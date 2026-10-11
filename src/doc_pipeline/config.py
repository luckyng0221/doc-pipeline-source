from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from . import __version__
from .errors import ConfigurationError
from .models import EFFORT_LEVELS, BoundaryRulesConfig, FieldSpec, ModelConfig

SUPPORTED_PROVIDERS = {"openai", "gemini", "glm"}
DEFAULT_GLM_URL = "https://open.bigmodel.cn/api/paas/v4/"
FIELD_TYPES = {"string", "decimal", "date", "integer", "boolean"}
SECOND_OPINION_MODES = {"when_needed", "always"}
DATE_ORDERS = {"day_first", "month_first", "unknown"}
FALLBACKS = {"new_document", "same_document"}
# Field names become Excel headers and JSON keys: letters, digits, spaces and
# a few separators, so "Tax / SST Amount" works.
FIELD_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _\-/()&.]{0,39}")
MAX_MODELS = 3
# Ceiling for limits.render_child_memory_mb: a render child shares the
# server's machine with every other session.
MAX_RENDER_CHILD_MEMORY_MB = 4096
# Ceiling for limits.render_timeout_seconds, an hour: a child that is allowed
# to run longer holds a run slot and a process for that long.
MAX_RENDER_TIMEOUT_SECONDS = 3600
# Ceiling for app.source_url, the public source offer (AGPL section 13).
MAX_SOURCE_URL_CHARS = 2048
# Limits that change what is sent to the models, and therefore the results.
RESULT_LIMITS = (
    "render_dpi",
    "jpeg_quality",
    "boundary_batch_pages",
    "boundary_pdf_batch_pages",
    "boundary_min_confidence",
    "extraction_batch_pages",
)


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    processing: Path
    split: Path
    done: Path
    errors: Path
    output: Path
    registry: Path

    def ensure_directories(self) -> None:
        for path in (
            self.processing,
            self.split,
            self.done,
            self.errors,
            self.output,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class LimitsConfig:
    max_pdf_mb: int = 50
    max_pages: int = 100
    render_dpi: int = 150
    jpeg_quality: int = 82
    # Each PDF is rendered in a disposable child process (POSIX only; elsewhere
    # in-process). The child is killed after this many seconds and its memory
    # is capped at this many MB, so a slow or bloated PDF fails on its own.
    render_timeout_seconds: float = 120.0
    render_child_memory_mb: int = 1024
    boundary_batch_pages: int = 6
    boundary_pdf_batch_pages: int = 40
    boundary_min_confidence: float = 0.65
    extraction_batch_pages: int = 12
    # Documents read at the same time. Results do not depend on it.
    extraction_workers: int = 3
    retry_attempts: int = 3
    retry_base_seconds: float = 1.0
    dhash_max_distance: int = 6
    visual_similarity_threshold: float = 0.85


@dataclass(frozen=True)
class PipelineConfig:
    path: Path
    paths: PathsConfig
    limits: LimitsConfig
    language: str
    # None: pages the rules cannot decide use boundary_rules.fallback.
    boundary_model: ModelConfig | None
    # Empty: fields are read from the PDF text by rules only.
    models: tuple[ModelConfig, ...]
    fields: tuple[FieldSpec, ...]
    fingerprint_fields: tuple[str, ...]
    boundary_rules: BoundaryRulesConfig = field(default_factory=BoundaryRulesConfig)
    # "when_needed": the first primary runs alone and the other models are
    # called only if its values cannot be confirmed from the PDF text.
    # "always": both primaries run on every document.
    second_opinion: str = "when_needed"
    # How numeric dates are written in the documents (02/12/2024 is 2 December
    # when day_first). "unknown" never trusts such a date from one model alone.
    date_order: str = "unknown"

    def fingerprint(self) -> str:
        """Hash every setting that can change the extracted results.

        Retry, size, and duplicate-matching limits and API-key variable names
        are left out, so changing them does not invalidate cached results and
        repeat model calls.
        """
        limits = asdict(self.limits)

        def model_identity(model: ModelConfig) -> dict[str, Any]:
            identity = asdict(model)
            identity.pop("api_key_env")
            return identity

        payload = {
            "pipeline_version": __version__,
            "result_schema_version": 4,
            "limits": {key: limits[key] for key in RESULT_LIMITS},
            "language": self.language,
            "boundary_model": model_identity(self.boundary_model) if self.boundary_model else None,
            "boundary_rules": asdict(self.boundary_rules),
            "second_opinion": self.second_opinion,
            "date_order": self.date_order,
            "models": [model_identity(model) for model in self.models],
            "fields": [asdict(spec) for spec in self.fields],
            "fingerprint_fields": list(self.fingerprint_fields),
        }
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _parse_model(raw: Any, *, default_name: str | None = None) -> ModelConfig:
    if not isinstance(raw, dict):
        raise ConfigurationError("Each model configuration must be a mapping.")
    provider = str(raw.get("provider", "")).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigurationError(f"Unsupported provider: {provider or '(missing)'}.")
    name = str(raw.get("name", default_name or provider)).strip()
    model = str(raw.get("model", "")).strip()
    if not name or not model:
        raise ConfigurationError("Every model needs a name and model identifier.")
    default_key = {"openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "glm": "GLM_API_KEY"}[provider]
    api_key_env = str(raw.get("api_key_env", default_key)).strip()
    if not api_key_env:
        raise ConfigurationError(f"Model {name} has an empty api_key_env.")
    base_url = raw.get("base_url")
    if provider == "glm" and not base_url:
        base_url = DEFAULT_GLM_URL
    structured_output = raw.get("structured_output", provider != "glm")
    role = str(raw.get("role", "primary")).strip().lower()
    if role not in {"primary", "arbiter"}:
        raise ConfigurationError(f"Model {name} has an invalid role; use primary or arbiter.")
    input_mode = str(raw.get("input", "images")).strip().lower()
    if input_mode not in {"images", "pdf"}:
        raise ConfigurationError(f"Model {name} has an invalid input mode; use images or pdf.")
    if input_mode == "pdf" and provider != "gemini":
        raise ConfigurationError(f"Model {name}: pdf input is only supported for the gemini provider.")
    effort = str(raw.get("effort", "default")).strip().lower()
    if effort not in EFFORT_LEVELS[provider]:
        choices = ", ".join(EFFORT_LEVELS[provider])
        raise ConfigurationError(f"Model {name}: effort for {provider} must be one of {choices}.")
    return ModelConfig(
        name=name,
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=str(base_url).strip() if base_url else None,
        structured_output=bool(structured_output),
        role=role,
        input=input_mode,
        effort=effort,
    )


def _parse_keywords(raw: Any, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list):
        raise ConfigurationError(f"Field {name}: keywords must be a list or a comma-separated text.")
    return tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))


def _parse_fields(raw: Any) -> tuple[FieldSpec, ...]:
    if not isinstance(raw, list) or not raw:
        raise ConfigurationError("fields must be a non-empty list.")
    fields: list[FieldSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ConfigurationError("Each field must be a mapping.")
        name = " ".join(str(item.get("name", "")).split())
        field_type = str(item.get("type", "string")).strip().lower()
        if not FIELD_NAME.fullmatch(name):
            raise ConfigurationError(
                f"Invalid field name {name!r}: use up to 40 letters, digits, spaces, or _-/()&. characters."
            )
        if field_type not in FIELD_TYPES:
            raise ConfigurationError(f"Unsupported type {field_type!r} for field {name}.")
        fields.append(
            FieldSpec(
                name=name,
                type=field_type,
                required=bool(item.get("required", False)),
                description=str(item.get("description", "")).strip(),
                verify_in_text=bool(item.get("verify_in_text", True)),
                keywords=_parse_keywords(item.get("keywords"), name),
            )
        )
    if len({spec.name.casefold() for spec in fields}) != len(fields):
        raise ConfigurationError("Field names must be unique.")
    return tuple(fields)


def _parse_limits(raw: Any) -> LimitsConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("limits must be a mapping.")
    defaults = LimitsConfig()
    values = {key: raw.get(key, getattr(defaults, key)) for key in asdict(defaults)}
    try:
        limits = LimitsConfig(
            max_pdf_mb=int(values["max_pdf_mb"]),
            max_pages=int(values["max_pages"]),
            render_dpi=int(values["render_dpi"]),
            jpeg_quality=int(values["jpeg_quality"]),
            render_timeout_seconds=float(values["render_timeout_seconds"]),
            render_child_memory_mb=int(values["render_child_memory_mb"]),
            boundary_batch_pages=int(values["boundary_batch_pages"]),
            boundary_pdf_batch_pages=int(values["boundary_pdf_batch_pages"]),
            boundary_min_confidence=float(values["boundary_min_confidence"]),
            extraction_batch_pages=int(values["extraction_batch_pages"]),
            extraction_workers=int(values["extraction_workers"]),
            retry_attempts=int(values["retry_attempts"]),
            retry_base_seconds=float(values["retry_base_seconds"]),
            dhash_max_distance=int(values["dhash_max_distance"]),
            visual_similarity_threshold=float(values["visual_similarity_threshold"]),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("One or more limits have an invalid value.") from exc
    positive_ints = (
        limits.max_pdf_mb,
        limits.max_pages,
        limits.render_dpi,
        limits.render_child_memory_mb,
        limits.boundary_batch_pages,
        limits.boundary_pdf_batch_pages,
        limits.extraction_batch_pages,
        limits.extraction_workers,
        limits.retry_attempts,
    )
    if any(value <= 0 for value in positive_ints):
        raise ConfigurationError("PDF, page, batch, worker, retry, and memory limits must be positive.")
    if limits.render_timeout_seconds <= 0:
        raise ConfigurationError("render_timeout_seconds must be positive.")
    if not math.isfinite(limits.render_timeout_seconds):
        raise ConfigurationError("render_timeout_seconds must be a finite number.")
    if limits.render_timeout_seconds > MAX_RENDER_TIMEOUT_SECONDS:
        raise ConfigurationError(f"render_timeout_seconds must be at most {MAX_RENDER_TIMEOUT_SECONDS}.")
    if limits.render_child_memory_mb > MAX_RENDER_CHILD_MEMORY_MB:
        raise ConfigurationError("render_child_memory_mb must be at most 4096.")
    if not 1 <= limits.jpeg_quality <= 100:
        raise ConfigurationError("jpeg_quality must be between 1 and 100.")
    if limits.retry_base_seconds < 0:
        raise ConfigurationError("retry_base_seconds cannot be negative.")
    if not 0 <= limits.boundary_min_confidence <= 1:
        raise ConfigurationError("boundary_min_confidence must be between 0 and 1.")
    if not 0 <= limits.visual_similarity_threshold <= 1:
        raise ConfigurationError("visual_similarity_threshold must be between 0 and 1.")
    if not 0 <= limits.dhash_max_distance <= 64:
        raise ConfigurationError("dhash_max_distance must be between 0 and 64.")
    return limits


def _parse_boundary_rules(raw: Any) -> BoundaryRulesConfig:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigurationError("boundary_rules must be a mapping.")
    defaults = BoundaryRulesConfig()
    blank_pages = str(raw.get("blank_pages", defaults.blank_pages)).strip().lower()
    if blank_pages not in {"ignore", "separator"}:
        raise ConfigurationError("boundary_rules.blank_pages must be ignore or separator.")
    try:
        blank_max_ink = float(raw.get("blank_max_ink", defaults.blank_max_ink))
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("boundary_rules.blank_max_ink must be a number.") from exc
    if not 0 <= blank_max_ink <= 1:
        raise ConfigurationError("boundary_rules.blank_max_ink must be between 0 and 1.")
    fallback = str(raw.get("fallback", defaults.fallback)).strip().lower()
    if fallback not in FALLBACKS:
        raise ConfigurationError("boundary_rules.fallback must be new_document or same_document.")
    titles_raw = raw.get("titles")
    titles = defaults.titles
    if titles_raw is not None:
        if not isinstance(titles_raw, dict) or not titles_raw:
            raise ConfigurationError("boundary_rules.titles must map document types to lists of title lines.")
        titles = {}
        for kind, phrases in titles_raw.items():
            if isinstance(phrases, str):
                phrases = [phrases]
            if not isinstance(phrases, list) or not all(isinstance(item, str) and item.strip() for item in phrases):
                raise ConfigurationError(f"boundary_rules.titles.{kind} must be a list of title lines.")
            titles[str(kind).strip()] = tuple(item.strip() for item in phrases)
    return BoundaryRulesConfig(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        blank_pages=blank_pages,
        blank_max_ink=blank_max_ink,
        fallback=fallback,
        titles=titles,
    )


def parse_source_url(raw: Any) -> str:
    """Validate app.source_url, the public source offer (AGPL section 13).

    Empty stays empty: the page then shows only the licence line. Any other
    value must be a plain https address with a host, without credentials,
    whitespace, control characters, a query string or a fragment. The
    messages never repeat the value, which may carry credentials that must
    not reach a log.
    """
    if raw is None or raw == "":
        return ""
    if not isinstance(raw, str):
        raise ConfigurationError("app.source_url must be text.")
    if len(raw) > MAX_SOURCE_URL_CHARS:
        raise ConfigurationError(f"app.source_url must be at most {MAX_SOURCE_URL_CHARS} characters.")
    if any(character.isspace() or unicodedata.category(character) == "Cc" for character in raw):
        raise ConfigurationError("app.source_url must not contain whitespace or control characters.")
    if "?" in raw or "#" in raw:
        raise ConfigurationError("app.source_url must not contain a query string or a fragment.")
    try:
        parsed = urlsplit(raw)
    except ValueError:
        # from None: the ValueError text can repeat the value.
        raise ConfigurationError("app.source_url must be a usable https address.") from None
    if parsed.scheme != "https":
        raise ConfigurationError("app.source_url must be an https address.")
    if not parsed.hostname:
        raise ConfigurationError("app.source_url needs a host name.")
    if "@" in parsed.netloc:
        raise ConfigurationError("app.source_url must not carry credentials.")
    return raw


def read_raw(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read config.yaml as a mapping without touching environment variables."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError("Could not read config.yaml.") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError("The configuration root must be a mapping.")
    return raw


def load_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Read and validate a configuration file (no environment variables are loaded)."""
    path = Path(config_path).expanduser().resolve()
    return build_config(read_raw(path), path)


def build_config(raw: dict[str, Any], path: Path) -> PipelineConfig:
    """Validate a raw configuration mapping; relative paths resolve beside path."""
    root = Path(str(raw.get("root", "."))).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    paths_raw = raw.get("paths", {})
    if not isinstance(paths_raw, dict):
        raise ConfigurationError("paths must be a mapping.")

    def resolve_path(key: str, default: str) -> Path:
        candidate = Path(str(paths_raw.get(key, default))).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    paths = PathsConfig(
        root=root,
        processing=resolve_path("processing", "processing"),
        split=resolve_path("split", "split"),
        done=resolve_path("done", "done"),
        errors=resolve_path("errors", "errors"),
        output=resolve_path("output", "output"),
        registry=resolve_path("registry", "registry.db"),
    )

    models_raw = raw.get("models") or []
    if not isinstance(models_raw, list) or len(models_raw) > MAX_MODELS:
        raise ConfigurationError(f"models must be a list of at most {MAX_MODELS} extraction models.")
    models = tuple(_parse_model(item) for item in models_raw)
    if len({model.name for model in models}) != len(models):
        raise ConfigurationError("Extraction model names must be unique.")
    roles = [model.role for model in models]
    if roles.count("arbiter") > 1:
        raise ConfigurationError("Configure at most one arbiter model.")
    if "arbiter" in roles and roles.count("primary") < 2:
        raise ConfigurationError("An arbiter model needs two primary models to settle.")
    boundary_raw = raw.get("boundary_model")
    boundary_model = _parse_model(boundary_raw, default_name="boundary") if boundary_raw else None
    fields = _parse_fields(raw.get("fields"))
    fingerprint_fields_raw = raw.get("fingerprint_fields", ["vendor", "invoice_no", "date", "total"])
    if not isinstance(fingerprint_fields_raw, list):
        raise ConfigurationError("fingerprint_fields must be a list.")
    fingerprint_fields = tuple(str(name).strip() for name in fingerprint_fields_raw)
    known_fields = {spec.name for spec in fields}
    if any(name not in known_fields for name in fingerprint_fields):
        raise ConfigurationError("Every fingerprint field must also appear in fields.")
    if len(set(fingerprint_fields)) != len(fingerprint_fields):
        raise ConfigurationError("fingerprint_fields must not contain duplicates.")

    limits = _parse_limits(raw.get("limits"))
    language = str(raw.get("language", "auto")).strip() or "auto"
    second_opinion = str(raw.get("second_opinion", "when_needed")).strip().lower()
    if second_opinion not in SECOND_OPINION_MODES:
        raise ConfigurationError("second_opinion must be when_needed or always.")
    date_order = str(raw.get("date_order", "unknown")).strip().lower()
    if date_order not in DATE_ORDERS:
        raise ConfigurationError("date_order must be day_first, month_first, or unknown.")
    # The app section belongs to app/settings.py, but the source offer is
    # checked here too, so load_config refuses a bad app.source_url as well.
    app_raw = raw.get("app")
    if isinstance(app_raw, dict):
        parse_source_url(app_raw.get("source_url"))
    return PipelineConfig(
        path=path,
        paths=paths,
        limits=limits,
        language=language,
        boundary_model=boundary_model,
        models=models,
        fields=fields,
        fingerprint_fields=fingerprint_fields,
        boundary_rules=_parse_boundary_rules(raw.get("boundary_rules")),
        second_opinion=second_opinion,
        date_order=date_order,
    )
