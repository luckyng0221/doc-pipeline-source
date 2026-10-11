"""Turn upload-app choices into a pipeline configuration (no Streamlit here)."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import PipelineConfig, build_config, parse_source_url, read_raw
from ..errors import ConfigurationError
from ..models import EFFORT_LEVELS
from ..pricing import Price, parse_prices

MAX_SLOTS = 3
PROVIDER_LABELS = {"gemini": "Gemini (Google)", "openai": "OpenAI", "glm": "GLM (Z.ai / Zhipu)"}
SLOT_TITLES = ("Model 1: runs first", "Model 2: second opinion", "Model 3: arbiter")
EFFORT_LABELS = {
    "default": "Provider default",
    "off": "Off (no thinking)",
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "on": "On (thinking)",
}
# Field editor type labels and the pipeline field types they stand for.
TYPE_LABELS = {"Text": "string", "Number": "decimal", "Date": "date", "Whole number": "integer", "Yes/No": "boolean"}
TYPE_NAMES = {value: label for label, value in TYPE_LABELS.items()}
DATE_ORDER_LABELS = {
    "day_first": "Day first (31/12/2025)",
    "month_first": "Month first (12/31/2025)",
    "unknown": "Not sure",
}
# Range of the "Documents read at once" slider.
MAX_WORKERS = 8
# Field-editor limits. The page sets some of them too, but a browser can send
# any value, so they are checked here.
MAX_FIELD_ROWS = 100
MAX_FIELDS = 30
MAX_KEYWORDS = 20
MAX_KEYWORD_CHARS = 60
MAX_DESCRIPTION_CHARS = 300
# Model IDs as providers write them, fine-tune IDs (ft:gpt-4o-mini:org:x:id) included.
MODEL_ID = re.compile(r"[A-Za-z0-9._:/-]{1,100}")
# Settings from config.yaml that the app replaces with its own.
APP_OWNED_KEYS = ("root", "paths", "models", "boundary_model", "fields", "fingerprint_fields", "app")


@dataclass(frozen=True)
class AppLimits:
    max_files: int = 5
    max_file_mb: int = 20
    max_pages: int = 50
    workspace_ttl_minutes: int = 120
    # Runs across all sessions of the server: how many work at once and how
    # many more may wait. Each user has at most one of them.
    max_concurrent_runs: int = 2
    max_waiting_runs: int = 4
    # A run stops here: files it finished are kept, the rest are reported as
    # not run. Must stay below the workspace TTL, so a stopped run's folder
    # is never swept while it works.
    run_timeout_minutes: int = 60
    # Runs each visitor may start per sliding hour and day: counted per
    # account when signed in and per IP address when not. In-memory counters,
    # so a restart resets them.
    max_runs_per_hour: int = 10
    max_runs_per_day: int = 40
    # Model tests (Test models) each visitor may run per sliding hour, under
    # its own counter: unlike a run, a test needs no uploads.
    max_model_tests_per_hour: int = 20


@dataclass(frozen=True)
class AppDefaults:
    base_raw: dict[str, Any]
    limits: AppLimits
    model_suggestions: dict[str, tuple[str, ...]]
    glm_endpoints: dict[str, str]
    field_presets: tuple[dict[str, Any], ...]
    date_order: str
    extraction_workers: int = 3
    # USD per 1M tokens by model ID (lower case), for the cost estimate.
    prices: dict[str, Price] = field(default_factory=dict)
    # Public source offer, AGPL section 13.
    source_url: str = ""


@dataclass
class ModelSlot:
    provider: str | None = None
    api_key: str = ""
    model: str = ""
    effort: str = "default"
    base_url: str | None = None


@dataclass
class AppChoices:
    slots: list[ModelSlot] = field(default_factory=list)
    # "rules" or the 1-based number of the slot that splits pages.
    splitting: str | int = "rules"
    second_opinion: str = "when_needed"
    date_order: str = "day_first"
    fallback: str = "new_document"
    blank_pages: str = "ignore"
    # Documents read at the same time when models are used.
    workers: int = 3
    fields: list[dict[str, Any]] = field(default_factory=list)

    def active_slots(self) -> list[ModelSlot]:
        return [slot for slot in self.slots if slot.provider]


def load_app_defaults(config_path: Path) -> AppDefaults:
    """Read config.yaml for the app; no environment variables or server keys are used."""
    raw = read_raw(config_path)
    app = raw.get("app") or {}
    if not isinstance(app, dict):
        raise ConfigurationError("app must be a mapping.")
    defaults = AppLimits()
    try:
        limits = AppLimits(
            max_files=int(app.get("max_files", defaults.max_files)),
            max_file_mb=int(app.get("max_file_mb", defaults.max_file_mb)),
            max_pages=int(app.get("max_pages", defaults.max_pages)),
            workspace_ttl_minutes=int(app.get("workspace_ttl_minutes", defaults.workspace_ttl_minutes)),
            max_concurrent_runs=int(app.get("max_concurrent_runs", defaults.max_concurrent_runs)),
            max_waiting_runs=int(app.get("max_waiting_runs", defaults.max_waiting_runs)),
            run_timeout_minutes=int(app.get("run_timeout_minutes", defaults.run_timeout_minutes)),
            max_runs_per_hour=int(app.get("max_runs_per_hour", defaults.max_runs_per_hour)),
            max_runs_per_day=int(app.get("max_runs_per_day", defaults.max_runs_per_day)),
            max_model_tests_per_hour=int(app.get("max_model_tests_per_hour", defaults.max_model_tests_per_hour)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("app limits must be whole numbers.") from exc
    if limits.max_concurrent_runs < 1 or limits.max_waiting_runs < 0:
        raise ConfigurationError("app.max_concurrent_runs must be at least 1 and app.max_waiting_runs at least 0.")
    if limits.run_timeout_minutes < 1:
        raise ConfigurationError("app.run_timeout_minutes must be at least 1.")
    if limits.max_runs_per_hour < 1 or limits.max_runs_per_day < 1:
        raise ConfigurationError("app.max_runs_per_hour must be at least 1 and app.max_runs_per_day at least 1.")
    if limits.max_runs_per_hour > limits.max_runs_per_day:
        raise ConfigurationError("app.max_runs_per_hour must not be greater than app.max_runs_per_day.")
    if limits.max_model_tests_per_hour < 1:
        raise ConfigurationError("app.max_model_tests_per_hour must be at least 1.")
    if limits.workspace_ttl_minutes <= limits.run_timeout_minutes:
        raise ConfigurationError(
            "app.workspace_ttl_minutes must be longer than app.run_timeout_minutes, so a run is not cleaned up "
            "while it works."
        )
    try:
        workers = int((raw.get("limits") or {}).get("extraction_workers", 3))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ConfigurationError("limits.extraction_workers must be a whole number.") from exc
    suggestions = app.get("models") or {}
    presets = app.get("field_presets") or []
    base_raw = {key: value for key, value in raw.items() if key not in APP_OWNED_KEYS}
    return AppDefaults(
        base_raw=base_raw,
        limits=limits,
        model_suggestions={
            provider: tuple(str(model) for model in suggestions.get(provider, ())) for provider in PROVIDER_LABELS
        },
        glm_endpoints={str(label): str(url) for label, url in (app.get("glm_endpoints") or {}).items()},
        field_presets=tuple(dict(item) for item in presets if isinstance(item, dict)),
        date_order=str(raw.get("date_order", "day_first")),
        extraction_workers=min(max(workers, 1), MAX_WORKERS),
        prices=parse_prices(app.get("prices")),
        source_url=parse_source_url(app.get("source_url")),
    )


def preset_rows(defaults: AppDefaults) -> list[dict[str, Any]]:
    """Starting rows for the field editor."""
    return [
        {
            "Include": bool(preset.get("include", False)),
            "Field": str(preset.get("name", "")),
            "Type": TYPE_NAMES.get(str(preset.get("type", "string")), "Text"),
            "Required": bool(preset.get("required", False)),
            "Duplicate key": bool(preset.get("duplicate_key", False)),
            "Keywords": ", ".join(str(item) for item in preset.get("keywords") or ()),
            "Description": str(preset.get("description", "")),
            "Must appear in text": bool(preset.get("verify_in_text", True)),
        }
        for preset in defaults.field_presets
    ]


def fields_from_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return raw field definitions and duplicate-key names for included rows."""
    if len(rows) > MAX_FIELD_ROWS:
        raise ConfigurationError(f"The field table can hold at most {MAX_FIELD_ROWS} rows.")
    fields: list[dict[str, Any]] = []
    fingerprint: list[str] = []
    for row in rows:
        name = _text(row.get("Field"))
        if not _flag(row.get("Include")) or not name:
            continue
        type_label = _text(row.get("Type")) or "Text"
        if type_label not in TYPE_LABELS:
            raise ConfigurationError(f"Field {name}: choose a type from {', '.join(TYPE_LABELS)}.")
        keywords, description = _text(row.get("Keywords")), _text(row.get("Description"))
        _check_field_text(name, keywords, description)
        fields.append(
            {
                "name": name,
                "type": TYPE_LABELS[type_label],
                "required": _flag(row.get("Required")),
                "description": description,
                "keywords": keywords,
                "verify_in_text": _flag(row.get("Must appear in text"), default=True),
            }
        )
        if _flag(row.get("Duplicate key")):
            fingerprint.append(" ".join(name.split()))
    if not fields:
        raise ConfigurationError("Include at least one field to extract.")
    if len(fields) > MAX_FIELDS:
        raise ConfigurationError(f"Include at most {MAX_FIELDS} fields ({len(fields)} are ticked).")
    return fields, fingerprint


def _check_field_text(name: str, keywords: str, description: str) -> None:
    labels = [label for label in (part.strip() for part in keywords.split(",")) if label]
    if len(labels) > MAX_KEYWORDS:
        raise ConfigurationError(f"Field {name}: use at most {MAX_KEYWORDS} keywords.")
    if any(len(label) > MAX_KEYWORD_CHARS for label in labels):
        raise ConfigurationError(f"Field {name}: keep each keyword to at most {MAX_KEYWORD_CHARS} characters.")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ConfigurationError(f"Field {name}: keep the description to at most {MAX_DESCRIPTION_CHARS} characters.")


def build_app_config(
    defaults: AppDefaults,
    choices: AppChoices,
    workspace: Path,
) -> tuple[PipelineConfig, dict[str, str]]:
    """Validate the choices; return the pipeline config and model name -> API key."""
    slots = choices.slots[:MAX_SLOTS]
    for index, slot in enumerate(slots):
        if not slot.provider and any(later.provider for later in slots[index + 1 :]):
            raise ConfigurationError(f"Fill the model slots in order: Model {index + 1} is empty but a later one is set.")
    active = [slot for slot in slots if slot.provider]
    models: list[dict[str, Any]] = []
    api_keys: dict[str, str] = {}
    for number, slot in enumerate(active, start=1):
        provider = str(slot.provider)
        if provider not in PROVIDER_LABELS:
            raise ConfigurationError(f"Model {number}: unsupported provider {provider}.")
        if not slot.api_key.strip():
            raise ConfigurationError(f"Model {number}: enter an API key or set the provider to None.")
        model_id = slot.model.strip()
        if not model_id:
            raise ConfigurationError(f"Model {number}: choose a model.")
        if not MODEL_ID.fullmatch(model_id):
            raise ConfigurationError(
                f"Model {number}: a model ID is up to 100 letters, digits, and . _ : / - characters."
            )
        name = f"model_{number}"
        model: dict[str, Any] = {
            "name": name,
            "provider": provider,
            "model": model_id,
            "effort": slot.effort,
            "role": "arbiter" if number == 3 else "primary",
            # Gemini reads native PDF pages at a flat ~258 tokens per page.
            "input": "pdf" if provider == "gemini" else "images",
            "structured_output": provider != "glm",
        }
        if provider == "glm" and slot.base_url:
            model["base_url"] = slot.base_url
        models.append(model)
        api_keys[name] = slot.api_key.strip()

    boundary_model = None
    if choices.splitting != "rules":
        number = int(choices.splitting)
        if not 1 <= number <= len(models):
            raise ConfigurationError("Page splitting uses a model slot that is not set.")
        boundary_model = dict(models[number - 1])

    fields, fingerprint = fields_from_rows(choices.fields)
    try:
        workers = min(max(int(choices.workers), 1), MAX_WORKERS)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("Documents read at once must be a whole number.") from exc
    raw = copy.deepcopy(defaults.base_raw)
    limits = dict(raw.get("limits") or {})
    limits.update(
        {
            "max_pages": defaults.limits.max_pages,
            "max_pdf_mb": defaults.limits.max_file_mb,
            "extraction_workers": workers,
        }
    )
    rules = dict(raw.get("boundary_rules") or {})
    rules.update({"fallback": choices.fallback, "blank_pages": choices.blank_pages})
    raw.update(
        {
            "root": str(workspace),
            "paths": {
                "processing": "processing",
                "split": "split",
                "done": "done",
                "errors": "errors",
                "output": "output",
                # Per-session registry: duplicates are found only among this
                # user's uploads, never across other people's files.
                "registry": "registry.db",
            },
            "models": models,
            "boundary_model": boundary_model,
            "fields": fields,
            "fingerprint_fields": fingerprint,
            "second_opinion": choices.second_opinion,
            "date_order": choices.date_order,
            "limits": limits,
            "boundary_rules": rules,
        }
    )
    return build_config(raw, workspace / "app-config.yaml"), api_keys


def describe_mode(choices: AppChoices) -> str:
    """One line for the banner: which models do what."""
    active = choices.active_slots()
    if not active:
        return "Rules only: no AI model"
    parts = []
    for number, slot in enumerate(active, start=1):
        role = "arbiter" if number == 3 else ("runs first" if number == 1 else "second opinion")
        effort = "" if slot.effort == "default" else f", effort {EFFORT_LABELS.get(slot.effort, slot.effort).lower()}"
        parts.append(f"Model {number} {PROVIDER_LABELS[str(slot.provider)]} {slot.model} ({role}{effort})")
    splitting = "rules" if choices.splitting == "rules" else f"rules, then Model {choices.splitting}"
    return f"{'; '.join(parts)}. Page splitting: {splitting}."


def effort_options(provider: str) -> tuple[str, ...]:
    return EFFORT_LEVELS[provider]


def _text(value: Any) -> str:
    if value is None or isinstance(value, float) and value != value:  # NaN from an empty editor cell
        return ""
    return " ".join(str(value).split())


def _flag(value: Any, default: bool = False) -> bool:
    if value is None or isinstance(value, float) and value != value:
        return default
    return bool(value)
