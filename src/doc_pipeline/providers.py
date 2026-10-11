from __future__ import annotations

import base64
import importlib
import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

from .config import LimitsConfig
from .errors import ConfigurationError, ProviderError
from .models import ModelConfig

logger = logging.getLogger(__name__)

USAGE_KEYS = ("prompt_tokens", "output_tokens", "thoughts_tokens", "cached_tokens", "total_tokens")
# The modules the providers import lazily inside _generate_text, imported
# eagerly so a running server keeps the code it loaded at start-up.
SDK_MODULES = ("openai", "google.genai", "google.genai.types", "httpx")
# Gemini 2.x models take a token budget instead of a thinking level.
GEMINI_THINKING_BUDGETS = {"minimal": 512, "low": 1024, "medium": 8192, "high": 24576}
# OpenAI reasoning models (o-series, GPT-5 family) accept only the default
# temperature, even at their default effort.
OPENAI_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")
# How long one provider request may take before it fails (and may be retried).
REQUEST_TIMEOUT_SECONDS = 120.0
# Pinned endpoints, passed to the SDKs explicitly: with no base_url the OpenAI
# SDK falls back to OPENAI_BASE_URL and google-genai to GOOGLE_GEMINI_BASE_URL,
# and Streamlit copies top-level string secrets into the environment live, so a
# same-uid process that can edit .streamlit/secrets.toml could otherwise move
# users' pages and API keys to another host without a restart.
OPENAI_BASE_URL = "https://api.openai.com/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/"
# Sent to google-genai as its httpx client args. trust_env=False keeps the
# environment's proxy settings (HTTPS_PROXY, ALL_PROXY, ...) away from provider
# requests, and verify=True keeps google-genai from building its SSL context
# from the environment's SSL_CERT_FILE/SSL_CERT_DIR (True is httpx's own
# certifi-backed default).
GEMINI_HTTP_CLIENT_ARGS = {"trust_env": False, "verify": True}


def import_provider_sdks() -> None:
    """Import every provider SDK once, up front.

    A failure is skipped here, because the lazy import in _generate_text
    surfaces the real error to that provider's user, but one constant WARNING
    names the modules that failed (they come from SDK_MODULES; the exception
    text is never logged).
    """
    failed: list[str] = []
    for name in SDK_MODULES:
        try:
            importlib.import_module(name)
        except Exception:
            failed.append(name)
    if failed:
        logger.warning("Provider SDK pre-import failed for: %s.", ", ".join(failed))


@dataclass(frozen=True)
class ProviderResponse:
    data: dict[str, Any]
    usage: dict[str, int]


def _empty_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS}


def _count(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _openai_usage(usage: Any) -> dict[str, int]:
    if usage is None:
        return _empty_usage()
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": _count(getattr(usage, "prompt_tokens", None)),
        "output_tokens": _count(getattr(usage, "completion_tokens", None)),
        "thoughts_tokens": 0,
        "cached_tokens": _count(getattr(details, "cached_tokens", None)),
        "total_tokens": _count(getattr(usage, "total_tokens", None)),
    }


def _gemini_usage(metadata: Any) -> dict[str, int]:
    if metadata is None:
        return _empty_usage()
    return {
        "prompt_tokens": _count(getattr(metadata, "prompt_token_count", None)),
        "output_tokens": _count(getattr(metadata, "candidates_token_count", None)),
        "thoughts_tokens": _count(getattr(metadata, "thoughts_token_count", None)),
        "cached_tokens": _count(getattr(metadata, "cached_content_token_count", None)),
        "total_tokens": _count(getattr(metadata, "total_token_count", None)),
    }


def _safe_error(exc: Exception) -> str:
    if isinstance(exc, ConfigurationError):
        return str(exc)  # our own message, e.g. a missing key; never provider text
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status is not None:
        return f"{type(exc).__name__} (HTTP {status})"
    return type(exc).__name__


def _status_code(exc: Exception) -> int | None:
    for name in ("status_code", "status", "code"):
        value = getattr(exc, name, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _is_transient(exc: Exception) -> bool:
    status = _status_code(exc)
    if status == 429 or (status is not None and 500 <= status <= 599):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.casefold()
    code_values: list[str] = []
    for attribute in ("status_code", "status", "code"):
        value = getattr(exc, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is not None:
            code_values.append(str(value).casefold())
    code = " ".join(code_values)
    return any(
        token in name or token in code
        for token in (
            "timeout",
            "connectionerror",
            "temporarilyunavailable",
            "servererror",
            "serviceunavailable",
            "resourceexhausted",
            "unavailable",
            "deadline_exceeded",
            "429",
            "500",
            "502",
            "503",
            "504",
        )
    )


def _parse_json(text: str) -> dict[str, Any]:
    value = text.strip()
    value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    start = value.find("{")
    end = value.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object found.")
    parsed = json.loads(value[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("JSON response must be an object.")
    return parsed


class VisionProvider:
    def __init__(self, config: ModelConfig, limits: LimitsConfig, api_key: str | None = None) -> None:
        self.config = config
        self.limits = limits
        # None reads the config.api_key_env variable; a string is the only
        # key used, so the upload app never falls back to the server's keys.
        self._explicit_key = api_key
        self._client: Any = None
        # Documents are read in parallel, so the first requests can race to
        # build the client.
        self._client_lock = threading.Lock()
        # Hooks for the app: on_reply with the tokens of each answer (for
        # progress and cost), on_retry with a sanitized note before each
        # backoff wait, and stop_check before and after each retry wait (a
        # request in flight is never interrupted).
        self.on_reply: Callable[[dict[str, int]], None] | None = None
        self.on_retry: Callable[[str], None] | None = None
        self.stop_check: Callable[[], None] | None = None

    def generate_json(self, prompt: str, parts: Sequence[bytes] = ()) -> ProviderResponse:
        text: str | None = None
        usage = _empty_usage()
        attempts = self.limits.retry_attempts
        for attempt in range(attempts):
            try:
                text, usage = self._generate_text(prompt, parts)
                break
            except Exception as exc:
                if not _is_transient(exc) or attempt + 1 >= attempts:
                    raise ProviderError(f"{self.config.name}: {_safe_error(exc)}") from exc
                # Spread the jitter with the delay: parallel requests that hit
                # a rate limit together should not all retry together.
                base = self.limits.retry_base_seconds * (2**attempt)
                delay = base + random.uniform(0, base * 0.25)
                if self.stop_check:
                    self.stop_check()
                if self.on_retry:
                    self.on_retry(
                        f"{self.config.name}: {_safe_error(exc)}; retrying in {delay:.0f} s "
                        f"(attempt {attempt + 2} of {attempts})"
                    )
                time.sleep(delay)
                if self.stop_check:
                    self.stop_check()
        if self.on_reply:
            self.on_reply(usage)
        if text is None:
            raise ProviderError(f"{self.config.name}: empty provider response.")
        try:
            data = _parse_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProviderError(f"{self.config.name}: response was not valid JSON.") from exc
        return ProviderResponse(data=data, usage=usage)

    def check_connection(self) -> ProviderResponse:
        """Send one tiny text-only request to check the key, model, and effort."""
        return self.generate_json('Reply with JSON only, exactly: {"ok": true}')

    def _api_key(self) -> str:
        if self._explicit_key is not None:
            value = self._explicit_key.strip()
            if not value:
                raise ConfigurationError(f"No API key entered for {self.config.name}.")
            return value
        value = os.environ.get(self.config.api_key_env, "").strip()
        if not value:
            raise ConfigurationError(f"Missing environment variable {self.config.api_key_env}.")
        return value

    def _generate_text(self, prompt: str, parts: Sequence[bytes]) -> tuple[str, dict[str, int]]:
        raise NotImplementedError


class OpenAICompatibleProvider(VisionProvider):
    def _generate_text(self, prompt: str, parts: Sequence[bytes]) -> tuple[str, dict[str, int]]:
        import httpx
        from openai import OpenAI

        with self._client_lock:
            if self._client is None:
                # Pinned endpoint and client: with no base_url the SDK would
                # honour OPENAI_BASE_URL, and without our own httpx client it
                # would apply the environment's proxies and CA settings
                # (trust_env). Built here only, once per provider instance.
                if self.config.provider != "openai" and not (self.config.base_url or "").strip():
                    raise ConfigurationError(f"No endpoint is configured for {self.config.name}.")
                options: dict[str, Any] = {
                    "api_key": self._api_key(),
                    "max_retries": 0,
                    "timeout": REQUEST_TIMEOUT_SECONDS,
                    # An OpenAI model ignores any configured base URL; a glm
                    # model must bring its own (checked above) - never the
                    # environment's.
                    "base_url": OPENAI_BASE_URL if self.config.provider == "openai" else self.config.base_url,
                    "http_client": httpx.Client(trust_env=False, timeout=REQUEST_TIMEOUT_SECONDS),
                }
                self._client = OpenAI(**options)
        client = self._client
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in parts:
            encoded = base64.b64encode(image).decode("ascii")
            image_url: dict[str, str] = {"url": f"data:image/jpeg;base64,{encoded}"}
            if self.config.provider == "openai":
                image_url["detail"] = "high"
            content.append({"type": "image_url", "image_url": image_url})
        request: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
        }
        effort = self.config.effort
        if self.config.provider == "openai" and effort != "default":
            # Reasoning models reject a custom temperature.
            request["reasoning_effort"] = effort
        elif not (
            self.config.provider == "openai" and self.config.model.casefold().startswith(OPENAI_REASONING_PREFIXES)
        ):
            request["temperature"] = 0
        if self.config.provider == "glm" and effort != "default":
            request["extra_body"] = {"thinking": {"type": "enabled" if effort == "on" else "disabled"}}
        if self.config.structured_output:
            request["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request)
        usage = _openai_usage(getattr(response, "usage", None))
        result = response.choices[0].message.content
        if isinstance(result, list):
            return "\n".join(str(getattr(part, "text", "")) for part in result), usage
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Provider returned no text content.")
        return result, usage


class GeminiProvider(VisionProvider):
    def _generate_text(self, prompt: str, parts: Sequence[bytes]) -> tuple[str, dict[str, int]]:
        from google import genai
        from google.genai import types

        with self._client_lock:
            if self._client is None:
                # The endpoint is pinned against GOOGLE_GEMINI_BASE_URL, and
                # client_args keeps the environment's proxies and CA files away
                # from the SDK's httpx clients. The SDK has no request timeout
                # by default; it takes milliseconds.
                self._client = genai.Client(
                    api_key=self._api_key(),
                    http_options=types.HttpOptions(
                        base_url=GEMINI_BASE_URL,
                        timeout=int(REQUEST_TIMEOUT_SECONDS * 1000),
                        client_args=GEMINI_HTTP_CLIENT_ARGS,
                        async_client_args=GEMINI_HTTP_CLIENT_ARGS,
                    ),
                )
        client = self._client
        mime_type = "application/pdf" if self.config.input == "pdf" else "image/jpeg"
        contents: list[Any] = [prompt]
        contents.extend(types.Part.from_bytes(data=part, mime_type=mime_type) for part in parts)
        config: dict[str, Any] = {"temperature": 0}
        thinking = _gemini_thinking(self.config.model, self.config.effort, types)
        if thinking is not None:
            config["thinking_config"] = thinking
        if self.config.structured_output:
            config["response_mime_type"] = "application/json"
        response = client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )
        usage = _gemini_usage(getattr(response, "usage_metadata", None))
        result = response.text
        if not isinstance(result, str) or not result.strip():
            raise ValueError("Provider returned no text content.")
        return result, usage


def _gemini_thinking(model: str, effort: str, types: Any) -> Any:
    if effort == "default":
        return None
    if effort == "off":
        return types.ThinkingConfig(thinking_budget=0)
    if model.startswith("gemini-2"):
        return types.ThinkingConfig(thinking_budget=GEMINI_THINKING_BUDGETS[effort])
    return types.ThinkingConfig(thinking_level=effort.upper())


def create_provider(config: ModelConfig, limits: LimitsConfig, api_key: str | None = None) -> VisionProvider:
    if config.provider == "glm" and not (config.base_url or "").strip():
        raise ConfigurationError(f"No endpoint is configured for {config.name}.")
    if config.provider in {"openai", "glm"}:
        return OpenAICompatibleProvider(config, limits, api_key)
    if config.provider == "gemini":
        return GeminiProvider(config, limits, api_key)
    raise ConfigurationError(f"Unsupported provider: {config.provider}.")
