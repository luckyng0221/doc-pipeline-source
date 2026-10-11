from __future__ import annotations

import logging
import sys
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

import doc_pipeline.providers as providers
from doc_pipeline.config import LimitsConfig
from doc_pipeline.errors import ConfigurationError, ProviderError, RunStopped
from doc_pipeline.models import ModelConfig
from doc_pipeline.providers import OpenAICompatibleProvider, create_provider

LIMITS = LimitsConfig(retry_attempts=1)
ATTACKER = "https://attacker.example"


class OpenAIStub:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content='{"ok": true}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)


class GeminiStub:
    def __init__(self) -> None:
        self.configs: list = []
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.configs.append(config)
        return SimpleNamespace(text='{"ok": true}', usage_metadata=None)


def _model(provider: str, effort: str = "default", model: str = "test-model") -> ModelConfig:
    config = ModelConfig(name="model_1", provider=provider, model=model, api_key_env="UNSET_TEST_KEY", effort=effort)
    # A glm model always brings its own endpoint (config.py fills the default).
    return replace(config, base_url="https://glm.example/api/v4/") if provider == "glm" else config


def _openai_request(provider: str, effort: str, model: str = "test-model") -> dict:
    stub = OpenAIStub()
    client = create_provider(_model(provider, effort, model), LIMITS, api_key="test-key")
    client._client = stub
    assert client.check_connection().data == {"ok": True}
    return stub.requests[0]


def _record_openai(monkeypatch) -> tuple[list[dict], type]:
    """Stand in for the OpenAI SDK class, recording the options it is built with."""
    import openai

    recorded: list[dict] = []

    class RecordingOpenAI(OpenAIStub):
        def __init__(self, **options: object) -> None:
            super().__init__()
            recorded.append(options)

    real = openai.OpenAI
    monkeypatch.setattr(openai, "OpenAI", RecordingOpenAI)
    return recorded, real


def test_openai_effort_replaces_temperature() -> None:
    assert "reasoning_effort" not in _openai_request("openai", "default")
    assert _openai_request("openai", "default")["temperature"] == 0
    request = _openai_request("openai", "low")
    assert request["reasoning_effort"] == "low"
    assert "temperature" not in request


@pytest.mark.parametrize("model", ["gpt-5-mini", "o4-mini"])
def test_openai_reasoning_models_never_get_a_temperature(model: str) -> None:
    # They reject any temperature but the default, even at Provider default effort.
    request = _openai_request("openai", "default", model)

    assert "temperature" not in request and "reasoning_effort" not in request
    assert _openai_request("glm", "default", "glm-4.5v")["temperature"] == 0


def test_glm_effort_toggles_thinking() -> None:
    assert _openai_request("glm", "off")["extra_body"] == {"thinking": {"type": "disabled"}}
    assert _openai_request("glm", "on")["extra_body"] == {"thinking": {"type": "enabled"}}
    assert "extra_body" not in _openai_request("glm", "default")


@pytest.mark.parametrize(
    ("model", "effort", "budget", "level"),
    [
        ("gemini-3.8-flash", "default", None, None),
        ("gemini-3.8-flash", "off", 0, None),
        ("gemini-3.8-flash", "low", None, "LOW"),
        ("gemini-2.5-flash", "high", 24576, None),
    ],
)
def test_gemini_effort_sets_thinking(model: str, effort: str, budget: int | None, level: str | None) -> None:
    stub = GeminiStub()
    client = create_provider(_model("gemini", effort, model), LIMITS, api_key="test-key")
    client._client = stub

    client.check_connection()

    thinking = stub.configs[0].thinking_config
    if budget is None and level is None:
        assert thinking is None
    else:
        assert thinking.thinking_budget == budget
        assert (thinking.thinking_level.value if thinking.thinking_level else None) == level


def test_explicit_key_never_falls_back_to_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("UNSET_TEST_KEY", "server-key")
    client = create_provider(_model("openai"), LIMITS, api_key="")

    with pytest.raises(ProviderError, match="No API key entered for model_1"):
        client.check_connection()
    assert create_provider(_model("openai"), LIMITS)._api_key() == "server-key"


class RateLimited(Exception):
    status_code = 429


def test_retries_and_replies_are_reported_without_provider_text() -> None:
    stub = OpenAIStub()
    answer = stub._create
    failures = [RateLimited("quota exceeded for key sk-secret")]

    def create(**request):
        if failures:
            raise failures.pop()
        response = answer(**request)
        response.usage = SimpleNamespace(prompt_tokens=120, completion_tokens=8, total_tokens=128)
        return response

    stub.chat.completions.create = create
    client = create_provider(_model("openai"), LimitsConfig(retry_attempts=3, retry_base_seconds=0), api_key="k")
    client._client = stub
    notes, replies = [], []
    client.on_retry = notes.append
    client.on_reply = replies.append

    assert client.check_connection().data == {"ok": True}
    assert notes == ["model_1: RateLimited (HTTP 429); retrying in 0 s (attempt 2 of 3)"]
    # One reply, with its tokens for the cost estimate.
    assert [(reply["prompt_tokens"], reply["output_tokens"]) for reply in replies] == [(120, 8)]


def test_a_stop_before_a_retry_wait_propagates() -> None:
    stub = OpenAIStub()
    original = stub._create
    attempts: list[dict] = []

    def create(**request):
        attempts.append(request)
        if len(attempts) == 1:
            raise RateLimited("quota exceeded for key sk-secret")
        return original(**request)

    stub.chat.completions.create = create
    client = create_provider(_model("openai"), LimitsConfig(retry_attempts=3, retry_base_seconds=0), api_key="k")
    client._client = stub

    def stop() -> None:
        raise RunStopped("the run was cancelled")

    client.stop_check = stop

    with pytest.raises(RunStopped, match="the run was cancelled"):
        client.check_connection()

    assert len(attempts) == 1  # the stop was honoured instead of the retry wait


def test_a_stop_after_a_retry_wait_prevents_the_next_attempt() -> None:
    stub = OpenAIStub()
    attempts: list[dict] = []

    def create(**request):
        attempts.append(request)
        raise RateLimited("quota exceeded for key sk-secret")

    stub.chat.completions.create = create
    client = create_provider(_model("openai"), LimitsConfig(retry_attempts=3, retry_base_seconds=0), api_key="k")
    client._client = stub
    notes: list[str] = []
    client.on_retry = notes.append
    checks = [0]

    def stop() -> None:
        checks[0] += 1  # the first check runs before the wait, the second after it
        if checks[0] == 2:
            raise RunStopped("the run was cancelled")

    client.stop_check = stop

    with pytest.raises(RunStopped, match="the run was cancelled"):
        client.check_connection()

    assert len(notes) == 1  # the retry wait had begun
    assert len(attempts) == 1  # the attempt after the wait never started


def test_gemini_requests_time_out(monkeypatch) -> None:
    from google import genai

    made: list[dict] = []

    def client(**options):
        made.append(options)
        return GeminiStub()

    monkeypatch.setattr(genai, "Client", client)

    create_provider(_model("gemini"), LIMITS, api_key="test-key").check_connection()

    # Without one, a stalled connection would hold its worker and its run forever.
    assert made[0]["http_options"].timeout == 120_000


def test_the_openai_client_targets_the_official_endpoint_without_the_environment(monkeypatch, tmp_path) -> None:
    recorded, real = _record_openai(monkeypatch)
    # A same-uid process can plant these: Streamlit copies top-level string
    # secrets into the environment live. The CA file is not there, so a
    # regression to trust_env=True fails loudly instead of quietly working.
    monkeypatch.setenv("OPENAI_BASE_URL", f"{ATTACKER}/v1")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.example:9")
    monkeypatch.setenv("ALL_PROXY", "http://attacker.example:9")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing-ca.pem"))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path / "missing-certs"))

    provider = create_provider(_model("openai"), LIMITS, api_key="test-key")

    assert provider.check_connection().data == {"ok": True}

    options = recorded[0]
    assert options["base_url"] == providers.OPENAI_BASE_URL
    assert options["max_retries"] == 0
    assert options["timeout"] == providers.REQUEST_TIMEOUT_SECONDS
    assert isinstance(options["http_client"], httpx.Client)
    assert options["http_client"].trust_env is False
    assert options["http_client"].timeout == httpx.Timeout(providers.REQUEST_TIMEOUT_SECONDS)
    # The real SDK builds the same pinned endpoint from these options.
    client = real(**options)
    assert str(client.base_url).rstrip("/") == providers.OPENAI_BASE_URL


def test_openai_ignores_a_configured_base_url(monkeypatch) -> None:
    recorded, _ = _record_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{ATTACKER}/v1")
    config = ModelConfig(
        name="model_1", provider="openai", model="test-model", api_key_env="UNSET_TEST_KEY", base_url=f"{ATTACKER}/v1"
    )

    create_provider(config, LIMITS, api_key="test-key").check_connection()

    assert recorded[0]["base_url"] == providers.OPENAI_BASE_URL


def test_glm_keeps_its_configured_endpoint(monkeypatch) -> None:
    recorded, _ = _record_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{ATTACKER}/v1")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.example:9")
    config = ModelConfig(
        name="model_1",
        provider="glm",
        model="test-model",
        api_key_env="UNSET_TEST_KEY",
        base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    provider = create_provider(config, LIMITS, api_key="test-key")

    assert provider.check_connection().data == {"ok": True}

    options = recorded[0]
    assert options["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"
    assert options["http_client"].trust_env is False


def test_the_sdk_clients_are_built_once_per_provider_instance(monkeypatch) -> None:
    built: list[str] = []

    class RecordingOpenAI(OpenAIStub):
        def __init__(self, **options: object) -> None:
            super().__init__()
            built.append("openai")

    class RecordingHttpxClient:
        def __init__(self, **options: object) -> None:
            built.append("httpx")

    monkeypatch.setattr("openai.OpenAI", RecordingOpenAI)
    monkeypatch.setattr(httpx, "Client", RecordingHttpxClient)
    provider = create_provider(_model("openai"), LIMITS, api_key="test-key")

    provider.check_connection()
    provider.check_connection()

    assert built.count("openai") == 1 and built.count("httpx") == 1


def test_the_gemini_client_targets_the_official_endpoint_without_the_environment(monkeypatch) -> None:
    import certifi
    from google import genai

    made: list[dict] = []

    def client(**options):
        made.append(options)
        return GeminiStub()

    real = genai.Client
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", f"{ATTACKER}/")
    monkeypatch.setenv("HTTPS_PROXY", "http://attacker.example:9")
    # An existing bundle: google-genai reads this variable while it builds the
    # client, whatever its client args say.
    monkeypatch.setenv("SSL_CERT_FILE", certifi.where())
    monkeypatch.setattr(genai, "Client", client)

    create_provider(_model("gemini"), LIMITS, api_key="test-key").check_connection()

    http_options = made[0]["http_options"]
    assert http_options.base_url == providers.GEMINI_BASE_URL
    assert http_options.timeout == 120_000
    assert providers.GEMINI_HTTP_CLIENT_ARGS == {"trust_env": False, "verify": True}
    assert http_options.client_args == providers.GEMINI_HTTP_CLIENT_ARGS
    assert http_options.async_client_args == providers.GEMINI_HTTP_CLIENT_ARGS
    # The real SDK builds the same pinned endpoint, and its httpx client takes
    # no proxies from the environment.
    api_client = real(**made[0]).models._api_client
    assert api_client._http_options.base_url == providers.GEMINI_BASE_URL
    assert api_client._httpx_client.trust_env is False


def test_the_sdk_import_imports_every_sdk_module(monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr(providers.importlib, "import_module", imported.append)

    providers.import_provider_sdks()

    assert imported == list(providers.SDK_MODULES)


@pytest.mark.parametrize("error", [ImportError("no such module"), RuntimeError("boom")], ids=repr)
def test_a_failed_sdk_import_is_tolerated_and_the_rest_still_imported(monkeypatch, error: Exception) -> None:
    real = providers.importlib.import_module
    imported: list[str] = []

    def failing_first(name: str, *args: object, **kwargs: object) -> object:
        imported.append(name)
        if name == providers.SDK_MODULES[0]:
            raise error
        return real(name)

    monkeypatch.setattr(providers.importlib, "import_module", failing_first)

    providers.import_provider_sdks()  # the failure never leaves this function

    assert imported == list(providers.SDK_MODULES)


def test_the_sdk_import_lands_every_sdk_module_in_sys_modules() -> None:
    # A misspelled module name would fail to import and never get here.
    providers.import_provider_sdks()

    assert all(name in sys.modules for name in providers.SDK_MODULES)


def _warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == providers.logger.name]


def test_failed_sdk_imports_log_one_warning_naming_only_the_modules(monkeypatch, caplog) -> None:
    first, third = providers.SDK_MODULES[0], providers.SDK_MODULES[2]

    def importing(name: str, *args: object, **kwargs: object) -> object:
        if name in (first, third):
            raise ImportError("secret detail from /home/someone/.venv")
        return None

    monkeypatch.setattr(providers.importlib, "import_module", importing)

    with caplog.at_level(logging.WARNING, logger=providers.logger.name):
        providers.import_provider_sdks()

    assert _warnings(caplog) == [f"Provider SDK pre-import failed for: {first}, {third}."]


def test_a_clean_sdk_import_logs_nothing(monkeypatch, caplog) -> None:
    monkeypatch.setattr(providers.importlib, "import_module", lambda name: None)

    with caplog.at_level(logging.WARNING, logger=providers.logger.name):
        providers.import_provider_sdks()

    assert _warnings(caplog) == []


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_a_glm_model_without_an_endpoint_is_refused(monkeypatch, base_url: str | None) -> None:
    recorded, _ = _record_openai(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", f"{ATTACKER}/v1")
    config = replace(_model("glm"), base_url=base_url)

    with pytest.raises(ConfigurationError, match="No endpoint is configured for model_1"):
        create_provider(config, LIMITS, "test-key")

    # Built directly, the provider still never falls back to OPENAI_BASE_URL.
    provider = OpenAICompatibleProvider(config, LIMITS, "test-key")
    with pytest.raises(ProviderError, match="No endpoint is configured for model_1"):
        provider.check_connection()
    assert recorded == []  # no OpenAI client was ever constructed
