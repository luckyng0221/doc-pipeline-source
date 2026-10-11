from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from doc_pipeline.config import MAX_SOURCE_URL_CHARS, build_config, parse_source_url, read_raw
from doc_pipeline.errors import ConfigurationError

PROJECT_CONFIG = Path(__file__).parents[1] / "config.yaml"


def _raw(**changes: object) -> dict:
    """The project config plus minimal fields, as the app would supply them."""
    raw = {key: value for key, value in read_raw(PROJECT_CONFIG).items() if key != "app"}
    raw.update(
        {
            "fields": [{"name": "vendor"}, {"name": "total", "type": "decimal"}],
            "fingerprint_fields": ["vendor", "total"],
        }
    )
    raw.update(changes)
    return raw


def test_project_config_holds_only_app_settings(tmp_path: Path) -> None:
    raw = read_raw(PROJECT_CONFIG)
    config = build_config(_raw(), tmp_path / "config.yaml")

    assert set(raw) == {"language", "date_order", "limits", "boundary_rules", "app"}
    assert config.models == () and config.boundary_model is None
    assert config.date_order == "day_first"
    assert config.boundary_rules.enabled
    assert config.boundary_rules.blank_pages == "ignore"
    assert config.limits.boundary_pdf_batch_pages == 40
    assert config.limits.extraction_workers == 3
    assert config.limits.render_timeout_seconds == 120
    assert config.limits.render_child_memory_mb == 1024
    with pytest.raises(ConfigurationError, match="must be positive"):
        build_config(_raw(limits={"extraction_workers": 0}), tmp_path / "config.yaml")
    assert config.paths.processing == tmp_path / "processing"


def test_fingerprint_ignores_settings_that_do_not_change_results(tmp_path: Path) -> None:
    config = build_config(_raw(models=[_model("a")], boundary_model=_model("a")), tmp_path / "config.yaml")
    tuned = replace(
        config,
        limits=replace(
            config.limits,
            retry_attempts=9,
            dhash_max_distance=3,
            extraction_workers=8,
            render_timeout_seconds=5.0,
            render_child_memory_mb=256,
        ),
    )
    renamed_key = replace(config, boundary_model=replace(config.boundary_model, api_key_env="OTHER_KEY"))

    assert tuned.fingerprint() == config.fingerprint()
    assert renamed_key.fingerprint() == config.fingerprint()
    assert replace(config, limits=replace(config.limits, render_dpi=200)).fingerprint() != config.fingerprint()
    assert replace(config, second_opinion="always").fingerprint() != config.fingerprint()


def test_render_child_limits_have_defaults_and_parse(tmp_path: Path) -> None:
    config = build_config(_raw(), tmp_path / "config.yaml")

    assert config.limits.render_timeout_seconds == 120.0
    assert config.limits.render_child_memory_mb == 1024

    tuned = build_config(
        _raw(limits={"render_timeout_seconds": 2.5, "render_child_memory_mb": 512}),
        tmp_path / "config.yaml",
    )

    assert tuned.limits.render_timeout_seconds == 2.5
    assert tuned.limits.render_child_memory_mb == 512


def test_render_child_limits_must_be_usable_numbers(tmp_path: Path) -> None:
    for value in (0, -1, -0.5):
        with pytest.raises(ConfigurationError, match="render_timeout_seconds must be positive"):
            build_config(_raw(limits={"render_timeout_seconds": value}), tmp_path / "config.yaml")
    for value in (0, -64):
        with pytest.raises(ConfigurationError, match="memory limits must be positive"):
            build_config(_raw(limits={"render_child_memory_mb": value}), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="invalid value"):
        build_config(_raw(limits={"render_timeout_seconds": "soon"}), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="invalid value"):
        build_config(_raw(limits={"render_child_memory_mb": "big"}), tmp_path / "config.yaml")


def test_render_timeout_has_an_upper_bound(tmp_path: Path) -> None:
    for value in (float("inf"), float("nan"), 0, -1, 3601):
        with pytest.raises(ConfigurationError, match="render_timeout_seconds must be"):
            build_config(_raw(limits={"render_timeout_seconds": value}), tmp_path / "config.yaml")
    for value in (120, 3600):
        accepted = build_config(_raw(limits={"render_timeout_seconds": value}), tmp_path / "config.yaml")
        assert accepted.limits.render_timeout_seconds == value


def test_render_child_limits_must_be_finite_and_bounded(tmp_path: Path) -> None:
    # -inf is refused earlier, as a non-positive timeout.
    refusals = (
        (float("nan"), "render_timeout_seconds must be a finite"),
        (float("inf"), "render_timeout_seconds must be a finite"),
        (float("-inf"), "render_timeout_seconds must be"),
    )
    for value, message in refusals:
        with pytest.raises(ConfigurationError, match=message):
            build_config(_raw(limits={"render_timeout_seconds": value}), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="render_child_memory_mb must be at most 4096"):
        build_config(_raw(limits={"render_child_memory_mb": 4097}), tmp_path / "config.yaml")

    tuned = build_config(
        _raw(limits={"render_timeout_seconds": 0.5, "render_child_memory_mb": 4096}),
        tmp_path / "config.yaml",
    )
    assert tuned.limits.render_timeout_seconds == 0.5
    assert tuned.limits.render_child_memory_mb == 4096
    # 1 MB is what the memory-kill test relies on; it stays configurable.
    floor = build_config(_raw(limits={"render_child_memory_mb": 1}), tmp_path / "config.yaml")
    assert floor.limits.render_child_memory_mb == 1


def _model(name: str, provider: str = "gemini", **extra: object) -> dict:
    return {"name": name, "provider": provider, "model": "test-model", **extra}


def test_zero_to_three_models_are_allowed(tmp_path: Path) -> None:
    rules_only = build_config(_raw(models=[], boundary_model=None), tmp_path / "config.yaml")
    assert rules_only.models == () and rules_only.boundary_model is None
    single = build_config(_raw(models=[_model("one")]), tmp_path / "config.yaml")
    assert [model.role for model in single.models] == ["primary"]
    arbitrated = build_config(
        _raw(models=[_model("a"), _model("b", "openai"), _model("c", "glm", role="arbiter")]),
        tmp_path / "config.yaml",
    )
    assert [model.role for model in arbitrated.models] == ["primary", "primary", "arbiter"]


def test_invalid_model_setups_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="needs two primary"):
        build_config(_raw(models=[_model("a"), _model("b", role="arbiter")]), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="at most 3"):
        build_config(_raw(models=[_model(str(index)) for index in range(4)]), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="effort for glm"):
        build_config(_raw(models=[_model("a", "glm", effort="high")]), tmp_path / "config.yaml")


def test_efforts_and_friendly_field_names(tmp_path: Path) -> None:
    config = build_config(
        _raw(
            models=[_model("a", "openai", effort="low"), _model("b", "glm", effort="off")],
            fields=[
                {"name": "Tax / SST Amount", "type": "decimal", "keywords": "SST, Service Tax"},
                {"name": "Document  Number", "keywords": ["Invoice No"]},
            ],
            fingerprint_fields=[],
        ),
        tmp_path / "config.yaml",
    )

    assert [model.effort for model in config.models] == ["low", "off"]
    assert [spec.name for spec in config.fields] == ["Tax / SST Amount", "Document Number"]
    assert config.fields[0].keywords == ("SST", "Service Tax")
    assert config.fingerprint_fields == ()
    for bad_name in ("=cmd", "", "x" * 41):
        with pytest.raises(ConfigurationError, match="field name"):
            build_config(_raw(fields=[{"name": bad_name}], fingerprint_fields=[]), tmp_path / "config.yaml")
    with pytest.raises(ConfigurationError, match="unique"):
        build_config(_raw(fields=[{"name": "Total"}, {"name": "total"}], fingerprint_fields=[]), tmp_path / "config.yaml")


def test_source_url_accepts_empty_and_plain_https_addresses() -> None:
    assert parse_source_url(None) == ""
    assert parse_source_url("") == ""
    valid = "https://example.com/source/tests"
    assert parse_source_url(valid) == valid
    at_the_limit = "https://example.com/" + "a" * (MAX_SOURCE_URL_CHARS - len("https://example.com/"))
    assert len(at_the_limit) == MAX_SOURCE_URL_CHARS
    assert parse_source_url(at_the_limit) == at_the_limit


def _credentialed_url() -> str:
    """A https address with credentials, built so no line holds one whole."""
    login = "no-such-user" + ":obviously-not-a-real-" + "password@"
    return f"https://{login}example.com/source"


def test_source_url_rejects_bad_values() -> None:
    overlong = "https://example.com/" + "a" * (MAX_SOURCE_URL_CHARS - len("https://example.com/") + 1)
    refusals = (
        "http://example.com/source",
        "javascript:alert(1)",
        "data:text/html,hello",
        "https://",
        _credentialed_url(),
        "https://example.com/a b",
        "https://example.com/a\tb",
        "https://example.com/a\x7fb",
        "https://example.com/s?x=1",
        "https://example.com/s?",
        "https://example.com/s#f",
        "   ",
        overlong,  # one character past the limit
        2048,
        ["https://example.com/source"],
    )

    assert len(overlong) == MAX_SOURCE_URL_CHARS + 1
    for value in refusals:
        with pytest.raises(ConfigurationError, match="app.source_url"):
            parse_source_url(value)


def test_source_url_with_a_query_or_fragment_names_the_setting_but_not_the_value() -> None:
    for value in ("https://example.com/s?x=1", "https://example.com/s?", "https://example.com/s#f"):
        with pytest.raises(ConfigurationError, match="app.source_url") as raised:
            parse_source_url(value)
        assert "example.com" not in str(raised.value)


def test_source_url_that_urlsplit_cannot_parse_hides_the_original_error() -> None:
    with pytest.raises(ConfigurationError, match="app.source_url") as raised:
        parse_source_url("https://[::1/source")

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


def test_source_url_errors_never_repeat_the_rejected_value() -> None:
    credentialed = _credentialed_url()

    with pytest.raises(ConfigurationError) as raised:
        parse_source_url(credentialed)

    # A refused address may carry credentials: the message must not echo it.
    assert "no-such-user" not in str(raised.value)
    assert "example.com" not in str(raised.value)


def test_build_config_checks_the_source_offer(tmp_path: Path) -> None:
    config = build_config(_raw(app={"source_url": "https://example.com/source"}), tmp_path / "config.yaml")

    assert config.path == tmp_path / "config.yaml"
    with pytest.raises(ConfigurationError, match="app.source_url"):
        build_config(_raw(app={"source_url": "http://example.com/source"}), tmp_path / "config.yaml")
