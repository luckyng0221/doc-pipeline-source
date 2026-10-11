from __future__ import annotations

from pathlib import Path

import pytest

from doc_pipeline.app.settings import load_app_defaults
from doc_pipeline.errors import ConfigurationError
from doc_pipeline.pricing import Price, add_usage, estimate, format_usd, parse_prices

ROOT = Path(__file__).parents[1]


def _usage(prompt: int, output: int, thoughts: int = 0, calls: int = 1) -> dict[str, int]:
    return {
        "prompt_tokens": prompt,
        "output_tokens": output,
        "thoughts_tokens": thoughts,
        "cached_tokens": 0,
        "total_tokens": prompt + output + thoughts,
        "calls": calls,
    }


def test_prices_are_read_by_model_id_and_validated() -> None:
    assert parse_prices({"Gemini-3.8-Flash": {"input": 0.75, "output": "3.75"}}) == {
        "gemini-3.8-flash": Price(0.75, 3.75)
    }
    assert parse_prices(None) == {}
    for bad in ([1, 2], {"m": {"input": 1}}, {"m": {"input": "x", "output": 1}}, {"m": {"input": -1, "output": 1}}):
        with pytest.raises(ConfigurationError):
            parse_prices(bad)


def test_cost_counts_thinking_as_output_and_leaves_unpriced_models_out() -> None:
    usage = {
        ("gemini", "gemini-3.8-flash"): _usage(1_000_000, 100_000, thoughts=100_000, calls=3),
        ("openai", "gpt-6-preview"): _usage(500, 20),
    }

    costs = estimate(usage, {"gemini-3.8-flash": Price(0.75, 3.75)})

    gemini, unpriced = costs
    assert (gemini.calls, gemini.input_tokens, gemini.output_tokens) == (3, 1_000_000, 200_000)
    assert gemini.usd == pytest.approx(0.75 + 0.75)
    assert unpriced.model_id == "gpt-6-preview" and unpriced.usd is None


def test_session_usage_adds_up_runs() -> None:
    key = ("gemini", "gemini-2.5-flash")
    first = {key: _usage(100, 10)}

    total = add_usage(add_usage({}, first), {key: _usage(50, 5, calls=2), ("glm", "glm-4.5v"): _usage(7, 1)})

    assert total[key]["prompt_tokens"] == 150 and total[key]["calls"] == 3
    assert total[("glm", "glm-4.5v")]["output_tokens"] == 1
    assert first[key]["prompt_tokens"] == 100  # inputs are not changed


@pytest.mark.parametrize(
    ("value", "text"),
    [(0, "$0.00"), (0.00004, "< $0.0001"), (0.0042, "$0.0042"), (0.1234, "$0.123"), (1234.567, "$1,234.57")],
)
def test_amounts_show_enough_digits_to_be_useful(value: float, text: str) -> None:
    assert format_usd(value) == text


def test_every_suggested_model_has_a_price() -> None:
    defaults = load_app_defaults(ROOT / "config.yaml")
    suggested = {model.casefold() for models in defaults.model_suggestions.values() for model in models}

    assert suggested <= set(defaults.prices)
