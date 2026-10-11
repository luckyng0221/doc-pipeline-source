"""Estimate what a run cost from the tokens each model reply reported.

Prices are USD per 1M tokens from config.yaml (app.prices). Thinking tokens
count as output. Cached-token, batch, and free-tier discounts are ignored, so
the estimate can be a little high; the provider's billing page is exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError
from .providers import USAGE_KEYS

# Tokens per (provider, model ID): USAGE_KEYS plus "calls".
ModelUsage = dict[tuple[str, str], dict[str, int]]


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float


@dataclass(frozen=True)
class ModelCost:
    provider: str
    model_id: str
    calls: int
    input_tokens: int
    output_tokens: int
    # None when config.yaml has no price for this model.
    usd: float | None


def parse_prices(raw: Any) -> dict[str, Price]:
    """Read app.prices: {model ID: {input: USD, output: USD}} per 1M tokens."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError("app.prices must map model IDs to input and output prices.")
    prices: dict[str, Price] = {}
    for model_id, entry in raw.items():
        if not isinstance(entry, dict) or not {"input", "output"} <= set(entry):
            raise ConfigurationError(f"app.prices.{model_id} needs an input and an output price.")
        try:
            price = Price(input=float(entry["input"]), output=float(entry["output"]))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"app.prices.{model_id} prices must be numbers.") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (price.input, price.output)):
            raise ConfigurationError(f"app.prices.{model_id} prices cannot be negative.")
        prices[str(model_id).strip().casefold()] = price
    return prices


def empty_usage() -> dict[str, int]:
    return {key: 0 for key in USAGE_KEYS} | {"calls": 0}


def add_usage(total: ModelUsage, more: ModelUsage) -> ModelUsage:
    """Return a new map with the tokens and calls of both."""
    merged = {key: dict(value) for key, value in total.items()}
    for key, usage in more.items():
        bucket = merged.setdefault(key, empty_usage())
        for name in (*USAGE_KEYS, "calls"):
            bucket[name] = bucket.get(name, 0) + int(usage.get(name, 0) or 0)
    return merged


def estimate(usage: ModelUsage, prices: dict[str, Price]) -> list[ModelCost]:
    costs = []
    for (provider, model_id), tokens in sorted(usage.items()):
        input_tokens = int(tokens.get("prompt_tokens", 0))
        # Gemini reports thinking separately and bills it as output; OpenAI's
        # output count already includes reasoning (its thoughts count is 0).
        output_tokens = int(tokens.get("output_tokens", 0)) + int(tokens.get("thoughts_tokens", 0))
        price = prices.get(model_id.casefold())
        usd = None if price is None else (input_tokens * price.input + output_tokens * price.output) / 1_000_000
        costs.append(ModelCost(provider, model_id, int(tokens.get("calls", 0)), input_tokens, output_tokens, usd))
    return costs


def format_usd(value: float) -> str:
    if value == 0:
        return "$0.00"
    if value < 0.0001:
        return "< $0.0001"
    if value < 0.01:
        return f"${value:.4f}"
    if value < 1:
        return f"${value:.3f}"
    return f"${value:,.2f}"
