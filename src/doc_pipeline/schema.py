from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

from .models import FieldSpec, ModelResult

# Longest text kept from one model value; anything longer is cut. Model output
# can be steered by text hidden in a PDF, so it is bounded like any input.
MAX_VALUE_CHARS = 1_000


def field_payload(response: dict[str, Any], specs: Iterable[FieldSpec]) -> dict[str, Any]:
    values = response.get("fields", response)
    if not isinstance(values, dict):
        raise ValueError("Model JSON must contain an object named fields.")
    output: dict[str, Any] = {}
    for spec in specs:
        value = values.get(spec.name)
        if isinstance(value, (dict, list)):
            value = None
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        if not (isinstance(value, (str, int, float, bool)) or value is None):
            value = str(value)
        output[spec.name] = value[:MAX_VALUE_CHARS] if isinstance(value, str) else value
    return output


def parse_decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    raw = unicodedata.normalize("NFKC", str(value)).strip().replace("−", "-")
    if not raw:
        return None
    wrapped_negative = raw.startswith("(") and raw.endswith(")")
    if wrapped_negative:
        inner = raw[1:-1].strip()
        wrapped_negative = not inner.startswith("-")
        raw = inner
    try:
        direct = Decimal(raw)
    except InvalidOperation:
        direct = None
    if direct is not None:
        if not direct.is_finite():
            return None
        return -direct if wrapped_negative else direct

    raw = re.sub(r"[^0-9,.-]", "", raw)
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            raw = raw.replace(".", "").replace(",", ".")
        else:
            raw = raw.replace(",", "")
    elif "," in raw:
        # Treat a single comma with one or two trailing digits as a decimal
        # separator; otherwise regard commas as thousands separators.
        tail = raw.rsplit(",", 1)[-1]
        raw = raw.replace(",", ".", 1) if len(tail) in (1, 2) else raw.replace(",", "")
    try:
        result = Decimal(raw)
        if not result.is_finite():
            return None
        return -result if wrapped_negative else result
    except InvalidOperation:
        return None


def _iso_date(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10]).isoformat()
    except ValueError:
        return None


def canonical_value(value: Any, field_type: str) -> str | None:
    if value is None or isinstance(value, str) and not value.strip():
        return None
    if field_type == "decimal":
        number = parse_decimal(value)
        if number is None:
            return None
        return "0" if number == 0 else format(number.normalize(), "f")
    if field_type == "integer":
        try:
            number = Decimal(str(value).strip())
            if number.is_finite() and number == number.to_integral_value():
                return str(int(number))
            return None
        except (InvalidOperation, TypeError, ValueError):
            return None
    if field_type == "date":
        return _iso_date(value)
    if field_type == "boolean":
        if isinstance(value, bool):
            return "true" if value else "false"
        normalized = str(value).strip().casefold()
        return normalized if normalized in {"true", "false", "yes", "no", "1", "0"} else None
    normalized = unicodedata.normalize("NFKC", str(value))
    return " ".join(normalized.split()).casefold()


def vote_fields(
    results: list[ModelResult],
    specs: tuple[FieldSpec, ...],
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Vote across usable model results.

    Returns (values, conflict_fields, minority_disagreement_fields). A field is
    a conflict when no two usable models agree on it; it is a minority
    disagreement when a majority exists but at least one usable model differs.
    The validation status is decided by the caller, which knows how many models
    were actually called and whether an arbiter participated.
    """
    available = [result for result in results if result.error is None]
    if len(available) < 2:
        raise ValueError("At least two extraction models must respond.")
    values: dict[str, Any] = {}
    conflicts: list[str] = []
    minority_disagreements: list[str] = []
    by_name = {spec.name: spec for spec in specs}
    for field_name, spec in by_name.items():
        candidates: list[tuple[str, Any]] = []
        for result in available:
            value = result.fields.get(field_name)
            canonical = canonical_value(value, spec.type)
            if canonical is not None:
                candidates.append((canonical, value))
        counts = Counter(key for key, _ in candidates)
        if not counts:
            values[field_name] = None
            continue
        winner, votes = counts.most_common(1)[0]
        if votes < 2:
            values[field_name] = None
            conflicts.append(field_name)
            continue
        values[field_name] = next(value for key, value in candidates if key == winner)
        if votes < len(available):
            minority_disagreements.append(field_name)
    return values, conflicts, minority_disagreements


def content_fingerprint(
    fields: dict[str, Any],
    specs: tuple[FieldSpec, ...],
    fingerprint_fields: tuple[str, ...],
) -> str | None:
    # Without duplicate-key fields every document would share one fingerprint.
    if not fingerprint_fields:
        return None
    by_name = {spec.name: spec for spec in specs}
    pieces: list[str] = []
    for name in fingerprint_fields:
        value = canonical_value(fields.get(name), by_name[name].type)
        if value is None:
            return None
        pieces.append(value.replace("\\", "\\\\").replace("|", "\\|"))
    canonical = "|".join(pieces)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def excel_value(value: Any, spec: FieldSpec) -> Any:
    if value is None:
        return None
    if spec.type == "decimal":
        number = parse_decimal(value)
        if number is not None:
            return float(number)
    elif spec.type == "integer":
        normalized = canonical_value(value, "integer")
        if normalized is not None:
            return int(normalized)
    elif spec.type == "boolean":
        normalized = canonical_value(value, "boolean")
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    elif spec.type == "date":
        normalized = _iso_date(value)
        if normalized:
            return date.fromisoformat(normalized)
    return excel_safe_string(str(value))


def excel_safe_string(value: str) -> str:
    # Strip XML-illegal control characters and force potentially executable
    # spreadsheet values to remain text.
    clean = "".join(char for char in value if char in "\t\n\r" or ord(char) >= 32)
    if clean.lstrip()[:1] in {"=", "+", "-", "@"}:
        return "'" + clean
    return clean


def audit_json(value: Any) -> str:
    return excel_safe_string(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
