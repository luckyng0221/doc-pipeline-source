from __future__ import annotations

import pytest

from doc_pipeline.models import FieldSpec, ModelResult
from doc_pipeline.schema import (
    MAX_VALUE_CHARS,
    canonical_value,
    content_fingerprint,
    excel_safe_string,
    field_payload,
    vote_fields,
)


FIELDS = (
    FieldSpec("vendor", "string", True),
    FieldSpec("invoice_no", "string"),
    FieldSpec("date", "date", True),
    FieldSpec("total", "decimal", True),
)


def _result(name: str, *, vendor: str = "Acme", total: str = "12.50") -> ModelResult:
    return ModelResult(
        name=name,
        fields={"vendor": vendor, "invoice_no": "A-1", "date": "2026-01-02", "total": total},
    )


def test_unanimous_models_report_no_conflicts() -> None:
    values, conflicts, minority = vote_fields([_result("a"), _result("b")], FIELDS)

    assert values["vendor"] == "Acme"
    assert conflicts == []
    assert minority == []


def test_no_majority_is_a_conflict() -> None:
    values, conflicts, minority = vote_fields(
        [_result("a", total="10"), _result("b", total="11"), _result("c", total="12")],
        FIELDS,
    )

    assert values["total"] is None
    assert "total" in conflicts
    assert minority == []


def test_majority_with_dissent_is_a_minority_disagreement() -> None:
    _, conflicts, minority = vote_fields(
        [_result("a"), _result("b"), _result("c", vendor="Other")],
        FIELDS,
    )

    assert conflicts == []
    assert minority == ["vendor"]


def test_single_model_cannot_vote() -> None:
    with pytest.raises(ValueError, match="At least two"):
        vote_fields([_result("a")], FIELDS)


def test_content_fingerprint_requires_every_field_and_normalizes_values() -> None:
    first = {"vendor": " ACME  Ltd ", "invoice_no": "A-1", "date": "2026-01-02", "total": "$1,234.50"}
    second = {"vendor": "acme ltd", "invoice_no": "A-1", "date": "2026-01-02", "total": "1234.5"}

    assert content_fingerprint(first, FIELDS, ("vendor", "invoice_no", "date", "total")) == content_fingerprint(
        second, FIELDS, ("vendor", "invoice_no", "date", "total")
    )
    european = {**second, "total": "1.234,50"}
    assert content_fingerprint(first, FIELDS, ("vendor", "invoice_no", "date", "total")) == content_fingerprint(
        european, FIELDS, ("vendor", "invoice_no", "date", "total")
    )
    assert content_fingerprint({**first, "invoice_no": None}, FIELDS, ("vendor", "invoice_no", "date", "total")) is None
    assert canonical_value("12.500", "decimal") == "12.5"


def test_no_duplicate_key_fields_means_no_fingerprint() -> None:
    # Otherwise every document would share the hash of an empty string.
    assert content_fingerprint({"vendor": "Acme"}, FIELDS, ()) is None


def test_excel_formula_prefixes_are_made_text() -> None:
    value = excel_safe_string("  =HYPERLINK(\"https://example.invalid\")")

    assert value.startswith("'")


def test_long_model_values_are_cut() -> None:
    payload = field_payload({"fields": {"vendor": "A" * 5_000, "total": 12.5, "date": ["2026-01-02"]}}, FIELDS)

    assert payload == {"vendor": "A" * MAX_VALUE_CHARS, "invoice_no": None, "date": None, "total": 12.5}
