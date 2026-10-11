from __future__ import annotations

import time

from doc_pipeline.models import DEFAULT_TITLES, FieldSpec
from doc_pipeline.rule_extract import extract_fields, find_dates

PAGE = """ACME TRADING SDN BHD
TAX INVOICE
Invoice No : INV-2024-001
Date: 02/12/2024
Bill To
TN Motor
Sub Total 40.00
SST 6% 2.40
Final Total
RM 42.40
"""
SPECS = (
    FieldSpec("Document Number", keywords=("Invoice No",)),
    FieldSpec("Document Date", "date", keywords=("Invoice Date", "Date")),
    FieldSpec("Customer Name", keywords=("Bill To",)),
    FieldSpec("Tax / SST Amount", "decimal", keywords=("SST",)),
    FieldSpec("Total Amount", "decimal", keywords=("Total",)),
    FieldSpec("Document Type"),
    FieldSpec("PO Number", keywords=("PO No",)),
)


def test_values_are_read_next_to_their_labels() -> None:
    result = extract_fields([PAGE], SPECS, "day_first", DEFAULT_TITLES)

    assert result.fields == {
        "Document Number": "INV-2024-001",
        "Document Date": "2024-12-02",
        "Customer Name": "TN Motor",
        "Tax / SST Amount": "2.40",
        "Total Amount": "42.40",
        "Document Type": "invoice",
        "PO Number": None,
    }
    assert result.notes == ["Not found in the PDF text: PO Number"]


def test_currency_marks_are_a_currency_value_but_not_other_text() -> None:
    specs = (FieldSpec("Currency", keywords=("Currency",)), FieldSpec("Customer Name", keywords=("Customer",)))

    def read(text: str) -> dict:
        return extract_fields([text], specs, "day_first", DEFAULT_TITLES).fields

    assert read("Currency: RM\nTotal 100.00")["Currency"] == "RM"
    assert read("Currency: MYR\nBill To: ABC Sdn Bhd")["Currency"] == "MYR"
    assert read("Customer RM\nABC Sdn Bhd")["Customer Name"] == "ABC Sdn Bhd"


def test_long_digit_lines_do_not_stall_rules_only_extraction() -> None:
    # The amount patterns backtrack in quadratic time on long runs of digits.
    specs = (FieldSpec("Total Amount", "decimal", keywords=("Total",)),)
    started = time.perf_counter()

    result = extract_fields(["Total\n" + "1" * 200_000 + "\nTotal 42.40"], specs, "day_first", DEFAULT_TITLES)

    assert time.perf_counter() - started < 1
    assert result.fields["Total Amount"] == "42.40"


def test_dates_follow_the_date_order_and_fall_back_to_the_first_date() -> None:
    assert extract_fields([PAGE], SPECS[1:2], "month_first", DEFAULT_TITLES).fields["Document Date"] == "2024-02-12"
    unlabelled = (FieldSpec("Due Date", "date", keywords=("Due Date",)),)
    assert extract_fields(["Printed 29 NOV 2024"], unlabelled, "unknown", DEFAULT_TITLES).fields["Due Date"] == "2024-11-29"
    assert [value.isoformat() for value in find_dates("02/12/2024, 13/05/24, 2025-01-31", "unknown")] == [
        "2024-05-13",
        "2025-01-31",
    ]
