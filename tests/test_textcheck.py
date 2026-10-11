from __future__ import annotations

from doc_pipeline.models import FieldSpec
from doc_pipeline.textcheck import check_against_text

FIELDS = (
    FieldSpec("vendor", "string", True),
    FieldSpec("invoice_no", "string"),
    FieldSpec("date", "date", True),
    FieldSpec("total", "decimal", True),
    FieldSpec("document_type", "string", True, verify_in_text=False),
)
# OCR text as a scanner writes it: a garbled letter, a split reference
# number, and "I" read for "1" in the total.
TEXT = """CONTOH TRAD�NG SDN, BHD.
INVOICE
Invoice No : INV-77 000315
Date : 05-11-2024
Total RM 1,234. I5
"""
FIELDS_READ = {
    "vendor": "CONTOH TRADING SDN. BHD.",
    "invoice_no": "INV-77000315",
    "date": "2024-11-05",
    "total": 1234.15,
    "document_type": "invoice",
}


def test_values_printed_in_the_text_are_confirmed() -> None:
    assert check_against_text(FIELDS_READ, FIELDS, TEXT, "day_first") == []


def test_ambiguous_numeric_dates_need_a_known_date_order() -> None:
    assert check_against_text(FIELDS_READ, FIELDS, TEXT) == ["date was not found in the PDF text"]
    # Written day first, 05-11-2024 is 5 November, never 11 May.
    assert check_against_text({**FIELDS_READ, "date": "2024-05-11"}, FIELDS, TEXT, "day_first") == [
        "date was not found in the PDF text"
    ]
    worded = TEXT.replace("05-11-2024", "5th Nov 2024")
    assert check_against_text(FIELDS_READ, FIELDS, worded) == []


def test_month_and_year_alone_do_not_confirm_a_day() -> None:
    # "March 2024" must not read as 20 March 2024 (day 20, year "24").
    text = TEXT + "Statement for March 2024\n"
    fields = (FieldSpec("date", "date"),)

    assert check_against_text({"date": "2024-03-20"}, fields, text) == ["date was not found in the PDF text"]
    assert check_against_text({"date": "2024-03-20"}, fields, text + "Paid on March 20, 2024\n") == []


def test_reference_numbers_never_match_part_of_a_longer_one() -> None:
    text = TEXT + "PO No: INV-2024-0015\nAccount 98765432\n"
    fields = (FieldSpec("ref", "string"),)

    assert check_against_text({"ref": "INV-2024-0015"}, fields, text) == []
    assert check_against_text({"ref": "INV 2024 0015"}, fields, text) == []
    for truncated in ("INV-2024-001", "NV-2024-0015", "8765432"):
        assert check_against_text({"ref": truncated}, fields, text) == ["ref was not found in the PDF text"]


def test_overlong_values_with_digits_are_not_matched() -> None:
    # The pattern grows with the value, and a value steered by text hidden in a
    # PDF could be long enough to stall the check.
    reference = "A1" * 150
    fields = (FieldSpec("ref", "string"),)

    assert check_against_text({"ref": reference}, fields, TEXT + reference + "\n") == ["ref was not found in the PDF text"]
    assert check_against_text({"ref": reference[:200]}, fields, TEXT + reference[:200] + "\n") == []


def test_wrong_or_missing_values_are_not_confirmed() -> None:
    fields = {**FIELDS_READ, "vendor": "SAMPLE SUPPLIES SDN BHD", "invoice_no": "INV-77000316", "date": None, "total": 1234.75}

    assert check_against_text(fields, FIELDS, TEXT, "day_first") == [
        "vendor was not found in the PDF text",
        "invoice_no was not found in the PDF text",
        "date is missing",
        "total was not found in the PDF text",
    ]
    assert check_against_text({**FIELDS_READ, "date": "2024-13-45"}, FIELDS, TEXT, "day_first") == [
        "date is not a valid date"
    ]


def test_zero_amounts_and_missing_text_layers_are_not_confirmed() -> None:
    assert check_against_text({**FIELDS_READ, "total": 0}, FIELDS, TEXT + "Balance 0.00\n", "day_first") == [
        "total was not found in the PDF text"
    ]
    assert check_against_text(FIELDS_READ, FIELDS, "", "day_first") == [
        "no PDF text layer to check vendor, invoice_no, date, total"
    ]


def test_trailing_table_border_does_not_change_an_amount() -> None:
    text = TEXT + "Final Total 3,565.40|\n"

    assert check_against_text({**FIELDS_READ, "total": "3565.40"}, FIELDS, text, "day_first") == []
