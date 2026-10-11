"""Fill fields from the PDF text layer without any AI model.

Used when no API key is entered. A value is taken from the text printed next
to one of the field's labels, so results depend on the scan's text layer and
are less reliable than a model reading the page.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from .models import FieldSpec
from .rules import first_title
from .schema import parse_decimal
from .textcheck import MONTHS, fix_ocr_digits

# How many lines below a label to look when the value is not on the label's line.
LOOKAHEAD_LINES = 3
# Lines are cut to this length. Values sit near their labels, and the amount
# patterns slow down sharply on very long runs of digits.
MAX_LINE_CHARS = 500
_SEPARATORS = re.compile(r"^[\s:.\-#=|]+")
_CENTS = re.compile(r"\d[\d,]*\.\d{2}(?!\d)|\d[\d.]*,\d{2}(?!\d)")
_DECIMAL = re.compile(r"\d[\d,]*\.\d{1,4}")
_INTEGER = re.compile(r"\d+")
# Currency marks are a Currency field's value, but noise after any other label.
_CURRENCY_MARKS = frozenset({"rm", "myr"})
# Column headings and currency marks that follow labels in scanned tables;
# they are never a text field's value.
_TABLE_WORDS = _CURRENCY_MARKS | frozenset(
    {"total", "subtotal", "amount", "qty", "quantity", "description", "price", "unit", "uom",
     "no", "date", "page", "item", "code", "balance", "discount"}
)
_ISO_DATE = re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)")
_NUMERIC_DATE = re.compile(r"(?<!\d)(\d{1,2})[-/.](\d{1,2})[-/.](\d{4}|\d{2})(?!\d)")
_MONTH_NAMES = "|".join(sorted({name for month in MONTHS for name in (month, month[:3], "sept")}, key=len, reverse=True))
_DAY_MONTH_YEAR = re.compile(
    rf"(?<!\d)(\d{{1,2}})(?:st|nd|rd|th)?[\s./-]*(?<![a-z])({_MONTH_NAMES})\.?[\s.,/-]*(\d{{4}}|\d{{2}})(?!\d)"
)
_MONTH_DAY_YEAR = re.compile(rf"(?<![a-z])({_MONTH_NAMES})\.?[\s./-]*(\d{{1,2}})(?:st|nd|rd|th)?[\s.,/-]*(\d{{4}})(?!\d)")


@dataclass
class RuleExtraction:
    fields: dict[str, Any]
    notes: list[str] = field(default_factory=list)


def extract_fields(
    page_texts: list[str],
    specs: tuple[FieldSpec, ...],
    date_order: str,
    titles: dict[str, tuple[str, ...]],
) -> RuleExtraction:
    lines = [" ".join(line.split())[:MAX_LINE_CHARS] for text in page_texts for line in text.splitlines()]
    lines = [line for line in lines if line]
    labels = _TABLE_WORDS | {_letters(keyword) for spec in specs for keyword in spec.keywords}
    values: dict[str, Any] = {}
    missing: list[str] = []
    for spec in specs:
        if _is_document_type(spec):
            value = first_title(page_texts[0], titles) if page_texts else None
        else:
            field_labels = labels - _CURRENCY_MARKS if _is_currency(spec) else labels
            value = _from_labels(lines, spec, date_order, field_labels)
            if value is None and spec.type == "date":
                dates = find_dates("\n".join(lines), date_order)
                value = dates[0].isoformat() if dates else None
        values[spec.name] = value
        if value is None:
            missing.append(spec.name)
    notes = [f"Not found in the PDF text: {', '.join(missing)}"] if missing else []
    return RuleExtraction(fields=values, notes=notes)


def find_dates(text: str, date_order: str) -> list[date]:
    """Dates in reading order; numeric dates with an unclear order need date_order."""
    clean = re.sub(r"(?<=\d)\s*([.,/-])\s*(?=\d)", r"\1", fix_ocr_digits(unicodedata.normalize("NFKC", text)))
    found: list[tuple[int, date]] = []
    for match in _ISO_DATE.finditer(clean):
        _add_date(found, match.start(), int(match[1]), int(match[2]), int(match[3]))
    for match in _NUMERIC_DATE.finditer(clean):
        first, second, year = int(match[1]), int(match[2]), _full_year(match[3])
        if first > 12 or date_order == "day_first":
            day, month = first, second
        elif second > 12 or date_order == "month_first":
            day, month = second, first
        else:
            continue
        _add_date(found, match.start(), year, month, day)
    folded = clean.casefold()
    for match in _DAY_MONTH_YEAR.finditer(folded):
        _add_date(found, match.start(), _full_year(match[3]), _month_number(match[2]), int(match[1]))
    for match in _MONTH_DAY_YEAR.finditer(folded):
        _add_date(found, match.start(), int(match[3]), _month_number(match[1]), int(match[2]))
    return [value for _, value in sorted(found, key=lambda item: item[0])]


def _is_document_type(spec: FieldSpec) -> bool:
    return re.sub(r"[\s_]+", "", spec.name.casefold()) == "documenttype"


def _is_currency(spec: FieldSpec) -> bool:
    return re.sub(r"[\s_]+", "", spec.name.casefold()) in {"currency", "curr"}


def _from_labels(lines: list[str], spec: FieldSpec, date_order: str, labels: frozenset[str]) -> Any:
    patterns = [_label_pattern(label) for label in spec.keywords or (spec.name.replace("_", " "),)]
    matches = [
        (index, match)
        for index, line in enumerate(lines)
        for pattern in patterns
        if (match := pattern.search(line))
    ]
    # Totals and tax usually appear last on a document; other values first.
    if spec.type == "decimal":
        matches.reverse()
    for index, match in matches:
        candidates = [_SEPARATORS.sub("", lines[index][match.end() :])]
        candidates += lines[index + 1 : index + 1 + LOOKAHEAD_LINES]
        # Amounts: an amount in cents on any nearby line beats a bare number.
        for cents_only in (True, False) if spec.type == "decimal" else (False,):
            for candidate in candidates:
                value = _parse(candidate, spec.type, date_order, labels, cents_only) if candidate else None
                if value is not None:
                    return value
    return None


def _label_pattern(label: str) -> re.Pattern[str]:
    tokens = re.findall(r"\w+|[^\w\s]", label)
    body = r"\s*".join(re.escape(token) for token in tokens)
    start = r"(?<![A-Za-z0-9])" if tokens and tokens[0][0].isalnum() else ""
    end = r"(?![A-Za-z0-9])" if tokens and tokens[-1][-1].isalnum() else ""
    return re.compile(start + body + end, re.IGNORECASE)


def _parse(candidate: str, field_type: str, date_order: str, labels: frozenset[str], cents_only: bool = False) -> Any:
    if field_type == "date":
        dates = find_dates(candidate, date_order)
        return dates[0].isoformat() if dates else None
    if field_type in ("decimal", "integer"):
        clean = re.sub(r"(?<=\d)\s*([.,])\s*(?=\d)", r"\1", fix_ocr_digits(candidate))
        if field_type == "integer":
            match = _INTEGER.search(clean)
            return int(match.group()) if match else None
        # Prefer an amount in cents ("45.00") over unit prices ("7.5006") and counts.
        match = _CENTS.search(clean)
        if match is None and not cents_only:
            match = _DECIMAL.search(clean) or _INTEGER.search(clean)
        amount = parse_decimal(match.group()) if match else None
        return str(amount) if amount is not None else None
    if field_type == "boolean":
        word = candidate.strip().split(" ")[0].casefold().rstrip(".")
        return {"yes": True, "y": True, "true": True, "no": False, "n": False, "false": False}.get(word)
    text = candidate.strip()
    # Punctuation, another label ("Date:", "Total"), or a bare amount is not a text value.
    if text.endswith(":") or not re.search(r"[A-Za-z0-9]", text) or _letters(text) in labels:
        return None
    if _CENTS.fullmatch(text.replace(" ", "")):
        return None
    return text[:120]


def _letters(value: str) -> str:
    return re.sub(r"[^a-z]+", "", value.casefold())


def _full_year(raw: str) -> int:
    year = int(raw)
    if len(raw) == 2:
        return 2000 + year if year < 70 else 1900 + year
    return year


def _month_number(name: str) -> int:
    return next(index for index, month in enumerate(MONTHS, start=1) if month.startswith(name[:3]))


def _add_date(found: list[tuple[int, date]], position: int, year: int, month: int, day: int) -> None:
    try:
        found.append((position, date(year, month, day)))
    except ValueError:
        pass
