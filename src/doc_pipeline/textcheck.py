"""Decide whether one model's answer can be trusted without asking another.

A value counts as confirmed only when it is printed in the PDF's own text
layer (digital text or the scanner's OCR). Nothing here calls a model; when a
check fails, the processor asks the next model instead.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal
from typing import Any

from .models import FieldSpec
from .schema import canonical_value, parse_decimal

MIN_TEXT_CHARS = 40
# Values with digits are matched with a pattern that grows with the value, so
# longer ones are not checked (no printed reference is this long) and the next
# model is asked instead.
MAX_PATTERN_CHARS = 200
MONTHS = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
# OCR often reads 1 as I/l/i and 0 as O/o inside numbers. A letter right
# after a number is left alone unless more digits or a separator follow.
_OCR_NEAR_DIGIT = re.compile(r"[IliOo](?=\d)|(?<=\d)[IliOo](?=[\d.,/-])")
_DIGIT_FIX = {"I": "1", "l": "1", "i": "1", "O": "0", "o": "0"}
_COMPACT_FIX = str.maketrans({"o": "0", "i": "1", "l": "1"})
_NUMBER = re.compile(r"\d[\d.,]*\d|\d")
_LETTERS = re.compile(r"[^\W\d_]+")


def check_against_text(
    fields: dict[str, Any],
    specs: tuple[FieldSpec, ...],
    text: str,
    date_order: str = "unknown",
) -> list[str]:
    """Return why the values are not confirmed by the text; empty when they all are.

    date_order ("day_first", "month_first", or "unknown") says how numeric
    dates such as 02/12/2024 are written in these documents.
    """
    document = DocumentText(text, date_order)
    problems: list[str] = []
    unchecked: list[str] = []
    for spec in specs:
        value = fields.get(spec.name)
        if value is None or isinstance(value, str) and not value.strip():
            if spec.required:
                problems.append(f"{spec.name} is missing")
            continue
        if canonical_value(value, spec.type) is None:
            problems.append(f"{spec.name} is not a valid {spec.type}")
            continue
        if not spec.verify_in_text or spec.type == "boolean":
            continue
        if not document.usable:
            unchecked.append(spec.name)
        elif not document.contains(value, spec.type):
            problems.append(f"{spec.name} was not found in the PDF text")
    if unchecked:
        problems.append(f"no PDF text layer to check {', '.join(unchecked)}")
    return problems


class DocumentText:
    def __init__(self, text: str, date_order: str = "unknown") -> None:
        self.date_order = date_order
        clean = unicodedata.normalize("NFKC", text).replace("�", "")
        self.usable = sum(char.isalnum() for char in clean) >= MIN_TEXT_CHARS
        self._compact = _compact(clean)
        # Like _compact, but punctuation and spaces are kept, so a reference
        # number can be matched whole and never as part of a longer one.
        self._spaced = clean.casefold().translate(_COMPACT_FIX)
        self._words = set(_LETTERS.findall(clean.casefold()))
        # Digits with OCR slips fixed and stray spaces around separators
        # removed, so "1,234. 15" reads 1,234.15 and "02/12/ 2024" a date.
        numeric = re.sub(r"(?<=\d)\s*([.,/-])\s*(?=\d)", r"\1", fix_ocr_digits(clean))
        self._numeric = numeric
        self._folded = " ".join(numeric.casefold().split())
        self._amounts: set[Decimal] = set()
        self._integers: set[int] = set()
        for match in _NUMBER.finditer(numeric):
            token = match.group()
            if token.isdigit():
                self._integers.add(int(token))
            elif re.search(r"(?:\.\d{1,4}|,\d{1,2})$", token):
                amount = parse_decimal(token)
                if amount is not None:
                    self._amounts.add(abs(amount))

    def contains(self, value: Any, field_type: str) -> bool:
        if field_type == "decimal":
            amount = parse_decimal(value)
            # A zero amount appears on almost every statement, so it proves nothing.
            return amount is not None and amount != 0 and abs(amount) in self._amounts
        if field_type == "integer":
            canonical = canonical_value(value, "integer")
            return canonical is not None and abs(int(canonical)) in self._integers
        if field_type == "date":
            canonical = canonical_value(value, "date")
            return canonical is not None and self._has_date(date.fromisoformat(canonical))
        return self._has_string(str(value))

    def _has_string(self, value: str) -> bool:
        stripped = "".join(value.split())
        if stripped.isdigit():
            return re.search(rf"(?<![^\W_]){re.escape(stripped)}(?![^\W_])", self._numeric) is not None
        compact = _compact(value)
        if any(char.isdigit() for char in value):
            # Reference numbers must match exactly, apart from OCR spacing and
            # punctuation: "INV-2024-0015" does not confirm "INV-2024-001".
            if not 4 <= len(compact) <= MAX_PATTERN_CHARS:
                return False
            body = r"[\W_]*".join(re.escape(char) for char in compact)
            return re.search(rf"(?<![^\W_]){body}(?![^\W_])", self._spaced) is not None
        if len(compact) >= 4 and compact in self._compact:
            return True
        words = [word for word in _LETTERS.findall(unicodedata.normalize("NFKC", value).casefold()) if len(word) >= 2]
        return bool(words) and all(self._has_word(word) for word in words)

    def _has_word(self, word: str) -> bool:
        if word in self._words:
            return True
        # Allow one OCR slip in longer name words ("M0TOR" for "MOTOR").
        return len(word) >= 5 and any(_within_one_edit(word, candidate) for candidate in self._words)

    def _has_date(self, day: date) -> bool:
        days = sorted({str(day.day), f"{day.day:02d}"})
        months = sorted({str(day.month), f"{day.month:02d}"})
        years = (f"{day.year:04d}", f"{day.year % 100:02d}")
        numeric = [f"{day.year:04d}{sep}{month}{sep}{d}" for sep in "-/." for month in months for d in days]
        # 02/12/2024 could be 2 December or 12 February, so day-and-month
        # numbers only count when the order cannot be misread or is known.
        if day.day > 12 or day.day == day.month:
            orders = ("day_first", "month_first")
        elif self.date_order in ("day_first", "month_first"):
            orders = (self.date_order,)
        else:
            orders = ()
        numeric += [
            f"{d}{sep}{month}{sep}{year}" if order == "day_first" else f"{month}{sep}{d}{sep}{year}"
            for order in orders
            for sep in "-/."
            for year in years
            for d in days
            for month in months
        ]
        pattern = "|".join(re.escape(candidate) for candidate in numeric)
        if re.search(rf"(?<!\d)(?:{pattern})(?!\d)", self._numeric):
            return True
        name = MONTHS[day.month - 1]
        names = "|".join(dict.fromkeys((name, "sept" if day.month == 9 else name, name[:3])))
        day_part = rf"0?{day.day}(?:st|nd|rd|th)?"
        year_part = rf"(?:{years[0]}|{years[1]})"
        # Month first, the day and year need a separator, or "March 2024"
        # would read as day 20 of 2024 (year "24").
        worded = (
            rf"(?<!\d){day_part}[\s./-]*(?<![a-z])(?:{names})\.?[\s.,/-]*{year_part}(?!\d)",
            rf"(?<![a-z])(?:{names})\.?[\s./-]*{day_part}[\s.,/-]+{year_part}(?!\d)",
        )
        return any(re.search(pattern, self._folded) for pattern in worded)


def _compact(value: str) -> str:
    folded = unicodedata.normalize("NFKC", value).casefold().replace("�", "")
    return "".join(char for char in folded if char.isalnum()).translate(_COMPACT_FIX)


def fix_ocr_digits(text: str) -> str:
    for _ in range(3):
        fixed = _OCR_NEAR_DIGIT.sub(lambda match: _DIGIT_FIX[match.group()], text)
        if fixed == text:
            break
        text = fixed
    return text


def _within_one_edit(first: str, second: str) -> bool:
    if abs(len(first) - len(second)) > 1:
        return False
    if len(first) > len(second):
        first, second = second, first
    index = 0
    while index < len(first) and first[index] == second[index]:
        index += 1
    if len(first) == len(second):
        return first[index + 1 :] == second[index + 1 :]
    return first[index:] == second[index + 1 :]
