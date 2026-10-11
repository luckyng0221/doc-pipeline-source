"""Local boundary rules that run before any boundary-model request.

Pages are classified from the PDF text layer and the rendered image only.
A page the rules cannot decide with certainty is left UNRESOLVED and is the
only kind of page sent to the boundary model.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .models import BoundaryRulesConfig
from .pdf import PageImage

START = "start"
CONTINUATION = "continuation"
BLANK = "blank"
UNRESOLVED = "unresolved"

@dataclass(frozen=True)
class PageFacts:
    number: int
    text_chars: int
    blank: bool
    marker: tuple[int, int | None] | None
    marker_conflict: bool
    titles: frozenset[str]

    def marker_label(self) -> str:
        if self.marker is None:
            return ""
        number, total = self.marker
        return f"{number} of {total}" if total else f"page {number}"


@dataclass(frozen=True)
class PageDecision:
    page: int
    decision: str
    method: str
    reason: str
    confidence: float | None = None


# OCR often reads "1" as I, i, or l.
_NUM = r"[0-9Iil]{1,3}"
# Whitespace is matched possessively (\s*+, \s++). With plain \s*, a long run
# of spaces after "Page" made the three optional gaps backtrack in cubic time,
# and one crafted page could stall the whole server.
_LABEL = r"(?:page|pg|sheet|halaman|muka\s++surat)"
_LABELLED_TOTAL = re.compile(
    rf"(?<![A-Za-z]){_LABEL}\s*+(?:no\.?)?\s*+[:.#]?\s*+(?P<n>{_NUM})\s*+(?:of|/|dari)\s*+(?P<m>{_NUM})(?![0-9A-Za-z])",
    re.IGNORECASE,
)
_LABELLED_ONLY = re.compile(rf"^\W*{_LABEL}\s*+(?:no\.?)?\s*+[:.#]?\s*+(?P<n>{_NUM})\W*$", re.IGNORECASE)
_BARE_TOTAL = re.compile(rf"^\W*(?P<n>{_NUM})\s*+(?:of|dari)\s*+(?P<m>{_NUM})\W*$", re.IGNORECASE)
_ONES = str.maketrans({"I": "1", "i": "1", "l": "1", "L": "1"})


def _ocr_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    digits = raw.translate(_ONES)
    return int(digits) if digits.isdigit() else None


def page_markers(text: str) -> set[tuple[int, int | None]]:
    """Return every "Page N of M" style marker printed on the page."""
    found: set[tuple[int, int | None]] = set()
    for line in text.splitlines():
        matches = list(_LABELLED_TOTAL.finditer(line))
        for pattern in (_LABELLED_ONLY, _BARE_TOTAL):
            match = pattern.match(line)
            if match:
                matches.append(match)
        for match in matches:
            number = _ocr_int(match.group("n"))
            total = _ocr_int(match.groupdict().get("m"))
            if number is None or number < 1 or (total is not None and not number <= total <= 999):
                continue
            found.add((number, total))
    return found


def _title_key(value: str) -> str:
    return " ".join(re.sub(r"[^A-Z]+", " ", unicodedata.normalize("NFKC", value).upper()).split())


def _title_lookup(titles: dict[str, tuple[str, ...]]) -> dict[str, str]:
    return {_title_key(phrase): kind for kind, phrases in titles.items() for phrase in phrases}


def page_titles(text: str, titles: dict[str, tuple[str, ...]]) -> frozenset[str]:
    lookup = _title_lookup(titles)
    return frozenset(lookup[key] for key in map(_title_key, text.splitlines()) if key in lookup)


def first_title(text: str, titles: dict[str, tuple[str, ...]]) -> str | None:
    """Return the document type of the first title line on the page, if any."""
    lookup = _title_lookup(titles)
    return next((lookup[key] for key in map(_title_key, text.splitlines()) if key in lookup), None)


def analyze_page(page: PageImage, rules: BoundaryRulesConfig) -> PageFacts:
    text = page.text or ""
    text_chars = sum(char.isalnum() for char in text)
    markers = page_markers(text)
    numbers = {number for number, _ in markers}
    totals = {total for _, total in markers if total is not None}
    marker: tuple[int, int | None] | None = None
    if markers and len(numbers) == 1 and len(totals) <= 1:
        marker = (numbers.pop(), next(iter(totals), None))
    return PageFacts(
        number=page.number,
        text_chars=text_chars,
        blank=text_chars < 10 and page.ink < rules.blank_max_ink and not markers,
        marker=marker,
        marker_conflict=bool(markers) and marker is None,
        titles=page_titles(text, rules.titles),
    )


def decide_pages(facts: list[PageFacts], *, blank_separator: bool) -> list[PageDecision]:
    """Classify each page as START, CONTINUATION, BLANK, or UNRESOLVED.

    Each clue has a strength: the page's own marker (3), what the previous
    marker implies or a change of title (2), and a blank separator (1). A page
    is decided only when no clue of strength 2 or more points the other way.
    """
    decisions: list[PageDecision] = []
    previous: PageFacts | None = None
    # Continuation pages still expected from the last "N of M" marker; None
    # when unknown.
    expected: int | None = None
    expected_source = ""
    after_blank = False
    for page in facts:
        if page.blank:
            decisions.append(PageDecision(page.number, BLANK, "rule", "blank page"))
            after_blank = previous is not None
            continue
        if previous is None:
            decision = PageDecision(page.number, START, "rule", "first page")
        else:
            clues: list[tuple[str, int, str]] = []
            if page.marker is not None:
                number, _ = page.marker
                clues.append((START if number == 1 else CONTINUATION, 3, f"marked {page.marker_label()}"))
            if expected is not None:
                if expected > 0:
                    clues.append((CONTINUATION, 2, f"{expected_source}, so more pages follow"))
                else:
                    clues.append((START, 2, f"{expected_source}, so that document has ended"))
            if previous.titles and page.titles and previous.titles.isdisjoint(page.titles):
                clues.append(
                    (START, 2, f"title changed from {'/'.join(sorted(previous.titles))} to {'/'.join(sorted(page.titles))}")
                )
            if after_blank and blank_separator:
                clues.append((START, 1, "follows a blank separator page"))
            decision = _combine(page, clues)
        decisions.append(decision)

        if page.marker is not None and page.marker[1] is not None and decision.decision != UNRESOLVED:
            number, total = page.marker
            expected = total - number
            expected_source = f"page {page.number} is marked {number} of {total}"
        elif decision.decision == CONTINUATION and expected:
            expected -= 1
        else:
            expected = None
        previous = page
        after_blank = False
    return decisions


def _combine(page: PageFacts, clues: list[tuple[str, int, str]]) -> PageDecision:
    if not clues:
        reason = "conflicting page markers" if page.marker_conflict else "no page marker or title change"
        return PageDecision(page.number, UNRESOLVED, "", reason)
    strongest = max(strength for _, strength, _ in clues)
    leaders = {value for value, strength, _ in clues if strength == strongest}
    winner = next(iter(leaders))
    opposed = any(value != winner and strength >= 2 for value, strength, _ in clues)
    if len(leaders) > 1 or opposed:
        detail = "; ".join(f"{reason} ({value})" for value, _, reason in clues)
        return PageDecision(page.number, UNRESOLVED, "", f"conflicting clues: {detail}")
    reasons = [reason for value, _, reason in clues if value == winner]
    return PageDecision(page.number, winner, "rule", "; ".join(reasons))
