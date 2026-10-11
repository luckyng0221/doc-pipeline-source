from __future__ import annotations

import time

from doc_pipeline.models import DEFAULT_TITLES, BoundaryRulesConfig
from doc_pipeline.pdf import PageImage
from doc_pipeline.rules import (
    BLANK,
    CONTINUATION,
    START,
    UNRESOLVED,
    analyze_page,
    decide_pages,
    page_markers,
    page_titles,
)


def _page(number: int, text: str = "", ink: float = 0.05) -> PageImage:
    return PageImage(number=number, jpeg=b"", dhash="0" * 16, text=text, ink=ink)


def _decide(texts: list[str], *, inks: list[float] | None = None, blank_pages: str = "ignore") -> list[str]:
    rules = BoundaryRulesConfig(blank_pages=blank_pages)
    inks = inks or [0.05] * len(texts)
    facts = [analyze_page(_page(number, text, ink), rules) for number, (text, ink) in enumerate(zip(texts, inks), start=1)]
    return [decision.decision for decision in decide_pages(facts, blank_separator=blank_pages == "separator")]


def test_page_markers_tolerate_ocr_noise() -> None:
    assert page_markers("Page\n1 of 3") == {(1, 3)}
    assert page_markers("l of 1") == {(1, 1)}
    assert page_markers("iof 1") == {(1, 1)}
    assert page_markers("Page : i of i") == {(1, 1)}
    assert page_markers(":3 of 3") == {(3, 3)}
    assert page_markers("Invoice No: 12  Page 2/4") == {(2, 4)}
    assert page_markers("Page 2") == {(2, None)}


def test_page_markers_ignore_other_numbers() -> None:
    assert page_markers("Copy 1 of 2") == set()
    assert page_markers("No. 18, Jalan 3/42A") == set()
    assert page_markers("see page 1 for terms") == set()
    assert page_markers("3 of 2") == set()


def test_long_whitespace_runs_do_not_stall_the_marker_search() -> None:
    # Plain \s* gaps backtracked in cubic time: 800 spaces after "Page" took
    # 1.5 s, and the regex holds the GIL, so one page froze every session.
    started = time.perf_counter()

    assert page_markers("Page" + " " * 20_000 + "x") == set()
    assert page_markers("Page" + " " * 20_000 + "2 of 4") == {(2, 4)}
    assert page_markers("Page   no.  :  2   of   4") == {(2, 4)}
    assert time.perf_counter() - started < 0.5


def test_titles_match_whole_lines_only() -> None:
    assert page_titles("OFFICIAL RECEIPT\nThis receipt is valid upon clearance", DEFAULT_TITLES) == {"receipt"}
    assert page_titles("INVOICE NO: 123\nRECEIVED THE ABOVE GOODS", DEFAULT_TITLES) == frozenset()
    assert page_titles("Tax Invoice", DEFAULT_TITLES) == {"invoice"}


def test_markers_and_page_counts_split_the_stream() -> None:
    # Continuation pages repeat the INVOICE title, so the title alone never splits.
    texts = ["INVOICE\n1 of 3", "INVOICE", "INVOICE", "INVOICE\nPage 1 of 1", "OFFICIAL RECEIPT"]

    assert _decide(texts) == [START, CONTINUATION, CONTINUATION, START, START]


def test_pages_without_clues_are_left_for_the_model() -> None:
    assert _decide(["INVOICE", "INVOICE"]) == [START, UNRESOLVED]


def test_conflicting_clues_are_left_for_the_model() -> None:
    # Page 1 promises a second page, but page 2 carries a different title.
    assert _decide(["INVOICE\n1 of 2", "OFFICIAL RECEIPT"]) == [START, UNRESOLVED]
    # A continuation marker right after a finished document.
    assert _decide(["INVOICE\n1 of 1", "INVOICE\n2 of 2"]) == [START, UNRESOLVED]


def test_blank_pages_are_skipped_or_used_as_separators() -> None:
    inks = [0.05, 0.0, 0.05]

    assert _decide(["INVOICE", "", "INVOICE"], inks=inks) == [START, BLANK, UNRESOLVED]
    assert _decide(["INVOICE", "", "INVOICE"], inks=inks, blank_pages="separator") == [START, BLANK, START]
    # Duplex scans: a page marked as a continuation overrides the separator.
    assert _decide(["INVOICE\n1 of 2", "", "INVOICE\n2 of 2"], inks=inks, blank_pages="separator") == [
        START,
        BLANK,
        CONTINUATION,
    ]


def test_image_only_scans_are_not_blank() -> None:
    rules = BoundaryRulesConfig()

    assert not analyze_page(_page(1, "", ink=0.08), rules).blank
    assert analyze_page(_page(1, "", ink=0.001), rules).blank
    assert not analyze_page(_page(1, "2 of 2", ink=0.001), rules).blank
