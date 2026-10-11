from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from doc_pipeline.boundaries import build_ranges, detect_boundaries, detect_document_ranges
from doc_pipeline.config import LimitsConfig
from doc_pipeline.errors import BoundaryDetectionError, RunStopped
from doc_pipeline.models import ModelConfig
from doc_pipeline.pdf import PageImage, render_pdf
from doc_pipeline.providers import ProviderResponse
from doc_pipeline.rules import BLANK, CONTINUATION, START, PageDecision


class FakeProvider:
    def __init__(self, responses: list[dict], config: ModelConfig | None = None) -> None:
        self.responses = iter(responses)
        self.config = config
        self.prompts: list[str] = []
        self.parts: list[list[bytes]] = []

    def generate_json(self, prompt: str, parts: list[bytes]) -> ProviderResponse:
        self.prompts.append(prompt)
        self.parts.append(list(parts))
        usage = {"prompt_tokens": 10, "output_tokens": 2, "thoughts_tokens": 0, "cached_tokens": 0, "total_tokens": 12}
        return ProviderResponse(data=next(self.responses), usage=usage)


def _pages(count: int, texts: list[str] | None = None) -> list[PageImage]:
    return [
        PageImage(number=index, jpeg=b"fake", dhash="0" * 16, text=texts[index - 1] if texts else "")
        for index in range(1, count + 1)
    ]


def _answer(*pages: tuple[int, bool]) -> dict:
    return {"pages": [{"page": page, "new_document": new, "confidence": 0.9} for page, new in pages]}


def test_boundary_detector_builds_contiguous_ranges() -> None:
    provider = FakeProvider([_answer((2, False), (3, True)), _answer((4, False), (5, True))])
    limits = LimitsConfig(boundary_batch_pages=2)

    assert detect_document_ranges(_pages(5), provider, limits) == [(1, 2), (3, 4), (5, 5)]
    # Page 1 always starts a document, so it is only sent as context.
    assert "1=PDF page 1 (context only)" in provider.prompts[0]


def test_boundary_detector_rejects_incomplete_classification() -> None:
    provider = FakeProvider([_answer((2, False))])

    with pytest.raises(BoundaryDetectionError, match="every candidate"):
        detect_boundaries(_pages(3), provider, LimitsConfig(boundary_batch_pages=2))


def test_rules_split_marked_pages_without_any_model_request() -> None:
    provider = FakeProvider([])
    texts = ["INVOICE\n1 of 2", "INVOICE", "OFFICIAL RECEIPT\n1 of 1", "INVOICE\nPage 1 of 1"]

    result = detect_boundaries(_pages(4, texts), provider, LimitsConfig())

    assert result.ranges == [(1, 2), (3, 3), (4, 4)]
    assert provider.prompts == []
    assert result.audit == []
    assert [row["decided_by"] for row in result.page_rows()] == ["rule"] * 4


def test_unresolved_pages_from_different_places_share_one_request() -> None:
    # Pages 2 and 6 have no clues; pages 3-5 are settled by the "1 of 2" marker.
    texts = ["INVOICE", "INVOICE", "INVOICE\n1 of 2", "INVOICE", "INVOICE", "INVOICE"]
    provider = FakeProvider([_answer((2, False), (6, True))])

    result = detect_boundaries(_pages(6, texts), provider, LimitsConfig(boundary_batch_pages=6))

    assert result.ranges == [(1, 2), (3, 4), (5, 5), (6, 6)]
    assert len(provider.prompts) == 1
    assert len(provider.parts[0]) == 4
    assert "1=PDF page 1 (context only), 2=PDF page 2, 3=PDF page 5 (context only), 4=PDF page 6" in provider.prompts[0]
    by_page = {row["page"]: row for row in result.page_rows()}
    assert by_page[6]["decided_by"] == "model"
    assert by_page[5]["decided_by"] == "rule"


def test_pdf_input_sends_only_the_needed_pages(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    document = pymupdf.open()
    for number in range(1, 6):
        page = document.new_page()
        page.insert_text((72, 72), f"Sample text {number}")
    document.save(source)
    document.close()
    pages = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=10)
    provider = FakeProvider(
        [_answer((2, False), (3, True)), _answer((4, False), (5, True))],
        config=ModelConfig(
            name="boundary",
            provider="gemini",
            model="gemini-3.8-flash",
            api_key_env="GEMINI_API_KEY",
            input="pdf",
        ),
    )

    assert detect_document_ranges(pages, provider, LimitsConfig(boundary_pdf_batch_pages=3), source=source) == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]
    # Later requests re-send the previous page as context inside the PDF part.
    assert "1=PDF page 3 (context only)" in provider.prompts[1]
    for parts in provider.parts:
        part = pymupdf.open(stream=parts[0], filetype="pdf")
        assert part.page_count == 3
        part.close()


def test_blank_pages_stay_inside_a_document_only_before_a_continuation() -> None:
    decisions = [
        PageDecision(1, START, "rule", ""),
        PageDecision(2, BLANK, "rule", ""),
        PageDecision(3, CONTINUATION, "rule", ""),
        PageDecision(4, BLANK, "rule", ""),
        PageDecision(5, START, "rule", ""),
        PageDecision(6, BLANK, "rule", ""),
    ]

    assert build_ranges(decisions) == [(1, 3), (5, 5)]


def test_a_pdf_of_blank_pages_is_rejected() -> None:
    pages = [PageImage(number=index, jpeg=b"", dhash="0" * 16, text="", ink=0.0) for index in (1, 2)]

    with pytest.raises(BoundaryDetectionError, match="blank"):
        detect_boundaries(pages, FakeProvider([]), LimitsConfig())


def test_a_stop_between_boundary_requests_propagates() -> None:
    provider = FakeProvider([_answer((2, False), (3, True))])
    checks = [0]

    def stop() -> None:
        checks[0] += 1
        if checks[0] == 2:  # the first request may run, the second may not
            raise RunStopped("the run was cancelled")

    with pytest.raises(RunStopped, match="the run was cancelled"):
        detect_boundaries(_pages(5), provider, LimitsConfig(boundary_batch_pages=2), stop_check=stop)

    assert len(provider.prompts) == 1  # the second boundary request was never sent


def test_without_a_model_unclear_pages_follow_the_fallback() -> None:
    from doc_pipeline.models import BoundaryRulesConfig

    texts = ["INVOICE\n1 of 1", "INVOICE", "INVOICE"]

    split = detect_boundaries(_pages(3, texts), None, LimitsConfig())
    merged = detect_boundaries(_pages(3, texts), None, LimitsConfig(), rules=BoundaryRulesConfig(fallback="same_document"))

    # Page 2 follows a finished "1 of 1" document; page 3 has no clue at all.
    assert split.ranges == [(1, 1), (2, 2), (3, 3)]
    assert merged.ranges == [(1, 1), (2, 3)]
    assert [row["decided_by"] for row in split.page_rows()] == ["rule", "rule", "guess"]
    assert split.page_rows()[2]["reason"] == "no AI model: guessed new document"
    assert split.audit == []
