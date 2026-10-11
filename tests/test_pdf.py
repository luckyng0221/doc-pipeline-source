from __future__ import annotations

import io
import json
from pathlib import Path

import pymupdf
import pytest
from PIL import Image

from doc_pipeline.errors import PipelineError
from doc_pipeline.pdf import MAX_PAGE_TEXT_CHARS, _page_text, _rebuild_text, extract_pdf_page_list, render_pdf, split_pdf


def _new_page(width: float = 595, height: float = 842) -> tuple[pymupdf.Document, pymupdf.Page]:
    document = pymupdf.open()
    return document, document.new_page(width=width, height=height)


def _save(document: pymupdf.Document, path: Path) -> Path:
    document.save(path)
    document.close()
    return path


def _plain_pdf(path: Path) -> Path:
    document, page = _new_page()
    page.insert_text((72, 72), "Plain page")
    return _save(document, path)


def _white_pixmap(width: int, height: int) -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pixmap.clear_with(255)
    return pixmap


def test_render_and_split_pdf(tmp_path: Path) -> None:
    source = tmp_path / "sample.pdf"
    document = pymupdf.open()
    for number in range(1, 4):
        page = document.new_page()
        page.insert_text((72, 72), f"Page {number}")
    document.new_page()
    document.save(source)
    document.close()

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=10)
    outputs = split_pdf(source, [(1, 2), (3, 3)], tmp_path / "split")

    assert [page.number for page in rendered] == [1, 2, 3, 4]
    assert all(page.jpeg and len(page.dhash) == 16 for page in rendered)
    assert all(page.hidden_text == 0 for page in rendered)
    assert rendered[0].text.strip() == "Page 1"
    assert 0 < rendered[0].ink < 0.05
    assert rendered[3].text == "" and rendered[3].ink == 0
    assert len(outputs) == 2
    page_counts = []
    for output in outputs:
        split_document = pymupdf.open(output)
        page_counts.append(split_document.page_count)
        split_document.close()
    assert page_counts == [2, 1]

    part = pymupdf.open(stream=extract_pdf_page_list(source, [1, 3, 4]), filetype="pdf")
    assert [page.get_text().strip() for page in part] == ["Page 1", "Page 3", ""]
    part.close()


def test_oversized_pages_render_within_the_pixel_limit(tmp_path: Path) -> None:
    # A page may be 200 x 200 inches: at 150 dpi, a 2.7 GB image for one page.
    # The text is large enough to stay readable (over 2 pt) at the reduced
    # render size.
    document, page = _new_page(14_400, 14_400)
    page.insert_text((72, 600), "Page 1", fontsize=400)
    source = _save(document, tmp_path / "huge.pdf")

    rendered = render_pdf(source, dpi=150, jpeg_quality=75, max_pages=1, max_page_pixels=1_000_000)[0]

    with Image.open(io.BytesIO(rendered.jpeg)) as image:
        assert image.width * image.height <= 1_010_000
    assert rendered.text.strip() == "Page 1"
    assert rendered.hidden_text == 0


def test_text_that_is_tiny_at_the_reduced_render_size_is_hidden(tmp_path: Path) -> None:
    # 11 pt text is plain on a normal page, but this page renders at about a
    # thirtieth of dpi, where the same text is far under MIN_VISIBLE_FONT_SIZE.
    document, page = _new_page(14_400, 14_400)
    page.insert_text((72, 600), "Planted 99.99", fontsize=11)
    source = _save(document, tmp_path / "huge_small_text.pdf")

    rendered = render_pdf(source, dpi=150, jpeg_quality=75, max_pages=1, max_page_pixels=1_000_000)[0]

    assert rendered.hidden_text == 1
    assert "Planted 99.99" not in rendered.text


def test_the_same_text_stays_visible_without_a_downscale() -> None:
    document, page = _new_page()
    page.insert_text((72, 72), "Total 12.50", fontsize=11)
    try:
        assert _page_text(page) == (page.get_text(), 0)
        assert _page_text(page, 1.0) == (page.get_text(), 0)
        text, hidden = _page_text(page, 0.1)
        assert hidden == 1
        assert text == ""
        assert _page_text(page, 0.2)[1] == 0  # 11 pt x 0.2 = 2.2 pt, still readable
    finally:
        document.close()


def test_pages_with_oversized_images_are_refused_before_decoding(tmp_path: Path) -> None:
    document, page = _new_page()
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 30), False)
    pixmap.clear_with(200)
    page.insert_image(page.rect, pixmap=pixmap)
    source = _save(document, tmp_path / "image.pdf")

    assert render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1, max_image_pixels=1200)
    with pytest.raises(PipelineError, match=r"Page 1 holds an image too large to process \(40 x 30 pixels\)"):
        render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1, max_image_pixels=1199)


def test_page_text_is_cut(tmp_path: Path) -> None:
    # Wide enough for one line of 30 000 x's, and at a size the text keeps:
    # under MIN_VISIBLE_FONT_SIZE it would count as hidden and be left out of
    # the text instead of cut.
    document, page = _new_page(170_000, 200)
    page.insert_text((10, 100), "x" * 30_000, fontsize=11)
    source = _save(document, tmp_path / "text.pdf")

    rendered = render_pdf(source, dpi=10, jpeg_quality=75, max_pages=1)[0]

    assert rendered.text == "x" * MAX_PAGE_TEXT_CHARS


def test_zero_opacity_text_is_hidden_and_left_out_of_the_page_text(tmp_path: Path) -> None:
    document, page = _new_page()
    page.insert_text((72, 72), "Visible receipt line")
    page.insert_text((72, 400), "Ghost 99.99", fill_opacity=0)
    source = _save(document, tmp_path / "ghost.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 1
    assert "Ghost 99.99" not in rendered.text
    assert rendered.text == "Visible receipt line"


@pytest.mark.parametrize(
    ("opacity", "hidden"),
    [(0.0, 1), (0.001, 1), (0.05, 1), (1.0, 0)],
)
def test_fill_opacity_under_a_tenth_is_hidden(tmp_path: Path, opacity: float, hidden: int) -> None:
    document, page = _new_page()
    planted = f"Ghost {opacity}"
    page.insert_text((72, 72), planted, fill_opacity=opacity)
    source = _save(document, tmp_path / "opacity.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == hidden
    if hidden:
        assert planted not in rendered.text
    else:
        assert planted in rendered.text


def test_tiny_text_is_hidden(tmp_path: Path) -> None:
    document, page = _new_page()
    page.insert_text((72, 72), "Visible receipt line")
    page.insert_text((72, 400), "Tiny 99.99", fontsize=1)
    source = _save(document, tmp_path / "tiny.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 1
    assert "Tiny 99.99" not in rendered.text
    assert "Visible receipt line" in rendered.text


def test_invisible_render_mode_text_without_an_image_is_hidden(tmp_path: Path) -> None:
    document, page = _new_page()
    page.insert_text((72, 72), "Visible receipt line")
    page.insert_text((72, 400), "Invisible 99.99", render_mode=3)
    source = _save(document, tmp_path / "invisible.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 1
    assert "Invisible 99.99" not in rendered.text
    assert "Visible receipt line" in rendered.text


def test_rotated_page_keeps_its_visible_text_and_drops_the_hidden_line(tmp_path: Path) -> None:
    # The trace reads a /Rotate 90 page with rotation 0, so span origins are
    # unrotated: the visible line, near the bottom of the portrait page, is
    # inside the page only when the box is derotated to match.
    document, page = _new_page()
    page.set_rotation(90)
    page.insert_text((72, 800), "Total: 12.50")
    page.insert_text((72, 100), "Secret 99.99", render_mode=3)
    source = _save(document, tmp_path / "rotated.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 1
    assert "Total: 12.50" in rendered.text
    assert "Secret 99.99" not in rendered.text


def test_rebuilt_text_drops_spans_outside_the_page_box(tmp_path: Path) -> None:
    # The extraction clips text whose origin lies outside the page box; the
    # trace does not, so the rebuild clips it the same way - otherwise an
    # off-page line could change an extracted value.
    document, page = _new_page()
    page.insert_text((72, 100), "Total: 12.50")
    page.insert_text((72, 100), "x", render_mode=3)
    page.insert_text((72, 900), "Total: 99.99")  # below the page box
    source = _save(document, tmp_path / "offpage.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 1
    assert "99.99" not in rendered.text
    assert "12.50" in rendered.text


def test_invisible_text_over_a_covering_image_is_an_ocr_layer(tmp_path: Path) -> None:
    document, page = _new_page()
    page.insert_image(page.rect, pixmap=_white_pixmap(595, 842))
    page.insert_text((72, 72), "OCR layer Total 12.50", render_mode=3)
    source = _save(document, tmp_path / "ocr.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 0
    assert "OCR layer Total 12.50" in rendered.text


def test_a_rotated_ocr_layer_over_a_covering_image_is_not_hidden(tmp_path: Path) -> None:
    # The image boxes follow the page's rotation while the trace's span boxes
    # do not; derotated, the scan's image covers the invisible layer again.
    document, page = _new_page()
    page.set_rotation(90)
    page.insert_image(page.rect, pixmap=_white_pixmap(842, 595), keep_proportion=False)
    page.insert_text((72, 800), "OCR layer Total 12.50", render_mode=3)
    source = _save(document, tmp_path / "rotated_ocr.pdf")

    rendered = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)[0]

    assert rendered.hidden_text == 0
    assert "OCR layer Total 12.50" in rendered.text


@pytest.mark.parametrize("method", ["get_texttrace", "get_image_info"])
def test_a_failure_while_detecting_hidden_text_fails_the_file(tmp_path: Path, monkeypatch, method: str) -> None:
    source = _plain_pdf(tmp_path / "plain.pdf")

    def broken(self: pymupdf.Page) -> object:
        raise RuntimeError("the page trace blew up")

    monkeypatch.setattr(pymupdf.Page, method, broken)

    with pytest.raises(PipelineError, match=r"^The PDF could not be rendered\.$"):
        render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)


def test_a_failure_while_rebuilding_visible_text_fails_the_file(tmp_path: Path, monkeypatch) -> None:
    source = _plain_pdf(tmp_path / "plain.pdf")
    # A hidden span makes the page's text be rebuilt from the trace, and a kept
    # span whose characters have no origin, as a confused MuPDF trace could,
    # makes the rebuild fail instead of reporting partial text. (A bad code or
    # origin value is dropped instead; see the next test.)
    hidden = {"type": 3, "opacity": 1.0, "size": 11.0, "bbox": (200.0, 400.0, 260.0, 411.0), "chars": [(120, 0, (200.0, 405.0), None)]}
    broken = {"type": 0, "opacity": 1.0, "size": 11.0, "bbox": (72.0, 63.0, 130.0, 74.0), "chars": [(120, 0, None, None)]}
    monkeypatch.setattr(pymupdf.Page, "get_texttrace", lambda self: [hidden, broken])

    with pytest.raises(PipelineError, match=r"^The PDF could not be rendered\.$"):
        render_pdf(source, dpi=72, jpeg_quality=75, max_pages=1)


def test_rebuilt_text_drops_malformed_chars_and_stays_encodable() -> None:
    def char(code: int, origin: tuple[float, float] = (72.0, 72.0)) -> tuple:
        return (code, 0, origin, None)

    span = {
        "type": 0,
        "opacity": 1.0,
        "size": 11.0,
        "bbox": (72.0, 63.0, 130.0, 74.0),
        "chars": [
            char(ord("A"), (72.0, 72.0)),
            char(0xD800, (80.0, 72.0)),  # a lone surrogate
            char(0x110000, (88.0, 72.0)),  # beyond Unicode
            char(-1, (96.0, 72.0)),  # negative
            char(ord("B"), (float("inf"), 72.0)),  # infinite origin
            char(ord("C"), (104.0, float("nan"))),  # NaN origin
            char(ord("D"), (112.0, 72.0)),
        ],
    }

    text = _rebuild_text([span], pymupdf.Rect(0, 0, 595, 842))

    assert text == "AD"
    assert json.dumps(text, ensure_ascii=False).encode("utf-8")
