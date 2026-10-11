from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import imagehash
import pymupdf
from PIL import Image

from .errors import PipelineError

# Limits for untrusted PDFs. A page may declare up to 200 x 200 inches, which
# at render_dpi would need gigabytes of memory, so a large page renders at a
# lower scale. Bigger embedded images are refused before they are decoded.
# Page text is cut, since every rule and check reads it; a dense real page
# holds well under 10,000 characters.
MAX_PAGE_PIXELS = 25_000_000
MAX_IMAGE_PIXELS = 100_000_000
MAX_PAGE_TEXT_CHARS = 20_000

# Hidden text: a span is hidden when MuPDF traced it as invisible text (PDF
# render mode 3), drew it at an opacity under MIN_VISIBLE_OPACITY, or set an
# effective font size under MIN_VISIBLE_FONT_SIZE, measured at the rendered
# size (a page rendered smaller than dpi counts its text smaller). Clip-only
# text (render mode 7) is a known limit: the trace has no span for it, so it
# is not detected. The count reported per page is capped.
MIN_VISIBLE_FONT_SIZE = 2.0
MIN_VISIBLE_OPACITY = 0.1
INVISIBLE_TEXT_TYPES = (3,)
MAX_HIDDEN_TEXT_SPANS = 1000


@dataclass(frozen=True)
class PageImage:
    number: int
    jpeg: bytes
    dhash: str
    # The page's *visible* text layer (digital text or a scanner's OCR);
    # text drawn invisibly is left out. Empty for image-only scans.
    text: str = ""
    # Share of dark pixels inside the page margins; close to 0 on blank pages.
    ink: float = 1.0
    # How many hidden text spans the page carries, capped at
    # MAX_HIDDEN_TEXT_SPANS; 0 on a page with none.
    hidden_text: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_pdf(
    path: Path,
    *,
    dpi: int,
    jpeg_quality: int,
    max_pages: int,
    max_page_pixels: int = MAX_PAGE_PIXELS,
    max_image_pixels: int = MAX_IMAGE_PIXELS,
) -> list[PageImage]:
    try:
        # Opened from memory: a failed open from the path keeps the file locked
        # on Windows while the error is handled, so it could not be archived.
        document = pymupdf.open(stream=path.read_bytes(), filetype="pdf")
    except Exception as exc:
        raise PipelineError("The file is not a readable PDF.") from exc
    try:
        if document.needs_pass:
            raise PipelineError("Password-protected PDFs are not supported.")
        if document.page_count < 1:
            raise PipelineError("The PDF contains no pages.")
        if document.page_count > max_pages:
            raise PipelineError(f"PDF exceeds the configured {max_pages}-page limit.")
        rendered: list[PageImage] = []
        for index, page in enumerate(document, start=1):
            _check_images(page, index, max_image_pixels)
            scale = _render_scale(page.rect, dpi, max_page_pixels)
            # 1.0 unless the page is rendered below dpi: a model reads the
            # rendered page, so tiny text is judged at that size.
            downscale = scale / (dpi / 72.0)
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            jpeg = pixmap.tobytes("jpeg", jpg_quality=jpeg_quality)
            with Image.open(BytesIO(jpeg)) as image:
                gray = image.convert("L")
                digest = str(imagehash.dhash(gray))
                ink = _ink_ratio(gray)
            text, hidden = _page_text(page, downscale)
            rendered.append(
                PageImage(number=index, jpeg=jpeg, dhash=digest, text=text, ink=ink, hidden_text=hidden)
            )
        return rendered
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("The PDF could not be rendered.") from exc
    finally:
        document.close()


def _render_scale(rect: pymupdf.Rect, dpi: int, max_pixels: int) -> float:
    """dpi / 72, lowered when the page would render to more than max_pixels."""
    scale = dpi / 72.0
    pixels = rect.width * rect.height * scale * scale
    return scale * math.sqrt(max_pixels / pixels) if pixels > max_pixels else scale


def _page_text(page: pymupdf.Page, downscale: float = 1.0) -> tuple[str, int]:
    """The page's visible text and its count of hidden text spans.

    A span is hidden when MuPDF traced it as invisible text (PDF render mode
    3), drew it at an opacity under MIN_VISIBLE_OPACITY, or set an effective
    font size under MIN_VISIBLE_FONT_SIZE, measured at the rendered size: the
    size times downscale, the factor (1.0 when not scaled down) by which the
    page is rendered below dpi. Clip-only text (render mode 7) is a known
    limit: the trace has no span for it, so it stays undetected.
    Invisible text lying over a raster image is excused: it is the scan's OCR
    layer, the page's own text, so it stays in the text and is not counted.
    Any image covering the span excuses it - a small planted one defeats the
    check - a limit accepted here, not coded around. A page with no hidden
    text keeps the text extraction of before, byte for byte.
    """
    # The trace reads the page with rotation 0, so span positions are
    # unrotated, while the page box and the image boxes follow the page's
    # rotation; both are derotated to be comparable with span positions.
    derotation = page.derotation_matrix
    images = [pymupdf.Rect(info["bbox"]) * derotation for info in page.get_image_info()]
    trace = page.get_texttrace()
    kept: list[dict] = []
    hidden = 0
    for span in trace:
        if not _is_hidden_span(span, downscale):
            kept.append(span)
        elif _over_an_image(span["bbox"], images):
            kept.append(span)  # a scanner's OCR layer: invisible, but the page's own text
        else:
            hidden += 1
    if not hidden:
        return page.get_text()[:MAX_PAGE_TEXT_CHARS], 0
    return _rebuild_text(kept, page.rect * derotation), min(hidden, MAX_HIDDEN_TEXT_SPANS)


def _is_hidden_span(span: dict, downscale: float = 1.0) -> bool:
    """Whether the span is drawn invisibly, or too small to read at the rendered size."""
    return (
        span["type"] in INVISIBLE_TEXT_TYPES
        or span["opacity"] < MIN_VISIBLE_OPACITY
        or span["size"] * downscale < MIN_VISIBLE_FONT_SIZE
    )


def _over_an_image(bbox: tuple[float, float, float, float], images: list[pymupdf.Rect]) -> bool:
    """Whether one of the page's raster images covers the whole span box."""
    span = pymupdf.Rect(bbox)
    return any(image.contains(span) for image in images)


def _usable_char(char: tuple) -> bool:
    """Whether a traced char is a real code point with a finite origin.

    A hostile PDF can trace a lone surrogate, a negative code or a code
    beyond Unicode, which would fail the UTF-8 encoding of the reply, and an
    infinite or NaN origin. Such chars are dropped, never a reason to fail
    the file.
    """
    code, origin = char[0], char[2]
    return (
        0 <= code <= 0x10FFFF
        and not 0xD800 <= code <= 0xDFFF
        and math.isfinite(origin[0])
        and math.isfinite(origin[1])
    )


def _rebuild_text(spans: list[dict], page_box: pymupdf.Rect) -> str:
    """The spans' text as lines read top to bottom, left to right.

    A span's chars usually share one line, but a multiline insertion puts
    them on several, so the chars are regrouped by the rounded y of their
    origins. Chars whose origin lies outside the page box are skipped: the
    extraction clips them against the page box and the trace does not. The
    box shares the trace's unrotated coordinates, so a rotated page's box is
    derotated first. Control characters and malformed chars (a lone
    surrogate, a code beyond Unicode, a non-finite origin) are dropped, runs
    of text on one line are joined with a space, and lines with a newline.
    """
    runs: list[tuple[int, float, str]] = []
    for span in spans:
        line: int | None = None
        chars: list[str] = []
        start = 0.0
        for char in span["chars"]:
            if not _usable_char(char) or char[2] not in page_box:
                continue
            y = round(char[2][1])
            if y != line:
                if chars:
                    runs.append((line, start, "".join(chars)))
                line, start, chars = y, char[2][0], []
            if char[0] >= 32:  # control characters (a recorded newline among them) are not text
                chars.append(chr(char[0]))
        if chars:
            runs.append((line, start, "".join(chars)))
    lines: dict[int, list[tuple[float, str]]] = {}
    for y, x, text in runs:
        lines.setdefault(y, []).append((x, text))
    return "\n".join(
        " ".join(text for _x, text in sorted(runs_of_line)) for _y, runs_of_line in sorted(lines.items())
    )[:MAX_PAGE_TEXT_CHARS]


def _check_images(page: pymupdf.Page, number: int, max_pixels: int) -> None:
    """Refuse images too large to decode safely, using their declared size only."""
    for image in page.get_images(full=True):
        width, height = image[2], image[3]
        if width * height > max_pixels:
            raise PipelineError(f"Page {number} holds an image too large to process ({width} x {height} pixels).")


def _ink_ratio(gray: Image.Image) -> float:
    width, height = gray.size
    margin_x, margin_y = width // 20, height // 20
    body = gray.crop((margin_x, margin_y, width - margin_x, height - margin_y)) if width > 20 and height > 20 else gray
    histogram = body.histogram()
    total = sum(histogram)
    return sum(histogram[:160]) / total if total else 0.0


def extract_pdf_pages(path: Path, page_start: int, page_end: int) -> bytes:
    """Return the 1-based inclusive page range of the source PDF as PDF bytes."""
    return extract_pdf_page_list(path, list(range(page_start, page_end + 1)))


def extract_pdf_page_list(path: Path, numbers: list[int]) -> bytes:
    """Return the listed 1-based pages of the source PDF, in order, as PDF bytes."""
    try:
        source = pymupdf.open(path)
    except Exception as exc:
        raise PipelineError("The PDF could not be reopened for page extraction.") from exc
    try:
        if not numbers or any(number < 1 or number > source.page_count for number in numbers):
            raise PipelineError("A page range is outside the source PDF.")
        part = pymupdf.open()
        try:
            run_start = previous = numbers[0]
            for number in [*numbers[1:], None]:
                if number is not None and number == previous + 1:
                    previous = number
                    continue
                part.insert_pdf(source, from_page=run_start - 1, to_page=previous - 1)
                if number is not None:
                    run_start = previous = number
            return part.tobytes(garbage=4, deflate=True)
        finally:
            part.close()
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("The PDF page range could not be extracted.") from exc
    finally:
        source.close()


def split_pdf(path: Path, ranges: list[tuple[int, int]], destination: Path) -> list[Path]:
    """Write 1-based inclusive page ranges into destination; return their paths."""
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    try:
        source = pymupdf.open(path)
    except Exception as exc:
        raise PipelineError("The PDF could not be reopened for splitting.") from exc
    try:
        for index, (page_start, page_end) in enumerate(ranges, start=1):
            if page_start < 1 or page_end < page_start or page_end > source.page_count:
                raise PipelineError("A document page range is outside the source PDF.")
            output = destination / f"document_{index:03d}_pages_{page_start:04d}-{page_end:04d}.pdf"
            temporary = _temporary_pdf_path(destination)
            part = pymupdf.open()
            try:
                part.insert_pdf(source, from_page=page_start - 1, to_page=page_end - 1)
                part.save(temporary, garbage=4, deflate=True)
                os.replace(temporary, output)
            finally:
                part.close()
                temporary.unlink(missing_ok=True)
            outputs.append(output)
        return outputs
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError("One or more split PDFs could not be written.") from exc
    finally:
        source.close()


def _temporary_pdf_path(directory: Path) -> Path:
    with tempfile.NamedTemporaryFile(prefix=".split-", suffix=".pdf", dir=directory, delete=False) as stream:
        return Path(stream.name)
