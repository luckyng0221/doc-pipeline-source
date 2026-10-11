"""Render untrusted PDFs in a disposable, resource-limited child process.

Every PDF file gets its own short-lived child: a fresh ``python -I``
interpreter (a spawn of a whole new process, the way ``multiprocessing``'s
spawn context would create one, minus everything it would drag along) with a
minimal environment, a session of its own, and a dedicated, light entry module
(``doc_pipeline._render_worker``). The child holds no API keys and no session
state, and the server process is non-dumpable on Linux when run as the
Streamlit app (``doc_pipeline.app.hardening``), so a hostile PDF that finds
a hole in the PDF library cannot read the server's memory through it - but
that is where the isolation ends: the child runs as the same OS user, with
that user's filesystem and network access. It can write the server's files
(this application's code, the venv's provider SDKs, other sessions'
workspaces) and read ``.streamlit/secrets.toml`` there and then. Its writes
to server code or config are not held back until a restart: the child can
force a live reload through the always-polled ``.streamlit/config.toml`` and
``.streamlit/secrets.toml``, so its changes can take effect in the running
server. Turning the file watcher off and importing the SDKs' top-level
modules (those that imported successfully) at start-up only narrow this (a
rerun does not re-import modules already in ``sys.modules``). The rlimits,
the timeout and its own process group bound it, they do not sandbox it (no
seccomp, no Landlock; a dedicated child user or a Landlock policy would
close this and is not done).
Rendering only: a page that renders forever is killed at the configured
timeout instead of stalling the Streamlit server for everyone (security
review M1/H2, plan §4.4).

The parent sends the path and the numeric render settings on stdin, then
drains a fixed framing of primitive data (a JSON header plus length-prefixed
raw JPEG bytes) from the child's stdout while polling the deadline and the
stop check. Nothing the child sends is ever unpickled: the parent rebuilds
``PageImage`` itself after checking the frame's shape and size, so a reply can
only ever be data. Text extraction and splitting stay in-process on purpose
(plan §9 row 4); non-POSIX platforms fall back to in-process rendering, where
the ``resource`` limits do not exist.
"""

from __future__ import annotations

import json
import logging
import os
import re
import selectors
import signal
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from ._render_worker import (
    ERROR,
    HEADER_PREFIX_BYTES,
    MAX_HEADER_BYTES,
    MAX_REQUEST_BYTES,
    OK,
)
from .errors import PipelineError
from .pdf import MAX_HIDDEN_TEXT_SPANS, MAX_PAGE_PIXELS, MAX_PAGE_TEXT_CHARS, PageImage, render_pdf

logger = logging.getLogger(__name__)

MAX_RENDER_CHILDREN = 2
# Server-wide, so two runs can never queue on each other for rendering
# (plan §9 row 3). Acquired per file, with timed polling so a run that is
# waiting for a slot still notices a Cancel.
_CHILD_SLOTS = threading.BoundedSemaphore(MAX_RENDER_CHILDREN)

# How often the wait loop wakes to check the deadline and the stop check.
POLL_SECONDS = 0.05
# SIGTERM to SIGKILL grace when a child ignores terminate.
TERMINATE_GRACE_SECONDS = 5.0
# How long a child that has closed its output is given to exit on its own, so
# its exit status can be told from a stop the parent caused.
EXIT_WAIT_SECONDS = 1.0
# Reply caps. A page renders to at most MAX_PAGE_PIXELS (25 MP, pdf.py) and is
# re-encoded as JPEG, and even a noisy 25-megapixel page stays a few MB at the
# configured quality: 16 MiB per page is several times that, and a JPEG
# claiming more is not a page this pipeline rendered. The header carries each
# page's text, cut at MAX_PAGE_TEXT_CHARS (20,000) characters per page, plus
# its hidden-text count, a capped integer (MAX_HIDDEN_TEXT_SPANS, 1000) that
# adds next to nothing beside the text; even with every character escaped to
# six bytes, plus the page's other fields rounded up to 4 KiB, that is ~121 KiB
# (124,096 bytes) per page - so a header is refused above
# _header_cap(max_pages), which stays well under MAX_HEADER_BYTES
# (32 MiB) until the page count itself approaches it. Together the caps bound
# what the parent will ever read into memory to what today's in-process render
# would have held anyway, whatever the page count.
MAX_REPLY_PAGE_BYTES = 16 * 1024 * 1024
MAX_REPLY_ERROR_CHARS = 1000
# The header's fixed part (status, or the message) plus the per-page budget
# _header_cap allows: every page of text, fully escaped, and a margin for the
# rest of its description.
_HEADER_ENVELOPE_BYTES = 8 * 1024
MAX_REPLY_PAGE_DESC_BYTES = MAX_PAGE_TEXT_CHARS * 6 + 4096
# imagehash.dhash renders to 16 lowercase hex characters (an 8x8 bit hash).
DHASH = re.compile(r"[0-9a-f]{16}")

RENDER_FAILED = "Rendering failed."
RENDER_TOOK_TOO_LONG = "Rendering took too long."


def _header_cap(max_pages: int) -> int:
    """The largest header length a reply for max_pages pages may declare.

    Derived per run from max_pages, so a child claiming a thousand pages
    cannot inflate the one number that decides how much the parent reads;
    MAX_HEADER_BYTES still bounds it when the page count is huge.
    """
    return min(MAX_HEADER_BYTES, _HEADER_ENVELOPE_BYTES + MAX_REPLY_PAGE_DESC_BYTES * max_pages)


def child_processes_supported() -> bool:
    """Whether the child process and its ``resource`` limits are available."""
    return os.name == "posix"


def render_pdf_isolated(
    path: Path,
    *,
    dpi: int,
    jpeg_quality: int,
    max_pages: int,
    timeout_seconds: float,
    memory_mb: int,
    stop_check: Callable[[], None] | None = None,
) -> list[PageImage]:
    """Render one PDF in a child process that is killed at the timeout.

    Only the file's failure is affected: a child that hangs, dies, or sends
    anything malformed fails its file with a content-free reason, and a Cancel
    or a run time limit raises RunStopped exactly as the rest of the pipeline
    does. On non-POSIX platforms the render runs in-process (no resource
    limits there) via the plain render_pdf.
    """
    if not child_processes_supported():
        return render_pdf(path, dpi=dpi, jpeg_quality=jpeg_quality, max_pages=max_pages)
    with _render_slot(stop_check):
        request = _request_bytes(
            path,
            dpi=dpi,
            jpeg_quality=jpeg_quality,
            max_pages=max_pages,
            timeout_seconds=timeout_seconds,
            memory_mb=memory_mb,
        )
        child = _start_child(request)
        try:
            status, payload = _await_child(
                child,
                timeout_seconds=timeout_seconds,
                max_pages=max_pages,
                stop_check=stop_check,
            )
        finally:
            _stop_child(child)
        if status == ERROR:
            raise PipelineError(payload)
        return payload


@contextmanager
def _render_slot(stop_check: Callable[[], None] | None = None):
    """Hold one of the server-wide render-child slots, honouring the stop."""
    while not _CHILD_SLOTS.acquire(timeout=POLL_SECONDS):
        if stop_check is not None:
            stop_check()
    try:
        yield
    finally:
        _CHILD_SLOTS.release()


def _child_env() -> dict[str, str]:
    """The child's whole environment: three chosen variables, nothing inherited.

    The interpreter is started by absolute path (sys.executable), so nothing
    of the server's environment - keys included, were they ever there - is
    passed along. The three thread caps are set deliberately: numpy (via
    imagehash, for the page hashes) otherwise reserves a BLAS arena of
    address space per CPU core, which would push the child over
    render_child_memory_mb before it rendered anything. The child hashes one
    small page image at a time and never wants those threads.
    """
    return {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def _bootstrap_code() -> str:
    """The ``-c`` code that turns a bare ``python -I`` into the render child.

    Isolated mode keeps the current directory off sys.path, so the folder
    above the package (this file's .../src) is put back first: the child then
    imports exactly the tree this module came from, and runs its light worker
    module - never the server's __main__.
    """
    package_root = str(Path(__file__).resolve().parent.parent)
    return (
        "import sys\n"
        f"sys.path.insert(0, {package_root!r})\n"
        "from doc_pipeline._render_worker import main\n"
        "main()\n"
    )


def _request_bytes(
    path: Path,
    *,
    dpi: int,
    jpeg_quality: int,
    max_pages: int,
    timeout_seconds: float,
    memory_mb: int,
) -> bytes:
    """The whole conversation with the child: JSON on stdin, then EOF.

    Only the PDF path and the numeric render settings cross - no API keys, no
    session state, and no file names beyond the path itself.
    """
    request = {
        "path": str(Path(path).resolve()),
        "dpi": int(dpi),
        "jpeg_quality": int(jpeg_quality),
        "max_pages": int(max_pages),
        "memory_mb": int(memory_mb),
        "timeout_seconds": float(timeout_seconds),
    }
    payload = json.dumps(request).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise PipelineError(RENDER_FAILED)
    return payload


def _start_child(request: bytes) -> subprocess.Popen:
    """Start one child: a fresh, isolated interpreter reading the request.

    The child leads its own session (and so its own process group), keeping it
    apart from the server's group: a timeout or a cancel can then reap the
    child together with anything it managed to spawn.
    """
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", _bootstrap_code()],
        env=_child_env(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        # Any MuPDF chatter is dropped; the reply alone is the result channel.
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
    )
    try:
        child.stdin.write(request)
        child.stdin.close()
    except OSError:
        # Died on start-up; the wait loop reports it.
        pass
    return child


def _await_child(
    child: subprocess.Popen,
    *,
    timeout_seconds: float,
    max_pages: int,
    stop_check: Callable[[], None] | None = None,
) -> tuple[str, object]:
    """Wait for the child's reply, killing it at the deadline or on a stop.

    Output is drained as it arrives (a selector here; no reader thread), so a
    large reply cannot deadlock on a full pipe and the total byte cap is
    enforced while reading. Everything the child sends is parsed as the fixed
    frame of primitives; an unparsable, truncated, or oversized frame means a
    child that died or was tampered with, and fails the file.
    """
    deadline = time.monotonic() + timeout_seconds
    limit = _header_cap(max_pages) + max_pages * MAX_REPLY_PAGE_BYTES
    data = bytearray()
    finished = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while not finished:
                if stop_check is not None:
                    stop_check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PipelineError(RENDER_TOOK_TOO_LONG)
                for _key, _events in selector.select(timeout=min(POLL_SECONDS, remaining)):
                    chunk = os.read(child.stdout.fileno(), 1024 * 1024)
                    if not chunk:
                        finished = True
                        break
                    data.extend(chunk)
                    if len(data) > limit:
                        raise PipelineError(RENDER_FAILED)
    except BaseException:
        _stop_child(child)
        raise
    _report_abnormal_exit(child, deadline)
    return _parse_reply(bytes(data), max_pages=max_pages)


def _report_abnormal_exit(child: subprocess.Popen, deadline: float) -> None:
    """Log one content-free WARNING when the child ended itself with an error.

    Called only after the child closed its output with nothing from the parent
    (no timeout, cancel or over-limit kill), so a non-zero status is the
    child's own: a refused limit, a crash or a kill by the kernel. Only the
    constant text and the integer exit code are logged - never a path, a file
    name or anything the child sent.
    """
    try:
        code = child.wait(timeout=max(0.0, min(EXIT_WAIT_SECONDS, deadline - time.monotonic())))
    except subprocess.TimeoutExpired:
        return  # still running: _stop_child ends it, which is the parent's doing
    if code != 0:
        logger.warning("A render child exited abnormally with exit code %d.", code)


def _parse_reply(data: bytes, *, max_pages: int) -> tuple[str, object]:
    """Validate the framed reply and rebuild the pages it claims to hold.

    The child is untrusted once a PDF has been parsed in it, so every field
    is type-checked here and the JPEG bytes are only ever copied around -
    nothing is deserialized.
    """
    if len(data) < HEADER_PREFIX_BYTES:
        raise PipelineError(RENDER_FAILED)
    (header_length,) = struct.unpack("<Q", data[:HEADER_PREFIX_BYTES])
    if not 0 < header_length <= _header_cap(max_pages):
        raise PipelineError(RENDER_FAILED)
    end = HEADER_PREFIX_BYTES + header_length
    if end > len(data):
        raise PipelineError(RENDER_FAILED)
    try:
        header = json.loads(data[HEADER_PREFIX_BYTES:end].decode("utf-8"))
    except Exception:  # bad UTF-8, bad JSON, or a nesting so deep it recursed
        raise PipelineError(RENDER_FAILED) from None
    if not isinstance(header, dict):
        raise PipelineError(RENDER_FAILED)
    status = header.get("status")
    if status == ERROR:
        message = header.get("message")
        if not isinstance(message, str) or not message or len(message) > MAX_REPLY_ERROR_CHARS:
            raise PipelineError(RENDER_FAILED)
        return ERROR, message
    if status != OK or not isinstance(header.get("pages"), list):
        raise PipelineError(RENDER_FAILED)
    described = header["pages"]
    if len(described) > max_pages:
        raise PipelineError(RENDER_FAILED)
    blob = memoryview(data)[end:]
    offset = 0
    pages: list[PageImage] = []
    for index, described_page in enumerate(described, start=1):
        jpeg, page = _page_from_reply(described_page, index, blob, offset)
        offset += len(jpeg)
        pages.append(page)
    if offset != len(blob):
        raise PipelineError(RENDER_FAILED)
    return OK, pages


def _page_from_reply(
    described: object,
    index: int,
    blob: memoryview,
    offset: int,
) -> tuple[bytes, PageImage]:
    """Rebuild one PageImage from a validated description of itself."""
    if not isinstance(described, dict):
        raise PipelineError(RENDER_FAILED)
    number, dhash, text, ink, hidden_text, jpeg_len = (
        described.get("number"),
        described.get("dhash"),
        described.get("text"),
        described.get("ink"),
        described.get("hidden_text"),
        described.get("jpeg_len"),
    )
    if (
        number != index
        or not isinstance(number, int)
        or isinstance(number, bool)
        or not isinstance(dhash, str)
        or not DHASH.fullmatch(dhash)
        or not isinstance(text, str)
        or len(text) > MAX_PAGE_TEXT_CHARS
        or not isinstance(ink, (int, float))
        or isinstance(ink, bool)
        or not 0.0 <= ink <= 1.0
        or not isinstance(hidden_text, int)
        or isinstance(hidden_text, bool)
        or not 0 <= hidden_text <= MAX_HIDDEN_TEXT_SPANS
        or not isinstance(jpeg_len, int)
        or isinstance(jpeg_len, bool)
        or not 0 < jpeg_len <= MAX_REPLY_PAGE_BYTES
    ):
        raise PipelineError(RENDER_FAILED)
    if offset + jpeg_len > len(blob):
        raise PipelineError(RENDER_FAILED)
    jpeg = bytes(blob[offset : offset + jpeg_len])
    dimensions = _jpeg_dimensions(jpeg)
    # MuPDF rounds the pixmap outward, so a page scaled to exactly
    # MAX_PAGE_PIXELS (pdf.py) arrives up to one pixel wider and taller; the
    # unrounded page is what has to fit the cap. A zero width or height is not
    # a page at all - and would slip past the product below as a negative.
    if (
        dimensions is None
        or dimensions[0] == 0
        or dimensions[1] == 0
        or (dimensions[0] - 1) * (dimensions[1] - 1) > MAX_PAGE_PIXELS
    ):
        raise PipelineError(RENDER_FAILED)
    return jpeg, PageImage(number=number, jpeg=jpeg, dhash=dhash, text=text, ink=float(ink), hidden_text=hidden_text)


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    """The width and height a JPEG's SOF marker declares, or None."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF or data[index + 1] in (0x00, 0xFF):
            return None
        marker = data[index + 1]
        if marker == 0xD8 or marker == 0x01 or 0xD0 <= marker <= 0xD7:
            index += 2
            continue
        (length,) = struct.unpack(">H", data[index + 2 : index + 4])
        if length < 2:
            return None
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if index + 9 > len(data):
                return None
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        if marker == 0xD9:
            return None
        index += 2 + length
    return None


def _signal_group(child: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, or the child alone as a fallback.

    A child from _start_child leads its own group, so anything it spawned is
    signalled with it. The fallback covers a child that is not a group leader
    - the stand-in children tests hand to _await_child - or one whose group is
    already gone, where killpg raises OSError.
    """
    try:
        os.killpg(child.pid, sig)
    except OSError:
        child.send_signal(sig)


def _stop_child(child: subprocess.Popen) -> None:
    """Reap the child and its group: terminate, a short grace, then kill."""
    try:
        if child.poll() is None:
            _signal_group(child, signal.SIGTERM)
            try:
                child.wait(timeout=TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            # The group is killed either way: members other than the child may
            # remain after the child itself died within the grace, and
            # _signal_group swallows the kill on a group that is already gone.
            _signal_group(child, signal.SIGKILL)
            child.wait()
    finally:
        for stream in (child.stdin, child.stdout):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
