"""The light child process that renders one untrusted PDF (see render_child).

It is started as a fresh ``python -I`` interpreter with an empty environment,
so it inherits none of the server's state: no Streamlit, no ``doc_pipeline.app``,
no provider SDKs, and no API keys. It reads one JSON request from stdin, sets
its POSIX resource limits, and only then imports PyMuPDF and renders. The
reply goes back over the original stdout, which is first moved to a private
descriptor so MuPDF chatter can never land in the result channel, as a fixed
framing of primitives:

    8-byte little-endian header length, the JSON header bytes, then the raw
    JPEG bytes of the pages, back to back in page order.

The parent never unpickles anything (render_child rebuilds ``PageImage``
itself after validating this frame), so this module imports only the standard
library and ``errors`` at the top level; ``pdf`` (PyMuPDF, Pillow) is imported
inside ``main`` after the limits are set.
"""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

from .errors import PipelineError

# The CPU limit sits this far above the parent's wall-clock timeout, so a
# child spinning in C code is reaped by the kernel even if the parent dies
# first, while a slow start or a slow render is still left to the timeout.
CPU_LIMIT_GRACE_SECONDS = 30
# The request is one JSON object; a path can hardly be longer than this.
MAX_REQUEST_BYTES = 64 * 1024
# The reply framing, mirrored by render_child's reader. Lengths are unsigned
# 64-bit little-endian; the header is UTF-8 JSON.
HEADER_PREFIX_BYTES = 8
MAX_HEADER_BYTES = 32 * 1024 * 1024

REQUEST_KEYS = {"path", "dpi", "jpeg_quality", "max_pages", "memory_mb", "timeout_seconds"}
OK = "ok"
ERROR = "error"


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _read_request() -> dict[str, object]:
    """Read the whole request from stdin and check every field's type.

    Anything unexpected is a bug or an attack on the channel: exit without a
    reply, and the parent fails the file with "Rendering failed.".
    """
    data = sys.stdin.buffer.read()
    if not data or len(data) > MAX_REQUEST_BYTES:
        raise SystemExit(2)
    try:
        request = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise SystemExit(2) from None
    if not isinstance(request, dict) or set(request) != REQUEST_KEYS:
        raise SystemExit(2)
    if not isinstance(request["path"], str) or not request["path"]:
        raise SystemExit(2)
    if not all(_positive(request[key]) for key in ("dpi", "jpeg_quality", "max_pages", "memory_mb")):
        raise SystemExit(2)
    if not _positive(request["timeout_seconds"]) and not (
        isinstance(request["timeout_seconds"], float) and request["timeout_seconds"] > 0
    ):
        raise SystemExit(2)
    return request


def _cpu_seconds(timeout_seconds: float) -> int:
    return max(1, int(timeout_seconds) + CPU_LIMIT_GRACE_SECONDS)


def _apply_limits(memory_mb: int, cpu_seconds: int) -> None:
    """Cap this child's address space and CPU time before PyMuPDF is imported.

    RLIMIT_AS turns a decompression bomb into a MemoryError inside the child
    instead of an out-of-memory kill that would take the whole server down;
    RLIMIT_CPU is the kernel-side backstop for the wall-clock timeout. Both are
    set as soft *and* hard limits, so code inside the child cannot raise its
    own cap again, and RLIMIT_CORE is zeroed so a crashing child writes no
    core file. A limit the kernel refuses is not rendered around: the child
    exits without a reply (SystemExit), and the parent reports "Rendering
    failed." for the file instead of running it uncapped.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX never spawns a child
        return
    wanted = (
        (memory_mb * 1024 * 1024, "RLIMIT_AS"),
        (cpu_seconds, "RLIMIT_CPU"),
        (0, "RLIMIT_CORE"),
    )
    for limit, name in wanted:
        kind = getattr(resource, name, None)
        if kind is None:
            continue
        try:
            _, hard = resource.getrlimit(kind)
            ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(kind, (ceiling, ceiling))
        except (OSError, ValueError, OverflowError):
            raise SystemExit(2) from None


def _redirect_stdout() -> int:
    """Move the result channel to a private descriptor; fd 1 becomes stderr.

    MuPDF may print to stdout, and a stray byte would corrupt the framed
    reply, so the reply is written to the saved descriptor instead.
    """
    result_fd = os.dup(1)
    os.dup2(2, 1)
    return result_fd


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _reply(header: dict[str, object], blob: bytes = b"") -> bytes:
    encoded = json.dumps(header, ensure_ascii=False).encode("utf-8")
    if len(encoded) > MAX_HEADER_BYTES:
        raise PipelineError("The rendered pages were too large to report.")
    return struct.pack("<Q", len(encoded)) + encoded + blob


def main() -> None:
    """Render the one PDF described by the request and report the result.

    Only a PipelineError (a message already phrased for the app's user, and
    free of document content) crosses the channel as an error. Any other
    exception kills the child, which the parent reports as "Rendering
    failed." without seeing its details.
    """
    request = _read_request()
    _apply_limits(int(request["memory_mb"]), _cpu_seconds(float(request["timeout_seconds"])))
    result_fd = _redirect_stdout()
    from .pdf import render_pdf

    try:
        pages = render_pdf(
            Path(request["path"]),
            dpi=int(request["dpi"]),
            jpeg_quality=int(request["jpeg_quality"]),
            max_pages=int(request["max_pages"]),
        )
    except PipelineError as exc:
        _write_all(result_fd, _reply({"status": ERROR, "message": str(exc)}))
        return
    header = {
        "status": OK,
        "pages": [
            {
                "number": page.number,
                "dhash": page.dhash,
                "text": page.text,
                "ink": page.ink,
                "hidden_text": page.hidden_text,
                "jpeg_len": len(page.jpeg),
            }
            for page in pages
        ],
    }
    _write_all(result_fd, _reply(header, b"".join(page.jpeg for page in pages)))


if __name__ == "__main__":  # pragma: no cover
    main()
