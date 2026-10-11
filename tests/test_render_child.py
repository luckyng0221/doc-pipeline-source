from __future__ import annotations

import io
import json
import logging
import os
import pickle
import signal
import struct
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Guarded: `resource` does not exist off POSIX, and this module's in-process
# tests must still run (and be collected) on Windows.
resource = __import__("resource") if os.name == "posix" else None

import pymupdf
import pytest
from PIL import Image

from doc_pipeline import _render_worker, render_child
from doc_pipeline.app.session import RunControl, Upload, new_workspace, run_uploads
from doc_pipeline.app.settings import AppChoices, build_app_config, load_app_defaults
from doc_pipeline.errors import PipelineError, RunStopped
from doc_pipeline.pdf import MAX_PAGE_PIXELS, MAX_PAGE_TEXT_CHARS, PageImage, render_pdf

ROOT = Path(__file__).parents[1]
DEFAULTS = load_app_defaults(ROOT / "config.yaml")
# Tiny render settings; the deadline is a few seconds and the timeout tests
# override it.
SETTINGS = dict(dpi=72, jpeg_quality=75, max_pages=10, timeout_seconds=5, memory_mb=1024)
RECEIPT = "OFFICIAL RECEIPT\nReceipt No: R-{no}\nDate: 25/02/2025\nTotal 12.50"

needs_posix = pytest.mark.skipif(
    os.name != "posix", reason="the render child and its resource limits are POSIX-only"
)

needs_linux = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="the grandchild probe reads /proc/<pid>/stat"
)


def _write_pdf(root: Path, text: str = RECEIPT.format(no=1)) -> Path:
    source = root / "sample.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    document.save(source)
    document.close()
    return source


def _pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    document.new_page().insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def _record_spawns(monkeypatch, on_spawn=None) -> list[subprocess.Popen]:
    """The children the module starts while the test runs, replaced Popen and all."""
    spawned: list[subprocess.Popen] = []
    real = subprocess.Popen

    def recording(*args, **kwargs) -> subprocess.Popen:
        child = real(*args, **kwargs)
        spawned.append(child)
        if on_spawn is not None:
            on_spawn()
        return child

    monkeypatch.setattr(render_child.subprocess, "Popen", recording)
    return spawned


def _bytes_child(path: Path) -> subprocess.Popen:
    """A real child whose whole output is the bytes of the given file."""
    return subprocess.Popen(
        [sys.executable, "-I", "-c", "import sys; sys.stdout.buffer.write(open(sys.argv[1], 'rb').read())", str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


def _frame(pages: list[dict], blob: bytes = b"", status: str = "ok") -> bytes:
    header = json.dumps({"status": status, "pages": pages}).encode("utf-8")
    return struct.pack("<Q", len(header)) + header + blob


def _jpeg(width: int = 2, height: int = 2) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height)).save(buffer, "JPEG")
    return buffer.getvalue()


def _page(jpeg: bytes, **changes: object) -> dict:
    described = {
        "number": 1,
        "dhash": "0" * 16,
        "text": "",
        "ink": 0.5,
        "hidden_text": 0,
        "jpeg_len": len(jpeg),
    }
    described.update(changes)
    return described


# Module level, so a pickle can reference it: pickle records the function's
# name, and only unpickling calls it.
DETONATIONS: list[str] = []


def _detonate(mark: str) -> None:
    DETONATIONS.append(mark)


@needs_posix
def test_the_child_render_equals_the_in_process_render(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path)
    expected = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=10)

    pages = render_child.render_pdf_isolated(source, **SETTINGS)

    assert pages == expected


@needs_posix
def test_hidden_text_crosses_the_real_child_channel(tmp_path: Path) -> None:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), RECEIPT)
    page.insert_text((72, 400), "Hidden 99.99", render_mode=3)
    source = tmp_path / "hidden.pdf"
    document.save(source)
    document.close()

    pages = render_child.render_pdf_isolated(source, **SETTINGS)

    assert pages == render_pdf(source, dpi=72, jpeg_quality=75, max_pages=10)
    assert pages[0].hidden_text == 1
    assert "Hidden 99.99" not in pages[0].text
    assert "OFFICIAL RECEIPT" in pages[0].text


@needs_posix
def test_a_tiny_timeout_kills_the_child_and_the_next_file_carries_on(tmp_path: Path, monkeypatch) -> None:
    source = _write_pdf(tmp_path)
    spawned = _record_spawns(monkeypatch)

    with pytest.raises(PipelineError, match=r"^Rendering took too long\.$"):
        render_child.render_pdf_isolated(source, **{**SETTINGS, "timeout_seconds": 0.05})

    assert spawned and all(child.poll() is not None for child in spawned)
    assert render_child._CHILD_SLOTS._value == render_child.MAX_RENDER_CHILDREN
    # The server and the next file are unaffected by the killed child.
    assert [page.number for page in render_child.render_pdf_isolated(source, **SETTINGS)] == [1]
    assert render_child._CHILD_SLOTS._value == render_child.MAX_RENDER_CHILDREN


@needs_posix
def test_a_dead_silent_child_means_rendering_failed() -> None:
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import os; os._exit(9)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._await_child(child, timeout_seconds=5, max_pages=10)

    assert child.poll() is not None


def _render_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == render_child.logger.name]


@needs_posix
def test_a_child_that_dies_on_its_own_logs_one_warning_with_only_its_exit_code(caplog) -> None:
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import os; os._exit(9)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )

    with caplog.at_level(logging.WARNING, logger=render_child.logger.name):
        with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
            render_child._await_child(child, timeout_seconds=5, max_pages=10)

    assert _render_warnings(caplog) == ["A render child exited abnormally with exit code 9."]


@needs_posix
def test_a_successful_render_logs_no_child_warning(tmp_path: Path, caplog) -> None:
    source = _write_pdf(tmp_path)

    with caplog.at_level(logging.WARNING, logger=render_child.logger.name):
        assert [page.number for page in render_child.render_pdf_isolated(source, **SETTINGS)] == [1]

    assert _render_warnings(caplog) == []


@needs_posix
def test_a_kill_by_the_parent_logs_no_child_warning(caplog) -> None:
    def sleeper() -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-I", "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )

    def cancel() -> None:
        raise RunStopped("the run was cancelled")

    with caplog.at_level(logging.WARNING, logger=render_child.logger.name):
        timed_out = sleeper()
        with pytest.raises(PipelineError, match=r"^Rendering took too long\.$"):
            render_child._await_child(timed_out, timeout_seconds=0.3, max_pages=10)
        cancelled = sleeper()
        with pytest.raises(RunStopped):
            render_child._await_child(cancelled, timeout_seconds=5, max_pages=10, stop_check=cancel)

    assert timed_out.poll() is not None and cancelled.poll() is not None
    assert _render_warnings(caplog) == []


@needs_posix
def test_a_pickle_reply_is_never_unpickled(tmp_path: Path) -> None:
    DETONATIONS.clear()

    class Bomb:
        def __reduce__(self) -> tuple[object, tuple[object, ...]]:
            # Pickling runs this to describe the bomb; only unpickling runs
            # what it returns.
            return _detonate, ("boom",)

    payload = pickle.dumps({"pages": [Bomb()]})
    frame = tmp_path / "reply.bin"
    # As the whole reply, where the JSON header belongs ...
    frame.write_bytes(struct.pack("<Q", len(payload)) + payload)
    child = _bytes_child(frame)
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._await_child(child, timeout_seconds=5, max_pages=10)
    assert child.poll() is not None
    # ... and as the page bytes of a well-formed header.
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([_page(payload, jpeg_len=len(payload))], payload), max_pages=10)
    assert DETONATIONS == []


@needs_posix
def test_a_broken_pdf_keeps_its_user_facing_message(tmp_path: Path) -> None:
    source = tmp_path / "broken.pdf"
    source.write_bytes(b"not a pdf")

    with pytest.raises(PipelineError, match=r"^The file is not a readable PDF\.$"):
        render_child.render_pdf_isolated(source, **SETTINGS)


@needs_posix
def test_a_cancel_during_the_render_kills_the_child_and_propagates(tmp_path: Path, monkeypatch) -> None:
    source = _write_pdf(tmp_path)
    started = threading.Event()
    spawned = _record_spawns(monkeypatch, started.set)

    def stop_check() -> None:
        if started.is_set():
            raise RunStopped("the run was cancelled")

    with pytest.raises(RunStopped, match="the run was cancelled"):
        render_child.render_pdf_isolated(source, **SETTINGS, stop_check=stop_check)

    assert spawned and all(child.poll() is not None for child in spawned)
    assert render_child._CHILD_SLOTS._value == render_child.MAX_RENDER_CHILDREN


@needs_posix
def test_a_cancel_during_a_render_stops_the_run_without_archiving_a_failure(tmp_path: Path, monkeypatch) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, AppChoices(fields=[{"Include": True, "Field": "Vendor"}]), workspace)
    # The run is cancelled at the moment its first render child exists.
    cancelled = threading.Event()
    _record_spawns(monkeypatch, cancelled.set)

    result = run_uploads(
        config,
        keys,
        [Upload("only.pdf", _pdf_bytes(RECEIPT.format(no=9)))],
        workspace,
        control=RunControl(cancelled=cancelled),
    )

    assert result.stopped == "the run was cancelled"
    assert result.rows == []
    assert result.failures and "cancelled" in result.failures[0][1]
    assert not list((workspace / "errors").iterdir())


@needs_posix
def test_two_renders_run_at_once_and_the_waiting_third_honours_the_stop(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        renders = [pool.submit(render_child.render_pdf_isolated, source, **SETTINGS) for _ in range(2)]
        both = [future.result() for future in renders]
    assert both[0] == both[1]
    assert render_child._CHILD_SLOTS._value == render_child.MAX_RENDER_CHILDREN

    def stop() -> None:
        raise RunStopped("the run was cancelled")

    with render_child._render_slot(), render_child._render_slot():
        assert render_child._CHILD_SLOTS._value == 0
        with pytest.raises(RunStopped):
            with render_child._render_slot(stop):
                pass  # pragma: no cover - the stop fires while waiting
    assert render_child._CHILD_SLOTS._value == render_child.MAX_RENDER_CHILDREN


def test_without_child_process_support_it_renders_in_process(tmp_path: Path, monkeypatch) -> None:
    source = _write_pdf(tmp_path)
    monkeypatch.setattr(render_child, "child_processes_supported", lambda: False)
    calls: list[dict] = []

    def recording_render(path: Path, **kwargs: object) -> list[PageImage]:
        calls.append(dict(kwargs))
        return [PageImage(number=1, jpeg=b"jpeg", dhash="0" * 16)]

    monkeypatch.setattr(render_child, "render_pdf", recording_render)

    pages = render_child.render_pdf_isolated(source, **SETTINGS)

    assert calls == [{"dpi": 72, "jpeg_quality": 75, "max_pages": 10}]
    assert pages == [PageImage(number=1, jpeg=b"jpeg", dhash="0" * 16)]


@needs_posix
def test_the_child_gets_a_minimal_environment_and_imports_nothing_of_the_app(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DOC_PIPELINE_RENDER_PROBE_SENTINEL", "leak-me-not")
    request = render_child._request_bytes(_write_pdf(tmp_path), **SETTINGS)
    probe = (
        render_child._bootstrap_code()
        + "import json, os, sys\n"
        "proc_environ = open('/proc/self/environ', 'rb').read().decode('utf-8', 'replace')\n"
        "print(json.dumps({\n"
        "    'environ': sorted(os.environ),\n"
        "    'sentinel_in_environ': [k for k in os.environ if 'SENTINEL' in k],\n"
        "    'sentinel_in_proc': 'SENTINEL' in proc_environ,\n"
        "    'fds': sorted(os.listdir('/proc/self/fd')),\n"
        "    'modules': sorted(sys.modules),\n"
        "}), file=sys.stderr)\n"
    )

    finished = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        input=request,
        capture_output=True,
        env=render_child._child_env(),
        timeout=60,
    )

    assert finished.returncode == 0
    reported = json.loads(finished.stderr.decode().splitlines()[-1])
    # Nothing inherited: the parent's sentinel variable is nowhere in the
    # child's environment or in the kernel's record of it.
    assert reported["sentinel_in_environ"] == [] and reported["sentinel_in_proc"] is False
    assert set(reported["environ"]) <= {"LC_CTYPE"} | set(render_child._child_env())
    # The child ran the light worker and PyMuPDF, and none of the server.
    modules = set(reported["modules"])
    assert not {
        "streamlit",
        "doc_pipeline.app",
        "doc_pipeline.app.ui",
        "doc_pipeline.app.session",
        "pandas",
        "openai",
        "google.genai",
    } & modules
    assert {"doc_pipeline._render_worker", "pymupdf"} <= modules
    # Only its own channels and the interpreter's few; a leaked server
    # descriptor would sit at a much higher number. (3 is the saved result
    # channel, 4 libffi's mapping, 5 the listing's own descriptor.)
    assert set(map(int, reported["fds"])) <= {0, 1, 2, 3, 4, 5}
    assert {0, 1, 2, 3} <= set(map(int, reported["fds"]))


@needs_posix
def test_a_real_child_reports_the_configured_resource_limits(tmp_path: Path) -> None:
    request = render_child._request_bytes(_write_pdf(tmp_path), **SETTINGS)
    probe = (
        render_child._bootstrap_code()
        + "import json, os, resource, sys\n"
        "limits = {\n"
        "    name: resource.getrlimit(getattr(resource, name))\n"
        "    for name in ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_CORE')\n"
        "}\n"
        "print(json.dumps({'limits': limits, 'group_leader': os.getpgrp() == os.getpid()}), file=sys.stderr)\n"
    )

    finished = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        input=request,
        capture_output=True,
        env=render_child._child_env(),
        timeout=30,
        start_new_session=True,  # as _start_child starts the real child
    )

    assert finished.returncode == 0
    reported = json.loads(finished.stderr.decode().splitlines()[-1])
    for name, want in (
        ("RLIMIT_AS", SETTINGS["memory_mb"] * 1024 * 1024),
        ("RLIMIT_CPU", _render_worker._cpu_seconds(SETTINGS["timeout_seconds"])),
    ):
        soft, hard = reported["limits"][name]
        # Soft and hard agree, so the child cannot raise its own cap again,
        # and the kernel gave it no more than was configured.
        assert soft == hard and 0 < soft <= want
    assert reported["limits"]["RLIMIT_CORE"] == [0, 0]
    assert reported["group_leader"] is True


@needs_posix
def test_a_real_child_exits_silently_when_a_limit_is_refused(tmp_path: Path) -> None:
    request = render_child._request_bytes(_write_pdf(tmp_path), **SETTINGS)
    # The patch lands between the bootstrap's import and its main() call, so
    # it is the child's own setrlimit that fails.
    patch = (
        "import resource\n"
        "def _refuse(*_args):\n"
        "    raise OSError('the kernel refused the limit')\n"
        "resource.setrlimit = _refuse\n"
    )
    probe = render_child._bootstrap_code().replace(
        "from doc_pipeline._render_worker import main\n",
        patch + "from doc_pipeline._render_worker import main\n",
    )

    finished = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        input=request,
        capture_output=True,
        env=render_child._child_env(),
        timeout=30,
    )

    # No reply: the parent has nothing to show but "Rendering failed.".
    assert finished.returncode == 2
    assert finished.stdout == b""


@needs_posix
def test_the_child_runs_in_its_own_process_group() -> None:
    child = render_child._start_child(b"{}")

    try:
        # A session (and so a process group) of its own, away from the server's.
        assert os.getpgid(child.pid) == child.pid
    finally:
        render_child._stop_child(child)


def _pid_state(pid: int) -> str | None:
    """A /proc/<pid>/stat state letter, or None once the pid is gone."""
    try:
        tail = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").rsplit(")", 1)[1]
    except OSError:
        return None
    return tail.split()[0]


def _gone_or_zombie(pid: int, seconds: float = 5.0) -> bool:
    """Whether the pid has died; a zombie counts (dead, even if not reaped)."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        state = _pid_state(pid)
        if state is None or state == "Z":
            return True
        time.sleep(0.05)
    return False


@needs_linux
def test_stopping_the_child_kills_its_whole_process_group() -> None:
    spawn = (
        "import subprocess, sys, time\n"
        "grandchild = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(60)'])\n"
        "print(grandchild.pid, flush=True)\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", spawn],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    grand_pid = int(child.stdout.readline())
    try:
        assert os.getpgid(grand_pid) == os.getpgid(child.pid) == child.pid

        render_child._stop_child(child)

        assert child.poll() is not None
        assert _gone_or_zombie(grand_pid), "the grandchild outlived the child's stop"
    finally:
        try:  # never leave the sleeper behind, even if an assert fired
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass
        child.wait()


@needs_linux
def test_the_group_is_killed_even_when_the_child_dies_within_the_grace() -> None:
    # The grandchild ignores SIGTERM and reports its pid only once that is in
    # place, so the SIGTERM within the grace kills the main child but never
    # it: the group must still get its SIGKILL afterwards.
    spawn = (
        "import subprocess, sys, time\n"
        "stubborn = (\n"
        "    'import os, signal, time; '\n"
        "    'signal.signal(signal.SIGTERM, signal.SIG_IGN); '\n"
        "    'print(os.getpid(), flush=True); time.sleep(60)'\n"
        ")\n"
        "subprocess.Popen([sys.executable, '-I', '-c', stubborn])\n"
        "time.sleep(60)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", spawn],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    grand_pid = int(child.stdout.readline())
    try:
        assert os.getpgid(child.pid) != os.getpgrp()  # the server's own group stays out of this
        assert os.getpgid(grand_pid) == os.getpgid(child.pid) == child.pid

        render_child._stop_child(child)

        assert child.poll() is not None
        assert _gone_or_zombie(grand_pid), "the grandchild outlived the child's stop"
    finally:
        try:  # never leave the sleeper behind, even if an assert fired
            os.killpg(child.pid, signal.SIGKILL)
        except OSError:
            pass
        child.wait()


@needs_posix
def test_the_child_sets_soft_and_hard_limits_and_no_core_dump(monkeypatch) -> None:
    # Recorded, never applied: lowering this interpreter's own limits (a real
    # setrlimit here) kills the test run itself. That a real child actually
    # runs under the limits is asserted by the probe above.
    hard = {
        resource.RLIMIT_AS: resource.RLIM_INFINITY,
        resource.RLIMIT_CPU: 600,
        resource.RLIMIT_CORE: resource.RLIM_INFINITY,
    }
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (resource.RLIM_INFINITY, hard[kind]))
    monkeypatch.setattr(resource, "setrlimit", lambda kind, ceiling: calls.append((kind, ceiling)))

    _render_worker._apply_limits(memory_mb=64, cpu_seconds=7)

    # Soft and hard get the same ceiling (the unbounded hard limit passes the
    # value through; a finite one caps it), and core dumps are off entirely.
    assert calls == [
        (resource.RLIMIT_AS, (64 * 1024 * 1024, 64 * 1024 * 1024)),
        (resource.RLIMIT_CPU, (7, 7)),
        (resource.RLIMIT_CORE, (0, 0)),
    ]


@needs_posix
def test_a_refused_limit_stops_the_child_instead_of_rendering_uncapped(monkeypatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    def refusing(kind: int, ceiling: tuple[int, int]) -> None:
        raise OSError("the kernel refused the limit")

    monkeypatch.setattr(resource, "setrlimit", refusing)

    with pytest.raises(SystemExit) as stopped:
        _render_worker._apply_limits(memory_mb=64, cpu_seconds=7)

    assert stopped.value.code == 2


def test_the_cpu_limit_sits_above_the_wall_clock_timeout() -> None:
    assert _render_worker._cpu_seconds(120.0) == 120 + _render_worker.CPU_LIMIT_GRACE_SECONDS
    assert _render_worker._cpu_seconds(0.05) == _render_worker.CPU_LIMIT_GRACE_SECONDS


@needs_posix
def test_a_memory_limit_below_the_interpreter_kills_the_child(tmp_path: Path) -> None:
    source = _write_pdf(tmp_path)

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child.render_pdf_isolated(source, **{**SETTINGS, "memory_mb": 1})


@needs_posix
def test_a_dead_child_fails_only_its_file_and_the_next_one_carries_on(tmp_path: Path, monkeypatch) -> None:
    workspace = new_workspace(tmp_path)
    config, keys = build_app_config(DEFAULTS, AppChoices(fields=[{"Include": True, "Field": "Vendor"}]), workspace)
    real = subprocess.Popen
    started = {"count": 0}

    def first_child_dies(*args: object, **kwargs: object) -> subprocess.Popen:
        started["count"] += 1
        if started["count"] == 1:
            return real(  # type: ignore[call-arg]
                [sys.executable, "-I", "-c", "import os; os._exit(9)"], **kwargs  # type: ignore[arg-type]
            )
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(render_child.subprocess, "Popen", first_child_dies)

    result = run_uploads(
        config,
        keys,
        [
            Upload("dies.pdf", _pdf_bytes(RECEIPT.format(no=2))),
            Upload("works.pdf", _pdf_bytes(RECEIPT.format(no=3))),
        ],
        workspace,
    )

    assert result.failures == [("dies.pdf", "Rendering failed.")]
    assert [row["Source PDF"] for row in result.rows] == ["works.pdf"]
    assert started["count"] == 2


def test_a_well_formed_reply_is_rebuilt_as_pages() -> None:
    jpeg = _jpeg()
    text = "Invoice No: A-1"

    status, pages = render_child._parse_reply(_frame([_page(jpeg, text=text, hidden_text=3)], jpeg), max_pages=10)

    assert status == "ok"
    assert pages == [PageImage(number=1, jpeg=jpeg, dhash="0" * 16, text=text, ink=0.5, hidden_text=3)]


def test_a_well_formed_error_keeps_its_message() -> None:
    header = json.dumps({"status": "error", "message": "The file is not a readable PDF."}).encode("utf-8")

    status, message = render_child._parse_reply(struct.pack("<Q", len(header)) + header, max_pages=10)

    assert (status, message) == ("error", "The file is not a readable PDF.")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00" * 4,
        # A header length that is not on the wire.
        struct.pack("<Q", render_child.MAX_HEADER_BYTES + 1) + b"{}",
        struct.pack("<Q", 100) + b"{}",
        struct.pack("<Q", 2) + b"x",
        b"\x00" * 8 + b"not json",
        struct.pack("<Q", 4) + b"null",
        struct.pack("<Q", 2) + b"[]",
    ],
)
def test_malformed_frames_mean_rendering_failed(data: bytes) -> None:
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(data, max_pages=10)


def test_a_header_over_the_page_derived_cap_is_refused_unparsed() -> None:
    # A run asking for one page reads no header sized for many more, even when
    # it arrives whole and would parse: the length alone is the refusal.
    header = json.dumps({"status": "error", "message": "The file is not a readable PDF."}).encode("utf-8")
    over = render_child._header_cap(1) + 1
    padded = header.ljust(over, b" ")
    frame = struct.pack("<Q", over) + padded

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(frame, max_pages=1)

    status, message = render_child._parse_reply(frame, max_pages=100)  # a run that allows it reads it
    assert (status, message) == ("error", "The file is not a readable PDF.")


def test_a_deeply_nested_header_means_rendering_failed() -> None:
    # Within the cap but past json's recursion: any parse error at all fails
    # the file with the same content-free message.
    header = b"[" * 200_000 + b"]" * 200_000
    assert len(header) <= render_child._header_cap(10)

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(struct.pack("<Q", len(header)) + header, max_pages=10)


@pytest.mark.parametrize(
    "page",
    [
        _page(_jpeg(), number=2),
        _page(_jpeg(), number=1.0),
        _page(_jpeg(), number=True),
        _page(_jpeg(), dhash="nothex"),
        _page(_jpeg(), text="x" * (MAX_PAGE_TEXT_CHARS + 1)),
        _page(_jpeg(), ink=1.5),
        _page(_jpeg(), ink="dark"),
        {key: value for key, value in _page(_jpeg()).items() if key != "hidden_text"},
        _page(_jpeg(), hidden_text=None),
        _page(_jpeg(), hidden_text=True),
        _page(_jpeg(), hidden_text="3"),
        _page(_jpeg(), hidden_text=-1),
        _page(_jpeg(), hidden_text=1.5),
        _page(_jpeg(), hidden_text=render_child.MAX_HIDDEN_TEXT_SPANS + 1),
        _page(_jpeg(), jpeg_len=0),
        _page(_jpeg(), jpeg_len=-1),
        _page(_jpeg(), jpeg_len=render_child.MAX_REPLY_PAGE_BYTES + 1),
        _page(_jpeg(), jpeg_len=True),
        _page(b"not a jpeg"),
        # A page whose declared pixels are over the render cap.
        _page(b"\xff\xd8\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", 65535, 65535) + bytes(16)),
        # A page claiming no width, or no height (and so no area at all).
        _page(b"\xff\xd8\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", 0, 65535) + bytes(16)),
        _page(b"\xff\xd8\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", 65535, 0) + bytes(16)),
    ],
)
def test_tampered_page_descriptions_mean_rendering_failed(page: dict) -> None:
    jpeg = _jpeg()
    blob = jpeg if page["jpeg_len"] == len(jpeg) else b""

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([page], blob), max_pages=10)


@pytest.mark.parametrize("dimensions", [(0, 100), (100, 0), (0, 0)])
def test_a_zero_sized_sof_is_refused_even_with_its_bytes_on_the_wire(dimensions: tuple[int, int]) -> None:
    # (width - 1) * (height - 1) would pass the pixel cap for a zero side, so
    # a zero width or height is refused on its own.
    forged = (
        b"\xff\xd8\xff\xc0"
        + struct.pack(">H", 17)
        + bytes([8])
        + struct.pack(">HH", dimensions[1], dimensions[0])
        + bytes(16)
    )

    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([_page(forged, jpeg_len=len(forged))], forged), max_pages=10)


def test_extra_pages_and_leftover_bytes_mean_rendering_failed() -> None:
    jpeg = _jpeg()
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([_page(jpeg)] * 3, jpeg), max_pages=2)
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([_page(jpeg)], jpeg + b"trailing"), max_pages=10)
    header = json.dumps({"status": "error", "message": "x" * 1001}).encode("utf-8")
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(struct.pack("<Q", len(header)) + header, max_pages=10)


def test_a_reply_cut_inside_the_last_page_means_rendering_failed() -> None:
    # A slice short of the declared jpeg_len used to clamp silently: the read
    # stopped at the end of the blob and the offset grew by what it read, so
    # every check passed on a reply cut mid-JPEG.
    jpeg = _jpeg()
    with pytest.raises(PipelineError, match=r"^Rendering failed\.$"):
        render_child._parse_reply(_frame([_page(jpeg)], jpeg[:-10]), max_pages=10)


def test_a_page_over_the_pixel_cap_is_refused_even_from_a_real_render(tmp_path: Path) -> None:
    # The dimensions of a page this pipeline actually rendered are accepted.
    source = _write_pdf(tmp_path)
    real = render_pdf(source, dpi=72, jpeg_quality=75, max_pages=10)[0]
    status, pages = render_child._parse_reply(
        _frame([_page(real.jpeg, text=real.text, dhash=real.dhash, ink=real.ink)], real.jpeg), max_pages=10
    )
    assert status == "ok" and pages[0].jpeg == real.jpeg
    assert MAX_PAGE_PIXELS > 0  # the cap the dimension check enforces


def test_a_page_rounded_just_over_the_pixel_cap_is_accepted() -> None:
    # pdf.py scales a page to exactly MAX_PAGE_PIXELS and MuPDF rounds the
    # pixmap outward, so a legitimate large page - a 2550 x 3300 pt page at
    # 150 dpi, say - arrives as 4396 x 5689, a few thousand pixels over.
    width, height = 4396, 5689
    assert width * height > MAX_PAGE_PIXELS  # what the check must tolerate
    jpeg = b"\xff\xd8\xff\xc0" + struct.pack(">H", 17) + bytes([8]) + struct.pack(">HH", height, width) + bytes(16)

    status, pages = render_child._parse_reply(_frame([_page(jpeg)], jpeg), max_pages=10)

    assert status == "ok"
    assert pages == [PageImage(number=1, jpeg=jpeg, dhash="0" * 16, text="", ink=0.5)]
