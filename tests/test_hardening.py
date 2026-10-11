"""Tests for app.hardening: the server process is made non-dumpable.

Every probe that really calls prctl runs in a spawned ``python -I``
subprocess: applying it to this pytest process would change the dumpable flag
the run's own children inherit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import doc_pipeline
import pytest

from doc_pipeline.app import hardening

SRC = str(Path(doc_pipeline.__file__).resolve().parent.parent)

needs_linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl is a Linux facility")

# Reports the dumpable flag before and after set_non_dumpable, and its result.
_DUMPABLE_PROBE = (
    "import ctypes, sys\n"
    f"sys.path.insert(0, {SRC!r})\n"
    "from doc_pipeline.app.hardening import PR_GET_DUMPABLE, set_non_dumpable\n"
    "prctl = ctypes.CDLL(None, use_errno=True).prctl\n"
    "prctl.argtypes = (ctypes.c_int,) + (ctypes.c_ulong,) * 4\n"
    "prctl.restype = ctypes.c_int\n"
    "before = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0)\n"
    "changed = set_non_dumpable()\n"
    "after = prctl(PR_GET_DUMPABLE, 0, 0, 0, 0)\n"
    "print(before, changed, after)\n"
)

# A child that hardens itself and sleeps, then a same-uid sibling that tries
# to read its /proc/<pid>/environ.
_TARGET = (
    "import sys, time\n"
    f"sys.path.insert(0, {SRC!r})\n"
    "from doc_pipeline.app.hardening import set_non_dumpable\n"
    "print(set_non_dumpable(), flush=True)\n"
    "time.sleep(30)\n"
)
_ENVIRON_PROBE = (
    "import subprocess, sys\n"
    f"target = subprocess.Popen([sys.executable, '-I', '-c', {_TARGET!r}], stdout=subprocess.PIPE)\n"
    "ready = target.stdout.readline().strip()\n"
    "try:\n"
    "    open(f'/proc/{target.pid}/environ', 'rb').read()\n"
    "except OSError:\n"
    "    outcome = 'refused'\n"
    "else:\n"
    "    outcome = 'read'\n"
    "target.kill()\n"
    "target.wait()\n"
    "print(ready.decode(), outcome)\n"
)


@needs_linux
def test_a_real_process_flips_to_non_dumpable_and_reports_true() -> None:
    finished = subprocess.run([sys.executable, "-I", "-c", _DUMPABLE_PROBE], capture_output=True, timeout=60)

    assert finished.returncode == 0, finished.stderr
    before, changed, after = finished.stdout.decode().split()
    assert (before, changed, after) == ("1", "True", "0")


@needs_linux
@pytest.mark.skipif(os.geteuid() == 0, reason="root reads another process's /proc files regardless")
def test_a_same_uid_process_cannot_read_a_non_dumpable_process_environ() -> None:
    finished = subprocess.run([sys.executable, "-I", "-c", _ENVIRON_PROBE], capture_output=True, timeout=60)

    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.decode().split() == ["True", "refused"]


def test_another_platform_is_left_alone(monkeypatch) -> None:
    monkeypatch.setattr(hardening.sys, "platform", "win32")

    assert hardening.set_non_dumpable() is False


def test_a_libc_that_cannot_be_loaded_returns_false_and_never_raises(monkeypatch) -> None:
    def refusing(*args: object, **kwargs: object) -> object:
        raise OSError("no libc here")

    monkeypatch.setattr(hardening.ctypes, "CDLL", refusing)

    assert hardening.set_non_dumpable() is False


def test_a_libc_without_prctl_returns_false(monkeypatch) -> None:
    class _LibcWithoutPrctl:
        pass

    monkeypatch.setattr(hardening.ctypes, "CDLL", lambda *args, **kwargs: _LibcWithoutPrctl())

    assert hardening.set_non_dumpable() is False


def test_a_refused_prctl_returns_false(monkeypatch) -> None:
    calls: list[tuple[object, ...]] = []

    class _RefusingPrctl:
        # A ctypes function accepts argtypes/restype; a refused call gives -1.
        def __call__(self, *args: object) -> int:
            calls.append(args)
            return -1

    class _RefusingLibc:
        prctl = _RefusingPrctl()

    # prctl is a Linux facility; without this the platform check would return
    # False before the fake is ever called.
    monkeypatch.setattr(hardening.sys, "platform", "linux")
    monkeypatch.setattr(hardening.ctypes, "CDLL", lambda *args, **kwargs: _RefusingLibc())

    assert hardening.set_non_dumpable() is False
    # The False came from the -1 return: prctl was reached and called once,
    # not skipped by an error the except clause swallowed.
    assert len(calls) == 1
