"""Process-level hardening for the Streamlit server (Linux, best-effort).

``set_non_dumpable`` clears the kernel's dumpable flag on the server process
(``prctl(PR_SET_DUMPABLE, 0)``). A non-dumpable process can neither be
ptraced nor have its ptrace-gated ``/proc/<pid>`` files (``mem``, ``environ``,
``maps``, ``fd``) read by another process of the same user, so a compromised
render child - which runs as the same user - cannot read the server's memory,
where every session's API keys sit, even where Yama ``ptrace_scope`` is 0.
The flag is set only when this runs as the Streamlit app on Linux (the app
calls it at start-up); using the package as a library leaves the process
dumpable. The call stays best-effort on purpose: a platform without ``prctl``,
or one that refuses it, keeps the previous behaviour instead of breaking
start-up.

That is where the hardening ends. Rendering is the only stage that runs in a
child, and the child is bounded, not sandboxed: still the same OS user,
limited by the render rlimits, a timeout and its own process group. A
compromised child can still write every file the server can write - this
application's own code, the venv's provider SDKs, other sessions'
workspaces - and read ``.streamlit/secrets.toml`` (the sign-in and cookie
secrets) immediately. Its writes to server code or config are not held back
until a restart: Streamlit always polls ``.streamlit/config.toml`` and
``.streamlit/secrets.toml``, so the child can touch either one to force a
live reload and have its changes take effect in the running server. Turning
the file watcher off and importing the SDKs' top-level modules at start-up
only narrow this (a rerun does not re-import modules already in
``sys.modules``); they are not a boundary. Running the render child under a
dedicated user account of its own, or a Landlock policy, would close this;
it is not done.
"""

from __future__ import annotations

import ctypes
import sys

# From <linux/prctl.h>: PR_SET_DUMPABLE sets the flag, PR_GET_DUMPABLE reads it.
PR_SET_DUMPABLE = 4
PR_GET_DUMPABLE = 3


def set_non_dumpable() -> bool:
    """Make this process non-dumpable; True only when the kernel accepted it.

    Every failure - another platform, no ``prctl`` in libc, a refused call -
    is reported as False and never raised: the hardening is a loss, not a
    defect, and the app must still start.
    """
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        prctl.restype = ctypes.c_int
        return (
            prctl(
                ctypes.c_int(PR_SET_DUMPABLE),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
                ctypes.c_ulong(0),
            )
            == 0
        )
    except Exception:
        return False
