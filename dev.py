"""Run a command in this project's virtualenv, creating or repairing it first.

    python3 dev.py pytest
    python3 dev.py streamlit run streamlit_app.py
    python3 dev.py         (prints the venv's python path)

The venv is .venv in the project folder (git-ignored), or under $WORKON_HOME
if that is set. A venv only works on the machine that made it, so each clone
builds its own. VS Code's Python extension finds .venv by itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
STAMP = "doc-pipeline-install.sha256"
# Set to 1 (by the pre-commit hook) to never create, delete or install anything.
NO_SETUP = "DOC_PIPELINE_NO_SETUP"


def venv_dir() -> Path:
    home = os.environ.get("WORKON_HOME")
    if not home:
        return (ROOT / ".venv").resolve()
    home = Path(home).expanduser()
    if not home.is_absolute():
        sys.exit("WORKON_HOME must be an absolute path.")
    return (home / ROOT.name).resolve()


def venv_python(env: Path) -> Path:
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def works(python: Path) -> bool:
    # A venv copied from another machine still has python.exe, but it can't start.
    # One whose creation failed (python3-venv missing) runs but has no pip.
    if not python.exists():
        return False
    return subprocess.run([python, "-c", "import pip"], capture_output=True).returncode == 0


def install_key() -> str:
    # The editable install points at this folder, so a move needs a reinstall too.
    # A change to the lock needs one as well.
    data = (ROOT / "pyproject.toml").read_bytes() + (ROOT / "requirements-dev.txt").read_bytes()
    data += str(ROOT).encode()
    return hashlib.sha256(data).hexdigest()


def foreign_venv(env: Path) -> bool:
    # True if pyvenv.cfg's `home =` is a Windows path on POSIX or a POSIX path on Windows.
    try:
        lines = (env / "pyvenv.cfg").read_text(errors="replace").splitlines()
    except OSError:
        return False
    for line in lines:
        key, _, value = line.partition("=")
        if key.strip().lower() == "home":
            value = value.strip()
            windows = bool(re.match(r"[A-Za-z]:", value)) or "\\" in value
            posix = value.startswith("/")
            return posix if os.name == "nt" else windows
    return False


def enable_hook() -> None:
    # Turns on the pre-commit hook in .githooks/ for this clone. Claude Code's
    # sandbox blocks writes to .git/config, so then say what to run instead.
    if not (ROOT / ".git").exists() or not shutil.which("git"):
        return
    git = ["git", "-C", str(ROOT), "config"]
    current = subprocess.run([*git, "--get", "core.hooksPath"], capture_output=True, text=True)
    value = current.stdout.strip()
    if current.returncode not in (0, 1):
        print(
            f"Couldn't read core.hooksPath (git exit {current.returncode}); the pre-commit hook may be off. "
            "Turn it on with: git config core.hooksPath .githooks",
            file=sys.stderr,
        )
        return
    if value == ".githooks":
        return
    if current.returncode == 1:
        if subprocess.run([*git, "core.hooksPath", ".githooks"], capture_output=True).returncode == 0:
            print("Turned on the pre-commit hook (git config core.hooksPath .githooks).", file=sys.stderr)
        else:
            print("The pre-commit hook is off. Turn it on with: git config core.hooksPath .githooks", file=sys.stderr)
    else:
        # An empty value counts too: git then runs no hooks at all.
        shown = value or "an empty value"
        print(
            f"core.hooksPath is already set to {shown}, so the project's pre-commit hook in .githooks/ isn't active. "
            "To turn it on for this clone: git config --local core.hooksPath .githooks",
            file=sys.stderr,
        )


def pip(python: Path, *args: str | Path) -> bool:
    # pip's output goes to stderr, so `python3 dev.py` prints only the venv path.
    return subprocess.run([python, "-m", "pip", *args], stdout=sys.stderr).returncode == 0


def install(python: Path, env: Path, created: bool) -> bool:
    # True if the venv now matches the lock and pyproject.toml.
    # Dependencies come from the hashed lock, as wheels only so no build script runs.
    # Only the project itself is built, from this folder.
    if not pip(python, "install", "--require-hashes", "--only-binary", ":all:", "-r", ROOT / "requirements-dev.txt"):
        if created:
            sys.exit(
                f"Couldn't install requirements-dev.txt into the new venv at {env}; "
                "check your network connection and the lock (see README)."
            )
        return False
    # With --no-index pip can only use what the lock installed, so an unmet
    # core, app or dev requirement (or version range) fails without network.
    ok = pip(python, "install", "--no-index", "--no-build-isolation", "-e", f"{ROOT}[app,dev]") and pip(python, "check")
    if ok:
        warn_extra_packages(python)
    return ok


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def warn_extra_packages(python: Path) -> None:
    # Packages installed by hand are not in the lock; say so, but leave them.
    listed = subprocess.run(
        [python, "-m", "pip", "list", "--format=json", "--exclude-editable"], capture_output=True, text=True
    )
    if listed.returncode != 0:
        return
    lock = (ROOT / "requirements-dev.txt").read_text()
    locked = {canonical(m[1]) for m in re.finditer(r"^([A-Za-z0-9._-]+)==", lock, re.MULTILINE)}
    extra = sorted({canonical(p["name"]) for p in json.loads(listed.stdout)} - locked - {"pip", "setuptools"})
    if extra:
        print(
            f"Installed but not in requirements-dev.txt: {', '.join(extra)}. "
            "They are not hash-checked; delete the venv to get a clean one.",
            file=sys.stderr,
        )


def ensure_venv() -> Path:
    env = venv_dir()
    python = venv_python(env)
    # The default .venv is allowed inside the project; any other venv must be outside it.
    if env == ROOT or env in ROOT.parents or (ROOT in env.parents and env != (ROOT / ".venv").resolve()):
        sys.exit(f"The venv folder {env} is this project, contains it, or is inside it; change WORKON_HOME.")
    if os.environ.get(NO_SETUP) == "1":
        stamp = env / STAMP
        if not works(python) or not stamp.exists() or stamp.read_text().strip() != install_key():
            sys.exit(f"The venv at {env} is missing or out of date, and {NO_SETUP}=1 forbids setting it up. Run python3 dev.py once from a terminal.")
        return python
    created = not works(python)
    if created:
        if sys.version_info < MIN_PYTHON:
            wanted = ".".join(map(str, MIN_PYTHON))
            sys.exit(f"Python {wanted}+ is needed to create the venv; this is {sys.version.split()[0]}.")
        if env.exists():
            if not (env / "pyvenv.cfg").exists():
                sys.exit(f"{env} is not a venv; remove it or change WORKON_HOME.")
            if foreign_venv(env):
                sys.exit(f"The venv at {env} was made on another OS; delete it yourself or set WORKON_HOME.")
            print(f"Rebuilding broken venv at {env}", file=sys.stderr, flush=True)
            shutil.rmtree(env)
        else:
            print(f"Creating venv at {env}", file=sys.stderr, flush=True)
        venv.create(env, with_pip=True)

    stamp = env / STAMP
    key = install_key()
    if not stamp.exists() or stamp.read_text().strip() != key:
        if install(python, env, created):
            stamp.write_text(key)
        else:
            # Warn but still run the command: it may be the one that fixes the lock.
            print(
                "requirements-dev.txt doesn't install or doesn't match pyproject.toml; check your network "
                f"connection, regenerate the locks (see README), or delete {env} if you installed packages by hand.",
                file=sys.stderr,
            )
    enable_hook()
    return python


def main() -> int:
    python = ensure_venv()
    if len(sys.argv) < 2:
        print(python)
        return 0
    env = None
    if sys.argv[1:4] == ["uv", "pip", "compile"] and "UV_CUSTOM_COMPILE_COMMAND" not in os.environ:
        # So the lock headers name this command, not a bare `uv pip compile`.
        env = {**os.environ, "UV_CUSTOM_COMPILE_COMMAND": "python3 dev.py " + shlex.join(sys.argv[1:])}
    try:
        # The hooks run in a folder of untrusted files; -I keeps them off the import path.
        isolate = ["-I"] if os.environ.get(NO_SETUP) == "1" else []
        return subprocess.run([python, *isolate, "-m", *sys.argv[1:]], env=env).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
