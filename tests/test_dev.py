from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from gitenv import scrub_process

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("dev", ROOT / "dev.py")
dev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dev)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("WORKON_HOME", raising=False)
    monkeypatch.delenv(dev.NO_SETUP, raising=False)
    # Whatever git variables the test run inherited must not point at the real repo.
    scrub_process(monkeypatch, str(tmp_path.parent))


def test_relative_workon_home_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORKON_HOME", "venvs")
    with pytest.raises(SystemExit):
        dev.ensure_venv()


def test_workon_home_at_the_projects_parent_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    # $WORKON_HOME/doc-pipeline would be the project folder itself.
    monkeypatch.setenv("WORKON_HOME", str(dev.ROOT.parent))
    with pytest.raises(SystemExit):
        dev.ensure_venv()


def test_a_folder_that_is_not_a_venv_is_left_alone(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / dev.ROOT.name
    folder.mkdir()
    (folder / "keep.txt").write_text("mine")
    monkeypatch.setenv("WORKON_HOME", str(tmp_path))
    with pytest.raises(SystemExit):
        dev.ensure_venv()
    assert (folder / "keep.txt").read_text() == "mine"


@pytest.mark.skipif(os.name == "nt", reason="a Windows home path is only foreign on POSIX")
def test_a_venv_made_on_windows_is_not_deleted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    folder = tmp_path / dev.ROOT.name
    folder.mkdir()
    (folder / "pyvenv.cfg").write_text("home = C:\\Users\\me\\Python314\nversion = 3.14.0\n")
    monkeypatch.setenv("WORKON_HOME", str(tmp_path))

    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not delete or build a venv")

    monkeypatch.setattr(dev.shutil, "rmtree", boom)
    monkeypatch.setattr(dev.venv, "create", boom)
    with pytest.raises(SystemExit):
        dev.ensure_venv()
    assert (folder / "pyvenv.cfg").exists()


def test_no_setup_never_creates_a_venv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("WORKON_HOME", str(tmp_path))
    monkeypatch.setenv(dev.NO_SETUP, "1")
    with pytest.raises(SystemExit) as info:
        dev.ensure_venv()
    assert "python3 dev.py" in str(info.value)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(not (ROOT / ".git").exists() or not shutil.which("git"), reason="needs a git checkout")
def test_clone_has_the_pre_commit_hook_turned_on() -> None:
    done = subprocess.run(["git", "-C", str(ROOT), "config", "--get", "core.hooksPath"], capture_output=True, text=True)
    assert done.stdout.strip() == ".githooks", "run: git config core.hooksPath .githooks"
    for name in ("pre-commit", "pre-merge-commit", "commit-msg", "pre-push"):
        hook = ROOT / ".githooks" / name
        assert os.access(hook, os.X_OK), f"git ignores a hook that isn't executable: chmod +x .githooks/{name}"
        staged = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-s", "--", f".githooks/{name}"], capture_output=True, text=True)
        # Not tracked yet is fine; once it is, git must have it as executable.
        assert not staged.stdout or staged.stdout.startswith("100755 "), f"run: git update-index --chmod=+x .githooks/{name}"


@pytest.mark.skipif(not shutil.which("git"), reason="needs git")
def test_empty_hooks_path_is_reported(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(dev, "ROOT", tmp_path)
    assert subprocess.run(["git", "init", "-q", str(tmp_path)]).returncode == 0
    assert subprocess.run(["git", "-C", str(tmp_path), "config", "core.hooksPath", ""]).returncode == 0
    dev.enable_hook()
    assert "empty value" in capsys.readouterr().err


def test_no_setup_runs_the_command_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    # The hooks run in a folder of untrusted files; -I keeps them off the import path.
    calls = []
    monkeypatch.setattr(dev, "ensure_venv", lambda: Path("venv-python"))
    monkeypatch.setattr(dev.subprocess, "run", lambda args, **kw: calls.append(args) or subprocess.CompletedProcess(args, 0))
    monkeypatch.setattr(dev.sys, "argv", ["dev.py", "detect_secrets", "--version"])
    monkeypatch.setenv(dev.NO_SETUP, "1")
    assert dev.main() == 0
    assert calls[-1] == [Path("venv-python"), "-I", "-m", "detect_secrets", "--version"]
    monkeypatch.delenv(dev.NO_SETUP)
    dev.main()
    assert "-I" not in calls[-1]


def test_the_empty_hooks_path_test_never_touches_the_real_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Even with git variables pointing at the real repository, enable_hook works on ROOT only.
    monkeypatch.setenv("GIT_DIR", str(ROOT / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(ROOT / ".git" / "index"))
    scrub_process(monkeypatch, str(tmp_path.parent))
    assert "GIT_DIR" not in os.environ and "GIT_INDEX_FILE" not in os.environ
    assert os.environ["GIT_CEILING_DIRECTORIES"] == str(tmp_path.parent)
