"""The pre-commit hook, end to end, in a throwaway git repo."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from gitenv import scrub

ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.skipif(not shutil.which("git") or not shutil.which("sh"), reason="needs git and sh")

# Fake keys are built here so that no key-shaped text sits in this file.
GEMINI = "AI" + "za" + "x" * 35
GLM = "0123456789abcdef" * 2 + "." + "AbCdEfGhIjKlMnOp"
OPENAI = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0" * 2
GOOGLE_OAUTH = "GOC" + "SPX-" + "x" * 28
MICROSOFT = "Kd2" + "8Q~" + ("Lm3Xv9Kd4Zq8" * 3)[:34]
PRIVATE_KEY = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
AWS = "AKIA" + "IOSFODNN7EXAMPLE"
FINDING = "pass" + "word = " + '"' + "Zq8Lm2Xv9Kd4" + '"\n'

# Stands in for dev.py: runs the module with the test's own Python, and logs whether the
# hook started it isolated (-I) when STUB_LOG is set.
STUB_DEV = """\
import os, subprocess, sys
if os.environ.get("STUB_LOG"):
    with open(os.environ["STUB_LOG"], "a") as log:
        log.write(f"isolated={{sys.flags.isolated}}\\n")
sys.exit(subprocess.run([{exe!r}, "-m", *sys.argv[1:]]).returncode)
"""
_ceiling = ""


@pytest.fixture(autouse=True)
def _fence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Git must not search above the test's folder, and never use the real repo's variables.
    monkeypatch.setattr(sys.modules[__name__], "_ceiling", str(tmp_path.parent))


def _env() -> dict[str, str]:
    env = dict(os.environ)
    scrub(env, _ceiling)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    # stdout and stderr together, as the user would see them.
    return subprocess.run(
        ["git", *args], cwd=repo, env={**_env(), **(env_extra or {})}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, check=False,
    )  # fmt: skip


def _commit(repo: Path, message: str = "x", hooks: bool = True) -> subprocess.CompletedProcess:
    # Without hooks, the way a commit made with the hooks off (or a bypass) looks to a later push.
    return _git(
        repo, *(["-c", "core.hooksPath=.githooks"] if hooks else []), "-c", "user.name=t",
        "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "commit", "-m", message,
    )  # fmt: skip


def _head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _scan(repo: Path, *args: str) -> None:
    done = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", "--no-verify", *args],
        cwd=repo, env=_env(), stdout=subprocess.PIPE, check=True,
    )  # fmt: skip
    if not args:
        (repo / ".secrets.baseline").write_bytes(done.stdout)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / ".githooks", tmp_path / ".githooks", ignore=shutil.ignore_patterns("__pycache__"))
    for hook in (tmp_path / ".githooks").iterdir():
        if not hook.suffix:
            hook.chmod(0o755)
    (tmp_path / "dev.py").write_text(STUB_DEV.format(exe=sys.executable))
    (tmp_path / ".gitignore").write_text("*.db\n")
    (tmp_path / "readme.txt").write_text("hello\n")
    assert _git(tmp_path, "init", "-q").returncode == 0
    _git(tmp_path, "add", ".gitignore", "dev.py", "readme.txt", ".githooks")
    _scan(tmp_path)
    _git(tmp_path, "add", ".secrets.baseline")
    done = _commit(tmp_path)
    assert done.returncode == 0, done.stdout
    return tmp_path


def _stage(repo: Path, name: str, content: bytes, force: bool = False) -> None:
    (repo / name).write_bytes(content)
    done = _git(repo, "add", *(["-f"] if force else []), "--", name)
    assert done.returncode == 0, done.stdout


# Each key line is one detect-secrets ignores (checked in the test), so only
# check_staged.py's own patterns can block it.
KEYS = {
    "gemini": ("app.py", f'client = f(api_key="{GEMINI}")\n', GEMINI, "gemini/google key"),
    "glm": ("glm.py", f'x = "{GLM}"\n', GLM, "glm key"),
    "openai": ("oai.py", f'client = f(api_key="{OPENAI}")\n', OPENAI, "openai key"),
    "openai-after-escape": ("oai2.py", f'x = 1  # \\n{OPENAI}\n', OPENAI, "openai key"),
    "openai-after-tab": ("oai4.py", f'x = 1  # \\t{OPENAI}\n', OPENAI, "openai key"),
    "openai-after-percent": ("oai3.py", f'x = "token%3D{OPENAI}"\n', OPENAI, "openai key"),
    "google-oauth": ("oauth.py", f'x = "{GOOGLE_OAUTH}"\n', GOOGLE_OAUTH, "google-oauth-secret"),
    "microsoft": ("ms.py", f'x = "{MICROSOFT}"\n', MICROSOFT, "microsoft client secret"),
    "private-key": ("pem.py", f'x = "{PRIVATE_KEY}"\n', PRIVATE_KEY, "private key"),
    "aws": ("aws.py", f'x = "{AWS}"\n', AWS, "aws access key id"),
}
# detect-secrets has detectors for these two, so it blocks them as well.
DETECT_SECRETS_CATCHES = {"private-key", "aws"}
BLOCKED = {
    "pdf-after-a-header": ("mail.txt", b"Subject: x\n" + b"-" * 200 + b"\n%PDF-1.7\n", 'looks like a PDF file'),
    "tar": ("t.txt", b"a" * 257 + b"ustar\x0000", 'looks like a tar file'),
    "xz": ("x.txt", b"\xfd7zXZ\x00\x00\x04", 'looks like a xz file'),
    "bzip2": ("b2.txt", b"BZh91AY&SY", 'looks like a bzip2 file'),
    "zstd": ("z.txt", b"\x28\xb5\x2f\xfd\x00", 'looks like a zstd file'),
    "bmp": ("bmp.txt", b"BM" + b"\x36\x00\x00\x00" + b"\0\0\0\0" + b"\x36\0\0\0" + b"\x28\0\0\0" + b"\0" * 20, 'looks like a BMP file'),
    "jpeg2000": ("j2.txt", b"\x00\x00\x00\x0cjP  \r\n\x87\n", 'looks like a JPEG 2000 file'),
    "rtf": ("r.txt", b"{\\rtf1\\ansi hello}", 'looks like a RTF file'),
    "postscript": ("p.txt", b"%!PS-Adobe-3.0\n", 'looks like a PostScript file'),
    "latin1-text": ("caf.txt", b"caf\xe9\n", 'not UTF-8 text'),
    "nul-byte": ("nul.txt", b"a\0b\n", 'not UTF-8 text'),
    "lfs-pointer": ("big.txt", b"version https://git-lfs.github.com/spec/v1\noid sha256:" + b"0" * 64 + b"\nsize 5\n", 'Git LFS pointer'),
    "eml-name": ("mail.eml", b"hello\n", 'document by name'),
    "tar-name": ("a.TAR", b"hello\n", 'document by name'),
    "msg-name": ("a.msg", b"hello\n", 'document by name'),
    "rtf-name": ("a.rtf", b"hello\n", 'document by name'),
    "ps-name": ("a.ps", b"hello\n", 'document by name'),
    "pdf-signature": ("notes.txt", b"%PDF-1.4 not really a pdf\n", 'looks like a PDF file'),
    "pdf-name": ("scan.PDF", b"hello\n", 'document by name'),
    "utf16": ("wide.txt", "hello\n".encode("utf-16"), 'UTF-16 file'),
    "gif": ("pic.txt", b"GIF89a\x01\x00", 'looks like a GIF file'),
    "gzip": ("data.txt", b"\x1f\x8b\x08\x00rest", 'looks like a gzip file'),
    "webp": ("pic2.txt", b"RIFF\x00\x00\x00\x00WEBPVP8 ", 'looks like a WEBP file'),
    "heic": ("pic3.txt", b"\x00\x00\x00\x18ftypheic\x00\x00", 'looks like a HEIC or AVIF file'),
    "7z": ("a.txt", b"7z\xbc\xaf\x27\x1c\x00\x04", 'looks like a 7z file'),
    "rar": ("b.txt", b"Rar!\x1a\x07\x00", 'looks like a RAR file'),
}


def _detect_secrets_alone(repo: Path, name: str) -> int:
    base = repo / ".secrets.baseline"
    copy = repo / "scratch-baseline.json"
    copy.write_bytes(base.read_bytes())
    done = subprocess.run(
        [sys.executable, "-m", "detect_secrets.pre_commit_hook", "--no-verify", "--baseline", str(copy), "--", name],
        cwd=repo, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )  # fmt: skip
    copy.unlink()
    return done.returncode


@pytest.mark.parametrize("case", KEYS)
def test_key_patterns_block(repo: Path, case: str) -> None:
    name, text, secret, pattern = KEYS[case]
    before = _head(repo)
    _stage(repo, name, text.encode())
    if case not in DETECT_SECRETS_CATCHES:
        assert _detect_secrets_alone(repo, name) == 0, "detect-secrets would catch this line itself"
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert f"{name}:1: {pattern}" in done.stdout
    assert secret not in done.stdout
    assert _head(repo) == before


@pytest.mark.parametrize("case", BLOCKED)
def test_blocked(repo: Path, case: str) -> None:
    name, content, reason = BLOCKED[case]
    before = _head(repo)
    _stage(repo, name, content)
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert _head(repo) == before
    assert f"{name}: {reason}" in done.stdout


def test_force_added_ignored_file_is_blocked(repo: Path) -> None:
    before = _head(repo)
    _stage(repo, "x.db", b"plain text\n", force=True)
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "x.db" in done.stdout
    assert _head(repo) == before


def test_odd_file_names_are_checked_literally(repo: Path) -> None:
    before = _head(repo)
    (repo / ":(x).db").write_bytes(b"plain text\n")
    assert _git(repo, "--literal-pathspecs", "add", "-f", "--", ":(x).db").returncode == 0
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert ":(x).db: ignored" in done.stdout
    assert _head(repo) == before


def test_the_report_names_the_line_but_not_the_key(repo: Path) -> None:
    _stage(repo, "app.py", f'x = 1\nclient = f(api_key="{GEMINI}")\n'.encode())
    done = _commit(repo)
    assert "app.py:2: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout


def test_a_textconv_driver_cannot_hide_a_key(repo: Path) -> None:
    if not shutil.which("sed"):
        pytest.skip("needs sed")
    assert _git(repo, "config", "diff.strip.textconv", "sed s/AIza/XXXX/").returncode == 0
    (repo / ".gitattributes").write_text("*.cfg diff=strip\n")
    before = _head(repo)
    _stage(repo, "a.cfg", f'key = "{GEMINI}"\n'.encode())
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "a.cfg:1: gemini/google key" in done.stdout
    assert _head(repo) == before


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_hidden_working_tree_changes_are_blocked(repo: Path, flag: str) -> None:
    # The scan reads the working tree, which git stops comparing to the index here.
    before = _head(repo)
    _stage(repo, "readme.txt", b"hello\nmore\n")
    assert _git(repo, "update-index", flag, "readme.txt").returncode == 0
    (repo / "readme.txt").write_text(f'hello\nmore\nx = "{GEMINI}"\n')
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "readme.txt: marked assume-unchanged or skip-worktree" in done.stdout
    assert _head(repo) == before


def test_new_detect_secrets_finding_is_blocked(repo: Path) -> None:
    before = _head(repo)
    _stage(repo, "cfg2.py", FINDING.encode())
    assert _detect_secrets_alone(repo, "cfg2.py") == 1
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "cfg2.py" in done.stdout
    assert FINDING.split('"')[1] not in done.stdout
    assert _head(repo) == before


def test_missing_staged_baseline_is_blocked(repo: Path) -> None:
    before = _head(repo)
    assert _git(repo, "rm", "-q", "--cached", ".secrets.baseline").returncode == 0
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert ".secrets.baseline" in done.stdout
    assert "ambiguous" not in done.stdout
    assert _head(repo) == before


def test_partially_staged_file_is_blocked(repo: Path) -> None:
    before = _head(repo)
    _stage(repo, "ok.py", b"x = 1\n")
    (repo / "ok.py").write_text("x = 1\ny = 2\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "ok.py" in done.stdout
    assert _head(repo) == before


def test_only_the_staged_baseline_counts(repo: Path) -> None:
    # A finding accepted in the working-tree baseline alone must not let the commit through.
    before = _head(repo)
    _stage(repo, "cfg.py", FINDING.encode())
    _scan(repo, "--baseline", ".secrets.baseline")
    assert "cfg.py" in (repo / ".secrets.baseline").read_text()
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "cfg.py" in done.stdout
    assert _head(repo) == before


def test_ordinary_change_is_allowed(repo: Path) -> None:
    before = _head(repo)
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    assert _head(repo) != before


def test_risk_based_text_is_not_a_key(repo: Path) -> None:
    _stage(repo, "notes.md", b"Use a risk-based-approach-for-scoring-documents here.\n")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout


def test_baseline_is_not_modified_when_line_numbers_move(repo: Path) -> None:
    _stage(repo, "cfg.py", FINDING.encode())
    _scan(repo, "--baseline", ".secrets.baseline")
    assert "cfg.py" in (repo / ".secrets.baseline").read_text()
    _git(repo, "add", ".secrets.baseline")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    baseline = (repo / ".secrets.baseline").read_bytes()
    # The finding moves down two lines, which detect-secrets would rewrite the baseline for.
    _stage(repo, "cfg.py", b"import os\n\n" + FINDING.encode())
    before = _head(repo)
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    assert _head(repo) != before
    assert "out of date" in done.stdout
    assert (repo / ".secrets.baseline").read_bytes() == baseline


STUB_EXIT_3 = """\
import sys
if {change}:
    with open(sys.argv[sys.argv.index("--baseline") + 1], "a") as f:
        f.write("\\n")
sys.exit(3)
"""


def test_exit_3_without_a_baseline_change_is_blocked(repo: Path) -> None:
    # On native Windows an aborted process also exits 3.
    (repo / "dev.py").write_text(STUB_EXIT_3.format(change=False))
    before = _head(repo)
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert _head(repo) == before


def test_exit_3_with_a_baseline_change_is_allowed_with_a_warning(repo: Path) -> None:
    (repo / "dev.py").write_text(STUB_EXIT_3.format(change=True))
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    assert "out of date" in done.stdout


def test_line_numbers_are_right_even_if_git_adds_context(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIFF_OPTS", "--unified=3")
    lines = [f"line {n}\n" for n in range(1, 11)]
    _stage(repo, "ten.txt", "".join(lines).encode())
    assert _commit(repo).returncode == 0
    lines[1] = "changed\n"
    lines[4] = f'client = f(api_key="{GEMINI}")\n'
    _stage(repo, "ten.txt", "".join(lines).encode())
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "ten.txt:5: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout


def test_environment_variables_that_point_at_another_repo_are_ignored(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_DIR", str(ROOT / ".git"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(ROOT / ".git" / "index"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "/nonexistent")
    top = _git(repo, "rev-parse", "--show-toplevel").stdout.strip()
    assert Path(top).resolve() == repo.resolve()
    assert _git(repo, "config", "--get", "core.hooksPath").stdout.strip() == ""


def test_hooks_start_python_isolated(repo: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path.parent / f"{tmp_path.name}-stub.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    assert log.read_text().split() == ["isolated=1"]


def test_text_with_accents_and_odd_bmp_lookalikes_is_allowed(repo: Path) -> None:
    _stage(repo, "caf.txt", "café\n".encode())
    _stage(repo, "bm.txt", b"BMW makes cars\n")
    done = _commit(repo)
    assert done.returncode == 0, done.stdout


def test_svg_with_a_keyword_secret_is_blocked(repo: Path) -> None:
    # detect-secrets skips .svg files unless its non-text filter is disabled.
    before = _head(repo)
    _stage(repo, "icon.svg", FINDING.encode())
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "icon.svg:1" in done.stdout
    assert FINDING.split('"')[1] not in done.stdout
    assert _head(repo) == before


def test_the_allowlist_pragma_does_not_hide_a_project_key(repo: Path) -> None:
    pragma = "pragma: " + "allowlist secret"
    _stage(repo, "app.py", f'x = "{GEMINI}"  # {pragma}\n'.encode())
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "app.py:1: gemini/google key" in done.stdout


def test_filter_attribute_is_blocked(repo: Path) -> None:
    (repo / ".gitattributes").write_text("*.dat filter=lfs\n")
    _stage(repo, "x.dat", b"plain text\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "x.dat: has a filter attribute" in done.stdout


def test_working_tree_encoding_attribute_is_blocked(repo: Path) -> None:
    (repo / ".gitattributes").write_text("*.cfg working-tree-encoding=UTF-16LE\n")
    _stage(repo, "enc.cfg", "x = 1\n".encode("utf-16-le"))
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert "enc.cfg: has a working-tree-encoding attribute" in done.stdout


def _edit_baseline(repo: Path, change) -> None:
    path = repo / ".secrets.baseline"
    baseline = json.loads(path.read_text())
    change(baseline)
    path.write_text(json.dumps(baseline, indent=2) + "\n")
    assert _git(repo, "add", ".secrets.baseline").returncode == 0


def _drop_keyword_detector(b: dict) -> None:
    b["plugins_used"] = [p for p in b["plugins_used"] if p["name"] != "KeywordDetector"]


def _loosen_hex_limit(b: dict) -> None:
    for plugin in b["plugins_used"]:
        if plugin["name"] == "HexHighEntropyString":
            plugin["limit"] = 5.0


def _exclude_files(b: dict) -> None:
    b["filters_used"].append({"path": "detect_secrets.filters.regex.should_exclude_file", "pattern": ["^cfg"]})


def _file_plugin(b: dict) -> None:
    b["plugins_used"].append({"name": "Mine", "path": "file:///tmp/mine.py"})


@pytest.mark.parametrize("change", [_drop_keyword_detector, _loosen_hex_limit, _exclude_files, _file_plugin])
def test_changed_baseline_settings_are_blocked(repo: Path, change) -> None:
    before = _head(repo)
    _edit_baseline(repo, change)
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert ".secrets.baseline: the detect-secrets settings" in done.stdout
    assert _head(repo) == before


@pytest.mark.skipif(os.name == "nt", reason="needs symlinks")
def test_file_to_symlink_and_back_is_not_a_false_block(repo: Path) -> None:
    (repo / "link.txt").unlink(missing_ok=True)
    _stage(repo, "link.txt", b"plain\n")
    assert _commit(repo).returncode == 0
    (repo / "link.txt").unlink()
    (repo / "link.txt").symlink_to("readme.txt")
    assert _git(repo, "add", "link.txt").returncode == 0
    done = _commit(repo)
    assert done.returncode == 0, done.stdout
    (repo / "link.txt").unlink()
    _stage(repo, "link.txt", f"first\nclient = f(api_key=\"{GEMINI}\")\n".encode())
    done = _commit(repo)
    # The new file's content is still scanned, at the right line.
    assert done.returncode != 0, done.stdout
    assert "link.txt:2: gemini/google key" in done.stdout
    assert "different files" not in done.stdout


def test_commit_message_with_a_key_is_blocked(repo: Path) -> None:
    before = _head(repo)
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo, message=f"fix\n\nuses {GEMINI}")
    assert done.returncode != 0, done.stdout
    assert "commit message:3: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout
    assert _head(repo) == before


def _side_branch_with_a_key(repo: Path) -> None:
    assert _git(repo, "checkout", "-q", "-b", "side").returncode == 0
    _stage(repo, "side.py", f'client = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    assert _git(repo, "checkout", "-q", "-").returncode == 0
    _stage(repo, "other.py", b"y = 2\n")
    assert _commit(repo).returncode == 0


def _merge(repo: Path) -> subprocess.CompletedProcess:
    return _git(
        repo, "-c", "core.hooksPath=.githooks", "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "-c", "commit.gpgsign=false", "merge", "--no-ff", "--no-edit", "side",
    )  # fmt: skip


def test_clean_merge_of_a_branch_with_a_key_is_blocked(repo: Path) -> None:
    _side_branch_with_a_key(repo)
    before = _head(repo)
    done = _merge(repo)
    assert done.returncode != 0, done.stdout
    assert "side.py:1: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout
    assert _head(repo) == before


# --- pre-push ---


@pytest.fixture
def remote(repo: Path, tmp_path: Path) -> Path:
    bare = tmp_path.parent / f"{tmp_path.name}-remote.git"
    assert _git(tmp_path.parent, "init", "--bare", "-q", str(bare)).returncode == 0
    assert _git(repo, "remote", "add", "origin", str(bare)).returncode == 0
    return bare


def _push(repo: Path, ref: str = "HEAD:refs/heads/main", hooks: bool = True) -> subprocess.CompletedProcess:
    return _git(repo, *(["-c", "core.hooksPath=.githooks"] if hooks else []), "push", "origin", ref)


def _remote_heads(bare: Path) -> str:
    return _git(bare, "for-each-ref", "--format=%(refname) %(objectname)", "refs/heads").stdout.strip()


def test_ordinary_push_is_allowed(repo: Path, remote: Path) -> None:
    done = _push(repo)
    assert done.returncode == 0, done.stdout
    assert _remote_heads(remote) == f"refs/heads/main {_head(repo)}"


def test_push_of_a_commit_made_without_hooks_is_blocked(repo: Path, remote: Path) -> None:
    assert _push(repo).returncode == 0
    before = _remote_heads(remote)
    _stage(repo, "app.py", f'x = 1\nclient = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert f"{_head(repo)[:10]} app.py:2: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout
    assert _remote_heads(remote) == before


def test_push_scans_the_whole_file_not_only_the_added_lines(repo: Path, remote: Path) -> None:
    _stage(repo, "app.py", f'client = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    (repo / "app.py").write_text(f'client = f(api_key="{GEMINI}")\nz = 1\n')
    assert _git(repo, "add", "app.py").returncode == 0
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0
    assert done.stdout.count("app.py:1: gemini/google key") == 2


def test_push_with_a_key_in_a_commit_message_is_blocked(repo: Path, remote: Path) -> None:
    _stage(repo, "ok.py", b"x = 1\n")
    assert _commit(repo, message=f"oops {OPENAI}", hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert f"{_head(repo)[:10]} commit message:1: openai key" in done.stdout
    assert OPENAI not in done.stdout
    assert _remote_heads(remote) == ""


@pytest.mark.parametrize(
    ("name", "content", "text"),
    [
        ("scan.pdf", b"hello\n", "document by name"),
        ("notes.txt", b"%PDF-1.4\n", "looks like a PDF"),
        ("wide.txt", "hi\n".encode("utf-16"), "UTF-16"),
        ("caf.txt", b"caf\xe9\n", "not UTF-8 text"),
        ("big.txt", b"version https://git-lfs.github.com/spec/v1\noid sha256:0\n", "Git LFS pointer"),
    ],
)
def test_push_blocks_documents_and_unscannable_files(repo: Path, remote: Path, name: str, content: bytes, text: str) -> None:
    _stage(repo, name, content)
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert f"{_head(repo)[:10]} {name}: " in done.stdout
    assert text in done.stdout


def test_push_blocks_ignored_and_filtered_files(repo: Path, remote: Path) -> None:
    _stage(repo, ".gitattributes", b"*.dat filter=lfs\n")
    _stage(repo, "x.dat", b"plain\n")
    _stage(repo, "y.db", b"plain\n", force=True)
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert "x.dat: has a filter attribute" in done.stdout
    assert "y.db: ignored by .gitignore" in done.stdout


def test_push_blocks_changed_baseline_settings(repo: Path, remote: Path) -> None:
    _edit_baseline(repo, _drop_keyword_detector)
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert ".secrets.baseline: the detect-secrets settings" in done.stdout


def test_push_runs_detect_secrets_on_the_pushed_files(repo: Path, remote: Path) -> None:
    # A keyword secret none of the project's own patterns know.
    _stage(repo, "cfg3.py", FINDING.encode())
    _stage(repo, "icon.svg", FINDING.encode())
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert f"{_head(repo)[:10]} detect-secrets: " in done.stdout
    assert "cfg3.py:1" in done.stdout
    assert "icon.svg:1" in done.stdout
    assert FINDING.split('"')[1] not in done.stdout
    assert _remote_heads(remote) == ""


def test_push_starts_detect_secrets_isolated(repo: Path, remote: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log = tmp_path.parent / f"{tmp_path.name}-stub.log"
    monkeypatch.setenv("STUB_LOG", str(log))
    assert _push(repo).returncode == 0
    assert "isolated=1" in log.read_text().split()
    assert set(log.read_text().split()) == {"isolated=1"}


@pytest.mark.parametrize(("change", "allowed"), [(False, False), (True, True)])
def test_push_accepts_exit_3_only_if_the_baseline_copy_changed(repo: Path, remote: Path, change: bool, allowed: bool) -> None:
    (repo / "dev.py").write_text(STUB_EXIT_3.format(change=change))
    _stage(repo, "ok.py", b"x = 1\n")
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert (done.returncode == 0) == allowed, done.stdout


def test_commits_already_on_a_remote_tracking_ref_are_not_scanned_again(repo: Path, remote: Path) -> None:
    # What the hooks can't see: a bad commit pushed with the hooks off stays pushed.
    _stage(repo, "app.py", f'client = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    assert _push(repo, hooks=False).returncode == 0
    _stage(repo, "ok.py", b"x = 1\n")
    assert _commit(repo).returncode == 0
    done = _push(repo)
    assert done.returncode == 0, done.stdout


def test_deleting_a_remote_branch_is_allowed(repo: Path, remote: Path) -> None:
    assert _push(repo, "HEAD:refs/heads/side").returncode == 0
    done = _push(repo, ":refs/heads/side")
    assert done.returncode == 0, done.stdout


def test_commits_mode_reports_without_a_push(repo: Path) -> None:
    _stage(repo, "app.py", f'client = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    done = subprocess.run(
        [sys.executable, "-I", ".githooks/check_staged.py", "--commits", _head(repo)],
        cwd=repo, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    assert done.returncode == 1
    assert f"{_head(repo)[:10]} app.py:1: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout


# --- round 3: baseline bypasses, baseline-less history, tags, more filters, messages ---

CONFIG = ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false")


def _duplicate_keyword_detector(b: dict) -> None:
    b["plugins_used"].append({"name": "KeywordDetector", "keyword_exclude": "."})


def _wrong_version(b: dict) -> None:
    b["version"] = "1.4.0"


def _custom_plugin_paths(b: dict) -> None:
    b["custom_plugin_paths"] = ["file:///tmp/mine.py"]


@pytest.mark.parametrize("change", [_duplicate_keyword_detector, _wrong_version, _custom_plugin_paths])
def test_baseline_bypasses_are_blocked(repo: Path, change) -> None:
    before = _head(repo)
    _edit_baseline(repo, change)
    _stage(repo, "ok.py", b"x = 1\n")
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert ".secrets.baseline: the detect-secrets settings" in done.stdout
    assert _head(repo) == before


@pytest.mark.parametrize("name", ["docs/swagger_notes.py", "poetry.lock"])
def test_files_detect_secrets_would_skip_are_still_scanned(repo: Path, remote: Path, name: str) -> None:
    before = _head(repo)
    (repo / "docs").mkdir(exist_ok=True)
    _stage(repo, name, FINDING.encode())
    done = _commit(repo)
    assert done.returncode != 0, done.stdout
    assert f"{name}:1" in done.stdout
    assert FINDING.split('"')[1] not in done.stdout
    assert _head(repo) == before
    # and the same file in a commit made with the hooks off is caught on push
    assert _commit(repo, hooks=False).returncode == 0
    pushed = _push(repo)
    assert pushed.returncode != 0, pushed.stdout
    assert f"{name}:1" in pushed.stdout


def _history_without_a_baseline_at_first(repo: Path, first: dict[str, bytes], later_removes: str | None = None) -> None:
    # A root commit with no .secrets.baseline, then one that adds it (like 05390d8 and cf6b40b).
    assert _git(repo, "checkout", "-q", "--orphan", "hist").returncode == 0
    assert _git(repo, "rm", "-rf", "-q", "--cached", ".").returncode == 0
    for name, content in first.items():
        _stage(repo, name, content)
    assert _commit(repo, message="first", hooks=False).returncode == 0
    if later_removes:
        assert _git(repo, "rm", "-q", "--cached", later_removes).returncode == 0
    assert _git(repo, "add", ".secrets.baseline", ".githooks", "dev.py", ".gitignore").returncode == 0
    assert _commit(repo, message="second", hooks=False).returncode == 0


def test_history_that_starts_without_a_baseline_pushes_with_a_warning(repo: Path, remote: Path) -> None:
    _history_without_a_baseline_at_first(repo, {"readme.txt": b"hello\n"})
    done = _push(repo, "HEAD:refs/heads/hist")
    assert done.returncode == 0, done.stdout
    assert "none in this commit; checked with the baseline of" in done.stdout


def test_a_baseline_less_commit_with_a_new_secret_is_still_blocked(repo: Path, remote: Path) -> None:
    _history_without_a_baseline_at_first(repo, {"old.py": FINDING.encode()}, later_removes="old.py")
    done = _push(repo, "HEAD:refs/heads/hist")
    assert done.returncode != 0, done.stdout
    assert "detect-secrets: old.py:1" in done.stdout
    assert FINDING.split('"')[1] not in done.stdout
    assert _remote_heads(remote) == ""


def test_no_baseline_anywhere_is_blocked(repo: Path, remote: Path) -> None:
    assert _git(repo, "checkout", "-q", "--orphan", "bare").returncode == 0
    assert _git(repo, "rm", "-rf", "-q", "--cached", ".").returncode == 0
    _stage(repo, "ok.py", b"x = 1\n")
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo, "HEAD:refs/heads/bare")
    assert done.returncode != 0, done.stdout
    assert "missing here, and the baseline of" in done.stdout


def _tag(repo: Path, name: str, *args: str) -> str:
    done = _git(repo, *CONFIG, "tag", *args, name, *([] if "-a" not in args else []))
    assert done.returncode == 0, done.stdout
    return _git(repo, "rev-parse", f"refs/tags/{name}").stdout.strip()


def test_tag_with_a_key_in_its_message_is_blocked(repo: Path, remote: Path) -> None:
    obj = _tag(repo, "v1", "-a", "-m", f"released with {GEMINI}")
    done = _push(repo, "refs/tags/v1")
    assert done.returncode != 0, done.stdout
    assert f"{obj[:10]} tag message:1: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout


def test_a_key_in_the_inner_tag_of_a_tag_is_blocked(repo: Path, remote: Path) -> None:
    _tag(repo, "inner", "-a", "-m", f"inner {OPENAI}")
    done = _git(repo, *CONFIG, "tag", "-a", "-m", "outer", "outer", "refs/tags/inner")
    assert done.returncode == 0, done.stdout
    assert _git(repo, "cat-file", "-p", "refs/tags/outer").stdout.splitlines()[1] == "type tag"
    pushed = _push(repo, "refs/tags/outer")
    assert pushed.returncode != 0, pushed.stdout
    assert "tag message:1: openai key" in pushed.stdout
    assert OPENAI not in pushed.stdout


def test_tag_of_a_tree_or_a_blob_is_blocked(repo: Path, remote: Path) -> None:
    (repo / "blob.bin").write_bytes(b"%PDF-1.4\n")
    blob = _git(repo, "hash-object", "-w", "blob.bin").stdout.strip()
    (repo / "blob.bin").unlink()
    for name, target, kind in [("t-tree", "HEAD^{tree}", "tree"), ("t-blob", blob, "blob")]:
        assert _git(repo, *CONFIG, "tag", "-a", "-m", "x", name, target).returncode == 0
        done = _push(repo, f"refs/tags/{name}")
        assert done.returncode != 0, done.stdout
        assert f"a {kind}" in done.stdout
    assert _remote_heads(remote) == ""


def test_lightweight_and_annotated_tags_on_a_clean_commit_are_allowed(repo: Path, remote: Path) -> None:
    assert _git(repo, "tag", "lw").returncode == 0
    assert _git(repo, *CONFIG, "tag", "-a", "-m", "release 1", "ann").returncode == 0
    for name in ("lw", "ann"):
        done = _push(repo, f"refs/tags/{name}")
        assert done.returncode == 0, done.stdout


def _editor(tmp_path: Path) -> Path:
    # An editor that puts $COMMIT_TEXT above what git wrote (as typing a message does).
    script = tmp_path.parent / f"{tmp_path.name}-editor.sh"
    script.write_text('#!/bin/sh\n{ printf "%s\\n\\n" "$COMMIT_TEXT"; cat "$1"; } > "$1.new" && mv "$1.new" "$1"\n')
    script.chmod(0o755)
    return script


def _commit_verbose(repo: Path, tmp_path: Path, text: str, *extra: str) -> subprocess.CompletedProcess:
    return _git(
        repo, "-c", "core.hooksPath=.githooks", *CONFIG, *extra, "commit", "-v",
        env_extra={"GIT_EDITOR": str(_editor(tmp_path)), "COMMIT_TEXT": text},
    )  # fmt: skip


def test_commit_v_may_remove_a_key_but_a_key_in_the_message_is_blocked(repo: Path, tmp_path: Path) -> None:
    _stage(repo, "app.py", f'x = 1\nclient = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    # The staged diff that `commit -v` appends shows the key being removed.
    _stage(repo, "app.py", b"x = 1\n")
    done = _commit_verbose(repo, tmp_path, "remove the key")
    assert done.returncode == 0, done.stdout
    _stage(repo, "ok.py", b"y = 2\n")
    before = _head(repo)
    done = _commit_verbose(repo, tmp_path, f"oops {GEMINI}")
    assert done.returncode != 0, done.stdout
    assert "commit message:1: gemini/google key" in done.stdout
    assert _head(repo) == before


def test_a_comment_line_with_a_key_is_still_blocked(repo: Path, tmp_path: Path) -> None:
    # With -m, git keeps lines that start with "#", so the hook scans them too.
    _stage(repo, "ok.py", b"y = 2\n")
    done = _commit(repo, message=f"# {GEMINI}")
    assert done.returncode != 0, done.stdout
    assert "commit message:1: gemini/google key" in done.stdout


@pytest.mark.parametrize("message", ["venv gone", "see " + GEMINI])
def test_push_shows_why_detect_secrets_could_not_run(repo: Path, remote: Path, message: str) -> None:
    (repo / "dev.py").write_text(f"import sys\nsys.stderr.write({message!r} + '\\n')\nsys.exit(1)\n")
    _stage(repo, "ok.py", b"x = 1\n")
    assert _commit(repo, hooks=False).returncode == 0
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert GEMINI not in done.stdout
    if message == "venv gone":
        assert "detect-secrets: exit 1: venv gone" in done.stdout
    else:
        assert "[line withheld: it looks like a key]" in done.stdout


def test_pre_push_needs_git_2_40(repo: Path, remote: Path, tmp_path: Path) -> None:
    shim = tmp_path.parent / f"{tmp_path.name}-shim"
    shim.mkdir()
    (shim / "git").write_text(
        f'#!/bin/sh\nif [ "$1" = version ]; then echo "git version 2.39.1"; exit 0; fi\nexec {shutil.which("git")} "$@"\n'
    )
    (shim / "git").chmod(0o755)
    # git puts its own folder ahead of PATH for hooks, so run the pre-push check directly.
    env = {**_env(), "PATH": str(shim) + os.pathsep + _env()["PATH"]}
    line = f"refs/heads/main {_head(repo)} refs/heads/main {'0' * 40}\n"
    done = subprocess.run(
        [sys.executable, "-I", ".githooks/check_staged.py", "--push"],
        cwd=repo, env=env, input=line, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    assert done.returncode == 1
    assert "pre-push needs git 2.40+" in done.stdout
    assert "the push is blocked" in done.stdout


def test_evil_merge_is_blocked_on_push(repo: Path, remote: Path) -> None:
    # A merge that adds content found in neither parent.
    assert _git(repo, "checkout", "-q", "-b", "side").returncode == 0
    _stage(repo, "side.py", b"s = 1\n")
    assert _commit(repo).returncode == 0
    assert _git(repo, "checkout", "-q", "-").returncode == 0
    _stage(repo, "other.py", b"o = 1\n")
    assert _commit(repo).returncode == 0
    assert _git(repo, *CONFIG, "merge", "--no-commit", "--no-ff", "side").returncode == 0
    _stage(repo, "evil.py", f'client = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    assert len(_git(repo, "rev-list", "--parents", "-n", "1", "HEAD").stdout.split()) == 3
    done = _push(repo)
    assert done.returncode != 0, done.stdout
    assert f"{_head(repo)[:10]} evil.py:1: gemini/google key" in done.stdout


def _commit_tree(repo: Path, files: dict[str, bytes]) -> str:
    # A root commit built with plumbing, which can hold paths `git add` refuses.
    def tree(entries: dict[str, object]) -> str:
        lines = []
        for name, value in entries.items():
            if isinstance(value, dict):
                lines.append(f"040000 tree {tree(value)}\t{name}")
            else:
                (repo / "tmp.blob").write_bytes(value)
                blob = _git(repo, "hash-object", "-w", "tmp.blob").stdout.strip()
                lines.append(f"100644 blob {blob}\t{name}")
        done = subprocess.run(["git", "mktree"], cwd=repo, env=_env(), input="\n".join(lines) + "\n", capture_output=True, text=True)
        return done.stdout.strip()

    root: dict[str, object] = {}
    for path, content in files.items():
        node = root
        *folders, leaf = path.split("/")
        for folder in folders:
            node = node.setdefault(folder, {})
        node[leaf] = content
    sha = _git(repo, *CONFIG, "commit-tree", tree(root), "-m", "plumbing").stdout.strip()
    (repo / "tmp.blob").unlink()
    return sha


@pytest.mark.parametrize("path", [".git/config", ".GIT/config", "sub/.Git/hooks/x"])
def test_push_refuses_paths_with_a_git_component(repo: Path, remote: Path, path: str) -> None:
    baseline = (repo / ".secrets.baseline").read_bytes()
    commit = _commit_tree(repo, {path: b"[core]\n", "ok.txt": b"hi\n", ".secrets.baseline": baseline})
    assert _git(repo, "update-ref", "refs/heads/odd", commit).returncode == 0
    done = _push(repo, "refs/heads/odd:refs/heads/odd")
    assert done.returncode != 0, done.stdout
    assert f"{commit[:10]} {path}: path with a .git component" in done.stdout


def test_commits_mode_does_not_talk_about_a_blocked_commit(repo: Path) -> None:
    done = subprocess.run(
        [sys.executable, "-I", ".githooks/check_staged.py", "--commits", "0" * 40],
        cwd=repo, env=_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )  # fmt: skip
    assert done.returncode == 1
    assert "The check failed" in done.stdout
    assert "blocked" not in done.stdout


def test_text_after_a_literal_scissors_line_is_blocked(repo: Path) -> None:
    # With -m, git keeps everything after a typed scissors line in the message.
    scissors = "# " + "-" * 24 + " >8 " + "-" * 24
    _stage(repo, "ok.py", b"y = 2\n")
    before = _head(repo)
    done = _commit(repo, message=f"fix\n{scissors}\nkey {GEMINI}")
    assert done.returncode != 0, done.stdout
    assert "commit message:3: gemini/google key" in done.stdout
    assert GEMINI not in done.stdout
    assert _head(repo) == before


def test_commit_v_may_remove_a_key_with_a_custom_comment_character(repo: Path, tmp_path: Path) -> None:
    _stage(repo, "app.py", f'x = 1\nclient = f(api_key="{GEMINI}")\n'.encode())
    assert _commit(repo, hooks=False).returncode == 0
    _stage(repo, "app.py", b"x = 1\n")
    done = _commit_verbose(repo, tmp_path, "remove the key", "-c", "core.commentChar=;")
    assert done.returncode == 0, done.stdout
