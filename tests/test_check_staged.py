"""Parts of .githooks/check_staged.py that are easier to test directly."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("check_staged", ROOT / ".githooks" / "check_staged.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _baseline(**changes: object) -> bytes:
    baseline = {
        "version": "1.5.0",
        "plugins_used": copy.deepcopy(cs.PINNED_PLUGINS),
        "filters_used": copy.deepcopy(cs.PINNED_FILTERS),
        "results": {},
    }
    baseline.update(changes)
    return json.dumps(baseline).encode()


def test_the_pinned_settings_are_what_the_project_baseline_uses() -> None:
    baseline = json.loads((ROOT / ".secrets.baseline").read_text())
    assert baseline["plugins_used"] == cs.PINNED_PLUGINS
    assert baseline["filters_used"] == cs.PINNED_FILTERS
    assert cs.baseline_problem((ROOT / ".secrets.baseline").read_bytes()) is None


def test_the_pinned_baseline_passes() -> None:
    assert cs.baseline_problem(_baseline()) is None


def test_stricter_baselines_pass() -> None:
    plugins = [*copy.deepcopy(cs.PINNED_PLUGINS), {"name": "SomeNewBuiltInDetector"}]
    assert cs.baseline_problem(_baseline(plugins_used=plugins, filters_used=cs.PINNED_FILTERS[1:])) is None


def _without(entries: list[dict], index: int) -> list[dict]:
    return entries[:index] + entries[index + 1 :]


@pytest.mark.parametrize(
    "baseline",
    [
        _baseline(plugins_used=_without(cs.PINNED_PLUGINS, 3)),
        _baseline(plugins_used=[{**p, "limit": 9.0} if "limit" in p else p for p in cs.PINNED_PLUGINS]),
        _baseline(plugins_used=[*cs.PINNED_PLUGINS, {"name": "Mine", "path": "plugins/mine.py"}]),
        _baseline(plugins_used=[*cs.PINNED_PLUGINS, {"name": "Mine", "other": "file:///tmp/mine.py"}]),
        _baseline(filters_used=[*cs.PINNED_FILTERS, {"path": "detect_secrets.filters.regex.should_exclude_file", "pattern": ["^x"]}]),
        _baseline(filters_used=[*cs.PINNED_FILTERS, {"path": "file:///tmp/filter.py"}]),
        _baseline(filters_used=[{**f, "pattern": ["^.*$"]} if "pattern" in f else f for f in cs.PINNED_FILTERS]),
        _baseline(exclude={"files": "^.*$"}),
        # A repeated entry can override the pinned one.
        _baseline(plugins_used=[*cs.PINNED_PLUGINS, {"name": "KeywordDetector", "keyword_exclude": "."}]),
        _baseline(plugins_used=[*cs.PINNED_PLUGINS, {"name": "Base64HighEntropyString", "limit": 8.0}]),
        _baseline(filters_used=[*cs.PINNED_FILTERS, cs.PINNED_FILTERS[2]]),
        _baseline(plugins_used=[*cs.PINNED_PLUGINS, {"name": ["x"]}]),
        # Another version runs detect-secrets' upgrade code; unknown keys are not allowed.
        _baseline(version="1.4.0"),
        _baseline(custom_plugin_paths=["file:///tmp/mine.py"]),
        _baseline(version=None),
        b"not json",
        json.dumps({"results": {}}).encode(),
        json.dumps({"plugins_used": {}, "filters_used": []}).encode(),
        None,
    ],
)
def test_changed_baseline_settings_are_reported(baseline: bytes | None) -> None:
    assert cs.baseline_problem(baseline)


@pytest.mark.parametrize(
    ("head", "kind"),
    [
        (b"BM" + b"\x36\0\0\0" + b"\0\0\0\0" + b"\x36\0\0\0" + b"\x28\0\0\0", "BMP"),
        (b"BMW makes cars and BMX bikes", None),
        (b"BZh9" + b"1AY&SY", "bzip2"),
        (b"BZh0", None),
        (b"BZhx", None),
        (b"a" * 257 + b"ustar", "tar"),
        (b"a" * 258 + b"ustar", None),
        (b"x" * 900 + b"%PDF-1.4", "PDF"),
        (b"x" * 1100 + b"%PDF-1.4", None),
        (b"plain text\n", None),
    ],
)
def test_signatures(head: bytes, kind: str | None) -> None:
    assert cs.by_signature(head) == kind


def test_key_hits_ignore_the_allowlist_pragma_and_report_each_pattern_once() -> None:
    key = "sk-" + "proj-" + "A1b2C3d4E5f6G7h8I9j0" * 2
    pragma = "pragma: " + "allowlist secret"
    hits = cs.key_hits([(1, f'x = "{key}"  # {pragma}'.encode()), (2, b"nothing here")])
    assert hits == [(1, "openai key")]


def test_the_scratch_repo_environment_does_not_point_at_another_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("GIT_DIR", "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_OBJECT_DIRECTORY", "GIT_CONFIG_PARAMETERS",
                 "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "GIT_DIFF_OPTS", "GIT_CEILING_DIRECTORIES"):  # fmt: skip
        monkeypatch.setenv(name, "x")
    monkeypatch.setenv("GIT_AUTHOR_NAME", "keep")
    env = cs.other_repo_env()
    assert not [k for k in env if k.startswith(("GIT_DIR", "GIT_INDEX", "GIT_WORK", "GIT_OBJECT", "GIT_CONFIG_PARAMETERS",
                                                "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY", "GIT_CONFIG_VALUE", "GIT_DIFF", "GIT_CEILING"))]  # fmt: skip
    assert env["GIT_AUTHOR_NAME"] == "keep"


def _verbose_message(comment: str = "#") -> bytes:
    scissors = f"{comment} " + "-" * 24 + " >8 " + "-" * 24
    return (
        f"fix\n\n{comment} Please enter the message\n{scissors}\n{comment} Do not modify or remove the line above.\n"
        "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n context\n-old = 1\n+new = 2\n"
    ).encode()


def test_editor_message_leaves_out_only_the_removed_lines_of_the_verbose_diff() -> None:
    kept = cs.editor_message(_verbose_message()).split(b"\n")
    assert kept[0] == b"fix"
    assert b"-old = 1" not in kept
    assert b"+new = 2" in kept and b" context" in kept and b"--- a/f" in kept
    assert len(kept) == len(_verbose_message().split(b"\n"))  # line numbers stay the same


def test_editor_message_uses_the_configured_comment_string() -> None:
    raw = _verbose_message(";")
    assert b"-old = 1" in cs.editor_message(raw, [b"#"]).split(b"\n")
    assert b"-old = 1" not in cs.editor_message(raw, [b";"]).split(b"\n")


def test_text_after_a_literal_scissors_line_is_kept() -> None:
    scissors = "# " + "-" * 24 + " >8 " + "-" * 24
    raw = f"fix\n{scissors}\nafter it\n-also kept\n".encode()
    assert cs.editor_message(raw) == raw


def test_safe_tail_withholds_key_shaped_lines_and_cuts_long_ones() -> None:
    key = "AI" + "za" + "x" * 35
    text = "\n".join(["one", f"see {key}", "x" * 500])
    shown = cs.safe_tail(text)
    assert key not in shown
    assert "[line withheld" in shown
    assert len(shown) < 400


@pytest.mark.parametrize(("version", "ok"), [("2.39.9", False), ("2.40.0", True), ("2.53.0", True), ("1.9.0", False), ("junk", False)])
def test_git_version_requirement(monkeypatch: pytest.MonkeyPatch, version: str, ok: bool) -> None:
    class Done:
        stdout = f"git version {version}\n".encode()

    monkeypatch.setattr(cs, "git", lambda *args, **kwargs: Done)
    if ok:
        cs.require_git_version()
    else:
        with pytest.raises(cs.GitError, match="git 2.40"):
            cs.require_git_version()
