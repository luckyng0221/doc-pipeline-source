"""Checks run by the hooks in .githooks/, on staged content or on commits about to be pushed.

Blocks documents (by name, by .gitignore, and by file signature), files that
detect-secrets would skip (UTF-16, other non-text), Git LFS and filtered files,
changed detect-secrets settings in .secrets.baseline, and this project's own API
key formats that detect-secrets misses. Standard library only, so it runs without
the venv. Run from the repo root with `python3 -I`:

    --staged            the index, for pre-commit and pre-merge-commit (default)
    --message FILE      a commit message, for commit-msg
    --commits SHA...    the files and messages of those commits, in full
    --push              like --commits for what pre-push reads on stdin (the
                        commits not yet on any remote-tracking ref), plus a
                        detect-secrets scan of their files

It reports `path:line: <pattern name>` and never the matched text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DOC_EXTENSIONS = (
    "pdf", "xls", "xlsx", "xlsm", "csv", "doc", "docx", "zip",
    "tif", "tiff", "jpg", "jpeg", "png", "heic", "webp",
    "tar", "tgz", "gz", "xz", "bz2", "zst", "7z", "rar", "gif", "bmp", "avif",
    "rtf", "odt", "ods", "odp", "eml", "msg", "ps",
)  # fmt: skip
# (offset, bytes, kind)
MAGIC = (
    (0, b"%PDF", "PDF"),
    (0, b"PK\x03\x04", "zip or Office file"),
    (0, b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", "old Office file"),
    (0, b"\xff\xd8\xff", "JPEG"),
    (0, b"\x89PNG\r\n\x1a\n", "PNG"),
    (0, b"II*\x00", "TIFF"),
    (0, b"MM\x00*", "TIFF"),
    (0, b"GIF87a", "GIF"),
    (0, b"GIF89a", "GIF"),
    (0, b"\x1f\x8b", "gzip"),
    (0, b"7z\xbc\xaf\x27\x1c", "7z"),
    (0, b"Rar!\x1a\x07", "RAR"),
    (0, b"\xfd7zXZ\x00", "xz"),
    (0, b"\x28\xb5\x2f\xfd", "zstd"),
    (0, b"\x00\x00\x00\x0cjP  ", "JPEG 2000"),
    (0, b"{\\rtf", "RTF"),
    (0, b"%!PS", "PostScript"),
    (257, b"ustar", "tar"),
)
# HEIC and AVIF: "ftyp" at offset 4, then one of these brands.
FTYP_BRANDS = (b"heic", b"heix", b"mif1", b"msf1", b"avif")
# DIB header sizes that follow a BMP file header.
BMP_HEADERS = (12, 40, 52, 56, 64, 108, 124)
HEAD = 1024
UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
KEY_PATTERNS = (
    ("gemini/google key", re.compile(rb"AIza[0-9A-Za-z_-]{35}")),
    ("openai key", re.compile(rb"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]{20,}")),
    # No lookbehind, so a key after "\n", "%3D" or a tab is caught too.
    ("openai key", re.compile(rb"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}")),
    ("glm key", re.compile(rb"(?<![0-9a-f])[0-9a-f]{32}\.[A-Za-z0-9]{16}")),
    # Split so that this file doesn't match its own pattern.
    ("google-oauth-secret", re.compile(rb"GOCSPX" rb"-")),
    ("private key", re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("aws access key id", re.compile(rb"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])")),
    (
        "microsoft client secret",
        re.compile(rb"(?<![A-Za-z0-9_~.-])[A-Za-z0-9_~.]{3}\dQ~[A-Za-z0-9_~.-]{31,34}(?![A-Za-z0-9_~.-])"),
    ),
)
# Filters that make detect-secrets skip a whole file; each can be used to hide a secret.
SKIP_FILTERS = (
    "detect_secrets.filters.heuristic.is_non_text_file",
    "detect_secrets.filters.heuristic.is_swagger_file",
    "detect_secrets.filters.heuristic.is_lock_file",
)
HUNK = re.compile(rb"^@@ -\S+ \+(\d+)(?:,\d+)? @@")
BASELINE = ".secrets.baseline"
# The detect-secrets settings of today's .secrets.baseline, copied exactly. A baseline
# that drops one of these plugins or changes its settings, or that adds a filter (such
# as an excluded-files regex) or any file:// plugin, is blocked until the user has
# reviewed it. Changing this pinned set is the user's decision, not an agent's.
# A different baseline version makes detect-secrets run its upgrade code, which can read
# custom_plugin_paths, so the version is pinned too, and only these top-level keys are allowed.
BASELINE_VERSION = "1.5.0"
BASELINE_KEYS = {"version", "plugins_used", "filters_used", "results", "generated_at"}
PINNED_PLUGINS = [
    {"name": "ArtifactoryDetector"},
    {"name": "AWSKeyDetector"},
    {"name": "AzureStorageKeyDetector"},
    {"name": "Base64HighEntropyString", "limit": 4.5},
    {"name": "BasicAuthDetector"},
    {"name": "CloudantDetector"},
    {"name": "DiscordBotTokenDetector"},
    {"name": "GitHubTokenDetector"},
    {"name": "GitLabTokenDetector"},
    {"name": "HexHighEntropyString", "limit": 3.0},
    {"name": "IbmCloudIamDetector"},
    {"name": "IbmCosHmacDetector"},
    {"name": "IPPublicDetector"},
    {"name": "JwtTokenDetector"},
    {"name": "KeywordDetector", "keyword_exclude": ""},
    {"name": "MailchimpDetector"},
    {"name": "NpmDetector"},
    {"name": "OpenAIDetector"},
    {"name": "PrivateKeyDetector"},
    {"name": "PypiTokenDetector"},
    {"name": "SendGridDetector"},
    {"name": "SlackDetector"},
    {"name": "SoftlayerDetector"},
    {"name": "SquareOAuthDetector"},
    {"name": "StripeDetector"},
    {"name": "TelegramBotTokenDetector"},
    {"name": "TwilioKeyDetector"},
]
PINNED_FILTERS = [
    {"path": "detect_secrets.filters.allowlist.is_line_allowlisted"},
    {"path": "detect_secrets.filters.common.is_baseline_file", "filename": ".secrets.baseline"},
    {"path": "detect_secrets.filters.heuristic.is_indirect_reference"},
    {"path": "detect_secrets.filters.heuristic.is_likely_id_string"},
    {"path": "detect_secrets.filters.heuristic.is_lock_file"},
    {"path": "detect_secrets.filters.heuristic.is_not_alphanumeric_string"},
    {"path": "detect_secrets.filters.heuristic.is_potential_uuid"},
    {"path": "detect_secrets.filters.heuristic.is_prefixed_with_dollar_sign"},
    {"path": "detect_secrets.filters.heuristic.is_sequential_string"},
    {"path": "detect_secrets.filters.heuristic.is_swagger_file"},
    {"path": "detect_secrets.filters.heuristic.is_templated_secret"},
    {"path": "detect_secrets.filters.regex.should_exclude_file", "pattern": ["^requirements(-dev)?\\.txt$"]},
]
# These would change how git reads the names passed to it. GIT_DIFF_OPTS would
# override -U0.
DROPPED_ENV = (
    "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS", "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS", "GIT_DIFF_OPTS",
)  # fmt: skip
ROOT = Path(__file__).resolve().parent.parent
SCAN_BATCH = 100


class GitError(Exception):
    pass


def git(
    *args: str, data: bytes | None = None, ok: tuple[int, ...] = (0,), cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:  # fmt: skip
    if env is None:
        env = {k: v for k, v in os.environ.items() if k not in DROPPED_ENV}
    done = subprocess.run(["git", *args], input=data, capture_output=True, env=env, cwd=cwd)
    if done.returncode not in ok:
        command = next(a for a in args if not a.startswith("-"))
        raise GitError(f"git {command} failed (exit {done.returncode})")
    return done


def other_repo_env() -> dict[str, str]:
    # For a scratch repo: nothing that points git at the real one.
    names = git("rev-parse", "--local-env-vars").stdout.decode().split()
    drop = {*names, *DROPPED_ENV, "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_CEILING_DIRECTORIES"}
    keep = {
        k: v for k, v in os.environ.items()
        if k not in drop and not k.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
    }  # fmt: skip
    return keep


def show(name: str) -> str:
    # Names can hold newlines or other odd characters; keep the report on one line.
    return name if name.isprintable() else ascii(name)


def by_name(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith("." + ext) for ext in DOC_EXTENSIONS)


def by_signature(data: bytes) -> str | None:
    head = data[:HEAD]
    for offset, magic, kind in MAGIC:
        if head[offset : offset + len(magic)] == magic:
            return kind
    if b"%PDF" b"-" in head:
        return "PDF"
    if head[4:8] == b"ftyp" and head[8:12] in FTYP_BRANDS:
        return "HEIC or AVIF"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    if head[:3] == b"BZh" and head[3:4].isdigit() and head[3:4] != b"0":
        return "bzip2"
    # "BM" alone starts ordinary words, so a BMP must also look like one.
    if head[:2] == b"BM" and head[6:10] == b"\0\0\0\0" and int.from_bytes(head[14:18], "little") in BMP_HEADERS:
        return "BMP"
    return None


def key_hits(lines: list[tuple[int, bytes]]) -> list[tuple[int, str]]:
    # (line number, pattern name) for each line that looks like a key.
    # Unlike detect-secrets, the allowlist pragma does not apply to these patterns.
    hits = []
    for number, line in lines:
        # One report per name and line, even if two patterns of that name match.
        names = dict.fromkeys(name for name, pattern in KEY_PATTERNS if pattern.search(line))
        hits.extend((number, name) for name in names)
    return hits


def numbered(data: bytes) -> list[tuple[int, bytes]]:
    return list(enumerate(data.split(b"\n"), 1))


def parse_raw(out: bytes) -> list[tuple[str, str, str]]:
    # (path, mode, blob id) from `git diff --raw -z --no-abbrev --no-renames` output.
    parts = out.split(b"\0")
    files = []
    for i in range(0, len(parts) - 1, 2):
        # ":<old mode> <new mode> <old id> <new id> <status>", then the path.
        fields = parts[i].lstrip(b":").split(b" ")
        if len(fields) != 5:
            raise GitError("unexpected git diff output")
        files.append((os.fsdecode(parts[i + 1]), fields[1].decode(), fields[3].decode()))
    return files


def staged_files() -> list[tuple[str, str, str]]:
    # Added, copied, modified and type-changed files in the index.
    out = git("diff", "--cached", "--raw", "-z", "--no-abbrev", "--no-renames", "--no-ext-diff", "--diff-filter=ACMRT").stdout
    return parse_raw(out)


def commit_files(commit: str) -> list[tuple[str, str, str]]:
    # -m so that a merge is compared with each parent; its own changes are scanned too.
    out = git(
        "diff-tree", "-r", "--root", "--raw", "-z", "--no-abbrev", "--no-renames", "--no-ext-diff", "--no-commit-id",
        "-m", "--diff-filter=ACMRT", commit,
    ).stdout  # fmt: skip
    return list(dict.fromkeys(parse_raw(out)))


def unscannable(names: list[str]) -> list[str]:
    # Paths marked assume-unchanged (lower-case tag) or skip-worktree ("S"). The
    # secret scan reads the working tree, which git then doesn't compare to the index.
    wanted = set(names)
    out = git("ls-files", "-v", "-z").stdout
    flagged = []
    for entry in out.split(b"\0"):
        tag, path = entry[:1], os.fsdecode(entry[2:])
        if path in wanted and (tag.islower() or tag == b"S"):
            flagged.append(path)
    return flagged


def blobs(ids: list[str]) -> dict[str, bytes]:
    wanted = sorted(set(ids))
    if not wanted:
        return {}
    out = git("cat-file", "--batch", data="".join(f"{i}\n" for i in wanted).encode()).stdout
    result, pos = {}, 0
    for blob_id in wanted:
        end = out.index(b"\n", pos)
        header = out[pos:end].split(b" ")
        if len(header) != 3 or header[0].decode() != blob_id:
            raise GitError(f"git cat-file could not read {blob_id[:12]}")
        size = int(header[2])
        result[blob_id] = out[end + 1 : end + 1 + size]
        pos = end + 1 + size + 1
    return result


def ignored(names: list[str]) -> list[str]:
    if not names:
        return []
    # The "./" stops a name like ":(x).bin" from being read as pathspec magic.
    data = b"".join(b"./" + os.fsencode(n) + b"\0" for n in names)
    done = git("check-ignore", "--no-index", "-z", "--stdin", data=data, ok=(0, 1))
    if done.returncode == 1:
        return []
    return [os.fsdecode(p)[2:] for p in done.stdout.split(b"\0") if p]


def filtered(names: list[str], *where: str) -> list[tuple[str, str]]:
    # (path, attribute) for paths with a `filter` (such as lfs) or `working-tree-encoding`
    # attribute: the blob is then not what the working tree holds, so the scan misses it.
    # `where` is one check-attr option per run ("" is the working tree's attributes).
    if not names:
        return []
    data = b"".join(os.fsencode(n) + b"\0" for n in names)
    found = []
    for option in where or ("",):
        extra = [option] if option else []
        out = git("check-attr", *extra, "-z", "--stdin", "filter", "working-tree-encoding", data=data).stdout
        parts = out.split(b"\0")
        for i in range(0, len(parts) - 2, 3):
            path, attribute, value = os.fsdecode(parts[i]), parts[i + 1].decode(), parts[i + 2]
            if value not in (b"unspecified", b"unset"):
                found.append((path, attribute))
    return list(dict.fromkeys(found))


def content_problem(mode: str, data: bytes) -> str | None:
    # The first reason this blob must not be committed, if any.
    kind = by_signature(data)
    if kind:
        return f"looks like a {kind} file; documents must not be committed"
    if data.startswith(UTF16_BOMS):
        return "UTF-16 file; detect-secrets can't scan it"
    if data.startswith(LFS_POINTER):
        return "Git LFS pointer; the real content is stored elsewhere and isn't scanned"
    # Symlinks and submodules are not files with content of their own.
    if mode.startswith("100"):
        if b"\0" in data:
            return "not UTF-8 text; detect-secrets would skip it"
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            return "not UTF-8 text; detect-secrets would skip it"
    return None


def baseline_problem(data: bytes | None) -> str | None:
    # Why this .secrets.baseline must be reviewed before it is trusted, if it must.
    if data is None:
        return "missing; the secret scan has no baseline to compare with"
    try:
        baseline = json.loads(data)
        plugins, filters = baseline["plugins_used"], baseline["filters_used"]
        entries = [*plugins, *filters]
        if not isinstance(plugins, list) or not isinstance(filters, list) or not all(isinstance(e, dict) for e in entries):
            raise ValueError
        if not all(isinstance(p.get("name"), str) for p in plugins) or not all(isinstance(f.get("path"), str) for f in filters):
            raise ValueError
    except (ValueError, KeyError, TypeError):
        return "unreadable, or it has no plugins_used and filters_used"
    names = [p.get("name") for p in plugins]
    paths = [f.get("path") for f in filters]
    # A repeated plugin or filter can override the pinned one, so each pinned plugin must
    # appear exactly once; the rest may only be stricter (extra built-in plugins, fewer filters).
    changed = (
        not set(baseline) <= BASELINE_KEYS
        or baseline.get("version") != BASELINE_VERSION
        or len(set(names)) != len(names)
        or len(set(paths)) != len(paths)
        or any(p not in plugins for p in PINNED_PLUGINS)
        or any(f not in PINNED_FILTERS for f in filters)
        or any("path" in p for p in plugins)
        or any("file://" in json.dumps(e) for e in entries)
    )
    if changed:
        return "the detect-secrets settings in the baseline changed (detectors or filters); the user must review it"
    return None


def file_problems(
    files: list[tuple[str, str, str]], contents: dict[str, bytes], prefix: str, *attr_options: str,
) -> list[str]:  # fmt: skip
    # Name, .gitignore, attribute and content checks, shared by every mode.
    problems = []

    def add(name: str, text: str) -> None:
        problems.append(f"{prefix}{show(name)}: {text}")

    names = [name for name, _, _ in files]
    named = {name for name in names if by_name(name)}
    for name in names:
        if name in named:
            add(name, "document by name; documents must not be committed")
    for name in ignored(names):
        add(name, "ignored by .gitignore; ignored files must not be committed")
    for name, attribute in filtered(names, *attr_options):
        add(name, f"has a {attribute} attribute; the blob isn't what the working tree holds, so it isn't scanned")
    for name, mode, blob_id in files:
        if mode == "160000":  # a submodule has no blob here
            continue
        # A document's name already said so; don't add "not UTF-8" on top.
        reason = content_problem(mode, contents[blob_id])
        if reason and not (name in named and "UTF-8" in reason):
            add(name, reason)
    return problems


def added_lines(names: list[str]) -> dict[str, list[tuple[int, bytes]]]:
    # Added lines per file, with new-file line numbers. Files appear in the diff
    # in the same order as in the raw list, which is how they are named.
    # No textconv (it can rewrite a key), no context, and submodules as plain hashes.
    out = git(
        "diff", "--cached", "-U0", "--inter-hunk-context=0", "--no-color", "--text", "--no-ext-diff",
        "--no-textconv", "--submodule=short", "--no-renames", "--diff-filter=ACMRT",
    ).stdout  # fmt: skip
    result: dict[str, list[tuple[int, bytes]]] = {n: [] for n in names}
    index, number, in_hunk = -1, 0, False
    for line in out.split(b"\n"):
        if line.startswith(b"diff --git "):
            index, in_hunk = index + 1, False
            continue
        if not in_hunk and line.startswith(b"deleted file mode "):
            # A type change (file to symlink and back) is a deletion followed by an
            # addition of the same path, which the raw list shows as one entry.
            index -= 1
            continue
        match = HUNK.match(line)
        if match:
            number, in_hunk = int(match[1]), True
        elif in_hunk and line[:1] in (b"+", b" "):
            if index >= len(names):
                raise GitError("git diff has more files than git diff --raw")
            if line.startswith(b"+"):
                result[names[index]].append((number, line[1:]))
            number += 1
    if index + 1 != len(names):
        raise GitError("git diff and git diff --raw list different files")
    return result


def message_problems(message: bytes, label: str) -> list[str]:
    return [f"{label}:{number}: {pattern}" for number, pattern in key_hits(numbered(message))]


def check_staged() -> list[str]:
    files = staged_files()
    names = [f[0] for f in files]
    contents = blobs([blob_id for _, mode, blob_id in files if mode != "160000"])
    # The working tree's attributes decide what `git add` did; the index's decide the rest.
    problems = file_problems(files, contents, "", "", "--cached")
    for name in unscannable(names):
        problems.append(
            f"{show(name)}: marked assume-unchanged or skip-worktree (core.ignoreStat=true marks every file this way); "
            "the scan would read the working tree, not what is staged"
        )
    for name, lines in added_lines(names).items():
        for number, pattern in key_hits(lines):
            problems.append(f"{show(name)}:{number}: {pattern}")
    done = git("cat-file", "blob", f":{BASELINE}", ok=(0, 128))
    reason = baseline_problem(done.stdout if done.returncode == 0 else None)
    if reason:
        problems.append(f"{BASELINE}: {reason}")
    return problems


MIN_GIT = (2, 40)


def commit_message(commit: str) -> bytes:
    raw = git("cat-file", "commit", commit).stdout
    return raw.partition(b"\n\n")[2]


AUTO_COMMENT_CHARS = "#;@!$%^&|:"


def comment_prefixes() -> list[bytes]:
    # The comment string git puts before the scissors line: core.commentString or
    # core.commentChar, "#" by default, and any of git's candidates for "auto".
    values = []
    for key in ("core.commentString", "core.commentChar"):
        done = git("config", "--get", key, ok=(0, 1))
        if done.returncode == 0 and done.stdout.strip():
            values.append(done.stdout.decode().strip())
    if "auto" in values:
        values += list(AUTO_COMMENT_CHARS)
    return [v.encode() for v in values if v != "auto"] or [b"#"]


def editor_message(raw: bytes, prefixes: list[bytes] | None = None) -> bytes:
    # The message as git would keep it. `git commit -v` puts the staged diff after a
    # scissors line, and its removed lines ("-...") can hold a key being taken out, so
    # inside that diff only those lines are left out (as blank lines, to keep line numbers).
    # With -m or -F git keeps text after a literal scissors line, so the rest is scanned.
    # Lines that start with "#" are scanned too: with -m, git keeps them in the message.
    scissors = [p + b" " + b"-" * 24 + b" >8 " + b"-" * 24 for p in (prefixes or [b"#"])]
    lines = raw.split(b"\n")
    after_scissors = in_hunk = False
    for number, line in enumerate(lines):
        if not after_scissors:
            after_scissors = line.rstrip() in scissors
        elif line.startswith(b"@@ "):
            in_hunk = True
        elif line.startswith(b"diff --git "):
            in_hunk = False
        elif in_hunk and line.startswith(b"-"):
            lines[number] = b""
    return b"\n".join(lines)


def require_git_version() -> None:
    # check-attr --source reads the attributes a pushed commit had.
    out = git("version").stdout.decode()
    match = re.search(r"(\d+)\.(\d+)", out)
    if not match or (int(match[1]), int(match[2])) < MIN_GIT:
        raise GitError("pre-push needs git 2.40+ (check-attr --source); this is " + out.strip())


def safe_tail(text: str) -> str:
    # The last lines of dev.py's or Python's own error output. It holds no scanned content,
    # but a line that looks like a key is withheld anyway.
    lines = []
    for line in text.strip().splitlines()[-8:]:
        line = line[:300]
        lines.append("[line withheld: it looks like a key]" if key_hits([(1, line.encode())]) else line)
    return " | ".join(lines)


def scan_with_detect_secrets(commit: str, files: list[tuple[str, str, str]], contents: dict[str, bytes], baseline: bytes) -> list[str]:
    # Writes the commit's regular files to a scratch repo and runs detect-secrets there,
    # against a copy of that commit's baseline. Exit 1 blocks; exit 3 is fine only if
    # the copy changed (on Windows an abort also exits 3).
    wanted = [(n, b) for n, m, b in files if m.startswith("100") and n != BASELINE]
    if not wanted:
        return []
    env = other_repo_env()
    env["DOC_PIPELINE_NO_SETUP"] = "1"
    short = commit[:10]
    problems = []
    with tempfile.TemporaryDirectory() as tree_dir, tempfile.TemporaryDirectory() as base_dir:
        tree = Path(tree_dir).resolve()
        git("init", "-q", cwd=str(tree), env=env)
        names = []
        for name, blob_id in wanted:
            if any(part.lower() == ".git" for part in name.split("/")):
                # Writing it would put files inside the scratch repo's own .git folder.
                problems.append(f"{short} {show(name)}: path with a .git component; it can't be scanned")
                continue
            target = (tree / name).resolve()
            if tree not in target.parents:
                raise GitError(f"unsafe path in {short}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contents[blob_id])
            names.append(name)
        copy = Path(base_dir) / "baseline.json"
        copy.write_bytes(baseline)
        for start in range(0, len(names), SCAN_BATCH):
            before = hashlib.sha256(copy.read_bytes()).digest()
            done = subprocess.run(
                [
                    sys.executable, "-I", str(ROOT / "dev.py"), "detect_secrets.pre_commit_hook", "--no-verify",
                    *[a for f in SKIP_FILTERS for a in ("--disable-filter", f)],
                    "--baseline", str(copy), "--", *names[start : start + SCAN_BATCH],
                ],
                cwd=tree, env=env, capture_output=True,
            )  # fmt: skip
            changed = hashlib.sha256(copy.read_bytes()).digest() != before
            if done.returncode == 0 or (done.returncode == 3 and changed):
                continue
            # detect-secrets names the type and place of each finding, never the value.
            text = done.stdout.decode("utf-8", "replace") + done.stderr.decode("utf-8", "replace")
            found = re.findall(r"^Location:\s+(.*)$", text, re.MULTILINE)
            if found:
                detail = ", ".join(found)
            else:
                # No finding, so this is an error such as a missing venv: show what it said.
                detail = f"exit {done.returncode}: " + safe_tail(done.stderr.decode("utf-8", "replace"))
            problems.append(f"{short} detect-secrets: {detail}")
    return problems


def check_commits(commits: dict[str, str], detect_secrets: bool) -> tuple[list[str], list[str]]:
    # `commits` maps each commit to the tip it is pushed under. A commit without a
    # .secrets.baseline is scanned against its tip's baseline (a warning), if that is usable.
    problems, warnings = [], []
    tip_baselines: dict[str, bytes | None] = {}

    def tip_baseline(tip: str) -> bytes | None:
        if tip not in tip_baselines:
            done = git("cat-file", "blob", f"{tip}:{BASELINE}", ok=(0, 128))
            data = done.stdout if done.returncode == 0 else None
            tip_baselines[tip] = data if data is not None and baseline_problem(data) is None else None
        return tip_baselines[tip]

    for commit, tip in commits.items():
        short = commit[:10]
        prefix = f"{short} "
        files = commit_files(commit)
        contents = blobs([blob_id for _, mode, blob_id in files if mode != "160000"])
        problems += file_problems(files, contents, prefix, f"--source={commit}")
        for name, mode, blob_id in files:
            if mode != "160000":
                for number, pattern in key_hits(numbered(contents[blob_id])):
                    problems.append(f"{prefix}{show(name)}:{number}: {pattern}")
        problems += [f"{prefix}{p}" for p in message_problems(commit_message(commit), "commit message")]
        done = git("cat-file", "blob", f"{commit}:{BASELINE}", ok=(0, 128))
        baseline = done.stdout if done.returncode == 0 else None
        if baseline is not None:
            reason = baseline_problem(baseline)
            if reason:
                problems.append(f"{prefix}{BASELINE}: {reason}")
                continue
        else:
            baseline = tip_baseline(tip)
            if baseline is None:
                problems.append(f"{prefix}{BASELINE}: missing here, and the baseline of {tip[:10]} is missing or not usable")
                continue
            warnings.append(f"{prefix}{BASELINE}: none in this commit; checked with the baseline of {tip[:10]}")
        if detect_secrets:
            problems += scan_with_detect_secrets(commit, files, contents, baseline)
    return problems, warnings


def peel(sha: str) -> tuple[str | None, list[str]]:
    # The commit a pushed object leads to, with problems found in tag messages on the way.
    # A tag of a tree or blob has no commits to check, so it is refused.
    problems = []
    obj = sha
    for _ in range(20):
        kind = git("cat-file", "-t", obj).stdout.decode().strip()
        if kind == "commit":
            return obj, problems
        if kind != "tag":
            problems.append(f"{sha[:10]} is (through its tags) a {kind}; this project doesn't push a tag of a {kind}")
            return None, problems
        header, _, message = git("cat-file", "tag", obj).stdout.partition(b"\n\n")
        problems += [f"{obj[:10]} {p}" for p in message_problems(message, "tag message")]
        target = re.search(rb"^object ([0-9a-f]{40,64})$", header, re.MULTILINE)
        if not target:
            raise GitError(f"unreadable tag {obj[:10]}")
        obj = target[1].decode()
    raise GitError("too many nested tags")


def pushed_commits(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    # The commits in a push that are not on any remote-tracking ref yet, oldest first, each
    # with the tip it is pushed under. With no remote-tracking refs at all, that is every
    # commit reachable from what is pushed.
    commits: dict[str, str] = {}
    problems = []
    for line in lines:
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 4:
            raise GitError("unexpected pre-push input")
        local_sha = fields[1]
        if set(local_sha) == {"0"}:  # deleting a ref pushes no commits
            continue
        tip, found = peel(local_sha)
        problems += found
        if tip is None:
            continue
        out = git("rev-list", "--reverse", tip, "--not", "--remotes").stdout.decode()
        for commit in out.split():
            commits.setdefault(commit, tip)
    return commits, problems


def run(args: argparse.Namespace) -> tuple[str, list[str], list[str]]:
    # (what is checked, problems, warnings)
    if args.message:
        message = editor_message(Path(args.message).read_bytes(), comment_prefixes())
        return "commit", message_problems(message, "commit message"), []
    if args.push:
        require_git_version()
        commits, problems = pushed_commits(sys.stdin.read().splitlines())
        more, warnings = check_commits(commits, detect_secrets=True)
        return "push", problems + more, warnings
    if args.commits is not None:
        require_git_version()
        problems, warnings = check_commits({c: "HEAD" for c in args.commits}, detect_secrets=False)
        return "", problems, warnings
    return "commit", check_staged(), []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--staged", action="store_true")
    group.add_argument("--message", metavar="FILE")
    group.add_argument("--commits", nargs="*", metavar="SHA")
    group.add_argument("--push", action="store_true")
    args = parser.parse_args()
    try:
        what, problems, warnings = run(args)
    except (GitError, OSError, ValueError) as error:
        blocked = "the push is blocked" if args.push else "the commit is blocked" if args.commits is None else "check not done"
        print(f"The check failed ({error}); {blocked}.")
        return 1
    for warning in warnings:
        print(f"Warning: {warning}")
    if problems:
        print(f"The {what} is blocked:" if what else "Problems found:")
        for problem in problems:
            print(f"  {problem}")
        print("Remove these. Nothing printed here shows a key's value.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
