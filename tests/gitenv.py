"""Keeps the tests' git commands away from the repository the tests run in.

Without this, pytest started from a git hook (or a shell with GIT_DIR set) would
point every git command in a test at the real repository.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import MutableMapping

import pytest

ALWAYS = (
    "GIT_INDEX_FILE", "GIT_WORK_TREE", "GIT_DIR", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT", "GIT_LITERAL_PATHSPECS", "GIT_GLOB_PATHSPECS",
    "GIT_NOGLOB_PATHSPECS", "GIT_ICASE_PATHSPECS", "GIT_DIFF_OPTS",
)  # fmt: skip


def scrub(env: MutableMapping[str, str], ceiling: str) -> None:
    """Remove git's repository-pointing variables from env, and fence git in at `ceiling`."""
    names = set(ALWAYS)
    try:
        listed = subprocess.run(["git", "rev-parse", "--local-env-vars"], capture_output=True, text=True)
        names.update(listed.stdout.split())
    except OSError:
        pass  # no git, so there is nothing to protect
    for key in list(env):
        if key in names or key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")):
            del env[key]
    env["GIT_CEILING_DIRECTORIES"] = ceiling
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull


def scrub_process(monkeypatch: pytest.MonkeyPatch, ceiling: str) -> None:
    """The same, for this test process and the commands it starts."""
    scrubbed = dict(os.environ)
    scrub(scrubbed, ceiling)
    for key in set(os.environ) - set(scrubbed):
        monkeypatch.delenv(key)
    for key, value in scrubbed.items():
        monkeypatch.setenv(key, value)
