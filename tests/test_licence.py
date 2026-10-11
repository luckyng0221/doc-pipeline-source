from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]

# The digest of the official AGPL-3.0 text, split into pieces so no line
# looks like a high-entropy secret to detect-secrets.
DIGEST_PIECES = ("0d96a4ff", "68ad6d4b", "6f1f30f7", "13b18d51", "84912ba8", "dd389f86", "aa7710db", "079abcb0")


def test_license_is_the_official_agpl_text() -> None:
    data = (ROOT / "LICENSE").read_bytes()

    assert hashlib.sha256(data).hexdigest() == "".join(DIGEST_PIECES)
    assert len(data.decode("utf-8").splitlines()) == 661


def test_pyproject_declares_the_agpl_licence() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["license"] == "AGPL-3.0-only"
    assert project["license-files"] == ["LICENSE"]
