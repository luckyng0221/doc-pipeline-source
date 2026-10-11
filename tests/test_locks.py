from __future__ import annotations

import re
import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

ROOT = Path(__file__).parents[1]
PIN = re.compile(r"^([A-Za-z0-9._-]+)==(\S+)(?: ; (.*?))?(?: \\)?$")


def _pins(path: Path | str) -> set[tuple[str, str, str | None]]:
    """(name, version, marker) for each pinned line in a lock file."""
    lines = (ROOT / path).read_text().splitlines()
    return {(canonicalize_name(m[1]), m[2], m[3]) for m in map(PIN.match, lines) if m}


def _hashes(path: Path | str) -> dict[tuple[str, str, str | None], frozenset[str]]:
    """The sha256 hashes of each pinned block in a lock file."""
    hashes: dict[tuple[str, str, str | None], frozenset[str]] = {}
    for block in re.split(r"\n(?=\S)", (ROOT / path).read_text()):
        pin = PIN.match(block.splitlines()[0])
        if pin:
            hashes[(canonicalize_name(pin[1]), pin[2], pin[3])] = frozenset(re.findall(r"--hash=sha256:([0-9a-f]{64})", block))
    return hashes


def _requirements() -> list[Requirement]:
    """Core dependencies plus the app and dev extras from pyproject.toml."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = project["optional-dependencies"]
    return [Requirement(r) for r in [*project["dependencies"], *extras["app"], *extras["dev"]]]


def _unmet(requirements: list[Requirement], pins: set[tuple[str, str, str | None]]) -> list[str]:
    unmet = []
    for req in requirements:
        if req.marker and not req.marker.evaluate():
            continue
        versions = [v for n, v, _ in pins if n == canonicalize_name(req.name)]
        if not any(req.specifier.contains(Version(v), prereleases=True) for v in versions):
            unmet.append(f"{req}: pinned {sorted(versions) or 'nothing'}")
    return unmet


def test_dev_lock_matches_the_deployed_lock() -> None:
    app, dev = _pins("requirements.txt"), _pins("requirements-dev.txt")
    assert app, "no pins found in requirements.txt"
    wrong = sorted(f"{n}=={v} ; {m}" for n, v, m in app - dev)
    assert not wrong, "requirements.txt has pins missing from requirements-dev.txt (regenerate both locks):\n" + "\n".join(wrong)


def test_dev_lock_meets_pyproject() -> None:
    unmet = _unmet(_requirements(), _pins("requirements-dev.txt"))
    assert not unmet, "requirements-dev.txt doesn't satisfy pyproject.toml (regenerate the locks):\n" + "\n".join(unmet)


def test_dev_lock_hashes_match_the_deployed_lock() -> None:
    app, dev = _hashes("requirements.txt"), _hashes("requirements-dev.txt")
    assert app, "no pins found in requirements.txt"
    for key, hashes in app.items():
        assert hashes, f"{key[0]}=={key[1]} has no hashes in requirements.txt"
    wrong = sorted(f"{n}=={v}" for (n, v, m), hashes in app.items() if dev.get((n, v, m)) != hashes)
    assert not wrong, "these hashes differ between the two locks (regenerate both):\n" + "\n".join(wrong)
