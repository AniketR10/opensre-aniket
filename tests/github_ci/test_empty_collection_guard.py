"""Guards against CI collecting zero tests under pytest-xdist.

Two failure modes have produced ``N workers [0 items]`` in CI:

1. A mangled ``PYTEST_MARKER_EXPR`` (e.g. boolean ``false``) deselects everything.
   ``tests/conftest.py`` forces exit code 5 in that case.

2. A missing path argument (file/dir deleted but still listed in
   ``.github/workflows/ci.yml``) makes xdist abort collection for the *whole*
   shard — even when other paths are valid. Seen after ``tests/github`` was
   removed while ``cli-runtime`` still referenced it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CI_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PATH_RE = re.compile(r"^(tests/\S+|gateway/tests)$")

# Directories under ``tests/`` that no shard may claim. ``tests/synthetic`` runs
# in synthetic-deterministic.yml and every shard ``--ignore``s it.
_SHARDED_ELSEWHERE = frozenset({"synthetic"})


def _shard_pytest_paths() -> list[tuple[str, str]]:
    """Return ``(shard, path)`` for every path token the CI matrix names."""
    workflow = yaml.safe_load(_CI_WORKFLOW.read_text(encoding="utf-8"))
    return [
        (entry.get("shard", "?"), token)
        for job in workflow.get("jobs", {}).values()
        for entry in ((job.get("strategy") or {}).get("matrix") or {}).get("include") or []
        if entry.get("pytest_paths")
        for token in str(entry["pytest_paths"]).split()
        if _PATH_RE.match(token)
    ]


def test_xdist_empty_marker_exits_no_tests_collected() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-n",
            "2",
            "-q",
            "tests/packaging",
            "-m",
            "false",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "0 items" in result.stdout or "0 items" in result.stderr
    assert result.returncode == 5, (
        f"expected ExitCode.NO_TESTS_COLLECTED (5), got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ci_pytest_paths_exist_in_git_tree() -> None:
    """Every ``matrix.pytest_paths`` entry must exist in the committed tree.

    Local empty leftover dirs (e.g. ``tests/github/`` with only ``__pycache__``)
    hide this; CI checkouts do not have them, and a missing path zeros xdist.
    """
    tracked = set(
        subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "ls-tree", "-r", "--name-only", "HEAD"],
            text=True,
        ).splitlines()
    )

    def _present(path: str) -> bool:
        if path in tracked:
            return True
        prefix = path.rstrip("/") + "/"
        return any(entry.startswith(prefix) for entry in tracked)

    missing = [f"{shard}: {token}" for shard, token in _shard_pytest_paths() if not _present(token)]

    assert not missing, (
        "CI pytest_paths missing from git tree (xdist will collect 0 items):\n"
        + "\n".join(f"  - {item}" for item in missing)
    )


def test_every_test_directory_runs_in_a_shard() -> None:
    """The reverse of the check above: every test directory must reach a shard.

    ``pytest_paths`` is a hand-maintained list, so a new directory runs nowhere
    until someone remembers to add it, and nothing fails when they don't — the
    tests simply never execute. ``tests/bootstrap`` sat unsharded that way until
    #5240, and ``tests/filestorage``, ``tests/surfaces`` and ``tests/quality``
    until #5349, between them 304 tests that no pull request ran.
    """
    claimed = {token for _shard, token in _shard_pytest_paths()}

    # Ancestor-or-self only: cli-runtime names ``tests/core/agent``, which must
    # not be read as covering the rest of ``tests/core``.
    uncovered = sorted(
        f"tests/{path.name}"
        for path in _REPO_ROOT.joinpath("tests").iterdir()
        if path.is_dir()
        and not path.name.startswith("__")
        and path.name not in _SHARDED_ELSEWHERE
        and any(path.rglob("test_*.py"))
        and f"tests/{path.name}" not in claimed
    )

    assert not uncovered, (
        "test directories in no CI shard (their tests never run on a PR):\n"
        + "\n".join(f"  - {item}" for item in uncovered)
        + "\nAdd each to a shard's pytest_paths in .github/workflows/ci.yml."
    )
