"""Tests for the coding-agent runner (orchestration).

The runner is driven here by a ``_FakeBackend`` that is **not** a CLI — no argv, no
subprocess, no exit code. That is deliberate: if the runner still works against it,
the orchestration really is transport-agnostic and a future MCP backend (OpenClaw)
can slot in without touching this module. Backend *resolution* is tested against the
real registry.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

from integrations.git import WorktreeDiff
from tools.coding_agent import run_coding_task, verify_coding_agent
from tools.coding_agent.backends.base import BackendOutcome, BackendProbe

_RESOLVE = "tools.coding_agent.runner.resolve_backend"
_WORKTREE_DIFF = "tools.coding_agent.runner.worktree_diff"


class _FakeBackend:
    """A coding backend with no transport at all — the point of the seam."""

    name = "fake"

    def __init__(
        self,
        *,
        probe: BackendProbe | None = None,
        outcome: BackendOutcome | None = None,
    ) -> None:
        self._probe = probe or BackendProbe(ready=True, detail="ok")
        self._outcome = outcome or BackendOutcome(summary="edited foo.py")
        self.run_kwargs: dict[str, object] = {}

    def detect(self) -> BackendProbe:
        return self._probe

    def run(
        self, task: str, *, workspace: str, model: str | None, timeout_sec: float
    ) -> BackendOutcome:
        self.run_kwargs = {
            "task": task,
            "workspace": workspace,
            "model": model,
            "timeout_sec": timeout_sec,
        }
        return self._outcome


_GIT_ENV = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _git_init(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=_GIT_ENV)


def _git_commit_all(path: Path) -> None:
    """Commit everything, so later edits show up as working-tree changes."""
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, env=_GIT_ENV)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=path,
        check=True,
        env=_GIT_ENV,
    )


class _WritingBackend:
    """A backend that actually edits the tree, for tests that use real git."""

    name = "writer"

    def __init__(self, writes: dict[str, str]) -> None:
        self._writes = writes

    def detect(self) -> BackendProbe:
        return BackendProbe(ready=True, detail="ok")

    def run(
        self, task: str, *, workspace: str, model: str | None, timeout_sec: float
    ) -> BackendOutcome:
        _ = (task, model, timeout_sec)
        for name, text in self._writes.items():
            (Path(workspace) / name).write_text(text)
        return BackendOutcome(summary=f"wrote {', '.join(self._writes)}")


def _diff(changed: list[str]) -> WorktreeDiff:
    return WorktreeDiff(
        changed_files=changed,
        diff="diff --git a/foo.py b/foo.py\n" if changed else "",
        truncated=False,
    )


# --------------------------------------------------------------------------- #
# backend resolution (real registry)
# --------------------------------------------------------------------------- #
def test_unknown_agent_fails_cleanly() -> None:
    result = run_coding_task(
        "x", workspace="/tmp", model=None, timeout_sec=60, provider="does-not-exist"
    )
    assert result.success is False
    assert "Unsupported coding agent" in (result.error or "")


def test_answer_only_agent_fails_cleanly() -> None:
    # codex is a registered LLM provider, but its adapter cannot edit a workspace.
    result = run_coding_task("x", workspace="/tmp", model=None, timeout_sec=60, provider="codex")
    assert result.success is False
    assert "cannot edit a workspace" in (result.error or "")


def test_pi_is_the_coding_capable_agent() -> None:
    from tools.coding_agent.backends.registry import coding_capable_backends

    assert coding_capable_backends() == ["pi"]


# --------------------------------------------------------------------------- #
# pre-flight
# --------------------------------------------------------------------------- #
def test_non_git_workspace_fails_before_the_agent_runs(tmp_path: Path) -> None:
    backend = _FakeBackend()
    with patch(_RESOLVE, return_value=backend):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="fake"
        )
    assert result.success is False
    assert "not a git repository" in (result.error or "")
    # The agent must not have been run at all — no diff could have been captured.
    assert backend.run_kwargs == {}


# --------------------------------------------------------------------------- #
# orchestration (with a non-CLI backend)
# --------------------------------------------------------------------------- #
def test_success_path_forwards_task_and_captures_diff(tmp_path: Path) -> None:
    _git_init(tmp_path)
    backend = _FakeBackend(outcome=BackendOutcome(summary="edited foo.py"))
    with (
        patch(_RESOLVE, return_value=backend),
        patch(_WORKTREE_DIFF, return_value=_diff(["foo.py"])),
    ):
        result = run_coding_task(
            "fix bug", workspace=str(tmp_path), model="a/b", timeout_sec=120, provider="fake"
        )

    assert result.success is True
    assert result.changed_files == ["foo.py"]
    assert "diff --git" in result.diff
    assert backend.run_kwargs == {
        "task": "fix bug",
        "workspace": str(tmp_path),
        "model": "a/b",
        "timeout_sec": 120,
    }


def test_backend_error_maps_to_failure(tmp_path: Path) -> None:
    _git_init(tmp_path)
    backend = _FakeBackend(outcome=BackendOutcome(summary="", error="agent exploded"))
    with (
        patch(_RESOLVE, return_value=backend),
        patch(_WORKTREE_DIFF, return_value=_diff([])),
    ):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="fake"
        )
    assert result.success is False
    assert result.error == "agent exploded"


def test_timeout_maps_to_failure(tmp_path: Path) -> None:
    _git_init(tmp_path)
    backend = _FakeBackend(
        outcome=BackendOutcome(summary="", error="fake timed out after 30s", timed_out=True)
    )
    with (
        patch(_RESOLVE, return_value=backend),
        patch(_WORKTREE_DIFF, return_value=_diff([])),
    ):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=30, provider="fake"
        )
    assert result.success is False
    assert result.timed_out is True


def test_non_cli_backend_needs_no_exit_code(tmp_path: Path) -> None:
    """An MCP backend has no exit code; the runner must not depend on one."""
    _git_init(tmp_path)
    backend = _FakeBackend(outcome=BackendOutcome(summary="done", exit_code=None))
    with (
        patch(_RESOLVE, return_value=backend),
        patch(_WORKTREE_DIFF, return_value=_diff(["foo.py"])),
    ):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="fake"
        )
    assert result.success is True
    assert result.returncode == 0


# --------------------------------------------------------------------------- #
# reporting only this run's changes (real git, no mocked diff)
#
# The workspace is a live checkout that may already hold a developer's
# work-in-progress. A post-run scan alone cannot tell that apart from the agent's
# edits, so the runner fingerprints the tree *before* running.
# --------------------------------------------------------------------------- #
def test_pre_existing_wip_is_not_reported_as_the_agents_work(tmp_path: Path) -> None:
    _git_init(tmp_path)
    (tmp_path / "existing.py").write_text("original\n")
    _git_commit_all(tmp_path)

    # A developer's uncommitted work, present before the agent runs.
    (tmp_path / "existing.py").write_text("developer WIP\n")
    (tmp_path / "scratch.txt").write_text("developer scratch file\n")

    backend = _WritingBackend({"agent.py": "# agent\n"})
    with patch(_RESOLVE, return_value=backend):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="writer"
        )

    assert result.changed_files == ["agent.py"]
    assert "developer WIP" not in result.diff
    assert "developer scratch file" not in result.diff


def test_agent_edit_to_an_already_dirty_file_is_still_reported(tmp_path: Path) -> None:
    """The baseline must compare *content*, not just presence.

    A file can be dirty before the run and still be edited by the agent — dropping it
    for being in the baseline would hide the agent's real work.
    """
    _git_init(tmp_path)
    (tmp_path / "shared.py").write_text("original\n")
    _git_commit_all(tmp_path)
    (tmp_path / "shared.py").write_text("developer WIP\n")

    backend = _WritingBackend({"shared.py": "agent rewrote this\n"})
    with patch(_RESOLVE, return_value=backend):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="writer"
        )

    assert result.changed_files == ["shared.py"]
    assert "agent rewrote this" in result.diff


def test_agent_that_changes_nothing_in_a_dirty_tree_is_not_a_success(tmp_path: Path) -> None:
    """Pre-existing WIP must not be mistaken for the agent having done something."""
    _git_init(tmp_path)
    (tmp_path / "existing.py").write_text("original\n")
    _git_commit_all(tmp_path)
    (tmp_path / "existing.py").write_text("developer WIP\n")

    backend = _WritingBackend({})  # writes nothing
    with patch(_RESOLVE, return_value=backend):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="writer"
        )

    assert result.changed_files == []
    assert result.diff == ""


# --------------------------------------------------------------------------- #
# verify_coding_agent
# --------------------------------------------------------------------------- #
def test_verify_ready() -> None:
    backend = _FakeBackend(probe=BackendProbe(ready=True, detail="ready"))
    with patch(_RESOLVE, return_value=backend):
        assert verify_coding_agent("fake") == (True, "ready")


def test_verify_not_ready() -> None:
    backend = _FakeBackend(probe=BackendProbe(ready=False, detail="not installed"))
    with patch(_RESOLVE, return_value=backend):
        ok, _ = verify_coding_agent("fake")
    assert ok is False


def test_verify_unknown_agent_is_clean() -> None:
    ok, detail = verify_coding_agent("does-not-exist")
    assert ok is False
    assert "Unsupported coding agent" in detail
