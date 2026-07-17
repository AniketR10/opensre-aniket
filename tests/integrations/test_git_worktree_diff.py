"""Tests for ``integrations.git.worktree_diff`` (working-tree diff capture).

Uses real git repos since the behavior under test is git's own (``git diff HEAD``
plus ``git diff --no-index`` for untracked files).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from integrations.git import worktree_diff, worktree_fingerprint


def _git_init_repo(repo: Path) -> None:
    """Init a repo with a single committed file (``hello.txt``)."""
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        env=env,
    )


def test_worktree_diff_includes_new_untracked_files(tmp_path: Path) -> None:
    """A newly created (untracked) file must appear in the diff, not just changed_files."""
    _git_init_repo(tmp_path)
    (tmp_path / "added.py").write_text("print('brand new file')\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path))
    assert "added.py" in result.changed_files
    assert "added.py" in result.diff  # rendered as added content
    assert "brand new file" in result.diff
    assert result.truncated is False


def test_worktree_diff_includes_tracked_edits(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("hello\nworld\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path))
    assert "hello.txt" in result.changed_files
    assert "world" in result.diff


def test_worktree_diff_truncates_large_diff(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    (tmp_path / "big.txt").write_text("x\n" * 20_000, encoding="utf-8")

    result = worktree_diff(str(tmp_path), max_chars=100)
    assert result.truncated is True
    assert len(result.diff) == 100


def test_worktree_diff_truncates_when_untracked_file_cap_is_hit(tmp_path: Path) -> None:
    """Dropping files for the *file* cap must set ``truncated``, like the char cap.

    The files here are tiny, so the char cap never fires — without its own signal the
    caller would be told a diff missing two files was complete.
    """
    _git_init_repo(tmp_path)
    for i in range(5):
        (tmp_path / f"new_{i}.py").write_text(f"# {i}\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path), max_untracked_files=3)
    assert result.truncated is True
    assert len(result.diff) < 20_000  # not the char cap doing the work
    # changed_files stays complete — only the rendered diff is capped.
    assert len([p for p in result.changed_files if p.startswith("new_")]) == 5


def test_worktree_diff_untracked_cap_not_hit_is_not_truncated(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    for i in range(3):
        (tmp_path / f"new_{i}.py").write_text(f"# {i}\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path), max_untracked_files=3)
    assert result.truncated is False


def test_worktree_diff_since_reports_only_changes_made_after_the_fingerprint(
    tmp_path: Path,
) -> None:
    """A dirty workspace: pre-existing WIP must not be attributed to a later run.

    Mirrors the coding-agent flow — fingerprint, let something edit the tree, diff.
    """
    _git_init_repo(tmp_path)
    # Work in progress that exists *before* the fingerprint is taken.
    (tmp_path / "hello.txt").write_text("developer WIP\n", encoding="utf-8")
    (tmp_path / "scratch.md").write_text("developer notes\n", encoding="utf-8")

    baseline = worktree_fingerprint(str(tmp_path))

    # Now an agent adds its own file.
    (tmp_path / "agent.py").write_text("print('agent work')\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path), since=baseline)
    assert result.changed_files == ["agent.py"]
    assert "agent work" in result.diff
    assert "developer WIP" not in result.diff
    assert "developer notes" not in result.diff


def test_worktree_diff_since_includes_a_file_that_was_already_dirty_then_edited(
    tmp_path: Path,
) -> None:
    """Presence in the baseline is not enough to exclude — content must be unchanged."""
    _git_init_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("developer WIP\n", encoding="utf-8")

    baseline = worktree_fingerprint(str(tmp_path))

    (tmp_path / "hello.txt").write_text("agent rewrote this\n", encoding="utf-8")

    result = worktree_diff(str(tmp_path), since=baseline)
    assert result.changed_files == ["hello.txt"]
    assert "agent rewrote this" in result.diff


def test_worktree_diff_since_with_no_new_changes_is_empty(tmp_path: Path) -> None:
    """A dirty tree nothing touched since the fingerprint reports no changes."""
    _git_init_repo(tmp_path)
    (tmp_path / "hello.txt").write_text("developer WIP\n", encoding="utf-8")
    (tmp_path / "scratch.md").write_text("developer notes\n", encoding="utf-8")

    baseline = worktree_fingerprint(str(tmp_path))

    result = worktree_diff(str(tmp_path), since=baseline)
    assert result.changed_files == []
    assert result.diff == ""
    assert result.truncated is False


def test_worktree_diff_clean_tree_is_empty(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    result = worktree_diff(str(tmp_path))
    assert result.changed_files == []
    assert result.diff == ""
    assert result.truncated is False
