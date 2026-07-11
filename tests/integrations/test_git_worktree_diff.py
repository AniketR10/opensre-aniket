"""Tests for ``integrations.git.worktree_diff`` (working-tree diff capture).

Uses real git repos since the behavior under test is git's own (``git diff HEAD``
plus ``git diff --no-index`` for untracked files).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from integrations.git import worktree_diff


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


def test_worktree_diff_clean_tree_is_empty(tmp_path: Path) -> None:
    _git_init_repo(tmp_path)
    result = worktree_diff(str(tmp_path))
    assert result.changed_files == []
    assert result.diff == ""
    assert result.truncated is False
