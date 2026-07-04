"""Thin, safe git wrapper for shipping a Sentry fix as a branch + commit + push.

Every call shells out to the ``git`` binary in the target *workspace* with an
explicit argument list (never ``shell=True``) and a bounded timeout. The push
path is deliberately narrow: it refuses to create or push a *protected* branch
(``main``/``master``/the repo default) and never uses ``--force``. This is the
structural half of the "never pushes to main" guarantee — the branch name is
namespaced and the protected-branch guard rejects anything that resolves to the
base branch.
"""

from __future__ import annotations

import base64
import os
import subprocess
from collections.abc import Sequence

from tools.cross_vendor.fix_sentry_issue.errors import (
    ERR_BRANCH_FAILED,
    ERR_COMMIT_FAILED,
    ERR_GIT_UNAVAILABLE,
    ERR_NOT_A_GIT_REPO,
    ERR_PROTECTED_BRANCH,
    ERR_PUSH_FAILED,
    FixIssueError,
)

_GIT_TIMEOUT_SEC = 60
# Branch names we refuse to create or push to, on top of the resolved default.
_PROTECTED_BRANCHES = frozenset({"main", "master", "develop", "trunk"})


def _run_git(
    workspace: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in *workspace*; raise FixIssueError if git is missing."""
    try:
        return subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            env=env,
        )
    except FileNotFoundError as exc:
        raise FixIssueError(ERR_GIT_UNAVAILABLE, "git is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise FixIssueError(
            ERR_GIT_UNAVAILABLE, f"git command timed out after {_GIT_TIMEOUT_SEC}s."
        ) from exc


def _token_auth_env(github_token: str) -> dict[str, str]:
    """Env that injects an HTTP Authorization header for this push only.

    Uses git's ``GIT_CONFIG_*`` env-config so the token never appears in argv,
    the remote URL, .git/config, or git's output. This makes the push use the
    *provided* token instead of whatever stale credential the local git
    credential helper might have cached (the usual cause of a 403 on push).
    """
    basic = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
    return {
        **os.environ,
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "http.extraheader",
        "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
    }


def is_git_repo(workspace: str) -> bool:
    """True when *workspace* is inside a git work tree."""
    result = _run_git(workspace, "rev-parse", "--is-inside-work-tree")
    return result.returncode == 0 and result.stdout.strip() == "true"


def ensure_git_repo(workspace: str) -> None:
    if not is_git_repo(workspace):
        raise FixIssueError(
            ERR_NOT_A_GIT_REPO, f"{workspace} is not a git repository; cannot open a PR."
        )


def current_branch(workspace: str) -> str:
    """Name of the currently checked-out branch (empty on detached HEAD)."""
    result = _run_git(workspace, "rev-parse", "--abbrev-ref", "HEAD")
    branch = result.stdout.strip()
    return "" if branch in ("", "HEAD") else branch


def default_branch(workspace: str) -> str:
    """Best-effort repo default branch (``origin/HEAD`` target), else current branch."""
    result = _run_git(workspace, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().removeprefix("origin/")
    return current_branch(workspace)


def short_head(workspace: str) -> str:
    """Short SHA of HEAD (used to make branch names unique)."""
    result = _run_git(workspace, "rev-parse", "--short", "HEAD")
    return result.stdout.strip()


def changed_paths(workspace: str) -> list[str]:
    """Paths with staged/unstaged/untracked changes (individual files, not dirs).

    Uses ``-z`` (NUL-separated) so paths are returned verbatim: git's default
    porcelain C-quotes filenames with spaces, quotes, or non-ASCII bytes, which
    would then not match on ``git add``/``hash-object``.
    """
    result = _run_git(workspace, "status", "--porcelain", "-z", "--untracked-files=all")
    tokens = result.stdout.split("\0")
    paths: list[str] = []
    i = 0
    while i < len(tokens):
        record = tokens[i]
        i += 1
        if len(record) < 3:
            continue
        # Porcelain: "XY <path>". Rename/copy (R/C) records are followed by the
        # original path in the next NUL-terminated token, which we skip.
        path = record[3:]
        if path:
            paths.append(path)
        if record[0] in ("R", "C"):
            i += 1
    return paths


def file_fingerprints(workspace: str, paths: Sequence[str]) -> dict[str, str]:
    """Map each path to a git hash of its current worktree content ("" if unreadable).

    Used to tell whether Pi actually *changed* a file that was already dirty before
    the run (hash differs) versus left it untouched (same hash) — so a fix Pi makes
    to a pre-existing WIP file is still committed, while untouched WIP is skipped.
    """
    fingerprints: dict[str, str] = {}
    for path in paths:
        result = _run_git(workspace, "hash-object", "--", path)
        fingerprints[path] = result.stdout.strip() if result.returncode == 0 else ""
    return fingerprints


def assert_not_protected(branch: str, *, protected_extra: str = "") -> None:
    """Raise unless *branch* is a safe, non-base feature branch to push to."""
    name = branch.strip()
    protected = set(_PROTECTED_BRANCHES)
    if protected_extra.strip():
        protected.add(protected_extra.strip())
    if not name or name in protected:
        raise FixIssueError(
            ERR_PROTECTED_BRANCH,
            f"Refusing to create or push protected branch '{name or '(empty)'}'. "
            "Fixes are always shipped on a fresh namespaced branch, never the base branch.",
        )


def create_branch(workspace: str, branch: str, *, base_default: str = "") -> None:
    """Create and switch to *branch* off the current HEAD (protected-name guarded)."""
    assert_not_protected(branch, protected_extra=base_default)
    result = _run_git(workspace, "checkout", "-b", branch)
    if result.returncode != 0:
        raise FixIssueError(
            ERR_BRANCH_FAILED, f"Could not create branch '{branch}': {result.stderr.strip()}"
        )


def commit_paths(workspace: str, paths: Sequence[str], message: str) -> None:
    """Stage and commit *only* the given paths, excluding any other WIP in the tree.

    ``git add`` registers the paths (so newly created files are tracked), and
    ``git commit --only`` commits exactly those paths — disregarding any other
    staged or unstaged changes the developer may have in the working tree.
    """
    if not paths:
        raise FixIssueError(ERR_COMMIT_FAILED, "no files to commit.")

    add = _run_git(workspace, "add", "--", *paths)
    if add.returncode != 0:
        raise FixIssueError(ERR_COMMIT_FAILED, f"git add failed: {add.stderr.strip()}")

    commit = _run_git(workspace, "commit", "--only", "-m", message, "--", *paths)
    if commit.returncode != 0:
        raise FixIssueError(ERR_COMMIT_FAILED, f"git commit failed: {commit.stderr.strip()}")


def push_branch(
    workspace: str,
    branch: str,
    *,
    remote: str = "origin",
    base_default: str = "",
    github_token: str | None = None,
) -> None:
    """Push *branch* to *remote* with upstream tracking. Never force, never base branch.

    When *github_token* is given, the push authenticates with that token (via an
    ephemeral HTTP header) instead of the machine's cached git credentials.
    """
    assert_not_protected(branch, protected_extra=base_default)
    env = _token_auth_env(github_token) if github_token else None
    result = _run_git(workspace, "push", "--set-upstream", remote, branch, env=env)
    if result.returncode != 0:
        raise FixIssueError(
            ERR_PUSH_FAILED, f"git push to {remote}/{branch} failed: {result.stderr.strip()}"
        )
