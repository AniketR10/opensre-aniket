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
from urllib.parse import urlsplit

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
# Networked lookups get a tighter bound so a slow/unreachable remote can't stall
# the whole flow (they always have a safe local fallback).
_REMOTE_TIMEOUT_SEC = 15
# Branch names we refuse to create or push to, on top of the resolved default.
_PROTECTED_BRANCHES = frozenset({"main", "master", "develop", "trunk"})


def _run_git(
    workspace: str,
    *args: str,
    env: dict[str, str] | None = None,
    timeout: float = _GIT_TIMEOUT_SEC,
) -> subprocess.CompletedProcess[str]:
    """Run ``git <args>`` in *workspace*; raise FixIssueError if git is missing."""
    try:
        return subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit
            ["git", *args],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise FixIssueError(ERR_GIT_UNAVAILABLE, "git is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise FixIssueError(
            ERR_GIT_UNAVAILABLE, f"git command timed out after {timeout:.0f}s."
        ) from exc


def _remote_https_base(workspace: str, remote: str = "origin") -> str:
    """``scheme://host/`` of *remote* when it uses HTTP(S), else "" (SSH/file/etc.)."""
    result = _run_git(workspace, "remote", "get-url", remote)
    if result.returncode != 0:
        return ""
    parsed = urlsplit(result.stdout.strip())
    if parsed.scheme in ("http", "https") and parsed.hostname:
        return f"{parsed.scheme}://{parsed.hostname}/"
    return ""


def _token_auth_env(github_token: str, base_url: str) -> dict[str, str]:
    """Env that injects an HTTP Authorization header scoped to *base_url* for this call.

    Uses git's ``GIT_CONFIG_*`` env-config so the token never appears in argv, the
    remote URL, .git/config, or git's output. The header is scoped via
    ``http.<base_url>.extraheader`` so the token is only sent to that host and never
    forwarded to other HTTPS remotes or redirects. This makes the request use the
    *provided* token instead of whatever stale credential the local git credential
    helper might have cached (the usual cause of a 403 on push).
    """
    basic = base64.b64encode(f"x-access-token:{github_token}".encode()).decode()
    env = dict(os.environ)
    # Append at the next free index rather than clobbering an existing
    # GIT_CONFIG_COUNT / GIT_CONFIG_KEY_* the caller may already rely on.
    try:
        count = int(env.get("GIT_CONFIG_COUNT", "0") or "0")
    except ValueError:
        count = 0
    env[f"GIT_CONFIG_KEY_{count}"] = f"http.{base_url}.extraheader"
    env[f"GIT_CONFIG_VALUE_{count}"] = f"Authorization: Basic {basic}"
    env["GIT_CONFIG_COUNT"] = str(count + 1)
    return env


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


def _remote_default_branch(workspace: str, github_token: str | None) -> str:
    """The remote's default branch via ``ls-remote --symref`` (authoritative).

    Bounded by a short timeout and returns "" on any failure/timeout, so a slow or
    unreachable remote never stalls or aborts shipping — the caller falls back to
    the current branch.
    """
    base = _remote_https_base(workspace, "origin")
    env = _token_auth_env(github_token, base) if (github_token and base) else None
    try:
        result = _run_git(
            workspace,
            "ls-remote",
            "--symref",
            "origin",
            "HEAD",
            env=env,
            timeout=_REMOTE_TIMEOUT_SEC,
        )
    except FixIssueError:
        return ""
    if result.returncode != 0:
        return ""
    for line in result.stdout.splitlines():
        # "ref: refs/heads/main\tHEAD"
        if line.startswith("ref:"):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].removeprefix("refs/heads/")
    return ""


def default_branch(workspace: str, *, github_token: str | None = None) -> str:
    """Resolve the repo's default branch (the PR base).

    Prefers the local ``origin/HEAD`` pointer; if it isn't configured (common on
    fresh clones), asks the remote directly so we don't silently target the user's
    current feature branch. Falls back to the current branch only when the remote
    is unreachable.
    """
    result = _run_git(workspace, "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().removeprefix("origin/")
    remote = _remote_default_branch(workspace, github_token)
    return remote or current_branch(workspace)


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
    fingerprints: dict[str, str] = dict.fromkeys(paths, "")
    # Hash all files in a single git invocation (one process, not one per file).
    # Deleted/unreadable paths are filtered out first so they don't fail the batch;
    # they keep the "" fingerprint.
    existing = [p for p in paths if os.path.isfile(os.path.join(workspace, p))]
    if not existing:
        return fingerprints
    result = _run_git(workspace, "hash-object", "--", *existing)
    hashes = result.stdout.splitlines()
    if result.returncode == 0 and len(hashes) == len(existing):
        for path, digest in zip(existing, hashes):
            fingerprints[path] = digest.strip()
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

    When *github_token* is given and *remote* is an HTTPS URL, the push
    authenticates with that token (via an ephemeral, host-scoped HTTP header)
    instead of the machine's cached git credentials. For SSH/other remotes the
    token is not injected (the transport authenticates itself).
    """
    assert_not_protected(branch, protected_extra=base_default)
    env = None
    if github_token:
        base = _remote_https_base(workspace, remote)
        if base:
            env = _token_auth_env(github_token, base)
    result = _run_git(workspace, "push", "--set-upstream", remote, branch, env=env)
    if result.returncode != 0:
        raise FixIssueError(
            ERR_PUSH_FAILED, f"git push to {remote}/{branch} failed: {result.stderr.strip()}"
        )
