"""Orchestrate a coding task: pick a backend, run it, capture what changed.

This is the agent-neutral entry point every caller uses. It owns the parts that are
invariant across coding agents — resolving ``CODING_AGENT``, validating the
workspace, capturing the git diff, and deciding success — and delegates *reaching
the agent* to a :class:`~tools.coding_agent.backends.base.CodingBackend`.

Nothing here knows how an agent is reached. A CLI subprocess (Pi today) and an MCP
session (OpenClaw later) are the same shape from this module's point of view, which
is what stops the coding path from being a CLI-only design.
"""

from __future__ import annotations

import os
from pathlib import Path

from integrations.git import GitCommandError, is_git_repo, worktree_diff
from tools.coding_agent.backends.base import CodingBackend
from tools.coding_agent.backends.registry import coding_capable_backends, resolve_backend
from tools.coding_agent.config import coding_agent_provider
from tools.coding_agent.models import CodingResult
from tools.coding_agent.results import build_coding_result


def _resolve(provider: str | None) -> tuple[str, CodingBackend | None, str]:
    """Resolve *provider* to a coding backend, or explain why it is unusable."""
    name = (provider or coding_agent_provider()).strip().lower()
    backend = resolve_backend(name)
    if backend is None:
        supported = ", ".join(coding_capable_backends()) or "(none)"
        return (
            name,
            None,
            f"Unsupported coding agent '{name}': it is not a registered agent, or it is "
            f"registered but cannot edit a workspace. Set CODING_AGENT to one of: {supported}.",
        )
    return name, backend, ""


def verify_coding_agent(provider: str | None = None) -> tuple[bool, str]:
    """Whether the configured coding agent is installed/ready (never raises)."""
    _name, backend, error = _resolve(provider)
    if backend is None:
        return False, error
    probe = backend.detect()
    return probe.ready, probe.detail


def run_coding_task(
    task: str,
    *,
    workspace: str,
    model: str | None,
    timeout_sec: float,
    provider: str | None = None,
) -> CodingResult:
    """Run the configured coding agent on *task* in *workspace*.

    Pre-flight failures (unsupported agent, missing binary, bad workspace) and
    execution failures (timeout, provider limit, no-op) all come back as a populated
    ``CodingResult`` with ``success=False`` and a human-readable ``error``; this
    function does not raise for expected conditions.
    """
    name, backend, error = _resolve(provider)
    if backend is None:
        return CodingResult(success=False, summary="", returncode=-1, error=error)

    ws = str(Path(workspace).expanduser()) if workspace else os.getcwd()
    if not Path(ws).is_dir():
        return CodingResult(
            success=False, summary="", returncode=-1, error=f"workspace is not a directory: {ws}"
        )

    # The contract is "edit + return a reviewable diff". Without git we can neither
    # capture nor review changes, so fail fast *before* editing rather than let the
    # agent loose and report a misleading success with no diff.
    try:
        if not is_git_repo(ws):
            return CodingResult(
                success=False,
                summary="",
                returncode=-1,
                error=f"workspace is not a git repository; the tool needs git to capture changes: {ws}",
            )
    except GitCommandError as exc:
        return CodingResult(success=False, summary="", returncode=-1, error=exc.message)

    outcome = backend.run(task, workspace=ws, model=model, timeout_sec=timeout_sec)

    try:
        diff = worktree_diff(ws)
    except GitCommandError as exc:
        return CodingResult(
            success=False, summary=outcome.summary, returncode=-1, error=exc.message
        )

    return build_coding_result(outcome, diff, agent=name)
