"""Provider-agnostic entry point for running a coding agent over a workspace.

Resolves the configured backend (``CODING_AGENT``, default ``pi``) through the
shared ``llm_cli`` adapter registry and drives it end to end: readiness →
agent-owned prompt → edit-capable invocation → polled execution → git diff capture →
neutral :class:`CodingResult`. Adding a coding agent is *registering an adapter*
(and marking it ``supports_coding``), not writing a new client — every caller keeps
using :func:`run_coding_task` / :func:`verify_coding_agent` unchanged.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from integrations.coding_agent.config import coding_agent_provider
from integrations.coding_agent.execution import run_polled_process
from integrations.coding_agent.models import CodingResult
from integrations.coding_agent.results import build_coding_result
from integrations.git import GitCommandError, is_git_repo, worktree_diff
from integrations.llm_cli.base import CodingCLIAdapter
from integrations.llm_cli.registry import (
    coding_capable_providers,
    get_cli_provider_registration,
)


def _resolve_adapter(provider: str | None) -> tuple[str, CodingCLIAdapter | None, str]:
    """Resolve *provider* to a coding-capable adapter.

    Returns ``(name, adapter, error)``. ``adapter`` is set only when the provider is
    registered *and* coding-capable; otherwise ``error`` explains why and lists the
    valid choices.
    """
    name = (provider or coding_agent_provider()).strip().lower()
    reg = get_cli_provider_registration(name)
    if reg is None:
        return name, None, _unsupported_message(name, "is not a registered coding agent")
    adapter = reg.adapter_factory()
    if not isinstance(adapter, CodingCLIAdapter) or not adapter.supports_coding:
        return (
            name,
            None,
            _unsupported_message(
                name, "is registered as an LLM provider but does not support coding tasks yet"
            ),
        )
    return name, adapter, ""


def _unsupported_message(name: str, reason: str) -> str:
    supported = ", ".join(coding_capable_providers()) or "(none)"
    return (
        f"Unsupported coding agent '{name}': it {reason}. "
        f"Set CODING_AGENT to a coding-capable provider: {supported}."
    )


def verify_coding_agent(provider: str | None = None) -> tuple[bool, str]:
    """Whether the configured coding agent is installed/ready (never raises)."""
    name, adapter, error = _resolve_adapter(provider)
    if adapter is None:
        return False, error
    probe = adapter.detect()
    if not probe.installed:
        return False, probe.detail
    if probe.logged_in is False:
        return False, probe.detail
    return True, probe.detail


def run_coding_task(
    task: str,
    *,
    workspace: str,
    model: str | None,
    timeout_sec: float,
    provider: str | None = None,
) -> CodingResult:
    """Run the configured coding agent on *task* in *workspace*.

    Pre-flight failures (unsupported provider, missing binary, bad workspace) and
    execution failures (timeout, provider limit, no-op) are all returned as a
    populated ``CodingResult`` with ``success=False`` and a human-readable ``error``;
    this function does not raise for expected conditions.
    """
    name, adapter, error = _resolve_adapter(provider)
    if adapter is None:
        return CodingResult(success=False, summary="", returncode=-1, error=error)

    ws = str(Path(workspace).expanduser()) if workspace else os.getcwd()
    if not Path(ws).is_dir():
        return CodingResult(
            success=False, summary="", returncode=-1, error=f"workspace is not a directory: {ws}"
        )

    # The tool's contract is "edit + return a reviewable diff". Without git we can
    # neither capture nor review changes, so fail fast *before* editing rather than
    # letting the agent edit files and reporting a misleading success with no diff.
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

    prompt = adapter.build_coding_prompt(task)
    try:
        invocation = adapter.build(prompt=prompt, model=model, workspace=ws, coding_mode=True)
    except RuntimeError as exc:
        # e.g. binary not found at build time — an expected pre-flight failure.
        return CodingResult(success=False, summary="", returncode=-1, error=str(exc))

    # The coding path uses the configured coding timeout, not the adapter's default
    # (which is tuned for short answer-mode calls).
    outcome = run_polled_process(replace(invocation, timeout_sec=timeout_sec))
    if outcome.spawn_error:
        return CodingResult(success=False, summary="", returncode=-1, error=outcome.spawn_error)

    try:
        diff = worktree_diff(ws)
    except GitCommandError as exc:
        return CodingResult(
            success=False, summary="", returncode=outcome.returncode, error=exc.message
        )

    return build_coding_result(outcome, diff, timeout_sec=timeout_sec, agent=name)
