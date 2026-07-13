"""Coding-agent orchestration.

Callers depend on this interface — a :class:`CodingResult`, a :func:`run_coding_task`
entry point, and :func:`verify_coding_agent` readiness — rather than on a specific
agent, and rather than on how that agent is reached.

The seam is :mod:`tools.coding_agent.backends.base`: a coding agent is anything that
can be probed and asked to edit a workspace. CLI agents (Pi today; codex/claude-code
once their adapters grow a coding mode) come in through
:mod:`tools.coding_agent.backends.cli`. A non-CLI agent — OpenClaw speaks MCP, not a
command line — registers its own backend without the runner changing.

Orchestration lives here in ``tools/`` (agent-callable logic); the clients it drives
live in ``integrations/`` (``llm_cli`` for the CLI transport, ``git`` for capturing
the diff).
"""

from __future__ import annotations

from tools.coding_agent.backends.base import BackendOutcome, BackendProbe, CodingBackend
from tools.coding_agent.config import (
    coding_agent_provider,
    coding_model,
    coding_timeout_seconds,
    coding_workspace,
)
from tools.coding_agent.models import CodingResult
from tools.coding_agent.runner import run_coding_task, verify_coding_agent

__all__ = [
    "BackendOutcome",
    "BackendProbe",
    "CodingBackend",
    "CodingResult",
    "coding_agent_provider",
    "coding_model",
    "coding_timeout_seconds",
    "coding_workspace",
    "run_coding_task",
    "verify_coding_agent",
]
