"""Coding backends: the transports a coding agent can be reached over.

``base`` defines the ``CodingBackend`` seam, ``cli`` implements it for CLI agents,
and ``registry`` maps a ``CODING_AGENT`` name onto one. The runner imports the seam,
never a concrete backend — which is what lets a non-CLI agent (OpenClaw over MCP)
land here without the orchestration changing.
"""

from __future__ import annotations

from tools.coding_agent.backends.base import BackendOutcome, BackendProbe, CodingBackend

__all__ = ["BackendOutcome", "BackendProbe", "CodingBackend"]
