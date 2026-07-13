"""Tests for the agent-neutral coding-agent config.

``CODING_*`` settings select/tune the backend, with legacy ``PI_CODING_*``
fallbacks so existing setups keep working.
"""

from __future__ import annotations

from unittest.mock import patch

from tools.coding_agent import coding_model, coding_timeout_seconds, coding_workspace
from tools.coding_agent.config import coding_agent_provider


def test_provider_defaults_to_pi_and_normalizes() -> None:
    assert coding_agent_provider({}) == "pi"
    assert coding_agent_provider({"CODING_AGENT": "  Codex "}) == "codex"


def test_model_prefers_coding_then_pi_fallback() -> None:
    assert coding_model({"CODING_MODEL": "  a/b  "}) == "a/b"
    assert coding_model({"PI_CODING_MODEL": "c/d"}) == "c/d"
    assert coding_model({"CODING_MODEL": "a/b", "PI_CODING_MODEL": "c/d"}) == "a/b"
    assert coding_model({}) is None


def test_workspace_prefers_coding_then_pi_fallback() -> None:
    assert coding_workspace({"CODING_WORKSPACE": "/repo"}) == "/repo"
    assert coding_workspace({"PI_CODING_WORKSPACE": "/legacy"}) == "/legacy"
    assert coding_workspace({"CODING_WORKSPACE": "/repo", "PI_CODING_WORKSPACE": "/legacy"}) == (
        "/repo"
    )


def test_timeout_clamped() -> None:
    with patch.dict("os.environ", {"CODING_TIMEOUT_SECONDS": "5"}, clear=False):
        assert coding_timeout_seconds() == 60.0  # clamped to minimum
    with patch.dict("os.environ", {"CODING_TIMEOUT_SECONDS": "99999"}, clear=False):
        assert coding_timeout_seconds() == 1800.0  # clamped to maximum
