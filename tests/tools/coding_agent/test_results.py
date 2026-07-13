"""Tests for coding-agent result building (``build_coding_result``).

This is where the runner decides *overall* success by combining what the agent said
with what the working tree shows. The subtle rule is the provider-limit one: a limit
signal only counts when nothing changed.
"""

from __future__ import annotations

from integrations.git import WorktreeDiff
from tools.coding_agent.backends.base import BackendOutcome
from tools.coding_agent.results import build_coding_result


def _diff(changed: list[str], *, diff: str = "", truncated: bool = False) -> WorktreeDiff:
    return WorktreeDiff(changed_files=changed, diff=diff, truncated=truncated)


def test_limit_signal_with_changes_is_not_a_limit() -> None:
    """An agent that successfully edits rate-limiting code echoes limit phrases.

    Regression guard: the signal must not be treated as a provider failure when the
    tree actually changed.
    """
    outcome = BackendOutcome(
        summary="Updated the quota manager and rate limit handling.",
        limit_signal=True,
        limit_detail="rate limit exceeded",
    )
    result = build_coding_result(
        outcome, _diff(["quota.py"], diff="diff --git a/quota.py b/quota.py\n"), agent="pi"
    )
    assert result.success is True
    assert result.error is None


def test_limit_signal_with_no_changes_is_a_failure() -> None:
    outcome = BackendOutcome(
        summary="",
        limit_signal=True,
        limit_detail='{"code":429,"status":"RESOURCE_EXHAUSTED"}',
    )
    result = build_coding_result(outcome, _diff([]), agent="pi")
    assert result.success is False
    assert "429" in (result.error or "")


def test_backend_error_is_a_failure() -> None:
    outcome = BackendOutcome(summary="", error="model not found")
    result = build_coding_result(outcome, _diff([]), agent="pi")
    assert result.success is False
    assert result.error == "model not found"


def test_timeout_is_a_failure() -> None:
    outcome = BackendOutcome(summary="", error="codex timed out after 60s", timed_out=True)
    result = build_coding_result(outcome, _diff([]), agent="codex")
    assert result.success is False
    assert result.timed_out is True
    assert "timed out" in (result.error or "")


def test_clean_run_with_no_changes_and_no_output_is_a_failure() -> None:
    outcome = BackendOutcome(summary="")
    result = build_coding_result(outcome, _diff([]), agent="pi")
    assert result.success is False
    assert "made no changes" in (result.error or "")


def test_diff_fields_carried_through() -> None:
    outcome = BackendOutcome(summary="done", exit_code=0)
    result = build_coding_result(outcome, _diff(["a.py"], diff="patch", truncated=True), agent="pi")
    assert result.success is True
    assert result.changed_files == ["a.py"]
    assert result.diff == "patch"
    assert result.diff_truncated is True


def test_missing_exit_code_becomes_zero() -> None:
    """A non-CLI backend (MCP) has no exit code; the result contract still needs one."""
    outcome = BackendOutcome(summary="done", exit_code=None)
    result = build_coding_result(outcome, _diff(["a.py"]), agent="openclaw")
    assert result.returncode == 0
    assert result.success is True
