"""Tests for coding-agent result classification (``build_coding_result``)."""

from __future__ import annotations

from integrations.coding_agent.execution import ProcessOutcome
from integrations.coding_agent.results import build_coding_result
from integrations.git import WorktreeDiff


def _diff(changed: list[str], *, diff: str = "", truncated: bool = False) -> WorktreeDiff:
    return WorktreeDiff(changed_files=changed, diff=diff, truncated=truncated)


def test_limit_word_in_successful_edit_is_not_a_limit() -> None:
    outcome = ProcessOutcome(
        stdout="Updated the quota manager and rate limit handling.",
        stderr="",
        returncode=0,
        timed_out=False,
    )
    result = build_coding_result(
        outcome,
        _diff(["quota.py"], diff="diff --git a/quota.py b/quota.py\n"),
        timeout_sec=60,
        agent="pi",
    )
    assert result.success is True
    assert result.error is None


def test_real_rate_limit_with_no_changes_is_a_failure() -> None:
    outcome = ProcessOutcome(
        stdout='{"error":{"code":429,"status":"RESOURCE_EXHAUSTED"}}',
        stderr="",
        returncode=1,
        timed_out=False,
    )
    result = build_coding_result(outcome, _diff([]), timeout_sec=60, agent="pi")
    assert result.success is False
    assert result.error


def test_timeout_is_a_failure_with_agent_name() -> None:
    outcome = ProcessOutcome(stdout="", stderr="", returncode=-1, timed_out=True)
    result = build_coding_result(outcome, _diff([]), timeout_sec=60, agent="codex")
    assert result.success is False
    assert result.timed_out is True
    assert "codex timed out after 60s" in (result.error or "")


def test_clean_exit_no_changes_no_output_is_a_failure() -> None:
    outcome = ProcessOutcome(stdout="", stderr="", returncode=0, timed_out=False)
    result = build_coding_result(outcome, _diff([]), timeout_sec=60, agent="pi")
    assert result.success is False
    assert "made no changes" in (result.error or "")


def test_diff_fields_carried_through() -> None:
    outcome = ProcessOutcome(stdout="done", stderr="", returncode=0, timed_out=False)
    result = build_coding_result(
        outcome,
        _diff(["a.py"], diff="patch", truncated=True),
        timeout_sec=60,
        agent="pi",
    )
    assert result.success is True
    assert result.changed_files == ["a.py"]
    assert result.diff == "patch"
    assert result.diff_truncated is True
