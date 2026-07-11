"""Tests for core LLM failure string classification."""

from __future__ import annotations

from core.llm.failure_classification import (
    classify_cli_failure_category_hint,
    classify_cli_failure_hint,
    is_context_length_overflow,
)


def test_is_context_length_overflow_distinguishes_timeouts() -> None:
    assert is_context_length_overflow("prompt is too long: 200001 tokens > 200000 maximum")
    assert not is_context_length_overflow("The request took too long to complete")


def test_classify_cli_failure_category_hint_quota() -> None:
    hint = classify_cli_failure_category_hint("", "rate limit exceeded", 1)
    assert hint is not None
    assert "rate limit" in hint


def test_classify_cli_failure_hint_silent_exit() -> None:
    hint = classify_cli_failure_hint("", "", 1)
    assert hint is not None
    assert "no error detail" in hint


def test_classify_cli_failure_hint_silent_exit_zero_is_explained() -> None:
    """A CLI that exits 0 with no output still means the run failed.

    Adapters call ``explain_failure`` for a non-zero exit *or* an exit-0 run whose
    output was unusable, so this is the case that most needs an explanation — it used
    to be the one case that got none.
    """
    hint = classify_cli_failure_hint("", "", 0)
    assert hint is not None
    assert "reported success but produced no output" in hint


def test_classify_cli_failure_hint_distinguishes_silent_crash_from_empty_success() -> None:
    """The two are different facts and must not share a message.

    A silent *crash* (non-zero) is usually quota/auth. A silent *success* (exit 0) is
    not — quota and auth failures exit non-zero — so leading with quota there would
    send the reader down the wrong path.
    """
    crashed = classify_cli_failure_hint("", "", 1)
    empty_success = classify_cli_failure_hint("", "", 0)
    assert crashed != empty_success
    # Neither may state a single cause as fact; both offer possibilities.
    for hint in (crashed, empty_success):
        assert hint is not None
        assert "may have" in hint or "common causes" in hint


def test_classify_cli_failure_hint_user_interrupt_gets_no_hint() -> None:
    """Exit 130 is the user pressing Ctrl+C — not a failure to blame on anything."""
    assert classify_cli_failure_hint("", "", 130) is None
