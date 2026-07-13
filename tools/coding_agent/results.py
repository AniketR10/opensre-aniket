"""Combine what the agent did with what the tree shows, into a :class:`CodingResult`.

Transport-agnostic by construction: it sees a :class:`BackendOutcome` and a
:class:`WorktreeDiff`, never a subprocess or an exit code. The one judgement it
makes that no backend can is whether a provider-limit *signal* was a real failure —
an agent that successfully edits rate-limiting code echoes the same phrases as one
that got 429'd, and only the diff tells them apart.
"""

from __future__ import annotations

from integrations.git import WorktreeDiff
from tools.coding_agent.backends.base import BackendOutcome
from tools.coding_agent.models import CodingResult


def build_coding_result(
    outcome: BackendOutcome,
    diff: WorktreeDiff,
    *,
    agent: str,
) -> CodingResult:
    """Decide overall success from the agent's report plus the working tree."""
    made_changes = bool(diff.changed_files)

    # A limit signal only counts when the agent produced nothing — otherwise the
    # phrase came from the code it was editing, not the provider.
    hit_limit = outcome.limit_signal and not made_changes

    success = (
        outcome.error is None
        and not outcome.timed_out
        and not hit_limit
        and (made_changes or bool(outcome.summary))
    )

    error = outcome.error
    if error is None and hit_limit:
        error = outcome.limit_detail
    elif error is None and not made_changes and not outcome.summary:
        error = (
            f"{agent} exited cleanly but made no changes and produced no output "
            "(the model may have hit a rate limit/quota or declined the task)."
        )

    return CodingResult(
        success=success,
        summary=outcome.summary,
        changed_files=diff.changed_files,
        diff=diff.diff,
        diff_truncated=diff.truncated,
        returncode=outcome.exit_code if outcome.exit_code is not None else 0,
        timed_out=outcome.timed_out,
        error=error,
    )
