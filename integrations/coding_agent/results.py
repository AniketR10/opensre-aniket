"""Classify a completed coding-agent run into the neutral :class:`CodingResult`.

Backend-agnostic: whatever agent produced the output, success/error is derived the
same way — from the exit code, the captured diff, and provider-limit signatures in
the output. Free-text fields are masked (the agent may echo env/secrets); the diff
is left verbatim since masking would corrupt code the caller needs to review.
"""

from __future__ import annotations

from integrations.coding_agent.execution import ProcessOutcome
from integrations.coding_agent.models import CodingResult
from integrations.git import WorktreeDiff
from platform.masking import MaskingContext, MaskingPolicy

_MAX_OUTPUT_CHARS = 8000

# Provider-side limit/error signatures agents print (often to stdout, exit 0). These
# are specific error phrases — NOT bare words like "quota" — so a task that edits
# quota/rate-limit code is not misread as a provider failure.
_LIMIT_MARKERS: tuple[str, ...] = (
    "resource_exhausted",
    "too many requests",
    "exceeded your current quota",
    "quota exceeded",
    "rate limit exceeded",
    "rate_limit_exceeded",
    "credit balance is too low",
    '"code":429',
    '"code": 429',
    '"code":413',
    '"code": 413',
)


def build_coding_result(
    outcome: ProcessOutcome,
    diff: WorktreeDiff,
    *,
    timeout_sec: float,
    agent: str,
) -> CodingResult:
    """Classify a run into success / error from output, exit code, and changes.

    *agent* is the backend name (e.g. ``pi``, ``codex``) used only to make error
    messages self-describing.
    """
    masker = MaskingContext(MaskingPolicy.from_env())
    out_text = outcome.stdout.strip()
    err_text = outcome.stderr.strip()
    summary = masker.mask(out_text[:_MAX_OUTPUT_CHARS])

    made_changes = bool(diff.changed_files)
    # Agents print provider errors (e.g. a 429 quota/rate-limit) to *stdout* and can
    # still exit 0, so detect limit/error signatures regardless of the exit code —
    # but only when nothing was produced, so a *successful* edit whose output
    # mentions a limit phrase is not misreported as a provider failure.
    lowered = f"{out_text}\n{err_text}".lower()
    hit_limit = (not made_changes) and any(marker in lowered for marker in _LIMIT_MARKERS)

    success = (
        (not outcome.timed_out)
        and outcome.returncode == 0
        and not hit_limit
        and (made_changes or bool(summary))
    )

    error: str | None = None
    if outcome.timed_out:
        error = f"{agent} timed out after {timeout_sec:.0f}s"
    elif outcome.returncode != 0 or hit_limit:
        detail = err_text or out_text or f"{agent} exited with code {outcome.returncode}"
        error = masker.mask(detail[:_MAX_OUTPUT_CHARS])
    elif not made_changes and not summary:
        error = (
            f"{agent} exited cleanly but made no changes and produced no output "
            "(the model may have hit a rate limit/quota or declined the task)."
        )

    return CodingResult(
        success=success,
        summary=summary,
        changed_files=diff.changed_files,
        diff=diff.diff,
        diff_truncated=diff.truncated,
        returncode=outcome.returncode,
        timed_out=outcome.timed_out,
        error=error,
    )
