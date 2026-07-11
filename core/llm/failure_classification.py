"""Shared LLM failure string classification (provider-agnostic)."""

from __future__ import annotations

import re

# Patterns ordered by specificity — first match wins in classify_cli_failure_category_hint.
_QUOTA_RE = re.compile(
    r"quota|rate.?limit|429|too many request|insufficient_quota|"
    r"out of credit|billing|usage limit|spending limit|plan limit|"
    r"exceeded.*limit|limit.*exceeded|maximum.*usage",
    re.IGNORECASE,
)
_AUTH_RE = re.compile(
    r"unauthorized|401|invalid.?api.?key|api.?key.*invalid|"
    r"authentication.?fail|not authenticated|not logged.?in|"
    r"no credentials|token.*expired|expired.*token|invalid.?token|"
    r"permission denied|access denied|403|forbidden",
    re.IGNORECASE,
)
_CONTEXT_OVERFLOW_RE = re.compile(
    r"context.?length|context.?window|max(?:imum)?\s+context\s+length|"
    r"max.?token|token.?limit|prompt.*too\s+long|prompt.*too.?large|"
    r"input.*exceed|reduce.*context|string too long",
    re.IGNORECASE,
)
_NETWORK_RE = re.compile(
    r"network.*error|connection.*refused|dns.*fail|unreachable|"
    r"no route to host|connection reset|ssl.*error|certificate.*error|"
    r"name.*resolution|getaddrinfo",
    re.IGNORECASE,
)
_ERROR_KEYWORD_RE = re.compile(r"error|fail|exception|invalid", re.IGNORECASE)

# Hints for a CLI that gave us nothing to go on. They lead with the *fact* (what the
# process did) and offer causes as possibilities — a confidently wrong cause sends the
# reader down the wrong path, which is worse than admitting we don't know.
_SILENT_FAILURE_HINT = (
    "no error detail from the CLI — common causes are an exhausted quota, expired auth, "
    "or a dropped request; check your plan/credits and auth, then retry"
)
# Exit 0 with no output is a *different* fact: the CLI believes it succeeded. Quota and
# auth failures usually exit non-zero, so leading with them here would mislead.
_EMPTY_SUCCESS_HINT = (
    "the CLI reported success but produced no output — the model may have returned an "
    "empty completion, or the provider may have dropped the request silently; check your "
    "plan/credits and auth, then retry"
)


def is_context_length_overflow(message: str) -> bool:
    """Return True when *message* indicates prompt/context/token limit exhaustion.

    Avoids bare ``too long`` so timeout strings like "request took too long"
    are not misclassified as context overflow.
    """
    return _CONTEXT_OVERFLOW_RE.search(message) is not None


def classify_cli_failure_category_hint(stdout: str, stderr: str, _returncode: int) -> str | None:
    """Return a category hint (quota/auth/context/network) when output matches."""
    combined = f"{stdout}\n{stderr}".strip()

    if _QUOTA_RE.search(combined):
        return "quota or rate limit exceeded — check your plan/billing or wait before retrying"
    if _AUTH_RE.search(combined):
        return "authentication failed — verify your API key or re-login with the provider CLI"
    if is_context_length_overflow(combined):
        return (
            "prompt too long — shorten the input or reduce accumulated context "
            "(/context to inspect)"
        )
    if _NETWORK_RE.search(combined):
        return "network error — check connectivity and provider status"
    return None


def classify_cli_failure_hint(stdout: str, stderr: str, returncode: int) -> str | None:
    """Return a short actionable hint for a known failure category, or None."""
    category = classify_cli_failure_category_hint(stdout, stderr, returncode)
    if category is not None:
        return category

    combined = f"{stdout}\n{stderr}".strip()
    # Reaching here at all means the run failed: adapters call ``explain_failure`` for a
    # non-zero exit *or* an exit-0 run whose output was unusable (empty). Exit 130 is the
    # user pressing Ctrl+C — not a failure to explain.
    if returncode == 130:
        return None
    if not combined or (len(combined) < 120 and not _ERROR_KEYWORD_RE.search(combined)):
        # A silent exit-0 means the CLI *thinks* it succeeded, which points somewhere
        # different from a silent crash — so say so rather than blaming quota for both.
        return _EMPTY_SUCCESS_HINT if returncode == 0 else _SILENT_FAILURE_HINT

    return None


__all__ = [
    "classify_cli_failure_category_hint",
    "classify_cli_failure_hint",
    "is_context_length_overflow",
]
