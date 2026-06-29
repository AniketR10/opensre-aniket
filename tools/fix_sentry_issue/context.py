"""Gather Sentry issue context and turn it into a coding task for Pi.

Resolves the Sentry config, parses the issue URL, fetches the issue, and compacts
it into a short, **masked** task description. The output is fed (as untrusted text)
to the Pi coding client, which adds its own safety rules + prompt-injection guard.
"""

from __future__ import annotations

from dataclasses import dataclass

from integrations.sentry import SentryConfig, get_sentry_issue, sentry_config_from_env
from integrations.sentry.issue_url import parse_sentry_issue_url
from platform.masking import MaskingContext, MaskingPolicy
from tools.fix_sentry_issue.errors import (
    ERR_INVALID_INPUT,
    ERR_ISSUE_NOT_FOUND,
    ERR_SENTRY_UNAVAILABLE,
    FixIssueError,
)

_MAX_VALUE_CHARS = 500


@dataclass(frozen=True)
class IssueContext:
    """Resolved issue identity + the masked task description handed to Pi."""

    issue_id: str
    task: str


def _resolve_config() -> SentryConfig:
    config = sentry_config_from_env()
    if config is None:
        raise FixIssueError(
            ERR_SENTRY_UNAVAILABLE,
            "Sentry is not configured. Set SENTRY_ORG_SLUG and SENTRY_AUTH_TOKEN "
            "(and SENTRY_URL for self-hosted).",
        )
    return config


def _build_task(issue: dict) -> str:
    """Compact a Sentry issue dict into a short, masked coding task for Pi."""
    masker = MaskingContext(MaskingPolicy.from_env())

    def field(value: object, *, limit: int | None = None) -> str:
        text = str(value or "").strip()
        if limit:
            text = text[:limit]
        return masker.mask(text)

    raw_meta = issue.get("metadata")
    meta = raw_meta if isinstance(raw_meta, dict) else {}
    title = field(issue.get("title"))
    culprit = field(issue.get("culprit"))
    etype = field(meta.get("type"))
    evalue = field(meta.get("value"), limit=_MAX_VALUE_CHARS)
    filename = field(meta.get("filename"))
    function = field(meta.get("function"))
    level = field(issue.get("level"))
    count = field(issue.get("count"))

    error = f"{etype}: {evalue}" if etype and evalue else (etype or evalue)
    if filename and function:
        location = f"{filename} in {function}"
    else:
        location = filename or function

    lines = ["Fix the root cause of this Sentry issue in the current repository.", ""]
    if title:
        lines.append(f"Issue: {title}")
    if error:
        lines.append(f"Error: {error}")
    if culprit:
        lines.append(f"Culprit: {culprit}")
    if location:
        lines.append(f"Location: {location}")
    if level:
        lines.append(f"Level: {level}")
    if count:
        lines.append(f"Times seen: {count}")
    lines += ["", "Make a minimal, correct fix and explain what you changed and why."]
    return "\n".join(lines)


def gather_issue_context(sentry_url: str | None) -> IssueContext:
    """Parse the URL, fetch the issue, and build the masked task. Raises FixIssueError."""
    ref = parse_sentry_issue_url(sentry_url)
    if ref is None:
        raise FixIssueError(
            ERR_INVALID_INPUT,
            "Not a recognizable Sentry issue URL (expected .../issues/<id>/).",
        )

    config = _resolve_config()
    issue = get_sentry_issue(config=config, issue_id=ref.issue_id)
    if not issue:
        raise FixIssueError(
            ERR_ISSUE_NOT_FOUND,
            f"Sentry issue {ref.issue_id} not found (check the URL and token access).",
        )
    return IssueContext(issue_id=ref.issue_id, task=_build_task(issue))
