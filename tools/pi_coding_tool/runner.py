"""Lifecycle and execution orchestration for the Pi coding tool.

These are the stages ``PiCodingTool.run`` drives, kept as small free functions so
the tool class stays a thin agent-facing contract:

- :func:`is_pi_coding_enabled` — the tool's opt-in gate (``PI_CODING_ENABLED``).
- :func:`ensure_enabled`    — raise unless opted in.
- :func:`ensure_cli_ready`  — coding agent installed and authenticated (via the seam).
- :func:`execute`           — run the coding agent (``integrations/coding_agent`` seam).
- :func:`to_output`         — shape a stable result dict (with ``error_kind``).
- :func:`error_output`      — the same dict shape for an early/expected failure.

Every return path goes through :func:`_base_output`, so the result dict always
carries the same keys — a caller reading ``diff`` or ``changed_files`` never has to
guard for a gate failure that short-circuited before the coding run.

The tool is agent-neutral: it routes through the ``integrations/coding_agent`` seam,
so ``CODING_AGENT`` (default ``pi``) selects the backend. The ``PI_CODING_ENABLED``
gate is this tool's own opt-in and is independent of provider selection.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Final

from integrations.coding_agent import (
    CodingResult,
    run_coding_task,
    verify_coding_agent,
)
from tools.pi_coding_tool.errors import (
    ERR_CLI_UNAVAILABLE,
    ERR_DISABLED,
    ERR_EXECUTION,
    ERR_TIMEOUT,
    PiCodingError,
)
from tools.pi_coding_tool.validation import ResolvedRequest

#: Evidence source tag stamped on every result dict.
SOURCE: Final = "knowledge"
_TRUTHY = {"1", "true", "yes", "on"}

_DISABLED_MESSAGE = (
    "Pi coding tool is disabled. Set PI_CODING_ENABLED=1 (and install/authenticate "
    "the coding agent) to enable it."
)


def is_pi_coding_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Whether the coding tool is opted in via ``PI_CODING_ENABLED``."""
    source = env if env is not None else os.environ
    return source.get("PI_CODING_ENABLED", "").strip().lower() in _TRUTHY


def ensure_enabled() -> None:
    if not is_pi_coding_enabled():
        raise PiCodingError(ERR_DISABLED, _DISABLED_MESSAGE)


def ensure_cli_ready() -> None:
    available, detail = verify_coding_agent()
    if not available:
        raise PiCodingError(ERR_CLI_UNAVAILABLE, f"Coding agent is not ready: {detail}")


def execute(request: ResolvedRequest) -> CodingResult:
    return run_coding_task(
        request.task,
        workspace=request.workspace,
        model=request.model,
        timeout_sec=request.timeout_sec,
    )


def _base_output() -> dict[str, Any]:
    """Stable output shape shared by every return path, so all keys are always present.

    Callers can read ``summary`` / ``changed_files`` / ``diff`` without guarding for
    early-gate failures (disabled, invalid input, CLI unavailable) that never reached
    the coding run. Defaults mirror :class:`CodingResult`'s own field defaults.
    """
    return {
        "source": SOURCE,
        "success": False,
        "error_kind": None,
        "summary": "",
        "changed_files": [],
        "diff": "",
        "diff_truncated": False,
        "returncode": 0,
        "timed_out": False,
        "error": None,
    }


def to_output(result: CodingResult) -> dict[str, Any]:
    error_kind: str | None = None
    if not result.success:
        error_kind = ERR_TIMEOUT if result.timed_out else ERR_EXECUTION
    return {
        **_base_output(),
        "success": result.success,
        "error_kind": error_kind,
        "summary": result.summary,
        "changed_files": result.changed_files,
        "diff": result.diff,
        "diff_truncated": result.diff_truncated,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "error": result.error,
    }


def error_output(kind: str, message: str) -> dict[str, Any]:
    return {**_base_output(), "error_kind": kind, "error": message}
