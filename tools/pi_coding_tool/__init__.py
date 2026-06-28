"""Pi coding-task tool: hand a coding task to the Pi agent so it implements the change.

This is the first **mutating** agent-callable tool, so it is deliberately gated,
mirroring how ``run_diagnostic_code`` ships disabled by default:

- ``side_effect_level = "mutating"``. ``requires_approval = True`` documents intent,
  but note it is only honored by the messaging-approval surface — the investigation
  tool loop does not enforce it — so the **real gate is ``is_available`` below**.
- ``is_available`` returns True only when ``PI_CODING_ENABLED`` is set, so it is
  never offered to the agent unless the operator opts in.
- ``surfaces = ("investigation",)`` — the surface the REPL assistant tool loop and
  the investigation pipeline actually consume (the ``chat`` surface has no live
  consumer). Reachability is gated by ``PI_CODING_ENABLED``, not by the surface.

Lifecycle (``run`` orchestrates these stages, each with a clear failure mode):

1. ``_ensure_enabled``    — opt-in gate (``PI_CODING_ENABLED``).
2. ``_resolve_request``   — validate + normalize ``task`` / ``workspace`` / ``model``.
3. ``_ensure_cli_ready``  — Pi binary installed and authenticated.
4. ``_execute``           — run the polled Pi process (``integrations/pi`` client).
5. ``_to_output``         — shape a stable result dict with an ``error_kind``.

Expected failures return a structured ``{"success": False, "error_kind": ...}`` dict;
any *unexpected* exception propagates to ``BaseTool.__call__``, which reports it to
Sentry (the global tool wrapper). It edits the working tree and returns a summary + git
diff; it never commits, pushes, or opens a PR (see ``integrations/pi``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.domain.types.evidence import EvidenceSource
from integrations.pi import (
    PiCodingResult,
    is_pi_coding_enabled,
    pi_coding_model,
    pi_coding_timeout_seconds,
    pi_coding_workspace,
    run_pi_coding_task,
    verify_pi_coding,
)
from tools.base import BaseTool

_SOURCE: EvidenceSource = "knowledge"
_MAX_TASK_CHARS = 4000

# Stable error categories surfaced in the tool's ``error_kind`` output field.
ERR_DISABLED = "disabled"
ERR_INVALID_INPUT = "invalid_input"
ERR_CLI_UNAVAILABLE = "cli_unavailable"
ERR_TIMEOUT = "timeout"
ERR_EXECUTION = "execution_error"


class _PiToolError(Exception):
    """An expected, user-actionable failure with a stable ``kind``."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


@dataclass(frozen=True)
class _ResolvedRequest:
    """A validated, fully-resolved coding request ready to execute."""

    task: str
    workspace: str
    model: str | None
    timeout_sec: float


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def _validate_task(task: str | None) -> str:
    cleaned = (task or "").strip()
    if not cleaned:
        raise _PiToolError(ERR_INVALID_INPUT, "task is required and must be non-empty.")
    if len(cleaned) > _MAX_TASK_CHARS:
        raise _PiToolError(
            ERR_INVALID_INPUT,
            f"task is too long ({len(cleaned)} chars); keep it under {_MAX_TASK_CHARS}.",
        )
    return cleaned


def _validate_workspace(workspace: str | None) -> str:
    resolved = (workspace or "").strip() or pi_coding_workspace()
    path = Path(resolved).expanduser()
    if not path.exists():
        raise _PiToolError(ERR_INVALID_INPUT, f"workspace does not exist: {path}")
    if not path.is_dir():
        raise _PiToolError(ERR_INVALID_INPUT, f"workspace is not a directory: {path}")
    return str(path)


def _validate_model(model: str | None) -> str | None:
    resolved = (model or "").strip() or pi_coding_model()
    if resolved is None:
        return None
    # Pi accepts "provider/model" and shorthands (e.g. "sonnet:high"); only reject
    # obviously malformed values (whitespace inside the token).
    if any(ch.isspace() for ch in resolved):
        raise _PiToolError(
            ERR_INVALID_INPUT,
            f"model must not contain whitespace; got {resolved!r}.",
        )
    return resolved


class PiCodingTool(BaseTool):
    """Submit a coding task to the Pi agent; it edits the workspace and returns a diff."""

    name = "pi_coding_task"
    display_name = "Pi coding task"
    source = _SOURCE
    side_effect_level = "mutating"
    surfaces = ("investigation",)
    requires_approval = True
    approval_reason = "Runs the Pi coding agent, which edits files in the target workspace."
    description = (
        "Submit a coding task to the Pi agent (pi.dev). Pi edits files in the workspace to "
        "implement the change and returns a summary plus the git diff. It does not commit, "
        "push, or open a PR. Disabled unless PI_CODING_ENABLED=1 and the Pi CLI is installed "
        "and authenticated."
    )
    use_cases = [
        "Apply a small, well-scoped fix identified during an investigation",
        "Make a targeted code change in the current repository and return the diff for review",
    ]
    anti_examples = [
        "Reading logs, metrics, or traces (use a read-only evidence tool instead)",
        "Large multi-file refactors that should be reviewed interactively",
    ]
    input_schema = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Natural-language description of the coding change Pi should make.",
            },
            "workspace": {
                "type": "string",
                "description": (
                    "Absolute path to the repository to edit. "
                    "Defaults to PI_CODING_WORKSPACE or the current directory."
                ),
                "nullable": True,
            },
            "model": {
                "type": "string",
                "description": (
                    "Optional Pi model override in provider/model form "
                    "(e.g. anthropic/claude-haiku-4-5). Defaults to PI_CODING_MODEL."
                ),
                "nullable": True,
            },
        },
        "required": ["task"],
    }
    outputs = {
        "success": "True when Pi completed and exited cleanly",
        "error_kind": "Stable failure category (disabled, invalid_input, cli_unavailable, "
        "timeout, execution_error) or None on success",
        "summary": "Pi's final message summarizing what it changed",
        "changed_files": "Files modified in the working tree (status porcelain)",
        "diff": "git diff of the changes vs HEAD (truncated if large)",
        "diff_truncated": "True when the diff was truncated",
        "error": "Human-readable error detail when the task failed",
    }

    # ----- availability ---------------------------------------------------- #
    def is_available(self, _sources: dict[str, dict]) -> bool:
        """Only available when explicitly opted in (cheap flag check)."""
        return is_pi_coding_enabled()

    # ----- lifecycle stages ------------------------------------------------ #
    def _ensure_enabled(self) -> None:
        if not is_pi_coding_enabled():
            raise _PiToolError(
                ERR_DISABLED,
                "Pi coding tool is disabled. Set PI_CODING_ENABLED=1 (and install/authenticate "
                "the Pi CLI) to enable it.",
            )

    def _resolve_request(
        self, task: str | None, workspace: str | None, model: str | None
    ) -> _ResolvedRequest:
        return _ResolvedRequest(
            task=_validate_task(task),
            workspace=_validate_workspace(workspace),
            model=_validate_model(model),
            timeout_sec=pi_coding_timeout_seconds(),
        )

    def _ensure_cli_ready(self) -> None:
        available, detail = verify_pi_coding()
        if not available:
            raise _PiToolError(ERR_CLI_UNAVAILABLE, f"Pi CLI is not ready: {detail}")

    def _execute(self, request: _ResolvedRequest) -> PiCodingResult:
        return run_pi_coding_task(
            request.task,
            workspace=request.workspace,
            model=request.model,
            timeout_sec=request.timeout_sec,
        )

    def _to_output(self, result: PiCodingResult) -> dict[str, Any]:
        error_kind: str | None = None
        if not result.success:
            error_kind = ERR_TIMEOUT if result.timed_out else ERR_EXECUTION
        return {
            "source": _SOURCE,
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

    def _error(self, kind: str, message: str) -> dict[str, Any]:
        return {"source": _SOURCE, "success": False, "error_kind": kind, "error": message}

    # ----- entrypoint ------------------------------------------------------ #
    def run(
        self,
        task: str,
        workspace: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        try:
            self._ensure_enabled()
            request = self._resolve_request(task, workspace, model)
            self._ensure_cli_ready()
        except _PiToolError as exc:
            return self._error(exc.kind, exc.message)

        # Expected execution failures (timeout, provider limit, no-op) come back as a
        # populated PiCodingResult; any *unexpected* exception propagates to
        # BaseTool.__call__, which reports it to Sentry (the global tool wrapper).
        return self._to_output(self._execute(request))


# Module-level instance so the tool registry auto-discovers it (see tools/registry.py).
pi_coding_task = PiCodingTool()
