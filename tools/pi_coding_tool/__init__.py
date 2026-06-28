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

It edits the working tree and returns a summary + git diff; it never commits,
pushes, or opens a PR (see ``integrations/pi``).
"""

from __future__ import annotations

from typing import Any

from integrations.pi import (
    is_pi_coding_enabled,
    pi_coding_model,
    pi_coding_timeout_seconds,
    pi_coding_workspace,
    run_pi_coding_task,
)
from tools.base import BaseTool

_DISABLED_MESSAGE = (
    "Pi coding tool is disabled. Set PI_CODING_ENABLED=1 (and install/authenticate "
    "the Pi CLI) to enable it."
)


class PiCodingTool(BaseTool):
    """Submit a coding task to the Pi agent; it edits the workspace and returns a diff."""

    name = "pi_coding_task"
    display_name = "Pi coding task"
    source = "knowledge"
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
        "summary": "Pi's final message summarizing what it changed",
        "changed_files": "Files modified in the working tree (status porcelain)",
        "diff": "git diff of the changes vs HEAD (truncated if large)",
        "diff_truncated": "True when the diff was truncated",
        "error": "Error detail when the task failed",
    }

    def is_available(self, _sources: dict[str, dict]) -> bool:
        """Only available when explicitly opted in (cheap flag check)."""
        return is_pi_coding_enabled()

    def run(
        self,
        task: str,
        workspace: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if not is_pi_coding_enabled():
            return {"source": "knowledge", "success": False, "error": _DISABLED_MESSAGE}
        if not (task or "").strip():
            return {"source": "knowledge", "success": False, "error": "task is required."}

        result = run_pi_coding_task(
            task,
            workspace=workspace or pi_coding_workspace(),
            model=model or pi_coding_model(),
            timeout_sec=pi_coding_timeout_seconds(),
        )
        return {
            "source": "knowledge",
            "success": result.success,
            "summary": result.summary,
            "changed_files": result.changed_files,
            "diff": result.diff,
            "diff_truncated": result.diff_truncated,
            "returncode": result.returncode,
            "timed_out": result.timed_out,
            "error": result.error,
        }


# Module-level instance so the tool registry auto-discovers it (see tools/registry.py).
pi_coding_task = PiCodingTool()
