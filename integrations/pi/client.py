"""Pi coding-task client.

Runs the Pi CLI (https://pi.dev) in headless agentic mode inside a target
workspace so it can implement a coding task (read/write/edit/bash), then captures
what changed via git. This is the *hands* role for Pi, the inverse of the
``integrations/llm_cli`` provider role (the *brain*).

Safety model (see issue: "Add Pi as an integration and tool for submitting
coding tasks"): the task prompt forbids commits/pushes and destructive git
commands, and the caller gates invocation (the ``tools`` layer only runs this when
``PI_CODING_ENABLED`` is set — off by default, since the tool is offered on the
investigation surface). This module only edits the working tree and reports the
diff; it never commits, pushes, or opens a PR.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from integrations.llm_cli.binary_resolver import (
    candidate_binary_names,
    default_cli_fallback_paths,
    resolve_cli_binary,
)
from integrations.llm_cli.env_overrides import PI_PROVIDER_ENV_KEYS, nonempty_env_values
from integrations.llm_cli.subprocess_env import build_cli_subprocess_env
from platform.masking import MaskingContext, MaskingPolicy

_GIT_TIMEOUT_SEC = 30.0
_MAX_DIFF_CHARS = 20000
_MAX_OUTPUT_CHARS = 8000
_INSTALL_HINT = "npm i -g @earendil-works/pi-coding-agent"


@dataclass(frozen=True)
class PiCodingResult:
    """Outcome of a Pi coding task run."""

    success: bool
    summary: str
    changed_files: list[str] = field(default_factory=list)
    diff: str = ""
    returncode: int = 0
    timed_out: bool = False
    error: str | None = None
    diff_truncated: bool = False


def _resolve_pi_binary() -> str | None:
    return resolve_cli_binary(
        explicit_env_key="PI_BIN",
        binary_names=candidate_binary_names("pi"),
        fallback_paths=lambda: default_cli_fallback_paths("pi"),
    )


def _pi_subprocess_env() -> dict[str, str]:
    """Color-free env with BYOK provider keys forwarded to the Pi subprocess."""
    env: dict[str, str] = {"NO_COLOR": "1"}
    env.update(nonempty_env_values(PI_PROVIDER_ENV_KEYS))
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        val = os.environ.get(key, "").strip()
        if val:
            env[key] = val
    return env


def _build_task_prompt(task: str) -> str:
    """Wrap the task with the same safety rules the Claude Code runner uses."""
    return (
        "You are the Pi coding agent working inside the given repository.\n\n"
        f"--- Task ---\n{task.strip()}\n\n"
        "--- Rules ---\n"
        "- Implement the requested change in this repository.\n"
        "- Follow AGENTS.md, existing project conventions, and local code style.\n"
        "- Do NOT create a git commit or push changes.\n"
        "- Do NOT run destructive git commands (reset --hard, checkout --, clean -fdx).\n"
        "- Preserve unrelated changes already in the working tree.\n"
        "- Run focused tests or lint checks when practical.\n"
        "- Finish with a concise summary of the files you changed and any verification you ran.\n"
    )


def _git(args: list[str], cwd: str) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1, ""
    return proc.returncode, proc.stdout or ""


def _is_git_repo(cwd: str) -> bool:
    rc, out = _git(["rev-parse", "--is-inside-work-tree"], cwd)
    return rc == 0 and out.strip() == "true"


def _changed_files(cwd: str) -> list[str]:
    """Working-tree changes (modified, added, deleted, untracked) via porcelain."""
    rc, out = _git(["status", "--porcelain"], cwd)
    if rc != 0:
        return []
    files: list[str] = []
    for line in out.splitlines():
        # porcelain format: "XY <path>" (path starts at column 3)
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if path:
            files.append(path)
    return files


def run_pi_coding_task(
    task: str,
    *,
    workspace: str,
    model: str | None,
    timeout_sec: float,
) -> PiCodingResult:
    """Run Pi against *task* in *workspace*; return summary + diff of what changed."""
    binary = _resolve_pi_binary()
    if not binary:
        return PiCodingResult(
            success=False,
            summary="",
            returncode=-1,
            error=f"Pi CLI not found on PATH or known locations. Install with: {_INSTALL_HINT} or set PI_BIN.",
        )

    ws = str(Path(workspace).expanduser()) if workspace else os.getcwd()
    if not Path(ws).is_dir():
        return PiCodingResult(
            success=False, summary="", returncode=-1, error=f"workspace is not a directory: {ws}"
        )

    is_git = _is_git_repo(ws)

    argv: list[str] = [binary, "-p", _build_task_prompt(task)]
    resolved_model = (model or "").strip()
    if resolved_model:
        argv.extend(["--model", resolved_model])

    env = build_cli_subprocess_env(_pi_subprocess_env())

    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=ws,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            env=env,
            check=False,
        )
        stdout, stderr, returncode = (proc.stdout or ""), (proc.stderr or ""), proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        returncode = -1
    except OSError as exc:
        return PiCodingResult(
            success=False, summary="", returncode=-1, error=f"failed to run pi: {exc}"
        )

    changed_files: list[str] = []
    diff = ""
    diff_truncated = False
    if is_git:
        changed_files = _changed_files(ws)
        _, diff = _git(["diff", "HEAD"], ws)
        if len(diff) > _MAX_DIFF_CHARS:
            diff = diff[:_MAX_DIFF_CHARS]
            diff_truncated = True

    # Mask free-text fields (Pi may echo env/secrets); the diff is left verbatim
    # since masking would corrupt code the caller needs to review.
    masker = MaskingContext(MaskingPolicy.from_env())
    out_text = (stdout or "").strip()
    err_text = (stderr or "").strip()
    summary = masker.mask(out_text[:_MAX_OUTPUT_CHARS])

    # Pi prints provider errors (e.g. a 429 quota/rate-limit) to *stdout* and can
    # still exit 0, so detect limit/error signatures in the combined output rather
    # than trusting the exit code alone.
    lowered = f"{out_text}\n{err_text}".lower()
    hit_limit = any(
        marker in lowered
        for marker in (
            "resource_exhausted",
            "quota",
            "rate limit",
            "too many requests",
            "credit balance",
            '"code":429',
            '"code": 429',
            '"code":413',
            '"code": 413',
        )
    )

    made_changes = bool(changed_files)
    # A real success edited something or at least produced a summary, with no
    # limit/error markers and a clean exit.
    success = (
        (not timed_out) and returncode == 0 and not hit_limit and (made_changes or bool(summary))
    )

    error: str | None = None
    if timed_out:
        error = f"pi timed out after {timeout_sec:.0f}s"
    elif returncode != 0 or hit_limit:
        detail = err_text or out_text or f"pi exited with code {returncode}"
        error = masker.mask(detail[:_MAX_OUTPUT_CHARS])
    elif not made_changes and not summary:
        error = (
            "Pi exited cleanly but made no changes and produced no output "
            "(the model may have hit a rate limit/quota or declined the task)."
        )

    return PiCodingResult(
        success=success,
        summary=summary,
        changed_files=changed_files,
        diff=diff,
        returncode=returncode,
        timed_out=timed_out,
        error=error,
        diff_truncated=diff_truncated,
    )
