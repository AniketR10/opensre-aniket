"""Polled subprocess execution for coding-agent runs.

A coding agent is a long-running child that streams verbose output (tool calls,
edits, progress). Unlike the brain-role ``CLIBackedLLMClient`` — a single blocking
``subprocess.run`` — we spawn it, drain both pipes in background threads, and poll
it to a deadline so a long task is bounded and the process is terminated gracefully
(SIGTERM, then SIGKILL) on timeout.

Backend-agnostic: it drives whatever :class:`CLIInvocation` an
:class:`~integrations.llm_cli.base.LLMCLIAdapter` builds, so every coding backend
shares one execution path.
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import IO

from integrations.llm_cli.base import CLIInvocation
from integrations.llm_cli.subprocess_env import build_cli_subprocess_env

_POLL_INTERVAL_SEC = 0.5
_TERMINATE_GRACE_SEC = 5.0


@dataclass(frozen=True)
class ProcessOutcome:
    """Raw result of polling a coding-agent subprocess to completion or deadline."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    spawn_error: str | None = None


def _terminate(proc: subprocess.Popen[str]) -> None:
    """Stop a still-running child: SIGTERM, then SIGKILL if it lingers."""
    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            proc.kill()


def _drain(pipe: IO[str] | None, buffer: list[str]) -> None:
    """Read *pipe* to EOF into *buffer*.

    A coding agent streams verbose output. If we polled without draining, that
    output would fill the OS pipe buffer (~64 KB), block the child on ``write()``,
    and cause a false timeout. Draining concurrently in a thread is the documented
    alternative to ``communicate()`` when we also need to watch a deadline.
    """
    if pipe is None:
        return
    try:
        for line in pipe:
            buffer.append(line)
    except (OSError, ValueError):
        # Draining is best-effort: the pipe may be closed mid-read when the process
        # is terminated on timeout (OSError) or already closed (ValueError). Either
        # way there is nothing more to read, so stop and let the caller proceed.
        pass
    finally:
        with contextlib.suppress(Exception):
            pipe.close()


def run_polled_process(invocation: CLIInvocation) -> ProcessOutcome:
    """Spawn *invocation*, drain its pipes, and poll to completion or its deadline.

    Polling (rather than a single blocking ``subprocess.run``) lets us enforce the
    deadline ourselves and terminate the process gracefully on timeout. stdout and
    stderr are drained by background threads throughout, so a chatty child can never
    deadlock on a full pipe buffer. Never raises: a spawn failure is reported via
    ``ProcessOutcome.spawn_error``.
    """
    merged_env = build_cli_subprocess_env(invocation.env)
    try:
        proc = subprocess.Popen(
            list(invocation.argv),
            cwd=invocation.cwd,
            env=merged_env,
            stdin=subprocess.DEVNULL if invocation.stdin is None else subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return ProcessOutcome("", "", -1, False, spawn_error=f"failed to run coding agent: {exc}")

    out_buf: list[str] = []
    err_buf: list[str] = []
    readers = (
        threading.Thread(target=_drain, args=(proc.stdout, out_buf), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, err_buf), daemon=True),
    )
    for reader in readers:
        reader.start()

    # Feed stdin only *after* the drain threads are running. An adapter that passes
    # its prompt on stdin (codex reads from ``-``) can exceed the OS pipe buffer, and
    # a chatty child fills its stdout buffer at the same time — writing before we
    # drain would deadlock both sides.
    if invocation.stdin is not None and proc.stdin is not None:
        with contextlib.suppress(Exception):
            proc.stdin.write(invocation.stdin)
            proc.stdin.close()

    deadline = time.monotonic() + max(invocation.timeout_sec, 0.0)
    timed_out = False
    while proc.poll() is None:
        if time.monotonic() >= deadline:
            timed_out = True
            _terminate(proc)
            break
        time.sleep(_POLL_INTERVAL_SEC)

    # Reap the process, then let the drain threads finish (the pipes hit EOF once
    # the child exits or is terminated).
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=_TERMINATE_GRACE_SEC)
    for reader in readers:
        reader.join(timeout=_TERMINATE_GRACE_SEC)

    return ProcessOutcome(
        stdout="".join(out_buf),
        stderr="".join(err_buf),
        returncode=proc.returncode if proc.returncode is not None else -1,
        timed_out=timed_out,
    )
