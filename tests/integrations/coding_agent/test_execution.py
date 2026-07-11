"""Tests for the polled coding-agent subprocess executor (``run_polled_process``)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from integrations.coding_agent.execution import run_polled_process
from integrations.llm_cli.base import CLIInvocation


def _invocation(
    argv: Sequence[str],
    *,
    cwd: str,
    timeout_sec: float = 30.0,
    stdin: str | None = None,
) -> CLIInvocation:
    return CLIInvocation(
        argv=tuple(argv),
        stdin=stdin,
        cwd=cwd,
        env={"NO_COLOR": "1"},
        timeout_sec=timeout_sec,
    )


def test_drains_large_output_without_deadlock(tmp_path: Path) -> None:
    """Regression: a child that writes more than the OS pipe buffer must not
    deadlock and time out. Without concurrent draining this hangs at ~64 KB."""
    payload = 256 * 1024  # 256 KB, well over the ~64 KB pipe buffer
    code = f"import sys; sys.stdout.write('x' * {payload}); sys.stderr.write('y' * {payload})"
    outcome = run_polled_process(_invocation([sys.executable, "-c", code], cwd=str(tmp_path)))
    assert outcome.timed_out is False
    assert outcome.returncode == 0
    assert len(outcome.stdout) == payload
    assert len(outcome.stderr) == payload


def test_large_stdin_with_chatty_child_does_not_deadlock(tmp_path: Path) -> None:
    """Regression: an adapter that feeds the prompt on stdin (codex reads from ``-``)
    can exceed the OS pipe buffer while the child simultaneously fills its stdout
    buffer. Writing stdin before the drain threads run would deadlock both sides.
    """
    payload = 512 * 1024  # 512 KB in, well over the ~64 KB pipe buffer
    code = (
        "import sys;"
        f"sys.stdout.write('y' * {payload});"  # child is chatty *while* we write stdin
        "data = sys.stdin.read();"
        "sys.stdout.write(str(len(data)))"
    )
    outcome = run_polled_process(
        _invocation([sys.executable, "-c", code], cwd=str(tmp_path), stdin="x" * payload)
    )
    assert outcome.timed_out is False
    assert outcome.returncode == 0
    assert str(payload) in outcome.stdout  # the child read the whole stdin


def test_times_out_and_terminates_a_hanging_child(tmp_path: Path) -> None:
    code = "import time; time.sleep(60)"
    outcome = run_polled_process(
        _invocation([sys.executable, "-c", code], cwd=str(tmp_path), timeout_sec=0.5)
    )
    assert outcome.timed_out is True


def test_spawn_error_on_missing_binary(tmp_path: Path) -> None:
    outcome = run_polled_process(_invocation(["/no/such/binary/xyz123"], cwd=str(tmp_path)))
    assert outcome.spawn_error is not None
    assert outcome.returncode == -1
    assert outcome.timed_out is False


def test_captures_nonzero_exit(tmp_path: Path) -> None:
    outcome = run_polled_process(
        _invocation([sys.executable, "-c", "import sys; sys.exit(3)"], cwd=str(tmp_path))
    )
    assert outcome.returncode == 3
    assert outcome.timed_out is False
    assert outcome.spawn_error is None
