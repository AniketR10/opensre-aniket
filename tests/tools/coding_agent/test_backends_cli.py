"""Tests for the CLI coding backend (``CLICodingBackend``).

This is the only place in the coding path that knows about argv, subprocesses, and
exit codes — so this is where those behaviours are pinned.
"""

from __future__ import annotations

from unittest.mock import patch

from integrations.llm_cli.base import CLIInvocation, CLIProbe
from integrations.llm_cli.polled_runner import ProcessOutcome
from tools.coding_agent.backends.cli import CLICodingBackend

_RUN_PROC = "tools.coding_agent.backends.cli.run_polled_process"


class _FakeAdapter:
    """Minimal coding-capable CLI adapter."""

    name = "fake"
    binary_env_key = "FAKE_BIN"
    install_hint = "install fake"
    auth_hint = "auth fake"
    min_version: str | None = None
    default_exec_timeout_sec = 10.0
    supports_coding = True

    def __init__(self, *, probe: CLIProbe | None = None, build_error: str | None = None) -> None:
        self._probe = probe or CLIProbe(
            installed=True, version="1", logged_in=True, bin_path="/bin/fake", detail="ok"
        )
        self._build_error = build_error
        self.build_kwargs: dict[str, object] = {}

    def detect(self) -> CLIProbe:
        return self._probe

    def build_coding_prompt(self, task: str) -> str:
        return f"PROMPT::{task}"

    def build(
        self,
        *,
        prompt: str,
        model: str | None,
        workspace: str,
        reasoning_effort: str | None = None,
        coding_mode: bool = False,
    ) -> CLIInvocation:
        _ = reasoning_effort
        self.build_kwargs = {"model": model, "workspace": workspace, "coding_mode": coding_mode}
        if self._build_error:
            raise RuntimeError(self._build_error)
        return CLIInvocation(
            argv=("/bin/fake", "-p", prompt),
            stdin=None,
            cwd=workspace,
            env={"NO_COLOR": "1"},
            timeout_sec=self.default_exec_timeout_sec,
        )

    def parse(self, *, stdout: str, stderr: str, returncode: int) -> str:
        _ = (stderr, returncode)
        return stdout.strip()

    def explain_failure(self, *, stdout: str, stderr: str, returncode: int) -> str:
        _ = returncode
        return stderr or stdout or "failed"


def _backend(adapter: _FakeAdapter | None = None) -> CLICodingBackend:
    return CLICodingBackend(adapter or _FakeAdapter(), name="fake")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# detect
# --------------------------------------------------------------------------- #
def test_detect_ready() -> None:
    assert _backend().detect().ready is True


def test_detect_not_installed() -> None:
    adapter = _FakeAdapter(
        probe=CLIProbe(
            installed=False, version=None, logged_in=None, bin_path=None, detail="missing"
        )
    )
    probe = _backend(adapter).detect()
    assert probe.ready is False
    assert probe.detail == "missing"


def test_detect_not_authenticated() -> None:
    adapter = _FakeAdapter(
        probe=CLIProbe(
            installed=True, version="1", logged_in=False, bin_path="/bin/fake", detail="no auth"
        )
    )
    assert _backend(adapter).detect().ready is False


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def test_run_uses_agent_prompt_coding_mode_and_timeout() -> None:
    adapter = _FakeAdapter()
    with patch(_RUN_PROC) as mock_run:
        mock_run.return_value = ProcessOutcome(
            stdout="edited foo.py", stderr="", returncode=0, timed_out=False
        )
        outcome = _backend(adapter).run("fix bug", workspace="/ws", model="a/b", timeout_sec=120)

    assert outcome.error is None
    assert outcome.summary == "edited foo.py"
    invocation = mock_run.call_args.args[0]
    # The agent's own prompt was used, and the coding timeout overrode the adapter default.
    assert invocation.argv == ("/bin/fake", "-p", "PROMPT::fix bug")
    assert invocation.timeout_sec == 120
    # Without coding_mode a write-gated agent (codex builds a read-only sandbox)
    # would run and silently edit nothing.
    assert adapter.build_kwargs == {"model": "a/b", "workspace": "/ws", "coding_mode": True}


def test_run_build_failure_is_a_clean_outcome() -> None:
    adapter = _FakeAdapter(build_error="fake binary not found")
    outcome = _backend(adapter).run("x", workspace="/ws", model=None, timeout_sec=60)
    assert "fake binary not found" in (outcome.error or "")


def test_run_spawn_error_is_a_clean_outcome() -> None:
    with patch(_RUN_PROC) as mock_run:
        mock_run.return_value = ProcessOutcome(
            stdout="", stderr="", returncode=-1, timed_out=False, spawn_error="failed to run: boom"
        )
        outcome = _backend().run("x", workspace="/ws", model=None, timeout_sec=60)
    assert "failed to run" in (outcome.error or "")


def test_run_timeout() -> None:
    with patch(_RUN_PROC) as mock_run:
        mock_run.return_value = ProcessOutcome(stdout="", stderr="", returncode=-1, timed_out=True)
        outcome = _backend().run("x", workspace="/ws", model=None, timeout_sec=30)
    assert outcome.timed_out is True
    assert "timed out after 30s" in (outcome.error or "")


def test_run_nonzero_exit_is_an_error() -> None:
    with patch(_RUN_PROC) as mock_run:
        mock_run.return_value = ProcessOutcome(
            stdout="", stderr="model not found: bogus", returncode=1, timed_out=False
        )
        outcome = _backend().run("x", workspace="/ws", model="bogus", timeout_sec=60)
    assert "model not found" in (outcome.error or "")


# --------------------------------------------------------------------------- #
# provider limits — reported as a *signal*, not a verdict
# --------------------------------------------------------------------------- #
def test_limit_phrases_on_clean_exit_are_a_signal_not_an_error() -> None:
    """The backend cannot tell a real 429 from an agent editing rate-limit code.

    It must hand the runner a signal (which the runner resolves using the diff), not
    a failure — otherwise a successful edit to quota code gets reported as a failure.
    """
    with patch(_RUN_PROC) as mock_run:
        mock_run.return_value = ProcessOutcome(
            stdout='{"code":429,"status":"RESOURCE_EXHAUSTED"}',
            stderr="",
            returncode=0,
            timed_out=False,
        )
        outcome = _backend().run("x", workspace="/ws", model=None, timeout_sec=60)

    assert outcome.limit_signal is True
    assert outcome.error is None  # not the backend's call
    assert "429" in outcome.limit_detail
