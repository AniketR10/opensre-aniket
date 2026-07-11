"""Tests for the coding-agent runner: registry dispatch + end-to-end orchestration.

A ``_FakeAdapter`` lets us exercise the runner's orchestration (prompt → build →
execute → diff → classify) without a real coding CLI. Registry-level dispatch
(unsupported / not-coding-capable providers) is tested against the *real* registry.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from integrations.coding_agent import run_coding_task, verify_coding_agent
from integrations.coding_agent.execution import ProcessOutcome
from integrations.git import WorktreeDiff
from integrations.llm_cli.base import CLIInvocation, CLIProbe
from integrations.llm_cli.registry import CLIProviderRegistration

_GET_REG = "integrations.coding_agent.runner.get_cli_provider_registration"
_RUN_PROC = "integrations.coding_agent.runner.run_polled_process"
_WORKTREE_DIFF = "integrations.coding_agent.runner.worktree_diff"


class _FakeAdapter:
    """Minimal coding-capable adapter for exercising the runner in isolation."""

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
        #: Records what the runner passed to build(), so tests can assert on it.
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


def _reg(adapter: _FakeAdapter) -> CLIProviderRegistration:
    return CLIProviderRegistration(adapter_factory=lambda: adapter, model_env_key="FAKE_MODEL")


def _git_init(path: Path) -> None:
    subprocess.run(
        ["git", "init", "-q"], cwd=path, check=True, env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    )


# --------------------------------------------------------------------------- #
# registry dispatch (real registry)
# --------------------------------------------------------------------------- #
def test_unsupported_provider_fails_cleanly() -> None:
    result = run_coding_task(
        "x", workspace="/tmp", model=None, timeout_sec=60, provider="does-not-exist"
    )
    assert result.success is False
    assert "Unsupported coding agent" in (result.error or "")


def test_registered_but_not_coding_capable_fails_cleanly() -> None:
    # codex is a registered LLM provider but its adapter is answer-only (read-only).
    result = run_coding_task("x", workspace="/tmp", model=None, timeout_sec=60, provider="codex")
    assert result.success is False
    assert "does not support coding" in (result.error or "")


# --------------------------------------------------------------------------- #
# workspace / pre-flight
# --------------------------------------------------------------------------- #
def test_non_git_workspace_fails_fast(tmp_path: Path) -> None:
    with patch(_GET_REG, return_value=_reg(_FakeAdapter())):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="fake"
        )
    assert result.success is False
    assert "not a git repository" in (result.error or "")


def test_build_runtime_error_is_clean_failure(tmp_path: Path) -> None:
    _git_init(tmp_path)
    adapter = _FakeAdapter(build_error="fake binary not found")
    with patch(_GET_REG, return_value=_reg(adapter)):
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=60, provider="fake"
        )
    assert result.success is False
    assert "fake binary not found" in (result.error or "")


# --------------------------------------------------------------------------- #
# end-to-end orchestration (execution + diff mocked at the runner boundary)
# --------------------------------------------------------------------------- #
def test_success_path_builds_prompt_and_applies_timeout(tmp_path: Path) -> None:
    _git_init(tmp_path)
    adapter = _FakeAdapter()
    with (
        patch(_GET_REG, return_value=_reg(adapter)),
        patch(_RUN_PROC) as mock_run,
        patch(_WORKTREE_DIFF) as mock_diff,
    ):
        mock_run.return_value = ProcessOutcome(
            stdout="edited foo.py", stderr="", returncode=0, timed_out=False
        )
        mock_diff.return_value = WorktreeDiff(
            changed_files=["foo.py"], diff="diff --git a/foo.py b/foo.py\n", truncated=False
        )
        result = run_coding_task(
            "fix bug", workspace=str(tmp_path), model="a/b", timeout_sec=120, provider="fake"
        )

    assert result.success is True
    assert result.changed_files == ["foo.py"]
    assert "diff --git" in result.diff
    # The agent's own prompt was used, and the coding timeout overrode the adapter default.
    invocation = mock_run.call_args.args[0]
    assert invocation.argv == ("/bin/fake", "-p", "PROMPT::fix bug")
    assert invocation.timeout_sec == 120
    # The runner must request an *edit-capable* invocation and forward the model +
    # resolved workspace. Without coding_mode a write-gated backend (e.g. codex, which
    # otherwise builds a read-only sandbox) would silently make no edits.
    assert adapter.build_kwargs == {
        "model": "a/b",
        "workspace": str(tmp_path),
        "coding_mode": True,
    }


def test_timeout_outcome_maps_to_failure(tmp_path: Path) -> None:
    _git_init(tmp_path)
    adapter = _FakeAdapter()
    with (
        patch(_GET_REG, return_value=_reg(adapter)),
        patch(_RUN_PROC) as mock_run,
        patch(_WORKTREE_DIFF) as mock_diff,
    ):
        mock_run.return_value = ProcessOutcome(stdout="", stderr="", returncode=-1, timed_out=True)
        mock_diff.return_value = WorktreeDiff(changed_files=[], diff="", truncated=False)
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=30, provider="fake"
        )
    assert result.success is False
    assert result.timed_out is True
    assert "fake timed out after 30s" in (result.error or "")


def test_spawn_error_is_clean_failure(tmp_path: Path) -> None:
    _git_init(tmp_path)
    adapter = _FakeAdapter()
    with (
        patch(_GET_REG, return_value=_reg(adapter)),
        patch(_RUN_PROC) as mock_run,
    ):
        mock_run.return_value = ProcessOutcome(
            stdout="",
            stderr="",
            returncode=-1,
            timed_out=False,
            spawn_error="failed to run coding agent: boom",
        )
        result = run_coding_task(
            "x", workspace=str(tmp_path), model=None, timeout_sec=30, provider="fake"
        )
    assert result.success is False
    assert "failed to run coding agent" in (result.error or "")


# --------------------------------------------------------------------------- #
# verify_coding_agent
# --------------------------------------------------------------------------- #
def test_verify_ready() -> None:
    adapter = _FakeAdapter(
        probe=CLIProbe(
            installed=True, version="1", logged_in=True, bin_path="/bin/fake", detail="ready"
        )
    )
    with patch(_GET_REG, return_value=_reg(adapter)):
        ok, detail = verify_coding_agent("fake")
    assert ok is True
    assert detail == "ready"


def test_verify_not_installed() -> None:
    adapter = _FakeAdapter(
        probe=CLIProbe(
            installed=False, version=None, logged_in=None, bin_path=None, detail="missing"
        )
    )
    with patch(_GET_REG, return_value=_reg(adapter)):
        ok, _ = verify_coding_agent("fake")
    assert ok is False


def test_verify_unsupported_provider_is_clean() -> None:
    ok, detail = verify_coding_agent("does-not-exist")
    assert ok is False
    assert "Unsupported coding agent" in detail


# --------------------------------------------------------------------------- #
# live (opt-in): real pi edits a temp git repo. Self-skips without pi/config.
# --------------------------------------------------------------------------- #
def _git_init_repo_with_commit(repo: Path) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    (repo / "hello.txt").write_text("hello\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        env=env,
    )


@pytest.mark.integration
@pytest.mark.live_llm
def test_live_pi_coding_edits_temp_repo(tmp_path: Path) -> None:
    from integrations.coding_agent import coding_model

    binary = shutil.which("pi") or os.environ.get("PI_BIN", "").strip()
    if not binary:
        pytest.skip("pi binary not installed; skipping live coding test")
    model = coding_model()
    if not model:
        pytest.skip("CODING_MODEL/PI_CODING_MODEL not set; skipping live coding test")

    _git_init_repo_with_commit(tmp_path)
    result = run_coding_task(
        "Append a new line containing exactly the word pong to hello.txt. Change nothing else.",
        workspace=str(tmp_path),
        model=model,
        timeout_sec=300,
        provider="pi",
    )

    # If the agent made no change the run is inconclusive for asserting edit
    # mechanics (rate limit/quota, or an underpowered model that declined) — skip.
    if not result.changed_files:
        detail = result.error or result.summary[:160]
        pytest.skip(f"agent made no changes; cannot assert edit mechanics: {detail!r}")

    assert result.success, result.error
    assert "hello.txt" in result.changed_files
    assert "pong" in (tmp_path / "hello.txt").read_text(encoding="utf-8").lower()
    # The tool must not commit: HEAD is still the single init commit.
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert len([ln for ln in log.stdout.splitlines() if ln.strip()]) == 1
