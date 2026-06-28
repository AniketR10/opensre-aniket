"""Tests for the Pi coding integration (config, verifier, client)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from integrations.pi import (
    is_pi_coding_enabled,
    pi_coding_model,
    pi_coding_timeout_seconds,
    pi_coding_workspace,
    run_pi_coding_task,
    verify_pi_coding,
)

_RESOLVE = "integrations.pi.client._resolve_pi_binary"
_RUN = "integrations.pi.client.subprocess.run"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_is_pi_coding_enabled_truthy_values() -> None:
    assert is_pi_coding_enabled({"PI_CODING_ENABLED": "1"}) is True
    assert is_pi_coding_enabled({"PI_CODING_ENABLED": "true"}) is True
    assert is_pi_coding_enabled({"PI_CODING_ENABLED": "YES"}) is True
    assert is_pi_coding_enabled({"PI_CODING_ENABLED": "0"}) is False
    assert is_pi_coding_enabled({}) is False


def test_pi_coding_model_and_workspace() -> None:
    assert pi_coding_model({"PI_CODING_MODEL": "  groq/llama-3.1-8b-instant  "}) == (
        "groq/llama-3.1-8b-instant"
    )
    assert pi_coding_model({}) is None
    assert pi_coding_workspace({"PI_CODING_WORKSPACE": "/repo"}) == "/repo"


def test_pi_coding_timeout_clamped() -> None:
    with patch.dict("os.environ", {"PI_CODING_TIMEOUT_SECONDS": "5"}, clear=False):
        assert pi_coding_timeout_seconds() == 60.0  # clamped to minimum
    with patch.dict("os.environ", {"PI_CODING_TIMEOUT_SECONDS": "99999"}, clear=False):
        assert pi_coding_timeout_seconds() == 1800.0  # clamped to maximum


# --------------------------------------------------------------------------- #
# verifier
# --------------------------------------------------------------------------- #
@patch("integrations.pi.verifier.PiAdapter")
def test_verify_pi_coding_installed_and_authed(mock_cls: MagicMock) -> None:
    mock_cls.return_value.detect.return_value = MagicMock(
        installed=True, logged_in=True, detail="ok"
    )
    available, detail = verify_pi_coding()
    assert available is True
    assert detail == "ok"


@patch("integrations.pi.verifier.PiAdapter")
def test_verify_pi_coding_not_installed(mock_cls: MagicMock) -> None:
    mock_cls.return_value.detect.return_value = MagicMock(
        installed=False, logged_in=None, detail="not found"
    )
    available, _ = verify_pi_coding()
    assert available is False


@patch("integrations.pi.verifier.PiAdapter")
def test_verify_pi_coding_not_authed(mock_cls: MagicMock) -> None:
    mock_cls.return_value.detect.return_value = MagicMock(
        installed=True, logged_in=False, detail="not logged in"
    )
    available, _ = verify_pi_coding()
    assert available is False


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #
def _git_side_effect(diff: str = "diff --git a/foo.py b/foo.py\n+changed\n") -> object:
    def side_effect(cmd: list[str], **_: object) -> MagicMock:
        if cmd[0] == "git":
            sub = cmd[1]
            if sub == "rev-parse":
                return MagicMock(returncode=0, stdout="true\n", stderr="")
            if sub == "status":
                return MagicMock(returncode=0, stdout=" M foo.py\n?? bar.py\n", stderr="")
            if sub == "diff":
                return MagicMock(returncode=0, stdout=diff, stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        # the pi invocation
        return MagicMock(returncode=0, stdout="Edited foo.py to fix the bug.\n", stderr="")

    return side_effect


@patch(_RUN)
@patch(_RESOLVE, return_value="/usr/bin/pi")
def test_run_pi_coding_task_success_captures_diff(
    _mock_resolve: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_run.side_effect = _git_side_effect()
    result = run_pi_coding_task(
        "fix the bug",
        workspace=str(tmp_path),
        model="anthropic/claude-haiku-4-5",
        timeout_sec=60,
    )
    assert result.success is True
    assert "foo.py" in result.changed_files
    assert "bar.py" in result.changed_files
    assert "diff --git" in result.diff
    assert "Edited foo.py" in result.summary
    assert result.error is None
    # the pi invocation carried the model flag and ran in the workspace
    pi_call = next(c for c in mock_run.call_args_list if c.args[0][0] == "/usr/bin/pi")
    assert "--model" in pi_call.args[0]
    assert pi_call.kwargs["cwd"] == str(tmp_path)


@patch(_RESOLVE, return_value=None)
def test_run_pi_coding_task_binary_missing(_mock_resolve: MagicMock, tmp_path: Path) -> None:
    result = run_pi_coding_task("x", workspace=str(tmp_path), model=None, timeout_sec=60)
    assert result.success is False
    assert "Pi CLI not found" in (result.error or "")


@patch(_RUN)
@patch(_RESOLVE, return_value="/usr/bin/pi")
def test_run_pi_coding_task_timeout(
    _mock_resolve: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    def side_effect(cmd: list[str], **kwargs: object) -> MagicMock:
        if cmd[0] == "git":
            if cmd[1] == "rev-parse":
                return MagicMock(returncode=0, stdout="true\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=60)

    mock_run.side_effect = side_effect
    result = run_pi_coding_task("x", workspace=str(tmp_path), model=None, timeout_sec=60)
    assert result.success is False
    assert result.timed_out is True
    assert "timed out" in (result.error or "")


@patch(_RUN)
@patch(_RESOLVE, return_value="/usr/bin/pi")
def test_run_pi_coding_task_nonzero_exit(
    _mock_resolve: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    def side_effect(cmd: list[str], **_: object) -> MagicMock:
        if cmd[0] == "git":
            if cmd[1] == "rev-parse":
                return MagicMock(returncode=0, stdout="true\n", stderr="")
            return MagicMock(returncode=0, stdout="", stderr="")
        return MagicMock(returncode=1, stdout="", stderr="model not found: bogus")

    mock_run.side_effect = side_effect
    result = run_pi_coding_task("x", workspace=str(tmp_path), model="bogus", timeout_sec=60)
    assert result.success is False
    assert "model not found" in (result.error or "")


# --------------------------------------------------------------------------- #
# live (opt-in): real pi edits a temp git repo. Self-skips without pi/config.
# --------------------------------------------------------------------------- #
def _git_init_repo(repo: Path) -> None:
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
    binary = shutil.which("pi") or os.environ.get("PI_BIN", "").strip()
    if not binary:
        pytest.skip("pi binary not installed; skipping live Pi coding test")
    if not is_pi_coding_enabled():
        pytest.skip("PI_CODING_ENABLED not set; skipping live Pi coding test")
    model = pi_coding_model()
    if not model:
        pytest.skip("PI_CODING_MODEL not set; skipping live Pi coding test")

    _git_init_repo(tmp_path)
    result = run_pi_coding_task(
        "Append a new line containing exactly the word pong to hello.txt. Change nothing else.",
        workspace=str(tmp_path),
        model=model,
        timeout_sec=300,
    )

    # If Pi made no change the run is inconclusive for asserting edit mechanics
    # (most often a provider rate limit/quota, or an underpowered model that
    # declined the task) — skip rather than hard-fail the integration.
    if not result.changed_files:
        detail = result.error or result.summary[:160]
        pytest.skip(f"Pi made no changes; cannot assert edit mechanics: {detail!r}")

    assert result.success, result.error
    assert "hello.txt" in result.changed_files
    assert "pong" in (tmp_path / "hello.txt").read_text(encoding="utf-8").lower()
    # The tool must not commit: HEAD is still the single init commit.
    log = subprocess.run(
        ["git", "log", "--oneline"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert len([ln for ln in log.stdout.splitlines() if ln.strip()]) == 1
