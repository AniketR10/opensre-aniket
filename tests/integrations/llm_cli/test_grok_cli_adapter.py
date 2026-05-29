"""Tests for the xAI Grok Build CLI adapter (detect / build / failure / env forwarding)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.integrations.llm_cli.binary_resolver import npm_prefix_bin_dirs
from app.integrations.llm_cli.grok_cli import (
    GrokCLIAdapter,
    _classify_grok_auth,
    _fallback_grok_paths,
    _grok_session_authenticated,
)
from tests.integrations.llm_cli.testing_helpers import write_fake_runnable_cli_bin

_VERSION_PROBE = "app.integrations.llm_cli.probe_utils.subprocess.run"
_WHICH = "app.integrations.llm_cli.binary_resolver.shutil.which"


def _posix_path_set(paths: list[str]) -> set[str]:
    return {Path(p).as_posix() for p in paths}


def _version_proc(version: str = "0.1.0") -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = f"grok {version}\n"
    m.stderr = ""
    return m


# ---------------------------------------------------------------------------
# Auth classification
# ---------------------------------------------------------------------------


def test_classify_auth_api_key_set() -> None:
    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test"}, clear=False):
        logged_in, detail = _classify_grok_auth()
    assert logged_in is True
    assert "XAI_API_KEY" in detail


def test_classify_auth_session_file_present(tmp_path: Path) -> None:
    config_dir = tmp_path / ".grok-build"
    config_dir.mkdir()
    (config_dir / "auth.json").write_text('{"token": "abc"}')

    with (
        patch.dict(os.environ, {"XAI_API_KEY": ""}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path.home", return_value=tmp_path),
    ):
        logged_in, detail = _classify_grok_auth()

    assert logged_in is True
    assert "session" in detail


def test_classify_auth_no_credentials_returns_false(tmp_path: Path) -> None:
    with (
        patch.dict(os.environ, {"XAI_API_KEY": ""}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path.home", return_value=tmp_path),
    ):
        logged_in, detail = _classify_grok_auth()

    assert logged_in is False
    assert "grok login" in detail


def test_classify_auth_api_key_wins_over_missing_session(tmp_path: Path) -> None:
    with (
        patch.dict(os.environ, {"XAI_API_KEY": "xai-key"}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path.home", return_value=tmp_path),
    ):
        logged_in, detail = _classify_grok_auth()

    assert logged_in is True
    assert "XAI_API_KEY" in detail


def test_session_unreadable_returns_none() -> None:
    with (
        patch.dict(os.environ, {"XAI_API_KEY": ""}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path") as mock_path,
    ):
        mock_creds = MagicMock()
        mock_creds.exists.return_value = True
        mock_creds.stat.side_effect = OSError("permission denied")
        mock_path.home.return_value.__truediv__.return_value.__truediv__.return_value = mock_creds
        logged_in, detail = _grok_session_authenticated()

    assert logged_in is None
    assert "unclear" in detail.lower()


# ---------------------------------------------------------------------------
# detect()
# ---------------------------------------------------------------------------


@patch(_VERSION_PROBE)
@patch(_WHICH)
def test_detect_logged_in_via_api_key(mock_which: MagicMock, mock_run: MagicMock) -> None:
    mock_which.return_value = "/usr/bin/grok"
    mock_run.return_value = _version_proc()

    with patch.dict(os.environ, {"XAI_API_KEY": "xai-test", "GROK_CLI_BIN": ""}, clear=False):
        probe = GrokCLIAdapter().detect()

    assert probe.installed is True
    assert probe.logged_in is True
    assert probe.bin_path == "/usr/bin/grok"
    assert probe.version == "0.1.0"
    assert "XAI_API_KEY" in probe.detail


@patch(_VERSION_PROBE)
@patch(_WHICH)
def test_detect_logged_in_via_session(
    mock_which: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_which.return_value = "/usr/bin/grok"
    mock_run.return_value = _version_proc()
    config_dir = tmp_path / ".grok"
    config_dir.mkdir()
    (config_dir / "session.json").write_text('{"token": "abc"}')

    with (
        patch.dict(os.environ, {"XAI_API_KEY": "", "GROK_CLI_BIN": ""}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path.home", return_value=tmp_path),
    ):
        probe = GrokCLIAdapter().detect()

    assert probe.installed is True
    assert probe.logged_in is True


@patch(_VERSION_PROBE)
@patch(_WHICH)
def test_detect_not_authenticated(
    mock_which: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_which.return_value = "/usr/bin/grok"
    mock_run.return_value = _version_proc()

    with (
        patch.dict(os.environ, {"XAI_API_KEY": "", "GROK_CLI_BIN": ""}, clear=False),
        patch("app.integrations.llm_cli.grok_cli.Path.home", return_value=tmp_path),
    ):
        probe = GrokCLIAdapter().detect()

    assert probe.installed is True
    assert probe.logged_in is False


@patch("app.integrations.llm_cli.grok_cli._fallback_grok_paths", return_value=[])
@patch(_WHICH, return_value=None)
def test_detect_not_installed(_mock_which: MagicMock, _mock_fallback: MagicMock) -> None:
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        probe = GrokCLIAdapter().detect()
    assert probe.installed is False
    assert probe.logged_in is None
    assert probe.bin_path is None
    assert "not found" in probe.detail.lower()


@patch(_VERSION_PROBE)
@patch(_WHICH)
def test_detect_version_command_fails(mock_which: MagicMock, mock_run: MagicMock) -> None:
    mock_which.return_value = "/usr/bin/grok"
    m = MagicMock()
    m.returncode = 1
    m.stdout = ""
    m.stderr = "some error\n"
    mock_run.return_value = m

    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        probe = GrokCLIAdapter().detect()

    assert probe.installed is False
    assert probe.logged_in is None


@patch(_VERSION_PROBE)
@patch(_WHICH)
def test_detect_version_timeout(mock_which: MagicMock, mock_run: MagicMock) -> None:
    mock_which.return_value = "/usr/bin/grok"
    mock_run.side_effect = subprocess.TimeoutExpired(
        cmd=["/usr/bin/grok", "--version"], timeout=5.0
    )

    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        probe = GrokCLIAdapter().detect()

    assert probe.installed is False
    assert probe.logged_in is None
    assert "--version" in probe.detail


# ---------------------------------------------------------------------------
# build()
# ---------------------------------------------------------------------------


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_basic_invocation(_mock_which: MagicMock) -> None:
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="explain this alert", model=None, workspace="")
    assert inv.argv[0] == "/usr/bin/grok"
    assert "-p" in inv.argv
    assert "explain this alert" in inv.argv
    assert "--output-format" in inv.argv
    assert "plain" in inv.argv
    assert "--no-auto-update" in inv.argv
    # Prompt is delivered as the -p argument, not stdin.
    assert inv.stdin is None
    assert inv.timeout_sec == 300.0


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_never_auto_approves(_mock_which: MagicMock) -> None:
    """OpenSRE uses Grok as a text responder; it must not auto-run Grok's own tools."""
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="p", model=None, workspace="")
    assert "--always-approve" not in inv.argv


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_adds_model_flag(_mock_which: MagicMock) -> None:
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="p", model="grok-build-0.1", workspace="")
    assert "-m" in inv.argv
    idx = inv.argv.index("-m")
    assert inv.argv[idx + 1] == "grok-build-0.1"


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_omits_model_flag_when_empty(_mock_which: MagicMock) -> None:
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="p", model="", workspace="")
    assert "-m" not in inv.argv


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_uses_provided_workspace(_mock_which: MagicMock) -> None:
    workspace = "/my/project"
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="p", model=None, workspace=workspace)
    assert Path(inv.cwd) == Path(workspace)


@patch(_WHICH, return_value="/usr/bin/grok")
def test_build_sets_no_color_env(_mock_which: MagicMock) -> None:
    with patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False):
        inv = GrokCLIAdapter().build(prompt="p", model=None, workspace="")
    assert inv.env is not None
    assert inv.env.get("NO_COLOR") == "1"


@patch("app.integrations.llm_cli.grok_cli._fallback_grok_paths", return_value=[])
@patch(_WHICH, return_value=None)
def test_build_raises_when_binary_not_found(
    _mock_which: MagicMock, _mock_fallback: MagicMock
) -> None:
    with (
        patch.dict(os.environ, {"GROK_CLI_BIN": ""}, clear=False),
        pytest.raises(RuntimeError, match="Grok Build CLI not found"),
    ):
        GrokCLIAdapter().build(prompt="p", model=None, workspace="")


# ---------------------------------------------------------------------------
# parse / explain_failure
# ---------------------------------------------------------------------------


def test_parse_returns_stripped_stdout() -> None:
    result = GrokCLIAdapter().parse(stdout="  hello world  \n", stderr="", returncode=0)
    assert result == "hello world"


def test_explain_failure_includes_returncode_and_stderr() -> None:
    msg = GrokCLIAdapter().explain_failure(stdout="", stderr="boom", returncode=1)
    assert "1" in msg
    assert "boom" in msg


def test_explain_failure_maps_auth_errors() -> None:
    msg = GrokCLIAdapter().explain_failure(stdout="", stderr="401 Unauthorized", returncode=1)
    assert "grok login" in msg.lower() or "xai_api_key" in msg.lower()


def test_explain_failure_falls_back_to_stdout() -> None:
    msg = GrokCLIAdapter().explain_failure(stdout="some output", stderr="", returncode=2)
    assert "some output" in msg


def test_auth_hint_mentions_login_and_api_key() -> None:
    adapter = GrokCLIAdapter()
    assert "grok login" in adapter.auth_hint
    assert "XAI_API_KEY" in adapter.auth_hint


# ---------------------------------------------------------------------------
# GROK_CLI_BIN env override
# ---------------------------------------------------------------------------


def test_detect_uses_grok_cli_bin_env(tmp_path: Path) -> None:
    fake_bin = write_fake_runnable_cli_bin(tmp_path, "my-grok")

    with (
        patch.dict(
            os.environ,
            {"GROK_CLI_BIN": str(fake_bin), "XAI_API_KEY": "xai-t"},
            clear=False,
        ),
        patch(_VERSION_PROBE) as mock_run,
    ):
        mock_run.return_value = _version_proc()
        probe = GrokCLIAdapter().detect()

    assert probe.bin_path == str(fake_bin)
    assert probe.installed is True


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_grok_cli_registry_entry() -> None:
    from app.integrations.llm_cli.registry import get_cli_provider_registration

    reg = get_cli_provider_registration("grok-cli")
    assert reg is not None
    assert reg.model_env_key == "GROK_CLI_MODEL"
    assert reg.adapter_factory().name == "grok-cli"


# ---------------------------------------------------------------------------
# Subprocess env forwarding — XAI_API_KEY must be scoped to the Grok subprocess
# ---------------------------------------------------------------------------


def test_xai_key_forwarded_via_build() -> None:
    """XAI_API_KEY is forwarded explicitly by build(), not via the blanket prefix allowlist."""
    with (
        patch.dict(
            os.environ,
            {
                "XAI_API_KEY": "xai-forward-me",
                "XAI_BASE_URL": "https://proxy.example.com",
                "GROK_CLI_BIN": "",
            },
            clear=False,
        ),
        patch(_WHICH, return_value="/usr/bin/grok"),
    ):
        inv = GrokCLIAdapter().build(prompt="p", model=None, workspace="")

    assert inv.env is not None
    assert inv.env["XAI_API_KEY"] == "xai-forward-me"
    assert inv.env["XAI_BASE_URL"] == "https://proxy.example.com"


def test_xai_key_not_in_blanket_subprocess_env() -> None:
    """XAI_API_KEY must NOT be forwarded via the global prefix allowlist (would leak to others)."""
    from app.integrations.llm_cli.subprocess_env import build_cli_subprocess_env

    with patch.dict(os.environ, {"XAI_API_KEY": "xai-secret"}, clear=False):
        env = build_cli_subprocess_env(None)

    assert "XAI_API_KEY" not in env


# ---------------------------------------------------------------------------
# Fallback paths
# ---------------------------------------------------------------------------


def test_fallback_paths_linux() -> None:
    npm_prefix_bin_dirs.cache_clear()
    with (
        patch("app.integrations.llm_cli.binary_resolver.sys.platform", "linux"),
        patch.dict(os.environ, {"npm_config_prefix": "/custom/npm"}, clear=False),
    ):
        paths = _fallback_grok_paths()

    normalized = _posix_path_set(paths)
    assert "/custom/npm/bin/grok" in normalized
