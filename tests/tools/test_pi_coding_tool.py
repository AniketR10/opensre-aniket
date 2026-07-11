"""Tests for the Pi coding tool: metadata, gating, validators, lifecycle, error_kind."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from integrations.coding_agent import CodingResult
from tools.pi_coding_tool import PiCodingTool, pi_coding_task
from tools.pi_coding_tool.errors import PiCodingError
from tools.pi_coding_tool.validation import validate_model, validate_task, validate_workspace

_VERIFY = "tools.pi_coding_tool.runner.verify_coding_agent"
_RUN = "tools.pi_coding_tool.runner.run_coding_task"


# --------------------------------------------------------------------------- #
# metadata + availability
# --------------------------------------------------------------------------- #
def test_metadata_is_mutating_on_investigation_surface() -> None:
    t = pi_coding_task
    assert t.name == "pi_coding_task"
    assert t.source == "knowledge"
    assert t.side_effect_level == "mutating"
    assert t.requires_approval is False
    assert t.surfaces == ("investigation",)
    assert t.input_schema["required"] == ["task"]
    assert "error_kind" in t.outputs
    assert t.metadata().name == "pi_coding_task"


def test_is_available_off_by_default_then_opt_in() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PI_CODING_ENABLED", None)
        assert pi_coding_task.is_available({}) is False
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        assert pi_coding_task.is_available({}) is True


# --------------------------------------------------------------------------- #
# validators
# --------------------------------------------------------------------------- #
def test_validate_task() -> None:
    assert validate_task("  do it  ") == "do it"
    with pytest.raises(PiCodingError) as empty:
        validate_task("   ")
    assert empty.value.kind == "invalid_input"
    with pytest.raises(PiCodingError) as too_long:
        validate_task("x" * 5000)
    assert too_long.value.kind == "invalid_input"


def test_validate_workspace(tmp_path: Path) -> None:
    assert validate_workspace(str(tmp_path)) == str(tmp_path)
    with pytest.raises(PiCodingError) as missing:
        validate_workspace("/no/such/path/xyz123")
    assert missing.value.kind == "invalid_input"


def test_validate_model() -> None:
    assert validate_model("anthropic/claude-haiku-4-5") == "anthropic/claude-haiku-4-5"
    with pytest.raises(PiCodingError) as bad:
        validate_model("bad model")
    assert bad.value.kind == "invalid_input"
    with patch.dict(os.environ, {"CODING_MODEL": "", "PI_CODING_MODEL": ""}, clear=False):
        assert validate_model("") is None


# --------------------------------------------------------------------------- #
# run() lifecycle + error_kind
# --------------------------------------------------------------------------- #
def test_run_disabled() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PI_CODING_ENABLED", None)
        out = pi_coding_task.run(task="do something")
    assert out["success"] is False
    assert out["error_kind"] == "disabled"


def test_run_invalid_task() -> None:
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="   ")
    assert out["success"] is False
    assert out["error_kind"] == "invalid_input"


@patch(_VERIFY, return_value=(False, "pi not installed"))
def test_run_cli_unavailable(_mock_verify: MagicMock, tmp_path: Path) -> None:
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="fix it", workspace=str(tmp_path))
    assert out["success"] is False
    assert out["error_kind"] == "cli_unavailable"


@patch(_RUN)
@patch(_VERIFY, return_value=(True, "ok"))
def test_run_success(_mock_verify: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = CodingResult(
        success=True,
        summary="edited foo.py",
        changed_files=["foo.py"],
        diff="diff --git a/foo.py b/foo.py\n",
        returncode=0,
    )
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(
            task="fix it", workspace=str(tmp_path), model="groq/llama-3.1-8b-instant"
        )
    assert out["success"] is True
    assert out["error_kind"] is None
    assert out["changed_files"] == ["foo.py"]
    assert "diff --git" in out["diff"]
    kwargs = mock_run.call_args.kwargs
    assert kwargs["workspace"] == str(tmp_path)
    assert kwargs["model"] == "groq/llama-3.1-8b-instant"


@patch(_RUN)
@patch(_VERIFY, return_value=(True, "ok"))
def test_run_error_kinds(_mock_verify: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = CodingResult(
        success=False, summary="", error="pi timed out", timed_out=True
    )
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="fix it", workspace=str(tmp_path))
    assert out["error_kind"] == "timeout"

    mock_run.return_value = CodingResult(success=False, summary="", error="model not found")
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="fix it", workspace=str(tmp_path))
    assert out["error_kind"] == "execution_error"


@patch(_VERIFY, return_value=(True, "ok"))
@patch(_RUN, side_effect=RuntimeError("boom"))
def test_run_unexpected_exception_returns_error_dict(
    _mock_run: MagicMock, _mock_verify: MagicMock, tmp_path: Path
) -> None:
    # __call__ wraps run(); an unexpected exception is reported + degraded to a dict.
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task(task="fix it", workspace=str(tmp_path))
    assert "error" in out


# --------------------------------------------------------------------------- #
# stable output shape — every path returns the same keys
# --------------------------------------------------------------------------- #
_RESULT_KEYS = {
    "source",
    "success",
    "error_kind",
    "summary",
    "changed_files",
    "diff",
    "diff_truncated",
    "returncode",
    "timed_out",
    "error",
}


@patch(_RUN)
@patch(_VERIFY, return_value=(True, "ok"))
def test_success_output_has_full_key_set(
    _mock_verify: MagicMock, mock_run: MagicMock, tmp_path: Path
) -> None:
    mock_run.return_value = CodingResult(success=True, summary="done", changed_files=["a.py"])
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="fix it", workspace=str(tmp_path))
    assert set(out) == _RESULT_KEYS


def test_early_gate_failures_still_have_full_key_set(tmp_path: Path) -> None:
    """A gate failure short-circuits before the coding run, but callers must still be
    able to read ``diff`` / ``changed_files`` without a KeyError."""
    # disabled
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PI_CODING_ENABLED", None)
        disabled = pi_coding_task.run(task="do something")

    # invalid input
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        invalid = pi_coding_task.run(task="   ")

    # cli unavailable
    with (
        patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False),
        patch(_VERIFY, return_value=(False, "not installed")),
    ):
        unavailable = pi_coding_task.run(task="fix it", workspace=str(tmp_path))

    for out in (disabled, invalid, unavailable):
        assert set(out) == _RESULT_KEYS
        assert out["success"] is False
        assert out["error_kind"]
        # the keys that used to be missing entirely
        assert out["changed_files"] == []
        assert out["diff"] == ""
        assert out["summary"] == ""


# --------------------------------------------------------------------------- #
# registry discovery
# --------------------------------------------------------------------------- #
def test_registry_discovers_pi_coding_on_investigation_surface() -> None:
    from tools.registry import get_registered_tool_map

    investigation = get_registered_tool_map("investigation")
    chat = get_registered_tool_map("chat")
    assert "pi_coding_task" in investigation
    assert "pi_coding_task" not in chat
    rt = investigation["pi_coding_task"]
    assert rt.requires_approval is False
    assert rt.side_effect_level == "mutating"


def test_tool_subclass_constructs() -> None:
    assert isinstance(pi_coding_task, PiCodingTool)
