"""Tests for the Pi coding tool (gated, mutating, approval-required)."""

from __future__ import annotations

import os
from unittest.mock import patch

from integrations.pi import PiCodingResult
from tools.pi_coding_tool import PiCodingTool, pi_coding_task


def test_metadata_is_mutating_and_approval_gated() -> None:
    t = pi_coding_task
    assert t.name == "pi_coding_task"
    assert t.source == "knowledge"
    assert t.side_effect_level == "mutating"
    assert t.requires_approval is True
    assert t.surfaces == ("investigation",)
    assert t.input_schema["required"] == ["task"]
    # metadata validates against the strict schema
    assert t.metadata().name == "pi_coding_task"


def test_is_available_off_by_default_then_opt_in() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PI_CODING_ENABLED", None)
        assert pi_coding_task.is_available({}) is False
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        assert pi_coding_task.is_available({}) is True


def test_run_disabled_returns_error() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PI_CODING_ENABLED", None)
        out = pi_coding_task.run(task="do something")
    assert out["success"] is False
    assert "disabled" in out["error"].lower()


def test_run_requires_task() -> None:
    with patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False):
        out = pi_coding_task.run(task="   ")
    assert out["success"] is False
    assert "task is required" in out["error"].lower()


@patch("tools.pi_coding_tool.run_pi_coding_task")
def test_run_success_shapes_output(mock_run: object) -> None:
    mock_run.return_value = PiCodingResult(  # type: ignore[attr-defined]
        success=True,
        summary="edited foo.py",
        changed_files=["foo.py"],
        diff="diff --git a/foo.py b/foo.py\n",
        returncode=0,
    )
    with patch.dict(
        os.environ, {"PI_CODING_ENABLED": "1", "PI_CODING_WORKSPACE": "/repo"}, clear=False
    ):
        out = pi_coding_task.run(task="fix it", model="groq/llama-3.1-8b-instant")

    assert out["success"] is True
    assert out["summary"] == "edited foo.py"
    assert out["changed_files"] == ["foo.py"]
    assert "diff --git" in out["diff"]
    # task forwarded with resolved workspace + model
    kwargs = mock_run.call_args.kwargs  # type: ignore[attr-defined]
    assert kwargs["workspace"] == "/repo"
    assert kwargs["model"] == "groq/llama-3.1-8b-instant"


def test_run_returns_error_dict_on_exception() -> None:
    # __call__ wraps run() and returns a structured error instead of raising.
    with (
        patch.dict(os.environ, {"PI_CODING_ENABLED": "1"}, clear=False),
        patch("tools.pi_coding_tool.run_pi_coding_task", side_effect=RuntimeError("boom")),
    ):
        out = pi_coding_task(task="fix it")
    assert "error" in out


def test_registry_discovers_pi_coding_on_investigation_surface() -> None:
    from tools.registry import get_registered_tool_map

    investigation = get_registered_tool_map("investigation")
    chat = get_registered_tool_map("chat")
    # On the investigation surface (consumed by the REPL assistant tool loop and
    # the investigation pipeline); not on the chat surface (which has no live consumer).
    assert "pi_coding_task" in investigation
    assert "pi_coding_task" not in chat
    rt = investigation["pi_coding_task"]
    assert rt.requires_approval is True
    assert rt.side_effect_level == "mutating"


def test_tool_subclass_constructs() -> None:
    assert isinstance(pi_coding_task, PiCodingTool)
