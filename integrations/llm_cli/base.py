"""Shared types for LLM CLI adapters (non-interactive subprocess execution)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CLIProbe:
    """Result of probing whether a CLI binary is usable (install + auth + version)."""

    installed: bool
    version: str | None
    logged_in: bool | None
    bin_path: str | None
    detail: str


@dataclass(frozen=True)
class CLIInvocation:
    """A single non-interactive subprocess call (no TTY)."""

    argv: tuple[str, ...]
    stdin: str | None
    cwd: str
    env: dict[str, str] | None
    timeout_sec: float


@runtime_checkable
class LLMCLIAdapter(Protocol):
    """Contract for one-shot, non-interactive LLM CLI execution."""

    name: str
    #: Env var for explicit binary path when not on PATH (e.g. ``CODEX_BIN``).
    binary_env_key: str
    install_hint: str
    auth_hint: str
    min_version: str | None
    default_exec_timeout_sec: float

    def detect(self) -> CLIProbe:
        """Resolve binary, version, and auth. Never raises; returns a structured probe."""
        pass

    def build(
        self,
        *,
        prompt: str,
        model: str | None,
        workspace: str,
        reasoning_effort: str | None = None,
    ) -> CLIInvocation:
        """Build argv for a non-interactive run (no approval prompts, no TTY)."""
        pass

    def parse(self, *, stdout: str, stderr: str, returncode: int) -> str:
        """Extract the model answer from a successful run."""
        pass

    def explain_failure(self, *, stdout: str, stderr: str, returncode: int) -> str:
        """Human-readable failure when returncode != 0 or output is unusable."""
        pass


@runtime_checkable
class CodingCLIAdapter(LLMCLIAdapter, Protocol):
    """An :class:`LLMCLIAdapter` that can also *edit a workspace* (coding mode).

    Most CLI adapters answer questions read-only; a coding-capable one additionally
    owns its agent prompt (:meth:`build_coding_prompt`) and, when built with
    ``coding_mode=True``, produces an invocation that edits the working tree. The
    coding runner resolves an adapter and only dispatches to it when it satisfies
    this protocol *and* ``supports_coding`` is True.
    """

    #: True when this adapter can edit a workspace (not just answer read-only).
    supports_coding: bool

    def build_coding_prompt(self, task: str) -> str:
        """Wrap the untrusted *task* in this agent's coding prompt (see coding_prompt)."""
        pass

    def build(
        self,
        *,
        prompt: str,
        model: str | None,
        workspace: str,
        reasoning_effort: str | None = None,
        coding_mode: bool = False,
    ) -> CLIInvocation:
        """Build argv; when ``coding_mode`` is set, produce an edit-capable invocation."""
        pass
