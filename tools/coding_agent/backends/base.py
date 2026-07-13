"""Transport-agnostic contract for a coding backend.

A coding backend is *whatever can edit a workspace*: a CLI subprocess today (Pi,
via ``integrations/llm_cli``), an MCP session tomorrow (OpenClaw, which is not a
CLI at all). The orchestration in :mod:`tools.coding_agent.runner` talks only to
this protocol, so **nothing here may mention argv, subprocesses, or exit codes** —
the moment it does, a non-CLI agent can no longer be a backend.

Division of labour:

* the **backend** knows how to reach its agent and what its failures look like;
* the **runner** owns everything invariant across agents — validating the
  workspace, capturing the git diff, and deciding overall success.

That split is why a backend reports what happened (:class:`BackendOutcome`) rather
than a finished :class:`~tools.coding_agent.models.CodingResult`: "produced a
reviewable change" is a contract every backend shares, so the runner enforces it
in one place instead of trusting each backend to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class BackendProbe:
    """Whether a backend is installed and authenticated."""

    ready: bool
    detail: str


@dataclass(frozen=True)
class BackendOutcome:
    """What a backend reports after running a coding task.

    The backend states what the agent *said* and whether the run itself failed. It
    does **not** decide overall success — the runner combines this with the git diff
    (did the working tree actually change?) to reach a verdict.
    """

    #: The agent's final message. Masked by the backend; may be empty.
    summary: str
    #: A definite failure — the agent crashed, timed out, or could not be started.
    #: ``None`` when the agent ran to completion, whatever it achieved.
    error: str | None = None
    timed_out: bool = False
    #: The provider signalled a quota/rate limit (e.g. a 429 printed to stdout while
    #: still exiting 0). Deliberately *not* a failure on its own: an agent that
    #: successfully edits rate-limiting code will echo those same phrases. Only the
    #: runner can judge it, because only the runner knows whether the tree changed.
    limit_signal: bool = False
    #: Masked detail the runner uses to explain a confirmed limit failure.
    limit_detail: str = ""
    #: Transport-specific completion signal, carried for diagnostics only. ``None``
    #: for transports with no such concept (e.g. MCP). The runner never branches on
    #: it — it only passes it through to the caller.
    exit_code: int | None = None


@runtime_checkable
class CodingBackend(Protocol):
    """Something that can run a coding task against a workspace."""

    #: Name that ``CODING_AGENT`` uses to select this backend.
    name: str

    def detect(self) -> BackendProbe:
        """Whether the backend is installed and authenticated. Never raises."""
        ...

    def run(
        self,
        task: str,
        *,
        workspace: str,
        model: str | None,
        timeout_sec: float,
    ) -> BackendOutcome:
        """Run *task* against *workspace*.

        Never raises for expected failures (agent missing, timeout, provider limit,
        refusal) — those come back as a populated :class:`BackendOutcome` with an
        ``error``. The backend owns its own agent prompt, so the tool layer never
        carries one.
        """
        ...
