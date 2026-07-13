"""Resolve a ``CODING_AGENT`` name to a :class:`CodingBackend`.

Today every coding backend is CLI-backed, so resolution walks the ``llm_cli``
provider registry and wraps any coding-capable adapter in a
:class:`~tools.coding_agent.backends.cli.CLICodingBackend`. A non-CLI agent
(OpenClaw over MCP) registers here as its own backend and the runner does not
change — that is the whole point of the :class:`CodingBackend` seam.
"""

from __future__ import annotations

from integrations.llm_cli.base import CodingCLIAdapter
from integrations.llm_cli.registry import CLI_PROVIDER_REGISTRY, get_cli_provider_registration
from tools.coding_agent.backends.base import CodingBackend
from tools.coding_agent.backends.cli import CLICodingBackend


def _cli_backend(name: str) -> CodingBackend | None:
    """Wrap the CLI adapter registered under *name*, if it can edit a workspace."""
    registration = get_cli_provider_registration(name)
    if registration is None:
        return None
    adapter = registration.adapter_factory()
    if not isinstance(adapter, CodingCLIAdapter) or not adapter.supports_coding:
        return None
    return CLICodingBackend(adapter, name=name)


def resolve_backend(name: str) -> CodingBackend | None:
    """Return the coding backend for *name*, or ``None`` when it cannot code.

    ``None`` covers both "not a known agent" and "known, but answer-only" — the
    caller reports the difference; here they are equally unusable.
    """
    return _cli_backend(name)


def coding_capable_backends() -> list[str]:
    """Names that ``CODING_AGENT`` accepts.

    Used to point users at a valid choice when they pick an unsupported or
    answer-only agent. Instantiates adapters (cheap constructors; no probes).
    """
    return sorted(name for name in CLI_PROVIDER_REGISTRY if _cli_backend(name) is not None)
