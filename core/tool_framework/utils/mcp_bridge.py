"""Shared unavailable-response builder for MCP bridge tools.

Every MCP bridge tool (``posthog_mcp``, ``sentry_mcp``, ``x_mcp``, ``openclaw``)
returns the same degraded payload when it can't reach its backend: the base
``tool_unavailable`` envelope, optionally annotated with the ``tool`` that was
being dispatched and the ``arguments`` it was called with. This module holds
that one shape so each bridge doesn't reconstruct it by hand.
"""

from __future__ import annotations

from core.tool_framework.utils.tool_availability import tool_unavailable

__all__ = ["MCPBridgeResponse", "unavailable_response"]

# MCP bridge payloads are open string-keyed maps; every bridge aliases this shape
# under its own name (``PostHogMCPResponse``, ``OpenClawBridgeResponse``, ...).
type MCPBridgeResponse = dict[str, object]


def unavailable_response(
    source: str,
    error: str,
    *,
    tool_name: str | None = None,
    arguments: dict[str, object] | None = None,
) -> MCPBridgeResponse:
    """Build the standard unavailable payload for an MCP bridge tool.

    Extends the base ``tool_unavailable`` envelope (``source``/``available``/
    ``error``) with the optional ``tool`` and ``arguments`` keys that a bridge
    attaches when a specific tool call couldn't be dispatched. ``tool`` is added
    only when ``tool_name`` is truthy; ``arguments`` is added whenever it is not
    ``None`` (an empty dict is still recorded).
    """
    payload: MCPBridgeResponse = tool_unavailable(source, error)
    if tool_name:
        payload["tool"] = tool_name
    if arguments is not None:
        payload["arguments"] = arguments
    return payload
