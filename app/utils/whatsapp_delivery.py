"""WhatsApp delivery helper — back-compat wrapper over twilio_delivery.

This module preserves the legacy public surface used by the publish-findings
node and existing tools. Internally it delegates to
:mod:`app.utils.twilio_delivery`, which provides the unified Twilio transport
shared with the SMS channel.
"""

from __future__ import annotations

import logging
from typing import Any

from app.utils.truncation import truncate
from app.utils.twilio_delivery import _WHATSAPP_LIMIT, post_twilio_message

logger = logging.getLogger(__name__)

__all__ = [
    "post_whatsapp_message_twilio",
    "send_whatsapp_report",
]


def post_whatsapp_message_twilio(
    to: str,
    text: str,
    account_sid: str,
    auth_token: str,
    from_number: str,
) -> tuple[bool, str, str]:
    """Send a WhatsApp message via Twilio Messaging API.

    Returns ``(success, error, message_id)``.
    """
    logger.debug("[whatsapp] post twilio message to %s", to)
    return post_twilio_message(
        channel="whatsapp",
        to=to,
        text=text,
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
    )


def send_whatsapp_report(
    report: str,
    whatsapp_ctx: dict[str, Any],
) -> tuple[bool, str]:
    """Send a truncated report to WhatsApp. Returns ``(success, error)``."""
    account_sid: str = str(whatsapp_ctx.get("account_sid") or "")
    auth_token: str = str(whatsapp_ctx.get("auth_token") or "")
    from_number: str = str(whatsapp_ctx.get("from_number") or "")
    to: str = str(whatsapp_ctx.get("to") or "")
    if not account_sid or not auth_token or not from_number or not to:
        return False, "Missing account_sid, auth_token, from_number, or to"

    text = truncate(report, _WHATSAPP_LIMIT, suffix="…")
    post_success, error, _ = post_whatsapp_message_twilio(
        to=to,
        text=text,
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
    )
    return (True, "") if post_success else (False, error)
