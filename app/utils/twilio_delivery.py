"""Twilio delivery helpers (SMS + WhatsApp transport).

Single shared transport for posting Twilio Messaging API requests. SMS uses
the raw E.164 number or a Messaging Service SID; the ``whatsapp`` channel
applies the ``whatsapp:`` prefix and exists so the standalone ``whatsapp``
integration can reuse this transport.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import httpx

from app.utils.truncation import truncate

logger = logging.getLogger(__name__)

_MESSAGE_LIMIT = 1600
_WHATSAPP_LIMIT = 4096
_TWILIO_BASE_URL = "https://api.twilio.com/2010-04-01/Accounts"

TwilioChannel = Literal["whatsapp", "sms"]


def _redact_token(text: str, token: str) -> str:
    """Replace access token with <redacted> to prevent accidental log leakage."""
    if token and token in text:
        return text.replace(token, "<redacted>")
    return text


def _normalize_endpoint(channel: TwilioChannel, value: str) -> str:
    """Apply the ``whatsapp:`` prefix for the WhatsApp channel.

    SMS numbers are passed through unchanged (Twilio expects E.164).
    """
    stripped = (value or "").strip()
    if channel != "whatsapp":
        return stripped
    return stripped if stripped.startswith("whatsapp:") else f"whatsapp:{stripped}"


def post_twilio_message(
    channel: TwilioChannel,
    to: str,
    text: str,
    account_sid: str,
    auth_token: str,
    from_number: str = "",
    messaging_service_sid: str = "",
    status_callback: str = "",
) -> tuple[bool, str, str]:
    """Send a Twilio Messaging API request for ``whatsapp`` or ``sms``.

    Returns ``(success, error, message_sid)``. Either ``from_number`` or
    ``messaging_service_sid`` must be set; if both are provided,
    ``messaging_service_sid`` wins (Twilio's documented precedence).
    """
    if not (from_number or messaging_service_sid):
        return False, "Missing from_number or messaging_service_sid.", ""

    logger.debug("[twilio] post %s message to %s", channel, to)
    url = f"{_TWILIO_BASE_URL}/{account_sid}/Messages.json"
    payload: dict[str, str] = {
        "To": _normalize_endpoint(channel, to),
        "Body": text,
    }
    if messaging_service_sid:
        payload["MessagingServiceSid"] = messaging_service_sid
    elif from_number:
        payload["From"] = _normalize_endpoint(channel, from_number)
    if status_callback:
        payload["StatusCallback"] = status_callback

    try:
        response = httpx.post(
            url,
            data=payload,
            auth=(account_sid, auth_token),
            timeout=15.0,
            follow_redirects=False,
        )
    except Exception as exc:
        error = _redact_token(str(exc), auth_token)
        logger.warning("[twilio] %s post exception: %s", channel, error)
        return False, error, ""

    parsed: dict[str, Any] = {}
    try:
        raw = response.json()
        if isinstance(raw, dict):
            parsed = raw
    except Exception:
        parsed = {}

    if response.status_code not in (200, 201):
        if parsed:
            error_message = str(
                parsed.get("message")
                or parsed.get("error_message")
                or f"HTTP {response.status_code}"
            )
        else:
            error_message = response.text or f"HTTP {response.status_code}"
        error_message = _redact_token(error_message, auth_token)
        logger.warning("[twilio] %s post failed: %s", channel, error_message)
        return False, error_message, ""

    return True, "", str(parsed.get("sid") or "")


def send_twilio_sms_report(
    report: str,
    sms_ctx: dict[str, Any],
) -> tuple[bool, str, str]:
    """Send a truncated report as SMS via Twilio.

    Returns ``(success, error, message_sid)``. ``sms_ctx`` must include
    ``account_sid``, ``auth_token``, ``to``, and either ``from_number`` or
    ``messaging_service_sid``. ``status_callback`` is optional.
    """
    account_sid = str(sms_ctx.get("account_sid") or "")
    auth_token = str(sms_ctx.get("auth_token") or "")
    to = str(sms_ctx.get("to") or "")
    from_number = str(sms_ctx.get("from_number") or "")
    messaging_service_sid = str(sms_ctx.get("messaging_service_sid") or "")
    status_callback = str(sms_ctx.get("status_callback") or "")

    if not account_sid or not auth_token or not to:
        return False, "Missing account_sid, auth_token, or to", ""
    if not (from_number or messaging_service_sid):
        return False, "Missing from_number or messaging_service_sid", ""

    text = truncate(report, _MESSAGE_LIMIT, suffix="…")
    return post_twilio_message(
        channel="sms",
        to=to,
        text=text,
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
        messaging_service_sid=messaging_service_sid,
        status_callback=status_callback,
    )
