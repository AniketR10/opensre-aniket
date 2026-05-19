"""Twilio SMS notification tool.

Lets the agent push a short SMS notification through a configured
Twilio integration. The investigation planner exposes this tool
whenever a Twilio integration is configured with the SMS channel
enabled.
"""

from __future__ import annotations

from typing import Any

from app.tools.base import BaseTool
from app.utils.twilio_delivery import send_twilio_sms_report


class TwilioNotifyTool(BaseTool):
    """Send a short SMS notification via the configured Twilio integration."""

    name = "twilio_notify"
    source = "twilio"
    description = (
        "Send a short SMS notification via the configured Twilio integration. "
        "Only available when a Twilio integration with the SMS channel enabled "
        "exists."
    )
    use_cases = [
        "Paging an on-call recipient with a one-line incident summary via SMS",
        "Sending a follow-up SMS when a critical-severity alert escalates",
    ]
    requires = ["twilio"]
    input_schema = {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": (
                    "Recipient phone number in E.164 (e.g. +14155551234). "
                    "Defaults to the channel default_to when omitted."
                ),
            },
            "body": {
                "type": "string",
                "description": "SMS body (truncated to the SMS limit).",
            },
        },
        "required": ["body"],
    }
    outputs = {
        "sid": "Twilio Message SID for the sent SMS",
        "status": "delivery dispatch status — 'sent' or 'failed'",
        "error": "error detail when status is 'failed'",
    }

    def is_available(self, sources: dict[str, Any]) -> bool:
        twilio = sources.get("twilio") or {}
        if not (twilio.get("account_sid") and twilio.get("auth_token")):
            return False
        sms = twilio.get("sms") or {}
        return bool(
            sms.get("enabled") and (sms.get("from_number") or sms.get("messaging_service_sid"))
        )

    def extract_params(self, sources: dict[str, Any]) -> dict[str, Any]:
        twilio = sources.get("twilio") or {}
        sms = twilio.get("sms") or {}
        return {
            "_account_sid": twilio.get("account_sid", ""),
            "_auth_token": twilio.get("auth_token", ""),
            "_from_number": sms.get("from_number", ""),
            "_messaging_service_sid": sms.get("messaging_service_sid", ""),
            "to": sms.get("default_to") or "",
        }

    def run(
        self,
        body: str,
        to: str = "",
        _account_sid: str = "",
        _auth_token: str = "",
        _from_number: str = "",
        _messaging_service_sid: str = "",
        **_kwargs: Any,
    ) -> dict[str, Any]:
        if not _account_sid or not _auth_token:
            return {
                "source": "twilio",
                "available": False,
                "status": "failed",
                "error": "Twilio integration is not configured.",
                "sid": "",
            }
        if not to:
            return {
                "source": "twilio",
                "available": True,
                "status": "failed",
                "error": "No recipient — pass 'to' or configure sms.default_to.",
                "sid": "",
            }
        if not (_from_number or _messaging_service_sid):
            return {
                "source": "twilio",
                "available": True,
                "status": "failed",
                "error": "Twilio SMS channel has no from_number or messaging_service_sid.",
                "sid": "",
            }

        ok, error, sid = send_twilio_sms_report(
            body,
            {
                "account_sid": _account_sid,
                "auth_token": _auth_token,
                "from_number": _from_number,
                "messaging_service_sid": _messaging_service_sid,
                "to": to,
            },
        )
        return {
            "source": "twilio",
            "available": True,
            "status": "sent" if ok else "failed",
            "error": "" if ok else error,
            "sid": sid,
        }


twilio_notify = TwilioNotifyTool()
