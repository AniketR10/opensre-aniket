"""Tests for app/tools/TwilioNotifyTool — SMS notification surface."""

from __future__ import annotations

from typing import Any

import pytest

from app.tools.TwilioNotifyTool import TwilioNotifyTool, twilio_notify


@pytest.fixture
def twilio_source() -> dict[str, Any]:
    return {
        "twilio": {
            "account_sid": "AC1",
            "auth_token": "tok",
            "sms": {
                "enabled": True,
                "from_number": "+14155551111",
                "messaging_service_sid": "",
                "default_to": "+14155550000",
            },
            "whatsapp": {"enabled": False},
        }
    }


def test_metadata_declares_twilio_source() -> None:
    metadata = TwilioNotifyTool.metadata()
    assert metadata.name == "twilio_notify"
    assert metadata.source == "twilio"


def test_is_available_true_when_sms_configured(
    twilio_source: dict[str, Any],
) -> None:
    assert twilio_notify.is_available(twilio_source) is True


def test_is_available_false_when_no_twilio() -> None:
    assert twilio_notify.is_available({}) is False


def test_is_available_false_when_sms_disabled(
    twilio_source: dict[str, Any],
) -> None:
    twilio_source["twilio"]["sms"]["enabled"] = False
    assert twilio_notify.is_available(twilio_source) is False


def test_is_available_false_when_no_sender(
    twilio_source: dict[str, Any],
) -> None:
    twilio_source["twilio"]["sms"]["from_number"] = ""
    twilio_source["twilio"]["sms"]["messaging_service_sid"] = ""
    assert twilio_notify.is_available(twilio_source) is False


def test_is_available_true_with_only_messaging_service(
    twilio_source: dict[str, Any],
) -> None:
    twilio_source["twilio"]["sms"]["from_number"] = ""
    twilio_source["twilio"]["sms"]["messaging_service_sid"] = "MG1"
    assert twilio_notify.is_available(twilio_source) is True


def test_extract_params_uses_default_recipient(
    twilio_source: dict[str, Any],
) -> None:
    params = twilio_notify.extract_params(twilio_source)
    assert params["to"] == "+14155550000"
    assert params["_account_sid"] == "AC1"
    assert params["_from_number"] == "+14155551111"


def test_run_dispatches_via_send_twilio_sms_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_send(report: str, ctx: dict[str, Any]) -> tuple[bool, str, str]:
        captured["report"] = report
        captured["ctx"] = ctx
        return True, "", "SM-SENT"

    monkeypatch.setattr("app.tools.TwilioNotifyTool.send_twilio_sms_report", _fake_send)

    result = twilio_notify.run(
        body="page on-call",
        to="+14155550000",
        _account_sid="AC1",
        _auth_token="tok",
        _from_number="+14155551111",
        _messaging_service_sid="",
    )

    assert result["status"] == "sent"
    assert result["sid"] == "SM-SENT"
    assert result["error"] == ""
    assert captured["report"] == "page on-call"
    assert captured["ctx"]["to"] == "+14155550000"


def test_run_returns_failed_when_missing_recipient() -> None:
    result = twilio_notify.run(
        body="hi",
        to="",
        _account_sid="AC1",
        _auth_token="tok",
        _from_number="+14155551111",
    )
    assert result["status"] == "failed"
    assert "recipient" in result["error"].lower()


def test_run_returns_failed_without_credentials() -> None:
    result = twilio_notify.run(body="hi", to="+1", _account_sid="", _auth_token="")
    assert result["status"] == "failed"
    assert "not configured" in result["error"].lower()


def test_run_propagates_send_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.tools.TwilioNotifyTool.send_twilio_sms_report",
        lambda _r, _c: (False, "twilio rejected", ""),
    )

    result = twilio_notify.run(
        body="hi",
        to="+14155550000",
        _account_sid="AC1",
        _auth_token="tok",
        _from_number="+14155551111",
    )
    assert result["status"] == "failed"
    assert result["error"] == "twilio rejected"
    assert result["sid"] == ""
