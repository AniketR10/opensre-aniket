"""Tests for the local-host pub/sub agent bus over a Unix-domain socket."""

from __future__ import annotations

import queue
import socket
import threading
import time
from pathlib import Path

import pytest

from app.agents import bus as bus_module
from app.agents.bus import (
    BUS_SCHEMA_VERSION,
    BusMessage,
    BusServer,
    _pid_file_for,
    _read_broker_pid,
    _socket_is_live,
    publish,
    subscribe,
)


@pytest.fixture
def sock_path(tmp_path: Path) -> Path:
    return tmp_path / "bus.sock"


@pytest.fixture(autouse=True)
def isolated_brokers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure each test starts with an empty broker registry, no cross-test bleed."""
    monkeypatch.setattr(bus_module, "_brokers", {})


def _drain_subscriber(
    sock_path: Path, into: queue.Queue[BusMessage], stop_after: int = 1
) -> threading.Thread:
    """Spawn a daemon thread that puts up to ``stop_after`` messages into ``into``."""

    def _loop() -> None:
        for count, msg in enumerate(subscribe(path=sock_path), start=1):
            into.put(msg)
            if count >= stop_after:
                return

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    # Give the subscriber a moment to bind/connect before tests publish.
    time.sleep(0.15)
    return thread


class TestBusMessage:
    def test_round_trip_preserves_all_fields(self) -> None:
        original = BusMessage(
            agent="claude-code:8421",
            topic="finding",
            summary="null deref on missing token",
            source="github",
            path="services/auth.py:42",
            data={"commit": "abc", "line": 42},
        )
        decoded = BusMessage.from_jsonl(original.to_jsonl())
        assert decoded == original

    def test_to_jsonl_ends_with_newline(self) -> None:
        msg = BusMessage(agent="a:1", topic="finding", summary="x")
        assert msg.to_jsonl().endswith(b"\n")

    def test_from_jsonl_supplies_defaults_for_optional_fields(self) -> None:
        # Minimum required fields only — source/path/data should default.
        wire = b'{"agent":"a:1","topic":"finding","summary":"x"}\n'
        msg = BusMessage.from_jsonl(wire)
        assert msg.source == ""
        assert msg.path == ""
        assert msg.data == {}
        assert msg.schema_version == BUS_SCHEMA_VERSION

    def test_from_jsonl_rejects_non_object_payload(self) -> None:
        with pytest.raises(ValueError):
            BusMessage.from_jsonl(b'["not","an","object"]')

    def test_from_jsonl_rejects_missing_required_field(self) -> None:
        with pytest.raises(KeyError):
            BusMessage.from_jsonl(b'{"agent":"a:1","topic":"finding"}')

    def test_is_not_hashable(self) -> None:
        # ``data`` is a mapping, so a value hash would be misleading and would
        # fail at call time on the auto-generated ``__hash__``. Disable it
        # explicitly so callers see the unhashability up front.
        msg = BusMessage(agent="a:1", topic="finding", summary="x", data={"k": 1})
        with pytest.raises(TypeError):
            hash(msg)

    def test_data_is_read_only_post_construction(self) -> None:
        msg = BusMessage(agent="a:1", topic="finding", summary="x", data={"k": 1})
        with pytest.raises(TypeError):
            msg.data["k"] = 2  # type: ignore[index]

    def test_data_is_isolated_from_caller_mutation(self) -> None:
        # External mutation of the originally-passed dict must not bleed into
        # the message — defensive copy in ``__post_init__`` enforces this.
        src: dict[str, object] = {"k": 1}
        msg = BusMessage(agent="a:1", topic="finding", summary="x", data=src)
        src["k"] = 999
        assert msg.data["k"] == 1


class TestBusServerLifecycle:
    def test_start_binds_socket_and_stop_unlinks(self, sock_path: Path) -> None:
        server = BusServer(sock_path)
        try:
            server.start()
            assert sock_path.exists()
            assert _socket_is_live(sock_path)
            assert server.is_running
        finally:
            server.stop()
        assert not sock_path.exists()
        assert not server.is_running

    def test_start_is_idempotent(self, sock_path: Path) -> None:
        server = BusServer(sock_path)
        try:
            server.start()
            server.start()  # second call should be a no-op, not raise
            assert server.is_running
        finally:
            server.stop()

    def test_stop_is_idempotent(self, sock_path: Path) -> None:
        server = BusServer(sock_path)
        server.start()
        server.stop()
        server.stop()  # second call should be a no-op, not raise

    def test_start_writes_pid_file_and_stop_removes_it(self, sock_path: Path) -> None:
        import os

        server = BusServer(sock_path)
        server.start()
        try:
            pid_path = _pid_file_for(sock_path)
            assert pid_path.exists()
            assert _read_broker_pid(sock_path) == os.getpid()
        finally:
            server.stop()
        assert not _pid_file_for(sock_path).exists()


class TestLivenessProbe:
    def test_socket_is_live_does_not_create_phantom_subscriber(self, sock_path: Path) -> None:
        # _socket_is_live used to make a real connection on every probe; under
        # publish/subscribe bursts that registered a short-lived subscriber +
        # reader thread per call. Verify the side-channel probe makes none.
        server = BusServer(sock_path)
        server.start()
        try:
            for _ in range(20):
                assert _socket_is_live(sock_path)
            with server._lock:
                assert len(server._subscribers) == 0
        finally:
            server.stop()

    def test_socket_is_live_false_when_pid_missing(self, sock_path: Path) -> None:
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock_path.touch()  # socket file present but no pid sidecar
        assert not _socket_is_live(sock_path)

    def test_socket_is_live_false_when_pid_dead(self, sock_path: Path) -> None:
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock_path.touch()
        # PID 999999 is almost certainly not a real process. The probe must
        # report not-live so the caller will unlink + rebind.
        _pid_file_for(sock_path).write_text("999999")
        assert not _socket_is_live(sock_path)


class TestPublishSubscribe:
    def test_round_trip_one_publisher_one_subscriber(self, sock_path: Path) -> None:
        received: queue.Queue[BusMessage] = queue.Queue()
        _drain_subscriber(sock_path, received)

        publish(
            BusMessage(
                agent="claude-code:8421",
                topic="finding",
                summary="null deref",
                path="services/auth.py:42",
            ),
            path=sock_path,
        )

        msg = received.get(timeout=2.0)
        assert msg.agent == "claude-code:8421"
        assert msg.summary == "null deref"
        assert msg.path == "services/auth.py:42"

    def test_one_publisher_multiple_subscribers_all_receive(self, sock_path: Path) -> None:
        n_subs = 3
        sub_queues: list[queue.Queue[BusMessage]] = [queue.Queue() for _ in range(n_subs)]
        for q in sub_queues:
            _drain_subscriber(sock_path, q)

        publish(BusMessage(agent="a:1", topic="finding", summary="hello"), path=sock_path)

        for q in sub_queues:
            msg = q.get(timeout=2.0)
            assert msg.summary == "hello"

    def test_publisher_does_not_receive_own_frame(self, sock_path: Path) -> None:
        # Self-elect a broker so we can attach as a single client that
        # both publishes and subscribes — and verify the broadcaster
        # does not echo the frame back to its origin.
        bus_module._ensure_broker(sock_path)
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(str(sock_path))
        client.settimeout(0.5)
        try:
            client.sendall(BusMessage(agent="a:1", topic="finding", summary="x").to_jsonl())
            with pytest.raises(socket.timeout):
                client.recv(4096)
        finally:
            client.close()

    def test_subscribe_skips_malformed_frames(self, sock_path: Path) -> None:
        received: queue.Queue[BusMessage] = queue.Queue()
        _drain_subscriber(sock_path, received)

        # Connect a raw publisher and inject a malformed frame followed by a valid one.
        bus_module._ensure_broker(sock_path)
        raw = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        raw.connect(str(sock_path))
        try:
            raw.sendall(b"this-is-not-json\n")
            raw.sendall(BusMessage(agent="a:1", topic="finding", summary="ok").to_jsonl())
        finally:
            # Keep the publisher alive briefly so the broker can forward both frames.
            time.sleep(0.1)
            raw.close()

        msg = received.get(timeout=2.0)
        assert msg.summary == "ok"

    def test_subscriber_disconnects_on_oversized_unterminated_stream(self, sock_path: Path) -> None:
        # Simulate a hostile broker that wins the bind race and streams unlimited
        # bytes without newlines. The subscriber must cap its buffer and bail
        # rather than grow memory unboundedly.
        from app.agents.bus import _MAX_FRAME_BYTES, BusServer, subscribe

        # Stand up a fake "hostile" broker that pushes garbage to every client.
        server = BusServer(sock_path)
        server.start()
        try:
            done = threading.Event()
            error: list[BaseException] = []

            def _consume() -> None:
                try:
                    # Drain until subscribe() returns (which it should, on cap breach).
                    for _ in subscribe(path=sock_path):
                        pass
                except BaseException as exc:  # noqa: BLE001
                    error.append(exc)
                finally:
                    done.set()

            t = threading.Thread(target=_consume, daemon=True)
            t.start()
            # Wait for the subscriber to attach to the broker.
            deadline = time.time() + 2.0
            while time.time() < deadline:
                with server._lock:
                    if server._subscribers:
                        break
                time.sleep(0.02)
            assert server._subscribers, "subscriber never attached"

            # Push a single oversized chunk through to all subscribers.
            payload = b"X" * (_MAX_FRAME_BYTES * 4 + 1024)
            with server._lock:
                victim = next(iter(server._subscribers))
            victim.sendall(payload)

            # Subscriber should disconnect on its own without raising.
            assert done.wait(timeout=2.0), "subscriber did not disconnect on cap breach"
            assert not error, f"subscribe() raised unexpectedly: {error}"
        finally:
            server.stop()


class TestBrokerSelfElection:
    def test_stale_socket_file_is_unlinked_and_rebound(self, sock_path: Path) -> None:
        sock_path.parent.mkdir(parents=True, exist_ok=True)
        sock_path.touch()  # stale: no listener
        assert sock_path.exists() and not _socket_is_live(sock_path)

        server = bus_module._ensure_broker(sock_path)
        try:
            assert server is not None
            assert server.is_running
            assert _socket_is_live(sock_path)
        finally:
            if server is not None:
                server.stop()

    def test_ensure_broker_returns_none_when_other_process_owns_it(self, sock_path: Path) -> None:
        # Stand up an "other" broker via direct BusServer (simulating another process),
        # then call _ensure_broker — it should detect the live socket and back off.
        other = BusServer(sock_path)
        other.start()
        try:
            # Pretend our process has not yet elected: empty the registry.
            bus_module._brokers.clear()
            result = bus_module._ensure_broker(sock_path)
            assert result is None
        finally:
            other.stop()

    def test_ensure_broker_idempotent_for_same_process(self, sock_path: Path) -> None:
        first = bus_module._ensure_broker(sock_path)
        second = bus_module._ensure_broker(sock_path)
        try:
            assert first is not None
            assert first is second
        finally:
            if first is not None:
                first.stop()


class TestSlashCommandFormatter:
    def test_format_includes_agent_and_summary(self) -> None:
        from app.cli.interactive_shell.command_registry.agents import _format_bus_message

        msg = BusMessage(agent="claude-code:8421", topic="finding", summary="hi")
        out = _format_bus_message(msg)
        assert "claude-code:8421" in out
        assert "hi" in out

    def test_format_includes_path_when_present(self) -> None:
        from app.cli.interactive_shell.command_registry.agents import _format_bus_message

        msg = BusMessage(agent="a:1", topic="finding", summary="x", path="services/auth.py:42")
        out = _format_bus_message(msg)
        assert "services/auth.py:42" in out
        assert "—" in out

    def test_format_omits_separator_when_no_path(self) -> None:
        from app.cli.interactive_shell.command_registry.agents import _format_bus_message

        msg = BusMessage(agent="a:1", topic="finding", summary="x")
        out = _format_bus_message(msg)
        assert "—" not in out
