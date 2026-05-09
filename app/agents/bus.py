"""Local-host pub/sub bus for cross-agent findings over a Unix-domain socket.

Carries the same shape as ``app/state/agent_state.py``'s ``evidence`` records so
findings published by one agent (claude-code, cursor, aider, ...) can later be
lifted into ``AgentState.evidence`` without re-mapping fields. See
``docs/agents.mdx`` for the on-the-wire schema.

Topology is a self-electing broker: the first ``publish`` or ``subscribe`` call
that finds no live socket binds it and runs an in-process daemon thread that
fans incoming JSONL messages out to every connected subscriber. Other processes
attach as plain clients. If the broker dies, the next operation re-elects.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import types
import uuid
from collections.abc import Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from app.constants import OPENSRE_HOME_DIR

logger = logging.getLogger(__name__)

DEFAULT_BUS_SOCKET_PATH: Path = OPENSRE_HOME_DIR / "agents-bus.sock"

#: Bus message wire-format version. Bump when ``BusMessage`` fields change shape.
BUS_SCHEMA_VERSION: int = 1

#: Max bytes per JSONL frame on the wire. Frames over this are dropped with a
#: warning; a finding payload that big is almost certainly a bug.
_MAX_FRAME_BYTES: int = 64 * 1024


@dataclass(frozen=True)
class BusMessage:
    """A single finding published on the agent bus.

    Field shape mirrors ``AgentState.evidence`` entries so a message can be
    folded into investigation state without renaming. ``agent`` follows the
    ``"<name>:<pid>"`` convention used by ``app.agents.conflicts.WriteEvent``.

    ``data`` is wrapped in ``types.MappingProxyType`` at construction so the
    payload is read-only post-init; mutating ``msg.data["x"] = 1`` raises
    ``TypeError``. ``__hash__`` is explicitly disabled because ``data`` is a
    mapping and would otherwise produce a misleading auto-generated hash that
    fails at call time.
    """

    agent: str
    topic: str
    summary: str
    source: str = ""
    path: str = ""
    data: Mapping[str, object] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    schema_version: int = BUS_SCHEMA_VERSION

    # Disable hashing: a BusMessage carries a mapping and is not a value-key.
    __hash__ = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        # Defensive copy + read-only view: protects against both external
        # mutation of the original dict and ``msg.data["x"] = 1`` after
        # construction. ``object.__setattr__`` bypasses the frozen check.
        object.__setattr__(self, "data", types.MappingProxyType(dict(self.data)))

    def to_jsonl(self) -> bytes:
        """Encode as a single newline-terminated JSON frame ready for the socket."""
        payload = {
            "agent": self.agent,
            "topic": self.topic,
            "summary": self.summary,
            "source": self.source,
            "path": self.path,
            "data": dict(self.data),
            "id": self.id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
        }
        return (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")

    @classmethod
    def from_jsonl(cls, line: bytes | str) -> BusMessage:
        """Decode one JSONL frame into a ``BusMessage``. Raises on malformed input."""
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("bus frame must be a JSON object")
        return cls(
            agent=str(data["agent"]),
            topic=str(data["topic"]),
            summary=str(data["summary"]),
            source=str(data.get("source", "")),
            path=str(data.get("path", "")),
            data=dict(data.get("data", {})),
            id=str(data.get("id", uuid.uuid4())),
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
            schema_version=int(data.get("schema_version", BUS_SCHEMA_VERSION)),
        )


def _pid_file_for(socket_path: Path) -> Path:
    """Return the sidecar PID-file path for a given bus socket path."""
    return socket_path.with_name(socket_path.name + ".pid")


def _read_broker_pid(socket_path: Path) -> int | None:
    """Read the broker PID from the sidecar file, or ``None`` if missing/garbled."""
    pid_path = _pid_file_for(socket_path)
    try:
        text = pid_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _process_is_alive(pid: int) -> bool:
    """``os.kill(pid, 0)`` probe: True iff the PID maps to a live process we can signal."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it. Treat as alive — we still can't
        # safely unlink the socket out from under whoever owns it.
        return True
    except OSError:
        return False
    return True


def _socket_is_live(path: Path) -> bool:
    """Return True if a broker is currently listening on ``path``.

    Uses a PID-file side channel rather than connecting to the socket: the
    broker writes its PID on ``start()`` and removes it on ``stop()``. We treat
    the broker as live iff the socket file exists, the PID file exists, and
    the recorded PID maps to a process we can signal. This avoids creating a
    short-lived phantom subscriber + reader thread on every ``publish()`` /
    ``subscribe()`` call by a non-owner process.

    A stale PID file (broker crashed without cleanup) is reported as not-live;
    the caller's ``_unlink_stale`` path will remove the socket file and rebind.
    """
    if not path.exists():
        return False
    pid = _read_broker_pid(path)
    if pid is None:
        return False
    return _process_is_alive(pid)


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)


def _unlink_stale(path: Path) -> None:
    """Remove a socket file (and its sidecar PID file) that has no live listener."""
    with suppress(FileNotFoundError, OSError):
        os.unlink(path)
    with suppress(FileNotFoundError, OSError):
        os.unlink(_pid_file_for(path))


def _write_pid_file_atomic(path: Path, pid: int) -> None:
    """Write ``pid`` to the sidecar atomically (tmpfile + rename)."""
    pid_path = _pid_file_for(path)
    tmp = pid_path.with_name(pid_path.name + ".tmp")
    try:
        tmp.write_text(str(pid), encoding="utf-8")
        with suppress(OSError):
            os.chmod(tmp, 0o600)
        os.replace(tmp, pid_path)
    except OSError:
        with suppress(FileNotFoundError, OSError):
            os.unlink(tmp)
        # PID file is best-effort: bus still works without it, ``_socket_is_live``
        # just falls back to "not live" and a peer might re-elect.
        logger.warning("failed to write bus pid file at %s", pid_path)


class BusServer:
    """In-process broker that fans JSONL frames out to every connected subscriber.

    The first publisher or subscriber on a given socket path elects itself as
    broker by calling ``BusServer(path).start()``. The server runs an accept
    loop and per-connection reader threads as daemons, so the host process
    exits without needing to join them. Subscribers that disconnect or fail to
    receive are removed from the fan-out set on the next broadcast.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._listener: socket.socket | None = None
        self._subscribers: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._accept_thread: threading.Thread | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def start(self) -> None:
        """Bind the socket and spawn the accept loop. Raises ``OSError`` on bind failure."""
        if self._running.is_set():
            return
        _ensure_parent_dir(self._path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self._path))
        except OSError:
            listener.close()
            raise
        with suppress(OSError):
            os.chmod(self._path, 0o600)
        listener.listen(16)
        self._listener = listener
        self._running.set()
        # Publish our PID via the sidecar so peers can answer "is the broker
        # live?" without making a real connection (which would otherwise spawn
        # a short-lived phantom subscriber on every probe).
        _write_pid_file_atomic(self._path, os.getpid())
        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="agents-bus-accept",
            daemon=True,
        )
        self._accept_thread.start()

    def stop(self) -> None:
        """Shut the broker down: close the listener, drop all subscribers, unlink the socket."""
        if not self._running.is_set():
            return
        self._running.clear()
        listener, self._listener = self._listener, None
        if listener is not None:
            with suppress(OSError):
                listener.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                listener.close()
        with self._lock:
            for sub in self._subscribers:
                with suppress(OSError):
                    sub.close()
            self._subscribers.clear()
        _unlink_stale(self._path)

    def _accept_loop(self) -> None:
        listener = self._listener
        if listener is None:
            return
        while self._running.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                # Listener closed during ``stop()`` — exit cleanly.
                return
            conn.setblocking(True)
            with self._lock:
                self._subscribers.add(conn)
            reader = threading.Thread(
                target=self._reader_loop,
                args=(conn,),
                name="agents-bus-reader",
                daemon=True,
            )
            reader.start()

    def _reader_loop(self, conn: socket.socket) -> None:
        """Read newline-delimited frames from one client and broadcast them."""
        buf = b""
        try:
            while self._running.is_set():
                chunk = conn.recv(4096)
                if not chunk:
                    return
                buf += chunk
                if len(buf) > _MAX_FRAME_BYTES * 4:
                    logger.warning("bus client exceeded buffer cap; disconnecting")
                    return
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line:
                        continue
                    if len(line) > _MAX_FRAME_BYTES:
                        logger.warning("dropping oversized bus frame (%d bytes)", len(line))
                        continue
                    self._broadcast(line + b"\n", origin=conn)
        except OSError:
            return
        finally:
            self._drop_subscriber(conn)

    def _broadcast(self, frame: bytes, origin: socket.socket | None) -> None:
        with self._lock:
            targets = list(self._subscribers)
        dead: list[socket.socket] = []
        for sub in targets:
            if sub is origin:
                # Don't echo a publisher's own frame back to itself.
                continue
            try:
                sub.sendall(frame)
            except OSError:
                dead.append(sub)
        for sub in dead:
            self._drop_subscriber(sub)

    def _drop_subscriber(self, conn: socket.socket) -> None:
        with self._lock:
            self._subscribers.discard(conn)
        with suppress(OSError):
            conn.close()


_broker_lock = threading.Lock()
_brokers: dict[Path, BusServer] = {}


def _ensure_broker(path: Path) -> BusServer | None:
    """Elect a broker for ``path`` if none is live, else return ``None``.

    Idempotent per-path: if this process already owns the broker, returns the
    existing instance. If another process owns it, returns ``None`` (the caller
    should connect as a client). If a stale socket file exists, unlinks it and
    retries the bind.
    """
    with _broker_lock:
        existing = _brokers.get(path)
        if existing is not None and existing.is_running:
            return existing
        if _socket_is_live(path):
            return None
        # Path either doesn't exist or is stale. Unlink any leftover and try to bind.
        _unlink_stale(path)
        server = BusServer(path)
        try:
            server.start()
        except OSError:
            # Lost the race to another process between liveness check and bind.
            return None
        _brokers[path] = server
        return server


def _connect_client(path: Path, timeout: float) -> socket.socket:
    """Open a blocking UDS connection to the broker at ``path``."""
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(timeout)
    try:
        client.connect(str(path))
    except OSError:
        with suppress(OSError):
            client.close()
        raise
    client.settimeout(None)
    return client


def publish(
    message: BusMessage,
    *,
    path: Path | None = None,
    connect_timeout: float = 1.0,
) -> None:
    """Publish ``message`` to every current subscriber on the bus.

    Self-elects a broker if none is running. Send is fire-and-forget: if no
    subscribers are attached, the frame is dropped by the broker (live-only,
    no replay buffer in v1).
    """
    target = path or DEFAULT_BUS_SOCKET_PATH
    _ensure_broker(target)
    client = _connect_client(target, timeout=connect_timeout)
    try:
        client.sendall(message.to_jsonl())
    finally:
        with suppress(OSError):
            client.close()


def subscribe(
    *,
    path: Path | None = None,
    connect_timeout: float = 1.0,
) -> Iterator[BusMessage]:
    """Yield ``BusMessage``s as they arrive on the bus until the broker disconnects.

    Self-elects a broker if none is running, then attaches as a subscriber and
    streams frames. Malformed lines are logged at WARNING and skipped — one
    misbehaving publisher should not kill an inspector REPL. The iterator ends
    cleanly on broker disconnect; ``KeyboardInterrupt`` propagates so callers
    (e.g. ``/agents bus``) can return to their prompt.

    A buffer cap mirrors the broker's ``_reader_loop`` guard: any process that
    can ``bind()`` the socket first (filesystem perms are the only auth) could
    otherwise stream unlimited bytes without newlines and exhaust subscriber
    memory. On overflow the subscriber logs a warning and disconnects.
    """
    target = path or DEFAULT_BUS_SOCKET_PATH
    _ensure_broker(target)
    client = _connect_client(target, timeout=connect_timeout)
    buf = b""
    try:
        while True:
            try:
                chunk = client.recv(4096)
            except OSError:
                return
            if not chunk:
                return
            buf += chunk
            if len(buf) > _MAX_FRAME_BYTES * 4:
                logger.warning(
                    "bus broker exceeded subscriber buffer cap (%d bytes); disconnecting",
                    len(buf),
                )
                return
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                if len(line) > _MAX_FRAME_BYTES:
                    logger.warning("dropping oversized bus frame (%d bytes)", len(line))
                    continue
                try:
                    yield BusMessage.from_jsonl(line)
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    logger.warning("dropping malformed bus frame: %s", line[:80])
    finally:
        with suppress(OSError):
            client.close()


__all__ = [
    "BUS_SCHEMA_VERSION",
    "BusMessage",
    "BusServer",
    "DEFAULT_BUS_SOCKET_PATH",
    "publish",
    "subscribe",
]
