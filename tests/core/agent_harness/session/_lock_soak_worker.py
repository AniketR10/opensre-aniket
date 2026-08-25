"""Standalone writer process for the session-lock soak matrix.

Run as a script, never imported by a test: the point is a genuinely separate
interpreter, so ``SIGKILL`` proves what a thread cannot and no monkeypatched
parent state leaks in. Configuration arrives as one JSON argument because the
matrix varies enough parameters that positional argv would be unreadable.

The worker never calls ``open_session``: that truncates an existing file
(#5474), which would destroy the very interleaving these tests measure. The
parent seeds the session file first.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.agent_harness.session.persistence import jsonl_store  # noqa: E402
from core.agent_harness.session.persistence.jsonl_store import JsonlSessionStore  # noqa: E402
from core.agent_harness.session.persistence.paths import session_path  # noqa: E402

_BARRIER_POLL_SECONDS = 0.01
_BARRIER_TIMEOUT_SECONDS = 120.0


def _session(session_id: str) -> Any:
    """Minimal stand-in for the persistence source the store reads."""
    return SimpleNamespace(
        session_id=session_id,
        started_at=0.0,
        agent=SimpleNamespace(messages=[]),
        accumulated_context={},
    )


def _await_go(ready_path: Path, go_path: Path) -> None:
    """Announce readiness, then block until the parent releases every worker.

    A wall-clock start time cannot do this job: interpreter startup is hundreds
    of milliseconds and unbounded on a loaded runner, so a worker can miss a
    fixed deadline and begin after a sibling has already finished. Signalling
    after the imports are paid makes the release order independent of load.
    """
    ready_path.write_text(str(os.getpid()), encoding="utf-8")
    deadline = time.monotonic() + _BARRIER_TIMEOUT_SECONDS
    while not go_path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"barrier {go_path} never opened")
        time.sleep(_BARRIER_POLL_SECONDS)


def _hold_lock_forever(session_id: str, held_path: Path, seconds: float) -> None:
    """Take the store's own write lock, announce it, and keep holding it.

    Through ``_locked`` rather than a ``FileLock`` built from a guessed filename:
    the caller needs to hold *whatever* lock the store uses, so that renaming or
    re-keying the lock cannot leave this quietly holding an unrelated file.

    The marker is written from inside the locked region, so a parent that sees
    it knows the lock is held right now — not that it might be soon.
    """
    store = JsonlSessionStore()
    path = session_path(session_id)
    with store._locked(path):  # noqa: SLF001
        held_path.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(seconds)


def main(config: dict[str, Any]) -> int:
    worker_id = str(config["worker_id"])
    session_ids: list[str] = list(config["session_ids"])
    text_chars = int(config.get("text_chars", 32))
    result_path = Path(config["result_path"])

    # Exactly one of the two is set: ``turns`` for a bounded run, ``duration``
    # when the run must still be going when something else happens to it.
    turns = config.get("turns")
    duration_seconds = config.get("duration_seconds")

    timeout_seconds = config.get("lock_timeout_seconds")
    if timeout_seconds is not None:
        jsonl_store._SESSION_LOCK_TIMEOUT_SECONDS = float(timeout_seconds)

    store = JsonlSessionStore()
    sessions = {session_id: _session(session_id) for session_id in session_ids}
    body = "x" * text_chars

    hold_seconds = config.get("hold_lock_seconds")
    if hold_seconds is not None:
        _hold_lock_forever(session_ids[0], Path(config["held_path"]), float(hold_seconds))
        return 0

    go_path = config.get("go_path")
    if go_path is not None:
        _await_go(Path(f"{result_path}.ready"), Path(go_path))

    written: list[str] = []
    first_write = time.time()
    deadline = time.monotonic() + float(duration_seconds) if duration_seconds else None
    index = 0
    while True:
        if deadline is None and index >= int(turns or 0):
            break
        if deadline is not None and time.monotonic() >= deadline:
            break
        session_id = session_ids[index % len(session_ids)]
        marker = f"{worker_id}-{index}"
        store.append_turn(sessions[session_id], "chat", f"{marker}:{body}")
        written.append(f"{session_id}/{marker}")
        index += 1
    last_write = time.time()

    result_path.write_text(
        json.dumps(
            {
                "worker_id": worker_id,
                "pid": os.getpid(),
                "first_write_ts": first_write,
                "last_write_ts": last_write,
                "written": written,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(json.loads(sys.argv[1])))
