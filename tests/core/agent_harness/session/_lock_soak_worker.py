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


def _session(session_id: str) -> Any:
    """Minimal stand-in for the persistence source the store reads."""
    return SimpleNamespace(
        session_id=session_id,
        started_at=0.0,
        agent=SimpleNamespace(messages=[]),
        accumulated_context={},
    )


def _wait_until(start_at: float) -> None:
    """Block until ``start_at`` so sibling workers contend instead of queueing."""
    delay = start_at - time.time()
    if delay > 0:
        time.sleep(delay)


def main(config: dict[str, Any]) -> int:
    worker_id = str(config["worker_id"])
    session_ids: list[str] = list(config["session_ids"])
    turns = int(config["turns"])
    text_chars = int(config.get("text_chars", 32))
    result_path = Path(config["result_path"])

    timeout_seconds = config.get("lock_timeout_seconds")
    if timeout_seconds is not None:
        jsonl_store._SESSION_LOCK_TIMEOUT_SECONDS = float(timeout_seconds)

    store = JsonlSessionStore()
    sessions = {session_id: _session(session_id) for session_id in session_ids}
    body = "x" * text_chars

    _wait_until(float(config.get("start_at", 0.0)))

    written: list[str] = []
    first_write = time.time()
    for index in range(turns):
        session_id = session_ids[index % len(session_ids)]
        marker = f"{worker_id}-{index}"
        store.append_turn(sessions[session_id], "chat", f"{marker}:{body}")
        written.append(f"{session_id}/{marker}")
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
