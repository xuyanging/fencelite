"""Disk contract for the low-priority line-type refresh worker.

The refresh worker is deliberately a separate process from gunicorn.  Its
status therefore cannot live in ``job.JOBS``; the web process needs one small,
atomic file that it can read without sharing locks or trusting a half-written
heartbeat.

Layout::

    _jobs/linetype_refresh/state.json
    _jobs/linetype_refresh/worker.lock

Keeping the files in a subdirectory is important.  ``job.resume_interrupted``
loads ``_jobs/*.json`` as upload cards on startup, so a top-level status JSON
would be mistaken for a resumable project.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from steps import store
from steps.legend_linetypes import sidecar as legend_sidecar
from steps.linetypes import sidecar

STATE_SCHEMA = 1
STATE_KIND = "linetype_refresh"
DIRECTORY_NAME = "linetype_refresh"
DEFAULT_STALE_AFTER = 45.0

__all__ = [
    "DEFAULT_STALE_AFTER",
    "RefreshStateWriter",
    "STATE_KIND",
    "STATE_SCHEMA",
    "WorkerAlreadyRunning",
    "current_engine_short",
    "is_fresh",
    "load_state",
    "page_refresh_status",
    "state_path",
    "worker_lock",
    "worker_lock_path",
]


def _directory() -> Path:
    # Resolve JOBS_DIR at call time.  Tests and offline tools patch the store
    # module's directory constants; binding it at import time would bypass the
    # patch and write into the real workspace.
    return Path(store.JOBS_DIR) / DIRECTORY_NAME


def state_path() -> Path:
    return _directory() / "state.json"


def worker_lock_path() -> Path:
    return _directory() / "worker.lock"


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat(
        timespec="seconds").replace("+00:00", "Z")


def load_state(path: str | os.PathLike | None = None):
    """Return the latest complete refresh status, or ``None`` when absent.

    ``store.load_json`` intentionally rejects malformed/truncated JSON rather
    than returning a partial object.  The writer uses the same directory-local
    atomic replace contract as every other cache in this service.
    """
    value = store.load_json(Path(path) if path is not None else state_path(),
                            None)
    return value if isinstance(value, dict) else None


def is_fresh(state, *, now: float | None = None,
             stale_after: float = DEFAULT_STALE_AFTER) -> bool:
    """Whether a state belongs to a live/recently-live worker heartbeat."""
    if not isinstance(state, dict):
        return False
    try:
        heartbeat = float(state.get("heartbeat_at"))
    except (TypeError, ValueError):
        return False
    current = time.time() if now is None else float(now)
    return 0.0 <= current - heartbeat <= max(0.0, float(stale_after))


def current_engine_short() -> str:
    """Combined producer identity for signature-free public queue rows.

    Queue/active rows intentionally omit their full (potentially bulky and
    input-specific) channel signature.  The worker repairs both ordinary
    arrow-bound caches and supervised legend caches, so the heartbeat must be
    invalidated when *either* producer changes.  Otherwise a web process from
    a new deploy could mistake an old worker's legend row for current work.
    """
    payload = {
        "arrow": str(sidecar.engine_digest()),
        "legend": str(legend_sidecar.producer_digest()),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def page_refresh_status(slug, page, state=None, *, channel=None):
    """Return ``queued``, ``running``, ``waiting``, or ``None`` for a page.

    Only a fresh status file from the current line-type engine is trusted.
    An active row wins over a queued row and remains ``running`` even when the
    worker is temporarily in its global ``waiting`` phase (already-started
    pages are deliberately allowed to finish).  A queued row is ``waiting``
    while foreground upload/rerun cards pause new starts, otherwise ``queued``.

    ``channel`` may be ``"arrow"`` or ``"legend"`` to disambiguate the two
    independent cache jobs on the same sheet.  Omitting it retains the legacy
    page-level view and reports activity in either channel.
    """
    value = load_state() if state is None else state
    if (not isinstance(value, dict)
            or value.get("schema") != STATE_SCHEMA
            or value.get("kind") != STATE_KIND
            or not is_fresh(value)
            or value.get("engine") != current_engine_short()):
        return None
    if not store.is_valid_slug(slug):
        return None
    try:
        wanted_page = int(page)
    except (TypeError, ValueError):
        return None
    if wanted_page < 1:
        return None
    if channel is not None:
        channel = str(channel).strip().lower()
        if channel not in ("arrow", "legend"):
            return None

    def matches(row):
        if not isinstance(row, dict) or row.get("slug") != slug:
            return False
        if channel is not None and row.get("channel") != channel:
            return False
        try:
            return int(row.get("page")) == wanted_page
        except (TypeError, ValueError):
            return False

    if any(matches(row) for row in value.get("active") or ()):
        return "running"
    if any(matches(row) for row in value.get("queued") or ()):
        return "waiting" if value.get("phase") == "waiting" else "queued"
    return None


class RefreshStateWriter:
    """Thread-safe, atomic top-level updates for the worker status file."""

    def __init__(self, path: str | os.PathLike | None = None, *, clock=None):
        self.path = Path(path) if path is not None else state_path()
        self._clock = clock or time.time
        self._lock = threading.Lock()
        self._state = {}

    def replace(self, payload):
        if not isinstance(payload, dict):
            raise TypeError("refresh state payload must be a dict")
        with self._lock:
            self._state = dict(payload)
            return self._write_locked()

    def update(self, **changes):
        with self._lock:
            self._state.update(changes)
            return self._write_locked()

    def snapshot(self):
        with self._lock:
            return dict(self._state)

    def _write_locked(self):
        now = float(self._clock())
        self._state.update({
            "schema": STATE_SCHEMA,
            "kind": STATE_KIND,
            "heartbeat_at": now,
            "heartbeat": _iso(now),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        store.save_json(self.path, self._state)
        return dict(self._state)


class WorkerAlreadyRunning(RuntimeError):
    pass


@contextmanager
def worker_lock(path: str | os.PathLike | None = None):
    """Take the singleton worker lock without waiting.

    The lock is advisory and process-scoped.  A crash automatically releases
    it, unlike a PID/sentinel file which can strand the refresh forever after a
    hard kill.  Both supported deployment families are covered; unusual
    platforms fail closed instead of allowing two writers.
    """
    target = Path(path) if path is not None else worker_lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = open(target, "a+b")  # noqa: SIM115 - held for the context lifetime
    unlock = None
    try:
        if os.name == "nt":
            import msvcrt                                      # noqa: PLC0415

            if target.stat().st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise WorkerAlreadyRunning(
                    "another line-type refresh worker holds the lock") from exc

            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            try:
                import fcntl                                  # noqa: PLC0415
                fcntl.flock(handle.fileno(),
                            fcntl.LOCK_EX | fcntl.LOCK_NB)
            except (ImportError, BlockingIOError, OSError) as exc:
                raise WorkerAlreadyRunning(
                    "another line-type refresh worker holds the lock") from exc

            def unlock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

        yield target
    finally:
        if unlock is not None:
            try:
                unlock()
            except OSError:
                pass
        handle.close()
