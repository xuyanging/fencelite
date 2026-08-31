"""Crash-safe concurrency primitives shared by web and maintenance workers."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class SlotWaitCancelled(RuntimeError):
    """Raised when cooperative cancellation wins while waiting for capacity."""


def shared_capacity_directory():
    """Stable directory used by provider-wide limits across processes."""
    configured = os.environ.get("FENCE_CAPACITY_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parent.parent / "_jobs" / ".capacity-slots"


def _fcntl_module():
    try:
        import fcntl                                      # noqa: PLC0415
        return fcntl
    except ImportError:
        return None


class SlotPool:
    """A bounded semaphore whose capacity is also enforced across processes.

    Every slot has a stable lock-file inode.  POSIX ``flock`` makes graceful
    gunicorn replacement and the line-type refresh service share the same
    capacity; the in-process semaphore supplies the equivalent contract on
    platforms without flock.
    """

    def __init__(self, directory, name, capacity):
        self.directory = Path(directory)
        self.name = str(name)
        self.capacity = max(1, int(capacity))
        self._local = threading.BoundedSemaphore(self.capacity)

    def _take_local(self, count, cancelled=None):
        taken = 0
        try:
            while taken < count:
                if cancelled and cancelled():
                    raise SlotWaitCancelled()
                if self._local.acquire(timeout=0.2):
                    taken += 1
            return taken
        except BaseException:
            for _ in range(taken):
                self._local.release()
            raise

    def _take_files(self, count, cancelled=None):
        fcntl = _fcntl_module()
        if fcntl is None:
            return []
        self.directory.mkdir(parents=True, exist_ok=True)
        held = {}
        try:
            while len(held) < count:
                if cancelled and cancelled():
                    raise SlotWaitCancelled()
                for index in range(self.capacity):
                    if index in held:
                        continue
                    handle = (self.directory /
                              f"{self.name}.{index}.lock").open("a+b")
                    try:
                        fcntl.flock(handle.fileno(),
                                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        handle.close()
                        continue
                    held[index] = handle
                    if len(held) >= count:
                        break
                if len(held) < count:
                    time.sleep(0.1)
            return list(held.values())
        except BaseException:
            self._release_files(held.values())
            raise

    @staticmethod
    def _release_files(handles):
        fcntl = _fcntl_module()
        for handle in reversed(list(handles)):
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    @contextmanager
    def slot(self, cancelled=None):
        # Serialize allocation (not execution).  This lets all_slots() stop new
        # entrants and drain existing holders without two drainers each keeping
        # a partial set of semaphore/file slots forever.
        with stable_named_lock(
                self.directory / f"{self.name}.allocator.lock",
                cancelled=cancelled):
            local = self._take_local(1, cancelled)
            try:
                handles = self._take_files(1, cancelled)
            except BaseException:
                for _ in range(local):
                    self._local.release()
                raise
        try:
            yield
        finally:
            self._release_files(handles)
            for _ in range(local):
                self._local.release()

    @contextmanager
    def all_slots(self):
        """Drain the whole pool for a restart/recovery consistency barrier."""
        with stable_named_lock(
                self.directory / f"{self.name}.allocator.lock"):
            local = self._take_local(self.capacity)
            try:
                handles = self._take_files(self.capacity)
            except BaseException:
                for _ in range(local):
                    self._local.release()
                raise
            try:
                yield
            finally:
                self._release_files(handles)
                for _ in range(local):
                    self._local.release()


_NAMED_LOCKS = {}
_NAMED_LOCKS_GUARD = threading.Lock()


@contextmanager
def stable_named_lock(path, cancelled=None):
    """Serialize one stable identity in-process and across processes."""
    if cancelled and cancelled():
        raise SlotWaitCancelled()
    path = Path(path)
    key = str(path.absolute())
    with _NAMED_LOCKS_GUARD:
        local = _NAMED_LOCKS.setdefault(key, threading.Lock())
    while not local.acquire(timeout=0.2):
        if cancelled and cancelled():
            raise SlotWaitCancelled()
    handle = None
    fcntl = _fcntl_module()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+b")
        if fcntl is not None:
            while True:
                if cancelled and cancelled():
                    raise SlotWaitCancelled()
                try:
                    fcntl.flock(handle.fileno(),
                                fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.1)
        yield
    finally:
        if handle is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        local.release()
