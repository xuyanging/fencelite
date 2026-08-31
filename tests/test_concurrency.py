"""Regression tests for the cross-thread/process capacity primitives.

These tests are deliberately independent of ``job.py`` so a scheduler change
cannot accidentally mock away the file-lock contract which protects gunicorn,
restart recovery, and the line-type maintenance process from one another.
"""
from __future__ import annotations

import multiprocessing
import os
import queue
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.concurrency import SlotPool, SlotWaitCancelled, _fcntl_module


def _process_slot_worker(root, name, capacity, ready, release, index):
    pool = SlotPool(Path(root), name, capacity)
    with pool.slot():
        ready.put(("enter", index))
        release.wait(5)
        ready.put(("exit", index))


def _crashing_slot_holder(root, ready):
    pool = SlotPool(Path(root), "crash", 1)
    with pool.slot():
        ready.set()
        # Event state is shared memory, but leave a short window for the parent
        # to observe it before bypassing all Python cleanup on purpose.
        time.sleep(0.05)
        os._exit(23)


def _slot_after_crash(root, connection):
    pool = SlotPool(Path(root), "crash", 1)
    started = time.monotonic()
    with pool.slot():
        connection.send(time.monotonic() - started)
    connection.close()


def _all_slots_worker(root, ready, start, events, index):
    pool = SlotPool(Path(root), "drain", 2)
    ready.put(index)
    start.wait(5)
    with pool.all_slots():
        events.put(("enter", index))
        time.sleep(0.1)
        # Publish this while the barrier is still owned.  A second "enter"
        # before this event therefore proves overlapping all_slots sections.
        events.put(("exit", index))


@unittest.skipUnless(
    _fcntl_module() is not None and "fork" in multiprocessing.get_all_start_methods(),
    "cross-process flock tests require POSIX fork",
)
class ProcessSlotPoolTests(unittest.TestCase):
    def setUp(self):
        self.context = multiprocessing.get_context("fork")
        self.processes = []
        self.addCleanup(self._stop_processes)

    def _start(self, target, *args):
        process = self.context.Process(target=target, args=args)
        process.start()
        self.processes.append(process)
        return process

    def _stop_processes(self):
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=2)

    def _join_cleanly(self, processes, timeout=3):
        for process in processes:
            process.join(timeout=timeout)
        stuck = [process.pid for process in processes if process.is_alive()]
        self.assertEqual(stuck, [], f"child processes did not finish: {stuck}")
        self.assertEqual([process.exitcode for process in processes],
                         [0] * len(processes))

    def test_capacity_two_is_global_across_processes(self):
        with tempfile.TemporaryDirectory() as root:
            events = self.context.Queue()
            release = self.context.Event()
            processes = [
                self._start(
                    _process_slot_worker, root, "project", 2,
                    events, release, index)
                for index in range(4)
            ]

            first = [events.get(timeout=3), events.get(timeout=3)]
            self.assertEqual([kind for kind, _index in first],
                             ["enter", "enter"])
            with self.assertRaises(queue.Empty):
                events.get(timeout=0.3)

            release.set()
            remaining = [events.get(timeout=3) for _ in range(6)]
            self._join_cleanly(processes)
            combined = first + remaining
            self.assertEqual(sum(kind == "enter" for kind, _ in combined), 4)
            self.assertEqual(sum(kind == "exit" for kind, _ in combined), 4)

    def test_holder_os_exit_releases_file_slot(self):
        with tempfile.TemporaryDirectory() as root:
            ready = self.context.Event()
            crashing = self._start(_crashing_slot_holder, root, ready)
            self.assertTrue(ready.wait(timeout=3))
            crashing.join(timeout=3)
            self.assertFalse(crashing.is_alive())
            self.assertEqual(crashing.exitcode, 23)

            receiver, sender = self.context.Pipe(duplex=False)
            follower = self._start(_slot_after_crash, root, sender)
            sender.close()
            self.assertTrue(receiver.poll(3), "released slot was not reusable")
            elapsed = receiver.recv()
            receiver.close()
            self._join_cleanly([follower])
            self.assertLess(elapsed, 1.0)

    def test_two_all_slots_barriers_serialize_without_deadlock(self):
        with tempfile.TemporaryDirectory() as root:
            ready = self.context.Queue()
            start = self.context.Event()
            events = self.context.Queue()
            processes = [
                self._start(
                    _all_slots_worker, root, ready, start, events, index)
                for index in range(2)
            ]
            self.assertEqual({ready.get(timeout=3), ready.get(timeout=3)},
                             {0, 1})
            start.set()
            observed = [events.get(timeout=3) for _ in range(4)]
            self._join_cleanly(processes)
            self.assertEqual([kind for kind, _index in observed],
                             ["enter", "exit", "enter", "exit"])
            self.assertNotEqual(observed[0][1], observed[2][1])


class ThreadSlotPoolTests(unittest.TestCase):
    def test_capacity_two_completes_thirty_tasks_without_exceeding_cap(self):
        with tempfile.TemporaryDirectory() as root:
            pool = SlotPool(Path(root), "project", 2)
            lock = threading.Lock()
            release_first_pair = threading.Event()
            pair_entered = threading.Event()
            state = {"live": 0, "peak": 0, "completed": 0}

            def work():
                with pool.slot():
                    with lock:
                        state["live"] += 1
                        state["peak"] = max(state["peak"], state["live"])
                        if state["live"] == 2:
                            pair_entered.set()
                    release_first_pair.wait(3)
                    time.sleep(0.005)
                    with lock:
                        state["live"] -= 1
                        state["completed"] += 1

            with ThreadPoolExecutor(max_workers=30) as executor:
                futures = [executor.submit(work) for _ in range(30)]
                self.assertTrue(pair_entered.wait(timeout=2))
                release_first_pair.set()
                for future in futures:
                    future.result(timeout=5)

            self.assertEqual(state,
                             {"live": 0, "peak": 2, "completed": 30})

    def test_waiting_for_a_slot_can_be_cancelled(self):
        with tempfile.TemporaryDirectory() as root:
            pool = SlotPool(Path(root), "project", 1)
            cancelled = threading.Event()
            waiting = threading.Event()

            def wait_for_slot():
                waiting.set()
                with pool.slot(cancelled=cancelled.is_set):
                    raise AssertionError("cancelled waiter entered the slot")

            with pool.slot():
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(wait_for_slot)
                    self.assertTrue(waiting.wait(timeout=1))
                    time.sleep(0.05)
                    cancelled.set()
                    with self.assertRaises(SlotWaitCancelled):
                        future.result(timeout=2)


if __name__ == "__main__":
    unittest.main()
