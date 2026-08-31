"""Offline contract tests for the low-priority line-type refresh worker.

No test invokes the line-type sidecar or reads a production PDF.  Production
inventory/recompute entry points are patched at their boundary, while the
state-file tests write only inside a temporary directory.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
import time
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

from steps.linetypes import refresh_state                         # noqa: E402
from tools import refresh_stale_linetypes as tool                 # noqa: E402


class MemoryWriter:
    def __init__(self):
        self.state = {}
        self.history = []

    def replace(self, payload):
        self.state = copy.deepcopy(payload)
        self.history.append(copy.deepcopy(self.state))
        return copy.deepcopy(self.state)

    def update(self, **changes):
        self.state.update(copy.deepcopy(changes))
        self.history.append(copy.deepcopy(self.state))
        return copy.deepcopy(self.state)


def candidate(page=2, *, slug="demo", sig="arrow+eengine|lt5"):
    return tool.Candidate(
        slug=slug,
        page=page,
        items=[{"text": "PROPOSED FENCE", "box_2d": [1, 2, 3, 4]}],
        arrow_entry={"items": {"0": {"targets": [{"tip": [5, 6]}]}}},
        sig=sig,
        cache_state="stale",
    )


class RefreshStateTests(unittest.TestCase):
    def test_writer_atomically_publishes_complete_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "state.json"
            clock = mock.Mock(side_effect=[100.0, 101.0])
            writer = refresh_state.RefreshStateWriter(path, clock=clock)

            first = writer.replace({"engine": "abc", "phase": "inventory"})
            self.assertEqual(first["schema"], refresh_state.STATE_SCHEMA)
            self.assertEqual(first["kind"], refresh_state.STATE_KIND)
            self.assertEqual(first["heartbeat_at"], 100.0)
            self.assertEqual(refresh_state.load_state(path), first)

            second = writer.update(phase="done", ok=True)
            self.assertEqual(second["heartbeat_at"], 101.0)
            self.assertEqual(refresh_state.load_state(path), second)
            self.assertFalse(list(path.parent.glob(".state.json.*.tmp")))

    def test_singleton_lock_rejects_a_second_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.lock"
            with refresh_state.worker_lock(path):
                with self.assertRaises(refresh_state.WorkerAlreadyRunning):
                    with refresh_state.worker_lock(path):
                        self.fail("a second holder must never enter")

    def test_page_status_requires_fresh_current_engine_state(self):
        state = {
            "schema": refresh_state.STATE_SCHEMA,
            "kind": refresh_state.STATE_KIND,
            "heartbeat_at": time.time(),
            "engine": "current-engine",
            "phase": "waiting",
            "queued": [{"slug": "demo", "page": 2,
                        "cache_state": "stale"}],
            "active": [{"slug": "demo", "page": 3,
                        "cache_state": "missing", "started_at": 10.0}],
        }
        with mock.patch.object(refresh_state, "current_engine_short",
                               return_value="current-engine"):
            self.assertEqual(
                refresh_state.page_refresh_status("demo", 2, state),
                "waiting")
            self.assertEqual(
                refresh_state.page_refresh_status("demo", 3, state),
                "running")

            queued = copy.deepcopy(state)
            queued["phase"] = "running"
            self.assertEqual(
                refresh_state.page_refresh_status("demo", "2", queued),
                "queued")
            self.assertIsNone(
                refresh_state.page_refresh_status("demo", 99, queued))

            wrong_engine = copy.deepcopy(state)
            wrong_engine["engine"] = "old-engine"
            self.assertIsNone(
                refresh_state.page_refresh_status("demo", 2, wrong_engine))

            stale = copy.deepcopy(state)
            stale["heartbeat_at"] = (
                time.time() - refresh_state.DEFAULT_STALE_AFTER - 1)
            self.assertIsNone(
                refresh_state.page_refresh_status("demo", 2, stale))

    def test_active_row_wins_even_during_global_wait(self):
        state = {
            "schema": refresh_state.STATE_SCHEMA,
            "kind": refresh_state.STATE_KIND,
            "heartbeat_at": time.time(),
            "engine": "engine",
            "phase": "waiting",
            "queued": [{"slug": "demo", "page": 4}],
            "active": [{"slug": "demo", "page": 4}],
        }
        with mock.patch.object(refresh_state, "current_engine_short",
                               return_value="engine"):
            self.assertEqual(
                refresh_state.page_refresh_status("demo", 4, state),
                "running")


class InventoryTests(unittest.TestCase):
    def test_current_signature_failures_are_not_retried(self):
        raw_jobs = [
            (1, ["items-1"], {"arrow": 1}, "sig-1"),
            (2, ["items-2"], {"arrow": 2}, "sig-2"),
            (3, ["items-3"], {"arrow": 3}, "sig-3"),
            (4, ["items-4"], {"arrow": 4}, "sig-4"),
        ]
        cache = {
            1: {"sig": "sig-1", "error": "deterministic failure"},
            2: {"sig": "old-sig", "error": "failure from old engine"},
            3: None,
            4: {"sig": "old-sig", "bindings": []},
        }
        with (
            mock.patch.object(tool, "project_slugs", return_value=["demo"]),
            mock.patch.object(tool.job, "_linetype_jobs",
                              return_value=(raw_jobs, ["upstream warning"])),
            mock.patch.object(tool.linetypes, "load_page",
                              side_effect=lambda _slug, page: cache[page]),
        ):
            found = tool.inventory()

        self.assertEqual([row.page for row in found.candidates], [2, 3, 4])
        self.assertEqual([row.cache_state for row in found.candidates],
                         ["stale", "missing", "stale"])
        self.assertEqual(found.current_failures, [{
            "slug": "demo", "page": 1, "error": "deterministic failure"}])
        self.assertEqual(found.warnings, ["demo: upstream warning"])

    def test_even_an_empty_current_error_marker_is_terminal(self):
        raw_job = (1, ["items"], {"arrow": 1}, "sig-1")
        with mock.patch.object(tool.linetypes, "load_page", return_value={
                "sig": "sig-1", "error": None}):
            row, failure = tool._candidate("demo", raw_job)
        self.assertIsNone(row)
        self.assertEqual(failure["page"], 1)

    def test_timeout_is_retried_only_when_current_budget_increased(self):
        raw_job = (24, ["items"], {"geometry": {"vector_paths": 49000}},
                   "sig-current")
        entry = {
            "sig": "sig-current",
            "error": "RuntimeError: linetype sidecar timeout after 600s (sheet 24)",
        }
        with (
            mock.patch.object(tool.linetypes, "load_page", return_value=entry),
            mock.patch.object(tool.job, "_linetype_timeout_for",
                              return_value=1800),
        ):
            row, failure = tool._candidate("bristol", raw_job)
        self.assertIsNone(failure)
        self.assertEqual(row.page, 24)
        self.assertEqual(row.cache_state, "stale")

        with (
            mock.patch.object(tool.linetypes, "load_page", return_value=entry),
            mock.patch.object(tool.job, "_linetype_timeout_for",
                              return_value=600),
        ):
            row, failure = tool._candidate("bristol", raw_job)
        self.assertIsNone(row)
        self.assertEqual(failure["page"], 24)

    def test_persisted_foreground_cards_are_read_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "active.json").write_text(json.dumps({
                "slug": "active", "done": False, "stage": "text"}),
                encoding="utf-8")
            (root / "fallback.json").write_text(json.dumps({
                "done": False, "detail": "restoring"}), encoding="utf-8")
            (root / "complete.json").write_text(json.dumps({
                "slug": "complete", "done": True}), encoding="utf-8")
            (root / "invalid name.json").write_text(json.dumps({
                "done": False}), encoding="utf-8")
            (root / "broken.json").write_text("{", encoding="utf-8")

            with mock.patch.object(tool.job, "JOBS_DIR", root):
                rows = tool.persisted_running_jobs()

        self.assertEqual([row["slug"] for row in rows],
                         ["active", "fallback"])


class PageExecutionTests(unittest.TestCase):
    def test_page_recompute_uses_job_entrypoint_and_revalidates(self):
        row = candidate()
        runner = tool.RefreshRunner(clock=lambda: 20.0)
        entry = {"sig": row.sig, "bindings": []}
        with (
            mock.patch.object(tool.job, "_linetype_one",
                              return_value=(row.page, 7, None)) as compute,
            mock.patch.object(tool.linetypes, "load_page", return_value=entry),
            mock.patch.object(runner, "_candidate_for_page", return_value=(
                None, "no longer stale/missing with current prerequisites")),
        ):
            result = runner._run_candidate(row)

        self.assertEqual(result.state, "completed")
        self.assertEqual(result.count, 7)
        compute.assert_called_once_with(
            row.slug, row.page, row.items, row.arrow_entry, row.sig)

    def test_success_return_is_rejected_when_cache_is_not_current(self):
        row = candidate()
        runner = tool.RefreshRunner(clock=lambda: 20.0)
        with (
            mock.patch.object(tool.job, "_linetype_one",
                              return_value=(row.page, 1, None)),
            mock.patch.object(tool.linetypes, "load_page",
                              return_value={"sig": "old", "bindings": []}),
            mock.patch.object(runner, "_candidate_for_page",
                              return_value=(row, None)),
        ):
            result = runner._run_candidate(row)
        self.assertEqual(result.state, "failed")
        self.assertIn("post-write validation", result.error)

    def test_foreground_card_pauses_before_any_page_starts(self):
        row = candidate()
        found = tool.Inventory([row], [], [])
        probes = deque([
            [{"slug": "upload", "stage": "text", "detail": None}],
            [],  # top-of-loop check after waiting
            [],  # inner pre-revalidation check
            [],  # immediate pre-submit check
        ])
        sleeps = []
        writer = MemoryWriter()
        runner = tool.RefreshRunner(
            workers=1,
            writer=writer,
            foreground_probe=lambda: probes.popleft() if probes else [],
            sleeper=sleeps.append,
            job_poll_seconds=0.01,
            heartbeat_seconds=0.01,
            clock=lambda: 50.0,
        )
        with (
            mock.patch.object(tool, "inventory", return_value=found),
            mock.patch.object(tool, "current_engine_short",
                              return_value="engine"),
            mock.patch.object(runner, "_candidate_for_page",
                              return_value=(row, None)) as revalidate,
            mock.patch.object(runner, "_run_candidate",
                              return_value=tool.PageResult(
                                  row.slug, row.page, "completed", count=2)),
        ):
            exit_code = runner.run_once()

        self.assertEqual(exit_code, 0)
        self.assertTrue(sleeps)
        self.assertTrue(any(state.get("phase") == "waiting"
                            for state in writer.history))
        self.assertEqual(writer.state["phase"], "done")
        self.assertTrue(writer.state["ok"])
        self.assertEqual(writer.state["progress"]["completed"], 1)
        self.assertEqual(writer.state["progress"]["foreground_waits"], 1)
        revalidate.assert_called_once_with(row.slug, row.page)
        queued_rows = [item for state in writer.history
                       for item in state.get("queued", [])]
        self.assertTrue(queued_rows)
        self.assertTrue(all("sig" not in item for item in queued_rows))

    def test_card_appearing_during_revalidation_defers_the_submit(self):
        row = candidate()
        found = tool.Inventory([row], [], [])
        probes = deque([
            [],  # top of first loop
            [],  # before revalidation
            [{"slug": "upload"}],  # immediately before submit
            [{"slug": "upload"}],  # next top-of-loop -> explicit wait
            [], [], [],  # next top, inner, immediate pre-submit
        ])
        writer = MemoryWriter()
        runner = tool.RefreshRunner(
            workers=1,
            writer=writer,
            foreground_probe=lambda: probes.popleft() if probes else [],
            sleeper=lambda _seconds: None,
            job_poll_seconds=0.01,
            heartbeat_seconds=0.01,
            clock=lambda: 70.0,
        )
        with (
            mock.patch.object(tool, "inventory", return_value=found),
            mock.patch.object(tool, "current_engine_short",
                              return_value="engine"),
            mock.patch.object(runner, "_candidate_for_page",
                              return_value=(row, None)) as revalidate,
            mock.patch.object(runner, "_run_candidate",
                              return_value=tool.PageResult(
                                  row.slug, row.page, "completed", count=1)),
        ):
            exit_code = runner.run_once()

        self.assertEqual(exit_code, 0)
        self.assertEqual(revalidate.call_count, 2)
        self.assertGreaterEqual(
            writer.state["progress"]["foreground_waits"], 1)


class CommandLineTests(unittest.TestCase):
    def test_workers_default_comes_from_environment_and_once_is_accepted(self):
        with mock.patch.dict(os.environ,
                             {"LINETYPE_REFRESH_WORKERS": "5"}):
            args = tool.build_parser().parse_args(["--once"])
        self.assertTrue(args.once)
        self.assertEqual(args.workers, 5)


if __name__ == "__main__":
    unittest.main()
