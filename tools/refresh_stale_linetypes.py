#!/usr/bin/env python3
"""Low-priority, one-pass repair of stale/missing line-type page caches.

This tool intentionally imports and calls the production orchestration
functions instead of growing a second implementation of line-type execution:

* ``job._linetype_jobs`` / ``job._legend_linetype_jobs`` own inventory;
* ``job._linetype_one`` / ``job._legend_linetype_one`` own recomputation;
* each channel's cache predicate revalidates every successful return.

Ordinary jobs still require current results + arrows and at least one terminal.
Legend jobs instead require a current symbol result with an explicitly boxed
line sample; they do not depend on arrows.  A failure entry stamped with the
*current* channel signature is normally reported but not retried; the next
timer tick would otherwise burn the same deadline forever.  The ordinary
channel retains its one exception for a recorded timeout whose old deadline is
lower than the page's current adaptive deadline.  A failure from an older
signature is stale evidence and is eligible again.

The worker gives uploads priority at page boundaries.  Before every submit it
scans persisted ``_jobs/*.json`` cards.  While any card says ``done=false``,
already-running maintenance pages may finish but no new page starts.  Deploy it
with systemd ``Nice``/``CPUWeight`` as an additional OS-level priority guard.

Typical timer invocation::

    venv/bin/python -B tools/refresh_stale_linetypes.py --once

``LINETYPE_REFRESH_WORKERS`` controls page concurrency (default 3).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

# Executing ``python tools/...py`` puts tools/, not the repository root, first.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import job                                                        # noqa: E402
from steps import legend_linetypes, linetypes, store               # noqa: E402
from steps.linetypes.refresh_state import (                        # noqa: E402
    RefreshStateWriter,
    WorkerAlreadyRunning,
    current_engine_short,
    worker_lock,
)

DEFAULT_WORKERS = 3
DEFAULT_JOB_POLL_SECONDS = 5.0
DEFAULT_HEARTBEAT_SECONDS = 10.0
MAX_RECENT_RESULTS = 200


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return int(default)
    try:
        return max(1, int(raw))
    except ValueError:
        return int(default)


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        value = float(raw)
    except ValueError:
        return float(default)
    return value if value > 0 else float(default)


@dataclass(frozen=True)
class Candidate:
    slug: str
    page: int
    items: list
    arrow_entry: dict
    sig: str
    cache_state: str
    channel: str = "arrow"
    samples: list | None = None

    def public(self):
        return {"slug": self.slug, "page": self.page,
                "channel": self.channel,
                "cache_state": self.cache_state}


@dataclass
class Inventory:
    candidates: list[Candidate]
    current_failures: list[dict]
    warnings: list[str]

    def public(self):
        missing = sum(row.cache_state == "missing"
                      for row in self.candidates)
        stale = len(self.candidates) - missing
        by_project = {}
        for row in self.candidates:
            bucket = by_project.setdefault(
                row.slug, {"stale": [], "missing": []})
            bucket[row.cache_state].append(row.page)
        channels = {}
        for row in self.candidates:
            bucket = channels.setdefault(
                row.channel, {"eligible": 0, "stale": 0, "missing": 0})
            bucket["eligible"] += 1
            bucket[row.cache_state] += 1
        return {
            "eligible": len(self.candidates),
            "stale": stale,
            "missing": missing,
            "current_failures": len(self.current_failures),
            "projects": by_project,
            "channels": channels,
            "candidates": [row.public() for row in self.candidates],
            "failure_pages": self.current_failures,
            "warnings": self.warnings,
        }


@dataclass(frozen=True)
class PageResult:
    slug: str
    page: int
    state: str
    count: int | None = None
    error: str | None = None
    seconds: float = 0.0
    channel: str = "arrow"

    def public(self):
        value = {"slug": self.slug, "page": self.page,
                 "channel": self.channel,
                 "state": self.state,
                 "seconds": round(float(self.seconds), 2)}
        if self.count is not None:
            value["line_types"] = int(self.count)
        if self.error:
            value["error"] = str(self.error)[:500]
        return value


def _current_failure(entry, sig):
    return bool(isinstance(entry, dict) and entry.get("sig") == sig
                and "error" in entry)


def _candidate(slug, raw_job):
    """Build one ordinary arrow-bound candidate (legacy helper name)."""
    page, items, arrow_entry, sig = raw_job
    entry = linetypes.load_page(slug, page)
    if _current_failure(entry, sig):
        # A timeout at an older, smaller budget is not the same deterministic
        # failure anymore.  Requeue exactly once under the larger current
        # budget; if that also times out its marker will carry that new budget
        # and future timer passes stop retrying it.
        if not job._linetype_failure_budget_increased(
                slug, page, arrow_entry, entry.get("error")):
            return None, {
                "slug": slug,
                "page": int(page),
                "error": str(entry.get("error"))[:500],
            }
    return Candidate(
        slug=slug,
        page=int(page),
        items=items,
        arrow_entry=arrow_entry,
        sig=sig,
        cache_state="missing" if entry is None else "stale",
    ), None


def _legend_candidate(slug, raw_job):
    """Build one supervised legend candidate, suppressing current failures."""
    page, samples, sig = raw_job
    entry = legend_linetypes.load_page(slug, page)
    if _current_failure(entry, sig):
        return None, {
            "slug": slug,
            "page": int(page),
            "channel": "legend",
            "error": str(entry.get("error"))[:500],
        }
    return Candidate(
        slug=slug,
        page=int(page),
        items=[],
        arrow_entry={},
        sig=sig,
        cache_state="missing" if entry is None else "stale",
        channel="legend",
        samples=samples,
    ), None


def project_slugs(selected=None):
    """Safe, deterministic project enumeration for an inventory pass."""
    if selected:
        return sorted({slug for slug in selected
                       if store.is_valid_slug(slug)})
    root = Path(job.DATA_DIR)
    if not root.is_dir():
        return []
    return sorted(path.name for path in root.iterdir()
                  if path.is_dir() and store.is_valid_slug(path.name))


def inventory(selected=None):
    """Inventory stale/missing pages using production prerequisites.

    Each production job builder owns its own prerequisites.  The only
    additional policy here is to skip a current-signature error entry so a
    deterministic failure cannot consume every timer pass forever.
    """
    candidates = []
    failures = []
    warnings = []
    for slug in project_slugs(selected):
        channels = (
            ("arrow", job._linetype_jobs, _candidate),
            ("legend", job._legend_linetype_jobs, _legend_candidate),
        )
        for channel, jobs_fn, candidate_fn in channels:
            try:
                raw_jobs, project_warnings = jobs_fn(slug)
            except Exception as exc:                          # noqa: BLE001
                prefix = (f"{slug}: inventory failed: " if channel == "arrow"
                          else f"{slug}: legend inventory failed: ")
                warnings.append(
                    f"{prefix}{type(exc).__name__}: {exc}")
                continue
            warnings.extend(f"{slug}: {message}"
                            for message in (project_warnings or ()))
            for raw_job in raw_jobs:
                row, failure = candidate_fn(slug, raw_job)
                if failure is not None:
                    # The ordinary helper predates channels; add its identity
                    # here without changing its standalone test/API contract.
                    failure.setdefault("channel", channel)
                    failures.append(failure)
                elif row is not None:
                    candidates.append(row)
    candidates.sort(key=lambda row: (row.slug, row.page, row.channel))
    failures.sort(key=lambda row: (
        row["slug"], row["page"], row.get("channel", "arrow")))
    return Inventory(candidates, failures, warnings)


def persisted_running_jobs():
    """All persisted foreground cards whose work has not completed."""
    root = Path(job.JOBS_DIR)
    if not root.is_dir():
        return []
    running = []
    for path in sorted(root.glob("*.json")):
        card = store.load_json(path, None)
        if not isinstance(card, dict) or card.get("done") is not False:
            continue
        slug = card.get("slug") or path.stem
        # Only genuine project cards can pause maintenance.  A malformed JSON
        # dropped in _jobs must not wedge the timer forever.
        if not store.is_valid_slug(slug):
            continue
        running.append({
            "slug": slug,
            "stage": card.get("stage"),
            "detail": card.get("detail"),
        })
    return running


class RefreshRunner:
    def __init__(self, *, workers=DEFAULT_WORKERS, selected=None,
                 writer=None, foreground_probe=None, sleeper=None,
                 job_poll_seconds=DEFAULT_JOB_POLL_SECONDS,
                 heartbeat_seconds=DEFAULT_HEARTBEAT_SECONDS,
                 clock=None):
        self.workers = max(1, int(workers))
        self.selected = list(selected or ())
        self.writer = writer or RefreshStateWriter()
        self.foreground_probe = foreground_probe or persisted_running_jobs
        self.sleeper = sleeper or time.sleep
        self.job_poll_seconds = max(0.01, float(job_poll_seconds))
        self.heartbeat_seconds = max(0.01, float(heartbeat_seconds))
        self.clock = clock or time.time
        self._results = []
        self._progress = {
            "total": 0, "completed": 0, "failed": 0, "skipped": 0,
            "remaining": 0, "foreground_waits": 0,
        }

    def _candidate_for_page(self, slug, page, channel="arrow"):
        """Re-read production prerequisites immediately before a page starts."""
        if channel == "arrow":
            jobs_fn, candidate_fn = job._linetype_jobs, _candidate
        elif channel == "legend":
            jobs_fn, candidate_fn = (
                job._legend_linetype_jobs, _legend_candidate)
        else:
            return None, f"unknown line-type refresh channel: {channel}"
        try:
            raw_jobs, _warnings = jobs_fn(slug)
        except Exception:                                      # noqa: BLE001
            return None, "prerequisite inventory failed"
        for raw in raw_jobs:
            if int(raw[0]) != int(page):
                continue
            row, failure = candidate_fn(slug, raw)
            if failure is not None:
                return None, "current-signature failure"
            return row, None
        return None, "no longer stale/missing with current prerequisites"

    def _revalidate(self, row):
        # Preserve the ordinary helper's historical two-argument invocation;
        # a number of offline callers patch this boundary directly.
        if row.channel == "arrow":
            return self._candidate_for_page(row.slug, row.page)
        return self._candidate_for_page(row.slug, row.page, row.channel)

    def _run_candidate(self, row):
        started = self.clock()
        try:
            if row.channel == "legend":
                page, count, error = job._legend_linetype_one(
                    row.slug, row.page, row.samples, row.sig)
            else:
                page, count, error = job._linetype_one(
                    row.slug, row.page, row.items, row.arrow_entry, row.sig)
        except Exception as exc:                              # noqa: BLE001
            return PageResult(
                row.slug, row.page, "failed", error=(
                    f"{type(exc).__name__}: {exc}"),
                seconds=self.clock() - started,
                channel=row.channel)
        if error:
            return PageResult(row.slug, page, "failed", error=str(error),
                              seconds=self.clock() - started,
                              channel=row.channel)
        if row.channel == "legend":
            entry = legend_linetypes.load_page(row.slug, page)
            is_current = legend_linetypes.has_current(entry, row.sig)
        else:
            entry = linetypes.load_page(row.slug, page)
            is_current = linetypes.has_current_linetypes(entry, row.sig)
        if not is_current:
            newer, reason = self._revalidate(row)
            if newer is not None and newer.sig != row.sig:
                return PageResult(
                    row.slug, page, "skipped",
                    error="prerequisites changed while the page ran",
                    seconds=self.clock() - started,
                    channel=row.channel)
            if reason == "no longer stale/missing with current prerequisites":
                return PageResult(
                    row.slug, page, "skipped",
                    error="captured page was superseded while it ran",
                    seconds=self.clock() - started,
                    channel=row.channel)
            return PageResult(
                row.slug, page, "failed",
                error=(("post-write validation rejected the line-type cache"
                        if row.channel == "arrow" else
                        "post-write validation rejected the legend "
                        "line-type cache")
                       if newer is not None
                       else reason or "post-write validation failed"),
                seconds=self.clock() - started,
                channel=row.channel)
        # If a concurrent project update changed the expected signature while
        # this page ran, _linetype_jobs will queue it again.  Do not claim the
        # old-input result repaired the live page; the next timer pass can pick
        # up the new candidate.
        newer, reason = self._revalidate(row)
        if newer is not None:
            return PageResult(
                row.slug, page, "skipped",
                error=("prerequisites changed while the page ran"
                       if newer.sig != row.sig
                       else "page remains stale after a validated write"),
                seconds=self.clock() - started,
                channel=row.channel)
        if reason == "current-signature failure":
            return PageResult(
                row.slug, page, "failed",
                error="page produced a current-signature failure entry",
                seconds=self.clock() - started,
                channel=row.channel)
        if reason != "no longer stale/missing with current prerequisites":
            return PageResult(
                row.slug, page, "failed",
                error=reason or "post-write prerequisite validation failed",
                seconds=self.clock() - started,
                channel=row.channel)
        return PageResult(row.slug, page, "completed", count=count,
                          seconds=self.clock() - started,
                          channel=row.channel)

    def _public_queue(self, pending):
        return [row.public() for row in pending]

    def _public_active(self, active):
        return [{**row.public(), "started_at": started}
                for _future, (row, started) in active.items()]

    def _record(self, result):
        if result.state == "completed":
            self._progress["completed"] += 1
        elif result.state == "failed":
            self._progress["failed"] += 1
        else:
            self._progress["skipped"] += 1
        self._results.append(result.public())
        if len(self._results) > MAX_RECENT_RESULTS:
            del self._results[:-MAX_RECENT_RESULTS]

    def _publish(self, *, phase, pending, active, foreground_jobs=None,
                 detail=None, inventory_payload=None, last_error=None):
        self._progress["remaining"] = len(pending) + len(active)
        changes = {
            "phase": phase,
            "detail": detail,
            "queued": self._public_queue(pending),
            "active": self._public_active(active),
            "foreground_jobs": list(foreground_jobs or ()),
            "progress": dict(self._progress),
            "recent_results": list(self._results),
        }
        if inventory_payload is not None:
            changes["inventory"] = inventory_payload
        if last_error is not None:
            changes["last_error"] = str(last_error)[:1000]
        self.writer.update(**changes)

    def run_once(self):
        started = self.clock()
        self.writer.replace({
            "pid": os.getpid(),
            "mode": "once",
            "phase": "inventory",
            "detail": "Scanning current line-type prerequisites and caches",
            "started_at": started,
            "engine": current_engine_short(),
            "workers": self.workers,
            "selected_slugs": self.selected,
            "queued": [],
            "active": [],
            "foreground_jobs": [],
            "progress": dict(self._progress),
            "recent_results": [],
        })
        found = inventory(self.selected)
        pending = deque(found.candidates)
        self._progress.update({
            "total": len(pending),
            "remaining": len(pending),
            "current_failures": len(found.current_failures),
        })
        self._publish(
            phase="queued" if pending else "done",
            pending=pending, active={},
            detail=(f"Queued {len(pending)} stale/missing pages"
                    if pending else "No stale/missing eligible pages"),
            inventory_payload=found.public())
        if not pending:
            self.writer.update(finished_at=self.clock(), ok=True)
            return 0

        active = {}
        with ThreadPoolExecutor(max_workers=self.workers,
                                thread_name_prefix="linetype-refresh") as pool:
            while pending or active:
                foreground = self.foreground_probe() or []
                # Existing maintenance pages are allowed to finish, but every
                # individual new submit is preceded by this persisted-card
                # check.  That is the upload-priority boundary.
                if foreground:
                    self._progress["foreground_waits"] += 1
                    self._publish(
                        phase="waiting", pending=pending, active=active,
                        foreground_jobs=foreground,
                        detail=("Foreground upload/rerun active; no new "
                                "maintenance page will start"))
                    if not active:
                        self.sleeper(self.job_poll_seconds)
                        continue
                else:
                    while pending and len(active) < self.workers:
                        foreground = self.foreground_probe() or []
                        if foreground:
                            break
                        original = pending.popleft()
                        row, reason = self._revalidate(original)
                        if row is None:
                            self._record(PageResult(
                                original.slug, original.page, "skipped",
                                error=reason,
                                channel=original.channel))
                            continue
                        # The prerequisite scan can touch several cache files.
                        # Close the gap between its earlier probe and the
                        # actual submit so a foreground card which appeared
                        # during revalidation still wins this page boundary.
                        foreground = self.foreground_probe() or []
                        if foreground:
                            pending.appendleft(original)
                            break
                        future = pool.submit(self._run_candidate, row)
                        active[future] = (row, self.clock())
                    if foreground:
                        self._progress["foreground_waits"] += 1
                    self._publish(
                        phase=("waiting" if foreground else
                               "running" if active else "queued"),
                        pending=pending, active=active,
                        foreground_jobs=foreground,
                        detail=(
                            "Foreground upload/rerun active; no new "
                            "maintenance page will start"
                            if foreground else
                            f"Refreshing {len(active)} page(s); "
                            f"{len(pending)} queued"))

                if not active:
                    if not pending:
                        break
                    # The foreground card may have appeared during the inner
                    # submit loop.  Recheck promptly rather than spinning.
                    self.sleeper(self.job_poll_seconds)
                    continue
                done, _not_done = wait(
                    tuple(active), timeout=self.heartbeat_seconds,
                    return_when=FIRST_COMPLETED)
                if not done:
                    self._publish(
                        phase="waiting" if foreground else "running",
                        pending=pending, active=active,
                        foreground_jobs=foreground,
                        detail=("Line-type refresh heartbeat; page engines "
                                "are still active"))
                    continue
                for future in done:
                    row, _page_started = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:                  # noqa: BLE001
                        result = PageResult(
                            row.slug, row.page, "failed",
                            error=f"{type(exc).__name__}: {exc}",
                            channel=row.channel)
                    self._record(result)

        failures = self._progress["failed"]
        phase = "done_with_errors" if failures else "done"
        self._publish(
            phase=phase, pending=deque(), active={},
            detail=(f"Refresh finished: {self._progress['completed']} current, "
                    f"{failures} failed, {self._progress['skipped']} skipped"))
        self.writer.update(finished_at=self.clock(), ok=not bool(failures))
        return 1 if failures else 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true",
        help="run one inventory/recompute pass and exit (the only mode; "
             "accepted explicitly for systemd timer readability)")
    parser.add_argument(
        "--workers", type=_positive_int,
        default=_env_int("LINETYPE_REFRESH_WORKERS", DEFAULT_WORKERS),
        help="concurrent page engines (env LINETYPE_REFRESH_WORKERS, default 3)")
    parser.add_argument(
        "--slug", action="append", default=[],
        help="limit the pass to one project slug; repeatable")
    parser.add_argument(
        "--job-poll-seconds", type=_positive_float,
        default=_env_float("LINETYPE_REFRESH_JOB_POLL_SECONDS",
                           DEFAULT_JOB_POLL_SECONDS),
        help="foreground job-card recheck interval (default 5s)")
    parser.add_argument(
        "--heartbeat-seconds", type=_positive_float,
        default=_env_float("LINETYPE_REFRESH_HEARTBEAT_SECONDS",
                           DEFAULT_HEARTBEAT_SECONDS),
        help="status heartbeat interval while page engines run (default 10s)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    writer = RefreshStateWriter()
    try:
        with worker_lock():
            runner = RefreshRunner(
                workers=args.workers,
                selected=args.slug,
                writer=writer,
                job_poll_seconds=args.job_poll_seconds,
                heartbeat_seconds=args.heartbeat_seconds)
            return runner.run_once()
    except WorkerAlreadyRunning as exc:
        # Do not overwrite the live worker's heartbeat merely to report that
        # this second timer/process lost the singleton race.
        print(f"refresh already running: {exc}", file=sys.stderr, flush=True)
        return 75
    except KeyboardInterrupt:
        writer.update(phase="stopped", detail="Interrupted", ok=False,
                      finished_at=time.time())
        return 130
    except Exception as exc:                                  # noqa: BLE001
        traceback.print_exc()
        try:
            writer.update(
                phase="error", detail="Refresh process failed", ok=False,
                last_error=f"{type(exc).__name__}: {exc}",
                finished_at=time.time())
        except Exception:                                     # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
