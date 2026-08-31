"""job.py 的离线回归 —— 严禁任何模型调用.

这里只测编排层自己的逻辑：slug 唯一化 / 目录布局、任务状态机与落盘、
重启后的 interrupted 语义、费用合并、进度映射、协作式取消。四个阶段函数
一律用假实现替换，所以整套测试不碰 PDF、不碰 Gemini、不写真实 data/。
"""
import io
import tempfile
import shutil
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

import job
from steps import store


class JobTestBase(unittest.TestCase):
    """每个用例一个临时的 projects/ data/ _jobs/，绝不污染真实目录。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fence_lite_job_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.projects = self.tmp / "projects"
        self.data = self.tmp / "data"
        self.jobs = self.tmp / "_jobs"
        for d in (self.projects, self.data, self.jobs):
            d.mkdir(parents=True)
        # job.py 与 steps/store.py 各自持有一份目录常量的绑定，两边都要换。
        for target, name, value in (
                (job, "PROJECTS_DIR", self.projects),
                (job, "DATA_DIR", self.data),
                (job, "JOBS_DIR", self.jobs),
                (store, "PROJECTS_DIR", self.projects),
                (store, "DATA_DIR", self.data),
                (store, "JOBS_DIR", self.jobs)):
            p = mock.patch.object(target, name, value)
            p.start()
            self.addCleanup(p.stop)
        job.JOBS.clear()
        job._CANCEL.clear()
        job._CANCEL_USERS.clear()
        job._RUNNING.clear()
        job._STARTING.clear()
        job._SLOT_POOLS.clear()
        self.addCleanup(job.JOBS.clear)
        self.addCleanup(job._CANCEL.clear)
        self.addCleanup(job._CANCEL_USERS.clear)
        self.addCleanup(job._RUNNING.clear)
        self.addCleanup(job._STARTING.clear)
        self.addCleanup(job._SLOT_POOLS.clear)

    def write_results(self, slug, extra=None):
        res = {"slug": slug, "fused_v": 2, "page_count": 1, "pages": {}}
        res.update(extra or {})
        store.save_json(store.results_path(slug), res)
        return res


# --------------------------------------------------------------- projects --
class TestProjectSetup(JobTestBase):

    def test_slug_uniquified_on_second_upload(self):
        a = job.create_project(b"%PDF-1.4 a", "Foo Bar.pdf")
        b = job.create_project(b"%PDF-1.4 b", "foo bar.PDF")
        self.assertEqual((a, b), ("foo_bar", "foo_bar_2"))
        self.assertEqual((self.projects / a / "input.pdf").read_bytes(),
                         b"%PDF-1.4 a")
        self.assertEqual((self.projects / b / "input.pdf").read_bytes(),
                         b"%PDF-1.4 b")

    def test_slug_collides_with_data_dir_only(self):
        (self.data / "sheets").mkdir()
        self.assertEqual(job.create_project(b"x", "sheets.pdf"), "sheets_2")

    def test_slug_fallback_and_length_cap(self):
        self.assertEqual(job.create_project(b"x", "___.pdf"), "project")
        long_slug = job.create_project(b"x", ("a" * 90) + ".pdf")
        self.assertEqual(long_slug, "a" * 60)
        self.assertTrue(job.is_valid_slug(long_slug))

    def test_stream_upload_writes_incrementally(self):
        payload = b"%PDF-1.4" + (b"x" * (2 * 1024 * 1024 + 17))

        class ChunkOnly(io.BytesIO):
            def read(self, size=-1):
                self.assert_chunk(size)
                return super().read(size)

            @staticmethod
            def assert_chunk(size):
                if not 0 < size <= 1024 * 1024:
                    raise AssertionError(f"unbounded stream read: {size}")

        slug = job.create_project_stream(ChunkOnly(payload), "large.pdf")
        self.assertEqual(slug, "large")
        self.assertEqual((self.projects / slug / "input.pdf").read_bytes(),
                         payload)

    def test_simultaneous_same_name_uploads_never_overwrite(self):
        payloads = [f"%PDF-1.4 upload {i}".encode() for i in range(8)]

        def upload(data):
            slug = job.create_project(data, "same.pdf")
            return slug, data

        with ThreadPoolExecutor(max_workers=8) as pool:
            uploaded = list(pool.map(upload, payloads))
        self.assertEqual(len({slug for slug, _data in uploaded}), 8)
        for slug, data in uploaded:
            self.assertEqual((self.projects / slug / "input.pdf").read_bytes(),
                             data)

    def test_delete_project_removes_all_three_places(self):
        slug = job.create_project(b"x", "gone.pdf")
        store.save_json(store.slug_dir(slug) / "results.json", {"slug": slug})
        job._set(slug, stage="done", done=True)
        self.assertTrue((self.jobs / f"{slug}.json").exists())
        self.assertTrue(job.delete_project(slug))
        self.assertFalse((self.projects / slug).exists())
        self.assertFalse((self.data / slug).exists())
        self.assertFalse((self.jobs / f"{slug}.json").exists())
        self.assertNotIn(slug, job.JOBS)
        self.assertFalse(job.delete_project(slug))

    def test_delete_project_rejects_bad_slug(self):
        with self.assertRaises(ValueError):
            job.delete_project("../etc")

    def test_delete_project_rejects_queued_or_running_owner(self):
        slug = job.create_project(b"x", "keep-running.pdf")
        job._set(slug, stage="queued", done=False)
        with self.assertRaises(job.JobStartError):
            job.delete_project(slug)
        self.assertTrue((self.projects / slug / "input.pdf").exists())
        self.assertTrue((self.jobs / f"{slug}.json").exists())
        self.assertNotIn(slug, job._STARTING)

    def test_page_count_of_unreadable_pdf_is_zero(self):
        slug = job.create_project(b"not a pdf", "broken.pdf")
        self.assertEqual(job.page_count_of(slug), 0)


# ------------------------------------------------------------- job status --
class TestJobState(JobTestBase):

    def test_set_persists_and_reloads(self):
        job._set("alpha", stage="text", progress=0.25, warnings=[])
        on_disk = store.load_json(self.jobs / "alpha.json", None)
        self.assertEqual(on_disk["slug"], "alpha")
        self.assertEqual(on_disk["stage"], "text")
        self.assertEqual(on_disk["progress"], 0.25)
        self.assertEqual(on_disk["warnings"], [])
        self.assertIsInstance(on_disk["updated_at"], float)
        job._set("alpha", progress=0.5)
        self.assertEqual(
            store.load_json(self.jobs / "alpha.json", None)["progress"], 0.5)

    def test_get_job_returns_copy_and_none(self):
        self.assertIsNone(job.get_job("nope"))
        job._set("alpha", stage="text")
        snap = job.get_job("alpha")
        snap["stage"] = "mutated"
        self.assertEqual(job.JOBS["alpha"]["stage"], "text")

    def test_job_running_only_until_done(self):
        job._set("alpha", done=False)
        self.assertTrue(job.job_running("alpha"))
        job._set("alpha", done=True)
        self.assertFalse(job.job_running("alpha"))
        self.assertFalse(job.job_running("unknown"))

    def test_all_jobs_newest_first(self):
        job._set("old", started=100.0)
        job._set("new", started=200.0)
        self.assertEqual([j["slug"] for j in job.all_jobs()], ["new", "old"])

    def test_warn_appends_without_dropping_earlier(self):
        job._set("alpha", warnings=[])
        job._warn("alpha", ["P1 boom"])
        job._warn("alpha", ["P2 boom"])
        job._warn("alpha", [])
        self.assertEqual(job.JOBS["alpha"]["warnings"], ["P1 boom", "P2 boom"])
        self.assertEqual(
            store.load_json(self.jobs / "alpha.json", None)["warnings"],
            ["P1 boom", "P2 boom"])

    def test_concurrent_persistence_cannot_replace_new_state_with_old(self):
        """Heartbeat/progress writers must preserve the newest disk snapshot."""
        job.JOBS["alpha"] = {"slug": "alpha", "done": False,
                             "progress": 0.25}
        first_inside_save = threading.Event()
        release_first = threading.Event()
        second_inside_save = threading.Event()
        call_lock = threading.Lock()
        calls = 0

        def controlled_save(path, payload):
            nonlocal calls
            with call_lock:
                calls += 1
                number = calls
            if number == 1:
                first_inside_save.set()
                self.assertTrue(release_first.wait(2))
            else:
                second_inside_save.set()
            store.save_json(path, payload)

        with mock.patch.object(job, "save_json", side_effect=controlled_save):
            older = threading.Thread(target=job._persist_job, args=("alpha",))
            older.start()
            self.assertTrue(first_inside_save.wait(2))
            with job._JOBS_LOCK:
                job.JOBS["alpha"].update(done=True, progress=1.0)
            newer = threading.Thread(target=job._persist_job, args=("alpha",))
            newer.start()
            # Snapshot+save is one serialized region: the newer writer cannot
            # even enter save_json until the older fsync/replace has finished.
            self.assertFalse(second_inside_save.wait(0.1))
            release_first.set()
            older.join(2)
            newer.join(2)

        self.assertFalse(older.is_alive())
        self.assertFalse(newer.is_alive())
        self.assertTrue(second_inside_save.is_set())
        persisted = store.load_json(self.jobs / "alpha.json", {})
        self.assertTrue(persisted["done"])
        self.assertEqual(persisted["progress"], 1.0)

    def test_request_cancel_without_running_job(self):
        out = job.request_cancel("alpha")
        self.assertFalse(out["was_running"])
        self.assertTrue(job.JOBS["alpha"]["cancel_requested"])

    def test_carry_baseline_uses_newer_zero_cost_card_ledger(self):
        slug = job.create_project(b"x", "ledger.pdf")
        self.write_results(slug, {
            "llm_summary": {"calls": 1, "model_seconds": 0.1,
                            "input_tokens": 0, "output_tokens": 0,
                            "thoughts_tokens": 0, "cost_usd": 0.0},
            "wall_seconds": 3.0,
        })
        store.save_json(self.jobs / f"{slug}.json", {
            "slug": slug,
            "llm": {"calls": 2, "model_seconds": 0.2,
                    "input_tokens": 0, "output_tokens": 0,
                    "thoughts_tokens": 0, "cost_usd": 0.0},
            "processing_started": 100.0,
            "wall_seconds": 4.0,
        })
        baseline, wall = job._carry_baseline(slug)
        self.assertEqual(baseline["calls"], 2)
        self.assertEqual(wall, 4.0)

    def test_carry_baseline_ignores_legacy_queue_inflated_wall(self):
        slug = job.create_project(b"x", "legacy-wall.pdf")
        store.save_json(self.jobs / f"{slug}.json", {
            "slug": slug, "done": False, "stage": "text",
            "wall_seconds": 1800.0,
        })
        _baseline, wall = job._carry_baseline(slug)
        self.assertEqual(wall, 0.0)

    def test_resume_interrupted_relaunches_from_page_checkpoints(self):
        store.save_json(self.jobs / "finished.json",
                        {"slug": "finished", "done": True, "ok": True,
                         "stage": "done"})
        (self.projects / "inflight").mkdir()
        (self.projects / "inflight" / "input.pdf").write_bytes(b"%PDF-1.4")
        store.save_json(self.jobs / "inflight.json",
                        {"slug": "inflight", "done": False, "stage": "symbols",
                         "progress": 0.6, "target": "find every fence",
                         "model": job.MODEL_NAME})
        store.save_json(self.jobs / "cancelled.json",
                        {"slug": "cancelled", "done": False,
                         "cancel_requested": True, "stage": "linetypes"})
        store.save_json(self.jobs / "junk.json", {"slug": "not/a/slug"})
        with mock.patch.object(job, "start_job") as start:
            self.assertIsNone(job.resume_interrupted())
        start.assert_called_once_with(
            "inflight", target="find every fence", model=job.MODEL_NAME,
            _resume=True)
        done = job.JOBS["finished"]
        self.assertEqual((done["done"], done["ok"], done["stage"]),
                         (True, True, "done"))
        hit = job.JOBS["inflight"]
        self.assertEqual((hit["done"], hit["ok"], hit["cancelled"],
                          hit["stage"]), (False, None, False, "queued"))
        # 这是交给 start_job 前的恢复卡；真实 start_job 会把本次
        # 进度条重置为 0，但逐页缓存仍会复用。
        self.assertEqual(hit["progress"], 0.6)
        self.assertEqual(
            store.load_json(self.jobs / "inflight.json", None)["stage"],
            "queued")
        stopped = job.JOBS["cancelled"]
        self.assertEqual(
            (stopped["done"], stopped["cancelled"], stopped["stage"]),
            (True, True, "cancelled"))
        self.assertNotIn("not/a/slug", job.JOBS)


# ------------------------------------------------------------ cost / time --
class TestLlmMerge(unittest.TestCase):

    def test_merge_sums_but_takes_max_peak(self):
        a = {"calls": 2, "model_seconds": 1.5, "peak_concurrency": 6,
             "input_tokens": 10, "output_tokens": 3, "thoughts_tokens": 1,
             "cost_usd": 0.125,
             "by_model": {"pro": {"calls": 2, "seconds": 1.5,
                                  "cost_usd": 0.125, "input_tokens": 10,
                                  "output_tokens": 3, "thoughts_tokens": 1}}}
        b = {"calls": 1, "model_seconds": 0.25, "peak_concurrency": 3,
             "input_tokens": 5, "output_tokens": 2, "thoughts_tokens": 0,
             "cost_usd": 0.0625,
             "by_model": {"pro": {"calls": 1, "seconds": 0.25,
                                  "cost_usd": 0.0625, "input_tokens": 5,
                                  "output_tokens": 2, "thoughts_tokens": 0},
                          "flash": {"calls": 4, "seconds": 2.0,
                                    "cost_usd": 0.5, "input_tokens": 1,
                                    "output_tokens": 1, "thoughts_tokens": 1}}}
        out = job._merge_llm(a, b)
        self.assertEqual(out["calls"], 3)
        self.assertEqual(out["model_seconds"], 1.75)
        self.assertEqual(out["peak_concurrency"], 6)      # max, not sum
        self.assertEqual(out["input_tokens"], 15)
        self.assertEqual(out["output_tokens"], 5)
        self.assertEqual(out["thoughts_tokens"], 1)
        self.assertEqual(out["cost_usd"], 0.1875)
        self.assertEqual(out["by_model"]["pro"],
                         {"calls": 3, "seconds": 1.75, "cost_usd": 0.1875,
                          "input_tokens": 15, "output_tokens": 5,
                          "thoughts_tokens": 1})
        self.assertEqual(out["by_model"]["flash"]["calls"], 4)

    def test_merge_tolerates_none(self):
        self.assertEqual(job._merge_llm(None, None), job._ZERO_LLM)
        self.assertEqual(job._merge_llm(None, {"calls": 1})["calls"], 1)


class TestStageProgress(JobTestBase):

    def test_linear_mapping_into_slice(self):
        cb = job._stage_progress("alpha", 0.55, 0.80)
        cb(0, 4)
        self.assertEqual(job.JOBS["alpha"]["progress"], 0.55)
        cb(2, 4)
        self.assertEqual(job.JOBS["alpha"]["progress"], 0.675)
        cb(4, 4)
        self.assertEqual(job.JOBS["alpha"]["progress"], 0.80)
        self.assertEqual((job.JOBS["alpha"]["stage_done"],
                          job.JOBS["alpha"]["stage_total"]), (4, 4))

    def test_zero_total_pins_to_lo(self):
        cb = job._stage_progress("alpha", 0.92, 1.0)
        cb(0, 0)
        self.assertEqual(job.JOBS["alpha"]["progress"], 0.92)


# --------------------------------------------------------- pipeline wiring --
class _FakeStages:
    """把六个阶段函数换成假实现，记录调用顺序与收到的 should_cancel。"""

    def __init__(self, case, text=None, on_text=None):
        self.calls = []
        self.text_warnings = text or []
        self.on_text = on_text
        for name in ("_stage_text", "_stage_symbols", "_stage_views",
                     "_stage_placements", "_stage_arrows",
                     "_stage_linetypes"):
            p = mock.patch.object(job, name, self._make(name))
            p.start()
            case.addCleanup(p.stop)
        for owner, name, value in (
                (job.arrows, "ENABLED", True),
                (job.linetypes, "ENABLED", True),
                (job, "require_full_pipeline_ready", mock.Mock(return_value=True))):
            p = mock.patch.object(owner, name, value)
            p.start()
            case.addCleanup(p.stop)

    def _make(self, name):
        def stage(slug, *args, **kw):
            self.calls.append(name)
            if name == "_stage_text":
                if self.on_text:
                    self.on_text(slug)
                return list(self.text_warnings)
            return []
        return stage


class TestPipeline(JobTestBase):

    def test_preflight_rejects_every_partial_toggle_combination(self):
        cases = ((False, False), (True, False), (False, True))
        for arrow_on, line_on in cases:
            with self.subTest(arrows=arrow_on, linetypes=line_on), \
                    mock.patch.object(job.arrows, "ENABLED", arrow_on), \
                    mock.patch.object(job.linetypes, "ENABLED", line_on), \
                    mock.patch.object(job.arrows, "sidecar_available",
                                      return_value=True), \
                    mock.patch.object(job.linetypes, "sidecar_available",
                                      return_value=True), \
                    mock.patch.object(job.legend_linetypes,
                                      "sidecar_available", return_value=True), \
                    mock.patch.object(job.linetypes.sidecar,
                                      "all_sidecar_available",
                                      return_value=True), \
                    mock.patch.object(job.linetypes.sidecar, "dep_versions",
                                      return_value={}):
                with self.assertRaisesRegex(
                        RuntimeError, "Full fence pipeline is not ready"):
                    job.require_full_pipeline_ready()

    def test_preflight_probes_complete_runtime(self):
        with mock.patch.object(job.arrows, "ENABLED", True), \
                mock.patch.object(job.linetypes, "ENABLED", True), \
                mock.patch.object(job.arrows, "sidecar_available",
                                  return_value=True), \
                mock.patch.object(job.arrows, "sidecar_probe") as arrow_probe, \
                mock.patch.object(job.linetypes, "sidecar_available",
                                  return_value=True), \
                mock.patch.object(job.legend_linetypes,
                                  "sidecar_available", return_value=True), \
                mock.patch.object(job.linetypes.sidecar,
                                  "all_sidecar_available", return_value=True), \
                mock.patch.object(job.linetypes.sidecar, "dep_versions",
                                  return_value={"numpy": "2"}), \
                mock.patch.object(job.linetypes.sidecar,
                                  "sidecar_probe") as line_probe:
            self.assertTrue(job.require_full_pipeline_ready(probe=True))
        arrow_probe.assert_called_once_with()
        line_probe.assert_called_once_with()

    def test_fence_mode_runs_six_stages_in_order(self):
        fake = _FakeStages(self)
        job._run_pipeline("alpha", None, lambda: False)
        self.assertEqual(fake.calls, ["_stage_text", "_stage_symbols",
                                      "_stage_views", "_stage_placements",
                                      "_stage_arrows", "_stage_linetypes"])
        self.assertEqual(job.JOBS["alpha"]["completed_stages"],
                         list(job.FULL_PIPELINE_STAGES))
        self.assertEqual(job.JOBS["alpha"]["stage"], "done")
        self.assertEqual(job.JOBS["alpha"]["progress"], 1.0)

    def test_custom_target_runs_text_only(self):
        fake = _FakeStages(self)
        job._run_pipeline("alpha", "find every manhole", lambda: False)
        self.assertEqual(fake.calls, ["_stage_text"])
        self.assertEqual(job.JOBS["alpha"]["progress"], 1.0)
        self.assertEqual(job.JOBS["alpha"]["detail"], "Text step done")

    def test_guard_raises_and_later_stages_never_run(self):
        fake = _FakeStages(self)
        slug = job.create_project(b"x", "cancelme.pdf")
        with self.assertRaises(job.Cancelled):
            job._run_pipeline(slug, None, lambda: True)
        self.assertEqual(fake.calls, ["_stage_text"])
        self.assertFalse((self.data / slug / "results.json").exists())

    def test_warnings_from_stages_land_on_the_job(self):
        _FakeStages(self, text=["P3 文字 VLM 失败：boom"])
        job._run_pipeline("alpha", None, lambda: False)
        self.assertEqual(job.JOBS["alpha"]["warnings"],
                         ["P3 文字 VLM 失败：boom"])

    def test_linetypes_runs_for_legend_when_arrows_toggle_is_off(self):
        fake = _FakeStages(self)
        stage_state = []

        def line_stage(slug, **_kwargs):
            stage_state.append((job.JOBS[slug]["stage"],
                                job.JOBS[slug]["progress"]))
            return []

        with mock.patch.object(job.arrows, "ENABLED", False), \
                mock.patch.object(job.linetypes, "ENABLED", True), \
                mock.patch.object(job, "_stage_arrows") as arrow_stage, \
                mock.patch.object(job, "_stage_linetypes",
                                  side_effect=line_stage) as line_types:
            job._run_pipeline("alpha", None, lambda: False,
                              allow_partial=True)

        self.assertEqual(fake.calls, ["_stage_text", "_stage_symbols",
                                      "_stage_views", "_stage_placements"])
        arrow_stage.assert_not_called()
        line_types.assert_called_once()
        self.assertEqual(stage_state, [("linetypes", 0.98)])
        self.assertEqual(job.JOBS["alpha"]["progress"], 1.0)

    def test_full_local_chain_keeps_existing_progress_slices(self):
        _FakeStages(self)
        stages = []

        def local_stage(name):
            def run(slug, **_kwargs):
                stages.append((name, job.JOBS[slug]["progress"]))
                return []
            return run

        with mock.patch.object(job.arrows, "ENABLED", True), \
                mock.patch.object(job.linetypes, "ENABLED", True), \
                mock.patch.object(job, "_stage_arrows",
                                  side_effect=local_stage("arrows")), \
                mock.patch.object(job, "_stage_linetypes",
                                  side_effect=local_stage("linetypes")):
            job._run_pipeline("alpha", None, lambda: False)

        self.assertEqual(stages, [("arrows", 0.96), ("linetypes", 0.98)])


class TestRun(JobTestBase):

    def test_same_slug_second_start_is_rejected_before_state_mutation(self):
        slug = job.create_project(b"x", "one-owner.pdf")
        entered = threading.Event()
        release = threading.Event()

        def owned(*_args, **_kwargs):
            entered.set()
            self.assertTrue(release.wait(2))

        with mock.patch.object(job, "_run_owned", side_effect=owned):
            first = job.start_job(slug)
            self.assertTrue(entered.wait(2))
            with job._CANCEL_LOCK:
                original_cancel = job._CANCEL[slug]
            with self.assertRaises(job.JobStartError):
                job.start_job(slug, target="find every manhole")
            with job._CANCEL_LOCK:
                self.assertIs(job._CANCEL[slug], original_cancel)
            current = job.get_job(slug)
            self.assertEqual(current["target"], first["target"])
            job.request_cancel(slug)
            self.assertTrue(original_cancel.is_set())
            release.set()

        for _ in range(100):
            with job._STARTING_LOCK:
                if slug not in job._STARTING:
                    break
            threading.Event().wait(0.01)
        self.assertNotIn(slug, job._STARTING)

    def test_restart_claim_covers_cache_reset(self):
        slug = job.create_project(b"x", "atomic-rerun.pdf")
        job._set(slug, done=True, stage="done")
        reset_entered = threading.Event()
        release_reset = threading.Event()

        def slow_reset(_slug):
            reset_entered.set()
            self.assertTrue(release_reset.wait(2))
            return ["old-cache.json"]

        def claimed(*_args, **_kwargs):
            job._release_job_start(slug)
            return {"slug": slug, "stage": "queued"}

        with mock.patch.object(job, "reset_project_cache",
                               side_effect=slow_reset) as reset, \
                mock.patch.object(job, "_start_job_claimed",
                                  side_effect=claimed), \
                ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(job.restart_job, slug)
            self.assertTrue(reset_entered.wait(2))
            with self.assertRaises(job.JobStartError):
                job.restart_job(slug)
            release_reset.set()
            status, cleared = first.result(timeout=2)

        self.assertEqual(status["stage"], "queued")
        self.assertEqual(cleared, ["old-cache.json"])
        reset.assert_called_once_with(slug)

    def test_pre_pipeline_lock_error_becomes_terminal_failure(self):
        slug = job.create_project(b"x", "lock-error.pdf")
        job._set(slug, started=job.time.time(), done=False, stage="queued")
        with mock.patch.object(job, "stable_named_lock",
                               side_effect=OSError("flock unavailable")):
            job._run(slug, None)
        status = job.get_job(slug)
        self.assertTrue(status["done"])
        self.assertFalse(status["ok"])
        self.assertEqual(status["stage"], "error")
        self.assertIn("flock unavailable", status["error"])

    def test_two_projects_run_together_but_third_waits_for_capacity(self):
        slugs = [job.create_project(b"x", f"parallel-{i}.pdf")
                 for i in range(3)]
        for slug in slugs:
            job._set(slug, started=job.time.time(), done=False,
                     stage="queued")
        lock = threading.Lock()
        release = threading.Event()
        two_live = threading.Event()
        counts = {"live": 0, "peak": 0, "entered": 0}

        def fake_owned(slug, *_args, **_kwargs):
            with lock:
                counts["live"] += 1
                counts["entered"] += 1
                counts["peak"] = max(counts["peak"], counts["live"])
                if counts["live"] == 2:
                    two_live.set()
            release.wait(timeout=3)
            with lock:
                counts["live"] -= 1

        with mock.patch.object(job, "MAX_PARALLEL_JOBS", 2), \
                mock.patch.object(job, "_run_owned", side_effect=fake_owned):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(job._run, slug, None)
                           for slug in slugs]
                self.assertTrue(two_live.wait(timeout=2))
                with lock:
                    self.assertEqual(counts["entered"], 2)
                    self.assertEqual(counts["peak"], 2)
                release.set()
                for future in futures:
                    future.result(timeout=3)
        self.assertEqual(counts["entered"], 3)
        self.assertEqual(counts["peak"], 2)

    def test_queued_project_can_cancel_before_a_slot_is_free(self):
        slugs = [job.create_project(b"x", f"cancel-queued-{i}.pdf")
                 for i in range(3)]
        for slug in slugs:
            job._set(slug, started=job.time.time(), done=False,
                     stage="queued")
        release = threading.Event()
        two_live = threading.Event()
        lock = threading.Lock()
        live = {"count": 0, "entered": []}

        def fake_owned(slug, *_args, **_kwargs):
            with lock:
                live["count"] += 1
                live["entered"].append(slug)
                if live["count"] == 2:
                    two_live.set()
            release.wait(timeout=3)
            with lock:
                live["count"] -= 1

        with mock.patch.object(job, "MAX_PARALLEL_JOBS", 2), \
                mock.patch.object(job, "_run_owned", side_effect=fake_owned):
            with ThreadPoolExecutor(max_workers=3) as executor:
                first = [executor.submit(job._run, slug, None)
                         for slug in slugs[:2]]
                self.assertTrue(two_live.wait(timeout=2))
                queued = executor.submit(job._run, slugs[2], None)
                for _ in range(20):
                    if slugs[2] in job._CANCEL:
                        break
                    threading.Event().wait(0.02)
                job.request_cancel(slugs[2])
                queued.result(timeout=2)
                self.assertTrue(job.JOBS[slugs[2]]["cancelled"])
                self.assertEqual(job.JOBS[slugs[2]]["stage"], "cancelled")
                with lock:
                    self.assertNotIn(slugs[2], live["entered"])
                release.set()
                for future in first:
                    future.result(timeout=3)

    def test_auto_resume_skips_card_finished_by_old_worker(self):
        slug = job.create_project(b"x", "overlap.pdf")
        # Simulate the replacement worker's stale in-memory queued card while
        # the old worker has just published the authoritative done card.
        job.JOBS[slug] = {"slug": slug, "done": False, "stage": "queued"}
        finished = {"slug": slug, "done": True, "ok": True,
                    "stage": "done", "progress": 1.0}
        store.save_json(self.jobs / f"{slug}.json", finished)
        with mock.patch.object(job, "_run_owned") as owned:
            job._run(slug, None, _resume=True)
        owned.assert_not_called()
        self.assertEqual(job.JOBS[slug], finished)
        self.assertNotIn(slug, job._CANCEL)

    def test_cancel_requested_before_thread_registration_is_honoured(self):
        slug = job.create_project(b"x", "early-cancel.pdf")
        fake = _FakeStages(self)
        job._set(slug, started=job.time.time(), cancel_requested=True,
                 done=False)
        job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["cancelled"], st["stage"]),
                         (True, False, True, "cancelled"))
        self.assertEqual(fake.calls, [])

    def test_successful_run_publishes_cost_and_wall(self):
        slug = job.create_project(b"x", "ok.pdf")
        self.write_results(slug)
        fake = _FakeStages(self)
        job._set(slug, started=job.time.time())
        with mock.patch.object(job, "page_count_of", return_value=7):
            job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["stage"], st["error"]),
                         (True, True, "done", None))
        self.assertEqual(st["outcome"], "success")
        self.assertTrue(st["results_available"])
        self.assertEqual(fake.calls, ["_stage_text", "_stage_symbols",
                                      "_stage_views", "_stage_placements",
                                      "_stage_arrows", "_stage_linetypes"])
        res = store.load_json(store.results_path(slug), None)
        self.assertEqual(res["llm_summary"], job._ZERO_LLM)
        self.assertIsInstance(res["wall_seconds"], float)
        self.assertNotIn(slug, job._RUNNING)
        self.assertNotIn(slug, job._CANCEL)
        self.assertEqual(st["pages_total"], 7)

    def test_completed_run_with_unresolved_warnings_is_partial(self):
        slug = job.create_project(b"x", "partial.pdf")
        self.write_results(slug)
        _FakeStages(self, text=["P3 image scan still incomplete"])
        job._set(slug, started=job.time.time())
        job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["outcome"]),
                         (True, True, "partial"))
        self.assertTrue(st["results_available"])
        self.assertEqual(st["warnings"],
                         ["P3 image scan still incomplete"])

    def test_missing_required_stage_can_never_finish_as_success(self):
        slug = job.create_project(b"x", "incomplete.pdf")
        self.write_results(slug)
        job._set(slug, started=job.time.time(),
                 required_stages=list(job.FULL_PIPELINE_STAGES),
                 completed_stages=list(job.FULL_PIPELINE_STAGES[:-1]))
        job._finish(slug, ok=True, summary=job._ZERO_LLM)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["outcome"], st["stage"]),
                         (True, False, "failed", "error"))
        self.assertIn("linetypes", st["error"])

    def test_cancel_during_text_stage_publishes_nothing(self):
        slug = job.create_project(b"x", "cancel.pdf")
        fake = _FakeStages(self, on_text=job.request_cancel)
        job._set(slug, started=job.time.time())
        job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["cancelled"], st["stage"]),
                         (True, False, True, "cancelled"))
        self.assertEqual(fake.calls, ["_stage_text"])
        self.assertFalse((self.data / slug / "results.json").exists())

    def test_stage_exception_marks_error(self):
        slug = job.create_project(b"x", "boom.pdf")
        with mock.patch.object(job, "_stage_text",
                               side_effect=RuntimeError("kaboom")):
            job._set(slug, started=job.time.time())
            job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["stage"]),
                         (True, False, "error"))
        self.assertEqual(st["error"], "RuntimeError: kaboom")
        self.assertEqual(st["outcome"], "failed")

    def test_start_job_initialises_state_and_carries_baseline(self):
        slug = job.create_project(b"x", "resume.pdf")
        self.write_results(slug, {
            "llm_summary": {"calls": 7, "cost_usd": 1.25,
                            "peak_concurrency": 4},
            "wall_seconds": 42.0})
        with mock.patch.object(job, "_run") as run, \
                mock.patch.object(job, "page_count_of") as page_count:
            st = job.start_job(slug)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], slug)
        self.assertEqual(st["mode"], "fence")
        self.assertEqual(st["stage"], "queued")
        self.assertEqual(st["progress"], 0.0)
        self.assertEqual(st["pages_total"], 0)
        self.assertEqual(st["warnings"], [])
        self.assertIsNone(st["outcome"])
        self.assertFalse(st["results_available"])
        self.assertIsNone(st["stage_unit"])
        self.assertEqual(st["llm"]["calls"], 7)
        self.assertEqual(st["wall_seconds"], 42.0)
        self.assertEqual(st["target"], job.TARGET_DEFAULT)
        page_count.assert_not_called()

    def test_start_job_custom_target_switches_mode(self):
        slug = job.create_project(b"x", "custom.pdf")
        with mock.patch.object(job, "_run"):
            st = job.start_job(slug, target="find every manhole")
        self.assertEqual(st["mode"], "custom")
        self.assertEqual(st["target"], "find every manhole")
        self.assertIsNone(st["llm"])


# ------------------------------------------------- 阶段排活（真模块、零调用） --
def _page_rec(text="6 FT CHAIN LINK FENCE", box=None):
    """results.json 里一条最小页记录（items_of 只看 box_2d/text/label/tbl）。"""
    return {"vlm_items": [{"text": text, "box_2d": box or [10, 10, 20, 200],
                           "label": "legend entry"}],
            "vec_added": [], "vec_covered": [], "has_text": True}


def _symbol_entry(sig, symbols, groups):
    """一条当期的 symbols.json 缓存条目（骗过 has_current_symbols 的全部闸）。"""
    from core.config import MODEL_NAME, resolve_model
    from steps.symbols import sweep_version
    from steps.versions import SYMBOL_PROMPT_V, SYMBOL_VERSION
    raw = {"groups": list(groups), "symbols": list(symbols)}
    # sweep_v：步骤②b 补扫也参与当期判定，缺了这一项整页会被重新排活。
    return {"sig": sig, "v": SYMBOL_VERSION, "pv": SYMBOL_PROMPT_V,
            "model": MODEL_NAME, "sweep_v": sweep_version() or 0, "raw": raw,
            "result": {"symbols": list(symbols), "groups": list(groups)},
            "_model_check": resolve_model(None)}


def _view_entry(groups, revision, view_type="plan"):
    from core.config import resolve_model
    from steps.versions import VIEW_VERSION
    from steps.views import _view_rows, view_signature
    return {"sig": view_signature(groups, revision),
            "v": VIEW_VERSION, "model": resolve_model(None),
            "views": [{"group_index": row["group_index"],
                       "view_type": view_type, "reason": "test fixture"}
                      for row in _view_rows(groups)]}


SHAPE_SYM = {"box_2d": [100, 100, 120, 120], "category": "shape",
             "value": "33", "type": "shape 33", "text_index": 0,
             "group_index": 0}
LINE_SYM = {"box_2d": [100, 100, 110, 200], "category": "line",
            "value": "SF", "type": "line SF", "text_index": 0,
            "group_index": 0}
LEGEND_GROUP = {"box_2d": [50, 50, 300, 400], "kind": "legend"}
VIEW_GROUP = {"box_2d": [400, 50, 900, 900], "kind": "view"}


class TestStageJobCollection(JobTestBase):

    def setUp(self):
        super().setUp()
        self.slug = job.create_project(b"%PDF-1.4 fake", "collect.pdf")
        self.revision = store.pdf_revision(store.pdf_path(self.slug))
        self.rec = _page_rec()
        self.sig = store.sig_of(store.items_of(self.rec), self.revision)

    def _publish(self, pages):
        store.save_json(store.results_path(self.slug),
                        {"slug": self.slug, "fused_v": 2,
                         "page_count": len(pages), "pages": pages})

    def _symbols(self, cache):
        store.save_json(store.slug_dir(self.slug) / "symbols.json", cache)

    def test_symbol_jobs_skips_current_and_itemless_pages(self):
        empty = {"vlm_items": [], "vec_added": [], "vec_covered": []}
        self._publish({"1": self.rec, "2": self.rec, "3": empty})
        self._symbols({"1": _symbol_entry(self.sig, [], [LEGEND_GROUP])})
        self.assertEqual([p for p, _i, _s in job._symbol_jobs(self.slug)], [2])

    def test_symbol_jobs_requeues_stale_signature(self):
        self._publish({"1": self.rec})
        self._symbols({"1": _symbol_entry("deadbeef", [], [LEGEND_GROUP])})
        self.assertEqual([p for p, _i, _s in job._symbol_jobs(self.slug)], [1])

    def test_view_jobs_only_for_shape_plus_view_group(self):
        pages = {str(p): self.rec for p in range(1, 6)}
        self._publish(pages)
        self._symbols({
            # 1: shape + 合法 view 组 → 排活
            "1": _symbol_entry(self.sig, [SHAPE_SYM],
                               [LEGEND_GROUP, VIEW_GROUP]),
            # 2: 只有 line 样例 → 没有要匹配的放置，不付分类费
            "2": _symbol_entry(self.sig, [LINE_SYM],
                               [LEGEND_GROUP, VIEW_GROUP]),
            # 3: 有 shape 但整页没有 view 组 → 无从分类
            "3": _symbol_entry(self.sig, [SHAPE_SYM], [LEGEND_GROUP]),
            # 4: 一个符号都没有 → 不排活
            "4": _symbol_entry(self.sig, [], [VIEW_GROUP]),
            # 5: 符号检测不当期 → 跳过并记 warning
        })
        jobs, warnings = job._view_jobs(self.slug)
        self.assertEqual([p for p, _pdf, _g, _r in jobs], [1])
        self.assertEqual(len(warnings), 1)
        self.assertIn("P5", warnings[0])

    def test_view_jobs_skips_pages_already_classified(self):
        groups = [LEGEND_GROUP, VIEW_GROUP]
        self._publish({"1": self.rec})
        self._symbols({"1": _symbol_entry(self.sig, [SHAPE_SYM], groups)})
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(groups, self.revision)})
        jobs, warnings = job._view_jobs(self.slug)
        self.assertEqual((jobs, warnings), ([], []))


class TestPlacementsStage(JobTestBase):
    """本地放置阶段的编排（匹配器本体用假实现替掉，只验编排与 fail-closed）。"""

    def setUp(self):
        super().setUp()
        self.slug = job.create_project(b"%PDF-1.4 fake", "place.pdf")
        self.revision = store.pdf_revision(store.pdf_path(self.slug))
        self.rec = _page_rec()
        self.sig = store.sig_of(store.items_of(self.rec), self.revision)
        self.groups = [LEGEND_GROUP, VIEW_GROUP]
        store.save_json(store.results_path(self.slug),
                        {"slug": self.slug, "fused_v": 2, "page_count": 1,
                         "pages": {"1": self.rec}})
        self.cache_path = store.slug_dir(self.slug) / "symbols.json"
        store.save_json(self.cache_path,
                        {"1": _symbol_entry(self.sig, [dict(SHAPE_SYM)],
                                            self.groups)})
        self.seen = []

        def fake_match(pdf, page_index, symbols, typed_groups, *, dbg=None):
            self.seen.append((page_index, [g.get("view_type")
                                           for g in typed_groups]))
            for s in symbols:
                s["placements"] = [[500, 500, 520, 520]]
            if dbg is not None:
                dbg.add("placements", {"symbol_index": 0})
            from steps.placements import placement_scope_signature
            from steps.versions import PLACEMENT_VERSION
            return {"shape": 1, "line": 0, "placed": 1,
                    "dropped_outside_plan": 3, "plan_groups": 1,
                    "plc_v": PLACEMENT_VERSION,
                    "plc_scope_sig": placement_scope_signature(
                        symbols, typed_groups)}

        p = mock.patch("steps.placements.match_placements", fake_match)
        p.start()
        self.addCleanup(p.stop)

    def test_typed_groups_are_passed_and_result_updated(self):
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(self.groups, self.revision)})
        self.assertEqual(job._stage_placements(self.slug), [])
        self.assertEqual(self.seen, [(0, [None, "plan"])])
        result = store.load_json(self.cache_path, {})["1"]["result"]
        self.assertEqual(result["placed"], 1)
        self.assertEqual(result["dropped_outside_plan"], 3)
        self.assertEqual(result["symbols"][0]["placements"],
                         [[500, 500, 520, 520]])
        self.assertTrue(store.load_json(self.cache_path, {})["1"]["debug"])
        # 第二次是当期，零工作
        self.assertEqual(job._stage_placements(self.slug), [])
        self.assertEqual(len(self.seen), 1)

    def test_changed_view_output_recomputes_same_version_placements(self):
        views_path = store.slug_dir(self.slug) / "view_types.json"
        store.save_json(views_path,
                        {"1": _view_entry(self.groups, self.revision)})
        self.assertEqual(job._stage_placements(self.slug), [])
        self.assertEqual(self.seen, [(0, [None, "plan"])])

        # view_signature signs classifier inputs; a forced rerun may change
        # its output under the same signature.  Placement scope must still
        # notice that the effective plan set changed.
        store.save_json(views_path, {"1": _view_entry(
            self.groups, self.revision, view_type="elevation")})
        self.assertEqual(job._stage_placements(self.slug), [])
        self.assertEqual(self.seen[-1], (0, [None, "elevation"]))
        self.assertEqual(len(self.seen), 2)

    def test_verified_row_code_is_added_before_the_production_matcher(self):
        rec = _page_rec()
        rec["vlm_items"][0]["vec_backed"] = True
        sig = store.sig_of(store.items_of(rec), self.revision)
        parent = {**SHAPE_SYM, "value": "4.0", "type": "shape 4.0"}
        store.save_json(store.results_path(self.slug),
                        {"slug": self.slug, "fused_v": 2, "page_count": 1,
                         "pages": {"1": rec}})
        store.save_json(self.cache_path,
                        {"1": _symbol_entry(sig, [parent], self.groups)})
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(self.groups, self.revision)})
        pdf = store.pdf_path(self.slug)
        lines = [{"text": "4.0", "box_2d": [100, 100, 110, 110]},
                 {"text": "4.6", "box_2d": [10, 20, 20, 30]}]
        store.save_json(store.slug_dir(self.slug) / "vec.json", {
            "schema": job.VEC_SCHEMA, "pdf_mtime": pdf.stat().st_mtime,
            "page_count": 1, "pages": {"1": {"lines": lines}}})

        def fake_snap(_pdf, _page, symbols):
            symbols[0]["snap"] = "shape"
            return {"snap_shape": 1}

        def fake_inherit(entry, items, vector_lines):
            self.assertEqual(items, store.items_of(rec))
            self.assertEqual(vector_lines, lines)
            entry["result"]["symbols"].append({
                "box_2d": [10, 20, 20, 30], "category": "shape",
                "value": "4.6", "text_index": 0, "source": "row_code",
                "snap": "inherited"})
            return 1

        with mock.patch("steps.snap_boxes.snap_symbol_boxes",
                        side_effect=fake_snap), \
                mock.patch("steps.symbols.inherit_row_code_symbols",
                           side_effect=fake_inherit) as inherit:
            self.assertEqual(job._stage_placements(self.slug), [])
        inherit.assert_called_once()
        result = store.load_json(self.cache_path, {})["1"]["result"]
        self.assertEqual([s["value"] for s in result["symbols"]],
                         ["4.0", "4.6"])
        # fake_match in setUp sees both formal symbols and runs the same
        # production-stage placement path on the derived one.
        self.assertEqual(result["symbols"][1]["placements"],
                         [[500, 500, 520, 520]])
        self.assertEqual(result["snap_inherited"], 1)

    def test_missing_vector_cache_does_not_stamp_row_code_page_current(self):
        rec = _page_rec()
        rec["vlm_items"][0]["vec_backed"] = True
        sig = store.sig_of(store.items_of(rec), self.revision)
        parent = {**SHAPE_SYM, "value": "4.0", "type": "shape 4.0"}
        store.save_json(store.results_path(self.slug),
                        {"slug": self.slug, "fused_v": 2, "page_count": 1,
                         "pages": {"1": rec}})
        store.save_json(self.cache_path,
                        {"1": _symbol_entry(sig, [parent], self.groups)})
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(self.groups, self.revision)})
        warnings = job._stage_placements(self.slug)
        self.assertEqual(len(warnings), 1)
        self.assertIn("native vector text is unavailable", warnings[0])
        result = store.load_json(self.cache_path, {})["1"]["result"]
        self.assertNotIn("plc_v", result)
        self.assertIn("row_code_error", result)
        self.assertEqual(self.seen, [])

    def test_missing_view_classification_is_fail_closed(self):
        warnings = job._stage_placements(self.slug)
        self.assertEqual(self.seen, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("view classification is stale", warnings[0])
        self.assertNotIn("plc_v", store.load_json(self.cache_path,
                                                  {})["1"]["result"])

    def test_line_only_page_runs_without_view_classification(self):
        """只有 line 样例的页不需要 plan 取景框，但**必须**盖上 plc_v.

        真实故障：taylor_3_12 P3 只有一条 line 样例，页上又有 view 组，于是
        放置阶段一直因"视图分类不当期"跳过、plc_v 永远是 None；webapp 的发布闸
        要求符号与放置都当期，结果把这一页**整个符号层**都扣住了 —— 用户看到的是
        "图例里那条线根本没框出来"。
        """
        line_sym = {"category": "line", "value": "X", "text_index": 0,
                    "group_index": 0, "box_2d": [400, 460, 410, 495]}
        store.save_json(self.cache_path,
                        {"1": _symbol_entry(self.sig, [line_sym], self.groups)})
        warnings = job._stage_placements(self.slug)
        self.assertEqual(warnings, [])            # 不是故障，不该报 warning
        self.assertEqual(self.seen, [(0, [None, None])])   # 用未合并的组
        result = store.load_json(self.cache_path, {})["1"]["result"]
        from steps.versions import PLACEMENT_VERSION
        self.assertEqual(result["plc_v"], PLACEMENT_VERSION)

    def test_shape_page_still_needs_view_classification(self):
        """有 shape 的页仍然 fail-closed —— 别把上面那条放宽成"全都不用分类"。"""
        warnings = job._stage_placements(self.slug)
        self.assertEqual(self.seen, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("view classification is stale", warnings[0])

    def test_page_level_matcher_blowup_is_a_warning_not_a_dead_job(self):
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(self.groups, self.revision)})
        with mock.patch("steps.placements.match_placements",
                        side_effect=RuntimeError("vector boom")):
            warnings = job._stage_placements(self.slug)
        self.assertEqual(len(warnings), 1)
        self.assertIn("RuntimeError: vector boom", warnings[0])
        self.assertNotIn("plc_v", store.load_json(self.cache_path,
                                                  {})["1"]["result"])

    def test_cancel_stops_before_touching_pages(self):
        store.save_json(store.slug_dir(self.slug) / "view_types.json",
                        {"1": _view_entry(self.groups, self.revision)})
        self.assertEqual(job._stage_placements(self.slug,
                                               should_cancel=lambda: True), [])
        self.assertEqual(self.seen, [])


class TestLinetypeCompletionBounds(JobTestBase):
    """A bad local clustering page must have a small, deterministic tail."""

    def setUp(self):
        super().setUp()
        self.slug = job.create_project(b"%PDF-1.4 fake", "lines.pdf")
        self.items = [{"text": "FENCE", "box_2d": [1, 1, 2, 2],
                       "label": "callout", "tbl": False}]
        self.write_results(self.slug, {"pages": {"1": {
            "vlm_items": [dict(self.items[0])], "vec_added": []}}})
        revision = store.pdf_revision(store.pdf_path(self.slug))
        arrows_sig = job.arrows.arrows_signature(
            self.items, revision, [])
        self.arrow_entry = {
            "sig": arrows_sig, "v": job.arrows.ARROWS_VERSION,
            "geometry": {"state": "vector",
                         "vector_paths": job.LINETYPE_DENSE_PATHS - 1},
            "items": {"0": {"targets": [{"tip": [10, 10]}]}}}
        store.save_json(store.slug_dir(self.slug) / "arrows.json",
                        {"1": self.arrow_entry})
        self.sig = job.linetypes.linetypes_signature(arrows_sig)
        self.success = {"sig": self.sig,
                        "v": job.linetypes.LINETYPE_VERSION,
                        "used_all": [3],
                        "bindings": [], "line_types": []}

    def test_current_fence_placement_schedules_without_an_arrow_target(self):
        symbols = [{
            "category": "shape", "text_index": 0,
            "placements": [[100, 200, 120, 240]],
        }]
        revision = store.pdf_revision(store.pdf_path(self.slug))
        symbol_entry = _symbol_entry(
            store.sig_of(self.items, revision), symbols, [])
        from steps.versions import PLACEMENT_VERSION
        symbol_entry["result"]["plc_v"] = PLACEMENT_VERSION
        from steps.placements import placement_scope_signature
        symbol_entry["result"]["plc_scope_sig"] = \
            placement_scope_signature(symbols, [])
        symbol_result = symbol_entry["result"]
        store.save_json(store.slug_dir(self.slug) / "symbols.json",
                        {"1": symbol_entry})
        extra = job._placement_anchors(symbol_result)
        arrows_sig = job.arrows.arrows_signature(
            self.items, revision, extra)
        arrow_entry = {
            "sig": arrows_sig, "v": job.arrows.ARROWS_VERSION,
            "geometry": {"state": "vector", "vector_paths": 10},
            # The symbol center is deliberately the only usable anchor.
            "items": {},
        }
        store.save_json(store.slug_dir(self.slug) / "arrows.json",
                        {"1": arrow_entry})

        jobs, warnings = job._linetype_jobs(self.slug)
        self.assertEqual(warnings, [])
        self.assertEqual(len(jobs), 1)
        page, items, captured_arrows, sig = jobs[0]
        self.assertEqual(page, 1)
        self.assertEqual(items, self.items)
        self.assertEqual(captured_arrows, arrow_entry)
        self.assertEqual(
            sig,
            job.linetypes.linetypes_signature(
                arrows_sig, symbol_result, self.items))
        anchors = job.linetypes.anchors_of(
            captured_arrows, symbol_result, items)
        self.assertEqual([anchor["anchor_kind"] for anchor in anchors],
                         ["symbol_center"])

    def test_linetype_worker_passes_current_symbol_result_to_compute(self):
        symbols = [{
            "category": "shape", "text_index": 0,
            "placements": [], "unit_marker": "current-symbol-result",
        }]
        revision = store.pdf_revision(store.pdf_path(self.slug))
        symbol_entry = _symbol_entry(
            store.sig_of(self.items, revision), symbols, [])
        from steps.versions import PLACEMENT_VERSION
        symbol_entry["result"]["plc_v"] = PLACEMENT_VERSION
        from steps.placements import placement_scope_signature
        symbol_entry["result"]["plc_scope_sig"] = \
            placement_scope_signature(symbols, [])
        symbol_result = symbol_entry["result"]
        store.save_json(store.slug_dir(self.slug) / "symbols.json",
                        {"1": symbol_entry})
        with mock.patch.object(job.linetypes, "compute_page_linetypes",
                               return_value=self.success) as compute:
            page, count, error = job._linetype_one(
                self.slug, 1, self.items, self.arrow_entry, self.sig)
        self.assertEqual((page, count, error), (1, 1, None))
        self.assertEqual(compute.call_args.kwargs["symbol_result"],
                         symbol_result)

    def test_timeout_is_adaptive_at_dense_path_boundary(self):
        self.assertEqual(job._linetype_timeout_for(
            self.slug, 1, self.arrow_entry), job.LINETYPE_TIMEOUT)
        dense = {**self.arrow_entry,
                 "geometry": {"vector_paths": job.LINETYPE_DENSE_PATHS}}
        self.assertEqual(job._linetype_timeout_for(
            self.slug, 1, dense), job.LINETYPE_DENSE_TIMEOUT)

        with mock.patch.object(job.linetypes, "compute_page_linetypes",
                               return_value=self.success) as compute:
            page, count, error = job._linetype_one(
                self.slug, 1, self.items, dense, self.sig)
        self.assertEqual((page, count, error), (1, 1, None))
        self.assertEqual(compute.call_args.kwargs["timeout"],
                         job.LINETYPE_DENSE_TIMEOUT)

    def test_old_timeout_marker_is_retryable_only_with_larger_budget(self):
        normal_error = "linetype sidecar timeout after 600s (sheet 1)"
        old_dense_error = "linetype sidecar timeout after 1800s (sheet 1)"
        current_dense_error = (
            f"linetype sidecar timeout after {job.LINETYPE_DENSE_TIMEOUT}s "
            "(sheet 1)"
        )
        dense = {**self.arrow_entry,
                 "geometry": {"vector_paths": job.LINETYPE_DENSE_PATHS}}
        self.assertFalse(job._linetype_failure_budget_increased(
            self.slug, 1, self.arrow_entry, normal_error))
        self.assertTrue(job._linetype_failure_budget_increased(
            self.slug, 1, dense, normal_error))
        self.assertTrue(job._linetype_failure_budget_increased(
            self.slug, 1, dense, old_dense_error))
        self.assertFalse(job._linetype_failure_budget_increased(
            self.slug, 1, dense, current_dense_error))
        self.assertFalse(job._linetype_failure_budget_increased(
            self.slug, 1, dense, "deterministic PAGE_IR_ERROR"))

    def test_arrow_success_persists_geometry_for_linetype_deadline(self):
        geometry = {"state": "vector", "vector_paths": 12345,
                    "images": 0, "image_coverage": 0.0}
        found = {0: {"targets": [{"tip": [10, 10]}]}}
        diagnostics = {0: {"state": "found"}}
        with mock.patch.object(job.arrows, "page_geometry_status",
                               return_value=geometry), \
                mock.patch.object(job.arrows, "find_page_arrows",
                                  return_value=(found, diagnostics)):
            page, count, error = job._arrow_one(
                self.slug, 1, self.items, "arrow-sig", [], [])
        self.assertEqual((page, count, error), (1, 1, None))
        cached = store.load_json(
            store.slug_dir(self.slug) / "arrows.json", {})["1"]
        self.assertEqual(cached["geometry"], geometry)

    def test_arrow_waiting_for_heavy_slot_can_cancel(self):
        cancelled = threading.Event()
        geometry_seen = threading.Event()

        def geometry(*_args, **_kwargs):
            geometry_seen.set()
            return {"state": "vector", "vector_paths": 10}

        with mock.patch.object(job, "HEAVY_SIDECAR_SLOTS", 1), \
                mock.patch.object(job.arrows, "page_geometry_status",
                                  side_effect=geometry), \
                mock.patch.object(job.arrows,
                                  "find_page_arrows") as compute, \
                job._slot_pool("heavy-sidecar", 1).slot(), \
                ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                job._arrow_one, self.slug, 1, self.items, "arrow-sig", [], [],
                cancelled.is_set)
            self.assertTrue(geometry_seen.wait(2))
            cancelled.set()
            with self.assertRaises(job.Cancelled):
                future.result(timeout=2)
        compute.assert_not_called()

    def test_linetype_waiting_for_heavy_slot_can_cancel(self):
        cancelled = threading.Event()
        with mock.patch.object(job, "HEAVY_SIDECAR_SLOTS", 1), \
                mock.patch.object(job.linetypes,
                                  "compute_page_linetypes") as compute, \
                job._slot_pool("heavy-sidecar", 1).slot(), \
                ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                job._linetype_one, self.slug, 1, self.items,
                self.arrow_entry, self.sig, cancelled.is_set)
            threading.Event().wait(0.1)
            cancelled.set()
            with self.assertRaises(job.Cancelled):
                future.result(timeout=2)
        compute.assert_not_called()

    def test_timeout_and_structured_errors_are_not_retried(self):
        errors = [
            "linetype sidecar timeout after 600s (sheet 1)",
            "linetype sidecar PAGE_IR_ERROR: bad page",
            "SourceAlignmentError: source paints disagree",
        ]
        for message in errors:
            with self.subTest(message=message), \
                    mock.patch.object(
                        job.linetypes, "compute_page_linetypes",
                        side_effect=RuntimeError(message)) as compute, \
                    mock.patch("time.sleep") as sleep:
                page, count, error = job._linetype_one(
                    self.slug, 1, self.items, self.arrow_entry, self.sig)
            self.assertEqual((page, count), (1, None))
            self.assertIn(message, error)
            self.assertEqual(compute.call_count, 1)
            sleep.assert_not_called()

    def test_transient_sidecar_exit_keeps_bounded_retry(self):
        with mock.patch.object(
                job.linetypes, "compute_page_linetypes",
                side_effect=[RuntimeError("sidecar exit 137 with no output"),
                             self.success]) as compute, \
                mock.patch("time.sleep") as sleep:
            page, count, error = job._linetype_one(
                self.slug, 1, self.items, self.arrow_entry, self.sig)
        self.assertEqual((page, count, error), (1, 1, None))
        self.assertEqual(compute.call_count, 2)
        sleep.assert_called_once_with(5)

    def test_stale_prerequisite_skips_before_compute_and_write(self):
        with mock.patch.object(job, "_linetype_jobs",
                               return_value=([], [])), \
                mock.patch.object(job.linetypes,
                                  "compute_page_linetypes") as compute, \
                mock.patch.object(job.linetypes, "save_page") as save:
            result = job._linetype_one(
                self.slug, 1, self.items, self.arrow_entry, self.sig)
        self.assertEqual(result, (1, 0, None))
        compute.assert_not_called()
        save.assert_not_called()

    def test_superseded_prerequisite_discards_computed_entry(self):
        captured = (1, self.items, self.arrow_entry, self.sig)
        with mock.patch.object(
                job, "_linetype_jobs",
                side_effect=[([captured], []), ([], [])]) as jobs, \
                mock.patch.object(
                    job.linetypes, "compute_page_linetypes",
                    return_value=self.success) as compute, \
                mock.patch.object(job.linetypes, "save_page") as save:
            result = job._linetype_one(
                self.slug, 1, self.items, self.arrow_entry, self.sig)
        self.assertEqual(result, (1, 0, None))
        self.assertEqual(jobs.call_count, 2)
        compute.assert_called_once()
        save.assert_not_called()

    def test_same_page_callers_compute_once_under_advisory_lock(self):
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()

        def compute(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release line-type compute")
            return self.success

        def second_call():
            second_started.set()
            return job._linetype_one(
                self.slug, 1, self.items, self.arrow_entry, self.sig)

        with mock.patch.object(
                job.linetypes, "compute_page_linetypes",
                side_effect=compute) as run_compute, \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                job._linetype_one, self.slug, 1, self.items,
                self.arrow_entry, self.sig)
            self.assertTrue(entered.wait(2))
            second = pool.submit(second_call)
            self.assertTrue(second_started.wait(2))
            release.set()
            self.assertEqual(first.result(timeout=2), (1, 1, None))
            self.assertEqual(second.result(timeout=2), (1, 0, None))

        self.assertEqual(run_compute.call_count, 1)
        self.assertTrue(job._linetype_page_lock_path(
            self.slug, 1).is_file())

    def test_materialize_all_valid_cache_skips_heavy_slot_and_sidecar(self):
        main = dict(self.success)
        cached = {"cache": "raw"}
        canonical = {"cache": "validated"}

        with mock.patch.object(job, "_current_main_all_source",
                               return_value=(main, self.arrow_entry)) as current, \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value=cached) as load_all, \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=canonical) as validate, \
                mock.patch.object(job, "_slot_pool") as slot_pool, \
                mock.patch.object(
                    job.linetypes,
                    "compute_all_page_geometry") as compute, \
                mock.patch.object(job.linetypes, "save_all_page") as save:
            result = job.materialize_all_linetypes(
                self.slug, 1, self.sig)

        self.assertIs(result, canonical)
        current.assert_called_once_with(self.slug, 1, self.sig)
        load_all.assert_called_once_with(self.slug, 1)
        validate.assert_called_once_with(cached, main, self.sig)
        slot_pool.assert_not_called()
        compute.assert_not_called()
        save.assert_not_called()

    def test_current_main_all_source_rechecks_live_prerequisites(self):
        job.linetypes.save_page(self.slug, 1, dict(self.success))
        main, arrow = job._current_main_all_source(
            self.slug, 1, self.sig)
        self.assertEqual(main, self.success)
        self.assertEqual(arrow, self.arrow_entry)

        results = store.load_json(store.results_path(self.slug), {})
        results["pages"]["1"]["vlm_items"][0]["text"] = "CHANGED FENCE"
        store.save_json(store.results_path(self.slug), results)
        self.assertEqual(
            job._current_main_all_source(self.slug, 1, self.sig),
            (None, None))

    def test_materialize_all_missing_cache_computes_verifies_and_saves(self):
        main = dict(self.success)
        generated = {"geometry": "fresh"}
        canonical = {"geometry": "verified"}
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()

        with mock.patch.object(
                job, "_current_main_all_source",
                side_effect=[(main, self.arrow_entry),
                             (main, self.arrow_entry)]) as current, \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value=None), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=None), \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool) as slot_pool, \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    return_value=generated) as compute, \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry",
                    return_value=canonical) as verify, \
                mock.patch.object(job.linetypes,
                                  "save_all_page") as save:
            result = job.materialize_all_linetypes(
                self.slug, 1, self.sig)

        self.assertIs(result, canonical)
        self.assertEqual(current.call_count, 2)
        slot_pool.assert_called_once_with(
            "heavy-sidecar", job.HEAVY_SIDECAR_SLOTS)
        heavy_pool.slot.assert_called_once_with()
        compute.assert_called_once_with(
            job.pdf_path(self.slug), 1, main, timeout=job.LINETYPE_TIMEOUT)
        verify.assert_called_once_with(main, generated)
        save.assert_called_once_with(self.slug, 1, canonical)

    def test_materialize_all_discards_result_if_main_changes_during_compute(
            self):
        main = dict(self.success)
        replacement = {**main, "sig": f"{self.sig}-replacement"}
        generated = {"geometry": "fresh"}
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()

        with mock.patch.object(
                job, "_current_main_all_source",
                side_effect=[(main, self.arrow_entry), (None, None)]), \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value=None), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=None), \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool), \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    return_value=generated) as compute, \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry") as verify, \
                mock.patch.object(job.linetypes,
                                  "save_all_page") as save:
            with self.assertRaisesRegex(
                    RuntimeError, "changed during generation"):
                job.materialize_all_linetypes(self.slug, 1, self.sig)

        compute.assert_called_once()
        verify.assert_not_called()
        save.assert_not_called()

    def test_materialize_all_same_page_callers_compute_once(self):
        main = dict(self.success)
        generated = {"geometry": "fresh"}
        canonical = {"geometry": "verified"}
        cached = {"entry": None}
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()

        def load_all(_slug, _page):
            return cached["entry"]

        def validate(entry, _main, _sig):
            return entry if entry is canonical else None

        def compute(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError(
                    "test did not release all-line-types compute")
            return generated

        def save(_slug, _page, entry):
            cached["entry"] = entry

        def second_call():
            second_started.set()
            return job.materialize_all_linetypes(
                self.slug, 1, self.sig)

        with mock.patch.object(job, "_current_main_all_source",
                               return_value=(main, self.arrow_entry)), \
                mock.patch.object(job.linetypes, "load_all_page",
                                  side_effect=load_all), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  side_effect=validate), \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool), \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    side_effect=compute) as run_compute, \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry",
                    return_value=canonical) as verify, \
                mock.patch.object(job.linetypes, "save_all_page",
                                  side_effect=save) as persist, \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                job.materialize_all_linetypes, self.slug, 1, self.sig)
            self.assertTrue(entered.wait(2))
            second = pool.submit(second_call)
            self.assertTrue(second_started.wait(2))
            release.set()
            self.assertIs(first.result(timeout=2), canonical)
            self.assertIs(second.result(timeout=2), canonical)

        run_compute.assert_called_once()
        verify.assert_called_once_with(main, generated)
        persist.assert_called_once_with(self.slug, 1, canonical)
        self.assertTrue(job._linetype_page_lock_path(
            self.slug, 1).is_file())


class TestLegendLinetypeOrchestration(JobTestBase):
    """Legend swatches are a supervised line-type channel, not arrow jobs."""

    def setUp(self):
        super().setUp()
        self.slug = job.create_project(b"%PDF-1.4 fake", "legend-lines.pdf")
        self.rec = _page_rec()
        self.items = store.items_of(self.rec)
        self.revision = store.pdf_revision(store.pdf_path(self.slug))
        self.symbol_sig = store.sig_of(self.items, self.revision)
        store.save_json(store.results_path(self.slug), {
            "slug": self.slug, "fused_v": 2, "page_count": 1,
            "pages": {"1": self.rec},
        })
        symbol_entry = _symbol_entry(
            self.symbol_sig, [dict(LINE_SYM)], [LEGEND_GROUP])
        store.save_json(store.slug_dir(self.slug) / "symbols.json", {
            "1": symbol_entry,
        })
        self.samples = job.legend_linetypes.samples_of(symbol_entry["result"])
        self.sig = job.legend_linetypes.signature(
            self.revision, self.samples)

    def _complete_legend_cache(self):
        return {
            "sig": self.sig, "v": job.legend_linetypes.VERSION,
            "ok": True, "line_types": [], "bindings": [],
            "page": {
                "base_line_types": 1,
                "page_fingerprint": "page",
                "owned_ops_sha1": "owned",
                "fused_ops_sha1": "fused",
                "path_ops": 1,
                "owned_path_ops": 1,
            },
            "engine_all_line_types": [{
                "line_type_number": 1,
                "signature_family": "motif_periodic",
                "recognition_source": "method1",
                "op_count": 1, "ops_sha1": "one",
                "segment_count": 1,
                "pattern_instance_count": 0,
                "pattern_instances": [],
            }],
        }

    def test_jobs_need_current_line_symbols_but_no_arrows_or_placements(self):
        self.assertFalse((store.slug_dir(self.slug) / "arrows.json").exists())
        self.assertNotIn("placements", self.samples[0])

        jobs, warnings = job._legend_linetype_jobs(self.slug)

        self.assertEqual(warnings, [])
        self.assertEqual(jobs, [(1, self.samples, self.sig)])

    def test_current_success_cache_skips_job(self):
        job.legend_linetypes.save_page(self.slug, 1, {
            "sig": self.sig, "v": job.legend_linetypes.VERSION,
            "ok": True, "line_types": [], "bindings": [],
        })
        self.assertEqual(job._legend_linetype_jobs(self.slug), ([], []))

    def test_current_legend_all_source_rechecks_symbols_and_projects_audit(self):
        cache = self._complete_legend_cache()
        job.legend_linetypes.save_page(self.slug, 1, cache)

        entry, audit = job._current_legend_all_source(
            self.slug, 1, self.sig)

        self.assertEqual(entry, cache)
        self.assertEqual(audit["sig"], self.sig)
        self.assertEqual(len(audit["all_line_types"]), 1)

        symbols_path = store.slug_dir(self.slug) / "symbols.json"
        symbols = store.load_json(symbols_path, {})
        symbols["1"]["result"]["symbols"][0]["value"] = "changed"
        store.save_json(symbols_path, symbols)
        self.assertEqual(
            job._current_legend_all_source(self.slug, 1, self.sig),
            (None, None))

    def test_materialize_all_from_legend_computes_pinned_and_revalidates(self):
        entry = self._complete_legend_cache()
        audit = job.legend_linetypes.all_audit_entry(entry, self.sig)
        latest = {**audit, "latest": True}
        generated = {"geometry": "fresh"}
        canonical = {"geometry": "verified"}
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()
        with mock.patch.object(
                job, "_current_legend_all_source",
                side_effect=[(entry, audit), (entry, latest)]) as current, \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value=None), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=None), \
                mock.patch.object(job, "_legend_linetype_timeout_for",
                                  return_value=123) as timeout, \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool) as slot_pool, \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    return_value=generated) as compute, \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry",
                    return_value=canonical) as verify, \
                mock.patch.object(job.linetypes, "save_all_page") as save:
            result = job.materialize_all_linetypes_from_legend(
                self.slug, 1, self.sig)

        self.assertIs(result, canonical)
        self.assertEqual(current.call_count, 2)
        timeout.assert_called_once_with(self.slug, 1)
        slot_pool.assert_called_once_with(
            "heavy-sidecar", job.HEAVY_SIDECAR_SLOTS)
        compute.assert_called_once_with(
            store.pdf_path(self.slug), 1, audit, timeout=123,
            cpu_budget=job.legend_linetypes.sidecar.CPU_BUDGET)
        verify.assert_called_once_with(latest, generated)
        save.assert_called_once_with(self.slug, 1, canonical)

    def test_materialize_all_from_legend_discards_changed_prerequisite(self):
        entry = self._complete_legend_cache()
        audit = job.legend_linetypes.all_audit_entry(entry, self.sig)
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()
        with mock.patch.object(
                job, "_current_legend_all_source",
                side_effect=[(entry, audit), (None, None)]), \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value=None), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=None), \
                mock.patch.object(job, "_legend_linetype_timeout_for",
                                  return_value=123), \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool), \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    return_value={"geometry": "fresh"}), \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry") as verify, \
                mock.patch.object(job.linetypes, "save_all_page") as save:
            with self.assertRaisesRegex(RuntimeError, "during generation"):
                job.materialize_all_linetypes_from_legend(
                    self.slug, 1, self.sig)
        verify.assert_not_called()
        save.assert_not_called()

    def test_materialize_all_from_legend_valid_cache_skips_sidecar(self):
        entry = self._complete_legend_cache()
        audit = job.legend_linetypes.all_audit_entry(entry, self.sig)
        cached = {"geometry": "verified"}
        with mock.patch.object(
                job, "_current_legend_all_source",
                return_value=(entry, audit)), \
                mock.patch.object(job.linetypes, "load_all_page",
                                  return_value={"raw": "cache"}), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  return_value=cached) as validate, \
                mock.patch.object(job, "_slot_pool") as slot_pool, \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry") as compute:
            result = job.materialize_all_linetypes_from_legend(
                self.slug, 1, self.sig)
        self.assertIs(result, cached)
        validate.assert_called_once()
        slot_pool.assert_not_called()
        compute.assert_not_called()

    def test_materialize_all_from_legend_same_page_callers_compute_once(self):
        entry = self._complete_legend_cache()
        audit = job.legend_linetypes.all_audit_entry(entry, self.sig)
        generated = {"geometry": "fresh"}
        canonical = {"geometry": "verified"}
        cache = {"entry": None}
        entered = threading.Event()
        release = threading.Event()
        heavy_pool = mock.Mock()
        heavy_pool.slot.return_value = mock.MagicMock()

        def validate(value, _audit, _sig):
            return canonical if value is canonical else None

        def compute(*_args, **_kwargs):
            entered.set()
            if not release.wait(2):
                raise AssertionError("test did not release legend All compute")
            return generated

        def save(_slug, _page, value):
            cache["entry"] = value

        with mock.patch.object(
                job, "_current_legend_all_source",
                return_value=(entry, audit)), \
                mock.patch.object(job.linetypes, "load_all_page",
                                  side_effect=lambda *_: cache["entry"]), \
                mock.patch.object(job.linetypes, "validated_all_page",
                                  side_effect=validate), \
                mock.patch.object(job, "_legend_linetype_timeout_for",
                                  return_value=123), \
                mock.patch.object(job, "_slot_pool",
                                  return_value=heavy_pool), \
                mock.patch.object(
                    job.linetypes, "compute_all_page_geometry",
                    side_effect=compute) as run_compute, \
                mock.patch.object(
                    job.linetypes, "verify_all_page_geometry",
                    return_value=canonical), \
                mock.patch.object(job.linetypes, "save_all_page",
                                  side_effect=save), \
                ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(
                job.materialize_all_linetypes_from_legend,
                self.slug, 1, self.sig)
            self.assertTrue(entered.wait(2))
            second = pool.submit(
                job.materialize_all_linetypes_from_legend,
                self.slug, 1, self.sig)
            release.set()
            self.assertIs(first.result(timeout=2), canonical)
            self.assertIs(second.result(timeout=2), canonical)
        run_compute.assert_called_once()

    def test_one_success_uses_pdf_geometry_timeout_and_saves(self):
        success = {
            "sig": self.sig, "v": job.legend_linetypes.VERSION,
            "ok": True, "line_types": [{"line_type_number": 7}],
            "bindings": [{"key": "s0:0"}],
        }
        geometry = {"vector_paths": job.LINETYPE_DENSE_PATHS}
        with mock.patch.object(job.arrows, "page_geometry_status",
                               return_value=geometry) as page_geometry, \
                mock.patch.object(job.legend_linetypes, "compute_page",
                                  return_value=success) as compute:
            result = job._legend_linetype_one(
                self.slug, 1, self.samples, self.sig)

        self.assertEqual(result, (1, 1, None))
        page_geometry.assert_called_once_with(store.pdf_path(self.slug), 0)
        compute.assert_called_once_with(
            store.pdf_path(self.slug), 1, self.samples, sig=self.sig,
            timeout=job.LINETYPE_DENSE_TIMEOUT)
        self.assertEqual(
            job.legend_linetypes.load_page(self.slug, 1), success)

    def test_one_failure_saves_retryable_non_success_shape(self):
        with mock.patch.object(job.arrows, "page_geometry_status",
                               return_value={"vector_paths": 1}), \
                mock.patch.object(
                    job.legend_linetypes, "compute_page",
                    side_effect=RuntimeError(
                        "SourceAlignmentError: source paints disagree")) \
                as compute:
            page, count, error = job._legend_linetype_one(
                self.slug, 1, self.samples, self.sig)

        self.assertEqual((page, count), (1, None))
        self.assertIn("SourceAlignmentError", error)
        compute.assert_called_once()
        failed = job.legend_linetypes.load_page(self.slug, 1)
        self.assertEqual(failed["sig"], self.sig)
        self.assertEqual(failed["v"], job.legend_linetypes.VERSION)
        self.assertIn("SourceAlignmentError", failed["error"])
        self.assertNotIn("ok", failed)
        self.assertFalse(job.legend_linetypes.has_current(failed, self.sig))

    def test_stage_schedules_both_channels_with_combined_progress_warning(self):
        arrow_job = (1, self.items, {"items": {}}, "arrow-sig")
        legend_job = (1, self.samples, self.sig)
        progress = []
        with mock.patch.object(job.arrows, "ENABLED", True), \
                mock.patch.object(job, "_linetype_jobs",
                                  return_value=([arrow_job], [])), \
                mock.patch.object(job, "_legend_linetype_jobs",
                                  return_value=([legend_job], [])), \
                mock.patch.object(job, "_linetype_one",
                                  return_value=(1, 2, None)) as ordinary, \
                mock.patch.object(
                    job, "_legend_linetype_one",
                    return_value=(1, None, "RuntimeError: no match")) as legend:
            warnings = job._stage_linetypes(
                self.slug, on_progress=lambda done, total:
                progress.append((done, total)))

        ordinary.assert_called_once_with(self.slug, *arrow_job, None)
        legend.assert_called_once_with(self.slug, *legend_job, None)
        self.assertEqual(progress[0], (0, 2))
        self.assertEqual(progress[-1], (2, 2))
        self.assertEqual(len(progress), 3)
        self.assertEqual(len(warnings), 1)
        self.assertIn("legend line-type matching failed", warnings[0])

    def test_stage_with_arrows_toggle_off_dispatches_only_legend(self):
        legend_job = (1, self.samples, self.sig)
        with mock.patch.object(job.arrows, "ENABLED", False), \
                mock.patch.object(job, "_linetype_jobs") as ordinary_jobs, \
                mock.patch.object(job, "_legend_linetype_jobs",
                                  return_value=([legend_job], [])), \
                mock.patch.object(job, "_linetype_one") as ordinary, \
                mock.patch.object(job, "_legend_linetype_one",
                                  return_value=(1, 1, None)) as legend:
            warnings = job._stage_linetypes(self.slug)

        self.assertEqual(warnings, [])
        ordinary_jobs.assert_not_called()
        ordinary.assert_not_called()
        legend.assert_called_once_with(self.slug, *legend_job, None)

    def test_cancelled_legend_work_stops_before_geometry_or_compute(self):
        with mock.patch.object(job.arrows, "page_geometry_status") as geometry, \
                mock.patch.object(job.legend_linetypes,
                                  "compute_page") as compute:
            with self.assertRaises(job.Cancelled):
                job._legend_linetype_one(
                    self.slug, 1, self.samples, self.sig,
                    should_cancel=lambda: True)
        geometry.assert_not_called()
        compute.assert_not_called()


class TestRemoteModelTimeoutBounds(JobTestBase):
    """Remote stalls get one recall retry; local deterministic stalls do not."""

    def test_serialized_timeout_names_remain_detectable(self):
        self.assertTrue(job.is_timeout_error(
            RuntimeError("ReadTimeout: upstream socket closed")))
        self.assertTrue(job.is_timeout_error(
            RuntimeError("DeadlineExceeded: provider request")))

    def test_symbol_timeout_recovers_on_one_bounded_retry(self):
        slug = job.create_project(b"%PDF-1.4 fake", "symbol-timeout.pdf")
        entry = {
            "sig": "sig", "v": 1, "pv": 1, "model": job.MODEL_NAME,
            "raw": {"groups": [], "symbols": []},
            "result": {"groups": [], "symbols": []},
        }
        with mock.patch(
                "steps.symbols.compute_page_symbols",
                side_effect=[TimeoutError("provider timed out"),
                             (entry, True)]) as compute, \
                mock.patch("steps.legend_sweep.sweep_needed",
                           return_value=[]), \
                mock.patch.object(job.time, "sleep") as sleep:
            result = job._symbol_one(
                slug, 1, [{"text": "FENCE", "box_2d": [1, 1, 2, 2]}],
                "sig")
        self.assertEqual((result[0], result[1], result[2]), (1, 0, None))
        self.assertEqual(compute.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_view_timeout_stops_after_two_total_attempts(self):
        slug = job.create_project(b"%PDF-1.4 fake", "view-timeout.pdf")
        with mock.patch(
                "steps.views.compute_view_types",
                side_effect=TimeoutError("provider timed out")) as compute, \
                mock.patch.object(job.time, "sleep") as sleep:
            page, summary, error = job._view_one(
                slug, 1, store.pdf_path(slug), [], "revision")
        self.assertEqual((page, summary), (1, None))
        self.assertIn("TimeoutError", error)
        self.assertEqual(compute.call_count, 2)
        sleep.assert_called_once_with(10)

    def test_text_judge_timeout_recovers_on_second_attempt(self):
        from steps.text import judge as text_judge

        response = mock.Mock(text="[]", usage_metadata=None)
        with mock.patch.object(
                text_judge, "gen_json",
                side_effect=[TimeoutError("provider timed out"), response]
                ) as generate, \
                mock.patch.object(text_judge.time, "sleep") as sleep:
            verdicts, _usage = text_judge.judge_strings(["SECURITY GATE"])
        self.assertEqual(verdicts, {"SECURITY GATE": False})
        self.assertEqual(generate.call_count, 2)
        sleep.assert_called_once_with(15)

    def test_text_judge_checkpoints_other_chunks_before_raising(self):
        from steps.text import judge as text_judge

        successful = {}

        def generate(_model, contents, **_kwargs):
            if "FAIL CHUNK" in contents[0]:
                raise TimeoutError("provider timed out")
            return mock.Mock(text="[]", usage_metadata=None)

        with mock.patch.object(text_judge, "gen_json", side_effect=generate), \
                mock.patch.object(text_judge.time, "sleep"):
            with self.assertRaisesRegex(RuntimeError, "1 of 2 judge chunks"):
                text_judge.judge_strings(
                    ["FAIL CHUNK", "OK CHUNK"], chunk=1,
                    on_chunk=successful.update)
        self.assertEqual(successful, {"OK CHUNK": False})


class TestTextStageOffline(JobTestBase):
    """真跑一遍 text 阶段：真 PDF、真矢量层、真融合，但一次模型调用都不许发生.

    做法：判词候选全被关键词地板吃掉（零 judge 调用），主模型和 Flash
    的 raw 都预置当期身份（零 VLM 调用），并把两个模块里的 gen_json
    直接换成会炸的假函数
    —— 只要还有任何付费路径被走到，测试就会红。
    """

    def setUp(self):
        super().setUp()
        import fitz
        from steps.text import (SECONDARY_UNION_ROLE, build_vlm_prompt,
                                make_vlm_record, vlm_identity)

        self.text = "6 FT CHAIN LINK FENCE"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 100), self.text, fontsize=11)
        pdf_bytes = doc.tobytes()
        doc.close()
        self.slug = job.create_project(pdf_bytes, "vector only.pdf")
        pdf = store.pdf_path(self.slug)
        self.revision = store.pdf_revision(pdf)

        from steps.text import vector_scan
        line = vector_scan(pdf, 0)["lines"][0]
        self.box = [int(line["box_2d"][0]), int(line["box_2d"][1]),
                    int(line["box_2d"][2]) + 1, int(line["box_2d"][3]) + 1]
        prompt = build_vlm_prompt(None)
        identity = vlm_identity(pdf, None, prompt)
        store.save_json(store.slug_dir(self.slug) / "vlm.json", {
            "1": make_vlm_record(
                identity=identity,
                items=[{"text": self.text, "box_2d": self.box,
                        "label": "note"}],
                elapsed=1.0,
                usage={"input_tokens": 10, "output_tokens": 5})})
        flash_identity = vlm_identity(pdf, job.FLASH_MODEL, prompt)
        store.save_json(store.slug_dir(self.slug) / "vlm_flash.json", {
            "1": make_vlm_record(
                identity=flash_identity, items=[], elapsed=1.0, usage={},
                role=SECONDARY_UNION_ROLE)})

        self.blocked_calls = []

        def boom(*_a, **_kw):                       # noqa: ANN002
            self.blocked_calls.append(_a)
            raise AssertionError("no model call is allowed in this test")

        for target in ("steps.text.judge.gen_json", "steps.text.vlm.gen_json"):
            p = mock.patch(target, boom)
            p.start()
            self.addCleanup(p.stop)

    def test_publishes_a_fused_page_without_paying(self):
        from steps.versions import FUSED_VERSION

        self.assertEqual(job._stage_text(self.slug, None), [])
        res = store.load_json(store.results_path(self.slug), None)
        self.assertEqual(res["fused_v"], FUSED_VERSION)
        self.assertEqual(res["pdf_revision"], self.revision)
        self.assertEqual(res["mode"], "fence")
        self.assertEqual(res["target"], job.TARGET_DEFAULT)
        self.assertIsNone(res["judge_error"])
        self.assertFalse(res["no_text_layer"])
        rec = res["pages"]["1"]
        self.assertEqual(len(rec["vlm_items"]), 1)
        self.assertEqual(rec["vlm_items"][0]["text"], self.text)
        self.assertTrue(rec["vlm_items"][0]["vec_backed"])
        # 关键词地板选出的矢量行被 VLM 框覆盖 → vec_covered，不再是独立 item
        self.assertEqual(len(rec["vec_covered"]), 1)
        self.assertEqual(rec["vec_added"], [])
        self.assertTrue(rec["has_text"])
        self.assertIsNone(rec["vlm_error"])
        # 矢量缓存落盘且不是 partial（可续跑的 checkpoint 已收尾）
        vec = store.load_json(store.slug_dir(self.slug) / "vec.json", None)
        self.assertEqual((vec["schema"], vec["page_count"]), (3, 1))
        self.assertNotIn("partial", vec)
        # 第二遍完全走缓存，同样零调用，结果一致
        self.assertEqual(job._stage_text(self.slug, None), [])
        again = store.load_json(store.results_path(self.slug), None)
        self.assertEqual(again["pages"], res["pages"])

    def test_malformed_vlm_retries_change_bytes_but_keep_base_identity(self):
        """A retry must escape provider caching without poisoning cache identity."""
        from steps.text import is_current_primary_record, vlm_identity

        base_prompt = "find every fence and return one JSON array"
        prompts = []
        item = {"text": "FENCE", "box_2d": [100, 100, 120, 180],
                "label": "callout"}

        def fake_scan(_pdf, _page, **kwargs):
            prompts.append(kwargs["prompt"])
            if len(prompts) < 3:
                raise RuntimeError("malformed JSON")
            return [item], 1.0, {"input_tokens": 1, "output_tokens": 1}

        raw = {}
        raw_path = store.slug_dir(self.slug) / "retry_nonce_test.json"
        with mock.patch("steps.text.scan_page", side_effect=fake_scan), \
                mock.patch.object(job.time, "sleep"), \
                mock.patch.object(job.time, "time_ns", side_effect=[101, 102]):
            page, count, _error = job._run_vlm(
                self.slug, store.pdf_path(self.slug), 1, raw, raw_path,
                prompt=base_prompt)

        self.assertEqual((page, count), (1, 1))
        self.assertEqual(prompts[0], base_prompt)
        self.assertIn("ATTEMPT 2", prompts[1])
        self.assertIn("ATTEMPT 3", prompts[2])
        self.assertEqual(len(set(prompts)), 3)
        expected = vlm_identity(
            store.pdf_path(self.slug), None, base_prompt)
        self.assertTrue(is_current_primary_record(raw["1"], expected))
        self.assertEqual(store.load_json(raw_path, {})["1"], raw["1"])

    def test_custom_target_needs_a_paid_scan_so_it_reports_the_failure(self):
        """换目标 → VLM 缓存身份失配 → 必须重扫（证明 prompt 参与缓存身份）."""
        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("time.sleep") as sleep:      # 别真等退避
            warnings = job._stage_text(self.slug, "find every manhole cover")
        self.assertEqual(len(warnings), 2, warnings)
        self.assertTrue(any("primary image scan incomplete" in w
                            for w in warnings), warnings)
        self.assertTrue(any("second image scan incomplete" in w
                            for w in warnings), warnings)
        # 两个独立模型各执行正常的 3 次策略，然后只对仍失败的来源各补
        # 一个带新 nonce 的最终修复请求。
        self.assertEqual(len(self.blocked_calls), 8)
        self.assertEqual([c.args[0] for c in sleep.call_args_list],
                         [20, 40, 20, 40])
        # 取消/失败都不能让已发布的结果消失：这页没有结果，但 raw 的 error 记录在
        vlm = store.load_json(store.slug_dir(self.slug) / "vlm.json", {})
        self.assertIn("AssertionError", vlm["1"]["error"])

    def test_primary_timeout_is_salvaged_by_independent_flash(self):
        """One timed-out model must not erase a result the other model found."""
        from steps.text.vlmcache import is_current_secondary_record
        from steps.text import build_vlm_prompt, vlm_identity

        (store.slug_dir(self.slug) / "vlm.json").unlink()
        (store.slug_dir(self.slug) / "vlm_flash.json").unlink()
        calls = []
        flash_item = {
            "text": "ORANGE PLASTIC SNOW FENCE",
            "box_2d": [700, 100, 730, 420], "label": "note"}

        def fake_scan(_pdf, _page, model=None, **_kwargs):
            calls.append(model)
            if model == job.FLASH_MODEL:
                return [flash_item], 1.0, {"input_tokens": 1}
            raise TimeoutError("model timed out")

        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("steps.text.scan_page", side_effect=fake_scan), \
                mock.patch("time.sleep") as sleep:
            warnings = job._stage_text(self.slug, None)
        self.assertEqual(calls, [job.MODEL_NAME, job.FLASH_MODEL,
                                 job.MODEL_NAME])
        # Flash gets its first chance before the timed-out Pro request is
        # repeated; the deferred retry itself is exactly one call.
        sleep.assert_not_called()
        self.assertTrue(any("primary image scan incomplete" in w
                            for w in warnings))
        result = store.load_json(store.results_path(self.slug), {})
        self.assertEqual(
            [item["text"] for item in result["pages"]["1"]["vlm_items"]],
            [flash_item["text"]])
        self.assertIn("TimeoutError", result["pages"]["1"]["vlm_error"])
        flash = store.load_json(store.slug_dir(self.slug) / "vlm_flash.json", {})
        expected = vlm_identity(
            store.pdf_path(self.slug), job.FLASH_MODEL,
            build_vlm_prompt(None))
        self.assertTrue(is_current_secondary_record(flash["1"], expected))

        # The failed primary remains explicit rework, but the successful Flash
        # raw is reused and its published detection cannot disappear.
        calls.clear()
        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("steps.text.scan_page", side_effect=fake_scan), \
                mock.patch("time.sleep"):
            job._stage_text(self.slug, None)
        self.assertEqual(calls, [job.MODEL_NAME, job.MODEL_NAME])
        again = store.load_json(store.results_path(self.slug), {})
        self.assertEqual(
            [item["text"] for item in again["pages"]["1"]["vlm_items"]],
            [flash_item["text"]])

    def test_malformed_source_is_repaired_once_without_warning(self):
        """All residual failures get one fresh request after both models run."""
        from steps.text import (build_vlm_prompt, is_current_primary_record,
                                vlm_identity)

        (store.slug_dir(self.slug) / "vlm.json").unlink()
        (store.slug_dir(self.slug) / "vlm_flash.json").unlink()
        primary_calls = []
        item = {"text": "FENCE", "box_2d": [100, 100, 120, 180],
                "label": "callout"}

        def fake_scan(_pdf, _page, model=None, **kwargs):
            if model == job.FLASH_MODEL:
                return [], 1.0, {"input_tokens": 1}
            primary_calls.append(kwargs["prompt"])
            if len(primary_calls) <= 3:
                raise RuntimeError("malformed provider JSON")
            return [item], 1.0, {"input_tokens": 1}

        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("steps.text.scan_page", side_effect=fake_scan), \
                mock.patch.object(job.time, "sleep"), \
                mock.patch.object(job.time, "time_ns",
                                  side_effect=[101, 102, 103]):
            warnings = job._stage_text(self.slug, None)

        self.assertEqual(warnings, [])
        self.assertEqual(len(primary_calls), 4)
        self.assertIn("ATTEMPT 2", primary_calls[1])
        self.assertIn("ATTEMPT 3", primary_calls[2])
        self.assertIn("ATTEMPT 4", primary_calls[3])
        identity = vlm_identity(store.pdf_path(self.slug), None,
                                build_vlm_prompt(None))
        raw = store.load_json(store.slug_dir(self.slug) / "vlm.json", {})
        self.assertTrue(is_current_primary_record(raw["1"], identity))

    def test_persisted_error_recovery_starts_with_fresh_nonce(self):
        """A reset=false rerun must not resend the failed base bytes."""
        from steps.text import (build_vlm_prompt, make_vlm_record,
                                vlm_identity)

        prompt = build_vlm_prompt(None)
        identity = vlm_identity(store.pdf_path(self.slug), None, prompt)
        path = store.slug_dir(self.slug) / "vlm.json"
        store.save_json(path, {"1": make_vlm_record(
            identity=identity, error="RuntimeError: previous bad response")})
        seen = []

        def fake_scan(_pdf, _page, **kwargs):
            seen.append(kwargs["prompt"])
            return [], 1.0, {"input_tokens": 1}

        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("steps.text.scan_page", side_effect=fake_scan), \
                mock.patch.object(job.time, "time_ns", return_value=777):
            self.assertEqual(job._stage_text(self.slug, None), [])
        self.assertEqual(len(seen), 1)
        self.assertIn("ATTEMPT 2", seen[0])
        self.assertIn("nonce=text-scan-retry-2-777", seen[0])

    def test_accuracy_mode_unions_both_models_on_text_layer_page(self):
        """Flash is not limited to empty Pro pages or raster-only pages."""
        (store.slug_dir(self.slug) / "vlm.json").unlink()
        (store.slug_dir(self.slug) / "vlm_flash.json").unlink()
        primary = {"text": "EXISTING RETAINING WALL & FENCE",
                   "box_2d": [300, 100, 330, 420], "label": "callout"}
        secondary = {"text": "STEEL FENCE POST @ 6 FT O.C.",
                     "box_2d": [700, 100, 730, 420], "label": "note"}

        def fake_scan(_pdf, _page, model=None, **_kwargs):
            item = secondary if model == job.FLASH_MODEL else primary
            return [item], 1.0, {"input_tokens": 1}

        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("steps.text.scan_page", side_effect=fake_scan):
            self.assertEqual(job._stage_text(self.slug, None), [])
        result = store.load_json(store.results_path(self.slug), {})
        texts = {item["text"] for item in
                 result["pages"]["1"]["vlm_items"]}
        self.assertEqual(texts, {primary["text"], secondary["text"]})


if __name__ == "__main__":
    unittest.main()
