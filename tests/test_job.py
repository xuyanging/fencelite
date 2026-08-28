"""job.py 的离线回归 —— 严禁任何模型调用.

这里只测编排层自己的逻辑：slug 唯一化 / 目录布局、任务状态机与落盘、
重启后的 interrupted 语义、费用合并、进度映射、协作式取消。四个阶段函数
一律用假实现替换，所以整套测试不碰 PDF、不碰 Gemini、不写真实 data/。
"""
import tempfile
import shutil
import unittest
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
        job._RUNNING.update({"slug": None, "base": None, "base_wall": 0.0})
        self.addCleanup(job.JOBS.clear)
        self.addCleanup(job._CANCEL.clear)

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

    def test_page_count_of_unreadable_pdf_is_zero(self):
        slug = job.create_project(b"not a pdf", "broken.pdf")
        self.assertEqual(job.page_count_of(slug), 0)


# ------------------------------------------------------------- job status --
class TestJobState(JobTestBase):

    def test_set_persists_and_reloads(self):
        job._set("alpha", stage="text", progress=0.25, warnings=[])
        on_disk = store.load_json(self.jobs / "alpha.json", None)
        self.assertEqual(on_disk, {"slug": "alpha", "stage": "text",
                                   "progress": 0.25, "warnings": []})
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

    def test_request_cancel_without_running_job(self):
        out = job.request_cancel("alpha")
        self.assertFalse(out["was_running"])
        self.assertTrue(job.JOBS["alpha"]["cancel_requested"])

    def test_resume_interrupted_marks_but_never_reruns(self):
        store.save_json(self.jobs / "finished.json",
                        {"slug": "finished", "done": True, "ok": True,
                         "stage": "done"})
        store.save_json(self.jobs / "inflight.json",
                        {"slug": "inflight", "done": False, "stage": "symbols",
                         "progress": 0.6})
        store.save_json(self.jobs / "junk.json", {"slug": "not/a/slug"})
        with mock.patch.object(job, "_run",
                               side_effect=AssertionError("must not re-run")):
            self.assertIsNone(job.resume_interrupted())
        done = job.JOBS["finished"]
        self.assertEqual((done["done"], done["ok"], done["stage"]),
                         (True, True, "done"))
        hit = job.JOBS["inflight"]
        self.assertEqual((hit["done"], hit["ok"], hit["cancelled"],
                          hit["stage"]), (True, False, True, "interrupted"))
        self.assertEqual(hit["progress"], 0.6)   # 保留原进度，只改状态
        self.assertEqual(
            store.load_json(self.jobs / "inflight.json", None)["stage"],
            "interrupted")
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
    """把四个阶段函数换成假实现，记录调用顺序与收到的 should_cancel。"""

    def __init__(self, case, text=None, on_text=None):
        self.calls = []
        self.text_warnings = text or []
        self.on_text = on_text
        for name in ("_stage_text", "_stage_symbols", "_stage_views",
                     "_stage_placements"):
            p = mock.patch.object(job, name, self._make(name))
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

    def test_fence_mode_runs_four_stages_in_order(self):
        fake = _FakeStages(self)
        job._run_pipeline("alpha", None, lambda: False)
        self.assertEqual(fake.calls, ["_stage_text", "_stage_symbols",
                                      "_stage_views", "_stage_placements"])
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


class TestRun(JobTestBase):

    def test_successful_run_publishes_cost_and_wall(self):
        slug = job.create_project(b"x", "ok.pdf")
        self.write_results(slug)
        fake = _FakeStages(self)
        job._set(slug, started=job.time.time())
        job._run(slug, None)
        st = job.JOBS[slug]
        self.assertEqual((st["done"], st["ok"], st["stage"], st["error"]),
                         (True, True, "done", None))
        self.assertEqual(len(fake.calls), 4)
        res = store.load_json(store.results_path(slug), None)
        self.assertEqual(res["llm_summary"], job._ZERO_LLM)
        self.assertIsInstance(res["wall_seconds"], float)
        self.assertIsNone(job._RUNNING["slug"])
        self.assertNotIn(slug, job._CANCEL)

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

    def test_start_job_initialises_state_and_carries_baseline(self):
        slug = job.create_project(b"x", "resume.pdf")
        self.write_results(slug, {
            "llm_summary": {"calls": 7, "cost_usd": 1.25,
                            "peak_concurrency": 4},
            "wall_seconds": 42.0})
        with mock.patch.object(job, "_run") as run:
            st = job.start_job(slug)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], slug)
        self.assertEqual(st["mode"], "fence")
        self.assertEqual(st["stage"], "queued")
        self.assertEqual(st["progress"], 0.0)
        self.assertEqual(st["pages_total"], 0)
        self.assertEqual(st["warnings"], [])
        self.assertEqual(st["llm"]["calls"], 7)
        self.assertEqual(st["wall_seconds"], 42.0)
        self.assertEqual(st["target"], job.TARGET_DEFAULT)

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
            from steps.versions import PLACEMENT_VERSION
            return {"shape": 1, "line": 0, "placed": 1,
                    "dropped_outside_plan": 3, "plan_groups": 1,
                    "plc_v": PLACEMENT_VERSION}

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

    def test_missing_view_classification_is_fail_closed(self):
        warnings = job._stage_placements(self.slug)
        self.assertEqual(self.seen, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("视图分类不当期", warnings[0])
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
        self.assertIn("视图分类不当期", warnings[0])

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


class TestTextStageOffline(JobTestBase):
    """真跑一遍 text 阶段：真 PDF、真矢量层、真融合，但一次模型调用都不许发生.

    做法：判词候选全被关键词地板吃掉（零 judge 调用），vlm.json 预置一条当期
    身份的 raw（零 VLM 调用），并把两个模块里的 gen_json 直接换成会炸的假函数
    —— 只要还有任何付费路径被走到，测试就会红。
    """

    def setUp(self):
        super().setUp()
        import fitz
        from steps.text import build_vlm_prompt, make_vlm_record, vlm_identity

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
        identity = vlm_identity(pdf, None, build_vlm_prompt(None))
        store.save_json(store.slug_dir(self.slug) / "vlm.json", {
            "1": make_vlm_record(
                identity=identity,
                items=[{"text": self.text, "box_2d": self.box,
                        "label": "note"}],
                elapsed=1.0,
                usage={"input_tokens": 10, "output_tokens": 5})})

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

    def test_custom_target_needs_a_paid_scan_so_it_reports_the_failure(self):
        """换目标 → VLM 缓存身份失配 → 必须重扫（证明 prompt 参与缓存身份）."""
        with mock.patch.object(job, "_judge_project",
                               return_value=(set(), None)), \
                mock.patch("time.sleep") as sleep:      # 别真等退避
            warnings = job._stage_text(self.slug, "find every manhole cover")
        self.assertTrue(any("文字 VLM 失败" in w for w in warnings), warnings)
        self.assertTrue(any("不当期" in w for w in warnings), warnings)
        # RETRIES=2 → 共 3 次尝试，退避 20*(n+1) 秒
        self.assertEqual(len(self.blocked_calls), 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [20, 40])
        # 取消/失败都不能让已发布的结果消失：这页没有结果，但 raw 的 error 记录在
        vlm = store.load_json(store.slug_dir(self.slug) / "vlm.json", {})
        self.assertIn("AssertionError", vlm["1"]["error"])


if __name__ == "__main__":
    unittest.main()
