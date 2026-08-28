"""tools/eval_symbols.py 的离线回归 —— 合成页 + 真数据冒烟，零 Gemini 调用.

合成部分不需要 PDF：eval_page 是纯函数，直接喂 items / symbols / groups /
strip_context 上下文，把每个指标的边界条件钉死（最小面积归属、量化重复、
恰好 0.5 的重叠、被框切断的线段、±2 组容差……）。
真数据部分全部 skipUnless：缺目录就跳过，绝不因为环境不同而假红。

跑法：
  cd C:\\Users\\Administrator\\fence_lite
  set PYTHONUTF8=1
  C:\\Users\\Administrator\\fence_detector\\venv\\Scripts\\python.exe -B -m unittest -v
"""
import copy
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")


def _load_tool():
    """tools/ 没有 __init__.py，按文件路径加载。"""
    path = BASE_DIR / "tools" / "eval_symbols.py"
    spec = importlib.util.spec_from_file_location("_eval_symbols_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EV = _load_tool()

# 真数据（只读）。缺了就跳过对应用例。
DIAG_DIR = Path(os.environ.get("FENCE_DIAG_DIR", str(
    Path(os.environ.get("TEMP", r"C:\Users\ADMINI~1\AppData\Local\Temp"))
    / "claude" / "C--Users-Administrator"
    / "5a059b30-437c-43d7-899c-bda2d5c22a41" / "scratchpad" / "diag")))
PONDEROSA = "2026_01_30_ponderosa_pines_all_buildings_ifc_set_lands_irr"
BINDER = "drawings_volume_4_binder"
DETECTOR_DIR = Path(r"C:\Users\Administrator\fence_detector")
HAS_TAYLOR = ((DETECTOR_DIR / "fence_fused" / "taylor_3_12"
               / "symbols.json").exists()
              and (DETECTOR_DIR / "projects" / "taylor_3_12"
                   / "input.pdf").exists())


def item(text, box):
    return {"text": text, "box_2d": list(box), "label": "", "tbl": False}


def sym(box, text_index, category="shape", value="", group_index=0):
    return {"box_2d": list(box), "text_index": text_index,
            "category": category, "value": value, "group_index": group_index}


def group(box, kind):
    return {"box_2d": list(box), "kind": kind}


# ------------------------------------------------------------ 归属 / 配对

class LegendTextTests(unittest.TestCase):
    def test_smallest_area_assignment_wins(self):
        """图例组套在 view 组里时，item 必须归给面积更小的图例组。"""
        groups = [group([0, 0, 1000, 1000], "view"),
                  group([100, 100, 200, 300], "legend")]
        items = [item("legend row", [140, 150, 150, 250]),
                 item("plan callout", [500, 500, 510, 560])]
        m = EV.eval_page(items, [], groups)
        self.assertEqual(m["legend_texts"], 1)
        self.assertEqual(m["unpaired_idx"], [0])

    def test_view_inside_legend_is_not_legend_text(self):
        """反过来：小的 view 组套在大 legend 组里，里面的文字不算图例文字。"""
        groups = [group([0, 0, 500, 500], "legend"),
                  group([100, 100, 200, 200], "view")]
        m = EV.eval_page([item("t", [140, 140, 150, 160])], [], groups)
        self.assertEqual(m["legend_texts"], 0)

    def test_schedule_and_note_cluster_count_too(self):
        groups = [group([0, 0, 100, 100], "schedule"),
                  group([200, 0, 300, 100], "note_cluster"),
                  group([400, 0, 500, 100], "title_block")]
        # 文字得是正经描述行：纯 marker 编码（"a" / "4CL"）按定义不算图例描述行，
        # 会被 legend_texts 排除掉 —— 那是另一条用例在验的事。
        items = [item("CHAIN LINK FENCE", [10, 10, 20, 20]),
                 item("SILT FENCE", [210, 10, 220, 20]),
                 item("SHEET INDEX", [410, 10, 420, 20])]
        m = EV.eval_page(items, [], groups)
        self.assertEqual(m["legend_texts"], 2)

    def test_bare_marker_code_rows_are_not_legend_descriptions(self):
        """"4CL" 这种漏进文字层的裸编码不是描述行，不该算进待配对口径。"""
        groups = [group([0, 0, 500, 500], "legend")]
        items = [item("4CL", [10, 100, 20, 130]),
                 item('4\'-0" TALL VINYL CHAIN LINK FENCING', [10, 200, 20, 400])]
        m = EV.eval_page(items, [], groups)
        self.assertEqual(m["legend_texts"], 1)
        self.assertEqual(m["unpaired_idx"], [1])

    def test_paired_unpaired_and_preview(self):
        groups = [group([0, 0, 500, 500], "legend")]
        items = [item("ROW ONE", [10, 100, 20, 200]),
                 item("ROW TWO WITH A VERY " + "LONG " * 20 + "TAIL",
                      [30, 100, 40, 200])]
        symbols = [sym([10, 50, 20, 70], 0)]
        m = EV.eval_page(items, symbols, groups)
        self.assertEqual((m["paired"], m["unpaired"]), (1, 1))
        self.assertEqual(m["unpaired_idx"], [1])
        self.assertEqual(len(m["unpaired_items"][0]["text"]),
                         EV.TEXT_PREVIEW)
        self.assertTrue(m["unpaired_items"][0]["text"].startswith("ROW TWO"))

    def test_owner_outside_legend_does_not_create_pairing(self):
        """symbol 认了一个不在图例组里的文字 → 图例文字仍然 0，不虚增 paired。"""
        groups = [group([0, 0, 500, 500], "view")]
        items = [item("plan text", [10, 10, 20, 100])]
        m = EV.eval_page(items, [sym([10, 200, 20, 210], 0)], groups)
        self.assertEqual((m["legend_texts"], m["paired"]), (0, 0))
        self.assertEqual(m["sym_total"], 1)


# ------------------------------------------------------------------ 重复

class DupBoxTests(unittest.TestCase):
    BOX = [144, 794, 152, 810]

    def test_same_box_two_owners_is_one_dup(self):
        groups = [group([0, 700, 300, 900], "legend")]
        items = [item("legend row", [146, 821, 150, 889]),
                 item("4CL", [146, 798, 150, 803])]
        symbols = [sym(self.BOX, 0), sym(self.BOX, 1)]
        m = EV.eval_page(items, symbols, groups)
        self.assertEqual(m["dup_boxes"], 1)

    def test_same_box_same_owner_is_not_dup(self):
        symbols = [sym(self.BOX, 0), sym(self.BOX, 0)]
        m = EV.eval_page([item("a", [0, 0, 1, 1])], symbols,
                         [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["dup_boxes"], 0)

    def test_quantization_merges_subunit_jitter(self):
        """0.4/1000 的抖动是同一个框；2/1000 的差别不是。"""
        near = [144.4, 794.2, 152.3, 810.1]
        far = [146, 796, 154, 812]
        m_near = EV.eval_page([item("a", [0, 0, 1, 1])],
                              [sym(self.BOX, 0), sym(near, 1)],
                              [group([0, 0, 1000, 1000], "legend")])
        m_far = EV.eval_page([item("a", [0, 0, 1, 1])],
                             [sym(self.BOX, 0), sym(far, 1)],
                             [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual((m_near["dup_boxes"], m_far["dup_boxes"]), (1, 0))

    def test_three_owners_one_box_is_still_one_group(self):
        symbols = [sym(self.BOX, i) for i in range(3)]
        m = EV.eval_page([item("a", [0, 0, 1, 1])], symbols,
                         [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["dup_boxes"], 1)


# ------------------------------------------------------- 文字框吃掉 marker

class InTextTests(unittest.TestCase):
    def test_taylor_p163_overlap_is_0_72(self):
        """线上真实数字：symbol∩text（inter/min 面积）= 0.72 > 0.5。"""
        items = [item("SEE DETAIL 8/E5.1", [166.4, 111.3, 184.5, 279.1])]
        symbols = [sym([166, 109, 178, 118], 0)]
        m = EV.eval_page(items, symbols, [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["sym_in_text"], 1)
        self.assertAlmostEqual(m["symbols"][0]["owner_overlap"], 0.72,
                               places=2)

    def test_disjoint_symbol_is_not_in_text(self):
        items = [item("caption", [100, 200, 110, 400])]
        symbols = [sym([100, 100, 110, 150], 0)]
        m = EV.eval_page(items, symbols, [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["sym_in_text"], 0)
        self.assertEqual(m["symbols"][0]["owner_overlap"], 0.0)

    def test_exactly_half_is_not_flagged(self):
        """阈值是严格大于 0.5：正好一半的重叠不算被吃掉。"""
        items = [item("caption", [100, 100, 110, 300])]
        symbols = [sym([100, 50, 110, 150], 0)]     # 一半在文字框里
        m = EV.eval_page(items, symbols, [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["symbols"][0]["owner_overlap"], 0.5)
        self.assertEqual(m["sym_in_text"], 0)

    def test_bad_owner_index_is_ignored_not_crashed(self):
        items = [item("caption", [100, 100, 110, 300])]
        symbols = [sym([100, 100, 110, 300], 7),        # 越界
                   sym([100, 100, 110, 300], None)]     # 缺主人
        m = EV.eval_page(items, symbols, [group([0, 0, 1000, 1000], "legend")])
        self.assertEqual(m["sym_in_text"], 0)
        self.assertNotIn("owner_overlap", m["symbols"][0])


# -------------------------------------------------------- 平面图 marker

class OutsideLegendTests(unittest.TestCase):
    GROUPS = [group([0, 0, 200, 200], "legend"),
              group([300, 300, 900, 900], "view")]

    def _one(self, box):
        return EV.eval_page([item("t", [10, 10, 20, 20])],
                            [sym(box, 0)], self.GROUPS)

    def test_center_in_view_is_outside_legend(self):
        self.assertEqual(self._one([500, 500, 510, 510])["sym_outside_legend"],
                         1)

    def test_center_in_legend_is_inside(self):
        self.assertEqual(self._one([50, 50, 60, 60])["sym_outside_legend"], 0)

    def test_tolerance_is_two_units(self):
        """与 steps.symbols 的几何兜底同口径：±2 之内还算在组里。"""
        just_in = self._one([199, 199, 202, 202])       # 中心 200.5
        just_out = self._one([203, 203, 206, 206])      # 中心 204.5
        self.assertEqual(just_in["sym_outside_legend"], 0)
        self.assertEqual(just_out["sym_outside_legend"], 1)

    def test_no_groups_at_all_means_outside(self):
        m = EV.eval_page([item("t", [10, 10, 20, 20])],
                         [sym([10, 10, 20, 20], 0)], [])
        self.assertEqual(m["sym_outside_legend"], 1)
        self.assertEqual(m["legend_texts"], 0)


# ------------------------------------------------------------------ box_fit

class BoxFitTests(unittest.TestCase):
    GROUPS = [group([0, 0, 1000, 1000], "legend")]
    ITEMS = [item("row", [100, 200, 110, 400])]

    def _fit(self, symbols, ctx):
        return EV.eval_page(self.ITEMS, symbols, self.GROUPS, ctx)

    def test_shape_takes_best_overlapping_mbox(self):
        ctx = {"mboxes": [[100, 100, 110, 120],       # 不沾
                          [100, 100, 112, 112]],      # 部分重叠
               "segs": []}
        m = self._fit([sym([100, 100, 110, 110], 0, "shape")], ctx)
        self.assertEqual(m["symbols"][0]["shape_fit"], 1.0)
        self.assertEqual(m["box_fit"]["shape_median"], 1.0)

    def test_shape_partial_overlap_ratio(self):
        # symbol 10x10=100，mbox 10x20=200，交 10x5=50 → 50/min(100,200)=0.5
        ctx = {"mboxes": [[100, 105, 110, 125]], "segs": []}
        m = self._fit([sym([100, 100, 110, 110], 0, "shape")], ctx)
        self.assertEqual(m["symbols"][0]["shape_fit"], 0.5)

    def test_shape_without_any_mbox_is_none(self):
        """页面没有矢量 marker（扫描页）→ 测不了，记 None，不是 0 分。"""
        m = self._fit([sym([100, 100, 110, 110], 0, "shape")],
                      {"mboxes": [], "segs": []})
        self.assertIsNone(m["symbols"][0]["shape_fit"])
        self.assertIsNone(m["box_fit"]["shape_median"])

    def test_shape_landing_on_nothing_is_zero(self):
        m = self._fit([sym([100, 100, 110, 110], 0, "shape")],
                      {"mboxes": [[800, 800, 810, 810]], "segs": []})
        self.assertEqual(m["symbols"][0]["shape_fit"], 0.0)

    def test_line_span_ratio_one_when_box_wraps_sample(self):
        segs = [{"ax": 100, "ay": 105, "bx": 140, "by": 105}]
        m = self._fit([sym([100, 100, 110, 140], 0, "line")],
                      {"mboxes": [], "segs": segs})
        fit = m["symbols"][0]["line_fit"]
        self.assertEqual((fit["segs"], fit["span_ratio"]), (1, 1.0))

    def test_line_span_ratio_collapses_when_box_cuts_the_sample(self):
        """框只切住样例中间一小截：被切断的线段不算，跨度比塌到 0。"""
        segs = [{"ax": 50, "ay": 105, "bx": 300, "by": 105}]
        m = self._fit([sym([100, 100, 110, 140], 0, "line")],
                      {"mboxes": [], "segs": segs})
        fit = m["symbols"][0]["line_fit"]
        self.assertEqual((fit["segs"], fit["span_ratio"]), (0, 0.0))
        # 同一行的线在框外还长得多 → row_cover 明显 < 1
        self.assertLess(fit["row_cover"], 0.2)

    def test_line_dashes_inside_box_keep_ratio_high(self):
        segs = [{"ax": 100 + i * 10, "ay": 105, "bx": 106 + i * 10, "by": 105}
                for i in range(4)]
        m = self._fit([sym([100, 100, 110, 140], 0, "line")],
                      {"mboxes": [], "segs": segs})
        self.assertGreater(m["symbols"][0]["line_fit"]["span_ratio"], 0.8)

    def test_line_without_any_seg_is_none(self):
        m = self._fit([sym([100, 100, 110, 140], 0, "line")],
                      {"mboxes": [], "segs": []})
        self.assertIsNone(m["symbols"][0]["line_fit"])
        self.assertIsNone(m["box_fit"]["line_span_median"])

    def test_no_ctx_skips_box_fit_but_keeps_other_metrics(self):
        m = self._fit([sym([100, 100, 110, 140], 0, "line")], None)
        self.assertFalse(m["box_fit"]["available"])
        self.assertEqual(m["box_fit"]["line_span_values"], [])
        self.assertNotIn("line_fit", m["symbols"][0])
        self.assertEqual(m["sym_line"], 1)

    def test_medians_use_all_values_not_page_averages(self):
        ctx = {"mboxes": [[100, 100, 110, 110]], "segs": []}
        symbols = [sym([100, 100, 110, 110], 0, "shape"),     # 1.0
                   sym([800, 800, 810, 810], 0, "shape"),     # 0.0
                   sym([100, 100, 104, 110], 0, "shape")]     # 1.0
        m = EV.eval_page(self.ITEMS, symbols, self.GROUPS, ctx)
        self.assertEqual(m["box_fit"]["shape_median"], 1.0)
        self.assertEqual(sorted(m["box_fit"]["shape_values"]),
                         [0.0, 1.0, 1.0])


# ------------------------------------------------------------ 布局 / 项目

def write_project(root, layout, slug, results, symbols):
    """在临时根里造一个某种布局的项目（不造 PDF —— box_fit 自动跳过）。"""
    if layout == "flat":
        cache = Path(root) / slug
    else:
        cache = Path(root) / layout / slug
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "results.json").write_text(json.dumps(results),
                                        encoding="utf-8")
    (cache / "symbols.json").write_text(json.dumps(symbols), encoding="utf-8")
    return cache


def tiny_project(pages=(1, 2)):
    """两页：P1 图例文字配到符号，P2 图例文字一个都没配到。"""
    groups = [group([0, 0, 300, 300], "legend"),
              group([400, 400, 900, 900], "view")]
    results = {"page_count": 9, "pages": {}}
    symbols = {}
    for page in pages:
        results["pages"][str(page)] = {
            "vlm_items": [{"text": f"ROW {page}", "box_2d": [50, 100, 60, 200]}],
            "vec_added": [{"text": "SECOND", "box_2d": [80, 100, 90, 200]}]}
        published = [] if page == 2 else [sym([50, 40, 60, 60], 0, "shape",
                                              "A", 0)]
        symbols[str(page)] = {"sig": "x", "v": 1, "pv": 1, "model": "m",
                              "raw": {"groups": groups,
                                      "symbols": published + []},
                              "result": {"groups": groups,
                                         "symbols": published}}
    return results, symbols


class ProjectLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.results, self.symbols = tiny_project()

    def test_three_layouts_resolve(self):
        for layout, slug in (("data", "a_data"), ("fence_fused", "b_fused"),
                             ("flat", "c_flat")):
            write_project(self.root, layout, slug, self.results, self.symbols)
        for layout, slug in (("data", "a_data"), ("fence_fused", "b_fused"),
                             ("flat", "c_flat")):
            cache, pdf, found = EV.resolve_project(self.root, slug)
            self.assertEqual(found, layout, slug)
            self.assertTrue(cache.exists())
            self.assertIsNone(pdf)
        self.assertEqual(EV.available_slugs(self.root),
                         ["a_data", "b_fused", "c_flat"])

    def test_fused_dir_given_directly_finds_sibling_pdf(self):
        """--from 直接指到 ...\\fence_fused 时，PDF 在上一级 projects/ 下。"""
        write_project(self.root, "fence_fused", "slug1", self.results,
                      self.symbols)
        pdf = self.root / "projects" / "slug1" / "input.pdf"
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b"%PDF-1.4 not a real pdf")
        cache, found_pdf, layout = EV.resolve_project(
            self.root / "fence_fused", "slug1")
        self.assertEqual(layout, "flat")
        self.assertEqual(found_pdf, pdf)
        self.assertEqual(cache, self.root / "fence_fused" / "slug1")

    def test_unknown_slug_raises(self):
        with self.assertRaises(FileNotFoundError):
            EV.eval_project("nope", root=self.root)

    def test_eval_project_metrics_and_page_filter(self):
        write_project(self.root, "data", "s", self.results, self.symbols)
        report = EV.eval_project("s", root=self.root)
        self.assertEqual(report["layout"], "data")
        self.assertEqual(sorted(report["pages"]), ["1", "2"])
        self.assertEqual(report["pages"]["1"]["paired"], 1)
        self.assertEqual(report["pages"]["2"]["unpaired"], 2)
        self.assertEqual(report["totals"]["pages_with_legend_text"], 2)
        self.assertEqual(report["totals"]["pages_all_paired"], 0)
        self.assertEqual(report["totals"]["unpaired"], 3)
        only2 = EV.eval_project("s", root=self.root, pages_filter=[2])
        self.assertEqual(list(only2["pages"]), ["2"])

    def test_union_index_order_matches_items_of(self):
        """text_index 锚在 vlm_items + vec_added 的拼接顺序上，必须一致。"""
        write_project(self.root, "data", "s", self.results, self.symbols)
        report = EV.eval_project("s", root=self.root)
        self.assertEqual(report["pages"]["2"]["unpaired_items"][0]["text"],
                         "ROW 2")
        self.assertEqual(report["pages"]["2"]["unpaired_items"][1]["text"],
                         "SECOND")

    def test_missing_symbol_entry_is_reported_not_hidden(self):
        results, symbols = tiny_project()
        symbols.pop("2")
        write_project(self.root, "data", "s", results, symbols)
        report = EV.eval_project("s", root=self.root)
        self.assertFalse(report["pages"]["2"]["has_symbol_entry"])
        self.assertEqual(report["totals"]["pages_without_symbol_entry"], 1)

    def test_gate_dropped_counts_raw_minus_published(self):
        results, symbols = tiny_project()
        symbols["1"]["raw"]["symbols"] = (symbols["1"]["raw"]["symbols"]
                                          + [sym([500, 500, 510, 510], 0)])
        write_project(self.root, "data", "s", results, symbols)
        report = EV.eval_project("s", root=self.root)
        self.assertEqual(report["pages"]["1"]["raw_total"], 2)
        self.assertEqual(report["pages"]["1"]["gate_dropped"], 1)


# ------------------------------------------------------------------ diff

class DiffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        results, symbols = tiny_project()
        write_project(self.root, "data", "s", results, symbols)
        self.before = EV.eval_root(["s"], root=self.root)

    def _after(self, mutate):
        after = copy.deepcopy(self.before)
        mutate(after["projects"]["s"]["pages"])
        return after

    def test_unpaired_fixed_is_an_improvement(self):
        def fix(pages):
            pages["2"]["unpaired"] = 0
            pages["2"]["unpaired_idx"] = []
            pages["2"]["paired"] = 2

        diff = EV.diff_reports(self.before, self._after(fix))
        self.assertEqual(diff["regressions"], [])
        self.assertEqual(diff["projects"]["s"]["pages"]["2"]["fixed_idx"],
                         [0, 1])
        self.assertTrue(any("unpaired" in tag
                            for tag in diff["improvements"]))

    def test_new_dup_and_in_text_are_regressions(self):
        def worsen(pages):
            pages["1"]["dup_boxes"] = 2
            pages["1"]["sym_in_text"] = 1

        diff = EV.diff_reports(self.before, self._after(worsen))
        self.assertEqual(len(diff["regressions"]), 2)
        text = EV.format_diff(diff)
        self.assertIn("[恶化]", text)
        self.assertIn("dup_boxes", text)

    def test_lost_pairing_is_a_regression(self):
        # 先把 before 摆成「P1 两条都配到了」，再让 after 丢掉其中一条。
        page = self.before["projects"]["s"]["pages"]["1"]
        page.update({"unpaired": 0, "unpaired_idx": [], "paired": 2})

        def worsen(pages):
            pages["1"]["unpaired"] = 1
            pages["1"]["unpaired_idx"] = [1]
            pages["1"]["paired"] = 1

        diff = EV.diff_reports(self.before, self._after(worsen))
        self.assertEqual(diff["projects"]["s"]["pages"]["1"]["broken_idx"],
                         [1])
        self.assertEqual(len(diff["regressions"]), 2)   # unpaired↑ + paired↓

    def test_box_fit_direction_and_epsilon(self):
        def better(pages):
            pages["1"]["box_fit"]["shape_median"] = 0.9

        def noise(pages):
            pages["1"]["box_fit"]["shape_median"] = 0.401

        def worse(pages):
            pages["1"]["box_fit"]["shape_median"] = 0.1

        for mutate in (better, noise, worse):
            self.before["projects"]["s"]["pages"]["1"]["box_fit"][
                "shape_median"] = 0.4
            diff = EV.diff_reports(self.before, self._after(mutate))
            changes = diff["projects"]["s"]["pages"].get("1", {}).get(
                "changes", [])
            verdicts = [c["verdict"] for c in changes
                        if c["metric"] == "shape_median"]
            if mutate is better:
                self.assertEqual(verdicts, ["better"])
            elif mutate is noise:
                self.assertEqual(verdicts, [])       # 抖动不算变化
            else:
                self.assertEqual(verdicts, ["worse"])

    def test_sym_total_change_alone_is_informational(self):
        def mutate(pages):
            pages["1"]["sym_total"] = 5

        diff = EV.diff_reports(self.before, self._after(mutate))
        self.assertEqual(diff["regressions"], [])
        self.assertEqual(diff["improvements"], [])

    def test_disappearing_page_is_flagged(self):
        after = copy.deepcopy(self.before)
        after["projects"]["s"]["pages"].pop("2")
        diff = EV.diff_reports(self.before, after)
        self.assertEqual(diff["pages_removed"], ["s#P2"])
        self.assertIn("[恶化]", EV.format_diff(diff))

    def test_missing_project_side_does_not_crash(self):
        after = copy.deepcopy(self.before)
        after["projects"] = {}
        diff = EV.diff_reports(self.before, after)
        self.assertEqual(diff["projects"]["s"]["missing_in"], "after")
        EV.format_diff(diff)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        results, symbols = tiny_project()
        write_project(self.root, "data", "s", results, symbols)

    def _run(self, argv):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = EV.main(argv)
        return code, buf.getvalue()

    def test_list_when_no_slug(self):
        code, out = self._run(["--from", str(self.root)])
        self.assertEqual(code, 2)
        self.assertIn("s", out)

    def test_json_out_is_machine_readable(self):
        out_path = self.root / "before.json"
        code, out = self._run(["s", "--from", str(self.root),
                               "--json", str(out_path), "--no-fit"])
        self.assertEqual(code, 0)
        self.assertIn("TOTAL", out)
        report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(report["report_version"], EV.REPORT_VERSION)
        self.assertEqual(report["totals"]["unpaired"], 3)

    def test_diff_cli_returns_1_on_regression(self):
        before_path = self.root / "a.json"
        after_path = self.root / "b.json"
        self._run(["s", "--from", str(self.root), "--json", str(before_path),
                   "--no-fit"])
        report = json.loads(before_path.read_text(encoding="utf-8"))
        good = copy.deepcopy(report)
        good["projects"]["s"]["pages"]["2"]["unpaired"] = 0
        good["projects"]["s"]["pages"]["2"]["unpaired_idx"] = []
        bad = copy.deepcopy(report)
        bad["projects"]["s"]["pages"]["1"]["sym_outside_legend"] = 3
        after_path.write_text(json.dumps(good), encoding="utf-8")
        code, out = self._run(["--diff", str(before_path), str(after_path)])
        self.assertEqual(code, 0)
        self.assertIn("没有恶化项", out)
        after_path.write_text(json.dumps(bad), encoding="utf-8")
        code, out = self._run(["--diff", str(before_path), str(after_path)])
        self.assertEqual(code, 1)
        self.assertIn("sym_outside_legend", out)


# ------------------------------------------------------------ 真数据冒烟

@unittest.skipUnless((DIAG_DIR / BINDER / "symbols.json").exists(),
                     f"缺 {DIAG_DIR / BINDER}")
class BinderRealDataTests(unittest.TestCase):
    def test_p5_has_duplicate_boxes(self):
        """4CL / 6CL 裸编码各自与真正的图例行同框 → dup_boxes >= 2。"""
        report = EV.eval_project(BINDER, root=DIAG_DIR, pages_filter=[5],
                                 use_pdf=False)
        page = report["pages"]["5"]
        self.assertGreaterEqual(page["dup_boxes"], 2)
        self.assertGreaterEqual(page["sym_in_text"], 2)


@unittest.skipUnless((DIAG_DIR / PONDEROSA / "symbols.json").exists(),
                     f"缺 {DIAG_DIR / PONDEROSA}")
class PonderosaRealDataTests(unittest.TestCase):
    def test_p2_two_legend_texts(self):
        report = EV.eval_project(PONDEROSA, root=DIAG_DIR, pages_filter=[2],
                                 use_pdf=False)
        self.assertEqual(report["pages"]["2"]["legend_texts"], 2)

    def test_p2_without_any_symbol_is_two_unpaired(self):
        """线上新缓存 P2 的 raw symbols = 0 —— 用真数据造出那个状态再量化。"""
        with tempfile.TemporaryDirectory() as tmp:
            src = DIAG_DIR / PONDEROSA
            dst = Path(tmp) / PONDEROSA
            dst.mkdir(parents=True)
            (dst / "results.json").write_bytes(
                (src / "results.json").read_bytes())
            symbols = json.loads((src / "symbols.json").read_text(
                encoding="utf-8"))
            symbols["2"]["raw"]["symbols"] = []
            symbols["2"]["result"]["symbols"] = []
            (dst / "symbols.json").write_text(json.dumps(symbols),
                                              encoding="utf-8")
            report = EV.eval_project(PONDEROSA, root=Path(tmp),
                                     pages_filter=[2], use_pdf=False)
        page = report["pages"]["2"]
        self.assertEqual((page["legend_texts"], page["paired"],
                          page["unpaired"]), (2, 0, 2))
        self.assertEqual([i["idx"] for i in page["unpaired_items"]], [0, 1])


@unittest.skipUnless(HAS_TAYLOR, f"缺 {DETECTOR_DIR}\\...\\taylor_3_12")
class TaylorRealDataTests(unittest.TestCase):
    """带真 PDF 的冒烟：box_fit 走真实矢量几何（本地、免费）。"""

    @classmethod
    def setUpClass(cls):
        cls.report = EV.eval_project("taylor_3_12", root=DETECTOR_DIR,
                                     pages_filter=[3, 163])

    def test_layout_and_pdf_found(self):
        self.assertEqual(self.report["layout"], "fence_fused")
        self.assertTrue(self.report["pdf"].endswith("input.pdf"))

    def test_p3_line_span_ratio_is_far_below_one(self):
        page = self.report["pages"]["3"]
        self.assertEqual(page["sym_line"], 1)
        self.assertTrue(page["box_fit"]["available"])
        self.assertLess(page["box_fit"]["line_span_median"], 0.5)

    def test_p163_symbol_is_swallowed_by_its_text_box(self):
        page = self.report["pages"]["163"]
        self.assertGreaterEqual(page["sym_in_text"], 1)
        self.assertGreater(page["symbols"][0]["owner_overlap"], 0.7)


if __name__ == "__main__":
    unittest.main()
