"""离线回归：步骤2 图例样例符号、步骤3 视图分类、步骤4 shape 放置匹配.

全部离线：只用本地 PDF、参考项目已有的缓存 JSON、纯几何 —— 一次 Gemini
调用都不发。参考数据目录由环境变量 FENCE_REF_DIR 定位（默认生产 5051 的本机
副本），目录不在就跳过那部分用例。

跑法：
  cd C:\\Users\\Administrator\\fence_lite
  set PYTHONUTF8=1
  C:\\Users\\Administrator\\fence_detector\\venv\\Scripts\\python.exe -B -m unittest -v
"""
import copy
import json
import os
import sys
import types
import unittest
from pathlib import Path

from core.config import MODEL_NAME
from steps.placements import (LINE_NOTE, NO_PLAN_NOTE, has_current_placements,
                              match_placements)
from steps.store import items_of, pdf_revision, sig_of
from steps.symbols import (_require_group_symbol_lists, can_reuse_raw,
                           compute_page_symbols, filter_owned_group_symbols,
                           has_current_symbols, merge_sweep,
                           parse_group_symbol_payload, symbol_in_allowed_group,
                           symbols_dropped_view, sweep_version)
from steps.versions import (PLACEMENT_VERSION, SYMBOL_PROMPT_V, SYMBOL_VERSION,
                            VIEW_VERSION)
from steps.views import (groups_need_classification, has_current_view_types,
                         merge_view_types, plan_boxes, view_signature)

REF_DIR = Path(os.environ.get(
    "FENCE_REF_DIR", r"C:\Users\Administrator\fence_takeoff_web"))
HAS_REF = (REF_DIR / "fence_fused").is_dir() and (REF_DIR / "projects").is_dir()

# 实测基线（2026-08-12，参考项目缓存 + 本地矢量匹配器）：
# 这 5 页原始 1245 个放置，落在 plan 视图内 1230 个，被滤掉的 15 个正是
# 图例原件 / 图签噪声。改动 plan 过滤口径必然打破这些数字。
REF_PAGES = (
    # slug, page(1-based), shape_symbols, placed_in_plan, dropped_outside_plan
    ("koch_tennis_center", 3, 2, 533, 0),
    # P4 曾经是 (2, 370, 14)：同一个样例框 [384,115,398,136] 被配给了两个
    # text_index，于是整批放置被算了两遍。SYMBOL_VERSION=19 的同框去重把它
    # 收敛成一条，数字正好减半 —— 这是修正，不是回归。
    ("koch_tennis_center", 4, 1, 185, 7),
    ("koch_tennis_center", 7, 3, 322, 0),
    ("lenexa_fuel_station", 3, 2, 2, 0),
    ("rapid_city", 11, 2, 3, 1),
)
REF_TOTAL_RAW = 1053
REF_TOTAL_PLACED = 1045

# 参考数据里唯一带 line 类样例的两页（rapid_city P4 连 view_types 都没有 →
# 顺便验 fail-closed 不会给 line 编出放置来）。
REF_LINE_PAGES = (("lenexa_fuel_station", 19, 1), ("rapid_city", 4, 3))


def _ref_json(slug, name):
    return json.loads((REF_DIR / "fence_fused" / slug / name)
                      .read_text(encoding="utf-8"))


def _ref_pdf(slug):
    return REF_DIR / "projects" / slug / "input.pdf"


class _ReplayMatcher:
    """把参考缓存里已记录的匹配器输出回放出来（零成本、完全确定）."""

    def __init__(self, by_box):
        self.by_box = {tuple(box): list(placements)
                       for box, placements in by_box}
        self.calls = []

    def find_symbol_placements(self, pdf_path, page_index, box_norm):
        self.calls.append(tuple(box_norm))
        key = tuple(box_norm)
        if key not in self.by_box:
            return {"error": "no recorded matcher output for this box"}
        return {"placements": self.by_box[key]}


def _install_matcher(test, matcher):
    """把 core.symbolmatch 换成 matcher（match_placements 里是函数内 import）."""
    module = types.ModuleType("core.symbolmatch")
    module.find_symbol_placements = matcher.find_symbol_placements
    saved = sys.modules.get("core.symbolmatch")
    sys.modules["core.symbolmatch"] = module

    def restore():
        if saved is None:
            sys.modules.pop("core.symbolmatch", None)
        else:
            sys.modules["core.symbolmatch"] = saved

    test.addCleanup(restore)


# --------------------------------------------------------- 步骤2：两条硬闸

class TestSymbolHardGates(unittest.TestCase):
    GROUPS = [
        {"kind": "view", "box_2d": [0, 0, 500, 1000]},
        {"kind": "legend", "box_2d": [600, 600, 800, 900]},
    ]

    def test_owner_index_must_be_in_range(self):
        symbols = [
            {"text_index": 0, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            {"text_index": 3, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            {"text_index": -1, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            {"text_index": True, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            {"text_index": 1.0, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            {"text_index": None, "group_index": 1, "box_2d": [700, 700, 710, 720]},
            "not a dict",
        ]
        kept = filter_owned_group_symbols(symbols, self.GROUPS, 3)
        self.assertEqual([s["text_index"] for s in kept], [0])

    def test_geometry_rescues_a_wrong_group_index(self):
        """group_index 指到 view 组，但框中心确实在 legend 组里 → 放行。"""
        symbol = {"text_index": 0, "group_index": 0,
                  "box_2d": [700, 700, 710, 720]}
        self.assertTrue(symbol_in_allowed_group(symbol, self.GROUPS))
        self.assertEqual(
            len(filter_owned_group_symbols([symbol], self.GROUPS, 1)), 1)

    def test_view_area_marker_is_dropped(self):
        """group_index 是 view 组、几何也不在任何图例组里 → 剔除。"""
        symbol = {"text_index": 0, "group_index": 0,
                  "box_2d": [100, 100, 110, 120]}
        self.assertFalse(symbol_in_allowed_group(symbol, self.GROUPS))
        self.assertEqual(
            filter_owned_group_symbols([symbol], self.GROUPS, 1), [])

    def test_geometry_tolerance_is_two(self):
        just_inside = {"text_index": 0, "group_index": 0,
                       "box_2d": [597, 598, 599, 602]}    # center 598,600
        just_outside = {"text_index": 0, "group_index": 0,
                        "box_2d": [594, 598, 596, 602]}   # center 595,600
        self.assertTrue(symbol_in_allowed_group(just_inside, self.GROUPS))
        self.assertFalse(symbol_in_allowed_group(just_outside, self.GROUPS))

    def test_no_groups_means_no_symbols(self):
        symbol = {"text_index": 0, "group_index": 0,
                  "box_2d": [700, 700, 710, 720]}
        self.assertEqual(filter_owned_group_symbols([symbol], [], 1), [])


# ------------------------------------------------- 步骤2：付费响应严校验

def _payload(groups=None, symbols=None):
    return json.dumps({"groups": groups if groups is not None else [],
                       "symbols": symbols if symbols is not None else []})


GOOD_GROUP = {"box_2d": [600, 600, 800, 900], "kind": "legend"}
GOOD_SYMBOL = {"text_index": 0, "box_2d": [700, 700, 710, 720],
               "category": "shape", "value": "12", "group_index": 0}


class TestGroupSymbolResponseContract(unittest.TestCase):
    def test_explicit_empty_arrays_are_a_complete_answer(self):
        self.assertEqual(_require_group_symbol_lists(
            {"groups": [], "symbols": []}), ([], []))
        parsed = parse_group_symbol_payload(_payload(), 0)
        self.assertEqual(parsed, {"groups": [], "symbols": []})

    def test_missing_or_non_array_keys_raise(self):
        for payload in ({}, {"groups": []}, {"symbols": []},
                        {"groups": [], "symbols": {}},
                        {"groups": None, "symbols": []},
                        [], "text", None):
            with self.subTest(payload=payload):
                with self.assertRaises(RuntimeError):
                    _require_group_symbol_lists(payload)

    def test_truncated_or_invalid_json_raises(self):
        for text in ("", "not json",
                     '{"groups": [], "symbols": [',
                     '```json\n{"groups": [], "symbols": []}',
                     '```json\n{"groups": [], "symbols": []'):
            with self.subTest(text=text):
                with self.assertRaises(RuntimeError):
                    parse_group_symbol_payload(text, 1)

    def test_fenced_json_is_accepted(self):
        text = '```json\n{"groups": [], "symbols": []}\n```'
        self.assertEqual(parse_group_symbol_payload(text, 0),
                         {"groups": [], "symbols": []})

    def test_symbol_rows_are_validated(self):
        bad_symbols = [
            {**GOOD_SYMBOL, "text_index": 5},          # out of range
            {**GOOD_SYMBOL, "text_index": -1},
            {**GOOD_SYMBOL, "text_index": True},
            {**GOOD_SYMBOL, "text_index": 0.0},
            {**GOOD_SYMBOL, "group_index": 1},         # out of range
            {**GOOD_SYMBOL, "box_2d": [700, 700, 700, 720]},   # zero height
            {**GOOD_SYMBOL, "box_2d": [700, 700, 1200, 720]},  # out of frame
            {**GOOD_SYMBOL, "box_2d": [710, 700, 700, 720]},   # inverted
            {**GOOD_SYMBOL, "box_2d": "nope"},
            {**GOOD_SYMBOL, "category": "marker"},     # 禁止发明子类型
            {**GOOD_SYMBOL, "value": 12},
            "not a dict",
        ]
        for symbol in bad_symbols:
            with self.subTest(symbol=symbol):
                with self.assertRaises(RuntimeError):
                    parse_group_symbol_payload(
                        _payload([GOOD_GROUP], [symbol]), 1)
        for missing in ("text_index", "box_2d", "category", "value",
                        "group_index"):
            partial = {k: v for k, v in GOOD_SYMBOL.items() if k != missing}
            with self.subTest(missing=missing):
                with self.assertRaises(RuntimeError):
                    parse_group_symbol_payload(
                        _payload([GOOD_GROUP], [partial]), 1)

    def test_group_rows_are_validated(self):
        for group in ({"kind": "legend"}, {"box_2d": [1, 1, 2, 2]},
                      {"box_2d": [1, 1, 2, 2], "kind": "table"},
                      {"box_2d": [1, 1, 2, 2], "kind": 7},
                      {"box_2d": [0, 0, 0, 0], "kind": "legend"},
                      {"box_2d": [1, 1, 2, 2], "kind": "view",
                       "view_type": "top-down"},
                      "not a dict"):
            with self.subTest(group=group):
                with self.assertRaises(RuntimeError):
                    parse_group_symbol_payload(_payload([group], []), 1)

    def test_view_type_is_validated_but_never_published(self):
        parsed = parse_group_symbol_payload(
            _payload([{"box_2d": [1, 1, 500, 500], "kind": "view",
                       "view_type": "plan"}], []), 0)
        self.assertEqual(parsed["groups"],
                         [{"box_2d": [1, 1, 500, 500], "kind": "view"}])

    def test_accepted_row_shape(self):
        parsed = parse_group_symbol_payload(
            _payload([GOOD_GROUP], [GOOD_SYMBOL]), 1)
        self.assertEqual(parsed["symbols"], [{
            "box_2d": [700, 700, 710, 720], "category": "shape",
            "value": "12", "type": "shape 12", "text_index": 0,
            "group_index": 0}])


class TestSymbolCachePredicates(unittest.TestCase):
    def _entry(self, **over):
        # sweep_v：步骤②b 补扫也是发布链的一环，当期条目必须盖过这个戳。
        entry = {"sig": "s1", "v": SYMBOL_VERSION, "pv": SYMBOL_PROMPT_V,
                 "model": MODEL_NAME, "sweep_v": sweep_version() or 0,
                 "raw": {"groups": [GOOD_GROUP], "symbols": [GOOD_SYMBOL]},
                 "result": {"symbols": [GOOD_SYMBOL], "groups": [GOOD_GROUP]}}
        entry.update(over)
        return entry

    def test_entry_without_sweep_stamp_is_not_current(self):
        """没跑过补扫的页 = 不当期（导入的旧缓存就是这种）。"""
        entry = self._entry()
        entry.pop("sweep_v")
        self.assertTrue(can_reuse_raw(entry, "s1"))   # raw 仍免费复用
        self.assertFalse(has_current_symbols(entry, "s1"))

    def test_current_entry(self):
        self.assertTrue(can_reuse_raw(self._entry(), "s1"))
        self.assertTrue(has_current_symbols(self._entry(), "s1"))

    def test_legacy_pure_field_is_still_current(self):
        """导入的旧缓存多一个 pure 字段 —— 仍然算当期，不许重付费。"""
        legacy = self._entry(pure=True)
        legacy["result"]["pure"] = True
        legacy["result"]["prop_v"] = 3
        self.assertTrue(can_reuse_raw(legacy, "s1"))
        self.assertTrue(has_current_symbols(legacy, "s1"))

    def test_bumping_version_is_a_free_refilter(self):
        stale = self._entry(v=SYMBOL_VERSION + 1)
        self.assertTrue(can_reuse_raw(stale, "s1"))       # raw 仍可复用（免费）
        self.assertFalse(has_current_symbols(stale, "s1"))

    def test_sig_prompt_model_and_raw_shape_all_participate(self):
        for over in ({"sig": "other"}, {"pv": SYMBOL_PROMPT_V + 1},
                     {"model": "gemini-2.5-flash"},
                     {"raw": {"groups": []}}, {"raw": {"symbols": []}},
                     {"raw": None}, {"raw": "x"}):
            with self.subTest(over=over):
                self.assertFalse(can_reuse_raw(self._entry(**over), "s1"))
                self.assertFalse(has_current_symbols(self._entry(**over), "s1"))
        self.assertFalse(has_current_symbols(self._entry(result=None), "s1"))
        self.assertFalse(can_reuse_raw(None, "s1"))

    def test_dropped_view_explains_each_stripped_symbol(self):
        raw_symbols = [
            GOOD_SYMBOL,
            {**GOOD_SYMBOL, "text_index": -1},
            {**GOOD_SYMBOL, "box_2d": [10, 10, 20, 20], "group_index": 1},
        ]
        groups = [GOOD_GROUP, {"box_2d": [0, 0, 500, 500], "kind": "view"}]
        entry = {"raw": {"groups": groups, "symbols": raw_symbols},
                 "result": {"groups": groups, "symbols": [GOOD_SYMBOL]}}
        view = symbols_dropped_view(entry)
        self.assertEqual(len(view["raw_symbols"]), 3)
        self.assertEqual(len(view["dropped"]), 2)
        self.assertIn("owner", view["dropped"][0]["reason"])
        self.assertIn("组内校验", view["dropped"][1]["reason"])


# ------------------------------------------------------- 步骤3：视图分类

class TestViewTypeCache(unittest.TestCase):
    GROUPS = [
        {"kind": "legend", "box_2d": [0, 0, 100, 100]},
        {"kind": "view", "box_2d": [100, 100, 900, 900]},
        {"kind": "view", "box_2d": [900, 100, 990, 900]},
    ]
    REV = "1f-2f"

    def _entry(self, views, **over):
        entry = {"sig": view_signature(self.GROUPS, self.REV),
                 "v": VIEW_VERSION, "model": MODEL_NAME, "views": views}
        entry.update(over)
        return entry

    def _good_views(self):
        return [{"group_index": 1, "view_type": "plan", "reason": "top-down"},
                {"group_index": 2, "view_type": "section", "reason": "cut"}]

    def test_complete_entry_is_current(self):
        self.assertTrue(has_current_view_types(
            self._entry(self._good_views()), self.GROUPS, self.REV))

    def test_missing_one_group_is_not_current(self):
        views = self._good_views()[:1]
        self.assertFalse(has_current_view_types(
            self._entry(views), self.GROUPS, self.REV))

    def test_extra_or_duplicate_group_is_not_current(self):
        for views in (self._good_views() + [
                          {"group_index": 0, "view_type": "plan",
                           "reason": "legend is not a view"}],
                      self._good_views() + [
                          {"group_index": 1, "view_type": "plan",
                           "reason": "dupe"}]):
            with self.subTest(views=views):
                self.assertFalse(has_current_view_types(
                    self._entry(views), self.GROUPS, self.REV))

    def test_empty_reason_is_not_current(self):
        views = self._good_views()
        views[0]["reason"] = "   "
        self.assertFalse(has_current_view_types(
            self._entry(views), self.GROUPS, self.REV))
        views[0]["reason"] = None
        self.assertFalse(has_current_view_types(
            self._entry(views), self.GROUPS, self.REV))

    def test_unknown_view_type_is_not_current(self):
        views = self._good_views()
        views[0]["view_type"] = "top-down"
        self.assertFalse(has_current_view_types(
            self._entry(views), self.GROUPS, self.REV))

    def test_version_model_revision_and_geometry_all_participate(self):
        good = self._good_views()
        self.assertFalse(has_current_view_types(
            self._entry(good, v=VIEW_VERSION + 1), self.GROUPS, self.REV))
        self.assertFalse(has_current_view_types(
            self._entry(good, model="gemini-2.5-flash"), self.GROUPS, self.REV))
        self.assertFalse(has_current_view_types(
            self._entry(good), self.GROUPS, "deadbeef"))
        moved = copy.deepcopy(self.GROUPS)
        moved[1]["box_2d"] = [101, 100, 900, 900]
        self.assertFalse(has_current_view_types(
            self._entry(good), moved, self.REV))
        self.assertFalse(has_current_view_types(
            None, self.GROUPS, self.REV))
        self.assertFalse(has_current_view_types(
            self._entry(views="nope"), self.GROUPS, self.REV))

    def test_groups_need_classification(self):
        self.assertTrue(groups_need_classification(self.GROUPS))
        self.assertFalse(groups_need_classification([]))
        self.assertFalse(groups_need_classification(
            [{"kind": "legend", "box_2d": [0, 0, 10, 10]}]))
        self.assertFalse(groups_need_classification(
            [{"kind": "view", "box_2d": [10, 10, 5, 5]}]))
        # 步骤2 顺带给的 view_type 不得让本步跳过分类
        self.assertTrue(groups_need_classification(
            [{"kind": "view", "box_2d": [0, 0, 10, 10],
              "view_type": "plan"}]))

    def test_merge_and_plan_boxes(self):
        typed = merge_view_types(self.GROUPS, self._entry(self._good_views()))
        self.assertEqual(typed[1]["view_type"], "plan")
        self.assertEqual(typed[1]["view_type_reason"], "top-down")
        self.assertNotIn("view_type", typed[0])
        self.assertEqual(plan_boxes(typed), [[100, 100, 900, 900]])
        # merge 是深拷贝，不动入参
        self.assertNotIn("view_type", self.GROUPS[1])

    def test_plan_boxes_is_fail_closed(self):
        self.assertEqual(plan_boxes(merge_view_types(self.GROUPS, None)), [])
        self.assertEqual(plan_boxes([]), [])
        self.assertEqual(plan_boxes(None), [])
        self.assertEqual(plan_boxes(
            [{"kind": "legend", "box_2d": [0, 0, 10, 10],
              "view_type": "plan"},
             {"kind": "view", "box_2d": [0, 0, 10, 10],
              "view_type": "elevation"},
             {"kind": "view", "box_2d": [10, 10, 5, 5],
              "view_type": "plan"},
             {"kind": "view", "view_type": "plan"}]), [])


# ------------------------------------------------- 步骤4：shape 放置匹配

SAMPLE_BOX = [700, 700, 710, 720]
PLAN_GROUPS = [{"kind": "view", "box_2d": [0, 0, 500, 500],
                "view_type": "plan"},
               {"kind": "legend", "box_2d": [600, 600, 900, 900]}]


class TestPlacements(unittest.TestCase):
    def _shape(self):
        return {"box_2d": SAMPLE_BOX, "category": "shape", "value": "12",
                "text_index": 0, "group_index": 1}

    def test_plan_filter_keeps_only_centers_inside_plan(self):
        matcher = _ReplayMatcher([(SAMPLE_BOX, [
            [100, 100, 110, 120],        # center 105 → inside
            [499, 100, 505, 120],        # center 502 → exactly on the ±2 slack
            [504, 100, 510, 120],        # center 507 → outside
            [700, 700, 710, 720],        # the legend original area
        ])])
        _install_matcher(self, matcher)
        symbols = [self._shape()]
        summary = match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertEqual(symbols[0]["placements"],
                         [[100.0, 100.0, 110.0, 120.0],
                          [499.0, 100.0, 505.0, 120.0]])
        self.assertEqual(symbols[0]["dropped_outside_plan"], 2)
        self.assertNotIn("placement_note", symbols[0])
        self.assertEqual(summary, {"shape": 1, "line": 0, "placed": 2,
                                   "dropped_outside_plan": 2,
                                   "plan_groups": 1,
                                   "plc_v": PLACEMENT_VERSION})

    def test_no_plan_view_keeps_nothing(self):
        matcher = _ReplayMatcher([(SAMPLE_BOX, [[100, 100, 110, 120]] * 3)])
        _install_matcher(self, matcher)
        symbols = [self._shape()]
        unclassified = [{"kind": "view", "box_2d": [0, 0, 500, 500]}]
        summary = match_placements("x.pdf", 0, symbols, unclassified)
        self.assertEqual(symbols[0]["placements"], [])
        self.assertEqual(symbols[0]["placement_note"], NO_PLAN_NOTE)
        self.assertEqual(symbols[0]["dropped_outside_plan"], 3)
        self.assertEqual(summary["plan_groups"], 0)
        self.assertEqual(summary["placed"], 0)

    def test_line_is_never_matched(self):
        matcher = _ReplayMatcher([])
        _install_matcher(self, matcher)
        symbols = [{"box_2d": SAMPLE_BOX, "category": "line", "value": "SF",
                    "text_index": 0, "group_index": 1}]
        summary = match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertEqual(symbols[0]["placements"], [])
        self.assertEqual(symbols[0]["placement_note"], LINE_NOTE)
        self.assertEqual(matcher.calls, [])
        self.assertEqual(summary["line"], 1)
        self.assertEqual(summary["shape"], 0)

    def test_matcher_error_is_reported_per_symbol(self):
        matcher = _ReplayMatcher([])           # 没有记录 → 返回 error
        _install_matcher(self, matcher)
        symbols = [self._shape()]
        summary = match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertIn("placement_error", symbols[0])
        self.assertNotIn("placements", symbols[0])
        self.assertEqual(summary["placed"], 0)

    def test_matcher_exception_is_contained(self):
        module = types.ModuleType("core.symbolmatch")

        def boom(*_a, **_k):
            raise ValueError("vector extraction failed")

        module.find_symbol_placements = boom
        saved = sys.modules.get("core.symbolmatch")
        sys.modules["core.symbolmatch"] = module
        self.addCleanup(
            lambda: sys.modules.__setitem__("core.symbolmatch", saved)
            if saved is not None
            else sys.modules.pop("core.symbolmatch", None))
        symbols = [self._shape()]
        match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertEqual(symbols[0]["placement_error"],
                         "ValueError: vector extraction failed")

    def test_stale_fields_are_cleared(self):
        matcher = _ReplayMatcher([(SAMPLE_BOX, [[100, 100, 110, 120]])])
        _install_matcher(self, matcher)
        symbols = [{**self._shape(), "placements": [[9, 9, 9, 9]],
                    "placement_error": "old", "placement_note": "old",
                    "dropped_outside_plan": 42}]
        match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertEqual(symbols[0]["placements"],
                         [[100.0, 100.0, 110.0, 120.0]])
        self.assertNotIn("placement_error", symbols[0])
        self.assertNotIn("placement_note", symbols[0])
        self.assertNotIn("dropped_outside_plan", symbols[0])

    def test_rerun_is_idempotent(self):
        matcher = _ReplayMatcher([(SAMPLE_BOX, [
            [100, 100, 110, 120], [700, 700, 710, 720]])])
        _install_matcher(self, matcher)
        symbols = [self._shape()]
        first = match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        snapshot = copy.deepcopy(symbols)
        second = match_placements("x.pdf", 0, symbols, PLAN_GROUPS)
        self.assertEqual(first, second)
        self.assertEqual(snapshot, symbols)

    def test_has_current_placements(self):
        self.assertFalse(has_current_placements(None))
        self.assertFalse(has_current_placements({}))
        self.assertFalse(has_current_placements({"plc_v": PLACEMENT_VERSION + 1}))
        result = {"symbols": []}
        result.update(match_placements("x.pdf", 0, [], PLAN_GROUPS))
        self.assertTrue(has_current_placements(result))

    def test_debug_channel_is_always_collected(self):
        from steps.debug import DebugSink
        matcher = _ReplayMatcher([(SAMPLE_BOX, [[100, 100, 110, 120],
                                                [700, 700, 710, 720]])])
        _install_matcher(self, matcher)
        dbg = DebugSink()
        symbols = [self._shape(),
                   {"box_2d": [800, 800, 810, 820], "category": "line",
                    "value": "", "text_index": 1, "group_index": 1}]
        match_placements("x.pdf", 0, symbols, PLAN_GROUPS, dbg=dbg)
        rows = dbg.data["placements"]
        self.assertEqual([r["symbol_index"] for r in rows], [0, 1])
        self.assertEqual(rows[0], {"symbol_index": 0, "category": "shape",
                                   "value": "12", "sample_box": SAMPLE_BOX,
                                   "placements": 1,
                                   "dropped_outside_plan": 1,
                                   "status": "accepted"})
        self.assertEqual(rows[1]["status"], LINE_NOTE)


# --------------------------------------- 参考数据回归（离线，只读缓存 + PDF）

@unittest.skipUnless(HAS_REF, f"reference data not found at {REF_DIR}")
class TestReferenceCacheCompatibility(unittest.TestCase):
    """导入的生产缓存：付费 raw 必须仍能复用（一次都不用重付步骤②），
    但发布链多了步骤②b 补扫，所以旧条目**故意**算不当期 —— 作业会免费复用
    raw、只为缺样例的图例块补扫一次。"""

    def test_prompt_bump_forces_a_repaid_step2_and_filter_stays_sane(self):
        """提示词版本 bump 之后，旧 raw 不再可复用 —— 这就是改提示词的代价。

        改 GROUP_SYMBOL_PROMPT 必须同步 bump SYMBOL_PROMPT_V，否则旧 raw 会被
        当成新提示词的答案继续服务。这条用例把「代价」写死：pv 一旦不同，
        can_reuse_raw 必须为假（作业会重新付费跑步骤②）；同时验证发布过滤
        本身的语义没坏 —— 它只允许剥掉不合规的、把同框的收敛成一条。
        """
        for slug, page, shape_count, _placed, _dropped in REF_PAGES:
            with self.subTest(slug=slug, page=page):
                revision = pdf_revision(_ref_pdf(slug))
                rec = _ref_json(slug, "results.json")["pages"][str(page)]
                items = items_of(rec)
                sig = sig_of(items, revision)
                entry = _ref_json(slug, "symbols.json")[str(page)]
                self.assertEqual(entry["sig"], sig)     # 文字锚本身没变
                self.assertNotEqual(entry["pv"], SYMBOL_PROMPT_V)
                self.assertFalse(can_reuse_raw(entry, sig))
                self.assertFalse(has_current_symbols(entry, sig))
                # pv 是唯一原因：对齐它就又能免费重过滤
                aligned = {**entry, "pv": SYMBOL_PROMPT_V}
                self.assertTrue(can_reuse_raw(aligned, sig))
                # 发布过滤（含同框去重）按当前口径重跑一遍
                result = _republished(slug, page)
                published = result["symbols"]
                self.assertEqual(
                    sum(1 for s in published if s["category"] == "shape"),
                    shape_count)
                old_boxes = [tuple(s["box_2d"])
                             for s in entry["result"]["symbols"]]
                new_boxes = [tuple(s["box_2d"]) for s in published]
                self.assertEqual(len(new_boxes), len(set(new_boxes)),
                                 "同一个样例框不该发布两次")
                self.assertTrue(set(new_boxes) <= set(old_boxes),
                                "重过滤不该凭空造出新框")
                # 顺带的 view_type 不进发布结果（旧 raw 里可能有）
                self.assertFalse(any("view_type" in g
                                     for g in result["groups"]))

    def test_imported_view_types_are_current(self):
        for slug, page, _shape, _placed, _dropped in REF_PAGES:
            with self.subTest(slug=slug, page=page):
                revision = pdf_revision(_ref_pdf(slug))
                groups = _ref_json(slug, "symbols.json")[str(page)]["result"]["groups"]
                entry = _ref_json(slug, "view_types.json").get(str(page))
                self.assertTrue(has_current_view_types(entry, groups, revision))
                self.assertEqual(len(plan_boxes(merge_view_types(groups, entry))), 1)


def _republished(slug, page):
    """把参考缓存的这一页按当前发布语义重跑一遍，返回 result 的深拷贝。

    旧缓存是 SYMBOL_VERSION=18 的产物，还带着「同一个样例框配给两个 text_index」
    这种重复；生产里它们一定会先被重过滤（复用 raw、零模型调用）。测试拿盘上
    那份直接算，等于在验一个生产永远不会出现的输入。
    """
    rec = _ref_json(slug, "results.json")["pages"][str(page)]
    items = items_of(rec)
    entry = _ref_json(slug, "symbols.json")[str(page)]
    # 直接跑发布过滤，不经过 compute_page_symbols：那一层还有 pv 闸，
    # 而参考缓存是老 SYMBOL_PROMPT_V 的产物 —— 一旦提示词版本 bump，它就会
    # 判「raw 不可复用」并真去发一次付费调用。这里要验的是**过滤语义**，
    # 与提示词版本无关，所以拿 raw 直接过一遍同一个过滤函数。
    raw = copy.deepcopy(entry["raw"])
    groups = raw["groups"]
    for group in groups:
        if isinstance(group, dict):
            group.pop("view_type", None)
            group.pop("view_type_reason", None)
    kept = filter_owned_group_symbols(raw["symbols"], groups, len(items),
                                      items, "page")
    return {"symbols": kept, "groups": groups}


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF_DIR}")
class TestReferencePlanFilter(unittest.TestCase):
    """用生产缓存里已记录的匹配器输出回放，只验 plan 过滤这一层的口径."""

    def _page(self, slug, page):
        # 必须先过一次重新发布：生产里旧缓存条目一定是不当期的（v/sweep_v），
        # 步骤4 拿到的永远是重过滤之后的符号集（含同框去重），不是盘上那份。
        result = _republished(slug, page)
        view_entry = _ref_json(slug, "view_types.json").get(str(page))
        typed = merge_view_types(result["groups"], view_entry)
        # 重新发布是从 raw 重建的，raw 上没有 placements（那是步骤4 写在 result
        # 上的）。回放要的是「匹配器当年对这个框输出了什么」，所以按框把旧记录
        # 接回到去重后的符号集上。
        by_box = {}
        for old in _ref_json(slug, "symbols.json")[str(page)]["result"]["symbols"]:
            by_box.setdefault(tuple(old["box_2d"]), old.get("placements") or [])
        recorded = [(s["box_2d"], by_box.get(tuple(s["box_2d"]), []))
                    for s in result["symbols"]]
        return result["symbols"], typed, recorded

    def test_recorded_placements_filter_to_the_expected_counts(self):
        total_raw = total_placed = 0
        for slug, page, shape_count, placed, dropped in REF_PAGES:
            with self.subTest(slug=slug, page=page):
                symbols, typed, recorded = self._page(slug, page)
                matcher = _ReplayMatcher(recorded)
                _install_matcher(self, matcher)
                summary = match_placements(
                    _ref_pdf(slug), page - 1, symbols, typed)
                self.assertEqual(summary["plan_groups"], 1)
                self.assertEqual(summary["shape"], shape_count)
                self.assertEqual(summary["placed"], placed)
                self.assertEqual(summary["dropped_outside_plan"], dropped)
                total_placed += summary["placed"]
                total_raw += sum(len(p) for _b, p in recorded)
        self.assertEqual(total_raw, REF_TOTAL_RAW)
        self.assertEqual(total_placed, REF_TOTAL_PLACED)

    def test_line_symbols_never_get_placements(self):
        for slug, page, line_count in REF_LINE_PAGES:
            with self.subTest(slug=slug, page=page):
                symbols, typed, recorded = self._page(slug, page)
                matcher = _ReplayMatcher(recorded)
                _install_matcher(self, matcher)
                summary = match_placements(
                    _ref_pdf(slug), page - 1, symbols, typed)
                self.assertEqual(summary["line"], line_count)
                lines = [s for s in symbols if s["category"] == "line"]
                self.assertEqual(len(lines), line_count)
                for s in lines:
                    self.assertEqual(s["placements"], [])
                    self.assertEqual(s["placement_note"], LINE_NOTE)


def _have_symbolmatch():
    try:
        import core.symbolmatch                                # noqa: F401
    except Exception:                                          # noqa: BLE001
        return False
    return True


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF_DIR}")
@unittest.skipUnless(_have_symbolmatch(), "core.symbolmatch not available yet")
class TestReferenceRealMatcher(unittest.TestCase):
    """真跑本地矢量匹配器（免费、无模型调用），端到端复现同样的数字."""

    def test_end_to_end_counts(self):
        total_placed = total_dropped = 0
        for slug, page, shape_count, placed, dropped in REF_PAGES:
            with self.subTest(slug=slug, page=page):
                result = _republished(slug, page)
                view_entry = _ref_json(slug, "view_types.json").get(str(page))
                typed = merge_view_types(result["groups"], view_entry)
                summary = match_placements(
                    _ref_pdf(slug), page - 1, result["symbols"], typed)
                self.assertEqual(summary["shape"], shape_count)
                self.assertEqual(summary["placed"], placed)
                self.assertEqual(summary["dropped_outside_plan"], dropped)
                result.update(summary)
                self.assertTrue(has_current_placements(result))
                total_placed += summary["placed"]
                total_dropped += summary["dropped_outside_plan"]
        self.assertEqual(total_placed, REF_TOTAL_PLACED)
        self.assertEqual(total_placed + total_dropped, REF_TOTAL_RAW)


if __name__ == "__main__":
    unittest.main()
