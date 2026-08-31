"""离线回归：步骤2 的发布闸（几何为准）+ 同框去重 + 步骤②b 补扫合并.

全部离线 —— 一次 Gemini 调用都不发：几何用手搓的组框，去重用线上真实数据里
那两个重复案例的形状（drawings_volume_4_binder P5 / koch_tennis_center P4），
补扫合并用假的 sweep 结果 dict。

跑法：
  cd C:\\Users\\Administrator\\fence_lite
  set PYTHONUTF8=1
  C:\\Users\\Administrator\\fence_detector\\venv\\Scripts\\python.exe -B -m unittest discover -s tests
"""
import copy
import sys
import types
import unittest

from core.config import MODEL_NAME
from steps.symbols import (SOURCE_PAGE, SOURCE_ROW_CODE, SOURCE_SWEEP,
                           classify_symbols,
                           dedupe_symbols, filter_owned_group_symbols,
                           has_current_symbols, inherit_row_code_symbols,
                           merge_sweep,
                           symbol_group_verdict, symbol_in_allowed_group,
                           symbols_dropped_view, sweep_version)
from steps.versions import SYMBOL_PROMPT_V, SYMBOL_VERSION

# 一页典型的分区：上半是平面图，右下角一块图例，另有一块 keyed 注记。
GROUPS = [
    {"kind": "view", "box_2d": [0, 0, 500, 1000]},
    {"kind": "legend", "box_2d": [600, 600, 800, 900]},
    {"kind": "note_cluster", "box_2d": [850, 600, 950, 900]},
]
IN_LEGEND = [700, 700, 710, 720]      # 中心 705,710 → legend 内
IN_VIEW = [100, 100, 110, 120]        # 中心 105,110 → view 内
NOWHERE = [520, 100, 530, 120]        # 中心 525,110 → 谁都不在


def _sym(box, text_index=0, group_index=1, **over):
    row = {"text_index": text_index, "group_index": group_index,
           "box_2d": list(box), "category": "shape", "value": "12",
           "type": "shape 12"}
    row.update(over)
    return row


def _item(box, text):
    return {"text": text, "box_2d": list(box), "label": "", "tbl": False}


# ------------------------------------------------------ 1. 组内闸：几何为准

class TestGeometryFirstGroupGate(unittest.TestCase):
    def test_claimed_legend_but_geometrically_in_view_is_dropped(self):
        """模型把 group_index 填成 legend，框却在平面图里 → 必须剥掉。"""
        symbol = _sym(IN_VIEW, group_index=1)          # 1 = legend
        self.assertFalse(symbol_in_allowed_group(symbol, GROUPS))
        self.assertEqual(
            filter_owned_group_symbols([symbol], GROUPS, 1, [_item(IN_VIEW, "x")]),
            [])
        ok, reason = symbol_group_verdict(symbol, GROUPS)
        self.assertFalse(ok)
        # 理由要说得出「它实际落在哪个 kind 的组里」
        self.assertIn("view", reason)
        self.assertIn("legend", reason)          # 模型声称的那个
        self.assertIn("in-group check", reason)

    def test_geometry_in_legend_wins_over_a_view_group_index(self):
        """几何为准的另一面：group_index 指向 view，框在图例里 → 放行。"""
        symbol = _sym(IN_LEGEND, group_index=0)        # 0 = view
        self.assertTrue(symbol_in_allowed_group(symbol, GROUPS))
        kept = filter_owned_group_symbols([symbol], GROUPS, 1)
        self.assertEqual(len(kept), 1)
        # group_index 降级成审计字段，不再是通行证
        self.assertEqual(kept[0]["claimed_group_index"], 0)
        self.assertNotIn("group_index", kept[0])
        self.assertEqual(kept[0]["source"], SOURCE_PAGE)

    def test_owner_index_must_be_in_range(self):
        symbols = [_sym(IN_LEGEND, 0), _sym(IN_LEGEND, 3), _sym(IN_LEGEND, -1),
                   _sym(IN_LEGEND, True), _sym(IN_LEGEND, 1.0),
                   _sym(IN_LEGEND, None), "not a dict"]
        kept = filter_owned_group_symbols(symbols, GROUPS, 3)
        self.assertEqual([s["text_index"] for s in kept], [0])

    def test_tolerance_is_two(self):
        just_inside = _sym([597, 598, 599, 602])       # 中心 598,600
        just_outside = _sym([594, 598, 596, 602])      # 中心 595,600
        self.assertTrue(symbol_in_allowed_group(just_inside, GROUPS))
        self.assertFalse(symbol_in_allowed_group(just_outside, GROUPS))


# --------------------------------------------- 2. 无几何证据时的 owner 兜底

class TestOwnerFallback(unittest.TestCase):
    def test_owner_in_note_cluster_publishes(self):
        """框中心不在任何组框里 → 用 owner 文字的位置判断（note_cluster → 放行）。"""
        symbol = _sym(NOWHERE, 0, group_index=0)
        items = [_item([880, 620, 890, 700], "6' CHAIN LINK FENCE - SEE SPECS")]
        self.assertEqual(symbol_group_verdict(symbol, GROUPS, items)[0], True)
        self.assertEqual(
            len(filter_owned_group_symbols([symbol], GROUPS, 1, items)), 1)

    def test_owner_in_view_is_dropped(self):
        symbol = _sym(NOWHERE, 0, group_index=1)
        items = [_item([200, 200, 210, 260], "SF-1")]     # callout in the plan
        ok, reason = symbol_group_verdict(symbol, GROUPS, items)
        self.assertFalse(ok)
        self.assertIn("owner", reason)
        self.assertIn("view", reason)
        self.assertEqual(
            filter_owned_group_symbols([symbol], GROUPS, 1, items), [])

    def test_no_owner_information_is_dropped(self):
        symbol = _sym(NOWHERE, 0)
        self.assertEqual(filter_owned_group_symbols([symbol], GROUPS, 1), [])

    def test_page_without_legend_groups_strips_everything(self):
        """fail-closed：本页一个图例类组都没有 → 全剥，理由写明。"""
        groups = [{"kind": "view", "box_2d": [0, 0, 1000, 1000]}]
        symbol = _sym(IN_LEGEND, 0, group_index=0)
        items = [_item([700, 600, 710, 690], "6' CHAIN LINK FENCE")]
        ok, reason = symbol_group_verdict(symbol, groups, items)
        self.assertFalse(ok)
        self.assertIn("no legend-type group on this sheet", reason)
        self.assertEqual(filter_owned_group_symbols([symbol], groups, 1, items),
                         [])

    def test_no_groups_at_all_means_no_symbols(self):
        self.assertEqual(
            filter_owned_group_symbols([_sym(IN_LEGEND)], [], 1), [])


# ------------------------------------------------------------ 3. 同框去重

# drawings_volume_4_binder P5 的真实形状：图例行 idx=7 是描述行，idx=10 是漏进
# 文字层的裸编码 "4CL"（它的框整个被样例框包住），两者拿到了同一个 symbol 框。
DUP_BOX = [144, 794, 152, 810]
DUP_ITEMS = {
    7: _item([146.2, 821.2, 150.1, 889.1],
             "4'-0\" TALL VINYL CHAIN LINK FENCING - SEE SPECS"),
    10: _item([146.1, 798.2, 149.8, 803.2], "4CL"),
}


def _drawings_p5_case():
    items = [_item([0, 0, 1, 1], "") for _ in range(11)]
    for index, item in DUP_ITEMS.items():
        items[index] = item
    groups = [{"kind": "view", "box_2d": [250, 480, 970, 890]},
              {"kind": "legend", "box_2d": [35, 780, 215, 890]}]
    symbols = [_sym(DUP_BOX, 7, group_index=1, value="4CL", type="shape 4CL"),
               _sym(DUP_BOX, 10, group_index=1, value="4CL", type="shape 4CL")]
    return symbols, groups, items


class TestDeterministicDedupe(unittest.TestCase):
    def test_description_row_wins_over_the_bare_code(self):
        symbols, groups, items = _drawings_p5_case()
        kept, dropped = classify_symbols(symbols, groups, len(items), items)
        self.assertEqual([s["text_index"] for s in kept], [7])
        self.assertEqual(len(dropped), 1)
        self.assertEqual(dropped[0]["text_index"], 10)
        self.assertIn("duplicate", dropped[0]["reason"])
        self.assertIn("text_index=7", dropped[0]["reason"])

    def test_bare_code_first_still_loses(self):
        """输入顺序反过来，赢家不变（规则是重叠度/长度，不是先来后到）。"""
        symbols, groups, items = _drawings_p5_case()
        kept, _ = classify_symbols(list(reversed(symbols)), groups,
                                   len(items), items)
        self.assertEqual([s["text_index"] for s in kept], [7])

    def test_koch_p4_shape(self):
        """koch_tennis_center P4 同一个病：owner 1 就是框里的 'SF-#'。"""
        box = [384, 115, 398, 136]
        items = [_item([388, 64, 394, 107], "SILT FENCE. REFERENCE DETAIL SHEET."),
                 _item([388, 119, 394, 132], "SF-#")]
        groups = [{"kind": "legend", "box_2d": [320, 10, 460, 150]},
                  {"kind": "view", "box_2d": [100, 150, 1000, 1000]}]
        symbols = [_sym(box, 0, group_index=0, value="SF-#"),
                   _sym(box, 1, group_index=0, value="SF-#")]
        kept, dropped = classify_symbols(symbols, groups, 2, items)
        self.assertEqual([s["text_index"] for s in kept], [0])
        self.assertEqual([s["text_index"] for s in dropped], [1])

    def test_distinct_boxes_are_all_kept(self):
        items = [_item([700, 730, 710, 800], "6' CHAIN LINK FENCE"),
                 _item([720, 730, 730, 800], "4' CHAIN LINK FENCE")]
        symbols = [_sym([700, 700, 710, 720], 0), _sym([720, 700, 730, 720], 1)]
        kept, dropped = classify_symbols(symbols, GROUPS, 2, items)
        self.assertEqual(len(kept), 2)
        self.assertEqual(dropped, [])

    def test_quantization_treats_sub_pixel_jitter_as_the_same_box(self):
        items = [_item([700, 730, 710, 800], "LONG DESCRIPTION ROW"),
                 _item([700, 730, 710, 800], "ANOTHER DESCRIPTION ROW HERE")]
        symbols = [_sym([700.0, 700.0, 710.0, 720.0], 0),
                   _sym([699.7, 700.2, 710.4, 719.6], 1)]
        kept, _ = classify_symbols(symbols, GROUPS, 2, items)
        self.assertEqual(len(kept), 1)

    def test_is_deterministic_and_order_independent(self):
        symbols, groups, items = _drawings_p5_case()
        symbols += [_sym([153, 794, 161, 810], 8, group_index=1, value="6CL"),
                    _sym([153, 794, 161, 810], 9, group_index=1, value="6CL")]
        items[8] = _item([154.9, 821.1, 158.8, 889.1],
                         "6'-0\" TALL VINYL CHAIN LINK FENCING - SEE SPECS")
        items[9] = _item([154.8, 798.2, 158.5, 803.2], "6CL")
        first = classify_symbols(symbols, groups, len(items), items)[0]
        second = classify_symbols(copy.deepcopy(symbols), groups,
                                  len(items), items)[0]
        self.assertEqual(first, second)              # 同输入两次 → 逐字节一致
        self.assertEqual([s["text_index"] for s in first], [7, 8])
        for order in ([3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]):
            shuffled = [symbols[i] for i in order]
            kept = classify_symbols(shuffled, groups, len(items), items)[0]
            self.assertEqual(sorted(s["text_index"] for s in kept), [7, 8])

    def test_dedupe_without_items_keeps_the_lowest_text_index(self):
        symbols, groups, _items = _drawings_p5_case()
        kept, dropped = dedupe_symbols(symbols)
        self.assertEqual([s["text_index"] for s in kept], [7])
        self.assertEqual(len(dropped), 1)

    def test_dropped_view_explains_a_dedupe_drop(self):
        symbols, groups, items = _drawings_p5_case()
        kept, _ = classify_symbols(symbols, groups, len(items), items)
        entry = {"raw": {"groups": groups, "symbols": symbols},
                 "result": {"groups": groups, "symbols": kept}}
        view = symbols_dropped_view(entry, items)
        self.assertEqual(len(view["dropped"]), 1)
        self.assertEqual(view["dropped"][0]["text_index"], 10)
        self.assertIn("duplicate", view["dropped"][0]["reason"])

    def test_dropped_view_explains_a_geometry_drop(self):
        raw_symbols = [_sym(IN_LEGEND, 0), _sym(IN_VIEW, 0, group_index=1)]
        entry = {"raw": {"groups": GROUPS, "symbols": raw_symbols},
                 "result": {"groups": GROUPS,
                            "symbols": [_sym(IN_LEGEND, 0)]}}
        view = symbols_dropped_view(entry)
        self.assertEqual(len(view["dropped"]), 1)
        reason = view["dropped"][0]["reason"]
        self.assertIn("view", reason)
        self.assertIn("in-group check", reason)


# --------------------------------------- 4.0 章节行号继承为 4.6 symbol

class TestInheritedRowCodeSymbols(unittest.TestCase):
    GROUPS = [{"kind": "schedule", "box_2d": [660, 61, 955, 570]}]
    PARENT_BOX = [706, 238, 717, 247]
    HEADER_GLYPH = [708.0, 239.6, 714.7, 244.5]
    ROW_GLYPH = [760.5, 256.1, 767.1, 261.0]

    def _items(self, *, vector_backed=True):
        return [
            {"text": "WALLS RAILINGS & FENCING",
             "box_2d": [708.0, 256.2, 714.7, 302.7],
             "source": "vector"},
            {"text": "5' ORNAMENTAL STEEL FENCE & GATE",
             "box_2d": [760.5, 275.0, 767.2, 335.5],
             "source": "vlm", "vec_backed": vector_backed},
        ]

    def _entry(self, *, snap="shape", value="4.0"):
        return {
            "result": {
                "groups": copy.deepcopy(self.GROUPS),
                "plc_v": 99,
                "symbols": [{
                    "box_2d": list(self.PARENT_BOX),
                    "category": "shape", "value": value,
                    "type": f"shape {value}", "text_index": 0,
                    "claimed_group_index": 0, "source": SOURCE_PAGE,
                    "snap": snap,
                }],
            }
        }

    def _lines(self, row_box=None, row_text="4.6"):
        return [
            {"text": "4.0", "box_2d": list(self.HEADER_GLYPH)},
            {"text": row_text,
             "box_2d": list(row_box or self.ROW_GLYPH)},
        ]

    def test_inherits_verified_parent_frame_and_exact_glyph(self):
        entry = self._entry()
        self.assertEqual(
            inherit_row_code_symbols(entry, self._items(), self._lines()), 1)
        symbols = entry["result"]["symbols"]
        self.assertEqual(len(symbols), 2)
        derived = symbols[1]
        self.assertEqual(derived["source"], SOURCE_ROW_CODE)
        self.assertEqual(derived["text_index"], 1)
        self.assertEqual(derived["value"], "4.6")
        self.assertEqual(derived["type"], "shape 4.6")
        self.assertEqual(derived["category"], "shape")
        self.assertEqual(derived["glyph_box_2d"], self.ROW_GLYPH)
        # 4.0 真实六边形 11x9，平移到 4.6 glyph 中心；不是 glyph 紧框。
        self.assertEqual(derived["box_2d"], [758.3, 254.1, 769.3, 263.1])
        self.assertEqual(derived["snap"], "inherited")
        self.assertEqual(derived["match_mode"], "exact_vector_text")
        self.assertEqual(derived["inherited_from"],
                         {"value": "4.0", "box_2d": self.PARENT_BOX,
                          "text_index": 0})
        self.assertNotIn("plc_v", entry["result"])

        snapshot = copy.deepcopy(entry)
        self.assertEqual(
            inherit_row_code_symbols(entry, self._items(), self._lines()), 0)
        self.assertEqual(entry, snapshot)

    def test_parent_must_be_a_vector_verified_closed_shape(self):
        for snap in (None, "glyph", "no_code_glyph"):
            with self.subTest(snap=snap):
                entry = self._entry(snap=snap)
                self.assertEqual(inherit_row_code_symbols(
                    entry, self._items(), self._lines()), 0)
                self.assertEqual(len(entry["result"]["symbols"]), 1)

        # Column fallback also identifies a real closed vector outline.
        entry = self._entry(snap="column")
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines()), 1)

    def test_owner_must_be_native_text_and_immediately_right_of_code(self):
        entry = self._entry()
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(vector_backed=False), self._lines()), 0)

        too_far = [760.5, 190.0, 767.1, 195.0]
        entry = self._entry()
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines(too_far)), 0)

    def test_major_mismatch_and_missing_native_header_are_rejected(self):
        entry = self._entry()
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines(row_text="5.6")), 0)

        entry = self._entry()
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), [self._lines()[1]]), 0)

    def test_an_intervening_section_header_closes_the_parent_section(self):
        lines = self._lines() + [
            {"text": "5.0", "box_2d": [735.0, 250.0, 742.0, 255.0]}]
        entry = self._entry()
        self.assertEqual(
            inherit_row_code_symbols(entry, self._items(), lines), 0)

    def test_ambiguous_row_codes_fail_closed(self):
        lines = self._lines() + [
            {"text": "4.7", "box_2d": [760.5, 264.0, 767.1, 269.0]}]
        entry = self._entry()
        self.assertEqual(
            inherit_row_code_symbols(entry, self._items(), lines), 0)

    def test_stale_derived_row_is_removed_and_rebuilt(self):
        entry = self._entry()
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines()), 1)
        entry["result"]["symbols"][1]["box_2d"] = [1, 1, 2, 2]
        entry["result"]["symbols"][1]["placements"] = [[3, 3, 4, 4]]
        entry["result"]["plc_v"] = 123
        # Same current evidence reconstructs the deterministic box and drops
        # downstream fields that must be recomputed for that new sample.
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines()), 1)
        derived = entry["result"]["symbols"][1]
        self.assertEqual(derived["box_2d"], [758.3, 254.1, 769.3, 263.1])
        self.assertNotIn("placements", derived)
        self.assertNotIn("plc_v", entry["result"])

        # If the parent is no longer a verified outline, stale derived data
        # is removed instead of surviving forever.
        entry["result"]["symbols"][0]["snap"] = "glyph"
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines()), 0)
        self.assertEqual(len(entry["result"]["symbols"]), 1)

    def test_parent_frame_must_not_swallow_owner_text(self):
        entry = self._entry()
        entry["result"]["symbols"][0]["box_2d"] = [706, 238, 717, 300]
        # Keep the exact 4.0 glyph inside the now-too-wide parent.
        self.assertEqual(inherit_row_code_symbols(
            entry, self._items(), self._lines()), 0)
        self.assertEqual(len(entry["result"]["symbols"]), 1)


# --------------------------------------------------- 4. 步骤②b 补扫结果合并

class _FakeSweepModule(types.ModuleType):
    pass


def _install_sweep_module(test, version=3):
    """假的 steps.legend_sweep（只提供 LEGEND_SWEEP_VERSION），零调用。"""
    module = _FakeSweepModule("steps.legend_sweep")
    module.LEGEND_SWEEP_VERSION = version
    saved = sys.modules.get("steps.legend_sweep")
    sys.modules["steps.legend_sweep"] = module

    def restore():
        if saved is None:
            sys.modules.pop("steps.legend_sweep", None)
        else:
            sys.modules["steps.legend_sweep"] = saved

    test.addCleanup(restore)
    return version


class TestMergeSweep(unittest.TestCase):
    def _entry(self, symbols=None, **over):
        groups = copy.deepcopy(GROUPS)
        entry = {"sig": "s1", "v": SYMBOL_VERSION, "pv": SYMBOL_PROMPT_V,
                 "model": MODEL_NAME,
                 "raw": {"groups": groups, "symbols": []},
                 "result": {"groups": groups,
                            "symbols": copy.deepcopy(symbols or [])}}
        entry.update(over)
        return entry

    def _items(self):
        return [_item([700, 730, 710, 800], "6' CHAIN LINK FENCE - SEE SPECS"),
                _item([720, 730, 730, 800], "4' ORNAMENTAL STEEL FENCE"),
                _item([200, 200, 210, 260], "SF-1")]

    def test_added_symbols_pass_the_gates_and_get_source_sweep(self):
        entry = self._entry()
        sweep = {"added": [
            {"box_2d": [700, 700, 710, 720], "category": "shape",
             "value": "12", "text_index": 0, "block_index": 1},
            # 平面图里的 marker：补扫也不许放行（几何闸对补扫一视同仁）
            {"box_2d": IN_VIEW, "category": "shape", "value": "9",
             "text_index": 2},
            # 坏行只跳过，绝不连坐这一页
            {"box_2d": [1, 2, 3], "category": "shape", "value": "",
             "text_index": 0},
            {"box_2d": [720, 700, 730, 720], "category": "circle",
             "value": "", "text_index": 1},
        ], "blocks": 2, "model": "gemini-3.5-pro", "errors": [],
            "skipped": ["P9 无图例组"]}
        version = _install_sweep_module(self)
        self.assertTrue(merge_sweep(entry, self._items(), sweep))
        published = entry["result"]["symbols"]
        self.assertEqual([s["text_index"] for s in published], [0])
        self.assertEqual(published[0]["source"], SOURCE_SWEEP)
        self.assertEqual(published[0]["block_index"], 1)
        self.assertEqual(published[0]["type"], "shape 12")
        self.assertEqual(entry["sweep_v"], version)
        self.assertEqual(entry["sweep"]["blocks"], 2)
        self.assertEqual(entry["sweep"]["model"], "gemini-3.5-pro")
        self.assertEqual(entry["sweep"]["skipped"], ["P9 无图例组"])
        self.assertEqual(entry["sweep"]["invalid_rows"], 2)

    def test_existing_symbols_are_kept_and_stamped_page(self):
        entry = self._entry([_sym([720, 700, 730, 720], 1)])
        _install_sweep_module(self)
        merge_sweep(entry, self._items(), {"added": []})
        published = entry["result"]["symbols"]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["source"], SOURCE_PAGE)

    def test_a_sweep_duplicate_of_a_published_box_collapses(self):
        """补扫又找到了同一个框 → 去重，保留描述行那条（已发布的 page 条目）。"""
        entry = self._entry([_sym([700, 700, 710, 720], 0,
                                  placements=[[1, 1, 2, 2]])])
        _install_sweep_module(self)
        sweep = {"added": [{"box_2d": [700, 700, 710, 720],
                            "category": "shape", "value": "12",
                            "text_index": 0}]}
        merge_sweep(entry, self._items(), sweep)
        published = entry["result"]["symbols"]
        self.assertEqual(len(published), 1)
        self.assertEqual(published[0]["source"], SOURCE_PAGE)
        self.assertEqual(published[0]["placements"], [[1, 1, 2, 2]])
        # 符号集没变 → 已经算好的放置不作废
        self.assertNotIn("plc_v", entry["result"])

    def test_new_symbols_invalidate_placements(self):
        entry = self._entry([_sym([720, 700, 730, 720], 1)])
        entry["result"]["plc_v"] = 1
        _install_sweep_module(self)
        sweep = {"added": [{"box_2d": [700, 700, 710, 720],
                            "category": "shape", "value": "12",
                            "text_index": 0}]}
        self.assertTrue(merge_sweep(entry, self._items(), sweep))
        self.assertEqual(len(entry["result"]["symbols"]), 2)
        self.assertNotIn("plc_v", entry["result"])

    def test_merging_the_same_sweep_twice_is_idempotent(self):
        entry = self._entry()
        _install_sweep_module(self)
        sweep = {"added": [{"box_2d": [700, 700, 710, 720],
                            "category": "shape", "value": "12",
                            "text_index": 0}], "blocks": 1}
        self.assertTrue(merge_sweep(entry, self._items(), sweep))
        snapshot = copy.deepcopy(entry)
        self.assertFalse(merge_sweep(entry, self._items(), sweep))
        self.assertEqual(entry, snapshot)

    def test_crop_box_is_a_valid_frame_for_swept_symbols(self):
        """裁剪窗比组框大一圈，样例落在组框外一点点也得认（几何仍然为准）。"""
        entry = self._entry()
        _install_sweep_module(self)
        just_outside = [806, 700, 814, 720]      # 中心 810 > legend 的 800+2
        sweep = {"added": [{"box_2d": just_outside, "category": "shape",
                            "value": "12", "text_index": 0}],
                 "blocks": [{"group_index": 1, "kind": "legend",
                             "box_2d": [600, 600, 800, 900],
                             "crop_box": [590, 590, 820, 910],
                             "asked": [0], "found": [{}], "elapsed": 3.1}]}
        self.assertTrue(merge_sweep(entry, self._items(), sweep))
        self.assertEqual(len(entry["result"]["symbols"]), 1)
        # 落痕压成一行一块的小结，不把符号在盘上存两遍
        self.assertEqual(entry["sweep"]["blocks"],
                         [{"group_index": 1, "kind": "legend",
                           "crop_box": [590, 590, 820, 910],
                           "asked": 1, "found": 1, "elapsed": 3.1}])
        # crop_box 只是取景框，不是免死金牌：落进平面图组里的照剥
        entry2 = self._entry()
        merge_sweep(entry2, self._items(),
                    {"added": [{"box_2d": IN_VIEW, "category": "shape",
                                "value": "9", "text_index": 2}],
                     "blocks": sweep["blocks"]})
        self.assertEqual(entry2["result"]["symbols"], [])

    def test_the_sweep_result_version_wins_over_the_module_version(self):
        entry = self._entry()
        _install_sweep_module(self, version=3)
        merge_sweep(entry, self._items(), {"version": 2, "added": []})
        self.assertEqual(entry["sweep_v"], 2)
        self.assertFalse(has_current_symbols(entry, "s1"))   # 旧版本 → 该重跑

    def test_empty_sweep_still_stamps_the_entry(self):
        entry = self._entry()
        version = _install_sweep_module(self)
        self.assertTrue(merge_sweep(entry, self._items(), {}))
        self.assertEqual(entry["sweep_v"], version)
        self.assertEqual(entry["result"]["symbols"], [])

    def test_a_broken_entry_is_a_no_op(self):
        self.assertFalse(merge_sweep(None, [], {"added": []}))
        self.assertFalse(merge_sweep({}, [], {"added": []}))
        self.assertFalse(merge_sweep({"result": "nope"}, [], {"added": []}))


# ------------------------------------------------- 5. 缓存当期性里的 sweep_v

class TestSweepCurrency(unittest.TestCase):
    def _entry(self, **over):
        entry = {"sig": "s1", "v": SYMBOL_VERSION, "pv": SYMBOL_PROMPT_V,
                 "model": MODEL_NAME,
                 "raw": {"groups": [], "symbols": []},
                 "result": {"groups": [], "symbols": []}}
        entry.update(over)
        return entry

    def test_missing_sweep_v_is_not_current(self):
        version = _install_sweep_module(self)
        self.assertEqual(sweep_version(), version)
        self.assertFalse(has_current_symbols(self._entry(), "s1"))
        self.assertFalse(
            has_current_symbols(self._entry(sweep_v=version - 1), "s1"))
        self.assertTrue(
            has_current_symbols(self._entry(sweep_v=version), "s1"))

    def test_absent_sweep_module_does_not_require_the_field(self):
        saved = sys.modules.pop("steps.legend_sweep", None)
        self.addCleanup(
            lambda: sys.modules.__setitem__("steps.legend_sweep", saved)
            if saved is not None else None)
        if sweep_version() is not None:      # 补扫模块真的存在 → 这条不适用
            self.skipTest("steps.legend_sweep is importable in this tree")
        self.assertTrue(has_current_symbols(self._entry(), "s1"))


if __name__ == "__main__":
    unittest.main()
