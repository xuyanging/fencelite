"""线型层的判据用例 —— 全部是纯函数，不起边车、不读 PDF、零花费.

覆盖：核心判据（拥有最近 op 的簇 / residual 是一等答案）、三条产品口径、
plan 只当显示闸这一条设计决定，以及一条跨模块一致性（归一化必须和
steps/arrows.py 的同文判据完全一致 —— 两处各写一份是有意的，一致性靠这里保证）。
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps.linetypes import bind, regroup                      # noqa: E402


def row(key, ti, tip, *, near=None, dist=0.2, ranked=(), own_ops=0):
    """构造一条边车 binding。near=None 表示最近的 op 是 residual。"""
    return {
        "key": key, "ti": ti, "tip": list(tip), "own_ops": own_ops,
        "nearest_op": None if dist is None else {
            "op_index": 0, "distance": dist, "owner": near},
        "ranked": [{"line_type_number": number, "distance": value}
                   for number, value in ranked],
    }


ITEMS = [
    {"text": "REMOVE EXISTING FENCE", "box_2d": [10, 10, 20, 90]},     # 0
    {"text": "remove   existing\nfence", "box_2d": [30, 10, 40, 90]},  # 1
    {"text": "REMOVE EXISTING GATE", "box_2d": [50, 10, 60, 90]},      # 2
]


class TextTokenTests(unittest.TestCase):
    def test_normalisation_folds_case_and_whitespace(self):
        self.assertEqual(bind.text_token("REMOVE EXISTING FENCE"),
                         bind.text_token("remove   existing\nfence"))

    def test_matches_arrows_module_normalisation(self):
        """与 steps/arrows.py 的同文判据必须一字不差地一致。

        arrows 用它隐藏弱重复项，线型用它归组。两边分歧的后果是静默的：
        arrows 认为两条是同一句话、线型认为不是（或反之），投票组就错了。
        """
        import re
        import unicodedata
        for sample in ["REMOVE EXISTING FENCE", "remove   existing\nfence",
                       "  Chain-Link  FENCE ", "ＣＬ FENCE", "", "   "]:
            # arrows.py 里的表达式，逐字复制过来比
            expected = re.sub(
                r"\s+", " ",
                unicodedata.normalize("NFKC", str(sample or ""))).strip().upper()
            self.assertEqual(bind.text_token(sample), expected)


class VerdictTests(unittest.TestCase):
    def test_owner_of_nearest_op_is_the_answer(self):
        state, number, distance = bind.verdict_of(
            row("0", 0, (11, 11), near=3, dist=0.4))
        self.assertEqual((state, number), ("bound", 3))
        self.assertAlmostEqual(distance, 0.4)

    def test_residual_is_a_first_class_answer(self):
        """tip 底下那段 ink 不属于任何簇时，正确答案是「这里没有线型」。

        不能退回「最近的簇」—— 实测一页只有 38% 的 path op 属于任何簇，
        那样会稳定地给出看着对的错高亮。
        """
        state, number, _ = bind.verdict_of(
            row("0", 0, (11, 11), near=None, dist=0.1,
                ranked=((3, 8.2), (4, 31.9))))
        self.assertEqual(state, "residual")
        self.assertIsNone(number)

    def test_too_far_when_the_tip_touches_nothing(self):
        state, number, distance = bind.verdict_of(
            row("0", 0, (11, 11), near=3,
                dist=bind.MAX_BIND_DISTANCE + 0.5))
        self.assertEqual(state, "too-far")
        self.assertIsNone(number)
        self.assertGreater(distance, bind.MAX_BIND_DISTANCE)

    def test_no_geometry_when_the_page_has_no_paths(self):
        state, number, _ = bind.verdict_of(
            {"key": "0", "ti": 0, "tip": [1, 1], "nearest_op": None})
        self.assertEqual(state, "no-geometry")
        self.assertIsNone(number)


class BindPageTests(unittest.TestCase):
    def test_residual_target_publishes_nothing(self):
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=None, dist=0.1, ranked=((3, 8.2),)),
        ])
        self.assertEqual(out["used_all"], [])
        self.assertEqual(out["bindings"][0]["state"], "residual")
        self.assertIsNone(out["bindings"][0]["line_type_number"])

    def test_multiple_targets_of_one_callout_share_one_line_type(self):
        """一条 callout 的多个末端必须落在同一个线型上（口径 2）。"""
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=0.1),
            row("0", 1, (80, 80), near=7, dist=0.4, ranked=((7, 0.4), (3, 2.0))),
            row("0", 2, (90, 90), near=3, dist=0.9),
        ])
        self.assertEqual(out["used_all"], [3])
        states = [b["state"] for b in out["bindings"]]
        self.assertEqual(states.count("reassigned"), 1)
        self.assertTrue(all(b["line_type_number"] == 3 for b in out["bindings"]))
        moved = next(b for b in out["bindings"] if b["state"] == "reassigned")
        self.assertEqual(moved["nearest"], 7)
        self.assertAlmostEqual(moved["distance_to_type"], 2.0)

    def test_same_text_callouts_share_one_line_type(self):
        """文字相同的不同 callout 归一组（口径 3），并把少数票拉回来。"""
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=0.1),
            row("1", 0, (31, 31), near=9, dist=0.2, ranked=((9, 0.2), (3, 1.5))),
            row("0", 1, (12, 12), near=3, dist=0.3),
        ])
        self.assertEqual(out["used_all"], [3])
        self.assertEqual(len(out["groups"]), 1)
        self.assertEqual(out["groups"][0]["votes_all"], {"3": 2, "9": 1})

    def test_residual_member_is_not_pulled_into_the_group_answer(self):
        """组里别人拿到了线型，也不给 residual 的那条外推 —— 那等于用别处的
        证据在这里画一条线。"""
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=0.1),
            row("0", 1, (80, 80), near=None, dist=0.1, ranked=((3, 6.0),)),
        ])
        self.assertEqual(out["used_all"], [3])
        residual = out["bindings"][1]
        self.assertEqual(residual["state"], "residual")
        self.assertIsNone(residual["line_type_number"])

    def test_different_text_stays_in_separate_groups(self):
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=0.1),
            row("2", 0, (51, 51), near=1, dist=0.2),
        ])
        self.assertEqual(out["used_all"], [1, 3])
        self.assertEqual(len(out["groups"]), 2)

    def test_placement_anchor_has_no_text_and_stands_alone(self):
        out = bind.bind_page(ITEMS, [
            row("s0:1", 0, (11, 11), near=4, dist=0.1),
            row("s0:2", 0, (12, 12), near=5, dist=0.1),
        ])
        self.assertEqual(len(out["groups"]), 2)
        self.assertEqual(out["used_all"], [4, 5])

    def test_tie_is_broken_deterministically_by_total_distance(self):
        rows = [row("0", 0, (11, 11), near=3, dist=5.0),
                row("0", 1, (12, 12), near=7, dist=1.0)]
        out = bind.bind_page(ITEMS, rows)
        self.assertTrue(out["groups"][0]["tie"])
        self.assertEqual(out["groups"][0]["line_type_number"], 7)
        again = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=5.0),
            row("0", 1, (12, 12), near=7, dist=1.0)])
        self.assertEqual(again["groups"][0]["line_type_number"], 7)

    def test_unreferenced_line_types_are_excluded(self):
        """没有被任何一组指到的线型不进 used_all —— 排除无关线型的判据。"""
        out = bind.bind_page(ITEMS, [
            row("0", 0, (11, 11), near=3, dist=0.1,
                ranked=((3, 0.1), (6, 30.0), (8, 90.0))),
        ])
        self.assertEqual(out["used_all"], [3])


def entry_of(rows, line_types=None, precision=0.0):
    """拼一个 linetypes.json 的 entry：缓存期只需要 bindings + 有哪些几何。"""
    numbers = set()
    for r in rows:
        n = (r.get("nearest_op") or {}).get("owner")
        if n is not None:
            numbers.add(n)
        for c in r.get("ranked") or ():
            numbers.add(c["line_type_number"])
        o = r.get("nearest_owned_op") or {}
        if o.get("owner") is not None:
            numbers.add(o["owner"])
    types = line_types if line_types is not None else [
        {"line_type_number": n, "polylines": [[[0, 0], [1, 1]]],
         "segment_count": 1, "signature_family": "compound_path_periodic"}
        for n in sorted(numbers)]
    return {"bindings": rows, "line_types": types,
            "page": {"tip_precision_pt": precision}}


class PlanDisplayGateTests(unittest.TestCase):
    def _entry(self):
        return entry_of([
            row("0", 0, (100, 100), near=3, dist=0.1),
            row("2", 0, (900, 900), near=1, dist=0.2),
        ])

    def test_only_terminals_inside_plan_are_visible(self):
        out = regroup.resolve(self._entry(), [[0, 0, 500, 500]], ITEMS)
        self.assertEqual(out["visible"], [3])
        inside = next(r for r in out["bindings"] if r["tip"] == [100, 100])
        outside = next(r for r in out["bindings"] if r["tip"] == [900, 900])
        self.assertTrue(inside["in_plan"])
        self.assertTrue(inside["visible"])
        self.assertFalse(outside["in_plan"])
        self.assertFalse(outside["visible"])

    def test_no_plan_hides_everything_but_keeps_the_data(self):
        entry = self._entry()
        out = regroup.resolve(entry, [], ITEMS)
        self.assertEqual(out["visible"], [])
        fence = [g for g in out["groups"] if g["scope"] == "fence"]
        gate = [g for g in out["groups"] if g["scope"] == "gate"]
        # fence 组是「被 plan 挡住」→ plan_fallback；gate 组是「不找线」→ 不是。
        self.assertTrue(all(g["plan_fallback"] for g in fence))
        self.assertTrue(gate and not any(g["plan_fallback"] for g in gate))
        self.assertEqual([g["line_type_number_all"] for g in fence], [3])

    def test_plan_gate_does_not_mutate_the_cached_entry(self):
        entry = self._entry()
        before = [dict(r) for r in entry["bindings"]]
        regroup.resolve(entry, [[0, 0, 500, 500]], ITEMS)
        self.assertEqual(entry["bindings"], before)

    def test_votes_are_recounted_among_in_plan_terminals_only(self):
        """plan 外的末端很可能就近咬到图例样例，不许它左右可见结果。"""
        entry = entry_of([
            row("0", 0, (100, 100), near=3, dist=0.1),
            row("0", 1, (900, 900), near=9, dist=0.1),
            row("0", 2, (901, 901), near=9, dist=0.1),
        ])
        self.assertEqual(
            regroup.resolve(entry, [[0, 0, 1000, 1000]], ITEMS)["visible"], [9])
        out = regroup.resolve(entry, [[0, 0, 105, 105]], ITEMS)
        self.assertEqual(out["visible"], [3])
        self.assertEqual(out["groups"][0]["votes_in_plan"], {"3": 1})

    def test_residual_terminal_inside_plan_yields_no_visible_type(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=None, dist=0.1, ranked=((3, 7.0),)),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], ITEMS)
        self.assertEqual(out["visible"], [])
        group = out["groups"][0]
        # 这一组压根没有答案（residual），不是「有答案但被 plan 挡住」。
        # 两者必须分开：前者是"这里没有线型"，后者是"这次不显示"。
        self.assertIsNone(group["line_type_number_all"])
        self.assertFalse(group["plan_fallback"])
        self.assertEqual(out["bindings"][0]["state"], "residual")


class SymbolAnchorGroupingTests(unittest.TestCase):
    """放置锚必须归到「它所属图例那一行文字」的组里（与前端 SCOPE_BY_SYMBOL 一致）。

    真实案例：rapid_city_2 P11 只有 3 个放置锚，其中 s1:0 与 s1:1 是**同一个**
    图例样例的两处放置，按旧的「每个放置各自成组」它们分别绑到了 #48 / #49；
    按产品口径必须是同一线型。
    """

    ITEMS = [
        {"text": "6' CHAIN LINK FENCE", "box_2d": [10, 10, 20, 90]},   # 0
        {"text": "DOUBLE SWING GATE", "box_2d": [30, 10, 40, 90]},     # 1
    ]
    OWNERS = {0: 1, 1: 0}       # symbol 0 属于 GATE 那行；symbol 1 属于 FENCE 那行

    def test_two_placements_of_one_symbol_vote_together(self):
        entry = entry_of([
            row("s1:0", 0, (100, 100), near=48, dist=0.07),
            row("s1:1", 0, (200, 200), near=49, dist=0.10,
                ranked=((49, 0.10), (48, 2.0))),
        ])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS,
                              self.OWNERS)
        self.assertEqual(len(out["groups"]), 1)
        self.assertEqual(out["groups"][0]["keys"], ["s1:0", "s1:1"])
        self.assertEqual(out["visible"], [48])
        moved = next(b for b in out["bindings"] if b["key"] == "s1:1")
        self.assertEqual(moved["state"], "reassigned")
        self.assertEqual(moved["line_type_number"], 48)

    def test_placement_inherits_the_legend_rows_gate_class(self):
        entry = entry_of([row("s0:0", 0, (100, 100), near=12, dist=0.05)])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS,
                              self.OWNERS)
        self.assertEqual(out["groups"][0]["scope"], "gate")
        self.assertEqual(out["visible"], [])
        self.assertEqual(out["bindings"][0]["state"], "gate")

    def test_placement_without_owner_groups_by_symbol_index(self):
        entry = entry_of([
            row("s3:0", 0, (100, 100), near=5, dist=0.1),
            row("s3:1", 0, (110, 110), near=5, dist=0.2),
            row("s4:0", 0, (120, 120), near=6, dist=0.1),
        ])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS, {})
        keys = sorted(g["group"] for g in out["groups"])
        self.assertEqual(keys, ["s:3", "s:4"])
        self.assertEqual(out["visible"], [5, 6])


class ReassignReachTests(unittest.TestCase):
    """改判不能把末端挪到一条它根本没碰到的线上。

    真实案例：rapid_city_2 P8 的 anchor 0 最近 op 在 20.3 pt 外、候选第三名还在
    31.6 pt（也就是离胜出的 #15 至少 31 pt），旧规则照样把它改判成了 #15。
    """

    def test_winner_out_of_reach_is_unresolved_not_reassigned(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=15, dist=0.013),
            row("0", 1, (600, 180), near=12, dist=20.309,
                ranked=((12, 20.309), (8, 29.828), (7, 31.576))),
        ], line_types=[{"line_type_number": n, "polylines": [[[0, 0], [1, 1]]],
                        "segment_count": 1, "signature_family": "x"}
                       for n in (7, 8, 12, 15)])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], ITEMS)
        self.assertEqual(out["visible"], [15])
        far = next(b for b in out["bindings"] if b["ti"] == 1)
        self.assertEqual(far["state"], "unresolved")
        self.assertIsNone(far["line_type_number"])

    def test_winner_within_reach_is_reassigned(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=15, dist=0.013),
            row("0", 1, (110, 110), near=12, dist=1.0,
                ranked=((12, 1.0), (15, 3.0))),
        ])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], ITEMS)
        self.assertEqual(out["visible"], [15])
        moved = next(b for b in out["bindings"] if b["ti"] == 1)
        self.assertEqual(moved["state"], "reassigned")
        self.assertEqual(moved["line_type_number"], 15)
        self.assertAlmostEqual(moved["distance_to_type"], 3.0)


class NeedsRecomputeTests(unittest.TestCase):
    def test_missing_geometry_is_reported_not_silently_empty(self):
        """旧缓存按当时的分组裁剪过折线；换了分组口径之后要显式说"要重算"。"""
        entry = entry_of(
            [row("0", 0, (100, 100), near=3, dist=0.1)],
            line_types=[{"line_type_number": 3, "segment_count": 0,
                         "signature_family": "x", "pruned": True}])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], ITEMS)
        self.assertEqual(out["visible"], [])
        self.assertEqual(out["needs_recompute"], [3])


if __name__ == "__main__":
    unittest.main()
