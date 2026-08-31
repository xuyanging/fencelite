"""线型层的判据用例 —— 全部是纯函数，不起边车、不读 PDF、零花费.

覆盖：核心判据（36 pt 内最近的已识别线型 / 无已识别线型时 residual）、三条产品口径、
plan 只当显示闸这一条设计决定，以及一条跨模块一致性（归一化必须和
steps/arrows.py 的同文判据完全一致 —— 两处各写一份是有意的，一致性靠这里保证）。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps import linetypes                                    # noqa: E402
from steps.linetypes import bind, regroup                      # noqa: E402


def all_geometry_pair():
    """Matching main-cache metadata and a full-geometry rerun."""
    page = {
        "page_fingerprint": "page-fingerprint",
        "owned_ops_sha1": "owned-ops",
        "fused_ops_sha1": "fused-ops",
        "path_ops": 6,
        "owned_path_ops": 3,
    }
    rows = [
        {"line_type_number": 1, "signature_family": "motif_periodic",
         "recognition_source": "method1", "op_count": 2,
         "ops_sha1": "type-one", "segment_count": 1},
        {"line_type_number": 2, "signature_family": "pdf_text_dash_line",
         "recognition_source": "method2", "op_count": 1,
         "ops_sha1": "type-two", "segment_count": 1,
         "pattern_instance_count": 1,
         "pattern_instances": [{
             "region_id": "pdf-text-1", "literal_text": "8'",
             "bbox": [10, 20, 30, 40],
         }]},
    ]
    main = {"sig": "current-signature", "v": 5, "page": dict(page),
            "all_line_types": [dict(row) for row in rows]}
    fresh_rows = []
    # Deliberately reverse the producer order: publication must be stable by
    # line_type_number rather than trusting incidental subprocess ordering.
    for row in reversed(rows):
        fresh_rows.append({
            **row,
            "by_run": [{
                "run_id": "1", "op_count": row["op_count"],
                "segment_count": 1, "bbox": [1, 2, 3, 4],
                "polylines": [[[1, 2], [3, 4]]],
            }],
        })
    fresh = {
        "ok": True, "page": dict(page), "engine": {"engine": "unit"},
        "types": fresh_rows,
        "residual": {"op_count": 3, "segment_count": 1,
                     "polylines": [[[5, 6], [7, 8]]]},
    }
    return main, fresh


class AllGeometryVerificationTests(unittest.TestCase):
    def test_exact_operation_identity_is_normalized_for_publication(self):
        main, fresh = all_geometry_pair()
        out = linetypes.verify_all_page_geometry(main, fresh)
        self.assertEqual(out["sig"], main["sig"])
        self.assertEqual(out["v"], main["v"])
        self.assertEqual(out["all_v"], linetypes.ALL_GEOMETRY_VERSION)
        self.assertEqual(out["producer_sha256"],
                         linetypes.sidecar.all_geometry_digest())
        self.assertEqual([row["line_type_number"] for row in out["types"]],
                         [1, 2])
        self.assertEqual(out["page"], fresh["page"])
        self.assertEqual(out["engine"], fresh["engine"])
        self.assertEqual(out["residual"], fresh["residual"])
        self.assertNotIn("ok", out)

    def test_every_page_ownership_fingerprint_is_required_and_exact(self):
        for key in ("page_fingerprint", "owned_ops_sha1", "fused_ops_sha1",
                    "path_ops", "owned_path_ops"):
            with self.subTest(key=key, condition="different"):
                main, fresh = all_geometry_pair()
                fresh["page"][key] = "different"
                with self.assertRaisesRegex(linetypes.AllGeometryMismatch, key):
                    linetypes.verify_all_page_geometry(main, fresh)
            with self.subTest(key=key, condition="missing"):
                main, fresh = all_geometry_pair()
                fresh["page"].pop(key)
                with self.assertRaisesRegex(linetypes.AllGeometryMismatch, key):
                    linetypes.verify_all_page_geometry(main, fresh)

    def test_type_number_sets_must_match_in_both_directions(self):
        main, fresh = all_geometry_pair()
        fresh["types"] = fresh["types"][:1]
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "missing type numbers"):
            linetypes.verify_all_page_geometry(main, fresh)

        main, fresh = all_geometry_pair()
        fresh["types"].append({
            "line_type_number": 9, "signature_family": "motif_periodic",
            "recognition_source": "method1", "op_count": 1,
            "ops_sha1": "unexpected", "segment_count": 0, "by_run": [],
        })
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "extra type numbers"):
            linetypes.verify_all_page_geometry(main, fresh)

    def test_each_type_operation_identity_field_must_match(self):
        replacements = {
            "signature_family": "different-family",
            "recognition_source": "different-source",
            "op_count": 999,
            "ops_sha1": "different-operation-set",
            "segment_count": 999,
        }
        for key, value in replacements.items():
            with self.subTest(key=key):
                main, fresh = all_geometry_pair()
                target = next(row for row in fresh["types"]
                              if row["line_type_number"] == 1)
                target[key] = value
                with self.assertRaisesRegex(
                        linetypes.AllGeometryMismatch,
                        r"type #1 operation identity differs"):
                    linetypes.verify_all_page_geometry(main, fresh)

    def test_method2_pattern_instances_must_match_main_result(self):
        main, fresh = all_geometry_pair()
        target = next(row for row in fresh["types"]
                      if row["line_type_number"] == 2)
        target["pattern_instances"] = [dict(target["pattern_instances"][0])]
        target["pattern_instances"][0]["bbox"] = [0, 0, 1, 1]
        with self.assertRaisesRegex(
                linetypes.AllGeometryMismatch,
                r"type #2 operation identity differs"):
            linetypes.verify_all_page_geometry(main, fresh)

    def test_duplicate_numbers_and_missing_run_geometry_are_rejected(self):
        main, fresh = all_geometry_pair()
        fresh["types"].append(dict(fresh["types"][0]))
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "duplicate line type"):
            linetypes.verify_all_page_geometry(main, fresh)

        main, fresh = all_geometry_pair()
        fresh["types"][0].pop("by_run")
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "has no run geometry"):
            linetypes.verify_all_page_geometry(main, fresh)

    def test_compute_wrapper_runs_full_sidecar_then_verifies(self):
        main, fresh = all_geometry_pair()
        with mock.patch.object(
                linetypes.sidecar, "run_all_page", return_value=fresh) as run:
            out = linetypes.compute_all_page_geometry(
                "/tmp/unit.pdf", 3, main, timeout=45, cpu_budget=2)
        self.assertEqual(out["sig"], main["sig"])
        run.assert_called_once_with(
            "/tmp/unit.pdf", 3, timeout=45, cpu_budget=2,
            residual=True, dbg=None)

    def test_zero_types_still_publishes_residual_geometry(self):
        page = {
            "page_fingerprint": "zero-page",
            "owned_ops_sha1": "empty-owner-set",
            "fused_ops_sha1": "empty-fused-set",
            "path_ops": 1,
            "owned_path_ops": 0,
        }
        main = {"sig": "zero", "v": 5, "page": dict(page),
                "all_line_types": []}
        fresh = {
            "page": dict(page), "types": [], "engine": {},
            "residual": {"op_count": 1, "segment_count": 1,
                         "polylines": [[[1, 2], [3, 4]]]},
        }
        out = linetypes.verify_all_page_geometry(main, fresh)
        self.assertEqual(out["types"], [])
        self.assertEqual(out["residual"]["op_count"], 1)

    def test_geometry_bucket_and_residual_counts_are_verified(self):
        main, fresh = all_geometry_pair()
        fresh["types"][0]["by_run"][0]["polylines"] = []
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "geometry count differs"):
            linetypes.verify_all_page_geometry(main, fresh)

        main, fresh = all_geometry_pair()
        fresh["residual"]["op_count"] = 2
        with self.assertRaisesRegex(linetypes.AllGeometryMismatch,
                                    "residual geometry count differs"):
            linetypes.verify_all_page_geometry(main, fresh)

    def test_cached_geometry_requires_current_producer_and_full_validation(self):
        main, fresh = all_geometry_pair()
        cached = linetypes.verify_all_page_geometry(main, fresh)
        self.assertIsNotNone(linetypes.validated_all_page(
            cached, main, main["sig"]))

        old = dict(cached, producer_sha256="old-producer")
        self.assertIsNone(linetypes.validated_all_page(old, main, main["sig"]))

        corrupt = dict(cached)
        corrupt["page"] = dict(cached["page"], owned_ops_sha1="corrupt")
        self.assertIsNone(linetypes.validated_all_page(
            corrupt, main, main["sig"]))

    def test_all_payload_rechecks_retained_binding_evidence_and_labels_origin(self):
        main, fresh = all_geometry_pair()
        main["page"]["tip_precision_pt"] = 0.1
        main["bindings"] = [{
            "key": 4, "ti": 0,
            "nearest_op": {"distance": 0.2, "owner": None},
            "nearest_owned_op": {"distance": 3.4, "owner": 2},
        }, {
            "key": "s3:0", "ti": 0,
            "nearest_op": {"distance": 0.1, "owner": 2},
            "line_type_number": 2, "distance_to_type": 0.1,
        }, {
            "source": "legend_template", "key": "s5:0", "ti": 0,
            "nearest_op": {"distance": 0.0, "owner": 2},
        }]
        payload = linetypes.all_payload(fresh, main)
        target = next(row for row in payload["types"]
                      if row["line_type_number"] == 2)
        self.assertEqual(
            [row["binding_kind"] for row in target["bound_by"]],
            ["text_callout", "symbol_callout", "legend_sample"])
        self.assertEqual(target["bound_by"][0]["distance"], 3.4)


def row(key, ti, tip, *, near=None, dist=0.2, ranked=(), own_ops=0):
    """构造一条边车 binding。near=None 表示最近的 op 是 residual。"""
    return {
        "key": key, "ti": ti, "tip": list(tip), "own_ops": own_ops,
        "nearest_op": None if dist is None else {
            "op_index": 0, "distance": dist, "owner": near},
        "ranked": [{"line_type_number": number, "distance": value}
                   for number, value in ranked],
    }


def legend_row(symbol_index, tip, *, tips=None, matched_runs=(), near=None,
               dist=0.2, ranked=()):
    """构造一份 legend 模板 binding；多个 tips 仍只是一份 symbol 证据。"""
    out = row(f"s{symbol_index}:0", 0, tip, near=near, dist=dist,
              ranked=ranked)
    out.update({
        "source": "legend_template",
        "tips": [list(value) for value in (tips or ())],
        "matched_runs": list(matched_runs),
    })
    return out


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

    def test_residual_without_an_authoritative_nearest_owned_op(self):
        """ranked 只是候选表；没有 nearest_owned_op 时不能凭候选硬绑定。"""
        state, number, _ = bind.verdict_of(
            row("0", 0, (11, 11), near=None, dist=0.1,
                ranked=((3, 8.2), (4, 31.9))))
        self.assertEqual(state, "residual")
        self.assertIsNone(number)

    def test_nearest_recognised_type_wins_over_closer_residual_ink(self):
        """final_plans P3 key 1：旧 tip-precision 闸误杀了 Method 2 #2。"""
        target = row("1", 0, (173, 656), near=None, dist=0.334,
                     ranked=((2, 3.437), (5, 4.042), (40, 108.474)))
        target["nearest_owned_op"] = {
            "op_index": 9290, "distance": 3.437, "owner": 2,
            "run_id": "83", "group_id": "197",
        }
        state, number, distance = bind.verdict_of(target, 1.512)
        self.assertEqual((state, number), ("bound", 2))
        self.assertAlmostEqual(distance, 3.437)

    def test_nearest_recognised_fallback_obeys_the_12pt_guard(self):
        for owned_distance, expected in (
                (bind.MAX_FALLBACK_BIND_DISTANCE, "bound"),
                (bind.MAX_FALLBACK_BIND_DISTANCE + 0.001, "residual")):
            with self.subTest(owned_distance=owned_distance):
                target = row("0", 0, (11, 11), near=None, dist=0.1)
                target["nearest_owned_op"] = {
                    "op_index": 2, "distance": owned_distance, "owner": 7,
                }
                state, number, _ = bind.verdict_of(target, 0.0)
                self.assertEqual(state, expected)
                self.assertEqual(number, 7 if expected == "bound" else None)

    def test_direct_owner_keeps_the_existing_36pt_guard(self):
        state, number, distance = bind.verdict_of(
            row("0", 0, (11, 11), near=7,
                dist=bind.MAX_BIND_DISTANCE))
        self.assertEqual((state, number), ("bound", 7))
        self.assertEqual(distance, bind.MAX_BIND_DISTANCE)

    def test_symbol_center_uses_its_dedicated_48pt_guard(self):
        """P4 的正确共同 #45 在 37.593pt，不能受箭头 36pt 护栏误杀。"""
        self.assertEqual(bind.MAX_SYMBOL_CENTER_DISTANCE, 48.0)
        for owned_distance, expected in (
                (bind.MAX_SYMBOL_CENTER_DISTANCE, "bound"),
                (bind.MAX_SYMBOL_CENTER_DISTANCE + 0.001, "too-far")):
            with self.subTest(owned_distance=owned_distance):
                target = row("s0:0", -1, (110, 220), near=None, dist=0.1)
                target["anchor_kind"] = "symbol_center"
                target["nearest_owned_op"] = {
                    "op_index": 9, "distance": owned_distance, "owner": 45,
                }
                state, number, distance = bind.verdict_of(target)
                self.assertEqual(state, expected)
                self.assertEqual(number, 45 if expected == "bound" else None)
                self.assertEqual(distance, owned_distance)

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


class GateTextClassificationTests(unittest.TestCase):
    def test_pure_gate_text_remains_gate(self):
        for text in ("GATE", "DOUBLE SWING GATE", "EXISTING GATES"):
            with self.subTest(text=text):
                self.assertTrue(bind.is_gate_text(text))
        self.assertFalse(bind.is_gate_text("AGGREGATE BASE"))

    def test_explicit_fence_wins_over_gate_in_the_same_text(self):
        for text in ("5' ORNAMENTAL STEEL FENCE & GATE",
                     "FENCING / GATES", "FENCED ACCESS GATE",
                     "FENCES AND GATES", "FENCELINE GATE",
                     "FENCE LINE / GATE"):
            with self.subTest(text=text):
                self.assertFalse(bind.is_gate_text(text))

    def test_fence_and_gate_group_remains_eligible_for_a_line_type(self):
        items = [{"text": "5' ORNAMENTAL STEEL FENCE & GATE",
                  "box_2d": [10, 10, 20, 90]}]
        entry = entry_of([row("0", 0, (100, 100), near=7, dist=0.1)])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], items)
        self.assertEqual(out["groups"][0]["scope"], "fence")
        self.assertEqual(out["groups"][0]["visible_line_type_number"], 7)
        self.assertEqual(out["bindings"][0]["state"], "bound")
        self.assertEqual(out["visible"], [7])


class SymbolCenterAnchorTests(unittest.TestCase):
    ITEMS = [
        {"text": "6' CHAIN LINK FENCE", "box_2d": [10, 10, 20, 90]},
        {"text": "DOUBLE SWING GATE", "box_2d": [30, 10, 40, 90]},
        {"text": "5' ORNAMENTAL STEEL FENCE & GATE",
         "box_2d": [50, 10, 60, 90]},
    ]

    @staticmethod
    def symbols(owner, placements):
        return {"symbols": [{"text_index": owner,
                              "placements": placements}]}

    def test_anchors_of_combines_arrow_tip_and_symbol_center(self):
        leader = [[100, 100], [105, 110]]
        arrow = [[105, 110], [108, 112]]
        arrow_entry = {"items": {"0": {
            "leader_strokes": [leader], "arrow_strokes": [arrow],
            "targets": [{"tip": [108, 112]}],
        }}}
        anchors = linetypes.anchors_of(
            arrow_entry,
            self.symbols(0, [[100, 200, 120, 240]]),
            self.ITEMS)
        self.assertEqual(len(anchors), 2)
        self.assertEqual(anchors[0], {
            "key": "0", "ti": 0, "tip": [108.0, 112.0],
            "own": [leader, arrow], "anchor_kind": "arrow_tip",
        })
        self.assertEqual(anchors[1], {
            "key": "s0:0", "ti": -1, "tip": [110.0, 220.0],
            "own": [], "anchor_kind": "symbol_center",
            "exclude_box": [100.0, 200.0, 120.0, 240.0],
        })

    def test_pure_gate_symbol_does_not_emit_a_center(self):
        anchors = linetypes.symbol_center_anchors(
            self.symbols(1, [[100, 200, 120, 240]]), self.ITEMS)
        self.assertEqual(anchors, [])

    def test_fence_and_gate_symbol_keeps_fence_priority(self):
        anchors = linetypes.symbol_center_anchors(
            self.symbols(2, [[100, 200, 120, 240]]), self.ITEMS)
        self.assertEqual(len(anchors), 1)
        self.assertEqual(anchors[0]["anchor_kind"], "symbol_center")
        self.assertEqual(anchors[0]["key"], "s0:0")

    def test_invalid_owner_and_bad_placement_boxes_fail_closed(self):
        bad_boxes = [
            None,
            [1, 2, 3],
            ["bad", 10, 20, 30],
            [float("nan"), 10, 20, 30],
            [10, 10, 10, 20],
            [10, 10, 20, 10],
        ]
        result = {"symbols": [
            {"text_index": 0, "placements": bad_boxes},
            {"text_index": True, "placements": [[1, 2, 3, 4]]},
            {"text_index": -1, "placements": [[1, 2, 3, 4]]},
            {"text_index": len(self.ITEMS),
             "placements": [[1, 2, 3, 4]]},
        ]}
        self.assertEqual(
            linetypes.symbol_center_anchors(result, self.ITEMS), [])

    def test_sidecar_wire_payload_preserves_center_protocol_fields(self):
        target = {
            "key": "s2:4", "ti": -1, "tip": [110, 220], "own": [],
            "anchor_kind": "symbol_center",
            "exclude_box": [100, 200, 120, 240],
        }
        with mock.patch.object(linetypes.sidecar, "sidecar_available",
                               return_value=True), \
                mock.patch.object(linetypes.sidecar, "_run_job",
                                  return_value={"ok": True}) as invoke:
            out = linetypes.sidecar.run_page("unit.pdf", 1, [target])
        self.assertEqual(out, {"ok": True})
        payload = invoke.call_args.args[1]
        self.assertEqual(payload["targets"], [{
            "key": "s2:4", "ti": -1, "tip": [110.0, 220.0], "own": [],
            "anchor_kind": "symbol_center",
            "exclude_box": [100, 200, 120, 240],
        }])


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

    def test_inside_terminal_uses_nearest_owned_type_despite_closer_residual(self):
        target = row("0", 0, (173, 656), near=None, dist=0.334,
                     ranked=((2, 3.437), (5, 4.042)))
        target["nearest_owned_op"] = {
            "op_index": 9290, "distance": 3.437, "owner": 2,
            "run_id": "83", "group_id": "197",
        }
        entry = entry_of([target], precision=1.512)
        out = regroup.resolve(entry, [[40, 90, 960, 840]], ITEMS)
        self.assertEqual(out["visible"], [2])
        self.assertEqual(out["groups"][0]["engine_runs"], ["83"])
        binding = out["bindings"][0]
        self.assertTrue(binding["in_plan"])
        self.assertEqual(binding["state"], "bound")
        self.assertEqual(binding["line_type_number"], 2)
        self.assertAlmostEqual(binding["distance_to_type"], 3.437)


class SymbolAnchorGroupingTests(unittest.TestCase):
    """放置锚必须归到「它所属图例那一行文字」的组里（与前端 SCOPE_BY_SYMBOL 一致）。

    真实案例：rapid_city_2 P11 只有 3 个放置锚，其中 s1:0 与 s1:1 是**同一个**
    图例样例的两处放置，按旧的「每个放置各自成组」它们分别绑到了 #48 / #49；
    按产品口径必须是同一线型。
    """

    ITEMS = [
        {"text": "6' CHAIN LINK FENCE", "box_2d": [10, 10, 20, 90]},   # 0
        {"text": "DOUBLE SWING GATE", "box_2d": [30, 10, 40, 90]},     # 1
        {"text": "5' ORNAMENTAL STEEL FENCE & GATE",
         "box_2d": [50, 10, 60, 90]},                                  # 2
    ]
    # symbol 0 属于纯 GATE；symbol 1 属于纯 FENCE；symbol 2 属于复合文字。
    OWNERS = {0: 1, 1: 0, 2: 2}

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

    def test_placement_inherits_fence_priority_from_compound_legend_text(self):
        entry = entry_of([row("s2:0", 0, (100, 100), near=21, dist=0.05)])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS,
                              self.OWNERS)
        self.assertEqual(out["groups"][0]["scope"], "fence")
        self.assertEqual(out["groups"][0]["visible_line_type_number"], 21)
        self.assertEqual(out["visible"], [21])
        self.assertEqual(out["bindings"][0]["state"], "bound")

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


class SymbolCenterConsensusTests(unittest.TestCase):
    """无引线中心只有跨 placement 形成共同可达共识后才能改判。"""

    ITEMS = [{"text": "6' DECORATIVE FENCE",
              "box_2d": [10, 10, 20, 90]}]
    OWNERS = {1: 0}

    @staticmethod
    def center(key, tip, *, near, dist, ranked):
        target = row(key, 0, tip, near=near, dist=dist, ranked=ranked)
        target["anchor_kind"] = "symbol_center"
        return target

    def resolve(self, rows):
        return regroup.resolve(entry_of(rows), [[0, 0, 1000, 1000]],
                               self.ITEMS, self.OWNERS)

    def test_common_reachable_candidate_overrides_one_false_arrow(self):
        """P4 形态：两中心个人最近不同，但 #45 是二者共同可达且更具体。"""
        out = self.resolve([
            self.center("s1:0", (100, 100), near=45, dist=1.0,
                        ranked=((45, 1.0), (19, 1.0))),
            self.center("s1:1", (200, 200), near=9, dist=2.0,
                        ranked=((9, 2.0), (45, 4.0), (19, 4.0))),
            row("s1:0", 0, (101, 101), near=30, dist=0.05,
                ranked=((30, 0.05), (45, 3.0))),
        ])
        group = out["groups"][0]
        self.assertEqual(group["votes_in_plan"], {"30": 1})
        self.assertEqual(group["symbol_center_coverage_in_plan"],
                         {"9": 1, "19": 2, "45": 2})
        self.assertEqual(group["symbol_center_first_votes_in_plan"],
                         {"9": 1, "45": 1})
        self.assertEqual(group["symbol_center_line_type_number"], 45)
        self.assertTrue(group["symbol_center_consensus_applied"])
        self.assertEqual(group["symbol_center_contribution"],
                         "consensus_override")
        self.assertEqual(group["visible_line_type_number"], 45)
        self.assertEqual(out["visible"], [45])

    def test_strict_majority_tolerates_one_center_outlier(self):
        out = self.resolve([
            self.center("s1:0", (100, 100), near=8, dist=1.0,
                        ranked=((8, 1.0),)),
            self.center("s1:1", (200, 200), near=8, dist=2.0,
                        ranked=((8, 2.0),)),
            self.center("s1:2", (300, 300), near=7, dist=1.0,
                        ranked=((7, 1.0),)),
            row("s1:0", 0, (101, 101), near=4, dist=0.1,
                ranked=((4, 0.1), (8, 2.0))),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_coverage_in_plan"],
                         {"7": 1, "8": 2})
        self.assertEqual(group["symbol_center_line_type_number"], 8)
        self.assertEqual(group["visible_line_type_number"], 8)

    def test_matching_arrow_is_reported_as_corroboration_not_override(self):
        out = self.resolve([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0),)),
            self.center("s1:1", (200, 200), near=3, dist=2.0,
                        ranked=((3, 2.0),)),
            row("s1:0", 0, (101, 101), near=3, dist=0.05,
                ranked=((3, 0.05),)),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_contribution"], "corroborated")
        self.assertFalse(group["symbol_center_consensus_applied"])
        self.assertEqual(group["visible_line_type_number"], 3)

    def test_equal_center_evidence_fails_closed_and_keeps_arrow(self):
        out = self.resolve([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0), (7, 2.0))),
            self.center("s1:1", (200, 200), near=7, dist=1.0,
                        ranked=((7, 1.0), (3, 2.0))),
            row("s1:0", 0, (101, 101), near=9, dist=0.1,
                ranked=((9, 0.1),)),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_finalists_in_plan"], [3, 7])
        self.assertIsNone(group["symbol_center_line_type_number"])
        self.assertFalse(group["symbol_center_consensus_applied"])
        self.assertEqual(group["visible_line_type_number"], 9)

    def test_single_center_does_not_change_existing_arrow_result(self):
        out = self.resolve([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0),)),
            row("s1:0", 0, (101, 101), near=9, dist=0.1,
                ranked=((9, 0.1),)),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_placement_count_in_plan"], 1)
        self.assertEqual(group["symbol_center_line_type_number"], 3)
        self.assertFalse(group["symbol_center_consensus_confirmed"])
        self.assertFalse(group["symbol_center_consensus_applied"])
        self.assertEqual(group["visible_line_type_number"], 9)

    def test_single_center_without_an_arrow_uses_nearest_type_as_fallback(self):
        out = self.resolve([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0), (7, 4.0))),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_line_type_number"], 3)
        self.assertFalse(group["symbol_center_consensus_confirmed"])
        self.assertFalse(group["symbol_center_consensus_applied"])
        self.assertEqual(group["visible_line_type_number"], 3)
        self.assertEqual(out["bindings"][0]["state"], "bound")
        self.assertEqual(out["visible"], [3])

    def test_center_can_use_a_ranked_type_beyond_arrow_residual_guard(self):
        target = self.center("s1:0", (100, 100), near=None, dist=0.2,
                             ranked=((3, 25.0),))
        target["nearest_owned_op"] = {
            "op_index": 9, "owner": 3, "distance": 25.0,
            "run_id": "center-run",
        }
        out = self.resolve([target])
        self.assertEqual(out["visible"], [3])
        self.assertEqual(out["bindings"][0]["state"], "bound")
        self.assertEqual(out["bindings"][0]["line_type_number"], 3)
        self.assertEqual(out["groups"][0]["engine_runs"], ["center-run"])

    def test_different_symbol_indexes_cannot_manufacture_consensus(self):
        entry = entry_of([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0),)),
            self.center("s2:0", (200, 200), near=3, dist=1.0,
                        ranked=((3, 1.0),)),
            row("s1:0", 0, (101, 101), near=9, dist=0.1,
                ranked=((9, 0.1),)),
        ])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS,
                              {1: 0, 2: 0})
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_symbol_indexes"], [1, 2])
        self.assertIsNone(group["symbol_center_line_type_number"])
        self.assertEqual(group["visible_line_type_number"], 9)

    def test_center_without_a_current_symbol_owner_fails_closed(self):
        entry = entry_of([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0),)),
        ])
        out = regroup.resolve(entry, [[0, 0, 1000, 1000]], self.ITEMS, {})
        self.assertEqual(out["visible"], [])
        self.assertEqual(out["groups"], [])
        self.assertEqual(out["bindings"], [])

    def test_two_agreeing_real_arrows_are_not_overridden(self):
        """中心只修正一个孤立误箭头，不能普遍压过多条一致的真实引线。"""
        out = self.resolve([
            self.center("s1:0", (100, 100), near=3, dist=1.0,
                        ranked=((3, 1.0), (9, 2.0))),
            self.center("s1:1", (200, 200), near=3, dist=1.0,
                        ranked=((3, 1.0), (9, 2.0))),
            row("s1:0", 0, (101, 101), near=9, dist=0.1,
                ranked=((9, 0.1), (3, 2.0))),
            row("s1:1", 0, (201, 201), near=9, dist=0.2,
                ranked=((9, 0.2), (3, 2.0))),
        ])
        group = out["groups"][0]
        self.assertEqual(group["symbol_center_line_type_number"], 3)
        self.assertFalse(group["symbol_center_consensus_applied"])
        self.assertEqual(group["symbol_center_contribution"],
                         "blocked_by_arrows")
        self.assertEqual(group["visible_line_type_number"], 9)

    def test_legacy_arrow_rows_without_anchor_kind_are_unchanged(self):
        out = self.resolve([
            row("s1:0", 0, (100, 100), near=5, dist=3.0),
            row("s1:1", 0, (200, 200), near=7, dist=1.0),
        ])
        group = out["groups"][0]
        self.assertNotIn("symbol_center_line_type_number", group)
        self.assertEqual(group["votes_in_plan"], {"5": 1, "7": 1})
        self.assertEqual(group["visible_line_type_number"], 7)


class LegendTemplateBindingTests(unittest.TestCase):
    """明确的 legend 线型模板优先于启发式箭头，但冲突时必须 fail closed。"""

    ITEMS = [
        {"text": "8' CHAIN LINK FENCE", "box_2d": [10, 10, 20, 90]},
        {"text": "DOUBLE SWING GATE", "box_2d": [30, 10, 40, 90]},
    ]

    def test_unique_template_overrides_arrow_majority_and_adds_matched_runs(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=9, dist=0.10),
            row("0", 1, (110, 110), near=9, dist=0.20),
            legend_row(0, (120, 120), tips=((120, 120),), near=3,
                       dist=0.05, matched_runs=("legend-match-a",
                                                "legend-match-b")),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        group = out["groups"][0]
        self.assertEqual(group["votes_in_plan"], {"3": 1, "9": 2})
        self.assertEqual(group["legend_votes_in_plan"], {"3": 1})
        self.assertEqual(group["legend_line_type_number"], 3)
        self.assertFalse(group["legend_conflict"])
        self.assertEqual(group["visible_line_type_number"], 3)
        self.assertEqual(group["engine_runs"],
                         ["legend-match-a", "legend-match-b"])
        self.assertEqual(out["visible"], [3])

    def test_many_occurrence_tips_from_one_symbol_cast_exactly_one_vote(self):
        entry = entry_of([
            legend_row(0, (900, 900),
                       tips=((900, 900), (100, 100), (200, 200)),
                       near=3, dist=0.05,
                       matched_runs=("run-1", "run-2", "run-3")),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        group = out["groups"][0]
        self.assertTrue(out["bindings"][0]["in_plan"])
        self.assertEqual(group["in_plan_count"], 1)
        self.assertEqual(group["votes_in_plan"], {"3": 1})
        self.assertEqual(group["legend_votes_in_plan"], {"3": 1})

    def test_duplicate_rows_for_one_symbol_are_still_only_one_vote(self):
        entry = entry_of([
            legend_row(0, (100, 100), tips=((100, 100),), near=3,
                       dist=0.20),
            legend_row(0, (200, 200), tips=((200, 200),), near=3,
                       dist=0.10),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        self.assertEqual(out["groups"][0]["votes_in_plan"], {"3": 1})
        self.assertEqual(out["groups"][0]["legend_votes_in_plan"], {"3": 1})

    def test_plan_uses_any_tip_and_falls_back_to_compatibility_tip(self):
        with_tips = entry_of([
            legend_row(0, (900, 900),
                       tips=((900, 900), (100, 100)), near=3, dist=0.05),
        ])
        out = regroup.resolve(with_tips, [[0, 0, 500, 500]], self.ITEMS,
                              {0: 0})
        self.assertTrue(out["bindings"][0]["in_plan"])
        self.assertEqual(out["visible"], [3])

        compatibility = legend_row(0, (100, 100), near=3, dist=0.05)
        compatibility.pop("tips")
        out = regroup.resolve(entry_of([compatibility]), [[0, 0, 500, 500]],
                              self.ITEMS, {0: 0})
        self.assertTrue(out["bindings"][0]["in_plan"])
        self.assertEqual(out["visible"], [3])

    def test_two_template_conclusions_in_one_group_fail_closed(self):
        entry = entry_of([
            legend_row(0, (100, 100), tips=((100, 100),), near=3,
                       dist=0.05, matched_runs=("run-three",)),
            legend_row(1, (200, 200), tips=((200, 200),), near=7,
                       dist=0.06, matched_runs=("run-seven",)),
        ])
        # 两个 symbol 都继承同一行文字，故意制造同组模板冲突。
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS,
                              {0: 0, 1: 0})
        group = out["groups"][0]
        self.assertTrue(group["legend_conflict"])
        self.assertEqual(group["legend_line_type_numbers_in_plan"], [3, 7])
        self.assertIsNone(group["legend_line_type_number"])
        self.assertIsNone(group["visible_line_type_number"])
        self.assertFalse(group["plan_fallback"])
        self.assertEqual(group["engine_runs"], [])
        self.assertEqual(out["visible"], [])
        self.assertTrue(all(row["state"] == "hidden"
                            for row in out["bindings"]))

    def test_gate_wins_over_a_valid_legend_template(self):
        entry = entry_of([
            legend_row(0, (100, 100), tips=((100, 100),), near=3,
                       dist=0.05, matched_runs=("must-not-show",)),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 1})
        group = out["groups"][0]
        self.assertEqual(group["scope"], "gate")
        self.assertIsNone(group["visible_line_type_number"])
        self.assertEqual(group["engine_runs"], [])
        self.assertEqual(out["visible"], [])
        self.assertEqual(out["bindings"][0]["state"], "gate")

    def test_out_of_plan_template_does_not_override_an_inside_arrow(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=9, dist=0.10),
            legend_row(0, (900, 900), tips=((900, 900),), near=3,
                       dist=0.05),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        group = out["groups"][0]
        self.assertEqual(group["legend_votes_in_plan"], {})
        self.assertEqual(group["visible_line_type_number"], 9)
        self.assertEqual(out["visible"], [9])

    def test_residual_template_does_not_override_an_inside_arrow(self):
        entry = entry_of([
            row("0", 0, (100, 100), near=9, dist=0.10),
            legend_row(0, (110, 110), tips=((110, 110),), near=None,
                       dist=0.05, ranked=((3, 0.10),)),
        ])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        group = out["groups"][0]
        self.assertEqual(group["legend_votes_in_plan"], {})
        self.assertEqual(group["visible_line_type_number"], 9)
        self.assertEqual(out["visible"], [9])

    def test_template_uses_the_same_nearest_owned_distance_guard_as_arrows(self):
        template = legend_row(0, (100, 100), tips=((100, 100),), near=None,
                              dist=0.10)
        template["nearest_owned_op"] = {
            "op_index": 1, "distance": 3.15, "owner": 3,
        }
        entry = entry_of([template], precision=0.01)
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS, {0: 0})
        self.assertEqual(out["groups"][0]["legend_votes_in_plan"], {"3": 1})
        self.assertEqual(out["visible"], [3])
        self.assertEqual(out["bindings"][0]["state"], "bound")

    def test_plain_old_cache_keeps_its_original_output_schema(self):
        entry = entry_of([row("0", 0, (100, 100), near=3, dist=0.1)])
        out = regroup.resolve(entry, [[0, 0, 500, 500]], self.ITEMS)
        group = out["groups"][0]
        self.assertEqual(set(group), {
            "group", "text", "scope", "keys", "votes_all",
            "votes_in_plan", "line_type_number_all",
            "visible_line_type_number", "tie", "plan_fallback",
            "in_plan_count", "engine_runs",
        })
        self.assertEqual(group["votes_all"], {"3": 1})
        self.assertEqual(group["visible_line_type_number"], 3)
        self.assertNotIn("legend_conflict", group)
        self.assertEqual(out["visible"], [3])


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
