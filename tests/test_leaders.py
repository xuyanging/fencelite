"""放置锚引线的几何判据回归 —— 零模型调用，PDF 现场合成.

每个用例自己画一页 PDF（标记外圈 + 引线 + 填充箭头 + 干扰物），所以判据是被
真正端到端验证的：core.vecgeom 的抽取、坐标帧换算、分类闸全都在链路里。

要钉住的是「不能误连」这一侧 —— 平面图上标记常压在围栏线上、周围几十条
hatch 短刺，判据一松就会到处连线：
  * hatch 短刺不能当引线（够不到 MIN_REACH）；
  * 气泡自己的扫描线填充条不能当箭头（最小边 <= SLIVER_PX）；
  * 没有箭头时默认不发布（require_head）。
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import fitz

from steps.leaders import marker_leaders

PW, PH = 600.0, 400.0
CX, CY, R = 100.0, 200.0, 9.0          # 标记外圈
TIPX = 200.0                            # 引线远端


def _box(cx=CX, cy=CY, r=R):
    """标记外圈 -> 页面帧 0-1000 的 box_2d。"""
    return [(cy - r) / PH * 1000, (cx - r) / PW * 1000,
            (cy + r) / PH * 1000, (cx + r) / PW * 1000]


class _Page:
    """一页合成 PDF，用完即删。"""

    def __init__(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.tmp.close()
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=PW, height=PH)

    def marker(self):
        self.page.draw_circle((CX, CY), R, color=(0, 0, 0), width=0.7)
        return self

    def leader(self, tipx=TIPX):
        self.page.draw_line((CX + R, CY), (tipx, CY), color=(0, 0, 0), width=0.7)
        return self

    def head(self, tipx=TIPX, size=4.0):
        self.page.draw_polyline(
            [(tipx, CY), (tipx - size, CY - size / 2),
             (tipx - size, CY + size / 2), (tipx, CY)],
            color=(0, 0, 0), fill=(0, 0, 0), width=0.3, closePath=True)
        return self

    def scanline_fill(self, n=31):
        """气泡的扫描线填充：n 条极薄的填充条，必须不能被当成箭头。"""
        for i in range(n):
            y = CY - R + i * (2 * R / n)
            self.page.draw_rect(fitz.Rect(CX - R, y, CX + R, y + 0.3),
                                color=None, fill=(1, 1, 1), width=0)
        return self

    def hatch(self, n=20):
        """围栏 hatch 短刺：很短，够不到 MIN_REACH。"""
        for i in range(n):
            x = CX + R + 2 + i * 3
            self.page.draw_line((x, CY + 6), (x + 1.2, CY + 7.2),
                                color=(0, 0, 0), width=0.24)
        return self

    def fence_line(self):
        self.page.draw_line((0, CY + 20), (PW, CY + 20), color=(0, 0, 0), width=1.4)
        return self

    def save(self):
        self.doc.save(self.tmp.name)
        self.doc.close()
        return self.tmp.name

    def cleanup(self):
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass


class LeaderGeometryTests(unittest.TestCase):
    def setUp(self):
        self.pages = []

    def tearDown(self):
        for p in self.pages:
            p.cleanup()

    def _run(self, page, **kw):
        self.pages.append(page)
        path = page.save()
        return marker_leaders(path, 0, [("s0:0", _box())], **kw)

    # ── 正例 ──────────────────────────────────────────────────────────────
    def test_finds_leader_and_arrowhead(self):
        got = self._run(_Page().marker().leader().head())
        self.assertIn("s0:0", got, "标记 + 引线 + 箭头应当被找到")
        ent = got["s0:0"]
        self.assertEqual(ent["targets"][0]["terminal_kind"], "arrowhead")
        self.assertEqual(ent["confidence"], "high")
        self.assertTrue(ent["leader_strokes"], "必须给出引线折线")
        self.assertTrue(ent["arrow_strokes"], "必须给出箭头笔画")

    def test_tip_lands_at_the_far_end(self):
        got = self._run(_Page().marker().leader().head())
        tip = got["s0:0"]["targets"][0]["tip"]          # [y, x] 0-1000
        self.assertAlmostEqual(tip[1], TIPX / PW * 1000, delta=12,
                               msg="末端 x 应当在引线远端")
        self.assertAlmostEqual(tip[0], CY / PH * 1000, delta=12)

    def test_target_box_is_on_the_arrowhead(self):
        got = self._run(_Page().marker().leader().head())
        box = got["s0:0"]["targets"][0]["box_2d"]
        self.assertEqual(len(box), 4)
        self.assertLess(box[0], box[2])                 # 正面积
        self.assertLess(box[1], box[3])
        cx = (box[1] + box[3]) / 2
        self.assertAlmostEqual(cx, TIPX / PW * 1000, delta=15)

    def test_survives_the_noisy_neighbourhood(self):
        # 实测环境：标记压在围栏线上，周围 hatch + 气泡扫描线填充。
        got = self._run(_Page().marker().scanline_fill().hatch()
                        .fence_line().leader().head())
        self.assertIn("s0:0", got)
        self.assertEqual(got["s0:0"]["targets"][0]["terminal_kind"], "arrowhead")

    # ── 反例 ──────────────────────────────────────────────────────────────
    def test_no_arrowhead_is_not_published_by_default(self):
        got = self._run(_Page().marker().leader())
        self.assertEqual(got, {}, "没有箭头时默认不发布")

    def test_no_arrowhead_can_be_opted_into(self):
        got = self._run(_Page().marker().leader(), require_head=False)
        self.assertIn("s0:0", got)
        self.assertEqual(got["s0:0"]["targets"][0]["terminal_kind"], "bare-end")
        self.assertEqual(got["s0:0"]["confidence"], "medium")

    def test_hatch_ticks_are_not_leaders(self):
        got = self._run(_Page().marker().hatch().head())
        self.assertEqual(got, {}, "hatch 短刺够不到 MIN_REACH，不能当引线")

    def test_scanline_strips_are_not_arrowheads(self):
        # 有引线、没有真箭头，只有气泡的扫描线填充条 -> 不发布。
        got = self._run(_Page().marker().scanline_fill().leader())
        self.assertEqual(got, {}, "薄填充条不能当箭头")

    def test_bare_marker_yields_nothing(self):
        got = self._run(_Page().marker())
        self.assertEqual(got, {})

    def test_fence_line_alone_is_not_a_leader(self):
        # 围栏长线离标记中心 20pt，根端够不上外圈，不能被当成引线。
        got = self._run(_Page().marker().fence_line().head())
        self.assertEqual(got, {})

    # ── 契约 ──────────────────────────────────────────────────────────────
    def test_empty_anchors_is_a_no_op(self):
        p = _Page().marker().leader().head()
        self.pages.append(p)
        self.assertEqual(marker_leaders(p.save(), 0, []), {})

    def test_malformed_box_is_skipped(self):
        p = _Page().marker().leader().head()
        self.pages.append(p)
        path = p.save()
        self.assertEqual(marker_leaders(path, 0, [("k", [1, 2])]), {})
        self.assertEqual(marker_leaders(path, 0, [("k", None)]), {})

    def test_coordinates_are_in_the_0_1000_frame(self):
        got = self._run(_Page().marker().leader().head())
        for pts in got["s0:0"]["leader_strokes"] + got["s0:0"]["arrow_strokes"]:
            for y, x in pts:
                self.assertGreaterEqual(y, 0.0)
                self.assertLessEqual(y, 1000.0)
                self.assertGreaterEqual(x, 0.0)
                self.assertLessEqual(x, 1000.0)


class ArrowsWiringTests(unittest.TestCase):
    def test_arrows_version_was_bumped(self):
        # 发布结果集变了就必须 bump，否则盘上的旧 arrows.json 会被当作当期。
        from steps.arrows import ARROWS_VERSION
        self.assertGreaterEqual(ARROWS_VERSION, 17)

    def test_diagnostic_early_returns_keep_the_tuple_contract(self):
        from steps import arrows

        self.assertEqual(
            arrows.find_page_arrows(
                "unused", 0, [], return_diagnostics=True),
            ({}, {}),
        )
        item = {"text": "FENCE", "box_2d": [100, 100, 120, 160]}
        with patch.object(arrows, "PLAN_GATE", True):
            self.assertEqual(
                arrows.find_page_arrows(
                    "unused", 0, [item], plan_regions=[],
                    return_diagnostics=True),
                ({}, {}),
            )
            self.assertEqual(
                arrows.find_page_arrows(
                    "unused", 0, [item],
                    plan_regions=[[700, 700, 800, 800]],
                    return_diagnostics=True),
                ({}, {}),
            )

        # Existing callers that do not request diagnostics keep the old shape.
        self.assertEqual(arrows.find_page_arrows("unused", 0, []), {})

    def test_arrow_signature_changes_when_label_changes(self):
        # Spatial/bare fallback eligibility depends on label, so two otherwise
        # identical anchors with different labels must never share a cache.
        from steps.arrows import arrows_signature

        callout = [{"text": "FENCE", "box_2d": [1, 2, 3, 4],
                    "label": "callout"}]
        title = [{**callout[0], "label": "view title"}]
        self.assertNotEqual(arrows_signature(callout, "rev"),
                            arrows_signature(title, "rev"))

    def test_arrow_signature_signs_effective_plan_gate_geometry(self):
        from steps import arrows

        items = [{"text": "FENCE", "box_2d": [1, 2, 3, 4],
                  "label": "callout"}]
        first = [[0, 0, 100, 100], [200, 200, 300, 300]]
        reordered = list(reversed(first))
        changed = [[0, 0, 101, 100], [200, 200, 300, 300]]
        with patch.object(arrows, "PLAN_GATE", False):
            self.assertEqual(
                arrows.arrows_signature(items, "rev", plan_regions=first),
                arrows.arrows_signature(items, "rev", plan_regions=changed))
        with patch.object(arrows, "PLAN_GATE", True):
            with self.assertRaises(TypeError):
                arrows.arrows_signature(items, "rev")
            self.assertEqual(
                arrows.arrows_signature(items, "rev", plan_regions=first),
                arrows.arrows_signature(
                    items, "rev", plan_regions=reordered))
            self.assertNotEqual(
                arrows.arrows_signature(items, "rev", plan_regions=first),
                arrows.arrows_signature(items, "rev", plan_regions=changed))

    def test_placement_and_text_fallbacks_keep_separate_key_spaces(self):
        # str 键仍只走 marker 外圈兜底；int 文字键走文字框端点图。
        import inspect

        from steps import arrows
        src = inspect.getsource(arrows.find_page_arrows)
        self.assertIn("isinstance(key, str)", src)
        self.assertIn("marker_leaders", src)
        self.assertIn("isinstance(key, int)", src)
        self.assertIn("text_box_leaders", src)
        self.assertIn("allow_bare_keys", src)

    def test_supplement_merge_never_removes_old_geometry(self):
        from steps.arrows import _merge_arrow_entry

        old = {
            "leader_strokes": [[[1, 1], [2, 2]]],
            "arrow_strokes": [[[2, 2], [2, 3]]],
            "targets": [{"tip": [2, 2], "box_2d": [1, 1, 3, 3],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "old",
        }
        extra = {
            "leader_strokes": [old["leader_strokes"][0],
                               [[2, 2], [4, 4]]],
            "arrow_strokes": [old["arrow_strokes"][0],
                              [[4, 4], [4, 5]]],
            "targets": [old["targets"][0],
                        {"tip": [4, 4], "box_2d": [3, 3, 5, 5],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "supplement",
        }
        got = _merge_arrow_entry(old, extra)
        self.assertEqual(got["leader_strokes"][0], old["leader_strokes"][0])
        self.assertEqual(got["arrow_strokes"][0], old["arrow_strokes"][0])
        self.assertGreaterEqual(len(got["leader_strokes"]),
                                len(old["leader_strokes"]))
        self.assertGreaterEqual(len(got["arrow_strokes"]),
                                len(old["arrow_strokes"]))
        self.assertGreaterEqual(len(got["targets"]), len(old["targets"]))

    def test_disconnected_supplement_is_not_borrowed(self):
        from steps.arrows import _merge_arrow_entry

        old = {
            "leader_strokes": [[[0, 0], [1, 1]]],
            "arrow_strokes": [],
            "targets": [{"tip": [1, 1], "box_2d": [0, 0, 2, 2],
                         "terminal_kind": "free-end"}],
            "confidence": "medium", "note": "old",
        }
        unrelated = {
            "leader_strokes": [[[10, 10], [11, 11]]],
            "arrow_strokes": [[[11, 11], [11, 12]]],
            "targets": [{"tip": [11, 11], "box_2d": [10, 10, 12, 12],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "other callout",
        }
        self.assertEqual(_merge_arrow_entry(old, unrelated), old)

    def test_same_target_repaint_does_not_thicken_existing_highlight(self):
        from steps.arrows import _merge_arrow_entry

        old = {
            "leader_strokes": [[[0, 0], [2, 2]]],
            "arrow_strokes": [[[2, 2], [2, 3]]],
            "targets": [{"tip": [2, 2], "box_2d": [1, 1, 3, 3],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "authored",
        }
        repaint = {
            "leader_strokes": [[[0.2, 0.1], [2.1, 2.1]]],
            "arrow_strokes": [[[2.1, 2.1], [2.2, 3.1]]],
            "targets": [{"tip": [2.1, 2.1], "box_2d": [1, 1, 3, 3],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "same branch",
        }
        self.assertEqual(_merge_arrow_entry(old, repaint), old)

    def test_bare_trace_through_existing_arrowhead_is_not_a_new_target(self):
        from steps.arrows import _merge_arrow_entry

        headed = {
            "leader_strokes": [[[0, 0], [2, 2]]],
            "arrow_strokes": [[[2, 2], [2, 3]]],
            "targets": [{"tip": [2, 2], "box_2d": [1, 1, 3, 3],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "authored",
        }
        through = {
            "leader_strokes": [[[0, 0], [2.2, 2.1], [5, 5]]],
            "arrow_strokes": [],
            "targets": [{"tip": [5, 5], "box_2d": [4, 4, 6, 6],
                         "terminal_kind": "bare-end"}],
            "confidence": "medium", "note": "continued trace",
        }
        self.assertEqual(_merge_arrow_entry(headed, through), headed)

    def test_continued_free_end_becomes_internal_not_a_second_target(self):
        from steps.arrows import _merge_arrow_entry

        old = {
            "leader_strokes": [[[0, 0], [1, 1]]],
            "arrow_strokes": [],
            "targets": [{"tip": [1, 1], "box_2d": [0, 0, 2, 2],
                         "terminal_kind": "free-end"}],
            "confidence": "medium", "note": "free elbow",
        }
        extension = {
            "leader_strokes": [[[1.2, 1.1], [2, 2]]],
            "arrow_strokes": [[[2, 2], [2, 2.4]]],
            "targets": [{"tip": [2, 2], "box_2d": [1, 1, 3, 3],
                         "terminal_kind": "arrowhead"}],
            "confidence": "high", "note": "real head",
        }
        got = _merge_arrow_entry(old, extension)
        self.assertEqual(len(got["leader_strokes"]), 2)
        self.assertEqual(len(got["targets"]), 1)
        self.assertEqual(got["targets"][0]["tip"], [2, 2])
        self.assertEqual(got["targets"][0]["terminal_kind"], "arrowhead")

    def test_text_endpoint_supplements_require_shared_topology(self):
        import inspect

        from steps import arrows
        src = inspect.getsource(arrows.find_page_arrows)
        self.assertIn("_merge_arrow_entry(out.get(key), entry)", src)
        merge_src = inspect.getsource(arrows._merge_arrow_entry)
        self.assertIn("_supplement_touches_current", merge_src)

    def test_frontend_target_mask_is_tip_centered_and_has_minimum_side(self):
        src = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
        self.assertIn("const targetDisplayBox=t=>", src)
        self.assertIn("Math.min(20,Math.max(12", src)
        self.assertIn("const box=targetDisplayBox(t)", src)

    def test_sidecar_enforces_global_and_row_level_leader_ownership(self):
        import inspect

        src = (ROOT / "tools" / "arrow_sidecar" / "sidecar.mjs").read_text(
            encoding="utf-8")
        self.assertIn("scopeCalloutLeadersToBox", src)
        self.assertIn("automaticCallouts.flatMap", src)
        self.assertIn("regularLeadersAt[index].length !== 0", src)
        self.assertIn("anchor_vec_backed", src)
        self.assertIn("resolveWrappedParagraph", src)
        self.assertIn("foreignDecodedTextOwnsRoot", src)
        self.assertIn("ownerText", src)
        self.assertIn("compactText", src)
        self.assertIn("carrier_is_text", src)

        from steps import arrows
        py_src = inspect.getsource(arrows.find_page_arrows)
        self.assertIn('diagnostic.get("source") == "text-only"', py_src)

    def test_arrow_signature_changes_when_vector_evidence_changes(self):
        from steps.arrows import arrows_signature

        weak = [{"text": "FENCE", "box_2d": [1, 2, 3, 4],
                 "label": "callout", "source": "vlm",
                 "vec_backed": False}]
        backed = [{**weak[0], "vec_backed": True}]
        self.assertNotEqual(arrows_signature(weak, "rev"),
                            arrows_signature(backed, "rev"))

    def test_only_weak_member_of_a_verified_duplicate_is_suppressed(self):
        from steps.arrows import suppressed_unverified_duplicates

        items = [
            {"text": "DEBRIS\nFENCE", "label": "callout",
             "source": "vlm", "vec_backed": False},
            {"text": "debris fence", "label": "callout",
             "source": "vlm", "vec_backed": False},
            {"text": "OTHER FENCE", "label": "callout",
             "source": "vlm", "vec_backed": False},
        ]
        diagnostics = {
            "0": {"source": "text-only", "carrier_is_text": False,
                  "has_leader": False},
            "1": {"source": "roi", "carrier_is_text": False,
                  "has_leader": True},
            "2": {"source": "text-only", "carrier_is_text": False,
                  "has_leader": False},
        }
        self.assertEqual(
            suppressed_unverified_duplicates(items, diagnostics), {0})


if __name__ == "__main__":
    unittest.main()
