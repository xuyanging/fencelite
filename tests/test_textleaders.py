"""Order-independent text leader recovery on synthetic PDF pages."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import fitz

from steps.textleaders import text_box_leaders

PW, PH = 600.0, 400.0
TEXT_RECT = (260.0, 180.0, 360.0, 220.0)


def _box(rect=TEXT_RECT):
    x0, y0, x1, y1 = rect
    return [y0 / PH * 1000, x0 / PW * 1000,
            y1 / PH * 1000, x1 / PW * 1000]


class _Page:
    def __init__(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        self.tmp.close()
        self.doc = fitz.open()
        self.page = self.doc.new_page(width=PW, height=PH)

    def line(self, a, b, width=0.7):
        self.page.draw_line(a, b, color=(0, 0, 0), width=width)
        return self

    def filled_head(self, tip, previous, size=4.0, stroke=True):
        vx, vy = tip[0] - previous[0], tip[1] - previous[1]
        length = (vx * vx + vy * vy) ** 0.5
        ux, uy = vx / length, vy / length
        nx, ny = -uy, ux
        base = (tip[0] - ux * size, tip[1] - uy * size)
        points = [tip,
                  (base[0] + nx * size / 2, base[1] + ny * size / 2),
                  (base[0] - nx * size / 2, base[1] - ny * size / 2), tip]
        self.page.draw_polyline(points,
                                color=(0, 0, 0) if stroke else None,
                                fill=(0, 0, 0),
                                width=0.7 if stroke else 0,
                                closePath=True)
        return self

    def filled_block(self, rect):
        self.page.draw_rect(fitz.Rect(*rect), color=None, fill=(0, 0, 0),
                            width=0)
        return self

    def open_head(self, tip, previous, size=3.0):
        vx, vy = tip[0] - previous[0], tip[1] - previous[1]
        length = (vx * vx + vy * vy) ** 0.5
        ux, uy = vx / length, vy / length
        nx, ny = -uy, ux
        base = (tip[0] - ux * size, tip[1] - uy * size)
        self.line(tip, (base[0] + nx * size / 2, base[1] + ny * size / 2))
        self.line(tip, (base[0] - nx * size / 2, base[1] - ny * size / 2))
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


class TextLeaderTests(unittest.TestCase):
    def setUp(self):
        self.pages = []

    def tearDown(self):
        for page in self.pages:
            page.cleanup()

    def _run(self, page, anchors=None, **kwargs):
        self.pages.append(page)
        anchors = anchors or [("k", _box(), "callout", "FENCE")]
        return text_box_leaders(page.save(), 0, anchors, **kwargs)

    def test_tracks_shoulder_and_diagonal_across_drawing_units(self):
        page = (_Page().line((256, 200), (220, 200))
                .line((220, 200), (120, 260))
                .filled_head((120, 260), (220, 200)))
        got = self._run(page)
        self.assertIn("k", got)
        self.assertEqual(len(got["k"]["leader_strokes"]), 2)
        self.assertTrue(got["k"]["arrow_strokes"])
        self.assertEqual(got["k"]["targets"][0]["terminal_kind"], "arrowhead")

    def test_uses_text_rectangle_boundary_not_center_radius(self):
        wide = (260.0, 180.0, 500.0, 220.0)
        page = (_Page().line((256, 200), (210, 200))
                .line((210, 200), (100, 230))
                .filled_head((100, 230), (210, 200)))
        got = self._run(page, [("wide", _box(wide), "callout", "FENCE")])
        self.assertIn("wide", got)

    def test_preserves_all_arrow_branches_for_one_callout(self):
        page = (_Page().line((364, 200), (400, 200))
                .line((400, 200), (480, 140))
                .line((400, 200), (480, 260))
                .filled_head((480, 140), (400, 200))
                .filled_head((480, 260), (400, 200)))
        got = self._run(page)
        self.assertIn("k", got)
        self.assertEqual(len(got["k"]["targets"]), 2)
        self.assertEqual(len(got["k"]["leader_strokes"]), 3)

    def test_recognizes_open_v_arrowhead(self):
        page = (_Page().line((256, 200), (180, 200))
                .open_head((180, 200), (256, 200)))
        got = self._run(page)
        self.assertIn("k", got)
        self.assertEqual(len(got["k"]["arrow_strokes"]), 2)
        self.assertEqual(got["k"]["targets"][0]["terminal_kind"], "arrowhead")

    def test_recognizes_fill_only_arrowhead(self):
        page = (_Page().line((256, 200), (180, 200))
                .filled_head((180, 200), (256, 200), stroke=False))
        got = self._run(page)
        self.assertIn("k", got)
        self.assertTrue(got["k"]["arrow_strokes"])

    def test_filled_rectangular_drawing_block_is_not_an_arrowhead(self):
        page = (_Page().line((256, 200), (180, 200))
                .filled_block((176, 196, 180, 204)))
        self.assertEqual(self._run(page), {})

    def test_complete_arrow_beats_a_nearer_bare_line(self):
        page = (_Page().line((256, 194), (210, 194))
                .line((256, 206), (220, 206))
                .line((220, 206), (120, 260))
                .filled_head((120, 260), (220, 206)))
        got = self._run(page, allow_bare_keys={"k"})
        self.assertEqual(got["k"]["targets"][0]["terminal_kind"], "arrowhead")

    def test_bare_leader_is_opt_in(self):
        page = _Page().line((256, 200), (150, 200))
        self.assertEqual(self._run(page), {})
        page = _Page().line((256, 200), (150, 200))
        got = self._run(page, allow_bare_keys={"k"})
        self.assertEqual(got["k"]["targets"][0]["terminal_kind"], "bare-end")
        self.assertFalse(got["k"]["arrow_strokes"])

    def test_short_bare_note_rule_is_rejected(self):
        # At extraction scale this clears the general 15 px path threshold,
        # but is too short to publish without an arrowhead.
        page = _Page().line((256, 200), (252, 200))
        got = self._run(page, allow_bare_keys={"k"})
        self.assertEqual(got, {})

    def test_parallel_grid_is_not_a_bare_leader(self):
        page = (_Page().line((256, 198.8), (150, 198.8), width=0.7)
                .line((256, 200), (150, 200), width=0.7)
                .line((256, 201.2), (150, 201.2), width=0.7))
        got = self._run(page, allow_bare_keys={"k"})
        self.assertEqual(got, {})

    def test_title_underline_is_not_a_bare_leader(self):
        page = _Page().line((275, 224), (345, 224))
        got = self._run(page, allow_bare_keys={"k"})
        self.assertEqual(got, {})

    def test_title_note_and_legend_anchors_are_filtered(self):
        anchors = [
            ("title", _box(), "view title", "FENCE DETAIL"),
            ("note", _box(), "note", "FENCE"),
            ("legend", _box(), "legend entry", "FENCE"),
            ("supp-title", _box(), "vector supplement", "FENCING DETAIL"),
        ]
        page = (_Page().line((256, 200), (180, 200))
                .filled_head((180, 200), (256, 200)))
        self.assertEqual(self._run(page, anchors), {})

    def test_bare_is_never_enabled_for_a_non_callout(self):
        page = _Page().line((256, 200), (150, 200))
        anchors = [("k", _box(), "vector supplement", "FENCE")]
        got = self._run(page, anchors, allow_bare_keys={"k"})
        self.assertEqual(got, {})

    def test_malformed_and_empty_anchors_are_no_ops(self):
        page = _Page().line((256, 200), (180, 200)).filled_head(
            (180, 200), (256, 200))
        self.pages.append(page)
        path = page.save()
        self.assertEqual(text_box_leaders(path, 0, []), {})
        self.assertEqual(text_box_leaders(path, 0, [("bad", None, "callout", "x")]), {})


if __name__ == "__main__":
    unittest.main()
