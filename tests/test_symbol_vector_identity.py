"""Compact symbol identity: native text exact, outlined glyph geometry strict.

These tests are pure geometry.  They pin the regression where a hexagon-5 or
circle-C/F template silently discarded the code strokes and matched every
copy of the common outer marker.
"""
import math
import os
import sys
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from core.symbolmatch import (  # noqa: E402
    _closed_contour_signature,
    _compact_line_identity,
    _geom_signature_close,
    _select_symbol_template,
    find_symbol_placements,
    match_template,
)
from core.vecgeom import PRIM_TOL  # noqa: E402


SHAPE_CLASS = ("S", "circle", 0, -1, 1.0, "")
OUTER_CLASS = ("S", "circle", 2, -1, 1.0, "")
LINE_CLASS = ("L", 0, 1.0, "", False)


def shape(src, x, y, bbox, *, cls=SHAPE_CLASS, size=10.0, signature=None):
    return {"src": src, "x": x, "y": y, "c": cls, "s": size,
            "o": None, "bbox": bbox, "g": signature}


def line(src, ax, ay, bx, by):
    return {"src": src, "x": (ax + bx) / 2, "y": (ay + by) / 2,
            "c": LINE_CLASS, "s": math.hypot(bx - ax, by - ay),
            "o": math.atan2(by - ay, bx - ax) % math.pi,
            "segment": (ax, ay, bx, by)}


def text(tid, value, x, y):
    return {"src": None, "tid": tid, "x": x, "y": y,
            "c": ("T", value), "s": 1.0, "o": None,
            "bbox": (x - 1, y - 2, x + 1, y + 2)}


def unit(src, bbox):
    x0, y0, x1, y1 = bbox
    return [src, x0, y0, x1, y1]


def full_unit(src, bbox, *, stroke=0, fill=-1, width=1.0):
    x0, y0, x1, y1 = bbox
    return [src, x0, y0, x1, y1, stroke, fill, width, 0, 0]


def indexes(prims):
    by_class = defaultdict(list)
    grid = defaultdict(list)
    for index, prim in enumerate(prims):
        by_class[prim["c"]].append(index)
        grid[(int(prim["x"] // PRIM_TOL),
              int(prim["y"] // PRIM_TOL))].append(index)
    return by_class, grid


def polygon_geom(points):
    return [["l", *left, *right]
            for left, right in zip(points, (*points[1:], points[0]))]


def transform(points, *, scale=1.0, angle=0.0, tx=0.0, ty=0.0,
              mirror=False):
    c, s = math.cos(angle), math.sin(angle)
    result = []
    for x, y in points:
        if mirror:
            x = -x
        result.append((tx + scale * (c * x - s * y),
                       ty + scale * (s * x + c * y)))
    return result


class TemplateSelectionTests(unittest.TestCase):
    def test_outlined_code_is_required_but_whole_background_path_is_not(self):
        prims = [
            shape(0, 0, 0, (-10, -10, 10, 10)),
            line(1, -3, -2, 3, -2),
            line(2, -3, 2, -3, -2),
            # This tiny visible piece is inside the marker, but its complete
            # PDF source path crosses the page and must be excluded.
            line(3, -2, 0, 2, 0),
            shape(4, 0, 0, (-2, -4, 2, 4),
                  cls=("S", "poly", 0, -1, 1.0, ""), size=3.0),
        ]
        units = [
            unit(0, (-10, -10, 10, 10)),
            unit(1, (-3, -2, 3, -2)),
            unit(2, (-3, -2, -3, 2)),
            unit(3, (-100, 0, 100, 0)),
            unit(4, (-2, -4, 2, 4)),
        ]
        allow, required, marker = _select_symbol_template(
            prims, list(range(len(prims))), units=units,
            content_bounds=(-12, -12, 12, 12))
        self.assertTrue(marker)
        self.assertEqual(required, {1, 2, 4})
        self.assertIn(0, allow)
        self.assertNotIn(3, allow)

    def test_native_text_is_the_only_required_identity(self):
        prims = [shape(0, 0, 0, (-10, -10, 10, 10)),
                 line(1, -3, 0, 3, 0),
                 shape(2, 0, 0, (-2, -4, 2, 4),
                       cls=("S", "poly", 0, -1, 1.0, ""), size=3.0),
                 text(0, "5", 0, 0)]
        units = [unit(0, (-10, -10, 10, 10)),
                 unit(1, (-3, 0, 3, 0)), unit(2, (-2, -4, 2, 4))]
        allow, required, marker = _select_symbol_template(
            prims, [0, 1, 2, 3], units=units,
            content_bounds=(-12, -12, 12, 12))
        self.assertTrue(marker)
        self.assertEqual(required, {3})
        self.assertEqual(set(allow), {0, 3})

    def test_glyph_can_share_one_pdf_path_with_its_outline(self):
        prims = [shape(0, 0, 0, (-10, -10, 10, 10)),
                 line(0, -3, -2, 3, -2), line(0, -3, 2, -3, -2)]
        # The same source also owns a leader outside the outline; the compact
        # primitives are still valid because that source drew the outline.
        units = [unit(0, (-30, -10, 10, 10))]
        allow, required, marker = _select_symbol_template(
            prims, [0, 1, 2], units=units,
            content_bounds=(-12, -12, 12, 12))
        self.assertTrue(marker)
        self.assertEqual(required, {1, 2})
        self.assertEqual(set(allow), {0, 1, 2})

    def test_wider_pad_prefers_complete_marker_over_closed_glyph_alone(self):
        # The VLM box hugs the closed inner glyph. Its centre is visible at
        # pad=2 while the slightly offset enclosing marker centre only enters
        # at pad=8. The inner loop must not be mistaken for the whole symbol.
        prims = [
            shape(1, 100, 100, (98, 96, 102, 104)),
            shape(0, 105, 100, (90, 90, 110, 110), cls=OUTER_CLASS,
                  size=20.0),
        ]
        by_class, grid = indexes(prims)
        data = {
            "w": 1000.0, "h": 1000.0,
            "units": [full_unit(0, (90, 90, 110, 110), stroke=2),
                      full_unit(1, (98, 96, 102, 104))],
            "texts": [], "geom": [[], []], "dashes": ["", ""],
        }
        empty_match = {
            "groups": [], "required_hits": [], "matched": set(),
            "count": 0, "rescued": 0, "template_prims": 2,
            "period": False, "single": False,
        }
        with patch("core.symbolmatch._extract_page", return_value=data), \
                patch("core.symbolmatch._get_prims",
                      return_value=(prims, by_class, grid)), \
                patch("core.symbolmatch.match_template",
                      return_value=empty_match) as mocked:
            find_symbol_placements("synthetic.pdf", 0,
                                   [96, 98, 104, 102])
        first = mocked.call_args_list[0]
        self.assertEqual(first.kwargs["prim_allow"], {0, 1})
        self.assertEqual(first.kwargs["prim_required"], {0})

    def test_small_fill_around_text_does_not_preempt_real_outer_marker(self):
        prims = [
            text(0, "5", 100, 100),
            shape(1, 100, 100, (97, 97, 103, 103), size=6.0),
            shape(0, 105, 100, (90, 90, 110, 110), cls=OUTER_CLASS,
                  size=20.0),
        ]
        by_class, grid = indexes(prims)
        data = {
            "w": 1000.0, "h": 1000.0,
            "units": [full_unit(0, (90, 90, 110, 110), stroke=2),
                      full_unit(1, (97, 97, 103, 103))],
            "texts": [{"id": 0, "text": "5", "x0": 99, "y0": 98,
                       "x1": 101, "y1": 102}],
            "geom": [[], []], "dashes": ["", ""],
        }
        empty_match = {
            "groups": [], "required_hits": [], "matched": set(),
            "count": 0, "rescued": 0, "template_prims": 3,
            "period": False, "single": False,
        }
        with patch("core.symbolmatch._extract_page", return_value=data), \
                patch("core.symbolmatch._get_prims",
                      return_value=(prims, by_class, grid)), \
                patch("core.symbolmatch.match_template",
                      return_value=empty_match) as mocked:
            find_symbol_placements("synthetic.pdf", 0,
                                   [98, 99, 102, 101])
        first = mocked.call_args_list[0]
        self.assertEqual(first.kwargs["prim_allow"], {0, 2})
        self.assertEqual(first.kwargs["prim_required"], {0})


class TemplateMatchTests(unittest.TestCase):
    def test_vector_strokes_distinguish_same_outer_shape(self):
        # Template F at x=0, another F at x=100, and a different glyph at
        # x=200 whose second stroke has the wrong orientation.
        prims = [
            shape(0, 0, 0, (-10, -10, 10, 10)),
            line(1, -3, -3, -3, 3), line(2, -3, -3, 3, -3),
            line(3, -3, 0, 1, 0),
            shape(4, 100, 0, (90, -10, 110, 10)),
            line(5, 97, -3, 97, 3), line(6, 97, -3, 103, -3),
            line(7, 97, 0, 101, 0),
            shape(8, 200, 0, (190, -10, 210, 10)),
            line(9, 197, -3, 197, 3), line(10, 197, -3, 197, 3),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(), prim_allow={0, 1, 2, 3},
            prim_required={1, 2, 3})
        self.assertEqual(result["count"], 2)
        self.assertEqual({frozenset(group) for group in result["groups"]},
                         {frozenset({0, 1, 2, 3}),
                          frozenset({4, 5, 6, 7})})
        self.assertEqual(len(result["required_hits"]), 2)
        for group, required_hits in zip(result["groups"],
                                        result["required_hits"]):
            self.assertEqual(len(required_hits), 3)
            self.assertTrue(required_hits <= group)
            self.assertTrue(all(prims[index]["c"][0] == "L"
                                for index in required_hits))

    def test_native_text_matches_exact_content(self):
        prims = [shape(0, 0, 0, (-10, -10, 10, 10)), text(0, "5", 0, 0),
                 shape(1, 100, 0, (90, -10, 110, 10)), text(1, "5", 100, 0),
                 shape(2, 200, 0, (190, -10, 210, 10)), text(2, "6", 200, 0)]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(), prim_allow={0, 1})
        self.assertEqual(result["count"], 2)
        matched_text = {next(iter(group - {0, 2, 4}), None)
                        for group in result["groups"]}
        self.assertNotIn(5, matched_text)

    def test_optional_primitives_are_one_to_one(self):
        # Three candidate shapes cannot satisfy a four-shape template by
        # reusing the same nearby primitive twice.
        prims = [
            shape(0, 0, 0, (-1, -1, 1, 1)),
            shape(1, 2.1, 0, (1.1, -1, 3.1, 1)),
            shape(2, 0, 10, (-1, 9, 1, 11)),
            shape(3, 10, 0, (9, -1, 11, 1)),
            shape(4, 100, 0, (99, -1, 101, 1)),
            shape(5, 100, 10, (99, 9, 101, 11)),
            shape(6, 110, 0, (109, -1, 111, 1)),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(),
            prim_allow={0, 1, 2, 3})
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["groups"], [{0, 1, 2, 3}])

    def test_required_assignment_uses_augmenting_path(self):
        # A nearest-first greedy choice consumes x=105.4 for expected x=105
        # and strands expected x=106. A valid crossed assignment exists.
        prims = [
            shape(0, 0, 0, (-1, -1, 1, 1)),
            shape(1, 0, 20, (-1, 19, 1, 21)),
            line(2, 4.5, 0, 5.5, 0),
            line(3, 5.5, 0, 6.5, 0),
            shape(4, 100, 0, (99, -1, 101, 1)),
            shape(5, 100, 20, (99, 19, 101, 21)),
            line(6, 103.5, 0, 104.5, 0),
            line(7, 104.9, 0, 105.9, 0),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(),
            prim_allow={0, 1, 2, 3}, prim_required={2, 3})
        self.assertEqual(result["count"], 2)

    def test_same_center_vector_code_can_rotate(self):
        prims = [
            shape(0, 0, 0, (-10, -10, 10, 10)),
            line(1, -3, 0, 3, 0),
            shape(2, 100, 0, (90, -10, 110, 10)),
            line(3, 100, -3, 100, 3),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(),
            prim_allow={0, 1}, prim_required={1})
        self.assertEqual(result["count"], 2)

    def test_repeated_native_text_keeps_multiplicity(self):
        prims = [
            shape(0, 0, 0, (-10, -10, 10, 10)),
            text(0, "1", -0.5, 0), text(1, "1", 0.5, 0),
            shape(1, 100, 0, (90, -10, 110, 10)),
            text(2, "1", 100, 0),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(),
            prim_allow={0, 1, 2}, prim_required={1, 2})
        self.assertEqual(result["count"], 1)


class ContourIdentityTests(unittest.TestCase):
    POINTS = ((0.0, 0.0), (5.0, 0.0), (5.0, 2.0),
              (2.0, 1.0), (0.0, 4.0))

    def test_contour_is_pose_start_and_direction_invariant(self):
        expected = _closed_contour_signature(polygon_geom(self.POINTS))
        moved = transform(self.POINTS[2:] + self.POINTS[:2], scale=2.7,
                          angle=0.73, tx=120, ty=-31, mirror=True)
        actual = _closed_contour_signature(polygon_geom(tuple(reversed(moved))))
        self.assertTrue(_geom_signature_close(expected, actual))

    def test_different_closed_vector_is_rejected(self):
        expected = _closed_contour_signature(polygon_geom(self.POINTS))
        different = ((0.0, 0.0), (5.0, 0.0), (5.0, 4.0),
                     (2.0, 3.8), (0.0, 4.0))
        actual = _closed_contour_signature(polygon_geom(different))
        self.assertFalse(_geom_signature_close(expected, actual))

    def test_single_required_closed_primitive_checks_contour(self):
        wanted = _closed_contour_signature(polygon_geom(self.POINTS))
        other = _closed_contour_signature(polygon_geom(
            ((0, 0), (5, 0), (5, 4), (2, 3.8), (0, 4))))
        prims = [
            shape(0, 0, 0, (-3, -3, 3, 3), signature=wanted),
            shape(1, 100, 0, (97, -3, 103, 3), signature=wanted),
            shape(2, 200, 0, (197, -3, 203, 3), signature=other),
        ]
        by_class, grid = indexes(prims)
        result = match_template(
            prims, by_class, grid, set(), set(), prim_allow={0},
            prim_required={0})
        self.assertEqual(result["count"], 2)

    def test_compact_extra_stroke_rejects_f_as_e_subset(self):
        prims = [
            line(0, 0, 0, 0, 8), line(1, 0, 0, 5, 0),
            line(2, 0, 4, 3, 4),
            line(3, 100, 0, 100, 8), line(4, 100, 0, 105, 0),
            line(5, 100, 4, 103, 4), line(6, 100, 8, 105, 8),
            # A long background line crosses the marker but is not compact.
            line(7, 80, 4, 120, 4),
        ]
        by_class, _grid = indexes(prims)
        classes = {LINE_CLASS}
        wanted = _compact_line_identity(
            prims, by_class, {0, 1, 2}, classes)
        candidate = _compact_line_identity(
            prims, by_class, {3, 4, 5}, classes)
        self.assertEqual(wanted[LINE_CLASS], 3)
        self.assertEqual(candidate[LINE_CLASS], 4)

    def test_independent_q_tail_is_rejected_but_older_background_is_not(self):
        # Three markers: template O, Q (same closed O plus a later compact
        # tail), and a true O crossed by an earlier background stub. The
        # positive contour match sees all three O loops; the conservative
        # connected-source negative check must keep only the true O.
        centers = (100.0, 300.0, 500.0)
        prims = []
        units = []
        geom = []

        def add_shape(src, cx, bbox, cls, points, stroke):
            prims.append(shape(src, cx, 100, bbox, cls=cls,
                               size=max(bbox[2] - bbox[0],
                                        bbox[3] - bbox[1])))
            units.append(full_unit(src, bbox, stroke=stroke))
            geom.append(polygon_geom(points))

        # Template: outline source 0, identity source 1.
        add_shape(0, centers[0], (90, 90, 110, 110), OUTER_CLASS,
                  ((90, 90), (110, 90), (110, 110), (90, 110)), 2)
        add_shape(1, centers[0], (96, 96, 104, 104), SHAPE_CLASS,
                  ((96, 96), (104, 96), (104, 104), (96, 104)), 0)
        # Q: outline 2, same O loop 3, independent later tail 4.
        add_shape(2, centers[1], (290, 90, 310, 110), OUTER_CLASS,
                  ((290, 90), (310, 90), (310, 110), (290, 110)), 2)
        add_shape(3, centers[1], (296, 96, 304, 104), SHAPE_CLASS,
                  ((296, 96), (304, 96), (304, 104), (296, 104)), 0)
        prims.append(line(4, 304, 102, 307, 105))
        units.append(full_unit(4, (304, 102, 307, 105)))
        geom.append([["l", 304, 102, 307, 105]])
        # True O: a compact background line was drawn *before* outline 6.
        # It touches the glyph but must not become symbol identity.
        prims.append(line(5, 504, 102, 507, 105))
        units.append(full_unit(5, (504, 102, 507, 105)))
        geom.append([["l", 504, 102, 507, 105]])
        add_shape(6, centers[2], (490, 90, 510, 110), OUTER_CLASS,
                  ((490, 90), (510, 90), (510, 110), (490, 110)), 2)
        add_shape(7, centers[2], (496, 96, 504, 104), SHAPE_CLASS,
                  ((496, 96), (504, 96), (504, 104), (496, 104)), 0)

        by_class, grid = indexes(prims)
        data = {
            "w": 1000.0, "h": 1000.0, "units": units, "texts": [],
            "geom": geom, "dashes": [""] * len(units),
        }
        with patch("core.symbolmatch._extract_page", return_value=data), \
                patch("core.symbolmatch._get_prims",
                      return_value=(prims, by_class, grid)):
            result = find_symbol_placements(
                "synthetic.pdf", 0, [90, 90, 110, 110])
        self.assertEqual(result.get("error"), None)
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["placements"], [[90, 490, 110, 510]])


class VersionTests(unittest.TestCase):
    def test_vector_identity_bumps_placement_cache(self):
        from steps.versions import PLACEMENT_VERSION
        self.assertGreaterEqual(PLACEMENT_VERSION, 5)


if __name__ == "__main__":
    unittest.main()
