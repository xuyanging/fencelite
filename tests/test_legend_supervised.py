from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parent.parent
SIDECAR = ROOT / "tools" / "linetype_sidecar"
ENGINE = SIDECAR / "engine"
for path in (SIDECAR, ENGINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from legend_supervised import (  # noqa: E402
    LegendTemplate, associate_template, embedded_cycle_atoms,
    extract_template, normalise_literal,
    pattern_instances_outside_samples)
from line_type_engine.ir import (  # noqa: E402
    BoundsIR, PathOperationIR, PathSegmentIR, TextOperationIR)
from line_type_engine.method1.unknown_pattern_split import (  # noqa: E402
    Atom, distance, mean, polyline_length, principal_frame,
    resample_polyline)


def atom_of(points, *, closed=False):
    samples = resample_polyline(points, 16)
    center = (mean(point[0] for point in samples),
              mean(point[1] for point in samples))
    scale = 2 * max(distance(point, center) for point in samples)
    direction, aspect = principal_frame(samples, center)
    return Atom(0, points, samples, polyline_length(points), center, scale,
                aspect, direction, closed, 0, len(points) - 1,
                "stroke", 0.6, 0, (0.0, 0.0, 0.0))


def path(order, y, *, operation_id=None, x0=0, x1=100):
    return PathOperationIR(
        operation_id or f"path:{order}", order, order,
        BoundsIR(x0, y - 1, x1, y + 1),
        (PathSegmentIR("move", (x0, y)),
         PathSegmentIR("line", (x1, y))),
        True, False, stroke_color=(0.0,), line_width=0.6,
        source_provenance_exact=True)


def enclosing_path(order):
    return PathOperationIR(
        f"container:{order}", order, order, BoundsIR(0, 0, 100, 100),
        (PathSegmentIR("move", (0, 0)),
         PathSegmentIR("line", (100, 0)),
         PathSegmentIR("line", (100, 100)),
         PathSegmentIR("line", (0, 100)),
         PathSegmentIR("close")),
        True, False, stroke_color=(0.0,), line_width=0.6,
        source_provenance_exact=True)


class Page:
    def __init__(self, operations):
        self.operations = tuple(operations)


class LegendExtractionTests(unittest.TestCase):
    def test_native_literal_normalisation_does_not_use_vlm_value(self):
        self.assertEqual(normalise_literal("  8\u2032\n"), "8'")
        self.assertEqual(normalise_literal(" sf "), "SF")

    def test_published_method2_instances_exclude_the_source_swatch(self):
        source = {"literal_text": "SF", "bbox": [10, 20, 12, 30]}
        target = {"literal_text": "SF", "bbox": [100, 200, 102, 210]}
        invalid = {"literal_text": "SF", "bbox": [1, 2, 1, 4]}
        self.assertEqual(pattern_instances_outside_samples(
            [source, target, invalid, {"literal_text": "SF"}],
            [[9, 19, 13, 31]]), [target])

    def test_latest_paint_tranche_discards_crossing_background(self):
        background = path(10, 50)
        sample = path(1000, 50)
        text = TextOperationIR(
            "text:1001", 1001, 0, 0, BoundsIR(40, 48, 50, 52), "8'", (),
            "Test", 8, (1, 0), 0, (0.0,), 1.0)
        page = Page((background, sample, text))
        template = extract_template(
            page,
            {"symbol_index": 3, "text_index": 7,
             "box_2d": [45, 0, 55, 100], "value": "wrong"},
            0, lambda x, y: [y, x])
        self.assertTrue(template.valid, template.reason)
        self.assertEqual(template.path_indices, (1,))
        self.assertEqual(template.text_indices, (2,))
        self.assertEqual(template.literals, ("8'",))
        self.assertEqual(template.discarded_earlier_operations, 1)
        self.assertGreater(template.horizontal_coverage, .95)

    def test_later_enclosing_border_does_not_replace_the_real_swatch(self):
        background = path(10, 50)
        sample = path(1000, 50)
        container = enclosing_path(1100)
        template = extract_template(
            Page((background, sample, container)),
            {"symbol_index": 3, "text_index": 7,
             "box_2d": [45, 20, 55, 80]},
            0, lambda x, y: [y, x])
        self.assertTrue(template.valid, template.reason)
        self.assertEqual(template.path_indices, (1,))
        self.assertEqual(template.paint_order_min, 1000)
        self.assertEqual(template.paint_order_max, 1000)
        self.assertEqual(template.discarded_earlier_operations, 1)

    def test_swatch_wholly_inside_box_counts_as_an_intersection(self):
        template = extract_template(
            Page((path(100, 50, x0=30, x1=50),)),
            {"symbol_index": 3, "text_index": 7,
             "box_2d": [45, 20, 55, 55]},
            0, lambda x, y: [y, x])
        self.assertTrue(template.valid, template.reason)
        self.assertEqual(template.path_indices, (0,))
        self.assertGreater(template.horizontal_coverage, .5)
        self.assertGreater(template.horizontal_span, .5)

    def test_embedded_closed_junction_is_recovered_from_open_carrier(self):
        # carrier -> square loop -> same junction -> carrier
        points = [(0, 0), (5, 0), (5, 2), (7, 2), (7, -2),
                  (3, -2), (3, 2), (5, 2), (5, 0), (10, 0)]
        cycles = embedded_cycle_atoms(atom_of(points))
        self.assertEqual(len(cycles), 1)
        self.assertTrue(cycles[0].closed)
        self.assertGreater(cycles[0].scale, 3)


class LegendAssociationTests(unittest.TestCase):
    def test_actual_pdf_text_maps_to_confirmed_method2_cluster(self):
        operation = path(5, 100)
        template = LegendTemplate(
            0, 2, 4, (10, 10, 20, 40), (0,), (1,), ("8'",),
            5, 6, 0, .8, .9, True, None)
        result = associate_template(
            template,
            pattern_instances={2: [{"literal_text": "8'",
                                    "bbox": [100, 100, 110, 120]}]},
            cluster_ops={2: [0]}, owner={}, operations=(operation,),
            run_of={0: "7"}, sample_boxes=[template.box_2d],
            to_page_frame=lambda x, y: [y, x])
        self.assertEqual(result["status"], "matched")
        self.assertEqual(result["match_kind"], "native_text")
        self.assertEqual(result["matched_line_type_numbers"], [2])
        self.assertEqual(result["matched_runs_by_line_type"], {"2": ["7"]})


if __name__ == "__main__":
    unittest.main()
