"""Projection regressions for confirmed Method2 pattern boxes."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run  # noqa: E402


def _recognition(regions, confirmed_count=None):
    if confirmed_count is None:
        confirmed_count = sum(
            region.get("line_type_confirmed") is True for region in regions
        )
    return SimpleNamespace(method2=SimpleNamespace(
        audit=SimpleNamespace(
            repeated_vector_text_family_clustering=SimpleNamespace(
                region_instances=tuple(regions),
                line_type_confirmed_instance_count=confirmed_count,
            )
        )
    ))


def _confirmed_region(**updates):
    region = {
        "region_id": "confirmed",
        "display_label": "P001",
        "group_id": "136",
        "op_indices": [10],
        "bounds": {"minX": 1.0, "minY": 2.0, "maxX": 5.0, "maxY": 4.0},
        "orientation_degrees": 15.0,
        "pattern_source": "pdf_text",
        "literal_text": "8'",
        "recovered": False,
        "line_type_confirmed": True,
        "global_type_id": "global_type_002",
    }
    region.update(updates)
    return region


def _payload(*sources):
    return {"global_line_types": [
        {
            "line_type_number": number,
            "recognition_source": "method2",
            "source_line_type_id": source,
        }
        for number, source in enumerate(sources, 1)
    ]}


class Method2PatternProjectionTests(unittest.TestCase):
    def test_nearest_point_keeps_ir_x_y_axes(self):
        self.assertEqual(
            run._nearest_point_on(
                (5.0, 3.0),
                (((0.0, 0.0), (10.0, 0.0)),),
                lambda x, y: [y, x],
            ),
            [0.0, 5.0],
        )

    def test_confirmed_source_identity_maps_to_fused_number_and_four_corners(self):
        regions = (
            _confirmed_region(),
            {
                "region_id": "unconfirmed",
                "bounds": {"minX": 8.0, "minY": 8.0, "maxX": 9.0, "maxY": 9.0},
                "line_type_confirmed": False,
                "global_type_id": "global_type_002",
            },
        )
        recognition = _recognition(regions)
        payload = {"global_line_types": [
            {
                "line_type_number": 9,
                "recognition_source": "method2",
                "source_line_type_id": "global_type_002",
            },
            {
                "line_type_number": 2,
                "recognition_source": "method1",
                "source_line_type_id": "global_type_002",
            },
        ]}

        projected = run._method2_pattern_instances(
            recognition,
            payload,
            lambda x, y: [x + y, x - y],
        )

        self.assertEqual(set(projected), {9})
        self.assertEqual(len(projected[9]), 1)
        self.assertEqual(projected[9][0]["region_id"], "confirmed")
        # The extrema come from crossed corners.  Mapping only (min,min) and
        # (max,max) would incorrectly produce [-1, 1] on the second axis.
        self.assertEqual(projected[9][0]["bbox"], [3.0, -3.0, 9.0, 3.0])

    def test_missing_method2_is_empty(self):
        self.assertEqual(run._method2_pattern_instances(
            SimpleNamespace(method2=None), {}, lambda x, y: [y, x]
        ), {})

    def test_duplicate_method2_source_id_fails_loudly(self):
        payload = _payload("global_type_002", "global_type_002")
        with self.assertRaisesRegex(ValueError, "duplicate Method2 source id"):
            run._method2_pattern_instances(
                _recognition((), 0), payload, lambda x, y: [y, x]
            )

    def test_confirmed_unmapped_source_fails_loudly(self):
        region = _confirmed_region(global_type_id="not-retained")
        with self.assertRaisesRegex(ValueError, "unmapped source id"):
            run._method2_pattern_instances(
                _recognition((region,)),
                _payload("global_type_002"),
                lambda x, y: [y, x],
            )

    def test_bad_or_degenerate_confirmed_bounds_fail_loudly(self):
        cases = {
            "missing": None,
            "non-finite": {
                "minX": 1.0, "minY": 2.0, "maxX": float("inf"), "maxY": 4.0,
            },
            "zero-width": {
                "minX": 1.0, "minY": 2.0, "maxX": 1.0, "maxY": 4.0,
            },
            "inverted": {
                "minX": 5.0, "minY": 2.0, "maxX": 1.0, "maxY": 4.0,
            },
        }
        for name, bounds in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "invalid bounds|non-finite|maxX|degenerate bounds"
            ):
                run._method2_pattern_instances(
                    _recognition((_confirmed_region(bounds=bounds),)),
                    _payload("global_type_002"),
                    lambda x, y: [y, x],
                )

    def test_projection_must_remain_visible_after_output_rounding(self):
        tiny = {
            "minX": 1.0,
            "minY": 2.0,
            "maxX": 1.001,
            "maxY": 2.001,
        }
        with self.assertRaisesRegex(ValueError, "projection.*degenerate"):
            run._method2_pattern_instances(
                _recognition((_confirmed_region(bounds=tiny),)),
                _payload("global_type_002"),
                lambda x, y: [y, x],
            )

    def test_mapped_count_must_equal_confirmed_audit_count(self):
        with self.assertRaisesRegex(ValueError, "pattern count disagrees"):
            run._method2_pattern_instances(
                _recognition((_confirmed_region(),), confirmed_count=2),
                _payload("global_type_002"),
                lambda x, y: [y, x],
            )


if __name__ == "__main__":
    unittest.main()
