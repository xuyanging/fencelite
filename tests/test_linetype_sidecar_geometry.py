"""Pure geometry tests for the line-type subprocess protocol.

These deliberately import ``run.py`` without executing ``main``.  They cover
the no-leader symbol-center rules without importing or running the expensive
line-type engine.
"""
from pathlib import Path
import importlib.util
import math
import unittest


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tools" / "linetype_sidecar" / "run.py"
SPEC = importlib.util.spec_from_file_location("linetype_sidecar_run", RUNNER)
sidecar = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sidecar)


class SymbolCenterExclusionTests(unittest.TestCase):
    def test_frame_box_is_projected_then_padded_equally_in_ir(self):
        # Mimic a rotated-page inverse: [frame_y, frame_x] -> (ir_x, ir_y).
        def to_ir(frame_y, frame_x):
            return (100.0 - frame_x, 200.0 - frame_y)

        # The tight box projects to x=60..80/y=170..190, then expands equally
        # by 36 PDF points on both axes (not anisotropically in page frame).
        self.assertEqual(
            sidecar._symbol_exclude_ir_box([10, 20, 30, 40], to_ir),
            (24.0, 134.0, 116.0, 226.0),
        )

    def test_invalid_center_box_fails_closed(self):
        identity = lambda y, x: (x, y)
        for box in (
            None,
            [1, 2, 3],
            [1, 2, 1, 4],
            [1, 2, 3, math.inf],
            [1, 2, 3, "not-a-number"],
        ):
            with self.subTest(box=box):
                self.assertIsNone(
                    sidecar._symbol_exclude_ir_box(box, identity)
                )

    def test_only_complete_ops_inside_neighbourhood_are_excluded(self):
        # Placement [20,20,30,30] becomes [-16,-16,66,66] in this identity IR.
        identity = lambda y, x: (x, y)
        geometry = {
            # A marker-local op: remove it.
            10: [[(10.0, 10.0), (20.0, 20.0), (40.0, 40.0)]],
            # A real long fence crossing the symbol: retain the complete op.
            11: [[(0.0, 25.0), (100.0, 25.0)]],
            # One local subpath plus one external subpath is still one op: retain it.
            12: [[(10.0, 12.0), (20.0, 12.0)],
                 [(20.0, 80.0), (30.0, 80.0)]],
            # Empty/non-path geometry is never classified as symbol ink.
            13: [],
        }
        self.assertEqual(
            sidecar._symbol_excluded_ops(
                geometry, [20, 20, 30, 30], identity
            ),
            {10},
        )


class SymbolCenterRankingTests(unittest.TestCase):
    def test_center_only_equal_distance_prefers_smaller_cluster(self):
        rows = [(6.5, 19, 352), (6.5, 45, 131)]
        center = sorted(
            rows,
            key=lambda row: sidecar._candidate_rank_key(
                row[0], row[1], row[2], True
            ),
        )
        leader = sorted(
            rows,
            key=lambda row: sidecar._candidate_rank_key(
                row[0], row[1], row[2], False
            ),
        )
        self.assertEqual([row[1] for row in center], [45, 19])
        # Existing leader behavior is unchanged: equal distance falls back to id.
        self.assertEqual([row[1] for row in leader], [19, 45])

    def test_center_does_not_let_specificity_beat_a_real_distance_gap(self):
        rows = [(1.0, 19, 352), (1.01, 45, 1)]
        ranked = sorted(
            rows,
            key=lambda row: sidecar._candidate_rank_key(
                row[0], row[1], row[2], True
            ),
        )
        self.assertEqual([row[1] for row in ranked], [19, 45])

    def test_center_nearest_op_tie_prefers_known_specific_cluster(self):
        residual = sidecar._op_rank_key(6.5, 1, None, 0, True)
        broad = sidecar._op_rank_key(6.5, 2, 19, 352, True)
        specific = sidecar._op_rank_key(6.5, 3, 45, 131, True)
        self.assertLess(specific, broad)
        self.assertLess(broad, residual)

    def test_leader_nearest_owned_still_uses_distance_before_identity(self):
        # The helper's non-center key mirrors the old nearest-op behavior:
        # it does not introduce cluster specificity into leader decisions.
        closer_large = sidecar._op_rank_key(1.0, 99, 19, 352, False)
        farther_small = sidecar._op_rank_key(1.01, 1, 45, 1, False)
        self.assertLess(closer_large, farther_small)


if __name__ == "__main__":
    unittest.main()
