"""Regressions for Method2 repeated inline measurement tokens."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from line_type_engine.ir import (  # noqa: E402
    BoundsIR,
    GroupingIR,
    GroupIR,
    PageIR,
    PathOperationIR,
    PathSegmentIR,
    TextOperationIR,
)
from line_type_engine.method2 import text_family  # noqa: E402
from line_type_engine.operation_index import PageOperationIndex  # noqa: E402


def _path(index: int, start: tuple[float, float], end: tuple[float, float]):
    return PathOperationIR(
        operation_id=f"path:{index}",
        paint_order=index,
        ordinal=index,
        bounds=BoundsIR(
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        ),
        segments=(PathSegmentIR("move", start), PathSegmentIR("line", end)),
        stroke=True,
        fill=False,
        stroke_color=(0.0,),
        line_width=0.6,
    )


def _text(index: int, literal: str, bounds: BoundsIR):
    return TextOperationIR(
        operation_id=f"text:{index}",
        paint_order=index,
        ordinal=index,
        span_index=index,
        bounds=bounds,
        literal_text=literal,
        characters=(),
        font_name="F1",
        font_size=2.0,
        direction=(1.0, 0.0),
        render_mode=0,
        color=(0.0,),
        opacity=1.0,
    )


def _page_and_grouping(
    operations: tuple[PathOperationIR | TextOperationIR, ...],
    group_ranges: tuple[tuple[str, int, int], ...],
) -> tuple[PageIR, GroupingIR]:
    page = PageIR(
        page_number=1,
        page_bounds=BoundsIR(0.0, 0.0, 100.0, 100.0),
        rotation_degrees=0,
        operations=operations,
        source_sha256="0" * 64,
    )
    groups = []
    assignments = []
    for group_id, start, end in group_ranges:
        owned = operations[start:end]
        bounds = owned[0].bounds
        for operation in owned[1:]:
            bounds = bounds.union(operation.bounds)
        groups.append(GroupIR(
            group_id,
            tuple(operation.operation_id for operation in owned),
            bounds,
            owned[0].paint_order,
            owned[-1].paint_order,
        ))
        assignments.extend((operation.operation_id, group_id) for operation in owned)
    return page, GroupingIR(page.fingerprint, tuple(groups), tuple(assignments))


def _straight_run(
    start_index: int,
    token_count: int,
    *,
    start_x: float,
    y: float,
    literal: str = "8'",
) -> tuple[PathOperationIR | TextOperationIR, ...]:
    operations: list[PathOperationIR | TextOperationIR] = []
    index = start_index
    x = start_x
    operations.append(_path(index, (x, y), (x + 4.0, y)))
    index += 1
    x += 4.0
    for _ in range(token_count):
        operations.append(_text(index, literal, BoundsIR(x, y - 1.0, x + 2.0, y + 1.0)))
        index += 1
        x += 2.0
        operations.append(_path(index, (x, y), (x + 4.0, y)))
        index += 1
        x += 4.0
    return tuple(operations)


class InlineMeasurementMethod2Tests(unittest.TestCase):
    def test_literal_exception_is_narrow(self) -> None:
        self.assertTrue(text_family._is_inline_feet_token("8'"))
        self.assertTrue(text_family._is_inline_feet_token("8.10\u2032"))
        self.assertFalse(text_family._is_inline_feet_token("8"))
        self.assertFalse(text_family._is_inline_feet_token("8'-0\""))
        self.assertFalse(text_family._is_inline_feet_token("GRID 8'"))

    def test_three_strict_tokens_own_only_internal_carriers(self) -> None:
        operations = _straight_run(0, 3, start_x=4.0, y=50.0)
        page, grouping = _page_and_grouping(
            operations, (("run", 0, len(operations)),)
        )
        recognized = text_family.recognize_repeated_text_pattern_families(
            page, grouping, ()
        )

        self.assertEqual(len(recognized.result.global_types), 1)
        global_type = recognized.result.global_types[0]
        self.assertEqual(global_type.signature_family, "pdf_text_dash_line")
        self.assertEqual(global_type.op_indices, (2, 4))
        instances = [
            item for item in recognized.audit.region_instances
            if item.literal_text == "8'"
        ]
        self.assertEqual(len(instances), 3)
        self.assertTrue(all(item.line_type_confirmed for item in instances))

    def test_entry_requires_three_distinct_authored_paint_events(self) -> None:
        operations = [
            replace(operation, paint_order=paint_order)
            for operation, paint_order in zip(
                _straight_run(0, 3, start_x=4.0, y=50.0),
                (0, 1, 1, 2, 2, 2, 3),
            )
        ]
        page, grouping = _page_and_grouping(
            tuple(operations), (("duplicate-trace", 0, len(operations)),)
        )

        recognized = text_family.recognize_repeated_text_pattern_families(
            page, grouping, ()
        )

        self.assertFalse(recognized.result.global_types)
        self.assertFalse([
            item for item in recognized.audit.region_instances
            if item.literal_text == "8'"
        ])

    def test_token_between_different_carrier_styles_is_not_admitted(self) -> None:
        operations = list(_straight_run(0, 1, start_x=4.0, y=50.0))
        operations[2] = replace(
            operations[2],
            stroke_color=(1.0, 0.0, 0.0),
            line_width=2.0,
        )
        page, grouping = _page_and_grouping(
            tuple(operations), (("mixed", 0, len(operations)),)
        )
        context = text_family._Context.build(page, grouping)

        self.assertIsNone(text_family._native_text_signature_for(
            context,
            operations[1],
            1,
            math.hypot(page.page_bounds.width, page.page_bounds.height),
        ))

    def test_complete_family_counts_distinct_paint_orders_per_group(self) -> None:
        operations = [
            replace(operation, paint_order=paint_order)
            for operation, paint_order in zip(
                _straight_run(0, 3, start_x=4.0, y=50.0),
                (0, 1, 1, 2, 2, 2, 3),
            )
        ]
        page, grouping = _page_and_grouping(
            tuple(operations), (("duplicate-trace", 0, len(operations)),)
        )
        context = text_family._Context.build(page, grouping)
        diagonal = math.hypot(page.page_bounds.width, page.page_bounds.height)
        signatures = tuple(
            signature
            for index in (1, 3, 5)
            if (
                signature := text_family._native_text_signature_for(
                    context, operations[index], index, diagonal
                )
            ) is not None
        )
        self.assertEqual(len(signatures), 3)

        self.assertFalse(
            text_family._complete_text_pattern_families(context, signatures)
        )

    def test_unqualified_group_cannot_ride_a_qualified_literal_family(self) -> None:
        first = _straight_run(0, 3, start_x=4.0, y=60.0)
        second = _straight_run(len(first), 1, start_x=40.0, y=30.0)
        operations = (*first, *second)
        page, grouping = _page_and_grouping(
            operations,
            (("qualified", 0, len(first)), ("coincidental", len(first), len(operations))),
        )
        recognized = text_family.recognize_repeated_text_pattern_families(
            page, grouping, ()
        )

        instances = [
            item for item in recognized.audit.region_instances
            if item.literal_text == "8'"
        ]
        self.assertEqual({item.group_id for item in instances}, {"qualified"})
        self.assertEqual(
            {member.case_id for member in recognized.result.global_types[0].members},
            {"qualified"},
        )

    def test_same_literal_with_different_carrier_styles_stays_separate(self) -> None:
        thin_black = _straight_run(0, 3, start_x=4.0, y=30.0)
        thick_red = tuple(
            replace(operation, stroke_color=(1.0, 0.0, 0.0), line_width=2.0)
            if isinstance(operation, PathOperationIR)
            else operation
            for operation in _straight_run(
                len(thin_black), 3, start_x=4.0, y=70.0
            )
        )
        operations = (*thin_black, *thick_red)
        page, grouping = _page_and_grouping(
            operations,
            (
                ("thin-black", 0, len(thin_black)),
                ("thick-red", len(thin_black), len(operations)),
            ),
        )

        recognized = text_family.recognize_repeated_text_pattern_families(
            page, grouping, ()
        )

        self.assertEqual(len(recognized.result.global_types), 2)
        self.assertEqual(
            {global_type.op_indices for global_type in recognized.result.global_types},
            {(2, 4), (9, 11)},
        )
        self.assertEqual(
            {
                tuple(member.case_id for member in global_type.members)
                for global_type in recognized.result.global_types
            },
            {("thin-black",), ("thick-red",)},
        )

    def test_same_carrier_style_still_merges_across_groups(self) -> None:
        first = _straight_run(0, 3, start_x=4.0, y=30.0)
        second = _straight_run(len(first), 3, start_x=4.0, y=70.0)
        operations = (*first, *second)
        page, grouping = _page_and_grouping(
            operations,
            (
                ("lower", 0, len(first)),
                ("upper", len(first), len(operations)),
            ),
        )

        recognized = text_family.recognize_repeated_text_pattern_families(
            page, grouping, ()
        )

        self.assertEqual(len(recognized.result.global_types), 1)
        self.assertEqual(
            {member.case_id for member in recognized.result.global_types[0].members},
            {"lower", "upper"},
        )

    def test_dense_internal_interval_can_contain_a_disconnected_bend(self) -> None:
        operations = (
            _path(0, (0.0, 50.0), (4.0, 50.0)),
            _text(1, "8'", BoundsIR(4.0, 49.0, 6.0, 51.0)),
            _path(2, (6.0, 50.0), (9.0, 50.0)),
            _path(3, (17.0, 50.0), (20.0, 50.0)),
            _text(4, "8'", BoundsIR(20.0, 49.0, 22.0, 51.0)),
            _path(5, (22.0, 50.0), (26.0, 50.0)),
        )
        page, grouping = _page_and_grouping(
            operations, (("turn", 0, len(operations)),)
        )
        context = text_family._Context(
            page,
            grouping,
            PageOperationIndex.build(page, grouping),
            tuple(range(len(operations))),
            {index: index for index in range(len(operations))},
        )
        diagonal = 100.0 * 2.0 ** 0.5
        signatures = [
            text_family._native_text_signature_for(
                context, operations[index], index, diagonal
            )
            for index in (1, 4)
        ]
        self.assertTrue(all(signature is not None for signature in signatures))
        family = text_family._Family([0, 1], 1.0)
        self.assertEqual(
            text_family._interior_measurement_carriers(
                context,
                family,
                tuple(signature for signature in signatures if signature is not None),
            ),
            {2, 3},
        )


if __name__ == "__main__":
    unittest.main()
