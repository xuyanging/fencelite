"""Regressions for conservative Method1 compound-carrier recovery."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
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
)
from line_type_engine.method1.postprocess_carrier import (  # noqa: E402
    recover_compound_carrier_support,
)
from line_type_engine.method1.serializer import (  # noqa: E402
    SerializedGroup,
    validate_group_classification,
)
from line_type_engine.fusion import (  # noqa: E402
    fuse_line_type_recognition_results,
)
from line_type_engine.results import (  # noqa: E402
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    NonLineType,
    RecognitionSummary,
    RecognizedGroup,
)


def _path(index: int, points: tuple[tuple[float, float], ...]) -> PathOperationIR:
    return PathOperationIR(
        operation_id=f"path-{index}",
        paint_order=index,
        ordinal=index,
        bounds=BoundsIR(
            min(point[0] for point in points),
            min(point[1] for point in points),
            max(point[0] for point in points),
            max(point[1] for point in points),
        ),
        segments=(
            PathSegmentIR("move", points[0]),
            *(PathSegmentIR("line", point) for point in points[1:]),
        ),
        stroke=True,
        fill=False,
        line_width=0.0,
        hairline=True,
    )


def _bounds(operations: tuple[PathOperationIR, ...], indices: tuple[int, ...]) -> BoundsIR:
    result = operations[indices[0]].bounds
    for index in indices[1:]:
        result = result.union(operations[index].bounds)
    return result


def _fixture(
    carrier_points: tuple[tuple[tuple[float, float], ...], ...],
    *,
    cross_group: bool = True,
    repeat_support: bool = True,
    second_carrier_points: tuple[
        tuple[tuple[float, float], ...], ...
    ] | None = None,
    simple_motif: bool = False,
    motif_count: int = 16,
) -> tuple[
    PageIR,
    GroupingIR,
    tuple[SerializedGroup, ...],
    LineTypeRecognitionResult,
    tuple[int, ...],
]:
    operations: list[PathOperationIR] = []
    first_motif = tuple(range(motif_count))

    def motif_points(x: float, y: float) -> tuple[tuple[float, float], ...]:
        if simple_motif:
            return ((x, y - 1.0), (x, y + 1.0))
        return (
            (x - 1.0, y - 1.0),
            (x - 1.0, y + 1.0),
            (x + 1.0, y + 1.0),
            (x + 1.0, y - 1.0),
            (x - 1.0, y - 1.0),
        )

    for x in range(0, motif_count * 10, 10):
        operations.append(_path(
            len(operations), motif_points(float(x), 0.0)
        ))
    repeated_carrier_points = (
        tuple(
            tuple((x, y + 100.0) for x, y in points)
            for points in (
                carrier_points
                if second_carrier_points is None
                else second_carrier_points
            )
        )
        if repeat_support
        else ()
    )
    all_carrier_points = (*carrier_points, *repeated_carrier_points)
    carrier_indices = tuple(
        range(len(operations), len(operations) + len(all_carrier_points))
    )
    for points in all_carrier_points:
        operations.append(_path(len(operations), points))
    second_start = len(operations)
    for x in range(0, motif_count * 10, 10):
        operations.append(_path(
            len(operations), motif_points(float(x), 100.0)
        ))
    second_motif = tuple(range(second_start, len(operations)))
    operation_tuple = tuple(operations)
    groups_indices = (first_motif, carrier_indices, second_motif)
    page = PageIR(
        page_number=1,
        page_bounds=BoundsIR(
            -20.0, -20.0, max(220.0, motif_count * 10 + 20.0), 140.0
        ),
        rotation_degrees=0,
        operations=operation_tuple,
        source_sha256="0" * 64,
    )
    group_ids = ("motif-a", "carrier", "motif-b")
    group_irs = tuple(
        GroupIR(
            group_id=group_id,
            operation_ids=tuple(operation_tuple[index].operation_id for index in indices),
            bounds=_bounds(operation_tuple, indices),
            first_paint_order=indices[0],
            last_paint_order=indices[-1],
        )
        for group_id, indices in zip(group_ids, groups_indices)
    )
    grouping = GroupingIR(
        page_fingerprint=page.fingerprint,
        groups=group_irs,
        assignments=tuple(
            (operation_tuple[index].operation_id, group_id)
            for group_id, indices in zip(group_ids, groups_indices)
            for index in indices
        ),
    )
    serialized = tuple(
        SerializedGroup(group_id, "", indices)
        for group_id, indices in zip(group_ids, groups_indices)
    )

    def motif_local(type_id: str, indices: tuple[int, ...]) -> LocalLineType:
        return LocalLineType(
            type_id=type_id,
            display_name="线型1",
            line_type_index=1,
            atom_count=len(indices),
            op_indices=indices,
            model="compound_path_chain",
            shape="非直线",
            shape_detail="复合周期 pattern",
        )

    first_local = motif_local("type_compound_001", first_motif)
    second_local = motif_local("type_compound_001", second_motif)
    recognized_groups = (
        RecognizedGroup(
            "motif-a", len(first_motif), (first_local,), NonLineType(0, ())
        ),
        RecognizedGroup(
            "carrier",
            len(carrier_indices),
            (),
            NonLineType(len(carrier_indices), carrier_indices),
        ),
        RecognizedGroup(
            "motif-b",
            len(second_motif),
            (second_local,) if cross_group else (),
            NonLineType(
                0 if cross_group else len(second_motif),
                () if cross_group else second_motif,
            ),
        ),
    )
    members = (
        GlobalLineTypeMember(
            "motif-a",
            first_local.type_id,
            first_local.display_name,
            first_local.atom_count,
            first_local.shape,
            first_local.model,
            first_local.shape_detail,
        ),
        *(
            (
                GlobalLineTypeMember(
                    "motif-b",
                    second_local.type_id,
                    second_local.display_name,
                    second_local.atom_count,
                    second_local.shape,
                    second_local.model,
                    second_local.shape_detail,
                ),
            )
            if cross_group
            else ()
        ),
    )
    global_ops = (*first_motif, *(second_motif if cross_group else ()))
    result = LineTypeRecognitionResult(
        groups=recognized_groups,
        global_types=(GlobalLineType(
            global_type_id="global-compound",
            signature_family="compound_path_periodic",
            minimum_pair_similarity=0.99,
            op_indices=tuple(global_ops),
            members=members,
        ),),
        summary=RecognitionSummary(
            input_group_count=3,
            processed_group_count=3,
            local_line_type_count=2 if cross_group else 1,
            signed_periodic_type_count=2 if cross_group else 1,
            unsigned_periodic_type_count=0,
            global_type_count=1,
            cross_group_global_type_count=1 if cross_group else 0,
        ),
    )
    return page, grouping, serialized, result, carrier_indices


class CompoundCarrierSupportTests(unittest.TestCase):
    def test_two_segment_carrier_extends_global_without_new_signature_member(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        global_type = output.global_types[0]
        self.assertTrue(set(carriers).issubset(global_type.op_indices))
        self.assertEqual(global_type.members, recognition.global_types[0].members)
        self.assertEqual(global_type.group_count, 2)
        carrier_group = next(group for group in output.groups if group.group_id == "carrier")
        self.assertEqual(carrier_group.non_linetype.op_indices, ())
        self.assertEqual(len(carrier_group.line_types), 1)
        self.assertEqual(carrier_group.line_types[0].model, "parallel_carrier_companion")
        self.assertEqual(carrier_group.line_types[0].op_indices, carriers)
        self.assertEqual(output.summary.local_line_type_count, 3)
        for source, analyzed in zip(serialized, output.groups):
            validate_group_classification(source, analyzed)

    def test_one_supported_edge_is_not_a_carrier_run(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (150.0, 0.0)),
        ))

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_cyclic_or_branched_support_component_is_rejected(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_single_group_compound_identity_cannot_seed_recovery(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ), cross_group=False)

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_support_must_repeat_at_every_eligible_member_group(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ), repeat_support=False)

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_repeated_components_must_have_the_same_operation_count(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (160.0, 0.0)),
        ), second_carrier_points=(
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (160.0, 0.0)),
            ((160.0, 0.0), (230.0, 0.0)),
        ), motif_count=24)

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_repeated_components_must_have_the_same_visual_style(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        styled_operations = tuple(
            replace(
                operation,
                stroke_color=((0.0,) if index in carriers[:2] else (1.0, 0.0, 0.0)),
                line_width=(0.6 if index in carriers[:2] else 4.0),
                hairline=False,
            )
            if index in carriers
            else operation
            for index, operation in enumerate(page.operations)
        )
        page = replace(page, operations=styled_operations)
        grouping = replace(grouping, page_fingerprint=page.fingerprint)

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_support_component_larger_than_four_operations_is_rejected(self) -> None:
        points = tuple(
            ((float(start), 0.0), (float(min(start + 80, 390)), 0.0))
            for start in range(0, 400, 80)
        )
        page, grouping, serialized, recognition, carriers = _fixture(
            points, motif_count=40
        )

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_simple_open_hatch_motif_cannot_seed_boundary_recovery(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ), simple_motif=True)

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_overlapping_support_paths_are_not_a_forward_route(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (150.0, 0.0)),
            ((0.0, 0.0), (150.0, 0.0)),
        ))

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        self.assertEqual(output, recognition)
        self.assertFalse(set(carriers) & set(output.global_types[0].op_indices))

    def test_support_is_appended_without_renaming_existing_locals(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        existing = LocalLineType(
            type_id="type_existing_001",
            display_name="线型1",
            line_type_index=1,
            atom_count=0,
            op_indices=(),
            model="existing_model",
            shape="非直线",
            shape_detail="existing detail",
        )
        carrier_group = recognition.groups[1]
        recognition = replace(
            recognition,
            groups=(
                recognition.groups[0],
                replace(carrier_group, line_types=(existing,)),
                recognition.groups[2],
            ),
            summary=replace(
                recognition.summary,
                local_line_type_count=recognition.summary.local_line_type_count + 1,
                signed_periodic_type_count=(
                    recognition.summary.signed_periodic_type_count + 1
                ),
            ),
        )

        output = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )

        updated = output.groups[1]
        self.assertEqual(updated.line_types[0], existing)
        self.assertEqual(updated.line_types[1].line_type_index, 2)
        self.assertEqual(updated.line_types[1].display_name, "线型2")
        self.assertEqual(updated.line_types[1].op_indices, carriers)
        validate_group_classification(serialized[1], updated)

    def test_fusion_projects_uniquely_owned_support_without_new_member(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        method1 = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )
        empty_method2 = LineTypeRecognitionResult(
            groups=(),
            global_types=(),
            summary=RecognitionSummary(0, 0, 0, 0, 0, 0, 0),
        )

        fused = fuse_line_type_recognition_results(method1, empty_method2).result

        carrier_group = next(
            group for group in fused.groups if group.group_id == "carrier"
        )
        supports = tuple(
            line_type
            for line_type in carrier_group.line_types
            if line_type.model == "parallel_carrier_companion"
        )
        self.assertEqual(len(supports), 1)
        self.assertEqual(supports[0].op_indices, carriers)
        self.assertTrue(supports[0].type_id.startswith("method1__"))
        self.assertFalse(set(carriers) & set(carrier_group.non_linetype.op_indices))
        global_line_ops = {
            op_index
            for global_type in fused.global_types
            for op_index in global_type.op_indices
        }
        local_line_ops = {
            op_index
            for group in fused.groups
            for line_type in group.line_types
            for op_index in line_type.op_indices
        }
        residual_ops = {
            op_index
            for group in fused.groups
            for op_index in group.non_linetype.op_indices
        }
        self.assertTrue(global_line_ops <= local_line_ops)
        self.assertFalse(global_line_ops & residual_ops)
        self.assertEqual(
            fused.global_types[0].members,
            tuple(
                replace(member, type_id=f"method1__{member.type_id}")
                for member in method1.global_types[0].members
            ),
        )

    def test_fusion_drops_support_with_ambiguous_global_ownership(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        method1 = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )
        duplicate_owner = replace(
            method1.global_types[0],
            global_type_id="global-compound-ambiguous",
        )
        method1 = replace(
            method1,
            global_types=(*method1.global_types, duplicate_owner),
            summary=replace(
                method1.summary,
                global_type_count=2,
                cross_group_global_type_count=2,
            ),
        )
        empty_method2 = LineTypeRecognitionResult(
            groups=(),
            global_types=(),
            summary=RecognitionSummary(0, 0, 0, 0, 0, 0, 0),
        )

        fused = fuse_line_type_recognition_results(method1, empty_method2).result

        carrier_group = next(
            group for group in fused.groups if group.group_id == "carrier"
        )
        self.assertFalse(any(
            line_type.model == "parallel_carrier_companion"
            for line_type in carrier_group.line_types
        ))
        self.assertEqual(carrier_group.non_linetype.op_indices, carriers)

    def test_fusion_counts_skipped_global_when_support_ownership_is_ambiguous(
        self,
    ) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        method1 = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )
        # Fusion is page-neutral, so this sentinel operation models an op
        # exclusively owned by the competing global and by Method2.  The
        # carrier ops themselves are deliberately shared with the retained
        # global; removing the competitor must not make that sharing look
        # unique after the fact.
        method2_overlap_op = 999
        skipped_owner = replace(
            method1.global_types[0],
            global_type_id="global-compound-skipped",
            op_indices=(*carriers, method2_overlap_op),
        )
        method1 = replace(
            method1,
            global_types=(*method1.global_types, skipped_owner),
            summary=replace(
                method1.summary,
                global_type_count=2,
                cross_group_global_type_count=2,
            ),
        )
        method2 = LineTypeRecognitionResult(
            groups=(),
            global_types=(GlobalLineType(
                global_type_id="global-method2",
                signature_family="method2_family",
                minimum_pair_similarity=1.0,
                op_indices=(method2_overlap_op,),
                members=(),
            ),),
            summary=RecognitionSummary(0, 0, 0, 0, 0, 1, 0),
        )

        fused = fuse_line_type_recognition_results(method1, method2).result

        carrier_group = next(
            group for group in fused.groups if group.group_id == "carrier"
        )
        self.assertFalse(any(
            line_type.model == "parallel_carrier_companion"
            for line_type in carrier_group.line_types
        ))
        self.assertEqual(carrier_group.non_linetype.op_indices, carriers)
        self.assertEqual(sum(
            global_type.recognition_source == "method1"
            for global_type in fused.global_types
        ), 1)

    def test_fusion_drops_support_when_method2_skips_its_global(self) -> None:
        page, grouping, serialized, recognition, carriers = _fixture((
            ((0.0, 0.0), (80.0, 0.0)),
            ((80.0, 0.0), (150.0, 0.0)),
        ))
        method1 = recover_compound_carrier_support(
            page, grouping, serialized, recognition
        )
        method2_local = LocalLineType(
            type_id="type_method2_001",
            display_name="线型1",
            line_type_index=1,
            atom_count=1,
            op_indices=(carriers[0],),
            model="method2_model",
            shape="非直线",
            shape_detail="method2 overlap",
        )
        method2 = LineTypeRecognitionResult(
            groups=(RecognizedGroup(
                "carrier",
                len(carriers),
                (method2_local,),
                NonLineType(1, (carriers[1],)),
            ),),
            global_types=(GlobalLineType(
                global_type_id="global-method2",
                signature_family="method2_family",
                minimum_pair_similarity=1.0,
                op_indices=(carriers[0],),
                members=(GlobalLineTypeMember(
                    "carrier",
                    method2_local.type_id,
                    method2_local.display_name,
                    method2_local.atom_count,
                    method2_local.shape,
                    method2_local.model,
                    method2_local.shape_detail,
                ),),
            ),),
            summary=RecognitionSummary(1, 1, 1, 1, 0, 1, 0),
        )

        fused = fuse_line_type_recognition_results(method1, method2).result

        carrier_group = next(
            group for group in fused.groups if group.group_id == "carrier"
        )
        self.assertFalse(any(
            line_type.model == "parallel_carrier_companion"
            for line_type in carrier_group.line_types
        ))
        self.assertEqual(
            tuple(line_type.model for line_type in carrier_group.line_types),
            ("method2_model",),
        )
        self.assertEqual(carrier_group.non_linetype.op_indices, carriers[1:])
        self.assertFalse(any(
            global_type.recognition_source == "method1"
            for global_type in fused.global_types
        ))


if __name__ == "__main__":
    unittest.main()
