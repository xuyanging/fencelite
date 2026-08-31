"""Recover carrier strokes that are proven by a compound periodic motif.

The compound recognizer deliberately clusters the repeated glyphs/modules and
does not use a long carrier as identity evidence.  Authored PDFs may place the
carrier in a neighbouring paint Group, or the route filter may demote a short
two-operation carrier run.  This final Method1 pass restores only carriers
whose geometry is supported by an already cross-Group-confirmed compound
identity.

Carrier locals are ownership/support records, not additional signature
members.  Consequently they extend ``GlobalLineType.op_indices`` for drawing
and callout binding while ``GlobalLineType.members`` continues to describe
only the independently clustered motif instances.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from statistics import median
from typing import Iterable, Sequence

from ..geometry import (
    bounds_gap,
    connection_tolerance,
    flatten_path,
    operation_contour_gap,
    page_diagonal,
)
from ..ir import GroupingIR, PageIR, PathOperationIR, PointIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    LineTypeRecognitionResult,
    LocalLineType,
    NonLineType,
    RecognizedGroup,
)
from .serializer import SerializedGroup


MIN_TARGET_PATH_COUNT = 12
MIN_CONTACT_COUNT = 8
MIN_TARGET_COVERAGE = 0.65
MIN_CARRIER_TO_MOTIF_INK = 2.2
MIN_CARRIER_STRAIGHTNESS = 0.98
MAX_CONTACT_ANGLE_DEGREES = 5.0
MAX_ROUTE_TURN_DEGREES = 30.0
MIN_SUPPORT_RUN_SIZE = 2
MAX_SUPPORT_RUN_SIZE = 4
MIN_SUPPORTED_TARGET_GROUPS = 2
MAX_SIMPLE_OPEN_NEAR_STRAIGHT_RATIO = 0.25


def _sorted_unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _carrier_style_key(operation: PathOperationIR) -> tuple[object, ...]:
    """Return the visual identity shared by one recovered carrier run.

    Carrier geometry proves adjacency, not line-type identity.  Repeated
    components with different color, opacity, width, hairline, cap/join or
    blend cannot support one global compound line type.  Layer and source
    provenance remain excluded because identical visible line types may be
    authored in different optional-content layers or form streams.
    """

    return (
        operation.stroke,
        operation.fill,
        tuple(round(value, 9) for value in (operation.stroke_color or ())),
        round(operation.stroke_opacity, 6),
        round(operation.line_width, 3),
        operation.hairline,
        operation.line_cap,
        round(operation.line_join, 3),
        operation.blend_mode,
    )


@dataclass(frozen=True, slots=True)
class _CarrierGeometry:
    op_index: int
    group_id: str
    start: PointIR
    end: PointIR
    direction: PointIR
    length: float


@dataclass(frozen=True, slots=True)
class _TargetMember:
    global_type: GlobalLineType
    group_id: str
    op_indices: tuple[int, ...]
    median_ink: float


@dataclass(frozen=True, slots=True)
class _SupportEvidence:
    carrier: _CarrierGeometry
    target: _TargetMember
    contacts: tuple[int, ...]

    @property
    def score(self) -> tuple[float, int]:
        return (len(self.contacts) / len(self.target.op_indices), len(self.contacts))


def _path_ink_length(operation: PathOperationIR, diagonal: float) -> float:
    return sum(
        math.dist(edge.start, edge.end)
        for edge in flatten_path(operation, diagonal)
    )


def _is_simple_open_near_straight(operation: PathOperationIR) -> bool:
    """Identify hatch primitives that must not seed boundary recovery."""

    if (
        operation.close_path
        or any(segment.kind in {"curve", "close"} for segment in operation.segments)
    ):
        return False
    moves = [segment for segment in operation.segments if segment.kind == "move"]
    lines = [segment for segment in operation.segments if segment.kind == "line"]
    if (
        len(moves) != 1
        or not lines
        or operation.segments[0].kind != "move"
        or any(segment.kind not in {"move", "line"} for segment in operation.segments)
    ):
        return False
    start = moves[0].end
    end = lines[-1].end
    if start is None or end is None:
        return False
    total = 0.0
    current = start
    for segment in lines:
        assert segment.end is not None
        total += math.dist(current, segment.end)
        current = segment.end
    return total > 1e-9 and math.dist(start, end) / total >= MIN_CARRIER_STRAIGHTNESS


def _carrier_geometry(
    op_index: int,
    group_id: str,
    operation: PathOperationIR,
) -> _CarrierGeometry | None:
    """Return an open, single-subpath, nearly straight stroke carrier."""

    if (
        not operation.stroke
        or operation.fill
        or operation.stroke_opacity <= 0.0
        or operation.dash_array
        or operation.close_path
        or any(segment.kind in {"curve", "close"} for segment in operation.segments)
    ):
        return None
    moves = [segment for segment in operation.segments if segment.kind == "move"]
    lines = [segment for segment in operation.segments if segment.kind == "line"]
    if (
        len(moves) != 1
        or not lines
        or len(lines) > 12
        or operation.segments[0].kind != "move"
        or any(segment.kind not in {"move", "line"} for segment in operation.segments)
    ):
        return None
    start = moves[0].end
    end = lines[-1].end
    if start is None or end is None:
        return None
    total = 0.0
    current = start
    for segment in lines:
        assert segment.end is not None
        total += math.dist(current, segment.end)
        current = segment.end
    chord = math.dist(start, end)
    if total <= 1e-9 or chord / total < MIN_CARRIER_STRAIGHTNESS:
        return None
    return _CarrierGeometry(
        op_index=op_index,
        group_id=group_id,
        start=start,
        end=end,
        direction=((end[0] - start[0]) / chord, (end[1] - start[1]) / chord),
        length=chord,
    )


def _operation_center(operation: PathOperationIR) -> PointIR:
    return (
        (operation.bounds.min_x + operation.bounds.max_x) / 2.0,
        (operation.bounds.min_y + operation.bounds.max_y) / 2.0,
    )


def _contact_axis_is_parallel(
    page: PageIR,
    contacts: Sequence[int],
    direction: PointIR,
    motif_ink: float,
) -> bool:
    """Require contacted motif centers to form a line along the carrier."""

    centers = tuple(
        _operation_center(operation)
        for op_index in contacts
        if isinstance((operation := page.operations[op_index]), PathOperationIR)
    )
    if len(centers) < MIN_CONTACT_COUNT:
        return False
    center = (
        sum(point[0] for point in centers) / len(centers),
        sum(point[1] for point in centers) / len(centers),
    )
    xx = sum((point[0] - center[0]) ** 2 for point in centers)
    yy = sum((point[1] - center[1]) ** 2 for point in centers)
    xy = sum(
        (point[0] - center[0]) * (point[1] - center[1])
        for point in centers
    )
    trace = xx + yy
    discriminant = math.sqrt(max(0.0, (xx - yy) ** 2 + 4.0 * xy * xy))
    largest = (trace + discriminant) / 2.0
    if largest <= 1e-9:
        return False
    principal = (xy, largest - xx)
    norm = math.hypot(*principal)
    if norm <= 1e-9:
        principal = (1.0, 0.0) if xx >= yy else (0.0, 1.0)
    else:
        principal = (principal[0] / norm, principal[1] / norm)
    alignment = abs(
        principal[0] * direction[0] + principal[1] * direction[1]
    )
    if alignment < math.cos(math.radians(MAX_CONTACT_ANGLE_DEGREES)):
        return False
    projections = tuple(
        point[0] * direction[0] + point[1] * direction[1]
        for point in centers
    )
    return max(projections) - min(projections) >= motif_ink * 0.85


def _member_targets(
    page: PageIR,
    recognition: LineTypeRecognitionResult,
    diagonal: float,
) -> tuple[_TargetMember, ...]:
    local_by_key = {
        (group.group_id, line_type.type_id): line_type
        for group in recognition.groups
        for line_type in group.line_types
    }
    targets: list[_TargetMember] = []
    for global_type in recognition.global_types:
        if (
            global_type.signature_family != "compound_path_periodic"
            or global_type.group_count < 2
        ):
            continue
        for member in global_type.members:
            # Reclaimed carrier-like locals use the same legacy model string;
            # only compound-stage members are independent motif identities.
            if (
                member.model != "compound_path_chain"
                or not member.type_id.startswith("type_compound_")
            ):
                continue
            local = local_by_key.get((member.case_id, member.type_id))
            if local is None:
                continue
            paths = tuple(
                op_index
                for op_index in local.op_indices
                if isinstance(page.operations[op_index], PathOperationIR)
            )
            if len(paths) < MIN_TARGET_PATH_COUNT:
                continue
            simple_path_count = sum(
                _is_simple_open_near_straight(page.operations[op_index])
                for op_index in paths
            )
            if (
                simple_path_count / len(paths)
                > MAX_SIMPLE_OPEN_NEAR_STRAIGHT_RATIO
            ):
                # Rows of diagonal hatch strokes can touch a material or wall
                # boundary hundreds of times.  That boundary is not a carrier
                # for the hatch.  P3's proven square-post motif is structurally
                # compound (2/49 and 1/25 simple paths), unlike the all-simple
                # Ponderosa P16 and Rapid City P24 hatch regressions.
                continue
            inks = tuple(
                _path_ink_length(page.operations[op_index], diagonal)
                for op_index in paths
            )
            positive_inks = tuple(value for value in inks if value > 1e-9)
            if not positive_inks:
                continue
            targets.append(_TargetMember(
                global_type=global_type,
                group_id=member.case_id,
                op_indices=paths,
                median_ink=median(positive_inks),
            ))
    return tuple(targets)


def _candidate_evidence(
    page: PageIR,
    target: _TargetMember,
    carrier: _CarrierGeometry,
    diagonal: float,
) -> _SupportEvidence | None:
    if carrier.length < target.median_ink * MIN_CARRIER_TO_MOTIF_INK:
        return None
    carrier_operation = page.operations[carrier.op_index]
    assert isinstance(carrier_operation, PathOperationIR)
    contacts: list[int] = []
    for op_index in target.op_indices:
        motif_operation = page.operations[op_index]
        assert isinstance(motif_operation, PathOperationIR)
        tolerance = connection_tolerance(carrier_operation, motif_operation, diagonal)
        # Cheap conservative rejection before exact contour flattening.
        visible_slack = (
            max(0.25 if carrier_operation.hairline else carrier_operation.line_width / 2.0, 0.0)
            + max(0.25 if motif_operation.hairline else motif_operation.line_width / 2.0, 0.0)
        )
        if bounds_gap(carrier_operation.bounds, motif_operation.bounds) > (
            tolerance + visible_slack
        ):
            continue
        if operation_contour_gap(
            carrier_operation, motif_operation, diagonal
        ) <= tolerance:
            contacts.append(op_index)
    if len(contacts) < MIN_CONTACT_COUNT:
        return None
    if not _contact_axis_is_parallel(
        page, contacts, carrier.direction, target.median_ink
    ):
        return None
    return _SupportEvidence(carrier, target, tuple(contacts))


def _connected_components(
    evidences: Sequence[_SupportEvidence],
    page: PageIR,
    diagonal: float,
) -> tuple[tuple[_SupportEvidence, ...], ...]:
    adjacency: list[set[int]] = [set() for _ in evidences]
    turn_limit = math.cos(math.radians(MAX_ROUTE_TURN_DEGREES))
    for left in range(len(evidences)):
        left_carrier = evidences[left].carrier
        left_operation = page.operations[left_carrier.op_index]
        assert isinstance(left_operation, PathOperationIR)
        for right in range(left + 1, len(evidences)):
            right_carrier = evidences[right].carrier
            right_operation = page.operations[right_carrier.op_index]
            assert isinstance(right_operation, PathOperationIR)
            alignment = abs(
                left_carrier.direction[0] * right_carrier.direction[0]
                + left_carrier.direction[1] * right_carrier.direction[1]
            )
            if alignment < turn_limit:
                continue
            endpoint_gap = min(
                math.dist(left_point, right_point)
                for left_point in (left_carrier.start, left_carrier.end)
                for right_point in (right_carrier.start, right_carrier.end)
            )
            if endpoint_gap > connection_tolerance(
                left_operation, right_operation, diagonal
            ):
                continue
            adjacency[left].add(right)
            adjacency[right].add(left)

    components: list[tuple[_SupportEvidence, ...]] = []
    unseen = set(range(len(evidences)))
    while unseen:
        seed = min(unseen)
        pending = [seed]
        unseen.remove(seed)
        component: list[int] = []
        while pending:
            current = pending.pop()
            component.append(current)
            for neighbour in sorted(adjacency[current], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    pending.append(neighbour)
        # A carrier support route must be one open, non-branching chain.
        edge_count = sum(len(adjacency[index]) for index in component) // 2
        if (
            len(component) < MIN_SUPPORT_RUN_SIZE
            or edge_count != len(component) - 1
            or any(len(adjacency[index]) > 2 for index in component)
        ):
            continue
        components.append(tuple(evidences[index] for index in sorted(component)))
    return tuple(components)


def _component_has_forward_progress(
    component: Sequence[_SupportEvidence],
    page: PageIR,
    diagonal: float,
) -> bool:
    """Reject coincident, folded-back, or heavily overlapping support paths."""

    for left in range(len(component)):
        left_carrier = component[left].carrier
        left_operation = page.operations[left_carrier.op_index]
        assert isinstance(left_operation, PathOperationIR)
        for right in range(left + 1, len(component)):
            right_carrier = component[right].carrier
            right_operation = page.operations[right_carrier.op_index]
            assert isinstance(right_operation, PathOperationIR)
            tolerance = connection_tolerance(
                left_operation, right_operation, diagonal
            )
            endpoint_gap = min(
                math.dist(left_point, right_point)
                for left_point in (left_carrier.start, left_carrier.end)
                for right_point in (right_carrier.start, right_carrier.end)
            )
            if endpoint_gap > tolerance:
                continue
            direction = left_carrier.direction
            left_projection = sorted(
                point[0] * direction[0] + point[1] * direction[1]
                for point in (left_carrier.start, left_carrier.end)
            )
            right_projection = sorted(
                point[0] * direction[0] + point[1] * direction[1]
                for point in (right_carrier.start, right_carrier.end)
            )
            overlap = max(
                0.0,
                min(left_projection[1], right_projection[1])
                - max(left_projection[0], right_projection[0]),
            )
            if overlap > tolerance:
                return False
    return True


def recover_compound_carrier_support(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Attach uniquely proven carrier runs to confirmed compound identities."""

    operation_index = PageOperationIndex.build(page, grouping)
    diagonal = page_diagonal(page.page_bounds)
    targets = _member_targets(page, recognition, diagonal)
    if not targets:
        return recognition

    group_position = {
        group.group_id: position for position, group in enumerate(grouping.groups)
    }
    residual_by_group = {
        group.group_id: set(group.non_linetype.op_indices)
        for group in recognition.groups
    }
    carriers: dict[int, _CarrierGeometry] = {}
    for group in recognition.groups:
        for op_index in residual_by_group[group.group_id]:
            operation = operation_index.operation(op_index)
            if not isinstance(operation, PathOperationIR):
                continue
            carrier = _carrier_geometry(op_index, group.group_id, operation)
            if carrier is not None:
                carriers[op_index] = carrier

    evidence_by_op: dict[int, list[_SupportEvidence]] = {}
    for target in targets:
        target_position = group_position.get(target.group_id)
        if target_position is None:
            continue
        allowed_groups = {
            grouping.groups[position].group_id
            for position in range(
                max(0, target_position - 1),
                min(len(grouping.groups), target_position + 2),
            )
        }
        for carrier in carriers.values():
            if carrier.group_id not in allowed_groups:
                continue
            evidence = _candidate_evidence(page, target, carrier, diagonal)
            if evidence is not None:
                evidence_by_op.setdefault(carrier.op_index, []).append(evidence)

    # A path that strongly supports more than one global identity is
    # geometrically ambiguous and stays residual. Multiple members of the same
    # global may overlap; retain only their best-supported member.
    unique: list[_SupportEvidence] = []
    for op_index, options in evidence_by_op.items():
        global_ids = {
            option.target.global_type.global_type_id for option in options
        }
        if len(global_ids) != 1:
            continue
        unique.append(max(options, key=lambda option: option.score))

    by_target: dict[tuple[str, str], list[_SupportEvidence]] = {}
    for evidence in unique:
        key = (
            evidence.target.global_type.global_type_id,
            evidence.target.group_id,
        )
        by_target.setdefault(key, []).append(evidence)

    target_groups_by_global: dict[str, set[str]] = {}
    for target in targets:
        target_groups_by_global.setdefault(
            target.global_type.global_type_id, set()
        ).add(target.group_id)

    accepted_by_global: dict[str, set[int]] = {}
    for global_type_id, target_group_ids in target_groups_by_global.items():
        # One geometric coincidence is not enough to extend a cross-Group
        # identity.  The same small support topology must independently recur
        # beside every eligible compound member of that global type.
        if len(target_group_ids) < MIN_SUPPORTED_TARGET_GROUPS:
            continue
        selected_components: list[tuple[_SupportEvidence, ...]] = []
        selected_component_styles: list[tuple[object, ...]] = []
        valid = True
        for group_id in sorted(target_group_ids):
            evidences = by_target.get((global_type_id, group_id), ())
            components = _connected_components(evidences, page, diagonal)
            if len(components) != 1:
                valid = False
                break
            component = components[0]
            if (
                len(component) > MAX_SUPPORT_RUN_SIZE
                or not _component_has_forward_progress(component, page, diagonal)
            ):
                valid = False
                break
            component_styles = {
                _carrier_style_key(operation)
                for evidence in component
                if isinstance(
                    operation := page.operations[evidence.carrier.op_index],
                    PathOperationIR,
                )
            }
            if len(component_styles) != 1:
                valid = False
                break
            target_ops = set(component[0].target.op_indices)
            contacted = {
                op_index
                for evidence in component
                for op_index in evidence.contacts
            }
            if len(contacted) / len(target_ops) < MIN_TARGET_COVERAGE:
                valid = False
                break
            selected_components.append(component)
            selected_component_styles.append(next(iter(component_styles)))
        if (
            not valid
            or len({len(component) for component in selected_components}) != 1
            or len(set(selected_component_styles)) != 1
        ):
            continue
        accepted_by_global[global_type_id] = {
            evidence.carrier.op_index
            for component in selected_components
            for evidence in component
        }
    if not accepted_by_global:
        return recognition

    # Recheck uniqueness after whole-run acceptance. This matters when two
    # nearby target members each produced a different valid component that
    # happens to share one operation through atom multiplicity.
    global_ids_by_op: dict[int, set[str]] = {}
    for global_type_id, op_indices in accepted_by_global.items():
        for op_index in op_indices:
            global_ids_by_op.setdefault(op_index, set()).add(global_type_id)
    accepted_by_global = {
        global_type_id: {
            op_index
            for op_index in op_indices
            if len(global_ids_by_op[op_index]) == 1
        }
        for global_type_id, op_indices in accepted_by_global.items()
    }
    accepted_by_global = {
        key: value for key, value in accepted_by_global.items() if value
    }
    if not accepted_by_global:
        return recognition

    owner_by_op = {
        op_index: global_type_id
        for global_type_id, op_indices in accepted_by_global.items()
        for op_index in op_indices
    }
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    additions_by_group: dict[str, dict[str, list[int]]] = {}
    for op_index, global_type_id in owner_by_op.items():
        additions_by_group.setdefault(operation_index.group_id(op_index), {}).setdefault(
            global_type_id, []
        ).append(op_index)

    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        additions = additions_by_group.get(group.group_id)
        serialized = serialized_by_group.get(group.group_id)
        if not additions or serialized is None:
            updated_groups.append(group)
            continue
        used_ids = {line_type.type_id for line_type in group.line_types}
        next_id = 1

        def fresh_type_id() -> str:
            nonlocal next_id
            while True:
                candidate = f"type_carrier_support_{next_id:03d}"
                next_id += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    return candidate

        added: list[LocalLineType] = []
        for global_type_id, op_indices in sorted(additions.items()):
            path_set = set(op_indices)
            line_type_index = len(group.line_types) + len(added) + 1
            added.append(LocalLineType(
                type_id=fresh_type_id(),
                display_name=f"线型{line_type_index}",
                line_type_index=line_type_index,
                atom_count=sum(
                    atom_index in path_set for atom_index in serialized.atom_op_indices
                ),
                op_indices=_sorted_unique(op_indices),
                model="parallel_carrier_companion",
                shape="非直线",
                shape_detail=(
                    "跨组复合周期 pattern 已确认；平行承载线仅作为支持覆盖"
                ),
            ))
        # Existing locals are referenced by GlobalLineTypeMember metadata.
        # Preserve their ids, names, and indices exactly; support-only locals
        # are appended after them and are intentionally not signature members.
        line_types = (*group.line_types, *added)
        assigned = {
            op_index for line_type in line_types for op_index in line_type.op_indices
        }
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=NonLineType(
                atom_count=sum(
                    atom_index not in assigned
                    for atom_index in serialized.atom_op_indices
                ),
                op_indices=_sorted_unique(
                    op_index
                    for op_index in group.non_linetype.op_indices
                    if op_index not in assigned
                ),
                display_name=group.non_linetype.display_name,
            ),
        ))

    global_types = tuple(
        replace(
            global_type,
            op_indices=_sorted_unique((
                *global_type.op_indices,
                *accepted_by_global.get(global_type.global_type_id, ()),
            )),
        )
        if global_type.global_type_id in accepted_by_global
        else global_type
        for global_type in recognition.global_types
    )
    local_count = sum(group.line_type_count for group in updated_groups)
    output = replace(
        recognition,
        groups=tuple(updated_groups),
        global_types=global_types,
        summary=replace(
            recognition.summary,
            local_line_type_count=local_count,
            signed_periodic_type_count=max(
                0,
                recognition.summary.signed_periodic_type_count
                + local_count
                - recognition.summary.local_line_type_count,
            ),
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in global_types
            ),
        ),
    )
    return LineTypeRecognitionResult.from_dict(output.to_dict())


__all__ = ["recover_compound_carrier_support"]
