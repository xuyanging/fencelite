"""Recover leftover strokes that match a page-confirmed Method1 line type.

This is the final frozen-r10 Method1 stage.  Earlier stages are intentionally
conservative when inventing a line type; once a type is proven, this pass uses
its per-stroke style, physical scale, and rotation-invariant radial ink profile
to reclaim matching contiguous stretches.  Result ownership is always the
dense ``PageIR.operations`` index, never PyMuPDF ``paint_order``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

from ..ir import GroupingIR, PageIR, PathOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    NonLineType,
    RecognizedGroup,
)
from .serializer import SerializedGroup


RECLAIM_SIZE_RATIO = 0.932
RECLAIM_SHAPE_FLOOR = 0.885
RECLAIM_RINGS = 6
RECLAIM_SAMPLES = 48


def _sorted_unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _median(values: Iterable[float]) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


def _number(value: float) -> str:
    finite = float(value) if math.isfinite(float(value)) else 0.0
    normalized = 0.0 if abs(finite) < 1e-10 else finite
    fixed = f"{normalized:.6f}".rstrip("0").rstrip(".")
    return "0" if fixed in {"", "-0"} else fixed


def _color_key(color: tuple[float, ...] | None) -> str:
    return "none" if color is None else ",".join(_number(value) for value in color)


def _path_style_key(operation: PathOperationIR) -> str:
    return "|".join((
        "S" if operation.stroke else "",
        "F" if operation.fill else "",
        _number(operation.line_width),
        _color_key(operation.stroke_color),
        _color_key(operation.fill_color),
    ))


def _path_ink_length(operation: PathOperationIR) -> float:
    length = 0.0
    current: tuple[float, float] | None = None
    for segment in operation.segments:
        if segment.kind == "move":
            current = segment.end
        elif segment.kind == "line":
            if current is not None and segment.end is not None:
                length += math.dist(current, segment.end)
            current = segment.end
        elif segment.kind == "curve":
            if (
                current is not None
                and segment.control_1 is not None
                and segment.control_2 is not None
                and segment.end is not None
            ):
                length += math.dist(current, segment.control_1)
                length += math.dist(segment.control_1, segment.control_2)
                length += math.dist(segment.control_2, segment.end)
            current = segment.end
    return length


def _stroke_rings(operation: PathOperationIR) -> tuple[float, ...] | None:
    corners = tuple(
        segment.end
        for segment in operation.segments
        if segment.kind != "close" and segment.end is not None
    )
    if len(corners) < 2:
        return None
    spans = tuple(math.dist(left, right) for left, right in zip(corners, corners[1:]))
    total = sum(spans)
    if total <= 1e-9:
        return None

    points: list[tuple[float, float]] = []
    for sample in range(RECLAIM_SAMPLES):
        along = total * sample / (RECLAIM_SAMPLES - 1)
        at = 0
        while at < len(spans) - 1 and along > spans[at]:
            along -= spans[at]
            at += 1
        share = along / spans[at] if spans[at] > 1e-9 else 0.0
        points.append((
            corners[at][0] + (corners[at + 1][0] - corners[at][0]) * share,
            corners[at][1] + (corners[at + 1][1] - corners[at][1]) * share,
        ))

    center = (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
    reach = max(math.dist(point, center) for point in points)
    if reach <= 1e-9:
        return None
    rings = [0] * RECLAIM_RINGS
    for point in points:
        radius = math.dist(point, center) / reach
        rings[min(RECLAIM_RINGS - 1, math.floor(radius * RECLAIM_RINGS))] += 1
    return tuple(count / len(points) for count in rings)


def _ring_profile(
    operations: PageOperationIndex,
    op_indices: Sequence[int],
) -> tuple[float, ...] | None:
    per_stroke: list[tuple[float, ...]] = []
    for op_index in op_indices:
        operation = operations.operation(op_index)
        if not isinstance(operation, PathOperationIR):
            continue
        rings = _stroke_rings(operation)
        if rings is not None:
            per_stroke.append(rings)
    if len(per_stroke) < 3:
        return None
    return tuple(
        _median(profile[ring] for profile in per_stroke)
        for ring in range(RECLAIM_RINGS)
    )


@dataclass(frozen=True, slots=True)
class _ReclaimSignature:
    style: str
    ink: float
    rings: tuple[float, ...]


def _signature(
    operations: PageOperationIndex,
    op_indices: Sequence[int],
) -> _ReclaimSignature | None:
    paths = tuple(
        op_index
        for op_index in op_indices
        if isinstance(operations.operation(op_index), PathOperationIR)
    )
    if len(paths) < 3:
        return None
    styles: dict[str, int] = {}
    inks: list[float] = []
    for op_index in paths:
        operation = operations.operation(op_index)
        assert isinstance(operation, PathOperationIR)
        style = _path_style_key(operation)
        styles[style] = styles.get(style, 0) + 1
        inks.append(_path_ink_length(operation))
    dominant = max(styles.items(), key=lambda item: item[1], default=None)
    if dominant is None or dominant[1] < len(paths) * 0.9:
        return None
    rings = _ring_profile(operations, paths)
    if rings is None:
        return None
    ink = _median(value for value in inks if value > 0.0)
    if not ink > 0.0:
        return None
    return _ReclaimSignature(dominant[0], ink, rings)


def _matches(left: _ReclaimSignature, right: _ReclaimSignature) -> bool:
    if left.style != right.style:
        return False
    if min(left.ink, right.ink) / max(1e-9, left.ink, right.ink) < RECLAIM_SIZE_RATIO:
        return False
    apart = sum(abs(a - b) for a, b in zip(left.rings, right.rings))
    return 1.0 - apart / 2.0 >= RECLAIM_SHAPE_FLOOR


def _member_from_local(group: RecognizedGroup, line_type: LocalLineType) -> GlobalLineTypeMember:
    return GlobalLineTypeMember(
        case_id=group.group_id,
        type_id=line_type.type_id,
        display_name=line_type.display_name,
        atom_count=line_type.atom_count,
        model=line_type.model,
        shape=line_type.shape,
        shape_detail=line_type.shape_detail,
    )


def reclaim_confirmed_line_types(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Return ``recognition`` with matching non-line runs reclaimed.

    This stage must run last: it is allowed to reuse only line types already
    confirmed by the preceding Method1 stages.
    """

    operations = PageOperationIndex.build(page, grouping)
    registered: list[tuple[GlobalLineType, _ReclaimSignature]] = []
    for global_type in recognition.global_types:
        signature = _signature(operations, global_type.op_indices)
        if signature is not None:
            registered.append((global_type, signature))
    if not registered:
        return recognition

    serialized_by_group = {group.group_id: group for group in serialized_groups}
    claimed = {
        op_index
        for group in recognition.groups
        for line_type in group.line_types
        for op_index in line_type.op_indices
    }

    leftovers: dict[str, list[tuple[int, ...]]] = {}
    for group in recognition.groups:
        spare = sorted(
            op_index
            for op_index in group.non_linetype.op_indices
            if op_index not in claimed
            and isinstance(operations.operation(op_index), PathOperationIR)
        )
        runs: list[tuple[int, ...]] = []
        current: list[int] = []
        for op_index in spare:
            operation = operations.operation(op_index)
            assert isinstance(operation, PathOperationIR)
            same = False
            if current:
                previous = operations.operation(current[-1])
                same = (
                    isinstance(previous, PathOperationIR)
                    and _path_style_key(previous) == _path_style_key(operation)
                    and op_index - current[-1] <= 4
                )
            if current and not same:
                runs.append(tuple(current))
                current = []
            current.append(op_index)
        if current:
            runs.append(tuple(current))
        leftovers[group.group_id] = [run for run in runs if len(run) >= 3]

    taken_by_group: dict[str, dict[str, list[int]]] = {}
    reclaimed_count = 0
    for group_id, runs in leftovers.items():
        for run in runs:
            signature = _signature(operations, run)
            if signature is None:
                continue
            match = next((item for item in registered if _matches(item[1], signature)), None)
            if match is None:
                continue
            by_type = taken_by_group.setdefault(group_id, {})
            by_type.setdefault(match[0].global_type_id, []).extend(run)
            reclaimed_count += len(run)

    if reclaimed_count == 0:
        return replace(
            recognition,
            summary=replace(recognition.summary, reclaimed_op_count=0),
        )

    added_local_keys: dict[str, list[tuple[str, str]]] = {}
    updated_groups: list[RecognizedGroup] = []
    globals_by_id = {item.global_type_id: item for item in recognition.global_types}
    for group in recognition.groups:
        by_type = taken_by_group.get(group.group_id)
        serialized = serialized_by_group.get(group.group_id)
        if not by_type or serialized is None:
            updated_groups.append(group)
            continue
        used_ids = {line_type.type_id for line_type in group.line_types}
        next_id = 1

        def fresh_type_id() -> str:
            nonlocal next_id
            while True:
                candidate = f"type_reclaim_{next_id:03d}"
                next_id += 1
                if candidate not in used_ids:
                    used_ids.add(candidate)
                    return candidate

        added: list[LocalLineType] = []
        for global_type_id, indices in by_type.items():
            source = globals_by_id.get(global_type_id)
            source_member = source.members[0] if source and source.members else None
            type_id = fresh_type_id()
            path_set = set(indices)
            line_type = LocalLineType(
                type_id=type_id,
                display_name="",
                line_type_index=1,
                atom_count=sum(index in path_set for index in serialized.atom_op_indices),
                op_indices=_sorted_unique(indices),
                model=source_member.model if source_member and source_member.model is not None
                else "compound_path_chain",
                shape=source_member.shape if source_member else "非直线",
                shape_detail=(
                    source_member.shape_detail
                    if source_member and source_member.shape_detail is not None
                    else "同页已确认线型"
                ),
            )
            added.append(line_type)
            added_local_keys.setdefault(global_type_id, []).append((group.group_id, type_id))

        ordered = sorted(
            (*group.line_types, *added),
            key=lambda item: min(item.op_indices, default=math.inf),
        )
        line_types = tuple(
            replace(item, display_name=f"线型{index}", line_type_index=index)
            for index, item in enumerate(ordered, start=1)
        )
        assigned = {index for item in line_types for index in item.op_indices}
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=NonLineType(
                atom_count=sum(index not in assigned for index in serialized.atom_op_indices),
                op_indices=_sorted_unique(
                    index for index in group.non_linetype.op_indices if index not in assigned
                ),
                display_name=group.non_linetype.display_name,
            ),
        ))

    local_by_key = {
        (group.group_id, line_type.type_id): (group, line_type)
        for group in updated_groups
        for line_type in group.line_types
    }
    global_types: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        keys = added_local_keys.get(global_type.global_type_id, ())
        if not keys:
            global_types.append(global_type)
            continue
        members = list(global_type.members)
        op_indices = list(global_type.op_indices)
        for key in keys:
            entry = local_by_key.get(key)
            if entry is None:
                continue
            group, line_type = entry
            members.append(_member_from_local(group, line_type))
            op_indices.extend(line_type.op_indices)
        global_types.append(replace(
            global_type,
            op_indices=_sorted_unique(op_indices),
            members=tuple(members),
        ))

    local_count = sum(group.line_type_count for group in updated_groups)
    output = replace(
        recognition,
        groups=tuple(updated_groups),
        global_types=tuple(global_types),
        summary=replace(
            recognition.summary,
            reclaimed_op_count=reclaimed_count,
            local_line_type_count=local_count,
            signed_periodic_type_count=max(
                0,
                recognition.summary.signed_periodic_type_count
                + local_count
                - recognition.summary.local_line_type_count,
            ),
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(item.group_count > 1 for item in global_types),
        ),
    )
    # Re-validate all result partitions and derived counters at this boundary.
    return LineTypeRecognitionResult.from_dict(output.to_dict())
