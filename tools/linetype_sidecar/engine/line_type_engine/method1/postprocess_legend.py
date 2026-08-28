"""Frozen Method1 vector-only and solid Legend recovery stages.

Result operation ids are dense positions in :class:`PageIR.operations`.
``paint_order`` remains authored ordering evidence only and is never used as
an ownership identity in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Iterable, Sequence

from ..ir import GroupingIR, PageIR, PathOperationIR, TextOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    RecognizedGroup,
)
from .serializer import SerializedGroup


def _sorted_unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _median(values: Iterable[float]) -> float:
    ordered = sorted(value for value in values if math.isfinite(value))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _bounds_center(operation: PathOperationIR | TextOperationIR) -> tuple[float, float]:
    return (
        (operation.bounds.min_x + operation.bounds.max_x) / 2.0,
        (operation.bounds.min_y + operation.bounds.max_y) / 2.0,
    )


def _normalized_line_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).upper()


def _javascript_string_length(text: str) -> int:
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def _atom_count_for_ops(serialized: SerializedGroup, op_indices: set[int]) -> int:
    return sum(op_index in op_indices for op_index in serialized.atom_op_indices)


def _index_local_types(
    groups: Sequence[RecognizedGroup],
) -> dict[tuple[str, str], LocalLineType]:
    return {
        (group.group_id, line_type.type_id): line_type
        for group in groups
        for line_type in group.line_types
    }


def _renumber_globals(
    global_types: Iterable[GlobalLineType],
) -> tuple[GlobalLineType, ...]:
    return tuple(
        replace(global_type, global_type_id=f"global_type_{index:03d}")
        for index, global_type in enumerate(global_types, start=1)
    )


def _count_local_types(groups: Sequence[RecognizedGroup]) -> int:
    return sum(group.line_type_count for group in groups)


def _path(
    operation_index: PageOperationIndex,
    op_index: int,
) -> PathOperationIR | None:
    if op_index < 0 or op_index >= len(operation_index.operations):
        return None
    operation = operation_index.operations[op_index]
    return operation if isinstance(operation, PathOperationIR) else None


@dataclass(frozen=True, slots=True)
class _VectorLegendRow:
    group_id: str
    center_y: float
    lane_min_x: float
    lane_max_x: float
    pitch: float
    op_indices: tuple[int, ...]
    signature: str


@dataclass(frozen=True, slots=True)
class _Anchor:
    op_index: int
    min_x: float
    max_x: float
    y: float
    width: float


@dataclass(frozen=True, slots=True)
class _LaneItem:
    min_x: float
    max_x: float
    y: float


def _vector_legend_row_signature(
    operation_index: PageOperationIndex,
    op_indices: Sequence[int],
    lane_min_x: float,
    lane_max_x: float,
    center_y: float,
    pitch: float,
) -> str:
    lane_width = max(0.001, lane_max_x - lane_min_x)
    tokens: list[str] = []
    for op_index in op_indices:
        operation = _path(operation_index, op_index)
        if operation is None:
            continue
        center_x, operation_center_y = _bounds_center(operation)
        token = ":".join((
            str(_js_round((center_x - lane_min_x) / lane_width * 32.0)),
            str(_js_round((operation_center_y - center_y) / max(0.001, pitch) * 8.0)),
            str(_js_round(operation.bounds.width / lane_width * 32.0)),
            str(_js_round(operation.bounds.height / max(0.001, pitch) * 8.0)),
            str(min(16, len(operation.segments))),
            "C" if any(segment.kind == "curve" for segment in operation.segments)
            else "L",
            "Z" if any(segment.kind == "close" for segment in operation.segments)
            else "O",
        ))
        tokens.append(token)
    return "|".join(sorted(tokens))


def _vector_legend_rows_for_group(
    operation_index: PageOperationIndex,
    serialized: SerializedGroup,
) -> tuple[_VectorLegendRow, ...]:
    try:
        group_indices = set(operation_index.group_indices(serialized.group_id))
    except KeyError:
        return ()
    path_indices = tuple(
        op_index
        for op_index in _sorted_unique(serialized.atom_op_indices)
        if op_index in group_indices and _path(operation_index, op_index) is not None
    )
    paths = tuple(
        (op_index, _path(operation_index, op_index))
        for op_index in path_indices
    )
    anchors: list[_Anchor] = []
    for op_index, candidate in paths:
        assert candidate is not None
        width = candidate.bounds.width
        height = candidate.bounds.height
        simple = (
            candidate.stroke
            and not candidate.fill
            and len(candidate.segments) == 2
            and candidate.segments[0].kind == "move"
            and candidate.segments[1].kind == "line"
        )
        if (
            not simple
            or width < max(8.0, candidate.line_width * 12.0)
            or width > 180.0
            or height > max(1.5, width * 0.04)
        ):
            continue
        anchors.append(_Anchor(
            op_index=op_index,
            min_x=candidate.bounds.min_x,
            max_x=candidate.bounds.max_x,
            y=(candidate.bounds.min_y + candidate.bounds.max_y) / 2.0,
            width=width,
        ))
    anchors.sort(key=lambda anchor: anchor.y)
    if len(anchors) < 16:
        return ()

    y_bands: list[list[_Anchor]] = []
    for anchor in anchors:
        band = y_bands[-1] if y_bands else None
        if band is None or abs(anchor.y - _median(item.y for item in band)) > 1.5:
            y_bands.append([anchor])
        else:
            band.append(anchor)

    interval_buckets: dict[str, list[_LaneItem]] = {}
    for band in y_bands:
        for left_index in range(len(band)):
            for right_index in range(left_index + 1, len(band)):
                left, right = sorted(
                    (band[left_index], band[right_index]),
                    key=lambda anchor: anchor.min_x,
                )
                gap = right.min_x - left.max_x
                width_ratio = max(left.width, right.width) / max(
                    0.001, min(left.width, right.width)
                )
                span = right.max_x - left.min_x
                if (
                    gap < 4.0
                    or gap > min(90.0, max(left.width, right.width) * 1.5)
                    or width_ratio > 1.40
                    or span < 50.0
                    or span > 240.0
                    or max(left.width, right.width) > span * 0.48
                ):
                    continue
                key = f"{_js_round(left.min_x / 4.0)}:{_js_round(right.max_x / 4.0)}"
                interval_buckets.setdefault(key, []).append(_LaneItem(
                    min_x=left.min_x,
                    max_x=right.max_x,
                    y=_median(item.y for item in band),
                ))

    lane_candidates: list[tuple[list[_LaneItem], float, float]] = []
    for items in interval_buckets.values():
        unique_by_y: dict[int, _LaneItem] = {}
        for item in items:
            unique_by_y[_js_round(item.y * 2.0)] = item
        unique_items = sorted(unique_by_y.values(), key=lambda item: item.y)
        if len(unique_items) < 8:
            continue
        lane_candidates.append((
            unique_items,
            _median(item.min_x for item in items),
            _median(item.max_x for item in items),
        ))
    lane_candidates.sort(key=lambda candidate: len(candidate[0]), reverse=True)
    if not lane_candidates:
        return ()
    lane_items, lane_min_x, lane_max_x = lane_candidates[0]

    differences = tuple(
        lane_items[index].y - lane_items[index - 1].y
        for index in range(1, len(lane_items))
        if 4.0 <= lane_items[index].y - lane_items[index - 1].y <= 60.0
    )
    lower_half = _median(differences)
    pitch = _median(
        difference
        for difference in differences
        if difference <= lower_half * 1.35
    )
    if pitch < 4.0 or pitch > 60.0:
        return ()
    consistent = 0
    for difference in differences:
        multiple = max(1, _js_round(difference / pitch))
        if multiple <= 3 and abs(difference - multiple * pitch) <= pitch * 0.20:
            consistent += 1
    if consistent < max(6.0, len(differences) * 0.70):
        return ()

    lane_width = lane_max_x - lane_min_x

    def row_ops_for(center_y: float) -> tuple[int, ...]:
        result: list[int] = []
        for op_index, candidate in paths:
            assert candidate is not None
            center_x, operation_center_y = _bounds_center(candidate)
            if (
                center_x >= lane_min_x - 2.0
                and center_x <= lane_max_x + 2.0
                and abs(operation_center_y - center_y) <= pitch * 0.43
                and candidate.bounds.width <= lane_width * 1.12
                and candidate.bounds.height <= pitch * 1.15
            ):
                result.append(op_index)
        return tuple(result)

    observed_min = lane_items[0].y
    observed_max = lane_items[-1].y
    centers: list[float] = []
    center = observed_min
    while center <= observed_max + pitch * 0.25:
        centers.append(center)
        center += pitch

    def extend(direction: int) -> None:
        center_y = (centers[0] if direction < 0 else centers[-1]) + direction * pitch
        for _attempt in range(4):
            op_indices = row_ops_for(center_y)
            row_paths = tuple(
                candidate
                for op_index in op_indices
                if (candidate := _path(operation_index, op_index)) is not None
            )
            coverage = (
                max(candidate.bounds.max_x for candidate in row_paths)
                - min(candidate.bounds.min_x for candidate in row_paths)
                if row_paths
                else 0.0
            )
            if len(op_indices) < 2 or coverage < lane_width * 0.55:
                break
            if direction < 0:
                centers.insert(0, center_y)
            else:
                centers.append(center_y)
            center_y += direction * pitch

    extend(-1)
    extend(1)

    rows = tuple(
        _VectorLegendRow(
            group_id=serialized.group_id,
            center_y=center_y,
            lane_min_x=lane_min_x,
            lane_max_x=lane_max_x,
            pitch=pitch,
            op_indices=op_indices,
            signature=_vector_legend_row_signature(
                operation_index,
                op_indices,
                lane_min_x,
                lane_max_x,
                center_y,
                pitch,
            ),
        )
        for center_y in centers
        if (op_indices := row_ops_for(center_y))
    )
    topology_diversity = len({
        (
            len(row.op_indices),
            sum(
                len(candidate.segments)
                for op_index in row.op_indices
                if (candidate := _path(operation_index, op_index)) is not None
            ),
        )
        for row in rows
    })
    return rows if len(rows) >= 8 and topology_diversity >= 4 else ()


def _rebuilt_global(
    source: GlobalLineType,
    members: Sequence[GlobalLineTypeMember],
    local_by_key: dict[tuple[str, str], LocalLineType],
) -> GlobalLineType:
    return replace(
        source,
        members=tuple(members),
        op_indices=_sorted_unique(
            op_index
            for member in members
            for op_index in local_by_key[(member.case_id, member.type_id)].op_indices
        ),
    )


def augment_vector_legend_samples(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Split a proven vector-only Legend swatch column into sample rows."""

    operation_index = PageOperationIndex.build(page, grouping)
    rows = tuple(
        row
        for serialized in serialized_groups
        for row in _vector_legend_rows_for_group(operation_index, serialized)
    )
    if not rows:
        return recognition
    rows_by_group: dict[str, list[_VectorLegendRow]] = {}
    for row in rows:
        rows_by_group.setdefault(row.group_id, []).append(row)
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    added: dict[tuple[str, str], _VectorLegendRow] = {}
    updated_groups: list[RecognizedGroup] = []

    for group in recognition.groups:
        group_rows = rows_by_group.get(group.group_id)
        serialized = serialized_by_group.get(group.group_id)
        if not group_rows or serialized is None:
            updated_groups.append(group)
            continue
        row_ops = {
            op_index for row in group_rows for op_index in row.op_indices
        }
        used_type_ids = {line_type.type_id for line_type in group.line_types}
        retained: list[LocalLineType] = []
        for line_type in group.line_types:
            hit_rows = sum(
                any(op_index in line_type.op_indices for op_index in row.op_indices)
                for row in group_rows
            )
            if hit_rows >= 4:
                continue
            op_indices = tuple(
                op_index
                for op_index in line_type.op_indices
                if op_index not in row_ops
            )
            paths = {
                op_index
                for op_index in op_indices
                if _path(operation_index, op_index) is not None
            }
            atom_count = _atom_count_for_ops(serialized, paths)
            if atom_count > 0:
                retained.append(replace(
                    line_type,
                    atom_count=atom_count,
                    op_indices=op_indices,
                ))

        additions: list[LocalLineType] = []
        for index, row in enumerate(group_rows, start=1):
            ordinal = index
            type_id = f"type_vector_legend_{ordinal:03d}"
            while type_id in used_type_ids:
                ordinal += 1
                type_id = f"type_vector_legend_{ordinal:03d}"
            used_type_ids.add(type_id)
            shape = "非直线" if any(
                candidate is not None
                and (
                    len(candidate.segments) > 2
                    or any(segment.kind == "curve" for segment in candidate.segments)
                )
                for op_index in row.op_indices
                for candidate in (_path(operation_index, op_index),)
            ) else "直线"
            local = LocalLineType(
                type_id=type_id,
                display_name="",
                line_type_index=1,
                atom_count=_atom_count_for_ops(serialized, set(row.op_indices)),
                op_indices=_sorted_unique(row.op_indices),
                model="legend_vector_sample",
                shape=shape,
                shape_detail="矢量 Legend 单行样例",
            )
            additions.append(local)
            added[(group.group_id, type_id)] = row

        line_types = tuple(
            replace(
                line_type,
                display_name=f"线型{index}",
                line_type_index=index,
            )
            for index, line_type in enumerate(
                sorted(
                    (*retained, *additions),
                    key=lambda item: min(item.op_indices, default=math.inf),
                ),
                start=1,
            )
        )
        assigned_paths = {
            op_index
            for line_type in line_types
            for op_index in line_type.op_indices
            if _path(operation_index, op_index) is not None
        }
        non_line_paths = {
            op_index
            for op_index in serialized.atom_op_indices
            if op_index not in assigned_paths
        }
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=_atom_count_for_ops(serialized, non_line_paths),
                op_indices=_sorted_unique(non_line_paths),
            ),
        ))

    groups = tuple(updated_groups)
    local_by_key = _index_local_types(groups)
    rebuilt_globals: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        members: list[GlobalLineTypeMember] = []
        for member in global_type.members:
            local = local_by_key.get((member.case_id, member.type_id))
            if local is None:
                continue
            members.append(replace(
                member,
                display_name=local.display_name,
                atom_count=local.atom_count,
                model=local.model,
                shape=local.shape,
                shape_detail=local.shape_detail,
            ))
        if members:
            rebuilt_globals.append(_rebuilt_global(
                global_type, members, local_by_key
            ))

    legend_buckets: dict[str, list[tuple[str, str]]] = {}
    for local_key, row in added.items():
        legend_buckets.setdefault(row.signature, []).append(local_key)
    for local_keys in legend_buckets.values():
        members = tuple(
            GlobalLineTypeMember(
                case_id=group_id,
                type_id=type_id,
                display_name=local_by_key[(group_id, type_id)].display_name,
                atom_count=local_by_key[(group_id, type_id)].atom_count,
                model=local_by_key[(group_id, type_id)].model,
                shape=local_by_key[(group_id, type_id)].shape,
                shape_detail=local_by_key[(group_id, type_id)].shape_detail,
            )
            for group_id, type_id in local_keys
        )
        rebuilt_globals.append(GlobalLineType(
            global_type_id="",
            signature_family="legend_vector_sample",
            minimum_pair_similarity=1.0,
            op_indices=_sorted_unique(
                op_index
                for local_key in local_keys
                for op_index in local_by_key[local_key].op_indices
            ),
            members=members,
        ))

    global_types = _renumber_globals(rebuilt_globals)
    local_count = _count_local_types(groups)
    return replace(
        recognition,
        groups=groups,
        global_types=global_types,
        summary=replace(
            recognition.summary,
            local_line_type_count=local_count,
            signed_periodic_type_count=local_count,
            unsigned_periodic_type_count=0,
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in global_types
            ),
        ),
    )


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


def _straight_legend_carrier(operation: PathOperationIR) -> bool:
    if any(segment.kind in {"curve", "close"} for segment in operation.segments):
        return False
    ink_length = _path_ink_length(operation)
    width = operation.bounds.width
    height = operation.bounds.height
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    for segment in operation.segments:
        if segment.kind == "move":
            if start is None:
                start = segment.end
            end = segment.end
        elif segment.kind == "line":
            end = segment.end
    chord = math.dist(start, end) if start is not None and end is not None else 0.0
    return (
        ink_length >= operation.line_width * 20.0
        and max(width, height) >= max(0.001, min(width, height)) * 8.0
        and chord / max(0.001, ink_length) >= 0.95
    )


def _legend_description_for_path(
    path: PathOperationIR,
    texts: Sequence[TextOperationIR],
) -> str | None:
    vertical = path.bounds.height >= path.bounds.width
    path_center = _bounds_center(path)
    path_start = path.bounds.min_y if vertical else path.bounds.min_x
    path_end = path.bounds.max_y if vertical else path.bounds.max_x
    path_major = max(0.001, path_end - path_start)
    candidates: list[tuple[float, str]] = []
    for text in texts:
        label = _normalized_line_label(text.literal_text)
        center = _bounds_center(text)
        text_start = text.bounds.min_y if vertical else text.bounds.min_x
        text_end = text.bounds.max_y if vertical else text.bounds.max_x
        longitudinal_gap = max(0.0, path_start - text_end, text_start - path_end)
        perpendicular_gap = abs(
            (path_center[0] if vertical else path_center[1])
            - (center[0] if vertical else center[1])
        )
        text_minor = text.bounds.width if vertical else text.bounds.height
        accepted = (
            label != "LEGEND"
            and _javascript_string_length(label) >= 4
            and longitudinal_gap <= max(8.0, path_major * 0.20)
            and perpendicular_gap <= max(18.0, text_minor * 2.0)
        )
        if accepted:
            candidates.append((longitudinal_gap + perpendicular_gap, label))
    candidates.sort(key=lambda candidate: candidate[0])
    return candidates[0][1] if candidates else None


def augment_legend_table_solid_samples(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Promote isolated solid Legend swatches with explicit text context."""

    operation_index = PageOperationIndex.build(page, grouping)
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    added_local_keys: dict[tuple[str, str], tuple[str, int]] = {}
    updated_groups: list[RecognizedGroup] = []

    for group in recognition.groups:
        try:
            dense_group_indices = operation_index.group_indices(group.group_id)
        except KeyError:
            updated_groups.append(group)
            continue
        texts = tuple(
            operation
            for op_index in dense_group_indices
            if isinstance(
                (operation := operation_index.operations[op_index]),
                TextOperationIR,
            )
        )
        if (
            not any(_normalized_line_label(text.literal_text) == "LEGEND" for text in texts)
            or len(texts) < 3
        ):
            updated_groups.append(group)
            continue
        candidates: list[tuple[int, PathOperationIR, str]] = []
        for op_index in group.non_linetype.op_indices:
            operation = _path(operation_index, op_index)
            if operation is None or not _straight_legend_carrier(operation):
                continue
            label = _legend_description_for_path(operation, texts)
            if label is not None:
                candidates.append((op_index, operation, label))
        if not candidates:
            updated_groups.append(group)
            continue

        serialized = serialized_by_group.get(group.group_id)
        used_type_ids = {line_type.type_id for line_type in group.line_types}

        def fresh_type_id(ordinal: int) -> str:
            suffix = ordinal
            type_id = f"type_legend_{suffix:03d}"
            while type_id in used_type_ids:
                suffix += 1
                type_id = f"type_legend_{suffix:03d}"
            used_type_ids.add(type_id)
            return type_id

        additions: list[LocalLineType] = []
        for index, (op_index, _operation, label) in enumerate(candidates, start=1):
            type_id = fresh_type_id(index)
            added_local_keys[(group.group_id, type_id)] = (label, op_index)
            atom_count = (
                sum(value == op_index for value in serialized.atom_op_indices)
                if serialized is not None
                else 0
            ) or 1
            additions.append(LocalLineType(
                type_id=type_id,
                display_name="",
                line_type_index=1,
                atom_count=atom_count,
                op_indices=(op_index,),
                model="legend_solid_sample",
                shape="直线",
                shape_detail=f"Legend 实线（{label}）",
            ))
        line_types = tuple(
            replace(
                line_type,
                display_name=f"线型{index}",
                line_type_index=index,
            )
            for index, line_type in enumerate(
                sorted(
                    (*group.line_types, *additions),
                    key=lambda item: min(item.op_indices, default=math.inf),
                ),
                start=1,
            )
        )
        promoted = {op_index for op_index, _operation, _label in candidates}
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=max(
                    0,
                    group.non_linetype.atom_count
                    - sum(line_type.atom_count for line_type in additions),
                ),
                op_indices=tuple(
                    op_index
                    for op_index in group.non_linetype.op_indices
                    if op_index not in promoted
                ),
            ),
        ))

    if not added_local_keys:
        return recognition
    groups = tuple(updated_groups)
    local_by_key = _index_local_types(groups)
    updated_globals = [
        replace(
            global_type,
            members=tuple(
                replace(
                    member,
                    display_name=(
                        local_by_key[(member.case_id, member.type_id)].display_name
                        if (member.case_id, member.type_id) in local_by_key
                        else member.display_name
                    ),
                )
                for member in global_type.members
            ),
        )
        for global_type in recognition.global_types
    ]
    for local_key, (_label, op_index) in added_local_keys.items():
        group_id, type_id = local_key
        local = local_by_key[local_key]
        updated_globals.append(GlobalLineType(
            global_type_id="",
            signature_family="legend_solid_sample",
            minimum_pair_similarity=1.0,
            op_indices=(op_index,),
            members=(GlobalLineTypeMember(
                case_id=group_id,
                type_id=type_id,
                display_name=local.display_name,
                atom_count=local.atom_count,
                model=local.model,
                shape=local.shape,
                shape_detail=local.shape_detail,
            ),),
        ))
    global_types = _renumber_globals(updated_globals)
    local_count = _count_local_types(groups)
    return replace(
        recognition,
        groups=groups,
        global_types=global_types,
        summary=replace(
            recognition.summary,
            local_line_type_count=local_count,
            signed_periodic_type_count=(
                recognition.summary.signed_periodic_type_count
                + local_count
                - recognition.summary.local_line_type_count
            ),
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in global_types
            ),
        ),
    )


__all__ = [
    "augment_legend_table_solid_samples",
    "augment_vector_legend_samples",
]
