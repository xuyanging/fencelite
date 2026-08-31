"""Frozen Method1 r10 decoded-text postprocessors.

This is the renderer-neutral Python port of ``augmentTextLabeledLineTypes``
and ``augmentShortInlineTextPatterns`` from the frozen TypeScript recognizer.
Result ``op_indices`` always use the dense position in ``PageIR.operations``.
PyMuPDF ``paint_order`` may repeat and is used only as authored sequence
evidence; it is never an ownership identity.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Iterable, Mapping, Sequence

from ..ir import BoundsIR, GroupingIR, PageIR, PathOperationIR, TextOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    RecognizedGroup,
)
from .serializer import SerializedGroup, serialize_path


_LOCAL_KEY_SEPARATOR = "\0"
_INLINE_FEET_LABEL = re.compile(r"^\d{1,3}(?:\.\d{1,3})?\s*['\u2019\u2032]$")


def _local_key(group_id: str, type_id: str) -> str:
    return f"{group_id}{_LOCAL_KEY_SEPARATOR}{type_id}"


def _sorted_unique(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _normalized_line_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).upper()


def _is_inline_feet_label(text: str) -> bool:
    """Whether text can be the numeric inline pattern protected below.

    The compound ownership exception exists for the narrow Method2 admission
    added for authored tokens such as ``8'``.  Ordinary text-labelled lines
    must retain frozen r10 competition semantics; protecting every compound
    type caused unrelated utility and hatch families to lose valid coverage
    across the regression corpus.
    """

    return _INLINE_FEET_LABEL.fullmatch(_normalized_line_label(text)) is not None


def _bounds_center(bounds: BoundsIR) -> tuple[float, float]:
    return (
        (bounds.min_x + bounds.max_x) / 2.0,
        (bounds.min_y + bounds.max_y) / 2.0,
    )


def _center_distance(left: BoundsIR, right: BoundsIR) -> float:
    left_center = _bounds_center(left)
    right_center = _bounds_center(right)
    return math.dist(left_center, right_center)


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _js_round(value: float) -> int:
    return math.floor(value + 0.5)


def _number(value: float) -> str:
    finite = float(value) if math.isfinite(float(value)) else 0.0
    normalized = 0.0 if abs(finite) < 1e-10 else finite
    fixed = f"{normalized:.6f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    return "0" if fixed in {"", "-0"} else fixed


def _color_key(color: tuple[float, ...] | None) -> str:
    if color is None:
        return "none"
    return ",".join(_number(channel) for channel in color)


def _path_style_key(operation: PathOperationIR) -> str:
    return "|".join((
        "S" if operation.stroke else "",
        "F" if operation.fill else "",
        _number(operation.line_width),
        _color_key(operation.stroke_color),
        _color_key(operation.fill_color),
    ))


def _text_label_key(operation: TextOperationIR) -> str:
    label = _normalized_line_label(operation.literal_text)
    # Frozen Scene identity uses the authored resource alias (for example F1
    # versus F2), not MuPDF's resolved BaseFont display name.  Legacy PageIR
    # has no source state and deliberately falls back through these properties.
    scale_bucket = _js_round(
        math.log(max(0.001, operation.canonical_font_size)) * 100.0
    )
    return _LOCAL_KEY_SEPARATOR.join(
        (label, operation.canonical_font_name, str(scale_bucket))
    )


def _index_local_types(
    groups: Sequence[RecognizedGroup],
) -> dict[str, LocalLineType]:
    return {
        _local_key(group.group_id, line_type.type_id): line_type
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


def _member_for_local(
    group_id: str,
    line_type: LocalLineType,
    previous: GlobalLineTypeMember | None = None,
) -> GlobalLineTypeMember:
    if previous is None:
        return GlobalLineTypeMember(
            case_id=group_id,
            type_id=line_type.type_id,
            display_name=line_type.display_name,
            atom_count=line_type.atom_count,
            model=line_type.model,
            shape=line_type.shape,
            shape_detail=line_type.shape_detail,
        )
    return replace(
        previous,
        case_id=group_id,
        type_id=line_type.type_id,
        display_name=line_type.display_name,
        atom_count=line_type.atom_count,
        model=line_type.model,
        shape=line_type.shape,
        shape_detail=line_type.shape_detail,
    )


@dataclass(frozen=True, slots=True)
class _PaintSequence:
    orders: tuple[int, ...]
    position_by_order: Mapping[int, int]
    operation_index: PageOperationIndex

    @classmethod
    def build(cls, operation_index: PageOperationIndex) -> "_PaintSequence":
        orders = tuple(operation_index.indices_by_paint_order)
        return cls(
            orders=orders,
            position_by_order={order: index for index, order in enumerate(orders)},
            operation_index=operation_index,
        )

    def batch_indices_at(self, position: int) -> tuple[int, ...]:
        if position < 0 or position >= len(self.orders):
            return ()
        return self.operation_index.indices_for_paint_order(self.orders[position])

    def nearby_batches(
        self,
        operation_index: int,
        radius: int,
    ) -> tuple[tuple[int, ...], ...]:
        operation = self.operation_index.operation(operation_index)
        position = self.position_by_order[operation.paint_order]
        return tuple(
            self.batch_indices_at(candidate)
            for candidate in range(
                max(0, position - radius),
                min(len(self.orders), position + radius + 1),
            )
        )

    def adjacent_batches(
        self,
        operation_index: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        operation = self.operation_index.operation(operation_index)
        position = self.position_by_order[operation.paint_order]
        return (
            self.batch_indices_at(position - 1),
            self.batch_indices_at(position + 1),
        )


@dataclass(frozen=True, slots=True)
class _TextContext:
    page: PageIR
    grouping: GroupingIR
    operation_index: PageOperationIndex
    paint_sequence: _PaintSequence
    serialized_by_group: Mapping[str, SerializedGroup]
    atom_multiplicity_by_group: Mapping[str, Mapping[int, int]]

    def atom_count(self, group_id: str, operation_indices: set[int]) -> int:
        multiplicity = self.atom_multiplicity_by_group[group_id]
        return sum(
            count
            for operation_index, count in multiplicity.items()
            if operation_index in operation_indices
        )

    def serialized_path_indices(self, group_id: str) -> set[int]:
        return set(self.atom_multiplicity_by_group[group_id])


def _build_context(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> _TextContext:
    operation_index = PageOperationIndex.build(page, grouping)
    grouping_ids = tuple(group.group_id for group in grouping.groups)
    recognition_ids = tuple(group.group_id for group in recognition.groups)
    serialized_ids = tuple(group.group_id for group in serialized_groups)
    if recognition_ids != grouping_ids:
        raise ValueError("recognition Groups do not match GroupingIR order")
    if serialized_ids != grouping_ids:
        raise ValueError("serialized Groups do not match GroupingIR order")

    for group in recognition.groups:
        owned = (
            *(index for line_type in group.line_types for index in line_type.op_indices),
            *group.non_linetype.op_indices,
        )
        for dense_index in owned:
            operation_index.operation(dense_index)
            if operation_index.group_id(dense_index) != group.group_id:
                raise ValueError(
                    f"group {group.group_id} owns dense operation {dense_index} "
                    "from a different Group"
                )
    for global_type in recognition.global_types:
        for dense_index in global_type.op_indices:
            operation_index.operation(dense_index)

    serialized_by_group = {group.group_id: group for group in serialized_groups}
    atom_multiplicity: dict[str, dict[int, int]] = {}
    for group_id, serialized in serialized_by_group.items():
        counts: dict[int, int] = {}
        for dense_index in operation_index.group_indices(group_id):
            operation = operation_index.operation(dense_index)
            if not isinstance(operation, PathOperationIR):
                continue
            serialized_path = serialize_path(operation)
            if serialized_path is None:
                continue
            counts[dense_index] = serialized_path[1]
        if sum(counts.values()) != len(serialized.atom_op_indices):
            raise ValueError(
                f"serialized Group {group_id} atom count does not match dense PageIR paths"
            )
        atom_multiplicity[group_id] = counts

    return _TextContext(
        page=page,
        grouping=grouping,
        operation_index=operation_index,
        paint_sequence=_PaintSequence.build(operation_index),
        serialized_by_group=serialized_by_group,
        atom_multiplicity_by_group=atom_multiplicity,
    )


def _validated(result: LineTypeRecognitionResult) -> LineTypeRecognitionResult:
    return LineTypeRecognitionResult.from_dict(result.to_dict())


@dataclass(frozen=True, slots=True)
class _TextLabelSeed:
    global_type_id: str
    label_key: str
    label: str
    support: int
    confidence: float
    period: float
    allowed_path_styles: frozenset[str]


def _paint_station_bounds(
    context: _TextContext,
    operation_indices: Sequence[int],
) -> tuple[BoundsIR, ...]:
    """Collapse shared-paint spans only for authored sequence evidence.

    Every span keeps its dense ownership index.  The union is used solely to
    stop one PyMuPDF text paint event from impersonating several repetitions.
    """

    stations: list[BoundsIR] = []
    previous_order: int | None = None
    for dense_index in sorted(
        operation_indices,
        key=lambda item: _paint_order_key(context, item),
    ):
        operation = context.operation_index.operation(dense_index)
        if operation.paint_order != previous_order:
            stations.append(operation.bounds)
            previous_order = operation.paint_order
        else:
            stations[-1] = stations[-1].union(operation.bounds)
    return tuple(stations)


def _periodic_text_run(
    context: _TextContext,
    operation_indices: Sequence[int],
    period: float,
) -> bool:
    stations = _paint_station_bounds(context, operation_indices)
    if len(stations) < 2 or period <= 0.0:
        return False
    inliers = 0
    for left, right in zip(stations, stations[1:]):
        spacing = _center_distance(left, right)
        multiplier = max(1, min(3, _js_round(spacing / period)))
        if abs(spacing / multiplier - period) / period <= 0.20:
            inliers += 1
    return inliers >= max(1, math.ceil((len(stations) - 1) * 0.70))


def _contains_path(
    context: _TextContext,
    dense_indices: Iterable[int],
    allowed: set[int],
) -> bool:
    return any(
        dense_index in allowed
        and isinstance(context.operation_index.operation(dense_index), PathOperationIR)
        for dense_index in dense_indices
    )


def _paint_order_key(
    context: _TextContext,
    dense_index: int,
) -> tuple[int, int]:
    return (context.operation_index.operation(dense_index).paint_order, dense_index)


def _has_intervening_text(
    context: _TextContext,
    left_index: int,
    right_index: int,
) -> bool:
    left_order = context.operation_index.operation(left_index).paint_order
    right_order = context.operation_index.operation(right_index).paint_order
    return any(
        left_order < context.operation_index.operation(dense_index).paint_order < right_order
        and isinstance(context.operation_index.operation(dense_index), TextOperationIR)
        for dense_index in range(left_index + 1, right_index)
    )


def _candidate_runs(
    context: _TextContext,
    matching: Sequence[int],
) -> tuple[tuple[int, ...], ...]:
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for dense_index in sorted(matching, key=lambda item: _paint_order_key(context, item)):
        if current and _has_intervening_text(context, current[-1], dense_index):
            runs.append(tuple(current))
            current = []
        current.append(dense_index)
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def _expanded_candidate_indices(
    context: _TextContext,
    group_id: str,
    candidate: Sequence[int],
) -> tuple[int, ...]:
    group_indices = context.operation_index.group_indices(group_id)
    group_set = set(group_indices)
    orders = tuple(dict.fromkeys(
        context.operation_index.operation(index).paint_order
        for index in group_indices
    ))
    position_by_order = {order: position for position, order in enumerate(orders)}
    start_order = context.operation_index.operation(candidate[0]).paint_order
    end_order = context.operation_index.operation(candidate[-1]).paint_order
    start_position = position_by_order[start_order]
    end_position = position_by_order[end_order]

    while start_position > 0:
        previous_order = orders[start_position - 1]
        previous_batch = tuple(
            index
            for index in context.operation_index.indices_for_paint_order(previous_order)
            if index in group_set
        )
        if any(
            isinstance(context.operation_index.operation(index), TextOperationIR)
            for index in previous_batch
        ):
            break
        start_position -= 1
    while end_position + 1 < len(orders):
        next_order = orders[end_position + 1]
        next_batch = tuple(
            index
            for index in context.operation_index.indices_for_paint_order(next_order)
            if index in group_set
        )
        if any(
            isinstance(context.operation_index.operation(index), TextOperationIR)
            for index in next_batch
        ):
            break
        end_position += 1

    allowed_orders = set(orders[start_position:end_position + 1])
    return tuple(
        index
        for index in group_indices
        if context.operation_index.operation(index).paint_order in allowed_orders
    )


def augment_text_labeled_line_types(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Complete proven carrier families with repeated decoded text tokens."""

    context = _build_context(page, grouping, serialized_groups, recognition)
    text_operations = tuple(
        (dense_index, operation)
        for dense_index, operation in context.operation_index.text_items()
        if len(_normalized_line_label(operation.literal_text)) >= 2
    )
    if not text_operations or not recognition.global_types:
        return recognition

    candidate_seeds: list[_TextLabelSeed] = []
    for global_type in recognition.global_types:
        path_indices = {
            dense_index
            for dense_index in global_type.op_indices
            if isinstance(context.operation_index.operation(dense_index), PathOperationIR)
        }
        if len(path_indices) < 5:
            continue
        buckets: dict[str, list[tuple[int, TextOperationIR]]] = {}
        for dense_index, operation in text_operations:
            previous_batch, next_batch = context.paint_sequence.adjacent_batches(dense_index)
            if not (
                _contains_path(context, previous_batch, path_indices)
                or _contains_path(context, next_batch, path_indices)
            ):
                continue
            buckets.setdefault(_text_label_key(operation), []).append(
                (dense_index, operation)
            )

        for label_key, occurrences in buckets.items():
            support = len({
                operation.paint_order for _dense_index, operation in occurrences
            })
            confidence = support / len(path_indices)
            if support < 5 or confidence < 0.50:
                continue
            by_group: dict[str, list[tuple[int, TextOperationIR]]] = {}
            for occurrence in occurrences:
                by_group.setdefault(
                    context.operation_index.group_id(occurrence[0]), []
                ).append(occurrence)
            spacings: list[float] = []
            for group_occurrences in by_group.values():
                stations = _paint_station_bounds(
                    context,
                    tuple(dense_index for dense_index, _op in group_occurrences),
                )
                spacings.extend(
                    _center_distance(left, right)
                    for left, right in zip(stations, stations[1:])
                )
            period = _median(spacing for spacing in spacings if spacing > 0.0)
            if period <= 0.0:
                continue
            allowed_styles: set[str] = set()
            for dense_index, _operation in occurrences:
                for batch in context.paint_sequence.nearby_batches(dense_index, 2):
                    for nearby_index in batch:
                        nearby = context.operation_index.operation(nearby_index)
                        if isinstance(nearby, PathOperationIR):
                            allowed_styles.add(_path_style_key(nearby))
            candidate_seeds.append(_TextLabelSeed(
                global_type_id=global_type.global_type_id,
                label_key=label_key,
                label=_normalized_line_label(occurrences[0][1].literal_text),
                support=support,
                confidence=confidence,
                period=period,
                allowed_path_styles=frozenset(allowed_styles),
            ))

    if not candidate_seeds:
        return recognition

    seed_by_label: dict[str, _TextLabelSeed] = {}
    ambiguous_labels: set[str] = set()
    for seed in sorted(
        candidate_seeds,
        key=lambda item: (-item.support, -item.confidence),
    ):
        if seed.label_key in ambiguous_labels:
            continue
        previous = seed_by_label.get(seed.label_key)
        if previous is None:
            seed_by_label[seed.label_key] = seed
        elif previous.global_type_id != seed.global_type_id:
            ratio = max(previous.period, seed.period) / max(
                0.001, min(previous.period, seed.period)
            )
            if ratio > 1.20:
                seed_by_label.pop(seed.label_key, None)
                ambiguous_labels.add(seed.label_key)

    result = recognition
    changed = False
    for seed in seed_by_label.values():
        target = next((
            global_type
            for global_type in result.global_types
            if global_type.global_type_id == seed.global_type_id
        ), None)
        if target is None:
            continue
        target_path_indices = {
            dense_index
            for dense_index in target.op_indices
            if isinstance(
                context.operation_index.operation(dense_index),
                PathOperationIR,
            )
        }
        protected_path_indices = (
            {
                dense_index
                for global_type in result.global_types
                if global_type.signature_family == "compound_path_periodic"
                and global_type.group_count >= 2
                for dense_index in global_type.op_indices
                if isinstance(
                    context.operation_index.operation(dense_index),
                    PathOperationIR,
                )
            }
            if _is_inline_feet_label(seed.label)
            else set()
        )
        matching_by_group: dict[str, list[int]] = {}
        for dense_index, operation in text_operations:
            if _text_label_key(operation) != seed.label_key:
                continue
            matching_by_group.setdefault(
                context.operation_index.group_id(dense_index), []
            ).append(dense_index)

        recovered_by_group: dict[str, tuple[set[int], set[int]]] = {}
        for group_id, matching in matching_by_group.items():
            recovered_paths: set[int] = set()
            recovered_texts: set[int] = set()
            for candidate in _candidate_runs(context, matching):
                if not _periodic_text_run(context, candidate, seed.period):
                    continue
                recovered_texts.update(candidate)
                for dense_index in _expanded_candidate_indices(
                    context, group_id, candidate
                ):
                    operation = context.operation_index.operation(dense_index)
                    if (
                        isinstance(operation, PathOperationIR)
                        and _path_style_key(operation) in seed.allowed_path_styles
                        # Ordinary text labels still resolve local competition
                        # exactly as frozen r10 did.  A short feet token admitted
                        # as an inline Method2 pattern may not steal a path from
                        # a compound identity already proven in multiple Groups:
                        # on final_plans P3, repeated 8' labels consumed the
                        # square-post SIDELINE FENCE motif after compound
                        # matching had joined both areas.
                        and (
                            dense_index not in protected_path_indices
                            or dense_index in target_path_indices
                        )
                    ):
                        recovered_paths.add(dense_index)
            if len(recovered_texts) >= 2 and len(recovered_paths) >= 2:
                recovered_by_group[group_id] = (
                    recovered_paths,
                    recovered_texts,
                )
        if not recovered_by_group:
            continue

        original_target_members = {
            member.case_id: member.type_id for member in target.members
        }
        updated_groups: list[RecognizedGroup] = []
        for group in result.groups:
            recovered = recovered_by_group.get(group.group_id)
            if recovered is None:
                updated_groups.append(group)
                continue
            recovered_paths, recovered_texts = recovered
            existing_target_id = original_target_members.get(group.group_id)
            existing_target = next((
                line_type
                for line_type in group.line_types
                if line_type.type_id == existing_target_id
            ), None)
            target_paths = set(recovered_paths)
            if existing_target is not None:
                target_paths.update(
                    dense_index
                    for dense_index in existing_target.op_indices
                    if isinstance(
                        context.operation_index.operation(dense_index),
                        PathOperationIR,
                    )
                )

            retained: list[LocalLineType] = []
            for line_type in group.line_types:
                if line_type.type_id == existing_target_id:
                    continue
                operation_indices = tuple(
                    index
                    for index in line_type.op_indices
                    if index not in target_paths
                )
                remaining_paths = {
                    index
                    for index in operation_indices
                    if isinstance(
                        context.operation_index.operation(index), PathOperationIR
                    )
                }
                atom_count = context.atom_count(group.group_id, remaining_paths)
                if atom_count > 0:
                    retained.append(replace(
                        line_type,
                        atom_count=atom_count,
                        op_indices=operation_indices,
                    ))

            used_type_ids = {line_type.type_id for line_type in group.line_types}
            recovered_type_id = (
                existing_target.type_id
                if existing_target is not None
                else "type_text_001"
            )
            suffix = 1
            while existing_target is None and recovered_type_id in used_type_ids:
                recovered_type_id = f"type_text_{suffix:03d}"
                suffix += 1
            target_line_type = LocalLineType(
                type_id=recovered_type_id,
                display_name="",
                line_type_index=1,
                atom_count=context.atom_count(group.group_id, target_paths),
                op_indices=_sorted_unique((*target_paths, *recovered_texts)),
                model="text_labeled_carrier_chain",
                shape=existing_target.shape if existing_target is not None else "非直线",
                shape_detail=(
                    existing_target.shape_detail
                    if existing_target is not None
                    else f"文字周期（{seed.label}）"
                ),
            )
            line_types = tuple(
                replace(
                    line_type,
                    display_name=f"线型{index}",
                    line_type_index=index,
                )
                for index, line_type in enumerate(
                    (*retained, target_line_type), start=1
                )
            )
            assigned_paths = {
                dense_index
                for line_type in line_types
                for dense_index in line_type.op_indices
                if isinstance(
                    context.operation_index.operation(dense_index), PathOperationIR
                )
            }
            non_line_paths = (
                context.serialized_path_indices(group.group_id) - assigned_paths
            )
            updated_groups.append(replace(
                group,
                line_types=line_types,
                non_linetype=replace(
                    group.non_linetype,
                    atom_count=context.atom_count(group.group_id, non_line_paths),
                    op_indices=_sorted_unique(non_line_paths),
                ),
            ))

        groups = tuple(updated_groups)
        local_by_key = _index_local_types(groups)
        target_members: dict[str, GlobalLineTypeMember] = {
            member.case_id: member for member in target.members
        }
        group_by_id = {group.group_id: group for group in groups}
        for group_id in recovered_by_group:
            group = group_by_id[group_id]
            target_type_id = original_target_members.get(group_id)
            line_type = next((
                candidate
                for candidate in group.line_types
                if candidate.type_id == target_type_id
                or candidate.model == "text_labeled_carrier_chain"
            ), None)
            if line_type is None:
                continue
            target_members[group_id] = _member_for_local(
                group_id,
                line_type,
                target_members.get(group_id),
            )

        updated_globals: list[GlobalLineType] = []
        for global_type in result.global_types:
            if global_type.global_type_id == target.global_type_id:
                members = tuple(target_members.values())
            else:
                members = tuple(
                    member
                    for member in global_type.members
                    if _local_key(member.case_id, member.type_id) in local_by_key
                )
            if not members:
                continue
            operation_indices = _sorted_unique(
                dense_index
                for member in members
                for dense_index in local_by_key[
                    _local_key(member.case_id, member.type_id)
                ].op_indices
            )
            updated_globals.append(replace(
                global_type,
                minimum_pair_similarity=(
                    min(global_type.minimum_pair_similarity, 0.99)
                    if global_type.global_type_id == target.global_type_id
                    else global_type.minimum_pair_similarity
                ),
                op_indices=operation_indices,
                members=members,
            ))

        local_count = sum(group.line_type_count for group in groups)
        global_types = tuple(updated_globals)
        result = replace(
            result,
            groups=groups,
            global_types=global_types,
            summary=replace(
                result.summary,
                local_line_type_count=local_count,
                signed_periodic_type_count=max(
                    0,
                    result.summary.signed_periodic_type_count
                    + local_count
                    - result.summary.local_line_type_count,
                ),
                global_type_count=len(global_types),
                cross_group_global_type_count=sum(
                    global_type.group_count > 1 for global_type in global_types
                ),
            ),
        )
        changed = True

    if not changed:
        return recognition
    return _validated(replace(result, global_types=_renumber_globals(result.global_types)))


@dataclass(frozen=True, slots=True)
class _InlineTextPattern:
    label: str
    text_operation_indices: tuple[int, ...]


def _point_to_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-12:
        return math.dist(point, start)
    projection = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / length_squared
    clamped = max(0.0, min(1.0, projection))
    return math.hypot(
        point[0] - (start[0] + clamped * dx),
        point[1] - (start[1] + clamped * dy),
    )


def _nearest_bracketing_path(
    context: _TextContext,
    text_index: int,
    path_indices: set[int],
    direction: int,
) -> int | None:
    operation = context.operation_index.operation(text_index)
    position = context.paint_sequence.position_by_order[operation.paint_order]
    for offset in range(1, 4):
        batch_position = position + direction * offset
        batch = context.paint_sequence.batch_indices_at(batch_position)
        candidates = tuple(
            dense_index
            for dense_index in batch
            if dense_index in path_indices
            and isinstance(
                context.operation_index.operation(dense_index), PathOperationIR
            )
        )
        if candidates:
            return max(candidates) if direction < 0 else min(candidates)
    return None


def _inline_text_pattern_for(
    context: _TextContext,
    group_id: str,
    line_type: LocalLineType,
) -> _InlineTextPattern | None:
    path_indices = {
        dense_index
        for dense_index in line_type.op_indices
        if isinstance(context.operation_index.operation(dense_index), PathOperationIR)
    }
    if len(path_indices) < 2:
        return None
    candidates: dict[str, list[int]] = {}
    for dense_index in context.operation_index.group_indices(group_id):
        operation = context.operation_index.operation(dense_index)
        if not isinstance(operation, TextOperationIR):
            continue
        label = _normalized_line_label(operation.literal_text)
        if len(label) < 1 or len(label) > 16 or re.search(r"[A-Z]", label) is None:
            continue
        previous_path = _nearest_bracketing_path(
            context, dense_index, path_indices, -1
        )
        next_path = _nearest_bracketing_path(
            context, dense_index, path_indices, 1
        )
        if previous_path is None or next_path is None:
            continue
        previous = context.operation_index.operation(previous_path)
        next_operation = context.operation_index.operation(next_path)
        assert isinstance(previous, PathOperationIR)
        assert isinstance(next_operation, PathOperationIR)
        text_center = _bounds_center(operation.bounds)
        previous_center = _bounds_center(previous.bounds)
        next_center = _bounds_center(next_operation.bounds)
        token_size = max(0.001, operation.bounds.width, operation.bounds.height)
        carrier_distance = _center_distance(previous.bounds, next_operation.bounds)
        if carrier_distance <= token_size * 0.35:
            continue
        if (
            _point_to_segment_distance(text_center, previous_center, next_center)
            > token_size * 0.75 + 2.0
        ):
            continue
        candidates.setdefault(label, []).append(dense_index)
    if len(candidates) != 1:
        return None
    label, text_indices = next(iter(candidates.items()))
    return _InlineTextPattern(label, tuple(text_indices))


def augment_short_inline_text_patterns(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Give bracketed one-period samples a decoded-text-aware identity."""

    context = _build_context(page, grouping, serialized_groups, recognition)
    patterns: dict[str, _InlineTextPattern] = {}
    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        assigned_texts: set[int] = set()
        line_types: list[LocalLineType] = []
        for line_type in group.line_types:
            pattern = _inline_text_pattern_for(
                context, group.group_id, line_type
            )
            if pattern is None:
                line_types.append(line_type)
                continue
            patterns[_local_key(group.group_id, line_type.type_id)] = pattern
            assigned_texts.update(pattern.text_operation_indices)
            detail = f"文字周期（{pattern.label}）"
            line_types.append(replace(
                line_type,
                op_indices=_sorted_unique((
                    *line_type.op_indices,
                    *pattern.text_operation_indices,
                )),
                model=(
                    line_type.model
                    if line_type.model == "text_labeled_carrier_chain"
                    else "short_text_labeled_pattern"
                ),
                shape_detail=(
                    line_type.shape_detail
                    if detail in line_type.shape_detail
                    else f"{line_type.shape_detail} · {detail}"
                ),
            ))
        updated_groups.append(replace(
            group,
            line_types=tuple(line_types),
            non_linetype=replace(
                group.non_linetype,
                op_indices=tuple(
                    index
                    for index in group.non_linetype.op_indices
                    if index not in assigned_texts
                ),
            ),
        ))
    if not patterns:
        return recognition

    groups = tuple(updated_groups)
    local_by_key = _index_local_types(groups)
    rebuilt_globals: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        buckets: dict[str, list[GlobalLineTypeMember]] = {}
        distinct_labels: set[str] = set()
        for member in global_type.members:
            local_key = _local_key(member.case_id, member.type_id)
            pattern = patterns.get(local_key)
            bucket_key = f"label:{pattern.label}" if pattern is not None else "unlabeled"
            if pattern is not None:
                distinct_labels.add(pattern.label)
            local = local_by_key.get(local_key)
            updated_member = (
                replace(
                    member,
                    atom_count=local.atom_count,
                    model=local.model,
                    shape=local.shape,
                    shape_detail=local.shape_detail,
                )
                if local is not None
                else member
            )
            buckets.setdefault(bucket_key, []).append(updated_member)

        if distinct_labels and len(buckets) >= 2:
            partitions = tuple(
                (bucket_key, tuple(members))
                for bucket_key, members in buckets.items()
            )
        else:
            partitions = ((
                "combined",
                tuple(member for members in buckets.values() for member in members),
            ),)

        for partition_index, (bucket_key, members) in enumerate(partitions):
            rebuilt_globals.append(replace(
                global_type,
                global_type_id=(
                    global_type.global_type_id if partition_index == 0 else ""
                ),
                signature_family=(
                    "text_labeled_short_period"
                    if bucket_key.startswith("label:")
                    else global_type.signature_family
                ),
                minimum_pair_similarity=(
                    min(global_type.minimum_pair_similarity, 0.99)
                    if bucket_key.startswith("label:")
                    else global_type.minimum_pair_similarity
                ),
                op_indices=_sorted_unique(
                    dense_index
                    for member in members
                    for dense_index in local_by_key[
                        _local_key(member.case_id, member.type_id)
                    ].op_indices
                ),
                members=members,
            ))

    global_types = _renumber_globals(rebuilt_globals)
    result = replace(
        recognition,
        groups=groups,
        global_types=global_types,
        summary=replace(
            recognition.summary,
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in global_types
            ),
        ),
    )
    return _validated(result)


__all__ = [
    "augment_short_inline_text_patterns",
    "augment_text_labeled_line_types",
]
