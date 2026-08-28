"""Frozen Method1 r10 outlined-text and riding-stitch postprocessors.

The implementation is a direct Python migration of the two stages in
``line-type-engine/method1/recognizer.ts``.  It consumes only canonical
``PageIR`` geometry, serialized atom ownership, and algorithm-neutral results;
there is no viewer, browser, PDF parser, cache, or TypeScript dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

from ..ir import BoundsIR, GroupingIR, PageIR, PathOperationIR
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


Point = tuple[float, float]


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


def _bounds_gap(left: BoundsIR, right: BoundsIR) -> float:
    dx = max(0.0, left.min_x - right.max_x, right.min_x - left.max_x)
    dy = max(0.0, left.min_y - right.max_y, right.min_y - left.max_y)
    return math.hypot(dx, dy)


def _bounds_center(bounds: BoundsIR) -> Point:
    return ((bounds.min_x + bounds.max_x) / 2.0, (bounds.min_y + bounds.max_y) / 2.0)


def _operation_paths(page: PageIR) -> dict[int, PathOperationIR]:
    """Index paths by their unique result identity, never by paint order."""

    return {
        dense_index: operation
        for dense_index, operation in enumerate(page.operations)
        if isinstance(operation, PathOperationIR)
    }


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


def _js_number(value: float) -> str:
    if not math.isfinite(value):
        return "0"
    if abs(value) < 1e-10:
        value = 0.0
    fixed = f"{value:.6f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    return "0" if fixed in {"", "-0"} else fixed


def _rgb(color: tuple[float, ...] | None) -> tuple[float, float, float]:
    if not color:
        return (0.0, 0.0, 0.0)
    channels = tuple(max(0.0, min(1.0, float(value))) for value in color)
    if len(channels) == 1:
        return (channels[0], channels[0], channels[0])
    if len(channels) >= 4:
        cyan, magenta, yellow, black = channels[:4]
        return (
            1.0 - min(1.0, cyan + black),
            1.0 - min(1.0, magenta + black),
            1.0 - min(1.0, yellow + black),
        )
    if len(channels) == 2:
        return (channels[0], channels[1], 0.0)
    return (channels[0], channels[1], channels[2])


def _css_rgb(color: tuple[float, ...] | None) -> str:
    channels = _rgb(color)
    # JavaScript Math.round is floor(x + 0.5) for the non-negative channels.
    integers = tuple(math.floor(channel * 255.0 + 0.5) for channel in channels)
    return f"rgb({integers[0]} {integers[1]} {integers[2]})"


def _path_style_key(operation: PathOperationIR) -> str:
    return "|".join((
        "S" if operation.stroke else "",
        "F" if operation.fill else "",
        _js_number(operation.line_width),
        _css_rgb(operation.stroke_color),
        _css_rgb(operation.fill_color),
    ))


def _is_outlined_stroke_text_type(
    paths: dict[int, PathOperationIR],
    line_type: LocalLineType,
) -> bool:
    if line_type.model != "compound_path_chain":
        return False
    path_ops = sorted(
        (
            (op_index, paths[op_index])
            for op_index in line_type.op_indices
            if op_index in paths
        ),
        key=lambda item: item[0],
    )
    if len(path_ops) < 20:
        return False
    curve_path_count = sum(
        any(segment.kind == "curve" for segment in operation.segments)
        for _, operation in path_ops
    )
    if curve_path_count > len(path_ops) * 0.05:
        return False

    union = path_ops[0][1].bounds
    for _, operation in path_ops[1:]:
        union = union.union(operation.bounds)
    union_width = union.width
    union_height = union.height
    union_major = max(union_width, union_height)
    union_minor = min(union_width, union_height)
    if union_major <= 0 or union_minor / union_major < 0.10:
        return False
    median_span = _median(
        max(operation.bounds.width, operation.bounds.height)
        for _, operation in path_ops
    )
    if median_span > union_major * 0.08:
        return False

    segment_count_frequency: dict[int, int] = {}
    for _, operation in path_ops:
        count = len(operation.segments)
        segment_count_frequency[count] = segment_count_frequency.get(count, 0) + 1
    dominant_segment_count_ratio = max(
        (0, *segment_count_frequency.values())
    ) / max(1, len(path_ops))
    median_segment_count = _median(
        len(operation.segments) for _, operation in path_ops
    )
    if median_segment_count >= 6 and dominant_segment_count_ratio >= 0.75:
        return False

    touching_pairs = 0
    for index in range(1, len(path_ops)):
        previous = path_ops[index - 1][1]
        current = path_ops[index][1]
        tolerance = max(0.25, max(previous.line_width, current.line_width) * 0.50)
        if _bounds_gap(previous.bounds, current.bounds) <= tolerance:
            touching_pairs += 1
    if touching_pairs / max(1, len(path_ops) - 1) < 0.55:
        return False

    angle_bins = [0] * 12
    segment_count = 0
    for _, operation in path_ops:
        current: Point | None = None
        for segment in operation.segments:
            if segment.kind == "move":
                current = segment.end
            elif segment.kind == "line":
                if current is not None and segment.end is not None:
                    dx = segment.end[0] - current[0]
                    dy = segment.end[1] - current[1]
                    if math.hypot(dx, dy) > 1e-6:
                        angle = math.atan2(dy, dx)
                        if angle < 0:
                            angle += math.pi
                        if angle >= math.pi:
                            angle -= math.pi
                        bin_index = min(11, math.floor(angle / math.pi * len(angle_bins)))
                        angle_bins[bin_index] += 1
                        segment_count += 1
                current = segment.end
    if segment_count < 20:
        return False
    ordered_bins = sorted(angle_bins, reverse=True)
    used_bins = sum(count > 0 for count in ordered_bins)
    dominant_fraction = ordered_bins[0] / segment_count
    top_two_fraction = (ordered_bins[0] + ordered_bins[1]) / segment_count
    return (
        used_bins >= 5
        and dominant_fraction <= 0.45
        and top_two_fraction <= 0.75
    )


def demote_outlined_stroke_text_line_types(
    page: PageIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Demote path-only CAD stroke labels mistaken for periodic line types."""

    paths = _operation_paths(page)
    removed_local_keys = {
        (group.group_id, line_type.type_id)
        for group in recognition.groups
        for line_type in group.line_types
        if _is_outlined_stroke_text_type(paths, line_type)
    }
    if not removed_local_keys:
        return recognition

    serialized_by_group = {group.group_id: group for group in serialized_groups}
    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        serialized = serialized_by_group.get(group.group_id)
        if serialized is None:
            updated_groups.append(group)
            continue
        line_types = tuple(
            replace(
                line_type,
                display_name=f"线型{index}",
                line_type_index=index,
            )
            for index, line_type in enumerate(
                (
                    line_type
                    for line_type in group.line_types
                    if (group.group_id, line_type.type_id) not in removed_local_keys
                ),
                start=1,
            )
        )
        assigned_paths = {
            op_index
            for line_type in line_types
            for op_index in line_type.op_indices
            if op_index in paths
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

    local_by_key = _index_local_types(updated_groups)
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
        if not members:
            continue
        op_indices = _sorted_unique(
            op_index
            for member in members
            for op_index in local_by_key[(member.case_id, member.type_id)].op_indices
        )
        rebuilt_globals.append(replace(
            global_type,
            op_indices=op_indices,
            members=tuple(members),
        ))
    global_types = _renumber_globals(rebuilt_globals)
    groups = tuple(updated_groups)
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


@dataclass(frozen=True, slots=True)
class _StitchCandidate:
    group_id: str
    op_index: int
    style: str
    center: Point
    tangent: Point
    scale: float
    outer_inner_ratio: float


@dataclass(frozen=True, slots=True)
class _StitchRun:
    candidates: tuple[_StitchCandidate, ...]
    outer_inner_ratio: float


@dataclass(frozen=True, slots=True)
class _StitchIdentity:
    local_key: tuple[str, str]
    group_id: str
    style: str
    outer_inner_ratio: float
    op_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _StitchAddition:
    style: str
    outer_inner_ratio: float
    op_indices: tuple[int, ...]


def _normalized_vector(x: float, y: float) -> Point | None:
    length = math.hypot(x, y)
    return (x / length, y / length) if length > 1e-9 else None


def _dot(left: Point, right: Point) -> float:
    return left[0] * right[0] + left[1] * right[1]


def _parallel_cosine(left: Point, right: Point) -> float:
    return abs(_dot(left, right))


def _pair_ratio(left: float, right: float) -> float:
    return max(left, right) / max(0.001, min(left, right))


def _repeated_stitch_candidate(
    paths: dict[int, PathOperationIR],
    group_id: str,
    op_index: int,
    page_diagonal: float,
) -> _StitchCandidate | None:
    operation = paths.get(op_index)
    if operation is None or not operation.stroke or operation.fill:
        return None
    points: list[Point] = []
    for segment in operation.segments:
        if segment.kind == "move":
            if points or segment.end is None:
                return None
            points.append(segment.end)
        elif segment.kind == "line":
            if not points or segment.end is None:
                return None
            points.append(segment.end)
        else:
            return None
    if len(points) != 5:
        return None
    vectors = tuple(
        (points[index + 1][0] - point[0], points[index + 1][1] - point[1])
        for index, point in enumerate(points[:-1])
    )
    lengths = tuple(math.hypot(*vector) for vector in vectors)
    minimum_length = max(
        0.02,
        operation.line_width * 0.35,
        page_diagonal * 0.00001,
    )
    if any(length < minimum_length for length in lengths):
        return None
    units = tuple(_normalized_vector(*vector) for vector in vectors)
    if any(unit is None for unit in units):
        return None
    outer_start, inner_start, inner_end, outer_end = units
    assert all(unit is not None for unit in units)
    same_direction_limit = math.cos(math.radians(22))
    if _dot(outer_start, outer_end) < same_direction_limit:  # type: ignore[arg-type]
        return None
    if _dot(inner_start, inner_end) < same_direction_limit:  # type: ignore[arg-type]
        return None
    outer = _normalized_vector(
        outer_start[0] + outer_end[0],  # type: ignore[index]
        outer_start[1] + outer_end[1],  # type: ignore[index]
    )
    tangent = _normalized_vector(
        inner_start[0] + inner_end[0],  # type: ignore[index]
        inner_start[1] + inner_end[1],  # type: ignore[index]
    )
    if outer is None or tangent is None or abs(_dot(outer, tangent)) > 0.46:
        return None
    if _pair_ratio(lengths[0], lengths[3]) > 1.45:
        return None
    if _pair_ratio(lengths[1], lengths[2]) > 1.45:
        return None
    outer_length = (lengths[0] + lengths[3]) / 2.0
    inner_length = (lengths[1] + lengths[2]) / 2.0
    outer_inner_ratio = outer_length / max(0.001, inner_length)
    if outer_inner_ratio < 1.10 or outer_inner_ratio > 3.25:
        return None
    scale = max(operation.bounds.width, operation.bounds.height)
    if scale < page_diagonal * 0.00015 or scale > page_diagonal * 0.012:
        return None
    ink_length = sum(lengths)
    chord = math.dist(points[-1], points[0])
    chord_ratio = chord / max(0.001, ink_length)
    if chord_ratio < 0.45 or chord_ratio > 0.88:
        return None
    return _StitchCandidate(
        group_id=group_id,
        op_index=op_index,
        style=_path_style_key(operation),
        center=_bounds_center(operation.bounds),
        tangent=tangent,
        scale=scale,
        outer_inner_ratio=outer_inner_ratio,
    )


def _stitch_geometry_connects(
    left: _StitchCandidate,
    right: _StitchCandidate,
) -> bool:
    if _pair_ratio(left.scale, right.scale) > 1.55:
        return False
    if _pair_ratio(left.outer_inner_ratio, right.outer_inner_ratio) > 1.35:
        return False
    if _parallel_cosine(left.tangent, right.tangent) < math.cos(math.radians(38)):
        return False
    delta = _normalized_vector(
        right.center[0] - left.center[0],
        right.center[1] - left.center[1],
    )
    if delta is None:
        return False
    distance = math.dist(left.center, right.center)
    if distance < min(left.scale, right.scale) * 0.12:
        return False
    if distance > max(left.scale, right.scale) * 1.45:
        return False
    route_alignment_limit = math.cos(math.radians(38))
    return (
        _parallel_cosine(delta, left.tangent) >= route_alignment_limit
        and _parallel_cosine(delta, right.tangent) >= route_alignment_limit
    )


def _stitch_run_qualifies(candidates: Sequence[_StitchCandidate]) -> bool:
    if len(candidates) < 8:
        return False
    typical_scale = max(0.001, _median(candidate.scale for candidate in candidates))
    xs = tuple(candidate.center[0] for candidate in candidates)
    ys = tuple(candidate.center[1] for candidate in candidates)
    route_extent = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    if route_extent < typical_scale * 3:
        return False
    steps = tuple(
        math.dist(candidates[index - 1].center, candidate.center)
        for index, candidate in enumerate(candidates[1:], start=1)
    )
    typical_step = max(0.001, _median(steps))
    accepted = sum(
        typical_step * 0.45 <= step <= typical_step * 2.2
        for step in steps
    )
    return accepted / max(1, len(steps)) >= 0.80


def _stitch_identity_similarity(
    left: _StitchIdentity,
    right: _StitchIdentity,
) -> float:
    ratio = _pair_ratio(left.outer_inner_ratio, right.outer_inner_ratio)
    if ratio > 1.35:
        return 0.0
    return max(0.01, 1.0 - (ratio - 1.0) / 0.35)


def augment_repeated_stitch_path_line_types(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Recover repeated four-segment riding-stitch routes from non-line ops."""

    operation_index = PageOperationIndex.build(page, grouping)
    paths = dict(operation_index.path_items())
    page_diagonal = max(1.0, math.hypot(
        page.page_bounds.width,
        page.page_bounds.height,
    ))
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    additions_by_group: dict[str, list[_StitchAddition]] = {}

    for group in recognition.groups:
        serialized = serialized_by_group.get(group.group_id)
        if serialized is None:
            continue
        candidates = tuple(
            candidate
            for op_index in group.non_linetype.op_indices
            if (
                candidate := _repeated_stitch_candidate(
                    paths, group.group_id, op_index, page_diagonal
                )
            ) is not None
        )
        by_style: dict[str, list[_StitchCandidate]] = {}
        for candidate in candidates:
            by_style.setdefault(candidate.style, []).append(candidate)
        for style, style_candidates in by_style.items():
            style_candidates.sort(key=lambda candidate: candidate.op_index)
            runs: list[_StitchRun] = []
            current: list[_StitchCandidate] = []

            def finish() -> None:
                nonlocal current
                if _stitch_run_qualifies(current):
                    runs.append(_StitchRun(
                        candidates=tuple(current),
                        outer_inner_ratio=_median(
                            candidate.outer_inner_ratio for candidate in current
                        ),
                    ))
                current = []

            for candidate in style_candidates:
                previous = current[-1] if current else None
                if previous is not None and (
                    candidate.op_index - previous.op_index > 12
                    or not _stitch_geometry_connects(previous, candidate)
                ):
                    finish()
                current.append(candidate)
            finish()

            run_clusters: list[list[_StitchRun]] = []
            for run in runs:
                cluster = next((
                    members
                    for members in run_clusters
                    if all(
                        _pair_ratio(member.outer_inner_ratio, run.outer_inner_ratio)
                        <= 1.35
                        for member in members
                    )
                ), None)
                if cluster is None:
                    run_clusters.append([run])
                else:
                    cluster.append(run)
            for cluster in run_clusters:
                seed_candidates = [
                    candidate
                    for run in cluster
                    for candidate in run.candidates
                ]
                if len(seed_candidates) < 12:
                    continue
                selected = {candidate.op_index for candidate in seed_candidates}
                family_ratio = _median(
                    candidate.outer_inner_ratio for candidate in seed_candidates
                )
                changed = True
                while changed:
                    changed = False
                    for candidate in style_candidates:
                        if candidate.op_index in selected:
                            continue
                        if _pair_ratio(family_ratio, candidate.outer_inner_ratio) > 1.35:
                            continue
                        if not any(
                            seed.op_index in selected
                            and _stitch_geometry_connects(seed, candidate)
                            for seed in seed_candidates
                        ):
                            continue
                        selected.add(candidate.op_index)
                        seed_candidates.append(candidate)
                        changed = True
                additions_by_group.setdefault(group.group_id, []).append(
                    _StitchAddition(
                        style=style,
                        outer_inner_ratio=family_ratio,
                        op_indices=_sorted_unique(selected),
                    )
                )

    if not additions_by_group:
        return recognition

    identities: list[_StitchIdentity] = []
    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        additions = additions_by_group.get(group.group_id)
        serialized = serialized_by_group.get(group.group_id)
        if not additions or serialized is None:
            updated_groups.append(group)
            continue
        used_type_ids = {line_type.type_id for line_type in group.line_types}
        next_stitch = 1

        def fresh_type_id() -> str:
            nonlocal next_stitch
            while True:
                candidate = f"type_stitch_{next_stitch:03d}"
                next_stitch += 1
                if candidate not in used_type_ids:
                    used_type_ids.add(candidate)
                    return candidate

        added_types: list[LocalLineType] = []
        for addition in additions:
            type_id = fresh_type_id()
            paths_for_type = set(addition.op_indices)
            line_type = LocalLineType(
                type_id=type_id,
                display_name="",
                line_type_index=1,
                atom_count=_atom_count_for_ops(serialized, paths_for_type),
                op_indices=addition.op_indices,
                model="repeated_stitch_path",
                shape="非直线",
                shape_detail="四段骑缝折线周期",
            )
            added_types.append(line_type)
            identities.append(_StitchIdentity(
                local_key=(group.group_id, type_id),
                group_id=group.group_id,
                style=addition.style,
                outer_inner_ratio=addition.outer_inner_ratio,
                op_indices=addition.op_indices,
            ))
        line_types = tuple(
            replace(
                line_type,
                display_name=f"线型{index}",
                line_type_index=index,
            )
            for index, line_type in enumerate(
                sorted(
                    (*group.line_types, *added_types),
                    key=lambda item: min(item.op_indices, default=math.inf),
                ),
                start=1,
            )
        )
        assigned = {
            op_index
            for line_type in line_types
            for op_index in line_type.op_indices
        }
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=sum(
                    op_index not in assigned
                    for op_index in serialized.atom_op_indices
                ),
                op_indices=_sorted_unique(
                    op_index
                    for op_index in group.non_linetype.op_indices
                    if op_index not in assigned
                ),
            ),
        ))

    groups = tuple(updated_groups)
    local_by_key = _index_local_types(groups)
    identity_clusters: list[list[_StitchIdentity]] = []
    for identity in identities:
        cluster = next((
            members
            for members in identity_clusters
            if all(
                _stitch_identity_similarity(member, identity) > 0
                for member in members
            )
        ), None)
        if cluster is None:
            identity_clusters.append([identity])
        else:
            cluster.append(identity)

    added_globals: list[GlobalLineType] = []
    for cluster in identity_clusters:
        members = tuple(
            GlobalLineTypeMember(
                case_id=identity.group_id,
                type_id=local_by_key[identity.local_key].type_id,
                display_name=local_by_key[identity.local_key].display_name,
                atom_count=local_by_key[identity.local_key].atom_count,
                model=local_by_key[identity.local_key].model,
                shape=local_by_key[identity.local_key].shape,
                shape_detail=local_by_key[identity.local_key].shape_detail,
            )
            for identity in cluster
        )
        similarities = tuple(
            _stitch_identity_similarity(left, right)
            for index, left in enumerate(cluster)
            for right in cluster[index + 1:]
        )
        added_globals.append(GlobalLineType(
            global_type_id="",
            signature_family="repeated_stitch_path",
            minimum_pair_similarity=min(similarities) if similarities else 1.0,
            op_indices=_sorted_unique(
                op_index
                for identity in cluster
                for op_index in identity.op_indices
            ),
            members=members,
        ))

    global_types = _renumber_globals((*recognition.global_types, *added_globals))
    local_count = _count_local_types(groups)
    return replace(
        recognition,
        groups=groups,
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


__all__ = [
    "augment_repeated_stitch_path_line_types",
    "demote_outlined_stroke_text_line_types",
]
