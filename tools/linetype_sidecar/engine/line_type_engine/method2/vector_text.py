"""OCR-free vector-text regions from the frozen Method2 r46 recognizer.

This is the browser-independent port of ``method2/vector-text.ts``.  Dense
positions in :class:`PageIR.operations` are the only operation identities
emitted by this module.  ``paint_order`` is kept separately and is used only
as authored sequence evidence, because multiple PageIR operations may share a
single paint event.

The low-complexity carrier-token-carrier detector stays an independent module.
It is reached through :class:`SequentialMultiPathDetector`, which keeps the
general vector-text detector testable and prevents an import cycle.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
import math
from typing import Iterable, Protocol, Sequence

from ..ir import BoundsIR, GroupingIR, PageIR, PathOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    LineTypeRecognitionResult,
    LocalLineType,
    RecognizedGroup,
)


REPEAT_RESCUE_MINIMUM_INSTANCES = 3
REPEAT_RESCUE_MINIMUM_CONFIDENCE = 0.74
REPEAT_RESCUE_SCALE_TOLERANCE = 0.25
_EPSILON = 1e-6
_SPATIAL_COMPONENT_INDEX_MINIMUM = 512
_SPATIAL_COMPONENT_MAX_DESCRIPTOR_CELLS = 64
_SPATIAL_COMPONENT_MAX_QUERY_CELLS = 256


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class VectorPathDescriptor:
    """Narrow per-path geometry used by vector-text candidate stages."""

    op_index: int
    paint_order: int
    operation: PathOperationIR
    bounds: BoundsIR
    center: Point
    major: float
    segment_count: int
    move_count: int
    curve_count: int
    angle_bins: frozenset[int]
    style: str


@dataclass(frozen=True, slots=True)
class VectorTextEvidence:
    path_count: int
    segment_count: int
    angle_bin_count: int
    paint_order_span: int
    aspect_ratio: float
    carrier_axis_degrees: float | None = None
    single_stroke_shape_key: str | None = None
    sequential_multi_path_shape_key: str | None = None
    sequential_multi_path_chain_key: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "path_count": self.path_count,
            "segment_count": self.segment_count,
            "angle_bin_count": self.angle_bin_count,
            "paint_order_span": self.paint_order_span,
            "aspect_ratio": self.aspect_ratio,
        }
        for name in (
            "carrier_axis_degrees",
            "single_stroke_shape_key",
            "sequential_multi_path_shape_key",
            "sequential_multi_path_chain_key",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class SequentialMultiPathRegion:
    """Unnumbered region returned by the guarded multi-path detector."""

    group_id: str
    op_indices: tuple[int, ...]
    bounds: BoundsIR
    orientation_degrees: float
    confidence: float
    evidence: VectorTextEvidence


@dataclass(frozen=True, slots=True)
class SequentialMultiPathRequest:
    page: PageIR
    operation_index: PageOperationIndex
    group_id: str
    descriptors: tuple[VectorPathDescriptor, ...]
    occupied_op_indices: frozenset[int]
    minimum_instances: int


class SequentialMultiPathDetector(Protocol):
    def __call__(
        self,
        request: SequentialMultiPathRequest,
    ) -> Sequence[SequentialMultiPathRegion]: ...


@dataclass(frozen=True, slots=True)
class VectorTextRegion:
    region_id: str
    group_id: str
    op_indices: tuple[int, ...]
    bounds: BoundsIR
    orientation_degrees: float
    confidence: float
    evidence: VectorTextEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "region_id": self.region_id,
            "group_id": self.group_id,
            "op_indices": list(self.op_indices),
            "bounds": {
                "minX": self.bounds.min_x,
                "minY": self.bounds.min_y,
                "maxX": self.bounds.max_x,
                "maxY": self.bounds.max_y,
            },
            "orientation_degrees": self.orientation_degrees,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class VectorTextProtectionAudit:
    region_count: int
    protected_region_count: int
    protected_op_count: int
    protected_op_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "region_count": self.region_count,
            "protected_region_count": self.protected_region_count,
            "protected_op_count": self.protected_op_count,
            "protected_op_indices": list(self.protected_op_indices),
        }


@dataclass(frozen=True, slots=True)
class VectorTextProtectionResult:
    result: LineTypeRecognitionResult
    audit: VectorTextProtectionAudit


class SerializedGroupLike(Protocol):
    group_id: str
    atom_op_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CarrierEndpoint:
    point: Point
    axis: Point


@dataclass(frozen=True, slots=True)
class _TextRunShortfall:
    region: SequentialMultiPathRegion
    signature: str
    confidence: float
    median_major: float


@dataclass(slots=True)
class _ComponentMajorYBucket:
    positions: list[int]
    cells: dict[int, list[int]]
    tall_positions: list[int]
    maximum_major: float
    maximum_line_width: float


@dataclass(slots=True)
class _ComponentSpatialIndex:
    minimum_xs: tuple[float, ...]
    cell_size: float
    major_buckets: dict[int, _ComponentMajorYBucket]


def _js_round(value: float, digits: int) -> float:
    scale = 10 ** digits
    result = math.floor(value * scale + 0.5) / scale
    return 0.0 if result == 0 else result


def _bounds_gap(left: BoundsIR, right: BoundsIR) -> float:
    return math.hypot(
        max(0.0, left.min_x - right.max_x, right.min_x - left.max_x),
        max(0.0, left.min_y - right.max_y, right.min_y - left.max_y),
    )


def _union_bounds(items: Sequence[VectorPathDescriptor]) -> BoundsIR:
    if not items:
        raise ValueError("cannot union an empty descriptor sequence")
    bounds = items[0].bounds
    for item in items[1:]:
        bounds = bounds.union(item.bounds)
    return bounds


def _color_key(color: tuple[float, ...] | None) -> str:
    if color is None:
        return ""
    return ",".join(f"{value:.9g}" for value in color)


def _path_style(operation: PathOperationIR) -> str:
    return "|".join((
        "S" if operation.stroke else "",
        "F" if operation.fill else "",
        f"{operation.line_width:.3f}",
        _color_key(operation.stroke_color),
        _color_key(operation.fill_color),
    ))


def _descriptor_for(
    op_index: int,
    operation: PathOperationIR,
) -> VectorPathDescriptor | None:
    points: list[Point] = []
    angle_bins: set[int] = set()
    segment_count = 0
    move_count = 0
    curve_count = 0
    current: Point | None = None
    for segment in operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = Point(*segment.end)
            points.append(current)
            move_count += 1
            continue
        if segment.kind == "close":
            continue
        assert segment.end is not None
        following = Point(*segment.end)
        if current is not None:
            dx = following.x - current.x
            dy = following.y - current.y
            if math.hypot(dx, dy) > _EPSILON:
                angle = math.atan2(dy, dx) % math.pi
                angle_bins.add(min(11, math.floor(angle / math.pi * 12)))
                segment_count += 1
        if segment.kind == "curve":
            curve_count += 1
        points.append(following)
        current = following
    if not points or segment_count == 0:
        return None
    width = max(0.0, operation.bounds.width)
    height = max(0.0, operation.bounds.height)
    major = max(width, height)
    if major <= _EPSILON:
        return None
    return VectorPathDescriptor(
        op_index=op_index,
        paint_order=operation.paint_order,
        operation=operation,
        bounds=operation.bounds,
        center=Point(
            (operation.bounds.min_x + operation.bounds.max_x) / 2,
            (operation.bounds.min_y + operation.bounds.max_y) / 2,
        ),
        major=major,
        segment_count=segment_count,
        move_count=move_count,
        curve_count=curve_count,
        angle_bins=frozenset(angle_bins),
        style=_path_style(operation),
    )


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _path_atom_count(operation: PathOperationIR) -> int:
    count = 0
    has_draw = False
    for segment in operation.segments:
        if segment.kind == "move":
            if has_draw:
                count += 1
            has_draw = False
        elif segment.kind != "close":
            has_draw = True
    return count + int(has_draw)


def _atom_count_for(
    operation_index: PageOperationIndex,
    op_indices: Iterable[int],
) -> int:
    count = 0
    for op_index in set(op_indices):
        operation = operation_index.operation(op_index)
        if isinstance(operation, PathOperationIR):
            count += _path_atom_count(operation)
    return count


def _sequence_key(descriptor: VectorPathDescriptor) -> tuple[int, int]:
    return descriptor.paint_order, descriptor.op_index


def _major_exponent(major: float) -> int:
    """Return an exact binary scale bucket for the frozen ratio denominator."""

    return math.frexp(max(0.001, major))[1] - 1


def _build_component_spatial_index(
    descriptors: Sequence[VectorPathDescriptor],
    ordered: Sequence[int],
) -> _ComponentSpatialIndex:
    ordered_descriptors = tuple(descriptors[index] for index in ordered)
    cell_size = max(
        2.0,
        _median([descriptor.major for descriptor in ordered_descriptors]) * 4.0,
        _median([
            descriptor.operation.line_width
            for descriptor in ordered_descriptors
        ]) * 10.0,
    )
    buckets: dict[int, _ComponentMajorYBucket] = {}
    for position, descriptor in enumerate(ordered_descriptors):
        exponent = _major_exponent(descriptor.major)
        bucket = buckets.get(exponent)
        if bucket is None:
            bucket = _ComponentMajorYBucket(
                positions=[],
                cells={},
                tall_positions=[],
                maximum_major=descriptor.major,
                maximum_line_width=descriptor.operation.line_width,
            )
            buckets[exponent] = bucket
        bucket.positions.append(position)
        bucket.maximum_major = max(bucket.maximum_major, descriptor.major)
        bucket.maximum_line_width = max(
            bucket.maximum_line_width,
            descriptor.operation.line_width,
        )
        first_cell = math.floor(descriptor.bounds.min_y / cell_size)
        last_cell = math.floor(descriptor.bounds.max_y / cell_size)
        if (
            last_cell - first_cell + 1
            > _SPATIAL_COMPONENT_MAX_DESCRIPTOR_CELLS
        ):
            bucket.tall_positions.append(position)
            continue
        for cell in range(first_cell, last_cell + 1):
            bucket.cells.setdefault(cell, []).append(position)
    return _ComponentSpatialIndex(
        minimum_xs=tuple(
            descriptor.bounds.min_x for descriptor in ordered_descriptors
        ),
        cell_size=cell_size,
        major_buckets=buckets,
    )


def _bounded_positions(
    positions: Sequence[int],
    left_position: int,
    right_end: int,
) -> Sequence[int]:
    first = bisect_right(positions, left_position)
    last = bisect_left(positions, right_end, lo=first)
    return positions[first:last]


def _indexed_component_candidate_positions(
    descriptors: Sequence[VectorPathDescriptor],
    ordered: Sequence[int],
    spatial_index: _ComponentSpatialIndex,
    left_position: int,
) -> list[int]:
    left = descriptors[ordered[left_position]]
    right_end = bisect_right(
        spatial_index.minimum_xs,
        left.bounds.max_x + max(2.0, left.major * 1.2),
        lo=left_position + 1,
    )
    if right_end <= left_position + 1:
        return []

    candidates: set[int] = set()
    left_exponent = _major_exponent(left.major)
    for exponent in range(left_exponent - 3, left_exponent + 4):
        bucket = spatial_index.major_buckets.get(exponent)
        if bucket is None:
            continue
        # Every pair accepted by the frozen ratio predicate has exponents no
        # more than three buckets apart.  This bound is deliberately only a
        # candidate filter: the exact <= 8 predicate remains below.
        query_tolerance = max(
            0.75,
            min(left.major, bucket.maximum_major) * 0.48,
            left.operation.line_width * 2.5,
            bucket.maximum_line_width * 2.5,
        )
        first_cell = math.floor(
            (left.bounds.min_y - query_tolerance) / spatial_index.cell_size
        )
        last_cell = math.floor(
            (left.bounds.max_y + query_tolerance) / spatial_index.cell_size
        )
        if last_cell - first_cell + 1 > _SPATIAL_COMPONENT_MAX_QUERY_CELLS:
            candidates.update(_bounded_positions(
                bucket.positions,
                left_position,
                right_end,
            ))
            continue
        for cell in range(first_cell, last_cell + 1):
            positions = bucket.cells.get(cell)
            if positions is not None:
                candidates.update(_bounded_positions(
                    positions,
                    left_position,
                    right_end,
                ))
        candidates.update(_bounded_positions(
            bucket.tall_positions,
            left_position,
            right_end,
        ))
    return sorted(candidates)


def _components_for(
    descriptors: Sequence[VectorPathDescriptor],
    use_sequential_paint_prior: bool = False,
) -> tuple[tuple[VectorPathDescriptor, ...], ...]:
    parent = list(range(len(descriptors)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def join(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_style: dict[str, list[int]] = {}
    for index, descriptor in enumerate(descriptors):
        by_style.setdefault(descriptor.style, []).append(index)

    for indices in by_style.values():
        ordered = sorted(
            indices,
            key=lambda index: (descriptors[index].bounds.min_x, descriptors[index].op_index),
        )
        ordered_minimum_xs = tuple(
            descriptors[index].bounds.min_x for index in ordered
        )
        spatial_index = (
            _build_component_spatial_index(descriptors, ordered)
            if len(ordered) >= _SPATIAL_COMPONENT_INDEX_MINIMUM
            else None
        )
        for left_order, left_index in enumerate(ordered):
            left = descriptors[left_index]
            left_bounds = left.bounds
            left_major = left.major
            left_line_width = left.operation.line_width
            if spatial_index is None:
                right_end = bisect_right(
                    ordered_minimum_xs,
                    left_bounds.max_x + max(2.0, left_major * 1.2),
                    lo=left_order + 1,
                )
                right_positions: Iterable[int] = range(left_order + 1, right_end)
            else:
                right_positions = _indexed_component_candidate_positions(
                    descriptors,
                    ordered,
                    spatial_index,
                    left_order,
                )
            for right_order in right_positions:
                right_index = ordered[right_order]
                right = descriptors[right_index]
                right_bounds = right.bounds
                right_major = right.major
                smaller_major = min(left_major, right_major)
                larger_major = max(left_major, right_major)
                size_ratio = larger_major / max(0.001, smaller_major)
                if size_ratio > 8:
                    continue
                gap_tolerance = max(
                    0.75,
                    smaller_major * 0.48,
                    left_line_width * 2.5,
                    right.operation.line_width * 2.5,
                )
                # Axis rejection is mathematically equivalent to the frozen
                # Euclidean bounds-gap test and avoids millions of hypot calls
                # on clearly separated paths in very large Groups.  The final
                # two-axis case still calls math.hypot, preserving its exact
                # floating-point boundary behavior.
                x_gap = max(
                    0.0,
                    left_bounds.min_x - right_bounds.max_x,
                    right_bounds.min_x - left_bounds.max_x,
                )
                if x_gap > gap_tolerance:
                    continue
                y_gap = max(
                    0.0,
                    left_bounds.min_y - right_bounds.max_y,
                    right_bounds.min_y - left_bounds.max_y,
                )
                if y_gap > gap_tolerance:
                    continue
                if x_gap > 0 and y_gap > 0 and math.hypot(x_gap, y_gap) > gap_tolerance:
                    continue
                center_limit = larger_major * 2.5 + gap_tolerance
                center_dx = abs(left.center.x - right.center.x)
                if center_dx > center_limit:
                    continue
                center_dy = abs(left.center.y - right.center.y)
                if center_dy > center_limit:
                    continue
                if (
                    center_dx > 0
                    and center_dy > 0
                    and math.hypot(center_dx, center_dy) > center_limit
                ):
                    continue
                join(left_index, right_index)

        paint_ordered = sorted(indices, key=lambda index: _sequence_key(descriptors[index])) \
            if use_sequential_paint_prior else []
        for order in range(1, len(paint_ordered)):
            left_index = paint_ordered[order - 1]
            right_index = paint_ordered[order]
            left = descriptors[left_index]
            right = descriptors[right_index]
            paint_gap = right.paint_order - left.paint_order
            if paint_gap < 1 or paint_gap > 3:
                continue
            size_ratio = max(left.major, right.major) / max(
                0.001, min(left.major, right.major)
            )
            if size_ratio > 5:
                continue
            left_simple = (
                left.segment_count == 1 and left.move_count == 1 and left.curve_count == 0
            )
            right_simple = (
                right.segment_count == 1 and right.move_count == 1 and right.curve_count == 0
            )
            simple_direction_transition = False
            if left_simple and right_simple:
                left_bin = next(iter(left.angle_bins), 0)
                right_bin = next(iter(right.angle_bins), 0)
                raw_delta = abs(left_bin - right_bin)
                axis_delta = min(raw_delta, 12 - raw_delta)
                if axis_delta <= 1 or size_ratio > 5:
                    continue
                simple_direction_transition = True
            if not simple_direction_transition:
                if left_simple and left.major > right.major * 1.8:
                    continue
                if right_simple and right.major > left.major * 1.8:
                    continue
            smaller_major = min(left.major, right.major)
            larger_major = max(left.major, right.major)
            relaxed_gap = max(
                2.0,
                smaller_major * 2.8,
                larger_major * 1.45,
                left.operation.line_width * 5,
                right.operation.line_width * 5,
            )
            if _bounds_gap(left.bounds, right.bounds) > relaxed_gap:
                continue
            center_gap = math.hypot(
                left.center.x - right.center.x,
                left.center.y - right.center.y,
            )
            if center_gap > larger_major * 4.25 + relaxed_gap:
                continue
            join(left_index, right_index)

    components: dict[int, list[VectorPathDescriptor]] = {}
    for index, descriptor in enumerate(descriptors):
        components.setdefault(find(index), []).append(descriptor)
    return tuple(tuple(component) for component in components.values())


def _straight_carrier_descriptor(descriptor: VectorPathDescriptor) -> bool:
    operation = descriptor.operation
    if not operation.stroke or descriptor.curve_count > 0 or descriptor.move_count != 1:
        return False
    current: Point | None = None
    direction: Point | None = None
    line_count = 0
    for segment in operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = Point(*segment.end)
            continue
        if segment.kind != "line" or current is None:
            continue
        assert segment.end is not None
        dx = segment.end[0] - current.x
        dy = segment.end[1] - current.y
        length = math.hypot(dx, dy)
        current = Point(*segment.end)
        if length <= _EPSILON:
            continue
        unit = Point(dx / length, dy / length)
        if direction is None:
            direction = unit
        elif abs(direction.x * unit.x + direction.y * unit.y) < 0.985:
            return False
        line_count += 1
    return 0 < line_count <= 12


def _route_carrier_descriptor(descriptor: VectorPathDescriptor) -> bool:
    if _straight_carrier_descriptor(descriptor):
        return True
    operation = descriptor.operation
    if (
        not operation.stroke
        or descriptor.curve_count > 0
        or descriptor.move_count != 1
        or descriptor.segment_count < 2
        or descriptor.segment_count > 12
    ):
        return False
    width = descriptor.bounds.width
    height = descriptor.bounds.height
    if max(width, height) / max(0.001, min(width, height)) >= 2.2:
        return True
    current: Point | None = None
    start: Point | None = None
    previous_direction: Point | None = None
    total_length = 0.0
    minimum_tangent_similarity = 1.0
    for segment in operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = Point(*segment.end)
            start = current
            continue
        if segment.kind != "line" or current is None:
            continue
        assert segment.end is not None
        end = Point(*segment.end)
        length = math.hypot(end.x - current.x, end.y - current.y)
        if length > _EPSILON:
            direction = Point((end.x - current.x) / length, (end.y - current.y) / length)
            if previous_direction is not None:
                minimum_tangent_similarity = min(
                    minimum_tangent_similarity,
                    previous_direction.x * direction.x + previous_direction.y * direction.y,
                )
            previous_direction = direction
            total_length += length
        current = end
    if start is None or current is None or total_length <= _EPSILON:
        return False
    endpoint_efficiency = math.hypot(current.x - start.x, current.y - start.y) / total_length
    return endpoint_efficiency >= 0.82 and minimum_tangent_similarity >= 0.72


def _carrier_endpoint_toward(
    descriptor: VectorPathDescriptor,
    target: Point,
) -> _CarrierEndpoint | None:
    operation = descriptor.operation
    sharp_open_carrier = (
        operation.stroke
        and not operation.fill
        and descriptor.move_count == 1
        and descriptor.curve_count == 0
        and 1 <= descriptor.segment_count <= 3
    )
    if (not _route_carrier_descriptor(descriptor) and not sharp_open_carrier) \
            or descriptor.curve_count > 0:
        return None
    points: list[Point] = []
    for segment in operation.segments:
        if segment.kind == "close":
            continue
        assert segment.end is not None
        points.append(Point(*segment.end))
    if len(points) < 2:
        return None
    first_distance = math.hypot(points[0].x - target.x, points[0].y - target.y)
    last_distance = math.hypot(points[-1].x - target.x, points[-1].y - target.y)
    endpoint = 0 if first_distance <= last_distance else len(points) - 1
    neighbor = 1 if endpoint == 0 else len(points) - 2
    dx = points[endpoint].x - points[neighbor].x
    dy = points[endpoint].y - points[neighbor].y
    length = math.hypot(dx, dy)
    if length <= _EPSILON:
        return None
    return _CarrierEndpoint(points[endpoint], Point(dx / length, dy / length))


def _single_stroke_shape_key(descriptor: VectorPathDescriptor) -> str | None:
    operation = descriptor.operation
    if (
        not operation.stroke
        or operation.fill
        or descriptor.move_count != 1
        or descriptor.curve_count != 0
        or descriptor.segment_count < 3
        or descriptor.segment_count > 10
        or len(descriptor.angle_bins) < 2
    ):
        return None
    points: list[Point] = []
    for segment in operation.segments:
        if segment.kind in {"close", "curve"}:
            return None
        assert segment.end is not None
        points.append(Point(*segment.end))
    if len(points) != descriptor.segment_count + 1:
        return None
    vectors = [
        Point(point.x - points[index].x, point.y - points[index].y)
        for index, point in enumerate(points[1:])
    ]
    lengths = [math.hypot(vector.x, vector.y) for vector in vectors]
    total_length = sum(lengths)
    endpoint_efficiency = math.hypot(
        points[-1].x - points[0].x,
        points[-1].y - points[0].y,
    ) / max(_EPSILON, total_length)
    if endpoint_efficiency >= 0.78:
        return None
    turns: list[tuple[int, int]] = []
    for index, vector in enumerate(vectors[1:]):
        previous = vectors[index]
        denominator = max(_EPSILON, lengths[index] * lengths[index + 1])
        dot = (previous.x * vector.x + previous.y * vector.y) / denominator
        cross = (previous.x * vector.y - previous.y * vector.x) / denominator
        turns.append((
            math.floor(max(-1.0, min(1.0, dot)) * 6 + 0.5),
            math.floor(max(-1.0, min(1.0, cross)) * 6 + 0.5),
        ))
    normalized_lengths = [
        math.floor(length / max(_EPSILON, total_length) * 32 + 0.5)
        for length in lengths
    ]
    base_variants = (
        (normalized_lengths, turns),
        (
            list(reversed(normalized_lengths)),
            [(dot, -cross) for dot, cross in reversed(turns)],
        ),
    )
    variants: list[str] = []
    for variant_lengths, variant_turns in base_variants:
        for reflected in (False, True):
            reflected_turns = [
                (dot, -cross if reflected else cross)
                for dot, cross in variant_turns
            ]
            variants.append(
                f"{descriptor.segment_count}:" +
                ",".join(str(value) for value in variant_lengths) + ":" +
                ";".join(f"{dot},{cross}" for dot, cross in reflected_turns)
            )
    return min(variants)


def _repeated_single_stroke_carrier_regions(
    group_id: str,
    descriptors: Sequence[VectorPathDescriptor],
    occupied_ops: frozenset[int] | set[int],
) -> tuple[SequentialMultiPathRegion, ...]:
    ordered = sorted(descriptors, key=_sequence_key)
    candidates: list[tuple[VectorPathDescriptor, Point, str]] = []
    for index in range(1, len(ordered) - 1):
        previous = ordered[index - 1]
        descriptor = ordered[index]
        following = ordered[index + 1]
        if (
            descriptor.op_index in occupied_ops
            or previous.paint_order + 1 != descriptor.paint_order
            or descriptor.paint_order + 1 != following.paint_order
        ):
            continue
        key = _single_stroke_shape_key(descriptor)
        previous_endpoint = _carrier_endpoint_toward(previous, descriptor.center)
        following_endpoint = _carrier_endpoint_toward(following, descriptor.center)
        if key is None or previous_endpoint is None or following_endpoint is None:
            continue
        if (
            descriptor.style != previous.style
            or descriptor.style != following.style
            or descriptor.major > min(previous.major, following.major) * 0.78
            or descriptor.major < descriptor.operation.line_width * 3
        ):
            continue
        previous_axis = previous_endpoint.axis
        aligned_following = following_endpoint.axis
        if previous_axis.x * aligned_following.x + previous_axis.y * aligned_following.y < 0:
            aligned_following = Point(-aligned_following.x, -aligned_following.y)
        axis_length = math.hypot(
            previous_axis.x + aligned_following.x,
            previous_axis.y + aligned_following.y,
        )
        if axis_length <= _EPSILON:
            continue
        axis = Point(
            (previous_axis.x + aligned_following.x) / axis_length,
            (previous_axis.y + aligned_following.y) / axis_length,
        )
        gap_tolerance = max(
            descriptor.major * 2.2,
            descriptor.operation.line_width * 8,
            2.0,
        )
        if (
            _bounds_gap(previous.bounds, descriptor.bounds) > gap_tolerance
            or _bounds_gap(descriptor.bounds, following.bounds) > gap_tolerance
        ):
            continue
        normal = Point(-axis.y, axis.x)

        def cross_offset(point: Point) -> float:
            return abs(
                (point.x - descriptor.center.x) * normal.x
                + (point.y - descriptor.center.y) * normal.y
            )

        maximum_cross_offset = descriptor.major * 1.2 + descriptor.operation.line_width * 3
        if (
            cross_offset(previous_endpoint.point) > maximum_cross_offset
            or cross_offset(following_endpoint.point) > maximum_cross_offset
        ):
            continue
        candidates.append((descriptor, axis, f"{'S' if descriptor.operation.stroke else ''}"
                           f"{'F' if descriptor.operation.fill else ''}:{key}"))

    by_shape: dict[str, list[tuple[VectorPathDescriptor, Point, str]]] = {}
    for candidate in candidates:
        by_shape.setdefault(candidate[2], []).append(candidate)
    result: list[SequentialMultiPathRegion] = []
    for bucket in by_shape.values():
        if len(bucket) < 3:
            continue
        for descriptor, axis, _key in bucket:
            width = descriptor.bounds.width
            height = descriptor.bounds.height
            orientation = _js_round(math.atan2(axis.y, axis.x) * 180 / math.pi, 2)
            result.append(SequentialMultiPathRegion(
                group_id=group_id,
                op_indices=(descriptor.op_index,),
                bounds=descriptor.bounds,
                orientation_degrees=orientation,
                confidence=min(0.99, 0.90 + min(0.08, len(bucket) * 0.005)),
                evidence=VectorTextEvidence(
                    path_count=1,
                    segment_count=descriptor.segment_count,
                    angle_bin_count=len(descriptor.angle_bins),
                    paint_order_span=1,
                    aspect_ratio=max(width, height) / max(0.001, min(width, height)),
                    carrier_axis_degrees=orientation,
                    single_stroke_shape_key=bucket[0][2],
                ),
            ))
    return tuple(result)


def _path_grid_occupancy(
    descriptors: Sequence[VectorPathDescriptor],
    bounds: BoundsIR,
    bins: int = 18,
) -> float:
    width = max(_EPSILON, bounds.width)
    height = max(_EPSILON, bounds.height)
    occupied: set[int] = set()

    def mark(point: Point) -> None:
        x = max(0, min(bins - 1, math.floor((point.x - bounds.min_x) / width * bins)))
        y = max(0, min(bins - 1, math.floor((point.y - bounds.min_y) / height * bins)))
        occupied.add(y * bins + x)

    for descriptor in descriptors:
        current: Point | None = None
        for segment in descriptor.operation.segments:
            if segment.kind == "move":
                assert segment.end is not None
                current = Point(*segment.end)
                mark(current)
                continue
            if segment.kind == "close" or current is None:
                continue
            assert segment.end is not None
            end = Point(*segment.end)
            steps = max(2, math.ceil(max(
                abs(end.x - current.x) / width,
                abs(end.y - current.y) / height,
            ) * bins * 2))
            for step in range(1, steps + 1):
                ratio = step / steps
                mark(Point(
                    current.x + (end.x - current.x) * ratio,
                    current.y + (end.y - current.y) * ratio,
                ))
            current = end
    return len(occupied) / (bins * bins)


def _split_paint_order_runs(
    component: Sequence[VectorPathDescriptor],
    use_sequential_carrier_delimiters: bool = False,
) -> tuple[tuple[VectorPathDescriptor, ...], ...]:
    ordered = sorted(component, key=_sequence_key)
    runs: list[list[VectorPathDescriptor]] = []
    for descriptor in ordered:
        if not runs or descriptor.paint_order - runs[-1][-1].paint_order > 8:
            runs.append([descriptor])
        else:
            runs[-1].append(descriptor)
    if not use_sequential_carrier_delimiters:
        return tuple(tuple(run) for run in runs)

    split_runs: list[tuple[VectorPathDescriptor, ...]] = []
    for run in runs:
        if len(run) < 4:
            split_runs.append(tuple(run))
            continue
        median_major = _median([descriptor.major for descriptor in run])
        route_delimiters = {
            index
            for index, descriptor in enumerate(run)
            if _route_carrier_descriptor(descriptor)
            and descriptor.major >= max(8.0, median_major * 2.25)
        }
        scale_outliers = {
            index
            for index, descriptor in enumerate(run)
            if (
                descriptor.operation.stroke
                and not descriptor.operation.fill
                and descriptor.move_count == 1
                and descriptor.curve_count == 0
                and 2 <= descriptor.segment_count <= 3
                and not any(segment.kind == "close" for segment in descriptor.operation.segments)
                and descriptor.major >= max(10.0, median_major * 3.25)
            )
        }
        all_delimiters = sorted(route_delimiters | scale_outliers)

        def compact_text_core(items: Sequence[VectorPathDescriptor]) -> bool:
            if len(items) < 3:
                return False
            core_median = _median([descriptor.major for descriptor in items])
            scale_inliers = [
                descriptor
                for descriptor in items
                if descriptor.major <= max(core_median * 2.4, core_median + 2)
            ]
            segment_count = sum(descriptor.segment_count for descriptor in items)
            angle_bins = set().union(*(descriptor.angle_bins for descriptor in items))
            return (
                len(scale_inliers) >= max(3, math.ceil(len(items) * 0.65))
                and segment_count >= 6
                and len(angle_bins) >= 3
            )

        verified_scale_delimiters: set[int] = set()
        for index in scale_outliers:
            boundary_order = all_delimiters.index(index)
            previous = all_delimiters[boundary_order - 1] if boundary_order > 0 else -1
            following = (
                all_delimiters[boundary_order + 1]
                if boundary_order + 1 < len(all_delimiters)
                else len(run)
            )
            left_core = run[previous + 1:index]
            right_core = run[index + 1:following]
            if (
                (index == 0 and compact_text_core(right_core))
                or (index == len(run) - 1 and compact_text_core(left_core))
                or (
                    index not in {0, len(run) - 1}
                    and compact_text_core(left_core)
                    and compact_text_core(right_core)
                )
            ):
                verified_scale_delimiters.add(index)

        text_run: list[VectorPathDescriptor] = []
        for index, descriptor in enumerate(run):
            if index in route_delimiters or index in verified_scale_delimiters:
                if text_run:
                    split_runs.append(tuple(text_run))
                    text_run = []
            else:
                text_run.append(descriptor)
        if text_run:
            split_runs.append(tuple(text_run))
    return tuple(split_runs)


def _run_shape_signature(run: Sequence[VectorPathDescriptor]) -> str:
    return "-".join(
        f"{descriptor.segment_count}.{descriptor.move_count}.{descriptor.curve_count}"
        for descriptor in run
    )


def _region_for_run(
    group_id: str,
    input_run: Sequence[VectorPathDescriptor],
    page_bounds: BoundsIR | None = None,
    reject_sparse_large_region: bool = False,
    shortfall: list[_TextRunShortfall] | None = None,
) -> SequentialMultiPathRegion | None:
    run = list(input_run)

    def straight_primitive(descriptor: VectorPathDescriptor) -> bool:
        return (
            descriptor.segment_count == 1
            and descriptor.move_count == 1
            and descriptor.curve_count == 0
            and descriptor.major >= max(3.0, descriptor.operation.line_width * 6)
        )

    for side in ("first", "last"):
        if len(run) < 4:
            break
        candidate = run[0] if side == "first" else run[-1]
        rest = run[1:] if side == "first" else run[:-1]
        if not straight_primitive(candidate):
            continue
        rest_bounds = _union_bounds(rest)
        rest_center = Point(
            (rest_bounds.min_x + rest_bounds.max_x) / 2,
            (rest_bounds.min_y + rest_bounds.max_y) / 2,
        )
        toward_rest = Point(
            rest_center.x - candidate.center.x,
            rest_center.y - candidate.center.y,
        )
        toward_length = math.hypot(toward_rest.x, toward_rest.y)
        angle_bin = next(iter(candidate.angle_bins), -1)
        theta = (angle_bin + 0.5) / 12 * math.pi
        alignment = (
            abs(
                (toward_rest.x * math.cos(theta) + toward_rest.y * math.sin(theta))
                / toward_length
            )
            if toward_length > _EPSILON else 0.0
        )
        rest_median_major = _median([descriptor.major for descriptor in rest])
        if (
            alignment >= 0.86
            and candidate.major >= rest_median_major * 1.15
            and toward_length >= rest_median_major * 0.90
        ):
            run = rest

    if not run or len(run) > 240:
        return None
    bounds = _union_bounds(run)
    width = bounds.width
    height = bounds.height
    if width <= _EPSILON or height <= _EPSILON:
        return None
    if page_bounds is not None:
        page_width = max(_EPSILON, page_bounds.width)
        page_height = max(_EPSILON, page_bounds.height)
        if width >= page_width * 0.8 or height >= page_height * 0.8:
            return None
        if reject_sparse_large_region:
            page_diagonal = math.hypot(page_width, page_height)
            region_diagonal = math.hypot(width, height)
            occupancy = _path_grid_occupancy(run, bounds)
            if (
                (region_diagonal >= page_diagonal * 0.025 and occupancy < 0.14)
                or (region_diagonal >= page_diagonal * 0.12 and occupancy < 0.24)
                or (region_diagonal >= page_diagonal * 0.08 and len(run) <= 3)
            ):
                return None

    segment_count = sum(descriptor.segment_count for descriptor in run)
    move_count = sum(descriptor.move_count for descriptor in run)
    curve_count = sum(descriptor.curve_count for descriptor in run)
    angle_bins = set().union(*(descriptor.angle_bins for descriptor in run))
    paint_order_span = run[-1].paint_order - run[0].paint_order + 1
    if paint_order_span > max(14, len(run) * 6):
        return None

    mean_x = sum(descriptor.center.x for descriptor in run) / len(run)
    mean_y = sum(descriptor.center.y for descriptor in run) / len(run)
    xx = yy = xy = 0.0
    for descriptor in run:
        dx = descriptor.center.x - mean_x
        dy = descriptor.center.y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    orientation = 0.5 * math.atan2(2 * xy, xx - yy)
    ux = math.cos(orientation)
    uy = math.sin(orientation)
    vx = -uy
    vy = ux
    corners = [
        Point(x, y)
        for descriptor in run
        for x, y in (
            (descriptor.bounds.min_x, descriptor.bounds.min_y),
            (descriptor.bounds.max_x, descriptor.bounds.min_y),
            (descriptor.bounds.min_x, descriptor.bounds.max_y),
            (descriptor.bounds.max_x, descriptor.bounds.max_y),
        )
    ]
    along = [point.x * ux + point.y * uy for point in corners]
    across = [point.x * vx + point.y * vy for point in corners]
    along_span = max(along) - min(along)
    across_span = max(across) - min(across)
    major = max(along_span, across_span)
    minor = min(along_span, across_span)
    aspect_ratio = major / max(0.001, minor)
    median_path_major = _median([descriptor.major for descriptor in run])
    if reject_sparse_large_region:
        page_diagonal = (
            math.hypot(page_bounds.width, page_bounds.height)
            if page_bounds is not None else math.inf
        )
        if minor > page_diagonal * 0.03:
            return None
        if minor > max(12.0, median_path_major * 4.5):
            return None
        if any(
            straight_primitive(descriptor)
            and descriptor.major >= max(10.0, median_path_major * 3.25)
            for descriptor in run
        ):
            return None

    is_multi_path_text = (
        len(run) >= 2
        and (len(run) >= 4 or segment_count >= 6)
        and len(angle_bins) >= (2 if curve_count > segment_count * 0.15 else 3)
        and 1.15 <= aspect_ratio <= 40
        and median_path_major <= max(minor * 2.2, major * 0.58)
    )
    is_single_compound_text = (
        len(run) == 1
        and move_count >= 6
        and segment_count >= 12
        and len(angle_bins) >= 3
        and 1.4 <= aspect_ratio <= 40
    )
    if not is_multi_path_text and not is_single_compound_text:
        return None
    if len(run) == 2 and (minor < 2.75 or major < 4):
        return None

    centers = sorted(
        descriptor.center.x * ux + descriptor.center.y * uy
        for descriptor in run
    )
    positive_gaps = [
        value - centers[index]
        for index, value in enumerate(centers[1:])
        if value - centers[index] > _EPSILON
    ]
    if positive_gaps and max(positive_gaps) > minor * 2.8 + 2:
        return None
    confidence = max(0.0, min(1.0,
        0.44
        + min(0.18, len(run) * 0.025)
        + min(0.16, len(angle_bins) * 0.025)
        + min(0.12, segment_count * 0.006)
        + (0.10 if paint_order_span <= len(run) * 2 else 0.0)
    ))
    region = SequentialMultiPathRegion(
        group_id=group_id,
        op_indices=tuple(sorted(descriptor.op_index for descriptor in run)),
        bounds=bounds,
        orientation_degrees=_js_round(orientation * 180 / math.pi, 2),
        confidence=_js_round(confidence, 3),
        evidence=VectorTextEvidence(
            path_count=len(run),
            segment_count=segment_count,
            angle_bin_count=len(angle_bins),
            paint_order_span=paint_order_span,
            aspect_ratio=_js_round(aspect_ratio, 3),
        ),
    )
    minimum_confidence = 0.84 if is_single_compound_text and segment_count >= 40 else 0.86
    if confidence < minimum_confidence:
        if shortfall is not None and confidence >= REPEAT_RESCUE_MINIMUM_CONFIDENCE:
            shortfall.append(_TextRunShortfall(
                region=region,
                signature=_run_shape_signature(run),
                confidence=confidence,
                median_major=_median([descriptor.major for descriptor in run]),
            ))
        return None
    return region


def _default_sequential_multi_path_detector(
    request: SequentialMultiPathRequest,
) -> Sequence[SequentialMultiPathRegion]:
    # Delayed import is intentional: the strict C-T-C recognizer is separately
    # testable and does not import this general vector-text module.
    from .multi_path_carrier import (
        CarrierDelimitedMultiPathDetectionInput,
        MultiPathCarrierDescriptor,
        MultiPathCarrierEndpoint,
        MultiPathCarrierPoint,
        detect_carrier_delimited_multi_path_regions,
    )

    by_index = {descriptor.op_index: descriptor for descriptor in request.descriptors}
    converted = tuple(MultiPathCarrierDescriptor(
        op_index=descriptor.op_index,
        bounds=descriptor.bounds,
        center=MultiPathCarrierPoint(descriptor.center.x, descriptor.center.y),
        major=descriptor.major,
        segment_count=descriptor.segment_count,
        move_count=descriptor.move_count,
        curve_count=descriptor.curve_count,
        angle_bins=descriptor.angle_bins,
        style=descriptor.style,
    ) for descriptor in request.descriptors)

    def is_route_carrier(external: MultiPathCarrierDescriptor, _operation: PathOperationIR) -> bool:
        return _route_carrier_descriptor(by_index[external.op_index])

    def endpoint_toward(
        external: MultiPathCarrierDescriptor,
        _operation: PathOperationIR,
        target: MultiPathCarrierPoint,
    ) -> MultiPathCarrierEndpoint | None:
        endpoint = _carrier_endpoint_toward(
            by_index[external.op_index],
            Point(target.x, target.y),
        )
        if endpoint is None:
            return None
        return MultiPathCarrierEndpoint(
            MultiPathCarrierPoint(endpoint.point.x, endpoint.point.y),
            MultiPathCarrierPoint(endpoint.axis.x, endpoint.axis.y),
        )

    external_regions = detect_carrier_delimited_multi_path_regions(
        CarrierDelimitedMultiPathDetectionInput(
            page=request.page,
            operation_index=request.operation_index,
            group_id=request.group_id,
            descriptors=converted,
            occupied_op_indices=request.occupied_op_indices,
            minimum_instances=request.minimum_instances,
            is_route_carrier=is_route_carrier,
            endpoint_toward=endpoint_toward,
        )
    )
    return tuple(SequentialMultiPathRegion(
        group_id=region.group_id,
        op_indices=region.op_indices,
        bounds=region.bounds,
        orientation_degrees=region.orientation_degrees,
        confidence=region.confidence,
        evidence=VectorTextEvidence(
            path_count=region.evidence.path_count,
            segment_count=region.evidence.segment_count,
            angle_bin_count=region.evidence.angle_bin_count,
            paint_order_span=region.evidence.paint_order_span,
            aspect_ratio=region.evidence.aspect_ratio,
            carrier_axis_degrees=region.evidence.carrier_axis_degrees,
            sequential_multi_path_shape_key=(
                region.evidence.sequential_multi_path_shape_key
            ),
            sequential_multi_path_chain_key=(
                region.evidence.sequential_multi_path_chain_key
            ),
        ),
    ) for region in external_regions)


def _validate_inputs(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
) -> None:
    if grouping.page_fingerprint != page.fingerprint:
        raise ValueError("GroupingIR does not belong to the supplied PageIR")
    if operation_index.operations != page.operations:
        raise ValueError("operation_index does not belong to the supplied PageIR")
    for group in grouping.groups:
        declared = tuple(
            operation_index.operation_index(operation_id)
            for operation_id in group.operation_ids
        )
        if declared != operation_index.group_indices(group.group_id):
            raise ValueError(f"operation_index differs from group {group.group_id!r}")


def _detect_vector_text_regions_with_prior(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
    use_sequential_paint_prior: bool,
    multi_path_detector: SequentialMultiPathDetector | None,
) -> tuple[VectorTextRegion, ...]:
    _validate_inputs(page, grouping, operation_index)
    regions: list[VectorTextRegion] = []
    for group_position, group in enumerate(grouping.groups, start=1):
        descriptors: list[VectorPathDescriptor] = []
        for op_index in operation_index.group_indices(group.group_id):
            operation = operation_index.operation(op_index)
            if not isinstance(operation, PathOperationIR):
                continue
            descriptor = _descriptor_for(op_index, operation)
            if descriptor is not None:
                descriptors.append(descriptor)

        accepted: list[SequentialMultiPathRegion] = []
        for component in _components_for(descriptors, use_sequential_paint_prior):
            shortfall: list[_TextRunShortfall] = []
            for run in _split_paint_order_runs(component, use_sequential_paint_prior):
                region = _region_for_run(
                    group.group_id,
                    run,
                    page.page_bounds,
                    use_sequential_paint_prior,
                    shortfall if use_sequential_paint_prior else None,
                )
                if region is not None:
                    accepted.append(region)
            by_shape: dict[str, list[_TextRunShortfall]] = {}
            for item in shortfall:
                by_shape.setdefault(item.signature, []).append(item)
            for bucket in by_shape.values():
                if len(bucket) < REPEAT_RESCUE_MINIMUM_INSTANCES:
                    continue
                scale_median = _median([item.median_major for item in bucket])
                consistent = [
                    item
                    for item in bucket
                    if abs(item.median_major - scale_median)
                    <= max(scale_median * REPEAT_RESCUE_SCALE_TOLERANCE, 0.5)
                ]
                if len(consistent) >= REPEAT_RESCUE_MINIMUM_INSTANCES:
                    accepted.extend(item.region for item in consistent)

        if use_sequential_paint_prior:
            occupied_ops = {index for region in accepted for index in region.op_indices}
            if multi_path_detector is not None:
                multi_path_regions = tuple(multi_path_detector(SequentialMultiPathRequest(
                    page=page,
                    operation_index=operation_index,
                    group_id=group.group_id,
                    descriptors=tuple(descriptors),
                    occupied_op_indices=frozenset(occupied_ops),
                    minimum_instances=REPEAT_RESCUE_MINIMUM_INSTANCES,
                )))
                for region in multi_path_regions:
                    if region.group_id != group.group_id:
                        raise ValueError("multi-path detector returned a region for another group")
                    if any(operation_index.group_id(index) != group.group_id
                           for index in region.op_indices):
                        raise ValueError("multi-path detector returned an out-of-group operation")
                accepted.extend(multi_path_regions)
                occupied_ops.update(
                    index for region in multi_path_regions for index in region.op_indices
                )
            accepted.extend(_repeated_single_stroke_carrier_regions(
                group.group_id,
                descriptors,
                occupied_ops,
            ))

        accepted.sort(key=lambda region: region.op_indices[0])
        for index, region in enumerate(accepted, start=1):
            regions.append(VectorTextRegion(
                region_id=(
                    f"vector_text_{group_position:04d}_{index:03d}"
                ),
                group_id=region.group_id,
                op_indices=region.op_indices,
                bounds=region.bounds,
                orientation_degrees=region.orientation_degrees,
                confidence=region.confidence,
                evidence=region.evidence,
            ))
    return tuple(regions)


def detect_vector_text_regions(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
) -> tuple[VectorTextRegion, ...]:
    """Historical detector retained for the legacy V2 rollback path."""

    return _detect_vector_text_regions_with_prior(
        page,
        grouping,
        operation_index,
        False,
        None,
    )


def detect_vector_text_regions_v2_sequential(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
    *,
    multi_path_detector: SequentialMultiPathDetector | None = (
        _default_sequential_multi_path_detector
    ),
) -> tuple[VectorTextRegion, ...]:
    """r46 detector with bounded paint-order and guarded carrier rescues."""

    return _detect_vector_text_regions_with_prior(
        page,
        grouping,
        operation_index,
        True,
        multi_path_detector,
    )


def _curve_ratio(
    operation_index: PageOperationIndex,
    op_indices: Iterable[int],
) -> float:
    path_count = 0
    curved_count = 0
    for op_index in op_indices:
        operation = operation_index.operation(op_index)
        if not isinstance(operation, PathOperationIR):
            continue
        path_count += 1
        if any(segment.kind == "curve" for segment in operation.segments):
            curved_count += 1
    return curved_count / path_count if path_count else 0.0


def protect_vector_text_regions(
    page: PageIR,
    operation_index: PageOperationIndex,
    serialized_groups: Sequence[SerializedGroupLike],
    recognition: LineTypeRecognitionResult,
    regions: Sequence[VectorTextRegion],
) -> VectorTextProtectionResult:
    """Remove fragmented vector text from an existing recognition result.

    This is the exact pre-family protection stage from the frozen TS module;
    it does not itself promote text regions to Method2 line types.
    """

    if operation_index.operations != page.operations:
        raise ValueError("operation_index does not belong to the supplied PageIR")
    group_by_id = {group.group_id: group for group in recognition.groups}
    protected_regions: list[VectorTextRegion] = []
    for region in regions:
        group = group_by_id.get(region.group_id)
        if group is None:
            continue
        owner_by_op = {
            op_index: line_type.type_id
            for line_type in group.line_types
            for op_index in line_type.op_indices
        }
        owners: set[str] = set()
        has_non_line = False
        for op_index in region.op_indices:
            owner = owner_by_op.get(op_index)
            if owner is None:
                has_non_line = True
            else:
                owners.add(owner)
        should_protect = False
        if len(owners) == 1 and not has_non_line:
            owner_id = next(iter(owners))
            owner = next(
                (item for item in group.line_types if item.type_id == owner_id),
                None,
            )
            if owner is not None:
                region_ops = set(region.op_indices)
                outside_ops = tuple(
                    op_index for op_index in owner.op_indices if op_index not in region_ops
                )
                owner_coverage = len(region.op_indices) / max(1, len(owner.op_indices))
                region_curve_ratio = _curve_ratio(operation_index, region.op_indices)
                outside_curve_ratio = _curve_ratio(operation_index, outside_ops)
                should_protect = (
                    len(region.op_indices) >= 6
                    and len(outside_ops) >= max(12, len(region.op_indices) * 0.5)
                    and owner_coverage <= 0.65
                    and region_curve_ratio <= 0.15
                    and outside_curve_ratio - region_curve_ratio >= 0.60
                )
        else:
            should_protect = len(owners) == 0 or len(owners) >= 2 or has_non_line
        if should_protect:
            protected_regions.append(region)

    protected_ops = {
        op_index
        for region in protected_regions
        for op_index in region.op_indices
    }
    protected_op_indices = tuple(sorted(protected_ops))
    audit = VectorTextProtectionAudit(
        region_count=len(regions),
        protected_region_count=len(protected_regions),
        protected_op_count=len(protected_op_indices),
        protected_op_indices=protected_op_indices,
    )
    if not protected_ops:
        return VectorTextProtectionResult(recognition, audit)

    serialized_by_group = {group.group_id: group for group in serialized_groups}
    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        serialized = serialized_by_group.get(group.group_id)
        if serialized is None:
            updated_groups.append(group)
            continue
        filtered: list[LocalLineType] = []
        for line_type in group.line_types:
            op_indices = tuple(
                op_index
                for op_index in line_type.op_indices
                if op_index not in protected_ops
            )
            atom_count = _atom_count_for(operation_index, op_indices)
            if atom_count > 0:
                filtered.append(replace(
                    line_type,
                    op_indices=op_indices,
                    atom_count=atom_count,
                ))
        line_types = tuple(
            replace(
                line_type,
                display_name=f"线型{index}",
                line_type_index=index,
            )
            for index, line_type in enumerate(filtered, start=1)
        )
        assigned = {
            op_index
            for line_type in line_types
            for op_index in line_type.op_indices
        }
        non_line_ops = tuple(sorted({
            op_index
            for op_index in serialized.atom_op_indices
            if op_index not in assigned
        }))
        non_line_set = set(non_line_ops)
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=sum(
                    op_index in non_line_set
                    for op_index in serialized.atom_op_indices
                ),
                op_indices=non_line_ops,
            ),
        ))

    local_by_key = {
        (group.group_id, line_type.type_id): line_type
        for group in updated_groups
        for line_type in group.line_types
    }
    retained_globals: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        members = []
        for member in global_type.members:
            local = local_by_key.get((member.case_id, member.type_id))
            if local is not None:
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
        op_indices = tuple(sorted({
            op_index
            for member in members
            for op_index in local_by_key[(member.case_id, member.type_id)].op_indices
        }))
        retained_globals.append(replace(
            global_type,
            members=tuple(members),
            op_indices=op_indices,
        ))
    global_types = tuple(
        replace(global_type, global_type_id=f"global_type_{index:03d}")
        for index, global_type in enumerate(retained_globals, start=1)
    )
    local_line_type_count = sum(group.line_type_count for group in updated_groups)
    summary = replace(
        recognition.summary,
        local_line_type_count=local_line_type_count,
        signed_periodic_type_count=max(
            0,
            recognition.summary.signed_periodic_type_count
            + local_line_type_count
            - recognition.summary.local_line_type_count,
        ),
        global_type_count=len(global_types),
        cross_group_global_type_count=sum(
            global_type.group_count > 1 for global_type in global_types
        ),
    )
    return VectorTextProtectionResult(
        LineTypeRecognitionResult(
            groups=tuple(updated_groups),
            global_types=global_types,
            summary=summary,
            errors=recognition.errors,
            schema_version=recognition.schema_version,
        ),
        audit,
    )


__all__ = [
    "REPEAT_RESCUE_MINIMUM_CONFIDENCE",
    "REPEAT_RESCUE_MINIMUM_INSTANCES",
    "REPEAT_RESCUE_SCALE_TOLERANCE",
    "SequentialMultiPathDetector",
    "SequentialMultiPathRegion",
    "SequentialMultiPathRequest",
    "VectorPathDescriptor",
    "VectorTextEvidence",
    "VectorTextProtectionAudit",
    "VectorTextProtectionResult",
    "VectorTextRegion",
    "detect_vector_text_regions",
    "detect_vector_text_regions_v2_sequential",
    "protect_vector_text_regions",
]
