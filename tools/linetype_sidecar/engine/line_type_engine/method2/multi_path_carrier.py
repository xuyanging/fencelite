"""Carrier-delimited low-complexity token detector from frozen Method2 r46.

The detector recognizes only the strict paint-sequence invariant

``carrier -> 2..12 token paths -> shared carrier -> token -> carrier``.

It deliberately does not decide whether arbitrary short paths are text.  Its
precision comes from shared-carrier continuity, non-overlapping token boxes,
complete-link geometry agreement, and endpoint-tangent checks on both sides
of every token.  All integer operation identities are dense positions in
``PageIR.operations``; ``paint_order`` remains evidence only.

This module is a parity port and is not yet connected to the Python pipeline.
Thresholds and evaluation order mirror ``multi-path-carrier.ts`` r46.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

from ..ir import BoundsIR, PageIR, PathOperationIR
from ..operation_index import PageOperationIndex


MINIMUM_TOKEN_PATHS = 2
MAXIMUM_TOKEN_PATHS = 12
DEFAULT_MINIMUM_INSTANCES = 3
MAXIMUM_SCALE_RATIO = 1.25
MAXIMUM_MEAN_GEOMETRY_DELTA = 0.12
MAXIMUM_SINGLE_GEOMETRY_DELTA = 0.30
EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class MultiPathCarrierPoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        x = float(self.x)
        y = float(self.y)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("carrier point coordinates must be finite")
        object.__setattr__(self, "x", 0.0 if x == 0 else x)
        object.__setattr__(self, "y", 0.0 if y == 0 else y)


@dataclass(frozen=True, slots=True)
class MultiPathCarrierDescriptor:
    """Narrow descriptor consumed by the C-T-C detector.

    ``op_index`` is the dense position in ``PageIR.operations``.  The path is
    resolved through ``PageOperationIndex`` at the API boundary instead of
    being duplicated in the descriptor.
    """

    op_index: int
    bounds: BoundsIR
    center: MultiPathCarrierPoint
    major: float
    segment_count: int
    move_count: int
    curve_count: int
    angle_bins: frozenset[int]
    style: str

    def __post_init__(self) -> None:
        if isinstance(self.op_index, bool) or not isinstance(self.op_index, int):
            raise TypeError("descriptor op_index must be an integer")
        if self.op_index < 0:
            raise ValueError("descriptor op_index must be non-negative")
        major = float(self.major)
        if not math.isfinite(major) or major < 0:
            raise ValueError("descriptor major must be finite and non-negative")
        object.__setattr__(self, "major", major)
        for field_name in ("segment_count", "move_count", "curve_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"descriptor {field_name} must be a non-negative integer")
        bins = frozenset(self.angle_bins)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in bins):
            raise TypeError("descriptor angle bins must be integers")
        object.__setattr__(self, "angle_bins", bins)


@dataclass(frozen=True, slots=True)
class MultiPathCarrierEndpoint:
    point: MultiPathCarrierPoint
    """Selected carrier endpoint."""

    axis: MultiPathCarrierPoint
    """Unit tangent pointing out of the selected carrier endpoint."""


RouteCarrierPredicate = Callable[
    [MultiPathCarrierDescriptor, PathOperationIR],
    bool,
]
EndpointToward = Callable[
    [MultiPathCarrierDescriptor, PathOperationIR, MultiPathCarrierPoint],
    MultiPathCarrierEndpoint | None,
]


@dataclass(frozen=True, slots=True)
class CarrierDelimitedMultiPathDetectionInput:
    page: PageIR
    operation_index: PageOperationIndex
    group_id: str
    descriptors: tuple[MultiPathCarrierDescriptor, ...]
    occupied_op_indices: frozenset[int]
    is_route_carrier: RouteCarrierPredicate
    endpoint_toward: EndpointToward
    minimum_instances: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "occupied_op_indices", frozenset(self.occupied_op_indices))
        if not self.group_id:
            raise ValueError("group_id must not be empty")
        if self.minimum_instances is not None and (
            isinstance(self.minimum_instances, bool)
            or not isinstance(self.minimum_instances, int)
        ):
            raise TypeError("minimum_instances must be an integer or None")


@dataclass(frozen=True, slots=True)
class CarrierDelimitedMultiPathEvidence:
    path_count: int
    segment_count: int
    angle_bin_count: int
    paint_order_span: int
    aspect_ratio: float
    carrier_axis_degrees: float
    sequential_multi_path_shape_key: str
    sequential_multi_path_chain_key: str


@dataclass(frozen=True, slots=True)
class CarrierDelimitedMultiPathRegion:
    group_id: str
    op_indices: tuple[int, ...]
    bounds: BoundsIR
    orientation_degrees: float
    confidence: float
    evidence: CarrierDelimitedMultiPathEvidence


GeometryProfile = tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _ResolvedDescriptor:
    descriptor: MultiPathCarrierDescriptor
    operation: PathOperationIR

    @property
    def op_index(self) -> int:
        return self.descriptor.op_index

    @property
    def bounds(self) -> BoundsIR:
        return self.descriptor.bounds

    @property
    def center(self) -> MultiPathCarrierPoint:
        return self.descriptor.center

    @property
    def major(self) -> float:
        return self.descriptor.major

    @property
    def segment_count(self) -> int:
        return self.descriptor.segment_count

    @property
    def move_count(self) -> int:
        return self.descriptor.move_count

    @property
    def curve_count(self) -> int:
        return self.descriptor.curve_count

    @property
    def angle_bins(self) -> frozenset[int]:
        return self.descriptor.angle_bins

    @property
    def style(self) -> str:
        return self.descriptor.style


@dataclass(frozen=True, slots=True)
class _CarrierDelimitedCandidate:
    token_descriptors: tuple[_ResolvedDescriptor, ...]
    token_bounds: BoundsIR
    center: MultiPathCarrierPoint
    intrinsic_scale: float
    carrier_axis: MultiPathCarrierPoint
    topology: str
    geometry: GeometryProfile
    segment_count: int
    angle_bin_count: int
    left_carrier: _ResolvedDescriptor
    right_carrier: _ResolvedDescriptor


@dataclass(frozen=True, slots=True)
class _GeometryDifference:
    mean: float
    maximum: float


@dataclass(frozen=True, slots=True)
class _PrincipalFrame:
    orientation_degrees: float
    projected_aspect: float


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _union_bounds(descriptors: Sequence[_ResolvedDescriptor]) -> BoundsIR:
    minimum_x = math.inf
    minimum_y = math.inf
    maximum_x = -math.inf
    maximum_y = -math.inf
    for descriptor in descriptors:
        minimum_x = min(minimum_x, descriptor.bounds.min_x)
        minimum_y = min(minimum_y, descriptor.bounds.min_y)
        maximum_x = max(maximum_x, descriptor.bounds.max_x)
        maximum_y = max(maximum_y, descriptor.bounds.max_y)
    return BoundsIR(minimum_x, minimum_y, maximum_x, maximum_y)


def _bounds_gap(left: BoundsIR, right: BoundsIR) -> float:
    return math.hypot(
        max(0.0, left.min_x - right.max_x, right.min_x - left.max_x),
        max(0.0, left.min_y - right.max_y, right.min_y - left.max_y),
    )


def _bounds_have_interior_overlap(left: BoundsIR, right: BoundsIR) -> bool:
    return (
        min(left.max_x, right.max_x) - max(left.min_x, right.min_x) > EPSILON
        and min(left.max_y, right.max_y) - max(left.min_y, right.min_y) > EPSILON
    )


def _path_geometry_points(
    descriptor: _ResolvedDescriptor,
) -> tuple[MultiPathCarrierPoint, ...]:
    points: list[MultiPathCarrierPoint] = []
    for segment in descriptor.operation.segments:
        if segment.kind == "close":
            continue
        if segment.kind == "curve":
            assert segment.control_1 is not None and segment.control_2 is not None
            points.append(MultiPathCarrierPoint(*segment.control_1))
            points.append(MultiPathCarrierPoint(*segment.control_2))
        assert segment.end is not None
        points.append(MultiPathCarrierPoint(*segment.end))
    return tuple(points)


def _descriptor_path_length(descriptor: _ResolvedDescriptor) -> float:
    current: tuple[float, float] | None = None
    length = 0.0
    for segment in descriptor.operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = segment.end
        elif segment.kind != "close" and current is not None:
            assert segment.end is not None
            length += math.hypot(segment.end[0] - current[0], segment.end[1] - current[1])
            current = segment.end
    return length


@dataclass(slots=True)
class _DescriptorGeometryCache:
    """Exact reusable geometry for overlapping candidate token windows."""

    points_by_op: dict[int, tuple[MultiPathCarrierPoint, ...]]
    path_length_by_op: dict[int, float]
    within_diameter_by_op: dict[int, float]
    cross_diameter_by_ops: dict[tuple[int, int], float]

    @classmethod
    def build(
        cls,
        descriptors: Sequence[_ResolvedDescriptor],
    ) -> _DescriptorGeometryCache:
        return cls(
            {
                descriptor.op_index: _path_geometry_points(descriptor)
                for descriptor in descriptors
            },
            {
                descriptor.op_index: _descriptor_path_length(descriptor)
                for descriptor in descriptors
            },
            {},
            {},
        )

    def path_length(self, descriptor: _ResolvedDescriptor) -> float:
        return self.path_length_by_op[descriptor.op_index]

    def intrinsic_diameter(
        self,
        descriptors: Sequence[_ResolvedDescriptor],
    ) -> float:
        point_groups = [
            self.points_by_op[descriptor.op_index] for descriptor in descriptors
        ]
        point_count = sum(len(points) for points in point_groups)
        if point_count > 256:
            # Preserve the frozen large-token farthest-point approximation.
            return _intrinsic_diameter(tuple(
                point for points in point_groups for point in points
            ))

        diameter = 0.0
        for left_order, (left_descriptor, left_points) in enumerate(zip(
            descriptors,
            point_groups,
        )):
            within = self.within_diameter_by_op.get(left_descriptor.op_index)
            if within is None:
                within = _intrinsic_diameter(left_points)
                self.within_diameter_by_op[left_descriptor.op_index] = within
            diameter = max(diameter, within)
            for right_descriptor, right_points in zip(
                descriptors[left_order + 1:],
                point_groups[left_order + 1:],
            ):
                key = (left_descriptor.op_index, right_descriptor.op_index)
                cross = self.cross_diameter_by_ops.get(key)
                if cross is None:
                    cross = 0.0
                    for left_point in left_points:
                        for right_point in right_points:
                            cross = max(cross, math.hypot(
                                left_point.x - right_point.x,
                                left_point.y - right_point.y,
                            ))
                    self.cross_diameter_by_ops[key] = cross
                diameter = max(diameter, cross)
        return diameter


def _intrinsic_diameter(points: Sequence[MultiPathCarrierPoint]) -> float:
    if len(points) > 256:
        def farthest_from(
            anchor: MultiPathCarrierPoint,
        ) -> tuple[MultiPathCarrierPoint, float]:
            farthest = points[0]
            maximum_squared_distance = -1.0
            for point in points:
                dx = point.x - anchor.x
                dy = point.y - anchor.y
                squared_distance = dx * dx + dy * dy
                if squared_distance > maximum_squared_distance:
                    maximum_squared_distance = squared_distance
                    farthest = point
            return farthest, math.sqrt(maximum_squared_distance)

        first_extreme, _ = farthest_from(points[0])
        second_extreme, second_distance = farthest_from(first_extreme)
        _, third_distance = farthest_from(second_extreme)
        return max(second_distance, third_distance)

    diameter = 0.0
    for left_index, left in enumerate(points):
        for right in points[left_index + 1:]:
            diameter = max(diameter, math.hypot(left.x - right.x, left.y - right.y))
    return diameter


def _topology_for(descriptors: Sequence[_ResolvedDescriptor]) -> str:
    parts: list[str] = []
    for descriptor in descriptors:
        segment_kinds = "".join(segment.kind[0] for segment in descriptor.operation.segments)
        parts.append(
            f"{segment_kinds}:{descriptor.segment_count}."
            f"{descriptor.move_count}.{descriptor.curve_count}"
        )
    return "|".join(parts)


def _geometry_for(
    descriptors: Sequence[_ResolvedDescriptor],
    intrinsic_scale: float,
) -> GeometryProfile:
    geometry = [
        _descriptor_path_length(descriptor) / intrinsic_scale
        for descriptor in descriptors
    ]
    for left_index, left in enumerate(descriptors):
        for right in descriptors[left_index + 1:]:
            geometry.append(
                math.hypot(left.center.x - right.center.x, left.center.y - right.center.y)
                / intrinsic_scale
            )
    return tuple(geometry)


def _geometry_difference(
    left: GeometryProfile,
    right: GeometryProfile,
) -> _GeometryDifference | None:
    if len(left) != len(right) or not left:
        return None
    total = 0.0
    maximum = 0.0
    for left_value, right_value in zip(left, right):
        difference = abs(left_value - right_value)
        if not math.isfinite(difference):
            return None
        total += difference
        maximum = max(maximum, difference)
    return _GeometryDifference(total / len(left), maximum)


def _geometries_are_compatible(left: GeometryProfile, right: GeometryProfile) -> bool:
    difference = _geometry_difference(left, right)
    return bool(
        difference is not None
        and difference.mean <= MAXIMUM_MEAN_GEOMETRY_DELTA
        and difference.maximum <= MAXIMUM_SINGLE_GEOMETRY_DELTA
    )


def _component_has_stable_geometry(
    candidates: Sequence[_CarrierDelimitedCandidate],
) -> bool:
    if not candidates:
        return False
    medoid_index = -1
    best_total = math.inf
    for candidate_index, candidate in enumerate(candidates):
        total = 0.0
        valid = True
        for other_index, other in enumerate(candidates):
            if candidate_index == other_index:
                continue
            difference = _geometry_difference(candidate.geometry, other.geometry)
            if difference is None:
                valid = False
                break
            total += difference.mean + difference.maximum * 0.25
        if valid and total < best_total:
            best_total = total
            medoid_index = candidate_index
    if medoid_index < 0:
        return False
    medoid = candidates[medoid_index]
    if any(
        not _geometries_are_compatible(medoid.geometry, candidate.geometry)
        for candidate in candidates
    ):
        return False
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            if not _geometries_are_compatible(left.geometry, right.geometry):
                return False
    return True


def _component_has_pairwise_disjoint_tokens(
    candidates: Sequence[_CarrierDelimitedCandidate],
) -> bool:
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1:]:
            if _bounds_have_interior_overlap(left.token_bounds, right.token_bounds):
                return False
    return True


def _principal_frame_for(
    descriptors: Sequence[_ResolvedDescriptor],
) -> _PrincipalFrame | None:
    points = [
        point
        for descriptor in descriptors
        for point in _path_geometry_points(descriptor)
    ]
    if len(points) < 2:
        return None
    mean_x = sum(point.x for point in points) / len(points)
    mean_y = sum(point.y for point in points) / len(points)
    xx = 0.0
    yy = 0.0
    xy = 0.0
    for point in points:
        dx = point.x - mean_x
        dy = point.y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    angle = 0.5 * math.atan2(2 * xy, xx - yy)

    def projected_spans(theta: float) -> tuple[float, float]:
        ux = math.cos(theta)
        uy = math.sin(theta)
        vx = -uy
        vy = ux
        minimum_u = math.inf
        maximum_u = -math.inf
        minimum_v = math.inf
        maximum_v = -math.inf
        for point in points:
            centered_x = point.x - mean_x
            centered_y = point.y - mean_y
            u = centered_x * ux + centered_y * uy
            v = centered_x * vx + centered_y * vy
            minimum_u = min(minimum_u, u)
            maximum_u = max(maximum_u, u)
            minimum_v = min(minimum_v, v)
            maximum_v = max(maximum_v, v)
        return maximum_u - minimum_u, maximum_v - minimum_v

    span_u, span_v = projected_spans(angle)
    if span_v > span_u:
        angle += math.pi / 2
        span_u, span_v = span_v, span_u
    if span_u <= EPSILON or span_v <= EPSILON:
        return None
    return _PrincipalFrame(
        _round_like_javascript(angle * 180 / math.pi, 2),
        span_u / span_v,
    )


def _carrier_pair_follows_token(
    candidate_center: MultiPathCarrierPoint,
    token_scale: float,
    maximum_line_width: float,
    left_endpoint: MultiPathCarrierEndpoint,
    right_endpoint: MultiPathCarrierEndpoint,
) -> bool:
    def side_follows_route(endpoint: MultiPathCarrierEndpoint) -> bool:
        dx = candidate_center.x - endpoint.point.x
        dy = candidate_center.y - endpoint.point.y
        distance = math.hypot(dx, dy)
        if distance <= EPSILON:
            return True
        forward = (dx * endpoint.axis.x + dy * endpoint.axis.y) / distance
        cross = abs(dx * endpoint.axis.y - dy * endpoint.axis.x)
        return (
            forward >= 0.5
            and cross <= token_scale * 0.35 + maximum_line_width * 2
        )

    return side_follows_route(left_endpoint) and side_follows_route(right_endpoint)


def _bisect_carrier_axes(
    left: MultiPathCarrierEndpoint,
    right: MultiPathCarrierEndpoint,
) -> MultiPathCarrierPoint | None:
    right_axis_x = right.axis.x
    right_axis_y = right.axis.y
    if left.axis.x * right_axis_x + left.axis.y * right_axis_y < 0:
        right_axis_x = -right_axis_x
        right_axis_y = -right_axis_y
    length = math.hypot(left.axis.x + right_axis_x, left.axis.y + right_axis_y)
    if length <= EPSILON:
        return None
    return MultiPathCarrierPoint(
        (left.axis.x + right_axis_x) / length,
        (left.axis.y + right_axis_y) / length,
    )


def _resolve_descriptors(
    data: CarrierDelimitedMultiPathDetectionInput,
) -> tuple[_ResolvedDescriptor, ...]:
    if data.operation_index.operations != data.page.operations:
        raise ValueError("operation_index does not belong to the supplied PageIR")
    # Also proves that the requested group exists before accepting an empty DTO.
    data.operation_index.group_indices(data.group_id)
    resolved: list[_ResolvedDescriptor] = []
    seen: set[int] = set()
    for descriptor in data.descriptors:
        if descriptor.op_index in seen:
            raise ValueError(f"duplicate descriptor op_index {descriptor.op_index}")
        seen.add(descriptor.op_index)
        operation = data.operation_index.operation(descriptor.op_index)
        if not isinstance(operation, PathOperationIR):
            raise TypeError(f"descriptor {descriptor.op_index} does not reference a path")
        if data.operation_index.group_id(descriptor.op_index) != data.group_id:
            raise ValueError(
                f"descriptor {descriptor.op_index} does not belong to group {data.group_id!r}"
            )
        resolved.append(_ResolvedDescriptor(descriptor, operation))
    return tuple(resolved)


def _build_candidates(
    data: CarrierDelimitedMultiPathDetectionInput,
    descriptors: Sequence[_ResolvedDescriptor],
) -> tuple[_CarrierDelimitedCandidate, ...]:
    ordered = sorted(descriptors, key=lambda descriptor: descriptor.op_index)
    geometry_cache = _DescriptorGeometryCache.build(ordered)
    route_carrier_cache: dict[int, bool] = {}

    def is_route_carrier(descriptor: _ResolvedDescriptor) -> bool:
        cached = route_carrier_cache.get(descriptor.op_index)
        if cached is not None:
            return cached
        result = data.is_route_carrier(descriptor.descriptor, descriptor.operation)
        route_carrier_cache[descriptor.op_index] = result
        return result

    candidates: list[_CarrierDelimitedCandidate] = []
    for left_order, left_carrier in enumerate(ordered):
        if (
            not left_carrier.operation.stroke
            or left_carrier.operation.fill
            or not is_route_carrier(left_carrier)
        ):
            continue
        for path_count in range(MINIMUM_TOKEN_PATHS, MAXIMUM_TOKEN_PATHS + 1):
            right_order = left_order + path_count + 1
            if right_order >= len(ordered):
                break
            right_carrier = ordered[right_order]
            token_descriptors = tuple(ordered[left_order + 1:right_order])
            packet = ordered[left_order:right_order + 1]
            if any(
                descriptor.op_index != left_carrier.op_index + offset
                for offset, descriptor in enumerate(packet)
            ):
                continue
            if (
                not right_carrier.operation.stroke
                or right_carrier.operation.fill
                or not is_route_carrier(right_carrier)
                or right_carrier.style != left_carrier.style
                or any(
                    descriptor.op_index in data.occupied_op_indices
                    or not descriptor.operation.stroke
                    or descriptor.operation.fill
                    or descriptor.style != left_carrier.style
                    for descriptor in token_descriptors
                )
            ):
                continue

            # The frozen structural gate is independent of bounds and point
            # geometry.  Evaluate it first so the common one-axis drafting
            # windows never build an expensive union/diameter merely to be
            # rejected for having no directional diversity.
            segment_count = sum(
                descriptor.segment_count for descriptor in token_descriptors
            )
            angle_bins = frozenset(
                angle_bin
                for descriptor in token_descriptors
                for angle_bin in descriptor.angle_bins
            )
            if segment_count < path_count or len(angle_bins) < 2:
                continue
            token_bounds = _union_bounds(token_descriptors)
            if token_bounds.width <= EPSILON or token_bounds.height <= EPSILON:
                continue
            intrinsic_scale = geometry_cache.intrinsic_diameter(token_descriptors)
            if intrinsic_scale <= EPSILON:
                continue
            center = MultiPathCarrierPoint(
                (token_bounds.min_x + token_bounds.max_x) / 2,
                (token_bounds.min_y + token_bounds.max_y) / 2,
            )
            token_median_major = _median([
                descriptor.major for descriptor in token_descriptors
            ])
            if token_median_major <= EPSILON or intrinsic_scale / token_median_major > 8:
                continue

            if (
                geometry_cache.path_length(left_carrier) < intrinsic_scale * 1.8
                or geometry_cache.path_length(right_carrier) < intrinsic_scale * 1.8
            ):
                continue
            maximum_line_width = max(
                left_carrier.operation.line_width,
                right_carrier.operation.line_width,
            )
            gap_tolerance = intrinsic_scale * 1.5 + maximum_line_width * 2
            if (
                _bounds_gap(left_carrier.bounds, token_bounds) > gap_tolerance
                or _bounds_gap(token_bounds, right_carrier.bounds) > gap_tolerance
                or _bounds_have_interior_overlap(left_carrier.bounds, token_bounds)
                or _bounds_have_interior_overlap(token_bounds, right_carrier.bounds)
            ):
                continue

            left_endpoint = data.endpoint_toward(
                left_carrier.descriptor,
                left_carrier.operation,
                center,
            )
            right_endpoint = data.endpoint_toward(
                right_carrier.descriptor,
                right_carrier.operation,
                center,
            )
            if (
                left_endpoint is None
                or right_endpoint is None
                or not _carrier_pair_follows_token(
                    center,
                    intrinsic_scale,
                    maximum_line_width,
                    left_endpoint,
                    right_endpoint,
                )
            ):
                continue
            carrier_axis = _bisect_carrier_axes(left_endpoint, right_endpoint)
            if carrier_axis is None:
                continue
            candidates.append(_CarrierDelimitedCandidate(
                token_descriptors=token_descriptors,
                token_bounds=token_bounds,
                center=center,
                intrinsic_scale=intrinsic_scale,
                carrier_axis=carrier_axis,
                topology=_topology_for(token_descriptors),
                geometry=_geometry_for(token_descriptors, intrinsic_scale),
                segment_count=segment_count,
                angle_bin_count=len(angle_bins),
                left_carrier=left_carrier,
                right_carrier=right_carrier,
            ))
    return tuple(candidates)


def _pair_can_share_chain(
    left: _CarrierDelimitedCandidate,
    right: _CarrierDelimitedCandidate,
) -> bool:
    if _bounds_have_interior_overlap(left.token_bounds, right.token_bounds):
        return False
    pitch = math.hypot(left.center.x - right.center.x, left.center.y - right.center.y)
    if pitch < max(left.intrinsic_scale, right.intrinsic_scale) * 2:
        return False
    scale_ratio = max(left.intrinsic_scale, right.intrinsic_scale) / max(
        EPSILON,
        min(left.intrinsic_scale, right.intrinsic_scale),
    )
    return (
        scale_ratio <= MAXIMUM_SCALE_RATIO
        and _geometries_are_compatible(left.geometry, right.geometry)
    )


def _shared_carrier_adjacency(
    bucket: Sequence[_CarrierDelimitedCandidate],
) -> tuple[set[int], ...]:
    starts_at_carrier: dict[int, list[int]] = {}
    for index, candidate in enumerate(bucket):
        starts_at_carrier.setdefault(candidate.left_carrier.op_index, []).append(index)
    adjacency = tuple(set() for _ in bucket)
    for left_index, candidate in enumerate(bucket):
        for right_index in starts_at_carrier.get(candidate.right_carrier.op_index, ()):
            if (
                left_index == right_index
                or not _pair_can_share_chain(candidate, bucket[right_index])
            ):
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)
    return adjacency


def _accepted_components(
    bucket: Sequence[_CarrierDelimitedCandidate],
    minimum_instances: int,
) -> tuple[tuple[_CarrierDelimitedCandidate, ...], ...]:
    adjacency = _shared_carrier_adjacency(bucket)
    accepted: list[tuple[_CarrierDelimitedCandidate, ...]] = []
    visited: set[int] = set()
    for seed in range(len(bucket)):
        if seed in visited:
            continue
        component_indices: list[int] = []
        pending = [seed]
        visited.add(seed)
        while pending:
            index = pending.pop()
            component_indices.append(index)
            for neighbor in adjacency[index]:
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                pending.append(neighbor)
        if len(component_indices) < minimum_instances:
            continue
        component_set = set(component_indices)
        edge_count = sum(
            sum(neighbor in component_set for neighbor in adjacency[index])
            for index in component_indices
        ) / 2
        if (
            edge_count != len(component_indices) - 1
            or any(len(adjacency[index]) > 2 for index in component_indices)
        ):
            continue
        component = [bucket[index] for index in component_indices]
        scale_median = _median([
            candidate.intrinsic_scale for candidate in component
        ])
        if (
            any(
                abs(candidate.intrinsic_scale - scale_median) > scale_median * 0.25
                for candidate in component
            )
            or not _component_has_pairwise_disjoint_tokens(component)
            or not _component_has_stable_geometry(component)
        ):
            continue
        accepted.append(tuple(sorted(
            component,
            key=lambda candidate: candidate.left_carrier.op_index,
        )))
    return tuple(accepted)


def _round_like_javascript(value: float, digits: int) -> float:
    """Match ``Math.round(value * 10**digits) / 10**digits`` exactly."""

    scale = 10 ** digits
    rounded = math.floor(value * scale + 0.5) / scale
    return 0.0 if rounded == 0 else rounded


def detect_carrier_delimited_multi_path_regions(
    data: CarrierDelimitedMultiPathDetectionInput,
) -> tuple[CarrierDelimitedMultiPathRegion, ...]:
    """Return strict repeated C-T-C regions without mutating pipeline state."""

    descriptors = _resolve_descriptors(data)
    candidates = _build_candidates(data, descriptors)
    by_topology: dict[str, list[_CarrierDelimitedCandidate]] = {}
    for candidate in candidates:
        by_topology.setdefault(candidate.topology, []).append(candidate)

    minimum_instances = max(
        DEFAULT_MINIMUM_INSTANCES,
        data.minimum_instances
        if data.minimum_instances is not None
        else DEFAULT_MINIMUM_INSTANCES,
    )
    accepted = [
        component
        for bucket in by_topology.values()
        for component in _accepted_components(bucket, minimum_instances)
    ]
    unique: dict[tuple[int, ...], tuple[_CarrierDelimitedCandidate, str]] = {}
    for component in accepted:
        chain_key = (
            f"{data.group_id}:{component[0].left_carrier.op_index}:"
            f"{component[-1].right_carrier.op_index}"
        )
        for candidate in component:
            token_key = tuple(
                descriptor.op_index for descriptor in candidate.token_descriptors
            )
            unique[token_key] = (candidate, chain_key)

    regions: list[CarrierDelimitedMultiPathRegion] = []
    for candidate, chain_key in unique.values():
        carrier_orientation_degrees = _round_like_javascript(
            math.atan2(candidate.carrier_axis.y, candidate.carrier_axis.x)
            * 180
            / math.pi,
            2,
        )
        principal_frame = _principal_frame_for(candidate.token_descriptors)
        if principal_frame is None:
            continue
        shape_orientation_degrees = (
            carrier_orientation_degrees
            if principal_frame.projected_aspect <= 1.12
            else principal_frame.orientation_degrees
        )
        token_indices = tuple(
            descriptor.op_index for descriptor in candidate.token_descriptors
        )
        regions.append(CarrierDelimitedMultiPathRegion(
            group_id=data.group_id,
            op_indices=token_indices,
            bounds=candidate.token_bounds,
            orientation_degrees=shape_orientation_degrees,
            confidence=0.96,
            evidence=CarrierDelimitedMultiPathEvidence(
                path_count=len(candidate.token_descriptors),
                segment_count=candidate.segment_count,
                angle_bin_count=candidate.angle_bin_count,
                paint_order_span=token_indices[-1] - token_indices[0] + 1,
                aspect_ratio=_round_like_javascript(
                    principal_frame.projected_aspect,
                    3,
                ),
                carrier_axis_degrees=carrier_orientation_degrees,
                sequential_multi_path_shape_key=candidate.topology,
                sequential_multi_path_chain_key=chain_key,
            ),
        ))
    return tuple(regions)


__all__ = [
    "CarrierDelimitedMultiPathDetectionInput",
    "CarrierDelimitedMultiPathEvidence",
    "CarrierDelimitedMultiPathRegion",
    "MultiPathCarrierDescriptor",
    "MultiPathCarrierEndpoint",
    "MultiPathCarrierPoint",
    "detect_carrier_delimited_multi_path_regions",
]
