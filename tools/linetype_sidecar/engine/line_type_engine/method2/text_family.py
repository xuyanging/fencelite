"""Repeated native/vector text-pattern families from frozen Method2 r46.

This module is the backend-only port of ``method2/text-family.ts``.  It owns
pattern signatures, complete-link identity, conservative missing-instance
completion, carrier attachment / endpoint routing, and the final Method2
result plus diagnostics.  OCR-free region detection remains in
``vector_text.py``.

Integer ownership is always the dense position in ``PageIR.operations``.
``paint_order`` is allowed to repeat and is used only as authored sequence
evidence; it is never emitted as an operation identity.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import heapq
import math
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from ..ir import BoundsIR, GroupingIR, PageIR, PathOperationIR, TextOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    NonLineType,
    RecognitionSummary,
    RecognizedGroup,
)
from .vector_text import VectorTextEvidence, VectorTextRegion


MATCH_THRESHOLD = 0.885
SEQUENTIAL_CARRIER_MATCH_THRESHOLD = 0.84
SEQUENTIAL_CARRIER_MAX_OP_GAP = 12
MAX_CARRIER_TO_PATTERN_MAJOR = 12
MAX_REGION_PATHS = 120
PAINT_ORDER_NEIGHBORHOOD = 96
CARRIER_RUN_STRAIGHTNESS = 0.7
MAX_DIAGNOSTIC_PAIRS = 12
MAX_CARRIER_MERGE_WORKERS = 8
MIN_PARALLEL_CARRIER_CANDIDATES = 64
MIN_PARALLEL_CARRIER_PAIRWISE_WORK = 25_000_000
_EPSILON = 1e-6
_DASH_NORMALIZATION = re.compile(r"[\u2010-\u2015\u2212]")
_INLINE_FEET_TOKEN = re.compile(r"^\d{1,3}(?:\.\d{1,3})?\s*['\u2019\u2032]$")


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class OrderedPathFeature:
    topology: str
    line_count: int
    curve_count: int
    subpath_count: int
    center_u: float
    center_v: float
    span_u: float
    span_v: float


@dataclass(frozen=True, slots=True)
class RegionFrame:
    ux: float
    uy: float
    vx: float
    vy: float
    min_u: float
    max_u: float
    min_v: float
    max_v: float
    major: float
    minor: float


@dataclass(frozen=True, slots=True)
class _PatternRegion:
    region_id: str
    group_id: str
    op_indices: tuple[int, ...]
    bounds: BoundsIR
    orientation_degrees: float
    confidence: float
    evidence: VectorTextEvidence
    pattern_source: str = "vector_strokes"
    literal_text: str | None = None

    @classmethod
    def from_vector(cls, region: VectorTextRegion) -> "_PatternRegion":
        return cls(
            region.region_id,
            region.group_id,
            tuple(region.op_indices),
            region.bounds,
            region.orientation_degrees,
            region.confidence,
            region.evidence,
        )


@dataclass(frozen=True, slots=True)
class RegionSignature:
    region: _PatternRegion
    frame: RegionFrame
    ordered: tuple[OrderedPathFeature, ...]
    occupancy: frozenset[int]
    path_count: int
    segment_count: int
    curve_count: int
    subpath_count: int
    aspect_ratio: float
    channel: str
    normalized_line_width: float
    canonical_key: str
    pattern_source: str
    literal_key: str | None = None
    inline_carrier_style_key: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class RegionSimilarityBreakdown:
    decision: str
    score: float
    match_threshold: float
    exact_ordered_topology: bool
    gates: Mapping[str, object]
    terms: Mapping[str, float | None]
    weights: Mapping[str, float]
    weighted_score: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "score": self.score,
            "match_threshold": self.match_threshold,
            "exact_ordered_topology": self.exact_ordered_topology,
            "gates": dict(self.gates),
            "terms": dict(self.terms),
            "weights": dict(self.weights),
            "weighted_score": self.weighted_score,
        }


@dataclass(frozen=True, slots=True)
class PatternInstanceDiagnostic:
    signature_index: int
    display_label: str
    group_id: str
    op_indices: tuple[int, ...]
    dimensions: Mapping[str, object]
    literal_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "signature_index": self.signature_index,
            "display_label": self.display_label,
            "group_id": self.group_id,
            "op_indices": list(self.op_indices),
            "dimensions": dict(self.dimensions),
        }
        if self.literal_text is not None:
            result["literal_text"] = self.literal_text
        return result


@dataclass(frozen=True, slots=True)
class TextPatternFamilyDiagnostic:
    global_type_id: str
    pattern_source: str
    minimum_pair_similarity: float
    instance_count: int
    pair_count: int
    instances: tuple[PatternInstanceDiagnostic, ...]
    pairs: tuple[Mapping[str, object], ...]
    literal_text: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "global_type_id": self.global_type_id,
            "pattern_source": self.pattern_source,
            "minimum_pair_similarity": self.minimum_pair_similarity,
            "instance_count": self.instance_count,
            "pair_count": self.pair_count,
            "instances": [item.to_dict() for item in self.instances],
            "pairs": [dict(item) for item in self.pairs],
        }
        if self.literal_text is not None:
            result["literal_text"] = self.literal_text
        return result


@dataclass(frozen=True, slots=True)
class PatternRegionInstance:
    display_label: str
    region_id: str
    group_id: str
    op_indices: tuple[int, ...]
    bounds: BoundsIR
    orientation_degrees: float
    confidence: float
    evidence: VectorTextEvidence
    matched: bool
    line_type_confirmed: bool
    recovered: bool
    pattern_source: str
    literal_text: str | None = None
    text_family_id: str | None = None
    global_type_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "display_label": self.display_label,
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
            "matched": self.matched,
            "line_type_confirmed": self.line_type_confirmed,
            "recovered": self.recovered,
            "pattern_source": self.pattern_source,
        }
        for name in ("literal_text", "text_family_id", "global_type_id"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result


@dataclass(frozen=True, slots=True)
class VectorTextFamilyAudit:
    family_diagnostics: tuple[TextPatternFamilyDiagnostic, ...]
    detected_region_count: int
    eligible_region_count: int
    matched_instance_count: int
    matched_family_count: int
    dash_connected_family_count: int
    line_type_confirmed_instance_count: int
    line_type_confirmed_text_op_count: int
    matched_text_op_count: int
    attached_dash_op_count: int
    bridged_route_op_count: int
    matched_text_op_indices: tuple[int, ...]
    line_type_confirmed_text_op_indices: tuple[int, ...]
    affected_group_ids: tuple[str, ...]
    region_instances: tuple[PatternRegionInstance, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "family_diagnostics": [item.to_dict() for item in self.family_diagnostics],
            "detected_region_count": self.detected_region_count,
            "eligible_region_count": self.eligible_region_count,
            "matched_instance_count": self.matched_instance_count,
            "matched_family_count": self.matched_family_count,
            "dash_connected_family_count": self.dash_connected_family_count,
            "line_type_confirmed_instance_count": self.line_type_confirmed_instance_count,
            "line_type_confirmed_text_op_count": self.line_type_confirmed_text_op_count,
            "matched_text_op_count": self.matched_text_op_count,
            "attached_dash_op_count": self.attached_dash_op_count,
            "bridged_route_op_count": self.bridged_route_op_count,
            "matched_text_op_indices": list(self.matched_text_op_indices),
            "line_type_confirmed_text_op_indices": list(
                self.line_type_confirmed_text_op_indices
            ),
            "affected_group_ids": list(self.affected_group_ids),
            "region_instances": [item.to_dict() for item in self.region_instances],
        }


@dataclass(frozen=True, slots=True)
class TextFamilyRecognition:
    result: LineTypeRecognitionResult
    audit: VectorTextFamilyAudit


@dataclass(slots=True)
class _Family:
    indices: list[int]
    minimum_similarity: float


@dataclass(frozen=True, slots=True)
class _Context:
    page: PageIR
    grouping: GroupingIR
    operation_index: PageOperationIndex
    paint_orders: tuple[int, ...]
    paint_position_by_order: Mapping[int, int]

    @classmethod
    def build(cls, page: PageIR, grouping: GroupingIR) -> "_Context":
        operation_index = PageOperationIndex.build(page, grouping)
        paint_orders = tuple(operation_index.indices_by_paint_order)
        return cls(
            page,
            grouping,
            operation_index,
            paint_orders,
            {order: index for index, order in enumerate(paint_orders)},
        )

    def paint_position(self, dense_index: int) -> int:
        operation = self.operation_index.operation(dense_index)
        return self.paint_position_by_order[operation.paint_order]


def _js_round(value: float, digits: int = 0) -> float:
    scale = 10 ** digits
    rounded = math.floor(value * scale + 0.5) / scale
    return 0.0 if rounded == 0 else rounded


def _median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _bounds_corners(bounds: BoundsIR) -> tuple[Point, ...]:
    return (
        Point(bounds.min_x, bounds.min_y),
        Point(bounds.max_x, bounds.min_y),
        Point(bounds.min_x, bounds.max_y),
        Point(bounds.max_x, bounds.max_y),
    )


def _path_samples(operation: PathOperationIR) -> tuple[Point, ...]:
    points: list[Point] = []
    current: Point | None = None
    subpath_start: Point | None = None
    for segment in operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = Point(*segment.end)
            subpath_start = current
            points.append(current)
            continue
        if segment.kind == "close":
            if current is not None and subpath_start is not None:
                for step in range(1, 5):
                    factor = step / 4.0
                    points.append(Point(
                        current.x + (subpath_start.x - current.x) * factor,
                        current.y + (subpath_start.y - current.y) * factor,
                    ))
                current = subpath_start
            continue
        assert segment.end is not None
        end = Point(*segment.end)
        if current is None:
            current = end
            subpath_start = end
            points.append(end)
            continue
        if segment.kind == "line":
            for step in range(1, 5):
                factor = step / 4.0
                points.append(Point(
                    current.x + (end.x - current.x) * factor,
                    current.y + (end.y - current.y) * factor,
                ))
        else:
            assert segment.control_1 is not None and segment.control_2 is not None
            control_1 = Point(*segment.control_1)
            control_2 = Point(*segment.control_2)
            for step in range(1, 8):
                factor = step / 7.0
                inverse = 1.0 - factor
                points.append(Point(
                    inverse ** 3 * current.x
                    + 3 * inverse * inverse * factor * control_1.x
                    + 3 * inverse * factor * factor * control_2.x
                    + factor ** 3 * end.x,
                    inverse ** 3 * current.y
                    + 3 * inverse * inverse * factor * control_1.y
                    + 3 * inverse * factor * factor * control_2.y
                    + factor ** 3 * end.y,
                ))
        current = end
    return tuple(points)


def _projection(point: Point, ux: float, uy: float) -> float:
    return point.x * ux + point.y * uy


def _topology_for(operation: PathOperationIR) -> str:
    return "".join({
        "move": "M", "line": "L", "curve": "C", "close": "Z",
    }[segment.kind] for segment in operation.segments)


def _channel_for(operation: PathOperationIR) -> str:
    return ("S" if operation.stroke else "") + ("F" if operation.fill else "")


def _frame_for(
    region: _PatternRegion,
    paths: Sequence[PathOperationIR],
) -> RegionFrame | None:
    points = tuple(point for operation in paths for point in _path_samples(operation))
    if not points:
        return None
    theta = region.orientation_degrees * math.pi / 180.0
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux

    def spans() -> tuple[list[float], list[float], float, float]:
        along = [_projection(point, ux, uy) for point in points]
        across = [_projection(point, vx, vy) for point in points]
        return along, across, max(along) - min(along), max(across) - min(across)

    along, across, span_u, span_v = spans()
    if span_v > span_u:
        theta += math.pi / 2.0
        ux, uy = math.cos(theta), math.sin(theta)
        vx, vy = -uy, ux
        along, across, span_u, span_v = spans()
    if span_u <= _EPSILON or span_v <= _EPSILON:
        return None
    return RegionFrame(
        ux, uy, vx, vy,
        min(along), max(along), min(across), max(across), span_u, span_v,
    )


def _carrier_axis_frame_for(
    region: _PatternRegion,
    paths: Sequence[PathOperationIR],
) -> RegionFrame | None:
    degrees = region.evidence.carrier_axis_degrees
    if degrees is None or not math.isfinite(degrees):
        return None
    points = tuple(point for operation in paths for point in _path_samples(operation))
    if not points:
        return None
    theta = degrees * math.pi / 180.0
    ux, uy = math.cos(theta), math.sin(theta)
    vx, vy = -uy, ux
    along = [_projection(point, ux, uy) for point in points]
    across = [_projection(point, vx, vy) for point in points]
    span_u = max(along) - min(along)
    span_v = max(across) - min(across)
    if span_u <= _EPSILON or span_v <= _EPSILON:
        return None
    return RegionFrame(
        ux, uy, vx, vy,
        min(along), max(along), min(across), max(across), span_u, span_v,
    )


def _quantize(value: float, bins: int) -> int:
    return max(0, min(bins - 1, int(_js_round(value * (bins - 1)))))


def _signature_variant(
    paths: Sequence[PathOperationIR],
    frame: RegionFrame,
    sign: int,
) -> tuple[tuple[OrderedPathFeature, ...], frozenset[int], str]:
    def normalize(point: Point) -> tuple[float, float]:
        raw_u = _projection(point, frame.ux, frame.uy)
        raw_v = _projection(point, frame.vx, frame.vy)
        u = (raw_u - frame.min_u) / frame.major
        v = (raw_v - frame.min_v) / frame.minor
        return (u, v) if sign > 0 else (1.0 - u, 1.0 - v)

    occupancy: set[int] = set()
    ordered: list[OrderedPathFeature] = []
    key_items: list[str] = []
    for operation in paths:
        for point in _path_samples(operation):
            u, v = normalize(point)
            occupancy.add(_quantize(v, 10) * 20 + _quantize(u, 20))
        normalized_corners = [normalize(point) for point in _bounds_corners(operation.bounds)]
        values_u = [point[0] for point in normalized_corners]
        values_v = [point[1] for point in normalized_corners]
        feature = OrderedPathFeature(
            _topology_for(operation),
            sum(segment.kind == "line" for segment in operation.segments),
            sum(segment.kind == "curve" for segment in operation.segments),
            sum(segment.kind == "move" for segment in operation.segments),
            (min(values_u) + max(values_u)) / 2.0,
            (min(values_v) + max(values_v)) / 2.0,
            max(values_u) - min(values_u),
            max(values_v) - min(values_v),
        )
        ordered.append(feature)
        key_items.append(":".join(map(str, (
            feature.topology,
            _quantize(feature.center_u, 24),
            _quantize(feature.center_v, 16),
            _quantize(feature.span_u, 16),
            _quantize(feature.span_v, 16),
        ))))
    key = "|".join(key_items) + "/" + ",".join(map(str, sorted(occupancy)))
    return tuple(ordered), frozenset(occupancy), key


def _signature_for(
    context: _Context,
    region: _PatternRegion,
    page_diagonal: float,
) -> RegionSignature | None:
    operations = [context.operation_index.operation(index) for index in region.op_indices]
    if any(not isinstance(operation, PathOperationIR) for operation in operations):
        return None
    paths = tuple(operation for operation in operations if isinstance(operation, PathOperationIR))
    if not paths or len(paths) > MAX_REGION_PATHS:
        return None
    frame = _frame_for(region, paths)
    if frame is None:
        return None
    if (
        frame.major > page_diagonal * 0.15
        or frame.minor > page_diagonal * 0.045
        or frame.major < page_diagonal * 0.00045
        or frame.minor < page_diagonal * 0.00012
    ):
        return None
    positive = _signature_variant(paths, frame, 1)
    negative = _signature_variant(paths, frame, -1)
    ordered, occupancy, canonical_key = min(positive, negative, key=lambda item: item[2])
    line_count = sum(
        segment.kind == "line" for path in paths for segment in path.segments
    )
    curve_count = sum(
        segment.kind == "curve" for path in paths for segment in path.segments
    )
    subpath_count = sum(
        segment.kind == "move" for path in paths for segment in path.segments
    )
    channel_counts: dict[str, int] = {}
    for path in paths:
        channel = _channel_for(path)
        channel_counts[channel] = channel_counts.get(channel, 0) + 1
    channel = min(channel_counts, key=lambda item: (-channel_counts[item], item))
    return RegionSignature(
        region, frame, ordered, occupancy, len(paths), line_count + curve_count,
        curve_count, subpath_count, frame.major / max(_EPSILON, frame.minor),
        channel,
        _median(max(0.0, path.line_width) for path in paths) / max(_EPSILON, frame.minor),
        canonical_key, "vector_strokes",
    )


def _normalized_pdf_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _DASH_NORMALIZATION.sub("-", normalized)
    return " ".join(normalized.strip().split()).upper()


def _javascript_string_length(value: str) -> int:
    """Return JavaScript ``String.length`` (UTF-16 code units)."""

    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def _unicode_letter(value: str) -> bool:
    # This is the Python spelling of frozen ``/\p{L}/u`` rather than the
    # slightly broader, implementation-defined ``str.isalpha`` predicate.
    return any(unicodedata.category(character).startswith("L") for character in value)


def _is_inline_feet_token(value: str) -> bool:
    """Whether a non-letter literal can be an authored inline line pattern.

    Native PDF text historically required a Unicode letter.  That keeps
    ordinary dimensions, coordinates and page numbers out of Method2, but it
    also rejects genuine repeated fence tokens such as ``8'`` before carrier
    evidence is considered.  Keep the exception deliberately narrow: a short
    number followed by an explicit feet mark.  Geometry below must still prove
    that the token is literally bracketed by its carrier.
    """

    return _INLINE_FEET_TOKEN.fullmatch(value) is not None


def _inline_carrier_paths(
    context: _Context,
    dense_index: int,
) -> tuple[PathOperationIR, PathOperationIR] | None:
    if dense_index <= 0 or dense_index + 1 >= len(context.page.operations):
        return None
    group_id = context.operation_index.group_id(dense_index)
    neighbor_indices = (dense_index - 1, dense_index + 1)
    if any(
        context.operation_index.group_id(index) != group_id
        for index in neighbor_indices
    ):
        return None
    neighbors = tuple(
        context.operation_index.operation(index) for index in neighbor_indices
    )
    if any(not isinstance(operation, PathOperationIR) for operation in neighbors):
        return None
    left, right = neighbors
    assert isinstance(left, PathOperationIR)
    assert isinstance(right, PathOperationIR)
    return left, right


def _inline_carrier_path_style(operation: PathOperationIR) -> tuple[object, ...]:
    """Visual style identity for a numeric token's surrounding carrier.

    Source provenance and layer names are intentionally absent: equal-looking
    line types may be authored in different streams or optional-content
    layers.  Paint channel, color, opacity, width, dash, cap/join, hairline and
    blend mode do change the visible carrier and therefore must prevent a
    literal-only cross-style merge.  Rounding only removes representation
    noise from source transforms; it is much narrower than the admission
    width-ratio guard.
    """

    return (
        _channel_for(operation),
        tuple(round(value, 9) for value in (operation.stroke_color or ())),
        round(operation.stroke_opacity, 6),
        round(operation.line_width, 3),
        operation.hairline,
        tuple(round(value, 3) for value in operation.dash_array),
        round(operation.dash_phase, 3),
        operation.line_cap,
        round(operation.line_join, 3),
        operation.blend_mode,
    )


def _inline_carrier_style_key(
    context: _Context,
    dense_index: int,
) -> tuple[object, ...] | None:
    paths = _inline_carrier_paths(context, dense_index)
    if paths is None:
        return None
    left_style, right_style = map(_inline_carrier_path_style, paths)
    return left_style if left_style == right_style else None


def _strict_inline_carrier_sandwich(
    context: _Context,
    dense_index: int,
    frame: RegionFrame,
    page_diagonal: float,
) -> bool:
    """Require ``carrier, token, carrier`` in one authored Group.

    The two carrier paths must be the dense operations immediately around the
    text, be open straight strokes, lie on opposite sides of the text frame,
    and agree with its writing axis.  This is intentionally stronger than the
    later family attachment search: it is the admission guard that prevents a
    repeated dimension value elsewhere on the sheet from entering Method2.
    """

    paths = _inline_carrier_paths(context, dense_index)
    if paths is None or _inline_carrier_style_key(context, dense_index) is None:
        return False
    if any(
        operation.fill
        or any(segment.kind in {"curve", "close"} for segment in operation.segments)
        or sum(segment.kind == "move" for segment in operation.segments) != 1
        for operation in paths
    ):
        return False
    metrics = tuple(_straight_path_metrics(operation) for operation in paths)
    if any(metric is None for metric in metrics):
        return False
    positive_widths = [operation.line_width for operation in paths if operation.line_width > 0]
    if len(positive_widths) == 2 and _ratio_similarity(*positive_widths) < 0.6:
        return False

    gap_tolerance = max(frame.minor * 2.2, page_diagonal * 0.0015)
    sides: set[str] = set()
    for operation, metric in zip(paths, metrics):
        assert metric is not None
        alignment = abs(metric.direction.x * frame.ux + metric.direction.y * frame.uy)
        if alignment < 0.93:
            return False
        corners = _bounds_corners(operation.bounds)
        values_u = [_projection(point, frame.ux, frame.uy) for point in corners]
        values_v = [_projection(point, frame.vx, frame.vy) for point in corners]
        v_gap = _interval_gap(
            min(values_v), max(values_v), frame.min_v, frame.max_v
        )
        if v_gap > frame.minor * 0.85:
            return False
        u_min, u_max = min(values_u), max(values_u)
        if u_max <= frame.min_u + frame.minor * 0.35:
            side = "left"
            gap = max(0.0, frame.min_u - u_max)
        elif u_min >= frame.max_u - frame.minor * 0.35:
            side = "right"
            gap = max(0.0, u_min - frame.max_u)
        else:
            return False
        if gap > max(gap_tolerance, frame.major * 5.0):
            return False
        sides.add(side)
    return sides == {"left", "right"}


def _contextual_inline_carrier_corner(
    context: _Context,
    dense_index: int,
    frame: RegionFrame,
) -> bool:
    """Admit a turn token only inside a run already proven by strict anchors."""

    paths = _inline_carrier_paths(context, dense_index)
    if paths is None or _inline_carrier_style_key(context, dense_index) is None:
        return False
    for index, operation in zip((dense_index - 1, dense_index + 1), paths):
        if (
            operation.fill
            or _carrier_geometry_for(operation, index) is None
        ):
            return False
    metrics = tuple(_straight_path_metrics(operation) for operation in paths)
    return any(
        metric is not None
        and abs(metric.direction.x * frame.ux + metric.direction.y * frame.uy) >= 0.93
        for metric in metrics
    )


def _source_text_point(
    matrix: tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> Point:
    return Point(
        matrix[0] * x + matrix[2] * y + matrix[4],
        matrix[1] * x + matrix[3] * y + matrix[5],
    )


def _native_text_signature_for(
    context: _Context,
    operation: TextOperationIR,
    dense_index: int,
    page_diagonal: float,
    *,
    allow_contextual_inline: bool = False,
) -> RegionSignature | None:
    literal_key = _normalized_pdf_text(operation.literal_text)
    contains_letter = _unicode_letter(literal_key)
    inline_feet_token = _is_inline_feet_token(literal_key)
    if (
        not literal_key
        or _javascript_string_length(literal_key) > 80
        or not (contains_letter or inline_feet_token)
    ):
        return None
    if operation.source_matrix is not None:
        assert operation.source_glyph_advance is not None
        assert operation.source_horizontal_scale is not None
        assert operation.source_rise is not None
        assert operation.source_font_size is not None
        matrix = operation.source_matrix
        advance = operation.source_glyph_advance * operation.source_horizontal_scale
        lower_y = operation.source_rise - operation.source_font_size * 0.25
        upper_y = operation.source_rise + operation.source_font_size * 0.88
        points = (
            _source_text_point(matrix, 0.0, lower_y),
            _source_text_point(matrix, advance, lower_y),
            _source_text_point(matrix, advance, upper_y),
            _source_text_point(matrix, 0.0, upper_y),
        )
        direction_length = math.hypot(matrix[0], matrix[1])
        if direction_length <= 1e-9:
            return None
        ux = matrix[0] / direction_length
        uy = matrix[1] / direction_length
    else:
        # Legacy/synthetic PageIR retains the pre-p10 trace-derived frame.
        direction_length = math.hypot(*operation.direction)
        if direction_length <= 1e-9:
            return None
        ux = operation.direction[0] / direction_length
        uy = operation.direction[1] / direction_length
        points = _bounds_corners(operation.bounds)
    vx, vy = -uy, ux
    along = [_projection(point, ux, uy) for point in points]
    across = [_projection(point, vx, vy) for point in points]
    frame = RegionFrame(
        ux, uy, vx, vy,
        min(along), max(along), min(across), max(across),
        max(along) - min(along), max(across) - min(across),
    )
    if (
        frame.major > page_diagonal * 0.15
        or frame.minor > page_diagonal * 0.045
        or frame.major < page_diagonal * 0.0002
        or frame.minor < page_diagonal * 0.0001
    ):
        return None
    if inline_feet_token:
        strict = _strict_inline_carrier_sandwich(
            context, dense_index, frame, page_diagonal
        )
        if not strict and not (
            allow_contextual_inline
            and _contextual_inline_carrier_corner(context, dense_index, frame)
        ):
            return None
        inline_carrier_style_key = _inline_carrier_style_key(context, dense_index)
        if inline_carrier_style_key is None:
            return None
    else:
        inline_carrier_style_key = None
    group_id = context.operation_index.group_id(dense_index)
    orientation = math.atan2(uy, ux) * 180.0 / math.pi
    region = _PatternRegion(
        f"pdf_text_{group_id.zfill(4)}_{dense_index}",
        group_id,
        (dense_index,),
        operation.bounds,
        orientation,
        1.0,
        VectorTextEvidence(
            0, 0, 0, 1,
            frame.major / max(_EPSILON, frame.minor),
            carrier_axis_degrees=orientation,
        ),
        "pdf_text",
        literal_key,
    )
    return RegionSignature(
        region, frame, (), frozenset(), 0, 0, 0, 0,
        frame.major / max(_EPSILON, frame.minor), "", 0.0,
        f"pdf-text:{literal_key}", "pdf_text", literal_key,
        inline_carrier_style_key,
    )


def _ratio_similarity(left: float, right: float) -> float:
    return min(abs(left), abs(right)) / max(_EPSILON, abs(left), abs(right))


def _path_feature_similarity(
    left: OrderedPathFeature,
    right: OrderedPathFeature,
) -> float:
    exact_topology = left.topology == right.topology
    topology = 1.0 if exact_topology else (
        _ratio_similarity(left.line_count + 1, right.line_count + 1) * 0.45
        + _ratio_similarity(left.curve_count + 1, right.curve_count + 1) * 0.35
        + _ratio_similarity(left.subpath_count + 1, right.subpath_count + 1) * 0.20
    )
    position_distance = math.hypot(
        left.center_u - right.center_u, left.center_v - right.center_v
    )
    size_distance = math.hypot(left.span_u - right.span_u, left.span_v - right.span_v)
    return (
        topology * 0.56
        + math.exp(-position_distance * 7.0) * 0.29
        + math.exp(-size_distance * 6.0) * 0.15
    )


def _ordered_similarity(
    left: Sequence[OrderedPathFeature],
    right: Sequence[OrderedPathFeature],
) -> float:
    longer, shorter = (left, right) if len(left) >= len(right) else (right, left)
    if not shorter:
        return 0.0
    total = 0.0
    for index, feature in enumerate(longer):
        expected = 0.0 if len(longer) == 1 else (
            index / (len(longer) - 1) * max(0, len(shorter) - 1)
        )
        center = int(_js_round(expected))
        best = 0.0
        for offset in (-1, 0, 1):
            candidate = center + offset
            if 0 <= candidate < len(shorter):
                best = max(best, _path_feature_similarity(feature, shorter[candidate]))
        total += best
    return total / len(longer) * _ratio_similarity(len(left), len(right))


def _occupancy_similarity(left: frozenset[int], right: frozenset[int]) -> float:
    intersection = len(left & right)
    return intersection / max(1, len(left) + len(right) - intersection)


def _has_exact_ordered_topology(left: RegionSignature, right: RegionSignature) -> bool:
    return (
        left.path_count == right.path_count
        and left.segment_count == right.segment_count
        and left.curve_count == right.curve_count
        and left.subpath_count == right.subpath_count
        and len(left.ordered) == len(right.ordered)
        and all(
            feature.topology == other.topology
            and feature.line_count == other.line_count
            and feature.curve_count == other.curve_count
            and feature.subpath_count == other.subpath_count
            for feature, other in zip(left.ordered, right.ordered)
        )
    )


_TERM_WEIGHTS = {
    "occupancy": 0.34, "order": 0.36, "topology": 0.15,
    "aspect": 0.10, "stroke": 0.05,
}


def _compare_region_signatures(
    left: RegionSignature,
    right: RegionSignature,
) -> tuple[float, RegionSimilarityBreakdown]:
    gates: dict[str, object] = {
        "channel_equal": left.channel == right.channel,
        "path_count_ratio": None,
        "path_count_limit": 0.72,
        "segment_count_ratio": None,
        "segment_count_limit": 0.76,
        "aspect_ratio_ratio": None,
        "aspect_ratio_limit": 0.66,
        "order": None,
        "order_limit": 0.76,
    }
    terms: dict[str, float | None] = {
        "occupancy": None, "order": None, "topology": None,
        "aspect": None, "stroke": None,
    }
    weighted_score: float | None = None

    def done(decision: str, score: float, exact: bool) -> tuple[float, RegionSimilarityBreakdown]:
        return score, RegionSimilarityBreakdown(
            decision, score, MATCH_THRESHOLD, exact,
            dict(gates), dict(terms), dict(_TERM_WEIGHTS), weighted_score,
        )

    if left.pattern_source == "pdf_text" or right.pattern_source == "pdf_text":
        score = float(
            left.pattern_source == right.pattern_source
            and left.literal_key == right.literal_key
            and (
                not _is_inline_feet_token(left.literal_key or "")
                or left.inline_carrier_style_key
                == right.inline_carrier_style_key
            )
        )
        return done("pdf_text_literal", score, False)
    if left.channel != right.channel:
        return done("channel_mismatch", 0.0, False)
    exact = _has_exact_ordered_topology(left, right)
    same_single = (
        left.path_count == right.path_count == 1
        and bool(left.region.evidence.single_stroke_shape_key)
        and left.region.evidence.single_stroke_shape_key
        == right.region.evidence.single_stroke_shape_key
        and left.region.evidence.carrier_axis_degrees is not None
        and right.region.evidence.carrier_axis_degrees is not None
    )
    same_multi = (
        left.path_count >= 2
        and right.path_count >= 2
        and left.region.group_id == right.region.group_id
        and bool(left.region.evidence.sequential_multi_path_chain_key)
        and left.region.evidence.sequential_multi_path_chain_key
        == right.region.evidence.sequential_multi_path_chain_key
        and left.region.evidence.sequential_multi_path_shape_key
        == right.region.evidence.sequential_multi_path_shape_key
    )
    gates["path_count_ratio"] = _ratio_similarity(left.path_count, right.path_count)
    if float(gates["path_count_ratio"]) < 0.72:
        return done("path_count_gate", 0.0, exact)
    gates["segment_count_ratio"] = _ratio_similarity(
        left.segment_count, right.segment_count
    )
    if float(gates["segment_count_ratio"]) < 0.76:
        return done("segment_count_gate", 0.0, exact)
    if same_single and exact:
        return done("guarded_single_stroke", MATCH_THRESHOLD + 0.055, exact)
    if same_multi and exact:
        return done("guarded_multi_path_chain", MATCH_THRESHOLD + 0.055, exact)
    gates["aspect_ratio_ratio"] = _ratio_similarity(left.aspect_ratio, right.aspect_ratio)
    if not exact and float(gates["aspect_ratio_ratio"]) < 0.66:
        return done("aspect_gate", 0.0, exact)
    order = _ordered_similarity(left.ordered, right.ordered)
    gates["order"] = order
    if order < 0.76:
        return done("order_gate", 0.0, exact)
    occupancy = _occupancy_similarity(left.occupancy, right.occupancy)
    topology = (
        _ratio_similarity(left.segment_count + 1, right.segment_count + 1) * 0.45
        + _ratio_similarity(left.curve_count + 1, right.curve_count + 1) * 0.30
        + _ratio_similarity(left.subpath_count + 1, right.subpath_count + 1) * 0.25
    )
    aspect = math.exp(-abs(math.log(
        max(_EPSILON, left.aspect_ratio) / max(_EPSILON, right.aspect_ratio)
    )) * 2.2)
    stroke = (
        1.0
        if max(left.normalized_line_width, right.normalized_line_width) <= _EPSILON
        else _ratio_similarity(left.normalized_line_width, right.normalized_line_width)
    )
    score = max(0.0, min(1.0,
        occupancy * 0.34 + order * 0.36 + topology * 0.15
        + aspect * 0.10 + stroke * 0.05
    ))
    terms.update({
        "occupancy": occupancy, "order": order, "topology": topology,
        "aspect": aspect, "stroke": stroke,
    })
    weighted_score = score
    if (
        exact and left.path_count >= 4 and order >= 0.72 and occupancy >= 0.22
        and _ratio_similarity(left.aspect_ratio, right.aspect_ratio) >= 0.5
    ):
        floor = MATCH_THRESHOLD + min(
            0.08, occupancy * 0.035 + order * 0.035 + aspect * 0.01
        )
        return done("exact_topology_floor", max(score, floor), exact)
    return done("weighted", score, exact)


def vector_text_region_similarity(left: RegionSignature, right: RegionSignature) -> float:
    return _compare_region_signatures(left, right)[0]


def explain_region_similarity(
    left: RegionSignature,
    right: RegionSignature,
) -> RegionSimilarityBreakdown:
    return _compare_region_signatures(left, right)[1]


def pattern_instance_dimensions(signature: RegionSignature) -> dict[str, object]:
    return {
        "path_count": signature.path_count,
        "segment_count": signature.segment_count,
        "curve_count": signature.curve_count,
        "subpath_count": signature.subpath_count,
        "aspect_ratio": signature.aspect_ratio,
        "normalized_line_width": signature.normalized_line_width,
        "channel": signature.channel,
        "occupancy_bins": len(signature.occupancy),
        "frame_major": signature.frame.major,
        "frame_minor": signature.frame.minor,
        "ordered_topology": [feature.topology for feature in signature.ordered],
    }


def rebuild_region_signature(
    page: PageIR,
    grouping: GroupingIR,
    region: VectorTextRegion,
) -> RegionSignature | None:
    context = _Context.build(page, grouping)
    diagonal = math.hypot(page.page_bounds.width, page.page_bounds.height)
    return _signature_for(context, _PatternRegion.from_vector(region), diagonal)


def explain_region_pair(
    page: PageIR,
    grouping: GroupingIR,
    left_region: VectorTextRegion,
    right_region: VectorTextRegion,
) -> dict[str, object] | None:
    left = rebuild_region_signature(page, grouping, left_region)
    right = rebuild_region_signature(page, grouping, right_region)
    if left is None or right is None:
        return None
    return {
        "left": pattern_instance_dimensions(left),
        "right": pattern_instance_dimensions(right),
        "left_literal": left.literal_key,
        "right_literal": right.literal_key,
        "detail": explain_region_similarity(left, right).to_dict(),
    }


def _complete_link_families(signatures: Sequence[RegionSignature]) -> list[_Family]:
    similarities: dict[tuple[int, int], float] = {}
    edges: list[tuple[float, int, int]] = []
    for left in range(len(signatures)):
        for right in range(left + 1, len(signatures)):
            similarity = vector_text_region_similarity(signatures[left], signatures[right])
            similarities[(left, right)] = similarity
            if similarity >= MATCH_THRESHOLD:
                edges.append((-similarity, left, right))
    edges.sort()
    parent = list(range(len(signatures)))
    members: dict[int, list[int]] = {index: [index] for index in parent}

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for _negative, left, right in edges:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            continue
        left_members = members[left_root]
        right_members = members[right_root]
        if not all(
            similarities.get(tuple(sorted((a, b))), 0.0) >= MATCH_THRESHOLD
            for a in left_members for b in right_members
        ):
            continue
        keep, drop = min(left_root, right_root), max(left_root, right_root)
        parent[drop] = keep
        members[keep] = sorted((*left_members, *right_members))
        del members[drop]
    result: list[_Family] = []
    for family in members.values():
        if len(family) < 2:
            continue
        minimum = min(
            similarities.get((left, right), 0.0)
            for order, left in enumerate(family)
            for right in family[order + 1:]
        )
        result.append(_Family(family, minimum))
    result.sort(key=lambda family: signatures[family.indices[0]].region.op_indices[0])
    return result


def _complete_text_pattern_families(
    context: _Context,
    signatures: Sequence[RegionSignature],
) -> list[_Family]:
    vector_indices = [
        index for index, signature in enumerate(signatures)
        if signature.pattern_source == "vector_strokes"
    ]
    vector_families = _complete_link_families(
        [signatures[index] for index in vector_indices]
    )
    for family in vector_families:
        family.indices[:] = [vector_indices[index] for index in family.indices]
    literal_buckets: dict[
        tuple[str, tuple[object, ...] | None], list[int]
    ] = {}
    for index, signature in enumerate(signatures):
        if signature.pattern_source == "pdf_text" and signature.literal_key:
            style_key = (
                signature.inline_carrier_style_key
                if _is_inline_feet_token(signature.literal_key)
                else None
            )
            literal_buckets.setdefault(
                (signature.literal_key, style_key), []
            ).append(index)
    literal_families: list[_Family] = []
    for (literal_key, _style_key), indices in literal_buckets.items():
        paint_orders = {
            context.operation_index.operation(
                signatures[index].region.op_indices[0]
            ).paint_order
            for index in indices
        }
        if len(paint_orders) < 2:
            continue
        if _is_inline_feet_token(literal_key):
            paint_orders_by_group: dict[str, set[int]] = {}
            for index in indices:
                signature = signatures[index]
                paint_orders_by_group.setdefault(
                    signature.region.group_id, set()
                ).add(
                    context.operation_index.operation(
                        signature.region.op_indices[0]
                    ).paint_order
                )
            # Numeric literals are weak identity on their own.  Require a real
            # repeated run in one authored Group rather than duplicate trace
            # spans or two coincidental dimension labels.  ``paint_order`` is
            # the authored event identity and may intentionally repeat across
            # multiple dense TextOperationIR spans.
            if max(map(len, paint_orders_by_group.values()), default=0) < 3:
                continue
        literal_families.append(_Family(indices, 1.0))
    result = [*vector_families, *literal_families]
    result.sort(key=lambda family: signatures[family.indices[0]].region.op_indices[0])
    return result


def _orientation_for_paths(paths: Sequence[PathOperationIR]) -> float:
    centers = [Point(
        (path.bounds.min_x + path.bounds.max_x) / 2.0,
        (path.bounds.min_y + path.bounds.max_y) / 2.0,
    ) for path in paths]
    mean_x = sum(point.x for point in centers) / max(1, len(centers))
    mean_y = sum(point.y for point in centers) / max(1, len(centers))
    xx = yy = xy = 0.0
    for point in centers:
        dx, dy = point.x - mean_x, point.y - mean_y
        xx += dx * dx
        yy += dy * dy
        xy += dx * dy
    return _js_round(0.5 * math.atan2(2 * xy, xx - yy) * 180.0 / math.pi, 2)


def _union_path_bounds(paths: Sequence[PathOperationIR]) -> BoundsIR:
    bounds = paths[0].bounds
    for path in paths[1:]:
        bounds = bounds.union(path.bounds)
    return bounds


def _recover_missing_family_signatures(
    context: _Context,
    regions: list[_PatternRegion],
    signatures: list[RegionSignature],
    families: list[_Family],
    page_diagonal: float,
    allowed_family_indices: set[int] | None = None,
) -> list[_PatternRegion]:
    occupied = {index for region in regions for index in region.op_indices}
    proposals: dict[tuple[str, int, int], tuple[
        int, _PatternRegion, RegionSignature, float,
    ]] = {}
    for family_index, family in enumerate(families):
        if allowed_family_indices is not None and family_index not in allowed_family_indices:
            continue
        if any(signatures[index].pattern_source != "vector_strokes" for index in family.indices):
            continue
        template_keys = {
            "|".join(feature.topology for feature in signatures[index].ordered)
            for index in family.indices
        }
        path_counts = sorted({signatures[index].path_count for index in family.indices})
        group_ids = tuple(dict.fromkeys(
            signatures[index].region.group_id for index in family.indices
        ))
        for group_id in group_ids:
            group_indices = context.operation_index.group_indices(group_id)
            for path_count in path_counts:
                for offset in range(0, len(group_indices) - path_count + 1):
                    op_indices = tuple(group_indices[offset:offset + path_count])
                    if any(index in occupied for index in op_indices):
                        continue
                    operations = [context.operation_index.operation(index) for index in op_indices]
                    if any(not isinstance(operation, PathOperationIR) for operation in operations):
                        continue
                    paths = tuple(
                        operation for operation in operations
                        if isinstance(operation, PathOperationIR)
                    )
                    if "|".join(_topology_for(path) for path in paths) not in template_keys:
                        continue
                    bounds = _union_path_bounds(paths)
                    segment_count = sum(
                        segment.kind not in {"move", "close"}
                        for path in paths for segment in path.segments
                    )
                    width, height = bounds.width, bounds.height
                    base = _PatternRegion(
                        f"vector_text_recovered_{group_id.zfill(4)}_{op_indices[0]}",
                        group_id,
                        op_indices,
                        bounds,
                        _orientation_for_paths(paths),
                        1.0,
                        VectorTextEvidence(
                            path_count,
                            segment_count,
                            0,
                            (
                                context.paint_position(op_indices[-1])
                                - context.paint_position(op_indices[0]) + 1
                            ),
                            max(width, height) / max(0.001, min(width, height)),
                        ),
                    )
                    best: tuple[_PatternRegion, RegionSignature, float] | None = None
                    orientations = {
                        base.orientation_degrees,
                        *(signatures[index].region.orientation_degrees for index in family.indices),
                    }
                    for orientation in orientations:
                        candidate = replace(base, orientation_degrees=orientation)
                        signature = _signature_for(context, candidate, page_diagonal)
                        if signature is None:
                            continue
                        minimum: float | None = 1.0
                        for index in family.indices:
                            similarity = vector_text_region_similarity(
                                signatures[index], signature
                            )
                            # Complete-link admission requires every pair to
                            # pass.  A failing orientation cannot win the
                            # proposal; accepted orientations still visit all
                            # members so their persisted minimum is unchanged.
                            if similarity < MATCH_THRESHOLD:
                                minimum = None
                                break
                            minimum = min(minimum, similarity)
                        if minimum is None:
                            continue
                        if best is None or minimum > best[2]:
                            best = (candidate, signature, minimum)
                    if best is None or best[2] < MATCH_THRESHOLD:
                        continue
                    key = (group_id, op_indices[0], op_indices[-1])
                    current = proposals.get(key)
                    if current is None or best[2] > current[3]:
                        proposals[key] = (family_index, *best)

    recovered: list[_PatternRegion] = []
    for family_index, region, signature, minimum in sorted(
        proposals.values(), key=lambda item: item[1].op_indices[0]
    ):
        if any(index in occupied for index in region.op_indices):
            continue
        signature_index = len(signatures)
        signatures.append(signature)
        regions.append(region)
        recovered.append(region)
        occupied.update(region.op_indices)
        family = families[family_index]
        family.indices.append(signature_index)
        family.indices.sort(key=lambda index: signatures[index].region.op_indices[0])
        family.minimum_similarity = min(family.minimum_similarity, minimum)
    return recovered


@dataclass(frozen=True, slots=True)
class _StraightPathMetrics:
    direction: Point
    total_length: float


def _straight_path_metrics(operation: PathOperationIR) -> _StraightPathMetrics | None:
    if not operation.stroke or any(segment.kind == "curve" for segment in operation.segments):
        return None
    current: Point | None = None
    direction: Point | None = None
    total_length = 0.0
    line_count = 0
    move_count = 0
    first: Point | None = None
    last: Point | None = None
    collinear = True
    for segment in operation.segments:
        if segment.kind == "move":
            assert segment.end is not None
            current = Point(*segment.end)
            move_count += 1
            continue
        if segment.kind != "line" or current is None:
            continue
        assert segment.end is not None
        end = Point(*segment.end)
        dx, dy = end.x - current.x, end.y - current.y
        length = math.hypot(dx, dy)
        if first is None:
            first = current
        current = end
        if length <= _EPSILON:
            continue
        last = current
        unit = Point(dx / length, dy / length)
        if direction is None:
            direction = unit
        elif abs(direction.x * unit.x + direction.y * unit.y) < 0.965:
            collinear = False
        total_length += length
        line_count += 1
    if direction is None or line_count == 0 or line_count > 12:
        return None
    if collinear:
        return _StraightPathMetrics(direction, total_length)
    if move_count != 1 or first is None or last is None:
        return None
    chord_x, chord_y = last.x - first.x, last.y - first.y
    chord = math.hypot(chord_x, chord_y)
    if chord <= _EPSILON or chord / total_length < CARRIER_RUN_STRAIGHTNESS:
        return None
    return _StraightPathMetrics(Point(chord_x / chord, chord_y / chord), chord)


def _interval_gap(
    left_min: float, left_max: float, right_min: float, right_max: float,
) -> float:
    return max(0.0, left_min - right_max, right_min - left_max)


@dataclass(frozen=True, slots=True)
class _DashCandidate:
    op_index: int
    score: float


@dataclass(frozen=True, slots=True)
class _DashProjection:
    side: str
    op_index: int
    score: float
    gap: float
    u_min: float
    u_max: float


@dataclass(frozen=True, slots=True)
class _DashAttachment:
    left: tuple[_DashCandidate, ...] = ()
    right: tuple[_DashCandidate, ...] = ()


def _dash_attachment_for(
    context: _Context,
    signature: RegionSignature,
    text_op_indices: set[int],
    page_diagonal: float,
) -> _DashAttachment:
    region = signature.region
    region_paths = tuple(
        operation
        for index in region.op_indices
        if isinstance((operation := context.operation_index.operation(index)), PathOperationIR)
    )
    attachment_frame = _carrier_axis_frame_for(region, region_paths) or signature.frame
    channel = signature.channel
    region_width = (
        0.0 if signature.pattern_source == "pdf_text"
        else _median(path.line_width for path in region_paths)
    )
    gap_tolerance = max(
        attachment_frame.minor * 2.2,
        page_diagonal * 0.0015,
        region_width * 6.0,
    )
    first_position = context.paint_position(region.op_indices[0])
    last_position = context.paint_position(region.op_indices[-1])
    start = max(0, first_position - PAINT_ORDER_NEIGHBORHOOD)
    end = min(len(context.paint_orders), last_position + PAINT_ORDER_NEIGHBORHOOD + 1)
    projections: list[_DashProjection] = []
    for position in range(start, end):
        for dense_index in context.operation_index.indices_for_paint_order(
            context.paint_orders[position]
        ):
            if dense_index in text_op_indices:
                continue
            # Match the frozen Scene-neighbourhood contract: a sequential
            # Group boundary may fall between a Pattern and its immediately
            # adjacent carrier, so attachment evidence is intentionally not
            # restricted to the Pattern's Group here.
            operation = context.operation_index.operation(dense_index)
            if not isinstance(operation, PathOperationIR):
                continue
            if channel and _channel_for(operation) != channel:
                continue
            metrics = _straight_path_metrics(operation)
            if metrics is None:
                continue
            alignment = abs(
                metrics.direction.x * attachment_frame.ux
                + metrics.direction.y * attachment_frame.uy
            )
            if alignment < 0.93:
                continue
            maximum_length = max(attachment_frame.major, attachment_frame.minor) * (
                MAX_CARRIER_TO_PATTERN_MAJOR
            )
            if (
                metrics.total_length < attachment_frame.minor * 0.25
                or metrics.total_length > maximum_length
            ):
                continue
            if (
                region_width > _EPSILON and operation.line_width > _EPSILON
                and _ratio_similarity(region_width, operation.line_width) < 0.4
            ):
                continue
            corners = _bounds_corners(operation.bounds)
            values_u = [_projection(point, attachment_frame.ux, attachment_frame.uy) for point in corners]
            values_v = [_projection(point, attachment_frame.vx, attachment_frame.vy) for point in corners]
            u_min, u_max = min(values_u), max(values_u)
            v_gap = _interval_gap(
                min(values_v), max(values_v), attachment_frame.min_v, attachment_frame.max_v
            )
            if v_gap > attachment_frame.minor * 0.85:
                continue
            if u_max <= attachment_frame.min_u + attachment_frame.minor * 0.35:
                side = "left"
                gap = max(0.0, attachment_frame.min_u - u_max)
            elif u_min >= attachment_frame.max_u - attachment_frame.minor * 0.35:
                side = "right"
                gap = max(0.0, u_min - attachment_frame.max_u)
            else:
                continue
            if gap > max(gap_tolerance, attachment_frame.major * 5.0):
                continue
            order_distance = min(abs(position - first_position), abs(position - last_position))
            score = (
                max(0.0, 1.0 - gap / max(_EPSILON, gap_tolerance)) * 0.72
                + alignment * 0.18
                + max(0.0, 1.0 - order_distance / PAINT_ORDER_NEIGHBORHOOD) * 0.10
            )
            projections.append(_DashProjection(
                side, dense_index, score, gap, u_min, u_max
            ))

    def best_chain(side: str) -> tuple[_DashCandidate, ...]:
        available = sorted(
            (item for item in projections if item.side == side),
            key=lambda item: (item.gap, -item.score, item.op_index),
        )
        seed = next((item for item in available if item.gap <= gap_tolerance), None)
        if seed is None:
            return ()
        chain = [seed]
        remaining = [item for item in available if item is not seed]
        changed = True
        while changed:
            changed = False
            for candidate in tuple(remaining):
                connected = any(
                    _interval_gap(
                        candidate.u_min, candidate.u_max, member.u_min, member.u_max
                    ) <= gap_tolerance
                    and abs(
                        context.paint_position(candidate.op_index)
                        - context.paint_position(member.op_index)
                    ) <= 8
                    for member in chain
                )
                if connected:
                    chain.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        return tuple(
            _DashCandidate(item.op_index, item.score)
            for item in sorted(chain, key=lambda item: item.op_index)
        )

    return _DashAttachment(best_chain("left"), best_chain("right"))


def _attachment_indices(attachment: _DashAttachment | None) -> tuple[int, ...]:
    if attachment is None:
        return ()
    return tuple(item.op_index for item in (*attachment.left, *attachment.right))


def _is_inline_measurement_family(
    family: _Family,
    signatures: Sequence[RegionSignature],
) -> bool:
    return bool(family.indices) and all(
        signatures[index].pattern_source == "pdf_text"
        and signatures[index].literal_key is not None
        and _is_inline_feet_token(signatures[index].literal_key or "")
        for index in family.indices
    )


def _interior_measurement_carriers(
    context: _Context,
    family: _Family,
    signatures: Sequence[RegionSignature],
) -> set[int]:
    """Return carriers genuinely between two repeated measurement tokens.

    A normal text line may legitimately extend its carrier past the terminal
    label.  A numeric measurement token is different: an exterior tail is
    weak evidence and can run straight into an unrelated periodic type.  Keep
    only the authored path interval between adjacent instances.  The interval
    must contain nothing except open carrier paths.  Strict sandwich anchors
    already proved this literal and Group before contextual corner instances
    were admitted, so dense source order is stronger evidence here than a
    single-axis attachment at a bend.  This preserves the visible
    ``line, 8', line`` chain while preventing route-tail extension from
    consuming a neighbouring square or dash family.
    """

    by_group: dict[str, list[int]] = {}
    for signature_index in family.indices:
        signature = signatures[signature_index]
        by_group.setdefault(signature.region.group_id, []).append(signature_index)
    interior: set[int] = set()
    for group_indices in by_group.values():
        ordered = sorted(
            group_indices,
            key=lambda index: signatures[index].region.op_indices[0],
        )
        for left_index, right_index in zip(ordered, ordered[1:]):
            left_op = signatures[left_index].region.op_indices[-1]
            right_op = signatures[right_index].region.op_indices[0]
            if (
                right_op <= left_op + 1
                or right_op - left_op > SEQUENTIAL_CARRIER_MAX_OP_GAP
            ):
                continue
            between = tuple(range(left_op + 1, right_op))
            if any(
                not isinstance(context.operation_index.operation(index), PathOperationIR)
                for index in between
            ):
                continue
            geometries = tuple(
                geometry
                for index in between
                if (
                    isinstance(
                        operation := context.operation_index.operation(index),
                        PathOperationIR,
                    )
                    and not operation.fill
                    and (geometry := _carrier_geometry_for(operation, index)) is not None
                )
            )
            if len(geometries) != len(between):
                continue
            interior.update(geometry.op_index for geometry in geometries)
    return interior


@dataclass(frozen=True, slots=True)
class _CarrierGeometry:
    op_index: int
    endpoints: tuple[Point, Point]
    tangents: tuple[Point, Point]
    length: float


def _point_distance(left: Point, right: Point) -> float:
    return math.hypot(left.x - right.x, left.y - right.y)


def _point_to_bounds_distance(point: Point, bounds: BoundsIR) -> float:
    return math.hypot(
        max(bounds.min_x - point.x, 0.0, point.x - bounds.max_x),
        max(bounds.min_y - point.y, 0.0, point.y - bounds.max_y),
    )


def _unit_vector(left: Point, right: Point) -> Point | None:
    dx, dy = right.x - left.x, right.y - left.y
    length = math.hypot(dx, dy)
    return Point(dx / length, dy / length) if length > _EPSILON else None


def _carrier_geometry_for(
    operation: PathOperationIR,
    dense_index: int,
) -> _CarrierGeometry | None:
    if (
        not operation.stroke
        or any(segment.kind == "close" for segment in operation.segments)
        or sum(segment.kind == "move" for segment in operation.segments) != 1
    ):
        return None
    samples = _path_samples(operation)
    if len(samples) < 2:
        return None
    length = sum(
        _point_distance(left, right) for left, right in zip(samples, samples[1:])
    )
    if length <= 1e-4:
        return None
    start_tangent = _unit_vector(samples[0], samples[1])
    end_tangent = _unit_vector(samples[-2], samples[-1])
    if start_tangent is None or end_tangent is None:
        return None
    return _CarrierGeometry(
        dense_index,
        (samples[0], samples[-1]),
        (start_tangent, end_tangent),
        length,
    )


def _attachment_length(
    context: _Context,
    candidates: Sequence[_DashCandidate],
) -> float:
    total = 0.0
    for dense_index in {candidate.op_index for candidate in candidates}:
        operation = context.operation_index.operation(dense_index)
        if isinstance(operation, PathOperationIR):
            geometry = _carrier_geometry_for(operation, dense_index)
            total += geometry.length if geometry is not None else 0.0
    return total


def _has_strong_two_sided_carrier(
    context: _Context,
    signature: RegionSignature,
    attachment: _DashAttachment | None,
    page_diagonal: float,
) -> bool:
    if attachment is None or not attachment.left or not attachment.right:
        return False
    minimum_length = max(page_diagonal * 0.0004, signature.frame.minor * 0.35)
    return (
        _attachment_length(context, attachment.left) >= minimum_length
        and _attachment_length(context, attachment.right) >= minimum_length
    )


@dataclass(frozen=True, slots=True)
class _CarrierEdge:
    index: int
    gap: float
    tangent_similarity: float


@dataclass(frozen=True, slots=True)
class _CarrierRouteGraph:
    geometries: tuple[_CarrierGeometry, ...]
    index_by_op: Mapping[int, int]
    adjacency: tuple[tuple[_CarrierEdge, ...], ...]


def _nearest_endpoint_connection(
    left: _CarrierGeometry,
    right: _CarrierGeometry,
) -> tuple[float, float]:
    best: tuple[float, float] | None = None
    for left_end in range(2):
        for right_end in range(2):
            gap = _point_distance(left.endpoints[left_end], right.endpoints[right_end])
            tangent = abs(
                left.tangents[left_end].x * right.tangents[right_end].x
                + left.tangents[left_end].y * right.tangents[right_end].y
            )
            if best is None or gap < best[0] or (
                abs(gap - best[0]) <= 1e-9 and tangent > best[1]
            ):
                best = (gap, tangent)
    assert best is not None
    return best


def _candidate_carrier_pairs(
    geometries: Sequence[_CarrierGeometry],
    junction_tolerance: float,
) -> Iterable[tuple[int, int]]:
    """Yield every pair that can have endpoints within the tolerance.

    The former route-graph builder compared every carrier with every other
    carrier.  A large CAD group can contain tens of thousands of unrelated
    paths, making that quadratic scan dominate the complete page run.  Index
    the two endpoints in tolerance-sized cells and return a strict superset
    of the possible neighbours instead.  The caller still performs the
    original distance/tangent calculation and exact threshold check.

    Pair order deliberately remains the old lexicographic ``left, right``
    order.  Dijkstra tie-breaking and therefore emitted ownership stay bit
    for bit deterministic.  The extra cell on every side protects boundary
    cases from floating-point rounding; it can only add candidates, never
    omit a pair accepted by the exact check.
    """

    count = len(geometries)
    if count < 2:
        return
    finite_endpoints = all(
        math.isfinite(point.x) and math.isfinite(point.y)
        for geometry in geometries
        for point in geometry.endpoints
    )
    if junction_tolerance <= 0.0 or not math.isfinite(junction_tolerance) \
            or not finite_endpoints:
        for left in range(count):
            for right in range(left + 1, count):
                yield left, right
        return

    cell_size = junction_tolerance
    spatial_index_safe = all(
        all(math.isfinite(value) for value in (
            point.x / cell_size,
            point.y / cell_size,
            (point.x - junction_tolerance) / cell_size,
            (point.x + junction_tolerance) / cell_size,
            (point.y - junction_tolerance) / cell_size,
            (point.y + junction_tolerance) / cell_size,
        ))
        for geometry in geometries
        for point in geometry.endpoints
    )
    if not spatial_index_safe:
        for left in range(count):
            for right in range(left + 1, count):
                yield left, right
        return

    endpoint_grid: dict[tuple[int, int], list[int]] = {}
    for index, geometry in enumerate(geometries):
        for point in geometry.endpoints:
            cell = (
                math.floor(point.x / cell_size),
                math.floor(point.y / cell_size),
            )
            endpoint_grid.setdefault(cell, []).append(index)

    for left, geometry in enumerate(geometries):
        candidates: set[int] = set()
        for point in geometry.endpoints:
            min_x = math.floor((point.x - junction_tolerance) / cell_size) - 1
            max_x = math.floor((point.x + junction_tolerance) / cell_size) + 1
            min_y = math.floor((point.y - junction_tolerance) / cell_size) - 1
            max_y = math.floor((point.y + junction_tolerance) / cell_size) + 1
            for cell_x in range(min_x, max_x + 1):
                for cell_y in range(min_y, max_y + 1):
                    candidates.update(endpoint_grid.get((cell_x, cell_y), ()))
        for right in sorted(index for index in candidates if index > left):
            yield left, right


def _carrier_route_graph_for(
    geometries: Sequence[_CarrierGeometry],
    junction_tolerance: float,
) -> _CarrierRouteGraph:
    adjacency: list[list[_CarrierEdge]] = [[] for _ in geometries]
    for left, right in _candidate_carrier_pairs(geometries, junction_tolerance):
        gap, tangent = _nearest_endpoint_connection(geometries[left], geometries[right])
        if gap > junction_tolerance:
            continue
        adjacency[left].append(_CarrierEdge(right, gap, tangent))
        adjacency[right].append(_CarrierEdge(left, gap, tangent))
    return _CarrierRouteGraph(
        tuple(geometries),
        {geometry.op_index: index for index, geometry in enumerate(geometries)},
        tuple(tuple(edges) for edges in adjacency),
    )


@dataclass(frozen=True, slots=True)
class _CarrierShortestPaths:
    source_indices: frozenset[int]
    previous: tuple[int, ...]
    settled_order: tuple[int, ...]


def _carrier_shortest_paths(
    graph: _CarrierRouteGraph,
    source_ops: set[int],
    junction_tolerance: float,
    stop_indices: set[int] | None = None,
) -> _CarrierShortestPaths | None:
    """Run the deterministic carrier Dijkstra to completion for one source.

    A family can ask thousands of target questions about the same source on
    the same graph.  Completing the search once preserves the exact heap key,
    relaxation rule and predecessor tree used by the former per-target run.
    ``settled_order`` lets each target query select the same first target that
    the early-exit implementation would have popped.
    """

    source_indices = [
        graph.index_by_op[index] for index in source_ops if index in graph.index_by_op
    ]
    if not source_indices:
        return None
    costs = [math.inf] * len(graph.geometries)
    previous = [-1] * len(graph.geometries)
    settled_order = [-1] * len(graph.geometries)
    queue: list[tuple[float, int, int]] = []
    for index in source_indices:
        costs[index] = graph.geometries[index].length
        heapq.heappush(queue, (
            costs[index], graph.geometries[index].op_index, index
        ))
    order = 0
    while queue:
        cost, _op_index, current = heapq.heappop(queue)
        if cost != costs[current]:
            continue
        settled_order[current] = order
        order += 1
        if stop_indices is not None and current in stop_indices:
            break
        for edge in graph.adjacency[current]:
            turn_penalty = (1.0 - edge.tangent_similarity) * junction_tolerance * 2.5
            next_cost = (
                cost + graph.geometries[edge.index].length
                + edge.gap * 8.0 + turn_penalty
            )
            if next_cost >= costs[edge.index]:
                continue
            costs[edge.index] = next_cost
            previous[edge.index] = current
            heapq.heappush(queue, (
                next_cost, graph.geometries[edge.index].op_index, edge.index
            ))
    return _CarrierShortestPaths(
        frozenset(source_indices), tuple(previous), tuple(settled_order)
    )


def _carrier_bridge_from_paths(
    graph: _CarrierRouteGraph,
    paths: _CarrierShortestPaths | None,
    source_ops: set[int],
    target_ops: set[int],
) -> tuple[_CarrierGeometry, ...] | None:
    if paths is None:
        return None
    targets = {
        graph.index_by_op[index] for index in target_ops if index in graph.index_by_op
    }
    reachable = [
        index for index in targets if paths.settled_order[index] >= 0
    ]
    if not reachable:
        return None
    reached = min(reachable, key=paths.settled_order.__getitem__)
    route: list[_CarrierGeometry] = []
    current = reached
    while current >= 0:
        route.append(graph.geometries[current])
        if current in paths.source_indices:
            break
        current = paths.previous[current]
    route.reverse()
    if not route or route[0].op_index not in source_ops:
        return None
    return tuple(route)


def _shortest_carrier_bridge(
    graph: _CarrierRouteGraph,
    source_ops: set[int],
    target_ops: set[int],
    junction_tolerance: float,
) -> tuple[_CarrierGeometry, ...] | None:
    """Run one historical early-exit query, including on unusual weights."""

    targets = {
        graph.index_by_op[index] for index in target_ops if index in graph.index_by_op
    }
    if not targets:
        return None
    return _carrier_bridge_from_paths(
        graph,
        _carrier_shortest_paths(
            graph, source_ops, junction_tolerance, stop_indices=targets
        ),
        source_ops,
        target_ops,
    )


def _carrier_steps_can_be_reused(
    graph: _CarrierRouteGraph,
    junction_tolerance: float,
) -> bool:
    """Whether settled predecessors are final for every directed graph step."""

    for edges in graph.adjacency:
        for edge in edges:
            step = (
                graph.geometries[edge.index].length
                + edge.gap * 8.0
                + (1.0 - edge.tangent_similarity) * junction_tolerance * 2.5
            )
            if not math.isfinite(step) or step < 0.0:
                return False
    return True


@dataclass(frozen=True, slots=True)
class _DashOwner:
    family_index: int
    group_id: str
    score: float


def _extend_sequential_carrier_tails(
    context: _Context,
    graph: _CarrierRouteGraph,
    family_index: int,
    group_id: str,
    best_owner: dict[int, _DashOwner],
) -> set[int]:
    extended: set[int] = set()

    def sequential_neighbors(index: int) -> tuple[_CarrierEdge, ...]:
        position = context.paint_position(graph.geometries[index].op_index)
        return tuple(
            edge for edge in graph.adjacency[index]
            if abs(
                position - context.paint_position(graph.geometries[edge.index].op_index)
            ) <= SEQUENTIAL_CARRIER_MAX_OP_GAP
        )

    queue = [
        index for index, geometry in enumerate(graph.geometries)
        if (owner := best_owner.get(geometry.op_index)) is not None
        and owner.family_index == family_index and owner.group_id == group_id
    ]
    visited: set[int] = set()
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        current_neighbors = sequential_neighbors(current)
        if len(current_neighbors) > 2:
            continue
        unowned = [
            edge for edge in current_neighbors
            if graph.geometries[edge.index].op_index not in best_owner
        ]
        if len(unowned) != 1:
            continue
        next_index = unowned[0].index
        next_neighbors = sequential_neighbors(next_index)
        if len(next_neighbors) > 2:
            continue
        if any(
            (owner := best_owner.get(graph.geometries[edge.index].op_index)) is not None
            and (owner.family_index != family_index or owner.group_id != group_id)
            for edge in next_neighbors
        ):
            continue
        op_index = graph.geometries[next_index].op_index
        best_owner[op_index] = _DashOwner(family_index, group_id, 0.74)
        extended.add(op_index)
        queue.append(next_index)
    return extended


def _frame_center(signature: RegionSignature) -> Point:
    u = (signature.frame.min_u + signature.frame.max_u) / 2.0
    v = (signature.frame.min_v + signature.frame.max_v) / 2.0
    return Point(
        signature.frame.ux * u + signature.frame.vx * v,
        signature.frame.uy * u + signature.frame.vy * v,
    )


def _frame_bounds(signature: RegionSignature) -> BoundsIR:
    corners = [
        Point(
            signature.frame.ux * u + signature.frame.vx * v,
            signature.frame.uy * u + signature.frame.vy * v,
        )
        for u in (signature.frame.min_u, signature.frame.max_u)
        for v in (signature.frame.min_v, signature.frame.max_v)
    ]
    return BoundsIR(
        min(point.x for point in corners), min(point.y for point in corners),
        max(point.x for point in corners), max(point.y for point in corners),
    )


def _clear_bounds_separation(left: RegionSignature, right: RegionSignature) -> float:
    left_bounds, right_bounds = _frame_bounds(left), _frame_bounds(right)
    return math.hypot(
        _interval_gap(
            left_bounds.min_x, left_bounds.max_x, right_bounds.min_x, right_bounds.max_x
        ),
        _interval_gap(
            left_bounds.min_y, left_bounds.max_y, right_bounds.min_y, right_bounds.max_y
        ),
    )


def _bridge_same_family_carrier_routes(
    context: _Context,
    signatures: Sequence[RegionSignature],
    families: Sequence[_Family],
    candidate_families: set[int],
    attachments: Mapping[int, _DashAttachment],
    detected_text_ops: set[int] | frozenset[int],
    page_diagonal: float,
    best_owner: dict[int, _DashOwner],
) -> tuple[set[int], set[int]]:
    bridged: set[int] = set()
    connected: set[int] = set()
    for family_index, family in enumerate(families):
        if family_index not in candidate_families:
            continue
        by_group: dict[str, list[int]] = {}
        for signature_index in family.indices:
            group_id = signatures[signature_index].region.group_id
            by_group.setdefault(group_id, []).append(signature_index)
        for group_id, group_signature_indices in by_group.items():
            group_signatures = [signatures[index] for index in group_signature_indices]
            region_paths = [
                operation
                for signature in group_signatures
                for dense_index in signature.region.op_indices
                if isinstance(
                    (operation := context.operation_index.operation(dense_index)),
                    PathOperationIR,
                )
            ]
            region_width = _median(path.line_width for path in region_paths)
            channel = group_signatures[0].channel if group_signatures else ""
            geometries: list[_CarrierGeometry] = []
            for dense_index in context.operation_index.group_indices(group_id):
                if dense_index in detected_text_ops:
                    continue
                operation = context.operation_index.operation(dense_index)
                if not isinstance(operation, PathOperationIR):
                    continue
                if channel and _channel_for(operation) != channel:
                    continue
                if (
                    region_width > _EPSILON and operation.line_width > _EPSILON
                    and _ratio_similarity(region_width, operation.line_width) < 0.35
                ):
                    continue
                geometry = _carrier_geometry_for(operation, dense_index)
                if geometry is not None:
                    geometries.append(geometry)
            if not geometries:
                continue
            geometry_widths = [
                context.operation_index.operation(item.op_index).line_width
                for item in geometries
                if isinstance(
                    context.operation_index.operation(item.op_index), PathOperationIR
                )
            ]
            line_width = (
                max(region_width, _median(geometry_widths))
                if group_signatures[0].pattern_source == "pdf_text"
                else max((region_width, *geometry_widths), default=region_width)
            )
            minimum_minor = min(signature.frame.minor for signature in group_signatures)
            junction_tolerance = max(
                0.2, page_diagonal * 0.00045,
                minimum_minor * 0.20, line_width * 3.0,
            )
            graph = _carrier_route_graph_for(geometries, junction_tolerance)
            can_reuse_shortest_paths = _carrier_steps_can_be_reused(
                graph, junction_tolerance
            )
            anchor_ops: dict[int, set[int]] = {}
            for signature_index in group_signature_indices:
                signature = signatures[signature_index]
                ops = set(_attachment_indices(attachments.get(signature_index)))
                base_tolerance = max(
                    0.5,
                    page_diagonal * 0.001,
                    signature.frame.minor * 2.2,
                    min(signature.frame.major * 0.4, signature.frame.minor * 3.2),
                    region_width * 5.0,
                )
                first_position = context.paint_position(signature.region.op_indices[0])
                last_position = context.paint_position(signature.region.op_indices[-1])
                for geometry in geometries:
                    position = context.paint_position(geometry.op_index)
                    order_distance = min(
                        abs(position - first_position), abs(position - last_position)
                    )
                    tolerance = base_tolerance * (
                        1.8 if order_distance <= SEQUENTIAL_CARRIER_MAX_OP_GAP else 1.0
                    )
                    if any(
                        _point_to_bounds_distance(point, signature.region.bounds) <= tolerance
                        for point in geometry.endpoints
                    ):
                        ops.add(geometry.op_index)
                anchor_ops[signature_index] = ops
            for left_order, left_index in enumerate(group_signature_indices):
                source_ops = anchor_ops.get(left_index, set())
                shortest_paths: _CarrierShortestPaths | None = None
                shortest_paths_ready = False
                for right_index in group_signature_indices[left_order + 1:]:
                    left_signature = signatures[left_index]
                    right_signature = signatures[right_index]
                    minimum_span = max(
                        page_diagonal * 0.0004,
                        min(left_signature.frame.minor, right_signature.frame.minor) * 0.5,
                        line_width * 3.0,
                    )
                    if _clear_bounds_separation(left_signature, right_signature) < minimum_span:
                        continue
                    if (
                        _has_strong_two_sided_carrier(
                            context, left_signature, attachments.get(left_index), page_diagonal
                        )
                        and _has_strong_two_sided_carrier(
                            context, right_signature, attachments.get(right_index), page_diagonal
                        )
                    ):
                        connected.add(family_index)
                    if can_reuse_shortest_paths:
                        if not shortest_paths_ready:
                            shortest_paths = _carrier_shortest_paths(
                                graph, source_ops, junction_tolerance
                            )
                            shortest_paths_ready = True
                        route = _carrier_bridge_from_paths(
                            graph,
                            shortest_paths,
                            source_ops,
                            anchor_ops.get(right_index, set()),
                        )
                    else:
                        route = _shortest_carrier_bridge(
                            graph,
                            source_ops,
                            anchor_ops.get(right_index, set()),
                            junction_tolerance,
                        )
                    if route is None:
                        continue
                    route_length = sum(geometry.length for geometry in route)
                    if route_length < minimum_span:
                        continue
                    connected.add(family_index)
                    new_route_ops = [
                        geometry.op_index for geometry in route
                        if geometry.op_index not in best_owner
                    ]
                    anchor_distance = _point_distance(
                        _frame_center(left_signature), _frame_center(right_signature)
                    )
                    route_score = 0.78 + 0.12 * min(
                        1.0, anchor_distance / max(_EPSILON, route_length)
                    )
                    for geometry in route:
                        current = best_owner.get(geometry.op_index)
                        if (
                            current is None or route_score > current.score
                            or (
                                route_score == current.score
                                and family_index < current.family_index
                            )
                        ):
                            best_owner[geometry.op_index] = _DashOwner(
                                family_index, group_id, route_score
                            )
                    bridged.update(new_route_ops)
            bridged.update(_extend_sequential_carrier_tails(
                context, graph, family_index, group_id, best_owner
            ))
    return bridged, connected


def _has_sequential_carrier_between(
    context: _Context,
    left: RegionSignature,
    right: RegionSignature,
    detected_text_ops: set[int],
) -> bool:
    if (
        left.pattern_source != "vector_strokes"
        or right.pattern_source != "vector_strokes"
        or left.region.group_id != right.region.group_id
        or not _has_exact_ordered_topology(left, right)
    ):
        return False
    left_start, left_end = left.region.op_indices[0], left.region.op_indices[-1]
    right_start, right_end = right.region.op_indices[0], right.region.op_indices[-1]
    gap_start = min(left_end, right_end) + 1
    gap_end = max(left_start, right_start)
    if gap_start >= gap_end:
        return False
    paint_gap = abs(
        context.paint_position(gap_end) - context.paint_position(gap_start)
    ) + 1
    if paint_gap > SEQUENTIAL_CARRIER_MAX_OP_GAP:
        return False
    for dense_index in range(gap_start, gap_end):
        if dense_index in detected_text_ops:
            return False
        operation = context.operation_index.operation(dense_index)
        if (
            not isinstance(operation, PathOperationIR)
            or _carrier_geometry_for(operation, dense_index) is None
        ):
            return False
    return True


def _admit_sequential_contextual_satellites(
    context: _Context,
    signatures: Sequence[RegionSignature],
    families: list[_Family],
    confirmed_family_indices: set[int],
    detected_text_ops: set[int],
) -> int:
    assigned = {index for family in families for index in family.indices}
    admitted = 0
    for signature_index, signature in enumerate(signatures):
        if signature_index in assigned or signature.pattern_source != "vector_strokes":
            continue
        start, end = signature.region.op_indices[0], signature.region.op_indices[-1]
        candidates: list[tuple[float, float, int]] = []
        for family_index in confirmed_family_indices:
            if family_index >= len(families):
                continue
            family = families[family_index]
            if len(family.indices) < 3:
                continue
            same_group = sorted(
                (
                    index for index in family.indices
                    if signatures[index].region.group_id == signature.region.group_id
                ),
                key=lambda index: signatures[index].region.op_indices[0],
            )
            left_index = next((
                index for index in reversed(same_group)
                if signatures[index].region.op_indices[-1] < start
            ), None)
            right_index = next((
                index for index in same_group
                if signatures[index].region.op_indices[0] > end
            ), None)
            if left_index is None or right_index is None:
                continue
            left, right = signatures[left_index], signatures[right_index]
            if (
                not _has_sequential_carrier_between(
                    context, left, signature, detected_text_ops
                )
                or not _has_sequential_carrier_between(
                    context, signature, right, detected_text_ops
                )
            ):
                continue
            left_similarity = vector_text_region_similarity(signature, left)
            right_similarity = vector_text_region_similarity(signature, right)
            adjacent_minimum = min(left_similarity, right_similarity)
            if adjacent_minimum < SEQUENTIAL_CARRIER_MATCH_THRESHOLD:
                continue
            strongest = max(
                vector_text_region_similarity(signature, signatures[index])
                for index in family.indices
            )
            if strongest < MATCH_THRESHOLD:
                continue
            candidates.append((strongest, adjacent_minimum, family_index))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if not candidates:
            continue
        winner = candidates[0]
        if len(candidates) > 1 and winner[0] - candidates[1][0] < 0.02:
            continue
        family = families[winner[2]]
        family.indices.append(signature_index)
        family.indices.sort(key=lambda index: signatures[index].region.op_indices[0])
        family.minimum_similarity = min(family.minimum_similarity, winner[1])
        assigned.add(signature_index)
        admitted += 1
    return admitted


@dataclass(frozen=True, slots=True)
class _CarrierMergeWorkerState:
    context: _Context
    signatures: tuple[RegionSignature, ...]
    seed_families: tuple[_Family, ...]
    attachments: Mapping[int, _DashAttachment]
    detected_text_ops: frozenset[int]
    page_diagonal: float


@dataclass(frozen=True, slots=True)
class _CarrierMergeParallelWork:
    candidate_count: int
    pairwise_work_units: int


_carrier_merge_worker_state: _CarrierMergeWorkerState | None = None


def _initialize_carrier_merge_worker(
    page: PageIR,
    grouping: GroupingIR,
    signatures: tuple[RegionSignature, ...],
    seed_families: tuple[_Family, ...],
    attachments: dict[int, _DashAttachment],
    detected_text_ops: frozenset[int],
    page_diagonal: float,
) -> None:
    global _carrier_merge_worker_state
    _carrier_merge_worker_state = _CarrierMergeWorkerState(
        _Context.build(page, grouping),
        signatures,
        seed_families,
        attachments,
        detected_text_ops,
        page_diagonal,
    )


def _carrier_merge_pair_decision_for(
    state: _CarrierMergeWorkerState,
    task: tuple[int, int],
) -> tuple[int, int, float | None]:
    left_family_index, right_family_index = task
    candidate_pairs = _carrier_merge_candidates_for(state, task)
    for similarity, left_index, right_index in candidate_pairs:
        if _carrier_merge_candidate_is_bridged_for(
            state,
            (similarity, left_index, right_index),
        ):
            return left_family_index, right_family_index, similarity
    return left_family_index, right_family_index, None


def _carrier_merge_candidates_for(
    state: _CarrierMergeWorkerState,
    task: tuple[int, int],
) -> tuple[tuple[float, int, int], ...]:
    left_family_index, right_family_index = task
    candidate_pairs: list[tuple[float, int, int]] = []
    for left_index in state.seed_families[left_family_index].indices:
        for right_index in state.seed_families[right_family_index].indices:
            left, right = state.signatures[left_index], state.signatures[right_index]
            if (
                left.pattern_source != right.pattern_source
                or left.region.group_id != right.region.group_id
            ):
                continue
            similarity = vector_text_region_similarity(left, right)
            sequential = _has_sequential_carrier_between(
                state.context, left, right, state.detected_text_ops
            )
            if similarity >= MATCH_THRESHOLD or (
                sequential and similarity >= SEQUENTIAL_CARRIER_MATCH_THRESHOLD
            ):
                candidate_pairs.append((similarity, left_index, right_index))
    candidate_pairs.sort(key=lambda item: (
        -item[0],
        state.signatures[item[1]].region.op_indices[0],
        state.signatures[item[2]].region.op_indices[0],
    ))
    return tuple(candidate_pairs)


def _carrier_merge_parallel_work_for(
    context: _Context,
    signatures: Sequence[RegionSignature],
    candidate_lists: Sequence[Sequence[tuple[float, int, int]]],
) -> _CarrierMergeParallelWork:
    group_operation_counts: dict[str, int] = {}
    candidate_count = 0
    pairwise_work_units = 0
    for candidates in candidate_lists:
        candidate_count += len(candidates)
        for _similarity, left_index, _right_index in candidates:
            group_id = signatures[left_index].region.group_id
            operation_count = group_operation_counts.get(group_id)
            if operation_count is None:
                operation_count = len(context.operation_index.group_indices(group_id))
                group_operation_counts[group_id] = operation_count
            pairwise_work_units += operation_count * (operation_count - 1) // 2
    return _CarrierMergeParallelWork(candidate_count, pairwise_work_units)


def _should_parallelize_carrier_merge(
    worker_count: int,
    work: _CarrierMergeParallelWork,
) -> bool:
    return (
        worker_count > 1
        and work.candidate_count >= MIN_PARALLEL_CARRIER_CANDIDATES
        and work.pairwise_work_units >= MIN_PARALLEL_CARRIER_PAIRWISE_WORK
    )


def _carrier_merge_candidate_is_bridged_for(
    state: _CarrierMergeWorkerState,
    task: tuple[float, int, int],
) -> bool:
    similarity, left_index, right_index = task
    bridged, _connected = _bridge_same_family_carrier_routes(
        state.context,
        state.signatures,
        [_Family([left_index, right_index], similarity)],
        {0},
        state.attachments,
        state.detected_text_ops,
        state.page_diagonal,
        {},
    )
    return bool(bridged)


def _carrier_merge_candidate_worker(task: tuple[float, int, int]) -> bool:
    state = _carrier_merge_worker_state
    if state is None:  # pragma: no cover - executor initializer owns this.
        raise RuntimeError("carrier merge worker is not initialized")
    return _carrier_merge_candidate_is_bridged_for(state, task)


def _merge_carrier_connected_pattern_families(
    context: _Context,
    signatures: Sequence[RegionSignature],
    seed_families: Sequence[_Family],
    confirmed_seed_families: set[int],
    attachments: Mapping[int, _DashAttachment],
    detected_text_ops: set[int],
    page_diagonal: float,
    worker_count: int,
) -> tuple[list[_Family], set[int]]:
    parent = list(range(len(seed_families)))
    merge_scores: list[tuple[int, int, float]] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int, similarity: float) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        keep, drop = min(left_root, right_root), max(left_root, right_root)
        parent[drop] = keep
        merge_scores.append((left, right, similarity))

    confirmed = sorted(confirmed_seed_families)
    pair_tasks: list[tuple[int, int]] = []
    for left_order, left_family_index in enumerate(confirmed):
        for right_family_index in confirmed[left_order + 1:]:
            if (
                left_family_index >= len(seed_families)
                or right_family_index >= len(seed_families)
            ):
                continue
            pair_tasks.append((left_family_index, right_family_index))

    state = _CarrierMergeWorkerState(
        context,
        tuple(signatures),
        tuple(seed_families),
        attachments,
        frozenset(detected_text_ops),
        page_diagonal,
    )
    if worker_count > 1:
        candidate_lists = [
            _carrier_merge_candidates_for(state, task) for task in pair_tasks
        ]
        parallel_work = _carrier_merge_parallel_work_for(
            context,
            signatures,
            candidate_lists,
        )
        process_count = (
            min(
                worker_count,
                MAX_CARRIER_MERGE_WORKERS,
                parallel_work.candidate_count,
            )
            if _should_parallelize_carrier_merge(worker_count, parallel_work)
            else 0
        )
    else:
        candidate_lists = []
        process_count = 0
    if process_count > 1:
        with ProcessPoolExecutor(
            max_workers=process_count,
            initializer=_initialize_carrier_merge_worker,
            initargs=(
                context.page,
                context.grouping,
                state.signatures,
                state.seed_families,
                dict(attachments),
                state.detected_text_ops,
                page_diagonal,
            ),
        ) as executor:
            # Bound speculation to one worker-sized batch at a time.  This
            # distributes a large family pair without evaluating every later
            # candidate after its first successful bridge.  Results are still
            # folded in the exact single-process pair/candidate order.
            for (left_family_index, right_family_index), candidates in zip(
                pair_tasks,
                candidate_lists,
            ):
                winning_similarity: float | None = None
                for start in range(0, len(candidates), process_count):
                    batch = candidates[start:start + process_count]
                    decisions = tuple(executor.map(
                        _carrier_merge_candidate_worker,
                        batch,
                        chunksize=1,
                    ))
                    winning_similarity = next((
                        similarity
                        for (similarity, _left_index, _right_index), bridged
                        in zip(batch, decisions)
                        if bridged
                    ), None)
                    if winning_similarity is not None:
                        break
                if winning_similarity is not None:
                    union(
                        left_family_index,
                        right_family_index,
                        winning_similarity,
                    )
    elif worker_count > 1:
        for (left_family_index, right_family_index), candidates in zip(
            pair_tasks,
            candidate_lists,
        ):
            for similarity, left_index, right_index in candidates:
                if _carrier_merge_candidate_is_bridged_for(
                    state,
                    (similarity, left_index, right_index),
                ):
                    union(left_family_index, right_family_index, similarity)
                    break
    else:
        for task in pair_tasks:
            left_family_index, right_family_index, similarity = (
                _carrier_merge_pair_decision_for(state, task)
            )
            if similarity is not None:
                union(left_family_index, right_family_index, similarity)

    components: dict[int, list[int]] = {}
    for family_index in range(len(seed_families)):
        components.setdefault(find(family_index), []).append(family_index)
    merged_with_sources: list[tuple[list[int], _Family]] = []
    for family_indices in components.values():
        family_set = set(family_indices)
        indices = sorted({
            index
            for family_index in family_indices
            for index in seed_families[family_index].indices
        }, key=lambda index: signatures[index].region.op_indices[0])
        supporting = [
            similarity for left, right, similarity in merge_scores
            if left in family_set and right in family_set
        ]
        minimum = min((
            *(seed_families[index].minimum_similarity for index in family_indices),
            *supporting,
        ))
        merged_with_sources.append((family_indices, _Family(indices, minimum)))
    merged_with_sources.sort(
        key=lambda item: signatures[item[1].indices[0]].region.op_indices[0]
    )
    confirmed_merged = {
        index for index, (source_indices, _family) in enumerate(merged_with_sources)
        if any(source in confirmed_seed_families for source in source_indices)
    }
    return [family for _sources, family in merged_with_sources], confirmed_merged


def _has_repeated_overlapping_pattern_instances(
    family: _Family,
    signatures: Sequence[RegionSignature],
) -> bool:
    by_group: dict[str, list[int]] = {}
    for signature_index in family.indices:
        signature = signatures[signature_index]
        by_group.setdefault(signature.region.group_id, []).append(signature_index)
    for group_indices in by_group.values():
        if len(group_indices) < 3:
            continue
        overlapping: set[int] = set()
        overlap_pairs = 0
        for left_order, left_index in enumerate(group_indices):
            for right_index in group_indices[left_order + 1:]:
                left = signatures[left_index].region.bounds
                right = signatures[right_index].region.bounds
                width = max(0.0, min(left.max_x, right.max_x) - max(left.min_x, right.min_x))
                height = max(0.0, min(left.max_y, right.max_y) - max(left.min_y, right.min_y))
                left_area = max(0.0, left.width) * max(0.0, left.height)
                right_area = max(0.0, right.width) * max(0.0, right.height)
                ratio = width * height / max(_EPSILON, min(left_area, right_area))
                if ratio < 0.12:
                    continue
                overlap_pairs += 1
                overlapping.update((left_index, right_index))
        if (
            overlap_pairs >= 2
            and len(overlapping) >= max(3, math.ceil(len(group_indices) * 0.5))
        ):
            return True
    return False


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


def _atom_count_for(context: _Context, indices: Iterable[int]) -> int:
    total = 0
    for dense_index in set(indices):
        operation = context.operation_index.operation(dense_index)
        if isinstance(operation, PathOperationIR):
            total += _path_atom_count(operation)
    return total


def _empty_audit() -> VectorTextFamilyAudit:
    return VectorTextFamilyAudit(
        (), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        (), (), (), (),
    )


def recognize_repeated_text_pattern_families(
    page: PageIR,
    grouping: GroupingIR,
    supplied_regions: Sequence[VectorTextRegion],
    *,
    worker_count: int = 1,
) -> TextFamilyRecognition:
    """Recognize repeated decoded/vector text only when carriers prove a line."""

    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ValueError("worker_count must be a positive integer")
    context = _Context.build(page, grouping)
    for region in supplied_regions:
        for dense_index in region.op_indices:
            context.operation_index.operation(dense_index)
            if context.operation_index.group_id(dense_index) != region.group_id:
                raise ValueError(
                    f"region {region.region_id!r} crosses Group ownership"
                )
    page_diagonal = math.hypot(page.page_bounds.width, page.page_bounds.height)
    if page_diagonal <= 0.0:
        groups = tuple(
            RecognizedGroup(group.group_id, 0, (), NonLineType(0, ()))
            for group in grouping.groups
        )
        result = LineTypeRecognitionResult(
            groups,
            (),
            RecognitionSummary(len(groups), len(groups), 0, 0, 0, 0, 0),
        )
        return TextFamilyRecognition(result, _empty_audit())

    regions = [_PatternRegion.from_vector(region) for region in supplied_regions]
    vector_signatures = [
        signature
        for region in regions
        if (signature := _signature_for(context, region, page_diagonal)) is not None
    ]
    native_signatures = [
        signature
        for dense_index, operation in context.operation_index.text_items()
        if (
            signature := _native_text_signature_for(
                context, operation, dense_index, page_diagonal
            )
        ) is not None
    ]
    strict_inline_by_key: dict[
        tuple[str, str, tuple[object, ...]], list[int]
    ] = {}
    for signature in native_signatures:
        literal_key = signature.literal_key or ""
        style_key = signature.inline_carrier_style_key
        if _is_inline_feet_token(literal_key) and style_key is not None:
            strict_inline_by_key.setdefault(
                (signature.region.group_id, literal_key, style_key), []
            ).append(signature.region.op_indices[0])
    qualified_inline_spans: dict[
        tuple[str, str, tuple[object, ...]], tuple[int, int]
    ] = {}
    for key, indices in strict_inline_by_key.items():
        distinct_paint_orders = {
            context.operation_index.operation(index).paint_order
            for index in indices
        }
        if len(distinct_paint_orders) >= 3:
            qualified_inline_spans[key] = (min(indices), max(indices))
    native_signatures = [
        signature
        for signature in native_signatures
        if not _is_inline_feet_token(signature.literal_key or "")
        or (
            signature.region.group_id,
            signature.literal_key or "",
            signature.inline_carrier_style_key,
        )
        in qualified_inline_spans
    ]
    strict_native_ops = {
        signature.region.op_indices[0] for signature in native_signatures
    }
    if qualified_inline_spans:
        for dense_index, operation in context.operation_index.text_items():
            if dense_index in strict_native_ops:
                continue
            literal_key = _normalized_pdf_text(operation.literal_text)
            group_id = context.operation_index.group_id(dense_index)
            if not _is_inline_feet_token(literal_key):
                continue
            signature = _native_text_signature_for(
                context,
                operation,
                dense_index,
                page_diagonal,
                allow_contextual_inline=True,
            )
            if signature is None or signature.inline_carrier_style_key is None:
                continue
            span = qualified_inline_spans.get((
                group_id,
                literal_key,
                signature.inline_carrier_style_key,
            ))
            if span is not None and span[0] < dense_index < span[1]:
                native_signatures.append(signature)
        native_signatures.sort(key=lambda item: item.region.op_indices[0])
    regions.extend(signature.region for signature in native_signatures)
    signatures: list[RegionSignature] = [*vector_signatures, *native_signatures]
    seed_families = _complete_text_pattern_families(context, signatures)
    original_matched = {
        index for family in seed_families for index in family.indices
    }
    original_detected_ops = {
        dense_index for region in regions for dense_index in region.op_indices
    }
    original_attachments = {
        index: _dash_attachment_for(
            context, signature, original_detected_ops, page_diagonal
        )
        for index, signature in enumerate(signatures)
        if index in original_matched
    }
    carrier_candidates: set[int] = set()
    cross_group_carriers: set[int] = set()
    for family_index, family in enumerate(seed_families):
        if _has_repeated_overlapping_pattern_instances(family, signatures):
            continue
        two_sided = [
            index for index in family.indices
            if _has_strong_two_sided_carrier(
                context, signatures[index], original_attachments.get(index), page_diagonal
            )
        ]
        if len(two_sided) < 2:
            continue
        carrier_candidates.add(family_index)
        if len({signatures[index].region.group_id for index in two_sided}) >= 2:
            cross_group_carriers.add(family_index)
    _initial_bridged, initial_connected = _bridge_same_family_carrier_routes(
        context,
        signatures,
        seed_families,
        carrier_candidates,
        original_attachments,
        original_detected_ops,
        page_diagonal,
        {},
    )
    seed_connected = {*cross_group_carriers, *initial_connected}
    _admit_sequential_contextual_satellites(
        context,
        signatures,
        seed_families,
        seed_connected,
        original_detected_ops,
    )
    _recover_missing_family_signatures(
        context,
        regions,
        signatures,
        seed_families,
        page_diagonal,
        seed_connected,
    )
    matched_signature_indices = {
        index for family in seed_families for index in family.indices
    }
    matched_signatures = [signatures[index] for index in sorted(matched_signature_indices)]
    detected_text_ops = {
        dense_index for region in regions for dense_index in region.op_indices
    }
    matched_text_ops = {
        dense_index
        for signature in matched_signatures
        for dense_index in signature.region.op_indices
    }
    attachments = {
        index: _dash_attachment_for(
            context, signature, detected_text_ops, page_diagonal
        )
        for index, signature in enumerate(signatures)
        if index in matched_signature_indices
    }
    families, connected_families = _merge_carrier_connected_pattern_families(
        context,
        signatures,
        seed_families,
        seed_connected,
        attachments,
        detected_text_ops,
        page_diagonal,
        worker_count,
    )
    best_owner: dict[int, _DashOwner] = {}
    for family_index, family in enumerate(families):
        if family_index not in connected_families:
            continue
        for signature_index in family.indices:
            attachment = attachments.get(signature_index)
            if attachment is None:
                continue
            for candidate in (*attachment.left, *attachment.right):
                current = best_owner.get(candidate.op_index)
                if (
                    current is None or candidate.score > current.score
                    or (
                        candidate.score == current.score
                        and family_index < current.family_index
                    )
                ):
                    best_owner[candidate.op_index] = _DashOwner(
                        family_index,
                        signatures[signature_index].region.group_id,
                        candidate.score,
                    )
    bridged_route_ops, _final_connected = _bridge_same_family_carrier_routes(
        context,
        signatures,
        families,
        connected_families,
        attachments,
        detected_text_ops,
        page_diagonal,
        best_owner,
    )
    measurement_interior_ops = {
        family_index: _interior_measurement_carriers(
            context,
            family,
            signatures,
        )
        for family_index, family in enumerate(families)
        if family_index in connected_families
        and _is_inline_measurement_family(family, signatures)
    }
    if measurement_interior_ops:
        best_owner = {
            op_index: owner
            for op_index, owner in best_owner.items()
            if owner.family_index not in measurement_interior_ops
            or op_index in measurement_interior_ops[owner.family_index]
        }
        bridged_route_ops.intersection_update(best_owner)
    families_with_owned_carrier = {
        owner.family_index for owner in best_owner.values()
    }
    connected_families.intersection_update(families_with_owned_carrier)

    confirmed_ordinal = {
        family_index: ordinal
        for ordinal, family_index in enumerate(sorted(connected_families))
    }
    group_order = {
        group.group_id: index for index, group in enumerate(grouping.groups)
    }
    local_by_group: dict[str, list[LocalLineType]] = {}
    global_types: list[GlobalLineType] = []
    for family_index, family in enumerate(families):
        if family_index not in connected_families:
            continue
        pattern_source = signatures[family.indices[0]].pattern_source
        ordinal = confirmed_ordinal[family_index]
        ops_by_group: dict[str, set[int]] = {}

        def add_to_source_group(dense_index: int) -> None:
            source_group_id = context.operation_index.group_id(dense_index)
            ops_by_group.setdefault(source_group_id, set()).add(dense_index)

        for signature_index in family.indices:
            signature = signatures[signature_index]
            for dense_index in signature.region.op_indices:
                if isinstance(
                    context.operation_index.operation(dense_index), PathOperationIR
                ):
                    add_to_source_group(dense_index)
        for dense_index, owner in best_owner.items():
            if owner.family_index == family_index:
                add_to_source_group(dense_index)
        members: list[GlobalLineTypeMember] = []
        for group_id, op_set in sorted(
            ops_by_group.items(), key=lambda item: group_order[item[0]]
        ):
            op_indices = tuple(sorted(op_set))
            existing = local_by_group.setdefault(group_id, [])
            type_id = f"type_v2_text_line_{ordinal + 1:03d}"
            local = LocalLineType(
                type_id=type_id,
                display_name=f"线型{len(existing) + 1}",
                line_type_index=len(existing) + 1,
                atom_count=_atom_count_for(context, op_indices),
                op_indices=op_indices,
                model=(
                    "repeated_pdf_text_with_dash"
                    if pattern_source == "pdf_text"
                    else "repeated_vector_text_with_dash"
                ),
                shape="文字型线型",
                shape_detail=(
                    "PDF 文字内容一致；同族文字锚点之间存在连续载线路径"
                    if pattern_source == "pdf_text"
                    else "完整文字绘制序列一致；同族文字锚点之间存在连续载线路径"
                ),
            )
            existing.append(local)
            members.append(GlobalLineTypeMember(
                case_id=group_id,
                type_id=type_id,
                display_name=local.display_name,
                atom_count=local.atom_count,
                model=local.model,
                shape=local.shape,
                shape_detail=local.shape_detail,
            ))
        global_op_indices = tuple(sorted({
            dense_index
            for member in members
            for local in local_by_group[member.case_id]
            if local.type_id == member.type_id
            for dense_index in local.op_indices
        }))
        global_types.append(GlobalLineType(
            global_type_id=f"global_type_{ordinal + 1:03d}",
            signature_family=(
                "pdf_text_dash_line"
                if pattern_source == "pdf_text"
                else "vector_text_dash_line"
            ),
            minimum_pair_similarity=_js_round(family.minimum_similarity, 3),
            op_indices=global_op_indices,
            members=tuple(members),
        ))

    groups: list[RecognizedGroup] = []
    for source_group in grouping.groups:
        path_ops = tuple(
            dense_index
            for dense_index in context.operation_index.group_indices(source_group.group_id)
            if isinstance(
                (operation := context.operation_index.operation(dense_index)),
                PathOperationIR,
            )
            and (operation.stroke or operation.fill)
            and _path_atom_count(operation) > 0
        )
        line_types = tuple(local_by_group.get(source_group.group_id, ()))
        assigned = {
            dense_index for line_type in line_types for dense_index in line_type.op_indices
        }
        non_line_ops = tuple(index for index in path_ops if index not in assigned)
        groups.append(RecognizedGroup(
            group_id=source_group.group_id,
            atom_count=_atom_count_for(context, path_ops),
            line_types=line_types,
            non_linetype=NonLineType(
                _atom_count_for(context, non_line_ops), non_line_ops
            ),
        ))
    local_count = sum(group.line_type_count for group in groups)
    result = LineTypeRecognitionResult(
        groups=tuple(groups),
        global_types=tuple(global_types),
        summary=RecognitionSummary(
            len(groups), len(groups), local_count, local_count, 0,
            len(global_types), sum(item.group_count > 1 for item in global_types),
        ),
    )
    result = LineTypeRecognitionResult.from_dict(result.to_dict())

    attached_dash_ops = tuple(sorted(
        dense_index for dense_index, owner in best_owner.items()
        if owner.family_index in connected_families
    ))
    affected_groups = tuple(sorted(
        {signature.region.group_id for signature in matched_signatures},
        key=lambda group_id: group_order[group_id],
    ))
    signature_index_by_region_id = {
        id(signature.region): index for index, signature in enumerate(signatures)
    }
    family_index_by_signature = {
        signature_index: family_index
        for family_index, family in enumerate(families)
        for signature_index in family.indices
    }
    confirmed_signature_indices = {
        signature_index
        for family_index in connected_families
        for signature_index in families[family_index].indices
    }
    confirmed_text_ops = {
        dense_index
        for signature_index in confirmed_signature_indices
        for dense_index in signatures[signature_index].region.op_indices
    }
    ordered_regions = sorted(regions, key=lambda region: (
        group_order[region.group_id], region.op_indices[0], region.region_id
    ))
    display_labels: dict[int, str] = {}
    vector_ordinal = pdf_ordinal = 0
    for region in ordered_regions:
        if region.pattern_source == "pdf_text":
            pdf_ordinal += 1
            label = f"P{pdf_ordinal:03d}"
        else:
            vector_ordinal += 1
            label = f"T{vector_ordinal:03d}"
        display_labels[id(region)] = label

    family_diagnostics: list[TextPatternFamilyDiagnostic] = []
    for family_index in sorted(connected_families):
        family = families[family_index]
        ordinal = confirmed_ordinal[family_index]
        pattern_source = signatures[family.indices[0]].pattern_source
        instances = tuple(
            PatternInstanceDiagnostic(
                signature_index,
                display_labels.get(id(signatures[signature_index].region), ""),
                signatures[signature_index].region.group_id,
                signatures[signature_index].region.op_indices,
                pattern_instance_dimensions(signatures[signature_index]),
                signatures[signature_index].literal_key,
            )
            for signature_index in family.indices
        )
        pair_items: list[tuple[float, int, int, RegionSimilarityBreakdown]] = []
        if pattern_source != "pdf_text":
            for order, left_index in enumerate(family.indices):
                for right_index in family.indices[order + 1:]:
                    detail = explain_region_similarity(
                        signatures[left_index], signatures[right_index]
                    )
                    pair_items.append((detail.score, left_index, right_index, detail))
            pair_items.sort(key=lambda item: item[0])
        pair_count = len(pair_items)
        if len(pair_items) > MAX_DIAGNOSTIC_PAIRS:
            pair_items = [
                *pair_items[:MAX_DIAGNOSTIC_PAIRS - 1], pair_items[-1]
            ]
        pairs: list[Mapping[str, object]] = []
        for _score, left_index, right_index, detail in pair_items:
            pairs.append({
                "left_label": display_labels.get(id(signatures[left_index].region), ""),
                "right_label": display_labels.get(id(signatures[right_index].region), ""),
                "left_group": signatures[left_index].region.group_id,
                "right_group": signatures[right_index].region.group_id,
                **detail.to_dict(),
            })
        family_diagnostics.append(TextPatternFamilyDiagnostic(
            f"global_type_{ordinal + 1:03d}",
            pattern_source,
            _js_round(family.minimum_similarity, 3),
            len(family.indices),
            pair_count,
            instances,
            tuple(pairs),
            (
                signatures[family.indices[0]].literal_key
                if pattern_source == "pdf_text" else None
            ),
        ))

    region_instances: list[PatternRegionInstance] = []
    for region in ordered_regions:
        signature_index = signature_index_by_region_id.get(id(region))
        family_index = (
            family_index_by_signature.get(signature_index)
            if signature_index is not None else None
        )
        confirmed = (
            confirmed_ordinal.get(family_index)
            if family_index is not None else None
        )
        region_instances.append(PatternRegionInstance(
            display_labels[id(region)],
            region.region_id,
            region.group_id,
            region.op_indices,
            region.bounds,
            region.orientation_degrees,
            region.confidence,
            region.evidence,
            family_index is not None,
            confirmed is not None,
            region.region_id.startswith("vector_text_recovered_"),
            region.pattern_source,
            region.literal_text,
            f"text_family_{family_index + 1:03d}" if family_index is not None else None,
            f"global_type_{confirmed + 1:03d}" if confirmed is not None else None,
        ))
    audit = VectorTextFamilyAudit(
        family_diagnostics=tuple(family_diagnostics),
        detected_region_count=len(regions),
        eligible_region_count=len(signatures),
        matched_instance_count=len(matched_signatures),
        matched_family_count=len(families),
        dash_connected_family_count=len(connected_families),
        line_type_confirmed_instance_count=len(confirmed_signature_indices),
        line_type_confirmed_text_op_count=len(confirmed_text_ops),
        matched_text_op_count=len(matched_text_ops),
        attached_dash_op_count=len(attached_dash_ops),
        bridged_route_op_count=len(bridged_route_ops),
        matched_text_op_indices=tuple(sorted(matched_text_ops)),
        line_type_confirmed_text_op_indices=tuple(sorted(confirmed_text_ops)),
        affected_group_ids=affected_groups,
        region_instances=tuple(region_instances),
    )
    return TextFamilyRecognition(result, audit)


recognize_repeated_vector_text_families = recognize_repeated_text_pattern_families


__all__ = [
    "MATCH_THRESHOLD",
    "PatternInstanceDiagnostic",
    "PatternRegionInstance",
    "RegionFrame",
    "RegionSignature",
    "RegionSimilarityBreakdown",
    "TextFamilyRecognition",
    "TextPatternFamilyDiagnostic",
    "VectorTextFamilyAudit",
    "explain_region_pair",
    "explain_region_similarity",
    "pattern_instance_dimensions",
    "rebuild_region_signature",
    "recognize_repeated_text_pattern_families",
    "recognize_repeated_vector_text_families",
    "vector_text_region_similarity",
]
