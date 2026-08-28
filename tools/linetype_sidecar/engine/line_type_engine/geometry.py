"""Geometry primitives for candidate sequential page grouping.

The module deliberately operates on :mod:`line_type_engine.ir` values only.
It has no PyMuPDF, browser, TypeScript-oracle, or cache dependency.  Path
curves are flattened deterministically before adjacent visible-contour gaps
are measured.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

from .ir import BoundsIR, OperationIR, PathOperationIR, PointIR


_LEAF_SIZE = 8
_MAX_CURVE_DEPTH = 12


@dataclass(frozen=True, slots=True)
class LineEdge:
    """One flattened path edge in original command order."""

    start: PointIR
    end: PointIR
    bounds: BoundsIR
    order: int


@dataclass(frozen=True, slots=True)
class _EdgeTree:
    bounds: BoundsIR
    edge_count: int
    edges: tuple[LineEdge, ...] = ()
    left: "_EdgeTree | None" = None
    right: "_EdgeTree | None" = None


def page_diagonal(bounds: BoundsIR) -> float:
    """Return the page scale used by all ratio-based grouping thresholds."""

    return max(1.0, math.hypot(bounds.width, bounds.height))


def bounds_gap(left: BoundsIR, right: BoundsIR) -> float:
    """Euclidean distance between two axis-aligned bounds (zero on overlap)."""

    dx = max(0.0, left.min_x - right.max_x, right.min_x - left.max_x)
    dy = max(0.0, left.min_y - right.max_y, right.min_y - left.max_y)
    return math.hypot(dx, dy)


def _edge(start: PointIR, end: PointIR, order: int) -> LineEdge:
    return LineEdge(
        start=start,
        end=end,
        bounds=BoundsIR(
            min(start[0], end[0]),
            min(start[1], end[1]),
            max(start[0], end[0]),
            max(start[1], end[1]),
        ),
        order=order,
    )


def _midpoint(left: PointIR, right: PointIR) -> PointIR:
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def _squared_point_segment_distance(
    point: PointIR,
    start: PointIR,
    end: PointIR,
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length_squared = dx * dx + dy * dy
    if length_squared <= 1e-24:
        return (point[0] - start[0]) ** 2 + (point[1] - start[1]) ** 2
    amount = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / length_squared,
        ),
    )
    nearest = (start[0] + amount * dx, start[1] + amount * dy)
    return (point[0] - nearest[0]) ** 2 + (point[1] - nearest[1]) ** 2


def _append_cubic(
    edges: list[LineEdge],
    start: PointIR,
    control_1: PointIR,
    control_2: PointIR,
    end: PointIR,
    flatness_squared: float,
) -> None:
    pending = [(start, control_1, control_2, end, 0)]
    while pending:
        curve_start, first, second, curve_end, depth = pending.pop()
        flatness = max(
            _squared_point_segment_distance(first, curve_start, curve_end),
            _squared_point_segment_distance(second, curve_start, curve_end),
        )
        if flatness <= flatness_squared or depth >= _MAX_CURVE_DEPTH:
            edges.append(_edge(curve_start, curve_end, len(edges)))
            continue

        start_first = _midpoint(curve_start, first)
        first_second = _midpoint(first, second)
        second_end = _midpoint(second, curve_end)
        left_control = _midpoint(start_first, first_second)
        right_control = _midpoint(first_second, second_end)
        split = _midpoint(left_control, right_control)
        next_depth = depth + 1
        # LIFO: enqueue the right half first to preserve PDF command order.
        pending.append((split, right_control, second_end, curve_end, next_depth))
        pending.append((curve_start, start_first, left_control, split, next_depth))


def flatten_path(operation: PathOperationIR, diagonal: float) -> tuple[LineEdge, ...]:
    """Flatten a path without mutating the IR or inferring semantic shapes."""

    flatness = max(1e-7, diagonal * 1e-7)
    edges: list[LineEdge] = []
    current: PointIR | None = None
    subpath_start: PointIR | None = None
    subpath_has_edge = False
    subpath_closed = False

    def append_edge(start: PointIR, end: PointIR) -> None:
        nonlocal subpath_has_edge
        edges.append(_edge(start, end, len(edges)))
        subpath_has_edge = True

    def close_subpath(force: bool = False) -> None:
        nonlocal current, subpath_closed
        should_close = force or operation.fill
        if (
            should_close
            and subpath_has_edge
            and not subpath_closed
            and current is not None
            and subpath_start is not None
        ):
            append_edge(current, subpath_start)
            current = subpath_start
            subpath_closed = True

    for segment in operation.segments:
        if segment.kind == "move":
            close_subpath()
            current = segment.end
            subpath_start = segment.end
            subpath_has_edge = False
            subpath_closed = False
        elif segment.kind == "line":
            if current is not None and segment.end is not None:
                append_edge(current, segment.end)
            elif segment.end is not None:
                subpath_start = segment.end
            current = segment.end
            subpath_closed = False
        elif segment.kind == "curve":
            if (
                current is not None
                and segment.control_1 is not None
                and segment.control_2 is not None
                and segment.end is not None
            ):
                before = len(edges)
                _append_cubic(
                    edges,
                    current,
                    segment.control_1,
                    segment.control_2,
                    segment.end,
                    flatness * flatness,
                )
                if len(edges) > before:
                    subpath_has_edge = True
            elif segment.end is not None:
                subpath_start = segment.end
            current = segment.end
            subpath_closed = False
        else:
            close_subpath(force=True)

    close_subpath(force=operation.close_path)
    return tuple(edges)


def _cross(first: PointIR, second: PointIR, third: PointIR) -> float:
    return (
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _point_on_segment(
    point: PointIR,
    start: PointIR,
    end: PointIR,
    epsilon: float,
) -> bool:
    return (
        abs(_cross(start, end, point)) <= epsilon
        and min(start[0], end[0]) - epsilon
        <= point[0]
        <= max(start[0], end[0]) + epsilon
        and min(start[1], end[1]) - epsilon
        <= point[1]
        <= max(start[1], end[1]) + epsilon
    )


def _edges_intersect(left: LineEdge, right: LineEdge) -> bool:
    coordinate_scale = max(
        1.0,
        *(abs(value) for point in (left.start, left.end, right.start, right.end) for value in point),
    )
    epsilon = math.ulp(1.0) * 64.0 * coordinate_scale * coordinate_scale
    first = _cross(left.start, left.end, right.start)
    second = _cross(left.start, left.end, right.end)
    third = _cross(right.start, right.end, left.start)
    fourth = _cross(right.start, right.end, left.end)
    if (
        ((first > epsilon and second < -epsilon) or (first < -epsilon and second > epsilon))
        and ((third > epsilon and fourth < -epsilon) or (third < -epsilon and fourth > epsilon))
    ):
        return True
    return (
        (abs(first) <= epsilon and _point_on_segment(right.start, left.start, left.end, epsilon))
        or (abs(second) <= epsilon and _point_on_segment(right.end, left.start, left.end, epsilon))
        or (abs(third) <= epsilon and _point_on_segment(left.start, right.start, right.end, epsilon))
        or (abs(fourth) <= epsilon and _point_on_segment(left.end, right.start, right.end, epsilon))
    )


def _edge_gap(left: LineEdge, right: LineEdge) -> float:
    if _edges_intersect(left, right):
        return 0.0
    return math.sqrt(
        min(
            _squared_point_segment_distance(left.start, right.start, right.end),
            _squared_point_segment_distance(left.end, right.start, right.end),
            _squared_point_segment_distance(right.start, left.start, left.end),
            _squared_point_segment_distance(right.end, left.start, left.end),
        )
    )


def _union_bounds(bounds: Iterable[BoundsIR]) -> BoundsIR:
    iterator = iter(bounds)
    result = next(iterator)
    for item in iterator:
        result = result.union(item)
    return result


def _bounds_of_edges(edges: tuple[LineEdge, ...]) -> BoundsIR:
    """一批边的包围盒 —— 用普通浮点扫一遍，只构造**一个** BoundsIR。

    等价于 _union_bounds(edge.bounds for edge in edges)：并集就是逐分量的
    min/max，一次扫描取四个极值和逐对 union 得到的是同一组数值。区别只在
    中间对象：原来每合并一条边就造一个 BoundsIR，n 条边造 n 个。

    为什么值得改：_build_tree 在每个节点上都要求一次子树包围盒，实测
    gladstone P2 单页分组阶段共 1780 万次 BoundsIR 构造（607k 次
    _build_tree 调用），仅 __post_init__ + _finite 就占整页 CPU 的 28%。
    改成一次扫描后，这里的构造数从 O(n·log n) 降到 O(节点数)。
    """
    iterator = iter(edges)
    first = next(iterator).bounds
    min_x, min_y = first.min_x, first.min_y
    max_x, max_y = first.max_x, first.max_y
    for edge in iterator:
        bounds = edge.bounds
        if bounds.min_x < min_x:
            min_x = bounds.min_x
        if bounds.min_y < min_y:
            min_y = bounds.min_y
        if bounds.max_x > max_x:
            max_x = bounds.max_x
        if bounds.max_y > max_y:
            max_y = bounds.max_y
    return BoundsIR(min_x, min_y, max_x, max_y)


def _build_tree(edges: tuple[LineEdge, ...]) -> _EdgeTree:
    tree_bounds = _bounds_of_edges(edges)
    if len(edges) <= _LEAF_SIZE:
        return _EdgeTree(tree_bounds, len(edges), edges=edges)
    split_x = tree_bounds.width >= tree_bounds.height
    ordered = tuple(
        sorted(
            edges,
            key=lambda edge: (
                (edge.bounds.min_x + edge.bounds.max_x) / 2.0
                if split_x
                else (edge.bounds.min_y + edge.bounds.max_y) / 2.0,
                edge.order,
            ),
        )
    )
    middle = len(ordered) // 2
    return _EdgeTree(
        tree_bounds,
        len(edges),
        left=_build_tree(ordered[:middle]),
        right=_build_tree(ordered[middle:]),
    )


def _path_centerline_gap(
    left_edges: tuple[LineEdge, ...],
    right_edges: tuple[LineEdge, ...],
) -> float:
    if not left_edges or not right_edges:
        return math.inf
    best = _edge_gap(left_edges[-1], right_edges[0])
    left_tree = _build_tree(left_edges)
    right_tree = _build_tree(right_edges)
    pending: list[tuple[float, int, _EdgeTree, _EdgeTree]] = []
    serial = 0

    def enqueue(left: _EdgeTree, right: _EdgeTree) -> None:
        nonlocal serial
        lower_bound = bounds_gap(left.bounds, right.bounds)
        if lower_bound < best:
            serial += 1
            heapq.heappush(pending, (lower_bound, serial, left, right))

    enqueue(left_tree, right_tree)
    while pending and best > 0.0:
        lower_bound, _, left_node, right_node = heapq.heappop(pending)
        if lower_bound >= best:
            continue
        left_leaf = bool(left_node.edges)
        right_leaf = bool(right_node.edges)
        if left_leaf and right_leaf:
            for left_edge in left_node.edges:
                for right_edge in right_node.edges:
                    best = min(best, _edge_gap(left_edge, right_edge))
                    if best == 0.0:
                        return 0.0
            continue
        split_left = not left_leaf and (
            right_leaf or left_node.edge_count >= right_node.edge_count
        )
        if split_left:
            assert left_node.left is not None and left_node.right is not None
            enqueue(left_node.left, right_node)
            enqueue(left_node.right, right_node)
        else:
            assert right_node.left is not None and right_node.right is not None
            enqueue(left_node, right_node.left)
            enqueue(left_node, right_node.right)
    return best


def _visible_stroke_radius(operation: PathOperationIR) -> float:
    if not operation.stroke or operation.stroke_opacity <= 0.0:
        return 0.0
    return 0.25 if operation.hairline else max(0.0, operation.line_width) / 2.0


def operation_contour_gap(
    left: OperationIR,
    right: OperationIR,
    diagonal: float,
) -> float:
    """Measure the visible gap between adjacent paint operations.

    Exact flattened path contours are used for path/path pairs.  Text and
    images currently expose bounds rather than outlines in PageIR v1, so mixed
    pairs intentionally fall back to their bounds.
    """

    if not isinstance(left, PathOperationIR) or not isinstance(right, PathOperationIR):
        return bounds_gap(left.bounds, right.bounds)
    left_edges = flatten_path(left, diagonal)
    right_edges = flatten_path(right, diagonal)
    if not left_edges or not right_edges:
        return bounds_gap(left.bounds, right.bounds)
    return max(
        0.0,
        _path_centerline_gap(left_edges, right_edges)
        - _visible_stroke_radius(left)
        - _visible_stroke_radius(right),
    )


def connection_tolerance(
    left: OperationIR,
    right: OperationIR,
    diagonal: float,
) -> float:
    """Small visible-gap tolerance; connected bends must remain contiguous."""

    def line_width(operation: OperationIR) -> float:
        value = getattr(operation, "line_width", 0.0)
        return max(0.0, float(value or 0.0))

    return max(
        0.0005 * diagonal,
        min(0.005 * diagonal, 2.0 * max(line_width(left), line_width(right))),
    )
