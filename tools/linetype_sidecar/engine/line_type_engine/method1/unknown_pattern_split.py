#!/usr/bin/env python3
"""Discover repeated vector line patterns in a PDF command snippet.

Put the commands in ``commands.txt`` beside this script or in the current
directory, then run:

    python unknown_pattern_split.py

An explicit input and output directory are also accepted:

    python unknown_pattern_split.py path/to/commands.txt -o result_dir

Process every ``*commands*.txt`` in the current directory with:

    python unknown_pattern_split.py --all

Without ``-o``, the colored image is written beside the input as
``<input-name>-split.svg``.

The implementation is a standard-library, pure-vector pattern detector.  It
keeps the original single-carrier fast path, adds a multi-carrier fallback,
and finally tries a guarded shared-reference-path model for attached motifs or
repeating ink/gap sequences.  No raster image is used by the classifier.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import colorsys
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.sax.saxutils import escape


EPS = 1e-8
MEASURE_EPS = 1e-9
CURVE_STEPS = 8
Point = tuple[float, float]


# ---------------------------------------------------------------------------
# Geometry


def add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def mul(p: Point, factor: float) -> Point:
    return p[0] * factor, p[1] * factor


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def normalize(p: Point) -> Point:
    length = math.hypot(*p)
    return mul(p, 1.0 / length) if length > EPS else (0.0, 0.0)


def acute_angle_degrees(a: Point, b: Point) -> float:
    cosine = max(-1.0, min(1.0, abs(dot(normalize(a), normalize(b)))))
    return math.degrees(math.acos(cosine))


def directed_angle_degrees(a: Point, b: Point) -> float:
    """Unsigned 0..180 degree turn; unlike ``acute_angle_degrees`` it detects U-turns."""
    cosine = max(-1.0, min(1.0, dot(normalize(a), normalize(b))))
    return math.degrees(math.acos(cosine))


def lateral_distance_to_line(point: Point, start: Point, end: Point) -> float:
    direction = normalize(sub(end, start))
    relative = sub(point, start)
    return abs(relative[0] * direction[1] - relative[1] * direction[0])


def first_polyline_tangent(points: list[Point]) -> Point:
    origin = points[0]
    for point in points[1:]:
        tangent = sub(point, origin)
        if math.hypot(*tangent) > EPS:
            return tangent
    return 0.0, 0.0


def last_polyline_tangent(points: list[Point]) -> Point:
    end = points[-1]
    for point in reversed(points[:-1]):
        tangent = sub(end, point)
        if math.hypot(*tangent) > EPS:
            return tangent
    return 0.0, 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(1, len(values))


def median(values: Iterable[float]) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def population_std(values: Iterable[float]) -> float:
    values = list(values)
    average = mean(values)
    return math.sqrt(mean((value - average) ** 2 for value in values))


def quantile(sorted_values: list[float], ratio: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * ratio
    lower, upper = math.floor(position), math.ceil(position)
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def polyline_length(points: list[Point]) -> float:
    return sum(distance(points[index - 1], points[index]) for index in range(1, len(points)))


def polyline_halfway_point(points: list[Point]) -> Point:
    halfway = polyline_length(points) / 2
    walked = 0.0
    for index in range(1, len(points)):
        segment_length = distance(points[index - 1], points[index])
        if walked + segment_length >= halfway:
            ratio = (halfway - walked) / segment_length if segment_length > EPS else 0.0
            return add(points[index - 1], mul(sub(points[index], points[index - 1]), ratio))
        walked += segment_length
    return points[-1]


def resample_polyline(points: list[Point], count: int = 16) -> list[Point]:
    if len(points) < 2 or count <= 1:
        return list(points)
    lengths = [distance(points[index - 1], points[index]) for index in range(1, len(points))]
    total = sum(lengths)
    if total <= EPS:
        return [points[0]] * count
    result: list[Point] = []
    segment_index = 0
    segment_start = 0.0
    for sample_index in range(count):
        target = total * sample_index / (count - 1)
        while segment_index < len(lengths) - 1 and segment_start + lengths[segment_index] < target:
            segment_start += lengths[segment_index]
            segment_index += 1
        length = lengths[segment_index]
        ratio = (target - segment_start) / length if length > EPS else 0.0
        result.append(add(points[segment_index], mul(sub(points[segment_index + 1], points[segment_index]), ratio)))
    return result


def cubic_point(start: Point, control1: Point, control2: Point, end: Point, t: float) -> Point:
    inverse = 1 - t
    return (
        inverse**3 * start[0] + 3 * inverse**2 * t * control1[0]
        + 3 * inverse * t**2 * control2[0] + t**3 * end[0],
        inverse**3 * start[1] + 3 * inverse**2 * t * control1[1]
        + 3 * inverse * t**2 * control2[1] + t**3 * end[1],
    )


def principal_frame(points: list[Point], center: Point) -> tuple[Point, float]:
    xx = mean((point[0] - center[0]) ** 2 for point in points)
    yy = mean((point[1] - center[1]) ** 2 for point in points)
    xy = mean((point[0] - center[0]) * (point[1] - center[1]) for point in points)
    trace = xx + yy
    discriminant = math.sqrt(max(0.0, (xx - yy) ** 2 + 4 * xy**2))
    major_value = max(0.0, (trace + discriminant) / 2)
    minor_value = max(0.0, (trace - discriminant) / 2)
    if abs(xy) > EPS:
        direction = normalize((major_value - yy, xy))
    else:
        direction = (1.0, 0.0) if xx >= yy else (0.0, 1.0)
    aspect = math.sqrt(minor_value / major_value) if major_value > EPS else 0.0
    return direction, aspect


def js_round_nonnegative(value: float) -> int:
    """Match JavaScript Math.round for the non-negative values used here."""
    return math.floor(value + 0.5)


# ---------------------------------------------------------------------------
# Minimal vector command parser


Matrix = tuple[float, float, float, float, float, float]


def multiply_matrix(left: Matrix, right: Matrix) -> Matrix:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def transform_point(matrix: Matrix, x: float, y: float) -> Point:
    return matrix[0] * x + matrix[2] * y + matrix[4], matrix[1] * x + matrix[3] * y + matrix[5]


def matrix_scale(matrix: Matrix) -> float:
    return max(0.000001, (math.hypot(matrix[0], matrix[1]) + math.hypot(matrix[2], matrix[3])) / 2)


@dataclass
class GraphicsState:
    ctm: Matrix = (1, 0, 0, 1, 0, 0)
    line_width: float = 1.0
    line_cap: int = 0
    stroke_color: tuple[float, ...] = (0.0, 0.0, 0.0)
    stroke_alpha: float = 1.0
    fill_alpha: float = 1.0
    clip: tuple[float, float, float, float] | None = None


@dataclass
class RawSubpath:
    points: list[Point]
    start_point: Point
    current_point: Point
    line_count: int = 0
    curve_count: int = 0
    explicitly_closed: bool = False


@dataclass
class Atom:
    id: int
    points: list[Point]
    samples: list[Point]
    length: float
    center: Point
    scale: float
    aspect_ratio: float
    principal_direction: Point
    closed: bool
    curve_segments: int
    line_segments: int
    paint_mode: str
    line_width: float
    line_cap: int
    stroke_color: tuple[float, ...]


def finalize_atom(
    atom_id: int,
    subpath: RawSubpath,
    measured_closed: bool,
    paint_mode: str,
    state: GraphicsState,
) -> Atom:
    points = list(subpath.points)
    samples = resample_polyline(points, 16)
    center = (mean(point[0] for point in samples), mean(point[1] for point in samples))
    scale = max(EPS, 2 * max(distance(point, center) for point in samples))
    principal_direction, aspect_ratio = principal_frame(samples, center)
    return Atom(
        id=atom_id,
        points=points,
        samples=samples,
        length=polyline_length(points),
        center=center,
        scale=scale,
        aspect_ratio=aspect_ratio,
        principal_direction=principal_direction,
        closed=measured_closed or distance(points[0], points[-1]) <= 1e-3,
        curve_segments=subpath.curve_count,
        line_segments=subpath.line_count,
        paint_mode=paint_mode,
        line_width=abs(state.line_width) * matrix_scale(state.ctm),
        line_cap=state.line_cap,
        stroke_color=state.stroke_color,
    )


NUMBER_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?$")
DELIMITERS = set("()<>[]{}/%")


def lexical_tokens(source: str) -> Iterable[tuple[str, Any]]:
    """Small PDF tokenizer sufficient for vector content streams."""
    index, size = 0, len(source)
    while index < size:
        char = source[index]
        if char.isspace():
            index += 1
            continue
        if char == "%":
            newline = source.find("\n", index)
            index = size if newline < 0 else newline + 1
            continue
        if char in "[]":
            yield char, char
            index += 1
            continue
        if char == "/":
            end = index + 1
            while end < size and not source[end].isspace() and source[end] not in DELIMITERS:
                end += 1
            yield "value", source[index:end]
            index = end
            continue
        if char == "(":
            depth, end = 1, index + 1
            while end < size and depth:
                if source[end] == "\\":
                    end += 2
                    continue
                if source[end] == "(":
                    depth += 1
                elif source[end] == ")":
                    depth -= 1
                end += 1
            yield "value", source[index:end]
            index = end
            continue
        if char == "<" and index + 1 < size and source[index + 1] != "<":
            end = source.find(">", index + 1)
            end = size - 1 if end < 0 else end
            yield "value", source[index : end + 1]
            index = end + 1
            continue
        if source.startswith("<<", index) or source.startswith(">>", index):
            yield "value", source[index : index + 2]
            index += 2
            continue
        end = index
        while end < size and not source[end].isspace() and source[end] not in DELIMITERS:
            end += 1
        token = source[index:end]
        if not token:  # Unknown delimiter: consume it safely.
            index += 1
            continue
        if NUMBER_RE.match(token):
            number = float(token)
            yield ("number", number) if math.isfinite(number) else ("value", token)
        else:
            yield "word", token
        index = end
        if token == "ID":
            # Inline-image bytes are not PDF commands.  Skip through the
            # whitespace-delimited EI terminator, like the production scanner.
            if index < size and source[index] == "\r":
                index += 1
            if index < size and source[index].isspace():
                index += 1
            match = re.search(r"(?<!\S)EI(?=\s|$)", source[index:])
            index = index + match.start() if match else size


KNOWN_OPERATORS = {
    "q", "Q", "cm", "w", "J", "j", "M", "d", "ri", "i", "gs",
    "m", "l", "c", "v", "y", "re", "h",
    "S", "s", "f", "F", "f*", "B", "B*", "b", "b*", "n", "W", "W*",
    "G", "g", "RG", "rg", "K", "k", "CS", "cs", "SC", "SCN", "sc", "scn",
    "BT", "ET", "Tc", "Tw", "Tz", "TL", "Tf", "Tr", "Ts", "Td", "TD", "Tm", "T*",
    "Tj", "TJ", "'", '"', "Do", "sh", "BI", "ID", "EI", "MP", "DP", "BMC", "BDC", "EMC",
}


def bounds_for_points(points: list[Point]) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def intersect_bounds(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    if left is None:
        return right
    if right is None:
        return left
    result = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    return result if result[0] <= result[2] and result[1] <= result[3] else None


def expand_bounds(bounds: tuple[float, float, float, float], amount: float) -> tuple[float, float, float, float]:
    return bounds[0] - amount, bounds[1] - amount, bounds[2] + amount, bounds[3] + amount


def parse_painted_atoms(source: str) -> list[Atom]:
    state = GraphicsState()
    state_stack: list[GraphicsState] = []
    subpaths: list[RawSubpath] = []
    current: RawSubpath | None = None
    path_bound_points: list[Point] = []
    pending_clip = False
    atoms: list[Atom] = []

    def clear_path() -> None:
        nonlocal subpaths, current, path_bound_points, pending_clip
        subpaths, current, path_bound_points, pending_clip = [], None, [], False

    def include_bounds(*points: Point) -> None:
        path_bound_points.extend(points)

    def begin(point: Point) -> None:
        nonlocal current
        current = RawSubpath(points=[point], start_point=point, current_point=point)
        subpaths.append(current)
        include_bounds(point)

    def close_current() -> None:
        if current is None or not current.points:
            return
        if distance(current.current_point, current.start_point) > MEASURE_EPS:
            current.points.append(current.start_point)
        current.explicitly_closed = True
        current.current_point = current.start_point

    def apply_pending_clip() -> None:
        nonlocal pending_clip
        if pending_clip:
            path_bounds = bounds_for_points(path_bound_points)
            if path_bounds is not None:
                state.clip = intersect_bounds(state.clip, path_bounds)
        pending_clip = False

    def paint(fill: bool, stroke: bool, close_first: bool = False) -> None:
        if close_first:
            close_current()
        path_bounds = bounds_for_points(path_bound_points)
        visible = path_bounds is not None
        visible_stroke = stroke and state.stroke_alpha > 0
        visible_fill = fill and state.fill_alpha > 0
        if visible and visible_stroke:
            width = abs(state.line_width) * matrix_scale(state.ctm)
            path_bounds = expand_bounds(path_bounds, max(width / 2, 0.25))
        if visible and state.clip is not None:
            visible = intersect_bounds(path_bounds, state.clip) is not None
        visible = visible and (visible_stroke or visible_fill)
        if visible:
            mode = "stroke-fill" if visible_stroke and visible_fill else "fill" if visible_fill else "stroke"
            for raw in subpaths:
                if len(raw.points) < 2:
                    continue
                length = polyline_length(raw.points)
                measured_closed = raw.explicitly_closed or distance(raw.points[0], raw.points[-1]) <= max(1e-7, length * 1e-5)
                atoms.append(finalize_atom(len(atoms), raw, measured_closed, mode, state))
        apply_pending_clip()
        clear_path()

    def numbers(operands: list[Any], count: int) -> list[float] | None:
        values = operands[-count:]
        return [float(value) for value in values] if len(values) == count and all(isinstance(value, (int, float)) for value in values) else None

    def execute(operator: str, operands: list[Any]) -> None:
        nonlocal state, current, pending_clip
        if operator == "q":
            state_stack.append(copy.deepcopy(state))
        elif operator == "Q":
            if state_stack:
                state = state_stack.pop()
        elif operator == "cm" and (values := numbers(operands, 6)):
            state.ctm = multiply_matrix(state.ctm, tuple(values))  # type: ignore[arg-type]
        elif operator == "w" and (values := numbers(operands, 1)):
            state.line_width = values[0]
        elif operator == "J" and (values := numbers(operands, 1)):
            state.line_cap = max(0, min(2, int(values[0])))
        elif operator == "G" and (values := numbers(operands, 1)):
            state.stroke_color = (values[0],)
        elif operator == "RG" and (values := numbers(operands, 3)):
            state.stroke_color = tuple(values)
        elif operator == "K" and (values := numbers(operands, 4)):
            state.stroke_color = tuple(values)
        elif operator == "m" and (values := numbers(operands, 2)):
            begin(transform_point(state.ctm, *values))
        elif operator == "l" and (values := numbers(operands, 2)):
            point = transform_point(state.ctm, *values)
            if current is None:
                begin(point)
            else:
                current.points.append(point)
                current.line_count += 1
                current.current_point = point
                include_bounds(point)
        elif operator == "c" and (values := numbers(operands, 6)):
            control1 = transform_point(state.ctm, values[0], values[1])
            control2 = transform_point(state.ctm, values[2], values[3])
            end = transform_point(state.ctm, values[4], values[5])
            if current is None:
                begin(control1)
            assert current is not None
            start = current.current_point
            current.points.extend(cubic_point(start, control1, control2, end, step / CURVE_STEPS) for step in range(1, CURVE_STEPS + 1))
            current.curve_count += 1
            current.current_point = end
            include_bounds(control1, control2, end)
        elif operator == "v" and current is not None and (values := numbers(operands, 4)):
            start = current.current_point
            control2 = transform_point(state.ctm, values[0], values[1])
            end = transform_point(state.ctm, values[2], values[3])
            current.points.extend(cubic_point(start, start, control2, end, step / CURVE_STEPS) for step in range(1, CURVE_STEPS + 1))
            current.curve_count += 1
            current.current_point = end
            include_bounds(control2, end)
        elif operator == "y" and current is not None and (values := numbers(operands, 4)):
            start = current.current_point
            control1 = transform_point(state.ctm, values[0], values[1])
            end = transform_point(state.ctm, values[2], values[3])
            current.points.extend(cubic_point(start, control1, end, end, step / CURVE_STEPS) for step in range(1, CURVE_STEPS + 1))
            current.curve_count += 1
            current.current_point = end
            include_bounds(control1, end)
        elif operator == "re" and (values := numbers(operands, 4)):
            x, y, width, height = values
            corners = [
                transform_point(state.ctm, x, y),
                transform_point(state.ctm, x + width, y),
                transform_point(state.ctm, x + width, y + height),
                transform_point(state.ctm, x, y + height),
            ]
            begin(corners[0])
            assert current is not None
            current.points.extend(corners[1:])
            current.line_count = 3
            current.current_point = corners[-1]
            include_bounds(*corners[1:])
            close_current()
        elif operator == "h":
            close_current()
        elif operator == "S":
            paint(False, True)
        elif operator == "s":
            paint(False, True, True)
        elif operator in {"f", "F", "f*"}:
            paint(True, False)
        elif operator in {"B", "B*"}:
            paint(True, True)
        elif operator in {"b", "b*"}:
            paint(True, True, True)
        elif operator in {"W", "W*"}:
            pending_clip = True
        elif operator == "n":
            apply_pending_clip()
            clear_path()

    operands: list[Any] = []
    array_stack: list[list[Any]] = []
    for kind, value in lexical_tokens(source):
        if kind == "[":
            array_stack.append([])
        elif kind == "]":
            if array_stack:
                completed = array_stack.pop()
                (array_stack[-1] if array_stack else operands).append(completed)
        elif kind in {"number", "value"}:
            (array_stack[-1] if array_stack else operands).append(value)
        elif kind == "word":
            if array_stack:
                array_stack[-1].append(value)
            elif value in KNOWN_OPERATORS:
                execute(value, operands)
                operands = []
                array_stack = []
            elif value not in {"true", "false", "null"}:
                # An unknown word in a content stream is normally an operator.
                operands = []
                array_stack = []
    return atoms


# ---------------------------------------------------------------------------
# Unknown motif discovery


@dataclass
class Candidate:
    id: int
    members: list[Atom]
    atom_ids: list[int]
    center: Point
    scale: float
    fingerprint: dict[str, Any]


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.component_size = [1] * size

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> bool:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return False
        if self.component_size[left_root] < self.component_size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.component_size[left_root] += self.component_size[right_root]
        return True


def radial_quantiles(points: list[Point], center: Point, scale: float) -> list[float]:
    radii = sorted(distance(point, center) / max(scale, EPS) for point in points)
    return [quantile(radii, ratio) for ratio in (0.1, 0.25, 0.5, 0.75, 0.9)]


def make_candidate(candidate_id: int, members: list[Atom]) -> Candidate:
    center = (mean(atom.center[0] for atom in members), mean(atom.center[1] for atom in members))
    samples = [point for atom in members for point in atom.samples]
    scale = max(EPS, 2 * max(distance(point, center) for point in samples))
    _, aspect_ratio = principal_frame(samples, center)
    total_length = sum(atom.length for atom in members)
    member_ratios = sorted((atom.length / max(total_length, EPS) for atom in members), reverse=True)
    center_distances: list[float] = []
    direction_angles: list[float] = []
    for left in range(len(members)):
        for right in range(left + 1, len(members)):
            center_distances.append(distance(members[left].center, members[right].center) / scale)
            direction_angles.append(acute_angle_degrees(members[left].principal_direction, members[right].principal_direction) / 90)
    fingerprint = {
        "member_count": len(members),
        "closed_count": sum(atom.closed for atom in members),
        "curved_count": sum(atom.curve_segments > 0 for atom in members),
        "filled_count": sum("fill" in atom.paint_mode for atom in members),
        "log_scale": math.log(scale),
        "aspect_ratio": aspect_ratio,
        "normalized_length": total_length / scale,
        "member_length_ratios": member_ratios,
        "center_distances": sorted(center_distances),
        "direction_angles": sorted(direction_angles),
        "radial_quantiles": radial_quantiles(samples, center, scale),
    }
    return Candidate(candidate_id, members, sorted(atom.id for atom in members), center, scale, fingerprint)


def compact_atoms(left: Atom, right: Atom) -> bool:
    smaller = min(left.scale, right.scale)
    ratio = max(left.scale, right.scale) / max(smaller, EPS)
    return distance(left.center, right.center) <= smaller * 0.4 and ratio <= 3


def same_junction_style(left: Atom, right: Atom) -> bool:
    if left.paint_mode != "stroke" or right.paint_mode != "stroke":
        return False
    if left.line_cap != right.line_cap:
        return False
    width_max = max(abs(left.line_width), abs(right.line_width), EPS)
    if abs(left.line_width - right.line_width) > max(1e-6, width_max * 0.05):
        return False
    if len(left.stroke_color) != len(right.stroke_color):
        return False
    return all(abs(a - b) <= 1e-6 for a, b in zip(left.stroke_color, right.stroke_color))


def junction_tolerance(left: Atom, right: Atom) -> float:
    smaller_scale = max(EPS, min(left.scale, right.scale))
    positive_widths = [
        abs(width) for width in (left.line_width, right.line_width)
        if abs(width) > EPS
    ]
    width_limit = min(positive_widths) * 0.1 if positive_widths else 0.0
    geometric_limit = min(width_limit, smaller_scale * 0.02) if width_limit else 0.0
    coordinates = [
        value
        for atom in (left, right)
        for point in atom.points
        for value in point
    ]
    numeric_limit = max(
        MEASURE_EPS,
        max((math.ulp(abs(value)) * 8 for value in coordinates), default=MEASURE_EPS),
    )
    return max(numeric_limit, geometric_limit)


def endpoint_hits_internal_line_vertex(endpoint_atom: Atom, internal_atom: Atom, tolerance: float) -> bool:
    if endpoint_atom.curve_segments or internal_atom.curve_segments:
        return False
    if internal_atom.line_segments < 2 or len(internal_atom.points) < 3:
        return False
    for endpoint in (endpoint_atom.points[0], endpoint_atom.points[-1]):
        branch = endpoint_segment_direction(endpoint_atom, endpoint)
        if math.hypot(*branch) <= EPS:
            continue
        for index in range(1, len(internal_atom.points) - 1):
            vertex = internal_atom.points[index]
            if distance(endpoint, vertex) > tolerance:
                continue
            before = sub(internal_atom.points[index - 1], vertex)
            after = sub(internal_atom.points[index + 1], vertex)
            if math.hypot(*before) <= EPS or math.hypot(*after) <= EPS:
                continue
            if min(
                acute_angle_degrees(branch, before),
                acute_angle_degrees(branch, after),
            ) < 12:
                continue
            return True
    return False


def junction_pair_atoms(left: Atom, right: Atom) -> bool:
    if abs(left.id - right.id) != 1 or left.closed or right.closed:
        return False
    if not same_junction_style(left, right):
        return False
    smaller_scale = max(EPS, min(left.scale, right.scale))
    if max(left.scale, right.scale) / smaller_scale > 3:
        return False
    if distance(left.center, right.center) > smaller_scale * 0.55:
        return False
    tolerance = junction_tolerance(left, right)
    forward = endpoint_hits_internal_line_vertex(left, right, tolerance)
    reverse = endpoint_hits_internal_line_vertex(right, left, tolerance)
    return forward != reverse


CANONICAL_SPLIT_MARKER_MINIMUM_TEMPLATE_SUPPORT = 8
CANONICAL_SPLIT_MARKER_MINIMUM_CARRIER_SCALE_RATIO = 2.0
CANONICAL_SPLIT_MARKER_MAXIMUM_CARRIER_SCALE_RATIO = 12.0


def _canonical_split_marker_candidate(
    candidate_id: int, carrier: Atom, marker: Atom,
) -> tuple[Candidate, float]:
    """Build one virtual Atom for a carrier and separately stroked marker.

    ``members`` and ``atom_ids`` keep the original PDF geometry auditable,
    while center/scale/fingerprint come from the logical continuous drawing.
    No source Atom is rewritten.
    """
    marker_points = (
        marker.points[:-1]
        if len(marker.points) > 1 and distance(marker.points[0], marker.points[-1]) <= 1e-3
        else marker.points
    )
    orientations: list[tuple[float, list[Point], int]] = []
    for reversed_carrier in (False, True):
        carrier_points = list(reversed(carrier.points)) if reversed_carrier else list(carrier.points)
        for marker_index, marker_point in enumerate(marker_points):
            orientations.append((
                distance(carrier_points[-1], marker_point),
                carrier_points,
                marker_index,
            ))
    gap, carrier_points, marker_index = min(orientations, key=lambda item: item[0])
    rotated_marker = (
        list(marker_points[marker_index:])
        + list(marker_points[:marker_index])
        + [marker_points[marker_index]]
    )
    points = carrier_points + rotated_marker
    samples = resample_polyline(points, 16)
    center = mean(point[0] for point in samples), mean(point[1] for point in samples)
    scale = max(EPS, 2 * max(distance(point, center) for point in samples))
    principal_direction, aspect_ratio = principal_frame(samples, center)
    virtual_atom = Atom(
        -1,
        points,
        samples,
        polyline_length(points),
        center,
        scale,
        aspect_ratio,
        principal_direction,
        False,
        carrier.curve_segments + marker.curve_segments,
        carrier.line_segments + marker.line_segments,
        carrier.paint_mode,
        carrier.line_width,
        carrier.line_cap,
        carrier.stroke_color,
    )
    virtual = make_candidate(candidate_id, [virtual_atom])
    return Candidate(
        candidate_id,
        [carrier, marker],
        [carrier.id, marker.id],
        virtual.center,
        virtual.scale,
        virtual.fingerprint,
    ), gap


def _cached_candidate(
    candidate_id: int,
    members: list[Atom],
    cache: dict[tuple[int, ...], Candidate] | None,
) -> Candidate:
    """Reuse geometry fingerprints while preserving the caller's candidate id."""
    key = tuple(sorted(atom.id for atom in members))
    cached = cache.get(key) if cache is not None else None
    if cached is None:
        cached = make_candidate(candidate_id, members)
        if cache is not None:
            cache[key] = cached
    if cached.id == candidate_id and cached.members == members:
        return cached
    return Candidate(
        candidate_id,
        members,
        list(cached.atom_ids),
        cached.center,
        cached.scale,
        cached.fingerprint,
    )


def _iter_motif_pair_neighbor_indices(atoms: list[Atom]):
    """Yield exact later-neighbor lists without retaining the whole dense graph."""
    spatial_atoms = sorted((atom.center[0], index) for index, atom in enumerate(atoms))
    spatial_x = [entry[0] for entry in spatial_atoms]
    tree = None
    if len(atoms) >= 512:
        try:
            import numpy as np
            from scipy.spatial import cKDTree
            tree = cKDTree(np.asarray([atom.center for atom in atoms], dtype=float))
        except ImportError:
            pass
    for left, atom in enumerate(atoms):
        radius = max(EPS, atom.scale) * 0.55 + MEASURE_EPS
        if tree is not None:
            possible = tree.query_ball_point(atom.center, radius)
        else:
            possible = (
                index
                for _, index in spatial_atoms[
                    bisect_left(spatial_x, atom.center[0] - radius):
                    bisect_right(spatial_x, atom.center[0] + radius)
                ]
                if abs(atoms[index].center[1] - atom.center[1]) <= radius
            )
        yield sorted(
            index
            for index in possible
            if index > left
            and distance(atom.center, atoms[index].center)
                <= max(EPS, min(atom.scale, atoms[index].scale)) * 0.55 + MEASURE_EPS
        )


def _motif_pair_neighbor_indices(atoms: list[Atom]) -> list[list[int]]:
    """Materialized compatibility helper used by exhaustive regression tests."""
    return list(_iter_motif_pair_neighbor_indices(atoms))


def _fingerprint_window_indices(
    candidates: list[Candidate],
) -> dict[tuple[int, int, int], tuple[list[float], list[tuple[float, int]]]]:
    buckets: dict[tuple[int, int, int], list[tuple[float, int]]] = {}
    for index, candidate in enumerate(candidates):
        fingerprint = candidate.fingerprint
        key = tuple(fingerprint[field] for field in ("member_count", "closed_count", "filled_count"))
        buckets.setdefault(key, []).append((fingerprint["log_scale"], index))
    result: dict[tuple[int, int, int], tuple[list[float], list[tuple[float, int]]]] = {}
    for key, entries in buckets.items():
        entries.sort()
        result[key] = ([entry[0] for entry in entries], entries)
    return result


def _matching_window_indices(
    fingerprint: dict[str, Any],
    index: dict[tuple[int, int, int], tuple[list[float], list[tuple[float, int]]]],
) -> list[int]:
    key = tuple(fingerprint[field] for field in ("member_count", "closed_count", "filled_count"))
    bucket = index.get(key)
    if bucket is None:
        return []
    scales, entries = bucket
    limit = math.log(1.3)
    start = bisect_left(scales, fingerprint["log_scale"] - limit)
    end = bisect_right(scales, fingerprint["log_scale"] + limit)
    return sorted(entry[1] for entry in entries[start:end])


def canonical_split_marker_candidates(
    atoms: list[Atom],
    candidate_cache: dict[tuple[int, ...], Candidate] | None = None,
) -> list[Candidate]:
    """Normalize strongly templated ``plain carrier`` + closed marker pairs.

    This is deliberately representation-specific: the marker must immediately
    follow a simple carrier in PDF paint order, touch its endpoint, and the
    virtual combined geometry must directly match many existing normal
    single-Atom examples of the same style.  Nearby symbols without a strong
    template family remain independent.
    """
    by_id = {atom.id: atom for atom in atoms}
    singletons = {
        atom.id: _cached_candidate(atom.id, [atom], candidate_cache)
        for atom in atoms
    }
    singleton_list = [singletons[atom.id] for atom in atoms]
    singleton_windows = _fingerprint_window_indices(singleton_list)
    proposals: list[Candidate] = []
    for marker in atoms:
        carrier = by_id.get(marker.id - 1)
        if (
            carrier is None
            or not marker.closed
            or marker.paint_mode != "stroke"
            or marker.curve_segments
            or not 3 <= marker.line_segments <= 8
            or carrier.closed
            or carrier.paint_mode != "stroke"
            or carrier.curve_segments
            or carrier.line_segments > 3
            or not same_junction_style(carrier, marker)
            or not (
                marker.scale * CANONICAL_SPLIT_MARKER_MINIMUM_CARRIER_SCALE_RATIO
                <= carrier.length
                <= marker.scale * CANONICAL_SPLIT_MARKER_MAXIMUM_CARRIER_SCALE_RATIO
            )
            or not 1.8 <= marker.length / max(marker.scale, EPS) <= 5.0
        ):
            continue
        proposal, gap = _canonical_split_marker_candidate(-1, carrier, marker)
        if gap > max(carrier.line_width * 2, marker.scale * 0.04):
            continue
        template_support = sum(
            atoms[index].id not in proposal.atom_ids
            and not atoms[index].closed
            and atoms[index].line_segments > marker.line_segments
            and same_junction_style(carrier, atoms[index])
            and fingerprints_match(proposal.fingerprint, singleton_list[index].fingerprint)
            for index in _matching_window_indices(proposal.fingerprint, singleton_windows)
        )
        if template_support >= CANONICAL_SPLIT_MARKER_MINIMUM_TEMPLATE_SUPPORT:
            proposals.append(proposal)

    owners: dict[int, int] = {}
    for proposal in proposals:
        for atom_id in proposal.atom_ids:
            owners[atom_id] = owners.get(atom_id, 0) + 1
    return [
        proposal
        for proposal in proposals
        if all(owners[atom_id] == 1 for atom_id in proposal.atom_ids)
    ]


def generate_motif_candidates(
    atoms: list[Atom],
    candidate_cache: dict[tuple[int, ...], Candidate] | None = None,
) -> list[Candidate]:
    canonical = canonical_split_marker_candidates(atoms, candidate_cache)
    canonical_atom_ids = {
        atom_id for candidate in canonical for atom_id in candidate.atom_ids
    }
    candidates: list[Candidate] = []
    for atom in atoms:
        if atom.id not in canonical_atom_ids:
            candidates.append(_cached_candidate(len(candidates), [atom], candidate_cache))
    atom_set = DisjointSet(len(atoms))
    compact_pairs: list[tuple[int, int]] = []
    junction_pairs: list[tuple[int, int]] = []
    # Both compact_atoms and junction_pair_atoms require the centers to be no
    # farther apart than 0.55 * min(scale).  Use that necessary condition as
    # an exact spatial pre-filter instead of testing every atom pair.  Nearby
    # indices are sorted back into the original order so candidate/type order
    # stays byte-for-byte deterministic.
    # On million-op CAD pages, dense coincident hatch geometry can produce
    # millions of generic composite candidates.  Keep the specialized
    # carrier+marker normalization above, but omit ambiguous generic pairs for
    # these extreme groups.  Normal groups retain byte-for-byte behavior.
    skip_generic_composites = len(atoms) >= 20_000
    if not skip_generic_composites:
        for left, right_neighbors in enumerate(_iter_motif_pair_neighbor_indices(atoms)):
            if atoms[left].id in canonical_atom_ids:
                continue
            for right in right_neighbors:
                if atoms[left].id in canonical_atom_ids or atoms[right].id in canonical_atom_ids:
                    continue
                if compact_atoms(atoms[left], atoms[right]):
                    compact_pairs.append((left, right))
                    atom_set.union(left, right)
                elif junction_pair_atoms(atoms[left], atoms[right]):
                    junction_pairs.append((left, right))
    components: dict[int, list[int]] = {}
    for index in range(len(atoms)):
        components.setdefault(atom_set.find(index), []).append(index)
    for left, right in compact_pairs + junction_pairs:
        candidates.append(_cached_candidate(
            len(candidates), [atoms[left], atoms[right]], candidate_cache,
        ))
    for component in components.values():
        if 3 <= len(component) <= 6:
            candidates.append(_cached_candidate(
                len(candidates), [atoms[index] for index in component], candidate_cache,
            ))
    for proposal in canonical:
        candidates.append(Candidate(
            len(candidates),
            proposal.members,
            proposal.atom_ids,
            proposal.center,
            proposal.scale,
            proposal.fingerprint,
        ))
    return candidates


def maximum_array_difference(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return math.inf
    return max([0.0] + [abs(value - right[index]) for index, value in enumerate(left)])


def fingerprints_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if any(left[key] != right[key] for key in ("member_count", "closed_count", "filled_count")):
        return False
    if abs(left["log_scale"] - right["log_scale"]) > math.log(1.3):
        return False
    if abs(left["aspect_ratio"] - right["aspect_ratio"]) > 0.2:
        return False
    if abs(left["normalized_length"] - right["normalized_length"]) > 0.3 * max(1, min(left["normalized_length"], right["normalized_length"])):
        return False
    return all(maximum_array_difference(left[key], right[key]) <= tolerance for key, tolerance in (
        ("member_length_ratios", 0.16),
        ("center_distances", 0.2),
        ("direction_angles", 0.2),
        ("radial_quantiles", 0.18),
    ))


def _union_matching_candidate_fingerprints(
    candidates: list[Candidate],
    sets: DisjointSet,
) -> None:
    # Exact duplicate fingerprints have identical matching behavior against
    # every other candidate.  Collapse them first, then compare one
    # representative per identity.  The remaining comparisons are bucketed
    # by the fields fingerprints_match requires to be exactly equal and are
    # windowed by its log-scale limit.  These are necessary conditions, so the
    # resulting connected components are identical to the exhaustive scan.
    identities: dict[tuple[Any, ...], list[int]] = {}
    for index, candidate in enumerate(candidates):
        fingerprint_identity = tuple(
            (key, tuple(value) if isinstance(value, list) else value)
            for key, value in sorted(candidate.fingerprint.items())
        )
        identities.setdefault(fingerprint_identity, []).append(index)

    representatives: list[int] = []
    for indices in identities.values():
        representative = indices[0]
        representatives.append(representative)
        for duplicate in indices[1:]:
            sets.union(representative, duplicate)

    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index in representatives:
        fingerprint = candidates[index].fingerprint
        bucket_key = tuple(fingerprint[key] for key in ("member_count", "closed_count", "filled_count"))
        buckets.setdefault(bucket_key, []).append(index)

    log_scale_limit = math.log(1.3)
    numeric_arrays: dict[tuple[int, str], Any] = {}
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        np = None
        cKDTree = None

    def numeric_array(index: int, key: str, values: list[float]) -> Any:
        cache_key = (index, key)
        cached = numeric_arrays.get(cache_key)
        if cached is None:
            assert np is not None
            cached = np.asarray(values, dtype=float)
            numeric_arrays[cache_key] = cached
        return cached

    def prepared_match(left_index: int, right_index: int) -> bool:
        left = candidates[left_index].fingerprint
        right = candidates[right_index].fingerprint
        if any(left[key] != right[key] for key in ("member_count", "closed_count", "filled_count")):
            return False
        if abs(left["log_scale"] - right["log_scale"]) > log_scale_limit:
            return False
        if abs(left["aspect_ratio"] - right["aspect_ratio"]) > 0.2:
            return False
        if abs(left["normalized_length"] - right["normalized_length"]) > 0.3 * max(
            1, min(left["normalized_length"], right["normalized_length"]),
        ):
            return False
        for key, tolerance in (
            ("member_length_ratios", 0.16),
            ("center_distances", 0.2),
            ("direction_angles", 0.2),
            ("radial_quantiles", 0.18),
        ):
            left_values, right_values = left[key], right[key]
            if len(left_values) != len(right_values):
                return False
            if np is not None and len(left_values) >= 64:
                left_array = numeric_array(left_index, key, left_values)
                right_array = numeric_array(right_index, key, right_values)
                if float(np.max(np.abs(left_array - right_array), initial=0.0)) > tolerance:
                    return False
            elif maximum_array_difference(left_values, right_values) > tolerance:
                return False
        return True

    for bucket in buckets.values():
        if (
            np is not None
            and cKDTree is not None
            and len(bucket) >= 512
        ):
            feature_rows: list[list[float]] = []
            for index in bucket:
                fingerprint = candidates[index].fingerprint
                feature_rows.append([
                    fingerprint["log_scale"] / log_scale_limit,
                    fingerprint["aspect_ratio"] / 0.2,
                    *(value / 0.16 for value in fingerprint["member_length_ratios"]),
                    *(value / 0.2 for value in fingerprint["center_distances"]),
                    *(value / 0.2 for value in fingerprint["direction_angles"]),
                    *(value / 0.18 for value in fingerprint["radial_quantiles"]),
                ])
            feature_matrix = np.asarray(feature_rows, dtype=float)
            normalized_lengths = np.asarray([
                candidates[index].fingerprint["normalized_length"]
                for index in bucket
            ], dtype=float)
            # A unit grid is an exact spatial decomposition for the scaled
            # Chebyshev predicate above: matching points can only occupy the
            # same or immediately adjacent cells.  Every pair in one cell
            # already satisfies all feature limits, so its 1-D length graph is
            # represented exactly by consecutive sorted edges.  This collapses
            # the very dense cells first; cross-cell searches then need only one
            # witness edge per pair of already-connected cell components.
            cell_rows, cell_ids = np.unique(
                np.floor(feature_matrix), axis=0, return_inverse=True,
            )
            cell_components: list[list[dict[str, Any]]] = []

            def lengths_match(left_length: float, right_length: float) -> bool:
                return abs(left_length - right_length) <= 0.3 * max(
                    1.0, min(left_length, right_length),
                )

            for cell_id in range(len(cell_rows)):
                locals_in_cell = np.flatnonzero(cell_ids == cell_id)
                order = locals_in_cell[np.argsort(
                    normalized_lengths[locals_in_cell], kind="stable",
                )]
                components: list[dict[str, Any]] = []
                component_start = 0
                for position in range(1, len(order) + 1):
                    if position < len(order) and lengths_match(
                        float(normalized_lengths[order[position - 1]]),
                        float(normalized_lengths[order[position]]),
                    ):
                        continue
                    component_locals = order[component_start:position]
                    representative_local = int(component_locals[0])
                    representative = bucket[representative_local]
                    for local_index in component_locals[1:]:
                        member = bucket[int(local_index)]
                        sets.union(min(representative, member), max(representative, member))
                    component_lengths = normalized_lengths[component_locals]
                    components.append({
                        "locals": component_locals,
                        "representative_local": representative_local,
                        "minimum_length": float(component_lengths[0]),
                        "maximum_length": float(component_lengths[-1]),
                        "tree": None,
                    })
                    component_start = position
                cell_components.append(components)

            adjacent_cells = cKDTree(cell_rows).query_pairs(
                1.0 + 1e-12, p=np.inf, output_type="ndarray",
            )
            for left_cell, right_cell in adjacent_cells:
                for left_component in cell_components[int(left_cell)]:
                    for right_component in cell_components[int(right_cell)]:
                        left_representative = bucket[left_component["representative_local"]]
                        right_representative = bucket[right_component["representative_local"]]
                        if sets.find(left_representative) == sets.find(right_representative):
                            continue
                        if left_component["maximum_length"] < right_component["minimum_length"]:
                            if not lengths_match(
                                left_component["maximum_length"],
                                right_component["minimum_length"],
                            ):
                                continue
                        elif right_component["maximum_length"] < left_component["minimum_length"]:
                            if not lengths_match(
                                right_component["maximum_length"],
                                left_component["minimum_length"],
                            ):
                                continue

                        query_component, indexed_component = left_component, right_component
                        if len(query_component["locals"]) > len(indexed_component["locals"]):
                            query_component, indexed_component = indexed_component, query_component
                        if indexed_component["tree"] is None:
                            indexed_component["tree"] = cKDTree(
                                feature_matrix[indexed_component["locals"]],
                            )
                        indexed_locals = indexed_component["locals"]
                        witness_found = False
                        for query_local in query_component["locals"]:
                            neighbor_positions = np.asarray(
                                indexed_component["tree"].query_ball_point(
                                    feature_matrix[int(query_local)],
                                    1.0 + 1e-12,
                                    p=np.inf,
                                ),
                                dtype=int,
                            )
                            if neighbor_positions.size == 0:
                                continue
                            neighbor_locals = indexed_locals[neighbor_positions]
                            exact_features = np.all(
                                np.abs(
                                    feature_matrix[neighbor_locals]
                                    - feature_matrix[int(query_local)]
                                ) <= 1.0,
                                axis=1,
                            )
                            query_length = normalized_lengths[int(query_local)]
                            neighbor_lengths = normalized_lengths[neighbor_locals]
                            exact_lengths = np.abs(query_length - neighbor_lengths) <= 0.3 * np.maximum(
                                1.0,
                                np.minimum(query_length, neighbor_lengths),
                            )
                            if bool(np.any(exact_features & exact_lengths)):
                                sets.union(
                                    min(left_representative, right_representative),
                                    max(left_representative, right_representative),
                                )
                                witness_found = True
                                break
                        if witness_found:
                            continue
            continue
        ordered = sorted(bucket, key=lambda index: (candidates[index].fingerprint["log_scale"], index))
        for position, left in enumerate(ordered):
            left_scale = candidates[left].fingerprint["log_scale"]
            for right in ordered[position + 1:]:
                if candidates[right].fingerprint["log_scale"] - left_scale > log_scale_limit:
                    break
                if prepared_match(left, right):
                    sets.union(min(left, right), max(left, right))


def cluster_repeated_candidates(candidates: list[Candidate], minimum_support: int = 3) -> list[list[Candidate]]:
    sets = DisjointSet(len(candidates))
    _union_matching_candidate_fingerprints(candidates, sets)
    groups: dict[int, list[Candidate]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(sets.find(index), []).append(candidate)
    return sorted((group for group in groups.values() if len(group) >= minimum_support), key=len, reverse=True)


def non_overlapping(candidates: list[Candidate]) -> bool:
    used: set[int] = set()
    for candidate in candidates:
        for atom_id in candidate.atom_ids:
            if atom_id in used:
                return False
            used.add(atom_id)
    return True


@dataclass
class Chain:
    ordered_items: list[Candidate]
    points: list[Point]
    edge_lengths: list[float]


def _chain_from_mst_edges(
    items: list[Candidate], edges: list[tuple[float, int, int]],
) -> Chain | None:
    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))
    sets = DisjointSet(len(items))
    tree: list[tuple[int, int]] = []
    for _, left, right in edges:
        if sets.union(left, right):
            tree.append((left, right))
            if len(tree) == len(items) - 1:
                break
    adjacency = [[] for _ in items]
    for left, right in tree:
        adjacency[left].append(right)
        adjacency[right].append(left)
    if max(map(len, adjacency)) > 2:
        return None
    endpoints = [index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1]
    if len(endpoints) != 2:
        return None
    endpoints.sort(key=lambda index: items[index].center)
    ordered: list[Candidate] = []
    previous, current = -1, endpoints[0]
    while current is not None:
        ordered.append(items[current])
        following = next((neighbor for neighbor in adjacency[current] if neighbor != previous), None)
        previous, current = current, following
    points = [candidate.center for candidate in ordered]
    return Chain(ordered, points, [distance(points[index - 1], points[index]) for index in range(1, len(points))])


def _delaunay_edges(items: list[Candidate]) -> list[tuple[float, int, int]] | None:
    """Return the exact Euclidean-MST candidate graph for a large 2-D set.

    Every Euclidean MST edge belongs to the Delaunay triangulation.  The
    recognizer therefore keeps the same Kruskal ordering and chain test while
    avoiding the complete graph's O(N²) edge materialization.  Coincident
    centers are first joined by the same minimum-index zero-edge star that
    complete-graph Kruskal would select, then represented by their minimum
    original index in the triangulation.  SciPy is used only for unusually
    large runtime Groups and the standard-library complete graph remains the
    deterministic fallback.
    """
    if len(items) < 512:
        return None
    try:
        import numpy as np
        from scipy.spatial import Delaunay, QhullError
    except ImportError:
        return None
    center_groups: dict[Point, list[int]] = {}
    for index, candidate in enumerate(items):
        center_groups.setdefault(candidate.center, []).append(index)
    unique_centers = list(center_groups)
    representatives = [min(center_groups[center]) for center in unique_centers]
    zero_edges = [
        (0.0, representative, index)
        for center, representative in zip(unique_centers, representatives)
        for index in center_groups[center]
        if index != representative
    ]
    if len(unique_centers) == 1:
        return zero_edges

    points = np.asarray(unique_centers, dtype=float)
    centered = points - points.mean(axis=0)
    if np.linalg.matrix_rank(centered, tol=EPS) < 2:
        axis = int(np.argmax(np.ptp(points, axis=0)))
        ordered = sorted(
            range(len(unique_centers)),
            key=lambda index: (points[index, axis], representatives[index]),
        )
        return zero_edges + [
            (
                distance(unique_centers[left], unique_centers[right]),
                min(representatives[left], representatives[right]),
                max(representatives[left], representatives[right]),
            )
            for left, right in zip(ordered, ordered[1:])
        ]
    try:
        triangulation = Delaunay(points)
    except QhullError:
        return None
    pairs: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (0, 2)):
            left_unique, right_unique = int(simplex[first]), int(simplex[second])
            left, right = sorted((representatives[left_unique], representatives[right_unique]))
            pairs.add((left, right))
    return zero_edges + [
        (distance(items[left].center, items[right].center), left, right)
        for left, right in sorted(pairs)
    ]


def mst_chain(items: list[Candidate]) -> Chain | None:
    if len(items) < 2:
        return None
    edges = _delaunay_edges(items)
    if edges is None:
        edges = [
            (distance(items[left].center, items[right].center), left, right)
            for left in range(len(items))
            for right in range(left + 1, len(items))
        ]
    return _chain_from_mst_edges(items, edges)


@dataclass
class CarrierSegment:
    start: Point
    end: Point
    direction: Point
    length: float
    arc_start: float
    index: int


@dataclass
class Projection:
    distance: float
    raw_t: float
    t: float
    point: Point
    direction: Point
    segment: CarrierSegment | None = None
    angle_degrees: float = 0.0
    side: str = ""


def project_point_to_segment(point: Point, start: Point, end: Point) -> Projection:
    delta = sub(end, start)
    squared_length = dot(delta, delta)
    raw_t = dot(sub(point, start), delta) / squared_length if squared_length > EPS else 0.0
    t = max(0.0, min(1.0, raw_t))
    projected = add(start, mul(delta, t))
    return Projection(distance(point, projected), raw_t, t, projected, normalize(delta))


def build_core_segments(points: list[Point]) -> list[CarrierSegment]:
    result: list[CarrierSegment] = []
    arc_start = 0.0
    for index in range(1, len(points)):
        length = distance(points[index - 1], points[index])
        result.append(CarrierSegment(points[index - 1], points[index], normalize(sub(points[index], points[index - 1])), length, arc_start, index - 1))
        arc_start += length
    return result


def best_projection(point: Point, segments: list[CarrierSegment], require_inside: bool = False) -> Projection | None:
    projections: list[Projection] = []
    for segment in segments:
        projection = project_point_to_segment(point, segment.start, segment.end)
        projection.segment = segment
        if not require_inside or 0 <= projection.raw_t <= 1:
            projections.append(projection)
    return min(projections, key=lambda projection: projection.distance) if projections else None


def atom_chord_direction(atom: Atom) -> Point:
    return normalize(sub(atom.points[-1], atom.points[0]))


@dataclass
class Follower:
    atom: Atom
    projection: Projection


def core_followers_for(
    atoms: list[Atom], excluded_ids: set[int], carrier_points: list[Point],
    spacing: float, motif_scale: float, line_width: float,
) -> tuple[list[Follower], float, float, list[CarrierSegment]]:
    segments = build_core_segments(carrier_points)
    corridor = max(motif_scale * 1.5, line_width * 4)
    maximum_length = spacing * 1.35
    followers: list[Follower] = []
    for atom in atoms:
        if atom.id in excluded_ids or atom.closed or atom.length > maximum_length:
            continue
        projection = best_projection(polyline_halfway_point(atom.points), segments, True)
        if projection is None or projection.distance > corridor:
            continue
        assert projection.segment is not None
        projection.angle_degrees = acute_angle_degrees(atom_chord_direction(atom), projection.segment.direction)
        if projection.angle_degrees <= 20:
            followers.append(Follower(atom, projection))
    return followers, corridor, maximum_length, segments


def orient_atom_points(atom: Atom, previous: Point, following: Point) -> list[Point]:
    return orient_points(atom.points, previous, following)


def orient_points(points: list[Point], previous: Point, following: Point) -> list[Point]:
    forward = distance(previous, points[0]) + distance(points[-1], following)
    reverse = distance(previous, points[-1]) + distance(points[0], following)
    return list(points) if forward <= reverse else list(reversed(points))


@dataclass
class RefinedCarrier:
    points: list[Point]
    motif_arc_gaps: list[float]
    start_tangent: Point
    end_tangent: Point


def refined_carrier_from(
    ordered: list[Candidate], followers: list[Follower], coarse: list[CarrierSegment],
    chord_only_follower_ids: set[int] | None = None,
) -> RefinedCarrier:
    chord_only_follower_ids = chord_only_follower_ids or set()

    def arc_position(point: Point) -> float:
        projection = best_projection(point, coarse, False)
        assert projection is not None and projection.segment is not None
        return projection.segment.arc_start + projection.t * projection.segment.length

    items: list[dict[str, Any]] = [
        {"kind": "motif", "center": candidate.center, "candidate": candidate, "arc": arc_position(candidate.center)}
        for candidate in ordered
    ] + [
        {"kind": "follower", "center": polyline_halfway_point(item.atom.points), "atom": item.atom,
         "arc": arc_position(polyline_halfway_point(item.atom.points))}
        for item in followers
    ]
    items.sort(key=lambda item: item["arc"])
    points: list[Point] = []
    motif_indices: list[int] = []
    oriented_followers: list[list[Point]] = []

    def append(point: Point) -> None:
        if not points or distance(points[-1], point) > EPS:
            points.append(point)

    for index, item in enumerate(items):
        if item["kind"] == "motif":
            append(item["center"])
            motif_indices.append(len(points) - 1)
        else:
            previous = items[index - 1]["center"] if index else item["center"]
            following = items[index + 1]["center"] if index + 1 < len(items) else item["center"]
            atom = item["atom"]
            carrier_points = (
                [atom.points[0], atom.points[-1]]
                if atom.id in chord_only_follower_ids else atom.points
            )
            oriented = orient_points(carrier_points, previous, following)
            oriented_followers.append(oriented)
            for point in oriented:
                append(point)
    gaps = [polyline_length(points[motif_indices[index - 1] : end + 1]) for index, end in enumerate(motif_indices[1:], 1)]
    start_tangent = normalize(sub(oriented_followers[0][1], oriented_followers[0][0])) if oriented_followers else normalize(sub(points[1], points[0]))
    end_tangent = normalize(sub(oriented_followers[-1][-1], oriented_followers[-1][-2])) if oriented_followers else normalize(sub(points[-1], points[-2]))
    return RefinedCarrier(points, gaps, start_tangent, end_tangent)


def endpoint_followers_for(
    atoms: list[Atom], excluded_ids: set[int], carrier: RefinedCarrier,
    spacing: float, corridor: float, maximum_length: float,
) -> list[Follower]:
    start, end = carrier.points[0], carrier.points[-1]
    extensions = [
        ("start", add(start, mul(carrier.start_tangent, -spacing * 1.1)), start, carrier.start_tangent),
        ("end", end, add(end, mul(carrier.end_tangent, spacing * 1.1)), carrier.end_tangent),
    ]
    followers: list[Follower] = []
    for atom in atoms:
        if atom.id in excluded_ids or atom.closed or atom.length > maximum_length:
            continue
        center = polyline_halfway_point(atom.points)
        matches: list[Projection] = []
        for side, extension_start, extension_end, direction in extensions:
            projection = project_point_to_segment(center, extension_start, extension_end)
            projection.side = side
            projection.angle_degrees = acute_angle_degrees(atom_chord_direction(atom), direction)
            if 0 <= projection.raw_t <= 1 and projection.distance <= corridor and projection.angle_degrees <= 15:
                matches.append(projection)
        if matches:
            followers.append(Follower(atom, min(matches, key=lambda match: match.distance)))
    return followers


def periodic_endpoint_fragment_followers(
    atoms: list[Atom], excluded_ids: set[int], chain: Chain,
    spacing: float, line_width: float, maximum_length: float,
    occupied_sides: set[str], spacing_cv: float, maximum_spacing_error: float,
) -> list[Follower]:
    """Absorb one uniquely supported incomplete drawing fragment per end.

    A boundary square or marker can be split across PDF ``S`` operations, so
    its residual fragment need not point along the carrier.  Unlike ordinary
    endpoint followers, this fallback therefore uses the fragment's position
    relative to the proven terminal motif rather than its own chord angle.
    It is restricted to long, stable chains and rejects an endpoint whenever
    more than one same-style fragment is plausible there.
    """
    if len(chain.ordered_items) < SKIPPED_PERIOD_MINIMUM_STATIONS:
        return []
    if spacing_cv > 0.12 or maximum_spacing_error > 0.20:
        return []
    endpoint_limit = max(spacing * 0.22, line_width * 4)
    result: list[Follower] = []
    used_ids: set[int] = set()
    terminal_specs = (
        ("start", chain.ordered_items[0], chain.ordered_items[1]),
        ("end", chain.ordered_items[-1], chain.ordered_items[-2]),
    )
    for side, terminal, neighbor in terminal_specs:
        if side in occupied_sides:
            continue
        outward = sub(terminal.center, neighbor.center)
        matches: list[Follower] = []
        for atom in atoms:
            if atom.id in excluded_ids or atom.id in used_ids or atom.closed or atom.length > maximum_length:
                continue
            if not same_junction_style(terminal.members[0], atom):
                continue
            center = polyline_halfway_point(atom.points)
            continuation = sub(center, terminal.center)
            center_distance = math.hypot(*continuation)
            if (
                center_distance > spacing * 1.1
                or center_distance < max(line_width * 2, spacing * 0.02)
                or directed_angle_degrees(outward, continuation) > 25
            ):
                continue
            endpoint_distance = min(
                _candidate_endpoint_distance(atom.points[0], terminal),
                _candidate_endpoint_distance(atom.points[-1], terminal),
            )
            if endpoint_distance > endpoint_limit:
                continue
            extension_end = add(terminal.center, mul(normalize(outward), spacing * 1.1))
            projection = project_point_to_segment(center, terminal.center, extension_end)
            projection.side = side
            matches.append(Follower(atom, projection))
        if len(matches) == 1:
            result.append(matches[0])
            used_ids.add(matches[0].atom.id)
    return result


def add_endpoint_followers_to_carrier(
    carrier: RefinedCarrier, followers: list[Follower],
    point_only_follower_ids: set[int] | None = None,
) -> list[Point]:
    point_only_follower_ids = point_only_follower_ids or set()
    starts = sorted((item for item in followers if item.projection.side == "start"),
                    key=lambda item: distance(polyline_halfway_point(item.atom.points), carrier.points[0]), reverse=True)
    ends = sorted((item for item in followers if item.projection.side == "end"),
                  key=lambda item: distance(polyline_halfway_point(item.atom.points), carrier.points[-1]))
    points: list[Point] = []
    for item in starts:
        if item.atom.id in point_only_follower_ids:
            points.append(polyline_halfway_point(item.atom.points))
        else:
            points.extend(orient_atom_points(item.atom, polyline_halfway_point(item.atom.points), carrier.points[0]))
    points.extend(carrier.points)
    for item in ends:
        if item.atom.id in point_only_follower_ids:
            points.append(polyline_halfway_point(item.atom.points))
        else:
            points.extend(orient_atom_points(item.atom, carrier.points[-1], polyline_halfway_point(item.atom.points)))
    return points


@dataclass
class Hypothesis:
    cluster: list[Candidate]
    motif_atom_ids: set[int]
    motif_member_count: int
    chain: Chain
    spacing: float
    spacing_cv: float
    maximum_relative_spacing_error: float
    core_followers: list[Follower]
    endpoint_followers: list[Follower]
    explained_atom_ids: set[int]
    refined_carrier: RefinedCarrier
    full_carrier: list[Point]
    refined_spacing: float
    refined_spacing_cv: float
    score: tuple[int, int, int, int]


@dataclass
class NetworkEdge:
    left_index: int
    right_index: int
    length: float


@dataclass
class OrientedFragment:
    atom: Atom
    points: list[Point]
    start: Point
    end: Point
    start_tangent: Point
    end_tangent: Point
    t_start: float
    t_end: float
    maximum_lateral: float
    reversed: bool


@dataclass
class BridgeMatch:
    fragments: tuple[OrientedFragment, ...]
    left_index: int
    right_index: int
    left_endpoint: Point
    right_endpoint: Point
    route_length: float
    ink_length: float
    total_gap: float
    maximum_lateral: float
    occupancy_signature: tuple[int, ...]
    fit_cost: float

    @property
    def atoms(self) -> tuple[Atom, ...]:
        return tuple(fragment.atom for fragment in self.fragments)

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return tuple(fragment.atom.id for fragment in self.fragments)


@dataclass
class NetworkHypothesis:
    """A repeated motif supported by a carrier graph, not one global chain."""

    cluster: list[Candidate]
    motif_atom_ids: set[int]
    motif_member_count: int
    station_count: int
    spacing: float
    spacing_cv: float
    maximum_relative_spacing_error: float
    network_edges: list[NetworkEdge]
    bridge_followers: list[BridgeMatch]
    endpoint_followers: list[Atom]
    explained_atom_ids: set[int]
    score: tuple[int, int, int, int]
    relaxed_terminal_edge_count: int = 0


@dataclass
class TwoInstanceExtension:
    """One uniquely validated outward continuation of a two-instance motif."""

    atom: Atom
    points: list[Point]
    center_endpoint: Point
    gap: float
    route_length: float
    angle_degrees: float


@dataclass
class TwoInstanceHypothesis:
    """High-precision fallback for one complete motif occurring exactly twice.

    The normal >=3-instance discovery path remains preferred; discovery only
    compares this guarded result with an otherwise empty round or with the
    weakest three-plain-line interpretation that it exactly subsumes.
    """

    cluster: list[Candidate]
    motif_atom_ids: set[int]
    motif_member_count: int
    spacing: float
    middle_bridge: BridgeMatch
    left_extension: TwoInstanceExtension
    right_extension: TwoInstanceExtension
    explained_atom_ids: set[int]
    score: tuple[int, int, int, int]


@dataclass
class SharedPathHypothesis:
    """A complete line type assembled from parts sharing one reference path.

    ``attached_repeat`` covers a real long carrier with repeated shapes attached
    to it.  ``ink_gap_period`` covers disconnected ink fragments whose ordered
    lengths and gaps repeat along one or more logical paths.  The same record
    also carries guarded self-contained repeat units, double-dot periods and a
    complete one-period command block at a Group boundary.
    """

    relation_kind: str
    instances: list[list[int]]
    repeated_atom_ids: set[int]
    reference_atom_ids: set[int]
    explained_atom_ids: set[int]
    period_length: float | None
    period_signature: tuple[tuple[float, float], ...] | None
    support_count: int
    fit_error: float
    score: tuple[int, int, int, int]
    # Logical motif stations, kept separately from ``instances`` (physical
    # carrier components).  This is needed when one PDF Atom contains both
    # the carrier dash and its attached marker: the repeated unit is the Atom
    # itself, rather than a follower attached to some other reference path.
    motif_instances: list[list[int]] | None = None


SKIPPED_PERIOD_MINIMUM_STATIONS = 8
SKIPPED_PERIOD_MAXIMUM_EDGES = 2
SKIPPED_PERIOD_MAXIMUM_MISSING_STATIONS = 2
SKIPPED_PERIOD_MAXIMUM_MULTIPLIER = 3
SKIPPED_PERIOD_MAXIMUM_UNIT_ERROR = 0.12


def _period_edge_multipliers(edge_lengths: list[float], spacing: float) -> list[int] | None:
    """Recognize a rare integer-period gap inside an otherwise stable chain.

    A long center-to-center edge is not accepted by geometry alone.  This
    helper only proposes the integer multiple; ``_skipped_period_bridges``
    must subsequently find unique painted ink connecting both neighboring
    motif candidates.
    """
    multipliers: list[int] = []
    for length in edge_lengths:
        ratio = length / max(spacing, EPS)
        multiplier = max(1, int(math.floor(ratio + 0.5))) if ratio >= 1.5 else 1
        if multiplier > SKIPPED_PERIOD_MAXIMUM_MULTIPLIER:
            return None
        multipliers.append(multiplier)
    skipped = [value for value in multipliers if value > 1]
    if not skipped:
        return multipliers
    if (
        len(edge_lengths) + 1 < SKIPPED_PERIOD_MINIMUM_STATIONS
        or len(skipped) > SKIPPED_PERIOD_MAXIMUM_EDGES
        or sum(value - 1 for value in skipped) > SKIPPED_PERIOD_MAXIMUM_MISSING_STATIONS
        or sum(value == 1 for value in multipliers) < math.ceil(len(multipliers) * 0.8)
    ):
        return None
    for length, multiplier in zip(edge_lengths, multipliers):
        if multiplier > 1 and abs(length / multiplier - spacing) / spacing > SKIPPED_PERIOD_MAXIMUM_UNIT_ERROR:
            return None
    return multipliers


def _candidate_endpoint_distance(point: Point, candidate: Candidate) -> float:
    return min(
        distance(point, endpoint)
        for atom in candidate.members
        for endpoint in (atom.points[0], atom.points[-1])
    )


def _skipped_period_bridges(
    chain: Chain, atoms: list[Atom], motif_ids: set[int], multipliers: list[int],
    spacing: float, motif_scale: float, line_width: float,
    coarse_segments: list[CarrierSegment],
) -> list[Follower] | None:
    """Find unique real ink for every proposed multi-period chain edge."""
    endpoint_limit = max(spacing * 0.22, line_width * 4)
    lateral_limit = max(motif_scale * 0.35, line_width * 4)
    result: list[Follower] = []
    used_ids: set[int] = set()
    for edge_index, multiplier in enumerate(multipliers):
        if multiplier == 1:
            continue
        left = chain.ordered_items[edge_index]
        right = chain.ordered_items[edge_index + 1]
        edge_length = chain.edge_lengths[edge_index]
        edge_direction = sub(right.center, left.center)
        matches: list[Follower] = []
        for atom in atoms:
            if atom.id in motif_ids or atom.id in used_ids or atom.closed:
                continue
            if (
                not same_junction_style(left.members[0], atom)
                or not same_junction_style(right.members[0], atom)
                or atom.length > edge_length * 1.05
                or distance(atom.points[0], atom.points[-1]) < edge_length * 0.4
                or acute_angle_degrees(atom_chord_direction(atom), edge_direction) > 20
            ):
                continue
            forward = max(
                _candidate_endpoint_distance(atom.points[0], left),
                _candidate_endpoint_distance(atom.points[-1], right),
            )
            reverse = max(
                _candidate_endpoint_distance(atom.points[-1], left),
                _candidate_endpoint_distance(atom.points[0], right),
            )
            if min(forward, reverse) > endpoint_limit:
                continue
            projection = best_projection(polyline_halfway_point(atom.points), coarse_segments, True)
            if projection is None or projection.distance > lateral_limit:
                continue
            assert projection.segment is not None
            projection.angle_degrees = acute_angle_degrees(atom_chord_direction(atom), projection.segment.direction)
            matches.append(Follower(atom, projection))
        # Ambiguous or absent bridge ink is evidence against skipping a period.
        if len(matches) != 1:
            return None
        result.append(matches[0])
        used_ids.add(matches[0].atom.id)
    return result


def build_hypothesis(cluster: list[Candidate], atoms: list[Atom]) -> Hypothesis | None:
    if not non_overlapping(cluster) or (chain := mst_chain(cluster)) is None:
        return None
    motif_scale = median(candidate.scale for candidate in cluster)
    spacing = median(chain.edge_lengths)
    if spacing <= motif_scale * 0.5:
        return None
    multipliers = _period_edge_multipliers(chain.edge_lengths, spacing)
    if multipliers is None:
        return None
    unit_lengths = [length / multiplier for length, multiplier in zip(chain.edge_lengths, multipliers)]
    maximum_error = max(abs(length - spacing) / spacing for length in unit_lengths)
    if maximum_error > 0.4:
        return None
    spacing_cv = population_std(unit_lengths) / max(mean(unit_lengths), EPS)
    motif_ids = {atom_id for candidate in cluster for atom_id in candidate.atom_ids}
    line_width = median((atom.line_width or 1) for candidate in cluster for atom in candidate.members)
    core, corridor, maximum_length, coarse_segments = core_followers_for(
        atoms, motif_ids, chain.points, spacing, motif_scale, line_width,
    )
    skipped_bridges = _skipped_period_bridges(
        chain, atoms, motif_ids | {item.atom.id for item in core}, multipliers,
        spacing, motif_scale, line_width, coarse_segments,
    )
    if skipped_bridges is None:
        return None
    core.extend(skipped_bridges)
    chord_only_ids = {item.atom.id for item in skipped_bridges}
    refined = refined_carrier_from(
        chain.ordered_items, core, coarse_segments, chord_only_ids,
    )
    core_ids = motif_ids | {item.atom.id for item in core}
    endpoints = endpoint_followers_for(atoms, core_ids, refined, spacing, corridor, maximum_length)
    boundary_fragments = (
        periodic_endpoint_fragment_followers(
            atoms,
            core_ids | {item.atom.id for item in endpoints},
            chain,
            spacing,
            line_width,
            maximum_length,
            {item.projection.side for item in endpoints},
            spacing_cv,
            maximum_error,
        )
        # Direction-free endpoint completion is intentionally coupled to the
        # already exceptional, ink-verified skipped-period case.  Ordinary
        # chains keep the stricter chord-aligned endpoint rule above.
        if skipped_bridges else []
    )
    endpoints.extend(boundary_fragments)
    explained = core_ids | {item.atom.id for item in endpoints}
    full_carrier = add_endpoint_followers_to_carrier(
        refined, endpoints, {item.atom.id for item in boundary_fragments},
    )
    refined_period_gaps = [
        gap / multiplier for gap, multiplier in zip(refined.motif_arc_gaps, multipliers)
    ]
    refined_spacing = median(refined_period_gaps)
    refined_cv = (
        population_std(refined_period_gaps) / max(mean(refined_period_gaps), EPS)
        if len(refined_period_gaps) > 1 else spacing_cv
    )
    score = (len(explained), len(cluster), -js_round_nonnegative(refined_cv * 10_000), -cluster[0].fingerprint["member_count"])
    return Hypothesis(
        cluster, motif_ids, cluster[0].fingerprint["member_count"], chain,
        spacing, spacing_cv, maximum_error, core, endpoints, explained,
        refined, full_carrier, refined_spacing, refined_cv, score,
    )


def dominant_nearest_spacing(cluster: list[Candidate]) -> tuple[float, list[float]] | None:
    """Estimate one local period without forcing all motif stations into one MST."""
    if len(cluster) < 3:
        return None
    nearest = sorted(_nearest_candidate_center_distances(cluster))
    modes: list[list[float]] = []
    for value in nearest:
        choices: list[tuple[float, list[float]]] = []
        for mode in modes:
            center = median(mode)
            error = abs(value - center) / max(center, EPS)
            if error <= 0.18:
                choices.append((error, mode))
        if choices:
            min(choices, key=lambda item: item[0])[1].append(value)
        else:
            modes.append([value])
    mode = max(
        modes,
        key=lambda values: (len(values), -population_std(values) / max(mean(values), EPS)),
    )
    if len(mode) < max(3, math.ceil(len(cluster) * 0.5)):
        return None
    return median(mode), mode


def endpoint_segment_direction(atom: Atom, endpoint: Point) -> Point:
    """Direction of the first non-zero primitive segment touching an endpoint."""
    if distance(endpoint, atom.points[0]) <= distance(endpoint, atom.points[-1]):
        origin = atom.points[0]
        points = atom.points[1:]
    else:
        origin = atom.points[-1]
        points = reversed(atom.points[:-1])
    for point in points:
        if distance(origin, point) > EPS:
            return sub(point, origin)
    return 0.0, 0.0


def oriented_fragments_for_edge(
    atom: Atom,
    left_center: Point,
    right_center: Point,
    spacing: float,
    maximum_length: float,
) -> list[OrientedFragment]:
    """Return the orientations that move monotonically from the left station to the right."""
    if atom.closed or atom.length < spacing * 0.02 or atom.length > maximum_length:
        return []
    chord = distance(atom.points[0], atom.points[-1])
    if atom.length > EPS and chord / atom.length < 0.45:
        return []
    result: list[OrientedFragment] = []
    for reverse in (False, True):
        points = list(reversed(atom.points)) if reverse else list(atom.points)
        progress = [
            project_point_to_segment(point, left_center, right_center).raw_t
            for point in points
        ]
        if progress[-1] <= progress[0] + EPS:
            continue
        if min(progress) < -0.12 or max(progress) > 1.12:
            continue
        if any(progress[index] + 0.05 < progress[index - 1] for index in range(1, len(progress))):
            continue
        maximum_lateral = max(
            lateral_distance_to_line(point, left_center, right_center)
            for point in points
        )
        if maximum_lateral > spacing * 0.30:
            continue
        result.append(OrientedFragment(
            atom=atom,
            points=points,
            start=points[0],
            end=points[-1],
            start_tangent=first_polyline_tangent(points),
            end_tangent=last_polyline_tangent(points),
            t_start=progress[0],
            t_end=progress[-1],
            maximum_lateral=maximum_lateral,
            reversed=reverse,
        ))
    return result


def bridge_occupancy_signature(
    fragments: tuple[OrientedFragment, ...],
    left_center: Point,
    right_center: Point,
    bins: int = 32,
) -> tuple[int, ...]:
    """Encode the ink/gap rhythm along one bridge, normalized by its route length."""
    cursor = distance(left_center, fragments[0].start)
    ink_ranges: list[tuple[float, float]] = []
    for index, fragment in enumerate(fragments):
        ink_ranges.append((cursor, cursor + fragment.atom.length))
        cursor += fragment.atom.length
        if index + 1 < len(fragments):
            cursor += distance(fragment.end, fragments[index + 1].start)
    route_length = cursor + distance(fragments[-1].end, right_center)
    if route_length <= EPS:
        return (0,) * bins
    return tuple(
        int(any(start <= route_length * (index + 0.5) / bins <= end for start, end in ink_ranges))
        for index in range(bins)
    )


def make_bridge_match(
    fragments: list[OrientedFragment],
    edge: NetworkEdge,
    left_center: Point,
    right_center: Point,
    spacing: float,
) -> BridgeMatch:
    frozen = tuple(fragments)
    gaps = [distance(left_center, frozen[0].start)]
    gaps.extend(distance(frozen[index - 1].end, frozen[index].start) for index in range(1, len(frozen)))
    gaps.append(distance(frozen[-1].end, right_center))
    ink_length = sum(fragment.atom.length for fragment in frozen)
    total_gap = sum(gaps)
    route_length = ink_length + total_gap
    maximum_lateral = max(fragment.maximum_lateral for fragment in frozen)
    turns = [
        directed_angle_degrees(frozen[index - 1].end_tangent, frozen[index].start_tangent)
        for index in range(1, len(frozen))
    ]
    fit_cost = (
        abs(route_length / spacing - 1)
        + 0.25 * total_gap / spacing
        + 0.25 * maximum_lateral / spacing
        + 0.10 * mean(turns) / 100
    )
    return BridgeMatch(
        frozen, edge.left_index, edge.right_index,
        frozen[0].start, frozen[-1].end,
        route_length, ink_length, total_gap, maximum_lateral,
        bridge_occupancy_signature(frozen, left_center, right_center), fit_cost,
    )


def find_multi_atom_bridge(
    edge: NetworkEdge,
    cluster: list[Candidate],
    atoms: list[Atom],
    excluded_ids: set[int],
    spacing: float,
) -> BridgeMatch | None:
    """Search a bounded, unambiguous 2..8-Atom path across one empty station edge."""
    left_center = cluster[edge.left_index].center
    right_center = cluster[edge.right_index].center
    maximum_gap = spacing * 0.12
    maximum_length = spacing * 0.45
    fragments = [
        fragment
        for atom in atoms
        if atom.id not in excluded_ids
        and not atom.closed
        and spacing * 0.02 <= atom.length <= maximum_length
        and -0.12 <= project_point_to_segment(atom.center, left_center, right_center).raw_t <= 1.12
        and lateral_distance_to_line(atom.center, left_center, right_center) <= spacing * 0.30
        for fragment in oriented_fragments_for_edge(
            atom, left_center, right_center, spacing, maximum_length,
        )
    ]
    starts = [fragment for fragment in fragments if distance(left_center, fragment.start) <= maximum_gap]
    states: list[tuple[tuple[OrientedFragment, ...], frozenset[int], float]] = [
        ((fragment,), frozenset((fragment.atom.id,)), distance(left_center, fragment.start))
        for fragment in starts
    ]
    completed: dict[frozenset[int], BridgeMatch] = {}
    for depth in range(1, 9):
        following_states: list[tuple[tuple[OrientedFragment, ...], frozenset[int], float]] = []
        for path, used_ids, accumulated_gap in states:
            last = path[-1]
            sink_gap = distance(last.end, right_center)
            if depth >= 2 and sink_gap <= maximum_gap:
                match = make_bridge_match(list(path), edge, left_center, right_center, spacing)
                internal_turns = [
                    directed_angle_degrees(path[index - 1].end_tangent, path[index].start_tangent)
                    for index in range(1, len(path))
                ]
                individual_gaps = [distance(left_center, path[0].start), sink_gap]
                individual_gaps.extend(
                    distance(path[index - 1].end, path[index].start)
                    for index in range(1, len(path))
                )
                if (
                    0.75 <= match.route_length / spacing <= 1.30
                    and match.ink_length / spacing >= 0.45
                    and match.total_gap / spacing <= 0.50
                    and max(individual_gaps) <= maximum_gap
                    and max(internal_turns, default=0.0) <= 100
                    and match.fit_cost <= 0.35
                ):
                    key = frozenset(match.atom_ids)
                    if key not in completed or match.fit_cost < completed[key].fit_cost:
                        completed[key] = match
            if depth == 8:
                continue
            for fragment in fragments:
                if fragment.atom.id in used_ids:
                    continue
                gap = distance(last.end, fragment.start)
                if gap > maximum_gap or fragment.t_start < last.t_end - 0.03:
                    continue
                if directed_angle_degrees(last.end_tangent, fragment.start_tangent) > 100:
                    continue
                if not same_junction_style(path[0].atom, fragment.atom):
                    continue
                following_states.append((
                    path + (fragment,),
                    used_ids | {fragment.atom.id},
                    accumulated_gap + gap,
                ))
        following_states.sort(key=lambda item: (item[2], -item[0][-1].t_end, item[0][-1].atom.id))
        states = following_states[:64]
        if not states:
            break
    ranked = sorted(completed.values(), key=lambda match: (match.fit_cost, match.atom_ids))
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[1].fit_cost - ranked[0].fit_cost < 0.08:
        return None
    return ranked[0]


def occupancy_iou(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    intersection = sum(a and b for a, b in zip(left, right))
    union = sum(a or b for a, b in zip(left, right))
    return intersection / union if union else 1.0


def bridge_profiles_match(left: BridgeMatch, right: BridgeMatch, spacing: float) -> bool:
    """Validate a two-station satellite against bridges learned from a >=3-station seed."""
    if len(left.fragments) != len(right.fragments):
        return False
    if abs(left.route_length - right.route_length) / spacing > 0.15:
        return False
    if abs(left.ink_length - right.ink_length) / spacing > 0.15:
        return False
    direct = occupancy_iou(left.occupancy_signature, right.occupancy_signature)
    reversed_iou = occupancy_iou(left.occupancy_signature, tuple(reversed(right.occupancy_signature)))
    return max(direct, reversed_iou) >= 0.75


def extend_terminal_bridge(
    center: Point,
    neighbor_center: Point,
    anchor: Atom,
    anchor_endpoint: Point,
    atoms: list[Atom],
    excluded_ids: set[int],
    spacing: float,
) -> list[Atom]:
    """Continue a validated terminal bridge through a unique chain of short fragments."""
    anchor_points = (
        list(anchor.points)
        if distance(anchor_endpoint, anchor.points[0]) <= distance(anchor_endpoint, anchor.points[-1])
        else list(reversed(anchor.points))
    )
    # Use the validated anchor's local tangent.  At a corner the chord to the
    # neighboring station can differ substantially from the true continuation.
    outward = normalize(last_polyline_tangent(anchor_points))
    if math.hypot(*outward) <= EPS:
        outward = normalize(sub(center, neighbor_center))
    current_atom = anchor
    current_end = anchor_points[-1]
    current_tangent = last_polyline_tangent(anchor_points)
    travelled = distance(center, anchor_points[0]) + anchor.length
    maximum_gap = spacing * 0.12
    extensions: list[Atom] = []
    while len(extensions) < 7:
        options: list[tuple[float, float, float, int, Atom, list[Point]]] = []
        for atom in atoms:
            if (
                atom.id in excluded_ids
                or atom.closed
                or atom.length > spacing * 0.45
                or not same_junction_style(anchor, atom)
            ):
                continue
            chord = distance(atom.points[0], atom.points[-1])
            if atom.length > EPS and chord / atom.length < 0.45:
                continue
            for reverse in (False, True):
                points = list(reversed(atom.points)) if reverse else list(atom.points)
                gap = distance(current_end, points[0])
                if gap > maximum_gap or travelled + gap + atom.length > spacing * 1.30:
                    continue
                progress = [dot(sub(point, center), outward) / spacing for point in points]
                if progress[-1] <= progress[0] + EPS:
                    continue
                if min(progress) < -0.12 or max(progress) > 1.30:
                    continue
                if any(progress[index] + 0.05 < progress[index - 1] for index in range(1, len(progress))):
                    continue
                maximum_lateral = max(
                    abs(sub(point, center)[0] * outward[1] - sub(point, center)[1] * outward[0])
                    for point in points
                )
                if maximum_lateral > spacing * 0.30:
                    continue
                turn = directed_angle_degrees(current_tangent, first_polyline_tangent(points))
                if turn > 100:
                    continue
                source_distance = abs(atom.id - current_atom.id)
                cost = gap / spacing + 0.10 * turn / 100 + 0.001 * min(source_distance, 10)
                options.append((cost, gap, turn, atom.id, atom, points))
        best_by_atom: dict[int, tuple[float, float, float, int, Atom, list[Point]]] = {}
        for option in options:
            atom_id = option[3]
            if atom_id not in best_by_atom or option[:4] < best_by_atom[atom_id][:4]:
                best_by_atom[atom_id] = option
        ranked = sorted(best_by_atom.values(), key=lambda option: option[:4])
        if not ranked:
            break
        if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.025:
            break
        _, gap, _, _, atom, points = ranked[0]
        extensions.append(atom)
        excluded_ids.add(atom.id)
        travelled += gap + atom.length
        current_atom = atom
        current_end = points[-1]
        current_tangent = last_polyline_tangent(points)
    return extensions


def _network_bridge_edge_indices(
    cluster: list[Candidate],
    edges: list[NetworkEdge],
    atoms: list[Atom],
    maximum_endpoint_gap: float,
) -> list[list[int]]:
    """Return exactly the station edges each Atom's endpoints can reach."""
    if not edges or not atoms:
        return [[] for _ in atoms]
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        return [
            [
                edge_index
                for edge_index, edge in enumerate(edges)
                if min(
                    max(
                        distance(atom.points[0], cluster[edge.left_index].center),
                        distance(atom.points[-1], cluster[edge.right_index].center),
                    ),
                    max(
                        distance(atom.points[-1], cluster[edge.left_index].center),
                        distance(atom.points[0], cluster[edge.right_index].center),
                    ),
                ) <= maximum_endpoint_gap
            ]
            for atom in atoms
        ]
    centers = np.asarray([candidate.center for candidate in cluster], dtype=float)
    tree = cKDTree(centers)
    queries = np.asarray([
        endpoint
        for atom in atoms
        for endpoint in (atom.points[0], atom.points[-1])
    ], dtype=float)
    nearby = tree.query_ball_point(queries, maximum_endpoint_gap + MEASURE_EPS)
    edges_by_node: dict[int, list[int]] = {}
    for edge_index, edge in enumerate(edges):
        edges_by_node.setdefault(edge.left_index, []).append(edge_index)
        edges_by_node.setdefault(edge.right_index, []).append(edge_index)
    result: list[list[int]] = []
    for atom_index, atom in enumerate(atoms):
        start_nodes = nearby[atom_index * 2]
        end_nodes = set(nearby[atom_index * 2 + 1])
        possible = {
            edge_index
            for node in start_nodes
            for edge_index in edges_by_node.get(node, ())
            if (
                edges[edge_index].right_index if edges[edge_index].left_index == node
                else edges[edge_index].left_index
            ) in end_nodes
        }
        result.append([
            edge_index
            for edge_index in sorted(possible)
            if min(
                max(
                    distance(atom.points[0], cluster[edges[edge_index].left_index].center),
                    distance(atom.points[-1], cluster[edges[edge_index].right_index].center),
                ),
                max(
                    distance(atom.points[-1], cluster[edges[edge_index].left_index].center),
                    distance(atom.points[0], cluster[edges[edge_index].right_index].center),
                ),
            ) <= maximum_endpoint_gap
        ])
    return result


def build_network_hypothesis(cluster: list[Candidate], atoms: list[Atom]) -> NetworkHypothesis | None:
    """Fallback for one motif class occurring on several straight/curved carriers.

    A robust local period creates a graph of adjacent motif stations.  Open
    atoms are then fitted as paths *between station endpoints*, so an L-shaped
    or finely segmented curve is judged by its real path rather than by one
    first-to-last chord.
    """
    if not non_overlapping(cluster) or (spacing_fit := dominant_nearest_spacing(cluster)) is None:
        return None
    spacing, spacing_samples = spacing_fit
    motif_scale = median(candidate.scale for candidate in cluster)
    if spacing <= EPS:
        return None
    spacing_cv = population_std(spacing_samples) / max(mean(spacing_samples), EPS)
    maximum_error = max(abs(value - spacing) / spacing for value in spacing_samples)
    if spacing <= motif_scale * 0.5 or spacing_cv > 0.12 or maximum_error > 0.25:
        return None

    eligible_edges: list[NetworkEdge] = []
    spatial_cluster = sorted((candidate.center[0], index) for index, candidate in enumerate(cluster))
    spatial_cluster_x = [entry[0] for entry in spatial_cluster]
    maximum_edge_length = spacing * 1.45
    for left in range(len(cluster)):
        nearby = sorted(
            index
            for _, index in spatial_cluster[
                bisect_left(spatial_cluster_x, cluster[left].center[0] - maximum_edge_length):
                bisect_right(spatial_cluster_x, cluster[left].center[0] + maximum_edge_length)
            ]
            if index > left
            and abs(cluster[index].center[1] - cluster[left].center[1]) <= maximum_edge_length
        )
        for right in nearby:
            length = distance(cluster[left].center, cluster[right].center)
            # A corner shortens Euclidean distance relative to arc distance;
            # actual carrier ink must validate every edge retained below.
            if spacing * 0.5 <= length <= spacing * 1.45:
                eligible_edges.append(NetworkEdge(left, right, length))
    eligible_nodes = {
        index
        for edge in eligible_edges
        for index in (edge.left_index, edge.right_index)
    }
    if len(eligible_nodes) < 3:
        return None

    all_cluster_atom_ids = {atom_id for candidate in cluster for atom_id in candidate.atom_ids}
    maximum_atom_length = spacing * 1.4
    maximum_endpoint_gap = spacing * 0.36
    candidate_bridges: list[BridgeMatch] = []
    possible_bridge_atoms = [
        atom
        for atom in atoms
        if atom.id not in all_cluster_atom_ids
        and not atom.closed
        and atom.length <= maximum_atom_length
        and not (
            atom.length > EPS
            and distance(atom.points[0], atom.points[-1]) / atom.length < 0.45
        )
    ]
    possible_edges = _network_bridge_edge_indices(
        cluster, eligible_edges, possible_bridge_atoms, maximum_endpoint_gap,
    )

    for atom, edge_indices in zip(possible_bridge_atoms, possible_edges):
        best: tuple[float, BridgeMatch] | None = None
        for edge_index in edge_indices:
            edge = eligible_edges[edge_index]
            left_center = cluster[edge.left_index].center
            right_center = cluster[edge.right_index].center
            for reverse in (False, True):
                left_endpoint, right_endpoint = (
                    (atom.points[-1], atom.points[0]) if reverse else (atom.points[0], atom.points[-1])
                )
                left_gap = distance(left_center, left_endpoint)
                right_gap = distance(right_center, right_endpoint)
                route_length = left_gap + atom.length + right_gap
                if max(left_gap, right_gap) > maximum_endpoint_gap:
                    continue
                if not spacing * 0.5 <= route_length <= spacing * 1.8:
                    continue
                oriented_points = list(reversed(atom.points)) if reverse else atom.points
                progress = [
                    project_point_to_segment(point, left_center, right_center).raw_t
                    for point in oriented_points
                ]
                if any(progress[index] + 0.05 < progress[index - 1] for index in range(1, len(progress))):
                    continue
                cost = abs(route_length / spacing - 1) + 0.2 * (left_gap + right_gap) / spacing
                maximum_lateral = max(
                    lateral_distance_to_line(point, left_center, right_center)
                    for point in oriented_points
                )
                fragment = OrientedFragment(
                    atom, list(oriented_points), left_endpoint, right_endpoint,
                    first_polyline_tangent(oriented_points), last_polyline_tangent(oriented_points),
                    progress[0], progress[-1], maximum_lateral, reverse,
                )
                match = BridgeMatch(
                    (fragment,), edge.left_index, edge.right_index,
                    left_endpoint, right_endpoint, route_length, atom.length,
                    left_gap + right_gap, maximum_lateral,
                    bridge_occupancy_signature((fragment,), left_center, right_center), cost,
                )
                if best is None or cost < best[0]:
                    best = cost, match
        if best is not None:
            candidate_bridges.append(best[1])

    # At most one atom owns a station interval.  This prevents a nearby
    # detouring annotation from being absorbed alongside the real bridge.
    best_by_edge: dict[tuple[int, int], BridgeMatch] = {}
    for match in candidate_bridges:
        key = match.left_index, match.right_index
        if key not in best_by_edge or match.fit_cost < best_by_edge[key].fit_cost:
            best_by_edge[key] = match

    # Freeze the proven single-Atom assignments.  True multi-Atom paths are
    # searched only for station intervals that still have no carrier ink.  A
    # station already belonging to a >=3-node single-bridge component is also
    # protected from extra edges; this keeps the established network topology
    # unchanged while still allowing mixed or wholly segmented carriers.
    single_adjacency: dict[int, set[int]] = {}
    for match in best_by_edge.values():
        single_adjacency.setdefault(match.left_index, set()).add(match.right_index)
        single_adjacency.setdefault(match.right_index, set()).add(match.left_index)
    protected_nodes: set[int] = set()
    seen_single: set[int] = set()
    for start in single_adjacency:
        if start in seen_single:
            continue
        component: set[int] = set()
        stack = [start]
        seen_single.add(start)
        while stack:
            current = stack.pop()
            component.add(current)
            for following in single_adjacency[current]:
                if following not in seen_single:
                    seen_single.add(following)
                    stack.append(following)
        if len(component) >= 3:
            protected_nodes.update(component)
    frozen_ids = all_cluster_atom_ids | {
        atom_id for match in best_by_edge.values() for atom_id in match.atom_ids
    }
    multi_candidates: list[BridgeMatch] = []
    # If a legacy single-bridge component already establishes the pattern,
    # return that topology unchanged.  Expanded search is a fallback for a
    # graph that the legacy builder could not establish at all.
    if not protected_nodes:
        for edge in eligible_edges:
            key = edge.left_index, edge.right_index
            if key in best_by_edge:
                continue
            match = find_multi_atom_bridge(edge, cluster, atoms, frozen_ids, spacing)
            if match is not None:
                multi_candidates.append(match)
    occupied_ids = set(frozen_ids)
    for match in sorted(multi_candidates, key=lambda item: (item.fit_cost, item.atom_ids)):
        key = match.left_index, match.right_index
        if key in best_by_edge or any(atom_id in occupied_ids for atom_id in match.atom_ids):
            continue
        best_by_edge[key] = match
        occupied_ids.update(match.atom_ids)

    bridges = list(best_by_edge.values())
    edge_lookup = {(edge.left_index, edge.right_index): edge for edge in eligible_edges}
    edges = [edge_lookup[key] for key in best_by_edge]

    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        adjacency.setdefault(edge.left_index, set()).add(edge.right_index)
        adjacency.setdefault(edge.right_index, set()).add(edge.left_index)
    components: list[set[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        component: set[int] = set()
        stack = [start]
        visited.add(start)
        while stack:
            current = stack.pop()
            component.add(current)
            for following in adjacency[current]:
                if following not in visited:
                    visited.add(following)
                    stack.append(following)
        components.append(component)

    seed_components = [component for component in components if len(component) >= 3]
    valid_nodes = {index for component in seed_components for index in component}
    seed_bridges = [
        match
        for match in bridges
        if any(match.left_index in component and match.right_index in component for component in seed_components)
    ]
    # A two-station instance cannot establish a pattern by itself.  It may be
    # attached as a satellite only when its bridge rhythm matches at least two
    # already validated seed bridges.
    bridge_by_key = {(match.left_index, match.right_index): match for match in bridges}
    if any(len(match.fragments) > 1 for match in bridges):
        for component in components:
            if len(component) != 2:
                continue
            left, right = sorted(component)
            satellite = bridge_by_key.get((left, right))
            if satellite is None:
                continue
            support = sum(bridge_profiles_match(satellite, seed, spacing) for seed in seed_bridges)
            if support >= 2:
                valid_nodes.update(component)
    if len(valid_nodes) < 3:
        return None

    edges = [
        edge for edge in edges
        if edge.left_index in valid_nodes and edge.right_index in valid_nodes
    ]
    valid_edge_keys = {(edge.left_index, edge.right_index) for edge in edges}
    bridges = [
        match for match in bridges
        if (match.left_index, match.right_index) in valid_edge_keys
    ]
    involved = {index for edge in edges for index in (edge.left_index, edge.right_index)}
    adjacency = {index: set() for index in involved}
    for edge in edges:
        adjacency[edge.left_index].add(edge.right_index)
        adjacency[edge.right_index].add(edge.left_index)

    motif_ids = {
        atom_id
        for index in involved
        for atom_id in cluster[index].atom_ids
    }
    explained = set(motif_ids) | {
        atom_id for match in bridges for atom_id in match.atom_ids
    }

    # A terminal atom may occupy one free carrier port.  Internal stations do
    # not have a free port and therefore cannot collect arbitrary nearby ink.
    endpoint_options: list[tuple[float, int, float, int, Atom, Point]] = []
    graph_endpoints = {index for index, neighbors in adjacency.items() if len(neighbors) == 1}
    for atom in atoms:
        if (
            atom.id in all_cluster_atom_ids
            or atom.id in explained
            or atom.closed
            or atom.length > maximum_atom_length
        ):
            continue
        for index in graph_endpoints:
            center = cluster[index].center
            neighbor = next(iter(adjacency[index]))
            inward = normalize(sub(cluster[neighbor].center, center))
            for endpoint in (atom.points[0], atom.points[-1]):
                gap = distance(center, endpoint)
                if gap > maximum_endpoint_gap:
                    continue
                outward = normalize(endpoint_segment_direction(atom, endpoint))
                signed_alignment = dot(outward, inward)
                if signed_alignment <= -math.cos(math.radians(30)):
                    angle = math.degrees(math.acos(max(-1.0, min(1.0, -signed_alignment))))
                    source_gap = min(abs(atom.id - atom_id) for atom_id in cluster[index].atom_ids)
                    endpoint_options.append((angle, source_gap, gap, index, atom, endpoint))

    endpoints: list[Atom] = []
    endpoint_records: list[tuple[int, Atom, Point]] = []
    used_ports: set[int] = set()
    used_atoms: set[int] = set()
    for _, _, _, index, atom, endpoint in sorted(
        endpoint_options,
        key=lambda item: (item[0], item[1], item[2], item[3], item[4].id),
    ):
        if index in used_ports or atom.id in used_atoms:
            continue
        used_ports.add(index)
        used_atoms.add(atom.id)
        endpoints.append(atom)
        endpoint_records.append((index, atom, endpoint))
        explained.add(atom.id)

    # Dashed/segmented carriers may need several separate painted subpaths at
    # an open end.  Extend only after a true multi-Atom bridge has validated
    # that representation, and stop on any local branching ambiguity.
    if any(len(match.fragments) > 1 for match in bridges):
        extension_excluded = set(explained) | all_cluster_atom_ids
        for index, anchor, endpoint in endpoint_records:
            neighbor = next(iter(adjacency[index]))
            extensions = extend_terminal_bridge(
                cluster[index].center, cluster[neighbor].center,
                anchor, endpoint, atoms, extension_excluded, spacing,
            )
            endpoints.extend(extensions)
            explained.update(atom.id for atom in extensions)

    member_count = cluster[0].fingerprint["member_count"]
    score = (
        len(explained),
        len(involved),
        -js_round_nonnegative(spacing_cv * 10_000),
        -member_count,
    )
    # ``cluster`` may contain look-alike Candidates that never joined a valid
    # carrier component.  Do not leak those unrelated Candidates—or sparse old
    # indices—into the accepted hypothesis.
    ordered_involved = sorted(involved)
    remapped_index = {old: new for new, old in enumerate(ordered_involved)}
    compact_cluster = [cluster[index] for index in ordered_involved]
    compact_edges = [
        NetworkEdge(remapped_index[edge.left_index], remapped_index[edge.right_index], edge.length)
        for edge in edges
    ]
    compact_bridges = [
        BridgeMatch(
            match.fragments,
            remapped_index[match.left_index],
            remapped_index[match.right_index],
            match.left_endpoint,
            match.right_endpoint,
            match.route_length,
            match.ink_length,
            match.total_gap,
            match.maximum_lateral,
            match.occupancy_signature,
            match.fit_cost,
        )
        for match in bridges
    ]
    return NetworkHypothesis(
        compact_cluster, motif_ids, member_count, len(compact_cluster), spacing, spacing_cv,
        maximum_error, compact_edges, compact_bridges, endpoints, explained, score,
    )


def _local_subset_candidate_sets(cluster: list[Candidate]) -> list[list[Candidate]]:
    """Propose large same-period subsets without deciding that they are valid.

    This deliberately uses centers only to *propose* work for
    ``build_network_hypothesis``.  A proposed edge has no evidential value
    until that builder finds real painted Atom paths between its endpoints.
    """
    minimum_stations = max(7, math.ceil(len(cluster) * 0.5))
    motif_scale = median(candidate.scale for candidate in cluster)
    pair_distances = [
        distance(cluster[left].center, cluster[right].center)
        for left in range(len(cluster))
        for right in range(left + 1, len(cluster))
        if distance(cluster[left].center, cluster[right].center) > motif_scale * 0.5
    ]
    proposals: dict[tuple[int, ...], tuple[float, list[Candidate]]] = {}
    for seed in pair_distances:
        edges = [
            (left, right, distance(cluster[left].center, cluster[right].center))
            for left in range(len(cluster))
            for right in range(left + 1, len(cluster))
            if abs(distance(cluster[left].center, cluster[right].center) / seed - 1) <= 0.12
        ]
        nodes = tuple(sorted({index for left, right, _ in edges for index in (left, right)}))
        if len(nodes) < minimum_stations:
            continue
        lengths = [length for left, right, length in edges if left in nodes and right in nodes]
        variation = population_std(lengths) / max(mean(lengths), EPS)
        proposal = [cluster[index] for index in nodes]
        previous = proposals.get(nodes)
        if previous is None or variation < previous[0]:
            proposals[nodes] = variation, proposal
    # Pair distances can create many near-identical proposals.  The cap keeps
    # this explicitly opt-in fallback bounded on a noisy group.
    ranked = sorted(
        proposals.values(),
        key=lambda item: (-len(item[1]), item[0], tuple(candidate.id for candidate in item[1])),
    )
    return [proposal for _, proposal in ranked[:32]]


def _verified_path_subset(hypothesis: NetworkHypothesis, source_size: int) -> bool:
    """Require a large, stable set of real bridges forming disjoint paths."""
    if hypothesis.station_count < max(7, math.ceil(source_size * 0.5)):
        return False
    if len(hypothesis.bridge_followers) < max(6, math.ceil(source_size * 0.4)):
        return False
    if hypothesis.spacing_cv > 0.06 or hypothesis.maximum_relative_spacing_error > 0.12:
        return False
    if len(hypothesis.network_edges) != len(hypothesis.bridge_followers):
        return False
    bridge_ids = [atom_id for match in hypothesis.bridge_followers for atom_id in match.atom_ids]
    if len(bridge_ids) != len(set(bridge_ids)):
        return False
    bridge_atoms = [atom for match in hypothesis.bridge_followers for atom in match.atoms]
    if not bridge_atoms or any(not same_junction_style(bridge_atoms[0], atom) for atom in bridge_atoms[1:]):
        return False

    adjacency: dict[int, set[int]] = {}
    for edge in hypothesis.network_edges:
        adjacency.setdefault(edge.left_index, set()).add(edge.right_index)
        adjacency.setdefault(edge.right_index, set()).add(edge.left_index)
    if len(adjacency) != hypothesis.station_count or any(len(neighbors) > 2 for neighbors in adjacency.values()):
        return False
    components: list[set[int]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen)
        component = {start}
        stack = [start]
        unseen.remove(start)
        while stack:
            current = stack.pop()
            for following in adjacency[current]:
                if following in unseen:
                    unseen.remove(following)
                    component.add(following)
                    stack.append(following)
        components.append(component)
    if any(len(component) < 3 for component in components):
        return False
    if len(hypothesis.network_edges) != hypothesis.station_count - len(components):
        return False

    relative_angles: list[float] = []
    for index, neighbors in adjacency.items():
        neighbor_list = sorted(neighbors)
        if len(neighbor_list) == 1:
            tangent = sub(hypothesis.cluster[neighbor_list[0]].center, hypothesis.cluster[index].center)
        else:
            tangent = sub(
                hypothesis.cluster[neighbor_list[1]].center,
                hypothesis.cluster[neighbor_list[0]].center,
            )
        samples = [point for atom in hypothesis.cluster[index].members for point in atom.samples]
        axis, _ = principal_frame(samples, hypothesis.cluster[index].center)
        relative_angles.append(acute_angle_degrees(axis, tangent))
    return max(relative_angles, default=0.0) - min(relative_angles, default=0.0) <= 20


def _network_endpoint_adjacency(hypothesis: NetworkHypothesis) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {}
    for edge in hypothesis.network_edges:
        adjacency.setdefault(edge.left_index, set()).add(edge.right_index)
        adjacency.setdefault(edge.right_index, set()).add(edge.left_index)
    return adjacency


def _network_port_already_occupied(
    hypothesis: NetworkHypothesis,
    index: int,
    neighbor: int,
) -> bool:
    center = hypothesis.cluster[index].center
    inward = normalize(sub(hypothesis.cluster[neighbor].center, center))
    maximum_gap = hypothesis.spacing * 0.36
    for atom in hypothesis.endpoint_followers:
        for endpoint in (atom.points[0], atom.points[-1]):
            if distance(center, endpoint) > maximum_gap:
                continue
            direction = normalize(endpoint_segment_direction(atom, endpoint))
            if dot(direction, inward) <= -math.cos(math.radians(30)):
                return True
    return False


def _relaxed_outer_bridge(
    hypothesis: NetworkHypothesis,
    endpoint_index: int,
    candidate: Candidate,
    atoms: list[Atom],
    excluded_ids: set[int],
) -> BridgeMatch | None:
    """Find one real Atom across a single, terminal 1.45T..1.55T interval."""
    center = hypothesis.cluster[endpoint_index].center
    target = candidate.center
    interval = distance(center, target)
    spacing = hypothesis.spacing
    if not 1.45 < interval / spacing <= 1.55:
        return None
    incident = [
        match for match in hypothesis.bridge_followers
        if endpoint_index in (match.left_index, match.right_index)
    ]
    if len(incident) != 1:
        return None
    reference = incident[0].atoms[0]
    edge = NetworkEdge(endpoint_index, len(hypothesis.cluster), interval)
    options: list[BridgeMatch] = []
    for atom in atoms:
        if (
            atom.id in excluded_ids
            or atom.closed
            or atom.length > spacing * 1.46
            or not same_junction_style(reference, atom)
        ):
            continue
        chord = distance(atom.points[0], atom.points[-1])
        if atom.length <= EPS or chord / atom.length < 0.70:
            continue
        for reverse in (False, True):
            points = list(reversed(atom.points)) if reverse else list(atom.points)
            left_gap = distance(center, points[0])
            right_gap = distance(points[-1], target)
            if max(left_gap, right_gap) > spacing * 0.12:
                continue
            route_length = left_gap + atom.length + right_gap
            if abs(route_length / interval - 1) > 0.08:
                continue
            progress = [project_point_to_segment(point, center, target).raw_t for point in points]
            if min(progress) < -0.05 or max(progress) > 1.05:
                continue
            if any(progress[position] + 0.03 < progress[position - 1] for position in range(1, len(progress))):
                continue
            maximum_lateral = max(lateral_distance_to_line(point, center, target) for point in points)
            if maximum_lateral > spacing * 0.12:
                continue
            fragment = OrientedFragment(
                atom, points, points[0], points[-1],
                first_polyline_tangent(points), last_polyline_tangent(points),
                progress[0], progress[-1], maximum_lateral, reverse,
            )
            match = make_bridge_match([fragment], edge, center, target, spacing)
            match.fit_cost = (
                abs(route_length / interval - 1)
                + (left_gap + right_gap + maximum_lateral) / spacing
            )
            options.append(match)
    options.sort(key=lambda match: (match.fit_cost, match.atom_ids))
    if not options:
        return None
    if len(options) > 1 and options[1].fit_cost - options[0].fit_cost < 0.05:
        return None
    return options[0]


def _unique_outer_endpoint_atom(
    center: Point,
    previous_center: Point,
    reference: Atom,
    atoms: list[Atom],
    excluded_ids: set[int],
    spacing: float,
) -> Atom | None:
    outward = normalize(sub(center, previous_center))
    options: list[tuple[float, int, Atom]] = []
    for atom in atoms:
        if (
            atom.id in excluded_ids
            or atom.closed
            or not spacing * 0.05 <= atom.length <= spacing * 1.10
            or not same_junction_style(reference, atom)
        ):
            continue
        chord = distance(atom.points[0], atom.points[-1])
        if atom.length <= EPS or chord / atom.length < 0.70:
            continue
        for endpoint in (atom.points[0], atom.points[-1]):
            gap = distance(center, endpoint)
            if gap > spacing * 0.12:
                continue
            direction = normalize(endpoint_segment_direction(atom, endpoint))
            alignment = dot(direction, outward)
            if alignment < math.cos(math.radians(25)):
                continue
            points = list(atom.points) if distance(endpoint, atom.points[0]) <= distance(endpoint, atom.points[-1]) else list(reversed(atom.points))
            lateral = max(
                abs(sub(point, center)[0] * outward[1] - sub(point, center)[1] * outward[0])
                for point in points
            )
            if lateral > spacing * 0.12:
                continue
            cost = gap / spacing + (1 - alignment) + lateral / spacing
            options.append((cost, atom.id, atom))
    best_by_atom: dict[int, tuple[float, int, Atom]] = {}
    for option in options:
        if option[1] not in best_by_atom or option < best_by_atom[option[1]]:
            best_by_atom[option[1]] = option
    ranked = sorted(best_by_atom.values())
    if not ranked:
        return None
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.05:
        return None
    return ranked[0][2]


def _complete_one_local_subset_terminal(
    hypothesis: NetworkHypothesis,
    source_cluster: list[Candidate],
    atoms: list[Atom],
) -> NetworkHypothesis:
    """Add at most one clipped outer interval; never recurse from the new end."""
    adjacency = _network_endpoint_adjacency(hypothesis)
    explained = set(hypothesis.explained_atom_ids)
    source_atom_ids = {atom_id for candidate in source_cluster for atom_id in candidate.atom_ids}
    involved_candidate_ids = set(hypothesis.motif_atom_ids)
    options: list[tuple[float, int, Candidate, BridgeMatch, Atom]] = []
    for endpoint_index, neighbors in adjacency.items():
        if len(neighbors) != 1:
            continue
        neighbor = next(iter(neighbors))
        if _network_port_already_occupied(hypothesis, endpoint_index, neighbor):
            continue
        center = hypothesis.cluster[endpoint_index].center
        previous_center = hypothesis.cluster[neighbor].center
        outward = sub(center, previous_center)
        endpoint_samples = [
            point for atom in hypothesis.cluster[endpoint_index].members for point in atom.samples
        ]
        endpoint_axis, _ = principal_frame(endpoint_samples, center)
        endpoint_angle = acute_angle_degrees(endpoint_axis, outward)
        endpoint_candidate = hypothesis.cluster[endpoint_index]
        for candidate in source_cluster:
            if any(atom_id in involved_candidate_ids or atom_id in explained for atom_id in candidate.atom_ids):
                continue
            # The source group was built by transitive similarity.  A relaxed
            # terminal addition must still match the actual endpoint directly.
            if (
                not fingerprints_match(endpoint_candidate.fingerprint, candidate.fingerprint)
                or not candidate_has_uniform_stroke_style(endpoint_candidate)
                or not candidate_has_uniform_stroke_style(candidate)
                or not same_junction_style(endpoint_candidate.members[0], candidate.members[0])
            ):
                continue
            continuation = sub(candidate.center, center)
            if directed_angle_degrees(outward, continuation) > 20:
                continue
            candidate_samples = [point for atom in candidate.members for point in atom.samples]
            candidate_axis, _ = principal_frame(candidate_samples, candidate.center)
            if abs(acute_angle_degrees(candidate_axis, continuation) - endpoint_angle) > 15:
                continue
            bridge = _relaxed_outer_bridge(
                hypothesis, endpoint_index, candidate, atoms,
                explained | source_atom_ids,
            )
            if bridge is None:
                continue
            outer = _unique_outer_endpoint_atom(
                candidate.center, center, bridge.atoms[0], atoms,
                explained | source_atom_ids | set(bridge.atom_ids),
                hypothesis.spacing,
            )
            if outer is None:
                continue
            cost = bridge.fit_cost + directed_angle_degrees(outward, continuation) / 180
            options.append((cost, endpoint_index, candidate, bridge, outer))
    # A relaxed boundary decision must be unambiguous over the whole validated
    # subset.  It is intentionally one-shot rather than one step per endpoint.
    if len(options) != 1:
        return hypothesis
    _, endpoint_index, candidate, bridge, outer = options[0]
    new_index = len(hypothesis.cluster)
    bridge.left_index = endpoint_index
    bridge.right_index = new_index
    new_edge = NetworkEdge(endpoint_index, new_index, distance(
        hypothesis.cluster[endpoint_index].center, candidate.center,
    ))
    new_motif_ids = set(hypothesis.motif_atom_ids) | set(candidate.atom_ids)
    new_explained = set(hypothesis.explained_atom_ids) | set(candidate.atom_ids) | set(bridge.atom_ids) | {outer.id}
    new_station_count = hypothesis.station_count + 1
    score = (
        len(new_explained), new_station_count,
        -js_round_nonnegative(hypothesis.spacing_cv * 10_000),
        -hypothesis.motif_member_count,
    )
    return NetworkHypothesis(
        list(hypothesis.cluster) + [candidate], new_motif_ids,
        hypothesis.motif_member_count, new_station_count,
        hypothesis.spacing, hypothesis.spacing_cv,
        hypothesis.maximum_relative_spacing_error,
        list(hypothesis.network_edges) + [new_edge],
        list(hypothesis.bridge_followers) + [bridge],
        list(hypothesis.endpoint_followers) + [outer],
        new_explained, score, hypothesis.relaxed_terminal_edge_count + 1,
    )


def build_reliable_local_subset_hypothesis(
    cluster: list[Candidate],
    atoms: list[Atom],
    complete_one_terminal: bool = True,
    standard_hypothesis_cache: dict[int, Any] | None = None,
) -> NetworkHypothesis | None:
    """Recover a bridge-proven local period from one contaminated shape group.

    Discovery calls this only as a final fallback.  The helper also verifies
    for itself that both standard builders reject the complete input cluster.
    """
    # The proposal stage intentionally compares several possible local
    # spacings.  Keep this conservative fallback bounded on very large text or
    # drawing clusters; those need a spatial pre-partition before local RANSAC.
    if not 10 <= len(cluster) <= 32 or not non_overlapping(cluster):
        return None
    cache_key = id(cluster)
    if standard_hypothesis_cache is not None and cache_key in standard_hypothesis_cache:
        if standard_hypothesis_cache[cache_key] is not None:
            return None
    elif build_hypothesis(cluster, atoms) is not None or build_network_hypothesis(cluster, atoms) is not None:
        return None
    recovered: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        NetworkHypothesis,
    ] = {}
    for subset in _local_subset_candidate_sets(cluster):
        hypothesis = build_network_hypothesis(subset, atoms)
        if hypothesis is None or not _verified_path_subset(hypothesis, len(cluster)):
            continue
        key = (
            tuple(sorted(hypothesis.motif_atom_ids)),
            tuple(sorted(atom_id for match in hypothesis.bridge_followers for atom_id in match.atom_ids)),
            tuple(sorted(atom.id for atom in hypothesis.endpoint_followers)),
        )
        recovered[key] = hypothesis
    ranked = sorted(
        recovered.values(),
        key=lambda hypothesis: (
            hypothesis.station_count,
            len(hypothesis.bridge_followers),
            len(hypothesis.explained_atom_ids),
            -hypothesis.spacing_cv,
        ),
        reverse=True,
    )
    if not ranked:
        return None
    if len(ranked) > 1:
        best, second = ranked[:2]
        if (
            best.station_count - second.station_count < 2
            or len(best.bridge_followers) - len(second.bridge_followers) < 2
        ):
            return None
    result = ranked[0]
    return _complete_one_local_subset_terminal(result, cluster, atoms) if complete_one_terminal else result


def ordinary_single_line_candidate(candidate: Candidate) -> bool:
    """Whether a Candidate is only one ordinary open straight segment."""
    if len(candidate.members) != 1:
        return False
    atom = candidate.members[0]
    chord = distance(atom.points[0], atom.points[-1])
    return (
        not atom.closed
        and atom.curve_segments == 0
        and atom.line_segments == 1
        and chord / max(atom.length, EPS) >= 0.98
    )


def candidate_has_uniform_stroke_style(candidate: Candidate) -> bool:
    if not candidate.members or candidate.members[0].paint_mode != "stroke":
        return False
    reference = candidate.members[0]
    return all(same_junction_style(reference, atom) for atom in candidate.members[1:])


def command_positions_are_contiguous(atom_ids: Iterable[int], positions: dict[int, int]) -> bool:
    atom_ids = list(atom_ids)
    selected = sorted(positions[atom_id] for atom_id in atom_ids if atom_id in positions)
    return bool(selected) and len(selected) == len(set(atom_ids)) and selected == list(range(selected[0], selected[-1] + 1))


def two_instance_candidate_pairs(
    candidates: list[Candidate],
    positions: dict[int, int],
) -> tuple[list[Candidate], list[tuple[Candidate, Candidate]], set[int]]:
    """Find isolated direct matches without using transitive clustering.

    Ordinary one-segment lines are deliberately excluded.  A Candidate must
    match exactly one peer directly; matching any third eligible Candidate is
    enough to leave this conservative two-instance branch.
    """
    eligible = [
        candidate
        for candidate in candidates
        if not ordinary_single_line_candidate(candidate)
        and candidate_has_uniform_stroke_style(candidate)
        and command_positions_are_contiguous(candidate.atom_ids, positions)
    ]
    adjacency = [set() for _ in eligible]
    buckets: dict[tuple[int, int, int, int], list[int]] = {}
    for index, candidate in enumerate(eligible):
        fingerprint = candidate.fingerprint
        key = (
            fingerprint["member_count"], fingerprint["closed_count"],
            fingerprint["curved_count"], fingerprint["filled_count"],
        )
        buckets.setdefault(key, []).append(index)
    log_window = math.log(1.3)
    start_positions = [
        min(positions[atom_id] for atom_id in candidate.atom_ids)
        for candidate in eligible
    ]
    fingerprint_identities = [
        tuple(
            (field, tuple(value) if isinstance(value, list) else value)
            for field, value in sorted(candidate.fingerprint.items())
        )
        for candidate in eligible
    ]
    exact_identities: dict[tuple[Any, ...], int] = {}
    global_scale_buckets: dict[tuple[int, int, int], list[tuple[float, int]]] = {}
    for index, candidate in enumerate(eligible):
        reference = candidate.members[0]
        exact_key = (
            fingerprint_identities[index],
            reference.paint_mode,
            reference.line_cap,
            reference.line_width,
            tuple(reference.stroke_color),
        )
        exact_identities[exact_key] = exact_identities.get(exact_key, 0) + 1
        fingerprint = candidate.fingerprint
        global_key = (
            fingerprint["member_count"],
            fingerprint["closed_count"],
            fingerprint["filled_count"],
        )
        global_scale_buckets.setdefault(global_key, []).append((
            fingerprint["log_scale"], index,
        ))
    global_scale_windows: dict[
        tuple[int, int, int], tuple[list[float], list[tuple[float, int]]]
    ] = {}
    for key, entries in global_scale_buckets.items():
        entries.sort()
        global_scale_windows[key] = ([entry[0] for entry in entries], entries)
    for indices in buckets.values():
        # The strict two-copy interpretation below already requires nearby
        # command blocks.  Sweep that bounded coordinate first so a large CAD
        # Group with many similarly-sized motifs does not scan an O(N²) scale
        # window merely to reject almost every pair as non-local.
        indices.sort(key=lambda index: (start_positions[index], index))
        for offset, left_index in enumerate(indices):
            left = eligible[left_index]
            for right_index in indices[offset + 1 :]:
                right = eligible[right_index]
                if start_positions[right_index] - start_positions[left_index] > 32:
                    break
                if abs(
                    right.fingerprint["log_scale"] - left.fingerprint["log_scale"]
                ) > log_window:
                    continue
                if not same_junction_style(left.members[0], right.members[0]):
                    continue
                if fingerprints_match(left.fingerprint, right.fingerprint):
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)
    pairs: list[tuple[Candidate, Candidate]] = []
    for left_index, peers in enumerate(adjacency):
        if len(peers) != 1:
            continue
        right_index = next(iter(peers))
        if left_index >= right_index or adjacency[right_index] != {left_index}:
            continue
        left, right = eligible[left_index], eligible[right_index]
        # Locality limits the expensive all-pairs proposal pass, but
        # "occurs exactly twice" is still a global statement inside this
        # coarse group.  A directly matching third Candidate anywhere rejects
        # the guarded two-copy interpretation.
        reference = left.members[0]
        exact_key = (
            fingerprint_identities[left_index],
            reference.paint_mode,
            reference.line_cap,
            reference.line_width,
            tuple(reference.stroke_color),
        )
        if exact_identities.get(exact_key, 0) >= 3:
            continue
        fingerprint = left.fingerprint
        global_key = (
            fingerprint["member_count"],
            fingerprint["closed_count"],
            fingerprint["filled_count"],
        )
        scales, entries = global_scale_windows[global_key]
        window_start = bisect_left(scales, fingerprint["log_scale"] - log_window)
        window_end = bisect_right(scales, fingerprint["log_scale"] + log_window)
        if any(
            other_index not in (left_index, right_index)
            and same_junction_style(reference, eligible[other_index].members[0])
            and fingerprints_match(fingerprint, eligible[other_index].fingerprint)
            for _, other_index in entries[window_start:window_end]
        ):
            continue
        if non_overlapping([left, right]):
            pairs.append((left, right))
    matched_atom_ids = {
        atom_id
        for index, peers in enumerate(adjacency)
        if peers
        for atom_id in eligible[index].atom_ids
    }
    return eligible, pairs, matched_atom_ids


def distance_to_candidate_ink(point: Point, candidate: Candidate) -> float:
    """Shortest distance from a connector endpoint to the motif's real ink."""
    distances: list[float] = []
    for atom in candidate.members:
        if len(atom.points) == 1:
            distances.append(distance(point, atom.points[0]))
            continue
        distances.extend(
            project_point_to_segment(point, atom.points[index - 1], atom.points[index]).distance
            for index in range(1, len(atom.points))
        )
    return min(distances, default=math.inf)


def unique_two_instance_middle_bridge(
    left: Candidate,
    right: Candidate,
    atoms: list[Atom],
    excluded_ids: set[int],
    spacing: float,
) -> BridgeMatch | None:
    """Return the sole real 1..8-Atom path between two Candidate centers."""
    start, end = left.center, right.center
    reference = left.members[0]
    end_gap_limit = spacing * 0.18
    internal_gap_limit = spacing * 0.12
    fragments = [
        fragment
        for atom in atoms
        if atom.id not in excluded_ids and same_junction_style(reference, atom)
        for fragment in oriented_fragments_for_edge(
            atom, start, end, spacing, spacing * 1.10,
        )
    ]
    states: list[tuple[tuple[OrientedFragment, ...], frozenset[int]]] = [
        ((fragment,), frozenset((fragment.atom.id,)))
        for fragment in fragments
        if distance(start, fragment.start) <= end_gap_limit
    ]
    completed: dict[tuple[tuple[int, bool], ...], BridgeMatch] = {}
    edge = NetworkEdge(0, 1, spacing)
    for depth in range(1, 9):
        following_states: list[tuple[tuple[OrientedFragment, ...], frozenset[int]]] = []
        for path, used_ids in states:
            sink_gap = distance(path[-1].end, end)
            if sink_gap <= end_gap_limit:
                if (
                    distance_to_candidate_ink(path[0].start, left) > internal_gap_limit
                    or distance_to_candidate_ink(path[-1].end, right) > internal_gap_limit
                ):
                    continue
                match = make_bridge_match(list(path), edge, start, end, spacing)
                internal_gaps = [
                    distance(path[index - 1].end, path[index].start)
                    for index in range(1, len(path))
                ]
                individual_gaps = [
                    distance(start, path[0].start), *internal_gaps, sink_gap,
                ]
                turns = [
                    directed_angle_degrees(path[index - 1].end_tangent, path[index].start_tangent)
                    for index in range(1, len(path))
                ]
                if (
                    0.85 <= match.route_length / spacing <= 1.35
                    and match.ink_length / spacing >= 0.45
                    and match.total_gap / spacing <= 0.36
                    and individual_gaps[0] <= end_gap_limit
                    and individual_gaps[-1] <= end_gap_limit
                    and max(individual_gaps[1:-1], default=0.0) <= internal_gap_limit
                    and max(turns, default=0.0) <= 100
                    and match.fit_cost <= 0.45
                ):
                    key = tuple((fragment.atom.id, fragment.reversed) for fragment in path)
                    if key not in completed or match.fit_cost < completed[key].fit_cost:
                        completed[key] = match
            if depth == 8:
                continue
            last = path[-1]
            for fragment in fragments:
                if fragment.atom.id in used_ids:
                    continue
                if distance(last.end, fragment.start) > internal_gap_limit:
                    continue
                if fragment.t_start < last.t_end - 0.03:
                    continue
                if directed_angle_degrees(last.end_tangent, fragment.start_tangent) > 100:
                    continue
                if len(following_states) >= 512:
                    return None
                following_states.append((
                    path + (fragment,), used_ids | {fragment.atom.id},
                ))
        # Overflow means the geometry itself is too ambiguous for a two-copy
        # decision; truncating here could incorrectly manufacture uniqueness.
        states = following_states
        if not states:
            break
    return next(iter(completed.values())) if len(completed) == 1 else None


def two_instance_extension_options(
    candidate: Candidate,
    center: Point,
    neighbor_center: Point,
    expected_tangent: Point,
    atoms: list[Atom],
    excluded_ids: set[int],
    reference: Atom,
    spacing: float,
) -> list[TwoInstanceExtension]:
    """Find all strict outward continuations at one end of a two-copy line."""
    expected = normalize(expected_tangent)
    inward = normalize(sub(neighbor_center, center))
    if math.hypot(*expected) <= EPS or math.hypot(*inward) <= EPS:
        return []
    options_by_atom: dict[int, tuple[float, TwoInstanceExtension]] = {}
    for atom in atoms:
        if (
            atom.id in excluded_ids
            or atom.closed
            or atom.length > spacing * 1.35
            or not same_junction_style(reference, atom)
        ):
            continue
        chord = distance(atom.points[0], atom.points[-1])
        if chord / max(atom.length, EPS) < 0.45:
            continue
        for reverse in (False, True):
            points = list(reversed(atom.points)) if reverse else list(atom.points)
            gap = distance(center, points[0])
            if gap > spacing * 0.18:
                continue
            if distance_to_candidate_ink(points[0], candidate) > spacing * 0.12:
                continue
            outward = normalize(first_polyline_tangent(points))
            tangent_dot = max(-1.0, min(1.0, dot(outward, expected)))
            angle = math.degrees(math.acos(tangent_dot))
            if angle > 15:
                continue
            if dot(outward, inward) > -math.cos(math.radians(15)):
                continue
            route_length = gap + atom.length
            if not spacing * 0.45 <= route_length <= spacing * 1.35:
                continue
            progress = [dot(sub(point, center), expected) / spacing for point in points]
            if progress[-1] <= progress[0] + EPS:
                continue
            if any(progress[index] + 0.03 < progress[index - 1] for index in range(1, len(progress))):
                continue
            lateral = max(
                abs(sub(point, center)[0] * expected[1] - sub(point, center)[1] * expected[0])
                for point in points
            )
            if lateral > spacing * 0.25:
                continue
            option = TwoInstanceExtension(atom, points, points[0], gap, route_length, angle)
            cost = gap / spacing + angle / 180 + abs(route_length / spacing - 1)
            if atom.id not in options_by_atom or cost < options_by_atom[atom.id][0]:
                options_by_atom[atom.id] = cost, option
    return [item[1] for item in sorted(options_by_atom.values(), key=lambda item: (item[0], item[1].atom.id))]


def two_instance_roles_are_command_local(
    positions: dict[int, int],
    left: Candidate,
    right: Candidate,
    middle: BridgeMatch,
    left_extension: TwoInstanceExtension,
    right_extension: TwoInstanceExtension,
) -> bool:
    """Require one unbroken local command block in either drawing direction."""
    roles: dict[int, str] = {
        **{atom_id: "A" for atom_id in left.atom_ids},
        **{atom_id: "B" for atom_id in right.atom_ids},
        **{atom_id: "M" for atom_id in middle.atom_ids},
        left_extension.atom.id: "L",
        right_extension.atom.id: "R",
    }
    if not command_positions_are_contiguous(roles, positions):
        return False
    ordered_roles: list[str] = []
    for atom_id in sorted(roles, key=positions.get):
        role = roles[atom_id]
        if not ordered_roles or ordered_roles[-1] != role:
            ordered_roles.append(role)
    return ordered_roles in (["L", "A", "M", "B", "R"], ["R", "B", "M", "A", "L"])


def build_two_instance_hypotheses(
    atoms: list[Atom],
    candidates: list[Candidate] | None = None,
    original_positions: dict[int, int] | None = None,
) -> list[TwoInstanceHypothesis]:
    """Build disjoint, high-confidence hypotheses for motifs seen exactly twice.

    ``atoms`` must be the currently unclaimed atoms.  This helper neither
    mutates them nor changes the normal discovery loop.
    """
    if len(atoms) < 5:
        return []
    positions = original_positions or {atom.id: index for index, atom in enumerate(atoms)}
    candidates = candidates if candidates is not None else generate_motif_candidates(atoms)
    eligible, pairs, matched_candidate_atom_ids = two_instance_candidate_pairs(candidates, positions)
    proposals: list[TwoInstanceHypothesis] = []
    for first, second in pairs:
        left, right = sorted((first, second), key=lambda candidate: candidate.center)
        spacing = distance(left.center, right.center)
        if spacing <= EPS or max(left.scale, right.scale) / spacing > 0.25:
            continue
        motif_ids = set(left.atom_ids) | set(right.atom_ids)
        # Candidate-shaped atoms belonging to another possible pattern are
        # protected rather than silently borrowed as carrier ink.
        protected_ids = matched_candidate_atom_ids - motif_ids
        middle = unique_two_instance_middle_bridge(
            left, right, atoms, motif_ids | protected_ids, spacing,
        )
        if middle is None:
            continue
        middle_ids = set(middle.atom_ids)
        shared_excluded = motif_ids | protected_ids | middle_ids
        left_options = two_instance_extension_options(
            left,
            left.center, right.center,
            mul(middle.fragments[0].start_tangent, -1),
            atoms, shared_excluded, left.members[0], spacing,
        )
        right_options = two_instance_extension_options(
            right,
            right.center, left.center,
            middle.fragments[-1].end_tangent,
            atoms, shared_excluded, left.members[0], spacing,
        )
        if len(left_options) != 1 or len(right_options) != 1:
            continue
        left_extension, right_extension = left_options[0], right_options[0]
        if left_extension.atom.id == right_extension.atom.id:
            continue
        if abs(left_extension.route_length - right_extension.route_length) / spacing > 0.15:
            continue
        if not two_instance_roles_are_command_local(
            positions, left, right, middle, left_extension, right_extension,
        ):
            continue
        explained = motif_ids | middle_ids | {
            left_extension.atom.id, right_extension.atom.id,
        }
        quality = (
            middle.fit_cost
            + abs(left_extension.route_length - right_extension.route_length) / spacing
            + (left_extension.angle_degrees + right_extension.angle_degrees) / 360
        )
        score = (
            len(explained), 2, -js_round_nonnegative(quality * 10_000),
            -left.fingerprint["member_count"],
        )
        proposals.append(TwoInstanceHypothesis(
            [left, right], motif_ids, left.fingerprint["member_count"], spacing,
            middle, left_extension, right_extension, explained, score,
        ))

    # If two proposals want any of the same Atom, neither may win by greedily
    # taking it first.  Ambiguity is safer to reject in a two-copy fallback.
    owners: dict[int, set[int]] = {}
    for index, proposal in enumerate(proposals):
        for atom_id in proposal.explained_atom_ids:
            owners.setdefault(atom_id, set()).add(index)
    conflicting = {
        index
        for indices in owners.values()
        if len(indices) > 1
        for index in indices
    }
    return sorted(
        (proposal for index, proposal in enumerate(proposals) if index not in conflicting),
        key=lambda proposal: proposal.score,
        reverse=True,
    )


@dataclass
class _PathProjection:
    distance: float
    arc: float
    point: Point
    tangent: Point


def _path_projections(point: Point, segments: list[CarrierSegment]) -> list[_PathProjection]:
    """Return the only two projections consumed by ambiguity detection."""
    best: list[_PathProjection] = []
    for segment in segments:
        projection = project_point_to_segment(point, segment.start, segment.end)
        item = _PathProjection(
            projection.distance,
            segment.arc_start + projection.t * segment.length,
            projection.point,
            segment.direction,
        )
        best.append(item)
        best.sort(key=lambda value: (value.distance, value.arc))
        if len(best) > 2:
            best.pop()
    return best


def _candidate_principal_direction(candidate: Candidate) -> Point:
    samples = [point for atom in candidate.members for point in atom.samples]
    return principal_frame(samples, candidate.center)[0]


def _point_segment_distance(point: Point, segment: CarrierSegment) -> float:
    delta_x = segment.end[0] - segment.start[0]
    delta_y = segment.end[1] - segment.start[1]
    squared_length = delta_x * delta_x + delta_y * delta_y
    raw_t = (
        ((point[0] - segment.start[0]) * delta_x + (point[1] - segment.start[1]) * delta_y)
        / squared_length
        if squared_length > EPS else 0.0
    )
    t = max(0.0, min(1.0, raw_t))
    return math.hypot(
        point[0] - (segment.start[0] + delta_x * t),
        point[1] - (segment.start[1] + delta_y * t),
    )


def _candidate_ink_distance_to_path(points: list[Point], segments: list[CarrierSegment]) -> float:
    return min(
        min(
            _point_segment_distance(point, segment)
            for segment in segments
        )
        for point in points
    )


def _nearest_candidate_center_distances(cluster: list[Candidate]) -> list[float]:
    """Return each Candidate's exact nearest-center distance without O(N²) storage/work."""
    if len(cluster) < 512:
        return [
            min(
                distance(candidate.center, other.center)
                for other in cluster
                if other is not candidate
            )
            for candidate in cluster
        ]
    try:
        import numpy as np
        from scipy.spatial import cKDTree
    except ImportError:
        return [
            min(
                distance(candidate.center, other.center)
                for other in cluster
                if other is not candidate
            )
            for candidate in cluster
        ]
    centers = np.asarray([candidate.center for candidate in cluster], dtype=float)
    nearest, _ = cKDTree(centers).query(centers, k=2, workers=1)
    return [float(row[1]) for row in nearest]


def _attached_path_hypotheses(
    atoms: list[Atom],
    clusters: list[list[Candidate]],
) -> list[SharedPathHypothesis]:
    """Find repeated shapes attached to one real, shared carrier Atom."""
    proposals: list[SharedPathHypothesis] = []
    for cluster in clusters:
        if len(cluster) < 5 or not non_overlapping(cluster):
            continue
        if not all(candidate_has_uniform_stroke_style(candidate) for candidate in cluster):
            continue
        motif_scale = median(candidate.scale for candidate in cluster)
        nearest = _nearest_candidate_center_distances(cluster)
        typical_gap = median(nearest)
        # At 85% required support, a cluster whose median nearest distance is
        # exactly zero cannot select enough unique stations: at least one pair
        # of matched candidates must retain the same center.  Their projected
        # arc gap is zero, so the existing min(arc_gaps) <= EPS guard below is
        # guaranteed to reject every carrier.  Reject here before scanning
        # every Atom against (occasionally tens of thousands of) candidates.
        if typical_gap == 0.0:
            continue
        minimum_support = max(5, math.ceil(len(cluster) * 0.85))
        motif_ids = {atom_id for candidate in cluster for atom_id in candidate.atom_ids}
        reference_style = cluster[0].members[0]
        matching_candidates = [
            candidate for candidate in cluster
            if all(same_junction_style(reference_style, member) for member in candidate.members)
        ]
        if len(matching_candidates) < minimum_support:
            continue
        candidate_ink_points = {
            id(candidate): [
                point
                for atom in candidate.members
                for point in (*atom.samples, atom.points[0], atom.points[-1])
            ]
            for candidate in matching_candidates
        }
        candidate_widths = {
            id(candidate): median(atom.line_width for atom in candidate.members)
            for candidate in matching_candidates
        }
        candidate_directions = {
            id(candidate): _candidate_principal_direction(candidate)
            for candidate in matching_candidates
        }
        candidate_reaches = {
            id(candidate): (
                max(
                    (distance(candidate.center, point) for point in candidate_ink_points[id(candidate)]),
                    default=0.0,
                )
                + max(abs(candidate_widths[id(candidate)]) * 2, candidate.scale * 0.12)
            )
            for candidate in matching_candidates
        }
        for carrier in atoms:
            if (
                carrier.id in motif_ids
                or carrier.closed
                or carrier.paint_mode != "stroke"
                or carrier.length < max(motif_scale * 4, typical_gap * 4)
                or not same_junction_style(reference_style, carrier)
            ):
                continue
            segments = [segment for segment in build_core_segments(carrier.points) if segment.length > EPS]
            if not segments:
                continue
            carrier_length = sum(segment.length for segment in segments)
            carrier_min_x = min(min(segment.start[0], segment.end[0]) for segment in segments)
            carrier_max_x = max(max(segment.start[0], segment.end[0]) for segment in segments)
            carrier_min_y = min(min(segment.start[1], segment.end[1]) for segment in segments)
            carrier_max_y = max(max(segment.start[1], segment.end[1]) for segment in segments)
            nearby_candidates = [
                candidate
                for candidate in matching_candidates
                if (
                    carrier_min_x - candidate_reaches[id(candidate)] <= candidate.center[0]
                    <= carrier_max_x + candidate_reaches[id(candidate)]
                    and carrier_min_y - candidate_reaches[id(candidate)] <= candidate.center[1]
                    <= carrier_max_y + candidate_reaches[id(candidate)]
                )
            ]
            if len(nearby_candidates) < minimum_support:
                continue
            matched: list[tuple[Candidate, _PathProjection, float, float, float, list[_PathProjection]]] = []
            for candidate in nearby_candidates:
                center_projections = _path_projections(candidate.center, segments)
                if not center_projections:
                    continue
                best = center_projections[0]
                contact_distance = _candidate_ink_distance_to_path(
                    candidate_ink_points[id(candidate)], segments,
                )
                candidate_width = candidate_widths[id(candidate)]
                contact_limit = max(abs(candidate_width) * 2, candidate.scale * 0.12)
                if contact_distance > contact_limit:
                    continue
                relative = sub(candidate.center, best.point)
                signed_offset = (
                    best.tangent[0] * relative[1] - best.tangent[1] * relative[0]
                ) / max(candidate.scale, EPS)
                direction = candidate_directions[id(candidate)]
                relative_angle = acute_angle_degrees(direction, best.tangent)
                matched.append((
                    candidate, best, contact_distance, signed_offset,
                    relative_angle, center_projections,
                ))

            if len(matched) < minimum_support:
                continue
            matched.sort(key=lambda item: item[1].arc)
            arcs = [item[1].arc for item in matched]
            arc_gaps = [arcs[index] - arcs[index - 1] for index in range(1, len(arcs))]
            if not arc_gaps or min(arc_gaps) <= EPS:
                continue
            spacing = median(arc_gaps)
            if (
                min(arc_gaps) < spacing * 0.35
                or max(arc_gaps) > spacing * 2.25
                or (arcs[-1] - arcs[0]) / max(carrier_length, EPS) < 0.70
                or arcs[0] > spacing * 2.5
                or carrier_length - arcs[-1] > spacing * 2.5
            ):
                continue

            # A self-intersection can give one motif two equally good but far
            # apart arc positions.  Adjacent polyline segments at one vertex
            # are harmless because their arc positions are almost identical.
            ambiguous_projection = False
            for candidate, best, _, _, _, alternatives in matched:
                if len(alternatives) < 2:
                    continue
                second = alternatives[1]
                ambiguity_distance = max(
                    median(atom.line_width for atom in candidate.members) * 2,
                    candidate.scale * 0.05,
                )
                if (
                    second.distance <= best.distance + ambiguity_distance
                    and abs(second.arc - best.arc) >= spacing * 0.5
                ):
                    ambiguous_projection = True
                    break
            if ambiguous_projection:
                continue

            offsets = [item[3] for item in matched]
            if population_std(offsets) > 0.12:
                continue
            elongated_angles = [
                item[4]
                for item in matched
                if item[0].fingerprint["aspect_ratio"] < 0.5
            ]
            if elongated_angles and max(elongated_angles) - min(elongated_angles) > 15:
                continue

            repeated_ids = {
                atom_id
                for candidate, *_ in matched
                for atom_id in candidate.atom_ids
            }
            explained = repeated_ids | {carrier.id}
            normalized_contact = max(
                contact / max(candidate.scale, EPS)
                for candidate, _, contact, *_ in matched
            )
            fit_error = normalized_contact + population_std(offsets)
            score = (
                len(explained), len(matched),
                -js_round_nonnegative(fit_error * 10_000),
                -cluster[0].fingerprint["member_count"],
            )
            proposals.append(SharedPathHypothesis(
                "attached_repeat",
                [[carrier.id], *[candidate.atom_ids for candidate, *_ in matched]],
                repeated_ids,
                {carrier.id},
                explained,
                spacing,
                None,
                len(matched),
                fit_error,
                score,
            ))
    return proposals


@dataclass
class _PathLink:
    left_id: int
    left_side: int
    right_id: int
    right_side: int
    gap: float
    cost: float

    def other(self, atom_id: int) -> int:
        return self.right_id if atom_id == self.left_id else self.left_id


@dataclass
class _InkPath:
    tokens: list["_InkToken"]
    gaps: list[float]


@dataclass
class _InkToken:
    """One logical dash, possibly emitted as several touching PDF paths."""

    atoms: tuple[Atom, ...]
    points: list[Point]
    length: float

    @property
    def id(self) -> int:
        return min(atom.id for atom in self.atoms)

    @property
    def atom_ids(self) -> tuple[int, ...]:
        return tuple(sorted(atom.id for atom in self.atoms))

    @property
    def paint_mode(self) -> str:
        return self.atoms[0].paint_mode

    @property
    def line_width(self) -> float:
        return self.atoms[0].line_width

    @property
    def line_cap(self) -> int:
        return self.atoms[0].line_cap

    @property
    def stroke_color(self) -> tuple[float, ...]:
        return self.atoms[0].stroke_color


@dataclass
class _InkGapProfile:
    path: _InkPath
    token_count: int
    period_length: float
    signature: tuple[tuple[float, float], ...]
    repeat_count: int
    fit_error: float


def _endpoint_point(atom: Atom, side: int) -> Point:
    return atom.points[0] if side == 0 else atom.points[-1]


def _endpoint_outward_direction(atom: Atom, side: int) -> Point:
    endpoint = _endpoint_point(atom, side)
    return mul(normalize(endpoint_segment_direction(atom, endpoint)), -1)


def _style_buckets(atoms: list[Atom]) -> list[list[Atom]]:
    buckets: list[list[Atom]] = []
    for atom in atoms:
        for bucket in buckets:
            if same_junction_style(bucket[0], atom):
                bucket.append(atom)
                break
        else:
            buckets.append([atom])
    return buckets


INK_TOKEN_MAXIMUM_TURN_DEGREES = 30
INK_PERIOD_MAXIMUM_CORNER_ANOMALY_RATIO = 0.18
INK_PERIOD_MAXIMUM_CORNER_ANOMALIES = 24


def _merge_touching_ink_tokens(atoms: list[Atom]) -> list[_InkToken]:
    """Merge command-local pieces that are visibly one logical dash.

    CAD writers sometimes approximate a rounded dash at a corner with three
    separately painted short segments.  Treating those segments as three
    period tokens destroys an otherwise exact ink/gap sequence.  A merge is
    allowed only at a shared endpoint, between adjacent painted Atom ids, with
    a unique continuation and a modest local turn.  Crossings, branches and a
    sharp independent stroke therefore remain separate.
    """
    if not atoms:
        return []
    atom_by_id = {atom.id: atom for atom in atoms}
    options: list[_PathLink] = []
    ordered = sorted(atoms, key=lambda atom: atom.id)
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1:]:
            if right.id - left.id > 1:
                break
            if right.id - left.id != 1 or not same_junction_style(left, right):
                continue
            tolerance = junction_tolerance(left, right)
            for left_side in (0, 1):
                left_point = _endpoint_point(left, left_side)
                left_outward = _endpoint_outward_direction(left, left_side)
                if math.hypot(*left_outward) <= EPS:
                    continue
                for right_side in (0, 1):
                    right_point = _endpoint_point(right, right_side)
                    gap = distance(left_point, right_point)
                    if gap > tolerance:
                        continue
                    right_outward = _endpoint_outward_direction(right, right_side)
                    if math.hypot(*right_outward) <= EPS:
                        continue
                    turn = directed_angle_degrees(left_outward, mul(right_outward, -1))
                    if turn > INK_TOKEN_MAXIMUM_TURN_DEGREES:
                        continue
                    options.append(_PathLink(
                        left.id, left_side, right.id, right_side, gap, turn,
                    ))

    by_endpoint: dict[tuple[int, int], list[int]] = {}
    for index, link in enumerate(options):
        by_endpoint.setdefault((link.left_id, link.left_side), []).append(index)
        by_endpoint.setdefault((link.right_id, link.right_side), []).append(index)
    unique_at_endpoint = {
        endpoint: indices[0]
        for endpoint, indices in by_endpoint.items()
        if len(indices) == 1
    }
    selected = [
        link
        for index, link in enumerate(options)
        if (
            unique_at_endpoint.get((link.left_id, link.left_side)) == index
            and unique_at_endpoint.get((link.right_id, link.right_side)) == index
        )
    ]

    adjacency: dict[int, list[_PathLink]] = {atom.id: [] for atom in atoms}
    for link in selected:
        adjacency[link.left_id].append(link)
        adjacency[link.right_id].append(link)

    tokens: list[_InkToken] = []
    visited: set[int] = set()
    for atom in ordered:
        if atom.id in visited:
            continue
        component: set[int] = set()
        pending = [atom.id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(link.other(current) for link in adjacency[current])
        visited.update(component)
        if len(component) == 1:
            tokens.append(_InkToken((atom,), list(atom.points), atom.length))
            continue
        if any(len(adjacency[atom_id]) > 2 for atom_id in component):
            for atom_id in sorted(component):
                member = atom_by_id[atom_id]
                tokens.append(_InkToken((member,), list(member.points), member.length))
            continue
        endpoints = [atom_id for atom_id in component if len(adjacency[atom_id]) == 1]
        if len(endpoints) != 2:
            for atom_id in sorted(component):
                member = atom_by_id[atom_id]
                tokens.append(_InkToken((member,), list(member.points), member.length))
            continue

        current = min(endpoints)
        previous: int | None = None
        member_ids: list[int] = []
        points: list[Point] = []
        while current is not None:
            member = atom_by_id[current]
            member_ids.append(current)
            following = next(
                (link for link in adjacency[current] if link.other(current) != previous),
                None,
            )
            if following is None:
                oriented = (
                    list(member.points)
                    if not points or distance(points[-1], member.points[0]) <= distance(points[-1], member.points[-1])
                    else list(reversed(member.points))
                )
                points.extend(oriented[1:] if points and distance(points[-1], oriented[0]) <= EPS else oriented)
                current = None
                continue
            connected_side = (
                following.left_side if following.left_id == current else following.right_side
            )
            oriented = list(member.points) if connected_side == 1 else list(reversed(member.points))
            points.extend(oriented[1:] if points and distance(points[-1], oriented[0]) <= EPS else oriented)
            previous, current = current, following.other(current)

        members = tuple(atom_by_id[atom_id] for atom_id in member_ids)
        tokens.append(_InkToken(members, points, sum(member.length for member in members)))
    return sorted(tokens, key=lambda token: token.id)


def _open_ink_paths(atoms: list[Atom], protected_ids: set[int]) -> list[_InkPath]:
    eligible = [
        atom for atom in atoms
        if (
            atom.id not in protected_ids
            and atom.paint_mode == "stroke"
            and not atom.closed
            and atom.length > max(EPS, abs(atom.line_width) * 0.5)
            and distance(atom.points[0], atom.points[-1]) / max(atom.length, EPS) >= 0.75
        )
    ]
    paths: list[_InkPath] = []
    for bucket in _style_buckets(eligible):
        tokens = _merge_touching_ink_tokens(bucket)
        # This is a deliberately guarded fallback.  Very large groups need a
        # spatial index before an all-pairs endpoint search is safe.
        if len(tokens) < 5 or len(tokens) > 512:
            continue
        link_options: list[_PathLink] = []
        for left_index in range(len(tokens)):
            left = tokens[left_index]
            for right_index in range(left_index + 1, len(tokens)):
                right = tokens[right_index]
                maximum_gap = max(
                    min(left.length, right.length) * 3,
                    max(abs(left.line_width), abs(right.line_width)) * 4,
                )
                for left_side in (0, 1):
                    left_point = _endpoint_point(left, left_side)
                    left_outward = _endpoint_outward_direction(left, left_side)
                    if math.hypot(*left_outward) <= EPS:
                        continue
                    for right_side in (0, 1):
                        right_point = _endpoint_point(right, right_side)
                        right_outward = _endpoint_outward_direction(right, right_side)
                        if math.hypot(*right_outward) <= EPS:
                            continue
                        gap = distance(left_point, right_point)
                        if gap > maximum_gap:
                            continue
                        if gap <= EPS:
                            # At a shared endpoint the two outward directions
                            # must oppose one another.  Same-side strokes form
                            # an overlap/branch, not one continuing path.
                            left_angle = right_angle = directed_angle_degrees(
                                left_outward, mul(right_outward, -1),
                            )
                        else:
                            gap_direction = normalize(sub(right_point, left_point))
                            left_angle = directed_angle_degrees(left_outward, gap_direction)
                            right_angle = directed_angle_degrees(right_outward, mul(gap_direction, -1))
                        if left_angle > 18 or right_angle > 18:
                            continue
                        cost = (
                            gap / max(min(left.length, right.length), EPS)
                            + (left_angle + right_angle) / 36
                        )
                        link_options.append(_PathLink(
                            left.id, left_side, right.id, right_side, gap, cost,
                        ))

        by_endpoint: dict[tuple[int, int], list[int]] = {}
        for index, link in enumerate(link_options):
            by_endpoint.setdefault((link.left_id, link.left_side), []).append(index)
            by_endpoint.setdefault((link.right_id, link.right_side), []).append(index)

        endpoint_best: dict[tuple[int, int], int] = {}
        for endpoint, indices in by_endpoint.items():
            # The first visible continuation owns the endpoint.  A farther
            # same-sized dash must not jump over a nearer short dash merely
            # because normalizing by its own length gives it a lower cost.
            ranked = sorted(indices, key=lambda index: (link_options[index].gap, link_options[index].cost))
            best_index = ranked[0]
            if len(ranked) > 1:
                best, second = link_options[best_index], link_options[ranked[1]]
                if (
                    second.gap <= max(best.gap, EPS) * 1.25
                    and second.cost - best.cost < 0.20
                ):
                    continue
            endpoint_best[endpoint] = best_index

        selected_indices: set[int] = set()
        for index, link in enumerate(link_options):
            if (
                endpoint_best.get((link.left_id, link.left_side)) == index
                and endpoint_best.get((link.right_id, link.right_side)) == index
            ):
                selected_indices.add(index)

        adjacency: dict[int, list[_PathLink]] = {token.id: [] for token in tokens}
        for index in selected_indices:
            link = link_options[index]
            adjacency[link.left_id].append(link)
            adjacency[link.right_id].append(link)
        token_by_id = {token.id: token for token in tokens}
        visited: set[int] = set()
        for token in tokens:
            if token.id in visited or not adjacency[token.id]:
                continue
            component: set[int] = set()
            pending = [token.id]
            while pending:
                current = pending.pop()
                if current in component:
                    continue
                component.add(current)
                pending.extend(link.other(current) for link in adjacency[current])
            visited.update(component)
            if len(component) < 5 or any(len(adjacency[atom_id]) > 2 for atom_id in component):
                continue
            endpoints = [atom_id for atom_id in component if len(adjacency[atom_id]) == 1]
            if len(endpoints) != 2:
                continue
            start = min(
                endpoints,
                key=lambda token_id: polyline_halfway_point(token_by_id[token_id].points),
            )
            ordered_ids: list[int] = []
            gaps: list[float] = []
            previous: int | None = None
            current: int | None = start
            while current is not None:
                ordered_ids.append(current)
                following = next(
                    (link for link in adjacency[current] if link.other(current) != previous),
                    None,
                )
                if following is None:
                    current = None
                    continue
                gaps.append(following.gap)
                previous, current = current, following.other(current)
            if len(ordered_ids) == len(component) and len(gaps) == len(component) - 1:
                paths.append(_InkPath([token_by_id[token_id] for token_id in ordered_ids], gaps))
    return paths


def _canonical_period_signature(
    ink_lengths: list[float],
    gap_lengths: list[float],
) -> tuple[tuple[float, float], ...]:
    period = sum(ink_lengths) + sum(gap_lengths)
    pairs = [
        (ink_lengths[index] / period, gap_lengths[index] / period)
        for index in range(len(ink_lengths))
    ]
    variants: list[tuple[tuple[float, float], ...]] = []
    for source in (
        pairs,
        [
            (ink_lengths[index] / period, gap_lengths[(index - 1) % len(gap_lengths)] / period)
            for index in reversed(range(len(ink_lengths)))
        ],
    ):
        for shift in range(len(source)):
            rotated = source[shift:] + source[:shift]
            variants.append(tuple((round(ink, 8), round(gap, 8)) for ink, gap in rotated))
    return min(variants)


def _fit_ink_gap_period(path: _InkPath) -> _InkGapProfile | None:
    if len(path.tokens) < 5:
        return None
    fits: list[tuple[int, float, int, list[float], list[float]]] = []
    # The last Atom has no following gap.  Optionally trim the first Atom too;
    # CAD exports commonly lengthen or clip both boundary dashes.
    for start_trim in (0, 1):
        token_indices = list(range(start_trim, len(path.tokens) - 1))
        for token_count in range(1, min(12, len(token_indices) // 3) + 1):
            groups = [token_indices[offset::token_count] for offset in range(token_count)]
            if min(map(len, groups), default=0) < 3:
                continue
            ink_template = [median(path.tokens[index].length for index in group) for group in groups]
            gap_template = [median(path.gaps[index] for index in group) for group in groups]
            period = sum(ink_template) + sum(gap_template)
            if period <= EPS:
                continue
            component_errors: list[float] = []
            relative_errors: list[float] = []
            corner_anomalies: set[int] = set()
            for offset, group in enumerate(groups):
                for index in group:
                    index_component_errors: list[float] = []
                    index_relative_errors: list[float] = []
                    for actual, expected in (
                        (path.tokens[index].length, ink_template[offset]),
                        (path.gaps[index], gap_template[offset]),
                    ):
                        index_component_errors.append(abs(actual - expected) / period)
                        index_relative_errors.append(abs(actual - expected) / max(expected, EPS))
                    if (
                        max(index_component_errors) > 0.10
                        or max(index_relative_errors) > 0.25
                    ):
                        near_composite_corner = any(
                            0 <= neighbor < len(path.tokens)
                            and (
                                len(path.tokens[neighbor].atoms) > 1
                                # Some exporters keep a whole bent dash in a
                                # single Atom.  It is just as much a corner
                                # anomaly as a dash split over several Atoms.
                                or any(
                                    atom.line_segments > 1 or atom.curve_segments > 0
                                    for atom in path.tokens[neighbor].atoms
                                )
                            )
                            for neighbor in (index - 1, index, index + 1)
                        )
                        if not near_composite_corner:
                            corner_anomalies = set(token_indices)
                            break
                        corner_anomalies.add(index)
                        continue
                    component_errors.extend(index_component_errors)
                    relative_errors.extend(index_relative_errors)
                if len(corner_anomalies) == len(token_indices):
                    break
            maximum_corner_anomalies = min(
                INK_PERIOD_MAXIMUM_CORNER_ANOMALIES,
                max(1, math.floor(len(token_indices) * INK_PERIOD_MAXIMUM_CORNER_ANOMALY_RATIO)),
            )
            if len(corner_anomalies) > maximum_corner_anomalies:
                continue
            good_counts = [
                sum(index not in corner_anomalies for index in group)
                for group in groups
            ]
            if min(good_counts, default=0) < 3:
                continue
            maximum_component_error = max(component_errors, default=math.inf)
            maximum_relative_error = max(relative_errors, default=math.inf)
            if maximum_component_error > 0.10 or maximum_relative_error > 0.25:
                continue

            # Boundary dashes may be clipped or lengthened, but they still
            # have a definite phase in the learned cycle.  Validate that phase
            # before retaining them; otherwise a tiny collinear annotation at
            # either end could be silently swallowed by the trimmed fit.
            first_phase = (token_count - 1) % token_count
            last_phase = len(token_indices) % token_count
            boundary_checks: list[tuple[float, float]] = [
                (path.tokens[-1].length, ink_template[last_phase]),
            ]
            if start_trim:
                boundary_checks.append((path.tokens[0].length, ink_template[first_phase]))
                first_gap_error = abs(path.gaps[0] - gap_template[first_phase])
                if (
                    first_gap_error / period > 0.10
                    or first_gap_error / max(gap_template[first_phase], EPS) > 0.25
                ):
                    continue
            if any(
                not 0.35 <= actual / max(expected, EPS) <= 1.75
                for actual, expected in boundary_checks
            ):
                continue
            anomaly_ratio = len(corner_anomalies) / max(len(token_indices), 1)
            fit_error = maximum_component_error + maximum_relative_error * 0.1 + anomaly_ratio
            fits.append((
                token_count, fit_error, min(good_counts), ink_template, gap_template,
            ))
    if not fits:
        return None
    token_count, fit_error, repeat_count, ink_template, gap_template = min(
        fits, key=lambda item: (item[0], item[1]),
    )
    period = sum(ink_template) + sum(gap_template)
    return _InkGapProfile(
        path,
        token_count,
        period,
        _canonical_period_signature(ink_template, gap_template),
        repeat_count,
        fit_error,
    )


def _ink_gap_profiles_match(left: _InkGapProfile, right: _InkGapProfile) -> bool:
    if left.token_count != right.token_count:
        return False
    if not same_junction_style(left.path.tokens[0], right.path.tokens[0]):
        return False
    period_ratio = max(left.period_length, right.period_length) / max(
        min(left.period_length, right.period_length), EPS,
    )
    if period_ratio > 1.3:
        return False
    return max(
        abs(left_value - right_value)
        for left_pair, right_pair in zip(left.signature, right.signature)
        for left_value, right_value in zip(left_pair, right_pair)
    ) <= 0.08


def _ink_gap_path_hypotheses(
    atoms: list[Atom],
    clusters: list[list[Candidate]],
) -> list[SharedPathHypothesis]:
    protected_ids = {
        atom_id
        for cluster in clusters
        for candidate in cluster
        if len(candidate.members) > 1
        for atom_id in candidate.atom_ids
    }
    profiles = [
        profile
        for path in _open_ink_paths(atoms, protected_ids)
        if (profile := _fit_ink_gap_period(path)) is not None
    ]
    groups: list[list[_InkGapProfile]] = []
    for profile in profiles:
        for group in groups:
            if all(_ink_gap_profiles_match(profile, other) for other in group):
                group.append(profile)
                break
        else:
            groups.append([profile])

    proposals: list[SharedPathHypothesis] = []
    for group in groups:
        explained = {
            atom_id
            for profile in group
            for token in profile.path.tokens
            for atom_id in token.atom_ids
        }
        support_count = sum(profile.repeat_count for profile in group)
        fit_error = max(profile.fit_error for profile in group)
        token_count = group[0].token_count
        score = (
            len(explained), support_count,
            -js_round_nonnegative(fit_error * 10_000), -token_count,
        )
        proposals.append(SharedPathHypothesis(
            "ink_gap_period",
            [
                [atom_id for token in profile.path.tokens for atom_id in token.atom_ids]
                for profile in group
            ],
            set(explained),
            set(),
            explained,
            median(profile.period_length for profile in group),
            group[0].signature,
            support_count,
            fit_error,
            score,
        ))
    return proposals


SELF_CARRIED_MINIMUM_SUPPORT = 3
SELF_CARRIED_MAXIMUM_NEIGHBOR_RATIO_ERROR = 0.22
SELF_CARRIED_MAXIMUM_ENDPOINT_GAP_RATIO = 0.22


def _self_carried_fingerprint(candidate: Candidate) -> bool:
    """Recognize an elongated repeat unit that contains its own carrier ink.

    The numerical window describes the observed representation, not a named
    symbol: an open, elongated logical unit whose total ink is moderately
    longer than its end-to-end carrier.  Plain lines, closed symbols and dense
    hatch fragments do not enter this fallback.
    """
    fingerprint = candidate.fingerprint
    return (
        fingerprint["member_count"] == 1
        and fingerprint["closed_count"] == 0
        and fingerprint["curved_count"] == 0
        and 0.04 <= fingerprint["aspect_ratio"] <= 0.20
        and 1.06 <= fingerprint["normalized_length"] <= 1.30
        and any(
            member.paint_mode == "stroke"
            and not member.closed
            and member.line_segments >= 2
            for member in candidate.members
        )
    )


def _strict_self_carried_match(left: Candidate, right: Candidate) -> bool:
    scale_ratio = max(left.scale, right.scale) / max(min(left.scale, right.scale), EPS)
    if scale_ratio > 1.08:
        return False
    if abs(left.fingerprint["aspect_ratio"] - right.fingerprint["aspect_ratio"]) > 0.045:
        return False
    if abs(
        left.fingerprint["normalized_length"]
        - right.fingerprint["normalized_length"]
    ) > 0.08:
        return False
    return maximum_array_difference(
        left.fingerprint["radial_quantiles"],
        right.fingerprint["radial_quantiles"],
    ) <= 0.055


def _self_carried_endpoints(candidate: Candidate) -> tuple[Point, Point] | None:
    carriers = [
        member
        for member in candidate.members
        if member.paint_mode == "stroke" and not member.closed and len(member.points) >= 2
    ]
    if not carriers:
        return None
    carrier = max(
        carriers,
        key=lambda member: distance(member.points[0], member.points[-1]),
    )
    return carrier.points[0], carrier.points[-1]


def _extend_self_carried_path(
    atoms: list[Atom],
    seed_ids: set[int],
    spacing: float,
) -> tuple[set[int], list[list[int]]]:
    """Complete clipped/corner units by physical endpoint continuity.

    Shape similarity found the seeds.  This second step therefore uses no
    fingerprint voting: a residual Atom joins only through reciprocal nearest
    endpoints in an unbranched, style-consistent component containing at
    least three seed Atoms.  Long construction/grid lines are excluded.
    """
    seed_atoms = [atom for atom in atoms if atom.id in seed_ids and not atom.closed]
    if len(seed_atoms) < SELF_CARRIED_MINIMUM_SUPPORT:
        return set(), []
    pool = [
        atom
        for atom in atoms
        if (
            atom.paint_mode == "stroke"
            and not atom.closed
            and atom.length > EPS
            and atom.length <= spacing * 2.5
            and same_junction_style(seed_atoms[0], atom)
        )
    ]
    options: list[_PathLink] = []
    def outward(atom: Atom, side: int) -> Point:
        if atom.id in seed_ids:
            chord = sub(atom.points[-1], atom.points[0])
            return normalize(mul(chord, -1) if side == 0 else chord)
        return _endpoint_outward_direction(atom, side)

    for left_index, left in enumerate(pool):
        for right in pool[left_index + 1:]:
            for left_side, left_point in enumerate((left.points[0], left.points[-1])):
                for right_side, right_point in enumerate((right.points[0], right.points[-1])):
                    gap = distance(left_point, right_point)
                    if gap > spacing * SELF_CARRIED_MAXIMUM_ENDPOINT_GAP_RATIO:
                        continue
                    left_outward, right_outward = outward(left, left_side), outward(right, right_side)
                    if math.hypot(*left_outward) <= EPS or math.hypot(*right_outward) <= EPS:
                        continue
                    if gap <= EPS:
                        left_angle = right_angle = directed_angle_degrees(
                            left_outward, mul(right_outward, -1),
                        )
                    else:
                        gap_direction = normalize(sub(right_point, left_point))
                        left_angle = directed_angle_degrees(left_outward, gap_direction)
                        right_angle = directed_angle_degrees(
                            right_outward, mul(gap_direction, -1),
                        )
                    maximum_angle = (
                        100
                        if gap <= spacing * 0.05
                        else TOPOLOGY_REPAIR_MAXIMUM_LINK_ANGLE
                    )
                    if max(left_angle, right_angle) > maximum_angle:
                        continue
                    options.append(_PathLink(
                        left.id, left_side, right.id, right_side,
                        gap, gap / spacing + (left_angle + right_angle) / 180,
                    ))
    by_endpoint: dict[tuple[int, int], list[int]] = {}
    for index, option in enumerate(options):
        by_endpoint.setdefault((option.left_id, option.left_side), []).append(index)
        by_endpoint.setdefault((option.right_id, option.right_side), []).append(index)
    endpoint_best: dict[tuple[int, int], int] = {}
    for endpoint, indices in by_endpoint.items():
        ranked = sorted(indices, key=lambda index: options[index].gap)
        best_index = ranked[0]
        if len(ranked) > 1 and options[ranked[1]].gap <= max(options[best_index].gap, EPS) * 1.25:
            continue
        endpoint_best[endpoint] = best_index
    adjacency: dict[int, set[int]] = {atom.id: set() for atom in pool}
    for index, option in enumerate(options):
        if (
            endpoint_best.get((option.left_id, option.left_side)) == index
            and endpoint_best.get((option.right_id, option.right_side)) == index
        ):
            adjacency[option.left_id].add(option.right_id)
            adjacency[option.right_id].add(option.left_id)
    accepted: set[int] = set()
    components: list[list[int]] = []
    visited: set[int] = set()
    for atom in pool:
        if atom.id in visited:
            continue
        component: set[int] = set()
        pending = [atom.id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current])
        visited.update(component)
        if (
            len(component & seed_ids) >= SELF_CARRIED_MINIMUM_SUPPORT
            and max((len(adjacency[atom_id]) for atom_id in component), default=0) <= 2
        ):
            accepted.update(component)
            components.append(sorted(component))
    return accepted - seed_ids, components


def _extend_self_carried_command_boundaries(
    atoms: list[Atom],
    explained_ids: set[int],
    spacing: float,
) -> set[int]:
    """Adopt export-boundary units with both command and spatial evidence."""
    atom_by_id = {atom.id: atom for atom in atoms}
    style_seeds = [
        atom_by_id[atom_id]
        for atom_id in explained_ids
        if atom_id in atom_by_id and atom_by_id[atom_id].paint_mode == "stroke"
    ]
    if not style_seeds:
        return set()
    accepted = set(explained_ids)
    changed = True
    while changed:
        changed = False
        for atom in atoms:
            if (
                atom.id in accepted
                or atom.paint_mode != "stroke"
                or atom.length > spacing * 2.5
                or not same_junction_style(style_seeds[0], atom)
                or not ({atom.id - 1, atom.id + 1} & accepted)
            ):
                continue
            # The touching owner must be the same command neighbor that
            # supplied the sequence evidence.  Otherwise an unrelated grid
            # segment can be adjacent in command order while merely crossing
            # some distant part of a large linetype.
            owner_atoms = [
                atom_by_id[atom_id]
                for atom_id in ({atom.id - 1, atom.id + 1} & accepted)
                if atom_id in atom_by_id
            ]
            gap = min(
                distance(point, owner_point)
                for point in (atom.points[0], atom.points[-1])
                for owner in owner_atoms
                for owner_point in owner.points
            )
            if gap <= spacing * SELF_CARRIED_MAXIMUM_ENDPOINT_GAP_RATIO:
                accepted.add(atom.id)
                changed = True
    return accepted - explained_ids


def _self_carried_repeat_hypotheses(
    atoms: list[Atom],
    clusters: list[list[Candidate]],
) -> list[SharedPathHypothesis]:
    """Find periodic lines whose repeated Atom already includes carrier ink.

    Existing carrier/follower models deliberately expect separate connector
    Atoms.  CAD exports also occur in the inverse representation: every Atom
    is one complete ``dash + marker`` unit.  Here proximity alone is not
    enough—each physical endpoint must choose the same neighboring endpoint,
    and only unbranched components with at least three stations are accepted.
    """
    proposals: list[SharedPathHypothesis] = []
    seen_groups: set[tuple[tuple[int, ...], ...]] = set()
    for cluster in clusters:
        eligible = [candidate for candidate in cluster if _self_carried_fingerprint(candidate)]
        for seed in eligible:
            family = [
                candidate for candidate in eligible
                if _strict_self_carried_match(seed, candidate)
            ]
            family_key = tuple(sorted(tuple(candidate.atom_ids) for candidate in family))
            if len(family) < SELF_CARRIED_MINIMUM_SUPPORT or family_key in seen_groups:
                continue
            seen_groups.add(family_key)
            endpoints = [_self_carried_endpoints(candidate) for candidate in family]
            if any(item is None for item in endpoints):
                continue
            nearest_distances = [
                min(
                    distance(candidate.center, other.center)
                    for other in family
                    if other is not candidate
                )
                for candidate in family
            ]
            spacing = median(nearest_distances)
            if spacing <= EPS:
                continue

            edge_options: list[tuple[int, int, int, int, float, float]] = []
            for left_index, left in enumerate(family):
                left_endpoints = endpoints[left_index]
                assert left_endpoints is not None
                for right_index in range(left_index + 1, len(family)):
                    right = family[right_index]
                    center_gap = distance(left.center, right.center)
                    relative_error = abs(center_gap - spacing) / spacing
                    if relative_error > SELF_CARRIED_MAXIMUM_NEIGHBOR_RATIO_ERROR:
                        continue
                    right_endpoints = endpoints[right_index]
                    assert right_endpoints is not None
                    endpoint_gap, left_side, right_side = min(
                        (
                            distance(left_point, right_point), li, ri
                        )
                        for li, left_point in enumerate(left_endpoints)
                        for ri, right_point in enumerate(right_endpoints)
                    )
                    if endpoint_gap > spacing * SELF_CARRIED_MAXIMUM_ENDPOINT_GAP_RATIO:
                        continue
                    edge_options.append((
                        left_index, left_side, right_index, right_side,
                        endpoint_gap, relative_error,
                    ))

            by_endpoint: dict[tuple[int, int], list[int]] = {}
            for option_index, option in enumerate(edge_options):
                left_index, left_side, right_index, right_side, _, _ = option
                by_endpoint.setdefault((left_index, left_side), []).append(option_index)
                by_endpoint.setdefault((right_index, right_side), []).append(option_index)
            endpoint_best: dict[tuple[int, int], int] = {}
            for endpoint, option_indices in by_endpoint.items():
                ranked = sorted(
                    option_indices,
                    key=lambda index: (edge_options[index][4], edge_options[index][5]),
                )
                best_index = ranked[0]
                if len(ranked) > 1:
                    best, second = edge_options[best_index], edge_options[ranked[1]]
                    if (
                        second[4] <= max(best[4], EPS) * 1.25
                        and second[5] - best[5] < 0.08
                    ):
                        continue
                endpoint_best[endpoint] = best_index

            adjacency: dict[int, set[int]] = {index: set() for index in range(len(family))}
            accepted_errors: list[float] = []
            for option_index, option in enumerate(edge_options):
                left_index, left_side, right_index, right_side, endpoint_gap, relative_error = option
                if (
                    endpoint_best.get((left_index, left_side)) == option_index
                    and endpoint_best.get((right_index, right_side)) == option_index
                ):
                    adjacency[left_index].add(right_index)
                    adjacency[right_index].add(left_index)
                    accepted_errors.append(relative_error + endpoint_gap / spacing)

            components: list[list[int]] = []
            visited: set[int] = set()
            for start in range(len(family)):
                if start in visited or not adjacency[start]:
                    continue
                component: set[int] = set()
                pending = [start]
                while pending:
                    current = pending.pop()
                    if current in component:
                        continue
                    component.add(current)
                    pending.extend(adjacency[current])
                visited.update(component)
                if (
                    len(component) >= SELF_CARRIED_MINIMUM_SUPPORT
                    and max(len(adjacency[index]) for index in component) <= 2
                ):
                    components.append(sorted(component))
            station_indices = sorted({index for component in components for index in component})
            if len(station_indices) < SELF_CARRIED_MINIMUM_SUPPORT:
                continue
            station_candidates = [family[index] for index in station_indices]
            repeated = {
                atom_id
                for candidate in station_candidates
                for atom_id in candidate.atom_ids
            }
            boundary_ids, physical_instances = _extend_self_carried_path(
                atoms, repeated, spacing,
            )
            command_boundary_ids = _extend_self_carried_command_boundaries(
                atoms, repeated | boundary_ids, spacing,
            )
            boundary_ids.update(command_boundary_ids)
            explained = repeated | boundary_ids
            if not physical_instances:
                physical_instances = [
                    sorted(
                        atom_id
                        for index in component
                        for atom_id in family[index].atom_ids
                    )
                    for component in components
                ]
            motif_instances = [list(candidate.atom_ids) for candidate in station_candidates]
            fit_error = max(accepted_errors, default=0.0)
            proposals.append(SharedPathHypothesis(
                "self_carried_repeat",
                physical_instances,
                set(repeated),
                set(boundary_ids),
                set(explained),
                spacing,
                None,
                len(station_candidates),
                fit_error,
                (
                    len(explained), len(station_candidates),
                    -js_round_nonnegative(fit_error * 10_000),
                    -len(components),
                ),
                motif_instances,
            ))
    return proposals


def _double_dot_period_hypotheses(atoms: list[Atom]) -> list[SharedPathHypothesis]:
    """Recognize ``carrier, point, point`` repeated exactly twice.

    Zero-length round-cap strokes are real painted dots, but are intentionally
    excluded from ordinary ink-path fitting.  Two copies are accepted here
    only with the complete seven-command context: matching point-pair pitch,
    a carrier on both outer sides, and one carrier between the pairs.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    proposals: list[SharedPathHypothesis] = []
    for first_id in sorted(atom_by_id):
        ids = list(range(first_id, first_id + 7))
        if any(atom_id not in atom_by_id for atom_id in ids):
            continue
        carriers = [atom_by_id[ids[index]] for index in (0, 3, 6)]
        points = [atom_by_id[ids[index]] for index in (1, 2, 4, 5)]
        if not all(
            atom.paint_mode == "stroke"
            and atom.length > max(EPS, abs(atom.line_width) * 2)
            and not atom.closed
            for atom in carriers
        ):
            continue
        if not all(
            atom.paint_mode == "stroke"
            and atom.length <= max(EPS, abs(atom.line_width) * 0.05)
            and atom.line_cap == carriers[0].line_cap
            for atom in points
        ):
            continue
        if not all(same_junction_style(carriers[0], atom) for atom in [*carriers[1:], *points]):
            continue
        pair_spacings = [distance(points[0].center, points[1].center), distance(points[2].center, points[3].center)]
        dot_spacing = median(pair_spacings)
        pair_centers = [
            (mean((points[index].center[0], points[index + 1].center[0])),
             mean((points[index].center[1], points[index + 1].center[1])))
            for index in (0, 2)
        ]
        period = distance(pair_centers[0], pair_centers[1])
        if (
            dot_spacing <= EPS
            or period < dot_spacing * 5
            or max(abs(value - dot_spacing) for value in pair_spacings) > period * 0.03
        ):
            continue
        sequence = [carriers[0], points[0], points[1], carriers[1], points[2], points[3], carriers[2]]
        sequence_gaps = [
            min(
                distance(left_point, right_point)
                for left_point in left.points
                for right_point in right.points
            )
            for left, right in zip(sequence, sequence[1:])
        ]
        if max(sequence_gaps) > period * 0.16:
            continue
        direction = sub(pair_centers[1], pair_centers[0])
        if any(
            acute_angle_degrees(atom_chord_direction(carrier), direction) > 12
            for carrier in carriers[:2]
        ):
            continue
        if not 0.35 <= carriers[1].length / period <= 0.90:
            continue
        explained = set(ids)
        fit_error = max(
            max(abs(value - dot_spacing) for value in pair_spacings) / period,
            max(sequence_gaps) / period,
        )
        proposals.append(SharedPathHypothesis(
            "double_dot_period",
            [ids],
            {points[0].id, points[1].id, points[2].id, points[3].id},
            {carrier.id for carrier in carriers},
            explained,
            period,
            _canonical_period_signature([0.0, 0.0], [dot_spacing, period - dot_spacing]),
            2,
            fit_error,
            (len(explained), 2, -js_round_nonnegative(fit_error * 10_000), -2),
            [[points[0].id, points[1].id], [points[2].id, points[3].id]],
        ))
    return proposals


def build_shared_reference_path_hypotheses(
    atoms: list[Atom],
    clusters: list[list[Candidate]],
) -> list[SharedPathHypothesis]:
    """Last-resort complete-line proposals; never rewrites a proven result."""
    proposals = [
        *_attached_path_hypotheses(atoms, clusters),
        *_ink_gap_path_hypotheses(atoms, clusters),
        *_self_carried_repeat_hypotheses(atoms, clusters),
        *_double_dot_period_hypotheses(atoms),
    ]
    return _resolve_shared_path_proposals(proposals)


def _resolve_shared_path_proposals(
    proposals: list[SharedPathHypothesis],
) -> list[SharedPathHypothesis]:
    """Collapse equivalent fits before rejecting genuinely ambiguous overlap.

    A branched or multi-run self-carried line can be reconstructed from more
    than one seed candidate.  Those fits may differ only in whether a clipped
    terminal Atom is recorded as a motif station or boundary ink while still
    explaining exactly the same painted Atoms.  Treating the duplicate fits
    as competing owners used to discard all of them, leaving an otherwise
    strongly repeated linetype classified as residual geometry.
    """
    equivalent: dict[tuple[str, frozenset[int]], SharedPathHypothesis] = {}
    for proposal in proposals:
        key = (proposal.relation_kind, frozenset(proposal.explained_atom_ids))
        previous = equivalent.get(key)
        if previous is None or proposal.score > previous.score:
            equivalent[key] = proposal
    proposals = list(equivalent.values())
    owners: dict[int, set[int]] = {}
    for index, proposal in enumerate(proposals):
        for atom_id in proposal.explained_atom_ids:
            owners.setdefault(atom_id, set()).add(index)
    conflicting = {
        index
        for indices in owners.values()
        if len(indices) > 1
        for index in indices
    }
    return sorted(
        (proposal for index, proposal in enumerate(proposals) if index not in conflicting),
        key=lambda proposal: proposal.score,
        reverse=True,
    )


PatternHypothesis = Hypothesis | NetworkHypothesis | TwoInstanceHypothesis | SharedPathHypothesis


def build_unknown_pattern_hypotheses(
    atoms: list[Atom],
    candidate_cache: dict[tuple[int, ...], Candidate] | None = None,
    standard_hypothesis_cache: dict[int, Any] | None = None,
) -> tuple[list[Candidate], list[list[Candidate]], list[PatternHypothesis]]:
    candidates = generate_motif_candidates(atoms, candidate_cache)
    clusters = cluster_repeated_candidates(candidates)
    hypotheses: list[PatternHypothesis] = []
    for cluster in clusters:
        # Preserve the proven single-chain behavior.  The network model is a
        # fallback only when the global chain is invalid (multiple instances,
        # a branch, or a turn-induced spacing failure).
        hypothesis: PatternHypothesis | None = build_hypothesis(cluster, atoms)
        if hypothesis is None:
            hypothesis = build_network_hypothesis(cluster, atoms)
        if standard_hypothesis_cache is not None:
            standard_hypothesis_cache[id(cluster)] = hypothesis
        if hypothesis is not None:
            hypotheses.append(hypothesis)
    hypotheses.sort(key=lambda hypothesis: hypothesis.score, reverse=True)
    return candidates, clusters, hypotheses


def hypothesis_summary(hypothesis: PatternHypothesis) -> dict[str, Any]:
    if isinstance(hypothesis, SharedPathHypothesis):
        return {
            "schema_version": 2,
            "model": hypothesis.relation_kind,
            "instances": hypothesis.instances,
            "repeated_atom_ids": sorted(hypothesis.repeated_atom_ids),
            "reference_atom_ids": sorted(hypothesis.reference_atom_ids),
            "explained_atom_ids": sorted(hypothesis.explained_atom_ids),
            "period_length": hypothesis.period_length,
            "period_signature": (
                [list(pair) for pair in hypothesis.period_signature]
                if hypothesis.period_signature is not None else None
            ),
            "support_count": hypothesis.support_count,
            "fit_error": hypothesis.fit_error,
            "score_tuple": list(hypothesis.score),
        }
    if isinstance(hypothesis, TwoInstanceHypothesis):
        return {
            "schema_version": 2,
            "model": "strict_two_instance_chain",
            "motif_member_count": hypothesis.motif_member_count,
            "motif_station_count": 2,
            "motif_atom_ids": sorted(hypothesis.motif_atom_ids),
            "middle_connector_atom_ids": sorted(hypothesis.middle_bridge.atom_ids),
            "endpoint_connector_atom_ids": sorted((
                hypothesis.left_extension.atom.id,
                hypothesis.right_extension.atom.id,
            )),
            "explained_atom_ids": sorted(hypothesis.explained_atom_ids),
            # Two centers provide a measuring scale, not enough samples to
            # claim a stable period.  Do not publish a misleading zero CV.
            "candidate_center_distance": hypothesis.spacing,
            "score_tuple": list(hypothesis.score),
        }
    if isinstance(hypothesis, NetworkHypothesis):
        bridge_ids = sorted(
            atom_id
            for item in hypothesis.bridge_followers
            for atom_id in item.atom_ids
        )
        maximum_all_edge_error = max(
            (
                abs(edge.length - hypothesis.spacing) / max(hypothesis.spacing, EPS)
                for edge in hypothesis.network_edges
            ),
            default=0.0,
        )
        return {
            "schema_version": 2,
            "model": "multi_carrier_network",
            "motif_member_count": hypothesis.motif_member_count,
            "motif_station_count": hypothesis.station_count,
            "motif_atom_ids": sorted(hypothesis.motif_atom_ids),
            "core_follower_atom_ids": bridge_ids,
            "bridge_follower_atom_ids": bridge_ids,
            "endpoint_follower_atom_ids": sorted(atom.id for atom in hypothesis.endpoint_followers),
            "explained_atom_ids": sorted(hypothesis.explained_atom_ids),
            "coarse_spacing": hypothesis.spacing,
            "coarse_spacing_cv": hypothesis.spacing_cv,
            "maximum_relative_coarse_spacing_error": hypothesis.maximum_relative_spacing_error,
            "refined_spacing": hypothesis.spacing,
            "refined_spacing_cv": hypothesis.spacing_cv,
            "spacing": hypothesis.spacing,
            "spacing_cv": hypothesis.spacing_cv,
            "maximum_relative_spacing_error": maximum_all_edge_error,
            "maximum_learned_edge_relative_error": hypothesis.maximum_relative_spacing_error,
            "relaxed_terminal_edge_count": hypothesis.relaxed_terminal_edge_count,
            "network_edge_count": len(hypothesis.network_edges),
            "multi_atom_bridge_count": sum(len(item.fragments) > 1 for item in hypothesis.bridge_followers),
            "score_tuple": list(hypothesis.score),
        }
    summary = {
        "schema_version": 2,
        "model": "single_carrier_chain",
        "motif_member_count": hypothesis.motif_member_count,
        "motif_station_count": len(hypothesis.cluster),
        "motif_atom_ids": sorted(hypothesis.motif_atom_ids),
        "core_follower_atom_ids": sorted(item.atom.id for item in hypothesis.core_followers),
        "endpoint_follower_atom_ids": sorted(item.atom.id for item in hypothesis.endpoint_followers),
        "explained_atom_ids": sorted(hypothesis.explained_atom_ids),
        "coarse_spacing": hypothesis.spacing,
        "coarse_spacing_cv": hypothesis.spacing_cv,
        "maximum_relative_coarse_spacing_error": hypothesis.maximum_relative_spacing_error,
        "refined_spacing": hypothesis.refined_spacing,
        "refined_spacing_cv": hypothesis.refined_spacing_cv,
        "score_tuple": list(hypothesis.score),
    }
    edge_multipliers = _period_edge_multipliers(
        hypothesis.chain.edge_lengths, hypothesis.spacing,
    ) or [1] * len(hypothesis.chain.edge_lengths)
    if any(multiplier > 1 for multiplier in edge_multipliers):
        summary["skipped_period_edge_count"] = sum(
            multiplier > 1 for multiplier in edge_multipliers
        )
        summary["implied_missing_station_count"] = sum(
            multiplier - 1 for multiplier in edge_multipliers
        )
        summary["skipped_period_bridge_atom_ids"] = sorted(
            item.atom.id
            for item in hypothesis.core_followers
            if item.atom.id not in hypothesis.motif_atom_ids
            and item.atom.length > hypothesis.spacing * 1.35
        )
    return summary


@dataclass
class PatternType:
    type_id: str
    kind: str
    atom_ids: set[int]
    hypothesis: PatternHypothesis | None = None


LINE_TYPE_SIGNATURE_SCHEMA_VERSION = 2
LINE_TYPE_SIGNATURE_MAXIMUM_SCALE_RATIO = 1.3
LINE_TYPE_SIGNATURE_MAXIMUM_PERIOD_RATIO = 1.3


def _aggregate_candidate_fingerprint(candidates: list[Candidate]) -> dict[str, Any]:
    """Return one deterministic prototype for a mutually similar cluster."""
    scalar_keys = (
        "log_scale", "aspect_ratio", "normalized_length",
    )
    array_keys = (
        "member_length_ratios", "center_distances",
        "direction_angles", "radial_quantiles",
    )
    first = candidates[0].fingerprint
    result: dict[str, Any] = {
        key: int(round(median(candidate.fingerprint[key] for candidate in candidates)))
        for key in ("member_count", "closed_count", "curved_count", "filled_count")
    }
    result.update({
        key: median(candidate.fingerprint[key] for candidate in candidates)
        for key in scalar_keys
    })
    for key in array_keys:
        size = len(first[key])
        result[key] = [
            median(candidate.fingerprint[key][index] for candidate in candidates)
            for index in range(size)
        ]
    return result


def _period_ink_fingerprint(atoms: list[Atom], period: float) -> dict[str, Any]:
    samples = [point for atom in atoms for point in atom.samples]
    center = (
        mean(point[0] for point in samples),
        mean(point[1] for point in samples),
    )
    scale = max(EPS, 2 * max(distance(point, center) for point in samples))
    _, aspect_ratio = principal_frame(samples, center)
    total_length = sum(atom.length for atom in atoms)
    return {
        "aspect_ratio": aspect_ratio,
        "normalized_length": total_length / scale,
        "radial_quantiles": radial_quantiles(samples, center, scale),
        "scale_to_period": scale / period,
        "length_to_period": total_length / period,
    }


def _period_ink_fingerprints_match(
    left: dict[str, Any], right: dict[str, Any], maximum_ratio: float = 1.3,
) -> bool:
    def ratio(a: float, b: float) -> float:
        return max(1.0, max(a, b) / max(min(a, b), EPS))

    return (
        abs(left["aspect_ratio"] - right["aspect_ratio"]) <= 0.2
        and abs(left["normalized_length"] - right["normalized_length"])
        <= 0.3 * max(1.0, min(left["normalized_length"], right["normalized_length"]))
        and maximum_array_difference(
            left["radial_quantiles"], right["radial_quantiles"],
        ) <= 0.18
        and ratio(left["scale_to_period"], right["scale_to_period"]) <= maximum_ratio
        and ratio(left["length_to_period"], right["length_to_period"]) <= maximum_ratio
    )


def _period_command_signature(
    motif_candidates: list[Candidate],
    assigned_atom_ids: set[int],
    atom_by_id: dict[int, Atom],
    period: float,
) -> dict[str, Any] | None:
    """Describe every painted token between consecutive motif stations.

    The motif detector may select only the long carrier in a decorated period
    and treat the intervening symbols as followers.  Recording paint-order
    phases prevents that decorated linetype from acquiring the same global
    identity as a plain repeated carrier.  Two complete periods are required;
    otherwise the extra evidence is omitted instead of guessed.
    """
    if period <= EPS or len(motif_candidates) < 3:
        return None
    if any(len(candidate.atom_ids) != 1 for candidate in motif_candidates):
        return None
    anchors = sorted(candidate.atom_ids[0] for candidate in motif_candidates)
    if len(set(anchors)) != len(anchors):
        return None
    intervals = [
        [
            atom_by_id[atom_id]
            for atom_id in sorted(assigned_atom_ids)
            if left <= atom_id < right and atom_id in atom_by_id
        ]
        for left, right in zip(anchors, anchors[1:])
    ]
    if not intervals or any(not interval for interval in intervals):
        return None
    def aggregate_ink(
        period_intervals: list[list[Atom]],
    ) -> dict[str, Any] | None:
        examples = [_period_ink_fingerprint(interval, period) for interval in period_intervals]
        if not examples or any(
            not _period_ink_fingerprints_match(examples[0], example)
            for example in examples[1:]
        ):
            return None
        result = {
            key: median(example[key] for example in examples)
            for key in ("aspect_ratio", "normalized_length", "scale_to_period", "length_to_period")
        }
        result["radial_quantiles"] = [
            median(example["radial_quantiles"][index] for example in examples)
            for index in range(len(examples[0]["radial_quantiles"]))
        ]
        return result

    period_ink_fingerprint = aggregate_ink(intervals)
    if period_ink_fingerprint is None:
        return None

    token_count = len(intervals[0])
    if any(len(interval) != token_count for interval in intervals):
        return {"ink_fingerprint": period_ink_fingerprint}

    phases: list[dict[str, Any]] = []
    for phase_index in range(token_count):
        examples = [interval[phase_index] for interval in intervals]
        candidates = [make_candidate(index, [atom]) for index, atom in enumerate(examples)]
        reference = candidates[0]
        if any(
            not same_junction_style(examples[0], atom)
            or not fingerprints_match(reference.fingerprint, candidate.fingerprint)
            for atom, candidate in zip(examples[1:], candidates[1:])
        ):
            return {"ink_fingerprint": period_ink_fingerprint}
        phases.append({
            "fingerprint": _aggregate_candidate_fingerprint(candidates),
            "scale_to_period": median(atom.scale for atom in examples) / period,
            "length_to_period": median(atom.length for atom in examples) / period,
            "line_cap": int(round(median(atom.line_cap for atom in examples))),
        })

    # The detector may anchor the same complete period on a different token in
    # another Group.  Rotate to a deterministic token and recompute the whole
    # period ink from that same command phase, so a cyclic shift cannot change
    # either the token sequence or its aggregate spatial fingerprint.
    canonical_phase = min(range(token_count), key=lambda index: (
        phases[index]["scale_to_period"],
        phases[index]["length_to_period"],
        phases[index]["fingerprint"]["aspect_ratio"],
        phases[index]["fingerprint"]["normalized_length"],
        tuple(phases[index]["fingerprint"]["radial_quantiles"]),
    ))
    if canonical_phase:
        phase_offset = intervals[0][canonical_phase].id - anchors[0]
        canonical_anchors = [anchor + phase_offset for anchor in anchors]
        if all(
            atom_id in assigned_atom_ids and atom_id in atom_by_id
            for atom_id in canonical_anchors
        ):
            canonical_intervals = [
                [
                    atom_by_id[atom_id]
                    for atom_id in sorted(assigned_atom_ids)
                    if left <= atom_id < right and atom_id in atom_by_id
                ]
                for left, right in zip(canonical_anchors, canonical_anchors[1:])
            ]
            canonical_ink = aggregate_ink(canonical_intervals)
            if canonical_ink is not None:
                period_ink_fingerprint = canonical_ink
        phases = phases[canonical_phase:] + phases[:canonical_phase]
    return {
        "ink_fingerprint": period_ink_fingerprint,
        "tokens": phases,
    }


def _stable_command_station_period(candidates: list[Candidate]) -> float | None:
    """Return the physical cadence of a fixed-stride command motif chain.

    Carrier refinement follows the full routed ink and can overestimate the
    repeat period around bends.  For a stable command sequence, consecutive
    motif centers are the representation-independent station cadence used by
    both ordinary and reconstructed composite patterns.
    """
    if (
        len(candidates) < 3
        or any(len(candidate.atom_ids) != 1 for candidate in candidates)
    ):
        return None
    ordered = sorted(candidates, key=lambda candidate: candidate.atom_ids[0])
    anchors = [candidate.atom_ids[0] for candidate in ordered]
    strides = [right - left for left, right in zip(anchors, anchors[1:])]
    if not strides or min(strides) < 1 or len(set(strides)) != 1:
        return None
    distances = [
        distance(left.center, right.center)
        for left, right in zip(ordered, ordered[1:])
    ]
    station_period = median(distances)
    if station_period <= EPS:
        return None
    if max(distances) / max(min(distances), EPS) > 1.3:
        return None
    return station_period


def _command_normalized_motif_signature(
    candidates: list[Candidate],
    assigned_atom_ids: set[int],
    atom_by_id: dict[int, Atom],
    fallback_period: float,
    model: str,
) -> dict[str, Any] | None:
    command_period = _stable_command_station_period(candidates)
    period = command_period or fallback_period
    period_command = _period_command_signature(
        candidates, assigned_atom_ids, atom_by_id, period,
    )
    return _motif_line_type_signature(
        candidates, period, model, period_command,
    )


def _motif_line_type_signature(
    candidates: list[Candidate], period: float, model: str,
    period_command: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not candidates or period <= EPS:
        return None
    motif_scale = median(candidate.scale for candidate in candidates)
    reference_atoms = [atom for candidate in candidates for atom in candidate.members]
    line_caps = [atom.line_cap for atom in reference_atoms]
    line_widths = [abs(atom.line_width) for atom in reference_atoms]
    signature = {
        "schema_version": LINE_TYPE_SIGNATURE_SCHEMA_VERSION,
        "family": "motif_periodic",
        "source_model": model,
        "support_count": len(candidates),
        "motif_fingerprint": _aggregate_candidate_fingerprint(candidates),
        "period_to_motif_scale": period / max(motif_scale, EPS),
        # Absolute values are retained so the default matcher can distinguish
        # different linetype scales.  Geometry remains position/rotation free.
        "absolute_motif_scale": motif_scale,
        "absolute_period": period,
        "line_cap": int(round(median(line_caps))) if line_caps else None,
        "line_width_to_motif_scale": (
            median(line_widths) / max(motif_scale, EPS) if line_widths else None
        ),
    }
    if period_command:
        signature["period_ink_fingerprint"] = period_command["ink_fingerprint"]
        if period_command.get("tokens"):
            signature["period_command_sequence"] = period_command["tokens"]
            signature["period_command_token_count"] = len(period_command["tokens"])
    return signature


def line_type_signature(
    pattern_type: PatternType,
    atom_by_id: dict[int, Atom],
) -> dict[str, Any] | None:
    """Build a reusable, group-independent identity for one proven line type.

    Residual geometry is intentionally unsigned: it has not yet supplied
    enough evidence to claim a repeat period, so globally registering it as a
    linetype would turn local leftovers into false cross-Group matches.
    """
    hypothesis = pattern_type.hypothesis
    if isinstance(hypothesis, Hypothesis):
        return _command_normalized_motif_signature(
            hypothesis.cluster,
            pattern_type.atom_ids,
            atom_by_id,
            hypothesis.refined_spacing,
            "single_carrier_chain",
        )
    if isinstance(hypothesis, NetworkHypothesis):
        return _command_normalized_motif_signature(
            hypothesis.cluster,
            pattern_type.atom_ids,
            atom_by_id,
            hypothesis.spacing,
            "multi_carrier_network",
        )
    if isinstance(hypothesis, TwoInstanceHypothesis):
        return _command_normalized_motif_signature(
            hypothesis.cluster,
            pattern_type.atom_ids,
            atom_by_id,
            hypothesis.spacing,
            "strict_two_instance_chain",
        )
    if not isinstance(hypothesis, SharedPathHypothesis):
        return None
    if hypothesis.relation_kind == "attached_repeat":
        instances = [
            [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
            for atom_ids in hypothesis.instances[1:]
        ]
        candidates = [
            make_candidate(index, members)
            for index, members in enumerate(instances)
            if members
        ]
        return _motif_line_type_signature(
            candidates,
            hypothesis.period_length or 0.0,
            "shared_path/attached_repeat",
        )
    if hypothesis.relation_kind == "self_carried_repeat" and hypothesis.motif_instances:
        instances = [
            [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
            for atom_ids in hypothesis.motif_instances
        ]
        candidates = [
            make_candidate(index, members)
            for index, members in enumerate(instances)
            if members
        ]
        return _motif_line_type_signature(
            candidates,
            hypothesis.period_length or 0.0,
            "shared_path/self_carried_repeat",
        )
    if hypothesis.relation_kind == "co_phased_modules" and hypothesis.motif_instances:
        instances = [
            [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
            for atom_ids in hypothesis.motif_instances
        ]
        # A reconstructed block may contain every token in the period, while a
        # normal Hypothesis stores one representative token plus the same
        # command interval.  Use one deterministic phase as the anchor and
        # serialize the complete interval so both models share one identity.
        candidates = [
            make_candidate(index, [members[0]])
            for index, members in enumerate(instances)
            if members
        ]
        return _command_normalized_motif_signature(
            candidates,
            pattern_type.atom_ids,
            atom_by_id,
            hypothesis.period_length or 0.0,
            "shared_path/co_phased_modules",
        )
    if hypothesis.relation_kind in ("ink_gap_period", "double_dot_period") and hypothesis.period_signature:
        reference_atoms = [
            atom_by_id[atom_id]
            for atom_id in hypothesis.explained_atom_ids
            if atom_id in atom_by_id
        ]
        return {
            "schema_version": LINE_TYPE_SIGNATURE_SCHEMA_VERSION,
            "family": "ink_gap_periodic",
            "source_model": f"shared_path/{hypothesis.relation_kind}",
            "support_count": hypothesis.support_count,
            "period_signature": [list(pair) for pair in hypothesis.period_signature],
            "absolute_period": hypothesis.period_length,
            "line_cap": (
                int(round(median(atom.line_cap for atom in reference_atoms)))
                if reference_atoms else None
            ),
            "line_width_to_period": (
                median(abs(atom.line_width) for atom in reference_atoms)
                / max(hypothesis.period_length or 0.0, EPS)
                if reference_atoms else None
            ),
        }
    return None


def _compare_period_command_sequences(
    left: list[dict[str, Any]] | None,
    right: list[dict[str, Any]] | None,
    maximum_scale_ratio: float,
) -> dict[str, Any]:
    if left is None and right is None:
        return {"matched": True, "error": 0.0, "reason": "not_available"}
    if left is None or right is None:
        return {"matched": True, "error": 0.0, "reason": "not_available_on_one_side"}
    if len(left) != len(right):
        return {"matched": False, "error": 1.0, "reason": "token_count_mismatch"}

    def value_ratio(a: float, b: float) -> float:
        return max(1.0, max(a, b) / max(min(a, b), EPS))

    def token_result(
        left_token: dict[str, Any], right_token: dict[str, Any],
    ) -> tuple[bool, float]:
        left_fp = dict(left_token["fingerprint"])
        right_fp = dict(right_token["fingerprint"])
        left_fp["log_scale"] = right_fp["log_scale"] = 0.0
        shape_match = fingerprints_match(left_fp, right_fp)
        cap_match = left_token.get("line_cap") == right_token.get("line_cap")
        scale_ratio = value_ratio(
            left_token["scale_to_period"], right_token["scale_to_period"],
        )
        length_ratio = value_ratio(
            left_token["length_to_period"], right_token["length_to_period"],
        )
        ratio_denominator = max(math.log(maximum_scale_ratio), EPS)
        error = mean((
            0.0 if shape_match else 1.0,
            0.0 if cap_match else 1.0,
            min(1.0, math.log(scale_ratio) / ratio_denominator),
            min(1.0, math.log(length_ratio) / ratio_denominator),
        ))
        return (
            shape_match
            and cap_match
            and scale_ratio <= maximum_scale_ratio
            and length_ratio <= maximum_scale_ratio,
            error,
        )

    alignments: list[tuple[bool, float, bool, int]] = []
    for reversed_order, ordered_right in (
        (False, right),
        (True, list(reversed(right))),
    ):
        for offset in range(len(right)):
            rotated = ordered_right[offset:] + ordered_right[:offset]
            token_results = [
                token_result(left_token, right_token)
                for left_token, right_token in zip(left, rotated)
            ]
            alignments.append((
                all(matched for matched, _ in token_results),
                mean(error for _, error in token_results),
                reversed_order,
                offset,
            ))
    matched, error, reversed_order, offset = min(
        alignments, key=lambda item: (not item[0], item[1], item[2], item[3]),
    )
    return {
        "matched": matched,
        "error": error,
        "reason": "matched" if matched else "token_mismatch",
        "token_count": len(left),
        "reversed": reversed_order,
        "cyclic_offset": offset,
    }


def _compare_period_ink_fingerprints(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    maximum_scale_ratio: float,
) -> dict[str, Any]:
    if left is None and right is None:
        return {"matched": True, "error": 0.0, "reason": "not_available"}
    if left is None or right is None:
        return {"matched": True, "error": 0.0, "reason": "not_available_on_one_side"}

    def ratio(a: float, b: float) -> float:
        return max(1.0, max(a, b) / max(min(a, b), EPS))

    scale_ratio = ratio(left["scale_to_period"], right["scale_to_period"])
    length_ratio = ratio(left["length_to_period"], right["length_to_period"])
    aspect_error = abs(left["aspect_ratio"] - right["aspect_ratio"])
    normalized_length_error = abs(
        left["normalized_length"] - right["normalized_length"],
    )
    radial_error = maximum_array_difference(
        left["radial_quantiles"], right["radial_quantiles"],
    )
    matched = _period_ink_fingerprints_match(
        left, right, maximum_scale_ratio,
    )
    ratio_denominator = max(math.log(maximum_scale_ratio), EPS)
    error = mean((
        min(1.0, aspect_error / 0.2),
        min(1.0, normalized_length_error / 0.3),
        min(1.0, radial_error / 0.18),
        min(1.0, math.log(scale_ratio) / ratio_denominator),
        min(1.0, math.log(length_ratio) / ratio_denominator),
    ))
    return {
        "matched": matched,
        "error": error,
        "reason": "matched" if matched else "period_ink_mismatch",
        "scale_ratio": scale_ratio,
        "length_ratio": length_ratio,
        "aspect_error": aspect_error,
        "normalized_length_error": normalized_length_error,
        "maximum_radial_quantile_error": radial_error,
    }


def compare_line_type_signatures(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
    *,
    maximum_scale_ratio: float = LINE_TYPE_SIGNATURE_MAXIMUM_SCALE_RATIO,
    maximum_period_ratio: float = LINE_TYPE_SIGNATURE_MAXIMUM_PERIOD_RATIO,
) -> dict[str, Any]:
    """Compare signatures from different Groups and report auditable errors."""
    if left is None or right is None:
        return {"matched": False, "similarity": 0.0, "reason": "missing_signature"}
    if left.get("schema_version") != right.get("schema_version"):
        return {"matched": False, "similarity": 0.0, "reason": "schema_mismatch"}
    if left.get("family") != right.get("family"):
        return {"matched": False, "similarity": 0.0, "reason": "family_mismatch"}

    def ratio(a: float, b: float) -> float:
        return max(a, b) / max(min(a, b), EPS)

    family = left["family"]
    if family == "motif_periodic":
        left_fp = dict(left["motif_fingerprint"])
        right_fp = dict(right["motif_fingerprint"])
        scale_ratio = ratio(left["absolute_motif_scale"], right["absolute_motif_scale"])
        period_ratio = ratio(left["absolute_period"], right["absolute_period"])
        normalized_period_ratio = ratio(
            left["period_to_motif_scale"], right["period_to_motif_scale"],
        )
        # Shape comparison is already translation/rotation/reversal invariant;
        # neutralize log scale because scale is checked explicitly above.
        left_fp["log_scale"] = right_fp["log_scale"] = 0.0
        shape_match = fingerprints_match(left_fp, right_fp)
        cap_match = left.get("line_cap") == right.get("line_cap")
        command_sequence = _compare_period_command_sequences(
            left.get("period_command_sequence"),
            right.get("period_command_sequence"),
            maximum_scale_ratio,
        )
        period_ink = _compare_period_ink_fingerprints(
            left.get("period_ink_fingerprint"),
            right.get("period_ink_fingerprint"),
            maximum_scale_ratio,
        )
        complete_command_sequences = (
            left.get("period_command_sequence") is not None
            and right.get("period_command_sequence") is not None
        )
        if complete_command_sequences:
            # A complete period is stronger and more stable than the detector's
            # arbitrary anchor token.  The token comparison is cyclic and
            # reversal invariant, checks every token shape/cap/relative scale,
            # and the period-ink fingerprint checks their combined geometry.
            # Absolute period retains the real drawing scale across Groups.
            matched = (
                command_sequence["matched"]
                and period_ink["matched"]
                and period_ratio <= maximum_period_ratio
            )
            errors = [
                min(1.0, math.log(period_ratio) / max(math.log(maximum_period_ratio), EPS)),
                command_sequence["error"],
                period_ink["error"],
            ]
            return {
                "matched": matched,
                "similarity": max(0.0, 1.0 - mean(errors)),
                "reason": "matched" if matched else "complete_period_sequence_mismatch",
                "identity_basis": "complete_period_command_sequence",
                "shape_match": shape_match,
                "line_cap_match": cap_match,
                "scale_ratio": scale_ratio,
                "period_ratio": period_ratio,
                "normalized_period_ratio": normalized_period_ratio,
                "period_command_sequence": command_sequence,
                "period_ink": period_ink,
            }
        if command_sequence["reason"] == "token_count_mismatch":
            command_pattern_match = period_ink["matched"]
            command_pattern_error = period_ink["error"]
        elif command_sequence["reason"].startswith("not_available"):
            command_pattern_match = period_ink["matched"]
            command_pattern_error = period_ink["error"]
        else:
            command_pattern_match = command_sequence["matched"]
            command_pattern_error = command_sequence["error"]
        matched = (
            shape_match
            and cap_match
            and command_pattern_match
            and scale_ratio <= maximum_scale_ratio
            and period_ratio <= maximum_period_ratio
            and normalized_period_ratio <= maximum_period_ratio
        )
        errors = [
            min(1.0, math.log(scale_ratio) / max(math.log(maximum_scale_ratio), EPS)),
            min(1.0, math.log(period_ratio) / max(math.log(maximum_period_ratio), EPS)),
            min(1.0, math.log(normalized_period_ratio) / max(math.log(maximum_period_ratio), EPS)),
            0.0 if shape_match else 1.0,
            0.0 if cap_match else 1.0,
            command_pattern_error,
        ]
        return {
            "matched": matched,
            "similarity": max(0.0, 1.0 - mean(errors)),
            "reason": "matched" if matched else "motif_period_or_sequence_mismatch",
            "shape_match": shape_match,
            "line_cap_match": cap_match,
            "scale_ratio": scale_ratio,
            "period_ratio": period_ratio,
            "normalized_period_ratio": normalized_period_ratio,
            "period_command_sequence": command_sequence,
            "period_ink": period_ink,
        }

    if family == "ink_gap_periodic":
        left_period = left["period_signature"]
        right_period = right["period_signature"]
        same_components = len(left_period) == len(right_period)
        maximum_component_error = (
            max(
                abs(left_value - right_value)
                for left_pair, right_pair in zip(left_period, right_period)
                for left_value, right_value in zip(left_pair, right_pair)
            ) if same_components and left_period else math.inf
        )
        period_ratio = ratio(left["absolute_period"], right["absolute_period"])
        cap_match = left.get("line_cap") == right.get("line_cap")
        matched = (
            same_components
            and maximum_component_error <= 0.08
            and period_ratio <= maximum_period_ratio
            and cap_match
        )
        component_score = 0.0 if not math.isfinite(maximum_component_error) else max(
            0.0, 1.0 - maximum_component_error / 0.08,
        )
        period_score = max(
            0.0,
            1.0 - math.log(period_ratio) / max(math.log(maximum_period_ratio), EPS),
        )
        return {
            "matched": matched,
            "similarity": mean((component_score, period_score, 1.0 if cap_match else 0.0)),
            "reason": "matched" if matched else "ink_period_mismatch",
            "same_component_count": same_components,
            "maximum_component_error": maximum_component_error,
            "line_cap_match": cap_match,
            "period_ratio": period_ratio,
        }
    return {"matched": False, "similarity": 0.0, "reason": "unsupported_family"}


@dataclass
class DiscoveryResult:
    types: list[PatternType]
    rounds: list[dict[str, Any]]


def cluster_residual_atoms(
    atoms: list[Atom],
    candidate_cache: dict[tuple[int, ...], Candidate] | None = None,
) -> list[list[Atom]]:
    if not atoms:
        return []
    candidates = [
        _cached_candidate(index, [atom], candidate_cache)
        for index, atom in enumerate(atoms)
    ]
    local_index_by_id = {atom.id: index for index, atom in enumerate(atoms)}
    sets = DisjointSet(len(candidates))
    _union_matching_candidate_fingerprints(candidates, sets)
    # Preserve the same command-boundary invariance used by periodic motif
    # discovery.  The original Atoms stay separate, but a validated virtual
    # carrier+marker pair joins the residual family of its normal templates.
    candidate_windows = _fingerprint_window_indices(candidates)
    for canonical in canonical_split_marker_candidates(atoms, candidate_cache):
        carrier_index = local_index_by_id[canonical.atom_ids[0]]
        marker_index = local_index_by_id[canonical.atom_ids[1]]
        sets.union(carrier_index, marker_index)
        carrier = atoms[carrier_index]
        for template_index in _matching_window_indices(canonical.fingerprint, candidate_windows):
            template = candidates[template_index]
            template_atom = atoms[template_index]
            if (
                template_atom.id not in canonical.atom_ids
                and same_junction_style(carrier, template_atom)
                and fingerprints_match(canonical.fingerprint, template.fingerprint)
            ):
                sets.union(carrier_index, template_index)
    groups: dict[int, list[Atom]] = {}
    for index, candidate in enumerate(candidates):
        groups.setdefault(sets.find(index), []).append(candidate.members[0])
    return sorted(groups.values(), key=lambda group: min(atom.id for atom in group))


TOPOLOGY_REPAIR_MINIMUM_SEED_ATOMS = 3
TOPOLOGY_REPAIR_MAXIMUM_POOL_ATOMS = 512
TOPOLOGY_REPAIR_MAXIMUM_GAP_PERIODS = 0.55
TOPOLOGY_REPAIR_MAXIMUM_LINK_ANGLE = 65
TOPOLOGY_REPAIR_MAXIMUM_ATOM_PERIODS = 2.5


def _hypothesis_period(hypothesis: PatternHypothesis) -> float:
    """Return the physical repeat scale learned by any periodic model."""
    if isinstance(hypothesis, Hypothesis):
        return hypothesis.refined_spacing
    if isinstance(hypothesis, (NetworkHypothesis, TwoInstanceHypothesis)):
        return hypothesis.spacing
    return hypothesis.period_length or 0.0


def _topology_repair_components(
    pool: list[Atom],
    seed_ids: set[int],
    period: float,
) -> list[tuple[set[int], set[int]]]:
    """Find unbranched ink paths that contain proven and residual atoms.

    Unlike periodic discovery, this pass is allowed to follow a real corner.
    It remains conservative in the other dimensions: endpoints must choose one
    another as their unique nearest continuation, gaps are bounded by the
    already learned period, and every component must be an unbranched path.
    """
    if (
        period <= EPS
        or len(pool) < TOPOLOGY_REPAIR_MINIMUM_SEED_ATOMS + 1
        or len(pool) > TOPOLOGY_REPAIR_MAXIMUM_POOL_ATOMS
    ):
        return []

    options: list[_PathLink] = []
    for left_index, left in enumerate(pool):
        for right in pool[left_index + 1:]:
            for left_side in (0, 1):
                left_point = _endpoint_point(left, left_side)
                left_outward = _endpoint_outward_direction(left, left_side)
                if math.hypot(*left_outward) <= EPS:
                    continue
                for right_side in (0, 1):
                    right_point = _endpoint_point(right, right_side)
                    right_outward = _endpoint_outward_direction(right, right_side)
                    if math.hypot(*right_outward) <= EPS:
                        continue
                    gap = distance(left_point, right_point)
                    if gap > period * TOPOLOGY_REPAIR_MAXIMUM_GAP_PERIODS:
                        continue
                    if gap <= EPS:
                        left_angle = right_angle = directed_angle_degrees(
                            left_outward, mul(right_outward, -1),
                        )
                    else:
                        gap_direction = normalize(sub(right_point, left_point))
                        left_angle = directed_angle_degrees(left_outward, gap_direction)
                        right_angle = directed_angle_degrees(
                            right_outward, mul(gap_direction, -1),
                        )
                    if max(left_angle, right_angle) > TOPOLOGY_REPAIR_MAXIMUM_LINK_ANGLE:
                        continue
                    options.append(_PathLink(
                        left.id,
                        left_side,
                        right.id,
                        right_side,
                        gap,
                        gap / period + (left_angle + right_angle) / 180,
                    ))

    by_endpoint: dict[tuple[int, int], list[int]] = {}
    for index, link in enumerate(options):
        by_endpoint.setdefault((link.left_id, link.left_side), []).append(index)
        by_endpoint.setdefault((link.right_id, link.right_side), []).append(index)

    endpoint_best: dict[tuple[int, int], int] = {}
    for endpoint, indices in by_endpoint.items():
        ranked = sorted(indices, key=lambda index: (options[index].gap, options[index].cost))
        best_index = ranked[0]
        if len(ranked) > 1:
            best, second = options[best_index], options[ranked[1]]
            if (
                second.gap <= max(best.gap, EPS) * 1.25
                and second.cost - best.cost < 0.20
            ):
                # A fork or crossing is not strong enough evidence to rewrite
                # an already accepted type assignment.
                continue
        endpoint_best[endpoint] = best_index

    adjacency: dict[int, set[int]] = {atom.id: set() for atom in pool}
    for index, link in enumerate(options):
        if (
            endpoint_best.get((link.left_id, link.left_side)) == index
            and endpoint_best.get((link.right_id, link.right_side)) == index
        ):
            adjacency[link.left_id].add(link.right_id)
            adjacency[link.right_id].add(link.left_id)

    components: list[tuple[set[int], set[int]]] = []
    visited: set[int] = set()
    for atom in pool:
        if atom.id in visited:
            continue
        component: set[int] = set()
        pending = [atom.id]
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current])
        visited.update(component)
        seeds = component & seed_ids
        residuals = component - seed_ids
        if (
            len(seeds) >= TOPOLOGY_REPAIR_MINIMUM_SEED_ATOMS
            and residuals
            and max((len(adjacency[atom_id]) for atom_id in component), default=0) <= 2
        ):
            components.append((seeds, residuals))
    return components


def reconcile_topology_residuals(atoms: list[Atom], types: list[PatternType]) -> list[dict[str, Any]]:
    """Absorb weak residual islands that continue one proven periodic path.

    The pass never merges two periodic hypotheses.  A residual component is
    moved only when exactly one periodic type claims it through a style-matched,
    mutually-nearest, unbranched endpoint path.  This encodes the practical
    prior that one line normally does not change type briefly at a corner or at
    one of its ends, without imposing a blanket majority vote on the Group.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    owner_by_atom = {
        atom_id: type_index
        for type_index, pattern_type in enumerate(types)
        for atom_id in pattern_type.atom_ids
    }
    proposals: list[tuple[int, set[int], set[int], float]] = []
    for type_index, pattern_type in enumerate(types):
        hypothesis = pattern_type.hypothesis
        if hypothesis is None:
            continue
        if (
            isinstance(hypothesis, SharedPathHypothesis)
            and hypothesis.relation_kind == "self_carried_repeat"
        ):
            # This model already completed its clipped/corner units with
            # endpoint + command evidence.  A second, more permissive generic
            # pass can otherwise walk from its endpoint into the next drawing
            # command (for example a grid edge following the linetype).
            continue
        period = _hypothesis_period(hypothesis)
        if period <= EPS:
            continue
        open_owner_atoms = [
            atom_by_id[atom_id]
            for atom_id in pattern_type.atom_ids
            if (
                atom_id in atom_by_id
                and atom_by_id[atom_id].paint_mode == "stroke"
                and not atom_by_id[atom_id].closed
                and atom_by_id[atom_id].length > EPS
            )
        ]
        for seed_atoms in _style_buckets(open_owner_atoms):
            if len(seed_atoms) < TOPOLOGY_REPAIR_MINIMUM_SEED_ATOMS:
                continue
            seed_ids = {atom.id for atom in seed_atoms}
            pool = [
                atom
                for atom in atoms
                if (
                    atom.paint_mode == "stroke"
                    and not atom.closed
                    and atom.length > EPS
                    and same_junction_style(seed_atoms[0], atom)
                    and (
                        owner_by_atom.get(atom.id) == type_index
                        or (
                            types[owner_by_atom[atom.id]].hypothesis is None
                            and atom.length <= period * TOPOLOGY_REPAIR_MAXIMUM_ATOM_PERIODS
                        )
                    )
                )
            ]
            for seeds, residuals in _topology_repair_components(pool, seed_ids, period):
                proposals.append((type_index, seeds, residuals, period))

    claims: dict[int, set[int]] = {}
    for type_index, _, residuals, _ in proposals:
        for atom_id in residuals:
            claims.setdefault(atom_id, set()).add(type_index)

    accepted: dict[int, dict[str, Any]] = {}
    for type_index, seeds, residuals, period in proposals:
        if any(claims.get(atom_id) != {type_index} for atom_id in residuals):
            continue
        record = accepted.setdefault(type_index, {
            "seed_atom_ids": set(),
            "absorbed_atom_ids": set(),
            "period": period,
        })
        record["seed_atom_ids"].update(seeds)
        record["absorbed_atom_ids"].update(residuals)

    for type_index, record in accepted.items():
        types[type_index].atom_ids.update(record["absorbed_atom_ids"])
    absorbed_ids = {
        atom_id
        for record in accepted.values()
        for atom_id in record["absorbed_atom_ids"]
    }
    if absorbed_ids:
        for pattern_type in types:
            if pattern_type.hypothesis is None:
                pattern_type.atom_ids.difference_update(absorbed_ids)
        types[:] = [pattern_type for pattern_type in types if pattern_type.atom_ids]
        for index, pattern_type in enumerate(types, start=1):
            pattern_type.type_id = f"type_{index:03d}"

    repairs: list[dict[str, Any]] = []
    for type_index, record in accepted.items():
        if not record["absorbed_atom_ids"]:
            continue
        target = next(
            pattern_type for pattern_type in types
            if record["absorbed_atom_ids"] <= pattern_type.atom_ids
            and pattern_type.hypothesis is not None
        )
        repairs.append({
            "model": "topology_continuity_repair",
            "target_type_id": target.type_id,
            "seed_atom_ids": sorted(record["seed_atom_ids"]),
            "absorbed_atom_ids": sorted(record["absorbed_atom_ids"]),
            "period": record["period"],
        })
    return repairs


SANDWICH_REPAIR_MAXIMUM_SOURCE_TYPE_ATOMS = 8
SANDWICH_REPAIR_MINIMUM_TARGET_ATOMS = 12
SANDWICH_REPAIR_MINIMUM_TARGET_RATIO = 4
SANDWICH_REPAIR_MAXIMUM_SOURCE_LENGTH_RATIO = 0.85
SANDWICH_REPAIR_MAXIMUM_GAP_SCALE_RATIO = 0.20
SANDWICH_REPAIR_CLOSE_CONTACT_SCALE_RATIO = 0.03
SANDWICH_REPAIR_MAXIMUM_DIRECTION_ANGLE = 30


@dataclass
class _SandwichAttachment:
    type_index: int
    atom: Atom
    projection: Projection


COMMAND_SEQUENCE_REPAIR_MINIMUM_TOKEN_SUPPORT = 2
COMMAND_SEQUENCE_REPAIR_MINIMUM_BOUNDARY_CARRIER_SUPPORT = 3
COMMAND_SEQUENCE_REPAIR_MINIMUM_BOUNDARY_LENGTH_RATIO = 0.35
COMMAND_SEQUENCE_REPAIR_MAXIMUM_BOUNDARY_LENGTH_RATIO = 1.35
COMMAND_SEQUENCE_REPAIR_MAXIMUM_CARRIER_ANGLE = 10


def _straight_carrier_atom(atom: Atom) -> bool:
    return (
        not atom.closed
        and atom.paint_mode == "stroke"
        and atom.curve_segments == 0
        and 1 <= atom.line_segments <= 3
        and distance(atom.points[0], atom.points[-1]) / max(atom.length, EPS) >= 0.98
    )


def reconcile_periodic_command_sequence_residuals(
    atoms: list[Atom],
    types: list[PatternType],
    candidate_cache: dict[tuple[int, ...], Candidate] | None = None,
) -> list[dict[str, Any]]:
    """Complete a proven periodic command sequence without majority voting.

    Internal tokens need accepted owner atoms immediately before and after in
    paint order plus at least two direct fingerprint templates.  A group-edge
    carrier may be clipped, but needs three full, aligned carrier templates
    and real contact with the periodic owner endpoint.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    owner_by_atom = {
        atom_id: type_index
        for type_index, pattern_type in enumerate(types)
        for atom_id in pattern_type.atom_ids
    }
    minimum_atom_id = min(atom_by_id, default=0)
    maximum_atom_id = max(atom_by_id, default=-1)
    singleton_candidates = {
        atom.id: _cached_candidate(-1, [atom], candidate_cache)
        for atom in atoms
    }
    proposals: list[tuple[int, int, int, str, dict[str, Any]]] = []
    for source_index, source_type in enumerate(types):
        if source_type.hypothesis is not None:
            continue
        for source_id in sorted(source_type.atom_ids):
            source = atom_by_id[source_id]
            possible_targets: list[tuple[int, str, dict[str, Any]]] = []
            for target_index, target_type in enumerate(types):
                hypothesis = target_type.hypothesis
                if hypothesis is None:
                    continue
                previous_owner = owner_by_atom.get(source_id - 1) == target_index
                following_owner = owner_by_atom.get(source_id + 1) == target_index
                internal_candidate = previous_owner and following_owner
                at_group_edge = source_id in (minimum_atom_id, maximum_atom_id)
                boundary_candidate = (
                    at_group_edge
                    and previous_owner != following_owner
                    and _straight_carrier_atom(source)
                )
                if not internal_candidate and not boundary_candidate:
                    continue
                owner_atoms = [
                    atom_by_id[atom_id]
                    for atom_id in target_type.atom_ids
                    if same_junction_style(source, atom_by_id[atom_id])
                ]
                if not owner_atoms:
                    continue
                if internal_candidate:
                    source_candidate = singleton_candidates[source.id]
                    direct_support = sum(
                        fingerprints_match(
                            source_candidate.fingerprint,
                            singleton_candidates[template.id].fingerprint,
                        )
                        for template in owner_atoms
                    )
                else:
                    direct_support = 0
                if internal_candidate and direct_support >= COMMAND_SEQUENCE_REPAIR_MINIMUM_TOKEN_SUPPORT:
                    possible_targets.append((target_index, "internal_repeated_token", {
                        "direct_template_support": direct_support,
                    }))
                    continue

                if not boundary_candidate:
                    continue
                carrier_templates = [
                    template
                    for template in owner_atoms
                    if _straight_carrier_atom(template)
                    and acute_angle_degrees(
                        atom_chord_direction(source), atom_chord_direction(template),
                    ) <= COMMAND_SEQUENCE_REPAIR_MAXIMUM_CARRIER_ANGLE
                ]
                if len(carrier_templates) < COMMAND_SEQUENCE_REPAIR_MINIMUM_BOUNDARY_CARRIER_SUPPORT:
                    continue
                typical_length = median(template.length for template in carrier_templates)
                length_ratio = source.length / max(typical_length, EPS)
                if not (
                    COMMAND_SEQUENCE_REPAIR_MINIMUM_BOUNDARY_LENGTH_RATIO
                    <= length_ratio
                    <= COMMAND_SEQUENCE_REPAIR_MAXIMUM_BOUNDARY_LENGTH_RATIO
                ):
                    continue
                period = _hypothesis_period(hypothesis)
                owner_candidate = Candidate(
                    -1,
                    owner_atoms,
                    sorted(atom.id for atom in owner_atoms),
                    (0.0, 0.0),
                    1.0,
                    {},
                )
                endpoint_gaps = [
                    distance_to_candidate_ink(endpoint, owner_candidate)
                    for endpoint in (source.points[0], source.points[-1])
                ]
                if min(endpoint_gaps) > period * 0.10:
                    continue
                possible_targets.append((target_index, "clipped_boundary_carrier", {
                    "carrier_template_support": len(carrier_templates),
                    "length_ratio": length_ratio,
                    "endpoint_gaps": endpoint_gaps,
                }))
            if len(possible_targets) == 1:
                target_index, reason, evidence = possible_targets[0]
                proposals.append((source_id, source_index, target_index, reason, evidence))

    moved: list[tuple[str, PatternType, int, str, dict[str, Any]]] = []
    for source_id, source_index, target_index, reason, evidence in proposals:
        source_type, target_type = types[source_index], types[target_index]
        if source_id not in source_type.atom_ids:
            continue
        source_type.atom_ids.remove(source_id)
        target_type.atom_ids.add(source_id)
        if target_type.hypothesis is not None:
            target_type.hypothesis.explained_atom_ids.add(source_id)
        moved.append((source_type.type_id, target_type, source_id, reason, evidence))
    if moved:
        types[:] = [pattern_type for pattern_type in types if pattern_type.atom_ids]
        for index, pattern_type in enumerate(types, start=1):
            pattern_type.type_id = f"type_{index:03d}"
    return [
        {
            "model": "periodic_command_sequence_repair",
            "source_type_id_before_repair": source_type_id,
            "target_type_id_after_repair": target_type.type_id,
            "absorbed_atom_ids": [source_id],
            "reason": reason,
            **evidence,
        }
        for source_type_id, target_type, source_id, reason, evidence in moved
    ]


def _nearest_projection_to_atom(point: Point, atom: Atom) -> Projection:
    if len(atom.points) < 2:
        return Projection(distance(point, atom.points[0]), 0.0, 0.0, atom.points[0], (0.0, 0.0))
    return min(
        (
            project_point_to_segment(point, atom.points[index - 1], atom.points[index])
            for index in range(1, len(atom.points))
        ),
        key=lambda projection: projection.distance,
    )


def reconcile_sandwiched_residuals(
    atoms: list[Atom],
    types: list[PatternType],
) -> list[dict[str, Any]]:
    """Absorb a short residual Atom uniquely bracketed by one large owner.

    This handles a local command-boundary anomaly even when the surrounding
    closed-loop linetype has not yet produced a periodic hypothesis.  Both
    ends must select the same target type, target atoms must be distinct, the
    continuation must point outward along the residual chord, and competing
    owners near either end make the decision ambiguous and therefore invalid.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    owner_by_atom = {
        atom_id: type_index
        for type_index, pattern_type in enumerate(types)
        for atom_id in pattern_type.atom_ids
    }
    proposals: list[tuple[int, int, int, float, float]] = []
    for source_index, source_type in enumerate(types):
        if (
            source_type.hypothesis is not None
            or len(source_type.atom_ids) > SANDWICH_REPAIR_MAXIMUM_SOURCE_TYPE_ATOMS
        ):
            continue
        for source_id in sorted(source_type.atom_ids):
            source = atom_by_id[source_id]
            if source.closed or source.paint_mode != "stroke" or len(source.points) < 2:
                continue
            chord = sub(source.points[-1], source.points[0])
            if math.hypot(*chord) <= EPS:
                continue

            options_by_side: list[list[_SandwichAttachment]] = []
            for endpoint in (source.points[0], source.points[-1]):
                options: list[_SandwichAttachment] = []
                for target_index, target_type in enumerate(types):
                    if target_index == source_index:
                        continue
                    target_atoms = [
                        atom_by_id[atom_id]
                        for atom_id in target_type.atom_ids
                        if same_junction_style(source, atom_by_id[atom_id])
                    ]
                    if (
                        len(target_atoms) < SANDWICH_REPAIR_MINIMUM_TARGET_ATOMS
                        or len(target_atoms) < len(source_type.atom_ids) * SANDWICH_REPAIR_MINIMUM_TARGET_RATIO
                    ):
                        continue
                    target_scale = median(atom.scale for atom in target_atoms)
                    target_length = median(atom.length for atom in target_atoms)
                    if source.length > target_length * SANDWICH_REPAIR_MAXIMUM_SOURCE_LENGTH_RATIO:
                        continue
                    best_atom, best_projection = min(
                        (
                            (atom, _nearest_projection_to_atom(endpoint, atom))
                            for atom in target_atoms
                        ),
                        key=lambda item: item[1].distance,
                    )
                    gap_limit = max(
                        abs(source.line_width) * 4,
                        target_scale * SANDWICH_REPAIR_MAXIMUM_GAP_SCALE_RATIO,
                    )
                    if best_projection.distance <= gap_limit:
                        options.append(_SandwichAttachment(
                            target_index, best_atom, best_projection,
                        ))
                options_by_side.append(sorted(
                    options,
                    key=lambda item: (item.projection.distance, item.type_index, item.atom.id),
                ))
            if any(not options for options in options_by_side):
                continue
            best_left, best_right = options_by_side[0][0], options_by_side[1][0]
            if best_left.type_index != best_right.type_index or best_left.atom.id == best_right.atom.id:
                continue
            target_index = best_left.type_index
            target_atoms = [
                atom_by_id[atom_id]
                for atom_id in types[target_index].atom_ids
                if same_junction_style(source, atom_by_id[atom_id])
            ]
            target_scale = median(atom.scale for atom in target_atoms)
            ambiguity = max(abs(source.line_width) * 4, target_scale * 0.05)
            if any(
                len(options) > 1
                and options[1].projection.distance <= options[0].projection.distance + ambiguity
                for options in options_by_side
            ):
                continue
            close_limit = max(
                abs(source.line_width) * 4,
                target_scale * SANDWICH_REPAIR_CLOSE_CONTACT_SCALE_RATIO,
            )
            if min(best_left.projection.distance, best_right.projection.distance) > close_limit:
                continue

            outward_directions = normalize(mul(chord, -1)), normalize(chord)
            valid_directions = True
            for endpoint, attachment, outward in zip(
                (source.points[0], source.points[-1]),
                (best_left, best_right),
                outward_directions,
            ):
                continuation = sub(attachment.projection.point, endpoint)
                if (
                    math.hypot(*continuation) > EPS
                    and directed_angle_degrees(outward, continuation)
                    > SANDWICH_REPAIR_MAXIMUM_DIRECTION_ANGLE
                ):
                    valid_directions = False
                    break
            if not valid_directions:
                continue
            proposals.append((
                source_id,
                source_index,
                target_index,
                best_left.projection.distance,
                best_right.projection.distance,
            ))

    # One Atom may not be claimed by competing targets.  This is normally
    # prevented above, but retaining the explicit owner check keeps the pass
    # safe if several source clusters propose the same geometry in the future.
    claims: dict[int, set[int]] = {}
    for source_id, _, target_index, _, _ in proposals:
        claims.setdefault(source_id, set()).add(target_index)
    accepted = [
        proposal for proposal in proposals
        if claims[proposal[0]] == {proposal[2]}
    ]
    moved_records: list[tuple[str, PatternType, int, float, float]] = []
    for source_id, source_index, target_index, left_gap, right_gap in accepted:
        source_type, target_type = types[source_index], types[target_index]
        if source_id not in source_type.atom_ids:
            continue
        source_type.atom_ids.remove(source_id)
        target_type.atom_ids.add(source_id)
        moved_records.append((source_type.type_id, target_type, source_id, left_gap, right_gap))
    if moved_records:
        types[:] = [pattern_type for pattern_type in types if pattern_type.atom_ids]
        for index, pattern_type in enumerate(types, start=1):
            pattern_type.type_id = f"type_{index:03d}"
    return [
        {
            "model": "sandwiched_residual_repair",
            "source_type_id_before_repair": source_type_id,
            "target_type_id_after_repair": target_type.type_id,
            "absorbed_atom_ids": [source_id],
            "endpoint_gaps": [left_gap, right_gap],
        }
        for source_type_id, target_type, source_id, left_gap, right_gap in moved_records
    ]


def reconcile_branched_command_carriers(
    atoms: list[Atom],
    types: list[PatternType],
) -> list[dict[str, Any]]:
    """Rejoin one boundary ``carrier, motif, carrier`` branch to its owner.

    The middle motif already has direct fingerprint evidence that it belongs
    to a proven periodic type.  When its two carrier neighbors remain residual
    only because that occurrence starts a spatially separated branch, keeping
    the motif's identity is stronger evidence than inventing a one-copy line
    type.  The complete three-command block must lie at the Group boundary,
    both comparable carriers must touch the motif, and the motif must be
    spatially isolated from its owner's other occurrences.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    owner_by_atom = {
        atom_id: type_index
        for type_index, pattern_type in enumerate(types)
        for atom_id in pattern_type.atom_ids
    }
    proposals: list[tuple[int, int, int, int]] = []
    minimum_atom_id = min(atom_by_id, default=0)
    maximum_atom_id = max(atom_by_id, default=-1)
    for middle_id in sorted(atom_by_id):
        left_id, right_id = middle_id - 1, middle_id + 1
        if left_id not in atom_by_id or right_id not in atom_by_id:
            continue
        if left_id != minimum_atom_id and right_id != maximum_atom_id:
            # A branch with only one motif is accepted only as a complete
            # Group-boundary command block; an arbitrary internal residual
            # pair must not be pulled into the periodic owner.
            continue
        middle_owner = owner_by_atom[middle_id]
        left_owner, right_owner = owner_by_atom[left_id], owner_by_atom[right_id]
        if (
            types[middle_owner].hypothesis is None
            or types[left_owner].hypothesis is not None
            or types[right_owner].hypothesis is not None
        ):
            continue
        left, middle, right = atom_by_id[left_id], atom_by_id[middle_id], atom_by_id[right_id]
        if not (
            middle.paint_mode == "stroke"
            and not middle.closed
            and 3 <= middle.line_segments <= 8
            and left.paint_mode == right.paint_mode == "stroke"
            and not left.closed and not right.closed
            and all(same_junction_style(middle, carrier) for carrier in (left, right))
            and left.length >= middle.length * 3
            and right.length >= middle.length * 3
            and max(left.length, right.length) / max(min(left.length, right.length), EPS) <= 1.6
        ):
            continue
        contact_limit = max(abs(middle.line_width) * 4, middle.scale * 0.8)
        contact_gaps = [
            min(distance(point, motif_point) for point in carrier.points for motif_point in middle.points)
            for carrier in (left, right)
        ]
        if max(contact_gaps) > contact_limit:
            continue
        other_owner_atoms = [
            atom_by_id[atom_id]
            for atom_id in types[middle_owner].atom_ids
            if atom_id != middle_id and atom_id in atom_by_id
        ]
        owner_separation = min((
            distance(point, owner_point)
            for point in middle.points
            for owner_atom in other_owner_atoms
            for owner_point in owner_atom.points
        ), default=math.inf)
        if owner_separation <= max(middle.scale * 5, max(contact_gaps) * 4):
            continue
        proposals.append((left_id, middle_id, right_id, middle_owner))

    claimed = {
        atom_id
        for left_id, middle_id, right_id, _ in proposals
        for atom_id in (left_id, middle_id, right_id)
    }
    if len(claimed) != len(proposals) * 3:
        return []
    records: list[dict[str, Any]] = []
    for left_id, middle_id, right_id, middle_owner in proposals:
        ids = {left_id, middle_id, right_id}
        target = types[middle_owner]
        target.atom_ids.update(ids)
        for pattern_type in types:
            if pattern_type.hypothesis is None:
                pattern_type.atom_ids.difference_update(ids)
        records.append({
            "model": "branched_command_carrier_repair",
            "target_type_id": target.type_id,
            "motif_atom_id": middle_id,
            "absorbed_carrier_atom_ids": [left_id, right_id],
        })
    if records:
        types[:] = [pattern_type for pattern_type in types if pattern_type.atom_ids]
        for index, pattern_type in enumerate(types, start=1):
            pattern_type.type_id = f"type_{index:03d}"
    return records


def _one_to_one_command_phase_evidence(
    carrier_cluster: list[Candidate],
    module_cluster: list[Candidate],
    carrier_anchors: list[int],
    stride: int,
) -> dict[str, Any] | None:
    """Prove two singleton chains are interleaved parts of one command period.

    Equal-frequency modules need a stronger guard than the integer-period
    branch below: two nearby parallel dashed lines may have the same period
    without being one linetype.  Require a fixed non-zero command phase, paired
    stations that advance together, a stable short longitudinal offset, and at
    most one clipped boundary pair.
    """
    if (
        len(module_cluster) < 3
        or any(
            len(candidate.atom_ids) != 1
            for candidate in module_cluster
        )
    ):
        return None
    module_by_anchor = {
        candidate.atom_ids[0]: candidate for candidate in module_cluster
    }
    module_anchors = sorted(module_by_anchor)
    module_strides = [
        right - left for left, right in zip(module_anchors, module_anchors[1:])
    ]
    if not module_strides or any(value != stride for value in module_strides):
        return None

    origin = carrier_anchors[0]
    phases = {(anchor - origin) % stride for anchor in module_anchors}
    if len(phases) != 1:
        return None
    phase = next(iter(phases))
    if phase == 0:
        return None

    carrier_by_anchor = {
        candidate.atom_ids[0]: candidate for candidate in carrier_cluster
    }
    pairs: list[tuple[Candidate, Candidate]] = []
    for module_anchor in module_anchors:
        block_index = (module_anchor - origin - phase) // stride
        carrier_anchor = origin + block_index * stride
        carrier_candidate = carrier_by_anchor.get(carrier_anchor)
        if carrier_candidate is not None:
            pairs.append((carrier_candidate, module_by_anchor[module_anchor]))
    minimum_pair_count = max(3, min(len(carrier_cluster), len(module_cluster)) - 1)
    if len(pairs) < minimum_pair_count:
        return None
    pairs.sort(key=lambda pair: pair[0].atom_ids[0])

    carrier_steps = [
        sub(right[0].center, left[0].center)
        for left, right in zip(pairs, pairs[1:])
    ]
    module_steps = [
        sub(right[1].center, left[1].center)
        for left, right in zip(pairs, pairs[1:])
    ]
    if not carrier_steps or any(math.hypot(*step) <= EPS for step in [*carrier_steps, *module_steps]):
        return None
    if any(
        acute_angle_degrees(carrier_step, module_step) > 20
        for carrier_step, module_step in zip(carrier_steps, module_steps)
    ):
        return None
    paired_step_ratios = [
        max(math.hypot(*carrier_step), math.hypot(*module_step))
        / max(min(math.hypot(*carrier_step), math.hypot(*module_step)), EPS)
        for carrier_step, module_step in zip(carrier_steps, module_steps)
    ]
    if max(paired_step_ratios) > 1.3:
        return None

    station_period = median([
        *(math.hypot(*step) for step in carrier_steps),
        *(math.hypot(*step) for step in module_steps),
    ])
    pair_gaps = [distance(carrier.center, module.center) for carrier, module in pairs]
    median_gap = median(pair_gaps)
    if median_gap > station_period * 0.45:
        return None
    if median_gap > EPS and population_std(pair_gaps) / median_gap > 0.15:
        return None

    # The paired module must sit along the local route, not on a neighboring
    # parallel route.  Permit one endpoint at a bend where a one-sided tangent
    # can temporarily make the longitudinal offset look lateral.
    longitudinal_pair_count = 0
    for index, (carrier, module) in enumerate(pairs):
        if index == 0:
            tangent = carrier_steps[0]
        elif index == len(pairs) - 1:
            tangent = carrier_steps[-1]
        else:
            tangent = sub(pairs[index + 1][0].center, pairs[index - 1][0].center)
        lateral = lateral_distance_to_line(
            module.center, carrier.center, add(carrier.center, tangent),
        )
        if lateral <= max(median_gap * 0.35, station_period * 0.03):
            longitudinal_pair_count += 1
    if longitudinal_pair_count < len(pairs) - 1:
        return None

    return {
        "phase": phase,
        "paired_station_count": len(pairs),
        "period": station_period,
        "fit_error": max(
            max(paired_step_ratios) - 1,
            population_std(pair_gaps) / max(median_gap, EPS),
        ),
    }


def reconcile_co_phased_periodic_modules(
    atoms: list[Atom],
    types: list[PatternType],
) -> list[dict[str, Any]]:
    """Merge periodic modules that form one stable command-period block.

    A CAD exporter may paint the carrier and its attached symbols as separate
    command families.  Both families can independently look periodic, but
    they are one linetype when every carrier period contains the same ordered
    module block.  Acceptance requires at least three complete blocks, a
    constant command stride, matching complete-block fingerprints, parallel
    progression and a small integer relationship between the two learned
    periods.  Direction alone is never sufficient.
    """
    atom_by_id = {atom.id: atom for atom in atoms}
    proposals: list[dict[str, Any]] = []
    for carrier_index, carrier_type in enumerate(types):
        carrier_hypothesis = carrier_type.hypothesis
        if not isinstance(carrier_hypothesis, Hypothesis):
            continue
        carrier_cluster = carrier_hypothesis.cluster
        if (
            len(carrier_cluster) < 3
            or any(len(candidate.atom_ids) != 1 for candidate in carrier_cluster)
        ):
            continue
        plain_carrier_cluster = all(
            ordinary_single_line_candidate(candidate) for candidate in carrier_cluster
        )
        anchors = sorted(candidate.atom_ids[0] for candidate in carrier_cluster)
        strides = [right - left for left, right in zip(anchors, anchors[1:])]
        if not strides or min(strides) < 2 or len(set(strides)) != 1:
            continue
        stride = strides[0]
        carrier_direction = sub(
            atom_by_id[anchors[-1]].center, atom_by_id[anchors[0]].center,
        )
        if math.hypot(*carrier_direction) <= EPS:
            continue
        for module_index, module_type in enumerate(types):
            if module_index == carrier_index or module_type.hypothesis is None:
                continue
            module_cluster = getattr(module_type.hypothesis, "cluster", [])
            if len(module_cluster) < 3:
                continue
            module_period = _hypothesis_period(module_type.hypothesis)
            carrier_period = carrier_hypothesis.refined_spacing
            period_multiple = round(carrier_period / max(module_period, EPS))
            integer_period_match = (
                plain_carrier_cluster
                and
                2 <= period_multiple <= 8
                and abs(carrier_period / max(module_period, EPS) - period_multiple)
                    <= period_multiple * 0.08
            )
            one_to_one = _one_to_one_command_phase_evidence(
                carrier_cluster, module_cluster, anchors, stride,
            )
            if not integer_period_match and one_to_one is None:
                continue
            if one_to_one is not None:
                # The 1:1 proof is symmetric.  Keep one canonical proposal so
                # pair ambiguity filtering does not discard the valid merge.
                if carrier_index > module_index:
                    continue
                period_multiple = 1
            module_direction = sub(
                min(module_cluster, key=lambda candidate: dot(candidate.center, carrier_direction)).center,
                max(module_cluster, key=lambda candidate: dot(candidate.center, carrier_direction)).center,
            )
            if acute_angle_degrees(carrier_direction, module_direction) > 5:
                continue

            owner_ids = carrier_type.atom_ids | module_type.atom_ids
            block_ids = [list(range(anchor, anchor + stride)) for anchor in anchors]
            complete_blocks = [
                ids for ids in block_ids
                if all(atom_id in atom_by_id and atom_id in owner_ids for atom_id in ids)
                and any(atom_id in carrier_type.atom_ids for atom_id in ids)
                and any(atom_id in module_type.atom_ids for atom_id in ids)
            ]
            if len(complete_blocks) < 3:
                continue
            block_candidates = [
                make_candidate(index, [atom_by_id[atom_id] for atom_id in ids])
                for index, ids in enumerate(complete_blocks)
            ]
            if any(
                not fingerprints_match(block_candidates[0].fingerprint, candidate.fingerprint)
                for candidate in block_candidates[1:]
            ):
                continue
            proposals.append({
                "target_index": min(carrier_index, module_index),
                "source_index": max(carrier_index, module_index),
                "carrier_index": carrier_index,
                "module_index": module_index,
                "period": one_to_one["period"] if one_to_one is not None else carrier_period,
                "period_multiple": period_multiple,
                "stride": stride,
                "blocks": complete_blocks,
                "block_candidates": block_candidates,
                "one_to_one": one_to_one,
            })

    pair_counts: dict[tuple[int, int], int] = {}
    for proposal in proposals:
        pair = proposal["target_index"], proposal["source_index"]
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    proposals = [
        proposal for proposal in proposals
        if pair_counts[(proposal["target_index"], proposal["source_index"])] == 1
    ]
    used_indices: set[int] = set()
    accepted: list[dict[str, Any]] = []
    for proposal in sorted(proposals, key=lambda item: (-len(item["blocks"]), item["target_index"])):
        target_index, source_index = proposal["target_index"], proposal["source_index"]
        if target_index in used_indices or source_index in used_indices:
            continue
        used_indices.update((target_index, source_index))
        target, source = types[target_index], types[source_index]
        carrier = types[proposal["carrier_index"]]
        explained = set(target.atom_ids | source.atom_ids)
        reference_ids = set(carrier.atom_ids)
        style_atom = atom_by_id[next(iter(reference_ids))]
        interior_boundary_ids: set[int] = set()
        for atom_id in range(min(explained), max(explained) + 1):
            atom = atom_by_id.get(atom_id)
            if (
                atom is None
                or atom_id in explained
                or atom.length > proposal["period"]
                or not same_junction_style(style_atom, atom)
            ):
                continue
            owner = next((
                pattern_type for pattern_type in types
                if atom_id in pattern_type.atom_ids
            ), None)
            if owner is None or owner.hypothesis is not None:
                continue
            contact_gap = min(
                distance(point, owner_point)
                for point in (atom.points[0], atom.points[-1])
                for explained_id in explained
                for owner_point in atom_by_id[explained_id].points
            )
            if contact_gap <= proposal["period"] * 0.05:
                interior_boundary_ids.add(atom_id)
        # A clipped phase may contain several fragments sharing the same
        # carrier endpoint, but it may not consume a whole extra command
        # period or jump outside the already proven block envelope.
        if len(interior_boundary_ids) >= proposal["stride"]:
            interior_boundary_ids.clear()
        explained.update(interior_boundary_ids)
        fit_error = (
            proposal["one_to_one"]["fit_error"]
            if proposal["one_to_one"] is not None
            else max(
                abs(proposal["period"] / max(_hypothesis_period(types[proposal["module_index"]].hypothesis), EPS)
                    - proposal["period_multiple"]) / proposal["period_multiple"],
                0.0,
            )
        )
        target.atom_ids = explained
        target.kind = "shared_reference_path_pattern"
        target.hypothesis = SharedPathHypothesis(
            "co_phased_modules",
            [list(ids) for ids in proposal["blocks"]],
            set(explained),
            reference_ids,
            set(explained),
            proposal["period"],
            None,
            len(proposal["blocks"]),
            fit_error,
            (
                len(explained), len(proposal["blocks"]),
                -js_round_nonnegative(fit_error * 10_000), -proposal["stride"],
            ),
            [list(ids) for ids in proposal["blocks"]],
        )
        source.atom_ids.clear()
        for pattern_type in types:
            if pattern_type.hypothesis is None:
                pattern_type.atom_ids.difference_update(interior_boundary_ids)
        accepted.append({
            "model": "co_phased_periodic_module_merge",
            "target_type_id_before_merge": target.type_id,
            "source_type_id_before_merge": source.type_id,
            "complete_block_count": len(proposal["blocks"]),
            "command_stride": proposal["stride"],
            "period": proposal["period"],
            "module_period_multiple": proposal["period_multiple"],
            "command_phase": (
                proposal["one_to_one"]["phase"]
                if proposal["one_to_one"] is not None else None
            ),
            "absorbed_boundary_atom_ids": sorted(interior_boundary_ids),
        })
    if accepted:
        types[:] = [pattern_type for pattern_type in types if pattern_type.atom_ids]
        for index, pattern_type in enumerate(types, start=1):
            pattern_type.type_id = f"type_{index:03d}"
    return accepted


def discover_unknown_pattern_types(atoms: list[Atom], minimum_explained_atoms: int = 3) -> DiscoveryResult:
    remaining_ids = {atom.id for atom in atoms}
    original_positions = {atom.id: index for index, atom in enumerate(atoms)}
    types: list[PatternType] = []
    rounds: list[dict[str, Any]] = []
    candidate_cache: dict[tuple[int, ...], Candidate] = {}
    while len(remaining_ids) >= minimum_explained_atoms:
        remaining = [atom for atom in atoms if atom.id in remaining_ids]
        standard_hypothesis_cache: dict[int, Any] = {}
        candidates, clusters, hypotheses = build_unknown_pattern_hypotheses(
            remaining, candidate_cache, standard_hypothesis_cache,
        )
        standard = next((
            hypothesis for hypothesis in hypotheses
            if len(hypothesis.explained_atom_ids) >= minimum_explained_atoms
        ), None)
        self_carried_already_accepted = any(
            isinstance(pattern_type.hypothesis, SharedPathHypothesis)
            and pattern_type.hypothesis.relation_kind == "self_carried_repeat"
            for pattern_type in types
        )
        newly_exposed_plain_three = (
            self_carried_already_accepted
            and isinstance(standard, Hypothesis)
            and len(standard.explained_atom_ids) == 3
            and standard.motif_member_count == 1
            and all(ordinary_single_line_candidate(candidate) for candidate in standard.cluster)
        )
        if newly_exposed_plain_three:
            # Removing a large self-carried line can leave three unrelated
            # construction segments that happen to share length.  Their only
            # evidence is the minimum cardinality itself; do not promote this
            # newly exposed coincidence to another linetype.
            standard = None

        local_subsets: list[NetworkHypothesis] = []
        if standard is None:
            local_subsets = [
                recovered
                for cluster in clusters
                if (recovered := build_reliable_local_subset_hypothesis(
                    cluster, remaining, standard_hypothesis_cache=standard_hypothesis_cache,
                )) is not None
                and len(recovered.explained_atom_ids) >= minimum_explained_atoms
                and not (
                    self_carried_already_accepted
                    and recovered.station_count == 3
                    and recovered.motif_member_count == 1
                    and all(ordinary_single_line_candidate(candidate) for candidate in recovered.cluster)
                )
            ]
            local_owners: dict[int, set[int]] = {}
            for index, recovered in enumerate(local_subsets):
                for atom_id in recovered.explained_atom_ids:
                    local_owners.setdefault(atom_id, set()).add(index)
            ambiguous_local_indices = {
                index
                for owners in local_owners.values()
                if len(owners) > 1
                for index in owners
            }
            local_subsets = [
                recovered for index, recovered in enumerate(local_subsets)
                if index not in ambiguous_local_indices
            ]
            local_subsets.sort(key=lambda hypothesis: hypothesis.score, reverse=True)

        accepted: PatternHypothesis | None = local_subsets[0] if local_subsets else standard

        # A strict two-copy line may subsume the weakest possible legacy
        # result: three plain line segments with no higher-level motif.  This
        # is what happens after Case 053's tick line has been removed.  Strong
        # legacy results and previously accepted Atom assignments remain
        # untouchable.
        weak_plain_three = (
            isinstance(standard, Hypothesis)
            and len(standard.explained_atom_ids) == minimum_explained_atoms == 3
            and standard.motif_member_count == 1
            and all(ordinary_single_line_candidate(candidate) for candidate in standard.cluster)
        )
        two_instance_hypotheses: list[TwoInstanceHypothesis] = []
        if accepted is None or (accepted is standard and weak_plain_three):
            two_instance_hypotheses = build_two_instance_hypotheses(
                remaining, candidates, original_positions,
            )
            if two_instance_hypotheses:
                guarded = two_instance_hypotheses[0]
                guarded_connector_ids = set(guarded.middle_bridge.atom_ids) | {
                    guarded.left_extension.atom.id,
                    guarded.right_extension.atom.id,
                }
                if accepted is None or (
                    weak_plain_three
                    and standard is not None
                    and standard.explained_atom_ids == guarded_connector_ids
                ):
                    accepted = guarded

        shared_path_hypotheses: list[SharedPathHypothesis] = []
        if accepted is None:
            shared_path_hypotheses = build_shared_reference_path_hypotheses(
                remaining, clusters,
            )
            if shared_path_hypotheses:
                accepted = shared_path_hypotheses[0]

        shown_hypotheses: list[PatternHypothesis] = [
            *hypotheses, *local_subsets, *two_instance_hypotheses,
            *shared_path_hypotheses,
        ]
        rounds.append({
            "input_atom_ids": [atom.id for atom in remaining],
            "candidate_count": len(candidates),
            "repeated_cluster_count": len(clusters),
            "hypotheses": [hypothesis_summary(hypothesis) for hypothesis in shown_hypotheses],
            "accepted": hypothesis_summary(accepted) if accepted else None,
        })
        if accepted is None:
            break
        accepted_kind = (
            "carrier_supported_two_instance_pattern"
            if isinstance(accepted, TwoInstanceHypothesis)
            else "shared_reference_path_pattern"
            if isinstance(accepted, SharedPathHypothesis)
            else "discovered_periodic_pattern"
        )
        types.append(PatternType(
            f"type_{len(types) + 1:03d}", accepted_kind,
            set(accepted.explained_atom_ids), accepted,
        ))
        remaining_ids.difference_update(accepted.explained_atom_ids)
    for group in cluster_residual_atoms(
        [atom for atom in atoms if atom.id in remaining_ids], candidate_cache,
    ):
        types.append(PatternType(f"type_{len(types) + 1:03d}", "residual_geometry_cluster", {atom.id for atom in group}))
    branched_carrier_repairs = reconcile_branched_command_carriers(atoms, types)
    if branched_carrier_repairs:
        rounds.append({
            "phase": "branched_command_carrier_repair",
            "repairs": branched_carrier_repairs,
        })
    command_sequence_repairs = reconcile_periodic_command_sequence_residuals(
        atoms, types, candidate_cache,
    )
    if command_sequence_repairs:
        rounds.append({
            "phase": "periodic_command_sequence_repair",
            "repairs": command_sequence_repairs,
        })
    sandwich_repairs = reconcile_sandwiched_residuals(atoms, types)
    if sandwich_repairs:
        rounds.append({
            "phase": "sandwiched_residual_repair",
            "repairs": sandwich_repairs,
        })
    module_merges = reconcile_co_phased_periodic_modules(atoms, types)
    if module_merges:
        rounds.append({
            "phase": "co_phased_periodic_module_merge",
            "repairs": module_merges,
        })
    repairs = reconcile_topology_residuals(atoms, types)
    if repairs:
        rounds.append({
            "phase": "topology_continuity_repair",
            "repairs": repairs,
        })
    return DiscoveryResult(types, rounds)


# ---------------------------------------------------------------------------
# Standard-library SVG and JSON output


PALETTE = ("#e45756", "#2f80ed", "#8e5ac8", "#d68b22", "#18a286", "#b34c9b")
NON_LINETYPE_COLOR = "#7b8188"


def type_color(index: int) -> str:
    """第 7 种以后的线型按黄金角取色。

    统一输出十六进制：`hsl(H S% L%)` 这种 CSS Color 4 空格写法浏览器认，
    但 SVG 光栅化器（PyMuPDF 等）不认，会把整条线画成黑色。
    """
    if index < len(PALETTE):
        return PALETTE[index]
    hue = ((index * 137.508) % 360) / 360
    red, green, blue = colorsys.hls_to_rgb(hue, 0.46, 0.65)
    return "#%02x%02x%02x" % (round(red * 255), round(green * 255), round(blue * 255))


def xml_attribute(value: object) -> str:
    return escape(str(value), {'"': "&quot;"})


@dataclass(frozen=True)
class CaseSourceInfo:
    case_id: str
    pdf_file_name: str | None = None
    original_page: int | None = None

    def label(self) -> str:
        parts = [f"Case {self.case_id}"]
        if self.pdf_file_name:
            parts.append(self.pdf_file_name)
        if self.original_page is not None:
            parts.append(f"PDF page {self.original_page}")
        return " · ".join(parts)


CASE_COMMAND_NAME = re.compile(r"^case_(\d+)_commands(?:[^.]*)\.txt$", re.IGNORECASE)


def case_source_info_for(input_path: Path) -> CaseSourceInfo | None:
    """Look up compact source information without making the index mandatory."""
    match = CASE_COMMAND_NAME.match(input_path.name)
    if match is None:
        return None
    case_id = f"{int(match.group(1)):03d}"
    fallback = CaseSourceInfo(case_id)
    index_paths = (
        input_path.parent / "case_index" / "cases.json",
        Path(__file__).resolve().with_name("case_index") / "cases.json",
    )
    checked: set[Path] = set()
    for index_path in index_paths:
        resolved = index_path.resolve()
        if resolved in checked or not resolved.is_file():
            continue
        checked.add(resolved)
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            cases = payload.get("cases")
            if not isinstance(cases, dict):
                continue
            record = cases.get(input_path.name)
            if not isinstance(record, dict):
                continue
            pdf = record.get("pdf", {})
            page = record.get("page", {})
            file_name = pdf.get("file_name") if isinstance(pdf, dict) else None
            original_page = page.get("original_page") if isinstance(page, dict) else None
            return CaseSourceInfo(
                str(record.get("case_id") or case_id),
                str(file_name) if file_name else None,
                int(original_page) if original_page is not None else None,
            )
        except (OSError, ValueError, TypeError):
            continue
    return fallback


def render_svg(
    atoms: list[Atom],
    result: DiscoveryResult,
    output: Path,
    source_info: CaseSourceInfo | None = None,
) -> None:
    source_label = source_info.label() if source_info else ""
    width, height, margin, legend_x = 1500, 880, 70, 1030
    header = 55 if source_label else 0
    all_points = [point for atom in atoms for point in atom.points]
    raw_width = max(point[0] for point in all_points) - min(point[0] for point in all_points)
    raw_height = max(point[1] for point in all_points) - min(point[1] for point in all_points)
    rotate = raw_height > raw_width * 1.4

    def display(point: Point) -> Point:
        return (-point[1], point[0]) if rotate else point

    displayed = [display(point) for point in all_points]
    min_x, max_x = min(point[0] for point in displayed), max(point[0] for point in displayed)
    min_y, max_y = min(point[1] for point in displayed), max(point[1] for point in displayed)
    world_width, world_height = max(1.0, max_x - min_x), max(1.0, max_y - min_y)
    drawing_width = legend_x - margin * 2
    scale = min(drawing_width / world_width, (height - header - margin) / world_height)
    offset_x = margin + (drawing_width - world_width * scale) / 2
    offset_y = header + (height - header - world_height * scale) / 2

    def mapped(point: Point) -> Point:
        x, y = display(point)
        return offset_x + (x - min_x) * scale, offset_y + (max_y - y) * scale

    periodic_types = [
        pattern_type for pattern_type in result.types
        if pattern_type.hypothesis is not None
    ]
    periodic_index_by_internal_id = {
        pattern_type.type_id: index
        for index, pattern_type in enumerate(periodic_types)
    }
    owner_by_atom = {
        atom_id: pattern_type
        for pattern_type in result.types
        for atom_id in pattern_type.atom_ids
    }
    root_attributes = ""
    if source_info:
        attributes = {
            "data-case-id": source_info.case_id,
            "data-pdf-file": source_info.pdf_file_name,
            "data-pdf-page": source_info.original_page,
        }
        root_attributes = "".join(
            f' {name}="{xml_attribute(value)}"'
            for name, value in attributes.items()
            if value is not None
        )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"{root_attributes}>',
        '<rect width="100%" height="100%" fill="#fbfaf6"/>',
    ]
    if source_label:
        escaped_label = escape(source_label)
        lines.append(f"<title>{escaped_label}</title>")
        lines.append(
            f'<text x="36" y="34" font-family="Arial,sans-serif" font-size="17" '
            f'font-weight="600" fill="#39424e">{escaped_label}</text>'
        )
    for atom in atoms:
        owner = owner_by_atom.get(atom.id)
        is_linetype = owner is not None and owner.hypothesis is not None
        color = (
            type_color(periodic_index_by_internal_id[owner.type_id])
            if is_linetype else NON_LINETYPE_COLOR
        )
        semantic_class = "linetype" if is_linetype else "non-linetype"
        internal_type_id = owner.type_id if owner is not None else "unassigned"
        semantic_attributes = (
            f' data-semantic-class="{semantic_class}"'
            f' data-type-id="{xml_attribute(internal_type_id)}"'
        )
        stroke_width = max(2.5, min(7.0, (atom.line_width or 1) * scale))
        if atom.length <= EPS:
            x, y = mapped(atom.points[0])
            lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{max(1.6, stroke_width / 2):.3f}" fill="{color}" data-atom-id="{atom.id}"{semantic_attributes} data-degenerate="true" data-line-cap="{atom.line_cap}"/>')
        else:
            points = " ".join(f"{x:.3f},{y:.3f}" for x, y in map(mapped, atom.points))
            line_cap = ("butt", "round", "square")[atom.line_cap]
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="{stroke_width:.3f}" stroke-linecap="{line_cap}" stroke-linejoin="round" data-atom-id="{atom.id}"{semantic_attributes}/>')
    legend_y = 34
    for index, pattern_type in enumerate(periodic_types):
        color = type_color(index)
        kind = (
            "two-instance, carrier-validated"
            if isinstance(pattern_type.hypothesis, TwoInstanceHypothesis)
            else "periodic"
        )
        display_name = f"线型{index + 1}"
        label = escape(f"{display_name}: {kind} ({len(pattern_type.atom_ids)} atoms)")
        lines.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="Arial,sans-serif" font-size="16" '
            f'font-weight="700" fill="{color}" data-type-id="{xml_attribute(pattern_type.type_id)}" '
            f'data-display-name="{display_name}" data-semantic-class="linetype">{label}</text>'
        )
        legend_y += 25
    non_linetype_atom_count = sum(
        len(pattern_type.atom_ids)
        for pattern_type in result.types
        if pattern_type.hypothesis is None
    )
    if non_linetype_atom_count:
        label = escape(f"非线型: unassigned ({non_linetype_atom_count} atoms)")
        lines.append(
            f'<text x="{legend_x}" y="{legend_y}" font-family="Arial,sans-serif" font-size="16" '
            f'font-weight="700" fill="{NON_LINETYPE_COLOR}" data-type-id="non_linetype" '
            f'data-display-name="非线型" data-semantic-class="non-linetype">{label}</text>'
        )
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_input_path() -> Path:
    candidates = [Path.cwd() / "commands.txt", Path(__file__).resolve().with_name("commands.txt")]
    return next((path for path in candidates if path.is_file()), candidates[0])


def source_warnings(source: str) -> list[str]:
    words = {value for kind, value in lexical_tokens(source) if kind == "word"}
    warnings: list[str] = []
    if "Do" in words:
        warnings.append("Form/Image XObject operator Do is not expanded because a standalone txt file has no resource dictionary.")
    if "BI" in words:
        warnings.append("Inline image data is skipped; only vector paths are classified.")
    if words & {"Tj", "TJ", "'", '"'}:
        warnings.append("Native PDF text is ignored unless the glyphs have already been converted to vector paths.")
    if "d" in words:
        warnings.append("Native PDF dash state is treated as one vector centerline; dash marks are not expanded into atoms.")
    if "gs" in words:
        warnings.append("ExtGState resources are unavailable in a standalone txt file; gs is ignored instead of guessing its alpha.")
    return warnings


def process_command_file(input_path: Path, split_path: Path) -> list[str]:
    """Classify one command file and overwrite its requested SVG result."""
    source = input_path.read_text(encoding="latin1")
    warnings = source_warnings(source)
    atoms = parse_painted_atoms(source)
    if not atoms:
        raise ValueError("no painted vector subpaths were found in the input")
    result = discover_unknown_pattern_types(atoms)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    render_svg(atoms, result, split_path, case_source_info_for(input_path))
    return warnings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Split unknown repeated vector line patterns from PDF commands (Python 3.10+)."
    )
    parser.add_argument("input", nargs="?", type=Path, help="Command text file; default: ./commands.txt")
    parser.add_argument("-o", "--output", type=Path, help="Optional output directory")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every *commands*.txt in the current directory (non-recursive)",
    )
    args = parser.parse_args()

    if args.all:
        if args.input is not None:
            parser.error("input cannot be used together with --all")
        input_paths = sorted(
            (path for path in Path.cwd().glob("*commands*.txt") if path.is_file()),
            key=lambda path: path.name.lower(),
        )
        if not input_paths:
            parser.error("no *commands*.txt files found in the current directory")
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
        succeeded = 0
        failed = 0
        for index, input_path in enumerate(input_paths, start=1):
            split_path = (
                args.output / f"{input_path.stem}-split.svg"
                if args.output
                else input_path.with_name(f"{input_path.stem}-split.svg")
            )
            try:
                warnings = process_command_file(input_path, split_path)
            except Exception as error:
                failed += 1
                print(
                    f"[{index}/{len(input_paths)}] Failed: {input_path.name}: {error}",
                    file=sys.stderr,
                )
                continue
            succeeded += 1
            for warning in warnings:
                print(f"[{input_path.name}] Warning: {warning}", file=sys.stderr)
            print(f"[{index}/{len(input_paths)}] Result image: {split_path.resolve()}")
        print(f"Batch complete: {succeeded} succeeded, {failed} failed")
        return 1 if failed else 0

    input_path = args.input or default_input_path()
    if not input_path.is_file():
        parser.error(f"input file not found: {input_path}\nPut commands in commands.txt or pass the path explicitly.")
    if args.output:
        split_path = args.output / "split-result.svg"
    else:
        split_path = input_path.with_name(f"{input_path.stem}-split.svg")
    try:
        warnings = process_command_file(input_path, split_path)
    except ValueError as error:
        parser.error(str(error))
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    print(f"Result image: {split_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
