"""Trusted legend-swatch extraction and supervised cluster association.

This module is deliberately separate from the two unsupervised recognizers.
The upstream symbol stage has already established that each supplied box is a
line-type sample; consequently a sample is allowed to contain only one or two
periods.  We still fail closed when the authored, top-painted geometry cannot
be extracted or when the page does not contain repeated compatible evidence.

All boxes and returned tips use the application's page frame
``[y_min, x_min, y_max, x_max]`` (0..1000).  Shape and period comparisons are
performed in the engine's unscaled PDF-point frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics
import unicodedata

from line_type_engine.method1.serializer import serialize_path
from line_type_engine.method1.unknown_pattern_split import (
    Atom,
    distance,
    make_candidate,
    maximum_array_difference,
    mean,
    parse_painted_atoms,
    polyline_length,
    principal_frame,
    resample_polyline,
)


PAINT_ORDER_GAP = 64
MIN_HORIZONTAL_COVERAGE = 0.20
MIN_HORIZONTAL_SPAN = 0.50
MIN_VECTOR_HITS = 3
MIN_VECTOR_SUPPORT = 0.60
MAX_PERIOD_RATIO = 1.25
MIN_SCALE_RATIO = 0.45
MAX_SCALE_RATIO = 2.20
MIN_NORMALISED_WIDTH_RATIO = 0.55
SAMPLE_EXCLUSION_MARGIN = 3.0


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid box_2d: {value!r}")
    out = tuple(float(part) for part in value)
    if (not all(math.isfinite(part) for part in out)
            or out[2] <= out[0] or out[3] <= out[1]):
        raise ValueError(f"invalid box_2d: {value!r}")
    return out


def _intersects(left, right):
    return (min(left[2], right[2]) >= max(left[0], right[0])
            and min(left[3], right[3]) >= max(left[1], right[1]))


def _center(box):
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _inside_any(point, boxes):
    y, x = point
    return any(box[0] <= y <= box[2] and box[1] <= x <= box[3]
               for box in boxes)


def operation_lines(operation):
    """Return authored path geometry as IR-frame polylines."""
    lines = []
    current = []
    for segment in getattr(operation, "segments", ()) or ():
        if segment.kind == "move":
            if len(current) > 1:
                lines.append(current)
            current = [tuple(segment.end)]
        elif segment.kind == "line":
            current.append(tuple(segment.end))
        elif segment.kind == "curve":
            # The regular sidecar uses controls as a compact display
            # approximation.  Matching itself uses the serializer's sampled
            # atoms, so this does not affect recognition.
            current.extend((tuple(segment.control_1), tuple(segment.control_2),
                            tuple(segment.end)))
        elif segment.kind == "close" and len(current) > 1:
            current.append(current[0])
    if len(current) > 1:
        lines.append(current)
    return lines


def projected_bounds(bounds, to_page_frame):
    corners = [to_page_frame(x, y)
               for x in (bounds.min_x, bounds.max_x)
               for y in (bounds.min_y, bounds.max_y)]
    ys = [point[0] for point in corners]
    xs = [point[1] for point in corners]
    return [min(ys), min(xs), max(ys), max(xs)]


def _segment_touches_box(left, right, box, margin=1.0):
    """Liang-Barsky segment/box test in page-frame ``[y,x]`` order."""
    y0, x0, y1, x1 = (box[0] - margin, box[1] - margin,
                      box[2] + margin, box[3] + margin)
    start = (float(left[0]), float(left[1]))
    delta = (float(right[0]) - start[0], float(right[1]) - start[1])
    lower, upper = 0.0, 1.0
    for value, change, minimum, maximum in (
            (start[0], delta[0], y0, y1),
            (start[1], delta[1], x0, x1)):
        if abs(change) <= 1e-12:
            if value < minimum or value > maximum:
                return False
            continue
        enter = (minimum - value) / change
        leave = (maximum - value) / change
        if enter > leave:
            enter, leave = leave, enter
        lower = max(lower, enter)
        upper = min(upper, leave)
        if lower > upper:
            return False
    return True


def _path_touches_box(operation, box, to_page_frame):
    """Whether authored stroke topology actually reaches a sample box.

    An enclosing legend/table border has a bounds rectangle covering every
    interior swatch even though none of its strokes enter those small boxes.
    Bounds-only paint selection therefore picked the late container instead
    of Rapid City P4's real X/O/W samples.  Test projected segments, retaining
    a one-unit allowance for integer VLM/snap boxes and stroke width.
    """
    for line in operation_lines(operation):
        projected = [to_page_frame(x, y) for x, y in line]
        if any(_segment_touches_box(left, right, box)
               for left, right in zip(projected, projected[1:])):
            return True
    return False


def _merged_length(intervals):
    total = 0.0
    end = None
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if end is None or left > end:
            total += right - left
            end = right
        elif right > end:
            total += right - end
            end = right
    return total


def normalise_literal(value):
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.translate(str.maketrans({"\u2018": "'", "\u2019": "'",
                                        "\u2032": "'", "\u02b9": "'"}))
    return re.sub(r"\s+", " ", text).strip().upper()


def pattern_instances_outside_samples(instances, sample_boxes):
    """Return valid Method-2 instances whose centres are not source swatches.

    The matcher and the published pattern-evidence layer must use the same
    exclusion rule.  Otherwise a native-text legend sample can be excluded
    while choosing the target cluster, then accidentally reappear in the UI
    as if it were one of the repeated full-page instances (final P9 exposed
    this as 69 target ``SF`` instances but 71 published pattern boxes).
    """
    out = []
    for instance in instances or ():
        if not isinstance(instance, dict):
            continue
        box = instance.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            box = tuple(float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in box) \
                or box[2] <= box[0] or box[3] <= box[1]:
            continue
        if not _inside_any(_center(box), sample_boxes):
            out.append(instance)
    return out


@dataclass(frozen=True)
class LegendTemplate:
    sample_index: int
    symbol_index: int
    text_index: int
    box_2d: tuple[float, float, float, float]
    path_indices: tuple[int, ...]
    text_indices: tuple[int, ...]
    literals: tuple[str, ...]
    paint_order_min: int | None
    paint_order_max: int | None
    discarded_earlier_operations: int
    horizontal_coverage: float
    horizontal_span: float
    valid: bool
    reason: str | None

    def audit(self):
        return {
            "sample_index": self.sample_index,
            "symbol_index": self.symbol_index,
            "text_index": self.text_index,
            "box_2d": list(self.box_2d),
            "path_op_indices": list(self.path_indices),
            "text_op_indices": list(self.text_indices),
            "literal_texts": list(self.literals),
            "paint_order_min": self.paint_order_min,
            "paint_order_max": self.paint_order_max,
            "discarded_earlier_operations": self.discarded_earlier_operations,
            "horizontal_coverage": round(self.horizontal_coverage, 4),
            "horizontal_span": round(self.horizontal_span, 4),
            "valid": self.valid,
            "reason": self.reason,
        }


def extract_template(page, sample, sample_index, to_page_frame):
    """Extract only the latest authored paint tranche inside one trusted box.

    Background plan strokes can cross a legend sample.  The legend ink is
    authored above them, so a large paint-order discontinuity is a structural
    separator and is substantially safer than choosing by colour or width.
    """
    box = _box(sample.get("box_2d"))
    hits = []
    for op_index, operation in enumerate(page.operations):
        kind = getattr(operation, "kind", None)
        if kind not in {"path", "text"}:
            continue
        bounds = getattr(operation, "bounds", None)
        if bounds is None:
            continue
        # Apply the same one-unit snap allowance to the cheap bounds gate and
        # the exact segment test.  Without this, a stroke 0.5 units outside an
        # integer VLM box would be rejected before the tolerant test ran.
        coarse_box = ((box[0] - 1.0, box[1] - 1.0,
                       box[2] + 1.0, box[3] + 1.0)
                      if kind == "path" else box)
        if not _intersects(
                projected_bounds(bounds, to_page_frame), coarse_box):
            continue
        if kind == "path" \
                and not _path_touches_box(operation, box, to_page_frame):
            continue
        hits.append((int(operation.paint_order), op_index, operation))

    if not hits:
        return LegendTemplate(
            sample_index, int(sample["symbol_index"]), int(sample["text_index"]),
            box, (), (), (), None, None, 0, 0.0, 0.0, False,
            "no authored path/text operation intersects the sample box")

    orders = sorted({row[0] for row in hits})
    tranche_start = orders[0]
    for previous, current in zip(orders, orders[1:]):
        if current - previous > PAINT_ORDER_GAP:
            tranche_start = current
    selected = [row for row in hits if row[0] >= tranche_start]
    paths = tuple(row[1] for row in selected
                  if getattr(row[2], "kind", None) == "path")
    texts = tuple(row[1] for row in selected
                  if getattr(row[2], "kind", None) == "text")
    literals = tuple(dict.fromkeys(
        normalise_literal(getattr(page.operations[index], "literal_text", ""))
        for index in texts
        if normalise_literal(getattr(page.operations[index], "literal_text", ""))
    ))

    x0, x1 = box[1], box[3]
    horizontal = []
    all_x = []
    for op_index in paths:
        for line in operation_lines(page.operations[op_index]):
            projected = [to_page_frame(x, y) for x, y in line]
            for point in projected:
                if box[0] - 1 <= point[0] <= box[2] + 1:
                    all_x.append(min(x1, max(x0, point[1])))
            for left, right in zip(projected, projected[1:]):
                dy = abs(right[0] - left[0])
                dx = abs(right[1] - left[1])
                if dx <= 1e-6 or dx < 2.0 * dy:
                    continue
                lo = max(x0, min(left[1], right[1]))
                hi = min(x1, max(left[1], right[1]))
                if hi > lo:
                    horizontal.append((lo, hi))
    width = x1 - x0
    coverage = _merged_length(horizontal) / width
    span = ((max(all_x) - min(all_x)) / width) if len(all_x) >= 2 else 0.0
    reason = None
    if not paths:
        reason = "top paint tranche contains no path ink"
    elif not horizontal:
        reason = "top paint tranche contains no horizontal carrier"
    elif coverage < MIN_HORIZONTAL_COVERAGE or span < MIN_HORIZONTAL_SPAN:
        reason = ("horizontal carrier is too short "
                  f"(coverage={coverage:.3f}, span={span:.3f})")

    return LegendTemplate(
        sample_index, int(sample["symbol_index"]), int(sample["text_index"]),
        box, paths, texts, literals,
        min((row[0] for row in selected), default=None),
        max((row[0] for row in selected), default=None),
        len(hits) - len(selected), coverage, span, reason is None, reason)


def _candidate_of(operation):
    serialised = serialize_path(operation)
    if serialised is None:
        return []
    return parse_painted_atoms(serialised[0])


def _polygon_area(points):
    return abs(sum(left[0] * right[1] - right[0] * left[1]
                   for left, right in zip(points, points[1:]))) / 2.0


def _cycle_atom(source, points, atom_id):
    samples = resample_polyline(points, 16)
    center = (mean(point[0] for point in samples),
              mean(point[1] for point in samples))
    scale = max(1e-8, 2 * max(distance(point, center) for point in samples))
    direction, aspect = principal_frame(samples, center)
    return Atom(
        atom_id, list(points), samples, polyline_length(points), center, scale,
        aspect, direction, True, source.curve_segments,
        max(0, len(points) - 1), source.paint_mode, source.line_width,
        source.line_cap, source.stroke_color)


def embedded_cycle_atoms(atom):
    """Extract closed motif loops embedded in an otherwise open carrier.

    CAD exporters commonly author ``carrier -> square -> same junction ->
    carrier`` as one Path.  The normal Method-1 Atom is therefore open and its
    fingerprint includes the carrier.  Repeated junction vertices expose the
    closed subchain without rewriting the production engine's Atom.
    """
    points = atom.points
    if len(points) < 4:
        return []
    tolerance = max(1e-3, abs(float(atom.line_width or 0.0)) * 0.05)
    found = []
    identities = set()
    for left in range(len(points) - 3):
        # Prefer the smallest loop returning to this junction; larger nested
        # carrier excursions otherwise duplicate the same motif.
        for right in range(left + 3, len(points)):
            if distance(points[left], points[right]) > tolerance:
                continue
            cycle = list(points[left:right + 1])
            xs = [point[0] for point in cycle]
            ys = [point[1] for point in cycle]
            width, height = max(xs) - min(xs), max(ys) - min(ys)
            if min(width, height) <= tolerance * 2:
                continue
            if _polygon_area(cycle) <= tolerance * tolerance * 4:
                continue
            identity = tuple(round(value / tolerance) for value in (
                min(xs), min(ys), max(xs), max(ys)))
            if identity in identities:
                continue
            identities.add(identity)
            found.append(_cycle_atom(atom, cycle, -(len(found) + 1)))
            break
    return found


@dataclass(frozen=True)
class MarkerEvidence:
    op_index: int
    atom: Atom
    candidate: object
    embedded: bool


def marker_evidence(operation, op_index):
    evidence = []
    seen = set()
    for atom in _candidate_of(operation):
        candidates = [(atom, False)] if atom.closed else []
        candidates.extend((cycle, True) for cycle in embedded_cycle_atoms(atom))
        for marker, embedded in candidates:
            candidate = make_candidate(len(evidence), [marker])
            identity = tuple(round(value, 5) for value in (
                marker.center[0], marker.center[1], marker.scale,
                marker.aspect_ratio))
            if identity in seen:
                continue
            seen.add(identity)
            evidence.append(MarkerEvidence(
                op_index, marker, candidate, embedded))
    return evidence


def _shape_matches(left, right):
    a, b = left.fingerprint, right.fingerprint
    if any(a[key] != b[key] for key in (
            "member_count", "closed_count", "curved_count", "filled_count")):
        return False
    if abs(a["aspect_ratio"] - b["aspect_ratio"]) > 0.12:
        return False
    if abs(a["normalized_length"] - b["normalized_length"]) > (
            0.18 * max(1.0, min(a["normalized_length"],
                                b["normalized_length"]))):
        return False
    return all(maximum_array_difference(a[key], b[key]) <= tolerance
               for key, tolerance in (
                   ("member_length_ratios", 0.10),
                   ("center_distances", 0.12),
                   ("direction_angles", 0.10),
                   ("radial_quantiles", 0.10),
               ))


def _style_matches(left, right):
    if left.paint_mode != right.paint_mode or left.line_cap != right.line_cap:
        return False
    if len(left.stroke_color) != len(right.stroke_color):
        return False
    if any(abs(a - b) > 0.08
           for a, b in zip(left.stroke_color, right.stroke_color)):
        return False
    scale_ratio = right.scale / max(left.scale, 1e-8)
    if not MIN_SCALE_RATIO <= scale_ratio <= MAX_SCALE_RATIO:
        return False
    left_width = abs(left.line_width) / max(left.scale, 1e-8)
    right_width = abs(right.line_width) / max(right.scale, 1e-8)
    if left_width <= 1e-9 or right_width <= 1e-9:
        return left_width <= 1e-9 and right_width <= 1e-9
    ratio = min(left_width, right_width) / max(left_width, right_width)
    return ratio >= MIN_NORMALISED_WIDTH_RATIO


def _marker_matches(sample, target):
    return (_shape_matches(sample.candidate, target.candidate)
            and _style_matches(sample.atom, target.atom))


def _period(centers):
    if len(centers) < 2:
        return None
    nearest = []
    for index, center in enumerate(centers):
        distances = [distance(center, other)
                     for other_index, other in enumerate(centers)
                     if other_index != index and distance(center, other) > 1e-6]
        if distances:
            nearest.append(min(distances))
    return statistics.median(nearest) if nearest else None


def _period_compatible(left, right):
    if left is None or right is None or left <= 1e-8 or right <= 1e-8:
        return False
    return max(left, right) / min(left, right) <= MAX_PERIOD_RATIO


def _style_identity(operation):
    """Exact authored visual style used only for short-run fringe adoption."""
    def colour(value):
        return tuple(round(float(part), 6) for part in (value or ()))

    return (
        bool(getattr(operation, "stroke", False)),
        bool(getattr(operation, "fill", False)),
        colour(getattr(operation, "stroke_color", None)),
        colour(getattr(operation, "fill_color", None)),
        round(float(getattr(operation, "line_width", 0.0) or 0.0), 6),
        bool(getattr(operation, "hairline", False)),
        tuple(getattr(operation, "line_cap", ()) or ()),
        round(float(getattr(operation, "line_join", 0.0) or 0.0), 6),
        tuple(round(float(value), 6)
              for value in (getattr(operation, "dash_array", ()) or ())),
        round(float(getattr(operation, "dash_phase", 0.0) or 0.0), 6),
        round(float(getattr(operation, "stroke_opacity", 1.0) or 0.0), 6),
        round(float(getattr(operation, "fill_opacity", 1.0) or 0.0), 6),
    )


def _point_segment_distance(point, left, right):
    dx, dy = right[0] - left[0], right[1] - left[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return distance(point, left)
    ratio = ((point[0] - left[0]) * dx
             + (point[1] - left[1]) * dy) / denominator
    ratio = max(0.0, min(1.0, ratio))
    return distance(point, (left[0] + ratio * dx, left[1] + ratio * dy))


def _closest_segment_alignment(left_lines, right_lines):
    """Return (minimum distance, absolute tangent dot) for two run geometries."""
    best = None
    for left_line in left_lines:
        for a, b in zip(left_line, left_line[1:]):
            ab = (b[0] - a[0], b[1] - a[1])
            ab_length = math.hypot(*ab)
            if ab_length <= 1e-9:
                continue
            for right_line in right_lines:
                for c, d in zip(right_line, right_line[1:]):
                    cd = (d[0] - c[0], d[1] - c[1])
                    cd_length = math.hypot(*cd)
                    if cd_length <= 1e-9:
                        continue
                    gap = min(
                        _point_segment_distance(a, c, d),
                        _point_segment_distance(b, c, d),
                        _point_segment_distance(c, a, b),
                        _point_segment_distance(d, a, b),
                    )
                    alignment = abs(
                        (ab[0] * cd[0] + ab[1] * cd[1])
                        / (ab_length * cd_length))
                    if best is None or gap < best[0] - 1e-8 \
                            or (abs(gap - best[0]) <= 1e-8
                                and alignment > best[1]):
                        best = gap, alignment
    return best if best is not None else (math.inf, 0.0)


def _expand_fringe_runs(row, *, cluster_ops, operations, run_of, group_of,
                        sample_boxes, to_page_frame, incompatible):
    """Adopt short same-style carrier fragments adjacent to proven runs.

    Compound CAD paths sometimes split end caps at a dash gap, so they cannot
    independently provide three motifs.  They may follow a proven run only
    with exact style, same engine group, adjacent authored order, compatible
    tangent, bounded spatial gap, and a small total-op budget.  Contradictory
    periodic runs and hairline/background companions are never adopted.
    """
    number = row["number"]
    path_indices = [op_index for op_index in _outside_ops(
        cluster_ops[number], operations, sample_boxes, to_page_frame)
        if getattr(operations[op_index], "kind", None) == "path"]
    by_run = {}
    for op_index in path_indices:
        by_run.setdefault(str(run_of.get(op_index, 1)), []).append(op_index)
    accepted = set(row["run_ids"])
    budget = max(8, int(math.ceil(row["hit_count"] * 0.25)))
    absorbed_ops = 0
    fringe = []

    def lines_of(run_id):
        return [line for op_index in by_run.get(run_id, ())
                for line in operation_lines(operations[op_index])]

    while True:
        changed = False
        for candidate in sorted(set(by_run) - accepted - set(incompatible)):
            indices = by_run[candidate]
            if absorbed_ops + len(indices) > budget:
                continue
            styles = {_style_identity(operations[index]) for index in indices}
            if len(styles) != 1:
                continue
            candidate_style = next(iter(styles))
            candidate_groups = {group_of.get(index) for index in indices
                                if group_of.get(index) is not None}
            candidate_orders = {int(operations[index].paint_order)
                                for index in indices}
            candidate_lines = lines_of(candidate)
            period_limit = MAX_PERIOD_RATIO * statistics.median(row["periods"])
            adopted = False
            for seed in sorted(accepted):
                seed_indices = by_run.get(seed, ())
                if not seed_indices:
                    continue
                if any(_style_identity(operations[index]) != candidate_style
                       for index in seed_indices):
                    continue
                seed_groups = {group_of.get(index) for index in seed_indices
                               if group_of.get(index) is not None}
                if not candidate_groups or not (candidate_groups & seed_groups):
                    continue
                seed_orders = {int(operations[index].paint_order)
                               for index in seed_indices}
                if min(abs(left - right) for left in candidate_orders
                       for right in seed_orders) > 1:
                    continue
                gap, alignment = _closest_segment_alignment(
                    candidate_lines, lines_of(seed))
                if gap <= period_limit and alignment >= 0.90:
                    adopted = True
                    break
            if not adopted:
                continue
            accepted.add(candidate)
            fringe.append(candidate)
            absorbed_ops += len(indices)
            changed = True
        if not changed:
            break
    row["run_ids"] = sorted(accepted)
    row["fringe_run_ids"] = sorted(fringe)
    return row


def _op_page_box(operation, to_page_frame):
    return projected_bounds(operation.bounds, to_page_frame)


def _outside_ops(indices, operations, sample_boxes, to_page_frame):
    out = []
    excluded = [(box[0] - SAMPLE_EXCLUSION_MARGIN,
                 box[1] - SAMPLE_EXCLUSION_MARGIN,
                 box[2] + SAMPLE_EXCLUSION_MARGIN,
                 box[3] + SAMPLE_EXCLUSION_MARGIN)
                for box in sample_boxes]
    for op_index in indices:
        if not 0 <= op_index < len(operations):
            continue
        # Tight VLM/snap boxes commonly end at the carrier centreline while
        # the authored stroke extends another 1–2 page-frame units.  Centre
        # tests therefore leak the swatch itself as apparent page evidence.
        # Exclude every op touching a small expanded sample neighbourhood.
        op_box = _op_page_box(operations[op_index], to_page_frame)
        if not any(_intersects(op_box, box) for box in excluded):
            out.append(op_index)
    return out


def _run_tips(op_indices, operations, run_of, to_page_frame):
    by_run = {}
    for op_index in op_indices:
        run_id = str(run_of.get(op_index, 1))
        box = _op_page_box(operations[op_index], to_page_frame)
        if run_id not in by_run:
            by_run[run_id] = list(box)
        else:
            current = by_run[run_id]
            current[0] = min(current[0], box[0])
            current[1] = min(current[1], box[1])
            current[2] = max(current[2], box[2])
            current[3] = max(current[3], box[3])
    return [_center(by_run[run_id]) for run_id in sorted(by_run)], sorted(by_run)


def _text_match(template, pattern_instances, cluster_ops, operations,
                run_of, sample_boxes, to_page_frame):
    queries = {literal for literal in template.literals
               if any(character.isalnum() for character in literal)}
    if not queries:
        return None
    candidates = []
    for number, instances in pattern_instances.items():
        matches = []
        for instance in pattern_instances_outside_samples(
                instances, sample_boxes):
            literal = normalise_literal(instance.get("literal_text"))
            if literal in queries:
                matches.append(instance)
        if matches:
            candidates.append((len(matches), int(number), matches))
    candidates.sort(reverse=True)
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return {"status": "ambiguous", "reason": "literal matches multiple clusters",
                "candidate_line_type_numbers": [row[1] for row in candidates]}
    _count, number, matches = candidates[0]
    outside = _outside_ops(cluster_ops.get(number, ()), operations,
                           sample_boxes, to_page_frame)
    if not outside:
        return None
    tips = [_center(instance["bbox"]) for instance in matches]
    _run_centers, run_ids = _run_tips(
        outside, operations, run_of, to_page_frame)
    return {
        "status": "matched",
        "match_kind": "native_text",
        "primary_line_type_number": number,
        "matched_line_type_numbers": [number],
        "matched_runs_by_line_type": {str(number): run_ids},
        "tips": tips,
        "sample_literals": sorted(queries),
        "target_instance_count": len(matches),
    }


def _owner_overlap_match(template, owner, cluster_ops, operations, run_of,
                         sample_boxes, to_page_frame):
    counts = {}
    for op_index in template.path_indices:
        number = owner.get(op_index)
        if number is not None:
            counts[number] = counts.get(number, 0) + 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda row: (-row[1], row[0]))
    number, count = ranked[0]
    if (count / max(1, len(template.path_indices)) < 0.60
            or (len(ranked) > 1 and ranked[1][1] == count)):
        return None
    outside = _outside_ops(cluster_ops.get(number, ()), operations,
                           sample_boxes, to_page_frame)
    if not outside:
        return None
    tips, run_ids = _run_tips(outside, operations, run_of, to_page_frame)
    return {
        "status": "matched",
        "match_kind": "owned_sample_continuation",
        "primary_line_type_number": int(number),
        "matched_line_type_numbers": [int(number)],
        "matched_runs_by_line_type": {str(number): run_ids},
        "tips": tips,
        "sample_owned_path_count": count,
    }


def _vector_match(template, cluster_ops, operations, run_of, group_of,
                  sample_boxes, to_page_frame):
    samples = [evidence
               for op_index in template.path_indices
               for evidence in marker_evidence(operations[op_index], op_index)]
    if not samples:
        return None

    # The sample period comes from repeated copies of one compatible marker;
    # a one-period sample intentionally leaves it unknown.
    exemplar = samples[0]
    family = [item for item in samples if _marker_matches(exemplar, item)]
    if len(family) < max(1, len(samples) // 2):
        families = []
        for candidate in samples:
            members = [item for item in samples if _marker_matches(candidate, item)]
            families.append((len(members), candidate, members))
        _size, exemplar, family = max(families, key=lambda row: row[0])
    sample_period = _period([item.atom.center for item in family])

    run_rows = []
    incompatible_runs = {}
    for number, raw_indices in cluster_ops.items():
        indices = _outside_ops(raw_indices, operations, sample_boxes, to_page_frame)
        if not indices:
            continue
        all_markers = [evidence for op_index in indices
                       if getattr(operations[op_index], "kind", None) == "path"
                       for evidence in marker_evidence(
                           operations[op_index], op_index)]
        # Closed cycles embedded in a carrier are direct evidence.  Standalone
        # markers remain eligible because many CAD exporters split the carrier
        # and marker into adjacent Path operations; ownership by an already
        # recognized line cluster supplies the attachment evidence there.
        structurally_eligible = [item for item in all_markers
                                 if item.atom.closed
                                 and item.atom.curve_segments
                                 == exemplar.atom.curve_segments
                                 and item.atom.paint_mode
                                 == exemplar.atom.paint_mode]
        # A global engine cluster may itself contain spatially disconnected
        # profiles with different periods.  Validate each connected run, then
        # union only compatible runs.  Final P3 #19 is the canonical case: r1
        # has a 13.0-pt period while r2 is the true 17.2-pt 4' fence sample.
        by_run = {}
        for item in structurally_eligible:
            by_run.setdefault(str(run_of.get(item.op_index, 1)), []).append(item)
        for run_id, eligible in by_run.items():
            hits = [item for item in eligible
                    if _marker_matches(exemplar, item)]
            # One op can contain more than one nested description of the same
            # junction.  Count it once in support, but retain every center for
            # period and visual evidence.
            hit_ops = sorted({item.op_index for item in hits})
            eligible_ops = {item.op_index for item in eligible}
            support = len(hit_ops) / max(1, len(eligible_ops))
            if len(hit_ops) < MIN_VECTOR_HITS or support < MIN_VECTOR_SUPPORT:
                continue
            target_period = _period([item.atom.center for item in hits])
            if target_period is None:
                continue
            if sample_period is not None \
                    and not _period_compatible(sample_period, target_period):
                incompatible_runs.setdefault(int(number), set()).add(run_id)
                continue
            run_rows.append({
                "number": int(number),
                "run_id": run_id,
                "hit_count": len(hit_ops),
                "support": support,
                "period": target_period,
                "tips": [to_page_frame(*item.atom.center) for item in hits],
                "hit_op_indices": hit_ops,
            })

    if not run_rows:
        return None
    run_rows.sort(key=lambda row: (
        -row["hit_count"], -row["support"], row["number"], row["run_id"]))
    primary_run = run_rows[0]
    if sample_period is None:
        compatible = []
        for row in run_rows:
            if _period_compatible(primary_run["period"], row["period"]):
                compatible.append(row)
            else:
                incompatible_runs.setdefault(row["number"], set()).add(
                    row["run_id"])
        run_rows = compatible
    by_number = {}
    for row in run_rows:
        aggregate = by_number.setdefault(row["number"], {
            "number": row["number"], "hit_count": 0, "supports": [],
            "periods": [], "run_ids": [], "tips": [], "hit_op_indices": [],
        })
        aggregate["hit_count"] += row["hit_count"]
        aggregate["supports"].append(row["support"])
        aggregate["periods"].append(row["period"])
        aggregate["run_ids"].append(row["run_id"])
        aggregate["tips"].extend(row["tips"])
        aggregate["hit_op_indices"].extend(row["hit_op_indices"])
    rows = list(by_number.values())
    rows.sort(key=lambda row: (
        -row["hit_count"], -statistics.mean(row["supports"]), row["number"]))
    primary = rows[0]
    # Add only structurally attached short carrier/end-cap runs.  Do not copy
    # an entire cluster: P3 #16 also owns two width-0 hairline companions, and
    # #19 owns a different-period run.
    for row in rows:
        _expand_fringe_runs(
            row, cluster_ops=cluster_ops, operations=operations,
            run_of=run_of, group_of=group_of or {},
            sample_boxes=sample_boxes, to_page_frame=to_page_frame,
            incompatible=incompatible_runs.get(row["number"], set()))
    numbers = [row["number"] for row in rows]
    return {
        "status": "matched",
        "match_kind": "vector_motif",
        "primary_line_type_number": primary["number"],
        "matched_line_type_numbers": numbers,
        "matched_runs_by_line_type": {
            str(row["number"]): row["run_ids"] for row in rows},
        "tips": [tip for row in rows for tip in row["tips"]],
        "sample_marker_count": len(family),
        "sample_period_pt": (round(sample_period, 4)
                             if sample_period is not None else None),
        "cluster_evidence": [{
            "line_type_number": row["number"],
            "hit_count": row["hit_count"],
            "support": round(statistics.mean(row["supports"]), 4),
            "period_pt": round(statistics.median(row["periods"]), 4),
            "run_ids": row["run_ids"],
            "fringe_run_ids": row.get("fringe_run_ids") or [],
            "hit_op_indices": sorted(set(row["hit_op_indices"])),
        } for row in rows],
    }


def associate_template(template, *, pattern_instances, cluster_ops, owner,
                       operations, run_of, sample_boxes, to_page_frame,
                       group_of=None):
    """Associate one trusted template with recognized full-page clusters.

    Native PDF text is the strongest identity (``8'``, ``SF``).  An already
    owned sample continuation is next.  Vector motifs are the guarded fallback
    and may merge multiple engine clusters when their period/style evidence is
    compatible; this repairs short-run splits without changing Method 1/2.
    """
    if not template.valid:
        return {"status": "invalid", "reason": template.reason}
    matched = _text_match(
        template, pattern_instances, cluster_ops, operations, run_of,
        sample_boxes, to_page_frame)
    if matched is None:
        matched = _owner_overlap_match(
            template, owner, cluster_ops, operations, run_of,
            sample_boxes, to_page_frame)
    if matched is None:
        matched = _vector_match(
            template, cluster_ops, operations, run_of, group_of,
            sample_boxes, to_page_frame)
    if matched is None:
        return {"status": "no_match",
                "reason": "no compatible repeated full-page cluster"}
    return matched


__all__ = [
    "LegendTemplate", "associate_template", "embedded_cycle_atoms",
    "extract_template", "marker_evidence", "normalise_literal",
    "operation_lines", "pattern_instances_outside_samples",
    "projected_bounds",
]
