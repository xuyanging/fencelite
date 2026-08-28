"""Recover text-callout leaders from page vector geometry.

The arrow sidecar intentionally uses PDF paint order as one of its clustering
signals.  CAD exports do not guarantee that a callout's shoulder, elbow and
arrowhead are painted next to each other, so a perfectly visible leader can be
left unresolved.  This module is the order-independent fallback: it starts at
the *boundary* of a known text rectangle and follows geometrically connected
strokes to their real terminal.

The fallback is deliberately conservative.  A complete filled or open-V
arrowhead wins over every bare candidate.  Bare leaders are opt-in per key and
must look like an isolated annotation stroke, not a table border, title
underline, hatch or drawing grid.

All public coordinates use the shared page frame ``[y, x]`` / ``[y0,x0,y1,x1]``
in 0..1000.  Geometry is evaluated in ``core.vecgeom`` render pixels (long side
about 3000 px), where page rotation has already been applied.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

from core.vecgeom import _extract_page

# Render-pixel thresholds (core.vecgeom renders the long page side near 3000).
ROOT_MIN = 12.0
ROOT_HEIGHT_FACTOR = 2.1
ROOT_MAX = 45.0
ROOT_PROGRESS = 2.0
ROOT_CORNER_SLACK = 8.0
CONNECT_MIN = 1.2
CONNECT_MAX = 2.0
MIN_EDGE = 1.0
MIN_RESULT_LENGTH = 15.0
MIN_OPEN_RESULT_LENGTH = 30.0
MIN_BARE_RESULT_LENGTH = 45.0
MAX_DEPTH = 7
MAX_PATH_LENGTH = 1500.0
MAX_BRANCH = 4

HEAD_MIN_SIDE = 1.2
HEAD_MAX_SIDE = 24.0
HEAD_TOUCH = 3.0
HEAD_GRID = 16.0
OPEN_HEAD_MIN = 2.0
OPEN_HEAD_MAX = 18.0
OPEN_HEAD_ANGLE_MIN = 20.0
OPEN_HEAD_ANGLE_MAX = 120.0

BARE_NORMAL_MIN = 0.45
PARALLEL_DISTANCE = 8.0
PARALLEL_ANGLE = 8.0
PARALLEL_OVERLAP = 0.50
TIP_BOX = 3.0

_REJECT_LABELS = {"view title", "note", "legend entry"}
_TITLE_TEXT = re.compile(r"\b(?:DETAIL|ELEVATION)\s*$", re.I)


def _poly(item):
    """A vecgeom item as a display polyline in render-pixel coordinates."""
    op = item[0]
    v = item[1:]
    if op == "l":
        return [(v[0], v[1]), (v[2], v[3])]
    if op == "c":
        # The controls preserve the visible curve for the frontend; graph
        # connectivity itself only uses the first and last points.
        return [(v[0], v[1]), (v[2], v[3]),
                (v[4], v[5]), (v[6], v[7])]
    if op == "re":
        x0, y0, x1, y1 = v[:4]
        return [(x0, y0), (x1, y0), (x1, y1),
                (x0, y1), (x0, y0)]
    if op == "qu":
        return [(v[0], v[1]), (v[2], v[3]),
                (v[4], v[5]), (v[6], v[7]), (v[0], v[1])]
    return []


def _path_length(points):
    return sum(math.dist(points[i], points[i + 1])
               for i in range(len(points) - 1))


def _to_frame(points, width, height):
    out = []
    for x, y in points:
        out.append([
            round(max(0.0, min(1000.0, y / height * 1000.0)), 1),
            round(max(0.0, min(1000.0, x / width * 1000.0)), 1),
        ])
    return out


def _box_to_pixels(box, width, height):
    y0, x0, y1, x1 = [float(v) for v in box]
    return (x0 / 1000.0 * width, y0 / 1000.0 * height,
            x1 / 1000.0 * width, y1 / 1000.0 * height)


def _outside_distance(point, rect):
    x, y = point
    x0, y0, x1, y1 = rect
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return math.hypot(dx, dy)


def _boundary_distance(point, rect):
    """Distance to the rectangle boundary, including points inside it."""
    x, y = point
    x0, y0, x1, y1 = rect
    if x0 <= x <= x1 and y0 <= y <= y1:
        return min(x - x0, x1 - x, y - y0, y1 - y)
    return _outside_distance(point, rect)


def _departure_ratio(root, other, rect):
    """Positive normal component of root->other relative to the nearest edge."""
    x, y = root
    x0, y0, x1, y1 = rect
    distances = [
        (abs(x - x0), (-1.0, 0.0)),
        (abs(x - x1), (1.0, 0.0)),
        (abs(y - y0), (0.0, -1.0)),
        (abs(y - y1), (0.0, 1.0)),
    ]
    nearest = min(v for v, _n in distances)
    # Boxes produced from multi-line OCR can put the visual attachment a few
    # pixels inside a corner.  Treat both adjacent sides as boundary normals;
    # otherwise a genuine diagonal can look tangent to the mathematically
    # nearest side (Taylor P36's TENSION WIRE).
    normals = [n for v, n in distances
               if v <= nearest + ROOT_CORNER_SLACK]
    vx, vy = other[0] - x, other[1] - y
    length = math.hypot(vx, vy)
    if length <= 0:
        return 0.0
    return max(0.0, max((vx * nx + vy * ny) / length
                        for nx, ny in normals))


def _style(unit, dash):
    return (unit[5], round(float(unit[7]), 2), str(dash or ""))


def _styles_compatible(first, second):
    if first[0] >= 0 and second[0] >= 0 and first[0] != second[0]:
        return False
    tolerance = max(0.12, 0.25 * max(first[1], second[1]))
    return abs(first[1] - second[1]) <= tolerance and first[2] == second[2]


def _anchor_allowed(label, text):
    label_norm = str(label or "").strip().lower()
    if label_norm in _REJECT_LABELS:
        return False
    # Vector scanning occasionally calls a drawing title a supplement.  The
    # title underline and nearby dimension arrow are a convincing geometric
    # false positive, so reject only title-shaped endings.  A VLM callout keeps
    # precedence even if its prose happens to contain the same words.
    if label_norm != "callout" and _TITLE_TEXT.search(str(text or "").strip()):
        return False
    return True


def _tip_box(point):
    x, y = point
    return (x - TIP_BOX, y - TIP_BOX, x + TIP_BOX, y + TIP_BOX)


def _frame_box(pixel_box, width, height):
    x0, y0, x1, y1 = pixel_box
    return [
        round(max(0.0, min(1000.0, y0 / height * 1000.0)), 1),
        round(max(0.0, min(1000.0, x0 / width * 1000.0)), 1),
        round(max(0.0, min(1000.0, y1 / height * 1000.0)), 1),
        round(max(0.0, min(1000.0, x1 / width * 1000.0)), 1),
    ]


def _parallel_repeat(edge_index, edges):
    """Whether a bare seed belongs to a repeated grid/table/hatch family."""
    edge = edges[edge_index]
    ax, ay = edge["a"]
    bx, by = edge["b"]
    vx, vy = bx - ax, by - ay
    length = math.hypot(vx, vy)
    if length < MIN_RESULT_LENGTH:
        return True
    ux, uy = vx / length, vy / length
    sin_limit = math.sin(math.radians(PARALLEL_ANGLE))
    for other_index, other in enumerate(edges):
        if other_index == edge_index:
            continue
        cx, cy = other["a"]
        dx, dy = other["b"]
        ovx, ovy = dx - cx, dy - cy
        other_length = math.hypot(ovx, ovy)
        if other_length < length * 0.5:
            continue
        oux, ouy = ovx / other_length, ovy / other_length
        if abs(ux * ouy - uy * oux) > sin_limit:
            continue
        # Perpendicular distance of the other line to this one.
        perpendicular = abs((cx - ax) * uy - (cy - ay) * ux)
        if not 0.8 <= perpendicular <= PARALLEL_DISTANCE:
            continue
        p0 = (cx - ax) * ux + (cy - ay) * uy
        p1 = (dx - ax) * ux + (dy - ay) * uy
        overlap = max(0.0, min(length, max(p0, p1)) - max(0.0, min(p0, p1)))
        if overlap >= PARALLEL_OVERLAP * min(length, other_length):
            return True
    return False


def text_box_leaders(pdf_path, page_index, anchors, allow_bare_keys=None):
    """Recover unresolved text leaders without using PDF paint order.

    ``anchors`` contains ``(key, box_2d, label, text)`` tuples.  Bare leaders
    are considered only for keys explicitly present in ``allow_bare_keys``;
    complete filled/open arrowheads are always preferred.  Returned entries are
    compatible with ``steps.arrows.find_page_arrows``.
    """
    if not anchors:
        return {}
    allow_bare = set(allow_bare_keys or ())
    usable = []
    for row in anchors:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            continue
        key, box, label, text = row
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        if _anchor_allowed(label, text):
            usable.append((key, box, str(label or ""), str(text or "")))
    if not usable:
        return {}

    data = _extract_page(pdf_path, page_index)
    units, geom = data["units"], data["geom"]
    dashes = data["dashes"]
    width, height = data["w"], data["h"]
    if not units or not width or not height:
        return {}

    edges = []
    heads = []
    endpoint_grid = defaultdict(list)
    seed_grid = defaultdict(list)
    head_grid = defaultdict(list)
    seed_cell = 8.0

    for unit_index, unit in enumerate(units):
        filled = unit[6] != -1
        uw, uh = unit[3] - unit[1], unit[4] - unit[2]
        unit_style = _style(unit, dashes[unit_index])
        if (filled and min(uw, uh) > HEAD_MIN_SIDE
                and max(uw, uh) <= HEAD_MAX_SIDE):
            strokes = [_poly(item) for item in geom[unit_index]]
            strokes = [points for points in strokes if len(points) >= 2]
            # Filled rectangles/quads are overwhelmingly posts, rail sections,
            # table cells and CAD masking blocks.  Arrowheads and dot terminals
            # contain an explicit line or curve path.
            has_open_geometry = any(item[0] in ("l", "c")
                                    for item in geom[unit_index])
            if strokes and has_open_geometry:
                head_index = len(heads)
                heads.append({
                    "bbox": (unit[1], unit[2], unit[3], unit[4]),
                    "strokes": strokes,
                    "style": unit_style,
                })
                x0, y0, x1, y1 = heads[-1]["bbox"]
                for gx in range(int((x0 - HEAD_TOUCH) // HEAD_GRID),
                                int((x1 + HEAD_TOUCH) // HEAD_GRID) + 1):
                    for gy in range(int((y0 - HEAD_TOUCH) // HEAD_GRID),
                                    int((y1 + HEAD_TOUCH) // HEAD_GRID) + 1):
                        head_grid[gx, gy].append(head_index)
        if filled:
            continue
        unit_style = _style(unit, dashes[unit_index])
        for item in geom[unit_index]:
            points = _poly(item)
            # Rectangle/quadrilateral outlines are overwhelmingly table,
            # border and hatch geometry.  A leader primitive is open.
            if item[0] not in ("l", "c") or len(points) < 2:
                continue
            length = _path_length(points)
            if length < MIN_EDGE:
                continue
            edge_index = len(edges)
            edge = {
                "a": points[0], "b": points[-1], "points": points,
                "length": length, "style": unit_style,
            }
            edges.append(edge)
            connection = max(CONNECT_MIN, min(
                CONNECT_MAX, 1.5 * max(unit_style[1], 0.1)))
            edge["connect"] = connection
            for point in (edge["a"], edge["b"]):
                endpoint_grid[round(point[0] / CONNECT_MAX),
                              round(point[1] / CONNECT_MAX)].append(edge_index)
                seed_grid[int(point[0] // seed_cell),
                          int(point[1] // seed_cell)].append(edge_index)

    if not edges:
        return {}

    def incident(point):
        gx = round(point[0] / CONNECT_MAX)
        gy = round(point[1] / CONNECT_MAX)
        found = []
        seen = set()
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for edge_index in endpoint_grid.get((ix, iy), ()):
                    if edge_index in seen:
                        continue
                    edge = edges[edge_index]
                    distances = (math.dist(point, edge["a"]),
                                 math.dist(point, edge["b"]))
                    tolerance = edge["connect"]
                    if min(distances) <= tolerance:
                        seen.add(edge_index)
                        found.append((edge_index,
                                      0 if distances[0] <= distances[1] else 1))
        return found

    def filled_head(point, incoming, leader_style):
        gx = int(point[0] // HEAD_GRID)
        gy = int(point[1] // HEAD_GRID)
        candidates = []
        seen = set()
        for ix in range(gx - 1, gx + 2):
            for iy in range(gy - 1, gy + 2):
                for head_index in head_grid.get((ix, iy), ()):
                    if head_index in seen:
                        continue
                    seen.add(head_index)
                    head = heads[head_index]
                    # A large class of CAD exports paints the triangle as a
                    # fill-only path (no stroke, width 0, empty dash).  Its
                    # stroke style therefore carries no compatibility signal.
                    # Arrow outlines are commonly painted thinner than the
                    # leader (and some exporters reset dash state at fills).
                    # Stroke colour is the only reliable style signal here.
                    if (head["style"][0] >= 0 and leader_style[0] >= 0
                            and head["style"][0] != leader_style[0]):
                        continue
                    x0, y0, x1, y1 = head["bbox"]
                    dx = max(x0 - point[0], 0.0, point[0] - x1)
                    dy = max(y0 - point[1], 0.0, point[1] - y1)
                    touch = math.hypot(dx, dy)
                    if touch > HEAD_TOUCH:
                        continue
                    # CADs disagree about whether the leader terminates at the
                    # triangle tip or its base.  Compactness + physical contact
                    # are stable; centroid direction is not.
                    candidates.append((touch, head_index))
        return min(candidates)[1] if candidates else None

    def open_head(point, previous, used, leader_style):
        incoming = (point[0] - previous[0], point[1] - previous[1])
        incoming_length = math.hypot(*incoming)
        if incoming_length <= 0:
            return None
        ux, uy = incoming[0] / incoming_length, incoming[1] / incoming_length
        arms = []
        for edge_index, endpoint in incident(point):
            if edge_index in used:
                continue
            edge = edges[edge_index]
            if not _styles_compatible(leader_style, edge["style"]):
                continue
            if not OPEN_HEAD_MIN <= edge["length"] <= OPEN_HEAD_MAX:
                continue
            remote = edge["b"] if endpoint == 0 else edge["a"]
            vx, vy = remote[0] - point[0], remote[1] - point[1]
            length = math.hypot(vx, vy)
            # Both arms must open behind the tip.
            if vx * ux + vy * uy > -0.10 * length:
                continue
            arms.append((edge_index, (vx / length, vy / length)))
        for i in range(len(arms)):
            for j in range(i + 1, len(arms)):
                dot = max(-1.0, min(1.0,
                    arms[i][1][0] * arms[j][1][0]
                    + arms[i][1][1] * arms[j][1][1]))
                angle = math.degrees(math.acos(dot))
                if OPEN_HEAD_ANGLE_MIN <= angle <= OPEN_HEAD_ANGLE_MAX:
                    return [arms[i][0], arms[j][0]]
        return None

    out = {}
    for key, box, label, _text in usable:
        rect = _box_to_pixels(box, width, height)
        root_gap = min(ROOT_MAX, max(
            ROOT_MIN, ROOT_HEIGHT_FACTOR * max(1.0, rect[3] - rect[1])))
        candidate_edges = set()
        for gx in range(int((rect[0] - root_gap) // seed_cell),
                        int((rect[2] + root_gap) // seed_cell) + 1):
            for gy in range(int((rect[1] - root_gap) // seed_cell),
                            int((rect[3] + root_gap) // seed_cell) + 1):
                candidate_edges.update(seed_grid.get((gx, gy), ()))

        seeds = []
        for edge_index in candidate_edges:
            edge = edges[edge_index]
            for orientation, (root, other) in enumerate(
                    ((edge["a"], edge["b"]), (edge["b"], edge["a"]))):
                gap = _boundary_distance(root, rect)
                if gap > root_gap or edge["length"] < 5.0:
                    continue
                departure = _departure_ratio(root, other, rect)
                # A shoulder parallel to the text edge is allowed only when it
                # actually travels out beyond the text rectangle; headed path
                # validation below then requires a second segment.
                if (_outside_distance(other, rect)
                        < _outside_distance(root, rect) + ROOT_PROGRESS):
                    continue
                seeds.append((gap, -departure, edge_index, orientation))

        headed_seeds = []
        bare_paths = []
        for gap, neg_departure, seed_index, orientation in sorted(seeds)[:60]:
            seed = edges[seed_index]
            root, other = ((seed["a"], seed["b"])
                           if orientation == 0 else (seed["b"], seed["a"]))
            departure = -neg_departure
            stack = [(other, root, [seed_index], seed["length"])]
            found_heads = []
            local_bare = []
            while stack:
                point, previous, path, path_length = stack.pop()
                incoming = (point[0] - previous[0], point[1] - previous[1])
                head_index = filled_head(point, incoming, seed["style"])
                if head_index is not None:
                    found_heads.append({
                        "path": path, "tip": point,
                        "kind": "filled", "head": head_index,
                        "incoming": incoming,
                    })
                    continue
                open_edges = open_head(point, previous, set(path), seed["style"])
                if open_edges:
                    found_heads.append({
                        "path": path, "tip": point,
                        "kind": "open", "head": open_edges,
                        "incoming": incoming,
                    })
                    continue
                if len(path) >= MAX_DEPTH or path_length >= MAX_PATH_LENGTH:
                    continue
                neighbours = []
                for next_index, endpoint in incident(point):
                    if next_index in path:
                        continue
                    next_edge = edges[next_index]
                    if not _styles_compatible(seed["style"], next_edge["style"]):
                        continue
                    remote = next_edge["b"] if endpoint == 0 else next_edge["a"]
                    if math.dist(point, remote) < OPEN_HEAD_MIN:
                        continue
                    # Do not reverse back along the incoming stroke.
                    vx, vy = incoming
                    nx, ny = remote[0] - point[0], remote[1] - point[1]
                    denom = math.hypot(vx, vy) * math.hypot(nx, ny)
                    if denom and (vx * nx + vy * ny) / denom < -0.2:
                        continue
                    neighbours.append((next_index, remote, next_edge["length"]))
                if not neighbours:
                    local_bare.append((path, point, path_length, "free"))
                    continue
                if len(neighbours) > MAX_BRANCH:
                    local_bare.append((path, point, path_length, "junction"))
                    continue
                for next_index, remote, next_length in neighbours:
                    stack.append((remote, point, path + [next_index],
                                  path_length + next_length))

            # A tangent shoulder is useful only when it leads through an elbow;
            # one tangent segment to a nearby dimension arrow is a title/table
            # false positive.
            accepted_heads = []
            for row in found_heads:
                total = sum(edges[i]["length"] for i in row["path"])
                minimum = (MIN_OPEN_RESULT_LENGTH if row["kind"] == "open"
                           else MIN_RESULT_LENGTH)
                if (total >= minimum
                        and (departure >= 0.2 or len(row["path"]) >= 2)):
                    accepted_heads.append(row)
            if accepted_heads:
                # Multiple branches from this one root all belong to the same
                # callout.  Dedupe coincident terminals, keeping the shorter path.
                by_tip = {}
                for row in accepted_heads:
                    token = (round(row["tip"][0] / 3.0),
                             round(row["tip"][1] / 3.0))
                    old = by_tip.get(token)
                    if old is None or len(row["path"]) < len(old["path"]):
                        by_tip[token] = row
                branches = list(by_tip.values())
                unique_edges = {edge_index for branch in branches
                                for edge_index in branch["path"]}
                headed_seeds.append((gap, -departure, len(unique_edges),
                                     -len(branches), seed_index, branches))
            elif (key in allow_bare and label.strip().lower() == "callout"
                  and departure >= BARE_NORMAL_MIN):
                for path, tip, path_length, terminal in local_bare:
                    # Bare terminals have no marker evidence, so a tiny rule
                    # protruding from a boxed note must not become a callout.
                    # Keep the headed threshold permissive; only the opt-in
                    # arrowless fallback needs this stronger minimum reach.
                    if path_length < MIN_BARE_RESULT_LENGTH:
                        continue
                    if _parallel_repeat(seed_index, edges):
                        continue
                    bare_paths.append((gap, -departure, len(path), path_length,
                                       path, tip, terminal))

        if headed_seeds:
            _gap, _dep, _edge_count, _n, _seed, branches = min(headed_seeds)
            leader_indices = []
            seen_leaders = set()
            arrow_strokes = []
            targets = []
            seen_arrow_parts = set()
            for branch in branches:
                for edge_index in branch["path"]:
                    if edge_index not in seen_leaders:
                        seen_leaders.add(edge_index)
                        leader_indices.append(edge_index)
                if branch["kind"] == "filled":
                    head = heads[branch["head"]]
                    for points in head["strokes"]:
                        token = tuple((round(x, 1), round(y, 1)) for x, y in points)
                        if token not in seen_arrow_parts:
                            seen_arrow_parts.add(token)
                            arrow_strokes.append(_to_frame(points, width, height))
                    target_box = head["bbox"]
                    incoming = branch["incoming"]
                    incoming_length = math.hypot(*incoming) or 1.0
                    ux, uy = (incoming[0] / incoming_length,
                              incoming[1] / incoming_length)
                    vertices = [point for stroke in head["strokes"]
                                for point in stroke]
                    # If the line attaches to the triangle base, the visual tip
                    # is the head vertex furthest forward along the incoming
                    # leader direction.  Tip-attached triangles naturally keep
                    # the graph terminal itself.
                    target_tip = max(
                        [branch["tip"]] + vertices,
                        key=lambda point: ((point[0] - branch["tip"][0]) * ux
                                           + (point[1] - branch["tip"][1]) * uy))
                else:
                    arm_boxes = []
                    for edge_index in branch["head"]:
                        edge = edges[edge_index]
                        if edge_index not in seen_arrow_parts:
                            seen_arrow_parts.add(edge_index)
                            arrow_strokes.append(_to_frame(
                                edge["points"], width, height))
                        xs = [p[0] for p in edge["points"]]
                        ys = [p[1] for p in edge["points"]]
                        arm_boxes.append((min(xs), min(ys), max(xs), max(ys)))
                    target_box = (
                        min(b[0] for b in arm_boxes), min(b[1] for b in arm_boxes),
                        max(b[2] for b in arm_boxes), max(b[3] for b in arm_boxes),
                    )
                    target_tip = branch["tip"]
                targets.append({
                    "tip": _to_frame([target_tip], width, height)[0],
                    "box_2d": _frame_box(target_box, width, height),
                    "terminal_kind": "arrowhead",
                })
            out[key] = {
                "leader_strokes": [_to_frame(edges[i]["points"], width, height)
                                   for i in leader_indices],
                "arrow_strokes": arrow_strokes,
                "targets": targets,
                "confidence": "high",
                "note": (f"geometric · text boundary · {len(targets)} "
                         "arrow target"),
            }
        elif bare_paths:
            _gap, _dep, _count, _length, path, tip, terminal = min(bare_paths)
            out[key] = {
                "leader_strokes": [_to_frame(edges[i]["points"], width, height)
                                   for i in path],
                "arrow_strokes": [],
                "targets": [{
                    "tip": _to_frame([tip], width, height)[0],
                    "box_2d": _frame_box(_tip_box(tip), width, height),
                    "terminal_kind": "bare-end",
                }],
                "confidence": "medium",
                "note": f"geometric · text boundary · bare {terminal}",
            }
    return out
