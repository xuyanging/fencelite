"""shape 符号匹配 —— 图例里的一个符号样例 → 整页所有同款放置.

输入是 VLM 给的图例符号框（0-1000 页面归一化），输出是这个符号在页面上
每一次放置的框。做法完全本地：拿 core/vecgeom.py 原子化出来的图元，做
平移 + 旋转 + 镜像不变的精确模板匹配（两锚点定位姿 → 全页逐图元验证），
文字内容参与判别，所以 circle-2 不会匹配到 circle-3。零模型调用、零费用。

只服务 shape（闭合外框 + 短码）。line（线型样例）走到 find_symbol_placements
会命中「框内只有线段」那条 error —— 线型的整线追踪本项目不做，样例框本身
由上层直接标出来。
"""
import math
from collections import Counter, defaultdict

from core.vecgeom import PRIM_TOL, _extract_page, _get_prims

# How much bigger than the legend template a candidate marker outline may be
# before it is rejected as "not the symbol". 1.35 covers the honest cases (a
# placement drawn slightly larger than the sample, or a template box the VLM
# cropped a hair tight) while still excluding plan features: on combined_bid
# P20 the circle is 18.1pt against a 18.1x17.3pt template (1.00x / 1.05x),
# and the next thing up that contains a marker is two orders of magnitude
# bigger.
OUTLINE_TOL = 1.35

# A marker code exported as vector outlines is made of tiny ``L``/``S``
# primitives rather than PDF text.  Those primitives are the identity of the
# marker (hexagon-5 is not hexagon-4), so they must survive template
# selection and matching.  48 comfortably covers the densest real code we
# have seen (koch P7's outlined ``C``: 12 outline + 17 glyph primitives)
# without turning a loose legend box into an unbounded page template.
MAX_TEMPLATE_PRIMS = 48


def _prim_bbox(p):
    """Return a primitive bbox, including straight segments."""
    box = p.get("bbox") or p.get("segment")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    x0, y0, x1, y1 = (float(v) for v in box)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _closed_contour_signature(items, bbox=None, *, return_edges=False):
    """Return an O(n), pose-invariant signature for one closed vector loop.

    ``S/poly`` primitives only expose a class and bbox, which is not enough
    to distinguish outlined letters such as D and E.  Rebuild the closed
    source runs lazily and select the run whose bbox produced the primitive.
    No signature is attached to the page-wide primitive cache: dense sheets
    can contain hundreds of thousands of primitives, so eager fingerprints
    are prohibitively expensive.

    Each vertex records the normalized lengths of its incoming/outgoing
    edges, the unsigned turn, and its normalized radius from the centroid.
    Matching tries every cyclic start and both traversal directions; the
    descriptor is therefore invariant to translation, uniform scale,
    rotation, mirror, path start, and path direction.
    """
    continuity_tol = 0.25  # coordinates are stored to 0.1 render pixel
    runs = []
    current = []

    def flush():
        nonlocal current
        if current:
            runs.append(current)
            current = []

    def append_edges(op, points):
        nonlocal current
        for left, right in zip(points, points[1:]):
            edge = (op, (float(left[0]), float(left[1])),
                    (float(right[0]), float(right[1])))
            if current:
                previous = current[-1][2]
                if (abs(previous[0] - edge[1][0]) > continuity_tol
                        or abs(previous[1] - edge[1][1]) > continuity_tol):
                    flush()
            current.append(edge)

    for item in items or ():
        op = item[0]
        if op == "l":
            append_edges(op, (item[1:3], item[3:5]))
        elif op == "c":
            # Sample a cubic at fixed parameters.  This preserves its actual
            # rendered contour instead of treating the control polygon as
            # straight ink.
            p0, p1, p2, p3 = (tuple(map(float, item[start:start + 2]))
                              for start in (1, 3, 5, 7))

            def point(t):
                q = 1.0 - t
                return (q ** 3 * p0[0] + 3 * q * q * t * p1[0]
                        + 3 * q * t * t * p2[0] + t ** 3 * p3[0],
                        q ** 3 * p0[1] + 3 * q * q * t * p1[1]
                        + 3 * q * t * t * p2[1] + t ** 3 * p3[1])

            append_edges(op, tuple(point(step / 4) for step in range(5)))
        elif op == "re":
            flush()
            x0, y0, x1, y1 = map(float, item[1:5])
            points = ((x0, y0), (x1, y0), (x1, y1), (x0, y1),
                      (x0, y0))
            append_edges(op, points)
            flush()
        elif op == "qu":
            flush()
            points = tuple(tuple(map(float, item[start:start + 2]))
                           for start in (1, 3, 5, 7))
            append_edges(op, (*points, points[0]))
            flush()
    flush()

    candidates = []
    for run in runs:
        if len(run) < 3 or len(run) > 96:
            continue
        first, last = run[0][1], run[-1][2]
        if (abs(first[0] - last[0]) > continuity_tol
                or abs(first[1] - last[1]) > continuity_tol):
            continue
        vertices = [edge[1] for edge in run]
        xs = [point[0] for point in vertices]
        ys = [point[1] for point in vertices]
        run_box = (min(xs), min(ys), max(xs), max(ys))
        if bbox is None:
            box_error = 0.0
        else:
            box_error = sum(abs(float(a) - float(b))
                            for a, b in zip(run_box, bbox))

        vectors = []
        lengths = []
        for _op, left, right in run:
            vx, vy = right[0] - left[0], right[1] - left[1]
            length = math.hypot(vx, vy)
            if length <= 1e-6:
                break
            vectors.append((vx, vy))
            lengths.append(length)
        if len(lengths) != len(run):
            continue
        perimeter = sum(lengths)
        if perimeter <= 1e-6:
            continue
        cx = sum(point[0] for point in vertices) / len(vertices)
        cy = sum(point[1] for point in vertices) / len(vertices)
        tokens = []
        for index, point in enumerate(vertices):
            previous = (index - 1) % len(run)
            incoming = (-vectors[previous][0], -vectors[previous][1])
            outgoing = vectors[index]
            denominator = (math.hypot(*incoming)
                           * math.hypot(*outgoing))
            cosine = ((incoming[0] * outgoing[0]
                       + incoming[1] * outgoing[1]) / denominator)
            turn = math.acos(max(-1.0, min(1.0, cosine))) / math.pi
            tokens.append((
                run[previous][0], run[index][0],
                lengths[previous] / perimeter,
                lengths[index] / perimeter,
                turn,
                math.hypot(point[0] - cx, point[1] - cy) / perimeter,
            ))
        candidates.append((
            box_error,
            tuple(tokens),
            tuple((edge[1], edge[2]) for edge in run),
        ))

    if not candidates:
        return None
    selected = min(candidates, key=lambda candidate: candidate[0])
    return selected[2] if return_edges else selected[1]


def _geom_signature_close(expected, actual):
    """Compare closed-contour signatures with cyclic/reverse invariance."""
    if expected is None:
        return True
    if actual is None or len(expected) != len(actual):
        return False
    count = len(expected)
    if not count:
        return False

    variants = [actual]
    variants.append(tuple((token[1], token[0], token[3], token[2],
                           token[4], token[5])
                          for token in reversed(actual)))
    for variant in variants:
        for shift in range(count):
            length_diffs = []
            turn_diffs = []
            radial_diffs = []
            valid = True
            for index, expected_token in enumerate(expected):
                token = variant[(index + shift) % count]
                if token[:2] != expected_token[:2]:
                    valid = False
                    break
                length_diffs.extend((abs(token[2] - expected_token[2]),
                                     abs(token[3] - expected_token[3])))
                turn_diffs.append(abs(token[4] - expected_token[4]))
                radial_diffs.append(abs(token[5] - expected_token[5]))
            if not valid:
                continue

            def close(diffs, mean_tol, max_tol):
                return (max(diffs, default=0.0) <= max_tol
                        and sum(diffs) / max(len(diffs), 1) <= mean_tol)

            if (close(length_diffs, 0.012, 0.04)
                    and close(turn_diffs, 0.045, 0.14)
                    and close(radial_diffs, 0.010, 0.03)):
                return True
    return False


def _compact_line_identity(prims, by_class, indices, classes, pad=0.6):
    """Class multiset of identity-style strokes inside a glyph's tight box.

    ``indices`` are the strokes already matched positively. Extra strokes of
    the same drawing style are included only when their complete geometry is
    inside that compact glyph box. This catches F-as-a-subset-of-E and
    C-as-a-subset-of-G without treating unrelated lines elsewhere in the
    surrounding marker as part of its code.
    """
    boxes = [_prim_bbox(prims[index]) for index in indices
             if prims[index]["c"][0] == "L"]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    gx0 = min(box[0] for box in boxes) - pad
    gy0 = min(box[1] for box in boxes) - pad
    gx1 = max(box[2] for box in boxes) + pad
    gy1 = max(box[3] for box in boxes) + pad
    signature = Counter()
    for cls in classes:
        for index in by_class.get(cls, ()):
            box = _prim_bbox(prims[index])
            if (box is not None and gx0 <= box[0] and gy0 <= box[1]
                    and box[2] <= gx1 and box[3] <= gy1):
                signature[cls] += 1
    return signature


def _connected_line_identity(prims, by_class, seed_indices, seed_edges,
                             classes, outline, *, units=None,
                             grid=None,
                             outline_source=None, draw_direction=0,
                             source_span=8, tolerance=0.25):
    """Expand glyph ink through conservatively attached line primitives.

    Positive matching alone permits subset errors: F can land on E, P on R,
    and a closed O contour on Q when the extra tail is a separate PDF path.
    Extra strokes are admitted only when all available evidence says they
    belong to the same marker block: their primitive *and complete source
    unit* fit inside the marker, their own endpoint touches existing glyph
    ink, and their draw order follows the template's outline/identity order.
    Missing metadata fails open instead of rejecting a real placement.
    """
    if outline is None:
        return None
    ox0, oy0, ox1, oy1 = outline
    inset = max(0.35, min(ox1 - ox0, oy1 - oy0) * 0.05)
    inner = (ox0 + inset, oy0 + inset, ox1 - inset, oy1 - inset)
    seed_sources = {
        prims[index].get("src") for index in seed_indices
        if isinstance(prims[index].get("src"), int)
    }

    def point_segment_distance(point, edge):
        px, py = point
        (ax, ay), (bx, by) = edge
        vx, vy = bx - ax, by - ay
        denominator = vx * vx + vy * vy
        if denominator <= 1e-9:
            return math.hypot(px - ax, py - ay)
        ratio = max(0.0, min(1.0,
                            ((px - ax) * vx + (py - ay) * vy)
                            / denominator))
        return math.hypot(px - (ax + ratio * vx),
                          py - (ay + ratio * vy))

    def attaches(extra, ink):
        # Only an endpoint of the *new* stroke may attach.  A background line
        # crossing the middle of glyph ink is not part of the glyph.
        return min(point_segment_distance(extra[0], ink),
                   point_segment_distance(extra[1], ink)) <= tolerance

    connected = {index for index in seed_indices
                 if prims[index]["c"][0] == "L"}
    ink_edges = list(seed_edges)
    for index in connected:
        raw = prims[index].get("segment")
        if raw is not None:
            ink_edges.append(((float(raw[0]), float(raw[1])),
                              (float(raw[2]), float(raw[3]))))

    if grid is None:
        pool = {index for cls in classes for index in by_class.get(cls, ())}
    else:
        pool = set()
        gx0, gy0 = int(inner[0] // PRIM_TOL), int(inner[1] // PRIM_TOL)
        gx1, gy1 = int(inner[2] // PRIM_TOL), int(inner[3] // PRIM_TOL)
        for gx in range(gx0 - 1, gx1 + 2):
            for gy in range(gy0 - 1, gy1 + 2):
                pool.update(grid.get((gx, gy), ()))

    candidates = []
    for index in pool:
        if prims[index]["c"] not in classes:
            continue
        if index in connected:
            continue
        p = prims[index]
        box = _prim_bbox(p)
        raw = p.get("segment")
        src = p.get("src")
        if (box is None or raw is None
                or not (inner[0] <= box[0] and inner[1] <= box[1]
                        and box[2] <= inner[2] and box[3] <= inner[3])):
            continue

        # A separately exported tail must be a compact unit.  If unit
        # metadata is unavailable, only a same-source stroke is safe.
        same_source = src in seed_sources
        if not same_source:
            if (units is None or not isinstance(src, int)
                    or not (0 <= src < len(units))
                    or len(units[src]) < 5):
                continue
            ux0, uy0, ux1, uy1 = map(float, units[src][1:5])
            slack = 0.25
            if not (inner[0] - slack <= min(ux0, ux1)
                    and inner[1] - slack <= min(uy0, uy1)
                    and max(ux0, ux1) <= inner[2] + slack
                    and max(uy0, uy1) <= inner[3] + slack):
                continue
            if not isinstance(outline_source, int) or not draw_direction:
                continue
            offset = src - outline_source
            if (offset * draw_direction <= 0
                    or abs(offset) > min(max(source_span, 1), 8)):
                continue

        candidates.append((
            index,
            ((float(raw[0]), float(raw[1])),
             (float(raw[2]), float(raw[3]))),
        ))

    changed = True
    while changed:
        changed = False
        remaining = []
        for index, edge in candidates:
            if any(attaches(edge, ink) for ink in ink_edges):
                connected.add(index)
                ink_edges.append(edge)
                changed = True
            else:
                remaining.append((index, edge))
        candidates = remaining

    return Counter(prims[index]["c"] for index in connected)


def _select_symbol_template(prims, inside, *, units=None, content_bounds=None):
    """Choose outline + marker identity from primitives inside a sample.

    Native PDF text is already an exact class (``("T", text)``), so its
    content is mandatory.  CAD-exported letters/numbers have no text layer;
    they appear as compact line/closed-shape geometry *inside* the enclosing
    marker.  Keep those inner primitives as mandatory identity geometry.

    Long leaders and table/background rules are deliberately excluded: their
    endpoints/bbox do not fit inside the inset of the largest closed marker
    outline.  Optional outline pieces retain the old small miss allowance;
    identity primitives do not.

    Returns ``(allow, required, has_marker_content)`` where both index lists
    refer to ``prims``.
    """
    raw_texts = [i for i in inside if prims[i]["c"][0] == "T"]
    shapes = [i for i in inside if prims[i]["c"][0] == "S"]
    segments = [i for i in inside if prims[i]["c"][0] == "L"]

    def content_text(index):
        if content_bounds is None:
            return True
        box = _prim_bbox(prims[index])
        if box is None:
            return False
        bx0, by0, bx1, by1 = content_bounds
        # VLM/snap boxes can be a render pixel tight.  The centre must still
        # be in the unpadded content box and at least 80% of the glyph bbox
        # must be covered on each non-degenerate axis.
        cx, cy = prims[index]["x"], prims[index]["y"]
        if not (bx0 <= cx <= bx1 and by0 <= cy <= by1):
            return False

        def covered(lo, hi, outer_lo, outer_hi):
            extent = hi - lo
            if extent <= 0.1:
                return outer_lo - 1.5 <= lo <= outer_hi + 1.5
            overlap = max(0.0, min(hi, outer_hi + 1.5)
                          - max(lo, outer_lo - 1.5))
            return overlap / extent >= 0.8

        return (covered(box[0], box[2], bx0, bx1)
                and covered(box[1], box[3], by0, by1))

    shape_boxes = [(i, _prim_bbox(prims[i])) for i in shapes]
    shape_boxes = [(i, b) for i, b in shape_boxes if b is not None]
    shapes = [i for i, _b in shape_boxes]
    texts = ([i for i in raw_texts if content_text(i)]
             if not shapes else list(raw_texts))
    has_marker_content = bool(shapes or texts)

    # A content-only derived row code has text but no closed marker in this
    # selection.  Its exact text class is sufficient and must remain the only
    # template primitive (steps.placements validates that contract).
    if not shapes:
        body = segments if not texts else []
        allow = body + texts
        return allow[:MAX_TEMPLATE_PRIMS], set(texts), has_marker_content

    outline = None
    outline_index = None
    if shape_boxes:
        outline_index, outline = max(
            shape_boxes,
            key=lambda ib: ((ib[1][2] - ib[1][0])
                            * (ib[1][3] - ib[1][1]), prims[ib[0]]["s"]),
        )

    identity = []
    suppressed_shapes = set()
    marker_texts = list(texts)
    if outline is not None:
        ox0, oy0, ox1, oy1 = outline
        ow, oh = ox1 - ox0, oy1 - oy0
        # The inset is large enough to reject ring/fill fragments that ride
        # on the boundary, but small enough to retain centred outlined glyphs.
        inset = max(0.5, min(ow, oh) * 0.08)
        ix0, iy0, ix1, iy1 = (ox0 + inset, oy0 + inset,
                              ox1 - inset, oy1 - inset)

        def wholly_inner(index):
            box = _prim_bbox(prims[index])
            if box is None or ix1 <= ix0 or iy1 <= iy0:
                return False
            if not (ix0 <= box[0] and iy0 <= box[1]
                    and box[2] <= ix1 and box[3] <= iy1):
                return False
            # A short primitive can be one piece of a long underlying path.
            # Requiring the complete PDF drawing unit to fit inside the
            # marker rejects those clipped table/background/leader pieces.
            src = prims[index].get("src")
            if (units is not None and isinstance(src, int)
                    and 0 <= src < len(units)):
                unit = units[src]
                if len(unit) >= 5:
                    ux0, uy0, ux1, uy1 = map(float, unit[1:5])
                    # Some CAD exporters put outline and glyph in one path,
                    # so the source bbox may equal the marker outline.  It
                    # still must not extend outside that outline like a
                    # leader/background path does.
                    source_is_outline = (outline_index is not None
                                         and src == prims[outline_index].get("src"))
                    slack = 1.0
                    if (not source_is_outline
                            and not (ox0 - slack <= min(ux0, ux1)
                                     and oy0 - slack <= min(uy0, uy1)
                                     and max(ux0, ux1) <= ox1 + slack
                                     and max(uy0, uy1) <= oy1 + slack)):
                        return False
            return True

        identity.extend(i for i in shapes if wholly_inner(i))
        identity.extend(i for i in segments if wholly_inner(i))

        # A neighbouring legend description can fall in a padded VLM box;
        # only text whose centre is inside the marker outline is its code.
        marker_texts = [
            i for i in raw_texts
            if ox0 <= prims[i]["x"] <= ox1
            and oy0 <= prims[i]["y"] <= oy1
        ]

        # When a native text code exists, its exact ``("T", text)`` class is
        # the identity.  Do not also require glyph-like vector fragments from
        # fill/scan geometry around it; that would make exact text matches
        # fail for otherwise equivalent placements.
        if marker_texts:
            suppressed_shapes.update(
                index for index in identity
                if prims[index]["c"][0] == "S")
            identity = []

    required = set(identity) | set(marker_texts)

    # Preserve every identity primitive.  If a very fragmented outline would
    # exceed the cap, trim optional outline pieces by size only.
    optional_shapes = [i for i in shapes
                       if i not in required and i not in suppressed_shapes]
    template_cap = MAX_TEMPLATE_PRIMS if identity else 24
    room = max(0, template_cap - len(required))
    optional_shapes = sorted(
        optional_shapes, key=lambda i: -prims[i]["s"]
    )[:room]
    allow = optional_shapes + sorted(required)
    return allow, required, has_marker_content


def match_template(prims, by_class, grid, sel, seltext, prim_allow=None,
                   prim_required=None, scale=1.0,
                   contour_signature_of=None):
    """The /match algorithm from pdf_viz.py, returning per-placement groups.

    sel: set of vector path ids (unit idx) inside the template box;
    seltext: set of text ids inside the box.
    prim_allow: optional set of prim indices — when given, the template is
      exactly these prims (finer than unit-level selection; lets the caller
      drop e.g. leader stubs whose length varies per placement).
    prim_required: optional set of vector prim indices whose identity cannot
      be consumed by the ordinary outline miss allowance. Used for outlined
      letters/digits made from tiny line/shape primitives.
    contour_signature_of: optional lazy callback ``prim index -> signature``
      for required closed vector glyphs. It is intentionally scoped to one
      symbol search instead of being stored on every primitive of a page.
    Symbol mode never does period detection/reduction — compact symbols are
      not periodic, and symmetric multi-piece glyphs (a circle drawn as a
      ring of tiny filled polys) vote out spurious periods that wreck the
      template. Non-periodic max_miss then scales with template size
      (0 for <=4 prims, ~1/8 of the prims otherwise; texts must always hit).
    scale: expected instance size relative to the template (legends are
      often drawn smaller than the plan callouts). The pose solve is a
      similarity transform already; scale only relaxes the size gates
      (candidate lclose, |ab| vs |AB|, near()'s size check) to that ratio.
    Returns dict with either "error" or:
      groups: [set(prim idx)] — one set per matched placement;
      matched: set(prim idx) — union of the groups;
      period (always False, kept for the caller's payload), count,
      rescued (always 0, ditto), template_prims, single (bool).
    """
    def in_tmpl(p):
        return (p.get("src") in sel) if p.get("src") is not None else (p.get("tid") in seltext)

    if prim_allow is not None:
        tmpl0 = sorted(prim_allow)
    else:
        tmpl0 = [idx for idx, p in enumerate(prims) if in_tmpl(p)]
    if not tmpl0:
        return {"error": "no recognizable primitive inside the box"}
    if len(tmpl0) > 1500:
        return {"error": "selection box too large"}

    # Tolerances (adaptive to template size further below).
    VTOL = 4.0
    LEN_ABS, LEN_REL = 4.0, 0.18

    def lclose(s1, s2, rel=LEN_REL):
        return abs(s1 - s2) <= max(LEN_ABS, rel * max(s1, s2))

    _cx = [prims[i]["x"] for i in tmpl0]
    _cy = [prims[i]["y"] for i in tmpl0]
    tscale = max(max(_cx) - min(_cx), max(_cy) - min(_cy),
                 max(prims[i]["s"] for i in tmpl0), 1.0)
    VTOL = max(1.0, min(4.0, 0.25 * tscale))
    LEN_ABS = max(1.0, min(4.0, 0.25 * tscale))
    NC = int(VTOL // PRIM_TOL) + 1

    required0 = set(prim_required or ())

    def signature_of(index):
        if contour_signature_of is not None:
            return contour_signature_of(index)
        return prims[index].get("g")

    def near_candidates(x, y, cls, sc, *, expected_o=None, expected_g=None,
                        strict=False):
        pos_tol = max(0.5, min(1.5, 0.08 * tscale)) if strict else VTOL

        def size_close(actual):
            # A normalized contour is rotation invariant, while ``S.s`` is
            # derived from its axis-aligned bbox and is not. The surrounding
            # marker pose already fixes scale, so do not reject an otherwise
            # identical rotated closed glyph on AABB size alone.
            if strict and expected_g is not None:
                return True
            if strict:
                return abs(actual - sc) <= max(0.35, 0.20 * max(actual, sc))
            return lclose(actual, sc)

        def angle_close(actual):
            if expected_o is None:
                return True
            if actual is None:
                return False
            delta = abs(float(actual) - float(expected_o)) % math.pi
            delta = min(delta, math.pi - delta)
            # Very short strokes suffer more from 0.1px endpoint rounding.
            limit = 0.40 if sc < 0.9 else 0.22
            return delta <= limit

        cx, cy = int(x // PRIM_TOL), int(y // PRIM_TOL)
        found = []
        seen = set()
        for dx in range(-NC, NC + 1):
            for dy in range(-NC, NC + 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    if j in seen:
                        continue
                    seen.add(j)
                    p = prims[j]
                    dist = abs(p["x"] - x) + abs(p["y"] - y)
                    if (p["c"] != cls or dist > pos_tol
                            or not size_close(p["s"])
                            or not angle_close(p.get("o"))
                            or (strict and not _geom_signature_close(
                                expected_g, signature_of(j)))):
                        continue
                    found.append(((dist, abs(p["s"] - sc), j), j))
        found.sort()
        return [j for _score, j in found]

    # 1) template reduction. The periodic branch (dominant translation vector
    # → motif) only ever ran for line-style samples; a symbol template is the
    # selection itself. Repeated equal text tokens remain separate identity
    # primitives: a marker containing two independently exported "1" words
    # must not collapse to a single "1".
    vec_idx = [t for t in tmpl0 if prims[t].get("src") is not None]
    txt_keep = sorted(
        (t for t in tmpl0 if prims[t].get("tid") is not None),
        key=lambda t: (prims[t]["y"], prims[t]["x"], t),
    )
    tmpl = vec_idx + txt_keep

    def cand_of(t):
        cls = prims[t]["c"]
        sc = prims[t]["s"] * scale
        contour = signature_of(t) if t in required0 else None
        if contour is not None:
            return [k for k in by_class[cls]
                    if abs(prims[k]["s"] - sc)
                    <= max(2.0, 0.40 * max(prims[k]["s"], sc))]
        return [k for k in by_class[cls]
                if lclose(prims[k]["s"], sc)]

    m = len(tmpl)
    if m == 0:
        return {"error": "no repeating unit found inside the box"}
    if m > MAX_TEMPLATE_PRIMS:
        return {"error": f"template holds {m} primitives, too many (the selection may not be tight enough)"}

    txt_anchors = [t for t in tmpl if prims[t].get("tid") is not None]
    txt_rep = [t for t in txt_anchors if len(cand_of(t)) >= 2]
    A = min(txt_rep or txt_anchors or tmpl, key=lambda t: len(cand_of(t)))
    ax0, ay0 = prims[A]["x"], prims[A]["y"]
    cand = [t for t in tmpl if t != A and (abs(prims[t]["x"] - ax0) + abs(prims[t]["y"] - ay0)) > 2]
    B = min(cand, key=lambda t: (len(cand_of(t)),
                                 -((prims[t]["x"] - ax0) ** 2 + (prims[t]["y"] - ay0) ** 2))) if cand else None

    toff = [(t, prims[t]["c"], prims[t]["s"],
             prims[t]["x"] - ax0, prims[t]["y"] - ay0,
             prims[t].get("tid") is not None, t in required0,
             prims[t].get("o"),
             signature_of(t) if t in required0 else None)
            for t in tmpl]

    # Symbol mode: multi-piece outer glyphs tolerate a proportional miss
    # (identical CAD blocks can still differ by an overlapped fill piece).
    # Text and inner vector-code identity never consume this allowance.
    optional_count = sum(1 for t in tmpl
                         if t not in required0
                         and prims[t].get("tid") is None)
    max_miss = min(optional_count,
                   0 if m <= 4 else max(1, m // 8))
    need = m - max_miss

    def verify_pose(ax, ay, c, s, mir, locked=None):
        """Verify a pose with a true one-to-one bipartite assignment."""
        options = {}
        mandatory = set()
        for (template_index, ck, sk, ox, oy, is_txt, is_required,
             template_o, template_g) in toff:
            if mir == 1:
                tx = ax + c * ox - s * oy
                ty = ay + s * ox + c * oy
            else:
                tx = ax + c * ox + s * oy
                ty = ay + s * ox - c * oy
            expected_o = None
            if template_o is not None:
                phi = math.atan2(s, c)
                expected_o = ((phi + template_o) if mir == 1
                              else (phi - template_o)) % math.pi
            options[template_index] = near_candidates(
                tx, ty, ck, sk * scale,
                expected_o=expected_o, expected_g=template_g,
                strict=is_required and not is_txt)
            if is_txt or is_required:
                mandatory.add(template_index)

        locked = dict(locked or {})
        if len(set(locked.values())) != len(locked):
            return None
        candidate_owner = {}
        assignment = {}
        for template_index, candidate_index in locked.items():
            if candidate_index not in options.get(template_index, ()):
                return None
            assignment[template_index] = candidate_index
            candidate_owner[candidate_index] = template_index

        locked_templates = set(locked)

        def augment(template_index, visiting):
            if template_index in visiting:
                return False
            visiting.add(template_index)
            for candidate_index in options[template_index]:
                owner = candidate_owner.get(candidate_index)
                if owner is None:
                    candidate_owner[candidate_index] = template_index
                    assignment[template_index] = candidate_index
                    return True
                if owner in locked_templates:
                    continue
                if augment(owner, visiting):
                    candidate_owner[candidate_index] = template_index
                    assignment[template_index] = candidate_index
                    return True
            return False

        # Rarest mandatory rows first; augmenting paths repair any greedy
        # choice when two similar strokes compete for the same candidate.
        for template_index in sorted(
                mandatory - locked_templates,
                key=lambda index: (len(options[index]), index)):
            if not options[template_index] or not augment(template_index, set()):
                return None
        optional = set(options) - mandatory - locked_templates
        for template_index in sorted(
                optional, key=lambda index: (len(options[index]), index)):
            if options[template_index]:
                augment(template_index, set())
        if len(assignment) < need:
            return None
        return (frozenset(assignment.values()),
                frozenset(assignment[index] for index in required0
                          if index in assignment))

    def finish(matches):
        matches.sort(key=lambda match: -len(match[0]))
        final = []
        final_required = []
        used = set()
        for group, required_hits in matches:
            if group & used:
                continue
            final.append(set(group))
            final_required.append(set(required_hits))
            used |= group
        return {"groups": final, "matched": set(used), "period": False,
                "count": len(final), "rescued": 0,
                "template_prims": m, "single": False,
                "required_hits": final_required}

    if m <= 1:
        # A required closed contour still carries identity even when it is
        # the only primitive. Do not downgrade it to a same-class search.
        expected = signature_of(A) if A in required0 else None
        hits = [candidate for candidate in cand_of(A)
                if _geom_signature_close(expected,
                                         signature_of(candidate))]
        return {"groups": [{h} for h in hits], "matched": set(hits),
                "period": False, "count": len(hits), "rescued": 0,
                "template_prims": m, "single": True,
                "required_hits": [{h} if A in required0 else set()
                                  for h in hits]}

    if B is None:
        # Several primitives can legitimately share a centre (circle + native
        # text, or circle + one closed outlined glyph).  The old fallback
        # silently treated that as a single-primitive template and never
        # verified the identity primitive.  With no stable baseline vector,
        # verify the whole template under translation + requested scale.
        matches = []
        oriented = next((row for row in toff if row[7] is not None), None)
        for a in cand_of(A):
            hypotheses = {(1, 0.0, None), (-1, 0.0, None)}
            if oriented is not None:
                oriented_index, cls = oriented[0], oriented[1]
                template_o = oriented[7]
                ax, ay = prims[a]["x"], prims[a]["y"]
                radius = VTOL + 3.0
                span = int(math.ceil(radius / PRIM_TOL)) + 1
                gcx, gcy = int(ax // PRIM_TOL), int(ay // PRIM_TOL)
                nearby = set()
                for dx in range(-span, span + 1):
                    for dy in range(-span, span + 1):
                        nearby.update(grid.get((gcx + dx, gcy + dy), ()))
                for candidate in nearby:
                    p = prims[candidate]
                    if (p["c"] != cls or p.get("o") is None
                            or abs(p["x"] - ax) + abs(p["y"] - ay) > radius):
                        continue
                    for mir in (1, -1):
                        phi = ((p["o"] - template_o) if mir == 1
                               else (p["o"] + template_o))
                        for angle in (phi, phi + math.pi):
                            hypotheses.add((mir, angle, candidate))
            for mir, angle, oriented_candidate in hypotheses:
                locks = {A: a}
                if (oriented_candidate is not None
                        and oriented[0] != A):
                    locks[oriented[0]] = oriented_candidate
                group = verify_pose(
                    prims[a]["x"], prims[a]["y"],
                    scale * math.cos(angle), scale * math.sin(angle), mir,
                    locked=locks)
                if group is not None:
                    matches.append(group)
        return finish(matches)

    Ax, Ay = ax0, ay0
    ABx, ABy = prims[B]["x"] - Ax, prims[B]["y"] - Ay
    ABlen = math.hypot(ABx, ABy)
    den = ABx * ABx + ABy * ABy
    candA = cand_of(A)
    candB = cand_of(B)
    if len(candA) > 4000:
        return {"error": "the anchor primitive is too common; the template is not distinctive enough"}

    cellB = max(ABlen * scale, 1.0)
    bgrid = defaultdict(list)
    for b in candB:
        bgrid[(int(prims[b]["x"] // cellB), int(prims[b]["y"] // cellB))].append(b)

    matches = []
    for a in candA:
        ax, ay = prims[a]["x"], prims[a]["y"]
        bcx, bcy = int(ax // cellB), int(ay // cellB)
        for dgx in (-2, -1, 0, 1, 2):
            for dgy in (-2, -1, 0, 1, 2):
                for b in bgrid.get((bcx + dgx, bcy + dgy), []):
                    if b == a:
                        continue
                    bx, by = prims[b]["x"], prims[b]["y"]
                    if abs(math.hypot(bx - ax, by - ay) - ABlen * scale) > VTOL:
                        continue
                    for mir in (1, -1):           # forward / mirrored; R*AB = ab
                        if mir == 1:
                            c = (ABx * (bx - ax) + ABy * (by - ay)) / den
                            s = (ABx * (by - ay) - ABy * (bx - ax)) / den
                        else:
                            c = (ABx * (bx - ax) - ABy * (by - ay)) / den
                            s = (ABx * (by - ay) + ABy * (bx - ax)) / den
                        group = verify_pose(ax, ay, c, s, mir,
                                            locked={A: a, B: b})
                        if group is not None:
                            matches.append(group)

    # The draw-order continuity rescue of the original algorithm was gated on
    # a line period; a compact symbol is exactly what the pose solve verified.
    return finish(matches)


def find_symbol_placements(pdf_path, page_index, box_norm, *,
                           content_box_norm=None):
    """Top-level entry: VLM legend-symbol box (0-1000 page-normalized) →
    every placement of that symbol on the page via vector matching.

    ``content_box_norm`` optionally narrows which primitives form the
    template while ``box_norm`` remains the marker-size reference and legend
    exclusion frame.  Derived schedule row codes use this split: exact 4.6
    glyph content for identity, inherited 4.0 outline dimensions for deciding
    whether a plan hit really sits inside the expected marker.

    Returns {placements: [[ymin,xmin,ymax,xmax] normalized, ...], count,
    rescued, template_prims, template_texts, period, single} or {error}.
    The group overlapping the input box (the legend original) is excluded
    from `placements`.
    """
    data = _extract_page(pdf_path, page_index)
    if not data["units"] and not data["texts"]:
        return {"error": "this sheet has no vector graphics (likely a scan); vector matching is unavailable"}
    w, h = data["w"], data["h"]

    by0, bx0, by1, bx1 = [float(v) for v in box_norm]
    px0, py0 = bx0 / 1000 * w, by0 / 1000 * h
    px1, py1 = bx1 / 1000 * w, by1 / 1000 * h
    if content_box_norm is None:
        cpx0, cpy0, cpx1, cpy1 = px0, py0, px1, py1
    else:
        try:
            cby0, cbx0, cby1, cbx1 = [float(v)
                                      for v in content_box_norm]
        except (TypeError, ValueError):
            return {"error": "invalid content_box_norm"}
        if cbx1 <= cbx0 or cby1 <= cby0:
            return {"error": "invalid content_box_norm"}
        cpx0, cpy0 = cbx0 / 1000 * w, cby0 / 1000 * h
        cpx1, cpy1 = cbx1 / 1000 * w, cby1 / 1000 * h

    prims, by_class, grid = _get_prims(pdf_path, page_index)

    contour_cache = {}
    contour_edge_cache = {}

    def contour_key_and_items(index):
        p = prims[index]
        if p["c"][0] != "S" or p.get("src") is None:
            return None, None, None
        box = _prim_bbox(p)
        key = (p["src"], tuple(round(value, 2) for value in box)
               if box is not None else None)
        src = p["src"]
        items = data["geom"][src] if 0 <= src < len(data["geom"]) else ()
        return key, items, box

    def contour_signature_of(index):
        key, items, box = contour_key_and_items(index)
        if key is None:
            return None
        if key not in contour_cache:
            contour_cache[key] = _closed_contour_signature(items, box)
        return contour_cache[key]

    def contour_edges_of(index):
        key, items, box = contour_key_and_items(index)
        if key is None:
            return ()
        if key not in contour_edge_cache:
            contour_edge_cache[key] = (
                _closed_contour_signature(items, box, return_edges=True)
                or ())
        return contour_edge_cache[key]

    # Template = prims whose center falls in the (padded) box. Native text is
    # exact identity. Outlined text has no PDF text primitive, so compact
    # vector geometry wholly inside the marker outline is mandatory identity;
    # external leaders/table rules are excluded by that containment gate.
    allow = []
    required = set()
    has_marker_content = False
    selections = []
    for pad in (2.0, 8.0):
        ax0, ay0 = cpx0 - pad, cpy0 - pad
        ax1, ay1 = cpx1 + pad, cpy1 + pad
        inside = [i for i, p in enumerate(prims)
                  if ax0 <= p["x"] <= ax1 and ay0 <= p["y"] <= ay1]
        if inside:
            candidate_allow, candidate_required, candidate_marker = (
                _select_symbol_template(
                prims, inside, units=data["units"],
                content_bounds=(cpx0, cpy0, cpx1, cpy1)))
            if not candidate_allow:
                continue
            text_identity = any(prims[i]["c"][0] == "T"
                                for i in candidate_required)
            shape_items = [
                (i, _prim_bbox(prims[i])) for i in candidate_allow
                if prims[i]["c"][0] == "S"
            ]
            shape_items = [(i, box) for i, box in shape_items
                           if box is not None]
            outline_item = max(
                shape_items,
                key=lambda item: ((item[1][2] - item[1][0])
                                  * (item[1][3] - item[1][1])),
                default=None)
            outline = outline_item[1] if outline_item is not None else None

            required_boxes = [_prim_bbox(prims[i])
                              for i in candidate_required]
            required_boxes = [box for box in required_boxes
                              if box is not None]
            credible_outline = False
            if required_boxes:
                rx0 = min(box[0] for box in required_boxes)
                ry0 = min(box[1] for box in required_boxes)
                rx1 = max(box[2] for box in required_boxes)
                ry1 = max(box[3] for box in required_boxes)
                required_width = max(rx1 - rx0, 0.75)
                required_height = max(ry1 - ry0, 0.75)
                required_area = max((rx1 - rx0) * (ry1 - ry0), 0.1)
                for index, box in shape_items:
                    if index in candidate_required:
                        continue
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    area = width * height
                    if (box[0] <= rx0 - 0.25 and box[1] <= ry0 - 0.25
                            and box[2] >= rx1 + 0.25
                            and box[3] >= ry1 + 0.25
                            and width >= required_width * 1.6
                            and height >= required_height * 1.6
                            and area >= required_area * 2.5):
                        credible_outline = True
                        break

            norm = max((px1 - px0) + (py1 - py0), 1.0)
            edge_error = (sum(abs(left - right)
                              for left, right in zip(
                                  outline, (px0, py0, px1, py1))) / norm
                          if outline is not None else float("inf"))
            if content_box_norm is not None and text_identity:
                # Derived schedule row codes intentionally use the exact
                # content box even when a wider pad happens to see a nearby
                # inherited marker outline.
                score = (-1, pad, 0.0)
            elif credible_outline:
                score = (0, edge_error, pad)
            elif candidate_required:
                score = (1, pad, edge_error)
            elif shape_items:
                # A lone closed glyph in the tight selection is not yet a
                # complete marker if a wider selection finds an enclosing
                # outline plus mandatory identity.
                score = (2, edge_error, pad)
            else:
                score = (3, pad, 0.0)
            selections.append((score, candidate_allow,
                               candidate_required, candidate_marker))
    if selections:
        _score, allow, required, has_marker_content = min(
            selections, key=lambda selection: selection[0])
    if not allow:
        return {"error": "no vector primitive inside the symbol box (the box may be offset, or the symbol is a raster)"}
    if not has_marker_content:
        # Segments-only content is a LINE-STYLE SAMPLE, not a marker: its
        # "placements" are the fence line itself (fenceline flow), and
        # stamping lookalike dash/X patches page-wide as symbol placements
        # litters the plan (rapid P7: manholes and tree marks boxed).
        return {"error": "the box holds only line segments (a line-type sample) - symbol matching does not apply; lines are located by the fenceline flow"}

    sel = {prims[i]["src"] for i in allow if prims[i].get("src") is not None}
    seltext = {prims[i]["tid"] for i in allow if prims[i].get("tid") is not None}
    units = data["units"]
    texts = data["texts"]

    def _enclosing_outline(gx0, gy0, gx1, gy1):
        """The marker outline the matched group sits inside, or None.

        Why this is needed: the group box is the union of the prims that
        actually matched, and for a coded marker (a number inside a bubble)
        the matcher can land the digits without the enclosing circle — on
        combined_bid P20 every placement came back exactly one digit-height
        tall (12.1pt) against an 18.1pt circle, i.e. 67% of the marker, so
        the published box cut the bubble off.

        The gate is anchored on the TEMPLATE, which is the legend sample and
        therefore the whole marker: a candidate must fully contain the group
        and must not exceed the template's own dimensions by much. That
        rejects the two things that would otherwise swallow a marker sitting
        on a plan feature — the long fence line whose bbox brushes past it,
        and the building outline that encloses half the sheet — without any
        page-specific tuning. Containment (not merely centre-in) also rejects
        the ~31 thin scanline strips that fill these bubbles: a 18x0.2 strip
        cannot contain a 7.8x12.1 digit box.
        """
        tw, th = px1 - px0, py1 - py0
        if tw <= 0 or th <= 0:
            return None
        gw, gh = gx1 - gx0, gy1 - gy0
        slack = 1.0                    # 1 render px of tolerance on containment
        best = None
        for u in units:
            ux0, uy0, ux1, uy1 = u[1], u[2], u[3], u[4]
            if not (ux0 <= gx0 + slack and uy0 <= gy0 + slack
                    and ux1 >= gx1 - slack and uy1 >= gy1 - slack):
                continue
            uw, uh = ux1 - ux0, uy1 - uy0
            if uw <= gw + slack and uh <= gh + slack:
                continue               # adds nothing
            if uw > tw * OUTLINE_TOL or uh > th * OUTLINE_TOL:
                continue               # bigger than the symbol itself
            area = uw * uh
            if best is None or area < best[0]:
                best = (area, ux0, uy0, ux1, uy1)
        return best[1:] if best else None

    def _placements_of(res):
        """Per-placement bbox: union of the group's underlying unit/text
        boxes, widened to the enclosing marker outline when the match landed
        only the inner glyphs (see _enclosing_outline); the legend original
        (overlap-over-smaller vs the input box) is counted separately."""
        placements = []
        template_hit = 0
        for grp in res["groups"]:
            gx0 = gy0 = float("inf")
            gx1 = gy1 = float("-inf")
            for pi in grp:
                p = prims[pi]
                if p.get("src") is not None:
                    u = units[p["src"]]
                    gx0 = min(gx0, u[1]); gy0 = min(gy0, u[2])
                    gx1 = max(gx1, u[3]); gy1 = max(gy1, u[4])
                elif p.get("tid") is not None:
                    t = texts[p["tid"]]
                    gx0 = min(gx0, t["x0"]); gy0 = min(gy0, t["y0"])
                    gx1 = max(gx1, t["x1"]); gy1 = max(gy1, t["y1"])
            if gx0 > gx1:
                continue
            # Widen to the marker outline BEFORE the legend-original test, so
            # the template overlap is measured on the box we actually publish.
            grown = _enclosing_outline(gx0, gy0, gx1, gy1)
            if grown is not None:
                gx0 = min(gx0, grown[0]); gy0 = min(gy0, grown[1])
                gx1 = max(gx1, grown[2]); gy1 = max(gy1, grown[3])
            iw = max(0.0, min(gx1, px1) - max(gx0, px0))
            ih = max(0.0, min(gy1, py1) - max(gy0, py0))
            smaller = min((gx1 - gx0) * (gy1 - gy0), max((px1 - px0) * (py1 - py0), 1e-6))
            if smaller > 0 and (iw * ih) / smaller > 0.5:
                template_hit += 1
                continue
            placements.append([
                max(0, min(1000, int(round(gy0 / h * 1000)))),
                max(0, min(1000, int(round(gx0 / w * 1000)))),
                max(0, min(1000, int(round(gy1 / h * 1000)))),
                max(0, min(1000, int(round(gx1 / w * 1000)))),
            ])
        return placements, template_hit

    def largest_outline_item(indices):
        outlines = []
        for index in indices:
            if prims[index]["c"][0] != "S":
                continue
            box = _prim_bbox(prims[index])
            if box is not None:
                outlines.append((index, box))
        if not outlines:
            return None
        return max(
            outlines,
            key=lambda ib: ((ib[1][2] - ib[1][0])
                            * (ib[1][3] - ib[1][1]),
                            prims[ib[0]]["s"]),
        )

    def largest_outline(indices):
        item = largest_outline_item(indices)
        return item[1] if item is not None else None

    line_required = {i for i in required if prims[i]["c"][0] == "L"}
    closed_required = {i for i in required if prims[i]["c"][0] == "S"}
    vector_required = line_required | closed_required
    text_required = {i for i in required if prims[i]["c"][0] == "T"}
    template_outline_item = largest_outline_item(allow)
    template_outline = (template_outline_item[1]
                        if template_outline_item is not None else None)
    template_outline_source = (
        prims[template_outline_item[0]].get("src")
        if template_outline_item is not None else None)
    prims_by_source = None

    def source_identity_signature(outline, sources):
        """Identity ink from matched glyph sources, excluding background.

        This is the negative half of vector-glyph matching.  C is a subset
        of G and F is a subset of E, so a positive stroke match alone is not
        enough.  Count every compact inner stroke/loop belonging to the
        sources that supplied the matched glyph.  Unrelated plan geometry
        inside the marker is deliberately ignored (Koch P3 has many valid
        markers drawn over background vectors), while an extra E/G stroke
        from the same glyph source is retained and rejects the subset match.
        """
        nonlocal prims_by_source
        if outline is None or not sources:
            return None
        if prims_by_source is None:
            prims_by_source = defaultdict(list)
            for index, p in enumerate(prims):
                if p.get("src") is not None:
                    prims_by_source[p["src"]].append(index)
        ox0, oy0, ox1, oy1 = outline
        inset = max(0.5, min(ox1 - ox0, oy1 - oy0) * 0.08)
        inner = (ox0 + inset, oy0 + inset, ox1 - inset, oy1 - inset)
        signature = Counter()
        for src in sources:
            for index in prims_by_source.get(src, ()):
                p = prims[index]
                if p["c"][0] not in ("L", "S"):
                    continue
                box = _prim_bbox(p)
                if (box is not None and inner[0] <= box[0]
                        and inner[1] <= box[1] and box[2] <= inner[2]
                        and box[3] <= inner[3]):
                    signature[p["c"]] += 1
        return signature

    template_vector_sources = {
        prims[index].get("src") for index in vector_required
        if prims[index].get("src") is not None
    }
    template_vector_signature = (
        source_identity_signature(template_outline, template_vector_sources)
        if vector_required else None)
    template_line_classes = {prims[index]["c"] for index in line_required}

    # A closed outlined letter (O/D/etc.) can gain an independently exported
    # tail and become another letter (Q/R).  The tail is an L primitive while
    # the loop is an S primitive, so derive compatible line styles from the
    # loop's stroke attributes and use its actual contour as connectivity
    # seed.  This is a negative superset check; the positive closed-contour
    # signature remains the primary identity test.
    identity_line_classes = set(template_line_classes)
    for index in closed_required:
        cls = prims[index]["c"]
        if len(cls) < 6:
            continue
        stroke, width, dash = cls[2], cls[4], cls[5]
        identity_line_classes.update(
            candidate_cls for candidate_cls in by_class
            if (candidate_cls[0] == "L" and len(candidate_cls) >= 5
                and candidate_cls[1] == stroke
                and candidate_cls[2] == width
                and candidate_cls[3] == dash))

    draw_offsets = [
        src - template_outline_source for src in template_vector_sources
        if (isinstance(src, int)
            and isinstance(template_outline_source, int)
            and src != template_outline_source)
    ]
    if draw_offsets and all(offset > 0 for offset in draw_offsets):
        identity_draw_direction = 1
    elif draw_offsets and all(offset < 0 for offset in draw_offsets):
        identity_draw_direction = -1
    else:
        identity_draw_direction = 0
    identity_source_span = min(
        8, max((abs(offset) for offset in draw_offsets), default=0) + 2)

    template_contour_edges = [
        edge for index in closed_required for edge in contour_edges_of(index)
    ]
    template_connected_signature = (
        _connected_line_identity(
            prims, by_class, vector_required, template_contour_edges,
            identity_line_classes, template_outline, units=units, grid=grid,
            outline_source=template_outline_source,
            draw_direction=identity_draw_direction,
            source_span=identity_source_span)
        if vector_required and identity_line_classes else None)

    template_compact_signature = (
        _compact_line_identity(prims, by_class, line_required,
                               template_line_classes)
        if line_required else None)
    template_text_signature = Counter(prims[i]["c"] for i in text_required)

    def _run(scale):
        matched = match_template(
            prims, by_class, grid, sel, seltext,
            prim_allow=set(allow), prim_required=required, scale=scale,
            contour_signature_of=contour_signature_of)
        if ("error" in matched
                or (not vector_required
                    and not (text_required and template_outline is not None))):
            return matched

        exact_groups = []
        exact_required_hits = []
        groups = matched.get("groups") or []
        required_groups = matched.get("required_hits") or [set()] * len(groups)
        for group, required_hits in zip(groups, required_groups):
            outline_item = largest_outline_item(group)
            if outline_item is None:
                continue
            outline_index, outline = outline_item
            candidate_outline_source = prims[outline_index].get("src")
            candidate_closed = {
                index for index in required_hits
                if prims[index]["c"][0] == "S"
            }
            if vector_required:
                candidate_identity = {
                    index for index in required_hits
                    if prims[index]["c"][0] == "L"
                } | candidate_closed
                candidate_sources = {
                    prims[index].get("src") for index in candidate_identity
                    if prims[index].get("src") is not None
                }
                if (source_identity_signature(outline, candidate_sources)
                        != template_vector_signature):
                    continue
                if identity_line_classes:
                    candidate_edges = [
                        edge for index in candidate_closed
                        for edge in contour_edges_of(index)
                    ]
                    if (_connected_line_identity(
                            prims, by_class, candidate_identity,
                            candidate_edges, identity_line_classes, outline,
                            units=units, grid=grid,
                            outline_source=candidate_outline_source,
                            draw_direction=identity_draw_direction,
                            source_span=identity_source_span)
                            != template_connected_signature):
                        continue
            if line_required:
                if (_compact_line_identity(
                        prims, by_class, group, template_line_classes)
                        != template_compact_signature):
                    continue
            if text_required:
                ox0, oy0, ox1, oy1 = outline
                candidate_text = Counter(
                    p["c"] for p in prims
                    if p["c"][0] == "T"
                    and ox0 <= p["x"] <= ox1
                    and oy0 <= p["y"] <= oy1)
                if candidate_text != template_text_signature:
                    continue
            exact_groups.append(group)
            exact_required_hits.append(set(required_hits))

        matched = dict(matched)
        matched["groups"] = exact_groups
        matched["matched"] = (set().union(*exact_groups)
                              if exact_groups else set())
        matched["count"] = len(exact_groups)
        matched["required_hits"] = exact_required_hits
        return matched

    used_scale = 1.0
    res = _run(1.0)
    if "error" in res:
        return res
    placements, template_hit = _placements_of(res)

    if not placements:
        # Scale hypotheses: legends are often drawn at a different size than
        # the plan callouts (paducah P137: keyed-note boxes are 1.4× larger
        # on the plan — zero matches at scale 1). The rarest template
        # class's page-wide size spread yields the candidate ratios.
        vecs = [i for i in allow if prims[i].get("src") is not None]
        if vecs:
            ref = min(vecs, key=lambda i: (len(by_class[prims[i]["c"]]),
                                           -prims[i]["s"]))
            ref_s = prims[ref]["s"] or 1.0
            buckets = defaultdict(list)
            for k in by_class[prims[ref]["c"]]:
                r = prims[k]["s"] / ref_s
                if 0.4 <= r <= 3.0 and abs(r - 1.0) > 0.12:
                    buckets[round(r, 1)].append(r)
            for grp_r in sorted(buckets.values(), key=len, reverse=True)[:3]:
                sc = sorted(grp_r)[len(grp_r) // 2]
                r2 = _run(sc)
                if "error" in r2:
                    continue
                p2, th2 = _placements_of(r2)
                if p2:
                    res, placements, template_hit, used_scale = r2, p2, th2, sc
                    break

    return {"placements": placements, "count": len(placements),
            "template_hit": template_hit, "rescued": res.get("rescued", 0),
            "template_prims": res.get("template_prims", 0),
            "template_texts": len(seltext), "period": res.get("period", False),
            "single": res.get("single", False), "scale": round(used_scale, 3)}
