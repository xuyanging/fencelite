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
from collections import defaultdict

from core.vecgeom import PRIM_TOL, _extract_page, _get_prims

# How much bigger than the legend template a candidate marker outline may be
# before it is rejected as "not the symbol". 1.35 covers the honest cases (a
# placement drawn slightly larger than the sample, or a template box the VLM
# cropped a hair tight) while still excluding plan features: on combined_bid
# P20 the circle is 18.1pt against a 18.1x17.3pt template (1.00x / 1.05x),
# and the next thing up that contains a marker is two orders of magnitude
# bigger.
OUTLINE_TOL = 1.35


def match_template(prims, by_class, grid, sel, seltext, prim_allow=None,
                   scale=1.0):
    """The /match algorithm from pdf_viz.py, returning per-placement groups.

    sel: set of vector path ids (unit idx) inside the template box;
    seltext: set of text ids inside the box.
    prim_allow: optional set of prim indices — when given, the template is
      exactly these prims (finer than unit-level selection; lets the caller
      drop e.g. leader stubs whose length varies per placement).
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

    def near(x, y, cls, sc):
        cx, cy = int(x // PRIM_TOL), int(y // PRIM_TOL)
        for dx in range(-NC, NC + 1):
            for dy in range(-NC, NC + 1):
                for j in grid.get((cx + dx, cy + dy), []):
                    p = prims[j]
                    if p["c"] == cls and abs(p["x"] - x) + abs(p["y"] - y) <= VTOL and lclose(p["s"], sc):
                        return j
        return -1

    # 1) template reduction. The periodic branch (dominant translation vector
    # → motif) only ever ran for line-style samples; a symbol template is the
    # selection itself, minus duplicate text classes.
    vec_idx = [t for t in tmpl0 if prims[t].get("src") is not None]
    txt_idx = [t for t in tmpl0 if prims[t].get("tid") is not None]
    txt_keep, seen_tc = [], set()
    for t in sorted(txt_idx, key=lambda t: (prims[t]["y"], prims[t]["x"])):
        cls = prims[t]["c"]
        if cls in seen_tc:
            continue
        seen_tc.add(cls)
        txt_keep.append(t)

    tmpl = vec_idx + txt_keep

    def cand_of(t):
        cls = prims[t]["c"]
        sc = prims[t]["s"] * scale
        return [k for k in by_class[cls] if lclose(prims[k]["s"], sc)]

    m = len(tmpl)
    if m == 0:
        return {"error": "no repeating unit found inside the box"}
    if m > 24:
        return {"error": f"template holds {m} primitives, too many (the selection may not be tight enough)"}

    txt_anchors = [t for t in tmpl if prims[t].get("tid") is not None]
    txt_rep = [t for t in txt_anchors if len(cand_of(t)) >= 2]
    A = min(txt_rep or txt_anchors or tmpl, key=lambda t: len(cand_of(t)))
    ax0, ay0 = prims[A]["x"], prims[A]["y"]
    cand = [t for t in tmpl if t != A and (abs(prims[t]["x"] - ax0) + abs(prims[t]["y"] - ay0)) > 2]
    B = min(cand, key=lambda t: (len(cand_of(t)),
                                 -((prims[t]["x"] - ax0) ** 2 + (prims[t]["y"] - ay0) ** 2))) if cand else None

    toff = [(prims[t]["c"], prims[t]["s"], prims[t]["x"] - ax0, prims[t]["y"] - ay0,
             prims[t].get("tid") is not None) for t in tmpl]

    if m <= 1 or B is None:
        # Degenerate single-prim template: every same-class instance is a hit.
        hits = cand_of(A)
        return {"groups": [{h} for h in hits], "matched": set(hits),
                "period": False, "count": len(hits), "rescued": 0,
                "template_prims": m, "single": True}

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

    # Symbol mode: multi-piece glyphs tolerate a proportional miss
    # (identical CAD blocks can still differ by an overlapped piece).
    max_miss = 0 if m <= 4 else max(1, m // 8)
    need = m - max_miss
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
                        miss = 0
                        hit = []
                        txt_ok = True
                        for (ck, sk, ox, oy, is_txt) in toff:
                            if mir == 1:
                                tx = ax + c * ox - s * oy
                                ty = ay + s * ox + c * oy
                            else:
                                tx = ax + c * ox + s * oy
                                ty = ay + s * ox - c * oy
                            j = near(tx, ty, ck, sk * scale)
                            if j < 0:
                                if is_txt:        # text must hit exactly
                                    txt_ok = False
                                    break
                                miss += 1
                                if miss > max_miss:
                                    break
                            else:
                                hit.append(j)
                        if txt_ok and miss <= max_miss and len(hit) >= need:
                            matches.append(frozenset(hit))

    matches.sort(key=lambda s: -len(s))           # dedupe + mutual exclusion
    final = []
    used = set()
    for s in matches:
        if s & used:
            continue
        final.append(set(s))
        used |= s
    matched = set(used)

    # 2) the draw-order (src) continuity rescue of the original algorithm was
    # gated on a period vector (Rg = 3.5 * |period|), i.e. line-style samples
    # only — a symbol match is exactly what the pose solve verified.
    return {"groups": final, "matched": matched, "period": False,
            "count": len(final), "rescued": 0,
            "template_prims": m, "single": False}


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

    # Template = prims whose center falls in the (padded) box. Symbol rules:
    #   - closed shapes + texts define the symbol;
    #   - free segments are leader stubs / table rules whose lengths vary
    #     per placement — drop them whenever any shape is present (only
    #     line-style markers with no closed shape keep their segments);
    #   - cap at 24 prims (largest shapes first) for multi-piece glyphs,
    #     e.g. a circle exported as a ring of tiny filled polys.
    allow = []
    has_marker_content = False
    for pad in (2.0, 8.0):
        ax0, ay0 = cpx0 - pad, cpy0 - pad
        ax1, ay1 = cpx1 + pad, cpy1 + pad
        inside = [i for i, p in enumerate(prims)
                  if ax0 <= p["x"] <= ax1 and ay0 <= p["y"] <= ay1]
        if inside:
            texts_sel = [i for i in inside if prims[i]["c"][0] == "T"]
            shapes_sel = [i for i in inside if prims[i]["c"][0] == "S"]
            segs_sel = [i for i in inside if prims[i]["c"][0] == "L"]
            has_marker_content = bool(shapes_sel or texts_sel)
            body = shapes_sel if shapes_sel else segs_sel
            if len(body) + len(texts_sel) > 24:
                body = sorted(body, key=lambda i: -prims[i]["s"])[:max(1, 24 - len(texts_sel))]
            allow = body + texts_sel
            if allow:
                break
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

    def _run(scale):
        return match_template(prims, by_class, grid, sel, seltext,
                              prim_allow=set(allow), scale=scale)

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
