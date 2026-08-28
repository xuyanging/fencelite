"""页矢量层 —— PDF 绘图命令 → 原子图元 + 索引（纯本地，零模型调用）.

这一层是 shape 符号匹配（core/symbolmatch.py）和步骤 1 文字清理
（marker code 剥离）共同的底座：把一页的 PDF 矢量绘图命令解析成
「渲染像素坐标」（每页自适应 zoom，页面 /Rotate 经 page.rotation_matrix
预先烘进坐标——这是原项目踩过的坐标坑），再原子化成能做模板匹配的图元。

为什么不栅格化：图例符号在整页图上只有几个像素，视觉匹配又贵又不准；
而矢量命令里同一个 CAD block 的每一次放置都是同样的几何，做平移／旋转／
镜像不变的精确模板匹配就能零成本找全，文字内容还能当判别器
（circle-2 永远不会匹配到 circle-3）。

提取很重（逐点矩阵变换），所以按 (path, mtime, page) 做 LRU 缓存；
所有 fitz 调用都在 core.pdfio.FITZ_LOCK 之下（MuPDF 不是线程安全的）。
"""
import math
import threading
from collections import OrderedDict, defaultdict

import fitz

# Same render-space convention as the source tool.
VEC_TARGET_LONG = 3000.0
VEC_ZOOM_MIN, VEC_ZOOM_MAX = 1.0, 6.0
PRIM_TOL = 2.0            # endpoint / position quantization tolerance (px)

# Extraction is heavy on dense pages (per-point transforms) — cache a few
# pages. Keyed by (path, mtime, page); LRU.
_VEC_CACHE: "OrderedDict[tuple, dict]" = OrderedDict()
_PRIMS_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_MAX = 6
_CACHE_LOCK = threading.Lock()


def _color_to_int(c):
    """PyMuPDF color (0..1 float tuple) -> 0xRRGGBB, -1 when absent."""
    if c is None:
        return -1
    try:
        if len(c) == 1:
            g = max(0, min(255, int(round(c[0] * 255))))
            return (g << 16) | (g << 8) | g
        if len(c) >= 3:
            r = max(0, min(255, int(round(c[0] * 255))))
            g = max(0, min(255, int(round(c[1] * 255))))
            b = max(0, min(255, int(round(c[2] * 255))))
            return (r << 16) | (g << 8) | b
    except Exception:
        return -1
    return -1


def _r1(v):
    return round(float(v), 1)


def _cache_get(cache, key):
    with _CACHE_LOCK:
        v = cache.get(key)
        if v is not None:
            cache.move_to_end(key)
        return v


def _cache_put(cache, key, value):
    with _CACHE_LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_MAX:
            cache.popitem(last=False)


def _extract_page(pdf_path, page_index):
    """Vector layer of one page in render-pixel coordinates.

    Returns {zoom, w, h, units, dashes, geom, texts}:
      units[i] = [idx, x0, y0, x1, y1, stroke_int, fill_int, width,
                  nl, nc, nre, nqu]           (bbox via rotation_matrix*zoom)
      dashes[i] = the path's dash pattern string ("" when solid)
      geom[i]  = [["l",x1,y1,x2,y2] | ["c",8 coords] | ["re",4] | ["qu",8]]
      texts    = word-level [{id,x0,y0,x1,y1,text,font,size,color}]
    """
    key = (str(pdf_path), _mtime(pdf_path), int(page_index))
    hit = _cache_get(_VEC_CACHE, key)
    if hit is not None:
        return hit

    from core.pdfio import FITZ_LOCK
    with FITZ_LOCK:
        return _extract_page_locked(pdf_path, page_index, key)


def _extract_page_locked(pdf_path, page_index, key):
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_index)
        long_side = max(page.rect.width, page.rect.height) or 1.0
        zoom = max(VEC_ZOOM_MIN, min(VEC_ZOOM_MAX, VEC_TARGET_LONG / long_side))
        # Unrotated PDF points -> render pixels (page /Rotate baked in).
        pmat = page.rotation_matrix * fitz.Matrix(zoom, zoom)
        # get_pixmap(matrix=scale) size == rect w/h * zoom (no swap on
        # /Rotate) and matches the pmat space — same pixel frame.
        w = int(round(page.rect.width * zoom))
        h = int(round(page.rect.height * zoom))

        units, dashes, geom = [], [], []
        for idx, d in enumerate(page.get_drawings()):
            rr = (d.get("rect") or fitz.Rect(0, 0, 0, 0)) * pmat
            x0, y0, x1, y1 = rr.x0, rr.y0, rr.x1, rr.y1
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            nl = nc = nr = nq = 0
            gitems = []
            for it in d["items"]:
                op = it[0]
                if op == "l":
                    nl += 1
                    p1 = it[1] * pmat
                    p2 = it[2] * pmat
                    gitems.append(["l", _r1(p1.x), _r1(p1.y), _r1(p2.x), _r1(p2.y)])
                elif op == "c":
                    nc += 1
                    p1 = it[1] * pmat
                    p2 = it[2] * pmat
                    p3 = it[3] * pmat
                    p4 = it[4] * pmat
                    gitems.append(["c", _r1(p1.x), _r1(p1.y), _r1(p2.x), _r1(p2.y),
                                   _r1(p3.x), _r1(p3.y), _r1(p4.x), _r1(p4.y)])
                elif op == "re":
                    nr += 1
                    er = it[1] * pmat
                    a0, b0, a1, b1 = er.x0, er.y0, er.x1, er.y1
                    if a1 < a0:
                        a0, a1 = a1, a0
                    if b1 < b0:
                        b0, b1 = b1, b0
                    gitems.append(["re", _r1(a0), _r1(b0), _r1(a1), _r1(b1)])
                elif op == "qu":
                    nq += 1
                    q = it[1]
                    ul = q.ul * pmat
                    ur = q.ur * pmat
                    lr = q.lr * pmat
                    ll = q.ll * pmat
                    gitems.append(["qu", _r1(ul.x), _r1(ul.y), _r1(ur.x), _r1(ur.y),
                                   _r1(lr.x), _r1(lr.y), _r1(ll.x), _r1(ll.y)])
            units.append([idx, _r1(x0), _r1(y0), _r1(x1), _r1(y1),
                          _color_to_int(d.get("color")), _color_to_int(d.get("fill")),
                          round(float(d.get("width") or 0), 2), nl, nc, nr, nq])
            dashes.append(d.get("dashes") or "")
            geom.append(gitems)

        # Word-level text layer (a span merges "LC ... LC" into one — words
        # let every repeated tag stand alone). Same pmat transform.
        texts = []

        def _emit_word(chars, font, size, color):
            s = "".join(c for c, _ in chars).strip()
            if not s:
                return
            xs0 = min(bb[0] for _, bb in chars)
            ys0 = min(bb[1] for _, bb in chars)
            xs1 = max(bb[2] for _, bb in chars)
            ys1 = max(bb[3] for _, bb in chars)
            rr2 = fitz.Rect(xs0, ys0, xs1, ys1) * pmat
            tx0, ty0, tx1, ty1 = rr2.x0, rr2.y0, rr2.x1, rr2.y1
            if tx1 < tx0:
                tx0, tx1 = tx1, tx0
            if ty1 < ty0:
                ty0, ty1 = ty1, ty0
            texts.append({"id": len(texts), "x0": _r1(tx0), "y0": _r1(ty0),
                          "x1": _r1(tx1), "y1": _r1(ty1),
                          "text": s, "font": font, "size": size, "color": color})

        try:
            for b in page.get_text("rawdict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                for ln in b.get("lines", []):
                    for sp in ln.get("spans", []):
                        font = sp.get("font")
                        size = round(float(sp.get("size") or 0), 1)
                        color = sp.get("color")
                        cur = []
                        for ch in sp.get("chars", []):
                            if (ch.get("c") or " ").isspace():
                                _emit_word(cur, font, size, color)
                                cur = []
                            else:
                                cur.append((ch["c"], ch["bbox"]))
                        _emit_word(cur, font, size, color)
        except Exception:
            pass

        data = {"zoom": zoom, "w": w, "h": h, "units": units,
                "dashes": dashes, "geom": geom, "texts": texts}
    finally:
        doc.close()

    _cache_put(_VEC_CACHE, key, data)
    return data


def _mtime(p):
    try:
        import os
        return int(os.path.getmtime(p))
    except OSError:
        return 0


def _decompose(units, geom, zoom):
    """Split every path into minimal parts: small closed shapes + free
    straight segments. shape=(src, kind, cx, cy, x0,y0,x1,y1, size_pt,
    color, width); seg=(src, ax,ay,bx,by, color, width). A combo path
    (line + square) naturally splits into {square + segments}."""
    shapes = []
    segs = []
    for u in units:
        i = u[0]
        col = u[5]
        w = u[7]
        g = geom[i]
        if not g:
            continue
        nl, nc, nr, nq = u[8], u[9], u[10], u[11]
        if nr or nq:
            for it in g:
                op = it[0]
                if op == "re":
                    x0, y0, x1, y1 = it[1], it[2], it[3], it[4]
                    if x1 < x0:
                        x0, x1 = x1, x0
                    if y1 < y0:
                        y0, y1 = y1, y0
                    sz = ((x1 - x0) + (y1 - y0)) / 4 / zoom
                    if 0.4 <= sz <= 25:
                        shapes.append((i, "rect", (x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1, y1, sz, col, w))
                elif op == "qu":
                    xs = (it[1], it[3], it[5], it[7])
                    ys = (it[2], it[4], it[6], it[8])
                    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                    sz = ((x1 - x0) + (y1 - y0)) / 4 / zoom
                    if 0.4 <= sz <= 25:
                        shapes.append((i, "quad", (x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1, y1, sz, col, w))
                elif op == "l":
                    segs.append((i, it[1], it[2], it[3], it[4], col, w))
            continue
        if nc >= 2 and nl == 0:                       # all curves = circle
            x0, y0, x1, y1 = u[1], u[2], u[3], u[4]
            sz = ((x1 - x0) + (y1 - y0)) / 4 / zoom
            if 0.4 <= sz <= 25:
                shapes.append((i, "circle", (x0 + x1) / 2, (y0 + y1) / 2, x0, y0, x1, y1, sz, col, w))
            continue
        if nl > 0:                                    # polyline: embedded loops -> shapes; rest -> segments
            # A single PDF drawing may contain several independent ``move``
            # subpaths. PyMuPDF exposes only their line/curve items here, so
            # flattening every endpoint into one V list invents a bridge from
            # the previous subpath to the next. Keep original item edges and
            # restart loop detection whenever geometric continuity breaks.
            runs = []
            run = []
            for it in g:
                if it[0] == "l":
                    edge = ((it[1], it[2]), (it[3], it[4]))
                elif it[0] == "c":
                    # Preserve the legacy curve chord representation (it is
                    # classed separately downstream), but never join it to an
                    # unrelated preceding subpath.
                    edge = ((it[1], it[2]), (it[7], it[8]))
                else:
                    continue
                if run:
                    px, py = run[-1][1]
                    ax, ay = edge[0]
                    if abs(px - ax) > 0.75 or abs(py - ay) > 0.75:
                        runs.append(run)
                        run = []
                run.append(edge)
            if run:
                runs.append(run)

            for run in runs:
                V = [run[0][0], *(edge[1] for edge in run)]
                n = len(V)
                used = [False] * len(run)
                vidx = {}
                for t in range(n):
                    key = (round(V[t][0]), round(V[t][1]))
                    kk = vidx.get(key)
                    if kk is not None and 3 <= t - kk <= 12:
                        lx = [V[s][0] for s in range(kk, t + 1)]
                        ly = [V[s][1] for s in range(kk, t + 1)]
                        x0, y0, x1, y1 = min(lx), min(ly), max(lx), max(ly)
                        sz = ((x1 - x0) + (y1 - y0)) / 4 / zoom
                        if 0.4 <= sz <= 25:
                            shapes.append((i, "poly", (x0 + x1) / 2,
                                           (y0 + y1) / 2, x0, y0, x1, y1,
                                           sz, col, w))
                            for s in range(kk, t):
                                used[s] = True
                    vidx[key] = t
                for t, edge in enumerate(run):
                    if not used[t]:
                        (ax, ay), (bx, by) = edge
                        if abs(bx - ax) + abs(by - ay) > 0.1:
                            segs.append((i, ax, ay, bx, by, col, w))
    return shapes, segs


def _build_prims(data):
    """Atomic primitives + indexes for template matching.
    prim = {src|tid, x, y (center), c (exact class), s (fuzzy scale),
    o (segment orientation in [0, pi) or None)}."""
    units = data["units"]
    zoom = data["zoom"]
    dashes = data["dashes"]
    texts = data["texts"]
    geom = data["geom"]

    def dash_of(src):
        return dashes[src] if 0 <= src < len(dashes) else ""

    shapes, segs = _decompose(units, geom, zoom)
    prims = []
    for s in shapes:
        src = s[0]
        prims.append({"src": src, "x": s[2], "y": s[3],
                      "c": ("S", s[1], units[src][5], units[src][6], round(units[src][7], 2), dash_of(src)),
                      "s": float(s[8]), "o": None,
                      "bbox": (float(s[4]), float(s[5]),
                               float(s[6]), float(s[7]))})
    for sg in segs:
        src = sg[0]
        ax, ay, bx, by = sg[1], sg[2], sg[3], sg[4]
        # A straight sub-segment of a curved path is a different class from
        # a pure straight line — pure-line templates never match arc paths.
        has_curve = (units[src][9] > 0) if 0 <= src < len(units) else False
        prims.append({"src": src, "x": (ax + bx) / 2, "y": (ay + by) / 2,
                      "c": ("L", sg[5], round(sg[6], 2), dash_of(src), has_curve),
                      "s": math.hypot(bx - ax, by - ay),
                      "o": math.atan2(by - ay, bx - ax) % math.pi,
                      "segment": (float(ax), float(ay),
                                  float(bx), float(by))})
    for t in texts:
        prims.append({"src": None, "tid": t["id"],
                      "x": (t["x0"] + t["x1"]) / 2, "y": (t["y0"] + t["y1"]) / 2,
                      "c": ("T", t["text"]),   # content-only identity (legend vs plan often differ in size)
                      "s": 1.0, "o": None,
                      "bbox": (float(t["x0"]), float(t["y0"]),
                               float(t["x1"]), float(t["y1"]))})

    by_class = defaultdict(list)
    grid = defaultdict(list)
    for idx, p in enumerate(prims):
        by_class[p["c"]].append(idx)
        grid[(int(p["x"] // PRIM_TOL), int(p["y"] // PRIM_TOL))].append(idx)
    return prims, by_class, grid


def _get_prims(pdf_path, page_index):
    key = (str(pdf_path), _mtime(pdf_path), int(page_index))
    hit = _cache_get(_PRIMS_CACHE, key)
    if hit is not None:
        return hit
    data = _extract_page(pdf_path, page_index)
    pack = _build_prims(data)
    _cache_put(_PRIMS_CACHE, key, pack)
    return pack


# Box → template selection. symbolmatch does its own center-point selection
# on prims (finer than unit level), so these two are the generic page-level
# helpers kept here for the marker / arrow layers that select by unit box.
def _covered(u_lo, u_hi, b_lo, b_hi):
    """One-axis coverage test (>=50% of the unit's own extent inside the
    box; zero-extent axes — thin lines — count as covered when inside)."""
    ext = u_hi - u_lo
    if ext <= 0.5:
        return b_lo - 0.5 <= u_lo <= b_hi + 0.5
    ov = min(u_hi, b_hi) - max(u_lo, b_lo)
    return ov / ext >= 0.5


def _select_in_box(data, x0, y0, x1, y1):
    """Template selection, mirroring the source frontend's rules: vector
    units need >=50% overlap on each axis; texts by center point."""
    sel = set()
    for u in data["units"]:
        if _covered(u[1], u[3], x0, x1) and _covered(u[2], u[4], y0, y1):
            sel.add(u[0])
    seltext = set()
    for t in data["texts"]:
        cx = (t["x0"] + t["x1"]) / 2
        cy = (t["y0"] + t["y1"]) / 2
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            seltext.add(t["id"])
    return sel, seltext
