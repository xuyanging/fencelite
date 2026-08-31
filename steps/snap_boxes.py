"""用矢量层把图例的两种框校准 —— 纯本地几何，零模型调用.

模型在整页图上给的框会漂：实测 drawings_volume_4_binder P4 的九个 shape 样例，
框全部落在同一列 x=790..800，而真实标记（包着编码字的那个小多边形）在
x=799.7..808.8，`4DFG` / `6DFG` 还纵向偏了 6 个单位。**但真值就在矢量层里**：

  * 编码字（"3G" / "4CL"）是文字层里的一个 token，位置是精确的；
  * 包住这个 token 的**最小闭合图形**就是那个标记（六边形 / 旗形 / 圆）。

于是有两件确定性的事可做，都不需要再问模型：

  snap_symbol_boxes   shape 样例的框 → 吸附到那个标记（找不到闭合图形就退到
                      编码字自身的框）。顺带把矫正前的框留在 box_raw。
  text_trim_boxes     图例行的文字框把行首编码含在里面时（"3G  3'-0\" WIDE
                      GATE LOCATION..."），把左边缘裁到标记右侧 —— 文字框只
                      该框文字，不该框住那个符号。

两者都**不改 results.json 里的 item**：``store.sig_of`` 把每个 item 的
(text, box_2d) 算进付费缓存签名，动一下就要重跑步骤② 的付费推理。symbol 的框
存在 symbols.json 里、不进签名，可以就地改；文字框的裁剪结果单独存成一张
{item 下标: 框} 的表，由发布层套用。
"""
from core.vecgeom import _extract_page, _get_prims

# 吸附时允许编码字与模型给的框在纵向上错开多少（0-1000 帧）。实测最大错开
# 6 个单位（4DFG），给 12 的余量；再大就不敢认了，宁可保留模型的框。
ROW_TOL = 12.0
# 找不到闭合图形、退到编码字自身框时，四周留一点余量（编码字比标记小一圈）。
GLYPH_PAD = 1.0
# 文字框裁掉编码之后，与标记右沿之间留的缝。
TRIM_GAP = 0.5


def _norm(text):
    return "".join(str(text or "").split()).upper()


def _page_geometry(pdf_path, page_index):
    """(编码字候选, 闭合图形框) —— 都换算到 0-1000 页面帧。两者都走
    vecgeom 的 LRU 缓存，同一页被反复问也只解析一次。"""
    data = _extract_page(pdf_path, page_index)
    width, height = data["w"], data["h"]
    texts = []
    for token in data["texts"]:
        raw = str(token.get("text") or "").strip()
        if not raw:
            continue
        texts.append({
            "key": _norm(raw),
            "box": [token["y0"] / height * 1000.0, token["x0"] / width * 1000.0,
                    token["y1"] / height * 1000.0, token["x1"] / width * 1000.0],
        })
    prims, _by_class, _grid = _get_prims(pdf_path, page_index)
    shapes = []
    for prim in prims:
        klass = prim.get("c")
        bbox = prim.get("bbox")
        if not bbox or not klass or klass[0] != "S":
            continue
        x0, y0, x1, y1 = bbox
        shapes.append([y0 / height * 1000.0, x0 / width * 1000.0,
                       y1 / height * 1000.0, x1 / width * 1000.0])
    return texts, shapes


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _center(box):
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _contains_center(outer, inner, pad=1.0):
    cy, cx = _center(inner)
    return (outer[0] - pad <= cy <= outer[2] + pad
            and outer[1] - pad <= cx <= outer[3] + pad)


def _code_glyph(value, box, texts):
    """与这个样例同一行、内容等于它 value 的那个编码字。

    先按「纵向与模型框重叠或错开不超过 ROW_TOL」筛，再取横向最近的一个 ——
    同一个编码在图纸上会印很多次（平面图上每处一个），必须锚在**这一行**。
    """
    key = _norm(value)
    if not key:
        return None
    row = []
    for token in texts:
        if token["key"] != key:
            continue
        gy0, gy1 = token["box"][0], token["box"][2]
        if gy1 < box[0] - ROW_TOL or gy0 > box[2] + ROW_TOL:
            continue
        row.append(token)
    if not row:
        return None
    _cy, cx = _center(box)
    return min(row, key=lambda t: abs(_center(t["box"])[1] - cx))["box"]


def _marker_box(glyph, shapes):
    """包住编码字的最小闭合图形 —— 那就是标记本体。"""
    hits = [s for s in shapes if _contains_center(s, glyph)]
    if not hits:
        return None
    return min(hits, key=_area)


def _int_box(box):
    out = [int(round(v)) for v in box]
    if out[0] >= out[2] or out[1] >= out[3]:
        return None
    return [max(0, min(1000, v)) for v in out]


def _marker_column(symbols):
    """本页已经吸附成功的那些标记撑出来的「标记列」x 范围。

    图例每一行的标记都排在同一列上，所以只要有几个吸附成功，这一列的位置就是
    已知的 —— 那些因为模型把码读错（实测 P4 的 8DMF 被读成 6DMP）而吸附不上的
    行，可以靠「这一列 + 这一行」把标记找回来。
    """
    xs = [(s["box_2d"][1], s["box_2d"][3]) for s in symbols
          if isinstance(s, dict) and s.get("snap") in ("shape", "glyph")
          and isinstance(s.get("box_2d"), (list, tuple)) and len(s["box_2d"]) == 4]
    if len(xs) < 2:
        return None
    return min(x0 for x0, _x1 in xs), max(x1 for _x0, x1 in xs)


def _snap_by_column(symbol, column, shapes):
    """靠「标记列 + 本行」找标记 —— 编码字对不上时的兜底。

    判据：闭合图形横向落在标记列内（各边留 2 个单位），纵向与模型给的框重叠。
    命中多个时按「行中心最近」选，而不是面积最小 —— 图例行挨得很近，
    ROW_TOL 的窗口里往往同时看得见上下两行的标记，取面积最小会随机挑到邻行
    （实测 P4 的 8DMF 就被挑成了下一行 FBE 的标记）。同距再比面积。
    """
    box = symbol.get("box_raw") or symbol.get("box_2d")
    if not (column and isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    lo, hi = column
    want = _center(box)[0]
    hits = []
    for shape in shapes:
        if shape[1] < lo - 2 or shape[3] > hi + 2:
            continue
        if shape[2] < box[0] - ROW_TOL or shape[0] > box[2] + ROW_TOL:
            continue
        hits.append(shape)
    if not hits:
        return None
    return min(hits, key=lambda m: (round(abs(_center(m)[0] - want), 1), _area(m)))


def snap_symbol_boxes(pdf_path, page_index, symbols):
    """就地把 shape 样例的框吸附到真实标记上，返回统计。

    只动 category=="shape" 且有 value 的样例：line 样例没有"编码字"这个锚，
    shape 里没读出 value 的也无从下手 —— 那些保留模型原框。
    幂等：每次都从 box_raw（若有）重新算，反复跑结果一致。
    """
    summary = {"snap_shape": 0, "snap_glyph": 0, "snap_column": 0,
               "snap_skipped": 0}
    rows = [s for s in (symbols or []) if isinstance(s, dict)]
    if not rows:
        return summary
    want = [s for s in rows
            if s.get("category") == "shape" and _norm(s.get("value"))]
    if not want:
        summary["snap_skipped"] = len(rows)
        return summary
    texts, shapes = _page_geometry(pdf_path, page_index)
    deferred = []
    for symbol in rows:
        base = symbol.get("box_raw") or symbol.get("box_2d")
        if not (isinstance(base, (list, tuple)) and len(base) == 4):
            summary["snap_skipped"] += 1
            continue
        if symbol not in want:
            summary["snap_skipped"] += 1
            continue
        # A schedule row-code sample deliberately has no outline in the
        # legend: its frame was synthesized from a vector-verified N.0 parent
        # and the exact glyph lives separately in glyph_box_2d.  Collapsing
        # it to the glyph here would lose the inherited dimensions that let
        # symbolmatch recover the real enclosing marker on plan instances.
        if (symbol.get("source") == "row_code"
                and symbol.get("snap") == "inherited"
                and isinstance(symbol.get("glyph_box_2d"), (list, tuple))
                and len(symbol["glyph_box_2d"]) == 4):
            summary["snap_skipped"] += 1
            continue
        glyph = _code_glyph(symbol.get("value"), list(base), texts)
        marker = _marker_box(glyph, shapes) if glyph is not None else None
        if marker is not None:
            snapped, kind = _int_box(marker), "shape"
        elif glyph is not None:
            snapped, kind = _int_box(
                [glyph[0] - GLYPH_PAD, glyph[1] - GLYPH_PAD,
                 glyph[2] + GLYPH_PAD, glyph[3] + GLYPH_PAD]), "glyph"
        else:
            # 编码字对不上（模型把码读错了 / 编码是描边字）—— 留到第二遍，
            # 那时同页其它行已经吸附好、标记列的位置才是已知的。
            deferred.append(symbol)
            continue
        _apply(symbol, base, snapped, kind, summary)

    # 第二遍：靠「标记列 + 本行」兜底。列必须由第一遍的成果撑出来，所以不能
    # 和第一遍混在一个循环里 —— 失败项排在前面时列还不存在。
    column = _marker_column(rows)
    for symbol in deferred:
        base = symbol.get("box_raw") or symbol.get("box_2d")
        fallback = _snap_by_column(symbol, column, shapes)
        snapped = _int_box(fallback) if fallback is not None else None
        if snapped is None:
            symbol["snap"] = "no_code_glyph"
            summary["snap_skipped"] += 1
            continue
        _apply(symbol, base, snapped, "column", summary)
    return summary


def _apply(symbol, base, snapped, kind, summary):
    if snapped is None:
        symbol["snap"] = "degenerate"
        summary["snap_skipped"] += 1
        return
    if list(base) != snapped:
        symbol.setdefault("box_raw", [round(float(v), 1) for v in base])
    symbol["box_2d"] = snapped
    symbol["snap"] = kind
    summary["snap_" + kind] += 1


def text_trim_boxes(pdf_path, page_index, items, symbols):
    """图例行文字框里含着行首编码时，把左边缘裁到标记右侧.

    返回 {item 下标: 裁剪后的框}；没有需要裁的就返回 {}。
    只处理「文字的第一个 token 等于某个样例的 value」这一种情形 —— 那正是
    "3G  3'-0\\" WIDE GATE LOCATION- VINYL CHAIN LINK" 这种排版。
    """
    trims = {}
    items = items if isinstance(items, list) else []
    codes = {}
    for symbol in (symbols or []):
        if not isinstance(symbol, dict):
            continue
        key = _norm(symbol.get("value"))
        box = symbol.get("box_2d")
        if key and isinstance(box, (list, tuple)) and len(box) == 4:
            codes.setdefault(key, []).append(list(box))
    if not codes:
        return trims
    # 本页所有标记按纵向排好，供 _row_band 找上下邻居
    ladder = sorted((m for boxes in codes.values() for m in boxes),
                    key=lambda m: m[0])
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        tokens = str(item.get("text") or "").split()
        box = item.get("box_2d")
        if len(tokens) < 2 or not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        marks = codes.get(_norm(tokens[0]))
        if not marks:
            continue
        # 同一行、且确实压在文字框左段里的那个标记
        same_row = [m for m in marks
                    if not (m[2] < box[0] - ROW_TOL or m[0] > box[2] + ROW_TOL)
                    and m[3] > box[1] and m[1] < box[3]]
        if not same_row:
            continue
        right = max(m[3] for m in same_row) + TRIM_GAP
        mine = min(same_row, key=lambda m: abs(_center(m)[0] - _center(box)[0]))
        top, bottom = _row_band(mine, ladder, box)
        left = right if box[1] < right < box[3] else box[1]
        if left >= box[3] or top >= bottom:
            continue      # 裁完就没了 —— 不动
        if [left, top, bottom] == [box[1], box[0], box[2]]:
            continue      # 没有任何变化，不必落表
        trims[index] = [round(float(top), 1), round(float(left), 1),
                        round(float(bottom), 1), round(float(box[3]), 1)]
    return trims


def _row_band(marker, ladder, box):
    """这一行文字纵向的合法范围 —— 用相邻标记的中点当行边界.

    图例是一行一个标记，所以「上一个标记的下沿」与「本标记的上沿」之间那条缝
    就是行的分界。步骤① 的吸附是按词命中率把矢量行并进来的，遇到相邻两行**第二
    行文字完全相同**（实测 P4 的 4DFG / 6DFG 都是
    "FENCE GATES W/ LOCKING HARDWARE- SEE SPECS"）就会把邻行也并进来，框于是
    纵向串到下一行去。用标记中点卡一刀，行与行就只会相切、不会重叠。
    """
    above = [m for m in ladder if m[2] <= marker[0] + 0.01]
    below = [m for m in ladder if m[0] >= marker[2] - 0.01]
    top = box[0]
    bottom = box[2]
    if above:
        edge = (max(m[2] for m in above) + marker[0]) / 2.0
        top = max(top, edge)
    if below:
        edge = (marker[2] + min(m[0] for m in below)) / 2.0
        bottom = min(bottom, edge)
    # 至少要把标记那一行本身留住，否则 2 行的说明会被削成 1 行
    return min(top, marker[0]), max(bottom, marker[2])
