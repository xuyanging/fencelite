"""放置锚的引线 / 箭头 —— 纯几何，不依赖绘制顺序.

为什么单独有这一步：`steps/arrows.py` 的边车靠「callout 的文字和它的引线在
内容流里挨着画」这个先验做簇形成，对文字 callout 很有效；但对**编号标记**
（圆圈里一个数字）在 combined_bid P20 上 12 个放置全部失败（`source:
"text-only"`、`roi_callout_count: 0`）。实测排除了三种可能：锚框太小（pad 到
12 单位无变化）、绘制顺序太远（成功与失败的 seqno 间隔都是 3）、箭头形状认不出
（是真实的 12 段填充多边形）。差异在边车内部的簇形成里，而它是个 291 KB 的
bundle。

所以这一步不去修簇形成，而是换一条**对放置锚成立的强得多的先验**：

    一个放置就是一个已知的编号标记 —— 引线必然从它的外框边缘出发，
    另一端有一个紧凑的填充箭头。

判据只用几何，与绘制顺序无关：
  * 引线 = 一条笔画，一端落在标记外框上（距中心 <= 半径 + ROOT_SLACK），
    另一端至少再远 MIN_REACH；
  * 箭头 = 引线远端附近的紧凑填充图元。**必须找到箭头才发布** —— 平面图上
    标记常压在围栏线上，附近几十条 hatch 短刺，没有箭头这条硬要求就会乱连；
  * 薄填充条（气泡自己的扫描线填充，实测每个标记 ~31 条、最薄 0.12pt）靠
    「最小边 > SLIVER_PX」排除，否则会被当成箭头。

坐标：进出都是页面帧 0-1000（[ymin, xmin, ymax, xmax] / [y, x]），与文字框、
symbol 框、边车输出同帧。中间在 core.vecgeom 的 render-pixel 帧里算，因为那一
层已经把 /Rotate 烘进去了，且结果有缓存。
"""
import math

from core.vecgeom import _extract_page

# 判据阈值，单位都是 render px（core.vecgeom 的帧，长边 ~3000px）。
ROOT_SLACK = 3.0      # 根端离标记外框的容差
MIN_REACH = 8.0       # 远端至少要比根端再远这么多，否则是外框自己的笔画
HEAD_WIN = 12.0       # 在远端多大范围内找箭头
SLIVER_PX = 1.2       # 最小边不超过这个的填充图元是扫描线条，不是箭头
MAX_HEAD_PX = 24.0    # 比这还大的填充块不是箭头（是色块 / 图案填充）
TIP_BOX_PX = 3.0      # 没有箭头框可用时，末端框的半边长


def _poly(item):
    """geom 的一条 item -> [(x, y), ...]（render-pixel 帧）。"""
    op = item[0]
    v = item[1:]
    if op == "l":
        return [(v[0], v[1]), (v[2], v[3])]
    if op == "c":
        return [(v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7])]
    if op == "re":
        x0, y0, x1, y1 = v[0], v[1], v[2], v[3]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    if op == "qu":
        return [(v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7]),
                (v[0], v[1])]
    return []


def _to_frame(pts, w, h):
    """render-pixel [(x, y), ...] -> 页面帧 [[y, x], ...]，0-1000。"""
    out = []
    for x, y in pts:
        out.append([round(max(0.0, min(1000.0, y / h * 1000)), 1),
                    round(max(0.0, min(1000.0, x / w * 1000)), 1)])
    return out


def _dist(p, cx, cy):
    return math.hypot(p[0] - cx, p[1] - cy)


def marker_leaders(pdf_path, page_index, anchors, require_head=True):
    """为每个放置锚找引线 + 箭头。

    anchors : [(key, box_2d), ...] —— box_2d 是页面帧 0-1000。key 不透明。
    返回    : {key: entry}，只包含真的找到引线的锚。entry 的结构与
              steps.arrows.find_page_arrows 一致，前端可直接吃：
              {leader_strokes, arrow_strokes, targets:[{tip, box_2d,
               terminal_kind}], confidence, note}
    """
    if not anchors:
        return {}
    data = _extract_page(pdf_path, page_index)
    units, geom = data["units"], data["geom"]
    w, h = data["w"], data["h"]
    if not units or not w or not h:
        return {}

    # 预分类：填充块（箭头候选）与笔画（引线候选）。
    heads, strokes = [], []
    for i, u in enumerate(units):
        ux0, uy0, ux1, uy1 = u[1], u[2], u[3], u[4]
        uw, uh = ux1 - ux0, uy1 - uy0
        filled = u[6] != -1
        if filled:
            if min(uw, uh) > SLIVER_PX and max(uw, uh) <= MAX_HEAD_PX:
                heads.append((i, ux0, uy0, ux1, uy1))
        else:
            strokes.append((i, ux0, uy0, ux1, uy1))

    out = {}
    for key, box in anchors:
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        by0, bx0, by1, bx1 = [float(v) for v in box]
        px0, py0 = bx0 / 1000 * w, by0 / 1000 * h
        px1, py1 = bx1 / 1000 * w, by1 / 1000 * h
        cx, cy = (px0 + px1) / 2, (py0 + py1) / 2
        # 标记「半径」用框的半长边 —— 放置框修好之后它就是外圈本身
        # （core/symbolmatch.py 的 _enclosing_outline）。
        radius = max(px1 - px0, py1 - py0) / 2.0
        if radius <= 0:
            continue
        root_max = radius + ROOT_SLACK

        best = None
        for (ui, ux0, uy0, ux1, uy1) in strokes:
            # 便宜的预筛：整条路径的 bbox 得够近
            if (ux1 < cx - root_max - 400 or ux0 > cx + root_max + 400
                    or uy1 < cy - root_max - 400 or uy0 > cy + root_max + 400):
                continue
            for item in geom[ui]:
                pts = _poly(item)
                if len(pts) < 2:
                    continue
                d0, d1 = _dist(pts[0], cx, cy), _dist(pts[-1], cx, cy)
                near, far = (pts[0], d0), (pts[-1], d1)
                if d1 < d0:
                    near, far = far, near
                if near[1] > root_max:
                    continue
                if far[1] < near[1] + MIN_REACH:
                    continue
                if best is None or far[1] > best[0]:
                    best = (far[1], pts, near, far, ui)
        if best is None:
            continue
        _reach, pts, near, far, ui = best

        # 箭头：远端附近最近的紧凑填充块
        fx, fy = far[0]
        head = None
        for (hi, hx0, hy0, hx1, hy1) in heads:
            if hi == ui:
                continue
            if hx1 < fx - HEAD_WIN or hx0 > fx + HEAD_WIN \
                    or hy1 < fy - HEAD_WIN or hy0 > fy + HEAD_WIN:
                continue
            hcx, hcy = (hx0 + hx1) / 2, (hy0 + hy1) / 2
            d = math.hypot(hcx - fx, hcy - fy)
            if head is None or d < head[0]:
                head = (d, hi, hx0, hy0, hx1, hy1)
        if head is None and require_head:
            continue

        if head is not None:
            _d, hi, hx0, hy0, hx1, hy1 = head
            arrow_strokes = [_to_frame(_poly(it), w, h) for it in geom[hi]
                             if len(_poly(it)) >= 2]
            tbox = (hx0, hy0, hx1, hy1)
            kind = "arrowhead"
        else:
            arrow_strokes = []
            tbox = (fx - TIP_BOX_PX, fy - TIP_BOX_PX,
                    fx + TIP_BOX_PX, fy + TIP_BOX_PX)
            kind = "bare-end"

        out[key] = {
            "leader_strokes": [_to_frame(pts, w, h)],
            "arrow_strokes": arrow_strokes,
            "targets": [{
                "tip": _to_frame([(fx, fy)], w, h)[0],
                "box_2d": [
                    round(max(0.0, min(1000.0, tbox[1] / h * 1000)), 1),
                    round(max(0.0, min(1000.0, tbox[0] / w * 1000)), 1),
                    round(max(0.0, min(1000.0, tbox[3] / h * 1000)), 1),
                    round(max(0.0, min(1000.0, tbox[2] / w * 1000)), 1),
                ],
                "terminal_kind": kind,
            }],
            "confidence": "high" if kind == "arrowhead" else "medium",
            "note": f"geometric · root on marker · {kind}",
        }
    return out
