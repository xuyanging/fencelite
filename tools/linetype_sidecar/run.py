"""线型聚类边车 —— 在**自己的 venv 里**跑 line_type_engine，一次性子进程.

为什么必须是独立进程，而不是在 webapp 里 import（三条独立的理由，任何一条都够）：

  1. 版本闸。引擎 fail-closed 要求 ``PyMuPDF>=1.28.2,<2`` 与 ``pypdf==6.14.2``
     （line_type_engine/runtime.py:28-37），而 fence_lite 跑的是 PyMuPDF
     1.27.2.3、没有 pypdf、没有 scipy。把服务升上去会改 page.get_drawings()
     的输出 → core/vecgeom.py 的图元变了 → 所有 placement 框可能静默位移。
  2. 崩进程 + 改 /Rotate。1.27.2.3 正是 PyMuPDF issue #5042 的受害版本
     （get_texttrace() 的原生引用计数缺陷**会终止解释器而不是抛异常**）；而且
     get_drawings/get_texttrace/get_bboxlog 都是「先 set_rotation(0)、算完再设
     回去」且**没有 try/finally**（pymupdf/__init__.py:11452-11497, 12198-12216），
     抽取中途抛异常会把页面永久留在 rotation=0 —— 之后宿主的 page.rect、
     rotation_matrix、get_pixmap 全都是错的而且看着正常。所以**绝不能**把
     fence_lite 还要复用的 fitz.Page 交给引擎，只给它路径、让它开自己的文档。
  3. spawn 会重入调用方的 __main__。cpu_budget>=2 时引擎会起 spawn 子进程
     （scheduling.py:86-96, 173-179），子进程以 __mp_main__ 重新执行调用方主模块
     的**模块级代码**。服务的 __main__ 是 webapp.py，那会重跑一遍 Flask 的
     import（并在 core/config.py:32-34 因缺 GEMINI_API_KEY 直接抛错）。

两个刻意的选择：

* **用 source-aligned 提取器，不用 pdf_adapter 的那条**。
  source_page_adapter.py:1-10 写得很明确：``source_content`` 才是 authored paint
  order / path topology / style 的权威，``pdf_adapter`` 的 path 列表"只作为诊断
  计数保留"。实测同一页（gladstone P2）两条路给出**不同的答案**：plain 8 个线型 /
  residual 2852，aligned 12 个 / residual 2103，type_uid 只重合 2 个。根因是
  pdf_adapter 从不设 source_provenance_exact（ir.py:131 默认 False），而
  grouping.py 的真实拆分/合并拿它当闸。代价是抽取从 5.4 s 涨到 15.0 s。
* **cpu_budget 钉死 1**。budget>=2 时 method2 的 carrier merge 走**结构不同**的
  代码路径（method2/text_family.py:2100-2180：先物化全部 candidate_lists 再按批
  speculative 求值，而 worker_count==1 是边算边 union），"结果与 budget 无关"
  没有被证明。钉成 1 才能让 budget 不必进缓存键，也顺带完全不 spawn。

绑定判据（这里只算，不做产品口径的投票，那在 steps/linetypes/bind.py）：

    一个末端的线型 = **拥有离 tip 最近那条 path op 的簇**。

  不是"最近的簇"。实测 gladstone P2 的 3414 条 path op 里只有 38% 属于任何簇，
  所以"最近的簇"经常是几 pt 之外一个真簇，而 tip 底下的那段 ink 其实是 residual
  —— 截图上完全看不出错。最近的 op 不属于任何簇时，正确答案是「这里没有线型」。
  另外必须先剔掉**这个 callout 自己的引线与箭头**：compound_path_periodic 找的
  正是周期性重复的相同图元，而重复的箭头 / 刻度短刺就是这种东西，不排除就会
  把一堆箭头认成"线型"再高亮回去。

坐标：进出都是**页面帧 0-1000**（框 [ymin,xmin,ymax,xmax]、点 [y,x]），与文字框、
symbol 框、arrows 的输出完全同帧。引擎侧是「y 向上、原点 = 未旋转 CropBox 左下、
/Rotate 只当元数据不施加」，这里负责换算。

协议：stdin 收一个 JSON job，stdout 吐一个 JSON。任何失败都以
``{"ok": false, "code": ..., "error": ...}`` + 非零退出码表达 —— **绝不吐空结果
冒充「这页没有线型」**。

job = {
  "pdf":    str,                 # 源 PDF 路径
  "pages": [                     # **一次一批**，见下面「为什么是多页」
    {"sheet": int,               #   **1-based** 页号
     "targets":[{"key": str, "ti": int, "tip": [y, x],
                 "own": [[[y,x], ...], ...]}]},   # own = 该 callout 自己的引线+箭头
    ...
  ],
  "top_k":  int,                 # 每个末端回传几个候选簇（审计用）
  "cpu_budget": int
}
# 兼容单页写法：顶层直接给 "sheet" + "targets"，等价于 pages 只有一项。

输出是 **NDJSON**，一页一行：``{"ok":true,"sheet":N,...}`` 或
``{"ok":false,"sheet":N,"code":...,"error":...}``，最后一行是
``{"ok":true,"done":true,"pages":N}``。

为什么是多页 + 流式：
  * PDF 只打开一次。SourceAlignedPdfDocument 的 docstring 明写「PDF snapshot is
    neither re-read nor re-opened per page」——逐页起进程的话，每页都要把整份
    PDF 读进来、重建 pypdf/PyMuPDF 两套文档，大图纸上这是纯浪费。
  * 进程启动与引擎 import（numpy/scipy/pymupdf/pypdf）只付一次，不是每页一次。
  * 算完一页就吐一行，父进程立刻落盘 —— 不用等整批结束，内存里也不会同时压着
    好几页的折线（单页可达 2.8 MB）。CPU 算下一页的同时，上一页的 JSON 解析与
    写盘在父进程里并行发生。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ENGINE = Path(os.environ.get("LINETYPE_ENGINE_PATH", str(_HERE / "engine")))

TOP_K_DEFAULT = 3
# 判定「这条 op 就是这个 callout 自己的引线/箭头」的容差与占比。两边的几何来自
# 同一份 PDF，坐标只差舍入（arrows.json 里的点是页帧整数），所以 1 pt 足够。
OWN_TOLERANCE_PT = 1.0
OWN_POINT_RATIO = 0.75       # op 的点有这么多落在自己笔画上就算自己的几何

# **所有距离都在 IR 帧（PDF 点）里算，不在 0-1000 页帧里算。**
# 0-1000 是逐轴归一的：gladstone P2 上 x 是 1000/2448 = 0.4085 单位/pt、
# y 是 1000/1584 = 0.6313 单位/pt，各向异性 1.545 倍 —— 在页帧里比距离会
# 系统性偏爱横向偏移的簇，而且偏多少取决于纸张长宽比，逐页不同。


def _fail(code, message):
    json.dump({"ok": False, "code": code, "error": str(message)[:2000]},
              sys.stdout, ensure_ascii=False)
    sys.stdout.flush()
    raise SystemExit(3)


def _point_to_segment(point, left, right):
    py, px = point
    ay, ax = left
    by, bx = right
    dy, dx = by - ay, bx - ax
    denom = dy * dy + dx * dx
    if denom <= 1e-12:
        return math.hypot(py - ay, px - ax)
    t = ((py - ay) * dy + (px - ax) * dx) / denom
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
    return math.hypot(py - (ay + t * dy), px - (ax + t * dx))


def _polylines_distance(point, polylines, cutoff=float("inf")):
    best = cutoff
    for line in polylines:
        for left, right in zip(line, line[1:]):
            distance = _point_to_segment(point, left, right)
            if distance < best:
                best = distance
                if best <= 0.0:
                    return 0.0
    return best


def _bbox(polylines):
    ys = [p[0] for line in polylines for p in line]
    xs = [p[1] for line in polylines for p in line]
    if not ys:
        return None
    return [round(min(ys), 2), round(min(xs), 2),
            round(max(ys), 2), round(max(xs), 2)]


def _is_own_geometry(lines, own_lines):
    """这条 op 是否就是调用方给的那几条自己的笔画（引线 / 箭头）。IR 帧、点为单位。"""
    if not own_lines:
        return False
    points = [p for line in lines for p in line]
    if not points:
        return False
    hits = 0
    for point in points:
        if _polylines_distance(point, own_lines,
                               OWN_TOLERANCE_PT * 2) <= OWN_TOLERANCE_PT:
            hits += 1
    return hits >= max(2, int(len(points) * OWN_POINT_RATIO))


# 判定「同一条走线」的接触容差，单位 PDF 点（IR 帧，等向）。
# 为什么是 0.5：实测 gladstone P4 线型 #5 的三段真实续接，几何最小距离是
# 0.02 / 0.13 页帧单位（≈0.04 / 0.25 pt）—— 它们是同一条走线被画断成多段，
# 端点在舍入误差内重合；而 lenexa P4 里那条**无关**的左侧长带与波浪围栏
# 最近 1.15 页帧单位（≈1.8~2.8 pt）—— 只是靠近，并没有接触。两者差近 10 倍，
# 0.5 pt 落在中间且离两边都远。判据是「接触」，不是「靠近」。
RUN_TOUCH_PT = float(os.environ.get("LINETYPE_RUN_TOUCH_PT", "0.5"))


def _connected_runs(op_indices, ir_geometry, tau=RUN_TOUCH_PT):
    """把一个线型的 op 按**几何接触**切成若干条走线（连通分量）。

    不用引擎的 group 号：一个 global 线型跨的几个 group 既可能是同一道围栏的
    连续段（gladstone P4 的中间横栏接在左侧竖栏上），也可能是图上完全分开的
    区域（lenexa P4 的四块）。组号分不开这两种，几何接触分得开。

    实现：按 tau 建网格索引所有线段，再拿每个 op 的顶点查邻近格，命中就并查集
    合并。点到**线段**的距离，因为 T 形接头是端点落在另一段的中间。
    """
    parent = {i: i for i in op_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    cell = max(tau * 4.0, 1.0)
    grid = {}
    for op_index in op_indices:
        for line in ir_geometry.get(op_index) or ():
            for (ax, ay), (bx, by) in zip(line, line[1:]):
                lo_x, hi_x = (ax, bx) if ax <= bx else (bx, ax)
                lo_y, hi_y = (ay, by) if ay <= by else (by, ay)
                for gx in range(int(lo_x // cell), int(hi_x // cell) + 1):
                    for gy in range(int(lo_y // cell), int(hi_y // cell) + 1):
                        grid.setdefault((gx, gy), []).append(
                            (op_index, ax, ay, bx, by))

    for op_index in op_indices:
        for line in ir_geometry.get(op_index) or ():
            for px, py in line:
                gx0, gy0 = int(px // cell), int(py // cell)
                for gx in range(gx0 - 1, gx0 + 2):
                    for gy in range(gy0 - 1, gy0 + 2):
                        for other, ax, ay, bx, by in grid.get((gx, gy), ()):
                            if other == op_index or find(other) == find(op_index):
                                continue
                            if _point_to_segment((py, px), (ay, ax), (by, bx)) <= tau:
                                union(op_index, other)

    runs = {}
    for op_index in op_indices:
        runs.setdefault(find(op_index), []).append(op_index)
    # 走线编号按「op 最多」排序，稳定且与输入顺序无关。
    ordered = sorted(runs.values(), key=lambda v: (-len(v), min(v)))
    return {op: index for index, ops in enumerate(ordered, 1) for op in ops}


def _nearest_point_on(point, lines, to_page_frame):
    """IR 帧里 point 到这些折线的最近落点，转成页面帧 [y, x]。可视化用。"""
    best = None
    py, px = point
    for line in lines:
        for (ax, ay), (bx, by) in zip(line, line[1:]):
            dy, dx = by - ay, bx - ax
            den = dy * dy + dx * dx
            if den <= 1e-12:
                qy, qx = ay, ax
            else:
                t = ((py - ay) * dy + (px - ax) * dx) / den
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                qy, qx = ay + t * dy, ax + t * dx
            d = math.hypot(py - qy, px - qx)
            if best is None or d < best[0]:
                best = (d, qx, qy)
    if best is None:
        return None
    return to_page_frame(best[1], best[2])


def main():
    try:
        job = json.load(sys.stdin)
    except Exception as error:                                  # noqa: BLE001
        _fail("BAD_JOB", f"cannot parse job JSON: {error}")

    pdf = str(job.get("pdf") or "")
    sheet = job.get("sheet")
    if not pdf or not Path(pdf).is_file():
        _fail("BAD_JOB", f"pdf not found: {pdf!r}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        _fail("BAD_JOB", f"sheet must be a 1-based int, got {sheet!r}")
    targets = job.get("targets") or []
    top_k = int(job.get("top_k") or TOP_K_DEFAULT)
    cpu_budget = int(job.get("cpu_budget") or 1)

    if not _ENGINE.is_dir():
        _fail("ENGINE_MISSING", f"engine source not found at {_ENGINE}")
    sys.path.insert(0, str(_ENGINE))

    try:
        import pymupdf
        import pypdf
        from line_type_engine.clustering_api import (
            CLUSTERING_API_SCHEMA_VERSION, project_line_type_clusters)
        from line_type_engine.engine import recognize_page
        from line_type_engine.source_page_adapter import (
            SOURCE_PAGE_ADAPTER_VERSION, source_aligned_page_ir_from_pdf_path)
        from line_type_engine.versions import (
            PAGE_IR_VERSION, PYTHON_ENGINE_VERSION)
    except Exception as error:                                  # noqa: BLE001
        _fail("IMPORT_ERROR", f"{type(error).__name__}: {error}")
    try:
        import scipy
        scipy_version = scipy.__version__
    except Exception:                                           # noqa: BLE001
        # 不是致命的（unknown_pattern_split 有纯 Python 回退），但**回退不是
        # 逐位等价**：_delaunay_edges 的等价代价平票能让一整个线型出现或消失。
        # 所以它必须进缓存签名，缺了也要显式记下来。
        scipy_version = None

    try:
        started = time.time()
        aligned = source_aligned_page_ir_from_pdf_path(pdf, sheet)
        page = aligned.page
        seconds_ir = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("PAGE_IR_ERROR", f"{type(error).__name__}: {error}")

    engine_stamp = {
        "page_ir": PAGE_IR_VERSION,
        "source_page_adapter": SOURCE_PAGE_ADAPTER_VERSION,
        "clustering_schema": CLUSTERING_API_SCHEMA_VERSION,
        "engine": PYTHON_ENGINE_VERSION,
        "producer": page.producer,
        "pymupdf": pymupdf.__version__,
        "pypdf": pypdf.__version__,
        "scipy": scipy_version,
        "cpu_budget": cpu_budget,
    }

    # 转帧的量取自**另一个**新开的文档：引擎的抽取路径会临时改 /Rotate 且异常时
    # 不恢复，所以绝不能复用它碰过的那个 page 对象。
    try:
        document = pymupdf.open(pdf)
        try:
            fitz_page = document[sheet - 1]
            rotation_matrix = fitz_page.rotation_matrix
            rotated_width = float(fitz_page.rect.width)
            rotated_height = float(fitz_page.rect.height)
            fitz_rotation = int(fitz_page.rotation)
        finally:
            document.close()
    except Exception as error:                                  # noqa: BLE001
        _fail("PAGE_FRAME_ERROR", f"{type(error).__name__}: {error}")
    if rotated_width <= 0 or rotated_height <= 0:
        _fail("PAGE_FRAME_ERROR", "page rect has non-positive extent")
    if fitz_rotation != page.rotation_degrees:
        _fail("PAGE_FRAME_ERROR",
              f"rotation disagrees: fitz {fitz_rotation} vs IR "
              f"{page.rotation_degrees} — refusing to guess the frame")

    bounds = page.page_bounds
    unrotated_height = bounds.max_y - bounds.min_y
    unrotated_width = bounds.max_x - bounds.min_x
    # 护栏：IR 的未旋转 CropBox 尺寸必须与 fitz 旋转后的 rect 对得上（90/270 交换
    # 两轴）。对不上说明 CropBox 越界或 UserUnit != 1 —— 那时整个转帧都是错的，
    # 宁可拒算也不要画一页看着合理的错几何。
    expected = ((rotated_height, rotated_width)
                if page.rotation_degrees % 180 else (rotated_width, rotated_height))
    if (abs(unrotated_width - expected[0]) > 1e-6
            or abs(unrotated_height - expected[1]) > 1e-6):
        _fail("PAGE_FRAME_ERROR",
              f"page_bounds {unrotated_width}x{unrotated_height} does not match "
              f"rotated rect {rotated_width}x{rotated_height} "
              f"(rot={page.rotation_degrees}) — CropBox offset or UserUnit != 1")

    def to_page_frame(x, y):
        point = pymupdf.Point(x + bounds.min_x,
                              unrotated_height - y + bounds.min_y) * rotation_matrix
        return [round(point.y / rotated_height * 1000, 2),
                round(point.x / rotated_width * 1000, 2)]

    inverse_rotation = ~rotation_matrix

    def to_ir_frame(frame_y, frame_x):
        """0-1000 页帧 [y,x] → IR 帧 (x, y)（PDF 点）。to_page_frame 的逆。"""
        point = pymupdf.Point(frame_x / 1000.0 * rotated_width,
                              frame_y / 1000.0 * rotated_height) * inverse_rotation
        return (point.x - bounds.min_x,
                unrotated_height - (point.y - bounds.min_y))

    # 一次拿三份输出。**fused 不是全部** —— fusion 按 global 类型整体裁决，
    # method2 只要赢下某个成员组，整个跨组类型就被溶解，连它在别的组里、
    # method2 从未碰过的部分一起消失。实测 gladstone P4：method1 认领 1772 op、
    # method2 认领 186（且是 method1 的真子集）、fused 只剩 897 —— 875 个**没有
    # 竞争者**的 op 凭空没了，其中 group 57 那条 58-op 的线 method2 一个 op 都
    # 没碰过。这 875 个里 99.1% 距离任何 method2 op 超过 36 pt（中位 517.7 pt），
    # 是图上完全不同的部位，不是"同一段线的边角"。
    # 所以：身份仍以 fused 为准（重叠judgement 归 method2，更具体的描述赢），
    # 但把「method1 认领 / method2 从未认领 / fused 却无主」的部分补回来。
    try:
        started = time.time()
        recognition = recognize_page(
            page, outputs=("method1", "method2", "fused"),
            method1_worker_count=cpu_budget, parallel_methods=True)
        if recognition.fused is None:
            raise RuntimeError("fused recognition did not produce a result")
        result = project_line_type_clusters(page, recognition.fused.result)
        seconds_cluster = time.time() - started
    except Exception as error:                                  # noqa: BLE001
        _fail("CLUSTER_ERROR", f"{type(error).__name__}: {error}")

    payload = result.to_dict()
    operations = page.operations

    def op_ir_lines(operation):
        """一条 op → **IR 帧**折线列表（(x, y) 元组，按 move 断开，close 回首点）。

        曲线按控制点折线化：高亮只要画得出形状，而引擎自己的判据也是在这批点上
        算的。绝不能整段丢掉 curve —— 虚线 / motif 线型经常是弧。
        """
        lines, current = [], []
        for segment in getattr(operation, "segments", ()) or ():
            if segment.kind == "move":
                if len(current) > 1:
                    lines.append(current)
                current = [tuple(segment.end)]
            elif segment.kind == "line":
                current.append(tuple(segment.end))
            elif segment.kind == "curve":
                current.append(tuple(segment.control_1))
                current.append(tuple(segment.control_2))
                current.append(tuple(segment.end))
            elif segment.kind == "close" and len(current) > 1:
                current.append(current[0])
        if len(current) > 1:
            lines.append(current)
        return lines

    # op_index → 它所属的引擎 group（引擎自己的结构划分，一页上百个）。
    # 这是限制高亮范围的关键粒度：一个 global 线型是多个 group 的局部簇按签名
    # 相似度**跨组合并**出来的（lenexa P4 的 #5 由 group 50/41/73/74 四块并成，
    # min_sim 0.9592，四块空间上完全分开），而 callout 的末端只落在其中一块。
    # 照 global 全画就会把另外三块（右上一大堆、左边两条带）一起点亮。
    group_of = {}
    for grp in payload["groups"]:
        gid = str(grp["group_id"])
        for local in grp["line_types"]:
            for op_index in local["commands"]["op_indices"]:
                group_of[op_index] = gid
        for op_index in grp["residual_vector_commands"]["op_indices"]:
            group_of.setdefault(op_index, gid)

    # op_index → 拥有它的全局线型编号（一个 op 最多属于一个全局簇）
    owner = {}
    for cluster in payload["global_line_types"]:
        number = int(cluster["line_type_number"])
        for op_index in cluster["commands"]["op_indices"]:
            if not 0 <= op_index < len(operations):
                _fail("OP_INDEX_ERROR",
                      f"op_index {op_index} out of range (ops={len(operations)})")
            owner[op_index] = number

    # ---- 补回被 fusion 连带丢弃的 method1 覆盖 ----------------------------
    # 判据是确定性的、不含任何阈值：**method1 认领了、method2 从未认领、而
    # fused 里无主** 的 op。method2 碰过的一律不补（那是它按策略赢下的），
    # fused 里已有主的也不补。补回来的类型单独编号、单独标记，前端能分辨。
    def _global_types(envelope):
        if envelope is None:
            return ()
        return getattr(envelope.result, "global_types", ()) or ()

    method2_ops = set()
    for cluster in _global_types(recognition.method2):
        method2_ops.update(cluster.op_indices)

    recovered = []
    next_number = max([int(c["line_type_number"])
                       for c in payload["global_line_types"]] or [0])
    for cluster in _global_types(recognition.method1):
        keep = [i for i in cluster.op_indices
                if i not in owner and i not in method2_ops
                and 0 <= i < len(operations)]
        if not keep:
            continue
        next_number += 1
        for op_index in keep:
            owner[op_index] = next_number
        recovered.append({
            "line_type_number": next_number,
            "source_global_type_id": cluster.global_type_id,
            "signature_family": cluster.signature_family,
            "minimum_pair_similarity": cluster.minimum_pair_similarity,
            "members": [{"case_id": m.case_id, "type_id": m.type_id,
                         "atom_count": m.atom_count} for m in cluster.members],
            "op_indices": keep,
            "op_count_in_method1": len(cluster.op_indices),
        })

    # 只有 path op 有 segments；文字 / 图片 op 自然得到空列表。
    ir_geometry = {}
    for op_index, operation in enumerate(operations):
        lines = op_ir_lines(operation)
        if lines:
            ir_geometry[op_index] = lines

    def page_lines_of(op_index):
        return [[to_page_frame(x, y) for x, y in line]
                for line in ir_geometry.get(op_index) or ()]

    def clipped(op_index):
        """这条 op 的 segments 是否伸出了它自己的 bounds —— 即有一部分被裁掉了.

        source_content 里 bounds 是和当前 clip 求过交的（source_content.py:1147-
        1153：painted_bounds 先按 max(line_width/2, 0.25) 外扩，再与 clip 求交；
        完全被裁掉的 paint 根本不产出 operation，这就是 aligned 3563 op 比
        display-list 3595 少的原因），而 segments **没有**被裁。所以在被裁的页上，
        照 segments 全画会在 PDF 上什么都没有的地方描出线来。这里只做计数并上报，
        不静默修改几何 —— 让它可见，比悄悄裁一刀好。
        """
        operation = operations[op_index]
        bounds = getattr(operation, "bounds", None)
        lines = ir_geometry.get(op_index) or ()
        if bounds is None or not lines:
            return False
        slack = max(float(getattr(operation, "line_width", 0.0) or 0.0) / 2.0,
                    0.25) + 1e-6
        for line in lines:
            for x, y in line:
                if (x < bounds.min_x - slack or x > bounds.max_x + slack
                        or y < bounds.min_y - slack or y > bounds.max_y + slack):
                    return True
        return False

    ops_by_cluster = {}
    for op_index, number in owner.items():
        ops_by_cluster.setdefault(number, []).append(op_index)

    # 每个线型切成若干条**连通走线**。高亮的单位是走线，不是引擎的 group ——
    # 详见 _connected_runs 的说明。只对候选线型算（其余不发几何，白算）。
    run_of = {}
    for number, ops in ops_by_cluster.items():
        for op_index, run_id in _connected_runs(sorted(ops), ir_geometry).items():
            run_of[op_index] = run_id

    ir_by_cluster = {}
    ir_by_cluster_run = {}
    clipped_by_cluster = {}
    for op_index, number in owner.items():
        lines = ir_geometry.get(op_index) or ()
        ir_by_cluster.setdefault(number, []).extend(lines)
        rid = str(run_of.get(op_index, 1))
        ir_by_cluster_run.setdefault(number, {}).setdefault(rid, []).extend(lines)
        if clipped(op_index):
            clipped_by_cluster[number] = clipped_by_cluster.get(number, 0) + 1

    meta = {}
    for cluster in payload["global_line_types"]:
        number = int(cluster["line_type_number"])
        lines = ir_by_cluster.get(number) or []
        page_lines = [[to_page_frame(x, y) for x, y in line] for line in lines]
        meta[number] = {
            "line_type_number": number,
            "line_type_id": cluster["line_type_id"],
            "type_uid": cluster["type_uid"],
            "signature_family": cluster["signature_family"],
            "recognition_source": cluster["recognition_source"],
            "minimum_pair_similarity": cluster["minimum_pair_similarity"],
            "member_count": len(cluster["members"]),
            "op_count": len(cluster["commands"]["op_indices"]),
            "range_count": len(cluster["commands"]["ranges"]),
            "segment_count": sum(max(0, len(line) - 1) for line in page_lines),
            # 这个类型拥有的 op 集合的指纹。**这才是「两边聚类结果相同」的判据** ——
            # member_count 不能用：TS 的 members 是「每个涉及的 group 一条」
            # （文字型线型上实测 19 个 op 对应 79 个 member，atom_count 全为 0），
            # 而 Python 的投影只列真正拥有矢量 op 的 group（同页 11 个）。
            # 那是投影口径差异，与 op 归属无关；比 op 集合才不会被它误导。
            "ops_sha1": hashlib.sha1(",".join(
                str(i) for i in sorted(cluster["commands"]["op_indices"])
            ).encode()).hexdigest(),
            # 框是把每条折线的**全部点**映射后取 min/max 得到的。y-flip 在所有
            # rotation 下都会颠倒 min/max 归属，90/270 还会交换轴，只映两个对角
            # 会得到「同两点、错装配」的框。
            "bbox": _bbox(page_lines),
            # >0 说明这个簇里有 op 的 segments 伸出了被 clip 裁过的 bounds，
            # 全画会在图上空白处描线。上报而不是静默裁掉。
            "clipped_ops": clipped_by_cluster.get(number, 0),
        }

    for row in recovered:
        number = row["line_type_number"]
        lines = ir_by_cluster.get(number) or []
        page_lines = [[to_page_frame(x, y) for x, y in line] for line in lines]
        meta[number] = {
            "line_type_number": number,
            "line_type_id": row["source_global_type_id"],
            # 补回的类型没有 fusion 赋予的 type_uid（那是 fused 阶段才有的），
            # 用 method1 的页内 id 合成一个，前缀标明来源。
            "type_uid": "m1:" + row["source_global_type_id"],
            "signature_family": row["signature_family"],
            "recognition_source": "method1",
            "minimum_pair_similarity": row["minimum_pair_similarity"],
            "member_count": len(row["members"]),
            "op_count": len(row["op_indices"]),
            "range_count": len(row["op_indices"]),
            "segment_count": sum(max(0, len(line) - 1) for line in page_lines),
            "ops_sha1": hashlib.sha1(",".join(
                str(i) for i in sorted(row["op_indices"])).encode()).hexdigest(),
            "bbox": _bbox(page_lines),
            "clipped_ops": clipped_by_cluster.get(number, 0),
            # 这一层是"被 fusion 连带丢弃后补回来的"，必须能分辨。
            "recovered_from_fusion": True,
            "op_count_in_method1": row["op_count_in_method1"],
        }

    bindings = []
    candidates = set()
    for target in targets:
        tip = target.get("tip")
        if not (isinstance(tip, (list, tuple)) and len(tip) >= 2):
            continue
        frame_tip = [float(tip[0]), float(tip[1])]
        ir_tip = to_ir_frame(frame_tip[0], frame_tip[1])
        own_lines = [[to_ir_frame(p[0], p[1]) for p in line]
                     for line in (target.get("own") or ())
                     if isinstance(line, (list, tuple)) and len(line) >= 2]

        # 先剔掉这个 callout 自己的引线 / 箭头笔画，再找离 tip 最近的 op。
        # 同时单独记下最近的**有主** op：tip 自身的量化精度就有 1 pt 量级
        # （arrows.json 里的 tip 是页帧整数），所以「最近的 op 是 residual、但
        # 有一条有主的 op 就在它旁边、差距小于 tip 精度」时，二者在输入分辨率
        # 之下无法区分，上层应该采信有主的那个。这不是调参，是输入精度决定的。
        nearest = None
        nearest_owned = None
        own_ops = 0
        for op_index, lines in ir_geometry.items():
            if _is_own_geometry(lines, own_lines):
                own_ops += 1
                continue
            distance = _polylines_distance(ir_tip, lines)
            if nearest is None or distance < nearest[0]:
                nearest = (distance, op_index)
            if op_index in owner and (nearest_owned is None
                                      or distance < nearest_owned[0]):
                nearest_owned = (distance, op_index)

        # 候选簇的距离（同样排除自己的几何），作审计与投票改判用。
        ranked = []
        for number, lines in ir_by_cluster.items():
            usable = [line for line in lines
                      if not _is_own_geometry([line], own_lines)]
            if not usable:
                continue
            ranked.append((_polylines_distance(ir_tip, usable), number))
        ranked.sort()
        head = ranked[:max(1, top_k)]
        candidates.update(number for _distance, number in head)

        row = {
            "key": target.get("key"),
            "ti": target.get("ti"),
            "tip": [round(frame_tip[0], 2), round(frame_tip[1], 2)],
            "own_ops": own_ops,
            # 单位是 **PDF 点**（等向），不是页帧千分比。
            "ranked": [{"line_type_number": number,
                        "distance": round(distance, 3)}
                       for distance, number in head],
        }
        if nearest is None:
            row["nearest_op"] = None
        else:
            distance, op_index = nearest
            row["nearest_op"] = {
                "op_index": op_index,
                "distance": round(distance, 3),
                # 高亮按 run 裁（连通走线）；group_id 只作审计。
                "run_id": str(run_of.get(op_index)) if op_index in run_of else None,
                "group_id": group_of.get(op_index),
                # 末端在这条 op 上的最近点，页面帧 —— 前端画「末端→线」的
                # 那一小段，让人直接看出判据用的是哪个距离。
                "nearest_point": _nearest_point_on(ir_tip, ir_geometry.get(op_index) or (),
                                                   to_page_frame),
                # None = tip 底下那段 ink 是 residual，不属于任何线型。
                # 这不是"没找到"，是"这里确实没有线型"。
                "owner": owner.get(op_index),
            }
            if owner.get(op_index) is not None:
                candidates.add(owner[op_index])
        if nearest_owned is None:
            row["nearest_owned_op"] = None
        else:
            distance, op_index = nearest_owned
            row["nearest_owned_op"] = {
                "op_index": op_index,
                "distance": round(distance, 3),
                "run_id": str(run_of.get(op_index)) if op_index in run_of else None,
                "group_id": group_of.get(op_index),
                "nearest_point": _nearest_point_on(ir_tip, ir_geometry.get(op_index) or (),
                                                   to_page_frame),
                "owner": owner[op_index],
            }
            candidates.add(owner[op_index])
        bindings.append(row)

    run_counts = {}
    for op_index, number in owner.items():
        key = (number, str(run_of.get(op_index, 1)))
        run_counts[key] = run_counts.get(key, 0) + 1

    line_types = []
    for number in sorted(candidates):
        entry = dict(meta[number])
        by_run = {}
        for rid, lines in (ir_by_cluster_run.get(number) or {}).items():
            polylines = [[to_page_frame(x, y) for x, y in line] for line in lines]
            by_run[rid] = {
                "run_id": rid,
                "op_count": run_counts.get((number, rid), 0),
                "segment_count": sum(max(0, len(line) - 1) for line in polylines),
                "bbox": _bbox(polylines),
                "polylines": polylines,
            }
        # **按连通走线分桶交付**：调用方只画「被这个 callout 的末端指到的走线」，
        # 同型但没接上的另一条走线留在这里供审计（有数量和 bbox），不画。
        entry["by_run"] = by_run
        line_types.append(entry)

    residual_ops = sum(len(group["residual_vector_commands"]["op_indices"])
                       for group in payload["groups"])
    audit = aligned.audit
    json.dump({
        "ok": True,
        "engine": engine_stamp,
        "page": {
            "sheet": sheet,
            "ops": payload["operation_count"],
            "path_ops": len(ir_geometry),
            "owned_path_ops": sum(1 for i in ir_geometry if i in owner),
            "rotation": page.rotation_degrees,
            "groups": payload["group_count"],
            "line_types": len(payload["global_line_types"]) + len(recovered),
            "residual_ops": residual_ops,
            "page_fingerprint": payload["page_fingerprint"],
            # tip 的量化精度（PDF 点）。arrows.json 的 tip 是 0-1000 页帧整数，
            # 半个单位就是这么多点；比这更小的距离差在输入分辨率之下。
            "tip_precision_pt": round(
                0.5 / 1000.0 * max(rotated_width, rotated_height), 3),
            "seconds_ir": round(seconds_ir, 2),
            "seconds_cluster": round(seconds_cluster, 2),
            "errors": payload["errors"],
            "clipped_ops": sum(clipped_by_cluster.values()),
            # 被任何线型认领的 op 集合的指纹。用它就能和 TS 侧
            # (method1 ∪ method2) 的同一指纹逐集合比对，而不是只比数量 ——
            # "不漏" 这件事必须能被机器判定，不能靠人核对计数。
            "owned_ops_sha1": hashlib.sha1(
                ",".join(str(i) for i in sorted(owner)).encode()).hexdigest(),
            "fused_ops_sha1": hashlib.sha1(
                ",".join(str(i) for i in sorted(
                    i for i, n in owner.items()
                    if n <= len(payload["global_line_types"]))).encode()).hexdigest(),
            "fused_line_types": len(payload["global_line_types"]),
            "recovered_line_types": len(recovered),
            "recovered_ops": sum(len(r["op_indices"]) for r in recovered),
            "method2_ops": len(method2_ops),
            "source_provenance_exact": bool(
                getattr(audit, "source_provenance_exact", True)),
        },
        "line_types": line_types,
        "all_line_types": [meta[number] for number in sorted(meta)],
        "bindings": bindings,
    }, sys.stdout, ensure_ascii=False)
    sys.stdout.flush()


if __name__ == "__main__":
    # 必须的 guard：cpu_budget>=2 时引擎会开 spawn 进程池，子进程重新 import
    # 本模块。没有这一层就是 _check_not_importing_main 死锁。
    main()
