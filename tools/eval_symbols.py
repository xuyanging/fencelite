"""步骤2（图例样例符号）的离线评测 —— 改动前后的对照证据，零 Gemini 调用.

为什么要这个工具：步骤2 现在是「整页一次推理」，实测有四类病：
  1. 漏检 —— 图例行的样例根本没找到（legend 文字 unpaired）；
  2. 框不准 —— line 样例只框住一小截（框里几乎没有完整线段）；
  3. 文字框吃掉 marker —— symbol 框和它主人的文字框几乎重合；
  4. 重复 —— 同一个框被配给两个 text_index（真正的图例行 + 裸编码行）；
  5. 平面图里的 marker 混进来 —— symbol 落在 view 区而不是图例区。
这五条全部可以**纯本地**量化：文字项、symbol、group 都在已付费的缓存 JSON 里，
真实矢量几何用 steps.text.markers.strip_context 免费拿。所以验收不必再花一分钱，
也不必看图：跑一遍出数字，改完再跑一遍 --diff 看哪些数字变好、哪些变坏。

**本工具从不调用模型，也从不写任何缓存**（只读 results.json / symbols.json + PDF）。

用法：
    python -B tools/eval_symbols.py <slug> [<slug> ...] [--from DIR]
                                    [--pages 2,5,163] [--json OUT] [--no-fit]
    python -B tools/eval_symbols.py --diff BEFORE.json AFTER.json
    python -B tools/eval_symbols.py --from DIR            # 列出可评的 slug

--from 给的是**数据根**，三种布局自动识别（只是目录名不同）：
    <ROOT>/data/<slug>/*.json        + <ROOT>/projects/<slug>/input.pdf   本项目
    <ROOT>/fence_fused/<slug>/*.json + <ROOT>/projects/<slug>/input.pdf   5051 / fence_detector
    <ROOT>/<slug>/*.json             + (同级或上级 projects/<slug>/input.pdf) 裸缓存副本
最后一种同时覆盖「--from 直接指到 ...\\fence_fused」和「只拷了几个 JSON 的诊断副本」。
PDF 缺失不是错误，只是 box_fit 那一栏记 None（其余指标照算）。

指标口径（全部页面帧 0-1000，[ymin,xmin,ymax,xmax]）：
  legend_texts   item 中心按**最小面积归属**落进 legend/schedule/note_cluster 组的数量
  paired/unpaired 这些 item 里有/没有 symbol 认它当 text_index 的数量
  dup_boxes      同一个量化框被配给 >1 个 text_index 的框数（重复症状）
  sym_in_text    symbol 框与其 owner 文字框 inter/min(area) > 0.5 的个数（吃掉 marker）
  sym_outside_legend  symbol 框中心不在任何图例类组内的个数（平面图 marker）
  box_fit.shape  与 symbol 框重叠最大的 mbox 的 inter/min(area)
  box_fit.line   symbol 框内**完整包住**的线段条数与它们的 x 跨度 / 框宽
                 —— 框只切住样例的一小截时，线段两端都被切在框外，跨度比塌到接近 0，
                    这正是「只框住一小段」的确定性信号
所有比值都做页级中位数与项目级中位数（个数少时中位数比均值稳）。
"""
import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    # 支持 `python -B tools/eval_symbols.py`：此时 sys.path[0] 是 tools/。
    sys.path.insert(0, str(BASE_DIR))

from steps.legend_sweep import _is_marker_code
from steps.prompts import SYMBOL_GROUP_KINDS
from steps.store import items_of, load_json

REPORT_VERSION = 1

# 组内几何判定的容差，与 steps.symbols.symbol_in_allowed_group 的几何兜底一致
# （±2/1000 页宽）—— 评测口径必须和被评测的闸门同一把尺子。
GROUP_TOL = 2.0
# symbol 框「被文字框吃掉」的判定阈值：inter/min(area)。
IN_TEXT_T = 0.5
# 线段「完整落在 symbol 框内」的端点容差（0-1000 帧，0.5 = 页宽的 0.05%）。
SEG_TOL = 0.5
# 框量化粒度：1/1000 页宽以下的差别没有物理意义，量化后才谈得上「同一个框」。
BOX_Q = 1.0
# 文字预览长度（unpaired 明细用）。
TEXT_PREVIEW = 60


# ------------------------------------------------------------------ 布局解析

def resolve_project(root, slug):
    """(缓存目录, PDF 路径或 None, 布局名)；三种布局都只是目录名不同。"""
    root = Path(root)
    for layout, cache_dir, pdf_roots in (
            ("data", root / "data" / slug, (root,)),
            ("fence_fused", root / "fence_fused" / slug, (root,)),
            ("flat", root / slug, (root, root.parent))):
        if not (cache_dir / "results.json").exists():
            continue
        pdf = None
        for pdf_root in pdf_roots:
            candidate = pdf_root / "projects" / slug / "input.pdf"
            if candidate.exists():
                pdf = candidate
                break
        return cache_dir, pdf, layout
    return None, None, None


def available_slugs(root):
    """数据根下算过东西（有 results.json）的 slug，去重按名字排序。"""
    root = Path(root)
    found = set()
    for parent in (root / "data", root / "fence_fused", root):
        if not parent.is_dir():
            continue
        try:
            children = list(parent.iterdir())
        except OSError:
            continue
        for child in children:
            if child.is_dir() and (child / "results.json").exists():
                found.add(child.name)
    return sorted(found)


# ------------------------------------------------------------------ 几何小工具

def _valid_box(box):
    return isinstance(box, (list, tuple)) and len(box) == 4 \
        and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in box)


def _norm_box(box):
    """[ymin,xmin,ymax,xmax]，顺序被写反时就地纠正（只为算面积，不改数据）。"""
    y0, x0, y1, x1 = (float(v) for v in box)
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 < x0:
        x0, x1 = x1, x0
    return [y0, x0, y1, x1]


def _area(box):
    y0, x0, y1, x1 = _norm_box(box)
    return max(0.0, y1 - y0) * max(0.0, x1 - x0)


def _inter_area(a, b):
    ay0, ax0, ay1, ax1 = _norm_box(a)
    by0, bx0, by1, bx1 = _norm_box(b)
    dy = min(ay1, by1) - max(ay0, by0)
    dx = min(ax1, bx1) - max(ax0, bx0)
    if dy <= 0 or dx <= 0:
        return 0.0
    return dy * dx


def overlap_min(a, b):
    """inter / min(area)：小框被大框包住时为 1，与谁大谁小无关。"""
    inter = _inter_area(a, b)
    if inter <= 0:
        return 0.0
    smallest = min(_area(a), _area(b))
    if smallest <= 0:
        return 0.0
    return inter / smallest


def _center(box):
    y0, x0, y1, x1 = _norm_box(box)
    return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)


def center_in(box, container, tol=GROUP_TOL):
    """框中心是否落在 container 里（±tol），与硬闸的几何兜底同口径。"""
    cy, cx = _center(box)
    y0, x0, y1, x1 = _norm_box(container)
    return y0 - tol <= cy <= y1 + tol and x0 - tol <= cx <= x1 + tol


def owning_group(box, groups, tol=GROUP_TOL):
    """最小面积归属：中心落在多个组里时，取面积最小的那个（最贴身的语境）。"""
    best_index = None
    best_area = None
    for index, group in enumerate(groups or []):
        if not isinstance(group, dict):
            continue
        gbox = group.get("box_2d")
        if not _valid_box(gbox) or not center_in(box, gbox, tol):
            continue
        area = _area(gbox)
        if best_area is None or area < best_area:
            best_index, best_area = index, area
    return best_index


def _qbox(box, q=BOX_Q):
    return tuple(round(float(v) / q) * q for v in _norm_box(box))


def _median(values):
    values = [v for v in values if v is not None]
    return round(statistics.median(values), 4) if values else None


def _preview(text):
    flat = " ".join(str(text or "").split())
    return flat[:TEXT_PREVIEW]


# ------------------------------------------------------------------ 矢量贴合度

def _seg_inside(seg, box, tol=SEG_TOL):
    """线段两端都在框内 —— 被框切断的线段**不算**，这正是要抓的症状。"""
    y0, x0, y1, x1 = _norm_box(box)
    return all(x0 - tol <= x <= x1 + tol and y0 - tol <= y <= y1 + tol
               for x, y in ((seg["ax"], seg["ay"]), (seg["bx"], seg["by"])))


def line_fit(box, segs):
    """line 样例的贴合度：框内完整线段条数 + 它们的 x 跨度 / 框宽。

    跨度比 ≈ 1  → 框刚好裹住整段样例；
    跨度比 ≪ 1 → 框里只剩零碎（或什么都没有），样例被切掉了大半。
    另外给 row_cover = 框宽 / 同一行线段的总跨度：<1 说明这一行的线在框外还
    延伸很长（框太短），>1 说明框比样例宽。这一项只作参考，不参与回归判定。
    """
    if not segs:
        return None
    width = max(_norm_box(box)[3] - _norm_box(box)[1], 1e-9)
    inside = [s for s in segs if _seg_inside(s, box)]
    span_ratio = None
    if inside:
        xs = [v for s in inside for v in (s["ax"], s["bx"])]
        span_ratio = round((max(xs) - min(xs)) / width, 4)
    else:
        span_ratio = 0.0
    y0, x0, y1, x1 = _norm_box(box)
    tol_y = max(GROUP_TOL, (y1 - y0) / 2.0)
    row = [s for s in segs
           if y0 - tol_y <= min(s["ay"], s["by"])
           and max(s["ay"], s["by"]) <= y1 + tol_y
           and max(s["ax"], s["bx"]) >= x0 and min(s["ax"], s["bx"]) <= x1]
    row_cover = None
    if row:
        xs = [v for s in row for v in (s["ax"], s["bx"])]
        span = max(xs) - min(xs)
        if span > 0:
            row_cover = round(width / span, 4)
    return {"segs": len(inside), "span_ratio": span_ratio,
            "row_segs": len(row), "row_cover": row_cover}


def shape_fit(box, mboxes):
    """shape 样例的贴合度：与它重叠最大的 mbox 的 inter/min(area)。

    页面没有任何 mbox（无矢量层 / 纯扫描页）→ None（测不了，不是 0 分）；
    有 mbox 但一个都不沾 → 0.0（框落在了没有闭合小图形的地方）。
    """
    if not mboxes:
        return None
    best = 0.0
    for mbox in mboxes:
        if not _valid_box(mbox):
            continue
        best = max(best, overlap_min(box, mbox))
    return round(best, 4)


# ------------------------------------------------------------------ 单页评测

def eval_page(items, symbols, groups, ctx=None):
    """一页的全部指标 —— 纯函数，不碰磁盘（离线测试直接喂合成数据）。

    items   步骤1 的 union item 列表（steps.store.items_of 的顺序即 text_index）
    symbols 步骤2 **发布后**的 symbol 列表（用户真正看到的那批）
    groups  同一页的 groups
    ctx     steps.text.markers.strip_context 的返回值，None = 没有 PDF，
            box_fit 整栏记 None（其余指标照算）
    """
    items = list(items or [])
    symbols = [s for s in (symbols or []) if isinstance(s, dict)]
    groups = [g for g in (groups or []) if isinstance(g, dict)]

    group_kinds = {}
    for group in groups:
        kind = group.get("kind")
        group_kinds[kind] = group_kinds.get(kind, 0) + 1
    legend_boxes = [g["box_2d"] for g in groups
                    if g.get("kind") in SYMBOL_GROUP_KINDS
                    and _valid_box(g.get("box_2d"))]

    # 1) legend 文字：中心最小面积归属到图例类组。
    # 纯 marker 编码的行（"4CL" 这种漏进文字层的）不算图例**描述行** —— 它自己
    # 就是那个 marker，样例的归属在旁边那条完整描述上。口径必须与
    # steps.legend_sweep 一致，否则"未配到"会把这些行算成漏检，
    # 让一个正确的修正在 diff 里显示成恶化。
    legend_idx = []
    for index, item in enumerate(items):
        box = item.get("box_2d")
        if not _valid_box(box):
            continue
        if _is_marker_code(item.get("text")):
            continue
        owner = owning_group(box, groups)
        if owner is not None and groups[owner].get("kind") in SYMBOL_GROUP_KINDS:
            legend_idx.append(index)

    owners = set()
    for symbol in symbols:
        ti = symbol.get("text_index")
        if isinstance(ti, int) and not isinstance(ti, bool):
            owners.add(ti)
    legend_set = set(legend_idx)
    paired = sorted(legend_set & owners)
    unpaired = sorted(legend_set - owners)

    # 2) 每个 symbol 的诊断行
    rows = []
    dup_map = {}
    sym_line = sym_shape = 0
    in_text = outside = 0
    shape_fits, line_spans, line_covers = [], [], []
    mboxes = (ctx or {}).get("mboxes")
    segs = (ctx or {}).get("segs")
    for index, symbol in enumerate(symbols):
        box = symbol.get("box_2d")
        category = symbol.get("category")
        ti = symbol.get("text_index")
        row = {"i": index, "category": category,
               "value": symbol.get("value", ""), "text_index": ti,
               "box_2d": list(box) if _valid_box(box) else None}
        if category == "line":
            sym_line += 1
        elif category == "shape":
            sym_shape += 1
        if not _valid_box(box):
            row["invalid_box"] = True
            rows.append(row)
            continue

        dup_map.setdefault(_qbox(box), set()).add(ti)

        # 文字框吃掉 marker：和它自己的主人比，不是和随便哪个文字比
        if isinstance(ti, int) and not isinstance(ti, bool) \
                and 0 <= ti < len(items) and _valid_box(items[ti].get("box_2d")):
            ov = round(overlap_min(box, items[ti]["box_2d"]), 4)
            row["owner_overlap"] = ov
            if ov > IN_TEXT_T:
                in_text += 1
                row["in_text"] = True

        # 平面图 marker：中心不在任何图例类组内
        if not any(center_in(box, gbox) for gbox in legend_boxes):
            outside += 1
            row["outside_legend"] = True

        if ctx is not None:
            if category == "shape":
                fit = shape_fit(box, mboxes)
                row["shape_fit"] = fit
                shape_fits.append(fit)
            elif category == "line":
                fit = line_fit(box, segs)
                row["line_fit"] = fit
                if fit is not None:
                    line_spans.append(fit["span_ratio"])
                    line_covers.append(fit["row_cover"])
                else:
                    line_spans.append(None)
                    line_covers.append(None)
        rows.append(row)

    dup_boxes = sum(1 for tis in dup_map.values()
                    if len({t for t in tis if t is not None}) > 1)

    return {
        "text_items": len(items),
        "legend_texts": len(legend_idx),
        "paired": len(paired),
        "unpaired": len(unpaired),
        "unpaired_idx": unpaired,
        "unpaired_items": [{"idx": i, "text": _preview(items[i].get("text"))}
                           for i in unpaired],
        "paired_idx": paired,
        "sym_total": len(symbols),
        "sym_line": sym_line,
        "sym_shape": sym_shape,
        "dup_boxes": dup_boxes,
        "sym_in_text": in_text,
        "sym_outside_legend": outside,
        "groups": len(groups),
        "group_kinds": group_kinds,
        "box_fit": {
            "available": ctx is not None,
            "shape_values": shape_fits,
            "line_span_values": line_spans,
            "line_cover_values": line_covers,
            "shape_median": _median(shape_fits),
            "line_span_median": _median(line_spans),
            "line_cover_median": _median(line_covers),
        },
        "symbols": rows,
    }


# ------------------------------------------------------------------ 项目评测

def _page_numbers(results, pages_filter=None):
    """有 fence 文字的页码（1-based int，稀疏），按数字序。"""
    numbers = []
    for key in (results.get("pages") or {}):
        try:
            numbers.append(int(key))
        except (TypeError, ValueError):
            continue
    numbers.sort()
    if pages_filter:
        wanted = set(pages_filter)
        numbers = [n for n in numbers if n in wanted]
    return numbers


def _page_context(pdf, page_index, need):
    """真实矢量几何；PDF 缺失 / 抽取失败 / 这页没 symbol → None（安静降级）。"""
    if pdf is None or not need:
        return None
    try:
        from steps.text.markers import strip_context
        return strip_context(str(pdf), page_index)
    except Exception as exc:                                    # noqa: BLE001
        # 评测工具绝不能因为一页抽取失败就中断整轮对照。
        return {"mboxes": [], "segs": [], "error": f"{type(exc).__name__}: {exc}"}


def eval_project(slug, root=BASE_DIR, pages_filter=None, use_pdf=True,
                 log=None):
    """一个项目的逐页指标 + 项目汇总。只读缓存与 PDF，零模型调用。"""
    cache_dir, pdf, layout = resolve_project(root, slug)
    if cache_dir is None:
        raise FileNotFoundError(
            f"no results.json for slug '{slug}' under {root}")
    results = load_json(cache_dir / "results.json", None) or {}
    symbols_all = load_json(cache_dir / "symbols.json", None) or {}
    if not use_pdf:
        pdf = None

    pages = {}
    for page in _page_numbers(results, pages_filter):
        rec = (results.get("pages") or {}).get(str(page)) or {}
        items = items_of(rec) if isinstance(rec, dict) else []
        entry = symbols_all.get(str(page))
        has_entry = isinstance(entry, dict) and isinstance(entry.get("result"),
                                                           dict)
        result = entry.get("result") if has_entry else {}
        published = result.get("symbols") or []
        groups = result.get("groups") or []
        raw = (entry or {}).get("raw") if isinstance(entry, dict) else None
        raw_total = len(raw.get("symbols") or []) if isinstance(raw, dict) else None
        ctx = _page_context(pdf, page - 1, bool(published))
        metrics = eval_page(items, published, groups, ctx)
        metrics["page"] = page
        metrics["has_symbol_entry"] = bool(has_entry)
        metrics["raw_total"] = raw_total
        metrics["gate_dropped"] = (None if raw_total is None
                                   else raw_total - metrics["sym_total"])
        if isinstance(ctx, dict) and ctx.get("error"):
            metrics["ctx_error"] = ctx["error"]
        pages[str(page)] = metrics
        if log:
            log(_page_line(metrics))
    return {
        "slug": slug,
        "layout": layout,
        "cache_dir": str(cache_dir),
        "pdf": str(pdf) if pdf else None,
        "page_count": results.get("page_count"),
        "pages": pages,
        "totals": summarize(pages),
    }


SUM_KEYS = ("text_items", "legend_texts", "paired", "unpaired", "sym_total",
            "sym_line", "sym_shape", "dup_boxes", "sym_in_text",
            "sym_outside_legend")


def summarize(pages):
    """把逐页指标加总；比值走全部原始值的中位数（不是中位数的平均）。"""
    totals = {key: 0 for key in SUM_KEYS}
    shape, span, cover = [], [], []
    pages_with_legend = pages_all_paired = pages_no_entry = 0
    for metrics in pages.values():
        for key in SUM_KEYS:
            totals[key] += metrics.get(key) or 0
        fit = metrics.get("box_fit") or {}
        shape += list(fit.get("shape_values") or [])
        span += list(fit.get("line_span_values") or [])
        cover += list(fit.get("line_cover_values") or [])
        if metrics.get("legend_texts"):
            pages_with_legend += 1
            if not metrics.get("unpaired"):
                pages_all_paired += 1
        if not metrics.get("has_symbol_entry"):
            pages_no_entry += 1
    totals.update({
        "pages": len(pages),
        "pages_with_legend_text": pages_with_legend,
        "pages_all_paired": pages_all_paired,
        "pages_without_symbol_entry": pages_no_entry,
        "shape_median": _median(shape),
        "line_span_median": _median(span),
        "line_cover_median": _median(cover),
        "shape_measured": len([v for v in shape if v is not None]),
        "line_measured": len([v for v in span if v is not None]),
    })
    return totals


def eval_root(slugs, root=BASE_DIR, pages_filter=None, use_pdf=True, log=None):
    """多个项目 → 一份可落盘、可 --diff 的报告。"""
    report = {"tool": "eval_symbols", "report_version": REPORT_VERSION,
              "root": str(root),
              "generated": datetime.now().isoformat(timespec="seconds"),
              "projects": {}}
    for slug in slugs:
        if log:
            log("")
        # 逐页表格在下面一次性打印，所以这里不让 eval_project 自己再打一遍。
        project = eval_project(slug, root=root, pages_filter=pages_filter,
                               use_pdf=use_pdf, log=None)
        report["projects"][slug] = project
        if log:
            log(f"== {slug}  [{project['layout']}] {project['cache_dir']}")
            log(f"   pdf: {project['pdf'] or '(缺 —— box_fit 跳过)'}")
            log(_page_header())
            for page in sorted(project["pages"], key=int):
                log(_page_line(project["pages"][page]))
            for page in sorted(project["pages"], key=int):
                metrics = project["pages"][page]
                for bad in metrics["unpaired_items"]:
                    log(f"   P{metrics['page']} unpaired idx={bad['idx']}"
                        f"  {bad['text']!r}")
                if metrics.get("ctx_error"):
                    log(f"   P{metrics['page']} [warn] 矢量抽取失败: "
                        f"{metrics['ctx_error']}")
            log(_totals_line(project["totals"]))
    report["totals"] = _merge_totals(report["projects"])
    if log:
        log("")
        log("==== 全部项目 ====")
        log(_totals_line(report["totals"]))
    return report


def _merge_totals(projects):
    merged = {key: 0 for key in SUM_KEYS}
    shape, span, cover = [], [], []
    for project in projects.values():
        for page in project.get("pages", {}).values():
            for key in SUM_KEYS:
                merged[key] += page.get(key) or 0
            fit = page.get("box_fit") or {}
            shape += list(fit.get("shape_values") or [])
            span += list(fit.get("line_span_values") or [])
            cover += list(fit.get("line_cover_values") or [])
    pages = sum(len(p.get("pages", {})) for p in projects.values())
    merged.update({
        "pages": pages,
        "projects": len(projects),
        "pages_with_legend_text": sum(
            p["totals"]["pages_with_legend_text"] for p in projects.values()),
        "pages_all_paired": sum(
            p["totals"]["pages_all_paired"] for p in projects.values()),
        "pages_without_symbol_entry": sum(
            p["totals"]["pages_without_symbol_entry"] for p in projects.values()),
        "shape_median": _median(shape),
        "line_span_median": _median(span),
        "line_cover_median": _median(cover),
        "shape_measured": len([v for v in shape if v is not None]),
        "line_measured": len([v for v in span if v is not None]),
    })
    return merged


# ------------------------------------------------------------------ 文本报表

def _fmt(value):
    return "-" if value is None else (f"{value:.2f}"
                                      if isinstance(value, float) else str(value))


def _page_header():
    return ("   page | txt lgnd pair unpr | sym(l/s) raw drop | dup inTxt "
            "outLg | shapeFit lineSpan lineCov")


def _page_line(m):
    return ("   {page:>4} | {t:>3} {l:>4} {p:>4} {u:>4} | {s:>3}({sl}/{ss})"
            " {raw:>3} {drop:>4} | {dup:>3} {it:>5} {ol:>5} | {sf:>8} {ls:>8}"
            " {lc:>7}").format(
        page=m.get("page", "?"), t=m["text_items"], l=m["legend_texts"],
        p=m["paired"], u=m["unpaired"], s=m["sym_total"], sl=m["sym_line"],
        ss=m["sym_shape"], raw=_fmt(m.get("raw_total")),
        drop=_fmt(m.get("gate_dropped")), dup=m["dup_boxes"],
        it=m["sym_in_text"], ol=m["sym_outside_legend"],
        sf=_fmt(m["box_fit"]["shape_median"]),
        ls=_fmt(m["box_fit"]["line_span_median"]),
        lc=_fmt(m["box_fit"]["line_cover_median"]))


def _totals_line(t):
    return ("   TOTAL {pages} 页(含图例文字 {wl}, 全配到 {ap}) | 文字 {ti}"
            " 图例文字 {lt} 配到 {p} 未配到 {u} | sym {s}({sl}/{ss})"
            " dup {dup} inTxt {it} outLegend {ol} | shapeFit {sf}"
            " lineSpan {ls} lineCov {lc}").format(
        pages=t["pages"], wl=t["pages_with_legend_text"],
        ap=t["pages_all_paired"], ti=t["text_items"], lt=t["legend_texts"],
        p=t["paired"], u=t["unpaired"], s=t["sym_total"], sl=t["sym_line"],
        ss=t["sym_shape"], dup=t["dup_boxes"], it=t["sym_in_text"],
        ol=t["sym_outside_legend"], sf=_fmt(t["shape_median"]),
        ls=_fmt(t["line_span_median"]), lc=_fmt(t["line_cover_median"]))


# ------------------------------------------------------------------ 对照 diff

# 指标方向：down = 越小越好，up = 越大越好，info = 只报不判。
METRIC_DIRECTION = (
    ("unpaired", "down"),
    ("paired", "up"),
    ("dup_boxes", "down"),
    ("sym_in_text", "down"),
    ("sym_outside_legend", "down"),
    ("sym_total", "info"),
    ("legend_texts", "info"),
)
FIT_DIRECTION = (("shape_median", "up"), ("line_span_median", "up"))
# 中位数的抖动阈值：比这更小的差别不算改善也不算恶化。
FIT_EPS = 0.005


def _classify(direction, before, after, eps=0.0):
    delta = after - before
    if abs(delta) <= eps:
        return "same"
    if direction == "info":
        return "info"
    better = delta < 0 if direction == "down" else delta > 0
    return "better" if better else "worse"


def diff_pages(before, after):
    """两页指标的差异明细：{changes:[...], improved:[idx], regressed:[idx]}。"""
    changes = []
    for key, direction in METRIC_DIRECTION:
        b, a = before.get(key) or 0, after.get(key) or 0
        if b == a:
            continue
        changes.append({"metric": key, "before": b, "after": a,
                        "verdict": _classify(direction, b, a)})
    bfit = before.get("box_fit") or {}
    afit = after.get("box_fit") or {}
    for key, direction in FIT_DIRECTION:
        b, a = bfit.get(key), afit.get(key)
        if b is None or a is None:
            if b != a:
                changes.append({"metric": key, "before": b, "after": a,
                                "verdict": "info"})
            continue
        verdict = _classify(direction, b, a, eps=FIT_EPS)
        if verdict != "same":
            changes.append({"metric": key, "before": b, "after": a,
                            "verdict": verdict})
    b_unpaired = set(before.get("unpaired_idx") or [])
    a_unpaired = set(after.get("unpaired_idx") or [])
    return {"changes": changes,
            "fixed_idx": sorted(b_unpaired - a_unpaired),
            "broken_idx": sorted(a_unpaired - b_unpaired)}


def diff_reports(before, after):
    """两份报告 → 逐页差异 + 恶化清单（恶化是这个工具存在的主要理由）。"""
    out = {"projects": {}, "regressions": [], "improvements": [],
           "pages_added": [], "pages_removed": []}
    slugs = sorted(set(before.get("projects", {}))
                   | set(after.get("projects", {})))
    for slug in slugs:
        bproj = (before.get("projects") or {}).get(slug)
        aproj = (after.get("projects") or {}).get(slug)
        if bproj is None or aproj is None:
            out["projects"][slug] = {
                "missing_in": "before" if bproj is None else "after"}
            continue
        bpages, apages = bproj.get("pages", {}), aproj.get("pages", {})
        pages = {}
        for page in sorted(set(bpages) | set(apages), key=int):
            if page not in bpages:
                out["pages_added"].append(f"{slug}#P{page}")
                continue
            if page not in apages:
                out["pages_removed"].append(f"{slug}#P{page}")
                continue
            entry = diff_pages(bpages[page], apages[page])
            if not entry["changes"] and not entry["fixed_idx"] \
                    and not entry["broken_idx"]:
                continue
            pages[page] = entry
            for change in entry["changes"]:
                tag = (f"{slug} P{page} {change['metric']}: "
                       f"{_fmt(change['before'])} -> {_fmt(change['after'])}")
                if change["verdict"] == "worse":
                    out["regressions"].append(tag)
                elif change["verdict"] == "better":
                    out["improvements"].append(tag)
        out["projects"][slug] = {
            "pages": pages,
            "totals": {"before": bproj.get("totals"),
                       "after": aproj.get("totals")},
        }
    return out


def format_diff(diff):
    lines = []
    for slug, project in diff["projects"].items():
        if project.get("missing_in"):
            lines.append(f"== {slug}: 只在 {'after' if project['missing_in'] == 'before' else 'before'} 里出现")
            continue
        lines.append(f"== {slug}")
        for page, entry in project["pages"].items():
            marks = []
            for change in entry["changes"]:
                flag = {"worse": " [恶化]", "better": " [改善]",
                        "info": "", "same": ""}[change["verdict"]]
                marks.append(f"{change['metric']} {_fmt(change['before'])}"
                             f"->{_fmt(change['after'])}{flag}")
            lines.append(f"   P{page}: " + "; ".join(marks))
            if entry["fixed_idx"]:
                lines.append(f"        配上了 idx {entry['fixed_idx']}")
            if entry["broken_idx"]:
                lines.append(f"        丢失配对 idx {entry['broken_idx']} [恶化]")
        totals = project.get("totals") or {}
        if totals.get("before") and totals.get("after"):
            lines.append("   before " + _totals_line(totals["before"]).strip())
            lines.append("   after  " + _totals_line(totals["after"]).strip())
    for page in diff["pages_added"]:
        lines.append(f"   [新增页] {page}")
    for page in diff["pages_removed"]:
        lines.append(f"   [消失页] {page} [恶化]")
    lines.append("")
    lines.append(f"改善 {len(diff['improvements'])} 项 / "
                 f"恶化 {len(diff['regressions'])} 项")
    if diff["regressions"]:
        lines.append("---- 恶化明细（不要引入新问题）----")
        for tag in diff["regressions"]:
            lines.append(f"   [恶化] {tag}")
    else:
        lines.append("没有恶化项。")
    return "\n".join(lines)


# ------------------------------------------------------------------ CLI

def _parse_pages(text):
    if not text:
        return None
    pages = []
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        pages.append(int(chunk))
    return pages or None


def main(argv=None, log=print):
    parser = argparse.ArgumentParser(
        description="步骤2 图例样例符号的离线评测（零 Gemini 调用）")
    parser.add_argument("slugs", nargs="*", help="项目 slug，可多个")
    parser.add_argument("--from", dest="root", default=str(BASE_DIR),
                        help=f"数据根（默认本项目 {BASE_DIR}）")
    parser.add_argument("--pages", default=None, help="只看这些页，如 2,5,163")
    parser.add_argument("--json", dest="json_out", default=None,
                        help="把机器可读结果写到这个文件（用于 --diff）")
    parser.add_argument("--no-fit", action="store_true",
                        help="不读 PDF、跳过 box_fit（快）")
    parser.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                        help="对照两份 --json 结果；有恶化项时返回码 1")
    args = parser.parse_args(argv)

    if args.diff:
        before = json.loads(Path(args.diff[0]).read_text(encoding="utf-8"))
        after = json.loads(Path(args.diff[1]).read_text(encoding="utf-8"))
        diff = diff_reports(before, after)
        log(f"==== diff: {args.diff[0]} -> {args.diff[1]} ====")
        log(format_diff(diff))
        return 1 if diff["regressions"] or diff["pages_removed"] else 0

    root = Path(args.root)
    if not args.slugs:
        found = available_slugs(root)
        log(f"可评的 slug（{root}）：" + (", ".join(found) or "无"))
        return 2

    report = eval_root(args.slugs, root=root,
                       pages_filter=_parse_pages(args.pages),
                       use_pdf=not args.no_fit, log=log)
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"\n已写出 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
