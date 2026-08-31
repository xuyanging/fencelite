"""根据文字 callout 找箭头 / 引线（矢量几何，零模型调用）.

实现方式是一个 Node 边车：算法本体是 pdf-command-visualizer 里那套矢量
callout 检测（content stream 解析 → 顺序分段 → 引线拓扑恢复 → 末端框），
以单文件 bundle 形式放在 tools/arrow_sidecar/。这里只负责取景、调用、回映射。

为什么是一次性子进程而不是常驻服务：本机 2 vCPU / 可用内存 ~600 MB 且已在
吃 swap；一页的峰值 RSS 120-280 MB，跑完退出就立刻归还，常驻反而要长期占住。

取景（务必遵守）：
  引线只在 **plan 视图**里找。一条从俯视图追到立面图 / 详图里的引线是错误归属。
  plan 框来自步骤3 的分类（steps.views.plan_boxes）。**没有分类就 fail-closed**
  —— 返回空结果，绝不退化成「整页都算 plan」，与步骤4 放置匹配的口径一致。

坐标：进出都是页面帧 [ymin, xmin, ymax, xmax] / [y, x]，0-1000 闭区间，
与文字框、symbol 框同帧。PDF 用户空间的往返换算在边车里，调用方不用关心。
"""
import hashlib
import json
import math
import os
import shutil
import re
import subprocess
import tempfile
import unicodedata
from pathlib import Path

import fitz

from core.pdfio import FITZ_LOCK
from steps.store import sig_of

# 接缝总开关。ARROWS=1 可在不改代码的情况下打开。
ENABLED = os.environ.get("ARROWS", "0") not in ("0", "", "false", "no", "off")

# plan 取景开关。默认**关**：先让每个框都去找引线，把问题充分暴露出来；
# 结果质量摸清之前，用取景先砍掉一半样本反而看不到问题。
# ARROWS_PLAN_GATE=1 恢复「只在俯视图里找」的行为（含 fail-closed 语义）。
PLAN_GATE = os.environ.get("ARROWS_PLAN_GATE", "0") not in ("0", "", "false", "no", "off")

# 改箭头算法语义时 bump，让 arrows.json 重算。
ARROWS_VERSION = 17

_BASE_DIR = Path(__file__).resolve().parent.parent
_SIDECAR = _BASE_DIR / "tools" / "arrow_sidecar" / "sidecar.mjs"


def _find_node():
    """按顺序找 node：显式 env → 本机自装目录 → PATH → 裸名。

    home 那条排在 which() 前面是刻意的：arrows_signature 既不含 node 版本也不含
    sidecar.mjs 摘要，换解释器结果变了缓存**不会**失效。所以宁可保持既有选择，
    也不要让 PATH 上碰巧另一个版本悄悄接管。

    Path.home() 单独 try：容器里 HOME 未设且 uid 不在 /etc/passwd 时它会抛
    RuntimeError，而 job.py 是无条件 import 本模块的 —— 那会让整个服务起不来。
    """
    explicit = os.environ.get("ARROWS_NODE", "").strip()
    if explicit:
        return Path(explicit)
    try:
        local = Path.home() / ".local" / "opt" / "node" / "bin" / "node"
        for candidate in (local, local.with_suffix(".exe")):
            if candidate.exists():
                return candidate
    except (RuntimeError, OSError):
        pass
    return Path(shutil.which("node") or "node")


_NODE = _find_node()

# 单页超时。重页实测 ~11 s（整份 22 MB PDF），留足余量。
_TIMEOUT = int(os.environ.get("ARROWS_TIMEOUT", "600"))
# 逐级加堆。绝大多数页 384 MB 够用（实测轻页峰值 ~120 MB）；重页会 OOM，
# 于是加到 768（实测 rapid_city P27 165,913 ops 需要 ~595 MB RSS），仍不够再到
# 1536、3072、6144。宁可为个别页多花几次解析，也不接受「静默无结果」——
# 空结果和没算过在下游是完全不同的两件事。
_HEAP_LADDER = [int(v) for v in os.environ.get(
    "ARROWS_HEAP_LADDER", "384,768,1536,3072,6144").split(",")]
# 超预算页的放宽阶梯。**堆和预算是两个独立的失败原因**，所以各走各的梯子：
# OOM 只加堆，PAGE_TOO_LARGE 加预算（并同时加一档堆，因为更多线段本身就更吃内存）。
#
# 为什么要加到这么高：lenexa_fuel_station P34（63,698 条 drawings / 71,104 个
# item / 5.9 MB content stream）在旧的两档预算下一律被拒（解析出的路径线段
# 超过 2,000,000 与 8,000,000 两条上限，错误里明写「包含裁剪路径」），而实测
# 把上限提到 6400 万之后，它在 heap=1536 MB 下 **5.4 秒**就算完、找到 33 个
# automatic callout。也就是说这页一点都不难算，纯粹是预算把它挡在门外。
# 本机 16 核 / 61 GB，宁可为这种页多花几秒，也不要白丢一页结果。
_BUDGET_LADDER = [
    None,                                  # 先用边车自己的默认预算
    {"maxSceneOps": 1_000_000, "maxPathSegments": 8_000_000,
     "maxDecodedBytes": 128 * 1024 * 1024,
     "maxDecodedStreamBytes": 64 * 1024 * 1024,
     "maxSourceLength": 128 * 1024 * 1024,
     "maxSegmentsPerPath": 400_000},
    {"maxSceneOps": 8_000_000, "maxPathSegments": 64_000_000,
     "maxDecodedBytes": 512 * 1024 * 1024,
     "maxDecodedStreamBytes": 256 * 1024 * 1024,
     "maxSourceLength": 512 * 1024 * 1024,
     "maxSegmentsPerPath": 2_000_000},
    {"maxSceneOps": 32_000_000, "maxPathSegments": 256_000_000,
     "maxDecodedBytes": 2048 * 1024 * 1024,
     "maxDecodedStreamBytes": 1024 * 1024 * 1024,
     "maxSourceLength": 2048 * 1024 * 1024,
     "maxSegmentsPerPath": 8_000_000},
]


_PLAN_REGIONS_UNSET = object()


def _signature_plan_regions(regions):
    canonical = set()
    for box in regions or ():
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            continue
        values = []
        for value in box:
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))):
                values = []
                break
            values.append(float(value))
        if values:
            canonical.add(tuple(values))
    return [list(box) for box in sorted(canonical)]


def arrows_signature(items, revision, extra_anchors=None,
                     plan_regions=_PLAN_REGIONS_UNSET):
    """arrows.json 的缓存签名：两类锚 + PDF 身份 + 算法版本。

    extra_anchors 是 [(key, box_2d), ...]，即 shape 样例矢量匹配出来的放置。
    它必须进签名：放置变了而文字没变时，箭头结果同样要重算。
    """
    base = sig_of(items, revision)
    # sig_of intentionally signs only text + box because most downstream
    # stages are label-agnostic.  Arrow recovery is not: callout/vector text
    # may use the spatial fallback while titles, notes and legend rows must not.
    # Include the label vector explicitly so a relabel cannot reuse arrows that
    # were produced under a different eligibility decision.
    label_digest = hashlib.sha1(json.dumps(
        [[str((item or {}).get("label") or ""),
          str((item or {}).get("source") or ""),
          bool((item or {}).get("vec_backed"))] for item in items],
        ensure_ascii=False).encode()).hexdigest()[:12]
    base = f"{base}+l{label_digest}"
    if extra_anchors:
        digest = hashlib.sha1(json.dumps(
            [[str(k), list(b)] for k, b in extra_anchors],
            sort_keys=True).encode()).hexdigest()[:12]
        base = f"{base}+{digest}"
    if PLAN_GATE:
        if plan_regions is _PLAN_REGIONS_UNSET:
            raise TypeError(
                "plan_regions is required when ARROWS_PLAN_GATE is enabled")
        # Gate-off keeps the established cache identity because regions do not
        # affect execution in that mode.  Gate-on has an explicit namespace
        # and signs the exact effective view geometry used by the sidecar.
        region_digest = hashlib.sha1(json.dumps(
            _signature_plan_regions(plan_regions),
            separators=(",", ":")).encode()).hexdigest()[:12]
        base = f"{base}+g1p{region_digest}"
    return f"{base}|v{ARROWS_VERSION}"


def has_current_arrows(entry, sig):
    return bool(isinstance(entry, dict) and entry.get("sig") == sig
                and isinstance(entry.get("items"), dict))


def _box_inside_any(box, regions):
    """页面帧的框是否完整落在任一取景框内（同帧、同为 [y0,x0,y1,x1]）。"""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return False
    y0, x0, y1, x1 = box
    return any(y0 >= r[0] and x0 >= r[1] and y1 <= r[2] and x1 <= r[3]
               for r in regions)


def sidecar_available():
    return _SIDECAR.exists() and _NODE.exists()


def sidecar_probe(timeout=15):
    """Prove the configured Node executable can actually be launched.

    ``sidecar_available`` intentionally stays a cheap filesystem check because
    it is used while assembling API status. Service startup calls this probe
    once so a stale/non-executable Node path fails before any paid PDF stages
    begin instead of failing on the first arrow page.
    """
    if not sidecar_available():
        raise RuntimeError(
            f"arrow sidecar missing: node={_NODE} sidecar={_SIDECAR}")
    try:
        proc = subprocess.run(
            [str(_NODE), "--version"], capture_output=True, text=True,
            timeout=max(1, int(timeout)), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"arrow Node probe failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "no output").strip()
        raise RuntimeError(
            f"arrow Node probe exited {proc.returncode}: {detail}")
    return (proc.stdout or proc.stderr or "node available").strip()


def page_geometry_status(pdf_path, page_index):
    """Classify pages whose drawing is only a large embedded raster image.

    Small logos do not make an otherwise text-only page a scan.  A page is
    marked image-only only when MuPDF exposes no vector paths at all and one
    large image (or a set of image tiles) covers most of the sheet.
    """
    with FITZ_LOCK:
        doc = fitz.open(pdf_path)
        try:
            if not 0 <= int(page_index) < doc.page_count:
                raise ValueError(
                    f"page {page_index} out of range (total {doc.page_count})")
            page = doc[int(page_index)]
            drawings = page.get_drawings()
            images = page.get_images(full=True)
            page_area = max(1.0, float(page.rect.width * page.rect.height))
            coverages = []
            seen = set()
            for image in images:
                xref = int(image[0])
                # The same XObject can be painted more than once; each placement
                # matters, but exact duplicate rects must not inflate coverage.
                for rect in page.get_image_rects(xref):
                    clipped = rect & page.rect
                    token = tuple(round(float(v), 3) for v in clipped)
                    if token in seen or clipped.is_empty:
                        continue
                    seen.add(token)
                    coverages.append(float(clipped.width * clipped.height)
                                     / page_area)
        finally:
            doc.close()
    largest = max(coverages, default=0.0)
    total = min(1.0, sum(coverages))
    image_only = not drawings and bool(coverages) and (
        largest >= 0.45 or total >= 0.75)
    return {
        "state": "image-only" if image_only else "vector",
        "vector_paths": len(drawings),
        "images": len(coverages),
        "image_coverage": round(max(largest, total), 3),
    }


def _stroke_token(stroke):
    """A direction-independent, stable token for one page-frame polyline."""
    try:
        points = tuple((round(float(p[0]), 2), round(float(p[1]), 2))
                       for p in stroke if len(p) >= 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 2:
        return None
    reverse = tuple(reversed(points))
    return min(points, reverse)


def _point_distance(left, right):
    try:
        return ((float(left[0]) - float(right[0])) ** 2
                + (float(left[1]) - float(right[1])) ** 2) ** .5
    except (TypeError, ValueError, IndexError):
        return float("inf")


def _stroke_endpoints(strokes):
    points = []
    for stroke in strokes or []:
        if not isinstance(stroke, (list, tuple)) or len(stroke) < 2:
            continue
        for point in (stroke[0], stroke[-1]):
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append(point)
    return points


def _supplement_touches_current(current, supplement, tolerance=1.5):
    """Require a supplemental component to share an authored endpoint/root."""
    old = _stroke_endpoints(current.get("leader_strokes") or [])
    new = _stroke_endpoints(supplement.get("leader_strokes") or [])
    if not old or not new:
        return True
    return any(_point_distance(left, right) <= tolerance
               for left in old for right in new)


def _is_internal_terminal(tip, strokes, tolerance=1.1):
    """A terminal has one outgoing ray; a continued line/junction has >=2."""
    rays = []
    for stroke in strokes or []:
        if not isinstance(stroke, (list, tuple)):
            continue
        for left, right in zip(stroke, stroke[1:]):
            for near, far in ((left, right), (right, left)):
                if _point_distance(tip, near) > tolerance:
                    continue
                try:
                    dy = float(far[0]) - float(near[0])
                    dx = float(far[1]) - float(near[1])
                except (TypeError, ValueError, IndexError):
                    continue
                length = (dy * dy + dx * dx) ** .5
                if length <= 1e-6:
                    continue
                ray = (dy / length, dx / length)
                # Repainted/duplicate strokes in the same direction are one
                # topological ray.  Opposite directions remain distinct.
                if not any(ray[0] * old[0] + ray[1] * old[1] > .94
                           for old in rays):
                    rays.append(ray)
    return len(rays) >= 2


def _has_arrowhead(entry):
    return any(isinstance(row, dict)
               and row.get("terminal_kind") == "arrowhead"
               for row in ((entry or {}).get("targets") or []))


def _has_new_target(current, supplement, tolerance=3.0):
    old_tips = [row.get("tip") for row in (current.get("targets") or [])
                if isinstance(row, dict)]
    for row in supplement.get("targets") or []:
        if not isinstance(row, dict):
            continue
        tip = row.get("tip")
        if not old_tips or all(_point_distance(tip, old) > tolerance
                               for old in old_tips):
            return True
    return False


def _point_to_segment_distance(point, left, right):
    """Euclidean distance from one page-frame point to a line segment."""
    try:
        py, px = float(point[0]), float(point[1])
        ay, ax = float(left[0]), float(left[1])
        by, bx = float(right[0]), float(right[1])
    except (TypeError, ValueError, IndexError):
        return float("inf")
    dy, dx = by - ay, bx - ax
    denom = dy * dy + dx * dx
    if denom <= 1e-12:
        return ((py - ay) ** 2 + (px - ax) ** 2) ** .5
    t = max(0.0, min(1.0, ((py - ay) * dy + (px - ax) * dx) / denom))
    qy, qx = ay + t * dy, ax + t * dx
    return ((py - qy) ** 2 + (px - qx) ** 2) ** .5


def _bare_supplement_crosses_head(current, supplement, tolerance=4.0):
    """Whether an arrowless trace merely runs through an existing head."""
    tips = [row.get("tip") for row in (current.get("targets") or [])
            if isinstance(row, dict)
            and row.get("terminal_kind") == "arrowhead"]
    for stroke in supplement.get("leader_strokes") or []:
        if not isinstance(stroke, (list, tuple)):
            continue
        for left, right in zip(stroke, stroke[1:]):
            if any(_point_to_segment_distance(tip, left, right) <= tolerance
                   for tip in tips):
                return True
    return False


def _merge_arrow_entry(current, supplement):
    """Merge a connected recovery component into an incomplete result.

    The authored sidecar remains authoritative.  A second detector may extend
    a free terminal to its real arrowhead or recover branches that share the
    same root.  A disconnected nearby line is a different callout and is
    rejected.  Strokes are retained; a free target that becomes an internal
    junction is removed because it is no longer a terminal.
    """
    if not isinstance(current, dict):
        return supplement
    if not isinstance(supplement, dict):
        return current
    # Once a real head exists, repainting the same branch in a second vector
    # representation is not a recovery.  It only thickens the highlight and
    # can drag neighbouring geometry into the result.  Continue only when the
    # supplement contributes a genuinely distinct terminal.
    if _has_arrowhead(current) and not _has_new_target(current, supplement):
        return current
    # The endpoint graph can trace the same authored branch through a filled
    # marker and continue into glyph/linework beyond it, yielding a fake bare
    # target after a real arrowhead.  A genuinely separate bare branch may
    # still merge when it diverges at the shared root; only a trace that
    # physically crosses the existing semantic head is rejected.
    if (_has_arrowhead(current) and not _has_arrowhead(supplement)
            and _bare_supplement_crosses_head(current, supplement)):
        return current
    if not _supplement_touches_current(current, supplement):
        return current

    merged = dict(current)
    for field in ("leader_strokes", "arrow_strokes"):
        strokes = list(current.get(field) or [])
        seen = {token for token in (_stroke_token(row) for row in strokes)
                if token is not None}
        for row in supplement.get(field) or []:
            token = _stroke_token(row)
            if token is not None and token not in seen:
                strokes.append(row)
                seen.add(token)
        merged[field] = strokes

    # A second detector often describes the same terminal with a slightly
    # different box.  Upgrade coincident terminals and append genuinely new
    # branch ends only.
    targets = [dict(row) for row in (current.get("targets") or [])
               if isinstance(row, dict)]
    for row in supplement.get("targets") or []:
        if not isinstance(row, dict):
            continue
        tip = row.get("tip")
        owner = None
        if isinstance(tip, (list, tuple)) and len(tip) >= 2:
            for index, old in enumerate(targets):
                old_tip = old.get("tip")
                if not (isinstance(old_tip, (list, tuple))
                        and len(old_tip) >= 2):
                    continue
                try:
                    distance = ((float(tip[0]) - float(old_tip[0])) ** 2
                                + (float(tip[1]) - float(old_tip[1])) ** 2) ** .5
                except (TypeError, ValueError):
                    continue
                if distance <= 3.0:
                    owner = index
                    break
        if owner is None:
            targets.append(dict(row))
        elif (row.get("terminal_kind") == "arrowhead"
              and targets[owner].get("terminal_kind") != "arrowhead"):
            targets[owner] = dict(row)
    # A sidecar free-end can be the elbow where the endpoint graph continues
    # to a real arrowhead.  Once two distinct rays meet there it is an internal
    # point, not an additional target box.  Keep every stroke, but remove the
    # stale semantic endpoint.
    merged["targets"] = [
        row for row in targets
        if (row.get("terminal_kind") == "arrowhead"
            or not _is_internal_terminal(
                row.get("tip"), merged.get("leader_strokes") or []))
    ]

    if (current.get("confidence") != "high"
            and supplement.get("confidence") == "high"):
        merged["confidence"] = "high"
    notes = [str(value) for value in (current.get("note"),
                                      supplement.get("note")) if value]
    merged["note"] = " + ".join(dict.fromkeys(notes))
    return merged


def find_page_arrows(pdf_path, page_index, items, *, plan_regions=None,
                     extra_anchors=None, dbg=None, return_diagnostics=False):
    """一页里为每个文字锚找箭头 / 引线。

    参数
      pdf_path      : pathlib.Path，源 PDF
      page_index    : 0-based 页号
      items         : steps.store.items_of(rec)，下标 = union index
      plan_regions  : [[ymin, xmin, ymax, xmax], ...] plan 视图取景框（页面帧）。
                      空 / None → fail-closed，直接返回 {}。
      extra_anchors : [(key, box_2d), ...] 第二类锚 —— shape 样例矢量匹配出来的
                      放置。key 是不透明字符串（约定 "s<symbol>:<placement>"），
                      与文字锚的 int union index 分处两个键空间，不会相撞。
      dbg           : steps.debug.DebugSink 或 None

    返回
      {union_index | key: {...}} —— 只包含真的找到箭头的锚。
      int 键 = 文字锚的 union index；str 键 = extra_anchors 传进来的 key。
    """
    if not items:
        return ({}, {}) if return_diagnostics else {}
    # 取景开启时才 fail-closed（与 steps.views.plan_boxes 的下游约定一致：
    # 分类缺失就一个结果都不给）。关闭时不看 plan，任何框都去找。
    if PLAN_GATE and not plan_regions:
        if dbg:
            dbg.note("arrows: no plan regions — fail-closed, skipped page")
        return ({}, {}) if return_diagnostics else {}
    # 取景判断先在这里做：纯坐标比较，零成本。落在 plan 外的锚本来就拿不到
    # 箭头，没必要为它们启动边车、解析整页、再跑一遍全页检测——实测一页要
    # 14-24 s。真实图纸里多数文字锚在标题栏 / 明细表 / 图例里，一页 12 个锚
    # 常常只有 1 个在俯视图内，整页一个都没有也很常见。
    anchors = [(index, it.get("box_2d")) for index, it in enumerate(items)]
    anchors += [(key, box) for key, box in (extra_anchors or [])]
    if PLAN_GATE:
        inside = [(key, box) for key, box in anchors
                  if _box_inside_any(box, plan_regions)]
        if not inside:
            if dbg:
                dbg.note(f"arrows: all {len(anchors)} anchors outside plan "
                         "views, sidecar not started")
            return ({}, {}) if return_diagnostics else {}
    else:
        inside = anchors

    if not sidecar_available():
        raise RuntimeError(
            f"arrow sidecar missing: node={_NODE} exists={_NODE.exists()} "
            f"sidecar={_SIDECAR} exists={_SIDECAR.exists()}")

    # 先抽单页再交给边车。整份文档解析是这条链路上唯一的内存悬崖：paducah 的
    # 169 MB 源文件会让边车堆爆（实测 --max-old-space-size=384 仍 OOM），而单页
    # 通常几百 KB。insert_pdf 保留该页的 CropBox / Rotate / UserUnit，
    # 所以页面帧与整份加载完全一致（见 tools/arrow_sidecar 的等价性验证）。
    with tempfile.TemporaryDirectory(prefix="arrows-") as work:
        one = Path(work) / "page.pdf"
        with FITZ_LOCK:
            src = fitz.open(pdf_path)
            try:
                if not 0 <= int(page_index) < src.page_count:
                    raise ValueError(
                        f"page {page_index} out of range (total {src.page_count})")
                dst = fitz.open()
                try:
                    dst.insert_pdf(src, from_page=int(page_index),
                                   to_page=int(page_index))
                    dst.save(one)
                finally:
                    dst.close()
            finally:
                src.close()

        job = {
            "pdf": str(one),
            "page": 1,                        # 单页文件，边车用 1-based
            # 只发 plan 内的锚；边车返回的 index 是这个子集的下标，
            # 下面再映射回 union index。
            "boxes": [list(box) for _key, box in inside],
            # 放置锚（"s<symbol>:<placement>" 键）与文字锚的末端语义不同，
            # 见边车里 arrowheadOnly 的说明。
            "anchor_kinds": ["placement" if isinstance(key, str) else "text"
                             for key, _box in inside],
            # The spatial marked-leader fallback is label-aware: table/title/
            # legend text never gets a geometry-only association.  Text and
            # labels therefore belong to this stage's cache identity (see
            # arrows_signature) and travel beside every anchor.
            "anchor_labels": [
                "placement" if isinstance(key, str)
                else str((items[int(key)] or {}).get("label") or "")
                for key, _box in inside
            ],
            "anchor_texts": [
                "" if isinstance(key, str)
                else str((items[int(key)] or {}).get("text") or "")
                for key, _box in inside
            ],
            # A decoded vector instance is strong independent evidence that a
            # supplied box really owns the nearby wording.  The sidecar uses
            # this only to reject geometry-only borrowing by an unbacked,
            # text-less VLM hallucination; normal automatic/ROI ownership and
            # every vec-backed outline callout keep their existing behaviour.
            "anchor_vec_backed": [
                False if isinstance(key, str)
                else bool((items[int(key)] or {}).get("vec_backed"))
                for key, _box in inside
            ],
            # 取景关闭时不传区域：边车对空列表的语义就是「不过滤」。
            "plan_regions": ([list(box) for box in plan_regions]
                             if PLAN_GATE else []),
        }
        payload = None
        last = None
        detail = None
        heap_index = 0
        budget_index = 0
        attempts = 0
        # 两个梯子各自升级：OOM 只加堆，PAGE_TOO_LARGE 加预算并跟着加一档堆。
        # 上限是两条梯子长度之和，防止某种反复交替把循环拖成无限。
        while (heap_index < len(_HEAP_LADDER)
               and budget_index < len(_BUDGET_LADDER)
               and attempts < len(_HEAP_LADDER) + len(_BUDGET_LADDER)):
            attempts += 1
            heap = _HEAP_LADDER[heap_index]
            budget = _BUDGET_LADDER[budget_index]
            if budget:
                job["budget"] = budget
            else:
                job.pop("budget", None)
            proc = subprocess.run(
                [str(_NODE), f"--max-old-space-size={heap}",
                 "--optimize-for-size", str(_SIDECAR)],
                input=json.dumps(job), capture_output=True, text=True,
                timeout=_TIMEOUT, check=False,
            )
            if proc.returncode == 0:
                try:
                    parsed = json.loads(proc.stdout)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"arrow sidecar bad output: {exc}; "
                        f"stdout={proc.stdout[:300]!r}") from exc
                if parsed.get("ok"):
                    payload = parsed
                    break
                last = parsed.get("code")
                # 边车的解释（哪条上限、超了多少）是排查这类页唯一的线索，
                # 必须留到最终的错误里 —— 只报「走完梯子」等于把原因丢掉。
                detail = parsed.get("error")
                if last != "PAGE_TOO_LARGE":
                    raise RuntimeError(
                        f"arrow sidecar {last}: {detail}")
                budget_index += 1
                heap_index = min(heap_index + 1, len(_HEAP_LADDER) - 1)
                if dbg:
                    dbg.note(f"arrows: over budget at heap={heap}, "
                             f"widening to budget rung {budget_index}")
                continue
            oom = (proc.returncode in (134, -6)
                   or "heap out of memory" in proc.stderr.lower())
            if not oom:
                raise RuntimeError(
                    f"arrow sidecar exit {proc.returncode}: "
                    f"{proc.stderr.strip()[:400]}")
            last = "OOM"
            heap_index += 1
            if dbg:
                dbg.note(f"arrows: OOM at heap={heap} MB, retrying higher")
        if payload is None:
            # 走完梯子仍算不出来。这必须是一个显式失败，让上层记警告并把这页
            # 标成未完成 —— 绝不能写一个空结果冒充「这页没有引线」。
            # 边车的原话一并带出去：只报「走完梯子」的话，到底是哪条上限、
            # 超了多少全都看不到，排查时只能靠猜。
            raise RuntimeError(
                f"arrow sidecar exhausted ladders (heap={_HEAP_LADDER}, "
                f"budget rungs={len(_BUDGET_LADDER)}, last={last})"
                + (f": {str(detail)[:300]}" if detail else ""))
        if dbg:
            dbg.note("arrows: " + json.dumps(payload.get("page", {})))

    out = {}
    anchor_diagnostics = {}
    resolved = set()
    for row in payload.get("results", []):
        index = inside[int(row["index"])][0]  # 子集下标 → union / placement key
        if isinstance(index, int):
            anchor_diagnostics[index] = {
                "source": str(row.get("source") or ""),
                "carrier_is_text": bool(row.get("carrier_is_text")),
                "has_leader": bool(row.get("has_leader")),
            }
        if not row.get("has_leader"):
            continue
        resolved.add(index)
        targets = row.get("targets") or []
        out[index] = {
            # 真实绘制的引线 / 箭头笔画（页面帧折线），供前端按 callout 上色。
            "leader_strokes": row.get("leader_strokes") or [],
            "arrow_strokes": row.get("arrow_strokes") or [],
            # 每个末端一视同仁：n 条引线就有 n 个完整条目。
            "targets": [{"tip": t["tip"], "box_2d": t["box_2d"],
                         "terminal_kind": t["terminal_kind"]} for t in targets],
            "confidence": "high" if row.get("source") == "automatic" else "medium",
            "note": f"{row.get('source')} · {row.get('leader_count')} leader"
                    f" · {len(targets)} target",
        }

    # Second vector pass: ignore PDF paint order and walk an endpoint graph
    # from each supplied text box.  This may add a real second branch to an
    # already-headed result, but _merge_arrow_entry accepts it only when the
    # two components share an authored endpoint/root.  Thus valid multi-leader
    # callouts remain recoverable while a nearby, disconnected dimension or
    # non-fence note cannot leak into the fence result.
    text_anchors = []
    for key, box in inside:
        if not isinstance(key, int):
            continue
        item = items[key] or {}
        diagnostic = anchor_diagnostics.get(key) or {}
        # Apply the same evidence gate to the order-independent fallback as
        # the sidecar spatial pass.  Otherwise it can correctly trace a real
        # neighbouring annotation's triangle but assign it to an unbacked VLM
        # hallucination whose supplied box contains no decoded text at all.
        if (str(item.get("source") or "").lower() == "vlm"
                and not bool(item.get("vec_backed"))
                and diagnostic.get("source") == "text-only"
                and not bool(diagnostic.get("carrier_is_text"))):
            continue
        text_anchors.append((key, box, item.get("label"), item.get("text")))
    if text_anchors:
        from steps import textleaders       # local import: optional fallback
        allow_bare = {key for key, _box, label, _text in text_anchors
                      if str(label or "").strip().lower() == "callout"}
        try:
            recovered = textleaders.text_box_leaders(
                pdf_path, page_index, text_anchors,
                allow_bare_keys=allow_bare)
        except Exception as exc:             # noqa: BLE001
            if dbg:
                dbg.note(f"arrows: text endpoint pass failed: {exc}")
            recovered = {}
        for key, entry in recovered.items():
            out[key] = _merge_arrow_entry(out.get(key), entry)
        if dbg and recovered:
            dbg.note(f"arrows: text endpoint pass recovered/supplemented "
                     f"{len(recovered)}/{len(text_anchors)} anchors")

    # 放置锚的兜底：边车靠「文字与引线绘制顺序相邻」做簇形成，对编号标记
    # （圆圈里一个数字）在 combined_bid P20 上 12 个放置全部失败。放置锚有
    # 强得多的先验可用 —— 它就是一个已知的标记，引线从它的外框出发、另一端
    # 有箭头 —— 所以对边车没解出来的放置锚再用纯几何找一遍（steps/leaders.py）。
    # 只补放置锚：文字锚没有「外框」这个锚点，同样的规则在文字上会乱连。
    todo = [(key, box) for key, box in inside
            if isinstance(key, str) and key not in resolved]
    if todo:
        from steps import leaders          # 局部 import：与边车解耦
        try:
            geo = leaders.marker_leaders(pdf_path, page_index, todo)
        except Exception as exc:           # noqa: BLE001
            if dbg:
                dbg.note(f"arrows: geometric pass failed: {exc}")
            geo = {}
        if dbg and geo:
            dbg.note(f"arrows: geometric pass recovered {len(geo)}/{len(todo)} "
                     "placement anchors the sidecar missed")
        out.update(geo)
    # The sidecar diagnostics precede the order-independent Python passes.
    # Reflect any later recovery so the UI never hides a real recovered member.
    for key in out:
        if isinstance(key, int) and key in anchor_diagnostics:
            anchor_diagnostics[key]["has_leader"] = True
    if return_diagnostics:
        return out, anchor_diagnostics
    return out


def suppressed_unverified_duplicates(items, anchor_diagnostics=None):
    """Indices of weak duplicate VLM members hidden from the default UI.

    Raw/fused items and their union indices stay untouched.  Suppression is
    deliberately narrow: only a VLM callout with no vector backing, no decoded
    sidecar carrier and no leader can be hidden, and only when an identical
    same-page text member has independent vector/automatic/ROI/leader evidence.
    Debug mode can still display the weak member for audit.
    """
    diagnostics = anchor_diagnostics or {}

    def diag(index):
        return diagnostics.get(str(index), diagnostics.get(index, {})) or {}

    def key(text):
        value = unicodedata.normalize("NFKC", str(text or ""))
        return re.sub(r"\s+", " ", value).strip().upper()

    def weak(index, item):
        row = diag(index)
        return (str(item.get("source") or "").lower() == "vlm"
                and str(item.get("label") or "").strip().lower() == "callout"
                and not bool(item.get("vec_backed"))
                and row.get("source") == "text-only"
                and not bool(row.get("carrier_is_text"))
                and not bool(row.get("has_leader")))

    groups = {}
    for index, item in enumerate(items or []):
        token = key((item or {}).get("text"))
        if token:
            groups.setdefault(token, []).append(index)

    hidden = set()
    for members in groups.values():
        if len(members) < 2:
            continue
        verified = []
        for index in members:
            item = items[index] or {}
            row = diag(index)
            if not weak(index, item) and (
                    bool(item.get("vec_backed"))
                    or row.get("source") in ("automatic", "roi")
                    or bool(row.get("carrier_is_text"))
                    or bool(row.get("has_leader"))):
                verified.append(index)
        if verified:
            hidden.update(index for index in members
                          if weak(index, items[index] or {}))
    return hidden
