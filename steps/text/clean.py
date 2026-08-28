"""检出清理 —— 符号码剥离 + VLM 框吸附收紧.

strip_marker_codes: 第 1 步（文字）不得抢占第 2 步（符号）的内容——
图例标记码（六边形 F-04、圆圈 33）、线内描边字母（-SF-SF- 的 SF）、
嵌在线里的短 token 都要剔除；散文里提到的 "F-04"/"CLF" 保留。
缺少矢量文字 backing 不是删除证据；必须实际位于闭合 marker 内，
或有从相对两侧逼近的线段。

snap_vlm_boxes: VLM 框漂移且常吞掉旁边的图例标记/keynote 气泡；把每个
VLM 框收缩为它真正转写过的矢量文字行的并集（原框存 box_raw）。
"""
import re

from core.parsing import is_normalized_box
from steps.text.judge import norm_text

_CODE_RE = re.compile(r"^[A-Za-z]{0,3}[-.]?\d{1,4}[A-Za-z]?$")
_TOKEN_RE = re.compile(r"^[A-Za-z]{1,3}$")


def _breaks_line(b, segs, keep=16):
    """True when linework approaches the box from two OPPOSITE sides — the
    text interrupts a line run (-SF-SF- and friends), any orientation.
    Only the NEAREST endpoints vote: glyph strokes of neighbouring rows sit
    a little further out and must not drown the true flanking line ends."""
    cy, cx = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    th = max(b[2] - b[0], b[3] - b[1], 2.0)
    pad = 1.6 * th
    cand = []
    for s in segs:
        for ex, ey in ((s["ax"], s["ay"]), (s["bx"], s["by"])):
            dy = max(b[0] - ey, 0.0, ey - b[2])
            dx = max(b[1] - ex, 0.0, ex - b[3])
            d = (dx * dx + dy * dy) ** 0.5
            if 0 < d <= pad:
                vx, vy = ex - cx, ey - cy
                n = (vx * vx + vy * vy) ** 0.5 or 1.0
                cand.append((d, vx / n, vy / n))
    cand.sort(key=lambda t: t[0])
    dirs = cand[:keep]
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            if dirs[i][1] * dirs[j][1] + dirs[i][2] * dirs[j][2] < -0.75:
                return True
    return False


def strip_marker_codes(vlm_items, instances, pdf_path, page_index, dbg=None):
    """Step 1 must NOT claim symbol content as fence text.  Two classes:
      1. marker codes — short code whose box sits inside a small closed
         shape (hexagon F-04, circle 33): graphics+code = symbol territory;
      2. line/stencil lettering — a short token that interrupts linework
         from two opposite sides (including the SF of an -SF-SF- line).
    Missing text-layer backing alone is deliberately NOT evidence: a mixed
    text/vector page can contain a legitimate VLM-only abbreviation such as
    CLF.  Every deletion therefore has closed-marker or flanking-line graphic
    evidence.  (That is why this function never looks at the page's vector
    text lines at all — they are not admissible as deletion evidence.)
    A bare "F-04" or "CLF" mention inside prose keeps its item.
    Returns (vlm_items, instances, n_stripped); zero vector work when the
    page has no code/token-shaped text.  `dbg` (optional DebugSink) records
    every dropped item with its reason."""
    def norm(it):
        return "".join(norm_text(it.get("text", "")).split())

    def is_cand(it):
        t = norm(it)
        return bool(_CODE_RE.match(t) or _TOKEN_RE.match(t))

    if not any(is_cand(it) for it in (vlm_items or []) + (instances or [])):
        return vlm_items, instances, 0
    from steps.text.markers import strip_context
    try:
        ctx = strip_context(pdf_path, page_index)
    except Exception:                                       # noqa: BLE001
        return vlm_items, instances, 0
    mboxes, segs = ctx["mboxes"], ctx["segs"]

    def _record(it, reason):
        if dbg is not None:
            dbg.add("stripped", {"text": it.get("text", ""),
                                 "box_2d": it.get("box_2d"),
                                 "reason": reason})

    def drop(it):
        t = norm(it)
        if not (_CODE_RE.match(t) or _TOKEN_RE.match(t)):
            return False
        b = it["box_2d"]
        cy, cx = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if any(
                mb[0] - 1 <= cy <= mb[2] + 1 and mb[1] - 1 <= cx <= mb[3] + 1
                for mb in mboxes):
            _record(it, "marker code: a short code inside a small closed shape (symbol territory)")
            return True
        if _breaks_line(b, segs):
            _record(it, "inline token: lines approach from two opposite directions")
            return True
        return False

    keep_v = [it for it in (vlm_items or []) if not drop(it)]
    # Vector instances are text-layer lines, but require the same explicit
    # closed-marker or flanking-line evidence as VLM items.
    keep_i = []
    for it in (instances or []):
        t = norm(it)
        bad = False
        if _CODE_RE.match(t) or _TOKEN_RE.match(t):
            b = it["box_2d"]
            cy, cx = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            in_marker = any(
                mb[0] - 1 <= cy <= mb[2] + 1 and mb[1] - 1 <= cx <= mb[3] + 1
                for mb in mboxes)
            bad = bool(in_marker) or _breaks_line(b, segs)
            if bad:
                _record(it, "marker code (vector row)" if in_marker else "inline token (vector row)")
        if not bad:
            keep_i.append(it)
    n = len(vlm_items or []) + len(instances or []) - len(keep_v) - len(keep_i)
    return keep_v, keep_i, n


_WORD_RE = re.compile(r"[A-Z0-9]+")


def _words(s):
    return _WORD_RE.findall(str(s).upper())


def filter_vlm_boxes(vlm_items):
    """Drop model detections whose normalized rectangle is not drawable.

    This is intentionally also applied to cached model output at the fuse
    boundary.  Old raw caches may predate strict response parsing, and
    clamping their coordinates would turn hallucinated/out-of-frame geometry
    into a misleading page-edge detection.
    """
    return [it for it in (vlm_items or [])
            if isinstance(it, dict)
            and is_normalized_box(it.get("box_2d"))]


def snap_vlm_boxes(vlm_items, lines):
    """VLM boxes drift and routinely swallow the adjacent legend marker /
    keynote bubble; the vector text layer is exact.  Shrink each VLM box to
    the union of the text lines it actually TRANSCRIBED: lines whose center
    sits in the box and whose words mostly (≥60%) appear in the item's text.
    The marker's code ("33") is not part of the transcription, so it drops
    out.  No text layer / no matching line → box kept as-is.
    Returns a new list; snapped items keep the original in box_raw."""
    out = []
    for it in vlm_items or []:
        box = it.get("box_2d")
        iw = set(_words(it.get("text", ""))) if box else set()
        cand = []
        if iw:
            for ln in lines or []:
                lb = ln["box_2d"]
                cy, cx = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                if not (box[0] - 2 <= cy <= box[2] + 2
                        and box[1] - 2 <= cx <= box[3] + 2):
                    continue
                lw = _words(ln["text"])
                if lw and sum(1 for w in lw if w in iw) / len(lw) >= 0.6:
                    cand.append(lb)
        if not cand:
            out.append(it)
            continue
        snapped = [round(min(b[0] for b in cand), 1),
                   round(min(b[1] for b in cand), 1),
                   round(max(b[2] for b in cand), 1),
                   round(max(b[3] for b in cand), 1)]
        # Vector text is PDF-native source geometry and can legitimately sit
        # a fraction outside the media box.  It may guide matching, but must
        # not leak that separate coordinate contract into a model/UI box.
        # Keep the original valid model box when the proposed snap is not a
        # valid normalized rectangle; never clamp either source.
        if not is_normalized_box(snapped):
            out.append(it)
            continue
        it2 = dict(it)
        it2["box_raw"] = box
        it2["box_2d"] = snapped
        out.append(it2)
    return out
