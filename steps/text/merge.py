"""融合 —— 矢量实例 ∪ VLM 检出，矢量兜底保证零遗漏.

Zero-miss guarantee: every text line the judge OR the floor marks as
fence-related is structurally guaranteed to appear in the fused item list —
either covered by a VLM box (vec_covered) or added as a "vector supplement"
item (vec_added).
"""
from steps.text.clean import filter_vlm_boxes, snap_vlm_boxes

COVER_TOL = 5.0  # 0-1000 units; same coverage metric as the audit / Notion doc


def intersects(a, b, tol=COVER_TOL):
    gy = max(a[0] - b[2], b[0] - a[2], 0.0)
    gx = max(a[1] - b[3], b[1] - a[3], 0.0)
    return gx <= tol and gy <= tol


def _table_col(b, lines, need=2):
    """Table-row signature: the box's LEFT edge aligns with ≥`need` other
    text lines that sit on clearly different rows — how legend / keynote /
    schedule captions stack.  Scattered plan callouts don't."""
    x0, h = b[1], max(b[2] - b[0], 2.0)
    n = 0
    for ln in lines or []:
        lb = ln["box_2d"]
        if abs(lb[1] - x0) > 4:
            continue
        if lb[2] < b[0] - 1.2 * h or lb[0] > b[2] + 1.2 * h:
            n += 1
            if n >= need:
                return True
    return False


def fuse(vlm_items, instances, lines=None):
    """Union merge.  Returns a page record:
      vlm_items   — VLM detections, each with vec_backed flag
      vec_added   — fence text lines NO VLM box covers (the misses, now items)
      vec_covered — fence text lines a VLM box already covers (green layer)
    Every instance lands in vec_added or vec_covered — nothing is dropped.
    When `lines` (the page's vector text lines) is given, VLM boxes are first
    snapped tight to the transcribed text (see snap_vlm_boxes) and every
    output item gets a `tbl` flag (sits in a table column — see _table_col),
    which the symbol step uses to restrict sample-symbol search to legend /
    keynote contexts."""
    # Validate before snapping.  Besides protecting fresh model responses,
    # this sanitizes legacy raw VLM caches every time the deterministic fused
    # result is rebuilt.  PDF-vector instances intentionally retain their
    # separate source-coordinate semantics below.
    vlm_items = filter_vlm_boxes(vlm_items)
    if lines:
        vlm_items = snap_vlm_boxes(vlm_items, lines)
    vlm_out = []
    vboxes = []
    for it in vlm_items or []:
        box = it.get("box_2d")
        if not box:
            continue
        vboxes.append(box)
        rec = {
            "text": it.get("text", ""),
            "box_2d": box,
            "label": it.get("label", "other"),
            "source": "vlm",
            "vec_backed": False,
        }
        if it.get("box_raw"):
            rec["box_raw"] = it["box_raw"]
        if lines:
            rec["tbl"] = _table_col(box, lines)
        vlm_out.append(rec)
    added, covered = [], []
    for inst in instances or []:
        hit = False
        for i, vb in enumerate(vboxes):
            if intersects(inst["box_2d"], vb):
                hit = True
                vlm_out[i]["vec_backed"] = True
        if hit:
            covered.append(inst)
        else:
            rec = {
                "text": inst["text"],
                "box_2d": inst["box_2d"],
                "label": "vector supplement",
                "source": "vector",
            }
            if lines:
                rec["tbl"] = _table_col(inst["box_2d"], lines)
            added.append(rec)
    return {"vlm_items": vlm_out, "vec_added": added, "vec_covered": covered}
