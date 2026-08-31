"""步骤4：图例 shape 样例 → plan 视图里的全部同款放置（纯本地矢量，零模型成本）.

每个 shape 样例框就是一个模板，core.symbolmatch 直接读 PDF 的矢量绘图命令做
平移/旋转/镜像不变的精确模板匹配，把整页同款符号都找出来（图例原件由匹配器
自己剔除，框内只有线段的样本也由它自己报错拒绝，这两条不要在这里重复实现）。

本步唯一的产品语义：**放置只在 plan 视图里算数**。匹配器是全页扫的，图签、
图例区、剖面/立面里的同形小图元都会命中；实测 5 页 1245 个放置里 1230 个落在
plan 内，被滤掉的正是图例原件与图签噪声。所以：
  * 中心落在任一 plan 组框 ±2 内 → 保留（±2 与全栈其它包含判定一致）；
  * 本页没有 plan 框 → fail-closed，一个都不留（未分类不猜 plan）。

``line`` 样例暂不匹配：整线追踪已下线，只标出图例样例本身。

这一步免费且可重复，所以 dbg 记录永远收集，PLACEMENT_VERSION bump 就是免费重算。
"""
from steps.versions import PLACEMENT_VERSION

# ±2 tolerance on plan containment — the same slack the symbol group gate and
# the rest of the stack use for "center inside this box".
PLAN_PAD = 2

LINE_NOTE = "line samples are not matched (only the legend sample is marked)"
NO_PLAN_NOTE = "no_plan_view"

# Written afresh on every run so a reused cache entry never keeps a stale
# placement field from an older algorithm version.
_STALE_KEYS = ("placements", "placement_error", "placement_note",
               "dropped_outside_plan", "dropped_without_outline")

# A derived row-code template is intentionally a native text glyph inside a
# parent-sized virtual frame.  A real plan symbol must therefore grow back to
# roughly that inherited frame; a naked decimal elsewhere on the plan is not
# a symbol even though its exact text class matches.
INHERITED_OUTLINE_MIN_RATIO = 0.75


def _has_inherited_outline(box, sample):
    if not (isinstance(box, (list, tuple)) and len(box) == 4
            and isinstance(sample, (list, tuple)) and len(sample) == 4):
        return False
    height = float(box[2]) - float(box[0])
    width = float(box[3]) - float(box[1])
    sample_height = float(sample[2]) - float(sample[0])
    sample_width = float(sample[3]) - float(sample[1])
    return (sample_height > 0 and sample_width > 0
            and height >= sample_height * INHERITED_OUTLINE_MIN_RATIO
            and width >= sample_width * INHERITED_OUTLINE_MIN_RATIO)


def _center_in_plan(box, plans):
    """Whether a placement's center falls inside any plan view box (±2)."""
    cy, cx = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for plan in plans:
        if plan[0] - PLAN_PAD <= cy <= plan[2] + PLAN_PAD \
                and plan[1] - PLAN_PAD <= cx <= plan[3] + PLAN_PAD:
            return True
    return False


def match_placements(pdf_path, page_index, symbols, typed_groups, *, dbg=None):
    """就地给每个 symbol 补 placements；返回一份汇总.

    参数
      pdf_path      : 源 PDF 路径
      page_index    : 0-based 页号
      symbols       : symbols.json 单页 result["symbols"]（原地修改）
      typed_groups  : steps.views.merge_view_types 之后的组（plan 框来源）
      dbg           : steps.debug.DebugSink 或 None

    返回
      {"shape", "line", "placed", "dropped_outside_plan", "plan_groups",
       "plc_v"} —— 调用方把它整份 merge 进 result（``result.update(summary)``），
      has_current_placements 就靠里面的 plc_v 判当期。
    """
    from core.symbolmatch import find_symbol_placements
    from steps.views import plan_boxes

    plans = plan_boxes(typed_groups)
    shape_count = line_count = placed = dropped_total = 0
    outline_dropped_total = 0
    for symbol_index, s in enumerate(symbols or []):
        for key in _STALE_KEYS:
            s.pop(key, None)              # idempotent on reused cache entries
        dropped = 0
        outline_dropped = 0
        if s.get("category") == "shape":
            shape_count += 1
            try:
                # shape sample → production compact-symbol template matcher:
                # every same-looking placement on the page (legend excluded)
                matcher_kwargs = {}
                if s.get("source") == "row_code":
                    matcher_kwargs["content_box_norm"] = s.get("glyph_box_2d")
                r = find_symbol_placements(str(pdf_path), page_index,
                                           s["box_2d"], **matcher_kwargs)
                if r.get("error"):
                    s["placement_error"] = r["error"]
                elif (s.get("source") == "row_code"
                      and (r.get("template_texts") != 1
                           or r.get("template_prims") != 1)):
                    s["placement_error"] = (
                        "derived row-code template is not exactly one native "
                        "text primitive")
                else:
                    kept = []
                    for raw_box in r.get("placements") or []:
                        box = [round(float(v), 1) for v in raw_box]
                        if (s.get("source") == "row_code"
                                and not _has_inherited_outline(
                                    box, s.get("box_2d"))):
                            outline_dropped += 1
                            continue
                        if _center_in_plan(box, plans):
                            kept.append(box)
                        else:
                            dropped += 1
                    s["placements"] = kept
                    if dropped:
                        s["dropped_outside_plan"] = dropped
                    if outline_dropped:
                        s["dropped_without_outline"] = outline_dropped
                    if not plans:
                        s["placement_note"] = NO_PLAN_NOTE
            except Exception as e:                          # noqa: BLE001
                s["placement_error"] = f"{type(e).__name__}: {e}"
        else:
            # line samples (the only other category the contract allows):
            # the sample box itself is the whole deliverable for now.
            line_count += 1
            s["placements"] = []
            s["placement_note"] = LINE_NOTE
        placed += len(s.get("placements") or [])
        dropped_total += dropped
        outline_dropped_total += outline_dropped
        if dbg is not None:
            dbg.add("placements", {
                "symbol_index": symbol_index,
                "category": s.get("category"),
                "value": s.get("value", ""),
                "sample_box": s.get("box_2d"),
                "placements": len(s.get("placements") or []),
                "dropped_outside_plan": dropped,
                "dropped_without_outline": outline_dropped,
                "status": (s.get("placement_error") or s.get("placement_note")
                           or "accepted"),
            })
    return {"shape": shape_count, "line": line_count, "placed": placed,
            "dropped_outside_plan": dropped_total, "plan_groups": len(plans),
            "dropped_without_outline": outline_dropped_total,
            "plc_v": PLACEMENT_VERSION}


def has_current_placements(result):
    """Whether this page's local shape matching is current (free to redo)."""
    return bool(isinstance(result, dict)
                and result.get("plc_v") == PLACEMENT_VERSION)
