"""步骤4：图例 shape 样例 → plan 视图里的全部同款放置（纯本地矢量，零模型成本）.

每个 shape 样例框就是一个模板，core.symbolmatch 直接读 PDF 的矢量绘图命令做
平移/旋转/镜像不变的精确模板匹配，把整页同款符号都找出来（图例原件由匹配器
自己剔除，框内只有线段的样本也由它自己报错拒绝，这两条不要在这里重复实现）。

本步唯一的产品语义：**放置只在 plan 视图里算数**。匹配器是全页扫的，图签、
图例区、剖面/立面里的同形小图元都会命中；实测 5 页 1245 个放置里 1230 个落在
plan 内，被滤掉的正是图例原件与图签噪声。所以：
  * 中心落在任一 plan 组框 ±2 内 → 保留（±2 与全栈其它包含判定一致）；
  * 本页没有 plan 框 → fail-closed，一个都不留（未分类不猜 plan）。

``line`` 样例不走 shape 放置传播；它由步骤6的 supervised legend line-type
通道直接提取样例线型并匹配全图。这里仅记录「不属于 placement 阶段」，不能再
写成只标框、不做全图匹配。

这一步免费且可重复，所以 dbg 记录永远收集，PLACEMENT_VERSION bump 就是免费重算。
"""
import hashlib
import json
import math

from steps.versions import PLACEMENT_VERSION

# ±2 tolerance on plan containment — the same slack the symbol group gate and
# the rest of the stack use for "center inside this box".
PLAN_PAD = 2

LINE_NOTE = "handled by supervised legend line-type matching (not shape placement)"
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


def _scope_box(box):
    """Stable numeric geometry for placement cache identity."""
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    out = []
    for value in box:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))):
            return None
        out.append(float(value))
    return out


def placement_scope_signature(symbols, typed_groups):
    """Identity of every input that can change local placement output.

    The symbol cache signature alone is insufficient: a forced classifier
    rerun can change a view from plan to elevation while retaining the same
    classifier input signature.  Placements must then be filtered again.
    Sign the effective plan boxes and the shape-template geometry actually
    consumed by :func:`match_placements`.

    Pages without shape samples use a fixed scope.  Line samples never create
    placements, so changing view classification must not make those pages
    perpetually stale.
    """
    shapes = []
    for index, symbol in enumerate(symbols or ()):
        if not isinstance(symbol, dict) or symbol.get("category") != "shape":
            continue
        source = str(symbol.get("source") or "")
        shapes.append({
            "index": index,
            "box": _scope_box(symbol.get("box_2d")),
            "source": source,
            "content_box": (_scope_box(symbol.get("glyph_box_2d"))
                            if source == "row_code" else None),
        })
    if shapes and typed_groups is None:
        # ``None`` means classification is pending, not "classified and found
        # no plan".  Those two states must never share a placement cache key.
        return None
    plans = []
    if shapes:
        from steps.views import plan_boxes

        plans = sorted({
            tuple(normalized)
            for box in plan_boxes(typed_groups)
            if (normalized := _scope_box(box)) is not None
        })
    payload = {
        "v": PLACEMENT_VERSION,
        "plan_pad": PLAN_PAD,
        "shapes": shapes,
        "plans": [list(box) for box in plans],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


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
       "plc_v", "plc_scope_sig"} —— 调用方把它整份 merge 进 result
      （``result.update(summary)``）；版本和输入 scope 必须同时当期。
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
            "plc_v": PLACEMENT_VERSION,
            "plc_scope_sig": placement_scope_signature(
                symbols, typed_groups)}


def has_current_placements(result, typed_groups):
    """Whether local shape matching matches the current templates and plans."""
    if not (isinstance(result, dict)
            and result.get("plc_v") == PLACEMENT_VERSION):
        return False
    expected = placement_scope_signature(
        result.get("symbols") or (), typed_groups)
    return bool(isinstance(expected, str)
                and result.get("plc_scope_sig") == expected)


def current_placement_context(symbol_entry, expected_symbol_sig,
                              view_entry, pdf_revision):
    """Validate symbol, view and placement caches as one prerequisite chain.

    Consumers previously validated only ``plc_v`` and could therefore publish
    placements filtered with an old plan/elevation decision.  This helper is
    intentionally shared by the job scheduler and web readers so they cannot
    disagree about which placement boxes enter arrow/line-type signatures.

    ``typed_groups`` is ``None`` while a page with view boxes awaits a current
    classifier result.  Shape placements then fail closed.  A line-only page
    remains current because it never creates placements; its independent view
    status is still reported through ``views_current`` and ``plan_regions``.
    """
    from steps.symbols import has_current_symbols
    from steps.views import (groups_need_classification,
                             has_current_view_types, merge_view_types,
                             plan_boxes)

    context = {
        "state": "symbols-stale",
        "result": {},
        "symbols_result": {},
        "placement_result": {},
        "typed_groups": None,
        "plan_regions": [],
        "scope_sig": None,
        "symbols_current": False,
        "views_current": False,
        "placements_current": False,
    }
    if not has_current_symbols(symbol_entry, expected_symbol_sig):
        return context

    result = (symbol_entry or {}).get("result") or {}
    groups = result.get("groups") or []
    context["result"] = result
    context["symbols_result"] = result
    context["symbols_current"] = True

    needs_views = groups_need_classification(groups)
    views_current = (not needs_views or has_current_view_types(
        view_entry, groups, pdf_revision))
    typed_groups = (merge_view_types(groups, view_entry)
                    if views_current else None)
    context["views_current"] = views_current
    context["typed_groups"] = typed_groups
    context["plan_regions"] = (plan_boxes(typed_groups)
                               if typed_groups is not None else [])

    has_shape = any(
        isinstance(symbol, dict) and symbol.get("category") == "shape"
        for symbol in (result.get("symbols") or ()))
    if has_shape and typed_groups is None:
        context["state"] = "views-pending"
        return context

    # The no-shape signature deliberately ignores plans, so raw groups are a
    # safe stand-in while an unrelated view classification is pending.
    scope_groups = typed_groups if typed_groups is not None else groups
    context["scope_sig"] = placement_scope_signature(
        result.get("symbols") or (), scope_groups)
    context["placements_current"] = has_current_placements(
        result, scope_groups)
    if context["placements_current"]:
        context["placement_result"] = result
    context["state"] = ("ok" if context["placements_current"]
                        else "placements-stale")
    return context
