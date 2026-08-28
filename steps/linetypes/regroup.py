"""读盘期的分组 / 投票 / 显示闸 —— 与聚类解耦，改判据零成本.

为什么单独一层：聚类一页要 100~240 s，而分组、投票、gate 判定、plan 取景全都是
纯坐标与字符串比较。把它们放在读盘期，意味着「谁和谁算一组」「gate 要不要找线」
「plan 内外」这些**产品口径**改了，不用重跑任何一页。缓存里只放贵的那部分：
每个末端到最近 op / 最近有主 op 的距离与归属（``steps/linetypes/bind.py`` 的
``verdict_of`` 消费它）。

分组模型 = **scope**，和前端 templates/index.html 完全同一套：

  * 文字锚：按归一化文字分组（``t:<TOKEN>``），NFKC + 折空白 + 大写。
  * 放置锚（``s<si>:<pi>``）：归到**它所属图例 symbol 的那行文字**的组里 ——
    前端的 SCOPE_BY_SYMBOL 就是这么做的。这条很要紧：rapid_city_2 P11 上
    ``s1:0`` 与 ``s1:1`` 是同一个图例样例的两处放置，按旧的「每个放置各自成组」
    它们分别绑到了 #48 和 #49；而按你的口径，同一个样例的多处放置必须是同一
    线型。归到同一组之后它们才会一起投票。
    顺带也解决了放置锚拿不到 gate/fence 分类的问题（继承图例那一行的文字）。

改判（reassign）多了一条硬要求：**胜出线型必须在这个末端自己的候选里、且在
MAX_BIND_DISTANCE 之内**。否则只标 ``unresolved``，不把它挪到一条它根本没碰到
的线上。实测 rapid_city_2 P8 的 anchor 0 最近 op 在 20.3 pt 外、候选第三名还在
31.6 pt，旧规则把它改判到了 #15（离它 >31 pt）—— 可见结果碰巧是对的，但规则
本身能把末端拉到远处的线上。
"""
from __future__ import annotations

import re

from steps.linetypes.bind import (MAX_BIND_DISTANCE, _anchor_text, _tally,
                                  _winner, is_gate_text, text_token,
                                  verdict_of)

_PLACEMENT = re.compile(r"^s(\d+):(\d+)$")


def _normalise_key(key):
    if isinstance(key, int) and not isinstance(key, bool):
        return key
    text = str(key)
    return int(text) if text.lstrip("-").isdigit() else text


def symbol_index_of(key):
    """``"s<si>:<pi>"`` → si；不是放置锚就返回 None。"""
    match = _PLACEMENT.match(str(key))
    return int(match.group(1)) if match else None


def group_of(items, key, symbol_owners=None):
    """这个锚归到哪一组。返回 (group_key, 组的代表文字)。"""
    key = _normalise_key(key)
    if isinstance(key, int):
        text = _anchor_text(items, key)
        token = text_token(text)
        return (f"t:{token}" if token else f"k:{key}"), text
    index = symbol_index_of(key)
    if index is not None:
        owner = (symbol_owners or {}).get(index)
        if owner is None:
            owner = (symbol_owners or {}).get(str(index))
        if owner is not None:
            text = _anchor_text(items, _normalise_key(owner))
            token = text_token(text)
            if token:
                return f"t:{token}", text
        # 图例样例没有合法主人时，退化成「同一个 symbol 一组」——
        # 仍然比「每个放置各自一组」正确。
        return f"s:{index}", ""
    return f"k:{key}", ""


def engine_run_of(row, number):
    """这个末端落在该线型的哪一条**连通走线**上。

    高亮按走线裁，不按引擎 group 号裁。一个 global 线型跨的几个 group 既可能是
    同一道围栏的连续段（gladstone P4：中间横栏接在左侧竖栏上，几何最小距离
    0.02~0.13 页帧单位 —— 真的贴着），也可能是图上完全分开的区域（lenexa P4：
    波浪围栏与左侧长带最近 1.15 —— 只是靠近，没接触）。组号分不开这两种，
    几何接触分得开，所以判据是走线。
    """
    if number is None:
        return None
    nearest = row.get("nearest_op") or {}
    if nearest.get("owner") == number:
        return nearest.get("run_id")
    owned = row.get("nearest_owned_op") or {}
    if owned.get("owner") == number:
        return owned.get("run_id")
    return None


def _reassignable(row, number):
    """胜出线型是否在这个末端自己的可达范围内。"""
    for candidate in row.get("ranked") or ():
        if candidate.get("line_type_number") == number:
            try:
                return float(candidate["distance"]) <= MAX_BIND_DISTANCE
            except (TypeError, ValueError):
                return False
    return False


def _distance_to(row, number):
    for candidate in row.get("ranked") or ():
        if candidate.get("line_type_number") == number:
            try:
                return float(candidate["distance"])
            except (TypeError, ValueError):
                return None
    return None


def tip_in_any(tip, regions):
    """末端点是否落在任一取景框内（页面帧，框是 [y0,x0,y1,x1]）。"""
    if not (isinstance(tip, (list, tuple)) and len(tip) >= 2):
        return False
    y, x = float(tip[0]), float(tip[1])
    return any(r[0] <= y <= r[2] and r[1] <= x <= r[3] for r in regions or ())


def resolve(entry, plan_regions, items=None, symbol_owners=None):
    """从缓存的 bindings 重新分组 + 投票 + 过 gate / plan 闸.

    闸门锚在末端 tip，不是 callout 的文字框：实测 callout 几乎都在图纸边缘 /
    明细表里，只有引线伸进俯视图（gladstone P2/P3/P4/P7 的全部末端都在 plan
    内，而 P8 是详图页、15 个末端一个都不在）。按文字框卡会把绝大多数正确绑定
    误杀。

    投票只数**落在 plan 内**的末端：全页聚类之后 plan 外的末端很可能就近咬到
    图例样例上，让它去左右可见结果就错了。

    返回 {"visible", "groups", "bindings", "needs_recompute"}
      needs_recompute = 胜出了但缓存里没有折线几何的线型编号（旧缓存按当时的
      分组裁剪过 polylines）。显式报出来，不要静默画不出线。
    """
    page = entry.get("page") or {}
    precision = float(page.get("tip_precision_pt") or 0.0)
    have_geometry = {row.get("line_type_number")
                     for row in (entry.get("line_types") or ())
                     if row.get("by_run") or row.get("polylines")}

    rows = []
    for source in entry.get("bindings") or ():
        row = dict(source)
        row["key"] = _normalise_key(row.get("key"))
        group, text = group_of(items, row["key"], symbol_owners)
        row["group"] = group
        row["group_text"] = text
        row["in_plan"] = tip_in_any(row.get("tip"), plan_regions)
        rows.append(row)

    buckets = {}
    for row in rows:
        buckets.setdefault(row["group"], []).append(row)

    groups = []
    visible = set()
    missing = set()
    for group in sorted(buckets):
        members = buckets[group]
        text = next((m["group_text"] for m in members if m.get("group_text")), "")
        gate = is_gate_text(text)
        inside = [row for row in members if row["in_plan"]]
        votes_all, weight_all = _tally(members, precision)
        votes, weight = _tally(inside, precision)
        winner = None if gate else _winner(votes, weight)
        if winner is not None and winner not in have_geometry:
            missing.add(winner)
            winner = None
        groups.append({
            "group": group,
            "text": text,
            "scope": "gate" if gate else "fence",
            "keys": sorted({str(m["key"]) for m in members}),
            "votes_all": {str(n): c for n, c in sorted(votes_all.items())},
            "votes_in_plan": {str(n): c for n, c in sorted(votes.items())},
            "line_type_number_all": _winner(votes_all, weight_all) if not gate else None,
            "visible_line_type_number": winner,
            "tie": bool(votes) and len([n for n, c in votes.items()
                                        if c == max(votes.values())]) > 1,
            "plan_fallback": winner is None and not gate and bool(votes_all),
            "in_plan_count": len(inside),
        })
        if winner is not None:
            visible.add(winner)
        engine_runs = []
        for row in members:
            state, number, distance = verdict_of(row, precision)
            row["scope"] = "gate" if gate else "fence"
            row["nearest"] = number
            row["distance"] = distance
            if gate:
                row["state"] = "gate"
                row["line_type_number"] = None
                row["distance_to_type"] = distance
            elif state != "bound":
                row["state"] = state
                row["line_type_number"] = None
                row["distance_to_type"] = distance
            elif winner is None:
                row["state"] = "hidden"
                row["line_type_number"] = None
                row["distance_to_type"] = distance
            elif number == winner:
                row["state"] = "bound"
                row["line_type_number"] = winner
                row["distance_to_type"] = distance
            elif _reassignable(row, winner):
                row["state"] = "reassigned"
                row["line_type_number"] = winner
                row["distance_to_type"] = _distance_to(row, winner)
            else:
                # 组里有答案，但这个末端离那条线太远 / 压根不在它的候选里。
                # 不挪 —— 挪过去就是画一条它没碰到的线。
                row["state"] = "unresolved"
                row["line_type_number"] = None
                row["distance_to_type"] = _distance_to(row, winner)
            row["visible"] = bool(row["line_type_number"] is not None
                                  and row["in_plan"])
            row["visible_line_type_number"] = (winner if row["visible"] else None)
            row["engine_run"] = engine_run_of(row, row["line_type_number"])
            if row["visible"] and row["engine_run"] is not None                     and row["engine_run"] not in engine_runs:
                engine_runs.append(row["engine_run"])
        # 这一组 callout 真正指到的走线 —— 高亮只画这些，同型但没接上的不画。
        groups[-1]["engine_runs"] = engine_runs

    return {"visible": sorted(visible), "groups": groups, "bindings": rows,
            "needs_recompute": sorted(missing)}
