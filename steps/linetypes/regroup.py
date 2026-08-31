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

from steps.linetypes.bind import (MAX_BIND_DISTANCE,
                                  MAX_SYMBOL_CENTER_DISTANCE, _anchor_text,
                                  _tally, _winner, is_gate_text, text_token,
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
    limit = (MAX_SYMBOL_CENTER_DISTANCE
             if _is_symbol_center(row) else MAX_BIND_DISTANCE)
    for candidate in row.get("ranked") or ():
        if candidate.get("line_type_number") == number:
            try:
                return float(candidate["distance"]) <= limit
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


def _is_legend_template(row):
    return row.get("source") == "legend_template"


def _in_plan(row, regions):
    """Legend 模板的多个命中点只共用一个显示闸。"""
    if not _is_legend_template(row):
        return tip_in_any(row.get("tip"), regions)
    tips = row.get("tips")
    if not isinstance(tips, (list, tuple)) or not tips:
        # ``tip`` 是协议的向后兼容字段；旧生产者没有 ``tips`` 时仍可读。
        tips = (row.get("tip"),)
    return any(tip_in_any(tip, regions) for tip in tips)


def _legend_identity(row):
    """同一份 symbol 无论带多少 occurrence，都只能投一票。"""
    index = symbol_index_of(row.get("key"))
    return f"s:{index}" if index is not None else f"k:{row.get('key')}"


def _legend_summary(rows, precision):
    """汇总有效 legend 结论，并对同一 symbol 去重。

    一个异常缓存若给同一 symbol 写了两个不同结论，也必须 fail closed，不能靠输入
    顺序选中其中一个。``numbers`` 保留这类冲突的完整证据供读盘审计。
    """
    by_symbol = {}
    for row in rows:
        if not _is_legend_template(row):
            continue
        state, number, distance = verdict_of(row, precision)
        if state != "bound":
            continue
        identity = _legend_identity(row)
        record = by_symbol.setdefault(identity, {})
        old = record.get(number)
        if old is None or (distance is not None and distance < old):
            record[number] = distance or 0.0

    numbers = sorted({number for record in by_symbol.values()
                      for number in record})
    votes = {}
    weight = {}
    ambiguous = []
    for identity, record in sorted(by_symbol.items()):
        if len(record) != 1:
            ambiguous.append(identity)
            continue
        number, distance = next(iter(record.items()))
        votes[number] = votes.get(number, 0) + 1
        weight[number] = weight.get(number, 0.0) + distance
    return {
        "votes": votes,
        "weight": weight,
        "numbers": numbers,
        "conflict": len(numbers) > 1,
        "winner": numbers[0] if len(numbers) == 1 else None,
        "ambiguous_symbols": ambiguous,
    }


def _tally_with_legend_dedup(rows, precision):
    """普通末端逐条计票；legend 按 symbol 计票。"""
    ordinary = [row for row in rows
                if not _is_legend_template(row)
                and not _is_symbol_center(row)]
    votes, weight = _tally(ordinary, precision)
    legend = _legend_summary(rows, precision)
    for number, count in legend["votes"].items():
        votes[number] = votes.get(number, 0) + count
        weight[number] = weight.get(number, 0.0) \
            + legend["weight"].get(number, 0.0)
    return votes, weight, legend


def _is_symbol_center(row):
    """是不是无引线 symbol 的框中心候选（新协议显式标记）。"""
    return row.get("anchor_kind") == "symbol_center" \
        and symbol_index_of(row.get("key")) is not None


def _ordinary_tally(rows, precision):
    """中心候选不能单独改变旧箭头票选；形成共识后才另行覆盖。"""
    return _tally([row for row in rows if not _is_symbol_center(row)],
                  precision)


def _row_verdict(row, precision):
    """中心用 48pt candidate 护栏；普通箭头保持 bind.py 的原判据。"""
    if _is_symbol_center(row):
        candidates, first = _center_candidates(row)
        if first is not None:
            return "bound", first, candidates[first]
    return verdict_of(row, precision)


def _center_candidates(row):
    """一个中心在 48pt 硬护栏内共同可达的线型，及其个人首选。

    中心并不是箭头尖端：框内的 symbol ink 往往先碰到 residual，因此这里不能用
    ``verdict_of`` 的 12pt residual fallback。中心共识的安全性来自多个不同
    placement 的共同可达性；单个中心永远不会触发共识。
    """
    ordered = []

    def add(number, distance):
        if number is None:
            return
        try:
            number = int(number)
            distance = float(distance)
        except (TypeError, ValueError):
            return
        if distance < 0.0 or distance > MAX_SYMBOL_CENTER_DISTANCE:
            return
        old = next((index for index, value in enumerate(ordered)
                    if value[0] == number), None)
        if old is None:
            ordered.append((number, distance))
        elif distance < ordered[old][1]:
            ordered[old] = (number, distance)

    # ranked 的顺序是边车已做过「距离、线型具体度」裁决后的个人偏好顺序。
    for candidate in row.get("ranked") or ():
        if isinstance(candidate, dict):
            add(candidate.get("line_type_number"), candidate.get("distance"))
    nearest = row.get("nearest_op") or {}
    add(nearest.get("owner"), nearest.get("distance"))
    owned = row.get("nearest_owned_op") or {}
    add(owned.get("owner"), owned.get("distance"))
    return dict(ordered), (ordered[0][0] if ordered else None)


def _symbol_center_summary(rows):
    """找多个 distinct placement 的保守中心共识。

    候选先按覆盖 placement 数排序；最大覆盖必须至少为 2 且超过全部有效中心的
    一半。覆盖相同再依次看个人首选票、总距离，最后仍相同就 fail closed。这样
    P4 上两个中心即使各自最近分别是 #45 / #9，只要 #45 是二者共同可达候选就能
    胜出；而两个互不相认的中心、完全平票都不会凭空造答案。单 placement 只在
    没有箭头结论时使用自己的首选作 fallback，不具备推翻箭头的资格。
    """
    placements = {}
    duplicate_conflicts = set()
    symbol_indexes = set()
    for row in rows:
        if not _is_symbol_center(row):
            continue
        identity = str(row.get("key"))
        candidates, first = _center_candidates(row)
        if not candidates:
            continue
        symbol_indexes.add(symbol_index_of(identity))
        old = placements.get(identity)
        record = {"candidates": candidates, "first": first}
        if old is None:
            placements[identity] = record
        elif old != record:
            # 一个 placement 正常只会有一个中心。异常缓存不能靠输入顺序定胜负。
            duplicate_conflicts.add(identity)

    for identity in duplicate_conflicts:
        placements.pop(identity, None)

    coverage = {}
    first_votes = {}
    distance = {}
    for record in placements.values():
        first = record["first"]
        if first is not None:
            first_votes[first] = first_votes.get(first, 0) + 1
        for number, value in record["candidates"].items():
            coverage[number] = coverage.get(number, 0) + 1
            distance[number] = distance.get(number, 0.0) + value

    placement_count = len(placements)
    winner = None
    confirmed = False
    finalists = []
    # 单个无引线 placement 也要有最近已知线型的保守 fallback；如果同时存在箭头，
    # 后面的决策仍让箭头优先。多 placement 才能形成可压过一个误箭头的强共识。
    if len(symbol_indexes) == 1 and placement_count == 1:
        record = next(iter(placements.values()))
        winner = record["first"]
        finalists = [winner] if winner is not None else []
    # 同一文字组里偶尔会挂多个不同 symbol；它们不能互相拼成“多 placement”。
    # 只有同一个 symbol index 的重复放置才是本策略所需的独立重复证据。
    elif len(symbol_indexes) == 1 and placement_count >= 2 and coverage:
        best_coverage = max(coverage.values())
        if best_coverage >= 2 and best_coverage * 2 > placement_count:
            finalists = [number for number, count in coverage.items()
                         if count == best_coverage]
            best_first = max(first_votes.get(number, 0)
                             for number in finalists)
            finalists = [number for number in finalists
                         if first_votes.get(number, 0) == best_first]
            best_distance = min(distance[number] for number in finalists)
            # 浮点噪声不能把本应相同的两份证据偷偷裁出一个赢家。
            finalists = [number for number in finalists
                         if abs(distance[number] - best_distance) <= 1e-9]
            if len(finalists) == 1:
                winner = finalists[0]
                confirmed = True

    return {
        "placement_count": placement_count,
        "coverage": coverage,
        "first_votes": first_votes,
        "distance": distance,
        "winner": winner,
        "confirmed": confirmed,
        "finalists": sorted(finalists),
        "symbol_indexes": sorted(symbol_indexes),
        "duplicate_conflicts": sorted(duplicate_conflicts),
    }


def _winner_with_center_consensus(votes, weight, center):
    """中心共识可纠正至多一个误箭头；多条相反箭头仍保留旧结论。"""
    ordinary = _winner(votes, weight)
    center_winner = center["winner"]
    if center_winner is None or ordinary == center_winner:
        return ordinary if center_winner is None else center_winner
    if not center["confirmed"]:
        # 单中心只在没有其他证据时兜底；不能推翻一条真正的箭头。
        return ordinary if ordinary is not None else center_winner
    conflicting = sum(count for number, count in votes.items()
                      if number != center_winner)
    if ordinary is None or conflicting <= 1:
        return center_winner
    return ordinary


def _center_contribution(center, ordinary, selected, *, gate=False,
                         legend=False):
    """Audit how center evidence affected the final group decision."""
    center_winner = center.get("winner")
    if center_winner is None:
        return "none"
    if gate:
        return "blocked_gate"
    if legend:
        return "blocked_legend_template"
    if not center.get("confirmed"):
        return ("single_fallback" if ordinary is None
                and selected == center_winner else "blocked_by_arrow")
    if selected != center_winner:
        return "blocked_by_arrows"
    if ordinary == center_winner:
        return "corroborated"
    if ordinary is None:
        return "consensus_fallback"
    return "consensus_override"


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
        if _is_symbol_center(row):
            # 中心结论必须继承**当前可发布** symbol 的主人文字，才能判断
            # fence/gate 并与同款 placement 分组。symbols stale 时 web 层会传
            # 空 owners；若继续走下面的 ``s:<index>`` 兼容 fallback，未知语义会
            # 被默认当 fence，反而把旧中心缓存发布出来。中心新协议在这里
            # fail closed；旧 placement-arrow 行仍保留原 fallback 兼容行为。
            index = symbol_index_of(row["key"])
            owner = (symbol_owners or {}).get(index)
            if owner is None:
                owner = (symbol_owners or {}).get(str(index))
            owner = _normalise_key(owner) if owner is not None else None
            if (not isinstance(owner, int)
                    or not 0 <= owner < len(items or ())
                    or not _anchor_text(items, owner).strip()):
                continue
        group, text = group_of(items, row["key"], symbol_owners)
        row["group"] = group
        row["group_text"] = text
        row["in_plan"] = _in_plan(row, plan_regions)
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
        center_members = [row for row in members if _is_symbol_center(row)]
        center_all = _symbol_center_summary(members)
        center_inside = _symbol_center_summary(inside)
        legend_members = [row for row in members
                          if _is_legend_template(row)]
        if legend_members:
            votes_all, weight_all, legend_all = \
                _tally_with_legend_dedup(members, precision)
            votes, weight, legend_inside = \
                _tally_with_legend_dedup(inside, precision)
            if gate or legend_inside["conflict"]:
                winner = None
            elif legend_inside["winner"] is not None:
                # 图例样例是明确模板，不参加普通箭头的多数票：唯一有效结论直接胜出。
                winner = legend_inside["winner"]
            else:
                winner = _winner_with_center_consensus(
                    votes, weight, center_inside)
            if gate or legend_all["conflict"]:
                winner_all = None
            elif legend_all["winner"] is not None:
                winner_all = legend_all["winner"]
            else:
                winner_all = _winner_with_center_consensus(
                    votes_all, weight_all, center_all)
        else:
            # 中心候选单独形成共识才参与决策；没有新协议字段的旧缓存仍走原路径。
            votes_all, weight_all = _ordinary_tally(members, precision)
            votes, weight = _ordinary_tally(inside, precision)
            winner = None if gate else _winner_with_center_consensus(
                votes, weight, center_inside)
            winner_all = None if gate else _winner_with_center_consensus(
                votes_all, weight_all, center_all)
        if winner is not None and winner not in have_geometry:
            missing.add(winner)
            winner = None
        ordinary_inside = _winner(votes, weight)
        center_contribution = _center_contribution(
            center_inside, ordinary_inside, winner,
            gate=gate, legend=bool(legend_members))
        group_row = {
            "group": group,
            "text": text,
            "scope": "gate" if gate else "fence",
            "keys": sorted({str(m["key"]) for m in members}),
            "votes_all": {str(n): c for n, c in sorted(votes_all.items())},
            "votes_in_plan": {str(n): c for n, c in sorted(votes.items())},
            "line_type_number_all": winner_all,
            "visible_line_type_number": winner,
            "tie": bool(votes) and len([n for n, c in votes.items()
                                        if c == max(votes.values())]) > 1,
            "plan_fallback": winner is None and not gate
                and winner_all is not None
                and not (legend_members and legend_inside["conflict"]),
            "in_plan_count": len(inside),
        }
        if center_members:
            # 新协议才追加；旧缓存输出 schema 完全不变。
            group_row.update({
                "symbol_center_keys": sorted({str(m["key"])
                                              for m in center_members}),
                "symbol_center_placement_count_all":
                    center_all["placement_count"],
                "symbol_center_placement_count_in_plan":
                    center_inside["placement_count"],
                "symbol_center_coverage_all": {
                    str(n): c for n, c in sorted(center_all["coverage"].items())},
                "symbol_center_coverage_in_plan": {
                    str(n): c
                    for n, c in sorted(center_inside["coverage"].items())},
                "symbol_center_first_votes_in_plan": {
                    str(n): c
                    for n, c in sorted(center_inside["first_votes"].items())},
                "symbol_center_line_type_number": center_inside["winner"],
                "symbol_center_consensus_confirmed": center_inside["confirmed"],
                "symbol_center_finalists_in_plan": center_inside["finalists"],
                "symbol_center_symbol_indexes": center_inside["symbol_indexes"],
                "symbol_center_duplicate_conflicts":
                    center_inside["duplicate_conflicts"],
                "symbol_center_contribution": center_contribution,
                "symbol_center_consensus_applied": (
                    center_contribution == "consensus_override"),
            })
        if legend_members:
            # 只在新协议存在时追加审计字段，确保普通旧缓存的 JSON 语义不变。
            group_row.update({
                "legend_template_keys": sorted({str(m["key"])
                                                for m in legend_members}),
                "legend_votes_all": {
                    str(n): c for n, c in sorted(legend_all["votes"].items())},
                "legend_votes_in_plan": {
                    str(n): c
                    for n, c in sorted(legend_inside["votes"].items())},
                "legend_line_type_numbers_in_plan": legend_inside["numbers"],
                "legend_line_type_number": legend_inside["winner"],
                "legend_conflict": legend_inside["conflict"],
                "legend_ambiguous_symbols": legend_inside["ambiguous_symbols"],
            })
        groups.append(group_row)
        if winner is not None:
            visible.add(winner)
        engine_runs = []
        for row in members:
            state, number, distance = _row_verdict(row, precision)
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
            if row["visible"] and row["engine_run"] is not None \
                    and row["engine_run"] not in engine_runs:
                engine_runs.append(row["engine_run"])
            if row["visible"] and _is_legend_template(row):
                for run_id in row.get("matched_runs") or ():
                    if run_id is not None and run_id not in engine_runs:
                        engine_runs.append(run_id)
        # 这一组 callout 真正指到的走线 —— 高亮只画这些，同型但没接上的不画。
        groups[-1]["engine_runs"] = engine_runs

    return {"visible": sorted(visible), "groups": groups, "bindings": rows,
            "needs_recompute": sorted(missing)}
