"""箭头末端 → 唯一线型 的绑定与投票（纯函数，无引擎依赖、无 IO）.

核心判据：

    一个有效箭头末端的线型 = **离 tip 最近的已识别线型**；
    直接命中允许 36 pt，越过 residual 寻找时只允许 12 pt。

``nearest_op`` 仍保留作几何审计，但 residual 不再否决附近的已识别线型。真实图纸
里箭头尖端经常先碰到围栏的载线、边框或其他未聚类 ink，而周期图案本身在几 pt
之外；final_plans P3 的 ``8' H FENCE`` 就是这种情况：最近 residual 为 0.334 pt，
正确的 Method 2 线型为 3.437 pt。只要最近的有主 op 在 12 pt fallback 护栏内，就采信它；
没有任何有主 op 时才保留 residual 结论。

另外两条同源的硬要求（都在边车里先做掉）：

  * 排除**这个 callout 自己的引线与箭头**再找最近的 op。compound_path_periodic
    找的正是周期性重复的相同图元，而重复的箭头 / 刻度短刺就是这种东西，不排除
    就会把一堆箭头认成"线型"再高亮回去。
  * 距离用**点到线段**，不用包围盒或质心。实测有个簇拥有 725 条散落全页的 op，
    它的并集包围盒几乎盖住整张图，按盒判会把所有末端都吸过去。

产品口径的三条（在这里）：

  1. 一个末端只有一个线型。
  2. 同一个 callout 的多个末端是同一个线型（gladstone P2 的 anchor "2" 有两个）。
  3. 文字相同的多个 callout 也是同一个线型。反过来，这种「重复指代」正是排除
     无关线型的依据：没有被任何一组指到的簇不该高亮。

组内用多数票把个别咬错的末端拉回来，被改判的必须留痕（``state="reassigned"``），
不能静默改掉。

坐标：全部页面帧 0-1000，点是 [y, x]。这里不做任何 plan 取景 —— plan 只在
webapp 读盘时当显示闸用（见 resolve_visible）。
"""
from __future__ import annotations

import re
import unicodedata

# tip 直接命中有主 op 的最大可接受距离，单位 **PDF 点**（边车在等向的 IR 帧里算，
# 见 tools/linetype_sidecar/run.py 的说明；0-1000 页帧是逐轴归一的，各向异性
# 可达 1.5 倍，在那里比距离会系统性偏爱横向偏移的簇）。
# 超了说明箭头末端离任何已识别线型都太远（引线追歪、末端框飘）。36 pt ≈ 0.5 英寸，
# 已经很宽松；final_plans P3 两个正确 Method 2 目标分别仅 3.437 / 6.838 pt。
MAX_BIND_DISTANCE = 36.0

# residual → 最近已识别线型是较弱证据，不能沿用 direct owner 的 36 pt。现有 plan
# 页旧规则已确认绑定的 fallback 距离 p95=2.537 pt、最大 2.970 pt；final_plans P3
# 两个确定正确的 Method 2 目标是 3.437 / 6.838 pt。12 pt 给已知正确最大值留约
# 1.75 倍余量，同时挡住审计中 25 个 12~36 pt 的高风险远距候选。
MAX_FALLBACK_BIND_DISTANCE = 12.0

# gate / fence 分类。门在图上是单独画的符号，不是周期重复的线型图案，所以
# **纯 gate 类别不去找线**（产品口径）。但一句话同时明确写了 fence 与 gate
# 时，它描述的仍是围栏系统，必须由 fence 语义优先；否则
# ``5' ORNAMENTAL STEEL FENCE & GATE`` 会仅因末尾的 GATE 被错误剥掉线型。
# 判据与前端 templates/index.html 的 isGateText 完全一致——两处各写一份，
# 一致性由 tests/test_linetypes.py 和 tools/check_frontend_build.mjs 钉住。
#
# 这条判定放在**读盘期**（resolve_visible），不进缓存：gate 与否是对文字的
# 判断，将来可能换更好的判据，而重算一页聚类要 100 s 以上。和 plan 显示闸
# 同一个道理。
FENCE_PATTERN = re.compile(r"\bFENC(?:E[DS]?|ING)\b", re.I)
GATE_PATTERN = re.compile(r"\bGATES?\b", re.I)


def is_gate_text(text):
    value = str(text or "")
    return bool(GATE_PATTERN.search(value) and not FENCE_PATTERN.search(value))


def text_token(text):
    """同文分组用的归一化键.

    与 steps/arrows.py: suppressed_unverified_duplicates 里的归一化**必须一致**
    （NFKC + 折叠空白 + 大写）。两处各写一份是有意的 —— 线型模块不 import arrows，
    保持可独立测试；一致性由 tests/test_linetypes.py 的对齐用例保证。
    """
    value = unicodedata.normalize("NFKC", str(text or ""))
    return re.sub(r"\s+", " ", value).strip().upper()


def _anchor_text(items, key):
    if not isinstance(key, int):
        return ""
    if not 0 <= key < len(items or ()):
        return ""
    return str((items[key] or {}).get("text") or "")


def group_key_of(items, key):
    """这个锚归到哪一组：有文字就按归一化文字，没文字就自己一组."""
    token = text_token(_anchor_text(items, key))
    return f"t:{token}" if token else f"k:{key}"


def verdict_of(row, tip_precision_pt=0.0):
    """一个末端自己的结论：(state, line_type_number, distance).

    state ∈
      no-geometry  这页没有任何可比的 path 几何（纯图片页之类）
      too-far      最近的 op 也在 MAX_BIND_DISTANCE 之外，末端没落在图元上
      residual     附近没有任何已识别线型
      bound        拿到线型

    ``tip_precision_pt`` 为旧调用协议保留；新口径不再用「有主 op 与 residual 的
    距离差必须小于 tip 精度」作否决条件。direct owner 仍由
    ``MAX_BIND_DISTANCE``（36 pt）控制；residual fallback 使用更严格的
    ``MAX_FALLBACK_BIND_DISTANCE``（12 pt）。
    """
    nearest = row.get("nearest_op")
    if not isinstance(nearest, dict):
        return "no-geometry", None, None
    distance = nearest.get("distance")
    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return "no-geometry", None, None
    if distance > MAX_BIND_DISTANCE:
        return "too-far", None, distance
    number = nearest.get("owner")
    if number is not None:
        return "bound", int(number), distance
    owned = row.get("nearest_owned_op")
    if isinstance(owned, dict) and owned.get("owner") is not None:
        try:
            owned_distance = float(owned["distance"])
        except (TypeError, ValueError):
            owned_distance = None
        if owned_distance is not None \
                and owned_distance <= MAX_FALLBACK_BIND_DISTANCE:
            return "bound", int(owned["owner"]), owned_distance
    return "residual", None, distance


def _tally(rows, tip_precision_pt=0.0):
    """(line_type_number -> 票数, 距离和)，只数 state == bound 的末端."""
    votes = {}
    weight = {}
    for row in rows:
        state, number, distance = verdict_of(row, tip_precision_pt)
        if state != "bound":
            continue
        votes[number] = votes.get(number, 0) + 1
        weight[number] = weight.get(number, 0.0) + (distance or 0.0)
    return votes, weight


def _winner(votes, weight):
    """多数票；平票时取「到该线型的距离总和更小」的那个，再平取编号小的.

    平票裁决必须确定性：同一份输入永远给同一答案，否则缓存和快照对比都失去意义。
    """
    if not votes:
        return None
    ordered = sorted(votes.items(),
                     key=lambda kv: (-kv[1], weight.get(kv[0], 0.0), kv[0]))
    return ordered[0][0]


def _distance_to(row, number):
    for candidate in row.get("ranked") or ():
        if candidate.get("line_type_number") == number:
            return float(candidate["distance"])
    return None


def bind_page(items, sidecar_bindings, tip_precision_pt=0.0):
    """缓存期绑定：**不看 plan**，产出每个末端的结论与全页同文投票.

    参数
      items            : steps.store.items_of(rec)，下标 = union index
      sidecar_bindings : 边车返回的 bindings，每条
                         {"key","ti","tip","own_ops","nearest_op","ranked"}

    返回 {"bindings": [...], "groups": [...], "used_all": [...]}
      bindings[i] 追加 {group, state, line_type_number, nearest, distance,
                        distance_to_type}
        state ∈ bound / residual / too-far / no-geometry / reassigned
      groups[j]   = {group, text, keys, votes_all, line_type_number, tie}
      used_all    = 被任何一组选中的线型编号（plan 闸之前的全集）
    """
    rows = []
    for row in sidecar_bindings or []:
        key = row.get("key")
        if isinstance(key, str) and key.lstrip("-").isdigit():
            key = int(key)
        merged = dict(row)
        merged["key"] = key
        merged["group"] = group_key_of(items, key)
        rows.append(merged)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], []).append(row)

    groups = []
    used = set()
    for group, members in sorted(grouped.items()):
        votes, weight = _tally(members, tip_precision_pt)
        winner = _winner(votes, weight)
        tie = bool(votes) and len([n for n, c in votes.items()
                                   if c == max(votes.values())]) > 1
        sample = next((m for m in members if isinstance(m["key"], int)), None)
        groups.append({
            "group": group,
            "text": _anchor_text(items, sample["key"]) if sample else "",
            "keys": sorted({str(m["key"]) for m in members}),
            "votes_all": {str(n): c for n, c in sorted(votes.items())},
            "line_type_number": winner,
            "tie": tie,
        })
        if winner is not None:
            used.add(winner)
        for row in members:
            state, number, distance = verdict_of(row, tip_precision_pt)
            row["nearest"] = number
            row["distance"] = distance
            if state != "bound" or winner is None:
                # residual / too-far / no-geometry 一律不给线型。组里别人拿到了
                # 也不外推 —— 那等于用别处的证据给这里画一条线。
                row["state"] = state
                row["line_type_number"] = None
                row["distance_to_type"] = distance
            elif number == winner:
                row["state"] = "bound"
                row["line_type_number"] = winner
                row["distance_to_type"] = distance
            else:
                row["state"] = "reassigned"
                row["line_type_number"] = winner
                row["distance_to_type"] = _distance_to(row, winner)

    return {"bindings": rows, "groups": groups, "used_all": sorted(used)}


# 分组 / 投票 / gate / plan 显示闸已经移到 steps/linetypes/regroup.py（读盘期）：
# 那些都是纯字符串与坐标比较，放在读盘期意味着改产品口径不用重跑任何一页聚类。
# 这里只保留缓存期要用的东西：verdict_of / _tally / _winner / text_token。
