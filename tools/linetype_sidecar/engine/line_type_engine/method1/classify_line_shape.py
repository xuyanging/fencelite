#!/usr/bin/env python3
"""把 unknown_pattern_split 拆出来的每个线型再分成「直线 / 非直线」两大类。

拆分脚本回答的是“同一个粗分组里哪些线段属于同一种线型”，本脚本回答的是
“这条线型走的路径是直的，还是折的/弯的”。

判定只看线型自己的骨架（spine），不看图案形状：

- 一串圆点沿直线排列          -> 直线
- 一条带三角标记的折线        -> 非直线（折线）
- 一条虚线画出来的圆弧        -> 非直线（曲线）

骨架的取法按拆分模型区分：

- single_carrier_chain     : 有序的图案站点中心（相同图案的偏移量一致，站点中心天然共线）
- multi_carrier_network    : 站点中心 + 网络边；出现分叉直接判非直线
- strict_two_instance_chain: 只有两个站点，改用连接线段（中间桥 + 两端延长）的墨迹
- shared_reference_path    : attached_repeat 用参考路径墨迹；ink_gap_period 每条实例路径各判一次
- residual_geometry_cluster: 不是周期线型，只做几何记录，不计入两大类统计

用法：

    cd scripts
    python classify_line_shape.py --all                  # 处理当前目录全部 case，写出 JSON + Markdown
    python classify_line_shape.py case_027_commands.txt  # 单个文件，打印明细
    python classify_line_shape.py --all --metrics        # 额外打印每个线型的原始指标，便于调阈值
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent

from . import unknown_pattern_split as ups
from .unknown_pattern_split import (
    Atom,
    Candidate,
    Hypothesis,
    NetworkHypothesis,
    PatternType,
    Point,
    SharedPathHypothesis,
    TwoInstanceHypothesis,
    distance,
    normalize,
    polyline_length,
    sub,
)

EPS = 1e-9

# --- 判定阈值 -------------------------------------------------------------
# 简化后仍然存在的横向偏离占骨架长度的比例；直线的残留偏离基本为 0。
STRAIGHT_LATERAL_RATIO = 0.01
# 简化后累计转角上限（度）。1 度以下的偏折已经在简化阶段被吃掉。
STRAIGHT_TOTAL_TURN_DEGREES = 3.0
# 简化容差下限，相对骨架长度：0.5% 对应大约 1.1 度的偏折，低于它视为噪声。
SIMPLIFY_RATIO = 0.005
# 简化容差上限，防止大图案的固定偏移把真实拐弯一起吃掉。
MAXIMUM_NOISE_RATIO = 0.05
# 非直线细分：单个拐角达到这个角度就算“折角”，否则算连续弯曲。
CORNER_TURN_DEGREES = 15.0
# 判为非直线、但最大转角仍然很小的，单独标成“缓弯”，方便人工复核边界样本。
SHALLOW_TURN_DEGREES = 8.0
# 墨迹之间的间隔超过常规间隔的这个倍数就认为不是同一条载体，分成两条骨架。
BREAK_GAP_FACTOR = 5.0
# 一个线型里长度不足最长骨架这个比例的碎片，只记录不参与该线型的两大类判定。
MINOR_SPINE_RATIO = 0.05
# 骨架端点外多远还算“这条线可能被拆断了”：按周期的倍数算。
ENDPOINT_REACH_PERIODS = 0.5


# ---------------------------------------------------------------------------
# 基础几何


def deduplicate(points: Sequence[Point], tolerance: float = 1e-9) -> list[Point]:
    cleaned: list[Point] = []
    for point in points:
        if not cleaned or distance(cleaned[-1], point) > tolerance:
            cleaned.append(point)
    return cleaned


def point_line_distance(point: Point, start: Point, end: Point) -> float:
    """点到无限长直线的距离；start == end 时退化为点距。"""
    dx, dy = end[0] - start[0], end[1] - start[1]
    norm = math.hypot(dx, dy)
    if norm <= EPS:
        return distance(point, start)
    return abs((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) / norm


def total_least_squares_axis(points: Sequence[Point]) -> tuple[Point, Point]:
    """返回 (中心, 主方向单位向量)。"""
    count = len(points)
    center = (sum(p[0] for p in points) / count, sum(p[1] for p in points) / count)
    sxx = syy = sxy = 0.0
    for x, y in points:
        dx, dy = x - center[0], y - center[1]
        sxx += dx * dx
        syy += dy * dy
        sxy += dx * dy
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    return center, (math.cos(angle), math.sin(angle))


def ramer_douglas_peucker(points: Sequence[Point], epsilon: float) -> list[Point]:
    """迭代版 RDP：骨架点数可能上万，递归会撞到解释器栈上限。"""
    if len(points) < 3:
        return list(points)
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        start, end = points[first], points[last]
        index, farthest = first, -1.0
        for position in range(first + 1, last):
            deviation = point_line_distance(points[position], start, end)
            if deviation > farthest:
                index, farthest = position, deviation
        if farthest > epsilon:
            keep[index] = True
            stack.append((first, index))
            stack.append((index, last))
    return [point for point, kept in zip(points, keep) if kept]


def signed_turn_degrees(before: Point, middle: Point, after: Point) -> float:
    first = sub(middle, before)
    second = sub(after, middle)
    if math.hypot(*first) <= EPS or math.hypot(*second) <= EPS:
        return 0.0
    cross = first[0] * second[1] - first[1] * second[0]
    dot = first[0] * second[0] + first[1] * second[1]
    return math.degrees(math.atan2(cross, dot))


# ---------------------------------------------------------------------------
# 把零散的点 / 折线串成一条骨架


def minimum_spanning_tree(points: Sequence[Point], costs: Any = None) -> list[list[int]]:
    """返回最小生成树的邻接表；costs(i, j) 可自定义两点之间的代价。"""
    count = len(points)
    adjacency: list[list[int]] = [[] for _ in range(count)]
    if count <= 1:
        return adjacency
    cost_of = costs or (lambda left, right: distance(points[left], points[right]))
    in_tree = [False] * count
    best_cost = [math.inf] * count
    best_parent = [-1] * count
    best_cost[0] = 0.0
    for _ in range(count):
        current = min((index for index in range(count) if not in_tree[index]), key=lambda index: best_cost[index])
        in_tree[current] = True
        if best_parent[current] >= 0:
            adjacency[current].append(best_parent[current])
            adjacency[best_parent[current]].append(current)
        for other in range(count):
            if in_tree[other]:
                continue
            cost = cost_of(current, other)
            if cost < best_cost[other]:
                best_cost[other] = cost
                best_parent[other] = current
    return adjacency


def tree_diameter_path(points: Sequence[Point], adjacency: list[list[int]], seed: int) -> list[int]:
    def farthest_from(source: int) -> tuple[int, list[int]]:
        parent = {source: -1}
        order = [source]
        queue = [source]
        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if neighbor not in parent:
                    parent[neighbor] = node
                    order.append(neighbor)
                    queue.append(neighbor)
        # 以几何距离而不是跳数选最远端点，避免树上短枝干扰。
        target = max(order, key=lambda node: distance(points[source], points[node]))
        path = [target]
        while parent[path[-1]] >= 0:
            path.append(parent[path[-1]])
        path.reverse()
        return target, path

    first, _ = farthest_from(seed)
    _, path = farthest_from(first)
    return path


def minimum_spanning_path(points: Sequence[Point]) -> list[int]:
    """用最小生成树的最长路径给点排序，弯折路径也能排对。"""
    if len(points) <= 2:
        return list(range(len(points)))
    return tree_diameter_path(points, minimum_spanning_tree(points), 0)


def polyline_link_cost(left: Sequence[Point], right: Sequence[Point]) -> float:
    """两段墨迹之间的最近端点距离：判断它们是不是同一条线的前后段。"""
    return min(
        distance(left[0], right[0]),
        distance(left[0], right[-1]),
        distance(left[-1], right[0]),
        distance(left[-1], right[-1]),
    )


def chain_polylines(polylines: Sequence[Sequence[Point]]) -> list[Point]:
    """把多段墨迹按空间顺序接成一条骨架（只在确认同属一条线时使用）。"""
    usable = [deduplicate(line) for line in polylines if len(deduplicate(line)) >= 2]
    if not usable:
        return []
    if len(usable) == 1:
        return list(usable[0])
    midpoints = [ups.polyline_halfway_point(list(line)) for line in usable]
    order = minimum_spanning_path(midpoints)
    ordered = [usable[index] for index in order]
    chained: list[Point] = list(ordered[0])
    for line in ordered[1:]:
        tail = chained[-1]
        forward = distance(tail, line[0])
        backward = distance(tail, line[-1])
        chained.extend(line if forward <= backward else list(reversed(line)))
    return deduplicate(chained)


def separate_spines(polylines: Sequence[Sequence[Point]]) -> list[list[Point]]:
    """按空间连通性把墨迹拆成若干条骨架，互不相连的载体绝不焊成一条。

    同一个线型可能画在两条相距很远的载体上（Case 342 相距 1e4，Case 306 是
    上下两条平行虚线）。把它们串成一条会凭空造出 90°/180° 的假拐角，所以先
    按最近端点距离建最小生成树，砍掉明显超出常规间距的边，再逐个连通块成骨架；
    连通块内部若有分叉，主干走最长路径，分支自己单独成一条骨架，绝不静默丢点。
    """
    usable = [deduplicate(line) for line in polylines if len(deduplicate(line)) >= 2]
    if len(usable) <= 1:
        return [list(line) for line in usable]

    midpoints = [ups.polyline_halfway_point(list(line)) for line in usable]
    adjacency = minimum_spanning_tree(
        midpoints, lambda left, right: polyline_link_cost(usable[left], usable[right])
    )
    edges = [
        (left, right, polyline_link_cost(usable[left], usable[right]))
        for left in range(len(usable))
        for right in adjacency[left]
        if left < right
    ]
    if not edges:
        return [list(line) for line in usable]
    typical_gap = ups.median(edge[2] for edge in edges)
    typical_length = ups.median(polyline_length(list(line)) for line in usable)
    break_limit = max(BREAK_GAP_FACTOR * typical_gap, BREAK_GAP_FACTOR * typical_length)
    kept: list[list[int]] = [[] for _ in usable]
    for left, right, gap in edges:
        if gap <= break_limit:
            kept[left].append(right)
            kept[right].append(left)

    spines: list[list[Point]] = []
    unassigned = set(range(len(usable)))
    while unassigned:
        seed = min(unassigned)
        component = [seed]
        stack = [seed]
        seen = {seed}
        while stack:
            node = stack.pop()
            for neighbor in kept[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.append(neighbor)
                    stack.append(neighbor)
        unassigned -= seen
        remaining = set(component)
        while remaining:
            local = sorted(remaining)
            local_index = {value: index for index, value in enumerate(local)}
            local_adjacency = [
                [local_index[neighbor] for neighbor in kept[value] if neighbor in remaining]
                for value in local
            ]
            path = tree_diameter_path([midpoints[value] for value in local], local_adjacency, 0)
            chosen = [local[index] for index in path] or local[:1]
            spines.append(chain_polylines([usable[index] for index in chosen]))
            remaining -= set(chosen)
    return [spine for spine in spines if len(spine) >= 2]


def split_at_long_jumps(points: Sequence[Point], limit: float) -> list[list[Point]]:
    """在明显超长的跳跃处切断骨架：一条载体不会一步跨过好几个周期。"""
    cleaned = deduplicate(points)
    if len(cleaned) < 2 or limit <= EPS:
        return [list(cleaned)] if len(cleaned) >= 2 else []
    pieces: list[list[Point]] = [[cleaned[0]]]
    for previous, current in zip(cleaned, cleaned[1:]):
        if distance(previous, current) > limit:
            pieces.append([current])
        else:
            pieces[-1].append(current)
    return [piece for piece in pieces if len(piece) >= 2]


# ---------------------------------------------------------------------------
# 骨架指标与两大类判定


def spine_metrics(points: Sequence[Point], noise_tolerance: float = 0.0) -> dict[str, Any] | None:
    """骨架指标。先按噪声容差简化，再量测偏离与转角。

    ``noise_tolerance`` 用来吃掉两类假抖动：图案中心相对载体线的固定偏移，
    以及虚线端点的量化误差。真实拐弯在长线上产生的偏离远大于它。
    """
    cleaned = deduplicate(points)
    if len(cleaned) < 2:
        return None
    path_length = polyline_length(cleaned)
    if path_length <= EPS:
        return None
    # 容差有下限（吃掉量化抖动）也有上限（图案再大也不能吃掉真实拐弯）。
    tolerance = max(min(noise_tolerance, path_length * MAXIMUM_NOISE_RATIO), path_length * SIMPLIFY_RATIO, EPS)
    simplified = ramer_douglas_peucker(cleaned, tolerance)
    chord = distance(simplified[0], simplified[-1])
    if chord >= 0.15 * path_length:
        anchor_start, anchor_end = simplified[0], simplified[-1]
        fit = "chord"
    else:
        # 闭合或折返的骨架用主轴拟合，端点连线没有意义。
        center, axis = total_least_squares_axis(simplified)
        anchor_start, anchor_end = center, (center[0] + axis[0], center[1] + axis[1])
        fit = "principal_axis"
    maximum_lateral = max(point_line_distance(point, anchor_start, anchor_end) for point in simplified)
    turns = [
        signed_turn_degrees(simplified[index - 1], simplified[index], simplified[index + 1])
        for index in range(1, len(simplified) - 1)
    ]
    absolute_turns = [abs(turn) for turn in turns]
    return {
        "point_count": len(cleaned),
        "path_length": path_length,
        "chord_length": chord,
        "noise_tolerance": tolerance,
        "tortuosity": path_length / max(chord, EPS),
        "maximum_lateral": maximum_lateral,
        "lateral_ratio": maximum_lateral / path_length,
        "fit": fit,
        "simplified_vertex_count": len(simplified),
        "simplified_points": [[round(x, 2), round(y, 2)] for x, y in simplified[:40]],
        "corner_count": sum(turn >= CORNER_TURN_DEGREES for turn in absolute_turns),
        "maximum_turn_degrees": max(absolute_turns, default=0.0),
        "total_turn_degrees": sum(absolute_turns),
        "net_turn_degrees": abs(sum(turns)),
    }


def classify_metrics(metrics: dict[str, Any] | None) -> tuple[str, str, str]:
    """返回 (两大类, 细分, 判定依据)。"""
    if metrics is None:
        return "未判定", "骨架点不足", "spine has fewer than two distinct points"
    lateral_ratio = metrics["lateral_ratio"]
    total_turn = metrics["total_turn_degrees"]
    if lateral_ratio <= STRAIGHT_LATERAL_RATIO and total_turn <= STRAIGHT_TOTAL_TURN_DEGREES:
        return (
            "直线",
            "直线",
            f"lateral_ratio={lateral_ratio:.4f} <= {STRAIGHT_LATERAL_RATIO}"
            f", total_turn={total_turn:.1f}° <= {STRAIGHT_TOTAL_TURN_DEGREES}°",
        )
    corner_count = metrics["corner_count"]
    maximum_turn = metrics["maximum_turn_degrees"]
    if corner_count == 0 and maximum_turn < SHALLOW_TURN_DEGREES:
        detail = "缓弯"
    elif corner_count == 0:
        detail = "曲线"
    elif corner_count <= 3 and maximum_turn >= CORNER_TURN_DEGREES:
        detail = "折线"
    else:
        detail = "折线+曲线"
    return (
        "非直线",
        detail,
        f"lateral_ratio={lateral_ratio:.4f}, total_turn={total_turn:.1f}°"
        f", corners>={CORNER_TURN_DEGREES}°: {corner_count}, max_turn={maximum_turn:.1f}°",
    )


# ---------------------------------------------------------------------------
# 各拆分模型 -> 骨架


def station_centers(cluster: Sequence[Candidate]) -> list[Point]:
    return [candidate.center for candidate in cluster]


def motif_noise_scale(cluster: Sequence[Candidate]) -> float:
    """图案中心相对载体线的偏移量级：用图案自身尺度估计。"""
    if not cluster:
        return 0.0
    return 0.75 * ups.median(candidate.scale for candidate in cluster)


def ink_noise_scale(atoms: Sequence[Atom]) -> float:
    """纯墨迹骨架只需要吃掉线宽级别的抖动。"""
    if not atoms:
        return 0.0
    return 2.0 * ups.median((atom.line_width or 1.0) for atom in atoms)


def network_spines(hypothesis: NetworkHypothesis) -> tuple[list[list[Point]], str, bool]:
    centers = station_centers(hypothesis.cluster)
    degree: dict[int, int] = {index: 0 for index in range(len(centers))}
    for edge in hypothesis.network_edges:
        degree[edge.left_index] = degree.get(edge.left_index, 0) + 1
        degree[edge.right_index] = degree.get(edge.right_index, 0) + 1
    branching = any(count >= 3 for count in degree.values())
    if branching:
        return [centers], "network_stations(branching)", True
    connectors = [list(fragment.points) for item in hypothesis.bridge_followers for fragment in item.fragments]
    connectors.extend(list(atom.points) for atom in hypothesis.endpoint_followers)
    if connectors:
        # 网络模型可能同时覆盖多条互不相连的载体（Case 306 上下两条平行虚线、
        # Case 342 相距 1e4 的两条圆圈嵌线），必须各自成骨架。
        return separate_spines(connectors), "network_connector_ink", False
    order = minimum_spanning_path(centers)
    ordered = [centers[index] for index in order]
    return split_at_long_jumps(ordered, 3.0 * hypothesis.spacing), "network_stations", False


def two_instance_spines(hypothesis: TwoInstanceHypothesis) -> tuple[list[list[Point]], str]:
    polylines: list[list[Point]] = [list(fragment.points) for fragment in hypothesis.middle_bridge.fragments]
    polylines.append(list(hypothesis.left_extension.points))
    polylines.append(list(hypothesis.right_extension.points))
    return separate_spines(polylines), "two_instance_connectors"


def shared_path_spines(
    hypothesis: SharedPathHypothesis, atom_by_id: dict[int, Atom]
) -> list[tuple[list[Point], str, float]]:
    if hypothesis.relation_kind == "attached_repeat" and hypothesis.reference_atom_ids:
        atoms = [atom_by_id[atom_id] for atom_id in sorted(hypothesis.reference_atom_ids) if atom_id in atom_by_id]
        noise = ink_noise_scale(atoms)
        return [
            (spine, "reference_path_ink", noise)
            for spine in separate_spines([list(atom.points) for atom in atoms])
        ]
    if hypothesis.relation_kind == "self_carried_repeat" and hypothesis.motif_instances:
        # Each motif Atom contains a carrier dash plus a small attached marker.
        # The marker must not bend the inferred line spine, so retain only the
        # longest open member's physical carrier chord at each station.
        carriers: list[Atom] = []
        for atom_ids in hypothesis.motif_instances:
            members = [atom_by_id[atom_id] for atom_id in atom_ids if atom_id in atom_by_id]
            open_members = [atom for atom in members if not atom.closed and len(atom.points) >= 2]
            if open_members:
                carriers.append(max(
                    open_members,
                    key=lambda atom: distance(atom.points[0], atom.points[-1]),
                ))
        boundary_atoms = [
            atom_by_id[atom_id]
            for atom_id in hypothesis.reference_atom_ids
            if atom_id in atom_by_id and not atom_by_id[atom_id].closed
        ]
        noise = ink_noise_scale([*carriers, *boundary_atoms])
        return [
            (spine, "self_carried_chords", noise)
            for spine in separate_spines([
                *([atom.points[0], atom.points[-1]] for atom in carriers),
                *(list(atom.points) for atom in boundary_atoms),
            ])
        ]
    if hypothesis.relation_kind == "double_dot_period" and hypothesis.reference_atom_ids:
        atoms = [
            atom_by_id[atom_id]
            for atom_id in sorted(hypothesis.reference_atom_ids)
            if atom_id in atom_by_id
        ]
        noise = ink_noise_scale(atoms)
        return [
            (spine, "double_dot_carrier_ink", noise)
            for spine in separate_spines([list(atom.points) for atom in atoms])
        ]
    if hypothesis.relation_kind == "co_phased_modules" and hypothesis.reference_atom_ids:
        atoms = [
            atom_by_id[atom_id]
            for atom_id in sorted(hypothesis.reference_atom_ids)
            if atom_id in atom_by_id
        ]
        noise = ink_noise_scale(atoms)
        return [
            (spine, "co_phased_carrier_ink", noise)
            for spine in separate_spines([list(atom.points) for atom in atoms])
        ]
    # 拆分算法会把一条线在拐角处切成多条 instance（Case 505 的横段和竖段），
    # 所以先把同一线型的全部墨迹放到一起，由空间连通性决定分成几条骨架。
    atoms = [
        atom_by_id[atom_id]
        for atom_id in sorted({atom_id for instance in hypothesis.instances for atom_id in instance})
        if atom_id in atom_by_id
    ]
    noise = ink_noise_scale(atoms)
    return [
        (spine, "ink_gap_instance_path", noise)
        for spine in separate_spines([list(atom.points) for atom in atoms])
    ]


def residual_spines(pattern_type: PatternType, atom_by_id: dict[int, Atom]) -> list[tuple[list[Point], str, float]]:
    atoms = [atom_by_id[atom_id] for atom_id in sorted(pattern_type.atom_ids) if atom_id in atom_by_id]
    noise = ink_noise_scale(atoms)
    if len(atoms) == 1:
        return [(list(atoms[0].points), "residual_single_atom_ink", noise)]
    return [
        (spine, "residual_atom_chain", noise)
        for spine in separate_spines([list(atom.points) for atom in atoms])
    ]


MODEL_NAMES = {
    "discovered_periodic_pattern": "periodic",
    "carrier_supported_two_instance_pattern": "two_instance",
    "shared_reference_path_pattern": "shared_path",
    "residual_geometry_cluster": "residual",
}


def endpoint_extension_suspects(
    parts: list[dict[str, Any]], spine_points: list[list[Point]], outside_atoms: Sequence[Atom], reach: float
) -> int:
    """数一下骨架两端附近还有多少没被这个线型收进来的墨迹。

    Case 504 / 513 就是这种：线在端点处拐了 90°，但拐弯后的那几段没被拆分算法
    归进同一个线型，于是只看线型本身会得到“直线”。这里不改判定，只打个标记。
    """
    if reach <= EPS or not outside_atoms:
        return 0
    tips: list[tuple[Point, Point]] = []
    for spine in spine_points:
        cleaned = deduplicate(spine)
        if len(cleaned) < 2:
            continue
        tips.append((cleaned[0], normalize(sub(cleaned[0], cleaned[1]))))
        tips.append((cleaned[-1], normalize(sub(cleaned[-1], cleaned[-2]))))
    if not tips:
        return 0
    count = 0
    for atom in outside_atoms:
        if atom.length <= EPS:
            continue
        for end in (atom.points[0], atom.points[-1]):
            for tip, outward in tips:
                offset = sub(end, tip)
                span = math.hypot(*offset)
                # 必须落在端点外侧、够近，才算“线可能还往前长了一截”。
                if span <= reach and (offset[0] * outward[0] + offset[1] * outward[1]) > 0.3 * span:
                    count += 1
                    break
            else:
                continue
            break
    return count


def describe_pattern_type(
    pattern_type: PatternType,
    atom_by_id: dict[int, Atom],
    outside_atoms: Sequence[Atom] = (),
) -> dict[str, Any]:
    hypothesis = pattern_type.hypothesis
    originally_explained = set(getattr(hypothesis, "explained_atom_ids", ()))
    topology_absorbed_ids = sorted(pattern_type.atom_ids - originally_explained) if hypothesis else []
    record: dict[str, Any] = {
        "type_id": pattern_type.type_id,
        "kind": pattern_type.kind,
        "atom_count": len(pattern_type.atom_ids),
        "is_periodic": pattern_type.kind != "residual_geometry_cluster",
        "semantic_class": (
            "linetype"
            if pattern_type.kind != "residual_geometry_cluster"
            else "non_linetype"
        ),
        "line_type_signature": ups.line_type_signature(pattern_type, atom_by_id),
        "topology_absorbed_atom_count": len(topology_absorbed_ids),
        "topology_absorbed_atom_ids": topology_absorbed_ids,
    }

    branching = False
    spines: list[tuple[list[Point], str, float]] = []
    if isinstance(hypothesis, Hypothesis):
        record["model"] = "single_carrier_chain"
        record["station_count"] = len(hypothesis.cluster)
        record["motif_member_count"] = hypothesis.motif_member_count
        record["spacing"] = hypothesis.refined_spacing
        # full_carrier 已经把连接线段的墨迹和图案中心按弧长顺序串好，
        # 只用图案中心会漏掉“最后一段拐 90 度”这类只有墨迹能证明的拐弯。
        noise = motif_noise_scale(hypothesis.cluster)
        spines = [
            (piece, "refined_carrier_ink", noise)
            for piece in split_at_long_jumps(list(hypothesis.full_carrier), 3.0 * max(hypothesis.refined_spacing, EPS))
        ]
    elif isinstance(hypothesis, NetworkHypothesis):
        record["model"] = "multi_carrier_network"
        record["station_count"] = hypothesis.station_count
        record["motif_member_count"] = hypothesis.motif_member_count
        record["spacing"] = hypothesis.spacing
        pieces, source, branching = network_spines(hypothesis)
        noise = motif_noise_scale(hypothesis.cluster)
        spines = [(piece, source, noise) for piece in pieces]
    elif isinstance(hypothesis, TwoInstanceHypothesis):
        record["model"] = "strict_two_instance_chain"
        record["station_count"] = 2
        record["motif_member_count"] = hypothesis.motif_member_count
        record["spacing"] = hypothesis.spacing
        pieces, source = two_instance_spines(hypothesis)
        noise = motif_noise_scale(hypothesis.cluster)
        spines = [(piece, source, noise) for piece in pieces]
    elif isinstance(hypothesis, SharedPathHypothesis):
        record["model"] = f"shared_path/{hypothesis.relation_kind}"
        record["station_count"] = hypothesis.support_count
        record["motif_member_count"] = None
        record["spacing"] = hypothesis.period_length
        spines = shared_path_spines(hypothesis, atom_by_id)
    else:
        record["model"] = "residual_geometry_cluster"
        record["station_count"] = None
        record["motif_member_count"] = None
        record["spacing"] = None
        spines = residual_spines(pattern_type, atom_by_id)

    if topology_absorbed_ids and spines:
        absorbed_atoms = [
            atom_by_id[atom_id]
            for atom_id in topology_absorbed_ids
            if atom_id in atom_by_id
        ]
        repaired = separate_spines([
            *(points for points, _, _ in spines),
            *(list(atom.points) for atom in absorbed_atoms),
        ])
        if repaired:
            repair_noise = max(
                [noise for _, _, noise in spines]
                + [ink_noise_scale(absorbed_atoms)],
            )
            spines = [
                (piece, "topology_repaired_ink", repair_noise)
                for piece in repaired
            ]

    parts: list[dict[str, Any]] = []
    for points, source, noise in spines:
        metrics = spine_metrics(points, noise)
        category, detail, reason = classify_metrics(metrics)
        if branching:
            category, detail = "非直线", "分叉网络"
            reason = "carrier graph has a node with degree >= 3" + (f"; {reason}" if metrics else "")
        parts.append({
            "spine_source": source,
            "metrics": metrics,
            "shape": category,
            "shape_detail": detail,
            "reason": reason,
            "significant": True,
        })

    # 同一线型可能画在多条载体上。长度不成比例的碎片只记录，不左右该线型的归类。
    longest = max((part["metrics"]["path_length"] for part in parts if part["metrics"]), default=0.0)
    for part in parts:
        length = part["metrics"]["path_length"] if part["metrics"] else 0.0
        part["significant"] = bool(longest > 0 and length >= MINOR_SPINE_RATIO * longest)

    record["spines"] = parts
    record["spine_count"] = len(parts)
    spacing = record.get("spacing") or 0.0
    record["endpoint_extension_suspects"] = endpoint_extension_suspects(
        parts, [points for points, _, _ in spines], outside_atoms, ENDPOINT_REACH_PERIODS * spacing
    ) if record["is_periodic"] else 0
    if not record["is_periodic"]:
        # Internal residual clusters are audit buckets, not linetypes.
        record["shape"] = "非线型"
        record["shape_detail"] = "未发现足够线型证据"
        return record
    decisive = [part for part in parts if part["significant"]] or parts
    categories = {part["shape"] for part in decisive}
    if not decisive:
        record["shape"] = "未判定"
        record["shape_detail"] = "无骨架"
    elif categories == {"直线"}:
        record["shape"] = "直线"
        record["shape_detail"] = "直线" if len(decisive) == 1 else f"直线（{len(decisive)} 条载体都是直线）"
    elif "非直线" in categories:
        record["shape"] = "非直线"
        details = [part["shape_detail"] for part in decisive if part["shape"] == "非直线"]
        record["shape_detail"] = "/".join(dict.fromkeys(details))
        if len(decisive) > 1 and "直线" in categories:
            record["shape_detail"] += "（同类型内含直线载体）"
    else:
        record["shape"] = "未判定"
        record["shape_detail"] = "未判定"
    return record


# ---------------------------------------------------------------------------
# 单个 case


def analyze_case(input_path: Path) -> dict[str, Any]:
    source = input_path.read_text(encoding="latin1")
    atoms = ups.parse_painted_atoms(source)
    if not atoms:
        raise ValueError("no painted vector subpaths were found in the input")
    result = ups.discover_unknown_pattern_types(atoms)
    atom_by_id = {atom.id: atom for atom in atoms}
    periodic_atom_ids = {
        atom_id
        for pattern_type in result.types
        if pattern_type.kind != "residual_geometry_cluster"
        for atom_id in pattern_type.atom_ids
    }
    types = [
        describe_pattern_type(
            pattern_type,
            atom_by_id,
            [atom for atom in atoms if atom.id not in pattern_type.atom_ids and atom.id not in periodic_atom_ids],
        )
        for pattern_type in result.types
    ]
    periodic = [item for item in types if item["is_periodic"]]
    line_type_index = 0
    for item in types:
        if item["is_periodic"]:
            line_type_index += 1
            item["display_name"] = f"线型{line_type_index}"
            item["line_type_index"] = line_type_index
        else:
            item["display_name"] = "非线型"
            item["line_type_index"] = None
    straight = [item for item in periodic if item["shape"] == "直线"]
    curved = [item for item in periodic if item["shape"] == "非直线"]
    non_linetype = [item for item in types if not item["is_periodic"]]
    if not periodic:
        case_shape = "无周期线型"
    elif not curved:
        case_shape = "直线"
    elif not straight:
        case_shape = "非直线"
    else:
        case_shape = "混合"

    info = ups.case_source_info_for(input_path)
    return {
        "case_file": input_path.name,
        "case_id": info.case_id if info else None,
        "pdf_file": info.pdf_file_name if info else None,
        "pdf_page": info.original_page if info else None,
        "atom_count": len(atoms),
        # A residual geometry cluster is an audit bucket, not a linetype.
        "type_count": len(periodic),
        "geometry_cluster_count": len(types),
        "periodic_type_count": len(periodic),
        "straight_type_count": len(straight),
        "non_straight_type_count": len(curved),
        "non_linetype_atom_count": sum(item["atom_count"] for item in non_linetype),
        "non_linetype_cluster_count": len(non_linetype),
        # Backward-compatible alias; these are clusters, not linetypes.
        "residual_type_count": len(non_linetype),
        "case_shape": case_shape,
        "line_types": periodic,
        "non_linetype": {
            "display_name": "非线型",
            "semantic_class": "non_linetype",
            "atom_count": sum(item["atom_count"] for item in non_linetype),
            "internal_cluster_count": len(non_linetype),
            "internal_cluster_ids": [item["type_id"] for item in non_linetype],
        },
        # Compatibility/audit view: contains line types and residual clusters.
        "types": types,
    }


# ---------------------------------------------------------------------------
# 输出


def format_metrics_line(part: dict[str, Any]) -> str:
    metrics = part["metrics"]
    if metrics is None:
        return f"      spine={part['spine_source']} metrics=<none>"
    return (
        f"      spine={part['spine_source']} pts={metrics['point_count']}"
        f" len={metrics['path_length']:.1f} lateral={metrics['maximum_lateral']:.2f}"
        f" ratio={metrics['lateral_ratio']:.4f} turn_total={metrics['total_turn_degrees']:.1f}°"
        f" turn_max={metrics['maximum_turn_degrees']:.1f}° corners={metrics['corner_count']}"
        f" tortuosity={metrics['tortuosity']:.3f}"
    )


def print_case(record: dict[str, Any], show_metrics: bool) -> None:
    header = f"{record['case_file']}  atoms={record['atom_count']}  case_shape={record['case_shape']}"
    print(header)
    for item in record["line_types"]:
        print(
            f"  · {item['display_name']} [{item['model']}] atoms={item['atom_count']}"
            f" -> {item['shape']}（{item['shape_detail']}）"
        )
        if show_metrics:
            for part in item["spines"]:
                print(format_metrics_line(part))
    non_linetype = record["non_linetype"]
    if non_linetype["atom_count"]:
        print(
            f"  - 非线型 atoms={non_linetype['atom_count']}"
            f"（内部审计簇 {non_linetype['internal_cluster_count']}）"
        )


def markdown_table(records: list[dict[str, Any]]) -> str:
    lines = [
        "| Case | 原 PDF | 页 | 线型数 | 非线型图元 | 直线 | 非直线 | case 归类 | 每个线型 |",
        "|---:|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        detail = "；".join(
            f"{item['display_name']}={item['shape']}"
            + (f"({item['shape_detail']})" if item["shape"] == "非直线" else "")
            for item in record["types"]
            if item["is_periodic"]
        ) or "—"
        lines.append(
            f"| {record['case_id'] or '—'} | {record['pdf_file'] or '—'} | {record['pdf_page'] or '—'} "
            f"| {record['type_count']} | {record['non_linetype_atom_count']} | {record['straight_type_count']} "
            f"| {record['non_straight_type_count']} | {record['case_shape']} | {detail} |"
        )
    return "\n".join(lines)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    periodic_types = [item for record in records for item in record["types"] if item["is_periodic"]]
    straight = [item for item in periodic_types if item["shape"] == "直线"]
    curved = [item for item in periodic_types if item["shape"] == "非直线"]
    detail_counts: dict[str, int] = {}
    for item in curved:
        detail_counts[item["shape_detail"]] = detail_counts.get(item["shape_detail"], 0) + 1
    case_counts: dict[str, int] = {}
    for record in records:
        case_counts[record["case_shape"]] = case_counts.get(record["case_shape"], 0) + 1
    return {
        "case_count": len(records),
        "type_count": sum(record["type_count"] for record in records),
        "geometry_cluster_count": sum(record["geometry_cluster_count"] for record in records),
        "periodic_type_count": len(periodic_types),
        "non_linetype_atom_count": sum(record["non_linetype_atom_count"] for record in records),
        "non_linetype_cluster_count": sum(record["non_linetype_cluster_count"] for record in records),
        "residual_type_count": sum(record["residual_type_count"] for record in records),
        "straight_type_count": len(straight),
        "non_straight_type_count": len(curved),
        "non_straight_detail_counts": detail_counts,
        "case_shape_counts": case_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify split line types as straight or non-straight.")
    parser.add_argument("input", nargs="?", type=Path, help="Command text file")
    parser.add_argument("--all", action="store_true", help="Process every *commands*.txt in the current directory")
    parser.add_argument("--metrics", action="store_true", help="Print raw spine metrics")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=SCRIPT_DIR / "shape_classification",
        help="Output directory for results.json / results.md (default: scripts/shape_classification)",
    )
    args = parser.parse_args()

    if args.all:
        if args.input is not None:
            parser.error("input cannot be used together with --all")
        input_paths = sorted(
            (path for path in Path.cwd().glob("*commands*.txt") if path.is_file()),
            key=lambda path: path.name.lower(),
        )
        if not input_paths:
            parser.error("no *commands*.txt files found in the current directory")
    else:
        input_path = args.input or (Path.cwd() / "commands.txt")
        if not input_path.is_file():
            parser.error(f"input file not found: {input_path}")
        input_paths = [input_path]

    records: list[dict[str, Any]] = []
    failures: list[tuple[str, str]] = []
    for index, path in enumerate(input_paths, start=1):
        try:
            record = analyze_case(path)
        except Exception as error:  # noqa: BLE001 - batch keeps going
            failures.append((path.name, str(error)))
            print(f"[{index}/{len(input_paths)}] Failed: {path.name}: {error}", file=sys.stderr)
            continue
        records.append(record)
        print_case(record, args.metrics)

    summary = summarize(records)
    print()
    print(
        f"Cases: {summary['case_count']}  线型: {summary['type_count']}"
        f"  非线型图元: {summary['non_linetype_atom_count']}"
        f"（内部审计簇 {summary['non_linetype_cluster_count']}）"
    )
    print(
        f"周期线型分类: 直线 {summary['straight_type_count']}"
        f" / 非直线 {summary['non_straight_type_count']} {summary['non_straight_detail_counts']}"
    )
    print(f"Case 归类: {summary['case_shape_counts']}")
    if failures:
        print(f"Failed: {len(failures)}", file=sys.stderr)

    if args.all or args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        payload = {
            "thresholds": {
                "straight_lateral_ratio": STRAIGHT_LATERAL_RATIO,
                "straight_total_turn_degrees": STRAIGHT_TOTAL_TURN_DEGREES,
                "simplify_ratio": SIMPLIFY_RATIO,
                "corner_turn_degrees": CORNER_TURN_DEGREES,
            },
            "summary": summary,
            "failures": [{"case_file": name, "error": message} for name, message in failures],
            "cases": records,
        }
        (args.output / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output / "results.md").write_text(markdown_table(records) + "\n", encoding="utf-8")
        print(f"Written: {(args.output / 'results.json').resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
