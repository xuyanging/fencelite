"""Assign stable global linetype IDs across independently processed Groups.

Input is ``classify_line_shape.py``'s results.json.  Each proven periodic type
contains a position/rotation-independent ``line_type_signature``.  This script
uses complete-link matching: a new type may join a global family only when it
matches every existing member, preventing transitive A~B~C drift.

Example:

    python match_line_types_across_groups.py \
        shape_classification/results.json \
        -o shape_classification/global_line_types.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent

from . import unknown_pattern_split as ups


def _case_sort_key(case: dict[str, Any]) -> tuple[int, str]:
    value = str(case.get("case_id") or "")
    return (int(value), value) if value.isdigit() else (10**12, value)


def global_linetype_registry(
    payload: dict[str, Any],
    maximum_scale_ratio: float = ups.LINE_TYPE_SIGNATURE_MAXIMUM_SCALE_RATIO,
    maximum_period_ratio: float = ups.LINE_TYPE_SIGNATURE_MAXIMUM_PERIOD_RATIO,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    unsigned_periodic: list[dict[str, Any]] = []
    for case in sorted(payload.get("cases", []), key=_case_sort_key):
        for pattern_type in case.get("line_types", case.get("types", [])):
            if not pattern_type.get("is_periodic"):
                continue
            entry = {
                "case_id": case.get("case_id"),
                "case_file": case.get("case_file"),
                "type_id": pattern_type.get("type_id"),
                "display_name": pattern_type.get("display_name"),
                "model": pattern_type.get("model"),
                "shape": pattern_type.get("shape"),
                "shape_detail": pattern_type.get("shape_detail"),
                "atom_count": pattern_type.get("atom_count"),
                "signature": pattern_type.get("line_type_signature"),
            }
            if entry["signature"] is None:
                unsigned_periodic.append({key: value for key, value in entry.items() if key != "signature"})
            else:
                entries.append(entry)

    groups: list[list[dict[str, Any]]] = []
    group_pair_comparisons: list[list[dict[str, Any]]] = []
    for entry in entries:
        choices: list[tuple[float, int, list[dict[str, Any]]]] = []
        for group_index, group in enumerate(groups):
            comparisons = [
                ups.compare_line_type_signatures(
                    entry["signature"], member["signature"],
                    maximum_scale_ratio=maximum_scale_ratio,
                    maximum_period_ratio=maximum_period_ratio,
                )
                for member in group
            ]
            if comparisons and all(item["matched"] for item in comparisons):
                choices.append((
                    min(item["similarity"] for item in comparisons),
                    group_index,
                    comparisons,
                ))
        if choices:
            _, group_index, comparisons = max(choices, key=lambda item: (item[0], -item[1]))
            groups[group_index].append(entry)
            group_pair_comparisons[group_index].extend(comparisons)
        else:
            groups.append([entry])
            group_pair_comparisons.append([])

    global_types: list[dict[str, Any]] = []
    for index, (group, comparisons) in enumerate(zip(groups, group_pair_comparisons), start=1):
        cases = {member["case_id"] for member in group}
        pairwise_matches: list[dict[str, Any]] = []
        for left_index, left in enumerate(group):
            for right in group[left_index + 1:]:
                comparison = ups.compare_line_type_signatures(
                    left["signature"], right["signature"],
                    maximum_scale_ratio=maximum_scale_ratio,
                    maximum_period_ratio=maximum_period_ratio,
                )
                pairwise_matches.append({
                    "left": {
                        "case_id": left["case_id"],
                        "type_id": left["type_id"],
                        "display_name": left["display_name"],
                    },
                    "right": {
                        "case_id": right["case_id"],
                        "type_id": right["type_id"],
                        "display_name": right["display_name"],
                    },
                    "comparison": comparison,
                })
        global_types.append({
            "global_type_id": f"global_type_{index:03d}",
            "signature_family": group[0]["signature"]["family"],
            "member_count": len(group),
            "group_count": len(cases),
            "minimum_pair_similarity": min(
                (item["similarity"] for item in comparisons),
                default=1.0,
            ),
            "representative_signature": group[0]["signature"],
            "pairwise_matches": pairwise_matches,
            "members": [
                {key: value for key, value in member.items() if key != "signature"}
                for member in group
            ],
        })
    return {
        "schema_version": 1,
        "matching_policy": {
            "algorithm": "deterministic_complete_link",
            "maximum_scale_ratio": maximum_scale_ratio,
            "maximum_period_ratio": maximum_period_ratio,
            "residual_types_are_registered": False,
        },
        "summary": {
            "signed_periodic_type_count": len(entries),
            "unsigned_periodic_type_count": len(unsigned_periodic),
            "global_type_count": len(global_types),
            "cross_group_global_type_count": sum(item["group_count"] > 1 for item in global_types),
        },
        "unsigned_periodic_types": unsigned_periodic,
        "global_types": global_types,
    }


def _ratio_text(value: Any) -> str:
    return f"{float(value):.6f}" if isinstance(value, (int, float)) else "—"


def _comparison_evidence(comparison: dict[str, Any]) -> str:
    if "scale_ratio" in comparison:
        command = comparison.get("period_command_sequence", {}).get("reason", "—")
        period_ink = comparison.get("period_ink", {}).get("reason", "—")
        return (
            f"scale={_ratio_text(comparison.get('scale_ratio'))}; "
            f"period={_ratio_text(comparison.get('period_ratio'))}; "
            f"normalized-period={_ratio_text(comparison.get('normalized_period_ratio'))}; "
            f"shape={'yes' if comparison.get('shape_match') else 'no'}; "
            f"cap={'yes' if comparison.get('line_cap_match') else 'no'}; "
            f"command={command}; period-ink={period_ink}"
        )
    return (
        f"period={_ratio_text(comparison.get('period_ratio'))}; "
        f"component-error={_ratio_text(comparison.get('maximum_component_error'))}; "
        f"cap={'yes' if comparison.get('line_cap_match') else 'no'}"
    )


def cross_group_markdown(registry: dict[str, Any]) -> str:
    """Human-readable audit of only the families spanning multiple Groups."""
    summary = registry["summary"]
    policy = registry["matching_policy"]
    lines = [
        "# 跨 Group 相同线型匹配结果",
        "",
        f"- 已签名本地线型：{summary['signed_periodic_type_count']}",
        f"- 未签名本地线型：{summary['unsigned_periodic_type_count']}",
        f"- 全局线型：{summary['global_type_count']}",
        f"- 含多个 Group 的全局线型：{summary['cross_group_global_type_count']}",
        f"- 匹配策略：complete-link；尺度比 ≤ {policy['maximum_scale_ratio']}；周期比 ≤ {policy['maximum_period_ratio']}",
        "- 非线型不参与全局注册。直线／非直线描述的是载体走向，不影响线型身份匹配。",
        "",
    ]
    cross_groups = [item for item in registry["global_types"] if item["group_count"] > 1]
    for group in cross_groups:
        lines.extend([
            f"## {group['global_type_id']}",
            "",
            f"签名族：`{group['signature_family']}`；Group 数：{group['group_count']}；"
            f"最低两两相似度：{group['minimum_pair_similarity']:.6f}",
            "",
            "| Case | 本地线型 | 模型 | 走向 | Atom 数 |",
            "|---:|---|---|---|---:|",
        ])
        for member in group["members"]:
            lines.append(
                f"| {member['case_id']} | {member.get('display_name') or member['type_id']} | "
                f"`{member['model']}` | {member['shape']} | {member['atom_count']} |"
            )
        lines.extend([
            "",
            "| 左侧 | 右侧 | 相似度 | 匹配证据 |",
            "|---|---|---:|---|",
        ])
        for pair in group["pairwise_matches"]:
            left, right, comparison = pair["left"], pair["right"], pair["comparison"]
            lines.append(
                f"| Case {left['case_id']} {left.get('display_name') or left['type_id']} | "
                f"Case {right['case_id']} {right.get('display_name') or right['type_id']} | "
                f"{comparison['similarity']:.6f} | {_comparison_evidence(comparison)} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Match periodic linetypes across independently processed Groups.",
    )
    parser.add_argument("results", type=Path, help="classify_line_shape.py results.json")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output global registry JSON")
    parser.add_argument("--markdown", type=Path, help="Optional human-readable cross-Group audit report")
    parser.add_argument("--maximum-scale-ratio", type=float, default=ups.LINE_TYPE_SIGNATURE_MAXIMUM_SCALE_RATIO)
    parser.add_argument("--maximum-period-ratio", type=float, default=ups.LINE_TYPE_SIGNATURE_MAXIMUM_PERIOD_RATIO)
    args = parser.parse_args()
    if args.maximum_scale_ratio < 1 or args.maximum_period_ratio < 1:
        parser.error("ratio limits must be >= 1")
    payload = json.loads(args.results.read_text(encoding="utf-8"))
    registry = global_linetype_registry(
        payload,
        maximum_scale_ratio=args.maximum_scale_ratio,
        maximum_period_ratio=args.maximum_period_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(cross_group_markdown(registry), encoding="utf-8")
    summary = registry["summary"]
    print(
        f"Signed periodic types: {summary['signed_periodic_type_count']}; "
        f"global types: {summary['global_type_count']}; "
        f"cross-Group matches: {summary['cross_group_global_type_count']}"
    )
    print(f"Written: {args.output.resolve()}")
    if args.markdown is not None:
        print(f"Written: {args.markdown.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
