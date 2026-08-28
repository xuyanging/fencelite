"""Compare native PageIR Grouping with a frozen TypeScript page snapshot.

The two PDF parsers do not share an operation-index space.  Their painted path
counts and order do match on migration fixtures, so this module joins paths by
their dense, zero-based *path ordinal*.  It never infers a TypeScript Scene
index from PyMuPDF ``paint_order`` / ``seqno``.

The diagnostic is intentionally read-only: it reports partition, atom, and
text-batch drift without changing either parser or any grouping threshold.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from typing import Any, Iterable, Mapping, Sequence

from ..ir import ImageOperationIR, PageIR, PathOperationIR, TextOperationIR
from ..operation_index import PageOperationIndex
from ..ir import GroupingIR


SCHEMA_VERSION = 1


def _drawable_subpath_count(operation: PathOperationIR) -> int:
    """Match Method1's atom multiplicity without serializing command text."""

    subpath_has_draw = False
    count = 0
    for segment in operation.segments:
        if segment.kind == "move":
            if subpath_has_draw:
                count += 1
            subpath_has_draw = False
        elif segment.kind in {"line", "curve"}:
            subpath_has_draw = True
    if subpath_has_draw:
        count += 1
    return count


def _kind_counts(operations: Iterable[object]) -> dict[str, int]:
    counts = {"path": 0, "text": 0, "image": 0}
    for operation in operations:
        if isinstance(operation, PathOperationIR):
            counts["path"] += 1
        elif isinstance(operation, TextOperationIR):
            counts["text"] += 1
        elif isinstance(operation, ImageOperationIR):
            counts["image"] += 1
        else:  # pragma: no cover - PageIR is a closed operation union.
            raise TypeError(f"unsupported PageIR operation: {type(operation)!r}")
    return counts


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items())
    }


def _segment_kind_counts(operation: PathOperationIR) -> dict[str, int]:
    counts = {"move": 0, "line": 0, "curve": 0, "close": 0}
    for segment in operation.segments:
        counts[segment.kind] += 1
    return counts


def _text_batch_size_histogram(operations: Iterable[object]) -> dict[str, int]:
    sizes = Counter(
        operation.paint_order
        for operation in operations
        if isinstance(operation, TextOperationIR)
    )
    return _histogram(sizes.values())


def describe_python_grouping(page: PageIR, grouping: GroupingIR) -> dict[str, Any]:
    """Return a compact, deterministic snapshot of one Python partition."""

    operation_index = PageOperationIndex.build(page, grouping)
    paths: list[dict[str, int]] = []
    path_ordinal_by_dense: dict[int, int] = {}
    for dense_op_index, operation in enumerate(page.operations):
        if not isinstance(operation, PathOperationIR):
            continue
        if operation.ordinal != len(paths):
            raise ValueError(
                "PageIR path ordinals must be a dense zero-based sequence in paint order"
            )
        entry = {
            "path_ordinal": operation.ordinal,
            "dense_op_index": dense_op_index,
            "paint_order": operation.paint_order,
            "atom_multiplicity": _drawable_subpath_count(operation),
            "segment_kind_counts": _segment_kind_counts(operation),
        }
        paths.append(entry)
        path_ordinal_by_dense[dense_op_index] = operation.ordinal

    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(grouping.groups):
        dense_indices = operation_index.group_indices(group.group_id)
        if not dense_indices:
            raise ValueError(f"Python Group {group.group_id} is empty")
        start = dense_indices[0]
        end = dense_indices[-1] + 1
        if dense_indices != tuple(range(start, end)):
            raise ValueError(f"Python Group {group.group_id} is not a dense operation range")
        operations = tuple(page.operations[index] for index in dense_indices)
        path_ordinals = [
            path_ordinal_by_dense[index]
            for index in dense_indices
            if index in path_ordinal_by_dense
        ]
        ordinal_range = (
            [path_ordinals[0], path_ordinals[-1]] if path_ordinals else None
        )
        if ordinal_range and ordinal_range[1] - ordinal_range[0] + 1 != len(path_ordinals):
            raise ValueError(
                f"Python Group {group.group_id} has a non-contiguous path ordinal range"
            )
        path_operations = tuple(
            operation
            for operation in operations
            if isinstance(operation, PathOperationIR)
        )
        kind_counts = _kind_counts(operations)
        text_paint_orders = {
            operation.paint_order
            for operation in operations
            if isinstance(operation, TextOperationIR)
        }
        previous_path = next(
            (
                path_ordinal_by_dense[index]
                for index in range(start - 1, -1, -1)
                if index in path_ordinal_by_dense
            ),
            None,
        )
        next_path = next(
            (
                path_ordinal_by_dense[index]
                for index in range(end, len(page.operations))
                if index in path_ordinal_by_dense
            ),
            None,
        )
        multiplicities = [
            _drawable_subpath_count(operation) for operation in path_operations
        ]
        groups.append({
            "group_id": group.group_id,
            "dense_op_range": [start, end],
            "operation_kind_counts": kind_counts,
            "path_ordinal_range": ordinal_range,
            "path_count": len(path_operations),
            "atom_count": sum(multiplicities),
            "atom_multiplicity_histogram": _histogram(multiplicities),
            "text_paint_batch_count": len(text_paint_orders),
            "pure_text_batch": (
                not path_operations
                and kind_counts["text"] > 0
                and kind_counts["image"] == 0
            ),
            "neighbor_path_ordinals": [previous_path, next_path],
            "boundary_before": None if group_index == 0 else {
                "reasons": list(group.split_reasons),
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "python-pageir-grouping-snapshot",
        "source_name": page.source_name,
        "page_number": page.page_number,
        "page_fingerprint": page.fingerprint,
        "operation_count": len(page.operations),
        "operation_kind_counts": _kind_counts(page.operations),
        "distinct_text_paint_batch_count": len({
            operation.paint_order
            for operation in page.operations
            if isinstance(operation, TextOperationIR)
        }),
        "text_paint_batch_size_histogram": _text_batch_size_histogram(page.operations),
        "path_count": len(paths),
        "group_count": len(groups),
        "paths": paths,
        "groups": groups,
    }


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{label} must be an integer")
    return value


def _validated_paths(
    snapshot: Mapping[str, Any],
    *,
    index_key: str,
    label: str,
) -> tuple[dict[str, int], ...]:
    raw_paths = snapshot.get("paths")
    if not isinstance(raw_paths, list):
        raise ValueError(f"{label}.paths must be a list")
    paths: list[dict[str, int]] = []
    seen_indices: set[int] = set()
    for expected_ordinal, raw in enumerate(raw_paths):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{label}.paths[{expected_ordinal}] must be an object")
        ordinal = _require_int(raw.get("path_ordinal"), "path_ordinal")
        index = _require_int(raw.get(index_key), index_key)
        multiplicity = _require_int(raw.get("atom_multiplicity"), "atom_multiplicity")
        if ordinal != expected_ordinal:
            raise ValueError(f"{label} path ordinals are not dense and ordered")
        if index < 0 or index in seen_indices:
            raise ValueError(f"{label} {index_key} values must be unique and non-negative")
        if multiplicity < 1:
            raise ValueError(f"{label} path atom multiplicity must be positive")
        seen_indices.add(index)
        paths.append({
            "path_ordinal": ordinal,
            index_key: index,
            "atom_multiplicity": multiplicity,
            "segment_kind_counts": dict(raw.get("segment_kind_counts", {})),
        })
    declared_count = _require_int(snapshot.get("path_count"), f"{label}.path_count")
    if declared_count != len(paths):
        raise ValueError(f"{label}.path_count does not match paths")
    return tuple(paths)


def _groups(snapshot: Mapping[str, Any], label: str) -> tuple[Mapping[str, Any], ...]:
    raw = snapshot.get("groups")
    if not isinstance(raw, list) or not all(isinstance(item, Mapping) for item in raw):
        raise ValueError(f"{label}.groups must be a list of objects")
    declared = _require_int(snapshot.get("group_count"), f"{label}.group_count")
    if declared != len(raw):
        raise ValueError(f"{label}.group_count does not match groups")
    return tuple(raw)


def _path_owners(
    groups: Sequence[Mapping[str, Any]],
    path_count: int,
    label: str,
) -> tuple[str, ...]:
    owners: list[str | None] = [None] * path_count
    for group in groups:
        ordinal_range = group.get("path_ordinal_range")
        if ordinal_range is None:
            continue
        if (
            not isinstance(ordinal_range, list)
            or len(ordinal_range) != 2
            or not all(isinstance(value, int) for value in ordinal_range)
        ):
            raise ValueError(f"{label} Group path_ordinal_range is invalid")
        start, end = ordinal_range
        group_id = str(group.get("group_id", ""))
        if not group_id or start < 0 or end < start or end >= path_count:
            raise ValueError(f"{label} Group path_ordinal_range is out of bounds")
        for ordinal in range(start, end + 1):
            if owners[ordinal] is not None:
                raise ValueError(f"{label} path ordinal {ordinal} has duplicate ownership")
            owners[ordinal] = group_id
    if any(owner is None for owner in owners):
        missing = next(index for index, owner in enumerate(owners) if owner is None)
        raise ValueError(f"{label} path ordinal {missing} has no Group owner")
    return tuple(str(owner) for owner in owners)


def _partition_boundaries(owners: Sequence[str]) -> set[int]:
    return {
        ordinal
        for ordinal in range(1, len(owners))
        if owners[ordinal] != owners[ordinal - 1]
    }


def _consecutive_ranges(values: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(values))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def _group_by_id(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for group in groups:
        group_id = str(group.get("group_id", ""))
        if not group_id or group_id in result:
            raise ValueError("Group ids must be present and unique")
        result[group_id] = group
    return result


def _boundary_details(
    positions: Iterable[int],
    primary_owners: Sequence[str],
    secondary_owners: Sequence[str],
    primary_groups: Sequence[Mapping[str, Any]],
    secondary_groups: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_by_id = _group_by_id(primary_groups)
    secondary_by_id = _group_by_id(secondary_groups)
    group_positions = {
        str(group["group_id"]): index for index, group in enumerate(primary_groups)
    }
    details: list[dict[str, Any]] = []
    def compact(group: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "group_id": str(group.get("group_id")),
            "path_ordinal_range": group.get("path_ordinal_range"),
            "path_count": int(group.get("path_count", 0)),
            "atom_count": int(group.get("atom_count", 0)),
            "operation_kind_counts": dict(group.get("operation_kind_counts", {})),
            "text_paint_batch_count": int(group.get("text_paint_batch_count", 0)),
            "pure_text_batch": group.get("pure_text_batch") is True,
        }
    for position in sorted(positions):
        left_id = primary_owners[position - 1]
        right_id = primary_owners[position]
        secondary_id = secondary_owners[position]
        right_group = primary_by_id[right_id]
        boundary = right_group.get("boundary_before")
        reasons = list(boundary.get("reasons", [])) if isinstance(boundary, Mapping) else []
        left_position = group_positions[left_id]
        right_position = group_positions[right_id]
        boundary_chain = []
        for group in primary_groups[left_position + 1:right_position + 1]:
            group_boundary = group.get("boundary_before")
            boundary_chain.append({
                "right_group_id": str(group.get("group_id")),
                "path_count": int(group.get("path_count", 0)),
                "pure_text_batch": group.get("pure_text_batch") is True,
                "reasons": (
                    list(group_boundary.get("reasons", []))
                    if isinstance(group_boundary, Mapping)
                    else []
                ),
                "reason_details": (
                    list(group_boundary.get("reason_details", []))
                    if isinstance(group_boundary, Mapping)
                    else []
                ),
            })
        details.append({
            "before_path_ordinal": position,
            "left_path_ordinal": position - 1,
            "right_path_ordinal": position,
            "primary_group_pair": [left_id, right_id],
            "primary_group_summaries": [
                compact(primary_by_id[left_id]),
                compact(primary_by_id[right_id]),
            ],
            "secondary_group": secondary_id,
            "secondary_group_summary": compact(secondary_by_id[secondary_id]),
            "primary_boundary_reasons": reasons,
            "primary_boundary_chain": boundary_chain,
            "intervening_non_path_group_ids": [
                str(group.get("group_id"))
                for group in primary_groups[left_position + 1:right_position]
                if int(group.get("path_count", 0)) == 0
            ],
        })
    return details


def _atom_pair_ranges(
    python_paths: Sequence[Mapping[str, int]],
    ts_paths: Sequence[Mapping[str, int]],
) -> tuple[
    list[dict[str, Any]],
    Counter[tuple[int, int]],
    Counter[str],
]:
    mismatch_rows: list[tuple[int, int, int]] = []
    pair_counts: Counter[tuple[int, int]] = Counter()
    mismatch_segment_signatures: Counter[str] = Counter()
    for ordinal, (python_path, ts_path) in enumerate(zip(python_paths, ts_paths)):
        pair = (
            int(ts_path["atom_multiplicity"]),
            int(python_path["atom_multiplicity"]),
        )
        pair_counts[pair] += 1
        if pair[0] != pair[1]:
            mismatch_rows.append((ordinal, pair[0], pair[1]))
            def signature(path: Mapping[str, Any]) -> str:
                counts = path.get("segment_kind_counts", {})
                return "/".join(
                    f"{code}{int(counts.get(kind, 0))}"
                    for kind, code in (
                        ("move", "M"),
                        ("line", "L"),
                        ("curve", "C"),
                        ("close", "Z"),
                    )
                )
            mismatch_segment_signatures[
                f"ts:{signature(ts_path)}->python:{signature(python_path)}"
            ] += 1

    ranges: list[dict[str, Any]] = []
    if mismatch_rows:
        start = previous = mismatch_rows[0][0]
        ts_value, python_value = mismatch_rows[0][1:]
        for ordinal, next_ts, next_python in mismatch_rows[1:]:
            if (
                ordinal != previous + 1
                or next_ts != ts_value
                or next_python != python_value
            ):
                ranges.append({
                    "path_ordinal_range": [start, previous],
                    "ts_atom_multiplicity": ts_value,
                    "python_atom_multiplicity": python_value,
                })
                start = ordinal
                ts_value, python_value = next_ts, next_python
            previous = ordinal
        ranges.append({
            "path_ordinal_range": [start, previous],
            "ts_atom_multiplicity": ts_value,
            "python_atom_multiplicity": python_value,
        })
    return ranges, pair_counts, mismatch_segment_signatures


def compare_grouping_snapshots(
    python_snapshot: Mapping[str, Any],
    frozen_ts_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a machine-readable path-ordinal parity report."""

    if python_snapshot.get("kind") != "python-pageir-grouping-snapshot":
        raise ValueError("python snapshot kind is invalid")
    if frozen_ts_snapshot.get("kind") != "frozen-ts-page-grouping-snapshot":
        raise ValueError("frozen TypeScript snapshot kind is invalid")
    if python_snapshot.get("page_number") != frozen_ts_snapshot.get("page_number"):
        raise ValueError("Python and frozen TypeScript snapshots describe different pages")
    python_paths = _validated_paths(
        python_snapshot, index_key="dense_op_index", label="python"
    )
    ts_paths = _validated_paths(
        frozen_ts_snapshot, index_key="scene_op_index", label="frozen_ts"
    )
    if len(python_paths) != len(ts_paths):
        raise ValueError(
            "path-ordinal mapping requires equal path counts: "
            f"Python {len(python_paths)}, TS {len(ts_paths)}"
        )
    python_groups = _groups(python_snapshot, "python")
    ts_groups = _groups(frozen_ts_snapshot, "frozen_ts")
    path_count = len(python_paths)
    python_owners = _path_owners(python_groups, path_count, "python")
    ts_owners = _path_owners(ts_groups, path_count, "frozen_ts")
    python_boundaries = _partition_boundaries(python_owners)
    ts_boundaries = _partition_boundaries(ts_owners)
    _group_by_id(python_groups)
    _group_by_id(ts_groups)

    mapping_payload = [
        [
            ordinal,
            python_path["dense_op_index"],
            ts_path["scene_op_index"],
        ]
        for ordinal, (python_path, ts_path) in enumerate(zip(python_paths, ts_paths))
    ]
    mapping_digest = sha256(json.dumps(
        mapping_payload, separators=(",", ":")
    ).encode("utf-8")).hexdigest()

    atom_ranges, atom_pairs, atom_segment_signatures = _atom_pair_ranges(
        python_paths, ts_paths
    )
    atom_mismatch_count = sum(
        count for pair, count in atom_pairs.items() if pair[0] != pair[1]
    )
    python_pure_text = [
        dict(group) for group in python_groups if group.get("pure_text_batch") is True
    ]
    ts_pure_text = [
        dict(group) for group in ts_groups if group.get("pure_text_batch") is True
    ]
    python_path_groups = sum(group.get("path_count", 0) > 0 for group in python_groups)
    ts_path_groups = sum(group.get("path_count", 0) > 0 for group in ts_groups)
    python_non_path_groups = len(python_groups) - python_path_groups
    ts_non_path_groups = len(ts_groups) - ts_path_groups
    python_pure_text_count = len(python_pure_text)
    ts_pure_text_count = len(ts_pure_text)
    python_other_non_path_count = python_non_path_groups - python_pure_text_count
    ts_other_non_path_count = ts_non_path_groups - ts_pure_text_count
    python_text_batches = python_snapshot.get("distinct_text_paint_batch_count")
    ts_text_batches = frozen_ts_snapshot.get("distinct_text_paint_batch_count")
    ts_only_boundary_positions = sorted(ts_boundaries - python_boundaries)
    python_only_details = _boundary_details(
        python_boundaries - ts_boundaries,
        python_owners,
        ts_owners,
        python_groups,
        ts_groups,
    )
    ts_only_details = _boundary_details(
        ts_boundaries - python_boundaries,
        ts_owners,
        python_owners,
        ts_groups,
        python_groups,
    )
    for detail in (*python_only_details, *ts_only_details):
        detail["path_index_mapping"] = [
            {
                "path_ordinal": ordinal,
                "python_dense_op_index": python_paths[ordinal]["dense_op_index"],
                "ts_scene_op_index": ts_paths[ordinal]["scene_op_index"],
            }
            for ordinal in (
                int(detail["left_path_ordinal"]),
                int(detail["right_path_ordinal"]),
            )
        ]

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pageir-vs-frozen-ts-grouping-parity",
        "source": {
            "python_source_name": python_snapshot.get("source_name", ""),
            "ts_source_name": frozen_ts_snapshot.get("source_name", ""),
            "page_number": python_snapshot.get("page_number"),
        },
        "summary": {
            "path_count": path_count,
            "python_operation_count": python_snapshot.get("operation_count"),
            "ts_scene_operation_count": frozen_ts_snapshot.get("scene_operation_count"),
            "python_group_count": len(python_groups),
            "ts_group_count": len(ts_groups),
            "group_count_delta_ts_minus_python": len(ts_groups) - len(python_groups),
        },
        "path_ordinal_mapping": {
            "mode": "path_ordinal",
            "entry_count": path_count,
            "bijection": True,
            "mapping_digest": mapping_digest,
            "paint_order_used_as_identity": False,
        },
        "group_count_reconciliation": {
            "python_path_bearing_group_count": python_path_groups,
            "ts_path_bearing_group_count": ts_path_groups,
            "path_bearing_group_delta": ts_path_groups - python_path_groups,
            "python_non_path_group_count": python_non_path_groups,
            "ts_non_path_group_count": ts_non_path_groups,
            "non_path_group_delta": ts_non_path_groups - python_non_path_groups,
            "python_pure_text_group_count": python_pure_text_count,
            "ts_pure_text_group_count": ts_pure_text_count,
            "pure_text_group_delta": ts_pure_text_count - python_pure_text_count,
            "other_non_path_group_delta": (
                ts_other_non_path_count - python_other_non_path_count
            ),
            "reconciled_delta": (
                ts_path_groups - python_path_groups
                + ts_non_path_groups - python_non_path_groups
            ),
        },
        "path_partition": {
            "python_boundary_count": len(python_boundaries),
            "ts_boundary_count": len(ts_boundaries),
            "common_boundary_count": len(python_boundaries & ts_boundaries),
            "python_only_boundary_ranges": _consecutive_ranges(
                python_boundaries - ts_boundaries
            ),
            "ts_only_boundary_ranges": _consecutive_ranges(
                ts_boundaries - python_boundaries
            ),
            "python_only_boundaries": python_only_details,
            "ts_only_boundaries": ts_only_details,
        },
        "atom_multiplicity": {
            "python_atom_count": sum(
                int(path["atom_multiplicity"]) for path in python_paths
            ),
            "ts_atom_count": sum(int(path["atom_multiplicity"]) for path in ts_paths),
            "matching_path_count": path_count - atom_mismatch_count,
            "mismatching_path_count": atom_mismatch_count,
            "pair_histogram": {
                f"ts:{ts_value}->python:{python_value}": count
                for (ts_value, python_value), count in sorted(atom_pairs.items())
            },
            "mismatch_segment_signature_histogram": dict(
                atom_segment_signatures.most_common()
            ),
            "mismatch_ranges": atom_ranges,
        },
        "text_batches": {
            "python_text_operation_count": python_snapshot.get(
                "operation_kind_counts", {}
            ).get("text"),
            "ts_text_operation_count": frozen_ts_snapshot.get(
                "operation_kind_counts", {}
            ).get("text"),
            "python_distinct_text_paint_batch_count": python_snapshot.get(
                "distinct_text_paint_batch_count"
            ),
            "ts_distinct_text_paint_batch_count": frozen_ts_snapshot.get(
                "distinct_text_paint_batch_count"
            ),
            "python_text_paint_batch_size_histogram": python_snapshot.get(
                "text_paint_batch_size_histogram", {}
            ),
            "ts_text_paint_batch_size_histogram": frozen_ts_snapshot.get(
                "text_paint_batch_size_histogram", {}
            ),
            "python_pure_text_groups": python_pure_text,
            "ts_pure_text_groups": ts_pure_text,
        },
        "diagnosis": {
            "group_delta_fully_reconciled": (
                len(ts_groups) - len(python_groups)
                == ts_path_groups - python_path_groups
                + ts_non_path_groups - python_non_path_groups
            ),
            "group_delta_components": [
                {
                    "kind": "additional-ts-path-bearing-boundary",
                    "count": len(ts_only_boundary_positions),
                    "before_path_ordinals": ts_only_boundary_positions,
                },
                {
                    "kind": "additional-ts-pure-text-group",
                    "count": ts_pure_text_count - python_pure_text_count,
                    "ts_group_ids": [
                        str(group.get("group_id")) for group in ts_pure_text
                    ],
                },
                {
                    "kind": "additional-ts-other-non-path-group",
                    "count": ts_other_non_path_count - python_other_non_path_count,
                },
            ],
            "text_paint_granularity_equivalent": (
                python_text_batches == ts_text_batches
            ),
            "atom_multiplicity_equivalent": atom_mismatch_count == 0,
            "grouping_thresholds_changed": False,
        },
    }


__all__ = [
    "SCHEMA_VERSION",
    "compare_grouping_snapshots",
    "describe_python_grouping",
]
