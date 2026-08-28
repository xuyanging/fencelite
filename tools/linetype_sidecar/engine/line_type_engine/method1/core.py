"""Package API for the validated Method1 Python base recognizer.

This is the algorithm formerly embedded in ``line-type-service-bridge.py``.
It accepts renderer-neutral serialized Group commands and returns the stable
schema-1 recognition result.  NDJSON, HTTP and process lifecycle concerns are
kept in adapters and do not live in the algorithm package.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import classify_line_shape as shape_classifier
from . import match_line_types_across_groups as cross_group_matcher
from . import unknown_pattern_split as pattern_splitter


ULTRA_DENSE_NON_LINETYPE_ATOM_COUNT = 1_000_000



def _error_text(error: object) -> str:
    """异常 -> **非空**消息文本。

    RecognitionError.message 走的是 results.py 的 _string(allow_empty=False)，
    而 `str(exc)` 在异常不带参数时是空串（`str(ValueError())` == ""）。于是
    「某个 group 失败」会被往返校验变成 `ValueError: error.message must be a
    string`，真实错因彻底丢失 —— 实测 taylor_3_12 P122 整页因此报废，日志里
    只剩这句毫无信息量的话。所以这里保底带上异常类名。
    """
    text = str(error).strip()
    name = type(error).__name__ if isinstance(error, BaseException) else "error"
    return f"{name}: {text}" if text else name

def _ordered_unique(values: Iterable[int]) -> list[int]:
    output: list[int] = []
    previous: int | None = None
    for raw_value in values:
        value = int(raw_value)
        if value != previous:
            output.append(value)
            previous = value
    return output


def _op_indices(
    atom_ids: Iterable[int],
    atom_op_indices: list[int],
) -> list[int]:
    return sorted({
        atom_op_indices[atom_id]
        for atom_id in atom_ids
        if 0 <= atom_id < len(atom_op_indices)
    })


def _compact_line_type(
    record: Mapping[str, Any],
    atom_ids: list[int],
    atom_op_indices: list[int],
    *,
    include_signature: bool,
) -> dict[str, Any]:
    compact = {
        "type_id": record["type_id"],
        "display_name": record["display_name"],
        "line_type_index": record["line_type_index"],
        "atom_count": record["atom_count"],
        "op_indices": _op_indices(atom_ids, atom_op_indices),
        "model": record.get("model"),
        "shape": record.get("shape"),
        "shape_detail": record.get("shape_detail"),
        "is_periodic": bool(record["is_periodic"]),
    }
    if include_signature:
        compact["line_type_signature"] = record.get("line_type_signature")
    return compact


def analyze_group(group: Mapping[str, Any]) -> dict[str, Any]:
    """Classify one already ordered Group without transport side effects."""

    group_id = str(group.get("group_id", ""))
    commands = str(group.get("commands", ""))
    atom_op_indices = [int(value) for value in group.get("atom_op_indices", [])]
    if (
        group.get("force_non_linetype") == "dense_2d_layer"
        or len(atom_op_indices) >= ULTRA_DENSE_NON_LINETYPE_ATOM_COUNT
    ):
        return {
            "case_id": group_id,
            "case_file": f"group_{group_id}",
            "group_id": group_id,
            "atom_count": len(atom_op_indices),
            "line_type_count": 0,
            "line_types": [],
            "non_linetype": {
                "display_name": "非线型",
                "atom_count": len(atom_op_indices),
                "op_indices": _ordered_unique(atom_op_indices),
            },
        }

    atoms = pattern_splitter.parse_painted_atoms(commands)
    if len(atoms) != len(atom_op_indices):
        raise ValueError(
            f"group {group_id}: serialized {len(atom_op_indices)} subpaths "
            f"but recognizer parsed {len(atoms)} atoms"
        )

    discovery = pattern_splitter.discover_unknown_pattern_types(atoms)
    atom_by_id = {atom.id: atom for atom in atoms}
    periodic_atom_ids = {
        atom_id
        for pattern_type in discovery.types
        if pattern_type.kind != "residual_geometry_cluster"
        for atom_id in pattern_type.atom_ids
    }
    records: list[dict[str, Any]] = []
    line_type_index = 0
    for pattern_type in discovery.types:
        if pattern_type.kind == "residual_geometry_cluster":
            described: dict[str, Any] = {
                "type_id": pattern_type.type_id,
                "atom_count": len(pattern_type.atom_ids),
                "is_periodic": False,
                "line_type_signature": None,
                "model": "residual_geometry_cluster",
                "shape": "非线型",
                "shape_detail": "未发现足够线型证据",
            }
        else:
            outside_atoms = [
                atom
                for atom in atoms
                if atom.id not in pattern_type.atom_ids
                and atom.id not in periodic_atom_ids
            ]
            described = shape_classifier.describe_pattern_type(
                pattern_type,
                atom_by_id,
                outside_atoms,
            )
        atom_ids = sorted(pattern_type.atom_ids)
        if described["is_periodic"]:
            line_type_index += 1
            described["display_name"] = f"线型{line_type_index}"
            described["line_type_index"] = line_type_index
        else:
            described["display_name"] = "非线型"
            described["line_type_index"] = None
        records.append(_compact_line_type(
            described,
            atom_ids,
            atom_op_indices,
            include_signature=True,
        ))

    line_types = [record for record in records if record["is_periodic"]]
    non_linetype = [record for record in records if not record["is_periodic"]]
    return {
        "case_id": group_id,
        "case_file": f"group_{group_id}",
        "group_id": group_id,
        "atom_count": len(atoms),
        "line_type_count": len(line_types),
        "line_types": line_types,
        "non_linetype": {
            "display_name": "非线型",
            "atom_count": sum(record["atom_count"] for record in non_linetype),
            "op_indices": sorted({
                op_index
                for record in non_linetype
                for op_index in record["op_indices"]
            }),
        },
    }


def finalize_registry(
    groups: list[dict[str, Any]],
    input_group_count: int,
    errors: list[dict[str, str]],
) -> dict[str, Any]:
    """Run complete-link cross-Group identity matching and public projection."""

    registry = cross_group_matcher.global_linetype_registry({"cases": groups})
    local_lookup = {
        (str(group["group_id"]), record["type_id"]): record
        for group in groups
        for record in group["line_types"]
    }
    for global_type in registry["global_types"]:
        op_indices: set[int] = set()
        for member in global_type["members"]:
            local = local_lookup.get((
                str(member.get("case_id", "")),
                member.get("type_id"),
            ))
            if local:
                op_indices.update(local["op_indices"])
        global_type["op_indices"] = sorted(op_indices)

    public_groups: list[dict[str, Any]] = []
    for group in groups:
        public_group = dict(group)
        public_group["line_types"] = [
            {key: value for key, value in record.items() if key != "line_type_signature"}
            for record in group["line_types"]
        ]
        public_groups.append(public_group)

    public_global_types = [
        {
            key: value
            for key, value in global_type.items()
            if key not in {"representative_signature", "pairwise_matches"}
        }
        for global_type in registry["global_types"]
    ]
    return {
        "schema_version": 1,
        "groups": public_groups,
        "global_types": public_global_types,
        "summary": {
            "input_group_count": input_group_count,
            "processed_group_count": len(groups),
            "local_line_type_count": sum(group["line_type_count"] for group in groups),
            **registry["summary"],
        },
        "errors": errors,
    }


def run(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute ``analyze``, ``registry`` or the full Method1 base pipeline."""

    mode = str(payload.get("mode", "full"))
    if mode == "registry":
        groups = list(payload.get("groups", []))
        return finalize_registry(
            groups,
            int(payload.get("input_group_count", len(groups))),
            list(payload.get("errors", [])),
        )
    if mode not in {"analyze", "full"}:
        raise ValueError(f"unsupported Method1 mode: {mode}")

    raw_groups = list(payload.get("groups", []))
    groups: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for raw_group in raw_groups:
        try:
            groups.append(analyze_group(raw_group))
        except Exception as error:  # One malformed Group must not hide the page.
            errors.append({
                "group_id": str(raw_group.get("group_id", "")),
                "message": _error_text(error),
            })
    if mode == "analyze":
        return {"schema_version": 1, "groups": groups, "errors": errors}
    return finalize_registry(groups, len(raw_groups), errors)
