"""Pure composition of independent Method 1 and Method 2 results.

This module is a Python parity port of the frozen TypeScript fusion policy in
``line-type-engine/fusion``.  It does not invoke either recognizer and it never
mutates their results.  Method 2 owns every operation it recognized; if any
operation of a Method 1 global type overlaps Method 2, that whole Method 1
global type is omitted from the fused display result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cmp_to_key
import json
import math
from typing import Any, Literal, Mapping, Sequence

from .results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    NonLineType,
    RecognitionSummary,
    RecognizedGroup,
)


RecognitionSource = Literal["method1", "method2"]
_METHOD1: Literal["method1"] = "method1"
_METHOD2: Literal["method2"] = "method2"


@dataclass(frozen=True, slots=True)
class LineTypeFusionAudit:
    method2_global_type_count: int
    method2_op_count: int
    retained_method1_global_type_count: int
    retained_method1_op_count: int
    skipped_duplicate_method1_global_type_count: int
    skipped_duplicate_method1_op_count: int
    duplicate_overlap_op_count: int
    skipped_method1_global_type_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "method2_global_type_count": self.method2_global_type_count,
            "method2_op_count": self.method2_op_count,
            "retained_method1_global_type_count": (
                self.retained_method1_global_type_count
            ),
            "retained_method1_op_count": self.retained_method1_op_count,
            "skipped_duplicate_method1_global_type_count": (
                self.skipped_duplicate_method1_global_type_count
            ),
            "skipped_duplicate_method1_op_count": (
                self.skipped_duplicate_method1_op_count
            ),
            "duplicate_overlap_op_count": self.duplicate_overlap_op_count,
            "skipped_method1_global_type_ids": list(
                self.skipped_method1_global_type_ids
            ),
        }


@dataclass(frozen=True, slots=True)
class FusedRecognition:
    result: LineTypeRecognitionResult
    audit: LineTypeFusionAudit

    def to_dict(self) -> dict[str, Any]:
        return {"result": self.result.to_dict(), "audit": self.audit.to_dict()}


@dataclass(frozen=True, slots=True)
class LineTypeResultDiff:
    exact_match: bool
    changed_group_ids: tuple[str, ...]
    changed_global_type_count: bool
    changed_line_type_count: int
    changed_op_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_match": self.exact_match,
            "changed_group_ids": list(self.changed_group_ids),
            "changed_global_type_count": self.changed_global_type_count,
            "changed_line_type_count": self.changed_line_type_count,
            "changed_op_count": self.changed_op_count,
        }


def _json_string(value: str) -> str:
    # JSON.stringify emits Unicode characters rather than ASCII escape pairs.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _javascript_number(value: int | float) -> str:
    """Serialize the result-contract numeric subset like JSON.stringify.

    Operation indices and counters are integers.  Integral floats occur in a
    few diagnostics and JavaScript writes them without a trailing ``.0``.
    The fallback is intentionally small because stable ids never include the
    recognizers' floating-point scores.
    """

    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "null"
    if value == 0:
        return "0"
    if value.is_integer() and abs(value) < 1e21:
        return str(int(value))
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    # JavaScript omits the optional plus-sign leading zero in small exponents.
    if "e" in encoded or "E" in encoded:
        mantissa, exponent = encoded.lower().split("e", 1)
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        return f"{mantissa}e{sign}{exponent}"
    return encoded


def stable_json(value: Any) -> str:
    """Return the deterministic JSON form used by the frozen TS contract."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("stable_json object keys must be strings")
        fields = (
            f"{_json_string(key)}:{stable_json(value[key])}"
            for key in sorted(value)
        )
        return "{" + ",".join(fields) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(stable_json(item) for item in value) + "]"
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _json_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _javascript_number(value)
    if isinstance(value, float):
        return _javascript_number(value)
    raise TypeError(f"stable_json does not support {type(value).__name__}")


def fnv1a64(value: str) -> str:
    """Hash JavaScript UTF-16 code units with unsigned 64-bit FNV-1a."""

    if not isinstance(value, str):
        raise TypeError("fnv1a64 value must be a string")
    payload = value.encode("utf-16-le", errors="surrogatepass")
    result = 0xCBF29CE484222325
    for offset in range(0, len(payload), 2):
        code_unit = payload[offset] | (payload[offset + 1] << 8)
        result ^= code_unit
        result = (result * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{result:016x}"


def _sorted_unique(values: Sequence[int]) -> tuple[int, ...]:
    return tuple(sorted(set(values)))


def _member_key(group_id: str, type_id: str) -> str:
    return f"{group_id}\0{type_id}"


def _namespaced_type_id(source: RecognitionSource, type_id: str) -> str:
    return f"{source}__{type_id}"


def _source_members(global_types: Sequence[GlobalLineType]) -> frozenset[str]:
    return frozenset(
        _member_key(str(member.case_id), member.type_id)
        for global_type in global_types
        for member in global_type.members
    )


def _unique_method1_support_locals(
    groups: Sequence[RecognizedGroup],
    all_global_types: Sequence[GlobalLineType],
    retained_global_types: Sequence[GlobalLineType],
) -> frozenset[str]:
    """Return carrier-support locals proven to belong to one retained global.

    Carrier companions extend a compound global's drawable coverage without
    becoming signature members.  Normal fusion projects locals through
    ``GlobalLineType.members``, so these deliberately memberless support
    records need one narrow projection rule.  Every support operation must be
    owned by the same single retained Method1 global; missing or competing
    ownership fails closed and leaves the local out of the fused line types.
    """

    retained_objects = {id(global_type) for global_type in retained_global_types}
    retained_positions = {
        position
        for position, global_type in enumerate(all_global_types)
        if id(global_type) in retained_objects
        and global_type.signature_family == "compound_path_periodic"
    }
    owners_by_op: dict[int, set[int]] = {}
    for global_position, global_type in enumerate(all_global_types):
        for op_index in global_type.op_indices:
            # Check against every Method1 global, including ones skipped by
            # Method2.  Otherwise an op shared with a skipped source could look
            # uniquely owned after that source was removed.  Tuple position is
            # used rather than the external id so malformed duplicate ids
            # cannot make two globals look like one owner.
            owners_by_op.setdefault(op_index, set()).add(global_position)

    support_keys: set[str] = set()
    for group in groups:
        for line_type in group.line_types:
            if (
                line_type.model != "parallel_carrier_companion"
                or not line_type.op_indices
            ):
                continue
            owners = tuple(
                owners_by_op.get(op_index, set())
                for op_index in line_type.op_indices
            )
            if (
                all(len(op_owners) == 1 for op_owners in owners)
                and len({next(iter(op_owners)) for op_owners in owners}) == 1
                and next(iter(owners[0])) in retained_positions
            ):
                support_keys.add(_member_key(group.group_id, line_type.type_id))
    return frozenset(support_keys)


def _clone_local_type(
    line_type: LocalLineType,
    source: RecognitionSource,
    line_type_index: int,
) -> LocalLineType:
    return replace(
        line_type,
        type_id=_namespaced_type_id(source, line_type.type_id),
        line_type_index=line_type_index,
        op_indices=_sorted_unique(line_type.op_indices),
    )


def _all_known_group_ops(*groups: RecognizedGroup | None) -> tuple[int, ...]:
    return _sorted_unique(
        tuple(
            op_index
            for group in groups
            if group is not None
            for line_type in group.line_types
            for op_index in line_type.op_indices
        )
        + tuple(
            op_index
            for group in groups
            if group is not None
            for op_index in group.non_linetype.op_indices
        )
    )


def _group_id_compare(left: str, right: str) -> int:
    # Mirrors Number(left) - Number(right) || left.localeCompare(right) for the
    # numeric Group ids emitted by the page segmenter, with a deterministic
    # lexical fallback for external fixtures.
    try:
        delta = float(left) - float(right)
    except ValueError:
        delta = math.nan
    if math.isfinite(delta) and delta != 0:
        return -1 if delta < 0 else 1
    return (left > right) - (left < right)


def _namespaced_member(
    member: GlobalLineTypeMember, source: RecognitionSource
) -> GlobalLineTypeMember:
    return replace(member, type_id=_namespaced_type_id(source, member.type_id))


def fuse_line_type_recognition_results(
    method1: LineTypeRecognitionResult,
    method2: LineTypeRecognitionResult,
) -> FusedRecognition:
    """Compose two complete recognizer results under Method2 ownership."""

    method2_owned_ops = {
        op_index
        for global_type in method2.global_types
        for op_index in global_type.op_indices
    }
    skipped_method1: list[GlobalLineType] = []
    retained_method1: list[GlobalLineType] = []
    for global_type in method1.global_types:
        if any(op_index in method2_owned_ops for op_index in global_type.op_indices):
            skipped_method1.append(global_type)
        else:
            retained_method1.append(global_type)

    retained_method1_members = _source_members(retained_method1)
    retained_method1_locals = (
        retained_method1_members
        | _unique_method1_support_locals(
            method1.groups,
            method1.global_types,
            retained_method1,
        )
    )
    method2_members = _source_members(method2.global_types)
    method1_groups = {str(group.group_id): group for group in method1.groups}
    method2_groups = {str(group.group_id): group for group in method2.groups}
    group_ids = sorted(
        set(method1_groups) | set(method2_groups),
        key=cmp_to_key(_group_id_compare),
    )

    global_specs: list[tuple[RecognitionSource, GlobalLineType]] = [
        *[(_METHOD2, global_type) for global_type in method2.global_types],
        *[(_METHOD1, global_type) for global_type in retained_method1],
    ]
    global_types = tuple(
        replace(
            global_type,
            global_type_id=f"global_type_{index:03d}",
            recognition_source=source,
            source_global_type_id=global_type.global_type_id,
            type_uid=None,
            op_indices=_sorted_unique(global_type.op_indices),
            members=tuple(
                _namespaced_member(member, source)
                for member in global_type.members
            ),
        )
        for index, (source, global_type) in enumerate(global_specs, start=1)
    )

    groups: list[RecognizedGroup] = []
    for group_id in group_ids:
        method1_group = method1_groups.get(group_id)
        method2_group = method2_groups.get(group_id)
        assigned: set[int] = set()
        line_types: list[LocalLineType] = []

        def append(
            source: RecognitionSource,
            group: RecognizedGroup | None,
            allowed_members: frozenset[str],
        ) -> None:
            if group is None:
                return
            for line_type in group.line_types:
                if _member_key(group_id, line_type.type_id) not in allowed_members:
                    continue
                clone = _clone_local_type(line_type, source, len(line_types) + 1)
                assigned.update(clone.op_indices)
                line_types.append(clone)

        append(_METHOD2, method2_group, method2_members)
        append(_METHOD1, method1_group, retained_method1_locals)
        known_ops = _all_known_group_ops(method1_group, method2_group)
        non_line_ops = tuple(
            op_index for op_index in known_ops if op_index not in assigned
        )
        if method1_group is not None:
            atom_count = method1_group.atom_count
        elif method2_group is not None:
            atom_count = method2_group.atom_count
        else:  # pragma: no cover - group_ids is formed from the two maps.
            atom_count = len(known_ops)
        line_atom_count = sum(item.atom_count for item in line_types)
        groups.append(
            RecognizedGroup(
                group_id=group_id,
                atom_count=atom_count,
                line_types=tuple(line_types),
                non_linetype=NonLineType(
                    atom_count=max(0, atom_count - line_atom_count),
                    op_indices=non_line_ops,
                ),
            )
        )

    local_line_type_count = sum(group.line_type_count for group in groups)
    result = LineTypeRecognitionResult(
        groups=tuple(groups),
        global_types=global_types,
        summary=RecognitionSummary(
            input_group_count=max(
                method1.summary.input_group_count,
                method2.summary.input_group_count,
                len(groups),
            ),
            processed_group_count=max(
                method1.summary.processed_group_count,
                method2.summary.processed_group_count,
                len(groups),
            ),
            local_line_type_count=local_line_type_count,
            signed_periodic_type_count=local_line_type_count,
            unsigned_periodic_type_count=0,
            global_type_count=len(global_types),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in global_types
            ),
        ),
        errors=method1.errors + method2.errors,
    )

    skipped_method1_ops = {
        op_index
        for global_type in skipped_method1
        for op_index in global_type.op_indices
    }
    retained_method1_ops = {
        op_index
        for global_type in retained_method1
        for op_index in global_type.op_indices
    }
    audit = LineTypeFusionAudit(
        method2_global_type_count=len(method2.global_types),
        method2_op_count=len(method2_owned_ops),
        retained_method1_global_type_count=len(retained_method1),
        retained_method1_op_count=len(retained_method1_ops),
        skipped_duplicate_method1_global_type_count=len(skipped_method1),
        skipped_duplicate_method1_op_count=len(skipped_method1_ops),
        duplicate_overlap_op_count=len(skipped_method1_ops & method2_owned_ops),
        skipped_method1_global_type_ids=tuple(
            global_type.global_type_id for global_type in skipped_method1
        ),
    )
    return FusedRecognition(result=result, audit=audit)


def _sorted_operation_key(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in sorted(set(values)))


def _comparable_group(group: RecognizedGroup | None) -> dict[str, Any] | None:
    if group is None:
        return None
    return {
        "group_id": group.group_id,
        "line_types": sorted(
            _sorted_operation_key(line_type.op_indices)
            for line_type in group.line_types
        ),
        "non_linetype": _sorted_operation_key(group.non_linetype.op_indices),
    }


def _comparable_result(result: LineTypeRecognitionResult) -> dict[str, Any]:
    return {
        "groups": sorted(
            (_comparable_group(group) for group in result.groups),
            key=lambda group: str(group["group_id"]),  # type: ignore[index]
        ),
        "global_types": sorted(
            _sorted_operation_key(global_type.op_indices)
            for global_type in result.global_types
        ),
    }


def _local_membership_by_operation(
    result: LineTypeRecognitionResult,
) -> dict[str, str]:
    membership: dict[str, str] = {}
    for group in result.groups:
        for line_type in group.line_types:
            partition = _sorted_operation_key(line_type.op_indices)
            for op_index in line_type.op_indices:
                membership[f"{group.group_id}:{op_index}"] = partition
    return membership


def compare_line_type_results(
    first: LineTypeRecognitionResult,
    second: LineTypeRecognitionResult,
) -> LineTypeResultDiff:
    """Compare geometry partitions while ignoring generated ids and sources."""

    first_groups = {group.group_id: group for group in first.groups}
    second_groups = {group.group_id: group for group in second.groups}
    all_group_ids = sorted(set(first_groups) | set(second_groups))
    changed_group_ids = tuple(
        group_id
        for group_id in all_group_ids
        if stable_json(_comparable_group(first_groups.get(group_id)))
        != stable_json(_comparable_group(second_groups.get(group_id)))
    )
    first_line_ops = _local_membership_by_operation(first)
    second_line_ops = _local_membership_by_operation(second)
    changed_op_count = sum(
        first_line_ops.get(key) != second_line_ops.get(key)
        for key in set(first_line_ops) | set(second_line_ops)
    )
    changed_line_type_count = sum(
        abs(
            (first_groups.get(group_id).line_type_count if group_id in first_groups else 0)
            - (
                second_groups.get(group_id).line_type_count
                if group_id in second_groups
                else 0
            )
        )
        for group_id in all_group_ids
    )
    return LineTypeResultDiff(
        exact_match=(
            stable_json(_comparable_result(first))
            == stable_json(_comparable_result(second))
        ),
        changed_group_ids=changed_group_ids,
        changed_global_type_count=(
            len(first.global_types) != len(second.global_types)
        ),
        changed_line_type_count=changed_line_type_count,
        changed_op_count=changed_op_count,
    )


def attach_stable_type_uids(
    result: LineTypeRecognitionResult,
    page_identity: str,
) -> LineTypeRecognitionResult:
    """Attach the same geometry-derived global type uid as the TS contract."""

    if not isinstance(page_identity, str):
        raise TypeError("page_identity must be a string")
    global_types: list[GlobalLineType] = []
    for global_type in result.global_types:
        members = [
            {
                "case_id": member.case_id,
                "model": member.model or "",
                "shape": member.shape,
                "shape_detail": member.shape_detail or "",
            }
            for member in global_type.members
        ]
        members.sort(key=stable_json)
        descriptor = stable_json(
            {
                "page": page_identity,
                "family": global_type.signature_family,
                "members": members,
                "op_indices": sorted(global_type.op_indices),
            }
        )
        global_types.append(
            replace(global_type, type_uid=f"lt2_{fnv1a64(descriptor)}")
        )
    return replace(result, global_types=tuple(global_types))


__all__ = [
    "FusedRecognition",
    "LineTypeFusionAudit",
    "LineTypeResultDiff",
    "attach_stable_type_uids",
    "compare_line_type_results",
    "fnv1a64",
    "fuse_line_type_recognition_results",
    "stable_json",
]
