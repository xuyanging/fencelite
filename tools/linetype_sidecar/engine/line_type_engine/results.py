"""Validated Python form of the viewer's algorithm-neutral result contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .versions import RESULT_SCHEMA_VERSION


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be a string")
    return value


def _indices(values: Any, label: str) -> tuple[int, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    result = tuple(sorted({_integer(value, label) for value in values}))
    if len(result) != len(values):
        raise ValueError(f"{label} must not contain duplicate operation indices")
    return result


@dataclass(frozen=True, slots=True)
class LocalLineType:
    type_id: str
    display_name: str
    line_type_index: int
    atom_count: int
    op_indices: tuple[int, ...]
    model: str
    shape: str
    shape_detail: str
    is_periodic: bool = True

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalLineType":
        if value.get("is_periodic") is not True:
            raise ValueError("local line type must be periodic")
        return cls(
            _string(value.get("type_id"), "type_id"),
            _string(value.get("display_name"), "display_name"),
            _integer(value.get("line_type_index"), "line_type_index", minimum=1),
            _integer(value.get("atom_count"), "atom_count"),
            _indices(value.get("op_indices"), "op_indices"),
            _string(value.get("model"), "model", allow_empty=True),
            _string(value.get("shape"), "shape", allow_empty=True),
            _string(value.get("shape_detail"), "shape_detail", allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type_id": self.type_id,
            "display_name": self.display_name,
            "line_type_index": self.line_type_index,
            "atom_count": self.atom_count,
            "op_indices": list(self.op_indices),
            "model": self.model,
            "shape": self.shape,
            "shape_detail": self.shape_detail,
            "is_periodic": True,
        }


@dataclass(frozen=True, slots=True)
class NonLineType:
    atom_count: int
    op_indices: tuple[int, ...]
    display_name: str = "非线型"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "NonLineType":
        return cls(
            _integer(value.get("atom_count"), "non_linetype.atom_count"),
            _indices(value.get("op_indices"), "non_linetype.op_indices"),
            _string(value.get("display_name"), "non_linetype.display_name"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "display_name": self.display_name,
            "atom_count": self.atom_count,
            "op_indices": list(self.op_indices),
        }


@dataclass(frozen=True, slots=True)
class RecognizedGroup:
    group_id: str
    atom_count: int
    line_types: tuple[LocalLineType, ...]
    non_linetype: NonLineType

    @property
    def line_type_count(self) -> int:
        return len(self.line_types)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecognizedGroup":
        raw_types = value.get("line_types")
        if not isinstance(raw_types, list):
            raise ValueError("group.line_types must be an array")
        result = cls(
            _string(value.get("group_id"), "group_id"),
            _integer(value.get("atom_count"), "group.atom_count"),
            tuple(LocalLineType.from_dict(item) for item in raw_types),
            NonLineType.from_dict(value.get("non_linetype", {})),
        )
        if value.get("line_type_count") != result.line_type_count:
            raise ValueError("group.line_type_count does not match line_types")
        # Operation ownership is intentionally not disjoint here. One PDF path
        # operation may contain several independently classified drawable
        # subpath atoms; two local types can therefore reference the same
        # op_index while owning different atoms. The Method1 serializer
        # boundary validates the atom partition with multiplicity. This
        # algorithm-neutral result cannot infer atom overlap from op ids alone.
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "atom_count": self.atom_count,
            "line_type_count": self.line_type_count,
            "line_types": [item.to_dict() for item in self.line_types],
            "non_linetype": self.non_linetype.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GlobalLineTypeMember:
    case_id: str
    type_id: str
    display_name: str
    atom_count: int
    shape: str
    model: str | None = None
    shape_detail: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlobalLineTypeMember":
        model = value.get("model")
        detail = value.get("shape_detail")
        return cls(
            _string(value.get("case_id"), "member.case_id"),
            _string(value.get("type_id"), "member.type_id"),
            _string(value.get("display_name"), "member.display_name", allow_empty=True),
            _integer(value.get("atom_count"), "member.atom_count"),
            _string(value.get("shape"), "member.shape", allow_empty=True),
            None if model is None else _string(model, "member.model", allow_empty=True),
            None if detail is None else _string(detail, "member.shape_detail", allow_empty=True),
        )

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "case_id": self.case_id,
            "type_id": self.type_id,
            "display_name": self.display_name,
            "atom_count": self.atom_count,
            "shape": self.shape,
        }
        if self.model is not None:
            output["model"] = self.model
        if self.shape_detail is not None:
            output["shape_detail"] = self.shape_detail
        return output


@dataclass(frozen=True, slots=True)
class GlobalLineType:
    global_type_id: str
    signature_family: str
    minimum_pair_similarity: float
    op_indices: tuple[int, ...]
    members: tuple[GlobalLineTypeMember, ...]
    recognition_source: str | None = None
    source_global_type_id: str | None = None
    type_uid: str | None = None

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def group_count(self) -> int:
        return len({member.case_id for member in self.members})

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GlobalLineType":
        raw_members = value.get("members")
        if not isinstance(raw_members, list):
            raise ValueError("global type members must be an array")
        similarity = value.get("minimum_pair_similarity")
        if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
            raise ValueError("minimum_pair_similarity must be numeric")
        result = cls(
            _string(value.get("global_type_id"), "global_type_id"),
            _string(value.get("signature_family"), "signature_family", allow_empty=True),
            float(similarity),
            _indices(value.get("op_indices"), "global_type.op_indices"),
            tuple(GlobalLineTypeMember.from_dict(item) for item in raw_members),
            value.get("recognition_source"),
            value.get("source_global_type_id"),
            value.get("type_uid"),
        )
        if value.get("member_count") != result.member_count:
            raise ValueError("global type member_count does not match members")
        if value.get("group_count") != result.group_count:
            raise ValueError("global type group_count does not match members")
        return result

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "global_type_id": self.global_type_id,
            "signature_family": self.signature_family,
            "member_count": self.member_count,
            "group_count": self.group_count,
            "minimum_pair_similarity": self.minimum_pair_similarity,
            "op_indices": list(self.op_indices),
            "members": [member.to_dict() for member in self.members],
        }
        for key in ("recognition_source", "source_global_type_id", "type_uid"):
            item = getattr(self, key)
            if item is not None:
                output[key] = item
        return output


@dataclass(frozen=True, slots=True)
class RecognitionSummary:
    input_group_count: int
    processed_group_count: int
    local_line_type_count: int
    signed_periodic_type_count: int
    unsigned_periodic_type_count: int
    global_type_count: int
    cross_group_global_type_count: int
    seedless_skipped_op_count: int | None = None
    discarded_route_op_count: int | None = None
    reclaimed_op_count: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecognitionSummary":
        required = (
            "input_group_count", "processed_group_count", "local_line_type_count",
            "signed_periodic_type_count", "unsigned_periodic_type_count",
            "global_type_count", "cross_group_global_type_count",
        )
        optional = (
            "seedless_skipped_op_count", "discarded_route_op_count", "reclaimed_op_count",
        )
        return cls(
            *(_integer(value.get(key), f"summary.{key}") for key in required),
            *(None if value.get(key) is None else _integer(value[key], f"summary.{key}")
              for key in optional),
        )

    def to_dict(self) -> dict[str, Any]:
        output = {
            "input_group_count": self.input_group_count,
            "processed_group_count": self.processed_group_count,
            "local_line_type_count": self.local_line_type_count,
            "signed_periodic_type_count": self.signed_periodic_type_count,
            "unsigned_periodic_type_count": self.unsigned_periodic_type_count,
            "global_type_count": self.global_type_count,
            "cross_group_global_type_count": self.cross_group_global_type_count,
        }
        for key in (
            "seedless_skipped_op_count", "discarded_route_op_count", "reclaimed_op_count",
        ):
            value = getattr(self, key)
            if value is not None:
                output[key] = value
        return output


@dataclass(frozen=True, slots=True)
class RecognitionError:
    group_id: str
    message: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecognitionError":
        return cls(
            _string(value.get("group_id"), "error.group_id", allow_empty=True),
            _string(value.get("message"), "error.message"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"group_id": self.group_id, "message": self.message}


@dataclass(frozen=True, slots=True)
class LineTypeRecognitionResult:
    groups: tuple[RecognizedGroup, ...]
    global_types: tuple[GlobalLineType, ...]
    summary: RecognitionSummary
    errors: tuple[RecognitionError, ...] = ()
    schema_version: int = RESULT_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LineTypeRecognitionResult":
        if value.get("schema_version") != RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported line-type result schema")
        raw_groups = value.get("groups")
        raw_global_types = value.get("global_types")
        raw_errors = value.get("errors")
        if not all(isinstance(item, list) for item in (raw_groups, raw_global_types, raw_errors)):
            raise ValueError("result groups, global_types and errors must be arrays")
        result = cls(
            tuple(RecognizedGroup.from_dict(item) for item in raw_groups),
            tuple(GlobalLineType.from_dict(item) for item in raw_global_types),
            RecognitionSummary.from_dict(value.get("summary", {})),
            tuple(RecognitionError.from_dict(item) for item in raw_errors),
        )
        if result.summary.global_type_count != len(result.global_types):
            raise ValueError("summary.global_type_count does not match global_types")
        if result.summary.local_line_type_count != sum(
            group.line_type_count for group in result.groups
        ):
            raise ValueError("summary.local_line_type_count does not match groups")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "groups": [group.to_dict() for group in self.groups],
            "global_types": [item.to_dict() for item in self.global_types],
            "summary": self.summary.to_dict(),
            "errors": [error.to_dict() for error in self.errors],
        }
