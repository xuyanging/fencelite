"""Persistable contract for the pure Method1/Method2 display composition.

This module is the Python counterpart of the frozen TypeScript
``fusion/contract.ts`` and the pure part of ``fusion/compose.ts``.  It only
combines already-complete recognition results.  Importing or calling it never
runs either recognizer, touches a cache, or depends on a browser/transport.
The envelope records a Python candidate implementation identity separately
from the frozen TypeScript composition used as its target specification.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MappingProxyType
from typing import Callable, Literal, Mapping

from .fusion import (
    LineTypeFusionAudit,
    LineTypeResultDiff,
    attach_stable_type_uids,
    compare_line_type_results,
    fnv1a64,
    fuse_line_type_recognition_results,
    stable_json,
)
from .method2.contract import (
    LINE_TYPE_METHOD2_CONFIG_HASH,
    METHOD2_ENGINE_VERSION,
    METHOD2_LOCAL_PROJECTION_VERSION,
    METHOD2_TARGET_SPEC_VERSION,
    LineTypeMethod2Envelope,
    VectorTextFamilyAuditPayload,
    validate_line_type_method2_envelope,
)
from .results import LineTypeRecognitionResult, RecognitionSummary
from .versions import (
    FROZEN_TS_FUSION_POLICY_VERSION,
    FROZEN_TS_METHOD1_ENGINE_VERSION,
    PYTHON_FUSION_ENGINE_VERSION,
    PYTHON_METHOD1_ENGINE_VERSION,
)


AuditLevel = Literal["summary", "full"]

LINE_TYPE_METHOD1_ENGINE_VERSION = PYTHON_METHOD1_ENGINE_VERSION
LINE_TYPE_METHOD2_ENGINE_VERSION = METHOD2_ENGINE_VERSION
LINE_TYPE_METHOD1_TARGET_SPEC_VERSION = FROZEN_TS_METHOD1_ENGINE_VERSION
LINE_TYPE_METHOD2_TARGET_SPEC_VERSION = METHOD2_TARGET_SPEC_VERSION
LINE_TYPE_FUSION_POLICY_VERSION = FROZEN_TS_FUSION_POLICY_VERSION
FUSED_LINE_TYPE_TARGET_SPEC_VERSION = (
    "fused-method1-r10-method2-r46-policy-v1-2026-08-24"
)
FUSED_LINE_TYPE_ENGINE_VERSION = PYTHON_FUSION_ENGINE_VERSION
FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION = 4

_FEATURE_ITEMS = (
    ("method2_first_overlap_ownership", True),
    ("drop_whole_overlapping_method1_global_type", True),
    ("retain_disjoint_method1_global_types", True),
)
FUSED_LINE_TYPE_FEATURES: Mapping[str, bool] = MappingProxyType(
    dict(_FEATURE_ITEMS)
)


def _validated_feature_copy(
    features: Mapping[str, bool] = FUSED_LINE_TYPE_FEATURES,
) -> dict[str, bool]:
    feature_copy: dict[str, bool] = {}
    for name, enabled in features.items():
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise TypeError("fusion features must map strings to booleans")
        feature_copy[name] = enabled
    return feature_copy


def fused_line_type_config_hash(
    features: Mapping[str, bool] = FUSED_LINE_TYPE_FEATURES,
) -> str:
    """Return the Python candidate fusion configuration hash."""

    feature_copy = _validated_feature_copy(features)
    return fnv1a64(
        stable_json(
            {
                "engine_version": FUSED_LINE_TYPE_ENGINE_VERSION,
                "target_spec_version": FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
                "method1_engine_version": LINE_TYPE_METHOD1_ENGINE_VERSION,
                "method1_target_spec_version": (
                    LINE_TYPE_METHOD1_TARGET_SPEC_VERSION
                ),
                "method2_engine_version": LINE_TYPE_METHOD2_ENGINE_VERSION,
                "method2_target_spec_version": (
                    LINE_TYPE_METHOD2_TARGET_SPEC_VERSION
                ),
                "method2_local_projection_version": (
                    METHOD2_LOCAL_PROJECTION_VERSION
                ),
                "fusion_policy_version": LINE_TYPE_FUSION_POLICY_VERSION,
                "features": feature_copy,
            }
        )
    )


FUSED_LINE_TYPE_CONFIG_HASH = fused_line_type_config_hash()


def frozen_ts_fused_line_type_config_hash(
    features: Mapping[str, bool] = FUSED_LINE_TYPE_FEATURES,
) -> str:
    """Return the frozen TS config hash used only as a parity anchor."""

    return fnv1a64(
        stable_json(
            {
                "method1_engine_version": LINE_TYPE_METHOD1_TARGET_SPEC_VERSION,
                "method2_engine_version": LINE_TYPE_METHOD2_TARGET_SPEC_VERSION,
                "fusion_policy_version": LINE_TYPE_FUSION_POLICY_VERSION,
                "features": _validated_feature_copy(features),
            }
        )
    )


FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH = frozen_ts_fused_line_type_config_hash()

# Compatibility names mirror the historical TypeScript V2 exports while new
# Python callers can use the unambiguous FUSED_* names above.
LINE_TYPE_V2_ENGINE_VERSION = FUSED_LINE_TYPE_ENGINE_VERSION
LINE_TYPE_V2_RESULT_SCHEMA_VERSION = FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION
LINE_TYPE_V2_FEATURES = FUSED_LINE_TYPE_FEATURES
LINE_TYPE_V2_CONFIG_HASH = FUSED_LINE_TYPE_CONFIG_HASH


def fused_line_type_replay_input_hash(value: object) -> str:
    return fnv1a64(stable_json(value))


def _safe_nonnegative_integer(value: object, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 9_007_199_254_740_991
    ):
        raise ValueError(f"{label} must be a non-negative safe integer")
    return value


def _finite_nonnegative_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite non-negative number")
    return float(value)


def _finite_number(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_array(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain strings")
    return tuple(value)


def _index_array(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(_safe_nonnegative_integer(item, label) for item in value)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return value


def _validate_json_value(value: object, label: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > 9_007_199_254_740_991:
            raise ValueError(f"{label} contains an unsafe integer")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} contains a non-string object key")
            _validate_json_value(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains a non-JSON value")


def _validate_optional_string(
    value: Mapping[str, object], key: str, label: str
) -> None:
    if key in value and not isinstance(value[key], str):
        raise ValueError(f"{label}.{key} must be a string")


def _validate_family_diagnostic(value: Mapping[str, object], index: int) -> None:
    label = f"family_diagnostics[{index}]"
    _required_string(value.get("global_type_id"), f"{label}.global_type_id")
    if value.get("pattern_source") not in ("vector_strokes", "pdf_text"):
        raise ValueError(f"{label}.pattern_source is invalid")
    _finite_number(
        value.get("minimum_pair_similarity"),
        f"{label}.minimum_pair_similarity",
    )
    instance_count = _safe_nonnegative_integer(
        value.get("instance_count"), f"{label}.instance_count"
    )
    pair_count = _safe_nonnegative_integer(
        value.get("pair_count"), f"{label}.pair_count"
    )
    raw_instances = value.get("instances")
    raw_pairs = value.get("pairs")
    if not isinstance(raw_instances, (list, tuple)) or not isinstance(
        raw_pairs, (list, tuple)
    ):
        raise ValueError(f"{label}.instances and pairs must be arrays")
    # Large vector families retain only a bounded diagnostic sample, while
    # pair_count records the complete comparison count.
    if instance_count != len(raw_instances) or pair_count < len(raw_pairs):
        raise ValueError(f"{label} count fields do not match their arrays")
    for item_index, raw_instance in enumerate(raw_instances):
        instance = _mapping(
            raw_instance, f"{label}.instances[{item_index}]"
        )
        _safe_nonnegative_integer(
            instance.get("signature_index"),
            f"{label}.instances[{item_index}].signature_index",
        )
        if not isinstance(instance.get("display_label"), str):
            raise ValueError(
                f"{label}.instances[{item_index}].display_label must be a string"
            )
        _required_string(
            instance.get("group_id"),
            f"{label}.instances[{item_index}].group_id",
        )
        _index_array(
            instance.get("op_indices"),
            f"{label}.instances[{item_index}].op_indices",
        )
        _mapping(
            instance.get("dimensions"),
            f"{label}.instances[{item_index}].dimensions",
        )
        _validate_optional_string(
            instance, "literal_text", f"{label}.instances[{item_index}]"
        )
        _validate_json_value(instance, f"{label}.instances[{item_index}]")
    for pair_index, raw_pair in enumerate(raw_pairs):
        pair = _mapping(raw_pair, f"{label}.pairs[{pair_index}]")
        _validate_json_value(pair, f"{label}.pairs[{pair_index}]")
    _validate_optional_string(value, "literal_text", label)


def _validate_region_instance(value: Mapping[str, object], index: int) -> None:
    label = f"region_instances[{index}]"
    if not isinstance(value.get("display_label"), str):
        raise ValueError(f"{label}.display_label must be a string")
    for key in ("region_id", "group_id"):
        _required_string(value.get(key), f"{label}.{key}")
    _index_array(value.get("op_indices"), f"{label}.op_indices")
    bounds = _mapping(value.get("bounds"), f"{label}.bounds")
    bounds_values = {
        key: _finite_number(bounds.get(key), f"{label}.bounds.{key}")
        for key in ("minX", "minY", "maxX", "maxY")
    }
    if (
        bounds_values["maxX"] < bounds_values["minX"]
        or bounds_values["maxY"] < bounds_values["minY"]
    ):
        raise ValueError(f"{label}.bounds is inverted")
    _finite_number(value.get("orientation_degrees"), f"{label}.orientation")
    _finite_number(value.get("confidence"), f"{label}.confidence")
    evidence = _mapping(value.get("evidence"), f"{label}.evidence")
    for key in (
        "path_count",
        "segment_count",
        "angle_bin_count",
        "paint_order_span",
    ):
        _safe_nonnegative_integer(evidence.get(key), f"{label}.evidence.{key}")
    _finite_nonnegative_number(
        evidence.get("aspect_ratio"), f"{label}.evidence.aspect_ratio"
    )
    if "carrier_axis_degrees" in evidence:
        _finite_number(
            evidence["carrier_axis_degrees"],
            f"{label}.evidence.carrier_axis_degrees",
        )
    for key in (
        "single_stroke_shape_key",
        "sequential_multi_path_shape_key",
        "sequential_multi_path_chain_key",
    ):
        _validate_optional_string(evidence, key, f"{label}.evidence")
    for key in ("matched", "line_type_confirmed", "recovered"):
        if not isinstance(value.get(key), bool):
            raise ValueError(f"{label}.{key} must be boolean")
    if value.get("pattern_source") not in ("vector_strokes", "pdf_text"):
        raise ValueError(f"{label}.pattern_source is invalid")
    for key in ("literal_text", "text_family_id", "global_type_id"):
        _validate_optional_string(value, key, label)
    _validate_json_value(value, label)


def _validate_family_audit_deep(
    payload: VectorTextFamilyAuditPayload,
) -> None:
    for index, diagnostic in enumerate(payload.family_diagnostics):
        _validate_family_diagnostic(diagnostic, index)
    for index, region in enumerate(payload.region_instances):
        _validate_region_instance(region, index)


@dataclass(frozen=True, slots=True)
class FusedLineTypeStatuses:
    line_type: int
    non_linetype: int
    uncertain: int
    skipped: int

    @classmethod
    def from_value(cls, value: object) -> "FusedLineTypeStatuses":
        raw = _mapping(value, "fusion statuses")
        expected = {"line_type", "non_linetype", "uncertain", "skipped"}
        if set(raw) != expected:
            raise ValueError("fusion statuses fields do not match the contract")
        return cls(
            *(
                _safe_nonnegative_integer(raw[key], f"statuses.{key}")
                for key in ("line_type", "non_linetype", "uncertain", "skipped")
            )
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "line_type": self.line_type,
            "non_linetype": self.non_linetype,
            "uncertain": self.uncertain,
            "skipped": self.skipped,
        }


def _fusion_audit_from_value(value: object) -> LineTypeFusionAudit:
    if isinstance(value, LineTypeFusionAudit):
        raw: Mapping[str, object] = value.to_dict()
    else:
        raw = _mapping(value, "fusion rule audit")
    count_keys = (
        "method2_global_type_count",
        "method2_op_count",
        "retained_method1_global_type_count",
        "retained_method1_op_count",
        "skipped_duplicate_method1_global_type_count",
        "skipped_duplicate_method1_op_count",
        "duplicate_overlap_op_count",
    )
    counts = {
        key: _safe_nonnegative_integer(raw.get(key), f"fusion audit.{key}")
        for key in count_keys
    }
    skipped_ids = _string_array(
        raw.get("skipped_method1_global_type_ids"),
        "fusion audit.skipped_method1_global_type_ids",
    )
    return LineTypeFusionAudit(
        **counts,
        skipped_method1_global_type_ids=skipped_ids,
    )


@dataclass(frozen=True, slots=True)
class FusedLineTypeRules:
    repeated_vector_text_family_clustering: VectorTextFamilyAuditPayload
    method1_method2_display_fusion: LineTypeFusionAudit

    @classmethod
    def from_value(cls, value: object) -> "FusedLineTypeRules":
        raw = _mapping(value, "fusion audit rules")
        family = VectorTextFamilyAuditPayload.from_value(
            raw.get("repeated_vector_text_family_clustering")
        )
        _validate_family_audit_deep(family)
        return cls(
            family,
            _fusion_audit_from_value(
                raw.get("method1_method2_display_fusion")
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "repeated_vector_text_family_clustering": (
                self.repeated_vector_text_family_clustering.to_dict()
            ),
            "method1_method2_display_fusion": (
                self.method1_method2_display_fusion.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class FusedLineTypeTraceEntry:
    global_type_id: str
    type_uid: str
    member_type_ids: tuple[str, ...]

    @classmethod
    def from_value(cls, value: object) -> "FusedLineTypeTraceEntry":
        raw = _mapping(value, "fusion trace entry")
        return cls(
            _required_string(raw.get("global_type_id"), "trace.global_type_id"),
            _required_string(raw.get("type_uid"), "trace.type_uid"),
            _string_array(raw.get("member_type_ids"), "trace.member_type_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "global_type_id": self.global_type_id,
            "type_uid": self.type_uid,
            "member_type_ids": list(self.member_type_ids),
        }


def _comparison_from_value(value: object) -> LineTypeResultDiff:
    if isinstance(value, LineTypeResultDiff):
        raw: Mapping[str, object] = value.to_dict()
    else:
        raw = _mapping(value, "fusion comparison")
    exact_match = raw.get("exact_match")
    changed_global_type_count = raw.get("changed_global_type_count")
    if not isinstance(exact_match, bool) or not isinstance(
        changed_global_type_count, bool
    ):
        raise ValueError("fusion comparison boolean fields are invalid")
    return LineTypeResultDiff(
        exact_match=exact_match,
        changed_group_ids=_string_array(
            raw.get("changed_group_ids"), "comparison.changed_group_ids"
        ),
        changed_global_type_count=changed_global_type_count,
        changed_line_type_count=_safe_nonnegative_integer(
            raw.get("changed_line_type_count"),
            "comparison.changed_line_type_count",
        ),
        changed_op_count=_safe_nonnegative_integer(
            raw.get("changed_op_count"), "comparison.changed_op_count"
        ),
    )


@dataclass(frozen=True, slots=True)
class FusedLineTypeAudit:
    level: AuditLevel
    input_hash: str
    deterministic_replay_key: str
    statuses: FusedLineTypeStatuses
    elapsed_ms: float
    rules: FusedLineTypeRules
    trace: tuple[FusedLineTypeTraceEntry, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "level": self.level,
            "input_hash": self.input_hash,
            "deterministic_replay_key": self.deterministic_replay_key,
            "statuses": self.statuses.to_dict(),
            "elapsed_ms": self.elapsed_ms,
            "rules": self.rules.to_dict(),
        }
        if self.trace is not None:
            output["trace"] = [item.to_dict() for item in self.trace]
        return output


def _validate_fused_result(result: LineTypeRecognitionResult) -> None:
    for global_type in result.global_types:
        if global_type.recognition_source not in ("method1", "method2"):
            raise ValueError("fused global type recognition_source is invalid")
        _required_string(
            global_type.source_global_type_id,
            "fused global type source_global_type_id",
        )
        _required_string(global_type.type_uid, "fused global type type_uid")
        _finite_number(
            global_type.minimum_pair_similarity,
            "fused global type minimum_pair_similarity",
        )


def _classified_operations(result: LineTypeRecognitionResult) -> set[int]:
    return {
        op_index
        for group in result.groups
        for op_index in (
            tuple(
                index
                for line_type in group.line_types
                for index in line_type.op_indices
            )
            + group.non_linetype.op_indices
        )
    }


def _expected_trace(
    result: LineTypeRecognitionResult,
) -> tuple[FusedLineTypeTraceEntry, ...]:
    return tuple(
        FusedLineTypeTraceEntry(
            global_type.global_type_id,
            global_type.type_uid or "",
            tuple(member.type_id for member in global_type.members),
        )
        for global_type in result.global_types
    )


def _validate_envelope_semantics(envelope: "FusedLineTypeEnvelope") -> None:
    _validate_fused_result(envelope.result)
    expected_replay_key = (
        f"{envelope.page_identity}:{FUSED_LINE_TYPE_CONFIG_HASH}:"
        f"{envelope.audit.input_hash}"
    )
    if envelope.audit.deterministic_replay_key != expected_replay_key:
        raise ValueError("fused envelope deterministic replay key does not match")

    line_ops = {
        op_index
        for global_type in envelope.result.global_types
        for op_index in global_type.op_indices
    }
    all_ops = _classified_operations(envelope.result)
    expected_statuses = FusedLineTypeStatuses(
        line_type=len(line_ops),
        non_linetype=max(0, len(all_ops) - len(line_ops)),
        uncertain=0,
        skipped=0,
    )
    if envelope.audit.statuses != expected_statuses:
        raise ValueError("fused envelope statuses do not match the result")

    fusion_audit = envelope.audit.rules.method1_method2_display_fusion
    source_method2 = tuple(
        item
        for item in envelope.result.global_types
        if item.recognition_source == "method2"
    )
    source_method1 = tuple(
        item
        for item in envelope.result.global_types
        if item.recognition_source == "method1"
    )
    if (
        fusion_audit.method2_global_type_count != len(source_method2)
        or fusion_audit.retained_method1_global_type_count != len(source_method1)
        or fusion_audit.method2_op_count
        != len({index for item in source_method2 for index in item.op_indices})
        or fusion_audit.retained_method1_op_count
        != len({index for item in source_method1 for index in item.op_indices})
        or fusion_audit.skipped_duplicate_method1_global_type_count
        != len(fusion_audit.skipped_method1_global_type_ids)
        or fusion_audit.duplicate_overlap_op_count
        > fusion_audit.skipped_duplicate_method1_op_count
    ):
        raise ValueError("fused envelope fusion audit does not match the result")

    if envelope.audit.trace is not None:
        if envelope.audit.trace != _expected_trace(envelope.result):
            raise ValueError("fused envelope trace does not match global types")


@dataclass(frozen=True, slots=True)
class FusedLineTypeEnvelope:
    schema_version: int
    engine_version: str
    target_spec_version: str
    config_hash: str
    features: Mapping[str, bool]
    method1_engine_version: str
    method1_target_spec_version: str
    method2_engine_version: str
    method2_target_spec_version: str
    method2_local_projection_version: str
    fusion_policy_version: str
    method2_config_hash: str
    page_identity: str
    result: LineTypeRecognitionResult
    audit: FusedLineTypeAudit
    comparison: LineTypeResultDiff | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def to_dict(self) -> dict[str, object]:
        output: dict[str, object] = {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "target_spec_version": self.target_spec_version,
            "config_hash": self.config_hash,
            "features": dict(self.features),
            "method1_engine_version": self.method1_engine_version,
            "method1_target_spec_version": self.method1_target_spec_version,
            "method2_engine_version": self.method2_engine_version,
            "method2_target_spec_version": self.method2_target_spec_version,
            "method2_local_projection_version": self.method2_local_projection_version,
            "fusion_policy_version": self.fusion_policy_version,
            "method2_config_hash": self.method2_config_hash,
            "page_identity": self.page_identity,
            "result": self.result.to_dict(),
            "audit": self.audit.to_dict(),
        }
        if self.comparison is not None:
            output["comparison"] = self.comparison.to_dict()
        return output

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FusedLineTypeEnvelope":
        if not isinstance(value, Mapping):
            raise ValueError("fused envelope must be an object")
        expected_identity = (
            value.get("schema_version") == FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION
            and value.get("engine_version") == FUSED_LINE_TYPE_ENGINE_VERSION
            and value.get("target_spec_version")
            == FUSED_LINE_TYPE_TARGET_SPEC_VERSION
            and value.get("config_hash") == FUSED_LINE_TYPE_CONFIG_HASH
            and value.get("method1_engine_version")
            == LINE_TYPE_METHOD1_ENGINE_VERSION
            and value.get("method1_target_spec_version")
            == LINE_TYPE_METHOD1_TARGET_SPEC_VERSION
            and value.get("method2_engine_version")
            == LINE_TYPE_METHOD2_ENGINE_VERSION
            and value.get("method2_target_spec_version")
            == LINE_TYPE_METHOD2_TARGET_SPEC_VERSION
            and value.get("method2_local_projection_version")
            == METHOD2_LOCAL_PROJECTION_VERSION
            and value.get("fusion_policy_version")
            == LINE_TYPE_FUSION_POLICY_VERSION
            and value.get("method2_config_hash")
            == LINE_TYPE_METHOD2_CONFIG_HASH
        )
        if not expected_identity:
            raise ValueError("fused envelope version or configuration is invalid")
        features = value.get("features")
        if not isinstance(features, Mapping) or dict(features) != dict(
            FUSED_LINE_TYPE_FEATURES
        ):
            raise ValueError("fused envelope features do not match")
        page_identity = _required_string(
            value.get("page_identity"), "fused envelope page_identity"
        )
        raw_result = _mapping(value.get("result"), "fused envelope result")
        result = LineTypeRecognitionResult.from_dict(raw_result)
        _validate_fused_result(result)

        raw_audit = _mapping(value.get("audit"), "fused envelope audit")
        level = raw_audit.get("level")
        if level not in ("summary", "full"):
            raise ValueError("fused envelope audit level is invalid")
        input_hash = _required_string(raw_audit.get("input_hash"), "audit.input_hash")
        replay_key = _required_string(
            raw_audit.get("deterministic_replay_key"),
            "audit.deterministic_replay_key",
        )
        elapsed_ms = _finite_nonnegative_number(
            raw_audit.get("elapsed_ms"), "audit.elapsed_ms"
        )
        statuses = FusedLineTypeStatuses.from_value(raw_audit.get("statuses"))
        rules = FusedLineTypeRules.from_value(raw_audit.get("rules"))
        raw_trace = raw_audit.get("trace")
        trace: tuple[FusedLineTypeTraceEntry, ...] | None
        if raw_trace is None:
            trace = None
        else:
            if not isinstance(raw_trace, (list, tuple)):
                raise ValueError("fusion trace must be an array")
            trace = tuple(
                FusedLineTypeTraceEntry.from_value(item) for item in raw_trace
            )
        raw_comparison = value.get("comparison")
        comparison = (
            None
            if raw_comparison is None
            else _comparison_from_value(raw_comparison)
        )
        envelope = cls(
            FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION,
            FUSED_LINE_TYPE_ENGINE_VERSION,
            FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
            FUSED_LINE_TYPE_CONFIG_HASH,
            FUSED_LINE_TYPE_FEATURES,
            LINE_TYPE_METHOD1_ENGINE_VERSION,
            LINE_TYPE_METHOD1_TARGET_SPEC_VERSION,
            LINE_TYPE_METHOD2_ENGINE_VERSION,
            LINE_TYPE_METHOD2_TARGET_SPEC_VERSION,
            METHOD2_LOCAL_PROJECTION_VERSION,
            LINE_TYPE_FUSION_POLICY_VERSION,
            LINE_TYPE_METHOD2_CONFIG_HASH,
            page_identity,
            result,
            FusedLineTypeAudit(
                level,
                input_hash,
                replay_key,
                statuses,
                elapsed_ms,
                rules,
                trace,
            ),
            comparison,
        )
        _validate_envelope_semantics(envelope)
        return envelope


def validate_fused_line_type_envelope(
    value: FusedLineTypeEnvelope | Mapping[str, object],
) -> FusedLineTypeEnvelope:
    """Deeply validate a persisted fused envelope and all nested metadata."""

    if isinstance(value, FusedLineTypeEnvelope):
        FusedLineTypeEnvelope.from_dict(value.to_dict())
        return value
    return FusedLineTypeEnvelope.from_dict(value)


def _empty_method1_result() -> LineTypeRecognitionResult:
    return LineTypeRecognitionResult(
        groups=(),
        global_types=(),
        summary=RecognitionSummary(0, 0, 0, 0, 0, 0, 0),
    )


def _recognition_result_from_value(
    value: LineTypeRecognitionResult | Mapping[str, object],
) -> LineTypeRecognitionResult:
    if isinstance(value, LineTypeRecognitionResult):
        LineTypeRecognitionResult.from_dict(value.to_dict())
        return value
    return LineTypeRecognitionResult.from_dict(value)


def _js_round(value: float, digits: int) -> float:
    scale = 10**digits
    result = math.floor(value * scale + 0.5) / scale
    return 0.0 if result == 0 else result


def fuse_line_type_results_for_display(
    method2_envelope: LineTypeMethod2Envelope | Mapping[str, object],
    method1_result: LineTypeRecognitionResult | Mapping[str, object] | None = None,
    audit_level: AuditLevel = "summary",
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> FusedLineTypeEnvelope:
    """Package the pure fusion of already-complete recognition results.

    ``clock`` returns seconds like :func:`time.perf_counter`; it is injectable
    solely so elapsed-time parity can be tested deterministically.
    """

    if audit_level not in ("summary", "full"):
        raise ValueError("audit_level must be 'summary' or 'full'")
    method2 = validate_line_type_method2_envelope(method2_envelope)
    has_method1_reference = method1_result is not None
    method1 = (
        _empty_method1_result()
        if method1_result is None
        else _recognition_result_from_value(method1_result)
    )

    started_at = clock()
    fusion = fuse_line_type_recognition_results(method1, method2.result)
    result = attach_stable_type_uids(fusion.result, method2.page_identity)
    line_type_ops = {
        op_index
        for global_type in result.global_types
        for op_index in global_type.op_indices
    }
    all_operations = _classified_operations(fusion.result)
    elapsed_ms = _js_round(
        method2.audit.elapsed_ms + (clock() - started_at) * 1000,
        2,
    )
    input_hash = method2.audit.input_hash
    trace = _expected_trace(result) if audit_level == "full" else None
    envelope = FusedLineTypeEnvelope(
        schema_version=FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION,
        engine_version=FUSED_LINE_TYPE_ENGINE_VERSION,
        target_spec_version=FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
        config_hash=FUSED_LINE_TYPE_CONFIG_HASH,
        features=FUSED_LINE_TYPE_FEATURES,
        method1_engine_version=LINE_TYPE_METHOD1_ENGINE_VERSION,
        method1_target_spec_version=LINE_TYPE_METHOD1_TARGET_SPEC_VERSION,
        method2_engine_version=LINE_TYPE_METHOD2_ENGINE_VERSION,
        method2_target_spec_version=LINE_TYPE_METHOD2_TARGET_SPEC_VERSION,
        method2_local_projection_version=METHOD2_LOCAL_PROJECTION_VERSION,
        fusion_policy_version=LINE_TYPE_FUSION_POLICY_VERSION,
        method2_config_hash=method2.config_hash,
        page_identity=method2.page_identity,
        result=result,
        audit=FusedLineTypeAudit(
            level=audit_level,
            input_hash=input_hash,
            deterministic_replay_key=(
                f"{method2.page_identity}:{FUSED_LINE_TYPE_CONFIG_HASH}:"
                f"{input_hash}"
            ),
            statuses=FusedLineTypeStatuses(
                line_type=len(line_type_ops),
                non_linetype=max(0, len(all_operations) - len(line_type_ops)),
                uncertain=0,
                skipped=0,
            ),
            elapsed_ms=elapsed_ms,
            rules=FusedLineTypeRules(
                method2.audit.repeated_vector_text_family_clustering,
                fusion.audit,
            ),
            trace=trace,
        ),
        comparison=(
            compare_line_type_results(method1, result)
            if has_method1_reference
            else None
        ),
    )
    return validate_fused_line_type_envelope(envelope)


# Historical TS names retained only at the Python contract boundary.
line_type_v2_config_hash = fused_line_type_config_hash
line_type_v2_replay_input_hash = fused_line_type_replay_input_hash
validate_fused_line_type_envelope_v2 = validate_fused_line_type_envelope


__all__ = [
    "AuditLevel",
    "FUSED_LINE_TYPE_CONFIG_HASH",
    "FUSED_LINE_TYPE_ENGINE_VERSION",
    "FUSED_LINE_TYPE_FEATURES",
    "FUSED_LINE_TYPE_RESULT_SCHEMA_VERSION",
    "FUSED_LINE_TYPE_TARGET_SPEC_VERSION",
    "FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH",
    "FusedLineTypeAudit",
    "FusedLineTypeEnvelope",
    "FusedLineTypeRules",
    "FusedLineTypeStatuses",
    "FusedLineTypeTraceEntry",
    "LINE_TYPE_FUSION_POLICY_VERSION",
    "LINE_TYPE_METHOD1_ENGINE_VERSION",
    "LINE_TYPE_METHOD1_TARGET_SPEC_VERSION",
    "LINE_TYPE_METHOD2_ENGINE_VERSION",
    "LINE_TYPE_METHOD2_TARGET_SPEC_VERSION",
    "LINE_TYPE_V2_CONFIG_HASH",
    "LINE_TYPE_V2_ENGINE_VERSION",
    "LINE_TYPE_V2_FEATURES",
    "LINE_TYPE_V2_RESULT_SCHEMA_VERSION",
    "fuse_line_type_results_for_display",
    "fused_line_type_config_hash",
    "frozen_ts_fused_line_type_config_hash",
    "fused_line_type_replay_input_hash",
    "line_type_v2_config_hash",
    "line_type_v2_replay_input_hash",
    "validate_fused_line_type_envelope",
    "validate_fused_line_type_envelope_v2",
]
