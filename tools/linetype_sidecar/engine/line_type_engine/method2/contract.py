"""Python Method2 candidate feature and persistence contract.

The envelope contains Method2 only.  It neither imports nor embeds Method1 or
display fusion state.  Its implementation identity is deliberately distinct
from the frozen TypeScript r46 target specification.  Configuration
fingerprints use the same UTF-16 FNV-1a writer as the frozen contract; the
PageIR input hash has its own domain tag because PageIR is a richer input
representation than the old browser ``Scene``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from ..results import LineTypeRecognitionResult
from ..versions import (
    FROZEN_TS_METHOD2_ENGINE_VERSION,
    PYTHON_METHOD2_ENGINE_VERSION,
    PYTHON_METHOD2_LOCAL_PROJECTION_VERSION,
)


METHOD2_ENGINE_VERSION = PYTHON_METHOD2_ENGINE_VERSION
METHOD2_TARGET_SPEC_VERSION = FROZEN_TS_METHOD2_ENGINE_VERSION
METHOD2_LOCAL_PROJECTION_VERSION = PYTHON_METHOD2_LOCAL_PROJECTION_VERSION
METHOD2_RESULT_SCHEMA_VERSION = 2

_FEATURE_ITEMS = (
    ("repeated_vector_text_family_clustering", True),
    ("native_pdf_text_pattern_clustering", True),
    ("vector_short_stroke_text_pattern_clustering", True),
    ("drawing_order_text_signature", True),
    ("dash_connected_text_line_assembly", True),
    ("same_family_carrier_route_search", True),
    ("carrier_connected_cross_family_merge", True),
    ("non_overlapping_pattern_instances", True),
    ("sequential_carrier_pattern_relaxation", True),
    ("sequential_contextual_pattern_satellites", True),
    ("sequential_terminal_carrier_extension", True),
    ("sequential_multi_path_pattern_rescue", True),
)

LINE_TYPE_METHOD2_FEATURES: Mapping[str, bool] = MappingProxyType(dict(_FEATURE_ITEMS))


def _javascript_number(value: int | float) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("fingerprint number must be int or float")
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "-0" if math.copysign(1.0, value) < 0 else "0"
    encoded = repr(value).lower()
    absolute = abs(value)
    if "e" in encoded and 1e-6 <= absolute < 1e21:
        mantissa, exponent_text = encoded.split("e", 1)
        exponent = int(exponent_text)
        negative = mantissa.startswith("-")
        digits = mantissa.lstrip("-").replace(".", "")
        decimal_position = mantissa.lstrip("-").find(".")
        if decimal_position < 0:
            decimal_position = len(digits)
        decimal_position += exponent
        if decimal_position <= 0:
            result = "0." + "0" * (-decimal_position) + digits
        elif decimal_position >= len(digits):
            result = digits + "0" * (decimal_position - len(digits))
        else:
            result = digits[:decimal_position] + "." + digits[decimal_position:]
        if "." in result:
            result = result.rstrip("0").rstrip(".")
        return "-" + result if negative else result
    if "e" in encoded:
        mantissa, exponent = encoded.split("e", 1)
        sign = ""
        if exponent.startswith(("+", "-")):
            sign, exponent = exponent[0], exponent[1:]
        exponent = exponent.lstrip("0") or "0"
        if sign == "+" or (not sign and int(exponent) >= 21):
            sign = "+"
        return f"{mantissa}e{sign}{exponent}"
    if value.is_integer():
        return str(int(value))
    return encoded


class LineTypeFingerprintWriter:
    """Unambiguous incremental hash compatible with the TS writer."""

    __slots__ = ("_hash",)

    def __init__(self) -> None:
        self._hash = 0xCBF29CE484222325

    def _update(self, value: str) -> None:
        # JavaScript hashes UTF-16 code units.  Walking Python characters
        # directly avoids allocating and indexing a temporary byte string for
        # every tiny token while preserving surrogate pairs and lone
        # surrogates exactly.
        hash_value = self._hash
        for character in value:
            code_point = ord(character)
            if code_point <= 0xFFFF:
                hash_value ^= code_point
                hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                continue
            code_point -= 0x10000
            high_surrogate = 0xD800 + (code_point >> 10)
            low_surrogate = 0xDC00 + (code_point & 0x3FF)
            hash_value ^= high_surrogate
            hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
            hash_value ^= low_surrogate
            hash_value = (hash_value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        self._hash = hash_value

    def string(self, value: str) -> "LineTypeFingerprintWriter":
        if not isinstance(value, str):
            raise TypeError("fingerprint string value must be str")
        length = (
            len(value)
            if value.isascii()
            else len(value.encode("utf-16-le", errors="surrogatepass")) // 2
        )
        self._update(f"s{length}:")
        self._update(value)
        return self

    def number(self, value: int | float) -> "LineTypeFingerprintWriter":
        normalized = _javascript_number(value)
        self._update(f"n{len(normalized)}:{normalized}")
        return self

    def boolean(self, value: bool) -> "LineTypeFingerprintWriter":
        if not isinstance(value, bool):
            raise TypeError("fingerprint boolean value must be bool")
        self._update("b1" if value else "b0")
        return self

    def null(self) -> "LineTypeFingerprintWriter":
        self._update("z")
        return self

    def begin(self, label: str) -> "LineTypeFingerprintWriter":
        self._update("{")
        return self.string(label)

    def end(self) -> "LineTypeFingerprintWriter":
        self._update("}")
        return self

    def digest(self) -> str:
        return f"{self._hash:016x}"


def line_type_method2_config_hash(
    features: Mapping[str, bool] = LINE_TYPE_METHOD2_FEATURES,
) -> str:
    writer = (
        LineTypeFingerprintWriter()
        .begin("python-method2-candidate-config-v2")
        .string(METHOD2_ENGINE_VERSION)
        .string(METHOD2_TARGET_SPEC_VERSION)
        .string(METHOD2_LOCAL_PROJECTION_VERSION)
    )
    for name, enabled in sorted(features.items()):
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise TypeError("Method2 features must map strings to booleans")
        writer.string(name).boolean(enabled)
    return writer.end().digest()


LINE_TYPE_METHOD2_CONFIG_HASH = line_type_method2_config_hash()


def frozen_ts_method2_config_hash(
    features: Mapping[str, bool] = LINE_TYPE_METHOD2_FEATURES,
) -> str:
    """Return the frozen r46 config hash used only as a parity anchor."""

    writer = LineTypeFingerprintWriter().begin("method2-config-v1").string(
        METHOD2_TARGET_SPEC_VERSION
    )
    for name, enabled in sorted(features.items()):
        if not isinstance(name, str) or not isinstance(enabled, bool):
            raise TypeError("Method2 features must map strings to booleans")
        writer.string(name).boolean(enabled)
    return writer.end().digest()


FROZEN_TS_METHOD2_CONFIG_HASH = frozen_ts_method2_config_hash()


class _AuditLike(Protocol):
    def to_dict(self) -> dict[str, object]: ...


_AUDIT_ARRAY_KEYS = (
    "family_diagnostics",
    "matched_text_op_indices",
    "line_type_confirmed_text_op_indices",
    "affected_group_ids",
    "region_instances",
)
_AUDIT_COUNT_KEYS = (
    "detected_region_count",
    "eligible_region_count",
    "matched_instance_count",
    "matched_family_count",
    "dash_connected_family_count",
    "line_type_confirmed_instance_count",
    "line_type_confirmed_text_op_count",
    "matched_text_op_count",
    "attached_dash_op_count",
    "bridged_route_op_count",
)


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _integer_array(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    return tuple(_nonnegative_integer(item, label) for item in value)


def _mapping_array(value: object, label: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be an array")
    result: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label} entries must be objects")
        result.append(MappingProxyType(deepcopy(dict(item))))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VectorTextFamilyAuditPayload:
    family_diagnostics: tuple[Mapping[str, object], ...]
    detected_region_count: int
    eligible_region_count: int
    matched_instance_count: int
    matched_family_count: int
    dash_connected_family_count: int
    line_type_confirmed_instance_count: int
    line_type_confirmed_text_op_count: int
    matched_text_op_count: int
    attached_dash_op_count: int
    bridged_route_op_count: int
    matched_text_op_indices: tuple[int, ...]
    line_type_confirmed_text_op_indices: tuple[int, ...]
    affected_group_ids: tuple[str, ...]
    region_instances: tuple[Mapping[str, object], ...]

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, object] | _AuditLike,
    ) -> "VectorTextFamilyAuditPayload":
        raw = value.to_dict() if hasattr(value, "to_dict") else value
        if not isinstance(raw, Mapping):
            raise ValueError("Method2 family audit must be an object")
        for key in _AUDIT_ARRAY_KEYS:
            if not isinstance(raw.get(key), (list, tuple)):
                raise ValueError(f"Method2 family audit {key} must be an array")
        counts = {
            key: _nonnegative_integer(raw.get(key), f"family audit {key}")
            for key in _AUDIT_COUNT_KEYS
        }
        affected = raw["affected_group_ids"]
        assert isinstance(affected, (list, tuple))
        if any(not isinstance(item, str) for item in affected):
            raise ValueError("family audit affected_group_ids must contain strings")
        return cls(
            family_diagnostics=_mapping_array(
                raw["family_diagnostics"], "family_diagnostics"
            ),
            **counts,
            matched_text_op_indices=_integer_array(
                raw["matched_text_op_indices"], "matched_text_op_indices"
            ),
            line_type_confirmed_text_op_indices=_integer_array(
                raw["line_type_confirmed_text_op_indices"],
                "line_type_confirmed_text_op_indices",
            ),
            affected_group_ids=tuple(affected),
            region_instances=_mapping_array(raw["region_instances"], "region_instances"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family_diagnostics": [deepcopy(dict(item)) for item in self.family_diagnostics],
            **{key: getattr(self, key) for key in _AUDIT_COUNT_KEYS},
            "matched_text_op_indices": list(self.matched_text_op_indices),
            "line_type_confirmed_text_op_indices": list(
                self.line_type_confirmed_text_op_indices
            ),
            "affected_group_ids": list(self.affected_group_ids),
            "region_instances": [deepcopy(dict(item)) for item in self.region_instances],
        }


@dataclass(frozen=True, slots=True)
class LineTypeMethod2Audit:
    input_hash: str
    deterministic_replay_key: str
    elapsed_ms: float
    repeated_vector_text_family_clustering: VectorTextFamilyAuditPayload

    def to_dict(self) -> dict[str, object]:
        return {
            "input_hash": self.input_hash,
            "deterministic_replay_key": self.deterministic_replay_key,
            "elapsed_ms": self.elapsed_ms,
            "repeated_vector_text_family_clustering": (
                self.repeated_vector_text_family_clustering.to_dict()
            ),
        }


@dataclass(frozen=True, slots=True)
class LineTypeMethod2Envelope:
    schema_version: int
    engine_version: str
    target_spec_version: str
    local_projection_version: str
    config_hash: str
    features: Mapping[str, bool]
    page_identity: str
    result: LineTypeRecognitionResult
    audit: LineTypeMethod2Audit

    def __post_init__(self) -> None:
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "target_spec_version": self.target_spec_version,
            "local_projection_version": self.local_projection_version,
            "config_hash": self.config_hash,
            "features": dict(self.features),
            "page_identity": self.page_identity,
            "result": self.result.to_dict(),
            "audit": self.audit.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "LineTypeMethod2Envelope":
        if not isinstance(value, Mapping):
            raise ValueError("Method2 envelope must be an object")
        if value.get("schema_version") != METHOD2_RESULT_SCHEMA_VERSION:
            raise ValueError("Method2 envelope schema version does not match")
        if value.get("engine_version") != METHOD2_ENGINE_VERSION:
            raise ValueError("Method2 envelope engine version does not match")
        if value.get("target_spec_version") != METHOD2_TARGET_SPEC_VERSION:
            raise ValueError("Method2 envelope target specification does not match")
        if value.get("local_projection_version") != METHOD2_LOCAL_PROJECTION_VERSION:
            raise ValueError("Method2 envelope local projection version does not match")
        if value.get("config_hash") != LINE_TYPE_METHOD2_CONFIG_HASH:
            raise ValueError("Method2 envelope config hash does not match")
        raw_features = value.get("features")
        if not isinstance(raw_features, Mapping) or dict(raw_features) != dict(
            LINE_TYPE_METHOD2_FEATURES
        ):
            raise ValueError("Method2 envelope features do not match")
        page_identity = value.get("page_identity")
        if not isinstance(page_identity, str) or not page_identity:
            raise ValueError("Method2 envelope page_identity is invalid")
        raw_result = value.get("result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("Method2 envelope result is invalid")
        result = LineTypeRecognitionResult.from_dict(raw_result)
        raw_audit = value.get("audit")
        if not isinstance(raw_audit, Mapping):
            raise ValueError("Method2 envelope audit is invalid")
        input_hash = raw_audit.get("input_hash")
        replay_key = raw_audit.get("deterministic_replay_key")
        elapsed_ms = raw_audit.get("elapsed_ms")
        if not isinstance(input_hash, str) or not input_hash:
            raise ValueError("Method2 envelope input_hash is invalid")
        if not isinstance(replay_key, str) or not replay_key:
            raise ValueError("Method2 envelope deterministic replay key is invalid")
        if (
            isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, (int, float))
            or not math.isfinite(elapsed_ms)
            or elapsed_ms < 0
        ):
            raise ValueError("Method2 envelope elapsed_ms is invalid")
        expected_replay = (
            f"{page_identity}:{LINE_TYPE_METHOD2_CONFIG_HASH}:{input_hash}"
        )
        if replay_key != expected_replay:
            raise ValueError("Method2 envelope replay key does not match")
        raw_family_audit = raw_audit.get("repeated_vector_text_family_clustering")
        if not isinstance(raw_family_audit, Mapping):
            raise ValueError("Method2 envelope family audit is invalid")
        audit = LineTypeMethod2Audit(
            input_hash,
            replay_key,
            float(elapsed_ms),
            VectorTextFamilyAuditPayload.from_value(raw_family_audit),
        )
        return cls(
            METHOD2_RESULT_SCHEMA_VERSION,
            METHOD2_ENGINE_VERSION,
            METHOD2_TARGET_SPEC_VERSION,
            METHOD2_LOCAL_PROJECTION_VERSION,
            LINE_TYPE_METHOD2_CONFIG_HASH,
            LINE_TYPE_METHOD2_FEATURES,
            page_identity,
            result,
            audit,
        )


def validate_line_type_method2_envelope(
    value: LineTypeMethod2Envelope | Mapping[str, object],
) -> LineTypeMethod2Envelope:
    """Validate nested result, audit, version and replay metadata."""

    if isinstance(value, LineTypeMethod2Envelope):
        LineTypeMethod2Envelope.from_dict(value.to_dict())
        return value
    return LineTypeMethod2Envelope.from_dict(value)


__all__ = [
    "FROZEN_TS_METHOD2_CONFIG_HASH",
    "LINE_TYPE_METHOD2_CONFIG_HASH",
    "LINE_TYPE_METHOD2_FEATURES",
    "LineTypeFingerprintWriter",
    "LineTypeMethod2Audit",
    "LineTypeMethod2Envelope",
    "METHOD2_ENGINE_VERSION",
    "METHOD2_LOCAL_PROJECTION_VERSION",
    "METHOD2_RESULT_SCHEMA_VERSION",
    "METHOD2_TARGET_SPEC_VERSION",
    "VectorTextFamilyAuditPayload",
    "frozen_ts_method2_config_hash",
    "line_type_method2_config_hash",
    "validate_line_type_method2_envelope",
]
