"""Strict, dependency-free JSON decoding for the canonical PageIR contract.

This module belongs to the reusable algorithm boundary.  It imports only the
renderer-neutral IR dataclasses and Python's standard library; PDF parsers,
caches, transports and UI code must remain outside it.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache
import json
import math
from types import NoneType, UnionType
from typing import (
    Any,
    Literal,
    Mapping,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from .ir import ImageOperationIR, PageIR, PathOperationIR, TextOperationIR
from .versions import PAGE_IR_VERSION


class IRCodecError(ValueError):
    """External JSON data does not exactly satisfy the canonical IR schema."""


def _json_value(value: object) -> object:
    """Return lossless JSON-compatible data for a frozen IR dataclass."""

    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("canonical", True)
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise IRCodecError("IR mappings must use string keys")
        return {
            key: _json_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IRCodecError("IR JSON does not support non-finite floats")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise IRCodecError(f"IR JSON does not support {type(value).__name__}")


def page_ir_to_dict(page: PageIR) -> dict[str, object]:
    """Return a lossless JSON-compatible dictionary for one PageIR."""

    if not isinstance(page, PageIR):
        raise TypeError("page must be a PageIR")
    value = _json_value(page)
    if not isinstance(value, dict):  # Defensive if PageIR stops being a dataclass.
        raise IRCodecError("PageIR encoding returned an unexpected type")
    decoded = page_ir_from_dict(value)
    if decoded != page:
        raise IRCodecError(
            "PageIR contains non-canonical runtime field types; "
            "JSON round-trip would change its value"
        )
    if decoded.fingerprint != page.fingerprint:
        raise IRCodecError(
            "PageIR contains non-canonical runtime field types; "
            "JSON round-trip would change its fingerprint"
        )
    return value


def _record(value: object, label: str, expected_keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise IRCodecError(f"{label} fields do not match the schema")
    if any(not isinstance(key, str) for key in value):
        raise IRCodecError(f"{label} keys must be strings")
    return value


@lru_cache(maxsize=None)
def _dataclass_hints(cls: type[object]) -> dict[str, object]:
    return get_type_hints(cls)


def _decode_typed(value: object, annotation: object, label: str) -> object:
    """Decode one JSON value through an exact runtime type annotation."""

    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if annotation is Any:
        raise IRCodecError(f"{label} has an unsafe Any schema")
    if annotation is NoneType:
        if value is not None:
            raise IRCodecError(f"{label} must be null")
        return None
    if annotation is bool:
        if type(value) is not bool:
            raise IRCodecError(f"{label} must be boolean")
        return value
    if annotation is int:
        if type(value) is not int:
            raise IRCodecError(f"{label} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IRCodecError(f"{label} must be a finite number")
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise IRCodecError(f"{label} must be a finite number") from error
        if not math.isfinite(number):
            raise IRCodecError(f"{label} must be a finite number")
        return number
    if annotation is str:
        if not isinstance(value, str):
            raise IRCodecError(f"{label} must be a string")
        return value
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in arguments):
            raise IRCodecError(f"{label} is not an allowed literal")
        return value
    if origin is tuple:
        if not isinstance(value, list):
            raise IRCodecError(f"{label} must be a JSON array")
        if len(arguments) == 2 and arguments[1] is Ellipsis:
            return tuple(
                _decode_typed(item, arguments[0], f"{label}[{index}]")
                for index, item in enumerate(value)
            )
        if len(value) != len(arguments):
            raise IRCodecError(f"{label} has the wrong tuple length")
        return tuple(
            _decode_typed(item, item_type, f"{label}[{index}]")
            for index, (item, item_type) in enumerate(zip(value, arguments))
        )
    if origin in (Union, UnionType):
        # Page operations use an explicit discriminator.  Do not try a
        # structurally surprising union arm when the claimed kind is known.
        if isinstance(value, Mapping) and isinstance(value.get("kind"), str):
            kinds = {
                "path": PathOperationIR,
                "text": TextOperationIR,
                "image": ImageOperationIR,
            }
            selected = kinds.get(value.get("kind"))
            if selected is not None and selected in arguments:
                return _decode_typed(value, selected, label)
        failures: list[Exception] = []
        for item_type in arguments:
            try:
                return _decode_typed(value, item_type, label)
            except (IRCodecError, TypeError, ValueError) as error:
                failures.append(error)
        cause = failures[-1] if failures else None
        raise IRCodecError(f"{label} does not match its union schema") from cause
    if isinstance(annotation, type) and is_dataclass(annotation):
        expected_fields = tuple(
            item
            for item in fields(annotation)
            if item.init and item.metadata.get("canonical", True)
        )
        record = _record(value, label, {item.name for item in expected_fields})
        hints = _dataclass_hints(annotation)
        decoded = {
            item.name: _decode_typed(
                record[item.name],
                hints[item.name],
                f"{label}.{item.name}",
            )
            for item in expected_fields
        }
        try:
            return annotation(**decoded)
        except (TypeError, ValueError) as error:
            raise IRCodecError(f"{label} violates its IR contract") from error
    raise IRCodecError(f"{label} uses an unsupported schema {annotation!r}")


def page_ir_from_dict(value: Mapping[str, object]) -> PageIR:
    """Reconstruct one PageIR from exact JSON-compatible dictionary data."""

    page = _decode_typed(value, PageIR, "PageIR")
    if not isinstance(page, PageIR):  # Defensive if annotations are changed.
        raise IRCodecError("PageIR reconstruction returned an unexpected type")
    if page.page_ir_version != PAGE_IR_VERSION:
        raise IRCodecError("PageIR version does not match this engine")
    kind_counts = {"path": 0, "text": 0, "image": 0}
    for operation in page.operations:
        expected_ordinal = kind_counts[operation.kind]
        if operation.ordinal != expected_ordinal:
            raise IRCodecError("PageIR operation ordinals must be dense per kind")
        kind_counts[operation.kind] += 1
    return page


def _reject_json_constant(value: str) -> None:
    raise IRCodecError(f"PageIR JSON contains invalid number {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise IRCodecError(f"PageIR JSON contains duplicate key {key!r}")
        value[key] = item
    return value


def page_ir_from_json(
    source: str | bytes | bytearray | memoryview,
    *,
    max_bytes: int | None = None,
) -> PageIR:
    """Decode strict UTF-8 JSON and reconstruct one canonical PageIR.

    ``max_bytes`` is an optional transport guard for untrusted integrations;
    it does not become part of PageIR identity or recognition semantics.
    """

    if max_bytes is not None and (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(source, str):
        try:
            encoded_size = len(source.encode("utf-8", errors="strict"))
        except UnicodeEncodeError as error:
            raise IRCodecError("PageIR JSON string is not valid Unicode") from error
        if max_bytes is not None and encoded_size > max_bytes:
            raise IRCodecError("PageIR JSON exceeds max_bytes")
        text = source
    elif isinstance(source, memoryview):
        if max_bytes is not None and source.nbytes > max_bytes:
            raise IRCodecError("PageIR JSON exceeds max_bytes")
        raw = source.tobytes()
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise IRCodecError("PageIR JSON must be valid UTF-8") from error
    elif isinstance(source, (bytes, bytearray)):
        if max_bytes is not None and len(source) > max_bytes:
            raise IRCodecError("PageIR JSON exceeds max_bytes")
        try:
            text = bytes(source).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise IRCodecError("PageIR JSON must be valid UTF-8") from error
    else:
        raise TypeError("PageIR JSON source must be str or bytes-like")
    try:
        value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_object,
        )
    except IRCodecError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise IRCodecError("PageIR JSON is not valid strict JSON") from error
    if not isinstance(value, Mapping):
        raise IRCodecError("PageIR JSON root must be an object")
    return page_ir_from_dict(value)


__all__ = [
    "IRCodecError",
    "page_ir_from_dict",
    "page_ir_from_json",
    "page_ir_to_dict",
]
