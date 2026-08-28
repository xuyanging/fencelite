"""Canonical corpus comparison against frozen TypeScript oracle outputs.

The comparator is deliberately read-only with respect to viewer caches and
persisted headless oracle runs.
It compares ownership partitions rather than generated ids, labels or timing
metadata, and refuses to treat dense operation indices as comparable unless
the serialized Method1 input hash proves that both engines saw the same input.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .corpus_runner import (
    CORPUS_RUN_SCHEMA_VERSION,
    validate_composed_corpus_page,
    validate_corpus_stage_page,
)
from .fusion_contract import (
    FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH,
    FUSED_LINE_TYPE_FEATURES,
    FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
)
from .method1.serializer import METHOD1_SERIALIZED_INPUT_HASH_SCHEMA
from .method2.contract import (
    FROZEN_TS_METHOD2_CONFIG_HASH,
    LINE_TYPE_METHOD2_FEATURES,
)
from .results import LineTypeRecognitionResult
from .versions import (
    FROZEN_TS_FUSION_POLICY_VERSION,
    FROZEN_TS_METHOD1_ENGINE_VERSION,
    FROZEN_TS_METHOD2_ENGINE_VERSION,
)


CORPUS_COMPARISON_SCHEMA_VERSION = 1
FROZEN_TS_METHOD1_CACHE_NAMESPACE = "fingerprint-grid-v6-method1-r10"
FROZEN_TS_FUSED_RESULT_SCHEMA_VERSION = 3
FROZEN_TS_HEADLESS_RUN_SCHEMA_VERSION = 1
FROZEN_TS_HEADLESS_PAGE_SCHEMA_VERSION = 1
FROZEN_TS_HEADLESS_OUTPUTS = ("method1", "method2", "fused")
FROZEN_TS_HEADLESS_ANALYZER_ID = (
    "line-type-page-preparation-v1+sequential-segmentation-v1+"
    f"{FROZEN_TS_METHOD1_ENGINE_VERSION}+"
    f"{FROZEN_TS_METHOD2_ENGINE_VERSION}+"
    f"{FROZEN_TS_FUSION_POLICY_VERSION}"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _load_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
    _atomic_write(path, _canonical_json(value) + b"\n")


def _atomic_write_jsonl(path: Path, values: Iterable[Mapping[str, object]]) -> None:
    _atomic_write(path, b"".join(_canonical_json(value) + b"\n" for value in values))


def _result(value: Mapping[str, Any], label: str) -> LineTypeRecognitionResult:
    try:
        return LineTypeRecognitionResult.from_dict(value)
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"{label} result contract is invalid: {error}") from error


def canonical_result_partitions(
    value: LineTypeRecognitionResult | Mapping[str, Any],
) -> dict[str, object]:
    """Discard display identities while retaining every ownership partition."""

    result = value if isinstance(value, LineTypeRecognitionResult) else _result(value, "line type")
    group_ownership: list[dict[str, object]] = []
    local_ownership: list[dict[str, object]] = []
    nonline_ownership: list[dict[str, object]] = []
    local_member_ownership: dict[tuple[str, str], dict[str, object]] = {}
    for group in result.groups:
        domain = tuple(sorted({
            op_index
            for line_type in group.line_types
            for op_index in line_type.op_indices
        } | set(group.non_linetype.op_indices)))
        domain_key = list(domain)
        group_ownership.append({
            "atom_count": group.atom_count,
            "op_indices": domain_key,
        })
        local_ownership.append({
            "group_atom_count": group.atom_count,
            "group_op_indices": domain_key,
            "partitions": sorted(
                (
                    {
                        "atom_count": line_type.atom_count,
                        "op_indices": list(line_type.op_indices),
                    }
                    for line_type in group.line_types
                ),
                key=_canonical_json,
            ),
        })
        for line_type in group.line_types:
            local_member_ownership[(group.group_id, line_type.type_id)] = {
                "group_op_indices": domain_key,
                "local_op_indices": list(line_type.op_indices),
                "atom_count": line_type.atom_count,
            }
        nonline_ownership.append({
            "group_atom_count": group.atom_count,
            "group_op_indices": domain_key,
            "atom_count": group.non_linetype.atom_count,
            "op_indices": list(group.non_linetype.op_indices),
        })

    global_ownership = [
        {
            "op_indices": list(global_type.op_indices),
            "member_count": global_type.member_count,
            "group_count": global_type.group_count,
            "member_atom_counts": sorted(
                member.atom_count for member in global_type.members
            ),
            "member_ownership": sorted(
                (
                    {
                        "declared_atom_count": member.atom_count,
                        "local_partition": local_member_ownership.get(
                            (member.case_id, member.type_id)
                        ),
                    }
                    for member in global_type.members
                ),
                key=_canonical_json,
            ),
        }
        for global_type in result.global_types
    ]
    return {
        "group_ownership": sorted(group_ownership, key=_canonical_json),
        "local_ownership": sorted(local_ownership, key=_canonical_json),
        "global_ownership": sorted(global_ownership, key=_canonical_json),
        "nonline_ownership": sorted(nonline_ownership, key=_canonical_json),
        "summary": result.summary.to_dict(),
        "error_count": len(result.errors),
    }


def compare_result_partitions(
    actual: LineTypeRecognitionResult | Mapping[str, Any],
    expected: LineTypeRecognitionResult | Mapping[str, Any],
) -> dict[str, object]:
    actual_partition = canonical_result_partitions(actual)
    expected_partition = canonical_result_partitions(expected)
    components = (
        "group_ownership",
        "local_ownership",
        "global_ownership",
        "nonline_ownership",
        "summary",
        "error_count",
    )
    equality = {
        f"{component}_equal": actual_partition[component]
        == expected_partition[component]
        for component in components
    }
    return {
        "exact_match": all(equality.values()),
        **equality,
        "actual_fingerprint": _fingerprint(actual_partition),
        "expected_fingerprint": _fingerprint(expected_partition),
        "actual_component_fingerprints": {
            component: _fingerprint(actual_partition[component])
            for component in components
        },
        "expected_component_fingerprints": {
            component: _fingerprint(expected_partition[component])
            for component in components
        },
    }


@dataclass(frozen=True, slots=True)
class CorpusCompareOptions:
    run_root: Path
    method1_baseline_root: Path | None = None
    method2_baseline_root: Path | None = None
    output_directory: Path | None = None
    headless_oracle_root: Path | None = None

    def __post_init__(self) -> None:
        run_root = Path(self.run_root).resolve()
        method1 = (
            None
            if self.method1_baseline_root is None
            else Path(self.method1_baseline_root).resolve()
        )
        method2 = (
            None
            if self.method2_baseline_root is None
            else Path(self.method2_baseline_root).resolve()
        )
        headless = (
            None
            if self.headless_oracle_root is None
            else Path(self.headless_oracle_root).resolve()
        )
        output = (
            run_root / "comparison"
            if self.output_directory is None
            else Path(self.output_directory).resolve()
        )
        if not run_root.is_dir():
            raise ValueError(f"corpus run root does not exist: {run_root}")
        if headless is None:
            if method1 is None or not method1.is_dir():
                raise ValueError(
                    f"Method1 baseline root does not exist: {method1}"
                )
            if method2 is None or not method2.is_dir():
                raise ValueError(
                    f"Method2 baseline root does not exist: {method2}"
                )
        elif not headless.is_dir():
            raise ValueError(
                f"TypeScript headless oracle root does not exist: {headless}"
            )
        try:
            output.relative_to(run_root)
        except ValueError as error:
            raise ValueError(
                "comparison output must stay inside the owned corpus run root"
            ) from error
        if output == run_root or any(
            output == protected or protected in output.parents
            for protected in (
                run_root / "sources",
                run_root / "stages",
                run_root / "composed",
                run_root / "logs",
            )
        ):
            raise ValueError("comparison output overlaps protected run artifacts")
        for protected in tuple(
            path for path in (method1, method2, headless) if path is not None
        ):
            try:
                output.relative_to(protected)
            except ValueError:
                continue
            raise ValueError("comparison output must not be inside a frozen cache")
        object.__setattr__(self, "run_root", run_root)
        object.__setattr__(self, "method1_baseline_root", method1)
        object.__setattr__(self, "method2_baseline_root", method2)
        object.__setattr__(self, "headless_oracle_root", headless)
        object.__setattr__(self, "output_directory", output)


def _index_python_pages(
    run_root: Path, stage: str
) -> dict[tuple[str, int], tuple[Path, ...]]:
    values: dict[tuple[str, int], list[Path]] = defaultdict(list)
    root = run_root / "stages" / stage
    if root.is_dir():
        for path in root.glob("*/pages-*/pages/page-*.json"):
            try:
                page_number = int(path.stem.removeprefix("page-"))
            except ValueError:
                continue
            values[(path.parents[2].name, page_number)].append(path)
    return {key: tuple(sorted(paths)) for key, paths in values.items()}


def _index_composed_pages(run_root: Path) -> dict[tuple[str, int], tuple[Path, ...]]:
    values: dict[tuple[str, int], list[Path]] = defaultdict(list)
    root = run_root / "composed"
    if root.is_dir():
        for path in root.glob("*/pages/page-*.json"):
            try:
                page_number = int(path.stem.removeprefix("page-"))
            except ValueError:
                continue
            values[(path.parents[1].name, page_number)].append(path)
    return {key: tuple(sorted(paths)) for key, paths in values.items()}


def _expected_pages(
    run_manifest: Mapping[str, Any],
) -> tuple[tuple[str, int], ...]:
    plan = run_manifest.get("plan")
    raw_documents = plan.get("documents") if isinstance(plan, Mapping) else None
    if not isinstance(plan, Mapping) or not isinstance(raw_documents, list):
        raise ValueError("corpus run plan is missing or malformed")
    expected: set[tuple[str, int]] = set()
    document_ids: set[str] = set()
    declared_page_count = 0
    for item in raw_documents:
        if not isinstance(item, Mapping):
            raise ValueError("corpus run plan contains a malformed document")
        document_id = item.get("document_id")
        pages = item.get("selected_pages")
        page_count = item.get("page_count")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id in document_ids
            or not isinstance(pages, list)
            or not pages
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 1
        ):
            raise ValueError("corpus run plan document identity/pages are invalid")
        document_ids.add(document_id)
        seen_pages: set[int] = set()
        for page in pages:
            if (
                isinstance(page, bool)
                or not isinstance(page, int)
                or page < 1
                or page > page_count
                or page in seen_pages
            ):
                raise ValueError("corpus run plan contains an invalid/duplicate page")
            seen_pages.add(page)
            expected.add((document_id, page))
        declared_page_count += len(pages)
    if not expected:
        raise ValueError("corpus run plan contains no pages")
    if plan.get("document_count") != len(document_ids):
        raise ValueError("corpus run plan document_count is inconsistent")
    if plan.get("page_count") != declared_page_count:
        raise ValueError("corpus run plan page_count is inconsistent")
    if run_manifest.get("document_count") != len(document_ids):
        raise ValueError("corpus run manifest document_count is inconsistent")
    if run_manifest.get("page_count") != declared_page_count:
        raise ValueError("corpus run manifest page_count is inconsistent")
    return tuple(sorted(expected, key=lambda item: (item[0], item[1])))


def _single_page(
    index: Mapping[tuple[str, int], tuple[Path, ...]],
    key: tuple[str, int],
) -> tuple[dict[str, Any] | None, str, str | None]:
    paths = index.get(key, ())
    if not paths:
        return None, "missing", None
    if len(paths) != 1:
        return None, "duplicate", ", ".join(str(path) for path in paths)
    value = _load_object(paths[0])
    if value is None:
        return None, "invalid", str(paths[0])
    return value, "available", str(paths[0])


def _source_hashes(
    run_manifest: Mapping[str, Any],
    run_root: Path,
    expected: Sequence[tuple[str, int]],
) -> dict[str, str]:
    raw_sources = run_manifest.get("sources")
    if not isinstance(raw_sources, Mapping):
        raise ValueError("corpus run source snapshot records are missing")
    document_ids = {document_id for document_id, _ in expected}
    if set(raw_sources) != document_ids:
        raise ValueError("corpus run source snapshot set differs from its plan")
    result: dict[str, str] = {}
    for document_id in sorted(document_ids):
        record = raw_sources.get(document_id)
        if not isinstance(record, Mapping):
            raise ValueError(f"source snapshot record is invalid: {document_id}")
        path_value = record.get("path")
        digest = record.get("sha256")
        byte_length = record.get("byte_length")
        expected_path = (run_root / "sources" / f"{document_id}.pdf").resolve()
        if (
            record.get("document_id") != document_id
            or not isinstance(path_value, str)
            or Path(path_value).resolve() != expected_path
            or not _is_lower_hex(digest, 64)
            or isinstance(byte_length, bool)
            or not isinstance(byte_length, int)
            or byte_length < 0
            or not expected_path.is_file()
            or expected_path.stat().st_size != byte_length
        ):
            raise ValueError(f"source snapshot identity is invalid: {document_id}")
        actual_digest = sha256()
        with expected_path.open("rb") as source:
            while chunk := source.read(8 * 1024 * 1024):
                actual_digest.update(chunk)
        if actual_digest.hexdigest() != digest:
            raise ValueError(f"source snapshot checksum failed: {document_id}")
        result[document_id] = str(digest)
    return result


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validated_stage_page(
    index: Mapping[tuple[str, int], tuple[Path, ...]],
    key: tuple[str, int],
    *,
    stage: str,
    source_sha256: str,
) -> tuple[dict[str, Any] | None, str, str | None, str | None]:
    value, availability, path = _single_page(index, key)
    if value is None:
        return None, availability, path, None
    try:
        validate_corpus_stage_page(
            value,
            stage=stage,
            source_sha256=source_sha256,
            page_number=key[1],
        )
    except (KeyError, TypeError, ValueError) as error:
        return (
            None,
            "invalid_contract",
            path,
            f"{type(error).__name__}: {error}",
        )
    return value, availability, path, None


def _method1_baseline(
    options: CorpusCompareOptions,
    document_id: str,
    page_number: int,
) -> tuple[dict[str, Any] | None, str, str]:
    path = options.method1_baseline_root / document_id / f"page-{page_number}.json"
    value = _load_object(path)
    if value is None:
        return None, "missing" if not path.exists() else "invalid", str(path)
    if value.get("cache_version") != FROZEN_TS_METHOD1_CACHE_NAMESPACE:
        return None, "incompatible_version", str(path)
    if value.get("document_id") != document_id or value.get("page") != page_number:
        return None, "invalid_identity", str(path)
    if not isinstance(value.get("result"), Mapping):
        return None, "invalid", str(path)
    try:
        _result(value["result"], "TypeScript Method1")
    except ValueError:
        return None, "invalid", str(path)
    return value, "available", str(path)


def _method2_baseline_index(
    root: Path,
) -> tuple[dict[str, list[tuple[Path, dict[str, Any], bool]]], int]:
    output: dict[str, list[tuple[Path, dict[str, Any], bool]]] = defaultdict(list)
    unindexed = 0
    for path in root.glob("*.json"):
        value = _load_object(path)
        if value is None or not isinstance(value.get("page_identity"), str):
            unindexed += 1
            continue
        exact = (
            value.get("schema_version") == 1
            and value.get("engine_version") == FROZEN_TS_METHOD2_ENGINE_VERSION
            and value.get("config_hash") == FROZEN_TS_METHOD2_CONFIG_HASH
            and isinstance(value.get("result"), Mapping)
        )
        output[value["page_identity"]].append((path, value, exact))
    return output, unindexed


def _method2_baseline(
    index: Mapping[str, Sequence[tuple[Path, dict[str, Any], bool]]],
    page_identity: str,
) -> tuple[dict[str, Any] | None, str, str | None]:
    candidates = index.get(page_identity, ())
    exact = [(path, value) for path, value, compatible in candidates if compatible]
    if len(exact) > 1:
        return None, "duplicate_current", ", ".join(str(path) for path, _ in exact)
    if len(exact) == 0:
        return (
            None,
            "incompatible_version" if candidates else "missing",
            ", ".join(str(path) for path, _, _ in candidates) or None,
        )
    path, value = exact[0]
    try:
        _result(value["result"], "TypeScript Method2")
    except ValueError:
        return None, "invalid", str(path)
    return value, "available", str(path)


_FAMILY_AUDIT_ARRAY_KEYS = (
    "family_diagnostics",
    "matched_text_op_indices",
    "line_type_confirmed_text_op_indices",
    "affected_group_ids",
    "region_instances",
)
_FAMILY_AUDIT_COUNT_KEYS = (
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
_FUSION_AUDIT_COUNT_KEYS = (
    "method2_global_type_count",
    "method2_op_count",
    "retained_method1_global_type_count",
    "retained_method1_op_count",
    "skipped_duplicate_method1_global_type_count",
    "skipped_duplicate_method1_op_count",
    "duplicate_overlap_op_count",
)


def _is_nonnegative_integer(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and 0 <= value <= 9_007_199_254_740_991
    )


def _is_exact_integer(value: object, expected: int) -> bool:
    return _is_nonnegative_integer(value) and value == expected


def _is_finite_nonnegative_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _valid_headless_input_identity(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"path", "file_name", "size", "sha256"}
        and isinstance(value.get("path"), str)
        and bool(value.get("path"))
        and isinstance(value.get("file_name"), str)
        and bool(value.get("file_name"))
        and _is_nonnegative_integer(value.get("size"))
        and _is_lower_hex(value.get("sha256"), 64)
    )


def _validate_family_audit(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if any(not isinstance(value.get(key), list) for key in _FAMILY_AUDIT_ARRAY_KEYS):
        raise ValueError(f"{label} array fields are invalid")
    if any(
        not _is_nonnegative_integer(value.get(key))
        for key in _FAMILY_AUDIT_COUNT_KEYS
    ):
        raise ValueError(f"{label} count fields are invalid")
    return value


def _validate_ts_method1_envelope(
    value: object,
    *,
    page_identity: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("headless Method1 envelope must be an object")
    result = value.get("result")
    if (
        not _is_exact_integer(value.get("schema_version"), 1)
        or value.get("engine_version") != FROZEN_TS_METHOD1_ENGINE_VERSION
        or value.get("page_identity") != page_identity
        or not _is_lower_hex(value.get("input_hash"), 64)
        or not isinstance(result, Mapping)
        or not _is_exact_integer(result.get("schema_version"), 1)
    ):
        raise ValueError("headless Method1 identity is invalid")
    _result(value["result"], "TypeScript headless Method1")
    return dict(value)


def _validate_ts_method2_envelope(
    value: object,
    *,
    page_identity: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("headless Method2 envelope must be an object")
    result = value.get("result")
    if (
        not _is_exact_integer(value.get("schema_version"), 1)
        or value.get("engine_version") != FROZEN_TS_METHOD2_ENGINE_VERSION
        or value.get("config_hash") != FROZEN_TS_METHOD2_CONFIG_HASH
        or value.get("page_identity") != page_identity
        or dict(value.get("features", {})) != dict(LINE_TYPE_METHOD2_FEATURES)
        or not isinstance(result, Mapping)
        or not _is_exact_integer(result.get("schema_version"), 1)
    ):
        raise ValueError("headless Method2 identity is invalid")
    _result(value["result"], "TypeScript headless Method2")
    audit = value.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("headless Method2 audit is invalid")
    input_hash = audit.get("input_hash")
    if (
        not _is_lower_hex(input_hash, 16)
        or audit.get("deterministic_replay_key")
        != f"{page_identity}:{FROZEN_TS_METHOD2_CONFIG_HASH}:{input_hash}"
        or not _is_finite_nonnegative_number(audit.get("elapsed_ms"))
    ):
        raise ValueError("headless Method2 replay identity is invalid")
    _validate_family_audit(
        audit.get("repeated_vector_text_family_clustering"),
        "headless Method2 family audit",
    )
    return dict(value)


def _validate_ts_fused_envelope(
    value: object,
    *,
    page_identity: str,
    method2_input_hash: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("headless fused envelope must be an object")
    result_value = value.get("result")
    if (
        not _is_exact_integer(
            value.get("schema_version"), FROZEN_TS_FUSED_RESULT_SCHEMA_VERSION
        )
        or value.get("engine_version") != FUSED_LINE_TYPE_TARGET_SPEC_VERSION
        or value.get("config_hash") != FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH
        or dict(value.get("features", {})) != dict(FUSED_LINE_TYPE_FEATURES)
        or value.get("method1_engine_version")
        != FROZEN_TS_METHOD1_ENGINE_VERSION
        or value.get("method2_engine_version")
        != FROZEN_TS_METHOD2_ENGINE_VERSION
        or value.get("fusion_policy_version")
        != FROZEN_TS_FUSION_POLICY_VERSION
        or value.get("method2_config_hash") != FROZEN_TS_METHOD2_CONFIG_HASH
        or value.get("page_identity") != page_identity
        or not isinstance(result_value, Mapping)
        or not _is_exact_integer(result_value.get("schema_version"), 1)
    ):
        raise ValueError("headless fused identity is invalid")
    result = _result(value["result"], "TypeScript headless fused")
    raw_global_types = value["result"].get("global_types")
    if not isinstance(raw_global_types, list):
        raise ValueError("headless fused global types are invalid")
    for global_type in raw_global_types:
        if (
            not isinstance(global_type, Mapping)
            or global_type.get("recognition_source") not in {"method1", "method2"}
            or not isinstance(global_type.get("source_global_type_id"), str)
            or not global_type.get("source_global_type_id")
            or not isinstance(global_type.get("type_uid"), str)
            or not global_type.get("type_uid")
        ):
            raise ValueError("headless fused source identity is invalid")
    if len(result.global_types) != len(raw_global_types):
        raise ValueError("headless fused global type parsing is incomplete")

    audit = value.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("headless fused audit is invalid")
    if (
        audit.get("level") not in {"summary", "full"}
        or audit.get("input_hash") != method2_input_hash
        or audit.get("deterministic_replay_key")
        != (
            f"{page_identity}:{FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH}:"
            f"{method2_input_hash}"
        )
        or not _is_finite_nonnegative_number(audit.get("elapsed_ms"))
    ):
        raise ValueError("headless fused replay identity is invalid")
    statuses = audit.get("statuses")
    if (
        not isinstance(statuses, Mapping)
        or set(statuses) != {"line_type", "non_linetype", "uncertain", "skipped"}
        or any(not _is_nonnegative_integer(item) for item in statuses.values())
    ):
        raise ValueError("headless fused statuses are invalid")
    rules = audit.get("rules")
    if not isinstance(rules, Mapping):
        raise ValueError("headless fused rules are invalid")
    _validate_family_audit(
        rules.get("repeated_vector_text_family_clustering"),
        "headless fused family audit",
    )
    fusion_audit = rules.get("method1_method2_display_fusion")
    if (
        not isinstance(fusion_audit, Mapping)
        or any(
            not _is_nonnegative_integer(fusion_audit.get(key))
            for key in _FUSION_AUDIT_COUNT_KEYS
        )
        or not isinstance(
            fusion_audit.get("skipped_method1_global_type_ids"), list
        )
        or any(
            not isinstance(item, str)
            for item in fusion_audit["skipped_method1_global_type_ids"]
        )
    ):
        raise ValueError("headless fused fusion audit is invalid")
    trace = audit.get("trace")
    if trace is not None:
        if not isinstance(trace, list):
            raise ValueError("headless fused trace is invalid")
        for item in trace:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("global_type_id"), str)
                or not item.get("global_type_id")
                or not isinstance(item.get("type_uid"), str)
                or not item.get("type_uid")
                or not isinstance(item.get("member_type_ids"), list)
                or any(
                    not isinstance(member, str)
                    for member in item["member_type_ids"]
                )
            ):
                raise ValueError("headless fused trace entry is invalid")
    return dict(value)


def _headless_unavailable(
    status: str,
    path: Path | None,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "method1": None,
        "method2": None,
        "fused": None,
        "availability": {
            "method1": status,
            "method2": status,
            "fused": status,
        },
        "path": str(path) if path is not None else None,
        "errors": {
            "method1": error,
            "method2": error,
            "fused": error,
        },
    }


def _headless_oracle_index(
    root: Path,
    expected: Sequence[tuple[str, int]],
    source_hashes: Mapping[str, str],
    source_sizes: Mapping[str, int],
    document_page_counts: Mapping[str, int],
) -> tuple[dict[tuple[str, int], dict[str, object]], Counter[str]]:
    expected_by_document: dict[str, tuple[int, ...]] = {}
    for document_id in sorted({document_id for document_id, _ in expected}):
        expected_by_document[document_id] = tuple(
            page for candidate, page in expected if candidate == document_id
        )

    index: dict[tuple[str, int], dict[str, object]] = {}
    issues: Counter[str] = Counter()
    expected_documents = set(expected_by_document)
    unexpected_runs = {
        path.parent.name
        for path in root.glob("*/run.json")
        if path.parent.name not in expected_documents
    }
    if unexpected_runs:
        issues["unexpected_document"] += len(unexpected_runs)

    for document_id, pages in expected_by_document.items():
        document_root = root / document_id
        run_path = document_root / "run.json"
        if (document_root.joinpath(".line-type-run.lock").exists()):
            error = "headless oracle run is locked"
            issues["locked"] += 1
            for page_number in pages:
                index[(document_id, page_number)] = _headless_unavailable(
                    "invalid_contract", run_path, error
                )
            continue
        run = _load_object(run_path)
        if run is None:
            status = "missing" if not run_path.exists() else "invalid_contract"
            if status != "missing":
                issues[status] += 1
            for page_number in pages:
                index[(document_id, page_number)] = _headless_unavailable(
                    status, run_path, "headless run.json is missing or malformed"
                )
            continue

        input_identity = run.get("input")
        summary = run.get("summary")
        manifest_pages = run.get("pages")
        expected_page_count = document_page_counts[document_id]
        expected_page_list = list(pages)
        run_error: str | None = None
        run_status = "invalid_contract"
        if (
            not _is_exact_integer(
                run.get("schema_version"), FROZEN_TS_HEADLESS_RUN_SCHEMA_VERSION
            )
            or run.get("status") != "complete"
            or not _is_exact_integer(run.get("page_count"), expected_page_count)
            or run.get("requested_pages") != expected_page_list
            or any(
                not _is_nonnegative_integer(page)
                for page in run.get("requested_pages", ())
            )
            or run.get("requested_outputs") != list(FROZEN_TS_HEADLESS_OUTPUTS)
            or run.get("analyzer_identity") != FROZEN_TS_HEADLESS_ANALYZER_ID
            or not _valid_headless_input_identity(input_identity)
            or not _is_nonnegative_integer(run.get("page_concurrency"))
            or run.get("page_concurrency") == 0
            or not _is_nonnegative_integer(run.get("workers_per_page"))
            or run.get("workers_per_page") == 0
            or not isinstance(run.get("resume"), bool)
            or not isinstance(run.get("started_at"), str)
            or not run.get("started_at")
            or not isinstance(run.get("completed_at"), str)
            or not run.get("completed_at")
            or not isinstance(summary, Mapping)
            or not isinstance(manifest_pages, list)
        ):
            run_error = "headless run contract/version/page set is invalid"
        elif (
            input_identity.get("sha256") != source_hashes[document_id]
            or input_identity.get("size") != source_sizes[document_id]
        ):
            run_status = "source_mismatch"
            run_error = "headless source SHA-256/size differs from Python snapshot"
        elif (
            summary.get("requested") != len(pages)
            or summary.get("failed") != 0
            or not _is_nonnegative_integer(summary.get("succeeded"))
            or not _is_nonnegative_integer(summary.get("resumed"))
            or summary.get("succeeded") + summary.get("resumed") != len(pages)
        ):
            run_error = "headless run summary is invalid"
        if run_error is not None:
            issues[run_status] += 1
            for page_number in pages:
                index[(document_id, page_number)] = _headless_unavailable(
                    run_status, run_path, run_error
                )
            continue

        manifest_by_page: dict[int, Mapping[str, Any]] = {}
        malformed_manifest = False
        for item in manifest_pages:
            if (
                not isinstance(item, Mapping)
                or not isinstance(item.get("page_number"), int)
                or isinstance(item.get("page_number"), bool)
                or item["page_number"] in manifest_by_page
            ):
                malformed_manifest = True
                break
            manifest_by_page[item["page_number"]] = item
        if malformed_manifest or set(manifest_by_page) != set(pages):
            issues["invalid_contract"] += 1
            for page_number in pages:
                index[(document_id, page_number)] = _headless_unavailable(
                    "invalid_contract", run_path,
                    "headless manifest page set is invalid",
                )
            continue

        pages_root = document_root / "pages"
        expected_names = {
            f"page-{page_number:06d}.json" for page_number in pages
        }
        actual_names = (
            {path.name for path in pages_root.iterdir()}
            if pages_root.is_dir() else set()
        )
        unexpected_names = actual_names - expected_names
        missing_names = expected_names - actual_names
        if unexpected_names:
            issues["unexpected_page"] += len(unexpected_names)
        if missing_names:
            issues["missing_manifest_page"] += len(missing_names)

        for page_number in pages:
            key = (document_id, page_number)
            path = pages_root / f"page-{page_number:06d}.json"
            manifest_page = manifest_by_page[page_number]
            expected_relative = f"pages/page-{page_number:06d}.json"
            if (
                manifest_page.get("status") != "success"
                or not isinstance(manifest_page.get("resumed"), bool)
                or manifest_page.get("result_path") != expected_relative
                or not _is_finite_nonnegative_number(
                    manifest_page.get("duration_ms")
                )
            ):
                issues["invalid_contract"] += 1
                index[key] = _headless_unavailable(
                    "invalid_contract", path,
                    "headless manifest page entry is invalid",
                )
                continue
            page = _load_object(path)
            if page is None:
                status = "invalid_contract"
                if path.exists():
                    issues[status] += 1
                index[key] = _headless_unavailable(
                    status, path, "headless page is missing or malformed"
                )
                continue
            if (
                not _is_exact_integer(
                    page.get("schema_version"),
                    FROZEN_TS_HEADLESS_PAGE_SCHEMA_VERSION,
                )
                or page.get("status") != "success"
                or not _is_exact_integer(page.get("page_number"), page_number)
                or not _is_exact_integer(
                    page.get("page_count"), expected_page_count
                )
                or page.get("requested_outputs")
                != list(FROZEN_TS_HEADLESS_OUTPUTS)
                or page.get("analyzer_identity") != FROZEN_TS_HEADLESS_ANALYZER_ID
                or not isinstance(page.get("input"), Mapping)
                or dict(page["input"]) != dict(input_identity)
                or not isinstance(page.get("generated_at"), str)
                or not page.get("generated_at")
                or not _is_finite_nonnegative_number(page.get("duration_ms"))
                or not _is_nonnegative_integer(page.get("workers_per_page"))
                or page.get("workers_per_page") == 0
                or not isinstance(page.get("outputs"), Mapping)
                or set(page["outputs"]) != set(FROZEN_TS_HEADLESS_OUTPUTS)
            ):
                issues["invalid_contract"] += 1
                index[key] = _headless_unavailable(
                    "invalid_contract", path,
                    "headless page identity/output set is invalid",
                )
                continue

            expected_identity = (
                f"sha256:{source_hashes[document_id]}:page:{page_number}"
            )
            availability: dict[str, str] = {}
            errors: dict[str, str | None] = {}
            values: dict[str, dict[str, Any] | None] = {}
            try:
                values["method1"] = _validate_ts_method1_envelope(
                    page["outputs"].get("method1"),
                    page_identity=expected_identity,
                )
                availability["method1"] = "available"
                errors["method1"] = None
            except (TypeError, ValueError, KeyError) as error:
                values["method1"] = None
                availability["method1"] = "invalid_contract"
                errors["method1"] = f"{type(error).__name__}: {error}"
                issues["invalid_contract"] += 1
            try:
                values["method2"] = _validate_ts_method2_envelope(
                    page["outputs"].get("method2"),
                    page_identity=expected_identity,
                )
                availability["method2"] = "available"
                errors["method2"] = None
            except (TypeError, ValueError, KeyError) as error:
                values["method2"] = None
                availability["method2"] = "invalid_contract"
                errors["method2"] = f"{type(error).__name__}: {error}"
                issues["invalid_contract"] += 1
            method2 = values["method2"]
            method2_audit = method2.get("audit") if method2 is not None else None
            try:
                if not isinstance(method2_audit, Mapping):
                    raise ValueError("validated Method2 audit is unavailable")
                values["fused"] = _validate_ts_fused_envelope(
                    page["outputs"].get("fused"),
                    page_identity=expected_identity,
                    method2_input_hash=str(method2_audit["input_hash"]),
                )
                availability["fused"] = "available"
                errors["fused"] = None
            except (TypeError, ValueError, KeyError) as error:
                values["fused"] = None
                availability["fused"] = "invalid_contract"
                errors["fused"] = f"{type(error).__name__}: {error}"
                issues["invalid_contract"] += 1
            index[key] = {
                **values,
                "availability": availability,
                "path": str(path),
                "errors": errors,
            }
    return index, issues


def _page_stage_result(
    page: Mapping[str, Any] | None,
    stage: str,
) -> Mapping[str, Any] | None:
    if page is None:
        return None
    envelope = page.get(stage)
    if not isinstance(envelope, Mapping):
        return None
    value = envelope.get("result")
    return value if isinstance(value, Mapping) else None


def _indices_within(result: LineTypeRecognitionResult, operation_count: int) -> bool:
    indices = (
        op_index
        for group in result.groups
        for values in (
            *(line_type.op_indices for line_type in group.line_types),
            group.non_linetype.op_indices,
        )
        for op_index in values
    )
    if any(index < 0 or index >= operation_count for index in indices):
        return False
    return all(
        0 <= index < operation_count
        for global_type in result.global_types
        for index in global_type.op_indices
    )


def assess_input_alignment(
    python_page: Mapping[str, Any] | None,
    ts_method1: Mapping[str, Any] | None,
    python_result: Mapping[str, Any] | None,
) -> dict[str, object]:
    """Return the only gate that authorizes direct dense-index comparison."""

    if python_page is None or ts_method1 is None:
        return {"status": "input_alignment_unproven", "reason": "input artifact or Method1 baseline missing"}
    python_hash = python_page.get("method1_input_hash")
    ts_hash = ts_method1.get("method1_input_hash")
    if not (
        python_page.get("method1_input_hash_schema")
        == METHOD1_SERIALIZED_INPUT_HASH_SCHEMA
        and _is_lower_hex(python_hash, 64)
        and _is_lower_hex(ts_hash, 64)
    ):
        return {"status": "input_alignment_unproven", "reason": "comparable Method1 input hash is unavailable"}
    if python_hash != ts_hash:
        return {
            "status": "input_mismatch",
            "reason": "serialized Method1 input hashes differ",
            "python_method1_input_hash": python_hash,
            "typescript_method1_input_hash": ts_hash,
        }
    if python_result is None or not isinstance(ts_method1.get("result"), Mapping):
        return {"status": "input_alignment_unproven", "reason": "result Group coverage is unavailable"}
    try:
        actual = _result(python_result, "Python")
        expected = _result(ts_method1["result"], "TypeScript Method1")
        operation_count = python_page.get("operation_count")
        group_count = python_page.get("group_count")
        if (
            isinstance(operation_count, bool)
            or not isinstance(operation_count, int)
            or operation_count < 0
            or isinstance(group_count, bool)
            or not isinstance(group_count, int)
            or group_count < 0
        ):
            raise ValueError("page operation/group counts are invalid")
        for candidate in (actual, expected):
            if (
                len(candidate.groups) != group_count
                or candidate.summary.input_group_count != group_count
                or candidate.summary.processed_group_count != group_count
                or len({group.group_id for group in candidate.groups}) != group_count
                or not _indices_within(candidate, operation_count)
            ):
                raise ValueError("result does not prove complete Group/index coverage")
        if {group.group_id for group in actual.groups} != {
            group.group_id for group in expected.groups
        }:
            raise ValueError("Python and TypeScript Group identities differ")
    except ValueError as error:
        return {"status": "input_alignment_unproven", "reason": str(error)}
    return {
        "status": "proven",
        "reason": "serialized input hash and complete Group coverage match",
        "method1_input_hash": python_hash,
        "operation_count": operation_count,
        "group_count": group_count,
    }


def _comparison_with_gate(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    tentative = compare_result_partitions(actual, expected)
    status = alignment.get("status")
    return {
        "status": (
            "match" if tentative["exact_match"] else "different"
        ) if status == "proven" else status,
        "officially_comparable": status == "proven",
        "tentative_partition_equal": tentative["exact_match"],
        "partition": tentative,
    }


def _safe_comparison_with_gate(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    alignment: Mapping[str, object],
) -> dict[str, object]:
    try:
        return _comparison_with_gate(actual, expected, alignment)
    except (TypeError, ValueError, KeyError) as error:
        return {
            "status": "invalid_contract",
            "officially_comparable": False,
            "error": f"{type(error).__name__}: {error}",
        }


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def compare_corpus(options: CorpusCompareOptions) -> dict[str, object]:
    if (options.run_root / "corpus-run.lock").exists():
        raise ValueError("corpus run is locked; comparison requires a quiescent run")
    run_manifest = _load_object(options.run_root / "corpus-run.json")
    if run_manifest is None:
        raise ValueError("corpus-run.json is missing or malformed")
    if run_manifest.get("schema_version") != CORPUS_RUN_SCHEMA_VERSION:
        raise ValueError("corpus-run.json schema is incompatible")
    run_status = run_manifest.get("status")
    if run_status in {"snapshotting", "running"}:
        raise ValueError("corpus run is still active")
    if not isinstance(run_status, str) or not run_status:
        raise ValueError("corpus run status is invalid")
    configuration = run_manifest.get("configuration")
    planned_stages = (
        configuration.get("stages") if isinstance(configuration, Mapping) else None
    )
    if (
        not isinstance(planned_stages, list)
        or not planned_stages
        or any(stage not in {"input", "method1", "method2", "compose", "fused"}
               for stage in planned_stages)
        or ("fused" in planned_stages and planned_stages != ["fused"])
    ):
        raise ValueError("corpus run stage configuration is invalid")
    plan = run_manifest.get("plan")
    if (
        not isinstance(plan, Mapping)
        or plan.get("plan_fingerprint") != run_manifest.get("plan_fingerprint")
    ):
        raise ValueError("corpus run plan fingerprint is inconsistent")
    input_index = _index_python_pages(options.run_root, "input")
    method1_index = _index_python_pages(options.run_root, "method1")
    method2_index = _index_python_pages(options.run_root, "method2")
    fused_index = _index_python_pages(options.run_root, "fused")
    composed_index = _index_composed_pages(options.run_root)
    keys = _expected_pages(run_manifest)
    expected_keys = set(keys)
    unexpected = {
        stage: sorted(set(index) - expected_keys)
        for stage, index in (
            ("input", input_index),
            ("method1", method1_index),
            ("method2", method2_index),
            ("fused", fused_index),
            ("composed", composed_index),
        )
        if set(index) - expected_keys
    }
    if unexpected:
        formatted = "; ".join(
            f"{stage}={','.join(f'{document}:P{page}' for document, page in pages)}"
            for stage, pages in sorted(unexpected.items())
        )
        raise ValueError(f"unexpected page artifacts outside the corpus plan: {formatted}")
    source_hashes = _source_hashes(run_manifest, options.run_root, keys)
    raw_sources = run_manifest["sources"]
    assert isinstance(raw_sources, Mapping)
    source_sizes = {
        document_id: int(raw_sources[document_id]["byte_length"])
        for document_id in source_hashes
    }
    raw_documents = plan["documents"]
    assert isinstance(raw_documents, list)
    document_page_counts = {
        str(document["document_id"]): int(document["page_count"])
        for document in raw_documents
        if isinstance(document, Mapping)
    }
    headless_oracles: dict[tuple[str, int], dict[str, object]] = {}
    headless_issues: Counter[str] = Counter()
    method2_baselines: dict[
        str, list[tuple[Path, dict[str, Any], bool]]
    ] = {}
    unindexed_method2_baselines = 0
    if options.headless_oracle_root is not None:
        headless_oracles, headless_issues = _headless_oracle_index(
            options.headless_oracle_root,
            keys,
            source_hashes,
            source_sizes,
            document_page_counts,
        )
    else:
        assert options.method2_baseline_root is not None
        method2_baselines, unindexed_method2_baselines = (
            _method2_baseline_index(options.method2_baseline_root)
        )

    records: list[dict[str, object]] = []
    method1_coverage: Counter[str] = Counter()
    method2_coverage: Counter[str] = Counter()
    fusion_coverage: Counter[str] = Counter()
    alignment_coverage: Counter[str] = Counter()
    method1_comparisons: Counter[str] = Counter()
    method2_comparisons: Counter[str] = Counter()
    fusion_comparisons: Counter[str] = Counter()
    method1_tentative: Counter[str] = Counter()
    method2_tentative: Counter[str] = Counter()
    fusion_tentative: Counter[str] = Counter()
    artifact_coverage: dict[str, Counter[str]] = {
        stage: Counter()
        for stage in ("input", "method1", "method2", "fused", "composed")
    }

    for document_id, page_number in keys:
        key = (document_id, page_number)
        page_identity = f"{document_id}:page:{page_number}"
        source_sha256 = source_hashes[document_id]
        (
            input_page, input_availability, input_path, input_error,
        ) = _validated_stage_page(
            input_index,
            key,
            stage="input",
            source_sha256=source_sha256,
        )
        (
            method1_page, method1_availability, method1_path, method1_error,
        ) = _validated_stage_page(
            method1_index,
            key,
            stage="method1",
            source_sha256=source_sha256,
        )
        (
            method2_page, method2_availability, method2_path, method2_error,
        ) = _validated_stage_page(
            method2_index,
            key,
            stage="method2",
            source_sha256=source_sha256,
        )
        (
            fused_page, fused_availability, fused_path, fused_error,
        ) = _validated_stage_page(
            fused_index,
            key,
            stage="fused",
            source_sha256=source_sha256,
        )
        composed_page, composed_availability, composed_path = _single_page(composed_index, key)
        composed_error: str | None = None
        if composed_page is not None:
            if method1_page is None or method2_page is None:
                composed_page = None
                composed_availability = "unverifiable_stage_inputs"
                composed_error = "validated Method1 and Method2 stage pages are required"
            else:
                try:
                    validate_composed_corpus_page(
                        composed_page,
                        document_id=document_id,
                        page_number=page_number,
                        method1_page=method1_page,
                        method2_page=method2_page,
                    )
                except (KeyError, TypeError, ValueError) as error:
                    composed_page = None
                    composed_availability = "invalid_contract"
                    composed_error = f"{type(error).__name__}: {error}"
        for stage, availability in (
            ("input", input_availability),
            ("method1", method1_availability),
            ("method2", method2_availability),
            ("fused", fused_availability),
            ("composed", composed_availability),
        ):
            artifact_coverage[stage][availability] += 1
        ts_method1: dict[str, Any] | None
        ts_method2: dict[str, Any] | None
        ts_fused: dict[str, Any] | None
        ts_method1_error: str | None = None
        ts_method2_error: str | None = None
        ts_fused_error: str | None = None
        if options.headless_oracle_root is not None:
            oracle = headless_oracles.get(key)
            if oracle is None:
                oracle = _headless_unavailable(
                    "missing", None, "headless page is not indexed"
                )
            raw_availability = oracle["availability"]
            raw_errors = oracle["errors"]
            assert isinstance(raw_availability, Mapping)
            assert isinstance(raw_errors, Mapping)
            ts_method1_envelope = oracle.get("method1")
            ts_method2 = oracle.get("method2")
            ts_fused = oracle.get("fused")
            ts_method1_availability = str(raw_availability["method1"])
            ts_method2_availability = str(raw_availability["method2"])
            ts_fused_availability = str(raw_availability["fused"])
            ts_method1_path = str(oracle["path"]) if oracle.get("path") else None
            ts_method2_path = ts_method1_path
            ts_fused_path = ts_method1_path
            ts_method1_error = (
                str(raw_errors["method1"])
                if raw_errors.get("method1") is not None else None
            )
            ts_method2_error = (
                str(raw_errors["method2"])
                if raw_errors.get("method2") is not None else None
            )
            ts_fused_error = (
                str(raw_errors["fused"])
                if raw_errors.get("fused") is not None else None
            )
            if isinstance(ts_method1_envelope, Mapping):
                ts_method1 = {
                    "method1_input_hash": ts_method1_envelope["input_hash"],
                    "result": ts_method1_envelope["result"],
                }
            else:
                ts_method1 = None
            if not isinstance(ts_method2, dict):
                ts_method2 = None
            if not isinstance(ts_fused, dict):
                ts_fused = None
        else:
            ts_method1, ts_method1_availability, ts_method1_path = (
                _method1_baseline(options, document_id, page_number)
            )
            ts_method2, ts_method2_availability, ts_method2_path = (
                _method2_baseline(method2_baselines, page_identity)
            )
            # Historical web caches do not contain an independently persisted
            # fused TypeScript payload.  Never synthesize that oracle here.
            ts_fused = None
            ts_fused_availability = "missing"
            ts_fused_path = None
        method1_coverage[ts_method1_availability] += 1
        method2_coverage[ts_method2_availability] += 1
        fusion_coverage[ts_fused_availability] += 1

        # A strict fused page is simultaneously the Python Method1, Method2,
        # and fusion artifact, all derived from one shared PageIR/Grouping.
        python_method1 = _page_stage_result(fused_page or method1_page, "method1")
        python_method2 = _page_stage_result(fused_page or method2_page, "method2")
        python_input = fused_page or method1_page or input_page
        alignment_result = python_method1 or python_method2
        alignment = assess_input_alignment(
            python_input, ts_method1, alignment_result
        )
        alignment_coverage[str(alignment["status"])] += 1

        record: dict[str, object] = {
            "document_id": document_id,
            "page_number": page_number,
            "page_identity": page_identity,
            "python_artifacts": {
                "input": {"availability": input_availability, "path": input_path, "error": input_error},
                "method1": {"availability": method1_availability, "path": method1_path, "error": method1_error},
                "method2": {"availability": method2_availability, "path": method2_path, "error": method2_error},
                "fused": {"availability": fused_availability, "path": fused_path, "error": fused_error},
                "composed": {"availability": composed_availability, "path": composed_path, "error": composed_error},
            },
            "input_alignment": alignment,
            "method1_baseline": {
                "availability": ts_method1_availability,
                "path": ts_method1_path,
                "error": ts_method1_error,
                "required_engine_version": FROZEN_TS_METHOD1_ENGINE_VERSION,
            },
            "method2_baseline": {
                "availability": ts_method2_availability,
                "path": ts_method2_path,
                "error": ts_method2_error,
                "required_engine_version": FROZEN_TS_METHOD2_ENGINE_VERSION,
                "required_config_hash": FROZEN_TS_METHOD2_CONFIG_HASH,
            },
            "fusion_baseline": {
                "availability": ts_fused_availability,
                "path": ts_fused_path,
                "error": ts_fused_error,
                "required_engine_version": FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
                "required_config_hash": FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH,
                "required_fusion_policy_version": FROZEN_TS_FUSION_POLICY_VERSION,
            },
        }

        if python_method1 is not None and ts_method1 is not None:
            comparison = _safe_comparison_with_gate(
                python_method1, ts_method1["result"], alignment
            )
            record["method1_comparison"] = comparison
            method1_comparisons[str(comparison["status"])] += 1
            if "tentative_partition_equal" in comparison:
                method1_tentative[
                    "equal"
                    if comparison["tentative_partition_equal"]
                    else "different"
                ] += 1
        else:
            if python_method1 is None:
                status = "python_result_unavailable"
            elif options.headless_oracle_root is not None:
                status = (
                    "oracle_missing"
                    if ts_method1_availability == "missing"
                    else "oracle_invalid"
                )
            else:
                status = "baseline_unavailable"
            record["method1_comparison"] = {
                "status": status,
                "officially_comparable": False,
                "oracle_availability": ts_method1_availability,
                "error": ts_method1_error,
            }
            method1_comparisons[status] += 1

        if python_method2 is not None and ts_method2 is not None:
            comparison = _safe_comparison_with_gate(
                python_method2, ts_method2["result"], alignment
            )
            record["method2_comparison"] = comparison
            method2_comparisons[str(comparison["status"])] += 1
            if "tentative_partition_equal" in comparison:
                method2_tentative[
                    "equal"
                    if comparison["tentative_partition_equal"]
                    else "different"
                ] += 1
        else:
            if python_method2 is None:
                status = "python_result_unavailable"
            elif options.headless_oracle_root is not None:
                status = (
                    "oracle_missing"
                    if ts_method2_availability == "missing"
                    else "oracle_invalid"
                )
            else:
                status = "baseline_unavailable"
            record["method2_comparison"] = {
                "status": status,
                "officially_comparable": False,
                "oracle_availability": ts_method2_availability,
                "error": ts_method2_error,
            }
            method2_comparisons[status] += 1

        python_fused: Mapping[str, Any] | None = None
        python_fused_page = fused_page or composed_page
        if python_fused_page is not None:
            fused = python_fused_page.get("fused")
            if isinstance(fused, Mapping) and isinstance(fused.get("result"), Mapping):
                python_fused = fused["result"]
        if python_fused is not None and ts_fused is not None:
            fusion = _safe_comparison_with_gate(
                python_fused, ts_fused["result"], alignment
            )
            record["fusion_oracle_comparison"] = fusion
            fusion_comparisons[str(fusion["status"])] += 1
            if "tentative_partition_equal" in fusion:
                fusion_tentative[
                    "equal" if fusion["tentative_partition_equal"] else "different"
                ] += 1
        else:
            if ts_fused is None:
                status = (
                    "oracle_missing"
                    if ts_fused_availability == "missing"
                    else "oracle_invalid"
                )
            else:
                status = "python_composed_unavailable"
            record["fusion_oracle_comparison"] = {
                "status": status,
                "officially_comparable": False,
                "oracle_availability": ts_fused_availability,
                "error": ts_fused_error,
            }
            fusion_comparisons[status] += 1
        records.append(record)

    required_artifacts = {
        "input": "input" in planned_stages,
        "method1": "method1" in planned_stages,
        "method2": "method2" in planned_stages,
        "fused": "fused" in planned_stages,
        "composed": "compose" in planned_stages,
    }
    artifact_failures = {
        stage: sum(count for availability, count in coverage.items()
                   if availability != "available")
        for stage, coverage in artifact_coverage.items()
        if required_artifacts[stage]
    }
    hard_comparison_statuses = {
        "different", "input_mismatch", "invalid_contract", "oracle_invalid"
    }
    hard_comparison_count = sum(
        counter[status]
        for counter in (method1_comparisons, method2_comparisons, fusion_comparisons)
        for status in hard_comparison_statuses
    )
    hard_failure = (
        run_status != "complete"
        or any(artifact_failures.values())
        or hard_comparison_count > 0
        or sum(headless_issues.values()) > 0
    )
    full_stage_sets = (
        {"input", "method1", "method2", "compose"},
        {"fused"},
    )
    inconclusive_statuses = {
        "input_alignment_unproven",
        "baseline_unavailable",
        "oracle_missing",
    }
    inconclusive_count = sum(
        counter[status]
        for counter in (method1_comparisons, method2_comparisons, fusion_comparisons)
        for status in inconclusive_statuses
    )
    verdict = (
        "failed"
        if hard_failure
        else "inconclusive"
        if set(planned_stages) not in full_stage_sets or inconclusive_count > 0
        else "pass"
    )

    aggregate: dict[str, object] = {
        "schema_version": CORPUS_COMPARISON_SCHEMA_VERSION,
        "status": verdict,
        "audit_completed": True,
        "release_gate_passed": verdict == "pass",
        "run_root": str(options.run_root),
        "run_status": run_status,
        "plan_fingerprint": run_manifest.get("plan_fingerprint"),
        "implementation": run_manifest.get("implementation"),
        "page_count": len(records),
        "candidate_artifact_coverage": {
            stage: _counter_dict(coverage)
            for stage, coverage in artifact_coverage.items()
        },
        "release_gate": {
            "required_artifacts": required_artifacts,
            "artifact_failure_count": sum(artifact_failures.values()),
            "artifact_failures_by_stage": artifact_failures,
            "hard_comparison_failure_count": hard_comparison_count,
            "oracle_validation_failure_count": sum(headless_issues.values()),
            "inconclusive_comparison_count": inconclusive_count,
            "reason": (
                "candidate run/artifact/comparison failure"
                if verdict == "failed"
                else "available frozen evidence does not prove complete parity"
                if verdict == "inconclusive"
                else "all required artifacts and available oracle comparisons passed"
            ),
        },
        "baseline_coverage": {
            "method1": _counter_dict(method1_coverage),
            "method2_r46": _counter_dict(method2_coverage),
            "fusion_r46": _counter_dict(fusion_coverage),
            "method2_unindexed_file_count": unindexed_method2_baselines,
            "headless_validation_issues": _counter_dict(headless_issues),
        },
        "input_alignment": _counter_dict(alignment_coverage),
        "comparisons": {
            "method1": _counter_dict(method1_comparisons),
            "method2": _counter_dict(method2_comparisons),
            "fusion_r46_oracle": _counter_dict(fusion_comparisons),
        },
        "fusion_oracle_provenance": (
            {
                "kind": "persisted_headless_typescript_output",
                "root": str(options.headless_oracle_root),
                "independent_frozen_fused_payload": True,
                "engine_version": FUSED_LINE_TYPE_TARGET_SPEC_VERSION,
                "config_hash": FROZEN_TS_FUSED_LINE_TYPE_CONFIG_HASH,
            }
            if options.headless_oracle_root is not None
            else {
                "kind": "unavailable_in_legacy_web_caches",
                "independent_frozen_fused_payload": False,
                "warning": (
                    "Legacy Method1/Method2 caches contain no independent "
                    "fused payload; fusion parity is therefore inconclusive."
                ),
            }
        ),
        "tentative_partition_equality": {
            "method1": _counter_dict(method1_tentative),
            "method2": _counter_dict(method2_tentative),
            "fusion_r46_oracle": _counter_dict(fusion_tentative),
            "warning": (
                "Tentative equality is diagnostic only; it is not a pass when "
                "input_alignment is unproven."
            ),
        },
        "outputs": {
            "per_page_jsonl": str(options.output_directory / "pages.jsonl"),
            "aggregate_json": str(options.output_directory / "aggregate.json"),
        },
    }
    _atomic_write_jsonl(options.output_directory / "pages.jsonl", records)
    _atomic_write_json(options.output_directory / "aggregate.json", aggregate)
    return aggregate


__all__ = [
    "CORPUS_COMPARISON_SCHEMA_VERSION",
    "FROZEN_TS_METHOD1_CACHE_NAMESPACE",
    "CorpusCompareOptions",
    "assess_input_alignment",
    "canonical_result_partitions",
    "compare_corpus",
    "compare_result_partitions",
]
