"""Fail-closed, algorithm-neutral cache for one page's PageIR and GroupingIR.

The cache deliberately stops before Method1, Method2, and fusion.  Its key is
the exact PDF snapshot, page, parser/grouping contracts, and parser source
implementation fingerprint.  A recognition-code edit can therefore reuse an
identical parse, while any parser or grouping edit invalidates it.

Only bounded JSON is accepted.  Pickle and other executable object formats are
never read.  Every hit is reconstructed through the public frozen dataclasses
and then checked against source identity, payload checksum, PageIR/Grouping
fingerprints, the complete Group partition, and the exact source-alignment
audit before it can reach an algorithm.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import threading
import time
from typing import (
    Any,
    Callable,
    Iterator,
    Literal,
    Mapping,
)

from .grouping import DEFAULT_SEQUENTIAL_GROUPING_OPTIONS, group_page_sequentially
from .ir import GroupingIR, PageIR
from .ir_codec import IRCodecError, _decode_typed
from .pdf_adapter import PDF_ADAPTER_VERSION
from .source_content import SOURCE_CONTENT_VERSION
from .source_page_adapter import (
    SOURCE_ALIGNED_PAGE_IR_PRODUCER,
    SOURCE_PAGE_ADAPTER_VERSION,
    SourceAlignmentSummary,
    current_source_aligned_producer_version,
    source_aligned_page_ir_from_pdf_bytes,
)
from .versions import GROUPING_IR_VERSION, PAGE_IR_VERSION


PAGE_ANALYSIS_CACHE_SCHEMA_VERSION = 1
PAGE_ANALYSIS_CACHE_KIND = "python-page-ir-grouping-cache"
# Strict dataclass reconstruction is intentionally more expensive than a bare
# ``json.loads``.  Admission therefore stops at 16 MiB: larger IR files trend
# toward the same wall time as a fresh parse/grouping and create poor disk
# pressure.  They remain protected by the full-result cache and in-flight
# request coalescing in the HTTP layer. The 10k-operation gate runs before JSON
# encoding; the synthetic strict-rehydration benchmark was 1.83 s at that size.
DEFAULT_PAGE_ANALYSIS_CACHE_MAX_FILE_BYTES = 16 * 1024 * 1024
DEFAULT_PAGE_ANALYSIS_CACHE_MAX_TOTAL_BYTES = 1024 * 1024 * 1024
DEFAULT_PAGE_ANALYSIS_CACHE_MAX_OPERATION_COUNT = 10_000
DEFAULT_PAGE_ANALYSIS_CACHE_LOCK_TIMEOUT_SECONDS = 30 * 60.0
_LOCK_STRIPE_COUNT = 64
_THREAD_LOCKS = tuple(threading.Lock() for _ in range(_LOCK_STRIPE_COUNT))
_PRUNE_THREAD_LOCK = threading.Lock()
_PARSER_IMPLEMENTATION_FILES = (
    "annotation_appearances.py",
    "bounded_content_stream.py",
    "geometry.py",
    "grouping.py",
    "ir.py",
    "ir_codec.py",
    "pdf_adapter.py",
    "runtime.py",
    "source_content.py",
    "source_page_adapter.py",
)


class PageAnalysisCacheError(ValueError):
    """A disk entry failed validation and must not be used."""


@dataclass(frozen=True, slots=True)
class CachedPageAnalysis:
    page: PageIR
    grouping: GroupingIR
    source_alignment_audit: SourceAlignmentSummary


@dataclass(frozen=True, slots=True)
class PageAnalysisCacheAccess:
    analysis: CachedPageAnalysis
    state: Literal["hit", "miss", "miss-not-stored", "disabled"]
    key: str
    file_bytes: int
    load_ms: float
    build_ms: float
    write_ms: float
    reason: str = ""

    def diagnostic_dict(self) -> dict[str, object]:
        return {
            "schema_version": PAGE_ANALYSIS_CACHE_SCHEMA_VERSION,
            "state": self.state,
            "key": self.key,
            "file_bytes": self.file_bytes,
            "load_ms": round(self.load_ms, 3),
            "build_ms": round(self.build_ms, 3),
            "write_ms": round(self.write_ms, 3),
            "reason": self.reason,
        }


def _json_value(value: Any) -> Any:
    """Return lossless JSON data while omitting non-contract dataclass caches."""

    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
            if item.metadata.get("canonical", True)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PageAnalysisCacheError("cache JSON does not support non-finite floats")
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise PageAnalysisCacheError(
        f"cache JSON does not support {type(value).__name__}"
    )


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> None:
    raise PageAnalysisCacheError(f"cache JSON contains invalid number {value}")


def _load_json_bytes(source: bytes) -> object:
    try:
        return json.loads(source, parse_constant=_reject_json_constant)
    except (RecursionError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PageAnalysisCacheError("cache file is not valid bounded UTF-8 JSON") from error


def _record(value: object, label: str, expected_keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise PageAnalysisCacheError(f"{label} fields do not match the schema")
    if any(not isinstance(key, str) for key in value):
        raise PageAnalysisCacheError(f"{label} keys must be strings")
    return value


def _implementation_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = sha256()
    for name in _PARSER_IMPLEMENTATION_FILES:
        path = root / name
        source = path.read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(source)).encode("ascii"))
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=1)
def current_page_analysis_cache_contracts() -> dict[str, object]:
    options = DEFAULT_SEQUENTIAL_GROUPING_OPTIONS
    return {
        "page_ir_version": PAGE_IR_VERSION,
        "grouping_ir_version": GROUPING_IR_VERSION,
        "source_content_version": SOURCE_CONTENT_VERSION,
        "pdf_adapter_version": PDF_ADAPTER_VERSION,
        "source_page_adapter_version": SOURCE_PAGE_ADAPTER_VERSION,
        "source_aligned_producer": SOURCE_ALIGNED_PAGE_IR_PRODUCER,
        "source_parser_runtime_version": current_source_aligned_producer_version(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "parser_implementation_sha256": _implementation_fingerprint(),
        "grouping_options": {
            "spatial_jump_ratio": options.spatial_jump_ratio,
            "structured_style_gap_ratio": options.structured_style_gap_ratio,
            "structured_gap_ratio": options.structured_gap_ratio,
        },
    }


def _cache_identity(
    *,
    source_sha256: str,
    source_byte_length: int,
    source_name: str,
    page_number: int,
) -> dict[str, object]:
    return {
        "source_sha256": source_sha256,
        "source_byte_length": source_byte_length,
        "source_name": source_name,
        "page_number": page_number,
        "contracts": current_page_analysis_cache_contracts(),
    }


def page_analysis_cache_key(
    *,
    source_sha256: str,
    source_byte_length: int,
    source_name: str,
    page_number: int,
) -> str:
    return sha256(_json_bytes(_cache_identity(
        source_sha256=source_sha256,
        source_byte_length=source_byte_length,
        source_name=source_name,
        page_number=page_number,
    ))).hexdigest()


def _cache_target(root: Path, key: str) -> Path:
    if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
        raise PageAnalysisCacheError("cache key is not lowercase SHA-256")
    return root / key[:2] / f"{key}.json"


def _validate_partition(page: PageIR, grouping: GroupingIR) -> None:
    if grouping.page_fingerprint != page.fingerprint:
        raise PageAnalysisCacheError("GroupingIR references a different PageIR")
    if grouping.grouping_version != GROUPING_IR_VERSION:
        raise PageAnalysisCacheError("GroupingIR version does not match")
    page_ids = tuple(operation.operation_id for operation in page.operations)
    group_ids = tuple(group.group_id for group in grouping.groups)
    if len(group_ids) != len(set(group_ids)):
        raise PageAnalysisCacheError("GroupingIR contains duplicate Group ids")
    flattened = tuple(
        operation_id
        for group in grouping.groups
        for operation_id in group.operation_ids
    )
    if flattened != page_ids:
        raise PageAnalysisCacheError("Groups are not an ordered complete PageIR partition")
    expected_assignments = tuple(
        (operation_id, group.group_id)
        for group in grouping.groups
        for operation_id in group.operation_ids
    )
    if grouping.assignments != expected_assignments:
        raise PageAnalysisCacheError("Grouping assignments do not match Group ownership")
    operation_by_id = {
        operation.operation_id: operation for operation in page.operations
    }
    for group in grouping.groups:
        operations = tuple(operation_by_id[item] for item in group.operation_ids)
        bounds = operations[0].bounds
        for operation in operations[1:]:
            bounds = bounds.union(operation.bounds)
        if bounds != group.bounds:
            raise PageAnalysisCacheError("Group bounds do not match its operations")
        if (
            group.first_paint_order != operations[0].paint_order
            or group.last_paint_order != operations[-1].paint_order
        ):
            raise PageAnalysisCacheError("Group paint range does not match its operations")


def _validate_analysis(
    analysis: CachedPageAnalysis,
    *,
    source_sha256: str,
    source_name: str,
    page_number: int,
) -> None:
    page = analysis.page
    if (
        page.source_sha256 != source_sha256
        or page.source_name != source_name
        or page.page_number != page_number
    ):
        raise PageAnalysisCacheError("PageIR source identity does not match")
    contracts = current_page_analysis_cache_contracts()
    if (
        page.page_ir_version != contracts["page_ir_version"]
        or page.producer != contracts["source_aligned_producer"]
        or page.producer_version != contracts["source_parser_runtime_version"]
    ):
        raise PageAnalysisCacheError("PageIR parser identity does not match")
    kind_counts = {"path": 0, "text": 0, "image": 0}
    for operation in page.operations:
        if not operation.source_provenance_exact:
            raise PageAnalysisCacheError("PageIR operation lacks exact source provenance")
        expected_ordinal = kind_counts[operation.kind]
        if operation.ordinal != expected_ordinal:
            raise PageAnalysisCacheError("PageIR operation ordinals are not dense")
        kind_counts[operation.kind] += 1
    _validate_partition(page, analysis.grouping)
    # from_dict is intentionally rerun even for an already constructed
    # summary.  This applies its exact field/count schema at both write and hit.
    SourceAlignmentSummary.from_dict(
        analysis.source_alignment_audit.to_dict(),
        expected_parser_version=page.producer_version,
        expected_operation_count=len(page.operations),
    )


def _payload(analysis: CachedPageAnalysis) -> dict[str, object]:
    return {
        "page": _json_value(analysis.page),
        "grouping": _json_value(analysis.grouping),
        "source_alignment_audit": analysis.source_alignment_audit.to_dict(),
    }


def _envelope(
    analysis: CachedPageAnalysis,
    *,
    key: str,
    identity: Mapping[str, object],
) -> dict[str, object]:
    payload = _payload(analysis)
    return {
        "schema_version": PAGE_ANALYSIS_CACHE_SCHEMA_VERSION,
        "kind": PAGE_ANALYSIS_CACHE_KIND,
        "key": key,
        "identity": identity,
        "page_fingerprint": analysis.page.fingerprint,
        "grouping_fingerprint": analysis.grouping.fingerprint,
        "operation_count": len(analysis.page.operations),
        "group_count": len(analysis.grouping.groups),
        "payload_sha256": sha256(_json_bytes(payload)).hexdigest(),
        "payload": payload,
    }


def _decode_envelope(
    raw: bytes,
    *,
    expected_key: str,
    expected_identity: Mapping[str, object],
) -> CachedPageAnalysis:
    value = _record(_load_json_bytes(raw), "cache envelope", {
        "schema_version",
        "kind",
        "key",
        "identity",
        "page_fingerprint",
        "grouping_fingerprint",
        "operation_count",
        "group_count",
        "payload_sha256",
        "payload",
    })
    if (
        value["schema_version"] != PAGE_ANALYSIS_CACHE_SCHEMA_VERSION
        or value["kind"] != PAGE_ANALYSIS_CACHE_KIND
        or value["key"] != expected_key
        or value["identity"] != expected_identity
    ):
        raise PageAnalysisCacheError("cache identity or contract does not match")
    payload = _record(value["payload"], "cache payload", {
        "page", "grouping", "source_alignment_audit",
    })
    payload_checksum = value["payload_sha256"]
    if (
        not isinstance(payload_checksum, str)
        or payload_checksum != sha256(_json_bytes(payload)).hexdigest()
    ):
        raise PageAnalysisCacheError("cache payload checksum does not match")
    try:
        page = _decode_typed(payload["page"], PageIR, "cache payload.page")
        grouping = _decode_typed(
            payload["grouping"], GroupingIR, "cache payload.grouping"
        )
    except IRCodecError as error:
        raise PageAnalysisCacheError(str(error)) from error
    if not isinstance(page, PageIR) or not isinstance(grouping, GroupingIR):
        raise PageAnalysisCacheError("cache IR reconstruction failed")
    audit_value = payload["source_alignment_audit"]
    if not isinstance(audit_value, Mapping):
        raise PageAnalysisCacheError("source alignment audit must be an object")
    audit = SourceAlignmentSummary.from_dict(
        audit_value,
        expected_parser_version=page.producer_version,
        expected_operation_count=len(page.operations),
    )
    analysis = CachedPageAnalysis(page, grouping, audit)
    _validate_analysis(
        analysis,
        source_sha256=str(expected_identity["source_sha256"]),
        source_name=str(expected_identity["source_name"]),
        page_number=int(expected_identity["page_number"]),
    )
    if (
        value["page_fingerprint"] != page.fingerprint
        or value["grouping_fingerprint"] != grouping.fingerprint
        or value["operation_count"] != len(page.operations)
        or value["group_count"] != len(grouping.groups)
    ):
        raise PageAnalysisCacheError("cache fingerprints or counts do not match")
    return analysis


def _read_entry(
    target: Path,
    *,
    key: str,
    identity: Mapping[str, object],
    max_file_bytes: int,
) -> tuple[CachedPageAnalysis, int]:
    size = target.stat().st_size
    if size < 2 or size > max_file_bytes:
        raise PageAnalysisCacheError("cache file exceeds its byte budget")
    raw = target.read_bytes()
    if len(raw) != size:
        raise PageAnalysisCacheError("cache file changed while it was read")
    analysis = _decode_envelope(
        raw,
        expected_key=key,
        expected_identity=identity,
    )
    try:
        os.utime(target, None)
    except OSError:
        pass
    return analysis, size


def _lock_stripe(key: str) -> int:
    return int(key[:8], 16) % _LOCK_STRIPE_COUNT


@contextmanager
def _file_lock(path: Path, timeout_seconds: float) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as error:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out acquiring cache lock {path.name}") from error
                time.sleep(0.05)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


@contextmanager
def _key_lock(root: Path, key: str, timeout_seconds: float) -> Iterator[None]:
    stripe = _lock_stripe(key)
    with _THREAD_LOCKS[stripe]:
        with _file_lock(root / ".locks" / f"{stripe:02x}.lock", timeout_seconds):
            yield


def _atomic_write(target: Path, body: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.tmp"
    )
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(body)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _prune_locked(root: Path, *, max_total_bytes: int, keep: Path) -> None:
    # Every writer holds the same cross-process prune lock before creating its
    # temporary file, so any temporary still present here is an orphan from a
    # terminated writer and cannot be an active publication.
    for temporary in root.glob("[0-9a-f][0-9a-f]/.*.tmp"):
        try:
            temporary.unlink()
        except OSError:
            pass
    entries: list[tuple[int, int, Path]] = []
    total = 0
    for path in root.glob("[0-9a-f][0-9a-f]/*.json"):
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        entries.append((stat.st_mtime_ns, stat.st_size, path))
    if total <= max_total_bytes:
        return
    for _modified, size, path in sorted(entries):
        if path == keep:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        total -= size
        if total <= max_total_bytes:
            return


def _build_analysis(
    source: bytes,
    page_number: int,
    source_name: str,
) -> CachedPageAnalysis:
    aligned = source_aligned_page_ir_from_pdf_bytes(
        source,
        page_number,
        source_name=source_name,
    )
    grouping = group_page_sequentially(aligned.page)
    audit = SourceAlignmentSummary.from_alignment_audit(
        aligned.audit,
        parser_version=current_source_aligned_producer_version(),
    )
    return CachedPageAnalysis(aligned.page, grouping, audit)


def load_or_build_page_analysis(
    source: bytes | bytearray | memoryview,
    page_number: int,
    *,
    source_name: str,
    cache_root: str | Path | None,
    max_file_bytes: int = DEFAULT_PAGE_ANALYSIS_CACHE_MAX_FILE_BYTES,
    max_total_bytes: int = DEFAULT_PAGE_ANALYSIS_CACHE_MAX_TOTAL_BYTES,
    max_operation_count: int = DEFAULT_PAGE_ANALYSIS_CACHE_MAX_OPERATION_COUNT,
    lock_timeout_seconds: float = DEFAULT_PAGE_ANALYSIS_CACHE_LOCK_TIMEOUT_SECONDS,
    builder: Callable[[], CachedPageAnalysis] | None = None,
) -> PageAnalysisCacheAccess:
    """Load one exact parse/grouping or build it once under a striped lock."""

    if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
        raise ValueError("page_number must be a positive one-based integer")
    if not isinstance(source_name, str):
        raise ValueError("source_name must be a string")
    if max_file_bytes < 1 or max_total_bytes < 1:
        raise ValueError("cache byte budgets must be positive")
    if (
        isinstance(max_operation_count, bool)
        or not isinstance(max_operation_count, int)
        or max_operation_count < 1
    ):
        raise ValueError("cache operation budget must be a positive integer")
    if not math.isfinite(lock_timeout_seconds) or lock_timeout_seconds <= 0:
        raise ValueError("cache lock timeout must be positive and finite")
    source_bytes = bytes(source)
    source_digest = sha256(source_bytes).hexdigest()
    identity = _cache_identity(
        source_sha256=source_digest,
        source_byte_length=len(source_bytes),
        source_name=source_name,
        page_number=page_number,
    )
    key = page_analysis_cache_key(
        source_sha256=source_digest,
        source_byte_length=len(source_bytes),
        source_name=source_name,
        page_number=page_number,
    )
    build = builder or (
        lambda: _build_analysis(source_bytes, page_number, source_name)
    )
    if cache_root is None:
        started = time.perf_counter()
        analysis = build()
        build_ms = (time.perf_counter() - started) * 1000.0
        _validate_analysis(
            analysis,
            source_sha256=source_digest,
            source_name=source_name,
            page_number=page_number,
        )
        return PageAnalysisCacheAccess(
            analysis, "disabled", key, 0, 0.0, build_ms, 0.0, "cache-disabled"
        )

    root = Path(cache_root).resolve()
    target = _cache_target(root, key)
    load_started = time.perf_counter()
    try:
        analysis, file_bytes = _read_entry(
            target,
            key=key,
            identity=identity,
            max_file_bytes=min(max_file_bytes, max_total_bytes),
        )
        return PageAnalysisCacheAccess(
            analysis,
            "hit",
            key,
            file_bytes,
            (time.perf_counter() - load_started) * 1000.0,
            0.0,
            0.0,
        )
    except (OSError, PageAnalysisCacheError, TypeError, ValueError) as error:
        miss_reason = type(error).__name__

    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        build_started = time.perf_counter()
        analysis = build()
        build_ms = (time.perf_counter() - build_started) * 1000.0
        _validate_analysis(
            analysis,
            source_sha256=source_digest,
            source_name=source_name,
            page_number=page_number,
        )
        return PageAnalysisCacheAccess(
            analysis,
            "miss-not-stored",
            key,
            0,
            0.0,
            build_ms,
            0.0,
            f"cache-root-{type(error).__name__}",
        )
    with _key_lock(root, key, lock_timeout_seconds):
        second_load_started = time.perf_counter()
        try:
            analysis, file_bytes = _read_entry(
                target,
                key=key,
                identity=identity,
                max_file_bytes=min(max_file_bytes, max_total_bytes),
            )
            return PageAnalysisCacheAccess(
                analysis,
                "hit",
                key,
                file_bytes,
                (time.perf_counter() - second_load_started) * 1000.0,
                0.0,
                0.0,
                "joined-concurrent-builder",
            )
        except (OSError, PageAnalysisCacheError, TypeError, ValueError):
            pass

        build_started = time.perf_counter()
        analysis = build()
        build_ms = (time.perf_counter() - build_started) * 1000.0
        _validate_analysis(
            analysis,
            source_sha256=source_digest,
            source_name=source_name,
            page_number=page_number,
        )
        if len(analysis.page.operations) > max_operation_count:
            return PageAnalysisCacheAccess(
                analysis,
                "miss-not-stored",
                key,
                0,
                0.0,
                build_ms,
                0.0,
                f"entry-exceeds-{max_operation_count}-operation-budget",
            )
        envelope = _envelope(analysis, key=key, identity=identity)
        body = _json_bytes(envelope)
        effective_limit = min(max_file_bytes, max_total_bytes)
        if len(body) > effective_limit:
            return PageAnalysisCacheAccess(
                analysis,
                "miss-not-stored",
                key,
                len(body),
                0.0,
                build_ms,
                0.0,
                f"entry-exceeds-{effective_limit}-byte-budget",
            )
        write_started = time.perf_counter()
        try:
            with _PRUNE_THREAD_LOCK:
                with _file_lock(
                    root / ".locks" / "prune.lock",
                    lock_timeout_seconds,
                ):
                    _atomic_write(target, body)
                    _prune_locked(root, max_total_bytes=max_total_bytes, keep=target)
            write_ms = (time.perf_counter() - write_started) * 1000.0
            return PageAnalysisCacheAccess(
                analysis,
                "miss",
                key,
                len(body),
                0.0,
                build_ms,
                write_ms,
                miss_reason,
            )
        except (OSError, TimeoutError) as error:
            return PageAnalysisCacheAccess(
                analysis,
                "miss-not-stored",
                key,
                len(body),
                0.0,
                build_ms,
                (time.perf_counter() - write_started) * 1000.0,
                f"cache-write-{type(error).__name__}",
            )
