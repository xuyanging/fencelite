"""Resumable, frontend-free whole-PDF recognition orchestration."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
from typing import Callable, Iterator, Mapping, Sequence

from .engine import (
    OutputKind,
    PageLineTypeRecognition,
    normalize_outputs,
    page_recognition_payload_hash,
    recognize_page,
)
from .pdf_adapter import (
    PDF_ADAPTER_VERSION,
)
from .source_content import SOURCE_CONTENT_VERSION
from .source_page_adapter import (
    SOURCE_ALIGNED_PAGE_IR_PRODUCER,
    SOURCE_PAGE_ADAPTER_VERSION,
    SourceAlignedPdfDocument,
    SourceAlignmentSummary,
    current_source_aligned_producer_version,
)
from .results import LineTypeRecognitionResult
from .runtime import assert_supported_pymupdf_runtime
from .versions import (
    FROZEN_TS_FUSION_POLICY_VERSION,
    FROZEN_TS_METHOD1_ENGINE_VERSION,
    FROZEN_TS_METHOD2_ENGINE_VERSION,
    DOCUMENT_RUN_SCHEMA_VERSION,
    GROUPING_IR_VERSION,
    PAGE_ANALYSIS_SCHEMA_VERSION,
    PAGE_IR_VERSION,
    PYTHON_FUSION_ENGINE_VERSION,
    PYTHON_ENGINE_VERSION,
    PYTHON_METHOD1_ENGINE_VERSION,
    PYTHON_METHOD2_ENGINE_VERSION,
    RESULT_SCHEMA_VERSION,
)


MAX_DOCUMENT_PAGES = 1_000_000
PageAnalyzer = Callable[..., PageLineTypeRecognition]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_page_selection(value: str, page_count: int) -> tuple[int, ...]:
    """Parse ``all`` or comma-separated one-based pages/ranges."""

    if not isinstance(page_count, int) or isinstance(page_count, bool):
        raise ValueError("page_count must be an integer")
    if page_count < 1 or page_count > MAX_DOCUMENT_PAGES:
        raise ValueError(
            f"PDF page count must be within 1..{MAX_DOCUMENT_PAGES}"
        )
    if not isinstance(value, str) or not value.strip():
        raise ValueError("page selection must not be empty")
    normalized = value.strip().lower()
    if normalized == "all":
        return tuple(range(1, page_count + 1))

    selected: set[int] = set()
    for raw_token in normalized.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("page selection contains an empty item")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise ValueError(f"invalid page range {token!r}")
            start, end = (int(part) for part in parts)
            if start < 1 or end < start:
                raise ValueError(f"invalid page range {token!r}")
            if end > page_count:
                raise ValueError(
                    f"page {end} is outside the document's 1..{page_count} range"
                )
            selected.update(range(start, end + 1))
        else:
            if not token.isdigit() or int(token) < 1:
                raise ValueError(f"invalid page number {token!r}")
            selected.add(int(token))
    if not selected:
        raise ValueError("page selection resolved to no pages")
    if max(selected) > page_count:
        raise ValueError(
            f"page {max(selected)} is outside the document's 1..{page_count} range"
        )
    return tuple(sorted(selected))


def _read_source_snapshot(path: Path) -> bytes:
    """Read one bounded-by-file-handle snapshot and detect in-place mutation."""

    with path.open("rb") as source_file:
        before = os.fstat(source_file.fileno())
        source = source_file.read()
        after = os.fstat(source_file.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(source) != before.st_size:
        raise RuntimeError("input PDF changed while its snapshot was being read")
    return source


def _current_page_ir_producer_version() -> str:
    return current_source_aligned_producer_version()


@contextmanager
def _open_pdf_snapshot(
    source: bytes,
    *,
    source_name: str,
) -> Iterator[SourceAlignedPdfDocument]:
    """Open both parsers once over the immutable whole-document snapshot."""

    with SourceAlignedPdfDocument(source, source_name=source_name) as document:
        yield document


@dataclass(frozen=True, slots=True)
class DocumentRunOptions:
    input_pdf: Path
    output_directory: Path
    pages: str = "all"
    outputs: tuple[str, ...] = ("fused",)
    method1_worker_count: int | None = None
    resume: bool = True

    def __post_init__(self) -> None:
        input_pdf = Path(self.input_pdf).resolve()
        output_directory = Path(self.output_directory).resolve()
        if not input_pdf.is_file():
            raise ValueError(f"input PDF does not exist: {input_pdf}")
        if input_pdf.suffix.lower() != ".pdf":
            raise ValueError("input file must have a .pdf extension")
        if self.method1_worker_count is not None and (
            isinstance(self.method1_worker_count, bool)
            or not isinstance(self.method1_worker_count, int)
            or self.method1_worker_count < 1
        ):
            raise ValueError("method1_worker_count must be a positive integer")
        object.__setattr__(self, "input_pdf", input_pdf)
        object.__setattr__(self, "output_directory", output_directory)
        object.__setattr__(self, "outputs", normalize_outputs(self.outputs))


@dataclass(frozen=True, slots=True)
class DocumentRunResult:
    status: str
    output_directory: Path
    source_sha256: str
    page_count: int
    requested_pages: tuple[int, ...]
    completed_pages: tuple[int, ...]
    resumed_pages: tuple[int, ...]
    failures: Mapping[int, str]

    @property
    def exit_code(self) -> int:
        return 0 if self.status == "complete" else 1

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "output_directory": str(self.output_directory),
            "source_sha256": self.source_sha256,
            "page_count": self.page_count,
            "requested_pages": list(self.requested_pages),
            "completed_pages": list(self.completed_pages),
            "resumed_pages": list(self.resumed_pages),
            "failures": {
                str(page_number): message
                for page_number, message in sorted(self.failures.items())
            },
        }


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            json.dump(
                payload,
                temporary,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            temporary.write("\n")
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


@contextmanager
def _run_lock(output_directory: Path) -> Iterator[None]:
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / "run.lock"
    token = secrets.token_hex(16)
    payload = json.dumps(
        {"pid": os.getpid(), "token": token, "created_at": _utc_now()},
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            0o600,
        )
    except FileExistsError as error:
        raise RuntimeError(
            f"output directory is already locked: {lock_path}; confirm no run is "
            "active before removing a stale lock"
        ) from error
    initialized = False
    try:
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("run lock metadata was only partially written")
            os.fsync(descriptor)
            initialized = True
        finally:
            os.close(descriptor)
        yield
    finally:
        if not initialized:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
        else:
            try:
                current = json.loads(lock_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                current = None
            if isinstance(current, Mapping) and current.get("token") == token:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass


def _page_path(output_directory: Path, page_number: int) -> Path:
    return output_directory / "pages" / f"page-{page_number:06d}.json"


def _load_json(path: Path) -> Mapping[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, Mapping) else None


def _is_strict_json(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_strict_json(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _is_strict_json(item)
            for key, item in value.items()
        )
    return False


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_result_for_page(
    raw: Mapping[str, object],
    *,
    operation_count: int,
    group_count: int,
    source_specific: bool,
) -> LineTypeRecognitionResult:
    result = LineTypeRecognitionResult.from_dict(raw)
    if len(result.groups) != group_count:
        raise ValueError("result Group count does not match the page grouping")
    if result.summary.input_group_count != group_count:
        raise ValueError("result input Group count does not match the page grouping")
    if result.summary.processed_group_count > group_count:
        raise ValueError("result processed Group count exceeds the page grouping")
    all_indices = {
        op_index
        for group in result.groups
        for op_index in (
            *group.non_linetype.op_indices,
            *(index for line_type in group.line_types for index in line_type.op_indices),
        )
    }
    all_indices.update(
        op_index
        for global_type in result.global_types
        for op_index in global_type.op_indices
    )
    if any(op_index >= operation_count for op_index in all_indices):
        raise ValueError("result references an operation outside the page")
    known_members = {
        (group.group_id, line_type.type_id)
        for group in result.groups
        for line_type in group.line_types
    }
    if any(
        (member.case_id, member.type_id) not in known_members
        for global_type in result.global_types
        for member in global_type.members
    ):
        raise ValueError("global result member does not reference a local line type")
    if source_specific and any(
        global_type.recognition_source is not None
        or global_type.source_global_type_id is not None
        or global_type.type_uid is not None
        for global_type in result.global_types
    ):
        raise ValueError("source-specific result contains fused identity fields")
    return result


def _validate_method1_audit(audit: Mapping[str, object], group_count: int) -> None:
    if audit.get("engine_version") != PYTHON_METHOD1_ENGINE_VERSION:
        raise ValueError("Method1 audit engine version does not match")
    base = audit.get("base")
    stages = audit.get("stages")
    elapsed_ms = audit.get("elapsed_ms")
    if not isinstance(base, Mapping) or not isinstance(stages, list):
        raise ValueError("Method1 audit shape is invalid")
    if (
        isinstance(elapsed_ms, bool)
        or not isinstance(elapsed_ms, (int, float))
        or not math.isfinite(elapsed_ms)
        or elapsed_ms < 0
    ):
        raise ValueError("Method1 audit elapsed time is invalid")
    expected_base = {
        "input_group_count": group_count,
        "processed_group_count": group_count,
        "failed_group_count": 0,
    }
    if any(
        not _nonnegative_integer(base.get(key)) or base.get(key) != expected
        for key, expected in expected_base.items()
    ):
        raise ValueError("Method1 base audit does not cover every page Group")
    if not _nonnegative_integer(base.get("serialized_atom_count")):
        raise ValueError("Method1 serialized atom count is invalid")
    worker_count = base.get("worker_count")
    if not _nonnegative_integer(worker_count) or worker_count == 0:
        raise ValueError("Method1 worker count is invalid")
    from .method1.pipeline import METHOD1_POSTPROCESS_STAGE_NAMES

    if tuple(
        stage.get("stage") if isinstance(stage, Mapping) else None
        for stage in stages
    ) != METHOD1_POSTPROCESS_STAGE_NAMES:
        raise ValueError("Method1 postprocess stage audit is incomplete")
    count_fields = (
        "local_type_count_before",
        "local_type_count_after",
        "global_type_count_before",
        "global_type_count_after",
        "owned_operation_count_before",
        "owned_operation_count_after",
    )
    for stage in stages:
        assert isinstance(stage, Mapping)
        stage_elapsed = stage.get("elapsed_ms")
        if (
            isinstance(stage_elapsed, bool)
            or not isinstance(stage_elapsed, (int, float))
            or not math.isfinite(stage_elapsed)
            or stage_elapsed < 0
        ):
            raise ValueError("Method1 stage audit elapsed_ms is invalid")
        if any(not _nonnegative_integer(stage.get(key)) for key in count_fields):
            raise ValueError("Method1 stage audit count is invalid")


def _validate_resumable_page(
    value: Mapping[str, object],
    *,
    source_sha256: str,
    page_number: int,
    outputs: tuple[OutputKind, ...],
    page_ir_producer_version: str,
) -> bool:
    try:
        if not _is_strict_json(value):
            return False
        if not _is_lower_hex(value.get("payload_sha256"), 64):
            return False
        if value.get("payload_sha256") != page_recognition_payload_hash(value):
            return False
        if value.get("status") != "candidate":
            return False
        if value.get("page_analysis_schema_version") != PAGE_ANALYSIS_SCHEMA_VERSION:
            return False
        if value.get("python_engine_version") != PYTHON_ENGINE_VERSION:
            return False
        if value.get("page_ir_version") != PAGE_IR_VERSION:
            return False
        if value.get("grouping_version") != GROUPING_IR_VERSION:
            return False
        if value.get("result_schema_version") != RESULT_SCHEMA_VERSION:
            return False
        if value.get("page_ir_producer") != SOURCE_ALIGNED_PAGE_IR_PRODUCER:
            return False
        if value.get("page_ir_producer_version") != page_ir_producer_version:
            return False
        if value.get("source_sha256") != source_sha256:
            return False
        if value.get("page_number") != page_number:
            return False
        if value.get("page_identity") != f"{source_sha256}:page:{page_number}":
            return False
        if value.get("requested_outputs") != list(outputs):
            return False
        versions = value.get("algorithm_versions")
        if not isinstance(versions, Mapping) or dict(versions) != {
            "method1": PYTHON_METHOD1_ENGINE_VERSION,
            "method2": PYTHON_METHOD2_ENGINE_VERSION,
            "fusion": PYTHON_FUSION_ENGINE_VERSION,
        }:
            return False
        target_specs = value.get("target_spec_versions")
        if not isinstance(target_specs, Mapping) or dict(target_specs) != {
            "method1": FROZEN_TS_METHOD1_ENGINE_VERSION,
            "method2": FROZEN_TS_METHOD2_ENGINE_VERSION,
            "fusion_policy": FROZEN_TS_FUSION_POLICY_VERSION,
        }:
            return False
        if not all(_is_lower_hex(value.get(key), 64) for key in (
            "page_fingerprint", "grouping_fingerprint"
        )):
            return False
        operation_count = value.get("operation_count")
        group_count = value.get("group_count")
        if not _nonnegative_integer(operation_count) or not _nonnegative_integer(group_count):
            return False
        assert isinstance(operation_count, int) and isinstance(group_count, int)
        alignment_audit = value.get("source_alignment_audit")
        if not isinstance(alignment_audit, Mapping):
            return False
        SourceAlignmentSummary.from_dict(
            alignment_audit,
            expected_parser_version=page_ir_producer_version,
            expected_operation_count=operation_count,
        )

        needs_method1 = "method1" in outputs or "fused" in outputs
        needs_method2 = "method2" in outputs or "fused" in outputs
        needs_input = "input" in outputs
        expected_sections = {
            "method1": needs_method1,
            "method2": needs_method2,
            "fused": "fused" in outputs,
        }
        if any((key in value) != expected for key, expected in expected_sections.items()):
            return False
        method1_result: LineTypeRecognitionResult | None = None
        method2_result: LineTypeRecognitionResult | None = None
        if needs_method1:
            method1 = value.get("method1")
            if not isinstance(method1, Mapping):
                return False
            result = method1.get("result")
            audit = method1.get("audit")
            if not isinstance(result, Mapping) or not isinstance(audit, Mapping):
                return False
            method1_result = _validate_result_for_page(
                result,
                operation_count=operation_count,
                group_count=group_count,
                source_specific=True,
            )
            _validate_method1_audit(audit, group_count)
        if needs_method1 or needs_input:
            from .method1.serializer import METHOD1_SERIALIZED_INPUT_HASH_SCHEMA

            if (
                value.get("method1_input_hash_schema")
                != METHOD1_SERIALIZED_INPUT_HASH_SCHEMA
                or not _is_lower_hex(value.get("method1_input_hash"), 64)
            ):
                return False
        elif (
            "method1_input_hash" in value
            or "method1_input_hash_schema" in value
        ):
            return False
        if needs_method2:
            from .fusion_contract import fuse_line_type_results_for_display
            from .method2.contract import validate_line_type_method2_envelope

            method2 = value.get("method2")
            if not isinstance(method2, Mapping):
                return False
            envelope = validate_line_type_method2_envelope(method2)
            if envelope.page_identity != f"{source_sha256}:page:{page_number}":
                return False
            if not _is_lower_hex(envelope.audit.input_hash, 16):
                return False
            method2_result = _validate_result_for_page(
                envelope.result.to_dict(),
                operation_count=operation_count,
                group_count=group_count,
                source_specific=True,
            )
            validation_ticks = iter((0.0, 0.0))
            fuse_line_type_results_for_display(
                envelope,
                audit_level="full",
                clock=lambda: next(validation_ticks),
            )
            family_audit = envelope.audit.repeated_vector_text_family_clustering
            if any(
                op_index >= operation_count
                for op_index in (
                    *family_audit.matched_text_op_indices,
                    *family_audit.line_type_confirmed_text_op_indices,
                )
            ):
                return False
        if "fused" in outputs:
            from .fusion_contract import (
                fuse_line_type_results_for_display,
                validate_fused_line_type_envelope,
            )
            from .method2.contract import validate_line_type_method2_envelope

            fused = value.get("fused")
            if not isinstance(fused, Mapping):
                return False
            assert method1_result is not None and method2_result is not None
            fused_envelope = validate_fused_line_type_envelope(fused)
            if fused_envelope.page_identity != f"{source_sha256}:page:{page_number}":
                return False
            assert needs_method2
            method2 = value.get("method2")
            assert isinstance(method2, Mapping)
            method2_envelope = validate_line_type_method2_envelope(method2)
            fusion_elapsed_ms = (
                fused_envelope.audit.elapsed_ms - method2_envelope.audit.elapsed_ms
            )
            if fusion_elapsed_ms < 0:
                return False
            ticks = iter((0.0, fusion_elapsed_ms / 1000.0))
            expected_fusion = fuse_line_type_results_for_display(
                method2_envelope,
                method1_result,
                fused_envelope.audit.level,
                clock=lambda: next(ticks),
            )
            if dict(fused) != expected_fusion.to_dict():
                return False
        return True
    except (AssertionError, TypeError, ValueError):
        return False


def _validate_output_owner(output_directory: Path, source_sha256: str) -> None:
    """Refuse an output tree whose source ownership cannot be established."""

    manifest_path = output_directory / "run.json"
    if not manifest_path.exists():
        pages_path = output_directory / "pages"
        if pages_path.exists() and (
            not pages_path.is_dir() or any(pages_path.iterdir())
        ):
            raise RuntimeError(
                "output directory contains page results but has no owning run.json"
            )
        return
    manifest = _load_json(manifest_path)
    if manifest is None:
        raise RuntimeError("existing run.json is malformed or unreadable")
    owner = manifest.get("source_sha256")
    if not _is_lower_hex(owner, 64):
        raise RuntimeError("existing run.json has no valid PDF source identity")
    if owner != source_sha256:
        raise RuntimeError("output directory belongs to a different PDF source hash")


def _manifest(
    *,
    status: str,
    options: DocumentRunOptions,
    source_sha256: str,
    page_count: int,
    requested_pages: tuple[int, ...],
    completed_pages: Sequence[int],
    resumed_pages: Sequence[int],
    failures: Mapping[int, str],
    started_at: str,
    page_ir_producer_version: str,
) -> dict[str, object]:
    return {
        "schema_version": DOCUMENT_RUN_SCHEMA_VERSION,
        "status": status,
        "python_engine_version": PYTHON_ENGINE_VERSION,
        "page_analysis_schema_version": PAGE_ANALYSIS_SCHEMA_VERSION,
        "page_ir_version": PAGE_IR_VERSION,
        "grouping_version": GROUPING_IR_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "pdf_adapter_version": PDF_ADAPTER_VERSION,
        "source_content_version": SOURCE_CONTENT_VERSION,
        "source_page_adapter_version": SOURCE_PAGE_ADAPTER_VERSION,
        "page_ir_producer": SOURCE_ALIGNED_PAGE_IR_PRODUCER,
        "page_ir_producer_version": page_ir_producer_version,
        "source_path": str(options.input_pdf),
        "source_sha256": source_sha256,
        "algorithm_versions": {
            "method1": PYTHON_METHOD1_ENGINE_VERSION,
            "method2": PYTHON_METHOD2_ENGINE_VERSION,
            "fusion": PYTHON_FUSION_ENGINE_VERSION,
        },
        "target_spec_versions": {
            "method1": FROZEN_TS_METHOD1_ENGINE_VERSION,
            "method2": FROZEN_TS_METHOD2_ENGINE_VERSION,
            "fusion_policy": FROZEN_TS_FUSION_POLICY_VERSION,
        },
        "page_count": page_count,
        "requested_pages": list(requested_pages),
        "requested_outputs": list(options.outputs),
        "method1_worker_count": options.method1_worker_count,
        "completed_pages": sorted(completed_pages),
        "resumed_pages": sorted(resumed_pages),
        "failures": {
            str(page_number): message
            for page_number, message in sorted(failures.items())
        },
        "started_at": started_at,
        "updated_at": _utc_now(),
    }


def _run_open_document(
    options: DocumentRunOptions,
    *,
    source_sha256: str,
    document: SourceAlignedPdfDocument,
    analyzer: PageAnalyzer,
) -> DocumentRunResult:
    if document.source_sha256 != source_sha256:
        raise RuntimeError("open parser session does not match the immutable source hash")
    page_count = int(document.page_count)
    pages = parse_page_selection(options.pages, page_count)
    page_ir_producer_version = _current_page_ir_producer_version()
    output_directory = options.output_directory
    started_at = _utc_now()
    completed: list[int] = []
    resumed: list[int] = []
    failures: dict[int, str] = {}

    with _run_lock(output_directory):
        _validate_output_owner(output_directory, source_sha256)
        _atomic_write_json(
            output_directory / "run.json",
            _manifest(
                status="running",
                options=options,
                source_sha256=source_sha256,
                page_count=page_count,
                requested_pages=pages,
                completed_pages=completed,
                resumed_pages=resumed,
                failures=failures,
                started_at=started_at,
                page_ir_producer_version=page_ir_producer_version,
            ),
        )
        try:
            for page_number in pages:
                # Drop the preceding page before extracting the next one.  In
                # particular, never retain a large rejected cache alongside a
                # fresh PageIR and recognition result.
                cached = None
                aligned = None
                page = None
                recognition = None
                payload = None
                page_path = _page_path(output_directory, page_number)
                if options.resume:
                    cached = _load_json(page_path)
                    if cached is not None and _validate_resumable_page(
                        cached,
                        source_sha256=source_sha256,
                        page_number=page_number,
                        outputs=options.outputs,
                        page_ir_producer_version=page_ir_producer_version,
                    ):
                        completed.append(page_number)
                        resumed.append(page_number)
                        continue
                    cached = None
                try:
                    aligned = document.page(page_number)
                    page = aligned.page
                    recognition = analyzer(
                        page,
                        outputs=options.outputs,
                        page_identity=f"{source_sha256}:page:{page_number}",
                        method1_worker_count=options.method1_worker_count,
                    )
                    recognition = replace(
                        recognition,
                        source_alignment_audit=(
                            SourceAlignmentSummary.from_alignment_audit(
                                aligned.audit,
                                parser_version=page_ir_producer_version,
                            )
                        ),
                    )
                    payload = recognition.to_dict()
                    if not _validate_resumable_page(
                        payload,
                        source_sha256=source_sha256,
                        page_number=page_number,
                        outputs=options.outputs,
                        page_ir_producer_version=page_ir_producer_version,
                    ):
                        raise RuntimeError("page analyzer returned an invalid durable payload")
                    _atomic_write_json(page_path, payload)
                    completed.append(page_number)
                except Exception as error:  # Each page is an explicit isolation boundary.
                    failures[page_number] = f"{type(error).__name__}: {error}"
                finally:
                    cached = None
                    aligned = None
                    page = None
                    recognition = None
                    payload = None
                _atomic_write_json(
                    output_directory / "run.json",
                    _manifest(
                        status="running",
                        options=options,
                        source_sha256=source_sha256,
                        page_count=page_count,
                        requested_pages=pages,
                        completed_pages=completed,
                        resumed_pages=resumed,
                        failures=failures,
                        started_at=started_at,
                        page_ir_producer_version=page_ir_producer_version,
                    ),
                )
        except KeyboardInterrupt:
            _atomic_write_json(
                output_directory / "run.json",
                _manifest(
                    status="aborted",
                    options=options,
                    source_sha256=source_sha256,
                    page_count=page_count,
                    requested_pages=pages,
                    completed_pages=completed,
                    resumed_pages=resumed,
                    failures=failures,
                    started_at=started_at,
                    page_ir_producer_version=page_ir_producer_version,
                ),
            )
            raise

        status = "complete" if len(completed) == len(pages) else "partial"
        final_manifest = _manifest(
            status=status,
            options=options,
            source_sha256=source_sha256,
            page_count=page_count,
            requested_pages=pages,
            completed_pages=completed,
            resumed_pages=resumed,
            failures=failures,
            started_at=started_at,
            page_ir_producer_version=page_ir_producer_version,
        )
        _atomic_write_json(output_directory / "run.json", final_manifest)

    return DocumentRunResult(
        status=status,
        output_directory=output_directory,
        source_sha256=source_sha256,
        page_count=page_count,
        requested_pages=pages,
        completed_pages=tuple(sorted(completed)),
        resumed_pages=tuple(sorted(resumed)),
        failures=dict(sorted(failures.items())),
    )


def run_document(
    options: DocumentRunOptions,
    *,
    analyzer: PageAnalyzer = recognize_page,
) -> DocumentRunResult:
    """Recognize selected pages from one immutable, once-opened PDF snapshot.

    A page failure is recorded and later pages still run; the returned status
    is ``partial`` until a resume fills every selected page.
    """

    assert_supported_pymupdf_runtime()
    source = _read_source_snapshot(options.input_pdf)
    source_sha256 = sha256(source).hexdigest()
    with _open_pdf_snapshot(source, source_name=options.input_pdf.name) as document:
        return _run_open_document(
            options,
            source_sha256=source_sha256,
            document=document,
            analyzer=analyzer,
        )


__all__ = [
    "DocumentRunOptions",
    "DocumentRunResult",
    "MAX_DOCUMENT_PAGES",
    "parse_page_selection",
    "run_document",
]
