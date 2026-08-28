"""Safe, resumable orchestration for the complete line-type PDF corpus.

This module coordinates the existing per-document CLI; it contains no line
recognition decisions.  Every run owns an immutable PDF snapshot tree and a
dedicated result tree, so it can never overwrite the viewer's frozen caches.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import subprocess
import sys
import tempfile
from typing import Callable, Iterator, Mapping, Sequence

from .document_runner import DOCUMENT_RUN_SCHEMA_VERSION, parse_page_selection
from .engine import page_recognition_payload_hash
from .fusion_contract import fuse_line_type_results_for_display
from .method2.contract import validate_line_type_method2_envelope
from .results import LineTypeRecognitionResult
from .runtime import describe_runtime_versions
from .versions import (
    FROZEN_TS_FUSION_POLICY_VERSION,
    FROZEN_TS_METHOD1_ENGINE_VERSION,
    FROZEN_TS_METHOD2_ENGINE_VERSION,
    GROUPING_IR_VERSION,
    PAGE_ANALYSIS_SCHEMA_VERSION,
    PAGE_IR_VERSION,
    PYTHON_ENGINE_VERSION,
    PYTHON_FUSION_ENGINE_VERSION,
    PYTHON_METHOD1_ENGINE_VERSION,
    PYTHON_METHOD2_ENGINE_VERSION,
    RESULT_SCHEMA_VERSION,
)


CORPUS_RUN_SCHEMA_VERSION = 1
COMPOSED_PAGE_SCHEMA_VERSION = 1
DEFAULT_SOURCE_ROOT = Path(
    os.environ.get(
        "LINE_TYPE_CORPUS_SOURCE_ROOT",
        str(Path.home() / "OneDrive" / "Desktop" / "fence_detector_projects_PDF"),
    )
)
STAGE_ORDER = ("input", "method1", "method2", "compose", "fused")
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, object]) -> None:
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
            temporary.write(_canonical_json(value))
            temporary.write(b"\n")
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


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(value)
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


def _load_object(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    document_id: str
    label: str
    relative_path: Path
    source_path: Path
    page_count: int
    selected_pages: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "label": self.label,
            "relative_path": self.relative_path.as_posix(),
            "source_path": str(self.source_path),
            "page_count": self.page_count,
            "selected_pages": list(self.selected_pages),
        }


@dataclass(frozen=True, slots=True)
class CorpusShard:
    stage: str
    document_id: str
    page_start: int
    page_end: int
    pages: tuple[int, ...]
    output_directory: Path

    @property
    def shard_id(self) -> str:
        return (
            f"{self.stage}:{self.document_id}:"
            f"{self.page_start:06d}-{self.page_end:06d}"
        )

    @property
    def pages_argument(self) -> str:
        ranges: list[str] = []
        start = previous = self.pages[0]
        for page in self.pages[1:]:
            if page == previous + 1:
                previous = page
                continue
            ranges.append(str(start) if start == previous else f"{start}-{previous}")
            start = previous = page
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        return ",".join(ranges)

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "stage": self.stage,
            "document_id": self.document_id,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "pages": list(self.pages),
            "output_directory": str(self.output_directory),
        }


@dataclass(frozen=True, slots=True)
class CorpusRunOptions:
    manifest_path: Path
    source_root: Path
    run_root: Path
    repository_root: Path
    cli_script: Path
    documents: tuple[str, ...] = ()
    pages: str = "all"
    stages: tuple[str, ...] = ("input", "method1", "method2", "compose")
    shard_pages: int = 32
    cpu_budget: int = max(1, os.cpu_count() or 1)
    input_jobs: int | None = None
    method1_jobs: int | None = None
    method1_workers: int | None = None
    method2_jobs: int | None = None
    fused_jobs: int | None = None
    fused_workers: int | None = None
    max_jobs_per_document: int = 2
    timeout_seconds: float | None = None
    resume: bool = True
    dry_run: bool = False
    python_executable: str = sys.executable

    def __post_init__(self) -> None:
        manifest_path = Path(self.manifest_path).resolve()
        source_root = Path(self.source_root).resolve()
        run_root = Path(self.run_root).resolve()
        repository_root = Path(self.repository_root).resolve()
        cli_script = Path(self.cli_script).resolve()
        if not manifest_path.is_file():
            raise ValueError(f"corpus manifest does not exist: {manifest_path}")
        if not source_root.is_dir():
            raise ValueError(f"corpus source root does not exist: {source_root}")
        if not cli_script.is_file():
            raise ValueError(f"recognition CLI does not exist: {cli_script}")
        if self.shard_pages < 1:
            raise ValueError("shard_pages must be positive")
        if self.cpu_budget < 1:
            raise ValueError("cpu_budget must be positive")
        input_jobs = self.input_jobs or min(12, self.cpu_budget)
        method1_workers = self.method1_workers or min(4, self.cpu_budget)
        method1_jobs = self.method1_jobs or min(
            6, max(1, self.cpu_budget // method1_workers)
        )
        method2_jobs = self.method2_jobs or min(12, self.cpu_budget)
        # A fused child owns this complete budget.  It reserves capacity for
        # Method2 and gives the balance to Method1, while multiple shards fill
        # the machine at corpus scale.  Four workers per page is deliberately
        # the default balance: enough for heavy-page inner pools without
        # sacrificing page-level parallelism or multiplying PageIR memory by
        # one process per logical CPU.
        for name, value in (
            ("fused_jobs", self.fused_jobs),
            ("fused_workers", self.fused_workers),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise ValueError(f"{name} must be a positive integer")
        fused_workers = (
            min(4, self.cpu_budget)
            if self.fused_workers is None
            else self.fused_workers
        )
        fused_jobs = (
            max(1, self.cpu_budget // fused_workers)
            if self.fused_jobs is None and fused_workers > 0
            else self.fused_jobs
        )
        # A nonpositive explicit worker value must reach the common validation
        # below without becoming a division-by-zero accident.
        if fused_jobs is None:
            fused_jobs = 0
        if min(
            input_jobs,
            method1_workers,
            method1_jobs,
            method2_jobs,
            fused_jobs,
            fused_workers,
        ) < 1:
            raise ValueError("stage jobs and workers must be positive")
        if input_jobs > self.cpu_budget:
            raise ValueError("input_jobs exceeds cpu_budget")
        if method1_jobs * method1_workers > self.cpu_budget:
            raise ValueError(
                "method1_jobs * method1_workers exceeds cpu_budget"
            )
        if method2_jobs > self.cpu_budget:
            raise ValueError("method2_jobs exceeds cpu_budget")
        if fused_jobs * fused_workers > self.cpu_budget:
            raise ValueError("fused_jobs * fused_workers exceeds cpu_budget")
        normalized_stages = tuple(
            stage for stage in STAGE_ORDER if stage in set(self.stages)
        )
        if not normalized_stages or set(self.stages) - set(STAGE_ORDER):
            raise ValueError(
                "stages must contain fused, or input/method1/method2/compose"
            )
        if "fused" in normalized_stages and normalized_stages != ("fused",):
            raise ValueError(
                "fused is a single-pass alternative and cannot be mixed with "
                "staged corpus outputs"
            )
        if {"method1", "method2", "compose"}.intersection(normalized_stages) and (
            "input" not in normalized_stages
        ):
            raise ValueError("algorithm stages require the input preflight stage")
        if "compose" in normalized_stages and not {
            "method1", "method2"
        }.issubset(normalized_stages):
            raise ValueError("compose requires method1 and method2 stages")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.max_jobs_per_document < 1:
            raise ValueError("max_jobs_per_document must be positive")
        protected = (
            repository_root / "data",
            repository_root / "public" / "group-samples",
            source_root,
        )
        if any(
            _is_relative_to(run_root, root.resolve())
            or _is_relative_to(root.resolve(), run_root)
            for root in protected
        ):
            raise ValueError("run_root must be outside source and production cache trees")
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "source_root", source_root)
        object.__setattr__(self, "run_root", run_root)
        object.__setattr__(self, "repository_root", repository_root)
        object.__setattr__(self, "cli_script", cli_script)
        object.__setattr__(self, "stages", normalized_stages)
        object.__setattr__(self, "input_jobs", input_jobs)
        object.__setattr__(self, "method1_workers", method1_workers)
        object.__setattr__(self, "method1_jobs", method1_jobs)
        object.__setattr__(self, "method2_jobs", method2_jobs)
        object.__setattr__(self, "fused_jobs", fused_jobs)
        object.__setattr__(self, "fused_workers", fused_workers)


@dataclass(frozen=True, slots=True)
class CorpusPlan:
    documents: tuple[CorpusDocument, ...]
    shards: tuple[CorpusShard, ...]
    manifest_sha256: str
    fingerprint: str
    implementation: Mapping[str, object]

    @property
    def page_count(self) -> int:
        return sum(len(document.selected_pages) for document in self.documents)

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "plan_fingerprint": self.fingerprint,
            "implementation": dict(self.implementation),
            "document_count": len(self.documents),
            "page_count": self.page_count,
            "documents": [document.to_dict() for document in self.documents],
            "shards": [shard.to_dict() for shard in self.shards],
        }


def _implementation_identity(options: CorpusRunOptions) -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    files = sorted(
        (
            path
            for path in package_root.rglob("*.py")
            if "__pycache__" not in path.parts
        ),
        key=lambda path: path.as_posix(),
    )
    if options.cli_script not in files:
        files.append(options.cli_script)
    digest = sha256()
    base = package_root.parent
    for path in files:
        try:
            name = path.relative_to(base).as_posix()
        except ValueError:
            name = str(path)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    runtime_versions = describe_runtime_versions()
    packages = runtime_versions["packages"]
    assert isinstance(packages, Mapping)
    dependency_versions = {
        "PyMuPDF": packages["pymupdf"],
        "pypdf": packages["pypdf"],
        "numpy": packages["numpy"],
        "scipy": packages["scipy"],
    }
    return {
        "python_engine_version": PYTHON_ENGINE_VERSION,
        "page_analysis_schema_version": PAGE_ANALYSIS_SCHEMA_VERSION,
        "page_ir_version": PAGE_IR_VERSION,
        "grouping_version": GROUPING_IR_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
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
        "python_source_fingerprint": digest.hexdigest(),
        "python_source_file_count": len(files),
        "runtime": {
            "executable": str(Path(sys.executable).resolve()),
            "requested_child_executable": str(
                Path(options.python_executable).resolve()
            ),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "cache_tag": getattr(sys.implementation, "cache_tag", None),
            "platform": platform.platform(),
            "dependencies": dependency_versions,
            "requirements": runtime_versions["requirements"],
            "pymupdf_runtime": runtime_versions["pymupdf_runtime"],
            "pypdf_runtime": runtime_versions["pypdf_runtime"],
        },
    }


@dataclass(frozen=True, slots=True)
class ShardOutcome:
    shard_id: str
    status: str
    return_code: int
    command: tuple[str, ...]
    stdout_tail: str
    stderr_tail: str
    started_at: str
    finished_at: str
    stdout_log: str = ""
    stderr_log: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "shard_id": self.shard_id,
            "status": self.status,
            "return_code": self.return_code,
            "command": list(self.command),
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
        }


def load_corpus_documents(options: CorpusRunOptions) -> tuple[CorpusDocument, ...]:
    raw = _load_object(options.manifest_path)
    if raw is None or not isinstance(raw.get("documents"), list):
        raise ValueError("corpus manifest is malformed")
    selected_ids = set(options.documents)
    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    for item in raw["documents"]:
        if not isinstance(item, Mapping) or item.get("valid") is not True:
            continue
        document_id = item.get("id")
        relative_value = item.get("relativePath")
        page_count = item.get("pageCount")
        if (
            not isinstance(document_id, str)
            or not _SAFE_ID.fullmatch(document_id)
            or document_id in seen
            or not isinstance(relative_value, str)
            or isinstance(page_count, bool)
            or not isinstance(page_count, int)
            or page_count < 1
        ):
            raise ValueError("valid corpus document metadata is malformed")
        seen.add(document_id)
        if selected_ids and document_id not in selected_ids:
            continue
        relative_path = Path(relative_value)
        source_path = (options.source_root / relative_path).resolve()
        if not _is_relative_to(source_path, options.source_root):
            raise ValueError(f"document {document_id} escapes source_root")
        if not source_path.is_file():
            raise ValueError(f"document source is missing: {source_path}")
        pages = parse_page_selection(options.pages, page_count)
        documents.append(CorpusDocument(
            document_id=document_id,
            label=str(item.get("label") or document_id),
            relative_path=relative_path,
            source_path=source_path,
            page_count=page_count,
            selected_pages=pages,
        ))
    missing = selected_ids - {document.document_id for document in documents}
    if missing:
        raise ValueError(f"unknown or invalid corpus document(s): {', '.join(sorted(missing))}")
    if not documents:
        raise ValueError("corpus selection contains no valid documents")
    return tuple(documents)


def _page_shards(pages: tuple[int, ...], maximum: int) -> tuple[tuple[int, ...], ...]:
    shards: list[tuple[int, ...]] = []
    current: list[int] = []
    for page in pages:
        if current and (len(current) >= maximum or page != current[-1] + 1):
            shards.append(tuple(current))
            current = []
        current.append(page)
    if current:
        shards.append(tuple(current))
    return tuple(shards)


def build_corpus_plan(options: CorpusRunOptions) -> CorpusPlan:
    documents = load_corpus_documents(options)
    implementation = _implementation_identity(options)
    shards: list[CorpusShard] = []
    for stage in (stage for stage in options.stages if stage != "compose"):
        for document in documents:
            for pages in _page_shards(document.selected_pages, options.shard_pages):
                output = (
                    options.run_root / "stages" / stage / document.document_id /
                    f"pages-{pages[0]:06d}-{pages[-1]:06d}"
                )
                shards.append(CorpusShard(
                    stage, document.document_id, pages[0], pages[-1], pages, output
                ))
    manifest_bytes = options.manifest_path.read_bytes()
    manifest_hash = sha256(manifest_bytes).hexdigest()
    identity = {
        "schema_version": CORPUS_RUN_SCHEMA_VERSION,
        "manifest_sha256": manifest_hash,
        "source_root": str(options.source_root),
        "documents": [
            {
                "id": document.document_id,
                "relative_path": document.relative_path.as_posix(),
                "page_count": document.page_count,
                "selected_pages": list(document.selected_pages),
            }
            for document in documents
        ],
        "stages": list(options.stages),
        "shard_pages": options.shard_pages,
        "method1_workers": options.method1_workers,
        "fused_workers": options.fused_workers,
        "implementation": implementation,
    }
    fingerprint = sha256(_canonical_json(identity)).hexdigest()
    return CorpusPlan(
        documents, tuple(shards), manifest_hash, fingerprint, implementation
    )


def _hash_file(path: Path) -> tuple[int, str]:
    digest = sha256()
    length = 0
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            length += len(chunk)
            digest.update(chunk)
    return length, digest.hexdigest()


def _snapshot_document(
    document: CorpusDocument,
    run_root: Path,
    existing: Mapping[str, object] | None,
) -> dict[str, object]:
    destination = run_root / "sources" / f"{document.document_id}.pdf"
    if destination.exists():
        if not isinstance(existing, Mapping):
            raise RuntimeError(f"unowned source snapshot already exists: {destination}")
        recorded_path = existing.get("path")
        if (
            existing.get("document_id") != document.document_id
            or not isinstance(recorded_path, str)
            or Path(recorded_path).resolve() != destination.resolve()
        ):
            raise RuntimeError(
                f"source snapshot ownership record is invalid: {destination}"
            )
        length, digest = _hash_file(destination)
        if existing.get("byte_length") != length or existing.get("sha256") != digest:
            raise RuntimeError(f"source snapshot integrity failed: {destination}")
        return dict(existing)
    if existing is not None:
        raise RuntimeError(f"recorded source snapshot is missing: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    digest = sha256()
    length = 0
    try:
        with document.source_path.open("rb") as source, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            before = os.fstat(source.fileno())
            while chunk := source.read(8 * 1024 * 1024):
                temporary.write(chunk)
                digest.update(chunk)
                length += len(chunk)
            after = os.fstat(source.fileno())
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
            if identity_before != identity_after or length != before.st_size:
                raise RuntimeError(
                    f"source PDF changed while snapshotting: {document.source_path}"
                )
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return {
        "document_id": document.document_id,
        "path": str(destination),
        "byte_length": length,
        "sha256": digest.hexdigest(),
        "source_path": str(document.source_path),
        "snapshotted_at": _utc_now(),
    }


@contextmanager
def _corpus_lock(run_root: Path) -> Iterator[None]:
    lock = run_root / "corpus-run.lock"
    token = secrets.token_hex(16)
    payload = _canonical_json({
        "pid": os.getpid(),
        "host": platform.node(),
        "token": token,
        "created_at": _utc_now(),
    })
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"corpus run is already locked: {lock}; confirm the recorded process "
            "is no longer active before removing a stale lock"
        ) from error
    initialized = False
    try:
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError("corpus lock metadata was only partially written")
            os.fsync(descriptor)
            initialized = True
        finally:
            os.close(descriptor)
        yield
    finally:
        if not initialized:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
        else:
            current = _load_object(lock)
            if current is not None and current.get("token") == token:
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass


def _base_manifest(
    options: CorpusRunOptions,
    plan: CorpusPlan,
    existing: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": CORPUS_RUN_SCHEMA_VERSION,
        "status": "snapshotting",
        "plan_fingerprint": plan.fingerprint,
        "manifest_path": str(options.manifest_path),
        "manifest_sha256": plan.manifest_sha256,
        "source_root": str(options.source_root),
        "run_root": str(options.run_root),
        "implementation": dict(plan.implementation),
        "plan": plan.to_dict(),
        "configuration": {
            "stages": list(options.stages),
            "shard_pages": options.shard_pages,
            "cpu_budget": options.cpu_budget,
            "input_jobs": options.input_jobs,
            "method1_jobs": options.method1_jobs,
            "method1_workers": options.method1_workers,
            "method2_jobs": options.method2_jobs,
            "fused_jobs": options.fused_jobs,
            "fused_workers": options.fused_workers,
            "max_jobs_per_document": options.max_jobs_per_document,
            "resume": options.resume,
        },
        "document_count": len(plan.documents),
        "page_count": plan.page_count,
        "shard_count": len(plan.shards),
        "sources": dict(existing.get("sources", {})) if existing else {},
        "jobs": dict(existing.get("jobs", {})) if existing else {},
        "compose": dict(existing.get("compose", {})) if existing else {},
        "failures": {},
        "started_at": existing.get("started_at", _utc_now()) if existing else _utc_now(),
        "updated_at": _utc_now(),
    }


def _ensure_owned_root(options: CorpusRunOptions, plan: CorpusPlan) -> dict[str, object] | None:
    manifest_path = options.run_root / "corpus-run.json"
    if not options.run_root.exists():
        return None
    existing = _load_object(manifest_path)
    entries = [item for item in options.run_root.iterdir() if item.name != "corpus-run.lock"]
    if existing is None:
        if entries:
            raise RuntimeError("run_root is non-empty but has no valid corpus-run.json")
        return None
    if existing.get("schema_version") != CORPUS_RUN_SCHEMA_VERSION:
        raise RuntimeError("run_root belongs to another corpus schema")
    if existing.get("plan_fingerprint") != plan.fingerprint:
        raise RuntimeError("run_root belongs to a different corpus plan")
    return existing


def _child_command(
    options: CorpusRunOptions,
    shard: CorpusShard,
    snapshot: Path,
) -> tuple[str, ...]:
    command = [
        options.python_executable,
        str(options.cli_script),
        "recognize",
        str(snapshot),
        "--output-dir",
        str(shard.output_directory),
        "--pages",
        shard.pages_argument,
        "--outputs",
        shard.stage,
    ]
    if shard.stage == "method1":
        command.extend(("--workers", str(options.method1_workers)))
    elif shard.stage == "fused":
        # In a fused run this is the total per-page CPU budget consumed by the
        # engine scheduler, not an extra pool layered on top of Method2.
        command.extend(("--workers", str(options.fused_workers)))
    if not options.resume:
        command.append("--no-resume")
    return tuple(command)


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_corpus_stage_page(
    value: Mapping[str, object],
    *,
    stage: str,
    source_sha256: str,
    page_number: int,
) -> None:
    """Validate the coordinator-visible contract of one durable stage page."""

    if stage not in {"input", "method1", "method2", "fused"}:
        raise ValueError(f"unsupported corpus stage: {stage}")
    if value.get("payload_sha256") != page_recognition_payload_hash(value):
        raise ValueError("page payload checksum failed")
    expected_scalars = {
        "status": "candidate",
        "python_engine_version": PYTHON_ENGINE_VERSION,
        "page_analysis_schema_version": PAGE_ANALYSIS_SCHEMA_VERSION,
        "page_ir_version": PAGE_IR_VERSION,
        "grouping_version": GROUPING_IR_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "page_number": page_number,
        "page_identity": f"{source_sha256}:page:{page_number}",
    }
    for key, expected in expected_scalars.items():
        if value.get(key) != expected:
            raise ValueError(f"page payload {key} does not match its shard")
    if value.get("requested_outputs") != [stage]:
        raise ValueError("page payload requested_outputs does not match its stage")
    expected_versions = {
        "method1": PYTHON_METHOD1_ENGINE_VERSION,
        "method2": PYTHON_METHOD2_ENGINE_VERSION,
        "fusion": PYTHON_FUSION_ENGINE_VERSION,
    }
    if value.get("algorithm_versions") != expected_versions:
        raise ValueError("page payload algorithm versions do not match")
    if not all(
        isinstance(value.get(key), int)
        and not isinstance(value.get(key), bool)
        and int(value[key]) >= 0
        for key in ("operation_count", "group_count")
    ):
        raise ValueError("page payload operation/group counts are invalid")
    if not all(
        _is_lower_hex(value.get(key), 64)
        for key in ("page_fingerprint", "grouping_fingerprint")
    ):
        raise ValueError("page payload fingerprints are invalid")
    if not isinstance(value.get("source_alignment_audit"), Mapping):
        raise ValueError("page payload source-alignment audit is missing")

    section_expectations = {
        "method1": stage in {"method1", "fused"},
        "method2": stage in {"method2", "fused"},
        "fused": stage == "fused",
    }
    if any((key in value) != present for key, present in section_expectations.items()):
        raise ValueError("page payload result sections do not match its stage")
    if stage == "input":
        if not _is_lower_hex(value.get("method1_input_hash"), 64):
            raise ValueError("input preflight has no valid Method1 input hash")
    elif stage != "fused":
        envelope = value.get(stage)
        if not isinstance(envelope, Mapping) or not isinstance(
            envelope.get("result"), Mapping
        ):
            raise ValueError(f"{stage} page result is missing")
    else:
        # The fused artifact is the release candidate for all three outputs.
        # Reuse the document runner's full durable-page verifier so resume and
        # corpus comparison accept exactly the same source-aligned contract,
        # including complete result ownership and deterministic fusion replay.
        from .document_runner import _validate_resumable_page
        from .source_page_adapter import current_source_aligned_producer_version

        if not _validate_resumable_page(
            value,
            source_sha256=source_sha256,
            page_number=page_number,
            outputs=("fused",),
            page_ir_producer_version=current_source_aligned_producer_version(),
        ):
            raise ValueError("fused page failed strict durable-payload validation")


def validate_corpus_shard_artifacts(
    shard: CorpusShard,
    *,
    source_sha256: str,
    document_page_count: int,
    expected_worker_count: int | None = None,
) -> None:
    """Require a successful child to have produced exactly its durable pages."""

    run_manifest = _load_object(shard.output_directory / "run.json")
    if run_manifest is None:
        raise ValueError("shard run.json is missing or malformed")
    expected_manifest = {
        "schema_version": DOCUMENT_RUN_SCHEMA_VERSION,
        "status": "complete",
        "python_engine_version": PYTHON_ENGINE_VERSION,
        "page_analysis_schema_version": PAGE_ANALYSIS_SCHEMA_VERSION,
        "page_ir_version": PAGE_IR_VERSION,
        "grouping_version": GROUPING_IR_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "source_sha256": source_sha256,
        "page_count": document_page_count,
        "requested_pages": list(shard.pages),
        "requested_outputs": [shard.stage],
        "completed_pages": list(shard.pages),
    }
    for key, expected in expected_manifest.items():
        if run_manifest.get(key) != expected:
            raise ValueError(f"shard run.json {key} does not match its plan")
    if (
        expected_worker_count is not None
        and run_manifest.get("method1_worker_count") != expected_worker_count
    ):
        raise ValueError("shard run.json worker budget does not match its plan")
    failures = run_manifest.get("failures")
    if not isinstance(failures, Mapping) or failures:
        raise ValueError("shard run.json contains failures")

    pages_directory = shard.output_directory / "pages"
    if not pages_directory.is_dir():
        raise ValueError("shard pages directory is missing")
    expected_names = {f"page-{page:06d}.json" for page in shard.pages}
    actual_entries = {path.name for path in pages_directory.iterdir()}
    if actual_entries != expected_names:
        missing = sorted(expected_names - actual_entries)
        extra = sorted(actual_entries - expected_names)
        raise ValueError(
            f"shard page set differs from its plan; missing={missing}, extra={extra}"
        )
    for page_number in shard.pages:
        page_path = pages_directory / f"page-{page_number:06d}.json"
        value = _load_object(page_path)
        if value is None:
            raise ValueError(f"page {page_number} artifact is malformed")
        validate_corpus_stage_page(
            value,
            stage=shard.stage,
            source_sha256=source_sha256,
            page_number=page_number,
        )


def run_shard_subprocess(
    options: CorpusRunOptions,
    shard: CorpusShard,
    snapshot: Path,
) -> ShardOutcome:
    command = _child_command(options, shard, snapshot)
    started = _utc_now()
    stdout = ""
    stderr = ""
    return_code = -1
    status = "failed"
    creation_flags = 0
    popen_options: dict[str, object] = {}
    if os.name == "nt":
        creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_options["start_new_session"] = True
    try:
        process = subprocess.Popen(
            command,
            cwd=options.repository_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=creation_flags,
            **popen_options,
        )
        try:
            stdout, stderr = process.communicate(timeout=options.timeout_seconds)
        except subprocess.TimeoutExpired as timeout_error:
            def timeout_text(value: object) -> str:
                if value is None:
                    return ""
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return str(value)

            stdout = timeout_text(timeout_error.stdout) or stdout
            stderr = timeout_text(timeout_error.stderr) or stderr
            tree_kill_note = ""
            if os.name == "nt":
                try:
                    killed = subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                    if killed.returncode != 0:
                        tree_kill_note = (
                            f"taskkill could not confirm tree termination "
                            f"(exit {killed.returncode})"
                        )
                except (OSError, subprocess.TimeoutExpired) as kill_error:
                    tree_kill_note = (
                        "taskkill could not confirm tree termination: "
                        f"{type(kill_error).__name__}: {kill_error}"
                    )
            else:
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except OSError as kill_error:
                    tree_kill_note = (
                        "process-group termination was not confirmed: "
                        f"{type(kill_error).__name__}: {kill_error}"
                    )
            if process.poll() is None:
                try:
                    process.kill()
                except OSError as kill_error:
                    tree_kill_note = "\n".join(filter(None, (
                        tree_kill_note,
                        "direct child termination failed: "
                        f"{type(kill_error).__name__}: {kill_error}",
                    )))
            try:
                timed_stdout, timed_stderr = process.communicate(timeout=5)
                stdout = timed_stdout or stdout
                stderr = timed_stderr or stderr
            except subprocess.TimeoutExpired as drain_error:
                stdout = timeout_text(drain_error.stdout) or stdout
                stderr = timeout_text(drain_error.stderr) or stderr
                tree_kill_note = "\n".join(filter(None, (
                    tree_kill_note,
                    "subprocess pipes did not close within 5 seconds",
                )))
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass
                try:
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            stderr = "\n".join(filter(None, (
                stderr,
                "subprocess timeout",
                tree_kill_note,
            )))
            return_code = -1
            status = "timeout"
        else:
            return_code = int(process.returncode)
            status = "complete" if return_code == 0 else "failed"
    except OSError as error:
        stderr = f"{type(error).__name__}: {error}"
    log_root = options.run_root / "logs" / shard.stage / shard.document_id
    log_stem = f"pages-{shard.page_start:06d}-{shard.page_end:06d}"
    stdout_path = log_root / f"{log_stem}.stdout.log"
    stderr_path = log_root / f"{log_stem}.stderr.log"
    _atomic_write_text(stdout_path, stdout)
    _atomic_write_text(stderr_path, stderr)
    return ShardOutcome(
        shard.shard_id,
        status,
        return_code,
        command,
        stdout[-32_768:],
        stderr[-65_536:],
        started,
        _utc_now(),
        str(stdout_path),
        str(stderr_path),
    )


def _page_index(run_root: Path, stage: str) -> dict[tuple[str, int], Path]:
    output: dict[tuple[str, int], Path] = {}
    stage_root = run_root / "stages" / stage
    if not stage_root.is_dir():
        return output
    for path in stage_root.glob("*/pages-*/pages/page-*.json"):
        document_id = path.parents[2].name
        try:
            page = int(path.stem.removeprefix("page-"))
        except ValueError:
            continue
        key = (document_id, page)
        if key in output:
            raise RuntimeError(f"duplicate {stage} page artifact for {document_id} P{page}")
        output[key] = path
    return output


def _composed_payload_hash(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return sha256(_canonical_json(payload)).hexdigest()


def validate_composed_corpus_page(
    value: Mapping[str, object],
    *,
    document_id: str,
    page_number: int,
    method1_page: Mapping[str, object],
    method2_page: Mapping[str, object],
) -> None:
    """Validate a persisted composition against both exact stage payloads."""

    if value.get("payload_sha256") != _composed_payload_hash(value):
        raise ValueError("composed page checksum failed")
    expected = {
        "schema_version": COMPOSED_PAGE_SCHEMA_VERSION,
        "status": "candidate",
        "document_id": document_id,
        "page_number": page_number,
        "source_sha256": method1_page.get("source_sha256"),
        "operation_count": method1_page.get("operation_count"),
        "group_count": method1_page.get("group_count"),
        "method1_payload_sha256": method1_page.get("payload_sha256"),
        "method2_payload_sha256": method2_page.get("payload_sha256"),
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"composed page {key} does not match its stages")
    raw_method1 = method1_page.get("method1")
    raw_method2 = method2_page.get("method2")
    raw_fused = value.get("fused")
    if not all(isinstance(item, Mapping) for item in (
        raw_method1, raw_method2, raw_fused
    )):
        raise ValueError("composed page or stage result is malformed")
    assert isinstance(raw_method1, Mapping)
    assert isinstance(raw_method2, Mapping)
    assert isinstance(raw_fused, Mapping)
    raw_method1_result = raw_method1.get("result")
    if not isinstance(raw_method1_result, Mapping):
        raise ValueError("Method1 result is malformed")
    method1_result = LineTypeRecognitionResult.from_dict(raw_method1_result)
    method2_envelope = validate_line_type_method2_envelope(raw_method2)
    if value.get("page_identity") != method2_envelope.page_identity:
        raise ValueError("composed page identity does not match Method2")
    ticks = iter((0.0, 0.0))
    expected_fused = fuse_line_type_results_for_display(
        method2_envelope,
        method1_result,
        "full",
        clock=lambda: next(ticks),
    ).to_dict()
    if dict(raw_fused) != expected_fused:
        raise ValueError("composed fusion does not match its exact stage inputs")


def compose_corpus_pages(
    run_root: Path,
    documents: Sequence[CorpusDocument],
) -> dict[str, object]:
    method1_pages = _page_index(run_root, "method1")
    method2_pages = _page_index(run_root, "method2")
    completed: list[str] = []
    resumed: list[str] = []
    failures: dict[str, str] = {}
    for document in documents:
        for page_number in document.selected_pages:
            key = (document.document_id, page_number)
            label = f"{document.document_id}:page:{page_number}"
            method1_path = method1_pages.get(key)
            method2_path = method2_pages.get(key)
            if method1_path is None or method2_path is None:
                failures[label] = "missing method1 or method2 stage artifact"
                continue
            try:
                method1_page = _load_object(method1_path)
                method2_page = _load_object(method2_path)
                if method1_page is None or method2_page is None:
                    raise ValueError("stage page JSON is malformed")
                for stage_name, stage_page in (
                    ("method1", method1_page),
                    ("method2", method2_page),
                ):
                    if (
                        stage_page.get("payload_sha256")
                        != page_recognition_payload_hash(stage_page)
                        or stage_page.get("requested_outputs") != [stage_name]
                    ):
                        raise ValueError(
                            f"{stage_name} stage page checksum/output contract failed"
                        )
                identity_fields = (
                    "source_sha256", "page_number", "page_fingerprint",
                    "grouping_fingerprint", "operation_count", "group_count",
                    "page_ir_producer", "page_ir_producer_version",
                )
                if any(method1_page.get(field) != method2_page.get(field) for field in identity_fields):
                    raise ValueError("stage page inputs do not have the same identity")
                if method1_page.get("page_number") != page_number:
                    raise ValueError("stage page number does not match its artifact path")
                raw_method1 = method1_page.get("method1")
                raw_method2 = method2_page.get("method2")
                if not isinstance(raw_method1, Mapping) or not isinstance(raw_method2, Mapping):
                    raise ValueError("stage page is missing its requested result")
                raw_method1_result = raw_method1.get("result")
                if not isinstance(raw_method1_result, Mapping):
                    raise ValueError("Method1 result is malformed")
                method1_result = LineTypeRecognitionResult.from_dict(raw_method1_result)
                method2_envelope = validate_line_type_method2_envelope(raw_method2)
                ticks = iter((0.0, 0.0))
                fused = fuse_line_type_results_for_display(
                    method2_envelope,
                    method1_result,
                    "full",
                    clock=lambda: next(ticks),
                )
                payload: dict[str, object] = {
                    "schema_version": COMPOSED_PAGE_SCHEMA_VERSION,
                    "status": "candidate",
                    "document_id": document.document_id,
                    "page_number": page_number,
                    "page_identity": method2_envelope.page_identity,
                    "source_sha256": method1_page["source_sha256"],
                    "operation_count": method1_page["operation_count"],
                    "group_count": method1_page["group_count"],
                    "method1_payload_sha256": method1_page.get("payload_sha256"),
                    "method2_payload_sha256": method2_page.get("payload_sha256"),
                    "method1_input_hash_schema": method1_page.get(
                        "method1_input_hash_schema"
                    ),
                    "method1_input_hash": method1_page.get("method1_input_hash"),
                    "fused": fused.to_dict(),
                }
                payload["payload_sha256"] = _composed_payload_hash(payload)
                validate_composed_corpus_page(
                    payload,
                    document_id=document.document_id,
                    page_number=page_number,
                    method1_page=method1_page,
                    method2_page=method2_page,
                )
                destination = (
                    run_root / "composed" / document.document_id / "pages" /
                    f"page-{page_number:06d}.json"
                )
                existing = _load_object(destination)
                existing_valid = False
                if existing is not None:
                    try:
                        validate_composed_corpus_page(
                            existing,
                            document_id=document.document_id,
                            page_number=page_number,
                            method1_page=method1_page,
                            method2_page=method2_page,
                        )
                        existing_valid = True
                    except (KeyError, TypeError, ValueError):
                        existing_valid = False
                if existing_valid:
                    resumed.append(label)
                else:
                    _atomic_write_json(destination, payload)
                completed.append(label)
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                failures[label] = f"{type(error).__name__}: {error}"
    return {
        "status": "complete" if not failures else "partial",
        "completed_count": len(completed),
        "resumed_count": len(resumed),
        "completed_pages": completed,
        "resumed_pages": resumed,
        "failures": failures,
        "updated_at": _utc_now(),
    }


ShardRunner = Callable[[CorpusRunOptions, CorpusShard, Path], ShardOutcome]


def run_corpus(
    options: CorpusRunOptions,
    *,
    shard_runner: ShardRunner = run_shard_subprocess,
) -> dict[str, object]:
    plan = build_corpus_plan(options)
    if options.dry_run:
        return {
            "schema_version": CORPUS_RUN_SCHEMA_VERSION,
            "status": "dry-run",
            **plan.to_dict(),
        }
    options.run_root.mkdir(parents=True, exist_ok=True)
    with _corpus_lock(options.run_root):
        existing = _ensure_owned_root(options, plan)
        manifest = _base_manifest(options, plan, existing)
        manifest_path = options.run_root / "corpus-run.json"
        _atomic_write_json(manifest_path, manifest)

        sources = manifest["sources"]
        assert isinstance(sources, dict)
        for document in plan.documents:
            previous = sources.get(document.document_id)
            try:
                snapshot = _snapshot_document(
                    document,
                    options.run_root,
                    previous if isinstance(previous, Mapping) else None,
                )
            except (OSError, RuntimeError) as error:
                manifest["status"] = "snapshot_failed"
                manifest["failures"] = {
                    f"snapshot:{document.document_id}": (
                        f"{type(error).__name__}: {error}"
                    )
                }
                manifest["updated_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                return manifest
            sources[document.document_id] = snapshot
            manifest["updated_at"] = _utc_now()
            _atomic_write_json(manifest_path, manifest)

        manifest["status"] = "running"
        jobs_payload = manifest["jobs"]
        assert isinstance(jobs_payload, dict)
        snapshots = {
            document_id: Path(value["path"])
            for document_id, value in sources.items()
            if isinstance(value, Mapping) and isinstance(value.get("path"), str)
        }
        snapshot_hashes = {
            document_id: str(value["sha256"])
            for document_id, value in sources.items()
            if isinstance(value, Mapping)
            and _is_lower_hex(value.get("sha256"), 64)
        }
        document_page_counts = {
            document.document_id: document.page_count for document in plan.documents
        }
        all_failures: dict[str, str] = {}
        blocked_after_stage: str | None = None
        implementation_changed = False
        for stage in (stage for stage in options.stages if stage != "compose"):
            tasks = [shard for shard in plan.shards if shard.stage == stage]
            if _implementation_identity(options) != dict(plan.implementation):
                implementation_changed = True
                all_failures["implementation"] = (
                    "Python source changed after the corpus plan was created"
                )
                for blocked in plan.shards:
                    if STAGE_ORDER.index(blocked.stage) < STAGE_ORDER.index(stage):
                        continue
                    jobs_payload[blocked.shard_id] = {
                        **blocked.to_dict(),
                        "status": "blocked_after_implementation_change",
                        "updated_at": _utc_now(),
                    }
                manifest["status"] = "implementation_changed"
                manifest["failures"] = all_failures
                manifest["updated_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                break
            maximum_jobs = {
                "input": options.input_jobs,
                "method1": options.method1_jobs,
                "method2": options.method2_jobs,
                "fused": options.fused_jobs,
            }[stage]
            assert maximum_jobs is not None
            futures: dict[Future[ShardOutcome], CorpusShard] = {}
            active_by_document: dict[str, int] = {}
            pending = list(tasks)
            with ThreadPoolExecutor(max_workers=maximum_jobs) as executor:
                while pending or futures:
                    submitted = False
                    while len(futures) < maximum_jobs:
                        eligible_index = next(
                            (
                                index
                                for index, candidate in enumerate(pending)
                                if active_by_document.get(candidate.document_id, 0)
                                < options.max_jobs_per_document
                            ),
                            None,
                        )
                        if eligible_index is None:
                            break
                        shard = pending.pop(eligible_index)
                        snapshot = snapshots[shard.document_id]
                        command = _child_command(options, shard, snapshot)
                        jobs_payload[shard.shard_id] = {
                            **shard.to_dict(),
                            "status": "running",
                            "command": list(command),
                            "updated_at": _utc_now(),
                        }
                        futures[executor.submit(
                            shard_runner, options, shard, snapshot
                        )] = shard
                        active_by_document[shard.document_id] = (
                            active_by_document.get(shard.document_id, 0) + 1
                        )
                        submitted = True
                    if submitted:
                        manifest["updated_at"] = _utc_now()
                        _atomic_write_json(manifest_path, manifest)
                    if not futures:
                        break
                    completed, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                    for future in completed:
                        shard = futures.pop(future)
                        active_by_document[shard.document_id] -= 1
                        try:
                            outcome = future.result()
                        except Exception as error:  # Coordinator failures are isolated too.
                            outcome = ShardOutcome(
                                shard.shard_id,
                                "failed",
                                -1,
                                _child_command(
                                    options, shard, snapshots[shard.document_id]
                                ),
                                "",
                                f"{type(error).__name__}: {error}",
                                _utc_now(),
                                _utc_now(),
                            )
                        if outcome.status == "complete":
                            try:
                                validate_corpus_shard_artifacts(
                                    shard,
                                    source_sha256=snapshot_hashes[shard.document_id],
                                    document_page_count=document_page_counts[
                                        shard.document_id
                                    ],
                                    expected_worker_count=(
                                        options.fused_workers
                                        if shard.stage == "fused"
                                        else None
                                    ),
                                )
                            except (KeyError, OSError, TypeError, ValueError) as error:
                                artifact_error = (
                                    "coordinator artifact validation failed: "
                                    f"{type(error).__name__}: {error}"
                                )
                                outcome = ShardOutcome(
                                    outcome.shard_id,
                                    "failed",
                                    outcome.return_code,
                                    outcome.command,
                                    outcome.stdout_tail,
                                    "\n".join(filter(None, (
                                        outcome.stderr_tail,
                                        artifact_error,
                                    ))),
                                    outcome.started_at,
                                    outcome.finished_at,
                                    outcome.stdout_log,
                                    outcome.stderr_log,
                                )
                        jobs_payload[shard.shard_id] = {
                            **shard.to_dict(),
                            **outcome.to_dict(),
                        }
                        if outcome.status != "complete":
                            all_failures[shard.shard_id] = (
                                outcome.stderr_tail
                                or f"child exit {outcome.return_code}"
                            )
                        manifest["failures"] = all_failures
                        manifest["updated_at"] = _utc_now()
                        _atomic_write_json(manifest_path, manifest)

            if _implementation_identity(options) != dict(plan.implementation):
                implementation_changed = True
                all_failures["implementation"] = (
                    f"Python source changed while the {stage} stage was running"
                )
                for blocked in plan.shards:
                    if STAGE_ORDER.index(blocked.stage) <= STAGE_ORDER.index(stage):
                        continue
                    jobs_payload[blocked.shard_id] = {
                        **blocked.to_dict(),
                        "status": "blocked_after_implementation_change",
                        "updated_at": _utc_now(),
                    }
                manifest["status"] = "implementation_changed"
                manifest["failures"] = all_failures
                manifest["updated_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                break

            if any(shard.shard_id in all_failures for shard in tasks):
                blocked_after_stage = stage
                for blocked in plan.shards:
                    if STAGE_ORDER.index(blocked.stage) <= STAGE_ORDER.index(stage):
                        continue
                    jobs_payload[blocked.shard_id] = {
                        **blocked.to_dict(),
                        "status": f"blocked_after_{stage}",
                        "updated_at": _utc_now(),
                    }
                manifest["status"] = (
                    "preflight_failed" if stage == "input" else f"{stage}_failed"
                )
                manifest["updated_at"] = _utc_now()
                _atomic_write_json(manifest_path, manifest)
                break

        if (
            "compose" in options.stages
            and blocked_after_stage is None
            and not implementation_changed
        ):
            compose = compose_corpus_pages(options.run_root, plan.documents)
            manifest["compose"] = compose
            for label, message in compose["failures"].items():
                all_failures[f"compose:{label}"] = message
        manifest["failures"] = all_failures
        manifest["status"] = (
            "implementation_changed"
            if implementation_changed
            else "preflight_failed"
            if blocked_after_stage == "input"
            else f"{blocked_after_stage}_failed"
            if blocked_after_stage is not None
            else ("complete" if not all_failures else "partial")
        )
        manifest["completed_shards"] = sum(
            isinstance(value, Mapping) and value.get("status") == "complete"
            for value in jobs_payload.values()
        )
        manifest["failed_shards"] = len(plan.shards) - int(
            manifest["completed_shards"]
        )
        manifest["updated_at"] = _utc_now()
        _atomic_write_json(manifest_path, manifest)
        return manifest


__all__ = [
    "COMPOSED_PAGE_SCHEMA_VERSION",
    "CORPUS_RUN_SCHEMA_VERSION",
    "DEFAULT_SOURCE_ROOT",
    "CorpusDocument",
    "CorpusPlan",
    "CorpusRunOptions",
    "CorpusShard",
    "ShardOutcome",
    "build_corpus_plan",
    "compose_corpus_pages",
    "load_corpus_documents",
    "run_corpus",
    "run_shard_subprocess",
    "validate_composed_corpus_page",
    "validate_corpus_shard_artifacts",
    "validate_corpus_stage_page",
]
