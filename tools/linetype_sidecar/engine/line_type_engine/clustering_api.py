"""UI-neutral public API for clustering canonical PDF vector commands.

This is the reusable package boundary: complete :class:`PageIR` commands go
in and command ownership comes out.  PDF transport, caches, HTTP, React,
Canvas, palettes and highlighting deliberately live outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .ir import ImageOperationIR, PageIR, PathOperationIR, TextOperationIR
from .results import LineTypeRecognitionResult
from .versions import PYTHON_ENGINE_VERSION


CLUSTERING_API_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class CommandRange:
    """Half-open dense command interval ``[start, end)``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or self.start < 0
            or self.end <= self.start
        ):
            raise ValueError("command range must be a non-empty half-open interval")

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end}


def command_ranges(op_indices: Iterable[int]) -> tuple[CommandRange, ...]:
    """Compress ordered command identities without changing their meaning."""

    indices = tuple(op_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError("operation indices must be non-negative integers")
    if any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("operation indices must be unique and increasing")
    if not indices:
        return ()
    ranges: list[CommandRange] = []
    start = indices[0]
    previous = start
    for index in indices[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(CommandRange(start, previous + 1))
        start = previous = index
    ranges.append(CommandRange(start, previous + 1))
    return tuple(ranges)


@dataclass(frozen=True, slots=True)
class CommandOwnership:
    op_indices: tuple[int, ...]
    operation_ids: tuple[str, ...]
    ranges: tuple[CommandRange, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "op_indices": list(self.op_indices),
            "operation_ids": list(self.operation_ids),
            "ranges": [item.to_dict() for item in self.ranges],
        }


@dataclass(frozen=True, slots=True)
class NonVectorSupport:
    """Auditable non-vector evidence retained by an algorithm line family."""

    kind: str
    op_index: int
    operation_id: str

    def __post_init__(self) -> None:
        if self.kind not in {"text", "image"}:
            raise ValueError("non-vector support kind must be text or image")
        if (
            isinstance(self.op_index, bool)
            or not isinstance(self.op_index, int)
            or self.op_index < 0
        ):
            raise ValueError("non-vector support index must be non-negative")
        if not isinstance(self.operation_id, str) or not self.operation_id:
            raise ValueError("non-vector support operation_id must not be empty")

    def to_dict(self) -> dict[str, str | int]:
        return {
            "kind": self.kind,
            "op_index": self.op_index,
            "operation_id": self.operation_id,
        }


@dataclass(frozen=True, slots=True)
class LocalLineTypeCluster:
    line_type_id: str
    line_type_number: int
    atom_count: int
    model: str
    shape: str
    commands: CommandOwnership
    non_vector_support: tuple[NonVectorSupport, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "line_type_id": self.line_type_id,
            "line_type_number": self.line_type_number,
            "atom_count": self.atom_count,
            "model": self.model,
            "shape": self.shape,
            "commands": self.commands.to_dict(),
            "non_vector_support": [
                item.to_dict() for item in self.non_vector_support
            ],
        }


@dataclass(frozen=True, slots=True)
class GroupLineTypeClusters:
    group_id: str
    atom_count: int
    line_types: tuple[LocalLineTypeCluster, ...]
    residual_vector_commands: CommandOwnership
    residual_non_vector_support: tuple[NonVectorSupport, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "atom_count": self.atom_count,
            "line_types": [item.to_dict() for item in self.line_types],
            "residual_vector_commands": self.residual_vector_commands.to_dict(),
            "residual_non_vector_support": [
                item.to_dict() for item in self.residual_non_vector_support
            ],
        }


@dataclass(frozen=True, slots=True)
class GlobalLineTypeMember:
    group_id: str
    local_line_type_id: str
    atom_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "local_line_type_id": self.local_line_type_id,
            "atom_count": self.atom_count,
        }


@dataclass(frozen=True, slots=True)
class GlobalLineTypeCluster:
    line_type_id: str
    line_type_number: int
    type_uid: str
    recognition_source: str | None
    source_line_type_id: str | None
    signature_family: str
    minimum_pair_similarity: float
    members: tuple[GlobalLineTypeMember, ...]
    commands: CommandOwnership
    non_vector_support: tuple[NonVectorSupport, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "line_type_id": self.line_type_id,
            "line_type_number": self.line_type_number,
            "type_uid": self.type_uid,
            "recognition_source": self.recognition_source,
            "source_line_type_id": self.source_line_type_id,
            "signature_family": self.signature_family,
            "minimum_pair_similarity": self.minimum_pair_similarity,
            "members": [item.to_dict() for item in self.members],
            "commands": self.commands.to_dict(),
            "non_vector_support": [
                item.to_dict() for item in self.non_vector_support
            ],
        }


@dataclass(frozen=True, slots=True)
class LineTypeClusteringResult:
    page_number: int
    page_fingerprint: str
    source_sha256: str
    operation_count: int
    group_count: int
    groups: tuple[GroupLineTypeClusters, ...]
    global_line_types: tuple[GlobalLineTypeCluster, ...]
    errors: tuple[tuple[str, str], ...]
    schema_version: int = CLUSTERING_API_SCHEMA_VERSION
    engine_version: str = PYTHON_ENGINE_VERSION

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "page_number": self.page_number,
            "page_fingerprint": self.page_fingerprint,
            "source_sha256": self.source_sha256,
            "operation_count": self.operation_count,
            "group_count": self.group_count,
            "groups": [item.to_dict() for item in self.groups],
            "global_line_types": [item.to_dict() for item in self.global_line_types],
            "errors": [
                {"group_id": group_id, "message": message}
                for group_id, message in self.errors
            ],
        }


@dataclass(frozen=True, slots=True)
class PageClusteringError:
    """Serializable exception summary for one independently processed page."""

    exception_type: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.exception_type, str) or not self.exception_type:
            raise ValueError("page clustering exception_type must not be empty")
        if not isinstance(self.message, str):
            raise ValueError("page clustering message must be a string")

    def to_dict(self) -> dict[str, str]:
        return {
            "exception_type": self.exception_type,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class PageClusteringOutcome:
    """Success or isolated failure for one input page."""

    page_number: int
    page_fingerprint: str
    result: LineTypeClusteringResult | None = None
    error: PageClusteringError | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.page_number, bool)
            or not isinstance(self.page_number, int)
            or self.page_number < 1
        ):
            raise ValueError("page clustering page_number must be positive")
        if (
            not isinstance(self.page_fingerprint, str)
            or len(self.page_fingerprint) != 64
            or self.page_fingerprint.lower() != self.page_fingerprint
            or any(
                character not in "0123456789abcdef"
                for character in self.page_fingerprint
            )
        ):
            raise ValueError("page clustering fingerprint must be lowercase SHA-256")
        if (self.result is None) == (self.error is None):
            raise ValueError("page clustering outcome requires one result or error")
        if self.result is not None and (
            self.result.page_number != self.page_number
            or self.result.page_fingerprint != self.page_fingerprint
        ):
            raise ValueError("page clustering result identity does not match outcome")

    @property
    def status(self) -> str:
        return "ok" if self.result is not None else "error"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "page_number": self.page_number,
            "page_fingerprint": self.page_fingerprint,
            "result": None if self.result is None else self.result.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }


class PageClusteringFailure(RuntimeError):
    """Raised by the explicit fail-fast multi-page mode."""

    def __init__(self, outcome: PageClusteringOutcome) -> None:
        if outcome.error is None:
            raise ValueError("PageClusteringFailure requires an error outcome")
        self.outcome = outcome
        super().__init__(
            f"page {outcome.page_number} clustering failed: "
            f"{outcome.error.exception_type}: {outcome.error.message}"
        )


def _project_ownership(
    page: PageIR,
    op_indices: Iterable[int],
) -> tuple[CommandOwnership, tuple[NonVectorSupport, ...]]:
    """Partition one exact recognizer ownership into vector and support data.

    Method1 text families can legitimately retain the text command that
    supports a recognized line family.  Downstream drawing integrations only
    consume Path commands, but the other owned operations must remain visible
    so the projection is lossless and auditable.
    """

    indices = tuple(op_indices)
    # Validate the recognizer's complete ownership before partitioning it.
    command_ranges(indices)
    path_indices: list[int] = []
    path_operation_ids: list[str] = []
    non_vector_support: list[NonVectorSupport] = []
    for index in indices:
        if index >= len(page.operations):
            raise ValueError(f"operation index {index} is outside PageIR")
        operation = page.operations[index]
        if isinstance(operation, PathOperationIR):
            path_indices.append(index)
            path_operation_ids.append(operation.operation_id)
        elif isinstance(operation, TextOperationIR):
            non_vector_support.append(
                NonVectorSupport("text", index, operation.operation_id)
            )
        elif isinstance(operation, ImageOperationIR):
            non_vector_support.append(
                NonVectorSupport("image", index, operation.operation_id)
            )
        else:
            raise ValueError(
                "line-type ownership references unsupported operation "
                f"kind at index {index}: {type(operation).__name__}"
            )
    commands = CommandOwnership(
        op_indices=tuple(path_indices),
        operation_ids=tuple(path_operation_ids),
        ranges=command_ranges(path_indices),
    )
    return commands, tuple(non_vector_support)


def project_line_type_clusters(
    page: PageIR,
    result: LineTypeRecognitionResult,
) -> LineTypeClusteringResult:
    """Project an algorithm result to typed command integration data.

    Path ownership is returned in ``commands`` for downstream visualization.
    Legitimate Text/Image evidence is retained separately as
    ``non_vector_support``.  Invalid indices and unknown operation variants
    fail closed instead of being silently filtered.
    """

    groups: list[GroupLineTypeClusters] = []
    for group in result.groups:
        local_line_types: list[LocalLineTypeCluster] = []
        for line_type in group.line_types:
            commands, support = _project_ownership(page, line_type.op_indices)
            local_line_types.append(LocalLineTypeCluster(
                line_type_id=line_type.type_id,
                line_type_number=line_type.line_type_index,
                atom_count=line_type.atom_count,
                model=line_type.model,
                shape=line_type.shape,
                commands=commands,
                non_vector_support=support,
            ))
        residual_commands, residual_support = _project_ownership(
            page,
            group.non_linetype.op_indices,
        )
        groups.append(GroupLineTypeClusters(
            group_id=group.group_id,
            atom_count=group.atom_count,
            line_types=tuple(local_line_types),
            residual_vector_commands=residual_commands,
            residual_non_vector_support=residual_support,
        ))
    global_line_types: list[GlobalLineTypeCluster] = []
    for number, line_type in enumerate(result.global_types, start=1):
        if not line_type.type_uid:
            raise ValueError(
                f"global line type {line_type.global_type_id!r} has no stable type_uid"
            )
        commands, support = _project_ownership(page, line_type.op_indices)
        global_line_types.append(GlobalLineTypeCluster(
            line_type_id=line_type.global_type_id,
            line_type_number=number,
            type_uid=line_type.type_uid,
            recognition_source=line_type.recognition_source,
            source_line_type_id=line_type.source_global_type_id,
            signature_family=line_type.signature_family,
            minimum_pair_similarity=line_type.minimum_pair_similarity,
            members=tuple(GlobalLineTypeMember(
                group_id=member.case_id,
                local_line_type_id=member.type_id,
                atom_count=member.atom_count,
            ) for member in line_type.members),
            commands=commands,
            non_vector_support=support,
        ))
    return LineTypeClusteringResult(
        page_number=page.page_number,
        page_fingerprint=page.fingerprint,
        source_sha256=page.source_sha256,
        operation_count=len(page.operations),
        group_count=len(groups),
        groups=tuple(groups),
        global_line_types=tuple(global_line_types),
        errors=tuple((error.group_id, error.message) for error in result.errors),
    )


def cluster_page_commands(
    page: PageIR,
    *,
    cpu_budget: int | None = None,
) -> LineTypeClusteringResult:
    """Cluster one canonical page with a bounded, execution-only CPU budget."""

    if not isinstance(page, PageIR):
        raise TypeError("page must be a PageIR; decode command JSON first")
    if cpu_budget is not None and (
        isinstance(cpu_budget, bool)
        or not isinstance(cpu_budget, int)
        or cpu_budget < 1
    ):
        raise ValueError("cpu_budget must be a positive integer")
    from .engine import recognize_page

    effective_cpu_budget = cpu_budget if _spawn_main_is_importable() else 1
    recognition = recognize_page(
        page,
        outputs=("fused",),
        method1_worker_count=effective_cpu_budget,
        parallel_methods=True,
    )
    if recognition.fused is None:
        raise RuntimeError("fused recognition did not produce a result")
    return project_line_type_clusters(page, recognition.fused.result)


def _page_execution_slots(
    page_count: int,
    *,
    cpu_budget: int | None,
    max_parallel_pages: int | None,
) -> tuple[int, ...]:
    """Return per-slot CPU budgets whose sum never exceeds the total budget."""

    if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
        raise ValueError("page_count must be a non-negative integer")
    if cpu_budget is not None and (
        isinstance(cpu_budget, bool)
        or not isinstance(cpu_budget, int)
        or cpu_budget < 1
    ):
        raise ValueError("cpu_budget must be a positive integer")
    if max_parallel_pages is not None and (
        isinstance(max_parallel_pages, bool)
        or not isinstance(max_parallel_pages, int)
        or max_parallel_pages < 1
    ):
        raise ValueError("max_parallel_pages must be a positive integer")

    from .scheduling import available_parallelism

    available = available_parallelism()
    total_budget = min(cpu_budget or available, available)
    if page_count == 0:
        return ()
    # Complete PageIR pages are independent.  Throughput-first defaults give
    # one CPU token to as many pages as possible; callers can set an explicit
    # page limit to trade parallel pages for more intra-page workers.
    requested_page_count = max_parallel_pages or total_budget
    slot_count = min(page_count, requested_page_count, total_budget)
    base, extra = divmod(total_budget, slot_count)
    return tuple(base + (index < extra) for index in range(slot_count))


def _page_error(page: PageIR, error: Exception) -> PageClusteringOutcome:
    return PageClusteringOutcome(
        page_number=page.page_number,
        page_fingerprint=page.fingerprint,
        error=PageClusteringError(
            exception_type=(
                f"{type(error).__module__}.{type(error).__qualname__}"
            ),
            message=str(error),
        ),
    )


def _cluster_page_task(page: PageIR, cpu_budget: int) -> PageClusteringOutcome:
    """Pickle-safe page task used by the Windows ``spawn`` executor."""

    try:
        result = cluster_page_commands(page, cpu_budget=cpu_budget)
    except Exception as error:
        return _page_error(page, error)
    return PageClusteringOutcome(
        page_number=page.page_number,
        page_fingerprint=page.fingerprint,
        result=result,
    )


def _spawn_main_is_importable() -> bool:
    """Whether ``spawn`` can reload the current main module safely.

    Python marks stdin and some embedded entry points with a pseudo-path such
    as ``<stdin>``.  Starting a process pool there crashes workers before the
    page task can isolate its own exception, so use the deterministic
    sequential path instead.
    """

    import __main__
    from pathlib import Path

    main_file = getattr(__main__, "__file__", None)
    if main_file is None:
        # ``python -c`` has no file to reload and is supported by spawn.
        return True
    if not isinstance(main_file, str) or not main_file:
        return False
    return Path(main_file).is_file()


def cluster_pages_commands(
    pages: Iterable[PageIR],
    *,
    cpu_budget: int | None = None,
    max_parallel_pages: int | None = None,
    fail_fast: bool = False,
) -> tuple[PageClusteringOutcome, ...]:
    """Cluster independent, already-parsed pages under one CPU budget.

    The returned tuple always follows input order.  By default, an exception
    is isolated in the corresponding :class:`PageClusteringOutcome` and other
    pages continue.  ``fail_fast=True`` instead raises
    :class:`PageClusteringFailure` after cancelling queued page work.

    Page processes receive disjoint CPU slots, and each slot passes its budget
    to :func:`cluster_page_commands`.  Therefore nested Method1/Method2
    parallelism cannot allocate more algorithm workers than the total budget.
    Scheduling values are execution hints only and are absent from results.
    """

    if not isinstance(fail_fast, bool):
        raise ValueError("fail_fast must be boolean")
    page_items = tuple(pages)
    if any(not isinstance(page, PageIR) for page in page_items):
        raise TypeError("pages must contain only PageIR values")
    slots = _page_execution_slots(
        len(page_items),
        cpu_budget=cpu_budget,
        max_parallel_pages=max_parallel_pages,
    )
    if not page_items:
        return ()

    spawn_main_importable = _spawn_main_is_importable()
    if len(slots) == 1 or not spawn_main_importable:
        # A non-importable main also cannot support the algorithm's nested
        # spawn workers, so keep the complete recognition path single-process.
        sequential_budget = slots[0] if spawn_main_importable else 1
        outcomes: list[PageClusteringOutcome] = []
        for page in page_items:
            outcome = _cluster_page_task(page, sequential_budget)
            if fail_fast and outcome.error is not None:
                raise PageClusteringFailure(outcome)
            outcomes.append(outcome)
        return tuple(outcomes)

    from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
    import multiprocessing

    ordered: list[PageClusteringOutcome | None] = [None] * len(page_items)
    next_index = 0
    failure: PageClusteringFailure | None = None
    try:
        executor = ProcessPoolExecutor(
            max_workers=len(slots),
            mp_context=multiprocessing.get_context("spawn"),
        )
    except Exception as error:
        outcomes = tuple(_page_error(page, error) for page in page_items)
        if fail_fast:
            raise PageClusteringFailure(outcomes[0]) from error
        return outcomes

    try:
        with executor:
            active: dict[Future[PageClusteringOutcome], tuple[int, int]] = {}

            def submit_next(slot_budget: int) -> None:
                """Fill one page slot, isolating executor submission failures."""

                nonlocal next_index, failure
                while next_index < len(page_items) and failure is None:
                    index = next_index
                    next_index += 1
                    try:
                        future = executor.submit(
                            _cluster_page_task,
                            page_items[index],
                            slot_budget,
                        )
                    except Exception as error:
                        outcome = _page_error(page_items[index], error)
                        ordered[index] = outcome
                        if fail_fast:
                            failure = PageClusteringFailure(outcome)
                        continue
                    active[future] = (index, slot_budget)
                    return

            for slot_budget in slots:
                submit_next(slot_budget)
                if failure is not None:
                    break

            while active and failure is None:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in sorted(completed, key=lambda item: active[item][0]):
                    index, slot_budget = active.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as error:
                        # Worker bootstrap failures and broken pools happen outside
                        # _cluster_page_task; they still belong to this page.
                        outcome = _page_error(page_items[index], error)
                    ordered[index] = outcome
                    if fail_fast and outcome.error is not None:
                        failure = PageClusteringFailure(outcome)
                        for pending in active:
                            pending.cancel()
                        break
                    submit_next(slot_budget)
    except Exception as error:
        # Shutdown/wait failures are executor infrastructure errors.  Preserve
        # completed results and attach the failure only to unfinished pages.
        for index, outcome in enumerate(ordered):
            if outcome is None:
                ordered[index] = _page_error(page_items[index], error)
        if fail_fast:
            first_error = next((
                item for item in ordered
                if item is not None and item.error is not None
            ), None)
            if first_error is not None:
                raise PageClusteringFailure(first_error) from error
    if failure is not None:
        raise failure
    if any(outcome is None for outcome in ordered):
        raise RuntimeError("multi-page clustering did not return every page")
    return tuple(outcome for outcome in ordered if outcome is not None)


__all__ = [
    "CLUSTERING_API_SCHEMA_VERSION",
    "CommandOwnership",
    "CommandRange",
    "GlobalLineTypeCluster",
    "GlobalLineTypeMember",
    "GroupLineTypeClusters",
    "LineTypeClusteringResult",
    "LocalLineTypeCluster",
    "NonVectorSupport",
    "PageClusteringError",
    "PageClusteringFailure",
    "PageClusteringOutcome",
    "cluster_page_commands",
    "cluster_pages_commands",
    "command_ranges",
    "project_line_type_clusters",
]
