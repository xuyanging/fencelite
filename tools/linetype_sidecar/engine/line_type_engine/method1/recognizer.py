"""Python Method1 composition root.

At this migration stage the module exposes the validated *base* recognizer:
PageIR -> GroupingIR -> command adapter -> existing Python Group discovery ->
Python cross-Group registry.  The frozen r10 post-processing stages are added
as separately testable modules before a production ``recognize_method1`` name
is enabled; callers cannot accidentally mistake a partial port for r10.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from typing import Any, Iterable

from ..ir import GroupingIR, PageIR
from .core import _error_text
from ..results import LineTypeRecognitionResult, RecognizedGroup
from .core import analyze_group, finalize_registry
from .serializer import SerializedGroup, serialize_groups, validate_group_classification


# A Windows ``spawn`` process pool has a measurable fixed cost even when every
# worker function is pure.  Real-page measurements put small pages (423 and
# 1,002 serialized atoms) firmly below that break-even point, while a 5,928
# atom page benefits substantially.  Keep the policy here, at the execution
# boundary, so it cannot leak into Method1 fingerprints or recognition rules.
_PROCESS_POOL_MIN_SERIALIZED_ATOMS = 4_096


@dataclass(frozen=True, slots=True)
class Method1BaseAudit:
    """Base-stage telemetry; ``worker_count`` is the effective pool size."""

    input_group_count: int
    processed_group_count: int
    failed_group_count: int
    serialized_atom_count: int
    worker_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            "input_group_count": self.input_group_count,
            "processed_group_count": self.processed_group_count,
            "failed_group_count": self.failed_group_count,
            "serialized_atom_count": self.serialized_atom_count,
            "worker_count": self.worker_count,
        }


@dataclass(frozen=True, slots=True)
class Method1BaseRecognition:
    result: LineTypeRecognitionResult
    audit: Method1BaseAudit
    serialized_groups: tuple[SerializedGroup, ...]


def _analyze_serialized(payload: dict[str, Any]) -> dict[str, Any]:
    return analyze_group(payload)


def _requested_worker_budget(requested: int | None) -> int:
    """Clamp one page's execution hint to the CPUs available to this process."""

    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
    ):
        raise ValueError("worker_count must be a positive integer")
    available = os.cpu_count() or 1
    return max(1, min(requested or available, available))


def _worker_budget(
    requested: int | None,
    serialized: tuple[SerializedGroup, ...],
) -> int:
    page_budget = _requested_worker_budget(requested)
    maximum = max(
        1,
        min(len(serialized) or 1, page_budget),
    )
    if maximum == 1:
        return 1

    # Only serialized path atoms are Method1 base-recognizer work.  Native
    # text/images and PDF page numbers must not make a cheap Group workload
    # look parallel-worthy.  Requiring at least two non-empty Groups also
    # proves there is work that this Group-level pool can distribute.
    nonempty_group_count = sum(bool(group.atom_op_indices) for group in serialized)
    serialized_atom_count = sum(
        len(group.atom_op_indices)
        for group in serialized
    )
    if (
        nonempty_group_count < 2
        or serialized_atom_count < _PROCESS_POOL_MIN_SERIALIZED_ATOMS
    ):
        return 1
    return maximum


def _analyze_groups(
    serialized: tuple[SerializedGroup, ...],
    worker_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    payloads = [group.to_dict() for group in serialized]
    groups: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    if worker_count == 1 or len(payloads) <= 1:
        outcomes: Iterable[dict[str, Any] | Exception]
        sequential: list[dict[str, Any] | Exception] = []
        for payload in payloads:
            try:
                sequential.append(_analyze_serialized(payload))
            except Exception as error:  # Isolate ordinary Group failures only.
                sequential.append(error)
        outcomes = sequential
    else:
        outcomes_list: list[dict[str, Any] | Exception] = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(_analyze_serialized, payload) for payload in payloads]
            for future in futures:  # Submission order, independent of completion order.
                try:
                    outcomes_list.append(future.result())
                except Exception as error:
                    outcomes_list.append(error)
        outcomes = outcomes_list

    for source, outcome in zip(serialized, outcomes):
        if isinstance(outcome, Exception):
            errors.append({"group_id": source.group_id,
                           "message": _error_text(outcome)})
            continue
        try:
            public_group = RecognizedGroup.from_dict(outcome)
            validate_group_classification(source, public_group)
            groups.append(outcome)
        except Exception as error:
            errors.append({"group_id": source.group_id,
                           "message": _error_text(error)})
    return groups, errors


def recognize_method1_base(
    page: PageIR,
    grouping: GroupingIR,
    *,
    worker_count: int | None = None,
) -> Method1BaseRecognition:
    """Run the complete pre-postprocessor Method1 Python pipeline.

    The explicit ``_base`` suffix is a safety boundary.  It is removed only
    after every r10 post-processing stage has an operation-level parity gate.
    """

    serialized = serialize_groups(page, grouping)
    workers = _worker_budget(worker_count, serialized)
    analyzed_groups, errors = _analyze_groups(serialized, workers)
    raw_result = finalize_registry(analyzed_groups, len(serialized), errors)
    result = LineTypeRecognitionResult.from_dict(raw_result)
    return Method1BaseRecognition(
        result=result,
        audit=Method1BaseAudit(
            input_group_count=len(serialized),
            processed_group_count=len(analyzed_groups),
            failed_group_count=len(errors),
            serialized_atom_count=sum(len(group.atom_op_indices) for group in serialized),
            worker_count=workers,
        ),
        serialized_groups=serialized,
    )
