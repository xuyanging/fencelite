"""Single composition root for the complete Python Method1 candidate.

Algorithm stages remain small, directly testable modules.  This file owns the
only legal production order and records stage-level audit counters so a bad
result can be localized without the viewer.  The ``candidate`` suffix remains
until PageIR/Grouping and operation-level frozen-r10 parity gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

from ..ir import GroupingIR, PageIR
from ..results import LineTypeRecognitionResult
from ..versions import PYTHON_METHOD1_ENGINE_VERSION
from .postprocess_compound import (
    augment_compound_path_line_types,
    enforce_extendable_route_line_types,
)
from .postprocess_legend import (
    augment_legend_table_solid_samples,
    augment_vector_legend_samples,
)
from .postprocess_reclaim import reclaim_confirmed_line_types
from .postprocess_stitch import (
    augment_repeated_stitch_path_line_types,
    demote_outlined_stroke_text_line_types,
)
from .postprocess_text import (
    augment_short_inline_text_patterns,
    augment_text_labeled_line_types,
)
from .recognizer import (
    Method1BaseAudit,
    _requested_worker_budget,
    recognize_method1_base,
)
from .serializer import SerializedGroup


METHOD1_POSTPROCESS_STAGE_NAMES = (
    "compound_path_augmentation",
    "extendable_route_filter",
    "outlined_stroke_text_demotion",
    "repeated_stitch_augmentation",
    "repeated_text_label_augmentation",
    "short_inline_text_identity",
    "vector_legend_samples",
    "solid_legend_samples",
    "confirmed_type_reclaim",
)


@dataclass(frozen=True, slots=True)
class Method1StageAudit:
    stage: str
    elapsed_ms: float
    local_type_count_before: int
    local_type_count_after: int
    global_type_count_before: int
    global_type_count_after: int
    owned_operation_count_before: int
    owned_operation_count_after: int

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "stage": self.stage,
            "elapsed_ms": self.elapsed_ms,
            "local_type_count_before": self.local_type_count_before,
            "local_type_count_after": self.local_type_count_after,
            "global_type_count_before": self.global_type_count_before,
            "global_type_count_after": self.global_type_count_after,
            "owned_operation_count_before": self.owned_operation_count_before,
            "owned_operation_count_after": self.owned_operation_count_after,
        }


@dataclass(frozen=True, slots=True)
class Method1CandidateAudit:
    engine_version: str
    base: Method1BaseAudit
    stages: tuple[Method1StageAudit, ...]
    elapsed_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "engine_version": self.engine_version,
            "base": self.base.to_dict(),
            "stages": [stage.to_dict() for stage in self.stages],
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass(frozen=True, slots=True)
class Method1CandidateRecognition:
    result: LineTypeRecognitionResult
    audit: Method1CandidateAudit
    serialized_groups: tuple[SerializedGroup, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the durable public payload, excluding internal Group commands."""

        return {
            "result": self.result.to_dict(),
            "audit": self.audit.to_dict(),
        }


def _owned_operation_count(result: LineTypeRecognitionResult) -> int:
    return len({
        op_index
        for group in result.groups
        for line_type in group.line_types
        for op_index in line_type.op_indices
    })


def _local_type_count(result: LineTypeRecognitionResult) -> int:
    return sum(group.line_type_count for group in result.groups)


def _validated(result: LineTypeRecognitionResult) -> LineTypeRecognitionResult:
    return LineTypeRecognitionResult.from_dict(result.to_dict())


def apply_method1_postprocessors(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: tuple[SerializedGroup, ...],
    base_result: LineTypeRecognitionResult,
    *,
    worker_count: int | None = None,
) -> tuple[LineTypeRecognitionResult, tuple[Method1StageAudit, ...]]:
    """Apply all frozen-r10 postprocessors in their canonical order."""

    stages: tuple[
        tuple[str, Callable[[LineTypeRecognitionResult], LineTypeRecognitionResult]], ...
    ] = (
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[0],
            lambda result: augment_compound_path_line_types(
                page,
                grouping,
                serialized_groups,
                result,
                worker_count=worker_count,
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[1],
            lambda result: enforce_extendable_route_line_types(
                page, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[2],
            lambda result: demote_outlined_stroke_text_line_types(
                page, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[3],
            lambda result: augment_repeated_stitch_path_line_types(
                page, grouping, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[4],
            lambda result: augment_text_labeled_line_types(
                page, grouping, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[5],
            lambda result: augment_short_inline_text_patterns(
                page, grouping, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[6],
            lambda result: augment_vector_legend_samples(
                page, grouping, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[7],
            lambda result: augment_legend_table_solid_samples(
                page, grouping, serialized_groups, result
            ),
        ),
        (
            METHOD1_POSTPROCESS_STAGE_NAMES[8],
            lambda result: reclaim_confirmed_line_types(
                page, grouping, serialized_groups, result
            ),
        ),
    )

    result = _validated(base_result)
    audits: list[Method1StageAudit] = []
    for stage_name, stage in stages:
        started = perf_counter()
        before = result
        try:
            result = stage(before)
            # Frozen r10 deliberately lets the compound stage leave summary
            # counters stale because the immediately following route stage
            # rebuilds every counter from Groups.  Validating that intermediate
            # object as a persisted result would reject correct oracle
            # behaviour; every other stage, and the final result, is strict.
            if stage_name != "compound_path_augmentation":
                result = _validated(result)
        except ValueError as error:
            raise ValueError(
                f"Method1 postprocessor {stage_name!r} produced an invalid result: {error}"
            ) from error
        audits.append(Method1StageAudit(
            stage=stage_name,
            elapsed_ms=round((perf_counter() - started) * 1000.0, 3),
            local_type_count_before=_local_type_count(before),
            local_type_count_after=_local_type_count(result),
            global_type_count_before=len(before.global_types),
            global_type_count_after=len(result.global_types),
            owned_operation_count_before=_owned_operation_count(before),
            owned_operation_count_after=_owned_operation_count(result),
        ))
    return result, tuple(audits)


def recognize_method1_candidate(
    page: PageIR,
    grouping: GroupingIR,
    *,
    worker_count: int | None = None,
) -> Method1CandidateRecognition:
    """Run complete Python Method1 without frontend, HTTP, or TypeScript.

    The API is deliberately marked candidate.  It becomes the production
    ``recognize_method1`` only after the named migration pages pass the frozen
    r10 operation-partition gate.
    """

    started = perf_counter()
    page_worker_budget = _requested_worker_budget(worker_count)
    base = recognize_method1_base(
        page,
        grouping,
        worker_count=page_worker_budget,
    )
    if (
        base.audit.processed_group_count != base.audit.input_group_count
        or base.audit.failed_group_count
        or base.result.errors
    ):
        details = "; ".join(
            f"Group {error.group_id}: {error.message}"
            for error in base.result.errors[:5]
        )
        suffix = f" ({details})" if details else ""
        raise RuntimeError(
            "Method1 base recognition did not cover every Group: "
            f"processed {base.audit.processed_group_count}/"
            f"{base.audit.input_group_count}, failed {base.audit.failed_group_count}"
            f"{suffix}"
        )
    result, stages = apply_method1_postprocessors(
        page,
        grouping,
        base.serialized_groups,
        base.result,
        worker_count=page_worker_budget,
    )
    return Method1CandidateRecognition(
        result=result,
        audit=Method1CandidateAudit(
            engine_version=PYTHON_METHOD1_ENGINE_VERSION,
            base=base.audit,
            stages=stages,
            elapsed_ms=round((perf_counter() - started) * 1000.0, 3),
        ),
        serialized_groups=base.serialized_groups,
    )
