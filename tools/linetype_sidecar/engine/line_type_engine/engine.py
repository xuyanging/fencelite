"""Public, frontend-independent composition root for line-type recognition.

The implementation modules remain separately testable, but callers should use
this file instead of assembling Method1, Method2 and fusion themselves.  It is
the boundary consumed by the whole-document runner and, later, by any viewer.
No browser, HTTP or cache transport is imported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import TYPE_CHECKING, Literal, Mapping, Sequence

from .grouping import group_page_sequentially
from .ir import GroupingIR, PageIR
from .scheduling import plan_single_page_execution, run_independent_methods
from .versions import (
    FROZEN_TS_FUSION_POLICY_VERSION,
    FROZEN_TS_METHOD1_ENGINE_VERSION,
    FROZEN_TS_METHOD2_ENGINE_VERSION,
    PAGE_ANALYSIS_SCHEMA_VERSION,
    PYTHON_FUSION_ENGINE_VERSION,
    PYTHON_ENGINE_VERSION,
    PYTHON_METHOD1_ENGINE_VERSION,
    PYTHON_METHOD2_ENGINE_VERSION,
    RESULT_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from .fusion_contract import FusedLineTypeEnvelope
    from .method1.pipeline import Method1CandidateRecognition
    from .method2.contract import LineTypeMethod2Envelope
    from .source_page_adapter import SourceAlignmentSummary


def _recognize_method2_payload_in_worker(
    page: PageIR,
    grouping: GroupingIR,
    page_identity: str,
    worker_count: int,
) -> dict[str, object]:
    """Return a pickle-safe Method2 payload from the Windows spawn worker."""

    from .method2.recognizer import recognize_line_types_method2_page

    return recognize_line_types_method2_page(
        page,
        grouping,
        page_identity=page_identity,
        worker_count=worker_count,
    ).to_dict()


def _recognize_method2_direct(
    page: PageIR,
    grouping: GroupingIR,
    page_identity: str,
    worker_count: int,
) -> LineTypeMethod2Envelope:
    from .method2.recognizer import recognize_line_types_method2_page

    return recognize_line_types_method2_page(
        page,
        grouping,
        page_identity=page_identity,
        worker_count=worker_count,
    )


OutputKind = Literal["input", "method1", "method2", "fused"]
OUTPUT_KINDS: tuple[OutputKind, ...] = ("input", "method1", "method2", "fused")


def page_recognition_payload_hash(value: Mapping[str, object]) -> str:
    """Hash a page envelope, excluding its own integrity field."""

    payload = dict(value)
    payload.pop("payload_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalize_outputs(outputs: Sequence[str]) -> tuple[OutputKind, ...]:
    """Validate and canonicalize requested outputs.

    ``fused`` is a derived output.  Its source recognizers are executed and
    retained in the returned page analysis so an operator can always audit
    which method produced each line type.
    """

    if isinstance(outputs, (str, bytes)):
        raise ValueError("outputs must be a sequence, not one string")
    requested = set(outputs)
    unknown = sorted(requested - set(OUTPUT_KINDS))
    if unknown:
        raise ValueError(f"unsupported output kind(s): {', '.join(unknown)}")
    if not requested:
        raise ValueError("at least one output kind is required")
    return tuple(kind for kind in OUTPUT_KINDS if kind in requested)


@dataclass(frozen=True, slots=True)
class PageLineTypeRecognition:
    page_identity: str
    page: PageIR
    grouping: GroupingIR
    requested_outputs: tuple[OutputKind, ...]
    method1: Method1CandidateRecognition | None
    method2: LineTypeMethod2Envelope | None
    fused: FusedLineTypeEnvelope | None
    method1_input_hash_schema: str | None = None
    method1_input_hash: str | None = None
    source_alignment_audit: SourceAlignmentSummary | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "status": "candidate",
            "page_analysis_schema_version": PAGE_ANALYSIS_SCHEMA_VERSION,
            "python_engine_version": PYTHON_ENGINE_VERSION,
            "page_ir_version": self.page.page_ir_version,
            "grouping_version": self.grouping.grouping_version,
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
            "page_identity": self.page_identity,
            "source_name": self.page.source_name,
            "source_sha256": self.page.source_sha256,
            "page_ir_producer": self.page.producer,
            "page_ir_producer_version": self.page.producer_version,
            "page_number": self.page.page_number,
            "page_fingerprint": self.page.fingerprint,
            "grouping_fingerprint": self.grouping.fingerprint,
            "operation_count": len(self.page.operations),
            "group_count": len(self.grouping.groups),
            "requested_outputs": list(self.requested_outputs),
        }
        if self.source_alignment_audit is not None:
            payload["source_alignment_audit"] = (
                self.source_alignment_audit.to_dict()
            )
        if self.method1 is not None:
            payload["method1"] = self.method1.to_dict()
        if self.method1_input_hash is not None:
            assert self.method1_input_hash_schema is not None
            payload["method1_input_hash_schema"] = self.method1_input_hash_schema
            payload["method1_input_hash"] = self.method1_input_hash
        if self.method2 is not None:
            payload["method2"] = self.method2.to_dict()
        if self.fused is not None:
            payload["fused"] = self.fused.to_dict()
        payload["payload_sha256"] = page_recognition_payload_hash(payload)
        return payload


def recognize_page(
    page: PageIR,
    *,
    grouping: GroupingIR | None = None,
    outputs: Sequence[str] = ("fused",),
    page_identity: str | None = None,
    method1_worker_count: int | None = None,
    parallel_methods: bool = True,
) -> PageLineTypeRecognition:
    """Run selected Python algorithms on one canonical page.

    The function performs no persistence. Method1 and Method2 stay independent
    and fusion is applied only after both complete. When both are needed they
    share one bounded CPU budget and may overlap. Requesting Method2 alone
    never starts Method1 workers.
    """

    requested = normalize_outputs(outputs)
    actual_grouping = grouping or group_page_sequentially(page)
    if actual_grouping.page_fingerprint != page.fingerprint:
        raise ValueError("GroupingIR does not belong to the supplied PageIR")
    identity = (
        f"{page.source_sha256}:page:{page.page_number}"
        if page_identity is None
        else page_identity
    )
    if not isinstance(identity, str) or not identity:
        raise ValueError("page_identity must be a non-empty string")

    needs_method1 = "method1" in requested or "fused" in requested
    needs_method2 = "method2" in requested or "fused" in requested
    needs_input = "input" in requested
    plan = plan_single_page_execution(
        needs_method1=needs_method1,
        needs_method2=needs_method2,
        cpu_budget=method1_worker_count,
        parallel_methods=parallel_methods,
    )

    def run_method1() -> Method1CandidateRecognition:
        # Keep a Method2-only process independent from Method1 imports and
        # their multiprocessing implementation.
        from .method1.pipeline import recognize_method1_candidate

        return recognize_method1_candidate(
            page,
            actual_grouping,
            worker_count=plan.method1_worker_count,
        )

    method1: Method1CandidateRecognition | None = None
    method2: LineTypeMethod2Envelope | None = None
    if needs_method1 and needs_method2:
        method1, method2_payload = run_independent_methods(
            run_method1,
            _recognize_method2_payload_in_worker,
            method2_arguments=(
                page,
                actual_grouping,
                identity,
                plan.method2_worker_count,
            ),
            concurrently=plan.concurrent_methods,
        )
        from .method2.contract import LineTypeMethod2Envelope

        method2 = LineTypeMethod2Envelope.from_dict(method2_payload)
    elif needs_method1:
        method1 = run_method1()
    elif needs_method2:
        method2 = _recognize_method2_direct(
            page,
            actual_grouping,
            identity,
            plan.method2_worker_count,
        )

    method1_input_hash_schema: str | None = None
    method1_input_hash: str | None = None
    if method1 is not None or needs_input:
        from .method1.serializer import (
            METHOD1_SERIALIZED_INPUT_HASH_SCHEMA,
            serialize_groups,
            serialized_method1_input_hash,
        )

        serialized_groups = (
            method1.serialized_groups
            if method1 is not None
            else serialize_groups(page, actual_grouping)
        )
        method1_input_hash_schema = METHOD1_SERIALIZED_INPUT_HASH_SCHEMA
        method1_input_hash = serialized_method1_input_hash(serialized_groups)

    fused: FusedLineTypeEnvelope | None = None
    if "fused" in requested:
        assert method1 is not None and method2 is not None
        from .fusion_contract import fuse_line_type_results_for_display

        fused = fuse_line_type_results_for_display(
            method2,
            method1.result,
            "full",
        )
    return PageLineTypeRecognition(
        page_identity=identity,
        page=page,
        grouping=actual_grouping,
        requested_outputs=requested,
        method1=method1,
        method2=method2,
        fused=fused,
        method1_input_hash_schema=method1_input_hash_schema,
        method1_input_hash=method1_input_hash,
    )


__all__ = [
    "OUTPUT_KINDS",
    "OutputKind",
    "PageLineTypeRecognition",
    "normalize_outputs",
    "page_recognition_payload_hash",
    "recognize_page",
]
