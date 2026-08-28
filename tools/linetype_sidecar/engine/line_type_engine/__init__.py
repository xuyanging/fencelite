"""Python-first PDF line-type recognition engine.

The public names are loaded lazily.  This is a correctness boundary, not just
an import-time optimization: requesting the independent Method2 backend must
not import Method1 (or its multiprocessing implementation), and vice versa.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .clustering_api import (
        CLUSTERING_API_SCHEMA_VERSION,
        CommandOwnership,
        CommandRange,
        GlobalLineTypeCluster,
        GlobalLineTypeMember,
        GroupLineTypeClusters,
        LineTypeClusteringResult,
        LocalLineTypeCluster,
        NonVectorSupport,
        PageClusteringError,
        PageClusteringFailure,
        PageClusteringOutcome,
        cluster_page_commands,
        cluster_pages_commands,
        command_ranges,
        project_line_type_clusters,
    )
    from .document_runner import (
        DocumentRunOptions,
        DocumentRunResult,
        parse_page_selection,
        run_document,
    )
    from .engine import (
        OUTPUT_KINDS,
        OutputKind,
        PageLineTypeRecognition,
        normalize_outputs,
        page_recognition_payload_hash,
        recognize_page,
    )
    from .fusion_contract import (
        FusedLineTypeEnvelope,
        fuse_line_type_results_for_display,
        validate_fused_line_type_envelope,
    )
    from .grouping import group_page, group_page_sequentially
    from .ir import (
        BoundsIR,
        GroupIR,
        GroupingIR,
        ImageOperationIR,
        PageIR,
        PathOperationIR,
        PathSegmentIR,
        TextCharacterIR,
        TextOperationIR,
    )
    from .ir_codec import (
        IRCodecError,
        page_ir_from_dict,
        page_ir_from_json,
        page_ir_to_dict,
    )
    from .method1 import recognize_method1_candidate
    from .method2.contract import LineTypeMethod2Envelope
    from .method2.recognizer import recognize_line_types_method2_page
    from .operation_index import PageOperationIndex
    from .pdf_adapter import (
        PDF_ADAPTER_VERSION,
        PdfAdapterError,
        page_ir_from_pdf_bytes,
        page_ir_from_pdf_path,
        page_ir_from_pymupdf_page,
        pdf_page_count_from_bytes,
    )
    from .results import LineTypeRecognitionResult
    from .source_content import (
        SOURCE_CONTENT_VERSION,
        SourceContentDocument,
        SourceContentError,
        SourceContentPageIR,
        source_content_page_from_pdf_bytes,
    )
    from .source_page_adapter import (
        SOURCE_ALIGNED_PAGE_IR_PRODUCER,
        SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION,
        SOURCE_PAGE_ADAPTER_VERSION,
        SourceAlignedPageIR,
        SourceAlignedPdfDocument,
        SourceAlignmentError,
        SourceAlignmentSummary,
        current_source_aligned_producer_version,
        source_aligned_page_ir_from_pdf_bytes,
        source_aligned_page_ir_from_pdf_path,
    )
    from .versions import (
        PAGE_IR_VERSION,
        PYTHON_ENGINE_VERSION,
        PYTHON_FUSION_ENGINE_VERSION,
        PYTHON_METHOD1_ENGINE_VERSION,
        PYTHON_METHOD2_ENGINE_VERSION,
        PYTHON_METHOD2_LOCAL_PROJECTION_VERSION,
    )


_PUBLIC_EXPORTS: dict[str, tuple[str, str]] = {
    "BoundsIR": (".ir", "BoundsIR"),
    "CLUSTERING_API_SCHEMA_VERSION": (
        ".clustering_api",
        "CLUSTERING_API_SCHEMA_VERSION",
    ),
    "CommandOwnership": (".clustering_api", "CommandOwnership"),
    "CommandRange": (".clustering_api", "CommandRange"),
    "DocumentRunOptions": (".document_runner", "DocumentRunOptions"),
    "DocumentRunResult": (".document_runner", "DocumentRunResult"),
    "FusedLineTypeEnvelope": (".fusion_contract", "FusedLineTypeEnvelope"),
    "GroupIR": (".ir", "GroupIR"),
    "GlobalLineTypeCluster": (".clustering_api", "GlobalLineTypeCluster"),
    "GlobalLineTypeMember": (".clustering_api", "GlobalLineTypeMember"),
    "GroupLineTypeClusters": (".clustering_api", "GroupLineTypeClusters"),
    "GroupingIR": (".ir", "GroupingIR"),
    "ImageOperationIR": (".ir", "ImageOperationIR"),
    "IRCodecError": (".ir_codec", "IRCodecError"),
    "LineTypeRecognitionResult": (".results", "LineTypeRecognitionResult"),
    "LineTypeClusteringResult": (".clustering_api", "LineTypeClusteringResult"),
    "LocalLineTypeCluster": (".clustering_api", "LocalLineTypeCluster"),
    "NonVectorSupport": (".clustering_api", "NonVectorSupport"),
    "PageClusteringError": (".clustering_api", "PageClusteringError"),
    "PageClusteringFailure": (".clustering_api", "PageClusteringFailure"),
    "PageClusteringOutcome": (".clustering_api", "PageClusteringOutcome"),
    "LineTypeMethod2Envelope": (".method2.contract", "LineTypeMethod2Envelope"),
    "OUTPUT_KINDS": (".engine", "OUTPUT_KINDS"),
    "OutputKind": (".engine", "OutputKind"),
    "PAGE_IR_VERSION": (".versions", "PAGE_IR_VERSION"),
    "PDF_ADAPTER_VERSION": (".pdf_adapter", "PDF_ADAPTER_VERSION"),
    "SOURCE_CONTENT_VERSION": (".source_content", "SOURCE_CONTENT_VERSION"),
    "SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION": (
        ".source_page_adapter",
        "SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION",
    ),
    "SOURCE_PAGE_ADAPTER_VERSION": (
        ".source_page_adapter",
        "SOURCE_PAGE_ADAPTER_VERSION",
    ),
    "SOURCE_ALIGNED_PAGE_IR_PRODUCER": (
        ".source_page_adapter",
        "SOURCE_ALIGNED_PAGE_IR_PRODUCER",
    ),
    "PYTHON_ENGINE_VERSION": (".versions", "PYTHON_ENGINE_VERSION"),
    "PYTHON_FUSION_ENGINE_VERSION": (
        ".versions",
        "PYTHON_FUSION_ENGINE_VERSION",
    ),
    "PYTHON_METHOD1_ENGINE_VERSION": (
        ".versions",
        "PYTHON_METHOD1_ENGINE_VERSION",
    ),
    "PYTHON_METHOD2_ENGINE_VERSION": (
        ".versions",
        "PYTHON_METHOD2_ENGINE_VERSION",
    ),
    "PYTHON_METHOD2_LOCAL_PROJECTION_VERSION": (
        ".versions",
        "PYTHON_METHOD2_LOCAL_PROJECTION_VERSION",
    ),
    "PageIR": (".ir", "PageIR"),
    "PageLineTypeRecognition": (".engine", "PageLineTypeRecognition"),
    "PageOperationIndex": (".operation_index", "PageOperationIndex"),
    "PathOperationIR": (".ir", "PathOperationIR"),
    "PathSegmentIR": (".ir", "PathSegmentIR"),
    "PdfAdapterError": (".pdf_adapter", "PdfAdapterError"),
    "SourceAlignedPageIR": (".source_page_adapter", "SourceAlignedPageIR"),
    "SourceAlignedPdfDocument": (
        ".source_page_adapter",
        "SourceAlignedPdfDocument",
    ),
    "SourceAlignmentError": (".source_page_adapter", "SourceAlignmentError"),
    "SourceAlignmentSummary": (
        ".source_page_adapter",
        "SourceAlignmentSummary",
    ),
    "SourceContentError": (".source_content", "SourceContentError"),
    "SourceContentDocument": (".source_content", "SourceContentDocument"),
    "SourceContentPageIR": (".source_content", "SourceContentPageIR"),
    "TextCharacterIR": (".ir", "TextCharacterIR"),
    "TextOperationIR": (".ir", "TextOperationIR"),
    "group_page": (".grouping", "group_page"),
    "group_page_sequentially": (".grouping", "group_page_sequentially"),
    "cluster_page_commands": (".clustering_api", "cluster_page_commands"),
    "cluster_pages_commands": (".clustering_api", "cluster_pages_commands"),
    "command_ranges": (".clustering_api", "command_ranges"),
    "current_source_aligned_producer_version": (
        ".source_page_adapter",
        "current_source_aligned_producer_version",
    ),
    "fuse_line_type_results_for_display": (
        ".fusion_contract",
        "fuse_line_type_results_for_display",
    ),
    "normalize_outputs": (".engine", "normalize_outputs"),
    "page_ir_from_pdf_bytes": (".pdf_adapter", "page_ir_from_pdf_bytes"),
    "page_ir_from_dict": (".ir_codec", "page_ir_from_dict"),
    "page_ir_from_json": (".ir_codec", "page_ir_from_json"),
    "page_ir_from_pdf_path": (".pdf_adapter", "page_ir_from_pdf_path"),
    "page_ir_from_pymupdf_page": (".pdf_adapter", "page_ir_from_pymupdf_page"),
    "page_ir_to_dict": (".ir_codec", "page_ir_to_dict"),
    "page_recognition_payload_hash": (".engine", "page_recognition_payload_hash"),
    "parse_page_selection": (".document_runner", "parse_page_selection"),
    "pdf_page_count_from_bytes": (".pdf_adapter", "pdf_page_count_from_bytes"),
    "recognize_method1_candidate": (".method1", "recognize_method1_candidate"),
    "recognize_line_types_method2_page": (
        ".method2.recognizer",
        "recognize_line_types_method2_page",
    ),
    "recognize_page": (".engine", "recognize_page"),
    "project_line_type_clusters": (
        ".clustering_api",
        "project_line_type_clusters",
    ),
    "source_aligned_page_ir_from_pdf_bytes": (
        ".source_page_adapter",
        "source_aligned_page_ir_from_pdf_bytes",
    ),
    "source_aligned_page_ir_from_pdf_path": (
        ".source_page_adapter",
        "source_aligned_page_ir_from_pdf_path",
    ),
    "source_content_page_from_pdf_bytes": (
        ".source_content",
        "source_content_page_from_pdf_bytes",
    ),
    "run_document": (".document_runner", "run_document"),
    "validate_fused_line_type_envelope": (
        ".fusion_contract",
        "validate_fused_line_type_envelope",
    ),
}

__all__ = sorted(_PUBLIC_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _PUBLIC_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
