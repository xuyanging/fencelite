"""Independent, headless Method2 r46 recognizer and input fingerprint.

The facade composes only vector/native text-pattern stages.  It does not know
about Method1, display fusion, browser state, HTTP or cache storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Protocol, Sequence

from ..ir import (
    BoundsIR,
    GroupingIR,
    ImageOperationIR,
    PageIR,
    PathOperationIR,
    PathSegmentIR,
    TextCharacterIR,
    TextOperationIR,
)
from ..operation_index import PageOperationIndex
from ..results import LineTypeRecognitionResult
from .contract import (
    LINE_TYPE_METHOD2_CONFIG_HASH,
    LINE_TYPE_METHOD2_FEATURES,
    METHOD2_ENGINE_VERSION,
    METHOD2_LOCAL_PROJECTION_VERSION,
    METHOD2_RESULT_SCHEMA_VERSION,
    METHOD2_TARGET_SPEC_VERSION,
    LineTypeFingerprintWriter,
    LineTypeMethod2Audit,
    LineTypeMethod2Envelope,
    VectorTextFamilyAuditPayload,
    validate_line_type_method2_envelope,
)
from .vector_text import VectorTextRegion


class VectorTextDetector(Protocol):
    def __call__(
        self,
        page: PageIR,
        grouping: GroupingIR,
        operation_index: PageOperationIndex,
    ) -> Sequence[VectorTextRegion]: ...


class FamilyAuditLike(Protocol):
    def to_dict(self) -> dict[str, object]: ...


class TextFamilyRecognitionLike(Protocol):
    result: LineTypeRecognitionResult
    audit: FamilyAuditLike


class TextFamilyRecognizer(Protocol):
    def __call__(
        self,
        page: PageIR,
        grouping: GroupingIR,
        regions: Sequence[VectorTextRegion],
    ) -> TextFamilyRecognitionLike: ...


ProgressCallback = Callable[[int, int, int], None]


@dataclass(frozen=True, slots=True)
class LineTypeMethod2Result:
    result: LineTypeRecognitionResult
    audit: FamilyAuditLike


def _write_bounds(writer: LineTypeFingerprintWriter, bounds: BoundsIR) -> None:
    writer.begin("bounds").number(bounds.min_x).number(bounds.min_y).number(
        bounds.max_x
    ).number(bounds.max_y).end()


def _write_optional_number(
    writer: LineTypeFingerprintWriter,
    value: float | None,
) -> None:
    if value is None:
        writer.null()
    else:
        writer.number(value)


def _write_color(
    writer: LineTypeFingerprintWriter,
    color: tuple[float, ...] | None,
) -> None:
    writer.begin("color")
    if color is None:
        writer.null()
    else:
        writer.number(len(color))
        for channel in color:
            writer.number(channel)
    writer.end()


def _write_segment(
    writer: LineTypeFingerprintWriter,
    segment: PathSegmentIR,
) -> None:
    if segment.kind == "move":
        assert segment.end is not None
        writer.string("M").number(segment.end[0]).number(segment.end[1])
    elif segment.kind == "line":
        assert segment.end is not None
        writer.string("L").number(segment.end[0]).number(segment.end[1])
    elif segment.kind == "curve":
        assert segment.control_1 is not None
        assert segment.control_2 is not None
        assert segment.end is not None
        writer.string("C").number(segment.control_1[0]).number(
            segment.control_1[1]
        ).number(segment.control_2[0]).number(segment.control_2[1]).number(
            segment.end[0]
        ).number(segment.end[1])
    else:
        writer.string("Z")


def _write_path(
    writer: LineTypeFingerprintWriter,
    operation: PathOperationIR,
) -> None:
    _write_bounds(writer, operation.bounds)
    writer.begin("segments").number(len(operation.segments))
    for segment in operation.segments:
        _write_segment(writer, segment)
    writer.end().boolean(operation.fill).boolean(operation.stroke).boolean(
        operation.even_odd
    )
    _write_color(writer, operation.fill_color)
    _write_color(writer, operation.stroke_color)
    writer.number(operation.fill_opacity).number(operation.stroke_opacity).number(
        operation.line_width
    ).boolean(operation.hairline).begin("line-cap").number(
        len(operation.line_cap)
    )
    for cap in operation.line_cap:
        writer.number(cap)
    writer.end().number(operation.line_join).begin("dash").number(
        len(operation.dash_array)
    )
    for value in operation.dash_array:
        writer.number(value)
    writer.end().number(operation.dash_phase).string(operation.blend_mode).boolean(
        operation.close_path
    ).string(operation.layer).number(operation.nesting_level)


def _write_character(
    writer: LineTypeFingerprintWriter,
    character: TextCharacterIR,
) -> None:
    writer.begin("character").number(character.codepoint).number(
        character.glyph_id
    ).number(character.origin[0]).number(character.origin[1])
    _write_bounds(writer, character.bounds)
    writer.end()


def _write_text(
    writer: LineTypeFingerprintWriter,
    operation: TextOperationIR,
) -> None:
    _write_bounds(writer, operation.bounds)
    writer.string(operation.literal_text).number(operation.render_mode)
    has_source = operation.source_matrix is not None
    writer.begin("authored-text-state").boolean(has_source)
    if has_source:
        assert operation.source_font_name is not None
        assert operation.source_font_size is not None
        assert operation.source_matrix is not None
        assert operation.source_glyph_advance is not None
        assert operation.source_horizontal_scale is not None
        assert operation.source_rise is not None
        assert operation.source_unclipped_bounds is not None
        assert operation.source_fill_color is not None
        assert operation.source_stroke_color is not None
        assert operation.source_fill_opacity is not None
        assert operation.source_stroke_opacity is not None
        assert operation.source_line_width is not None
        assert operation.source_blend_mode is not None
        writer.string(operation.source_font_name).number(operation.source_font_size)
        writer.begin("matrix")
        for value in operation.source_matrix:
            writer.number(value)
        writer.end().number(operation.source_glyph_advance).number(
            operation.source_horizontal_scale
        ).number(operation.source_rise)
        _write_bounds(writer, operation.source_unclipped_bounds)
        _write_color(writer, operation.source_fill_color)
        _write_color(writer, operation.source_stroke_color)
        writer.number(operation.source_fill_opacity).number(
            operation.source_stroke_opacity
        ).number(operation.source_line_width).string(operation.source_blend_mode)
    writer.end().begin("display-trace").string(operation.font_name).number(
        operation.font_size
    ).number(operation.direction[0]).number(operation.direction[1])
    _write_color(writer, operation.color)
    writer.number(operation.opacity)
    _write_optional_number(writer, operation.line_width)
    writer.number(operation.writing_mode).number(operation.flags).number(
        operation.bidi_level
    ).number(operation.bidi_direction).number(operation.span_index).string(
        operation.layer
    ).begin("characters").number(len(operation.characters))
    for character in operation.characters:
        _write_character(writer, character)
    writer.end().end()


def _write_image(
    writer: LineTypeFingerprintWriter,
    operation: ImageOperationIR,
) -> None:
    # Image payloads do not participate in text-pattern identity, but their
    # bounds and dense position can change Group sequence context.
    _write_bounds(writer, operation.bounds)
    writer.string(operation.target_kind).number(operation.alpha).string(
        operation.blend_mode
    )


def _write_operation(
    writer: LineTypeFingerprintWriter,
    dense_index: int,
    operation: PathOperationIR | TextOperationIR | ImageOperationIR,
) -> None:
    writer.begin("operation").number(dense_index).string(operation.kind).number(
        operation.paint_order
    ).number(operation.ordinal)
    if isinstance(operation, PathOperationIR):
        _write_path(writer, operation)
    elif isinstance(operation, TextOperationIR):
        _write_text(writer, operation)
    else:
        _write_image(writer, operation)
    writer.end()


def line_type_method2_input_hash(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex | None = None,
) -> str:
    """Fingerprint every canonical input consumed by Python Method2.

    Dense assignments are written in page order.  Operation ids and source
    filenames are deliberately excluded: consistently renaming either does not
    alter recognition, while geometry, native text, style or Group placement
    always changes the digest.
    """

    index = operation_index or PageOperationIndex.build(page, grouping)
    if grouping.page_fingerprint != page.fingerprint:
        raise ValueError("GroupingIR does not belong to the supplied PageIR")
    if index.operations != page.operations:
        raise ValueError("operation_index does not belong to the supplied PageIR")

    writer = LineTypeFingerprintWriter().begin("method2-page-ir-input-v2-source-text-image")
    writer.string(page.page_ir_version)
    _write_bounds(writer, page.page_bounds)
    writer.begin("operations").number(len(page.operations))
    for dense_index, operation in enumerate(page.operations):
        _write_operation(writer, dense_index, operation)
    writer.end().begin("grouping").string(grouping.grouping_version)
    writer.begin("dense-assignments").number(len(page.operations))
    for dense_index in range(len(page.operations)):
        writer.string(index.group_id(dense_index))
    writer.end().begin("groups").number(len(grouping.groups))
    for group in grouping.groups:
        dense_indices = index.group_indices(group.group_id)
        writer.begin("group").string(group.group_id).number(len(dense_indices))
        for dense_index in dense_indices:
            writer.number(dense_index)
        _write_bounds(writer, group.bounds)
        writer.number(group.first_paint_order).number(group.last_paint_order).begin(
            "split-reasons"
        ).number(len(group.split_reasons))
        for reason in group.split_reasons:
            writer.string(reason)
        writer.end().end()
    return writer.end().end().end().digest()


def _default_vector_text_detector(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
) -> Sequence[VectorTextRegion]:
    from .vector_text import detect_vector_text_regions_v2_sequential

    return detect_vector_text_regions_v2_sequential(page, grouping, operation_index)


def _default_text_family_recognizer(
    page: PageIR,
    grouping: GroupingIR,
    regions: Sequence[VectorTextRegion],
    *,
    worker_count: int = 1,
) -> TextFamilyRecognitionLike:
    from .text_family import recognize_repeated_text_pattern_families

    return recognize_repeated_text_pattern_families(
        page,
        grouping,
        regions,
        worker_count=worker_count,
    )


def _validate_worker_count(worker_count: int) -> int:
    if (
        isinstance(worker_count, bool)
        or not isinstance(worker_count, int)
        or worker_count < 1
    ):
        raise ValueError("worker_count must be a positive integer")
    return worker_count


def recognize_line_types_method2(
    page: PageIR,
    grouping: GroupingIR,
    *,
    worker_count: int = 1,
    vector_text_detector: VectorTextDetector = _default_vector_text_detector,
    text_family_recognizer: TextFamilyRecognizer = _default_text_family_recognizer,
) -> LineTypeMethod2Result:
    """Run the two Method2-only stages without packaging or persistence."""

    operation_index = PageOperationIndex.build(page, grouping)
    return _recognize_line_types_method2_with_index(
        page,
        grouping,
        operation_index,
        vector_text_detector,
        text_family_recognizer,
        _validate_worker_count(worker_count),
    )


def _recognize_line_types_method2_with_index(
    page: PageIR,
    grouping: GroupingIR,
    operation_index: PageOperationIndex,
    vector_text_detector: VectorTextDetector,
    text_family_recognizer: TextFamilyRecognizer,
    worker_count: int,
) -> LineTypeMethod2Result:
    regions = tuple(vector_text_detector(page, grouping, operation_index))
    if text_family_recognizer is _default_text_family_recognizer:
        recognized = _default_text_family_recognizer(
            page,
            grouping,
            regions,
            worker_count=worker_count,
        )
    else:
        # Custom recognizers keep the established three-argument injection
        # contract.  ``worker_count`` is an execution hint for the built-in
        # implementation only and never changes the result contract.
        recognized = text_family_recognizer(page, grouping, regions)
    if not isinstance(recognized.result, LineTypeRecognitionResult):
        raise TypeError("text-family recognizer returned an invalid result")
    VectorTextFamilyAuditPayload.from_value(recognized.audit)
    return LineTypeMethod2Result(recognized.result, recognized.audit)


def _js_round(value: float, digits: int) -> float:
    scale = 10 ** digits
    result = math.floor(value * scale + 0.5) / scale
    return 0.0 if result == 0 else result


def recognize_line_types_method2_page(
    page: PageIR,
    grouping: GroupingIR,
    *,
    page_identity: str,
    worker_count: int = 1,
    on_progress: ProgressCallback | None = None,
    vector_text_detector: VectorTextDetector = _default_vector_text_detector,
    text_family_recognizer: TextFamilyRecognizer = _default_text_family_recognizer,
    clock: Callable[[], float] = time.perf_counter,
) -> LineTypeMethod2Envelope:
    """Run and package Method2 without consulting another recognizer."""

    if not isinstance(page_identity, str) or not page_identity:
        raise ValueError("page_identity must be a non-empty string")
    started_at = clock()
    if on_progress is not None:
        on_progress(0, 1, 1)
    operation_index = PageOperationIndex.build(page, grouping)
    input_hash = line_type_method2_input_hash(page, grouping, operation_index)
    method2 = _recognize_line_types_method2_with_index(
        page,
        grouping,
        operation_index,
        vector_text_detector,
        text_family_recognizer,
        _validate_worker_count(worker_count),
    )
    if on_progress is not None:
        on_progress(1, 1, 0)
    elapsed_ms = _js_round((clock() - started_at) * 1000, 2)
    family_audit = VectorTextFamilyAuditPayload.from_value(method2.audit)
    replay_key = f"{page_identity}:{LINE_TYPE_METHOD2_CONFIG_HASH}:{input_hash}"
    envelope = LineTypeMethod2Envelope(
        schema_version=METHOD2_RESULT_SCHEMA_VERSION,
        engine_version=METHOD2_ENGINE_VERSION,
        target_spec_version=METHOD2_TARGET_SPEC_VERSION,
        local_projection_version=METHOD2_LOCAL_PROJECTION_VERSION,
        config_hash=LINE_TYPE_METHOD2_CONFIG_HASH,
        features=LINE_TYPE_METHOD2_FEATURES,
        page_identity=page_identity,
        result=method2.result,
        audit=LineTypeMethod2Audit(
            input_hash=input_hash,
            deterministic_replay_key=replay_key,
            elapsed_ms=elapsed_ms,
            repeated_vector_text_family_clustering=family_audit,
        ),
    )
    return validate_line_type_method2_envelope(envelope)


__all__ = [
    "FamilyAuditLike",
    "LineTypeMethod2Result",
    "ProgressCallback",
    "TextFamilyRecognizer",
    "VectorTextDetector",
    "line_type_method2_input_hash",
    "recognize_line_types_method2",
    "recognize_line_types_method2_page",
]
