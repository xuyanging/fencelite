"""Fail-closed alignment of source content events with visible PageIR geometry.

``source_content`` is authoritative for authored paint order, path topology
and style, text-show geometry, image identity and structural provenance.
``pdf_adapter`` supplies a glyph/style trace plus independently decoded image
occurrences; its path list is retained only as a diagnostic count.  This module
joins the two views by explicit source events, never by PyMuPDF ``seqno`` or
display-path ordinal.  The output is an ordinary ``PageIR`` whose dense
operation position remains the ownership identity and whose ``paint_order`` is
the recovered source paint order.

Alignment is deliberately strict: count drift, unsupported paints, ambiguous
font-code consumption or mixed trace styles raise ``SourceAlignmentError``.
Returning a plausible but mis-owned page would be worse than retaining the
original PyMuPDF PageIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Mapping
import unicodedata

from .ir import (
    BoundsIR,
    ImageOperationIR,
    PageIR,
    PathOperationIR,
    TextCharacterIR,
    TextOperationIR,
)
from .pdf_adapter import PdfAdapterError, page_ir_from_pymupdf_page
from .runtime import (
    assert_supported_pymupdf_runtime,
    assert_supported_pypdf_runtime,
)
from .source_content import (
    SOURCE_CONTENT_VERSION,
    SourceContentDocument,
    SourceContentPageIR,
    SourceImagePaintEventIR,
    SourceInlineImageSkipEventIR,
    SourcePathPaintEventIR,
    SourceTextShowEventIR,
)


SOURCE_PAGE_ADAPTER_VERSION = (
    "source-aligned-page-ir-r11-coalesced-render2-runs-2026-08-29"
)
SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION = 4
SOURCE_ALIGNED_PAGE_IR_PRODUCER = (
    f"PyMuPDF+pypdf/{SOURCE_CONTENT_VERSION}/{SOURCE_PAGE_ADAPTER_VERSION}"
)


class SourceAlignmentError(RuntimeError):
    """Raised when source provenance cannot be joined without guessing."""


@lru_cache(maxsize=1)
def _pymupdf() -> object:
    assert_supported_pymupdf_runtime()
    try:
        import pymupdf
    except ImportError as error:  # pragma: no cover - guarded above.
        raise SourceAlignmentError(
            "PyMuPDF is required for source-aligned PageIR"
        ) from error
    return pymupdf


def current_source_aligned_producer_version() -> str:
    """Return the exact two-parser identity used by durable resume checks."""

    pymupdf = _pymupdf()
    pypdf = assert_supported_pypdf_runtime()
    pymupdf_version = str(
        getattr(pymupdf, "VersionBind", "")
        or getattr(pymupdf, "__version__", "")
    )
    return f"{pymupdf_version}+{pypdf.module_version}"


@dataclass(frozen=True, slots=True)
class SourcePageAlignmentAudit:
    source_paint_count: int
    aligned_operation_count: int
    source_path_count: int
    display_path_cross_audit_count: int
    source_atom_count: int
    source_visible_text_show_count: int
    source_visible_text_glyph_count: int
    display_visible_text_span_count: int
    display_visible_text_glyph_count: int
    source_hidden_text_show_count: int
    source_image_count: int
    display_image_count: int
    exact_source_path_construction: bool
    exact_text_event_join: bool
    exact_source_image_join: bool
    mixed_trace_style_event_count: int = 0
    source_annotation_appearance_count: int = 0
    collapsed_fill_stroke_trace_glyph_count: int = 0
    consumed_hidden_trace_glyph_count: int = 0
    source_inline_image_skip_count: int = 0
    display_inline_image_count: int = 0
    source_only_image_count: int = 0


@dataclass(frozen=True, slots=True)
class SourceAlignmentSummary:
    """Compact durable proof that a page used exact source alignment.

    The in-memory audit deliberately retains separate source/display counters.
    Persisted pages retain those mechanically reconcilable counters plus the
    exact parser version. Construction fails unless all fail-closed alignment
    guarantees hold, so the summary cannot make an ambiguous join look
    successful.
    """

    schema_version: int
    parser_version: str
    operation_count: int
    path_count: int
    display_path_cross_audit_count: int
    path_atom_count: int
    visible_text_show_count: int
    visible_text_glyph_count: int
    display_trace_span_count: int
    display_trace_glyph_count: int
    collapsed_fill_stroke_trace_glyph_count: int
    consumed_hidden_trace_glyph_count: int
    hidden_text_show_count: int
    image_count: int
    display_image_count: int
    inline_image_skip_count: int
    source_only_image_count: int
    annotation_appearance_count: int

    @classmethod
    def from_alignment_audit(
        cls,
        audit: SourcePageAlignmentAudit,
        *,
        parser_version: str,
    ) -> "SourceAlignmentSummary":
        if not isinstance(parser_version, str) or not parser_version:
            raise ValueError("source alignment parser_version must be non-empty")
        if not (
            audit.exact_source_path_construction
            and audit.exact_text_event_join
            and audit.exact_source_image_join
            and audit.mixed_trace_style_event_count == 0
            and audit.source_paint_count == audit.aligned_operation_count
            and audit.display_visible_text_glyph_count
            == (
                audit.source_visible_text_glyph_count
                + audit.collapsed_fill_stroke_trace_glyph_count
                + audit.consumed_hidden_trace_glyph_count
            )
            and audit.source_inline_image_skip_count
            == audit.display_inline_image_count
            and audit.display_image_count
            == (
                audit.source_image_count
                - audit.source_only_image_count
                + audit.display_inline_image_count
            )
        ):
            raise ValueError("source alignment audit is not exact")
        summary = cls(
            schema_version=SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION,
            parser_version=parser_version,
            operation_count=audit.aligned_operation_count,
            path_count=audit.source_path_count,
            display_path_cross_audit_count=audit.display_path_cross_audit_count,
            path_atom_count=audit.source_atom_count,
            visible_text_show_count=audit.source_visible_text_show_count,
            visible_text_glyph_count=audit.source_visible_text_glyph_count,
            display_trace_span_count=audit.display_visible_text_span_count,
            display_trace_glyph_count=audit.display_visible_text_glyph_count,
            collapsed_fill_stroke_trace_glyph_count=(
                audit.collapsed_fill_stroke_trace_glyph_count
            ),
            consumed_hidden_trace_glyph_count=(
                audit.consumed_hidden_trace_glyph_count
            ),
            hidden_text_show_count=audit.source_hidden_text_show_count,
            image_count=audit.source_image_count,
            display_image_count=audit.display_image_count,
            inline_image_skip_count=audit.source_inline_image_skip_count,
            source_only_image_count=audit.source_only_image_count,
            annotation_appearance_count=(
                audit.source_annotation_appearance_count
            ),
        )
        summary._validate_counts()
        return summary

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        expected_parser_version: str,
        expected_operation_count: int,
    ) -> "SourceAlignmentSummary":
        expected_keys = {
            "schema_version",
            "parser_version",
            "operation_count",
            "path_count",
            "display_path_cross_audit_count",
            "path_atom_count",
            "visible_text_show_count",
            "visible_text_glyph_count",
            "display_trace_span_count",
            "display_trace_glyph_count",
            "collapsed_fill_stroke_trace_glyph_count",
            "consumed_hidden_trace_glyph_count",
            "hidden_text_show_count",
            "image_count",
            "display_image_count",
            "inline_image_skip_count",
            "source_only_image_count",
            "annotation_appearance_count",
        }
        if set(value) != expected_keys:
            raise ValueError("source alignment audit fields do not match the schema")

        def count(name: str) -> int:
            item = value.get(name)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise ValueError(f"source alignment {name} must be non-negative")
            return item

        parser_version = value.get("parser_version")
        if parser_version != expected_parser_version:
            raise ValueError("source alignment parser version does not match")
        if value.get("schema_version") != SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION:
            raise ValueError("source alignment audit schema version does not match")
        summary = cls(
            schema_version=SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION,
            parser_version=expected_parser_version,
            operation_count=count("operation_count"),
            path_count=count("path_count"),
            display_path_cross_audit_count=count(
                "display_path_cross_audit_count"
            ),
            path_atom_count=count("path_atom_count"),
            visible_text_show_count=count("visible_text_show_count"),
            visible_text_glyph_count=count("visible_text_glyph_count"),
            display_trace_span_count=count("display_trace_span_count"),
            display_trace_glyph_count=count("display_trace_glyph_count"),
            collapsed_fill_stroke_trace_glyph_count=count(
                "collapsed_fill_stroke_trace_glyph_count"
            ),
            consumed_hidden_trace_glyph_count=count(
                "consumed_hidden_trace_glyph_count"
            ),
            hidden_text_show_count=count("hidden_text_show_count"),
            image_count=count("image_count"),
            display_image_count=count("display_image_count"),
            inline_image_skip_count=count("inline_image_skip_count"),
            source_only_image_count=count("source_only_image_count"),
            annotation_appearance_count=count("annotation_appearance_count"),
        )
        if summary.operation_count != expected_operation_count:
            raise ValueError("source alignment operation count does not match the page")
        summary._validate_counts()
        return summary

    def _validate_counts(self) -> None:
        counts = (
            self.operation_count,
            self.path_count,
            self.display_path_cross_audit_count,
            self.path_atom_count,
            self.visible_text_show_count,
            self.visible_text_glyph_count,
            self.display_trace_span_count,
            self.display_trace_glyph_count,
            self.collapsed_fill_stroke_trace_glyph_count,
            self.consumed_hidden_trace_glyph_count,
            self.hidden_text_show_count,
            self.image_count,
            self.display_image_count,
            self.inline_image_skip_count,
            self.source_only_image_count,
            self.annotation_appearance_count,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        ):
            raise ValueError("source alignment counts must be non-negative integers")
        if self.operation_count != (
            self.path_count + self.visible_text_show_count + self.image_count
        ):
            raise ValueError("source alignment paint counts do not form the page")
        if self.visible_text_show_count > self.visible_text_glyph_count:
            raise ValueError("source text shows exceed aligned glyphs")
        if self.display_trace_span_count > self.display_trace_glyph_count:
            raise ValueError("display trace spans exceed display trace glyphs")
        if self.display_trace_glyph_count != (
            self.visible_text_glyph_count
            + self.collapsed_fill_stroke_trace_glyph_count
            + self.consumed_hidden_trace_glyph_count
        ):
            raise ValueError("display trace glyph counts do not reconcile")
        if self.source_only_image_count > self.image_count:
            raise ValueError("source-only images exceed authored image paints")
        if self.display_image_count != (
            self.image_count - self.source_only_image_count + self.inline_image_skip_count
        ):
            raise ValueError("display/source image counts do not reconcile")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "operation_count": self.operation_count,
            "path_count": self.path_count,
            "display_path_cross_audit_count": self.display_path_cross_audit_count,
            "path_atom_count": self.path_atom_count,
            "visible_text_show_count": self.visible_text_show_count,
            "visible_text_glyph_count": self.visible_text_glyph_count,
            "display_trace_span_count": self.display_trace_span_count,
            "display_trace_glyph_count": self.display_trace_glyph_count,
            "collapsed_fill_stroke_trace_glyph_count": (
                self.collapsed_fill_stroke_trace_glyph_count
            ),
            "consumed_hidden_trace_glyph_count": self.consumed_hidden_trace_glyph_count,
            "hidden_text_show_count": self.hidden_text_show_count,
            "image_count": self.image_count,
            "display_image_count": self.display_image_count,
            "inline_image_skip_count": self.inline_image_skip_count,
            "source_only_image_count": self.source_only_image_count,
            "annotation_appearance_count": self.annotation_appearance_count,
        }


@dataclass(frozen=True, slots=True)
class SourceAlignedPageIR:
    page: PageIR
    source: SourceContentPageIR
    audit: SourcePageAlignmentAudit


@dataclass(frozen=True, slots=True)
class _TraceCharacter:
    character: TextCharacterIR
    span: TextOperationIR


def _near(left: float, right: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def _same_bounds(left: BoundsIR, right: BoundsIR) -> bool:
    return all(
        _near(left_value, right_value)
        for left_value, right_value in zip(
            (left.min_x, left.min_y, left.max_x, left.max_y),
            (right.min_x, right.min_y, right.max_x, right.max_y),
        )
    )


def _same_display_image_bounds(left: BoundsIR, right: BoundsIR) -> bool:
    # MuPDF stores image bbox coordinates as float32, while pypdf retains the
    # authored decimal numbers.  One thousandth of a PDF point is far below a
    # device pixel and only reconciles that mechanical representation drift.
    return all(
        math.isclose(left_value, right_value, rel_tol=0.0, abs_tol=1e-3)
        for left_value, right_value in zip(
            (left.min_x, left.min_y, left.max_x, left.max_y),
            (right.min_x, right.min_y, right.max_x, right.max_y),
        )
    )


def _text_style_key(span: TextOperationIR) -> tuple[object, ...]:
    return (
        span.font_name,
        round(span.font_size, 7),
        tuple(round(value, 7) for value in span.direction),
        span.render_mode,
        span.color,
        round(span.opacity, 7),
        None if span.line_width is None else round(span.line_width, 7),
        span.writing_mode,
        span.flags,
        span.bidi_level,
        span.bidi_direction,
        span.layer,
    )


def _literal(characters: tuple[TextCharacterIR, ...]) -> str:
    result: list[str] = []
    for character in characters:
        codepoint = character.codepoint
        if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
            result.append("\N{REPLACEMENT CHARACTER}")
        else:
            result.append(chr(codepoint))
    return "".join(result)


def _trace_text_matches_source(
    event: SourceTextShowEventIR,
    records: tuple[_TraceCharacter, ...],
) -> bool:
    trace_text = _literal(tuple(record.character for record in records))
    source_text = event.decoded_text
    if trace_text == source_text:
        return True
    if unicodedata.normalize("NFKC", trace_text) == unicodedata.normalize(
        "NFKC", source_text
    ):
        return True
    # WinAnsi control code 0x09 is routinely mapped to the font's space glyph
    # by MuPDF while the frozen source decoder preserves the authored tab.
    if source_text.replace("\t", " ") == trace_text:
        return True
    if len(source_text) == len(trace_text) and all(
        left == right
        or (left == "\t" and right == " ")
        or left in {"\N{WHITE SQUARE}", "\N{REPLACEMENT CHARACTER}"}
        or right == "\N{REPLACEMENT CHARACTER}"
        for left, right in zip(source_text, trace_text)
    ):
        return True
    return False


def _same_trace_glyph(left: _TraceCharacter, right: _TraceCharacter) -> bool:
    return (
        left.character.codepoint == right.character.codepoint
        and left.character.glyph_id == right.character.glyph_id
        and left.character.origin == right.character.origin
        and left.character.bounds == right.character.bounds
    )


def _source_text_unclipped_bounds(event: SourceTextShowEventIR) -> BoundsIR:
    """Rebuild the exact un-clipped frame consumed by frozen Method2.

    The browser stores absolute, floored ``fontSize`` / ``glyphAdvance`` on
    TextDrawOp and Method2 reconstructs from those stored values.  This can
    intentionally differ from the raw pre-normalization paint envelope for a
    malformed negative size, so retain both on their respective source IRs.
    """

    font_size = max(0.1, abs(event.font_size))
    glyph_advance = max(0.001, abs(event.glyph_advance))
    advance = glyph_advance * event.horizontal_scale
    lower_y = event.rise - font_size * 0.25
    upper_y = event.rise + font_size * 0.88
    a, b, c, d, e, f = event.text_matrix
    points = tuple(
        (a * x + c * y + e, b * x + d * y + f)
        for x, y in (
            (0.0, lower_y),
            (advance, lower_y),
            (advance, upper_y),
            (0.0, upper_y),
        )
    )
    return BoundsIR(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _aligned_text_operations(
    base_texts: tuple[TextOperationIR, ...],
    source_events: tuple[SourceTextShowEventIR, ...],
) -> tuple[tuple[TextOperationIR, ...], int, int, int, int, int]:
    """Join authored shows to the MuPDF glyph trace without changing identity.

    MuPDF exposes render mode 2 as geometrically identical mode 0/1 traces.
    Usually one source show becomes one adjacent fill/stroke pair.  It may also
    coalesce consecutive shows in one text object into a whole fill run followed
    by the matching whole stroke run.  Both forms are consumed only after every
    source slice and every duplicate glyph have been joined exactly.  Fully
    clipped shows and render mode 3 are also consumed from the trace but
    deliberately do not emit PageIR operations.  Every other unexplained
    cardinality or style difference fails closed.
    """

    trace_spans = tuple(span for span in base_texts if span.characters)
    trace = tuple(
        _TraceCharacter(character, span)
        for span in trace_spans
        for character in span.characters
    )

    aligned: list[TextOperationIR] = []
    cursor = 0
    mixed_style_count = 0
    collapsed_duplicate_glyph_count = 0
    consumed_hidden_glyph_count = 0
    event_index = 0
    while event_index < len(source_events):
        event = source_events[event_index]
        if event.glyph_count <= 0:
            raise SourceAlignmentError(
                f"source text show {event.text_show_ordinal} has no joinable glyphs"
            )
        end = cursor + event.glyph_count
        records = trace[cursor:end]
        if len(records) != event.glyph_count:
            raise SourceAlignmentError(
                "source text show has no complete PyMuPDF trace slice: "
                f"show {event.text_show_ordinal}, source {event.glyph_count}, "
                f"remaining {len(trace) - cursor}"
            )
        if not _trace_text_matches_source(event, records):
            raise SourceAlignmentError(
                "source text content does not match its PyMuPDF trace slice: "
                f"show {event.text_show_ordinal}, source {event.decoded_text!r}, "
                f"trace {_literal(tuple(record.character for record in records))!r}"
            )
        cursor = end
        joined_events = [(event, records)]
        next_event_index = event_index + 1

        # MuPDF traces PDF render mode 2 as a fill pass followed by a stroke
        # pass, while the authored/frozen Scene retains one operation.  A
        # display span may coalesce several consecutive source shows: in that
        # case its ordering is all fills, then all strokes, rather than an
        # adjacent pair per show.  Extend the fill run only across source shows
        # whose exact text slices are present as mode-0 trace glyphs.  The
        # equally sized mode-1 run below must then duplicate every glyph exactly
        # or alignment still fails closed.
        if event.render_mode == 2 and all(
            record.span.render_mode == 0 for record in records
        ):
            while next_event_index < len(source_events):
                candidate = source_events[next_event_index]
                if candidate.render_mode != 2 or candidate.glyph_count <= 0:
                    break
                candidate_end = cursor + candidate.glyph_count
                candidate_records = trace[cursor:candidate_end]
                if (
                    len(candidate_records) != candidate.glyph_count
                    or not all(
                        record.span.render_mode == 0
                        for record in candidate_records
                    )
                    or not _trace_text_matches_source(candidate, candidate_records)
                ):
                    break
                joined_events.append((candidate, candidate_records))
                cursor = candidate_end
                next_event_index += 1

            fill_records = tuple(
                record
                for _joined_event, joined_records in joined_events
                for record in joined_records
            )
            duplicate_end = cursor + len(fill_records)
            duplicates = trace[cursor:duplicate_end]
            if (
                len(duplicates) != len(fill_records)
                or not all(record.span.render_mode == 1 for record in duplicates)
                or not all(
                    _same_trace_glyph(left, right)
                    for left, right in zip(fill_records, duplicates)
                )
            ):
                raise SourceAlignmentError(
                    "render mode 2 text is not an exact adjacent MuPDF fill/stroke pair: "
                    f"show {event.text_show_ordinal}"
                )
            cursor = duplicate_end
            collapsed_duplicate_glyph_count += len(fill_records)
        elif any(record.span.render_mode != event.render_mode for record in records):
            modes = sorted({record.span.render_mode for record in records})
            raise SourceAlignmentError(
                "source and PyMuPDF render modes disagree for text show "
                f"{event.text_show_ordinal}: {event.render_mode} vs {modes}"
            )

        for joined_event, joined_records in joined_events:
            if not joined_event.visible:
                consumed_hidden_glyph_count += joined_event.glyph_count
                continue

            styles = {_text_style_key(record.span) for record in joined_records}
            if len(styles) != 1:
                mixed_style_count += 1
                raise SourceAlignmentError(
                    "one source text show crosses incompatible PyMuPDF trace styles: "
                    f"show {joined_event.text_show_ordinal}, {len(styles)} styles"
                )
            first = joined_records[0].span
            characters = tuple(record.character for record in joined_records)
            if joined_event.paint_order is None:
                raise SourceAlignmentError("visible source text show has no paint order")
            if joined_event.visible_bounds is None:
                raise SourceAlignmentError(
                    f"visible source text show {joined_event.text_show_ordinal} "
                    "has no source bounds"
                )
            ordinal = len(aligned)
            aligned.append(TextOperationIR(
                operation_id=f"text:{ordinal:08d}",
                paint_order=joined_event.paint_order,
                ordinal=ordinal,
                span_index=first.span_index,
                # Geometry comes from the authored text matrix/font widths and
                # the frozen Scene ascent/descent envelope.  PyMuPDF remains
                # the glyph/style trace, but its font-engine bbox is not the
                # grouping contract and can drift across the exact split threshold.
                bounds=joined_event.visible_bounds,
                literal_text=joined_event.decoded_text,
                characters=characters,
                font_name=first.font_name,
                font_size=first.font_size,
                direction=first.direction,
                render_mode=joined_event.render_mode,
                color=first.color,
                opacity=first.opacity,
                line_width=first.line_width,
                writing_mode=first.writing_mode,
                flags=first.flags,
                bidi_level=first.bidi_level,
                bidi_direction=first.bidi_direction,
                layer=first.layer,
                structure_before=joined_event.structure_before,
                content_stream_index=(
                    joined_event.location.page_content_stream_index
                ),
                form_instance_path=joined_event.location.form_instance_path,
                source_provenance_exact=True,
                source_font_name=joined_event.font_resource_name,
                source_font_size=max(0.1, abs(joined_event.font_size)),
                source_matrix=joined_event.text_matrix,
                source_glyph_advance=max(0.001, abs(joined_event.glyph_advance)),
                source_horizontal_scale=joined_event.horizontal_scale,
                source_rise=joined_event.rise,
                source_unclipped_bounds=_source_text_unclipped_bounds(joined_event),
                source_fill_color=joined_event.fill_color,
                source_stroke_color=joined_event.stroke_color,
                source_fill_opacity=joined_event.fill_opacity,
                source_stroke_opacity=joined_event.stroke_opacity,
                source_line_width=max(0.25, joined_event.line_width),
                source_blend_mode=joined_event.blend_mode,
            ))
        event_index = next_event_index
    if cursor != len(trace):
        raise SourceAlignmentError(
            "PyMuPDF text trace remains after strict source-event join: "
            f"consumed {cursor}, trace {len(trace)}"
        )
    return (
        tuple(aligned),
        len(trace_spans),
        len(trace),
        mixed_style_count,
        collapsed_duplicate_glyph_count,
        consumed_hidden_glyph_count,
    )


def _aligned_path_operations(
    source_paths: tuple[SourcePathPaintEventIR, ...],
) -> tuple[PathOperationIR, ...]:
    """Build every path solely from the authored source paint.

    MuPDF's drawing list may merge an adjacent fill/stroke pair, omit exact
    duplicates, or add renderer-only paths.  It is retained only as an audit
    count by the caller and never carries a source ordinal or path style.
    """

    paths: list[PathOperationIR] = []
    for ordinal, event in enumerate(source_paths):
        if event.path_ordinal != ordinal:
            raise SourceAlignmentError(
                "source path ordinals are not a dense ordered partition: "
                f"expected {ordinal}, received {event.path_ordinal}"
            )
        paths.append(PathOperationIR(
            operation_id=f"path:{ordinal:08d}",
            paint_order=event.paint_order,
            ordinal=ordinal,
            bounds=event.visible_bounds,
            segments=event.segments,
            stroke=event.stroke,
            fill=event.fill,
            stroke_color=event.stroke_color,
            fill_color=event.fill_color,
            line_width=event.line_width,
            hairline=event.stroke and event.line_width == 0.0,
            line_cap=(event.line_cap, event.line_cap, event.line_cap),
            line_join=float(event.line_join),
            miter_limit=event.miter_limit,
            dash_array=event.dash_array,
            dash_phase=event.dash_phase,
            stroke_opacity=event.stroke_opacity,
            fill_opacity=event.fill_opacity,
            even_odd=event.even_odd,
            close_path=event.close_path,
            blend_mode=event.blend_mode,
            structure_before=event.structure_before,
            content_stream_index=event.location.page_content_stream_index,
            form_instance_path=event.location.form_instance_path,
            source_provenance_exact=True,
        ))
    return tuple(paths)


def _source_colorspace_components(name: str) -> int:
    return {
        "/DeviceGray": 1,
        "/G": 1,
        "/DeviceRGB": 3,
        "/RGB": 3,
        "/DeviceCMYK": 4,
        "/CMYK": 4,
    }.get(name, 0)


def _source_image_transform(
    event: SourceImagePaintEventIR,
) -> tuple[float, float, float, float, float, float]:
    lower_left, lower_right, _upper_right, upper_left = event.corners
    return (
        lower_right[0] - lower_left[0],
        lower_right[1] - lower_left[1],
        upper_left[0] - lower_left[0],
        upper_left[1] - lower_left[1],
        lower_left[0],
        lower_left[1],
    )


def _display_image_matches_source(
    display: ImageOperationIR,
    event: SourceImagePaintEventIR,
) -> bool:
    return (
        display.xref != 0
        and display.pixel_width == event.pixel_width
        and display.pixel_height == event.pixel_height
        and (
            _same_display_image_bounds(display.bounds, event.bounds)
            or (
                event.visible_bounds is not None
                and _same_display_image_bounds(display.bounds, event.visible_bounds)
            )
        )
    )


def _display_image_matches_inline(
    display: ImageOperationIR,
    event: SourceInlineImageSkipEventIR,
) -> bool:
    return (
        display.xref == 0
        and display.pixel_width == event.pixel_width
        and display.pixel_height == event.pixel_height
        and (
            _same_display_image_bounds(display.bounds, event.bounds)
            or (
                event.visible_bounds is not None
                and _same_display_image_bounds(display.bounds, event.visible_bounds)
            )
        )
    )


def _aligned_image_operations(
    base_images: tuple[ImageOperationIR, ...],
    source: SourceContentPageIR,
) -> tuple[tuple[ImageOperationIR, ...], int, int]:
    """Emit authored XObject images and strictly consume skipped inline images."""

    expected = sorted(
        (
            *source.image_events,
            *(
                event
                for event in source.inline_image_events
                if event.visible_bounds is not None
            ),
        ),
        key=lambda event: event.event_index,
    )
    display_cursor = 0
    inline_consumed = 0
    source_only_images = 0
    images: list[ImageOperationIR] = []
    for event in expected:
        display = (
            base_images[display_cursor]
            if display_cursor < len(base_images)
            else None
        )
        if isinstance(event, SourceInlineImageSkipEventIR):
            if display is None or not _display_image_matches_inline(display, event):
                raise SourceAlignmentError(
                    "visible source inline image has no exact xref-0 display occurrence: "
                    f"inline ordinal {event.inline_image_ordinal}"
                )
            display_cursor += 1
            inline_consumed += 1
            continue

        if event.visible_bounds is None:  # pragma: no cover - source filters these.
            raise SourceAlignmentError(
                f"source image {event.image_ordinal} has no visible bounds"
            )
        if display is not None and _display_image_matches_source(display, event):
            display_cursor += 1
        elif event.target_placeholder_reason in {
            "IMAGE_PIXEL_BUDGET",
            "IMAGE_MASK_UNSUPPORTED",
            "IMAGE_JPX_UNSUPPORTED",
        }:
            # The frozen browser deliberately represents all three structured
            # image omissions as placeholders. Source identity remains
            # complete even if MuPDF exposes no display-list occurrence.
            display = None
            source_only_images += 1
        else:
            raise SourceAlignmentError(
                "source image has no exact display occurrence and is not a proven structured "
                f"placeholder: image ordinal {event.image_ordinal}"
            )

        ordinal = len(images)
        if event.image_ordinal != ordinal:
            raise SourceAlignmentError(
                "source image ordinals are not a dense ordered partition: "
                f"expected {ordinal}, received {event.image_ordinal}"
            )
        images.append(ImageOperationIR(
            operation_id=f"image:{ordinal:08d}",
            paint_order=event.paint_order,
            ordinal=ordinal,
            bounds=event.visible_bounds,
            pixel_width=event.pixel_width,
            pixel_height=event.pixel_height,
            colorspace=(
                display.colorspace
                if display is not None
                else _source_colorspace_components(event.color_space_name)
            ),
            bits_per_component=event.bits_per_component,
            color_space_name=event.color_space_name,
            source_filters=event.source_filters,
            xref=event.xref,
            digest="" if display is None else display.digest,
            # Authored CTM is the semantic value. MuPDF's transform is only a
            # display cross-audit and must never feed the browser join/hash.
            transform=_source_image_transform(event),
            resource_name=event.resource_name,
            resource_id=event.resource_id,
            resource_scope=event.location.resource_scope,
            corners=event.corners,
            visible_bounds=event.visible_bounds,
            paint_operator=event.paint_operator,
            image_mask=event.image_mask,
            target_placeholder_reason=event.target_placeholder_reason,
            alpha=event.alpha,
            blend_mode=event.blend_mode,
            structure_before=event.structure_before,
            content_stream_index=event.location.page_content_stream_index,
            form_instance_path=event.location.form_instance_path,
            source_provenance_exact=True,
        ))

    if display_cursor != len(base_images):
        raise SourceAlignmentError(
            "unexplained PyMuPDF image occurrences remain after source join: "
            f"consumed {display_cursor}, display {len(base_images)}"
        )
    return tuple(images), inline_consumed, source_only_images


def align_page_ir_with_source_content(
    page: PageIR,
    source: SourceContentPageIR,
) -> SourceAlignedPageIR:
    """Return source-ordered PageIR or raise on any ambiguous ordinal join."""

    if page.page_number != source.page_number:
        raise SourceAlignmentError("PageIR and source content describe different pages")
    if page.source_sha256 != source.source_sha256:
        raise SourceAlignmentError("PageIR and source content have different source digests")
    if not _same_bounds(page.page_bounds, source.page_bounds):
        raise SourceAlignmentError(
            f"page bounds disagree: PageIR {page.page_bounds}, source {source.page_bounds}"
        )
    if page.rotation_degrees != source.rotation_degrees:
        raise SourceAlignmentError("page rotations disagree")
    if source.issues:
        raise SourceAlignmentError(
            "source parsing reported non-exact semantics: " + "; ".join(source.issues[:5])
        )
    if source.unsupported_paint_events:
        details = ", ".join(
            f"{event.operator}@{event.location.global_operator_index}"
            for event in source.unsupported_paint_events[:5]
        )
        raise SourceAlignmentError(f"unsupported source paints prevent alignment: {details}")

    base_paths = tuple(
        operation for operation in page.operations if isinstance(operation, PathOperationIR)
    )
    base_texts = tuple(
        operation for operation in page.operations if isinstance(operation, TextOperationIR)
    )
    base_images = tuple(
        operation for operation in page.operations if isinstance(operation, ImageOperationIR)
    )

    paths = _aligned_path_operations(source.path_events)
    (
        texts,
        trace_span_count,
        trace_glyph_count,
        mixed_style_count,
        collapsed_trace_glyph_count,
        consumed_hidden_trace_glyph_count,
    ) = (
        _aligned_text_operations(base_texts, source.text_events)
    )
    images, inline_image_skip_count, source_only_image_count = (
        _aligned_image_operations(base_images, source)
    )
    operations = [*paths, *texts, *images]
    operations.sort(key=lambda operation: (operation.paint_order, operation.ordinal))
    if len(operations) != len(source.paint_events):
        raise SourceAlignmentError(
            "aligned operation count does not cover every source paint: "
            f"aligned {len(operations)}, source {len(source.paint_events)}"
        )
    if [operation.paint_order for operation in operations] != list(range(len(operations))):
        raise SourceAlignmentError("aligned paint orders are not a dense source partition")

    aligned_page = PageIR(
        page_number=page.page_number,
        page_bounds=page.page_bounds,
        rotation_degrees=page.rotation_degrees,
        operations=tuple(operations),
        source_sha256=page.source_sha256,
        source_name=page.source_name,
        producer=SOURCE_ALIGNED_PAGE_IR_PRODUCER,
        producer_version=f"{page.producer_version}+{source.producer_version}",
    )
    source_glyph_count = sum(
        event.glyph_count for event in source.visible_text_events
    )
    audit = SourcePageAlignmentAudit(
        source_paint_count=len(source.paint_events),
        aligned_operation_count=len(aligned_page.operations),
        source_path_count=len(source.path_events),
        display_path_cross_audit_count=len(base_paths),
        source_atom_count=sum(
            event.atom_multiplicity for event in source.path_events
        ),
        source_visible_text_show_count=len(source.visible_text_events),
        source_visible_text_glyph_count=source_glyph_count,
        display_visible_text_span_count=trace_span_count,
        display_visible_text_glyph_count=trace_glyph_count,
        source_hidden_text_show_count=sum(
            not event.visible for event in source.text_events
        ),
        source_image_count=len(source.image_events),
        display_image_count=len(base_images),
        exact_source_path_construction=True,
        exact_text_event_join=(
            trace_glyph_count
            == source_glyph_count
            + collapsed_trace_glyph_count
            + consumed_hidden_trace_glyph_count
        ),
        exact_source_image_join=True,
        mixed_trace_style_event_count=mixed_style_count,
        collapsed_fill_stroke_trace_glyph_count=collapsed_trace_glyph_count,
        consumed_hidden_trace_glyph_count=consumed_hidden_trace_glyph_count,
        source_annotation_appearance_count=source.annotation_appearance_count,
        source_inline_image_skip_count=sum(
            event.visible_bounds is not None
            for event in source.inline_image_events
        ),
        display_inline_image_count=inline_image_skip_count,
        source_only_image_count=source_only_image_count,
    )
    return SourceAlignedPageIR(aligned_page, source, audit)


def source_aligned_page_ir_from_pdf_bytes(
    source: bytes | bytearray | memoryview,
    page_number: int = 1,
    *,
    source_name: str = "",
) -> SourceAlignedPageIR:
    """Extract and strictly join visible and authored views of one PDF page."""

    with SourceAlignedPdfDocument(source, source_name=source_name) as document:
        return document.page(page_number)


def source_aligned_page_ir_from_pdf_path(
    path: str | Path,
    page_number: int = 1,
) -> SourceAlignedPageIR:
    source_path = Path(path)
    return source_aligned_page_ir_from_pdf_bytes(
        source_path.read_bytes(),
        page_number,
        source_name=source_path.name,
    )


class SourceAlignedPdfDocument:
    """Once-opened PyMuPDF + pypdf document session.

    Whole-document recognition owns one instance for the entire run.  The PDF
    snapshot is neither re-read nor re-opened per page; only page-local display
    geometry and content streams are materialized.
    """

    def __init__(
        self,
        source: bytes | bytearray | memoryview,
        *,
        source_name: str = "",
    ) -> None:
        self._source = bytes(source)
        self.source_name = source_name
        # Runtime validation precedes both document allocations so an unsafe
        # native binding fails explicitly without leaving an open pypdf buffer.
        pymupdf = _pymupdf()
        self._source_document = SourceContentDocument(
            self._source,
            source_name=source_name,
        )
        try:
            self._display_document = pymupdf.open(  # type: ignore[attr-defined]
                stream=self._source,
                filetype="pdf",
            )
        except Exception as error:
            self._source_document.close()
            raise PdfAdapterError(
                "PyMuPDF could not open the input PDF snapshot"
            ) from error
        self._closed = False
        if self._display_document.page_count != self._source_document.page_count:
            self.close()
            raise SourceAlignmentError(
                "PyMuPDF and pypdf disagree on the document page count"
            )

    @property
    def source_sha256(self) -> str:
        return self._source_document.source_sha256

    @property
    def page_count(self) -> int:
        return self._source_document.page_count

    @property
    def producer(self) -> str:
        return SOURCE_ALIGNED_PAGE_IR_PRODUCER

    @property
    def producer_version(self) -> str:
        return current_source_aligned_producer_version()

    def page(self, page_number: int) -> SourceAlignedPageIR:
        if self._closed:
            raise SourceAlignmentError("source-aligned document is closed")
        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError("page_number must be a positive one-based integer")
        if page_number > self.page_count:
            raise IndexError(
                f"page {page_number} is outside the document's 1..{self.page_count} range"
            )
        provenance = self._source_document.page(page_number)
        display = page_ir_from_pymupdf_page(
            self._display_document[page_number - 1],
            source_sha256=self.source_sha256,
            source_name=self.source_name,
            page_number=page_number,
        )
        return align_page_ir_with_source_content(display, provenance)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self._display_document.close()
            finally:
                self._source_document.close()

    def __enter__(self) -> "SourceAlignedPdfDocument":
        if self._closed:
            raise SourceAlignmentError("source-aligned document is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "SOURCE_ALIGNMENT_AUDIT_SCHEMA_VERSION",
    "SOURCE_PAGE_ADAPTER_VERSION",
    "SOURCE_ALIGNED_PAGE_IR_PRODUCER",
    "SourceAlignedPageIR",
    "SourceAlignedPdfDocument",
    "SourceAlignmentError",
    "SourceAlignmentSummary",
    "SourcePageAlignmentAudit",
    "align_page_ir_with_source_content",
    "current_source_aligned_producer_version",
    "source_aligned_page_ir_from_pdf_bytes",
    "source_aligned_page_ir_from_pdf_path",
]
