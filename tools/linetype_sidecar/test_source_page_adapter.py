"""Offline regressions for strict source/display text alignment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from line_type_engine.ir import (  # noqa: E402
    BoundsIR,
    TextCharacterIR,
    TextOperationIR,
)
from line_type_engine.source_content import (  # noqa: E402
    SourceLocationIR,
    SourceTextShowEventIR,
)
from line_type_engine.source_page_adapter import (  # noqa: E402
    SourceAlignmentError,
    _aligned_text_operations,
)


def _characters(text: str, start_x: float = 0.0) -> tuple[TextCharacterIR, ...]:
    return tuple(
        TextCharacterIR(
            codepoint=ord(character),
            glyph_id=ord(character),
            origin=(start_x + index, 5.0),
            bounds=BoundsIR(
                start_x + index,
                0.0,
                start_x + index + 0.8,
                10.0,
            ),
        )
        for index, character in enumerate(text)
    )


def _trace_span(
    text: str,
    *,
    render_mode: int,
    span_index: int,
    characters: tuple[TextCharacterIR, ...],
) -> TextOperationIR:
    return TextOperationIR(
        operation_id=f"trace:{span_index}",
        paint_order=span_index,
        ordinal=span_index,
        span_index=span_index,
        bounds=BoundsIR(0.0, 0.0, float(len(text)), 10.0),
        literal_text=text,
        characters=characters,
        font_name="BradleyHandITC",
        font_size=10.0,
        direction=(1.0, 0.0),
        render_mode=render_mode,
        color=(0.1, 0.4, 0.7),
        opacity=1.0,
        line_width=0.41,
    )


def _source_show(
    text: str,
    *,
    ordinal: int,
    start_x: float,
) -> SourceTextShowEventIR:
    return SourceTextShowEventIR(
        event_index=ordinal,
        paint_order=ordinal,
        text_show_ordinal=ordinal,
        show_operator="Tj",
        array_item_index=None,
        raw_bytes=text.encode("ascii"),
        decoded_text=text,
        glyph_count=len(text),
        visible=True,
        text_object_id="form/text@1",
        font_resource_name="BradleyHandITC",
        font_size=10.0,
        render_mode=2,
        text_matrix=(1.0, 0.0, 0.0, 1.0, start_x, 5.0),
        glyph_advance=float(len(text)),
        horizontal_scale=1.0,
        rise=0.0,
        unclipped_bounds=BoundsIR(start_x, 0.0, start_x + len(text), 10.0),
        fill_color=(0.1, 0.4, 0.7),
        stroke_color=(0.1, 0.4, 0.7),
        fill_opacity=1.0,
        stroke_opacity=1.0,
        line_width=0.41,
        blend_mode="Normal",
        visible_bounds=BoundsIR(start_x, 0.0, start_x + len(text), 10.0),
        location=SourceLocationIR(
            content_stream_id="obj:1:0",
            stream_operator_index=10 + ordinal * 2,
            global_operator_index=10 + ordinal * 2,
        ),
    )


class RenderModeTwoTraceTests(unittest.TestCase):
    def test_consecutive_shows_can_share_one_exact_fill_stroke_run(self) -> None:
        first = "05/06"
        second = "/2026"
        combined = first + second
        glyphs = _characters(combined)
        traces = (
            _trace_span(
                combined,
                render_mode=0,
                span_index=0,
                characters=glyphs,
            ),
            _trace_span(
                combined,
                render_mode=1,
                span_index=1,
                characters=glyphs,
            ),
        )
        events = (
            _source_show(first, ordinal=0, start_x=0.0),
            _source_show(second, ordinal=1, start_x=5.0),
        )

        aligned, span_count, trace_count, mixed, collapsed, hidden = (
            _aligned_text_operations(traces, events)
        )

        self.assertEqual(
            [operation.literal_text for operation in aligned],
            [event.decoded_text for event in events],
        )
        self.assertEqual([operation.paint_order for operation in aligned], [0, 1])
        self.assertEqual([len(operation.characters) for operation in aligned], [5, 5])
        self.assertEqual((span_count, trace_count), (2, 20))
        self.assertEqual((mixed, collapsed, hidden), (0, 10, 0))

    def test_coalesced_stroke_run_must_match_every_glyph_exactly(self) -> None:
        combined = "05/06/2026"
        fill_glyphs = _characters(combined)
        bad_stroke = list(fill_glyphs)
        bad_stroke[7] = replace(
            bad_stroke[7],
            origin=(bad_stroke[7].origin[0] + 0.01, bad_stroke[7].origin[1]),
        )
        traces = (
            _trace_span(
                combined,
                render_mode=0,
                span_index=0,
                characters=fill_glyphs,
            ),
            _trace_span(
                combined,
                render_mode=1,
                span_index=1,
                characters=tuple(bad_stroke),
            ),
        )
        events = (
            _source_show("05/06", ordinal=0, start_x=0.0),
            _source_show("/2026", ordinal=1, start_x=5.0),
        )

        with self.assertRaisesRegex(
            SourceAlignmentError,
            "not an exact adjacent MuPDF fill/stroke pair",
        ):
            _aligned_text_operations(traces, events)

    def test_single_show_adjacent_pair_remains_supported(self) -> None:
        text = "FENCE"
        glyphs = _characters(text)
        traces = (
            _trace_span(text, render_mode=0, span_index=0, characters=glyphs),
            _trace_span(text, render_mode=1, span_index=1, characters=glyphs),
        )

        aligned, _spans, trace_count, _mixed, collapsed, _hidden = (
            _aligned_text_operations(
                traces,
                (_source_show(text, ordinal=0, start_x=0.0),),
            )
        )

        self.assertEqual([operation.literal_text for operation in aligned], [text])
        self.assertEqual((trace_count, collapsed), (10, 5))


if __name__ == "__main__":
    unittest.main()
