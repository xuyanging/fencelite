"""Candidate native-Python sequential grouping for :class:`PageIR`.

This is a forward, complete partition of paint order.  Source-aligned PageIR
supplies the same top-level content-stream, root Form and structural close/open
provenance used by the frozen TypeScript partitioner.  Legacy PageIR without
that proof retains the older display-list structural hints.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .geometry import (
    bounds_gap,
    connection_tolerance,
    operation_contour_gap,
    page_diagonal,
)
from .ir import (
    BoundsIR,
    GroupIR,
    GroupingIR,
    ImageOperationIR,
    OperationIR,
    PageIR,
    PathOperationIR,
    TextOperationIR,
)


@dataclass(frozen=True, slots=True)
class SequentialGroupingOptions:
    """Page-scale-independent thresholds for the candidate partitioner."""

    spatial_jump_ratio: float = 0.20
    structured_style_gap_ratio: float = 0.025
    structured_gap_ratio: float = 0.05

    def __post_init__(self) -> None:
        for name, value in (
            ("spatial_jump_ratio", self.spatial_jump_ratio),
            ("structured_style_gap_ratio", self.structured_style_gap_ratio),
            ("structured_gap_ratio", self.structured_gap_ratio),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative ratio")


DEFAULT_SEQUENTIAL_GROUPING_OPTIONS = SequentialGroupingOptions()


@dataclass(frozen=True, slots=True)
class _PaintBatch:
    """All IR operations emitted by one PyMuPDF display-list paint event.

    ``get_texttrace()`` may expose hundreds of spans for one authored text
    paint.  They must stay available as individual TextOperationIR values for
    Method2, but they are one indivisible event for sequential Grouping.
    """

    paint_order: int
    operations: tuple[OperationIR, ...]
    bounds: BoundsIR
    structure_before: int
    content_stream_index: int
    form_instance_path: tuple[str, ...]
    source_provenance_exact: bool


def _paint_batches(operations: tuple[OperationIR, ...]) -> tuple[_PaintBatch, ...]:
    batches: list[_PaintBatch] = []
    current: list[OperationIR] = []
    current_order = -1

    def append_current() -> None:
        if not current:
            return
        bounds = current[0].bounds
        for operation in current[1:]:
            bounds = bounds.union(operation.bounds)
        exact_values = {operation.source_provenance_exact for operation in current}
        if len(exact_values) != 1:
            raise ValueError("one paint batch mixes exact and legacy provenance")
        source_provenance_exact = exact_values == {True}
        stream_indices = {operation.content_stream_index for operation in current}
        form_paths = {operation.form_instance_path for operation in current}
        structure_masks = {operation.structure_before for operation in current}
        if source_provenance_exact and (
            len(stream_indices) != 1
            or len(form_paths) != 1
            or len(structure_masks) != 1
        ):
            raise ValueError("one source paint batch has inconsistent provenance")
        batches.append(_PaintBatch(
            current_order,
            tuple(current),
            bounds,
            next(iter(structure_masks), 0),
            next(iter(stream_indices), 0),
            next(iter(form_paths), ()),
            source_provenance_exact,
        ))

    for operation in operations:
        if current and operation.paint_order != current_order:
            append_current()
            current = []
        if not current:
            current_order = operation.paint_order
        current.append(operation)
    append_current()
    return tuple(batches)


def _ratio_at_least(left: float, right: float, threshold: float) -> bool:
    small = min(abs(left), abs(right))
    large = max(abs(left), abs(right))
    if small == 0.0:
        return large > 0.0
    return large / small >= threshold


def _effective_path_colors(operation: PathOperationIR) -> tuple[tuple[float, ...], ...]:
    colors = []
    if operation.fill and operation.fill_color is not None:
        colors.append(tuple(operation.fill_color))
    if operation.stroke and operation.stroke_color is not None:
        colors.append(tuple(operation.stroke_color))
    return tuple(sorted(set(colors)))


def _effective_path_opacities(operation: PathOperationIR) -> tuple[float, ...]:
    values = []
    if operation.fill:
        values.append(operation.fill_opacity)
    if operation.stroke:
        values.append(operation.stroke_opacity)
    return tuple(sorted(set(values)))


def _opacity_changed(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    if not left or not right:
        return False
    return abs(min(left) - min(right)) > 0.1 or abs(max(left) - max(right)) > 0.1


def _text_channels(render_mode: int) -> tuple[bool, bool]:
    """Return frozen Scene (fill, stroke) paint-channel membership."""

    return (
        render_mode in {0, 2, 4, 6},
        render_mode in {1, 2, 5, 6},
    )


def _effective_text_colors(operation: TextOperationIR) -> tuple[tuple[float, ...], ...]:
    fill, stroke = _text_channels(operation.render_mode)
    if operation.source_fill_color is None or operation.source_stroke_color is None:
        return (() if operation.color is None or not (fill or stroke) else (operation.color,))
    colors = []
    if fill:
        colors.append(operation.source_fill_color)
    if stroke:
        colors.append(operation.source_stroke_color)
    return tuple(sorted(set(colors)))


def _effective_text_opacities(operation: TextOperationIR) -> tuple[float, ...]:
    fill, stroke = _text_channels(operation.render_mode)
    if (
        operation.source_fill_opacity is None
        or operation.source_stroke_opacity is None
    ):
        return () if not (fill or stroke) else (operation.opacity,)
    values = []
    if fill:
        values.append(operation.source_fill_opacity)
    if stroke:
        values.append(operation.source_stroke_opacity)
    return tuple(sorted(set(values)))


def _strong_style_changed(left: OperationIR, right: OperationIR) -> bool:
    # A kind switch or a fill/stroke channel switch alone is not a strong
    # style boundary.  This keeps vector text and its carrier in one run.
    if type(left) is not type(right):
        return False
    if isinstance(left, PathOperationIR) and isinstance(right, PathOperationIR):
        if _effective_path_colors(left) != _effective_path_colors(right):
            return True
        if _opacity_changed(
            _effective_path_opacities(left),
            _effective_path_opacities(right),
        ):
            return True
        if left.blend_mode != right.blend_mode:
            return True
        if left.stroke and right.stroke:
            return (
                left.hairline != right.hairline
                or _ratio_at_least(left.line_width, right.line_width, 2.0)
                or left.dash_array != right.dash_array
                or left.dash_phase != right.dash_phase
            )
        return False
    if isinstance(left, TextOperationIR) and isinstance(right, TextOperationIR):
        return (
            _effective_text_colors(left) != _effective_text_colors(right)
            or _opacity_changed(
                _effective_text_opacities(left),
                _effective_text_opacities(right),
            )
            or (left.source_blend_mode or "source-over")
            != (right.source_blend_mode or "source-over")
            or left.canonical_font_name != right.canonical_font_name
            or _ratio_at_least(
                left.canonical_font_size, right.canonical_font_size, 1.5
            )
        )
    if isinstance(left, ImageOperationIR) and isinstance(right, ImageOperationIR):
        # Frozen placeholders have no alpha/blend visual style contract.  A
        # kind transition alone also never creates a strong-style restart.
        if left.target_kind != right.target_kind or left.target_kind == "placeholder":
            return False
        return (
            abs(left.alpha - right.alpha) > 0.1
            or left.blend_mode != right.blend_mode
        )
    return False


def _batch_structure_evidence(
    left: _PaintBatch,
    right: _PaintBatch,
) -> tuple[str, ...]:
    if right.source_provenance_exact:
        evidence: list[str] = []
        for flag, name in (
            (1 << 0, "graphics-state"),
            (1 << 1, "text-object"),
            (1 << 2, "marked-content"),
            (1 << 3, "compatibility-section"),
        ):
            if right.structure_before & flag:
                evidence.append(name)
        return tuple(evidence)

    evidence: list[str] = []
    left_layers = {getattr(operation, "layer", "") for operation in left.operations}
    right_layers = {getattr(operation, "layer", "") for operation in right.operations}
    if left_layers != right_layers and any(left_layers | right_layers):
        evidence.append("layer-change")

    left_levels = {
        operation.nesting_level
        for operation in left.operations
        if isinstance(operation, PathOperationIR)
    }
    right_levels = {
        operation.nesting_level
        for operation in right.operations
        if isinstance(operation, PathOperationIR)
    }
    if left_levels and right_levels and left_levels != right_levels:
        evidence.append("nesting-level-change")
    if right.paint_order > left.paint_order + 1:
        evidence.append("paint-order-gap")
    return tuple(evidence)


def _root_form(batch: _PaintBatch) -> str | None:
    return batch.form_instance_path[0] if batch.form_instance_path else None


def _command_stream_key(batch: _PaintBatch) -> tuple[int, str]:
    return (batch.content_stream_index, _root_form(batch) or "page")


def _batch_strong_style_changed(left: _PaintBatch, right: _PaintBatch) -> bool:
    # One compatible cross-batch pairing is sufficient to avoid treating a
    # heterogeneous text paint as a strong style restart.  This is the batch
    # equivalent of the former adjacent-operation comparison without choosing
    # an arbitrary last/first span.
    return all(
        _strong_style_changed(left_operation, right_operation)
        for left_operation in left.operations
        for right_operation in right.operations
    )


def _batch_contour_gap(
    left: _PaintBatch,
    right: _PaintBatch,
    diagonal: float,
) -> float:
    if len(left.operations) == 1 and len(right.operations) == 1:
        return operation_contour_gap(left.operations[0], right.operations[0], diagonal)
    # Multi-span text paints are atomic authored events.  Their complete union
    # must participate in the next boundary decision; selecting the final span
    # makes the partition depend on extraction order inside one text paint.
    return bounds_gap(left.bounds, right.bounds)


def _batch_connection_tolerance(
    left: _PaintBatch,
    right: _PaintBatch,
    diagonal: float,
) -> float:
    def line_width(operation: OperationIR) -> float:
        return max(0.0, float(getattr(operation, "line_width", 0.0) or 0.0))

    left_widest = max(left.operations, key=line_width)
    right_widest = max(right.operations, key=line_width)
    return connection_tolerance(left_widest, right_widest, diagonal)


def _boundary_reasons(
    left: _PaintBatch,
    right: _PaintBatch,
    diagonal: float,
    options: SequentialGroupingOptions,
) -> tuple[str, ...]:
    gap = _batch_contour_gap(left, right, diagonal)
    epsilon = max(1e-12, diagonal * 1e-9)
    if gap <= max(epsilon, _batch_connection_tolerance(left, right, diagonal)):
        return ()

    reasons: list[str] = []
    if gap > max(epsilon, options.spatial_jump_ratio * diagonal):
        reasons.append("spatial-jump")

    structure = _batch_structure_evidence(left, right)
    strong_style = _batch_strong_style_changed(left, right)
    structured_threshold = (
        options.structured_style_gap_ratio
        if strong_style
        else options.structured_gap_ratio
    ) * diagonal
    if structure and gap > max(epsilon, structured_threshold):
        reasons.append("structure-transition")
        reasons.extend(f"structure:{item}" for item in structure)
        if strong_style:
            reasons.append("strong-style-change")
    return tuple(reasons)


def _group_id(index: int) -> str:
    # Match the algorithm/result contract used by the frozen recognizer and UI:
    # Group ids are one-based decimal strings.  Ordering is carried by the
    # Group tuple itself, not by a presentation prefix.
    return str(index)


def _make_group(
    index: int,
    operations: list[OperationIR],
    split_reasons: tuple[str, ...],
) -> GroupIR:
    bounds: BoundsIR = operations[0].bounds
    for operation in operations[1:]:
        bounds = bounds.union(operation.bounds)
    return GroupIR(
        group_id=_group_id(index),
        operation_ids=tuple(operation.operation_id for operation in operations),
        bounds=bounds,
        first_paint_order=operations[0].paint_order,
        last_paint_order=operations[-1].paint_order,
        split_reasons=split_reasons,
    )


def group_page_sequentially(
    page: PageIR,
    options: SequentialGroupingOptions | None = None,
) -> GroupingIR:
    """Return a deterministic contiguous and complete partition of ``page``."""

    active_options = options or DEFAULT_SEQUENTIAL_GROUPING_OPTIONS
    operations = page.operations
    if not operations:
        return GroupingIR(page.fingerprint, (), ())

    diagonal = page_diagonal(page.page_bounds)
    batches = _paint_batches(operations)
    groups: list[GroupIR] = []
    current_operations = list(batches[0].operations)
    current_split_reasons: tuple[str, ...] = ()

    for left, right in zip(batches, batches[1:]):
        if (
            left.source_provenance_exact
            and right.source_provenance_exact
            and _command_stream_key(left) != _command_stream_key(right)
        ):
            split_reasons = ("command-stream-switch",)
        elif (
            left.source_provenance_exact
            and right.source_provenance_exact
            and _root_form(left) is not None
            and _root_form(left) == _root_form(right)
        ):
            # Match the frozen TS contract: one top-level Form invocation is
            # indivisible even when nested child paints are spatially distant.
            split_reasons = ()
        else:
            split_reasons = _boundary_reasons(
                left,
                right,
                diagonal,
                active_options,
            )
        if split_reasons:
            groups.append(
                _make_group(len(groups) + 1, current_operations, current_split_reasons)
            )
            current_operations = list(right.operations)
            current_split_reasons = split_reasons
        else:
            current_operations.extend(right.operations)

    groups.append(_make_group(len(groups) + 1, current_operations, current_split_reasons))
    assignments = tuple(
        (operation_id, group.group_id)
        for group in groups
        for operation_id in group.operation_ids
    )
    return GroupingIR(
        page_fingerprint=page.fingerprint,
        groups=tuple(groups),
        assignments=assignments,
    )


# Short alias for callers that do not need the implementation detail in their
# domain vocabulary.  Both names return the same candidate GroupingIR.
group_page = group_page_sequentially
