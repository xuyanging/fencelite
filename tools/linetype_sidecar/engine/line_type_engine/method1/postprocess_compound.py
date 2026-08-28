"""Frozen Method1 r10 compound-path and extendable-route post-processing.

This is a renderer-neutral port of the compound/route stages in the frozen
TypeScript oracle.  Numeric ownership is the dense position in
``PageIR.operations``; ``paint_order`` remains ordering evidence only and may
repeat.  Every public stage returns new immutable result values and never
mutates PageIR, GroupingIR, serialized Groups, or the incoming result.
"""

from __future__ import annotations

from array import array
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from itertools import combinations
import math
import multiprocessing
import os
from typing import Iterable, Literal, Mapping, Sequence

from ..ir import BoundsIR, GroupingIR, PageIR, PathOperationIR
from ..operation_index import PageOperationIndex
from ..results import (
    GlobalLineType,
    GlobalLineTypeMember,
    LineTypeRecognitionResult,
    LocalLineType,
    RecognizedGroup,
)
from .serializer import SerializedGroup


_LOCAL_KEY_SEPARATOR = "\0"
_COMPOUND_MATCHED_MASS_MINIMUM = 0.5
_COMPOUND_STRONG_MATCH_MINIMUM = 0.90


def _local_type_key(group_id: str, type_id: str) -> str:
    return f"{group_id}{_LOCAL_KEY_SEPARATOR}{type_id}"


def _index_local_types(
    groups: Sequence[RecognizedGroup],
) -> dict[str, LocalLineType]:
    return {
        _local_type_key(group.group_id, line_type.type_id): line_type
        for group in groups
        for line_type in group.line_types
    }


def _sorted_unique_indices(indices: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(indices)))


def _atom_count_for_ops(
    serialized: SerializedGroup,
    operation_indices: set[int],
) -> int:
    # One PDF path may contain several drawable subpaths.  The serializer
    # intentionally repeats its dense result index once per atom.
    return sum(index in operation_indices for index in serialized.atom_op_indices)


def _number(value: float) -> str:
    finite = float(value) if math.isfinite(float(value)) else 0.0
    normalized = 0.0 if abs(finite) < 1e-10 else finite
    fixed = f"{normalized:.6f}".rstrip("0").rstrip(".")
    return "0" if fixed in {"", "-0"} else fixed


def _color_key(color: tuple[float, ...] | None) -> str:
    return "none" if color is None else ",".join(_number(value) for value in color)


def _path_style_key(operation: PathOperationIR) -> str:
    return "|".join((
        "S" if operation.stroke else "",
        "F" if operation.fill else "",
        _number(operation.line_width),
        _color_key(operation.stroke_color),
        _color_key(operation.fill_color),
    ))


def _bounds_center(bounds: BoundsIR) -> tuple[float, float]:
    return (
        (bounds.min_x + bounds.max_x) / 2.0,
        (bounds.min_y + bounds.max_y) / 2.0,
    )


def _bounds_gap(left: BoundsIR, right: BoundsIR) -> float:
    dx = max(0.0, left.min_x - right.max_x, right.min_x - left.max_x)
    dy = max(0.0, left.min_y - right.max_y, right.min_y - left.max_y)
    return math.hypot(dx, dy)


def _median_number(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _lower_half_median(values: Iterable[float]) -> float:
    ordered = sorted(
        float(value)
        for value in values
        if math.isfinite(float(value)) and float(value) > 0.001
    )
    if not ordered:
        return 0.0
    return _median_number(ordered[: max(1, math.ceil(len(ordered) / 2))])


def _js_round(value: float) -> int:
    """Match JavaScript ``Math.round`` for finite values used in buckets."""

    return math.floor(value + 0.5)


def _path_ink_length(operation: PathOperationIR) -> float:
    length = 0.0
    current: tuple[float, float] | None = None
    for segment in operation.segments:
        if segment.kind == "move":
            current = segment.end
        elif segment.kind == "line":
            if current is not None and segment.end is not None:
                length += math.hypot(
                    segment.end[0] - current[0],
                    segment.end[1] - current[1],
                )
            current = segment.end
        elif segment.kind == "curve":
            if (
                current is not None
                and segment.control_1 is not None
                and segment.control_2 is not None
                and segment.end is not None
            ):
                length += math.dist(current, segment.control_1)
                length += math.dist(segment.control_1, segment.control_2)
                length += math.dist(segment.control_2, segment.end)
            current = segment.end
    return length


def _path_index(page: PageIR) -> dict[int, PathOperationIR]:
    """Index paths by dense result id; repeated paint order is valid."""

    return {
        dense_index: operation
        for dense_index, operation in enumerate(page.operations)
        if isinstance(operation, PathOperationIR)
    }


@dataclass(frozen=True, slots=True)
class CompoundPathProfile:
    family: Literal["equal_dash", "short_long", "point_carrier"]
    zero_ratio: float
    dominant_fraction: float
    long_ratio: float
    curve_ratio: float
    aspect_ratio: float
    segment_median: float
    complex_path_ratio: float
    straight_ratio: float
    closed_ratio: float
    ordered_length_change_ratio: float
    ordered_topology_change_ratio: float
    relative_angle_fractions: tuple[float, float, float]
    relative_angle_change_ratio: float
    paint_order_jump_ratio: float
    ordered_tokens: tuple[int, ...]
    ordered_token_fractions: tuple[float, ...]
    ordered_transition_fractions: tuple[float, ...]
    ordered_motif: tuple[int, ...]
    ordered_motif_confidence: float


@dataclass(frozen=True, slots=True)
class _CompoundPathRun:
    group_id: str
    style: str
    op_indices: tuple[int, ...]
    excluded_op_indices: tuple[int, ...]
    profile: CompoundPathProfile
    assigned_count: int


@dataclass(frozen=True, slots=True)
class _CompoundLocalIdentity:
    local_key: str
    broad_keys: tuple[str, ...]
    runs: tuple[_CompoundPathRun, ...]
    op_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _CompoundRunTask:
    """One frozen-style contiguous run, independent of every other run.

    The main process creates these in the original Group/paint order using the
    unchanged style, non-atom and spatial-break rules.  A very large authored
    Group can therefore use many CPUs without allowing workers to decide a
    sequential boundary.  Each path object belongs to one task, avoiding a
    complete PageIR copy per Windows ``spawn`` worker.
    """

    group_id: str
    style: str
    paths: tuple[tuple[int, PathOperationIR], ...]
    operation_indices: tuple[int, ...]
    assigned_path_indices: tuple[int, ...]


_compound_identity_worker_inputs: tuple[_CompoundLocalIdentity, ...] | None = None


def _compound_worker_budget(
    group_count: int,
    path_count: int,
    requested: int | None,
) -> int:
    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
    ):
        raise ValueError("worker_count must be a positive integer")
    available = os.cpu_count() or 1
    desired = requested or available
    # Process startup is slower than the exact serial loop on ordinary pages.
    # A daemon cannot legally create children (for example when a caller has
    # already assigned one process per document), so it retains serial parity.
    if (
        multiprocessing.current_process().daemon
        or group_count < 32
        or path_count < 2_048
    ):
        return 1
    useful_by_mass = max(1, path_count // 512)
    return max(1, min(desired, available, group_count, useful_by_mass))


def _compound_identity_worker_budget(
    identity_count: int,
    pair_count: int,
    requested: int | None,
) -> int:
    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
    ):
        raise ValueError("worker_count must be a positive integer")
    available = os.cpu_count() or 1
    desired = requested or available
    # Each pair is independent, but Windows spawn and the one-time initializer
    # copy are slower than the exact serial loop for a small matrix.  A daemon
    # cannot create children, matching the discovery-stage fallback above.
    if multiprocessing.current_process().daemon or pair_count < 8_192:
        return 1
    useful_by_mass = max(1, pair_count // 4_096)
    return max(1, min(desired, available, identity_count, useful_by_mass))


def _initialize_compound_identity_worker(
    identities: tuple[_CompoundLocalIdentity, ...],
) -> None:
    global _compound_identity_worker_inputs
    _compound_identity_worker_inputs = identities


def _compound_identity_pair_similarity(pair: tuple[int, int]) -> float:
    identities = _compound_identity_worker_inputs
    if identities is None:  # pragma: no cover - guarded by ProcessPool initializer.
        raise RuntimeError("compound identity worker was not initialized")
    left_index, right_index = pair
    return _compound_local_identity_similarity(
        identities[left_index],
        identities[right_index],
    )


@dataclass(frozen=True, slots=True)
class _CompoundIdentitySimilarityMatrix:
    """Dense upper triangle in stable identity-index order."""

    identity_count: int
    values: array

    def similarity(self, left_index: int, right_index: int) -> float:
        if left_index == right_index:
            return 1.0
        if left_index > right_index:
            left_index, right_index = right_index, left_index
        offset = (
            left_index * (2 * self.identity_count - left_index - 1) // 2
            + right_index
            - left_index
            - 1
        )
        return self.values[offset]


def _precompute_compound_identity_similarities(
    identities: Sequence[_CompoundLocalIdentity],
    worker_count: int | None,
) -> _CompoundIdentitySimilarityMatrix:
    """Compute each symmetric identity pair exactly once, in stable order."""

    frozen = tuple(identities)
    pair_count = len(frozen) * (len(frozen) - 1) // 2
    workers = _compound_identity_worker_budget(
        len(frozen),
        pair_count,
        worker_count,
    )
    pairs = combinations(range(len(frozen)), 2)
    if workers == 1:
        values = array(
            "d",
            (
                _compound_local_identity_similarity(frozen[left], frozen[right])
                for left, right in pairs
            ),
        )
    else:
        chunksize = max(1, pair_count // (workers * 8))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_compound_identity_worker,
            initargs=(frozen,),
        ) as executor:
            values = array(
                "d",
                executor.map(
                    _compound_identity_pair_similarity,
                    pairs,
                    chunksize=chunksize,
                ),
            )
    if len(values) != pair_count:  # pragma: no cover - executor.map is exhaustive.
        raise RuntimeError("compound identity similarity matrix is incomplete")
    return _CompoundIdentitySimilarityMatrix(len(frozen), values)


def _split_spatial_route_runs(
    paths_by_order: Mapping[int, PathOperationIR],
    operation_indices: Sequence[int],
    family: Literal["equal_dash", "short_long", "point_carrier"] | None = None,
) -> tuple[tuple[int, ...], ...]:
    paths = tuple(sorted(index for index in operation_indices if index in paths_by_order))
    if not paths:
        return ()
    lengths = tuple(_path_ink_length(paths_by_order[index]) for index in paths)
    widths = tuple(paths_by_order[index].line_width for index in paths)
    consecutive_gaps = tuple(
        _bounds_gap(paths_by_order[paths[index]].bounds, paths_by_order[paths[index + 1]].bounds)
        for index in range(len(paths) - 1)
    )
    positive_lengths = tuple(
        length for length in lengths if math.isfinite(length) and length > 0.001
    )
    atom_scale = (
        _median_number(positive_lengths)
        if family is not None and family != "point_carrier"
        else _lower_half_median(lengths)
    )
    jump_limit = max(
        6.0,
        _median_number(widths) * 16.0,
        atom_scale * 2.75,
        min(_lower_half_median(consecutive_gaps) * 4.0, atom_scale * 6.0),
    )
    runs: list[tuple[int, ...]] = []
    current: list[int] = []
    for index in paths:
        if (
            current
            and _bounds_gap(paths_by_order[current[-1]].bounds, paths_by_order[index].bounds)
            > jump_limit
        ):
            runs.append(tuple(current))
            current = []
        current.append(index)
    if current:
        runs.append(tuple(current))
    return tuple(runs)


def _is_zero_length_path(operation: PathOperationIR, typical_width: float) -> bool:
    return _path_ink_length(operation) <= max(0.01, typical_width * 0.20)


def _has_point_carrier_core(
    paths_by_order: Mapping[int, PathOperationIR],
    operation_indices: Sequence[int],
) -> bool:
    if len(operation_indices) < 4:
        return False
    paths = tuple(
        paths_by_order[index] for index in operation_indices if index in paths_by_order
    )
    typical_width = _median_number(operation.line_width for operation in paths)
    zero = tuple(_is_zero_length_path(operation, typical_width) for operation in paths)
    first_carrier = -1
    last_carrier = -1
    start = 0
    while start < len(zero):
        if not zero[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(zero) and zero[end + 1]:
            end += 1
        before = start - 1
        after = end + 1
        if end - start + 1 >= 2 and (
            (before >= 0 and not zero[before])
            or (after < len(zero) and not zero[after])
        ):
            if first_carrier < 0:
                first_carrier = before if before >= 0 else 0
            last_carrier = after if after < len(zero) else len(zero) - 1
        start = end + 1
    return first_carrier >= 0 and last_carrier > first_carrier


def _is_extendable_route_run(
    paths_by_order: Mapping[int, PathOperationIR],
    operation_indices: Sequence[int],
) -> bool:
    if len(operation_indices) < 4:
        return False
    paths = tuple(
        paths_by_order[index] for index in operation_indices if index in paths_by_order
    )
    if len(paths) < 4:
        return False
    centers = tuple(_bounds_center(operation.bounds) for operation in paths)
    center_span = math.hypot(
        max(center[0] for center in centers) - min(center[0] for center in centers),
        max(center[1] for center in centers) - min(center[1] for center in centers),
    )
    atom_spans = tuple(
        max(operation.bounds.width, operation.bounds.height) for operation in paths
    )
    typical_atom_span = _median_number(atom_spans)
    maximum_atom_span = max(atom_spans)
    typical_width = _median_number(operation.line_width for operation in paths)
    moving_steps = sum(
        math.dist(centers[index - 1], centers[index])
        > max(0.25, typical_width * 0.5)
        for index in range(1, len(centers))
    )
    return moving_steps >= 3 and center_span >= max(
        typical_atom_span * 1.80,
        maximum_atom_span * 1.75,
        typical_width * 10.0,
    )


def _normalized_histogram(values: Sequence[int], size: int) -> tuple[float, ...]:
    histogram = [0] * size
    for value in values:
        if 0 <= value < size:
            histogram[value] += 1
    total = sum(histogram)
    return tuple(value / max(1, total) for value in histogram)


def _canonical_cyclic_sequence(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        return ()
    variants: list[tuple[int, ...]] = []
    for source in (tuple(values), tuple(reversed(values))):
        for offset in range(len(source)):
            variants.append(source[offset:] + source[:offset])
    return min(variants)


def _ordered_motif_for(tokens: Sequence[int]) -> tuple[tuple[int, ...], float]:
    if len(tokens) < 2:
        return (tuple(tokens), 1.0)
    best_motif: tuple[int, ...] | None = None
    best_confidence = 1.0
    best_score = -math.inf
    maximum_period = min(12, len(tokens) // 2)
    for period in range(1, maximum_period + 1):
        motif: list[int] = []
        for phase in range(period):
            frequencies: dict[int, int] = {}
            for index in range(phase, len(tokens), period):
                token = tokens[index]
                frequencies[token] = frequencies.get(token, 0) + 1
            motif.append(sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))[0][0])
        confidence = sum(
            token == motif[index % period] for index, token in enumerate(tokens)
        ) / len(tokens)
        score = confidence - period * 0.02
        if (
            best_motif is None
            or score > best_score + 1e-9
            or (abs(score - best_score) <= 1e-9 and period < len(best_motif))
        ):
            best_motif = tuple(motif)
            best_confidence = confidence
            best_score = score
    return (_canonical_cyclic_sequence(best_motif or tuple(tokens)), best_confidence)


def _trim_isolated_compound_outliers(
    paths_by_order: Mapping[int, PathOperationIR],
    operation_indices: Sequence[int],
) -> tuple[int, ...]:
    if len(operation_indices) < 10:
        return tuple(operation_indices)
    metrics = []
    for index in operation_indices:
        operation = paths_by_order.get(index)
        if operation is None:
            metrics.append((index, 0.0, 0.0, 0, False))
        else:
            metrics.append((
                index,
                _path_ink_length(operation),
                max(operation.bounds.width, operation.bounds.height),
                len(operation.segments),
                any(segment.kind == "curve" for segment in operation.segments),
            ))
    positive_lengths = tuple(metric[1] for metric in metrics if metric[1] > 0.001)
    typical_length = max(0.001, _median_number(positive_lengths))
    typical_span = max(0.001, _median_number(metric[2] for metric in metrics))
    typical_segments = max(1.0, _median_number(metric[3] for metric in metrics))
    frequencies: dict[int, int] = {}
    for metric in metrics:
        frequencies[metric[3]] = frequencies.get(metric[3], 0) + 1
    stable_topology = max(frequencies.values(), default=0) / len(metrics) >= 0.60
    mostly_linear = sum(metric[4] for metric in metrics) / len(metrics) <= 0.10
    retained = tuple(
        metric[0]
        for metric in metrics
        if not (
            (
                metric[1] > typical_length * 8.0
                and metric[2] > typical_span * 6.0
                and (
                    metric[3] > typical_segments + 8.0
                    or metric[1] > typical_length * 20.0
                    or metric[2] > typical_span * 20.0
                )
            )
            or (stable_topology and mostly_linear and metric[4])
        )
    )
    return retained if len(retained) >= 5 else tuple(operation_indices)


def _compound_path_profile(
    paths_by_order: Mapping[int, PathOperationIR],
    operation_indices: Sequence[int],
) -> CompoundPathProfile | None:
    if len(operation_indices) < 4:
        return None
    lengths = tuple(
        _path_ink_length(paths_by_order[index]) if index in paths_by_order else 0.0
        for index in operation_indices
    )
    typical_width = _median_number(
        paths_by_order[index].line_width if index in paths_by_order else 0.0
        for index in operation_indices
    )
    zero_limit = max(0.01, typical_width * 0.20)
    zero_count = sum(length <= zero_limit for length in lengths)
    positive = tuple(length for length in lengths if length > zero_limit)
    if len(positive) < 2 or zero_count / len(lengths) > 0.75:
        return None
    path_operations = tuple(
        paths_by_order[index] for index in operation_indices if index in paths_by_order
    )
    curve_ratio = sum(
        any(segment.kind == "curve" for segment in operation.segments)
        for operation in path_operations
    ) / max(1, len(path_operations))
    aspect_ratio = _median_number(
        min(operation.bounds.width, operation.bounds.height)
        / max(0.001, max(operation.bounds.width, operation.bounds.height))
        for operation in path_operations
    )
    segment_median = _median_number(len(operation.segments) for operation in path_operations)
    complex_path_ratio = sum(
        len(operation.segments) >= 10 for operation in path_operations
    ) / max(1, len(path_operations))

    endpoint_metrics: list[tuple[bool, bool]] = []
    for operation in path_operations:
        start: tuple[float, float] | None = None
        end: tuple[float, float] | None = None
        explicitly_closed = False
        for segment in operation.segments:
            if segment.kind == "move":
                if start is None:
                    start = segment.end
                end = segment.end
            elif segment.kind in {"line", "curve"}:
                end = segment.end
            else:
                explicitly_closed = True
        ink_length = max(0.001, _path_ink_length(operation))
        chord = math.dist(start, end) if start is not None and end is not None else 0.0
        endpoint_metrics.append((
            chord / ink_length >= 0.90,
            explicitly_closed or chord <= max(0.01, operation.line_width * 0.20),
        ))
    straight_ratio = sum(metric[0] for metric in endpoint_metrics) / max(1, len(endpoint_metrics))
    closed_ratio = sum(metric[1] for metric in endpoint_metrics) / max(1, len(endpoint_metrics))

    ordered_positive = sorted(positive)
    dominant: list[float] = []
    window_start = 0
    for window_end in range(len(ordered_positive)):
        while (
            window_start < window_end
            and ordered_positive[window_end] / max(0.001, ordered_positive[window_start]) > 1.33
        ):
            window_start += 1
        if window_end - window_start + 1 > len(dominant):
            dominant = ordered_positive[window_start : window_end + 1]
    dominant_length = max(0.001, _median_number(dominant))
    long_fraction = sum(length > dominant_length * 2.20 for length in positive) / len(positive)
    length_classes = tuple(
        0 if length <= zero_limit else 2 if length > dominant_length * 2.20 else 1
        for length in lengths
    )
    topology_classes = tuple(
        0 if len(operation.segments) <= 3 else 1 if len(operation.segments) <= 8 else 2
        for operation in path_operations
    )

    def adjacent_change_ratio(values: Sequence[int]) -> float:
        if len(values) < 2:
            return 0.0
        return sum(values[index] != values[index - 1] for index in range(1, len(values))) / (len(values) - 1)

    centers = tuple(_bounds_center(operation.bounds) for operation in path_operations)
    chord_angles: list[float | None] = []
    for operation in path_operations:
        start = None
        end = None
        for segment in operation.segments:
            if segment.kind == "move":
                if start is None:
                    start = segment.end
                end = segment.end
            elif segment.kind in {"line", "curve"}:
                end = segment.end
        chord_angles.append(
            math.atan2(end[1] - start[1], end[0] - start[0])
            if start is not None and end is not None and math.dist(start, end) > 0.001
            else None
        )
    relative_angle_classes: list[int] = []
    for index, angle in enumerate(chord_angles):
        if angle is None or len(centers) < 2:
            relative_angle_classes.append(3)
            continue
        before_index = max(0, index - 1)
        after_index = min(len(centers) - 1, index + 1)
        if before_index == after_index:
            relative_angle_classes.append(3)
            continue
        before = centers[before_index]
        after = centers[after_index]
        tangent = math.atan2(after[1] - before[1], after[0] - before[0])
        difference = abs(angle - tangent) % math.pi
        difference = min(difference, math.pi - difference)
        degrees = difference * 180.0 / math.pi
        relative_angle_classes.append(0 if degrees <= 22.5 else 1 if degrees <= 67.5 else 2)
    known_angle_count = max(1, sum(value != 3 for value in relative_angle_classes))
    relative_angle_fractions = tuple(
        sum(value == bucket for value in relative_angle_classes) / known_angle_count
        for bucket in range(3)
    )
    ordered_tokens = tuple(
        length_class * 12
        + (topology_classes[index] if index < len(topology_classes) else 0) * 4
        + (relative_angle_classes[index] if index < len(relative_angle_classes) else 3)
        for index, length_class in enumerate(length_classes)
    )
    ordered_transitions = tuple(
        min(ordered_tokens[index - 1], ordered_tokens[index]) * 36
        + max(ordered_tokens[index - 1], ordered_tokens[index])
        for index in range(1, len(ordered_tokens))
    )
    ordered_motif, motif_confidence = _ordered_motif_for(ordered_tokens)
    center_steps = tuple(math.dist(centers[index - 1], centers[index]) for index in range(1, len(centers)))
    typical_step = max(0.001, _median_number(center_steps))
    paint_order_jump_ratio = sum(step > typical_step * 6.0 for step in center_steps) / max(1, len(center_steps))

    values = dict(
        zero_ratio=zero_count / len(lengths),
        dominant_fraction=len(dominant) / len(positive),
        long_ratio=max(positive) / dominant_length,
        curve_ratio=curve_ratio,
        aspect_ratio=aspect_ratio,
        segment_median=segment_median,
        complex_path_ratio=complex_path_ratio,
        straight_ratio=straight_ratio,
        closed_ratio=closed_ratio,
        ordered_length_change_ratio=adjacent_change_ratio(length_classes),
        ordered_topology_change_ratio=adjacent_change_ratio(topology_classes),
        relative_angle_fractions=relative_angle_fractions,
        relative_angle_change_ratio=adjacent_change_ratio(relative_angle_classes),
        paint_order_jump_ratio=paint_order_jump_ratio,
        ordered_tokens=ordered_tokens,
        ordered_token_fractions=_normalized_histogram(ordered_tokens, 36),
        ordered_transition_fractions=_normalized_histogram(ordered_transitions, 36 * 36),
        ordered_motif=ordered_motif,
        ordered_motif_confidence=motif_confidence,
    )
    if zero_count >= 2 and zero_count / len(lengths) >= 0.15:
        return CompoundPathProfile(family="point_carrier", **values)
    short_alternating_legend = (
        5 <= len(operation_indices) <= 7
        and len(dominant) >= 2
        and sum(length > dominant_length * 2.20 for length in lengths) >= 2
        and all(length <= dominant_length * 1.33 or length > dominant_length * 2.20 for length in lengths)
        and sum(
            (lengths[index] > dominant_length * 2.20)
            != (lengths[index - 1] > dominant_length * 2.20)
            for index in range(1, len(lengths))
        ) >= 3
    )
    if (
        (len(dominant) < 3 or len(dominant) / len(positive) < 0.30)
        and not short_alternating_legend
    ):
        return None
    values["zero_ratio"] = 0.0
    return CompoundPathProfile(
        family="short_long" if long_fraction >= 0.12 else "equal_dash",
        **values,
    )


def _maximum_fraction_difference(left: Sequence[float], right: Sequence[float]) -> float:
    return max(
        (abs(value - (right[index] if index < len(right) else 0.0)) for index, value in enumerate(left)),
        default=-math.inf,
    )


def _histogram_variation(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        abs(value - (right[index] if index < len(right) else 0.0))
        for index, value in enumerate(left)
    ) / 2.0


def _reliable_ordered_motif(profile: CompoundPathProfile) -> bool:
    return (
        profile.ordered_motif_confidence >= 0.68
        and len(profile.ordered_tokens) / max(1, len(profile.ordered_motif)) >= 2.5
    )


def _point_carrier_mark_count(profile: CompoundPathProfile) -> int:
    zero = tuple(token // 12 == 0 for token in profile.ordered_tokens)
    counts: list[int] = []
    start = 0
    while start < len(zero):
        if not zero[start]:
            start += 1
            continue
        end = start
        while end + 1 < len(zero) and zero[end + 1]:
            end += 1
        counts.append(end - start + 1)
        start = end + 1
    return _js_round(_median_number(counts))


def _compound_profile_similarity(
    left: CompoundPathProfile,
    right: CompoundPathProfile,
    allow_slender_family_change: bool = False,
) -> float:
    if left.family != right.family and not allow_slender_family_change:
        return 0.0
    if left.family == "point_carrier" and right.family == "point_carrier":
        left_marks = _point_carrier_mark_count(left)
        right_marks = _point_carrier_mark_count(right)
        if left_marks < 2 or left_marks != right_marks:
            return 0.0
        if abs(left.curve_ratio - right.curve_ratio) > 0.18:
            return 0.0
        if abs(left.segment_median - right.segment_median) > 2.0:
            return 0.0
        return max(
            0.01,
            1.0
            - (
                abs(left.zero_ratio - right.zero_ratio)
                + abs(left.curve_ratio - right.curve_ratio)
                + abs(left.straight_ratio - right.straight_ratio)
            )
            / 3.0,
        )
    if abs(left.zero_ratio - right.zero_ratio) > 0.20:
        return 0.0
    if abs(left.dominant_fraction - right.dominant_fraction) > 0.35:
        return 0.0
    if abs(left.curve_ratio - right.curve_ratio) > 0.18:
        return 0.0
    if abs(left.complex_path_ratio - right.complex_path_ratio) > 0.20:
        return 0.0
    if abs(left.straight_ratio - right.straight_ratio) > 0.25:
        return 0.0
    if abs(left.closed_ratio - right.closed_ratio) > 0.20:
        return 0.0
    if abs(left.segment_median - right.segment_median) > 2.0:
        return 0.0
    if abs(left.ordered_length_change_ratio - right.ordered_length_change_ratio) > 0.28:
        return 0.0
    if abs(left.ordered_topology_change_ratio - right.ordered_topology_change_ratio) > 0.25:
        return 0.0
    if _maximum_fraction_difference(
        left.relative_angle_fractions,
        right.relative_angle_fractions,
    ) > 0.30:
        return 0.0
    if abs(left.relative_angle_change_ratio - right.relative_angle_change_ratio) > 0.30:
        return 0.0
    if abs(left.paint_order_jump_ratio - right.paint_order_jump_ratio) > 0.22:
        return 0.0
    token_variation = _histogram_variation(
        left.ordered_token_fractions,
        right.ordered_token_fractions,
    )
    transition_variation = _histogram_variation(
        left.ordered_transition_fractions,
        right.ordered_transition_fractions,
    )
    if not allow_slender_family_change and token_variation > 0.42:
        return 0.0
    if not allow_slender_family_change and transition_variation > 0.48:
        return 0.0
    if (
        not allow_slender_family_change
        and _reliable_ordered_motif(left)
        and _reliable_ordered_motif(right)
        and left.ordered_motif != right.ordered_motif
    ):
        return 0.0
    if left.family == "short_long" and right.family == "short_long":
        ratio = max(left.long_ratio, right.long_ratio) / max(
            0.001,
            min(left.long_ratio, right.long_ratio),
        )
        if ratio > 2.25:
            return 0.0
    differences = (
        abs(left.zero_ratio - right.zero_ratio),
        abs(left.dominant_fraction - right.dominant_fraction),
        abs(left.curve_ratio - right.curve_ratio),
        abs(left.complex_path_ratio - right.complex_path_ratio),
        abs(left.straight_ratio - right.straight_ratio),
        abs(left.closed_ratio - right.closed_ratio),
        abs(left.ordered_length_change_ratio - right.ordered_length_change_ratio),
        abs(left.ordered_topology_change_ratio - right.ordered_topology_change_ratio),
        _maximum_fraction_difference(left.relative_angle_fractions, right.relative_angle_fractions),
        abs(left.relative_angle_change_ratio - right.relative_angle_change_ratio),
        abs(left.paint_order_jump_ratio - right.paint_order_jump_ratio),
        token_variation,
        transition_variation,
    )
    return max(0.01, 1.0 - sum(differences) / len(differences))


def inspect_compound_path_profile_similarity(
    page: PageIR,
    left_operation_indices: Sequence[int],
    right_operation_indices: Sequence[int],
) -> tuple[CompoundPathProfile | None, CompoundPathProfile | None, float]:
    """Expose the exact profiles used by recognition for parity diagnostics."""

    paths_by_order = _path_index(page)
    left = _compound_path_profile(paths_by_order, left_operation_indices)
    right = _compound_path_profile(paths_by_order, right_operation_indices)
    return (
        left,
        right,
        _compound_profile_similarity(left, right) if left and right else 0.0,
    )


def _trim_compound_periodic_fringes(
    operation_indices: Sequence[int],
    profile: CompoundPathProfile,
) -> tuple[int, ...]:
    if (
        profile.family == "point_carrier"
        or len(profile.ordered_tokens) < 8
        or len(profile.ordered_motif) > 6
        or profile.ordered_motif_confidence < 0.55
        or profile.segment_median > 3.5
        or profile.curve_ratio > 0.10
    ):
        return tuple(operation_indices)
    variants: list[tuple[int, ...]] = []
    for source in (profile.ordered_motif, tuple(reversed(profile.ordered_motif))):
        for offset in range(len(source)):
            variants.append(source[offset:] + source[:offset])
    best: tuple[int, int, int, int] | None = None  # start, end, matches, score
    for variant in variants:
        start = 0
        score = 0
        matches = 0
        for index, token in enumerate(profile.ordered_tokens):
            matched = token == variant[index % len(variant)]
            score += 1 if matched else -2
            matches += 1 if matched else 0
            if score <= 0:
                start = index + 1
                score = 0
                matches = 0
                continue
            candidate = (start, index, matches, score)
            if (
                best is None
                or candidate[3] > best[3]
                or (
                    candidate[3] == best[3]
                    and candidate[1] - candidate[0] > best[1] - best[0]
                )
            ):
                best = candidate
    if best is None:
        return tuple(operation_indices)
    start, end, matches, _ = best
    core_length = end - start + 1
    removed_count = len(operation_indices) - core_length
    core_confidence = matches / core_length
    removed_tokens = tuple(
        token
        for index, token in enumerate(profile.ordered_tokens)
        if index < start or index > end
    )
    removes_complex_glyph = any((token % 12) // 4 >= 2 for token in removed_tokens)
    if (
        removed_count < 2
        or removes_complex_glyph
        or core_length < max(5, len(profile.ordered_motif) * 2)
        or core_length / len(operation_indices) < 0.60
        or core_confidence < 0.72
        or core_confidence - profile.ordered_motif_confidence < 0.10
    ):
        return tuple(operation_indices)
    return tuple(operation_indices[start : end + 1])


def _compound_cluster_key(run: _CompoundPathRun) -> str:
    profile = run.profile
    if profile.family == "point_carrier":
        return _LOCAL_KEY_SEPARATOR.join((run.style, profile.family))
    shape = (
        "carrier"
        if profile.straight_ratio >= 0.50
        else (
            f"shape_{_js_round(profile.aspect_ratio * 3.0)}_"
            + (
                "simple"
                if profile.segment_median <= 2.5
                else "compound"
                if profile.segment_median <= 6.0
                else "complex"
            )
        )
    )
    return _LOCAL_KEY_SEPARATOR.join((
        run.style,
        profile.family,
        str(_js_round(profile.straight_ratio * 2.0)),
        str(_js_round(profile.closed_ratio * 2.0)),
        shape,
    ))


def _is_slender_open_complex_motif(profile: CompoundPathProfile) -> bool:
    return (
        profile.zero_ratio == 0.0
        and profile.curve_ratio <= 0.15
        and profile.closed_ratio <= 0.10
        and profile.straight_ratio <= 0.20
        and profile.aspect_ratio <= 0.30
        and 6.0 <= profile.segment_median <= 12.0
    )


def _compound_global_cluster_key(run: _CompoundPathRun) -> str:
    profile = run.profile
    alternating_open_closed_route = (
        profile.family == "equal_dash"
        and 0.30 <= profile.curve_ratio <= 0.70
        and 0.30 <= profile.closed_ratio <= 0.70
        and 0.30 <= profile.straight_ratio <= 0.70
        and abs(profile.curve_ratio - profile.closed_ratio) <= 0.12
        and profile.ordered_topology_change_ratio >= 0.70
        and profile.relative_angle_change_ratio >= 0.70
        and _reliable_ordered_motif(profile)
        and len(profile.ordered_motif) == 2
    )
    if alternating_open_closed_route:
        paint = "|".join(run.style.split("|")[:2])
        return _LOCAL_KEY_SEPARATOR.join((
            paint,
            profile.family,
            "alternating_open_closed_route",
            "_".join(str(token) for token in profile.ordered_motif),
        ))
    if _is_slender_open_complex_motif(profile):
        return _LOCAL_KEY_SEPARATOR.join((
            run.style,
            "open_slender_complex_motif",
            f"segments_{_js_round(profile.segment_median / 2.0) * 2}",
        ))
    open_angular_alternation = (
        profile.family in {"equal_dash", "short_long"}
        and profile.curve_ratio <= 0.10
        and profile.closed_ratio <= 0.20
        and 0.20 < profile.straight_ratio < 0.80
        and 2.5 < profile.segment_median <= 6.0
    )
    if not open_angular_alternation:
        return _compound_cluster_key(run)
    paint = "|".join(run.style.split("|")[:2])
    length_band = (
        f"ratio_{_js_round(math.log2(max(1.0, profile.long_ratio)))}"
        if profile.family == "short_long"
        else "uniform"
    )
    glyph_complexity = (
        "complex_glyph" if profile.complex_path_ratio >= 0.18 else "simple_glyph"
    )
    return _LOCAL_KEY_SEPARATOR.join((
        paint,
        profile.family,
        "open_angular_alternation",
        length_band,
        glyph_complexity,
    ))


def _compound_local_identity_similarity(
    left: _CompoundLocalIdentity,
    right: _CompoundLocalIdentity,
) -> float:
    if not any(key in right.broad_keys for key in left.broad_keys):
        return 0.0
    allow_slender_family_change = (
        all(_is_slender_open_complex_motif(run.profile) for run in left.runs)
        and all(_is_slender_open_complex_motif(run.profile) for run in right.runs)
    )

    def directed_similarity(
        source: Sequence[_CompoundPathRun],
        target: Sequence[_CompoundPathRun],
    ) -> float:
        total_ops = 0
        matched_ops = 0
        weighted = 0.0
        strongest = 0.0
        weakest = math.inf
        for source_run in source:
            best = max(
                (
                    _compound_profile_similarity(
                        source_run.profile,
                        target_run.profile,
                        allow_slender_family_change,
                    )
                    for target_run in target
                ),
                default=0.0,
            )
            operation_count = len(source_run.op_indices)
            total_ops += operation_count
            weakest = min(weakest, best)
            if best <= 0.0:
                continue
            matched_ops += operation_count
            weighted += operation_count * best
            strongest = max(strongest, best)
        if total_ops <= 0 or matched_ops <= 0:
            return 0.0
        if matched_ops == total_ops:
            return weakest
        if matched_ops <= total_ops * _COMPOUND_MATCHED_MASS_MINIMUM:
            return 0.0
        if strongest < _COMPOUND_STRONG_MATCH_MINIMUM:
            return 0.0
        return weighted / total_ops

    return min(
        directed_similarity(left.runs, right.runs),
        directed_similarity(right.runs, left.runs),
    )


def _renumber_global_type(global_type: GlobalLineType, index: int) -> GlobalLineType:
    return replace(global_type, global_type_id=f"global_type_{index + 1:03d}")


def _build_global_type(
    base: GlobalLineType | None,
    members: Sequence[GlobalLineTypeMember],
    local_by_key: Mapping[str, LocalLineType],
    signature_family: str | None = None,
    minimum_similarity: float | None = None,
) -> GlobalLineType:
    member_tuple = tuple(members)
    operation_indices = _sorted_unique_indices(
        index
        for member in member_tuple
        for index in local_by_key[
            _local_type_key(member.case_id, member.type_id)
        ].op_indices
    )
    if base is None:
        return GlobalLineType(
            global_type_id="",
            signature_family=signature_family or "motif_periodic",
            minimum_pair_similarity=1.0 if minimum_similarity is None else minimum_similarity,
            op_indices=operation_indices,
            members=member_tuple,
        )
    return replace(
        base,
        signature_family=signature_family or base.signature_family,
        minimum_pair_similarity=(
            base.minimum_pair_similarity
            if minimum_similarity is None
            else minimum_similarity
        ),
        op_indices=operation_indices,
        members=member_tuple,
    )


def _discover_compound_runs_for_task(
    task: _CompoundRunTask,
) -> tuple[_CompoundPathRun, ...]:
    """Profile one pre-separated run without shared state or reordering."""

    group_id = task.group_id
    paths_by_order = dict(task.paths)
    assigned = set(task.assigned_path_indices)
    runs: list[_CompoundPathRun] = []
    current_style = task.style
    current_operations = list(task.operation_indices)

    def finish_run() -> None:
        nonlocal current_style, current_operations
        whole_profile = _compound_path_profile(paths_by_order, current_operations)
        spatial_runs = (
            _split_spatial_route_runs(
                paths_by_order,
                current_operations,
                whole_profile.family,
            )
            if whole_profile is not None and whole_profile.family != "point_carrier"
            else ()
        )
        if whole_profile is None:
            candidate_runs = (tuple(current_operations),) if current_operations else ()
        elif whole_profile.family == "point_carrier":
            candidate_runs = _split_spatial_route_runs(
                paths_by_order,
                current_operations,
            )
        elif len(spatial_runs) > 1 and all(len(run) >= 4 for run in spatial_runs):
            candidate_runs = spatial_runs
        else:
            candidate_runs = (tuple(current_operations),) if current_operations else ()

        for spatial_run in candidate_runs:
            operation_indices = _trim_isolated_compound_outliers(
                paths_by_order,
                spatial_run,
            )
            retained_set = set(operation_indices)
            excluded_indices = tuple(
                index for index in spatial_run if index not in retained_set
            )
            profile = _compound_path_profile(paths_by_order, operation_indices)
            if (
                profile is not None
                and profile.family == "point_carrier"
                and not _has_point_carrier_core(paths_by_order, operation_indices)
            ):
                continue
            if (
                profile is not None
                and profile.family != "point_carrier"
                and not _is_extendable_route_run(paths_by_order, operation_indices)
            ):
                continue
            assigned_operations = tuple(
                index for index in spatial_run if index in assigned
            )
            assigned_profile = _compound_path_profile(
                paths_by_order,
                assigned_operations,
            )
            if profile is not None:
                periodic_core = _trim_compound_periodic_fringes(
                    operation_indices,
                    profile,
                )
                if len(periodic_core) < len(operation_indices):
                    operation_indices = periodic_core
                    retained_set = set(operation_indices)
                    excluded_indices = tuple(
                        index for index in spatial_run if index not in retained_set
                    )
                    profile = (
                        _compound_path_profile(paths_by_order, operation_indices)
                        or profile
                    )
            if (
                profile is not None
                and profile.family == "short_long"
                and assigned_profile is not None
                and assigned_profile.family == "equal_dash"
                and len(assigned_operations) / len(spatial_run) >= 0.50
                and _is_slender_open_complex_motif(assigned_profile)
            ):
                typical_length = _median_number(
                    _path_ink_length(paths_by_order[index])
                    if index in paths_by_order
                    else 0.0
                    for index in assigned_operations
                )
                typical_segments = _median_number(
                    len(paths_by_order[index].segments)
                    if index in paths_by_order
                    else 0
                    for index in assigned_operations
                )
                filtered: list[int] = []
                for index in spatial_run:
                    if index in assigned:
                        filtered.append(index)
                        continue
                    operation = paths_by_order.get(index)
                    if operation is None:
                        continue
                    length_ratio = _path_ink_length(operation) / max(
                        0.001,
                        typical_length,
                    )
                    has_curve = any(
                        segment.kind == "curve" for segment in operation.segments
                    )
                    if (
                        0.60 <= length_ratio <= 1.45
                        and abs(len(operation.segments) - typical_segments) <= 2.0
                        and not has_curve
                    ):
                        filtered.append(index)
                operation_indices = tuple(filtered)
                retained_set = set(operation_indices)
                excluded_indices = tuple(
                    index for index in spatial_run if index not in retained_set
                )
                profile = (
                    _compound_path_profile(paths_by_order, operation_indices)
                    or assigned_profile
                )
            if profile is not None:
                runs.append(_CompoundPathRun(
                    group_id=group_id,
                    style=current_style,
                    op_indices=tuple(operation_indices),
                    excluded_op_indices=tuple(excluded_indices),
                    profile=profile,
                    assigned_count=sum(index in assigned for index in operation_indices),
                ))
        current_style = ""
        current_operations = []

    finish_run()
    return tuple(runs)


def augment_compound_path_line_types(
    page: PageIR,
    grouping: GroupingIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
    *,
    worker_count: int | None = None,
) -> LineTypeRecognitionResult:
    """Complete seeded multi-operation path periods and cluster complete-link."""

    operation_index = PageOperationIndex.build(page, grouping)
    paths_by_order = dict(operation_index.path_items())
    recognized_by_group = {group.group_id: group for group in recognition.groups}
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    tasks: list[_CompoundRunTask] = []
    for grouping_group in grouping.groups:
        group_id = grouping_group.group_id
        recognized = recognized_by_group.get(group_id)
        serialized = serialized_by_group.get(group_id)
        if recognized is None or serialized is None:
            continue
        dense_indices = operation_index.group_indices(group_id)
        atom_operations = set(serialized.atom_op_indices)
        assigned = {
            index
            for line_type in recognized.line_types
            for index in line_type.op_indices
            if index in paths_by_order
        }
        current_style = ""
        current_operations: list[int] = []

        def queue_run() -> None:
            nonlocal current_style, current_operations
            if current_operations:
                tasks.append(_CompoundRunTask(
                    group_id=group_id,
                    style=current_style,
                    paths=tuple(
                        (index, paths_by_order[index])
                        for index in current_operations
                    ),
                    operation_indices=tuple(current_operations),
                    assigned_path_indices=tuple(
                        index for index in current_operations if index in assigned
                    ),
                ))
            current_style = ""
            current_operations = []

        for dense_index in dense_indices:
            operation = operation_index.operation(dense_index)
            if (
                not isinstance(operation, PathOperationIR)
                or dense_index not in atom_operations
            ):
                queue_run()
                continue
            style = _path_style_key(operation)
            previous = (
                paths_by_order.get(current_operations[-1])
                if current_operations
                else None
            )
            length_ratio = (
                max(_path_ink_length(previous), _path_ink_length(operation))
                / max(
                    0.001,
                    min(_path_ink_length(previous), _path_ink_length(operation)),
                )
                if previous is not None
                else 1.0
            )
            spatial_break = (
                previous is not None
                and len(previous.segments) <= 3
                and len(operation.segments) <= 3
                and length_ratio > 8.0
                and _bounds_gap(previous.bounds, operation.bounds)
                > max(6.0, max(previous.line_width, operation.line_width) * 16.0)
            )
            if current_operations and (style != current_style or spatial_break):
                queue_run()
            if not current_operations:
                current_style = style
            current_operations.append(dense_index)
        queue_run()

    workers = _compound_worker_budget(len(tasks), len(paths_by_order), worker_count)
    if workers == 1:
        discovered = map(_discover_compound_runs_for_task, tasks)
    else:
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            discovered = executor.map(
                _discover_compound_runs_for_task,
                tasks,
                chunksize=chunksize,
            )
            # Materialize before shutting the executor down; map preserves the
            # exact input Group order independently of worker completion order.
            runs = [run for group_runs in discovered for run in group_runs]
    if workers == 1:
        runs = [run for group_runs in discovered for run in group_runs]

    selected_ids: set[int] = set()
    selected_order: list[_CompoundPathRun] = []

    def select(run: _CompoundPathRun) -> None:
        identity = id(run)
        if identity not in selected_ids:
            selected_ids.add(identity)
            selected_order.append(run)

    for run in runs:
        if run.assigned_count >= 3:
            select(run)
    seeded_groups = {run.group_id for run in selected_order}
    seedless_skipped_operation_count = sum(
        len(run.op_indices) for run in runs if run.group_id not in seeded_groups
    )
    if not selected_order:
        return replace(
            recognition,
            summary=replace(
                recognition.summary,
                seedless_skipped_op_count=seedless_skipped_operation_count,
            ),
        )

    by_group_cluster: dict[str, list[_CompoundPathRun]] = {}
    for run in runs:
        key = f"{run.group_id}{_LOCAL_KEY_SEPARATOR}{_compound_cluster_key(run)}"
        by_group_cluster.setdefault(key, []).append(run)
    for candidates in by_group_cluster.values():
        seeds = tuple(run for run in candidates if id(run) in selected_ids)
        if len(seeds) < 2:
            continue
        for candidate in candidates:
            if any(
                _compound_profile_similarity(seed.profile, candidate.profile) > 0.0
                for seed in seeds
            ):
                select(candidate)

    page_seeds: dict[str, list[_CompoundPathRun]] = {}
    for run in selected_order:
        page_seeds.setdefault(_compound_cluster_key(run), []).append(run)
    for key, seeds in page_seeds.items():
        group_count = len({run.group_id for run in seeds})
        operation_count = sum(len(run.op_indices) for run in seeds)
        if group_count < 2 or operation_count < 10:
            continue
        for candidate in runs:
            if id(candidate) in selected_ids or _compound_cluster_key(candidate) != key:
                continue
            if any(
                _compound_profile_similarity(seed.profile, candidate.profile) > 0.0
                for seed in seeds
            ):
                select(candidate)

    selected_by_group: dict[str, dict[str, list[_CompoundPathRun]]] = {}
    for run in selected_order:
        clusters = selected_by_group.setdefault(run.group_id, {})
        clusters.setdefault(_compound_cluster_key(run), []).append(run)

    compound_local_identities: list[_CompoundLocalIdentity] = []
    updated_groups: list[RecognizedGroup] = []
    for group in recognition.groups:
        clusters = selected_by_group.get(group.group_id)
        serialized = serialized_by_group.get(group.group_id)
        if clusters is None or serialized is None:
            updated_groups.append(group)
            continue
        selected_operations = {
            index
            for cluster_runs in clusters.values()
            for run in cluster_runs
            for index in run.op_indices
        }
        excluded_operations = {
            index
            for cluster_runs in clusters.values()
            for run in cluster_runs
            for index in run.excluded_op_indices
        }
        used_type_ids = {line_type.type_id for line_type in group.line_types}
        next_compound = 1

        def fresh_compound_type_id() -> str:
            nonlocal next_compound
            while True:
                candidate = f"type_compound_{next_compound:03d}"
                next_compound += 1
                if candidate not in used_type_ids:
                    used_type_ids.add(candidate)
                    return candidate

        retained: list[LocalLineType] = []
        for line_type in group.line_types:
            removed = any(index in selected_operations for index in line_type.op_indices)
            if not removed:
                retained.append(line_type)
                continue
            operation_indices = tuple(
                index
                for index in line_type.op_indices
                if index not in selected_operations and index not in excluded_operations
            )
            path_set = {index for index in operation_indices if index in paths_by_order}
            atom_count = _atom_count_for_ops(serialized, path_set)
            if atom_count > 0:
                retained.append(replace(
                    line_type,
                    atom_count=atom_count,
                    op_indices=_sorted_unique_indices(operation_indices),
                ))

        compound: list[LocalLineType] = []
        for cluster_runs in clusters.values():
            path_set = {
                index for run in cluster_runs for index in run.op_indices
            }
            source_type = max(
                group.line_types,
                key=lambda line_type: sum(
                    index in path_set for index in line_type.op_indices
                ),
                default=None,
            )
            type_id = fresh_compound_type_id()
            compound.append(LocalLineType(
                type_id=type_id,
                display_name="",
                line_type_index=0,
                atom_count=_atom_count_for_ops(serialized, path_set),
                op_indices=_sorted_unique_indices(path_set),
                model="compound_path_chain",
                shape=source_type.shape if source_type is not None else "非直线",
                shape_detail=(
                    source_type.shape_detail
                    if source_type is not None
                    else "复合路径周期"
                ),
            ))
            local_key = _local_type_key(group.group_id, type_id)
            compound_local_identities.append(_CompoundLocalIdentity(
                local_key=local_key,
                broad_keys=tuple(dict.fromkeys(
                    _compound_global_cluster_key(run) for run in cluster_runs
                )),
                runs=tuple(cluster_runs),
                op_indices=_sorted_unique_indices(path_set),
            ))

        line_types = tuple(
            replace(line_type, display_name=f"线型{index + 1}", line_type_index=index + 1)
            for index, line_type in enumerate(sorted(
                (*retained, *compound),
                key=lambda item: min(item.op_indices, default=math.inf),
            ))
        )
        assigned_paths = {
            index
            for line_type in line_types
            for index in line_type.op_indices
            if index in paths_by_order
        }
        non_line_paths = {
            index
            for index in serialized.atom_op_indices
            if index not in assigned_paths
        }
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=_atom_count_for_ops(serialized, non_line_paths),
                op_indices=_sorted_unique_indices(non_line_paths),
            ),
        ))

    local_by_key = _index_local_types(updated_groups)
    identity_similarities = _precompute_compound_identity_similarities(
        compound_local_identities,
        worker_count,
    )
    identity_index = {
        id(identity): index for index, identity in enumerate(compound_local_identities)
    }
    identity_clusters: list[list[_CompoundLocalIdentity]] = []
    for identity in compound_local_identities:
        current_index = identity_index[id(identity)]
        choices: list[tuple[int, tuple[float, ...]]] = []
        for index, cluster in enumerate(identity_clusters):
            similarities = tuple(
                identity_similarities.similarity(
                    current_index,
                    identity_index[id(member)],
                )
                for member in cluster
            )
            if all(similarity > 0.0 for similarity in similarities):
                choices.append((index, similarities))
        choices.sort(key=lambda choice: (-min(choice[1]), choice[0]))
        if choices:
            identity_clusters[choices[0][0]].append(identity)
        else:
            identity_clusters.append([identity])

    cluster_local_keys: dict[str, tuple[str, ...]] = {}
    cluster_operation_indices: dict[str, set[int]] = {}
    cluster_minimum_similarity: dict[str, float] = {}
    for index, cluster in enumerate(identity_clusters):
        cluster_key = f"compound_complete_link_{index + 1}"
        cluster_local_keys[cluster_key] = tuple(identity.local_key for identity in cluster)
        cluster_operation_indices[cluster_key] = {
            operation_index
            for identity in cluster
            for operation_index in identity.op_indices
        }
        similarities = tuple(
            identity_similarities.similarity(
                identity_index[id(left)],
                identity_index[id(right)],
            )
            for left_index, left in enumerate(cluster)
            for right in cluster[left_index + 1 :]
        )
        cluster_minimum_similarity[cluster_key] = min(similarities, default=1.0)

    original_global_operations = {
        global_type.global_type_id: set(global_type.op_indices)
        for global_type in recognition.global_types
    }
    cluster_preferences = []
    for cluster_key, operation_indices in cluster_operation_indices.items():
        overlaps = [
            (
                global_type,
                sum(
                    index in original_global_operations[global_type.global_type_id]
                    for index in operation_indices
                ),
            )
            for global_type in recognition.global_types
        ]
        overlaps.sort(key=lambda item: -item[1])
        cluster_preferences.append((cluster_key, overlaps[0] if overlaps else None))
    cluster_preferences.sort(key=lambda item: -(item[1][1] if item[1] else 0))

    target_by_cluster: dict[str, str] = {}
    used_targets: set[str] = set()
    for cluster_key, _ in cluster_preferences:
        candidates = [
            (
                global_type.global_type_id,
                sum(
                    index in original_global_operations[global_type.global_type_id]
                    for index in cluster_operation_indices.get(cluster_key, set())
                ),
            )
            for global_type in recognition.global_types
        ]
        candidates.sort(key=lambda item: -item[1])
        target = next(
            (
                candidate
                for candidate in candidates
                if candidate[1] > 0 and candidate[0] not in used_targets
            ),
            None,
        )
        if target is not None:
            target_by_cluster[cluster_key] = target[0]
            used_targets.add(target[0])
    cluster_by_target = {
        target: cluster_key for cluster_key, target in target_by_cluster.items()
    }

    def member_for_local_key(local_key: str) -> GlobalLineTypeMember:
        case_id, type_id = local_key.split(_LOCAL_KEY_SEPARATOR, 1)
        line_type = local_by_key[local_key]
        return GlobalLineTypeMember(
            case_id=case_id,
            type_id=type_id,
            display_name=line_type.display_name,
            atom_count=line_type.atom_count,
            shape=line_type.shape,
            model=line_type.model,
            shape_detail=line_type.shape_detail,
        )

    updated_globals: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        retained_members = tuple(
            member
            for member in global_type.members
            if _local_type_key(member.case_id, member.type_id) in local_by_key
        )
        cluster_key = cluster_by_target.get(global_type.global_type_id)
        if cluster_key is None:
            if retained_members:
                updated_globals.append(_build_global_type(
                    global_type,
                    retained_members,
                    local_by_key,
                ))
            continue
        cluster_members = tuple(
            member_for_local_key(local_key)
            for local_key in cluster_local_keys.get(cluster_key, ())
        )
        updated_globals.append(_build_global_type(
            global_type,
            cluster_members,
            local_by_key,
            "compound_path_periodic",
            cluster_minimum_similarity.get(cluster_key, 1.0),
        ))
        if retained_members:
            updated_globals.append(_build_global_type(
                replace(global_type, global_type_id=""),
                retained_members,
                local_by_key,
            ))
    for cluster_key, local_keys in cluster_local_keys.items():
        if cluster_key in target_by_cluster:
            continue
        updated_globals.append(_build_global_type(
            None,
            tuple(member_for_local_key(local_key) for local_key in local_keys),
            local_by_key,
            "compound_path_periodic",
            cluster_minimum_similarity.get(cluster_key, 1.0),
        ))

    return replace(
        recognition,
        groups=tuple(updated_groups),
        global_types=tuple(
            _renumber_global_type(global_type, index)
            for index, global_type in enumerate(updated_globals)
        ),
        summary=replace(
            recognition.summary,
            seedless_skipped_op_count=seedless_skipped_operation_count,
        ),
    )


def enforce_extendable_route_line_types(
    page: PageIR,
    serialized_groups: Sequence[SerializedGroup],
    recognition: LineTypeRecognitionResult,
) -> LineTypeRecognitionResult:
    """Demote generic candidates with no spatially extendable route core."""

    paths_by_order = _path_index(page)
    serialized_by_group = {group.group_id: group for group in serialized_groups}
    discarded_route_operation_count = 0
    updated_groups: list[RecognizedGroup] = []

    for group in recognition.groups:
        serialized = serialized_by_group.get(group.group_id)
        if serialized is None:
            updated_groups.append(group)
            continue
        retained_line_types: list[LocalLineType] = []
        for line_type in group.line_types:
            if line_type.model not in {"single_carrier_chain", "compound_path_chain"}:
                retained_line_types.append(line_type)
                continue
            retained_operations: list[int] = []
            for run in _split_spatial_route_runs(
                paths_by_order,
                line_type.op_indices,
            ):
                profile = _compound_path_profile(paths_by_order, run)
                if profile is not None and profile.family == "point_carrier":
                    if _has_point_carrier_core(paths_by_order, run):
                        retained_operations.extend(run)
                    # Frozen r10 counts generic rejected route pieces only;
                    # preserve that diagnostic behaviour for parity.
                    continue
                if _is_extendable_route_run(paths_by_order, run):
                    retained_operations.extend(run)
                else:
                    discarded_route_operation_count += len(run)
            operation_indices = _sorted_unique_indices(retained_operations)
            if not operation_indices:
                continue
            path_set = {index for index in operation_indices if index in paths_by_order}
            retained_line_types.append(replace(
                line_type,
                atom_count=_atom_count_for_ops(serialized, path_set),
                op_indices=operation_indices,
            ))

        line_types = tuple(
            replace(line_type, display_name=f"线型{index + 1}", line_type_index=index + 1)
            for index, line_type in enumerate(retained_line_types)
        )
        assigned = {
            index
            for line_type in line_types
            for index in line_type.op_indices
            if index in paths_by_order
        }
        non_line = {
            index for index in serialized.atom_op_indices if index not in assigned
        }
        updated_groups.append(replace(
            group,
            line_types=line_types,
            non_linetype=replace(
                group.non_linetype,
                atom_count=_atom_count_for_ops(serialized, non_line),
                op_indices=_sorted_unique_indices(non_line),
            ),
        ))

    local_by_key = _index_local_types(updated_groups)
    global_types: list[GlobalLineType] = []
    for global_type in recognition.global_types:
        members: list[GlobalLineTypeMember] = []
        for member in global_type.members:
            local = local_by_key.get(_local_type_key(member.case_id, member.type_id))
            if local is None:
                continue
            members.append(replace(
                member,
                display_name=local.display_name,
                atom_count=local.atom_count,
                model=local.model,
                shape=local.shape,
                shape_detail=local.shape_detail,
            ))
        if members:
            global_types.append(_build_global_type(
                global_type,
                members,
                local_by_key,
            ))
    renumbered_globals = tuple(
        _renumber_global_type(global_type, index)
        for index, global_type in enumerate(global_types)
    )
    local_line_type_count = sum(len(group.line_types) for group in updated_groups)
    return replace(
        recognition,
        groups=tuple(updated_groups),
        global_types=renumbered_globals,
        summary=replace(
            recognition.summary,
            discarded_route_op_count=discarded_route_operation_count,
            local_line_type_count=local_line_type_count,
            signed_periodic_type_count=local_line_type_count,
            unsigned_periodic_type_count=0,
            global_type_count=len(renumbered_globals),
            cross_group_global_type_count=sum(
                global_type.group_count > 1 for global_type in renumbered_globals
            ),
        ),
    )
