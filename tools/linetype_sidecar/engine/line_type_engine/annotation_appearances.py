"""Mechanical selection and placement of visible annotation appearances.

The browser PDF extractor appends each eligible normal appearance as a
synthetic ``q /__AnnotationN Do Q`` after the page ``/Contents`` streams.
This module mirrors only the PDF-object selection and matrix calculation.  It
does not parse paths, group operations, or recognize line types.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, TypeAlias


Matrix: TypeAlias = tuple[float, float, float, float, float, float]
Bounds: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class AnnotationAppearancePlan:
    """One visible normal appearance in page annotation-array order."""

    annotation_index: int
    resource_name: str
    resource_id: str
    subtype: str
    appearance_reference: Any
    appearance_stream: Any
    matrix: Matrix
    bbox: Bounds


def _object(value: Any) -> Any:
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _number(value: Any) -> float | None:
    value = _object(value)
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _array_numbers(value: Any, count: int) -> tuple[float, ...] | None:
    value = _object(value)
    if value is None or not hasattr(value, "__len__") or len(value) < count:
        return None
    numbers = tuple(_number(value[index]) for index in range(count))
    if any(number is None for number in numbers):
        return None
    return numbers  # type: ignore[return-value]


def _name(value: Any) -> str | None:
    value = _object(value)
    if type(value).__name__ != "NameObject":
        return None
    return str(value).lstrip("/")


def _is_dictionary(value: Any) -> bool:
    return value is not None and hasattr(value, "get")


def _is_stream(value: Any) -> bool:
    return _is_dictionary(value) and callable(getattr(value, "get_data", None))


def _matrix_from_stream(stream: Any) -> Matrix:
    values = _array_numbers(stream.get("/Matrix"), 6)
    return (values or (1.0, 0.0, 0.0, 1.0, 0.0, 0.0))  # type: ignore[return-value]


def _bounds_from_stream(stream: Any) -> Bounds:
    values = _array_numbers(stream.get("/BBox"), 4) or (0.0, 0.0, 1.0, 1.0)
    return (
        min(values[0], values[2]),
        min(values[1], values[3]),
        max(values[0], values[2]),
        max(values[1], values[3]),
    )


def _multiply(left: Matrix, right: Matrix) -> Matrix:
    return (
        left[0] * right[0] + left[2] * right[1],
        left[1] * right[0] + left[3] * right[1],
        left[0] * right[2] + left[2] * right[3],
        left[1] * right[2] + left[3] * right[3],
        left[0] * right[4] + left[2] * right[5] + left[4],
        left[1] * right[4] + left[3] * right[5] + left[5],
    )


def _transformed_bounds(matrix: Matrix, bounds: Bounds) -> Bounds:
    points = tuple(
        (
            matrix[0] * x + matrix[2] * y + matrix[4],
            matrix[1] * x + matrix[3] * y + matrix[5],
        )
        for x, y in (
            (bounds[0], bounds[1]),
            (bounds[2], bounds[1]),
            (bounds[2], bounds[3]),
            (bounds[0], bounds[3]),
        )
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _placed_matrix(stream: Any, rectangle: tuple[float, ...] | None) -> tuple[Matrix, Bounds]:
    matrix = _matrix_from_stream(stream)
    bbox = _bounds_from_stream(stream)
    if rectangle is None:
        return matrix, bbox
    rect_bounds = (
        min(rectangle[0], rectangle[2]),
        min(rectangle[1], rectangle[3]),
        max(rectangle[0], rectangle[2]),
        max(rectangle[1], rectangle[3]),
    )
    painted = _transformed_bounds(matrix, bbox)
    painted_width = max(0.000001, painted[2] - painted[0])
    painted_height = max(0.000001, painted[3] - painted[1])
    scale_x = (rect_bounds[2] - rect_bounds[0]) / painted_width
    scale_y = (rect_bounds[3] - rect_bounds[1]) / painted_height
    placement: Matrix = (
        scale_x,
        0.0,
        0.0,
        scale_y,
        rect_bounds[0] - painted[0] * scale_x,
        rect_bounds[1] - painted[1] * scale_y,
    )
    return _multiply(placement, matrix), bbox


def annotation_appearance_plans(page: Any) -> tuple[AnnotationAppearancePlan, ...]:
    """Return exactly the normal appearances appended by the frozen TS reader.

    Invisible, hidden and no-view annotations (flags 1, 2 and 32), malformed
    dictionaries, absent appearance states and non-Form streams are skipped.
    Array position, rather than accepted-count position, determines the
    synthetic resource name.
    """

    annotations = _object(page.get("/Annots"))
    if annotations is None or not hasattr(annotations, "__iter__"):
        return ()
    plans: list[AnnotationAppearancePlan] = []
    for index, raw_annotation in enumerate(annotations):
        annotation = _object(raw_annotation)
        if not _is_dictionary(annotation) or _is_stream(annotation):
            continue
        flags_number = _number(annotation.get("/F"))
        flags = int(flags_number) if flags_number is not None else 0
        if flags & (1 | 2 | 32):
            continue

        appearances = _object(annotation.get("/AP"))
        if not _is_dictionary(appearances) or _is_stream(appearances):
            continue
        raw_normal = appearances.get("/N")
        normal = _object(raw_normal)
        selected_reference: Any | None = raw_normal if _is_stream(normal) else None
        stream: Any | None = normal if _is_stream(normal) else None
        if stream is None and _is_dictionary(normal):
            state_name = _name(annotation.get("/AS")) or "Off"
            state_value = normal.get(f"/{state_name}")
            selected = _object(state_value)
            if _is_stream(selected):
                selected_reference = state_value
                stream = selected
        if stream is None or _name(stream.get("/Subtype")) != "Form":
            continue

        rectangle = _array_numbers(annotation.get("/Rect"), 4)
        matrix, bbox = _placed_matrix(stream, rectangle)
        if hasattr(raw_annotation, "idnum"):
            resource_id = (
                f"{int(raw_annotation.idnum)} "
                f"{int(getattr(raw_annotation, 'generation', 0) or 0)} R:appearance"
            )
        else:
            resource_id = f"direct:annotation:{index}:appearance"
        plans.append(AnnotationAppearancePlan(
            annotation_index=index,
            resource_name=f"__Annotation{index + 1}",
            resource_id=resource_id,
            subtype=_name(annotation.get("/Subtype")) or "",
            appearance_reference=(
                selected_reference if selected_reference is not None else stream
            ),
            appearance_stream=stream,
            matrix=matrix,
            bbox=bbox,
        ))
    return tuple(plans)


__all__ = ["AnnotationAppearancePlan", "annotation_appearance_plans"]
