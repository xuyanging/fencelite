"""Native PyMuPDF adapter for the canonical :mod:`line_type_engine.ir` model.

PyMuPDF exposes page geometry in an unrotated, top-left coordinate system.
The recognition IR deliberately uses page-local PDF coordinates instead:
``(0, 0)`` is the bottom-left of the crop box and page rotation is retained as
metadata.  No recognition or grouping decision belongs in this module.

``get_drawings(extended=True)`` normalizes rectangle and quadrilateral PDF
operators into ``re`` and ``qu`` items.  PageIR treats the spelling of a PDF
operator as parser provenance, not pattern identity, so those items are
losslessly expanded to move / line / close segments.  Curves remain cubic.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from hashlib import sha256
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .ir import (
    BoundsIR,
    ImageOperationIR,
    PageIR,
    PathOperationIR,
    PathSegmentIR,
    PointIR,
    TextCharacterIR,
    TextOperationIR,
)
from .runtime import assert_supported_pymupdf_runtime


PDF_ADAPTER_VERSION = "pymupdf-page-ir-r2-text-trace-audit-2026-08-24"


class PdfAdapterError(RuntimeError):
    """Raised when PyMuPDF data cannot be represented without guessing."""


@lru_cache(maxsize=1)
def _pymupdf() -> Any:
    assert_supported_pymupdf_runtime()
    try:
        import pymupdf  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - guarded above.
        raise PdfAdapterError(
            "PyMuPDF is required to extract PageIR (install PyMuPDF>=1.28.2,<2)"
        ) from error
    return pymupdf


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise PdfAdapterError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise PdfAdapterError(f"{label} must be finite, got {value!r}")
    return 0.0 if number == 0 else number


def _point(value: Any, page_height: float, label: str = "point") -> PointIR:
    try:
        x, y = value[0], value[1]
    except (IndexError, KeyError, TypeError) as error:
        raise PdfAdapterError(f"{label} is not a two-dimensional point: {value!r}") from error
    x_value = _finite(x, f"{label}.x")
    y_value = _finite(y, f"{label}.y")
    return (x_value, _finite(page_height - y_value, f"{label}.converted_y"))


def _bounds(value: Any, page_height: float, label: str = "bounds") -> BoundsIR:
    try:
        raw_x0, raw_y0, raw_x1, raw_y1 = value[:4]
    except (IndexError, KeyError, TypeError) as error:
        raise PdfAdapterError(f"{label} is not a four-coordinate rectangle: {value!r}") from error
    x0, x1 = sorted((_finite(raw_x0, f"{label}.x0"), _finite(raw_x1, f"{label}.x1")))
    y0, y1 = sorted((_finite(raw_y0, f"{label}.y0"), _finite(raw_y1, f"{label}.y1")))
    return BoundsIR(x0, page_height - y1, x1, page_height - y0)


def _color(value: Any, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        channels = tuple(
            _finite(channel, f"{label}[{index}]")
            for index, channel in enumerate(value)
        )
    except TypeError as error:
        raise PdfAdapterError(f"{label} must be a color sequence or None") from error
    return channels


def _same_point(left: PointIR | None, right: PointIR, tolerance: float = 1e-7) -> bool:
    return (
        left is not None
        and abs(left[0] - right[0]) <= tolerance
        and abs(left[1] - right[1]) <= tolerance
    )


def _quad_corners(quad: Any, page_height: float) -> tuple[PointIR, PointIR, PointIR, PointIR]:
    try:
        values = (quad.ul, quad.ur, quad.lr, quad.ll)
    except AttributeError as error:
        raise PdfAdapterError(f"PyMuPDF qu item has no UL/UR/LR/LL corners: {quad!r}") from error
    return tuple(
        _point(value, page_height, f"quadrilateral.{name}")
        for name, value in zip(("ul", "ur", "lr", "ll"), values)
    )  # type: ignore[return-value]


def _rectangle_corners(
    rectangle: Any, orientation: Any, page_height: float
) -> tuple[PointIR, PointIR, PointIR, PointIR]:
    try:
        x0, y0, x1, y1 = rectangle[:4]
    except (IndexError, KeyError, TypeError) as error:
        raise PdfAdapterError(f"PyMuPDF re item has an invalid rectangle: {rectangle!r}") from error
    x0, x1 = sorted((_finite(x0, "rectangle.x0"), _finite(x1, "rectangle.x1")))
    y0, y1 = sorted((_finite(y0, "rectangle.y0"), _finite(y1, "rectangle.y1")))
    top_left = _point((x0, y0), page_height, "rectangle.top_left")
    top_right = _point((x1, y0), page_height, "rectangle.top_right")
    bottom_right = _point((x1, y1), page_height, "rectangle.bottom_right")
    bottom_left = _point((x0, y1), page_height, "rectangle.bottom_left")
    try:
        direction = int(orientation)
    except (TypeError, ValueError) as error:
        raise PdfAdapterError(
            f"rectangle orientation is not an integer: {orientation!r}"
        ) from error
    if direction == 0:
        raise PdfAdapterError("rectangle orientation must not be zero")
    if direction > 0:
        return top_left, top_right, bottom_right, bottom_left
    return top_left, bottom_left, bottom_right, top_right


def _closed_polygon_segments(corners: Sequence[PointIR]) -> list[PathSegmentIR]:
    if len(corners) < 3:
        raise PdfAdapterError("closed polygon requires at least three corners")
    return [
        PathSegmentIR("move", corners[0]),
        *(PathSegmentIR("line", corner) for corner in corners[1:]),
        PathSegmentIR("close"),
    ]


def _path_segments(
    items: Iterable[Any], close_path: bool, page_height: float
) -> tuple[PathSegmentIR, ...]:
    segments: list[PathSegmentIR] = []
    current: PointIR | None = None
    subpath_start: PointIR | None = None

    for item_index, item in enumerate(items):
        if not isinstance(item, (tuple, list)) or not item:
            raise PdfAdapterError(f"drawing item {item_index} is malformed: {item!r}")
        command = str(item[0]).lower()
        if command == "l":
            if len(item) != 3:
                raise PdfAdapterError(f"line item must contain start and end points: {item!r}")
            start = _point(item[1], page_height, f"items[{item_index}].line_start")
            end = _point(item[2], page_height, f"items[{item_index}].line_end")
            if not _same_point(current, start):
                segments.append(PathSegmentIR("move", start))
                subpath_start = start
            segments.append(PathSegmentIR("line", end))
            current = end
            continue

        if command == "c":
            if len(item) != 5:
                raise PdfAdapterError(
                    f"cubic item must contain start, two controls and an end: {item!r}"
                )
            start = _point(item[1], page_height, f"items[{item_index}].curve_start")
            control_1 = _point(item[2], page_height, f"items[{item_index}].control_1")
            control_2 = _point(item[3], page_height, f"items[{item_index}].control_2")
            end = _point(item[4], page_height, f"items[{item_index}].curve_end")
            if not _same_point(current, start):
                segments.append(PathSegmentIR("move", start))
                subpath_start = start
            segments.append(PathSegmentIR("curve", end, control_1, control_2))
            current = end
            continue

        if command == "qu":
            if len(item) != 2:
                raise PdfAdapterError(f"quadrilateral item must contain one Quad: {item!r}")
            polygon = _closed_polygon_segments(_quad_corners(item[1], page_height))
            segments.extend(polygon)
            current = polygon[0].end
            subpath_start = current
            continue

        if command == "re":
            if len(item) != 3:
                raise PdfAdapterError(f"rectangle item must contain Rect and orientation: {item!r}")
            polygon = _closed_polygon_segments(
                _rectangle_corners(item[1], item[2], page_height)
            )
            segments.extend(polygon)
            current = polygon[0].end
            subpath_start = current
            continue

        raise PdfAdapterError(
            f"unsupported PyMuPDF drawing command {command!r} at item {item_index}"
        )

    if close_path and segments and segments[-1].kind != "close":
        segments.append(PathSegmentIR("close"))
        current = subpath_start
    if not segments:
        raise PdfAdapterError("painted drawing contains no representable path segments")
    return tuple(segments)


_DASH_PATTERN = re.compile(r"^\s*\[\s*(.*?)\s*\]\s*([^\s]+)\s*$")


def _dash(value: Any) -> tuple[tuple[float, ...], float]:
    if value is None:
        return (), 0.0
    raw = str(value)
    match = _DASH_PATTERN.fullmatch(raw)
    if match is None:
        raise PdfAdapterError(f"unsupported PyMuPDF dash syntax: {raw!r}")
    array_text, phase_text = match.groups()
    try:
        dash_array = tuple(
            _finite(token, f"dash[{index}]")
            for index, token in enumerate(array_text.split())
        )
        phase = _finite(phase_text, "dash phase")
    except PdfAdapterError:
        raise
    return dash_array, phase


def _line_cap(value: Any) -> tuple[int, int, int]:
    if value is None:
        return (0, 0, 0)
    if isinstance(value, (int, float)):
        cap = int(value)
        return (cap, cap, cap)
    try:
        caps = tuple(int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise PdfAdapterError(f"invalid line-cap tuple: {value!r}") from error
    if len(caps) != 3:
        raise PdfAdapterError(f"line-cap tuple must contain three entries: {value!r}")
    return caps  # type: ignore[return-value]


def _is_painted_path(drawing_type: Any) -> bool:
    value = str(drawing_type or "").lower()
    return bool(value) and set(value).issubset({"f", "s"})


def _path_operations(page: Any, page_height: float) -> list[PathOperationIR]:
    try:
        drawings = page.get_drawings(extended=True)
    except Exception as error:  # PyMuPDF raises several implementation-specific types.
        raise PdfAdapterError("PyMuPDF get_drawings(extended=True) failed") from error

    operations: list[PathOperationIR] = []
    for drawing_index, drawing in enumerate(drawings):
        drawing_type = drawing.get("type")
        if not _is_painted_path(drawing_type):
            # Extended clips and transparency groups are structural records,
            # not visible path operations and must never enter recognition.
            continue
        if "seqno" not in drawing:
            raise PdfAdapterError(
                f"painted drawing {drawing_index} has no PyMuPDF paint sequence number"
            )
        paint_order = int(drawing["seqno"])
        if paint_order < 0:
            raise PdfAdapterError(f"drawing {drawing_index} has negative seqno {paint_order}")
        ordinal = len(operations)
        stroke = "s" in str(drawing_type).lower()
        fill = "f" in str(drawing_type).lower()
        dash_array, dash_phase = _dash(drawing.get("dashes"))
        line_width = _finite(drawing.get("width") or 0.0, "line width")
        operations.append(PathOperationIR(
            operation_id=f"path:{ordinal:08d}",
            paint_order=paint_order,
            ordinal=ordinal,
            bounds=_bounds(drawing.get("rect"), page_height, f"drawing[{drawing_index}].rect"),
            segments=_path_segments(
                drawing.get("items", ()), bool(drawing.get("closePath")), page_height
            ),
            stroke=stroke,
            fill=fill,
            stroke_color=_color(drawing.get("color"), "stroke color"),
            fill_color=_color(drawing.get("fill"), "fill color"),
            line_width=line_width,
            hairline=stroke and line_width == 0.0,
            line_cap=_line_cap(drawing.get("lineCap")),
            line_join=_finite(drawing.get("lineJoin") or 0.0, "line join"),
            dash_array=dash_array,
            dash_phase=dash_phase,
            stroke_opacity=_finite(
                drawing.get("stroke_opacity") if drawing.get("stroke_opacity") is not None else 1.0,
                "stroke opacity",
            ),
            fill_opacity=_finite(
                drawing.get("fill_opacity") if drawing.get("fill_opacity") is not None else 1.0,
                "fill opacity",
            ),
            even_odd=bool(drawing.get("even_odd")),
            close_path=bool(drawing.get("closePath")),
            blend_mode=str(drawing.get("blendmode") or "Normal"),
            layer=str(drawing.get("layer") or ""),
            nesting_level=int(drawing.get("level") or 0),
        ))
    return operations


def _literal_character(codepoint: int) -> str:
    if codepoint > 0x10FFFF or 0xD800 <= codepoint <= 0xDFFF:
        return "\N{REPLACEMENT CHARACTER}"
    try:
        return chr(codepoint)
    except ValueError:
        return "\N{REPLACEMENT CHARACTER}"


def _text_operations(page: Any, page_height: float) -> list[TextOperationIR]:
    try:
        spans = page.get_texttrace()
    except Exception as error:
        raise PdfAdapterError("PyMuPDF get_texttrace() failed") from error

    operations: list[TextOperationIR] = []
    for span_index, span in enumerate(spans):
        if "seqno" not in span:
            raise PdfAdapterError(f"text span {span_index} has no paint sequence number")
        characters: list[TextCharacterIR] = []
        literal: list[str] = []
        for char_index, character in enumerate(span.get("chars", ())):
            if len(character) < 4:
                raise PdfAdapterError(
                    f"text span {span_index} character {char_index} is malformed: {character!r}"
                )
            codepoint = int(character[0])
            if codepoint < 0:
                raise PdfAdapterError(
                    f"text span {span_index} character {char_index} has negative codepoint"
                )
            literal.append(_literal_character(codepoint))
            characters.append(TextCharacterIR(
                codepoint=codepoint,
                glyph_id=int(character[1]),
                origin=_point(
                    character[2], page_height,
                    f"text[{span_index}].characters[{char_index}].origin",
                ),
                bounds=_bounds(
                    character[3], page_height,
                    f"text[{span_index}].characters[{char_index}].bbox",
                ),
            ))
        ordinal = len(operations)
        direction = _point(
            (span.get("dir") or (1.0, 0.0)),
            0.0,
            f"text[{span_index}].direction",
        )
        # _point reflects y around its supplied height.  Height zero therefore
        # converts a direction vector as (dx, -dy), without adding translation.
        operations.append(TextOperationIR(
            operation_id=f"text:{ordinal:08d}",
            paint_order=int(span["seqno"]),
            ordinal=ordinal,
            span_index=span_index,
            bounds=_bounds(span.get("bbox"), page_height, f"text[{span_index}].bbox"),
            literal_text="".join(literal),
            characters=tuple(characters),
            font_name=str(span.get("font") or ""),
            font_size=_finite(span.get("size") or 0.0, "font size"),
            direction=direction,
            render_mode=int(span.get("type") or 0),
            color=_color(span.get("color"), "text color"),
            opacity=_finite(
                span.get("opacity") if span.get("opacity") is not None else 1.0,
                "text opacity",
            ),
            line_width=(
                None
                if span.get("linewidth") is None
                else _finite(span.get("linewidth"), "text line width")
            ),
            writing_mode=int(span.get("wmode") or 0),
            flags=int(span.get("flags") or 0),
            bidi_level=int(span.get("bidi_lvl") or 0),
            bidi_direction=int(span.get("bidi_dir") or 0),
            layer=str(span.get("layer") or ""),
        ))
    return operations


def _bbox_key(value: Any) -> tuple[float, float, float, float]:
    try:
        return tuple(
            round(_finite(component, "image bbox"), 5)
            for component in value[:4]
        )  # type: ignore[return-value]
    except (IndexError, KeyError, TypeError) as error:
        raise PdfAdapterError(f"invalid image bbox: {value!r}") from error


def _image_paint_orders(page: Any, images: Sequence[Mapping[str, Any]]) -> list[int]:
    if not images:
        return []
    get_bboxlog = getattr(page, "get_bboxlog", None)
    if not callable(get_bboxlog):
        raise PdfAdapterError("PyMuPDF page has images but does not expose get_bboxlog()")
    try:
        bbox_log = get_bboxlog(layers=True)
    except TypeError:
        bbox_log = get_bboxlog()
    except Exception as error:
        raise PdfAdapterError("PyMuPDF get_bboxlog() failed while ordering images") from error

    logged: dict[tuple[float, float, float, float], list[int]] = defaultdict(list)
    for paint_order, entry in enumerate(bbox_log):
        if not entry or str(entry[0]) not in {"fill-image", "fill-imgmask"}:
            continue
        logged[_bbox_key(entry[1])].append(paint_order)

    requested: dict[tuple[float, float, float, float], list[int]] = defaultdict(list)
    for image_index, image in enumerate(images):
        requested[_bbox_key(image.get("bbox"))].append(image_index)

    orders = [-1] * len(images)
    for key, image_indices in requested.items():
        candidates = logged.get(key, [])
        if len(candidates) != len(image_indices):
            raise PdfAdapterError(
                "cannot map image occurrences to paint order without guessing: "
                f"bbox {key!r} has {len(image_indices)} image-info entries but "
                f"{len(candidates)} bbox-log entries"
            )
        for image_index, paint_order in zip(image_indices, candidates):
            orders[image_index] = paint_order
    if any(order < 0 for order in orders):
        raise PdfAdapterError("one or more image occurrences have no paint-order mapping")
    return orders


def _image_transform(
    value: Any, page_height: float
) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    try:
        a, b, c, d, e, f = value[:6]
    except (IndexError, KeyError, TypeError) as error:
        raise PdfAdapterError(f"invalid image transform: {value!r}") from error
    a_value = _finite(a, "image transform.a")
    b_value = _finite(b, "image transform.b")
    c_value = _finite(c, "image transform.c")
    d_value = _finite(d, "image transform.d")
    e_value = _finite(e, "image transform.e")
    f_value = _finite(f, "image transform.f")
    return (
        a_value,
        _finite(-b_value, "image transform.converted_b"),
        c_value,
        _finite(-d_value, "image transform.converted_d"),
        e_value,
        _finite(page_height - f_value, "image transform.converted_f"),
    )


def _image_operations(page: Any, page_height: float) -> list[ImageOperationIR]:
    get_image_info = getattr(page, "get_image_info", None)
    if not callable(get_image_info):
        return []
    try:
        images = get_image_info(hashes=True, xrefs=True)
    except TypeError:
        # PyMuPDF versions at the low end of the supported range may not expose
        # xrefs.  Digest and occurrence geometry remain available.
        images = get_image_info(hashes=True)
    except Exception as error:
        raise PdfAdapterError("PyMuPDF get_image_info() failed") from error
    paint_orders = _image_paint_orders(page, images)

    operations: list[ImageOperationIR] = []
    for ordinal, (image, paint_order) in enumerate(zip(images, paint_orders)):
        digest = image.get("digest", "")
        if isinstance(digest, (bytes, bytearray, memoryview)):
            digest_text = bytes(digest).hex()
        else:
            digest_text = str(digest or "")
        operations.append(ImageOperationIR(
            operation_id=f"image:{ordinal:08d}",
            paint_order=paint_order,
            ordinal=ordinal,
            bounds=_bounds(image.get("bbox"), page_height, f"image[{ordinal}].bbox"),
            pixel_width=int(image.get("width") or 0),
            pixel_height=int(image.get("height") or 0),
            colorspace=int(image.get("colorspace") or 0),
            xref=int(image.get("xref") or 0),
            digest=digest_text,
            transform=_image_transform(image.get("transform"), page_height),
        ))
    return operations


def page_ir_from_pymupdf_page(
    page: Any,
    *,
    source_sha256: str,
    source_name: str = "",
    page_number: int | None = None,
) -> PageIR:
    """Convert one attached PyMuPDF page to canonical PageIR.

    ``source_sha256`` must describe the original PDF bytes.  Re-serializing a
    PyMuPDF document here would make the fingerprint depend on library output,
    so callers must supply the source digest explicitly.
    """

    pymupdf = _pymupdf()
    if getattr(page, "parent", None) is None:
        raise PdfAdapterError("cannot extract an orphaned PyMuPDF page")
    try:
        crop_box = page.cropbox
        page_width = _finite(crop_box.width, "page width")
        page_height = _finite(crop_box.height, "page height")
    except Exception as error:
        if isinstance(error, PdfAdapterError):
            raise
        raise PdfAdapterError("cannot read the unrotated PyMuPDF crop box") from error
    if page_width <= 0 or page_height <= 0:
        raise PdfAdapterError(f"page crop box must be non-empty, got {page_width} x {page_height}")

    operations = [
        *_path_operations(page, page_height),
        *_text_operations(page, page_height),
        *_image_operations(page, page_height),
    ]
    operations.sort(
        key=lambda operation: (operation.paint_order, operation.ordinal, operation.kind)
    )
    actual_page_number = int(page_number if page_number is not None else page.number + 1)
    producer_version = str(
        getattr(pymupdf, "VersionBind", "")
        or getattr(pymupdf, "__version__", "")
    )
    return PageIR(
        page_number=actual_page_number,
        page_bounds=BoundsIR(0.0, 0.0, page_width, page_height),
        rotation_degrees=int(page.rotation) % 360,
        operations=tuple(operations),
        source_sha256=source_sha256,
        source_name=source_name,
        producer=f"PyMuPDF/{PDF_ADAPTER_VERSION}",
        producer_version=producer_version,
    )


def page_ir_from_pdf_bytes(
    source: bytes | bytearray | memoryview,
    page_number: int = 1,
    *,
    source_name: str = "",
) -> PageIR:
    """Extract a one-based page from PDF bytes without frontend involvement."""

    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError("page_number must be a positive one-based integer")
    source_bytes = bytes(source)
    pymupdf = _pymupdf()
    try:
        document = pymupdf.open(stream=source_bytes, filetype="pdf")
    except Exception as error:
        raise PdfAdapterError("PyMuPDF could not open the supplied PDF bytes") from error
    try:
        if page_number > document.page_count:
            raise IndexError(
                f"page {page_number} is outside the document's 1..{document.page_count} range"
            )
        return page_ir_from_pymupdf_page(
            document[page_number - 1],
            source_sha256=sha256(source_bytes).hexdigest(),
            source_name=source_name,
            page_number=page_number,
        )
    finally:
        document.close()


def pdf_page_count_from_bytes(source: bytes | bytearray | memoryview) -> int:
    """Return the PDF page count without creating recognition state."""

    source_bytes = bytes(source)
    pymupdf = _pymupdf()
    try:
        document = pymupdf.open(stream=source_bytes, filetype="pdf")
    except Exception as error:
        raise PdfAdapterError("PyMuPDF could not open the supplied PDF bytes") from error
    try:
        count = int(document.page_count)
        if count < 1:
            raise PdfAdapterError("PDF must contain at least one page")
        return count
    finally:
        document.close()


def page_ir_from_pdf_path(path: str | Path, page_number: int = 1) -> PageIR:
    """Read ``path`` once and extract a one-based page into PageIR."""

    source_path = Path(path)
    return page_ir_from_pdf_bytes(
        source_path.read_bytes(),
        page_number,
        source_name=source_path.name,
    )


__all__ = [
    "PDF_ADAPTER_VERSION",
    "PdfAdapterError",
    "page_ir_from_pdf_bytes",
    "page_ir_from_pdf_path",
    "page_ir_from_pymupdf_page",
    "pdf_page_count_from_bytes",
]
