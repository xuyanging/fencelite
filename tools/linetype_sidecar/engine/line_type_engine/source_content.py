"""Native PDF content-stream provenance for :class:`~line_type_engine.ir.PageIR`.

PyMuPDF deliberately exposes a normalized display list.  That is excellent
for visible bounds and paint style, but it cannot retain two source semantics
needed by sequential grouping:

* one display-list text ``seqno`` may contain many authored ``Tj`` / ``TJ``
  string shows; and
* rectangle / shorthand curve operators are normalized before
  ``get_drawings()`` returns them.

This module reads the PDF content streams directly with pypdf.  It does not
recognize line types and it does not depend on the browser or TypeScript.  The
result is a provenance layer: every visible paint has a unique dense
``paint_order`` while hidden text shows retain a unique ``event_index`` but no
paint identity.  Original path operators and Form invocation paths remain
available beside normalized :class:`~line_type_engine.ir.PathSegmentIR`
geometry.

The parser is intentionally fail-observable.  Unsupported painting operators,
cyclic Forms and malformed commands are recorded in ``issues`` or as explicit
``SourceUnsupportedPaintEventIR`` values; consumers that require exact PageIR
alignment must reject those records instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import math
import re
from typing import Any, Literal, TypeAlias

from .annotation_appearances import annotation_appearance_plans
from .bounded_content_stream import iter_content_stream_operations
from .ir import BoundsIR, PathSegmentIR, PointIR
from .runtime import UnsupportedRuntimeError, assert_supported_pypdf_runtime


SOURCE_CONTENT_VERSION = (
    "pypdf-source-content-r10-source-text-image-style-2026-08-24"
)

# Keep the individual decoded-image budget in lockstep with ``app/pdf-file``.
# The cumulative page budget is verified by the browser-side semantic join;
# any cumulative-only placeholder therefore fails closed rather than being
# treated as an ordinary image with guessed style semantics.
_MAX_BROWSER_IMAGE_PIXELS = 20_000_000

_BLEND_MODES = {
    "Normal": "source-over",
    "Compatible": "source-over",
    "Multiply": "multiply",
    "Screen": "screen",
    "Overlay": "overlay",
    "Darken": "darken",
    "Lighten": "lighten",
    "ColorDodge": "color-dodge",
    "ColorBurn": "color-burn",
    "HardLight": "hard-light",
    "SoftLight": "soft-light",
    "Difference": "difference",
    "Exclusion": "exclusion",
    "Hue": "hue",
    "Saturation": "saturation",
    "Color": "color",
    "Luminosity": "luminosity",
}

# These values deliberately mirror ``OpStructureBoundary`` in the frozen
# TypeScript parser.  They describe a close/open restart observed between two
# painted operations; an isolated opener or closer is not a restart by itself.
SOURCE_STRUCTURE_GRAPHICS_STATE = 1 << 0
SOURCE_STRUCTURE_TEXT_OBJECT = 1 << 1
SOURCE_STRUCTURE_MARKED_CONTENT = 1 << 2
SOURCE_STRUCTURE_COMPATIBILITY = 1 << 3

MatrixIR: TypeAlias = tuple[float, float, float, float, float, float]


class SourceContentError(RuntimeError):
    """Raised when a PDF cannot be parsed into source-content provenance."""


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise SourceContentError(f"{label} must be numeric, got {value!r}") from error
    if not math.isfinite(number):
        raise SourceContentError(f"{label} must be finite, got {value!r}")
    return 0.0 if number == 0.0 else number


def multiply_matrix(left: MatrixIR, right: MatrixIR) -> MatrixIR:
    """Return the PDF affine product ``left * right``."""

    return (
        left[0] * right[0] + left[2] * right[1],
        left[1] * right[0] + left[3] * right[1],
        left[0] * right[2] + left[2] * right[3],
        left[1] * right[2] + left[3] * right[3],
        left[0] * right[4] + left[2] * right[5] + left[4],
        left[1] * right[4] + left[3] * right[5] + left[5],
    )


def transform_point(matrix: MatrixIR, x: float, y: float) -> PointIR:
    return (
        matrix[0] * x + matrix[2] * y + matrix[4],
        matrix[1] * x + matrix[3] * y + matrix[5],
    )


def _js_round_non_negative(value: float) -> int:
    """Match ``Math.round`` for the non-negative values used by PDF colours."""

    return math.floor(value + 0.5)


def _rgb(red: float, green: float, blue: float) -> tuple[float, float, float]:
    """Mirror the frozen TypeScript parser's 8-bit CSS colour quantization."""

    return tuple(
        _js_round_non_negative(max(0.0, min(1.0, channel)) * 255.0) / 255.0
        for channel in (red, green, blue)
    )  # type: ignore[return-value]


def _gray(value: float) -> tuple[float, float, float]:
    return _rgb(value, value, value)


def _cmyk(
    cyan: float,
    magenta: float,
    yellow: float,
    black: float,
) -> tuple[float, float, float]:
    return _rgb(
        1.0 - min(1.0, cyan + black),
        1.0 - min(1.0, magenta + black),
        1.0 - min(1.0, yellow + black),
    )


def _matrix_scale(matrix: MatrixIR, user_unit: float) -> float:
    """Match the frozen parser's five-place effective CTM scale."""

    x_scale = math.hypot(matrix[0] * user_unit, matrix[1] * user_unit)
    y_scale = math.hypot(matrix[2] * user_unit, matrix[3] * user_unit)
    scale = (x_scale + y_scale) / 2.0
    quantized = _js_round_non_negative(scale * 100_000.0) / 100_000.0
    return max(0.000001, quantized)


@dataclass(frozen=True, slots=True)
class SourceLocationIR:
    """Stable location of an operator inside page or Form content."""

    content_stream_id: str
    stream_operator_index: int
    global_operator_index: int
    form_instance_path: tuple[str, ...] = ()
    # Canonical indirect-object ids of the containing Form resource scopes.
    # Unlike ``form_instance_path`` this intentionally omits parser-specific
    # invocation offsets so another PDF parser can compare the resource chain.
    resource_scope: tuple[str, ...] = ()
    graphics_depth: int = 0
    page_content_stream_index: int = 0


@dataclass(frozen=True, slots=True)
class SourcePathCommandIR:
    """One original PDF path command plus its normalized page geometry."""

    operator: Literal["m", "l", "c", "v", "y", "re", "h"]
    operands: tuple[float, ...]
    segments: tuple[PathSegmentIR, ...]
    location: SourceLocationIR


@dataclass(frozen=True, slots=True)
class SourcePathPaintEventIR:
    event_index: int
    paint_order: int
    path_ordinal: int
    paint_operator: Literal["S", "s", "f", "F", "f*", "B", "B*", "b", "b*"]
    commands: tuple[SourcePathCommandIR, ...]
    segments: tuple[PathSegmentIR, ...]
    # ``bounds`` is the authored control-point envelope.  ``visible_bounds``
    # is the frozen Scene contract: stroke expansion followed by the current
    # rectangular clip.  Keeping both avoids changing provenance semantics.
    bounds: BoundsIR
    visible_bounds: BoundsIR
    atom_multiplicity: int
    stroke: bool
    fill: bool
    even_odd: bool
    close_path: bool
    line_width: float
    line_cap: int
    line_join: int
    miter_limit: float
    dash_array: tuple[float, ...]
    dash_phase: float
    blend_mode: str
    stroke_opacity: float
    fill_opacity: float
    stroke_color: tuple[float, float, float]
    fill_color: tuple[float, float, float]
    location: SourceLocationIR
    structure_before: int = 0
    kind: Literal["path"] = "path"


@dataclass(frozen=True, slots=True)
class SourceTextShowEventIR:
    event_index: int
    paint_order: int | None
    text_show_ordinal: int
    show_operator: Literal["Tj", "TJ", "'", '"']
    array_item_index: int | None
    raw_bytes: bytes
    decoded_text: str
    glyph_count: int
    visible: bool
    text_object_id: str
    font_resource_name: str
    font_size: float
    render_mode: int
    text_matrix: MatrixIR
    glyph_advance: float
    horizontal_scale: float
    rise: float
    unclipped_bounds: BoundsIR
    fill_color: tuple[float, float, float]
    stroke_color: tuple[float, float, float]
    fill_opacity: float
    stroke_opacity: float
    line_width: float
    blend_mode: str
    visible_bounds: BoundsIR | None
    location: SourceLocationIR
    structure_before: int = 0
    kind: Literal["text"] = "text"


@dataclass(frozen=True, slots=True)
class SourceImagePaintEventIR:
    event_index: int
    paint_order: int
    image_ordinal: int
    resource_name: str
    pixel_width: int
    pixel_height: int
    bits_per_component: int
    color_space_name: str
    source_filters: tuple[str, ...]
    xref: int
    resource_id: str
    corners: tuple[PointIR, PointIR, PointIR, PointIR]
    bounds: BoundsIR
    visible_bounds: BoundsIR | None
    image_mask: bool
    target_placeholder_reason: str | None
    alpha: float
    blend_mode: str
    location: SourceLocationIR
    structure_before: int = 0
    paint_operator: Literal["Do"] = "Do"
    kind: Literal["image"] = "image"


@dataclass(frozen=True, slots=True)
class SourceInlineImageSkipEventIR:
    """Authored inline image intentionally omitted by the frozen Scene.

    PyMuPDF still exposes a display-list image occurrence with ``xref == 0``.
    Retaining this non-paint identity lets the adapter consume that occurrence
    without stealing pending q/Q structure from the next real Scene paint.
    """

    event_index: int
    paint_order: None
    inline_image_ordinal: int
    pixel_width: int
    pixel_height: int
    image_mask: bool
    bits_per_component: int
    color_space: str
    filters: tuple[str, ...]
    data_sha256: str
    corners: tuple[PointIR, PointIR, PointIR, PointIR]
    bounds: BoundsIR
    visible_bounds: BoundsIR | None
    location: SourceLocationIR
    kind: Literal["inline-image-skip"] = "inline-image-skip"


@dataclass(frozen=True, slots=True)
class SourceUnsupportedPaintEventIR:
    """A painting command that PageIR cannot represent without guessing."""

    event_index: int
    paint_order: int
    operator: str
    detail: str
    location: SourceLocationIR
    structure_before: int = 0
    kind: Literal["unsupported"] = "unsupported"


SourceEventIR: TypeAlias = (
    SourcePathPaintEventIR
    | SourceTextShowEventIR
    | SourceImagePaintEventIR
    | SourceInlineImageSkipEventIR
    | SourceUnsupportedPaintEventIR
)


@dataclass(frozen=True, slots=True)
class SourceBoundaryIR:
    boundary_index: int
    kind: Literal[
        "content-stream",
        "form",
        "text-object",
        "graphics-state",
        "marked-content",
        "compatibility",
    ]
    entering: bool
    location: SourceLocationIR
    next_event_index: int
    next_paint_order: int
    identity: str = ""


@dataclass(frozen=True, slots=True)
class SourceContentPageIR:
    page_number: int
    source_sha256: str
    source_name: str
    page_bounds: BoundsIR
    rotation_degrees: int
    events: tuple[SourceEventIR, ...]
    boundaries: tuple[SourceBoundaryIR, ...]
    issues: tuple[str, ...]
    annotation_appearance_count: int = 0
    source_content_version: str = SOURCE_CONTENT_VERSION
    producer: str = "pypdf"
    producer_version: str = ""

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a SHA-256 digest")
        if self.annotation_appearance_count < 0:
            raise ValueError("annotation_appearance_count must be non-negative")
        event_indices = [event.event_index for event in self.events]
        if event_indices != list(range(len(event_indices))):
            raise ValueError("source event indices must be dense and ordered")
        paint_orders = [
            event.paint_order
            for event in self.events
            if event.paint_order is not None
        ]
        if paint_orders != list(range(len(paint_orders))):
            raise ValueError("visible source paint orders must be dense and ordered")

    @property
    def paint_events(self) -> tuple[
        SourcePathPaintEventIR
        | SourceTextShowEventIR
        | SourceImagePaintEventIR
        | SourceUnsupportedPaintEventIR,
        ...,
    ]:
        return tuple(event for event in self.events if event.paint_order is not None)

    @property
    def path_events(self) -> tuple[SourcePathPaintEventIR, ...]:
        return tuple(
            event for event in self.events if isinstance(event, SourcePathPaintEventIR)
        )

    @property
    def text_events(self) -> tuple[SourceTextShowEventIR, ...]:
        return tuple(
            event for event in self.events if isinstance(event, SourceTextShowEventIR)
        )

    @property
    def visible_text_events(self) -> tuple[SourceTextShowEventIR, ...]:
        return tuple(event for event in self.text_events if event.visible)

    @property
    def image_events(self) -> tuple[SourceImagePaintEventIR, ...]:
        return tuple(
            event for event in self.events if isinstance(event, SourceImagePaintEventIR)
        )

    @property
    def inline_image_events(self) -> tuple[SourceInlineImageSkipEventIR, ...]:
        return tuple(
            event
            for event in self.events
            if isinstance(event, SourceInlineImageSkipEventIR)
        )

    @property
    def unsupported_paint_events(self) -> tuple[SourceUnsupportedPaintEventIR, ...]:
        return tuple(
            event
            for event in self.events
            if isinstance(event, SourceUnsupportedPaintEventIR)
        )


@dataclass(slots=True)
class _GraphicsState:
    ctm: MatrixIR = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line_width: float = 1.0
    line_cap: int = 0
    line_join: int = 0
    miter_limit: float = 10.0
    dash_array: tuple[float, ...] = ()
    dash_phase: float = 0.0
    blend_mode: str = "source-over"
    stroke_opacity: float = 1.0
    fill_opacity: float = 1.0
    stroke_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fill_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    clip: BoundsIR | None = None
    # These defaults are part of the frozen TypeScript Scene parser contract.
    # A conforming PDF normally selects a font before showing text, but keeping
    # the same defaults makes malformed/marginal streams deterministic too.
    font_resource_name: str = "sans-serif"
    font_size: float = 12.0
    char_spacing: float = 0.0
    word_spacing: float = 0.0
    horizontal_scale: float = 1.0
    leading: float = 0.0
    rise: float = 0.0
    render_mode: int = 0

    def clone(self) -> "_GraphicsState":
        return _GraphicsState(
            ctm=self.ctm,
            line_width=self.line_width,
            line_cap=self.line_cap,
            line_join=self.line_join,
            miter_limit=self.miter_limit,
            dash_array=self.dash_array,
            dash_phase=self.dash_phase,
            blend_mode=self.blend_mode,
            stroke_opacity=self.stroke_opacity,
            fill_opacity=self.fill_opacity,
            stroke_color=self.stroke_color,
            fill_color=self.fill_color,
            clip=self.clip,
            font_resource_name=self.font_resource_name,
            font_size=self.font_size,
            char_spacing=self.char_spacing,
            word_spacing=self.word_spacing,
            horizontal_scale=self.horizontal_scale,
            leading=self.leading,
            rise=self.rise,
            render_mode=self.render_mode,
        )


@dataclass(slots=True)
class _PathBuilder:
    commands: list[SourcePathCommandIR] = field(default_factory=list)
    segments: list[PathSegmentIR] = field(default_factory=list)
    current: PointIR | None = None
    subpath_start: PointIR | None = None

    def clear(self) -> None:
        self.commands.clear()
        self.segments.clear()
        self.current = None
        self.subpath_start = None


@dataclass(slots=True)
class _Frame:
    state: _GraphicsState
    path: _PathBuilder = field(default_factory=_PathBuilder)
    state_stack: list[_GraphicsState] = field(default_factory=list)
    active_text_object_id: str = ""
    pending_clip: bool = False
    pending_structure_closures: int = 0
    structure_before_next_paint: int = 0
    text_matrix: MatrixIR = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    line_matrix: MatrixIR = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class _CodeSpace:
    ranges: tuple[tuple[int, int, int], ...]
    fallback_width: int
    to_unicode: tuple[tuple[str, str], ...] = ()

    def decode(self, raw: bytes) -> tuple[str, tuple[int, ...]]:
        if not raw:
            return "", ()
        mapping = dict(self.to_unicode)
        if mapping:
            lengths = sorted(
                {
                    *(width for _, _, width in self.ranges if width > 0),
                    *(max(1, len(key) // 2) for key in mapping),
                },
                reverse=True,
            )
            fallback = lengths[-1] if lengths else max(1, self.fallback_width)
            offset = 0
            text: list[str] = []
            codes: list[int] = []
            while offset < len(raw):
                consumed = 0
                decoded: str | None = None
                code = 0
                for width in lengths:
                    if offset + width > len(raw):
                        continue
                    key = raw[offset:offset + width].hex().upper()
                    if key in mapping:
                        consumed = width
                        decoded = mapping[key]
                        code = int(key, 16)
                        break
                if not consumed:
                    consumed = min(fallback, len(raw) - offset)
                    key = raw[offset:offset + consumed].hex().upper()
                    code = int(key or "0", 16)
                    decoded = mapping.get(key)
                    if decoded is None:
                        decoded = chr(code) if 32 <= code <= 126 else "□"
                codes.append(code)
                text.append(decoded)
                offset += consumed
            return "".join(text), tuple(codes)

        if len(raw) >= 2 and raw[:2] == b"\xfe\xff":
            codes = tuple(
                int.from_bytes(raw[index:index + 2], "big")
                for index in range(2, len(raw) - 1, 2)
            )
            return "".join(chr(code) for code in codes), codes
        zero_high_bytes = sum(
            raw[index] == 0 for index in range(0, len(raw), 2)
        )
        two_byte_cid = (
            len(raw) >= 2
            and len(raw) % 2 == 0
            and zero_high_bytes > 0
            and zero_high_bytes >= len(raw) // 2 - 1
        )
        if two_byte_cid:
            codes = tuple(
                int.from_bytes(raw[index:index + 2], "big")
                for index in range(0, len(raw), 2)
            )
            return "".join(
                chr(code + 29) if 3 <= code <= 97 else "□"
                for code in codes
            ), codes
        latin = raw.decode("latin-1")
        printable = sum(
            character in "\t\n" or 32 <= ord(character) <= 126
            for character in latin
        )
        text = latin if printable >= len(latin) * 0.8 else "□" * max(1, len(raw))
        return text, tuple(raw)

    def count(self, raw: bytes) -> int:
        if self.to_unicode:
            return len(self.decode(raw)[1])
        offset = 0
        count = 0
        while offset < len(raw):
            matches: list[int] = []
            for lower, upper, width in self.ranges:
                if offset + width > len(raw):
                    continue
                value = int.from_bytes(raw[offset:offset + width], "big")
                if lower <= value <= upper:
                    matches.append(width)
            width = max(matches) if matches else self.fallback_width
            if width < 1 or offset + width > len(raw):
                raise SourceContentError(
                    "font code space cannot consume a complete source text string"
                )
            offset += width
            count += 1
        return count


@dataclass(frozen=True, slots=True)
class _FontResource:
    code_space: _CodeSpace
    widths: dict[int, float]
    default_width: float
    found: bool = True


_CODE_SPACE_SECTION = re.compile(
    rb"begincodespacerange(?P<body>.*?)endcodespacerange",
    re.DOTALL | re.IGNORECASE,
)
_CODE_SPACE_PAIR = re.compile(rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>")
_BF_CHAR_SECTION = re.compile(
    rb"beginbfchar(?P<body>.*?)endbfchar", re.DOTALL | re.IGNORECASE
)
_BF_RANGE_SECTION = re.compile(
    rb"beginbfrange(?P<body>.*?)endbfrange", re.DOTALL | re.IGNORECASE
)
_BF_RANGE = re.compile(
    rb"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*"
    rb"(?:<([0-9A-Fa-f]+)>|\[(.*?)\])",
    re.DOTALL,
)
_HEX_VALUE = re.compile(rb"<([0-9A-Fa-f]+)>")


def _unicode_from_hex(value: bytes) -> str:
    raw = value + (b"0" if len(value) % 2 else b"")
    decoded = bytes.fromhex(raw.decode("ascii"))
    if decoded.startswith(b"\xfe\xff"):
        decoded = decoded[2:]
    return "".join(
        chr(decoded[index] * 256 + decoded[index + 1])
        for index in range(0, len(decoded) - 1, 2)
    )


def _to_unicode_map(data: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    for section in _BF_CHAR_SECTION.finditer(data):
        for source, destination in _CODE_SPACE_PAIR.findall(section.group("body")):
            result[source.decode("ascii").upper()] = _unicode_from_hex(destination)
    for section in _BF_RANGE_SECTION.finditer(data):
        for match in _BF_RANGE.finditer(section.group("body")):
            start_raw, end_raw, destination_raw, destination_array = match.groups()
            start = int(start_raw, 16)
            end = int(end_raw, 16)
            count = min(65536, max(0, end - start + 1))
            destinations = (
                _HEX_VALUE.findall(destination_array) if destination_array else []
            )
            for offset in range(count):
                key = f"{start + offset:0{len(start_raw)}X}"
                destination = (
                    f"{int(destination_raw, 16) + offset:0{len(destination_raw)}X}".encode()
                    if destination_raw
                    else destinations[offset] if offset < len(destinations) else None
                )
                if destination is not None:
                    result[key] = _unicode_from_hex(destination)
    return result


def _object(value: Any) -> Any:
    getter = getattr(value, "get_object", None)
    return getter() if callable(getter) else value


def _strict_pdf_boolean(value: Any, label: str, *, default: bool = False) -> bool:
    """Read a PDF boolean without Python object-truthiness coercion.

    ``pypdf.generic.BooleanObject(False)`` is itself truthy.  Calling
    ``bool(value)`` therefore turns an authored ``false`` into ``true`` and
    misclassifies a normal image as an ImageMask placeholder.
    """

    resolved = _object(value)
    if resolved is None:
        return default
    try:
        from pypdf.generic import BooleanObject
    except ImportError as error:  # pragma: no cover - runtime guard owns this.
        raise SourceContentError("pypdf BooleanObject is unavailable") from error
    if not isinstance(resolved, BooleanObject) or type(resolved.value) is not bool:
        raise SourceContentError(f"{label} must be a PDF boolean")
    return resolved.value


def _stream_id(value: Any, fallback: str) -> str:
    if hasattr(value, "idnum"):
        return f"obj:{int(value.idnum)}:{int(getattr(value, 'generation', 0))}"
    return fallback


def _xref(value: Any) -> int:
    return int(getattr(value, "idnum", 0) or 0)


def _resource_id(value: Any) -> str:
    """Return the cross-parser PDF indirect-object identity, if available."""

    if not hasattr(value, "idnum"):
        return ""
    return (
        f"{int(value.idnum)} {int(getattr(value, 'generation', 0) or 0)} R"
    )


def _name(value: Any) -> str:
    return str(value or "").lstrip("/")


_FILTER_NAMES = {
    "/AHx": "/ASCIIHexDecode",
    "/A85": "/ASCII85Decode",
    "/LZW": "/LZWDecode",
    "/Fl": "/FlateDecode",
    "/RL": "/RunLengthDecode",
    "/CCF": "/CCITTFaxDecode",
    "/DCT": "/DCTDecode",
    "/JPX": "/JPXDecode",
}


def _canonical_filter_name(value: Any) -> str:
    name = str(_object(value) or "")
    return _FILTER_NAMES.get(name, name)


def _canonical_pdf_value(value: Any) -> str:
    """Return a parser-instance-independent identity for a PDF value."""

    if hasattr(value, "idnum"):
        return _resource_id(value)
    resolved = _object(value)
    if isinstance(resolved, (list, tuple)):
        return "[" + " ".join(_canonical_pdf_value(item) for item in resolved) + "]"
    if isinstance(resolved, dict):
        items = sorted(
            (str(key), _canonical_pdf_value(item))
            for key, item in resolved.items()
        )
        return "<<" + " ".join(f"{key} {item}" for key, item in items) + ">>"
    if isinstance(resolved, bool):
        return "true" if resolved else "false"
    if isinstance(resolved, (int, float)):
        return format(_finite(resolved, "PDF value"), ".15g")
    return str(resolved or "")


def _numbers(operands: Any, count: int, operator: str) -> tuple[float, ...]:
    if len(operands) < count:
        raise SourceContentError(f"{operator} requires {count} numeric operands")
    return tuple(
        _finite(value, f"{operator}[{index}]")
        for index, value in enumerate(operands[-count:])
    )


def _numeric_operands(operands: Any, operator: str) -> tuple[float, ...]:
    """Return numeric operands while ignoring Pattern/colour-space names."""

    result: list[float] = []
    for index, value in enumerate(operands):
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(number):
            raise SourceContentError(
                f"{operator}[{index}] must be finite, got {value!r}"
            )
        result.append(0.0 if number == 0.0 else number)
    return tuple(result)


def _matrix(value: Any, label: str) -> MatrixIR:
    values = _numbers(value, 6, label)
    return values  # type: ignore[return-value]


def _raw_string_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    original = getattr(value, "original_bytes", None)
    if original is not None:
        return bytes(original)
    getter = getattr(value, "get_original_bytes", None)
    if callable(getter):
        return bytes(getter())
    encoded = getattr(value, "get_encoded_bytes", None)
    if callable(encoded):
        return bytes(encoded())
    if isinstance(value, str):
        return value.encode("latin-1", errors="replace")
    raise SourceContentError(f"text-show operand is not a PDF string: {value!r}")


def _is_pdf_string(value: Any) -> bool:
    return isinstance(value, (bytes, bytearray, memoryview)) or type(value).__name__ in {
        "ByteStringObject",
        "TextStringObject",
    }


def _bounds_for_segments(segments: tuple[PathSegmentIR, ...]) -> BoundsIR | None:
    points = [
        point
        for segment in segments
        for point in (segment.end, segment.control_1, segment.control_2)
        if point is not None
    ]
    if not points:
        return None
    return BoundsIR(
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _expand_bounds(bounds: BoundsIR, amount: float) -> BoundsIR:
    return BoundsIR(
        bounds.min_x - amount,
        bounds.min_y - amount,
        bounds.max_x + amount,
        bounds.max_y + amount,
    )


def _intersect_bounds(
    left: BoundsIR | None,
    right: BoundsIR | None,
) -> BoundsIR | None:
    """Mirror the frozen parser's rectangular ``intersectBounds`` helper."""

    if left is None:
        return right
    if right is None:
        return left
    minimum_x = max(left.min_x, right.min_x)
    minimum_y = max(left.min_y, right.min_y)
    maximum_x = min(left.max_x, right.max_x)
    maximum_y = min(left.max_y, right.max_y)
    if minimum_x > maximum_x or minimum_y > maximum_y:
        return None
    return BoundsIR(minimum_x, minimum_y, maximum_x, maximum_y)


def _drawable_subpath_count(segments: tuple[PathSegmentIR, ...]) -> int:
    """Count source atoms exactly as drawable move-delimited subpaths."""

    count = 0
    has_draw = False
    for segment in segments:
        if segment.kind == "move":
            if has_draw:
                count += 1
            has_draw = False
        elif segment.kind in {"line", "curve"}:
            has_draw = True
    return count + int(has_draw)


class _SourceParser:
    def __init__(
        self,
        reader: Any,
        page: Any,
        page_number: int,
        source_sha256: str,
        source_name: str,
        producer_version: str,
    ) -> None:
        self.reader = reader
        self.page = page
        self.page_number = page_number
        self.source_sha256 = source_sha256
        self.source_name = source_name
        self.producer_version = producer_version
        crop = _object(page.get("/CropBox") or page.get("/MediaBox"))
        if crop is None or len(crop) != 4:
            raise SourceContentError("page has no valid CropBox or MediaBox")
        self.crop_left = _finite(crop[0], "crop.left")
        self.crop_bottom = _finite(crop[1], "crop.bottom")
        right = _finite(crop[2], "crop.right")
        top = _finite(crop[3], "crop.top")
        self.user_unit = _finite(page.get("/UserUnit", 1.0), "UserUnit")
        if self.user_unit <= 0.0:
            raise SourceContentError("UserUnit must be positive")
        self.page_bounds = BoundsIR(
            0.0,
            0.0,
            (right - self.crop_left) * self.user_unit,
            (top - self.crop_bottom) * self.user_unit,
        )
        self.rotation_degrees = int(page.get("/Rotate", 0) or 0) % 360
        self.events: list[SourceEventIR] = []
        self.boundaries: list[SourceBoundaryIR] = []
        self.issues: list[str] = []
        self.global_operator_index = 0
        self.next_paint_order = 0
        self.path_ordinal = 0
        self.text_show_ordinal = 0
        self.image_ordinal = 0
        self.inline_image_ordinal = 0
        self.text_object_sequence = 0
        self.form_invocation_sequence = 0
        self.font_resources: dict[tuple[int, int] | int, _FontResource] = {}

    def location(
        self,
        stream_id: str,
        stream_operator_index: int,
        form_path: tuple[str, ...],
        resource_scope: tuple[str, ...],
        graphics_depth: int,
        page_content_stream_index: int,
        global_operator_index: int | None = None,
    ) -> SourceLocationIR:
        return SourceLocationIR(
            content_stream_id=stream_id,
            stream_operator_index=stream_operator_index,
            global_operator_index=(
                self.global_operator_index
                if global_operator_index is None
                else global_operator_index
            ),
            form_instance_path=form_path,
            resource_scope=resource_scope,
            graphics_depth=graphics_depth,
            page_content_stream_index=page_content_stream_index,
        )

    def boundary(
        self,
        kind: SourceBoundaryIR.__annotations__["kind"],
        entering: bool,
        location: SourceLocationIR,
        identity: str = "",
    ) -> None:
        self.boundaries.append(SourceBoundaryIR(
            boundary_index=len(self.boundaries),
            kind=kind,
            entering=entering,
            location=location,
            next_event_index=len(self.events),
            next_paint_order=self.next_paint_order,
            identity=identity,
        ))

    @staticmethod
    def _mark_structure_open(frame: _Frame, flag: int) -> None:
        if frame.pending_structure_closures & flag:
            frame.structure_before_next_paint |= flag

    @staticmethod
    def _mark_structure_close(frame: _Frame, flag: int) -> None:
        frame.pending_structure_closures |= flag

    @staticmethod
    def _take_structure_before(frame: _Frame) -> int:
        value = frame.structure_before_next_paint
        frame.structure_before_next_paint = 0
        frame.pending_structure_closures = 0
        return value

    def _normalized_point(self, state: _GraphicsState, x: float, y: float) -> PointIR:
        raw_x, raw_y = transform_point(state.ctm, x, y)
        return (
            (raw_x - self.crop_left) * self.user_unit,
            (raw_y - self.crop_bottom) * self.user_unit,
        )

    def _normalized_matrix(self, matrix: MatrixIR) -> MatrixIR:
        """Map a source-space matrix into the normalized PageIR coordinate space."""

        return (
            matrix[0] * self.user_unit,
            matrix[1] * self.user_unit,
            matrix[2] * self.user_unit,
            matrix[3] * self.user_unit,
            (matrix[4] - self.crop_left) * self.user_unit,
            (matrix[5] - self.crop_bottom) * self.user_unit,
        )

    @staticmethod
    def _optional_number(value: Any) -> float | None:
        value = _object(value)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def _font_resource(self, resources: Any, font_name: str) -> _FontResource:
        """Return the frozen Scene font decoder and width table for ``font_name``."""

        resources = _object(resources) or {}
        fonts = _object(resources.get("/Font")) if hasattr(resources, "get") else None
        reference = fonts.get(f"/{font_name}") if fonts and hasattr(fonts, "get") else None
        if reference is None:
            self.issues.append(f"missing font resource /{font_name}")
            return _FontResource(_CodeSpace((), 1), {}, 500.0, found=False)
        key: tuple[int, int] | int
        if hasattr(reference, "idnum"):
            key = (int(reference.idnum), int(getattr(reference, "generation", 0)))
        else:
            key = id(reference)
        cached = self.font_resources.get(key)
        if cached is not None:
            return cached
        font = _object(reference)
        if not hasattr(font, "get"):
            raise SourceContentError(f"font /{font_name} is not a dictionary")
        subtype = str(font.get("/Subtype", ""))
        encoding = str(_object(font.get("/Encoding", "")))
        fallback_width = 2 if subtype == "/Type0" or encoding in {
            "/Identity-H", "/Identity-V"
        } else 1
        ranges: list[tuple[int, int, int]] = []
        to_unicode_map: dict[str, str] = {}
        to_unicode = font.get("/ToUnicode")
        if to_unicode is not None:
            try:
                data = bytes(_object(to_unicode).get_data())
                for section in _CODE_SPACE_SECTION.finditer(data):
                    for match in _CODE_SPACE_PAIR.finditer(section.group("body")):
                        lower_raw, upper_raw = match.groups()
                        if len(lower_raw) != len(upper_raw) or len(lower_raw) % 2:
                            continue
                        width = len(lower_raw) // 2
                        ranges.append((
                            int(lower_raw, 16),
                            int(upper_raw, 16),
                            width,
                        ))
                to_unicode_map = _to_unicode_map(data)
            except Exception as error:
                self.issues.append(
                    f"font /{font_name} ToUnicode parse failed: {error}"
                )

        widths: dict[int, float] = {}
        descendants = _object(font.get("/DescendantFonts"))
        descendant = (
            _object(descendants[0])
            if descendants is not None and len(descendants) > 0
            else None
        )
        if descendant is not None and hasattr(descendant, "get"):
            width_array = _object(descendant.get("/W"))
            if width_array is not None:
                index = 0
                while index < len(width_array):
                    first_code = self._optional_number(width_array[index])
                    if first_code is None:
                        break
                    first = int(first_code)
                    if index + 1 >= len(width_array):
                        break
                    width_list = _object(width_array[index + 1])
                    if type(width_list).__name__ == "ArrayObject" or isinstance(
                        width_list, (list, tuple)
                    ):
                        for offset, raw_width in enumerate(width_list):
                            width = self._optional_number(raw_width)
                            if width is not None:
                                widths[first + offset] = width
                        index += 2
                        continue
                    if index + 2 >= len(width_array):
                        break
                    last_code = self._optional_number(width_array[index + 1])
                    width = self._optional_number(width_array[index + 2])
                    if last_code is None or width is None:
                        break
                    last = int(last_code)
                    for code in range(first, min(last, first + 65_535) + 1):
                        widths[code] = width
                    index += 3
            default_width = self._optional_number(descendant.get("/DW"))
            if default_width is None:
                default_width = 1000.0
        else:
            first_char = self._optional_number(font.get("/FirstChar"))
            first = int(first_char) if first_char is not None else 0
            width_array = _object(font.get("/Widths"))
            if width_array is not None:
                for offset, raw_width in enumerate(width_array):
                    width = self._optional_number(raw_width)
                    if width is not None:
                        widths[first + offset] = width
            descriptor = _object(font.get("/FontDescriptor"))
            missing_width = (
                self._optional_number(descriptor.get("/MissingWidth"))
                if descriptor is not None and hasattr(descriptor, "get")
                else None
            )
            default_width = missing_width if missing_width is not None else 500.0

        result = _FontResource(
            code_space=_CodeSpace(
                tuple(ranges),
                fallback_width,
                tuple(to_unicode_map.items()),
            ),
            widths=widths,
            default_width=default_width,
        )
        self.font_resources[key] = result
        return result

    def _add_path_event(
        self,
        frame: _Frame,
        operator: str,
        location: SourceLocationIR,
    ) -> None:
        if not frame.path.segments:
            frame.path.clear()
            frame.pending_clip = False
            return
        segments = tuple(frame.path.segments)
        bounds = _bounds_for_segments(segments)
        if bounds is None:
            frame.path.clear()
            frame.pending_clip = False
            return
        stroke = operator in {"S", "s", "B", "B*", "b", "b*"}
        fill = operator in {"f", "F", "f*", "B", "B*", "b", "b*"}
        scale = _matrix_scale(frame.state.ctm, self.user_unit)
        line_width = (
            0.0
            if frame.state.line_width == 0.0
            else abs(frame.state.line_width) * scale
        )
        painted_bounds = (
            _expand_bounds(bounds, max(line_width / 2.0, 0.25))
            if stroke
            else bounds
        )
        visible_bounds = _intersect_bounds(painted_bounds, frame.state.clip)
        if visible_bounds is not None:
            event = SourcePathPaintEventIR(
                event_index=len(self.events),
                paint_order=self.next_paint_order,
                path_ordinal=self.path_ordinal,
                paint_operator=operator,  # type: ignore[arg-type]
                commands=tuple(frame.path.commands),
                segments=segments,
                bounds=bounds,
                visible_bounds=visible_bounds,
                atom_multiplicity=_drawable_subpath_count(segments),
                stroke=stroke,
                fill=fill,
                even_odd=operator.endswith("*"),
                close_path=operator in {"s", "b", "b*"} or (
                    bool(segments) and segments[-1].kind == "close"
                ),
                line_width=line_width,
                line_cap=frame.state.line_cap,
                line_join=frame.state.line_join,
                miter_limit=frame.state.miter_limit,
                dash_array=tuple(
                    abs(value) * scale for value in frame.state.dash_array
                ),
                dash_phase=frame.state.dash_phase * scale,
                blend_mode=frame.state.blend_mode,
                stroke_opacity=frame.state.stroke_opacity,
                fill_opacity=frame.state.fill_opacity,
                stroke_color=frame.state.stroke_color,
                fill_color=frame.state.fill_color,
                location=location,
                structure_before=self._take_structure_before(frame),
            )
            self.events.append(event)
            self.next_paint_order += 1
            self.path_ordinal += 1
        self._apply_pending_clip(frame)
        frame.path.clear()

    def _apply_pending_clip(self, frame: _Frame) -> None:
        if not frame.pending_clip:
            return
        path_bounds = _bounds_for_segments(tuple(frame.path.segments))
        if path_bounds is not None:
            frame.state.clip = _intersect_bounds(frame.state.clip, path_bounds)
        frame.pending_clip = False

    def _close_path(self, frame: _Frame) -> None:
        if not frame.path.segments or frame.path.subpath_start is None:
            return
        frame.path.segments.append(PathSegmentIR("close"))
        frame.path.current = frame.path.subpath_start

    def _path_command(
        self,
        frame: _Frame,
        operator: str,
        operands: Any,
        location: SourceLocationIR,
    ) -> None:
        state = frame.state
        path = frame.path
        values: tuple[float, ...]
        normalized: list[PathSegmentIR] = []
        if operator == "m":
            values = _numbers(operands, 2, operator)
            point = self._normalized_point(state, values[0], values[1])
            normalized.append(PathSegmentIR("move", point))
            path.current = point
            path.subpath_start = point
        elif operator == "l":
            values = _numbers(operands, 2, operator)
            point = self._normalized_point(state, values[0], values[1])
            if path.current is None:
                normalized.append(PathSegmentIR("move", point))
                path.subpath_start = point
            else:
                normalized.append(PathSegmentIR("line", point))
            path.current = point
        elif operator == "c":
            values = _numbers(operands, 6, operator)
            first = self._normalized_point(state, values[0], values[1])
            second = self._normalized_point(state, values[2], values[3])
            end = self._normalized_point(state, values[4], values[5])
            if path.current is None:
                normalized.append(PathSegmentIR("move", first))
                path.subpath_start = first
            normalized.append(PathSegmentIR("curve", end, first, second))
            path.current = end
        elif operator == "v":
            values = _numbers(operands, 4, operator)
            if path.current is None:
                self.issues.append(
                    f"v without current point at source operator {location.global_operator_index}"
                )
                return
            second = self._normalized_point(state, values[0], values[1])
            end = self._normalized_point(state, values[2], values[3])
            normalized.append(PathSegmentIR("curve", end, path.current, second))
            path.current = end
        elif operator == "y":
            values = _numbers(operands, 4, operator)
            first = self._normalized_point(state, values[0], values[1])
            end = self._normalized_point(state, values[2], values[3])
            if path.current is None:
                normalized.append(PathSegmentIR("move", first))
                path.subpath_start = first
            normalized.append(PathSegmentIR("curve", end, first, end))
            path.current = end
        elif operator == "re":
            values = _numbers(operands, 4, operator)
            x, y, width, height = values
            points = tuple(
                self._normalized_point(state, px, py)
                for px, py in (
                    (x, y),
                    (x + width, y),
                    (x + width, y + height),
                    (x, y + height),
                )
            )
            normalized.extend((
                PathSegmentIR("move", points[0]),
                PathSegmentIR("line", points[1]),
                PathSegmentIR("line", points[2]),
                PathSegmentIR("line", points[3]),
                PathSegmentIR("close"),
            ))
            path.current = points[0]
            path.subpath_start = points[0]
        else:
            values = ()
            if path.segments:
                normalized.append(PathSegmentIR("close"))
                path.current = path.subpath_start
        path.commands.append(SourcePathCommandIR(
            operator=operator,  # type: ignore[arg-type]
            operands=values,
            segments=tuple(normalized),
            location=location,
        ))
        path.segments.extend(normalized)

    def _show_text(
        self,
        frame: _Frame,
        resources: Any,
        operator: str,
        value: Any,
        array_item_index: int | None,
        location: SourceLocationIR,
    ) -> None:
        raw = _raw_string_bytes(value)
        if not raw:
            return
        font = self._font_resource(resources, frame.state.font_resource_name)
        decoded_text, decoded_codes = font.code_space.decode(raw)
        # ``glyph_count`` is the strict join cardinality for the PyMuPDF trace,
        # not the frozen parser's display-decoder code count.  In particular,
        # its legacy two-byte-CID heuristic intentionally renders a two-byte
        # simple-font string as one placeholder while MuPDF still exposes two
        # glyphs.  Authored code-space width preserves that lossless join.
        try:
            glyph_count = font.code_space.count(raw)
        except SourceContentError as error:
            self.issues.append(
                f"text show {self.text_show_ordinal} code-space mismatch: {error}"
            )
            glyph_count = len(decoded_codes)
        # Mirror the browser parser's explicit protection against pathological
        # single-show strings.  The ellipsis is display text only; widths use
        # at most the first 300 source codes, exactly like the frozen parser.
        safe_text = (
            f"{decoded_text[:299]}…"
            if len(decoded_text) > 300
            else decoded_text
        )
        if not safe_text:
            return
        safe_codes = decoded_codes[:300]
        if font.found:
            glyph_width = (
                sum(
                    font.widths.get(code, font.default_width)
                    for code in safe_codes
                )
                / 1000.0
            ) * frame.state.font_size
        else:
            glyph_width = max(0.35, len(safe_text) * 0.56) * frame.state.font_size
        spaces = sum(character == " " for character in safe_text)
        glyph_advance = (
            glyph_width
            + max(0, len(safe_codes) - 1) * frame.state.char_spacing
            + spaces * frame.state.word_spacing
        )
        advance = glyph_advance * frame.state.horizontal_scale
        text_matrix = self._normalized_matrix(multiply_matrix(
            frame.state.ctm,
            frame.text_matrix,
        ))
        bottom = frame.state.rise - frame.state.font_size * 0.25
        top = frame.state.rise + frame.state.font_size * 0.88
        corners = tuple(
            transform_point(text_matrix, x, y)
            for x, y in (
                (0.0, bottom),
                (advance, bottom),
                (advance, top),
                (0.0, top),
            )
        )
        bounds = BoundsIR(
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
        visible_bounds = _intersect_bounds(bounds, frame.state.clip)
        visible = frame.state.render_mode != 3 and visible_bounds is not None
        paint_order = self.next_paint_order if visible else None
        text_object_id = frame.active_text_object_id or "outside-text-object"
        event = SourceTextShowEventIR(
            event_index=len(self.events),
            paint_order=paint_order,
            text_show_ordinal=self.text_show_ordinal,
            show_operator=operator,  # type: ignore[arg-type]
            array_item_index=array_item_index,
            raw_bytes=raw,
            decoded_text=safe_text,
            glyph_count=glyph_count,
            visible=visible,
            text_object_id=text_object_id,
            font_resource_name=frame.state.font_resource_name,
            font_size=frame.state.font_size,
            render_mode=frame.state.render_mode,
            text_matrix=text_matrix,
            glyph_advance=glyph_advance,
            horizontal_scale=frame.state.horizontal_scale,
            rise=frame.state.rise,
            unclipped_bounds=bounds,
            fill_color=frame.state.fill_color,
            stroke_color=frame.state.stroke_color,
            fill_opacity=frame.state.fill_opacity,
            stroke_opacity=frame.state.stroke_opacity,
            line_width=frame.state.line_width,
            blend_mode=frame.state.blend_mode,
            visible_bounds=visible_bounds if visible else None,
            location=location,
            structure_before=(
                self._take_structure_before(frame) if visible else 0
            ),
        )
        self.events.append(event)
        self.text_show_ordinal += 1
        if visible:
            self.next_paint_order += 1
        frame.text_matrix = multiply_matrix(
            frame.text_matrix,
            (1.0, 0.0, 0.0, 1.0, advance, 0.0),
        )

    def _add_image(
        self,
        reference: Any,
        resource_name: str,
        location: SourceLocationIR,
        frame: _Frame,
    ) -> None:
        image = _object(reference)
        corners = tuple(
            self._normalized_point(frame.state, x, y)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        )
        bounds = BoundsIR(
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
        visible_bounds = _intersect_bounds(bounds, frame.state.clip)
        if visible_bounds is None:
            return
        image_mask = _strict_pdf_boolean(
            image.get("/ImageMask"),
            f"image /{resource_name}.ImageMask",
        )
        filter_value = _object(image.get("/Filter"))
        if isinstance(filter_value, (list, tuple)):
            source_filters = tuple(
                _canonical_filter_name(item) for item in filter_value
            )
        elif filter_value is None:
            source_filters = ()
        else:
            source_filters = (_canonical_filter_name(filter_value),)
        color_space_value = image.get("/ColorSpace")
        color_space_name = (
            _canonical_pdf_value(_object(color_space_value))
            if color_space_value is not None
            else ""
        )
        target_reason = (
            "IMAGE_MASK_UNSUPPORTED"
            if image_mask
            else (
                "IMAGE_JPX_UNSUPPORTED"
                if "/JPXDecode" in source_filters
                else (
                    "IMAGE_PIXEL_BUDGET"
                    if int(image.get("/Width", 0) or 0)
                    * int(image.get("/Height", 0) or 0)
                    > _MAX_BROWSER_IMAGE_PIXELS
                    else None
                )
            )
        )
        self.events.append(SourceImagePaintEventIR(
            event_index=len(self.events),
            paint_order=self.next_paint_order,
            image_ordinal=self.image_ordinal,
            resource_name=resource_name,
            pixel_width=int(image.get("/Width", 0) or 0),
            pixel_height=int(image.get("/Height", 0) or 0),
            bits_per_component=int(image.get("/BitsPerComponent", 0) or 0),
            color_space_name=color_space_name,
            source_filters=source_filters,
            xref=_xref(reference),
            resource_id=_resource_id(reference),
            corners=corners,
            bounds=bounds,
            visible_bounds=visible_bounds,
            image_mask=image_mask,
            target_placeholder_reason=target_reason,
            alpha=frame.state.fill_opacity,
            blend_mode=frame.state.blend_mode,
            location=location,
            structure_before=self._take_structure_before(frame),
        ))
        self.next_paint_order += 1
        self.image_ordinal += 1

    @staticmethod
    def _inline_setting(settings: Any, *names: str) -> Any:
        for name in names:
            value = settings.get(f"/{name}") if hasattr(settings, "get") else None
            if value is not None:
                return _object(value)
        return None

    def _add_inline_image_skip(
        self,
        operands: Any,
        location: SourceLocationIR,
        frame: _Frame,
    ) -> None:
        if not isinstance(operands, dict):
            raise SourceContentError("INLINE IMAGE operands are not a dictionary")
        settings = _object(operands.get("settings"))
        data = operands.get("data")
        if not hasattr(settings, "get") or not isinstance(
            data, (bytes, bytearray, memoryview)
        ):
            raise SourceContentError("INLINE IMAGE settings/data are malformed")

        def integer_setting(label: str, *names: str) -> int:
            value = self._inline_setting(settings, *names)
            number = self._optional_number(value)
            if number is None or int(number) != number or number < 0:
                raise SourceContentError(
                    f"INLINE IMAGE {label} must be a non-negative integer"
                )
            return int(number)

        width = integer_setting("width", "W", "Width")
        height = integer_setting("height", "H", "Height")
        image_mask = _strict_pdf_boolean(
            self._inline_setting(settings, "IM", "ImageMask"),
            "INLINE IMAGE ImageMask",
        )
        bits_value = self._inline_setting(settings, "BPC", "BitsPerComponent")
        bits = 1 if image_mask and bits_value is None else integer_setting(
            "bits-per-component", "BPC", "BitsPerComponent"
        )
        if image_mask and bits != 1:
            raise SourceContentError(
                "INLINE IMAGE stencil bits-per-component must be 1"
            )
        color_value = self._inline_setting(settings, "CS", "ColorSpace")
        color_space = str(color_value or "")
        filter_value = self._inline_setting(settings, "F", "Filter")
        if type(filter_value).__name__ == "ArrayObject" or isinstance(
            filter_value, (list, tuple)
        ):
            filters = tuple(_canonical_filter_name(item) for item in filter_value)
        elif filter_value is None:
            filters = ()
        else:
            filters = (_canonical_filter_name(filter_value),)
        corners = tuple(
            self._normalized_point(frame.state, x, y)
            for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        )
        bounds = BoundsIR(
            min(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[0] for point in corners),
            max(point[1] for point in corners),
        )
        self.events.append(SourceInlineImageSkipEventIR(
            event_index=len(self.events),
            paint_order=None,
            inline_image_ordinal=self.inline_image_ordinal,
            pixel_width=width,
            pixel_height=height,
            image_mask=image_mask,
            bits_per_component=bits,
            color_space=color_space,
            filters=filters,
            data_sha256=sha256(bytes(data)).hexdigest(),
            corners=corners,  # type: ignore[arg-type]
            bounds=bounds,
            visible_bounds=_intersect_bounds(bounds, frame.state.clip),
            location=location,
        ))
        self.inline_image_ordinal += 1

    def _unsupported_paint(
        self,
        operator: str,
        detail: str,
        location: SourceLocationIR,
        frame: _Frame,
    ) -> None:
        self.events.append(SourceUnsupportedPaintEventIR(
            event_index=len(self.events),
            paint_order=self.next_paint_order,
            operator=operator,
            detail=detail,
            location=location,
            structure_before=self._take_structure_before(frame),
        ))
        self.next_paint_order += 1

    def _walk_form(
        self,
        reference: Any,
        resource_name: str,
        parent_resources: Any,
        parent_frame: _Frame,
        parent_location: SourceLocationIR,
        form_path: tuple[str, ...],
        active_forms: frozenset[str],
        *,
        identity_override: str | None = None,
        resource_id_override: str | None = None,
        matrix_override: MatrixIR | None = None,
        bbox_override: tuple[float, float, float, float] | None = None,
    ) -> None:
        invocation_structure_before = self._take_structure_before(parent_frame)
        form = _object(reference)
        identity = identity_override or _stream_id(
            reference, f"direct-form:{resource_name}"
        )
        if identity in active_forms:
            self.issues.append(f"cyclic Form XObject {identity} was not expanded")
            return
        self.form_invocation_sequence += 1
        instance = (
            f"{identity}/{resource_name}@{self.form_invocation_sequence}:"
            f"{parent_location.global_operator_index}"
        )
        child_path = (*form_path, instance)
        form_resource_id = (
            resource_id_override
            if resource_id_override is not None
            else _resource_id(reference)
        )
        child_resource_scope = (
            (*parent_location.resource_scope, form_resource_id)
            if form_resource_id
            else (*parent_location.resource_scope, "")
        )
        form_matrix_raw = (
            matrix_override
            if matrix_override is not None
            else form.get("/Matrix", (1, 0, 0, 1, 0, 0))
        )
        child_state = parent_frame.state.clone()
        child_state.ctm = multiply_matrix(
            parent_frame.state.ctm,
            _matrix(form_matrix_raw, "Form.Matrix"),
        )
        form_bbox = (
            bbox_override
            if bbox_override is not None
            else _object(form.get("/BBox"))
        )
        if form_bbox is not None and len(form_bbox) == 4:
            left, bottom, right, top = (
                _finite(value, f"Form.BBox[{index}]")
                for index, value in enumerate(form_bbox)
            )
            corners = tuple(
                self._normalized_point(child_state, x, y)
                for x, y in (
                    (left, bottom),
                    (right, bottom),
                    (right, top),
                    (left, top),
                )
            )
            form_bounds = BoundsIR(
                min(point[0] for point in corners),
                min(point[1] for point in corners),
                max(point[0] for point in corners),
                max(point[1] for point in corners),
            )
            child_state.clip = _intersect_bounds(child_state.clip, form_bounds)
        child_frame = _Frame(
            state=child_state,
            structure_before_next_paint=invocation_structure_before,
        )
        resources = form.get("/Resources") or parent_resources
        stream_id = _stream_id(reference, f"form:{self.form_invocation_sequence}")
        location = SourceLocationIR(
            content_stream_id=stream_id,
            stream_operator_index=-1,
            global_operator_index=parent_location.global_operator_index,
            form_instance_path=child_path,
            resource_scope=child_resource_scope,
            graphics_depth=parent_location.graphics_depth + 1,
            page_content_stream_index=parent_location.page_content_stream_index,
        )
        self.boundary("form", True, location, instance)
        self._walk_stream(
            reference,
            stream_id,
            resources,
            child_frame,
            child_path,
            child_resource_scope,
            parent_location.graphics_depth + 1,
            active_forms | {identity},
            parent_location.page_content_stream_index,
            content_boundary=False,
        )
        exit_location = SourceLocationIR(
            content_stream_id=stream_id,
            stream_operator_index=-1,
            global_operator_index=max(0, self.global_operator_index - 1),
            form_instance_path=child_path,
            resource_scope=child_resource_scope,
            graphics_depth=parent_location.graphics_depth + 1,
            page_content_stream_index=parent_location.page_content_stream_index,
        )
        self.boundary("form", False, exit_location, instance)

    def _walk_annotation_appearances(
        self,
        resources: Any,
        frame: _Frame,
        page_content_stream_index: int,
    ) -> int:
        """Append the browser's synthetic ``q /__AnnotationN Do Q`` stream."""

        plans = annotation_appearance_plans(self.page)
        stream_id = f"page:{self.page_number}:annotation-appearances"
        synthetic_operator_index = 0
        for plan in plans:
            q_location = self.location(
                stream_id,
                synthetic_operator_index,
                (),
                (),
                len(frame.state_stack),
                page_content_stream_index,
                self.global_operator_index,
            )
            synthetic_operator_index += 1
            self.global_operator_index += 1
            self._mark_structure_open(frame, SOURCE_STRUCTURE_GRAPHICS_STATE)
            self.boundary("graphics-state", True, q_location)
            frame.state_stack.append(frame.state.clone())

            do_location = self.location(
                stream_id,
                synthetic_operator_index,
                (),
                (),
                len(frame.state_stack),
                page_content_stream_index,
                self.global_operator_index,
            )
            synthetic_operator_index += 1
            self.global_operator_index += 1
            self._walk_form(
                plan.appearance_reference,
                plan.resource_name,
                resources,
                frame,
                do_location,
                (),
                frozenset(),
                identity_override=plan.resource_id,
                resource_id_override=plan.resource_id,
                matrix_override=plan.matrix,
                bbox_override=plan.bbox,
            )

            close_location = self.location(
                stream_id,
                synthetic_operator_index,
                (),
                (),
                len(frame.state_stack),
                page_content_stream_index,
                self.global_operator_index,
            )
            synthetic_operator_index += 1
            self.global_operator_index += 1
            self._mark_structure_close(frame, SOURCE_STRUCTURE_GRAPHICS_STATE)
            self.boundary("graphics-state", False, close_location)
            if frame.state_stack:
                frame.state = frame.state_stack.pop()
            else:  # pragma: no cover - synthetic q/Q is constructed here.
                self.issues.append("annotation graphics-state stack underflow")
        return len(plans)

    def _walk_stream(
        self,
        stream_reference: Any,
        stream_id: str,
        resources: Any,
        frame: _Frame,
        form_path: tuple[str, ...],
        resource_scope: tuple[str, ...],
        depth_base: int,
        active_forms: frozenset[str],
        page_content_stream_index: int,
        *,
        content_boundary: bool,
    ) -> None:
        start_location = SourceLocationIR(
            content_stream_id=stream_id,
            stream_operator_index=-1,
            global_operator_index=self.global_operator_index,
            form_instance_path=form_path,
            resource_scope=resource_scope,
            graphics_depth=depth_base + len(frame.state_stack),
            page_content_stream_index=page_content_stream_index,
        )
        if content_boundary:
            self.boundary("content-stream", True, start_location, stream_id)
        try:
            operation_iterator = iter_content_stream_operations(
                stream_reference, self.reader
            )
        except Exception as error:
            raise SourceContentError(f"cannot parse content stream {stream_id}") from error

        stream_operator_count = 0
        while True:
            # Generator decoding/tokenization is lazy. Isolate only iterator
            # failures as content-stream parse errors; semantic errors in the
            # operation body keep their existing per-command fail-closed path.
            try:
                operands, raw_operator = next(operation_iterator)
            except StopIteration:
                break
            except Exception as error:
                raise SourceContentError(
                    f"cannot parse content stream {stream_id}"
                ) from error
            stream_operator_index = stream_operator_count
            stream_operator_count += 1
            operator = raw_operator.decode("latin-1", errors="replace")
            global_index = self.global_operator_index
            self.global_operator_index += 1
            location = self.location(
                stream_id,
                stream_operator_index,
                form_path,
                resource_scope,
                depth_base + len(frame.state_stack),
                page_content_stream_index,
                global_index,
            )
            try:
                if operator == "q":
                    self._mark_structure_open(
                        frame, SOURCE_STRUCTURE_GRAPHICS_STATE
                    )
                    self.boundary("graphics-state", True, location)
                    frame.state_stack.append(frame.state.clone())
                    continue
                if operator == "Q":
                    self._mark_structure_close(
                        frame, SOURCE_STRUCTURE_GRAPHICS_STATE
                    )
                    self.boundary("graphics-state", False, location)
                    if frame.state_stack:
                        frame.state = frame.state_stack.pop()
                    else:
                        self.issues.append(
                            f"graphics-state underflow at source operator {global_index}"
                        )
                    continue
                if operator == "cm":
                    frame.state.ctm = multiply_matrix(
                        frame.state.ctm, _matrix(operands, "cm")
                    )
                    continue
                if operator == "w":
                    frame.state.line_width = _numbers(operands, 1, operator)[0]
                    continue
                if operator == "J":
                    frame.state.line_cap = max(
                        0, min(2, int(_numbers(operands, 1, operator)[0]))
                    )
                    continue
                if operator == "j":
                    # Match the frozen TypeScript parser's bitwise integer
                    # coercion for the three PDF line-join enum values.
                    frame.state.line_join = max(
                        0, min(2, int(_numbers(operands, 1, operator)[0]))
                    )
                    continue
                if operator == "M":
                    frame.state.miter_limit = max(
                        1.0, _numbers(operands, 1, operator)[0]
                    )
                    continue
                if operator == "d":
                    if len(operands) < 2:
                        raise SourceContentError("d requires a dash array and phase")
                    raw_dash = _object(operands[-2])
                    if not isinstance(raw_dash, (list, tuple)):
                        raise SourceContentError("d dash value is not an array")
                    frame.state.dash_array = tuple(
                        _finite(_object(value), f"d[{index}]")
                        for index, value in enumerate(raw_dash)
                    )
                    frame.state.dash_phase = _finite(
                        _object(operands[-1]), "d.phase"
                    )
                    continue
                if operator == "gs":
                    resource_name = _name(operands[-1]) if operands else ""
                    if not resource_name:
                        raise SourceContentError("gs requires an ExtGState name")
                    resource_dict = _object(resources) or {}
                    ext_states = (
                        _object(resource_dict.get("/ExtGState"))
                        if hasattr(resource_dict, "get")
                        else None
                    )
                    reference = (
                        ext_states.get(f"/{resource_name}")
                        if ext_states is not None and hasattr(ext_states, "get")
                        else None
                    )
                    if reference is None:
                        # Source-aligned PageIR is fail-closed: unlike the
                        # viewer's diagnostic name heuristic, a missing
                        # resource cannot prove authored opacity semantics.
                        raise SourceContentError(
                            f"missing ExtGState resource /{resource_name}"
                        )
                    ext_state = _object(reference)
                    if not hasattr(ext_state, "get"):
                        raise SourceContentError(
                            f"ExtGState /{resource_name} is not a dictionary"
                        )
                    stroke_alpha = ext_state.get("/CA")
                    fill_alpha = ext_state.get("/ca")
                    if stroke_alpha is not None:
                        frame.state.stroke_opacity = _finite(
                            _object(stroke_alpha), f"ExtGState /{resource_name}.CA"
                        )
                    if fill_alpha is not None:
                        frame.state.fill_opacity = _finite(
                            _object(fill_alpha), f"ExtGState /{resource_name}.ca"
                        )
                    blend_value = _object(ext_state.get("/BM"))
                    if isinstance(blend_value, (list, tuple)):
                        blend_value = _object(blend_value[0]) if blend_value else None
                    blend_name = _name(blend_value) if blend_value is not None else ""
                    mapped_blend = _BLEND_MODES.get(blend_name)
                    if mapped_blend is not None:
                        frame.state.blend_mode = mapped_blend
                    continue
                if operator == "RG":
                    frame.state.stroke_color = _rgb(
                        *_numbers(operands, 3, operator)
                    )
                    continue
                if operator == "G":
                    frame.state.stroke_color = _gray(
                        _numbers(operands, 1, operator)[0]
                    )
                    continue
                if operator == "K":
                    frame.state.stroke_color = _cmyk(
                        *_numbers(operands, 4, operator)
                    )
                    continue
                if operator in {"SC", "SCN"}:
                    values = _numeric_operands(operands, operator)
                    if len(values) >= 3:
                        frame.state.stroke_color = _rgb(*values[-3:])
                    elif len(values) == 1:
                        frame.state.stroke_color = _gray(values[0])
                    continue
                if operator == "rg":
                    frame.state.fill_color = _rgb(
                        *_numbers(operands, 3, operator)
                    )
                    continue
                if operator == "g":
                    frame.state.fill_color = _gray(
                        _numbers(operands, 1, operator)[0]
                    )
                    continue
                if operator == "k":
                    frame.state.fill_color = _cmyk(
                        *_numbers(operands, 4, operator)
                    )
                    continue
                if operator in {"sc", "scn"}:
                    values = _numeric_operands(operands, operator)
                    if len(values) >= 3:
                        frame.state.fill_color = _rgb(*values[-3:])
                    elif len(values) == 1:
                        frame.state.fill_color = _gray(values[0])
                    continue
                if operator in {"m", "l", "c", "v", "y", "re", "h"}:
                    self._path_command(frame, operator, operands, location)
                    continue
                if operator in {"s", "b", "b*"}:
                    self._close_path(frame)
                    self._add_path_event(frame, operator, location)
                    continue
                if operator in {"S", "f", "F", "f*", "B", "B*"}:
                    self._add_path_event(frame, operator, location)
                    continue
                if operator == "n":
                    self._apply_pending_clip(frame)
                    frame.path.clear()
                    continue
                if operator in {"W", "W*"}:
                    frame.pending_clip = True
                    continue
                if operator == "BT":
                    self._mark_structure_open(frame, SOURCE_STRUCTURE_TEXT_OBJECT)
                    self.text_object_sequence += 1
                    frame.active_text_object_id = "/".join((
                        *form_path,
                        f"text@{self.text_object_sequence}",
                    ))
                    frame.text_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    frame.line_matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
                    self.boundary(
                        "text-object", True, location, frame.active_text_object_id
                    )
                    continue
                if operator == "ET":
                    self._mark_structure_close(frame, SOURCE_STRUCTURE_TEXT_OBJECT)
                    self.boundary(
                        "text-object", False, location, frame.active_text_object_id
                    )
                    frame.active_text_object_id = ""
                    continue
                if operator == "Tf":
                    if len(operands) >= 2:
                        frame.state.font_resource_name = _name(operands[-2])
                        frame.state.font_size = _finite(operands[-1], "Tf.font_size")
                    continue
                if operator == "Tm":
                    frame.text_matrix = _matrix(operands, "Tm")
                    frame.line_matrix = frame.text_matrix
                    continue
                if operator in {"Td", "TD"}:
                    tx, ty = _numbers(operands, 2, operator)
                    if operator == "TD":
                        frame.state.leading = -ty
                    frame.line_matrix = multiply_matrix(
                        frame.line_matrix,
                        (1.0, 0.0, 0.0, 1.0, tx, ty),
                    )
                    frame.text_matrix = frame.line_matrix
                    continue
                if operator == "T*":
                    frame.line_matrix = multiply_matrix(
                        frame.line_matrix,
                        (1.0, 0.0, 0.0, 1.0, 0.0, -frame.state.leading),
                    )
                    frame.text_matrix = frame.line_matrix
                    continue
                if operator == "Tc":
                    frame.state.char_spacing = _numbers(operands, 1, operator)[0]
                    continue
                if operator == "Tw":
                    frame.state.word_spacing = _numbers(operands, 1, operator)[0]
                    continue
                if operator == "Tz":
                    frame.state.horizontal_scale = (
                        _numbers(operands, 1, operator)[0] / 100.0
                    )
                    continue
                if operator == "TL":
                    frame.state.leading = _numbers(operands, 1, operator)[0]
                    continue
                if operator == "Ts":
                    frame.state.rise = _numbers(operands, 1, operator)[0]
                    continue
                if operator == "Tr":
                    frame.state.render_mode = max(
                        0, min(7, int(_numbers(operands, 1, "Tr")[0]))
                    )
                    continue
                if operator == "Tj" and operands:
                    self._show_text(
                        frame, resources, operator, operands[-1], None, location
                    )
                    continue
                if operator == "TJ" and operands:
                    array = _object(operands[-1])
                    for item_index, item in enumerate(array):
                        if _is_pdf_string(item):
                            self._show_text(
                                frame,
                                resources,
                                operator,
                                item,
                                item_index,
                                location,
                            )
                        else:
                            adjustment_value = self._optional_number(item)
                            if adjustment_value is not None:
                                adjustment = (
                                    (-adjustment_value / 1000.0)
                                    * frame.state.font_size
                                    * frame.state.horizontal_scale
                                )
                                frame.text_matrix = multiply_matrix(
                                    frame.text_matrix,
                                    (1.0, 0.0, 0.0, 1.0, adjustment, 0.0),
                                )
                    continue
                if operator == "'":
                    frame.line_matrix = multiply_matrix(
                        frame.line_matrix,
                        (
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            -frame.state.leading,
                        ),
                    )
                    frame.text_matrix = frame.line_matrix
                    if operands and _is_pdf_string(operands[-1]):
                        self._show_text(
                            frame, resources, operator, operands[-1], None, location
                        )
                    continue
                if operator == '"':
                    if len(operands) >= 3:
                        word_spacing = self._optional_number(operands[-3])
                        char_spacing = self._optional_number(operands[-2])
                        if word_spacing is not None:
                            frame.state.word_spacing = word_spacing
                        if char_spacing is not None:
                            frame.state.char_spacing = char_spacing
                    frame.line_matrix = multiply_matrix(
                        frame.line_matrix,
                        (
                            1.0,
                            0.0,
                            0.0,
                            1.0,
                            0.0,
                            -frame.state.leading,
                        ),
                    )
                    frame.text_matrix = frame.line_matrix
                    if operands and _is_pdf_string(operands[-1]):
                        self._show_text(
                            frame, resources, operator, operands[-1], None, location
                        )
                    continue
                if operator == "Do" and operands:
                    resource_name = _name(operands[-1])
                    resource_dict = _object(resources) or {}
                    xobjects = _object(resource_dict.get("/XObject")) or {}
                    reference = xobjects.get(f"/{resource_name}")
                    if reference is None:
                        self._unsupported_paint(
                            operator,
                            f"missing XObject /{resource_name}",
                            location,
                            frame,
                        )
                        continue
                    xobject = _object(reference)
                    subtype = str(xobject.get("/Subtype", ""))
                    if subtype == "/Image":
                        self._add_image(reference, resource_name, location, frame)
                    elif subtype == "/Form":
                        self._walk_form(
                            reference,
                            resource_name,
                            resources,
                            frame,
                            location,
                            form_path,
                            active_forms,
                        )
                    else:
                        self._unsupported_paint(
                            operator,
                            f"unsupported XObject subtype {subtype or '<missing>'}",
                            location,
                            frame,
                        )
                    continue
                if operator == "INLINE IMAGE":
                    self._add_inline_image_skip(operands, location, frame)
                    continue
                if operator == "sh":
                    self._unsupported_paint(
                        operator,
                        "shading paint is not representable in PageIR",
                        location,
                        frame,
                    )
                    continue
                if operator in {"BMC", "BDC"}:
                    self._mark_structure_open(
                        frame, SOURCE_STRUCTURE_MARKED_CONTENT
                    )
                    self.boundary("marked-content", True, location, operator)
                    continue
                if operator == "EMC":
                    self._mark_structure_close(
                        frame, SOURCE_STRUCTURE_MARKED_CONTENT
                    )
                    self.boundary("marked-content", False, location, operator)
                    continue
                if operator == "BX":
                    self._mark_structure_open(
                        frame, SOURCE_STRUCTURE_COMPATIBILITY
                    )
                    self.boundary("compatibility", True, location)
                    continue
                if operator == "EX":
                    self._mark_structure_close(
                        frame, SOURCE_STRUCTURE_COMPATIBILITY
                    )
                    self.boundary("compatibility", False, location)
                    continue
                # Remaining styling, text-position and marked-point operators
                # do not create paint identities. The source-alignment style
                # fields (w/J/j, colours and ExtGState opacity) are handled
                # above; pypdf has already validated the rest lexically.
            except SourceContentError as error:
                self.issues.append(
                    f"{stream_id} operator {stream_operator_index} {operator}: {error}"
                )

        end_location = SourceLocationIR(
            content_stream_id=stream_id,
            stream_operator_index=stream_operator_count,
            global_operator_index=max(0, self.global_operator_index - 1),
            form_instance_path=form_path,
            resource_scope=resource_scope,
            graphics_depth=depth_base + len(frame.state_stack),
            page_content_stream_index=page_content_stream_index,
        )
        if content_boundary:
            self.boundary("content-stream", False, end_location, stream_id)

    def parse(self) -> SourceContentPageIR:
        contents = self.page.get("/Contents")
        resources = self.page.get("/Resources") or {}
        frame = _Frame(state=_GraphicsState())
        streams: list[Any] = []
        if contents is not None:
            resolved = _object(contents)
            streams = list(resolved) if type(resolved).__name__ == "ArrayObject" else [contents]
            for index, stream in enumerate(streams):
                stream_id = _stream_id(stream, f"page:{self.page_number}:stream:{index}")
                self._walk_stream(
                    stream,
                    stream_id,
                    resources,
                    frame,
                    (),
                    (),
                    0,
                    frozenset(),
                    index,
                    content_boundary=True,
                )
        annotation_appearance_count = self._walk_annotation_appearances(
            resources,
            frame,
            len(streams),
        )
        return SourceContentPageIR(
            page_number=self.page_number,
            source_sha256=self.source_sha256,
            source_name=self.source_name,
            page_bounds=self.page_bounds,
            rotation_degrees=self.rotation_degrees,
            events=tuple(self.events),
            boundaries=tuple(self.boundaries),
            issues=tuple(self.issues),
            annotation_appearance_count=annotation_appearance_count,
            producer_version=self.producer_version,
        )


def source_content_page_from_pdf_bytes(
    source: bytes | bytearray | memoryview,
    page_number: int = 1,
    *,
    source_name: str = "",
) -> SourceContentPageIR:
    """Recover one page's authored content-stream paint provenance."""

    with SourceContentDocument(source, source_name=source_name) as document:
        return document.page(page_number)


class SourceContentDocument:
    """Once-opened pypdf view used by whole-document recognition.

    The immutable source bytes are copied into one ``BytesIO`` only once.
    Page extraction then parses only that page's content streams and resources;
    it never re-reads or re-opens the complete PDF.
    """

    def __init__(
        self,
        source: bytes | bytearray | memoryview,
        *,
        source_name: str = "",
    ) -> None:
        self._source = bytes(source)
        self.source_sha256 = sha256(self._source).hexdigest()
        self.source_name = source_name
        self._buffer = BytesIO(self._source)
        try:
            pypdf_runtime = assert_supported_pypdf_runtime()
            from pypdf import PdfReader
        except UnsupportedRuntimeError:
            self._buffer.close()
            raise
        except ImportError as error:  # pragma: no cover - dependency guard.
            self._buffer.close()
            raise SourceContentError(
                "pypdf is required for source-content extraction "
                "(install pypdf==6.14.2)"
            ) from error
        try:
            self._reader = PdfReader(self._buffer, strict=False)
        except Exception as error:
            self._buffer.close()
            raise SourceContentError("pypdf could not open the supplied PDF bytes") from error
        if self._reader.is_encrypted:
            self._buffer.close()
            raise SourceContentError(
                "encrypted PDFs require decryption before source parsing"
            )
        self.producer_version = pypdf_runtime.module_version
        self._closed = False

    @property
    def page_count(self) -> int:
        return len(self._reader.pages)

    def page(self, page_number: int) -> SourceContentPageIR:
        if self._closed:
            raise SourceContentError("source-content document is closed")
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
        return _SourceParser(
            self._reader,
            self._reader.pages[page_number - 1],
            page_number,
            self.source_sha256,
            self.source_name,
            self.producer_version,
        ).parse()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._buffer.close()

    def __enter__(self) -> "SourceContentDocument":
        if self._closed:
            raise SourceContentError("source-content document is closed")
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


__all__ = [
    "SOURCE_CONTENT_VERSION",
    "SOURCE_STRUCTURE_COMPATIBILITY",
    "SOURCE_STRUCTURE_GRAPHICS_STATE",
    "SOURCE_STRUCTURE_MARKED_CONTENT",
    "SOURCE_STRUCTURE_TEXT_OBJECT",
    "SourceBoundaryIR",
    "SourceContentError",
    "SourceContentDocument",
    "SourceContentPageIR",
    "SourceEventIR",
    "SourceImagePaintEventIR",
    "SourceInlineImageSkipEventIR",
    "SourceLocationIR",
    "SourcePathCommandIR",
    "SourcePathPaintEventIR",
    "SourceTextShowEventIR",
    "SourceUnsupportedPaintEventIR",
    "multiply_matrix",
    "source_content_page_from_pdf_bytes",
    "transform_point",
]
