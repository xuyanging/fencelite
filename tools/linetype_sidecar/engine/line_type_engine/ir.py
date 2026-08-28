"""Versioned, browser-neutral PDF page intermediate representation.

Coordinates use a PDF-style bottom-left origin.  ``paint_order`` is the
PyMuPDF display-list sequence number and may repeat; ``ordinal`` is a dense
per-kind index and is never used as a substitute for paint order.  The dense
position in ``PageIR.operations`` is the unique integer operation identity used
by recognition results.  The representation stores only
data required by recognition or reproducible visualization, not PyMuPDF
objects, DOM objects, or frontend state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, Mapping, TypeAlias

from .versions import GROUPING_IR_VERSION, PAGE_IR_VERSION


ColorIR: TypeAlias = tuple[float, ...]
PointIR: TypeAlias = tuple[float, float]
MatrixIR: TypeAlias = tuple[float, float, float, float, float, float]

_INDIRECT_OBJECT_ID = re.compile(r"^([1-9]\d*) (0|[1-9]\d*) R$")
_RESOURCE_SCOPE_ID = re.compile(
    r"^[1-9]\d* (?:0|[1-9]\d*) R(?::appearance)?$"
)


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return 0.0 if number == 0 else number


@dataclass(frozen=True, slots=True)
class BoundsIR:
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def __post_init__(self) -> None:
        values = tuple(_finite(value, "bounds coordinate") for value in (
            self.min_x, self.min_y, self.max_x, self.max_y
        ))
        if values[0] > values[2] or values[1] > values[3]:
            raise ValueError("bounds minimum must not exceed maximum")
        object.__setattr__(self, "min_x", values[0])
        object.__setattr__(self, "min_y", values[1])
        object.__setattr__(self, "max_x", values[2])
        object.__setattr__(self, "max_y", values[3])

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    def union(self, other: "BoundsIR") -> "BoundsIR":
        """两个已校验 bounds 的并集，**跳过 __post_init__ 的重复校验**。

        这是恒等变换，不是放宽校验：
          * 有限性 —— min/max 只会返回两个输入之一，而两者在各自构造时都已
            通过 _finite；不产生新数值，就不可能引入 inf/nan。
          * min <= max —— self 与 other 各自满足 min_x <= max_x，于是
            min(a.min_x, b.min_x) <= min(a.max_x, b.max_x) <= max(a.max_x, b.max_x)，
            y 轴同理。
          * -0.0 归一 —— _finite 已把输入里的 -0.0 变成 0.0；min/max 只做选择，
            不会凭空造出 -0.0。
        为什么值得这么写：分组阶段实测 1780 万次 union（gladstone P2 单页），
        走完整构造时 __post_init__ + _finite + 生成器合计占整页 CPU 的 28%
        （38s / 133s），而上面三条说明那份校验对 union 的结果永远为真。
        """
        result = object.__new__(BoundsIR)
        setter = object.__setattr__
        a, b = self.min_x, other.min_x
        setter(result, "min_x", a if a <= b else b)
        a, b = self.min_y, other.min_y
        setter(result, "min_y", a if a <= b else b)
        a, b = self.max_x, other.max_x
        setter(result, "max_x", a if a >= b else b)
        a, b = self.max_y, other.max_y
        setter(result, "max_y", a if a >= b else b)
        return result


@dataclass(frozen=True, slots=True)
class PathSegmentIR:
    kind: Literal["move", "line", "curve", "close"]
    end: PointIR | None = None
    control_1: PointIR | None = None
    control_2: PointIR | None = None

    def __post_init__(self) -> None:
        if self.kind in {"move", "line"} and self.end is None:
            raise ValueError(f"{self.kind} segment requires an endpoint")
        if self.kind == "curve" and (
            self.end is None or self.control_1 is None or self.control_2 is None
        ):
            raise ValueError("curve segment requires two controls and an endpoint")
        if self.kind == "close" and any((self.end, self.control_1, self.control_2)):
            raise ValueError("close segment must not contain points")
        if self.kind not in {"move", "line", "curve", "close"}:
            raise ValueError(f"unsupported path segment kind: {self.kind}")
        for name in ("end", "control_1", "control_2"):
            point = getattr(self, name)
            if point is not None:
                object.__setattr__(self, name, (
                    _finite(point[0], f"{name}.x"),
                    _finite(point[1], f"{name}.y"),
                ))


@dataclass(frozen=True, slots=True)
class PathOperationIR:
    operation_id: str
    paint_order: int
    ordinal: int
    bounds: BoundsIR
    segments: tuple[PathSegmentIR, ...]
    stroke: bool
    fill: bool
    stroke_color: ColorIR | None = None
    fill_color: ColorIR | None = None
    line_width: float = 0.0
    hairline: bool = False
    line_cap: tuple[int, int, int] = (0, 0, 0)
    line_join: float = 0.0
    miter_limit: float = 10.0
    dash_array: tuple[float, ...] = ()
    dash_phase: float = 0.0
    stroke_opacity: float = 1.0
    fill_opacity: float = 1.0
    even_odd: bool = False
    close_path: bool = False
    blend_mode: str = "Normal"
    layer: str = ""
    nesting_level: int = 0
    structure_before: int = 0
    content_stream_index: int = 0
    form_instance_path: tuple[str, ...] = ()
    source_provenance_exact: bool = False

    kind: Literal["path"] = "path"

    def __post_init__(self) -> None:
        _validate_operation_identity(self.operation_id, self.paint_order, self.ordinal)
        _validate_source_provenance(
            self.structure_before,
            self.content_stream_index,
            self.form_instance_path,
            self.source_provenance_exact,
        )
        if not self.stroke and not self.fill:
            raise ValueError("path operation must stroke, fill, or both")
        if not self.segments:
            raise ValueError("path operation requires at least one segment")
        object.__setattr__(self, "line_width", _finite(self.line_width, "line_width"))
        object.__setattr__(self, "line_join", _finite(self.line_join, "line_join"))
        object.__setattr__(self, "miter_limit", _finite(self.miter_limit, "miter_limit"))
        object.__setattr__(self, "dash_array", tuple(
            _finite(value, "dash_array") for value in self.dash_array
        ))
        object.__setattr__(self, "dash_phase", _finite(self.dash_phase, "dash_phase"))
        object.__setattr__(self, "stroke_opacity", _finite(self.stroke_opacity, "stroke_opacity"))
        object.__setattr__(self, "fill_opacity", _finite(self.fill_opacity, "fill_opacity"))


@dataclass(frozen=True, slots=True)
class TextCharacterIR:
    codepoint: int
    glyph_id: int
    origin: PointIR
    bounds: BoundsIR

    def __post_init__(self) -> None:
        if not isinstance(self.codepoint, int) or self.codepoint < 0:
            raise ValueError("text codepoint must be a non-negative integer")
        if not isinstance(self.glyph_id, int):
            raise ValueError("glyph_id must be an integer")
        object.__setattr__(self, "origin", (
            _finite(self.origin[0], "origin.x"),
            _finite(self.origin[1], "origin.y"),
        ))


@dataclass(frozen=True, slots=True)
class TextOperationIR:
    operation_id: str
    paint_order: int
    ordinal: int
    span_index: int
    bounds: BoundsIR
    literal_text: str
    characters: tuple[TextCharacterIR, ...]
    font_name: str
    font_size: float
    direction: PointIR
    render_mode: int
    color: ColorIR | None
    opacity: float
    line_width: float | None = None
    writing_mode: int = 0
    flags: int = 0
    bidi_level: int = 0
    bidi_direction: int = 0
    layer: str = ""
    structure_before: int = 0
    content_stream_index: int = 0
    form_instance_path: tuple[str, ...] = ()
    source_provenance_exact: bool = False

    # The fields above are the independent MuPDF text trace retained for
    # glyph/content audit.  Source-aligned pages additionally carry the exact
    # authored values consumed by the frozen browser Scene.  Keeping the two
    # domains separate prevents a display-font name or clipped trace bbox from
    # silently becoming recognition input.
    source_font_name: str | None = None
    source_font_size: float | None = None
    source_matrix: MatrixIR | None = None
    source_glyph_advance: float | None = None
    source_horizontal_scale: float | None = None
    source_rise: float | None = None
    source_unclipped_bounds: BoundsIR | None = None
    source_fill_color: ColorIR | None = None
    source_stroke_color: ColorIR | None = None
    source_fill_opacity: float | None = None
    source_stroke_opacity: float | None = None
    source_line_width: float | None = None
    source_blend_mode: str | None = None

    kind: Literal["text"] = "text"

    def __post_init__(self) -> None:
        _validate_operation_identity(self.operation_id, self.paint_order, self.ordinal)
        _validate_source_provenance(
            self.structure_before,
            self.content_stream_index,
            self.form_instance_path,
            self.source_provenance_exact,
        )
        if self.span_index < 0:
            raise ValueError("span_index must be non-negative")
        object.__setattr__(self, "font_size", _finite(self.font_size, "font_size"))
        object.__setattr__(self, "opacity", _finite(self.opacity, "text opacity"))
        object.__setattr__(self, "direction", (
            _finite(self.direction[0], "direction.x"),
            _finite(self.direction[1], "direction.y"),
        ))
        if self.line_width is not None:
            object.__setattr__(self, "line_width", _finite(self.line_width, "text line_width"))
        if not isinstance(self.render_mode, int) or isinstance(self.render_mode, bool) or not (
            0 <= self.render_mode <= 7
        ):
            raise ValueError("text render_mode must be an integer from 0 through 7")
        source_fields = (
            self.source_font_name,
            self.source_font_size,
            self.source_matrix,
            self.source_glyph_advance,
            self.source_horizontal_scale,
            self.source_rise,
            self.source_unclipped_bounds,
            self.source_fill_color,
            self.source_stroke_color,
            self.source_fill_opacity,
            self.source_stroke_opacity,
            self.source_line_width,
            self.source_blend_mode,
        )
        if self.source_provenance_exact and any(value is None for value in source_fields):
            raise ValueError("source-aligned text requires the complete authored text state")
        if any(value is not None for value in source_fields):
            if any(value is None for value in source_fields):
                raise ValueError("authored text state must be entirely present or absent")
            assert self.source_font_name is not None
            assert self.source_font_size is not None
            assert self.source_matrix is not None
            assert self.source_glyph_advance is not None
            assert self.source_horizontal_scale is not None
            assert self.source_rise is not None
            assert self.source_fill_color is not None
            assert self.source_stroke_color is not None
            assert self.source_fill_opacity is not None
            assert self.source_stroke_opacity is not None
            assert self.source_line_width is not None
            assert self.source_blend_mode is not None
            if not self.source_font_name or not self.source_blend_mode:
                raise ValueError("authored text font and blend mode must be non-empty")
            source_font_size = _finite(self.source_font_size, "source_font_size")
            source_glyph_advance = _finite(
                self.source_glyph_advance, "source_glyph_advance"
            )
            if source_font_size < 0.1 or source_glyph_advance < 0.001:
                raise ValueError("authored text size/advance violate the frozen Scene floor")
            object.__setattr__(self, "source_font_size", source_font_size)
            object.__setattr__(self, "source_glyph_advance", source_glyph_advance)
            object.__setattr__(
                self,
                "source_matrix",
                tuple(_finite(value, "source_matrix") for value in self.source_matrix),
            )
            object.__setattr__(
                self,
                "source_horizontal_scale",
                _finite(self.source_horizontal_scale, "source_horizontal_scale"),
            )
            object.__setattr__(self, "source_rise", _finite(self.source_rise, "source_rise"))
            object.__setattr__(
                self,
                "source_fill_color",
                tuple(_finite(value, "source_fill_color") for value in self.source_fill_color),
            )
            object.__setattr__(
                self,
                "source_stroke_color",
                tuple(
                    _finite(value, "source_stroke_color")
                    for value in self.source_stroke_color
                ),
            )
            object.__setattr__(
                self,
                "source_fill_opacity",
                _finite(self.source_fill_opacity, "source_fill_opacity"),
            )
            object.__setattr__(
                self,
                "source_stroke_opacity",
                _finite(self.source_stroke_opacity, "source_stroke_opacity"),
            )
            source_line_width = _finite(self.source_line_width, "source_line_width")
            if source_line_width < 0.25:
                raise ValueError("authored text line width violates the frozen Scene floor")
            object.__setattr__(self, "source_line_width", source_line_width)

    @property
    def canonical_font_name(self) -> str:
        return self.source_font_name or self.font_name

    @property
    def canonical_font_size(self) -> float:
        return self.source_font_size if self.source_font_size is not None else self.font_size


@dataclass(frozen=True, slots=True)
class ImageOperationIR:
    operation_id: str
    paint_order: int
    ordinal: int
    bounds: BoundsIR
    pixel_width: int
    pixel_height: int
    colorspace: int
    bits_per_component: int = 0
    color_space_name: str = ""
    source_filters: tuple[str, ...] = ()
    xref: int = 0
    digest: str = ""
    transform: tuple[float, float, float, float, float, float] | None = None
    # Source-content identity is deliberately separate from the display-list
    # digest/transform.  MuPDF may report an image or mask xref that differs
    # from the XObject referenced by the authored ``Do`` operator.
    resource_name: str = ""
    resource_id: str = ""
    resource_scope: tuple[str, ...] = ()
    corners: tuple[PointIR, ...] = ()
    visible_bounds: BoundsIR | None = None
    paint_operator: str = ""
    image_mask: bool = False
    target_placeholder_reason: str | None = None
    alpha: float = 1.0
    blend_mode: str = "source-over"
    structure_before: int = 0
    content_stream_index: int = 0
    form_instance_path: tuple[str, ...] = ()
    source_provenance_exact: bool = False

    kind: Literal["image"] = "image"

    def __post_init__(self) -> None:
        _validate_operation_identity(self.operation_id, self.paint_order, self.ordinal)
        _validate_source_provenance(
            self.structure_before,
            self.content_stream_index,
            self.form_instance_path,
            self.source_provenance_exact,
        )
        if self.pixel_width < 0 or self.pixel_height < 0:
            raise ValueError("image dimensions must be non-negative")
        if self.bits_per_component < 0:
            raise ValueError("image bits_per_component must be non-negative")
        if not isinstance(self.color_space_name, str):
            raise ValueError("image color_space_name must be a string")
        if not isinstance(self.source_filters, tuple) or any(
            not isinstance(item, str) or not item.startswith("/")
            for item in self.source_filters
        ):
            raise ValueError("image source_filters must contain canonical PDF names")
        if isinstance(self.xref, bool) or not isinstance(self.xref, int) or self.xref < 0:
            raise ValueError("image xref must be a non-negative integer")
        if not isinstance(self.image_mask, bool):
            raise ValueError("image_mask must be boolean")
        object.__setattr__(self, "alpha", _finite(self.alpha, "image alpha"))
        if not isinstance(self.blend_mode, str) or not self.blend_mode:
            raise ValueError("image blend_mode must be a non-empty string")
        expected_reason = (
            "IMAGE_MASK_UNSUPPORTED"
            if self.image_mask
            else (
                "IMAGE_JPX_UNSUPPORTED"
                if "/JPXDecode" in self.source_filters
                else None
            )
        )
        if self.target_placeholder_reason == "IMAGE_PIXEL_BUDGET" and expected_reason is None:
            expected_reason = "IMAGE_PIXEL_BUDGET"
        if self.target_placeholder_reason != expected_reason:
            raise ValueError(
                "target_placeholder_reason does not match source mask/filter semantics"
            )
        if self.transform is not None:
            if len(self.transform) != 6:
                raise ValueError("image transform must contain exactly six values")
            object.__setattr__(self, "transform", tuple(
                _finite(value, "image transform") for value in self.transform
            ))
        if not isinstance(self.resource_scope, tuple) or any(
            not isinstance(item, str) for item in self.resource_scope
        ):
            raise ValueError("image resource_scope must contain strings")
        if self.corners:
            if len(self.corners) != 4:
                raise ValueError("image corners must contain exactly four points")
            object.__setattr__(self, "corners", tuple(
                (
                    _finite(point[0], "image corner.x"),
                    _finite(point[1], "image corner.y"),
                )
                for point in self.corners
            ))
        if self.source_provenance_exact:
            if not self.resource_name or self.paint_operator != "Do":
                raise ValueError("source-aligned image requires its resource name and Do operator")
            resource_match = _INDIRECT_OBJECT_ID.fullmatch(self.resource_id)
            if resource_match is None or int(resource_match.group(1)) != self.xref:
                raise ValueError(
                    "source-aligned image requires an indirect resource id matching xref"
                )
            if any(_RESOURCE_SCOPE_ID.fullmatch(item) is None for item in self.resource_scope):
                raise ValueError(
                    "source-aligned image resource_scope must contain indirect object ids "
                    "or strict annotation-appearance ids"
                )
            if len(self.corners) != 4 or self.visible_bounds is None:
                raise ValueError("source-aligned image requires exact corners and visible bounds")
            if self.bounds != self.visible_bounds:
                raise ValueError("source-aligned image bounds must equal visible_bounds")

    @property
    def target_kind(self) -> Literal["image", "placeholder"]:
        return "placeholder" if self.target_placeholder_reason is not None else "image"


OperationIR: TypeAlias = PathOperationIR | TextOperationIR | ImageOperationIR


def _validate_operation_identity(operation_id: str, paint_order: int, ordinal: int) -> None:
    if not operation_id:
        raise ValueError("operation_id must not be empty")
    if not isinstance(paint_order, int) or paint_order < 0:
        raise ValueError("paint_order must be a non-negative integer")
    if not isinstance(ordinal, int) or ordinal < 0:
        raise ValueError("ordinal must be a non-negative integer")


def _validate_source_provenance(
    structure_before: int,
    content_stream_index: int,
    form_instance_path: tuple[str, ...],
    source_provenance_exact: bool,
) -> None:
    if (
        isinstance(structure_before, bool)
        or not isinstance(structure_before, int)
        or structure_before < 0
        or structure_before & ~0xF
    ):
        raise ValueError("structure_before must be a supported non-negative bit mask")
    if (
        isinstance(content_stream_index, bool)
        or not isinstance(content_stream_index, int)
        or content_stream_index < 0
    ):
        raise ValueError("content_stream_index must be non-negative")
    if not isinstance(form_instance_path, tuple) or any(
        not isinstance(item, str) or not item for item in form_instance_path
    ):
        raise ValueError("form_instance_path must contain non-empty strings")
    if not isinstance(source_provenance_exact, bool):
        raise ValueError("source_provenance_exact must be boolean")


@dataclass(frozen=True, slots=True)
class PageIR:
    page_number: int
    page_bounds: BoundsIR
    rotation_degrees: int
    operations: tuple[OperationIR, ...]
    source_sha256: str
    source_name: str = ""
    page_ir_version: str = PAGE_IR_VERSION
    producer: str = "PyMuPDF"
    producer_version: str = ""
    _fingerprint_cache: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"canonical": False},
    )

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("page_number must be one-based")
        if self.rotation_degrees % 90:
            raise ValueError("page rotation must be a multiple of 90 degrees")
        if len(self.source_sha256) != 64:
            raise ValueError("source_sha256 must be a lowercase SHA-256 hex digest")
        if self.source_sha256.lower() != self.source_sha256 or any(
            char not in "0123456789abcdef" for char in self.source_sha256
        ):
            raise ValueError("source_sha256 must be lowercase hexadecimal")
        ids = [operation.operation_id for operation in self.operations]
        if len(ids) != len(set(ids)):
            raise ValueError("operation_id values must be unique within a page")
        order = [(operation.paint_order, operation.ordinal) for operation in self.operations]
        if order != sorted(order):
            raise ValueError("page operations must be sorted by paint_order then ordinal")

    @property
    def fingerprint(self) -> str:
        cached = self._fingerprint_cache
        if cached is None:
            cached = canonical_fingerprint(self)
            object.__setattr__(self, "_fingerprint_cache", cached)
        return cached


@dataclass(frozen=True, slots=True)
class GroupIR:
    group_id: str
    operation_ids: tuple[str, ...]
    bounds: BoundsIR
    first_paint_order: int
    last_paint_order: int
    split_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.group_id or not self.operation_ids:
            raise ValueError("group requires an id and at least one operation")
        if len(self.operation_ids) != len(set(self.operation_ids)):
            raise ValueError("group operation ids must be unique")
        if self.first_paint_order > self.last_paint_order:
            raise ValueError("group paint-order range is inverted")


@dataclass(frozen=True, slots=True)
class GroupingIR:
    page_fingerprint: str
    groups: tuple[GroupIR, ...]
    assignments: tuple[tuple[str, str], ...]
    grouping_version: str = GROUPING_IR_VERSION
    _fingerprint_cache: str | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
        metadata={"canonical": False},
    )

    def __post_init__(self) -> None:
        if len(self.page_fingerprint) != 64:
            raise ValueError("page_fingerprint must be a SHA-256 digest")
        operation_ids = [operation_id for operation_id, _ in self.assignments]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("an operation may be assigned only once")
        known_groups = {group.group_id for group in self.groups}
        if any(group_id not in known_groups for _, group_id in self.assignments):
            raise ValueError("assignment references an unknown group")

    @property
    def fingerprint(self) -> str:
        cached = self._fingerprint_cache
        if cached is None:
            cached = canonical_fingerprint(self)
            object.__setattr__(self, "_fingerprint_cache", cached)
        return cached


def _canonical_value(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
            if field.metadata.get("canonical", True)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical JSON does not support non-finite floats")
        rounded = round(value, 9)
        return 0.0 if rounded == 0 else rounded
    return value


def canonical_json(value: Any) -> str:
    """Return deterministic UTF-8 JSON for hashes, caches and parity tools."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_fingerprint(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def page_ir_to_dict(page: PageIR) -> dict[str, Any]:
    """Return the lossless public JSON mapping (fingerprints stay canonical)."""

    # Runtime import avoids an ir -> codec -> ir import cycle while retaining
    # the historical direct ``line_type_engine.ir`` entry point.
    from .ir_codec import page_ir_to_dict as encode_page_ir

    return encode_page_ir(page)
