"""Serialize Python PageIR Groups for the validated Method1 atom parser.

The base Method1 recognizer consumes a compact subset of PDF path commands.
Keeping this adapter isolated makes the legacy command parser replaceable
without coupling PDF extraction, Grouping, or higher Method1 stages to it.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Any, Iterable

from ..ir import GroupingIR, PageIR, PathOperationIR
from ..operation_index import PageOperationIndex
from ..results import RecognizedGroup


@dataclass(frozen=True, slots=True)
class SerializedGroup:
    group_id: str
    commands: str
    atom_op_indices: tuple[int, ...]
    force_non_linetype: str | None = None

    def to_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "group_id": self.group_id,
            "commands": self.commands,
            "atom_op_indices": list(self.atom_op_indices),
        }
        if self.force_non_linetype is not None:
            output["force_non_linetype"] = self.force_non_linetype
        return output


METHOD1_SERIALIZED_INPUT_HASH_SCHEMA = "method1-serialized-input-v1"


def serialized_method1_input_hash(
    groups: Iterable[SerializedGroup],
) -> str:
    """Match the frozen TS ``createMethod1InputHash`` byte for byte.

    The already-produced serializer payload is used, so corpus runs persist
    input-space proof without parsing a page a second time.
    """

    serialized = tuple(groups)
    digest = sha256()
    digest.update(METHOD1_SERIALIZED_INPUT_HASH_SCHEMA.encode("utf-8") + b"\0")

    def field(value: object) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b":")
        digest.update(encoded)
        digest.update(b";")

    field(len(serialized))
    for group in serialized:
        field(group.group_id)
        field(group.commands)
        field(group.force_non_linetype or "")
        field(len(group.atom_op_indices))
        for op_index in group.atom_op_indices:
            field(op_index)
    return digest.hexdigest()


def _number(value: float) -> str:
    finite = float(value) if math.isfinite(float(value)) else 0.0
    if abs(finite) < 1e-10:
        finite = 0.0
    fixed = f"{finite:.6f}"
    if "." in fixed:
        fixed = fixed.rstrip("0").rstrip(".")
    return "0" if fixed in {"", "-0"} else fixed


def _rgb(color: tuple[float, ...] | None) -> tuple[float, float, float]:
    if not color:
        return (0.0, 0.0, 0.0)
    channels = tuple(max(0.0, min(1.0, float(value))) for value in color)
    if len(channels) == 1:
        return (channels[0], channels[0], channels[0])
    if len(channels) >= 4:
        cyan, magenta, yellow, black = channels[:4]
        return (
            1.0 - min(1.0, cyan + black),
            1.0 - min(1.0, magenta + black),
            1.0 - min(1.0, yellow + black),
        )
    if len(channels) == 2:
        return (channels[0], channels[1], 0.0)
    return (channels[0], channels[1], channels[2])


def serialize_path(operation: PathOperationIR) -> tuple[str, int] | None:
    """Return command text and drawable subpath multiplicity."""

    lines = [f"{_number(operation.line_width)} w", f"{operation.line_cap[0]} J"]
    red, green, blue = _rgb(operation.stroke_color or operation.fill_color)
    lines.append(f"{_number(red)} {_number(green)} {_number(blue)} RG")
    subpath_has_draw = False
    valid_subpath_count = 0
    for segment in operation.segments:
        if segment.kind == "move":
            if subpath_has_draw:
                valid_subpath_count += 1
            subpath_has_draw = False
            assert segment.end is not None
            lines.append(f"{_number(segment.end[0])} {_number(segment.end[1])} m")
        elif segment.kind == "line":
            subpath_has_draw = True
            assert segment.end is not None
            lines.append(f"{_number(segment.end[0])} {_number(segment.end[1])} l")
        elif segment.kind == "curve":
            subpath_has_draw = True
            assert segment.control_1 is not None
            assert segment.control_2 is not None
            assert segment.end is not None
            lines.append(
                f"{_number(segment.control_1[0])} {_number(segment.control_1[1])} "
                f"{_number(segment.control_2[0])} {_number(segment.control_2[1])} "
                f"{_number(segment.end[0])} {_number(segment.end[1])} c"
            )
        else:
            lines.append("h")
    if subpath_has_draw:
        valid_subpath_count += 1
    if valid_subpath_count == 0:
        return None
    if operation.stroke and operation.fill:
        paint = "B*" if operation.even_odd else "B"
    elif operation.stroke:
        paint = "S"
    else:
        paint = "f*" if operation.even_odd else "f"
    lines.append(paint)
    return ("\n".join(lines), valid_subpath_count)


def _is_dense_two_dimensional_layer(
    operations: tuple[PathOperationIR, ...],
    atom_count: int,
) -> bool:
    if atom_count < 20_000 or not operations:
        return False
    stride = max(1, math.ceil(len(operations) / 100_000))
    centers = tuple(
        (
            (operation.bounds.min_x + operation.bounds.max_x) / 2.0,
            (operation.bounds.min_y + operation.bounds.max_y) / 2.0,
        )
        for operation in operations[::stride]
    )
    if len(centers) < 512:
        return False
    min_x = min(point[0] for point in centers)
    max_x = max(point[0] for point in centers)
    min_y = min(point[1] for point in centers)
    max_y = max(point[1] for point in centers)
    span_x = max_x - min_x
    span_y = max_y - min_y
    if min(span_x, span_y) <= max(span_x, span_y) * 0.08:
        return False
    occupied_cells = {
        min(63, math.floor((x - min_x) / max(1e-9, span_x) * 64))
        + 64 * min(63, math.floor((y - min_y) / max(1e-9, span_y) * 64))
        for x, y in centers
    }
    return len(occupied_cells) >= 512 or (
        len(occupied_cells) >= 100 and atom_count / len(occupied_cells) >= 50
    )


def serialize_groups(page: PageIR, grouping: GroupingIR) -> tuple[SerializedGroup, ...]:
    """Build a complete ordered Group payload for Method1 base recognition.

    Numeric ownership is the dense position in ``PageIR.operations``.
    ``paint_order`` remains authored ordering evidence and may repeat, so it
    must never identify a recognition-result operation.
    """

    operation_index = PageOperationIndex.build(page, grouping)

    result: list[SerializedGroup] = []
    for group in grouping.groups:
        command_blocks: list[str] = []
        atom_op_indices: list[int] = []
        path_operations: list[PathOperationIR] = []
        for dense_index in operation_index.group_indices(group.group_id):
            operation = operation_index.operation(dense_index)
            if not isinstance(operation, PathOperationIR):
                continue
            path_operations.append(operation)
            serialized = serialize_path(operation)
            if serialized is None:
                continue
            commands, valid_subpath_count = serialized
            command_blocks.append(commands)
            atom_op_indices.extend([dense_index] * valid_subpath_count)
        dense = _is_dense_two_dimensional_layer(
            tuple(path_operations),
            len(atom_op_indices),
        )
        result.append(SerializedGroup(
            group_id=group.group_id,
            commands="\n".join(command_blocks),
            atom_op_indices=tuple(atom_op_indices),
            force_non_linetype="dense_2d_layer" if dense else None,
        ))
    return tuple(result)


def validate_group_classification(
    source: SerializedGroup,
    analyzed: RecognizedGroup,
) -> None:
    """Reject atom loss, duplicate ownership, or wrong Group responses."""

    if analyzed.group_id != source.group_id:
        raise ValueError("Method1 returned a different group_id")
    if analyzed.atom_count != len(source.atom_op_indices):
        raise ValueError(
            f"Group {source.group_id} atom count mismatch: "
            f"input {len(source.atom_op_indices)}, result {analyzed.atom_count}"
        )
    classified_atom_count = (
        sum(line_type.atom_count for line_type in analyzed.line_types)
        + analyzed.non_linetype.atom_count
    )
    if classified_atom_count != analyzed.atom_count:
        raise ValueError(f"Group {source.group_id} classified atom count is incomplete")
    expected = sorted(set(source.atom_op_indices))
    actual = sorted({
        index
        for line_type in analyzed.line_types
        for index in line_type.op_indices
    } | set(analyzed.non_linetype.op_indices))
    if actual != expected:
        raise ValueError(f"Group {source.group_id} operation partition is incomplete")
