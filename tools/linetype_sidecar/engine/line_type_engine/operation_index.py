"""Unique result-operation indexing over a :class:`PageIR`.

``paint_order`` is authored display-list evidence and is deliberately allowed
to repeat: PyMuPDF may expose many text spans for one paint event.  Recognition
results therefore use the dense position in ``PageIR.operations`` as their
integer ``op_indices`` identity.  This table is the single conversion boundary
between stable operation ids, dense result ids, paint order, and Groups.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterator, Mapping

from .ir import GroupingIR, OperationIR, PageIR, PathOperationIR, TextOperationIR


@dataclass(frozen=True, slots=True)
class PageOperationIndex:
    """Validated, read-only indexes for one PageIR/GroupingIR pair."""

    operations: tuple[OperationIR, ...]
    index_by_operation_id: Mapping[str, int]
    group_id_by_index: tuple[str, ...]
    indices_by_group_id: Mapping[str, tuple[int, ...]]
    indices_by_paint_order: Mapping[int, tuple[int, ...]]

    @classmethod
    def build(cls, page: PageIR, grouping: GroupingIR) -> "PageOperationIndex":
        if grouping.page_fingerprint != page.fingerprint:
            raise ValueError("GroupingIR does not belong to the supplied PageIR")

        index_by_id = {
            operation.operation_id: index
            for index, operation in enumerate(page.operations)
        }
        if len(index_by_id) != len(page.operations):
            raise ValueError("PageIR operation ids are not unique")

        assignment_by_id = dict(grouping.assignments)
        if len(assignment_by_id) != len(grouping.assignments):
            raise ValueError("GroupingIR assigns an operation more than once")
        expected_ids = set(index_by_id)
        if set(assignment_by_id) != expected_ids:
            missing = len(expected_ids - set(assignment_by_id))
            extra = len(set(assignment_by_id) - expected_ids)
            raise ValueError(
                "GroupingIR is not a complete PageIR partition "
                f"(missing={missing}, extra={extra})"
            )

        group_id_by_index = tuple(
            assignment_by_id[operation.operation_id]
            for operation in page.operations
        )
        indices_by_group: dict[str, list[int]] = {
            group.group_id: [] for group in grouping.groups
        }
        for index, group_id in enumerate(group_id_by_index):
            if group_id not in indices_by_group:
                raise ValueError(f"GroupingIR references unknown group {group_id!r}")
            indices_by_group[group_id].append(index)

        flattened: list[int] = []
        for group in grouping.groups:
            indices = indices_by_group[group.group_id]
            if not indices:
                raise ValueError(f"GroupingIR group {group.group_id!r} is empty")
            expected = list(range(indices[0], indices[-1] + 1))
            if indices != expected:
                raise ValueError(
                    f"GroupingIR group {group.group_id!r} is not contiguous"
                )
            declared_ids = tuple(group.operation_ids)
            actual_ids = tuple(page.operations[index].operation_id for index in indices)
            if declared_ids != actual_ids:
                raise ValueError(
                    f"GroupingIR group {group.group_id!r} operation order differs from PageIR"
                )
            flattened.extend(indices)
        if flattened != list(range(len(page.operations))):
            raise ValueError("GroupingIR Groups are not an ordered complete partition")

        by_paint_order: dict[int, list[int]] = {}
        for index, operation in enumerate(page.operations):
            by_paint_order.setdefault(operation.paint_order, []).append(index)

        return cls(
            operations=page.operations,
            index_by_operation_id=MappingProxyType(index_by_id),
            group_id_by_index=group_id_by_index,
            indices_by_group_id=MappingProxyType({
                group_id: tuple(indices)
                for group_id, indices in indices_by_group.items()
            }),
            indices_by_paint_order=MappingProxyType({
                paint_order: tuple(indices)
                for paint_order, indices in by_paint_order.items()
            }),
        )

    def operation(self, op_index: int) -> OperationIR:
        if isinstance(op_index, bool) or not isinstance(op_index, int):
            raise TypeError("operation index must be an integer")
        if op_index < 0 or op_index >= len(self.operations):
            raise IndexError(f"operation index {op_index} is outside the page")
        return self.operations[op_index]

    def operation_index(self, operation_id: str) -> int:
        try:
            return self.index_by_operation_id[operation_id]
        except KeyError as error:
            raise KeyError(f"unknown PageIR operation id {operation_id!r}") from error

    def group_id(self, op_index: int) -> str:
        self.operation(op_index)
        return self.group_id_by_index[op_index]

    def group_indices(self, group_id: str) -> tuple[int, ...]:
        try:
            return self.indices_by_group_id[group_id]
        except KeyError as error:
            raise KeyError(f"unknown GroupIR id {group_id!r}") from error

    def group_span(self, group_id: str) -> tuple[int, int]:
        indices = self.group_indices(group_id)
        return (indices[0], indices[-1] + 1)

    def indices_for_paint_order(self, paint_order: int) -> tuple[int, ...]:
        return self.indices_by_paint_order.get(paint_order, ())

    def path_items(self) -> Iterator[tuple[int, PathOperationIR]]:
        for index, operation in enumerate(self.operations):
            if isinstance(operation, PathOperationIR):
                yield index, operation

    def text_items(self) -> Iterator[tuple[int, TextOperationIR]]:
        for index, operation in enumerate(self.operations):
            if isinstance(operation, TextOperationIR):
                yield index, operation
