"""Read-only migration diagnostics for the native Python line-type engine."""

from .grouping_parity import (
    compare_grouping_snapshots,
    describe_python_grouping,
)

__all__ = ["compare_grouping_snapshots", "describe_python_grouping"]
