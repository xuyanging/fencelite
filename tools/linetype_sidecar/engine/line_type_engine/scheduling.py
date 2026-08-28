"""CPU-budgeted scheduling for independent single-page recognizers.

Method1 and Method2 consume the same immutable PageIR but do not depend on one
another. A fused page can overlap them without changing either algorithm.
Both methods may own process workers.  A fused request reserves roughly one
quarter of the logical processors (capped by Method2's proven useful limit) for
Method2 and gives the remainder to Method1.  This mirrors the measured heavy-
page work ratio while keeping the total runnable algorithm workers bounded.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.connection import Connection
import multiprocessing
import os
from typing import Callable, TypeVar


Method1Result = TypeVar("Method1Result")
Method2Result = TypeVar("Method2Result")
DEFAULT_METHOD2_PARALLEL_CAP = 8


def available_parallelism() -> int:
    """Return logical processors actually available to this process."""

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return max(1, len(get_affinity(0)))
        except (OSError, TypeError, ValueError):
            pass
    return max(1, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class SinglePageExecutionPlan:
    total_cpu_budget: int
    method1_worker_count: int
    method2_worker_count: int
    concurrent_methods: bool

    def __post_init__(self) -> None:
        if self.total_cpu_budget < 1:
            raise ValueError("total_cpu_budget must be positive")
        if min(self.method1_worker_count, self.method2_worker_count) < 0:
            raise ValueError("single-page worker counts must be nonnegative")
        if (
            self.concurrent_methods
            and self.method1_worker_count + self.method2_worker_count
            > self.total_cpu_budget
        ):
            raise ValueError("concurrent workers exceed total_cpu_budget")
        if self.concurrent_methods and min(
            self.method1_worker_count,
            self.method2_worker_count,
        ) < 1:
            raise ValueError("concurrent methods require both workers")


def plan_single_page_execution(
    *,
    needs_method1: bool,
    needs_method2: bool,
    cpu_budget: int | None = None,
    method2_parallel_cap: int = DEFAULT_METHOD2_PARALLEL_CAP,
    parallel_methods: bool = True,
) -> SinglePageExecutionPlan:
    """Allocate a bounded CPU budget without changing algorithm semantics."""

    if cpu_budget is not None and (
        isinstance(cpu_budget, bool)
        or not isinstance(cpu_budget, int)
        or cpu_budget < 1
    ):
        raise ValueError("cpu_budget must be a positive integer")
    if (
        isinstance(method2_parallel_cap, bool)
        or not isinstance(method2_parallel_cap, int)
        or method2_parallel_cap < 1
    ):
        raise ValueError("method2_parallel_cap must be a positive integer")
    available = available_parallelism()
    budget = min(cpu_budget or available, available)
    can_reserve = (
        parallel_methods and needs_method1 and needs_method2 and budget >= 2
    )
    if not needs_method2:
        method2_workers = 0
    elif can_reserve:
        method2_workers = min(
            method2_parallel_cap,
            max(1, budget // 4),
            budget - 1,
        )
    else:
        method2_workers = min(method2_parallel_cap, budget)
    method1_workers = (
        0
        if not needs_method1
        else budget - method2_workers
        if can_reserve
        else budget
    )
    return SinglePageExecutionPlan(
        total_cpu_budget=budget,
        method1_worker_count=method1_workers,
        method2_worker_count=method2_workers,
        concurrent_methods=can_reserve,
    )


class ParallelMethodError(RuntimeError):
    """A child recognizer failed without a safely transferable exception."""


def _process_method_entry(
    connection: Connection,
    task: Callable[..., Method2Result],
    arguments: tuple[object, ...],
) -> None:
    try:
        result = task(*arguments)
    except BaseException as error:
        try:
            connection.send(("error", error))
        except (BrokenPipeError, EOFError, OSError, TypeError):
            try:
                connection.send((
                    "untransferable_error",
                    f"{type(error).__module__}.{type(error).__qualname__}: {error}",
                ))
            except (BrokenPipeError, EOFError, OSError):
                pass
    else:
        try:
            connection.send(("ok", result))
        except (BrokenPipeError, EOFError, OSError):
            # The parent cancelled Method1 and deliberately closed the pipe.
            pass
    finally:
        connection.close()


def _stop_process(process: multiprocessing.Process) -> None:
    if process.is_alive():
        process.terminate()
    process.join()
    close = getattr(process, "close", None)
    if close is not None:
        close()


def run_independent_methods(
    method1_task: Callable[[], Method1Result],
    method2_task: Callable[..., Method2Result],
    *,
    method2_arguments: tuple[object, ...] = (),
    concurrently: bool,
) -> tuple[Method1Result, Method2Result]:
    """Run recognizers concurrently while retaining stable error priority.

    Method1 is awaited first because that is the failure callers observed in
    the former sequential composition. On failure, queued work is cancelled
    and running work is joined before returning, so no child pool is orphaned.
    The HTTP boundary can still cancel the complete process tree on disconnect.
    """

    if not concurrently:
        return method1_task(), method2_task(*method2_arguments)

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_process_method_entry,
        args=(sender, method2_task, method2_arguments),
        name="line-type-page-method2",
    )
    process.start()
    sender.close()
    try:
        method1 = method1_task()
    except BaseException:
        receiver.close()
        _stop_process(process)
        raise
    try:
        try:
            status, payload = receiver.recv()
        except EOFError as error:
            raise ParallelMethodError(
                f"Method2 worker exited without a result (exitcode={process.exitcode})"
            ) from error
    finally:
        receiver.close()
        process.join()
        exitcode = process.exitcode
        close = getattr(process, "close", None)
        if close is not None:
            close()
    if status == "ok":
        return method1, payload
    if status == "error" and isinstance(payload, BaseException):
        raise payload
    raise ParallelMethodError(f"Method2 worker failed: {payload}; exitcode={exitcode}")


__all__ = [
    "DEFAULT_METHOD2_PARALLEL_CAP",
    "SinglePageExecutionPlan",
    "ParallelMethodError",
    "available_parallelism",
    "plan_single_page_execution",
    "run_independent_methods",
]
