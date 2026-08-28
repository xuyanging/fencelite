"""机器规格探测 —— 让并发旋钮跟着机器走，而不是写死在代码里.

为什么需要它：这套管线要能在不同配置的服务器上直接跑（弱机器也不许炸）。
硬编码的并发数在 32 线程的机器上浪费、在 4 核机器上则会 OOM 或者把机器压死。

两个量、两种用途，别混：

* ``cpu_threads()`` —— 决定**纯算**的并发（线型聚类的页级并发、每页 worker）。
* ``total_ram_gb()`` —— 决定**吃内存**的并发。箭头边车是 Node 进程，heap 阶梯
  最高会升到 6144 MB（steps/arrows.py 的 _HEAP_LADDER），所以它的并发上限由
  内存而不是核数决定；按核数开会在小内存机器上直接 OOM。

**网络型阶段（文字 VLM / 判词 / 图例符号 / 视图分类）不在这里** —— 它们受
模型侧延迟与配额限制，弱 CPU 机器上一样能开 8 路，跟着 CPU 走只会白白变慢。

不引第三方依赖（psutil 不在 requirements 里，而这套东西必须在最小环境里可用）。
"""
from __future__ import annotations

import os


def cpu_threads() -> int:
    """本进程实际可用的逻辑处理器数。

    优先 ``sched_getaffinity``：容器 / cgroup 里 ``cpu_count()`` 报的是宿主的核数，
    照它开并发会严重超订。Windows 上没有这个调用，退回 ``cpu_count()``。
    """
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            return max(1, len(get_affinity(0)))
        except (OSError, TypeError, ValueError):
            pass
    return max(1, os.cpu_count() or 1)


def total_ram_gb() -> float:
    """物理内存总量（GB）。探测不到时返回 0.0，调用方必须能接受"未知"。

    返回 0 而不是猜一个值：猜大了会在小机器上 OOM，那比不做自适应更糟。
    调用方看到 0 应当退回"只按 CPU 推导"的保守分支。
    """
    # POSIX
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and size > 0:
            return pages * size / (1024 ** 3)
    except (AttributeError, ValueError, OSError):
        pass
    # Windows：GlobalMemoryStatusEx
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / (1024 ** 3)
    except Exception:                                           # noqa: BLE001
        pass
    return 0.0


def clamp(value, low, high) -> int:
    """把推导出来的并发数夹进 [low, high]。上下限都必须给：\n
    没有下限，弱机器会算出 0 个 worker；没有上限，强机器会超订。"""
    return int(max(low, min(high, value)))


def describe() -> str:
    ram = total_ram_gb()
    return (f"cpu_threads={cpu_threads()} "
            f"ram={'unknown' if ram <= 0 else f'{ram:.1f}GB'}")
