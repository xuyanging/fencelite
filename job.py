"""上传作业编排 + 时间/费用统计 —— 全项目唯一的付费入口.

网页层只做「校验 + 建 slug + 起后台线程」，生产默认的六阶段链在这里按固定
依赖顺序跑完：

    text        0.00 → 0.55   矢量文字层(进程池) → 判词(线程池)
                              → 整页 VLM(线程池) → 本地融合
    symbols     0.55 → 0.80   每页一次 group+symbol 付费推理(线程池)
    views       0.80 → 0.92   只给「有 shape 样例 且 有合法 view 组」的页
                              做投影分类(线程池)
    placements  0.92 → 0.96   本地 shape 模板匹配 + plan 过滤(零费用)
    arrows      0.96 → 0.98   本地 Node 边车恢复引线与箭头末端
    linetypes   0.98 → 1.00   本地 Python 边车聚类线型并绑定末端

阶段之间严格串行（后一步吃前一步的产物），阶段内部按页并发。自定义检测目标
（target 非默认）只跑 text 一步（0→1.0）—— 后五步是 fence 专属语义。

为什么这些机制不能省：
  * ``_PROC_SLOTS``：按机器容量限制同时运行的 PDF；每个作业拥有独立的
    RECORDER 会话和模型上下文，同一 slug 仍由稳定文件锁保持单写者。
  * ``RECORDER.on_update``：每笔付费调用一完成就把累计 cost/wall 落盘，
    进程崩了也不会丢已经花掉的钱。
  * 增量 checkpoint：vec 每批落盘、判词每 chunk 落盘、VLM raw 每页落盘、
    symbols/view_types 每页写回 —— 崩溃/取消后重跑只补没做完的那部分，
    绝不重复付费。
  * 取消是协作式的：阶段边界 guard() 抛 Cancelled，should_cancel 透传进每个
    阶段让它停止排新页；已经在飞的模型调用会跑完（钱已经花了）。text 阶段被
    取消时不发布 results.json，但付费 raw 留在 vlm.json 里供下次复用。
"""
import os

# 并发旋钮必须在各阶段常量求值之前 setdefault：steps/text/judge.py 这类模块在
# import 期就把 os.environ 读成模块级常量，运行期再改环境变量无效。本文件的
# 阶段实现模块都是在阶段函数里才 import 的，所以这里的 setdefault 一定更早。
os.environ.setdefault("TEXT_WORKERS", "8")      # 单页图像 VLM：并发页数
os.environ.setdefault("JUDGE_WORKERS", "4")     # 判词分块并发（纯文本，网络受限）
os.environ.setdefault("SYMBOLS_WORKERS", "8")   # 图例 group+symbol：并发页数
os.environ.setdefault("VIEW_WORKERS", "6")      # 视图投影分类：并发页数
# 矢量文字抽取跨*进程*并发（MuPDF 非线程安全），按核数封顶
# ---- CPU 型旋钮：跟机器走，不写死 -----------------------------------------
# 这套要能在不同配置的服务器上直接跑。硬编码的并发数在 32 线程机器上浪费、在
# 4 核机器上会把机器压死或 OOM。两个量分开用（见 core/hw.py）：纯算的并发看
# cpu_threads，吃内存的并发看 total_ram_gb。
# 上面四个网络型旋钮**故意不跟 CPU 走** —— 它们受模型延迟与配额限制，弱 CPU
# 机器上一样能开 8 路，跟着 CPU 走只会白白变慢。
from core import hw as _hw

_CPU = _hw.cpu_threads()
_RAM = _hw.total_ram_gb()

os.environ.setdefault("VEC_WORKERS", str(_hw.clamp(_CPU, 1, 6)))
# 箭头边车是 Node 进程，heap 阶梯最高升到 6144 MB（steps/arrows.py 的
# _HEAP_LADDER），所以它的上限由**内存**定而不是核数 —— 按核数开会在小内存机器上
# 直接 OOM。原来默认是 1，也就是箭头阶段完全串行（实测 grand_island 37 页 274 s）。
os.environ.setdefault("ARROWS_WORKERS", str(_hw.clamp(
    min(_CPU // 6, int(_RAM // 6) if _RAM > 0 else 1), 1, 6)))

import re
import shutil
import threading
import time
import traceback
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import fitz

from core.concurrency import (SlotPool, SlotWaitCancelled,
                              stable_named_lock)
from core.config import (MODEL_NAME, PRICING, PROJECTS_DIR, compute_cost,
                         resolve_model, set_model_override,
                         submit_with_context)
from core.gemini import (RECORDER, is_timeout_error,
                         should_retry_model_error)
from core.pdfio import FITZ_LOCK
from steps import arrows, legend_linetypes, linetypes
from steps.debug import DebugSink
from steps.store import (DATA_DIR, JOBS_DIR, is_valid_slug, items_of, load_json,
                         pdf_path, pdf_revision, results_path, save_json,
                         sig_of, slug_dir)
from steps.text.target import TARGET_DEFAULT, is_default_target
from steps.versions import FUSED_VERSION

TEXT_WORKERS = int(os.environ["TEXT_WORKERS"])
SYMBOLS_WORKERS = int(os.environ["SYMBOLS_WORKERS"])
VIEW_WORKERS = int(os.environ["VIEW_WORKERS"])
# 箭头边车是独立 Node 进程，单页峰值 120-280 MB；2 vCPU / ~1 GB 可用内存
# 下并发只会互相抢内存并触发 swap，默认串行。
ARROWS_WORKERS = int(os.environ.get("ARROWS_WORKERS", "1"))
# 线型：同时跑几页。每页一个边车子进程，子进程内部还有引擎自己的并行度
# （LINETYPE_CPU_BUDGET），所以这里不是越大越好 —— 4 页 × budget 16 在
# 16 核 / 32 线程上已经是 2 倍超订，靠的是各页的单线程阶段互相错开。
def _linetype_page_workers():
    """线型的页级并发。跟机器走。

    为什么是 cpu/2.5：实测单页有效占用约 1.6 个核（引擎的
    group_page_sequentially 单线程、占单页 76% 的时间），留余量给页内 worker。
    上限 12 是**实测拐点**：本机 32 线程上 4→8 路吞吐 ×1.59、8→16 路只再 ×1.23，
    16 路时整机 CPU 已 79.5%（峰值 100%）—— 再往上是超订不是加速。
    下限 2 是让弱机器也有一点重叠。内存那一路按每页约 2 GB 估。
    """
    ram = _RAM
    by_ram = int(ram // 2) if ram > 0 else 2
    return _hw.clamp(min(round(_CPU / 2.5), by_ram), 2, 12)


LINETYPE_PAGE_WORKERS = int(os.environ.get("LINETYPE_PAGE_WORKERS", "")
                            or _linetype_page_workers())
# 普通页 10 分钟硬上限；密页保留 60 分钟。Bristol P24 约 49k paths，
# 但聚类复杂度明显高于普通页，30 分钟仍会稳定截断有效答案。阈值放到
# 40k 后它和更密的图纸都走长预算；轻页仍维持有界的 10 分钟。
LINETYPE_TIMEOUT = max(1, int(os.environ.get("LINETYPE_TIMEOUT", "600")))
LINETYPE_DENSE_TIMEOUT = max(
    LINETYPE_TIMEOUT,
    int(os.environ.get("LINETYPE_DENSE_TIMEOUT", "3600")))
LINETYPE_DENSE_PATHS = max(
    1, int(os.environ.get("LINETYPE_DENSE_PATHS", "40000")))
VEC_WORKERS = int(os.environ["VEC_WORKERS"])

VEC_SCHEMA = 3   # 3 = native typographic lines (v2's homegrown clustering
                 # chained whole elevation sheets into one mega box)
FLASH_MODEL = "gemini-3.5-flash"   # 整页视觉的第二模型（并集保险）
# 准确率优先：默认每页都用主模型 + Flash 读图。CAD 常把标题栏做成
# 可提取文字，却把真正的 fence 标注画成纯描边 path；“页上有任意文字”
# 绝不能再当成跳过视觉识别的依据。只有明确接受这类假阴性时才设 0。
SCAN_ALL_PAGES = os.environ.get("SCAN_ALL_PAGES", "1") not in (
    "0", "", "false", "no", "off")
# 没有文字层的页（扫描件）默认必须读图：矢量+判词那条免费通道在这类页上什么也
# 看不到，不读图就会「一页都没找到」却还报成功。这个开关只在
# SCAN_ALL_PAGES=0 的选择性模式下生效。
SCAN_NO_TEXT_PAGES = os.environ.get("SCAN_NO_TEXT_PAGES", "1") not in (
    "0", "", "false", "no", "off")

RETRIES = 2      # 共 3 次尝试；退避系数各阶段不同，见各处 time.sleep

_IO_LOCK = threading.Lock()      # symbols.json / view_types.json 读-改-写串行
_VLM_LOCK = threading.Lock()     # vlm.json / vlm_flash.json 读-改-写串行
# 线型回填脚本与 Web 作业是不同进程，_IO_LOCK 拦不住它们；同页由下方
# stable_named_lock 在进程内与跨进程统一串行。
class Cancelled(Exception):
    """Raised to unwind a run the user asked to cancel."""


class JobStartError(RuntimeError):
    """A background worker could not safely be claimed or started.

    For a newly-created unique upload, this is raised only before
    ``Thread.start`` succeeds, so its caller may clean that upload up. Existing
    project reruns use ``restart_job`` and translate a competing claim to 409.
    """


# ---------------------------------------------------------------- job state --
JOBS = {}                       # slug -> status dict
_JOBS_LOCK = threading.Lock()
# Snapshot + atomic replace must be one ordered operation.  Heartbeats,
# progress callbacks and RECORDER.on_update run on different threads; locking
# only the in-memory dict lets an older snapshot finish its replace after a
# newer one and resurrect ``done=False`` on disk after the process is already
# complete.  Keep this lock separate from _JOBS_LOCK so slow fsync never blocks
# read-only /api/job snapshots.
_JOB_PERSIST_LOCK = threading.Lock()
# Two PDFs may overlap on this 8-core / 15-GB host.  The expensive local
# sidecars still share their old single-job capacity, and vector extraction has
# its own CPU gate below; this fills network/CPU idle gaps without multiplying
# the worst-case memory footprint.
MAX_PARALLEL_JOBS = max(1, int(os.environ.get("MAX_PARALLEL_JOBS", "")
                               or _hw.clamp(
                                   min(_CPU // 4, int(_RAM // 7)), 1, 3)))
HEAVY_SIDECAR_SLOTS = max(1, int(os.environ.get("HEAVY_SIDECAR_SLOTS", "")
                                  or max(ARROWS_WORKERS,
                                         LINETYPE_PAGE_WORKERS)))
_SLOT_POOLS = {}
_SLOT_POOLS_LOCK = threading.Lock()


def _slot_pool(name, capacity):
    # JOBS_DIR is patched to a temporary root in tests; keying the registry by
    # its current value prevents a test run from touching production lock files.
    directory = JOBS_DIR / ".capacity-slots"
    key = (str(directory.absolute()), str(name), int(capacity))
    with _SLOT_POOLS_LOCK:
        return _SLOT_POOLS.setdefault(
            key, SlotPool(directory, name, capacity))

# Per-running-slug baseline and processing clock.  A single dict was safe only
# while jobs were serialized; a map keeps live cost/wall snapshots isolated.
_RUNNING = {}
_RUNNING_LOCK = threading.Lock()

# A reservation spans cache reset, queueing and the complete worker lifetime.
# It makes the route's "is it running? -> reset -> start" transaction atomic
# within the single gunicorn process, so two simultaneous reruns cannot wipe
# each other's paid checkpoints before one is rejected.
_STARTING = set()
_STARTING_LOCK = threading.Lock()

_CANCEL = {}                    # slug -> shared threading.Event
_CANCEL_USERS = {}              # direct/internal duplicate-run hardening
_CANCEL_LOCK = threading.Lock()

_ZERO_LLM = {"calls": 0, "model_seconds": 0.0, "peak_concurrency": 0,
             "input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
             "cost_usd": 0.0, "by_model": {}}


@contextmanager
def _interprocess_proc_lock(*, all_slots=False, cancelled=None):
    """Take one project slot, or drain all slots for restart recovery."""
    pool = _slot_pool("project", MAX_PARALLEL_JOBS)
    guard = (pool.all_slots() if all_slots
             else pool.slot(cancelled=cancelled))
    with guard:
        yield


def _job_run_lock_path(slug):
    if not is_valid_slug(slug):
        raise ValueError(f"invalid project slug: {slug!r}")
    return JOBS_DIR / ".job-run-locks" / f"{slug}.lock"


def _linetype_page_lock_path(slug, page):
    """Stable lock inode for one project sheet, outside deletable data dirs."""
    if not is_valid_slug(slug):
        raise ValueError(f"invalid project slug: {slug!r}")
    page = int(page)
    if page < 1:
        raise ValueError(f"page must be 1-based, got {page!r}")
    return JOBS_DIR / ".linetype-page-locks" / f"{slug}.{page}.lock"


@contextmanager
def _linetype_page_lock(slug, page, cancelled=None):
    """Serialize one sheet across web workers and external backfills.

    The lock lives under ``_jobs`` rather than ``data/<slug>``: deleting a
    project's result directory while a worker still owns an open lock file
    must not let another process recreate a different inode and enter the same
    sheet concurrently.  POSIX ``flock`` is advisory and crash-safe; platforms
    without it still retain the in-process mutex used by tests/development.
    """
    try:
        with stable_named_lock(_linetype_page_lock_path(slug, page),
                               cancelled=cancelled):
            yield
    except SlotWaitCancelled as exc:
        raise Cancelled() from exc


def _claim_job_start(slug, *, resume=False):
    """Reserve one slug before any status/cache mutation.

    Automatic recovery is idempotent and may quietly observe an existing
    owner. Explicit upload/rerun starts instead fail without touching that
    owner's status, cancellation event or cache.
    """
    return _claim_job_starts([slug], resume=resume)


def _claim_job_starts(slugs, *, resume=False):
    """Atomically reserve one or more project identities."""
    slugs = tuple(dict.fromkeys(slugs))
    with _STARTING_LOCK:
        busy = [slug for slug in slugs if slug in _STARTING]
        if busy:
            if resume:
                return False
            names = ", ".join(repr(slug) for slug in busy)
            raise JobStartError(
                f"project {names} is already queued or running")
        _STARTING.update(slugs)
    return True


def _release_job_start(slug):
    with _STARTING_LOCK:
        _STARTING.discard(slug)


def _release_job_starts(slugs):
    with _STARTING_LOCK:
        _STARTING.difference_update(slugs)


def _register_cancel_event(slug):
    """Share cancellation across any internal duplicate callers safely."""
    with _CANCEL_LOCK:
        event = _CANCEL.get(slug)
        if event is None:
            event = threading.Event()
            _CANCEL[slug] = event
        _CANCEL_USERS[slug] = _CANCEL_USERS.get(slug, 0) + 1
        return event


def _unregister_cancel_event(slug, event):
    with _CANCEL_LOCK:
        users = max(0, _CANCEL_USERS.get(slug, 1) - 1)
        if users:
            _CANCEL_USERS[slug] = users
        else:
            _CANCEL_USERS.pop(slug, None)
            if _CANCEL.get(slug) is event:
                _CANCEL.pop(slug, None)


def _merge_llm(a, b):
    """Sum two LLM summaries (peak_concurrency is a max, not a sum)."""
    a = a or {}
    b = b or {}
    out = {
        "calls": int(a.get("calls", 0)) + int(b.get("calls", 0)),
        "model_seconds": round(float(a.get("model_seconds", 0))
                               + float(b.get("model_seconds", 0)), 2),
        "peak_concurrency": max(int(a.get("peak_concurrency", 0)),
                                int(b.get("peak_concurrency", 0))),
        "input_tokens": int(a.get("input_tokens", 0)) + int(b.get("input_tokens", 0)),
        "output_tokens": int(a.get("output_tokens", 0)) + int(b.get("output_tokens", 0)),
        "thoughts_tokens": int(a.get("thoughts_tokens", 0)) + int(b.get("thoughts_tokens", 0)),
        "cost_usd": round(float(a.get("cost_usd", 0)) + float(b.get("cost_usd", 0)), 4),
        "by_model": {},
    }
    for src in (a.get("by_model") or {}, b.get("by_model") or {}):
        for m, v in src.items():
            d = out["by_model"].setdefault(
                m, {"calls": 0, "seconds": 0.0, "cost_usd": 0.0,
                    "input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0})
            d["calls"] += int(v.get("calls", 0))
            d["seconds"] = round(d["seconds"] + float(v.get("seconds", 0)), 2)
            d["cost_usd"] = round(d["cost_usd"] + float(v.get("cost_usd", 0)), 4)
            d["input_tokens"] += int(v.get("input_tokens", 0))
            d["output_tokens"] += int(v.get("output_tokens", 0))
            d["thoughts_tokens"] += int(v.get("thoughts_tokens", 0))
    return out


def _running_state(slug):
    with _RUNNING_LOCK:
        return dict(_RUNNING.get(slug) or {})


def _effective_llm(slug):
    """Cumulative cost/time for the running job = carried baseline + this
    session's RECORDER total."""
    state = _running_state(slug)
    return _merge_llm(state.get("base"), RECORDER.summary(slug))


def _flush_running(slug):
    """Persist the running job's cumulative cost/time after each completed
    model call (RECORDER.on_update hook).  Keeps the on-disk total exact to the
    last completed call, so a crash+resume never loses already-counted spend.
    No-op when no processing run is active."""
    state = _running_state(slug)
    if not state:
        return
    upd = {"llm": _effective_llm(slug)}
    processing_started = state.get("processing_started")
    if processing_started:
        upd["wall_seconds"] = round(
            state.get("base_wall", 0.0)
            + (time.time() - processing_started), 1)
    _set(slug, **upd)


# every counted Gemini call flushes the cumulative total to disk
RECORDER.on_update = _flush_running


def _job_file(slug):
    return JOBS_DIR / f"{slug}.json"


def _persist_job(slug):
    """Mirror a job's status to disk so a server restart can show it.
    Best-effort: persistence must never break processing."""
    try:
        # The snapshot is intentionally taken AFTER acquiring the persistence
        # lock.  Taking it first and serializing only save_json still permits
        # an old waiter to replace a newer file last.
        with _JOB_PERSIST_LOCK:
            with _JOBS_LOCK:
                job = dict(JOBS.get(slug) or {})
            if job:
                JOBS_DIR.mkdir(parents=True, exist_ok=True)
                save_json(_job_file(slug), job)
    except Exception:                                          # noqa: BLE001
        pass


def _set(slug, **kw):
    # ``updated_at`` is a real liveness signal for the browser.  A paid model
    # call or a dense local line-type page can legitimately take minutes; the
    # old UI looked frozen because the last visible percentage did not move.
    # Keep this server-authored timestamp out of callers so every state change
    # (including heartbeat-only changes) follows the same persistence path.
    kw["updated_at"] = time.time()
    with _JOBS_LOCK:
        job = JOBS.setdefault(slug, {"slug": slug})
        job.update(kw)
    _persist_job(slug)


def _warn(slug, messages):
    """Append stage warnings (per-page failures / skips) to the job.

    The reference implementation threw every batch's return code away, so a
    partially failed run still showed up as a clean success in the gallery.
    Collecting them here is the fix: whatever succeeded remains published, but
    ``_finish`` marks the job outcome as ``partial`` and the web layer keeps an
    actionable retry card instead of hiding it as a clean success."""
    if not messages:
        return
    with _JOBS_LOCK:
        job = JOBS.setdefault(slug, {"slug": slug})
        job["warnings"] = list(job.get("warnings") or []) + list(messages)
        job["updated_at"] = time.time()
    _persist_job(slug)


def get_job(slug):
    """Public status for one slug, merged with the LIVE cumulative cost/time
    while this slug's run is in flight (so the UI sees it grow in real time)."""
    with _JOBS_LOCK:
        job = dict(JOBS.get(slug) or {})
    state = _running_state(slug)
    if job and state and not job.get("done"):
        job["llm"] = _effective_llm(slug)
        if state.get("processing_started"):
            job["wall_seconds"] = round(
                state.get("base_wall", 0.0)
                + (time.time() - state["processing_started"]), 1)
    return job or None


def all_jobs():
    """Every known job (newest first), each with live totals merged in."""
    with _JOBS_LOCK:
        slugs = list(JOBS)
    jobs = [j for j in (get_job(s) for s in slugs) if j]
    jobs.sort(key=lambda j: j.get("started") or 0, reverse=True)
    return jobs


def job_running(slug):
    """True while a job for this slug is still processing (not done).  Used to
    suppress partial-result loading in the viewer until the run finishes."""
    with _JOBS_LOCK:
        j = JOBS.get(slug)
    return bool(j) and not j.get("done")


def request_cancel(slug):
    """Ask a running job to stop.  Cooperative: stages stop scheduling new
    pages and the run unwinds at the next stage boundary."""
    with _CANCEL_LOCK:
        ev = _CANCEL.get(slug)
    if ev is not None:
        ev.set()
    _set(slug, cancel_requested=True, detail="Cancelling…")
    return {"was_running": ev is not None, "job": get_job(slug)}


def _wall_now(slug):
    state = _running_state(slug)
    processing_started = state.get("processing_started")
    if not processing_started:
        return None
    return round(state.get("base_wall", 0.0)
                 + (time.time() - processing_started), 1)


def _finish(slug, ok, summary, error=None):
    total = _merge_llm(_running_state(slug).get("base"), summary)
    with _JOBS_LOCK:
        warnings = list((JOBS.get(slug) or {}).get("warnings") or [])
    # ``ok`` remains the backwards-compatible "pipeline did not crash" bit.
    # ``outcome`` tells the UI whether every requested source actually
    # completed.  This lets useful partial results remain viewable without
    # presenting unresolved sheets as a clean success.
    outcome = ("failed" if not ok else
               ("partial" if warnings else "success"))
    _set(slug, done=True, ok=ok, outcome=outcome, error=error,
         stage="done" if ok else "error", stage_unit=None,
         results_available=results_path(slug).exists(),
         repairing=None, llm=total, finished=time.time())
    wall = _wall_now(slug)
    if wall is not None:
        _set(slug, wall_seconds=wall)


@contextmanager
def _job_heartbeat(slug):
    """Persist liveness while one long page is inside a model/sidecar call.

    Progress is deliberately left untouched: a heartbeat proves the worker is
    alive but must never pretend a sheet has completed.  The browser can now
    distinguish a legitimate ten-minute dense page from a dead connection.
    """
    stop = threading.Event()

    def beat():
        while not stop.wait(10):
            with _JOBS_LOCK:
                status = JOBS.get(slug) or {}
                if status.get("done"):
                    return
            _set(slug, heartbeat_at=time.time())

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=1)


def _snapshot(slug):
    _set(slug, llm=_effective_llm(slug))


def _persist_llm(slug, summary):
    """Store the CUMULATIVE model time + cost inside results.json so the gallery
    keeps showing it after the in-memory job is gone (server restart), and so a
    later re-run can carry it forward as the baseline."""
    path = results_path(slug)
    res = load_json(path, None)
    if not isinstance(res, dict):
        return
    res["llm_summary"] = _merge_llm(
        _running_state(slug).get("base"), summary)
    wall = _wall_now(slug)
    if wall is not None:
        res["wall_seconds"] = wall
    try:
        save_json(path, res)
    except Exception:                                          # noqa: BLE001
        pass


def _carry_baseline(slug):
    """(llm, wall) already spent on this slug by an earlier run.

    Money is per project, not per attempt: a run that was interrupted (or
    cancelled) already paid for its VLM raw, and the next run reuses those
    caches.  Reading the published totals back as the new run's baseline keeps
    the gallery's cost/time monotonic instead of resetting to zero.

    results.json only gets ``llm_summary`` from ``_persist_llm`` at _finish, so
    a run killed before that (server restart, crash, cancel) leaves the money
    spent but unbooked.  The job card is the other half of the ledger — it is
    rewritten by ``_flush_running`` after every single paid call precisely so a
    dead process cannot lose what it already spent — so fall back to it and
    take whichever total is larger.  Without this the next re-run reports a
    project that cost $1.05 as costing nothing.
    """
    res = load_json(results_path(slug), None)
    llm = None
    wall = 0.0
    if isinstance(res, dict):
        published = res.get("llm_summary")
        llm = published if isinstance(published, dict) else None
        wall = float(res.get("wall_seconds") or 0.0)
    card = load_json(_job_file(slug), None)
    if isinstance(card, dict):
        spent = card.get("llm")
        if isinstance(spent, dict):
            # Calls/tokens/time are monotonic too. Cost alone is insufficient:
            # a completed zero-usage response or four-decimal rounding can
            # advance the durable card without changing cost_usd.
            def progress_key(summary):
                summary = summary or {}
                tokens = sum(int(summary.get(key) or 0) for key in (
                    "input_tokens", "output_tokens", "thoughts_tokens"))
                return (int(summary.get("calls") or 0), tokens,
                        float(summary.get("model_seconds") or 0.0),
                        float(summary.get("cost_usd") or 0.0))

            if progress_key(spent) > progress_key(llm):
                llm = spent
        # Pre-parallel cards measured wall time from upload, so a PDF which
        # waited half an hour contributed that whole queue delay on recovery.
        # New cards carry processing_started and measure only occupied-slot
        # time. Ignore an unfinished legacy card's incompatible wall number;
        # completed results remain historical data and are left untouched.
        if card.get("processing_started") or card.get("done"):
            wall = max(wall, float(card.get("wall_seconds") or 0.0))
    return llm, wall


# ------------------------------------------------------------ project setup --
def _slugify(filename):
    base = Path(filename or "").stem.lower()
    base = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
    if not base:
        base = "project"
    return base[:60]


def _claim_project_dir(filename):
    """Atomically reserve a unique project directory for one upload."""
    base = _slugify(filename)
    n = 1
    while True:
        slug = base if n == 1 else f"{base}_{n}"
        n += 1
        if (DATA_DIR / slug).exists():
            continue
        directory = PROJECTS_DIR / slug
        try:
            # exist_ok=False is the cross-thread/process slug lock.  The old
            # exists()+mkdir(exist_ok=True) sequence let simultaneous uploads
            # of the same filename overwrite each other's input.pdf.
            directory.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        if (DATA_DIR / slug).exists():
            # A legacy data-only project appeared between the check and mkdir.
            # This directory is ours and still empty, so releasing it is safe.
            directory.rmdir()
            continue
        return slug, directory


def create_project(pdf_bytes, filename):
    """Write projects/<slug>/input.pdf under a fresh unique slug; return slug.
    A re-upload of the same file name gets a ``_2``/``_3`` suffix so multiple
    runs (e.g. different targets) coexist in the gallery."""
    slug, directory = _claim_project_dir(filename)
    try:
        (directory / "input.pdf").write_bytes(pdf_bytes)
        return slug
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def create_project_stream(stream, filename):
    """Stream an upload to disk without holding the entire PDF in RAM."""
    slug, directory = _claim_project_dir(filename)
    try:
        with (directory / "input.pdf").open("xb") as output:
            shutil.copyfileobj(stream, output, length=1024 * 1024)
        return slug
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


# A comparison run lives in its own slug, "<base>__<model-id>", so the two
# providers' caches and results never share a directory. _slugify() collapses
# runs of non-alphanumerics to a single "_", so it can never emit "__" — the
# separator is unambiguous.
VARIANT_SEP = "__"


def variant_base(slug):
    """Base slug this one was forked from, or None if it isn't a variant."""
    if not isinstance(slug, str) or VARIANT_SEP not in slug:
        return None
    return slug.split(VARIANT_SEP, 1)[0] or None


def variant_model(slug):
    """Model id a variant slug pins, or None."""
    if not isinstance(slug, str) or VARIANT_SEP not in slug:
        return None
    tag = slug.split(VARIANT_SEP, 1)[1]
    return tag if tag in PRICING else None


def run_model(slug, res=None):
    """Which model produced this project's results.

    A variant slug names it directly; otherwise fall back to what the recorder
    actually billed, then to the process default for pre-existing runs.
    """
    pinned = variant_model(slug)
    if pinned:
        return pinned
    by_model = ((res or {}).get("llm_summary") or {}).get("by_model") or {}
    if by_model:
        # Attribute to the model that did the most paid work.
        return max(by_model.items(),
                   key=lambda kv: (kv[1] or {}).get("calls", 0))[0]
    return MODEL_NAME


def _variant_identity(base_slug, model):
    """Validate a requested comparison and return its stable slug."""
    if not is_valid_slug(base_slug):
        raise ValueError("invalid slug")
    if variant_base(base_slug):
        raise ValueError("cannot fork a variant of a variant")
    if model not in PRICING:
        raise ValueError(f"unknown model: {model}")
    slug = f"{base_slug}{VARIANT_SEP}{model}"
    if not is_valid_slug(slug):
        raise ValueError(f"variant slug is not a valid path segment: {slug}")
    return slug


def _create_variant_unclaimed(base_slug, model, slug):
    """Create/reuse the sibling while base+variant reservations are held."""
    src = PROJECTS_DIR / base_slug / "input.pdf"
    if not src.exists():
        raise FileNotFoundError("base project has no input.pdf")
    directory = PROJECTS_DIR / slug
    directory.mkdir(parents=True, exist_ok=True)
    dst = directory / "input.pdf"
    if not dst.exists():
        try:
            os.link(src, dst)      # same inode: no extra disk, same revision
        except OSError:
            shutil.copy2(src, dst)
    return slug


def create_variant(base_slug, model):
    """Fork a comparison run: same PDF, different model, separate cache dir.

    The whole point is that the original run stays untouched on disk — a
    variant is a sibling project (projects/<base>__<model>/ +
    data/<base>__<model>/), never a re-run in place. Both providers' results
    therefore coexist and the UI can switch between them.

    Idempotent: forking an existing variant returns the same slug, so the
    caller can re-run it without wiping and recreating.
    """
    slug = _variant_identity(base_slug, model)
    _claim_job_starts([base_slug, slug])
    try:
        if job_running(base_slug) or job_running(slug):
            raise JobStartError("base project or comparison is in progress")
        return _create_variant_unclaimed(base_slug, model, slug)
    finally:
        _release_job_starts([base_slug, slug])


def start_variant(base_slug, model, target=None):
    """Atomically create/reset/start a comparison without a delete race."""
    slug = _variant_identity(base_slug, model)
    _claim_job_starts([base_slug, slug])
    worker_started = False
    try:
        if job_running(base_slug) or job_running(slug):
            raise JobStartError("base project or comparison is in progress")
        _create_variant_unclaimed(base_slug, model, slug)
        cleared = reset_project_cache(slug)
        status = _start_job_claimed(slug, target=target, model=model)
        worker_started = True
        return slug, status, cleared
    finally:
        _release_job_start(base_slug)
        if not worker_started:
            # _start_job_claimed may already have released this on failure.
            _release_job_start(slug)


def variants_of(slug):
    """Comparison-run slugs forked from this one, as found on disk.

    The UI shows one row per PDF, so a variant is not separately deletable
    there — without this, deleting the parent would leave its variants as
    invisible orphans still occupying disk and still listed by the API.
    """
    if not is_valid_slug(slug) or variant_base(slug):
        return []
    prefix = f"{slug}{VARIANT_SEP}"
    found = set()
    for root in (PROJECTS_DIR, DATA_DIR):
        if not root.exists():
            continue
        for d in root.iterdir():
            if d.is_dir() and d.name.startswith(prefix) and is_valid_slug(d.name):
                found.add(d.name)
    return sorted(found)


def _delete_project_unclaimed(slug):
    """Delete exactly one slug while its caller owns the reservation."""
    removed = False
    for root in (PROJECTS_DIR / slug, DATA_DIR / slug):
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
            removed = True
    # Serialize deletion with any heartbeat/progress writer. Otherwise an
    # already-started _persist_job can recreate the status file immediately.
    with _JOB_PERSIST_LOCK:
        try:
            status_file = _job_file(slug)
            if status_file.exists():
                status_file.unlink()
                removed = True
        except OSError:
            pass
        with _JOBS_LOCK:
            JOBS.pop(slug, None)
    return removed


def delete_project(slug, cascade=True):
    """Remove a stopped project and, by default, all comparison siblings.

    ``cascade`` also removes this project's comparison runs, which is what the
    single-row gallery needs; pass False to delete exactly one slug. A queued
    or running owner is rejected before any file disappears.
    """
    if not is_valid_slug(slug):
        raise ValueError("invalid slug")
    claimed = [slug]
    _claim_job_start(slug)
    try:
        # Base reservation prevents a concurrent create_variant from appearing
        # between this inventory and the cascade.
        variants = variants_of(slug) if cascade else []
        if variants:
            _claim_job_starts(variants)
            claimed.extend(variants)
        busy = [name for name in claimed if job_running(name)]
        if busy:
            raise JobStartError(
                f"project {', '.join(busy)} is still in progress")
        removed = False
        for name in variants:
            removed = _delete_project_unclaimed(name) or removed
        return _delete_project_unclaimed(slug) or removed
    finally:
        _release_job_starts(claimed)


def reset_project_cache(slug):
    """Delete every cached artifact for a slug, keeping only the source PDF.

    This is the "重新跑" guarantee: after this call nothing on disk can steer
    the next run's output — no vector layer, no judge verdicts, no paid VLM
    raws, no symbol/view/placement results, no rendered base images.  The next
    job therefore re-derives (and re-pays for) everything from the PDF alone.
    Reusing a cache is normally the whole point; this exists for the moments
    when you need to *prove* the current code produced what you are looking at.
    """
    if not is_valid_slug(slug):
        raise ValueError("invalid slug")
    if job_running(slug):
        raise RuntimeError("this project is in progress — cancel it before resetting")
    root = DATA_DIR / slug
    removed = []
    if root.is_dir():
        for child in sorted(root.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                else:
                    child.unlink()
                removed.append(child.name)
            except OSError:
                pass
    return removed


def stored_target(slug):
    """这个项目上次是用哪个检测目标跑的（重新跑时预填，找不到就用默认）。"""
    res = load_json(DATA_DIR / slug / "results.json", None) or {}
    return res.get("target") or TARGET_DEFAULT


def page_count_of(slug):
    try:
        with FITZ_LOCK:
            with fitz.open(pdf_path(slug)) as doc:
                return doc.page_count
    except Exception:                                          # noqa: BLE001
        return 0


# ------------------------------------------------------- stage 1: text step --
def _vec_scan_project(slug, pdf, on_progress=None, should_cancel=None):
    """All pages' text lines, disk-cached (keyed on pdf mtime + schema).

    Pages are extracted across ``VEC_WORKERS`` processes (MuPDF is not
    thread-safe, so real parallelism needs processes), each opening the PDF
    once for its slice.  Checkpoints incrementally ("partial": true) so a
    killed run resumes where it stopped instead of rescanning from page 1."""
    from steps.text import vector_scan_pages

    cache_path = slug_dir(slug) / "vec.json"
    mtime = pdf.stat().st_mtime
    cached = load_json(cache_path, None)
    if cached and cached.get("pdf_mtime") == mtime \
            and cached.get("schema") == VEC_SCHEMA:
        # "partial" 这个标记不足以采信：取消后的 _final() 从来不写它，
        # 于是半份 vec.json 会永远看起来是完整的。真正的判据是页数够不够 ——
        # 这条同时治得了已经写坏的存量缓存。
        if not cached.get("partial") \
                and len(cached.get("pages") or {}) == cached.get("page_count"):
            return cached
    else:
        cached = None
    with FITZ_LOCK:
        with fitz.open(pdf) as doc:
            page_count = doc.page_count
    pages = (cached or {}).get("pages", {})
    if pages:
        print(f"  [vec] {slug} resuming at {len(pages)}/{page_count} cached",
              flush=True)
    missing = [p - 1 for p in range(1, page_count + 1) if str(p) not in pages]

    def _final():
        data = {"schema": VEC_SCHEMA, "pdf_mtime": mtime,
                "page_count": page_count, "pages": pages}
        # 取消是 break 出扫描后落到这里的。缺页就直说，别让短了一截的页集
        # 冒充跑完的扫描 —— 缺的那些页下一轮会被当扫描件重新付费读图。
        if len(pages) != page_count:
            data["partial"] = True
        save_json(cache_path, data)
        return data

    if not missing:
        return _final()

    t0 = time.perf_counter()
    done = [len(pages)]

    def on_chunk(batch):
        for idx, rec in batch.items():
            pages[str(idx + 1)] = rec
        done[0] += len(batch)
        # incremental checkpoint for crash/cancel resume
        save_json(cache_path, {"schema": VEC_SCHEMA, "pdf_mtime": mtime,
                               "page_count": page_count, "pages": pages,
                               "partial": True})
        if on_progress:
            on_progress(done[0], page_count)
        print(f"  [vec] {slug} {done[0]}/{page_count} "
              f"({time.perf_counter() - t0:.0f}s, x{VEC_WORKERS})", flush=True)

    vector_scan_pages(pdf, missing, workers=VEC_WORKERS,
                      on_chunk=on_chunk, should_cancel=should_cancel)
    return _final()


def _judge_project(slug, vpages, judge_prompt=None, use_kw_floor=True):
    """Text-judge the project's unique strings (cached, incremental).
    Returns (flagged_set, judge_error_or_None).  ``judge_prompt``/``use_kw_floor``
    are threaded from the detection target so the vector-text judge matches the
    exact same target as the image VLM."""
    from steps.text import judge_candidates, judge_strings, prepare_judge_cache

    cache_path = slug_dir(slug) / "textjudge.json"
    stored = load_json(cache_path, {})
    cache = prepare_judge_cache(stored, resolve_model(None))
    if cache != stored:
        save_json(cache_path, cache)
    verdicts = cache.get("verdicts", {})
    cand = judge_candidates([v.get("lines", []) for v in vpages.values()],
                            use_kw_floor=use_kw_floor)
    todo = sorted(set(cand) - set(verdicts))
    print(f"  [judge] unique strings={len(cand)}  cached={len(cand) - len(todo)}"
          f"  to judge={len(todo)}", flush=True)
    err = None
    if todo:
        def checkpoint(chunk_verdicts):
            verdicts.update(chunk_verdicts)
            cache["verdicts"] = verdicts
            cache["model"] = resolve_model(None)
            save_json(cache_path, cache)
        try:
            _, usage = judge_strings(
                todo,
                on_progress=lambda done, total: print(
                    f"  [judge] {slug} {done}/{total}", flush=True),
                on_chunk=checkpoint, judge_prompt=judge_prompt)
            cost = compute_cost(resolve_model(None), usage)
            print(f"  [judge] done  in={usage['input_tokens']} "
                  f"out={usage['output_tokens']} "
                  f"cost=${(cost or {}).get('total_usd', 0):.3f}", flush=True)
        except Exception as e:                              # noqa: BLE001
            # degrade to floor + already-cached verdicts, never lose the run
            err = f"{type(e).__name__}: {e}"
            print(f"  [judge] FAILED (floor-only fallback): {err}", flush=True)
    flagged = {s for s, v in verdicts.items() if v}
    n_flag = sum(1 for s in cand if s in flagged)
    print(f"  [judge] fence-related strings: {n_flag}", flush=True)
    return flagged, err


def _vlm_attempt_prompt(base_prompt, attempt):
    """Return the semantic prompt, with cache-busting repair metadata on retry.

    Gemini is deterministic at temperature zero and may also cache identical
    request bytes.  Re-sending a malformed response verbatim three times is
    therefore not a retry in practice (final_plans P17 returned ``[]\n[]`` on
    every attempt).  The suffix does not change the detection target; it only
    asks for the same answer in exactly one valid JSON document.  Durable VLM
    identity is still computed from ``base_prompt`` below, never this nonce.
    """
    if attempt <= 0:
        return base_prompt
    number = attempt + 1
    return (
        base_prompt
        + f"\n\nRETRY JSON VALIDATION ATTEMPT {number} "
        + f"(nonce=text-scan-retry-{number}-{time.time_ns()}). "
        + "The previous response failed strict validation. Return the same "
        + "requested detection result as exactly ONE complete JSON array. "
        + "Do not emit two top-level values, markdown, commentary, or a "
        + "truncated prefix."
    )


def _run_vlm(slug, pdf, page, store, store_path, model=None, role=None,
             prompt=None, max_attempts=None, timeout_retries=1,
             attempt_offset=0):
    """One page's paid image scan → a stamped record in ``store`` (on disk).

    Retries then stores an explicit error record: a failed page must stay
    visible as rework, never look like a cache hit.  ``role`` omitted → the
    primary role make_vlm_record defaults to."""
    from steps.text import make_vlm_record, scan_page, vlm_identity

    role_kw = {} if role is None else {"role": role}
    identity = vlm_identity(pdf, model, prompt)
    items = None
    last_err = None
    attempts = max(1, int(max_attempts or (RETRIES + 1)))
    for attempt in range(attempts):
        try:
            # 540s: dense casino/spec sheets kept blowing the 300s default
            items, elapsed, usage = scan_page(
                pdf, page - 1, model=identity["model"], timeout_ms=540_000,
                prompt=_vlm_attempt_prompt(prompt, attempt_offset + attempt))
            break
        except Exception as e:                              # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if should_retry_model_error(
                    e, attempt, attempts, timeout_retries=timeout_retries):
                time.sleep(20 * (attempt + 1))
            else:
                break
    with _VLM_LOCK:
        if items is None:
            store[str(page)] = make_vlm_record(
                identity=identity, error=last_err, **role_kw)
        else:
            store[str(page)] = make_vlm_record(
                identity=identity, items=items, elapsed=elapsed, usage=usage,
                **role_kw)
        save_json(store_path, store)
    return page, (len(items) if items is not None else None), last_err


def _stage_text(slug, target, on_progress=None, should_cancel=None):
    """矢量文字层 ∪ 判词 ∪ 整页 VLM → 本地融合 → 发布 results.json.

    ``target`` 是那唯一一处用户可编辑的检测目标（steps/text/target.py）。它被
    包进两套固定脚手架，分别生成图像 VLM 提示词与判词提示词，保证两条通道找的
    是同一件事。目标被折进 VLM 缓存 identity，换目标绝不会复用别的目标的应答。
    fence 关键词地板只对默认目标生效，自定义目标完全靠判词。

    返回 warning 列表（每页失败原因）。被取消时直接返回、不发布 results.json：
    已付费的 raw 仍留在 vlm.json / vlm_flash.json 里供下次复用。
    """
    from steps.text import (SECONDARY_UNION_ROLE, build_judge_prompt,
                            build_vlm_prompt, fuse_page,
                            is_current_primary_record,
                            is_current_secondary_record, select_instances,
                            union_vlm, vlm_identity, vlm_needed)

    warnings = []
    target = target or TARGET_DEFAULT
    is_default = is_default_target(target)
    vlm_prompt = build_vlm_prompt(target)
    judge_prompt = build_judge_prompt(target)
    pdf = pdf_path(slug)
    if not pdf.exists():
        return [f"{slug}: input.pdf does not exist"]

    # coarse text-step progress on a 0..100 scale so the bar creeps forward
    # during the (page-less) vector scan + judge call, not just the VLM loop
    def _p(pct):
        if on_progress:
            on_progress(int(pct), 100)

    print(f"\n=== {slug} === (mode={'fence' if is_default else 'custom'})",
          flush=True)
    _sub = {}                     # sub-step wall times for the text stage
    _t = time.perf_counter()
    _p(2)
    # vector extraction creeps the bar 2→8% (parallel across processes)
    try:
        with _slot_pool("vector", 1).slot(cancelled=should_cancel):
            vec = _vec_scan_project(
                slug, pdf,
                on_progress=lambda d, t: _p(2 + 6 * d / max(t, 1)),
                should_cancel=should_cancel)
    except SlotWaitCancelled as exc:
        raise Cancelled() from exc
    if should_cancel and should_cancel():
        return warnings
    vpages = vec["pages"]
    _p(8)
    _sub["vec"] = time.perf_counter() - _t
    _t = time.perf_counter()
    flagged, judge_err = _judge_project(slug, vpages, judge_prompt=judge_prompt,
                                        use_kw_floor=is_default)
    if judge_err:
        warnings.append(f"verdict pass failed as a whole; fell back to the keyword floor + plus cached verdicts: {judge_err}")
    _p(22)
    _sub["judge"] = time.perf_counter() - _t
    _t = time.perf_counter()

    vlm_path = slug_dir(slug) / "vlm.json"
    vlm = load_json(vlm_path, {})
    primary_identity = vlm_identity(pdf, None, vlm_prompt)
    flash_identity = vlm_identity(pdf, FLASH_MODEL, vlm_prompt)

    inst_by_page = {
        p: select_instances(vpages.get(str(p), {}).get("lines", []), flagged,
                            use_kw_floor=is_default)
        for p in range(1, vec["page_count"] + 1)
    }
    def _has_text(page):
        # 关掉 SCAN_NO_TEXT_PAGES 时把每页都当成「有文字层」，扫描页就退回
        # 「只扫矢量命中页」的省钱口径（代价见常量处的说明）。
        if not SCAN_NO_TEXT_PAGES:
            return True
        return bool(vpages.get(str(page), {}).get("has_text"))

    need = [p for p in range(1, vec["page_count"] + 1)
            if vlm_needed(p, inst_by_page[p], vlm, primary_identity,
                          has_text=_has_text(p), scan_all=SCAN_ALL_PAGES)]

    # 第二模型是独立的视觉保险，不以主模型成功为前提。否则主模型
    # 某页超时时，明明 Flash 可以读到，整页仍会被静默丢掉。
    def _secondary_enabled(page):
        if SCAN_ALL_PAGES:
            return True
        return (SCAN_NO_TEXT_PAGES
                and not vpages.get(str(page), {}).get("has_text"))

    flash_path = slug_dir(slug) / "vlm_flash.json"
    flash = load_json(flash_path, {})
    secondary_pages = [p for p in range(1, vec["page_count"] + 1)
                       if _secondary_enabled(p)]
    fneed = [p for p in secondary_pages
             if not is_current_secondary_record(flash.get(str(p)),
                                                flash_identity)]

    n_hit_pages = sum(1 for v in inst_by_page.values() if v)
    n_new = sum(1 for p in need if str(p) not in vlm)
    n_image_only = sum(1 for p in range(1, vec["page_count"] + 1)
                       if not vpages.get(str(p), {}).get("has_text"))
    print(f"  pages={vec['page_count']}  fence-text pages={n_hit_pages}  "
          f"image-only pages={n_image_only}  "
          f"vision={'all-pages' if SCAN_ALL_PAGES else 'selective'}  "
          f"VLM top-up needed={len(need)}+{len(fneed)} "
          f"(primary new pages={n_new})", flush=True)
    if (n_image_only and not SCAN_ALL_PAGES
            and not SCAN_NO_TEXT_PAGES):
        # 静默出空结果是最坏的失败方式，至少让它出现在作业卡片上。
        warnings.append(
            f"{n_image_only} sheets have no text layer and SCAN_NO_TEXT_PAGES=0 - "
            "these sheets were never read as images, so their results are certainly empty")

    # Pro + Flash 的首轮共用 22%..90%。一个 Pro 超时不能先原地
    # 再等 540s 才启动 Flash：先让两个独立模型各扫一轮，再对真正
    # timeout 的那一侧补最多一次。补跑占 90%..95%。
    paid_total = len(need) + len(fneed)

    def _paid_progress(done):
        if paid_total:
            _p(22 + 68 * done / paid_total)

    if need and not (should_cancel and should_cancel()):
        with ThreadPoolExecutor(max_workers=TEXT_WORKERS) as ex:
            futs = [submit_with_context(
                        ex, _run_vlm, slug, pdf, p, vlm, vlm_path,
                        prompt=vlm_prompt, timeout_retries=0,
                        attempt_offset=(1 if
                            (vlm.get(str(p)) or {}).get("error") else 0))
                    for p in need]
            for i, f in enumerate(as_completed(futs), 1):
                page, n, err = f.result()
                msg = f"ERROR {err}" if err and n is None else f"{n} items"
                print(f"  [vlm {i}/{len(need)}] P{page}: {msg}", flush=True)
                _paid_progress(i)
                if should_cancel and should_cancel():
                    for fu in futs:
                        fu.cancel()
                    break
    _sub["vlm"] = time.perf_counter() - _t
    _t = time.perf_counter()

    # 第二模型看同一页，结果取并集。准确率模式扫全页；选择性模式
    # 仅对无文字层扫描页做这层保险。
    if fneed and not (should_cancel and should_cancel()):
        print(f"  [flash-union] scan pages to double-check: {len(fneed)}",
              flush=True)
        with ThreadPoolExecutor(max_workers=TEXT_WORKERS) as ex:
            futs = [submit_with_context(
                        ex, _run_vlm, slug, pdf, p, flash, flash_path,
                        FLASH_MODEL, SECONDARY_UNION_ROLE,
                        prompt=vlm_prompt, timeout_retries=0,
                        attempt_offset=(1 if
                            (flash.get(str(p)) or {}).get("error") else 0))
                    for p in fneed]
            for i, f in enumerate(as_completed(futs), 1):
                page, n, err = f.result()
                msg = f"ERROR {err}" if err and n is None else f"{n} items"
                print(f"  [flash {i}/{len(fneed)}] P{page}: {msg}", flush=True)
                _paid_progress(len(need) + i)
                if should_cancel and should_cancel():
                    for fu in futs:
                        fu.cancel()
                    break
    # Deferred accuracy repair: both models get their first complete retry
    # policy before we top up ONLY the sources which are still invalid.  The
    # previous implementation selected timeouts alone and resent the exact
    # same base request bytes; malformed/truncated provider replies therefore
    # survived as a red warning.  Every residual source now gets one final
    # cache-busting request while durable identity stays on ``vlm_prompt``.
    retry_jobs = []
    for page in need:
        if not is_current_primary_record(vlm.get(str(page)),
                                         primary_identity):
            retry_jobs.append((page, vlm, vlm_path, None, None, "primary"))
    for page in fneed:
        if not is_current_secondary_record(flash.get(str(page)),
                                           flash_identity):
            retry_jobs.append((page, flash, flash_path, FLASH_MODEL,
                               SECONDARY_UNION_ROLE, "flash"))
    if retry_jobs and not (should_cancel and should_cancel()):
        print(f"  [vision-repair] incomplete image scans: {len(retry_jobs)}",
              flush=True)
        _set(slug, repairing={"stage": "text", "total": len(retry_jobs)},
             detail=(f"Automatically repairing {len(retry_jobs)} failed "
                     "image scan(s); completed sheets are being reused…"))
        with ThreadPoolExecutor(max_workers=TEXT_WORKERS) as ex:
            futs = [submit_with_context(
                ex, _run_vlm, slug, pdf, page, target_store, target_path,
                retry_model, retry_role, prompt=vlm_prompt, max_attempts=1,
                timeout_retries=0, attempt_offset=RETRIES + 1)
                for (page, target_store, target_path, retry_model, retry_role,
                     _label) in retry_jobs]
            labels = {future: retry_jobs[index][5]
                      for index, future in enumerate(futs)}
            for index, future in enumerate(as_completed(futs), 1):
                page, count, error = future.result()
                label = labels[future]
                message = (f"ERROR {error}" if error and count is None
                           else f"{count} items")
                print(f"  [vision-repair {index}/{len(futs)}] "
                      f"{label} P{page}: {message}", flush=True)
                _p(90 + 5 * index / len(futs))
                if should_cancel and should_cancel():
                    for pending in futs:
                        pending.cancel()
                    break
        _set(slug, repairing=None,
             detail="Automatic image-scan repair finished; validating results…")
    _p(95)

    # Report only final residuals.  A first-pass error repaired above never
    # reaches the user, and each unresolved source appears exactly once.
    for page in need:
        record = vlm.get(str(page)) or {}
        if not is_current_primary_record(record, primary_identity):
            warnings.append(
                f"P{page} primary image scan incomplete after automatic repair: "
                f"{record.get('error') or 'stale VLM cache identity'}")
    for page in fneed:
        record = flash.get(str(page)) or {}
        if not is_current_secondary_record(record, flash_identity):
            warnings.append(
                f"P{page} second image scan incomplete after automatic repair: "
                f"{record.get('error') or 'stale VLM cache identity'}")
    _sub["flash"] = time.perf_counter() - _t
    _t = time.perf_counter()

    # ---- merge ----
    # Cancelled during the (paid) VLM phase → skip the local merge over every
    # page and unwind now, so the job flips to cancelled promptly instead of
    # grinding vector work for a huge sheet set.  No results are published.
    if should_cancel and should_cancel():
        return warnings
    pages_out = {}
    tot = {"vlm": 0, "added": 0, "covered": 0}
    for p in range(1, vec["page_count"] + 1):
        v = vpages.get(str(p), {})
        inst = inst_by_page[p]
        stored = vlm.get(str(p))
        rec = stored if is_current_primary_record(stored, primary_identity) \
            else None
        flash_stored = flash.get(str(p)) if _secondary_enabled(p) else None
        flash_rec = (flash_stored
                     if is_current_secondary_record(flash_stored,
                                                    flash_identity)
                     else None)
        if rec is None and flash_rec is None and not inst:
            continue                   # neither model nor vector has evidence
        vitems = (rec or {}).get("items", [])
        flash_error = None
        if flash_rec is not None:
            vitems = union_vlm(vitems, flash_rec.get("items"))
        elif isinstance(flash_stored, dict):
            flash_error = (flash_stored.get("error")
                           or "stale VLM cache identity")
        dbg = DebugSink()
        # 单页组装（选实例 → 剥符号码 → 融合 → 调试视图）只有 steps.text.page
        # 这一份实现，将来的单页重扫共用它，绝不会出现第二条会漂移的合并循环。
        fused_rec = fuse_page(pdf, p - 1, v, flagged, vitems,
                              use_kw_floor=is_default, dbg=dbg)
        if not (fused_rec["vlm_items"] or fused_rec["vec_added"]
                or fused_rec["vec_covered"] or rec is not None
                or flash_rec is not None):
            continue
        fused_rec["vlm_error"] = (
            None if rec is not None
            else (stored or {}).get("error") or "stale VLM cache identity")
        if flash_error:
            fused_rec["vlm_flash_error"] = flash_error
        pages_out[str(p)] = fused_rec
        tot["vlm"] += len(fused_rec["vlm_items"])
        tot["added"] += len(fused_rec["vec_added"])
        tot["covered"] += len(fused_rec["vec_covered"])

    result = {
        "slug": slug,
        "fused_v": FUSED_VERSION,
        "pdf_revision": pdf_revision(pdf),
        "page_count": vec["page_count"],
        "no_text_layer": not any(v.get("has_text") for v in vpages.values()),
        "judge_error": judge_err,
        "mode": "fence" if is_default else "custom",
        "target": target,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "pages": pages_out,
    }
    save_json(results_path(slug), result)
    _sub["merge"] = time.perf_counter() - _t
    print(f"  merged: {len(pages_out)} pages | vlm={tot['vlm']} "
          f"vec_added={tot['added']} vec_covered={tot['covered']}", flush=True)
    print(f"  [timing:text] {[(k, round(v, 1)) for k, v in _sub.items()]}",
          flush=True)

    return warnings


# ---------------------------------------------------- stage 2: legend symbols --
def _symbol_jobs(slug):
    """Strictly enumerate pages whose symbol *detection* is not current."""
    from steps.symbols import has_current_symbols

    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    if not res or not pdf.exists():
        return []
    revision = pdf_revision(pdf)
    cache = load_json(slug_dir(slug) / "symbols.json", {})
    jobs = []
    for pstr, rec in sorted(res.get("pages", {}).items(),
                            key=lambda pair: int(pair[0])):
        items = items_of(rec)
        if not items:
            continue
        sig = sig_of(items, revision)
        if has_current_symbols(cache.get(pstr), sig):
            continue
        jobs.append((int(pstr), items, sig))
    return jobs


def _symbol_one(slug, page, items, sig):
    """Run one detection job and publish it, returning a result tuple.

    两步：
      ② 整页一次 group+symbol 付费推理（compute_page_symbols）；
      ②b 图例块裁剪补扫（legend_sweep）—— 只对「还有 fence 文字没配到样例」的
         legend/schedule/note_cluster 块，把那一块裁出来按高 DPI 重问一次。
         全页图上几十像素高的样例，在裁剪图上是它的 3~8 倍，召回与框精度是
         另一个量级。没有待补的块就一次调用都不发。

    Model exceptions are retried and then returned as an explicit per-page
    failure — one bad page never takes the whole stage down."""
    from steps.legend_sweep import sweep_needed, sweep_page
    from steps.symbols import compute_page_symbols, merge_sweep

    pdf = pdf_path(slug)
    cache_path = slug_dir(slug) / "symbols.json"
    with _IO_LOCK:
        ent = load_json(cache_path, {}).get(str(page))
    entry = None
    err = None
    fresh = False
    for attempt in range(RETRIES + 1):
        try:
            entry, fresh = compute_page_symbols(pdf, page - 1, items, ent, sig)
            break
        except Exception as exc:                            # noqa: BLE001
            err = f"{type(exc).__name__}: {exc}"
            if should_retry_model_error(exc, attempt, RETRIES + 1):
                time.sleep(15 * (attempt + 1))
            else:
                break
    if entry is None:
        return page, None, err, False, {}, 0, []

    result = entry["result"]
    usage = dict(result.get("usage") or {})

    # ②b 补扫。**每一页都要调一次 merge_sweep**（哪怕这页无事可做、sweep 是空
    # 字典）—— 那一下才把 sweep_v 盖上去，否则 has_current_symbols 永远判它
    # 不当期，下次作业又从头跑一遍。
    sweep = {}
    sweep_err = None
    try:
        groups = result.get("groups") or []
        symbols = result.get("symbols") or []
        if any(not b.get("skipped")
               for b in sweep_needed(items, groups, symbols)):
            sweep = sweep_page(pdf, page - 1, items, groups, symbols)
    except Exception as exc:                                # noqa: BLE001
        # sweep_page 自己号称不抛，这里只是不让补扫的意外把已经付过钱的
        # 步骤② 结果一起废掉。
        sweep_err = f"{type(exc).__name__}: {exc}"
    merge_sweep(entry, items, sweep)
    for key in usage:
        usage[key] = usage.get(key, 0) or 0
    for key, value in (sweep.get("usage") or {}).items():
        usage[key] = (usage.get(key) or 0) + (value or 0)
    errors = list(sweep.get("errors") or [])
    if sweep_err:
        errors.append({"group_index": None, "error": sweep_err})
    if errors:
        # 补扫失败的页不盖当期戳：下次作业会重试这一块（步骤② 的 raw 仍复用，
        # 不重付整页推理），并在阶段末尾的复核里显式报出来。
        entry.pop("sweep_v", None)

    with _IO_LOCK:
        cache = load_json(cache_path, {})
        cache[str(page)] = entry
        save_json(cache_path, cache)
    n_sweep = sum(1 for s in result.get("symbols") or []
                  if isinstance(s, dict) and s.get("source") == "sweep")
    return page, len(result["symbols"]), None, bool(fresh), usage, n_sweep, errors


def _stage_symbols(slug, on_progress=None, should_cancel=None):
    """每页一次 group+symbol 付费推理（线程池），逐页写回 symbols.json."""
    jobs = _symbol_jobs(slug)
    print(f"symbol jobs: {len(jobs)}  workers={SYMBOLS_WORKERS}", flush=True)
    if on_progress:
        on_progress(0, len(jobs))
    warnings = []
    usage_tot = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
    fresh_calls = 0
    sweep_calls = 0          # 补扫里失败的块数（成功的块数见每页日志）
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, SYMBOLS_WORKERS)) as ex:
        futures = {submit_with_context(ex, _symbol_one, slug, *job): job[0]
                   for job in jobs}
        for future in as_completed(futures):
            submitted_page = futures[future]
            done += 1
            if on_progress:
                on_progress(done, len(jobs))
            if should_cancel and should_cancel():
                for f in futures:
                    f.cancel()
                break
            try:
                page, count, err, fresh, usage, n_sweep, sweep_errs = \
                    future.result()
            except Exception as exc:                        # noqa: BLE001
                page, count, fresh, usage = submitted_page, None, False, {}
                n_sweep, sweep_errs = 0, []
                err = f"{type(exc).__name__}: {exc}"
            if err is not None or count is None:
                warnings.append(f"P{page} legend symbol detection failed: {err or 'unknown failure'}")
                message = f"ERROR {err or 'unknown failure'}"
            else:
                fresh_calls += int(fresh)
                for key in usage_tot:
                    usage_tot[key] += usage.get(key, 0) or 0
                sweep_calls += len(sweep_errs)
                message = (f"{count} symbols"
                           f"{'' if fresh else ' (VLM reused)'}"
                           f"{f' +{n_sweep} top-up scan' if n_sweep else ''}")
            for row in sweep_errs:
                warnings.append(
                    f"P{page} legend block top-up scan failed"
                    f"（group {row.get('group_index')}): {row.get('error')}")
            print(f"[{done}/{len(jobs)}] {slug} P{page}: {message}", flush=True)

    cost = compute_cost(resolve_model(None), usage_tot)
    print(f"  [symbols] fresh VLM calls={fresh_calls}  "
          f"cost=${(cost or {}).get('total_usd', 0):.2f}", flush=True)
    # Publication is complete only when a new disk read confirms that every
    # non-empty page satisfies the shared detection predicate.
    remaining = _symbol_jobs(slug)
    if remaining:
        warnings.append(f"legend symbols still have {len(remaining)} stale sheets: "
                        f"{[p for p, _i, _s in remaining][:12]}")
    return warnings


# ------------------------------------------------- stage 3: view projections --
def _view_jobs(slug):
    """只给「有 shape 样例符号 且 有合法 kind=view 组」的页排活.

    这条排活条件比参考实现更严：没有 shape 样例就没有要匹配的放置，那页的
    plan/elevation 分类对最终交付毫无影响 —— 付这笔钱纯属浪费。
    """
    from steps.symbols import has_current_symbols
    from steps.views import groups_need_classification, has_current_view_types

    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    if not res or not pdf.exists():
        return [], []
    revision = pdf_revision(pdf)
    symbols = load_json(slug_dir(slug) / "symbols.json", {})
    views = load_json(slug_dir(slug) / "view_types.json", {})
    jobs, warnings = [], []
    for pstr, rec in sorted(res.get("pages", {}).items(),
                            key=lambda pair: int(pair[0])):
        items = items_of(rec)
        if not items:
            continue
        entry = symbols.get(pstr)
        if not has_current_symbols(entry, sig_of(items, revision)):
            warnings.append(f"P{pstr} skipped view classification: legend symbol detection is stale")
            continue
        result = entry.get("result") or {}
        if not any(s.get("category") == "shape"
                   for s in result.get("symbols") or []) and not arrows.ENABLED:
            # no shape sample → nothing to place → don't pay.  但箭头步同样只在
            # plan 视图里找引线，开启后这页仍然需要分类，否则整页 fail-closed。
            continue
        groups = result.get("groups") or []
        if not groups_need_classification(groups):
            continue
        if has_current_view_types(views.get(pstr), groups, revision):
            continue
        jobs.append((int(pstr), pdf, groups, revision))
    return jobs, warnings


def _view_one(slug, page, pdf, groups, revision):
    from steps.views import compute_view_types

    cache_path = slug_dir(slug) / "view_types.json"
    with _IO_LOCK:
        cached = load_json(cache_path, {}).get(str(page))
    entry = None
    error = None
    for attempt in range(RETRIES + 1):
        try:
            entry, _fresh = compute_view_types(
                pdf, page - 1, groups, revision, cached=cached)
            break
        except Exception as exc:                            # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if should_retry_model_error(exc, attempt, RETRIES + 1):
                time.sleep(10 * (attempt + 1))
            else:
                break
    if entry is None:
        return page, None, error
    with _IO_LOCK:
        cache = load_json(cache_path, {})
        cache[str(page)] = entry
        save_json(cache_path, cache)
    summary = ", ".join(f"g{row['group_index']}={row['view_type']}"
                        for row in entry.get("views", [])) or "no views"
    return page, summary, None


def _stage_views(slug, on_progress=None, should_cancel=None):
    """kind=view 的组框细分为 plan/elevation/section/detail/other（独立付费缓存）."""
    jobs, warnings = _view_jobs(slug)
    print(f"view-type jobs: {len(jobs)}  workers={VIEW_WORKERS}", flush=True)
    if on_progress:
        on_progress(0, len(jobs))
    with ThreadPoolExecutor(max_workers=max(1, VIEW_WORKERS)) as pool:
        futures = {submit_with_context(pool, _view_one, slug, *job): job[0]
                   for job in jobs}
        for number, future in enumerate(as_completed(futures), 1):
            submitted_page = futures[future]
            if on_progress:
                on_progress(number, len(jobs))
            if should_cancel and should_cancel():
                for f in futures:
                    f.cancel()
                break
            try:
                page, summary, error = future.result()
            except Exception as exc:                        # noqa: BLE001
                page, summary = submitted_page, None
                error = f"{type(exc).__name__}: {exc}"
            if error:
                warnings.append(f"P{page} view classification failed: {error}")
                print(f"[{number}/{len(jobs)}] {slug} P{page}: ERROR {error}",
                      flush=True)
            else:
                print(f"[{number}/{len(jobs)}] {slug} P{page}: {summary}",
                      flush=True)
    remaining, _ = _view_jobs(slug)
    if remaining:
        warnings.append(f"view classification still has {len(remaining)} stale sheets: "
                        f"{[p for p, _pdf, _g, _r in remaining][:12]}")
    return warnings


# ------------------------------------------- stage 4: local shape placements --
def _stage_placements(slug, on_progress=None, should_cancel=None):
    """本地 shape 模板匹配 + plan 视图过滤 —— 纯几何、零模型调用.

    单线程：core.symbolmatch 读 PDF 矢量层且带页级 LRU 缓存，多线程只会互相抢
    MuPDF 的锁。写回也只在末尾落一次盘（同一个 symbols.json 大字典）。
    """
    from steps.placements import has_current_placements, match_placements
    from steps.snap_boxes import snap_symbol_boxes, text_trim_boxes
    from steps.symbols import has_current_symbols, inherit_row_code_symbols
    from steps.views import (groups_need_classification,
                             has_current_view_types, merge_view_types)

    warnings = []
    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    cache_path = slug_dir(slug) / "symbols.json"
    cache = load_json(cache_path, {})
    if not res or not pdf.exists() or not cache:
        if on_progress:
            on_progress(0, 0)
        return warnings
    revision = pdf_revision(pdf)
    views = load_json(slug_dir(slug) / "view_types.json", {})
    vec_cache = load_json(slug_dir(slug) / "vec.json", {})
    vec_pages = vec_cache.get("pages") or {}
    vec_current = bool(
        isinstance(vec_pages, dict)
        and vec_cache.get("schema") == VEC_SCHEMA
        and vec_cache.get("pdf_mtime") == pdf.stat().st_mtime
        and not vec_cache.get("partial")
        and vec_cache.get("page_count") == res.get("page_count")
        and len(vec_pages) == vec_cache.get("page_count"))
    pages = sorted(cache, key=lambda p: int(p))
    total = len(pages)
    if on_progress:
        on_progress(0, total)
    dirty = False
    n_pages = n_plc = n_dropped = 0
    for number, pstr in enumerate(pages, 1):
        if should_cancel and should_cancel():
            break
        entry = cache.get(pstr)
        rec = (res.get("pages") or {}).get(pstr)
        if not isinstance(entry, dict) or not rec:
            if on_progress:
                on_progress(number, total)
            continue
        if not has_current_symbols(entry, sig_of(items_of(rec), revision)):
            if on_progress:
                on_progress(number, total)
            continue
        result = entry.get("result") or {}
        groups = result.get("groups") or []
        page_items = items_of(rec)
        # plan 过滤要求带 classifier 出处的 view_type：缺分类就不猜 plan
        # （fail-closed，与参考实现同一条政策）。
        # 只有 line 样例（或压根没有样例）的页不需要 plan 取景框：line 不做匹配，
        # 没有 shape 就没有要过滤的放置。这种页**必须照样跑完并盖上 plc_v** ——
        # 否则 has_current_placements 永远为假，webapp 的发布闸会把这一页**整个
        # 符号层**都扣住，用户看到的是"图例里的线一个都没框出来"。
        # （真实案例：taylor_3_12 P3 只有一条 line 样例，却因为该页有 view 组、
        #  而 views 阶段按"有没有 shape"排活从不分类它，plc_v 一直是 None。）
        has_shape = any(isinstance(s, dict) and s.get("category") == "shape"
                        for s in (result.get("symbols") or []))
        if has_shape and groups_need_classification(groups):
            ventry = views.get(pstr)
            if not has_current_view_types(ventry, groups, revision):
                warnings.append(f"P{pstr} skipped placement matching: view classification is stale")
                if on_progress:
                    on_progress(number, total)
                continue
            typed_groups = merge_view_types(groups, ventry)
        else:
            typed_groups = groups
        if has_current_placements(result):
            if on_progress:
                on_progress(number, total)
            continue
        dbg = DebugSink()
        # An empty symbol list is a valid completed result: there is no local
        # vector work to do, but plc_v must still be recorded — without that
        # marker every read treats the page as perpetually pending.
        # match_placements already turns every per-symbol matcher failure into
        # that symbol's placement_error; a page-level blowup must still not
        # take the other pages' (already paid for) publication down with it.
        # 先用矢量层把样例框校准，再拿它当模板去匹配：模型在整页图上给的框会漂
        # （实测 P4 九个样例整列偏左 ~10 单位、两个纵向偏 6 单位），拿漂了的框
        # 当模板，匹配自然也差。校准是纯本地几何、零成本。
        # 它是**锦上添花**的一步：拿不到矢量层（扫描页 / PDF 读不出来）时，
        # 放置匹配照样用模型的框跑下去 —— 所以单独 try，失败只记在这一页的
        # result.snap_error 上，不升级成作业级 warning（那是"需要人处理"的语义）。
        symbols_now = result.get("symbols") or []
        may_have_row_code = (
            any(isinstance(s, dict) and s.get("category") == "shape"
                and re.fullmatch(r"\d{1,3}\.0", str(s.get("value") or "").strip())
                for s in symbols_now)
            and any(isinstance(item, dict)
                    and (item.get("vec_backed") is True
                         or item.get("source") == "vector")
                    for item in page_items))
        vector_page = vec_pages.get(pstr) if isinstance(vec_pages, dict) else None
        vector_lines = ((vector_page or {}).get("lines")
                        if isinstance(vector_page, dict) else None)
        if may_have_row_code and (not vec_current
                                  or not isinstance(vector_lines, list)):
            error = ("current native vector text is unavailable; refusing to "
                     "stamp row-code placement cache as complete")
            result["row_code_error"] = error
            warnings.append(f"P{pstr} skipped placement matching: {error}")
            dirty = True
            if on_progress:
                on_progress(number, total)
            continue
        snap_summary = {}
        snap_failed = False
        try:
            snap_summary = dict(snap_symbol_boxes(pdf, int(pstr) - 1,
                                                 symbols_now))
            # 只在父级 N.0 已经被矢量层证实是闭合 shape 后，才从同一
            # schedule 行左侧的原生 PDF 文字派生 N.k。新样例会继承父级
            # 外框尺寸，但保留精确 glyph 框作为文字身份；下面的生产
            # matcher 会和其他 shape 一样全页扫描并做 plan 过滤。
            inherited = inherit_row_code_symbols(
                entry, page_items,
                vector_lines if isinstance(vector_lines, list) else [])
            if inherited:
                snap_summary["snap_inherited"] = inherited
            symbols_now = result.get("symbols") or []
            result.pop("row_code_error", None)
            result.pop("snap_error", None)
        except Exception as exc:                                # noqa: BLE001
            result["snap_error"] = f"{type(exc).__name__}: {exc}"
            snap_failed = True
            print(f"  [snap] P{pstr} skipped sample-box calibration: {result['snap_error']}",
                  flush=True)
        if snap_failed and may_have_row_code:
            error = ("row-code inheritance could not be verified: "
                     f"{result['snap_error']}")
            result["row_code_error"] = error
            warnings.append(f"P{pstr} skipped placement matching: {error}")
            dirty = True
            if on_progress:
                on_progress(number, total)
            continue
        if not snap_failed:
            try:
                trims = text_trim_boxes(pdf, int(pstr) - 1,
                                        page_items, symbols_now)
                result.pop("trim_error", None)
                if trims:
                    # 文字框的裁剪结果单独存表，不去改 results.json 里的 item ——
                    # 那会动 store.sig_of，让整页的步骤② 重新付费。
                    result["text_trim"] = {str(k): v for k, v in trims.items()}
                else:
                    result.pop("text_trim", None)
            except Exception as exc:                            # noqa: BLE001
                # 文字框裁剪是显示优化，不是 row-code 继承的证据；
                # 失败时不得挡住已验证的 symbol 进入生产 matcher。
                result["trim_error"] = f"{type(exc).__name__}: {exc}"
        try:
            summary = dict(snap_summary)
            summary.update(match_placements(pdf, int(pstr) - 1, symbols_now,
                                            typed_groups, dbg=dbg))
        except Exception as exc:                            # noqa: BLE001
            warnings.append(f"P{pstr} placement matching failed: "
                            f"{type(exc).__name__}: {exc}")
            if on_progress:
                on_progress(number, total)
            continue
        result.update(summary)
        entry["debug"] = dbg.data
        dirty = True
        n_pages += 1
        n_plc += int(summary.get("placed") or 0)
        n_dropped += int(summary.get("dropped_outside_plan") or 0)
        if on_progress:
            on_progress(number, total)
    if dirty:
        with _IO_LOCK:
            save_json(cache_path, cache)
    print(f"  [placements] pages={n_pages} shape placements={n_plc} "
          f"dropped outside plan={n_dropped}", flush=True)
    return warnings


# ----------------------------------------------------- stage 5 (seam): arrows --
def _arrow_jobs(slug):
    """只给「文字锚当期 且 plan 取景框可用」的页排活.

    取景框来自步骤3 的分类。没有分类就不排活也不报错地跳过 —— 与
    steps.views.plan_boxes 的下游约定一致：缺分类时一个结果都不给，
    绝不把整页当 plan。
    """
    from steps.views import merge_view_types, plan_boxes

    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    if not res or not pdf.exists():
        return [], []
    revision = pdf_revision(pdf)
    symbols = load_json(slug_dir(slug) / "symbols.json", {})
    views = load_json(slug_dir(slug) / "view_types.json", {})
    cache = load_json(slug_dir(slug) / "arrows.json", {})
    jobs, warnings = [], []
    for pstr, rec in sorted(res.get("pages", {}).items(),
                            key=lambda pair: int(pair[0])):
        items = items_of(rec)
        if not items:
            continue
        result = (symbols.get(pstr) or {}).get("result") or {}
        # 第二类锚：shape 样例矢量匹配出来的放置。它们和文字框一样落在 plan 上，
        # 同样要走后续步骤，只是键空间分开（"s<symbol>:<placement>"）以免和
        # 文字锚的 union index 相撞。
        extra = _placement_anchors(result)
        sig = arrows.arrows_signature(items, revision, extra)
        if arrows.has_current_arrows(cache.get(pstr), sig):
            continue
        groups = result.get("groups") or []
        regions = plan_boxes(merge_view_types(groups, views.get(pstr)))
        # 取景开启时没有 plan 框就 fail-closed；关闭时任何有锚的页都要跑，
        # 这样被取景挡掉的锚也能暴露出问题来。
        if arrows.PLAN_GATE and not regions:
            warnings.append(f"P{pstr} skipped arrows: no plan view box")
            continue
        if not items and not extra:
            continue
        jobs.append((int(pstr), items, sig, regions, extra))
    return jobs, warnings


def _placement_anchors(result):
    """symbols.json 的 result → [(key, box_2d), ...] 放置锚。"""
    anchors = []
    for si, symbol in enumerate(result.get("symbols") or []):
        for pi, box in enumerate(symbol.get("placements") or []):
            if isinstance(box, (list, tuple)) and len(box) == 4:
                anchors.append((f"s{si}:{pi}", list(box)))
    return anchors


def _raise_if_cancelled(should_cancel):
    if should_cancel and should_cancel():
        raise Cancelled()


def _retry_pause(seconds, should_cancel=None):
    """Retry backoff which remains responsive to a queued-job cancellation."""
    if should_cancel is None:
        time.sleep(seconds)
        return
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        _raise_if_cancelled(should_cancel)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(0.2, remaining))


def _arrow_one(slug, page, items, sig, regions, extra_anchors,
               should_cancel=None):
    """一页的箭头检测，结果原子写回 arrows.json。"""
    _raise_if_cancelled(should_cancel)
    cache_path = slug_dir(slug) / "arrows.json"
    geometry = arrows.page_geometry_status(pdf_path(slug), page - 1)
    if geometry.get("state") == "image-only":
        # The user explicitly wants raster-only sheets identified, not guessed
        # at with pixel tracing.  Keep an empty, current cache entry so reruns do
        # not repeatedly launch the vector sidecar for the same scan.
        with _IO_LOCK:
            cache = load_json(cache_path, {})
            cache[str(page)] = {
                "sig": sig, "v": arrows.ARROWS_VERSION, "items": {},
                "page_kind": "image-only",
                "geometry": geometry,
            }
            save_json(cache_path, cache)
        return page, 0, None
    found = None
    error = None
    for attempt in range(RETRIES + 1):
        try:
            with _slot_pool(
                    "heavy-sidecar", HEAVY_SIDECAR_SLOTS).slot(
                        cancelled=should_cancel):
                _raise_if_cancelled(should_cancel)
                found, anchor_diagnostics = arrows.find_page_arrows(
                    pdf_path(slug), page - 1, items, plan_regions=regions,
                    extra_anchors=extra_anchors, return_diagnostics=True)
            break
        except SlotWaitCancelled as exc:
            raise Cancelled() from exc
        except Cancelled:
            raise
        except Exception as exc:                            # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            if attempt < RETRIES and not is_timeout_error(exc):
                _retry_pause(5 * (attempt + 1), should_cancel)
            else:
                break
    if found is None:
        # 失败必须落盘。没有条目和「算过但没找到」在界面上是两回事，而且不写
        # 的话下一轮排活看不出这页试过又失败了。故意不写 items —— 缺 items 的
        # 条目不满足 has_current_arrows，下次会自动重试。
        with _IO_LOCK:
            cache = load_json(cache_path, {})
            cache[str(page)] = {"sig": sig, "v": arrows.ARROWS_VERSION,
                                "error": error, "geometry": geometry}
            save_json(cache_path, cache)
        return page, None, error
    with _IO_LOCK:
        cache = load_json(cache_path, {})
        # 键必须是 str，与 items_of 的 union index 对应关系写在 steps/arrows.py。
        cache[str(page)] = {"sig": sig, "v": arrows.ARROWS_VERSION,
                            "items": {str(k): v for k, v in found.items()},
                            "anchors": {str(k): v for k, v
                                        in anchor_diagnostics.items()},
                            # 线型阶段用这个已经免费算出的 path 数
                            # 选 10 / 30 分钟的自适应截止，避免再打开 PDF。
                            "geometry": geometry}
        save_json(cache_path, cache)
    return page, len(found), None


def _stage_arrows(slug, on_progress=None, should_cancel=None):
    """按文字 callout 找箭头 / 引线（矢量几何，零模型调用）。

    只在 plan 视图里找；取景与 fail-closed 语义见 steps/arrows.py。
    """
    jobs, warnings = _arrow_jobs(slug)
    print(f"arrow jobs: {len(jobs)}  workers={ARROWS_WORKERS}", flush=True)
    if on_progress:
        on_progress(0, len(jobs))
    total = 0
    with ThreadPoolExecutor(max_workers=max(1, ARROWS_WORKERS)) as pool:
        futures = {
            pool.submit(_arrow_one, slug, *page_job, should_cancel): page_job[0]
            for page_job in jobs
        }
        for number, future in enumerate(as_completed(futures), 1):
            submitted_page = futures[future]
            if on_progress:
                on_progress(number, len(jobs))
            if should_cancel and should_cancel():
                for f in futures:
                    f.cancel()
                break
            try:
                page, count, error = future.result()
            except Exception as exc:                        # noqa: BLE001
                page, count = submitted_page, None
                error = f"{type(exc).__name__}: {exc}"
            if error:
                warnings.append(f"P{page} arrow detection failed: {error}")
                print(f"[{number}/{len(jobs)}] {slug} P{page}: ERROR {error}",
                      flush=True)
            else:
                total += count or 0
                print(f"[{number}/{len(jobs)}] {slug} P{page}: {count} arrows",
                      flush=True)
    print(f"arrows total: {total}", flush=True)
    return warnings


# ------------------------------------------- stage 6: line types (sidecar) --
def _linetype_jobs(slug):
    """给「箭头结果当期 且 这页有末端」的页排活.

    **不看 plan** —— plan 只是显示闸（见 steps/linetypes 的 docstring）。
    也不看 arrows 的取景开关：末端在哪算在哪，可见性留给 webapp 决定。
    """
    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    if not res or not pdf.exists():
        return [], []
    revision = pdf_revision(pdf)
    symbols = load_json(slug_dir(slug) / "symbols.json", {})
    arrow_cache = load_json(slug_dir(slug) / "arrows.json", {})
    jobs, warnings = [], []
    for pstr, rec in sorted(res.get("pages", {}).items(),
                            key=lambda pair: int(pair[0])):
        items = items_of(rec)
        if not items:
            continue
        result = (symbols.get(pstr) or {}).get("result") or {}
        extra = _placement_anchors(result)
        arrows_sig = arrows.arrows_signature(items, revision, extra)
        arrow_entry = arrow_cache.get(pstr)
        if not arrows.has_current_arrows(arrow_entry, arrows_sig):
            # 箭头层不当期 —— 这页的末端还没定下来，算线型没有意义。不报错：
            # 上一阶段自己会把失败页记进 warnings。
            continue
        anchors = linetypes.anchors_of(arrow_entry)
        if not anchors:
            continue
        sig = linetypes.linetypes_signature(arrows_sig)
        if linetypes.has_current_linetypes(
                linetypes.load_page(slug, int(pstr)), sig):
            continue
        jobs.append((int(pstr), items, arrow_entry, sig))
    return jobs, warnings


def _legend_linetype_jobs(slug):
    """Schedule every current, explicitly boxed legend line sample.

    This is deliberately independent from ``arrows.json`` and placements.  A
    legend swatch is itself the supervised anchor, so requiring an arrow
    terminal (or a shape placement) would silently restore the exact missing
    pipeline edge this channel exists to fill.
    """
    from steps.symbols import has_current_symbols

    res = load_json(results_path(slug), None)
    pdf = pdf_path(slug)
    if not res or not pdf.exists():
        return [], []
    revision = pdf_revision(pdf)
    symbols = load_json(slug_dir(slug) / "symbols.json", {})
    jobs, warnings = [], []
    for pstr, rec in sorted(res.get("pages", {}).items(),
                            key=lambda pair: int(pair[0])):
        items = items_of(rec)
        if not items:
            continue
        symbol_entry = symbols.get(pstr)
        if not has_current_symbols(symbol_entry, sig_of(items, revision)):
            # Symbol detection owns the warning for a stale/failed page.  Do
            # not duplicate it here; merely refuse to consume stale boxes.
            continue
        result = (symbol_entry or {}).get("result") or {}
        samples = legend_linetypes.samples_of(result)
        if not samples:
            continue
        sig = legend_linetypes.signature(revision, samples)
        page = int(pstr)
        if legend_linetypes.has_current(
                legend_linetypes.load_page(slug, page), sig):
            continue
        jobs.append((page, samples, sig))
    return jobs, warnings


def _linetype_timeout_for(slug, page, arrow_entry):
    """Choose a bounded deadline without penalising genuinely huge sheets."""
    geometry = (arrow_entry or {}).get("geometry")
    if not isinstance(geometry, dict):
        # Old arrow caches predate the geometry field.  One cheap MuPDF pass
        # upgrades the decision in memory; if even that fails, favour recall
        # and use the dense deadline rather than timing out a valid huge page.
        try:
            geometry = arrows.page_geometry_status(pdf_path(slug), page - 1)
        except Exception:                                      # noqa: BLE001
            return LINETYPE_DENSE_TIMEOUT
    try:
        paths = int(geometry.get("vector_paths") or 0)
    except (TypeError, ValueError):
        return LINETYPE_DENSE_TIMEOUT
    return (LINETYPE_DENSE_TIMEOUT
            if paths >= LINETYPE_DENSE_PATHS else LINETYPE_TIMEOUT)


def _legend_linetype_timeout_for(slug, page):
    """Choose the legend sidecar deadline from the PDF, never arrow cache.

    Legend work must remain schedulable when a page has no callout/terminal and
    therefore no ``arrows.json`` entry.  ``page_geometry_status`` is the same
    cheap MuPDF count used by the arrow stage; failure favours recall and uses
    the dense budget.
    """
    try:
        geometry = arrows.page_geometry_status(pdf_path(slug), int(page) - 1)
    except Exception:                                      # noqa: BLE001
        return LINETYPE_DENSE_TIMEOUT
    try:
        paths = int((geometry or {}).get("vector_paths") or 0)
    except (AttributeError, TypeError, ValueError):
        return LINETYPE_DENSE_TIMEOUT
    return (LINETYPE_DENSE_TIMEOUT
            if paths >= LINETYPE_DENSE_PATHS else LINETYPE_TIMEOUT)


_LINETYPE_TIMEOUT_BUDGET_RE = re.compile(
    r"\blinetype\s+sidecar\s+timeout\s+after\s+([0-9]+(?:\.[0-9]+)?)s\b",
    re.I)


def _linetype_failed_timeout_budget(error):
    """Deadline embedded in a persisted sidecar timeout marker, if any."""
    match = _LINETYPE_TIMEOUT_BUDGET_RE.search(str(error or ""))
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def _linetype_failure_budget_increased(slug, page, arrow_entry, error):
    """Whether a timeout failure is eligible under a larger current budget."""
    previous = _linetype_failed_timeout_budget(error)
    return (previous is not None
            and previous < _linetype_timeout_for(slug, page, arrow_entry))


def _linetype_retryable(error):
    """Only retry failures which might change in a fresh sidecar process.

    A timeout has already consumed its complete deadline.  Named sidecar
    errors (PAGE_IR_ERROR, CLUSTER_ERROR, etc.) are deterministic for the same
    PDF bytes and engine version.  Retrying either three times was the direct
    cause of web uploads sitting at 99% for hours.
    """
    if is_timeout_error(error):
        return False
    message = str(error)
    if re.search(r"linetype sidecar [A-Z][A-Z0-9_]+:", message):
        return False
    return "SourceAlignmentError" not in message


def _legend_linetype_retryable(error):
    """Apply the ordinary bounded policy plus legend structured errors."""
    if not _linetype_retryable(error):
        return False
    return re.search(
        r"legend line-type sidecar [A-Z][A-Z0-9_]+:",
        str(error)) is None


def _linetype_job_still_current(slug, page, sig):
    """Return whether this captured sheet/signature is still schedulable.

    Reusing ``_linetype_jobs`` keeps this check on exactly the same prerequisite
    contract as normal scheduling: current results, current arrows (including
    placement anchors and PDF revision), at least one terminal, and no already
    current successful line-type cache.  A failed cache remains eligible because
    it intentionally has no ``bindings`` key, so this lookup does not turn a
    retryable failure into a permanent cache hit.
    """
    jobs, _warnings = _linetype_jobs(slug)
    return any(candidate[0] == int(page) and candidate[3] == sig
               for candidate in jobs)


def _legend_linetype_job_still_current(slug, page, sig):
    """Whether the captured supervised samples are still the current work."""
    jobs, _warnings = _legend_linetype_jobs(slug)
    return any(candidate[0] == int(page) and candidate[2] == sig
               for candidate in jobs)


def _linetype_one(slug, page, items, arrow_entry, sig, should_cancel=None):
    """一页的线型聚类 + 绑定，结果原子写回 linetypes.json。

    page 是 1-based（与 results.json 的页键、引擎 API 一致）。
    """
    with _linetype_page_lock(slug, page, cancelled=should_cancel):
        _raise_if_cancelled(should_cancel)
        # 等锁时另一个进程可能已经算完这页，也可能新的
        # results/arrows 已经把捕获的 sig 作废。两种情况都安静丢弃，
        # 不把「已被取代」误报成页面失败。
        if not _linetype_job_still_current(slug, page, sig):
            return page, 0, None

        entry = None
        error = None
        timeout = _linetype_timeout_for(slug, page, arrow_entry)
        for attempt in range(RETRIES + 1):
            try:
                with _slot_pool(
                        "heavy-sidecar", HEAVY_SIDECAR_SLOTS).slot(
                            cancelled=should_cancel):
                    _raise_if_cancelled(should_cancel)
                    entry = linetypes.compute_page_linetypes(
                        pdf_path(slug), page, items, arrow_entry, sig=sig,
                        timeout=timeout)
                break
            except SlotWaitCancelled as exc:
                raise Cancelled() from exc
            except Cancelled:
                raise
            except Exception as exc:                        # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < RETRIES and _linetype_retryable(exc):
                    _retry_pause(5 * (attempt + 1), should_cancel)
                else:
                    break

        # 聚类可能跑几分钟。落盘前立即重读先决条件：如果这期间
        # PDF/results/arrows/placements/engine 任一变了，这份结果就不再属于
        # 当前页面。无论成功还是失败记录都不允许覆盖新一代缓存。
        if not _linetype_job_still_current(slug, page, sig):
            return page, 0, None

        # 一页一个文件，同页由 advisory lock 保护，不需要跨页
        # _IO_LOCK。失败同样必须落盘，且形状要被当期判据拒绝：
        # 不写 bindings，has_current_linetypes 判假，下次自动重试。
        linetypes.save_page(slug, page, entry if entry is not None else {
            "sig": sig, "v": linetypes.LINETYPE_VERSION, "error": error})
        if entry is None:
            return page, None, error
        return page, len(entry.get("used_all") or ()), None


def _legend_linetype_one(slug, page, samples, sig, should_cancel=None):
    """Extract and match one page's supervised legend swatches.

    The successful cache is owned by :mod:`steps.legend_linetypes`; a failure
    marker intentionally omits ``ok: true`` so ``has_current`` rejects it and
    a later run retries.  Both generations are guarded by the same stable page
    lock as ordinary line-type work, preventing two heavy engines from parsing
    one sheet concurrently in different web/backfill processes.
    """
    with _linetype_page_lock(slug, page, cancelled=should_cancel):
        _raise_if_cancelled(should_cancel)
        if not _legend_linetype_job_still_current(slug, page, sig):
            return page, 0, None

        entry = None
        error = None
        timeout = _legend_linetype_timeout_for(slug, page)
        for attempt in range(RETRIES + 1):
            try:
                with _slot_pool(
                        "heavy-sidecar", HEAVY_SIDECAR_SLOTS).slot(
                            cancelled=should_cancel):
                    _raise_if_cancelled(should_cancel)
                    entry = legend_linetypes.compute_page(
                        pdf_path(slug), page, samples, sig=sig,
                        timeout=timeout)
                break
            except SlotWaitCancelled as exc:
                raise Cancelled() from exc
            except Cancelled:
                raise
            except Exception as exc:                        # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                if attempt < RETRIES and _legend_linetype_retryable(exc):
                    _retry_pause(5 * (attempt + 1), should_cancel)
                else:
                    break

        # A symbol re-detection can replace the supervised box while this
        # dense page is running.  Never publish either generation against a
        # superseded sample signature.
        if not _legend_linetype_job_still_current(slug, page, sig):
            return page, 0, None

        if entry is not None and (
                not isinstance(entry, dict) or entry.get("ok") is not True):
            error = "legend line-type sidecar returned no successful payload"
            entry = None
        if entry is None:
            error = error or "legend line-type sidecar returned no result"
            entry = {"sig": sig, "v": legend_linetypes.VERSION,
                     "error": error}
        legend_linetypes.save_page(slug, page, entry)
        if entry.get("ok") is not True:
            return page, None, error
        return page, len(entry.get("line_types") or ()), None


def materialize_all_linetypes(slug, page, sig):
    """Build the optional all-line-types geometry for one current main result.

    The viewer calls this only after its cache-only GET reports missing or
    stale geometry.  The expensive rerun is serialized with normal line-type
    work for the same sheet, shares the global heavy-sidecar capacity, and is
    published only after operation-set parity is checked again against the
    latest main cache.
    """
    page = int(page)
    with _linetype_page_lock(slug, page):
        main_entry = linetypes.load_page(slug, page)
        if not linetypes.has_current_linetypes(main_entry, sig):
            raise RuntimeError("main line-type result changed before generation")
        cached = linetypes.validated_all_page(
            linetypes.load_all_page(slug, page), main_entry, sig)
        if cached is not None:
            return cached

        arrow_entry = load_json(
            slug_dir(slug) / "arrows.json", {}).get(str(page)) or {}
        timeout = _linetype_timeout_for(slug, page, arrow_entry)
        with _slot_pool("heavy-sidecar", HEAVY_SIDECAR_SLOTS).slot():
            generated = linetypes.compute_all_page_geometry(
                pdf_path(slug), page, main_entry, timeout=timeout)

        # A rerun or upload can replace prerequisites during the minutes spent
        # in the sidecar.  Verify against the latest cache, not merely the
        # object captured before computation, before the atomic write.
        latest = linetypes.load_page(slug, page)
        if not linetypes.has_current_linetypes(latest, sig):
            raise RuntimeError("main line-type result changed during generation")
        generated = linetypes.verify_all_page_geometry(latest, generated)
        linetypes.save_all_page(slug, page, generated)
        return generated


def _stage_linetypes(slug, on_progress=None, should_cancel=None):
    """Run arrow-bound and supervised-legend line matching together.

    页级并发由 ``LINETYPE_PAGE_WORKERS`` 控制（生产前台为 2）；每个线程等待一个
    单页边车，边车内部再按 ``LINETYPE_CPU_BUDGET`` 使用 worker 预算（生产前台 4，
    低优先级 refresh 2）。每页独立缓存，并受跨进程页锁与共享 heavy slot 保护。
    """
    # ARROWS=0 used to mean the ordinary terminal-bound channel never ran.
    # Keep that feature-toggle contract even though the independent supervised
    # legend channel now makes this stage reachable without arrows.
    jobs, warnings = (_linetype_jobs(slug) if arrows.ENABLED else ([], []))
    legend_jobs, legend_warnings = _legend_linetype_jobs(slug)
    warnings.extend(legend_warnings)
    total_jobs = len(jobs) + len(legend_jobs)
    print(f"linetype jobs: {len(jobs)} arrow + {len(legend_jobs)} legend  "
          f"page_workers={LINETYPE_PAGE_WORKERS}", flush=True)
    if on_progress:
        on_progress(0, total_jobs)
    total = 0
    legend_total = 0
    done = 0
    # 页级并行。每个线程只是等一个边车子进程，GIL 不碍事；真正吃 CPU 的是
    # 子进程。串行时实测整机 CPU 只有 13~16%（16 核 / 32 线程），因为单页内部
    # 有大段单线程阶段（PageIR 抽取、序列化、进程启停），并行跑多页正好把这些
    # 空档填上。一页一个缓存文件，所以并发写盘不会互相等。
    with ThreadPoolExecutor(max_workers=max(1, LINETYPE_PAGE_WORKERS)) as pool:
        futures = {}
        for page_job in jobs:
            future = pool.submit(
                _linetype_one, slug, *page_job, should_cancel)
            futures[future] = ("arrow", page_job[0])
        for page_job in legend_jobs:
            future = pool.submit(
                _legend_linetype_one, slug, *page_job, should_cancel)
            futures[future] = ("legend", page_job[0])
        heartbeat_stop = threading.Event()

        def heartbeat():
            # A valid dense page can spend minutes inside the local engine.
            # Keep the job card visibly alive even when no page has completed
            # since the previous browser poll; this is not fake completion
            # progress, just an explicit bounded-work heartbeat.
            while not heartbeat_stop.wait(10):
                if should_cancel and should_cancel():
                    return
                remaining = sum(not future.done() for future in futures)
                if not remaining:
                    return
                _set(
                    slug,
                    detail=(f"Line-type engine active: {done}/{total_jobs} "
                            f"sheets finished, {remaining} remaining "
                            f"(per-sheet limit {LINETYPE_TIMEOUT}s; dense "
                            f"{LINETYPE_DENSE_TIMEOUT}s)"))

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
        try:
            for future in as_completed(futures):
                source, submitted = futures[future]
                if should_cancel and should_cancel():
                    for pending in futures:
                        pending.cancel()
                    break
                try:
                    page, count, error = future.result()
                except Exception as exc:                    # noqa: BLE001
                    page, count = submitted, None
                    error = f"{type(exc).__name__}: {exc}"
                done += 1
                if on_progress:
                    on_progress(done, total_jobs)
                if error:
                    if source == "legend":
                        warnings.append(
                            f"P{page} legend line-type matching failed: {error}")
                    else:
                        warnings.append(
                            f"P{page} line-type clustering failed: {error}")
                    print(f"[{done}/{total_jobs}] {slug} P{page} "
                          f"({source}): ERROR {error}",
                          flush=True)
                else:
                    if source == "legend":
                        legend_total += count or 0
                    else:
                        total += count or 0
                    print(f"[{done}/{total_jobs}] {slug} P{page} "
                          f"({source}): {count} line types", flush=True)
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
    print(f"linetypes total: {total} arrow + {legend_total} legend", flush=True)
    return warnings


# ------------------------------------------------------------- run pipeline --
def _start_job_claimed(slug, target=None, model=None, _resume=False):
    """Start a worker after the caller has reserved ``slug``."""
    target = (target or "").strip()
    model = model if (model and model in PRICING) else None
    is_default = is_default_target(target)
    mode = "fence" if is_default else "custom"
    try:
        base_llm, base_wall = _carry_baseline(slug)
        _set(slug, mode=mode, target=(target or TARGET_DEFAULT),
             stage="queued", detail="Queued…", progress=0.0,
             stage_done=0, stage_total=0, stage_unit=None, pages_total=0,
             done=False, ok=None, error=None, cancelled=False,
             outcome=None, results_available=False, repairing=None,
             cancel_requested=False, warnings=[], heartbeat_at=time.time(),
             started=time.time(), finished=None,
             llm=(base_llm or None), wall_seconds=(base_wall or None),
             model=(model or MODEL_NAME),
             created=datetime.now().isoformat(timespec="seconds"))
        # Take the response snapshot before start(). Once start() returns,
        # ownership of the reservation belongs to _run_reserved.
        status = get_job(slug)
        thread = threading.Thread(
            target=_run_reserved,
            args=(slug, (target or None), base_llm, base_wall, model, _resume),
            daemon=True)
        thread.start()
    except Exception as exc:                                  # noqa: BLE001
        _release_job_start(slug)
        try:
            _set(slug, done=True, ok=False, outcome="failed", stage="error",
                 stage_unit=None, detail="Background worker did not start",
                 error=f"{type(exc).__name__}: {exc}", finished=time.time())
        except Exception:                                     # noqa: BLE001
            pass
        raise JobStartError(f"{type(exc).__name__}: {exc}") from exc
    return status


def start_job(slug, target=None, model=None, _resume=False):
    """Kick off background processing for ONE detection target.

    ``target`` is the user-editable "what to detect" text (see
    steps/text/target.py).  Blank / equal-to-default → fence mode (the full
    six-stage text → symbols → views → placements → arrows → linetypes chain
    in production); anything else → custom mode, which runs the text step only,
    because the other five stages are fence-specific.

    ``model`` pins every paid call in this run to one model id (see
    core.config.set_model_override).  None keeps the process default, so
    existing callers are unaffected.
    """
    if not _claim_job_start(slug, resume=_resume):
        return get_job(slug)
    return _start_job_claimed(slug, target=target, model=model,
                              _resume=_resume)


def restart_job(slug, target=None, model=None, *, reset=True):
    """Atomically reserve, optionally clear caches, and rerun one project."""
    _claim_job_start(slug)
    try:
        if job_running(slug):
            raise JobStartError("this project is already in progress")
        cleared = reset_project_cache(slug) if reset else []
        status = _start_job_claimed(slug, target=target, model=model)
        return status, cleared
    except Exception:
        # Harmless if _start_job_claimed already released after a start error.
        _release_job_start(slug)
        raise


def _run_reserved(slug, target, base_llm=None, base_wall=0.0, model=None,
                  _resume=False):
    try:
        _run(slug, target, base_llm, base_wall, model, _resume)
    finally:
        _release_job_start(slug)


def _run(slug, target, base_llm=None, base_wall=0.0, model=None,
         _resume=False):
    cancel_ev = _register_cancel_event(slug)
    # request_cancel() can win the tiny race between Thread.start() and this
    # registration.  Honour the persisted/in-memory bit as well as the Event.
    with _JOBS_LOCK:
        if (JOBS.get(slug) or {}).get("cancel_requested"):
            cancel_ev.set()
    try:
        # Do not persist a queued detail for an automatic resume before its
        # post-lock done recheck.  During graceful replacement the old worker
        # may already have published the authoritative terminal card.
        if not _resume:
            _set(slug, stage="queued",
                 detail=(f"Waiting for a processing slot "
                         f"({MAX_PARALLEL_JOBS} PDFs can run together)…"))
        # Same-slug first, then shared capacity: a duplicate resume never holds
        # one of the scarce project slots while waiting for the live owner.
        with stable_named_lock(_job_run_lock_path(slug),
                               cancelled=cancel_ev.is_set):
            with _interprocess_proc_lock(cancelled=cancel_ev.is_set):
                # A replacement worker may have queued this resume while the
                # old worker was finishing. Re-read only automatic resumes once
                # both locks are ours; explicit reruns still run.
                if _resume:
                    persisted = load_json(_job_file(slug), None)
                    if isinstance(persisted, dict) and persisted.get("done"):
                        with _JOBS_LOCK:
                            JOBS[slug] = persisted
                        return
                with _JOBS_LOCK:
                    if (JOBS.get(slug) or {}).get("cancel_requested"):
                        cancel_ev.set()
                _run_owned(slug, target, cancel_ev.is_set,
                           base_llm=base_llm, base_wall=base_wall, model=model)
    except SlotWaitCancelled:
        total = _merge_llm(base_llm, None)
        _set(slug, done=True, ok=False, cancelled=True, stage="cancelled",
             outcome="failed", results_available=results_path(slug).exists(),
             stage_unit=None, detail="Cancelled while queued", llm=total,
             wall_seconds=(float(base_wall) if base_wall else None),
             finished=time.time())
    except Exception as exc:                                  # noqa: BLE001
        # Lock-directory/flock failures happen before _run_owned's pipeline
        # exception boundary. Never leave such a worker looking queued forever.
        traceback.print_exc()
        total = _merge_llm(base_llm, None)
        _set(slug, done=True, ok=False, cancelled=False, stage="error",
             outcome="failed", results_available=results_path(slug).exists(),
             stage_unit=None, detail="Processing could not start", llm=total,
             error=f"{type(exc).__name__}: {exc}",
             wall_seconds=(float(base_wall) if base_wall else None),
             finished=time.time())
    finally:
        _unregister_cancel_event(slug, cancel_ev)


def _run_owned(slug, target, should_cancel, base_llm=None, base_wall=0.0,
               model=None):
    """Run one job while its slug lock and one project slot are held."""
    processing_started = time.time()
    with _RUNNING_LOCK:
        _RUNNING[slug] = {
            "base": (base_llm or dict(_ZERO_LLM)),
            "base_wall": float(base_wall or 0.0),
            "processing_started": processing_started,
        }
    with _JOBS_LOCK:
        queued_at = (JOBS.get(slug) or {}).get("started")
    _set(slug, processing_started=processing_started,
         queue_seconds=(round(processing_started - queued_at, 1)
                        if queued_at else 0.0),
         detail="Processing started…")
    try:
        # Both values are ContextVars; paid worker submissions propagate them.
        set_model_override(model)
        RECORDER.start(slug)
        # cumulative cost is flushed to disk on every completed call via
        # RECORDER.on_update (_flush_running), so a crash keeps the exact spend
        if should_cancel():
            raise Cancelled()
        # MuPDF open/page-count can wait behind a render's FITZ_LOCK and should
        # never delay the HTTP upload response.  It belongs to the background
        # run; the queued card legitimately starts with pages_total=0.
        with _job_heartbeat(slug):
            _set(slug, pages_total=page_count_of(slug))
            _run_pipeline(slug, target, should_cancel)
        summary = RECORDER.stop(slug)
        _persist_llm(slug, summary)
        _finish(slug, ok=True, summary=summary)
    except Cancelled:
        summary = RECORDER.stop(slug)
        _persist_llm(slug, summary)     # keep whatever spend already happened
        total = _merge_llm(_running_state(slug).get("base"), summary)
        _set(slug, done=True, ok=False, cancelled=True, stage="cancelled",
             outcome="failed", results_available=results_path(slug).exists(),
             stage_unit=None, detail="Cancelled", llm=total,
             finished=time.time())
        wall = _wall_now(slug)
        if wall is not None:
            _set(slug, wall_seconds=wall)
    except Exception as e:                                     # noqa: BLE001
        summary = RECORDER.stop(slug)
        traceback.print_exc()
        _finish(slug, ok=False, summary=summary,
                error=f"{type(e).__name__}: {e}")
    finally:
        with _RUNNING_LOCK:
            _RUNNING.pop(slug, None)
        set_model_override(None)


def _stage_progress(slug, lo, hi):
    """Return an on_progress(done, total) that maps a stage's page completion
    onto the [lo, hi] slice of the overall 0..1 progress bar."""
    def cb(done, total):
        frac = (done / total) if total else 0.0
        _set(slug, progress=round(lo + (hi - lo) * frac, 4),
             stage_done=int(done), stage_total=int(total))
    return cb


def _run_pipeline(slug, target, should_cancel=None):
    """text → symbols → views → placements（fence 目标），custom 目标只跑 text.

    每阶段的 wall 时间打成 ``[timing] <name> = Xs``（排查主耗时用），阶段边界
    guard() 检查取消，每阶段的失败页收进 job["warnings"]。
    """
    sc = should_cancel or (lambda: False)

    def guard():
        if sc():
            raise Cancelled()

    _timings = []

    def timed(name, fn):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        _timings.append((name, dt))
        print(f"  [timing] {name} = {dt:.1f}s", flush=True)
        return out

    def done_msg():
        print(f"  [timing] TOTAL = {sum(d for _, d in _timings):.1f}s  "
              f"breakdown={[(n, round(d, 1)) for n, d in _timings]}",
              flush=True)

    is_default = is_default_target(target)
    text_hi = 0.55 if is_default else 1.0
    _set(slug, stage="text", progress=0.0, stage_done=0, stage_total=0,
         stage_unit="percent",
         detail="Vector text layer judge + image VLM detection (same target), fusing…")
    _warn(slug, timed("text(vec+judge+vlm)", lambda: _stage_text(
        slug, target,
        on_progress=_stage_progress(slug, 0.0, text_hi), should_cancel=sc)))
    guard()
    _snapshot(slug)
    if not is_default:
        # custom target: the fence-specific symbol/view/placement steps do not
        # apply to an arbitrary detection target.
        _set(slug, stage="done", detail="Text step done", progress=1.0,
             stage_done=0, stage_total=0, stage_unit=None)
        done_msg()
        return

    _set(slug, stage="symbols", detail="Legend symbol detection (VLM)…",
         progress=0.55, stage_done=0, stage_total=0, stage_unit="sheets")
    _warn(slug, timed("symbols", lambda: _stage_symbols(
        slug, on_progress=_stage_progress(slug, 0.55, 0.80), should_cancel=sc)))
    guard()
    _snapshot(slug)

    _set(slug, stage="views", detail="View classification (VLM)…",
         progress=0.80, stage_done=0, stage_total=0, stage_unit="sheets")
    _warn(slug, timed("views", lambda: _stage_views(
        slug, on_progress=_stage_progress(slug, 0.80, 0.92), should_cancel=sc)))
    guard()
    _snapshot(slug)

    # Legend line samples are independent from arrows.  When only LINETYPES is
    # on, reserve the same final 2% slice that line types already occupied in
    # the full chain; when both local seams are off, placements still reaches
    # 100% exactly as before.
    plc_hi = (0.96 if arrows.ENABLED else
              (0.98 if linetypes.ENABLED else 1.0))
    _set(slug, stage="placements",
         detail="Shape placement matching (local, no model)…",
         progress=0.92, stage_done=0, stage_total=0, stage_unit="sheets")
    _warn(slug, timed("placements", lambda: _stage_placements(
        slug, on_progress=_stage_progress(slug, 0.92, plc_hi),
        should_cancel=sc)))
    guard()

    linetype_lo = plc_hi
    if arrows.ENABLED:
        # 普通线型仍然绑箭头末端；legend 线型走独立的 supervised channel。
        arw_hi = 0.98 if linetypes.ENABLED else 1.0
        _set(slug, stage="arrows", detail="Arrow / leader detection…",
             progress=plc_hi, stage_done=0, stage_total=0,
             stage_unit="sheets")
        _warn(slug, timed("arrows", lambda: _stage_arrows(
            slug, on_progress=_stage_progress(slug, plc_hi, arw_hi),
            should_cancel=sc)))
        guard()
        _snapshot(slug)
        linetype_lo = arw_hi

    if linetypes.ENABLED:
        _set(slug, stage="linetypes",
             detail=("Line-type clustering and legend matching "
                     "(local sidecar, no model)…"),
             progress=linetype_lo, stage_done=0, stage_total=0,
             stage_unit="sheets")
        _warn(slug, timed("linetypes", lambda: _stage_linetypes(
            slug, on_progress=_stage_progress(slug, linetype_lo, 1.0),
            should_cancel=sc)))
        guard()
        _snapshot(slug)

    _set(slug, stage="done", detail="Done", progress=1.0,
         stage_done=0, stage_total=0, stage_unit=None)
    done_msg()


def resume_interrupted():
    """Reload persisted cards and resume work interrupted by a server restart.

    Every paid/raw stage checkpoints by page and validates cache identity, so
    continuing the same upload neither reuses another PDF's result nor repeats
    completed calls.  A user-requested cancellation remains cancelled; only an
    unfinished, non-cancelled job with its input PDF still present is resumed.
    """
    if not JOBS_DIR.exists():
        return
    # During a graceful worker replacement, the old worker can still be
    # checkpointing its final page while the new worker imports wsgi.py.  Wait
    # for that run before deciding what is genuinely unfinished; otherwise the
    # new worker can overwrite a just-finished card with done=false.
    with _interprocess_proc_lock(all_slots=True):
        resumable = _restore_interrupted_cards()
    for slug, target, model in resumable:
        start_job(slug, target=target, model=model, _resume=True)
    if resumable:
        print(f"[resume] relaunched {len(resumable)} unfinished job(s): "
              f"{[slug for slug, _target, _model in resumable]}", flush=True)


def _restore_interrupted_cards():
    """Restore cards and return resumable jobs while the process lock is held."""
    resumable = []
    for f in sorted(JOBS_DIR.glob("*.json")):
        job = load_json(f, None)
        if not isinstance(job, dict):
            continue
        slug = job.get("slug") or f.stem
        if not is_valid_slug(slug):
            continue
        with _JOBS_LOCK:
            JOBS[slug] = job                 # restore for the gallery/cards
        if job.get("done"):
            continue                          # already finished — leave as-is
        if job.get("cancel_requested"):
            _set(slug, done=True, ok=False, cancelled=True,
                 outcome="failed", stage="cancelled", stage_unit=None,
                 results_available=results_path(slug).exists(),
                 detail="Cancelled")
            continue
        if not pdf_path(slug).exists():
            _set(slug, done=True, ok=False, cancelled=False, stage="error",
                 outcome="failed", stage_unit=None,
                 results_available=results_path(slug).exists(),
                 error="input.pdf is missing; cannot resume after restart",
                 detail="Resume failed: input PDF is missing")
            continue
        resumable.append((slug, job.get("target"), job.get("model")))
        _set(slug, done=False, ok=None, outcome=None,
             results_available=False, cancelled=False, stage="queued",
             stage_unit=None, repairing=None,
             detail="Server restarted — resuming from saved page checkpoints…")
    return resumable
