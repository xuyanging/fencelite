"""调用线型聚类边车（独立 venv 的一次性子进程）.

边车本体和它的解释器都在 ``tools/linetype_sidecar/``，理由写在 run.py 的
docstring 里（PyMuPDF 版本闸 / issue #5042 会崩进程 / spawn 会重入调用方的
__main__）。这里只负责：拼 job、起子进程、把失败变成显式异常。

**绝不把失败写成空结果**：空结果的意思是"这页确实没有线型可绑"，和"这页没算过"
在下游是完全不同的两件事 —— arrows 那一层已经为此付过代价（OOM 被写成空结果，
一整页的缺失完全不可见）。
"""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_CLIENT = Path(__file__).resolve()
_SIDECAR_DIR = _BASE_DIR / "tools" / "linetype_sidecar"
_RUNNER = _SIDECAR_DIR / "run.py"
_ALL_RUNNER = _SIDECAR_DIR / "run_all.py"
_VENDORED_ENGINE_DIR = _SIDECAR_DIR / "engine" / "line_type_engine"
# run.py supports an explicit engine checkout for controlled verification.  The
# cache digest and availability check must follow the same directory; hashing
# the vendored tree while executing an override would publish an untraceable mix.
_ENGINE_DIR = Path(os.environ.get(
    "LINETYPE_ENGINE_PATH", str(_VENDORED_ENGINE_DIR))).resolve()


def _venv_python(venv_dir):
    """venv 里的解释器。Windows 是 Scripts/python.exe，POSIX 是 bin/python ——
    写死任何一个都会让另一个平台上边车"missing"。"""
    windows = venv_dir / "Scripts" / "python.exe"
    posix = venv_dir / "bin" / "python"
    if windows.is_file():
        return windows
    if posix.is_file():
        return posix
    # 都不存在时按当前平台给出**期望**路径，好让 sidecar_available() 的报错
    # 指向该装的地方，而不是指向另一个平台的路径。
    return windows if os.name == "nt" else posix


_PYTHON = Path(os.environ.get("LINETYPE_PYTHON", "")
               or _venv_python(_SIDECAR_DIR / "venv"))

# 直接调用边车时的普通页上限。Web 编排层会用 path 数分档：普通页
# 600s，>=40k path 的超密页 3600s。这里默认 600s，防止脱离 web 的
# 调用者重新引入无界等待。
TIMEOUT = int(os.environ.get("LINETYPE_TIMEOUT", "600"))
# 每页给引擎的 worker 数。**跟机器走**，并且**不进缓存签名**（见 engine_digest）。
#
# budget 不变性已经证明：tools/linetype_sidecar/verify_budget_invariance.py 在
# 6 页 × budget{1,2,8,16,32} = 30 次运行上比整份输出逐字段（只忽略两个计时字段
# 和 budget 自己），全部逐位相同。测试点按 plan_single_page_execution 的分支挑，
# 覆盖了 budget=1 的非并发计划、budget=8 让 method2 进多 worker（就是那条结构
# 不同的代码路径：method2/text_family.py:2100-2180 先物化全部 candidate_lists
# 再按批 speculative 求值，而 worker_count==1 是边算边 union）、budget=32 顶到
# method2 的 cap=8；6 页全都真的产出了 method2 类型，组数从 164 跨到 7854。
#
# 取值不再是 16：同一份实测显示 1→8 大约快一倍，8→32 基本平、甚至退化
# （wasd_pollard P5 是 46.8s → 48.4s → 54.1s）。每页给多了纯属浪费，核该拿去
# 开更多**页**（job.py 的 LINETYPE_PAGE_WORKERS）。页级并发本来就在抢核，
# 每页更该压小。
def _default_cpu_budget():
    from core import hw
    return hw.clamp(hw.cpu_threads() // 8, 2, 6)


CPU_BUDGET = int(os.environ.get("LINETYPE_CPU_BUDGET", "")
                 or _default_cpu_budget())
# 每个末端回传几个候选线型。投票只可能从各末端的候选里选，所以这个数决定了
# 上层判据的闭合性：调小到 1 就等于禁掉"被多数票改判"。
TOP_K = int(os.environ.get("LINETYPE_TOP_K", "3"))
RUN_TOUCH_PT = os.environ.get("LINETYPE_RUN_TOUCH_PT", "0.5")

_DIGEST_LOCK = threading.Lock()
_DIGEST = {"value": None, "stamp": None}


def sidecar_available():
    return _RUNNER.is_file() and _PYTHON.is_file() and _ENGINE_DIR.is_dir()


def all_sidecar_available():
    """Whether the optional full-page geometry producer can run."""
    return (_ALL_RUNNER.is_file() and _PYTHON.is_file()
            and _ENGINE_DIR.is_dir())


def sidecar_probe(timeout=20):
    """Prove the isolated interpreter and required algorithm deps import.

    File existence alone cannot distinguish a healthy venv from one that was
    copied without its packages. This startup-only subprocess uses the same
    cleared ``PYTHONPATH`` contract as real line-type jobs and catches that
    failure before text/symbol model calls spend time or money.
    """
    if not sidecar_available():
        raise RuntimeError(
            f"linetype sidecar missing: python={_PYTHON} runner={_RUNNER} "
            f"engine={_ENGINE_DIR}")
    probe = (
        "import fitz,numpy,pypdf,scipy;"
        "print(fitz.__version__,numpy.__version__,"
        "pypdf.__version__,scipy.__version__)"
    )
    try:
        proc = subprocess.run(
            [str(_PYTHON), "-B", "-c", probe],
            capture_output=True, text=True,
            timeout=max(1, int(timeout)), check=False,
            env={**os.environ, "PYTHONPATH": "", "PYTHONUTF8": "1"})
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"linetype Python probe failed: {exc}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "no output").strip()
        raise RuntimeError(
            f"linetype Python probe exited {proc.returncode}: {detail}")
    return (proc.stdout or "linetype dependencies available").strip()


def all_geometry_digest():
    """Content identity of the optional full-geometry producer.

    ``run_all.py`` is intentionally excluded from the main line-type cache
    signature so that viewer-only changes never invalidate expensive primary
    results.  Its own sidecar cache still needs an identity, otherwise an old
    coordinate projection can survive a producer update under the same main
    ``sig``.
    """
    return hashlib.sha256(_ALL_RUNNER.read_bytes()).hexdigest()


def engine_digest():
    """边车这一整套「能改变结果的东西」的摘要 —— 进缓存签名.

    包含三部分，缺一不可：

      * vendored 引擎源码树。动了算法（哪怕一个阈值）就换摘要。
      * ``run.py`` 与本客户端协议。前者决定提取/绑定几何，后者决定哪些 target
        字段真的越过进程边界；任一不进摘要都会让旧缓存静默复用错误协议。
      * 边车 venv 里 PyMuPDF / pypdf / scipy 的版本。scipy 尤其重要：
        unknown_pattern_split 里 5 处 ``try: import scipy`` 的纯 Python 回退
        **不是逐位等价** —— _delaunay_edges 的等价代价平票能让一整个线型
        出现或消失。版本从 dist-info 目录名读，纯文件系统、不起子进程。
    """
    # memo 必须能失效。只按「算过一次」缓存的话，改了 run.py 或重新 vendor
    # 引擎之后，长跑的 webapp 进程会一直拿旧摘要算期望签名 —— 盘上明明是当期
    # 结果，界面却全部报 stale，而且没有任何报错。用一次 stat 扫描（路径 +
    # mtime_ns + size）当 memo 的键：改动必然改 stat，内容哈希只在真变了时才重算。
    stamp = _stat_stamp()
    with _DIGEST_LOCK:
        if _DIGEST["value"] and _DIGEST.get("stamp") == stamp:
            return _DIGEST["value"]
        hasher = hashlib.sha256()
        hasher.update(b"engine-tree\0")
        hasher.update(_tree_digest(_ENGINE_DIR).encode())
        hasher.update(b"\0runner\0")
        try:
            hasher.update(hashlib.sha256(_RUNNER.read_bytes()).hexdigest().encode())
        except OSError:
            hasher.update(b"no-runner")
        hasher.update(b"\0client\0")
        try:
            hasher.update(hashlib.sha256(_CLIENT.read_bytes()).hexdigest().encode())
        except OSError:
            hasher.update(b"no-client")
        # Environment-driven values are executable algorithm inputs, not mere
        # performance tuning.  P4 center consensus needs the second ranked
        # candidate, so TOP_K=1 and TOP_K=3 must never share a cache key.
        hasher.update(b"\0runtime-config\0")
        hasher.update(f"top_k={TOP_K}\0".encode())
        hasher.update(f"run_touch_pt={RUN_TOUCH_PT}\0".encode())
        hasher.update(b"\0deps\0")
        for name, version in sorted(dep_versions().items()):
            hasher.update(f"{name}={version}\0".encode())
        # **cpu_budget 刻意不进签名。**
        #
        # 原来它在里面，理由是"不变性只在两页上实测过，没有普遍证明"。现在证明
        # 了（见 CPU_BUDGET 处的说明：6 页 × 5 档 = 30 次运行逐位相同）。
        #
        # 拿掉它同时修掉一个**现存的谎**：引擎里
        # budget = min(cpu_budget or available, available)（scheduling.py:84-85）
        # —— 4 核机器上实际只有 4，而签名里写的是 16。留着它既不反映真实执行，
        # 又会让「按机器核数决定 budget」变成「换台机器全部缓存作废」，
        # 与「弱机器也能正常跑」正好相反。
        _DIGEST["value"] = hasher.hexdigest()
        _DIGEST["stamp"] = stamp
        return _DIGEST["value"]


def _stat_stamp():
    """引擎树 + runner/client 的 (路径, mtime_ns, size) 摘要 —— 纯 stat。"""
    parts = []
    try:
        stat = _CLIENT.stat()
        parts.append(f"client:{stat.st_mtime_ns}:{stat.st_size}")
    except OSError:
        parts.append("client:missing")
    try:
        stat = _RUNNER.stat()
        parts.append(f"run:{stat.st_mtime_ns}:{stat.st_size}")
    except OSError:
        parts.append("run:missing")
    if _ENGINE_DIR.is_dir():
        for path in sorted(_ENGINE_DIR.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def dep_versions():
    """边车 venv 里影响结果的依赖版本，从 dist-info 目录名读（不起子进程）。"""
    out = {}
    wanted = {"pymupdf": "pymupdf", "pypdf": "pypdf", "scipy": "scipy",
              "numpy": "numpy"}
    # site-packages 的位置分平台：Windows 是 <venv>/Lib/site-packages，
    # POSIX 是 <venv>/lib/pythonX.Y/site-packages。只写 Windows 那个的话，
    # Linux 上这里会全部返回 "missing" —— 后果不是报错而是**静默**：
    # scipy 版本这道闸失效（unknown_pattern_split 里 5 处纯 Python 回退
    # 不是逐位等价），而且 digest 会和 Windows 上算出来的不一样。
    root = _PYTHON.parent.parent
    candidates = [root / "Lib" / "site-packages"]
    candidates.extend(sorted((root / "lib").glob("python*/site-packages"))
                      if (root / "lib").is_dir() else [])
    for site in candidates:
        if not site.is_dir():
            continue
        for path in site.glob("*.dist-info"):
            stem = path.name[:-len(".dist-info")]
            name, _, version = stem.rpartition("-")
            key = wanted.get(name.lower().replace("_", "-"))
            if key:
                out[key] = version
    for key in wanted.values():
        out.setdefault(key, "missing")
    return out


def _tree_digest(root):
    if not root.is_dir():
        return "no-engine"
    hasher = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix in (".pyc", ".pyo"):
            continue
        if "__pycache__" in path.parts:
            continue
        hasher.update(path.relative_to(root).as_posix().encode())
        hasher.update(b"\0")
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def _kill_tree(pid):
    """连同所有后代一起杀掉 —— **两个平台都必须真的杀到孙进程**。

    为什么不能只杀直接子进程：cpu_budget>=2 时引擎会开 multiprocessing spawn 池，
    那些孙进程继承了 stdout/stderr 的写端。只杀直接子进程的话管道永不关闭，
    communicate() 无限阻塞 —— 实测 bristol P24 在"30 分钟超时"下卡了 10.5 小时，
    孙进程满核空转 196 分钟。

    POSIX：子进程用 start_new_session=True 起，自成一个进程组，killpg 一刀到底。
    Windows：没有进程组语义可用（spawn 池不是我们创建的，拿不到 Job Object），
    taskkill /F /T 是唯一能沿父子关系整棵杀的手段，而且**必须趁直接子进程还活着**
    调用 —— 它一死后代就成孤儿，/T 再也沿不到。
    """
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass                     # 组没了或拿不到，退回单进程
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:                                       # noqa: BLE001
            pass
        return
    try:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                       capture_output=True, timeout=60, check=False)
    except Exception:                                           # noqa: BLE001
        try:
            os.kill(pid, 9)
        except Exception:                                       # noqa: BLE001
            pass


def _run_job(runner, payload, *, sheet, timeout, dbg, label):
    """Run one sidecar entry point and kill its complete process tree on timeout."""
    # **不能用 subprocess.run(timeout=)**。它超时后只 kill 直接子进程
    # （run.py），而 cpu_budget>=2 时引擎开的 multiprocessing spawn 孙进程
    # 继承了 stdout/stderr 的写端 —— 直接子进程死了、管道却还被孙进程握着，
    # 内部的 communicate() 于是无限阻塞：超时保护把自己挂死。实测 bristol
    # P24 在「30 分钟超时」下卡了 10.5 小时，孙进程满核空转 196 分钟，
    # 另有一批孤儿活了 21.7 小时。
    #
    # 正确做法：超时后**先**整棵树一起杀，再回收管道。顺序不能反 —— 先杀
    # 直接子进程会把孙进程变成孤儿，taskkill /T 就沿不到它们了。
    limit = int(timeout if timeout is not None else TIMEOUT)
    # start_new_session：POSIX 上让子进程自成进程组，_kill_tree 才能 killpg
    # 一刀连孙进程一起杀。**不能省** —— 不建新组的话 killpg 会把 webapp 自己
    # 也杀掉（同组）。Windows 上这个参数无效，那边走 taskkill /T。
    proc = subprocess.Popen(
        [str(_PYTHON), "-B", str(runner)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, encoding="utf-8",
        errors="replace",
        start_new_session=(os.name != "nt"),
        # 边车自己会加 sys.path，别把宿主的 site-packages 混进去：
        # 宿主的 PyMuPDF 1.27.2.3 会被引擎的版本闸拒掉。
        env={**os.environ, "PYTHONPATH": "", "PYTHONUTF8": "1"},
    )
    try:
        stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=limit)
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try:
            # 树杀干净了管道就会关，这一步应当立刻返回；给一点余量兜底。
            stdout, stderr = proc.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = "", ""
        raise RuntimeError(
            f"{label} timeout after {limit}s (sheet {sheet})")

    class _Done:
        pass

    result = _Done()
    result.stdout = stdout or ""
    result.stderr = stderr or ""
    result.returncode = proc.returncode
    proc = result

    if proc.stdout:
        try:
            parsed = json.loads(proc.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{label} bad output: {error}; "
                f"stdout={proc.stdout[:300]!r} "
                f"stderr={proc.stderr.strip()[:300]!r}") from error
        if parsed.get("ok"):
            if dbg:
                dbg.note(f"{label}: " + json.dumps(parsed.get("page", {})))
            return parsed
        raise RuntimeError(
            f"{label} {parsed.get('code')}: {parsed.get('error')}")

    raise RuntimeError(
        f"{label} exit {proc.returncode} with no output: "
        f"{proc.stderr.strip()[:400]}")


def run_page(pdf_path, sheet, targets, *, top_k=None, cpu_budget=None,
             timeout=None, dbg=None):
    """跑一页。sheet 是 **1-based**（引擎 API 就是 1-based，别传 page_index）.

    targets = [{"key": union_index | "s<i>:<j>", "ti": int, "tip": [y, x]}, ...]
    返回边车的完整 dict；任何失败抛 RuntimeError。
    """
    if not sidecar_available():
        raise RuntimeError(
            f"linetype sidecar missing: python={_PYTHON} "
            f"exists={_PYTHON.is_file()} runner={_RUNNER} "
            f"exists={_RUNNER.is_file()} engine={_ENGINE_DIR} "
            f"exists={_ENGINE_DIR.is_dir()}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        raise ValueError(f"sheet must be a 1-based int, got {sheet!r}")

    payload = {
        "pdf": str(pdf_path),
        "sheet": int(sheet),
        "targets": [{"key": str(row["key"]), "ti": int(row.get("ti") or 0),
                     "tip": [float(row["tip"][0]), float(row["tip"][1])],
                     # 这个 callout 自己的引线 + 箭头笔画。边车要先把它们对应的
                     # op 剔掉再比距离，否则会把重复的箭头本身认成"线型"。
                     "own": [list(line) for line in (row.get("own") or ())],
                     # 无引线 symbol 的协议字段必须原样越过进程边界。旧白名单
                     # 丢掉这两项后，run.py 会把中心当普通箭头，稳定吸回 marker。
                     "anchor_kind": row.get("anchor_kind"),
                     "exclude_box": (list(row["exclude_box"])
                                     if isinstance(row.get("exclude_box"),
                                                   (list, tuple))
                                     else row.get("exclude_box"))}
                    for row in targets or ()],
        "top_k": int(top_k if top_k is not None else TOP_K),
        "cpu_budget": int(cpu_budget if cpu_budget is not None else CPU_BUDGET),
    }
    if not payload["targets"]:
        raise ValueError("no targets: caller must skip pages with no anchors")
    return _run_job(
        _RUNNER, payload, sheet=sheet, timeout=timeout, dbg=dbg,
        label="linetype sidecar")


def run_all_page(pdf_path, sheet, *, cpu_budget=None, timeout=None,
                 residual=True, dbg=None):
    """Re-run one page and return geometry for every recognized line type.

    This is deliberately separate from :func:`run_page`: callers must verify
    every emitted operation-set fingerprint against the current main cache
    before publishing it.
    """
    if not all_sidecar_available():
        raise RuntimeError(
            f"all-line-type sidecar missing: python={_PYTHON} "
            f"exists={_PYTHON.is_file()} runner={_ALL_RUNNER} "
            f"exists={_ALL_RUNNER.is_file()} engine={_ENGINE_DIR} "
            f"exists={_ENGINE_DIR.is_dir()}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        raise ValueError(f"sheet must be a 1-based int, got {sheet!r}")
    payload = {
        "pdf": str(pdf_path),
        "sheet": int(sheet),
        "cpu_budget": int(
            cpu_budget if cpu_budget is not None else CPU_BUDGET),
        "residual": bool(residual),
    }
    return _run_job(
        _ALL_RUNNER, payload, sheet=sheet, timeout=timeout, dbg=dbg,
        label="all-line-type sidecar")
