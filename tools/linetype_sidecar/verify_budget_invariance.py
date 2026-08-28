"""证明 cpu_budget 不改变线型聚类结果 —— 或者证明它会改.

为什么这件事必须先证：

  * ``sidecar.py:85`` 把 ``cpu_budget`` 拌进 ``engine_digest()``，而 digest 进缓存
    签名。于是「按机器核数决定 budget」会让**换一台机器 = 全部缓存作废**，
    正好与「换到弱机器也能正常用」相反。
  * 更要紧的是：引擎里 ``budget = min(cpu_budget or available, available)``
    （scheduling.py:84-85）—— **budget 本来就被机器并行度截断**。所以如果 budget
    真的影响结果，那今天盘上的缓存**已经是跟机器绑定的，而签名里写的 16 是假的**。
  * 反过来，一旦证明不变，就可以把它从签名里拿掉，budget 才能随机器自由调。

测哪几个取值不是随手挑的，按 ``plan_single_page_execution`` 的分支挑（本机 32 核）：

    budget=1   m1=1  m2=1  concurrent=False  ← 唯一的非并发计划；method2 单 worker
    budget=2   m1=1  m2=1  concurrent=True   ← 同 worker 数但并发
    budget=8   m1=6  m2=2  concurrent=True   ← **method2 多 worker**：run.py 的
                                               docstring 点名的风险路径（worker>=2
                                               先物化全部 candidate_lists 再按批
                                               speculative 求值，==1 是边算边 union）
    budget=16  m1=12 m2=4                    ← 当前默认
    budget=32  m1=24 m2=8                    ← method2 顶到 cap

比什么：整份输出，**只忽略三个与结果无关的字段** —— ``page.seconds_ir`` /
``page.seconds_cluster``（计时）与 ``engine.cpu_budget``（就是自变量本身）。
其余一律逐位比：每个类型的 ops_sha1 / op_count / min_sim / bbox / 段数，
页级的 owned_ops_sha1 / fused_ops_sha1 / page_fingerprint，以及每个末端的
最近 op / 距离 / 候选排序。任何一处不同都算「budget 会改变结果」。

    python verify_budget_invariance.py                 # 默认页集
    python verify_budget_invariance.py <slug> <page> ...
    python verify_budget_invariance.py --budgets 1,8,16

退出码：0 = 全部不变；1 = 存在差异；2 = 用法/环境错误。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent.parent
PY = HERE / "venv" / "Scripts" / "python.exe"
RUNNER = HERE / "run.py"

# 刻意挑的：都真的产出了 method2 类型（否则 method2 的 worker 分支根本不执行），
# 并且组数从 55 到 7854 跨三个数量级（method1 的 worker 并行度吃组数）。
DEFAULT_PAGES = [
    ("vallivue_academy", 5),                    # 1137 组, 18 个 method2 类型, 14 个补回
    ("taylor_3_12", 7),                         # 164 组, 14 个 method2
    ("wasd_pollard", 5),                        # 606 组, 38834 op
    ("grand_island_casino", 250),               # 287 组, 81 个类型
    ("civil_ifb_167263", 2),                    # 492 组, 22 个补回
    ("fence_report_paducah_ky", 164),           # 7854 组 —— method1 并行度的极端
]
DEFAULT_BUDGETS = [1, 2, 8, 16, 32]

# 与结果无关、必须忽略的字段（路径用 / 分隔）
IGNORE = {"page/seconds_ir", "page/seconds_cluster", "engine/cpu_budget"}


def anchors_of(slug, sheet):
    arrows = json.loads(
        (FL / "data" / slug / "arrows.json").read_text(encoding="utf-8"))
    entry = arrows.get(str(sheet)) or {}
    out = []
    for key, item in (entry.get("items") or {}).items():
        own = [list(line) for line in
               list(item.get("leader_strokes") or ())
               + list(item.get("arrow_strokes") or ())
               if isinstance(line, (list, tuple)) and len(line) >= 2]
        for index, target in enumerate(item.get("targets") or ()):
            tip = target.get("tip")
            if isinstance(tip, (list, tuple)) and len(tip) >= 2:
                out.append({"key": str(key), "ti": index,
                            "tip": [float(tip[0]), float(tip[1])],
                            "own": own})
    return out


def run_one(slug, sheet, budget):
    job = {"pdf": str(FL / "projects" / slug / "input.pdf"),
           "sheet": int(sheet), "targets": anchors_of(slug, sheet),
           "top_k": 3, "cpu_budget": int(budget)}
    started = time.time()
    proc = subprocess.run(
        [str(PY), "-B", str(RUNNER)], input=json.dumps(job),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env={**os.environ, "PYTHONPATH": "", "PYTHONUTF8": "1"})
    wall = time.time() - started
    if not proc.stdout.strip():
        return None, wall, f"no output (exit {proc.returncode}) {proc.stderr[:200]}"
    payload = json.loads(proc.stdout)
    if not payload.get("ok"):
        return None, wall, f"{payload.get('code')}: {str(payload.get('error'))[:200]}"
    return payload, wall, None


def diffs(left, right, path=""):
    """逐字段比，返回差异路径列表。IGNORE 里的路径跳过。"""
    if path in IGNORE:
        return []
    if type(left) is not type(right):
        return [f"{path}: 类型 {type(left).__name__} vs {type(right).__name__}"]
    if isinstance(left, dict):
        out = []
        for key in sorted(set(left) | set(right)):
            child = f"{path}/{key}" if path else key
            if key not in left:
                out.append(f"{child}: 只在 B 有")
            elif key not in right:
                out.append(f"{child}: 只在 A 有")
            else:
                out.extend(diffs(left[key], right[key], child))
        return out
    if isinstance(left, list):
        if len(left) != len(right):
            return [f"{path}: 长度 {len(left)} vs {len(right)}"]
        out = []
        for index, (a, b) in enumerate(zip(left, right)):
            out.extend(diffs(a, b, f"{path}[{index}]"))
        return out
    if isinstance(left, float) and isinstance(right, float):
        # 浮点也要求逐位相同：这套结果本来就是确定性的，容差会掩盖真差异
        return [] if left == right else [f"{path}: {left!r} vs {right!r}"]
    return [] if left == right else [f"{path}: {left!r} vs {right!r}"]


def main(argv):
    args = argv[1:]
    budgets = DEFAULT_BUDGETS
    if "--budgets" in args:
        index = args.index("--budgets")
        budgets = [int(x) for x in args[index + 1].split(",")]
        del args[index:index + 2]
    pages = DEFAULT_PAGES
    if len(args) >= 2:
        pages = [(args[0], int(p)) for p in args[1:]]

    print(f"  {len(pages)} 页 x {len(budgets)} 个 budget = "
          f"{len(pages) * len(budgets)} 次运行；基准 budget={budgets[0]}", flush=True)
    failures = 0
    for slug, sheet in pages:
        if not (FL / "projects" / slug / "input.pdf").is_file():
            print(f"  SKIP {slug} P{sheet}: 没有 PDF")
            continue
        if not (FL / "data" / slug / "arrows.json").is_file():
            print(f"  SKIP {slug} P{sheet}: 没有 arrows.json")
            continue
        print(f"\n  == {slug} P{sheet}", flush=True)
        base = None
        base_budget = budgets[0]
        for budget in budgets:
            payload, wall, error = run_one(slug, sheet, budget)
            if payload is None:
                print(f"     ERR  budget={budget}: {error}", flush=True)
                failures += 1
                continue
            page = payload.get("page") or {}
            stamp = (f"型={page.get('line_types')} owned={page.get('owned_path_ops')} "
                     f"sha={str(page.get('owned_ops_sha1'))[:10]}")
            if base is None:
                base = payload
                print(f"     基准 budget={budget:<3} {wall:6.1f}s  {stamp}", flush=True)
                continue
            found = diffs(base, payload)
            if found:
                failures += 1
                print(f"     **DIFF budget={budget} vs {base_budget}: "
                      f"{len(found)} 处** {wall:6.1f}s  {stamp}", flush=True)
                for line in found[:10]:
                    print(f"        {line}", flush=True)
            else:
                print(f"     OK   budget={budget:<3} {wall:6.1f}s  逐位相同",
                      flush=True)

    verdict = ("BUDGET INVARIANT: 所有 budget 给出逐位相同的结果"
               if not failures else
               f"BUDGET NOT INVARIANT: {failures} 处差异/错误")
    print(f"\n{verdict}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
