"""线型引擎的**结果等价性**台账 —— 改算法之前后各跑一次，逐字段说差异.

为什么需要它：算法模块是允许改的，但「改完结果没问题」必须能证明，不能靠看图。
这个脚本把边车在一组固定页上的输出冻结成基线快照，改完再跑一次对比，**逐字段**
报告差异，而不是给一句「不一样」。

它刻意复用生产路径：锚点直接取自 ``data/<slug>/arrows.json`` 的真实末端，边车、
解释器、vendored 引擎都和跑作业时是同一套。基线因此就是生产会看到的东西。

对比时忽略的字段（只有这些，其余一律逐值比较）：
    page.seconds_ir / page.seconds_cluster     机器负载相关，天然波动

**cpu_budget 不变性**是单独一条要验的性质：语料工具和服务用的预算可能不同，
如果结果随预算变，那所有快照都失去意义。用 ``--cpu`` 跑两次再 compare 即可
（``snapshot --cpu 1`` 然后 ``compare --cpu 8``）。

用法：
    python verify_engine.py cases                       # 列出会跑哪些页
    python verify_engine.py snapshot [--cpu N] [--out baseline.json]
    python verify_engine.py compare  [--cpu N] [--baseline baseline.json]

退出码：0 = 完全一致；1 = 有差异；2 = 用法/环境错误。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent.parent
RUNNER = HERE / "run.py"
PYTHON = HERE / "venv" / "Scripts" / "python.exe"
DEFAULT_BASELINE = HERE / "baseline.json"

# 固定用例。选取理由写在旁边 —— 换用例等于换基线，别随手改。
CASES = [
    # /Rotate 270 的平面图：转帧最容易错的一类，且末端实测能打到 0.00
    ("gladstone_dog_park", 2),
    # 详图页：12 个视图框、15 个末端全在 section/elevation 里（plan 闸的反例）
    ("gladstone_dog_park", 8),
    # 图例样例密集的一页
    ("drawings_volume_4_binder", 4),
]

IGNORED = {("page", "seconds_ir"), ("page", "seconds_cluster")}


def anchors_for(slug, sheet):
    path = FL / "data" / slug / "arrows.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text(encoding="utf-8")).get(str(sheet))
    if not isinstance(entry, dict):
        return None
    out = []
    for key, item in (entry.get("items") or {}).items():
        # own 必须带上，否则边车不会剔掉这个 callout 自己的引线 / 箭头，
        # 结果就和生产路径不一致 —— 基线必须是生产会看到的东西。
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


def run_case(slug, sheet, cpu_budget):
    pdf = FL / "projects" / slug / "input.pdf"
    anchors = anchors_for(slug, sheet)
    if not pdf.is_file():
        return {"skipped": f"no pdf at {pdf}"}
    if anchors is None:
        return {"skipped": "no arrows.json entry for this sheet"}
    if not anchors:
        return {"skipped": "no arrow terminals on this sheet"}
    job = {"pdf": str(pdf), "sheet": int(sheet), "targets": anchors,
           "top_k": 3, "cpu_budget": int(cpu_budget)}
    # 生产用的是 cpu_budget=1（见 steps/linetypes/sidecar.py 的说明）。
    # 用别的值跑 compare 就同时在验 budget 不变性 —— 那是有意的，见 --cpu。
    proc = subprocess.run([str(PYTHON), "-B", str(RUNNER)],
                          input=json.dumps(job), capture_output=True,
                          text=True, check=False)
    if not proc.stdout:
        return {"failed": f"exit {proc.returncode}: {proc.stderr.strip()[:400]}"}
    try:
        parsed = json.loads(proc.stdout)
    except json.JSONDecodeError as error:
        return {"failed": f"bad output: {error}"}
    if not parsed.get("ok"):
        return {"failed": f"{parsed.get('code')}: {parsed.get('error')}"}
    return parsed


def canonical(payload):
    """只留下**该稳定**的部分，顺序固定，可直接逐值比较。"""
    if "skipped" in payload or "failed" in payload:
        return dict(payload)
    page = dict(payload.get("page") or {})
    for _section, field in IGNORED:
        page.pop(field, None)
    return {
        "engine": payload.get("engine"),
        "page": page,
        "all_line_types": payload.get("all_line_types"),
        "line_types": [
            {key: value for key, value in row.items()}
            for row in payload.get("line_types") or ()
        ],
        "bindings": payload.get("bindings"),
    }


def walk(node, path=""):
    """把嵌套结构摊平成 {路径: 值}，这样差异能定位到具体字段。"""
    if isinstance(node, dict):
        for key in sorted(node):
            yield from walk(node[key], f"{path}.{key}")
    elif isinstance(node, list):
        yield f"{path}#len", len(node)
        for index, value in enumerate(node):
            yield from walk(value, f"{path}[{index}]")
    else:
        yield path or ".", node


def diff_case(name, left, right):
    left_flat = dict(walk(left))
    right_flat = dict(walk(right))
    keys = sorted(set(left_flat) | set(right_flat))
    problems = []
    for key in keys:
        if key not in left_flat:
            problems.append(f"    + {key} = {right_flat[key]!r}")
        elif key not in right_flat:
            problems.append(f"    - {key} = {left_flat[key]!r}")
        elif left_flat[key] != right_flat[key]:
            problems.append(f"    ~ {key}: {left_flat[key]!r} -> {right_flat[key]!r}")
    if not problems:
        print(f"  OK   {name}  ({len(left_flat)} fields identical)")
        return 0
    print(f"  DIFF {name}  ({len(problems)} field(s) differ)")
    for line in problems[:40]:
        print(line)
    if len(problems) > 40:
        print(f"    … and {len(problems) - 40} more")
    return 1


def cmd_cases():
    for slug, sheet in CASES:
        anchors = anchors_for(slug, sheet)
        count = "no arrows.json" if anchors is None else str(len(anchors))
        pdf = FL / "projects" / slug / "input.pdf"
        print(f"  {slug} P{sheet}: terminals={count} pdf={'yes' if pdf.is_file() else 'MISSING'}")
    return 0


def collect(cpu_budget):
    out = {}
    for slug, sheet in CASES:
        name = f"{slug}:P{sheet}"
        print(f"  running {name} (cpu_budget={cpu_budget})…", flush=True)
        payload = run_case(slug, sheet, cpu_budget)
        note = payload.get("skipped") or payload.get("failed")
        if note:
            print(f"    {note}")
        out[name] = canonical(payload)
    return out


def cmd_snapshot(cpu_budget, path):
    data = collect(cpu_budget)
    Path(path).write_text(json.dumps(
        {"cpu_budget": cpu_budget, "cases": data},
        ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
    ok = sum(1 for value in data.values()
             if "skipped" not in value and "failed" not in value)
    print(f"snapshot: {ok}/{len(data)} case(s) captured -> {path}")
    return 0


def cmd_compare(cpu_budget, path):
    baseline_file = Path(path)
    if not baseline_file.is_file():
        print(f"no baseline at {baseline_file}; run `snapshot` first")
        return 2
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    recorded = baseline.get("cases") or {}
    if baseline.get("cpu_budget") != cpu_budget:
        print(f"note: baseline cpu_budget={baseline.get('cpu_budget')} vs "
              f"this run {cpu_budget} — this run also tests budget invariance")
    fresh = collect(cpu_budget)
    failures = 0
    for name in sorted(set(recorded) | set(fresh)):
        if name not in recorded:
            print(f"  NEW  {name} (not in baseline)")
            failures += 1
            continue
        if name not in fresh:
            print(f"  GONE {name} (in baseline, not produced now)")
            failures += 1
            continue
        failures += diff_case(name, recorded[name], fresh[name])
    print(f"\n{'IDENTICAL' if not failures else 'DIFFERENCES FOUND'}: "
          f"{failures} case(s) differ of {len(fresh)}")
    return 1 if failures else 0


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",
                        choices=("cases", "snapshot", "compare"))
    parser.add_argument("--cpu", type=int, default=1,
                        help="cpu_budget passed to cluster_page_commands")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv[1:])
    if not PYTHON.is_file():
        print(f"sidecar interpreter missing: {PYTHON}")
        return 2
    if args.action == "cases":
        return cmd_cases()
    if args.action == "snapshot":
        return cmd_snapshot(args.cpu, args.out or args.baseline)
    return cmd_compare(args.cpu, args.baseline)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
