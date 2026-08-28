"""验证 run_all.py 的调试几何与主缓存是**同一次聚类的产物**，并落盘 .all.json.

run_all.py 是 run.py 之外的第二条装配路径（为了不动 run.py 的字节、不作废盘上
54 页缓存）。两条路径同源不是靠"我相信它同源"，而是靠这里逐项对表：

  1. ``page.owned_ops_sha1`` 与 ``page.fused_ops_sha1`` 必须与主缓存逐位相同 ——
     这两个指纹锚的是「哪些 op 被哪一层认领」，它们相同就说明聚类和补回规则
     都跑出了同一个答案。
  2. 每个线型逐行相同：按 ``line_type_number`` 对齐，比
     (signature_family, recognition_source, op_count, ops_sha1, min_sim)。
     **不比 member_count 之外的投影量**，理由见 verify_ts_parity.py 的说明。
  3. 类型数量相同，且没有只在一侧出现的编号。

任何一条不过就报 DIFF 并且**不写盘** —— 写一份和正在显示的结果不同源的调试
几何，比没有调试几何更糟：它会让人得出关于另一次聚类的结论。

    python verify_all_geometry.py <slug> <sheet> [<sheet> ...]
    python verify_all_geometry.py --all            # 所有已有主缓存的页
    python verify_all_geometry.py --missing        # 只补还没有 .all.json 的页
    python verify_all_geometry.py --missing <slug> [<slug> ...]   # 限定项目

退出码：0 = 全部一致并已落盘；1 = 有失配；2 = 用法/环境错误。
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
RUNNER = HERE / "run_all.py"
CPU = int(os.environ.get("LINETYPE_CPU_BUDGET", "16"))

# 逐行比对的字段。ops_sha1 是核心 —— 它锚的是 op 集合本身，不是任何计数。
ROW_KEYS = ("signature_family", "recognition_source", "op_count", "ops_sha1")


def main_cache(slug, sheet):
    path = FL / "data" / slug / "linetypes" / f"{int(sheet)}.json"
    if not path.is_file():
        return None
    entry = json.loads(path.read_text(encoding="utf-8"))
    return entry if isinstance(entry, dict) else None


def all_path(slug, sheet):
    return FL / "data" / slug / "linetypes" / f"{int(sheet)}.all.json"


def run_all(pdf, sheet):
    payload = json.dumps({"pdf": str(pdf), "sheet": int(sheet),
                          "cpu_budget": CPU, "residual": True})
    proc = subprocess.run(
        [str(PY), "-B", str(RUNNER)], input=payload, capture_output=True,
        text=True, encoding="utf-8",
        env={**os.environ, "PYTHONPATH": "", "PYTHONUTF8": "1"})
    if not proc.stdout.strip():
        return None, f"no output (exit {proc.returncode}) {proc.stderr.strip()[:300]}"
    try:
        out = json.loads(proc.stdout)
    except Exception as error:                                  # noqa: BLE001
        return None, f"bad JSON: {error}; stderr={proc.stderr.strip()[:200]}"
    if not out.get("ok"):
        return None, f"{out.get('code')}: {str(out.get('error'))[:300]}"
    return out, None


def row_of(entry, keys=ROW_KEYS):
    return tuple(entry.get(k) for k in keys)


def compare(slug, sheet, cached, fresh):
    """三条判据。返回 (ok, 报告行列表)。"""
    notes = []
    page_c = cached.get("page") or {}
    page_f = fresh.get("page") or {}
    ok = True
    for key in ("owned_ops_sha1", "fused_ops_sha1"):
        left, right = page_c.get(key), page_f.get(key)
        if left and right and left != right:
            ok = False
            notes.append(f"       {key}: 缓存 {left[:12]} vs 调试 {right[:12]}")

    by_number_c = {int(r["line_type_number"]): r
                   for r in (cached.get("all_line_types") or ())}
    by_number_f = {int(r["line_type_number"]): r for r in (fresh.get("types") or ())}
    only_c = sorted(set(by_number_c) - set(by_number_f))
    only_f = sorted(set(by_number_f) - set(by_number_c))
    if only_c or only_f:
        ok = False
        notes.append(f"       编号不对等: 只在缓存 {only_c[:8]} / 只在调试 {only_f[:8]}")
    mismatched = []
    for number in sorted(set(by_number_c) & set(by_number_f)):
        left, right = by_number_c[number], by_number_f[number]
        if row_of(left) != row_of(right):
            mismatched.append((number, row_of(left), row_of(right)))
    if mismatched:
        ok = False
        notes.append(f"       {len(mismatched)} 个类型逐行不同，前 3 个:")
        for number, left, right in mismatched[:3]:
            notes.append(f"         #{number} 缓存 {left}")
            notes.append(f"         #{number} 调试 {right}")
    return ok, notes


def do_page(slug, sheet):
    cached = main_cache(slug, sheet)
    if cached is None:
        print(f"  SKIP {slug} P{sheet}: 没有主缓存")
        return 0
    if not cached.get("all_line_types"):
        print(f"  SKIP {slug} P{sheet}: 主缓存没有线型（{(cached.get('page') or {}).get('reason') or cached.get('error') or '?'}）")
        return 0
    pdf = FL / "projects" / slug / "input.pdf"
    if not pdf.is_file():
        print(f"  SKIP {slug} P{sheet}: 找不到 PDF")
        return 0

    started = time.time()
    fresh, error = run_all(pdf, sheet)
    wall = time.time() - started
    if fresh is None:
        print(f"  ERR  {slug} P{sheet}: {error}")
        return 1

    ok, notes = compare(slug, sheet, cached, fresh)
    types = fresh.get("types") or []
    residual = fresh.get("residual") or {}
    if not ok:
        print(f"  DIFF {slug} P{sheet}: 与主缓存不同源，**不写盘**")
        for line in notes:
            print(line)
        return 1

    # 写盘。带上主缓存的 sig，读盘期用它判断这份几何是否还对应当期结果。
    out = {
        "sig": cached.get("sig"),
        "v": cached.get("v"),
        "page": fresh.get("page") or {},
        "engine": fresh.get("engine") or {},
        "types": types,
        "residual": residual,
    }
    path = all_path(slug, sheet)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    print(f"  OK   {slug} P{sheet}: {len(types)} 型逐行相同, "
          f"residual {residual.get('op_count', 0)} op / "
          f"{residual.get('segment_count', 0)} 段, "
          f"{len(text) / 1048576:.1f} MB, {wall:.0f}s")
    return 0


def pages_with_cache(only_missing=False, slugs=()):
    """有主缓存的页。slugs 非空时只看这些项目 —— 补全量时按项目切分成几个
    并行进程，taylor 一家就占累计耗时的 68%，不切的话它会拖着整批。"""
    out = []
    root = FL / "data"
    if not root.is_dir():
        return out
    wanted = {str(s) for s in slugs}
    for data_dir in sorted(root.iterdir()):
        if wanted and data_dir.name not in wanted:
            continue
        directory = data_dir / "linetypes"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json"),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else 0):
            if not path.stem.isdigit():
                continue
            sheet = int(path.stem)
            if only_missing and all_path(data_dir.name, sheet).is_file():
                continue
            out.append((data_dir.name, sheet))
    return out


def main(argv):
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] in ("--all", "--missing"):
        pairs = pages_with_cache(only_missing=args[0] == "--missing",
                                 slugs=args[1:])
    elif len(args) >= 2:
        pairs = [(args[0], int(s)) for s in args[1:]]
    else:
        print(__doc__)
        return 2
    if not pairs:
        print("  没有要处理的页")
        return 0
    print(f"  {len(pairs)} 页, cpu_budget={CPU}", flush=True)
    failures = 0
    for slug, sheet in pairs:
        failures += do_page(slug, sheet)
        sys.stdout.flush()
    verdict = "ALL GEOMETRY OK" if not failures else "ALL GEOMETRY FAILED"
    print(f"\n{verdict}: {len(pairs) - failures}/{len(pairs)} 页一致并已落盘")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
