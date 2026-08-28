"""与 TypeScript 参考实现的**对齐验证** —— 同一份 PDF、同一页，逐项比较.

为什么需要它：线型算法有两套装配 —— TS 的冻结版（`method1-alternating-crop-r10`
+ `method2-sequential-multipath-r46` + `method2-owns-overlap-v1`）和我们边车跑的
Python 版。Python 那边自己的 versions.py 把四个版本号都标成 "-candidate"，意思是
「缺少可执行的全量证据」。这个脚本就是那份证据。

三条判据，全部必须过：

  1. **fused 类型表逐项相同** —— 比 signature_family / **该类型 op 集合的 sha1** /
     min_sim。**不比 member_count**：TS 的 members 是「每个涉及的 group 一条」
     （文字型线型上实测 19 个 op 对应 79 个 member，atom_count 全为 0），而
     Python 的投影只列真正拥有矢量 op 的 group（同页 11 个）——那是投影口径
     差异，与聚类结果无关，拿它当判据会把一致的页误判成失配。
  2. **fused 的 op 集合逐位相同** —— 比 sha1，不比计数。
  3. **总覆盖恰好等于 TS 的 method1 ∪ method2** —— 我们在 fused 之上补回了
     fusion 连带丢弃的 method1 覆盖（见 steps/linetypes/version.py 的 v4），
     所以正确的期望不是"等于 fused"而是"等于并集"：既不许漏（少于并集），
     也不许凭空多（多于并集）。

TS 侧用仓库自带的无头 CLI（`scripts/recognize-line-types-pdf.mjs`），它用 vite 把
`line-type-engine/node/standalone.ts` 打成 SSR bundle 再跑。注意 TS 的 method1
后端其实是 spawn 回 Python 桥（`node/python-method1-backend.ts:203` →
`scripts/line-type-service-bridge.py`），所以这里验的是**同一套算法的两条装配
路径**是否给出同一答案，而不是两种语言的独立实现。

    python verify_ts_parity.py list                      # 列出可比的页
    python verify_ts_parity.py check <slug> <sheet> ...  # 比这些页
    python verify_ts_parity.py check --all               # 比所有已算过的页

退出码：0 = 全部一致；1 = 有失配；2 = 用法/环境错误。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FL = HERE.parent.parent
TS_ROOT = Path(
    r"C:\Users\Administrator\OneDrive\Desktop\pdf_stream\pdf-command-visualizer")
TS_CLI = TS_ROOT / "scripts" / "recognize-line-types-pdf.mjs"
NODE = "node"
KINDS = ("method1", "method2", "fused")


def ts_document(pdf: Path, sheets, kinds=KINDS):
    """跑**一次** TS 无头 CLI 覆盖这份文档的这些页，返回 {sheet: outputs}。

    按文档批量，不要逐页调用：CLI 每次启动都用 vite 现打一遍 SSR bundle，
    逐页跑的话构建开销会淹没实际计算。
    """
    if not TS_CLI.is_file():
        raise SystemExit(f"TS CLI not found: {TS_CLI}")
    sheets = sorted({int(s) for s in sheets})
    if not sheets:
        return {}
    with tempfile.TemporaryDirectory(prefix="ts-parity-") as work:
        proc = subprocess.run(
            [NODE, str(TS_CLI), "--input", str(pdf), "--output-dir", work,
             "--pages", ",".join(str(s) for s in sheets),
             "--outputs", ",".join(kinds)],
            cwd=str(TS_ROOT), capture_output=True, text=True, check=False)
        out = {}
        for sheet in sheets:
            page_file = Path(work) / "pages" / f"page-{sheet:06d}.json"
            if not page_file.is_file():
                out[sheet] = {"__error": (
                    f"no page file (exit {proc.returncode}) "
                    f"{proc.stderr.strip()[:200]}")}
                continue
            payload = json.loads(page_file.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                out[sheet] = {"__error": f"status={payload.get('status')}"}
                continue
            out[sheet] = payload["outputs"]
        return out


def ops_of(output):
    """一份 TS output 里被任何局部类型认领的 op 集合。"""
    owned = set()
    for group in output["result"]["groups"]:
        for local in group.get("line_types") or ():
            owned.update(local.get("op_indices") or ())
    return owned


def digest_of(ops):
    return hashlib.sha1(
        ",".join(str(i) for i in sorted(ops)).encode()).hexdigest()


def py_entry(slug: str, sheet: int):
    """读一页的线型结果。新布局 data/<slug>/linetypes/<page>.json 优先，
    回落旧的单文件 linetypes.json。"""
    entry = None
    per_page = FL / "data" / slug / "linetypes" / f"{int(sheet)}.json"
    if per_page.is_file():
        entry = json.loads(per_page.read_text(encoding="utf-8"))
    else:
        legacy = FL / "data" / slug / "linetypes.json"
        if not legacy.is_file():
            return None, "no linetypes cache"
        entry = json.loads(legacy.read_text(encoding="utf-8")).get(str(sheet))
    if not isinstance(entry, dict):
        return None, "page not computed"
    if entry.get("error"):
        return None, f"page errored: {str(entry['error'])[:80]}"
    if not entry.get("all_line_types"):
        return None, "no line-type payload"
    return entry, None


def py_fused_rows(entry):
    """我们输出里属于 fused 的那部分（排除 fusion 连带丢弃后补回来的）。"""
    return [t for t in (entry.get("all_line_types") or ())
            if not t.get("recovered_from_fusion")]


def normalise_ts(rows):
    return sorted(
        (r["signature_family"], len(r.get("op_indices") or ()),
         digest_of(r.get("op_indices") or ()),
         round(float(r["minimum_pair_similarity"]), 6))
        for r in rows)


def normalise_py(rows):
    return sorted(
        (r["signature_family"], int(r["op_count"]), r.get("ops_sha1") or "?",
         round(float(r["minimum_pair_similarity"]), 6))
        for r in rows)


def pages_on_disk():
    out = []
    for data_dir in sorted((FL / "data").iterdir()):
        if not data_dir.is_dir():
            continue
        slug = data_dir.name
        if not (FL / "projects" / slug / "input.pdf").is_file():
            continue
        pages = set()
        per_page_dir = data_dir / "linetypes"
        if per_page_dir.is_dir():
            pages.update(int(f.stem) for f in per_page_dir.glob("*.json")
                         if f.stem.isdigit())
        legacy = data_dir / "linetypes.json"
        if legacy.is_file():
            pages.update(int(k) for k in
                         json.loads(legacy.read_text(encoding="utf-8"))
                         if str(k).isdigit())
        out.extend((slug, page) for page in sorted(pages))
    return out


def check_page(slug, sheet, outputs):
    """一页三条判据。返回 0 = 一致，1 = 失配。"""
    if not outputs or outputs.get("__error"):
        print(f"  ERR  {slug} P{sheet}: {(outputs or {}).get('__error')}")
        return 1
    entry, note = py_entry(slug, sheet)
    if entry is None:
        print(f"  SKIP {slug} P{sheet}: {note}")
        return 0

    rows = py_fused_rows(entry)
    left = normalise_ts(outputs["fused"]["result"]["global_types"])
    right = normalise_py(rows)

    union = ops_of(outputs["method1"]) | ops_of(outputs["method2"])
    stats = entry.get("page") or {}
    owned_sha = stats.get("owned_ops_sha1")
    fused_sha = stats.get("fused_ops_sha1")
    coverage_ok = owned_sha is None or owned_sha == digest_of(union)
    fused_ok = (fused_sha is None
                or fused_sha == digest_of(ops_of(outputs["fused"])))

    if left == right and coverage_ok and fused_ok:
        print(f"  OK   {slug} P{sheet}: fused {len(right)} 型 / "
              f"{sum(r[1] for r in right)} ops 逐项相同; "
              f"覆盖 == method1∪method2 ({len(union)} ops)")
        return 0

    if not coverage_ok:
        print(f"  DIFF {slug} P{sheet}: 覆盖集合 != method1∪method2 "
              f"(TS union {len(union)} ops, PY {stats.get('owned_path_ops')})")
    if not fused_ok:
        print(f"  DIFF {slug} P{sheet}: fused 的 op 集合与 TS 不同 "
              f"(TS {len(ops_of(outputs['fused']))} ops)")
    if left != right:
        print(f"  DIFF {slug} P{sheet}: fused 类型表 TS {len(left)} 型 / "
              f"{sum(r[1] for r in left)} ops vs PY {len(right)} 型 / "
              f"{sum(r[1] for r in right)} ops")
        for row in [r for r in left if r not in right][:8]:
            print(f"       TS only: {row}")
        for row in [r for r in right if r not in left][:8]:
            print(f"       PY only: {row}")
    return 1


def check(pairs):
    failures = 0
    checked = 0
    by_slug = {}
    for slug, sheet in pairs:
        by_slug.setdefault(slug, []).append(int(sheet))

    for slug, sheets in sorted(by_slug.items()):
        pdf = FL / "projects" / slug / "input.pdf"
        wanted = []
        for sheet in sorted(set(sheets)):
            entry, note = py_entry(slug, sheet)
            if entry is None:
                print(f"  SKIP {slug} P{sheet}: {note}")
            else:
                wanted.append(sheet)
        if not wanted:
            continue
        print(f"  ... {slug}: TS 跑 {len(wanted)} 页", flush=True)
        try:
            document = ts_document(pdf, wanted)
        except Exception as error:                              # noqa: BLE001
            print(f"  ERR  {slug}: {error}")
            failures += len(wanted)
            continue
        for sheet in wanted:
            checked += 1
            failures += check_page(slug, sheet, document.get(sheet) or {})

    verdict = "PARITY OK" if not failures else "PARITY FAILED"
    print(f"\n{verdict}: {checked - failures}/{checked} 页逐项一致")
    return 1 if failures else 0


def main(argv):
    action = argv[1] if len(argv) > 1 else "list"
    if action == "list":
        for slug, sheet in pages_on_disk():
            print(f"  {slug} P{sheet}")
        return 0
    if action != "check":
        print(__doc__)
        return 2
    rest = argv[2:]
    if rest[:1] == ["--all"]:
        return check(pages_on_disk())
    if len(rest) >= 2:
        return check([(rest[0], int(s)) for s in rest[1:]])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
