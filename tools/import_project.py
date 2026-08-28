"""把生产 5051（fence_takeoff_web）已经算好的项目搬进 fence_lite —— 零 Gemini 花费.

为什么需要这个工具：所有付费缓存（vlm.json 的整页 VLM raw、symbols.json 的
图例推理 raw）的主键里都混了 ``pdf_revision = f"{size:x}-{mtime_ns:x}"``。
文件一拷贝 mtime 就变，于是内容完全相同的 PDF 也会得到一个新 revision，
所有 sig 全部失配 —— 管线会把已经付过钱的推理再买一遍。

做法与参考项目的 tools_migrate_rev.py 一致：**重算签名而不是重新付费**。
签名一律用 fence_lite 自己的函数算（steps.store.sig_of / steps.views.view_signature），
公式只有一处、天然正确。

用法：
    python -B tools/import_project.py <slug> [<slug> ...] [--from DIR] [--dry-run]

搬什么：
    <REF>/projects/<slug>/input.pdf   → projects/<slug>/input.pdf
    <REF>/fence_fused/<slug>/*.json   → data/<slug>/*.json（vlm_extra.json 改名 vlm.json）

不搬什么（都是已砍掉的东西，或会自动重生的东西）：
    callout_selections.json / fencelines.json   步骤4 fenceline 全套，已砍
    base_P*.jpg                                 前端底图，按新 revision 自动重生

另外会**剥掉**旧 symbols.json 里 propagate 阶段留下的字段（placements / trace /
line_type / …）：那是旧语义（没有 plan 视图过滤、line 类也跑过整线传播），
在 fence_lite 里必须由本地免费的 placements 阶段重算，不能直接端上去给用户看。
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    # 支持 `python -B tools/import_project.py`：此时 sys.path[0] 是 tools/，
    # 不把项目根塞进去 `from steps.x import y` 会直接 ImportError。
    sys.path.insert(0, str(BASE_DIR))

from steps.store import (DATA_DIR, items_of, load_json, pdf_revision,
                         require_slug, save_json, sig_of)
from core.config import PROJECTS_DIR

try:
    from steps.views import view_signature as _view_signature
except Exception:                                               # noqa: BLE001
    # steps/views.py 由并行开发；缺失时把 view_types 的重签名降级为「跳过并提示」，
    # 而不是猜一个公式写死（猜错 = 静默重付分类费用）。
    _view_signature = None

REF_DIR_DEFAULT = Path(r"C:\Users\Administrator\fence_takeoff_web")

# <REF>/fence_fused/<slug>/<源文件名> → data/<slug>/<目标文件名>
CACHE_MAP = {
    "results.json": "results.json",
    "vec.json": "vec.json",
    "textjudge.json": "textjudge.json",
    "vlm_extra.json": "vlm.json",
    "vlm_flash.json": "vlm_flash.json",
    "symbols.json": "symbols.json",
    "view_types.json": "view_types.json",
}
# 没有这个文件就没有任何可看的东西，视为「这个 slug 没算过」。
REQUIRED_SRC = "results.json"
# 明确不搬（列出来是为了让「为什么少了这几个文件」一眼可查）。
SKIP_SRC = ("callout_selections.json", "fencelines.json", "base_P*.jpg")

# 旧 propagate / 整线追踪阶段写在 symbol 上的字段，全是已砍语义。
STALE_SYMBOL_KEYS = ("placements", "trace", "sample_evidence", "vec_error",
                     "line_type", "snapped_box", "vec_note")
STALE_RESULT_KEYS = ("prop_v",)
STALE_ENTRY_KEYS = ("debug",)



def ref_paths(slug, ref_dir):
    """(源 PDF, 源缓存目录)。"""
    ref_dir = Path(ref_dir)
    return (ref_dir / "projects" / slug / "input.pdf",
            ref_dir / "fence_fused" / slug)


def available_slugs(ref_dir):
    """参考目录里算过东西（有 results.json）的 slug，按名字排序。"""
    fused = Path(ref_dir) / "fence_fused"
    if not fused.is_dir():
        return []
    return sorted(d.name for d in fused.iterdir()
                  if d.is_dir() and (d / REQUIRED_SRC).exists())


def _strip_stale_propagation(symbols):
    """剥掉旧语义的放置结果，返回被剥掉的字段计数。

    Old placements came from the retired propagation step: no plan-view gate
    and the line category was swept too.  Keeping them would show the user
    stale boxes before the (free) placements stage ever runs.
    """
    counts = {}
    for page_entry in (symbols or {}).values():
        if not isinstance(page_entry, dict):
            continue
        for key in STALE_ENTRY_KEYS:
            if page_entry.pop(key, None) is not None:
                counts[key] = counts.get(key, 0) + 1
        result = page_entry.get("result")
        if not isinstance(result, dict):
            continue
        for key in STALE_RESULT_KEYS:
            if result.pop(key, None) is not None:
                counts[key] = counts.get(key, 0) + 1
        for symbol in result.get("symbols") or []:
            if not isinstance(symbol, dict):
                continue
            for key in STALE_SYMBOL_KEYS:
                if symbol.pop(key, None) is not None:
                    counts[key] = counts.get(key, 0) + 1
    return counts


def _rewrite_results(results, revision, source_dir=None):
    """results.json 里所有带 pdf_revision 的地方改成新值，并打上「搬来的」标记。

    这个标记不是装饰：步骤1/2 的结果是**旧流水线**花钱跑出来的，虽然算法逐字
    相同（回归里逐字段验过），但它不是这套代码在这台机器上产出的。前端据此
    显示「旧缓存」徽标，用户想确认就点「重新跑」（会先清空缓存）。
    """
    results["pdf_revision"] = revision
    if source_dir is not None:
        results["imported"] = {
            "from": str(source_dir),
            "at": datetime.now().isoformat(timespec="seconds"),
            "note": "步骤1/2 结果来自旧服务的付费缓存；放置结果由本服务本地重算",
        }
    for rec in (results.get("pages") or {}).values():
        if not isinstance(rec, dict):
            continue
        for source in rec.get("vlm_sources") or []:
            identity = source.get("identity") if isinstance(source, dict) else None
            if isinstance(identity, dict):
                identity["pdf_revision"] = revision


def _rewrite_vlm(records, revision):
    """vlm.json / vlm_flash.json 每条 raw 记录的身份换成新 revision。"""
    fixed = 0
    for entry in (records or {}).values():
        if isinstance(entry, dict) and isinstance(entry.get("vlm_identity"), dict):
            entry["vlm_identity"]["pdf_revision"] = revision
            fixed += 1
    return fixed


def _symbol_sigs(symbols, pages, revision):
    """symbols.json 每页的 sig 用新 revision 重算，返回重算页数。"""
    fixed = 0
    for page, entry in (symbols or {}).items():
        rec = (pages or {}).get(str(page))
        if not isinstance(rec, dict) or not isinstance(entry, dict):
            continue
        entry["sig"] = sig_of(items_of(rec), revision)
        fixed += 1
    return fixed


def _view_sigs(view_types, symbols, revision):
    """view_types.json 每页的 sig 重算（含 model 与 VIEW_VERSION）。

    返回重算页数，或 None 表示 steps.views 还没就绪、整步跳过。
    """
    if _view_signature is None:
        return None
    fixed = 0
    for page, entry in (view_types or {}).items():
        sym_entry = (symbols or {}).get(str(page))
        if not isinstance(entry, dict) or not isinstance(sym_entry, dict):
            continue
        raw_groups = (sym_entry.get("result") or {}).get("groups", [])
        entry["sig"] = _view_signature(raw_groups, revision, entry.get("model"))
        fixed += 1
    return fixed


def _tally(results, symbols):
    """自检用的计数：页数 / 文字项 / symbol / placements。"""
    pages = (results or {}).get("pages") or {}
    text_items = sum(len(items_of(rec)) for rec in pages.values()
                     if isinstance(rec, dict))
    sym_count = plc_count = 0
    for entry in (symbols or {}).values():
        for symbol in ((entry or {}).get("result") or {}).get("symbols") or []:
            sym_count += 1
            plc_count += len(symbol.get("placements") or [])
    return {
        "page_count": (results or {}).get("page_count"),
        "pages_with_text": len(pages),
        "text_items": text_items,
        "symbols": sym_count,
        "placements": plc_count,
    }


def _verify(slug, revision, log):
    """从磁盘重读，独立重算一遍签名并比对（真检查，不是回读自己刚写的变量）。"""
    ddir = DATA_DIR / slug
    results = load_json(ddir / "results.json", {}) or {}
    pages = results.get("pages") or {}
    symbols = load_json(ddir / "symbols.json", {}) or {}
    views = load_json(ddir / "view_types.json", {}) or {}
    out = {"results_revision_ok": results.get("pdf_revision") == revision}

    bad = [p for p, e in symbols.items()
           if isinstance(e, dict) and isinstance(pages.get(str(p)), dict)
           and e.get("sig") != sig_of(items_of(pages[str(p)]), revision)]
    out["symbols_sig_ok"] = (not bad) if symbols else None
    if bad:
        log(f"  [warn] symbols sig 失配页: {', '.join(sorted(bad))}")

    if not views:
        out["view_sig_ok"] = None
    elif _view_signature is None:
        out["view_sig_ok"] = None
    else:
        bad_v = []
        for page, entry in views.items():
            sym_entry = symbols.get(str(page))
            if not isinstance(entry, dict) or not isinstance(sym_entry, dict):
                continue
            groups = (sym_entry.get("result") or {}).get("groups", [])
            if entry.get("sig") != _view_signature(groups, revision,
                                                   entry.get("model")):
                bad_v.append(page)
        out["view_sig_ok"] = not bad_v
        if bad_v:
            log(f"  [warn] view_types sig 失配页: {', '.join(sorted(bad_v))}")

    vlm = load_json(ddir / "vlm.json", {}) or {}
    out["vlm_revision_ok"] = all(
        (e.get("vlm_identity") or {}).get("pdf_revision") == revision
        for e in vlm.values() if isinstance(e, dict)) if vlm else None
    vec = load_json(ddir / "vec.json", None)
    if isinstance(vec, dict):
        try:
            mtime = (PROJECTS_DIR / slug / "input.pdf").stat().st_mtime
        except OSError:
            mtime = None
        out["vec_mtime_ok"] = vec.get("pdf_mtime") == mtime
    else:
        out["vec_mtime_ok"] = None
    return out


def _recompute_placements(slug, revision, log):
    """顺手把步骤4 重算一遍（纯本地矢量几何、零模型调用）。

    旧的 placements 已经在导入时剥离，不重算的话用户会看到「一个放置都没有」。
    这里只调 steps.placements / steps.views 的公开函数，plan 过滤与 fail-closed
    语义完全由它们决定，本工具不复制任何阈值。返回汇总 dict 或 None（跳过）。
    """
    try:
        from steps.placements import match_placements
        from steps.views import (groups_need_classification,
                                 has_current_view_types, merge_view_types)
    except Exception as exc:                                    # noqa: BLE001
        # 步骤4 / 步骤3 由并行开发；缺失时只提示，不猜语义。
        log(f"  [todo] 旧 placements 已剥离，steps.placements 尚不可用 "
            f"({type(exc).__name__}) —— 请在导入后跑一次 placements 阶段"
            f"（本地几何，免费）")
        return None

    ddir = DATA_DIR / slug
    pdf = PROJECTS_DIR / slug / "input.pdf"
    symbols = load_json(ddir / "symbols.json", None)
    if not isinstance(symbols, dict) or not symbols:
        return None
    views = load_json(ddir / "view_types.json", {}) or {}
    total = {"pages": 0, "placed": 0, "dropped_outside_plan": 0, "pending": 0}
    for page, entry in symbols.items():
        result = (entry or {}).get("result")
        if not isinstance(result, dict):
            continue
        groups = result.get("groups") or []
        if groups_need_classification(groups):
            view_entry = views.get(str(page))
            if not has_current_view_types(view_entry, groups, revision):
                # 视图分类还没算 → fail-closed，这一页留给管线自己跑
                total["pending"] += 1
                continue
            typed = merge_view_types(groups, view_entry)
        else:
            typed = groups
        summary = match_placements(pdf, int(page) - 1,
                                   result.get("symbols") or [], typed)
        result.update(summary)
        total["pages"] += 1
        total["placed"] += summary["placed"]
        total["dropped_outside_plan"] += summary["dropped_outside_plan"]
    save_json(ddir / "symbols.json", symbols)
    log(f"  placements 已本地重算（零模型调用）：{total['pages']} 页, "
        f"{total['placed']} 个放置, {total['dropped_outside_plan']} 个被 plan "
        f"过滤丢弃" + (f", {total['pending']} 页等视图分类" if total["pending"]
                       else ""))
    return total


def import_project(slug, ref_dir=REF_DIR_DEFAULT, dry_run=False, log=print):
    """把一个 slug 从参考项目搬进 fence_lite，并重算全部 PDF-aware 签名。

    返回一份自检报告 dict。已存在同名项目直接 ValueError（绝不静默覆盖）。
    """
    require_slug(slug)
    src_pdf, src_dir = ref_paths(slug, ref_dir)
    if not src_pdf.exists():
        raise FileNotFoundError(f"missing source PDF: {src_pdf}")
    if not (src_dir / REQUIRED_SRC).exists():
        raise FileNotFoundError(f"missing source cache: {src_dir / REQUIRED_SRC}")

    dst_pdf = PROJECTS_DIR / slug / "input.pdf"
    dst_dir = DATA_DIR / slug
    existing = [dst_pdf] if dst_pdf.exists() else []
    existing += sorted(dst_dir.glob("*.json")) if dst_dir.is_dir() else []
    if existing:
        raise ValueError(
            f"project '{slug}' already exists here ("
            + ", ".join(p.name for p in existing[:3])
            + "); delete projects/{0} and data/{0} first".format(slug))

    present = {src: dst for src, dst in CACHE_MAP.items()
               if (src_dir / src).exists()}
    report = {"slug": slug, "dry_run": bool(dry_run),
              "old_revision": (load_json(src_dir / REQUIRED_SRC, {}) or {})
              .get("pdf_revision"),
              "files": dict(present)}

    if dry_run:
        log(f"[{slug}] DRY RUN")
        log(f"  pdf   {src_pdf}  ->  {dst_pdf}")
        for src, dst in present.items():
            log(f"  cache {src:>18}  ->  data/{slug}/{dst}")
        for src in CACHE_MAP:
            if src not in present:
                log(f"  cache {src:>18}  ->  (源里没有，跳过)")
        for name in SKIP_SRC:
            log(f"  skip  {name:>18}  (已砍 / 自动重生)")
        log(f"  old_revision={report['old_revision']}  "
            f"新 revision 与所有 sig 会在真正导入时重算")
        return report

    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    # copyfile（不是 copy2）：故意让目标拿一个新的 mtime，走「统一重算 revision」
    # 这一条路径，而不是依赖 mtime 是否碰巧被保住。
    shutil.copyfile(src_pdf, dst_pdf)
    revision = pdf_revision(dst_pdf)
    report["pdf_revision"] = revision
    log(f"[{slug}] old_revision={report['old_revision']}  "
        f"new_revision={revision}")

    results = load_json(src_dir / REQUIRED_SRC, {}) or {}
    _rewrite_results(results, revision, source_dir=src_dir)
    results["slug"] = slug
    save_json(dst_dir / "results.json", results)
    pages = results.get("pages") or {}

    symbols = None
    for src, dst in present.items():
        # results.json 已经写完；view_types.json 要等 symbols 重算完才能签名。
        if src in (REQUIRED_SRC, "view_types.json"):
            continue
        data = load_json(src_dir / src, None)
        if src in ("vlm_extra.json", "vlm_flash.json"):
            fixed = _rewrite_vlm(data, revision)
            log(f"  {dst}: {fixed} 条记录的 vlm_identity.pdf_revision 已更新")
        elif src == "vec.json" and isinstance(data, dict):
            data["pdf_mtime"] = dst_pdf.stat().st_mtime
            log(f"  {dst}: pdf_mtime 已更新（否则矢量层会全量重抽，免费但慢）")
        elif src == "symbols.json":
            stripped = _strip_stale_propagation(data)
            fixed = _symbol_sigs(data, pages, revision)
            symbols = data
            log(f"  {dst}: {fixed} 页 sig 已重算；剥离旧字段 "
                + (", ".join(f"{k}×{v}" for k, v in sorted(stripped.items()))
                   or "无"))
        save_json(dst_dir / dst, data)

    if "view_types.json" in present:
        views = load_json(src_dir / "view_types.json", None)
        if symbols is None:
            symbols = load_json(dst_dir / "symbols.json", {}) or {}
        fixed = _view_sigs(views, symbols, revision)
        if fixed is None:
            log("  view_types.json: steps.views 尚未就绪，sig 未重算 —— "
                "跑起来后视图分类会按新 revision 重新付费一次")
        else:
            log(f"  view_types.json: {fixed} 页 sig 已重算")
        save_json(dst_dir / "view_types.json", views)

    for name in SKIP_SRC:
        log(f"  skip {name}（已砍 / 自动重生）")

    report.update(_verify(slug, revision, log))
    report["placements_recomputed"] = \
        _recompute_placements(slug, revision, log) is not None
    # 计数从磁盘重读，才反映 placements 重算之后的真实状态。
    report.update(_tally(results, load_json(dst_dir / "symbols.json", {})))
    return report


def _print_selfcheck(reports, log=print):
    log("")
    log("==== 自检 ====")
    header = ("slug", "页", "有文字页", "文字项", "symbol", "placements",
              "sym sig", "view sig")
    log("  " + " | ".join(header))
    for rep in reports:
        if rep.get("dry_run"):
            log(f"  {rep['slug']} | (dry-run，未落地)")
            continue

        def mark(value):
            return {True: "OK", False: "FAIL", None: "n/a"}[value]

        log("  " + " | ".join(str(v) for v in (
            rep["slug"], rep.get("page_count"), rep.get("pages_with_text"),
            rep.get("text_items"), rep.get("symbols"), rep.get("placements"),
            mark(rep.get("symbols_sig_ok")), mark(rep.get("view_sig_ok")))))
        for key in ("results_revision_ok", "vlm_revision_ok", "vec_mtime_ok"):
            if rep.get(key) is False:
                log(f"    [FAIL] {key}")
    if any(r.get("placements_recomputed") is False for r in reports
           if not r.get("dry_run")):
        log("")
        log("提示：placements 为 0 是预期的 —— 旧的放置结果是旧语义（没有 plan "
            "视图过滤、line 类也跑过整线传播），已在导入时剥离；步骤4 未能就地"
            "重算，请跑一次 placements 阶段（本地矢量几何，零 Gemini 花费）。")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="把 fence_takeoff_web 已算好的项目搬进 fence_lite（零花费）")
    parser.add_argument("slugs", nargs="*", help="项目 slug，可多个")
    parser.add_argument("--from", dest="ref_dir", default=str(REF_DIR_DEFAULT),
                        help=f"参考项目根目录（默认 {REF_DIR_DEFAULT}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印会做什么，不落地")
    args = parser.parse_args(argv)

    ref_dir = Path(args.ref_dir)
    if not args.slugs:
        found = available_slugs(ref_dir)
        print(f"可导入的 slug（{ref_dir}）：" + (", ".join(found) or "无"))
        return 2

    reports, failed = [], 0
    for slug in args.slugs:
        try:
            reports.append(import_project(slug, ref_dir=ref_dir,
                                          dry_run=args.dry_run))
        except (ValueError, OSError) as exc:
            failed += 1
            print(f"[{slug}] 失败: {type(exc).__name__}: {exc}")
    if reports:
        _print_selfcheck(reports)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
