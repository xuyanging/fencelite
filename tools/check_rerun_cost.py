"""重跑这个项目会不会花钱？—— 只读盘，零调用。

    venv/bin/python tools/check_rerun_cost.py <slug>

Will a rerun of this project cost money?

Every paid stage caches under a version stamp. If each stage's stored stamp is
already current, a rerun with reset=false re-publishes from cache and issues no
model calls; only the stages whose stamp moved are recomputed. Checked BEFORE
running, because last time an unchecked rerun quietly completed 64 unfinished
pages of a 73-page PDF and cost $1.15.
"""
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import MODEL_NAME
from steps.store import DATA_DIR, load_json, pdf_path
from steps.text import (TARGET_DEFAULT, build_vlm_prompt,
                        is_current_primary_record,
                        is_current_secondary_record, is_default_target,
                        select_instances, vlm_identity)
from steps.versions import (FUSED_VERSION, PLACEMENT_VERSION, SYMBOL_PROMPT_V,
                            SYMBOL_VERSION, TEXT_JUDGE_VERSION, VIEW_VERSION)

import job as _job                                            # noqa: E402


def counts(store, key, path=None):
    def get(e):
        if not isinstance(e, dict):
            return None
        if path:
            e = e.get(path) or {}
            if not isinstance(e, dict):
                return None
        return e.get(key)
    return dict(collections.Counter(get(e) for e in store.values()))


def _models_of(store):
    models = set()
    for record in (store or {}).values():
        if not isinstance(record, dict):
            continue
        identity = record.get("vlm_identity") or {}
        if not isinstance(identity, dict):
            identity = {}
        model = identity.get("model") or record.get("model")
        if model:
            models.add(model)
    return models


def _page_list(pages):
    values = list(pages)
    if not values:
        return "-"
    shown = ",".join(f"P{page}" for page in values[:12])
    return shown + (f",…(+{len(values) - 12})" if len(values) > 12 else "")


def _vision_status(slug, res, judge, vec, vlm, flash, effective):
    """Mirror the production page/identity gates without issuing model calls."""
    source = pdf_path(slug)
    result_pages = int(res.get("page_count") or 0)
    actual_pages = _job.page_count_of(slug) if source.is_file() else 0
    page_count = actual_pages or result_pages
    all_pages = list(range(1, page_count + 1))
    target = res.get("target") or TARGET_DEFAULT
    prompt = build_vlm_prompt(target)
    try:
        primary_identity = vlm_identity(source, effective, prompt)
        flash_identity = vlm_identity(source, _job.FLASH_MODEL, prompt)
        identity_error = None
    except OSError as exc:
        primary_identity = flash_identity = None
        identity_error = f"{type(exc).__name__}: {exc}"

    if _job.SCAN_ALL_PAGES:
        primary_pages = secondary_pages = all_pages
    else:
        vpages = (vec or {}).get("pages") or {}
        flagged = {text for text, verdict in
                   ((judge or {}).get("verdicts") or {}).items() if verdict}
        default_target = is_default_target(target)

        def has_text(page):
            if not _job.SCAN_NO_TEXT_PAGES:
                return True
            return bool((vpages.get(str(page)) or {}).get("has_text"))

        primary_pages = []
        secondary_pages = []
        for page in all_pages:
            page_vec = vpages.get(str(page)) or {}
            instances = select_instances(
                page_vec.get("lines") or [], flagged,
                use_kw_floor=default_target)
            if (not has_text(page)) or instances or str(page) in vlm:
                primary_pages.append(page)
            if (_job.SCAN_NO_TEXT_PAGES
                    and not page_vec.get("has_text")):
                secondary_pages.append(page)

    if identity_error:
        primary_due = list(primary_pages)
        flash_due = list(secondary_pages)
    else:
        primary_due = [
            page for page in primary_pages
            if not is_current_primary_record(
                vlm.get(str(page)), primary_identity)
        ]
        flash_due = [
            page for page in secondary_pages
            if not is_current_secondary_record(
                flash.get(str(page)), flash_identity)
        ]
    return {
        "page_count": page_count,
        "mode": "all-pages" if _job.SCAN_ALL_PAGES else "selective",
        "primary_required": primary_pages,
        "primary_due": primary_due,
        "flash_required": secondary_pages,
        "flash_due": flash_due,
        "identity_error": identity_error,
    }


def main(slug):
    d = DATA_DIR / slug
    res = load_json(d / "results.json", {}) or {}
    sym = load_json(d / "symbols.json", {}) or {}
    vt = load_json(d / "view_types.json", {}) or {}
    arw = load_json(d / "arrows.json", {}) or {}
    vlm = load_json(d / "vlm.json", {}) or {}
    flash = load_json(d / "vlm_flash.json", {}) or {}
    judge = load_json(d / "textjudge.json", {}) or {}
    vec = load_json(d / "vec.json", {}) or {}

    pinned = _job.variant_model(slug)
    effective = pinned or MODEL_NAME
    vision = _vision_status(slug, res, judge, vec, vlm, flash, effective)
    pages = vision["page_count"]
    print(f"{slug}: {pages} pages")
    print()
    print(f"  results.fused_v   = {res.get('fused_v')}   (current {FUSED_VERSION})")
    print(f"  textjudge.v       = {judge.get('v')}   (current {TEXT_JUDGE_VERSION})")
    print(f"  vision mode       = {vision['mode']}")
    print(f"  vlm.json current  = "
          f"{len(vision['primary_required']) - len(vision['primary_due'])} / "
          f"{len(vision['primary_required'])} required"
          f"   due={_page_list(vision['primary_due'])}")
    print(f"  vlm_flash current = "
          f"{len(vision['flash_required']) - len(vision['flash_due'])} / "
          f"{len(vision['flash_required'])} required"
          f"   due={_page_list(vision['flash_due'])}")
    print(f"  symbols pages     = {len(sym)} / {pages}")
    print(f"    .pv  = {counts(sym,'pv')}   (current {SYMBOL_PROMPT_V})  <- 付费")
    print(f"    .v   = {counts(sym,'v')}   (current {SYMBOL_VERSION})   <- 免费重过滤")
    print(f"    .plc_v = {counts(sym,'plc_v',path='result')}   (current {PLACEMENT_VERSION})  <- 免费")
    print(f"  view_types pages  = {len(vt)}")
    print(f"    .v   = {counts(vt,'v')}   (current {VIEW_VERSION})   <- 付费")
    print(f"  arrows pages      = {len(arw)}  sig tails "
          f"{sorted({str(e.get('sig'))[-4:] for e in arw.values()})}"
          f"   <- 免费（本地边车）")
    print()

    paid = []
    if res.get("fused_v") != FUSED_VERSION:
        paid.append("fused 语义变了 -> 融合本身免费，但新 items/sig 可能使"
                    " symbols/views 失效，未回放前无法保证零付费")
    if judge.get("v") != TEXT_JUDGE_VERSION:
        paid.append("判词版本变了 -> 重判所有字符串（付费）")
    if vision["identity_error"]:
        paid.append("无法按当前 input.pdf 计算 VLM identity -> 不能证明图像缓存可复用: "
                    + vision["identity_error"])
    else:
        if vision["primary_due"]:
            paid.append(
                f"vlm.json 有 {len(vision['primary_due'])} 个必扫页缺失、失败或 identity/role 不当期 "
                f"({_page_list(vision['primary_due'])}) -> 主模型要付费扫描")
        if vision["flash_due"]:
            paid.append(
                f"vlm_flash.json 有 {len(vision['flash_due'])} 个必扫页缺失、失败或 identity/role 不当期 "
                f"({_page_list(vision['flash_due'])}) -> Flash 要付费扫描")
    if set(counts(sym, "pv")) - {SYMBOL_PROMPT_V}:
        paid.append("symbols 提示词版本不当期 -> 每页重新付费推理")
    if set(counts(vt, "v")) - {VIEW_VERSION}:
        paid.append("视图分类版本不当期 -> 每页重新付费分类")

    # VLM/Flash 复用已由必扫页的完整 identity 决定；超出当前
    # PDF 页范围的孤儿记录不应因模型不同被二次判成付费。
    symbol_models = {model for model in counts(sym, "model") if model}
    print(f"  symbols 缓存模型  = {sorted(symbol_models) or ['-']}")
    print(f"  主 VLM 缓存模型   = {sorted(_models_of(vlm)) or ['-']}")
    print(f"  Flash 缓存里的模型  = {sorted(_models_of(flash)) or ['-']}"
          f"   (expected {_job.FLASH_MODEL})")
    print(f"  重跑主模型          = {effective}"
          + (f"   (变体 slug 钉定)" if pinned else "   (进程默认)"))
    if symbol_models - {effective}:
        paid.append(f"symbols 缓存模型 {sorted(symbol_models)} != 重跑模型 {effective}"
                    f" -> 会用 {effective} 重新付费并覆盖原结果")

    if paid:
        print("会产生付费调用：")
        for reason in paid:
            print("  !", reason)
    else:
        print("不会产生付费调用：所有付费步的版本戳和逐页 identity 都当期，"
              "rerun 只会重算 arrows/linetypes（本地边车 + 纯几何）"
              "与免费的重过滤/放置。")


if __name__ == "__main__":
    selected = (sys.argv[1] if len(sys.argv) > 1
                else "combined_bid_set_ten_mile_storage_phase_1")
    main(selected)
