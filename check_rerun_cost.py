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

from steps.store import DATA_DIR, load_json
from core.config import MODEL_NAME
from steps.versions import (FUSED_VERSION, PLACEMENT_VERSION, SYMBOL_PROMPT_V,
                            SYMBOL_VERSION, TEXT_JUDGE_VERSION, VIEW_VERSION)

SLUG = sys.argv[1] if len(sys.argv) > 1 else "combined_bid_set_ten_mile_storage_phase_1"
d = DATA_DIR / SLUG

res = load_json(d / "results.json", {}) or {}
sym = load_json(d / "symbols.json", {}) or {}
vt = load_json(d / "view_types.json", {}) or {}
arw = load_json(d / "arrows.json", {}) or {}
vlm = load_json(d / "vlm.json", {}) or {}
judge = load_json(d / "textjudge.json", {}) or {}

pages = int(res.get("page_count") or 0)
print(f"{SLUG}: {pages} pages")
print()


def counts(store, key, path=None):
    def get(e):
        if path:
            e = (e or {}).get(path) or {}
        return (e or {}).get(key)
    return dict(collections.Counter(get(e) for e in store.values()))


print(f"  results.fused_v   = {res.get('fused_v')}   (current {FUSED_VERSION})")
print(f"  textjudge.v       = {judge.get('v')}   (current {TEXT_JUDGE_VERSION})")
# vlm.json 的页数**不**该等于总页数：有文字层但没有目标文字的页是故意不扫的
# （steps/text/page.py 的 vlm_needed），那正是省钱的地方。所以只看它是否
# 与「有结果的页」一致，不看它是否等于总页数。
with_text = sum(1 for r in (res.get("pages") or {}).values()
                if (r or {}).get("vlm_items"))
print(f"  vlm.json pages    = {len(vlm)}   有 vlm_items 的页 = {with_text}"
      f"   <- 有文字层且无目标文字的页故意不扫，不是缺页")
print(f"  symbols pages     = {len(sym)} / {pages}")
print(f"    .pv  = {counts(sym,'pv')}   (current {SYMBOL_PROMPT_V})  <- 付费")
print(f"    .v   = {counts(sym,'v')}   (current {SYMBOL_VERSION})   <- 免费重过滤")
print(f"    .plc_v = {counts(sym,'plc_v',path='result')}   (current {PLACEMENT_VERSION})  <- 免费")
print(f"  view_types pages  = {len(vt)}")
print(f"    .v   = {counts(vt,'v')}   (current {VIEW_VERSION})   <- 付费")
print(f"  arrows pages      = {len(arw)}  sig tails {sorted({str(e.get('sig'))[-4:] for e in arw.values()})}"
      f"   <- 免费（本地边车）")
print()

paid = []
if res.get("fused_v") != FUSED_VERSION:
    paid.append("fused 语义变了 -> 重新融合（免费，但会连带重算）")
if judge.get("v") != TEXT_JUDGE_VERSION:
    paid.append("判词版本变了 -> 重判所有字符串（付费）")
if len(vlm) < with_text:
    paid.append(f"vlm.json ({len(vlm)}) 少于有结果的页 ({with_text}) -> 缺的页要付费扫描")
if set(counts(sym, "pv")) - {SYMBOL_PROMPT_V}:
    paid.append("symbols 提示词版本不当期 -> 每页重新付费推理")
if set(counts(vt, "v")) - {VIEW_VERSION}:
    paid.append("视图分类版本不当期 -> 每页重新付费分类")

# 模型这一维和版本戳一样会作废缓存，而且更隐蔽：symbols / vlm / 判词都按
# resolve_model(None) 校验，重跑时若没把模型钉回去，整份缓存都读作过期，
# 会用默认模型重新付费**并覆盖掉**原来那份结果。对比运行尤其致命。
import job as _job                                            # noqa: E402
pinned = _job.variant_model(SLUG)
effective = pinned or MODEL_NAME
stored = {m for m in counts(sym, "model") if m} \
    | {m for m in {((e or {}).get("vlm_identity") or {}).get("model")
                   for e in vlm.values()} if m}
if judge.get("model"):
    stored.add(judge["model"])
print(f"  缓存里的模型        = {sorted(stored) or ['-']}")
print(f"  重跑会用的模型      = {effective}"
      + (f"   (变体 slug 钉定)" if pinned else "   (进程默认)"))
if stored - {effective}:
    paid.append(f"缓存模型 {sorted(stored)} != 重跑模型 {effective}"
                f" -> 会用 {effective} 重新付费并覆盖原结果")

if paid:
    print("会产生付费调用：")
    for p in paid:
        print("  !", p)
else:
    print("不会产生付费调用：所有付费步的版本戳都当期，rerun 只会重算 arrows"
          "（本地边车 + 纯几何）与免费的重过滤/放置。")
