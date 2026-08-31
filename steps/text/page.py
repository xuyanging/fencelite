"""单页文字步的组装 —— 判词 + 矢量行 + VLM items → 一条融合页记录.

把「一页的完整文字步」收成两个纯本地函数，作业(job.py)和以后的单页重扫
共用同一段代码，永远不会出现两条会漂移的合并循环：

  fuse_page    判词命中的矢量行 → select_instances → 剥符号码 → 融合 → 调试视图。
               零模型调用、零落盘：谁调用谁负责把付费 raw 存进 vlm.json、
               把结果写进 results.json。
  vlm_needed   这一页要不要花钱读图。当期身份的 primary 记录不存在，且满足
               四条触发中的任一条：准确率模式强制全页扫描／本页没有文字层
               （扫描页，矢量通道天生看不见任何东西）／本页有矢量命中实例／
               本页已经被记过一次。
               「已经被记过」那一半是关键 —— 旧身份 / 报错 / 结构损坏的记录
               必须显式变成重扫工作，不能悄悄当成缓存命中。
"""
from steps.text.clean import strip_marker_codes
from steps.text.debug_view import attach_text_debug
from steps.text.judge import select_instances
from steps.text.merge import fuse
from steps.text.vlmcache import is_current_primary_record


def _flagged(verdicts):
    """Accept the flagged normalized-string set, or the raw judge verdict map.

    ``select_instances`` tests plain membership, so handing it a
    ``{norm: bool}`` map would silently promote every *rejected* string to a
    match.  Normalize here instead of trusting every call site.
    """
    if isinstance(verdicts, dict):
        return {s for s, v in verdicts.items() if v}
    return verdicts or set()


def fuse_page(pdf, page_index, vec_page, verdicts, vlm_items, *,
              use_kw_floor=True, dbg=None, judged_new=None):
    """One page's fused text record.  Purely local — no model call, no I/O
    beyond reading the PDF's vector geometry for the marker-code strip.

    ``page_index`` is 0-based (the same frame ``vector_scan`` uses);
    ``vec_page`` is that page's ``{"lines", "has_text"}`` vector record;
    ``verdicts`` are the judge's fence-related normalized strings;
    ``vlm_items`` is the raw (already unioned, if any) VLM item list.
    Returns ``{vlm_items, vec_added, vec_covered, has_text
    [, codes_stripped][, debug]}`` — the caller adds the cache-provenance
    fields (vlm_from / vlm_sources / vlm_error).
    """
    vec_page = vec_page or {}
    lines = vec_page.get("lines")
    inst = select_instances(lines or [], _flagged(verdicts),
                            use_kw_floor=use_kw_floor)
    vlm_items, inst, n_code = strip_marker_codes(
        vlm_items, inst, pdf, page_index, dbg=dbg)
    rec = fuse(vlm_items, inst, lines=lines)
    attach_text_debug(rec, dbg, inst, judged_new, use_kw_floor=use_kw_floor)
    if n_code:
        rec["codes_stripped"] = n_code
    rec["has_text"] = bool(vec_page.get("has_text"))
    return rec


def vlm_needed(page, instances, vlm_store, expected_identity, *,
               has_text=True, scan_all=False):
    """True when this 1-based page still owes one paid primary scan.

    Four independent triggers:
      * ``scan_all`` is True — accuracy-first mode.  A page can contain an
        ordinary extractable title block while the relevant CAD lettering is
        made only from stroked paths.  In that mixed-page case ``has_text`` is
        true but the vector text channel is blind, so every page must be read
        visually.
      * ``has_text`` is False — the page carries no vector text layer at all
        (scanned raster sheet), so the free vector+judge path can never see
        anything and reading the image is the ONLY way to honour "find the
        matching text on every page".  Without this the page would silently
        publish zero results while the job still reports success.
      * the free vector+judge path found matching text on this page;
      * the page is already known to ``vlm.json`` but its stored record is
        stale / errored / structurally invalid — that is explicit rework, never
        a silent cache hit.
    With ``scan_all=False``, a page that has a text layer but no matching text
    is deliberately not scanned.  That selective mode is retained as an
    explicit cost-saving option; it is not safe for mixed CAD pages.
    """
    key = str(page)
    store = vlm_store or {}
    if is_current_primary_record(store.get(key), expected_identity):
        return False
    return bool(scan_all) or (not has_text) or bool(instances) or key in store
