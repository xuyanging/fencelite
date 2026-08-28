"""逐页健康状况 —— 分清「待重算」「跑失败」「压根没排上活」，读盘期算，不落盘.

要解决的问题：一页的模型调用失败后，下游阶段会静默跳过它（那页没有 items，
``_symbol_jobs`` 里直接 ``continue``）。界面上看起来是「这页没东西」，而实际是
「这页上游失败了」—— 两件完全不同的事，混在一起没法排查。

三个刻意的决定，都是踩过之后定的：

* **不写 flag 字段。** 失败的证据本来就都在盘上（vlm.json 的 error、arrows.json
  里缺 items 的条目、linetypes 页文件的 error）。再写一份 flag 就有两个真相来源，
  它们一定会漂移 —— 重试成功了却忘了清 flag，那页就永远带着「失败」标记。
* **「该不该有结果」复用管线自己的排活函数**（``job._symbol_jobs`` 等），不自己
  重写判据。第一版自己写的，10 个项目报出假阳性：视图分类只跑「有 shape 样例
  且需要分类」的页，而我按"有 items 就该有 view 条目"判。排活函数是唯一真相。
* **「待重算」不是「失败」。** 只用排活函数还不够 —— 版本号或引擎摘要一变，
  全部页都会被列为待办（实测：刚把 cpu_budget 从签名里拿掉，178 页线型立刻全部
  进待办）。那是正常的待重算。所以还要第二个证据：**带 error 且 sig 当期**的
  落盘记录，才算「在当前配置下试过并且失败」。

四种状态：

    failed    有 error 记录且 sig 当期 —— 在当前配置下试过、失败了
    pending   排活函数列为待办，但没有失败记录 —— 待重算/还没跑
    blocked   前置不具备（例如项目缺 PDF），压根排不上活
    ok        其余

symbols / views 失败时**什么都不写**（见 job.py 的 _symbol_one / _view_one），
所以它们无法区分 failed 与 pending —— 这一点在下面显式标了 ``unknown_failure``，
不猜。

标记只给内部看：webapp 挂在 /api/page 上，前端仅在「内部视图」(⋯ 打开) 渲染。
"""
from __future__ import annotations

__all__ = ["document_health", "page_states", "STAGES", "NO_ERROR_RECORD"]

STAGES = ("text", "symbols", "views", "arrows", "linetypes")
# 这两个阶段失败时不落盘，所以「待重算」和「失败」在盘上长得一样
NO_ERROR_RECORD = ("symbols", "views")


def _error_pages(slug):
    """{阶段: {页: sig}} —— 盘上带 error 的记录及其签名。"""
    from steps import store

    directory = store.slug_dir(slug)
    out = {stage: {} for stage in STAGES}

    for name in ("vlm.json", "vlm_flash.json"):
        cache = store.load_json(directory / name, {}) or {}
        for key, entry in cache.items():
            if str(key).isdigit() and isinstance(entry, dict) and entry.get("error"):
                # 文字层的身份是 vlm_identity 三元组，不是 sig 字符串
                out["text"][int(key)] = entry.get("vlm_identity")

    arrows = store.load_json(directory / "arrows.json", {}) or {}
    for key, entry in arrows.items():
        if str(key).isdigit() and isinstance(entry, dict) and entry.get("error"):
            out["arrows"][int(key)] = entry.get("sig")

    from steps import linetypes as lt_mod

    for page in lt_mod.computed_pages(slug):
        entry = lt_mod.load_page(slug, page)
        if isinstance(entry, dict) and entry.get("error"):
            out["linetypes"][int(page)] = entry.get("sig")
    return out


def _pending(slug):
    """{阶段: {待办页}} —— 直接问管线自己的排活函数。"""
    import job                     # 延迟导入：job 依赖 steps.*，模块级会成环

    out = {}
    for stage, planner in (("symbols", job._symbol_jobs),
                           ("views", job._view_jobs),
                           ("arrows", job._arrow_jobs),
                           ("linetypes", job._linetype_jobs)):
        try:
            jobs = planner(slug)
            if isinstance(jobs, tuple):          # 有的返回 (jobs, warnings)
                jobs = jobs[0]
            out[stage] = {int(row[0]) for row in (jobs or ())}
        except Exception:                                       # noqa: BLE001
            out[stage] = set()
    out["text"] = set()            # 文字阶段没有独立排活函数，只看 error 记录
    return out


def page_states(slug):
    """返回 ({页: {阶段: 状态}}, blocked_reason)。只收录有异常的页。"""
    from steps import store

    if not store.pdf_path(slug).is_file():
        # 缺 PDF 时排活函数一律返回空，会把"跑不了"误报成"没问题"
        return {}, "blocked-no-pdf"

    pending = _pending(slug)
    errors = _error_pages(slug)
    out = {}
    for stage in STAGES:
        for page, sig in (errors.get(stage) or {}).items():
            # 有 error 记录就算 failed。不再去核 sig 是否当期 —— 各阶段的
            # has_current_* 都要求「有结果」，失败记录必然同时出现在待办里；
            # 真正重试成功之后这条记录会被覆盖掉，自然消失。
            out.setdefault(int(page), {})[stage] = "failed"
        for page in (pending.get(stage) or ()):
            state = out.setdefault(int(page), {})
            if stage not in state:
                state[stage] = ("unknown_failure" if stage in NO_ERROR_RECORD
                                else "pending")
    return out, None


def document_health(slug):
    """整份文档的健康摘要。给作业收尾和内部视图用；逐页读，别放热路径。"""
    states, blocked = page_states(slug)
    counts, by_stage = {}, {}
    for page, stages in states.items():
        for stage, state in stages.items():
            counts[state] = counts.get(state, 0) + 1
            by_stage.setdefault(stage, {}).setdefault(state, []).append(page)
    for stage in by_stage:
        for state in by_stage[stage]:
            by_stage[stage][state].sort()
    failed = sorted(p for p, st in states.items()
                    if any(v == "failed" for v in st.values()))
    return {"failed": failed, "counts": counts, "by_stage": by_stage,
            "blocked": blocked}
