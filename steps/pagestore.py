"""按页分文件的缓存布局 —— 一页一个文件，页与页互不影响.

为什么要它：原来 vlm.json / symbols.json / view_types.json / arrows.json 都是
``{页号: entry}`` 的**单个大字典**，每算完一页就整文件重写一遍。三个后果：

1. **并发被锁掉**。读-改-写必须串行，所以 job.py 用 ``_IO_LOCK`` / ``_VLM_LOCK``
   把所有页的写入排成一队（job.py:86-87）。页级并发再高，落盘这一段也是单线程。
2. **累计重写量是 O(N²)**。taylor 那种 19 页的项目，线型结果单页能到 2.8 MB，
   整份 28 MB —— 跑一遍累计重写约 266 MB。页数越多越离谱。
3. **一页能连累整份**。写入过程中崩溃 / 一页的畸形数据，影响的是整个文件。

改成一页一文件之后：写盘只碰自己那一页，不需要任何锁；重写量与页数成正比；
一页坏掉不动别的页。``steps/linetypes`` 已经先用这个布局验证过了（见那边的
docstring），这里把它抽成公用的。

**读取兼容旧的单文件**：老项目照样能显示，不做隐式自动迁移 —— 迁移是显式的
一次性动作（``split_legacy``），因为它要动盘上已经付过钱的结果，必须能被审计。

布局：

    data/<slug>/<kind>/<page>.json       新（一页一文件）
    data/<slug>/<kind>.json              旧（{页号: entry} 单文件，只读回落）

``kind`` 用不带后缀的名字："vlm" / "symbols" / "view_types" / "arrows" / …
"""
from __future__ import annotations

import json
from pathlib import Path

from steps import store

__all__ = ["dir_of", "legacy_path", "load_page", "loaded_pages", "page_path",
           "pages_of", "save_page", "split_legacy", "load_all", "drop_page"]


def dir_of(slug, kind):
    return store.slug_dir(slug) / str(kind)


def page_path(slug, kind, page):
    return dir_of(slug, kind) / f"{int(page)}.json"


def legacy_path(slug, kind):
    return store.slug_dir(slug) / f"{kind}.json"


def load_page(slug, kind, page, default=None):
    """读一页。新布局优先，回落旧单文件。

    回落是**只读**的：不会顺手把旧文件拆开。见模块 docstring。
    """
    path = page_path(slug, kind, page)
    if path.is_file():
        entry = store.load_json(path, None)
        if entry is not None:
            return entry
    legacy = store.load_json(legacy_path(slug, kind), None)
    if isinstance(legacy, dict):
        entry = legacy.get(str(page))
        if entry is not None:
            return entry
    return default


def save_page(slug, kind, page, entry):
    """写一页。**不需要锁** —— 每页各写自己的文件，save_json 保证单文件原子。"""
    path = page_path(slug, kind, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_json(path, entry)


def drop_page(slug, kind, page):
    """删一页（重置某一页时用）。不存在也不报错。"""
    path = page_path(slug, kind, page)
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def pages_of(slug, kind):
    """这个 kind 下已经有结果的页号（两种布局都算上），升序。"""
    pages = set()
    directory = dir_of(slug, kind)
    if directory.is_dir():
        for path in directory.glob("*.json"):
            if path.stem.isdigit():
                pages.add(int(path.stem))
    legacy = store.load_json(legacy_path(slug, kind), None)
    if isinstance(legacy, dict):
        for key in legacy:
            if str(key).isdigit():
                pages.add(int(key))
    return sorted(pages)


def loaded_pages(slug, kind):
    """{页号: entry}，新布局覆盖旧单文件的同名页。

    给那些**确实需要整份视图**的地方用（/api/overview 的汇总、跨页统计）。
    热路径不要用它 —— 逐页读才是这个布局的意义。
    """
    out = {}
    legacy = store.load_json(legacy_path(slug, kind), None)
    if isinstance(legacy, dict):
        for key, value in legacy.items():
            if str(key).isdigit():
                out[int(key)] = value
    directory = dir_of(slug, kind)
    if directory.is_dir():
        for path in sorted(directory.glob("*.json")):
            if not path.stem.isdigit():
                continue
            entry = store.load_json(path, None)
            if entry is not None:
                out[int(path.stem)] = entry
    return out


# 兼容旧名字（load_all 语义就是 loaded_pages）
load_all = loaded_pages


def split_legacy(slug, kind, *, keep_legacy=True, dry_run=False):
    """把旧的单文件拆成一页一文件。**显式的一次性迁移**，不在读路径里偷偷做。

    ``keep_legacy=True``（默认）保留旧文件不删 —— 它是这次迁移唯一的回退凭据。
    拆完之后读路径会优先命中新布局，旧文件只是躺着。

    返回 {"pages": n, "skipped": m, "written": [...]}；``skipped`` 是新布局里
    已经存在、内容一致的页（重复迁移是幂等的）。
    """
    legacy = store.load_json(legacy_path(slug, kind), None)
    if not isinstance(legacy, dict):
        return {"pages": 0, "skipped": 0, "written": [], "note": "no legacy file"}
    written, skipped = [], 0
    for key, value in sorted(legacy.items(),
                             key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else -1):
        if not str(key).isdigit():
            continue
        page = int(key)
        target = page_path(slug, kind, page)
        if target.is_file():
            existing = store.load_json(target, None)
            # 逐字节比 json 序列化结果，不比对象 —— 键序不同不算差异
            if json.dumps(existing, sort_keys=True) == json.dumps(value, sort_keys=True):
                skipped += 1
                continue
        if not dry_run:
            save_page(slug, kind, page, value)
        written.append(page)
    if not keep_legacy and not dry_run and written:
        legacy_path(slug, kind).unlink(missing_ok=True)
    return {"pages": len(written) + skipped, "skipped": skipped,
            "written": written}
