"""线型层 —— 把箭头末端框里的那一种线，在全页高亮出来（免费，零模型调用）.

这是一个**独立可关掉**的模块：算法本体（line_type_engine）在
``tools/linetype_sidecar/engine/`` 里 vendored 一份，由自己 venv 的一次性子进程
执行；本包只做三件事 —— 拼锚、绑定投票、缓存当期判定。既有模块一行都没动。

一页的链路：

    arrows.json[p].items            ← 已有的箭头/引线结果
        │  每个 target 的 tip [y,x]（页面帧 0-1000）
        ▼
    tools/linetype_sidecar/run.py   ← 子进程：source-aligned PageIR → 聚类 →
        │  反解几何 → 每个末端的「最近 op / 最近有主 op」+ 候选线型的页面帧折线
        ▼
    steps/linetypes/bind.py         ← 一末端一线型 / 同 callout 同线型 / 同文同线型
        │
        ▼
    data/<slug>/linetypes.json      ← 落盘（**不含 plan 信息**）
        │
        ▼
    webapp /api/page                ← plan 只在这里当**显示闸**（bind.resolve_visible）

三条设计上的硬决定：

* **算全页、显示才过 plan 闸**。plan 框不进缓存签名，所以步骤3 重分类 /
  VIEW_VERSION bump **不会**作废线型缓存 —— 闸门是读盘时现算的。代价是详图页也
  会算（gladstone P8 的 15 个末端全在 section/elevation 里），换来的是它们仍然
  可审计：Debug 层能看到"它本来会绑到哪个线型"。
* **只发布被指到的线型**。gladstone P2 实测全页 12 个线型，其中一个就有 725 条
  op / 4350 段；全发是 1 MB+ 的 SVG。真正被指到的只有 1~2 个。所以「用重复指代
  排除无关线型」既是产品口径，也是这层能上前端的前提。
* **一页只有 38% 的 path op 属于任何线型**（P2 实测 1311/3414）。所以判据是
  「拥有离 tip 最近那条 op 的簇」而不是「最近的簇」，最近的 op 是 residual 时
  就给 residual —— 详见 bind.py。
* **失败必须落盘成"被当期判据拒绝的形状"**：写 ``{sig, v, error}`` 且**不带
  ``page``/``bindings``**，于是 has_current_linetypes 判假、下次自动重试。写成
  成功形状就是永久假缓存。
"""
from __future__ import annotations

import hashlib
import json
import os

from steps import store
from steps.linetypes import bind, regroup, sidecar
from steps.linetypes.version import LINETYPE_VERSION

# 总开关。默认**关**，和 ARROWS 同一个风格：不设就完全不影响现有结果。
ENABLED = os.environ.get("LINETYPES", "0") not in ("0", "", "false", "no", "off")

# 是否把高亮裁到「末端所在的那条连通走线」。**默认关 = 画整个线型**，与 TS
# 的显示口径一致、不漏。
#
# 为什么不默认开：这个闸在连续折线上很好用（gladstone P4 的线型 #5 会正确地
# 合成两条走线、覆盖 376/378 op），但在**虚线**上失效 —— lenexa P4 那条围栏
# 由互不相接的短划组成，按 0.5 pt 接触判据会碎成几十条 1-op 走线；而那页真正
# 该排除的无关长带，与围栏的最近距离只有 1.15 页帧单位，**比短划自身的间距
# 还小**。也就是说距离判据在那一页上无论怎么取阈值都分不开，却会实打实地
# 害了连续折线的页。
#
# 走线分解仍然照算并交付（by_run / dropped_runs / 末端的 run_id），作为信息
# 和排查依据；要试这个闸就设 LINETYPE_RUN_GATE=1。
RUN_GATE = os.environ.get("LINETYPE_RUN_GATE", "0") not in (
    "0", "", "false", "no", "off")

__all__ = ["ALL_PAGE_SUFFIX", "ENABLED", "LINETYPE_VERSION",
           "all_page_path", "all_payload", "anchors_of", "bind",
           "computed_pages", "load_page", "page_dir", "page_path", "save_page",
           "compute_page_linetypes", "has_current_linetypes",
           "load_all_page",
           "linetypes_signature", "page_payload", "regroup",
           "resolve_visible", "sidecar", "sidecar_available",
           "symbol_owners_of"]


# ---------------------------------------------------------------- 磁盘布局 --
# **一页一个文件**：data/<slug>/linetypes/<page>.json
#
# 原来是单个 linetypes.json 存 {页: 结果}，每算完一页就整文件重写一遍。线型
# 结果里带折线几何，单页能到 2.8 MB，taylor 那种 19 页的项目整份 28 MB ——
# 于是跑一个项目累计重写约 266 MB，纯属浪费，而且页面并行时所有线程都要抢同
# 一把文件锁、互相等。拆开之后：写盘只碰自己那一页，页级并行才真的能并行。
#
# 读取兼容旧的单文件（还没重跑的项目照样能显示），但不做自动迁移 —— 重跑一次
# 自然就落到新布局了。
LEGACY_NAME = "linetypes.json"
PAGE_DIR_NAME = "linetypes"


def page_dir(slug):
    return store.slug_dir(slug) / PAGE_DIR_NAME


def page_path(slug, page):
    return page_dir(slug) / f"{int(page)}.json"


# ---- 调试产物：全部线型的几何（.all.json）--------------------------------
# 正常视图只发**被指到**的那几个线型，这是它能上前端的前提（一个线型能有 725
# 条 op / 4350 段，全发是 1 MB+ 的 SVG）。但只发这些的代价是：一个 callout 没
# 绑上线型时，图上分不出两种完全不同的原因 ——
#   (a) 那条线根本没被聚成线型；
#   (b) 聚出来了，只是离末端更远、没被选中。
# rapid_city_2 P11 的 callout ② 属于 (b)：方块 pattern 线是 #48（518 op），但
# 末端落在 hatch 带里，最近的 ink 是 hatch 短划（0.26 pt），#48 在 3.4 pt 外。
#
# 所以全部线型的几何单独存一份，**单独一个文件、单独一个接口、前端按需拉**：
# 正常页面加载的体积一点不变，调试时才付那份代价。
#
# 产出者是 tools/linetype_sidecar/run_all.py（不是 run.py —— 后者的字节进
# engine_digest()，改一个字节全站缓存作废），落盘前由 verify_all_geometry.py
# 逐类型比 ops_sha1 证明与主缓存同源。文件里带主缓存当时的 sig，读盘期用它
# 判当期：sig 不符就是**另一次聚类的几何**，宁可不显示也不能拿它下结论。
ALL_PAGE_SUFFIX = ".all.json"


def all_page_path(slug, page):
    return page_dir(slug) / f"{int(page)}{ALL_PAGE_SUFFIX}"


def load_all_page(slug, page):
    path = all_page_path(slug, page)
    if not path.is_file():
        return None
    entry = store.load_json(path, None)
    return entry if isinstance(entry, dict) else None


def all_payload(all_entry, main_entry=None):
    """/api/linetypes_all 的响应体：全部线型 + residual + 谁被绑到了.

    每个类型把各条走线的折线并成一条 ``polylines``（调试视图要看的是"这个线型
    在图上是哪些 ink"，不是走线切分），走线的计数与 bbox 留在 ``runs`` 里。

    ``bound`` 标出这个类型有没有被某个末端选中、被哪些选中 —— 这正是「存在但
    没被选中」和「压根没有」的分界，必须在列表里一眼看到。
    """
    bound = {}
    for row in (main_entry or {}).get("bindings") or ():
        number = row.get("line_type_number")
        if number is None:
            continue
        bound.setdefault(int(number), []).append(
            {"key": row.get("key"), "ti": row.get("ti"),
             "distance": row.get("distance_to_type")})
    types = []
    for row in all_entry.get("types") or ():
        keep = {key: value for key, value in row.items() if key != "by_run"}
        buckets = row.get("by_run") or []
        keep["polylines"] = [line for bucket in buckets
                             for line in (bucket.get("polylines") or ())]
        keep["runs"] = [{key: bucket.get(key) for key in
                         ("run_id", "op_count", "segment_count", "bbox")}
                        for bucket in buckets]
        keep["bound_by"] = bound.get(int(row.get("line_type_number") or 0)) or []
        types.append(keep)
    return {
        "page": all_entry.get("page") or {},
        "engine": all_entry.get("engine") or {},
        "types": types,
        "residual": all_entry.get("residual") or None,
    }


def load_page(slug, page):
    """读一页的线型结果。新布局优先，回落旧的单文件。"""
    path = page_path(slug, page)
    if path.is_file():
        entry = store.load_json(path, None)
        if isinstance(entry, dict):
            return entry
    legacy = store.load_json(store.slug_dir(slug) / LEGACY_NAME, {})
    entry = (legacy or {}).get(str(page))
    return entry if isinstance(entry, dict) else None


def save_page(slug, page, entry):
    """写一页。目录按需建；save_json 自己保证单文件原子。"""
    path = page_path(slug, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    store.save_json(path, entry)


def computed_pages(slug):
    """这个项目已经算过哪些页（两种布局都算上）。"""
    pages = set()
    directory = page_dir(slug)
    if directory.is_dir():
        for path in directory.glob("*.json"):
            if path.stem.isdigit():
                pages.add(int(path.stem))
    legacy = store.load_json(store.slug_dir(slug) / LEGACY_NAME, {})
    for key in (legacy or {}):
        if str(key).isdigit():
            pages.add(int(key))
    return sorted(pages)


def sidecar_available():
    return sidecar.sidecar_available()


def linetypes_signature(arrows_sig):
    """linetypes.json 的缓存签名 = 箭头结果 + 引擎源码 + 本层版本.

    * ``arrows_sig``：直接串上 arrows.arrows_signature(...) 的结果。它自己已经
      锚了 (文字, 框) + label 摘要 + 放置锚 + pdf_revision + ARROWS_VERSION，
      所以箭头一变、PDF 一换，线型绑定必须重算 —— 否则会把旧末端算出来的高亮
      画到新箭头上。
    * ``engine_digest()``：vendored 算法源码树 + ``run.py`` 本身 + 边车 venv 里
      PyMuPDF / pypdf / scipy / numpy 的版本，一起做摘要。改了算法（哪怕一个
      阈值）、换了提取器、或者 scipy 版本变了，都自动作废 —— 这正是"改了要能
      验证"的第一道保障。scipy 必须在里面：unknown_pattern_split 里 5 处
      ``try: import scipy`` 的纯 Python 回退**不是逐位等价**。
    * **plan 框不在里面**，这是有意的：plan 只是显示闸，重分类不该让人重跑
      一页 100 秒的聚类。
    """
    digest = hashlib.sha1(str(sidecar.engine_digest()).encode()).hexdigest()[:12]
    return f"{arrows_sig}+e{digest}|lt{LINETYPE_VERSION}"


def has_current_linetypes(entry, sig):
    """当期可发布？签名相同 且 有 bindings 列表（失败记录没有这个键）。"""
    return bool(isinstance(entry, dict) and entry.get("sig") == sig
                and isinstance(entry.get("bindings"), list))


def anchors_of(arrow_entry):
    """arrows.json 的一页 entry → 线型层要处理的末端列表.

    一个 callout 可以有多个末端，身份是 (key, ti) —— 只用 key 会把同一句话的
    两个末端并成一个，静默丢掉一半绑定（gladstone P2 的 anchor "2" 就是两个）。

    每个末端带上**它所属 callout 自己的引线与箭头笔画**（``own``）。边车要先
    把这些几何对应的 op 剔掉再找最近的 op：compound_path_periodic 认的正是
    周期性重复的相同图元，重复的箭头 / 刻度短刺就是这种东西，不排除就会把
    一堆箭头认成"线型"再高亮回去。
    """
    out = []
    for key, item in ((arrow_entry or {}).get("items") or {}).items():
        own = [list(line) for line in
               list(item.get("leader_strokes") or ())
               + list(item.get("arrow_strokes") or ())
               if isinstance(line, (list, tuple)) and len(line) >= 2]
        for index, target in enumerate(item.get("targets") or ()):
            tip = target.get("tip")
            if isinstance(tip, (list, tuple)) and len(tip) >= 2:
                out.append({"key": key, "ti": index,
                            "tip": [float(tip[0]), float(tip[1])],
                            "own": own})
    return out


def compute_page_linetypes(pdf_path, sheet, items, arrow_entry, *, sig,
                           dbg=None, **kwargs):
    """算一页，返回**待落盘的 entry**（不含 plan 任何信息）。

    sheet 是 **1-based**（results.json 的页键、引擎 API 都是 1-based）。
    没有末端时返回一个显式的空 entry —— 那是"确实没有可绑的对象"，不是失败。
    """
    anchors = anchors_of(arrow_entry)
    if not anchors:
        return {"sig": sig, "v": LINETYPE_VERSION, "bindings": [],
                "groups": [], "line_types": [], "used_all": [],
                "page": {"sheet": int(sheet), "reason": "no-anchors"}}

    payload = sidecar.run_page(pdf_path, int(sheet), anchors, dbg=dbg, **kwargs)
    page_info = payload.get("page") or {}
    bound = bind.bind_page(items, payload.get("bindings") or [],
                           float(page_info.get("tip_precision_pt") or 0.0))
    # **候选线型的折线全部留着**，不按当次分组裁剪。分组 / 投票 / gate / plan
    # 现在都是读盘期算的（steps/linetypes/regroup.py），裁剪会让「改了分组口径
    # 之后新的胜出线型没有几何可画」。候选集本身有界（每个末端 top-K + 最近
    # 有主 op），实测一页几个到十几个，不是全页 60+ 个簇。
    # /api/page 仍然只发可见的那些，所以浏览器端的体积不受影响。
    line_types = [dict(row) for row in (payload.get("line_types") or [])]
    return {
        "sig": sig,
        "v": LINETYPE_VERSION,
        "engine": payload.get("engine") or {},
        "page": payload.get("page") or {},
        "line_types": line_types,
        "all_line_types": payload.get("all_line_types") or [],
        "bindings": bound["bindings"],
        "groups": bound["groups"],
        "used_all": bound["used_all"],
    }


def symbol_owners_of(symbol_result):
    """symbols.json 的 result → {symbol 下标: 它的 text_index}.

    放置锚要靠这张表归到「它所属图例那一行文字」的组里（与前端
    SCOPE_BY_SYMBOL 同一口径）。
    """
    owners = {}
    for index, symbol in enumerate((symbol_result or {}).get("symbols") or ()):
        owner = (symbol or {}).get("text_index")
        if isinstance(owner, int) and not isinstance(owner, bool):
            owners[index] = owner
    return owners


def resolve_visible(entry, plan_regions, items=None, symbol_owners=None):
    """读盘期的分组 / 投票 / gate / plan 闸；见 steps/linetypes/regroup.py。"""
    return regroup.resolve(entry or {}, plan_regions or [], items or [],
                           symbol_owners or {})


def page_payload(entry, plan_regions, items=None, symbol_owners=None):
    """/api/page 要挂的东西：只带可见线型的折线 + 逐末端绑定 + 分组投票。

    折线只取**被这个 callout 的末端指到的那条连通走线**。一个 global 线型可能
    在图上有好几条互不相连的走线（lenexa P4 的 #5 有 4 条，callout 只指其中一
    条的波浪围栏），也可能一条走线跨了好几个引擎 group（gladstone P4 的 #5 跨
    3 个 group，但那是同一道围栏的连续段，几何上贴着，必须一起画）。所以判据
    是几何连通，不是 group 号。没被指到的走线留在 dropped_runs 里报数量和
    bbox，不静默丢掉。
    """
    resolved = resolve_visible(entry, plan_regions, items, symbol_owners)
    visible = set(resolved["visible"])
    # 线型编号 → 该画的走线集合（同文多处指代时是各处的并集）。
    # RUN_GATE 默认**关**：画整个线型，与 TS 一致、不漏。
    wanted = {}
    for group in resolved["groups"]:
        number = group.get("visible_line_type_number")
        if number is None:
            continue
        wanted.setdefault(number, set()).update(group.get("engine_runs") or ())
    line_types = []
    for row in entry.get("line_types") or ():
        number = row.get("line_type_number")
        keep = dict(row)
        buckets = keep.pop("by_run", None) or {}
        if number not in visible:
            keep.pop("polylines", None)
            keep["hidden"] = True
            line_types.append(keep)
            continue
        want = wanted.get(number) or set()
        polylines, kept, dropped = [], [], []
        for rid, bucket in sorted(buckets.items()):
            if not RUN_GATE or rid in want:
                polylines.extend(bucket.get("polylines") or ())
                kept.append({"run_id": rid,
                             "segment_count": bucket.get("segment_count"),
                             "bbox": bucket.get("bbox")})
            else:
                dropped.append({"run_id": rid,
                                "segment_count": bucket.get("segment_count"),
                                "bbox": bucket.get("bbox")})
        if buckets:
            keep["polylines"] = polylines
            keep["segment_count"] = sum(max(0, len(line) - 1)
                                        for line in polylines)
            keep["kept_runs"] = kept
            keep["dropped_runs"] = dropped
        line_types.append(keep)
    return {
        "line_types": line_types,
        "bindings": resolved["bindings"],
        "groups": resolved["groups"],
        "visible": sorted(visible),
        "needs_recompute": resolved.get("needs_recompute") or [],
        "page": entry.get("page") or {},
        "engine": entry.get("engine") or {},
    }


def dumps_compact(entry):
    """落盘用的紧凑序列化（折线点很多，别让缩进把文件翻倍）。"""
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
