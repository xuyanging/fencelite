"""fence_lite 的 Flask 层(5060)：上传 PDF → 后台作业跑完整链 → 只读可视化.

这一层只做四件事：
  * 上传编排：收 PDF + 用户可编辑的「检测目标」→ job.create_project /
    job.start_job，返回 slug 供前端轮询；
  * 作业状态 / 费用的只读出口（/api/jobs、/api/job/<slug>、/api/cancel）；
  * 装配画廊（/api/overview）和单页的全部图层（/api/page）；
  * 页面底图的渲染缓存（/img）。

**这里绝不发起任何付费调用。** 所有 Gemini 调用都在上传作业内部跑完，网页端
只读 data/<slug>/ 里已经算好的缓存，所以 5051 那套「点一下才检测」的
GET/POST 付费接口（/api/symbols、/api/fenceline、/api/rescan）在这里完全
不存在；/api/page 一次把该页所有图层给全。

发布纪律（照抄 5051 的理由）：结果必须同时对上「当前 PDF 的 pdf_revision」
与「当期版本号」才发布。底图缓存是按 pdf_revision 建的，PDF 换了以后还把
旧的归一化框发出去，会把旧几何画到新渲染的图纸上 —— 看起来是对的，几何上
是错的。对不上就报 pending，让作业重跑。

Run:  venv\\Scripts\\python.exe webapp.py   →  http://127.0.0.1:5060/
"""
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import (Flask, abort, jsonify, make_response, render_template,
                   request, send_file)
from werkzeug.exceptions import HTTPException
from PIL import Image

import job
from core.config import BASE_DIR, MODEL_NAME, PRICING
from core.pdfio import render_pdf_page
from steps import arrows, linetypes, store
from steps.linetypes import refresh_state as linetype_refresh_state
from steps.placements import has_current_placements
from steps.symbols import (has_current_symbols, marker_code_indices,
                          symbols_dropped_view)
from steps.text.target import TARGET_DEFAULT, is_default_target
from steps.versions import FUSED_VERSION
from steps.views import (groups_need_classification, has_current_view_types,
                         merge_view_types, plan_boxes)

BASE_LONG = 5000          # base render long side (px) — sharp enough to zoom
JPEG_Q = 80

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True


# ------------------------------------------------------------- publish gates --
def _results_state(slug):
    """Return only project results that match the current PDF and schema.

    The base-image cache is keyed by the live PDF revision. Returning an old
    page record after that PDF changes would draw stale normalized boxes over
    a newly rendered sheet, which looks valid but is geometrically false.
    """
    if not store.is_valid_slug(slug):
        return None, "bad_slug"
    res = store.load_results(slug)
    if not isinstance(res, dict):
        return None, "missing"
    if res.get("fused_v") != FUSED_VERSION:
        return None, "fused_version"
    try:
        revision = store.pdf_revision(store.pdf_path(slug))
    except OSError:
        return None, "pdf_missing"
    if res.get("pdf_revision") != revision:
        return None, "pdf_revision"
    return res, None


def _results(slug):
    return _results_state(slug)[0]


def _fused_pending(reason):
    return jsonify({
        "pending": True,
        "stage": "fused",
        "reason": reason,
        "error": "fused results are stale; re-run the upload job",
    }), 409


def _symbols_publishable(entry, sig):
    """符号 + 放置是否当期可发布 —— 直接用写缓存那两步自己的判定函数.

    has_current_symbols 锚定 (items, pdf_revision) + 提示词/过滤版本，
    has_current_placements 锚定本地放置匹配版本。任何一项不当期都视作
    「这一页还没算」，返回空符号层，绝不把旧几何画到新底图上。
    """
    return bool(has_current_symbols(entry, sig)
                and has_current_placements(entry.get("result")))


def _typed_groups(slug, page, groups, revision):
    """Cache-only read.  Return typed groups, or ``None`` when pending."""
    if not groups_need_classification(groups):
        return merge_view_types(groups, None)
    entry = store.load_json(
        store.slug_dir(slug) / "view_types.json", {}).get(str(page))
    if not has_current_view_types(entry, groups, revision):
        return None
    return merge_view_types(groups, entry)


def _page_symbols(slug, page, items, revision):
    """一页的符号层：图例样例 symbol + plan 视图内的放置 + 免费的剥离解释.

    纯读盘：symbols.json（付费步的产物）+ view_types.json（分类步的产物）。
    分类不当期时照发分区框，但一个 plan 框都不给 —— 没有取景框就 fail-closed，
    绝不退化成「整页都算 plan」。
    """
    empty = {"groups": [], "symbols": []}
    if not items:
        return empty, [], []
    entry = store.load_json(
        store.slug_dir(slug) / "symbols.json", {}).get(str(page))
    if not _symbols_publishable(entry, store.sig_of(items, revision)):
        # 盘上明明有符号、只是版本戳对不上当前算法（改了算法/版本号之后就会这样）
        # —— 这种"陈旧"必须显式说出来。静默返回 0 会让人以为"模型没找到"，
        # 那是最误导的失败方式：真实案例里用户按着截图问"这条线怎么没框出来"，
        # 其实是这一页整个符号层被扣着没发。
        stale_count = len([s for s in ((entry or {}).get("result") or {})
                           .get("symbols") or [] if isinstance(s, dict)])
        if stale_count:
            return ({"groups": [], "symbols": [], "stale": True,
                     "stale_symbols": stale_count}, [], [])
        return empty, [], []
    result = entry["result"]
    raw_groups = result.get("groups") or []
    typed = _typed_groups(slug, page, raw_groups, revision)
    payload = dict(result)
    payload["symbols"] = [symbol for symbol in (result.get("symbols") or [])
                          if isinstance(symbol, dict)]
    if typed is None:
        # Only classifier-versioned view types may reach an API response.
        groups = merge_view_types(raw_groups, None)
        for group in groups:
            if group.get("kind") == "view":
                group.pop("view_type", None)
                group.pop("view_type_reason", None)
        payload["groups"] = groups
        payload["view_types_pending"] = True
        plans = []
    else:
        payload["groups"] = typed
        plans = plan_boxes(typed)
    # symbols_dropped_view 返回 {raw_symbols, dropped}；前端只要 dropped 那一层。
    dropped = symbols_dropped_view(entry) or {}
    return payload, list(dropped.get("dropped") or []), plans


def _placement_anchors_for(slug, page):
    """与 job._placement_anchors 同源：shape 放置也是箭头步的锚，必须进签名。"""
    result = ((store.load_json(store.slug_dir(slug) / "symbols.json", {})
               .get(str(page)) or {}).get("result") or {})
    anchors = []
    for si, symbol in enumerate(result.get("symbols") or []):
        for pi, box in enumerate(symbol.get("placements") or []):
            if isinstance(box, (list, tuple)) and len(box) == 4:
                anchors.append((f"s{si}:{pi}", list(box)))
    return anchors


def _attach_arrows(record, slug, page, items, revision, plan_regions=None):
    """箭头步的结果与**分层状态**.

    只挂结果是不够的：一个空结果可能是「这页确实没有引线」，也可能是任何一层
    根本没跑到（分类缺失 / 没有 plan 取景 / 锚全在 plan 外 / 边车算失败 /
    结果不当期）。这几种在界面上必须能区分，否则排查时无从下手 —— 之前
    正是因为 OOM 被写成空结果，一整页的缺失完全不可见。

    state 取值与含义：
      disabled          接缝没打开（ARROWS=1 未设）
      no-anchors        这页没有任何文字锚或放置锚，没有可处理的对象
      views-pending     步骤3 没跑到这页，没有分类就 fail-closed
      no-plan           分类跑了，但这页没有 plan 视图
      no-anchor-in-plan 有 plan 框，但没有任何锚完整落在里面
      not-run           该跑却没有结果文件条目
      failed            边车算失败（含走完堆梯子仍 OOM），detail 里有原因
      stale             有结果但签名/版本不当期，页面不会显示它
      image-only        只有嵌入图片、没有矢量路径；按规则标记后不追箭头
      ok                算过了；count 为找到箭头的锚数（0 表示确实没有）
    """
    if not arrows.ENABLED:
        record["arrows_status"] = {"state": "disabled"}
        return
    extra = _placement_anchors_for(slug, page)
    entry = store.load_json(
        store.slug_dir(slug) / "arrows.json", {}).get(str(page))
    sig = arrows.arrows_signature(items, revision, extra)

    if isinstance(entry, dict) and entry.get("error"):
        record["arrows_status"] = {"state": "failed",
                                   "detail": str(entry["error"])[:200]}
        return
    if arrows.has_current_arrows(entry, sig):
        record["arrows"] = entry["items"]
        record["arrow_anchors"] = entry.get("anchors") or {}
        if entry.get("page_kind") == "image-only":
            record["arrows_status"] = {
                "state": "image-only",
                "anchors": len(items) + len(extra),
                "detail": "this sheet has only an embedded image and no vector paths; arrow tracing was skipped by rule",
            }
            return
        record["arrows_status"] = {"state": "ok", "count": len(entry["items"]),
                                   "anchors": len(items) + len(extra)}
        return
    # 没有当期结果 —— 逐层说明卡在哪。
    if not items and not extra:
        # 这页压根没有可锚定的东西，不是任何一层"没跑"。
        record["arrows_status"] = {"state": "no-anchors"}
        return
    if plan_regions is None:
        record["arrows_status"] = {"state": "not-run"}
        return
    # 取景关闭时 plan 框与否不再决定跑不跑，所以不能拿它解释「为什么没有结果」。
    # 这时只剩两种可能：结果不当期（版本/签名变了）或压根没跑过。
    if not arrows.PLAN_GATE:
        record["arrows_status"] = {
            "state": "stale" if entry is not None else "not-run",
            "anchors": len(items) + len(extra)}
        return
    if not plan_regions:
        typed = store.load_json(
            store.slug_dir(slug) / "view_types.json", {}).get(str(page))
        record["arrows_status"] = {
            "state": "no-plan" if typed else "views-pending"}
        return
    anchors = [it.get("box_2d") for it in items] + [b for _k, b in extra]
    inside = sum(1 for b in anchors if arrows._box_inside_any(b, plan_regions))
    if not inside:
        record["arrows_status"] = {"state": "no-anchor-in-plan",
                                   "anchors": len(anchors)}
    elif entry is None:
        record["arrows_status"] = {"state": "not-run", "anchors": inside}
    else:
        record["arrows_status"] = {"state": "stale", "anchors": inside}


def _attach_linetypes(record, slug, page, items, revision, plan_regions=None):
    """线型层的结果与**分层状态**（只读盘，零模型调用）.

    plan 在这里、而且只在这里起作用：它是**显示闸**，不进缓存签名。所以步骤3
    重分类或 VIEW_VERSION bump 会立刻改变可见范围，却不会作废一页 80 秒的聚类。
    闸门锚在末端 tip 而不是 callout 的文字框 —— callout 几乎都在图纸边缘 /
    明细表里，只有引线伸进俯视图，按文字框卡会把绝大多数正确绑定误杀。

    state 取值：
      disabled            接缝没打开（LINETYPES=1 未设）
      no-arrows           箭头层不当期，或这页没有任何末端 —— 没有可绑的对象
      not-run             该跑却没有结果条目
      failed              边车算失败，detail 里有原因
      updating            结果缺失或签名/版本不当期，后台正排队重算当前版本
      hidden-no-plan      算过了，但这页没有 plan 框 → 一条都不显示
      hidden-outside-plan 有 plan 框，但没有末端落在里面（详图页的常态）
      all-gate            这页的 callout 全是 gate —— gate 不找线，不是失败
      no-line-type        末端在 plan 内、但它指的那段 ink 不属于任何线型
                          （residual / 太远 / 组内答案够不着），也就是"这里
                          确实没有可高亮的线型"，与"没算过"和"被 plan 挡住"
                          是三件不同的事
      ok                  有可见线型；visible 是被指到的线型编号

    这几个「空」必须分得清：真实案例 civil_ifb_167263 P3 的末端**就在 plan
    内**（in_plan=1），旧代码却报 hidden-outside-plan —— 照着那个状态去查
    plan 分类是白费功夫，真实原因是 tip 底下 0.012 pt 那条 op 是 residual。
    """
    if not linetypes.ENABLED:
        record["linetypes_status"] = {"state": "disabled"}
        return
    extra = _placement_anchors_for(slug, page)
    arrows_sig = arrows.arrows_signature(items, revision, extra)
    arrow_entry = store.load_json(
        store.slug_dir(slug) / "arrows.json", {}).get(str(page))
    if not arrows.has_current_arrows(arrow_entry, arrows_sig):
        record["linetypes_status"] = {"state": "no-arrows"}
        return
    anchors = linetypes.anchors_of(arrow_entry)
    if not anchors:
        record["linetypes_status"] = {"state": "no-arrows"}
        return
    sig = linetypes.linetypes_signature(arrows_sig)
    entry = linetypes.load_page(slug, page)
    # Signature mismatch is never publishable, including an error produced by
    # an older engine.  The low-priority refresh worker will replace it
    # atomically; keep text/arrows visible and tell the browser to poll instead
    # of silently presenting an empty FENCELINE section.
    signature_matches = bool(
        isinstance(entry, dict) and entry.get("sig") == sig)
    retry_with_larger_budget = bool(
        signature_matches and entry.get("error")
        and job._linetype_failure_budget_increased(
            slug, page, arrow_entry, entry.get("error")))
    if signature_matches and entry.get("error") \
            and not retry_with_larger_budget:
        record["linetypes_status"] = {"state": "failed",
                                     "detail": str(entry["error"])[:200]}
        return
    if (not signature_matches
            or not linetypes.has_current_linetypes(entry, sig)):
        refresh = linetype_refresh_state.page_refresh_status(slug, page)
        detail = {
            "running": "Current line-type engine is refreshing this sheet",
            "waiting": "Queued; a foreground upload/rerun has priority",
            "queued": "Queued for the current line-type engine",
        }.get(refresh, "Queued for the next automatic line-type refresh scan")
        record["linetypes_status"] = {
            "state": "updating", "refresh": refresh or "queued",
            "detail": detail, "targets": len(anchors)}
        return

    owners = linetypes.symbol_owners_of(
        (store.load_json(store.slug_dir(slug) / "symbols.json", {})
         .get(str(page)) or {}).get("result") or {})
    payload = linetypes.page_payload(entry, plan_regions or [], items, owners)
    record["linetypes"] = payload
    page_info = payload.get("page") or {}
    status = {"targets": len(anchors),
              "bound": len(entry.get("used_all") or ()),
              "clusters": page_info.get("line_types"),
              "residual_ops": page_info.get("residual_ops"),
              "seconds": page_info.get("seconds_cluster")}
    if payload.get("needs_recompute"):
        # 旧缓存按当时的分组裁剪过折线，换了分组口径之后新的胜出线型没有几何
        # 可画。显式说出来 —— 静默画不出线是最误导的失败方式。
        status["state"] = "needs-recompute"
        status["needs_recompute"] = payload["needs_recompute"]
    elif payload["visible"]:
        status["state"] = "ok"
        status["visible"] = payload["visible"]
    else:
        groups = payload.get("groups") or []
        fence = [g for g in groups if g.get("scope") != "gate"]
        # 逐层判：先看有没有非 gate 的组，再看它们的末端在不在 plan 内，
        # 最后才是"在里面但那段 ink 不属于任何线型"。顺序反了就会像旧代码
        # 那样把 residual 报成 plan 问题。
        inside = sum(int(g.get("in_plan_count") or 0) for g in fence)
        by_state = {}
        for row in payload.get("bindings") or ():
            if row.get("scope") == "gate":
                continue
            by_state[row.get("state")] = by_state.get(row.get("state"), 0) + 1
        status["binding_states"] = by_state
        if groups and not fence:
            status["state"] = "all-gate"
        elif not plan_regions:
            status["state"] = "hidden-no-plan"
        elif not inside:
            status["state"] = "hidden-outside-plan"
        else:
            status["state"] = "no-line-type"
            status["in_plan_targets"] = inside
    record["linetypes_status"] = status


# 上传体积上限。上传路径会把整份 PDF 同时持有在内存里（read() 一份 +
# write_bytes 一份），所以这个数字直接决定最小机器规格。可用环境变量调，
# 反代（Caddy 的 request_body max_size / nginx client_max_body_size）必须配同一个数，
# 否则先被反代截断，用户拿到的是反代的 HTML 错误页而不是这里的 JSON。
MAX_UPLOAD_MB = int(os.environ.get("FENCE_MAX_UPLOAD_MB", "512"))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.errorhandler(413)
def _too_large(_e):
    """Flask 默认的 413 是 HTML。前端只按 JSON 解 error 字段，不转的话
    用户看到的是「解析失败」而不是「文件太大」。"""
    return jsonify({"error": f"PDF too large (limit {MAX_UPLOAD_MB} MB)"}), 413


def _cross_site_write_blocked():
    """Reject browser cross-site writes while keeping CLI/test clients usable."""
    if request.headers.get("Sec-Fetch-Site", "").lower() == "cross-site":
        return True
    origin = request.headers.get("Origin")
    return bool(origin and origin.rstrip("/") != request.host_url.rstrip("/"))


def _empty_rec():
    """Placeholder record for a sheet with no detected text box, so the viewer
    can still display the bare page (default view) or hide it (toggle)."""
    return {"vlm_items": [], "vec_added": [], "vec_covered": [],
            "has_text": None, "vlm_error": None, "empty": True}


# ------------------------------------------------------------------- base img --
def _base_img_path(slug, page):
    return (store.slug_dir(slug)
            / f"base_P{page}_{store.pdf_revision(store.pdf_path(slug))}.jpg")


def _drop_stale_base(slug, page, keep):
    """Delete this page's base images left over from earlier pdf_revisions.

    ``pdf_revision`` carries the PDF's mtime, so re-uploading the same drawing
    renames every base image and nothing ever removed the old ones.  Only files
    for *this* page whose name is not the current one are touched -- no code
    path can reach those, since the filename is always derived from the current
    revision."""
    try:
        for old in store.slug_dir(slug).glob(f"base_P{page}_*.jpg"):
            if old.name != keep.name:
                old.unlink(missing_ok=True)
    except OSError:
        pass            # a reader may still hold the handle on Windows; retry next time


def _ensure_base(slug, page):
    """Render + cache the page base image (render_pdf_page owns FITZ_LOCK)."""
    pdf = store.pdf_path(slug)
    if not pdf.is_file():
        # Do this BEFORE _base_img_path: slug_dir() creates the directory, so an
        # unknown slug used to leave an empty data/<slug>/ behind and then 500.
        abort(404)
    f = _base_img_path(slug, page)
    if f.exists():
        with Image.open(f) as im:
            return f, im.size
    img = render_pdf_page(pdf, page - 1, dpi=600, max_px=BASE_LONG)
    # Write to a temp name and rename into place.  A direct save leaves a
    # permanently truncated file if the process dies mid-write (f.exists() is
    # then true forever), and lets a concurrent reader send half a JPEG with a
    # one-day Cache-Control on it.
    tmp = f.with_name(f"{f.name}.{os.getpid()}.tmp")
    try:
        img.save(tmp, "JPEG", quality=JPEG_Q, optimize=True)
        os.replace(tmp, f)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
    _drop_stale_base(slug, page, f)
    return f, img.size


# ------------------------------------------------------------------- gallery --
def _overview_row(res, slug):
    """One gallery row.  ``pages`` lists EVERY sheet 1..page_count — sheets
    with no detected text box get a zero-count placeholder so the frontend can
    show them by default and hide them with the "只看有结果的页" toggle."""
    page_count = int(res.get("page_count") or 0)
    have = res.get("pages", {}) or {}
    try:
        revision = store.pdf_revision(store.pdf_path(slug))
    except OSError:
        revision = None
    directory = store.slug_dir(slug)
    symbol_cache = store.load_json(directory / "symbols.json", {})
    arrow_cache = store.load_json(directory / "arrows.json", {})
    pages = []
    tot_added = tot_vlm = tot_cov = tot_sym = tot_plc = tot_stale = 0
    for p in range(1, page_count + 1):
        rec = have.get(str(p))
        if rec is None:
            pages.append({"page": p, "added": 0, "vlm": 0, "covered": 0,
                          "sym": 0, "plc": 0, "err": None, "has_text": None,
                          "present": False})
            continue
        raw_a = len(rec.get("vec_added", []))
        raw_v = len(rec.get("vlm_items", []))
        a, v = raw_a, raw_v
        c = len(rec.get("vec_covered", []))
        items = store.items_of(rec)
        entry = symbol_cache.get(str(p)) if revision else None
        symbols = []
        if items and revision \
                and _symbols_publishable(entry, store.sig_of(items, revision)):
            symbols = [s for s in (entry["result"].get("symbols") or [])
                       if isinstance(s, dict)]
        elif (entry or {}).get("result", {}).get("symbols"):
            # 有结果但版本戳不当期：算进 stale 计数，让画廊显示「符号待重跑」
            # 而不是让这一页看起来"什么都没找到"。
            tot_stale += len([s for s in entry["result"]["symbols"]
                              if isinstance(s, dict)])
        placements = sum(len(s.get("placements") or []) for s in symbols)
        raw_result = (entry or {}).get("result") or {}
        arrow_extra = []
        for si, symbol in enumerate(raw_result.get("symbols") or []):
            for pi, box in enumerate(symbol.get("placements") or []):
                if isinstance(box, (list, tuple)) and len(box) == 4:
                    arrow_extra.append((f"s{si}:{pi}", list(box)))
        arrow_entry = arrow_cache.get(str(p)) if revision else None
        arrow_current = bool(
            revision and arrow_entry and arrows.has_current_arrows(
                arrow_entry, arrows.arrows_signature(items, revision, arrow_extra)))
        image_only = bool(arrow_current
                          and arrow_entry.get("page_kind") == "image-only")
        if arrow_current:
            hidden = arrows.suppressed_unverified_duplicates(
                items, arrow_entry.get("anchors"))
            v -= sum(1 for index in hidden if index < raw_v)
            a -= sum(1 for index in hidden if index >= raw_v)
        tot_added += a; tot_vlm += v; tot_cov += c
        tot_sym += len(symbols); tot_plc += placements
        pages.append({"page": p, "added": a, "vlm": v, "covered": c,
                      "sym": len(symbols), "plc": placements,
                      "err": rec.get("vlm_error"),
                      "has_text": bool(rec.get("has_text")),
                      "image_only": image_only,
                      "present": True})
    status = job.get_job(slug) or {}
    # Prefer the live/session job stats; fall back to the summary persisted in
    # results.json so cost/time survive a server restart.
    llm = status.get("llm") or res.get("llm_summary")
    wall = status.get("wall_seconds") or res.get("wall_seconds")
    return {"slug": res.get("slug", slug), "page_count": page_count,
            "no_text_layer": bool(res.get("no_text_layer")),
            "mode": res.get("mode", "fence"),
            "target": res.get("target"),
            "generated": res.get("generated"),
            "vlm": tot_vlm, "added": tot_added, "covered": tot_cov,
            "symbols": tot_sym, "placements": tot_plc,
            "llm": llm, "wall_seconds": wall,
            # 结果是从别的服务搬进来的（旧流水线跑的付费缓存），还是这套代码
            # 自己跑出来的？前端据此打标，用户想确认就点「重新跑」。
            "imported": res.get("imported") or None,
            # 盘上有符号但算法版本戳不当期 → 画廊打「符号待重跑」，
            # 否则「符号 0」会被读成「这份图纸没有图例符号」
            "stale_symbols": tot_stale,
            "model": job.run_model(slug, res),
            "variant_of": job.variant_base(slug),
            "pages": pages}


def _processing_row(slug, status):
    """Gallery row for a project whose results are not ready yet (uploading /
    processing / failed).  Its PDF exists, so every page is previewable now."""
    pc = int(status.get("pages_total") or job.page_count_of(slug) or 0)
    pages = [{"page": p, "added": 0, "vlm": 0, "covered": 0, "sym": 0,
              "plc": 0, "err": None, "has_text": None, "present": False}
             for p in range(1, pc + 1)]
    failed = bool(status.get("done") and not status.get("ok"))
    return {"slug": slug, "page_count": pc, "no_text_layer": False,
            "mode": status.get("mode", "fence"),
            "target": status.get("target"), "generated": None,
            "vlm": 0, "added": 0, "covered": 0, "symbols": 0, "placements": 0,
            "llm": status.get("llm"),
            "wall_seconds": status.get("wall_seconds"),
            "processing": not status.get("done"),
            "job_error": status.get("error") if failed else None,
            "model": status.get("model") or job.run_model(slug),
            "variant_of": job.variant_base(slug),
            "pages": pages}


# -------------------------------------------------------------------- routes --
@app.route("/")
def index():
    # 禁缓存。模板这边开了 TEMPLATES_AUTO_RELOAD（第 47-48 行），改完立刻重渲染，
    # 但**响应上一个缓存头都没有**时浏览器会按启发式规则自行缓存整份 HTML ——
    # 而这一页的 JS 是内联的，于是前端改动看起来"没生效"，实际是压根没取到。
    # 实测就这么误判过一次：服务端已经返回新代码，界面还是旧行为。
    response = make_response(render_template("index.html"))
    response.headers["Cache-Control"] = "no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/favicon.ico")
def favicon():
    return ("", 204)


@app.route("/api/preset_target")
def preset_target():
    """The default, user-editable detection TARGET (what to find), shown
    pre-filled in the upload dialog.  Only this part is editable — the JSON
    output-format scaffolding is fixed server-side and never exposed, so an
    edit can change WHAT is detected but never break the data format.  The one
    target drives both the image VLM and the vector-text judge."""
    return jsonify({"target": TARGET_DEFAULT})


@app.route("/api/overview")
def overview():
    out = []
    seen = set()
    for d in sorted(store.DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        slug = d.name
        # A still-running job shows as "processing" (bare pages, no partial
        # results) even if a partial results.json already exists — results only
        # surface once the whole run finishes.
        if job.job_running(slug):
            out.append(_processing_row(slug, job.get_job(slug) or {}))
            seen.add(slug)
            continue
        res = _results(slug)
        if not res:
            continue
        out.append(_overview_row(res, slug))
        seen.add(slug)
    # projects still processing with no results.json yet — previewable now.
    for status in job.all_jobs():
        slug = status.get("slug")
        if slug in seen or not store.is_valid_slug(slug):
            continue
        if not store.pdf_path(slug).exists():
            continue
        out.append(_processing_row(slug, status))
    out.sort(key=lambda r: r["slug"])
    return jsonify(out)


@app.route("/api/jobs")
def jobs():
    """All processing jobs (running + finished this session) for the gallery
    progress cards."""
    return jsonify(job.all_jobs())


@app.route("/api/job/<slug>")
def job_status(slug):
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    status = job.get_job(slug)
    if status is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(status)


@app.route("/api/cancel/<slug>", methods=["POST"])
def cancel_job(slug):
    """Request cooperative cancellation of a running job (user got tired of
    waiting on a big PDF).  Stages stop scheduling new pages and the run
    unwinds; a few in-flight model calls may still finish."""
    if _cross_site_write_blocked():
        return jsonify({"error": "cross-site write rejected"}), 403
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    if job.get_job(slug) is None:
        return jsonify({"error": "job not found"}), 404
    result = job.request_cancel(slug)
    return jsonify({"ok": True, "was_running": bool(result.get("was_running")),
                    "job": result.get("job")})


_UPLOAD_TOKEN_LOCKS = {}
_UPLOAD_TOKEN_LOCKS_GUARD = threading.Lock()


def _valid_upload_token(token):
    """A browser-generated opaque id used only as a safe path component."""
    return (16 <= len(token) <= 128
            and all(c.isalnum() or c in "-_" for c in token))


def _upload_token_path(token):
    return store.JOBS_DIR / ".upload-tokens" / f"{token}.json"


def _upload_token_lock_path(token):
    return store.JOBS_DIR / ".upload-tokens" / f"{token}.lock"


@contextmanager
def _upload_token_lock(token):
    """Serialize one idempotency key across threads and gunicorn workers.

    The lock file is stable and is never unlinked, so stale-marker recovery
    cannot suffer an unlink/recreate ABA race.  ``flock`` is released by the
    kernel on process death; the in-process mutex is the portable fallback.
    """
    lock_path = _upload_token_lock_path(token)
    key = str(lock_path.absolute())
    with _UPLOAD_TOKEN_LOCKS_GUARD:
        local_lock = _UPLOAD_TOKEN_LOCKS.setdefault(key, threading.Lock())
    with local_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as handle:
            try:
                import fcntl                                  # noqa: PLC0415
            except ImportError:
                yield
                return
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_upload_token_dir(path):
    """Make marker create/replace/unlink durable across a machine crash."""
    fd = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        fd = os.open(path.parent, flags)
        os.fsync(fd)
    except OSError:
        pass
    finally:
        if fd is not None:
            os.close(fd)


def _save_upload_token(path, payload):
    store.save_json(path, payload)
    _fsync_upload_token_dir(path)


def _release_upload_token(path):
    if path is None:
        return
    try:
        path.unlink()
    except OSError:
        return
    _fsync_upload_token_dir(path)


def _upload_target_identity(target):
    return "" if is_default_target(target) else (target or "").strip()


def _upload_request_identity(filename, size, target):
    clean_name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return {"filename": clean_name, "size": int(size),
            "target": _upload_target_identity(target)}


def _token_project(marker):
    """Return ``(slug, status)`` only while marker still owns that PDF."""
    slug = marker.get("slug") if isinstance(marker, dict) else None
    if not isinstance(slug, str) or not store.is_valid_slug(slug):
        return None, None
    pdf = store.pdf_path(slug)
    try:
        current = pdf.exists() and store.pdf_revision(pdf)
    except OSError:
        current = None
    if not current or current != marker.get("pdf_revision"):
        return None, None
    status = (job.get_job(slug)
              or store.load_json(store.JOBS_DIR / f"{slug}.json", None)
              or marker.get("job"))
    return slug, status if isinstance(status, dict) else None


def _finish_upload_marker(path, token, identity, slug, target, status):
    revision = store.pdf_revision(store.pdf_path(slug))
    _save_upload_token(path, {
        "token": token, "state": "started", "request": identity,
        "slug": slug, "pdf_revision": revision, "target": target,
        "job": status, "updated_at": time.time(),
    })


def _replay_or_recover_upload(path, token, identity, target):
    """Return a prior/recovered response, or None to create a fresh project.

    Caller holds ``_upload_token_lock(token)``.  Therefore a ``created``
    marker with no job can only be residue from a dead owner and is safe to
    resume exactly once.
    """
    marker = store.load_json(path, None)
    if not isinstance(marker, dict):
        _release_upload_token(path)
        return None
    previous_identity = marker.get("request")
    if previous_identity != identity:
        return ({"error": "upload token was already used for a different "
                           "filename, file size, or detection target"}, 409)
    slug, status = _token_project(marker)
    if slug and status:
        try:
            _finish_upload_marker(path, token, identity, slug, target, status)
        except Exception as exc:                              # noqa: BLE001
            print(f"[upload] could not refresh token {token}: {exc}", flush=True)
        return ({"slug": slug, "job": status, "deduplicated": True}, 200)
    if slug and marker.get("state") == "created":
        # Source PDF was committed but the former process died before its
        # worker was registered.  Recover in place; never upload/pay twice.
        try:
            status = job.start_job(slug, target=target)
        except job.JobStartError as exc:
            try:
                job.delete_project(slug, cascade=False)
            except Exception:                                 # noqa: BLE001
                pass
            _release_upload_token(path)
            return ({"error": f"{type(exc).__name__}: {exc}"}, 500)
        except Exception as exc:                              # noqa: BLE001
            return ({"error": f"{type(exc).__name__}: {exc}"}, 500)
        try:
            _finish_upload_marker(path, token, identity, slug, target, status)
        except Exception as exc:                              # noqa: BLE001
            print(f"[upload] could not finalize recovered token {token}: {exc}",
                  flush=True)
        return ({"slug": slug, "job": status, "deduplicated": True,
                 "recovered": True}, 200)
    # A stable lock proves no live request still owns a receiving marker.
    # A missing/replaced PDF similarly means this token no longer owns a job.
    _release_upload_token(path)
    return None


def _create_uploaded_project(stream, filename, target, token=None,
                             token_path=None, identity=None):
    slug = None
    try:
        slug = job.create_project_stream(stream, filename)
        if token_path is not None:
            revision = store.pdf_revision(store.pdf_path(slug))
            _save_upload_token(token_path, {
                "token": token, "state": "created", "request": identity,
                "slug": slug, "pdf_revision": revision, "target": target,
                "updated_at": time.time(),
            })
    except Exception as exc:                                  # noqa: BLE001
        if slug:
            try:
                job.delete_project(slug, cascade=False)
            except Exception:                                 # noqa: BLE001
                pass
        _release_upload_token(token_path)
        return ({"error": f"{type(exc).__name__}: {exc}"}, 500)
    try:
        status = job.start_job(slug, target=target)
    except job.JobStartError as exc:
        try:
            job.delete_project(slug, cascade=False)
        except Exception:                                     # noqa: BLE001
            pass
        _release_upload_token(token_path)
        return ({"error": f"{type(exc).__name__}: {exc}"}, 500)
    except Exception as exc:                                  # noqa: BLE001
        # Leave the created marker and exact source in place. A retry under
        # the stable token lock will inspect/recover it without duplicating.
        return ({"error": f"{type(exc).__name__}: {exc}"}, 500)
    if token_path is not None:
        try:
            _finish_upload_marker(
                token_path, token, identity, slug, target, status)
        except Exception as exc:                              # noqa: BLE001
            # The task is live. Returning success avoids inviting a duplicate;
            # its created marker can be finalized by any later replay.
            print(f"[upload] could not finalize token {token}: {exc}", flush=True)
    return ({"slug": slug, "job": status}, 200)


@app.route("/api/upload", methods=["POST"])
def upload():
    """Accept a PDF + optional edited detection target, start background
    processing, and return the new slug for the gallery to poll."""
    if _cross_site_write_blocked():
        return jsonify({"error": "cross-site write rejected"}), 403
    f = request.files.get("pdf")
    if f is None or not f.filename:
        return jsonify({"error": "no PDF uploaded (field 'pdf')"}), 400
    stream = f.stream
    try:
        header = stream.read(5)
        stream.seek(0, os.SEEK_END)
        file_size = stream.tell()
        stream.seek(0)
    except (AttributeError, OSError, TypeError):
        return jsonify({"error": "uploaded PDF stream is not readable"}), 400
    if header != b"%PDF-":
        return jsonify({"error": "not a PDF file"}), 400
    # editable detection target ("what to find")
    target = request.form.get("target", "")
    upload_token = request.form.get("upload_token", "").strip()
    if not upload_token:
        payload, code = _create_uploaded_project(stream, f.filename, target)
        return jsonify(payload), code
    if not _valid_upload_token(upload_token):
        return jsonify({"error": "invalid upload token"}), 400
    identity = _upload_request_identity(f.filename, file_size, target)
    token_path = _upload_token_path(upload_token)
    try:
        with _upload_token_lock(upload_token):
            replay = _replay_or_recover_upload(
                token_path, upload_token, identity, target)
            if replay is not None:
                payload, code = replay
                return jsonify(payload), code
            _save_upload_token(token_path, {
                "token": upload_token, "state": "receiving",
                "request": identity, "updated_at": time.time(),
            })
            payload, code = _create_uploaded_project(
                stream, f.filename, target, token=upload_token,
                token_path=token_path, identity=identity)
            return jsonify(payload), code
    except Exception as exc:                                  # noqa: BLE001
        # This catches lock/marker I/O failures before a project can be safely
        # associated with the token. Never turn them into an HTML 500.
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/rerun/<slug>", methods=["POST"])
def rerun(slug):
    """Re-run an existing PDF.

    ``reset`` (default true) wipes every cached artifact for the project first,
    so the new run cannot be shaped by anything an earlier/older pipeline left
    on disk — the only survivor is projects/<slug>/input.pdf.  Pass
    ``{"reset": false}`` to reuse whatever is still current (cheap, but then
    the result is only as new as the cache).
    """
    if _cross_site_write_blocked():
        return jsonify({"error": "cross-site write rejected"}), 403
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    if not store.pdf_path(slug).exists():
        return jsonify({"error": "project not found"}), 404
    body = request.get_json(silent=True) or {}
    target = body.get("target")
    if target is None:
        target = job.stored_target(slug)
    # A comparison run must re-run on ITS OWN model. Without this the job falls
    # back to the process default: every model-gated cache (symbols' stored
    # model, vlm_identity, the judge cache) would then read as stale, and the
    # re-run would re-pay on the default model AND overwrite the variant's
    # results — silently destroying the comparison it exists for.
    model = body.get("model")
    if model is None:
        model = job.variant_model(slug)
    if model is not None and model not in PRICING:
        return jsonify({"error": f"unknown model: {model}"}), 400
    try:
        status, cleared = job.restart_job(
            slug, target=target, model=model,
            reset=body.get("reset", True))
    except job.JobStartError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:                                  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 409
    return jsonify({"slug": slug, "cleared": cleared, "model": model,
                    "job": status})


@app.route("/api/models")
def models_list():
    """Model registry for the compare switch, grouped by provider."""
    out = []
    for mid, p in PRICING.items():
        out.append({
            "id": mid,
            "display": p.get("display", mid),
            "note": p.get("note", ""),
            "provider": p.get("provider", "gemini"),
            "in_low": p["in_low"],
            "in_high": p.get("in_high", p["in_low"]),
            "out_low": p["out_low"],
            "out_high": p.get("out_high", p["out_low"]),
            "tiered": not p.get("flat", False),
        })
    return jsonify({"default": MODEL_NAME, "models": out})


@app.route("/api/variant/<slug>", methods=["POST"])
def make_variant(slug):
    """Run the same PDF under a different model, as a sibling project.

    This is the compare switch's write half: the base project's results are
    never touched, the variant gets its own cache directory, and the gallery
    then shows both so they can be flipped between.
    """
    if _cross_site_write_blocked():
        return jsonify({"error": "cross-site write rejected"}), 403
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    if not store.pdf_path(slug).exists():
        return jsonify({"error": "project not found"}), 404
    body = request.get_json(silent=True) or {}
    model = body.get("model")
    if model not in PRICING:
        return jsonify({"error": f"unknown model: {model}"}), 400
    target = body.get("target")
    if target is None:
        target = job.stored_target(slug)   # compare like-for-like
    try:
        vslug, status, cleared = job.start_variant(
            slug, model, target=target)
    except job.JobStartError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as exc:                                  # noqa: BLE001
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
    return jsonify({"slug": vslug, "base": slug, "model": model,
                    "cleared": cleared, "job": status})


@app.route("/api/project/<slug>", methods=["DELETE"])
def delete_project(slug):
    if _cross_site_write_blocked():
        return jsonify({"error": "cross-site write rejected"}), 403
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    try:
        removed = job.delete_project(slug)
    except job.JobStartError as exc:
        return jsonify({"error": str(exc)}), 409
    except Exception as e:                                     # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500
    return jsonify({"ok": True, "removed": removed})


@app.route("/api/page/<slug>/<int:page>")
def page_data(slug, page):
    """该页的全部图层，一次给全（全部来自缓存，零模型调用）。"""
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    res, stale_reason = _results_state(slug)
    if stale_reason not in (None, "missing"):
        return _fused_pending(stale_reason)
    # While a job is still running, only preview the bare sheet — do NOT surface
    # partial results.  Treat it the same as "no results yet" even if a partial
    # results.json already exists.
    if job.job_running(slug):
        res = None
    # Preview mode: results not produced yet (still processing / failed) but the
    # PDF is on disk — render the sheet with an empty overlay so it is viewable.
    if res is None:
        if not store.pdf_path(slug).exists():
            return jsonify({"error": "page not found"}), 404
        pc = job.page_count_of(slug)
        if not (1 <= page <= pc):
            return jsonify({"error": "page not found"}), 404
        try:
            _, (w, h) = _ensure_base(slug, page)
        except Exception as e:                              # noqa: BLE001
            return jsonify({"error": f"render failed: {e}"}), 500
        return jsonify({"page": page, "page_count": pc, "mode": "fence",
                        "w": w, "h": h, "img": f"/img/{slug}/{page}",
                        "record": _empty_rec(), "items": [],
                        "symbols": {"groups": [], "symbols": []},
                        "dropped_symbols": [], "plan_boxes": [],
                        "counts": {"text": 0, "symbols": 0, "placements": 0,
                                   "plan_groups": 0},
                        "processing": True})
    page_count = int(res.get("page_count") or 0)
    # A page absent from results.json but within the PDF is a legitimately
    # empty sheet — synthesize a blank record instead of 404 so it is viewable.
    if res.get("pages", {}).get(str(page)) is None \
            and not 1 <= page <= page_count:
        return jsonify({"error": "page not found"}), 404
    try:
        _, (w, h) = _ensure_base(slug, page)
    except Exception as e:                                  # noqa: BLE001
        return jsonify({"error": f"render failed: {e}"}), 500
    # A reset=false rerun can begin while the base image is rendering and leave
    # the previous results.json intact.  Rechecking only that file would then
    # publish the old overlays during an active run.  Match the entry check and
    # return the same bare-sheet preview until the new job finishes.
    if job.job_running(slug):
        pc = job.page_count_of(slug)
        if not (1 <= page <= pc):
            return jsonify({"error": "page not found"}), 404
        return jsonify({"page": page, "page_count": pc, "mode": "fence",
                        "w": w, "h": h, "img": f"/img/{slug}/{page}",
                        "record": _empty_rec(), "items": [],
                        "symbols": {"groups": [], "symbols": []},
                        "dropped_symbols": [], "plan_boxes": [],
                        "counts": {"text": 0, "symbols": 0, "placements": 0,
                                   "plan_groups": 0},
                        "processing": True})
    # Re-check after rendering: the PDF or the results cache can be replaced
    # while this request is producing the base image.
    res, stale_reason = _results_state(slug)
    if stale_reason is not None:
        return _fused_pending(stale_reason)
    page_count = int(res.get("page_count") or 0)
    rec = res.get("pages", {}).get(str(page))
    if rec is None:
        if not 1 <= page <= page_count:
            return jsonify({"error": "page not found"}), 404
        rec = _empty_rec()
    record = dict(rec)
    items = store.items_of(record)
    revision = res.get("pdf_revision")
    symbols, dropped, plan_boxes = ({"groups": [], "symbols": []}, [], [])
    plan_boxes_now = []
    if res.get("mode", "fence") == "fence":
        # 图例符号 / plan 放置是 fence 专属的；自定义检测目标只有文字层。
        symbols, dropped, plan_boxes = _page_symbols(
            slug, page, items, revision)
        plan_boxes_now = plan_boxes
    _attach_arrows(record, slug, page, items, revision, plan_boxes_now)
    # 线型层跟在箭头之后：它绑的就是箭头末端。plan 在这里只当显示闸。
    _attach_linetypes(record, slug, page, items, revision, plan_boxes_now)
    suppressed_items = arrows.suppressed_unverified_duplicates(
        items, record.get("arrow_anchors"))
    placements = sum(len(s.get("placements") or [])
                     for s in symbols["symbols"])
    marker_codes = marker_code_indices(items, symbols.get("groups") or [],
                                       symbols.get("symbols") or [])
    # 图例行文字框把行首编码（"3G  3'-0\" WIDE GATE..."）含在里面时，套用
    # 放置阶段算好的裁剪：文字框只该框文字，不该框住那个标记。裁剪结果单独
    # 存表、不改 results.json 里的 item —— 动 item 就要重跑步骤② 的付费推理。
    for key, trimmed in (symbols.get("text_trim") or {}).items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(items) and isinstance(trimmed, list) \
                and len(trimmed) == 4:
            items[index] = {**items[index], "box_2d": trimmed,
                            "box_raw": items[index].get("box_2d")}
    return jsonify({"page": page, "page_count": page_count,
                    "mode": res.get("mode", "fence"),
                    "w": w, "h": h, "img": f"/img/{slug}/{page}",
                    "record": record, "items": items,
                    "symbols": symbols, "dropped_symbols": dropped,
                    "plan_boxes": plan_boxes,
                    # Preserve raw union indices/caches; the default viewer
                    # hides only weak duplicate members rejected by arrow/text
                    # ownership validation. Debug mode still renders them.
                    "suppressed_items": sorted(suppressed_items),
                    # 其实是图例编码标记、不是独立 fence 文字的那些 item 下标。
                    # 只标记不删除：删 item 会让步骤② 的付费缓存签名失效（每页
                    # 重新付费），还会让 union index 错位。前端默认不把它们画进
                    # 文字层，Debug 里单独一层看。
                    "marker_codes": sorted(marker_codes),
                    # 盘上有符号但版本戳不是当期 → 显式说「要重跑」，
                    # 不要让前端把它显示成「没找到」
                    "symbols_stale": bool(symbols.get("stale")),
                    "stale_symbols": int(symbols.get("stale_symbols") or 0),
                    "counts": {"text": len(items)
                               - len(marker_codes | suppressed_items),
                               "marker_codes": len(marker_codes),
                               "symbols": len(symbols["symbols"]),
                               "placements": placements,
                               "plan_groups": len(plan_boxes)}})


@app.route("/api/linetypes_all/<slug>/<int:page>")
def linetypes_all(slug, page):
    """调试视图：这一页**全部**线型的几何 + residual ink（前端按需拉）.

    为什么单独一个接口而不挂进 /api/page：正常视图只发被指到的那几个线型，
    这是它能上前端的前提。全部线型是它的 1.8 倍（rapid_city_2 P11 实测
    6.3 → 11.2 MB），塞进每次页面加载就是给所有人付调试的代价。

    它回答的是「这个 callout 没找到线，到底是没聚出来还是聚出来了没被选中」——
    正常视图里这两种原因看起来一模一样。所以除了全部线型，还发 residual：
    不属于任何线型的 path ink。residual 里有那条线 = 没聚出来；某个线型盖着
    那条线但末端没绑上 = 聚出来了没被选中。

    state：
      disabled      接缝没打开
      no-arrows     箭头层不当期 / 这页没有末端 —— 主结果本身就没算
      not-run       没有 .all.json（跑 tools/linetype_sidecar/
                    verify_all_geometry.py 补）
      stale         .all.json 的 sig 与当期主结果不符 —— 那是**另一次聚类**
                    的几何，拿它下结论会得出关于别的结果的结论，所以不发
      ok            types / residual 可用
    """
    if not store.is_valid_slug(slug):
        return jsonify({"error": "bad slug"}), 400
    if not linetypes.ENABLED:
        return jsonify({"state": "disabled"})
    res, stale_reason = _results_state(slug)
    if res is None or stale_reason not in (None, "missing"):
        return jsonify({"state": "no-arrows", "detail": stale_reason})
    record = (res.get("pages") or {}).get(str(page)) or {}
    # 并集索引口径：items_of(rec) = vlm_items ++ vec_added。arrows_signature
    # 锚的就是这个列表，用别的口径算出来的签名一定对不上。
    items = store.items_of(record)
    revision = res.get("pdf_revision")
    extra = _placement_anchors_for(slug, page)
    arrows_sig = arrows.arrows_signature(items, revision, extra)
    arrow_entry = store.load_json(
        store.slug_dir(slug) / "arrows.json", {}).get(str(page))
    if not arrows.has_current_arrows(arrow_entry, arrows_sig):
        return jsonify({"state": "no-arrows"})
    sig = linetypes.linetypes_signature(arrows_sig)
    main_entry = linetypes.load_page(slug, page)
    if not linetypes.has_current_linetypes(main_entry, sig):
        return jsonify({"state": "no-arrows", "detail": "main result not current"})

    all_entry = linetypes.load_all_page(slug, page)
    if all_entry is None:
        return jsonify({"state": "not-run"})
    if all_entry.get("sig") != sig:
        return jsonify({"state": "stale"})
    payload = linetypes.all_payload(all_entry, main_entry)
    payload["state"] = "ok"
    return jsonify(payload)


@app.route("/img/<slug>/<int:page>")
def img(slug, page):
    if "/" in slug or "\\" in slug or ".." in slug \
            or not store.is_valid_slug(slug):
        return "bad slug", 400
    try:
        f, _ = _ensure_base(slug, page)
    except HTTPException:
        raise                       # abort(404) from _ensure_base -- not a render failure
    except Exception as e:                                  # noqa: BLE001
        # Detail goes to the log, not to the client: the exception text carries
        # absolute filesystem paths.
        print(f"[img] {slug} p{page} render failed: {e}", flush=True)
        return "render failed", 500
    resp = send_file(f, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "public, max-age=86400"
    return resp


if __name__ == "__main__":
    # Resume jobs interrupted by a previous server stop from their per-page
    # identity-checked checkpoints.  Explicit user cancellations stay stopped.
    try:
        job.resume_interrupted()
    except Exception as exc:                                    # noqa: BLE001
        print(f"[resume] skipped: {exc}", flush=True)
    host = os.environ.get("FENCE_LITE_HOST", "127.0.0.1")
    port = int(os.environ.get("FENCE_LITE_PORT", "5060"))
    print(f"fence_lite  →  http://{host}:{port}/", flush=True)
    app.run(host=host, port=port, threaded=True)
