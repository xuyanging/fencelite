"""步骤3：给已有的 ``kind=="view"`` 组框判投影类型（plan / elevation / …）.

为什么单独一次付费调用、单独一份缓存（data/<slug>/view_types.json）：
组框在步骤2 已经定下来了，这里唯一的问题是「这张图是俯视图吗」。步骤2 那次
推理有时会顺口给一个 view_type，但它没有分类器的版本与模型出处，一旦被采信，
bump VIEW_VERSION 就无法真正重分类 —— 所以本步不信任它（steps.symbols 也已
不写入该字段），只认这份带 sig/v/model 的独立缓存。

下游（步骤4 shape 放置匹配）只在 plan 视图里保留放置，缺分类就 fail-closed：
plan_boxes 返回空 → 一个放置都不留，绝不猜。

缓存 entry = {sig, v, model, views[, elapsed, usage]}
  views = [{group_index, view_type, reason}, ...]，必须覆盖本页每一个合法
  view 组、不多不少；漏一个就是整页作废重试（_parse_response）。
"""
import hashlib
import json
import time

from google.genai import types

from core.config import resolve_model
from core.gemini import (_encode_image_for_gemini, gen_json,
                         usage_from_response)
from core.pdfio import render_pdf_page
from steps.prompts import (VIEW_CLASSIFIER_PROMPT, VIEW_CLASSIFIER_SCHEMA,
                           VIEW_TYPES)
from steps.versions import VIEW_VERSION


def _valid_box(box):
    return bool(
        isinstance(box, (list, tuple)) and len(box) == 4
        and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in box)
        and 0 <= box[0] < box[2] <= 1000
        and 0 <= box[1] < box[3] <= 1000
    )


def _view_rows(groups):
    return [
        {"group_index": index, "box_2d": group.get("box_2d")}
        for index, group in enumerate(groups or [])
        if group.get("kind") == "view" and _valid_box(group.get("box_2d"))
    ]


def view_signature(groups, pdf_revision, model=None):
    resolved_model = resolve_model(model)
    payload = {
        "version": VIEW_VERSION,
        "model": resolved_model,
        "pdf": pdf_revision,
        "views": _view_rows(groups),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def groups_need_classification(groups):
    """Whether this page has a view governed by the independent classifier.

    A ``view_type`` embedded in ``symbols.json`` has no classifier version or
    model provenance, so it must not bypass this cache.  This also guarantees
    that bumping ``VIEW_VERSION`` really reclassifies every view.
    """
    return any(
        group.get("kind") == "view"
        and _valid_box(group.get("box_2d"))
        for group in (groups or [])
    )


def has_current_view_types(entry, groups, pdf_revision, model=None):
    resolved_model = resolve_model(model)
    rows = _view_rows(groups)
    expected = {row["group_index"] for row in rows}
    if not (
        isinstance(entry, dict)
        and entry.get("sig") == view_signature(
            groups, pdf_revision, resolved_model)
        and entry.get("v") == VIEW_VERSION
        and entry.get("model") == resolved_model
        and isinstance(entry.get("views"), list)
    ):
        return False
    seen = set()
    for row in entry["views"]:
        if not isinstance(row, dict):
            return False
        index = row.get("group_index")
        view_type = str(row.get("view_type") or "").lower()
        reason = row.get("reason")
        if isinstance(index, bool) or not isinstance(index, int) \
                or index not in expected or index in seen \
                or view_type not in VIEW_TYPES \
                or not isinstance(reason, str) or not reason.strip():
            return False
        seen.add(index)
    return seen == expected


def _parse_response(text, expected_indices):
    try:
        payload = json.loads((text or "").strip())
    except (TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("view classifier returned invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("views"), list):
        raise RuntimeError("view classifier response must contain views[]")
    out, seen = [], set()
    for row_index, row in enumerate(payload["views"]):
        if not isinstance(row, dict):
            raise RuntimeError(
                f"view classifier row {row_index} must be an object")
        index = row.get("group_index")
        view_type = str(row.get("view_type") or "").strip().lower()
        raw_reason = row.get("reason")
        if not isinstance(raw_reason, str):
            raise RuntimeError(
                f"view classifier row {row_index} has invalid reason")
        reason = raw_reason.strip()
        if isinstance(index, bool) or not isinstance(index, int) \
                or index not in expected_indices or index in seen:
            raise RuntimeError(
                f"view classifier row {row_index} has invalid group_index")
        if view_type not in VIEW_TYPES:
            raise RuntimeError(
                f"view classifier row {row_index} has invalid view_type")
        if not reason:
            raise RuntimeError(
                f"view classifier row {row_index} has empty reason")
        seen.add(index)
        out.append({"group_index": index, "view_type": view_type,
                    "reason": reason})
    if seen != set(expected_indices):
        raise RuntimeError("view classifier did not classify every view group")
    return out


def compute_view_types(pdf_path, page_index, groups, pdf_revision,
                       cached=None, model=None, timeout_ms=180_000):
    """Return ``(entry, fresh_call)`` for one page's existing view boxes."""
    resolved_model = resolve_model(model)
    sig = view_signature(groups, pdf_revision, resolved_model)
    if has_current_view_types(
            cached, groups, pdf_revision, resolved_model):
        return cached, False
    view_rows = _view_rows(groups)
    if not view_rows:
        return {"sig": sig, "v": VIEW_VERSION,
                "model": resolved_model, "views": [], "elapsed": 0.0}, False

    image = render_pdf_page(pdf_path, page_index)
    data, mime = _encode_image_for_gemini(image)
    part = types.Part.from_bytes(data=data, mime_type=mime)
    prompt = VIEW_CLASSIFIER_PROMPT.format(
        groups_json=json.dumps(view_rows, ensure_ascii=False))
    t0 = time.perf_counter()
    response = gen_json(
        resolved_model, [part, prompt],
        timeout_ms=timeout_ms, response_json_schema=VIEW_CLASSIFIER_SCHEMA)
    elapsed = time.perf_counter() - t0
    views = _parse_response(
        response.text or "", {row["group_index"] for row in view_rows})
    entry = {"sig": sig, "v": VIEW_VERSION,
             "model": resolved_model, "views": views,
             "elapsed": round(elapsed, 1),
             "usage": usage_from_response(response)}
    return entry, True


def merge_view_types(groups, entry):
    """Return copied groups enriched with cached type and audit reason."""
    copied = json.loads(json.dumps(groups or []))
    by_index = {
        row["group_index"]: row for row in (entry or {}).get("views", [])
        if isinstance(row, dict) and isinstance(row.get("group_index"), int)
    }
    for index, group in enumerate(copied):
        row = by_index.get(index)
        if group.get("kind") == "view" and row:
            group["view_type"] = row["view_type"]
            group["view_type_reason"] = row.get("reason", "")
    return copied


def plan_boxes(typed_groups):
    """merge_view_types 之后的组 → 全部 plan 视图框（步骤4 的唯一取景框）.

    只认这里写进去的 view_type：没分类、非 plan、框非法的一律不算 plan，
    调用方拿到空列表就该 fail-closed（不留任何放置），而不是退化成整页。
    """
    return [list(group["box_2d"]) for group in (typed_groups or [])
            if isinstance(group, dict) and group.get("kind") == "view"
            and str(group.get("view_type") or "").lower() == "plan"
            and _valid_box(group.get("box_2d"))]
