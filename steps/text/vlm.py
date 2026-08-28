"""VLM 读图找 fence 文字 —— 生产单任务提示词 + 扫描页双模型 union.

One job only: hand Gemini a page image and ask for every piece of TEXT that
mentions a fence.  No groups, no structure pass, no legends/symbols — the
single-task prompt keeps recall high.  Renderer and normalized 0-1000 frame
are exactly the main pipeline's, so boxes are directly comparable.

union_vlm: scan pages have no vector floor, so a SECOND model's finds are
unioned in (Pro often goes half-blind on rasters where Flash reads fine).
"""
import json
import re
import time
from pathlib import Path

from google.genai import types

from core.config import resolve_model
from core.gemini import _encode_image_for_gemini, gen_json, usage_from_response
from core.parsing import _coerce_box
from core.pdfio import render_pdf_page
from steps.text.judge import norm_text
from steps.text.merge import intersects
from steps.text.target import TARGET_DEFAULT, build_vlm_prompt

# Default image-scan prompt = the fixed scaffolding wrapped around the default
# fence target.  scan_page(prompt=...) accepts any prompt built by
# target.build_vlm_prompt(user_target); the output-format contract lives in the
# scaffolding, never in the user-editable target.  (See target.py.)
PROMPT = build_vlm_prompt(TARGET_DEFAULT)


def _parse_scan_response(text):
    """Parse one *complete* scan response.

    Fence-text scans are persisted as resumable cache entries.  Recovering
    the complete object prefix of a response cut off midway through its JSON
    array would therefore turn a transport failure into a durable false
    negative.  Accept the two documented payload shapes (and complete
    markdown fences for provider compatibility), but reject every incomplete
    or trailing-garbage response so the caller retries it.
    """
    raw = (text or "").strip()
    if not raw:
        raise RuntimeError("empty Gemini response")
    if raw.startswith("```"):
        if not raw.endswith("```") or len(raw) < 6:
            raise RuntimeError("incomplete fenced Gemini response")
        raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1,
                     flags=re.IGNORECASE)
        raw = raw[:-3].strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"incomplete/unparseable Gemini response: {raw[:200]!r}"
        ) from exc
    if isinstance(parsed, dict):
        parsed = parsed.get("items")
    if not isinstance(parsed, list):
        raise RuntimeError(
            f"Gemini response must be an array or items object: {raw[:200]!r}"
        )
    return parsed


def scan_page(pdf_path, page_index, model=None, timeout_ms=300_000,
              prompt=None):
    """Render one page exactly like the main pipeline does, run the
    single-task prompt, return (items, elapsed, usage).

    ``prompt`` defaults to the preset fence prompt, so callers that omit it get
    byte-identical behaviour to the original pipeline.  The web layer passes a
    user-edited prompt here to detect an arbitrary target instead of fences;
    the prompt is also folded into the VLM cache identity (see vlm_cache), so a
    changed prompt never reuses a different prompt's cached response."""
    img = render_pdf_page(Path(pdf_path), page_index)
    data, mime = _encode_image_for_gemini(img)
    part = types.Part.from_bytes(data=data, mime_type=mime)
    t0 = time.perf_counter()
    resp = gen_json(resolve_model(model), [part, prompt or PROMPT],
                    timeout_ms=timeout_ms)
    elapsed = time.perf_counter() - t0

    # A silent [] must mean "the model saw no fence text", never "the response
    # was empty/unparseable".  In particular, never cache a valid prefix from
    # a JSON array truncated by the model output cap.
    parsed = _parse_scan_response(resp.text)
    items = []
    for row_index, it in enumerate(parsed):
        if not isinstance(it, dict):
            raise RuntimeError(
                f"Gemini scan row {row_index} must be an object")
        if "text" not in it or not isinstance(it["text"], str):
            raise RuntimeError(
                f"Gemini scan row {row_index} has no valid text")
        txt = it["text"].strip()
        if not txt:
            raise RuntimeError(
                f"Gemini scan row {row_index} has empty text")
        if "box_2d" not in it:
            raise RuntimeError(
                f"Gemini scan row {row_index} has no box_2d")
        box = _coerce_box(it["box_2d"])
        if box is None:
            raise RuntimeError(
                f"Gemini scan row {row_index} has an invalid box_2d")
        items.append({"text": txt, "box_2d": box,
                      "label": str(it.get("label", "other")).strip() or "other"})
    return items, elapsed, usage_from_response(resp)


def _iou(a, b):
    iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ix = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ar = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ar if ar > 0 else 0.0


def union_vlm(primary, secondary):
    """Scan pages have no vector floor, so TWO different models look at the
    page and their finds are unioned (Pro often goes half-blind on rasters
    where Flash reads fine — proven on construction_pkg1 P5).  A secondary
    item is a duplicate when it lands on a primary box or repeats the same
    text nearby; everything else is added with a model tag."""
    out = list(primary or [])
    for it in (secondary or []):
        b = it.get("box_2d")
        if not b:
            continue
        nt = " ".join(norm_text(it.get("text", "")).split())
        dup = False
        for p in (primary or []):
            pb = p.get("box_2d")
            if not pb:
                continue
            if _iou(b, pb) > 0.45 or (
                    nt and nt == " ".join(norm_text(p.get("text", "")).split())
                    and intersects(b, pb, tol=15)):
                dup = True
                break
        if not dup:
            it = dict(it)
            it.setdefault("model", "gemini-3.5-flash")
            out.append(it)
    return out
