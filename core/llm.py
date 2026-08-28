"""Anthropic Claude backend for the pipeline's single paid-call chokepoint.

`core.gemini.gen_json` dispatches here whenever the resolved model id is a
Claude id (see `core.config.provider_of`). The contract is deliberately the
same as the google-genai response the rest of the code already consumes:

    resp.text            -> str
    resp.usage_metadata  -> object with Gemini's attribute names

so `core.gemini.usage_from_response`, the CallRecorder and every
`compute_cost` caller keep working untouched.

Deliberate non-goals:

  * Prompt text is NOT rewritten per provider — the prompts in steps/prompts.py
    are sent byte-identical to both backends, which is what makes the
    comparison meaningful. The provider-specific framing Claude needs (emit
    bare JSON; box_2d is normalised 0-1000, not pixels) goes in a system
    prompt, which is an adapter concern rather than a prompt change.
  * No fallback to Gemini on error. A failed Claude call must surface as a
    failed Claude call, otherwise a comparison run silently mixes providers.
"""
import base64
import io
import json
import math
import os
import sys
import threading

from PIL import Image

# ── Anthropic request limits ────────────────────────────────────────────────
# Per-image cap is 10 MB *base64*, and base64 inflates by 4/3 — bound the raw
# bytes at 5 MB. (Gemini's limit in core/config.py is 90 MB, far above this,
# so pages that sail through the Gemini path do need re-encoding here.)
_MAX_IMAGE_BYTES = 5 * 1024 * 1024
_MAX_EDGE = 2576          # high-resolution tier long-edge limit
# Visual-token budget for one image (28x28 px per token). For drawing scans
# this is what actually binds, not the edge limit.
_MAX_TOKENS_VISUAL = 4784
# More than 20 image blocks in one request drops the per-image dimension limit
# to ~2000 px; oversized images are then rejected outright with an
# invalid_request_error mentioning "many-image requests".
_MANY_IMAGE_COUNT = 20
_MANY_IMAGE_MAX_EDGE = 2000
# Standard endpoints cap the whole request at 32 MB.
_MAX_REQUEST_BYTES = 28 * 1024 * 1024
_JPEG_LADDER = (92, 85, 78, 70, 60)

# Concurrency cap for this provider.
#
# The pipeline fans stages out across worker threads (TEXT_WORKERS=6,
# SYMBOLS_WORKERS=6, VIEW_WORKERS=6 in the unit file) and the Gemini runs peak
# around 5 simultaneous calls. Anthropic accounts also carry a *concurrent
# connections* limit, and exceeding it returns 429 "Number of concurrent
# connections has exceeded your rate limit" — which, unlike a token-rate 429,
# is not fixed by waiting a moment: every retry from every worker collides
# again. Gate the provider here instead of lowering the shared *_WORKERS knobs,
# so the Gemini path keeps its tuning.
#
# Default 1, measured rather than guessed: on the key this box uses, two
# simultaneous streams already 429 (probed at 2/3/4/6/8 concurrent — exactly
# one call succeeded every time). Streaming holds the connection for the whole
# generation, so 1 makes the Claude path fully serial: its wall-clock is the
# sum of its calls while Gemini still fans out ~5 wide. That is an account-tier
# limit, not a model property — raise ANTHROPIC_MAX_CONCURRENCY once the limit
# is lifted and the two providers' wall-times become comparable again.
MAX_CONCURRENCY = max(1, int(os.environ.get("ANTHROPIC_MAX_CONCURRENCY", "1")))
_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)


def slot():
    """The provider's concurrency slot, as a context manager.

    Held by core.gemini.gen_json around the whole call — deliberately outside
    the recorder's timer, so a thread waiting for the connection is not counted
    as an in-flight call nor billed as model time.
    """
    return _slots

# Reasoning depth. Sonnet 5 / Opus 5 default to "high"; drop to "medium" if the
# gunicorn --timeout 600 becomes the binding constraint on big pages.
EFFORT = os.environ.get("ANTHROPIC_EFFORT", "high")
# Ceiling on thinking + visible output combined. Legend sweeps can return long
# JSON arrays, so keep this generous.
MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "16000"))
# Coordinate frame to ask Claude for. "pixel" follows Anthropic's coordinates
# guide (Claude is documented to do poorly with normalized coordinates) and
# converts back to the pipeline's 0-1000 frame in _to_normalized();
# "normalized" asks for 0-1000 directly, as the shared prompts do for Gemini.
COORD_MODE = os.environ.get("ANTHROPIC_COORD_MODE", "normalized")


def _system_json(dims):
    """System prompt for the JSON calls, with the image's real dimensions.

    The shared prompts in steps/prompts.py speak Gemini's normalized 0-1000
    box_2d frame, and they are sent to both providers byte-identical so the
    comparison stays honest. Claude needs more scaffolding to hit that frame:
    observed failure on a 4896x3168 sheet was y values of 1163-1256 while x
    stayed in range — i.e. the y axis divided by something other than the
    image height. Naming both denominators explicitly, with the actual pixel
    numbers, is the fix; clamping the result is not (a clamped box lands at a
    page edge and looks plausible while being geometrically false).
    """
    if dims:
        w, h = dims
        axes = (
            f"This image is exactly {w} pixels wide and {h} pixels tall. "
            f"The two axes therefore have DIFFERENT denominators — use the "
            f"width for x and the height for y, never one for both:\n"
            f"    xmin = round(1000 * pixel_x / {w})    (x divides by {w})\n"
            f"    ymin = round(1000 * pixel_y / {h})    (y divides by {h})\n"
            f"and likewise for xmax / ymax. Every one of the four values must "
            f"land in 0-1000 inclusive. If any value comes out above 1000 you "
            f"have divided by the wrong number — recompute it, do not clip it."
        )
    else:
        axes = ("Derive each value from the pixel positions you observe: "
                "xmin = round(1000 * pixel_x / image_width), "
                "ymin = round(1000 * pixel_y / image_height), and likewise "
                "for the max corner. All four values must be in 0-1000.")
    return (
        "You are a precise visual-extraction engine for construction and "
        "architectural drawings.\n\n"
        "OUTPUT FORMAT: reply with raw JSON only. No markdown code fences, no "
        "prose, no explanation before or after the JSON. The first character "
        "of your reply must be '{' or '['.\n\n"
        "COORDINATES: every box_2d is [ymin, xmin, ymax, xmax] — four "
        "integers in a 0-1000 frame measured against the full image, where "
        "[0,0] is the top-left corner and [1000,1000] the bottom-right.\n"
        + axes +
        "\nNever emit raw pixel values, and never emit a box with zero width "
        "or zero height: if a feature is thinner than one unit, widen it to "
        "at least one so xmax > xmin and ymax > ymin."
    )


def _system_pixel(dims):
    """Pixel-coordinate variant of the system prompt.

    The shared prompts ask for a normalized 0-1000 box_2d because that is
    Gemini's convention. Anthropic's own coordinates guide says the opposite
    for Claude — "Claude works best with absolute pixel coordinates... Claude
    does not work well when you ask for normalized coordinates" — so this mode
    overrides that one instruction and converts the answer back to 0-1000 in
    _to_normalized() before the pipeline ever sees it. The task instructions
    (what to look for) are untouched; only the coordinate frame changes.

    Overriding an instruction from the user turn has to be explicit, or the two
    directives simply conflict and the model picks one at random.
    """
    w, h = dims
    return (
        "You are a precise visual-extraction engine for construction and "
        "architectural drawings.\n\n"
        "OUTPUT FORMAT: reply with raw JSON only. No markdown code fences, no "
        "prose, no explanation before or after the JSON. The first character "
        "of your reply must be '{' or '['.\n\n"
        "COORDINATE OVERRIDE — this overrides the coordinate scaling rule in "
        "the instructions that follow. Those instructions ask for box_2d "
        f"values normalized to 0-1000. Do NOT normalize. This image is exactly "
        f"{w} x {h} pixels; report every box_2d as ABSOLUTE PIXEL positions in "
        f"that image, still ordered [ymin, xmin, ymax, xmax], with x in 0-{w} "
        f"and y in 0-{h}. Origin [0,0] is the top-left pixel. A downstream "
        "wrapper rescales your pixel values, so pixels are what it expects — "
        "everything else in the instructions below still applies exactly as "
        "written.\n"
        "Never emit a box with zero width or zero height: if a feature is "
        "thinner than one pixel, widen it so xmax > xmin and ymax > ymin."
    )


def _rescale_boxes(node, sx, sy):
    """Rescale every box_2d in a parsed JSON tree, in place. Returns a count."""
    n = 0
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "box_2d" and isinstance(v, list) and len(v) == 4:
                try:
                    y0, x0, y1, x1 = (float(c) for c in v)
                except (TypeError, ValueError):
                    continue
                node[k] = [round(y0 * sy), round(x0 * sx),
                           round(y1 * sy), round(x1 * sx)]
                n += 1
            else:
                n += _rescale_boxes(v, sx, sy)
    elif isinstance(node, list):
        for v in node:
            n += _rescale_boxes(v, sx, sy)
    return n


def _collect_boxes(node, out):
    """Every 4-element box_2d in a parsed JSON tree."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "box_2d" and isinstance(v, list) and len(v) == 4:
                out.append(v)
            else:
                _collect_boxes(v, out)
    elif isinstance(node, list):
        for v in node:
            _collect_boxes(v, out)
    return out


def _valid_normalized(box):
    try:
        y0, x0, y1, x1 = (float(v) for v in box)
    except (TypeError, ValueError):
        return False
    if any(v != v or v in (float("inf"), float("-inf")) for v in (y0, x0, y1, x1)):
        return False
    return 0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000


def _repair_pixel_frame(text, dims):
    """Rescue a reply that came back in pixel units instead of 0-1000.

    Why this is needed: the shared prompts all specify the normalized frame,
    and Claude honours it for the text-scan prompt — but for the group+symbol
    prompt it reliably answers in pixels instead (measured on gladstone P3
    across repeated attempts: boxes like [0, 1975, 1540, 2380], which is
    exactly the right-hand title block in pixels of the 2380x1540 frame this
    adapter sent). Those replies are correct content in the wrong unit, and
    the frame is known here, so they are recoverable.

    The conversion is deliberately ALL-OR-NOTHING. It is applied only when
    every box in the reply becomes a valid normalized box afterwards; if even
    one does not, the original text is returned so the pipeline's validator
    rejects it as before. That distinction matters: a reply that merely looks
    pixel-ish but is actually scrambled (one observed case had xmin > xmax and
    mixed units in a single box) must NOT be silently rewritten into a
    plausible box at an unrelated page edge — the same reasoning core/parsing.py
    gives for refusing to clamp.
    """
    if not dims or not text:
        return text, 0
    w, h = dims
    if not w or not h:
        return text, 0
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return text, 0
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return text, 0

    boxes = _collect_boxes(parsed, [])
    if not boxes:
        return text, 0
    # Only consider a pixel reading when something actually overflows the
    # normalized frame — an in-range reply is taken at its word.
    overflow = False
    for b in boxes:
        try:
            if any(float(v) > 1000 for v in b):
                overflow = True
                break
        except (TypeError, ValueError):
            return text, 0
    if not overflow:
        return text, 0

    sx, sy = 1000.0 / w, 1000.0 / h
    candidate = json.loads(stripped)          # fresh copy to mutate
    _rescale_boxes(candidate, sx, sy)
    if not all(_valid_normalized(b) for b in _collect_boxes(candidate, [])):
        _log("reply overflows 0-1000 but does not resolve as a pixel frame "
             "either — leaving it for the validator to reject")
        return text, 0
    _log(f"reply was in pixel units for a {w}x{h} frame; converted "
         f"{len(boxes)} box(es) to 0-1000")
    return json.dumps(candidate, ensure_ascii=False), len(boxes)


def _to_normalized(text, dims):
    """Convert a pixel-frame reply into the 0-1000 frame the pipeline expects.

    Returns the text unchanged if it cannot be parsed — the callers' own
    parsers (core/parsing.py) already strip fences and extract the outermost
    value, and a validator rejecting a box is a far better outcome than this
    silently rescaling something it misread.
    """
    if not dims or not text:
        return text
    w, h = dims
    if not w or not h:
        return text
    stripped = text.strip()
    if not stripped.startswith(("{", "[")):
        return text
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return text
    n = _rescale_boxes(parsed, 1000.0 / w, 1000.0 / h)
    if not n:
        return text
    return json.dumps(parsed, ensure_ascii=False)


def _log(msg):
    print(f"[llm] {msg}", file=sys.stderr, flush=True)


# ── Client (lazy + thread-safe: the batch stages fan out across threads) ────
_client = None
_client_lock = threading.Lock()


def get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                import anthropic

                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError(
                        "ANTHROPIC_API_KEY is required to run a Claude model. "
                        "Add it to fence_lite/.env and restart the service."
                    )
                # max_retries above the SDK default of 2: a burst of
                # concurrent-connection 429s should be ridden out rather than
                # failing a page, and the semaphore below keeps the burst small.
                _client = anthropic.Anthropic(api_key=key, timeout=600.0,
                                              max_retries=6)
    return _client


# ── Image conversion ────────────────────────────────────────────────────────
def _encode(im, fmt, quality=None):
    buf = io.BytesIO()
    if fmt == "PNG":
        # compress_level=1 for the same reason core/gemini.py uses it: PNG is
        # lossless at every level, so the pixels the model sees are identical,
        # and level 1 cuts CPU encode time markedly on a single-core box.
        im.save(buf, format="PNG", compress_level=1)
        return buf.getvalue(), "image/png"
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _visual_tokens(w, h):
    """Anthropic bills images in 28x28 patches; one patch = one visual token."""
    return math.ceil(w / 28) * math.ceil(h / 28)


def _target_size(w, h, max_edge, max_tokens):
    """The size Anthropic would itself downscale this image to.

    Reference implementation from the coordinates guide: the largest
    aspect-preserving size satisfying BOTH the edge limit and the visual-token
    budget. Worth computing rather than just scaling to the edge — for nearly
    all drawing scans it is the *token* budget that binds, so a 4896x3168 sheet
    lands near 2380x1540 rather than at the 2576 edge, and assuming otherwise
    puts every returned coordinate off target.
    """
    def fits(a, b):
        return (math.ceil(a / 28) * 28 <= max_edge
                and math.ceil(b / 28) * 28 <= max_edge
                and _visual_tokens(a, b) <= max_tokens)

    if fits(w, h):
        return w, h
    if h > w:
        th, tw = _target_size(h, w, max_edge, max_tokens)
        return tw, th
    aspect = w / h
    lo, hi = 1, w                      # lo always fits, hi never does
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect), 1)


def _prepare_image(data, mime, max_edge, max_bytes, max_tokens=None):
    """Resize one image to exactly the frame Claude will see, and fit the byte cap.

    Why resize here rather than let the API do it: the coordinates Claude
    returns are positions in the image *after* the server's own downscale. The
    docs' recommended fix is to pre-resize so the image you hold is the image
    Claude sees — otherwise the frame it measures against is one this code
    never computed, which is exactly how a 4896x3168 sheet produced y values
    above 1000. Downscaling costs nothing here: box_2d is a 0-1000 fraction,
    so it is scale-invariant, and a smaller payload uploads faster.

    Line art survives PNG far better than JPEG, so PNG is tried first.
    """
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
    except Exception:                                          # noqa: BLE001
        return data, mime, None  # undecodable here; let the API judge it

    w, h = im.size
    tw, th = _target_size(w, h, max_edge, max_tokens or _MAX_TOKENS_VISUAL)
    if (tw, th) == (w, h) and len(data) <= max_bytes:
        return data, mime, (w, h)

    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    if (tw, th) != (w, h):
        im = im.resize((tw, th), Image.LANCZOS)
        _log(f"pre-resized {w}x{h} -> {tw}x{th} "
             f"({_visual_tokens(w, h)} -> {_visual_tokens(tw, th)} visual tokens)")

    out, out_mime = _encode(im, "PNG")
    if len(out) <= max_bytes:
        return out, out_mime, im.size
    for q in _JPEG_LADDER:
        out, out_mime = _encode(im, "JPEG", q)
        if len(out) <= max_bytes:
            _log(f"PNG over budget, used JPEG q{q} ({len(out) / 1e6:.1f} MB)")
            return out, out_mime, im.size
    while max(im.size) > 512:
        im = im.resize((max(1, im.width // 2), max(1, im.height // 2)),
                       Image.LANCZOS)
        out, out_mime = _encode(im, "JPEG", 80)
        if len(out) <= max_bytes:
            _log(f"halved to {im.width}x{im.height} to fit byte budget")
            return out, out_mime, im.size
    return out, out_mime, im.size


def _split_contents(contents):
    """Split a google-genai ``contents`` list into (image blobs, text chunks)."""
    images, texts = [], []
    for part in contents or []:
        if isinstance(part, str):
            texts.append(part)
            continue
        blob = getattr(part, "inline_data", None)
        if blob is not None and getattr(blob, "data", None):
            images.append((blob.data, blob.mime_type or "image/png"))
            continue
        txt = getattr(part, "text", None)
        if txt:
            texts.append(txt)
    return images, texts


def _build_content(contents):
    """Returns (content blocks, dims of the first image or None).

    The dims travel with the request so the system prompt can name the exact
    pixel denominators for the 0-1000 conversion. The first image is the one
    the prompts describe ("this page"); later parts are crops or extra pages.
    """
    images, texts = _split_contents(contents)
    many = len(images) > _MANY_IMAGE_COUNT
    max_edge = _MANY_IMAGE_MAX_EDGE if many else _MAX_EDGE
    # Above 20 images the binding constraint is the per-image *dimension* cap,
    # not the single-image token budget, so pass a bound loose enough that the
    # edge cap is what actually decides the size. (No call site in this repo
    # sends more than one image today — every gen_json caller passes exactly
    # one page or one crop — so this branch is here for correctness if a batch
    # stage is ever added, not because it is exercised.)
    max_tokens_visual = (_MANY_IMAGE_MAX_EDGE // 28) ** 2 if many else _MAX_TOKENS_VISUAL
    budget = min(_MAX_IMAGE_BYTES,
                 max(_MAX_REQUEST_BYTES // max(len(images), 1), 256 * 1024))

    blocks, total, first_dims = [], 0, None
    for data, mime in images:
        data, mime, dims = _prepare_image(data, mime, max_edge, budget,
                                          max_tokens_visual)
        if first_dims is None:
            first_dims = dims
        total += len(data)
        blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime,
                "data": base64.standard_b64encode(data).decode("ascii"),
            },
        })
    if total > _MAX_REQUEST_BYTES:
        _log(f"WARNING: {total / 1e6:.1f} MB of images may exceed the request cap")
    # Images before text — matches the existing call sites and the documented
    # image-then-text preference.
    for t in texts:
        blocks.append({"type": "text", "text": t})
    return blocks, first_dims


# ── Schema translation ──────────────────────────────────────────────────────
def _strict_schema(node):
    """Deep-copy a Gemini response schema into the shape Anthropic requires.

    The pipeline's schemas (steps/prompts.py, steps/legend_sweep.py) are plain
    object/array/enum trees, which structured outputs supports — but every
    object needs an explicit ``additionalProperties: false``.
    """
    if isinstance(node, dict):
        out = {k: _strict_schema(v) for k, v in node.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(node, list):
        return [_strict_schema(v) for v in node]
    return node


def _effort_for(thinking_budget):
    """Map Gemini's thinking-token budget onto Claude's effort levels.

    Sonnet 5 / Opus 5 removed fixed thinking budgets (budget_tokens returns a
    400); effort is the replacement knob. Callers only pass a budget to make a
    cheap stage cheap, so preserve that intent rather than the exact number.
    """
    if not thinking_budget:
        return EFFORT
    try:
        b = int(thinking_budget)
    except (TypeError, ValueError):
        return EFFORT
    if b <= 2048:
        return "low"
    if b <= 8192:
        return "medium"
    return EFFORT


# ── Response shim ───────────────────────────────────────────────────────────
class _Usage:
    """Anthropic token counts under the Gemini attribute names that
    core.gemini.usage_from_response reads."""

    __slots__ = ("prompt_token_count", "candidates_token_count",
                 "thoughts_token_count", "cached_content_token_count",
                 "total_token_count")

    def __init__(self, usage):
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        thinking = int(getattr(details, "thinking_tokens", 0) or 0) if details else 0

        # Anthropic's output_tokens already includes thinking tokens, whereas
        # Gemini reports them separately and compute_cost sums
        # candidates + thoughts. Split so the billed total stays correct.
        self.prompt_token_count = inp + cache_read + cache_write
        self.candidates_token_count = max(out - thinking, 0)
        self.thoughts_token_count = thinking
        self.cached_content_token_count = cache_read
        self.total_token_count = self.prompt_token_count + out


class _Response:
    __slots__ = ("text", "usage_metadata", "stop_reason", "model")

    def __init__(self, text, usage, stop_reason, model):
        self.text = text
        self.usage_metadata = _Usage(usage) if usage is not None else None
        self.stop_reason = stop_reason
        self.model = model


# ── Entry point ─────────────────────────────────────────────────────────────
def generate_json(model, contents, timeout_ms=None, thinking_budget=None,
                 response_json_schema=None):
    """Claude equivalent of the google-genai JSON call in core.gemini.gen_json."""
    blocks, dims = _build_content(contents)
    client = get_client()
    if timeout_ms:
        client = client.with_options(timeout=float(timeout_ms) / 1000.0)

    kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        # temperature is intentionally absent: Sonnet 5 / Opus 5 reject a
        # non-default temperature with a 400, so the Gemini call's
        # temperature=0.0 cannot be carried over.
        "thinking": {"type": "adaptive"},
        "system": (_system_pixel(dims)
                   if (COORD_MODE == "pixel" and dims) else _system_json(dims)),
        "messages": [{"role": "user", "content": blocks}],
    }
    output_config = {"effort": _effort_for(thinking_budget)}
    if response_json_schema:
        output_config["format"] = {
            "type": "json_schema",
            "schema": _strict_schema(response_json_schema),
        }
    kwargs["output_config"] = output_config

    def _send(kw):
        # Stream and collect: a large max_tokens on a non-streaming request
        # risks the HTTP idle timeout, and the SDK refuses such requests.
        # The concurrency slot is NOT taken here — core.gemini.gen_json holds
        # it around this whole call so that queue-wait stays out of the
        # recorder's timing. Acquiring again at cap 1 would self-deadlock.
        with client.messages.stream(**kw) as stream:
            return stream.get_final_message()

    try:
        msg = _send(kwargs)
    except Exception as e:                                     # noqa: BLE001
        # A schema the pipeline wrote for Gemini may be rejected here (the
        # supported JSON Schema subset is narrower). Retry once without it —
        # the prompt still demands JSON, and core/parsing.py strips fences and
        # extracts the outermost object as a safety net.
        if response_json_schema and _is_schema_error(e):
            _log(f"schema rejected ({e}); retrying without response format")
            kwargs["output_config"] = {"effort": _effort_for(thinking_budget)}
            msg = _send(kwargs)
        else:
            raise

    if msg.stop_reason == "refusal":
        detail = getattr(msg, "stop_details", None)
        raise RuntimeError(f"{model} declined the request "
                           f"(category={getattr(detail, 'category', None)})")
    if msg.stop_reason == "max_tokens":
        _log(f"WARNING: {model} hit max_tokens={MAX_TOKENS}; JSON likely "
             f"truncated — raise ANTHROPIC_MAX_TOKENS")

    text = "".join(b.text for b in msg.content if b.type == "text")
    if COORD_MODE == "pixel" and dims:
        text = _to_normalized(text, dims)
    else:
        # Asked for 0-1000 but some prompts come back in pixels anyway; rescue
        # those, but only when the whole reply resolves cleanly (see
        # _repair_pixel_frame).
        text, _n = _repair_pixel_frame(text, dims)
    return _Response(text, msg.usage, msg.stop_reason, msg.model)


def _is_schema_error(exc):
    s = str(exc).lower()
    return "schema" in s or "output_config" in s or "format" in s
