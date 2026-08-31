"""Shared Gemini JSON call, usage, and image-encoding helpers for 5054.

fence_takeoff_web addition: ``gen_json`` is the single chokepoint every paid
Gemini call in the whole pipeline passes through (verified: exactly one
``generate_content`` site in the codebase).  We therefore record each call's
wall-time + token usage + USD cost into a thread-safe, job-partitioned
``RECORDER`` so the web layer can report each concurrently processing PDF's
model time and spend. The recorder is a pure observer.
"""
import io
import os
import threading
import time
from contextvars import ContextVar

from google import genai
from google.genai import types
from PIL import Image

from core.concurrency import SlotPool, shared_capacity_directory
from core.config import (API_KEY, GEMINI_INLINE_BYTES_LIMIT, compute_cost,
                         provider_of)

client = genai.Client(api_key=API_KEY)

_USAGE_FIELDS = (
    ("prompt_token_count", "input_tokens"),
    ("candidates_token_count", "output_tokens"),
    ("thoughts_token_count", "thoughts_tokens"),
    ("cached_content_token_count", "cached_tokens"),
    ("total_token_count", "total_tokens"),
)


_RECORDER_SESSION = ContextVar("fence_lite_recorder_session", default=None)


def _empty_recording(on=False):
    return {
        "on": bool(on), "calls": 0, "model_seconds": 0.0,
        "input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0,
        "cost_usd": 0.0, "by_model": {}, "active": 0,
        "peak_concurrency": 0,
    }


def _recording_summary(state):
    state = state or _empty_recording()
    return {
        "calls": state["calls"],
        "model_seconds": round(state["model_seconds"], 2),
        "peak_concurrency": state["peak_concurrency"],
        "input_tokens": state["input_tokens"],
        "output_tokens": state["output_tokens"],
        "thoughts_tokens": state["thoughts_tokens"],
        "cost_usd": round(state["cost_usd"], 4),
        "by_model": {
            key: {**value, "seconds": round(value["seconds"], 2),
                  "cost_usd": round(value["cost_usd"], 4)}
            for key, value in state["by_model"].items()
        },
    }


class CallRecorder:
    """Keep an independent cost ledger for every concurrently running PDF.

    A ContextVar identifies the current job.  The orchestrator propagates that
    context into every paid ThreadPool callback, allowing calls from multiple
    PDFs to overlap without combining their usage or peak-concurrency totals.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sessions = {}
        self._sequence = 0
        # Callback(session_key), fired outside the recorder lock after a paid
        # call is accounted, so job.py can durably flush the correct PDF.
        self.on_update = None

    def _key(self, session=None):
        return session if session is not None else _RECORDER_SESSION.get()

    def start(self, session=None):
        with self._lock:
            if session is None:
                self._sequence += 1
                session = ("legacy", threading.get_ident(), self._sequence)
            self._sessions[session] = _empty_recording(on=True)
        _RECORDER_SESSION.set(session)
        return session

    def stop(self, session=None):
        key = self._key(session)
        with self._lock:
            state = self._sessions.pop(key, None) if key is not None else None
            if state is not None:
                state["on"] = False
            out = _recording_summary(state)
        if _RECORDER_SESSION.get() == key:
            _RECORDER_SESSION.set(None)
        return out

    def enter(self):
        key = self._key()
        if key is None:
            return
        with self._lock:
            state = self._sessions.get(key)
            if not state or not state["on"]:
                return
            state["active"] += 1
            state["peak_concurrency"] = max(
                state["peak_concurrency"], state["active"])

    def leave(self):
        key = self._key()
        if key is None:
            return
        with self._lock:
            state = self._sessions.get(key)
            if state and state["active"] > 0:
                state["active"] -= 1

    def add(self, model, elapsed, usage):
        key = self._key()
        if key is None:
            return
        cost = compute_cost(model, usage) or {}
        in_tok = int((usage or {}).get("input_tokens") or 0)
        out_tok = int((usage or {}).get("output_tokens") or 0)
        th_tok = int((usage or {}).get("thoughts_tokens") or 0)
        usd = float(cost.get("total_usd") or 0.0)
        with self._lock:
            state = self._sessions.get(key)
            if not state or not state["on"]:
                return
            state["calls"] += 1
            state["model_seconds"] += float(elapsed or 0.0)
            state["input_tokens"] += in_tok
            state["output_tokens"] += out_tok
            state["thoughts_tokens"] += th_tok
            state["cost_usd"] += usd
            by_model = state["by_model"].setdefault(
                model, {"calls": 0, "seconds": 0.0, "cost_usd": 0.0,
                        "input_tokens": 0, "output_tokens": 0,
                        "thoughts_tokens": 0})
            by_model["calls"] += 1
            by_model["seconds"] += float(elapsed or 0.0)
            by_model["cost_usd"] += usd
            by_model["input_tokens"] += in_tok
            by_model["output_tokens"] += out_tok
            by_model["thoughts_tokens"] += th_tok
        cb = self.on_update
        if cb:
            try:
                cb(key)
            except Exception:                                  # noqa: BLE001
                pass  # persistence must never break a paid call

    def summary(self, session=None):
        key = self._key(session)
        with self._lock:
            return _recording_summary(self._sessions.get(key))


RECORDER = CallRecorder()

# Two project pipelines can each fan out six or eight Gemini calls.  Bound the
# combined provider pressure while still allowing more overlap than the old
# single-project scheduler.  Queue wait stays outside _record(), so it is not
# mislabeled as provider/model time.
GEMINI_MAX_CONCURRENCY = max(
    1, int(os.environ.get("GEMINI_MAX_CONCURRENCY", "8")))
_GEMINI_SLOTS = SlotPool(
    shared_capacity_directory(), "provider-gemini", GEMINI_MAX_CONCURRENCY)


def is_timeout_error(error) -> bool:
    """Return whether an exception chain represents a hard deadline.

    Retrying malformed model output can help.  A remote provider deadline gets
    one bounded accuracy retry, while a deterministic local-engine deadline is
    not repeated; identifying the two prevents the old three-full-deadline
    progress stall.  Keep this test provider-agnostic (Google, Anthropic and
    ``subprocess`` use different exception classes) and walk the chained
    cause/context as SDKs often wrap their transport timeout.
    """
    seen = set()
    current = error
    while isinstance(current, BaseException) and id(current) not in seen:
        seen.add(id(current))
        name = type(current).__name__.lower()
        message = str(current).lower()
        # A malformed model payload can legitimately contain the word
        # "timeout" in its own text.  It is still a completed provider call,
        # not a transport deadline; keep it on the malformed-response retry
        # path.  The string prefix covers error records reloaded from JSON,
        # where the original exception type is no longer available.
        if (getattr(current, "retry_as_malformed", False)
                or message.startswith("vlmresponseerror:")):
            return False
        if ("timeout" in name or "timeout" in message
                or "deadline exceeded" in message
                or "deadline_exceeded" in message
                or "deadlineexceeded" in message):
            return True
        current = current.__cause__ or current.__context__
    return False


def should_retry_model_error(error, attempt, total_attempts,
                             timeout_retries=1) -> bool:
    """Bound retries without treating a provider timeout as deterministic.

    ``attempt`` is zero-based and describes the call which just failed.
    Malformed/transient responses may use every configured attempt.  A remote
    model timeout gets only ``timeout_retries`` additional calls (one by
    default): this preserves recall after a one-off transport/provider stall
    while keeping every page's wall time finite.  Local deterministic engine
    timeouts intentionally do not use this helper.
    """
    if attempt + 1 >= max(1, int(total_attempts)):
        return False
    if getattr(error, "retry_as_malformed", False):
        return True
    if is_timeout_error(error):
        return attempt < max(0, int(timeout_retries))
    return True


def gen_json(model, contents, timeout_ms=None, thinking_budget=None,
             response_json_schema=None):
    """The one shared entry point for every Gemini call in the pipeline:
    temperature=0 + forced JSON output. timeout_ms optionally caps the HTTP
    request (the SDK default has none, so a hung connection stalls forever);
    thinking_budget optionally caps reasoning tokens — thinking time is the
    dominant, highly variable share of latency on vision calls.

    Still the one chokepoint, now for two providers: a Claude model id is
    routed to core/llm.py (Anthropic Messages API) instead of google-genai.
    Both paths return an object exposing ``.text`` and ``.usage_metadata``, so
    the recorder / costing below is provider-agnostic.

    The provider's concurrency slot is taken BEFORE the timer starts and before
    RECORDER.enter(). That ordering is load-bearing for the numbers this
    recorder exists to report: acquire it later and a thread queued behind the
    single allowed connection counts as an in-flight call (inflating
    peak_concurrency) and bills its wait as model time (inflating
    model_seconds — a 413 s run reported 747 s of "model" time that way)."""
    if provider_of(model) == "anthropic":
        from core import llm  # local import: keeps `anthropic` optional

        with llm.slot():
            return _record(model, lambda: llm.generate_json(
                model, contents,
                timeout_ms=timeout_ms,
                thinking_budget=thinking_budget,
                response_json_schema=response_json_schema,
            ))
    with _GEMINI_SLOTS.slot():
        return _record(model, lambda: client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
                response_json_schema=response_json_schema,
                http_options=(types.HttpOptions(timeout=timeout_ms)
                              if timeout_ms else None),
                thinking_config=(
                    types.ThinkingConfig(thinking_budget=thinking_budget)
                    if thinking_budget else None),
            ),
        ))


def _record(model, send):
    """Time one paid call and fold it into RECORDER. Pure observer."""
    t0 = time.perf_counter()
    RECORDER.enter()
    try:
        resp = send()
    finally:
        RECORDER.leave()
    try:
        RECORDER.add(model, time.perf_counter() - t0, usage_from_response(resp))
    except Exception:                                          # noqa: BLE001
        pass  # instrumentation must never break a paid call
    return resp


def usage_from_response(resp, include_cached=True):
    """Map SDK usage_metadata onto our own field names. include_cached=False
    reproduces the 4-field variant used by cross-ref / secondary-ref
    responses (kept for response-schema compatibility)."""
    usage = getattr(resp, "usage_metadata", None)
    out = {}
    if usage is None:
        return out
    for src, dst in _USAGE_FIELDS:
        if not include_cached and dst == "cached_tokens":
            continue
        v = getattr(usage, src, None)
        if v is not None:
            out[dst] = int(v)
    return out


def _encode_image_for_gemini(image: Image.Image):
    """PNG by default; JPEG fallback if PNG exceeds the inline byte limit.

    compress_level=1: PNG is lossless at every level, so the DECODED pixels the
    model sees are byte-identical to the zlib default (level 6) — only the file
    size and, crucially, the CPU encode time differ.  On a single-core box the
    per-call PNG encode of a 15-MP CAD page (~1.25s at level 6) serializes under
    the GIL and throttles LLM concurrency; level 1 cuts that ~40% (~0.74s) for a
    <2% size increase.  Not an algorithm change — the image content is identical."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", compress_level=1)
    data = buf.getvalue()
    if len(data) <= GEMINI_INLINE_BYTES_LIMIT:
        return data, "image/png"
    # too big: re-encode as JPEG quality 92 (lossy but still pixel-precise for line art)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue(), "image/jpeg"
