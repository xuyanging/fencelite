"""LLM 文字判词 —— 纯文本（零图片token）判断字符串是否 fence 相关.

LLM text judge = semantics without a keyword dictionary.  Each project's
unique line strings go to Gemini as PLAIN TEXT and it decides which read as
fence-related.  Verdicts are cached per project (textjudge.json), so re-runs
and rescans are free.

KW floor = a tiny literal-word regex kept as a zero-cost deterministic
safety net under the judge.  Not meant to be maintained or extended —
semantics belong to the judge.
"""
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import resolve_model, submit_with_context
from core.gemini import gen_json, should_retry_model_error
from core.parsing import parse_json_value
from steps.text.target import TARGET_DEFAULT, build_judge_prompt
from steps.versions import TEXT_JUDGE_VERSION

# Literal floor only. Semantic terms (gate, barbed, mesh, CLF...) are the
# judge's job — do NOT grow this list.  Fence-specific, so it is only applied
# for the default fence target (use_kw_floor); a custom target has no floor.
KW = re.compile(r"fenc|chain[ -]?link", re.I)


def prepare_judge_cache(cache, model):
    """Normalize the text-judge cache and enforce its prompt version.

    Version-less pre-refactor caches are v1, so this upgrade is free. A
    future version bump intentionally clears verdicts made under different
    prompt semantics.
    """
    if not isinstance(cache, dict) or cache.get("v", 1) != TEXT_JUDGE_VERSION:
        return {"v": TEXT_JUDGE_VERSION, "model": model, "verdicts": {}}
    verdicts = cache.get("verdicts")
    if not isinstance(verdicts, dict):
        verdicts = {}
    return {**cache, "v": TEXT_JUDGE_VERSION, "model": model,
            "verdicts": verdicts}


def norm_text(s):
    return " ".join(str(s).split()).upper()


def judge_candidates(lines_by_page, use_kw_floor=True):
    """Unique normalized strings worth sending to the judge: at least one
    letter, 3-400 chars.  With the fence keyword floor on (default), strings the
    floor already catches are skipped; for a custom target (floor off) every
    string is judged."""
    cand = {}
    for lines in lines_by_page:
        for ln in lines:
            t = norm_text(ln["text"])
            if not (3 <= len(t) <= 400) or not re.search(r"[A-Za-z]", t):
                continue
            if use_kw_floor and KW.search(t):
                continue
            cand.setdefault(t, ln["text"])
    return cand  # {norm: original example}


# Default judge prompt = fixed scaffolding wrapped around the default fence
# target (same target the image VLM uses).  Pass a custom-target prompt to
# judge_strings(judge_prompt=...) to unify both tasks under one edited target.
JUDGE_PROMPT = build_judge_prompt(TARGET_DEFAULT)

JUDGE_CHUNK = 400
# Short-string classification needs no deep reasoning; uncapped, thinking
# tokens dominated the judge's cost (~3× the input cost on grand_island).
# 128 is the documented minimum for pro models.
JUDGE_THINKING_BUDGET = 128
# Independent chunks run concurrently (text-only calls, network-bound).  This
# does NOT change verdicts — every string is judged by the same prompt in the
# same chunk; only the wall time drops.  env-tunable (too high → 429 backoff).
JUDGE_WORKERS = int(os.environ.get("JUDGE_WORKERS", "4"))


def judge_strings(norm_strings, model=None, chunk=JUDGE_CHUNK, on_progress=None,
                  on_chunk=None, judge_prompt=None):
    """Classify normalized strings via text-only Gemini calls.
    Returns ({norm: bool}, usage_totals).  on_chunk(chunk_verdicts) fires after
    every successful chunk so callers can checkpoint paid results.  Raises only
    if a whole chunk fails after retries — caller decides how to degrade.
    ``judge_prompt`` defaults to the fence prompt; pass a custom-target prompt
    (target.build_judge_prompt) to judge against the same target the image VLM
    uses.  Chunks are judged concurrently; the merged verdict set is identical
    to a serial run (each chunk classifies a disjoint set of strings)."""
    jp = judge_prompt or JUDGE_PROMPT
    verdicts = {}
    usage_tot = {"input_tokens": 0, "output_tokens": 0, "thoughts_tokens": 0}
    strings = sorted(norm_strings)
    mdl = resolve_model(model)
    batches = [(ci, strings[ci:ci + chunk])
               for ci in range(0, len(strings), chunk)]
    if not batches:
        return verdicts, usage_tot

    def run_chunk(ci, batch):
        payload = json.dumps([{"id": i, "t": s} for i, s in enumerate(batch)],
                             ensure_ascii=False)
        last_err = None
        for attempt in range(3):
            try:
                resp = gen_json(mdl, [jp + "\n\n" + payload],
                                timeout_ms=300_000,
                                thinking_budget=JUDGE_THINKING_BUDGET)
                ids = parse_json_value(resp.text or "")
                if not isinstance(ids, list):
                    raise ValueError(f"unparseable judge response: "
                                     f"{(resp.text or '')[:160]!r}")
                flagged = {int(i) for i in ids if isinstance(i, (int, float))}
                cv = {s: (i in flagged) for i, s in enumerate(batch)}
                return cv, getattr(resp, "usage_metadata", None)
            except Exception as e:                          # noqa: BLE001
                last_err = e
                if not should_retry_model_error(e, attempt, 3):
                    raise RuntimeError(
                        f"judge chunk {ci // chunk} failed: {last_err}") from e
                time.sleep(15 * (attempt + 1))

    lock = threading.Lock()
    done = [0]
    failures = []
    workers = max(1, min(JUDGE_WORKERS, len(batches)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [submit_with_context(ex, run_chunk, ci, batch)
                for ci, batch in batches]
        for f in as_completed(futs):
            try:
                cv, um = f.result()
            except Exception as exc:                         # noqa: BLE001
                # Do not abandon the other paid chunks.  They may already have
                # succeeded; consuming and checkpointing every future is what
                # makes a partial provider outage resumable instead of losing
                # valid verdicts and paying for them again.
                failures.append(exc)
                continue
            with lock:
                verdicts.update(cv)
                if um:
                    usage_tot["input_tokens"] += um.prompt_token_count or 0
                    usage_tot["output_tokens"] += um.candidates_token_count or 0
                    usage_tot["thoughts_tokens"] += \
                        getattr(um, "thoughts_token_count", 0) or 0
                if on_chunk:
                    on_chunk(cv)
                done[0] += len(cv)
                if on_progress:
                    on_progress(min(done[0], len(strings)), len(strings))
    if failures:
        first = failures[0]
        raise RuntimeError(
            f"{len(failures)} of {len(batches)} judge chunks failed; "
            f"first error: {first}") from first
    return verdicts, usage_tot


def select_instances(lines, flagged, use_kw_floor=True):
    """Matching vector text lines = (fence keyword floor, if on) ∪ judge-flagged.
    For a custom target the floor is off, so only judge-flagged lines count."""
    return [ln for ln in lines
            if (use_kw_floor and KW.search(ln["text"]))
            or norm_text(ln["text"]) in flagged]
