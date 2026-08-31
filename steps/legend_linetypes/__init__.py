"""Independent supervised line-type channel for legend swatches.

The existing :mod:`steps.linetypes` cache represents line types selected by
arrow terminals.  A line sample explicitly boxed in a legend is stronger
evidence: it is guaranteed to be a line type even when the box contains only
one or two periods.  Keeping that evidence in ``legend_linetypes/<page>.json``
prevents either algorithm from masquerading as the other and lets the two
outputs be merged explicitly at publication time.

Cache shape (version 1)::

    {"sig": "...", "v": 1, "ok": true, ...sidecar payload fields...}

An error record may be persisted for diagnostics, but it has no ``ok: true``
and :func:`has_current` will reject it.  Therefore a crash, timeout, malformed
response, or missing sidecar can never become a successful empty cache.
"""
from __future__ import annotations

import hashlib
import json
import math

from steps import pagestore, store
from steps.legend_linetypes import sidecar

VERSION = 1
CACHE_KIND = "legend_linetypes"


def _symbol_result(value):
    """Accept either ``symbols.json[p].result`` or the whole page entry."""
    if not isinstance(value, dict):
        return {}
    result = value.get("result")
    return result if isinstance(result, dict) else value


def _normal_box(box):
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in box):
        return None
    out = [float(value) for value in box]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def samples_of(symbol_result):
    """Return canonical supervised samples from published ``line`` symbols.

    ``symbol_index`` always refers to the original symbols array, including
    intervening shape symbols.  Invalid rows are skipped fail-closed: the
    symbol publication contract requires a real owner ``text_index`` and a
    finite positive-area box, and inventing either would make later ownership
    or cache invalidation ambiguous.
    """
    out = []
    symbols = _symbol_result(symbol_result).get("symbols")
    for symbol_index, symbol in enumerate(
            symbols if isinstance(symbols, list) else ()):
        if not isinstance(symbol, dict) \
                or str(symbol.get("category") or "").strip().lower() != "line":
            continue
        text_index = symbol.get("text_index")
        if (isinstance(text_index, bool) or not isinstance(text_index, int)
                or text_index < 0):
            continue
        box = _normal_box(symbol.get("box_2d"))
        if box is None:
            continue
        out.append({
            "symbol_index": symbol_index,
            "text_index": text_index,
            "box_2d": box,
            "value": str(symbol.get("value") or ""),
            "source": str(symbol.get("source") or ""),
        })
    return out


def _canonical_samples(samples):
    """Copy canonical rows and reject signature/protocol disagreement."""
    out = []
    seen = set()
    for row in samples if isinstance(samples, (list, tuple)) else ():
        wire = sidecar._wire_sample(row)  # noqa: SLF001
        index = wire["symbol_index"]
        if index in seen:
            raise ValueError(f"duplicate legend symbol_index: {index}")
        seen.add(index)
        box = wire["box_2d"]
        if (not all(math.isfinite(value) for value in box)
                or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError(f"invalid legend sample box_2d: {box!r}")
        out.append(wire)
    return out


def signature(pdf_revision, samples):
    """Cache identity: PDF revision + every sample field + complete producer.

    The caller normally passes :func:`samples_of`.  Accepting a symbols result
    as a convenience is safe because it is normalised through that same
    function before hashing.
    """
    if isinstance(samples, dict):
        samples = samples_of(samples)
    canonical = _canonical_samples(samples)
    payload = {
        "pdf_revision": str(pdf_revision or ""),
        "samples": canonical,
        "producer": sidecar.producer_digest(),
        "version": VERSION,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# Descriptive alias for callers that use the ``*_signature`` convention.
legend_linetypes_signature = signature


def has_current(entry, sig):
    """Whether ``entry`` is an explicit successful result for ``sig``."""
    return bool(isinstance(entry, dict) and entry.get("sig") == sig
                and entry.get("v") == VERSION and entry.get("ok") is True)


def page_path(slug, page):
    return pagestore.page_path(slug, CACHE_KIND, page)


def load(slug, page):
    return pagestore.load_page(slug, CACHE_KIND, page, None)


def save(slug, page, entry):
    """Atomically save one page; failures remain retryable via has_current()."""
    pagestore.save_page(slug, CACHE_KIND, page, entry)


# Explicit page-suffixed aliases make orchestration code self-documenting.
load_page = load
save_page = save


def computed_pages(slug):
    return pagestore.pages_of(slug, CACHE_KIND)


def sidecar_available():
    return sidecar.sidecar_available()


def compute_page(pdf_path, sheet, samples_or_symbol_result, *, sig=None,
                 pdf_revision=None, cpu_budget=None, timeout=None, dbg=None):
    """Compute and wrap one successful supervised sidecar response.

    Nothing is written here.  The orchestrator saves the returned entry, so an
    exception cannot accidentally overwrite a previous good result.  ``sig``
    can be supplied by the job collector; otherwise it is derived from the PDF
    on disk and the exact samples sent to the sidecar.
    """
    # Job collectors commonly materialise samples once to calculate the
    # signature and then pass that same list here.  Accept the symbols result
    # too for direct callers, but in both cases send the exact canonical rows
    # that were (or will be) signed.
    if isinstance(samples_or_symbol_result, (list, tuple)):
        samples = _canonical_samples(samples_or_symbol_result)
    else:
        samples = samples_of(samples_or_symbol_result)
    if not samples:
        raise ValueError("no legend line samples")
    if sig is None:
        revision = (pdf_revision if pdf_revision is not None
                    else store.pdf_revision(pdf_path))
        sig = signature(revision, samples)
    payload = sidecar.run_page(
        pdf_path, sheet, samples, cpu_budget=cpu_budget,
        timeout=timeout, dbg=dbg)
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        # _run_job enforces this in production; retain the check at this API
        # boundary so a replaced/test runner cannot create a false cache.
        raise RuntimeError("legend line-type sidecar returned no successful payload")
    entry = dict(payload)
    # Sidecar-owned metadata may never spoof the caller's cache identity.
    entry["sig"] = str(sig)
    entry["v"] = VERSION
    return entry


__all__ = [
    "CACHE_KIND", "VERSION", "computed_pages", "compute_page",
    "has_current", "legend_linetypes_signature", "load", "load_page",
    "page_path", "samples_of", "save", "save_page", "sidecar",
    "sidecar_available", "signature",
]
