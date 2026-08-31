"""Supervised legend-line matcher sidecar transport.

This module deliberately has its own runner and cache identity.  The ordinary
``steps.linetypes`` runner starts from arrow terminals; legend samples are a
different, supervised input and must not silently change (or invalidate) that
pipeline's result format.

Only the process-management machinery is shared.  In particular,
``steps.linetypes.sidecar._run_job`` owns the cross-platform timeout contract:
the complete multiprocessing process tree is killed before its pipes are
reaped.  Reimplementing that subtle behaviour here would make the two runners
fail differently on dense sheets.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from steps.linetypes import sidecar as _base_sidecar

_BASE_DIR = Path(__file__).resolve().parent.parent.parent
_SIDECAR_DIR = _BASE_DIR / "tools" / "linetype_sidecar"
_RUNNER = _SIDECAR_DIR / "run_legend.py"
_HELPER = _SIDECAR_DIR / "legend_supervised.py"
_ENGINE_DIR = _SIDECAR_DIR / "engine" / "line_type_engine"

# Method2 has structurally different implementations for worker_count == 1
# and worker_count >= 2.  The ordinary runner has corpus evidence for its own
# scheduling plan, but that does not prove invariance for this new supervised
# composition path.  Pin this channel to one worker until such a proof exists;
# accepting a larger value without putting it in the signature could otherwise
# let two different algorithms share one cache key.
TIMEOUT = int(os.environ.get(
    "LEGEND_LINETYPE_TIMEOUT", str(_base_sidecar.TIMEOUT)))
CPU_BUDGET = 1


def _single_worker_budget(requested):
    """Return the only supported budget, rejecting argument/env overrides."""
    source = "cpu_budget argument"
    raw = requested
    if raw is None:
        source = "LEGEND_LINETYPE_CPU_BUDGET"
        raw = os.environ.get("LEGEND_LINETYPE_CPU_BUDGET", str(CPU_BUDGET))
    if isinstance(raw, bool):
        raise ValueError(f"{source} must be 1 for deterministic legend matching")
    try:
        budget = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{source} must be 1 for deterministic legend matching") from exc
    # Do not accept 1.5 merely because int(1.5) truncates to 1.  Environment
    # values are strings; direct callers should provide the actual integer.
    if budget != CPU_BUDGET or str(raw).strip() not in ("1", "1.0"):
        raise ValueError(f"{source} must be 1 for deterministic legend matching")
    return CPU_BUDGET


def sidecar_available():
    """Whether every executable part of the supervised sidecar is installed."""
    return (Path(_RUNNER).is_file() and Path(_HELPER).is_file()
            and Path(_base_sidecar._PYTHON).is_file()  # noqa: SLF001
            and Path(_ENGINE_DIR).is_dir())


def _file_digest(path):
    path = Path(path)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        # A missing producer must still have a deterministic identity.  It will
        # fail sidecar_available(), but callers can safely calculate expected
        # cache signatures while reporting that state.
        return "missing"


def producer_digest():
    """Content identity of the supervised legend-line producer.

    ``engine_digest`` already covers the complete vendored engine tree and the
    PyMuPDF/pypdf/scipy/numpy versions in its private venv.  We add the two
    files unique to this channel: the JSON composition root and the supervised
    extraction/matching helper.  Consequently a matcher threshold, paint-order
    rule, protocol projection, engine implementation, or numerical dependency
    update invalidates this cache automatically.

    The shared digest conservatively also contains the ordinary ``run.py``.
    That may cause an extra supervised-cache refresh when the arrow runner
    alone changes, but never permits a stale result; importantly, changes here
    do *not* alter the ordinary line-type cache identity.
    """
    payload = {
        "runner": _file_digest(_RUNNER),
        "helper": _file_digest(_HELPER),
        "engine_and_deps": _base_sidecar.engine_digest(),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _wire_sample(row):
    """Validate and copy one already-normalised sample for the JSON protocol."""
    if not isinstance(row, dict):
        raise ValueError(f"legend line sample must be an object, got {row!r}")
    symbol_index = row.get("symbol_index")
    text_index = row.get("text_index")
    if (isinstance(symbol_index, bool) or not isinstance(symbol_index, int)
            or symbol_index < 0):
        raise ValueError(f"invalid legend symbol_index: {symbol_index!r}")
    if (isinstance(text_index, bool) or not isinstance(text_index, int)
            or text_index < 0):
        raise ValueError(f"invalid legend text_index: {text_index!r}")
    box = row.get("box_2d")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        raise ValueError(f"invalid legend sample box_2d: {box!r}")
    try:
        box = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid legend sample box_2d: {box!r}") from exc
    if (not all(math.isfinite(value) for value in box)
            or box[2] <= box[0] or box[3] <= box[1]):
        raise ValueError(f"invalid legend sample box_2d: {box!r}")
    return {
        "symbol_index": symbol_index,
        "text_index": text_index,
        "box_2d": box,
        "value": str(row.get("value") or ""),
        "source": str(row.get("source") or ""),
    }


def run_page(pdf_path, sheet, samples, *, cpu_budget=None, timeout=None,
             dbg=None):
    """Run one 1-based PDF sheet through ``run_legend.py``.

    The wire input is intentionally limited to the documented four fields:
    ``pdf``, ``sheet``, the fully normalised ``samples``, and ``cpu_budget``.
    Any sidecar failure is raised by the shared runner and is therefore never
    confused with a valid page containing zero matches.
    """
    budget = _single_worker_budget(cpu_budget)
    if not sidecar_available():
        python = Path(_base_sidecar._PYTHON)  # noqa: SLF001
        raise RuntimeError(
            "legend line-type sidecar missing: "
            f"python={python} exists={python.is_file()} "
            f"runner={_RUNNER} exists={Path(_RUNNER).is_file()} "
            f"helper={_HELPER} exists={Path(_HELPER).is_file()} "
            f"engine={_ENGINE_DIR} exists={Path(_ENGINE_DIR).is_dir()}")
    if not isinstance(sheet, int) or isinstance(sheet, bool) or sheet < 1:
        raise ValueError(f"sheet must be a 1-based int, got {sheet!r}")
    wire_samples = [_wire_sample(row) for row in (samples or ())]
    if not wire_samples:
        raise ValueError("no legend line samples")
    payload = {
        "pdf": str(pdf_path),
        "sheet": int(sheet),
        "samples": wire_samples,
        "cpu_budget": budget,
    }
    return _base_sidecar._run_job(  # noqa: SLF001
        _RUNNER, payload, sheet=sheet,
        timeout=timeout if timeout is not None else TIMEOUT,
        dbg=dbg, label="legend line-type sidecar")


__all__ = [
    "CPU_BUDGET", "TIMEOUT", "producer_digest", "run_page",
    "sidecar_available",
]
