"""矢量文字层提取 —— PDF 原生排版行 → 0-1000 归一化文字框.

Text comes out as the PDF's native typographic LINES (rawdict
blocks→lines→chars), split further on big in-line gaps — so a box always
hugs one visible run of text, never a page-wide cluster.  Page /Rotate is
baked in (same frame the VLM sees).

Parallelism: MuPDF is NOT thread-safe (even across separate Document
objects — a shared global allocator/font state segfaulted the server on
concurrent renders).  So multi-page extraction is fanned out across
*processes* (each has its own MuPDF context), never threads.  Each worker
opens the PDF once for its whole slice instead of re-opening per page.
"""
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import fitz

from core.pdfio import FITZ_LOCK

GAP_SPLIT = 2.5  # split a native line where the char gap exceeds this × size
                 # (CAD PDFs sometimes draw a whole table row as one text op)


def _emit(frag, mat, W, H, out):
    """One run of chars → normalized 0-1000 box + text."""
    text = "".join(c for c, _ in frag).strip()
    if not text:
        return
    # trim leading/trailing space glyphs so their advance width doesn't
    # widen the box past the visible text
    a, b = 0, len(frag)
    while a < b and frag[a][0].isspace():
        a += 1
    while b > a and frag[b - 1][0].isspace():
        b -= 1
    core = frag[a:b]
    x0 = min(bb[0] for _, bb in core)
    y0 = min(bb[1] for _, bb in core)
    x1 = max(bb[2] for _, bb in core)
    y1 = max(bb[3] for _, bb in core)
    r = fitz.Rect(x0, y0, x1, y1) * mat
    bx0, bx1 = sorted((r.x0, r.x1))
    by0, by1 = sorted((r.y0, r.y1))
    out.append({
        "box_2d": [round(by0 / H * 1000, 1), round(bx0 / W * 1000, 1),
                   round(by1 / H * 1000, 1), round(bx1 / W * 1000, 1)],
        "text": text,
    })


def _extract_page(page):
    """Core extraction from an already-open page (no open, no lock).
    Returns {"lines": [{"box_2d", "text"}], "has_text": bool}."""
    mat = page.rotation_matrix          # unrotated pts → rotated frame
    W = page.rect.width or 1.0          # rect already reflects /Rotate
    H = page.rect.height or 1.0
    lines, has_text = [], False
    for b in page.get_text("rawdict").get("blocks", []):
        if b.get("type") != 0:
            continue
        for ln in b.get("lines", []):
            dx, dy = (ln.get("dir") or (1, 0))[:2]
            horiz = abs(dx) >= abs(dy)
            frag, prev = [], None
            for sp in ln.get("spans", []):
                for ch in sp.get("chars", []):
                    c = ch.get("c") or " "
                    bb = ch["bbox"]
                    if c.isspace():
                        frag.append((" ", bb))
                        continue
                    has_text = True
                    if prev is not None:
                        size = max(bb[3] - bb[1], bb[2] - bb[0], 1e-3)
                        gap = (bb[0] - prev[2]) if horiz \
                            else (bb[1] - prev[3])
                        if gap > GAP_SPLIT * size:
                            _emit(frag, mat, W, H, lines)
                            frag = []
                    frag.append((c, bb))
                    prev = bb
            _emit(frag, mat, W, H, lines)
    lines.sort(key=lambda it: (it["box_2d"][0], it["box_2d"][1]))
    return {"lines": lines, "has_text": has_text}


def vector_scan(pdf_path, page_index):
    """One page's text lines in the 0-1000 normalized render frame.
    Single-shot, thread-locked (safe to call from the web request threads)."""
    with FITZ_LOCK:
        doc = fitz.open(str(pdf_path))
        try:
            return _extract_page(doc.load_page(page_index))
        finally:
            doc.close()


# Per-worker state: the PDF is opened ONCE per process in the pool
# initializer and reused across every chunk that process handles (re-opening
# a big PDF per chunk would erase the parallelism win).
_WORKER = {}


def _worker_init(pdf_path):
    """Runs once per pool process (own MuPDF context → no lock needed)."""
    _WORKER["doc"] = fitz.open(pdf_path)


def _worker_scan(indices):
    """Scan a slice of 0-based page indices reusing this process's open doc."""
    doc = _WORKER["doc"]
    out = {}
    for idx in indices:
        try:
            out[idx] = _extract_page(doc.load_page(idx))
        except Exception as e:                                  # noqa: BLE001
            out[idx] = {"lines": [], "has_text": False,
                        "error": f"{type(e).__name__}: {e}"}
    return out


def _mp_context():
    """forkserver on POSIX (clean single-threaded parent → avoids the
    fork-from-a-threaded-gunicorn-worker deadlock hazard); default (spawn)
    on Windows."""
    try:
        if os.name == "posix":
            return mp.get_context("forkserver")
    except Exception:                                           # noqa: BLE001
        pass
    return None


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def vector_scan_pages(pdf_path, page_indices, workers=1,
                      on_chunk=None, should_cancel=None):
    """Extract many pages, fanning out across ``workers`` processes when
    >1 (MuPDF forbids threads).  Opens the PDF once per worker.

    ``on_chunk(batch)`` is invoked with each freshly-scanned
    ``{page_index: rec}`` batch so the caller can checkpoint/report progress
    incrementally.  ``should_cancel()`` truthy → stop launching/collecting
    more work (in-flight chunks still finish).  Returns ``{page_index: rec}``.
    """
    indices = list(page_indices)
    results = {}
    if not indices:
        return results
    eff = max(1, min(int(workers or 1), len(indices)))

    def _serial(idx_list):
        buf = {}
        with FITZ_LOCK:
            doc = fitz.open(str(pdf_path))
            try:
                for idx in idx_list:
                    if should_cancel and should_cancel():
                        break
                    try:
                        rec = _extract_page(doc.load_page(idx))
                    except Exception as e:                      # noqa: BLE001
                        rec = {"lines": [], "has_text": False,
                               "error": f"{type(e).__name__}: {e}"}
                    results[idx] = rec
                    buf[idx] = rec
                    if on_chunk and len(buf) >= 20:
                        on_chunk(dict(buf))
                        buf = {}
            finally:
                doc.close()
        if on_chunk and buf:
            on_chunk(dict(buf))

    # below this many pages the process-spawn overhead outweighs the win, so
    # scan in-process (still opens the PDF only once).  env-tunable.
    min_par = int(os.environ.get("VEC_MIN_PARALLEL", "24"))
    if eff == 1 or len(indices) < min_par:
        _serial(indices)
        return results

    # ~4 chunks per worker for load balance; bounded so checkpoints stay
    # frequent and per-task pickle payloads stay small.
    chunk = min(25, max(8, (len(indices) // (eff * 4)) or 1))
    parts = _chunks(indices, chunk)
    try:
        with ProcessPoolExecutor(max_workers=eff, mp_context=_mp_context(),
                                 initializer=_worker_init,
                                 initargs=(str(pdf_path),)) as ex:
            futs = {ex.submit(_worker_scan, part): part for part in parts}
            for f in as_completed(futs):
                batch = f.result()
                results.update(batch)
                if on_chunk:
                    on_chunk(batch)
                if should_cancel and should_cancel():
                    for fu in futs:
                        fu.cancel()
                    break
    except Exception as e:                                      # noqa: BLE001
        # any pool/pickle/spawn failure → finish the rest in-process, never
        # fail the run over a parallelism problem.
        rest = [i for i in indices if i not in results]
        print(f"  [vec] parallel pool failed ({type(e).__name__}: {e}); "
              f"finishing {len(rest)} pages serially", flush=True)
        _serial(rest)
    return results
