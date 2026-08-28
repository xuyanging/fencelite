"""Thread-safe PDF page rendering for the 5054 pipeline."""
import threading
from pathlib import Path

import fitz
from PIL import Image

# MuPDF is NOT thread-safe even across separate Document objects (global
# allocator/font state). Concurrent renders segfaulted the server on
# 2026-07-03 (access violation in mupdfcpp64.dll) — every fitz call site in
# the process must hold this lock.
FITZ_LOCK = threading.RLock()

from core.config import MAX_RENDER_PX, RENDER_DPI


def render_pdf_page(pdf_path: Path, page_index: int, dpi: int = RENDER_DPI,
                    max_px: int = MAX_RENDER_PX, doc=None) -> Image.Image:
    """Rasterize one page. Pass an already-open `doc` to skip the per-call
    fitz.open — callers that loop over many pages of the same PDF should own
    one document per thread (fitz docs are not thread-safe across threads)."""
    with FITZ_LOCK:
        own_doc = doc is None
        d = fitz.open(pdf_path) if own_doc else doc
        try:
            if page_index < 0 or page_index >= d.page_count:
                raise IndexError(f"page {page_index} out of range (total {d.page_count})")
            page = d.load_page(page_index)
            # Start from the requested DPI, then shrink if the resulting raster would
            # exceed max_px on its long side. Single-pass matrix scaling — no second
            # resample step, no extra memory.
            zoom = dpi / 72
            long_side = max(page.rect.width, page.rect.height) * zoom
            if max_px and long_side > max_px:
                zoom *= max_px / long_side
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            if own_doc:
                d.close()
