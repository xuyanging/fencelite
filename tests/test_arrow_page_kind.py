"""Image-only arrow-page classification and UI status regression."""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")
os.environ.setdefault("TEXT_WORKERS", "1")
os.environ.setdefault("SYMBOLS_WORKERS", "1")
os.environ.setdefault("VIEW_WORKERS", "1")
os.environ.setdefault("VEC_WORKERS", "1")

import fitz
from PIL import Image

from steps.arrows import page_geometry_status


class _Pdf:
    def __init__(self, image_rect, *, vector=False):
        handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        handle.close()
        self.path = handle.name
        picture = Image.new("RGB", (600, 400), "white")
        stream = io.BytesIO()
        picture.save(stream, "PNG")
        doc = fitz.open()
        page = doc.new_page(width=600, height=400)
        page.insert_image(fitz.Rect(*image_rect), stream=stream.getvalue())
        if vector:
            page.draw_line((20, 20), (580, 380), color=(0, 0, 0), width=1)
        doc.save(self.path)
        doc.close()

    def close(self):
        try:
            os.unlink(self.path)
        except OSError:
            pass


class PageGeometryTests(unittest.TestCase):
    def setUp(self):
        self.docs = []

    def tearDown(self):
        for doc in self.docs:
            doc.close()

    def _make(self, rect, **kw):
        doc = _Pdf(rect, **kw)
        self.docs.append(doc)
        return doc.path

    def test_full_page_raster_is_marked_image_only(self):
        got = page_geometry_status(self._make((0, 0, 600, 400)), 0)
        self.assertEqual(got["state"], "image-only")
        self.assertEqual(got["vector_paths"], 0)
        self.assertGreaterEqual(got["image_coverage"], .9)

    def test_small_logo_does_not_mark_the_page(self):
        got = page_geometry_status(self._make((0, 0, 60, 40)), 0)
        self.assertEqual(got["state"], "vector")

    def test_any_real_vector_path_keeps_vector_processing_enabled(self):
        got = page_geometry_status(
            self._make((0, 0, 600, 400), vector=True), 0)
        self.assertEqual(got["state"], "vector")
        self.assertGreater(got["vector_paths"], 0)


class ImageOnlyStatusTests(unittest.TestCase):
    def test_current_image_only_cache_is_exposed_as_its_own_state(self):
        import webapp

        entry = {"sig": "same", "items": {}, "page_kind": "image-only"}
        record = {}
        with patch.object(webapp.arrows, "ENABLED", True), \
                patch.object(webapp.arrows, "arrows_signature",
                             return_value="same"), \
                patch.object(webapp, "_placement_anchors_for", return_value=[]), \
                patch.object(webapp.store, "load_json",
                             return_value={"1": entry}):
            webapp._attach_arrows(
                record, "sample", 1,
                [{"text": "FENCE", "box_2d": [1, 2, 3, 4]}], "rev", [])
        self.assertEqual(record["arrows_status"]["state"], "image-only")
        self.assertEqual(record["arrows"], {})


if __name__ == "__main__":
    unittest.main()
