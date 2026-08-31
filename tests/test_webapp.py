"""webapp.py 的离线回归 —— 全程零模型调用.

只用：临时目录 + fitz 现造的空白 PDF + 手写的当期缓存 JSON + 纯几何断言。
data/ 和 projects/ 都被 monkeypatch 到 tmp 目录，绝不碰真实项目数据。

并行开发期的兜底：`job` 与 `steps.symbols` 由别的模块负责。缺哪个就为它装一个
最小 stub（严格按契约实现），并在 STUBBED 里记下来，这样这一层的路由/发布闸/
装配逻辑可以先自证，不必等整个管线合并完。
"""
import io
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import threading
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz

from core.config import MODEL_NAME
from core.pdfio import FITZ_LOCK
from steps import store
from steps.versions import (FUSED_VERSION, PLACEMENT_VERSION, SYMBOL_PROMPT_V,
                            SYMBOL_VERSION, VIEW_VERSION)
from steps.views import view_signature

STUBBED = []


def _stub_job():
    """job.py 的最小契约替身（不跑任何管线、不花钱）。"""
    module = types.ModuleType("job")

    def page_count_of(slug):
        try:
            with FITZ_LOCK:
                with fitz.open(store.pdf_path(slug)) as doc:
                    return doc.page_count
        except Exception:                                       # noqa: BLE001
            return 0

    def delete_project(slug):
        removed = []
        for root in (store.PROJECTS_DIR / slug, store.DATA_DIR / slug):
            if root.exists():
                shutil.rmtree(root, ignore_errors=True)
                removed.append(str(root))
        marker = store.JOBS_DIR / f"{slug}.json"
        if marker.exists():
            marker.unlink()
            removed.append(str(marker))
        return removed

    module.get_job = lambda slug: {}
    module.all_jobs = lambda: {}
    module.job_running = lambda slug: False
    module.page_count_of = page_count_of
    module.request_cancel = lambda slug: False
    module.create_project = lambda data, filename: "stub"
    module.start_job = lambda slug, target=None: {"slug": slug}
    module.delete_project = delete_project
    module.resume_interrupted = lambda: []
    return module


try:
    import job                                                  # noqa: F401
except ImportError:
    sys.modules["job"] = _stub_job()
    STUBBED.append("job")

try:
    from steps.symbols import symbols_dropped_view               # noqa: F401
    from steps.symbols import sweep_version as _sweep_version
except ImportError:
    _symbols = types.ModuleType("steps.symbols")
    _symbols.symbols_dropped_view = lambda entry: []
    _symbols.sweep_version = lambda: None
    sys.modules["steps.symbols"] = _symbols
    _sweep_version = _symbols.sweep_version
    STUBBED.append("steps.symbols")

import job as job_module
import webapp

webapp.app.config["TESTING"] = True

SLUG = "unit_demo"
XSITE = {"Sec-Fetch-Site": "cross-site"}
_PATCHED_DIRS = ("DATA_DIR", "PROJECTS_DIR", "JOBS_DIR")


class WebappCase(unittest.TestCase):
    """把 store（以及 job 里同名的目录常量）指向临时目录。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fence_lite_web_"))
        self.saved = {}
        dirs = {"DATA_DIR": self.tmp / "data",
                "PROJECTS_DIR": self.tmp / "projects",
                "JOBS_DIR": self.tmp / "_jobs"}
        for value in dirs.values():
            value.mkdir(parents=True, exist_ok=True)
        for module in (store, job_module):
            for name in _PATCHED_DIRS:
                if hasattr(module, name):
                    self.saved[(module, name)] = getattr(module, name)
                    setattr(module, name, dirs[name])
        self.dirs = dirs
        self.client = webapp.app.test_client()

    def tearDown(self):
        for (module, name), value in self.saved.items():
            setattr(module, name, value)
        shutil.rmtree(self.tmp, ignore_errors=True)

    # --------------------------------------------------------------- fixtures --
    def make_pdf(self, slug=SLUG, pages=1):
        directory = self.dirs["PROJECTS_DIR"] / slug
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "input.pdf"
        with FITZ_LOCK:
            doc = fitz.open()
            for _ in range(pages):
                doc.new_page(width=612, height=792)
            doc.save(str(path))
            doc.close()
        return path

    def write_results(self, rec=None, slug=SLUG, **over):
        """写一份「当期」results.json（真实 pdf_revision + 当期 fused_v）。"""
        pdf = store.pdf_path(slug)
        payload = {"slug": slug, "fused_v": FUSED_VERSION,
                   "pdf_revision": store.pdf_revision(pdf),
                   "page_count": 1, "mode": "fence",
                   "pages": {"1": rec} if rec is not None else {}}
        payload.update(over)
        store.save_json(store.results_path(slug), payload)
        return payload

    def sample_rec(self):
        return {
            "vlm_items": [
                {"text": "6' CHAIN LINK FENCE", "box_2d": [100, 100, 120, 300],
                 "label": "legend", "tbl": True},
                {"text": "FENCE LINE", "box_2d": [400, 500, 415, 610],
                 "label": "callout"},
            ],
            "vec_added": [
                {"text": "GATE", "box_2d": [700, 200, 715, 260], "label": ""},
            ],
            "vec_covered": [
                {"text": "FENCE LINE", "box_2d": [401, 501, 414, 609]},
            ],
            "has_text": True, "codes_stripped": 2,
            "debug": {"vector_candidates": []},
        }

    def write_symbols(self, rec, slug=SLUG, sig=None, placements=2):
        items = store.items_of(rec)
        revision = store.pdf_revision(store.pdf_path(slug))
        symbol = {"category": "shape", "value": "F1", "text_index": 0,
                  "group_index": 0, "box_2d": [100, 60, 118, 90],
                  "placements": [[500 + 10 * i, 300, 512 + 10 * i, 316]
                                 for i in range(placements)]}
        line = {"category": "line", "value": "X", "text_index": 1,
                "group_index": 0, "box_2d": [400, 460, 410, 495]}
        groups = [{"kind": "legend", "box_2d": [80, 40, 200, 400]},
                  {"kind": "view", "box_2d": [300, 100, 900, 900]}]
        entry = {
            "sig": sig if sig is not None else store.sig_of(items, revision),
            "v": SYMBOL_VERSION, "pv": SYMBOL_PROMPT_V, "model": MODEL_NAME,
            # sweep_v：步骤②b 补扫也参与当期判定
            "sweep_v": _sweep_version() or 0,
            "raw": {"groups": groups, "symbols": [symbol, line]},
            "result": {"groups": groups, "symbols": [symbol, line],
                       "plc_v": PLACEMENT_VERSION,
                       "placement_note": "2 placements inside plan"},
        }
        store.save_json(store.slug_dir(slug) / "symbols.json", {"1": entry})
        self.write_view_types(groups, revision, slug=slug)
        return entry

    def write_view_types(self, groups, revision, slug=SLUG, **over):
        """当期 view_types.json：签名用 steps.views.view_signature 真算。"""
        entry = {"sig": view_signature(groups, revision), "v": VIEW_VERSION,
                 "model": MODEL_NAME,
                 "views": [{"group_index": 1, "view_type": "plan",
                            "reason": "top-down site plan"}]}
        entry.update(over)
        store.save_json(store.slug_dir(slug) / "view_types.json", {"1": entry})
        return entry


class UploadGuardTests(WebappCase):
    def test_valid_upload_uses_streaming_project_writer(self):
        with mock.patch.object(
                job_module, "create_project_stream",
                return_value="streamed") as create, \
                mock.patch.object(
                    job_module, "start_job",
                    return_value={"slug": "streamed", "done": False}):
            r = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 streamed"), "large.pdf")},
                content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["slug"], "streamed")
        create.assert_called_once()
        self.assertEqual(create.call_args.args[1], "large.pdf")

    def test_job_start_failure_removes_orphaned_upload(self):
        variant = "orphan__legacy_model"
        (self.dirs["PROJECTS_DIR"] / variant).mkdir()
        (self.dirs["PROJECTS_DIR"] / variant / "input.pdf").write_bytes(
            b"%PDF-1.4 historical variant")
        (self.dirs["DATA_DIR"] / variant).mkdir()
        (self.dirs["DATA_DIR"] / variant / "keep.json").write_text("{}")
        with mock.patch.object(
                job_module.threading.Thread, "start",
                side_effect=RuntimeError("thread unavailable")):
            r = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 streamed"), "orphan.pdf")},
                content_type="multipart/form-data")
        self.assertEqual(r.status_code, 500)
        self.assertIn("thread unavailable", r.get_json()["error"])
        self.assertFalse((self.dirs["PROJECTS_DIR"] / "orphan").exists())
        self.assertFalse((self.dirs["DATA_DIR"] / "orphan").exists())
        self.assertFalse((self.dirs["JOBS_DIR"] / "orphan.json").exists())
        self.assertNotIn("orphan", job_module.JOBS)
        self.assertTrue((self.dirs["PROJECTS_DIR"] / variant / "input.pdf").exists())
        self.assertTrue((self.dirs["DATA_DIR"] / variant / "keep.json").exists())

    def test_unknown_start_error_does_not_delete_possible_live_project(self):
        with mock.patch.object(
                job_module, "start_job",
                side_effect=RuntimeError("after start uncertainty")):
            r = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 streamed"), "possible_live.pdf")},
                content_type="multipart/form-data")
        self.assertEqual(r.status_code, 500)
        self.assertTrue(
            (self.dirs["PROJECTS_DIR"] / "possible_live" / "input.pdf").exists())

    def test_upload_token_replays_same_job_without_duplicate_project(self):
        token = "batchtoken0123456789abcdef"
        status = {"slug": "idempotent", "stage": "queued", "done": False}
        with mock.patch.object(job_module, "start_job",
                               return_value=status) as start:
            first = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 first"), "idempotent.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
            second = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 first"), "idempotent.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["slug"], first.get_json()["slug"])
        self.assertTrue(second.get_json()["deduplicated"])
        start.assert_called_once()
        self.assertEqual(
            sorted(p.name for p in self.dirs["PROJECTS_DIR"].iterdir()),
            ["idempotent"])

    def test_bad_upload_token_rejected_before_project_creation(self):
        r = self.client.post("/api/upload", data={
            "pdf": (io.BytesIO(b"%PDF-1.4 streamed"), "bad_token.pdf"),
            "upload_token": "../bad",
        }, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertFalse((self.dirs["PROJECTS_DIR"] / "bad_token").exists())

    def test_upload_token_never_attaches_to_reused_slug(self):
        token = "reusedslug0123456789abcdef"
        with mock.patch.object(
                job_module, "start_job",
                side_effect=lambda slug, target="": {
                    "slug": slug, "stage": "queued", "done": False,
                }) as start:
            first = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 original"), "reused.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
            # Simulate deletion/reuse or replacement of the path after the
            # original response was lost.  The old token must not claim this
            # now-different PDF merely because its slug is the same.
            (self.dirs["PROJECTS_DIR"] / "reused" / "input.pdf").write_bytes(
                b"%PDF-1.4 different project bytes")
            second = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 original"), "reused.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.get_json()["slug"], "reused_2")
        self.assertFalse(second.get_json().get("deduplicated", False))
        self.assertEqual(start.call_count, 2)

    def test_upload_token_rejects_different_detection_target(self):
        token = "targetbound0123456789abcdef"
        status = {"slug": "target_bound", "stage": "queued", "done": False}
        with mock.patch.object(job_module, "start_job",
                               return_value=status) as start:
            first = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 same"), "target_bound.pdf"),
                "target": "doors", "upload_token": token,
            }, content_type="multipart/form-data")
            second = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(b"%PDF-1.4 same"), "target_bound.pdf"),
                "target": "trees", "upload_token": token,
            }, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 409)
        self.assertIn("different", second.get_json()["error"])
        start.assert_called_once()

    def test_created_upload_marker_recovers_worker_without_new_project(self):
        token = "recovercreated0123456789abcdef"
        payload = b"%PDF-1.4 recover"
        with mock.patch.object(
                job_module, "start_job",
                side_effect=lambda slug, target="": {
                    "slug": slug, "stage": "queued", "done": False,
                }) as start:
            first = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(payload), "recover.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
            marker_path = webapp._upload_token_path(token)
            marker = store.load_json(marker_path, {})
            marker["state"] = "created"
            marker.pop("job", None)
            webapp._save_upload_token(marker_path, marker)
            job_module.JOBS.pop("recover", None)
            try:
                (self.dirs["JOBS_DIR"] / "recover.json").unlink()
            except FileNotFoundError:
                pass
            second = self.client.post("/api/upload", data={
                "pdf": (io.BytesIO(payload), "recover.pdf"),
                "upload_token": token,
            }, content_type="multipart/form-data")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(second.get_json()["deduplicated"])
        self.assertTrue(second.get_json()["recovered"])
        self.assertEqual(start.call_count, 2)
        self.assertEqual(
            sorted(p.name for p in self.dirs["PROJECTS_DIR"].iterdir()),
            ["recover"])

    def test_concurrent_same_token_starts_exactly_one_job(self):
        token = "concurrent0123456789abcdef"
        payload = b"%PDF-1.4 concurrent"
        entered = threading.Event()
        release = threading.Event()

        def start(slug, target=""):
            entered.set()
            release.wait(timeout=2)
            return {"slug": slug, "stage": "queued", "done": False}

        def post_once(_index):
            with webapp.app.test_client() as client:
                return client.post("/api/upload", data={
                    "pdf": (io.BytesIO(payload), "concurrent.pdf"),
                    "upload_token": token,
                }, content_type="multipart/form-data")

        with mock.patch.object(job_module, "start_job", side_effect=start) as call:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(post_once, 1)
                self.assertTrue(entered.wait(timeout=2))
                second = pool.submit(post_once, 2)
                release.set()
                responses = [first.result(timeout=3), second.result(timeout=3)]
        self.assertEqual([r.status_code for r in responses], [200, 200])
        self.assertEqual(len({r.get_json()["slug"] for r in responses}), 1)
        self.assertEqual(sum(bool(r.get_json().get("deduplicated"))
                             for r in responses), 1)
        call.assert_called_once()

    def test_non_pdf_bytes_rejected(self):
        r = self.client.post("/api/upload", data={
            "pdf": (io.BytesIO(b"hello, not a pdf"), "sheet.pdf")},
            content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "not a PDF file")

    def test_missing_file_rejected(self):
        r = self.client.post("/api/upload", data={},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)

    def test_cross_site_header_rejected(self):
        r = self.client.post("/api/upload", headers=XSITE, data={
            "pdf": (io.BytesIO(b"%PDF-1.4 ..."), "sheet.pdf")},
            content_type="multipart/form-data")
        self.assertEqual(r.status_code, 403)

    def test_cross_origin_rejected(self):
        r = self.client.post("/api/upload",
                             headers={"Origin": "http://evil.example"},
                             data={"pdf": (io.BytesIO(b"%PDF-1.4"), "a.pdf")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 403)

    def test_cancel_and_delete_are_write_protected(self):
        self.assertEqual(
            self.client.post(f"/api/cancel/{SLUG}", headers=XSITE).status_code,
            403)
        self.assertEqual(
            self.client.delete(f"/api/project/{SLUG}",
                               headers=XSITE).status_code, 403)

    def test_bad_slug_rejected_on_writes(self):
        self.assertEqual(self.client.post("/api/cancel/..").status_code, 400)
        self.assertEqual(
            self.client.delete("/api/project/a b").status_code, 400)
        self.assertEqual(self.client.get("/api/job/a b").status_code, 400)

    def test_unknown_job_and_cancel_return_404(self):
        with mock.patch.object(job_module, "get_job", return_value=None):
            self.assertEqual(self.client.get("/api/job/missing").status_code,
                             404)
            self.assertEqual(
                self.client.post("/api/cancel/missing").status_code, 404)


class PresetAndOverviewTests(WebappCase):
    def test_preset_target_is_non_empty(self):
        body = self.client.get("/api/preset_target").get_json()
        self.assertIn("target", body)
        self.assertGreater(len(body["target"]), 20)

    def test_overview_empty_dir(self):
        r = self.client.get("/api/overview")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json(), [])

    def test_overview_row_counts(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        self.write_symbols(rec)
        rows = self.client.get("/api/overview").get_json()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["slug"], SLUG)
        self.assertEqual(row["page_count"], 1)
        self.assertEqual(row["mode"], "fence")
        self.assertEqual((row["vlm"], row["added"], row["covered"]), (2, 1, 1))
        self.assertEqual((row["symbols"], row["placements"]), (2, 2))
        page = row["pages"][0]
        self.assertEqual((page["page"], page["vlm"], page["added"],
                          page["sym"], page["plc"], page["present"]),
                         (1, 2, 1, 2, 2, True))

    def test_overview_skips_stale_project(self):
        self.make_pdf()
        self.write_results(self.sample_rec(), fused_v=FUSED_VERSION + 1)
        self.assertEqual(self.client.get("/api/overview").get_json(), [])


class PageTests(WebappCase):
    def test_bad_slug(self):
        r = self.client.get("/api/page/a b/1")
        self.assertEqual(r.status_code, 400)
        self.assertEqual(r.get_json()["error"], "bad slug")

    def test_unknown_project_is_404(self):
        r = self.client.get("/api/page/nope/1")
        self.assertEqual(r.status_code, 404)

    def test_stale_schema_is_409(self):
        self.make_pdf()
        self.write_results(self.sample_rec(), fused_v=FUSED_VERSION + 1)
        r = self.client.get(f"/api/page/{SLUG}/1")
        self.assertEqual(r.status_code, 409)
        body = r.get_json()
        self.assertTrue(body["pending"])
        self.assertEqual(body["stage"], "fused")
        self.assertEqual(body["reason"], "fused_version")

    def test_stale_pdf_revision_is_409(self):
        self.make_pdf()
        self.write_results(self.sample_rec(), pdf_revision="deadbeef-1")
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["reason"], "pdf_revision")

    def test_out_of_range_page_is_404(self):
        self.make_pdf()
        self.write_results(self.sample_rec())
        self.assertEqual(
            self.client.get(f"/api/page/{SLUG}/7").status_code, 404)

    def test_page_without_record_is_synthesized(self):
        self.make_pdf()
        self.write_results(None)
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["items"], [])
        self.assertTrue(body["record"]["empty"])
        self.assertEqual(body["counts"]["text"], 0)

    def test_full_page_payload(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        self.write_symbols(rec)
        r = self.client.get(f"/api/page/{SLUG}/1")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        for key in ("page", "page_count", "mode", "w", "h", "img", "record",
                    "items", "symbols", "dropped_symbols", "plan_boxes",
                    "counts"):
            self.assertIn(key, body)
        self.assertEqual(body["page"], 1)
        self.assertEqual(body["page_count"], 1)
        self.assertEqual(body["mode"], "fence")
        self.assertEqual(body["img"], f"/img/{SLUG}/1")
        self.assertGreater(body["w"], 0)
        self.assertGreater(body["h"], 0)
        # union index 契约：items 与 store.items_of 逐项相同（vlm 在前）
        self.assertEqual(body["items"], store.items_of(rec))
        self.assertEqual(body["items"][2]["text"], "GATE")
        self.assertEqual(body["record"]["codes_stripped"], 2)
        self.assertEqual(len(body["record"]["vec_covered"]), 1)
        # 符号层 + plan 过滤范围
        self.assertEqual(len(body["symbols"]["symbols"]), 2)
        self.assertEqual(body["plan_boxes"], [[300, 100, 900, 900]])
        self.assertEqual(body["symbols"]["groups"][1]["view_type"], "plan")
        self.assertNotIn("view_type", body["symbols"]["groups"][0])
        self.assertEqual(body["counts"], {"text": 3, "symbols": 2,
                                          "placements": 2, "plan_groups": 1,
                                          "marker_codes": 0})
        self.assertEqual(body["marker_codes"], [])
        self.assertIsInstance(body["dropped_symbols"], list)

    def test_stale_symbols_are_not_published(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        self.write_symbols(rec, sig="not-the-current-signature")
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["symbols"]["symbols"], [])
        self.assertEqual(body["symbols"]["groups"], [])
        self.assertEqual(body["plan_boxes"], [])
        self.assertEqual(body["counts"]["symbols"], 0)
        self.assertEqual(body["counts"]["text"], 3)
        # 一个框都不发，但必须明说「这是陈旧、要重跑」而不是「没找到」
        self.assertTrue(body["symbols_stale"])
        self.assertEqual(body["stale_symbols"], 2)

    def test_a_page_with_no_symbols_at_all_is_not_reported_stale(self):
        """真的没有符号 ≠ 陈旧。两种 0 必须能区分开。"""
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        self.write_symbols(rec, sig="not-the-current-signature")
        path = store.slug_dir(SLUG) / "symbols.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["1"]["result"]["symbols"] = []
        path.write_text(json.dumps(cache), encoding="utf-8")
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertFalse(body["symbols_stale"])
        self.assertEqual(body["stale_symbols"], 0)

    def test_stale_placement_version_is_not_published(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        self.write_symbols(rec)
        path = store.slug_dir(SLUG) / "symbols.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        cache["1"]["result"]["plc_v"] = PLACEMENT_VERSION + 1
        store.save_json(path, cache)
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["symbols"]["symbols"], [])

    def test_custom_target_skips_symbol_layer(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec, mode="custom")
        self.write_symbols(rec)
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["mode"], "custom")
        self.assertEqual(body["symbols"]["symbols"], [])
        self.assertEqual(body["counts"]["text"], 3)

    def test_stale_view_types_fail_closed(self):
        """分类不当期：符号照发、分区框照发，但一个 plan 框都不给。"""
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        entry = self.write_symbols(rec)
        revision = store.pdf_revision(store.pdf_path(SLUG))
        self.write_view_types(entry["result"]["groups"], revision,
                              v=VIEW_VERSION + 1)
        body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(body["plan_boxes"], [])
        self.assertTrue(body["symbols"]["view_types_pending"])
        self.assertEqual(len(body["symbols"]["symbols"]), 2)
        self.assertNotIn("view_type", body["symbols"]["groups"][1])
        self.assertEqual(body["counts"]["plan_groups"], 0)

    def test_processing_project_returns_bare_page(self):
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        original = job_module.job_running
        job_module.job_running = lambda slug: slug == SLUG
        try:
            body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        finally:
            job_module.job_running = original
        self.assertTrue(body["processing"])
        self.assertEqual(body["items"], [])
        self.assertEqual(body["record"]["vlm_items"], [])
        self.assertEqual(body["page_count"], 1)

    def test_job_starting_during_render_does_not_publish_old_results(self):
        """A reset=false rerun may retain old results while render is in flight."""
        self.make_pdf()
        rec = self.sample_rec()
        self.write_results(rec)
        running = mock.Mock(side_effect=[False, True])
        with mock.patch.object(job_module, "job_running", running):
            body = self.client.get(f"/api/page/{SLUG}/1").get_json()
        self.assertEqual(running.call_count, 2)
        self.assertTrue(body["processing"])
        self.assertEqual(body["items"], [])
        self.assertEqual(body["record"]["vlm_items"], [])


class AllLineTypesRouteTests(WebappCase):
    @contextmanager
    def route_context(self, *, all_entry=None, generated=None,
                      generation_error=None):
        """Reach the route with current prerequisites and no real sidecar."""
        sig = "lt-current"
        main = {
            "sig": sig, "v": 5, "bindings": [],
            "page": {
                "page_fingerprint": "page",
                "owned_ops_sha1": "owned",
                "fused_ops_sha1": "fused",
                "path_ops": 1,
                "owned_path_ops": 1,
            },
            "all_line_types": [{
                "line_type_number": 1,
                "signature_family": "motif_periodic",
                "recognition_source": "method1",
                "op_count": 1, "ops_sha1": "one", "segment_count": 1,
            }],
        }
        if generated is None:
            generated = {
                "sig": sig, "v": 5,
                "all_v": webapp.linetypes.ALL_GEOMETRY_VERSION,
                "producer_sha256": (
                    webapp.linetypes.sidecar.all_geometry_digest()),
                "page": dict(main["page"]),
                "engine": {"engine": "unit"},
                "types": [{
                    "line_type_number": 1,
                    "signature_family": "motif_periodic",
                    "recognition_source": "method1",
                    "op_count": 1, "ops_sha1": "one",
                    "segment_count": 1,
                    "by_run": [{
                        "run_id": "1", "op_count": 1,
                        "segment_count": 1, "bbox": [1, 2, 3, 4],
                        "polylines": [[[1, 2], [3, 4]]],
                    }],
                }],
                "residual": {"op_count": 0, "segment_count": 0,
                             "polylines": []},
            }
        results = {"pages": {"1": {}}, "pdf_revision": "pdf-current"}
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(webapp.linetypes, "ENABLED", True))
            stack.enter_context(mock.patch.object(
                webapp, "_results_state", return_value=(results, None)))
            stack.enter_context(mock.patch.object(
                webapp, "_placement_anchors_for", return_value=[]))
            stack.enter_context(mock.patch.object(
                webapp.arrows, "arrows_signature", return_value="arrow-current"))
            stack.enter_context(mock.patch.object(
                webapp.arrows, "has_current_arrows", return_value=True))
            stack.enter_context(mock.patch.object(
                webapp.store, "load_json", return_value={"1": {}}))
            stack.enter_context(mock.patch.object(
                webapp.linetypes, "linetypes_signature", return_value=sig))
            stack.enter_context(mock.patch.object(
                webapp.linetypes, "load_page", return_value=main))
            stack.enter_context(mock.patch.object(
                webapp.linetypes, "load_all_page", return_value=all_entry))
            materialize = stack.enter_context(mock.patch.object(
                job_module, "materialize_all_linetypes",
                return_value=generated, side_effect=generation_error))
            yield materialize, generated

    def test_get_missing_is_cache_only_and_never_computes(self):
        with self.route_context(all_entry=None) as (materialize, _generated):
            response = self.client.get(
                f"/api/linetypes_all/{SLUG}/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"state": "not-run"})
        materialize.assert_not_called()

    def test_post_missing_materializes_and_returns_full_geometry(self):
        with self.route_context(all_entry=None) as (materialize, generated):
            response = self.client.post(
                f"/api/linetypes_all/{SLUG}/1")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body["state"], "ok")
        self.assertEqual(len(body["types"]), 1)
        self.assertEqual(body["types"][0]["polylines"],
                         [[[1, 2], [3, 4]]])
        materialize.assert_called_once_with(SLUG, 1, generated["sig"])

    def test_invalid_existing_cache_is_not_published_and_post_rebuilds(self):
        with self.route_context() as (_materialize, valid):
            invalid = dict(valid)
            invalid["page"] = dict(valid["page"], owned_ops_sha1="wrong")

        with self.route_context(all_entry=invalid) as (materialize, _):
            response = self.client.get(f"/api/linetypes_all/{SLUG}/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"state": "stale"})
        materialize.assert_not_called()

        with self.route_context(all_entry=invalid) as (materialize, generated):
            response = self.client.post(f"/api/linetypes_all/{SLUG}/1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["state"], "ok")
        materialize.assert_called_once_with(SLUG, 1, generated["sig"])

    def test_cross_site_post_is_rejected_before_generation(self):
        with mock.patch.object(
                job_module, "materialize_all_linetypes") as materialize:
            for headers in (XSITE, {"Origin": "http://evil.example"}):
                with self.subTest(headers=headers):
                    response = self.client.post(
                        f"/api/linetypes_all/{SLUG}/1", headers=headers)
                    self.assertEqual(response.status_code, 403)
                    self.assertEqual(response.get_json()["error"],
                                     "cross-site write rejected")
        materialize.assert_not_called()

    def test_generation_error_does_not_leak_sidecar_or_file_details(self):
        secret = "/srv/private/final_plans/input.pdf: API_TOKEN=secret"
        with self.route_context(
                all_entry=None,
                generation_error=RuntimeError(secret)) as (materialize, _), \
                mock.patch("builtins.print"):
            response = self.client.post(
                f"/api/linetypes_all/{SLUG}/1")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json(), {
            "state": "error",
            "error": "full line-type geometry generation failed",
        })
        self.assertNotIn(secret, response.get_data(as_text=True))
        self.assertNotIn("RuntimeError", response.get_data(as_text=True))
        materialize.assert_called_once()


class LineTypeRefreshStatusTests(WebappCase):
    def _attach(self, entry, refresh="running", text="PROPOSED FENCE"):
        record = {}
        items = [{"text": text,
                  "box_2d": [100, 100, 120, 180]}]
        with mock.patch.object(webapp.linetypes, "ENABLED", True), \
                mock.patch.object(webapp.arrows, "arrows_signature",
                               return_value="arrow-current"), \
                mock.patch.object(webapp.arrows, "has_current_arrows",
                                  return_value=True), \
                mock.patch.object(webapp.linetypes, "anchors_of",
                                  return_value=[{"key": "0", "ti": 0,
                                                 "tip": [200, 300]}]), \
                mock.patch.object(webapp.linetypes, "linetypes_signature",
                                  return_value="lt-current"), \
                mock.patch.object(webapp.linetypes, "load_page",
                                  return_value=entry), \
                mock.patch.object(
                    webapp.linetype_refresh_state, "page_refresh_status",
                    return_value=refresh):
            webapp._attach_linetypes(
                record, SLUG, 1, items, "pdf-current",
                plan_regions=[[0, 0, 1000, 1000]])
        return record["linetypes_status"]

    def test_stale_cache_is_reported_as_automatic_update(self):
        status = self._attach({"sig": "lt-old", "bindings": []})
        self.assertEqual(status["state"], "updating")
        self.assertEqual(status["refresh"], "running")
        self.assertEqual(status["targets"], 1)

    def test_current_failure_is_not_mislabeled_as_old_cache(self):
        status = self._attach({"sig": "lt-current", "error": "timeout"})
        self.assertEqual(status["state"], "failed")
        self.assertIn("timeout", status["detail"])

    def test_timeout_from_smaller_budget_keeps_polling_for_retry(self):
        entry = {"sig": "lt-current", "error": (
            "RuntimeError: linetype sidecar timeout after 600s (sheet 24)")}
        with mock.patch.object(
                job_module, "_linetype_failure_budget_increased",
                return_value=True):
            status = self._attach(entry, refresh=None)
        self.assertEqual(status["state"], "updating")
        self.assertEqual(status["refresh"], "queued")
        self.assertIn("automatic", status["detail"].lower())

    def test_current_incomplete_cache_is_reported_as_automatic_update(self):
        status = self._attach({"sig": "lt-current", "page": {}})
        self.assertEqual(status["state"], "updating")
        self.assertEqual(status["refresh"], "running")
        self.assertEqual(status["targets"], 1)

    @staticmethod
    def _owned_entry():
        return {
            "sig": "lt-current",
            "bindings": [{
                "key": "0", "ti": 0, "tip": [200, 300], "own_ops": 0,
                "nearest_op": {"op_index": 8, "distance": 0.1, "owner": 7},
                "ranked": [{"line_type_number": 7, "distance": 0.1}],
            }],
            "line_types": [{
                "line_type_number": 7,
                "signature_family": "compound_path_periodic",
                "polylines": [[[200, 300], [210, 320]]],
            }],
            "used_all": [7],
            "page": {"line_types": 1, "residual_ops": 0,
                     "seconds_cluster": 1.0},
        }

    def test_fence_and_gate_text_is_not_mislabeled_all_gate(self):
        status = self._attach(
            self._owned_entry(), text="5' ORNAMENTAL STEEL FENCE & GATE")
        self.assertEqual(status["state"], "ok")
        self.assertEqual(status["visible"], [7])

    def test_pure_gate_text_remains_all_gate(self):
        status = self._attach(self._owned_entry(), text="DOUBLE SWING GATE")
        self.assertEqual(status["state"], "all-gate")


class ImageTests(WebappCase):
    def test_base_image_is_cached_jpeg(self):
        self.make_pdf()
        self.write_results(self.sample_rec())
        r = self.client.get(f"/img/{SLUG}/1")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.mimetype, "image/jpeg")
        self.assertEqual(r.headers["Cache-Control"], "public, max-age=86400")
        revision = store.pdf_revision(store.pdf_path(SLUG))
        cached = store.slug_dir(SLUG) / f"base_P1_{revision}.jpg"
        self.assertTrue(cached.is_file())
        r.close()      # send_file 的文件句柄：Windows 上不关就删不掉 tmp 目录

    def test_bad_slug_image(self):
        self.assertEqual(self.client.get("/img/a b/1").status_code, 400)


class DeleteTests(WebappCase):
    def test_delete_removes_pdf_results_and_job_file(self):
        self.make_pdf()
        self.write_results(self.sample_rec())
        marker = store.JOBS_DIR / f"{SLUG}.json"
        store.save_json(marker, {"slug": SLUG, "done": True, "ok": True})
        project = store.PROJECTS_DIR / SLUG
        data = store.DATA_DIR / SLUG
        self.assertTrue(project.is_dir() and data.is_dir() and marker.is_file())
        r = self.client.delete(f"/api/project/{SLUG}")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.get_json()["ok"])
        self.assertFalse(project.exists())
        self.assertFalse(data.exists())
        self.assertFalse(marker.exists())
        self.assertEqual(self.client.get("/api/overview").get_json(), [])


REF_DIR = Path(os.environ.get("FENCE_REF_DIR",
                              r"C:\Users\Administrator\fence_takeoff_web"))
REF_SLUG = "gladstone_dog_park"
REF_PDF = REF_DIR / "projects" / REF_SLUG / "input.pdf"
REF_RESULTS = REF_DIR / "fence_fused" / REF_SLUG / "results.json"


@unittest.skipUnless(REF_PDF.is_file() and REF_RESULTS.is_file(),
                     f"reference project not available under {REF_DIR}")
class ReferenceDataTests(WebappCase):
    """拿 5051 生产项目的真实 PDF + results.json 跑一次装配（仍然零模型调用）。"""

    def test_real_sheet_payload(self):
        target = self.dirs["PROJECTS_DIR"] / REF_SLUG
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REF_PDF, target / "input.pdf")
        res = json.loads(REF_RESULTS.read_text(encoding="utf-8"))
        # 跨目录拷贝会改 mtime_ns → pdf_revision 变；按本地文件重算，
        # 否则发布闸会（正确地）判它陈旧。见 tools/import_project.py 的同一个坑。
        res["pdf_revision"] = store.pdf_revision(store.pdf_path(REF_SLUG))
        store.save_json(store.results_path(REF_SLUG), res)
        rec = res["pages"]["3"]
        body = self.client.get(f"/api/page/{REF_SLUG}/3").get_json()
        self.assertEqual(body["page_count"], res["page_count"])
        self.assertEqual(body["items"], store.items_of(rec))
        self.assertGreater(body["counts"]["text"], 0)
        self.assertGreater(min(body["w"], body["h"]), 1000)
        # 5051 的 symbols.json 用 prop_v 而不是 plc_v：发布闸必须拒收
        symbols = REF_DIR / "fence_fused" / REF_SLUG / "symbols.json"
        if symbols.is_file():
            shutil.copyfile(symbols, store.slug_dir(REF_SLUG) / "symbols.json")
            again = self.client.get(f"/api/page/{REF_SLUG}/3").get_json()
            self.assertEqual(again["symbols"], {"groups": [], "symbols": []})


class IndexTests(WebappCase):
    def test_index_renders_single_page_ui(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)
        self.assertIn("no-store", response.headers["Cache-Control"])
        self.assertIn("cache:'no-store'", html)
        self.assertIn("fence_lite", html)
        # "Only sheets with results" 曾是这里的标记，随统计框/状态栏一起删掉了。
        # 换成仍然在页面上的同一块 UI（画廊的命中页计数）。
        for marker in ("#2563eb", "#7c3aed", "#16a34a", "#64748b",
                       "sheets with results", "/api/page/"):
            self.assertIn(marker, html)

    def test_favicon(self):
        self.assertEqual(self.client.get("/favicon.ico").status_code, 204)


if __name__ == "__main__":
    if STUBBED:
        print(f"[stubbed for offline test] {', '.join(STUBBED)}", flush=True)
    unittest.main(verbosity=2)
