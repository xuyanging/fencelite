"""重新跑一个已有 PDF 的回归 —— 全程零模型调用.

这条路径的产品承诺是「结果不受任何旧缓存影响」，所以测的重点只有两件事：
  1. reset 真的把 data/<slug>/ 下的东西全清了，且**没碰** projects/<slug>/input.pdf；
  2. 路由把 reset 与 target 正确传下去，跑着的项目不给重跑。
job.start_job 被替换成不启线程的记录器，绝不触发真实管线。
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

import fitz

import job
import webapp
from steps import store


def _blank_pdf(path):
    with fitz.open() as doc:
        doc.new_page()
        doc.save(str(path))


class RerunTests(unittest.TestCase):
    SLUG = "demo_project"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.projects = root / "projects"
        self.data = root / "data"
        self.jobs = root / "_jobs"
        for d in (self.projects, self.data, self.jobs):
            d.mkdir()
        self._saved = {}
        for mod, name, value in (
                (store, "PROJECTS_DIR", self.projects),
                (store, "DATA_DIR", self.data),
                (store, "JOBS_DIR", self.jobs),
                (job, "PROJECTS_DIR", self.projects),
                (job, "DATA_DIR", self.data),
                (job, "JOBS_DIR", self.jobs)):
            self._saved[(mod, name)] = getattr(mod, name)
            setattr(mod, name, value)

        self.pdf = self.projects / self.SLUG / "input.pdf"
        self.pdf.parent.mkdir()
        _blank_pdf(self.pdf)
        self.slug_data = self.data / self.SLUG
        self.slug_data.mkdir()
        for name in ("results.json", "symbols.json", "vec.json",
                     "textjudge.json", "vlm.json", "view_types.json"):
            (self.slug_data / name).write_text(
                json.dumps({"target": "老目标", "pages": {}}),
                encoding="utf-8")
        (self.slug_data / "base_P1_deadbeef.jpg").write_bytes(b"\xff\xd8jpeg")

        self.started = []
        self._real_start = job.start_job
        # 签名要跟 job.start_job 一致：/api/rerun 现在会把变体的模型钉回去
        # （见 tests/test_variants.py 的 test_rerun_of_a_variant_keeps_its_model）。
        job.start_job = lambda slug, target=None, model=None: (
            self.started.append((slug, target)) or {"slug": slug,
                                                    "stage": "queued"})
        self._real_running = job.job_running
        job.job_running = lambda slug: False
        webapp.app.config["TESTING"] = True
        self.client = webapp.app.test_client()

    def tearDown(self):
        job.start_job = self._real_start
        job.job_running = self._real_running
        for (mod, name), value in self._saved.items():
            setattr(mod, name, value)
        self.tmp.cleanup()

    # ---- reset_project_cache ------------------------------------------------

    def test_reset_wipes_every_cached_artifact(self):
        removed = job.reset_project_cache(self.SLUG)
        self.assertIn("results.json", removed)
        self.assertIn("base_P1_deadbeef.jpg", removed)
        self.assertEqual(sorted(p.name for p in self.slug_data.iterdir()), [])

    def test_reset_keeps_the_source_pdf(self):
        job.reset_project_cache(self.SLUG)
        self.assertTrue(self.pdf.exists(), "重置绝不能动源 PDF")

    def test_reset_refuses_while_running(self):
        job.job_running = lambda slug: True
        with self.assertRaises(RuntimeError):
            job.reset_project_cache(self.SLUG)
        self.assertTrue((self.slug_data / "results.json").exists())

    def test_reset_rejects_bad_slug(self):
        with self.assertRaises(ValueError):
            job.reset_project_cache("../etc")

    # ---- POST /api/rerun ----------------------------------------------------

    def test_rerun_clears_then_starts(self):
        r = self.client.post(f"/api/rerun/{self.SLUG}",
                             json={"target": "找门"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertIn("results.json", body["cleared"])
        self.assertEqual(self.started, [(self.SLUG, "找门")])
        self.assertEqual(list(self.slug_data.iterdir()), [])
        self.assertTrue(self.pdf.exists())

    def test_rerun_without_reset_keeps_cache(self):
        r = self.client.post(f"/api/rerun/{self.SLUG}",
                             json={"reset": False})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["cleared"], [])
        self.assertTrue((self.slug_data / "results.json").exists())

    def test_rerun_defaults_to_the_projects_stored_target(self):
        self.client.post(f"/api/rerun/{self.SLUG}", json={})
        self.assertEqual(self.started, [(self.SLUG, "老目标")])

    def test_rerun_running_project_is_409(self):
        job.job_running = lambda slug: True
        r = self.client.post(f"/api/rerun/{self.SLUG}", json={})
        self.assertEqual(r.status_code, 409)
        self.assertEqual(self.started, [])

    def test_rerun_unknown_project_is_404(self):
        r = self.client.post("/api/rerun/nope", json={})
        self.assertEqual(r.status_code, 404)

    def test_rerun_bad_slug_is_400(self):
        r = self.client.post("/api/rerun/..%2Fetc", json={})
        self.assertIn(r.status_code, (400, 404))

    def test_rerun_is_cross_site_write_protected(self):
        r = self.client.post(f"/api/rerun/{self.SLUG}", json={},
                             headers={"Sec-Fetch-Site": "cross-site"})
        self.assertEqual(r.status_code, 403)
        self.assertEqual(self.started, [])
        self.assertTrue((self.slug_data / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
