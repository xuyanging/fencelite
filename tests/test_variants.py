"""对比运行（variant）的磁盘语义与级联删除 —— 全程零模型调用.

界面上一个 PDF 只有一行，对比运行折叠进那一行的模型 chip 里。于是有两条
必须钉住的约束：

  1. **派生不动原项目**：variant 是兄弟目录 data/<slug>__<model>/，原项目的
     缓存与结果一个字节都不许碰 —— 否则"保留原结果做对比"这个前提就没了。
  2. **删除必须级联**：既然 variant 在界面上没有自己的行，删掉原项目时必须
     连它一起删；漏了就变成看不见、又占着盘、还被 /api/overview 列出来的孤儿。

job.start_job 换成不启线程的记录器，绝不触发真实管线。
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


class VariantTests(unittest.TestCase):
    SLUG = "demo_project"
    MODEL = "claude-sonnet-5"

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
        for name in ("results.json", "symbols.json", "vlm.json"):
            (self.slug_data / name).write_text(
                json.dumps({"marker": "原项目结果", "pages": {}}),
                encoding="utf-8")

        self.started = []
        self._real_start = job.start_job
        job.start_job = lambda slug, target=None, model=None: (
            self.started.append((slug, target, model))
            or {"slug": slug, "stage": "queued", "model": model})
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

    @property
    def vslug(self):
        return f"{self.SLUG}{job.VARIANT_SEP}{self.MODEL}"

    # ── create_variant ────────────────────────────────────────────────────
    def test_creates_sibling_with_the_same_pdf(self):
        slug = job.create_variant(self.SLUG, self.MODEL)
        self.assertEqual(slug, self.vslug)
        self.assertTrue((self.projects / slug / "input.pdf").exists())
        self.assertEqual((self.projects / slug / "input.pdf").read_bytes(),
                         self.pdf.read_bytes())

    def test_base_project_is_untouched(self):
        before = {p.name: p.read_bytes()
                  for p in self.slug_data.iterdir() if p.is_file()}
        pdf_before = self.pdf.read_bytes()
        job.create_variant(self.SLUG, self.MODEL)
        after = {p.name: p.read_bytes()
                 for p in self.slug_data.iterdir() if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(self.pdf.read_bytes(), pdf_before)

    def test_variant_gets_its_own_empty_cache_dir(self):
        # Sharing a cache dir would let one provider's cached raw satisfy the
        # other's currency check, silently mixing the two in one comparison.
        slug = job.create_variant(self.SLUG, self.MODEL)
        store.slug_dir(slug)          # created on demand
        self.assertNotEqual(store.slug_dir(slug), self.slug_data)
        self.assertEqual(list((self.data / slug).glob("*.json")), [])

    def test_is_idempotent(self):
        a = job.create_variant(self.SLUG, self.MODEL)
        b = job.create_variant(self.SLUG, self.MODEL)
        self.assertEqual(a, b)

    def test_rejects_unknown_model(self):
        with self.assertRaises(ValueError):
            job.create_variant(self.SLUG, "gpt-4")

    def test_rejects_forking_a_variant(self):
        v = job.create_variant(self.SLUG, self.MODEL)
        with self.assertRaises(ValueError):
            job.create_variant(v, "claude-opus-5")

    def test_rejects_missing_project(self):
        with self.assertRaises(FileNotFoundError):
            job.create_variant("no_such_project", self.MODEL)

    # ── variants_of ───────────────────────────────────────────────────────
    def test_variants_of_finds_the_fork(self):
        job.create_variant(self.SLUG, self.MODEL)
        self.assertEqual(job.variants_of(self.SLUG), [self.vslug])

    def test_variants_of_is_empty_for_a_plain_project(self):
        self.assertEqual(job.variants_of(self.SLUG), [])

    def test_variants_of_a_variant_is_empty(self):
        v = job.create_variant(self.SLUG, self.MODEL)
        self.assertEqual(job.variants_of(v), [])

    def test_variants_of_does_not_match_a_name_prefix(self):
        # "demo_project_2" must not be mistaken for a variant of "demo_project"
        (self.data / f"{self.SLUG}_2").mkdir()
        self.assertEqual(job.variants_of(self.SLUG), [])

    # ── cascade delete ────────────────────────────────────────────────────
    def test_delete_cascades_to_variants(self):
        v = job.create_variant(self.SLUG, self.MODEL)
        store.slug_dir(v)
        self.assertTrue(job.delete_project(self.SLUG))
        for root in (self.projects, self.data):
            self.assertFalse((root / self.SLUG).exists())
            self.assertFalse((root / v).exists(), f"orphan left in {root.name}")

    def test_delete_cascade_can_be_disabled(self):
        v = job.create_variant(self.SLUG, self.MODEL)
        job.delete_project(self.SLUG, cascade=False)
        self.assertTrue((self.projects / v).exists())

    def test_deleting_a_variant_leaves_the_base(self):
        v = job.create_variant(self.SLUG, self.MODEL)
        job.delete_project(v)
        self.assertTrue((self.projects / self.SLUG / "input.pdf").exists())
        self.assertTrue((self.data / self.SLUG / "results.json").exists())

    # ── /api/variant ──────────────────────────────────────────────────────
    def test_endpoint_starts_the_run_with_the_model_pinned(self):
        r = self.client.post(f"/api/variant/{self.SLUG}",
                             json={"model": self.MODEL})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        body = r.get_json()
        self.assertEqual(body["slug"], self.vslug)
        self.assertEqual(body["model"], self.MODEL)
        self.assertEqual(body["base"], self.SLUG)
        # The variant inherits the base project's detection target, so the two
        # runs differ only by model — that is the whole point of the comparison.
        self.assertEqual(len(self.started), 1)
        slug, target, model = self.started[0]
        self.assertEqual(slug, self.vslug)
        self.assertEqual(model, self.MODEL)
        self.assertEqual(target, job.stored_target(self.SLUG))

    def test_endpoint_rejects_unknown_model(self):
        r = self.client.post(f"/api/variant/{self.SLUG}", json={"model": "gpt-4"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.started, [])

    def test_endpoint_rejects_missing_project(self):
        r = self.client.post("/api/variant/nope", json={"model": self.MODEL})
        self.assertEqual(r.status_code, 404)

    # ── /api/rerun 必须把变体的模型钉回去 ────────────────────────────────
    def test_rerun_of_a_variant_keeps_its_model(self):
        # 不钉的话重跑会用进程默认模型：symbols / vlm / 判词都按
        # resolve_model(None) 校验，整份缓存读作过期 -> 用默认模型重新付费
        # 并覆盖掉这份对比结果。
        v = job.create_variant(self.SLUG, self.MODEL)
        (self.data / v).mkdir(exist_ok=True)
        r = self.client.post(f"/api/rerun/{v}", json={"reset": False})
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        self.assertEqual(r.get_json()["model"], self.MODEL)
        self.assertEqual(self.started[-1][0], v)
        self.assertEqual(self.started[-1][2], self.MODEL)

    def test_rerun_of_a_plain_project_pins_nothing(self):
        r = self.client.post(f"/api/rerun/{self.SLUG}", json={"reset": False})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.get_json()["model"])
        self.assertIsNone(self.started[-1][2])

    def test_rerun_rejects_an_unknown_explicit_model(self):
        r = self.client.post(f"/api/rerun/{self.SLUG}",
                             json={"reset": False, "model": "gpt-4"})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.started, [])

    def test_endpoint_404s_on_a_bad_slug(self):
        r = self.client.post("/api/variant/..%2Fetc", json={"model": self.MODEL})
        self.assertIn(r.status_code, (400, 404))
        self.assertEqual(self.started, [])


if __name__ == "__main__":
    unittest.main()
