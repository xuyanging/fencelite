"""tools/import_project.py 的离线回归 —— 造一个假的参考项目再搬进临时目录.

完全离线：一页 PDF 由 PyMuPDF 现场生成，缓存 JSON 手写最小体，零 Gemini 调用。
DATA_DIR / PROJECTS_DIR 都打到临时目录，绝不碰真的 data/ 与 projects/。
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")

import fitz

from steps.store import items_of, sig_of


def _load_tool():
    """tools/ 没有 __init__.py，按文件路径加载。"""
    path = BASE_DIR / "tools" / "import_project.py"
    spec = importlib.util.spec_from_file_location("_import_project_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOL = _load_tool()

SLUG = "fake_site"
OLD_REV = "deadbe-ef00ef00"


def _write_pdf(path):
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "6' CHAIN LINK FENCE", fontsize=11)
    doc.save(str(path))
    doc.close()


def _fake_ref(root):
    """<root>/projects/<slug>/input.pdf + <root>/fence_fused/<slug>/*.json"""
    pdir = root / "projects" / SLUG
    fdir = root / "fence_fused" / SLUG
    pdir.mkdir(parents=True)
    fdir.mkdir(parents=True)
    _write_pdf(pdir / "input.pdf")

    rec = {
        "vlm_items": [{"text": "6' CHAIN LINK FENCE", "box_2d": [100, 80, 112, 300],
                       "label": "legend entry", "source": "vlm", "tbl": True}],
        "vec_added": [{"text": "SILT FENCE", "box_2d": [400, 80, 410, 200],
                       "label": "vector supplement", "source": "vector"}],
        "vec_covered": [],
        "has_text": True,
        "vlm_error": None,
        "vlm_from": "extra",
        "vlm_sources": [{"store": "extra", "role": "primary",
                         "identity": {"pdf_revision": OLD_REV,
                                      "model": "gemini-3.1-pro-preview",
                                      "prompt_sha256": "a" * 64}}],
    }
    results = {"slug": "SOME_OTHER_NAME", "fused_v": 2, "pdf_revision": OLD_REV,
               "page_count": 1, "no_text_layer": False, "judge_error": None,
               "pages": {"1": rec}}
    (fdir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    (fdir / "vlm_extra.json").write_text(json.dumps({
        "1": {"items": rec["vlm_items"], "elapsed": 3.2, "model": "gemini-3.1-pro-preview",
              "usage": {"input_tokens": 10, "output_tokens": 5},
              "vlm_identity": {"pdf_revision": OLD_REV,
                               "model": "gemini-3.1-pro-preview",
                               "prompt_sha256": "a" * 64},
              "vlm_role": "primary"}}), encoding="utf-8")

    (fdir / "vlm_flash.json").write_text(json.dumps({
        "1": {"items": [], "model": "gemini-3.5-flash",
              "vlm_identity": {"pdf_revision": OLD_REV,
                               "model": "gemini-3.5-flash",
                               "prompt_sha256": "a" * 64}}}), encoding="utf-8")

    (fdir / "vec.json").write_text(json.dumps({
        "schema": 3, "pdf_mtime": 1.0, "page_count": 1,
        "pages": {"1": {"lines": [{"box_2d": [100, 80, 112, 300],
                                   "text": "6' CHAIN LINK FENCE"}],
                        "has_text": True}}}), encoding="utf-8")

    (fdir / "textjudge.json").write_text(json.dumps({
        "v": 1, "model": "gemini-3.1-pro-preview",
        "verdicts": {"6' chain link fence": True}}), encoding="utf-8")

    groups = [{"box_2d": [90, 70, 200, 400], "kind": "legend"},
              {"box_2d": [300, 0, 900, 1000], "kind": "view",
               "view_type": "plan"}]
    symbol = {"box_2d": [100, 40, 112, 70], "category": "shape", "value": "SF",
              "type": "shape SF", "text_index": 0, "group_index": 0,
              # 旧 propagate 阶段的字段，导入时必须被剥掉
              "placements": [[500, 500, 510, 510], [600, 600, 610, 610]],
              "trace": {"segments": [1, 2, 3]}, "line_type": "dashed",
              "sample_evidence": {"n": 1}, "vec_error": None}
    (fdir / "symbols.json").write_text(json.dumps({
        "1": {"sig": "STALESIG0000", "v": 18, "pv": 16, "pure": True,
              "model": "gemini-3.1-pro-preview",
              "raw": {"groups": groups, "symbols": [dict(symbol)]},
              "result": {"symbols": [dict(symbol)], "groups": groups,
                         "pure": True, "prop_v": 13},
              "debug": {"matching": [{"symbol_index": 0}]}}}),
        encoding="utf-8")

    (fdir / "view_types.json").write_text(json.dumps({
        "1": {"sig": "STALEVIEW000", "v": 1, "model": "gemini-3.1-pro-preview",
              "views": [{"group_index": 1, "view_type": "plan",
                         "reason": "top-down site"}]}}), encoding="utf-8")

    # 明确不该被搬走的两个已砍文件
    (fdir / "callout_selections.json").write_text("{}", encoding="utf-8")
    (fdir / "fencelines.json").write_text("{}", encoding="utf-8")
    (fdir / f"base_P1_{OLD_REV}.jpg").write_bytes(b"\xff\xd8\xff")
    return root


class ImportProjectTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.ref = _fake_ref(root / "ref")
        self.data = root / "data"
        self.projects = root / "projects"
        self.data.mkdir()
        self.projects.mkdir()
        patches = [mock.patch.object(TOOL, "DATA_DIR", self.data),
                   mock.patch.object(TOOL, "PROJECTS_DIR", self.projects)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, **kw):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rep = TOOL.import_project(SLUG, ref_dir=self.ref, **kw)
        return rep, buf.getvalue()

    # ---- 文件名映射与不该搬的文件 -------------------------------------
    def test_file_mapping(self):
        rep, _ = self._run()
        names = sorted(p.name for p in (self.data / SLUG).iterdir())
        self.assertEqual(names, ["results.json", "symbols.json",
                                 "textjudge.json", "vec.json",
                                 "view_types.json", "vlm.json",
                                 "vlm_flash.json"])
        self.assertFalse((self.data / SLUG / "vlm_extra.json").exists())
        self.assertFalse((self.data / SLUG / "callout_selections.json").exists())
        self.assertFalse((self.data / SLUG / "fencelines.json").exists())
        self.assertEqual(list((self.data / SLUG).glob("base_P*.jpg")), [])
        self.assertTrue((self.projects / SLUG / "input.pdf").exists())
        self.assertEqual(rep["old_revision"], OLD_REV)

    # ---- revision 重算 ------------------------------------------------
    def test_revision_recomputed(self):
        rep, _ = self._run()
        new_rev = rep["pdf_revision"]
        self.assertNotEqual(new_rev, OLD_REV)
        pdf = self.projects / SLUG / "input.pdf"
        stat = pdf.stat()
        self.assertEqual(new_rev, f"{stat.st_size:x}-{stat.st_mtime_ns:x}")

        results = json.loads((self.data / SLUG / "results.json").read_text("utf-8"))
        self.assertEqual(results["pdf_revision"], new_rev)
        self.assertEqual(results["slug"], SLUG)  # 顶层 slug 跟随新项目名
        self.assertEqual(
            results["pages"]["1"]["vlm_sources"][0]["identity"]["pdf_revision"],
            new_rev)

        for name in ("vlm.json", "vlm_flash.json"):
            data = json.loads((self.data / SLUG / name).read_text("utf-8"))
            self.assertEqual(data["1"]["vlm_identity"]["pdf_revision"], new_rev)

        vec = json.loads((self.data / SLUG / "vec.json").read_text("utf-8"))
        self.assertEqual(vec["pdf_mtime"], pdf.stat().st_mtime)
        self.assertEqual(vec["schema"], 3)

    # ---- sig 重算 -----------------------------------------------------
    def test_symbol_sig_recomputed(self):
        rep, _ = self._run()
        new_rev = rep["pdf_revision"]
        results = json.loads((self.data / SLUG / "results.json").read_text("utf-8"))
        symbols = json.loads((self.data / SLUG / "symbols.json").read_text("utf-8"))
        expect = sig_of(items_of(results["pages"]["1"]), new_rev)
        self.assertEqual(symbols["1"]["sig"], expect)
        self.assertNotEqual(symbols["1"]["sig"], "STALESIG0000")
        self.assertIs(rep["symbols_sig_ok"], True)
        self.assertIs(rep["results_revision_ok"], True)
        self.assertIs(rep["vlm_revision_ok"], True)
        self.assertIs(rep["vec_mtime_ok"], True)

    def test_view_sig_recomputed_or_skipped(self):
        rep, out = self._run()
        views = json.loads((self.data / SLUG / "view_types.json").read_text("utf-8"))
        if TOOL._view_signature is None:
            # steps.views 未就绪：明确跳过并提示，不许猜公式
            self.assertEqual(views["1"]["sig"], "STALEVIEW000")
            self.assertIsNone(rep["view_sig_ok"])
            self.assertIn("steps.views 尚未就绪", out)
        else:
            symbols = json.loads(
                (self.data / SLUG / "symbols.json").read_text("utf-8"))
            groups = symbols["1"]["result"]["groups"]
            self.assertEqual(
                views["1"]["sig"],
                TOOL._view_signature(groups, rep["pdf_revision"],
                                     views["1"]["model"]))
            self.assertIs(rep["view_sig_ok"], True)

    # ---- 旧语义 placements 必须剥离 -----------------------------------
    def test_stale_placements_stripped(self):
        rep, out = self._run()
        symbols = json.loads((self.data / SLUG / "symbols.json").read_text("utf-8"))
        entry = symbols["1"]
        sym = entry["result"]["symbols"][0]
        for key in TOOL.STALE_SYMBOL_KEYS:
            self.assertNotIn(key, sym, key)
        self.assertNotIn("prop_v", entry["result"])
        self.assertNotIn("debug", entry)
        # 付费的 raw 一个字都不许动
        self.assertIn("placements", entry["raw"]["symbols"][0])
        self.assertEqual(sym["text_index"], 0)
        self.assertEqual(rep["symbols"], 1)
        self.assertEqual(rep["text_items"], 2)
        self.assertEqual(rep["pages_with_text"], 1)
        if rep["placements_recomputed"]:
            # 步骤4 就地重算过：plc_v 落盘，placements 是新语义（本页假 PDF
            # 里没有可匹配的矢量形状，所以 0 是正确答案）
            self.assertIn("plc_v", entry["result"])
            self.assertEqual(rep["placements"], 0)
        else:
            self.assertNotIn("plc_v", entry["result"])
            self.assertEqual(rep["placements"], 0)
            self.assertIn("steps.placements 尚不可用", out)

    # ---- dry-run 不落地 ------------------------------------------------
    def test_dry_run_writes_nothing(self):
        rep, out = self._run(dry_run=True)
        self.assertTrue(rep["dry_run"])
        self.assertFalse((self.data / SLUG).exists())
        self.assertFalse((self.projects / SLUG).exists())
        self.assertIn("DRY RUN", out)
        self.assertIn("vlm.json", out)
        self.assertNotIn("pdf_revision", rep)

    # ---- 重复导入被拒 --------------------------------------------------
    def test_refuses_existing(self):
        self._run()
        with self.assertRaises(ValueError) as ctx:
            self._run()
        self.assertIn("already exists", str(ctx.exception))

    def test_missing_slug_raises(self):
        with self.assertRaises(FileNotFoundError):
            TOOL.import_project("no_such_slug", ref_dir=self.ref)

    def test_bad_slug_rejected(self):
        with self.assertRaises(ValueError):
            TOOL.import_project("../evil", ref_dir=self.ref)

    # ---- CLI ----------------------------------------------------------
    def test_available_slugs(self):
        self.assertEqual(TOOL.available_slugs(self.ref), [SLUG])
        self.assertEqual(TOOL.available_slugs(self.ref / "nope"), [])

    def test_main_dry_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = TOOL.main([SLUG, "--from", str(self.ref), "--dry-run"])
        self.assertEqual(rc, 0)
        self.assertFalse((self.data / SLUG).exists())

    def test_main_no_slugs_lists(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = TOOL.main(["--from", str(self.ref)])
        self.assertEqual(rc, 2)
        self.assertIn(SLUG, buf.getvalue())

    def test_main_reports_failure(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = TOOL.main(["../evil", "--from", str(self.ref)])
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
