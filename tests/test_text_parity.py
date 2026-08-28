"""步骤1 的离线回归 —— 用新代码重算生产结果，逐字段比对.

完全离线：只读参考项目（默认 C:\\Users\\Administrator\\fence_takeoff_web）的
本地 PDF + 已付费缓存 JSON，零 Gemini 调用。

FENCE_REF_DIR 指向参考项目根；目录不存在就整体跳过。

四类断言：
  1. fuse_page parity —— 拿 vec.json / textjudge.json / vlm_extra.json 重算
     每一页，和生产 results.json 里的页记录整字典相等（vlm_items /
     vec_added / vec_covered / codes_stripped / has_text /
     debug.vector_candidates）。gladstone 8页 + lenexa + rapid_city 共 25 页。
     vlm_from == "textscan" 的页跳过：那些 raw 存在已砍掉的 fence_text_scan
     历史库里（本机该目录是空的），无法离线重建；koch 整个项目的缓存还早于
     target 重构，prompt 摘要不同，身份闸按设计全部拒绝。
  2. strip_marker_codes parity —— koch P4（唯一 codes_stripped != 0 的参考页）
     用生产 debug 记录的 31 条被剥项 + 16 条存活项重放，逐条对齐理由。
  3. markers 源码 parity —— steps/text/markers.py 除模块 docstring 与那一行
     import 之外，与参考项目的 steps/legends/geometric.py 逐字节相同。
  4. vlm_needed / debug_view 的纯逻辑单元测试。
"""
import ast
import json
import os
import unittest
from pathlib import Path

from core.parsing import is_normalized_box
from steps.debug import DebugSink
from steps.text import (TARGET_DEFAULT, fuse_page, is_default_target,
                        build_vlm_prompt, union_vlm, vlm_needed)
from steps.text.debug_view import attach_text_debug
from steps.text.vlmcache import (PRIMARY_ROLE, SECONDARY_UNION_ROLE,
                                 is_current_primary_record,
                                 is_current_secondary_record,
                                 vlm_identity_for_revision)

REF = Path(os.environ.get("FENCE_REF_DIR",
                          r"C:\Users\Administrator\fence_takeoff_web"))
HAS_REF = (REF / "fence_fused").is_dir() and (REF / "projects").is_dir()

ITEM_BUCKETS = ("vlm_items", "vec_added", "vec_covered")


def _load(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _has_vecgeom():
    """markers.strip_context 只依赖 core.vecgeom._extract_page（另一模块提供）。"""
    try:
        from core.vecgeom import _extract_page  # noqa: F401
        return True
    except Exception:                                           # noqa: BLE001
        return False


HAS_VECGEOM = _has_vecgeom()


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF}")
class TestFusePageParity(unittest.TestCase):
    """新 fuse_page 必须逐字段重现生产 results.json 的页记录。"""

    def _pages_for(self, slug):
        res = _load(REF / "fence_fused" / slug / "results.json")
        self.assertIsNotNone(res, f"{slug}: no results.json")
        vec = _load(REF / "fence_fused" / slug / "vec.json", {})
        judge = _load(REF / "fence_fused" / slug / "textjudge.json", {})
        extra = _load(REF / "fence_fused" / slug / "vlm_extra.json", {})
        flash = _load(REF / "fence_fused" / slug / "vlm_flash.json", {})
        flagged = {s for s, v in (judge.get("verdicts") or {}).items() if v}
        target = res.get("target") or TARGET_DEFAULT
        vlm_prompt = build_vlm_prompt(target)
        # 用 results.json 里记下的 revision，而不是重新 stat 本机 PDF ——
        # 跨机拷贝会改 mtime，但缓存身份是当时那份文档的。
        revision = res["pdf_revision"]
        primary_identity = vlm_identity_for_revision(revision, None, vlm_prompt)
        flash_identity = vlm_identity_for_revision(
            revision, "gemini-3.5-flash", vlm_prompt)
        return {
            "res": res, "vec": vec, "flagged": flagged, "extra": extra,
            "flash": flash, "target": target,
            "use_kw_floor": is_default_target(res.get("target")),
            "primary_identity": primary_identity,
            "flash_identity": flash_identity,
            "pdf": REF / "projects" / slug / "input.pdf",
        }

    def _recompute(self, ctx, page):
        """复刻生产合并循环的输入：primary raw (+ 扫描页 flash union)。"""
        record = ctx["extra"].get(str(page))
        # **刻意忽略 prompt_sha256**：这几条用例验的是本地融合（判词命中的矢量行
        # → 剥符号码 → 三桶融合）能否从录好的 raw 复现出录好的页记录，与提示词
        # 无关。而 target.py 的提示词已经按产品要求改过（编码标记不再当文字），
        # 与参考集那一版不同 —— 拿身份闸去筛就会一页都不剩，把这层回归废掉。
        if not (isinstance(record, dict) and not record.get("error")
                and isinstance(record.get("items"), list)
                and record.get("vlm_role") == PRIMARY_ROLE
                and (record.get("vlm_identity") or {}).get("pdf_revision")
                == ctx["res"]["pdf_revision"]):
            return None
        vitems = record.get("items", [])
        vpage = (ctx["vec"].get("pages") or {}).get(str(page), {})
        if not vpage.get("has_text"):
            fr = ctx["flash"].get(str(page))
            if is_current_secondary_record(fr, ctx["flash_identity"]):
                vitems = union_vlm(vitems, fr.get("items"))
        dbg = DebugSink()
        return fuse_page(ctx["pdf"], page - 1, vpage, ctx["flagged"], vitems,
                         use_kw_floor=ctx["use_kw_floor"], dbg=dbg)

    def _check_slug(self, slug, expect_compared):
        ctx = self._pages_for(slug)
        self.assertTrue(ctx["pdf"].exists(), f"{slug}: input.pdf missing")
        compared, skipped = 0, []
        for page_str, expect in sorted(ctx["res"]["pages"].items(),
                                       key=lambda kv: int(kv[0])):
            page = int(page_str)
            with self.subTest(slug=slug, page=page):
                if expect.get("vlm_from") == "textscan":
                    skipped.append((page, "textscan store (removed)"))
                    continue
                got = self._recompute(ctx, page)
                if got is None:
                    skipped.append((page, "raw cache identity not current"))
                    continue
                for bucket in ITEM_BUCKETS:
                    self.assertEqual(
                        got[bucket], expect[bucket],
                        f"{slug} P{page} {bucket} differs")
                self.assertEqual(got.get("codes_stripped"),
                                 expect.get("codes_stripped"),
                                 f"{slug} P{page} codes_stripped differs")
                self.assertEqual(got["has_text"], expect["has_text"],
                                 f"{slug} P{page} has_text differs")
                self.assertEqual(
                    (got.get("debug") or {}).get("vector_candidates"),
                    (expect.get("debug") or {}).get("vector_candidates"),
                    f"{slug} P{page} debug.vector_candidates differs")
                compared += 1
        print(f"  [parity] {slug}: {compared} pages field-identical, "
              f"{len(skipped)} skipped {skipped}")
        self.assertEqual(compared, expect_compared,
                         f"{slug}: skipped={skipped}")

    def test_gladstone_every_recorded_page(self):
        self._check_slug("gladstone_dog_park", 5)

    def test_rapid_city_every_recorded_page(self):
        self._check_slug("rapid_city", 11)

    def test_lenexa_every_recorded_page(self):
        self._check_slug("lenexa_fuel_station", 9)

    def test_prompt_has_deliberately_diverged_from_the_reference_set(self):
        """步骤① 的提示词已经按产品要求改过，与参考集那一版不再相同。

        改的内容：编码标记（印在小闭合标记里、或独占图例 SYMBOL 列的短码）是
        符号不是文字 —— 图纸上那些标记不再输出成文字项，图例行也不再把行首的
        码含进 text/box。所以参考集的 raw 是**另一版提示词**的产物，上面几条
        parity 用例才刻意忽略 prompt_sha256。
        这条用例把「已分叉」这件事写死：哪天有人把规则删回去，这里会红。
        """
        prompt = build_vlm_prompt(TARGET_DEFAULT)
        self.assertIn("CODE MARKERS ARE SYMBOLS, NOT TEXT", prompt)
        self.assertIn("exclude the code from", prompt)
        ours = vlm_identity_for_revision("x", None, prompt)["prompt_sha256"]
        for slug in ("gladstone_dog_park", "lenexa_fuel_station", "rapid_city"):
            extra = _load(REF / "fence_fused" / slug / "vlm_extra.json", {})
            digests = {r.get("vlm_identity", {}).get("prompt_sha256")
                       for r in extra.values() if isinstance(r, dict)}
            self.assertTrue(digests, slug)
            self.assertNotIn(ours, digests, slug)

    def test_legacy_prompt_raws_are_never_reused_in_production(self):
        """提示词摘要不同的旧 raw，生产的身份闸必须拒绝复用（会重新付费扫）。

        koch 的缓存整体早于 target 重构（results.json 连 target/mode 都没有），
        是天然的样本。这里验的是**生产口径**：is_current_primary_record 为假。
        上面几条 parity 用例走的是"忽略提示词身份"的回放路径 —— 那是另一件事
        （验本地融合），两者不能混。
        """
        ctx = self._pages_for("koch_tennis_center")
        self.assertNotIn("target", ctx["res"])
        for page in ("16", "21"):
            record = ctx["extra"][page]
            self.assertEqual(record.get("vlm_role"), PRIMARY_ROLE)
            self.assertEqual(record["vlm_identity"]["pdf_revision"],
                             ctx["res"]["pdf_revision"])
            self.assertNotEqual(
                record["vlm_identity"]["prompt_sha256"],
                ctx["primary_identity"]["prompt_sha256"])
            self.assertFalse(is_current_primary_record(
                record, ctx["primary_identity"]))

    def test_fusion_is_prompt_independent(self):
        """koch 那两页 raw 虽然是旧提示词的产物，回放进 fuse_page 仍然逐字段
        复现它当时的页记录 —— 融合逻辑与提示词无关，这正是上面几条 parity
        用例敢忽略 prompt_sha256 的依据。"""
        self._check_slug("koch_tennis_center", 2)


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF}")
class TestMarkersSourceParity(unittest.TestCase):
    """markers.py = 参考项目 legends/geometric.py，只换 import 与 docstring。"""

    def test_body_is_verbatim(self):
        ours = Path("steps/text/markers.py").read_text(encoding="utf-8")
        theirs = (REF / "fence_takeoff" / "steps" / "legends"
                  / "geometric.py").read_text(encoding="utf-8")
        expect = _after_docstring(theirs).replace(
            "from core.vecmatch import _extract_page",
            "from core.vecgeom import _extract_page")
        self.assertEqual(_after_docstring(ours), expect)


def _after_docstring(src):
    """源码去掉模块 docstring（比对函数体是否逐字节一致）。"""
    tree = ast.parse(src)
    body = tree.body
    lines = src.splitlines(True)
    if body and isinstance(body[0], ast.Expr) \
            and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        start = body[1].lineno - 1 if len(body) > 1 else len(lines)
    else:
        start = 0
    return "".join(lines[start:])


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF}")
@unittest.skipUnless(HAS_VECGEOM, "core.vecgeom not available")
class TestStripContextSmoke(unittest.TestCase):
    """strip_context 在真实页上跑出结构正确的 marker 上下文（含非空 mboxes）。"""

    def test_koch_p4(self):
        from steps.text.markers import strip_context
        ctx = strip_context(
            REF / "projects" / "koch_tennis_center" / "input.pdf", 3)
        self.assertEqual(set(ctx), {"mboxes", "segs"})
        self.assertTrue(ctx["mboxes"], "no marker box harvested")
        self.assertTrue(ctx["segs"], "no segment harvested")
        for box in ctx["mboxes"]:
            self.assertTrue(is_normalized_box(box), box)
        for seg in ctx["segs"]:
            self.assertEqual(set(seg), {"ax", "ay", "bx", "by", "dash", "src"})
        print(f"  [markers] koch P4: mboxes={len(ctx['mboxes'])} "
              f"segs={len(ctx['segs'])}")


@unittest.skipUnless(HAS_REF, f"reference data not found at {REF}")
@unittest.skipUnless(HAS_VECGEOM, "core.vecgeom not available")
class TestStripMarkerCodesParity(unittest.TestCase):
    """剥符号码的真实回归 —— koch P4 是参考数据里唯一 codes_stripped != 0 的页.

    它的 raw 存在已砍的 textscan 历史库里，但生产 debug 记下了被剥掉的 31 条
    （text + 剥前的框 + 理由），而 results.json 里留着活下来的 16 条。把两边
    拼回去重放 strip_marker_codes，就能验证 mboxes / _breaks_line 的全部阈值：
    同样码形的 SF-43 / SF-4 必须留下，-SF-SF- 线里的 31 个必须掉。
    """

    def test_koch_p4(self):
        from steps.text.clean import strip_marker_codes
        res = _load(REF / "fence_fused" / "koch_tennis_center" / "results.json")
        rec = res["pages"]["4"]
        # strip 跑在 snap 之前，所以复原剥前的框要优先取 box_raw。
        kept = [{"text": it["text"], "box_2d": it.get("box_raw") or it["box_2d"],
                 "label": it.get("label", "other")}
                for it in rec["vlm_items"] if it.get("source") == "vlm"]
        dropped = [{"text": d["text"], "box_2d": d["box_2d"],
                    "label": "other"}
                   for d in rec["debug"]["stripped"]]
        self.assertEqual(len(dropped), rec["codes_stripped"])
        self.assertTrue(all("矢量行" not in d["reason"]
                            for d in rec["debug"]["stripped"]))
        instances = [dict(it) for it in rec["vec_covered"]]

        def key(it):
            return (it["text"], tuple(it["box_2d"]))

        dbg = DebugSink()
        keep_v, keep_i, n = strip_marker_codes(
            kept + dropped, instances,
            REF / "projects" / "koch_tennis_center" / "input.pdf", 3, dbg=dbg)
        self.assertEqual(n, rec["codes_stripped"])
        self.assertEqual([key(it) for it in keep_v], [key(it) for it in kept])
        self.assertEqual(keep_i, instances)
        self.assertEqual({(d["text"], tuple(d["box_2d"]), d["reason"])
                          for d in dbg.data["stripped"]},
                         {(d["text"], tuple(d["box_2d"]), d["reason"])
                          for d in rec["debug"]["stripped"]})
        print(f"  [strip] koch P4: dropped {n}, kept {len(keep_v)} vlm "
              f"({[it['text'] for it in keep_v if len(it['text']) <= 6]} "
              "are code-shaped survivors)")


class TestVlmNeeded(unittest.TestCase):
    """付费判据：当期 primary 记录缺失 且（有矢量命中 或 本页记过）。"""

    IDENT = {"pdf_revision": "aa-bb", "model": "m", "prompt_sha256": "d"}

    def _current(self):
        return {"items": [{"text": "FENCE",
                           "box_2d": [1, 1, 2, 2]}],
                "vlm_identity": dict(self.IDENT), "model": "m",
                "vlm_role": PRIMARY_ROLE}

    def test_unseen_page_without_instances_is_free(self):
        self.assertFalse(vlm_needed(3, [], {}, self.IDENT))

    def test_unseen_page_with_instances_is_paid(self):
        self.assertTrue(vlm_needed(3, [{"text": "FENCE"}], {}, self.IDENT))

    def test_current_record_is_a_cache_hit(self):
        store = {"3": self._current()}
        self.assertFalse(vlm_needed(3, [{"text": "FENCE"}], store, self.IDENT))

    def test_known_page_with_stale_identity_is_rework(self):
        rec = self._current()
        rec["vlm_identity"]["pdf_revision"] = "cc-dd"
        self.assertTrue(vlm_needed(3, [], {"3": rec}, self.IDENT))

    def test_known_page_with_error_is_rework(self):
        rec = self._current()
        rec["error"] = "Timeout"
        self.assertTrue(vlm_needed(3, [], {"3": rec}, self.IDENT))

    def test_secondary_role_in_primary_store_is_not_a_hit(self):
        rec = self._current()
        rec["vlm_role"] = SECONDARY_UNION_ROLE
        self.assertTrue(vlm_needed(3, [], {"3": rec}, self.IDENT))

    def test_empty_items_record_is_a_valid_negative(self):
        rec = self._current()
        rec["items"] = []
        self.assertFalse(vlm_needed(3, [{"text": "FENCE"}], {"3": rec},
                                    self.IDENT))

    def test_page_without_text_layer_is_always_paid(self):
        """扫描页矢量通道天生看不见东西，不读图就是静默出空结果。"""
        self.assertTrue(vlm_needed(3, [], {}, self.IDENT, has_text=False))

    def test_no_text_layer_still_respects_a_current_record(self):
        """已经扫过且身份当期，就不该因为「没有文字层」再付一次。"""
        store = {"3": self._current()}
        self.assertFalse(
            vlm_needed(3, [], store, self.IDENT, has_text=False))

    def test_text_layer_without_instances_stays_free(self):
        """有文字层但判词没命中：确定性矢量地板已覆盖，这里就是省钱的地方。"""
        self.assertFalse(vlm_needed(3, [], {}, self.IDENT, has_text=True))


class TestDebugViewFloorFlag(unittest.TestCase):
    """自定义 target（地板关闭）下不得把候选标成 keyword_floor。"""

    ITEM = {"text": "6' CHAIN LINK FENCE", "box_2d": [10, 10, 20, 200]}

    def _judge_tag(self, use_kw_floor):
        rec = {"vec_covered": []}
        sink = DebugSink()
        attach_text_debug(rec, sink, [self.ITEM], None,
                          use_kw_floor=use_kw_floor)
        return rec["debug"]["vector_candidates"][0]["judge"]

    def test_floor_on_reports_keyword_floor(self):
        self.assertEqual(self._judge_tag(True), "keyword_floor")

    def test_floor_off_reports_judge(self):
        self.assertEqual(self._judge_tag(False), "judge_cache")


if __name__ == "__main__":
    unittest.main(verbosity=2)
