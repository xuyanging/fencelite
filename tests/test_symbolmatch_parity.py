"""shape 符号匹配的逐字段回归 —— 对齐生产 5051 的 vecmatch 输出.

tests/fixtures/placements_expected.json 是用参考实现
（fence_takeoff_web/core/vecmatch.py::find_symbol_placements）在参考项目的
每一个图例 symbol 框 + 一组固定探针框上跑出来的**冻结**结果。本测试用
core.symbolmatch.find_symbol_placements 复算同样的输入，要求逐字段（含
error 字符串、placements 每个整数、scale / template_hit / count）完全相同。

同时校验 core.vecgeom 的提取层没有漂移：每页的 zoom / w / h / units /
texts / prims 都有 sha1 指纹，图元原子化只要有一处变了就会失败。

完全离线：只读本地 PDF 与已冻结的 JSON，零 Gemini 调用。
参考数据目录用环境变量 FENCE_REF_DIR 定位，缺失时整个类跳过。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vecgeom import _extract_page, _get_prims  # noqa: E402
from core.symbolmatch import find_symbol_placements  # noqa: E402

REF_DIR = Path(os.environ.get("FENCE_REF_DIR",
                              r"C:\Users\Administrator\fence_takeoff_web"))
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "placements_expected.json"


def _load_fixture():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


def _sha1(obj):
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def _canon(value):
    """Type-strict canonical form: 805 and 805.0 must not compare equal."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _first_diff(expected, actual, path="result"):
    """Human-readable location of the first difference, or None."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}: unexpected key = {actual[key]!r}"
            if key not in actual:
                return f"{path}.{key}: missing (expected {expected[key]!r})"
            sub = _first_diff(expected[key], actual[key], f"{path}.{key}")
            if sub:
                return sub
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        for index in range(min(len(expected), len(actual))):
            sub = _first_diff(expected[index], actual[index],
                              f"{path}[{index}]")
            if sub:
                return sub
        if len(expected) != len(actual):
            return (f"{path}: length {len(actual)} != expected "
                    f"{len(expected)}")
        return None
    if _canon(expected) != _canon(actual):
        return f"{path}: {actual!r} != expected {expected!r}"
    return None


def _pdf_of(slug):
    return REF_DIR / "projects" / slug / "input.pdf"


@unittest.skipUnless(REF_DIR.is_dir(),
                     f"reference project not available at {REF_DIR} "
                     f"(set FENCE_REF_DIR)")
class SymbolPlacementParityTests(unittest.TestCase):
    """Frozen-output parity against the production matcher."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = _load_fixture()

    def test_fixture_covers_every_reference_symbol(self):
        """The frozen set must still match the reference symbols.json."""
        frozen = {(c["slug"], c["page"], c["symbol_index"])
                  for c in self.fixture["cases"] if c["kind"] == "symbol"}
        found = set()
        for slug in sorted({c["slug"] for c in self.fixture["cases"]}):
            path = REF_DIR / "fence_fused" / slug / "symbols.json"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as fh:
                entries = json.load(fh)
            for page_str, entry in entries.items():
                symbols = ((entry.get("raw") or {}).get("symbols")) or []
                for index in range(len(symbols)):
                    found.add((slug, int(page_str), index))
        self.assertTrue(frozen, "fixture has no symbol cases")
        self.assertEqual(sorted(found - frozen), [],
                         "reference symbols missing from the fixture — "
                         "regenerate tests/fixtures/placements_expected.json")

    def test_page_vector_layer_fingerprints(self):
        """core.vecgeom extraction / prim atomisation must not drift."""
        mismatches = []
        for page in self.fixture["pages"]:
            pdf = _pdf_of(page["slug"])
            if not pdf.exists():
                self.skipTest(f"missing reference PDF {pdf}")
            index = page["page"] - 1
            data = _extract_page(str(pdf), index)
            prims, by_class, grid = _get_prims(str(pdf), index)
            canon = [[p.get("src"), p.get("tid"), p["x"], p["y"],
                      list(p["c"]), p["s"], p.get("o"),
                      list(p.get("bbox") or p.get("segment") or ())]
                     for p in prims]
            actual = {
                "zoom": data["zoom"], "w": data["w"], "h": data["h"],
                "n_units": len(data["units"]), "n_texts": len(data["texts"]),
                "n_prims": len(prims), "n_classes": len(by_class),
                "n_cells": len(grid),
                "units_sha1": _sha1(data["units"]),
                "texts_sha1": _sha1(data["texts"]),
                "prims_sha1": _sha1(canon),
            }
            expected = {k: v for k, v in page.items()
                        if k not in ("slug", "page")}
            diff = _first_diff(expected, actual,
                               f"{page['slug']} P{page['page']}")
            if diff:
                mismatches.append(diff)
        if mismatches:
            self.fail(f"{len(mismatches)}/{len(self.fixture['pages'])} page(s) "
                      f"drifted; first: {mismatches[0]}")

    def test_symbol_placements_match_frozen_output(self):
        """find_symbol_placements must reproduce the frozen result verbatim."""
        # Group per page so the LRU extraction cache is hit, not thrashed.
        by_page = OrderedDict()
        for case in self.fixture["cases"]:
            by_page.setdefault((case["slug"], case["page_index"]), []).append(case)

        mismatches = []
        checked = 0
        for (slug, page_index), cases in by_page.items():
            pdf = _pdf_of(slug)
            if not pdf.exists():
                self.skipTest(f"missing reference PDF {pdf}")
            for case in cases:
                actual = find_symbol_placements(str(pdf), page_index,
                                                case["box_2d"])
                checked += 1
                label = (f"{slug} P{case['page']} {case['kind']}"
                         f"/{case['symbol_index']} box={case['box_2d']}")
                diff = _first_diff(case["result"], actual, label)
                if diff:
                    mismatches.append(diff)
        self.assertEqual(len(self.fixture["cases"]), checked)
        if mismatches:
            self.fail(f"{len(mismatches)}/{checked} case(s) differ; "
                      f"first: {mismatches[0]}")

    def test_line_sample_box_still_refuses_symbol_matching(self):
        """A segments-only legend sample must keep hitting the line error."""
        line_cases = [c for c in self.fixture["cases"]
                      if c["kind"] == "symbol" and c["category"] == "line"]
        self.assertTrue(line_cases, "fixture lost its line-category cases")
        refused = 0
        for case in line_cases:
            pdf = _pdf_of(case["slug"])
            if not pdf.exists():
                self.skipTest(f"missing reference PDF {pdf}")
            result = find_symbol_placements(str(pdf), case["page_index"],
                                            case["box_2d"])
            self.assertEqual(case["result"], result, case["box_2d"])
            if str(result.get("error", "")).startswith("the box holds only line segments"):
                refused += 1
        self.assertGreater(refused, 0,
                           "no line sample reached the segments-only branch")


if __name__ == "__main__":
    unittest.main()
