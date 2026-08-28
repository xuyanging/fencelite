"""矢量层校准图例两种框（steps/snap_boxes.py）—— 全离线、零模型调用.

真实故障（drawings_volume_4_binder P4）：模型在整页图上给的九个 shape 样例框
全落在同一列 x=790..800，而真实标记（包着编码字的小多边形）在 x=799.7..808.8，
`4DFG` / `6DFG` 还纵向偏了 6 个单位；同时图例行的文字框从编码开始，把标记
含在了框里。真值在矢量层：编码字是文字层的一个 token，包住它的最小闭合图形
就是标记本体。

这里用 fitz 现造一张最小图纸来验，不依赖任何外部数据。
"""
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

from steps.snap_boxes import snap_symbol_boxes, text_trim_boxes

# 页面 600x600 pt，两行图例：标记(矩形+编码字) + 右侧说明文字。
PAGE = 600.0
ROWS = (  # (码, 标记左上 x, y)
    ("4CL", 300.0, 200.0),
    ("6DF", 300.0, 240.0),
)
MARK_W, MARK_H = 40.0, 20.0


def _norm(v, total=PAGE):
    return v / total * 1000.0


class SnapBoxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.pdf = Path(cls.tmp.name) / "legend.pdf"
        doc = fitz.open()
        page = doc.new_page(width=PAGE, height=PAGE)
        for code, x, y in ROWS:
            page.draw_rect(fitz.Rect(x, y, x + MARK_W, y + MARK_H),
                           color=(0, 0, 0), width=0.8)
            page.insert_text((x + 6, y + 14), code, fontsize=9)
            page.insert_text((x + MARK_W + 14, y + 14),
                             code + " 4'-0\" TALL VINYL CHAIN LINK FENCING",
                             fontsize=9)
        doc.save(str(cls.pdf))
        doc.close()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _wrong_symbol(self, code, x, y, dx=-30.0, dy=2.0):
        """模型给的那种"整列偏左、还纵向漂一点"的框。"""
        return {"category": "shape", "value": code, "text_index": 0,
                "box_2d": [_norm(y + dy), _norm(x + dx),
                           _norm(y + dy + MARK_H), _norm(x + dx + MARK_W)]}

    def test_snaps_onto_the_marker_outline(self):
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol(code, x, y)]
        summary = snap_symbol_boxes(self.pdf, 0, symbols)
        self.assertEqual(summary["snap_shape"], 1)
        box = symbols[0]["box_2d"]
        self.assertEqual(symbols[0]["snap"], "shape")
        # 吸附后应当就是那个矩形标记（容差 2 个归一化单位）
        for got, want in zip(box, (_norm(y), _norm(x),
                                   _norm(y + MARK_H), _norm(x + MARK_W))):
            self.assertAlmostEqual(got, want, delta=2.0)
        self.assertIn("box_raw", symbols[0])      # 原框留痕

    def test_is_idempotent(self):
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol(code, x, y)]
        snap_symbol_boxes(self.pdf, 0, symbols)
        first = list(symbols[0]["box_2d"])
        snap_symbol_boxes(self.pdf, 0, symbols)
        self.assertEqual(symbols[0]["box_2d"], first)

    def test_anchors_on_the_right_row(self):
        """同一个编码在图纸上会印很多次，必须锚在这一行。"""
        code, x, y = ROWS[1]
        symbols = [self._wrong_symbol(code, x, y)]
        snap_symbol_boxes(self.pdf, 0, symbols)
        self.assertAlmostEqual(symbols[0]["box_2d"][0], _norm(y), delta=2.0)

    def test_unknown_code_is_left_alone(self):
        """矢量层里没有这个编码字（模型读错码）→ 保留原框，不许乱猜。"""
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol("9ZZ", x, y)]
        before = list(symbols[0]["box_2d"])
        summary = snap_symbol_boxes(self.pdf, 0, symbols)
        self.assertEqual(summary["snap_shape"], 0)
        self.assertEqual(symbols[0]["box_2d"], before)
        self.assertEqual(symbols[0]["snap"], "no_code_glyph")

    def test_line_symbols_are_not_touched(self):
        symbols = [{"category": "line", "value": "", "text_index": 0,
                    "box_2d": [100, 100, 110, 200]}]
        before = list(symbols[0]["box_2d"])
        snap_symbol_boxes(self.pdf, 0, symbols)
        self.assertEqual(symbols[0]["box_2d"], before)

    def test_text_box_left_edge_is_trimmed_past_the_marker(self):
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol(code, x, y)]
        snap_symbol_boxes(self.pdf, 0, symbols)
        # 图例行文字框：从标记左边一直到说明文字末尾（含编码）
        items = [{"text": code + " 4'-0\" TALL VINYL CHAIN LINK FENCING",
                  "box_2d": [_norm(y), _norm(x + 4),
                             _norm(y + MARK_H), _norm(x + 240)]}]
        trims = text_trim_boxes(self.pdf, 0, items, symbols)
        self.assertIn(0, trims)
        self.assertGreater(trims[0][1], symbols[0]["box_2d"][3] - 0.01)
        # 右边缘不动；上下缘只允许被"行边界"收紧（或对齐到标记那一行），
        # 不允许越过标记本身
        self.assertAlmostEqual(trims[0][3], items[0]["box_2d"][3], places=1)
        self.assertLessEqual(trims[0][0], symbols[0]["box_2d"][0] + 0.01)
        self.assertGreaterEqual(trims[0][2], symbols[0]["box_2d"][2] - 0.01)

    def test_text_box_that_does_not_contain_the_marker_is_untouched(self):
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol(code, x, y)]
        snap_symbol_boxes(self.pdf, 0, symbols)
        items = [{"text": code + " 4'-0\" TALL FENCING",
                  "box_2d": [_norm(y), _norm(x + MARK_W + 10),
                             _norm(y + MARK_H), _norm(x + 240)]}]
        self.assertEqual(text_trim_boxes(self.pdf, 0, items, symbols), {})

    def test_single_token_item_is_not_trimmed(self):
        """裸编码那条 item 不是"图例行"，不该被裁（它自己就是标记）。"""
        code, x, y = ROWS[0]
        symbols = [self._wrong_symbol(code, x, y)]
        snap_symbol_boxes(self.pdf, 0, symbols)
        items = [{"text": code, "box_2d": [_norm(y), _norm(x),
                                           _norm(y + MARK_H), _norm(x + MARK_W)]}]
        self.assertEqual(text_trim_boxes(self.pdf, 0, items, symbols), {})

    def test_malformed_input_is_tolerated(self):
        self.assertEqual(snap_symbol_boxes(self.pdf, 0, None),
                         {"snap_shape": 0, "snap_glyph": 0, "snap_column": 0,
                          "snap_skipped": 0})
        self.assertEqual(text_trim_boxes(self.pdf, 0, None, None), {})
        # 非 dict 的行在入口就被滤掉，只有那个 dict 计入 skipped
        self.assertEqual(
            snap_symbol_boxes(self.pdf, 0, [{"category": "shape"}, 3, None]),
            {"snap_shape": 0, "snap_glyph": 0, "snap_column": 0,
             "snap_skipped": 1})


if __name__ == "__main__":
    unittest.main()
