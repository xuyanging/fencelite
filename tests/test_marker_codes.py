"""图例编码标记的识别（marker_code_indices）—— 全离线、零模型调用.

背景（真实故障）：`drawings_volume_4_binder` P4 里，平面图上每处要装 4' 链条网
围栏的地方都印一个 "4CL" 编码标记。判词把 "4CL" 判成了 fence 相关（语义上不算
错 —— 4' chain link 确实是围栏），于是这 24 个编码各自成了一条独立文字项，前端
把它们画成蓝色文字框，看起来就是"平面图上的 symbol 被框成了文字"。图例那一行
（"4CL 4'-0\" TALL VINYL CHAIN LINK FENCING"）也因此把编码含在行内。

判据是"这条文字整体就是一个编码 token，且这个编码等于某个已发布 symbol 的 value
（或某个图例行的首个 token）"。**只标记不删除**：删 item 会让 store.sig_of 变化、
每页重跑步骤② 的付费推理，还会让 union index 整体错位。
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from steps.symbols import marker_code_indices

LEGEND = {"kind": "legend", "box_2d": [630, 780, 830, 900]}
VIEW = {"kind": "view", "box_2d": [10, 30, 620, 780]}


def item(text, box):
    return {"text": text, "box_2d": box, "label": "legend entry", "tbl": True}


class MarkerCodeTests(unittest.TestCase):
    # ---- 排版 A：编码写在图例描述行的开头（drawings P4） -------------------
    A_ITEMS = [
        item('4CL\n4\'-0" TALL VINYL CHAIN LINK FENCING- SEE SPECS',
             [728, 801, 733, 892]),
        item("4CL", [80, 644, 83, 650]),       # 平面图上的编码标记
        item("4CL", [93, 765, 101, 771]),      # 同上
    ]

    def test_layout_a_marks_plan_instances_only(self):
        idx = marker_code_indices(self.A_ITEMS, [LEGEND, VIEW], [])
        self.assertEqual(idx, {1, 2})

    # ---- 排版 B：编码是独立一条、描述行不含它（drawings P5） ---------------
    B_ITEMS = [
        item('4\'-0" TALL VINYL CHAIN LINK FENCING - SEE SPECS',
             [146, 821, 150, 889]),
        item("4CL", [146, 798, 150, 803]),
    ]
    B_SYMBOLS = [{"category": "shape", "value": "4CL", "text_index": 0,
                  "box_2d": [144, 794, 152, 810]}]

    def test_layout_b_uses_the_symbol_value(self):
        idx = marker_code_indices(self.B_ITEMS, [LEGEND], self.B_SYMBOLS)
        self.assertEqual(idx, {1})

    def test_layout_b_without_symbols_finds_nothing(self):
        """没有已发布 symbol、描述行也不含编码时，没有任何依据 —— 宁可不标。"""
        self.assertEqual(marker_code_indices(self.B_ITEMS, [LEGEND], []), set())

    # ---- 不许误伤 ---------------------------------------------------------
    def test_real_fence_text_is_never_marked(self):
        items = [item('4CL\n4\'-0" TALL VINYL CHAIN LINK FENCING',
                      [728, 801, 733, 892]),
                 item("6' HIGH BLACK VINYL CHAIN LINK FENCE - TYP.",
                      [298, 219, 307, 252]),
                 item("NEW CHAIN LINK FENCE", [269, 839, 277, 893])]
        self.assertEqual(marker_code_indices(items, [LEGEND, VIEW], []), set())

    def test_single_token_that_is_not_code_shaped_is_kept(self):
        """单 token 但不是编码形状（"FENCING"）不该被当成标记。"""
        items = [item("FENCING FENCE DETAILS AND NOTES", [700, 800, 710, 890]),
                 item("FENCING", [300, 200, 310, 240])]
        self.assertEqual(marker_code_indices(items, [LEGEND, VIEW], []), set())

    def test_no_legend_group_and_no_symbols_marks_nothing(self):
        items = [item("4CL", [80, 644, 83, 650])]
        self.assertEqual(marker_code_indices(items, [VIEW], []), set())

    def test_single_token_legend_row_cannot_seed_itself(self):
        """图例里孤零零一条 "4CL" 不能自己把自己认成"图例行的编码"。"""
        items = [item("4CL", [700, 800, 705, 806])]
        self.assertEqual(marker_code_indices(items, [LEGEND], []), set())

    def test_digit_first_codes_are_recognised(self):
        """4CL / 6DMP / 3GP 这类"数字在前"的形态正是 clean.py 正则漏掉的。"""
        for code in ("4CL", "6DMP", "3GP", "F-04", "33", "SF"):
            with self.subTest(code=code):
                items = [item(code + " SOME FENCE DESCRIPTION HERE",
                              [700, 800, 710, 890]),
                         item(code, [300, 200, 310, 240])]
                self.assertEqual(
                    marker_code_indices(items, [LEGEND, VIEW], []), {1})

    def test_case_insensitive(self):
        items = [item('4cl 4\'-0" TALL FENCE', [700, 800, 710, 890]),
                 item("4CL", [300, 200, 310, 240])]
        self.assertEqual(marker_code_indices(items, [LEGEND, VIEW], []), {1})

    def test_malformed_input_is_tolerated(self):
        for bad in (None, "x", [None, 3, {"text": "4CL"}]):
            with self.subTest(bad=bad):
                self.assertEqual(marker_code_indices(bad, [LEGEND], []), set())
        self.assertEqual(marker_code_indices(self.A_ITEMS, None, None), set())


if __name__ == "__main__":
    unittest.main()
