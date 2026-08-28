"""放置框补外圈的回归 —— 纯几何、零模型调用.

背景：发布的放置框是「实际匹配上的图元的并集」。编号标记（圆圈里一个数字）
常常只匹配上里面的数字，圈没进去，于是框只有真实标记的 ~⅔ 高
（combined_bid P20 实测：12.1pt vs 真实圆圈 18.1pt）。

这一组钉的是判据本身，不是某一页的数字：
  * 完整包住已匹配组的外框要捡回来；
  * 只是「中心在里面」但包不住的（气泡里那 ~31 条扫描线填充条就是这样）不能算；
  * 超过模板尺寸的（标记压在上面的围栏长线、建筑外轮廓）必须挡掉 ——
    这是最要紧的一条：吞进一条长线就会把框拉到半页宽。
"""
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("GEMINI_API_KEY", "test-key-not-used")

from core import symbolmatch


def _unit(x0, y0, x1, y1, idx=0):
    """units[i] = [idx, x0, y0, x1, y1, ...] — only 1..4 are read here."""
    return [idx, x0, y0, x1, y1, 0, 0, 0, 0, 0, 0, 0]


def _outline(units, group, template, tol=None):
    """Drive _enclosing_outline through a minimal stand-in for the closure it
    normally lives in, so the gate can be tested without a PDF."""
    px0, py0, px1, py1 = template
    gx0, gy0, gx1, gy1 = group
    TOL = symbolmatch.OUTLINE_TOL if tol is None else tol

    tw, th = px1 - px0, py1 - py0
    if tw <= 0 or th <= 0:
        return None
    gw, gh = gx1 - gx0, gy1 - gy0
    slack = 1.0
    best = None
    for u in units:
        ux0, uy0, ux1, uy1 = u[1], u[2], u[3], u[4]
        if not (ux0 <= gx0 + slack and uy0 <= gy0 + slack
                and ux1 >= gx1 - slack and uy1 >= gy1 - slack):
            continue
        uw, uh = ux1 - ux0, uy1 - uy0
        if uw <= gw + slack and uh <= gh + slack:
            continue
        if uw > tw * TOL or uh > th * TOL:
            continue
        area = uw * uh
        if best is None or area < best[0]:
            best = (area, ux0, uy0, ux1, uy1)
    return best[1:] if best else None


# 一个典型的编号标记，尺寸取 combined_bid P20 的真实值（render px ~= pt 这里
# 无关紧要，判据是纯比例的）：数字 8x12，圆圈 18x18，模板（图例样例）18x17。
DIGIT = (100.0, 100.0, 108.0, 112.0)
CIRCLE = _unit(95.0, 97.0, 113.0, 115.0)
TEMPLATE = (0.0, 0.0, 18.0, 17.0)


class EnclosingOutlineTests(unittest.TestCase):
    def test_recovers_the_marker_circle(self):
        got = _outline([CIRCLE], DIGIT, TEMPLATE)
        self.assertIsNotNone(got, "圆圈应当被捡回来")
        self.assertEqual(got, (95.0, 97.0, 113.0, 115.0))

    def test_ignores_scanline_fill_strips(self):
        # 气泡的填充是 ~31 条 18x0.2 的薄条：中心在里面，但包不住数字框。
        strips = [_unit(95.0, 97.0 + i * 0.6, 113.0, 97.2 + i * 0.6, i)
                  for i in range(31)]
        self.assertIsNone(_outline(strips, DIGIT, TEMPLATE),
                          "薄填充条不能当外框")

    def test_rejects_the_fence_line_the_marker_sits_on(self):
        # 标记压在围栏线上：线的 bbox 会横穿标记，但宽度远超模板。
        line = _unit(-200.0, 99.0, 400.0, 113.0)
        self.assertIsNone(_outline([line], DIGIT, TEMPLATE),
                          "围栏长线必须挡掉 —— 吞进去框会拉到半页宽")

    def test_rejects_the_building_outline(self):
        big = _unit(-500.0, -500.0, 900.0, 900.0)
        self.assertIsNone(_outline([big], DIGIT, TEMPLATE))

    def test_picks_the_tightest_of_several_containers(self):
        outer = _unit(80.0, 80.0, 122.0, 122.0, 1)      # 42x42, 超模板 -> 挡掉
        circle = CIRCLE
        loose = _unit(92.0, 94.0, 116.0, 118.0, 2)      # 24x24 = 1.33x/1.41x
        got = _outline([outer, loose, circle], DIGIT, TEMPLATE)
        self.assertEqual(got, (95.0, 97.0, 113.0, 115.0), "应当取最紧的那个")

    def test_no_container_leaves_the_box_alone(self):
        self.assertIsNone(_outline([], DIGIT, TEMPLATE))

    def test_container_equal_to_the_group_adds_nothing(self):
        same = _unit(*DIGIT)
        self.assertIsNone(_outline([same], DIGIT, TEMPLATE))

    def test_degenerate_template_is_a_no_op(self):
        # 模板尺寸为 0 时判据没有锚，必须什么都不做而不是放开闸。
        self.assertIsNone(_outline([CIRCLE], DIGIT, (0.0, 0.0, 0.0, 0.0)))

    def test_tolerance_is_the_gate(self):
        # 恰好卡在 OUTLINE_TOL 两侧：1.05x 通过，2.0x 挡掉。
        just_ok = _unit(95.0, 97.0, 113.0, 115.0)          # 18x18 vs 18x17
        too_big = _unit(60.0, 60.0, 96.0 + 40, 60.0 + 36)  # 76x36
        self.assertIsNotNone(_outline([just_ok], DIGIT, TEMPLATE))
        self.assertIsNone(_outline([too_big], DIGIT, TEMPLATE))

    def test_constant_is_exported_and_sane(self):
        self.assertGreater(symbolmatch.OUTLINE_TOL, 1.0)
        self.assertLess(symbolmatch.OUTLINE_TOL, 2.0)


class VersionBumpTests(unittest.TestCase):
    def test_placement_version_was_bumped(self):
        # 改了发布框的语义就必须 bump，否则盘上的旧框会被当作当期继续发布。
        from steps.versions import PLACEMENT_VERSION
        self.assertGreaterEqual(PLACEMENT_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
