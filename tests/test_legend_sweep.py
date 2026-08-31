"""离线回归：步骤②b 图例块裁剪补扫（steps/legend_sweep.py）.

全部离线 —— 一次真实 Gemini 调用都不许发：模型出口 gen_json 一律换成假函数，
整页渲染要么换成一张内存里的 PIL 图，要么用参考项目的真 PDF **只渲染、不推理**。

参考数据（有就跑、没有就跳过）：
  FENCE_REF_DIR          默认 C:\\Users\\Administrator\\fence_takeoff_web
                         （真 PDF + 真 symbols/results 缓存，用于几何冒烟）
  FL_SWEEP_DIAG_DIR      线上项目缓存副本，用于复现用户报的漏检页
                         （ponderosa P2：整页那次 raw symbols = 0）

跑法：
  cd C:\\Users\\Administrator\\fence_lite
  set PYTHONUTF8=1
  C:\\Users\\Administrator\\fence_detector\\venv\\Scripts\\python.exe -B -m unittest -v
"""
import json
import os
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from steps.debug import DebugSink
from steps.legend_sweep import (LEGEND_SWEEP_VERSION, SWEEP_ATTEMPTS,
                                SWEEP_MIN_CROP_PX, SWEEP_PAD,
                                crop_box_to_page, crop_window,
                                page_box_to_crop, parse_sweep_payload,
                                sweep_needed, sweep_page)
from steps.store import items_of

REF_DIR = Path(os.environ.get(
    "FENCE_REF_DIR", r"C:\Users\Administrator\fence_takeoff_web"))
HAS_REF = (REF_DIR / "fence_fused").is_dir() and (REF_DIR / "projects").is_dir()

DIAG_DIR = Path(os.environ.get(
    "FL_SWEEP_DIAG_DIR",
    r"C:\Users\Administrator\AppData\Local\Temp\claude"
    r"\C--Users-Administrator\5a059b30-437c-43d7-899c-bda2d5c22a41"
    r"\scratchpad\diag"))
PONDEROSA = DIAG_DIR / "2026_01_30_ponderosa_pines_all_buildings_ifc_set_lands_irr"
HAS_PONDEROSA = (PONDEROSA / "symbols.json").is_file() \
    and (PONDEROSA / "results.json").is_file()


class _FakeResponse:
    """gen_json 的返回值只被用到 .text 和 usage_metadata（没有就当没用量）。"""

    def __init__(self, text):
        self.text = text


def _install_fake_gen_json(test, answers):
    """把 steps.legend_sweep.gen_json 换掉，返回记录每次调用的 list.

    answers 里放响应文本（或要抛的异常），按顺序消费，最后一条会一直重复 ——
    「三次都返回坏 JSON」写成一条就够。
    """
    calls = []

    def fake_gen_json(model, contents, timeout_ms=None,
                      response_json_schema=None, **kwargs):
        answer = answers[min(len(calls), len(answers) - 1)]
        calls.append({"model": model, "prompt": contents[-1],
                      "timeout_ms": timeout_ms,
                      "schema": response_json_schema})
        if isinstance(answer, Exception):
            raise answer
        return _FakeResponse(answer)

    patcher = mock.patch("steps.legend_sweep.gen_json", fake_gen_json)
    patcher.start()
    test.addCleanup(patcher.stop)
    # 退避真睡 1.5s/3.0s 会让重试用例慢得没必要 —— 逻辑与 sleep 时长无关。
    backoff = mock.patch("steps.legend_sweep.SWEEP_BACKOFF_S", 0.0)
    backoff.start()
    test.addCleanup(backoff.stop)
    return calls


def _install_fake_render(test, size=(2000, 1000)):
    """整页渲染换成一张内存白图（本文件的逻辑用例不需要真像素）。"""
    rendered = []

    def fake_render(pdf_path, page_index, dpi=None, max_px=None, doc=None):
        rendered.append({"pdf": pdf_path, "page": page_index, "dpi": dpi,
                         "max_px": max_px})
        return Image.new("RGB", size, "white")

    patcher = mock.patch("steps.legend_sweep.render_pdf_page", fake_render)
    patcher.start()
    test.addCleanup(patcher.stop)
    return rendered


def _item(box, text="6' CHAIN LINK FENCE"):
    return {"text": text, "box_2d": list(box), "label": "", "tbl": False}


# ------------------------------------------------------------------ 选块

class TestSweepNeeded(unittest.TestCase):
    GROUPS = [
        {"kind": "view", "box_2d": [0, 0, 900, 500]},
        {"kind": "legend", "box_2d": [100, 600, 300, 900]},
        {"kind": "schedule", "box_2d": [400, 600, 700, 900]},
        {"kind": "title_block", "box_2d": [900, 600, 990, 900]},
    ]

    def test_only_legend_context_groups_are_swept(self):
        items = [
            _item([200, 620, 210, 700]),   # 0 → legend(1)
            _item([500, 620, 510, 700]),   # 1 → schedule(2)
            _item([400, 100, 410, 200]),   # 2 → view，不补扫
            _item([950, 620, 960, 700]),   # 3 → title_block，不补扫
        ]
        blocks = sweep_needed(items, self.GROUPS, [])
        self.assertEqual([(b["group_index"], b["kind"], b["missing"])
                          for b in blocks],
                         [(1, "legend", [0]), (2, "schedule", [1])])
        self.assertTrue(all(b["skipped"] is None for b in blocks))

    def test_items_with_a_published_symbol_are_not_asked_again(self):
        items = [_item([200, 620, 210, 700]), _item([250, 620, 260, 700])]
        symbols = [{"text_index": 0, "box_2d": [200, 605, 210, 615],
                    "category": "shape", "value": "1", "group_index": 1}]
        blocks = sweep_needed(items, self.GROUPS, symbols)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["missing"], [1])

    def test_a_fully_covered_group_is_not_swept_at_all(self):
        items = [_item([200, 620, 210, 700])]
        symbols = [{"text_index": 0, "box_2d": [200, 605, 210, 615],
                    "category": "shape", "value": "1", "group_index": 1}]
        self.assertEqual(sweep_needed(items, self.GROUPS, symbols), [])

    def test_out_of_range_owner_index_does_not_cover_an_item(self):
        items = [_item([200, 620, 210, 700])]
        for bad in (5, -1, True, 1.0, None, "0"):
            with self.subTest(text_index=bad):
                symbols = [{"text_index": bad, "box_2d": [200, 605, 210, 615]}]
                blocks = sweep_needed(items, self.GROUPS, symbols)
                self.assertEqual([b["missing"] for b in blocks], [[0]])

    def test_smallest_containing_group_owns_the_item(self):
        """大 schedule 里嵌一个小 legend：小块更具体，行归小块（_owner_view 口径）。"""
        groups = [
            {"kind": "schedule", "box_2d": [100, 600, 800, 900]},
            {"kind": "legend", "box_2d": [200, 620, 400, 880]},
        ]
        items = [_item([300, 640, 310, 700]),    # 落在两个组里 → 归小的 legend
                 _item([700, 640, 710, 700])]    # 只落在 schedule
        blocks = sweep_needed(items, groups, [])
        self.assertEqual({b["group_index"]: b["missing"] for b in blocks},
                         {1: [0], 0: [1]})

    def test_invalid_boxes_are_ignored_on_both_sides(self):
        groups = [{"kind": "legend", "box_2d": [100, 600, 100, 900]},   # 零高
                  {"kind": "legend", "box_2d": "nope"},
                  {"kind": "legend", "box_2d": [100, 600, 300, 900]}]
        items = [_item([200, 620, 210, 700]),
                 {"text": "no box", "box_2d": None},
                 {"text": "bad box", "box_2d": [1, 2, 3]}]
        blocks = sweep_needed(items, groups, [])
        self.assertEqual([(b["group_index"], b["missing"]) for b in blocks],
                         [(2, [0])])

    def test_center_tolerance_matches_the_step2_gate(self):
        """中心刚好在组框外 2 单位内算落进来（与 symbol_in_allowed_group 同口径）。"""
        groups = [{"kind": "legend", "box_2d": [100, 600, 300, 900]}]
        inside = _item([96, 620, 100, 700])     # 中心 y=98 → 组框上沿 -2
        outside = _item([90, 620, 94, 700])     # 中心 y=92 → 出界
        self.assertEqual(sweep_needed([inside], groups, [])[0]["missing"], [0])
        self.assertEqual(sweep_needed([outside], groups, []), [])

    def test_max_blocks_truncates_but_records_every_dropped_block(self):
        groups, items, expected = [], [], []
        for n in range(5):
            top = 100 * n + 10
            groups.append({"kind": "legend",
                           "box_2d": [top, 600, top + 50, 900]})
            # 第 n 个组放 (5-n) 行 → missing 数降序就是 group_index 升序
            for _ in range(5 - n):
                items.append(_item([top + 10, 620, top + 20, 700]))
            expected.append(5 - n)
        blocks = sweep_needed(items, groups, [], max_blocks=2)
        self.assertEqual([len(b["missing"]) for b in blocks], expected)
        self.assertEqual([b["skipped"] for b in blocks],
                         [None, None, "max_blocks", "max_blocks",
                          "max_blocks"])
        self.assertEqual([b["group_index"] for b in blocks if b["skipped"]],
                         [2, 3, 4])

    def test_env_default_max_blocks_is_honoured(self):
        groups = [{"kind": "legend", "box_2d": [100 * n + 10, 600,
                                                100 * n + 60, 900]}
                  for n in range(6)]
        items = [_item([100 * n + 20, 620, 100 * n + 30, 700])
                 for n in range(6)]
        with mock.patch("steps.legend_sweep.SWEEP_MAX_BLOCKS", 3):
            blocks = sweep_needed(items, groups, [])
        self.assertEqual(sum(1 for b in blocks if not b["skipped"]), 3)
        self.assertEqual(sum(1 for b in blocks if b["skipped"]), 3)

    def test_returned_box_is_a_copy_of_the_group_box(self):
        groups = [{"kind": "legend", "box_2d": [100, 600, 300, 900]}]
        blocks = sweep_needed([_item([200, 620, 210, 700])], groups, [])
        blocks[0]["box_2d"][0] = 999
        self.assertEqual(groups[0]["box_2d"], [100, 600, 300, 900])


# ------------------------------------------------------------ 坐标帧换算

class TestCropFrameMath(unittest.TestCase):
    # 整页渲染图 1000x500 px，裁剪窗 = 页面帧 [100,100,500,500] 那块
    WINDOW = (100, 50, 400, 200)     # cx0, cy0, cw, ch
    SIZE = (1000, 500)               # W, H

    def _to_page(self, box):
        return crop_box_to_page(box, *self.WINDOW, *self.SIZE)

    def test_crop_corners_map_to_the_window_corners(self):
        self.assertEqual(self._to_page([0, 0, 1000, 1000]),
                         [100, 100, 500, 500])

    def test_known_inner_box_round_trips(self):
        page = self._to_page([500, 250, 750, 500])
        self.assertEqual(page, [300, 200, 400, 300])
        back = page_box_to_crop(page, *self.WINDOW, *self.SIZE)
        self.assertEqual(back, [500, 250, 750, 500])

    def test_page_box_outside_the_window_is_clamped_for_the_model(self):
        # 一行长文字伸出裁剪窗 → 截到窗边（这是模型真正看到的样子）
        self.assertEqual(page_box_to_crop([300, 50, 400, 900],
                                          *self.WINDOW, *self.SIZE),
                         [500, 0, 750, 1000])

    def test_tiny_marker_never_collapses_to_zero_area(self):
        """页面帧比裁剪帧粗：四舍五入后必须仍是正面积，否则下游当非法框丢掉。"""
        window, size = (0, 0, 20, 20), (1000, 1000)
        page = crop_box_to_page([500, 500, 505, 505], *window, *size)
        self.assertIsNotNone(page)
        self.assertLess(page[0], page[2])
        self.assertLess(page[1], page[3])

    def test_degenerate_window_returns_none(self):
        self.assertIsNone(crop_box_to_page([0, 0, 10, 10], 0, 0, 0, 100,
                                           1000, 500))
        self.assertIsNone(page_box_to_crop([0, 0, 10, 10], 0, 0, 100, 0,
                                           1000, 500))

    def test_crop_window_pads_and_clamps(self):
        crop_box, window = crop_window([100, 200, 300, 400], 1000, 1000)
        self.assertEqual(crop_box, [100 - SWEEP_PAD, 200 - SWEEP_PAD,
                                    300 + SWEEP_PAD, 400 + SWEEP_PAD])
        self.assertEqual(window, (int(200 - SWEEP_PAD), int(100 - SWEEP_PAD),
                                  int(200 + 2 * SWEEP_PAD),
                                  int(200 + 2 * SWEEP_PAD)))
        edge_box, _edge_window = crop_window([0, 0, 1000, 1000], 800, 600)
        self.assertEqual(edge_box, [0.0, 0.0, 1000.0, 1000.0])


# ------------------------------------------------------------ 响应严校验

class TestParseSweepPayload(unittest.TestCase):
    ASKED = (3, 7)
    GOOD = {"symbols": [{"idx": 3, "box_2d": [100, 100, 200, 400],
                         "category": "line", "value": "SF"}]}

    def test_happy_path(self):
        rows = parse_sweep_payload(json.dumps(self.GOOD), self.ASKED)
        self.assertEqual(rows, [{"idx": 3, "box_2d": [100, 100, 200, 400],
                                 "category": "line", "value": "SF"}])

    def test_markdown_fence_and_empty_answer(self):
        self.assertEqual(
            parse_sweep_payload('```json\n{"symbols": []}\n```', self.ASKED),
            [])

    def test_exact_duplicate_rows_collapse(self):
        payload = {"symbols": [self.GOOD["symbols"][0],
                               dict(self.GOOD["symbols"][0])]}
        self.assertEqual(len(parse_sweep_payload(json.dumps(payload),
                                                 self.ASKED)), 1)

    def test_two_different_samples_for_one_row_both_survive(self):
        payload = {"symbols": [
            {"idx": 3, "box_2d": [100, 100, 200, 400], "category": "line",
             "value": ""},
            {"idx": 3, "box_2d": [100, 420, 200, 460], "category": "shape",
             "value": "A"}]}
        self.assertEqual(len(parse_sweep_payload(json.dumps(payload),
                                                 self.ASKED)), 2)

    def test_every_protocol_violation_raises(self):
        bad = {
            "not json": "definitely not json",
            "top level array": "[]",
            "missing symbols key": '{"found": []}',
            "symbols not a list": '{"symbols": {}}',
            "row not an object": '{"symbols": ["x"]}',
            "missing idx": json.dumps({"symbols": [
                {"box_2d": [1, 1, 2, 2], "category": "line", "value": ""}]}),
            "missing box": json.dumps({"symbols": [
                {"idx": 3, "category": "line", "value": ""}]}),
            "missing category": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2, 2], "value": ""}]}),
            "missing value": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2, 2], "category": "line"}]}),
            "idx not asked": json.dumps({"symbols": [
                {"idx": 4, "box_2d": [1, 1, 2, 2], "category": "line",
                 "value": ""}]}),
            "idx is bool": json.dumps({"symbols": [
                {"idx": True, "box_2d": [1, 1, 2, 2], "category": "line",
                 "value": ""}]}),
            "idx is float": json.dumps({"symbols": [
                {"idx": 3.0, "box_2d": [1, 1, 2, 2], "category": "line",
                 "value": ""}]}),
            "third category": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2, 2], "category": "hatch",
                 "value": ""}]}),
            "value not a string": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2, 2], "category": "line",
                 "value": 4}]}),
            "box out of range": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2, 1400], "category": "line",
                 "value": ""}]}),
            "box inverted": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [200, 1, 100, 400], "category": "line",
                 "value": ""}]}),
            "box too short": json.dumps({"symbols": [
                {"idx": 3, "box_2d": [1, 1, 2], "category": "line",
                 "value": ""}]}),
        }
        for name, text in bad.items():
            with self.subTest(case=name):
                with self.assertRaises(RuntimeError):
                    parse_sweep_payload(text, self.ASKED)


# ------------------------------------------------------------ sweep_page

class TestSweepPage(unittest.TestCase):
    GROUPS = [{"kind": "view", "box_2d": [0, 0, 900, 500]},
              {"kind": "legend", "box_2d": [400, 600, 600, 900]}]
    ITEMS = [_item([500, 700, 510, 800], "6' CHAIN LINK FENCE")]

    def _answer(self, rows):
        return json.dumps({"symbols": rows})

    def test_publishes_page_frame_symbols_shaped_like_step2(self):
        calls = _install_fake_gen_json(self, [self._answer(
            [{"idx": 0, "box_2d": [400, 100, 600, 300], "category": "line",
              "value": "SF"}])])
        rendered = _install_fake_render(self)
        dbg = DebugSink()
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [], dbg=dbg)

        self.assertEqual(out["version"], LEGEND_SWEEP_VERSION)
        self.assertEqual(out["errors"], [])
        self.assertEqual(out["skipped"], [])
        self.assertEqual(out["calls"], 1)
        self.assertEqual(len(rendered), 1)
        self.assertEqual(len(out["added"]), 1)
        symbol = out["added"][0]
        self.assertEqual(sorted(symbol),
                         ["block_index", "box_2d", "category", "group_index",
                          "text_index", "type", "value"])
        self.assertEqual(symbol["block_index"], 0)
        self.assertEqual((symbol["text_index"], symbol["group_index"],
                          symbol["category"], symbol["value"],
                          symbol["type"]),
                         (0, 1, "line", "SF", "line SF"))
        # 裁剪帧 [400,100,600,300] 必须落回图例组框附近的页面帧
        block = out["blocks"][0]
        self.assertEqual(block["asked"], [0])
        self.assertEqual(block["crop_box"],
                         [400 - SWEEP_PAD, 600 - SWEEP_PAD,
                          600 + SWEEP_PAD, 900 + SWEEP_PAD])
        y0, x0, y1, x1 = symbol["box_2d"]
        self.assertTrue(block["crop_box"][0] <= y0 < y1 <= block["crop_box"][2])
        self.assertTrue(block["crop_box"][1] <= x0 < x1 <= block["crop_box"][3])
        # 送进模型的行框是裁剪帧，不是页面帧
        prompt = calls[0]["prompt"]
        self.assertIn("TEXT ROWS IN THIS CROP:", prompt)
        rows = json.loads(prompt.split("TEXT ROWS IN THIS CROP:\n", 1)[1])
        self.assertEqual([r["idx"] for r in rows], [0])
        self.assertEqual(rows[0]["text"], "6' CHAIN LINK FENCE")
        self.assertNotEqual(rows[0]["box_2d"], self.ITEMS[0]["box_2d"])
        self.assertTrue(all(0 <= v <= 1000 for v in rows[0]["box_2d"]))
        # dbg 通道
        self.assertEqual(len(dbg.data["legend_sweep"]), 1)
        self.assertEqual(dbg.data["legend_sweep"][0]["found"], 1)
        self.assertIsNone(dbg.data["legend_sweep"][0]["skipped"])

    def test_no_missing_item_means_no_render_and_no_call(self):
        calls = _install_fake_gen_json(self, ['{"symbols": []}'])
        rendered = _install_fake_render(self)
        symbols = [{"text_index": 0, "box_2d": [500, 650, 510, 660],
                    "category": "shape", "value": "1", "group_index": 1}]
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, symbols)
        self.assertEqual((out["added"], out["blocks"], out["errors"]),
                         ([], [], []))
        self.assertEqual((calls, rendered), ([], []))

    def test_three_bad_answers_go_to_errors_instead_of_raising(self):
        # 一条格式完整、但 idx 不在本次 asked 里的行 —— 最像模型真会犯的错
        calls = _install_fake_gen_json(self, [self._answer(
            [{"idx": 9, "box_2d": [10, 10, 90, 90], "category": "shape",
              "value": "9"}])])
        _install_fake_render(self)
        dbg = DebugSink()
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [], dbg=dbg)
        self.assertEqual(len(calls), SWEEP_ATTEMPTS)
        self.assertEqual(out["calls"], SWEEP_ATTEMPTS)
        self.assertEqual(out["added"], [])
        self.assertEqual(len(out["errors"]), 1)
        self.assertEqual(out["errors"][0]["group_index"], 1)
        self.assertIn("invalid idx", out["errors"][0]["error"])
        # 花了钱的块仍然记一条 block（found 空），不许假装没跑过
        self.assertEqual(len(out["blocks"]), 1)
        self.assertEqual(out["blocks"][0]["found"], [])
        self.assertEqual(dbg.data["legend_sweep"][0]["found"], 0)

    def test_retry_nonce_changes_the_request_bytes(self):
        """Gemini 对逐字节相同的请求会返回同一份坏答案 —— 重试必须改字节。"""
        calls = _install_fake_gen_json(self, ["not json at all"])
        _install_fake_render(self)
        sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [])
        prompts = [c["prompt"] for c in calls]
        self.assertEqual(len(prompts), SWEEP_ATTEMPTS)
        self.assertEqual(len(set(prompts)), SWEEP_ATTEMPTS)
        self.assertNotIn("RETRY VALIDATION ATTEMPT", prompts[0])
        self.assertIn("nonce=sweep-retry-2", prompts[1])
        self.assertIn("nonce=sweep-retry-3", prompts[2])
        self.assertTrue(prompts[1].startswith(prompts[0]))

    def test_transport_failure_also_retries_then_records(self):
        calls = _install_fake_gen_json(self, [RuntimeError("504 deadline")])
        _install_fake_render(self)
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [])
        self.assertEqual(len(calls), SWEEP_ATTEMPTS)
        self.assertIn("504 deadline", out["errors"][0]["error"])

    def test_provider_timeout_gets_one_retry_not_three(self):
        calls = _install_fake_gen_json(
            self, [TimeoutError("provider timed out")])
        _install_fake_render(self)
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["calls"], 2)
        self.assertIn("provider timed out", out["errors"][0]["error"])

    def test_recovers_on_the_second_attempt(self):
        calls = _install_fake_gen_json(self, [
            "truncated {",
            self._answer([{"idx": 0, "box_2d": [10, 10, 900, 900],
                           "category": "shape", "value": "A"}])])
        _install_fake_render(self)
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(out["errors"], [])
        self.assertEqual(len(out["added"]), 1)
        self.assertEqual(out["added"][0]["value"], "A")

    def test_tiny_block_is_skipped_without_paying(self):
        calls = _install_fake_gen_json(self, ['{"symbols": []}'])
        _install_fake_render(self, size=(200, 200))
        groups = [{"kind": "legend", "box_2d": [500, 500, 505, 508]}]
        items = [_item([501, 501, 504, 507])]
        out = sweep_page("x.pdf", 0, items, groups, [])
        self.assertEqual(calls, [])
        self.assertEqual(out["added"], [])
        self.assertEqual(out["skipped"][0]["reason"], "crop_too_small")
        self.assertLess(min(out["skipped"][0]["crop_px"][2:]),
                        SWEEP_MIN_CROP_PX)

    def test_dropped_block_over_the_cap_is_recorded_not_silent(self):
        _install_fake_gen_json(self, ['{"symbols": []}'])
        _install_fake_render(self)
        groups = [{"kind": "legend", "box_2d": [100, 600, 300, 900]},
                  {"kind": "legend", "box_2d": [400, 600, 600, 900]}]
        items = [_item([200, 620, 210, 700]), _item([500, 620, 510, 700])]
        dbg = DebugSink()
        out = sweep_page("x.pdf", 0, items, groups, [], dbg=dbg, max_blocks=1)
        self.assertEqual(len(out["blocks"]), 1)
        self.assertEqual([(s["group_index"], s["reason"])
                          for s in out["skipped"]], [(1, "max_blocks")])
        self.assertEqual([d["skipped"] for d in dbg.data["legend_sweep"]],
                         ["max_blocks", None])

    def test_render_failure_never_takes_the_page_down(self):
        calls = _install_fake_gen_json(self, ['{"symbols": []}'])
        patcher = mock.patch("steps.legend_sweep.render_pdf_page",
                             mock.Mock(side_effect=IndexError("page 99")))
        patcher.start()
        self.addCleanup(patcher.stop)
        out = sweep_page("x.pdf", 99, self.ITEMS, self.GROUPS, [])
        self.assertEqual(calls, [])
        self.assertEqual(out["added"], [])
        self.assertEqual(out["errors"],
                         [{"group_index": None, "error": "IndexError: page 99"}])

    def test_model_and_schema_reach_gen_json(self):
        calls = _install_fake_gen_json(self, ['{"symbols": []}'])
        _install_fake_render(self)
        with mock.patch("steps.legend_sweep.SWEEP_MODEL", "gemini-2.5-flash"):
            out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [],
                             timeout_ms=1234)
        self.assertEqual(out["model"], "gemini-2.5-flash")
        self.assertEqual(calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(calls[0]["timeout_ms"], 1234)
        self.assertEqual(calls[0]["schema"]["required"], ["symbols"])

    def test_unknown_model_id_falls_back_to_a_priced_one(self):
        """价目表外的 id 会让 RECORDER 算不出 USD —— resolve_model 挡在前面。"""
        calls = _install_fake_gen_json(self, ['{"symbols": []}'])
        _install_fake_render(self)
        out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [],
                         model="totally-made-up")
        from core.config import PRICING
        self.assertIn(out["model"], PRICING)
        self.assertEqual(calls[0]["model"], out["model"])

    def test_page_render_uses_the_high_dpi_knobs(self):
        _install_fake_gen_json(self, ['{"symbols": []}'])
        rendered = _install_fake_render(self)
        from steps.legend_sweep import SWEEP_DPI, SWEEP_PAGE_MAX_PX
        sweep_page("x.pdf", 4, self.ITEMS, self.GROUPS, [])
        self.assertEqual(rendered[0], {"pdf": "x.pdf", "page": 4,
                                       "dpi": SWEEP_DPI,
                                       "max_px": SWEEP_PAGE_MAX_PX})

    def test_oversized_crop_is_downscaled_but_mapping_is_unchanged(self):
        answer = self._answer([{"idx": 0, "box_2d": [0, 0, 1000, 1000],
                                "category": "shape", "value": "1"}])
        _install_fake_gen_json(self, [answer])
        _install_fake_render(self, size=(8000, 4000))
        with mock.patch("steps.legend_sweep.SWEEP_MAX_PX", 500):
            out = sweep_page("x.pdf", 0, self.ITEMS, self.GROUPS, [])
        block = out["blocks"][0]
        self.assertLessEqual(max(block["crop_size"]), 500)
        # 缩放不参与换算：整框仍然映回裁剪窗本身
        self.assertEqual(out["added"][0]["box_2d"],
                         [int(round(v)) for v in block["crop_box"]])


# --------------------------------------------------- 与步骤② 发布闸的握手

class TestPublishGateHandshake(unittest.TestCase):
    """补扫找到的框必须真的能过步骤② 的发布闸.

    裁剪窗是组框外扩 SWEEP_PAD 的框，所以样例可能落在原组框外一点点 —— 闸门
    对补扫符号认 blocks[].crop_box 当取景框（steps.symbols._sweep_gate_groups）。
    这一条握手断了，补扫等于白花钱：找到了、然后被闸门吃掉。
    """

    def setUp(self):
        try:
            from steps.symbols import SOURCE_SWEEP, merge_sweep
        except ImportError as exc:                             # noqa: BLE001
            self.skipTest(f"steps.symbols has no sweep merge yet: {exc}")
        self.merge_sweep = merge_sweep
        self.source_sweep = SOURCE_SWEEP

    def test_a_sample_just_outside_the_group_box_still_publishes(self):
        groups = [{"kind": "legend", "box_2d": [400, 600, 600, 900]}]
        items = [_item([500, 700, 510, 800], "6' CHAIN LINK FENCE")]
        # 样例落在组框左沿之外几个单位（裁剪窗内、原组框外）
        _install_fake_gen_json(self, [json.dumps({"symbols": [
            {"idx": 0, "box_2d": [480, 5, 520, 60], "category": "shape",
             "value": "12"}]})])
        _install_fake_render(self)
        sweep = sweep_page("x.pdf", 0, items, groups, [])
        self.assertEqual(len(sweep["added"]), 1)
        self.assertLess(sweep["added"][0]["box_2d"][1], groups[0]["box_2d"][1])

        entry = {"result": {"symbols": [], "groups": groups}}
        self.merge_sweep(entry, items, sweep)
        published = entry["result"]["symbols"]
        self.assertEqual([s["text_index"] for s in published], [0])
        self.assertEqual(published[0]["source"], self.source_sweep)


# ------------------------------------------------- 真实数据：线上漏检那一页

@unittest.skipUnless(HAS_PONDEROSA, f"no cached diag project at {PONDEROSA}")
class TestPonderosaMissedPage(unittest.TestCase):
    """ponderosa P2：整页那次推理 raw symbols = 0，两行 fence 文字一个都没配到.

    这正是补扫要救的场景 —— 用线上缓存断言选块口径把这一页挑出来。
    """

    @classmethod
    def setUpClass(cls):
        cls.results = json.loads(
            (PONDEROSA / "results.json").read_text(encoding="utf-8"))
        cls.symbols = json.loads(
            (PONDEROSA / "symbols.json").read_text(encoding="utf-8"))
        cls.items = items_of(cls.results["pages"]["2"])
        cls.groups = cls.symbols["2"]["result"]["groups"]

    def test_zero_symbols_page_selects_exactly_the_schedule_block(self):
        blocks = sweep_needed(self.items, self.groups, [])
        self.assertEqual(len(self.items), 2)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["kind"], "schedule")
        self.assertEqual(blocks[0]["missing"], [0, 1])
        self.assertIsNone(blocks[0]["skipped"])
        group_box = self.groups[blocks[0]["group_index"]]["box_2d"]
        self.assertEqual(blocks[0]["box_2d"], group_box)

    def test_the_one_symbol_the_old_cache_found_shrinks_the_ask(self):
        published = self.symbols["2"]["result"]["symbols"]
        self.assertEqual([s["text_index"] for s in published], [1])
        blocks = sweep_needed(self.items, self.groups, published)
        self.assertEqual([b["missing"] for b in blocks], [[0]])

    def test_crop_window_of_that_block_is_a_sane_slice_of_the_page(self):
        block = sweep_needed(self.items, self.groups, [])[0]
        crop_box, (cx0, cy0, cw, ch) = crop_window(block["box_2d"], 7000, 4500)
        self.assertTrue(all(0 <= v <= 1000 for v in crop_box))
        self.assertGreaterEqual(min(cw, ch), SWEEP_MIN_CROP_PX)
        self.assertLess(cw * ch, 7000 * 4500)      # 只是一块，不是整页
        self.assertLessEqual(cx0 + cw, 7000)
        self.assertLessEqual(cy0 + ch, 4500)


# --------------------------------------------- 真实几何冒烟（渲染，不调模型）

@unittest.skipUnless(HAS_REF, f"no reference project dir at {REF_DIR}")
class TestRealPdfCropGeometry(unittest.TestCase):
    """真 PDF + 真缓存组框：只渲染 + 裁剪，一次模型调用都不发。"""

    SLUG, PAGE = "rapid_city", "11"

    @classmethod
    def setUpClass(cls):
        base = REF_DIR / "fence_fused" / cls.SLUG
        if not (base / "symbols.json").is_file():
            raise unittest.SkipTest(f"no cached symbols for {cls.SLUG}")
        cls.pdf = REF_DIR / "projects" / cls.SLUG / "input.pdf"
        if not cls.pdf.is_file():
            raise unittest.SkipTest(f"no source pdf for {cls.SLUG}")
        cls.symbols = json.loads(
            (base / "symbols.json").read_text(encoding="utf-8"))
        results = json.loads(
            (base / "results.json").read_text(encoding="utf-8"))
        cls.items = items_of(results["pages"][cls.PAGE])
        cls.groups = cls.symbols[cls.PAGE]["result"]["groups"]

    def test_real_page_blocks_crop_to_sane_windows(self):
        from steps.legend_sweep import SWEEP_DPI, SWEEP_MAX_PX, SWEEP_PAGE_MAX_PX
        from core.pdfio import render_pdf_page

        blocks = sweep_needed(self.items, self.groups, [])
        self.assertTrue(blocks, "reference page should need at least one block")
        page_image = render_pdf_page(self.pdf, int(self.PAGE) - 1,
                                     dpi=SWEEP_DPI, max_px=SWEEP_PAGE_MAX_PX)
        width, height = page_image.size
        self.assertLessEqual(max(width, height), SWEEP_PAGE_MAX_PX)
        for block in blocks:
            with self.subTest(group=block["group_index"]):
                crop_box, (cx0, cy0, cw, ch) = crop_window(block["box_2d"],
                                                           width, height)
                self.assertTrue(all(0 <= v <= 1000 for v in crop_box))
                self.assertLess(crop_box[0], crop_box[2])
                self.assertLess(crop_box[1], crop_box[3])
                self.assertGreaterEqual(min(cw, ch), SWEEP_MIN_CROP_PX)
                self.assertLessEqual(cx0 + cw, width)
                self.assertLessEqual(cy0 + ch, height)
                crop = page_image.crop((cx0, cy0, cx0 + cw, cy0 + ch))
                self.assertEqual(crop.size, (cw, ch))
                # 裁剪块比整页小得多（这一步的全部意义），且不至于大到浪费 token
                self.assertLess(cw * ch, width * height / 4)
                self.assertLessEqual(max(crop.size), SWEEP_MAX_PX)
                # 每一行都能换进裁剪帧，再换回来还在原处附近
                for item_index in block["missing"]:
                    box = self.items[item_index]["box_2d"]
                    in_crop = page_box_to_crop(box, cx0, cy0, cw, ch,
                                               width, height)
                    self.assertIsNotNone(in_crop)
                    back = crop_box_to_page(in_crop, cx0, cy0, cw, ch,
                                            width, height)
                    self.assertIsNotNone(back)
                    for got, want in zip(back, box):
                        self.assertLess(abs(got - want), 3)

    def test_real_page_ownership_matches_the_step2_symbols(self):
        """页上已发布的 symbol 所属的行，不该再被补扫问一遍。"""
        published = self.symbols[self.PAGE]["result"]["symbols"]
        self.assertTrue(published)
        asked = {index for block in sweep_needed(self.items, self.groups,
                                                 published)
                 for index in block["missing"]}
        self.assertFalse(asked & {s["text_index"] for s in published})


class TestAlreadyPairedRowsCostNothing(unittest.TestCase):
    """已经配到样例的行不再花钱重问 —— 包括那些框看起来"窄"的.

    真实案例 taylor_3_12 P3：图例块 x 690..927 里排了两列条目，
    NEW CHAIN LINK FENCE 的文字在 x 839.5，它的 line 样例框 [269,796,277,835]
    只有 37 宽。曾经按「框宽 < (文字左沿 − 块左沿) × 40%」判它"没框全"并重问，
    结果补扫给回几乎一样的框（37 vs 39）、矢量层也证实样例本来就那么长 ——
    基准取错了（把隔壁那一列的宽度也算进了样例区）。这条用例锁住"别再自作聪明"。
    """

    GROUPS = [{"kind": "legend", "box_2d": [28, 690, 282, 927]}]
    ITEMS = [{"box_2d": [269.1, 839.5, 277.1, 893.0],
              "text": "NEW CHAIN LINK FENCE"}]

    def test_narrow_line_box_is_not_re_asked(self):
        symbols = [{"category": "line", "box_2d": [269, 796, 277, 835],
                    "text_index": 0}]
        self.assertEqual(sweep_needed(self.ITEMS, self.GROUPS, symbols), [])

    def test_bare_marker_code_rows_are_never_asked(self):
        """裸编码行不是图例描述行，样例归属在旁边那条完整描述上。

        真实案例 drawings_volume_4_binder P5："4CL" / "6CL" 漏进了文字层成了
        独立 item。不排除的话，同框去重刚把重复收敛掉，补扫又会把它当成
        "还没配到样例"、再配一次同一个 marker。
        """
        items = [{"box_2d": [146.1, 798.2, 149.8, 803.2], "text": "4CL"},
                 {"box_2d": [146.2, 821.2, 150.1, 889.1],
                  "text": '4\'-0" TALL VINYL CHAIN LINK FENCING - SEE SPECS'}]
        groups = [{"kind": "legend", "box_2d": [35, 780, 215, 890]}]
        blocks = sweep_needed(items, groups, [])
        self.assertEqual([b["missing"] for b in blocks], [[1]])

    def test_marker_code_shapes_recognised(self):
        from steps.legend_sweep import _is_marker_code
        for code in ("4CL", "6DMP", "F-04", "33", "A12b", "SF", "3GP"):
            self.assertTrue(_is_marker_code(code), code)
        for text in ("NEW CHAIN LINK FENCE", "4' TALL FENCE", "GATE 12 WIDE"):
            self.assertFalse(_is_marker_code(text), text)

    def test_only_unpaired_rows_are_asked(self):
        # 第二行也要落在图例块的 y 范围（28..282）里，否则它压根不属于这一块
        items = self.ITEMS + [{"box_2d": [250.0, 839.5, 258.0, 893.0],
                               "text": "PROPERTY LINE"}]
        symbols = [{"category": "line", "box_2d": [269, 796, 277, 835],
                    "text_index": 0}]
        blocks = sweep_needed(items, self.GROUPS, symbols)
        self.assertEqual([b["missing"] for b in blocks], [[1]])


if __name__ == "__main__":
    unittest.main()
