"""离线回归：多提供方分派（Gemini / Anthropic Claude）与对比运行的 slug 语义.

全部离线 —— 一次付费调用都不发：Anthropic 那一侧用假 client，Gemini 那一侧
用假 generate_content，图片用 PIL 现搓。

这一组要钉住的东西，每一条都对应一个真会静默出错的地方：
  * gen_json 必须按 model id 分派 —— 分派错了不会报错，只会把跑分算到
    另一个提供方头上；
  * usage 映射必须把 thinking 从 output_tokens 里拆出来 —— Anthropic 的
    output_tokens 已含 thinking，而 compute_cost 会把 candidates + thoughts
    相加，不拆就double count；
  * >20 张图的那一档尺寸闸 —— 越界不是降级而是整个请求 400；
  * 模型覆盖的优先级 —— 覆盖失效时整条流水线会静默留在 Gemini 上。

跑法：
  cd /home/ubuntu/fence_lite
  venv/bin/python -m pytest tests/test_llm_dispatch.py -q
"""
import base64
import io
import json
import sys
import time
import types
import unittest
from unittest import mock

from PIL import Image

from core import gemini as core_gemini
from core import llm
from core.config import (MODEL_NAME, PRICING, compute_cost, get_model_override,
                         provider_of, resolve_model, set_model_override)


def _part(data, mime="image/png"):
    """A stand-in for google.genai types.Part.from_bytes (only .inline_data is read)."""
    blob = types.SimpleNamespace(data=data, mime_type=mime)
    return types.SimpleNamespace(inline_data=blob, text=None)


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (240, 240, 240)).save(buf, format="PNG")
    return buf.getvalue()


class TestRegistry(unittest.TestCase):
    def test_claude_models_registered_as_anthropic(self):
        for mid in ("claude-sonnet-5", "claude-opus-5"):
            self.assertIn(mid, PRICING)
            self.assertEqual(provider_of(mid), "anthropic")

    def test_gemini_models_keep_default_provider(self):
        self.assertEqual(provider_of("gemini-3.1-pro-preview"), "gemini")
        self.assertEqual(provider_of("nonexistent-model"), "gemini")

    def test_claude_pricing_is_costable(self):
        # An id missing from PRICING makes compute_cost return None, which the
        # recorder turns into a silent $0 — the exact failure this guards.
        cost = compute_cost("claude-sonnet-5",
                            {"input_tokens": 1_000_000,
                             "output_tokens": 1_000_000})
        self.assertIsNotNone(cost)
        self.assertAlmostEqual(cost["input_usd"], 2.00, places=6)
        self.assertAlmostEqual(cost["output_usd"], 10.00, places=6)

    def test_thinking_tokens_are_billed_as_output(self):
        cost = compute_cost("claude-sonnet-5",
                            {"input_tokens": 0, "output_tokens": 500_000,
                             "thoughts_tokens": 500_000})
        self.assertAlmostEqual(cost["output_usd"], 10.00, places=6)


class TestModelOverride(unittest.TestCase):
    def tearDown(self):
        set_model_override(None)

    def test_default_is_process_model(self):
        set_model_override(None)
        self.assertEqual(resolve_model(None), MODEL_NAME)

    def test_override_applies_when_no_explicit_model(self):
        set_model_override("claude-sonnet-5")
        self.assertEqual(resolve_model(None), "claude-sonnet-5")
        self.assertEqual(get_model_override(), "claude-sonnet-5")

    def test_explicit_model_outranks_override(self):
        set_model_override("claude-sonnet-5")
        self.assertEqual(resolve_model("gemini-2.5-flash"), "gemini-2.5-flash")

    def test_unknown_override_is_rejected_not_stored(self):
        set_model_override("not-a-model")
        self.assertIsNone(get_model_override())
        self.assertEqual(resolve_model(None), MODEL_NAME)

    def test_clearing_restores_default(self):
        set_model_override("claude-opus-5")
        set_model_override(None)
        self.assertEqual(resolve_model(None), MODEL_NAME)


class TestGenJsonDispatch(unittest.TestCase):
    """gen_json 是全项目唯一的付费入口，分派对不对全看这里。"""

    def test_claude_id_goes_to_anthropic_backend(self):
        sentinel = object()
        with mock.patch.object(llm, "generate_json",
                               return_value=sentinel) as fake_llm, \
             mock.patch.object(core_gemini.client.models, "generate_content",
                               side_effect=AssertionError(
                                   "Gemini must not be called for a Claude id")):
            out = core_gemini.gen_json("claude-sonnet-5", ["hi"],
                                       timeout_ms=1234, thinking_budget=512,
                                       response_json_schema={"type": "object"})
        self.assertIs(out, sentinel)
        self.assertEqual(fake_llm.call_args.args[0], "claude-sonnet-5")
        self.assertEqual(fake_llm.call_args.kwargs["timeout_ms"], 1234)
        self.assertEqual(fake_llm.call_args.kwargs["thinking_budget"], 512)
        self.assertEqual(fake_llm.call_args.kwargs["response_json_schema"],
                         {"type": "object"})

    def test_gemini_id_still_goes_to_google_genai(self):
        fake_resp = types.SimpleNamespace(text="{}", usage_metadata=None)
        with mock.patch.object(core_gemini.client.models, "generate_content",
                               return_value=fake_resp) as fake_gc, \
             mock.patch.object(llm, "generate_json",
                               side_effect=AssertionError(
                                   "Anthropic must not be called for a Gemini id")):
            out = core_gemini.gen_json("gemini-2.5-flash", ["hi"])
        self.assertIs(out, fake_resp)
        self.assertEqual(fake_gc.call_args.kwargs["model"], "gemini-2.5-flash")

    def test_recorder_counts_claude_calls(self):
        usage = types.SimpleNamespace(input_tokens=1000, output_tokens=200,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0,
                                      output_tokens_details=None)
        resp = llm._Response("{}", usage, "end_turn", "claude-sonnet-5")
        core_gemini.RECORDER.start()
        try:
            with mock.patch.object(llm, "generate_json", return_value=resp):
                core_gemini.gen_json("claude-sonnet-5", ["hi"])
        finally:
            summary = core_gemini.RECORDER.stop()
        self.assertEqual(summary["calls"], 1)
        self.assertIn("claude-sonnet-5", summary["by_model"])
        self.assertGreater(summary["cost_usd"], 0)


class TestUsageMapping(unittest.TestCase):
    def test_thinking_is_split_out_of_output_tokens(self):
        # Anthropic's output_tokens INCLUDES thinking; Gemini reports them
        # separately and compute_cost sums candidates + thoughts. Without the
        # split the run is billed for thinking twice.
        usage = types.SimpleNamespace(
            input_tokens=1000, output_tokens=500,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
            output_tokens_details=types.SimpleNamespace(thinking_tokens=300))
        shim = llm._Usage(usage)
        self.assertEqual(shim.candidates_token_count, 200)
        self.assertEqual(shim.thoughts_token_count, 300)
        self.assertEqual(shim.candidates_token_count
                         + shim.thoughts_token_count, 500)

    def test_cache_tokens_fold_into_prompt_total(self):
        usage = types.SimpleNamespace(
            input_tokens=100, output_tokens=10,
            cache_read_input_tokens=700, cache_creation_input_tokens=200,
            output_tokens_details=None)
        shim = llm._Usage(usage)
        self.assertEqual(shim.prompt_token_count, 1000)
        self.assertEqual(shim.cached_content_token_count, 700)

    def test_maps_onto_gemini_field_names(self):
        usage = types.SimpleNamespace(
            input_tokens=7, output_tokens=3,
            cache_read_input_tokens=0, cache_creation_input_tokens=0,
            output_tokens_details=None)
        resp = llm._Response("{}", usage, "end_turn", "claude-sonnet-5")
        # This is the function every step uses to read usage.
        out = core_gemini.usage_from_response(resp)
        self.assertEqual(out["input_tokens"], 7)
        self.assertEqual(out["output_tokens"], 3)


class TestSchemaTranslation(unittest.TestCase):
    def test_additional_properties_injected_at_every_depth(self):
        src = {"type": "object",
               "properties": {"rows": {"type": "array", "items": {
                   "type": "object",
                   "properties": {"v": {"type": "string"}},
                   "required": ["v"]}}},
               "required": ["rows"]}
        out = llm._strict_schema(src)
        self.assertFalse(out["additionalProperties"])
        self.assertFalse(out["properties"]["rows"]["items"]["additionalProperties"])
        # must not mutate the caller's schema (they are module-level constants)
        self.assertNotIn("additionalProperties", src)

    def test_existing_value_is_preserved(self):
        out = llm._strict_schema({"type": "object", "additionalProperties": True})
        self.assertTrue(out["additionalProperties"])

    def test_real_pipeline_schemas_translate(self):
        from steps.legend_sweep import SWEEP_RESPONSE_SCHEMA
        from steps.prompts import (GROUP_SYMBOL_RESPONSE_SCHEMA,
                                   VIEW_CLASSIFIER_SCHEMA)
        for schema in (SWEEP_RESPONSE_SCHEMA, GROUP_SYMBOL_RESPONSE_SCHEMA,
                       VIEW_CLASSIFIER_SCHEMA):
            out = llm._strict_schema(schema)
            self.assertFalse(out["additionalProperties"])


class TestImagePreparation(unittest.TestCase):
    def _blocks(self, contents):
        blocks, _dims = llm._build_content(contents)
        return blocks

    def _decoded_sizes(self, blocks):
        sizes = []
        for b in blocks:
            if b["type"] != "image":
                continue
            raw = base64.standard_b64decode(b["source"]["data"])
            sizes.append(Image.open(io.BytesIO(raw)).size)
        return sizes

    def test_small_image_passes_through_untouched(self):
        data = _png(800, 600)   # 638 visual tokens: under budget, no resize
        blocks = self._blocks([_part(data), "prompt"])
        self.assertEqual(self._decoded_sizes(blocks), [(800, 600)])
        self.assertEqual(base64.standard_b64decode(
            blocks[0]["source"]["data"]), data)

    def test_text_comes_after_images(self):
        blocks = self._blocks([_part(_png(10, 10)), "prompt"])
        self.assertEqual([b["type"] for b in blocks], ["image", "text"])

    def test_many_image_request_is_capped_at_2000px(self):
        # >20 images drops the per-image dimension limit; exceeding it is a
        # hard invalid_request_error, not a downgrade.
        parts = [_part(_png(3000, 2000)) for _ in range(21)]
        blocks = self._blocks(parts + ["prompt"])
        for w, h in self._decoded_sizes(blocks):
            self.assertLessEqual(max(w, h), llm._MANY_IMAGE_MAX_EDGE)

    def test_large_image_is_pre_resized_to_the_visual_token_budget(self):
        # Pre-resizing is the point: the coordinates Claude returns are in the
        # frame it actually saw, so this code must send the frame it computed
        # rather than let the server pick one it never sees.
        parts = [_part(_png(3000, 2000)) for _ in range(3)]
        blocks = self._blocks(parts + ["prompt"])
        for w, h in self._decoded_sizes(blocks):
            self.assertLessEqual(llm._visual_tokens(w, h), llm._MAX_TOKENS_VISUAL)
            self.assertAlmostEqual(w / h, 1.5, delta=0.02)   # aspect preserved

    def test_target_size_matches_the_documented_algorithm(self):
        # The token budget binds long before the edge limit on drawing scans:
        # a 4896x3168 sheet lands at 2380x1540, not at the 2576 edge.
        self.assertEqual(llm._target_size(4896, 3168, 2576, 4784), (2380, 1540))
        self.assertEqual(llm._target_size(800, 600, 2576, 4784), (800, 600))

    def test_dims_are_reported_for_the_first_image(self):
        blocks, dims = llm._build_content([_part(_png(4896, 3168)), "p"])
        self.assertEqual(dims, (2380, 1540))

    def test_system_prompt_names_both_denominators(self):
        # The observed failure was y normalized against the wrong dimension,
        # so both numbers must appear explicitly.
        sysmsg = llm._system_json((2380, 1540))
        self.assertIn("2380", sysmsg)
        self.assertIn("1540", sysmsg)

    def test_oversized_edge_is_resized(self):
        blocks = self._blocks([_part(_png(9000, 1000))])
        (w, h), = self._decoded_sizes(blocks)
        self.assertLessEqual(max(w, h), llm._MAX_EDGE)
        self.assertAlmostEqual(w / h, 9.0, delta=0.2)  # aspect preserved

    def test_undecodable_bytes_pass_through(self):
        blocks = self._blocks([_part(b"not an image", "image/png")])
        self.assertEqual(base64.standard_b64decode(
            blocks[0]["source"]["data"]), b"not an image")


class TestCoordConversion(unittest.TestCase):
    """pixel 模式：把像素框换回流水线要的 0-1000 帧。

    默认是 normalized —— 在 gladstone P2 上实测 pixel 模式更差（平均 IoU
    0.189 vs 0.287，中心误差 86.5 vs 41.5 单位），与官方通用建议相反，所以
    这条路留着但不作默认。"""

    def test_default_mode_is_normalized(self):
        self.assertEqual(llm.COORD_MODE, "normalized")

    def test_pixels_rescale_to_the_0_1000_frame(self):
        body = json.dumps([{"text": "x", "box_2d": [770, 1190, 1540, 2380]}])
        out = json.loads(llm._to_normalized(body, (2380, 1540)))
        # y/1540*1000 and x/2380*1000
        self.assertEqual(out[0]["box_2d"], [500, 500, 1000, 1000])

    def test_nested_boxes_are_found(self):
        body = json.dumps({"groups": [{"box_2d": [0, 0, 1540, 2380]}],
                           "symbols": [{"box_2d": [770, 1190, 1540, 2380]}]})
        out = json.loads(llm._to_normalized(body, (2380, 1540)))
        self.assertEqual(out["groups"][0]["box_2d"], [0, 0, 1000, 1000])
        self.assertEqual(out["symbols"][0]["box_2d"], [500, 500, 1000, 1000])

    def test_unparseable_text_is_returned_unchanged(self):
        # Better that a validator rejects a box than that this silently
        # rescales something it misread.
        for bad in ("I could not find anything", "```json\n[]\n```", ""):
            self.assertEqual(llm._to_normalized(bad, (2380, 1540)), bad)

    def test_missing_dims_is_a_no_op(self):
        body = json.dumps([{"box_2d": [1, 2, 3, 4]}])
        self.assertEqual(llm._to_normalized(body, None), body)

    def test_non_numeric_box_is_left_alone(self):
        body = json.dumps([{"box_2d": ["a", "b", "c", "d"]}])
        out = llm._to_normalized(body, (2380, 1540))
        self.assertEqual(json.loads(out)[0]["box_2d"], ["a", "b", "c", "d"])

    def test_pixel_system_prompt_states_the_override(self):
        msg = llm._system_pixel((2380, 1540))
        self.assertIn("2380", msg)
        self.assertIn("1540", msg)
        # the override must be explicit or it just conflicts with the user turn
        self.assertIn("OVERRIDE", msg.upper())


class TestPixelFrameRepair(unittest.TestCase):
    """像素帧兜底：group+symbol 那条提示词下 Claude 会不按 0-1000 答、直接给
    2380x1540 帧的像素框。内容对、单位错，且帧是已知的，所以可以救。

    关键在**全有或全无**：只有整个回复换算后每个框都合法才换算。混合单位的
    回复（实测约 2/3 是这种）必须继续失败 —— 否则就会造出一个"看着像真的、
    几何上是错的"框贴在页边，正是 core/parsing.py 拒绝 clamp 的那个理由。"""

    DIMS = (2380, 1540)

    def test_clean_pixel_reply_is_converted(self):
        body = json.dumps({"groups": [{"box_2d": [0, 1975, 1540, 2380],
                                       "kind": "title_block"}]})
        out, n = llm._repair_pixel_frame(body, self.DIMS)
        self.assertEqual(n, 1)
        box = json.loads(out)["groups"][0]["box_2d"]
        self.assertEqual(box, [0, 830, 1000, 1000])
        self.assertTrue(llm._valid_normalized(box))

    def test_mixed_unit_reply_is_refused(self):
        # observed: xmax normalized while xmin is pixels -> nonsense either way
        body = json.dumps({"groups": [{"box_2d": [0, 1980, 1000, 1000],
                                       "kind": "title_block"}]})
        out, n = llm._repair_pixel_frame(body, self.DIMS)
        self.assertEqual(n, 0)
        self.assertEqual(out, body)      # untouched: let the validator reject

    def test_observed_mixed_reply_is_refused_in_full(self):
        # Verbatim from a live gladstone P3 attempt. Three of the four boxes
        # read fine as pixels; the title_block runs x 1980 -> 1000, which is
        # inverted under either reading. One incoherent box must veto the whole
        # conversion, or the other three get rewritten on a false premise.
        body = json.dumps({"groups": [
            {"box_2d": [55, 326, 1000, 1980], "kind": "view"},
            {"box_2d": [0, 0, 570, 1000], "kind": "note_cluster"},
            {"box_2d": [0, 1980, 1000, 1000], "kind": "title_block"},
            {"box_2d": [55, 0, 140, 1000], "kind": "other"}]})
        out, n = llm._repair_pixel_frame(body, self.DIMS)
        self.assertEqual(n, 0)
        self.assertEqual(out, body)

    def test_a_coherent_all_pixel_reply_is_converted(self):
        # The counterpart: every box resolves, so the rescue is safe to apply.
        body = json.dumps({"groups": [
            {"box_2d": [55, 326, 1000, 1980], "kind": "view"},
            {"box_2d": [0, 0, 570, 1000], "kind": "note_cluster"}]})
        out, n = llm._repair_pixel_frame(body, self.DIMS)
        self.assertEqual(n, 2)
        for g in json.loads(out)["groups"]:
            self.assertTrue(llm._valid_normalized(g["box_2d"]))

    def test_in_range_reply_is_taken_at_its_word(self):
        body = json.dumps({"groups": [{"box_2d": [10, 20, 30, 40],
                                       "kind": "view"}]})
        out, n = llm._repair_pixel_frame(body, self.DIMS)
        self.assertEqual(n, 0)
        self.assertEqual(out, body)

    def test_non_json_is_untouched(self):
        for bad in ("", "no idea", "```json\n{}\n```"):
            self.assertEqual(llm._repair_pixel_frame(bad, self.DIMS), (bad, 0))

    def test_missing_dims_is_a_no_op(self):
        body = json.dumps({"groups": [{"box_2d": [0, 1975, 1540, 2380]}]})
        self.assertEqual(llm._repair_pixel_frame(body, None), (body, 0))

    def test_valid_normalized_rejects_degenerate_boxes(self):
        self.assertFalse(llm._valid_normalized([10, 20, 10, 40]))   # zero height
        self.assertFalse(llm._valid_normalized([10, 20, 30, 20]))   # zero width
        self.assertFalse(llm._valid_normalized([10, 20, 30, 1001]))  # out of range
        self.assertFalse(llm._valid_normalized([-1, 20, 30, 40]))   # negative
        self.assertTrue(llm._valid_normalized([10, 20, 30, 40]))


class TestEffortMapping(unittest.TestCase):
    def test_no_budget_uses_configured_effort(self):
        self.assertEqual(llm._effort_for(None), llm.EFFORT)

    def test_small_budget_maps_to_low(self):
        self.assertEqual(llm._effort_for(1024), "low")

    def test_mid_budget_maps_to_medium(self):
        self.assertEqual(llm._effort_for(4096), "medium")

    def test_garbage_budget_falls_back(self):
        self.assertEqual(llm._effort_for("nonsense"), llm.EFFORT)


class TestConcurrencyGate(unittest.TestCase):
    """并发闸：账号的 concurrent-connections 限额比流水线的 worker 数低，
    越限返回 429 且重试也会继续相撞，所以必须在这一层挡住。"""

    def test_semaphore_matches_configured_cap(self):
        self.assertEqual(llm._slots._initial_value, llm.MAX_CONCURRENCY)

    def test_cap_is_at_least_one(self):
        self.assertGreaterEqual(llm.MAX_CONCURRENCY, 1)

    def test_sends_never_exceed_the_cap(self):
        import threading as _t

        live = {"now": 0, "peak": 0}
        lock = _t.Lock()

        def fake_stream(**kwargs):
            class Ctx:
                def __enter__(self_inner):
                    with lock:
                        live["now"] += 1
                        live["peak"] = max(live["peak"], live["now"])
                    time.sleep(0.05)
                    return self_inner

                def __exit__(self_inner, *a):
                    with lock:
                        live["now"] -= 1

                def get_final_message(self_inner):
                    return types.SimpleNamespace(
                        stop_reason="end_turn", model="claude-sonnet-5",
                        content=[types.SimpleNamespace(type="text", text="{}")],
                        usage=types.SimpleNamespace(
                            input_tokens=1, output_tokens=1,
                            cache_read_input_tokens=0,
                            cache_creation_input_tokens=0,
                            output_tokens_details=None))
            return Ctx()

        fake_client = types.SimpleNamespace(
            messages=types.SimpleNamespace(stream=fake_stream),
            with_options=lambda **kw: fake_client)

        with mock.patch.object(llm, "get_client", return_value=fake_client):
            # gen_json, not generate_json: the slot is taken there so that
            # queue-wait stays outside the recorder's timing.
            threads = [_t.Thread(target=core_gemini.gen_json,
                                 args=("claude-sonnet-5", ["x"]))
                       for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        self.assertLessEqual(live["peak"], llm.MAX_CONCURRENCY)
        self.assertEqual(live["now"], 0)   # every slot released


class TestVariantSlugs(unittest.TestCase):
    def setUp(self):
        import job
        self.job = job

    def test_round_trip(self):
        slug = f"gladstone_dog_park{self.job.VARIANT_SEP}claude-sonnet-5"
        self.assertEqual(self.job.variant_base(slug), "gladstone_dog_park")
        self.assertEqual(self.job.variant_model(slug), "claude-sonnet-5")

    def test_plain_slug_is_not_a_variant(self):
        self.assertIsNone(self.job.variant_base("gladstone_dog_park"))
        self.assertIsNone(self.job.variant_model("gladstone_dog_park"))

    def test_unknown_model_tag_is_not_a_variant_model(self):
        self.assertIsNone(self.job.variant_model("x__not-a-model"))

    def test_variant_slug_is_a_valid_path_segment(self):
        from steps.store import is_valid_slug
        for mid in ("claude-sonnet-5", "claude-opus-5"):
            self.assertTrue(is_valid_slug(f"proj{self.job.VARIANT_SEP}{mid}"))

    def test_cannot_fork_a_variant(self):
        with self.assertRaises(ValueError):
            self.job.create_variant("a__claude-sonnet-5", "claude-opus-5")

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            self.job.create_variant("gladstone_dog_park", "gpt-4")

    def test_run_model_prefers_the_pinned_variant_model(self):
        slug = f"p{self.job.VARIANT_SEP}claude-sonnet-5"
        res = {"llm_summary": {"by_model": {"gemini-3.1-pro-preview":
                                            {"calls": 9}}}}
        self.assertEqual(self.job.run_model(slug, res), "claude-sonnet-5")

    def test_run_model_falls_back_to_billed_model(self):
        res = {"llm_summary": {"by_model": {
            "gemini-2.5-flash": {"calls": 1},
            "gemini-3.1-pro-preview": {"calls": 14}}}}
        self.assertEqual(self.job.run_model("p", res), "gemini-3.1-pro-preview")

    def test_run_model_defaults_when_nothing_recorded(self):
        self.assertEqual(self.job.run_model("p", {}), MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
