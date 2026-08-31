"""Cache/protocol tests for the independent supervised legend-line channel."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steps import legend_linetypes, store                       # noqa: E402
from steps.legend_linetypes import sidecar                       # noqa: E402


def complete_audit_payload():
    return {
        "page": {
            "base_line_types": 1,
            "page_fingerprint": "page",
            "owned_ops_sha1": "owned",
            "fused_ops_sha1": "fused",
            "path_ops": 2,
            "owned_path_ops": 1,
        },
        "engine_all_line_types": [{
            "line_type_number": 1,
            "signature_family": "motif_periodic",
            "recognition_source": "method1",
            "op_count": 1,
            "ops_sha1": "one",
            "segment_count": 1,
            "pattern_instance_count": 0,
            "pattern_instances": [],
        }],
    }


class SampleTests(unittest.TestCase):
    def test_only_valid_line_symbols_are_normalised_with_original_index(self):
        result = {"symbols": [
            {"category": "shape", "text_index": 0,
             "box_2d": [1, 2, 3, 4], "value": "33"},
            {"category": "line", "text_index": 7,
             "box_2d": (10, 20, 12, 45), "value": "SF",
             "source": "sweep"},
            {"category": "LINE", "text_index": 8,
             "box_2d": [30.5, 40, 32, 80], "value": None},
            {"category": "line", "text_index": True,
             "box_2d": [1, 2, 3, 4]},
            {"category": "line", "text_index": 9,
             "box_2d": [4, 3, 2, 1]},
        ]}
        self.assertEqual(legend_linetypes.samples_of(result), [
            {"symbol_index": 1, "text_index": 7,
             "box_2d": [10.0, 20.0, 12.0, 45.0],
             "value": "SF", "source": "sweep"},
            {"symbol_index": 2, "text_index": 8,
             "box_2d": [30.5, 40.0, 32.0, 80.0],
             "value": "", "source": ""},
        ])

    def test_whole_symbols_entry_is_accepted(self):
        entry = {"sig": "symbol-sig", "result": {"symbols": [{
            "category": "line", "text_index": 2,
            "box_2d": [1, 2, 3, 4], "value": "X", "source": "page",
        }]}}
        self.assertEqual(legend_linetypes.samples_of(entry)[0]["text_index"], 2)


class SignatureTests(unittest.TestCase):
    SAMPLE = [{"symbol_index": 2, "text_index": 7,
               "box_2d": [10.0, 20.0, 12.0, 45.0],
               "value": "SF", "source": "sweep"}]

    def test_every_input_and_producer_identity_changes_signature(self):
        with mock.patch.object(sidecar, "producer_digest", return_value="producer-a"):
            baseline = legend_linetypes.signature("pdf-a", self.SAMPLE)
            self.assertNotEqual(
                baseline, legend_linetypes.signature("pdf-b", self.SAMPLE))
            for key, value in {
                    "symbol_index": 3, "text_index": 8,
                    "box_2d": [10, 20, 13, 45], "value": "CL",
                    "source": "page"}.items():
                changed = [dict(self.SAMPLE[0], **{key: value})]
                self.assertNotEqual(
                    baseline, legend_linetypes.signature("pdf-a", changed), key)
        with mock.patch.object(sidecar, "producer_digest", return_value="producer-b"):
            self.assertNotEqual(
                baseline, legend_linetypes.signature("pdf-a", self.SAMPLE))

    def test_signature_accepts_symbols_result_through_same_normaliser(self):
        result = {"symbols": [{"category": "line", "text_index": 7,
                               "box_2d": [10, 20, 12, 45], "value": "SF",
                               "source": "sweep"}]}
        canonical = [dict(self.SAMPLE[0], symbol_index=0)]
        with mock.patch.object(sidecar, "producer_digest", return_value="p"):
            self.assertEqual(legend_linetypes.signature("pdf", result),
                             legend_linetypes.signature("pdf", canonical))

    def test_duplicate_sample_identity_is_rejected(self):
        duplicate = self.SAMPLE + [dict(self.SAMPLE[0])]
        with self.assertRaisesRegex(ValueError, "duplicate legend symbol_index"):
            legend_linetypes.signature("pdf", duplicate)

    def test_producer_digest_covers_runner_helper_and_shared_engine_deps(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "run_legend.py"
            helper = root / "legend_supervised.py"
            runner.write_text("runner-v1", encoding="utf-8")
            helper.write_text("helper-v1", encoding="utf-8")
            with mock.patch.object(sidecar, "_RUNNER", runner), \
                    mock.patch.object(sidecar, "_HELPER", helper), \
                    mock.patch.object(sidecar._base_sidecar, "engine_digest",
                                      return_value="engine-deps-v1"):
                first = sidecar.producer_digest()
                runner.write_text("runner-v2", encoding="utf-8")
                self.assertNotEqual(first, sidecar.producer_digest())
                second = sidecar.producer_digest()
                helper.write_text("helper-v2", encoding="utf-8")
                self.assertNotEqual(second, sidecar.producer_digest())
                third = sidecar.producer_digest()
                with mock.patch.object(sidecar._base_sidecar, "engine_digest",
                                      return_value="engine-deps-v2"):
                    self.assertNotEqual(third, sidecar.producer_digest())


class CacheTests(unittest.TestCase):
    def test_only_explicit_success_with_current_version_is_current(self):
        good = {"sig": "wanted", "v": legend_linetypes.VERSION,
                "ok": True, "matches": [], **complete_audit_payload()}
        self.assertTrue(legend_linetypes.has_current(good, "wanted"))
        for entry in (
                dict(good, sig="old"),
                dict(good, v=0),
                dict(good, ok=False),
                {"sig": "wanted", "v": legend_linetypes.VERSION,
                 "error": "timeout"},
                None):
            self.assertFalse(legend_linetypes.has_current(entry, "wanted"))

    def test_subset_or_incomplete_engine_audit_cannot_publish_all(self):
        good = {"sig": "wanted", "v": legend_linetypes.VERSION,
                "ok": True, **complete_audit_payload()}
        incomplete = {key: value for key, value in good.items()
                      if key != "engine_all_line_types"}
        self.assertTrue(legend_linetypes.has_current(incomplete, "wanted"))
        self.assertIsNone(
            legend_linetypes.all_audit_entry(incomplete, "wanted"))
        subset = dict(good)
        subset["page"] = dict(good["page"], base_line_types=2)
        self.assertIsNone(
            legend_linetypes.all_audit_entry(subset, "wanted"))
        malformed = dict(good)
        malformed["engine_all_line_types"] = [
            {**good["engine_all_line_types"][0],
             "pattern_instances": None}]
        self.assertIsNone(
            legend_linetypes.all_audit_entry(malformed, "wanted"))

    def test_all_audit_projection_uses_complete_engine_rows(self):
        entry = {"sig": "wanted", "v": legend_linetypes.VERSION,
                 "ok": True, "line_types": [{"line_type_number": 1}],
                 "bindings": [{"key": "s0:0"}],
                 **complete_audit_payload()}
        audit = legend_linetypes.all_audit_entry(entry, "wanted")
        self.assertEqual(audit["sig"], "wanted")
        self.assertEqual(len(audit["all_line_types"]), 1)
        self.assertIsNot(audit["all_line_types"],
                         entry["engine_all_line_types"])
        self.assertIsNone(legend_linetypes.all_audit_entry(entry, "old"))

    def test_page_cache_uses_independent_atomic_pagestore_layout(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(store, "DATA_DIR", Path(tmp)):
            entry = {"sig": "s", "v": 1, "ok": True, "matches": []}
            legend_linetypes.save("fixture", 3, entry)
            path = Path(tmp) / "fixture" / "legend_linetypes" / "3.json"
            self.assertTrue(path.is_file())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), entry)
            self.assertEqual(legend_linetypes.load("fixture", 3), entry)
            self.assertEqual(legend_linetypes.computed_pages("fixture"), [3])


class ComputeTests(unittest.TestCase):
    RESULT = {"symbols": [
        {"category": "shape", "text_index": 0,
         "box_2d": [1, 1, 2, 2]},
        {"category": "line", "text_index": 4,
         "box_2d": [100, 200, 110, 300], "value": "8'",
         "source": "page"},
    ]}

    def test_success_payload_is_wrapped_without_allowing_identity_spoofing(self):
        payload = {"ok": True, "sig": "sidecar-spoof", "v": 999,
                   "matches": [{"symbol_index": 1}],
                   **complete_audit_payload()}
        payload["page"] = dict(payload["page"], sheet=3)
        with mock.patch.object(sidecar, "run_page", return_value=payload) as run:
            entry = legend_linetypes.compute_page(
                "/tmp/input.pdf", 3, self.RESULT, sig="expected",
                cpu_budget=1, timeout=45)
        self.assertEqual(entry["sig"], "expected")
        self.assertEqual(entry["v"], legend_linetypes.VERSION)
        self.assertTrue(entry["ok"])
        run.assert_called_once_with(
            "/tmp/input.pdf", 3,
            [{"symbol_index": 1, "text_index": 4,
              "box_2d": [100.0, 200.0, 110.0, 300.0],
              "value": "8'", "source": "page"}],
            cpu_budget=1, timeout=45, dbg=None)

    def test_failure_or_malformed_success_is_never_returned_as_cache(self):
        with mock.patch.object(sidecar, "run_page",
                              side_effect=RuntimeError("engine crash")):
            with self.assertRaisesRegex(RuntimeError, "engine crash"):
                legend_linetypes.compute_page(
                    "/tmp/input.pdf", 3, self.RESULT, sig="s")
        with mock.patch.object(
                sidecar, "run_page", return_value={"ok": True,
                                                    "line_types": []}):
            with self.assertRaisesRegex(RuntimeError,
                                        "no complete engine audit"):
                legend_linetypes.compute_page(
                    "/tmp/input.pdf", 3, self.RESULT, sig="s")

    def test_compute_accepts_the_same_canonical_sample_list_that_was_signed(self):
        samples = legend_linetypes.samples_of(self.RESULT)
        with mock.patch.object(sidecar, "run_page",
                              return_value={"ok": True, "matches": [],
                                            **complete_audit_payload()}) as run:
            entry = legend_linetypes.compute_page(
                "/tmp/input.pdf", 3, samples, sig="sample-sig")
        self.assertTrue(legend_linetypes.has_current(entry, "sample-sig"))
        self.assertEqual(run.call_args.args[2], samples)
        with mock.patch.object(sidecar, "run_page",
                              return_value={"ok": False, "error": "nope"}):
            with self.assertRaisesRegex(RuntimeError, "no successful payload"):
                legend_linetypes.compute_page(
                    "/tmp/input.pdf", 3, self.RESULT, sig="s")

    def test_no_samples_is_not_a_successful_empty_cache(self):
        with mock.patch.object(sidecar, "run_page") as run:
            with self.assertRaisesRegex(ValueError, "no legend line samples"):
                legend_linetypes.compute_page(
                    "/tmp/input.pdf", 3, {"symbols": []}, sig="s")
        run.assert_not_called()


class SidecarProtocolTests(unittest.TestCase):
    def test_exact_wire_payload_and_shared_process_runner(self):
        sample = {"symbol_index": 1, "text_index": 4,
                  "box_2d": [1, 2, 3, 4], "value": "SF",
                  "source": "page"}
        with mock.patch.object(sidecar, "sidecar_available", return_value=True), \
                mock.patch.object(sidecar._base_sidecar, "_run_job",
                                  return_value={"ok": True}) as run:
            result = sidecar.run_page(
                "/tmp/in.pdf", 9, [sample], cpu_budget=1, timeout=77)
        self.assertEqual(result, {"ok": True})
        args, kwargs = run.call_args
        self.assertEqual(args[0], sidecar._RUNNER)
        self.assertEqual(args[1], {
            "pdf": "/tmp/in.pdf", "sheet": 9,
            "samples": [{"symbol_index": 1, "text_index": 4,
                         "box_2d": [1.0, 2.0, 3.0, 4.0],
                         "value": "SF", "source": "page"}],
            "cpu_budget": 1,
        })
        self.assertEqual(kwargs, {"sheet": 9, "timeout": 77, "dbg": None,
                                  "label": "legend line-type sidecar"})

    def test_sheet_and_empty_samples_are_rejected_before_spawn(self):
        with mock.patch.object(sidecar, "sidecar_available", return_value=True), \
                mock.patch.object(sidecar._base_sidecar, "_run_job") as run:
            with self.assertRaisesRegex(ValueError, "1-based int"):
                sidecar.run_page("x.pdf", 0, [{}])
            with self.assertRaisesRegex(ValueError, "no legend line samples"):
                sidecar.run_page("x.pdf", 1, [])
        run.assert_not_called()

    def test_multiworker_argument_and_environment_are_rejected(self):
        sample = {"symbol_index": 1, "text_index": 4,
                  "box_2d": [1, 2, 3, 4], "value": "SF",
                  "source": "page"}
        with mock.patch.object(sidecar, "sidecar_available", return_value=True), \
                mock.patch.object(sidecar._base_sidecar, "_run_job") as run:
            with self.assertRaisesRegex(ValueError, "must be 1"):
                sidecar.run_page("x.pdf", 1, [sample], cpu_budget=2)
            with mock.patch.dict(
                    os.environ, {"LEGEND_LINETYPE_CPU_BUDGET": "6"}):
                with self.assertRaisesRegex(ValueError, "must be 1"):
                    sidecar.run_page("x.pdf", 1, [sample])
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
