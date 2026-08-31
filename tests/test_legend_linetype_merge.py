import copy
import unittest

from steps.legend_linetypes.merge import LegendMergeError, merge_entries


class LegendLineTypeMergeTests(unittest.TestCase):
    def test_independent_rows_and_bindings_are_combined(self):
        ordinary = {
            "page": {"sheet": 3, "line_types": 20},
            "line_types": [{"line_type_number": 2,
                            "recognition_source": "method2",
                            "by_run": {"1": {"polylines": [[[1, 1], [2, 2]]]}}}],
            "bindings": [{"key": 4}],
        }
        legend = {
            "page": {"sheet": 3, "legend_samples": 3, "legend_matches": 3},
            "line_types": [{"line_type_number": 16,
                            "base_line_type_number": 16,
                            "recognition_source": "legend_template",
                            "base_recognition_source": "method1",
                            "matched_line_type_numbers": [16],
                            "matched_cluster_sources": {"16": "method1"},
                            "by_run": {"lt16:r1": {"polylines": []}}}],
            "bindings": [{"key": "s0:0"}],
            "samples": [{"sample_index": 0}],
        }
        merged = merge_entries(ordinary, legend)
        self.assertEqual([row["line_type_number"]
                          for row in merged["line_types"]], [2, 16])
        self.assertEqual([row["key"] for row in merged["bindings"]],
                         [4, "s0:0"])
        self.assertEqual(merged["page"]["line_types"], 20)
        self.assertEqual(merged["page"]["legend_samples"], 3)
        self.assertEqual(merged["legend_samples"], [{"sample_index": 0}])

    def test_same_base_number_replaces_duplicate_base_geometry(self):
        ordinary = {"line_types": [{
            "line_type_number": 16, "recognition_source": "method1",
            "by_run": {"1": {"polylines": ["arrow"]}},
        }]}
        legend = {"line_types": [{
            "line_type_number": 16, "base_line_type_number": 16,
            "recognition_source": "legend_template",
            "base_recognition_source": "method1",
            "matched_line_type_numbers": [16, 19],
            "matched_cluster_sources": {
                "16": "method1", "19": "method1"},
            "by_run": {"lt16:r1": {"polylines": ["legend"]},
                       "lt19:r1": {"polylines": ["satellite"]}},
        }]}
        before = copy.deepcopy((ordinary, legend))
        row = merge_entries(ordinary, legend)["line_types"][0]
        self.assertEqual(set(row["by_run"]), {"lt16:r1", "lt19:r1"})
        self.assertEqual(row["replaced_arrow_run_count"], 1)
        self.assertEqual(row["recognition_source"], "legend_template")
        self.assertEqual(row["arrow_recognition_source"], "method1")
        self.assertEqual((ordinary, legend), before)

    def test_source_disagreement_and_spoofed_rows_fail_closed(self):
        ordinary = {"line_types": [{
            "line_type_number": 2, "recognition_source": "method2"}]}
        bad = {"line_types": [{
            "line_type_number": 2, "base_line_type_number": 2,
            "recognition_source": "legend_template",
            "base_recognition_source": "method1"}]}
        with self.assertRaises(LegendMergeError):
            merge_entries(ordinary, bad)
        with self.assertRaises(LegendMergeError):
            merge_entries({}, {"line_types": [{
                "line_type_number": 7,
                "recognition_source": "method1"}]})

    def test_complete_engine_audit_must_identify_every_supervised_cluster(self):
        ordinary = {
            "all_line_types": [
                {"line_type_number": 3, "recognition_source": "method1",
                 "type_uid": "base-three", "line_type_id": "id-three"},
                {"line_type_number": 7, "recognition_source": "method2",
                 "type_uid": "base-seven", "line_type_id": "id-seven"},
            ],
        }
        legend_type = {
            "line_type_number": 3, "base_line_type_number": 3,
            "recognition_source": "legend_template",
            "base_recognition_source": "method1",
            "type_uid": "legend:base-three",
            "line_type_id": "legend:id-three",
            "matched_line_type_numbers": [3, 7],
            "matched_cluster_sources": {"3": "method1", "7": "method2"},
            "by_run": {},
        }
        merged = merge_entries(ordinary, {"line_types": [legend_type]})
        self.assertEqual(merged["line_types"][0]["line_type_number"], 3)
        self.assertEqual(merged["all_line_types"], ordinary["all_line_types"])

        for mutation, message in (
                ({"type_uid": "legend:different"}, "type_uid differs"),
                ({"matched_cluster_sources": {
                    "3": "method1", "7": "method1"}}, "source differs"),
                ({"matched_line_type_numbers": [3, 8],
                  "matched_cluster_sources": {
                      "3": "method1", "8": "method2"}},
                 "missing engine type")):
            with self.subTest(mutation=mutation):
                row = dict(legend_type, **mutation)
                with self.assertRaisesRegex(LegendMergeError, message):
                    merge_entries(ordinary, {"line_types": [row]})


if __name__ == "__main__":
    unittest.main()
