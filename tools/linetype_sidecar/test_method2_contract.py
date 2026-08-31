from __future__ import annotations

import unittest
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from line_type_engine.method2.contract import (  # noqa: E402
    FROZEN_TS_METHOD2_CONFIG_HASH,
    FROZEN_TS_METHOD2_FEATURES,
    LINE_TYPE_METHOD2_FEATURES,
    frozen_ts_method2_config_hash,
)


class Method2ContractTests(unittest.TestCase):
    def test_frozen_ts_features_remain_distinct_from_candidate_features(self) -> None:
        candidate_only = {
            "native_inline_feet_pattern_clustering",
            "measurement_internal_carrier_ownership",
        }

        self.assertEqual(len(FROZEN_TS_METHOD2_FEATURES), 12)
        self.assertEqual(len(LINE_TYPE_METHOD2_FEATURES), 14)
        self.assertTrue(candidate_only.isdisjoint(FROZEN_TS_METHOD2_FEATURES))
        self.assertTrue(candidate_only.issubset(LINE_TYPE_METHOD2_FEATURES))

    def test_frozen_ts_config_hash_is_unchanged(self) -> None:
        self.assertEqual(FROZEN_TS_METHOD2_CONFIG_HASH, "7211c00a7a782be7")
        self.assertEqual(
            frozen_ts_method2_config_hash(),
            FROZEN_TS_METHOD2_CONFIG_HASH,
        )


if __name__ == "__main__":
    unittest.main()
