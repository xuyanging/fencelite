"""Scope regressions for Method1's inline-feet compound ownership guard."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))

from line_type_engine.method1 import postprocess_text  # noqa: E402


class InlineFeetCompoundGuardTests(unittest.TestCase):
    def test_short_feet_tokens_are_the_only_protected_labels(self) -> None:
        for value in ("8'", "8.10\u2032", " 12 \u2019 "):
            with self.subTest(value=value):
                self.assertTrue(postprocess_text._is_inline_feet_label(value))

        for value in ("8", "8'-0\"", "GRID 8'", "FM", "SIDELINE FENCE"):
            with self.subTest(value=value):
                self.assertFalse(postprocess_text._is_inline_feet_label(value))


if __name__ == "__main__":
    unittest.main()
