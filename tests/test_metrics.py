from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.metrics import structural_metrics
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "complex_eight_room_bfg.json"


class StructuralMetricsTests(unittest.TestCase):
    def test_metrics_capture_graph_shapes_progression_and_height(self) -> None:
        document = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        validation = validate_document(document)
        plan = compile_to_plan(document)
        wad_path = ROOT / "build" / "test_structural_metrics.wad"
        compile_to_wad(document, wad_path)
        wad_report = validate_wad(wad_path)

        report = structural_metrics(
            document,
            prompt="Create a big eight-room Doom map with blue and red keys.",
            plan=plan,
            wad_report=wad_report,
            dsl_valid=validation.valid,
        )

        self.assertEqual(report["space_count"], 8)
        self.assertGreaterEqual(report["unique_room_shape_count"], 4)
        self.assertEqual(report["graph"]["nodes"], 8)
        self.assertEqual(report["graph"]["branch_nodes"], 0)
        self.assertGreaterEqual(report["graph"]["cycle_count"], 0)
        self.assertEqual(report["progression"]["key_colors"], ["blue", "red"])
        self.assertEqual(report["progression"]["missing_key_for_gate_colors"], [])
        self.assertTrue(report["height"]["varied"])
        self.assertEqual(report["geometry"]["sectors"], wad_report["sectors"])
        self.assertTrue(report["validation"]["dsl_valid"])
        self.assertTrue(report["validation"]["wad_valid"])
        self.assertTrue(report["prompt"]["space_count_exact_match"])

    def test_metrics_tolerate_partial_failed_generation(self) -> None:
        report = structural_metrics(
            {"spaces": [{"id": "start", "shape": "rectangle"}], "connections": []},
            prompt="make an eight-room Doom map with a red key",
            dsl_valid=False,
            raw_output='{"spaces": [',
            output_token_count=32,
            max_new_tokens=32,
        )

        self.assertEqual(report["space_count"], 1)
        self.assertEqual(report["graph"]["connected_components"], 1)
        self.assertTrue(report["output"]["generation_truncated"])
        self.assertEqual(report["prompt"]["missing_requested_keys"], ["red"])
        self.assertFalse(report["validation"]["dsl_valid"])


if __name__ == "__main__":
    unittest.main()
