from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.compiler import compile_to_layout, compile_to_plan, compile_to_wad
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "blue_lock_processing.json"


def load_example() -> dict:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


class SchemaTests(unittest.TestCase):
    def test_example_is_valid(self) -> None:
        report = validate_document(load_example())
        self.assertTrue(report.valid, report.errors)

    def test_missing_matching_key_fails(self) -> None:
        document = load_example()
        document["progression"]["keys"] = []

        report = validate_document(document)

        self.assertFalse(report.valid)
        self.assertTrue(any("no matching key exists" in error for error in report.errors))

    def test_required_key_in_secret_fails(self) -> None:
        document = load_example()
        document["progression"]["keys"][0]["location"] = "armor_secret"

        report = validate_document(document)

        self.assertFalse(report.valid)
        self.assertTrue(any("secret space" in error for error in report.errors))

    def test_compile_to_plan_preserves_progression_gate(self) -> None:
        plan = compile_to_plan(load_example())

        locked_edges = [edge for edge in plan["edges"] if edge["type"] == "locked_door"]
        self.assertEqual(
            locked_edges,
            [
                {
                    "id": "blue_locked_door",
                    "from": "central_hub",
                    "to": "exit_wing",
                    "type": "locked_door",
                    "key": "blue",
                    "required_for_exit": True,
                }
            ],
        )

    def test_compile_to_layout_keeps_semantic_progression(self) -> None:
        layout = compile_to_layout(load_example())
        flattened = [token for row in layout["grid"] for token in row]

        self.assertEqual(flattened.count("start"), 1)
        self.assertEqual(flattened.count("exit"), 1)
        self.assertEqual(flattened.count("key"), 1)
        self.assertGreaterEqual(flattened.count("door"), 1)
        self.assertIn("blue", layout["key_colors"].values())
        self.assertIn("blue", layout["door_locks"].values())
        self.assertIn(192, {height for row in layout["floor_heights"] for height in row})
        self.assertIn(-24, {height for row in layout["floor_heights"] for height in row})

    def test_compile_to_wad_writes_valid_udmf_pwad(self) -> None:
        output = Path(__file__).resolve().parents[1] / "build" / "test_blue_lock_processing.wad"
        report = compile_to_wad(load_example(), output)
        reread = validate_wad(output)

        self.assertTrue(output.exists())
        self.assertTrue(report["valid"])
        self.assertEqual(report["geometry_profile"], "sector_primitives_v0.3")
        self.assertEqual(reread["player_starts"], 1)
        self.assertEqual(reread["keys"], 1)
        self.assertEqual(reread["exits"], 1)
        self.assertGreaterEqual(reread["locked_doors"], 1)
        self.assertEqual(reread["locked_door_linedefs"], reread["bidirectional_door_linedefs"])
        self.assertGreaterEqual(reread["closed_door_sectors"], 1)
        self.assertGreater(reread["monsters"], 0)

    def test_sector_primitive_compiler_uses_rooms_and_height_feature_sectors(self) -> None:
        output = Path(__file__).resolve().parents[1] / "build" / "test_sector_primitives.wad"
        report = compile_to_wad(load_example(), output)

        self.assertEqual(report["room_sectors"], 7)
        self.assertEqual(report["door_sectors"], 1)
        self.assertEqual(report["height_feature_sectors"], 6)
        self.assertLess(report["sectors"], 60)
        self.assertIn(-24, report["floor_height_levels"])
        self.assertIn(32, report["floor_height_levels"])
        self.assertIn(192, report["floor_height_levels"])


if __name__ == "__main__":
    unittest.main()
