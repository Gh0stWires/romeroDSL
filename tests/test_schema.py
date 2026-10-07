from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.compiler import compile_to_plan
from romerodsl.schema import validate_document

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


if __name__ == "__main__":
    unittest.main()
