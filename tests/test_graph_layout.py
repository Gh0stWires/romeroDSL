from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from romerodsl.compiler import _place_rooms, compile_to_plan, compile_to_wad
from romerodsl.metrics import structural_metrics
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "phase2_graph_layout.json"


class GraphLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.document = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_graph_layout_preserves_branches_loop_and_required_gate(self) -> None:
        validation = validate_document(self.document)
        self.assertTrue(validation.valid, validation.errors)
        plan = compile_to_plan(self.document)
        boxes = _place_rooms(
            self.document["spaces"],
            self.document["progression"],
            self.document["connections"],
            layout_seed=self.document["layout_seed"],
        )
        self.assertEqual(len(boxes), 8)
        self.assertEqual(plan["layout"]["strategy"], "graph_bfs_layers")
        self.assertEqual(plan["layout"]["overlap_count"], 0)
        self.assertGreater(plan["layout"]["connection_orientations"]["horizontal"], 0)
        self.assertGreater(plan["layout"]["connection_orientations"]["vertical"], 0)
        self.assertEqual(structural_metrics(self.document)["graph"]["cycle_count"], 1)
        self.assertGreaterEqual(structural_metrics(self.document)["graph"]["branch_nodes"], 1)
        _assert_no_room_overlap(self, boxes)

    def test_same_seed_is_reproducible_and_seed_changes_layout_orientation(self) -> None:
        first = compile_to_plan(self.document)
        second = compile_to_plan(copy.deepcopy(self.document))
        self.assertEqual(first["layout"], second["layout"])
        self.assertEqual(first["rooms"], second["rooms"])

        alternate = copy.deepcopy(self.document)
        alternate["layout_seed"] = self.document["layout_seed"] + 1
        alternate_plan = compile_to_plan(alternate)
        self.assertNotEqual(first["rooms"], alternate_plan["rooms"])
        self.assertEqual(alternate_plan["layout"]["seed"], alternate["layout_seed"])

    def test_invalid_layout_seed_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.document)
        invalid["layout_seed"] = "17"
        validation = validate_document(invalid)
        self.assertFalse(validation.valid)
        self.assertIn("layout_seed must be an integer", " ".join(validation.errors))

    def test_graph_layout_exports_valid_keyed_wad(self) -> None:
        output = Path(__file__).resolve().parents[1] / "build" / "test_phase2_graph_layout.wad"
        report = compile_to_wad(self.document, output)
        reread = validate_wad(output)
        self.assertTrue(report["valid"], report)
        self.assertTrue(reread["valid"], reread)
        self.assertEqual(report["room_sectors"], 8)
        self.assertEqual(reread["keys"], 1)
        self.assertEqual(reread["locked_doors"], 1)
        self.assertEqual(reread["locked_door_colors"], {"blue": 1})
        self.assertTrue(reread["key_lock_pairs_valid"])
        self.assertEqual(reread["invalid_side_references"], [])
        self.assertEqual(reread["unreferenced_sidedefs"], [])


def _assert_no_room_overlap(test: unittest.TestCase, boxes: dict[str, dict[str, int]]) -> None:
    ids = sorted(boxes)
    for index, left_id in enumerate(ids):
        left = boxes[left_id]
        for right_id in ids[index + 1 :]:
            right = boxes[right_id]
            overlap = not (
                left["c"] + left["w"] + 1 <= right["c"]
                or right["c"] + right["w"] + 1 <= left["c"]
                or left["r"] + left["h"] + 1 <= right["r"]
                or right["r"] + right["h"] + 1 <= left["r"]
            )
            test.assertFalse(overlap, f"rooms overlap: {left_id}, {right_id}")


if __name__ == "__main__":
    unittest.main()
