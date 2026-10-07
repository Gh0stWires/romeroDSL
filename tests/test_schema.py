from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.compiler import _place_rooms, compile_to_layout, compile_to_plan, compile_to_wad
from romerodsl.geometry import build_geometry_textmap
from romerodsl.schema import validate_document
from romerodsl.wad import _parse_textmap, validate_wad

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

    def test_one_sided_linedefs_face_their_own_sector(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        vertices = groups["vertex"]
        sidedefs = groups["sidedef"]
        linedefs = groups["linedef"]
        sector_points: dict[int, list[tuple[int, int]]] = {index: [] for index, _sector in enumerate(groups["sector"])}
        for line in linedefs:
            for side_key in ("sidefront", "sideback"):
                side_id = line.get(side_key, -1)
                if side_id < 0:
                    continue
                sector_id = sidedefs[side_id]["sector"]
                sector_points[sector_id].append((vertices[line["v1"]]["x"], vertices[line["v1"]]["y"]))
                sector_points[sector_id].append((vertices[line["v2"]]["x"], vertices[line["v2"]]["y"]))
        centroids = {
            sector_id: (
                sum(x for x, _y in points) / len(points),
                sum(y for _x, y in points) / len(points),
            )
            for sector_id, points in sector_points.items()
            if points
        }

        wrong_facing = []
        for index, line in enumerate(linedefs):
            if line.get("sideback", -1) >= 0:
                continue
            sector_id = sidedefs[line["sidefront"]]["sector"]
            centroid_x, centroid_y = centroids[sector_id]
            v1 = vertices[line["v1"]]
            v2 = vertices[line["v2"]]
            dx = v2["x"] - v1["x"]
            dy = v2["y"] - v1["y"]
            px = centroid_x - v1["x"]
            py = centroid_y - v1["y"]
            cross = dx * py - dy * px
            # Doom renders the front sidedef on the right side of a linedef.
            if cross >= 0:
                wrong_facing.append(index)

        self.assertEqual(wrong_facing, [])

    def test_sector_primitive_two_sided_lines_have_upper_and_lower_textures(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        missing = []
        for index, line in enumerate(groups["linedef"]):
            if line.get("sideback", -1) < 0:
                continue
            for side_key in ("sidefront", "sideback"):
                side = groups["sidedef"][line[side_key]]
                if side.get("texturetop") == "-" or side.get("texturebottom") == "-":
                    missing.append((index, side_key, side.get("texturetop"), side.get("texturebottom")))

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
