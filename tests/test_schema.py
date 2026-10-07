from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.compiler import _place_rooms, compile_to_layout, compile_to_plan, compile_to_wad
from romerodsl.geometry import _box_to_rect, _pillar_rects, build_geometry_textmap
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

    def test_invalid_space_material_fails(self) -> None:
        document = load_example()
        document["spaces"][0]["materials"] = {"wall": "NOT_A_TEXTURE"}

        report = validate_document(document)

        self.assertFalse(report.valid)
        self.assertTrue(any("invalid wall material" in error for error in report.errors))

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

    def test_compile_to_layout_preserves_material_layer(self) -> None:
        document = load_example()
        document["spaces"][0]["materials"] = {
            "floor": "FLOOR0_1",
            "ceiling": "CEIL3_5",
            "wall": "STARTAN2",
        }

        layout = compile_to_layout(document)
        start_box = layout["rooms"]["start_room"]
        cell = (start_box["r"] + 1, start_box["c"] + 1)

        self.assertEqual(layout["surface_materials"]["floor"][cell[0]][cell[1]], "FLOOR0_1")
        self.assertEqual(layout["surface_materials"]["ceiling"][cell[0]][cell[1]], "CEIL3_5")
        self.assertEqual(layout["surface_materials"]["wall"][cell[0]][cell[1]], "STARTAN2")

    def test_compile_to_wad_writes_valid_udmf_pwad(self) -> None:
        output = Path(__file__).resolve().parents[1] / "build" / "test_blue_lock_processing.wad"
        report = compile_to_wad(load_example(), output)
        reread = validate_wad(output)

        self.assertTrue(output.exists())
        self.assertTrue(report["valid"])
        self.assertEqual(report["geometry_profile"], "sector_primitives_v0.5")
        self.assertEqual(reread["player_starts"], 1)
        self.assertEqual(reread["keys"], 1)
        self.assertEqual(reread["exits"], 1)
        self.assertGreaterEqual(reread["locked_doors"], 1)
        self.assertGreaterEqual(reread["bidirectional_door_linedefs"], reread["locked_door_linedefs"])
        self.assertGreaterEqual(reread["closed_door_sectors"], 1)
        self.assertEqual(reread["invalid_side_references"], [])
        self.assertEqual(reread["unreferenced_sidedefs"], [])
        self.assertEqual(reread["missing_visible_textures"], 0)
        self.assertEqual(reread["unknown_flats"], [])
        self.assertEqual(reread["unknown_textures"], [])
        self.assertGreater(reread["monsters"], 0)

    def test_sector_primitive_compiler_uses_rooms_and_height_feature_sectors(self) -> None:
        output = Path(__file__).resolve().parents[1] / "build" / "test_sector_primitives.wad"
        report = compile_to_wad(load_example(), output)

        self.assertEqual(report["room_sectors"], 7)
        self.assertEqual(report["door_sectors"], 2)
        self.assertEqual(report["height_feature_sectors"], 8)
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

    def test_two_sided_linedef_sidedefs_face_their_assigned_sectors(self) -> None:
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
            if line.get("sideback", -1) < 0:
                continue
            v1 = vertices[line["v1"]]
            v2 = vertices[line["v2"]]
            dx = v2["x"] - v1["x"]
            dy = v2["y"] - v1["y"]
            sector_id = sidedefs[line["sidefront"]]["sector"]
            centroid_x, centroid_y = centroids[sector_id]
            cross = dx * (centroid_y - v1["y"]) - dy * (centroid_x - v1["x"])
            if abs(cross) < 1e-6:
                continue
            if cross >= 0:
                wrong_facing.append((index, "sidefront", sector_id))

        self.assertEqual(wrong_facing, [])

    def test_sector_primitive_linedefs_do_not_cross_without_vertices(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        vertices = groups["vertex"]
        linedefs = groups["linedef"]

        def segment(index: int) -> tuple[int, int, int, int]:
            line = linedefs[index]
            v1 = vertices[line["v1"]]
            v2 = vertices[line["v2"]]
            return v1["x"], v1["y"], v2["x"], v2["y"]

        def strictly_between(value: int, a: int, b: int) -> bool:
            return min(a, b) < value < max(a, b)

        crossings = []
        for left in range(len(linedefs)):
            x1, y1, x2, y2 = segment(left)
            for right in range(left + 1, len(linedefs)):
                x3, y3, x4, y4 = segment(right)
                if x1 == x2 and y3 == y4:
                    if strictly_between(x1, x3, x4) and strictly_between(y3, y1, y2):
                        crossings.append((left, right, x1, y3))
                elif y1 == y2 and x3 == x4:
                    if strictly_between(x3, x1, x2) and strictly_between(y1, y3, y4):
                        crossings.append((left, right, x3, y1))

        self.assertEqual(crossings, [])

    def test_exported_linedefs_reference_valid_used_sidedefs(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        sidedefs = groups["sidedef"]
        invalid_refs = []
        used_sidedefs = set()
        for line_index, line in enumerate(groups["linedef"]):
            for side_key in ("sidefront", "sideback"):
                side_id = line.get(side_key, -1)
                if side_key == "sideback" and side_id < 0:
                    continue
                if not 0 <= side_id < len(sidedefs):
                    invalid_refs.append((line_index, side_key, side_id))
                    continue
                used_sidedefs.add(side_id)

        unreferenced_sidedefs = sorted(set(range(len(sidedefs))) - used_sidedefs)

        self.assertEqual(invalid_refs, [])
        self.assertEqual(unreferenced_sidedefs, [])

    def test_irregular_octagonal_rooms_emit_diagonal_linedefs(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        vertices = groups["vertex"]
        diagonal_linedefs = []
        for index, line in enumerate(groups["linedef"]):
            v1 = vertices[line["v1"]]
            v2 = vertices[line["v2"]]
            if v1["x"] != v2["x"] and v1["y"] != v2["y"]:
                diagonal_linedefs.append(index)

        self.assertGreaterEqual(len(diagonal_linedefs), 4)

    def test_stair_access_emits_intermediate_height_sectors(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)
        heights = sorted({sector["heightfloor"] for sector in groups["sector"]})

        self.assertIn(16, heights)
        self.assertIn(-12, heights)

    def test_stair_connected_floor_deltas_are_doom_step_safe(self) -> None:
        document = load_example()
        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)

        sectors = groups["sector"]
        sidedefs = groups["sidedef"]
        graph: dict[int, set[int]] = {index: set() for index, _sector in enumerate(sectors)}
        for index, line in enumerate(groups["linedef"]):
            if line.get("sideback", -1) < 0:
                continue
            front = sidedefs[line["sidefront"]]["sector"]
            back = sidedefs[line["sideback"]]["sector"]
            delta = abs(sectors[front]["heightfloor"] - sectors[back]["heightfloor"])
            if delta <= 24:
                graph[front].add(back)
                graph[back].add(front)

        reachable = set()
        frontier = [index for index, sector in enumerate(sectors) if sector["heightfloor"] == 0]
        while frontier:
            sector_id = frontier.pop()
            if sector_id in reachable:
                continue
            reachable.add(sector_id)
            frontier.extend(sorted(graph[sector_id] - reachable))

        reachable_heights = {sectors[index]["heightfloor"] for index in reachable}
        self.assertIn(32, reachable_heights)
        self.assertIn(-24, reachable_heights)

    def test_thing_placement_avoids_blocking_height_features(self) -> None:
        document = load_example()
        start_room = document["spaces"][0]
        start_room["height_topology"] = {
            "features": [
                {
                    "id": "start_room_blockers",
                    "type": "pillar_cluster",
                    "count": 5,
                    "floor_height": 128,
                    "ceiling_height": 128,
                    "blocks_movement": True,
                }
            ]
        }
        start_room["things"]["items"] = ["shotgun"] + ["shells_small"] * 24

        room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        textmap, _stats = build_geometry_textmap(document, room_boxes)
        groups = _parse_textmap(textmap)
        start_rect = _box_to_rect(room_boxes["start_room"], 128)
        blockers = _pillar_rects(start_rect, 5)

        blocked_things = []
        for thing in groups["thing"]:
            point = (thing["x"], thing["y"])
            if any(x1 <= point[0] <= x2 and y1 <= point[1] <= y2 for x1, y1, x2, y2 in blockers):
                blocked_things.append((thing["type"], point))

        self.assertEqual(blocked_things, [])


if __name__ == "__main__":
    unittest.main()
