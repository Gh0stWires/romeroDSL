from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.compiler import _place_rooms, compile_to_wad
from romerodsl.geometry import _box_to_rect, _connection_portals, _room_boundary_points
from romerodsl.schema import canonical_orientation, canonical_room_shape, validate_document
from romerodsl.wad import _read_textmap, validate_wad


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "blue_lock_processing.json"


def _point_on_boundary(point: tuple[int, int], boundary: list[tuple[int, int]]) -> bool:
    px, py = point
    for (ax, ay), (bx, by) in zip(boundary, boundary[1:] + boundary[:1]):
        cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
        if cross == 0 and min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by):
            return True
    return False


class GeometryShapeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = json.loads(EXAMPLE.read_text(encoding="utf-8"))

    def test_supported_shapes_export_valid_wads(self) -> None:
        shapes = (
            "rectangle_with_side_alcoves",
            "l_shaped",
            "t_shaped",
            "irregular_orthogonal",
            "inset_chamber",
            "raised_center",
            "depressed_floor",
            "irregular_octagonal",
            "bent_corridor",
            "narrow_hall_with_cover",
            "closet",
        )
        textmaps: set[str] = set()
        with tempfile.TemporaryDirectory() as temporary:
            for shape in shapes:
                document = copy.deepcopy(self.source)
                document["spaces"][1]["shape"] = shape
                document["spaces"][1]["orientation"] = "east" if shape in {"l_shaped", "t_shaped"} else 0
                schema = validate_document(document)
                self.assertTrue(schema.valid, schema.errors)
                output = Path(temporary) / f"{shape}.wad"
                report = compile_to_wad(document, output)
                reread = validate_wad(output)
                self.assertTrue(report["valid"], report)
                self.assertTrue(reread["valid"], reread)
                textmaps.add(_read_textmap(output))

        self.assertGreaterEqual(len(textmaps), 5)

    def test_phase2_shapes_have_distinct_physical_boundaries(self) -> None:
        shapes = (
            "rectangle_with_side_alcoves",
            "closet",
            "bent_corridor",
            "narrow_hall_with_cover",
        )
        signatures = {
            shape: tuple(_room_boundary_points((0, 0, 1024, 1024), shape))
            for shape in shapes
        }
        self.assertEqual(len(set(signatures.values())), len(shapes))
        self.assertNotEqual(len(signatures["closet"]), len(signatures["bent_corridor"]))

    def test_narrow_hall_with_cover_emits_cover_geometry(self) -> None:
        document = copy.deepcopy(self.source)
        document["spaces"][1]["shape"] = "narrow_hall_with_cover"
        document["spaces"][1]["height_topology"] = {"features": []}
        with tempfile.TemporaryDirectory() as temporary:
            report = compile_to_wad(document, Path(temporary) / "cover.wad")
        self.assertGreaterEqual(report["height_feature_sectors"], 2)

    def test_height_shapes_emit_a_central_feature_sector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for shape, expected_delta in (("raised_center", 32), ("depressed_floor", -24)):
                document = copy.deepcopy(self.source)
                document["spaces"][1]["shape"] = shape
                output = Path(temporary) / f"{shape}.wad"
                report = compile_to_wad(document, output)
                self.assertTrue(report["valid"], report)
                self.assertIn(expected_delta, report["floor_height_levels"])
                self.assertGreaterEqual(report["height_feature_sectors"], 9)

    def test_connections_land_on_rotated_shape_boundaries(self) -> None:
        document = copy.deepcopy(self.source)
        document["spaces"][1]["shape"] = "l_shaped"
        document["spaces"][1]["orientation"] = "east"
        document["spaces"][2]["shape"] = "t_shaped"
        document["spaces"][2]["orientation"] = 180
        boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
        spaces = {space["id"]: space for space in document["spaces"]}
        rects = {space_id: _box_to_rect(box, 128) for space_id, box in boxes.items()}
        for connection in document["connections"]:
            source = str(connection["from"])
            target = str(connection["to"])
            _source_side, _target_side, source_point, target_point, _source_opening, _target_opening = _connection_portals(
                rects[source],
                rects[target],
                source_shape=canonical_room_shape(spaces[source].get("shape", "rectangle")) or "rectangle",
                target_shape=canonical_room_shape(spaces[target].get("shape", "rectangle")) or "rectangle",
                source_orientation=canonical_orientation(spaces[source].get("orientation", 0)) or 0,
                target_orientation=canonical_orientation(spaces[target].get("orientation", 0)) or 0,
            )
            self.assertTrue(_point_on_boundary(source_point, _room_boundary_points(rects[source], spaces[source].get("shape", "rectangle"), canonical_orientation(spaces[source].get("orientation", 0)) or 0)))
            self.assertTrue(
                _point_on_boundary(
                    target_point,
                    _room_boundary_points(
                        rects[target],
                        spaces[target].get("shape", "rectangle"),
                        canonical_orientation(spaces[target].get("orientation", 0)) or 0,
                    ),
                )
            )

    def test_unsupported_shape_is_rejected_instead_of_becoming_rectangle(self) -> None:
        document = copy.deepcopy(self.source)
        document["spaces"][1]["shape"] = "circle"
        report = validate_document(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("unsupported shape" in error for error in report.errors))

    def test_unsupported_orientation_is_rejected(self) -> None:
        document = copy.deepcopy(self.source)
        document["spaces"][1]["orientation"] = 45
        report = validate_document(document)
        self.assertFalse(report.valid)
        self.assertTrue(any("unsupported orientation" in error for error in report.errors))


if __name__ == "__main__":
    unittest.main()
