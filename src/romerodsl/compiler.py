"""Abstract compiler for romeroDSL v0.1.

This module deliberately stops before WAD/UDMF export. v0.1 converts a semantic
map contract into a deterministic geometry plan that later compiler stages can
turn into real sectors, linedefs, sidedefs, and things.
"""

from __future__ import annotations

from typing import Any

from romerodsl.schema import validate_document


def compile_to_plan(document: dict[str, Any]) -> dict[str, Any]:
    """Compile a valid romeroDSL document to an abstract geometry plan."""

    report = validate_document(document)
    report.raise_for_errors()

    spaces = document["spaces"]
    connections = document["connections"]

    room_boxes = _place_rooms(spaces)
    return {
        "compiler_version": "0.1",
        "source_title": document.get("title"),
        "target": "abstract_geometry_plan",
        "units": "doom_map_units",
        "rooms": [
            {
                "id": space["id"],
                "role": space.get("role", "unspecified"),
                "box": room_boxes[space["id"]],
                "floor_height": space.get("floor_height", 0),
                "ceiling_height": space.get("ceiling_height", 128),
                "height_features": space.get("height_topology", {}).get("features", []),
            }
            for space in spaces
        ],
        "edges": [
            {
                "id": connection.get("id") or f"{connection['from']}->{connection['to']}",
                "from": connection["from"],
                "to": connection["to"],
                "type": connection.get("type", "doorway"),
                "key": connection.get("key"),
                "required_for_exit": bool(connection.get("required_for_exit", False)),
            }
            for connection in connections
        ],
        "progression": document["progression"],
        "theme": document["theme"],
        "notes": [
            "This is an abstract plan, not a WAD.",
            "The model-authored DSL remains the source of truth for progression.",
        ],
    }


def _place_rooms(spaces: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Place rooms on a simple deterministic row for early compiler testing."""

    boxes: dict[str, dict[str, int]] = {}
    cursor_x = 0
    gap = 192
    for space in spaces:
        width, height = _size_to_dimensions(str(space.get("size", "medium")))
        boxes[space["id"]] = {
            "x": cursor_x,
            "y": 0,
            "width": width,
            "height": height,
        }
        cursor_x += width + gap
    return boxes


def _size_to_dimensions(size: str) -> tuple[int, int]:
    match size:
        case "tiny":
            return 256, 256
        case "small":
            return 384, 384
        case "medium":
            return 512, 512
        case "large":
            return 768, 640
        case "huge":
            return 1024, 768
        case _:
            return 512, 512
