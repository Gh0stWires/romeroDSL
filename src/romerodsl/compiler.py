"""Compiler for romeroDSL v0.1.

The compiler bridges model-authored semantic Doom design contracts toward real
geometry. v0.1 still keeps the geometry deliberately simple: rooms become cell
rectangles, connections become corridors, and WAD export uses one UDMF sector per
cell. That is enough to prove the new semantic path can preserve progression,
keys, locks, height features, and monsters without asking a raster model to
invent them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from romerodsl.geometry import write_geometry_wad
from romerodsl.schema import validate_document
from romerodsl.wad import (
    AMMO,
    DOOR,
    EXIT,
    FLOOR,
    HEALTH,
    KEY,
    MONSTER,
    START,
    WEAPON,
)

Cell = tuple[int, int]
Box = dict[str, int]


def compile_to_plan(document: dict[str, Any]) -> dict[str, Any]:
    """Compile a valid romeroDSL document to an abstract geometry plan."""

    report = validate_document(document)
    report.raise_for_errors()

    spaces = document["spaces"]
    connections = document["connections"]
    room_boxes = _place_rooms(spaces, document["progression"], connections)
    return {
        "compiler_version": "0.2",
        "source_title": document.get("title"),
        "target": "abstract_geometry_plan",
        "units": "doom_map_units",
        "rooms": [
            {
                "id": space["id"],
                "role": space.get("role", "unspecified"),
                "box": _grid_box_to_map_units(room_boxes[space["id"]]),
                "grid_box": room_boxes[space["id"]],
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


def compile_to_layout(document: dict[str, Any]) -> dict[str, Any]:
    """Compile romeroDSL into a cell-level semantic layout for WAD export."""

    report = validate_document(document)
    report.raise_for_errors()

    spaces = {space["id"]: space for space in document["spaces"]}
    room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
    height = max(box["r"] + box["h"] for box in room_boxes.values()) + 3
    width = max(box["c"] + box["w"] for box in room_boxes.values()) + 3
    grid = [["empty" for _ in range(width)] for _ in range(height)]
    floor_heights = [[0 for _ in range(width)] for _ in range(height)]
    key_colors: dict[str, str] = {}
    door_locks: dict[str, str] = {}
    room_centers: dict[str, Cell] = {}

    for space_id, box in room_boxes.items():
        space = spaces[space_id]
        floor_height = int(space.get("floor_height", 0))
        for r in range(box["r"], box["r"] + box["h"]):
            for c in range(box["c"], box["c"] + box["w"]):
                grid[r][c] = FLOOR
                floor_heights[r][c] = floor_height
        room_centers[space_id] = (box["r"] + box["h"] // 2, box["c"] + box["w"] // 2)
        _apply_height_features(space, box, grid, floor_heights)

    for connection in document["connections"]:
        source = str(connection["from"])
        target = str(connection["to"])
        if source not in room_centers or target not in room_centers:
            continue
        door_cell = _carve_corridor(
            grid,
            floor_heights,
            room_centers[source],
            room_centers[target],
            source_height=int(spaces[source].get("floor_height", 0)),
            target_height=int(spaces[target].get("floor_height", 0)),
            token=DOOR if connection.get("type") in {"door", "locked_door", "keyed_door"} else FLOOR,
        )
        if connection.get("type") in {"locked_door", "keyed_door"}:
            color = str(connection.get("key", "blue"))
            door_locks[_cell_key(door_cell)] = color

    _place_progression_things(document, room_boxes, grid, floor_heights, key_colors)
    _place_space_things(document, room_boxes, grid)

    return {
        "compiler_version": "0.2",
        "source_title": document.get("title"),
        "target": "cell_layout",
        "grid": grid,
        "floor_heights": floor_heights,
        "rooms": room_boxes,
        "key_colors": key_colors,
        "door_locks": door_locks,
    }


def compile_to_wad(document: dict[str, Any], path: Path, cell_size: int = 128) -> dict[str, Any]:
    """Compile romeroDSL directly to a sector-primitive UDMF PWAD."""

    report = validate_document(document)
    report.raise_for_errors()
    room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"])
    return write_geometry_wad(document, room_boxes, Path(path), cell_size=cell_size)


def _place_rooms(spaces: list[dict[str, Any]], progression: dict[str, Any], connections: list[dict[str, Any]]) -> dict[str, Box]:
    main_order = _main_path_order(spaces, progression)
    boxes: dict[str, Box] = {}
    cursor_c = 2
    base_r = 2
    gap = 5

    for space_id in main_order:
        space = next(space for space in spaces if space["id"] == space_id)
        w, h = _size_to_cells(str(space.get("size", "medium")))
        boxes[space_id] = {"r": base_r, "c": cursor_c, "w": w, "h": h}
        cursor_c += w + gap

    branch_slots: dict[str, int] = {}
    for space in spaces:
        space_id = space["id"]
        if space_id in boxes:
            continue
        anchor = _find_anchor(space_id, connections, boxes) or main_order[0]
        anchor_box = boxes[anchor]
        slot = branch_slots.get(anchor, 0)
        branch_slots[anchor] = slot + 1
        w, h = _size_to_cells(str(space.get("size", "medium")))
        # Alternate branches above and below the anchor so key rooms and secrets do not overlap.
        if slot % 2 == 0:
            r = anchor_box["r"] + anchor_box["h"] + 5 + (slot // 2) * (h + 3)
        else:
            r = max(2, anchor_box["r"] - h - 5 - (slot // 2) * (h + 3))
        c = max(2, anchor_box["c"] + anchor_box["w"] // 2 - w // 2)
        boxes[space_id] = {"r": r, "c": c, "w": w, "h": h}

    return boxes


def _main_path_order(spaces: list[dict[str, Any]], progression: dict[str, Any]) -> list[str]:
    space_ids = {space["id"] for space in spaces}
    key_locations = {
        key.get("location") for key in progression.get("keys", []) if isinstance(key, dict)
    }
    start = str(progression.get("start"))
    exit_space = str(progression.get("exit"))
    order: list[str] = []
    for node in progression.get("critical_path", []):
        if node not in space_ids or node in order:
            continue
        if node in key_locations and node not in {start, exit_space}:
            continue
        order.append(node)
    if start in space_ids and start not in order:
        order.insert(0, start)
    if exit_space in space_ids and exit_space not in order:
        order.append(exit_space)
    return order or [spaces[0]["id"]]


def _find_anchor(space_id: str, connections: list[dict[str, Any]], boxes: dict[str, Box]) -> str | None:
    for connection in connections:
        source = connection.get("from")
        target = connection.get("to")
        if source == space_id and target in boxes:
            return str(target)
        if target == space_id and source in boxes:
            return str(source)
    return None


def _grid_box_to_map_units(box: Box, cell_size: int = 128) -> dict[str, int]:
    return {
        "x": box["c"] * cell_size,
        "y": -box["r"] * cell_size,
        "width": box["w"] * cell_size,
        "height": box["h"] * cell_size,
    }


def _size_to_cells(size: str) -> tuple[int, int]:
    match size:
        case "tiny":
            return 3, 3
        case "small":
            return 5, 5
        case "medium":
            return 7, 7
        case "large":
            return 9, 7
        case "huge":
            return 11, 9
        case _:
            return 7, 7


def _apply_height_features(space: dict[str, Any], box: Box, grid: list[list[str]], floor_heights: list[list[int]]) -> None:
    base = int(space.get("floor_height", 0))
    features = space.get("height_topology", {}).get("features", [])
    if not isinstance(features, list):
        return
    for feature in features:
        if not isinstance(feature, dict):
            continue
        feature_type = feature.get("type")
        if feature_type in {"ceiling_pillar_cluster", "pillar", "pillar_cluster"}:
            count = int(feature.get("count", 1))
            pillar_height = int(feature.get("floor_height", base + 128))
            for cell in _pillar_cells(box, count):
                r, c = cell
                grid[r][c] = FLOOR
                floor_heights[r][c] = pillar_height
        elif feature_type in {"raised_platform", "platform"}:
            platform_height = int(feature.get("floor_height", base + 32))
            for r in range(box["r"] + 1, min(box["r"] + 3, box["r"] + box["h"] - 1)):
                for c in range(box["c"] + box["w"] - 3, box["c"] + box["w"] - 1):
                    grid[r][c] = FLOOR
                    floor_heights[r][c] = platform_height
        elif feature_type in {"shallow_pit", "pit"}:
            pit_height = int(feature.get("floor_height", base - 24))
            for r in range(box["r"] + box["h"] - 3, box["r"] + box["h"] - 1):
                for c in range(box["c"] + 1, min(box["c"] + 4, box["c"] + box["w"] - 1)):
                    grid[r][c] = FLOOR
                    floor_heights[r][c] = pit_height


def _pillar_cells(box: Box, count: int) -> list[Cell]:
    center_r = box["r"] + box["h"] // 2
    center_c = box["c"] + box["w"] // 2
    candidates = [
        (center_r - 1, center_c - 1),
        (center_r - 1, center_c + 1),
        (center_r + 1, center_c - 1),
        (center_r + 1, center_c + 1),
        (center_r, center_c),
    ]
    cells = [
        (r, c)
        for r, c in candidates
        if box["r"] < r < box["r"] + box["h"] - 1 and box["c"] < c < box["c"] + box["w"] - 1
    ]
    return cells[: max(0, min(count, len(cells)))]


def _carve_corridor(
    grid: list[list[str]],
    floor_heights: list[list[int]],
    start: Cell,
    end: Cell,
    source_height: int,
    target_height: int,
    token: str,
) -> Cell:
    path: list[Cell] = []
    r, c = start
    end_r, end_c = end
    step_c = 1 if end_c >= c else -1
    while c != end_c:
        path.append((r, c))
        c += step_c
    step_r = 1 if end_r >= r else -1
    while r != end_r:
        path.append((r, c))
        r += step_r
    path.append(end)

    door_index = max(1, min(len(path) - 2, len(path) // 2)) if len(path) > 2 else 0
    door_cell = path[door_index]
    for index, (pr, pc) in enumerate(path):
        grid[pr][pc] = token if token == DOOR and index == door_index else FLOOR
        floor_heights[pr][pc] = source_height if index <= door_index else target_height
    return door_cell


def _place_progression_things(
    document: dict[str, Any],
    room_boxes: dict[str, Box],
    grid: list[list[str]],
    floor_heights: list[list[int]],
    key_colors: dict[str, str],
) -> None:
    start_space = document["progression"].get("start")
    exit_space = document["progression"].get("exit")
    if start_space in room_boxes:
        _place_token(grid, room_boxes[str(start_space)], START)
    if exit_space in room_boxes:
        _place_exit_token(grid, room_boxes[str(exit_space)])

    for key in document["progression"].get("keys", []):
        if not isinstance(key, dict):
            continue
        location = key.get("location")
        color = str(key.get("color", "blue"))
        if location in room_boxes:
            cell = _place_token(grid, room_boxes[str(location)], KEY)
            key_colors[_cell_key(cell)] = color
            floor_heights[cell[0]][cell[1]] = floor_heights[cell[0]][cell[1]]


def _place_space_things(document: dict[str, Any], room_boxes: dict[str, Box], grid: list[list[str]]) -> None:
    for space in document["spaces"]:
        box = room_boxes[space["id"]]
        for item in _extract_items(space):
            token = _item_to_token(item)
            if token:
                _place_token(grid, box, token)
        for _monster_name, count in _extract_monsters(space):
            for _ in range(min(count, 8)):
                _place_token(grid, box, MONSTER)


def _extract_items(space: dict[str, Any]) -> list[str]:
    items: list[str] = []
    things = space.get("things", {})
    contains = space.get("contains", {})
    if isinstance(things, dict):
        raw_items = things.get("items", [])
        if isinstance(raw_items, list):
            items.extend(str(item) for item in raw_items)
    if isinstance(contains, dict):
        raw_items = contains.get("items", [])
        if isinstance(raw_items, list):
            items.extend(str(item) for item in raw_items)
    return items


def _extract_monsters(space: dict[str, Any]) -> list[tuple[str, int]]:
    encounter = space.get("encounter", {})
    if not isinstance(encounter, dict):
        return []
    monsters = encounter.get("monsters", [])
    result: list[tuple[str, int]] = []
    if not isinstance(monsters, list):
        return result
    for entry in monsters:
        if isinstance(entry, dict):
            for name, count in entry.items():
                result.append((str(name), int(count)))
    return result


def _item_to_token(item: str) -> str | None:
    lowered = item.lower()
    if "shotgun" in lowered or "chaingun" in lowered or "weapon" in lowered:
        return WEAPON
    if "shell" in lowered or "ammo" in lowered or "clip" in lowered:
        return AMMO
    if "health" in lowered or "stim" in lowered or "med" in lowered:
        return HEALTH
    return None


def _place_exit_token(grid: list[list[str]], box: Box) -> Cell:
    candidates = [
        (box["r"] + box["h"] // 2, box["c"] + box["w"] - 1),
        (box["r"] + box["h"] - 1, box["c"] + box["w"] // 2),
        (box["r"], box["c"] + box["w"] // 2),
        (box["r"] + box["h"] // 2, box["c"]),
    ]
    for r, c in candidates:
        if grid[r][c] == FLOOR:
            grid[r][c] = EXIT
            return (r, c)
    return _place_token(grid, box, EXIT, reverse=True)


def _place_token(grid: list[list[str]], box: Box, token: str, reverse: bool = False) -> Cell:
    candidates = [
        (r, c)
        for r in range(box["r"] + 1, box["r"] + box["h"] - 1)
        for c in range(box["c"] + 1, box["c"] + box["w"] - 1)
    ]
    if reverse:
        candidates = list(reversed(candidates))
    for cell in candidates:
        r, c = cell
        if grid[r][c] == FLOOR:
            grid[r][c] = token
            return cell
    center = (box["r"] + box["h"] // 2, box["c"] + box["w"] // 2)
    grid[center[0]][center[1]] = token
    return center


def _cell_key(cell: Cell) -> str:
    return f"{cell[0]},{cell[1]}"
