"""Compiler for romeroDSL v0.1.

The compiler bridges model-authored semantic Doom design contracts toward real
geometry. v0.1 still keeps the geometry deliberately simple: rooms become cell
rectangles, connections become corridors, and WAD export uses one UDMF sector per
cell. That is enough to prove the new semantic path can preserve progression,
keys, locks, height features, and monsters without asking a raster model to
invent them.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import random
from typing import Any

from romerodsl.geometry import write_geometry_wad
from romerodsl.materials import SURFACE_ROLES, resolve_materials
from romerodsl.schema import canonical_room_shape, validate_document
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
    layout_seed = _layout_seed(document)
    room_boxes = _place_rooms(spaces, document["progression"], connections, layout_seed=layout_seed)
    layout = _layout_report(room_boxes, connections, layout_seed)
    return {
        "compiler_version": "0.2",
        "source_title": document.get("title"),
        "target": "abstract_geometry_plan",
        "units": "doom_map_units",
        "layout": layout,
        "rooms": [
            {
                "id": space["id"],
                "role": space.get("role", "unspecified"),
                "shape": space.get("shape", "rectangle"),
                "orientation": space.get("orientation", 0),
                "box": _grid_box_to_map_units(room_boxes[space["id"]]),
                "grid_box": room_boxes[space["id"]],
                "floor_height": space.get("floor_height", 0),
                "ceiling_height": space.get("ceiling_height", 128),
                "height_features": space.get("height_topology", {}).get("features", []),
                "materials": resolve_materials(document, space),
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
    layout_seed = _layout_seed(document)
    room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"], layout_seed=layout_seed)
    height = max(box["r"] + box["h"] for box in room_boxes.values()) + 3
    width = max(box["c"] + box["w"] for box in room_boxes.values()) + 3
    grid = [["empty" for _ in range(width)] for _ in range(height)]
    floor_heights = [[0 for _ in range(width)] for _ in range(height)]
    surface_materials = {
        role: [["" for _ in range(width)] for _ in range(height)] for role in SURFACE_ROLES
    }
    key_colors: dict[str, str] = {}
    door_locks: dict[str, str] = {}
    room_centers: dict[str, Cell] = {}
    connection_doors: dict[str, Cell] = {}
    routed_cells: set[Cell] = set()

    for space_id, box in room_boxes.items():
        space = spaces[space_id]
        floor_height = int(space.get("floor_height", 0))
        materials = resolve_materials(document, space)
        for r in range(box["r"], box["r"] + box["h"]):
            for c in range(box["c"], box["c"] + box["w"]):
                grid[r][c] = FLOOR
                floor_heights[r][c] = floor_height
                for role, texture in materials.items():
                    surface_materials[role][r][c] = texture
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
            room_boxes=room_boxes,
            routed_cells=routed_cells,
        )
        source_materials = resolve_materials(document, spaces[source])
        _apply_path_materials(surface_materials, room_centers[source], room_centers[target], source_materials)
        if connection.get("type") in {"locked_door", "keyed_door"}:
            color = str(connection.get("key", "blue"))
            door_locks[_cell_key(door_cell)] = color
            surface_materials["door"][door_cell[0]][door_cell[1]] = source_materials["door"]
        connection_id = connection.get("id") or f"{source}->{target}"
        if connection.get("type") in {"locked_door", "keyed_door"}:
            connection_doors[str(connection_id)] = door_cell

    _place_progression_things(document, room_boxes, grid, floor_heights, key_colors)
    _place_space_things(document, room_boxes, grid)

    return {
        "compiler_version": "0.2",
        "source_title": document.get("title"),
        "target": "cell_layout",
        "grid": grid,
        "floor_heights": floor_heights,
        "rooms": room_boxes,
        "layout": _layout_report(room_boxes, document["connections"], layout_seed),
        "surface_materials": surface_materials,
        "key_colors": key_colors,
        "door_locks": door_locks,
        "connection_doors": {key: _cell_key(value) for key, value in connection_doors.items()},
        "physical_progression": _physical_progression_report(
            document, grid, connection_doors, key_colors
        ),
    }


def validate_physical_progression(document: dict[str, Any]) -> dict[str, Any]:
    """Check route accessibility on the compiled cell layout without repairing it."""

    return compile_to_layout(document)["physical_progression"]


def _physical_progression_report(
    document: dict[str, Any],
    grid: list[list[str]],
    connection_doors: dict[str, Cell],
    key_colors: dict[str, str],
) -> dict[str, Any]:
    starts = _find_cells(grid, START)
    exits = _find_cells(grid, EXIT)
    passable = {
        (row, column)
        for row, values in enumerate(grid)
        for column, token in enumerate(values)
        if token != "empty"
    }
    errors: list[str] = []
    if not starts:
        errors.append("compiled layout has no player start")
    if not exits:
        errors.append("compiled layout has no exit")
    reachable = _reachable_cells(passable, starts)
    if not reachable.intersection(exits):
        errors.append("compiled exit is not reachable from start")

    progression = document["progression"]
    gate_checks: list[dict[str, Any]] = []
    key_cells_by_color: dict[str, set[Cell]] = {}
    for cell, color in key_colors.items():
        key_cells_by_color.setdefault(color, set()).add(_parse_cell_key(cell))
    for gate in progression.get("gates", []):
        if not isinstance(gate, dict) or gate.get("type") not in {"locked_door", "keyed_door"}:
            continue
        if not gate.get("required_for_exit", False):
            continue
        gate_id = str(gate.get("id"))
        color = str(gate.get("key"))
        door = connection_doors.get(gate_id)
        if door is None:
            errors.append(f"compiled layout has no door cell for required gate {gate_id!r}")
            continue
        blocked_reachable = _reachable_cells(passable - {door}, starts)
        key_reachable = bool(blocked_reachable & key_cells_by_color.get(color, set()))
        exit_reachable = bool(blocked_reachable.intersection(exits))
        gate_checks.append({
            "gate_id": gate_id,
            "color": color,
            "key_reachable_before_gate": key_reachable,
            "exit_reachable_without_gate": exit_reachable,
        })
        if not key_reachable:
            errors.append(f"{color} key is not reachable before compiled gate {gate_id!r}")
        if exit_reachable:
            errors.append(f"compiled exit bypasses required gate {gate_id!r}")

    return {
        "valid": not errors,
        "start_reachable_cells": len(reachable),
        "gate_checks": gate_checks,
        "errors": errors,
    }


def _find_cells(grid: list[list[str]], token: str) -> set[Cell]:
    return {
        (row, column)
        for row, values in enumerate(grid)
        for column, value in enumerate(values)
        if value == token
    }


def _reachable_cells(passable: set[Cell], starts: set[Cell]) -> set[Cell]:
    reachable: set[Cell] = set()
    frontier = list(starts & passable)
    while frontier:
        cell = frontier.pop()
        if cell in reachable or cell not in passable:
            continue
        reachable.add(cell)
        row, column = cell
        frontier.extend(((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)))
    return reachable


def _parse_cell_key(value: str) -> Cell:
    row, column = value.split(",", 1)
    return int(row), int(column)


def compile_to_wad(document: dict[str, Any], path: Path, cell_size: int = 128) -> dict[str, Any]:
    """Compile romeroDSL directly to a sector-primitive UDMF PWAD."""

    report = validate_document(document)
    report.raise_for_errors()
    layout_seed = _layout_seed(document)
    room_boxes = _place_rooms(document["spaces"], document["progression"], document["connections"], layout_seed=layout_seed)
    return write_geometry_wad(document, room_boxes, Path(path), cell_size=cell_size)


def _layout_seed(document: dict[str, Any]) -> int:
    raw_seed = document.get("layout_seed", 0)
    if isinstance(raw_seed, bool):
        return int(raw_seed)
    if isinstance(raw_seed, int):
        return raw_seed
    raise ValueError("layout_seed must be an integer when provided")


def _place_rooms(
    spaces: list[dict[str, Any]],
    progression: dict[str, Any],
    connections: list[dict[str, Any]],
    *,
    layout_seed: int = 0,
) -> dict[str, Box]:
    return _graph_place_rooms(spaces, progression, connections, layout_seed)


def _graph_place_rooms(
    spaces: list[dict[str, Any]],
    progression: dict[str, Any],
    connections: list[dict[str, Any]],
    layout_seed: int,
) -> dict[str, Box]:
    """Place rooms from graph layers, rather than from a linear room cursor.

    BFS layers provide a deterministic scaffold from the start room. Rooms that
    share a layer are stacked around the layer's center, so branches and loop
    reconnections get distinct spatial positions. The seed only controls stable
    tie ordering and mirroring; it never adds unconstrained random offsets.
    """

    if not spaces:
        return {}
    space_by_id = {str(space["id"]): space for space in spaces}
    adjacency = _layout_adjacency(space_by_id, connections)
    start = str(progression.get("start"))
    if start not in space_by_id:
        start = min(space_by_id)
    levels = _graph_levels(adjacency, start, layout_seed)
    critical_branch_nodes = _critical_return_branches(progression, set(space_by_id))
    critical_order = _main_path_order(spaces, progression)
    critical_rank = {node: index for index, node in enumerate(critical_order)}
    seeded_rank = _seeded_node_ranks(space_by_id, layout_seed)

    max_width = max(_size_to_cells(str(space.get("size", "medium")))[0] for space in spaces)
    max_height = max(_size_to_cells(str(space.get("size", "medium")))[1] for space in spaces)
    column_step = max_width + 6
    row_step = max_height + 5
    layer_nodes: dict[int, list[str]] = {}
    for node, level in levels.items():
        layer_nodes.setdefault(level, []).append(node)
    max_level = max(layer_nodes)
    max_slots = max((len(nodes) - 1) // 2 for nodes in layer_nodes.values())
    base_row = 2 + max_slots * row_step
    base_column = 2
    mirror = layout_seed % 2 != 0
    boxes: dict[str, Box] = {}
    slot_pattern = _center_out_slots(max(len(nodes) for nodes in layer_nodes.values()))

    for level in sorted(layer_nodes):
        nodes = sorted(
            layer_nodes[level],
            key=lambda node: (
                0 if node in critical_rank else 1,
                critical_rank.get(node, 10**6),
                _role_layout_rank(space_by_id[node].get("role")),
                seeded_rank[node],
                node,
            ),
        )
        for index, node in enumerate(nodes):
            space = space_by_id[node]
            width, height = _size_to_cells(str(space.get("size", "medium")))
            slot = slot_pattern[index]
            column_level = max_level - level if mirror else level
            branch_column_offset = 0
            if node in critical_branch_nodes or node not in critical_rank:
                parent_candidates = [
                    neighbor
                    for neighbor in adjacency[node]
                    if neighbor in boxes and levels[neighbor] < levels[node]
                ]
                if parent_candidates:
                    parent = min(parent_candidates, key=lambda candidate: (levels[candidate], seeded_rank[candidate], candidate))
                    parent_box = boxes[parent]
                    parent_center = parent_box["c"] + parent_box["w"] // 2
                    branch_column_offset = parent_center - (base_column + column_level * column_step) - width // 2
            boxes[node] = {
                "r": base_row + slot * row_step + (max_height - height) // 2,
                "c": base_column + column_level * column_step + branch_column_offset,
                "w": width,
                "h": height,
            }

    _resolve_layout_overlaps(boxes, levels, seeded_rank, row_step, column_step)
    _validate_layout_boxes(boxes, connections)
    return boxes


def _layout_adjacency(
    space_by_id: dict[str, dict[str, Any]], connections: list[dict[str, Any]]
) -> dict[str, set[str]]:
    adjacency = {space_id: set() for space_id in space_by_id}
    for connection in connections:
        source = str(connection.get("from"))
        target = str(connection.get("to"))
        if source in adjacency and target in adjacency and source != target:
            adjacency[source].add(target)
            adjacency[target].add(source)
    return adjacency


def _graph_levels(adjacency: dict[str, set[str]], start: str, seed: int) -> dict[str, int]:
    levels = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        neighbors = sorted(adjacency.get(node, set()), key=lambda neighbor: (_stable_seed_rank(neighbor, seed), neighbor))
        for neighbor in neighbors:
            if neighbor not in levels:
                levels[neighbor] = levels[node] + 1
                queue.append(neighbor)
    next_level = max(levels.values(), default=0) + 2
    for node in sorted(adjacency):
        if node not in levels:
            levels[node] = next_level
            next_level += 2
    return levels


def _role_layout_rank(role: Any) -> int:
    role_order = {
        "player_start": 0,
        "combat_hub": 1,
        "connector": 2,
        "key_room": 3,
        "combat_room": 4,
        "reward_room": 5,
        "secret_reward": 6,
        "final_combat": 7,
        "exit": 8,
    }
    return role_order.get(str(role), 9)


def _critical_return_branches(progression: dict[str, Any], space_ids: set[str]) -> set[str]:
    raw_path = progression.get("critical_path", [])
    if not isinstance(raw_path, list):
        return set()
    spaces = [str(node) for node in raw_path if str(node) in space_ids]
    branches: set[str] = set()
    for index in range(len(spaces) - 2):
        if spaces[index] == spaces[index + 2] and spaces[index + 1] in space_ids:
            branches.add(spaces[index + 1])
    return branches


def _seeded_node_ranks(space_by_id: dict[str, dict[str, Any]], seed: int) -> dict[str, int]:
    nodes = sorted(space_by_id)
    randomizer = random.Random(seed)
    randomizer.shuffle(nodes)
    return {node: index for index, node in enumerate(nodes)}


def _stable_seed_rank(value: str, seed: int) -> int:
    accumulator = seed & 0xFFFFFFFF
    for character in value:
        accumulator = (accumulator * 33 + ord(character)) & 0xFFFFFFFF
    return accumulator


def _center_out_slots(count: int) -> list[int]:
    slots = [0]
    offset = 1
    while len(slots) < count:
        slots.extend((offset, -offset))
        offset += 1
    return slots[:count]


def _resolve_layout_overlaps(
    boxes: dict[str, Box],
    levels: dict[str, int],
    seeded_rank: dict[str, int],
    row_step: int,
    column_step: int,
) -> None:
    """Resolve residual footprint collisions without changing graph layers."""

    nodes = sorted(boxes, key=lambda node: (levels[node], seeded_rank[node], node))
    for _ in range(len(nodes) * 3 + 1):
        changed = False
        for index, left_id in enumerate(nodes):
            left = boxes[left_id]
            for right_id in nodes[index + 1 :]:
                right = boxes[right_id]
                if not _boxes_overlap(left, right):
                    continue
                changed = True
                if levels[left_id] == levels[right_id]:
                    right["r"] = left["r"] + left["h"] + 5
                else:
                    right["c"] = left["c"] + left["w"] + 5
        if not changed:
            return
        for box in boxes.values():
            box["r"] = max(2, box["r"])
            box["c"] = max(2, box["c"])
    _validate_layout_boxes(boxes, [])


def _boxes_overlap(left: Box, right: Box) -> bool:
    return not (
        left["c"] + left["w"] + 1 <= right["c"]
        or right["c"] + right["w"] + 1 <= left["c"]
        or left["r"] + left["h"] + 1 <= right["r"]
        or right["r"] + right["h"] + 1 <= left["r"]
    )


def _validate_layout_boxes(boxes: dict[str, Box], connections: list[dict[str, Any]]) -> None:
    nodes = sorted(boxes)
    for index, left_id in enumerate(nodes):
        for right_id in nodes[index + 1 :]:
            if _boxes_overlap(boxes[left_id], boxes[right_id]):
                raise ValueError(f"layout placed overlapping rooms {left_id!r} and {right_id!r}")
    for connection in connections:
        source = str(connection.get("from"))
        target = str(connection.get("to"))
        if source not in boxes or target not in boxes:
            raise ValueError(f"layout connection references unplaced rooms {source!r}, {target!r}")


def _layout_report(
    boxes: dict[str, Box], connections: list[dict[str, Any]], layout_seed: int
) -> dict[str, Any]:
    orientation_counts = {"horizontal": 0, "vertical": 0, "diagonal": 0}
    for connection in connections:
        source = boxes.get(str(connection.get("from")))
        target = boxes.get(str(connection.get("to")))
        if source is None or target is None:
            continue
        source_center = (source["c"] + source["w"] // 2, source["r"] + source["h"] // 2)
        target_center = (target["c"] + target["w"] // 2, target["r"] + target["h"] // 2)
        if source_center[0] == target_center[0]:
            orientation_counts["vertical"] += 1
        elif source_center[1] == target_center[1]:
            orientation_counts["horizontal"] += 1
        else:
            orientation_counts["diagonal"] += 1
    return {
        "strategy": "graph_bfs_layers",
        "seed": layout_seed,
        "room_count": len(boxes),
        "connection_count": len(connections),
        "connection_orientations": orientation_counts,
        "overlap_count": 0,
    }


def _main_path_order(spaces: list[dict[str, Any]], progression: dict[str, Any]) -> list[str]:
    space_ids = {space["id"] for space in spaces}
    start = str(progression.get("start"))
    exit_space = str(progression.get("exit"))
    critical_spaces = [node for node in progression.get("critical_path", []) if node in space_ids]
    order: list[str] = []
    for index, node in enumerate(critical_spaces):
        if node in order:
            continue
        next_space = next((future for future in critical_spaces[index + 1 :] if future != node), None)
        if next_space in order and node not in {start, exit_space}:
            # The critical path went into a side room and then returned to an
            # already-placed space. Keep that room as a branch instead of laying
            # it across the main route and creating crossing corridors.
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
        features = []
    features = list(features)
    shape = canonical_room_shape(space.get("shape", "rectangle"))
    if not features and shape in {"raised_center", "depressed_floor"}:
        features.append({"type": shape, "floor_height": base + (32 if shape == "raised_center" else -24)})
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
        elif feature_type in {"raised_center", "depressed_floor", "raised_central_structure", "central_platform", "central_pit"}:
            center_height = int(feature.get("floor_height", base + (32 if "raised" in str(feature_type) else -24)))
            center_r = box["r"] + box["h"] // 2
            center_c = box["c"] + box["w"] // 2
            for r in range(max(box["r"] + 1, center_r - 1), min(box["r"] + box["h"] - 1, center_r + 2)):
                for c in range(max(box["c"] + 1, center_c - 1), min(box["c"] + box["w"] - 1, center_c + 2)):
                    grid[r][c] = FLOOR
                    floor_heights[r][c] = center_height


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
    *,
    room_boxes: dict[str, Box] | None = None,
    routed_cells: set[Cell] | None = None,
) -> Cell:
    if room_boxes is None:
        path = _manhattan_path(start, end)
    else:
        path = _obstacle_aware_path(grid, start, end, room_boxes, routed_cells or set())
    if len(path) < 2:
        raise ValueError(f"could not route corridor between {start!r} and {end!r}")

    endpoint_cells = _box_cells(room_boxes, start, end) if room_boxes else {start, end}
    exterior_path = [cell for cell in path if cell not in endpoint_cells]
    door_index = len(exterior_path) // 2 if exterior_path else len(path) // 2
    door_cell = exterior_path[door_index] if exterior_path else path[door_index]
    for index, (pr, pc) in enumerate(path):
        grid[pr][pc] = token if token == DOOR and (pr, pc) == door_cell else FLOOR
        floor_heights[pr][pc] = source_height if index <= path.index(door_cell) else target_height
    if routed_cells is not None:
        routed_cells.update(cell for cell in path if cell not in endpoint_cells)
    return door_cell


def _manhattan_path(start: Cell, end: Cell) -> list[Cell]:
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
    return path


def _box_cells(room_boxes: dict[str, Box] | None, start: Cell, end: Cell) -> set[Cell]:
    if not room_boxes:
        return {start, end}
    cells: set[Cell] = set()
    for box in room_boxes.values():
        if box["r"] <= start[0] < box["r"] + box["h"] and box["c"] <= start[1] < box["c"] + box["w"]:
            cells.update(_cells_in_box(box))
        if box["r"] <= end[0] < box["r"] + box["h"] and box["c"] <= end[1] < box["c"] + box["w"]:
            cells.update(_cells_in_box(box))
    return cells


def _cells_in_box(box: Box) -> set[Cell]:
    return {
        (row, column)
        for row in range(box["r"], box["r"] + box["h"])
        for column in range(box["c"], box["c"] + box["w"])
    }


def _obstacle_aware_path(
    grid: list[list[str]],
    start: Cell,
    end: Cell,
    room_boxes: dict[str, Box],
    routed_cells: set[Cell],
) -> list[Cell]:
    path = _search_grid_path(grid, start, end, room_boxes, routed_cells, block_routed=True)
    if path is None:
        path = _search_grid_path(grid, start, end, room_boxes, routed_cells, block_routed=False)
    if path is None:
        raise ValueError(f"no obstacle-aware corridor route between {start!r} and {end!r}")
    return path


def _search_grid_path(
    grid: list[list[str]],
    start: Cell,
    end: Cell,
    room_boxes: dict[str, Box],
    routed_cells: set[Cell],
    *,
    block_routed: bool,
) -> list[Cell] | None:
    endpoint_cells = _box_cells(room_boxes, start, end)
    room_cells = set().union(*(_cells_in_box(box) for box in room_boxes.values()))
    blocked = (room_cells - endpoint_cells) | (routed_cells if block_routed else set())
    height = len(grid)
    width = len(grid[0]) if height else 0
    frontier = deque([start])
    previous: dict[Cell, Cell | None] = {start: None}
    while frontier:
        cell = frontier.popleft()
        if cell == end:
            break
        row, column = cell
        neighbors = ((row, column + 1), (row + 1, column), (row, column - 1), (row - 1, column))
        for neighbor in neighbors:
            nr, nc = neighbor
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            if neighbor in previous or neighbor in blocked:
                continue
            if grid[nr][nc] != "empty" and neighbor not in endpoint_cells and neighbor not in routed_cells:
                continue
            previous[neighbor] = cell
            frontier.append(neighbor)
    if end not in previous:
        return None
    path: list[Cell] = []
    current: Cell | None = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path



def _apply_path_materials(
    surface_materials: dict[str, list[list[str]]],
    start: Cell,
    end: Cell,
    materials: dict[str, str],
) -> None:
    r, c = start
    end_r, end_c = end
    step_c = 1 if end_c >= c else -1
    while c != end_c:
        _set_cell_materials(surface_materials, r, c, materials)
        c += step_c
    step_r = 1 if end_r >= r else -1
    while r != end_r:
        _set_cell_materials(surface_materials, r, c, materials)
        r += step_r
    _set_cell_materials(surface_materials, end_r, end_c, materials)


def _set_cell_materials(
    surface_materials: dict[str, list[list[str]]],
    r: int,
    c: int,
    materials: dict[str, str],
) -> None:
    for role, texture in materials.items():
        surface_materials[role][r][c] = texture


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
