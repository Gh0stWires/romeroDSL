"""Minimal UDMF/PWAD writer for romeroDSL compiler output.

This is intentionally small and dependency-free. It turns a cell-level semantic
layout into a ZDoom UDMF TEXTMAP PWAD. The semantic DSL remains the authoring
source of truth; this module only serializes compiler-produced geometry.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

from romerodsl.materials import VALID_DOOM2_FLATS, VALID_DOOM2_TEXTURES

EMPTY = "empty"
FLOOR = "floor"
DOOR = "door"
KEY = "key"
MONSTER = "monster"
HEALTH = "health"
AMMO = "ammo"
WEAPON = "weapon"
START = "start"
EXIT = "exit"
OBSTACLE = "obstacle"

WALKABLE = {FLOOR, DOOR, KEY, MONSTER, HEALTH, AMMO, WEAPON, START, EXIT, OBSTACLE}
THING_TYPES = {
    KEY: 5,
    MONSTER: 3001,
    HEALTH: 2011,
    AMMO: 2008,
    WEAPON: 2001,
    OBSTACLE: 2035,
}
KEY_COLOR_THING_TYPES = {"blue": 5, "red": 13, "yellow": 6}
DOOR_LOCK_NUMBERS = {"blue": 2, "red": 3, "yellow": 4}
WALL_TEXTURE = "STARTAN2"
FLOOR_TEXTURE = "FLOOR0_1"
DOOR_TEXTURE = "BIGDOOR2"
EXIT_TEXTURE = "SW1EXIT"
SPECIAL_TEXTURE = "SW1COMM"

Cell = tuple[int, int]
Grid = list[list[str]]


def write_wad_from_layout(layout: dict[str, Any], path: Path, cell_size: int = 128) -> dict[str, Any]:
    """Write a UDMF PWAD from a compiler layout and return validation metadata."""

    grid: Grid = layout["grid"]
    height_map: list[list[int]] = layout["floor_heights"]
    key_colors: dict[str, str] = layout.get("key_colors", {})
    door_locks: dict[str, str] = layout.get("door_locks", {})
    textmap = build_textmap(grid, height_map, key_colors, door_locks, cell_size=cell_size)
    _write_textmap_wad(Path(path), textmap)
    report = validate_wad(path)
    report["source_title"] = layout.get("source_title")
    report["cell_size"] = cell_size
    return report


def build_textmap(
    grid: Grid,
    height_map: list[list[int]],
    key_colors: dict[str, str],
    door_locks: dict[str, str],
    cell_size: int = 128,
) -> str:
    """Build a ZDoom UDMF TEXTMAP string from a cell grid."""

    _validate_grid(grid, height_map)
    cells = [(r, c) for r, row in enumerate(grid) for c, token in enumerate(row) if token in WALKABLE]
    if not cells:
        raise ValueError("layout has no walkable cells")

    vertices: list[dict[str, int]] = []
    vertex_ids: dict[tuple[int, int], int] = {}
    sidedefs: list[dict[str, Any]] = []
    linedefs: list[dict[str, Any]] = []
    edge_ids: dict[tuple[int, int], int] = {}
    sectors: list[dict[str, Any]] = []
    ordered_cells = sorted(cells)
    sector_for_cell = {cell: index for index, cell in enumerate(ordered_cells)}

    for sector_id, (r, c) in enumerate(ordered_cells):
        token = grid[r][c]
        floor_height = int(height_map[r][c])
        ceiling_height = floor_height if token == DOOR else floor_height + 128
        sector = {
            "heightfloor": floor_height,
            "heightceiling": ceiling_height,
            "texturefloor": FLOOR_TEXTURE,
            "textureceiling": FLOOR_TEXTURE,
            "lightlevel": 192,
        }
        sectors.append(sector)

        x = c * cell_size
        y = -r * cell_size
        corners = [(x, y), (x + cell_size, y), (x + cell_size, y - cell_size), (x, y - cell_size)]
        for point in corners:
            if point not in vertex_ids:
                vertex_ids[point] = len(vertices)
                vertices.append({"x": point[0], "y": point[1]})

        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
            va = vertex_ids[a]
            vb = vertex_ids[b]
            edge = tuple(sorted((va, vb)))
            side_id = len(sidedefs)
            sidedefs.append(
                {
                    "sector": sector_id,
                    "texturemiddle": _line_texture(token, boundary=True),
                    "texturebottom": _line_texture(token, boundary=True),
                    "texturetop": _line_texture(token, boundary=True),
                }
            )
            if edge in edge_ids:
                line = linedefs[edge_ids[edge]]
                line.update(sideback=side_id, twosided=True, blocking=False)
                _update_two_sided_textures(line, sidedefs, sectors, token)
                _maybe_mark_door(line, sidedefs, sectors, ordered_cells, grid, door_locks)
            else:
                edge_ids[edge] = len(linedefs)
                linedefs.append(
                    {
                        "v1": va,
                        "v2": vb,
                        "sidefront": side_id,
                        "sideback": -1,
                        "blocking": True,
                        "twosided": False,
                    }
                )

    _mark_exit_line(linedefs, sidedefs, ordered_cells, grid)
    things = _build_things(ordered_cells, grid, sector_for_cell, key_colors, cell_size)

    groups = {
        "vertex": vertices,
        "sector": sectors,
        "sidedef": sidedefs,
        "linedef": linedefs,
        "thing": things,
    }
    return 'namespace = "ZDoom";\n' + "".join(
        _block(kind, fields) for kind, blocks in groups.items() for fields in blocks
    )


def _validate_grid(grid: Grid, height_map: list[list[int]]) -> None:
    if not grid or not grid[0]:
        raise ValueError("grid must be non-empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid rows must have equal widths")
    if len(height_map) != len(grid) or any(len(row) != width for row in height_map):
        raise ValueError("height_map must match grid shape")


def _block(kind: str, fields: dict[str, Any]) -> str:
    return kind + "\n{\n" + "".join(
        f"    {key} = {json.dumps(value)};\n" for key, value in fields.items()
    ) + "}\n"


def _line_texture(token: str, boundary: bool) -> str:
    if token == DOOR:
        return DOOR_TEXTURE
    if token == EXIT:
        return EXIT_TEXTURE
    return WALL_TEXTURE if boundary else "-"


def _update_two_sided_textures(
    line: dict[str, Any],
    sidedefs: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
    back_token: str,
) -> None:
    front_side = sidedefs[line["sidefront"]]
    back_side = sidedefs[line["sideback"]]
    front_sector = sectors[front_side["sector"]]
    back_sector = sectors[back_side["sector"]]
    front_side["texturemiddle"] = "-"
    back_side["texturemiddle"] = "-"
    if front_sector["heightfloor"] != back_sector["heightfloor"]:
        front_side["texturebottom"] = WALL_TEXTURE
        back_side["texturebottom"] = WALL_TEXTURE
    else:
        front_side["texturebottom"] = "-"
        back_side["texturebottom"] = "-"
    if front_sector["heightceiling"] != back_sector["heightceiling"]:
        texture = DOOR_TEXTURE if back_token == DOOR else WALL_TEXTURE
        front_side["texturetop"] = texture
        back_side["texturetop"] = texture
    else:
        front_side["texturetop"] = "-"
        back_side["texturetop"] = "-"


def _maybe_mark_door(
    line: dict[str, Any],
    sidedefs: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
    ordered_cells: list[Cell],
    grid: Grid,
    door_locks: dict[str, str],
) -> None:
    front_sector_id = sidedefs[line["sidefront"]]["sector"]
    back_sector_id = sidedefs[line["sideback"]]["sector"]
    front_cell = ordered_cells[front_sector_id]
    back_cell = ordered_cells[back_sector_id]
    door_cell = None
    if grid[front_cell[0]][front_cell[1]] == DOOR:
        door_cell = front_cell
        door_sector = sectors[front_sector_id]
    elif grid[back_cell[0]][back_cell[1]] == DOOR:
        door_cell = back_cell
        door_sector = sectors[back_sector_id]
    else:
        return

    cell_key = _cell_key(door_cell)
    lock_color = door_locks.get(cell_key)
    line.update(
        special=12,
        playeruse=True,
        playeruseback=True,
        repeatspecial=True,
        arg0=sectors.index(door_sector) + 1,
        arg1=16,
        arg2=150,
        arg3=0,
    )
    if lock_color:
        line["locknumber"] = DOOR_LOCK_NUMBERS[lock_color]
    for side_id in (line["sidefront"], line["sideback"]):
        sidedefs[side_id]["texturemiddle"] = "-"
        sidedefs[side_id]["texturebottom"] = DOOR_TEXTURE
        sidedefs[side_id]["texturetop"] = DOOR_TEXTURE


def _mark_exit_line(linedefs: list[dict[str, Any]], sidedefs: list[dict[str, Any]], ordered_cells: list[Cell], grid: Grid) -> None:
    exit_cells = [cell for cell in ordered_cells if grid[cell[0]][cell[1]] == EXIT]
    if not exit_cells:
        raise ValueError("layout has no exit cell")
    exit_sector_id = ordered_cells.index(exit_cells[0])
    for line in linedefs:
        if line["sideback"] == -1 and sidedefs[line["sidefront"]]["sector"] == exit_sector_id:
            line.update(special=243, playeruse=True, repeatspecial=False, arg0=0)
            sidedefs[line["sidefront"]]["texturemiddle"] = EXIT_TEXTURE
            return
    raise ValueError("exit cell has no boundary line")


def _build_things(
    ordered_cells: list[Cell],
    grid: Grid,
    sector_for_cell: dict[Cell, int],
    key_colors: dict[str, str],
    cell_size: int,
) -> list[dict[str, Any]]:
    things: list[dict[str, Any]] = []
    for cell in ordered_cells:
        token = grid[cell[0]][cell[1]]
        if token == START:
            thing_type = 1
        elif token in THING_TYPES:
            thing_type = _thing_type(token, cell, key_colors)
        else:
            continue
        r, c = cell
        things.append(
            {
                "x": (c + 0.5) * cell_size,
                "y": -(r + 0.5) * cell_size,
                "type": thing_type,
                "angle": 0,
                "height": 0,
                "skill1": True,
                "skill2": True,
                "skill3": True,
                "skill4": True,
                "skill5": True,
                "single": True,
                "coop": True,
                "dm": True,
                "sector": sector_for_cell[cell],
            }
        )
    return things


def _thing_type(token: str, cell: Cell, key_colors: dict[str, str]) -> int:
    if token == KEY:
        return KEY_COLOR_THING_TYPES[key_colors.get(_cell_key(cell), "blue")]
    return THING_TYPES[token]


def _write_textmap_wad(path: Path, text: str) -> None:
    payload = text.encode("ascii")
    directory = 12 + len(payload)
    data = struct.pack("<4sii", b"PWAD", 3, directory) + payload
    data += struct.pack("<ii8s", 12, 0, b"MAP01")
    data += struct.pack("<ii8s", 12, len(payload), b"TEXTMAP")
    data += struct.pack("<ii8s", directory, 0, b"ENDMAP")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def validate_wad(path: Path) -> dict[str, Any]:
    """Validate the narrow WAD profile emitted by this package."""

    groups = _parse_textmap(_read_textmap(Path(path)))
    sectors = groups["sector"]
    sidedefs = groups["sidedef"]
    linedefs = groups["linedef"]
    things = groups["thing"]
    if not sectors or not linedefs:
        raise ValueError("missing geometry")
    for line in linedefs:
        if line["sidefront"] >= len(sidedefs) or line["sideback"] >= len(sidedefs):
            raise ValueError("linedef references invalid sidedef")
    door_lines = [line for line in linedefs if line.get("special") == 12]
    door_tags = {line.get("arg0") for line in door_lines if line.get("arg0", 0) > 0}
    locked_door_tags = {
        line.get("arg0") for line in door_lines if line.get("arg0", 0) > 0 and line.get("locknumber", 0) > 0
    }
    missing_visible_textures = _missing_visible_textures(linedefs, sidedefs, sectors)
    unknown_flats = sorted(
        {
            texture
            for sector in sectors
            for texture in (sector.get("texturefloor"), sector.get("textureceiling"))
            if isinstance(texture, str) and texture not in VALID_DOOM2_FLATS and texture != "-"
        }
    )
    unknown_textures = sorted(
        {
            texture
            for side in sidedefs
            for texture in (side.get("texturemiddle"), side.get("texturebottom"), side.get("texturetop"))
            if isinstance(texture, str) and texture not in VALID_DOOM2_TEXTURES and texture != "-"
        }
    )
    return {
        "valid": True,
        "namespace": "ZDoom",
        "sectors": len(sectors),
        "vertices": len(groups["vertex"]),
        "linedefs": len(linedefs),
        "sidedefs": len(sidedefs),
        "things": len(things),
        "player_starts": sum(thing["type"] == 1 for thing in things),
        "keys": sum(thing["type"] in {5, 6, 13, 38, 39, 40} for thing in things),
        "exits": sum(line.get("special") == 243 for line in linedefs),
        "door_linedefs": len(door_lines),
        "doors": len(door_tags),
        "locked_doors": len(locked_door_tags),
        "locked_door_linedefs": sum(line.get("special") == 12 and line.get("locknumber", 0) > 0 for line in linedefs),
        "bidirectional_door_linedefs": sum(
            line.get("special") == 12 and line.get("playeruse") is True and line.get("playeruseback") is True
            for line in linedefs
        ),
        "closed_door_sectors": sum(
            sector["heightfloor"] == sector["heightceiling"] for sector in sectors
        ),
        "floor_height_levels": sorted({sector["heightfloor"] for sector in sectors}),
        "missing_visible_textures": len(missing_visible_textures),
        "missing_visible_texture_linedefs": missing_visible_textures[:20],
        "unknown_flats": unknown_flats,
        "unknown_textures": unknown_textures,
        "monsters": sum(thing["type"] in {3001, 3002, 3003, 3004, 3005, 3006} for thing in things),
        "engine_tested": False,
    }


def _missing_visible_textures(
    linedefs: list[dict[str, Any]],
    sidedefs: list[dict[str, Any]],
    sectors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for index, line in enumerate(linedefs):
        front = sidedefs[line["sidefront"]]
        if line.get("sideback", -1) < 0:
            if front.get("texturemiddle") == "-":
                missing.append({"linedef": index, "side": "front", "surface": "middle"})
            continue
        back = sidedefs[line["sideback"]]
        front_sector = sectors[front["sector"]]
        back_sector = sectors[back["sector"]]
        if front.get("texturemiddle") != "-" or back.get("texturemiddle") != "-":
            missing.append({"linedef": index, "side": "both", "surface": "middle_should_be_clear"})
        if front_sector["heightfloor"] != back_sector["heightfloor"]:
            if front.get("texturebottom") == "-":
                missing.append({"linedef": index, "side": "front", "surface": "lower"})
            if back.get("texturebottom") == "-":
                missing.append({"linedef": index, "side": "back", "surface": "lower"})
        if front_sector["heightceiling"] != back_sector["heightceiling"]:
            if front.get("texturetop") == "-":
                missing.append({"linedef": index, "side": "front", "surface": "upper"})
            if back.get("texturetop") == "-":
                missing.append({"linedef": index, "side": "back", "surface": "upper"})
    return missing


def _read_textmap(path: Path) -> str:
    data = path.read_bytes()
    if len(data) < 12:
        raise ValueError("WAD too short")
    magic, count, directory = struct.unpack_from("<4sii", data)
    if magic != b"PWAD" or count != 3 or directory < 12 or directory + 16 * count != len(data):
        raise ValueError("invalid WAD directory")
    names = []
    text = None
    for index in range(count):
        offset, size, name = struct.unpack_from("<ii8s", data, directory + index * 16)
        label = name.rstrip(b"\0")
        names.append(label)
        if size and (offset < 12 or offset + size > directory):
            raise ValueError("lump outside data region")
        if label == b"TEXTMAP":
            text = data[offset : offset + size].decode("ascii")
    if names != [b"MAP01", b"TEXTMAP", b"ENDMAP"] or text is None:
        raise ValueError("unsupported generated WAD profile")
    return text


def _parse_textmap(text: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {
        kind: [] for kind in ("vertex", "sector", "sidedef", "linedef", "thing")
    }
    for kind, body in re.findall(r"(\w+)\s*\{([^{}]*)\}", text):
        if kind in groups:
            groups[kind].append(
                {key: json.loads(value) for key, value in re.findall(r"(\w+)\s*=\s*([^;]+);", body)}
            )
    return groups


def _cell_key(cell: Cell) -> str:
    return f"{cell[0]},{cell[1]}"
