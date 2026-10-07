"""Sector-primitive geometry compiler for romeroDSL.

This is the v0.3 path: it keeps the high-level DSL as the source of truth, but
exports cleaner UDMF than the debug cell-grid writer. Rooms become main sectors,
connections become corridor/door sectors, and intra-room height features become
subsectors inside the room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from romerodsl.materials import resolve_materials
from romerodsl.wad import (
    DOOR_LOCK_NUMBERS,
    DOOR_TEXTURE,
    EXIT_TEXTURE,
    FLOOR_TEXTURE,
    KEY_COLOR_THING_TYPES,
    WALL_TEXTURE,
    _write_textmap_wad,
    validate_wad,
)

Side = Literal["left", "right", "top", "bottom"]
Point = tuple[int, int]
Rect = tuple[int, int, int, int]

THING_TYPES = {
    "player_start": 1,
    "key": 5,
    "monster": 3001,
    "health": 2011,
    "ammo": 2008,
    "weapon": 2001,
}


@dataclass(slots=True)
class Opening:
    side: Side
    start: int
    end: int


@dataclass(slots=True)
class GeometryStats:
    room_sectors: int = 0
    corridor_sectors: int = 0
    door_sectors: int = 0
    height_feature_sectors: int = 0
    things_by_kind: dict[str, int] = field(default_factory=dict)


class UdmfBuilder:
    """Small UDMF builder that merges exact shared line segments into portals."""

    def __init__(self) -> None:
        self.vertices: list[dict[str, int]] = []
        self.vertex_ids: dict[Point, int] = {}
        self.sectors: list[dict[str, Any]] = []
        self.sidedefs: list[dict[str, Any]] = []
        self.linedefs: list[dict[str, Any]] = []
        self.things: list[dict[str, Any]] = []
        self.edge_ids: dict[tuple[int, int], int] = {}

    def add_sector(
        self,
        *,
        floor: int,
        ceiling: int,
        texturefloor: str = FLOOR_TEXTURE,
        textureceiling: str = FLOOR_TEXTURE,
        lightlevel: int = 192,
        tag: int | None = None,
    ) -> int:
        sector: dict[str, Any] = {
            "heightfloor": floor,
            "heightceiling": ceiling,
            "texturefloor": texturefloor,
            "textureceiling": textureceiling,
            "lightlevel": lightlevel,
        }
        if tag is not None:
            sector["id"] = tag
        self.sectors.append(sector)
        return len(self.sectors) - 1

    def add_line(
        self,
        a: Point,
        b: Point,
        sector: int,
        *,
        texture: str = WALL_TEXTURE,
        blocking: bool = True,
        special: dict[str, Any] | None = None,
    ) -> int:
        va = self._vertex(a)
        vb = self._vertex(b)
        side_id = len(self.sidedefs)
        self.sidedefs.append(
            {
                "sector": sector,
                "texturemiddle": texture,
                "texturebottom": texture,
                "texturetop": texture,
            }
        )
        edge = tuple(sorted((va, vb)))
        if edge in self.edge_ids:
            line_id = self.edge_ids[edge]
            line = self.linedefs[line_id]
            line.update(sideback=side_id, twosided=True, blocking=False)
            self._apply_two_sided_textures(line_id)
        else:
            line_id = len(self.linedefs)
            self.edge_ids[edge] = line_id
            line = {
                "v1": va,
                "v2": vb,
                "sidefront": side_id,
                "sideback": -1,
                "blocking": blocking,
                "twosided": False,
            }
            self.linedefs.append(line)
        if special:
            self.linedefs[line_id].update(special)
            if self.linedefs[line_id].get("special") == 12:
                self._apply_door_textures(line_id)
        return line_id

    def add_inner_rect(self, rect: Rect, sector: int, parent_sector: int, *, texture: str = WALL_TEXTURE) -> None:
        x1, y1, x2, y2 = rect
        # Emit clockwise so the front sidedef, assigned to the inner feature
        # sector, faces the feature interior. Reversed inner loops expose backsides
        # to the player and can render as hall-of-mirrors in Zandronum.
        corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
        for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
            front_side = len(self.sidedefs)
            self.sidedefs.append(
                {
                    "sector": sector,
                    "texturemiddle": "-",
                    "texturebottom": texture,
                    "texturetop": texture,
                }
            )
            back_side = len(self.sidedefs)
            self.sidedefs.append(
                {
                    "sector": parent_sector,
                    "texturemiddle": "-",
                    "texturebottom": texture,
                    "texturetop": texture,
                }
            )
            line = {
                "v1": self._vertex(a),
                "v2": self._vertex(b),
                "sidefront": front_side,
                "sideback": back_side,
                "blocking": False,
                "twosided": True,
            }
            self.linedefs.append(line)
            self._apply_two_sided_textures(len(self.linedefs) - 1)

    def add_thing(self, x: int, y: int, kind: str, *, color: str | None = None) -> None:
        thing_type = KEY_COLOR_THING_TYPES[color or "blue"] if kind == "key" else THING_TYPES[kind]
        self.things.append(
            {
                "x": x,
                "y": y,
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
            }
        )

    def textmap(self) -> str:
        groups = {
            "vertex": self.vertices,
            "sector": self.sectors,
            "sidedef": self.sidedefs,
            "linedef": self.linedefs,
            "thing": self.things,
        }
        return 'namespace = "ZDoom";\n' + "".join(
            _block(kind, fields) for kind, blocks in groups.items() for fields in blocks
        )

    def _vertex(self, point: Point) -> int:
        if point not in self.vertex_ids:
            self.vertex_ids[point] = len(self.vertices)
            self.vertices.append({"x": point[0], "y": point[1]})
        return self.vertex_ids[point]

    def _apply_two_sided_textures(self, line_id: int) -> None:
        line = self.linedefs[line_id]
        if line["sideback"] < 0:
            return
        front = self.sidedefs[line["sidefront"]]
        back = self.sidedefs[line["sideback"]]
        front_sector = self.sectors[front["sector"]]
        back_sector = self.sectors[back["sector"]]
        front["texturemiddle"] = "-"
        back["texturemiddle"] = "-"
        # Keep upper/lower textures populated even when a same-height portal
        # does not currently expose them. Zandronum treats any later-exposed
        # upper/lower gap as a missing texture/HOM surface, and harmless hidden
        # textures are safer than sparse '-' placeholders for generated maps.
        front["texturebottom"] = front.get("texturebottom") if front.get("texturebottom") != "-" else WALL_TEXTURE
        back["texturebottom"] = back.get("texturebottom") if back.get("texturebottom") != "-" else WALL_TEXTURE
        front["texturetop"] = front.get("texturetop") if front.get("texturetop") != "-" else WALL_TEXTURE
        back["texturetop"] = back.get("texturetop") if back.get("texturetop") != "-" else WALL_TEXTURE

    def _apply_door_textures(self, line_id: int) -> None:
        line = self.linedefs[line_id]
        for key in ("sidefront", "sideback"):
            side_id = line.get(key, -1)
            if side_id >= 0:
                self.sidedefs[side_id]["texturemiddle"] = "-"
                self.sidedefs[side_id]["texturebottom"] = DOOR_TEXTURE
                self.sidedefs[side_id]["texturetop"] = DOOR_TEXTURE


def write_geometry_wad(
    document: dict[str, Any],
    room_boxes: dict[str, dict[str, int]],
    path: Path,
    *,
    cell_size: int = 128,
) -> dict[str, Any]:
    """Compile semantic rooms/connections to sector primitives and write a PWAD."""

    textmap, stats = build_geometry_textmap(document, room_boxes, cell_size=cell_size)
    _write_textmap_wad(Path(path), textmap)
    report = validate_wad(path)
    report.update(
        {
            "source_title": document.get("title"),
            "cell_size": cell_size,
            "geometry_profile": "sector_primitives_v0.3",
            "room_sectors": stats.room_sectors,
            "corridor_sectors": stats.corridor_sectors,
            "door_sectors": stats.door_sectors,
            "height_feature_sectors": stats.height_feature_sectors,
            "things_by_kind": stats.things_by_kind,
        }
    )
    return report


def build_geometry_textmap(
    document: dict[str, Any],
    room_boxes: dict[str, dict[str, int]],
    *,
    cell_size: int = 128,
) -> tuple[str, GeometryStats]:
    spaces = {space["id"]: space for space in document["spaces"]}
    rects = {space_id: _box_to_rect(box, cell_size) for space_id, box in room_boxes.items()}
    openings = _collect_openings(document["connections"], rects)
    builder = UdmfBuilder()
    stats = GeometryStats()
    room_sector_ids: dict[str, int] = {}

    for space_id, space in spaces.items():
        floor = int(space.get("floor_height", 0))
        ceiling = int(space.get("ceiling_height", floor + 128))
        materials = resolve_materials(document, space)
        sector_id = builder.add_sector(
            floor=floor,
            ceiling=ceiling,
            texturefloor=materials["floor"],
            textureceiling=materials["ceiling"],
        )
        room_sector_ids[space_id] = sector_id
        _add_room_boundary(builder, rects[space_id], sector_id, openings.get(space_id, []), materials)
        stats.room_sectors += 1

    door_tag = 100
    for connection in document["connections"]:
        source = str(connection["from"])
        target = str(connection["to"])
        if source not in rects or target not in rects:
            continue
        created = _add_connection(
            builder,
            rects[source],
            rects[target],
            connection,
            door_tag,
            resolve_materials(document, spaces[source]),
        )
        stats.corridor_sectors += created["corridors"]
        stats.door_sectors += created["doors"]
        door_tag += created["doors"]

    for space_id, space in spaces.items():
        stats.height_feature_sectors += _add_height_features(
            builder,
            space,
            rects[space_id],
            room_sector_ids[space_id],
            resolve_materials(document, space),
        )

    _add_progression_things(builder, document, rects, stats)
    _add_space_things(builder, document, rects, stats)
    _add_exit_special(builder, rects[str(document["progression"]["exit"])])
    return builder.textmap(), stats


def _collect_openings(connections: list[dict[str, Any]], rects: dict[str, Rect]) -> dict[str, list[Opening]]:
    openings: dict[str, list[Opening]] = {space_id: [] for space_id in rects}
    for connection in connections:
        source = str(connection["from"])
        target = str(connection["to"])
        if source not in rects or target not in rects:
            continue
        source_side, target_side = _connection_sides(rects[source], rects[target])
        openings[source].append(_opening_for_side(rects[source], source_side))
        openings[target].append(_opening_for_side(rects[target], target_side))
    return openings


def _connection_sides(source: Rect, target: Rect) -> tuple[Side, Side]:
    sx, sy = _center(source)
    tx, ty = _center(target)
    if abs(tx - sx) >= abs(ty - sy):
        return ("right", "left") if tx >= sx else ("left", "right")
    return ("bottom", "top") if ty <= sy else ("top", "bottom")


def _opening_for_side(rect: Rect, side: Side, width: int = 128) -> Opening:
    x1, y1, x2, y2 = rect
    cx, cy = _center(rect)
    half = width // 2
    if side in {"left", "right"}:
        return Opening(side=side, start=cy - half, end=cy + half)
    return Opening(side=side, start=cx - half, end=cx + half)


def _add_room_boundary(
    builder: UdmfBuilder,
    rect: Rect,
    sector: int,
    openings: list[Opening],
    materials: dict[str, str],
) -> None:
    x1, y1, x2, y2 = rect
    by_side = {side: [opening for opening in openings if opening.side == side] for side in ("left", "right", "top", "bottom")}
    _add_vertical_side(builder, x1, y1, y2, sector, by_side["left"], materials["wall"])
    _add_horizontal_side(builder, x1, x2, y2, sector, by_side["bottom"], materials["wall"])
    _add_vertical_side(builder, x2, y2, y1, sector, by_side["right"], materials["wall"])
    _add_horizontal_side(builder, x2, x1, y1, sector, by_side["top"], materials["wall"])


def _add_vertical_side(
    builder: UdmfBuilder,
    x: int,
    start_y: int,
    end_y: int,
    sector: int,
    openings: list[Opening],
    texture: str,
) -> None:
    for a, b in _split_axis(start_y, end_y, [(opening.start, opening.end) for opening in openings]):
        builder.add_line((x, a), (x, b), sector, texture=texture)


def _add_horizontal_side(
    builder: UdmfBuilder,
    start_x: int,
    end_x: int,
    y: int,
    sector: int,
    openings: list[Opening],
    texture: str,
) -> None:
    for a, b in _split_axis(start_x, end_x, [(opening.start, opening.end) for opening in openings]):
        builder.add_line((a, y), (b, y), sector, texture=texture)


def _split_axis(start: int, end: int, openings: list[tuple[int, int]]) -> list[tuple[int, int]]:
    direction = 1 if end >= start else -1
    lo, hi = sorted((start, end))
    clipped = sorted((max(lo, a), min(hi, b)) for a, b in openings if max(lo, a) < min(hi, b))
    points = {lo, hi}
    for a, b in clipped:
        points.add(a)
        points.add(b)
    ordered = sorted(points)
    segments = [(a, b) for a, b in zip(ordered, ordered[1:]) if a != b]
    if direction < 0:
        return [(b, a) for a, b in reversed(segments)]
    return segments


def _add_connection(
    builder: UdmfBuilder,
    source: Rect,
    target: Rect,
    connection: dict[str, Any],
    door_tag: int,
    materials: dict[str, str],
) -> dict[str, int]:
    source_side, target_side = _connection_sides(source, target)
    sx, sy = _opening_center(source, source_side)
    tx, ty = _opening_center(target, target_side)
    floor = 0
    is_door = connection.get("type") in {"door", "locked_door", "keyed_door"}
    lock_color = str(connection.get("key")) if connection.get("type") in {"locked_door", "keyed_door"} else None
    corridors = 0
    doors = 0

    if source_side in {"left", "right"} and sy == ty:
        x_start, x_end = sorted((sx, tx))
        if is_door:
            door_width = 128
            door_x1 = (x_start + x_end - door_width) // 2
            door_x2 = door_x1 + door_width
            if door_x1 > x_start:
                _add_corridor_rect(builder, (x_start, sy - 64, door_x1, sy + 64), floor, materials)
                corridors += 1
            _add_door_rect(builder, (door_x1, sy - 64, door_x2, sy + 64), floor, door_tag, lock_color, materials)
            doors += 1
            if door_x2 < x_end:
                _add_corridor_rect(builder, (door_x2, sy - 64, x_end, sy + 64), floor, materials)
                corridors += 1
        else:
            _add_corridor_rect(builder, (x_start, sy - 64, x_end, sy + 64), floor, materials)
            corridors += 1
    elif source_side in {"top", "bottom"} and sx == tx:
        y_start, y_end = sorted((sy, ty))
        if is_door:
            door_height = 128
            door_y1 = (y_start + y_end - door_height) // 2
            door_y2 = door_y1 + door_height
            if door_y1 > y_start:
                _add_corridor_rect(builder, (sx - 64, y_start, sx + 64, door_y1), floor, materials)
                corridors += 1
            _add_door_rect(builder, (sx - 64, door_y1, sx + 64, door_y2), floor, door_tag, lock_color, materials)
            doors += 1
            if door_y2 < y_end:
                _add_corridor_rect(builder, (sx - 64, door_y2, sx + 64, y_end), floor, materials)
                corridors += 1
        else:
            _add_corridor_rect(builder, (sx - 64, y_start, sx + 64, y_end), floor, materials)
            corridors += 1
    else:
        # L-shaped fallback for future less-regular room placement.
        mid = (tx, sy)
        _add_corridor_rect(builder, _rect_between_points((sx, sy), mid), floor, materials)
        _add_corridor_rect(builder, _rect_between_points(mid, (tx, ty)), floor, materials)
        corridors += 2
    return {"corridors": corridors, "doors": doors}


def _add_corridor_rect(builder: UdmfBuilder, rect: Rect, floor: int, materials: dict[str, str]) -> int:
    sector = builder.add_sector(
        floor=floor,
        ceiling=floor + 128,
        texturefloor=materials["floor"],
        textureceiling=materials["ceiling"],
    )
    _add_plain_rect(builder, rect, sector, texture=materials["wall"])
    return sector


def _add_door_rect(
    builder: UdmfBuilder,
    rect: Rect,
    floor: int,
    tag: int,
    lock_color: str | None,
    materials: dict[str, str],
) -> int:
    sector = builder.add_sector(
        floor=floor,
        ceiling=floor,
        texturefloor=materials["floor"],
        textureceiling=materials["ceiling"],
        tag=tag,
    )
    _add_plain_rect(builder, rect, sector, texture=materials["door_track"])
    special = {
        "special": 12,
        "playeruse": True,
        "playeruseback": True,
        "repeatspecial": True,
        "arg0": tag,
        "arg1": 16,
        "arg2": 150,
        "arg3": 0,
    }
    if lock_color:
        special["locknumber"] = DOOR_LOCK_NUMBERS[lock_color]
    # Mark every portal touching this door sector as an usable door linedef.
    for index, line in enumerate(builder.linedefs):
        if line.get("sideback", -1) < 0:
            continue
        sectors = {
            builder.sidedefs[line["sidefront"]]["sector"],
            builder.sidedefs[line["sideback"]]["sector"],
        }
        if sector in sectors:
            line.update(special)
            builder._apply_door_textures(index)
    return sector


def _add_plain_rect(builder: UdmfBuilder, rect: Rect, sector: int, *, texture: str = WALL_TEXTURE) -> None:
    x1, y1, x2, y2 = rect
    # Doom renders a one-sided line's front sidedef on the right side of
    # the linedef. Emit rectangles clockwise so their visible wall faces point
    # inward; otherwise corridor/door side walls show as hall-of-mirrors from
    # inside the map.
    corners = [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    for a, b in zip(corners, corners[1:] + corners[:1], strict=True):
        builder.add_line(a, b, sector, texture=texture)


def _add_height_features(
    builder: UdmfBuilder,
    space: dict[str, Any],
    rect: Rect,
    parent_sector: int,
    materials: dict[str, str],
) -> int:
    features = space.get("height_topology", {}).get("features", [])
    if not isinstance(features, list):
        return 0
    count = 0
    base = int(space.get("floor_height", 0))
    ceiling = int(space.get("ceiling_height", base + 128))
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        feature_type = feature.get("type")
        if feature_type in {"ceiling_pillar_cluster", "pillar", "pillar_cluster"}:
            requested = int(feature.get("count", 1))
            for pillar_rect in _pillar_rects(rect, requested):
                sector = builder.add_sector(
                    floor=int(feature.get("floor_height", ceiling)),
                    ceiling=int(feature.get("ceiling_height", ceiling)),
                    texturefloor=materials["floor"],
                    textureceiling=materials["ceiling"],
                )
                builder.add_inner_rect(pillar_rect, sector, parent_sector, texture=materials["lower_wall"])
                count += 1
        elif feature_type in {"raised_platform", "platform"}:
            sector = builder.add_sector(
                floor=int(feature.get("floor_height", base + 32)),
                ceiling=ceiling,
                texturefloor=materials["floor"],
                textureceiling=materials["ceiling"],
            )
            builder.add_inner_rect(_feature_rect(rect, "north", index), sector, parent_sector, texture=materials["lower_wall"])
            count += 1
        elif feature_type in {"shallow_pit", "pit"}:
            sector = builder.add_sector(
                floor=int(feature.get("floor_height", base - 24)),
                ceiling=ceiling,
                texturefloor=materials["floor"],
                textureceiling=materials["ceiling"],
            )
            builder.add_inner_rect(_feature_rect(rect, "south", index), sector, parent_sector, texture=materials["lower_wall"])
            count += 1
    return count


def _pillar_rects(rect: Rect, count: int) -> list[Rect]:
    x1, y1, x2, y2 = rect
    cx, cy = _center(rect)
    size = 96
    offsets = [(-160, 160), (160, 160), (-160, -160), (160, -160), (0, 0)]
    result = []
    for dx, dy in offsets[: max(0, min(count, len(offsets)))]:
        px = cx + dx
        py = cy + dy
        if x1 + 128 < px < x2 - 128 and y1 + 128 < py < y2 - 128:
            result.append((px - size // 2, py - size // 2, px + size // 2, py + size // 2))
    return result


def _feature_rect(rect: Rect, placement: str, index: int) -> Rect:
    x1, y1, x2, y2 = rect
    width = min(256, max(128, (x2 - x1) // 3))
    height = min(192, max(128, (y2 - y1) // 4))
    if placement == "north":
        return (x2 - width - 128, y2 - height - 128 - index * 16, x2 - 128, y2 - 128 - index * 16)
    return (x1 + 128, y1 + 128 + index * 16, x1 + 128 + width, y1 + 128 + height + index * 16)


def _add_progression_things(builder: UdmfBuilder, document: dict[str, Any], rects: dict[str, Rect], stats: GeometryStats) -> None:
    start_room = str(document["progression"].get("start"))
    exit_room = str(document["progression"].get("exit"))
    if start_room in rects:
        x, y = _center(rects[start_room])
        builder.add_thing(x, y, "player_start")
        _count_thing(stats, "player_start")
    for key in document["progression"].get("keys", []):
        if not isinstance(key, dict):
            continue
        location = str(key.get("location"))
        if location in rects:
            x, y = _safe_room_point(rects[location], 0)
            builder.add_thing(x, y, "key", color=str(key.get("color", "blue")))
            _count_thing(stats, "key")
    if exit_room in rects:
        # The exit switch is a line special, but count the intent for reports.
        stats.things_by_kind["exit_switch"] = stats.things_by_kind.get("exit_switch", 0) + 1


def _add_space_things(builder: UdmfBuilder, document: dict[str, Any], rects: dict[str, Rect], stats: GeometryStats) -> None:
    for space in document["spaces"]:
        rect = rects[space["id"]]
        offset = 1
        for item in _extract_items(space):
            kind = _item_kind(item)
            if kind:
                x, y = _safe_room_point(rect, offset)
                builder.add_thing(x, y, kind)
                _count_thing(stats, kind)
                offset += 1
        for _name, monster_count in _extract_monsters(space):
            for _ in range(min(monster_count, 8)):
                x, y = _safe_room_point(rect, offset)
                builder.add_thing(x, y, "monster")
                _count_thing(stats, "monster")
                offset += 1


def _add_exit_special(builder: UdmfBuilder, exit_rect: Rect) -> None:
    x1, _y1, x2, y2 = exit_rect
    exit_mid = ((x1 + x2) // 2, y2)
    best_line_id = None
    best_distance = None
    for index, line in enumerate(builder.linedefs):
        if line.get("sideback", -1) >= 0:
            continue
        v1 = builder.vertices[line["v1"]]
        v2 = builder.vertices[line["v2"]]
        midpoint = ((v1["x"] + v2["x"]) // 2, (v1["y"] + v2["y"]) // 2)
        distance = abs(midpoint[0] - exit_mid[0]) + abs(midpoint[1] - exit_mid[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_line_id = index
    if best_line_id is None:
        raise ValueError("could not find exit boundary line")
    line = builder.linedefs[best_line_id]
    line.update(special=243, playeruse=True, repeatspecial=False, arg0=0)
    side = builder.sidedefs[line["sidefront"]]
    side["texturemiddle"] = EXIT_TEXTURE
    side["texturebottom"] = EXIT_TEXTURE
    side["texturetop"] = EXIT_TEXTURE


def _extract_items(space: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("things", "contains"):
        value = space.get(key, {})
        if isinstance(value, dict) and isinstance(value.get("items"), list):
            items.extend(str(item) for item in value["items"])
    return items


def _extract_monsters(space: dict[str, Any]) -> list[tuple[str, int]]:
    encounter = space.get("encounter", {})
    monsters = encounter.get("monsters", []) if isinstance(encounter, dict) else []
    result = []
    if isinstance(monsters, list):
        for entry in monsters:
            if isinstance(entry, dict):
                for name, count in entry.items():
                    result.append((str(name), int(count)))
    return result


def _item_kind(item: str) -> str | None:
    lowered = item.lower()
    if "shotgun" in lowered or "chaingun" in lowered or "weapon" in lowered:
        return "weapon"
    if "shell" in lowered or "ammo" in lowered or "clip" in lowered:
        return "ammo"
    if "health" in lowered or "stim" in lowered or "med" in lowered:
        return "health"
    return None


def _count_thing(stats: GeometryStats, kind: str) -> None:
    stats.things_by_kind[kind] = stats.things_by_kind.get(kind, 0) + 1


def _safe_room_point(rect: Rect, offset: int) -> Point:
    x1, y1, x2, y2 = rect
    cols = max(1, (x2 - x1 - 256) // 96)
    col = offset % cols
    row = offset // cols
    x = min(x2 - 128, x1 + 128 + col * 96)
    y = min(y2 - 128, y1 + 128 + row * 96)
    return x, y


def _box_to_rect(box: dict[str, int], cell_size: int) -> Rect:
    x1 = box["c"] * cell_size
    y2 = -box["r"] * cell_size
    x2 = (box["c"] + box["w"]) * cell_size
    y1 = -(box["r"] + box["h"]) * cell_size
    return x1, y1, x2, y2


def _center(rect: Rect) -> Point:
    x1, y1, x2, y2 = rect
    return (x1 + x2) // 2, (y1 + y2) // 2


def _opening_center(rect: Rect, side: Side) -> Point:
    x1, y1, x2, y2 = rect
    cx, cy = _center(rect)
    if side == "left":
        return x1, cy
    if side == "right":
        return x2, cy
    if side == "top":
        return cx, y2
    return cx, y1


def _rect_between_points(a: Point, b: Point, half_width: int = 64) -> Rect:
    ax, ay = a
    bx, by = b
    if ay == by:
        return min(ax, bx), ay - half_width, max(ax, bx), ay + half_width
    return ax - half_width, min(ay, by), ax + half_width, max(ay, by)


def _block(kind: str, fields: dict[str, Any]) -> str:
    return kind + "\n{\n" + "".join(
        f"    {key} = {json.dumps(value)};\n" for key, value in fields.items()
    ) + "}\n"
