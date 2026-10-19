"""Sector-primitive geometry compiler for romeroDSL.

This path keeps the high-level DSL as the source of truth, but exports cleaner
UDMF than the debug cell-grid writer. Rooms become main sectors, connections
become corridor/door sectors, and intra-room height features become subsectors
inside the room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from romerodsl.materials import resolve_materials
from romerodsl.schema import canonical_orientation, canonical_room_shape
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
    "bfg": 2006,
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

    def add_inner_rect_cluster(self, rects: list[tuple[Rect, int]], parent_sector: int, *, texture: str = WALL_TEXTURE) -> None:
        """Emit touching inner sectors while assigning shared edges to both feature sectors.

        Isolated inner rectangles use parent-sector backs on every edge. Stair
        strips need adjacent feature sectors to share a real two-sided linedef;
        otherwise duplicated parent-backed edges can create odd visual seams and
        movement artifacts.
        """

        local_edges: dict[tuple[int, int], int] = {}
        for rect, sector in rects:
            x1, y1, x2, y2 = rect
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
                va = self._vertex(a)
                vb = self._vertex(b)
                edge = (min(va, vb), max(va, vb))
                if edge in local_edges:
                    line_id = local_edges[edge]
                    line = self.linedefs[line_id]
                    line.update(sideback=front_side, twosided=True, blocking=False)
                    self._apply_two_sided_textures(line_id)
                    continue

                back_side = len(self.sidedefs)
                self.sidedefs.append(
                    {
                        "sector": parent_sector,
                        "texturemiddle": "-",
                        "texturebottom": texture,
                        "texturetop": texture,
                    }
                )
                line_id = len(self.linedefs)
                local_edges[edge] = line_id
                self.linedefs.append(
                    {
                        "v1": va,
                        "v2": vb,
                        "sidefront": front_side,
                        "sideback": back_side,
                        "blocking": False,
                        "twosided": True,
                    }
                )
                self._apply_two_sided_textures(line_id)

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
        self._compact_sidedefs()
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

    def _compact_sidedefs(self) -> None:
        used_ids: list[int] = []
        seen: set[int] = set()
        for line in self.linedefs:
            for side_key in ("sidefront", "sideback"):
                side_id = line.get(side_key, -1)
                if side_id >= 0 and side_id not in seen:
                    used_ids.append(side_id)
                    seen.add(side_id)
        remap = {old_id: new_id for new_id, old_id in enumerate(used_ids)}
        if len(remap) == len(self.sidedefs):
            return
        self.sidedefs = [self.sidedefs[old_id] for old_id in used_ids]
        for line in self.linedefs:
            line["sidefront"] = remap[line["sidefront"]]
            if line.get("sideback", -1) >= 0:
                line["sideback"] = remap[line["sideback"]]


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
            "geometry_profile": "sector_primitives_v0.5",
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
    shapes = {}
    for space_id, space in spaces.items():
        raw_shape = space.get("shape", "rectangle")
        shape = canonical_room_shape(raw_shape)
        if shape is None:
            raise ValueError(f"unsupported room shape {raw_shape!r} in space {space_id!r}")
        shapes[space_id] = shape
    orientations = {}
    for space_id, space in spaces.items():
        raw_orientation = space.get("orientation", 0)
        orientation = canonical_orientation(raw_orientation)
        if orientation is None:
            raise ValueError(f"unsupported orientation {raw_orientation!r} in space {space_id!r}")
        orientations[space_id] = orientation
    openings = _collect_openings(document["connections"], rects, shapes, orientations)
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
        _add_room_boundary(
            builder,
            rects[space_id],
            sector_id,
            openings.get(space_id, []),
            materials,
            shape=shapes[space_id],
            orientation=orientations[space_id],
        )
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
            source_shape=shapes[source],
            target_shape=shapes[target],
            source_orientation=orientations[source],
            target_orientation=orientations[target],
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


def _collect_openings(
    connections: list[dict[str, Any]],
    rects: dict[str, Rect],
    shapes: dict[str, str] | None = None,
    orientations: dict[str, int] | None = None,
) -> dict[str, list[Opening]]:
    shapes = shapes or {}
    orientations = orientations or {}
    openings: dict[str, list[Opening]] = {space_id: [] for space_id in rects}
    for connection in connections:
        source = str(connection["from"])
        target = str(connection["to"])
        if source not in rects or target not in rects:
            continue
        _source_side, _target_side, _source_point, _target_point, source_opening, target_opening = _connection_portals(
            rects[source],
            rects[target],
            source_shape=shapes.get(source, "rectangle"),
            target_shape=shapes.get(target, "rectangle"),
            source_orientation=orientations.get(source, 0),
            target_orientation=orientations.get(target, 0),
        )
        openings[source].append(source_opening)
        openings[target].append(target_opening)
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


def _connection_portals(
    source: Rect,
    target: Rect,
    width: int = 128,
    *,
    source_shape: str = "rectangle",
    target_shape: str = "rectangle",
    source_orientation: int = 0,
    target_orientation: int = 0,
) -> tuple[Side, Side, Point, Point, Opening, Opening]:
    source_side, target_side = _connection_sides(source, target)
    source_opening = _opening_for_shape(source, source_side, source_shape, width, source_orientation)
    target_opening = _opening_for_shape(target, target_side, target_shape, width, target_orientation)
    source_axis = _opening_axis(source_opening)
    target_axis = _opening_axis(target_opening)
    if source_opening.start <= target_opening.end and target_opening.start <= source_opening.end:
        aligned = max(
            max(source_opening.start, target_opening.start),
            min(min(source_opening.end, target_opening.end), (source_axis + target_axis) // 2),
        )
        source_opening = _opening_at_axis(source_opening, aligned)
        target_opening = _opening_at_axis(target_opening, aligned)
        source_axis = target_axis = aligned
    else:
        # When openings do not overlap, keep the L-shaped route attached to
        # the nearest valid edge of each portal. Leaving both axes at their
        # independent centers makes the corridor side run through the room
        # boundary instead of through the declared doorway opening.
        portal_width = min(width, 64)
        target_half = max(1, portal_width // 2)
        target_axis = max(
            target_opening.start + target_half,
            min(target_opening.end - target_half, source_axis),
        )
        source_half = max(1, portal_width // 2)
        source_axis = max(
            source_opening.start + source_half,
            min(source_opening.end - source_half, target_axis),
        )
        source_opening = _opening_at_axis(source_opening, source_axis, portal_width)
        target_opening = _opening_at_axis(target_opening, target_axis, portal_width)
        source_axis = _opening_axis(source_opening)
        target_axis = _opening_axis(target_opening)

    if source_side in {"left", "right"}:
        source_point = (source[0] if source_side == "left" else source[2], source_axis)
        target_point = (target[0] if target_side == "left" else target[2], target_axis)
    else:
        source_point = (source_axis, source[3] if source_side == "top" else source[1])
        target_point = (target_axis, target[3] if target_side == "top" else target[1])
    return source_side, target_side, source_point, target_point, source_opening, target_opening


def _opening_for_shape(
    rect: Rect,
    side: Side,
    shape: str,
    width: int = 128,
    orientation: int = 0,
) -> Opening:
    intervals = _shape_side_intervals(rect, shape, side, orientation)
    if not intervals:
        return _opening_for_side(rect, side, width)
    desired = _axis_center(rect[1], rect[3]) if side in {"left", "right"} else _axis_center(rect[0], rect[2])
    interval = min(intervals, key=lambda bounds: 0 if bounds[0] <= desired <= bounds[1] else min(abs(desired - bounds[0]), abs(desired - bounds[1])))
    lo, hi = interval
    opening_width = min(width, hi - lo)
    half = opening_width // 2
    center = max(lo + half, min(hi - half, desired))
    return Opening(side=side, start=center - half, end=center + half)


def _opening_at_axis(opening: Opening, axis: int, width: int | None = None) -> Opening:
    span = opening.end - opening.start
    half = max(1, (min(span, width) if width is not None else span) // 2)
    center = max(opening.start + half, min(opening.end - half, axis))
    return Opening(opening.side, center - half, center + half)


def _opening_axis(opening: Opening) -> int:
    return (opening.start + opening.end) // 2


def _shape_side_intervals(rect: Rect, shape: str, side: Side, orientation: int = 0) -> list[tuple[int, int]]:
    points = _room_boundary_points(rect, shape, orientation)
    x1, y1, x2, y2 = rect
    intervals: list[tuple[int, int]] = []
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        sx, sy = start
        ex, ey = end
        if side == "left" and sx == ex == x1:
            lo, hi = sorted((sy, ey))
            intervals.append((lo, hi))
        elif side == "right" and sx == ex == x2:
            lo, hi = sorted((sy, ey))
            intervals.append((lo, hi))
        elif side == "top" and sy == ey == y2:
            lo, hi = sorted((sx, ex))
            intervals.append((lo, hi))
        elif side == "bottom" and sy == ey == y1:
            lo, hi = sorted((sx, ex))
            intervals.append((lo, hi))
    return [(lo, hi) for lo, hi in intervals if hi - lo >= 64]


def _aligned_axis_position(source_lo: int, source_hi: int, target_lo: int, target_hi: int, half_width: int) -> int:
    lo = max(source_lo + half_width, target_lo + half_width)
    hi = min(source_hi - half_width, target_hi - half_width)
    desired = (_axis_center(source_lo, source_hi) + _axis_center(target_lo, target_hi)) // 2
    if lo <= hi:
        return max(lo, min(hi, desired))
    # No perpendicular overlap. Use the midpoint between the two room centers;
    # the L-shaped fallback will route outside the rooms.
    return desired


def _axis_center(lo: int, hi: int) -> int:
    return (lo + hi) // 2


def _add_room_boundary(
    builder: UdmfBuilder,
    rect: Rect,
    sector: int,
    openings: list[Opening],
    materials: dict[str, str],
    *,
    shape: str = "rectangle",
    orientation: int = 0,
) -> None:
    by_side = {side: [opening for opening in openings if opening.side == side] for side in ("left", "right", "top", "bottom")}
    points = _room_boundary_points(rect, shape, orientation)
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        side = _boundary_side(start, end, rect)
        segment_openings = by_side.get(side, []) if side else []
        if start[0] == end[0]:
            _add_vertical_side(builder, start[0], start[1], end[1], sector, segment_openings, materials["wall"])
        elif start[1] == end[1]:
            _add_horizontal_side(builder, start[0], end[0], start[1], sector, segment_openings, materials["wall"])
        else:
            builder.add_line(start, end, sector, texture=materials["wall"])


def _boundary_side(start: Point, end: Point, rect: Rect) -> Side | None:
    x1, y1, x2, y2 = rect
    if start[0] == end[0] == x1:
        return "left"
    if start[0] == end[0] == x2:
        return "right"
    if start[1] == end[1] == y1:
        return "bottom"
    if start[1] == end[1] == y2:
        return "top"
    return None


def _clockwise(points: list[Point]) -> list[Point]:
    area = sum(
        start[0] * end[1] - end[0] * start[1]
        for start, end in zip(points, points[1:] + points[:1], strict=True)
    )
    return list(reversed(points)) if area > 0 else points


def _canonical_room_boundary_points(rect: Rect, shape: str) -> list[Point]:
    x1, y1, x2, y2 = rect
    canonical = canonical_room_shape(shape)
    if canonical is None:
        raise ValueError(f"unsupported room shape {shape!r}")
    if canonical in {"rectangle", "raised_center", "depressed_floor"}:
        return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    if canonical == "rectangle_with_side_alcoves":
        bevel = min(128, max(64, min(x2 - x1, y2 - y1) // 8))
        side_bevel = min(192, bevel + 32)
        return _clockwise([
            (x1, y1 + bevel),
            (x1, y2 - bevel),
            (x1 + bevel, y2),
            (x2 - side_bevel, y2),
            (x2, y2 - side_bevel),
            (x2, y1 + side_bevel),
            (x2 - side_bevel, y1),
            (x1 + bevel, y1),
        ])
    if canonical == "closet":
        closet_depth = min(256, max(160, (y2 - y1) // 3))
        closet_width = min(320, max(192, (x2 - x1) // 2))
        center = (x1 + x2) // 2
        neck_left = center - closet_width // 2
        neck_right = center + closet_width // 2
        body_top = y2 - closet_depth
        return _clockwise([
            (x1, y1),
            (x2, y1),
            (x2, body_top),
            (neck_right, body_top),
            (neck_right, y2),
            (neck_left, y2),
            (neck_left, body_top),
            (x1, body_top),
        ])
    if canonical == "bent_corridor":
        bend = min(256, max(128, (y2 - y1) // 4))
        return _clockwise([
            (x1, y1 + bend),
            (x1 + bend, y1),
            (x2 - bend // 2, y1),
            (x2, y1 + bend // 2),
            (x2 - bend, y2),
            (x1 + bend // 2, y2),
            (x1, y2 - bend // 2),
        ])
    if canonical == "narrow_hall_with_cover":
        half_width = min(128, max(64, (y2 - y1) // 8))
        center = (y1 + y2) // 2
        low, high = center - half_width, center + half_width
        bay = min(192, max(128, (x2 - x1) // 6))
        return _clockwise([
            (x1, low),
            (x1 + bay, low),
            (x1 + bay, y1),
            (x2 - bay, y1),
            (x2 - bay, low),
            (x2, low),
            (x2, high),
            (x2 - bay, high),
            (x2 - bay, y2),
            (x1 + bay, y2),
            (x1 + bay, high),
            (x1, high),
        ])
    if canonical == "l_shaped":
        notch_width = max(192, (x2 - x1) // 3)
        notch_depth = max(192, (y2 - y1) // 3)
        return [
            (x1, y1),
            (x1, y2),
            (x2 - notch_width, y2),
            (x2 - notch_width, y2 - notch_depth),
            (x2, y2 - notch_depth),
            (x2, y1),
        ]
    if canonical == "t_shaped":
        bar_height = max(192, (y2 - y1) // 3)
        stem_width = max(192, (x2 - x1) // 3)
        cx = (x1 + x2) // 2
        return [
            (x1, y1),
            (x1, y2),
            (x2, y2),
            (x2, y1 + bar_height),
            (cx + stem_width // 2, y1 + bar_height),
            (cx + stem_width // 2, y1),
            (cx - stem_width // 2, y1),
            (cx - stem_width // 2, y1 + bar_height),
            (x1, y1 + bar_height),
        ]
    if canonical in {"irregular_orthogonal", "inset_chamber"}:
        depth = min(128, max(64, (y2 - y1) // 7))
        first = x1 + max(128, (x2 - x1) // 5)
        second = x1 + max(first + 128, (x2 - x1) * 3 // 5)
        return [
            (x1, y1),
            (x1, y2),
            (first, y2),
            (first, y2 - depth),
            (second, y2 - depth),
            (second, y2),
            (x2, y2),
            (x2, y1),
        ]
    if canonical == "irregular_octagonal":
        chamfer = min(160, max(96, min(x2 - x1, y2 - y1) // 5))
        return [
            (x1, y1 + chamfer),
            (x1, y2 - chamfer),
            (x1 + chamfer, y2),
            (x2 - chamfer, y2),
            (x2, y2 - chamfer),
            (x2, y1 + chamfer),
            (x2 - chamfer, y1),
            (x1 + chamfer, y1),
        ]
    raise ValueError(f"unsupported room shape {shape!r}")


def _room_boundary_points(rect: Rect, shape: str, orientation: int = 0) -> list[Point]:
    points = _canonical_room_boundary_points(rect, shape)
    turn = orientation % 360
    if turn not in {0, 90, 180, 270}:
        raise ValueError(f"orientation must be a quarter turn, got {orientation!r}")
    if turn == 0:
        return points
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    rotated: list[Point] = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        if turn == 90:
            rotated.append((cx - dy, cy + dx))
        elif turn == 180:
            rotated.append((cx - dx, cy - dy))
        else:
            rotated.append((cx + dy, cy - dx))
    min_x = min(x for x, _y in rotated)
    max_x = max(x for x, _y in rotated)
    min_y = min(y for _x, y in rotated)
    max_y = max(y for _x, y in rotated)
    span_x = max(1, max_x - min_x)
    span_y = max(1, max_y - min_y)
    return [
        (
            rect[0] + round((x - min_x) * (rect[2] - rect[0]) / span_x),
            rect[1] + round((y - min_y) * (rect[3] - rect[1]) / span_y),
        )
        for x, y in rotated
    ]


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
    *,
    source_shape: str = "rectangle",
    target_shape: str = "rectangle",
    source_orientation: int = 0,
    target_orientation: int = 0,
) -> dict[str, int]:
    source_side, _target_side, source_point, target_point, _source_opening, _target_opening = _connection_portals(
        source,
        target,
        source_shape=source_shape,
        target_shape=target_shape,
        source_orientation=source_orientation,
        target_orientation=target_orientation,
    )
    sx, sy = source_point
    tx, ty = target_point
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
            door_sector = _add_door_rect(
                builder, (door_x1, sy - 64, door_x2, sy + 64), floor, door_tag, lock_color, materials
            )
            doors += 1
            if door_x2 < x_end:
                _add_corridor_rect(builder, (door_x2, sy - 64, x_end, sy + 64), floor, materials)
                corridors += 1
            _mark_door_portals(builder, door_sector, door_tag, lock_color)
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
            door_sector = _add_door_rect(
                builder, (sx - 64, door_y1, sx + 64, door_y2), floor, door_tag, lock_color, materials
            )
            doors += 1
            if door_y2 < y_end:
                _add_corridor_rect(builder, (sx - 64, door_y2, sx + 64, y_end), floor, materials)
                corridors += 1
            _mark_door_portals(builder, door_sector, door_tag, lock_color)
        else:
            _add_corridor_rect(builder, (sx - 64, y_start, sx + 64, y_end), floor, materials)
            corridors += 1
    else:
        # L-shaped fallback for rooms whose side openings do not overlap. Put the
        # actual door on one leg instead of silently downgrading a locked-door
        # connection into ordinary corridor sectors.
        mid = (tx, sy)
        first = _add_connection_segment(
            builder, source_point, mid, False, door_tag, lock_color, materials
        )
        second = _add_connection_segment(
            builder, mid, target_point, is_door, door_tag, lock_color, materials
        )
        if is_door and second["doors"] == 0:
            first = _add_connection_segment(
                builder, source_point, mid, True, door_tag, lock_color, materials
            )
        corridors += first["corridors"] + second["corridors"]
        doors += first["doors"] + second["doors"]
    return {"corridors": corridors, "doors": doors}


def _add_connection_segment(
    builder: UdmfBuilder,
    start: Point,
    end: Point,
    is_door: bool,
    door_tag: int,
    lock_color: str | None,
    materials: dict[str, str],
) -> dict[str, int]:
    sx, sy = start
    tx, ty = end
    if sx == tx and sy == ty:
        return {"corridors": 0, "doors": 0}
    if sx != tx and sy != ty:
        raise ValueError("connection segment must be axis-aligned")

    corridors = 0
    doors = 0
    if sy == ty:
        x_start, x_end = sorted((sx, tx))
        if is_door:
            door_width = min(128, x_end - x_start)
            door_x1 = (x_start + x_end - door_width) // 2
            door_x2 = door_x1 + door_width
            if door_x1 > x_start:
                _add_corridor_rect(builder, (x_start, sy - 64, door_x1, sy + 64), 0, materials)
                corridors += 1
            door_sector = _add_door_rect(
                builder, (door_x1, sy - 64, door_x2, sy + 64), 0, door_tag, lock_color, materials
            )
            doors += 1
            if door_x2 < x_end:
                _add_corridor_rect(builder, (door_x2, sy - 64, x_end, sy + 64), 0, materials)
                corridors += 1
            _mark_door_portals(builder, door_sector, door_tag, lock_color)
        else:
            _add_corridor_rect(builder, (x_start, sy - 64, x_end, sy + 64), 0, materials)
            corridors += 1
    else:
        y_start, y_end = sorted((sy, ty))
        if is_door:
            door_height = min(128, y_end - y_start)
            door_y1 = (y_start + y_end - door_height) // 2
            door_y2 = door_y1 + door_height
            if door_y1 > y_start:
                _add_corridor_rect(builder, (sx - 64, y_start, sx + 64, door_y1), 0, materials)
                corridors += 1
            door_sector = _add_door_rect(
                builder, (sx - 64, door_y1, sx + 64, door_y2), 0, door_tag, lock_color, materials
            )
            doors += 1
            if door_y2 < y_end:
                _add_corridor_rect(builder, (sx - 64, door_y2, sx + 64, y_end), 0, materials)
                corridors += 1
            _mark_door_portals(builder, door_sector, door_tag, lock_color)
        else:
            _add_corridor_rect(builder, (sx - 64, y_start, sx + 64, y_end), 0, materials)
            corridors += 1
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
    _mark_door_portals(builder, sector, tag, lock_color)
    return sector


def _door_special(tag: int, lock_color: str | None) -> dict[str, Any]:
    special: dict[str, Any] = {
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
    return special


def _mark_door_portals(builder: UdmfBuilder, sector: int, tag: int, lock_color: str | None) -> None:
    special = _door_special(tag, lock_color)
    # Mark every portal touching this door sector as a usable door linedef.
    # Zandronum reliably activates Door_Raise from the front side; do not rely
    # on back-side activation flags for the far face of a generated door.
    for index, line in enumerate(builder.linedefs):
        if line.get("sideback", -1) < 0:
            continue
        front_sector = builder.sidedefs[line["sidefront"]]["sector"]
        back_sector = builder.sidedefs[line["sideback"]]["sector"]
        if sector not in {front_sector, back_sector}:
            continue
        if front_sector == sector:
            line["sidefront"], line["sideback"] = line["sideback"], line["sidefront"]
            line["v1"], line["v2"] = line["v2"], line["v1"]
        line.update(special)
        builder._apply_door_textures(index)


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
    features = list(features)
    shape = canonical_room_shape(space.get("shape", "rectangle"))
    if not features and shape in {"raised_center", "depressed_floor"}:
        features.append(
            {
                "type": "raised_central_structure" if shape == "raised_center" else "depressed_floor",
                "floor_height": int(space.get("floor_height", 0)) + (32 if shape == "raised_center" else -24),
                "placement": "central",
            }
        )
    elif not features and shape == "rectangle_with_side_alcoves":
        features = [
            {"type": "side_alcove", "placement": "west_alcove"},
            {"type": "side_alcove", "placement": "east_alcove"},
        ]
    elif not features and shape == "narrow_hall_with_cover":
        features = [
            {"type": "cover_block", "placement": "west_cover"},
            {"type": "cover_block", "placement": "east_cover"},
        ]
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
        elif feature_type == "side_alcove":
            x1, y1, x2, y2 = rect
            alcove_width = min(192, max(128, (x2 - x1) // 6))
            alcove_height = min(256, max(160, (y2 - y1) // 3))
            center_y = (y1 + y2) // 2
            if feature.get("placement") == "east_alcove":
                alcove_rect = (x2 - alcove_width - 96, center_y - alcove_height // 2, x2 - 96, center_y + alcove_height // 2)
            else:
                alcove_rect = (x1 + 96, center_y - alcove_height // 2, x1 + alcove_width + 96, center_y + alcove_height // 2)
            sector = builder.add_sector(
                floor=base,
                ceiling=max(base + 96, ceiling - 32),
                texturefloor=materials["floor"],
                textureceiling=materials["ceiling"],
            )
            builder.add_inner_rect(alcove_rect, sector, parent_sector, texture=materials["lower_wall"])
            count += 1
        elif feature_type == "cover_block":
            cx, cy = _center(rect)
            cover_width = 96
            cover_height = 96
            cover_x = cx + (-192 if index % 2 == 0 else 192)
            cover_rect = (
                cover_x - cover_width // 2,
                cy - cover_height // 2,
                cover_x + cover_width // 2,
                cy + cover_height // 2,
            )
            sector = builder.add_sector(
                floor=base,
                ceiling=min(ceiling, base + 64),
                texturefloor=materials["floor"],
                textureceiling=materials["ceiling"],
            )
            builder.add_inner_rect(cover_rect, sector, parent_sector, texture=materials["lower_wall"])
            count += 1
        elif feature_type in {"raised_platform", "platform"}:
            platform_floor = int(feature.get("floor_height", base + 32))
            feature_rect = _feature_rect(rect, "north", index)
            if feature.get("access") == "stairs":
                count += _add_stair_feature(
                    builder,
                    feature_rect,
                    parent_sector,
                    base,
                    platform_floor,
                    ceiling,
                    "south_to_north",
                    materials,
                )
            else:
                sector = builder.add_sector(
                    floor=platform_floor,
                    ceiling=ceiling,
                    texturefloor=materials["floor"],
                    textureceiling=materials["ceiling"],
                )
                builder.add_inner_rect(feature_rect, sector, parent_sector, texture=materials["lower_wall"])
                count += 1
        elif feature_type in {"shallow_pit", "pit"}:
            pit_floor = int(feature.get("floor_height", base - 24))
            feature_rect = _feature_rect(rect, "south", index)
            if feature.get("escape") == "stairs":
                count += _add_stair_feature(
                    builder,
                    feature_rect,
                    parent_sector,
                    base,
                    pit_floor,
                    ceiling,
                    "north_to_south",
                    materials,
                )
            else:
                sector = builder.add_sector(
                    floor=pit_floor,
                    ceiling=ceiling,
                    texturefloor=materials["floor"],
                    textureceiling=materials["ceiling"],
                )
                builder.add_inner_rect(feature_rect, sector, parent_sector, texture=materials["lower_wall"])
                count += 1
        elif feature_type in {"raised_central_structure", "depressed_floor", "central_platform", "central_pit"}:
            feature_floor = int(feature.get("floor_height", base + (32 if "raised" in str(feature_type) else -24)))
            feature_rect = _feature_rect(rect, "central", index)
            sector = builder.add_sector(
                floor=feature_floor,
                ceiling=ceiling,
                texturefloor=materials["floor"],
                textureceiling=materials["ceiling"],
            )
            builder.add_inner_rect(feature_rect, sector, parent_sector, texture=materials["lower_wall"])
            count += 1
    return count


def _add_stair_feature(
    builder: UdmfBuilder,
    feature_rect: Rect,
    parent_sector: int,
    base_floor: int,
    target_floor: int,
    ceiling: int,
    direction: Literal["south_to_north", "north_to_south"],
    materials: dict[str, str],
) -> int:
    x1, y1, x2, y2 = feature_rect
    stair_depth = min(96, max(64, (y2 - y1) // 2))
    step_floor = base_floor + ((target_floor - base_floor) // 2)
    if direction == "south_to_north":
        step_rect = (x1, y1 - stair_depth, x2, y1)
    else:
        step_rect = (x1, y2, x2, y2 + stair_depth)
    step_sector = builder.add_sector(
        floor=step_floor,
        ceiling=ceiling,
        texturefloor=materials["floor"],
        textureceiling=materials["ceiling"],
    )
    feature_sector = builder.add_sector(
        floor=target_floor,
        ceiling=ceiling,
        texturefloor=materials["floor"],
        textureceiling=materials["ceiling"],
    )
    builder.add_inner_rect_cluster(
        [(step_rect, step_sector), (feature_rect, feature_sector)],
        parent_sector,
        texture=materials["lower_wall"],
    )
    return 2


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
    width = min(192, max(128, (x2 - x1) // 4))
    height = min(160, max(128, (y2 - y1) // 4))
    if placement == "north":
        return (x2 - width - 128, y2 - height - 128 - index * 16, x2 - 128, y2 - 128 - index * 16)
    if placement == "central":
        cx, cy = _center(rect)
        return (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)
    return (x1 + 128, y1 + 128 + index * 16, x1 + 128 + width, y1 + 128 + height + index * 16)


def _add_progression_things(builder: UdmfBuilder, document: dict[str, Any], rects: dict[str, Rect], stats: GeometryStats) -> None:
    spaces = {space["id"]: space for space in document["spaces"]}
    start_room = str(document["progression"].get("start"))
    exit_room = str(document["progression"].get("exit"))
    if start_room in rects:
        x, y = _safe_room_point(rects[start_room], 0, _blocking_feature_rects(spaces[start_room], rects[start_room]))
        builder.add_thing(x, y, "player_start")
        _count_thing(stats, "player_start")
    for key in document["progression"].get("keys", []):
        if not isinstance(key, dict):
            continue
        location = str(key.get("location"))
        if location in rects:
            x, y = _safe_room_point(rects[location], 0, _blocking_feature_rects(spaces[location], rects[location]))
            builder.add_thing(x, y, "key", color=str(key.get("color", "blue")))
            _count_thing(stats, "key")
    if exit_room in rects:
        # The exit switch is a line special, but count the intent for reports.
        stats.things_by_kind["exit_switch"] = stats.things_by_kind.get("exit_switch", 0) + 1


def _add_space_things(builder: UdmfBuilder, document: dict[str, Any], rects: dict[str, Rect], stats: GeometryStats) -> None:
    for space in document["spaces"]:
        rect = rects[space["id"]]
        blocked_rects = _blocking_feature_rects(space, rect)
        offset = 1
        for item in _extract_items(space):
            kind = _item_kind(item)
            if kind:
                if kind == "bfg":
                    center = _center(rect)
                    x, y = center if not _point_in_any_rect(center, blocked_rects) else _safe_room_point(rect, offset, blocked_rects)
                else:
                    x, y = _safe_room_point(rect, offset, blocked_rects)
                builder.add_thing(x, y, kind)
                _count_thing(stats, kind)
                offset += 1
        for _name, monster_count in _extract_monsters(space):
            for _ in range(min(monster_count, 8)):
                x, y = _safe_room_point(rect, offset, blocked_rects)
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
    if "bfg" in lowered:
        return "bfg"
    if "shotgun" in lowered or "chaingun" in lowered or "weapon" in lowered:
        return "weapon"
    if "shell" in lowered or "ammo" in lowered or "clip" in lowered:
        return "ammo"
    if "health" in lowered or "stim" in lowered or "med" in lowered:
        return "health"
    return None


def _count_thing(stats: GeometryStats, kind: str) -> None:
    stats.things_by_kind[kind] = stats.things_by_kind.get(kind, 0) + 1


def _safe_room_point(rect: Rect, offset: int, blocked_rects: list[Rect] | None = None) -> Point:
    candidates = _room_point_candidates(rect)
    if not candidates:
        return _center(rect)
    if not blocked_rects:
        return candidates[offset % len(candidates)]
    for shift in range(len(candidates)):
        point = candidates[(offset + shift) % len(candidates)]
        if not _point_in_any_rect(point, blocked_rects):
            return point
    return candidates[offset % len(candidates)]


def _room_point_candidates(rect: Rect) -> list[Point]:
    x1, y1, x2, y2 = rect
    cols = max(1, (x2 - x1 - 256) // 96)
    rows = max(1, (y2 - y1 - 256) // 96)
    return [
        (min(x2 - 128, x1 + 128 + col * 96), min(y2 - 128, y1 + 128 + row * 96))
        for row in range(rows)
        for col in range(cols)
    ]


def _blocking_feature_rects(space: dict[str, Any], rect: Rect) -> list[Rect]:
    features = space.get("height_topology", {}).get("features", [])
    if not isinstance(features, list):
        features = []
    features = list(features)
    shape = canonical_room_shape(space.get("shape", "rectangle"))
    if not features and shape in {"raised_center", "depressed_floor"}:
        features.append({"type": shape, "placement": "central", "blocks_movement": False})
    elif not features and shape == "narrow_hall_with_cover":
        features = [
            {"type": "cover_block", "placement": "west_cover", "blocks_movement": True},
            {"type": "cover_block", "placement": "east_cover", "blocks_movement": True},
        ]
    blocked: list[Rect] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        feature_type = feature.get("type")
        if feature_type in {"ceiling_pillar_cluster", "pillar", "pillar_cluster"}:
            floor = int(feature.get("floor_height", space.get("ceiling_height", 128)))
            ceiling = int(feature.get("ceiling_height", space.get("ceiling_height", 128)))
            if bool(feature.get("blocks_movement", floor >= ceiling)):
                blocked.extend(_pillar_rects(rect, int(feature.get("count", 1))))
        elif feature_type == "cover_block":
            cx, cy = _center(rect)
            cover_x = cx + (-192 if index % 2 == 0 else 192)
            blocked.append((cover_x - 48, cy - 48, cover_x + 48, cy + 48))
        elif bool(feature.get("blocks_movement", False)):
            if feature_type in {"raised_platform", "platform"}:
                blocked.append(_feature_rect(rect, "north", index))
            elif feature_type in {"shallow_pit", "pit"}:
                blocked.append(_feature_rect(rect, "south", index))
            elif feature_type in {"raised_central_structure", "depressed_floor", "central_platform", "central_pit"}:
                blocked.append(_feature_rect(rect, "central", index))
    return blocked


def _point_in_any_rect(point: Point, rects: list[Rect]) -> bool:
    x, y = point
    return any(x1 <= x <= x2 and y1 <= y <= y2 for x1, y1, x2, y2 in rects)


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
