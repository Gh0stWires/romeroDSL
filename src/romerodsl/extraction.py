"""Extract canonical romeroDSL training pairs from WAD graph-cache samples.

v0.6 does not try to recreate exact linedefs from source WADs. The graph cache is
used as weak semantic supervision: theme, scale, key/lock colors, route-gating,
height variation, and combat/resource density become a valid high-level DSL
contract that the compiler can then turn into clean Doom geometry.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from romerodsl.metrics import structural_metrics


class TopologyExtractionError(ValueError):
    """Raised when graph-cache evidence cannot support a valid extracted target."""


TOPOLOGY_EXTRACTION_MODE = "topology_preserving"
LEGACY_EXTRACTION_MODE = "legacy_canonical"

KEY_COLOR_ORDER = ("blue", "red", "yellow")
VALID_THEME_GUESSES = {"techbase", "hell"}


def dsl_from_graph_sample(
    sample: dict[str, Any], *, mode: str = LEGACY_EXTRACTION_MODE
) -> dict[str, Any]:
    """Convert a graph-cache sample using the selected extraction mode.

    ``legacy_canonical`` preserves the historical small synthetic scaffold.
    ``topology_preserving`` uses sector groups and traversable graph evidence.
    """

    if mode == LEGACY_EXTRACTION_MODE:
        return _legacy_dsl_from_graph_sample(sample)
    if mode == TOPOLOGY_EXTRACTION_MODE:
        return topology_preserving_dsl_from_graph_sample(sample)
    raise ValueError(f"unsupported extraction mode {mode!r}")


def _legacy_dsl_from_graph_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one WAD graph-cache sample into a valid canonical DSL document.

    Start, exit, and a route between them are the core contract. Keyed progression
    is only authored into the target when the source graph says a lock gates the
    exit; optional/ambiguous key-lock observations stay in labels instead of being
    faked into required progression.
    """

    theme = _theme_from_sample(sample)
    colors = _required_key_lock_colors(sample)
    height_levels = _height_levels(sample)
    counts = _mapping(sample.get("counts"))
    things = _mapping(sample.get("things_by_category"))
    family = str(sample.get("family") or "unknown")
    map_name = str(sample.get("map") or "MAP01")
    has_height_variation = len(height_levels) > 1
    has_locked_progression = bool(colors)

    spaces = _spaces_for_sample(
        theme=theme,
        colors=colors,
        has_height_variation=has_height_variation,
        monster_count=int(things.get("monsters", 0) or 0),
        weapon_count=int(things.get("weapons", 0) or 0),
    )
    connections = _connections_for_colors(colors)
    critical_path = _critical_path_for_colors(colors)
    prompt = prompt_from_graph_sample(sample, colors)

    return {
        "dsl_version": "0.1",
        "game": "doom2",
        "format_target": "udmf",
        "title": _title_for_sample(theme, colors, int(counts.get("sectors", 0) or 0)),
        "scale": _scale_from_sector_count(int(counts.get("sectors", 0) or 0)),
        "difficulty": _difficulty_from_monsters(int(things.get("monsters", 0) or 0)),
        "intended_playtime": "short" if int(counts.get("sectors", 0) or 0) < 64 else "medium",
        "metadata": {
            "dataset_class": "synthetic",
            "extraction_mode": LEGACY_EXTRACTION_MODE,
            "source": "graph_cache_v1",
            "source_wad": sample.get("source_wad"),
            "source_family": family,
            "source_map": map_name,
            "source_counts": counts,
            "source_height_levels": height_levels,
        },
        "design_intent": {
            "summary": prompt,
            "topology": "keyed_progression_route" if has_locked_progression else "linear_route",
            "progression_style": "required_key_lock" if has_locked_progression else "reachable_exit",
            "combat_style": _combat_style(int(things.get("monsters", 0) or 0)),
            "extraction_note": "Canonicalized from graph-cache semantics; not exact source linedefs.",
        },
        "theme": _theme_block(theme),
        "progression": {
            "start": "start_room",
            "exit": "exit_room",
            "critical_path": critical_path,
            "keys": [
                {
                    "color": color,
                    "location": f"{color}_key_room",
                    "reachable_before": [f"{color}_locked_door"],
                }
                for color in colors
            ],
            "gates": [
                {
                    "id": f"{color}_locked_door",
                    "type": "locked_door",
                    "key": color,
                    "blocks": _gate_blocks_phrase(colors, index),
                    "required_for_exit": True,
                }
                for index, color in enumerate(colors)
            ],
        },
        "spaces": spaces,
        "connections": connections,
        "validation": {
            "required": [
                "exactly_one_player_start",
                "exactly_one_exit",
                "start_can_reach_exit",
                "keys_reachable_before_matching_locks",
                "locked_doors_are_partition_edges",
                "no_required_keys_in_secrets",
            ]
        },
    }


def topology_preserving_dsl_from_graph_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Extract a DSL document from source sector/edge evidence.

    The graph cache contains enough information for approximate room connectivity,
    room-local entities, height bands, and coarse shape/scale. It does not contain
    the original linedef vertex loops, so this mode deliberately records geometry
    as approximate rather than pretending to reproduce exact source polygons.
    """

    topology = _build_topology_summary(sample)
    if not topology["supported"]:
        raise TopologyExtractionError("; ".join(topology["unsupported_reasons"]))

    spaces = topology["spaces"]
    connections = topology["connections"]
    start_id = topology["start_room"]
    exit_id = topology["exit_room"]
    colors = topology["required_colors"]
    critical_path = topology["critical_path"]
    progression = {
        "start": start_id,
        "exit": exit_id,
        "critical_path": critical_path,
        "keys": [
            {
                "color": color,
                "location": topology["key_rooms"][color],
                "reachable_before": [topology["gate_ids"][color]],
            }
            for color in colors
        ],
        "gates": [
            {
                "id": topology["gate_ids"][color],
                "type": "locked_door",
                "key": color,
                "blocks": topology["gate_blocks"][color],
                "required_for_exit": True,
            }
            for color in colors
        ],
    }
    source_counts = _mapping(sample.get("counts"))
    theme = _theme_from_sample(sample)
    return {
        "dsl_version": "0.1",
        "game": "doom2",
        "format_target": "udmf",
        "title": "Extracted Doom topology",
        "scale": _scale_from_sector_count(int(source_counts.get("sectors", 0) or 0)),
        "difficulty": _difficulty_from_monsters(int(_mapping(sample.get("things_by_category")).get("monsters", 0) or 0)),
        "intended_playtime": "short" if len(spaces) < 6 else "medium",
        "metadata": {
            "dataset_class": "extracted",
            "extraction_mode": TOPOLOGY_EXTRACTION_MODE,
            "source": "graph_cache_v1",
            "source_wad": sample.get("source_wad"),
            "source_family": sample.get("family"),
            "source_map": sample.get("map"),
            "source_counts": source_counts,
            "source_topology": topology["source_metrics"],
            "information_preserved": topology["information_preserved"],
            "information_approximated": topology["information_approximated"],
            "information_unavailable": topology["information_unavailable"],
        },
        "design_intent": {
            "summary": _topology_prompt(sample, topology),
            "topology": topology["topology_label"],
            "progression_style": "required_key_lock" if colors else "reachable_exit",
            "combat_style": _combat_style(int(_mapping(sample.get("things_by_category")).get("monsters", 0) or 0)),
            "extraction_note": "Room groups and connections are derived from source sector/edge evidence; room geometry is approximate.",
        },
        "theme": _theme_block(theme),
        "progression": progression,
        "spaces": spaces,
        "connections": connections,
        "validation": {
            "required": [
                "exactly_one_player_start",
                "exactly_one_exit",
                "start_can_reach_exit",
                "keys_reachable_before_matching_locks",
                "locked_doors_are_partition_edges",
                "no_required_keys_in_secrets",
            ]
        },
    }


def _build_topology_summary(sample: dict[str, Any]) -> dict[str, Any]:
    sectors = {
        int(record["id"]): record
        for record in sample.get("sectors", [])
        if isinstance(record, dict) and isinstance(record.get("id"), int | float)
    }
    edges = [edge for edge in sample.get("edges", []) if isinstance(edge, dict)]
    if not sectors:
        return _unsupported_topology("graph cache has no sector records")

    adjacency = _source_adjacency(sectors, edges)
    start_sectors = {int(value) for value in _mapping(sample.get("progression")).get("start_sectors", []) if str(value).lstrip("-").isdigit()}
    exit_sectors = {int(value) for value in _mapping(sample.get("progression")).get("exit_sectors", []) if str(value).lstrip("-").isdigit()}
    start_sectors &= set(sectors)
    exit_sectors &= set(sectors)
    if not start_sectors or not exit_sectors:
        return _unsupported_topology("graph cache does not identify both start and exit sectors")

    components = _group_sectors(sectors, edges)
    sector_to_component = {sector_id: index for index, group in enumerate(components) for sector_id in group}
    start_components = {sector_to_component[sector] for sector in start_sectors}
    exit_components = {sector_to_component[sector] for sector in exit_sectors}
    if not start_components & exit_components and not _component_reachable(adjacency, start_sectors, exit_sectors):
        return _unsupported_topology("source sector graph has no traversable start-to-exit route")

    component_edges = _component_edges(sectors, edges, sector_to_component)
    source_metrics = _source_topology_metrics(components, component_edges, start_components, exit_components)
    component_ids, role_data = _allocate_component_ids(
        components,
        sector_to_component,
        start_components,
        exit_components,
        sample,
    )
    spaces = _spaces_from_components(components, component_ids, role_data, sectors, sample)
    room_edges = _room_edges(component_edges, component_ids)
    required_edges = _required_lock_edges(
        sample,
        room_edges,
        start_components,
        exit_components,
        sector_to_component,
    )
    required_colors = []
    key_rooms: dict[str, str] = {}
    for color, key_components in _key_components_by_color(sample, sector_to_component).items():
        required_for_color = [edge for edge in required_edges if edge.get("lock") == color]
        if required_for_color and key_components:
            required_colors.append(color)
            key_rooms[color] = component_ids[min(key_components)]
    required_colors.sort(key=KEY_COLOR_ORDER.index)
    connections, gate_ids, gate_blocks = _connections_from_room_edges(
        room_edges,
        required_edges,
        required_colors,
        component_ids,
    )
    start_id = component_ids[min(start_components)]
    exit_id = component_ids[min(exit_components)]
    critical_path = _critical_room_path(room_edges, start_components, exit_components, component_ids)
    topology_label = _topology_label(source_metrics)
    if not critical_path:
        return _unsupported_topology("room-level extraction could not derive a start-to-exit path")
    return {
        "supported": True,
        "unsupported_reasons": [],
        "spaces": spaces,
        "connections": connections,
        "start_room": start_id,
        "exit_room": exit_id,
        "required_colors": required_colors,
        "key_rooms": key_rooms,
        "gate_ids": gate_ids,
        "gate_blocks": gate_blocks,
        "critical_path": critical_path,
        "topology_label": topology_label,
        "source_metrics": source_metrics,
        "information_preserved": [
            "sector-group connectivity",
            "branches and cycles in the grouped traversable graph",
            "approximate room scale and shape",
            "observed floor/ceiling height bands",
            "source-associated starts, exits, keys, locks, combat, and resources",
        ],
        "information_approximated": [
            "room boundaries and shape from sector centers, counts, and vertex counts",
            "doorway placement within a room boundary",
            "room role when source evidence contains multiple competing roles",
        ],
        "information_unavailable": [
            "exact linedef vertex loops and original wall coordinates",
            "full thing positions and exact encounter placement",
            "unambiguous optional-lock semantics when the cache does not prove route necessity",
        ],
    }


def _unsupported_topology(reason: str) -> dict[str, Any]:
    return {"supported": False, "unsupported_reasons": [reason]}


def _source_adjacency(sectors: dict[int, dict[str, Any]], edges: list[dict[str, Any]]) -> dict[int, set[int]]:
    adjacency = {sector_id: set() for sector_id in sectors}
    for edge in edges:
        source = _int_or_none(edge.get("from_sector"))
        target = _int_or_none(edge.get("to_sector"))
        if source not in sectors or target not in sectors or source == target:
            continue
        if edge.get("passable", edge.get("kind") in {"connection", "door", "exit"}) is False:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _group_sectors(sectors: dict[int, dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[int]]:
    parent = {sector_id: sector_id for sector_id in sectors}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        source = _int_or_none(edge.get("from_sector"))
        target = _int_or_none(edge.get("to_sector"))
        if source not in sectors or target not in sectors or source == target:
            continue
        if edge.get("kind") != "connection" or edge.get("passable", True) is False:
            continue
        left, right = sectors[source], sectors[target]
        if (
            int(left.get("floor_height", 0)) == int(right.get("floor_height", 0))
            and int(left.get("ceiling_height", 0)) == int(right.get("ceiling_height", 0))
        ):
            union(source, target)
    groups: dict[int, list[int]] = defaultdict(list)
    for sector_id in sorted(sectors):
        groups[find(sector_id)].append(sector_id)
    return sorted(groups.values(), key=lambda group: group[0])


def _component_edges(
    sectors: dict[int, dict[str, Any]],
    edges: list[dict[str, Any]],
    sector_to_component: dict[int, int],
) -> list[dict[str, Any]]:
    result = []
    for edge in edges:
        source = _int_or_none(edge.get("from_sector"))
        target = _int_or_none(edge.get("to_sector"))
        if source not in sector_to_component or target not in sector_to_component or source == target:
            continue
        if edge.get("passable", edge.get("kind") in {"connection", "door", "exit"}) is False:
            continue
        left = sector_to_component[source]
        right = sector_to_component[target]
        if left == right:
            continue
        result.append(
            {
                "source_component": min(left, right),
                "target_component": max(left, right),
                "kind": str(edge.get("kind") or "connection"),
                "lock": edge.get("lock"),
                "linedef_id": edge.get("linedef_id"),
            }
        )
    return result


def _room_edges(component_edges: list[dict[str, Any]], component_ids: dict[int, str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for edge in component_edges:
        grouped[(edge["source_component"], edge["target_component"])].append(edge)
    result = []
    for (source, target), candidates in sorted(grouped.items()):
        selected = sorted(
            candidates,
            key=lambda edge: (
                edge.get("kind") != "door",
                edge.get("lock") is None,
                edge.get("linedef_id") if isinstance(edge.get("linedef_id"), int) else 10**9,
            ),
        )[0]
        result.append(
            {
                "from": component_ids[source],
                "to": component_ids[target],
                "source_component": source,
                "target_component": target,
                "kind": selected["kind"],
                "lock": selected.get("lock"),
                "edge_count": len(candidates),
                "source_edges": [candidate.get("linedef_id") for candidate in candidates],
            }
        )
    return result


def _connections_from_room_edges(room_edges: list[dict[str, Any]], required_edges: list[dict[str, Any]], required_colors: list[str], component_ids: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, str]]:
    required_pairs = {(edge["source_component"], edge["target_component"]): edge for edge in required_edges}
    connections = []
    gate_ids: dict[str, str] = {}
    gate_blocks: dict[str, str] = {}
    for edge in room_edges:
        pair = (edge["source_component"], edge["target_component"])
        required = required_pairs.get(pair)
        color = required.get("lock") if required else None
        connection_id = f"{color}_locked_door" if color in required_colors else f"room_edge_{len(connections):04d}"
        connection_type = "locked_door" if color in required_colors else ("door" if edge["kind"] == "door" else "hallway")
        record = {
            "id": connection_id,
            "from": edge["from"],
            "to": edge["to"],
            "type": connection_type,
            "required_for_exit": bool(color in required_colors),
            "source_evidence": {
                "sector_edge_count": edge["edge_count"],
                "linedef_ids": edge["source_edges"],
                "source_kind": edge["kind"],
                "source_lock_color": edge.get("lock"),
            },
        }
        if color in required_colors:
            record["key"] = color
            gate_ids[color] = connection_id
            gate_blocks[color] = f"{edge['from']} -> {edge['to']}"
        elif edge.get("lock"):
            record["source_lock_color"] = edge["lock"]
        connections.append(record)
    return connections, gate_ids, gate_blocks


def _required_lock_edges(
    sample: dict[str, Any],
    room_edges: list[dict[str, Any]],
    start_components: set[int],
    exit_components: set[int],
    sector_to_component: dict[int, int],
) -> list[dict[str, Any]]:
    progression = _mapping(sample.get("progression"))
    if not progression.get("exit_requires_any_lock"):
        return []
    graph: dict[int, set[int]] = defaultdict(set)
    for room_edge in room_edges:
        graph[room_edge["source_component"]].add(room_edge["target_component"])
        graph[room_edge["target_component"]].add(room_edge["source_component"])
    key_components = _key_components_by_color(sample, sector_to_component)
    required = []
    for edge in room_edges:
        color = edge.get("lock")
        if color not in key_components or not key_components[color]:
            continue
        blocked_pair = (edge["source_component"], edge["target_component"])
        reachable = _room_reachable(graph, start_components, blocked_pair=blocked_pair)
        if not (reachable & exit_components):
            required.append(edge)
    return required


def _key_components_by_color(sample: dict[str, Any], sector_to_component: dict[int, int]) -> dict[str, set[int]]:
    result: dict[str, set[int]] = defaultdict(set)
    for key in sample.get("keys", []):
        if not isinstance(key, dict):
            continue
        sector = _int_or_none(key.get("sector"))
        color = key.get("color")
        if sector in sector_to_component and color in KEY_COLOR_ORDER:
            result[str(color)].add(sector_to_component[sector])
    return result


def _room_graph(room_edges: list[dict[str, Any]]) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for edge in room_edges:
        graph[edge["from"]].add(edge["to"])
        graph[edge["to"]].add(edge["from"])
    return graph


def _room_reachable(graph: dict[Any, set[Any]], starts: set[Any], blocked_pair: tuple[Any, Any] | None = None) -> set[Any]:
    seen = set(starts)
    queue = deque(starts)
    blocked = {blocked_pair, (blocked_pair[1], blocked_pair[0])} if blocked_pair else set()
    while queue:
        current = queue.popleft()
        for target in graph.get(current, set()):
            if (current, target) in blocked or target in seen:
                continue
            seen.add(target)
            queue.append(target)
    return seen


def _critical_room_path(room_edges: list[dict[str, Any]], starts: set[int], exits: set[int], component_ids: dict[int, str]) -> list[str]:
    graph: dict[int, set[int]] = defaultdict(set)
    for edge in room_edges:
        graph[edge["source_component"]].add(edge["target_component"])
        graph[edge["target_component"]].add(edge["source_component"])
    queue = deque(starts)
    parent: dict[int, int | None] = {start: None for start in starts}
    found = None
    while queue:
        current = queue.popleft()
        if current in exits:
            found = current
            break
        for target in sorted(graph.get(current, set())):
            if target not in parent:
                parent[target] = current
                queue.append(target)
    if found is None:
        return []
    path = [found]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return [component_ids[component] for component in reversed(path)]


def _allocate_component_ids(components: list[list[int]], sector_to_component: dict[int, int], starts: set[int], exits: set[int], sample: dict[str, Any]) -> tuple[dict[int, str], dict[int, list[str]]]:
    roles: dict[int, list[str]] = defaultdict(list)
    for sector_text, sector_roles in _mapping(sample.get("sector_roles")).items():
        sector_id = _int_or_none(sector_text)
        if sector_id not in sector_to_component or not isinstance(sector_roles, list):
            continue
        roles[sector_to_component[sector_id]].extend(str(role) for role in sector_roles)
    for component in starts:
        roles[component].append("start")
    for component in exits:
        roles[component].append("exit")
    key_components = _key_components_by_color(sample, sector_to_component)
    for color, component_ids in key_components.items():
        for component in component_ids:
            roles[component].append(f"{color}_key")
    names: dict[int, str] = {}
    used: set[str] = set()
    for component in sorted(range(len(components))):
        role_values = roles.get(component, [])
        if component in starts:
            base = "start_room"
        elif component in exits:
            base = "exit_room"
        else:
            key_roles = [role for role in role_values if role.endswith("_key") and role.split("_")[0] in KEY_COLOR_ORDER]
            base = f"{key_roles[0].split('_')[0]}_key_room" if key_roles else f"source_room_{component:03d}"
        candidate = base
        suffix = 2
        while candidate in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate)
        names[component] = candidate
    return names, roles


def _spaces_from_components(components: list[list[int]], component_ids: dict[int, str], roles: dict[int, list[str]], sectors: dict[int, dict[str, Any]], sample: dict[str, Any]) -> list[dict[str, Any]]:
    spaces = []
    for component_index, sector_ids in enumerate(components):
        records = [sectors[sector_id] for sector_id in sector_ids]
        floors = [int(record.get("floor_height", 0)) for record in records]
        ceilings = [int(record.get("ceiling_height", 128)) for record in records]
        centers = [record.get("center") for record in records if isinstance(record.get("center"), list) and len(record["center"]) == 2]
        xs = [float(center[0]) for center in centers]
        ys = [float(center[1]) for center in centers]
        role_values = roles.get(component_index, [])
        role = _space_role(role_values)
        floor_height = _mode_int(floors)
        ceiling_height = max(max(ceilings) if ceilings else 128, floor_height + 64)
        room = {
            "id": component_ids[component_index],
            "role": role,
            "size": _component_size(len(sector_ids), xs, ys),
            "shape": _component_shape(records, xs, ys),
            "floor_height": floor_height,
            "ceiling_height": ceiling_height,
            "source_evidence": {
                "sector_ids": sector_ids,
                "sector_count": len(sector_ids),
                "role_signals": sorted(set(role_values)),
                "center_bounds": [min(xs), min(ys), max(xs), max(ys)] if xs and ys else None,
            },
        }
        if "start" in role_values:
            room["things"] = {"player_start": True}
        if "exit" in role_values:
            room.setdefault("contains", {})["exit_switch"] = True
        key_roles = [value for value in role_values if value.endswith("_key")]
        if key_roles:
            room.setdefault("contains", {})["key"] = key_roles[0].split("_", 1)[0]
        monster_count = role_values.count("monsters")
        if monster_count:
            room["encounter"] = {"intensity": _density_bucket(monster_count, 8, 40), "monsters": [{"imp": monster_count}]}
        resources = []
        for role_name, item_name in (("weapons", "weapon"), ("ammunitions", "ammo"), ("powerups", "health")):
            if role_name in role_values:
                resources.append(item_name)
        if resources:
            room["contains"] = dict(room.get("contains", {}), items=resources)
        unique_floors = sorted(set(floors))
        if len(unique_floors) > 1:
            room["height_topology"] = {
                "base_floor": min(unique_floors),
                "ceiling": max(ceilings) if ceilings else 128,
                "features": [{"type": "observed_height_regions", "floor_levels": unique_floors, "access": "source_evidence"}],
            }
        spaces.append(room)
    return spaces


def _space_role(role_values: list[str]) -> str:
    if "start" in role_values:
        return "player_start"
    if "exit" in role_values:
        return "exit"
    if any(value.endswith("_key") for value in role_values):
        return "key_room"
    if "monsters" in role_values:
        return "combat_room"
    if any(value in role_values for value in ("weapons", "ammunitions", "powerups")):
        return "reward_room"
    return "connector"


def _component_shape(records: list[dict[str, Any]], xs: list[float], ys: list[float]) -> str:
    if len(records) <= 2:
        return "closet"
    if xs and ys:
        width, height = max(xs) - min(xs), max(ys) - min(ys)
        if max(width, height) > max(1.0, min(width, height)) * 2.5:
            return "bent_corridor"
    average_vertices = sum(int(record.get("vertex_count", 0) or 0) for record in records) / len(records)
    if average_vertices >= 8:
        return "irregular_octagonal"
    if len({int(record.get("floor_height", 0)) for record in records}) > 1:
        return "irregular_orthogonal"
    return "rectangle"


def _component_size(count: int, xs: list[float], ys: list[float]) -> str:
    spread = max((max(xs) - min(xs)) * (max(ys) - min(ys)), 0.0) if xs and ys else 0.0
    if count <= 2 or spread < 4096:
        return "small"
    if count >= 32 or spread >= 250000:
        return "large"
    return "medium"


def _source_topology_metrics(components: list[list[int]], edges: list[dict[str, Any]], starts: set[int], exits: set[int]) -> dict[str, Any]:
    nodes = set(range(len(components)))
    pairs = {(edge["source_component"], edge["target_component"]) for edge in edges}
    adjacency = {node: set() for node in nodes}
    for source, target in pairs:
        adjacency[source].add(target)
        adjacency[target].add(source)
    connected_components = _count_components(adjacency)
    branch_nodes = sum(len(neighbors) > 2 for neighbors in adjacency.values())
    cycle_count = max(0, len(pairs) - len(nodes) + connected_components)
    return {
        "sector_group_count": len(components),
        "sector_count": sum(len(component) for component in components),
        "room_connection_count": len(pairs),
        "connected_components": connected_components,
        "branch_nodes": branch_nodes,
        "cycle_count": cycle_count,
        "largest_room_sector_count": max((len(component) for component in components), default=0),
        "start_room_count": len(starts),
        "exit_room_count": len(exits),
    }


def _topology_label(metrics: dict[str, Any]) -> str:
    if metrics["cycle_count"] and metrics["branch_nodes"]:
        return "branched_looped_source_topology"
    if metrics["branch_nodes"]:
        return "branched_source_topology"
    if metrics["cycle_count"]:
        return "looped_source_topology"
    return "connected_source_route"


def _topology_prompt(sample: dict[str, Any], topology: dict[str, Any]) -> str:
    metrics = topology["source_metrics"]
    parts = [
        "preserve the extracted Doom map organization",
        f"with {metrics['sector_group_count']} grouped spaces",
    ]
    if metrics["branch_nodes"]:
        parts.append("with branching routes")
    if metrics["cycle_count"]:
        parts.append("with looped connectivity")
    if topology["required_colors"]:
        parts.append("with source-supported required key progression")
    if len(_height_levels(sample)) > 1:
        parts.append("with observed height changes")
    return ", ".join(parts)


def _count_components(adjacency: dict[Any, set[Any]]) -> int:
    remaining = set(adjacency)
    count = 0
    while remaining:
        count += 1
        queue = [remaining.pop()]
        while queue:
            current = queue.pop()
            for neighbor in adjacency.get(current, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return count


def _component_reachable(adjacency: dict[int, set[int]], starts: set[int], exits: set[int]) -> bool:
    seen = set(starts)
    queue = deque(starts)
    while queue:
        current = queue.popleft()
        if current in exits:
            return True
        for neighbor in adjacency.get(current, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return False


def _mode_int(values: list[int]) -> int:
    return Counter(values).most_common(1)[0][0] if values else 0


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _legacy_dataset_class(document: dict[str, Any]) -> str:
    return str(_mapping(document.get("metadata")).get("dataset_class") or "synthetic")


def training_record_from_graph_sample(
    sample: dict[str, Any], *, sample_path: str, prompt_variant_index: int = 0, mode: str = LEGACY_EXTRACTION_MODE
) -> dict[str, Any]:
    """Build one prompt-to-DSL training record from a graph-cache sample."""

    required_colors = _required_key_lock_colors(sample)
    observed_colors = _paired_key_lock_colors(sample)
    document = dsl_from_graph_sample(sample, mode=mode)
    counts = _mapping(sample.get("counts"))
    progression = _mapping(sample.get("progression"))
    prompt_variants = (
        _topology_prompt_variants(sample, document)
        if mode == TOPOLOGY_EXTRACTION_MODE
        else prompt_variants_from_graph_sample(sample)
    )
    prompt = prompt_variants[prompt_variant_index % len(prompt_variants)]
    strict = bool(
        progression.get("start_reaches_exit")
        and progression.get("exit_requires_any_lock")
        and counts.get("keys", 0)
        and counts.get("locked_doors", 0)
        and required_colors
    )
    bucket = _bucket_for_sample(sample)
    source_topology = _mapping(_mapping(document.get("metadata")).get("source_topology"))
    target_graph = _mapping(structural_metrics(document).get("graph"))
    return {
        "schema": "romerodsl_training_pair_v0.8",
        "source_sample": sample_path,
        "source_wad": sample.get("source_wad"),
        "family": sample.get("family"),
        "map": sample.get("map"),
        "prompt": prompt,
        "prompt_variant_index": prompt_variant_index,
        "dsl": document,
        "labels": {
            "dataset_class": _legacy_dataset_class(document),
            "extraction_mode": mode,
            "bucket": bucket,
            "strict_progression_candidate": strict,
            "colors": required_colors,
            "observed_paired_colors": observed_colors,
            "theme": document["theme"]["primary"],
            "start_reaches_exit": bool(progression.get("start_reaches_exit")),
            "exit_requires_any_lock": bool(progression.get("exit_requires_any_lock")),
            "height_variation": len(_height_levels(sample)) > 1,
            "monster_density": _density_bucket(int(_mapping(sample.get("things_by_category")).get("monsters", 0) or 0), 8, 40),
        },
        "features": {
            "sector_count": int(counts.get("sectors", 0) or 0),
            "connection_count": int(counts.get("sector_connections", 0) or 0),
            "door_count": int(counts.get("doors", 0) or 0),
            "locked_door_count": int(counts.get("locked_doors", 0) or 0),
            "key_count": int(counts.get("keys", 0) or 0),
            "height_level_count": len(_height_levels(sample)),
            "monster_count": int(_mapping(sample.get("things_by_category")).get("monsters", 0) or 0),
            "source_room_count": int(source_topology.get("sector_group_count", 0) or 0),
            "source_branch_nodes": int(source_topology.get("branch_nodes", 0) or 0),
            "source_cycle_count": int(source_topology.get("cycle_count", 0) or 0),
            "target_room_count": len(document.get("spaces", [])),
            "target_connection_count": len(document.get("connections", [])),
            "target_branch_nodes": int(target_graph.get("branch_nodes", 0) or 0),
            "target_cycle_count": int(target_graph.get("cycle_count", 0) or 0),
        },
    }


def build_dsl_training_cache(
    graph_cache: Path,
    output: Path,
    *,
    strict_only: bool = False,
    prompt_variants_per_sample: int = 1,
    mode: str = LEGACY_EXTRACTION_MODE,
) -> dict[str, Any]:
    """Build a manifest and sample files of prompt-to-romeroDSL records."""

    if prompt_variants_per_sample < 1:
        raise ValueError("prompt_variants_per_sample must be at least 1")
    graph_cache = Path(graph_cache)
    output = Path(output)
    samples_dir = output / "samples"
    output.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.json"):
        old.unlink()
    for old in samples_dir.glob("*.json"):
        old.unlink()

    manifest_path = graph_cache / "manifest.jsonl"
    records = []
    color_counter: Counter[str] = Counter()
    observed_color_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    strict_count = 0
    source_count = 0
    supported_count = 0
    rejection_counter: Counter[str] = Counter()

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_rel = str(row["sample"])
        source_path = graph_cache / source_rel
        sample = json.loads(source_path.read_text(encoding="utf-8"))
        source_count += 1
        try:
            base_record = training_record_from_graph_sample(
                sample,
                sample_path=source_rel.replace("\\", "/"),
                mode=mode,
            )
        except TopologyExtractionError as exc:
            rejection_counter.update([str(exc)])
            continue
        supported_count += 1
        if strict_only and not base_record["labels"]["strict_progression_candidate"]:
            continue

        for variant_index in range(prompt_variants_per_sample):
            record = training_record_from_graph_sample(
                sample,
                sample_path=source_rel.replace("\\", "/"),
                prompt_variant_index=variant_index,
                mode=mode,
            )
            strict_count += int(record["labels"]["strict_progression_candidate"])
            sample_out = samples_dir / f"dsl_{len(records):05d}.json"
            sample_out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            manifest_record = {
                "path": str(sample_out.relative_to(output)).replace("\\", "/"),
                "source_sample": record["source_sample"],
                "prompt": record["prompt"],
                "prompt_variant_index": record["prompt_variant_index"],
                "family": record["family"],
                "map": record["map"],
                "labels": record["labels"],
                "features": record["features"],
            }
            records.append(manifest_record)
            color_counter.update(record["labels"]["colors"])
            observed_color_counter.update(record["labels"].get("observed_paired_colors", []))
            theme_counter.update([record["labels"]["theme"]])
            bucket_counter.update([record["labels"]["bucket"]])

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema": "romerodsl_training_pair_v0.8",
        "source_graph_cache": str(graph_cache),
        "source_samples": source_count,
        "supported_samples": supported_count,
        "rejected_samples": source_count - supported_count,
        "rejection_reasons": dict(rejection_counter),
        "extraction_mode": mode,
        "samples": len(records),
        "prompt_variants_per_sample": prompt_variants_per_sample,
        "strict_progression_candidates": strict_count,
        "strict_only": strict_only,
        "bucket_counts": dict(bucket_counter),
        "color_counts": dict(color_counter),
        "observed_color_counts": dict(observed_color_counter),
        "theme_counts": dict(theme_counter),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def audit_graph_cache(graph_cache: Path) -> dict[str, Any]:
    """Summarize full graph-cache coverage before building SFT data."""

    graph_cache = Path(graph_cache)
    manifest_path = graph_cache / "manifest.jsonl"
    bucket_counter: Counter[str] = Counter()
    theme_counter: Counter[str] = Counter()
    observed_color_counter: Counter[str] = Counter()
    target_color_counter: Counter[str] = Counter()
    source_samples = 0
    core_candidates = 0
    start_count = 0
    exit_count = 0
    start_reaches_exit_count = 0
    height_variation_count = 0

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = json.loads((graph_cache / str(row["sample"])).read_text(encoding="utf-8"))
        counts = _mapping(sample.get("counts"))
        progression = _mapping(sample.get("progression"))
        bucket = _bucket_for_sample(sample)
        observed_colors = _paired_key_lock_colors(sample)
        target_colors = _required_key_lock_colors(sample)

        source_samples += 1
        start_count += int(int(counts.get("player_starts", 0) or 0) > 0 or bool(progression.get("start_sectors")))
        exit_count += int(int(counts.get("exit_sectors", 0) or 0) > 0 or bool(progression.get("exit_sectors")))
        start_reaches_exit = bool(progression.get("start_reaches_exit"))
        start_reaches_exit_count += int(start_reaches_exit)
        core_candidates += int(start_reaches_exit)
        height_variation_count += int(len(_height_levels(sample)) > 1)
        bucket_counter.update([bucket])
        theme_counter.update([_theme_from_sample(sample)])
        observed_color_counter.update(observed_colors)
        target_color_counter.update(target_colors)

    return {
        "schema": "romerodsl_graph_cache_audit_v0.8",
        "source_graph_cache": str(graph_cache),
        "source_samples": source_samples,
        "samples_with_start": start_count,
        "samples_with_exit": exit_count,
        "start_reaches_exit": start_reaches_exit_count,
        "core_start_exit_route_candidates": core_candidates,
        "height_variation": height_variation_count,
        "bucket_counts": dict(bucket_counter),
        "theme_counts": dict(theme_counter),
        "observed_color_counts": dict(observed_color_counter),
        "target_required_color_counts": dict(target_color_counter),
    }


def audit_topology_extraction(graph_cache: Path, *, limit: int = 25) -> dict[str, Any]:
    """Compare grouped source topology with topology-preserving DSL targets."""

    graph_cache = Path(graph_cache)
    rows = []
    rejected: Counter[str] = Counter()
    manifest_rows = [
        json.loads(line)
        for line in (graph_cache / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        manifest_rows = manifest_rows[:limit]
    for row in manifest_rows:
        sample_path = graph_cache / str(row["sample"])
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        try:
            document = topology_preserving_dsl_from_graph_sample(sample)
        except TopologyExtractionError as exc:
            rejected.update([str(exc)])
            continue
        source = _mapping(_mapping(document.get("metadata")).get("source_topology"))
        target = _mapping(structural_metrics(document).get("graph"))
        rows.append(
            {
                "source_sample": str(row["sample"]).replace("\\", "/"),
                "source_wad": sample.get("source_wad"),
                "family": sample.get("family"),
                "map": sample.get("map"),
                "dataset_class": "extracted",
                "source": {
                    "room_count": source.get("sector_group_count", 0),
                    "connection_count": source.get("room_connection_count", 0),
                    "connected_components": source.get("connected_components", 0),
                    "branch_nodes": source.get("branch_nodes", 0),
                    "cycle_count": source.get("cycle_count", 0),
                    "largest_room_sector_count": source.get("largest_room_sector_count", 0),
                },
                "dsl": {
                    "room_count": target.get("nodes", 0),
                    "connection_count": target.get("edges", 0),
                    "connected_components": target.get("connected_components", 0),
                    "branch_nodes": target.get("branch_nodes", 0),
                    "cycle_count": target.get("cycle_count", 0),
                    "shape_count": len(structural_metrics(document).get("unique_room_shapes", [])),
                },
                "preserved": {
                    "branch_signal": bool(source.get("branch_nodes")) == bool(target.get("branch_nodes")),
                    "cycle_signal": bool(source.get("cycle_count")) == bool(target.get("cycle_count")),
                    "connected_signal": source.get("connected_components") == target.get("connected_components"),
                },
            }
        )
    return {
        "schema": "romerodsl_topology_extraction_audit_v1",
        "source_graph_cache": str(graph_cache),
        "requested_samples": limit,
        "audited_samples": len(manifest_rows),
        "supported_samples": len(rows),
        "rejected_samples": len(manifest_rows) - len(rows),
        "rejection_reasons": dict(rejected),
        "representative_comparisons": rows,
        "information_preserved": [
            "grouped sector connectivity",
            "room-level branches and cycles represented by DSL connections",
            "coarse room scale, aspect, shape, and height bands",
            "source-associated entities and route-supported key/lock dependencies",
        ],
        "information_approximated": [
            "room boundaries, exact portal positions, and encounter placement",
            "room roles when several source signals share a group",
        ],
        "information_unavailable_without_source_wad_reading": [
            "exact linedef vertex loops and wall coordinates",
            "full thing coordinates and precise spatial combat layout",
        ],
    }


def prompt_from_graph_sample(sample: dict[str, Any], colors: list[str] | None = None) -> str:
    """Create the default grounded natural-language prompt from observed facts."""

    if colors is None:
        return prompt_variants_from_graph_sample(sample, variant_count=1)[0]
    return _prompt_variant(sample, colors=colors, style=0)


def _topology_prompt_variants(sample: dict[str, Any], document: dict[str, Any]) -> list[str]:
    summary = str(document.get("design_intent", {}).get("summary", "preserve the extracted Doom map organization"))
    topology = str(document.get("design_intent", {}).get("topology", "connected_source_route"))
    theme = _theme_from_sample(sample)
    return [
        f"{summary} in a {theme} Doom map",
        f"rebuild the source-supported {topology} as a playable Doom level with the observed start and exit route",
        f"preserve the source map's grouped rooms, real branches or loops, height changes, and supported progression dependencies in {theme} style",
        f"generate a topology-preserving Doom layout from the source map evidence without inventing unsupported keys, locks, or geometry",
    ]


def prompt_variants_from_graph_sample(sample: dict[str, Any], *, variant_count: int = 4) -> list[str]:
    """Create deterministic, user-style prompts grounded in source-map facts.

    The variants make the core level requirement explicit: one player start, one
    exit, and a route between them. Keys/locks are requested only when the source
    provides required exit-gating evidence.
    """

    if variant_count < 1:
        raise ValueError("variant_count must be at least 1")
    colors = _required_key_lock_colors(sample)
    variants = [_prompt_variant(sample, colors=colors, style=style) for style in range(4)]
    return [variants[index % len(variants)] for index in range(variant_count)]


def _prompt_variant(sample: dict[str, Any], *, colors: list[str], style: int) -> str:
    counts = _mapping(sample.get("counts"))
    things = _mapping(sample.get("things_by_category"))
    theme = _theme_from_sample(sample)
    scale = _scale_from_sector_count(int(counts.get("sectors", 0) or 0))
    monster_bucket = _density_bucket(int(things.get("monsters", 0) or 0), 8, 40)
    has_supplies = bool(things.get("weapons", 0) or things.get("ammunitions", 0) or things.get("powerups", 0))
    has_height = len(_height_levels(sample)) > 1
    combat_phrase = f"{monster_bucket} monster pressure" if monster_bucket != "none" else "low combat"
    supply_phrase = "with weapons, ammo, and health" if has_supplies else "with sparse supplies"
    height_phrase = " and height changes" if has_height else ""

    if colors:
        keyed_phrase = " then ".join(f"{color} key" for color in colors)
        lock_phrase = " then ".join(f"{color} locked door" for color in colors)
        if style == 1:
            return (
                f"create a {scale} {theme} Doom II level with one player start and one exit, "
                f"where progression requires {keyed_phrase} before {lock_phrase}, {combat_phrase}{height_phrase}"
            )
        if style == 2:
            return (
                f"build a {theme} map with a playable route from start to exit, "
                f"gated by matching {'/'.join(colors)} key and locked-door stages, {supply_phrase}"
            )
        if style == 3:
            return (
                f"make a Doom map that has exactly one start and one exit, a {scale} connected layout, "
                f"and required {' then '.join(colors)} key progression on the main route"
            )
        return (
            f"make a Doom map with a {theme} theme, with a {scale} connected room layout, "
            f"with {', '.join(f'a {color} key before a {color} locked door' for color in colors)}, "
            f"with {combat_phrase}, {supply_phrase}, where progression gates the exit"
        )

    if style == 1:
        return (
            f"create a {scale} {theme} Doom II map with exactly one player start, one exit, "
            f"a clear route between them, and no locked doors"
        )
    if style == 2:
        return (
            f"build a short {theme} level where the player can reach the exit without collecting keys, "
            f"using {combat_phrase}{height_phrase}"
        )
    if style == 3:
        return (
            f"make a valid Doom level with a start room, an exit room, connected spaces, "
            f"no required key gates, and {supply_phrase}"
        )
    return (
        f"make a Doom map with a {theme} theme, with a {scale} connected room layout, "
        f"with one player start and one exit, with no locked doors, with {combat_phrase}, {supply_phrase}"
    )


def _spaces_for_sample(
    *,
    theme: str,
    colors: list[str],
    has_height_variation: bool,
    monster_count: int,
    weapon_count: int,
) -> list[dict[str, Any]]:
    spaces: list[dict[str, Any]] = [
        {
            "id": "start_room",
            "role": "player_start",
            "size": "small",
            "shape": "rectangle",
            "floor_height": 0,
            "ceiling_height": 128,
            "things": {"player_start": True, "items": ["shotgun"] if weapon_count else ["clip_small"]},
        },
        {
            "id": "combat_hub",
            "role": "combat_hub",
            "size": "large" if monster_count > 8 else "medium",
            "shape": "irregular_octagonal" if has_height_variation else "rectangle",
            "floor_height": 0,
            "ceiling_height": 192 if has_height_variation else 128,
            "materials": _accent_materials(theme),
            "encounter": {
                "intensity": _density_bucket(monster_count, 8, 40),
                "monsters": _monster_mix(monster_count),
            },
        },
    ]
    if has_height_variation:
        spaces[1]["height_topology"] = {
            "base_floor": 0,
            "ceiling": 192,
            "features": [
                {
                    "id": "extracted_height_platform",
                    "type": "raised_platform",
                    "placement": "north_wall",
                    "floor_height": 32,
                    "access": "stairs",
                    "purpose": "observed_source_height_variation",
                }
            ],
        }

    for index, color in enumerate(colors):
        spaces.append(
            {
                "id": f"{color}_key_room",
                "role": "key_room",
                "size": "medium",
                "shape": "rectangle_with_side_alcoves",
                "floor_height": -16 if index % 2 == 0 else 16,
                "ceiling_height": 144,
                "contains": {"key": color},
                "encounter": {
                    "intensity": "medium" if monster_count else "low",
                    "monsters": [{"imp": 3 + index}],
                },
            }
        )

    spaces.append(
        {
            "id": "exit_room",
            "role": "exit",
            "size": "small",
            "shape": "rectangle",
            "floor_height": 0,
            "ceiling_height": 128,
            "contains": {"exit_switch": True},
            "materials": {"exit_switch": "SW1EXIT"},
        }
    )
    return spaces


def _connections_for_colors(colors: list[str]) -> list[dict[str, Any]]:
    connections = [
        {"id": "start_to_hub", "from": "start_room", "to": "combat_hub", "type": "doorway"},
    ]
    previous = "combat_hub"
    if not colors:
        connections.append({"id": "hub_to_exit", "from": "combat_hub", "to": "exit_room", "type": "doorway"})
        return connections

    for index, color in enumerate(colors):
        key_room = f"{color}_key_room"
        if previous != key_room:
            connections.append(
                {"id": f"{color}_key_route", "from": previous, "to": key_room, "type": "doorway"}
            )
        target = f"{colors[index + 1]}_key_room" if index + 1 < len(colors) else "exit_room"
        connections.append(
            {
                "id": f"{color}_locked_door",
                "from": key_room,
                "to": target,
                "type": "locked_door",
                "key": color,
                "required_for_exit": True,
            }
        )
        previous = target
    return connections


def _critical_path_for_colors(colors: list[str]) -> list[str]:
    path = ["start_room", "start_to_hub", "combat_hub"]
    if not colors:
        return path + ["hub_to_exit", "exit_room"]
    previous = "combat_hub"
    for index, color in enumerate(colors):
        key_room = f"{color}_key_room"
        if previous != key_room:
            path.append(f"{color}_key_route")
        path.extend([key_room, f"{color}_locked_door"])
        previous = f"{colors[index + 1]}_key_room" if index + 1 < len(colors) else "exit_room"
    path.append("exit_room")
    return path


def _paired_key_lock_colors(sample: dict[str, Any]) -> list[str]:
    progression = _mapping(sample.get("progression"))
    colors = _mapping(progression.get("colors"))
    paired = [
        color
        for color in KEY_COLOR_ORDER
        if _mapping(colors.get(color)).get("has_key_and_lock")
    ]
    return sorted(
        paired,
        key=lambda color: (
            not bool(_mapping(colors.get(color)).get("key_reachable_without_locks")),
            KEY_COLOR_ORDER.index(color),
        ),
    )


def _required_key_lock_colors(sample: dict[str, Any]) -> list[str]:
    progression = _mapping(sample.get("progression"))
    if not progression.get("exit_requires_any_lock"):
        return []
    return _paired_key_lock_colors(sample)


def _bucket_for_sample(sample: dict[str, Any]) -> str:
    counts = _mapping(sample.get("counts"))
    progression = _mapping(sample.get("progression"))
    required_colors = _required_key_lock_colors(sample)
    observed_colors = _paired_key_lock_colors(sample)
    if len(required_colors) >= 2:
        return "required_multi_lock"
    if len(required_colors) == 1:
        return "required_single_lock"
    if observed_colors or int(counts.get("locked_doors", 0) or 0) or int(counts.get("keys", 0) or 0):
        return "optional_key_lock"
    if progression.get("start_reaches_exit"):
        return "basic_route"
    return "layout_style_only"


def _title_for_sample(theme: str, colors: list[str], sector_count: int) -> str:
    scale = _scale_from_sector_count(sector_count)
    if colors:
        return f"{scale.title()} {theme.title()} Key Route"
    return f"{scale.title()} {theme.title()} Route"


def _theme_from_sample(sample: dict[str, Any]) -> str:
    theme = str(sample.get("theme_guess") or "unknown")
    return theme if theme in VALID_THEME_GUESSES else "techbase"


def _theme_block(theme: str) -> dict[str, Any]:
    if theme == "hell":
        return {
            "primary": "hell",
            "secondary": "stone",
            "mood": "hostile",
            "texture_palette": {
                "walls": ["STONE2", "MARBLE", "METAL"],
                "floors": ["FLAT5_4", "FLOOR0_5"],
                "ceilings": ["CEIL3_5"],
                "doors": ["BIGDOOR2"],
            },
            "materials": {
                "floor": "FLAT5_4",
                "ceiling": "CEIL3_5",
                "wall": "STONE2",
                "upper_wall": "STONE2",
                "lower_wall": "STONE2",
                "door": "BIGDOOR2",
                "door_track": "DOORTRAK",
                "exit_switch": "SW1EXIT",
            },
        }
    return {
        "primary": "techbase",
        "secondary": "industrial",
        "mood": "tense",
        "texture_palette": {
            "walls": ["STARTAN", "METAL", "SUPPORT"],
            "floors": ["FLOOR0_1", "FLAT20"],
            "ceilings": ["CEIL3_5"],
            "doors": ["BIGDOOR2"],
        },
        "materials": {
            "floor": "FLOOR0_1",
            "ceiling": "CEIL3_5",
            "wall": "STARTAN2",
            "upper_wall": "STARTAN2",
            "lower_wall": "STEP1",
            "door": "BIGDOOR2",
            "door_track": "DOORTRAK",
            "exit_switch": "SW1EXIT",
        },
    }


def _accent_materials(theme: str) -> dict[str, str]:
    if theme == "hell":
        return {"floor": "FLAT5_4", "ceiling": "CEIL3_5", "wall": "STONE2", "lower_wall": "STONE2"}
    return {"floor": "FLAT20", "ceiling": "CEIL3_5", "wall": "METAL2", "lower_wall": "STEP1"}


def _gate_blocks_phrase(colors: list[str], index: int) -> str:
    source = f"{colors[index]}_key_room"
    target = f"{colors[index + 1]}_key_room" if index + 1 < len(colors) else "exit_room"
    return f"{source} -> {target}"


def _height_levels(sample: dict[str, Any]) -> list[int]:
    levels = sample.get("height_levels", [])
    if not isinstance(levels, list):
        return []
    return [int(level) for level in levels if isinstance(level, int | float)]


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _scale_from_sector_count(sector_count: int) -> str:
    if sector_count < 8:
        return "tiny"
    if sector_count < 64:
        return "small"
    if sector_count < 256:
        return "medium"
    return "large"


def _difficulty_from_monsters(monster_count: int) -> str:
    if monster_count <= 8:
        return "easy"
    if monster_count <= 40:
        return "medium"
    return "hard"


def _combat_style(monster_count: int) -> str:
    if monster_count <= 0:
        return "exploration"
    if monster_count <= 8:
        return "light_pressure"
    if monster_count <= 40:
        return "readable_pressure"
    return "heavy_pressure"


def _density_bucket(count: int, low: int, high: int) -> str:
    if count <= 0:
        return "none"
    if count <= low:
        return "light"
    if count <= high:
        return "medium"
    return "heavy"


def _monster_mix(monster_count: int) -> list[dict[str, int]]:
    if monster_count <= 0:
        return []
    if monster_count <= 8:
        return [{"zombieman": max(1, monster_count)}]
    return [{"imp": min(monster_count, 8)}, {"shotgun_guy": max(1, monster_count // 4)}]
