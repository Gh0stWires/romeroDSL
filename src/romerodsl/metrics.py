"""Structural metrics for romeroDSL documents and compiled outputs.

These metrics intentionally describe authored structure rather than treating raw
sector count as a proxy for complexity.  They are dependency-free so they can
be used during dataset construction, compiler tests, and no-repair evaluation.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any

from romerodsl.evaluation import evaluate_generated_dsl_intent


def structural_metrics(
    document: dict[str, Any],
    *,
    prompt: str | None = None,
    plan: dict[str, Any] | None = None,
    wad_report: dict[str, Any] | None = None,
    dsl_valid: bool | None = None,
    raw_output: str | None = None,
    output_token_count: int | None = None,
    max_new_tokens: int | None = None,
    generation_truncated: bool | None = None,
) -> dict[str, Any]:
    """Return reproducible semantic, graph, geometry, and output metrics.

    ``document`` may be a partially parsed model object.  The function avoids
    raising on malformed optional fields so it can also describe failed model
    generations.  Validation itself remains the responsibility of
    :func:`romerodsl.schema.validate_document`.
    """

    spaces = [space for space in document.get("spaces", []) if isinstance(space, dict)]
    connections = [
        connection for connection in document.get("connections", []) if isinstance(connection, dict)
    ]
    space_ids = {str(space.get("id")) for space in spaces if isinstance(space.get("id"), str)}
    adjacency: dict[str, set[str]] = {space_id: set() for space_id in space_ids}
    directed_adjacency: dict[str, set[str]] = {space_id: set() for space_id in space_ids}
    edge_pairs: set[tuple[str, str]] = set()
    for connection in connections:
        source = connection.get("from")
        target = connection.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        directed_adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        edge_pairs.add((min(source, target), max(source, target)))

    components = _connected_components(adjacency)
    cycle_count = max(0, len(edge_pairs) - len(space_ids) + len(components))
    progression = document.get("progression")
    progression = progression if isinstance(progression, dict) else {}
    critical_path = progression.get("critical_path")
    critical_path = critical_path if isinstance(critical_path, list) else []

    shapes = Counter(
        str(space.get("shape"))
        for space in spaces
        if isinstance(space.get("shape"), str)
    )
    roles = Counter(
        str(space.get("role"))
        for space in spaces
        if isinstance(space.get("role"), str)
    )
    connection_types = Counter(
        str(connection.get("type"))
        for connection in connections
        if isinstance(connection.get("type"), str)
    )
    height_feature_types = Counter()
    height_levels: set[int] = set()
    for space in spaces:
        for field_name in ("floor_height", "ceiling_height"):
            value = space.get(field_name)
            if isinstance(value, (int, float)):
                height_levels.add(int(value))
        topology = space.get("height_topology")
        features = topology.get("features", []) if isinstance(topology, dict) else []
        if not isinstance(features, list):
            continue
        for feature in features:
            if not isinstance(feature, dict):
                continue
            feature_type = feature.get("type")
            if isinstance(feature_type, str):
                height_feature_types[feature_type] += 1
            for field_name in ("floor_height", "ceiling_height"):
                value = feature.get(field_name)
                if isinstance(value, (int, float)):
                    height_levels.add(int(value))

    materials = set()
    for space in spaces:
        declared = space.get("materials")
        if isinstance(declared, dict):
            materials.update(str(value) for value in declared.values() if isinstance(value, str))
    theme = document.get("theme")
    if isinstance(theme, dict) and isinstance(theme.get("materials"), dict):
        materials.update(str(value) for value in theme["materials"].values() if isinstance(value, str))

    keys = [key for key in progression.get("keys", []) if isinstance(key, dict)]
    gates = [gate for gate in progression.get("gates", []) if isinstance(gate, dict)]
    key_colors = sorted({str(key.get("color")) for key in keys if isinstance(key.get("color"), str)})
    gate_colors = sorted(
        {
            str(gate.get("key"))
            for gate in gates
            if gate.get("type") in {"locked_door", "keyed_door"}
            and isinstance(gate.get("key"), str)
        }
        | {
            str(connection.get("key"))
            for connection in connections
            if connection.get("type") in {"locked_door", "keyed_door"}
            and isinstance(connection.get("key"), str)
        }
    )
    required_gate_count = sum(
        gate.get("type") in {"locked_door", "keyed_door"} and bool(gate.get("required_for_exit"))
        for gate in gates
    )
    features = set(shapes) | set(roles) | set(connection_types) | set(height_feature_types) | materials

    result: dict[str, Any] = {
        "space_count": len(spaces),
        "connection_count": len(connections),
        "unique_room_shapes": sorted(shapes),
        "room_shape_counts": dict(sorted(shapes.items())),
        "unique_room_shape_count": len(shapes),
        "room_role_counts": dict(sorted(roles.items())),
        "connection_type_counts": dict(sorted(connection_types.items())),
        "graph": {
            "nodes": len(space_ids),
            "edges": len(edge_pairs),
            "connected_components": len(components),
            "connected": len(components) <= 1 and bool(space_ids),
            "branch_nodes": sum(len(targets) > 1 for targets in directed_adjacency.values()),
            "branch_edges": sum(max(0, len(targets) - 1) for targets in directed_adjacency.values()),
            "cycle_count": cycle_count,
            "degree_sequence": sorted((len(neighbors) for neighbors in adjacency.values()), reverse=True),
        },
        "critical_path": {
            "length": len(critical_path),
            "space_nodes": sum(node in space_ids for node in critical_path),
            "entries": critical_path,
        },
        "progression": {
            "key_colors": key_colors,
            "gate_colors": gate_colors,
            "required_gate_count": required_gate_count,
            "matching_key_colors": sorted(set(key_colors) & set(gate_colors)),
            "missing_key_for_gate_colors": sorted(set(gate_colors) - set(key_colors)),
            "keys": len(keys),
            "gates": len(gates),
        },
        "height": {
            "levels": sorted(height_levels),
            "level_count": len(height_levels),
            "varied": len(height_levels) > 1,
            "feature_types": dict(sorted(height_feature_types.items())),
            "feature_count": sum(height_feature_types.values()),
        },
        "architectural_feature_count": len(features),
        "architectural_features": sorted(features),
        "geometry": _geometry_metrics(plan, wad_report),
        "validation": {
            "dsl_valid": dsl_valid,
            "wad_valid": wad_report.get("valid") if isinstance(wad_report, dict) else None,
            "export_present": wad_report is not None,
        },
    }
    if prompt is not None:
        result["prompt"] = evaluate_generated_dsl_intent(prompt, document)
    if output_token_count is not None or max_new_tokens is not None or raw_output is not None:
        result["output"] = {
            "raw_character_count": len(raw_output) if raw_output is not None else None,
            "output_token_count": output_token_count,
            "max_new_tokens": max_new_tokens,
            "generation_truncated": (
                generation_truncated
                if generation_truncated is not None
                else bool(
                    output_token_count is not None
                    and max_new_tokens is not None
                    and output_token_count >= max_new_tokens
                )
            ),
        }
    return result


def _geometry_metrics(plan: dict[str, Any] | None, wad_report: dict[str, Any] | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(plan, dict):
        rooms = plan.get("rooms")
        edges = plan.get("edges")
        result.update(
            {
                "plan_room_count": len(rooms) if isinstance(rooms, list) else None,
                "plan_edge_count": len(edges) if isinstance(edges, list) else None,
            }
        )
    if isinstance(wad_report, dict):
        for key in (
            "sectors",
            "vertices",
            "linedefs",
            "sidedefs",
            "things",
            "room_sectors",
            "corridor_sectors",
            "door_sectors",
            "height_feature_sectors",
            "floor_height_levels",
            "missing_visible_textures",
            "invalid_side_references",
            "unreferenced_sidedefs",
            "engine_tested",
        ):
            if key in wad_report:
                result[key] = wad_report[key]
    return result


def _connected_components(adjacency: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(adjacency)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining)
        component: set[str] = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            queue.extend(sorted(adjacency.get(node, set()) - component))
        components.append(component)
    return components
