"""Validation for romeroDSL v0.1 documents.

The schema is intentionally semantic rather than geometric. It checks that a
model-authored map contract is internally consistent before any compiler tries
to turn it into sectors, linedefs, and things.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from romerodsl.materials import validate_material_declaration

SUPPORTED_DSL_VERSION = "0.1"
KEY_COLORS = {"blue", "red", "yellow"}
REQUIRED_TOP_LEVEL_KEYS = {
    "dsl_version",
    "game",
    "format_target",
    "title",
    "design_intent",
    "theme",
    "progression",
    "spaces",
    "connections",
    "validation",
}


@dataclass(slots=True)
class ValidationReport:
    """Result of validating one romeroDSL document."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.valid:
            joined = "\n".join(f"- {error}" for error in self.errors)
            raise ValueError(f"Invalid romeroDSL document:\n{joined}")


def validate_document(document: dict[str, Any]) -> ValidationReport:
    """Validate a romeroDSL document.

    This is not a complete formal schema yet. It focuses on invariants that
    matter for Doom authorship: references, progression gates, key-before-lock
    logic, and the rule that required progression cannot be hidden in secrets.
    """

    errors: list[str] = []
    warnings: list[str] = []

    _validate_mapping(document, "document", errors)
    if errors:
        return ValidationReport(valid=False, errors=errors, warnings=warnings)

    missing = REQUIRED_TOP_LEVEL_KEYS - document.keys()
    for key in sorted(missing):
        errors.append(f"missing top-level key: {key}")

    version = str(document.get("dsl_version", ""))
    if version != SUPPORTED_DSL_VERSION:
        errors.append(f"unsupported dsl_version {version!r}; expected {SUPPORTED_DSL_VERSION!r}")

    progression = _mapping(document.get("progression"), "progression", errors)
    spaces = _sequence(document.get("spaces"), "spaces", errors)
    connections = _sequence(document.get("connections"), "connections", errors)

    space_ids = _collect_space_ids(spaces, errors)
    connection_ids = _collect_connection_ids(connections, warnings)
    secret_space_ids = {
        str(space.get("id"))
        for space in spaces
        if isinstance(space, dict) and bool(space.get("secret", False))
    }

    start_id = str(progression.get("start", ""))
    exit_id = str(progression.get("exit", ""))
    if start_id not in space_ids:
        errors.append(f"progression.start references unknown space {start_id!r}")
    if exit_id not in space_ids:
        errors.append(f"progression.exit references unknown space {exit_id!r}")

    _validate_spaces(spaces, errors, warnings)
    _validate_connections(connections, space_ids, errors)
    _validate_theme_materials(document.get("theme"), errors)
    _validate_critical_path(progression, space_ids, connection_ids, errors, warnings)
    _validate_keys_and_gates(progression, space_ids, connection_ids, secret_space_ids, errors, warnings)
    _validate_validation_contract(document.get("validation"), warnings)

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)


def _validate_mapping(value: Any, name: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")


def _mapping(value: Any, name: str, errors: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _sequence(value: Any, name: str, errors: list[str]) -> list[Any]:
    if not isinstance(value, list):
        errors.append(f"{name} must be a list")
        return []
    return value


def _collect_space_ids(spaces: list[Any], errors: list[str]) -> set[str]:
    ids: set[str] = set()
    for index, space in enumerate(spaces):
        if not isinstance(space, dict):
            errors.append(f"spaces[{index}] must be an object")
            continue
        space_id = space.get("id")
        if not isinstance(space_id, str) or not space_id:
            errors.append(f"spaces[{index}] needs a non-empty string id")
            continue
        if space_id in ids:
            errors.append(f"duplicate space id: {space_id}")
        ids.add(space_id)
    return ids


def _collect_connection_ids(connections: list[Any], warnings: list[str]) -> set[str]:
    ids: set[str] = set()
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            continue
        connection_id = connection.get("id")
        if isinstance(connection_id, str) and connection_id:
            if connection_id in ids:
                warnings.append(f"duplicate connection id {connection_id!r}; gate references may be ambiguous")
            ids.add(connection_id)
        else:
            from_id = connection.get("from")
            to_id = connection.get("to")
            if isinstance(from_id, str) and isinstance(to_id, str):
                ids.add(f"{from_id}->{to_id}")
    return ids


def _validate_spaces(spaces: list[Any], errors: list[str], warnings: list[str]) -> None:
    player_start_count = 0
    exit_count = 0

    for index, space in enumerate(spaces):
        if not isinstance(space, dict):
            continue
        space_id = str(space.get("id", f"spaces[{index}]"))

        if "role" not in space:
            warnings.append(f"space {space_id!r} has no role")

        floor = space.get("floor_height")
        ceiling = space.get("ceiling_height")
        if isinstance(floor, int) and isinstance(ceiling, int) and ceiling <= floor:
            errors.append(f"space {space_id!r} ceiling_height must be greater than floor_height")

        role = space.get("role")
        things = space.get("things", {})
        contains = space.get("contains", {})
        if role == "player_start" or (isinstance(things, dict) and things.get("player_start")):
            player_start_count += 1
        if role == "exit" or (isinstance(contains, dict) and contains.get("exit_switch")):
            exit_count += 1

        height_topology = space.get("height_topology")
        if isinstance(height_topology, dict):
            _validate_height_topology(space_id, height_topology, errors)

        validate_material_declaration(space.get("materials"), f"space {space_id!r}", errors)

    if player_start_count != 1:
        errors.append(f"expected exactly one player start space, found {player_start_count}")
    if exit_count != 1:
        errors.append(f"expected exactly one exit space, found {exit_count}")


def _validate_height_topology(space_id: str, height_topology: dict[str, Any], errors: list[str]) -> None:
    features = height_topology.get("features", [])
    if not isinstance(features, list):
        errors.append(f"space {space_id!r} height_topology.features must be a list")
        return
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            errors.append(f"space {space_id!r} height feature {index} must be an object")
            continue
        if "type" not in feature:
            errors.append(f"space {space_id!r} height feature {index} is missing type")
        floor = feature.get("floor_height")
        ceiling = feature.get("ceiling_height")
        if isinstance(floor, int) and isinstance(ceiling, int) and ceiling < floor:
            errors.append(
                f"space {space_id!r} height feature {index} has ceiling lower than floor"
            )


def _validate_theme_materials(value: Any, errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    validate_material_declaration(value.get("materials"), "theme", errors)


def _validate_connections(connections: list[Any], space_ids: set[str], errors: list[str]) -> None:
    for index, connection in enumerate(connections):
        if not isinstance(connection, dict):
            errors.append(f"connections[{index}] must be an object")
            continue
        from_id = connection.get("from")
        to_id = connection.get("to")
        if from_id not in space_ids:
            errors.append(f"connections[{index}].from references unknown space {from_id!r}")
        if to_id not in space_ids:
            errors.append(f"connections[{index}].to references unknown space {to_id!r}")

        connection_type = connection.get("type")
        if connection_type in {"locked_door", "keyed_door"}:
            key = connection.get("key")
            if key not in KEY_COLORS:
                errors.append(f"locked connection {connection.get('id', index)!r} has invalid key {key!r}")


def _validate_critical_path(
    progression: dict[str, Any],
    space_ids: set[str],
    connection_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    critical_path = progression.get("critical_path", [])
    if not isinstance(critical_path, list) or not critical_path:
        errors.append("progression.critical_path must be a non-empty list")
        return

    for node in critical_path:
        if node in space_ids or node in connection_ids:
            continue
        warnings.append(
            f"critical_path entry {node!r} is not a known space or connection id; "
            "this may be intentional for abstract beats, but compiler support may be limited"
        )


def _validate_keys_and_gates(
    progression: dict[str, Any],
    space_ids: set[str],
    connection_ids: set[str],
    secret_space_ids: set[str],
    errors: list[str],
    warnings: list[str],
) -> None:
    keys = progression.get("keys", [])
    gates = progression.get("gates", [])
    if not isinstance(keys, list):
        errors.append("progression.keys must be a list")
        keys = []
    if not isinstance(gates, list):
        errors.append("progression.gates must be a list")
        gates = []

    key_locations_by_color: dict[str, set[str]] = {}
    for index, key in enumerate(keys):
        if not isinstance(key, dict):
            errors.append(f"progression.keys[{index}] must be an object")
            continue
        color = key.get("color")
        location = key.get("location")
        if color not in KEY_COLORS:
            errors.append(f"progression.keys[{index}] has invalid color {color!r}")
        if location not in space_ids:
            errors.append(f"progression.keys[{index}] references unknown location {location!r}")
        elif location in secret_space_ids:
            errors.append(f"required progression key {color!r} is placed in secret space {location!r}")
        if isinstance(color, str) and isinstance(location, str):
            key_locations_by_color.setdefault(color, set()).add(location)

    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            errors.append(f"progression.gates[{index}] must be an object")
            continue
        gate_id = gate.get("id")
        gate_type = gate.get("type")
        color = gate.get("key")
        if gate_type in {"locked_door", "keyed_door"}:
            if color not in KEY_COLORS:
                errors.append(f"gate {gate_id!r} has invalid key {color!r}")
            elif color not in key_locations_by_color:
                errors.append(f"gate {gate_id!r} requires {color!r} key but no matching key exists")
        if isinstance(gate_id, str) and gate_id not in connection_ids:
            warnings.append(
                f"gate {gate_id!r} has no matching connection id; compiler may not know where it is"
            )


def _validate_validation_contract(value: Any, warnings: list[str]) -> None:
    if not isinstance(value, dict):
        return
    required = value.get("required", [])
    if isinstance(required, list):
        needed = {
            "exactly_one_player_start",
            "exactly_one_exit",
            "start_can_reach_exit",
            "locked_doors_are_partition_edges",
        }
        missing = needed - set(required)
        for invariant in sorted(missing):
            warnings.append(f"validation.required does not include {invariant!r}")
