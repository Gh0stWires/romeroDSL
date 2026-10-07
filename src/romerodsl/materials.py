"""Material resolution for Doom II surface textures/flats.

The DSL can stay semantic while the compiler works with concrete Doom II
surface names. Materials are deliberately surface-oriented: Doom assigns flats
to sector floors/ceilings and textures to sidedef middle/upper/lower surfaces.
"""

from __future__ import annotations

from typing import Any

FLAT_ROLES = {"floor", "ceiling"}
TEXTURE_ROLES = {"wall", "upper_wall", "lower_wall", "door", "door_track", "exit_switch"}
SURFACE_ROLES = tuple(sorted(FLAT_ROLES | TEXTURE_ROLES))

# Narrow Doom II catalog for the textures/flats this compiler is allowed to emit
# today. Keep this conservative until we parse the selected IWAD dynamically.
VALID_DOOM2_FLATS = {
    "CEIL3_5",
    "FLAT20",
    "FLAT5_4",
    "FLOOR0_1",
    "FLOOR0_2",
    "FLOOR0_3",
    "FLOOR0_5",
    "FLOOR0_6",
    "FLOOR0_7",
    "STEP1",
    "STEP2",
}
VALID_DOOM2_TEXTURES = {
    "BIGDOOR2",
    "DOORBLU",
    "DOORRED",
    "DOORTRAK",
    "DOORYEL",
    "METAL",
    "METAL2",
    "STARTAN2",
    "STARTAN3",
    "STEP1",
    "STEP2",
    "STONE2",
    "STONE3",
    "SUPPORT3",
    "SW1COMM",
    "SW1EXIT",
}

DEFAULT_MATERIALS = {
    "floor": "FLOOR0_1",
    "ceiling": "CEIL3_5",
    "wall": "STARTAN2",
    "upper_wall": "STARTAN2",
    "lower_wall": "STARTAN2",
    "door": "BIGDOOR2",
    "door_track": "DOORTRAK",
    "exit_switch": "SW1EXIT",
}

MATERIAL_ID_PRESETS: dict[int, dict[str, str]] = {
    5: {"floor": "FLOOR0_1"},
    6: {"ceiling": "CEIL3_5"},
    7: {"floor": "FLAT20"},
    10: {"wall": "STARTAN2", "upper_wall": "STARTAN2", "lower_wall": "STARTAN2"},
    11: {"wall": "METAL2", "upper_wall": "METAL2", "lower_wall": "METAL2"},
    12: {"wall": "STONE2", "upper_wall": "STONE2", "lower_wall": "STONE2"},
    20: {"door": "BIGDOOR2"},
    21: {"door_track": "DOORTRAK"},
    30: {"exit_switch": "SW1EXIT"},
}

TYPE_PRESETS: dict[str, dict[str, str]] = {
    "techbase": {
        "floor": "FLOOR0_1",
        "ceiling": "CEIL3_5",
        "wall": "STARTAN2",
        "upper_wall": "STARTAN2",
        "lower_wall": "STARTAN2",
        "door": "BIGDOOR2",
        "door_track": "DOORTRAK",
        "exit_switch": "SW1EXIT",
    },
    "industrial": {
        "floor": "FLAT20",
        "ceiling": "CEIL3_5",
        "wall": "METAL2",
        "upper_wall": "METAL2",
        "lower_wall": "METAL2",
        "door": "BIGDOOR2",
        "door_track": "DOORTRAK",
        "exit_switch": "SW1COMM",
    },
    "stone": {
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


def resolve_materials(document: dict[str, Any], space: dict[str, Any] | None = None) -> dict[str, str]:
    """Resolve document/space material declarations to concrete Doom names."""

    resolved = dict(DEFAULT_MATERIALS)
    theme = document.get("theme", {})
    if isinstance(theme, dict):
        primary = theme.get("primary")
        if isinstance(primary, str) and primary in TYPE_PRESETS:
            resolved.update(TYPE_PRESETS[primary])
        _apply_declaration(resolved, theme.get("materials"))
    if space is not None:
        _apply_declaration(resolved, space.get("materials"))
    return resolved


def validate_material_declaration(value: Any, context: str, errors: list[str]) -> None:
    """Validate a material declaration and append human-readable errors."""

    if value is None:
        return
    if not isinstance(value, dict):
        errors.append(f"{context}.materials must be an object")
        return
    for role, raw in value.items():
        if role not in SURFACE_ROLES:
            errors.append(f"{context}.materials has unknown surface role {role!r}")
            continue
        texture = _resolve_value(str(role), raw)
        if texture is None:
            errors.append(f"{context} has invalid {role} material {raw!r}")
        elif not _is_valid_for_role(str(role), texture):
            errors.append(f"{context} has invalid {role} material {texture!r}")


def _apply_declaration(resolved: dict[str, str], value: Any) -> None:
    if not isinstance(value, dict):
        return
    for role, raw in value.items():
        if role not in SURFACE_ROLES:
            continue
        texture = _resolve_value(str(role), raw)
        if texture and _is_valid_for_role(str(role), texture):
            resolved[str(role)] = texture


def _resolve_value(role: str, value: Any) -> str | None:
    if isinstance(value, str):
        if value in TYPE_PRESETS:
            return TYPE_PRESETS[value].get(role)
        return value
    if isinstance(value, int):
        return MATERIAL_ID_PRESETS.get(value, {}).get(role)
    if isinstance(value, dict):
        material_type = value.get("type")
        if isinstance(material_type, str) and material_type in TYPE_PRESETS:
            return TYPE_PRESETS[material_type].get(role)
        texture_id = value.get("texture_id", value.get("id"))
        if isinstance(texture_id, int):
            resolved = MATERIAL_ID_PRESETS.get(texture_id, {}).get(role)
            if resolved:
                return resolved
        key = "flat" if role in FLAT_ROLES else "texture"
        named = value.get(key, value.get("texture"))
        if isinstance(named, str):
            return named
    return None


def _is_valid_for_role(role: str, texture: str) -> bool:
    if role in FLAT_ROLES:
        return texture in VALID_DOOM2_FLATS
    return texture in VALID_DOOM2_TEXTURES
