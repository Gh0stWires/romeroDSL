"""Extract canonical romeroDSL training pairs from WAD graph-cache samples.

v0.6 does not try to recreate exact linedefs from source WADs. The graph cache is
used as weak semantic supervision: theme, scale, key/lock colors, route-gating,
height variation, and combat/resource density become a valid high-level DSL
contract that the compiler can then turn into clean Doom geometry.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

KEY_COLOR_ORDER = ("blue", "red", "yellow")
VALID_THEME_GUESSES = {"techbase", "hell"}


def dsl_from_graph_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Convert one WAD graph-cache sample into a valid canonical DSL document."""

    family = str(sample.get("family") or "unknown")
    map_name = str(sample.get("map") or "MAP01")
    theme = _theme_from_sample(sample)
    colors = _paired_key_lock_colors(sample)
    height_levels = _height_levels(sample)
    counts = _mapping(sample.get("counts"))
    things = _mapping(sample.get("things_by_category"))
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
        "title": f"Extracted {family} {map_name} DSL Seed",
        "scale": _scale_from_sector_count(int(counts.get("sectors", 0) or 0)),
        "difficulty": _difficulty_from_monsters(int(things.get("monsters", 0) or 0)),
        "intended_playtime": "short" if int(counts.get("sectors", 0) or 0) < 64 else "medium",
        "metadata": {
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


def training_record_from_graph_sample(sample: dict[str, Any], *, sample_path: str) -> dict[str, Any]:
    """Build one prompt-to-DSL training record from a graph-cache sample."""

    colors = _paired_key_lock_colors(sample)
    document = dsl_from_graph_sample(sample)
    counts = _mapping(sample.get("counts"))
    progression = _mapping(sample.get("progression"))
    prompt = prompt_from_graph_sample(sample, colors)
    strict = bool(
        progression.get("start_reaches_exit")
        and progression.get("exit_requires_any_lock")
        and counts.get("keys", 0)
        and counts.get("locked_doors", 0)
        and colors
    )
    return {
        "schema": "romerodsl_training_pair_v0.6",
        "source_sample": sample_path,
        "source_wad": sample.get("source_wad"),
        "family": sample.get("family"),
        "map": sample.get("map"),
        "prompt": prompt,
        "dsl": document,
        "labels": {
            "strict_progression_candidate": strict,
            "colors": colors,
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
        },
    }


def build_dsl_training_cache(graph_cache: Path, output: Path, *, strict_only: bool = False) -> dict[str, Any]:
    """Build a manifest and sample files of prompt-to-romeroDSL records."""

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
    theme_counter: Counter[str] = Counter()
    strict_count = 0

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_rel = str(row["sample"])
        source_path = graph_cache / source_rel
        sample = json.loads(source_path.read_text(encoding="utf-8"))
        record = training_record_from_graph_sample(sample, sample_path=source_rel.replace("\\", "/"))
        if strict_only and not record["labels"]["strict_progression_candidate"]:
            continue

        strict_count += int(record["labels"]["strict_progression_candidate"])
        sample_out = samples_dir / f"dsl_{len(records):05d}.json"
        sample_out.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        manifest_record = {
            "path": str(sample_out.relative_to(output)).replace("\\", "/"),
            "source_sample": record["source_sample"],
            "prompt": record["prompt"],
            "family": record["family"],
            "map": record["map"],
            "labels": record["labels"],
            "features": record["features"],
        }
        records.append(manifest_record)
        color_counter.update(record["labels"]["colors"])
        theme_counter.update([record["labels"]["theme"]])

    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema": "romerodsl_training_pair_v0.6",
        "source_graph_cache": str(graph_cache),
        "samples": len(records),
        "strict_progression_candidates": strict_count,
        "strict_only": strict_only,
        "color_counts": dict(color_counter),
        "theme_counts": dict(theme_counter),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def prompt_from_graph_sample(sample: dict[str, Any], colors: list[str] | None = None) -> str:
    """Create a grounded natural-language prompt from observed graph-cache facts."""

    colors = _paired_key_lock_colors(sample) if colors is None else colors
    counts = _mapping(sample.get("counts"))
    things = _mapping(sample.get("things_by_category"))
    theme = _theme_from_sample(sample)
    parts = ["make a Doom map"]
    if theme != "unknown":
        parts.append(f"with a {theme} theme")
    sector_count = int(counts.get("sectors", 0) or 0)
    if sector_count:
        parts.append(f"with a {_scale_from_sector_count(sector_count)} connected room layout")
    for color in colors:
        parts.append(f"with a {color} key before a {color} locked door")
    monster_bucket = _density_bucket(int(things.get("monsters", 0) or 0), 8, 40)
    if monster_bucket != "none":
        parts.append(f"with {monster_bucket} monster encounters")
    if things.get("weapons", 0) or things.get("ammunitions", 0) or things.get("powerups", 0):
        parts.append("with weapons ammo and health supplies")
    if len(_height_levels(sample)) > 1:
        parts.append("with height changes")
    if _mapping(sample.get("progression")).get("exit_requires_any_lock"):
        parts.append("where progression gates the exit")
    return ", ".join(parts)


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
    for color in colors:
        path.extend([f"{color}_key_route", f"{color}_key_room", f"{color}_locked_door"])
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
