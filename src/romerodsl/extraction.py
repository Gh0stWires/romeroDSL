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


def training_record_from_graph_sample(
    sample: dict[str, Any], *, sample_path: str, prompt_variant_index: int = 0
) -> dict[str, Any]:
    """Build one prompt-to-DSL training record from a graph-cache sample."""

    required_colors = _required_key_lock_colors(sample)
    observed_colors = _paired_key_lock_colors(sample)
    document = dsl_from_graph_sample(sample)
    counts = _mapping(sample.get("counts"))
    progression = _mapping(sample.get("progression"))
    prompt_variants = prompt_variants_from_graph_sample(sample)
    prompt = prompt_variants[prompt_variant_index % len(prompt_variants)]
    strict = bool(
        progression.get("start_reaches_exit")
        and progression.get("exit_requires_any_lock")
        and counts.get("keys", 0)
        and counts.get("locked_doors", 0)
        and required_colors
    )
    bucket = _bucket_for_sample(sample)
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
        },
    }


def build_dsl_training_cache(
    graph_cache: Path,
    output: Path,
    *,
    strict_only: bool = False,
    prompt_variants_per_sample: int = 1,
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

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        source_rel = str(row["sample"])
        source_path = graph_cache / source_rel
        sample = json.loads(source_path.read_text(encoding="utf-8"))
        base_record = training_record_from_graph_sample(sample, sample_path=source_rel.replace("\\", "/"))
        if strict_only and not base_record["labels"]["strict_progression_candidate"]:
            continue

        source_count += 1
        for variant_index in range(prompt_variants_per_sample):
            record = training_record_from_graph_sample(
                sample,
                sample_path=source_rel.replace("\\", "/"),
                prompt_variant_index=variant_index,
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


def prompt_from_graph_sample(sample: dict[str, Any], colors: list[str] | None = None) -> str:
    """Create the default grounded natural-language prompt from observed facts."""

    if colors is None:
        return prompt_variants_from_graph_sample(sample, variant_count=1)[0]
    return _prompt_variant(sample, colors=colors, style=0)


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
