"""Evaluation helpers for generated romeroDSL documents."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Iterable, Sequence

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad


PHASE6_BENCHMARK_VERSION = "phase6_no_repair_v1"


def _benchmark_case(
    case_id: str,
    tier: str,
    family: str,
    prompt: str,
    room_count: int,
    keys: Sequence[str] = (),
    *,
    locked_doors: Sequence[str] | None = None,
    min_branch_nodes: int = 0,
    min_cycles: int = 0,
    min_height_levels: int = 1,
    required_features: Sequence[str] = (),
) -> dict[str, Any]:
    """Construct an immutable-by-convention held-out benchmark case."""

    return {
        "id": case_id,
        "complexity_tier": tier,
        "design_family": family,
        "prompt": prompt,
        "expected_room_count": room_count,
        "expected_key_colors": list(keys),
        "expected_locked_door_colors": list(keys if locked_doors is None else locked_doors),
        "max_keys": 0 if not keys else None,
        "max_locked_doors": 0 if locked_doors == () else None,
        "min_branch_nodes": min_branch_nodes,
        "min_cycles": min_cycles,
        "min_height_levels": min_height_levels,
        "required_features": list(required_features),
    }


# These prompts are deliberately separate from training records and the older v0.8
# smoke suite. Keep their ids and wording stable: changing them invalidates benchmark
# comparisons. There are eight held-out design families in each complexity tier.
_PHASE6_CASE_SPECS = [
    ("b01", "basic", "courtyard_route", "Build a four-room courtyard route with a red keycard and a red locked exit door.", 4, ("red",), (), 0, 0, 1, ()),
    ("b02", "basic", "maintenance_loop", "Build a three-room maintenance loop with no keys and no locked doors, connecting the start to the exit.", 3, (), (), 0, 1, 1, ()),
    ("b03", "basic", "blue_storage", "Build a five-room blue storage route: place a blue key before one blue locked doorway and put the exit beyond it.", 5, ("blue",), ("blue",), 0, 0, 1, ()),
    ("b04", "basic", "yellow_shaft", "Build a four-room yellow access map with stairs, a yellow key, and a matching yellow locked exit.", 4, ("yellow",), ("yellow",), 0, 0, 2, ("stairs",)),
    ("b05", "basic", "plain_arena", "Build a three-room plain arena route with no key requirement, no locked doors, and a small combat encounter.", 3, (), (), 0, 0, 1, ()),
    ("b06", "basic", "red_armory", "Build a five-room red armory route with a red key in a side room and a red locked final room.", 5, ("red",), ("red",), 1, 0, 1, ()),
    ("b07", "basic", "blue_pit", "Build a four-room blue pit map with a shallow pit, a blue key, and a blue lock on the exit route.", 4, ("blue",), ("blue",), 0, 0, 2, ("pit",)),
    ("b08", "basic", "yellow_hub", "Build a five-room yellow hub-and-branch map with one yellow key and one matching locked door.", 5, ("yellow",), ("yellow",), 1, 0, 1, ()),
    ("i01", "intermediate", "red_crossroads", "Build a seven-room industrial crossroads with branching combat rooms, a red key, and a red locked exit gate.", 7, ("red",), ("red",), 1, 0, 1, ()),
    ("i02", "intermediate", "blue_ring", "Build an eight-room blue ring layout with a loop, a blue key on the loop, and a blue locked door leading to the exit.", 8, ("blue",), ("blue",), 0, 1, 1, ()),
    ("i03", "intermediate", "yellow_stairwell", "Build a six-room yellow industrial stairwell with multiple floor heights, a yellow key, and a required yellow lock.", 6, ("yellow",), ("yellow",), 0, 0, 3, ("stairs",)),
    ("i04", "intermediate", "dual_key_depot", "Build a nine-room depot where a blue key opens the blue gate and a red key opens the red exit gate in sequence.", 9, ("blue", "red"), ("blue", "red"), 1, 0, 1, ()),
    ("i05", "intermediate", "secret_catwalk", "Build a seven-room exploration map with a non-required secret side room, a yellow key, and a yellow locked final corridor.", 7, ("yellow",), ("yellow",), 1, 0, 1, ("secret",)),
    ("i06", "intermediate", "arena_loop", "Build an eight-room arena loop with a blue key before a blue locked exit and at least one branch for resources.", 8, ("blue",), ("blue",), 1, 1, 1, ()),
    ("i07", "intermediate", "red_lift", "Build a six-room red lift approach with a raised platform, a red key, and a red locked door on the critical path.", 6, ("red",), ("red",), 0, 0, 2, ("platform",)),
    ("i08", "intermediate", "no_lock_lab", "Build a seven-room laboratory with a connected start-to-exit route, varied rooms, and explicitly no keys or locked doors.", 7, (), (), 1, 0, 1, ()),
    ("a01", "advanced", "three_gate_fortress", "Build a twelve-room fortress with sequential blue, red, then yellow keys and matching required locked gates before the exit.", 12, ("blue", "red", "yellow"), ("blue", "red", "yellow"), 1, 0, 1, ()),
    ("a02", "advanced", "hell_loop", "Build an eleven-room hell loop with a red key branch, a red locked exit, multiple combat spaces, and a return route.", 11, ("red",), ("red",), 1, 1, 1, ()),
    ("a03", "advanced", "vertical_uac", "Build a thirteen-room vertical UAC complex with stairs, a pit, a raised platform, and a blue locked final exit.", 13, ("blue",), ("blue",), 1, 1, 3, ("stairs", "pit", "platform")),
    ("a04", "advanced", "industrial_maze", "Build a ten-room branching industrial maze with a yellow key behind a combat room and a yellow required gate.", 10, ("yellow",), ("yellow",), 2, 0, 1, ()),
    ("a05", "advanced", "dual_route", "Build a fourteen-room map with two connected routes, a blue key, a blue gate, and a secret reward room that is not required for the exit.", 14, ("blue",), ("blue",), 2, 1, 1, ("secret",)),
    ("a06", "advanced", "red_blue_bastion", "Build a twelve-room bastion where the red key opens a red gate before the blue key, then the blue gate leads to the exit.", 12, ("red", "blue"), ("red", "blue"), 1, 0, 1, ()),
    ("a07", "advanced", "elevation_gallery", "Build a fifteen-room map with three distinct height levels, stairs between them, a yellow key, and a yellow locked exit.", 15, ("yellow",), ("yellow",), 2, 0, 3, ("stairs",)),
    ("a08", "advanced", "resource_bypass", "Build a ten-room combat map with a loopback resource route, no locked doors, no required keys, and a reachable exit.", 10, (), (), 1, 1, 1, ()),
    ("e01", "expert", "four_stage_citadel", "Build a twenty-room citadel with sequential blue, red, yellow, and blue-key gates, branches, and a final exit beyond all required gates.", 20, ("blue", "red", "yellow"), ("blue", "red", "yellow"), 2, 1, 1, ()),
    ("e02", "expert", "multi_loop_ruins", "Build an eighteen-room ruin with two independent loops, a red key side route, a red locked exit, secrets, and varied elevation.", 18, ("red",), ("red",), 2, 2, 2, ("secret",)),
    ("e03", "expert", "vertical_foundry", "Build a twenty-two-room foundry spanning four height levels with stairs, pits, platforms, a blue key, and a blue required exit lock.", 22, ("blue",), ("blue",), 2, 1, 4, ("stairs", "pit", "platform")),
    ("e04", "expert", "three_wing_labyrinth", "Build a twenty-four-room labyrinth with three branching wings, a yellow key in one wing, a yellow gate, and a loopback to the hub.", 24, ("yellow",), ("yellow",), 3, 2, 1, ()),
    ("e05", "expert", "dual_sequence_keep", "Build a sixteen-room keep where blue then red keys open matching sequential doors, with a secret reward branch and an exit after red.", 16, ("blue", "red"), ("blue", "red"), 2, 1, 1, ("secret",)),
    ("e06", "expert", "open_expedition", "Build a nineteen-room expedition map with multiple loops, varied height features, combat branches, and explicitly no keys or locked doors.", 19, (), (), 2, 2, 2, ()),
    ("e07", "expert", "colorless_arena", "Build a seventeen-room arena complex with three combat branches, a raised center, no keycards, no locked doors, and a connected exit route.", 17, (), (), 2, 1, 2, ("platform",)),
    ("e08", "expert", "final_procession", "Build a twenty-three-room procession with blue, yellow, then red keys on the critical path, matching gates, loops, secrets, and a final exit.", 23, ("blue", "yellow", "red"), ("blue", "yellow", "red"), 2, 2, 1, ("secret",)),
]
PHASE6_BENCHMARK_CASES: list[dict[str, Any]] = [
    _benchmark_case(*spec[:6], locked_doors=spec[6], min_branch_nodes=spec[7], min_cycles=spec[8], min_height_levels=spec[9], required_features=spec[10])
    for spec in _PHASE6_CASE_SPECS
]


V08_PROMPT_CASES: list[dict[str, Any]] = [
    {
        "prompt": "make a compact techbase Doom map with a red key before a red locked door, medium monsters, ammo, health, and the exit gated behind that door",
        "expected_key_colors": {"red"},
        "expected_locked_door_colors": {"red"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a small hell-themed Doom map with a blue key in a trap room, a blue locked door, demons and imps, and the exit beyond the locked door",
        "expected_key_colors": {"blue"},
        "expected_locked_door_colors": {"blue"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a UAC storage depot map with a yellow key, a yellow locked exit wing, light combat, shotgun ammo, and health before the lock",
        "expected_key_colors": {"yellow"},
        "expected_locked_door_colors": {"yellow"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a tiny connected Doom map with no locked door, one clear route from start to exit, low monsters, and no required key",
        "expected_key_colors": set(),
        "expected_locked_door_colors": set(),
        "min_keys": 0,
        "min_locked_doors": 0,
        "max_keys": 0,
        "max_locked_doors": 0,
    },
    {
        "prompt": "make a Doom map with two keys, blue then red, where the blue door gates the red key and the red door gates the exit",
        "expected_key_colors": {"blue", "red"},
        "expected_locked_door_colors": {"blue", "red"},
        "min_keys": 2,
        "min_locked_doors": 2,
    },
    {
        "prompt": "make a vertical techbase map with stairs, a shallow pit, a raised exit platform, a red key, and a red locked door on the critical path",
        "expected_key_colors": {"red"},
        "expected_locked_door_colors": {"red"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a low-combat exploration Doom map with a yellow key side room, optional secret armor, and a yellow locked exit corridor",
        "expected_key_colors": {"yellow"},
        "expected_locked_door_colors": {"yellow"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a high-pressure arena Doom map with a blue key visible early, resources before the fight, and a blue locked door that gates the exit",
        "expected_key_colors": {"blue"},
        "expected_locked_door_colors": {"blue"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a loopback base map where the player finds the red key on a side route, returns to the hub, and opens a red locked door to the exit",
        "expected_key_colors": {"red"},
        "expected_locked_door_colors": {"red"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
    {
        "prompt": "make a small industrial Doom map with branching rooms, medium monster encounters, one weapon pickup, and a blue key before a blue locked final door",
        "expected_key_colors": {"blue"},
        "expected_locked_door_colors": {"blue"},
        "min_keys": 1,
        "min_locked_doors": 1,
    },
]


def v08_prompt_suite() -> list[str]:
    """Return a fixed multi-prompt suite for v0.8 Qwen adapter evaluation."""

    return [str(case["prompt"]) for case in V08_PROMPT_CASES]


def v08_prompt_expectations() -> list[dict[str, Any]]:
    """Return intent expectations aligned with :func:`v08_prompt_suite`."""

    return [{key: value for key, value in case.items() if key != "prompt"} for case in V08_PROMPT_CASES]


def evaluate_prompt_intent(result: dict[str, Any], expectation: dict[str, Any]) -> dict[str, Any]:
    """Check whether a generated result satisfies prompt-level key/lock intent."""

    errors: list[str] = []
    key_colors = set((result.get("key_colors") or {}).keys())
    locked_door_colors = set((result.get("locked_door_colors") or {}).keys())
    keys = int(result.get("keys") or 0)
    locked_doors = int(result.get("locked_doors") or 0)

    for color in sorted(expectation.get("expected_key_colors", set())):
        if color not in key_colors:
            errors.append(f"missing {color} key")
    for color in sorted(expectation.get("expected_locked_door_colors", set())):
        if color not in locked_door_colors:
            errors.append(f"missing {color} locked door")

    min_keys = expectation.get("min_keys")
    if isinstance(min_keys, int) and keys < min_keys:
        errors.append(f"keys < {min_keys}")
    max_keys = expectation.get("max_keys")
    if isinstance(max_keys, int) and keys > max_keys:
        errors.append(f"keys > {max_keys}")
    min_locked_doors = expectation.get("min_locked_doors")
    if isinstance(min_locked_doors, int) and locked_doors < min_locked_doors:
        errors.append(f"locked_doors < {min_locked_doors}")
    max_locked_doors = expectation.get("max_locked_doors")
    if isinstance(max_locked_doors, int) and locked_doors > max_locked_doors:
        errors.append(f"locked_doors > {max_locked_doors}")

    for field in ("dsl_valid", "wad_valid", "key_lock_pairs_valid"):
        if result.get(field) is not True:
            errors.append(f"{field} is not true")
    if result.get("zandronum_exit_code") != 0:
        errors.append("zandronum_exit_code != 0")

    return {"prompt_intent_valid": not errors, "prompt_intent_errors": errors}


KEY_COLORS = ("blue", "red", "yellow")
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def requested_key_colors_from_prompt(prompt: str) -> list[str]:
    """Return key colors explicitly requested by the prompt."""

    text = prompt.lower()
    requested = {
        color
        for color in KEY_COLORS
        if re.search(rf"\b{color}\s+(?:key|keycard|skull\s*key|skull)\b", text)
    }
    return sorted(requested)


def requested_space_count_from_prompt(prompt: str) -> int | None:
    """Return an explicit requested room/space count from prompt text, if any."""

    text = prompt.lower()
    numeric = re.search(r"\b(\d{1,2})\s*[- ]?\s*(?:room|rooms|space|spaces)\b", text)
    if numeric:
        return int(numeric.group(1))
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s*[- ]?\s*(?:room|rooms|space|spaces)\b", text):
            return value
    return None


def evaluate_generated_dsl_intent(prompt: str, dsl: dict[str, Any]) -> dict[str, Any]:
    """Summarize prompt-contract checks directly from generated DSL JSON."""

    progression = dsl.get("progression", {}) if isinstance(dsl.get("progression"), dict) else {}
    spaces = dsl.get("spaces", []) if isinstance(dsl.get("spaces"), list) else []
    connections = dsl.get("connections", []) if isinstance(dsl.get("connections"), list) else []
    requested_keys = set(requested_key_colors_from_prompt(prompt))
    requested_space_count = requested_space_count_from_prompt(prompt)
    generated_key_colors = {
        str(key.get("color"))
        for key in progression.get("keys", [])
        if isinstance(key, dict) and isinstance(key.get("color"), str)
    }
    generated_gate_colors = {
        str(gate.get("key"))
        for gate in progression.get("gates", [])
        if isinstance(gate, dict) and isinstance(gate.get("key"), str)
    }
    generated_gate_colors.update(
        str(connection.get("key"))
        for connection in connections
        if isinstance(connection, dict)
        and connection.get("type") in {"locked_door", "keyed_door"}
        and isinstance(connection.get("key"), str)
    )
    space_count = len(spaces)
    space_count_exact_match = requested_space_count is None or space_count == requested_space_count
    space_count_tolerance = _space_count_tolerance(requested_space_count)
    space_count_pass = requested_space_count is None or abs(space_count - requested_space_count) <= space_count_tolerance
    if requested_space_count is None:
        space_count_note = "not_requested"
    elif space_count_exact_match:
        space_count_note = "exact_match"
    elif space_count_pass:
        space_count_note = "soft_pass_within_tolerance"
    else:
        space_count_note = "outside_tolerance"
    missing_requested_keys = requested_keys - generated_key_colors
    missing_requested_gates = requested_keys - generated_gate_colors
    design_intent = dsl.get("design_intent", {}) if isinstance(dsl.get("design_intent"), dict) else {}
    public_design_notes = {
        key: design_intent.get(key)
        for key in ("summary", "topology", "progression_style", "combat_style", "reward_style")
        if isinstance(design_intent.get(key), str)
    }
    return {
        "requested_key_colors": sorted(requested_keys),
        "generated_key_colors": sorted(generated_key_colors),
        "generated_gate_colors": sorted(generated_gate_colors),
        "missing_requested_keys": sorted(missing_requested_keys),
        "missing_requested_gates": sorted(missing_requested_gates),
        "requested_space_count": requested_space_count,
        "space_count": space_count,
        "space_count_exact_match": space_count_exact_match,
        "space_count_tolerance": space_count_tolerance,
        "space_count_pass": space_count_pass,
        "space_count_note": space_count_note,
        "critical_path": progression.get("critical_path", []) if isinstance(progression.get("critical_path"), list) else [],
        "public_design_notes": public_design_notes,
        "prompt_contract_valid": not missing_requested_keys and not missing_requested_gates and space_count_pass,
    }


def _space_count_tolerance(requested_space_count: int | None) -> int:
    if requested_space_count is None:
        return 0
    return max(1, requested_space_count // 4)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract one JSON object from raw model text."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.I)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("generated JSON is not an object")
    return value


def template_similarity_report(generated: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    """Return lightweight overlap signals against a known example/template DSL."""

    generated_spaces = _ids(generated.get("spaces"))
    template_spaces = _ids(template.get("spaces"))
    generated_connections = _connection_ids(generated.get("connections"))
    template_connections = _connection_ids(template.get("connections"))
    generated_key_colors = _key_colors(generated.get("progression"))
    template_key_colors = _key_colors(template.get("progression"))

    generated_roles = _field_values(generated.get("spaces"), "role")
    template_roles = _field_values(template.get("spaces"), "role")
    generated_shapes = _field_values(generated.get("spaces"), "shape")
    template_shapes = _field_values(template.get("spaces"), "shape")
    generated_theme = _theme_primary(generated.get("theme"))
    template_theme = _theme_primary(template.get("theme"))

    space_overlap = _jaccard(generated_spaces, template_spaces)
    connection_overlap = _jaccard(generated_connections, template_connections)
    key_color_overlap = _jaccard(generated_key_colors, template_key_colors)
    role_overlap = _jaccard(generated_roles, template_roles)
    shape_overlap = _jaccard(generated_shapes, template_shapes)
    theme_overlap = 1.0 if generated_theme and generated_theme == template_theme else 0.0
    space_count_similarity = _count_similarity(len(generated_spaces), len(template_spaces))
    connection_count_similarity = _count_similarity(len(generated_connections), len(template_connections))
    gate_count_similarity = _count_similarity(
        len((generated.get("progression") or {}).get("gates") or []),
        len((template.get("progression") or {}).get("gates") or []),
    )
    copied_title = str(generated.get("title", "")).strip().lower() == str(template.get("title", "")).strip().lower()
    source_seed_title = "extracted" in str(generated.get("title", "")).lower() and "seed" in str(generated.get("title", "")).lower()

    # Lexical leakage catches exact copied titles/ids. Structural leakage catches the
    # case that matters visually: different ids/colors, but the same room-role and
    # room-shape skeleton compiled into a near-identical map.
    lexical_score = (space_overlap + connection_overlap + key_color_overlap + (1.0 if copied_title else 0.0)) / 4.0
    structural_score = (
        role_overlap
        + shape_overlap
        + theme_overlap
        + space_count_similarity
        + connection_count_similarity
        + gate_count_similarity
    ) / 6.0
    score = max(lexical_score, structural_score)
    if score >= 0.65 or source_seed_title:
        label = "high"
    elif score >= 0.4:
        label = "medium"
    else:
        label = "low"
    return {
        "template_similarity": label,
        "template_similarity_score": round(score, 3),
        "template_lexical_score": round(lexical_score, 3),
        "template_structural_score": round(structural_score, 3),
        "copied_source_title": copied_title or source_seed_title,
        "space_id_overlap": round(space_overlap, 3),
        "connection_id_overlap": round(connection_overlap, 3),
        "key_color_overlap": round(key_color_overlap, 3),
        "role_overlap": round(role_overlap, 3),
        "shape_overlap": round(shape_overlap, 3),
        "theme_overlap": round(theme_overlap, 3),
        "space_count_similarity": round(space_count_similarity, 3),
        "connection_count_similarity": round(connection_count_similarity, 3),
        "gate_count_similarity": round(gate_count_similarity, 3),
        "generated_space_ids": sorted(generated_spaces),
        "generated_connection_ids": sorted(generated_connections),
        "generated_key_colors": sorted(generated_key_colors),
    }


def _ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item["id"] for item in value if isinstance(item, dict) and isinstance(item.get("id"), str)}


def _connection_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    ids: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("id"), str):
            ids.add(item["id"])
            continue
        source = item.get("from")
        target = item.get("to")
        if isinstance(source, str) and isinstance(target, str):
            ids.add(f"{source}->{target}")
    return ids


def _key_colors(progression: Any) -> set[str]:
    if not isinstance(progression, dict) or not isinstance(progression.get("keys"), list):
        return set()
    return {key["color"] for key in progression["keys"] if isinstance(key, dict) and isinstance(key.get("color"), str)}


def _field_values(value: Any, field: str) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item[field] for item in value if isinstance(item, dict) and isinstance(item.get(field), str)}


def _theme_primary(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    primary = value.get("primary")
    return primary if isinstance(primary, str) else None


def _count_similarity(left: int, right: int) -> float:
    if left == right == 0:
        return 1.0
    maximum = max(left, right)
    if maximum == 0:
        return 0.0
    return max(0.0, 1.0 - abs(left - right) / maximum)


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


# ---------------------------------------------------------------------------
# Phase 6 strict no-repair benchmark


def evaluate_no_repair_output(
    case: dict[str, Any],
    raw_output: str,
    *,
    training_templates: Iterable[dict[str, Any]] | None = None,
    wad_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate one model response without modifying or repairing it.

    Parsing is deliberately stricter than :func:`extract_json_object`: surrounding
    prose and markdown are failures. Compilation and WAD validation only run after
    the unmodified JSON passes the DSL schema. ``raw_output`` is echoed in the
    result so callers can persist an auditable copy beside the report.
    """

    prompt = str(case.get("prompt", ""))
    result: dict[str, Any] = {
        "benchmark_version": PHASE6_BENCHMARK_VERSION,
        "case_id": str(case.get("id", "unknown")),
        "complexity_tier": case.get("complexity_tier"),
        "design_family": case.get("design_family"),
        "prompt": prompt,
        "raw_output": raw_output,
        "evaluation_mode": "strict_no_repair",
        "repair_applied": False,
        "iterative_correction_success": None,
        "json_parse_success": False,
        "json_valid": False,
        "dsl_schema_valid": False,
        "dsl_valid": False,
        "exact_prompt_constraints_satisfied": False,
        "prompt_compliance": False,
        "correct_room_count": False,
        "correct_key_lock_behavior": False,
        "valid_progression_and_connectivity": False,
        "geometry_compilation_success": False,
        "wad_export_valid": False,
        "wad_valid": False,
        "failure_categories": [],
        "errors": [],
        "architectural_diversity": {"available": False, "reason": "DSL did not pass validation"},
        "template_similarity": {"available": False, "nearest": None, "scores": []},
        "progression": {"available": False, "reason": "DSL did not pass validation"},
    }
    document: dict[str, Any] | None = None
    try:
        parsed = json.loads(raw_output.strip())
        if not isinstance(parsed, dict):
            raise ValueError("JSON completion must be an object")
        document = parsed
        result["json_parse_success"] = True
        result["json_valid"] = True
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        result["errors"].append(f"JSON parse failure: {exc}")
        result["failure_categories"] = ["syntax_failure"]
        result["first_pass_success"] = False
        return result

    validation = validate_document(document)
    result["schema_errors"] = list(validation.errors)
    result["schema_warnings"] = list(validation.warnings)
    result["dsl_schema_valid"] = validation.valid
    result["dsl_valid"] = validation.valid
    if not validation.valid:
        result["errors"].extend(validation.errors)
        result["failure_categories"] = _failure_categories_for_schema(validation.errors)
        result["first_pass_success"] = False
        return result

    intent = evaluate_generated_dsl_intent(prompt, document)
    constraints = _evaluate_benchmark_constraints(case, document, intent)
    result["prompt_intent"] = intent
    result["prompt_constraints"] = constraints
    result["exact_prompt_constraints_satisfied"] = bool(constraints["valid"])
    result["prompt_compliance"] = bool(constraints["valid"])
    result["correct_room_count"] = bool(constraints["room_count_exact"])
    result["correct_key_lock_behavior"] = bool(constraints["key_lock_behavior_valid"])

    progression = _evaluate_progression_and_connectivity(document)
    result["progression"] = progression
    result["valid_progression_and_connectivity"] = bool(progression["valid"])

    # Import locally to avoid the metrics -> evaluation import cycle.
    from romerodsl.metrics import structural_metrics

    structural = structural_metrics(document, prompt=prompt, dsl_valid=True)
    architecture = _architectural_diversity_metrics(structural)
    result["architectural_diversity"] = architecture

    templates = list(training_templates or [])
    result["template_similarity"] = _nearest_template_similarity(document, templates)

    plan: dict[str, Any] | None = None
    try:
        plan = compile_to_plan(document)
        result["geometry_compilation_success"] = True
        result["plan"] = plan
    except Exception as exc:  # compiler errors are benchmark evidence, not harness crashes
        result["errors"].append(f"Geometry compilation failure: {exc}")

    if plan is not None:
        export_path: Path
        temporary: TemporaryDirectory[str] | None = None
        if wad_path is None:
            temporary = TemporaryDirectory(prefix="romerodsl_phase6_")
            export_path = Path(temporary.name) / "MAP01.wad"
        else:
            export_path = Path(wad_path)
        try:
            export_report = compile_to_wad(document, export_path)
            wad_report = validate_wad(export_path)
            result["wad_export_report"] = export_report
            result["wad_validation"] = wad_report
            result["wad_export_valid"] = bool(export_report.get("valid", True) and wad_report.get("valid"))
            result["wad_valid"] = result["wad_export_valid"]
            result["correct_key_lock_behavior"] = bool(
                result["correct_key_lock_behavior"] and wad_report.get("key_lock_pairs_valid", False)
            )
            if not result["wad_export_valid"]:
                result["errors"].append("WAD export or validation report is invalid")
        except Exception as exc:  # invalid geometry/materials must be visible per case
            result["errors"].append(f"WAD export failure: {exc}")
        finally:
            if temporary is not None:
                temporary.cleanup()

    if not result["exact_prompt_constraints_satisfied"]:
        result["errors"].append("one or more exact prompt constraints were not satisfied")
    if not result["correct_key_lock_behavior"]:
        result["errors"].append("key/lock behavior does not match the held-out contract")
    if not result["valid_progression_and_connectivity"]:
        result["errors"].append("progression or connectivity is invalid")
    if architecture["degenerate"]:
        result["errors"].append("architecture is repetitive or degenerate")

    result["failure_categories"] = _failure_categories_for_result(result)
    result["first_pass_success"] = all(
        bool(result[field])
        for field in (
            "json_parse_success",
            "dsl_schema_valid",
            "exact_prompt_constraints_satisfied",
            "correct_room_count",
            "correct_key_lock_behavior",
            "valid_progression_and_connectivity",
            "geometry_compilation_success",
            "wad_export_valid",
        )
    ) and not architecture["degenerate"]
    return result


def run_no_repair_benchmark(
    generate: Callable[[str], str],
    output_dir: Path,
    *,
    cases: Sequence[dict[str, Any]] | None = None,
    training_templates: Iterable[dict[str, Any]] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run and persist the permanent Phase 6 benchmark.

    ``generate`` receives only the held-out prompt. Each exact model response is
    written to ``<case-id>/raw_output.txt`` before its report is written. No
    correction, normalization, JSON extraction, or compiler repair is performed.
    The default suite is permanent and contains 32 cases, eight per complexity tier.
    Existing non-empty output directories are protected unless ``overwrite=True``.
    """

    selected = list(PHASE6_BENCHMARK_CASES if cases is None else cases)
    if cases is None and len(selected) < 30:
        raise ValueError("the permanent Phase 6 benchmark must contain at least 30 cases")
    if len({str(case.get("id")) for case in selected}) != len(selected):
        raise ValueError("benchmark case ids must be unique")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"benchmark output directory is not empty: {output_dir}; pass overwrite=True to replace it"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for case in selected:
        case_dir = output_dir / str(case["id"])
        case_dir.mkdir(parents=True, exist_ok=True)
        raw_output = generate(str(case["prompt"]))
        if not isinstance(raw_output, str):
            raw_output = str(raw_output)
        (case_dir / "raw_output.txt").write_text(raw_output, encoding="utf-8")
        result = evaluate_no_repair_output(
            case,
            raw_output,
            training_templates=training_templates,
            wad_path=case_dir / "generated.wad",
        )
        (case_dir / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        results.append(result)

    failure_counts: Counter[str] = Counter(
        category for result in results for category in result.get("failure_categories", [])
    )
    tier_summary: dict[str, dict[str, int]] = {}
    for result in results:
        tier = str(result.get("complexity_tier", "unknown"))
        summary = tier_summary.setdefault(tier, {"total": 0, "first_pass_success": 0})
        summary["total"] += 1
        summary["first_pass_success"] += int(bool(result.get("first_pass_success")))
    report = {
        "benchmark_version": PHASE6_BENCHMARK_VERSION,
        "evaluation_mode": "strict_no_repair",
        "repair_applied": False,
        "iterative_correction_evaluated": False,
        "output_dir": str(output_dir),
        "counts": {
            "total": len(results),
            "first_pass_success": sum(bool(result.get("first_pass_success")) for result in results),
            "json_parse_success": sum(bool(result.get("json_parse_success")) for result in results),
            "dsl_schema_valid": sum(bool(result.get("dsl_schema_valid")) for result in results),
            "wad_export_valid": sum(bool(result.get("wad_export_valid")) for result in results),
        },
        "failure_category_counts": dict(sorted(failure_counts.items())),
        "tier_summary": tier_summary,
        "cases": results,
    }
    (output_dir / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _evaluate_benchmark_constraints(
    case: dict[str, Any], document: dict[str, Any], intent: dict[str, Any]
) -> dict[str, Any]:
    spaces = document.get("spaces") if isinstance(document.get("spaces"), list) else []
    progression = document.get("progression") if isinstance(document.get("progression"), dict) else {}
    keys = progression.get("keys") if isinstance(progression.get("keys"), list) else []
    gates = progression.get("gates") if isinstance(progression.get("gates"), list) else []
    generated_keys = {str(item.get("color")) for item in keys if isinstance(item, dict)}
    generated_gates = {
        str(item.get("key"))
        for item in gates
        if isinstance(item, dict) and item.get("type") in {"locked_door", "keyed_door"}
    }
    expected_keys = {str(color) for color in case.get("expected_key_colors", [])}
    expected_gates = {str(color) for color in case.get("expected_locked_door_colors", [])}
    expected_count = case.get("expected_room_count")
    room_count_exact = isinstance(expected_count, int) and len(spaces) == expected_count
    no_extra_keys = case.get("max_keys") != 0 or not generated_keys
    no_extra_gates = case.get("max_locked_doors") != 0 or not generated_gates
    missing_features: list[str] = []
    feature_text = json.dumps(document, sort_keys=True).lower()
    for feature in case.get("required_features", []):
        if str(feature).lower() not in feature_text:
            missing_features.append(str(feature))
    directed: dict[str, set[str]] = {}
    undirected: set[tuple[str, str]] = set()
    for connection in document.get("connections", []):
        if not isinstance(connection, dict):
            continue
        source, target = connection.get("from"), connection.get("to")
        if not isinstance(source, str) or not isinstance(target, str):
            continue
        directed.setdefault(source, set()).add(target)
        undirected.add((min(source, target), max(source, target)))
    branch_nodes = sum(len(targets) > 1 for targets in directed.values())
    graph_for_cycles: dict[str, set[str]] = {
        str(space.get("id")): set()
        for space in spaces
        if isinstance(space, dict) and isinstance(space.get("id"), str)
    }
    for source, target in undirected:
        graph_for_cycles.setdefault(source, set()).add(target)
        graph_for_cycles.setdefault(target, set()).add(source)
    graph_components = _graph_components(graph_for_cycles)
    cycle_count = max(0, len(undirected) - len(graph_for_cycles) + len(graph_components))
    height_levels: set[int] = set()
    for space in spaces:
        if not isinstance(space, dict):
            continue
        for field in ("floor_height", "ceiling_height"):
            if isinstance(space.get(field), (int, float)):
                height_levels.add(int(space[field]))
        topology = space.get("height_topology")
        features = topology.get("features", []) if isinstance(topology, dict) else []
        for feature in features if isinstance(features, list) else []:
            if not isinstance(feature, dict):
                continue
            for field in ("floor_height", "ceiling_height"):
                if isinstance(feature.get(field), (int, float)):
                    height_levels.add(int(feature[field]))
    branch_pass = branch_nodes >= int(case.get("min_branch_nodes", 0) or 0)
    cycle_pass = cycle_count >= int(case.get("min_cycles", 0) or 0)
    height_pass = len(height_levels) >= int(case.get("min_height_levels", 1) or 1)
    key_lock_behavior_valid = (
        generated_keys == expected_keys
        and generated_gates == expected_gates
        and no_extra_keys
        and no_extra_gates
        and not (expected_gates - generated_keys)
    )
    valid = (
        bool(intent.get("prompt_contract_valid", True))
        and room_count_exact
        and key_lock_behavior_valid
        and branch_pass
        and cycle_pass
        and height_pass
        and not missing_features
    )
    return {
        "valid": valid,
        "room_count_exact": room_count_exact,
        "expected_room_count": expected_count,
        "actual_room_count": len(spaces),
        "expected_key_colors": sorted(expected_keys),
        "generated_key_colors": sorted(generated_keys),
        "expected_locked_door_colors": sorted(expected_gates),
        "generated_locked_door_colors": sorted(generated_gates),
        "key_lock_behavior_valid": key_lock_behavior_valid,
        "branch_nodes": branch_nodes,
        "min_branch_nodes": int(case.get("min_branch_nodes", 0) or 0),
        "branch_requirement_pass": branch_pass,
        "cycle_count": cycle_count,
        "min_cycles": int(case.get("min_cycles", 0) or 0),
        "cycle_requirement_pass": cycle_pass,
        "height_level_count": len(height_levels),
        "min_height_levels": int(case.get("min_height_levels", 1) or 1),
        "height_requirement_pass": height_pass,
        "missing_required_features": missing_features,
        "prompt_contract_valid": bool(intent.get("prompt_contract_valid", True)),
    }


def _evaluate_progression_and_connectivity(document: dict[str, Any]) -> dict[str, Any]:
    progression = document.get("progression") if isinstance(document.get("progression"), dict) else {}
    spaces = {
        str(space.get("id"))
        for space in document.get("spaces", [])
        if isinstance(space, dict) and isinstance(space.get("id"), str)
    }
    connections = [item for item in document.get("connections", []) if isinstance(item, dict)]
    start = progression.get("start")
    exit_space = progression.get("exit")
    graph: dict[str, set[str]] = {space: set() for space in spaces}
    for connection in connections:
        source, target = connection.get("from"), connection.get("to")
        if isinstance(source, str) and isinstance(target, str):
            graph.setdefault(source, set()).add(target)
            graph.setdefault(target, set()).add(source)
    reachable = _reachable_nodes(graph, start if isinstance(start, str) else "")
    errors: list[str] = []
    if not isinstance(start, str) or not isinstance(exit_space, str) or exit_space not in reachable:
        errors.append("exit is not reachable from start")
    keys = [item for item in progression.get("keys", []) if isinstance(item, dict)]
    gates = [item for item in progression.get("gates", []) if isinstance(item, dict)]
    key_locations = {
        str(item.get("color")): str(item.get("location"))
        for item in keys
        if isinstance(item.get("color"), str) and isinstance(item.get("location"), str)
    }
    for gate in gates:
        if gate.get("type") not in {"locked_door", "keyed_door"} or not gate.get("required_for_exit"):
            continue
        color = gate.get("key")
        edge = _gate_edge(gate, connections)
        if not isinstance(color, str) or color not in key_locations:
            errors.append(f"missing key before required {color} gate")
            continue
        if edge is None:
            errors.append(f"required gate {gate.get('id')} has no concrete connection")
            continue
        before = _reachable_nodes(graph, str(start), blocked_edge=edge)
        if key_locations[color] not in before:
            errors.append(f"{color} key is not reachable before its gate")
        if isinstance(exit_space, str) and exit_space in before:
            errors.append(f"exit is reachable without {color} gate")
    connected = len(_graph_components(graph)) <= 1 and bool(graph)
    if not connected:
        errors.append("space graph is disconnected")
    return {
        "valid": not errors,
        "connected": connected,
        "start": start,
        "exit": exit_space,
        "reachable_from_start": sorted(reachable),
        "required_gate_count": sum(
            item.get("type") in {"locked_door", "keyed_door"} and bool(item.get("required_for_exit"))
            for item in gates
        ),
        "errors": errors,
    }


def _reachable_nodes(graph: dict[str, set[str]], start: str, blocked_edge: tuple[str, str] | None = None) -> set[str]:
    if not start:
        return set()
    reachable: set[str] = set()
    frontier = [start]
    while frontier:
        node = frontier.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for neighbor in graph.get(node, set()):
            if blocked_edge is not None and {node, neighbor} == set(blocked_edge):
                continue
            if neighbor not in reachable:
                frontier.append(neighbor)
    return reachable


def _gate_edge(gate: dict[str, Any], connections: list[dict[str, Any]]) -> tuple[str, str] | None:
    gate_id = gate.get("id")
    blocked = gate.get("blocks")
    for connection in connections:
        if gate_id and connection.get("id") == gate_id:
            source, target = connection.get("from"), connection.get("to")
            if isinstance(source, str) and isinstance(target, str):
                return source, target
    if isinstance(blocked, str) and "->" in blocked:
        source, target = (part.strip() for part in blocked.split("->", 1))
        if source and target:
            return source, target
    return None


def _graph_components(graph: dict[str, set[str]]) -> list[set[str]]:
    remaining = set(graph)
    components: list[set[str]] = []
    while remaining:
        start = min(remaining)
        component = _reachable_nodes(graph, start)
        components.append(component)
        remaining -= component
    return components


def _architectural_diversity_metrics(structural: dict[str, Any]) -> dict[str, Any]:
    graph = structural.get("graph", {}) if isinstance(structural.get("graph"), dict) else {}
    space_count = int(structural.get("space_count", 0) or 0)
    shapes = structural.get("unique_room_shape_count", 0)
    roles = len(structural.get("room_role_counts", {}) or {})
    features = int(
        (structural.get("height", {}) or {}).get("feature_count", 0)
        if isinstance(structural.get("height"), dict)
        else 0
    )
    branch_nodes = int(graph.get("branch_nodes", 0) or 0)
    cycles = int(graph.get("cycle_count", 0) or 0)
    connection_type_diversity = len(structural.get("connection_type_counts", {}) or {})
    degenerate = (
        space_count >= 6
        and shapes <= 1
        and roles <= 2
        and features == 0
        and branch_nodes == 0
        and cycles == 0
    )
    diversity_components = shapes + roles + connection_type_diversity + features + branch_nodes + cycles
    return {
        "space_count": space_count,
        "unique_shape_count": shapes,
        "role_diversity": roles,
        "connection_type_diversity": connection_type_diversity,
        "architectural_feature_count": features,
        "branch_nodes": branch_nodes,
        "cycle_count": cycles,
        "degenerate": degenerate,
        "diversity_score": round(min(1.0, float(diversity_components) / max(6.0, space_count)), 3),
    }


def _nearest_template_similarity(document: dict[str, Any], templates: list[dict[str, Any]]) -> dict[str, Any]:
    if not templates:
        return {"available": False, "nearest": None, "scores": []}
    scored: list[dict[str, Any]] = []
    for index, template in enumerate(templates):
        candidate = template.get("dsl") if isinstance(template.get("dsl"), dict) else template
        if not isinstance(candidate, dict):
            continue
        similarity = template_similarity_report(document, candidate)
        scored.append(
            {
                "template_id": template.get("id", index) if isinstance(template, dict) else index,
                **similarity,
            }
        )
    scored.sort(key=lambda item: float(item["template_similarity_score"]), reverse=True)
    return {
        "available": bool(scored),
        "nearest": scored[0] if scored else None,
        "scores": scored,
    }


def _failure_categories_for_schema(errors: Sequence[str]) -> list[str]:
    categories: set[str] = set()
    for error in errors:
        text = str(error).lower()
        if "missing top-level key" in text:
            categories.add("missing_required_field")
            continue
        if "missing" in text and ("field" in text or "key" in text):
            categories.add("missing_required_field")
        if "material" in text or "texture" in text or "flat" in text:
            categories.add("invalid_material")
        if "unsupported shape" in text or "unsupported orientation" in text or "unknown feature" in text:
            categories.add("unsupported_feature")
        if "unsupported dsl_version" in text:
            categories.add("unsupported_feature")
        if "progression" in text or "gate" in text or "reachable" in text or "key" in text:
            categories.add("incorrect_progression")
        if "connection" in text and "references unknown" in text:
            categories.add("invalid_geometry")
        if "self-connection" in text or "geometry" in text:
            categories.add("invalid_geometry")
    return sorted(categories or {"missing_required_field"})


def _failure_categories_for_result(result: dict[str, Any]) -> list[str]:
    categories = set(str(category) for category in result.get("failure_categories", []))
    if not result.get("exact_prompt_constraints_satisfied"):
        categories.add("unmet_prompt_requirement")
    if not result.get("correct_key_lock_behavior") or not result.get("valid_progression_and_connectivity"):
        categories.add("incorrect_progression")
    if result.get("json_parse_success") and not result.get("geometry_compilation_success"):
        categories.add("invalid_geometry")
    if result.get("geometry_compilation_success") and not result.get("wad_export_valid"):
        categories.add("invalid_geometry")
    if result.get("architectural_diversity", {}).get("degenerate"):
        categories.add("repetitive_or_degenerate_architecture")
    return sorted(categories)
