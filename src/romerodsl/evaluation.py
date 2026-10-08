"""Evaluation helpers for generated romeroDSL documents."""

from __future__ import annotations

import json
import re
from typing import Any


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

    space_overlap = _jaccard(generated_spaces, template_spaces)
    connection_overlap = _jaccard(generated_connections, template_connections)
    key_color_overlap = _jaccard(generated_key_colors, template_key_colors)
    copied_title = str(generated.get("title", "")).strip().lower() == str(template.get("title", "")).strip().lower()
    source_seed_title = "extracted" in str(generated.get("title", "")).lower() and "seed" in str(generated.get("title", "")).lower()
    score = (space_overlap + connection_overlap + key_color_overlap + (1.0 if copied_title else 0.0)) / 4.0
    if score >= 0.75 or source_seed_title:
        label = "high"
    elif score >= 0.4:
        label = "medium"
    else:
        label = "low"
    return {
        "template_similarity": label,
        "template_similarity_score": round(score, 3),
        "copied_source_title": copied_title or source_seed_title,
        "space_id_overlap": round(space_overlap, 3),
        "connection_id_overlap": round(connection_overlap, 3),
        "key_color_overlap": round(key_color_overlap, 3),
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


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
