from __future__ import annotations

import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from romerodsl.schema import validate_document

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "build/v08_sft/romerodsl_sft_v08_full.jsonl"
GOLDEN = ROOT / "examples/complex_eight_room_bfg.json"
OUT = ROOT / "build/v11_gemma_sft/romerodsl_sft_gemma_light_real_wad_only.jsonl"
SUMMARY = OUT.with_suffix(OUT.suffix + ".summary.json")
SEED = 20261010
SYSTEM_PROMPT = (
    "You generate valid romeroDSL JSON for Doom II maps. "
    "Return only one JSON object. Do not wrap it in markdown. "
    "Use only valid Doom II material names that appear in training examples. "
    "BFG is an item reward named bfg9000, never a key or gate type. "
    "Required locked gates must be real route gates; do not create bypass connections around them."
)
BUCKET_QUOTAS = {
    "required_multi_lock": 1200,
    "required_single_lock": 900,
    "optional_key_lock": 500,
    "layout_style_only": 450,
    "basic_route": 450,
}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    records = load_records(SOURCE)
    selected, bucket_counts, rejection_counts = select_records_for_training(records, rng=rng)

    golden_records = build_golden_records()
    selected.extend(golden_records)
    rng.shuffle(selected)

    validation = validate_records(selected)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2))

    OUT.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in selected), encoding="utf-8")
    summary = {
        "source": str(SOURCE),
        "output": str(OUT),
        "records": len(selected),
        "bucket_counts": bucket_counts,
        "golden_records": len(golden_records),
        "rejection_counts": rejection_counts,
        "validation": validation,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def load_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_records_for_training(
    records: list[dict[str, Any]],
    *,
    rng: random.Random,
    quotas: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    """Normalize, quality-filter, then quota-sample training records by bucket."""

    quotas = dict(BUCKET_QUOTAS if quotas is None else quotas)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejection_counts: Counter[str] = Counter()

    for record in records:
        normalized = normalize_record(record)
        rejection_reasons = record_rejection_reasons(normalized)
        if rejection_reasons:
            rejection_counts.update(rejection_reasons)
            continue
        bucket = str(normalized.get("metadata", {}).get("labels", {}).get("bucket", "unknown"))
        by_bucket[bucket].append(normalized)

    selected: list[dict[str, Any]] = []
    bucket_counts: dict[str, int] = {}
    for bucket, quota in quotas.items():
        values = list(by_bucket.get(bucket, []))
        rng.shuffle(values)
        take = values[:quota]
        selected.extend(take)
        bucket_counts[bucket] = len(take)

    return selected, bucket_counts, dict(rejection_counts)


def record_rejection_reasons(record: dict[str, Any]) -> list[str]:
    """Return stable quality-gate reasons that make a record unsafe for SFT."""

    reasons: list[str] = []
    document = _assistant_document(record)
    if document is None:
        return ["invalid_json"]

    if "spaces" not in document:
        reasons.append("missing_top_level_spaces")

    connections = document.get("connections", [])
    if isinstance(connections, list) and any(
        isinstance(connection, dict)
        and isinstance(connection.get("from"), str)
        and connection.get("from") == connection.get("to")
        for connection in connections
    ):
        reasons.append("self_connection")

    report = validate_document(document)
    for error in report.errors:
        if "invalid" in error and " material " in error:
            reasons.append("invalid_material")
        if error == "missing top-level key: spaces":
            reasons.append("missing_top_level_spaces")
        if "exit is reachable without gate" in error:
            reasons.append("route_bypass")
        if "self-connection" in error:
            reasons.append("self_connection")

    if not report.valid and not reasons:
        reasons.append("invalid_dsl")

    return _unique_preserving_order(reasons)


def _assistant_document(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3 or not isinstance(messages[-1], dict):
        return None
    assistant = messages[-1].get("content")
    if not isinstance(assistant, str):
        return None
    try:
        document = json.loads(assistant)
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Return a Gemma-friendly compact training record with strict system text."""

    normalized = copy.deepcopy(record)
    messages = normalized.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return normalized
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        messages[0]["content"] = SYSTEM_PROMPT
    assistant = messages[-1]
    if isinstance(assistant, dict) and isinstance(assistant.get("content"), str):
        document = json.loads(assistant["content"])
        assistant["content"] = json.dumps(
            document, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    normalized["schema"] = "romerodsl_sft_v0.11_gemma_light_real_wad_only"
    return normalized


def build_golden_records() -> list[dict[str, Any]]:
    """Return hand-authored records for experiments that explicitly allow synthetic golden data.

    Sam wants Gemma trained only from human-made WAD-derived levels right now, so
    the default light SFT builder must not inject hand-authored examples.
    """

    return []


def make_yellow_refinery_variant(source: dict[str, Any]) -> dict[str, Any]:
    doc = copy.deepcopy(source)
    doc["title"] = "Corrupted Refinery Yellow Red BFG Route"
    doc["design_intent"]["summary"] = (
        "A big eight-room corrupted refinery with separate combat areas, yellow and red keycard rooms, "
        "a yellow-locked BFG armory on the route to the red key, and a red-locked exit wing."
    )
    doc["design_intent"]["progression_style"] = "yellow_key_then_bfg_armory_then_red_key_then_red_locked_exit"
    doc["theme"]["primary"] = "techbase"
    doc["theme"]["secondary"] = "industrial"
    doc["theme"]["mood"] = "corrupted_refinery"

    replacements = {
        "blue_key_room": "yellow_key_room",
        "blue_locked_route": "yellow_locked_route",
        "blue": "yellow",
    }
    replace_strings(doc, replacements)

    for space in doc.get("spaces", []):
        if isinstance(space, dict) and space.get("id") == "yellow_key_room":
            materials = space.setdefault("materials", {})
            materials["floor"] = "FLOOR0_2"
            materials["wall"] = "STARTAN2"
    return doc


def make_red_yellow_bfg_variant(source: dict[str, Any]) -> dict[str, Any]:
    """Return an eight-room route where red gates BFG and yellow gates exit."""

    doc = copy.deepcopy(source)
    replace_strings_atomic(doc, {"blue": "red", "red": "yellow"})
    doc["title"] = "Eight Room Red Yellow BFG Refinery"
    doc["design_intent"]["summary"] = (
        "A big eight-room refinery where a red key opens the BFG armory, the route continues to "
        "a yellow key room, and a yellow locked door gates the final exit wing."
    )
    doc["design_intent"]["progression_style"] = "red_key_then_bfg_armory_then_yellow_key_then_yellow_locked_exit"
    doc["theme"]["mood"] = "hellish_refinery"
    return doc


def make_blue_red_combat_loop_variant(source: dict[str, Any]) -> dict[str, Any]:
    """Return a distinct blue/red eight-room variant with combat-loop wording."""

    doc = copy.deepcopy(source)
    doc["title"] = "Eight Room Blue Red Combat Loop"
    doc["design_intent"]["summary"] = (
        "A big eight-room techbase where the blue key gates a dangerous combat loop and BFG reward "
        "before the red key, then the red locked door gates the final exit."
    )
    doc["design_intent"]["topology"] = "eight_room_two_key_combat_loop_with_bfg_reward"
    doc["design_intent"]["progression_style"] = "blue_key_then_combat_loop_then_red_key_then_red_locked_exit"
    for space in doc.get("spaces", []):
        if isinstance(space, dict) and space.get("id") == "bfg_armory":
            space["role"] = "weapon_combat_room"
            space["shape"] = "irregular_octagonal"
            space.setdefault("encounter", {})["intensity"] = "heavy"
    return doc


def make_no_bfg_variant(source: dict[str, Any], *, title: str) -> dict[str, Any]:
    """Return an eight-room two-key route with arsenal rewards but no BFG token."""

    doc = copy.deepcopy(source)
    replace_strings(doc, {"bfg_to_red_key": "arsenal_to_second_key", "bfg_armory": "arsenal_room"})
    replace_strings(doc, {"BFG9000": "chaingun", "BFG": "arsenal", "bfg9000": "chaingun", "bfg": "arsenal"})
    doc["title"] = title
    doc["design_intent"]["summary"] = (
        "A big eight-room two-key route with an arsenal reward room between the first locked door "
        "and the second key, using ammo and weapon rewards instead of a superweapon reward."
    )
    doc["design_intent"]["reward_style"] = "arsenal_room_behind_existing_key_lock_with_standard_weapon_reward"
    doc["design_intent"]["progression_style"] = (
        str(doc["design_intent"].get("progression_style", "two_key_route")).replace("bfg_armory", "arsenal_room")
    )
    for space in doc.get("spaces", []):
        if isinstance(space, dict) and space.get("id") == "arsenal_room":
            space["role"] = "weapon_room"
            contains = space.setdefault("contains", {})
            contains["items"] = ["chaingun", "cell_pack", "medkit"]
    return doc


def replace_strings(value: Any, replacements: dict[str, str]) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if isinstance(child, str):
                for old, new in replacements.items():
                    child = child.replace(old, new)
                value[key] = child
            else:
                replace_strings(child, replacements)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                for old, new in replacements.items():
                    child = child.replace(old, new)
                value[index] = child
            else:
                replace_strings(child, replacements)


def replace_strings_atomic(value: Any, replacements: dict[str, str]) -> None:
    """Replace strings without cascading when replacement text contains another source token."""

    sentinels = {old: f"__ROMERODSL_REPLACE_{index}__" for index, old in enumerate(replacements)}

    def replace_text(text: str) -> str:
        for old, sentinel in sentinels.items():
            text = text.replace(old, sentinel)
        for old, sentinel in sentinels.items():
            text = text.replace(sentinel, replacements[old])
        return text

    if isinstance(value, dict):
        for key, child in list(value.items()):
            if isinstance(child, str):
                value[key] = replace_text(child)
            else:
                replace_strings_atomic(child, replacements)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, str):
                value[index] = replace_text(child)
            else:
                replace_strings_atomic(child, replacements)


def validate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    invalid_json = 0
    invalid_dsl = 0
    invalid_shape = 0
    errors = []
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            invalid_shape += 1
            errors.append({"index": index, "error": "missing messages"})
            continue
        assistant = messages[-1].get("content") if isinstance(messages[-1], dict) else None
        if not isinstance(assistant, str):
            invalid_shape += 1
            errors.append({"index": index, "error": "assistant content is not text"})
            continue
        try:
            document = json.loads(assistant)
        except json.JSONDecodeError as exc:
            invalid_json += 1
            errors.append({"index": index, "error": str(exc)})
            continue
        report = validate_document(document)
        if not report.valid:
            invalid_dsl += 1
            errors.append({"index": index, "error": report.errors[:10]})
    return {
        "valid": invalid_json == 0 and invalid_dsl == 0 and invalid_shape == 0,
        "invalid_json": invalid_json,
        "invalid_dsl": invalid_dsl,
        "invalid_shape": invalid_shape,
        "errors": errors[:20],
    }


if __name__ == "__main__":
    raise SystemExit(main())
