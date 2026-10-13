from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from romerodsl.extraction import generate_synthetic_training_records, structural_layout_signature
from romerodsl.finetune import sft_record_from_training_record
from romerodsl.metrics import structural_metrics
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build quality-gated Gemma-light SFT data.")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--synthetic-per-tier", type=int, default=0)
    parser.add_argument("--synthetic-seed", type=int, default=SEED)
    parser.add_argument(
        "--synthetic-count",
        action="append",
        default=[],
        metavar="TIER=COUNT",
        help="override one curriculum tier count; repeat for basic/intermediate/advanced/expert",
    )
    args = parser.parse_args(argv)
    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    records = load_records(source)
    selected, bucket_counts, rejection_counts = select_records_for_training(records, rng=rng)
    synthetic_records, synthetic_counts, synthetic_rejections = build_synthetic_records_for_sft(
        per_tier=args.synthetic_per_tier,
        overrides=args.synthetic_count,
        seed=args.synthetic_seed,
    )
    selected.extend(synthetic_records)
    rejection_counts.update(synthetic_rejections)

    golden_records = build_golden_records()
    selected.extend(golden_records)
    rng.shuffle(selected)

    validation = validate_records(selected)
    if not validation["valid"]:
        raise SystemExit(json.dumps(validation, indent=2))

    output.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in selected), encoding="utf-8")
    summary = {
        "source": str(source),
        "output": str(output),
        "records": len(selected),
        "bucket_counts": bucket_counts,
        "synthetic_tier_counts": synthetic_counts,
        "golden_records": len(golden_records),
        "rejection_counts": dict(rejection_counts),
        "provenance_counts": provenance_distribution(selected),
        "curriculum_distribution": curriculum_distribution(selected),
        "validation": validation,
    }
    SUMMARY_PATH = output.with_suffix(output.suffix + ".summary.json")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def build_synthetic_records_for_sft(
    *,
    per_tier: int,
    overrides: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    if per_tier < 0:
        raise ValueError("--synthetic-per-tier must be non-negative")
    counts = {tier: per_tier for tier in ("basic", "intermediate", "advanced", "expert")}
    for raw in overrides:
        if "=" not in raw:
            raise ValueError(f"invalid --synthetic-count {raw!r}; expected TIER=COUNT")
        tier, value = raw.split("=", 1)
        if tier not in counts:
            raise ValueError(f"invalid synthetic tier {tier!r}")
        counts[tier] = int(value)
        if counts[tier] < 0:
            raise ValueError("synthetic tier counts must be non-negative")

    selected: list[dict[str, Any]] = []
    signatures: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    for index, tier in enumerate(counts):
        training_records = generate_synthetic_training_records(
            tier,
            count=counts[tier],
            seed=seed + index * 100003,
        )
        for training_record in training_records:
            signature = structural_layout_signature(training_record["dsl"])
            if signature in signatures:
                rejection_counts["duplicate_layout_signature"] += 1
                continue
            candidate = normalize_record(sft_record_from_training_record(training_record))
            reasons = record_rejection_reasons(candidate)
            if reasons:
                rejection_counts.update(reasons)
                continue
            signatures.add(signature)
            selected.append(candidate)
    return selected, {tier: counts[tier] for tier in counts}, dict(rejection_counts)


def provenance_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"synthetic": 0, "extracted": 0, "hand-authored": 0, "unknown": 0}
    for record in records:
        metadata = record.get("metadata", {})
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
        provenance = labels.get("dataset_class") if isinstance(labels, dict) else None
        counts[str(provenance) if provenance in counts else "unknown"] += 1
    return counts


def curriculum_distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    distributions = {
        "room_counts": Counter(),
        "shape_types": Counter(),
        "branch_counts": Counter(),
        "cycle_counts": Counter(),
        "height_features": Counter(),
        "progression_types": Counter(),
        "prompt_lengths": Counter(),
        "assistant_output_lengths": Counter(),
    }
    for record in records:
        metadata = record.get("metadata", {})
        metrics = metadata.get("structural_metrics", {}) if isinstance(metadata, dict) else {}
        graph = metrics.get("graph", {}) if isinstance(metrics, dict) else {}
        height = metrics.get("height", {}) if isinstance(metrics, dict) else {}
        progression = metrics.get("progression", {}) if isinstance(metrics, dict) else {}
        distributions["room_counts"][str(metrics.get("space_count", 0))] += 1
        for shape in metrics.get("unique_room_shapes", []) if isinstance(metrics, dict) else []:
            distributions["shape_types"][str(shape)] += 1
        distributions["branch_counts"][str(graph.get("branch_nodes", 0))] += 1
        distributions["cycle_counts"][str(graph.get("cycle_count", 0))] += 1
        distributions["height_features"][str(height.get("feature_count", 0))] += 1
        progression_type = (
            metadata.get("features", {}).get("progression_type")
            if isinstance(metadata.get("features"), dict)
            else None
        ) or ("required_key_lock" if progression.get("required_gate_count", 0) else ("optional_key_lock" if progression.get("optional_gate_count", 0) else "reachable_exit"))
        distributions["progression_types"][str(progression_type)] += 1
        distributions["prompt_lengths"][str(metadata.get("prompt_length", 0))] += 1
        distributions["assistant_output_lengths"][str(metadata.get("assistant_output_length", 0))] += 1
    return {name: dict(counter) for name, counter in distributions.items()}


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
    metadata = normalized.get("metadata")
    if isinstance(metadata, dict):
        document = json.loads(messages[-1]["content"])
        labels = metadata.get("labels")
        if not isinstance(labels, dict):
            labels = {}
            metadata["labels"] = labels
        dataset_class = labels.get("dataset_class") or metadata.get("dataset_class")
        if dataset_class not in {"synthetic", "extracted", "hand-authored"}:
            if metadata.get("source_wad") or metadata.get("source_sample"):
                dataset_class = "extracted"
            elif metadata.get("hand_authored") is True or metadata.get("source_kind") == "hand-authored":
                dataset_class = "hand-authored"
            else:
                dataset_class = "unknown"
        labels["dataset_class"] = dataset_class
        metadata["dataset_class"] = dataset_class
        metadata["structural_metrics"] = structural_metrics(document)
        prompt_message = messages[1] if len(messages) > 1 and isinstance(messages[1], dict) else {}
        prompt_text = prompt_message.get("content") if isinstance(prompt_message, dict) else ""
        assistant_text = messages[-1].get("content") if isinstance(messages[-1], dict) else ""
        metadata["prompt_length"] = len(prompt_text) if isinstance(prompt_text, str) else 0
        metadata["assistant_output_length"] = len(assistant_text) if isinstance(assistant_text, str) else 0
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
