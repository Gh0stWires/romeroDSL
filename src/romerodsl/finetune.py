"""Build and validate v0.7 supervised fine-tuning JSONL records.

The v0.7 target is deliberately strict: a model receives a natural-language Doom
map prompt and must answer with only a valid romeroDSL JSON object. The compiler
and validator remain the gate after generation.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from romerodsl.metrics import structural_metrics
from romerodsl.schema import validate_document

SFT_SCHEMA = "romerodsl_sft_v0.7"
SYSTEM_PROMPT = (
    "You generate valid romeroDSL JSON for Doom II maps. "
    "Return only one JSON object. Do not wrap it in markdown. "
    "The JSON must pass the romerodsl validator and must preserve requested keys, locks, "
    "progression, theme, combat, resources, materials, and height topology. "
    "Do not include source WAD provenance metadata or long source height arrays."
)


def canonical_sft_dsl(dsl: dict[str, Any]) -> dict[str, Any]:
    """Return the model target DSL without bulky extraction provenance.

    Training records keep source metadata outside the assistant message. The model
    should learn the authored design contract, not memorize long source-WAD audit
    arrays such as hundreds of raw height levels.
    """

    document = copy.deepcopy(dsl)
    document.pop("metadata", None)
    design_intent = document.get("design_intent")
    if isinstance(design_intent, dict):
        design_intent.pop("extraction_note", None)
    report = validate_document(document)
    report.raise_for_errors()
    return document


def sft_record_from_training_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one v0.6 training record into chat-style SFT JSONL format."""

    prompt = str(record.get("prompt") or "make a Doom map")
    dsl = record.get("dsl")
    if not isinstance(dsl, dict):
        raise ValueError("training record is missing a DSL object")
    report = validate_document(dsl)
    report.raise_for_errors()
    assistant_dsl = canonical_sft_dsl(dsl)
    labels = record.get("labels") if isinstance(record.get("labels"), dict) else {}
    features = record.get("features") if isinstance(record.get("features"), dict) else {}
    source_metadata: dict[str, Any] = (
        dict(dsl["metadata"]) if isinstance(dsl.get("metadata"), dict) else {}
    )
    generator_seed = record.get("generator_seed") or labels.get("generator_seed") or source_metadata.get("generator_seed")
    design_family = record.get("design_family") or labels.get("design_family") or source_metadata.get("design_family")
    assistant_content = json.dumps(assistant_dsl, indent=2, sort_keys=True, allow_nan=False)
    metrics = structural_metrics(assistant_dsl)
    return {
        "schema": SFT_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {
            "source_sample": record.get("source_sample"),
            "source_wad": record.get("source_wad"),
            "family": record.get("family"),
            "map": record.get("map"),
            "labels": labels,
            "features": features,
            "dataset_class": labels.get("dataset_class", "unknown"),
            "complexity_tier": labels.get("complexity_tier"),
            "generator_seed": generator_seed,
            "design_family": design_family,
            "prompt_length": len(prompt),
            "assistant_output_length": len(assistant_content),
            "structural_metrics": metrics,
        },
    }


def build_sft_jsonl(training_cache: Path, output: Path) -> dict[str, Any]:
    """Build a v0.7 SFT JSONL file from a v0.6 DSL training cache."""

    training_cache = Path(training_cache)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    theme_counter: Counter[str] = Counter()
    color_counter: Counter[str] = Counter()
    manifest_path = training_cache / "manifest.jsonl"

    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample_path = training_cache / str(row["path"])
        training_record = json.loads(sample_path.read_text(encoding="utf-8"))
        sft_record = sft_record_from_training_record(training_record)
        records.append(sft_record)
        labels = sft_record["metadata"].get("labels", {})
        if isinstance(labels, dict):
            theme = labels.get("theme")
            if isinstance(theme, str):
                theme_counter.update([theme])
            colors = labels.get("colors", [])
            if isinstance(colors, list):
                color_counter.update(str(color) for color in colors)

    output.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "schema": SFT_SCHEMA,
        "source_training_cache": str(training_cache),
        "output": str(output),
        "records": len(records),
        "theme_counts": dict(theme_counter),
        "color_counts": dict(color_counter),
    }
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def build_complexity_balanced_sft(
    output: Path,
    *,
    seed: int = 20261020,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a validated synthetic curriculum SFT file with provenance reports."""

    from romerodsl.extraction import (
        SYNTHETIC_TIERS,
        generate_synthetic_training_records,
        structural_layout_signature,
    )

    requested = {tier: 4 for tier in SYNTHETIC_TIERS} if counts is None else dict(counts)
    unknown = set(requested) - set(SYNTHETIC_TIERS)
    if unknown:
        raise ValueError(f"unsupported synthetic tiers: {sorted(unknown)}")
    if any(not isinstance(value, int) or value < 0 for value in requested.values()):
        raise ValueError("synthetic tier counts must be non-negative integers")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    signatures: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    for tier in SYNTHETIC_TIERS:
        for training_record in generate_synthetic_training_records(
            tier,
            count=int(requested.get(tier, 0)),
            seed=seed + SYNTHETIC_TIERS.index(tier) * 100003,
        ):
            signature = structural_layout_signature(training_record["dsl"])
            if signature in signatures:
                rejection_counts["duplicate_layout_signature"] += 1
                continue
            signatures.add(signature)
            records.append(sft_record_from_training_record(training_record))

    output.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in records), encoding="utf-8")
    validation = validate_sft_jsonl(output)
    room_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    branch_counts: Counter[str] = Counter()
    cycle_counts: Counter[str] = Counter()
    height_feature_counts: Counter[str] = Counter()
    progression_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter({"synthetic": 0, "extracted": 0, "hand-authored": 0})
    tier_counts: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    assistant_lengths: list[int] = []
    for record in records:
        metadata = record.get("metadata", {})
        labels = metadata.get("labels", {}) if isinstance(metadata, dict) else {}
        metrics = metadata.get("structural_metrics", {}) if isinstance(metadata, dict) else {}
        graph = metrics.get("graph", {}) if isinstance(metrics, dict) else {}
        height = metrics.get("height", {}) if isinstance(metrics, dict) else {}
        progression = metrics.get("progression", {}) if isinstance(metrics, dict) else {}
        provenance = str(metadata.get("dataset_class", "unknown")) if isinstance(metadata, dict) else "unknown"
        provenance_counts[provenance] += 1
        tier_counts[str(labels.get("complexity_tier", "unknown"))] += 1
        room_counts[str(metrics.get("space_count", 0))] += 1
        for shape in metrics.get("unique_room_shapes", []) if isinstance(metrics, dict) else []:
            shape_counts[str(shape)] += 1
        branch_counts[str(graph.get("branch_nodes", 0))] += 1
        cycle_counts[str(graph.get("cycle_count", 0))] += 1
        height_feature_counts[str(height.get("feature_count", 0))] += 1
        progression_counts["required_key_lock" if progression.get("required_gate_count", 0) else "reachable_exit"] += 1
        prompt_lengths.append(int(metadata.get("prompt_length", 0)))
        assistant_lengths.append(int(metadata.get("assistant_output_length", 0)))

    summary = {
        "schema": "romerodsl_phase4_complexity_balanced_sft_v1",
        "output": str(output),
        "seed": seed,
        "records": len(records),
        "requested_tier_counts": requested,
        "tier_counts": dict(tier_counts),
        "room_count_distribution": dict(room_counts),
        "shape_distribution": dict(shape_counts),
        "branch_count_distribution": dict(branch_counts),
        "cycle_count_distribution": dict(cycle_counts),
        "height_feature_distribution": dict(height_feature_counts),
        "progression_distribution": dict(progression_counts),
        "prompt_length_distribution": _length_distribution(prompt_lengths),
        "assistant_output_length_distribution": _length_distribution(assistant_lengths),
        "provenance_counts": dict(provenance_counts),
        "duplicate_rejections": dict(rejection_counts),
        "valid": validation["valid"],
        "validation": validation,
    }
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return summary


def _length_distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0, "values": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "values": sorted(values),
    }


def validate_sft_jsonl(path: Path) -> dict[str, Any]:
    """Validate that every SFT record contains parseable, valid romeroDSL output."""

    path = Path(path)
    records = 0
    invalid_json = 0
    invalid_dsl = 0
    invalid_shape = 0
    errors: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        records += 1
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            invalid_json += 1
            errors.append({"line": line_number, "kind": "record_json", "error": str(exc)})
            continue
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            invalid_shape += 1
            errors.append({"line": line_number, "kind": "shape", "error": "missing chat messages"})
            continue
        assistant = messages[-1]
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            invalid_shape += 1
            errors.append({"line": line_number, "kind": "shape", "error": "last message is not assistant"})
            continue
        content = assistant.get("content")
        if not isinstance(content, str):
            invalid_shape += 1
            errors.append({"line": line_number, "kind": "shape", "error": "assistant content is not text"})
            continue
        try:
            document = json.loads(content)
        except json.JSONDecodeError as exc:
            invalid_json += 1
            errors.append({"line": line_number, "kind": "assistant_json", "error": str(exc)})
            continue
        report = validate_document(document)
        if not report.valid:
            invalid_dsl += 1
            errors.append({"line": line_number, "kind": "dsl", "error": report.errors[:5]})
    return {
        "schema": SFT_SCHEMA,
        "path": str(path),
        "records": records,
        "invalid_json": invalid_json,
        "invalid_dsl": invalid_dsl,
        "invalid_shape": invalid_shape,
        "valid": invalid_json == 0 and invalid_dsl == 0 and invalid_shape == 0,
        "errors": errors[:20],
    }
