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
    return {
        "schema": SFT_SCHEMA,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": json.dumps(assistant_dsl, indent=2, sort_keys=True, allow_nan=False)},
        ],
        "metadata": {
            "source_sample": record.get("source_sample"),
            "source_wad": record.get("source_wad"),
            "family": record.get("family"),
            "map": record.get("map"),
            "labels": labels,
            "features": features,
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
