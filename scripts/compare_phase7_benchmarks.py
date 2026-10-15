from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

BOOLEAN_FIELDS = (
    "json_parse_success",
    "dsl_schema_valid",
    "exact_prompt_constraints_satisfied",
    "correct_room_count",
    "correct_key_lock_behavior",
    "valid_progression_and_connectivity",
    "geometry_compilation_success",
    "wad_export_valid",
    "first_pass_success",
)


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("benchmark_version") != "phase6_no_repair_v1":
        raise ValueError(f"{path} is not a Phase 6 strict no-repair report")
    results = report.get("cases")
    if not isinstance(results, list) or len(results) != 32:
        raise ValueError(f"{path} does not contain the complete 32-case benchmark")
    return report


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    results = report["cases"]
    tiers: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result.get("complexity_tier"))].append(result)
    for tier, values in sorted(grouped.items()):
        tiers[tier] = summarize_results(values)
    return {"overall": summarize_results(results), "tiers": tiers}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    available_diversity = [
        result["architectural_diversity"].get("diversity_score")
        for result in results
        if isinstance(result.get("architectural_diversity"), dict)
        and isinstance(result["architectural_diversity"].get("diversity_score"), (int, float))
    ]
    template_scores = [
        result["template_similarity"]["nearest"].get("template_similarity_score")
        for result in results
        if isinstance(result.get("template_similarity"), dict)
        and isinstance(result["template_similarity"].get("nearest"), dict)
        and isinstance(result["template_similarity"]["nearest"].get("template_similarity_score"), (int, float))
    ]
    room_counts = [
        result["architectural_diversity"].get("space_count")
        for result in results
        if isinstance(result.get("architectural_diversity"), dict)
        and isinstance(result["architectural_diversity"].get("space_count"), int)
    ]
    return {
        "cases": len(results),
        "pass_counts": {field: sum(bool(result.get(field)) for result in results) for field in BOOLEAN_FIELDS},
        "failure_categories": dict(Counter(category for result in results for category in result.get("failure_categories", []))),
        "first_pass_success_rate": sum(bool(result.get("first_pass_success")) for result in results) / max(1, len(results)),
        "mean_diversity_score": mean(available_diversity) if available_diversity else None,
        "mean_nearest_template_similarity": mean(template_scores) if template_scores else None,
        "mean_generated_space_count": mean(room_counts) if room_counts else None,
        "architecture_metrics_available": len(available_diversity),
        "template_metrics_available": len(template_scores),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two complete strict Phase 6 benchmark reports.")
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior = load_report(args.prior)
    new = load_report(args.new)
    comparison = {
        "benchmark_version": "phase6_no_repair_v1",
        "prior_report": str(args.prior.resolve()),
        "new_report": str(args.new.resolve()),
        "prior_run_metadata": prior.get("run_metadata", {}),
        "new_run_metadata": new.get("run_metadata", {}),
        "prior": summarize(prior),
        "new": summarize(new),
    }
    comparison["delta_overall"] = {
        field: comparison["new"]["overall"]["pass_counts"][field]
        - comparison["prior"]["overall"]["pass_counts"][field]
        for field in BOOLEAN_FIELDS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(comparison["delta_overall"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
