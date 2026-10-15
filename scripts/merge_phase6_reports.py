from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from romerodsl.evaluation import PHASE6_BENCHMARK_CASES, PHASE6_BENCHMARK_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge resumed per-case Phase 6 reports into one complete report.")
    parser.add_argument("--root", action="append", required=True, help="case-output root; repeat for partial/resume roots")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = [Path(root).resolve() for root in args.root]
    results: list[dict[str, Any]] = []
    missing: list[str] = []
    for case in PHASE6_BENCHMARK_CASES:
        case_id = str(case["id"])
        found = None
        for root in roots:
            candidate = root / case_id / "result.json"
            if candidate.is_file():
                found = candidate
                break
        if found is None:
            missing.append(case_id)
            continue
        result = json.loads(found.read_text(encoding="utf-8"))
        if result.get("case_id") != case_id:
            raise ValueError(f"case id mismatch in {found}")
        results.append(result)
    if missing:
        raise ValueError(f"incomplete Phase 6 case set, missing: {missing}")
    failure_counts = Counter(category for result in results for category in result.get("failure_categories", []))
    tier_summary: dict[str, dict[str, int]] = {}
    for result in results:
        tier = str(result.get("complexity_tier", "unknown"))
        summary = tier_summary.setdefault(tier, {"total": 0, "first_pass_success": 0})
        summary["total"] += 1
        summary["first_pass_success"] += int(bool(result.get("first_pass_success")))
    metadata: dict[str, Any] = {
        "evaluation_mode": "strict_no_repair",
        "benchmark_version": PHASE6_BENCHMARK_VERSION,
        "source_roots": [str(root) for root in roots],
        "resumed": len(roots) > 1,
    }
    for root in roots:
        candidate = root / "benchmark_report.json"
        if candidate.is_file():
            partial = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(partial.get("run_metadata"), dict):
                metadata.update(partial["run_metadata"])
    report = {
        "benchmark_version": PHASE6_BENCHMARK_VERSION,
        "evaluation_mode": "strict_no_repair",
        "repair_applied": False,
        "iterative_correction_evaluated": False,
        "output_dir": str(args.output.resolve().parent),
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
        "run_metadata": metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(results), "counts": report["counts"], "tier_summary": tier_summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
