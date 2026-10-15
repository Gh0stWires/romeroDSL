from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an explicit no-truncation Phase 7 training subset.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heldout-long", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.source.read_text(encoding="utf-8").splitlines() if line.strip()]
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    examples = diagnostics.get("examples", [])
    excluded = {
        int(example["index"])
        for example in examples
        if example.get("overlength") or example.get("critical_json_truncated")
    }
    if any(index < 0 or index >= len(records) for index in excluded):
        raise ValueError("token diagnostics contains an index outside the source dataset")
    kept = [record for index, record in enumerate(records) if index not in excluded]
    long_records = [record for index, record in enumerate(records) if index in excluded]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.heldout_long.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in kept), encoding="utf-8")
    args.heldout_long.write_text("".join(json.dumps(record, allow_nan=False) + "\n" for record in long_records), encoding="utf-8")
    report: dict[str, Any] = {
        "strategy": "explicit_staged_long_example_holdout",
        "source": str(args.source.resolve()),
        "source_sha256": sha256_file(args.source),
        "diagnostics": str(args.diagnostics.resolve()),
        "diagnostic_max_length": diagnostics.get("max_length"),
        "records_before": len(records),
        "records_after": len(kept),
        "excluded_long_records": len(long_records),
        "excluded_indices": sorted(excluded),
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "heldout_long": str(args.heldout_long.resolve()),
        "heldout_long_sha256": sha256_file(args.heldout_long),
        "policy": "Do not truncate complete JSON targets; train the 2500-token-safe subset and reserve long expert records for a separately approved context or staged-generation experiment.",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
