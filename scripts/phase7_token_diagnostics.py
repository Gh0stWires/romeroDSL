from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from romerodsl.qwen_lora import build_tokenization_diagnostics, load_sft_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure Phase 7 Gemma SFT token lengths before training.")
    parser.add_argument("--sft-jsonl", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_sft_records(args.sft_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    report = build_tokenization_diagnostics(records, tokenizer, max_length=args.max_length)
    report["sft_jsonl"] = str(args.sft_jsonl.resolve())
    report["sft_jsonl_sha256"] = hashlib.sha256(args.sft_jsonl.read_bytes()).hexdigest()
    report["tokenizer"] = args.model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": report["records"],
        "max_length": report["max_length"],
        "overlength_examples": report["overlength_examples"],
        "truncated_critical_json_completions": report["truncated_critical_json_completions"],
        "supervised_target_tokens_discarded": report["supervised_target_tokens_discarded"],
        "output": str(args.output.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
