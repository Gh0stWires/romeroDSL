from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from romerodsl.evaluation import PHASE6_BENCHMARK_CASES, run_no_repair_benchmark
from romerodsl.finetune import SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_ADAPTER = ROOT / "runs/gemma4_e4b_v10_light_clean/adapter"
DEFAULT_OUTPUT = ROOT / "build/phase7_baseline_gemma_v10"
DEFAULT_TEMPLATES = ROOT / "build/v11_gemma_sft/romerodsl_sft_gemma_light_real_wad_only.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_training_templates(path: Path, limit: int) -> list[dict[str, Any]]:
    templates: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        assistant = messages[-1] if isinstance(messages[-1], dict) else {}
        content = assistant.get("content")
        if not isinstance(content, str):
            continue
        try:
            document = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            templates.append(document)
        if len(templates) >= limit:
            break
    if not templates:
        raise ValueError(f"no JSON training templates found in {path}")
    return templates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the strict Phase 6 benchmark against a Gemma adapter.")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--template-limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=2500)
    parser.add_argument("--case-id", action="append", default=[], help="run only this Phase 6 case; repeat for resume runs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    adapter = args.adapter.resolve()
    output = args.output.resolve()
    templates_path = args.templates.resolve()
    if not adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter}")
    if not templates_path.is_file():
        raise FileNotFoundError(f"template dataset does not exist: {templates_path}")
    if args.template_limit <= 0:
        raise ValueError("--template-limit must be positive")

    print(f"loading tokenizer from {adapter}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("loading Gemma 4 E4B in 4-bit mode", flush=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    templates = load_training_templates(templates_path, args.template_limit)

    generation_stats: dict[str, Any] = {
        "cases": 0,
        "elapsed_seconds": 0.0,
        "max_cuda_allocated_bytes": 0,
        "max_cuda_reserved_bytes": 0,
    }

    def generate(prompt: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(next(model.parameters()).device)
        started = time.perf_counter()
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generation_stats["cases"] += 1
        generation_stats["elapsed_seconds"] += time.perf_counter() - started
        if torch.cuda.is_available():
            generation_stats["max_cuda_allocated_bytes"] = max(
                generation_stats["max_cuda_allocated_bytes"], torch.cuda.max_memory_allocated()
            )
            generation_stats["max_cuda_reserved_bytes"] = max(
                generation_stats["max_cuda_reserved_bytes"], torch.cuda.max_memory_reserved()
            )
        return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()

    selected_cases = None
    if args.case_id:
        wanted = set(args.case_id)
        known = {str(case["id"]) for case in PHASE6_BENCHMARK_CASES}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"unknown Phase 6 case ids: {unknown}")
        selected_cases = [case for case in PHASE6_BENCHMARK_CASES if str(case["id"]) in wanted]

    report = run_no_repair_benchmark(
        generate,
        output,
        cases=selected_cases,
        training_templates=templates,
        overwrite=args.overwrite,
    )
    report["run_metadata"] = {
        "evaluation_mode": "strict_no_repair",
        "base_model": MODEL_ID,
        "adapter": str(adapter),
        "adapter_sha256": sha256_file(adapter / "adapter_model.safetensors"),
        "template_dataset": str(templates_path),
        "template_dataset_sha256": sha256_file(templates_path),
        "template_count": len(templates),
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "system_prompt": SYSTEM_PROMPT,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "generation_memory": generation_stats,
    }
    (output / "benchmark_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"aggregate": report.get("aggregate"), "run_metadata": report["run_metadata"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
