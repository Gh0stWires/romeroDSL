"""Command line interface for romeroDSL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.extraction import audit_graph_cache, build_dsl_training_cache
from romerodsl.finetune import build_sft_jsonl, validate_sft_jsonl
from romerodsl.qwen_lora import SFTTrainConfig, train_qwen_lora, training_run_plan
from romerodsl.schema import validate_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="romerodsl")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a romeroDSL JSON file")
    validate_parser.add_argument("path", type=Path)

    compile_parser = subparsers.add_parser("compile", help="emit an abstract geometry plan")
    compile_parser.add_argument("path", type=Path)
    compile_parser.add_argument("--output", "-o", type=Path, required=True)

    wad_parser = subparsers.add_parser("export-wad", help="compile a romeroDSL JSON file to a UDMF PWAD")
    wad_parser.add_argument("path", type=Path)
    wad_parser.add_argument("--output", "-o", type=Path, required=True)
    wad_parser.add_argument("--cell-size", type=int, default=128)

    training_parser = subparsers.add_parser(
        "build-training-pairs",
        help="extract prompt-to-romeroDSL training pairs from a graph cache",
    )
    training_parser.add_argument("--graph-cache", type=Path, required=True)
    training_parser.add_argument("--output", "-o", type=Path, required=True)
    training_parser.add_argument("--strict-only", action="store_true")
    training_parser.add_argument("--prompt-variants-per-sample", type=int, default=1)

    audit_parser = subparsers.add_parser(
        "audit-graph-cache",
        help="audit full graph-cache coverage and stratification buckets",
    )
    audit_parser.add_argument("--graph-cache", type=Path, required=True)
    audit_parser.add_argument("--output", "-o", type=Path)

    sft_parser = subparsers.add_parser(
        "build-sft-jsonl",
        help="convert v0.6 training pairs into v0.7 chat-style SFT JSONL",
    )
    sft_parser.add_argument("--training-cache", type=Path, required=True)
    sft_parser.add_argument("--output", "-o", type=Path, required=True)

    validate_sft_parser = subparsers.add_parser(
        "validate-sft-jsonl",
        help="validate v0.7 SFT JSONL assistant outputs as romeroDSL",
    )
    validate_sft_parser.add_argument("path", type=Path)

    train_parser = subparsers.add_parser(
        "train-qwen-lora",
        help="train a Qwen LoRA adapter on v0.7 prompt-to-romeroDSL SFT JSONL",
    )
    train_parser.add_argument("--sft-jsonl", type=Path, required=True)
    train_parser.add_argument("--output", "-o", type=Path, required=True)
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    train_parser.add_argument("--max-length", type=int, default=4096)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--epochs", type=float, default=1.0)
    train_parser.add_argument("--max-steps", type=int)
    train_parser.add_argument("--lora-rank", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--plan-only", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate_command(args.path)
    if args.command == "compile":
        return _compile_command(args.path, args.output)
    if args.command == "export-wad":
        return _export_wad_command(args.path, args.output, args.cell_size)
    if args.command == "build-training-pairs":
        return _build_training_pairs_command(
            args.graph_cache,
            args.output,
            args.strict_only,
            args.prompt_variants_per_sample,
        )
    if args.command == "audit-graph-cache":
        return _audit_graph_cache_command(args.graph_cache, args.output)
    if args.command == "build-sft-jsonl":
        return _build_sft_jsonl_command(args.training_cache, args.output)
    if args.command == "validate-sft-jsonl":
        return _validate_sft_jsonl_command(args.path)
    if args.command == "train-qwen-lora":
        return _train_qwen_lora_command(args)
    raise AssertionError(f"unhandled command: {args.command}")


def _validate_command(path: Path) -> int:
    document = _load_json(path)
    report = validate_document(document)
    for warning in report.warnings:
        print(f"warning: {warning}")
    if report.valid:
        print(f"valid: {path}")
        return 0
    for error in report.errors:
        print(f"error: {error}")
    return 1


def _compile_command(path: Path, output: Path) -> int:
    document = _load_json(path)
    plan = compile_to_plan(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {output}")
    return 0


def _export_wad_command(path: Path, output: Path, cell_size: int) -> int:
    document = _load_json(path)
    report = compile_to_wad(document, output, cell_size=cell_size)
    print(f"wrote: {output}")
    print(json.dumps(report, indent=2))
    return 0


def _build_training_pairs_command(
    graph_cache: Path,
    output: Path,
    strict_only: bool,
    prompt_variants_per_sample: int,
) -> int:
    summary = build_dsl_training_cache(
        graph_cache,
        output,
        strict_only=strict_only,
        prompt_variants_per_sample=prompt_variants_per_sample,
    )
    print(f"wrote: {output}")
    print(json.dumps(summary, indent=2))
    return 0


def _audit_graph_cache_command(graph_cache: Path, output: Path | None) -> int:
    summary = audit_graph_cache(graph_cache)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"wrote: {output}")
    print(json.dumps(summary, indent=2))
    return 0


def _build_sft_jsonl_command(training_cache: Path, output: Path) -> int:
    summary = build_sft_jsonl(training_cache, output)
    print(f"wrote: {output}")
    print(json.dumps(summary, indent=2))
    return 0


def _validate_sft_jsonl_command(path: Path) -> int:
    summary = validate_sft_jsonl(path)
    print(json.dumps(summary, indent=2))
    return 0 if summary["valid"] else 1


def _train_qwen_lora_command(args: argparse.Namespace) -> int:
    config = SFTTrainConfig(
        sft_jsonl=args.sft_jsonl,
        output_dir=args.output,
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    if args.plan_only:
        print(json.dumps(training_run_plan(config), indent=2))
        return 0
    metrics = train_qwen_lora(config)
    print(json.dumps({"adapter_dir": metrics.get("adapter_dir"), "steps": len(metrics.get("steps", []))}, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
