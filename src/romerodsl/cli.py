"""Command line interface for romeroDSL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.extraction import build_dsl_training_cache
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

    args = parser.parse_args(argv)

    if args.command == "validate":
        return _validate_command(args.path)
    if args.command == "compile":
        return _compile_command(args.path, args.output)
    if args.command == "export-wad":
        return _export_wad_command(args.path, args.output, args.cell_size)
    if args.command == "build-training-pairs":
        return _build_training_pairs_command(args.graph_cache, args.output, args.strict_only)
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


def _build_training_pairs_command(graph_cache: Path, output: Path, strict_only: bool) -> int:
    summary = build_dsl_training_cache(graph_cache, output, strict_only=strict_only)
    print(f"wrote: {output}")
    print(json.dumps(summary, indent=2))
    return 0


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
