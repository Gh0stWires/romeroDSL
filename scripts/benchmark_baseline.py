"""Build a reproducible RomeroDSL v0.11 structural baseline report.

The script evaluates hand-authored examples without repair and, when the local
Gemma runtime is available, runs one loaded adapter against a fixed 20-prompt
suite.  Existing datasets, adapters, and benchmark directories are never
modified unless ``--overwrite`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.evaluation import extract_json_object, template_similarity_report
from romerodsl.metrics import structural_metrics
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "build" / "benchmarks" / "v0.11_baseline_run"
DEFAULT_ADAPTER = ROOT / "runs" / "gemma4_e4b_v10_light_clean" / "adapter"
MODEL_ID = "google/gemma-4-E4B-it"

PROMPTS: list[dict[str, str]] = [
    {"id": "basic_three_room", "tier": "basic", "prompt": "Make a three-room Doom map with a start, a combat room, and a reachable exit."},
    {"id": "basic_no_lock", "tier": "basic", "prompt": "Create a four-room techbase route with no locked doors, light monsters, ammo, and health."},
    {"id": "basic_blue_lock", "tier": "basic", "prompt": "Make a five-room map where a blue key is reachable before a blue locked door gates the exit."},
    {"id": "basic_height", "tier": "basic", "prompt": "Build a compact three-room map with a raised platform, stairs, monsters, and a reachable exit."},
    {"id": "intermediate_eight_techbase", "tier": "intermediate", "prompt": "Create an eight-room interconnected techbase with a hub, two branches, varied room shapes, and a valid exit."},
    {"id": "intermediate_two_keys", "tier": "intermediate", "prompt": "Make an eight-room Doom map with blue then red keys, matching required locks, and no bypass around either gate."},
    {"id": "intermediate_loop", "tier": "intermediate", "prompt": "Build a seven-room base with a branch from the hub and a loop that reconnects before the final exit."},
    {"id": "intermediate_secret", "tier": "intermediate", "prompt": "Create an eight-space industrial map with an optional secret armor route, a yellow key, and a yellow locked exit."},
    {"id": "intermediate_bfg", "tier": "intermediate", "prompt": "Make an eight-room techbase where a blue key opens a BFG9000 reward room and a red key gates the final exit."},
    {"id": "intermediate_vertical", "tier": "intermediate", "prompt": "Create a nine-room facility with stairs, a shallow pit, a raised platform, and a red locked route to the exit."},
    {"id": "advanced_twelve_nonlinear", "tier": "advanced", "prompt": "Build a twelve-room nonlinear Doom facility with two hubs, multiple branches, one loop, combat arenas, and a reachable exit."},
    {"id": "advanced_three_keys", "tier": "advanced", "prompt": "Make a twelve-space map with blue, red, and yellow keys, sequential matching locked gates, and no route bypasses."},
    {"id": "advanced_optional_routes", "tier": "advanced", "prompt": "Create a ten-room exploration map with optional side routes, one secret, varied silhouettes, height changes, and required red progression."},
    {"id": "advanced_irregular", "tier": "advanced", "prompt": "Design an eleven-room industrial map using L-shaped, T-shaped, alcove, and irregular orthogonal chambers with a locked exit."},
    {"id": "advanced_reward", "tier": "advanced", "prompt": "Build a fourteen-room techbase with a blue-gated armory, a red-gated final arena, monsters, ammo, health, and a valid exit."},
    {"id": "expert_sixteen_hell", "tier": "expert", "prompt": "Create a sixteen-room nonlinear Hell complex with interconnected hubs, loops, distinct silhouettes, elevation changes, and locked progression."},
    {"id": "expert_exploration", "tier": "expert", "prompt": "Make an eighteen-space Doom level focused on exploration, optional secrets, multiple branches, two reconnections, and a final combat arena."},
    {"id": "expert_multi_lock", "tier": "expert", "prompt": "Build a twenty-room map with blue, red, and yellow required key locks, hub reconnections, height features, and no bypass paths."},
    {"id": "expert_techbase", "tier": "expert", "prompt": "Design a twenty-four-room UAC techbase with multiple architectural zones, alcoves, L-shaped chambers, platforms, pits, resources, and a gated exit."},
    {"id": "expert_materials", "tier": "expert", "prompt": "Create an eighteen-room Hell facility with distinct stone materials by zone, meaningful elevation, optional branches, combat distribution, and locked escape."},
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--max-new-tokens", type=int, default=3500)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite non-empty baseline directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema": "romerodsl_baseline_report_v0.11",
        "created_by": "scripts/benchmark_baseline.py",
        "repository": str(ROOT),
        "git": _git_state(),
        "generation_parameters": {
            "model": MODEL_ID,
            "adapter": str(args.adapter.resolve()),
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": None,
            "top_k": None,
            "repair": False,
        },
        "engine_environment": _engine_environment(),
        "examples": _evaluate_examples(output / "examples"),
        "model_suite": {
            "suite_size": len(PROMPTS),
            "prompts": PROMPTS,
        },
    }

    if args.skip_model:
        report["model_suite"].update({"status": "skipped", "reason": "--skip-model"})
    else:
        report["model_suite"].update(
            _evaluate_model_suite(output / "model_suite", args.adapter.resolve(), args.max_new_tokens)
        )

    engine_smoke_results = [
        example.get("engine_smoke", {}).get("passed", False)
        for example in report["examples"]
        if "engine_smoke" in example
    ]
    report["acceptance"] = {
        "tests_run_without_retraining": True,
        "example_exports_recorded": bool(report["examples"]),
        "fixed_prompt_suite_recorded": len(PROMPTS) >= 20,
        "adapter_status": report["model_suite"].get("status"),
        "engine_available": report["engine_environment"]["available"],
        "engine_smoke_test": bool(engine_smoke_results) and all(engine_smoke_results),
    }
    report_path = output / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote: {report_path}")
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0


def _evaluate_examples(output: Path) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for source in sorted((ROOT / "examples").glob("*.json")):
        case_dir = output / source.stem
        case_dir.mkdir(parents=True, exist_ok=True)
        result: dict[str, Any] = {"source": str(source), "id": source.stem}
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
            validation = validate_document(document)
            result["dsl_validation"] = {
                "valid": validation.valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
            if validation.valid:
                plan = compile_to_plan(document)
                plan_path = case_dir / "map.plan.json"
                plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                wad_path = case_dir / "map.wad"
                compile_report = compile_to_wad(document, wad_path)
                wad_report = validate_wad(wad_path)
                metrics = structural_metrics(
                    document,
                    plan=plan,
                    wad_report=wad_report,
                    dsl_valid=validation.valid,
                )
                result.update(
                    {
                        "plan_path": str(plan_path),
                        "wad_path": str(wad_path),
                        "compile_report": compile_report,
                        "wad_report": wad_report,
                        "structural_metrics": metrics,
                    }
                )
                result["engine_smoke"] = _run_engine_smoke(wad_path, case_dir / "zandronum.log")
        except Exception as exc:  # noqa: BLE001
            result["error"] = f"{type(exc).__name__}: {exc}"
        (case_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(result)
    return results


def _evaluate_model_suite(output: Path, adapter: Path, max_new_tokens: int) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    if not adapter.exists():
        return {"status": "unavailable", "reason": f"adapter does not exist: {adapter}"}
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"runtime import failed: {type(exc).__name__}: {exc}"}

    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
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
    except Exception as exc:  # noqa: BLE001
        return {"status": "unavailable", "reason": f"model load failed: {type(exc).__name__}: {exc}"}

    results = []
    for case in PROMPTS:
        case_dir = output / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        result: dict[str, Any] = {**case, "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False}}
        try:
            messages = [
                {"role": "system", "content": "Generate only one valid romeroDSL JSON object for Doom II. Do not repair or explain the output."},
                {"role": "user", "content": case["prompt"]},
            ]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            (case_dir / "prompt.txt").write_text(rendered, encoding="utf-8")
            inputs = tokenizer(rendered, return_tensors="pt")
            input_tokens = int(inputs["input_ids"].shape[1])
            inputs = inputs.to(next(model.parameters()).device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0, input_tokens:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            (case_dir / "raw.txt").write_text(raw, encoding="utf-8")
            result["input_token_count"] = input_tokens
            result["output_token_count"] = int(new_tokens.shape[0])
            result["generation_truncated"] = int(new_tokens.shape[0]) >= max_new_tokens
            try:
                document = extract_json_object(raw)
                result["json_valid"] = True
            except Exception as exc:  # noqa: BLE001
                document = {}
                result.update({"json_valid": False, "failure_category": "syntax_failure", "error": str(exc)})
            validation = validate_document(document)
            result["dsl_validation"] = {
                "valid": validation.valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
            if result.get("json_valid") and not validation.valid:
                result["failure_category"] = _validation_failure_category(validation.errors)
            metrics = structural_metrics(
                document,
                prompt=case["prompt"],
                dsl_valid=validation.valid,
                raw_output=raw,
                output_token_count=result["output_token_count"],
                max_new_tokens=max_new_tokens,
                generation_truncated=result["generation_truncated"],
            )
            result["structural_metrics"] = metrics
            if validation.valid:
                plan = compile_to_plan(document)
                (case_dir / "map.plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                wad_path = case_dir / "map.wad"
                compile_report = compile_to_wad(document, wad_path)
                wad_report = validate_wad(wad_path)
                result.update(
                    {
                        "wad_path": str(wad_path),
                        "compile_report": compile_report,
                        "wad_report": wad_report,
                        "template_similarity": template_similarity_report(
                            document,
                            json.loads((ROOT / "examples" / "complex_eight_room_bfg.json").read_text(encoding="utf-8")),
                        ),
                    }
                )
                result["structural_metrics"] = structural_metrics(
                    document,
                    prompt=case["prompt"],
                    plan=plan,
                    wad_report=wad_report,
                    dsl_valid=validation.valid,
                    raw_output=raw,
                    output_token_count=result["output_token_count"],
                    max_new_tokens=max_new_tokens,
                    generation_truncated=result["generation_truncated"],
                )
                if not wad_report.get("valid", False):
                    result["failure_category"] = "invalid_geometry"
            result["elapsed_seconds"] = round(time.time() - started, 3)
        except Exception as exc:  # noqa: BLE001
            result.update({"status": "error", "failure_category": "runtime_failure", "error": f"{type(exc).__name__}: {exc}"})
        (case_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results.append(result)

    valid = sum(bool(result.get("dsl_validation", {}).get("valid")) for result in results)
    wad_valid = sum(bool(result.get("wad_report", {}).get("valid")) for result in results)
    return {
        "status": "complete",
        "adapter": str(adapter),
        "model": MODEL_ID,
        "prompt_count": len(results),
        "json_valid_count": sum(bool(result.get("json_valid")) for result in results),
        "dsl_valid_count": valid,
        "wad_valid_count": wad_valid,
        "results": results,
    }


def _validation_failure_category(errors: list[str]) -> str:
    joined = " ".join(errors).lower()
    if "material" in joined:
        return "invalid_material"
    if "connection" in joined or "reachable" in joined or "gate" in joined:
        return "incorrect_progression"
    if "missing top-level" in joined or "missing" in joined:
        return "missing_required_field"
    return "schema_failure"


def _run_engine_smoke(wad_path: Path, log_path: Path) -> dict[str, Any]:
    zandronum = Path("C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe")
    doom2 = Path("C:/Users/gh0st/Downloads/DOOM2.WAD")
    if not (zandronum.exists() and doom2.exists()):
        return {"tested": False, "reason": "Zandronum or DOOM2.WAD unavailable"}
    try:
        result = subprocess.run(
            [
                str(zandronum),
                "-iwad",
                str(doom2),
                "-file",
                str(wad_path),
                "+logfile",
                str(log_path),
                "+map",
                "MAP01",
                "+quit",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return {
            "tested": True,
            "passed": result.returncode == 0,
            "exit_code": result.returncode,
            "log": str(log_path),
        }
    except Exception as exc:  # noqa: BLE001
        return {"tested": True, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _git_state() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
        value = result.stdout.strip()
        return value if result.returncode == 0 else None

    return {"branch": run("branch", "--show-current"), "commit": run("rev-parse", "HEAD"), "status": run("status", "--short")}


def _engine_environment() -> dict[str, Any]:
    zandronum = Path("C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe")
    doom2 = Path("C:/Users/gh0st/Downloads/DOOM2.WAD")
    return {"available": zandronum.exists() and doom2.exists(), "zandronum": str(zandronum), "doom2_iwad": str(doom2)}


if __name__ == "__main__":
    raise SystemExit(main())
