from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from romerodsl.compiler import compile_to_wad
from romerodsl.evaluation import (
    evaluate_prompt_intent,
    extract_json_object,
    template_similarity_report,
    v08_prompt_expectations,
    v08_prompt_suite,
)
from romerodsl.finetune import SYSTEM_PROMPT
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad


BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_RELATIVE = "runs/qwen2p5_coder_1p5b_v07_slim/adapter"
ZANDRONUM = Path("C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe")
DOOM2 = Path("C:/Users/gh0st/Downloads/DOOM2.WAD")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    adapter = root / ADAPTER_RELATIVE
    out_dir = root / "build/v08_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    template = json.loads((root / "examples/blue_lock_processing.json").read_text(encoding="utf-8"))

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    print("loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    print("loading adapter", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    summaries = []
    prompts = v08_prompt_suite()
    expectations = v08_prompt_expectations()
    for index, (prompt, expectation) in enumerate(zip(prompts, expectations), start=1):
        slug = f"prompt_{index:02d}"
        print(f"[{index}/{len(prompts)}] generating {slug}", flush=True)
        summary = evaluate_prompt(model, tokenizer, prompt, template, out_dir, slug)
        summary.update(evaluate_prompt_intent(summary, expectation))
        summaries.append(summary)
        status = {
            "slug": slug,
            "json_valid": summary["json_valid"],
            "dsl_valid": summary["dsl_valid"],
            "wad_valid": summary["wad_valid"],
            "zandronum_exit_code": summary["zandronum_exit_code"],
            "template_similarity": summary.get("template_similarity"),
            "keys": summary.get("keys"),
            "locked_doors": summary.get("locked_doors"),
            "key_lock_pairs_valid": summary.get("key_lock_pairs_valid"),
            "prompt_intent_valid": summary.get("prompt_intent_valid"),
        }
        print(json.dumps(status, sort_keys=True), flush=True)

    aggregate = aggregate_summaries(summaries)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({"aggregate": aggregate, "results": summaries}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    print(f"summary={summary_path}", flush=True)
    return 0 if aggregate["fully_passing"] > 0 else 1


def evaluate_prompt(model: Any, tokenizer: Any, prompt: str, template: dict[str, Any], out_dir: Path, slug: str) -> dict[str, Any]:
    raw_path = out_dir / f"{slug}_raw.txt"
    dsl_path = out_dir / f"{slug}.dsl.json"
    wad_path = out_dir / f"{slug}.wad"
    zandronum_log = out_dir / f"{slug}_zandronum.log"
    summary: dict[str, Any] = {
        "slug": slug,
        "prompt": prompt,
        "raw_path": str(raw_path),
        "dsl_path": str(dsl_path),
        "wad_path": str(wad_path),
        "json_valid": False,
        "dsl_valid": False,
        "wad_valid": False,
        "zandronum_exit_code": None,
    }
    generated = generate(model, tokenizer, prompt)
    raw_path.write_text(generated, encoding="utf-8")

    try:
        dsl = extract_json_object(generated)
    except Exception as exc:  # noqa: BLE001 - persisted eval summary should record parse failures.
        summary["error"] = f"json_parse_failed: {exc}"
        return summary
    summary["json_valid"] = True
    dsl_path.write_text(json.dumps(dsl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary.update(template_similarity_report(dsl, template))

    dsl_report = validate_document(dsl)
    summary["dsl_valid"] = dsl_report.valid
    summary["dsl_errors"] = dsl_report.errors[:10]
    summary["dsl_warnings"] = dsl_report.warnings[:10]
    if not dsl_report.valid:
        return summary

    compile_report = compile_to_wad(dsl, wad_path)
    wad_report = validate_wad(wad_path)
    summary.update(
        {
            "wad_valid": bool(wad_report.get("valid")),
            "geometry_profile": wad_report.get("geometry_profile", compile_report.get("geometry_profile")),
            "sectors": wad_report.get("sectors"),
            "things": wad_report.get("things"),
            "keys": wad_report.get("keys"),
            "key_colors": wad_report.get("key_colors"),
            "locked_doors": wad_report.get("locked_doors"),
            "locked_door_colors": wad_report.get("locked_door_colors"),
            "key_lock_pairs_valid": wad_report.get("key_lock_pairs_valid"),
            "key_lock_pair_errors": wad_report.get("key_lock_pair_errors"),
            "missing_visible_textures": wad_report.get("missing_visible_textures"),
            "unknown_flats": wad_report.get("unknown_flats"),
            "unknown_textures": wad_report.get("unknown_textures"),
            "invalid_side_references": wad_report.get("invalid_side_references"),
            "unreferenced_sidedefs": wad_report.get("unreferenced_sidedefs"),
        }
    )
    if wad_report.get("valid"):
        summary["zandronum_exit_code"] = run_zandronum_parser(wad_path, zandronum_log)
        summary["zandronum_log"] = str(zandronum_log)
    return summary


def generate(model: Any, tokenizer: Any, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def run_zandronum_parser(wad_path: Path, log_path: Path) -> int | None:
    if not ZANDRONUM.exists() or not DOOM2.exists():
        return None
    result = subprocess.run(
        [
            str(ZANDRONUM),
            "-iwad",
            str(DOOM2),
            "-file",
            str(wad_path),
            "+logfile",
            str(log_path),
            "+map",
            "MAP01",
            "+quit",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result.returncode


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompts": len(summaries),
        "json_valid": sum(1 for item in summaries if item.get("json_valid")),
        "dsl_valid": sum(1 for item in summaries if item.get("dsl_valid")),
        "wad_valid": sum(1 for item in summaries if item.get("wad_valid")),
        "zandronum_ok": sum(1 for item in summaries if item.get("zandronum_exit_code") == 0),
        "key_lock_pairs_valid": sum(1 for item in summaries if item.get("key_lock_pairs_valid")),
        "prompt_intent_valid": sum(1 for item in summaries if item.get("prompt_intent_valid")),
        "high_template_similarity": sum(1 for item in summaries if item.get("template_similarity") == "high"),
        "clean_non_template_success": sum(
            1
            for item in summaries
            if item.get("prompt_intent_valid") and item.get("template_similarity") != "high"
        ),
        "fully_passing": sum(
            1
            for item in summaries
            if item.get("prompt_intent_valid")
        ),
    }


if __name__ == "__main__":
    raise SystemExit(main())
