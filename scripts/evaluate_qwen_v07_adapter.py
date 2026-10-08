from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from romerodsl.compiler import compile_to_wad
from romerodsl.qwen_lora import load_sft_records, split_records
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    adapter = root / "runs/qwen2p5_coder_1p5b_v07_slim/adapter"
    out_dir = root / "build/v07_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_sft_records(root / "build/v07_sft/romerodsl_sft_v07.jsonl")
    split = split_records(records, validation_fraction=0.08, seed=1337)
    record = split["validation"][0]
    prompt = record["messages"][1]["content"]
    messages = record["messages"][:-1]
    base = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    print("loading base model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    print("loading adapter", flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print("generating", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=2400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, inputs["input_ids"].shape[1] :]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    raw_path = out_dir / "qwen_v07_smoke_output.txt"
    raw_path.write_text(generated, encoding="utf-8")
    print(f"raw_output={raw_path}", flush=True)
    try:
        dsl = json.loads(generated)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", generated, re.S)
        if not match:
            print("json_valid=false")
            return 2
        dsl = json.loads(match.group(0))
    report = validate_document(dsl)
    print(f"dsl_valid={report.valid}", flush=True)
    if not report.valid:
        print(json.dumps([error.__dict__ for error in report.errors[:10]], indent=2), flush=True)
        return 3
    dsl_path = out_dir / "qwen_v07_smoke.dsl.json"
    dsl_path.write_text(json.dumps(dsl, indent=2) + "\n", encoding="utf-8")
    wad_path = out_dir / "qwen_v07_smoke.wad"
    compile_report = compile_to_wad(dsl, wad_path)
    wad_report = validate_wad(wad_path)
    summary = {
        "prompt": prompt,
        "raw_output": str(raw_path),
        "dsl_path": str(dsl_path),
        "wad_path": str(wad_path),
        "dsl_valid": report.valid,
        "wad_valid": wad_report.get("valid"),
        "geometry_profile": wad_report.get("geometry_profile", compile_report.get("geometry_profile")),
        "sectors": wad_report.get("sectors"),
        "things": wad_report.get("things"),
        "keys": wad_report.get("keys"),
        "locked_doors": wad_report.get("locked_doors"),
        "missing_visible_textures": wad_report.get("missing_visible_textures"),
        "unknown_flats": wad_report.get("unknown_flats"),
        "unknown_textures": wad_report.get("unknown_textures"),
        "invalid_side_references": wad_report.get("invalid_side_references"),
        "unreferenced_sidedefs": wad_report.get("unreferenced_sidedefs"),
    }
    summary_path = out_dir / "qwen_v07_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
