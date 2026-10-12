from __future__ import annotations

import json
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.evaluation import extract_json_object, template_similarity_report
from romerodsl.finetune import SYSTEM_PROMPT
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/v08_complex_prompt_step850"
ADAPTER = ROOT / "runs/qwen2p5_coder_1p5b_v08_full/adapter-step-850"
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
PROMPT = (
    "make a big eight room Doom 2 techbase level with a clear start and exit, "
    "a few separate combat areas, two different keycard rooms, a blue key and a red key, "
    "the blue key should open the route to the red key room, the red key should open the final exit wing, "
    "one optional side room should contain a BFG behind a locked door or key gate, "
    "include ammo, health, monsters, height variation, and make the route loop back through a hub instead of being a straight line"
)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"prompt": PROMPT, "adapter": str(ADAPTER)}
    print("loading tokenizer", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    print("loading model", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    print(f"loading adapter {ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    print("generating", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=3200,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    raw = tok.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    raw_path = OUT / "raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    summary["raw_path"] = str(raw_path)

    try:
        dsl = extract_json_object(raw)
        summary["json_valid"] = True
        dsl_path = OUT / "map.dsl.json"
        dsl_path.write_text(json.dumps(dsl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["dsl_path"] = str(dsl_path)
    except Exception as exc:  # noqa: BLE001 - persisted eval summary should record parse failures.
        summary.update(json_valid=False, error=f"json_parse_failed: {exc}")
        write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    report = validate_document(dsl)
    summary["dsl_valid"] = report.valid
    summary["dsl_errors"] = report.errors[:20]
    summary["dsl_warnings"] = report.warnings[:20]
    summary["space_count"] = len(dsl.get("spaces", [])) if isinstance(dsl.get("spaces"), list) else None
    summary["connection_count"] = len(dsl.get("connections", [])) if isinstance(dsl.get("connections"), list) else None
    summary["spaces"] = [
        {"id": s.get("id"), "role": s.get("role"), "size": s.get("size"), "shape": s.get("shape")}
        for s in dsl.get("spaces", [])
        if isinstance(s, dict)
    ]
    summary["connections"] = [
        {"id": c.get("id"), "from": c.get("from"), "to": c.get("to"), "type": c.get("type"), "key": c.get("key")}
        for c in dsl.get("connections", [])
        if isinstance(c, dict)
    ]
    summary["progression"] = dsl.get("progression")
    try:
        template = json.loads((ROOT / "examples/blue_lock_processing.json").read_text(encoding="utf-8"))
        summary.update(template_similarity_report(dsl, template))
    except Exception as exc:  # noqa: BLE001
        summary["template_similarity_error"] = str(exc)

    if report.valid:
        plan = compile_to_plan(dsl)
        (OUT / "map.plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        wad_path = OUT / "map.wad"
        compile_report = compile_to_wad(dsl, wad_path)
        wad_report = validate_wad(wad_path)
        summary["wad_path"] = str(wad_path)
        summary["compile_report"] = {
            key: compile_report.get(key)
            for key in ["valid", "room_sectors", "corridor_sectors", "door_sectors", "height_feature_sectors", "things_by_kind"]
        }
        summary["wad_report"] = {
            key: wad_report.get(key)
            for key in [
                "valid",
                "keys",
                "key_colors",
                "doors",
                "locked_doors",
                "locked_door_colors",
                "exits",
                "monsters",
                "things",
                "sectors",
                "missing_visible_textures",
                "invalid_side_references",
                "unreferenced_sidedefs",
                "floor_height_levels",
            ]
        }
        zand = Path("C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe")
        doom2 = Path("C:/Users/gh0st/Downloads/DOOM2.WAD")
        if zand.exists() and doom2.exists():
            log = OUT / "zandronum.log"
            result = subprocess.run(
                [str(zand), "-iwad", str(doom2), "-file", str(wad_path), "+logfile", str(log), "+map", "MAP01", "+quit"],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            summary["zandronum_exit_code"] = result.returncode
            summary["zandronum_log"] = str(log)
    else:
        summary["wad_path"] = None

    write_summary(summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def write_summary(summary: dict[str, object]) -> None:
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
