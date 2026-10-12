from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.evaluation import extract_json_object, template_similarity_report
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "google/gemma-4-E4B-it"
DEFAULT_ADAPTER = ROOT / "runs/gemma4_e4b_v09_light/adapter-step-25"
DEFAULT_OUTPUT = ROOT / "build/gemma4_e4b_v09_light_step25_eval"
PROMPT = (
    "Create a big eight-room Doom 2 corrupted techbase/refinery level with a clear start and exit. "
    "Use a hub-like route with several combat areas, two keycard rooms, a yellow key and a red key. "
    "The yellow key must open a yellow locked route into a BFG armory room. "
    "The BFG armory must contain a BFG9000 as an item reward, not as a key. "
    "After the BFG armory the route must continue to the red key room. "
    "The red key must open the final exit wing. "
    "Include ammo, health, monsters, height variation, and distinct room shapes."
)
SYSTEM_PROMPT = (
    "You generate valid romeroDSL JSON for Doom II maps. "
    "Return only one JSON object. Do not wrap it in markdown. "
    "Use only valid Doom II material names from the training schema. "
    "BFG is an item reward named bfg9000, never a key or gate type. "
    "Required locked gates must be real route gates; do not create bypass connections around them."
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-new-tokens", type=int, default=3500)
    args = parser.parse_args()

    adapter = args.adapter.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model": MODEL_ID, "adapter": str(adapter), "prompt": PROMPT}
    if not adapter.exists():
        raise FileNotFoundError(f"adapter directory does not exist: {adapter}")

    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("loading base 4bit", flush=True)
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
    print("loading adapter", adapter, flush=True)
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    (out / "prompt.txt").write_text(text, encoding="utf-8")
    inputs = tokenizer(text, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)
    print(f"prompt_tokens {inputs['input_ids'].shape[1]}", flush=True)
    print("generating", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    raw_path = out / "raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    summary["raw_path"] = str(raw_path)

    try:
        dsl = extract_json_object(raw)
        summary["json_valid"] = True
        dsl_path = out / "map.dsl.json"
        dsl_path.write_text(json.dumps(dsl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["dsl_path"] = str(dsl_path)
    except Exception as exc:  # noqa: BLE001
        summary.update(json_valid=False, error=f"json_parse_failed: {exc}")
        write_summary(out, summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    report = validate_document(dsl)
    summarize_dsl(summary, dsl, report)
    try:
        example = json.loads((ROOT / "examples/complex_eight_room_bfg.json").read_text(encoding="utf-8"))
        summary.update(template_similarity_report(dsl, example))
    except Exception as exc:  # noqa: BLE001
        summary["template_similarity_error"] = str(exc)

    if report.valid:
        plan = compile_to_plan(dsl)
        (out / "map.plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        wad_path = out / "map.wad"
        compile_report = compile_to_wad(dsl, wad_path)
        wad_report = validate_wad(wad_path)
        summary["wad_path"] = str(wad_path)
        summary["compile_report"] = compact_compile_report(compile_report)
        summary["wad_report"] = compact_wad_report(wad_report)
        maybe_run_zandronum(summary, out, wad_path)
    else:
        summary["wad_path"] = None

    write_summary(out, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


def summarize_dsl(summary: dict[str, Any], dsl: dict[str, Any], report: Any) -> None:
    spaces = dsl.get("spaces", []) if isinstance(dsl.get("spaces"), list) else []
    connections = dsl.get("connections", []) if isinstance(dsl.get("connections"), list) else []
    progression = dsl.get("progression", {}) if isinstance(dsl.get("progression"), dict) else {}
    summary["dsl_valid"] = report.valid
    summary["dsl_errors"] = report.errors[:40]
    summary["dsl_warnings"] = report.warnings[:40]
    summary["space_count"] = len(spaces)
    summary["connection_count"] = len(connections)
    summary["spaces"] = [
        {"id": s.get("id"), "role": s.get("role"), "size": s.get("size"), "shape": s.get("shape")}
        for s in spaces
        if isinstance(s, dict)
    ]
    summary["connections"] = [
        {"id": c.get("id"), "from": c.get("from"), "to": c.get("to"), "type": c.get("type"), "key": c.get("key")}
        for c in connections
        if isinstance(c, dict)
    ]
    summary["progression"] = progression
    summary["intent_checks"] = intent_checks(dsl)


def intent_checks(dsl: dict[str, Any]) -> dict[str, Any]:
    spaces = dsl.get("spaces", []) if isinstance(dsl.get("spaces"), list) else []
    connections = dsl.get("connections", []) if isinstance(dsl.get("connections"), list) else []
    progression = dsl.get("progression", {}) if isinstance(dsl.get("progression"), dict) else {}
    space_ids = [str(s.get("id")) for s in spaces if isinstance(s, dict)]
    key_colors = {str(k.get("color")) for k in progression.get("keys", []) if isinstance(k, dict)}
    gate_colors = {str(g.get("key")) for g in progression.get("gates", []) if isinstance(g, dict)}
    item_strings = json.dumps(spaces).lower()
    critical_path = [str(node) for node in progression.get("critical_path", [])] if isinstance(progression.get("critical_path"), list) else []
    return {
        "has_8_spaces": len(spaces) == 8,
        "has_yellow_key": "yellow" in key_colors,
        "has_red_key": "red" in key_colors,
        "has_only_legal_key_colors": key_colors <= {"blue", "red", "yellow"} and gate_colors <= {"blue", "red", "yellow"},
        "has_bfg_item": "bfg9000" in item_strings,
        "does_not_use_bfg_as_key": "bfg" not in key_colors and "bfg" not in gate_colors,
        "has_bfg_armory_space": any("bfg" in sid and "armory" in sid for sid in space_ids),
        "bfg_before_red_key_in_critical_path": _contains_before(critical_path, "bfg", "red_key"),
        "no_self_connections": all(c.get("from") != c.get("to") for c in connections if isinstance(c, dict)),
        "unsupported_connection_types": sorted(
            {
                str(c.get("type"))
                for c in connections
                if isinstance(c, dict) and c.get("type") not in {"doorway", "door", "stairs", "hallway", "locked_door", "keyed_door", "hidden_wall"}
            }
        ),
    }


def _contains_before(values: list[str], left_part: str, right_part: str) -> bool:
    left_indexes = [i for i, value in enumerate(values) if left_part in value]
    right_indexes = [i for i, value in enumerate(values) if right_part in value]
    return bool(left_indexes and right_indexes and min(left_indexes) < min(right_indexes))


def compact_compile_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: report.get(key) for key in ["valid", "geometry_profile", "room_sectors", "corridor_sectors", "door_sectors", "height_feature_sectors", "things_by_kind"]}


def compact_wad_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report.get(key)
        for key in [
            "valid",
            "geometry_profile",
            "keys",
            "key_colors",
            "doors",
            "door_linedefs",
            "bidirectional_door_linedefs",
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


def maybe_run_zandronum(summary: dict[str, Any], out: Path, wad_path: Path) -> None:
    zand = Path("C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe")
    doom2 = Path("C:/Users/gh0st/Downloads/DOOM2.WAD")
    if not (zand.exists() and doom2.exists()):
        return
    log = out / "zandronum.log"
    result = subprocess.run(
        [str(zand), "-iwad", str(doom2), "-file", str(wad_path), "+logfile", str(log), "+map", "MAP01", "+quit"],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    summary["zandronum_exit_code"] = result.returncode
    summary["zandronum_log"] = str(log)


def write_summary(out: Path, summary: dict[str, Any]) -> None:
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
