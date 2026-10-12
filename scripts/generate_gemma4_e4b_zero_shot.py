from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.evaluation import extract_json_object, template_similarity_report
from romerodsl.schema import validate_document
from romerodsl.wad import validate_wad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build/gemma4_e4b_zero_shot_complex"
MODEL_ID = "google/gemma-4-E4B-it"
PROMPT = (
    "Make a big eight room Doom 2 techbase level with a clear start and exit. "
    "It must have a few separate combat areas, two keycard rooms, a blue key and a red key. "
    "The blue key must open a blue locked route into a BFG armory room. "
    "The BFG armory must contain a BFG9000 as an item reward, not as a key. "
    "After the BFG armory the route must continue to the red key room. "
    "The red key must open the final exit wing. "
    "Include ammo, health, monsters, height variation, and a hub-like flow."
)
STRICT_SYSTEM_PROMPT = """You generate valid romeroDSL JSON for Doom II maps.
Return only one JSON object. Do not use markdown.
The JSON must use only this DSL shape:
- top-level keys: dsl_version, game, format_target, title, scale, difficulty, intended_playtime, design_intent, theme, progression, spaces, connections, validation.
- dsl_version must be "0.1"; game must be "doom2"; format_target must be "udmf".
- progression.start and progression.exit must reference space ids.
- progression.keys is a list of key objects with color one of: blue, red, yellow. No other key colors are legal.
- progression.gates is a list of gate objects. Required locked gates must use type "locked_door" and key one of: blue, red, yellow.
- BFG is never a key. BFG must appear only as an item string "bfg9000" inside a room's contains.items or things.items list.
- Every connection.from and connection.to must reference an existing space id.
- Connection type for locked doors must be "locked_door". Do not invent key_gate, locked_gate, extractable_item, or bfg key types.
- No self-connections.
- Exactly one player_start room and exactly one exit room.
For this prompt, create 8 spaces, including start_room, central_hub, blue_key_room, bfg_armory, red_key_room, exit_wing, and exit_room.
The critical path should place bfg_armory after blue_key_room and before red_key_room.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"model": MODEL_ID, "prompt": PROMPT}
    print("loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("loading model", flush=True)
    offload = OUT / "offload"
    offload.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else "auto",
        device_map="auto",
        max_memory={0: "10GiB", "cpu": "48GiB"} if torch.cuda.is_available() else {"cpu": "48GiB"},
        offload_folder=str(offload),
        trust_remote_code=True,
    )
    model.eval()

    messages = [
        {"role": "system", "content": STRICT_SYSTEM_PROMPT},
        {"role": "user", "content": PROMPT},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    print("generating", flush=True)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=4200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    raw_path = OUT / "raw.txt"
    raw_path.write_text(raw, encoding="utf-8")
    summary["raw_path"] = str(raw_path)

    try:
        dsl = extract_json_object(raw)
        summary["json_valid"] = True
        dsl_path = OUT / "map.dsl.json"
        dsl_path.write_text(json.dumps(dsl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary["dsl_path"] = str(dsl_path)
    except Exception as exc:  # noqa: BLE001
        summary.update(json_valid=False, error=f"json_parse_failed: {exc}")
        write_summary(summary)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0

    report = validate_document(dsl)
    summary["dsl_valid"] = report.valid
    summary["dsl_errors"] = report.errors[:30]
    summary["dsl_warnings"] = report.warnings[:30]
    spaces = dsl.get("spaces", []) if isinstance(dsl.get("spaces"), list) else []
    connections = dsl.get("connections", []) if isinstance(dsl.get("connections"), list) else []
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
    summary["progression"] = dsl.get("progression")
    summary["intent_checks"] = intent_checks(dsl)
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
        "has_blue_key": "blue" in key_colors,
        "has_red_key": "red" in key_colors,
        "has_only_legal_key_colors": key_colors <= {"blue", "red", "yellow"} and gate_colors <= {"blue", "red", "yellow"},
        "has_bfg_item": "bfg9000" in item_strings,
        "does_not_use_bfg_as_key": "bfg" not in key_colors and "bfg" not in gate_colors,
        "has_bfg_armory_space": "bfg_armory" in space_ids,
        "bfg_before_red_key_in_critical_path": _before(critical_path, "bfg_armory", "red_key_room"),
        "no_self_connections": all(c.get("from") != c.get("to") for c in connections if isinstance(c, dict)),
        "unsupported_connection_types": sorted(
            {
                str(c.get("type"))
                for c in connections
                if isinstance(c, dict) and c.get("type") not in {"doorway", "door", "stairs", "hallway", "locked_door", "keyed_door", "hidden_wall"}
            }
        ),
    }


def _before(values: list[str], left: str, right: str) -> bool:
    return left in values and right in values and values.index(left) < values.index(right)


def write_summary(summary: dict[str, Any]) -> None:
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
