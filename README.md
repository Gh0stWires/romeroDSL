# romeroDSL

A high-level semantic DSL for Doom map generation.

`romeroDSL` is the new graph-first path for RomeroGAN: instead of asking a model to emit raster cells or linedefs directly, it asks a model to author a structured Doom design contract that can be validated, compiled into geometry, exported to WAD/UDMF, and playtested.

The boundary is deliberate:

- The **model** authors design: spaces, progression, keys, locks, height topology, combat beats, resources, secrets, and theme.
- The **compiler** authors mechanics: vertices, sectors, linedefs, tags, sidedefs, UDMF formatting, and texture application.
- The **validator** rejects missing or fake progression instead of silently repairing it.

This keeps room topology, height/elevation, and key progression as first-class learned concepts while leaving room to reuse the existing RomeroGAN raster/graph experiments later as local geometry/detail generators.

## Current status

This is v0.10-pre: a GitHub-ready semantic Doom DSL pipeline plus early no-repair model-training experiments. The compiler/validator path is solid enough to prove hand-authored maps with real rooms, keyed progression, BFG rewards, material validation, and UDMF/WAD export. The model path is improving, but it is not yet a solved `prompt -> playable WAD` system.

The current best training run is the clean Gemma 4 E4B 4-bit QLoRA adapter at `runs/gemma4_e4b_v10_light_clean/adapter`. It trained on the rebuilt Gemma-light SFT set after fixing the multi-key self-connection bug and adding quality gates for self-connections, invalid materials, missing `spaces`, and route bypasses. The run completed 100 steps with final train loss `0.01657` and validation loss `0.01367` on the canonical SFT distribution.

The latest no-repair Gemma eval is a partial success, not a pass: raw generation produced valid JSON and followed several prompt intents, but failed romeroDSL validation because it created a red locked gate without declaring a matching red key in `progression.keys`. It also collapsed an eight-room request to six spaces, so no WAD was compiled. This means the current blocker has moved from syntax/JSON format to complete prompt intent: requested key colors, matching keys/gates, and explicit room-count obedience.

The package currently contains:

- a canonical JSON form of the DSL
- an example Doom II techbase map contract
- a Python validator for references and progression invariants
- a debug cell-layout compiler
- a sector-primitive geometry compiler that turns semantic rooms/connections into UDMF sectors
- a material layer for floor, ceiling, wall, upper/lower, door, door-track, and exit-switch surfaces
- render-surface validation for missing visible textures and invalid Doom II flats/textures
- safer placement for starts, keys, items, and monsters around blocking height features
- chamfered/octagonal room boundaries for authored irregular spaces
- stair-step sectors for raised platforms and pits that declare stair access/escape
- a minimal dependency-free UDMF/PWAD exporter
- unit tests
- graph-cache extraction into canonical prompt-to-romeroDSL training pairs
- full graph-cache audit and v0.8 stratified buckets for no-lock, optional-lock, single-lock, multi-lock, and layout/style-only samples
- robust prompt variants that make start/exit universal while keeping keys/locks prompt-conditioned
- chat-style v0.7/v0.8 SFT export/validation helpers
- a Qwen2.5-Coder LoRA training harness generalized enough to run Gemma 4 E4B 4-bit QLoRA
- v0.8 WAD validation that reports concrete key colors, locked-door colors, and mismatched key/lock pairs
- Gemma-light SFT builder quality gates for self-connections, invalid materials, missing `spaces`, and route bypasses
- no-repair Gemma zero-shot, one-shot, checkpoint, and final-adapter eval scripts

The v0.5 WAD exporter uses real room sectors, corridor sectors, tagged door sectors, chamfered irregular-room boundaries, and intra-room height-feature sectors for pillars, stair-access platforms, and stair-escape pits. It resolves semantic material declarations into concrete Doom II flats/textures, rejects missing visible surface textures that can create hall-of-mirrors artifacts in Zandronum, rejects crossing linedefs, and avoids placing actors/items inside blocking height features.

The v0.6/v0.8 extraction layer turns RomeroGAN `graph_cache_v1` samples into canonical prompt-to-DSL records. v0.8 no longer treats keyed progression as the default level contract: it audits the full cache, emits stratified buckets, creates multiple grounded prompt phrasings per source map, and keeps keys/locks out of the assistant target unless the source graph has required exit-gating evidence. The universal target is a valid start, exit, and connected route; what happens between them is controlled by the prompt.

The first v0.7 slice converts those validated prompt-to-DSL records into chat-style supervised fine-tuning JSONL and validates every assistant output as parseable, canonical romeroDSL before training. The slim Qwen adapter can generate valid romeroDSL that compiles to a Zandronum-loadable WAD, but the current generated samples are not yet proof of broad map-design generalization.

The first v0.8 slice fixes and validates concrete ZDoom keycard/lock mechanics. A reported red/blue/yellow locked door must now use a lock number that matches the exported visible keycard color, and mismatched key/lock pairs mark the WAD report invalid instead of passing as a misleading object count. v0.8 has also started semantic route-gating checks: a required locked gate must be a concrete connection, its key must be reachable before crossing that gate, and the exit must not be reachable while that gate is blocked.

The v0.9/v0.10 Gemma-light work uses `google/gemma-4-E4B-it` as a stronger planner/DSL generator candidate. Zero-shot Gemma showed promising semantic planning but failed strict JSON. One-shot Gemma produced valid JSON but invalid romeroDSL due unsupported material names and a route bypass. The clean v0.10 QLoRA run fixed the dataset self-connection contamination and successfully learned the compact canonical JSON style; the remaining failure is higher-level prompt completion, especially declaring every requested key and preserving explicit room counts.

## Install for development

```bash
python -m pip install -e .
```

No runtime dependencies are required for the compiler/validator path. Training requires the optional `train` dependency group.

## Validate the example

```bash
python -m romerodsl validate examples/blue_lock_processing.json
```

Expected result:

```text
valid: examples/blue_lock_processing.json
```

## Export a first playable UDMF PWAD

```bash
python -m romerodsl export-wad examples/blue_lock_processing.json --output build/blue_lock_processing.wad
```

The exported WAD should contain one player start, one exit, a blue key, at least one blue-locked door, monsters, items, real room/corridor/door sectors, sector height variation from the DSL, zero missing visible textures, and no unknown Doom II flats/textures. In v0.8, the WAD report also includes `key_colors`, `locked_door_colors`, `key_lock_pairs_valid`, and `key_lock_pair_errors` so a mismatched visible key and UDMF lock number cannot pass as a valid keyed-door report.

## Emit an abstract geometry plan

```bash
python -m romerodsl compile examples/blue_lock_processing.json --output build/blue_lock_processing.plan.json
```

The emitted plan is a debug bridge between semantic authorship and UDMF/WAD export.

## Audit and build prompt-to-DSL training pairs

Audit the full graph cache before training so the dataset shape is explicit:

```bash
python -m romerodsl audit-graph-cache \
  --graph-cache C:/Users/gh0st/Documents/RomeroGan/romerogan_v2/data/graph_cache_v1 \
  --output build/v08_full_dataset_audit.json
```

Build the v0.8 full-dataset cache with prompt variants instead of the old strict-key-only slice:

```bash
python -m romerodsl build-training-pairs \
  --graph-cache C:/Users/gh0st/Documents/RomeroGan/romerogan_v2/data/graph_cache_v1 \
  --output build/v08_full_dsl_training_pairs \
  --prompt-variants-per-sample 4
```

The command writes:

- `manifest.jsonl` with prompt, prompt variant, labels, features, and sample path
- `summary.json` with sample/theme/color/bucket counts
- `samples/dsl_*.json` training records containing `{prompt, dsl, labels, features}`

`--strict-only` is now only for reproducing the older keyed-progression subset. The v0.8 default uses the full graph cache and buckets samples by what they can honestly supervise: `basic_route`, `optional_key_lock`, `required_single_lock`, `required_multi_lock`, and `layout_style_only`. Start, exit, and a playable route are the core generated contract; keys and locks are only placed in the assistant target when the source graph says a lock gates exit progression.

## Build v0.8 supervised fine-tuning JSONL

```bash
python -m romerodsl build-sft-jsonl \
  --training-cache build/v08_full_dsl_training_pairs \
  --output build/v08_sft/romerodsl_sft_v08_full.jsonl

python -m romerodsl validate-sft-jsonl build/v08_sft/romerodsl_sft_v08_full.jsonl
```

The SFT file uses chat-style records with a system instruction, the user prompt, and an assistant response that is only the canonical DSL JSON object. `validate-sft-jsonl` reparses every assistant response and runs `validate_document()` before the file is considered trainable. The assistant target omits source-WAD provenance, raw source height arrays, and extracted source titles, so the model learns the authored level contract instead of memorizing seed names.

## Train v0.8 Qwen LoRA adapter

The first training target was `Qwen/Qwen2.5-Coder-1.5B-Instruct` because romeroDSL is strict JSON/code-like output and the 1.5B model is small enough for local iteration.

```bash
python -m romerodsl train-qwen-lora \
  --sft-jsonl build/v08_sft/romerodsl_sft_v08_full.jsonl \
  --output runs/qwen2p5_coder_1p5b_v08_full \
  --max-length 2400 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1 \
  --learning-rate 0.0002
```

The trainer writes `train_config.json`, `metrics.json`, periodic `adapter-step-*` checkpoints, and the final LoRA adapter under the output directory. The v0.8 target uses the full graph cache plus prompt variants and intentionally includes no-lock/basic-route prompts so the model learns that only start, exit, and a connected route are universal. Keys, locked doors, multi-key gates, combat, resources, and height changes are prompt-conditioned features, not mandatory defaults.

## Train Gemma 4 E4B light QLoRA adapter

The stronger local planner candidate is `google/gemma-4-E4B-it`. On the local 12GB RTX 4070 setup, use 4-bit QLoRA rather than BF16 LoRA with CPU/meta offload.

Build the compact, quality-gated Gemma-light SFT subset:

```bash
python scripts/build_gemma_light_sft.py
```

Then train with the Gemma-capable LoRA harness:

```bash
python -m romerodsl train-qwen-lora \
  --sft-jsonl build/v09_gemma_sft/romerodsl_sft_gemma_light.jsonl \
  --output runs/gemma4_e4b_v10_light_clean \
  --model google/gemma-4-E4B-it \
  --max-length 2500 \
  --batch-size 1 \
  --gradient-accumulation-steps 4 \
  --max-steps 100 \
  --validation-fraction 0.02 \
  --eval-every-steps 50 \
  --save-every-steps 25 \
  --learning-rate 0.0001 \
  --lora-rank 8 \
  --lora-alpha 16 \
  --load-in-4bit
```

The known clean run completed successfully with 3,508 SFT records, 3,438 train records, 70 validation records, and final validation loss `0.01367`. Low loss here means the adapter learned the compact canonical SFT distribution; it does not by itself prove free-generation quality. After training, evaluate with no repair: raw text must parse as JSON, pass `validate_document()`, compile to WAD, and pass WAD validation before the adapter counts as useful.

## Design goals

The DSL should be high-level, but not vague. It should capture what a Doom mapper means:

- map scale and theme
- start/exit progression
- critical path
- rooms/spaces and semantic roles
- doors, locked doors, stairs, lifts, drops, teleporters, secrets
- keys reachable before matching locks
- height topology inside rooms: stairs, pits, pillars, platforms, ledges
- combat pacing and monster groups
- weapon/ammo/health economy
- material intent for floors, ceilings, walls, steps, doors, tracks, and switches
- validation requirements

The compiler may make valid geometry from those decisions, but it should not invent missing progression.

## Roadmap

1. v0.1: DSL schema, examples, validator, abstract plan compiler.
2. v0.2: WAD/UDMF compiler for a constrained subset: rooms, hallways, doors, blue/red/yellow locks, starts, exits, monsters, items.
3. v0.3: Replace the cell-block compiler with sector-primitive geometry: room sectors, corridor sectors, tagged door sectors, and intra-room height-feature sectors.
4. v0.4: Add material-aware surfaces, render-surface validation, and fix Zandronum HOM defects found during engine playtesting.
5. v0.5: Improve geometry authorship: polygonal/irregular rooms, doorway cutouts, stairs, safer monster/item placement, and deeper engine playtesting. Lift specials are deferred until the compiler has a namespace-verified mover model instead of a guessed action number.
6. v0.6: Extract real WAD graph caches into canonical prompt-to-DSL training pairs.
7. v0.7: LoRA fine-tune target: `prompt -> romeroDSL`.
8. v0.8: Stricter playability validation: concrete ZDoom key/lock mechanics, route-gated progression checks, template-leak detection, multi-prompt evaluation, full-dataset stratified SFT rebuild, and no-repair retraining.
9. v0.9: Gemma 4 E4B no-repair zero-shot/one-shot experiments and light QLoRA setup. This phase proved Gemma has stronger semantic planning than the earlier Qwen slice, but raw zero-shot/one-shot output was still invalid under the DSL contract.
10. v0.10: Clean Gemma-light rebuild after fixing multi-key self-connections and adding SFT quality gates. The final clean adapter emits valid JSON and follows the BFG/yellow-gate route better, but still fails strict DSL when it omits a requested red key and collapses explicit eight-room prompts to six spaces.
11. Next: tighten no-repair eval for requested key colors and requested room counts, add more varied eight-room/multi-key golden examples, rebuild SFT, and run a targeted shorter fine-tune before moving on to richer geometry layers.

## Repository layout

```text
examples/                  Example DSL documents
src/romerodsl/             Python package
  schema.py                DSL validation rules
  compiler.py              Abstract geometry-plan compiler
  extraction.py            Graph-cache to prompt-to-DSL training-pair extraction
  finetune.py              v0.7 SFT JSONL build/validation helpers
  qwen_lora.py             Qwen/Gemma LoRA and 4-bit QLoRA training harness
  evaluation.py            No-repair generation, prompt-intent, and template-similarity helpers
  cli.py                   Command line interface
scripts/
  build_gemma_light_sft.py                 Quality-gated Gemma-light SFT subset builder
  generate_gemma4_e4b_zero_shot.py         Gemma zero-shot no-repair eval
  generate_gemma4_e4b_one_shot.py          Gemma one-shot no-repair eval
  generate_gemma4_e4b_lora_checkpoint.py   Gemma adapter no-repair eval
tests/                     Unit tests
```
