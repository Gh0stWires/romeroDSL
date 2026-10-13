# romeroDSL

A high-level semantic DSL for Doom map generation.

`romeroDSL` is the new graph-first path for RomeroGAN: instead of asking a model to emit raster cells or linedefs directly, it asks a model to author a structured Doom design contract that can be validated, compiled into geometry, exported to WAD/UDMF, and playtested.

The boundary is deliberate:

- The **model** authors design: spaces, progression, keys, locks, height topology, combat beats, resources, secrets, and theme.
- The **compiler** authors mechanics: vertices, sectors, linedefs, tags, sidedefs, UDMF formatting, and texture application.
- The **validator** rejects missing or fake progression instead of silently repairing it.

This keeps room topology, height/elevation, and key progression as first-class learned concepts while leaving room to reuse the existing RomeroGAN raster/graph experiments later as local geometry/detail generators.

## Current status

This is v0.13: a GitHub-ready semantic Doom DSL pipeline plus early no-repair model-training experiments, a deterministic complexity-balanced curriculum builder, and leakage-resistant SFT splitting/token diagnostics. The compiler/validator path is solid enough to prove hand-authored maps with real rooms, keyed progression, BFG rewards, material validation, and UDMF/WAD export. The model path is improving, but it is not yet a solved `prompt -> playable WAD` system.

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
- shape-aware L/T/inset/alcove/corridor/closet footprints with north/east/south/west orientation
- central raised-platform and depressed-floor features inferred from the room shape
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

## Phase 1 geometry gallery

The compiler accepts canonical room shapes including `rectangle`, `rectangle_with_side_alcoves`, `l_shaped`, `t_shaped`, `irregular_orthogonal`, `inset_chamber`, `raised_center`, `depressed_floor`, `irregular_octagonal`, `bent_corridor`, `narrow_hall_with_cover`, and `closet`. Shapes may use `0`, `90`, `180`, or `270` degree orientation values, or the aliases `north`, `east`, `south`, and `west`. Unsupported shape and orientation values are rejected by validation rather than silently compiled as rectangles.

A hand-authored export covering representative Phase 1 geometry cases is available at `examples/phase1_geometry_gallery.json`:

```bash
python -m romerodsl validate examples/phase1_geometry_gallery.json
python -m romerodsl export-wad examples/phase1_geometry_gallery.json --output build/phase1_geometry_gallery.wad
```

The gallery is intentionally an authored compiler fixture, not a model-quality claim. Generated model output remains subject to strict no-repair validation.

## Phase 2 graph-aware layout

Room placement is graph-driven rather than a fixed horizontal cursor. The compiler builds deterministic BFS layers from the start room, groups related branch rooms around their graph parent, preserves loop and reconnection edges, resolves footprint collisions, and records the layout seed and corridor-orientation counts in the abstract plan. Set `layout_seed` in a DSL document to reproduce a layout or generate a deterministic mirrored arrangement.

The hand-authored acceptance fixture is `examples/phase2_graph_layout.json`. It contains eight spaces, multiple branches, a return loop, a secret branch, and a required blue locked exit gate:

```bash
python -m romerodsl validate examples/phase2_graph_layout.json
python -m romerodsl compile examples/phase2_graph_layout.json --output build/phase2_graph_layout.plan.json
python -m romerodsl export-wad examples/phase2_graph_layout.json --output build/phase2_graph_layout.wad
```

The layout stage rejects invalid seed types and overlapping room boxes before geometry export. It does not invent connections or alter semantic progression.

## Phase 3 topology-preserving source extraction

Phase 3 adds an explicit source-grounded extraction mode without changing the legacy default. `legacy_canonical` remains available for regression comparisons, but its output is labeled `dataset_class: synthetic` because it is a procedural scaffold. Use `topology_preserving` when the graph cache contains enough evidence:

```bash
python -m romerodsl build-training-pairs \
  --graph-cache C:/Users/gh0st/Documents/RomeroGan/romerogan_v2/data/graph_cache_v1 \
  --output build/phase3_topology_training_pairs \
  --extraction-mode topology_preserving \
  --prompt-variants-per-sample 4
```

This mode groups same-height sectors joined by source `connection` edges into approximate spaces, reconstructs room adjacency from traversable source edges, and preserves observed branches, cycles, height bands, room roles, entities, exits, and route-required key/lock evidence. Samples without reliable start/exit route evidence are rejected with `TopologyExtractionError`; the extractor does not fabricate a canonical route and call it extracted. Training records carry both `dataset_class: extracted` and `extraction_mode: topology_preserving` labels.

The graph cache does not contain original linedef vertex loops or complete thing coordinates. Therefore exact wall polygons, portal positions, and encounter placement are explicitly approximated. The source WAD path is retained as provenance for supported samples and is the next input for exact geometry recovery, but Phase 3 intentionally uses only graph-cache evidence.

Generate the deterministic source-versus-target audit report with:

```bash
python -m romerodsl audit-topology-extraction \
  --graph-cache C:/Users/gh0st/Documents/RomeroGan/romerogan_v2/data/graph_cache_v1 \
  --output reports/phase3_topology_extraction_report.json \
  --limit 25
```

The report compares source and DSL room counts, connections, connected components, branch signals, cycle counts, and spatial shape complexity. It also records supported/rejected samples and lists information preserved, approximated, and unavailable. The checked-in representative report is evidence about extraction coverage, not model-generation quality.

Dataset classes are intentionally separate:

- `extracted`: grounded in graph-cache/source-map evidence.
- `synthetic`: procedurally generated and validated, including the legacy canonical extractor.
- `hand-authored`: deliberate compiler fixtures such as the Phase 1 and Phase 2 examples.


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

## Phase 4 complexity-balanced curriculum data

Phase 4 adds a deterministic synthetic curriculum for teaching varied authored structure without confusing synthetic examples with extracted WAD evidence. The generator supports four tiers:

- `basic`: 3–5 spaces with simple but varied routes.
- `intermediate`: 6–9 spaces with branches, multiple silhouettes, and height variation.
- `advanced`: 10–15 spaces with loops, optional areas, and multi-step progression.
- `expert`: 16–24 spaces with multiple hubs and meaningful exploration.

Each generated document is independently varied across topology, room size and shape, theme/material preset, height features, encounters, resources, secrets, locks, connection types, and progression. Prompts are generated from measured target statistics, including the exact authored room count. A requested twelve-room prompt therefore targets exactly twelve spaces.

The generator validates every target, compiles it through the abstract geometry compiler, and rejects ID-independent duplicate layout signatures. It is available directly for experiments:

```python
from romerodsl.extraction import generate_synthetic_training_records

records = generate_synthetic_training_records("advanced", count=100, seed=20261020)
```

Build a compact synthetic-only SFT smoke set and its distribution report:

```python
from pathlib import Path
from romerodsl.finetune import build_complexity_balanced_sft

build_complexity_balanced_sft(
    Path("build/phase4_synthetic_sft/romerodsl_synthetic.jsonl"),
    seed=20261020,
    counts={"basic": 20, "intermediate": 20, "advanced": 20, "expert": 20},
)
```

The summary reports room counts, shape types, branch counts, cycle counts, height features, progression types, prompt lengths, assistant output lengths, duplicate rejections, and provenance counts for `synthetic`, `extracted`, and `hand-authored` records. The checked Phase 4 smoke report is `reports/phase4_synthetic_curriculum_smoke_summary.json`; it contains eight valid records, two per tier, with zero duplicate-layout rejections.

The Gemma-light builder can add a synthetic curriculum to the extracted WAD-derived selection without relabeling it as extracted:

```bash
python scripts/build_gemma_light_sft.py \
  --source build/v08_sft/romerodsl_sft_v08_full.jsonl \
  --output build/phase4_gemma_sft/romerodsl_sft_phase4.jsonl \
  --synthetic-per-tier 100 \
  --synthetic-seed 20261020
```

Use `--synthetic-count basic=50 --synthetic-count expert=150` to deliberately overweight a tier. The output summary records tier and provenance distributions. Synthetic curriculum data is a controlled augmentation, not evidence that the model learned those layouts from human-authored WADs; keep that distinction in training and evaluation reports.
## Phase 5 leakage-resistant splitting and token safety

Phase 5 replaces record-level random splitting with deterministic design-family splitting. Records are grouped by source WAD/map identity, synthetic generator seed or design family, and hand-authored family. Prompt variants from one source design therefore stay together. An optional `--test-fraction` holds out additional complete design groups for testing; `split_group_report()` fails if any group appears in multiple partitions.

The Qwen trainer now tokenizes every chat record before loading the model and writes `tokenization_diagnostics.json` containing prompt, target, and total sequence lengths, overlength examples, discarded supervised target tokens, discarded-target fraction, and truncated critical JSON completion counts. The default `--overlength-policy reject` refuses overlength records. `--overlength-policy preserve_target` may trim only prompt-prefix tokens when the complete assistant JSON target still fits; targets are never silently truncated.

```bash
python -m romerodsl train-qwen-lora \
  --sft-jsonl build/v08_sft/romerodsl_sft_v08_full.jsonl \
  --output runs/qwen2p5_coder_1p5b_v08_grouped \
  --max-length 2400 \
  --validation-fraction 0.08 \
  --test-fraction 0.02 \
  --overlength-policy reject \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --epochs 1
```

A validation score from this split measures held-out source designs, not alternate phrasings of designs already seen during training. Do not increase context length to hide truncation without checking 12GB GPU memory and the diagnostic report first.

A real diagnostic against `Qwen/Qwen2.5-Coder-1.5B-Instruct` on all 23,792 v0.8 records found p50/p95/max total lengths of `1,629 / 1,928 / 1,935` tokens. At `--max-length 2400`, zero examples were overlength, zero supervised target tokens were discarded, and zero critical JSON completions were truncated. The reproducible report is `reports/phase5_qwen_tokenization_report.json`.


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
11. v0.11 / Phase 3: topology-preserving graph-cache extraction, explicit extracted/synthetic/hand-authored provenance, source-versus-DSL topology audit, and honest rejection of unsupported source maps. Exact linedef geometry remains a future source-WAD parsing task.
12. v0.12 / Phase 4: deterministic complexity-balanced synthetic curriculum with exact prompt/target room-count alignment, tiered topology, independent architectural/combat/progression variation, duplicate-layout rejection, and SFT distribution reporting. Synthetic examples remain separately labeled from extracted WAD examples.
13. v0.13 / Phase 5: leakage-resistant design-family train/validation/test splitting, source/synthetic/hand-authored group keys, token-length diagnostics, critical JSON truncation detection, and explicit reject/preserve-target policies. No GPU training is started until the diagnostics are clean.
14. Next: evaluate held-out source-design families with the no-repair suite before any new fine-tune.

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
