# romeroDSL

A high-level semantic DSL for Doom map generation.

`romeroDSL` is the new graph-first path for RomeroGAN: instead of asking a model to emit raster cells or linedefs directly, it asks a model to author a structured Doom design contract that can be validated, compiled into geometry, exported to WAD/UDMF, and playtested.

The boundary is deliberate:

- The **model** authors design: spaces, progression, keys, locks, height topology, combat beats, resources, secrets, and theme.
- The **compiler** authors mechanics: vertices, sectors, linedefs, tags, sidedefs, UDMF formatting, and texture application.
- The **validator** rejects missing or fake progression instead of silently repairing it.

This keeps room topology, height/elevation, and key progression as first-class learned concepts while leaving room to reuse the existing RomeroGAN raster/graph experiments later as local geometry/detail generators.

## Current status

This is v0.1: a GitHub-ready foundation containing:

- a canonical JSON form of the DSL
- an example Doom II techbase map contract
- a Python validator for references and progression invariants
- a small abstract geometry-plan compiler stub
- unit tests

It does **not** export WADs yet. That is the next milestone.

## Install for development

```bash
python -m pip install -e .
```

No runtime dependencies are required for v0.1.

## Validate the example

```bash
python -m romerodsl validate examples/blue_lock_processing.json
```

Expected result:

```text
valid: examples/blue_lock_processing.json
```

## Emit an abstract geometry plan

```bash
python -m romerodsl compile examples/blue_lock_processing.json --output build/blue_lock_processing.plan.json
```

The emitted plan is intentionally not a WAD. It is the first bridge between semantic authorship and later UDMF/WAD export.

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
- validation requirements

The compiler may make valid geometry from those decisions, but it should not invent missing progression.

## Roadmap

1. v0.1: DSL schema, examples, validator, abstract plan compiler.
2. v0.2: WAD/UDMF compiler for a constrained subset: rooms, hallways, doors, blue/red/yellow locks, starts, exits, monsters, items.
3. v0.3: Extraction from real WAD graph caches into canonical DSL training pairs.
4. v0.4: LoRA fine-tune target: `prompt -> romeroDSL`.
5. v0.5: Validator-guided generation loop and playability reports.

## Repository layout

```text
examples/                  Example DSL documents
src/romerodsl/             Python package
  schema.py                DSL validation rules
  compiler.py              Abstract geometry-plan compiler
  cli.py                   Command line interface
tests/                     Unit tests
```
