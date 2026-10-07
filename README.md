# romeroDSL

A high-level semantic DSL for Doom map generation.

`romeroDSL` is the new graph-first path for RomeroGAN: instead of asking a model to emit raster cells or linedefs directly, it asks a model to author a structured Doom design contract that can be validated, compiled into geometry, exported to WAD/UDMF, and playtested.

The boundary is deliberate:

- The **model** authors design: spaces, progression, keys, locks, height topology, combat beats, resources, secrets, and theme.
- The **compiler** authors mechanics: vertices, sectors, linedefs, tags, sidedefs, UDMF formatting, and texture application.
- The **validator** rejects missing or fake progression instead of silently repairing it.

This keeps room topology, height/elevation, and key progression as first-class learned concepts while leaving room to reuse the existing RomeroGAN raster/graph experiments later as local geometry/detail generators.

## Current status

This is v0.3: a GitHub-ready foundation containing:

- a canonical JSON form of the DSL
- an example Doom II techbase map contract
- a Python validator for references and progression invariants
- a debug cell-layout compiler
- a sector-primitive geometry compiler that turns semantic rooms/connections into UDMF sectors
- a minimal dependency-free UDMF/PWAD exporter
- unit tests

The v0.3 WAD exporter uses real room sectors, corridor sectors, tagged door sectors, and intra-room height-feature sectors for pillars, platforms, and pits. Geometry is still simple and has not been engine-playtested, but the path now proves semantic DSL authorship can survive into much cleaner Doom geometry than the cell-grid prototype.

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

## Export a first playable UDMF PWAD

```bash
python -m romerodsl export-wad examples/blue_lock_processing.json --output build/blue_lock_processing.wad
```

The exported WAD should contain one player start, one exit, a blue key, at least one blue-locked door, monsters, items, real room/corridor/door sectors, and sector height variation from the DSL. It has not been engine-playtested yet.

## Emit an abstract geometry plan

```bash
python -m romerodsl compile examples/blue_lock_processing.json --output build/blue_lock_processing.plan.json
```

The emitted plan is a debug bridge between semantic authorship and UDMF/WAD export.

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
3. v0.3: Replace the cell-block compiler with sector-primitive geometry: room sectors, corridor sectors, tagged door sectors, and intra-room height-feature sectors.
4. v0.4: Improve geometry authorship: polygonal/irregular rooms, doorway cutouts, stairs, lifts, safer monster/item placement, and engine playtesting.
5. v0.5: Extraction from real WAD graph caches into canonical DSL training pairs.
6. v0.6: LoRA fine-tune target: `prompt -> romeroDSL`.
7. v0.7: Validator-guided generation loop and playability reports.

## Repository layout

```text
examples/                  Example DSL documents
src/romerodsl/             Python package
  schema.py                DSL validation rules
  compiler.py              Abstract geometry-plan compiler
  cli.py                   Command line interface
tests/                     Unit tests
```
