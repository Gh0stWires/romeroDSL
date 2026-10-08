# No-Repair romeroDSL Model Improvement Plan

> **For Hermes:** This is a planning document only. Do not implement from this file until Sam asks to execute.

**Goal:** Improve the Qwen `prompt -> romeroDSL` adapter so raw model outputs satisfy the DSL, compile to valid WADs, and match prompt intent without post-generation repair.

**Hard constraint:** No repair. Invalid outputs must fail. The compiler may deterministically compile valid model-authored DSL, but it must not invent missing keys, locks, exits, gates, rooms, route order, or prompt intent after generation.

**Current evidence:** v0.8 multi-prompt eval produced `10/10` JSON-valid responses, but only `2/10` DSL-valid/WAD-valid responses, only `1/10` prompt-intent-valid response, and `0/10` clean non-template successes. `9/10` showed high template similarity. The model learned JSON shape, not robust Doom-map design.

---

## Diagnosis

### 1. The current target is too free-form for a 998-record LoRA

The model has to emit a large nested JSON document with:

- top-level metadata-like fields such as `title`
- open-ended `materials`
- arbitrary `space.id` strings
- arbitrary `connection.id` strings
- `progression.keys`, `progression.gates`, and `connections` that must agree exactly
- required semantic graph invariants

The eval errors show the model understands the *style* but not the constraints:

- `progression.exit references unknown space ''`
- `connections[x].to references unknown space None`
- `progression.keys[0] must be an object`
- `expected exactly one exit space, found 2`
- invented material roles like `locked_door`, `door_track_pin`, `exit_wall`, `game_over_sign`

This is not random. It means the model is mixing natural Doom words into schema keys.

### 2. The dataset likely teaches source/template leakage

The generated passing outputs often keep source-like titles such as extracted seed titles and repeated room ids:

- `start_room`
- `combat_hub`
- `red_key_room`
- `exit_room`

Those are fine as internal examples but bad as the only distribution. The model is learning examples as forms, not design rules.

### 3. There are not enough negative or contrastive examples

The model needs to learn that:

- no-lock prompts should produce no keys and no gates
- one-key prompts should produce exactly one matching key/gate pair
- two-key prompts require two gates, not just two keys
- key color and lock color must match
- exit must be behind the requested final gate

The current training likely over-represents strict key/lock examples and under-represents no-lock, one-lock, and multi-lock contrastive variations.

### 4. The compiler currently masks some design weakness

The compiler can compile valid DSL into a WAD, but it cannot prove the model learned interesting topology. A small number of valid outputs can still be template-like.

No-repair means the compiler should remain a compiler, not a fixer. It can lay out valid geometry from the DSL, but cannot decide missing semantic intent.

---

## Strategic Direction

Do **not** train longer on the same target.

The next improvement should be a stricter, smaller, more learnable target:

```text
prompt
-> canonical no-repair authoring DSL
-> validator rejects invalid raw output
-> compiler emits WAD only after validation
```

The model target should still be romeroDSL, but we should narrow the authoring surface to reduce invalid degrees of freedom.

---

## Phase 1: Define a Canonical Authoring Profile

**Goal:** Make model outputs easier to learn and harder to partially corrupt.

Create a documented `authoring_profile: v0.8_no_repair` or `dsl_profile: canonical_v0.8` with these rules:

1. No source-derived titles.
   - Bad: `Extracted TROG MAP01 DSL Seed`
   - Good: generated clean title from prompt, such as `Red Key Techbase Gate`.

2. Controlled material surface roles only.
   - Allowed roles: `floor`, `ceiling`, `wall`, `upper_wall`, `lower_wall`, `door`, `door_track`, `exit_switch`.
   - Model must not invent roles like `exit_wall`, `locked_door`, `door_track_pin`, or `game_over_sign`.

3. Controlled key/gate object schema.
   - `progression.keys` must always be a list of objects.
   - `progression.gates` must always be a list of objects.
   - No raw strings inside `progression.keys`.

4. Stable semantic roles, variable labels.
   - Encourage roles like `player_start`, `combat_hub`, `key_room`, `connector`, `exit`.
   - Do not require fixed ids like `combat_hub` every time.

5. Explicit progression contract.
   Every keyed prompt output must include:
   - key object
   - matching gate object
   - matching locked connection
   - critical path containing key before gate before exit

6. Explicit no-lock contract.
   No-lock prompts must include:
   - `progression.keys: []`
   - `progression.gates: []`
   - no locked connections

**No repair rule:** If the model fails any of these, validation fails. Do not normalize it into compliance.

---

## Phase 2: Strengthen Dataset Construction Before Retraining

**Goal:** Teach valid variation, not just one seed pattern.

### 2.1 Remove source leakage from targets

Strip or regenerate from assistant targets:

- source WAD titles
- source family identifiers
- extraction-seed wording
- overly repeated room ids where they are artifacts, not semantics

Keep provenance only in SFT metadata, never in assistant JSON.

### 2.2 Add deterministic authored augmentations

For each valid training DSL, create multiple no-repair variants that remain valid before training:

- rename spaces consistently
- change title cleanly
- vary theme labels within supported materials
- vary prompt wording
- vary key colors where the graph supports it
- create no-lock contrast examples from simple connected maps
- create one-key and two-key controlled examples

Important: these are **training data augmentations**, not output repair. Every augmented target must pass `validate_document()` before entering the SFT file.

### 2.3 Add synthetic gold examples

The 998 extracted samples are too narrow. Add a small hand-authored gold set, maybe 50-150 records, covering:

- no key / no locked door
- red only
- blue only
- yellow only
- blue then red
- red then yellow
- key visible but reachable later
- loopback hub route
- vertical route with stairs/pit/platform
- optional secret that is not required for progression

Each gold example should be raw-valid and compile-valid. This is not cheating. It is explicit curriculum for the semantics we care about.

### 2.4 Add prompt-intent labels into eval metadata

The eval already has expectations. Training records should also expose intent labels outside the assistant JSON:

- expected key colors
- expected locked door colors
- min keys
- min locked doors
- no-lock flag
- route style
- theme

Do not put these labels inside the assistant JSON unless they are part of the DSL. Use them for filtering, balancing, and reporting.

---

## Phase 3: Improve SFT Format Without Repair

**Goal:** Reduce output entropy while keeping raw generation accountable.

### Option A: Keep full JSON, but add stricter system prompt

Current system prompt is too broad. Add precise constraints:

- Use only allowed surface roles.
- Every id referenced by progression or connections must exist.
- `progression.keys` and `progression.gates` must be arrays of objects.
- If prompt says no locked door, emit empty keys/gates and no locked connections.
- If prompt says two keys, emit two keys, two gates, and two locked connections.
- Do not include source WAD names, extraction notes, or seed titles.

This is the fastest path, but may still fail because small models are bad at long-form schema consistency.

### Option B: Train a smaller authoring DSL, then compile to full DSL

Example target:

```json
{
  "title": "Red Key Techbase Gate",
  "theme": "techbase",
  "layout_motif": "hub_with_key_branch",
  "progression": [
    {"space": "start", "role": "player_start"},
    {"space": "hub", "role": "combat_hub"},
    {"space": "red_key_room", "role": "key_room", "key": "red"},
    {"gate": "red_locked_door", "key": "red"},
    {"space": "exit", "role": "exit"}
  ],
  "combat": "medium",
  "resources": ["weapon", "ammo", "health"],
  "height_features": ["stairs", "pit"]
}
```

Then a deterministic expander converts this *valid smaller DSL* into the current full romeroDSL.

This does not violate no repair if:

- the smaller DSL is the declared model target,
- it has its own strict validator,
- invalid small DSL is rejected,
- the expander only compiles valid fields and does not invent missing intent.

This is probably the strongest long-term path because it reduces the output burden on the model.

### Recommendation

Do Phase 3A immediately because it is cheap, but plan Phase 3B as the real architecture if full JSON remains brittle.

---

## Phase 4: Upgrade Evaluation Gates

**Goal:** Make success mean what Sam actually cares about.

The multi-prompt eval should report these as separate metrics:

1. `json_valid`
2. `dsl_valid`
3. `prompt_intent_valid`
4. `wad_valid`
5. `zandronum_parser_ok`
6. `template_similarity`
7. `clean_non_template_success`
8. `route_gate_valid`
9. `no_repair_used`

A clean success should require:

```text
json_valid == true
dsl_valid == true
prompt_intent_valid == true
wad_valid == true
zandronum_parser_ok == true
template_similarity != high
no_repair_used == true
```

Do not count generic WAD validity as model success.

---

## Phase 5: Retrain v0.8 Only After Dataset Audit Passes

Before training, run a dataset audit:

- Count no-lock examples.
- Count one-key examples by color.
- Count two-key examples by color order.
- Count examples with real route-gated locked exits.
- Count unique titles.
- Count unique space-id sets.
- Count invalid material roles in targets.
- Count source-title leakage.
- Count assistant outputs over token budget.

Do not train if:

- no-lock examples are nearly absent,
- multi-key examples are nearly absent,
- source-title leakage remains,
- assistant targets contain invalid material roles,
- key/gate/connection consistency is not guaranteed.

Then train:

```text
runs/qwen2p5_coder_1p5b_v08_no_repair
```

Evaluate with the same v0.8 prompt suite. Do not move goalposts.

---

## Phase 6: Success Thresholds

Do not claim the model works until it hits something like:

```text
JSON valid:                 >= 10/10
DSL valid:                  >= 8/10
Prompt intent valid:        >= 7/10
WAD valid:                  >= 7/10
Zandronum parser OK:        >= 7/10
Clean non-template success: >= 5/10
```

For v0.8, a reasonable next milestone is:

```text
DSL valid:                  5/10
Prompt intent valid:        3/10
Clean non-template success: 2/10
```

That would be real progress over the current baseline.

---

## Immediate Next Implementation Steps

1. Add dataset audit script for source-title leakage, invalid material roles, key/gate counts, and prompt-intent distribution.
2. Patch `canonical_sft_dsl()` so it removes/regenerates source-derived titles and enforces the canonical v0.8 authoring profile.
3. Add tests proving SFT targets contain no source-seed titles and no unknown material roles.
4. Create 20-50 hand-authored gold no-repair DSL examples covering no-lock, one-lock, and two-lock cases.
5. Rebuild SFT JSONL.
6. Run the audit.
7. Only then train `qwen2p5_coder_1p5b_v08_no_repair`.

---

## Non-Negotiables

- No post-generation semantic repair.
- No exporter-invented missing keys or locks.
- No counting WAD validity as prompt success.
- No hiding template similarity.
- No retraining on known-bad targets.
- Invalid raw model output fails, visibly.
