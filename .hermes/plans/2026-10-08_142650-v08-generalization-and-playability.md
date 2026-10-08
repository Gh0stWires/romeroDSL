# romeroDSL v0.8 Generalization and Playability Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Make Qwen-generated romeroDSL maps less template-like and make validation prove real playable progression, not just object counts.

**Architecture:** Split the next work into three tracks: stronger validation, more diverse data/evaluation, and compiler layout variation. The model should author the semantic contract; the compiler should preserve that contract without silently inventing or flattening it; validators should verify concrete UDMF mechanics and route-gated progression.

**Tech Stack:** Python stdlib/unittest, romeroDSL schema/compiler/WAD validator, Qwen LoRA adapter, Zandronum parser smoke with Doom II IWAD.

---

## Current Context

- The slim Qwen adapter can emit parseable/valid romeroDSL JSON.
- The compiler can export a Zandronum-loadable WAD.
- A real exporter bug was found: red locks were using the wrong ZDoom lock number. Fixed mapping should be red=1, blue=2, yellow=3.
- The latest generated DSL still looked template-like: title `Extracted TROG MAP01 DSL Seed`, room ids like `start_room`, `combat_hub`, `red_key_room`, `exit_room`.
- Current validator counted keys and locked doors but did not fully prove playable keyed progression from the player's perspective.

---

## Proposed v0.8 Acceptance Criteria

A v0.8 sample should only count as success if all of these pass:

1. JSON parses and `validate_document()` passes.
2. WAD compiles and `validate_wad()` passes.
3. UDMF contains concrete key thing types matching lock numbers:
   - red keycard `type=13` -> `locknumber=1`
   - blue keycard `type=5` -> `locknumber=2`
   - yellow keycard `type=6` -> `locknumber=3`
4. The key is reachable from the player start before crossing its matching locked door.
5. The exit is not reachable from the player start unless the matching locked door is crossed/opened.
6. The locked door is a real partition edge between route components, not a decorative door/pillar inside a room.
7. The generated DSL is not an obvious copy of `examples/blue_lock_processing.json` or a single cached seed.
8. At least 10 evaluation prompts are run, including prompts not directly mirrored by training examples.
9. At least 3 generated WADs are provided for user visual playtest, with honest pass/fail reports.

---

## Task 1: Strengthen WAD Progression Validation

**Objective:** Make `validate_wad()` prove concrete keyed-door mechanics and route gating.

**Files:**
- Modify: `src/romerodsl/wad.py`
- Test: `tests/test_schema.py` or new `tests/test_wad_progression.py`

**Steps:**
1. Add tests that fail if a red keycard is paired with a yellow lock number.
2. Add helper logic to parse UDMF things/linedefs/sectors into gameplay facts:
   - key things by color/type
   - locked linedefs by lock number/color
   - door sectors by tag/arg0
3. Add a report section such as:
   - `key_lock_pairs_valid`
   - `key_lock_pair_errors`
   - `locked_door_colors`
   - `key_colors`
4. Run:
   ```bash
   PYTHONPATH=src python -m unittest discover -s tests
   ```

---

## Task 2: Add Route-Gated Progression Checks at DSL/Compiler Level

**Objective:** Verify that the semantic graph requires the key before the locked door and requires the door before exit.

**Files:**
- Modify: `src/romerodsl/schema.py`
- Modify: `src/romerodsl/compiler.py`
- Test: new/updated tests under `tests/`

**Steps:**
1. Build a graph from `spaces` and `connections`.
2. Mark locked connections as blocked unless the required key has been acquired.
3. Validate:
   - start can reach the key without crossing that key's lock
   - exit cannot be reached without crossing the required lock
   - exit can be reached after acquiring the key
4. Add tests for false positives:
   - key exists but is behind its own door
   - door exists but is not on exit route
   - exit is reachable without door
5. Run full tests.

---

## Task 3: Detect Template Leakage / Seed Copying

**Objective:** Stop calling template-shaped outputs a model-quality win.

**Files:**
- Modify: `src/romerodsl/finetune.py` or add `src/romerodsl/evaluation.py`
- Test: new `tests/test_evaluation.py`
- Script: update `scripts/evaluate_qwen_v07_adapter.py` or create `scripts/evaluate_qwen_v08_suite.py`

**Steps:**
1. Load `examples/blue_lock_processing.json` and generated DSL.
2. Compare structural fingerprints:
   - title/source family strings
   - space IDs
   - space count
   - connection graph shape
   - key/gate color sequence
   - combat/resource counts
3. Report a similarity score, not as a hard fail at first.
4. Flag outputs with copied titles like `Extracted ... Seed`.
5. Add evaluator output fields:
   - `template_similarity`
   - `copied_source_title`
   - `space_id_overlap`

---

## Task 4: Build a Multi-Prompt Eval Suite

**Objective:** Evaluate generalization instead of one prompt.

**Files:**
- Create: `scripts/evaluate_qwen_v08_suite.py`
- Create: `build/v08_eval/` outputs
- Test: lightweight unit tests for prompt manifest parsing if needed

**Prompt set should include:**
1. Tiny techbase red key door exit gate.
2. Small hell map with blue key and trap room.
3. UAC storage depot with yellow key and optional secret.
4. Compact map with two combat rooms and one locked exit wing.
5. Map with key visible early but unreachable until loopback.
6. Map with no locked door requested, to ensure the model does not always add one.
7. Map with two keys, blue then red.
8. Vertical map with stairs, pit, and raised exit platform.
9. Low-monster exploration map.
10. High-pressure arena with resources before lock.

**Report per prompt:**
- DSL valid
- WAD valid
- key/lock mechanics valid
- route-gated progression valid
- template similarity
- Zandronum parser exit code
- WAD path

---

## Task 5: Improve Training Targets for Variation

**Objective:** Reduce overfitting to extraction seed names and template room IDs.

**Files:**
- Modify: `src/romerodsl/finetune.py`
- Possibly modify: `src/romerodsl/extraction.py`
- Test: `tests/test_finetune.py`

**Steps:**
1. Remove or normalize source-derived titles from assistant targets.
2. Replace seed-like room IDs with varied semantic names where safe.
3. Add prompt augmentation so the same DSL structure can be asked in multiple human-style ways.
4. Consider adding negative/no-lock examples if the current strict-only dataset teaches every prompt to include a key door.
5. Rebuild SFT JSONL and inspect fingerprints before retraining.

---

## Task 6: Add Compiler Layout Variation Without Breaking Semantics

**Objective:** Make maps from different DSLs produce visibly different layouts while preserving route correctness.

**Files:**
- Modify: `src/romerodsl/compiler.py`
- Modify: `src/romerodsl/geometry.py`
- Tests: existing geometry tests plus new variation tests

**Steps:**
1. Let the DSL specify or imply layout motifs:
   - hub-and-spoke
   - linear gauntlet
   - loopback
   - branching optional room
   - vertical stair route
2. Add deterministic variation from DSL content, not random hidden repair.
3. Validate that key/door/exit route remains correct after layout variation.
4. Export at least three generated WADs for visual comparison.

---

## Task 7: Retrain Only After Validation/Data Fixes

**Objective:** Avoid wasting GPU time training against weak targets.

**Steps:**
1. Complete Tasks 1-5 first.
2. Rebuild SFT JSONL.
3. Check token length stats.
4. Run one-step smoke train.
5. Train new adapter under:
   ```text
   runs/qwen2p5_coder_1p5b_v08
   ```
6. Evaluate with the v0.8 suite, not a single prompt.

---

## Verification Commands

```bash
cd C:/Users/gh0st/Documents/RomeroGan/romeroDSL
PYTHONPATH=src python -m unittest discover -s tests
PYTHONPATH=src python -m romerodsl validate examples/blue_lock_processing.json
PYTHONPATH=src python scripts/evaluate_qwen_v08_suite.py
```

Zandronum parser smoke should use:

```bash
"C:/Users/gh0st/AppData/Local/Zandronum/zandronum.exe" -iwad "C:/Users/gh0st/Downloads/DOOM2.WAD" -file "<generated.wad>" +map MAP01 +quit
```

Do not launch interactive playtests unless the user explicitly asks. Provide WAD paths and commands for user visual testing.

---

## Risks and Tradeoffs

- Stronger validators may reveal that many existing training examples are weak. That is good: it prevents fake progress.
- Multi-prompt evaluation will likely make v0.7 look worse than the single smoke result. That is also good: it tells us what to fix.
- Compiler variation can accidentally become another hidden authoring layer. Keep it deterministic and driven by model-authored DSL fields.
- A new retrain before fixing the data/evaluator will mostly teach the model the same template behavior.

---

## Recommended Next Step

Start with Task 1 and Task 2. The next milestone should be a truthful validator that can say: `this red key opens this red door, the key is reachable first, and the exit is actually gated behind that door.`

Only after that should we retrain or claim model improvement.
