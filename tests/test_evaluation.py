from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.evaluation import (
    evaluate_generated_dsl_intent,
    evaluate_prompt_intent,
    extract_json_object,
    requested_key_colors_from_prompt,
    requested_space_count_from_prompt,
    template_similarity_report,
    v08_prompt_expectations,
    v08_prompt_suite,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "blue_lock_processing.json"


class EvaluationTests(unittest.TestCase):
    def test_extract_json_object_accepts_plain_or_wrapped_json(self) -> None:
        expected = {"dsl_version": "0.1", "title": "Example"}

        self.assertEqual(extract_json_object(json.dumps(expected)), expected)
        self.assertEqual(extract_json_object(f"```json\n{json.dumps(expected)}\n```"), expected)

    def test_v08_prompt_suite_covers_varied_progression_requests(self) -> None:
        prompts = v08_prompt_suite()
        expectations = v08_prompt_expectations()

        self.assertGreaterEqual(len(prompts), 10)
        self.assertEqual(len(expectations), len(prompts))
        self.assertTrue(any("no locked door" in prompt.lower() for prompt in prompts))
        self.assertTrue(any("two keys" in prompt.lower() for prompt in prompts))
        self.assertTrue(any("yellow" in prompt.lower() for prompt in prompts))

    def test_prompt_intent_requires_requested_locks_not_just_valid_wad(self) -> None:
        two_key_result = {
            "dsl_valid": True,
            "wad_valid": True,
            "zandronum_exit_code": 0,
            "key_lock_pairs_valid": True,
            "key_colors": {"blue": 1, "red": 1},
            "locked_door_colors": {},
            "locked_doors": 0,
        }

        report = evaluate_prompt_intent(two_key_result, v08_prompt_expectations()[4])

        self.assertFalse(report["prompt_intent_valid"])
        self.assertIn("locked_doors < 2", report["prompt_intent_errors"])

    def test_prompt_intent_accepts_matching_requested_key_and_lock(self) -> None:
        red_key_result = {
            "dsl_valid": True,
            "wad_valid": True,
            "zandronum_exit_code": 0,
            "key_lock_pairs_valid": True,
            "keys": 1,
            "key_colors": {"red": 1},
            "locked_door_colors": {"red": 1},
            "locked_doors": 1,
        }

        report = evaluate_prompt_intent(red_key_result, v08_prompt_expectations()[0])

        self.assertTrue(report["prompt_intent_valid"])
        self.assertEqual(report["prompt_intent_errors"], [])

    def test_prompt_request_extracts_key_colors_and_room_count(self) -> None:
        prompt = (
            "Create a big eight-room Doom 2 level with two keycard rooms, "
            "a yellow key and a red key, then a red locked final exit."
        )

        self.assertEqual(requested_key_colors_from_prompt(prompt), ["red", "yellow"])
        self.assertEqual(requested_space_count_from_prompt(prompt), 8)

    def test_generated_dsl_intent_reports_missing_requested_key_but_treats_room_count_as_soft(self) -> None:
        prompt = (
            "Create a big eight-room Doom 2 level with two keycard rooms, "
            "a yellow key and a red key. The red key must open the final exit wing."
        )
        dsl = {
            "design_intent": {
                "summary": "A public route plan for the generated map.",
                "progression_style": "yellow_key_then_bfg_then_red_key_then_exit",
            },
            "progression": {
                "start": "start_room",
                "exit": "exit",
                "critical_path": ["start_room", "yellow_key_room", "yellow_gate", "bfg_armory", "red_key_room", "red_gate", "exit"],
                "keys": [{"color": "yellow", "location": "yellow_key_room"}],
                "gates": [
                    {"id": "yellow_gate", "type": "locked_door", "key": "yellow"},
                    {"id": "red_gate", "type": "locked_door", "key": "red"},
                ],
            },
            "spaces": [
                {"id": "start_room"},
                {"id": "combat_hub"},
                {"id": "yellow_key_room"},
                {"id": "bfg_armory"},
                {"id": "red_key_room"},
                {"id": "exit"},
            ],
            "connections": [],
        }

        report = evaluate_generated_dsl_intent(prompt, dsl)

        self.assertEqual(report["requested_key_colors"], ["red", "yellow"])
        self.assertEqual(report["generated_key_colors"], ["yellow"])
        self.assertEqual(report["generated_gate_colors"], ["red", "yellow"])
        self.assertEqual(report["missing_requested_keys"], ["red"])
        self.assertEqual(report["missing_requested_gates"], [])
        self.assertEqual(report["requested_space_count"], 8)
        self.assertEqual(report["space_count"], 6)
        self.assertFalse(report["space_count_exact_match"])
        self.assertTrue(report["space_count_pass"])
        self.assertEqual(report["space_count_note"], "soft_pass_within_tolerance")
        self.assertEqual(report["public_design_notes"]["summary"], "A public route plan for the generated map.")
        self.assertFalse(report["prompt_contract_valid"])

    def test_template_similarity_flags_copied_seed_title_and_space_ids(self) -> None:
        example = json.loads(EXAMPLE.read_text(encoding="utf-8"))

        report = template_similarity_report(example, example)

        self.assertTrue(report["copied_source_title"])
        self.assertEqual(report["space_id_overlap"], 1.0)
        self.assertEqual(report["connection_id_overlap"], 1.0)
        self.assertEqual(report["template_similarity"], "high")
    def test_template_similarity_flags_same_structure_with_different_ids(self) -> None:
        example = json.loads(EXAMPLE.read_text(encoding="utf-8"))
        generated = {
            "title": "Small Techbase Key Route",
            "theme": {"primary": "techbase"},
            "progression": {
                "keys": [{"color": "red", "location": "red_key_room"}],
                "gates": [{"id": "red_locked_door", "key": "red"}],
            },
            "spaces": [
                {"id": "start", "role": "player_start", "shape": "rectangle"},
                {"id": "hub", "role": "combat_hub", "shape": "irregular_octagonal"},
                {"id": "key", "role": "key_room", "shape": "rectangle_with_side_alcoves"},
                {"id": "exit", "role": "exit", "shape": "rectangle"},
            ],
            "connections": [
                {"from": "start", "to": "hub"},
                {"from": "hub", "to": "key"},
                {"from": "key", "to": "exit"},
            ],
        }

        report = template_similarity_report(generated, example)

        self.assertEqual(report["space_id_overlap"], 0.0)
        self.assertGreaterEqual(report["template_structural_score"], 0.65)
        self.assertEqual(report["template_similarity"], "high")


if __name__ == "__main__":
    unittest.main()
