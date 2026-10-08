from __future__ import annotations

import json
import unittest
from pathlib import Path

from romerodsl.evaluation import (
    evaluate_prompt_intent,
    extract_json_object,
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

    def test_template_similarity_flags_copied_seed_title_and_space_ids(self) -> None:
        example = json.loads(EXAMPLE.read_text(encoding="utf-8"))

        report = template_similarity_report(example, example)

        self.assertTrue(report["copied_source_title"])
        self.assertEqual(report["space_id_overlap"], 1.0)
        self.assertEqual(report["connection_id_overlap"], 1.0)
        self.assertEqual(report["template_similarity"], "high")


if __name__ == "__main__":
    unittest.main()
