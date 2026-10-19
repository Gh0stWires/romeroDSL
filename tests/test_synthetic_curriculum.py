import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.compiler import compile_to_plan, compile_to_wad
from romerodsl.extraction import (
    SYNTHETIC_TIERS,
    generate_synthetic_dsl,
    generate_synthetic_training_records,
    structural_layout_signature,
)
from romerodsl.finetune import build_complexity_balanced_sft, validate_sft_jsonl
from romerodsl.metrics import structural_metrics
from romerodsl.schema import validate_document


class SyntheticCurriculumTests(unittest.TestCase):
    def test_each_tier_is_valid_compilable_and_hits_complexity_bounds(self) -> None:
        expected_bounds = {
            "basic": (3, 5),
            "intermediate": (6, 9),
            "advanced": (10, 15),
            "expert": (16, 24),
        }
        for tier, (minimum, maximum) in expected_bounds.items():
            with self.subTest(tier=tier):
                document = generate_synthetic_dsl(tier, seed=700 + minimum)
                report = validate_document(document)
                self.assertTrue(report.valid, report.errors)
                self.assertGreaterEqual(len(document["spaces"]), minimum)
                self.assertLessEqual(len(document["spaces"]), maximum)
                compile_to_plan(document)
                metrics = structural_metrics(document)
                self.assertEqual(metrics["space_count"], len(document["spaces"]))
                self.assertEqual(document["metadata"]["dataset_class"], "synthetic")
                self.assertEqual(document["metadata"]["complexity_tier"], tier)

    def test_curriculum_adds_required_structural_signals(self) -> None:
        intermediate = structural_metrics(generate_synthetic_dsl("intermediate", seed=802))
        advanced = structural_metrics(generate_synthetic_dsl("advanced", seed=803))
        expert = structural_metrics(generate_synthetic_dsl("expert", seed=804))
        self.assertGreaterEqual(intermediate["unique_room_shape_count"], 2)
        self.assertTrue(intermediate["height"]["varied"])
        self.assertGreaterEqual(intermediate["graph"]["branch_nodes"], 1)
        self.assertGreaterEqual(advanced["graph"]["cycle_count"], 1)
        self.assertGreaterEqual(advanced["room_role_counts"].get("secret_reward", 0), 1)
        self.assertGreaterEqual(expert["graph"]["branch_nodes"], 2)
        self.assertGreaterEqual(expert["graph"]["cycle_count"], 2)

    def test_optional_branch_lock_reuses_existing_branch_edge(self) -> None:
        document = generate_synthetic_dsl("expert", seed=804)
        report = validate_document(document)
        self.assertTrue(report.valid, report.errors)
        optional_gates = [
            gate for gate in document["progression"]["gates"]
            if not gate.get("required_for_exit", True)
        ]
        self.assertEqual(len(optional_gates), 1)
        optional = optional_gates[0]
        matching = [
            connection for connection in document["connections"]
            if connection["id"] == optional["id"]
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["type"], "locked_door")
        self.assertEqual(matching[0]["key"], optional["key"])
        self.assertFalse(matching[0].get("required_for_exit", True))
        self.assertEqual(
            [key["color"] for key in document["progression"]["keys"]].count(optional["key"]),
            1,
        )
        compile_to_plan(document)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = compile_to_wad(document, Path(temp_dir) / "optional_branch.wad")
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["locked_door_colors"], {"blue": 1, "red": 1, "yellow": 1})
        self.assertTrue(report["key_lock_pairs_valid"])

    def test_prompts_truthfully_describe_exact_authored_room_count(self) -> None:
        records = generate_synthetic_training_records("advanced", count=4, seed=901)
        self.assertEqual(len(records), 4)
        for record in records:
            document = record["dsl"]
            prompt = record["prompt"]
            room_count = len(document["spaces"])
            self.assertIn(str(room_count), prompt)
            self.assertEqual(record["labels"]["dataset_class"], "synthetic")
            self.assertEqual(record["labels"]["complexity_tier"], "advanced")
            self.assertEqual(record["features"]["room_count"], room_count)

    def test_generated_records_have_no_duplicate_layout_signatures(self) -> None:
        records = generate_synthetic_training_records("expert", count=8, seed=1001)
        signatures = [structural_layout_signature(record["dsl"]) for record in records]
        self.assertEqual(len(signatures), len(set(signatures)))

    def test_balanced_sft_builder_reports_provenance_and_valid_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "phase4_sft.jsonl"
            summary = build_complexity_balanced_sft(
                output,
                seed=1100,
                counts={tier: 2 for tier in SYNTHETIC_TIERS},
            )
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["records"], 8)
            self.assertEqual(
                summary["provenance_counts"],
                {"synthetic": 8, "extracted": 0, "hand-authored": 0},
            )
            self.assertEqual(summary["tier_counts"], {tier: 2 for tier in SYNTHETIC_TIERS})
            validation = validate_sft_jsonl(output)
            self.assertTrue(validation["valid"], validation["errors"])
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all(row["metadata"]["labels"]["dataset_class"] == "synthetic" for row in rows))
            self.assertTrue(all(row["metadata"]["prompt_length"] > 0 for row in rows))
            self.assertTrue(all(row["metadata"]["assistant_output_length"] > 0 for row in rows))


if __name__ == "__main__":
    unittest.main()
