from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.cli import main
from romerodsl.finetune import build_sft_jsonl, sft_record_from_training_record, validate_sft_jsonl
from romerodsl.schema import validate_document
from tests.test_training_pairs import progression_sample
from romerodsl.extraction import build_dsl_training_cache, training_record_from_graph_sample


class FinetuneTests(unittest.TestCase):
    def test_sft_record_contains_prompt_and_valid_canonical_dsl_output(self) -> None:
        training_record = training_record_from_graph_sample(
            progression_sample(), sample_path="samples/example_MAP06.json"
        )

        sft_record = sft_record_from_training_record(training_record)
        assistant_json = json.loads(sft_record["messages"][-1]["content"])
        report = validate_document(assistant_json)

        self.assertEqual(sft_record["schema"], "romerodsl_sft_v0.7")
        self.assertEqual(sft_record["messages"][0]["role"], "system")
        self.assertEqual(sft_record["messages"][1]["role"], "user")
        self.assertIn("blue key before a blue locked door", sft_record["messages"][1]["content"])
        self.assertTrue(report.valid, report.errors)
        self.assertEqual(sft_record["metadata"]["source_sample"], "samples/example_MAP06.json")
        self.assertNotIn("metadata", assistant_json)

    def test_sft_output_omits_bulky_source_provenance_metadata(self) -> None:
        sample = progression_sample()
        sample["heights"] = {"levels": list(range(-128, 4096, 16))}
        training_record = training_record_from_graph_sample(
            sample, sample_path="samples/example_MAP06.json"
        )

        sft_record = sft_record_from_training_record(training_record)
        assistant_json = json.loads(sft_record["messages"][-1]["content"])
        report = validate_document(assistant_json)

        self.assertTrue(report.valid, report.errors)
        self.assertNotIn("metadata", assistant_json)
        self.assertNotIn("source_height_levels", sft_record["messages"][-1]["content"])
        self.assertLess(len(sft_record["messages"][-1]["content"]), 6000)

    def test_build_and_validate_sft_jsonl_from_training_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            (samples_dir / "example_MAP06.json").write_text(
                json.dumps(progression_sample()) + "\n", encoding="utf-8"
            )
            (graph_cache / "manifest.jsonl").write_text(
                json.dumps({"sample": "samples/example_MAP06.json"}) + "\n", encoding="utf-8"
            )
            training_cache = root / "dsl_training_v06"
            build_dsl_training_cache(graph_cache, training_cache, strict_only=True)
            output = root / "romerodsl_sft_v07.jsonl"

            build_summary = build_sft_jsonl(training_cache, output)
            validate_summary = validate_sft_jsonl(output)

        self.assertEqual(build_summary["records"], 1)
        self.assertEqual(validate_summary["records"], 1)
        self.assertEqual(validate_summary["invalid_json"], 0)
        self.assertEqual(validate_summary["invalid_dsl"], 0)

    def test_cli_build_and_validate_sft_jsonl_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            (samples_dir / "example_MAP06.json").write_text(
                json.dumps(progression_sample()) + "\n", encoding="utf-8"
            )
            (graph_cache / "manifest.jsonl").write_text(
                json.dumps({"sample": "samples/example_MAP06.json"}) + "\n", encoding="utf-8"
            )
            training_cache = root / "dsl_training_v06"
            build_dsl_training_cache(graph_cache, training_cache, strict_only=True)
            output = root / "romerodsl_sft_v07.jsonl"

            build_exit = main(
                ["build-sft-jsonl", "--training-cache", str(training_cache), "--output", str(output)]
            )
            validate_exit = main(["validate-sft-jsonl", str(output)])
            output_exists = output.exists()

        self.assertEqual(build_exit, 0)
        self.assertEqual(validate_exit, 0)
        self.assertTrue(output_exists)


if __name__ == "__main__":
    unittest.main()
