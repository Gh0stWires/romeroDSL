from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.finetune import build_sft_jsonl
from romerodsl.qwen_lora import SFTTrainConfig, load_sft_records, split_records, training_run_plan
from tests.test_finetune import progression_sample
from romerodsl.extraction import build_dsl_training_cache


class QwenLoraTests(unittest.TestCase):
    def test_split_records_is_deterministic_and_keeps_validation_records(self) -> None:
        records = [{"metadata": {"source_sample": f"sample_{index}"}} for index in range(20)]

        first = split_records(records, validation_fraction=0.2, seed=7)
        second = split_records(records, validation_fraction=0.2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first["train"]), 16)
        self.assertEqual(len(first["validation"]), 4)
        self.assertEqual(
            {record["metadata"]["source_sample"] for record in first["train"]}
            & {record["metadata"]["source_sample"] for record in first["validation"]},
            set(),
        )

    def test_load_sft_records_reads_validated_jsonl_records(self) -> None:
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
            sft_path = root / "romerodsl_sft_v07.jsonl"
            build_sft_jsonl(training_cache, sft_path)

            records = load_sft_records(sft_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["schema"], "romerodsl_sft_v0.7")
        self.assertEqual(records[0]["messages"][-1]["role"], "assistant")

    def test_training_run_plan_names_qwen_and_output_paths(self) -> None:
        config = SFTTrainConfig(
            sft_jsonl=Path("build/v07_sft/romerodsl_sft_v07.jsonl"),
            output_dir=Path("runs/qwen_v07_lora"),
            max_steps=3,
        )

        plan = training_run_plan(config)

        self.assertEqual(plan["base_model"], "Qwen/Qwen2.5-Coder-1.5B-Instruct")
        self.assertEqual(plan["output_dir"], "runs/qwen_v07_lora")
        self.assertEqual(plan["max_steps"], 3)
        self.assertIn("adapter", plan["artifacts"])


if __name__ == "__main__":
    unittest.main()
