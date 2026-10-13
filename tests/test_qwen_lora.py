from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.finetune import build_sft_jsonl
from romerodsl.qwen_lora import (
    SFTTrainConfig,
    build_tokenization_diagnostics,
    load_sft_records,
    split_group_report,
    split_records,
    training_run_plan,
)
from tests.test_finetune import progression_sample
from romerodsl.extraction import build_dsl_training_cache


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "|".join(message["role"] + ":" + message["content"] for message in messages)
        return text + ("|assistant:" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        class Encoded:
            input_ids = [[index for index, _ in enumerate(text)]]

        return Encoded()


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

    def test_split_records_keeps_all_prompt_variants_together(self) -> None:
        records = []
        for variant in range(3):
            records.append(
                {
                    "metadata": {
                        "source_wad": "packs/alpha.wad",
                        "map": "MAP01",
                        "labels": {"dataset_class": "extracted"},
                        "prompt_variant_index": variant,
                    }
                }
            )
        for seed in (11, 12):
            records.append(
                {
                    "metadata": {
                        "labels": {
                            "dataset_class": "synthetic",
                            "generator_seed": seed,
                            "design_family": "curriculum_family_a",
                        }
                    }
                }
            )
        records.append(
            {
                "metadata": {
                    "labels": {"dataset_class": "hand-authored", "design_family": "golden_route"}
                }
            }
        )

        split = split_records(records, validation_fraction=0.5, seed=17)
        locations = {}
        for partition, values in split.items():
            for record in values:
                metadata = record["metadata"]
                if metadata.get("source_wad"):
                    key = (metadata["source_wad"], metadata["map"])
                elif metadata.get("labels", {}).get("dataset_class") == "synthetic":
                    key = ("synthetic", metadata["labels"]["design_family"], metadata["labels"]["generator_seed"])
                else:
                    key = ("hand-authored", metadata["labels"]["design_family"])
                locations.setdefault(key, set()).add(partition)
        self.assertTrue(all(len(partitions) == 1 for partitions in locations.values()))

    def test_optional_test_partition_also_holds_out_whole_design_groups(self) -> None:
        records = [
            {"metadata": {"source_wad": f"wad_{index}.wad", "map": "MAP01"}}
            for index in range(10)
        ]
        split = split_records(records, validation_fraction=0.2, test_fraction=0.2, seed=9)
        self.assertIn("test", split)
        self.assertEqual(sum(len(values) for values in split.values()), len(records))
        report = split_group_report(split)
        self.assertEqual(report["cross_partition_groups"], [])
        self.assertEqual(report["partition_group_counts"]["test"], len(split["test"]))

    def test_tokenization_diagnostics_identify_discarded_json_targets(self) -> None:
        records = [
            {
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "short"},
                    {"role": "assistant", "content": '{"spaces": []}'},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "S"},
                    {"role": "user", "content": "long prompt " * 10},
                    {"role": "assistant", "content": '{"spaces": ["' + ("x" * 80) + '"]}'},
                ]
            },
        ]

        diagnostics = build_tokenization_diagnostics(records, _FakeTokenizer(), max_length=50)

        self.assertEqual(diagnostics["records"], 2)
        self.assertEqual(diagnostics["overlength_examples"], 1)
        self.assertGreater(diagnostics["supervised_target_tokens_discarded"], 0)
        self.assertEqual(diagnostics["truncated_critical_json_completions"], 1)
        self.assertGreater(diagnostics["fraction_of_supervised_target_tokens_discarded"], 0.0)

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
        self.assertEqual(plan["split_strategy"], "grouped_design_family")
        self.assertEqual(plan["overlength_policy"], "reject")
        self.assertIn("adapter", plan["artifacts"])


if __name__ == "__main__":
    unittest.main()
