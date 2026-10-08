from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.cli import main
from romerodsl.extraction import (
    audit_graph_cache,
    build_dsl_training_cache,
    dsl_from_graph_sample,
    prompt_variants_from_graph_sample,
    training_record_from_graph_sample,
)
from romerodsl.schema import validate_document


def progression_sample() -> dict:
    return {
        "source_wad": "C:/wads/example.wad",
        "family": "example",
        "map": "MAP06",
        "counts": {
            "sectors": 22,
            "sector_connections": 18,
            "doors": 7,
            "locked_doors": 2,
            "keys": 2,
            "player_starts": 1,
            "exit_sectors": 2,
        },
        "theme_guess": "hell",
        "height_levels": [0, 30, 32, 72],
        "things_by_category": {"monsters": 9, "weapons": 2, "keys": 2},
        "keys": [
            {"color": "blue", "sector": 10},
            {"color": "red", "sector": 21},
        ],
        "progression": {
            "start_sectors": [3],
            "exit_sectors": [15, 17],
            "start_to_exit_path": [3, 0, 4, 6, 11, 14, 15],
            "start_reaches_exit": True,
            "exit_reachable_without_locks": False,
            "exit_requires_any_lock": True,
            "colors": {
                "blue": {"keys": 1, "locks": 1, "key_reachable_without_locks": True, "has_key_and_lock": True},
                "red": {"keys": 1, "locks": 1, "key_reachable_without_locks": False, "has_key_and_lock": True},
                "yellow": {"keys": 0, "locks": 0, "key_reachable_without_locks": False, "has_key_and_lock": False},
            },
            "has_keyed_route_signal": True,
        },
    }


def basic_route_sample() -> dict:
    return {
        "source_wad": "C:/wads/basic.wad",
        "family": "basic",
        "map": "MAP01",
        "counts": {
            "sectors": 10,
            "sector_connections": 8,
            "doors": 2,
            "locked_doors": 0,
            "keys": 0,
            "player_starts": 1,
            "exit_sectors": 1,
        },
        "theme_guess": "techbase",
        "height_levels": [0],
        "things_by_category": {"monsters": 3, "weapons": 1, "keys": 0},
        "progression": {
            "start_sectors": [1],
            "exit_sectors": [9],
            "start_to_exit_path": [1, 4, 9],
            "start_reaches_exit": True,
            "exit_reachable_without_locks": True,
            "exit_requires_any_lock": False,
            "colors": {
                "blue": {"keys": 0, "locks": 0, "key_reachable_without_locks": False, "has_key_and_lock": False},
                "red": {"keys": 0, "locks": 0, "key_reachable_without_locks": False, "has_key_and_lock": False},
                "yellow": {"keys": 0, "locks": 0, "key_reachable_without_locks": False, "has_key_and_lock": False},
            },
            "has_keyed_route_signal": False,
        },
    }


def optional_lock_sample() -> dict:
    sample = progression_sample()
    sample["counts"] = dict(sample["counts"], locked_doors=1, keys=1)
    sample["things_by_category"] = {"monsters": 4, "weapons": 1, "keys": 1}
    sample["progression"] = dict(sample["progression"], exit_requires_any_lock=False, exit_reachable_without_locks=True)
    return sample


class ExtractionTests(unittest.TestCase):
    def test_graph_sample_converts_to_valid_canonical_dsl_with_key_lock_order(self) -> None:
        document = dsl_from_graph_sample(progression_sample())
        report = validate_document(document)

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(document["metadata"]["source_family"], "example")
        self.assertEqual(document["theme"]["primary"], "hell")
        self.assertEqual([key["color"] for key in document["progression"]["keys"]], ["blue", "red"])
        self.assertEqual([gate["key"] for gate in document["progression"]["gates"]], ["blue", "red"])
        self.assertLess(
            document["progression"]["critical_path"].index("blue_key_room"),
            document["progression"]["critical_path"].index("blue_locked_door"),
        )
        self.assertIn("locked_doors_are_partition_edges", document["validation"]["required"])

    def test_training_record_preserves_prompt_source_labels_and_valid_dsl(self) -> None:
        record = training_record_from_graph_sample(progression_sample(), sample_path="samples/example_MAP06.json")

        self.assertEqual(record["source_sample"], "samples/example_MAP06.json")
        self.assertIn("blue key", record["prompt"])
        self.assertTrue(record["labels"]["strict_progression_candidate"])
        self.assertEqual(record["labels"]["bucket"], "required_multi_lock")
        self.assertTrue(validate_document(record["dsl"]).valid)

    def test_basic_route_sample_keeps_keys_and_gates_empty(self) -> None:
        record = training_record_from_graph_sample(basic_route_sample(), sample_path="samples/basic_MAP01.json")
        document = record["dsl"]

        self.assertEqual(record["labels"]["bucket"], "basic_route")
        self.assertEqual(record["labels"]["colors"], [])
        self.assertEqual(document["progression"]["keys"], [])
        self.assertEqual(document["progression"]["gates"], [])
        self.assertIn("no locked doors", record["prompt"])
        self.assertTrue(validate_document(document).valid)

    def test_optional_lock_sample_does_not_fake_required_exit_gate(self) -> None:
        record = training_record_from_graph_sample(optional_lock_sample(), sample_path="samples/optional_MAP06.json")
        document = record["dsl"]

        self.assertEqual(record["labels"]["bucket"], "optional_key_lock")
        self.assertEqual(record["labels"]["colors"], [])
        self.assertEqual(record["labels"]["observed_paired_colors"], ["blue", "red"])
        self.assertEqual(document["progression"]["keys"], [])
        self.assertEqual(document["progression"]["gates"], [])
        self.assertTrue(validate_document(document).valid)

    def test_prompt_variants_make_prompt_control_explicit(self) -> None:
        no_lock_prompts = prompt_variants_from_graph_sample(basic_route_sample(), variant_count=3)
        keyed_prompts = prompt_variants_from_graph_sample(progression_sample(), variant_count=3)

        self.assertEqual(len(no_lock_prompts), 3)
        self.assertTrue(any("no locked doors" in prompt for prompt in no_lock_prompts))
        self.assertTrue(any("start" in prompt and "exit" in prompt for prompt in no_lock_prompts))
        self.assertTrue(any("blue key" in prompt and "red key" in prompt for prompt in keyed_prompts))

    def test_build_dsl_training_cache_writes_manifest_samples_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            sample_path = samples_dir / "example_MAP06.json"
            sample_path.write_text(json.dumps(progression_sample()) + "\n", encoding="utf-8")
            (graph_cache / "manifest.jsonl").write_text(
                json.dumps({"sample": "samples/example_MAP06.json", "family": "example", "map": "MAP06"}) + "\n",
                encoding="utf-8",
            )

            output = root / "dsl_training_v06"
            summary = build_dsl_training_cache(graph_cache, output, strict_only=True)
            manifest_lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
            record = json.loads((output / "samples" / "dsl_00000.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["source_samples"], 1)
        self.assertEqual(summary["strict_progression_candidates"], 1)
        self.assertEqual(len(manifest_lines), 1)
        self.assertEqual(record["dsl"]["progression"]["gates"][0]["id"], "blue_locked_door")

    def test_build_dsl_training_cache_can_expand_prompt_variants_without_strict_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            (samples_dir / "basic_MAP01.json").write_text(json.dumps(basic_route_sample()) + "\n", encoding="utf-8")
            (samples_dir / "example_MAP06.json").write_text(json.dumps(progression_sample()) + "\n", encoding="utf-8")
            (graph_cache / "manifest.jsonl").write_text(
                "".join(
                    [
                        json.dumps({"sample": "samples/basic_MAP01.json", "family": "basic", "map": "MAP01"}) + "\n",
                        json.dumps({"sample": "samples/example_MAP06.json", "family": "example", "map": "MAP06"}) + "\n",
                    ]
                ),
                encoding="utf-8",
            )

            output = root / "dsl_training_v08"
            summary = build_dsl_training_cache(graph_cache, output, strict_only=False, prompt_variants_per_sample=3)
            manifest_lines = (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()

        self.assertEqual(summary["source_samples"], 2)
        self.assertEqual(summary["samples"], 6)
        self.assertEqual(summary["prompt_variants_per_sample"], 3)
        self.assertEqual(summary["bucket_counts"]["basic_route"], 3)
        self.assertEqual(summary["bucket_counts"]["required_multi_lock"], 3)
        self.assertEqual(len(manifest_lines), 6)

    def test_audit_graph_cache_reports_full_dataset_buckets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            (samples_dir / "basic_MAP01.json").write_text(json.dumps(basic_route_sample()) + "\n", encoding="utf-8")
            (samples_dir / "optional_MAP06.json").write_text(json.dumps(optional_lock_sample()) + "\n", encoding="utf-8")
            (graph_cache / "manifest.jsonl").write_text(
                "".join(
                    [
                        json.dumps({"sample": "samples/basic_MAP01.json"}) + "\n",
                        json.dumps({"sample": "samples/optional_MAP06.json"}) + "\n",
                    ]
                ),
                encoding="utf-8",
            )

            audit = audit_graph_cache(graph_cache)

        self.assertEqual(audit["source_samples"], 2)
        self.assertEqual(audit["core_start_exit_route_candidates"], 2)
        self.assertEqual(audit["bucket_counts"], {"basic_route": 1, "optional_key_lock": 1})

    def test_cli_build_training_pairs_command_writes_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            graph_cache = root / "graph_cache"
            samples_dir = graph_cache / "samples"
            samples_dir.mkdir(parents=True)
            (samples_dir / "example_MAP06.json").write_text(
                json.dumps(progression_sample()) + "\n",
                encoding="utf-8",
            )
            (graph_cache / "manifest.jsonl").write_text(
                json.dumps({"sample": "samples/example_MAP06.json"}) + "\n",
                encoding="utf-8",
            )
            output = root / "dsl_training_v06"

            exit_code = main(
                [
                    "build-training-pairs",
                    "--graph-cache",
                    str(graph_cache),
                    "--output",
                    str(output),
                    "--strict-only",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "manifest.jsonl").exists())
            self.assertTrue((output / "samples" / "dsl_00000.json").exists())


if __name__ == "__main__":
    unittest.main()
