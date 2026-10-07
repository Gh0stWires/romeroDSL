from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from romerodsl.cli import main
from romerodsl.extraction import build_dsl_training_cache, dsl_from_graph_sample, training_record_from_graph_sample
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
        self.assertIn("blue key before a blue locked door", record["prompt"])
        self.assertTrue(record["labels"]["strict_progression_candidate"])
        self.assertTrue(validate_document(record["dsl"]).valid)

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
        self.assertEqual(summary["strict_progression_candidates"], 1)
        self.assertEqual(len(manifest_lines), 1)
        self.assertEqual(record["dsl"]["progression"]["gates"][0]["id"], "blue_locked_door")

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
