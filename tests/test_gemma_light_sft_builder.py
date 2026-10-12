from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_gemma_light_sft.py"
EXAMPLE = ROOT / "examples" / "blue_lock_processing.json"

spec = importlib.util.spec_from_file_location("build_gemma_light_sft", SCRIPT)
assert spec is not None and spec.loader is not None
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def load_example() -> dict[str, Any]:
    return json.loads(EXAMPLE.read_text(encoding="utf-8"))


def sft_record(document: dict[str, Any], *, bucket: str = "required_single_lock") -> dict[str, Any]:
    return {
        "schema": "romerodsl_sft_v0.8_full",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "make a map"},
            {"role": "assistant", "content": json.dumps(document, allow_nan=False)},
        ],
        "metadata": {"labels": {"bucket": bucket}},
    }


class GemmaLightSftBuilderTests(unittest.TestCase):
    def test_record_rejection_reasons_catch_bad_training_targets(self) -> None:
        valid = load_example()
        self.assertEqual(builder.record_rejection_reasons(sft_record(valid)), [])

        self_connected = copy.deepcopy(valid)
        self_connected["connections"].append(
            {"id": "fake_route", "from": "central_hub", "to": "central_hub", "type": "doorway"}
        )
        self.assertIn(
            "self_connection",
            builder.record_rejection_reasons(sft_record(self_connected)),
        )

        invalid_material = copy.deepcopy(valid)
        invalid_material["spaces"][0]["materials"] = {"wall": "CORRUPT_WALL"}
        self.assertIn(
            "invalid_material",
            builder.record_rejection_reasons(sft_record(invalid_material)),
        )

        missing_spaces = copy.deepcopy(valid)
        del missing_spaces["spaces"]
        self.assertIn(
            "missing_top_level_spaces",
            builder.record_rejection_reasons(sft_record(missing_spaces)),
        )

        bypass = copy.deepcopy(valid)
        bypass["connections"].append(
            {"id": "shortcut", "from": "central_hub", "to": "exit_room", "type": "doorway"}
        )
        self.assertIn("route_bypass", builder.record_rejection_reasons(sft_record(bypass)))

    def test_select_records_for_training_skips_rejected_records_before_filling_quota(self) -> None:
        valid_a = sft_record(load_example(), bucket="required_single_lock")
        valid_b = sft_record(load_example(), bucket="required_single_lock")
        invalid = sft_record(load_example(), bucket="required_single_lock")
        broken = json.loads(invalid["messages"][-1]["content"])
        broken["connections"].append(
            {"id": "fake_route", "from": "central_hub", "to": "central_hub", "type": "doorway"}
        )
        invalid["messages"][-1]["content"] = json.dumps(broken, allow_nan=False)

        selected, bucket_counts, rejections = builder.select_records_for_training(
            [invalid, valid_a, valid_b],
            rng=builder.random.Random(builder.SEED),
            quotas={"required_single_lock": 2},
        )

        self.assertEqual(len(selected), 2)
        self.assertEqual(bucket_counts, {"required_single_lock": 2})
        self.assertEqual(rejections["self_connection"], 1)


if __name__ == "__main__":
    unittest.main()
