"""romeroDSL: semantic Doom map design contracts."""

from romerodsl.compiler import compile_to_layout, compile_to_plan, compile_to_wad
from romerodsl.extraction import (
    build_dsl_training_cache,
    dsl_from_graph_sample,
    training_record_from_graph_sample,
)
from romerodsl.finetune import (
    build_sft_jsonl,
    canonical_sft_dsl,
    sft_record_from_training_record,
    validate_sft_jsonl,
)
from romerodsl.qwen_lora import (
    SFTTrainConfig,
    build_tokenization_diagnostics,
    record_group_key,
    split_group_report,
    split_records,
    train_qwen_lora,
    training_run_plan,
)
from romerodsl.schema import ValidationReport, validate_document

__all__ = [
    "ValidationReport",
    "SFTTrainConfig",
    "build_tokenization_diagnostics",
    "record_group_key",
    "split_group_report",
    "split_records",
    "build_dsl_training_cache",
    "build_sft_jsonl",
    "canonical_sft_dsl",
    "compile_to_layout",
    "compile_to_plan",
    "compile_to_wad",
    "dsl_from_graph_sample",
    "sft_record_from_training_record",
    "training_record_from_graph_sample",
    "train_qwen_lora",
    "training_run_plan",
    "validate_document",
    "validate_sft_jsonl",
]
