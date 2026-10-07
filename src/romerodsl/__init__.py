"""romeroDSL: semantic Doom map design contracts."""

from romerodsl.compiler import compile_to_layout, compile_to_plan, compile_to_wad
from romerodsl.extraction import (
    build_dsl_training_cache,
    dsl_from_graph_sample,
    training_record_from_graph_sample,
)
from romerodsl.schema import ValidationReport, validate_document

__all__ = [
    "ValidationReport",
    "build_dsl_training_cache",
    "compile_to_layout",
    "compile_to_plan",
    "compile_to_wad",
    "dsl_from_graph_sample",
    "training_record_from_graph_sample",
    "validate_document",
]
