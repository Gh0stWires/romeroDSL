"""Qwen LoRA training utilities for v0.7 prompt-to-romeroDSL learning."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_QWEN_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(slots=True)
class SFTTrainConfig:
    """Configuration for a Qwen LoRA SFT run."""

    sft_jsonl: Path
    output_dir: Path
    model_name: str = DEFAULT_QWEN_MODEL
    validation_fraction: float = 0.08
    test_fraction: float = 0.0
    seed: int = 1337
    max_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    epochs: float = 1.0
    max_steps: int | None = None
    save_every_steps: int = 50
    eval_every_steps: int = 25
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bf16: bool = True
    load_in_4bit: bool = False
    overlength_policy: str = "reject"


def load_sft_records(path: Path) -> list[dict[str, Any]]:
    """Load SFT JSONL records."""

    records = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not a JSON object")
        messages = value.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"line {line_number} has no chat messages")
        records.append(value)
    if not records:
        raise ValueError(f"no records found in {path}")
    return records


def record_group_key(record: dict[str, Any]) -> str:
    """Return a leakage-resistant design-family key for one SFT record."""

    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    labels = metadata.get("labels") if isinstance(metadata.get("labels"), dict) else {}
    dataset_class = str(labels.get("dataset_class") or metadata.get("dataset_class") or "unknown")
    source_wad = metadata.get("source_wad")
    source_sample = metadata.get("source_sample")
    map_name = metadata.get("map") or labels.get("map")
    if source_wad:
        return f"extracted-wad:{source_wad}|map:{map_name or 'unknown'}"
    if source_sample and dataset_class == "extracted":
        return f"extracted-sample:{source_sample}|map:{map_name or 'unknown'}"

    design_family = labels.get("design_family") or metadata.get("design_family")
    if design_family and dataset_class in {"synthetic", "hand-authored"}:
        return f"{dataset_class}-family:{design_family}"
    generator_seed = labels.get("generator_seed") or metadata.get("generator_seed")
    if generator_seed is None and isinstance(source_sample, str) and dataset_class == "synthetic":
        generator_seed = source_sample
    if generator_seed is not None and dataset_class == "synthetic":
        return f"synthetic-seed:{generator_seed}"
    if source_sample:
        return f"{dataset_class}-sample:{source_sample}"

    stable_payload = {
        "dataset_class": dataset_class,
        "messages": record.get("messages"),
        "title": metadata.get("title"),
    }
    digest = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:20]
    return f"{dataset_class}-anonymous:{digest}"


def split_records(
    records: list[dict[str, Any]], *, validation_fraction: float, seed: int, test_fraction: float = 0.0
) -> dict[str, list[dict[str, Any]]]:
    """Split whole design groups, never individual prompt variants."""

    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("validation_fraction and test_fraction must be in [0, 1)")
    if validation_fraction + test_fraction >= 1 and len(records) > 1:
        raise ValueError("validation_fraction + test_fraction must be less than 1")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record_group_key(record), []).append(record)
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    validation_target = 0 if len(records) <= 1 else max(1, round(len(records) * validation_fraction))
    test_target = 0 if len(records) <= 1 or test_fraction == 0 else max(1, round(len(records) * test_fraction))

    def take_groups(target: int, available: list[tuple[str, list[dict[str, Any]]]]) -> set[str]:
        selected: set[str] = set()
        selected_size = 0
        for key, group in available:
            if selected_size >= target:
                break
            current_distance = abs(selected_size - target)
            next_distance = abs(selected_size + len(group) - target)
            if not selected or next_distance <= current_distance:
                selected.add(key)
                selected_size += len(group)
        return selected

    validation_keys = take_groups(validation_target, group_items)
    remaining_groups = [(key, group) for key, group in group_items if key not in validation_keys]
    test_keys = take_groups(test_target, remaining_groups)
    partitions = {
        "train": [record for record in records if record_group_key(record) not in validation_keys | test_keys],
        "validation": [record for record in records if record_group_key(record) in validation_keys],
    }
    if test_fraction:
        partitions["test"] = [record for record in records if record_group_key(record) in test_keys]
    return partitions


def split_group_report(splits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Report partition groups and assert no design family crosses the split."""

    partition_keys = {
        partition: sorted({record_group_key(record) for record in records})
        for partition, records in splits.items()
    }
    locations: dict[str, set[str]] = {}
    for partition, keys in partition_keys.items():
        for key in keys:
            locations.setdefault(key, set()).add(partition)
    overlap = sorted(key for key, partitions in locations.items() if len(partitions) > 1)
    if overlap:
        raise ValueError(f"design groups cross dataset partitions: {overlap[:5]}")
    return {
        "strategy": "grouped_design_family",
        "partition_group_counts": {partition: len(keys) for partition, keys in partition_keys.items()},
        "partition_group_keys": partition_keys,
        "train_groups": len(partition_keys.get("train", [])),
        "validation_groups": len(partition_keys.get("validation", [])),
        "test_groups": len(partition_keys.get("test", [])),
        "cross_partition_groups": overlap,
        "train_group_keys": partition_keys.get("train", []),
        "validation_group_keys": partition_keys.get("validation", []),
        "test_group_keys": partition_keys.get("test", []),
    }


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(values, "tolist"):
        values = values.tolist()
    while values and isinstance(values[0], list):
        values = values[0]
    return list(values)


def build_tokenization_diagnostics(
    records: list[dict[str, Any]], tokenizer: Any, *, max_length: int
) -> dict[str, Any]:
    """Measure full, prompt, target, and safely discardable token lengths."""

    if max_length <= 0:
        raise ValueError("max_length must be positive")
    examples: list[dict[str, Any]] = []
    total_target = 0
    discarded_target = 0
    critical_truncations = 0
    for index, record in enumerate(records):
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            raise ValueError(f"record {index} has no complete chat messages")
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
        full_ids = _token_ids(tokenizer, full_text)
        prompt_ids = _token_ids(tokenizer, prompt_text)
        prompt_length = len(prompt_ids)
        total_length = len(full_ids)
        target_length = max(0, total_length - prompt_length)
        kept_target = max(0, min(total_length, max_length) - min(prompt_length, max_length))
        target_discarded = max(0, target_length - kept_target)
        assistant_content = messages[-1].get("content") if isinstance(messages[-1], dict) else ""
        critical_json = False
        if isinstance(assistant_content, str):
            try:
                json.loads(assistant_content)
                critical_json = True
            except json.JSONDecodeError:
                critical_json = False
        if critical_json and target_discarded:
            critical_truncations += 1
        total_target += target_length
        discarded_target += target_discarded
        examples.append(
            {
                "index": index,
                "prompt_tokens": prompt_length,
                "target_tokens": target_length,
                "total_sequence_tokens": total_length,
                "target_tokens_discarded": target_discarded,
                "overlength": total_length > max_length,
                "critical_json_completion": critical_json,
                "critical_json_truncated": critical_json and target_discarded > 0,
            }
        )
    overlength = [example for example in examples if example["overlength"]]
    return {
        "max_length": max_length,
        "records": len(records),
        "prompt_token_lengths": [example["prompt_tokens"] for example in examples],
        "target_token_lengths": [example["target_tokens"] for example in examples],
        "total_sequence_lengths": [example["total_sequence_tokens"] for example in examples],
        "overlength_examples": len(overlength),
        "overlength_indices": [example["index"] for example in overlength],
        "supervised_target_tokens": total_target,
        "supervised_target_tokens_discarded": discarded_target,
        "fraction_of_supervised_target_tokens_discarded": discarded_target / total_target if total_target else 0.0,
        "truncated_critical_json_completions": critical_truncations,
        "examples": examples,
    }


def training_run_plan(config: SFTTrainConfig) -> dict[str, Any]:
    """Return a dependency-free description of what a training run will produce."""

    return {
        "base_model": config.model_name,
        "sft_jsonl": str(config.sft_jsonl).replace("\\", "/"),
        "output_dir": str(config.output_dir).replace("\\", "/"),
        "max_length": config.max_length,
        "overlength_policy": config.overlength_policy,
        "split_strategy": "grouped_design_family",
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "max_steps": config.max_steps,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "lora": {
            "rank": config.lora_rank,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
        },
        "load_in_4bit": config.load_in_4bit,
        "artifacts": {
            "adapter": str(config.output_dir / "adapter").replace("\\", "/"),
            "metrics": str(config.output_dir / "metrics.json").replace("\\", "/"),
            "config": str(config.output_dir / "train_config.json").replace("\\", "/"),
        },
    }


def train_qwen_lora(config: SFTTrainConfig) -> dict[str, Any]:
    """Train a LoRA adapter. Imports ML dependencies lazily so normal tests stay light."""

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.nn.utils.rnn import pad_sequence
    from torch.utils.data import DataLoader
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

    if config.overlength_policy not in {"reject", "preserve_target"}:
        raise ValueError("overlength_policy must be 'reject' or 'preserve_target'")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2) + "\n", encoding="utf-8"
    )
    records = load_sft_records(config.sft_jsonl)
    splits = split_records(
        records,
        validation_fraction=config.validation_fraction,
        test_fraction=config.test_fraction,
        seed=config.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenization = build_tokenization_diagnostics(records, tokenizer, max_length=config.max_length)
    (config.output_dir / "tokenization_diagnostics.json").write_text(
        json.dumps(tokenization, indent=2) + "\n", encoding="utf-8"
    )
    if config.overlength_policy == "reject" and tokenization["overlength_examples"]:
        raise ValueError(
            "overlength SFT examples rejected; see tokenization_diagnostics.json "
            f"({tokenization['overlength_examples']} examples, "
            f"{tokenization['truncated_critical_json_completions']} critical JSON completions truncated)"
        )
    if tokenization["truncated_critical_json_completions"]:
        raise ValueError(
            "critical JSON completions would be truncated; increase data quality or use a larger "
            "verified max_length, not silent truncation"
        )

    dtype = torch.bfloat16 if config.bf16 and torch.cuda.is_available() else torch.float16
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora_target_modules = _resolve_lora_target_modules(model, config.model_name)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = _ChatDataset(splits["train"], tokenizer, config.max_length, config.overlength_policy)
    val_dataset = (
        _ChatDataset(splits["validation"], tokenizer, config.max_length, config.overlength_policy)
        if splits["validation"]
        else None
    )

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        labels = pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=-100)
        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    val_loader = (
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate)
        if val_dataset is not None
        else None
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / config.gradient_accumulation_steps))
    total_steps = config.max_steps or max(1, math.ceil(updates_per_epoch * config.epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    warmup_steps = max(1, int(total_steps * config.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    device = next(model.parameters()).device
    scaler = None
    global_step = 0
    running_loss = 0.0
    metrics: dict[str, Any] = {
        "plan": training_run_plan(config),
        "records": len(records),
        "train_records": len(splits["train"]),
        "validation_records": len(splits["validation"]),
        "test_records": len(splits.get("test", [])),
        "split_report": split_group_report(splits),
        "tokenization": tokenization,
        "steps": [],
    }

    model.train()
    while global_step < total_steps:
        for batch_index, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu())
            should_step = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or (batch_index + 1) == len(train_loader)
            )
            if should_step:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                step_metrics = {
                    "step": global_step,
                    "train_loss": running_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                running_loss = 0.0
                if val_loader is not None and global_step % config.eval_every_steps == 0:
                    step_metrics["validation_loss"] = _evaluate_loss(model, val_loader, device)
                    model.train()
                metrics["steps"].append(step_metrics)
                _write_metrics(config.output_dir, metrics)
                print(json.dumps(step_metrics), flush=True)
                if global_step % config.save_every_steps == 0:
                    model.save_pretrained(config.output_dir / f"adapter-step-{global_step}")
                    tokenizer.save_pretrained(config.output_dir / f"adapter-step-{global_step}")
                if global_step >= total_steps:
                    break
    adapter_dir = config.output_dir / "adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    metrics["adapter_dir"] = str(adapter_dir)
    _write_metrics(config.output_dir, metrics)
    return metrics


def _resolve_lora_target_modules(model: Any, model_name: str) -> list[str]:
    """Return PEFT target modules, avoiding Gemma 4's vision/audio wrappers."""

    lowered = model_name.lower()
    if "gemma-4" not in lowered:
        return list(LORA_TARGET_MODULES)

    targets = []
    suffixes = tuple(f".{module_name}" for module_name in LORA_TARGET_MODULES)
    for name, module in model.named_modules():
        if not name.startswith("model.language_model."):
            continue
        if not name.endswith(suffixes):
            continue
        targets.append(name)
    if not targets:
        raise ValueError("no Gemma 4 language-model LoRA target modules found")
    return targets


class _ChatDataset:
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int, overlength_policy: str) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.overlength_policy = overlength_policy

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import torch

        messages = self.records[index]["messages"]
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True
        )
        full_ids = _token_ids(self.tokenizer, full_text)
        prompt_ids = _token_ids(self.tokenizer, prompt_text)
        prompt_length = len(prompt_ids)
        target_ids = full_ids[prompt_length:]
        if len(target_ids) > self.max_length:
            raise ValueError(
                f"record {index} target has {len(target_ids)} tokens, exceeding max_length={self.max_length}; "
                "complete JSON targets cannot be truncated"
            )
        if len(full_ids) > self.max_length:
            if self.overlength_policy != "preserve_target":
                raise ValueError(f"record {index} exceeds max_length without an explicit preserve_target policy")
            prompt_keep = self.max_length - len(target_ids)
            input_ids = prompt_ids[-prompt_keep:] + target_ids if prompt_keep else target_ids
            prompt_length = prompt_keep
        else:
            input_ids = full_ids
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels = input_tensor.clone()
        labels[:prompt_length] = -100
        if torch.all(labels == -100):
            raise ValueError(f"record {index} has no supervised target tokens after tokenization")
        return {"input_ids": input_tensor, "labels": labels}


def _evaluate_loss(model: Any, loader: Any, device: Any) -> float:
    import torch

    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            losses.append(float(model(**batch).loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def _write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


def _jsonable_config(config: SFTTrainConfig) -> dict[str, Any]:
    value = asdict(config)
    value["sft_jsonl"] = str(config.sft_jsonl)
    value["output_dir"] = str(config.output_dir)
    return value
