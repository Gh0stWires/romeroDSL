"""Qwen LoRA training utilities for v0.7 prompt-to-romeroDSL learning."""

from __future__ import annotations

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


def split_records(
    records: list[dict[str, Any]], *, validation_fraction: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    """Deterministically split records into train/validation partitions."""

    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    validation_count = 0 if len(records) == 1 else max(1, round(len(records) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    return {
        "train": [record for index, record in enumerate(records) if index not in validation_indices],
        "validation": [record for index, record in enumerate(records) if index in validation_indices],
    }


def training_run_plan(config: SFTTrainConfig) -> dict[str, Any]:
    """Return a dependency-free description of what a training run will produce."""

    return {
        "base_model": config.model_name,
        "sft_jsonl": str(config.sft_jsonl).replace("\\", "/"),
        "output_dir": str(config.output_dir).replace("\\", "/"),
        "max_length": config.max_length,
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_config.json").write_text(
        json.dumps(_jsonable_config(config), indent=2) + "\n", encoding="utf-8"
    )
    records = load_sft_records(config.sft_jsonl)
    splits = split_records(records, validation_fraction=config.validation_fraction, seed=config.seed)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    train_dataset = _ChatDataset(splits["train"], tokenizer, config.max_length)
    val_dataset = _ChatDataset(splits["validation"], tokenizer, config.max_length) if splits["validation"] else None

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
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

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
        full = self.tokenizer(full_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        prompt = self.tokenizer(prompt_text, truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        prompt_length = min(prompt["input_ids"].shape[1], labels.shape[0])
        labels[:prompt_length] = -100
        if torch.all(labels == -100):
            labels[-1] = input_ids[-1]
        return {"input_ids": input_ids, "labels": labels}


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
