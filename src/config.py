"""Typed fine-tuning configs loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QuantConfig:
    enabled: bool = False
    bits: int = 4
    compute_dtype: str = "bfloat16"


@dataclass
class TrainConfig:
    output_dir: str = "./outputs"
    num_epochs: int = 3
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # bf16 | fp16 | no
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 42


@dataclass
class FinetuneConfig:
    run_name: str
    base_model: str
    dataset_path: str
    text_field: str = "text"
    val_ratio: float = 0.05
    chat_template: str = "chatml"  # chatml | alpaca | llama3
    lora: LoraConfig = field(default_factory=LoraConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_benchmarks: List[str] = field(default_factory=list)

    @property
    def method(self) -> str:
        return "qlora" if self.quant.enabled else "lora"


def _from_dict(cls, data: Dict[str, Any]):
    """Instantiate a dataclass, ignoring unknown keys."""
    import dataclasses

    known = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: str | Path) -> FinetuneConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return FinetuneConfig(
        run_name=raw["run_name"],
        base_model=raw["base_model"],
        dataset_path=raw["dataset_path"],
        text_field=raw.get("text_field", "text"),
        val_ratio=float(raw.get("val_ratio", 0.05)),
        chat_template=raw.get("chat_template", "chatml"),
        lora=_from_dict(LoraConfig, raw.get("lora", {})),
        quant=_from_dict(QuantConfig, raw.get("quant", {})),
        train=_from_dict(TrainConfig, raw.get("train", {})),
        eval_benchmarks=list(raw.get("eval_benchmarks", [])),
    )
