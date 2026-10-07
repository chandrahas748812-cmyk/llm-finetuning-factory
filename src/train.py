"""LoRA/QLoRA fine-tuning with HuggingFace TRL + PEFT.

Usage:
    python -m src.train --config configs/lora-llama3-8b.yaml
    python -m src.train --config configs/lora-llama3-8b.yaml --dry-run   # validate pipeline, no GPU/torch needed

Dry-run mode exercises config loading, dataset prep, stats, tracking, and
registry writes with synthetic artifacts — the full MLOps loop minus the
actual gradient steps. Training mode requires torch, transformers, peft,
trl, and (for QLoRA) bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.dataset import build_dataset, tokenization_stats
from src.evaluate import perplexity_from_nll
from src.registry import ModelRegistry
from src.tracking import ExperimentTracker


def _require_deps(quantized: bool):
    missing = []
    for mod in ["torch", "transformers", "peft", "trl", "datasets"]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if quantized:
        try:
            __import__("bitsandbytes")
        except ImportError:
            missing.append("bitsandbytes")
    if missing:
        raise RuntimeError(
            f"Training requires missing packages: {', '.join(missing)}. "
            "pip install -r requirements.txt  (or use --dry-run)"
        )


def dry_run(cfg, args) -> Path:
    """Validate the whole pipeline without torch/GPU."""
    print(f"[dry-run] config: {cfg.run_name} | method={cfg.method} | base={cfg.base_model}")

    # 1. dataset
    raw_records = json.loads(Path(cfg.dataset_path).read_text())
    data = build_dataset(raw_records, template=cfg.chat_template,
                         val_ratio=cfg.val_ratio, seed=cfg.train.seed)
    stats = tokenization_stats(data["train"], max_seq_length=cfg.train.max_seq_length)
    print(f"[dry-run] train={len(data['train'])} val={len(data['validation'])} "
          f"truncation_rate={stats['truncation_rate']}")

    # 2. tracking
    tracker = ExperimentTracker(args.log_dir, cfg.run_name)
    tracker.log_params({
        "base_model": cfg.base_model, "method": cfg.method,
        "lora_r": cfg.lora.r, "lora_alpha": cfg.lora.lora_alpha,
        "lr": cfg.train.learning_rate, "epochs": cfg.train.num_epochs,
        "max_seq_length": cfg.train.max_seq_length,
    })
    for step in (10, 20, 30):
        tracker.log_metrics({"train/loss": 2.5 - step * 0.03, "eval/perplexity": 12.0 - step * 0.2}, step=step)
    tracker.log_summary({"status": "dry-run-ok", **stats})

    # 3. registry with synthetic adapter dir
    fake_adapter = Path(args.log_dir) / "fake_adapter"
    fake_adapter.mkdir(parents=True, exist_ok=True)
    (fake_adapter / "adapter_config.json").write_text(json.dumps({"r": cfg.lora.r, "dry_run": True}))
    registry = ModelRegistry(args.registry_dir)
    version_path = registry.register(
        model_name=cfg.base_model,
        artifact_dir=fake_adapter,
        metadata={"run_name": cfg.run_name, "method": cfg.method,
                  "dry_run": True, "dataset_stats": stats},
    )
    print(f"[dry-run] registered {version_path}")
    return version_path


def train(cfg, args):
    _require_deps(quantized=cfg.quant.enabled)

    import torch
    from datasets import Dataset
    from peft import LoraConfig as PeftLoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTTrainer, SFTConfig

    print(f"[train] {cfg.run_name} | {cfg.method} on {cfg.base_model}")
    raw_records = json.loads(Path(cfg.dataset_path).read_text())
    data = build_dataset(raw_records, template=cfg.chat_template,
                         val_ratio=cfg.val_ratio, seed=cfg.train.seed)
    print(f"[train] n_train={len(data['train'])} n_val={len(data['validation'])}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.train.mixed_precision, None)
    quant_config = None
    if cfg.quant.enabled:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=cfg.quant.bits == 4, load_in_8bit=cfg.quant.bits == 8,
            bnb_4bit_compute_dtype=dtype or torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model, quantization_config=quant_config,
        torch_dtype=dtype, device_map="auto",
    )
    if cfg.quant.enabled:
        model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = PeftLoraConfig(
        r=cfg.lora.r, lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=cfg.lora.target_modules,
        bias=cfg.lora.bias, task_type=cfg.lora.task_type,
    )

    sft_config = SFTConfig(
        output_dir=cfg.train.output_dir,
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.per_device_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        max_seq_length=cfg.train.max_seq_length,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        bf16=cfg.train.mixed_precision == "bf16",
        fp16=cfg.train.mixed_precision == "fp16",
        warmup_ratio=cfg.train.warmup_ratio,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps,
        eval_strategy="steps",
        seed=cfg.train.seed,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=Dataset.from_dict({"text": data["train"]}),
        eval_dataset=Dataset.from_dict({"text": data["validation"]}),
        peft_config=peft_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    tracker = ExperimentTracker(args.log_dir, cfg.run_name, use_mlflow=args.mlflow)
    tracker.log_params({"base_model": cfg.base_model, "method": cfg.method})

    trainer.train()

    adapter_dir = Path(cfg.train.output_dir) / "final_adapter"
    trainer.model.save_pretrained(adapter_dir)
    registry = ModelRegistry(args.registry_dir)
    version_path = registry.register(
        model_name=cfg.base_model, artifact_dir=adapter_dir,
        metadata={"run_name": cfg.run_name, "method": cfg.method},
    )
    print(f"[train] done → {version_path}")
    return version_path


def main():
    parser = argparse.ArgumentParser(description="LoRA/QLoRA fine-tuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-dir", default="./runs")
    parser.add_argument("--registry-dir", default="./registry")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        dry_run(cfg, args)
    else:
        train(cfg, args)


if __name__ == "__main__":
    main()
