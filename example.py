"""End-to-end dry run: config -> dataset -> stats -> tracking -> registry.

Exercises the full MLOps loop without torch/GPU.
Run: python example.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import load_config
from src.dataset import build_dataset, tokenization_stats
from src.evaluate import benchmark_accuracy, perplexity
from src.registry import ModelRegistry
from src.tracking import ExperimentTracker

# --- synthetic instruction dataset ---
RECORDS = [
    {
        "instruction": "Explain what a vector database is.",
        "input": "",
        "output": "A vector database stores embeddings and supports similarity search.",
    },
    {
        "instruction": "What is LoRA fine-tuning?",
        "input": "",
        "output": "LoRA freezes base weights and trains low-rank adapter matrices.",
    },
    {
        "instruction": "Define perplexity.",
        "input": "",
        "output": "Perplexity is exp(mean negative log-likelihood); lower is better.",
    },
    {
        "instruction": "What does RAG stand for?",
        "input": "",
        "output": "Retrieval-Augmented Generation.",
    },
]


def main():
    workdir = Path(tempfile.mkdtemp(prefix="ftf-demo-"))
    print(f"workdir: {workdir}")

    # 1. config
    cfg = load_config(Path(__file__).resolve().parent / "configs" / "lora-phi3-mini.yaml")
    print(f"config: {cfg.run_name} | method={cfg.method} | base={cfg.base_model}")

    # 2. dataset prep
    data = build_dataset(RECORDS, template=cfg.chat_template, val_ratio=cfg.val_ratio, seed=42)
    stats = tokenization_stats(data["train"], max_seq_length=cfg.train.max_seq_length)
    print(f"train={len(data['train'])} val={len(data['validation'])} "
          f"mean_tokens={stats['mean_est_tokens']} truncation_rate={stats['truncation_rate']}")

    # 3. experiment tracking
    tracker = ExperimentTracker(workdir / "runs", cfg.run_name)
    tracker.log_params({"base_model": cfg.base_model, "method": cfg.method, "lora_r": cfg.lora.r})
    for step, loss in enumerate([2.4, 1.9, 1.5], start=1):
        tracker.log_metrics({"train/loss": loss, "eval/perplexity": 10.0 - step}, step=step * 10)
    tracker.log_summary({"status": "demo-ok"})

    # 4. eval hooks (offline math)
    print(f"demo perplexity([-0.7]*100) = {perplexity([-0.7] * 100):.2f}")
    acc = benchmark_accuracy(["Paris", "london"], ["paris", "London"])
    print(f"demo benchmark accuracy = {acc['accuracy']:.2f}")

    # 5. registry
    fake_adapter = workdir / "adapter"
    fake_adapter.mkdir()
    (fake_adapter / "adapter_config.json").write_text(json.dumps({"r": cfg.lora.r}))
    registry = ModelRegistry(workdir / "registry")
    version_path = registry.register(cfg.base_model, fake_adapter, {"run_name": cfg.run_name})
    print(f"registered: {version_path}")
    print(f"versions: {registry.list_versions(cfg.base_model)}")

    shutil.rmtree(workdir, ignore_errors=True)
    print("\nDry run complete — full pipeline validated without GPU.")


if __name__ == "__main__":
    main()
