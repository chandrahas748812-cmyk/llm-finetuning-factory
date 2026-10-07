# LLM Fine-Tuning Factory

An **end-to-end fine-tuning pipeline** for instruction-tuning open LLMs with **LoRA/QLoRA**. Covers the full MLOps loop: YAML configs per base model → dataset preparation (instruction formatting, train/val splits, tokenization stats) → training with HuggingFace **TRL + PEFT** (gradient checkpointing, mixed precision) → evaluation hooks (perplexity, task benchmarks, eval-harness-compatible) → experiment tracking (**MLflow** or zero-dependency **JSONL**) → **versioned model registry** → adapter inference.

## Quickstart

```bash
pip install -r requirements.txt        # core (no GPU needed)
python example.py                      # full dry-run: config → data → tracking → registry
pytest tests/ -v
```

Validate a real training config without a GPU:

```bash
# put your instruction data at ./data/instructions.json first
python -m src.train --config configs/lora-phi3-mini.yaml --dry-run
```

Real training (needs GPU + training deps):

```bash
pip install torch transformers peft trl datasets   # + bitsandbytes for QLoRA
python -m src.train --config configs/qlora-mistral-7b.yaml
```

Test a fine-tuned adapter:

```bash
python -m src.inference --base-model mistralai/Mistral-7B-v0.3 \
    --adapter ./registry/mistralai-mistral-7b-v0-3/v1/adapter \
    --prompt "Explain QLoRA in one sentence."
```

## Configs

| Config | Base model | Method | Target hardware |
|---|---|---|---|
| `configs/lora-llama3-8b.yaml` | Meta-Llama-3-8B | LoRA (r=16) | 1× 40GB+ GPU |
| `configs/qlora-mistral-7b.yaml` | Mistral-7B-v0.3 | QLoRA 4-bit (r=32) | 1× 24GB GPU |
| `configs/lora-phi3-mini.yaml` | Phi-3-mini-4k | LoRA (r=8) | 1× 16GB GPU |

Key knobs per config: `lora.r` / `lora_alpha` / `target_modules`, `quant.bits`, `train.learning_rate`, `train.max_seq_length`, `train.gradient_checkpointing`, `train.mixed_precision`.

## Pipeline stages

```
configs/*.yaml
      │  src/config.py — typed dataclasses, validated on load
      ▼
data/instructions.json  ({"instruction","input","output"} or {"text"})
      │  src/dataset.py — format_instruction (chatml|alpaca|llama3)
      │                   train_val_split (deterministic), tokenization_stats
      ▼
src/train.py  (TRL SFTTrainer + PEFT LoRA)
      │  gradient checkpointing · bf16/fp16 · warmup · eval steps
      ▼
src/evaluate.py — perplexity (sliding window) + benchmark_accuracy
      │  interface mirrors llm-eval-harness for drop-in eval suites
      ▼
src/tracking.py — MLflow when configured, else JSONL (events.jsonl + summary.json)
      ▼
src/registry.py — registry/<model-slug>/v<N>/{adapter,metadata.json,README.md}
      ▼
src/inference.py — load base + adapter, generate
```

## Results table template

Copy into your experiment notes after each run:

| run_name | base_model | method | epochs | lr | train_loss | eval_ppl | benchmark | adapter |
|---|---|---|---|---|---|---|---|---|
| phi3-mini-lora-v1 | Phi-3-mini-4k | lora r=8 | 3 | 3e-4 | | | | registry/.../v1 |
| mistral-7b-qlora-v1 | Mistral-7B-v0.3 | qlora r=32 | 2 | 2e-4 | | | | registry/.../v1 |

## Project layout

```
configs/               # per-model LoRA/QLoRA YAML configs
src/
  config.py            # FinetuneConfig dataclasses + YAML loader
  dataset.py           # instruction formatting, splits, token stats
  train.py             # TRL/PEFT training (+ --dry-run, no GPU needed)
  evaluate.py          # perplexity + benchmark hooks
  tracking.py          # MLflow / JSONL experiment tracker
  registry.py          # versioned model registry
  inference.py         # adapter inference (+ --mock offline mode)
tests/
  test_dataset.py
  test_registry.py     # tracking + registry + eval math
example.py             # end-to-end dry run demo
```

## Dataset format

`data/instructions.json` — a JSON array of either shape:

```json
[
  {"instruction": "Summarize this ticket.", "input": "…", "output": "…"},
  {"text": "<already formatted training string>"}
]
```

## Design notes

- **Dry-run first**: `python -m src.train --config <cfg> --dry-run` validates configs, data, tracking, and registry writes before you book GPU time.
- **Deterministic splits**: `train_val_split` is seeded; same data + seed = same split, every run.
- **Registry immutability**: versions are never overwritten (`FileExistsError` on collision); metadata captures config snapshot, metrics, and lineage.
- **Eval compatibility**: `evaluate.py` exposes the same `perplexity` / `benchmark_accuracy` interface as the `llm-eval-harness` project, so one eval suite serves both.
