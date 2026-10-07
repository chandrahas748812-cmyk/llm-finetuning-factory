"""Dataset preparation: instruction formatting, splits, tokenization stats.

Works with plain dict records: {"instruction": ..., "input": ..., "output": ...}
or pre-formatted {"text": ...} records.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Tuple


CHAT_TEMPLATES = {
    "chatml": (
        "<|im_start|>system\n{system}<|im_end|>\n"
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n{output}<|im_end|>"
    ),
    "alpaca": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"
    ),
    "llama3": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}"
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"
    ),
}

DEFAULT_SYSTEM = "You are a helpful assistant."


def format_instruction(
    record: Dict[str, Any],
    template: str = "chatml",
    system: str = DEFAULT_SYSTEM,
) -> str:
    """Format one record into a training string."""
    if "text" in record and "instruction" not in record:
        return str(record["text"])
    tmpl = CHAT_TEMPLATES.get(template, CHAT_TEMPLATES["chatml"])
    return tmpl.format(
        system=record.get("system", system),
        instruction=record.get("instruction", ""),
        input=record.get("input", ""),
        output=record.get("output", ""),
    ).strip()


def train_val_split(
    records: List[Dict[str, Any]],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic train/validation split."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def tokenization_stats(
    texts: List[str],
    max_seq_length: int = 2048,
    approx_chars_per_token: float = 4.0,
) -> Dict[str, Any]:
    """Estimate token-length distribution without loading a tokenizer.

    Uses the ~4 chars/token heuristic; swap in a real tokenizer count for
    exact numbers before setting max_seq_length for production runs.
    """
    est_tokens = [max(1, int(len(t) / approx_chars_per_token)) for t in texts]
    truncated = sum(1 for n in est_tokens if n > max_seq_length)
    counter = Counter()
    for n in est_tokens:
        bucket = min(n // 256 * 256, 4096)
        counter[bucket] += 1
    return {
        "n_texts": len(texts),
        "mean_est_tokens": round(sum(est_tokens) / len(est_tokens), 1) if est_tokens else 0,
        "max_est_tokens": max(est_tokens) if est_tokens else 0,
        "p95_est_tokens": sorted(est_tokens)[int(len(est_tokens) * 0.95)] if est_tokens else 0,
        "truncated_at_max_seq_length": truncated,
        "truncation_rate": round(truncated / len(est_tokens), 4) if est_tokens else 0,
        "length_histogram_256b": dict(sorted(counter.items())),
    }


def build_dataset(
    records: List[Dict[str, Any]],
    template: str = "chatml",
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Format records and split into train/val text lists."""
    texts = [format_instruction(r, template=template) for r in records]
    train, val = train_val_split(
        [{"text": t} for t in texts], val_ratio=val_ratio, seed=seed
    )
    return {
        "train": [r["text"] for r in train],
        "validation": [r["text"] for r in val],
    }
