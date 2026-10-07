"""Evaluation hooks: perplexity + task benchmarks.

Two modes:
- `*_with_model` functions run against a real HF model/tokenizer.
- `perplexity` / `benchmark_accuracy` (no model) compute from supplied
  per-token log-probs / predictions, so eval-harness-style pipelines can
  plug in without torch.

This mirrors the interface of the llm-eval-harness project.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def perplexity_from_nll(mean_nll: float) -> float:
    """Perplexity from mean negative log-likelihood per token."""
    return math.exp(mean_nll)


def perplexity(token_logprobs: Sequence[float]) -> float:
    """Corpus perplexity from per-token log probabilities."""
    if not token_logprobs:
        raise ValueError("token_logprobs must be non-empty")
    mean_nll = -sum(token_logprobs) / len(token_logprobs)
    return perplexity_from_nll(mean_nll)


def benchmark_accuracy(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Exact-match accuracy; normalizes whitespace/case."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")
    norm = lambda s: " ".join(s.strip().lower().split())
    hits = sum(1 for p, r in zip(predictions, references) if norm(p) == norm(r))
    total = len(references)
    return {"accuracy": hits / total if total else 0.0, "n": total, "correct": hits}


def perplexity_with_model(
    model, tokenizer, texts: List[str], max_length: int = 2048, stride: int = 512
) -> float:
    """Sliding-window perplexity with a HF causal LM (requires torch)."""
    import torch

    total_nll_tokens = 0.0  # sum of (loss * n_tokens) over chunks
    total_tokens = 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt")
        input_ids = enc.input_ids[0]
        for i in range(0, input_ids.size(0), stride):
            chunk = input_ids[i : i + max_length].unsqueeze(0)
            with torch.no_grad():
                outputs = model(chunk, labels=chunk)
            n_tokens = chunk.size(1)
            total_nll_tokens += float(outputs.loss) * n_tokens
            total_tokens += n_tokens
    mean_nll = total_nll_tokens / max(1, total_tokens)
    return math.exp(mean_nll)


def run_benchmarks(
    benchmarks: List[str],
    predict_fn,
    datasets: Dict[str, List[Dict[str, str]]],
) -> Dict[str, Dict[str, float]]:
    """Run named benchmarks. `predict_fn(prompt) -> prediction`."""
    results: Dict[str, Dict[str, float]] = {}
    for name in benchmarks:
        items = datasets.get(name, [])
        preds = [predict_fn(d["prompt"]) for d in items]
        refs = [d["reference"] for d in items]
        results[name] = benchmark_accuracy(preds, refs)
    return results
