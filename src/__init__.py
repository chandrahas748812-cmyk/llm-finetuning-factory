"""End-to-end LLM fine-tuning pipeline: configs, data prep, training, eval, registry."""

from .config import FinetuneConfig, load_config
from .dataset import (
    format_instruction,
    train_val_split,
    tokenization_stats,
    build_dataset,
)
from .tracking import ExperimentTracker, JsonlTracker
from .registry import ModelRegistry
from .evaluate import perplexity, benchmark_accuracy

__all__ = [
    "FinetuneConfig",
    "load_config",
    "format_instruction",
    "train_val_split",
    "tokenization_stats",
    "build_dataset",
    "ExperimentTracker",
    "JsonlTracker",
    "ModelRegistry",
    "perplexity",
    "benchmark_accuracy",
]
