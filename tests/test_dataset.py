"""Tests for dataset preparation utilities."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.dataset import (
    build_dataset,
    format_instruction,
    tokenization_stats,
    train_val_split,
)

RECORDS = [
    {"instruction": f"Question {i}", "input": "", "output": f"Answer {i}"}
    for i in range(20)
]


def test_format_instruction_chatml():
    text = format_instruction(RECORDS[0], template="chatml")
    assert "<|im_start|>user" in text
    assert "Question 0" in text
    assert "Answer 0" in text


def test_format_instruction_alpaca():
    text = format_instruction(RECORDS[0], template="alpaca")
    assert "### Instruction:" in text
    assert "### Response:" in text


def test_format_instruction_passthrough():
    text = format_instruction({"text": "already formatted"})
    assert text == "already formatted"


def test_train_val_split_sizes():
    train, val = train_val_split(RECORDS, val_ratio=0.1, seed=1)
    assert len(train) + len(val) == 20
    assert len(val) == 2  # max(1, int(20 * 0.1))


def test_train_val_split_deterministic():
    t1, v1 = train_val_split(RECORDS, seed=7)
    t2, v2 = train_val_split(RECORDS, seed=7)
    assert t1 == t2 and v1 == v2


def test_train_val_split_no_overlap():
    train, val = train_val_split(RECORDS, seed=3)
    train_ids = {id(r) for r in train}
    val_ids = {id(r) for r in val}
    assert not train_ids & val_ids


def test_train_val_split_bad_ratio():
    with pytest.raises(ValueError):
        train_val_split(RECORDS, val_ratio=0.0)


def test_tokenization_stats():
    texts = ["hello world", "x" * 10000]
    stats = tokenization_stats(texts, max_seq_length=2048)
    assert stats["n_texts"] == 2
    assert stats["max_est_tokens"] >= 2000
    assert stats["truncated_at_max_seq_length"] == 1
    assert 0.0 < stats["truncation_rate"] <= 1.0


def test_build_dataset():
    data = build_dataset(RECORDS, template="chatml", val_ratio=0.1, seed=1)
    assert set(data.keys()) == {"train", "validation"}
    assert len(data["train"]) + len(data["validation"]) == 20
    assert all(isinstance(t, str) for t in data["train"])
