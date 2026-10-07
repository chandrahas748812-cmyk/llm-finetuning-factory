"""Tests for experiment tracking and the model registry."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from src.evaluate import benchmark_accuracy, perplexity
from src.registry import ModelRegistry
from src.tracking import ExperimentTracker, JsonlTracker


def test_perplexity_known_value():
    # uniform-ish logprobs: mean_nll = 1.0 -> ppl = e
    import math

    assert perplexity([-1.0, -1.0, -1.0]) == pytest.approx(math.e)


def test_perplexity_empty_raises():
    with pytest.raises(ValueError):
        perplexity([])


def test_benchmark_accuracy():
    result = benchmark_accuracy(
        ["  Paris ", "london", "Berlin"],
        ["paris", "London", "Munich"],
    )
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["correct"] == 2
    assert result["n"] == 3


def test_benchmark_accuracy_length_mismatch():
    with pytest.raises(ValueError):
        benchmark_accuracy(["a"], ["a", "b"])


def test_jsonl_tracker_roundtrip(tmp_path):
    tracker = JsonlTracker(tmp_path / "runs", "test-run")
    tracker.log_params({"lr": 2e-4, "epochs": 3})
    tracker.log_metrics({"loss": 1.5}, step=10)
    tracker.log_metrics({"loss": 1.2}, step=20)
    tracker.log_summary({"status": "ok"})

    assert json.loads((tmp_path / "runs" / "params.json").read_text())["epochs"] == 3
    events = tracker.read_events()
    assert len(events) == 2
    assert events[0]["metrics"]["loss"] == 1.5
    assert json.loads((tmp_path / "runs" / "summary.json").read_text())["status"] == "ok"


def test_experiment_tracker_defaults_to_jsonl(tmp_path):
    tracker = ExperimentTracker(tmp_path, "run-1")
    assert isinstance(tracker.impl, JsonlTracker)


def test_registry_register_and_list(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_text("fake-weights")

    v1 = registry.register("llama3-8b", adapter, {"run": "a"})
    assert v1.name == "v1"
    assert (v1 / "metadata.json").exists()
    assert (v1 / "README.md").exists()

    v2 = registry.register("llama3-8b", adapter, {"run": "b"})
    assert v2.name == "v2"
    assert registry.list_versions("llama3-8b") == ["v1", "v2"]
    assert registry.latest("llama3-8b") == "v2"
    assert registry.get_metadata("llama3-8b", "v1")["run"] == "a"


def test_registry_duplicate_version_raises(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    registry.register("m", adapter, {}, version="v1")
    with pytest.raises(FileExistsError):
        registry.register("m", adapter, {}, version="v1")


def test_registry_unknown_model(tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    assert registry.list_versions("nope") == []
    assert registry.latest("nope") is None
