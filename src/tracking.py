"""Experiment tracking: MLflow when available, JSONL fallback otherwise.

The JSONL tracker writes one JSON object per line:
{"step": ..., "metrics": {...}, "timestamp": ...}
plus a final summary.json — enough to compare runs without infra.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JsonlTracker:
    def __init__(self, log_dir: str | Path, run_name: str):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.events_path = self.dir / "events.jsonl"
        self.params: Dict[str, Any] = {}

    def log_params(self, params: Dict[str, Any]) -> None:
        self.params.update(params)
        (self.dir / "params.json").write_text(json.dumps(self.params, indent=2))

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        event = {
            "run": self.run_name,
            "step": step,
            "metrics": metrics,
            "timestamp": time.time(),
        }
        with self.events_path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def log_summary(self, summary: Dict[str, Any]) -> None:
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def read_events(self) -> list[dict]:
        if not self.events_path.exists():
            return []
        return [json.loads(line) for line in self.events_path.read_text().splitlines() if line.strip()]


class MlflowTracker:
    """Thin wrapper; raises a clear error if mlflow is not installed."""

    def __init__(self, tracking_uri: Optional[str], experiment: str, run_name: str):
        try:
            import mlflow  # type: ignore
        except ImportError as e:
            raise RuntimeError("mlflow is not installed; use JsonlTracker") from e
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self._run = mlflow.start_run(run_name=run_name)
        self._mlflow = mlflow

    def log_params(self, params: Dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_summary(self, summary: Dict[str, Any]) -> None:
        self._mlflow.log_dict(summary, "summary.json")

    def close(self) -> None:
        self._mlflow.end_run()


class ExperimentTracker:
    """Factory: MLflow when configured, JSONL otherwise."""

    def __init__(
        self,
        log_dir: str | Path,
        run_name: str,
        use_mlflow: bool = False,
        tracking_uri: Optional[str] = None,
        experiment: str = "finetuning",
    ):
        if use_mlflow:
            self.impl = MlflowTracker(tracking_uri, experiment, run_name)
        else:
            self.impl = JsonlTracker(log_dir, run_name)

    def __getattr__(self, name: str):
        return getattr(self.impl, name)
