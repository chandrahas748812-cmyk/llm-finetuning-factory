"""Versioned model registry: each run gets an immutable versioned directory.

Layout:
    registry/
      <model-slug>/
        v1/
          adapter/            # LoRA adapter weights (or full model)
          metadata.json       # config snapshot + metrics + lineage
          README.md           # auto-generated model card stub
        v2/
          ...
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "model"


class ModelRegistry:
    def __init__(self, root: str | Path = "./registry"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _model_dir(self, model_name: str) -> Path:
        return self.root / _slug(model_name)

    def next_version(self, model_name: str) -> str:
        model_dir = self._model_dir(model_name)
        existing = [
            p.name for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("v")
        ] if model_dir.exists() else []
        nums = [int(n[1:]) for n in existing if n[1:].isdigit()]
        return f"v{max(nums, default=0) + 1}"

    def register(
        self,
        model_name: str,
        artifact_dir: str | Path,
        metadata: Dict[str, Any],
        version: Optional[str] = None,
    ) -> Path:
        """Copy artifacts into a new versioned dir and write metadata."""
        version = version or self.next_version(model_name)
        dest = self._model_dir(model_name) / version
        if dest.exists():
            raise FileExistsError(f"Version {version} already registered for {model_name}")
        shutil.copytree(artifact_dir, dest / "adapter")

        full_metadata = {
            "model_name": model_name,
            "version": version,
            "registered_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        (dest / "metadata.json").write_text(json.dumps(full_metadata, indent=2))
        (dest / "README.md").write_text(
            f"# {model_name} {version}\n\n"
            f"Registered {full_metadata['registered_at']}.\n\n"
            f"See `metadata.json` for config, metrics, and lineage.\n"
        )
        return dest

    def list_versions(self, model_name: str) -> List[str]:
        model_dir = self._model_dir(model_name)
        if not model_dir.exists():
            return []
        versions = [p.name for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("v")]
        return sorted(versions, key=lambda v: int(v[1:]) if v[1:].isdigit() else 0)

    def get_metadata(self, model_name: str, version: str) -> Dict[str, Any]:
        path = self._model_dir(model_name) / version / "metadata.json"
        return json.loads(path.read_text())

    def latest(self, model_name: str) -> Optional[str]:
        versions = self.list_versions(model_name)
        return versions[-1] if versions else None
