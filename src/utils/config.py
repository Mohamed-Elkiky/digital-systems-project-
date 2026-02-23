"""YAML configuration loader with merging."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import configs_dir


# ---------------------------------------------------------------------------
# Deep-merge helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(algo_config: str | Path | None = None) -> dict[str, Any]:
    """Load base.yaml and optionally merge with an algorithm-specific config."""
    base = load_yaml(configs_dir() / "base.yaml")
    if algo_config is not None:
        algo_path = Path(algo_config)
        if not algo_path.is_absolute():
            algo_path = configs_dir() / algo_path
        override = load_yaml(algo_path)
        base = _deep_merge(base, override)
    return base


def save_config_snapshot(cfg: dict[str, Any], run_dir: Path) -> Path:
    """Save the resolved config dict as YAML inside the run directory."""
    out = run_dir / "config.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return out
