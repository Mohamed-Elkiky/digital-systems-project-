"""Centralised path helpers — all directory references go through here."""

from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the repository root (two levels up from this file)."""
    return Path(__file__).resolve().parents[2]


def data_raw_dir(symbol: str | None = None) -> Path:
    d = project_root() / "data" / "raw"
    if symbol:
        d = d / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_processed_dir(symbol: str | None = None) -> Path:
    d = project_root() / "data" / "processed"
    if symbol:
        d = d / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d


def runs_dir() -> Path:
    d = project_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def configs_dir() -> Path:
    return project_root() / "configs"


def new_run_dir(tag: str = "") -> Path:
    """Create a timestamped run directory under runs/."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = f"{ts}_{tag}" if tag else ts
    d = runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    return d
