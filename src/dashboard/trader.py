"""Model loading helpers for the dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR     = PROJECT_ROOT / "runs"


def list_runs() -> list[dict]:
    if not RUNS_DIR.exists():
        return []
    result = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        name  = d.name
        parts = name.split("_")
        algo  = next((p for p in parts if p in ("dqn", "ppo", "qlearning")), "unknown")
        raw_date = parts[0] if parts else ""
        date_fmt = (
            f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
            if len(raw_date) >= 8 else raw_date
        )
        result.append({
            "name":  name,
            "algo":  algo.upper(),
            "date":  date_fmt,
            "label": f"{algo.upper()} · {date_fmt}",
            "path":  d,
        })
    return result


def _detect_algo(run_dir: Path) -> str:
    name = run_dir.name.lower()
    if "qlearning" in name:
        return "qlearning"
    if "ppo" in name:
        return "ppo"
    return "dqn"


def load_model(run_dir: Path) -> dict[str, Any]:
    algo      = _detect_algo(run_dir)
    model_dir = run_dir / "model"

    if algo == "qlearning":
        from src.agents.qlearning import QLearningAgent
        agent = QLearningAgent.load(model_dir)
        meta  = json.loads((model_dir / "meta.json").read_text())
        return {"algo": "qlearning", "agent": agent, "meta": meta, "run": run_dir.name}

    if algo == "dqn":
        meta = json.loads((model_dir / "meta.json").read_text())
        from src.agents.dqn import DQNAgent
        agent = DQNAgent.load(model_dir, obs_dim=meta["obs_dim"])
        return {"algo": "dqn", "agent": agent, "meta": meta, "run": run_dir.name}

    from src.agents.ppo import load_ppo
    model  = load_ppo(run_dir)
    meta_p = run_dir / "ppo_meta.json"
    meta   = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {"algo": "ppo", "agent": model, "meta": meta, "run": run_dir.name}
