"""Read configs/retrace.yaml, create a run id and store a copy of the config in the run folder."""
import shutil
import time
from pathlib import Path

import yaml

from src.utils import ROOT
from .common import RUNS_DIR, load_adv_config

DEFAULT_CONFIG = {
    "run": {"name": "pilot", "seed": 17},
    "models": {"graph": ["main", "no_cl", "no_mamba"], "text": []},
    "keep": [0.2, 0.4, 0.6, 0.8, 1.0],
    "eligible": {"tier": "lenient", "density_min": 2.0, "per_group_cap": 25, "pilot_n": 50,
                 "use_llm_cti": False},
    "search": {"strategy": "random", "n": 5, "greedy_max_items": 10},
    "llm": {"model": "qwen3:8b", "temperature": 0.7, "seed": 17, "think": False, "num_ctx": 4096},
    "validate": {"len_ratio": [0.7, 1.3], "nli": False, "judge": False},
    "insert_window": 128,
    "confident_tau": 0.5,
    "ops": {},
}


def get_config(path=None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path) if path else (ROOT / "configs" / "retrace.yaml")
    if p.exists():
        user = load_adv_config(p)
        cfg = _deep_merge(cfg, user)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def run_dir(name: str) -> Path:
    d = RUNS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def snapshot_config(cfg: dict, name: str):
    d = run_dir(name)
    cfg = dict(cfg)
    cfg["_snapshot_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(d / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return d
