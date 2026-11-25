"""Shared utilities for ReTrace: paths, config, JSONL(.gz), hashing, logging."""
import gzip
import hashlib
import json
import time
from pathlib import Path

from src.utils import OUTPUTS, ROOT

ADV_DIR = OUTPUTS / "retrace"
RUNS_DIR = ADV_DIR / "runs"
CACHE_DIR = ADV_DIR / "cache"
POOLS_DIR = ADV_DIR / "pools"
BASELINE_DIR = ADV_DIR / "baselines"
HUMAN_DIR = ADV_DIR / "human"
ADV_CONFIG = ROOT / "configs" / "retrace.yaml"

KEEPS_DEFAULT = [0.2, 0.4, 0.6, 0.8, 1.0]


def load_adv_config(path=None) -> dict:
    import yaml
    p = Path(path) if path else ADV_CONFIG
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def sha256_file(path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def open_any(path, mode="rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding=None if "b" in mode else "utf-8")
    return open(path, mode, encoding=None if "b" in mode else "utf-8")


def read_jsonl_any(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open_any(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # the last line may be truncated by an interrupted run - skip it
                    continue
    return rows


def iter_jsonl_any(path):
    p = Path(path)
    if not p.exists():
        return
    with open_any(p, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


class JsonlAppender:
    """Append-only JSONL writer (optionally .gz). Every flush closes the file so that an
    interrupted run can be resumed safely."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buf: list[str] = []

    def add(self, row: dict):
        self.buf.append(json.dumps(row, ensure_ascii=False))

    def flush(self):
        if not self.buf:
            return
        with open_any(self.path, "at") as f:
            f.write("\n".join(self.buf) + "\n")
        self.buf = []


def param_key(params: dict) -> str:
    """Stable string that identifies a parameter set; used in variant ids and when grouping tables."""
    if not params:
        return "-"
    parts = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, (list, tuple)):
            v = "+".join(str(x) for x in v)
        parts.append(f"{k}={v}")
    return ",".join(parts)


def expand_grid(spec: dict) -> list[dict]:
    """{'k': [1, 3], 'mode': 'x'} -> [{'k':1,'mode':'x'}, {'k':3,'mode':'x'}]."""
    import itertools
    if not spec:
        return [{}]
    keys = sorted(spec)
    vals = [spec[k] if isinstance(spec[k], list) else [spec[k]] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def __str__(self):
        s = int(time.time() - self.t0)
        return f"{s // 3600:d}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def log(msg: str, fp=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fp is not None:
        Path(fp).parent.mkdir(parents=True, exist_ok=True)
        with open(fp, "a", encoding="utf-8") as f:
            f.write(line + "\n")
