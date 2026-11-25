"""Mitigation probe: build CSKGs from text in which every GROUP NAME is masked, so that a model
trained on them can no longer "read the name". This separates the accuracy that comes from
real behaviour from the accuracy that comes from group names.

    python -m retrace.mitigate build-masked                 # -> data/cskg_masked/*.pt
    # then:
    python -m src.train    --cskg-dir data/cskg_masked --suffix masked
    python -m src.evaluate main_masked --cskg-dir data/cskg_masked --robustness

Compare macro-F1 and the robustness curve of main_masked with main: the drop is the share of
the model that relied on group names.
"""
import argparse

import torch

from src.utils import PROCESSED, load_json, read_jsonl
from .common import Timer, log
from .pipeline import AdvPipeline
from .runner import mask_names_text


def build_masked(device=None, split="all", limit=None):
    from pathlib import Path
    from src.utils import CSKG_DIR
    out_dir = Path(str(CSKG_DIR) + "_masked")
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = load_json(PROCESSED / "groups.json")
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    if split != "all":
        docs = [d for d in docs if d["split"] == split]
    if limit:
        docs = docs[:limit]
    pipe = AdvPipeline(device=device, warm_docs=docs)
    t = Timer()
    for k, d in enumerate(docs):
        masked = mask_names_text(d["text"], groups)
        g, _ = pipe.build(d["doc_id"], d["group"], masked)
        torch.save(g, out_dir / f"{d['doc_id']}.pt")
        if (k + 1) % 200 == 0:
            log(f"[mitigate] {k+1}/{len(docs)} | {t}")
    pipe.save_cache()
    log(f"[mitigate] done: {len(docs)} name-masked graphs -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build-masked"])
    ap.add_argument("--split", default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    build_masked(device=args.device, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()
