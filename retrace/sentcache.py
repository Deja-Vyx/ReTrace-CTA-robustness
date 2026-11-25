"""Sentence-level embedding cache (sha1 of the sentence -> 768-d float32 vector).

Why:
  - The repository's encode_doc_cached caches per DOCUMENT, so every variant would re-encode
    all of its sentences and write a new .pt file. With tens of thousands of variants this is
    slow and wastes disk space.
  - TTPs are scored sentence by sentence, so an unchanged sentence must get EXACTLY the same
    embedding as in the original. A per-sentence cache guarantees this (same tensor,
    independent of batching).

warm_from_doc_cache() preloads embeddings from data/processed/emb_cache (the {doc_id}_{hash}.pt
files written by Stage 4). Rebuilding an original document therefore reproduces the graph in
data/cskg BIT-FOR-BIT, and unchanged sentences need almost no GPU time.
"""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np

EMB_DIM = 768


def _key(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


class CachedSentenceEncoder:
    is_fake = False
    skip_disk_cache = True          # encode_doc_cached then calls encode() directly (see src/encoder.py)
    dim = EMB_DIM

    def __init__(self, base, store: Path | None = None, autosave_every: int = 20000):
        self.base = base
        self.store = Path(store) if store else None
        self.mem: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._new = 0
        self.autosave_every = autosave_every
        if self.store and self.store.exists():
            self.load()

    # ------------------------------------------------------------------ io
    def load(self):
        with open(self.store, "rb") as f:
            data = pickle.load(f)
        self.mem.update(data)

    def save(self):
        if not self.store or self._new == 0:
            return
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.mem, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.store)
        self._new = 0

    # ------------------------------------------------------------------ encode
    def encode(self, sents: list[str]):
        import torch
        if not sents:
            return torch.zeros(0, EMB_DIM)
        keys = [_key(s) for s in sents]
        todo: dict[str, str] = {}
        for k, s in zip(keys, sents):
            if k in self.mem:
                self.hits += 1
            elif k not in todo:
                todo[k] = s
        if todo:
            self.misses += len(todo)
            ks = list(todo)
            emb = self.base.encode([todo[k] for k in ks]).float().cpu().numpy()
            for k, row in zip(ks, emb):
                self.mem[k] = row.astype(np.float32)
            self._new += len(ks)
            if self.store and self._new >= self.autosave_every:
                self.save()
        arr = np.stack([self.mem[k] for k in keys]).astype(np.float32)
        return torch.from_numpy(arr)

    def has(self, s: str) -> bool:
        return _key(s) in self.mem

    def stats(self) -> dict:
        tot = self.hits + self.misses
        return {"cached": len(self.mem), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / tot, 4) if tot else None}

    # ------------------------------------------------------------------ warm
    def warm_from_doc_cache(self, docs: list[dict], cache_dir=None, max_sents: int = 256) -> int:
        """Load embeddings from the repository's per-document cache (Stage 4). Returns the
        number of documents loaded."""
        from src.encoder import CACHE
        from src.utils import sent_split, torch_load
        cdir = Path(cache_dir) if cache_dir else CACHE
        n = 0
        if not cdir.exists():
            return 0
        for d in docs:
            sents = sent_split(d["text"])[:max_sents]
            if not sents:
                continue
            key16 = hashlib.sha1("\n".join(sents).encode("utf-8", "ignore")).hexdigest()[:16]
            fp = cdir / f"{d['doc_id']}_{key16}.pt"
            if not fp.exists():
                continue
            emb = torch_load(fp, weights_only=True).float().cpu().numpy()
            if emb.shape[0] != len(sents):
                continue
            for s, row in zip(sents, emb):
                k = _key(s)
                if k not in self.mem:
                    self.mem[k] = row.astype(np.float32)
                    self._new += 1
            n += 1
        return n
