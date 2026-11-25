"""Local embedding models replacing OpenAI text-embedding-3-large (original TTP-Match code).

- HFEmbedder: transformers AutoModel (default BAAI/bge-large-en-v1.5), CLS or mean pooling,
  L2-normalised -> dot product = cosine (like np.matmul(embeddings_matrix, query) in the original code).
- FakeEmbedder: ONLY for CPU tests (CTA_SYSTEMS_FAKE_EMB=1 or THREATMAMBA_FAKE_ENCODER=1):
  hashed bag of words -> sentences sharing vocabulary get a high cosine, enough to test the matching logic.
- CachedEmbedder: cache keyed by sha1(text) (float16) in outputs/cta_systems/cache/, because variants
  change only a few sentences, so most windows are identical to the original.
"""
import hashlib
import os
import pickle
import re

import numpy as np

from ..common import CACHE_DIR, log


class HFEmbedder:
    def __init__(self, name: str, device=None, pooling: str = "cls", max_length: int = 512,
                 batch_size: int = 64, fp16: bool = True, prefix: str = ""):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).eval().to(self.device)
        if fp16 and str(self.device).startswith("cuda"):
            self.model = self.model.half()
        self.pooling = pooling
        self.max_length = max_length
        self.batch_size = batch_size
        self.prefix = prefix or ""
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        # sort by length to reduce padding, then restore the original order
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.no_grad():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                batch = [self.prefix + texts[j][:20000] for j in idx]
                enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_length,
                               return_tensors="pt").to(self.device)
                hid = self.model(**enc).last_hidden_state
                if self.pooling == "mean":
                    m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
                    e = (hid * m).sum(1) / m.sum(1).clamp(min=1)
                else:
                    e = hid[:, 0]
                e = torch.nn.functional.normalize(e.float(), dim=-1)
                out.append((idx, e.cpu().numpy()))
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, e in out:
            arr[idx] = e
        return arr


class FakeEmbedder:
    """Hashing bag-of-words (dim 256), deterministic. Not for real results."""
    name = "fake-hashing-256"
    dim = 256
    _tok = re.compile(r"[a-z0-9]+")

    def encode(self, texts: list[str]) -> np.ndarray:
        arr = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in self._tok.findall(t.lower()):
                if len(w) < 3:
                    continue
                h = int.from_bytes(hashlib.sha1(w.encode()).digest()[:4], "little")
                arr[i, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
            n = np.linalg.norm(arr[i])
            if n > 0:
                arr[i] /= n
        return arr


class CachedEmbedder:
    def __init__(self, base, store=None, autosave_every: int = 50000, tag: str = ""):
        self.base = base
        self.name = getattr(base, "name", "?")
        self.dim = base.dim
        # the cache file name includes the settings (pooling, max_length, fp16) -> changing them never reuses a stale cache
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{self.name}_{tag}")
        self.store = store or (CACHE_DIR / f"ttp_match_emb_{slug}.pkl")
        self.mem: dict[str, np.ndarray] = {}
        self._new = 0
        self.hits = self.misses = 0
        self.autosave_every = autosave_every
        if self.store.exists():
            with open(self.store, "rb") as f:
                self.mem = pickle.load(f)

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode("utf-8", "ignore")).hexdigest() for t in texts]
        todo = {}
        for k, t in zip(keys, texts):
            if k in self.mem:
                self.hits += 1
            elif k not in todo:
                todo[k] = t
        if todo:
            self.misses += len(todo)
            ks = list(todo)
            emb = self.base.encode([todo[k] for k in ks])
            for k, e in zip(ks, emb):
                self.mem[k] = e.astype(np.float16)
            self._new += len(ks)
            if self._new >= self.autosave_every:
                self.save()
        if not keys:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.mem[k] for k in keys]).astype(np.float32)

    def save(self):
        if self._new == 0:
            return
        self.store.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self.mem, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.store)
        self._new = 0

    def stats(self) -> dict:
        return {"cached": len(self.mem), "hits": self.hits, "misses": self.misses}


def is_fake() -> bool:
    return bool(os.getenv("CTA_SYSTEMS_FAKE_EMB") or os.getenv("THREATMAMBA_FAKE_ENCODER"))


def embedder_settings(gcfg: dict) -> dict:
    """Settings that determine the embedding VALUES (stored in ttp_match.pkl, checked when reloading)."""
    if is_fake():
        return {"embedder": FakeEmbedder.name, "pooling": "-", "max_length": 0, "fp16": False}
    return {"embedder": gcfg["embedder"], "pooling": gcfg.get("pooling", "cls"),
            "max_length": int(gcfg.get("max_length", 512)), "fp16": bool(gcfg.get("fp16", True))}


def get_embedder(gcfg: dict, device=None, settings: dict | None = None) -> CachedEmbedder:
    """settings (from ttp_match.pkl) take priority over the config -> scoring uses EXACTLY the training settings."""
    st = settings or embedder_settings(gcfg)
    if is_fake():
        return CachedEmbedder(FakeEmbedder(), tag="fake")
    log(f"[ttp_match] loading embedding model {st['embedder']} (pooling={st['pooling']}, max_length={st['max_length']}, "
        f"fp16={st['fp16']}) ...")
    base = HFEmbedder(st["embedder"], device=device, pooling=st["pooling"], max_length=int(st["max_length"]),
                      batch_size=int(gcfg.get("batch_size", 64)), fp16=bool(st["fp16"]))
    return CachedEmbedder(base, tag=f"{st['pooling']}_{st['max_length']}_{'fp16' if st['fp16'] else 'fp32'}")
