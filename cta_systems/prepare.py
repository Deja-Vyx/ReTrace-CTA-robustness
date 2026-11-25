"""Shared preparation for both case-study systems: rebuild ThreatMAMBA's CSKG for every text to obtain

  1. timeline cut-offs: index of the LAST sentence kept at each 20/40/60/80% cut-off (identical to
     cskg_builder.truncate_graph and retrace/models.text_cut), so text models are scored at the
     EXACT cut position ThreatMAMBA sees;
  2. the ATT&CK techniques that ThreatMAMBA's TTP extractor assigns to the text (with the first
     sentence in which each occurs) - the input features of TTP-Vec (Hussain & Petrova).

Records are cached by sha1(text) in outputs/cta_systems/cache/prepared.jsonl.gz (resumable).
Per-sentence SecureBERT embeddings are preloaded from the ReTrace cache (read-only) so that the
rebuilt graphs match the graphs the ReTrace runs scored.
"""
import os
import pickle

from src.cskg_builder import NODE_TYPES
from src.utils import CSKG_DIR, sent_split, torch_load

from .common import CACHE_DIR, KEEPS, ADV_DIR, JsonlAppender, iter_jsonl_any, log, sha1

TECH_T = NODE_TYPES.index("Technique")
PREP_PATH = CACHE_DIR / "prepared.jsonl.gz"


def join_sents(sents: list[str]) -> str:
    # same as retrace/pipeline.join_sents (text after clean_text has no line breaks)
    return " ".join(s.strip() for s in sents if s and s.strip())


def cut_sent(g: dict, keep: float):
    """Index of the last sentence kept at cut-off `keep` (None = full text). Formula of truncate_graph."""
    states = g["states"]
    if keep >= 1.0 or not states:
        return None
    k = max(1, int(round(len(states) * keep)))
    return int(g["node_sent"][states[k - 1]])


def rec_from_graph(g: dict) -> dict:
    tech = sorted({(str(v), int(s)) for t, v, s in zip(g["node_types"], g["node_values"], g["node_sent"])
                   if t == TECH_T}, key=lambda x: (x[1], x[0]))
    return {"n_sents": int(g.get("n_sents", 0)), "n_states": len(g["states"]),
            "cut": [cut_sent(g, k) for k in KEEPS], "tech": [list(x) for x in tech]}


def techs_at(rec: dict, keep: float) -> set:
    """Techniques ThreatMAMBA extracts from the part of the text that remains at cut-off `keep`."""
    i = KEEPS.index(keep)
    c = rec["cut"][i]
    return {t for t, s in rec["tech"] if c is None or s <= c}


def sents_at(sents: list[str], rec: dict, keep: float) -> list[str]:
    i = KEEPS.index(keep)
    c = rec["cut"][i]
    return sents if c is None else sents[: c + 1]


class Preparer:
    """Cache {sha1(text): record}. The ThreatMAMBA pipeline is only loaded when a new text appears."""

    def __init__(self, device=None, path=PREP_PATH):
        self.device = device
        self.path = path
        self.mem: dict[str, dict] = {}
        for r in iter_jsonl_any(path):
            self.mem[r["key"]] = {k: v for k, v in r.items() if k != "key"}
        self._app = JsonlAppender(path)
        self._pipe = None
        self.n_built = 0          # graphs that had to be REBUILT (needs pipeline/GPU)
        self.n_cskg = 0           # records taken from data/cskg/*.pt (no GPU needed)
        self._fast: dict[str, set] = {}

    # ------------------------------------------------------------------ pipeline (lazy)
    @property
    def pipe(self):
        if self._pipe is None:
            from retrace.pipeline import AdvPipeline
            from .common import load_docs
            fake = bool(os.getenv("THREATMAMBA_FAKE_ENCODER"))
            self._pipe = AdvPipeline(cfg=None, device=self.device,
                                     cache_path=CACHE_DIR / "sent_emb_securebert.pkl",
                                     warm_docs=None if fake else load_docs())
            orig = ADV_DIR / "cache" / "sent_emb.pkl"
            if not fake and orig.exists():
                # load the ReTrace sentence cache (READ-ONLY) -> rebuilt graphs match the original run
                with open(orig, "rb") as f:
                    data = pickle.load(f)
                # the ReTrace cache takes priority (as in the original Runner: the store is loaded
                # first, warm_from_doc_cache only adds missing sentences) -> bit-for-bit identical embeddings
                self._pipe.enc.mem.update(data)
                log(f"[prepare] loaded {len(data)} sentence embeddings from {orig} (read-only)")
        return self._pipe

    # ------------------------------------------------------------------ api
    def get(self, text: str) -> dict:
        key = sha1(text)
        rec = self.mem.get(key)
        if rec is None:
            g, _tr = self.pipe.build("cta_systems", "?", text)
            rec = rec_from_graph(g)
            self.n_built += 1
            self._put(key, rec)
        return rec

    def techs_fast(self, text: str) -> set:
        """Techniques at the 100% cut-off WITHOUT building the graph (skips IOC extraction, the slowest
        part). Used by the adaptive search for TTP-Vec: GraphBuilder creates Technique nodes from the
        non-TA labels in ttp['seq'], so the result equals techs_at(get(text), 1.0) (checked in tests/test_cta_systems.py)."""
        key = sha1(text)
        if key in self.mem:
            return techs_at(self.mem[key], 1.0)
        ts = self._fast.get(key)
        if ts is None:
            sents = sent_split(text)[: self.pipe.builder.max_sents]
            emb = self.pipe.enc.encode(sents).float()
            ttp = self.pipe.ttp.extract(text, sents, emb) if sents else {"seq": []}
            ts = {lab for _s, lab in ttp["seq"] if not lab.startswith("TA")}
            if len(self._fast) > 50000:
                self._fast.clear()
            self._fast[key] = ts
        return ts

    def get_doc(self, doc: dict) -> tuple[str, dict]:
        """Original train/test document: prefer the graph stored in data/cskg (no GPU needed)."""
        text = join_sents(sent_split(doc["text"]))
        key = sha1(text)
        rec = self.mem.get(key)
        if rec is None:
            fp = CSKG_DIR / f"{doc['doc_id']}.pt"
            if fp.exists() and not os.getenv("THREATMAMBA_FAKE_ENCODER"):
                rec = rec_from_graph(torch_load(fp))
                rec["src"] = "cskg"
                self.n_cskg += 1
                self._put(key, rec)
            else:
                rec = self.get(text)
        return text, rec

    def _put(self, key: str, rec: dict):
        self.mem[key] = rec
        self._app.add({"key": key, **rec})
        if len(self._app.buf) >= 200:
            self.flush()

    def flush(self):
        """Write only the new records (small). The sentence-embedding cache (large) is written by save()."""
        self._app.flush()

    def save(self):
        """Write the per-sentence SecureBERT embedding cache (large file) - call at the end of a step / when interrupted."""
        self._app.flush()
        if self._pipe is not None:
            self._pipe.save_cache()
