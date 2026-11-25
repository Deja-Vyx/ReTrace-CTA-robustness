"""Text -> CSKG for ReTrace, using EXACTLY the repository's GraphBuilder + TTPService.

- AdvPipeline.build(doc_id, group, text) -> (graph, trace)
- AdvPipeline.make_doc(doc) -> Doc (rebuilt original + extra information for the operators)
- AdvPipeline.ttp_per_sentence(sents) -> TTP labels of EVERY sentence (needed by E2; the original
  function only keeps the first sentence of each label)
- CLI parity: rebuild N test documents and compare them with data/cskg (pipeline parity check)

    python -m retrace.pipeline parity --n 20
    python -m retrace.pipeline parity --n 20 --models main,no_cl,no_mamba   # also compares predictions
"""
import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from src.cskg_builder import NODE_TYPES, REL, GraphBuilder
from src.ioc_extract import extract_iocs
from src.utils import CSKG_DIR, OUTPUTS, load_config, read_jsonl, PROCESSED, sent_split, torch_load
from .common import CACHE_DIR, log
from .sentcache import CachedSentenceEncoder

STATE_T = NODE_TYPES.index("State")
TECH_T = NODE_TYPES.index("Technique")
TAC_T = NODE_TYPES.index("Tactic")
IOC_TYPES = [t for t in NODE_TYPES if t not in ("State", "Technique", "Tactic")]
IOC_T_IDS = {NODE_TYPES.index(t) for t in IOC_TYPES}


def join_sents(sents: list[str]) -> str:
    """Text after clean_text has no line breaks, so joining with one space reproduces the text exactly."""
    return " ".join(s.strip() for s in sents if s and s.strip())


# ---------------------------------------------------------------------- trace / graph comparison
def graph_trace(g: dict) -> dict:
    ioc = sorted({f"{NODE_TYPES[t]}:{str(v).lower()}" for t, v in zip(g["node_types"], g["node_values"])
                  if t in IOC_T_IDS})
    ttp = sorted({str(v) for t, v in zip(g["node_types"], g["node_values"]) if t in (TECH_T, TAC_T)})
    tech = sorted({str(v) for t, v in zip(g["node_types"], g["node_values"]) if t == TECH_T})
    return {"n_sents": int(g.get("n_sents", 0)), "n_states": len(g["states"]),
            "n_nodes": len(g["node_types"]), "n_edges": int(len(g["edges"])),
            "state_sents": [int(g["node_sent"][s]) for s in g["states"]],
            "ioc": ioc, "ttp": ttp, "tech": tech}


def _jacc(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def diff_trace(t0: dict, t1: dict) -> dict:
    return {"n_sents": t1["n_sents"], "n_states": t1["n_states"], "n_nodes": t1["n_nodes"],
            "n_edges": t1["n_edges"], "d_states": t1["n_states"] - t0["n_states"],
            "d_nodes": t1["n_nodes"] - t0["n_nodes"],
            "ioc_jacc": round(_jacc(t0["ioc"], t1["ioc"]), 4),
            "ttp_jacc": round(_jacc(t0["ttp"], t1["ttp"]), 4),
            "tech_jacc": round(_jacc(t0["tech"], t1["tech"]), 4),
            "ioc_lost": sorted(set(t0["ioc"]) - set(t1["ioc"]))[:20],
            "ioc_new": sorted(set(t1["ioc"]) - set(t0["ioc"]))[:20],
            "tech_lost": sorted(set(t0["tech"]) - set(t1["tech"]))[:20],
            "tech_new": sorted(set(t1["tech"]) - set(t0["tech"]))[:20]}


def _node_keys(g: dict) -> list:
    # GraphBuilder merges nodes by (type, value.lower()) -> this key identifies a node within a graph
    return [(int(t), str(v).lower()) for t, v in zip(g["node_types"], g["node_values"])]


def graph_struct_equal(a: dict, b: dict, atol: float = 1e-2,
                       strict_order: bool = False) -> tuple[bool, str]:
    """Compare two CSKGs by node IDENTITY rather than position.

    ioc-finder returns IOCs in an order that depends on Python's random hash seed (PYTHONHASHSEED),
    so two IOCs of the same type in one sentence may swap places between runs. The graph is then
    only a permutation of node indices: same nodes, same features, same edges -> GAT (mean/max
    pooling) and the State sequence give the same prediction. This function matches nodes by
    (type, value), maps edges/states/x through that permutation and then compares.
    strict_order=True keeps the old behaviour (order is compared too).
    """
    na, nb = len(a["node_types"]), len(b["node_types"])
    if na != nb:
        return False, f"node count differs ({na} vs {nb})"
    ka, kb = _node_keys(a), _node_keys(b)
    ca, cb = Counter(ka), Counter(kb)
    if ca != cb:
        only_orig = sorted(v for (_t, v) in (ca - cb).elements())[:3]
        only_rebuilt = sorted(v for (_t, v) in (cb - ca).elements())[:3]
        return False, f"node_values differ (only in original: {only_orig}, only in rebuilt: {only_rebuilt})"
    pos = defaultdict(list)
    for i, k in enumerate(ka):
        pos[k].append(i)
    perm = [pos[k].pop(0) for k in kb]          # perm[j] = index in a of node j of b
    reordered = perm != list(range(na))
    if strict_order and reordered:
        return False, "node order differs (strict_order)"
    for j, i in enumerate(perm):
        if a["node_sent"][i] != b["node_sent"][j]:
            return False, f"node_sent differs at {kb[j][1]!r}"
    if [perm[s] for s in b["states"]] != list(a["states"]):
        return False, "states differ"
    # co_occurs is semantically undirected (the builder links earlier IOC -> later IOC in a sentence,
    # so the direction follows IOC order); model._prep_edges also adds the reverse edge -> normalise (min,max).
    co = REL["co_occurs"]

    def _canon(u, v, r):
        return (min(u, v), max(u, v), r) if r == co else (u, v, r)

    ea = sorted(_canon(u, v, r) for u, v, r in a["edges"].tolist())
    eb = sorted(_canon(perm[u], perm[v], r) for u, v, r in b["edges"].tolist())
    if ea != eb:
        id2rel = {v: k for k, v in REL.items()}
        ca_, cb_ = Counter(ea), Counter(eb)
        lost = Counter(id2rel.get(r, r) for _u, _v, r in (ca_ - cb_).elements())
        new = Counter(id2rel.get(r, r) for _u, _v, r in (cb_ - ca_).elements())
        rels = sorted(set(lost) | set(new))
        detail = ", ".join(f"{r} -{lost.get(r, 0)}/+{new.get(r, 0)}" for r in rels)
        note = ""
        if rels == ["co_occurs"]:
            # the builder only links co_occurs for the FIRST 6 IOCs of a sentence (ids[:6]); for a
            # sentence with > 6 IOCs that set depends on ioc-finder's order -> a few edges differ,
            # which is not a ReTrace error
            note = "; due to the 6-IOC-per-sentence co_occurs limit"
        return False, f"edges differ ({len(ea)} vs {len(eb)}: {detail}{note})"
    if a["x"].shape != b["x"].shape:
        return False, "x shape differs"
    if a["x"].numel():
        xa = a["x"].float()[torch.tensor(perm, dtype=torch.long)]
        if float((xa - b["x"].float()).abs().max()) > atol:
            return False, "x differs > atol"
    return True, ("ok (node order permuted only)" if reordered else "ok")


# ---------------------------------------------------------------------- fake TTP extractor for CPU tests
class FakeTTP:
    """Keyword-based TTP extractor, ONLY for unit tests / CPU dry runs (THREATMAMBA_FAKE_ENCODER=1)."""
    RULES = [("phish", "T1566"), ("spearphish", "T1566.001"), ("powershell", "T1059.001"),
             ("command", "T1059"), ("macro", "T1204.002"), ("registry", "T1112"),
             ("rdp", "T1021.001"), ("lateral", "TA0008"), ("dns", "T1071.004"),
             ("mimikatz", "T1003"), ("credential", "TA0006"), ("scheduled task", "T1053.005"),
             ("exfiltrat", "TA0010"), ("backdoor", "T1505"), ("persistence", "TA0003"),
             ("dropper", "T1105"), ("download", "T1105"), ("encrypt", "T1486"),
             ("scan", "T1046"), ("inject", "T1055")]
    max_sents = 128

    def per_sentence(self, sents, emb=None) -> list[set]:
        out = []
        for s in sents[: self.max_sents]:
            low = s.lower()
            labs = [lab for kw, lab in self.RULES if kw in low][:5]
            out.append(set(labs))
        return out

    def extract(self, text, sents=None, emb=None):
        sents = sents if sents is not None else sent_split(text)
        best = {}
        for i, labs in enumerate(self.per_sentence(sents)):
            for lab in labs:
                best.setdefault(lab, (0.9, i))
        top = list(best.items())[:40]
        return {"ttps": [(lab, p) for lab, (p, _s) in top],
                "seq": sorted(((s, lab) for lab, (_p, s) in top)), "n_sents": len(sents)}


# ---------------------------------------------------------------------- Doc
@dataclass
class Doc:
    doc_id: str
    group: str
    text: str
    sents: list
    graph: dict
    trace: dict
    iocs: list = field(default_factory=list)          # [{value,type,sent_idx}]
    active: list = field(default_factory=list)        # indices of State sentences (in order)
    meta: dict = field(default_factory=dict)
    _ttp_sets: list | None = None

    def state_set(self) -> set:
        return set(self.active)


# ---------------------------------------------------------------------- pipeline
class AdvPipeline:
    def __init__(self, cfg=None, device=None, fake: bool | None = None, cache_path=None,
                 attck_override: dict | None = None, warm_docs: list | None = None):
        import os
        self.cfg = cfg or load_config()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fake = bool(os.getenv("THREATMAMBA_FAKE_ENCODER")) if fake is None else fake
        from src.encoder import FakeEncoder, get_encoder
        base = FakeEncoder() if self.fake else get_encoder(device=str(self.device))
        store = None if self.fake else (cache_path or (CACHE_DIR / "sent_emb.pkl"))
        self.enc = CachedSentenceEncoder(base, store=store)
        if warm_docs and not self.fake:
            n = self.enc.warm_from_doc_cache(warm_docs)
            log(f"[pipeline] preloaded embeddings from emb_cache for {n}/{len(warm_docs)} documents")
        if self.fake:
            self.ttp = FakeTTP()
        else:
            self.ttp = self._load_ttp_service()
        self.builder = GraphBuilder(encoder=self.enc,
                                    ttp_fn=lambda t, s, e=None: self.ttp.extract(t, s, e),
                                    mode="train")
        if attck_override:
            self.builder.tech2tac = attck_override.get("tech2tac", self.builder.tech2tac)
            self.builder.tac_order = attck_override.get("tac_order", self.builder.tac_order)
            self.builder.tacid2short = attck_override.get("tacid2short", self.builder.tacid2short)
        self.max_ttp_sents = int(self.cfg["ttp"].get("max_sents", 128))

    def _load_ttp_service(self):
        """Load TTPService but share the cached encoder (avoids loading SecureBERT twice)."""
        from src import ttp_extract as te
        ckpt = OUTPUTS / "ttp_extractor.pt"
        if not ckpt.exists():
            raise SystemExit("outputs/ttp_extractor.pt not found - run Stage 3 of the reproduction first")
        orig = te.get_encoder
        te.get_encoder = lambda device=None: self.enc
        try:
            svc = te.TTPService(device=str(self.device))
        finally:
            te.get_encoder = orig
        return svc

    # ---- build
    def build(self, doc_id: str, group: str, text: str) -> tuple[dict, dict]:
        g = self.builder.build({"doc_id": doc_id, "group": group, "text": text})
        return g, graph_trace(g)

    def make_doc(self, doc: dict) -> Doc:
        sents = sent_split(doc["text"])
        text = join_sents(sents)
        g, tr = self.build(doc["doc_id"], doc["group"], text)
        iocs = extract_iocs(text, sents)
        active = [int(g["node_sent"][s]) for s in g["states"]]
        return Doc(doc_id=doc["doc_id"], group=doc["group"], text=text, sents=sents, graph=g,
                   trace=tr, iocs=iocs, active=active)

    # ---- TTPs of every sentence
    def ttp_per_sentence(self, sents: list[str]) -> list[set]:
        sents = sents[: self.max_ttp_sents]
        if not sents:
            return []
        if self.fake:
            return self.ttp.per_sentence(sents)
        svc = self.ttp
        emb = self.enc.encode(sents)
        bs = int(svc.cfg.get("batch_size", 32))
        max_per_sent = int(svc.cfg.get("max_ttps_per_sent", 5))
        out: list[set] = [set() for _ in sents]
        with torch.no_grad():
            for i in range(0, len(sents), bs):
                x = emb[i:i + bs].float().unsqueeze(1).to(svc.device)
                mask = torch.ones(x.shape[0], 1, device=svc.device)
                probs = torch.sigmoid(svc.model(x, mask)[0]).float().cpu().numpy()
                for r in range(probs.shape[0]):
                    hit = np.nonzero(probs[r] >= svc._thr_vec)[0]
                    if len(hit) > max_per_sent:
                        hit = hit[np.argsort(-probs[r][hit])[:max_per_sent]]
                    out[i + r] = {svc.labels[j] for j in hit}
        return out

    def doc_ttp_sets(self, d: Doc) -> list[set]:
        if d._ttp_sets is None:
            d._ttp_sets = self.ttp_per_sentence(d.sents)
        return d._ttp_sets

    def technique_name(self, tid: str) -> str:
        return self._names().get(tid, "")

    def _names(self) -> dict:
        if not hasattr(self, "_tech_names"):
            from src.cskg_builder import _attck
            self._tech_names = {k: v.get("name", "") for k, v in _attck()["techniques"].items()}
        return self._tech_names

    def save_cache(self):
        self.enc.save()


# ---------------------------------------------------------------------- CLI: parity
def cmd_parity(args):
    cfg = load_config(args.config)
    docs = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == args.split]
    docs = docs[: args.n]
    pipe = AdvPipeline(cfg, device=args.device, warm_docs=docs)
    ok = 0
    rows = []
    pairs = []            # (doc, original graph, rebuilt graph) to compare predictions with --models
    for d in docs:
        fp = CSKG_DIR / f"{d['doc_id']}.pt"
        if not fp.exists():
            rows.append((d["doc_id"], "MISSING"))
            continue
        g0 = torch_load(fp)
        doc = pipe.make_doc(d)
        same, why = graph_struct_equal(g0, doc.graph)
        ok += int(same)
        rows.append((d["doc_id"], why))
        pairs.append((d, g0, doc.graph))
    for did, why in rows:
        print(f"  {did}: {why}")
    if args.models and pairs:
        # final criterion: does every model give the same prediction on the original and rebuilt graphs?
        from .models import ModelZoo
        tags = args.models.split(",")
        zoo = ModelZoo(tags, device=args.device)
        g2i = {g: i for i, g in enumerate(zoo.groups)}
        keeps = [0.2, 0.4, 0.6, 0.8, 1.0]
        s0 = zoo.score_graphs([p[1] for p in pairs], keeps)
        s1 = zoo.score_graphs([p[2] for p in pairs], keeps)
        print("  -- predictions data/cskg vs rebuilt (cut-offs 20..100%) --")
        pred_summary = {}
        for t in tags:
            dmax, flips, worst = 0.0, 0, None
            for k in keeps:
                a, b = s0[(t, k)], s1[(t, k)]
                diff = np.abs(a - b).max(axis=1)
                j = int(diff.argmax())
                if float(diff[j]) > dmax:
                    dmax, worst = float(diff[j]), f"{pairs[j][0]['doc_id']}@{int(k*100)}%"
                flips += int((a.argmax(1) != b.argmax(1)).sum())
            print(f"  {t}: max |dp| = {dmax:.2e} (largest at {worst}) | top-1 label changes = {flips}")
            pred_summary[t] = {"max_dp": dmax, "label_changes": flips, "worst": worst}
        args._pred_summary = pred_summary
    n_perm = sum(1 for _d, why in rows if "permuted" in why)
    print(f"PARITY: {ok}/{len(docs)} documents structurally identical to data/cskg "
          f"({n_perm} differ only in IOC node order - harmless) | cache {pipe.enc.stats()}")
    pipe.save_cache()
    need = len(docs) - 1
    passed = ok >= need and all(v["label_changes"] == 0 for v in getattr(args, "_pred_summary", {}).values())
    from .common import ADV_DIR
    ADV_DIR.mkdir(parents=True, exist_ok=True)
    (ADV_DIR / "parity.json").write_text(json.dumps({
        "n": len(docs), "struct_ok": ok, "n_permuted_only": n_perm, "passed": passed,
        "rows": rows, "pred": getattr(args, "_pred_summary", {})}, indent=1), encoding="utf-8")
    print("=> " + ("PASS" if passed else "FAIL - see the Troubleshooting section of docs/PIPELINE.md"))
    if not passed:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["parity"])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--models", default=None,
                    help="e.g. main,no_cl,no_mamba: also compare predictions on original vs rebuilt graphs (needs GPU)")
    args = ap.parse_args()
    {"parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main()
