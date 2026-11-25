"""Execute one run: for every eligible document, generate variants -> validity checks -> rebuild
the CSKG -> score every model at 5 timeline cut-offs -> write JSONL. Resumable (existing vids are skipped).

    python -m retrace.runner --run pilot --docs pilot \\
        --ops ID,T3,T6,E1,E4,E7,E8,G1,G3,G5 --strategy random --models main,no_cl,no_mamba

Outputs in outputs/retrace/runs/<run>/:
  variants.jsonl.gz  : every variant (text / parameters / check results / trace)
  scores.jsonl.gz    : one row per (variant, model, cut-off)
"""
import argparse
import gzip
import random

import numpy as np

from src.cskg_builder import truncate_graph
from src.utils import PROCESSED, load_json, read_jsonl
from .common import ADV_DIR, JsonlAppender, Timer, iter_jsonl_any, log, param_key, sha1
from .config import get_config, run_dir, snapshot_config
from .models import ModelZoo, preprocess_of, summarize_probs, text_cut
from .operators import P0_OPS, get_op
from .operators.base import Ctx
from .pipeline import AdvPipeline, diff_trace, join_sents
from .validate import check_variant


def _load_docs(which: str) -> list[str]:
    if which == "pilot":
        fp = ADV_DIR / "pilot.txt"
    elif which == "eligible":
        return [r["doc_id"] for r in iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl")]
    else:
        fp = ADV_DIR / which
    return [l.strip() for l in open(fp, encoding="utf-8") if l.strip() and not l.startswith("#")]


def _elig_meta() -> dict:
    return {r["doc_id"]: r for r in iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl")}


def mask_names_text(text: str, groups) -> str:
    from .lexicons import find_group_mentions, mask_spans
    spans = []
    for g in groups:
        spans += find_group_mentions(text, g)
    spans = sorted(set(spans))
    clean, last = [], -1
    for s, e, a in spans:
        if s >= last:
            clean.append((s, e, a)); last = e
    return mask_spans(text, clean, "the threat actor")


class Runner:
    def __init__(self, cfg, run_name, models, device=None, strategy="random", search_n=5,
                 no_llm=False):
        self.cfg = cfg
        self.run_name = run_name
        self.dir = run_dir(run_name)
        self.strategy = strategy
        self.search_n = search_n
        self.keeps = cfg.get("keep", [0.2, 0.4, 0.6, 0.8, 1.0])
        self.groups = load_json(PROCESSED / "groups.json")
        self.g2i = {g: i for i, g in enumerate(self.groups)}
        self.zoo = ModelZoo(models, device=device, groups=self.groups)
        docs_all = read_jsonl(PROCESSED / "docs.jsonl")
        self.docs_by_id = {d["doc_id"]: d for d in docs_all}
        self._gcache = {}
        self.pipe = AdvPipeline(cfg=None, device=device, warm_docs=docs_all)
        from .pools import Pools
        self.pools = Pools(self.pipe)
        self.ctx = Ctx(pipeline=self.pipe, cfg=cfg, pools=self.pools, zoo=self.zoo,
                       groups=self.groups)
        if not no_llm:
            self.ctx.llm = None  # created on demand
        self.elig = _elig_meta()
        self._donors_built = False
        self.v_out = JsonlAppender(self.dir / "variants.jsonl.gz")
        self.s_out = JsonlAppender(self.dir / "scores.jsonl.gz")
        self.done = self._load_done()

    def _load_done(self) -> set:
        done = set()
        fp = self.dir / "scores.jsonl.gz"
        if fp.exists():
            for r in iter_jsonl_any(fp):
                done.add((r["vid"], r["model"], r["keep"]))
        return done

    def build_donors(self, doc_ids):
        """For G4: a few sample Docs per group (from training documents)."""
        if self._donors_built:
            return
        train = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == "train"]
        by_g = {}
        for d in train:
            by_g.setdefault(d["group"], []).append(d)
        donors = {}
        for g, ds in by_g.items():
            donors[g] = [self.pipe.make_doc(d) for d in ds[:2]]
        self.ctx.donors = donors
        self._donors_built = True

    # ---- score one variant with every model x cut-off
    def _preprocessed_graph(self, doc, variant, preprocess: str):
        if variant.graph is not None:
            return variant.graph, None, None
        text = variant.text
        sents_full = None
        if preprocess == "mask_names":
            text = mask_names_text(text, self.groups)
        key = (sha1(text), preprocess)
        if key in self._gcache:
            return self._gcache[key]
        g, tr = self.pipe.build(f"{doc.doc_id}|{variant.op}|{variant.cand}", doc.group, text)
        from src.utils import sent_split
        sents_full = sent_split(text)
        if len(self._gcache) > 512:
            self._gcache.clear()
        self._gcache[key] = (g, tr, sents_full)
        return g, tr, sents_full

    def score_variant(self, doc, v, write=True) -> dict:
        """Return {(tag,keep): probs_row}. If write, also append to scores.jsonl.gz."""
        y = self.g2i[doc.group]
        target = self.g2i.get(v.meta.get("target")) if v.meta else None
        results = {}
        graphs_by_pp = {}
        sents_by_pp = {}
        for pp, tags in self.zoo.preprocess_groups().items():
            g, tr, sents = self._preprocessed_graph(doc, v, pp)
            graphs_by_pp[pp] = g
            sents_by_pp[pp] = sents
            if pp == "none":
                v.meta = dict(v.meta or {})
                v.meta.setdefault("trace", tr if tr is not None else None)
            # build the graph once, score every model of this preprocess group at every cut-off
            sc = self.zoo.score_graphs([g], self.keeps, tags=tags)
            for (t, keep), arr in sc.items():
                results[(t, keep)] = arr[0]
        # text baselines (text cut exactly at the sentence truncate_graph uses as the cut-off)
        if self.zoo.text_tags:
            base_g = graphs_by_pp.get("none")
            from src.utils import sent_split
            sents = sents_by_pp.get("none") or (sent_split(v.text) if v.text else doc.sents)
            texts_by_keep = {keep: [text_cut(sents, base_g, keep)] for keep in self.keeps} \
                if base_g is not None else {keep: [v.text or join_sents(doc.sents)] for keep in self.keeps}
            tprobs = self.zoo.score_texts(texts_by_keep)
            for (t, keep), arr in tprobs.items():
                results[(t, keep)] = arr[0]
        if write:
            for (t, keep), ps in results.items():
                row = {"vid": v.meta.get("vid"), "model": t, "keep": keep,
                       **summarize_probs(ps, y, target)}
                self.s_out.add(row)
        return results

    def score_one(self, doc, v, model, keep):
        """Score one model at one cut-off (used during search; ~15x cheaper than score_variant)."""
        if model in self.zoo.text_tags:
            return self.score_variant(doc, v, write=False)[(model, keep)]
        g, tr, _s = self._preprocessed_graph(doc, v, preprocess_of(model))
        if v.graph is None and preprocess_of(model) == "none":
            v.meta = dict(v.meta or {}); v.meta.setdefault("trace", tr)
        return self.zoo.score_graphs([g], [keep], tags=[model])[(model, keep)][0]

    def p_true(self, doc, v, model, keep) -> float:
        return float(self.score_one(doc, v, model, keep)[self.g2i[doc.group]])

    # ---- run one operator on one document
    def run_op(self, doc, op_code, params, rng):
        op = get_op(op_code, self.ctx)
        if op_code == "G4":
            self.build_donors(None)
        search_model = self.zoo.graph_tags[0]
        search_keep = 1.0
        cands = op.candidates(doc, params, self.search_n if self.strategy != "random" else 1, rng)
        variants = []
        if not cands:
            return 0
        if op.expect == "TGT":
            from .search import pick_targeted

            def _pair(vv):
                ps = self.score_one(doc, vv, search_model, search_keep)
                return (float(ps[self.g2i[doc.group]]),
                        float(ps[self.g2i.get(vv.meta.get("target"), self.g2i[doc.group])]))
            variants = pick_targeted(cands, _pair)
        elif self.strategy == "greedy" and op.item_based:
            from .search import greedy_k
            k_max = int(params.get("k", 3)) if str(params.get("k", "")).isdigit() else 3
            variants = greedy_k(op, doc, k_max,
                                lambda vv: self.p_true(doc, vv, search_model, search_keep),
                                rng, self.cfg.get("search", {}).get("greedy_max_items", 10),
                                base_params=params)
        elif self.strategy in ("best_of_n", "greedy") and op_code not in ("ID",):
            # greedy for non item-based operators -> best_of_n over N candidates
            from .search import pick_best_of_n
            variants = pick_best_of_n(cands, lambda vv: self.p_true(doc, vv, search_model, search_keep))
        else:
            variants = cands[:1] if op_code == "ID" else cands

        n = 0
        for v in variants:
            v.meta = dict(v.meta or {})
            vid = f"{doc.doc_id}|{v.op}|{param_key(v.params)}|c{v.cand}"
            v.meta["vid"] = vid
            if all((vid, t, k) in self.done for t in self.zoo.tags for k in self.keeps):
                continue
            val = check_variant(join_sents(doc.sents), v, self.cfg) if v.text is not None \
                else {"passed": True, "checks": {"graph_level": True}}
            trace = None
            if v.text is not None:
                g0 = doc.trace
                _res = self.score_variant(doc, v, write=True)
                tr1 = v.meta.get("trace")
                trace = diff_trace(g0, tr1) if tr1 else None
            else:
                self.score_variant(doc, v, write=True)
            rec = {"run": self.run_name, "vid": vid, "doc_id": doc.doc_id, "group": doc.group,
                   "op": v.op, "level": v.level, "expect": v.expect, "params": v.params,
                   "strategy": v.strategy, "cand": v.cand, "aligned": v.aligned,
                   "runner_up": self.elig.get(doc.doc_id, {}).get("runner_up"),
                   "valid": val, "trace": trace,
                   "edits": v.edits[:8], "meta": {k: vv for k, vv in v.meta.items()
                                                  if k not in ("trace",)}}
            if v.text is not None:
                rec["text"] = v.text
            self.v_out.add(rec)
            for t in self.zoo.tags:
                for k in self.keeps:
                    self.done.add((vid, t, k))
            n += 1
        return n

    def run(self, doc_ids, ops, params_per_op):
        t = Timer()
        # write the ORIG document once
        for i, did in enumerate(doc_ids):
            doc_raw = self.docs_by_id.get(did)
            if doc_raw is None:
                continue
            self._gcache.clear()
            doc = self.pipe.make_doc(doc_raw)
            doc.meta["runner_up"] = self.elig.get(did, {}).get("runner_up")
            # ORIG
            orig_vid = f"{did}|ORIG|-|c0"
            if not all((orig_vid, tg, k) in self.done for tg in self.zoo.tags for k in self.keeps):
                from .operators.base import Variant
                ov = Variant(doc_id=did, group=doc.group, op="ORIG", level=0, expect="INV",
                             params={}, text=join_sents(doc.sents))
                ov.meta = {"vid": orig_vid}
                self.score_variant(doc, ov, write=True)
                self.v_out.add({"run": self.run_name, "vid": orig_vid, "doc_id": did, "group": doc.group,
                                "op": "ORIG", "text": join_sents(doc.sents),
                                "runner_up": doc.meta["runner_up"], "trace": doc.trace})
            # Python's hash() changes per process (PYTHONHASHSEED) -> use sha1 so that every
            # run/resume produces the same variants for the same (seed, document)
            rng = random.Random(int(sha1(f"{self.cfg['run']['seed']}|{did}")[:8], 16))
            for op_code in ops:
                plist = params_per_op.get(op_code, [{}])
                if self.strategy == "greedy":
                    plist = _greedy_grid(op_code, plist)
                for params in plist:
                    try:
                        self.run_op(doc, op_code, params, rng)
                    except Exception as e:  # noqa: BLE001
                        log(f"[runner] ERROR {op_code} {params} on {did}: {type(e).__name__}: {e}",
                            fp=self.dir / "log.txt")
            self.v_out.flush(); self.s_out.flush()
            if (i + 1) % 5 == 0:
                log(f"[runner] {i+1}/{len(doc_ids)} documents | {t} | cache {self.pipe.enc.stats()}",
                    fp=self.dir / "log.txt")
        self.pipe.save_cache()
        log(f"[runner] done: {len(doc_ids)} documents in {t}", fp=self.dir / "log.txt")


def _greedy_grid(op_code, plist):
    """Greedy only needs to run ONCE per combination of the non-k parameters, with k = the largest
    k in the grid (greedy itself yields the prefixes k=1..k_max). Non item-based operators keep the grid."""
    from .operators import REGISTRY
    if not getattr(REGISTRY.get(op_code), "item_based", False):
        return plist
    groups = {}
    for p in plist:
        rest = tuple(sorted((k, str(v)) for k, v in p.items() if k != "k"))
        kv = p.get("k", 3)
        kn = int(kv) if str(kv).isdigit() else 3
        base = {k: v for k, v in p.items() if k != "k"}
        if rest not in groups or kn > groups[rest]["k"]:
            groups[rest] = {**base, "k": kn}
    return list(groups.values())


def _params_for(ops, cfg, section="ops"):
    from .common import expand_grid
    out = {}
    ops_cfg = cfg.get(section, {}) or {}
    for code in ops:
        spec = ops_cfg.get(code, None)
        if spec is None:
            from .operators import REGISTRY
            spec = getattr(REGISTRY.get(code), "default_grid", {})
        out[code] = expand_grid(spec) if spec else [{}]
    return out


def cmd_run(args):
    cfg = get_config(args.config)
    cfg["run"]["name"] = args.run
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    ops = args.ops.split(",") if args.ops else P0_OPS
    models = args.models.split(",")
    snapshot_config(cfg, args.run)
    r = Runner(cfg, args.run, models, device=args.device, strategy=args.strategy,
               search_n=args.n or cfg.get("search", {}).get("n", 5), no_llm=args.no_llm)
    doc_ids = _load_docs(args.docs)
    if args.limit:
        doc_ids = doc_ids[:args.limit]
    params = _params_for(ops, cfg, args.ops_section)
    log(f"[runner] run={args.run} | {len(doc_ids)} documents | ops={ops} | models={models} | "
        f"strategy={args.strategy} | keeps={cfg.get('keep')}")
    r.run(doc_ids, ops, params)
    print(f"-> {r.dir}/variants.jsonl.gz + scores.jsonl.gz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--docs", default="pilot", help="pilot | eligible | <file name in outputs/retrace>")
    ap.add_argument("--ops", default=None, help="comma-separated; default: P0")
    ap.add_argument("--models", default="main,no_cl,no_mamba")
    ap.add_argument("--strategy", default="random", choices=["random", "best_of_n", "greedy"])
    ap.add_argument("--n", type=int, default=None, help="number of candidates for best_of_n")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--ops-section", default="ops",
                    help="parameter grid section in configs/retrace.yaml (ops | ops_llm)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
