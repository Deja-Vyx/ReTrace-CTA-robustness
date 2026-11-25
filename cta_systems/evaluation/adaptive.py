"""ADAPTIVE attacks: search the strongest variants with the scores of the case-study system ITSELF.

The ReTrace `worst` / `llm` runs select variants with the scores of ThreatMAMBA main, so for the other
systems they are only transfer attacks (which underestimate their vulnerability). This module repeats
EXACTLY the procedure of retrace/runner (same operators, same parameter grid in configs/retrace.yaml,
same per-document seed, same greedy-k / best-of-N / targeted, same validity checks) but uses the
case-study system as the black-box target; the E8 decoy B = the second-ranked group OF THAT SYSTEM.

    python -m cta_systems.evaluation.adaptive --model ttp_match
    python -m cta_systems.evaluation.adaptive --model ttp_vec_rf --limit 10     # dry run on 10 documents
    python -m cta_systems.evaluation.adaptive --model ttp_match --llm                # add T1,T2,E2,E6 (Ollama)

Writes outputs/cta_systems/runs/adapt_<tag>/{variants,scores,extract}.jsonl.gz (same format as the
ReTrace runs). Resumable (finished documents are skipped).
"""
import argparse
import json
import random
import time

import numpy as np

from retrace.common import param_key
from retrace.config import get_config
from retrace.operators import get_op
from retrace.operators.base import Ctx
from retrace.runner import _greedy_grid, _params_for
from retrace.search import greedy_k, pick_best_of_n, pick_targeted
from retrace.validate import check_variant

from ..common import (ADV_DIR, KEEPS, RUNS_DIR, JsonlAppender, groups, iter_jsonl_any, load_cfg, log,
                      model_fingerprint, save_json, sha1)
from ..prepare import Preparer, join_sents, techs_at
from .zoo import Item, jaccard, load_scorers, summarize_probs

LLM_OPS = ["T1", "T2", "E2", "E6"]


def load_doc_ids(which: str) -> list[str]:
    """Same as retrace/runner._load_docs: eligible | pilot | file name in outputs/retrace."""
    if which == "eligible":
        return [r["doc_id"] for r in iter_jsonl_any(ADV_DIR / "eligible_v0.jsonl")]
    fp = ADV_DIR / ("pilot.txt" if which == "pilot" else which)
    return [ln.strip() for ln in open(fp, encoding="utf-8") if ln.strip() and not ln.startswith("#")]


class AdaptiveRunner:
    def __init__(self, tag: str, scorer, prep: Preparer, adv_cfg: dict, ccfg: dict):
        self.tag, self.sc, self.prep = tag, scorer, prep
        self.cfg = adv_cfg
        self.keeps = [float(k) for k in ccfg["keeps"]]
        self.G = groups(); self.g2i = {g: i for i, g in enumerate(self.G)}
        self.dir = RUNS_DIR / f"adapt_{tag}"
        fpf = self.dir / "fingerprint.json"
        cur = model_fingerprint(tag)
        if fpf.exists() and json.load(open(fpf, encoding="utf-8")).get("fingerprint") != cur:
            stale = self.dir.with_name(f"{self.dir.name}.stale_{time.strftime('%Y%m%d_%H%M%S')}")
            self.dir.rename(stale)
            log(f"[adaptive] {tag} was retrained -> old results moved to {stale.name}, restarting from scratch")
        save_json(fpf, {"fingerprint": cur, "model": tag})
        self.v_out = JsonlAppender(self.dir / "variants.jsonl.gz")
        self.s_out = JsonlAppender(self.dir / "scores.jsonl.gz")
        self.x_out = JsonlAppender(self.dir / "extract.jsonl.gz")
        from ..common import load_docs
        cnt = np.bincount([self.g2i[d["group"]] for d in load_docs("train")], minlength=len(self.G))
        self.train_prior = cnt / max(1, cnt.sum())            # tie-breaking when choosing the decoy B
        self.done_docs = {x["doc_id"] for x in iter_jsonl_any(self.dir / "done.jsonl")}
        self.done_app = JsonlAppender(self.dir / "done.jsonl")
        from retrace.pools import Pools
        self.pools = Pools(prep.pipe)
        self.ctx = Ctx(pipeline=prep.pipe, cfg=adv_cfg, pools=self.pools, groups=self.G)
        self.n_calls = 0

    # ------------------------------------------------------------------ scoring
    def _item(self, text: str, need_rec: bool) -> Item:
        rec = self.prep.get(text) if need_rec else {"cut": [None] * len(KEEPS), "tech": [], "n_states": None}
        return Item(text, rec)

    def probs_full(self, text: str) -> np.ndarray:
        """100% cut-off only (used during search): TTP-Match needs no graph; TTP-Vec only needs the
        technique set (fast path, skips IOC extraction)."""
        self.n_calls += 1
        if self.sc.kind == "ttp_vec":
            rec = {"cut": [None] * len(KEEPS), "tech": [[t, 0] for t in sorted(self.prep.techs_fast(text))],
                   "n_states": None}
            it = Item(text, rec)
        else:
            it = self._item(text, need_rec=False)
        return self.sc.score([it], [1.0])[1.0][0]

    def _text(self, doc, v) -> str:
        return v.text if v.text is not None else join_sents(doc.sents)

    def p_true(self, doc, v) -> float:
        return float(self.probs_full(self._text(doc, v))[self.g2i[doc.group]])

    def write(self, doc, v, orig_rec: dict, val: dict):
        text = self._text(doc, v)
        it = self._item(text, need_rec=True)
        P = self.sc.score([it], self.keeps)
        tgt = (v.meta or {}).get("target")
        for k in self.keeps:
            self.s_out.add({"vid": v.meta["vid"], "model": self.tag, "keep": k,
                            **summarize_probs(P[k][0], self.g2i[doc.group], self.g2i.get(tgt) if tgt else None)})
        self.x_out.add({"vid": v.meta["vid"], "model": self.tag, "x": self.sc.extract(it)})
        trace = {"n_states": it.rec["n_states"], "d_states": it.rec["n_states"] - orig_rec["n_states"],
                 "tech_jacc": round(jaccard(techs_at(it.rec, 1.0), techs_at(orig_rec, 1.0)), 4)}
        rec = {"run": self.dir.name, "vid": v.meta["vid"], "doc_id": doc.doc_id, "group": doc.group,
               "op": v.op, "level": v.level, "expect": v.expect, "params": v.params, "strategy": v.strategy,
               "cand": v.cand, "aligned": v.aligned, "runner_up": doc.meta.get("runner_up"),
               "valid": val, "trace": trace, "edits": v.edits[:8],
               "meta": {k: vv for k, vv in (v.meta or {}).items() if k != "trace"}, "text": v.text,
               "search_model": self.tag}
        self.v_out.add(rec)

    # ------------------------------------------------------------------ one operator on one document
    def run_op(self, doc, op_code: str, params: dict, rng: random.Random, strategy: str, n: int, orig_rec):
        op = get_op(op_code, self.ctx)
        cands = op.candidates(doc, params, n if strategy != "random" else 1, rng)
        if not cands:
            return 0
        if op.expect == "TGT":
            def _pair(vv):
                ps = self.probs_full(self._text(doc, vv))
                return (float(ps[self.g2i[doc.group]]),
                        float(ps[self.g2i.get(vv.meta.get("target"), self.g2i[doc.group])]))
            variants = pick_targeted(cands, _pair)
        elif strategy == "greedy" and op.item_based:
            k_max = int(params.get("k", 3)) if str(params.get("k", "")).isdigit() else 3
            variants = greedy_k(op, doc, k_max, lambda vv: self.p_true(doc, vv), rng,
                                self.cfg.get("search", {}).get("greedy_max_items", 10), base_params=params)
        elif strategy in ("best_of_n", "greedy"):
            variants = pick_best_of_n(cands, lambda vv: self.p_true(doc, vv))
        else:
            variants = cands
        for v in variants:
            v.meta = dict(v.meta or {})
            v.meta["vid"] = f"{doc.doc_id}|{v.op}|{param_key(v.params)}|c{v.cand}"
            val = check_variant(join_sents(doc.sents), v, self.cfg)
            self.write(doc, v, orig_rec, val)
        return len(variants)

    # ------------------------------------------------------------------ one document, every pass
    def run_doc(self, raw: dict, passes: list[tuple[str, list[str], dict, int, str]]):
        doc = self.prep.pipe.make_doc(raw)
        text0 = join_sents(doc.sents)
        orig_rec = self.prep.get(text0)
        it0 = Item(text0, orig_rec)
        P0 = self.sc.score([it0], self.keeps)
        p0 = P0[1.0][0]
        # ties (KNN/DT 0-1, RF) -> prefer the group more frequent in train (more plausible decoy), then the index
        order = sorted(range(len(self.G)), key=lambda i: (-p0[i], -self.train_prior[i], i))
        # decoy B = the highest-probability group OF THIS MODEL other than the true group and the current
        # top-1 (correct model -> B = rank 2, as in eligible.py; wrong model -> do not use the wrongly
        # predicted group itself, avoiding a "free" ASR; the report only counts ASR on correctly attributed documents)
        truth = self.g2i[doc.group]
        doc.meta["runner_up"] = self.G[next(i for i in order if i != truth and i != order[0])]
        ovid = f"{doc.doc_id}|ORIG|-|c0"
        for k in self.keeps:
            self.s_out.add({"vid": ovid, "model": self.tag, "keep": k,
                            **summarize_probs(P0[k][0], self.g2i[doc.group])})
        self.x_out.add({"vid": ovid, "model": self.tag, "x": self.sc.extract(it0)})
        self.v_out.add({"run": self.dir.name, "vid": ovid, "doc_id": doc.doc_id, "group": doc.group, "op": "ORIG",
                        "text": text0, "runner_up": doc.meta["runner_up"], "search_model": self.tag})
        seed = int(self.cfg["run"]["seed"])
        for _name, ops, params_per_op, n, strategy in passes:
            # each pass uses its own per-document rng, LIKE the original runner (each runner command is one pass)
            rng = random.Random(int(sha1(f"{seed}|{doc.doc_id}")[:8], 16))
            for op_code in ops:
                plist = params_per_op.get(op_code, [{}])
                if strategy == "greedy":
                    plist = _greedy_grid(op_code, plist)
                for params in plist:
                    try:
                        self.run_op(doc, op_code, params, rng, strategy, n, orig_rec)
                    except SystemExit:
                        raise
                    except Exception as e:  # noqa: BLE001
                        log(f"[adaptive] ERROR {op_code} {params} on {doc.doc_id}: {type(e).__name__}: {e}")
        self.v_out.flush(); self.s_out.flush(); self.x_out.flush()
        self.done_app.add({"doc_id": doc.doc_id}); self.done_app.flush()
        self.prep.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="ttp_match | ttp_vec_<knn|svm|rf|dt|nb>")
    ap.add_argument("--docs", default="eligible", help="eligible | pilot (as in the ReTrace runner)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--llm", action="store_true", help="add T1,T2,E2 neutral,E6 (best-of-3, needs Ollama)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    ccfg = load_cfg(args.config)
    acfg = ccfg["adaptive"]
    adv_cfg = get_config()                              # configs/retrace.yaml (read-only)
    scorer = load_scorers([args.model], device=args.device, gcfg=ccfg["ttp_match"])[args.model]
    prep = Preparer(device=args.device)
    ar = AdaptiveRunner(args.model, scorer, prep, adv_cfg, ccfg)
    passes = [("greedy", list(acfg["ops_greedy"]), _params_for(acfg["ops_greedy"], adv_cfg), int(acfg["n"]), "greedy"),
              ("bon", list(acfg["ops_bon"]), _params_for(acfg["ops_bon"], adv_cfg), int(acfg["n"]), "best_of_n")]
    llm_ops = list(acfg.get("ops_llm") or []) or (LLM_OPS if args.llm else [])
    if llm_ops:
        passes.append(("llm", llm_ops, _params_for(llm_ops, adv_cfg, "ops_llm"), int(acfg["n_llm"]), "best_of_n"))
    doc_ids = load_doc_ids(args.docs)
    if args.limit:
        doc_ids = doc_ids[:args.limit]
    from ..common import load_docs
    raw = {d["doc_id"]: d for d in load_docs()}
    t0 = time.time()
    log(f"[adaptive] model={args.model} | {len(doc_ids)} documents | passes: " +
        "; ".join(f"{p[0]}={','.join(p[1])}" for p in passes))
    for i, did in enumerate(doc_ids):
        if did in ar.done_docs or did not in raw:
            continue
        ar.run_doc(raw[did], passes)
        if (i + 1) % 20 == 0:
            prep.save()
            if hasattr(scorer, "emb"):
                scorer.emb.save()
        if (i + 1) % 5 == 0:
            log(f"[adaptive] {i + 1}/{len(doc_ids)} documents | {int(time.time() - t0)}s | {ar.n_calls} model calls")
    prep.save()
    if hasattr(scorer, "emb"):
        scorer.emb.save()
    save_json(ar.dir / "meta.json", {"model": args.model, "n_docs": len(doc_ids), "passes": [p[:2] for p in passes],
                                     "seconds": round(time.time() - t0, 1), "n_model_calls": ar.n_calls})
    log(f"[adaptive] done -> {ar.dir}")


if __name__ == "__main__":
    main()
