"""Rescore the STORED VARIANTS of the ReTrace runs with the case-study systems (transfer attacks).

    python -m cta_systems.evaluation.rescore                           # runs rand,worst,llm; every model
    python -m cta_systems.evaluation.rescore --runs rand --models ttp_match,ttp_vec_rf
    python -m cta_systems.evaluation.rescore --limit 200               # dry run: first 200 variants of each run

Reads : outputs/retrace/runs/<run>/variants.jsonl.gz (variant text + parameters + validity checks)
Writes: outputs/cta_systems/runs/<run>/scores.jsonl.gz   same format as the ReTrace scores
        outputs/cta_systems/runs/<run>/extract.jsonl.gz  the model's actual input at 100%
                                                         (technique set) -> stage decomposition
        outputs/cta_systems/runs/<run>/meta.json         number of variants, graph-match check, timing
Only level 0/1/2 variants have text (level 3 edits graphs and does not apply to text models).
Timeline cut-off = the sentence ThreatMAMBA uses as the cut-off (rebuilt graph, cache prepared.jsonl.gz).
Resumable: existing (vid, model, cut-off) entries are skipped.
"""
import argparse
import json
import time

from ..common import (ADV_RUNS, RUNS_DIR, JsonlAppender, groups, iter_jsonl_any, load_cfg, log, model_fingerprint,
                      purge_model_rows, save_json, sha1)
from ..prepare import Preparer
from .zoo import Item, load_scorers, summarize_probs


def default_tags(cfg: dict) -> list[str]:
    return [f"ttp_vec_{a}" for a in cfg["ttp_vec"]["algos"]] + ["ttp_match"]


def iter_text_variants(run: str, levels: set):
    for v in iter_jsonl_any(ADV_RUNS / run / "variants.jsonl.gz"):
        if v.get("text") is None:
            continue
        if v.get("op") == "ORIG" or (v.get("level") in levels):
            yield v


def rescore_run(run: str, scorers: dict, prep: Preparer, cfg: dict, batch: int = 64, limit: int | None = None):
    src = ADV_RUNS / run / "variants.jsonl.gz"
    if not src.exists():
        log(f"[rescore] skipping run '{run}': {src} not found")
        return None
    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    fpf = out / "fingerprints.json"
    fps = json.load(open(fpf, encoding="utf-8")) if fpf.exists() else {}
    for t in scorers:
        cur = model_fingerprint(t)
        if t in fps and fps[t] != cur:
            n1 = purge_model_rows(out / "scores.jsonl.gz", t)
            n2 = purge_model_rows(out / "extract.jsonl.gz", t)
            log(f"[rescore] {run}: model {t} was retrained -> dropping {n1} old score rows + {n2} old extract rows, rescoring")
        fps[t] = cur
    save_json(fpf, fps)
    done = {(s["vid"], s["model"], s["keep"]) for s in iter_jsonl_any(out / "scores.jsonl.gz")}
    xdone = {(x["vid"], x["model"]) for x in iter_jsonl_any(out / "extract.jsonl.gz")}
    s_out, x_out = JsonlAppender(out / "scores.jsonl.gz"), JsonlAppender(out / "extract.jsonl.gz")
    G = groups(); g2i = {g: i for i, g in enumerate(G)}
    keeps = [float(k) for k in cfg["keeps"]]
    levels = set(cfg["levels"])
    tags = list(scorers)
    t0 = time.time()
    stats = {"run": run, "n_variants": 0, "n_scored": 0, "n_prepared_new": 0, "state_check": 0, "state_match": 0}
    pending: list[dict] = []

    def check_states(v, rec):
        # the State count of the rebuilt graph must equal the count recorded by the ReTrace run (trace) -> same graph
        n_st = (v.get("trace") or {}).get("n_states")
        if n_st is not None and rec is not None:
            stats["state_check"] += 1
            stats["state_match"] += int(n_st == rec["n_states"])

    def flush_batch():
        if not pending:
            return
        items = []
        for v in pending:
            rec = prep.get(v["text"])
            check_states(v, rec)
            items.append(Item(v["text"], rec))
        for t in tags:
            todo = [i for i, v in enumerate(pending) if not all((v["vid"], t, k) in done for k in keeps)]
            if todo:
                P = scorers[t].score([items[i] for i in todo], keeps)
                for j, i in enumerate(todo):
                    v = pending[i]
                    tgt = (v.get("meta") or {}).get("target")
                    for k in keeps:
                        if (v["vid"], t, k) in done:
                            continue
                        row = {"vid": v["vid"], "model": t, "keep": k,
                               **summarize_probs(P[k][j], g2i[v["group"]], g2i.get(tgt) if tgt else None)}
                        s_out.add(row)
                        done.add((v["vid"], t, k))
            for i, v in enumerate(pending):
                if (v["vid"], t) not in xdone:
                    x_out.add({"vid": v["vid"], "model": t, "x": scorers[t].extract(items[i])})
                    xdone.add((v["vid"], t))
        stats["n_scored"] += len(pending)
        pending.clear()
        s_out.flush(); x_out.flush(); prep.flush()

    for v in iter_text_variants(run, levels):
        if limit is not None and stats["n_variants"] >= limit:
            break
        stats["n_variants"] += 1
        if all((v["vid"], t, k) in done for t in tags for k in keeps) and all((v["vid"], t) in xdone for t in tags):
            check_states(v, prep.mem.get(sha1(v["text"])))
            continue
        pending.append(v)
        if len(pending) >= batch:
            flush_batch()
            if stats["n_scored"] % (batch * 10) == 0:
                log(f"[rescore] {run}: {stats['n_scored']} new variants | {int(time.time() - t0)}s | "
                    f"graphs rebuilt: {prep.n_built}")
    flush_batch()
    prep.save()
    for sc in scorers.values():
        if hasattr(sc, "emb"):
            sc.emb.save()
    stats["n_prepared_new"] = prep.n_built
    stats["seconds"] = round(time.time() - t0, 1)
    stats["models"] = tags
    stats["state_match_rate"] = (round(stats["state_match"] / stats["state_check"], 4)
                                 if stats["state_check"] else None)
    save_json(out / "meta.json", stats)
    log(f"[rescore] done {run}: {stats['n_variants']} variants ({stats['n_scored']} newly scored) in "
        f"{stats['seconds']}s | State counts matching the original run: {stats['state_match']}/{stats['state_check']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=None, help="default from configs/cta_systems.yaml (rand,worst,llm)")
    ap.add_argument("--models", default=None, help="default: ttp_vec_<5 algorithms>,ttp_match")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="only score the first N variants of each run (dry run)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    runs = args.runs.split(",") if args.runs else list(cfg["runs"])
    tags = args.models.split(",") if args.models else default_tags(cfg)
    scorers = load_scorers(tags, device=args.device, gcfg=cfg["ttp_match"])
    prep = Preparer(device=args.device)
    log(f"[rescore] runs={runs} | models={tags} | cut-offs={cfg['keeps']} | levels={cfg['levels']}")
    for run in runs:
        rescore_run(run, scorers, prep, cfg, batch=args.batch, limit=args.limit)
    prep.save()


if __name__ == "__main__":
    main()
