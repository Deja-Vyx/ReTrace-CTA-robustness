"""Build the eligible document set: keep only documents that (a) are real CTI and (b) every
model under test attributes CORRECTLY at the 100% cut-off, so that every perturbation is
measured against a model that was right to begin with.

    python -m retrace.eligible build --models main,no_cl,no_mamba --tier lenient --cap 25 --pilot 50

Writes outputs/retrace/eligible_v0.jsonl (one document per line) + eligible_summary.csv + pilot.txt.
Key columns:
  - is_cti       : passes the CTI filter (term density, optionally an LLM check)
  - in_authors_clean : present in the dataset authors' cleaned_CTI2Attacker.csv (content match)
  - correct_100 / correct_20 : the models (intersection of all) are correct at 100% / 20%
  - name_in_text / name_in_graph : the group name appears in the text / in an evidence sentence
  - runner_up    : second-ranked group (decoy for targeted E8)
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict

from src.utils import PROCESSED, load_config, load_json, read_jsonl, sent_split, clean_text
from .common import ADV_DIR, log
from .lexicons import dataset_groups, find_group_mentions

SECURITY_RE = re.compile(
    r"\b(malware|attacker|threat actor|phishing|spearphish|c2|command[ -]?and[ -]?control|exploit|"
    r"backdoor|trojan|payload|apt\d*|campaign|compromise|intrusion|ransomware|botnet|dropper|"
    r"implant|lateral movement|persistence|exfiltrat|beacon|obfuscat|reconnaissance|privilege "
    r"escalation|credential|c2 server|malicious)\w*", re.I)


def security_density(text: str) -> float:
    words = max(1, len(text.split()))
    return len(SECURITY_RE.findall(text)) / words * 1000


def _authors_clean_keys():
    fp = PROCESSED.parent / "raw" / "MuscleFish" / "cleaned_CTI2Attacker.csv"
    keys = set()
    if not fp.exists():
        return keys
    import pandas as pd
    df = pd.read_csv(fp, encoding="utf-8", encoding_errors="replace")
    col = "cti_content" if "cti_content" in df.columns else df.columns[-3]
    for v in df[col].dropna():
        keys.add(re.sub(r"[^a-z0-9]", "", clean_text(str(v)).lower())[:400])
    return keys


def build(models, tier="lenient", cap=25, pilot_n=50, use_llm_cti=False, device=None, seed=17):
    import numpy as np
    from .models import ModelZoo
    from .pipeline import AdvPipeline, join_sents

    cfg = load_config()
    density_min = {"strict": 3.0, "lenient": 2.0, "off": -1.0}[tier]
    docs = [d for d in read_jsonl(PROCESSED / "docs.jsonl") if d["split"] == "test"]
    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}

    clean_keys = _authors_clean_keys()
    llm = None
    if use_llm_cti:
        from .llm import P_IS_CTI, check_is_cti, get_llm
        llm = get_llm(cfg)

    # 1) CTI filter
    rows = []
    for d in docs:
        dens = security_density(d["text"])
        ck = re.sub(r"[^a-z0-9]", "", d["text"].lower())[:400]
        in_clean = ck in clean_keys
        is_cti = in_clean or dens >= density_min
        if use_llm_cti and not in_clean and llm is not None:
            r = llm.json(P_IS_CTI.format(text=d["text"][:2500]), check=check_is_cti, temperature=0.0,
                         tag="cti")
            is_cti = bool(r and r.get("is_cti"))
        rows.append({"doc": d, "density": round(dens, 2), "in_authors_clean": int(in_clean),
                     "is_cti": int(bool(is_cti))})

    # 2) predictions of each model at 100% and 20% (intersection of all -> correct)
    from src.cskg_builder import truncate_graph
    from src.utils import CSKG_DIR, torch_load
    zoo = ModelZoo(models, device=device, groups=groups)
    graphs, gi = [], []
    for i, r in enumerate(rows):
        fp = CSKG_DIR / f"{r['doc']['doc_id']}.pt"
        if fp.exists():
            graphs.append(torch_load(fp)); gi.append(i)
    sc = zoo.score_graphs(graphs, [0.2, 1.0])
    for k, i in enumerate(gi):
        y = g2i[rows[i]["doc"]["group"]]
        c100 = all(sc[(t, 1.0)][k].argmax() == y for t in models)
        c20 = all(sc[(t, 0.2)][k].argmax() == y for t in models)
        # runner_up according to the first model (main)
        p = sc[(models[0], 1.0)][k]
        order = np.argsort(-p)
        rows[i]["correct_100"] = int(c100)
        rows[i]["correct_20"] = int(c20)
        rows[i]["runner_up"] = groups[int(order[1])]
        rows[i]["margin"] = round(float(p[order[0]] - p[order[1]]), 4)
        rows[i]["has_graph"] = 1

    # 3) group-name leakage
    for r in rows:
        d = r["doc"]
        sents = sent_split(d["text"])
        active = set()
        # evidence sentences = sentences that appear in the graph states -> read from the graph
        fp = CSKG_DIR / f"{d['doc_id']}.pt"
        if fp.exists():
            g = torch_load(fp)
            active = {int(g["node_sent"][s]) for s in g["states"]}
        spans = find_group_mentions(d["text"], d["group"])
        r["name_in_text"] = int(bool(spans))
        # does the name occur in an evidence sentence?
        in_graph = False
        for s, e, _a in spans:
            # find the sentence that contains position s
            pos = 0
            for si, sent in enumerate(sents):
                nxt = pos + len(sent) + 1
                if pos <= s < nxt:
                    in_graph = in_graph or (si in active)
                    break
                pos = nxt
        r["name_in_graph"] = int(in_graph)

    # 4) select eligible documents: is_cti & correct_100 & has a graph
    elig = [r for r in rows if r.get("is_cti") and r.get("correct_100") and r.get("has_graph")]
    # cap per group
    # Draw a RANDOM (seeded) sample of at most `cap` documents per group. Do not take the
    # highest-margin documents: that keeps only the "easiest" ones and underestimates flip rates.
    import random as _random
    _rng = _random.Random(seed)
    by_g = defaultdict(list)
    for r in sorted(elig, key=lambda r: r["doc"]["doc_id"]):
        by_g[r["doc"]["group"]].append(r)
    capped = []
    for g in sorted(by_g):
        lst = by_g[g]
        capped += lst if len(lst) <= cap else _rng.sample(lst, cap)

    ADV_DIR.mkdir(parents=True, exist_ok=True)
    with open(ADV_DIR / "eligible_v0.jsonl", "w", encoding="utf-8") as f:
        for r in sorted(capped, key=lambda r: r["doc"]["doc_id"]):
            f.write(json.dumps({
                "doc_id": r["doc"]["doc_id"], "group": r["doc"]["group"],
                "runner_up": r.get("runner_up"), "margin": r.get("margin"),
                "density": r["density"], "in_authors_clean": r["in_authors_clean"],
                "correct_20": r.get("correct_20", 0), "name_in_text": r.get("name_in_text", 0),
                "name_in_graph": r.get("name_in_graph", 0),
            }, ensure_ascii=False) + "\n")
    with open(ADV_DIR / "eligible_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "n_test", "n_cti", "n_correct100", "n_eligible", "n_name_in_graph"])
        tot = Counter(); cti = Counter(); cor = Counter(); nig = Counter()
        for r in rows:
            g = r["doc"]["group"]; tot[g] += 1
            cti[g] += r.get("is_cti", 0); cor[g] += r.get("correct_100", 0)
        eset = {r["doc"]["doc_id"] for r in capped}
        eg = Counter(r["doc"]["group"] for r in capped)
        for r in capped:
            nig[r["doc"]["group"]] += r.get("name_in_graph", 0)
        for g in groups:
            w.writerow([g, tot[g], cti[g], cor[g], eg[g], nig[g]])

    # pilot: stratified by group
    import random
    rng = random.Random(seed)
    by_g2 = defaultdict(list)
    for r in capped:
        by_g2[r["doc"]["group"]].append(r["doc"]["doc_id"])
    pilot = []
    pool = {g: rng.sample(v, len(v)) for g, v in by_g2.items()}
    while len(pilot) < pilot_n and any(pool.values()):
        for g in sorted(pool):
            if pool[g]:
                pilot.append(pool[g].pop())
                if len(pilot) >= pilot_n:
                    break
    (ADV_DIR / "pilot.txt").write_text("\n".join(pilot) + "\n", encoding="utf-8")

    log(f"[eligible] test={len(rows)} | is_cti={sum(r.get('is_cti',0) for r in rows)} | "
        f"correct100={sum(r.get('correct_100',0) for r in rows)} | eligible={len(capped)} "
        f"over {len(eg)} groups | pilot={len(pilot)}")
    log(f"[eligible] name_in_graph among eligible: {sum(r.get('name_in_graph',0) for r in capped)}/{len(capped)}")
    log(f"-> {ADV_DIR/'eligible_v0.jsonl'}, eligible_summary.csv, pilot.txt")
    if len(capped) < 100:
        log("[eligible] WARNING: fewer than 100 eligible documents. Lower --tier to lenient/off or raise --cap.")
    return capped


def cmd_build(args):
    build(models=args.models.split(","), tier=args.tier, cap=args.cap, pilot_n=args.pilot,
          use_llm_cti=args.use_llm_cti, device=args.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--models", default="main,no_cl,no_mamba")
    ap.add_argument("--tier", default="lenient", choices=["strict", "lenient", "off"])
    ap.add_argument("--cap", type=int, default=25)
    ap.add_argument("--pilot", type=int, default=50)
    ap.add_argument("--use-llm-cti", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    {"build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
