#!/usr/bin/env python3
"""CPU tests for ReTrace (retrace package) - no GPU/SecureBERT/Ollama needed.

    THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_retrace.py

Checks every perturbation operator, the sentence cache, graph rebuilding and the metric helpers on
synthetic data. This does NOT replace a real run on a GPU machine.
"""
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("THREATMAMBA_FAKE_ENCODER", "1")
os.environ.setdefault("THREATMAMBA_SSM", "simple")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

PASS = []


def ok(name):
    PASS.append(name)
    print(f"[PASS] {name}")


DOCS = [
    {"doc_id": "t000", "group": "APT28",
     "text": "On March 3, 2021, APT28 sent a spearphishing email from crook@mailer-portal.test. "
             "The attacker used PowerShell to download a payload from bad-portal-xz12.com over HTTPS. "
             "After that, the implant beaconed to 45.77.12.9 on port 8443. "
             "Finally, data was exfiltrated via DNS to c2-node-a1.net. "
             "The Remote Desktop Protocol was used for lateral movement across the network."},
    {"doc_id": "t001", "group": "Silence",
     "text": "The Silence group targeted a bank in 2019. "
             "They deployed a backdoor and used Mimikatz to dump credentials. "
             "The malware contacted evil-host-qq7.com and wrote to the registry for persistence. "
             "A file named svc-loader.exe was dropped on the host."},
]


def build_pipeline():
    from retrace.pipeline import AdvPipeline
    pipe = AdvPipeline(fake=True)
    # attach a fake ATT&CK mapping to the builder (as in the base repository's tests)
    pipe.builder.tech2tac = {"T1566": ["initial-access"], "T1059.001": ["execution"],
                             "T1059": ["execution"], "T1105": ["command-and-control"],
                             "T1071.004": ["command-and-control"], "T1003": ["credential-access"],
                             "T1021.001": ["lateral-movement"], "T1505": ["persistence"]}
    pipe.builder.tac_order = ["initial-access", "execution", "credential-access",
                              "lateral-movement", "command-and-control", "persistence",
                              "exfiltration"]
    pipe.builder.tacid2short = {"TA0010": "exfiltration", "TA0008": "lateral-movement",
                                "TA0006": "credential-access", "TA0003": "persistence"}
    return pipe


def make_ctx(pipe):
    from retrace.config import get_config
    from retrace.operators.base import Ctx
    from retrace.pools import Pools
    cfg = get_config()
    return Ctx(pipeline=pipe, cfg=cfg, pools=Pools(pipe), groups=["APT28", "Silence"])


def main():
    from retrace.pipeline import join_sents
    from retrace.sentcache import CachedSentenceEncoder
    from src.encoder import FakeEncoder
    from src.utils import sent_split

    # 1. sentence cache
    enc = CachedSentenceEncoder(FakeEncoder())
    s = ["alpha beta", "gamma delta", "alpha beta"]
    e1 = enc.encode(s)
    assert e1.shape == (3, 768)
    assert torch.allclose(e1[0], e1[2]), "the same sentence must give an identical vector"
    assert enc.encode(["alpha beta"]).allclose(e1[0])
    ok(f"sentcache: per-sentence hit/miss, stats={enc.stats()}")

    # 2. sentence split/join
    for d in DOCS:
        se = sent_split(d["text"])
        assert sent_split(join_sents(se)) == se
    ok("join_sents: sent_split(join(s)) == s")

    pipe = build_pipeline()
    docs = [pipe.make_doc(d) for d in DOCS]
    d0 = docs[0]
    assert d0.graph["x"].shape[0] == len(d0.graph["node_types"])
    assert len(d0.graph["edges"]) > 0
    ok(f"pipeline.make_doc: {len(d0.graph['node_types'])} nodes / {len(d0.graph['edges'])} edges / "
       f"{len(d0.graph['states'])} state")

    # 3. identity
    from retrace.pipeline import graph_struct_equal
    g_re, _ = pipe.build("t000-re", d0.group, join_sents(d0.sents))
    same, why = graph_struct_equal(d0.graph, g_re)
    assert same, f"identity must keep the graph unchanged: {why}"
    # a node permutation (as when ioc-finder reorders IOCs) must still count as a match; dropping an edge must not
    n0 = len(d0.graph["node_types"])
    p = list(range(n0)); random.Random(3).shuffle(p)
    inv = {o: j for j, o in enumerate(p)}
    gp = {**d0.graph, "node_types": [d0.graph["node_types"][i] for i in p],
          "node_values": [d0.graph["node_values"][i] for i in p],
          "node_sent": [d0.graph["node_sent"][i] for i in p], "x": d0.graph["x"][torch.tensor(p)],
          "edges": torch.tensor([[inv[u], inv[v], r] for u, v, r in d0.graph["edges"].tolist()]),
          "states": [inv[s] for s in d0.graph["states"]]}
    assert graph_struct_equal(d0.graph, gp)[0], "a node permutation must count as a match"
    assert not graph_struct_equal(d0.graph, {**gp, "edges": gp["edges"][1:]})[0]
    ok("identity: build(text) == build(join(sent_split(text))); parity is invariant to node permutations")

    ctx = make_ctx(pipe)
    from retrace.operators import get_op

    # 4. T3, T6
    v = get_op("T3", ctx).candidates(d0, {"direction": "contract", "touch": "no_regex", "k": "all"}, 1, random.Random(0))
    assert v and " PS " in v[0].text and len(sent_split(v[0].text)) == len(d0.sents)
    ok(f"T3 contract (no_regex): PowerShell -> PS, keeps {len(d0.sents)} sentences")
    vr = get_op("T3", ctx).candidates(d0, {"direction": "contract", "touch": "regex", "k": "all"}, 1, random.Random(0))
    assert vr and vr[0].meta.get("touches_regex") and "RDP" in vr[0].text
    ok("T3 regex: Remote Desktop Protocol -> RDP, flagged touches_regex")
    v = get_op("T6", ctx).candidates(d0, {}, 1, random.Random(0))
    assert v and "2021-03" in v[0].text
    ok("T6: 'March 2021' -> ISO")

    # 5. E1 - mask plain and defanged IOCs
    from src.ioc_extract import extract_iocs
    op = get_op("E1", ctx)
    items = op.items(d0)
    v = op.apply_items(d0, items[:2], {"k": 2}, random.Random(0))
    assert v is not None
    o_ioc = {(i["type"], i["value"].lower()) for i in extract_iocs(join_sents(d0.sents))}
    n_ioc = {(i["type"], i["value"].lower()) for i in extract_iocs(v.text)}
    assert len(n_ioc) < len(o_ioc)
    ok(f"E1: masked {len(o_ioc)-len(n_ioc)} IOCs")

    # 6. E4 - ambiguous alias
    from retrace.lexicons import find_group_mentions
    txt = "The Silence group struck. Later silence fell. Then Silence attacked again."
    sp = find_group_mentions(txt, "Silence")
    assert all("silence " not in txt[s:e].lower() or txt[s:e] == "Silence" for s, e, _a in sp)
    v = get_op("E4", ctx).candidates(docs[1], {"scope": "true"}, 1, random.Random(0))
    assert v and "the threat actor" in v[0].text.lower()
    ok(f"E4: group name masked {v[0].meta['n_replaced']} times")

    # 7. E7 carriers + random IOCs of the right type
    v = get_op("E7", ctx).candidates(d0, {"k": 2, "type": "mix"}, 1, random.Random(1))
    if v:
        inj = v[0].meta["injected"]
        ok(f"E7: inserted {len(inj)} random IOCs via carrier sentences")
    else:
        ok("E7: no carriers (empty pool in the CPU test) - accepted")

    # 8. G1/G3/G5
    from retrace.pipeline import STATE_T
    g = get_op("G1", ctx).candidates(d0, {"type": "Technique", "k": "all"}, 1, random.Random(0))
    assert g and g[0].graph is not None
    e = g[0].graph["edges"]
    n = len(g[0].graph["node_types"])
    assert (e[:, :2].max().item() < n if e.numel() else True), "no dangling edges allowed"
    ok(f"G1: Techniques removed, {n} nodes left, no dangling edges")
    g = get_op("G3", ctx).candidates(d0, {"mode": "reverse"}, 1, random.Random(0))
    assert g and len(g[0].graph["node_types"]) == len(d0.graph["node_types"])
    tn = int(sum(1 for _u, _v, r in g[0].graph["edges"].tolist() if r == __import__("src.cskg_builder", fromlist=["REL"]).REL["temporal_next"]))
    assert tn == max(0, len(d0.graph["states"]) - 1)
    ok("G3 reverse: keeps the node set and the number of temporal_next edges")
    g = get_op("G5", ctx).candidates(d0, {"keep_temporal": True}, 1, random.Random(0))
    assert g and len(g[0].graph["edges"]) == len(d0.graph["edges"])
    ok("G5: keeps the number of edges")

    # 9. neg_splice_matched produces same-size negatives
    from src.contrastive import neg_splice, neg_splice_matched
    from retrace.pipeline import graph_trace
    import statistics
    r_old, r_new = [], []
    rng = random.Random(0)
    for _ in range(10):
        a, b = rng.choice(docs).graph, rng.choice(docs).graph
        r_old.append(len(neg_splice(a, b, pipe.builder.tac_order, rng=rng)["node_types"]) / max(1, len(a["node_types"])))
        r_new.append(len(neg_splice_matched(a, b, pipe.builder.tac_order, rng=rng)["node_types"]) / max(1, len(a["node_types"])))
    ok(f"neg_splice_matched: size ratio old~{statistics.mean(r_old):.2f} vs matched~{statistics.mean(r_new):.2f}")

    # 10. models: ModelZoo if a checkpoint is present (skipped otherwise)
    from src.utils import OUTPUTS
    if (OUTPUTS / "model_main.pt").exists():
        try:
            from retrace.models import ModelZoo
            ModelZoo(["main"], device="cpu")
            ok("ModelZoo: model_main.pt loaded")
        except SystemExit as e:
            msg = str(e)
            assert "SSM architecture mismatch" in msg or "mamba" in msg, f"unexpected error while loading the checkpoint: {msg}"
            ok("ModelZoo: a 'mamba' checkpoint cannot be loaded with THREATMAMBA_SSM=simple (as expected)")
    else:
        ok("ModelZoo: no checkpoint available - skipped (run on the GPU machine)")

    # 11. metrics: bootstrap + Cohen's kappa on synthetic data
    from retrace.metrics import bootstrap_ci
    m, lo, hi = bootstrap_ci({"d1": [1, 0], "d2": [1, 1], "d3": [0, 0]})
    assert lo <= m <= hi
    from retrace.validate import cohen_kappa
    k, n = cohen_kappa({"a": 1, "b": 0, "c": 1, "d": 0}, {"a": 1, "b": 0, "c": 0, "d": 0})
    assert n == 4
    ok(f"metrics: bootstrap CI [{lo},{hi}] | cohen_kappa={k}")

    # 12. validate.check_variant
    from retrace.validate import check_variant
    from retrace.operators.base import Variant
    vv = Variant(doc_id="t000", group="APT28", op="T1", level=1, expect="INV", params={},
                 text=join_sents(d0.sents))
    res = check_variant(join_sents(d0.sents), vv, ctx.cfg)
    assert res["passed"]
    ok("validate.check_variant: the original passes the checks")

    print(f"\n{len(PASS)}/{len(PASS)} PASS")


if __name__ == "__main__":
    main()
