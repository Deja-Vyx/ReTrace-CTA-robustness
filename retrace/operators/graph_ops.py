"""Level 3 - direct perturbations of the CSKG (G1-G5), used to DIAGNOSE the classifier (bypasses the extractor).

G1 drop nodes      : remove k (or all) nodes of one type: IOC | Technique | Tactic | <IOC type name>
G2 ioc_ttp         : mode=drop removes ioc_ttp edges; mode=rewire links IOCs to random techniques
G3 order           : permute the sentence order of the States (reverse | shuffle | swap1); rebuilds
                     temporal_next, tech_temporal, node_sent; keeps x and all other edges
G4 noise nodes     : add k noise IOC nodes with features taken from other documents (source=random)
                     or from group B (source=runner_up); graph-level twin of E7/E8
G5 relation shuffle: shuffle relation labels across edges, keeping endpoints, node count, edge count and degrees
"""
import random

import torch

from src.cskg_builder import NODE_TYPES, NT, REL
from ..pipeline import IOC_T_IDS, STATE_T, TAC_T, TECH_T, Doc
from .base import Op, register


def _clone(g: dict) -> dict:
    return {**g, "node_types": list(g["node_types"]), "node_values": list(g["node_values"]),
            "node_sent": list(g["node_sent"]), "x": g["x"].clone(), "edges": g["edges"].clone(),
            "states": list(g["states"])}


def remove_nodes(g: dict, drop: set) -> dict:
    keep = [i for i in range(len(g["node_types"])) if i not in drop]
    remap = {o: n for n, o in enumerate(keep)}
    edges = [(remap[u], remap[v], r) for u, v, r in g["edges"].tolist() if u in remap and v in remap]
    return {**g, "node_types": [g["node_types"][i] for i in keep],
            "node_values": [g["node_values"][i] for i in keep],
            "node_sent": [g["node_sent"][i] for i in keep], "x": g["x"][keep],
            "edges": torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 3, dtype=torch.long),
            "states": [remap[s] for s in g["states"] if s in remap]}


def _type_ids(name: str) -> set:
    if name == "IOC":
        return set(IOC_T_IDS)
    if name == "Technique":
        return {TECH_T}
    if name == "Tactic":
        return {TAC_T}
    if name in NT:
        return {NT[name]}
    raise ValueError(f"invalid node type: {name}")


@register
class G1DropNodes(Op):
    code, level, expect = "G1", 3, "DIR"
    item_based = True
    default_grid = {"type": ["IOC", "Technique", "Tactic"], "k": [1, "all"]}

    def items(self, doc: Doc) -> list:
        # the type parameter is passed via self._type (set by candidates before the call)
        tids = _type_ids(getattr(self, "_type", "IOC"))
        return [i for i, t in enumerate(doc.graph["node_types"]) if t in tids]

    def candidates(self, doc, params, n, rng):
        self._type = params.get("type", "IOC")
        return super().candidates(doc, params, n, rng)

    def apply_items(self, doc, items, params, rng):
        if not items:
            return None
        g = remove_nodes(doc.graph, set(items))
        vals = [f"{NODE_TYPES[doc.graph['node_types'][i]]}:{doc.graph['node_values'][i]}" for i in items]
        return self.make_graph_variant(doc, g, params, meta={"items": vals[:50], "n_removed": len(items)})


@register
class G2IocTtp(Op):
    code, level, expect = "G2", 3, "DIR"
    default_grid = {"mode": ["drop", "rewire"]}

    def candidates(self, doc, params, n, rng):
        g = _clone(doc.graph)
        e = g["edges"].tolist()
        it = REL["ioc_ttp"]
        n_it = sum(1 for _u, _v, r in e if r == it)
        if n_it == 0:
            return []
        mode = params.get("mode", "drop")
        out = []
        techs = [i for i, t in enumerate(g["node_types"]) if t == TECH_T]
        for c in range(max(1, n) if mode == "rewire" else 1):
            if mode == "drop":
                new = [x for x in e if x[2] != it]
            else:
                if len(techs) < 2:
                    return []
                new = []
                for u, v, r in e:
                    if r == it:
                        choices = [t for t in techs if t != v]
                        v = rng.choice(choices)
                    new.append((u, v, r))
            g2 = {**g, "edges": torch.tensor(new, dtype=torch.long) if new else torch.zeros(0, 3, dtype=torch.long)}
            v = self.make_graph_variant(doc, g2, params, meta={"n_ioc_ttp": n_it})
            v.cand = c
            out.append(v)
        return out


def reorder_states(g: dict, perm: list[int]) -> dict:
    """perm[i] = new position of the i-th State (in the old order). Rebuilds the order-dependent edges."""
    g = _clone(g)
    old_sents = [g["node_sent"][s] for s in g["states"]]
    mapping = {old_sents[i]: old_sents[perm[i]] for i in range(len(old_sents))}
    g["node_sent"] = [mapping.get(s, s) for s in g["node_sent"]]
    states = sorted(g["states"], key=lambda n: g["node_sent"][n])
    tn, tt = REL["temporal_next"], REL["tech_temporal"]
    edges = [(u, v, r) for u, v, r in g["edges"].tolist() if r not in (tn, tt)]
    edges += [(a, b, tn) for a, b in zip(states, states[1:])]
    techs = sorted([i for i, t in enumerate(g["node_types"]) if t == TECH_T],
                   key=lambda i: (g["node_sent"][i], str(g["node_values"][i])))
    edges += [(a, b, tt) for a, b in zip(techs, techs[1:])]
    g["states"] = states
    g["edges"] = torch.tensor(edges, dtype=torch.long) if edges else torch.zeros(0, 3, dtype=torch.long)
    return g


@register
class G3Order(Op):
    code, level, expect = "G3", 3, "DIR"
    default_grid = {"mode": ["reverse", "shuffle", "swap1"]}

    def candidates(self, doc, params, n, rng):
        S = len(doc.graph["states"])
        if S < 2:
            return []
        mode = params.get("mode", "reverse")
        out = []
        reps = 1 if mode == "reverse" else max(1, n)
        for c in range(reps):
            if mode == "reverse":
                perm = list(range(S))[::-1]
            elif mode == "shuffle":
                perm = list(range(S)); rng.shuffle(perm)
                if perm == list(range(S)):
                    perm = perm[1:] + perm[:1]
            else:
                perm = list(range(S)); i = rng.randrange(S - 1)
                perm[i], perm[i + 1] = perm[i + 1], perm[i]
            g = reorder_states(doc.graph, perm)
            v = self.make_graph_variant(doc, g, params, meta={"perm_head": perm[:20]})
            v.cand = c
            out.append(v)
        return out


@register
class G4NoiseNodes(Op):
    code, level, expect = "G4", 3, "INV"
    default_grid = {"k": [1, 3, 5], "source": ["random", "runner_up"]}

    def candidates(self, doc, params, n, rng):
        src = params.get("source", "random")
        donors = self.ctx.donors or {}
        if src == "runner_up":
            b = doc.meta.get("runner_up")
            pool = donors.get(b, [])
        else:
            pool = [d for g, ds in donors.items() if g != doc.group for d in ds]
        if not pool or not doc.graph["states"]:
            return []
        k = int(params.get("k", 1))
        out = []
        ioc_types = [NT[t] for t in ("Domain", "IP", "Hash", "URL")]
        for c in range(max(1, n)):
            g = _clone(doc.graph)
            added = []
            for j in range(k):
                d = rng.choice(pool)
                cand = [i for i, t in enumerate(d.graph["node_types"]) if t in IOC_T_IDS] or \
                       list(range(len(d.graph["node_types"])))
                src_i = rng.choice(cand)
                st = rng.choice(g["states"])
                nid = len(g["node_types"])
                t = d.graph["node_types"][src_i] if d.graph["node_types"][src_i] in IOC_T_IDS else rng.choice(ioc_types)
                g["node_types"].append(t)
                g["node_values"].append(f"noise{j}:{d.graph['node_values'][src_i]}")
                g["node_sent"].append(g["node_sent"][st])
                g["x"] = torch.cat([g["x"], d.graph["x"][src_i:src_i + 1].to(g["x"].dtype)])
                e = g["edges"].tolist() + [(st, nid, REL["dependency"])]
                g["edges"] = torch.tensor(e, dtype=torch.long)
                added.append(f"{d.doc_id}:{d.group}")
            v = self.make_graph_variant(doc, g, params, meta={"donors": added,
                                                              "target": doc.meta.get("runner_up") if src == "runner_up" else None})
            v.cand = c
            out.append(v)
        return out


@register
class G5RelShuffle(Op):
    code, level, expect = "G5", 3, "DIR"
    default_grid = {"keep_temporal": [True, False]}

    def candidates(self, doc, params, n, rng):
        e = doc.graph["edges"].tolist()
        if len(e) < 2:
            return []
        keep_t = bool(params.get("keep_temporal", True))
        tn = REL["temporal_next"]
        idx = [i for i, (_u, _v, r) in enumerate(e) if not (keep_t and r == tn)]
        out = []
        for c in range(max(1, n)):
            rels = [e[i][2] for i in idx]
            rng.shuffle(rels)
            new = [list(x) for x in e]
            for i, r in zip(idx, rels):
                new[i][2] = r
            g = {**doc.graph, "edges": torch.tensor(new, dtype=torch.long)}
            v = self.make_graph_variant(doc, g, params, meta={"n_shuffled": len(idx)})
            v.cand = c
            out.append(v)
        return out
