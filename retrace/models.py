"""Load and score models for ReTrace.

ModelZoo loads the checkpoints outputs/model_<tag>.pt (like evaluate.load_model, but it accepts
another folder so that tests can run on CPU). A graph is built ONCE and then scored by every
model at every timeline cut-off (truncate_graph).

Tag conventions:
  - main, no_cl, no_mamba, no_iochunter, main_s7, main_negm, ...  -> graph models
  - tags ending in "_masked" -> models trained on name-masked text; at scoring time the variant
    text must also be name-masked before the graph is built (preprocess = "mask_names")
  - b1_tfidf, b2_sbert -> text baselines (baselines.py)
"""
from pathlib import Path

import numpy as np
import torch

from src.cskg_builder import truncate_graph
from src.model import ThreatMambaModel, collate
from src.train import HUNT_RELS
from src.utils import OUTPUTS, PROCESSED, load_json, torch_load
from .pipeline import join_sents

TEXT_TAGS = ("b1_tfidf", "b2_sbert")


def preprocess_of(tag: str) -> str:
    return "mask_names" if tag.endswith("_masked") else "none"


def load_classifier(path: Path, device):
    ck = torch_load(path, map_location=device)
    m = ck["model_cfg"]; fl = ck["flags"]
    impl = ck.get("ssm_impl") or m.get("ssm_fallback", "auto")
    model = ThreatMambaModel(len(ck["groups"]), d=int(m["gat_dim"]), heads=int(m["gat_heads"]),
                             gat_layers=int(m["gat_layers"]), ssm_layers=int(m["ssm_layers"]),
                             dropout=float(m["dropout"]), ssm_mode=impl,
                             no_mamba=fl["no_mamba"]).to(device)
    try:
        model.load_state_dict(ck["state_dict"])
    except RuntimeError as e:
        raise SystemExit(
            f"Cannot load {path}: SSM architecture mismatch (checkpoint: {impl}). "
            f"A 'mamba' checkpoint needs a GPU + mamba-ssm; check `echo $THREATMAMBA_SSM` (must be empty). "
            f"Original error: {str(e)[:200]}") from None
    model.eval()
    return model, ck["groups"], fl, ck


def text_cut(sents: list[str], g: dict, keep: float) -> str:
    """Cut the text EXACTLY at the sentence that truncate_graph uses as the cut-off, so that text
    baselines are compared fairly with graph models at the same timeline cut-off."""
    states = g["states"]
    if keep >= 1.0 or not states:
        return join_sents(sents)
    k = max(1, int(round(len(states) * keep)))
    max_sent = int(g["node_sent"][states[k - 1]])
    return join_sents(sents[: max_sent + 1])


class ModelZoo:
    def __init__(self, tags: list[str], device=None, models_dir: Path | None = None,
                 groups: list[str] | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models_dir = Path(models_dir) if models_dir else OUTPUTS
        self.groups = groups or load_json(PROCESSED / "groups.json")
        self.graph_tags = [t for t in tags if t not in TEXT_TAGS]
        self.text_tags = [t for t in tags if t in TEXT_TAGS]
        self.models, self.drop, self.info = {}, {}, {}
        for t in self.graph_tags:
            fp = self.models_dir / f"model_{t}.pt"
            if not fp.exists():
                raise SystemExit(f"{fp} not found. Available tags: "
                                 f"{sorted(p.stem[6:] for p in self.models_dir.glob('model_*.pt'))}")
            model, groups, fl, ck = load_classifier(fp, self.device)
            if list(groups) != list(self.groups):
                raise SystemExit(f"Group order in {fp} differs from groups.json")
            self.models[t] = model
            self.drop[t] = HUNT_RELS if fl.get("no_iochunter") else None
            self.info[t] = {"ssm_impl": ck.get("ssm_impl"), "train_cfg": ck.get("train_cfg", {})}
        self.text_models = {}
        if self.text_tags:
            from .baselines import load_baseline
            for t in self.text_tags:
                self.text_models[t] = load_baseline(t, device=self.device)

    @property
    def tags(self) -> list[str]:
        return self.graph_tags + self.text_tags

    def preprocess_groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for t in self.graph_tags:
            out.setdefault(preprocess_of(t), []).append(t)
        return out

    @torch.no_grad()
    def score_graphs(self, graphs: list[dict], keeps: list[float], tags: list[str] | None = None,
                     bs: int = 64) -> dict:
        """-> {(tag, keep): probs (n, C)}"""
        tags = tags or self.graph_tags
        out = {}
        for keep in keeps:
            gs = [truncate_graph(g, keep) if keep < 1.0 else g for g in graphs]
            for t in tags:
                model = self.models[t]
                ps = []
                for i in range(0, len(gs), bs):
                    batch = collate(gs[i:i + bs], self.device, self.drop[t])
                    logits, _z = model(batch)
                    ps.append(torch.sigmoid(logits).float().cpu())
                out[(t, keep)] = torch.cat(ps).numpy() if ps else np.zeros((0, len(self.groups)))
        return out

    def score_texts(self, texts_by_keep: dict, tags: list[str] | None = None) -> dict:
        """texts_by_keep: {keep: [text,...]} -> {(tag, keep): probs}"""
        tags = tags or self.text_tags
        out = {}
        for keep, texts in texts_by_keep.items():
            for t in tags:
                out[(t, keep)] = self.text_models[t].score_texts(texts)
        return out


def summarize_probs(p: np.ndarray, true_idx: int, target_idx: int | None = None) -> dict:
    order = np.argsort(-p)
    rank = int(np.where(order == true_idx)[0][0]) + 1
    row = {"top1": int(order[0]), "p_top1": round(float(p[order[0]]), 5),
           "p_true": round(float(p[true_idx]), 5), "rank_true": rank,
           "top3": [int(i) for i in order[:3]],
           "probs": [round(float(x), 4) for x in p]}
    if target_idx is not None:
        row["p_target"] = round(float(p[target_idx]), 5)
    return row
