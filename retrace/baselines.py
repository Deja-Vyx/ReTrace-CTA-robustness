"""Text-only baselines (no graph). They answer a reviewer-style question: is a weakness found by
ReTrace specific to the graph model, or does it affect any attribution model?

B1  b1_tfidf : TF-IDF (1-2 grams) + class-balanced LogisticRegression
B2  b2_sbert : mean sentence embedding (from the SecureBERT cache) + small MLP, 3 seeds

    python -m retrace.baselines train --which b1,b2
    python -m retrace.baselines eval  --which b1,b2
"""
import argparse
import json
import pickle

import numpy as np

from src.utils import PROCESSED, load_json, read_jsonl, sent_split
from .common import BASELINE_DIR, log


def _split():
    docs = read_jsonl(PROCESSED / "docs.jsonl")
    groups = load_json(PROCESSED / "groups.json")
    g2i = {g: i for i, g in enumerate(groups)}
    tr = [(d["text"], g2i[d["group"]]) for d in docs if d["split"] == "train"]
    te = [(d["text"], g2i[d["group"]]) for d in docs if d["split"] == "test"]
    return tr, te, groups


class B1TfIdf:
    tag = "b1_tfidf"

    def __init__(self, vec=None, clf=None, groups=None):
        self.vec, self.clf, self.groups = vec, clf, groups

    @classmethod
    def train(cls, seed=42):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        tr, _te, groups = _split()
        X = [t for t, _y in tr]; y = [yy for _t, yy in tr]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
        Xv = vec.fit_transform(X)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
        clf.fit(Xv, y)
        return cls(vec, clf, groups)

    def score_texts(self, texts):
        if not texts:
            return np.zeros((0, len(self.groups)))
        Xv = self.vec.transform(texts)
        p = self.clf.predict_proba(Xv)
        # expand to all 20 classes in the order of `groups`
        full = np.zeros((len(texts), len(self.groups)))
        for j, c in enumerate(self.clf.classes_):
            full[:, c] = p[:, j]
        return full

    def save(self, path):
        pickle.dump({"vec": self.vec, "clf": self.clf, "groups": self.groups}, open(path, "wb"))

    @classmethod
    def load(cls, path, device=None):
        d = pickle.load(open(path, "rb"))
        return cls(d["vec"], d["clf"], d["groups"])


class B2Sbert:
    tag = "b2_sbert"

    def __init__(self, model=None, groups=None, device="cpu"):
        self.model, self.groups, self.device = model, groups, device
        self._enc = None

    def _encoder(self):
        if self._enc is None:
            from src.encoder import get_encoder
            from .sentcache import CachedSentenceEncoder
            from .common import CACHE_DIR
            self._enc = CachedSentenceEncoder(get_encoder(device=self.device),
                                              store=CACHE_DIR / "sent_emb.pkl")
        return self._enc

    def _doc_vec(self, text):
        import torch
        sents = sent_split(text)[:256]
        if not sents:
            return torch.zeros(768)
        return self._encoder().encode(sents).mean(0)

    @classmethod
    def train(cls, seed=42, device="cpu", epochs=40):
        import torch
        import torch.nn as nn
        from src.utils import set_seed
        set_seed(seed)
        tr, _te, groups = _split()
        obj = cls(None, groups, device)
        X = torch.stack([obj._doc_vec(t) for t, _y in tr]).float()
        y = torch.tensor([yy for _t, yy in tr])
        cnt = torch.bincount(y, minlength=len(groups)).float().clamp(min=1)
        w = (len(y) / (len(groups) * cnt))
        net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, len(groups)))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss(weight=w)
        net.train()
        for _ep in range(epochs):
            perm = torch.randperm(len(y))
            for i in range(0, len(y), 64):
                idx = perm[i:i + 64]
                opt.zero_grad()
                lossf(net(X[idx]), y[idx]).backward()
                opt.step()
        net.eval()
        obj.model = net
        obj._enc = None  # do not pickle the encoder
        return obj

    def score_texts(self, texts):
        import torch
        if not texts:
            return np.zeros((0, len(self.groups)))
        X = torch.stack([self._doc_vec(t) for t in texts]).float()
        with torch.no_grad():
            return torch.softmax(self.model(X), -1).cpu().numpy()

    def save(self, path):
        import torch
        torch.save({"state": self.model.state_dict(), "groups": self.groups}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        import torch
        import torch.nn as nn
        d = torch.load(path, map_location=device, weights_only=False)
        net = nn.Sequential(nn.Linear(768, 256), nn.ReLU(), nn.Dropout(0.2),
                            nn.Linear(256, len(d["groups"])))
        net.load_state_dict(d["state"]); net.eval()
        obj = cls(net, d["groups"], device)
        return obj


REGISTRY = {"b1_tfidf": (B1TfIdf, "b1_tfidf.pkl"), "b2_sbert": (B2Sbert, "b2_sbert.pt")}


def load_baseline(tag, device="cpu"):
    cls, fn = REGISTRY[tag]
    fp = BASELINE_DIR / fn
    if not fp.exists():
        raise SystemExit(f"{fp} not found. Run: python -m retrace.baselines train --which {tag[:2]}")
    return cls.load(fp, device=device)


def cmd_train(args):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    which = set(args.which.split(","))
    if "b1" in which:
        m = B1TfIdf.train()
        m.save(BASELINE_DIR / "b1_tfidf.pkl"); log("[b1] saved b1_tfidf.pkl")
    if "b2" in which:
        m = B2Sbert.train(device=args.device or "cpu")
        m.save(BASELINE_DIR / "b2_sbert.pt"); log("[b2] saved b2_sbert.pt")


def cmd_eval(args):
    from sklearn.metrics import f1_score
    _tr, te, groups = _split()
    which = set(args.which.split(","))
    tags = [t for t in ("b1_tfidf", "b2_sbert") if t[:2] in which]
    for tag in tags:
        m = load_baseline(tag, device=args.device or "cpu")
        probs = m.score_texts([t for t, _y in te])
        y = np.array([yy for _t, yy in te]); pred = probs.argmax(1)
        print(f"[{tag}] micro-F1={f1_score(y, pred, average='micro'):.4f} "
              f"macro-F1={f1_score(y, pred, labels=list(range(len(groups))), average='macro', zero_division=0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name); p.add_argument("--which", default="b1,b2"); p.add_argument("--device", default=None)
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
