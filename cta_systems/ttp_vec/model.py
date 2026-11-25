"""TTP-Vec (Hussain & Petrova): one-hot TTPs -> StandardScaler -> rarity weights -> classifier.

Keeps the authors' parameters exactly:
  knn  KNeighborsClassifier(n_neighbors=1)
  svm  SVC(kernel='linear', random_state=7)          (+ calibrated probabilities, see below;
       max_iter from the config: with (1/u)^3 weights up to 20^3 = 8000, libsvm converges very slowly on
       20 groups - the original code sets no limit; accuracy is stable well before the limit, see README)
  rf   RandomForestClassifier(random_state=7)
  dt   DecisionTreeClassifier(random_state=7)
  nb   GaussianNB()
Weights (weighting="rarity3", function get_amplified_rarity_weights / *-weighted-ttp.py):
  w(f) = (1/u(f))^3 / (1/max_u)^3, u(f) = number of DISTINCT groups having feature f; u = 0 -> w = 0.

Probabilities: ReTrace needs probabilities (delta_p, "confident flip" > 0.5). SVC is Platt-calibrated (sigmoid) with
CalibratedClassifierCV(SVC, method="sigmoid", cv=5, ensemble=False) - the approach recommended by sklearn >= 1.9
instead of SVC(probability=True) (deprecated in 1.9, removed in 1.11). The top-1 prediction is the
argmax of the probabilities EVERYWHERE (clean evaluation and robustness tests) for consistency. KNN k=1
and DT give 0/1 probabilities - inherent to those algorithms.
"""
import numpy as np

ALGOS = ("knn", "svm", "rf", "dt", "nb")


def make_classifier(algo: str, knn_k: int = 1, svm_kernel: str = "linear", svm_probability: bool = True,
                    svm_max_iter: int = -1, y=None):
    """Classifier with EXACTLY the authors' parameters. The `probability` argument is NOT passed to SVC
    (sklearn >= 1.9 warns even for probability=False); SVM probabilities come from CalibratedClassifierCV."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    if algo == "knn":
        return KNeighborsClassifier(n_neighbors=knn_k)
    if algo == "svm":
        base = SVC(kernel=svm_kernel, random_state=7, max_iter=svm_max_iter)
        if not svm_probability:
            return base
        return calibrated_svm(base, y)
    if algo == "rf":
        return RandomForestClassifier(random_state=7)
    if algo == "dt":
        return DecisionTreeClassifier(random_state=7)
    if algo == "nb":
        return GaussianNB()
    raise ValueError(algo)


def calibrated_svm(base, y):
    """Platt (sigmoid) on the SVC decision scores, stratified 5-fold CV then refit on all of train
    (the same idea as libsvm's SVC(probability=True)). Classes with too few samples (< 2) -> plain SVC;
    TTPVecModel.predict_proba then uses a softmax of the decision scores (only on tiny test subsets)."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold
    if y is None:
        return base
    cnt = np.bincount(np.asarray(y))
    k = int(min(5, cnt[cnt > 0].min()))
    if k < 2:
        return base
    return CalibratedClassifierCV(estimator=base, method="sigmoid", ensemble=False,
                                  cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=7))


def rarity_weights(X: np.ndarray, y: np.ndarray, power: float) -> np.ndarray:
    """Number of distinct groups using each feature -> weights; only called with the TRAINING X, y."""
    V = X.shape[1]
    usage = np.zeros(V)
    for c in np.unique(y):
        usage += (X[y == c].sum(axis=0) > 0)
    w = np.zeros(V)
    if usage.max() > 0:
        nz = usage > 0
        w[nz] = (1.0 / usage[nz]) ** power / (1.0 / usage.max()) ** power
    return w


class TTPVecModel:
    """One classifier (one algorithm, one weighting scheme) over the technique vocabulary of the training set."""

    def __init__(self, algo: str, weighting: str = "rarity3", n_classes: int = 20, knn_k: int = 1,
                 svm_kernel: str = "linear", svm_max_iter: int = -1):
        self.algo = algo
        self.weighting = weighting
        self.n_classes = n_classes
        self.knn_k = knn_k
        self.svm_kernel = svm_kernel
        self.svm_probability = True     # False only inside CV (labels only)
        self.svm_max_iter = int(svm_max_iter)
        self.n_convergence_warnings = 0
        self.vocab: list[str] = []
        self.index: dict[str, int] = {}
        self.scaler = None
        self.w = None
        self.clf = None

    @property
    def tag(self) -> str:
        return f"ttp_vec_{self.algo}" + ("" if self.weighting == "rarity3" else f"_{self.weighting}")

    # ------------------------------------------------------------------ features
    def featurize(self, tech_sets: list) -> np.ndarray:
        X = np.zeros((len(tech_sets), len(self.vocab)), dtype=np.float64)
        for i, ts in enumerate(tech_sets):
            for t in ts:
                j = self.index.get(t)
                if j is not None:
                    X[i, j] = 1.0
        return X

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(X)
        return Z * self.w if self.w is not None else Z

    # ------------------------------------------------------------------ training / prediction
    def fit(self, tech_sets: list, y: np.ndarray, vocab: list[str] | None = None):
        from sklearn.preprocessing import StandardScaler
        self.vocab = sorted(vocab if vocab is not None else set().union(*[set(t) for t in tech_sets]))
        self.index = {t: i for i, t in enumerate(self.vocab)}
        X = self.featurize(tech_sets)
        self.scaler = StandardScaler().fit(X)                  # TRAIN only (fixes the leakage of the original code)
        if self.weighting.startswith("rarity"):
            power = float(self.weighting[len("rarity"):] or 1)
            self.w = rarity_weights(X, np.asarray(y), power)
        else:
            self.w = None
        self.clf = make_classifier(self.algo, self.knn_k, self.svm_kernel, self.svm_probability, self.svm_max_iter,
                                   y=np.asarray(y))
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            self.clf.fit(self._transform(X), np.asarray(y))
        self.n_convergence_warnings = sum(1 for w in caught if issubclass(w.category, ConvergenceWarning))
        return self

    def predict_proba(self, tech_sets: list) -> np.ndarray:
        if not tech_sets:
            return np.zeros((0, self.n_classes))
        Z = self._transform(self.featurize(tech_sets))
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(Z)
        else:                                   # plain SVC (tiny sets): softmax of the ovr decision scores
            d = np.atleast_2d(self.clf.decision_function(Z))
            e = np.exp(d - d.max(axis=1, keepdims=True))
            p = e / e.sum(axis=1, keepdims=True)
        full = np.zeros((len(tech_sets), self.n_classes))
        for j, c in enumerate(self.clf.classes_):
            full[:, int(c)] = p[:, j]
        return full

    def describe(self) -> dict:
        nz = int((self.w > 0).sum()) if self.w is not None else len(self.vocab)
        return {"tag": self.tag, "algo": self.algo, "weighting": self.weighting, "vocab": len(self.vocab),
                "features_with_weight": nz, "convergence_warnings": self.n_convergence_warnings,
                "max_weight": float(self.w.max()) if self.w is not None and len(self.w) else None}
