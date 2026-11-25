# Case-study systems: TTP-Match and TTP-Vec

`cta_systems/` re-implements two recent attribution systems from their authors' code, trains them
on **the same CTI2Attacker data** that ThreatMAMBA uses, and scores them with **the same ReTrace
benchmark** (levels 1 and 2, same metrics, same confidence intervals, same verdict rule). The
names TTP-Match and TTP-Vec are the names used in the paper.

| Name in the paper | Tag | Original work | Original code |
| --- | --- | --- | --- |
| TTP-Match | `ttp_match` | Guru, Moss & Kochenderfer, *On Technique Identification and Threat-Actor Attribution using LLMs and Embedding Models*, Cyber-AI 2025 (arXiv 2505.11547) | github.com/kylag/ttp_attribution |
| TTP-Vec | `ttp_vec_<knn\|svm\|rf\|dt\|nb>` | Hussain & Petrova, *Threat Actor Attribution Applying a Tactics–Techniques–Procedures Approach: An Empirical Investigation*, Future Internet 18(8):433, 2026 | Supplementary Materials (`futureinternet-18-00433-s001`) |

Level 3 (direct graph edits) does not apply because neither system uses a graph. The code only
**reads** `data/` and `outputs/retrace/` and only **writes** `outputs/cta_systems/`.

---

## TTP-Vec (`cta_systems/ttp_vec/`)

**Kept from the authors' code** (`*-weighted-ttp.py`):

- one-hot features over ATT&CK techniques and sub-techniques;
- `StandardScaler`, then multiplication by the rarity weights `(1/number of groups using it)^3 / (1/max)^3`
  (`get_amplified_rarity_weights`);
- the five classifiers `KNeighborsClassifier(n_neighbors=1)`, `SVC(kernel='linear', random_state=7)`,
  `RandomForestClassifier(random_state=7)`, `DecisionTreeClassifier(random_state=7)`, `GaussianNB()`;
- the stratified 10-fold CV protocol with `random_state=7`.

**Changed, and why:**

| In the paper / original code | Here | Reason |
| --- | --- | --- |
| 471 columns: TTPs + victim country/sector/sub-sector (EuRepoC) | TTPs only | CTI2Attacker has no victim fields. The paper reports 55–66% for the TTP-only configuration |
| TTPs extracted by ChatGPT-4 and reviewed by hand | TTPs assigned by **ThreatMAMBA's TTP extractor** (Technique nodes of the CSKG) | Re-runnable for thousands of variants. Both systems share the same features and differ only in the classifier, which isolates the role of the representation |
| Scaler and weights computed on all labelled data before the split | Computed on train only | The original code leaks labels; the CV table reports both variants |
| Random 70/30 split | ThreatMAMBA's train/test split (2,515 / 629) | Same evaluation data |
| SVC without probabilities | Platt calibration with `CalibratedClassifierCV(SVC, method="sigmoid", cv=5, ensemble=False)`; top-1 is the argmax of the probabilities everywhere | ReTrace needs probabilities for Δp and the "confident flip > 0.5" rule. `SVC(probability=True)` is deprecated from scikit-learn 1.9 |
| SVC without an iteration limit | `max_iter = 1e6` (`svm_max_iter`) | Weights up to 20³ = 8000 make libsvm converge very slowly on 20 groups; on synthetic data of the same size, 1e5–1e7 iterations give the same accuracy. Non-convergence warnings are counted in the log and in `ttp_vec_meta.json` |

All algorithms are scored. The main tables use `ttp_vec_knn`, `ttp_vec_svm` and `ttp_vec_rf` (KNN
is the paper's main algorithm, RF the best TTP-only one); DT and NB are in the report appendix.

---

## TTP-Match (`cta_systems/ttp_match/`)

**Kept from the authors' algorithm** (`everything.py` / `everythingHYDE.py`):

- the report is split into non-overlapping windows of 3 units;
- each window is matched to the nearest ATT&CK technique descriptions by cosine similarity (top-k,
  optional threshold), and matches are counted;
- the group profile P(t|A) is the normalised sum of counts over the group's training reports;
- score(A) = P(A) × Σ_t P(t|A) × y_t, with the "expert prior" = group frequency in train;
- parent techniques are dropped when a sub-technique exists (`rid_technique.py`).

**Changed, and why:**

| In the original code | Here | Reason |
| --- | --- | --- |
| OpenAI `text-embedding-3-large` | `BAAI/bge-large-en-v1.5` run locally (CLS + L2 normalisation) | No paid API; configurable in `configs/cta_systems.yaml` |
| 3-**line** windows | 3-**sentence** windows | CTI2Attacker texts contain no line breaks, so line windows would cover the whole report |
| Techniques keyed by **name** | Keyed by ATT&CK v14 **ID** (531 entries, local STIX) | 23 names are duplicated (e.g. "Spearphishing Attachment" is T1566.001 and T1598.002) |
| Fixed cosine threshold 0.65 | Threshold from cosine quantiles on train, chosen on validation | 0.65 only suits the OpenAI scale |
| "Test" loop run on validation; `argmax` picks the worst split; ties broken by folder order | Separate stratified validation split (15%); test = the 629 reports; tie-aware ranks | Bug fixes |
| Reports without any match score 0 for every group | 1e-6 smoothing on the likelihood, so they fall back to the prior | Avoids arbitrary predictions; the ranking is unchanged when there are matches |
| No probabilities | Temperature T chosen on validation: p = softmax(log score / T) | ReTrace needs probabilities; T does not change top-1 |
| HyDE (GPT-4 pseudo passages) | Not used | Needs an LLM; per the paper HyDE only moves the mean rank from 7.75 to 7.55 |

The clean-accuracy table lists the paper's Algorithm 1 (top-1, no threshold, expert prior), the
`everything.py` configuration (top-5, no threshold, no prior) and the configuration selected on
validation, which is the `ttp_match` tag used in every robustness table.

---

## How the evaluation matches ReTrace

- **Same variants.** `evaluation/rescore.py` rescores the variants stored in
  `outputs/retrace/runs/{rand,worst,llm}/variants.jsonl.gz` (levels 0–2, valid variants only). No
  variant is regenerated and no LLM is called.
- **Same prefixes.** Each variant's CSKG is rebuilt to find the sentence ThreatMAMBA uses as the
  20/40/60/80% cut-off, and the text is cut exactly there (as `retrace.models.text_cut`). The
  report checks that the rebuilt State counts match the original run.
- **Same metrics.** `evaluation/rows.py` uses the definitions of `retrace.metrics.build_table`; a
  test checks that both functions agree.
- **flip_c.** The 144-report benchmark is selected because ThreatMAMBA attributes all of it
  correctly, so other systems may already be wrong on part of it. Robustness tables therefore
  compute flips on the reports that *the system itself* attributes correctly.
- **Paired comparison.** Section 7 of the report pairs documents on the intersection (reports
  both systems attribute correctly) and reports the flip_c difference with a CI and a verdict.
- **Ties.** RF, KNN and DT often give equal probabilities; top-1 uses a stable sort, so the group
  only changes when the probabilities actually change.
- **Retraining.** Every output folder stores the sha256 fingerprint of the model file; after
  retraining, the old scores of that model are dropped and rescored.
- **Transfer vs adaptive.** The strongest variants of the `worst`/`llm` runs are chosen with
  ThreatMAMBA's scores, so for the other systems they are transfer attacks.
  `evaluation/adaptive.py` repeats the procedure of `retrace.runner` (same operators, same grid in
  `configs/retrace.yaml`, same per-document seed, same greedy-k / best-of-5 / targeted search) with
  the case-study system as the black box; the E8 decoy B is that system's second-ranked group.

---

## Limitations

1. No 2025–2026 work reports results on CTI2Attacker, so every number here comes from our own
   training and evaluation and is not directly comparable with the original papers.
2. TTP-Vec uses only the TTP-only configuration, with TTPs from ThreatMAMBA's extractor instead
   of ChatGPT plus manual review.
3. TTP-Match uses open embeddings instead of OpenAI and no HyDE.
4. The benchmark is selected with ThreatMAMBA. flip_c and the intersection-based comparison
   reduce, but do not remove, this bias.
