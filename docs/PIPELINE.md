# ReTrace pipeline: step by step

This page details what `scripts/run_retrace.sh` and `scripts/run_cta_systems.sh` do, how to run
each step by hand, what they write, and how to troubleshoot them. Installation is covered in the
[README](../README.md).

Run every command from the root of the ThreatMamba-reproduction checkout (after the overlay and
the patch are in place), with the virtual environment active and `export PYTHONHASHSEED=0`.

---

## 1. Prerequisites

ReTrace evaluates the trained ThreatMAMBA checkpoints of the reproduction. It needs:

| Path | Produced by (reproduction) |
| --- | --- |
| `data/processed/docs.jsonl`, `data/processed/groups.json` | Stage 1 (`python -m src.data_prep all`) |
| `data/attck/attck_v14.json`, `data/attck/enterprise-attack-14.1.json` | Stage 1 |
| `outputs/ttp_extractor.pt` | Stage 3 (`python -m src.ttp_extract ...`) |
| `data/cskg/*.pt` (3,144 graphs) | Stage 4 (`python -m src.cskg_builder build`) |
| `outputs/model_main.pt`, `model_no_cl.pt`, `model_no_mamba.pt` | Stage 5 (`bash scripts/train_all.sh`) |

The checkpoints are trained with the `mamba-ssm` CUDA kernel, so real runs need the same GPU
environment as the reproduction. The unit tests run on CPU.

**LLM operators (optional).** T1, T2, E2-neutral and E6 call a local LLM through Ollama
(default `qwen3.5:9b`, set in `configs/retrace.yaml` → `llm.model`). Without an LLM, the script
skips these operators and still produces the full report for the others.

```bash
ollama serve &              # if it is not running yet
ollama pull qwen3.5:9b
python -m retrace.llm       # must print "OK" and "PASS"
```

If Ollama runs on another host (for example on Windows while the experiments run in WSL), set
`OLLAMA_HOST=http://<host-ip>:11434` in the shell or in the repository `.env`; ReTrace only
reads `OLLAMA_HOST` and `OLLAMA_MODEL` from that file.

---

## 2. One command

```bash
DOCS=pilot LLM=0 bash scripts/run_retrace.sh     # quick check on 50 pilot documents -> outputs/retrace/REPORT_pilot.md
bash scripts/run_retrace.sh                      # full run on the eligible set -> outputs/retrace/REPORT.md
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `DOCS` | `eligible` | `eligible` (about 144 documents) or `pilot` (50) |
| `LLM` | `auto` | `auto`: run the LLM operators if `python -m retrace.llm` passes · `1`: required (stop otherwise) · `0`: skip |
| `LLM_DOCS` | `pilot` | document set for the (slow) LLM operators; `eligible` for full coverage |
| `FORCE` | empty | `1` re-runs every step; existing runs are **resumed**, not deleted |
| `MODELS` | `main,no_cl,no_mamba` | checkpoints under test |

The script is resumable: finished steps are marked in `outputs/retrace/.done/`, and the runner
skips variants that are already scored. Per-step logs are in `outputs/retrace/logs/`. It stops
immediately if `preflight`, the unit tests or `parity` fail.

---

## 3. Steps

### A. Environment and pipeline fidelity

| Step | Command | Passes when |
| --- | --- | --- |
| A1 | `python -m retrace.preflight` | prints `PASS`; writes `freeze.json` (git commit, SHA-256 of the checkpoints, library versions) |
| A2 | `THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_retrace.py` | `17/17 PASS` |
| A3 | `python -m retrace.pipeline parity --n 20 --models main,no_cl,no_mamba` | ≥19/20 structurally identical **and** `top-1 label changes = 0` for every model; writes `parity.json` |

In A3, `ok (node order permuted only)` and `edges differ (... co_occurs ...)` are harmless (see
Troubleshooting). The deciding criterion is that predictions do not change.

### B. Evaluation data

| Step | Command | Output |
| --- | --- | --- |
| B1 | `python -m retrace.lexicons export-review` | `alias_review.csv`: aliases of the 20 groups. Optionally set `keep=0` for false-positive aliases; a reviewed file is never overwritten |
| B2 | `python -m retrace.eligible build --models main,no_cl,no_mamba --tier lenient --cap 25 --pilot 50` | `eligible_v0.jsonl`: test reports that pass the CTI filter **and** that all three models attribute correctly at 100%, a seeded random sample of at most 25 per group; `pilot.txt` (50 documents) |
| B3 | `python -m retrace.pools build --what carriers,distractors,false-flag` | `pools/`: benign sentences (T5), IOC carrier sentences (E7), traces of decoy groups taken from existing training reports (E8) |

### C. Generate and score variants

Every variant is rebuilt into a CSKG with **exactly** the reproduction pipeline and scored by all
three models at five report prefixes (20/40/60/80/100%).

| Step | Run | Command (abridged) | Purpose |
| --- | --- | --- | --- |
| C1 | `rand` | `runner --run rand --ops ID,T3,T4,T5,T6,E1,E2,E3,E4,E5,E7,E8,G1,G2,G3,G4,G5 --strategy random` | **average** robustness and curves over k |
| C2 | `worst` | `runner --run worst --ops E1,E2,E3,G1 --strategy greedy` | **strongest** item-based attack: each step picks the item that lowers p(true group) the most |
| C3 | `worst` | `runner --run worst --ops T5,E4,E5,E7,E8,G3,G4 --strategy best_of_n --n 5` | strongest of 5 candidates; E8 keeps the candidate that raises p(B) the most (targeted) |
| C4 | `llm` | `runner --run llm --docs $LLM_DOCS --ops T1,T2,E2,E6 --ops-section ops_llm --strategy best_of_n --n 3` | LLM rewrites (only with Ollama) |
| C5 | `llm` | `python -m retrace.validate nli --run llm` | bidirectional NLI check (DeBERTa-v3) for T1/T2; a failure only warns |

The search only uses the output probabilities of `main` (black-box). `no_cl` and `no_mamba` are
scored on the same variants, which measures transfer.

### D. Aggregation, analysis, report (always re-run, fast)

| Step | Command | Output |
| --- | --- | --- |
| D1 | `python -m retrace.metrics summarize --run <run>` | `runs/<run>/tables/*.csv`: flip, violations, Δp with 95% CIs per operator / parameter k / level / prefix |
| D2 | `python -m retrace.analysis size-probe --n 2000` | S1: graph-size shortcut in contrastive learning |
| D3 | `python -m retrace.analysis ttp-fp` | S2: share of benign sentences labelled by the TTP extractor |
| D4 | `python -m retrace.analysis all --run <run>` | S3–S6, stage decomposition, per-group robustness, curves, figures, LaTeX table |
| D5 | `python -m retrace.report` | **`outputs/retrace/REPORT.md`** + `report_numbers.json` |

---

## 4. Reading the results

Every number in `REPORT.md` comes with a 95% document-level bootstrap CI (1,000 resamples).
Verdicts are computed from the data with two conditions: (a) the 95% CI excludes 0, (b) the
effect is at least 0.01 (one percentage point).

- **supported**: both conditions hold.
- **weak**: only one holds.
- **not_supported**: neither holds.
- **na**: fewer than 5 documents.

| Metric | Definition |
| --- | --- |
| flip | the variant's top-1 label differs from the original's (same model, same prefix) |
| flip_c | flip restricted to reports the model attributes correctly at 100% |
| violation | the operator's expectation is violated. INV: flip. DIR: **confident** flip (p > 0.5). TGT: steered to the decoy group B |
| Δp | p(true group \| original) − p(true group \| variant); positive = less confident in the true group |
| ASR | (E8) share of variants attributed to the decoy group B |
| Δacc | top-1 accuracy of the originals minus that of the variants (same model, same prefix). Equal to flip at 100%; smaller than flip at early prefixes, where many originals are already wrong |
| micro / macro | micro = mean over documents (default); macro = unweighted mean over groups |

Shortcut hypotheses (section 4 of the report):

| Code | Question | Source |
| --- | --- | --- |
| S1 | Can contrastive learning be solved by **counting nodes**? (splice negatives are about twice as large as positives) | `size-probe` |
| S2 | Does the TTP extractor label **benign sentences**? | distractor pool |
| S3a | Does the model **rely on the true group name**? (E4 masks it → does p(true group) drop?) | run `rand` |
| S3b | Does **another group's name** pull the prediction? (E8 inserts B's name → does p(B) rise?) | runs `rand`, `worst` |
| S4 | Is the Mamba branch **order-sensitive**? (G3 reverse: main vs no_mamba) | run `rand` |
| S5 | Do some groups **attract** wrong predictions? | run `rand` |
| S6 | Does the model rely on **specific** or **common** techniques? (E2 drop) | runs `worst`/`rand` |

Figures are in `runs/rand/figures/` (`violation_by_op`, `flip_vs_k_main`,
`flip_vs_timeline_by_level`, `heatmap_op_keep_<model>`, each as `.pdf` and `.png`); the LaTeX
table is `runs/rand/tables/flip_table.tex`.

**Health checks before trusting the numbers** (section 0 of the report): `ID` must have flip
0.000 for every model, `T6` must be close to 0, and parity must pass.

---

## 5. Case-study systems (`scripts/run_cta_systems.sh`)

Run after the ReTrace runs exist. See [CASE_STUDIES.md](CASE_STUDIES.md) for what was kept from
and changed in the authors' code.

| Step | Command | What it does |
| --- | --- | --- |
| A1 | `CTA_SYSTEMS_TEST=1 python tests/test_cta_systems.py` | CPU tests with synthetic data in a temporary folder |
| B1 | `python -m cta_systems.ttp_vec.train` | technique features from `data/cskg`; 5 algorithms × 2 weightings; 10-fold CV |
| B2 | `python -m cta_systems.ttp_match.train` | embeds 531 techniques and ~40k windows; tunes on validation |
| C1 | `python -m cta_systems.evaluation.rescore` | rescores the stored level 0–2 variants of rand/worst/llm at 5 prefixes (transfer) |
| C2 | `python -m cta_systems.evaluation.adaptive --model ttp_match` (and `ttp_vec_rf`) | adaptive search against each system itself |
| D | `python -m cta_systems.evaluation.report [--fmt pct]` | comparison report, tables, figures |

| Variable | Meaning |
| --- | --- |
| `LIMIT=200` | dry run: 200 variants per run, 5 documents for the adaptive search |
| `ADAPT=0` | skip the adaptive attacks (C2) |
| `FORCE=1` | re-run every step (existing results are resumed) |

The first run of B2 downloads `BAAI/bge-large-en-v1.5` (~1.3 GB) from Hugging Face. Rebuilding
the CSKG of every variant (to find the prefix cut and the technique set) is cached in
`outputs/cta_systems/cache/prepared.jsonl.gz`.

---

## 6. Outputs

```
outputs/retrace/
  REPORT.md  report_numbers.json         # summary report + every number in it
  freeze.json  parity.json               # frozen baseline, parity result
  eligible_v0.jsonl  eligible_summary.csv  pilot.txt  alias_review.csv
  pools/{distractors,carriers,false_flag}.json
  analysis/{size_probe,ttp_fp}.json
  cache/sent_emb.pkl  cache/llm.sqlite   # caches (safe to delete, rebuilt on demand)
  logs/  .done/                          # script logs and step markers
  runs/{rand,worst,llm}/
    variants.jsonl.gz  scores.jsonl.gz  config.yaml  log.txt
    tables/*.csv  tables/flip_table.tex  analysis/*.json  figures/*.png|pdf

outputs/cta_systems/
  REPORT_CTA_SYSTEMS.md  REPORT_CTA_SYSTEMS_pct.md  report_numbers.json
  models/{ttp_vec,ttp_match}.pkl
  tables/  figures/  runs/<run>/  runs/adapt_<tag>/  cache/  logs/
```

ReTrace only **reads** `data/cskg`, `data/processed`, `outputs/model_*.pt`,
`outputs/ttp_extractor.pt` and the embedding cache, so it never overwrites reproduction results.
Do not run `python -m src.train` without `--suffix` (it overwrites `model_main.pt`) or
`python -m src.evaluate all` (it overwrites `metrics_all.csv`).

---

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `parity` prints `ok (node order permuted only)` | Normal. ioc-finder returns same-type IOCs of a sentence in a hash-dependent order, so a rebuilt graph only differs in node indices. Predictions do not change |
| `parity` prints `edges differ (...; due to the 6-IOC-per-sentence co_occurs limit)` | Not a ReTrace error. `cskg_builder` links `co_occurs` for the first 6 IOCs of a sentence only; for sentences with more IOCs, which 6 are linked depends on the order. Verified with the real models: Δp ≤ 4e-7, no label change |
| `parity` FAILS with label changes | Environment drift: compare `ioc-finder`, `transformers` and `ttp_extractor.pt` with `freeze.json` |
| `Cannot load model_*.pt: SSM architecture mismatch` | A `mamba` checkpoint needs a GPU + `mamba-ssm`. `echo $THREATMAMBA_SSM` must be empty |
| `ID` has a non-zero flip in the report | Stop: the pipeline is not deterministic. Re-run A3 to find the cause |
| `[pools] ERROR: no carrier template for ...` | Report the issue with `outputs/retrace/logs/B3_pools.log` |
| E4 masks ordinary words ("silence", "tick") | Review `alias_review.csv`, set `keep=0`, then re-run from B2: `rm outputs/retrace/.done/B2* outputs/retrace/.done/C*` |
| `[llm] cannot connect to Ollama at ...` | Ollama is not running or not reachable; see section 1, then `python -m retrace.llm` until it prints `PASS` |
| `[llm] ... does not have model 'qwen3.5:9b'` | `ollama pull qwen3.5:9b` (update Ollama if it is too old) |
| Invalid JSON or very slow LLM calls | Check `ollama ps`. Run with `LLM=0` first and add the LLM operators later with `LLM=1` (other steps are skipped) |
| Re-run a run from scratch | `mv outputs/retrace/runs/rand outputs/retrace/runs/rand_old && rm outputs/retrace/.done/C1_rand` |
| `BAAI/bge-large-en-v1.5` cannot be downloaded | Check access to huggingface.co, or switch to `BAAI/bge-base-en-v1.5` in `configs/cta_systems.yaml` and re-run from B2 |
| Out of GPU memory (TTP-Match) | Lower `ttp_match.batch_size` (64 → 16) in `configs/cta_systems.yaml` |
| Low "State counts matching the original run" | The rebuilt graphs differ from the ReTrace run: check `PYTHONHASHSEED=0` and that `outputs/ttp_extractor.pt` did not change |

---

## 8. Optional experiments

Not part of the main scripts.

- **Human validation of LLM rewrites:** `python -m retrace.validate export-human --run llm --n 150`,
  two annotators fill in `same_meaning`/`same_actor`, then
  `python -m retrace.validate kappa --a human_A.csv --b human_B.csv` (target κ ≥ 0.6).
- **Seed variance and same-size contrastive negatives:** `bash scripts/train_seeds.sh`, then add
  the new tags to `MODELS=`.
- **Text-only baselines:** `python -m retrace.baselines train --which b1,b2`, then
  `MODELS=main,no_cl,no_mamba,b1_tfidf,b2_sbert`.
- **Training on name-masked text:** `python -m retrace.mitigate build-masked`, then
  `python -m src.train --cskg-dir data/cskg_masked --suffix masked`.
