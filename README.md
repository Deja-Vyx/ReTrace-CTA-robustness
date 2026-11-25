# ReTrace: Adversarial Robustness Evaluation of Cyber Threat Attribution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Conference](https://img.shields.io/badge/NSS--SocialSec-2026-blue.svg)](https://nsclab.org/nss2026/)
[![Paper](https://img.shields.io/badge/paper-coming%20soon-lightgrey.svg)](#citation)

Official code of the paper

> **Rewriting the Traces: Adversarial Robustness Evaluation of Temporal Graph-Based Cyber Threat Attribution**
> Duong Chi Vy, Hien Do Hoang, Van-Hau Pham, Phan The Duy.
> Accepted at [NSS-SocialSec 2026](https://nsclab.org/nss2026/) (Springer LNCS).
> Paper link: *to be added once the proceedings are published.*

> 🇻🇳 Bản tiếng Việt: [README.vi.md](README.vi.md)

This repository contains **only the ReTrace code**. ReTrace evaluates attribution models that are
trained with the ThreatMAMBA reproduction, so it runs on top of
**[Deja-Vyx/ThreatMamba-reproduction](https://github.com/Deja-Vyx/ThreatMamba-reproduction)**
(data preparation, TTP extractor, CSKG builder, ThreatMAMBA models). The reproduction is not
duplicated here.

---

## Contents

- [What ReTrace does](#what-retrace-does)
- [Main findings](#main-findings)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Running ReTrace](#running-retrace)
- [Perturbation operators](#perturbation-operators)
- [Metrics and protocol](#metrics-and-protocol)
- [Paper-to-code map](#paper-to-code-map)
- [Changes to the reproduction (patch)](#changes-to-the-reproduction-patch)
- [Testing](#testing)
- [Citation](#citation)
- [Acknowledgments and license](#acknowledgments-and-license)

---

## What ReTrace does

Cyber threat attribution (CTA) systems map cyber threat intelligence (CTI) reports to the threat
group behind them. They are usually evaluated on clean or truncated reports, although attribution
is adversarial: groups change their traces and plant false flags, and reports change with them.
ReTrace is a methodology for evaluating the adversarial robustness of CTA systems:

- **Threat model.** Adversarial CTA is formalised as a constrained black-box problem over report
  variants. Three sources of report change are separated: reporting variation (R), operational
  manipulation by the attacker (A) and direct report manipulation (M).
- **19 perturbation operators** at three levels (text, evidence, graph), each with a testable
  expectation (INV, DIR, TGT) and automatic validity checks (IOC preservation, no new group names,
  length ratio, bidirectional NLI for LLM rewrites).
- **Black-box search** (random, greedy-k, best-of-N, targeted) in **transfer** and **adaptive**
  modes.
- **Scoring on report prefixes**: the first x% of evidence units in narrative order.
- **A paired protocol** that compares systems with different clean accuracy on the reports both
  attribute correctly, with document-level bootstrap confidence intervals.
- **Stage diagnostics**: stage decomposition (extraction / representation / classifier), graph
  diagnosis and six shortcut hypotheses (S1–S6).

ReTrace is applied to three case-study systems: **ThreatMAMBA** (temporal knowledge graph),
**TTP-Match** (embedding-based technique matching, Guru et al. 2025) and **TTP-Vec** (technique-vector
classifiers, Hussain & Petrova 2026).

## Main findings

From the paper (12,538 valid variants of 144 real CTI reports from 12 groups):

- **Technique sentences are the main attack surface.** Removing the sentences of one to three
  techniques, chosen by black-box search, flips 22–72% of correct attributions (graph-based system:
  22% flipped, only 6% confidently misattributed). Targeted technique false flags succeed 14–30%.
- **Most flips come with a change in the extracted evidence**: intended when technique sentences
  are removed, an extraction error when benign or IOC-carrying sentences receive techniques (the TTP
  extractor labels 82% of benign sentences). Direct graph edits flip at most 4%.
- **The evidence representation decides how far a change propagates.** On reports both systems
  attribute correctly, adaptive manipulation flips 12.4–15.0 percentage points more attributions of
  the technique-based systems than of the graph-based system, a gap that transfer attacks
  underestimate (+3.1 / +4.5 pp).

## Repository layout

```
ReTrace-CTA-robustness/
├── retrace/                    # ReTrace methodology (evaluates the ThreatMAMBA checkpoints)
│   ├── operators/              #   T1–T6 (text), E1–E8 (evidence), G1–G5 (graph) + ID control
│   ├── pipeline.py             #   text -> CSKG with the reproduction's GraphBuilder; parity check
│   ├── validate.py             #   validity checks, NLI, human-evaluation export, Cohen's kappa
│   ├── search.py  runner.py    #   black-box search strategies; generate + score runs (resumable)
│   ├── models.py               #   load/score checkpoints at the report prefixes
│   ├── eligible.py  pools.py   #   evaluation pool; benign / carrier / false-flag pools
│   ├── lexicons.py  llm.py     #   aliases, abbreviations, dates; Ollama client with SQLite cache
│   ├── metrics.py  analysis.py #   flip, Δp, ASR, Δacc + bootstrap CIs; S1–S6, stage decomposition
│   ├── report.py               #   outputs/retrace/REPORT.md
│   └── preflight.py  sentcache.py  baselines.py  mitigate.py  config.py  common.py
├── cta_systems/                # case-study systems evaluated with ReTrace
│   ├── ttp_match/              #   TTP-Match (Guru et al. 2025)
│   ├── ttp_vec/                #   TTP-Vec (Hussain & Petrova 2026)
│   ├── evaluation/             #   transfer rescoring, adaptive attacks, comparison report
│   └── prepare.py  common.py
├── configs/
│   ├── retrace.yaml            # operator grids, LLM, validity thresholds, prefixes
│   └── cta_systems.yaml        # TTP-Match / TTP-Vec settings, adaptive attacks
├── scripts/
│   ├── run_retrace.sh          # full ReTrace evaluation of ThreatMAMBA (one command)
│   ├── run_cta_systems.sh      # TTP-Match and TTP-Vec on the ReTrace benchmark (one command)
│   └── train_seeds.sh          # optional: extra seeds, same-size contrastive negatives
├── tests/
│   ├── test_retrace.py         # 17 CPU tests
│   └── test_cta_systems.py     # 14 CPU tests
├── patches/
│   └── threatmamba-reproduction.patch   # small, flag-guarded changes to 4 reproduction files
└── docs/
    ├── PIPELINE.md             # every step, outputs, troubleshooting
    └── CASE_STUDIES.md         # what was kept from / changed in the TTP-Match and TTP-Vec code
```

## Installation

**1. Set up the reproduction.** Follow the README of
[ThreatMamba-reproduction](https://github.com/Deja-Vyx/ThreatMamba-reproduction) (GPU,
`mamba-ssm`, data download) and run its pipeline through training, so that the following exist:
`data/processed/`, `data/attck/`, `data/cskg/*.pt`, `outputs/ttp_extractor.pt`,
`outputs/model_main.pt`, `outputs/model_no_cl.pt`, `outputs/model_no_mamba.pt`.

**2. Add ReTrace on top and apply the patch.**

```bash
git clone https://github.com/Deja-Vyx/ThreatMamba-reproduction.git
git clone https://github.com/Deja-Vyx/ReTrace-CTA-robustness.git

cd ThreatMamba-reproduction
git apply ../ReTrace-CTA-robustness/patches/threatmamba-reproduction.patch
cp -r ../ReTrace-CTA-robustness/retrace ../ReTrace-CTA-robustness/cta_systems .
cp ../ReTrace-CTA-robustness/configs/*.yaml configs/
cp ../ReTrace-CTA-robustness/scripts/*.sh scripts/
cp ../ReTrace-CTA-robustness/tests/*.py tests/
```

The patch was made against commit `61cc452` of the reproduction. Its changes are guarded by flags
and keep the default behaviour, so existing checkpoints and results stay valid
(see [below](#changes-to-the-reproduction-patch)).

**3. Dependencies.** No packages beyond the reproduction's `requirements.txt` are needed. Optional
components:

| Component | Used for | Setup |
| --- | --- | --- |
| [Ollama](https://ollama.com) + `qwen3.5:9b` | LLM operators T1, T2, E2-neutral, E6 | `ollama pull qwen3.5:9b`; set `OLLAMA_HOST` if Ollama runs on another host |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | bidirectional NLI check of LLM rewrites | downloaded from Hugging Face on first use |
| `BAAI/bge-large-en-v1.5` | TTP-Match embeddings | downloaded from Hugging Face on first use |

## Running ReTrace

```bash
export PYTHONHASHSEED=0

# ReTrace on the ThreatMAMBA checkpoints (main, no_cl, no_mamba)
DOCS=pilot LLM=0 bash scripts/run_retrace.sh   # quick check: 50 documents, no LLM
bash scripts/run_retrace.sh                     # full run -> outputs/retrace/REPORT.md

# case-study systems on the same benchmark (after the ReTrace runs exist)
bash scripts/run_cta_systems.sh                 # -> outputs/cta_systems/REPORT_CTA_SYSTEMS.md
```

Both scripts are resumable: run the same command again after an interruption and finished steps
are skipped. Useful variables: `DOCS`, `LLM`, `LLM_DOCS`, `FORCE`, `MODELS` for
`run_retrace.sh`; `LIMIT`, `ADAPT`, `FORCE` for `run_cta_systems.sh`. Every step can also be run by
hand; see [docs/PIPELINE.md](docs/PIPELINE.md).

## Perturbation operators

Scenario: R = reporting variation, A = operational manipulation, D = diagnostic only.
Expectation: **INV** the label must not change · **DIR** p(true group) may drop but no confident
flip · **TGT** the report must not be steered to the decoy group B.

| Level | Code | Operator | Scenario | Expectation |
| --- | --- | --- | --- | --- |
| 0 | ID | Rebuild the unchanged report (determinism control) | – | INV |
| 1 | T1 | LLM paraphrase of k sentences (IOCs and names kept) | R | INV |
| 1 | T2 | LLM split/merge of k sentences, active/passive voice | R | INV |
| 1 | T3 | Abbreviation ↔ full form (e.g. RDP) | R | INV |
| 1 | T4 | Move the IOC appendix to the top | R | INV |
| 1 | T5 | Insert k benign sentences (no IOC, tool or group) | R | INV |
| 1 | T6 | Change the date format (negative control) | R | INV |
| 2 | E1 | Mask k IOCs with generic phrases | A | DIR |
| 2 | E2 | Drop or neutralise (LLM) the sentences of k techniques | A | DIR |
| 2 | E3 | Replace tool names with another ATT&CK alias | A | INV |
| 2 | E4 | Mask the true group name or all group names | R | DIR |
| 2 | E5 | Remove temporal connectives or swap k sentences | A | DIR |
| 2 | E6 | LLM generalisation of tool names/commands | A | DIR |
| 2 | E7 | Insert k valid but unrelated IOCs | A | INV |
| 2 | E8 | False flag: insert IOCs, TTP sentences or the name of B | A | TGT |
| 3 | G1 | Delete k (or all) IOC, technique or tactic nodes | D | DIR |
| 3 | G2 | Delete or randomly rewire IOC–technique edges | D | DIR |
| 3 | G3 | Reorder evidence nodes (reverse, shuffle, swap) | D | DIR |
| 3 | G4 | Insert k nodes from another group's graph | D | INV |
| 3 | G5 | Shuffle edge relation types | D | DIR |

Parameter grids are in `configs/retrace.yaml`.

## Metrics and protocol

| Metric | Definition |
| --- | --- |
| flip / flip_c | top-1 label of the variant differs from the original; flip_c only on reports the system attributes correctly |
| Δp | p(true group \| original) − p(true group \| variant) |
| violation | the operator's expectation is violated (INV: flip, DIR: confident flip with p > 0.5, TGT: attributed to B) |
| ASR | (E8) share of variants attributed to the decoy group B |
| Δacc | top-1 accuracy of the originals minus that of the variants |

Every rate is averaged per document and reported with a 95% document-level bootstrap CI (1,000
resamples). An effect is **supported** when the CI excludes 0 and the effect is at least 1 pp,
**weak** when only one condition holds, and **not supported** otherwise. The **paired protocol**
compares two systems by their flip_c difference on the reports both attribute correctly.

## Paper-to-code map

| Paper | Code |
| --- | --- |
| Threat model, operators (Table 1) | `retrace/operators/`, `configs/retrace.yaml` |
| Variant generation, validity checks | `retrace/llm.py`, `retrace/pools.py`, `retrace/lexicons.py`, `retrace/validate.py` |
| Black-box search (transfer / adaptive) | `retrace/search.py`, `retrace/runner.py`, `cta_systems/evaluation/adaptive.py` |
| Report prefixes | `retrace/models.py` (`text_cut`), `cta_systems/prepare.py` |
| Metrics, bootstrap, verdicts | `retrace/metrics.py`, `retrace/analysis.py` (`verdict`) |
| Paired protocol | `cta_systems/evaluation/rows.py`, `cta_systems/evaluation/report.py` (section 7) |
| Stage decomposition, graph diagnosis, S1–S6 | `retrace/analysis.py` |
| Evaluation pool (CTI filter, 144 reports) | `retrace/eligible.py` |
| Case-study systems | `cta_systems/ttp_match/`, `cta_systems/ttp_vec/` |

## Changes to the reproduction (patch)

`patches/threatmamba-reproduction.patch` adds the hooks ReTrace needs. Every change is behind a
flag, and the defaults reproduce the original behaviour.

| File | Change |
| --- | --- |
| `src/encoder.py` | `encode_doc_cached` skips the per-document disk cache for encoders with `skip_disk_cache=True` (ReTrace caches per sentence) |
| `src/contrastive.py` | adds `neg_splice_matched()` (same-size contrastive negatives, Eq. 8) and `truncate_tail()`; `neg_splice` is unchanged |
| `src/train.py` | new flags `--cl-neg {splice,matched}`, `--cskg-dir`, `--aug-dir` |
| `src/evaluate.py` | new flag `--cskg-dir` |
| `configs/default.yaml` | `train.cl_neg_mode: splice` |

## Testing

```bash
THREATMAMBA_FAKE_ENCODER=1 THREATMAMBA_SSM=simple python tests/test_retrace.py   # 17/17 PASS
CTA_SYSTEMS_TEST=1 python tests/test_cta_systems.py                              # 14 tests PASS, ~1 min
```

The tests run on CPU with a fake encoder, TTP extractor, embedder and LLM, and write only to a
temporary folder. They need `data/processed/` and `data/attck/` from the reproduction. They do not
replace a real run on the GPU machine.

## Citation

The paper is accepted at NSS-SocialSec 2026; the DOI and page numbers will be added once the
proceedings are published.

```bibtex
@inproceedings{duong2026retrace,
  title     = {Rewriting the Traces: Adversarial Robustness Evaluation of Temporal Graph-Based Cyber Threat Attribution},
  author    = {{Duong Chi Vy} and {Hien Do Hoang} and {Van-Hau Pham} and {Phan The Duy}},
  booktitle = {Proceedings of NSS-SocialSec 2026},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

Please also cite the systems evaluated here:
ThreatMAMBA (Ge et al., *IEEE TIFS* 21:4386–4401, 2026, [10.1109/TIFS.2026.3685967](https://doi.org/10.1109/TIFS.2026.3685967)),
TTP-Match (Guru et al., Cyber-AI 2025, [10.1109/Cyber-AI66431.2025.11233715](https://doi.org/10.1109/Cyber-AI66431.2025.11233715))
and TTP-Vec (Hussain & Petrova, *Future Internet* 18(8):433, 2026, [10.3390/fi18080433](https://doi.org/10.3390/fi18080433)).

## Acknowledgments and license

This research is funded by Vietnam National University Ho Chi Minh City (VNU-HCM) under grant
number NCM2025-26-01. The CTI2Attacker data and the ThreatMAMBA design come from Ge et al.; the
TTP-Match and TTP-Vec ports follow the code released by their authors (see
[docs/CASE_STUDIES.md](docs/CASE_STUDIES.md)). The dataset is not redistributed here; it is
downloaded by the reproduction's `scripts/download_data.sh`.

ReTrace operates only on public CTI reports and never executes attacks; false-flag variants are
generated offline for evaluation only.

Released under the [MIT License](LICENSE).
