"""TTP-Match: Guru, Moss & Kochenderfer (2025), "On Technique Identification and Threat-Actor
Attribution using LLMs and Embedding Models", arXiv 2505.11547 (code: github.com/kylag/ttp_attribution).

Algorithm (the authors' everything.py / everythingHYDE.py):
  1. split the report into 3-line windows, embed each window and find the nearest ATT&CK techniques by
     cosine against the embedded technique descriptions (top-k per window, optional threshold)
  2. count techniques over the whole report -> normalised frequency y
  3. group profile P(t | A) = summed counts over the TRAINING reports of group A, normalised
  4. score(A) = P(A) * sum_t P(t | A) * y_t ; rank the groups

Changes needed to run correctly on CTI2Attacker (see README.md):
  - OpenAI text-embedding-3-large -> an open embedding model run locally (default BAAI/bge-large-en-v1.5)
  - 3-SENTENCE windows (CTI2Attacker texts have no line breaks, so "3 lines" = the whole report)
  - techniques keyed by ATT&CK v14 ID (the original code keys by NAME -> 23 duplicate names were merged)
  - technique list from the local v14 STIX, parent techniques dropped when a sub-technique exists (rid_technique.py)
  - bug fixes: the "test" loop ran on validation, argmax picked the worst split, ties broken by folder order
  - top-k/threshold/prior chosen on a validation split of train; a temperature T makes probabilities meaningful
  - HyDE (GPT-4-generated pseudo passages) NOT used (needs an LLM; paper: mean rank 7.55 vs 7.75)
"""
