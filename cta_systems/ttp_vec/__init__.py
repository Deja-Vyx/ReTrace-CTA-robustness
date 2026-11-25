"""TTP-Vec: Hussain & Petrova (2026), "Threat Actor Attribution Applying a Tactics-Techniques-Procedures
Approach: An Empirical Investigation", Future Internet 18(8):433.

Ported from the authors' code (Supplementary Materials, futureinternet-18-00433-s001):
  - one-hot ATT&CK technique/sub-technique features ("ttps_*")
  - StandardScaler -> multiply by rarity weights (1/number of groups using the feature)^3, max-normalised to 1
  - DecisionTree / KNN (k=1) / GaussianNB / RandomForest / SVC(kernel='linear'), random_state=7

Changes needed to run correctly on CTI2Attacker (documented in README.md):
  - CTI2Attacker has no victim fields (country/sector) -> only the paper's "TTP-only" configuration
  - TTPs are not extracted with ChatGPT + manual review but with ThreatMAMBA's OWN TTP extractor
    (same features, only the classifier differs -> "graph + Mamba" vs "bag of TTPs")
  - scaler and rarity weights are computed on the TRAINING set only (the original code uses all labelled data)
  - uses ThreatMAMBA's train/test split (2,515 / 629) instead of a random 70/30 split
"""
