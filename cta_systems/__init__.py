"""Case-study CTA systems evaluated with ReTrace on the same CTI2Attacker data and the same
ReTrace benchmark (levels 1 and 2) as ThreatMAMBA.

    ttp_vec/       TTP-Vec: Hussain & Petrova 2026 (Future Internet 18(8):433): one-hot TTPs + DT/KNN/NB/RF/SVM
    ttp_match/     TTP-Match: Guru, Moss & Kochenderfer 2025 (arXiv 2505.11547): embedding-based technique
                   matching + P(technique | group) profiles + prior
    evaluation/    rescoring of ReTrace variants (transfer), adaptive attacks, comparison report

This package only READS data/results of the base repository (src/, outputs/retrace/) and writes to
outputs/cta_systems/. It does not modify any ThreatMAMBA or ReTrace file.
"""
