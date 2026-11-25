"""ReTrace: adversarial robustness evaluation for cyber threat attribution (CTA) systems.

This package does NOT modify the inference path of the attribution model. It only reuses
existing components (GraphBuilder, TTPService, load_model/predict, truncate_graph,
node_contributions) to:

  1. build the eligible document set (eligible.py)
  2. generate validity-checked variants at three levels: text (T), evidence (E), graph (G)
     (operators/)
  3. check validity (validate.py), rebuild the CSKG (pipeline.py), score every model at five
     timeline cut-offs (models.py) and store the results (runner.py)
  4. aggregate the metrics and run the root-cause / shortcut analyses (metrics.py, analysis.py)

See README.md at the repository root for the order in which to run the steps.
"""
