"""ReTrace evaluation (levels 1 and 2) of the case-study systems.

    zoo.py       load models by tag, score texts at 5 timeline cut-offs
    rows.py      metric rows: EXACTLY the definitions of retrace/metrics.build_table
    rescore.py   rescore the stored variants of the ReTrace runs (rand/worst/llm) -> transfer attacks
    adaptive.py  search the strongest variants against the case-study system ITSELF -> adaptive attacks
    report.py    comparison report outputs/cta_systems/REPORT_CTA_SYSTEMS.md
"""
