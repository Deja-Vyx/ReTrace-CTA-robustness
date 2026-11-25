"""Registry of perturbation operators. `get_op("E1", ctx)` returns an instance."""
from .base import REGISTRY, Ctx, Op, Variant  # noqa: F401
from . import text_ops, evidence_ops, graph_ops  # noqa: F401  (registers the operators)

ALL_OPS = ["ID", "T1", "T2", "T3", "T4", "T5", "T6",
           "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8",
           "G1", "G2", "G3", "G4", "G5"]
P0_OPS = ["ID", "T3", "T6", "E1", "E4", "E7", "E8", "G1", "G3", "G5"]
LLM_OPS = {"T1", "T2", "E6"}          # E2 needs the LLM only when mode=neutral


def get_op(code: str, ctx: Ctx) -> Op:
    if code not in REGISTRY:
        raise SystemExit(f"Unknown operator '{code}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[code](ctx)
