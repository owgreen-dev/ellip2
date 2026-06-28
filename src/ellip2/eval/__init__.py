"""Evaluation utilities: leakage-safe splits, PU metrics, invariant checks."""

from ellip2.eval.splits import (
    SplitConfig,
    SplitResult,
    generate_split,
    load_components,
    verify_split,
)

__all__ = [
    "SplitConfig",
    "SplitResult",
    "generate_split",
    "load_components",
    "verify_split",
]
