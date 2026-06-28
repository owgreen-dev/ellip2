"""Stage 2 — Positive-Unlabeled (PU) learning components.

Holds the cluster-level PU machinery: the non-negative PU risk estimator
(:mod:`ellip2.pu.nnpu_loss`, Kiryo et al. 2017) and, in later tasks, the
class-prior estimator, GNN encoder, and trainer.

Per plan.md Resolved decision #2: genuine nnPU lives at the **cluster** level
(the ~49M unlabeled clusters are a true unlabeled set, π_p small ~1e-3 or less),
NOT at the subgraph level (whose 2.27% base rate calls for supervised weighted
BCE instead).
"""

from __future__ import annotations
