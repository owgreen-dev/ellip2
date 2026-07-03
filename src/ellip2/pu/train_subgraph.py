"""Stage 2 — supervised subgraph classifier (gradient-boosted trees). PRIMARY detector.

plan.md decision #1: at the subgraph level detection is imbalanced *supervised*
classification (reliable licit negatives), which is what every Elliptic2 SOTA does. A
diagnostic on the real graph showed this beats the cluster-level nnPU GNN ~10× in test
PR-AUC (0.30 vs 0.03) — see ``docs/stage2-model-choice.md``. This trains a
``HistGradientBoostingClassifier`` on per-subgraph pooled ``cluster_features``
(:mod:`ellip2.pu.subgraph_pool`) against the suspicious/licit label on the train split,
class-weighting the rare positives, and reports PR-AUC/F1 on the eval split.

Example:
    python scripts/train_subgraph.py \
        --features artifacts/features/cluster_features.parquet \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/subgraph_model.pkl
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ellip2.eval.pu_metrics import pu_metric_report
from ellip2.pu.subgraph_pool import POOLS, pool_subgraph_features


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: supervised subgraph classifier (HistGradientBoosting).",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--subgraphs", required=True, type=Path, help="subgraphs.parquet")
    p.add_argument("--split-csv", required=True, type=Path, help="split.csv (id,label,split)")
    p.add_argument("--out", required=True, type=Path, help="output model .pkl")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--l2", type=float, default=0.0, help="L2 regularization")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    pooled = pool_subgraph_features(args.features, args.subgraphs, split_csv=args.split_csv)
    tr = pooled.split == args.train_split
    ev = pooled.split == args.eval_split
    n_pos = int(pooled.y[tr].sum())
    n_neg = int((pooled.y[tr] == 0).sum())
    if n_pos == 0:
        raise SystemExit(f"no suspicious subgraphs in '{args.train_split}' split")

    # class-weight the rare positives (neg/pos), like the diagnostic.
    weights = np.where(pooled.y[tr] == 1, n_neg / max(1, n_pos), 1.0)
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=args.seed,
    )
    print(
        f"[train_subgraph] subgraphs={len(pooled.ccids):,} feat_dim={pooled.X.shape[1]} "
        f"train_pos={n_pos} train_neg={n_neg} eval={args.eval_split}({int(ev.sum())})",
        flush=True,
    )
    clf.fit(pooled.X[tr], pooled.y[tr], sample_weight=weights)

    if ev.any():
        scores = clf.predict_proba(pooled.X[ev])[:, 1]
        report = pu_metric_report(scores, pooled.y[ev])
        base = float(pooled.y[ev].mean())
        print(
            f"[train_subgraph] eval ({args.eval_split}, n={int(ev.sum()):,}, "
            f"base_rate={base:.4f}): "
            + ", ".join(f"{k}={v:.4f}" for k, v in report.items())
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(
            {
                "model": clf,
                "framing": "supervised_subgraph_hgbm",
                "pools": list(POOLS),
                "feature_names": pooled.feature_names,
            },
            fh,
        )
    print(f"[train_subgraph] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
