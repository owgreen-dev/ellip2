"""Stage 2 — score all labeled subgraphs with a trained supervised classifier.

Loads a :mod:`ellip2.pu.train_subgraph` model and writes ``subgraph_scores.parquet``
(``ccId``, ``score``, ``label``, ``split``) — the per-subgraph detection scores that are
the actual deliverable (validate against RevClassify's held-out numbers). Optionally
reports PR-AUC/F1 on an eval split.

Example:
    python scripts/score_subgraph.py \
        --model artifacts/pu/subgraph_model.pkl \
        --features artifacts/features/cluster_features.parquet \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/subgraph_scores.parquet --eval-split test
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ellip2.eval.pu_metrics import pu_metric_report
from ellip2.pu.subgraph_pool import POOLS, pool_subgraph_features


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: score subgraphs with a trained supervised classifier.",
    )
    p.add_argument("--model", required=True, type=Path, help="model .pkl from train_subgraph")
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--subgraphs", required=True, type=Path, help="subgraphs.parquet")
    p.add_argument("--out", required=True, type=Path, help="output subgraph_scores.parquet")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--eval-split", default="test")
    args = p.parse_args(argv)

    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    clf = bundle["model"]
    pools = tuple(bundle.get("pools", POOLS))

    pooled = pool_subgraph_features(
        args.features, args.subgraphs, split_csv=args.split_csv, pools=pools
    )
    scores = clf.predict_proba(pooled.X)[:, 1].astype(np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "ccId": pa.array(pooled.ccids),
                "score": pa.array(scores),
                "label": pa.array(pooled.y),
                "split": pa.array([str(s) for s in pooled.split]),
            }
        ),
        args.out,
    )
    print(f"[score_subgraph] wrote {len(pooled.ccids):,} subgraph scores -> {args.out}")

    if args.split_csv is not None:
        ev = pooled.split == args.eval_split
        if ev.any():
            report = pu_metric_report(scores[ev], pooled.y[ev])
            base = float(pooled.y[ev].mean())
            print(
                f"[score_subgraph] eval ({args.eval_split}, n={int(ev.sum()):,}, "
                f"base_rate={base:.4f}): "
                + ", ".join(f"{k}={v:.4f}" for k, v in report.items())
            )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
