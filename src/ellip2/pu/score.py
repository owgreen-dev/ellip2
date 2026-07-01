"""Stage 2 (score) — per-cluster PU scores from a trained checkpoint.

Loads a checkpoint written by :mod:`ellip2.pu.train`, recomputes per-cluster
scores (sigmoid of the scorer logits) over the full graph, and writes an ``(N,)``
``.npy`` array — the exact ``--scores`` contract Stage 3
:mod:`ellip2.discovery.discover` consumes (one score per cluster, index-aligned
to ``node_features`` / ``edge_index``).

Optionally reports subgraph-level PU metrics (PR-AUC / F1 / recall) by MIL
max-pooling cluster scores to subgraphs (:func:`~ellip2.pu.trainer.max_pool_to_subgraph`)
and comparing against the suspicious/licit labels — restricted to an eval split
when ``--split-csv`` is given, to validate against RevClassify's held-out numbers.

Example:
    python scripts/score.py \
        --model artifacts/pu/model.pt \
        --features artifacts/features/cluster_features.parquet \
        --edge-index artifacts/ingest/edge_index.npy \
        --out artifacts/pu/scores.npy \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --eval-split test
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from ellip2.data import schema
from ellip2.eval.pu_metrics import pu_metric_report
from ellip2.pu.train import EncoderConfig, build_scorer, load_features
from ellip2.pu.trainer import max_pool_to_subgraph


def _subgraph_scores(
    subgraphs_path: Path,
    cluster_scores: np.ndarray,
    *,
    split_csv: Path | None,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool cluster scores to per-subgraph scores + suspicious labels.

    Returns ``(scores, labels)`` over the selected subgraphs (all subgraphs, or
    just those in ``split_name`` when ``split_csv`` is given). ``labels`` is 1 for
    suspicious, 0 for licit.
    """
    table = pq.read_table(subgraphs_path, columns=["ccId", "ccLabel", "member_idx"])
    cc_ids = [str(v) for v in table.column("ccId").to_pylist()]
    cc_labels = table.column("ccLabel").to_pylist()
    members_col = table.column("member_idx").to_pylist()

    allowed: set[str] | None = None
    if split_csv is not None:
        with open(split_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            allowed = {row["id"] for row in reader if row["split"] == split_name}

    member_scores: list[float] = []
    member_subgraph: list[int] = []
    labels: list[int] = []
    j = 0
    for cc_id, label, members in zip(cc_ids, cc_labels, members_col, strict=True):
        if allowed is not None and cc_id not in allowed:
            continue
        for m in members:
            member_scores.append(float(cluster_scores[int(m)]))
            member_subgraph.append(j)
        labels.append(1 if label == schema.LABEL_SUSPICIOUS else 0)
        j += 1
    scores = max_pool_to_subgraph(
        torch.tensor(member_scores, dtype=torch.float64),
        torch.tensor(member_subgraph, dtype=torch.long),
        num_subgraphs=j,
    ).numpy()
    return scores, np.asarray(labels, dtype=int)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: score clusters with a trained nnPU checkpoint.",
    )
    p.add_argument("--model", required=True, type=Path, help="checkpoint .pt from train")
    p.add_argument("--features", required=True, type=Path,
                   help="cluster_features.parquet (same columns as training)")
    p.add_argument("--edge-index", required=True, type=Path, help="(2, E) edge_index.npy")
    p.add_argument("--out", required=True, type=Path,
                   help="(N,) scores .npy — discover.py --scores contract")
    p.add_argument("--subgraphs", type=Path, default=None,
                   help="subgraphs.parquet; enables subgraph-level metrics")
    p.add_argument("--split-csv", type=Path, default=None,
                   help="split.csv; report metrics only on --eval-split")
    p.add_argument("--eval-split", default="test", help="split to evaluate (default: test)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args(argv)

    feature_columns, X = load_features(args.features)
    edge_index = np.load(args.edge_index)

    device = torch.device(args.device)
    ckpt = torch.load(str(args.model), map_location=device, weights_only=False)
    extra = ckpt.get("extra", {})
    cfg = EncoderConfig(**extra["encoder"]) if "encoder" in extra else EncoderConfig()
    expected_cols = extra.get("feature_columns")
    if expected_cols is not None and list(expected_cols) != feature_columns:
        raise SystemExit(
            "feature columns differ from training; refusing to score "
            f"(train={len(expected_cols)} cols, now={len(feature_columns)})"
        )
    in_dim = int(extra.get("in_dim", X.shape[1]))
    model = build_scorer(in_dim, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x = torch.from_numpy(X).to(device)
    ei = torch.from_numpy(np.asarray(edge_index)).long().to(device)
    with torch.no_grad():
        logits = model(x, ei)
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, scores)
    print(f"[score] wrote {scores.shape[0]:,} cluster scores -> {args.out}")

    if args.subgraphs is not None:
        sg_scores, sg_labels = _subgraph_scores(
            args.subgraphs, scores,
            split_csv=args.split_csv, split_name=args.eval_split,
        )
        report = pu_metric_report(sg_scores, sg_labels)
        tag = args.eval_split if args.split_csv else "all"
        print(
            f"[score] subgraph metrics ({tag}, n={len(sg_labels):,}, "
            f"pos={int(sg_labels.sum()):,}): "
            + ", ".join(f"{k}={v:.4f}" for k, v in report.items())
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
