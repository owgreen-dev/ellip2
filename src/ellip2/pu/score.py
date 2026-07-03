"""Stage 2 (score) — per-cluster PU scores from a trained checkpoint.

Loads a checkpoint written by :mod:`ellip2.pu.train` and computes per-cluster
scores (sigmoid of the scorer logits) over **all** clusters via the same
fanout-capped neighbor sampling used in training — so scoring the 49M-node graph
never materializes a full-graph forward. Writes an ``(N,)`` ``.npy`` array (the
``--scores`` contract Stage 3 :mod:`ellip2.discovery.discover` consumes).

Optionally reports subgraph-level PU metrics (PR-AUC / F1 / recall) by MIL
max-pooling cluster scores to subgraphs and comparing against suspicious/licit
labels on a held-out split.

Example:
    python scripts/score.py \
        --model artifacts/pu/model.pt \
        --features artifacts/features/cluster_features.parquet \
        --edge-index artifacts/ingest/edge_index.npy \
        --out artifacts/pu/scores.npy \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv --eval-split test \
        --device cuda
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
from ellip2.pu.train import EncoderConfig, SeedBatcher, build_scorer, load_features
from ellip2.pu.trainer import max_pool_to_subgraph


def _subgraph_scores(
    subgraphs_path: Path,
    cluster_scores: np.ndarray,
    *,
    split_csv: Path | None,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Max-pool cluster scores to per-subgraph scores + suspicious labels."""
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
        description="Stage 2: score clusters with a trained nnPU checkpoint (minibatch).",
    )
    p.add_argument("--model", required=True, type=Path, help="checkpoint .pt from train")
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--edge-index", required=True, type=Path, help="(2, E) edge_index.npy")
    p.add_argument("--out", required=True, type=Path,
                   help="(N,) scores .npy — discover.py --scores contract")
    p.add_argument("--subgraphs", type=Path, default=None,
                   help="subgraphs.parquet; enables subgraph-level metrics")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--batch-size", type=int, default=4096, help="seed nodes per minibatch")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    args = p.parse_args(argv)

    feature_columns, X = load_features(args.features)
    n_nodes = X.shape[0]
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
    num_neighbors = extra.get("num_neighbors") or [15] * cfg.num_layers
    model = build_scorer(in_dim, cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    x_all = torch.from_numpy(X)
    ei_all = torch.from_numpy(np.asarray(edge_index)).long()
    all_seeds = torch.arange(n_nodes, dtype=torch.long)

    batcher = SeedBatcher(
        x_all, ei_all, all_seeds, num_neighbors=num_neighbors,
        batch_size=args.batch_size, shuffle=False, device=device,
    )
    scores = np.zeros(n_nodes, dtype=np.float64)
    done = 0
    with torch.no_grad():
        for x_b, ei_b, n_id, bs in batcher:
            logits = model(x_b, ei_b)[:bs]
            scores[n_id[:bs].numpy()] = torch.sigmoid(logits).cpu().numpy()
            done += bs
            if done % (args.batch_size * 200) < bs:
                print(f"[score] {done:,}/{n_nodes:,} clusters scored", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, scores)
    print(f"[score] wrote {scores.shape[0]:,} cluster scores -> {args.out}")

    if args.subgraphs is not None:
        sg_scores, sg_labels = _subgraph_scores(
            args.subgraphs, scores, split_csv=args.split_csv, split_name=args.eval_split,
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
