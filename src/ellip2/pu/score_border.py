"""Stage 2 (border model) — score all subgraphs with a trained border classifier.

Rebuilds the :class:`~ellip2.pu.trainer.SupervisedSubgraphModel` from a
:mod:`ellip2.pu.train_border` checkpoint, re-extracts border sets (deterministic, same cap),
and writes ``subgraph_scores.parquet`` (``ccId``, ``score``, ``label``, ``split``) — the
per-subgraph detection scores — plus optional PR-AUC/F1 on an eval split. Scoring runs in
subgraph chunks so the per-batch border tensors stay bounded.

Example:
    python scripts/score_border.py \
        --model artifacts/pu/border_model.pt \
        --artifacts-dir artifacts/ingest \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/border_scores.parquet --eval-split test
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ellip2.eval.pu_metrics import pu_metric_report
from ellip2.pu.border_assembly import (
    build_subgraph_batch,
    extract_border_sets,
    load_internal_edge_features,
)
from ellip2.pu.train_border import load_subgraphs
from ellip2.pu.trainer import SupervisedSubgraphModel


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: score subgraphs with a trained border model.",
    )
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--artifacts-dir", required=True, type=Path)
    p.add_argument("--subgraphs", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output border_scores.parquet")
    p.add_argument("--split-csv", type=Path, default=None)
    p.add_argument("--eval-split", default="test")
    p.add_argument("--internal-edges", type=Path, default=None,
                   help="internal_edge_features.parquet (required if the model used edges)")
    p.add_argument("--chunk-size", type=int, default=20_000, help="subgraphs per forward")
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    ccids, members, labels, split = load_subgraphs(args.subgraphs, args.split_csv)
    edge_index = np.load(args.artifacts_dir / "edge_index.npy")
    node_features = np.load(args.artifacts_dir / "node_features.npy", mmap_mode="r")
    n_nodes = node_features.shape[0]

    device = torch.device(args.device)
    ckpt = torch.load(str(args.model), map_location=device, weights_only=False)
    extra = ckpt["extra"]
    model = SupervisedSubgraphModel(
        int(extra["node_dim"]), int(extra["edge_dim"]),
        set_hidden=int(extra["set_hidden"]), set_out=int(extra["set_out"]),
        mlp_hidden=tuple(extra["mlp_hidden"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(extra["feat_mean"], dtype=np.float32)
    std = np.asarray(extra["feat_std"], dtype=np.float32)
    edge_dim = int(extra["edge_dim"])

    edge_feats = None
    edge_mean = edge_std = None
    if extra.get("use_edges"):
        if args.internal_edges is None:
            raise SystemExit("model was trained with edges; pass --internal-edges")
        edge_feats = load_internal_edge_features(args.internal_edges)
        edge_mean = np.asarray(extra["edge_mean"], dtype=np.float32)
        edge_std = np.asarray(extra["edge_std"], dtype=np.float32)

    print(f"[score_border] extracting border (cap={extra['border_cap']}) "
          f"over E={edge_index.shape[1]:,}...", flush=True)
    border = extract_border_sets(edge_index, members, n_nodes, cap=int(extra["border_cap"]))
    del edge_index

    scores = np.zeros(len(ccids), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(ccids), args.chunk_size):
            pos = list(range(start, min(start + args.chunk_size, len(ccids))))
            batch = build_subgraph_batch(
                pos, border, node_features, mean=mean, std=std, edge_dim=edge_dim,
                edge_features_by_sg=edge_feats, edge_mean=edge_mean, edge_std=edge_std,
            )
            scores[pos] = torch.sigmoid(model(batch)).cpu().numpy()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ccId": pa.array(ccids),
            "score": pa.array(scores),
            "label": pa.array(labels),
            "split": pa.array([str(s) for s in split]),
        }),
        args.out,
    )
    print(f"[score_border] wrote {len(ccids):,} subgraph scores -> {args.out}")

    if args.split_csv is not None:
        ev = split == args.eval_split
        if ev.any():
            report = pu_metric_report(scores[ev], labels[ev])
            base = float(labels[ev].mean())
            print(f"[score_border] eval ({args.eval_split}, n={int(ev.sum()):,}, "
                  f"base_rate={base:.4f}): "
                  + ", ".join(f"{k}={v:.4f}" for k, v in report.items()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
