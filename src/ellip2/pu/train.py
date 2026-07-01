"""Stage 2 (train) — cluster-level nnPU scorer over Elliptic2 clusters.

Trains the heterophily-tolerant encoder (:class:`~ellip2.pu.encoder.HeterophilyEncoder`)
plus a linear PU head (:class:`~ellip2.pu.trainer.ClusterScorer`) with the
non-negative PU risk (Kiryo 2017) — the **cluster-level** framing of plan.md
Resolved decision #2. Positives are the clusters that are members of a
*suspicious* subgraph; every other cluster is unlabeled. The checkpoint written
here is consumed by :mod:`ellip2.pu.score`, whose per-cluster scores feed Stage 3
discovery (:mod:`ellip2.discovery.discover`).

Why cluster-nnPU and not the supervised subgraph model? Stage 3 (`discover.py`)
consumes ONE score per cluster (an ``(N,)`` array), and the ingest artifacts
carry no per-edge feature array — both of which the border-Deep-Sets supervised
model in :mod:`ellip2.pu.trainer` would require. The cluster scorer maps cleanly
onto the existing ``cluster_features`` / ``edge_index`` / ``subgraphs`` artifacts.

Example:
    python scripts/train.py \
        --features artifacts/features/cluster_features.parquet \
        --edge-index artifacts/ingest/edge_index.npy \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/model.pt
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from ellip2.data import schema
from ellip2.pu.encoder import HeterophilyEncoder
from ellip2.pu.prior_estimation import tice_prior
from ellip2.pu.trainer import ClusterScorer, save_checkpoint, train_cluster_nnpu


@dataclass(frozen=True)
class EncoderConfig:
    """Encoder hyper-parameters, persisted in the checkpoint so :mod:`ellip2.pu.score`
    can rebuild an identical model before loading weights."""

    hidden: int = 64
    emb_dim: int = 32
    num_layers: int = 2
    aggr: str = "mean"
    dropout: float = 0.0
    normalize: bool = False


def load_features(path: Path) -> tuple[list[str], np.ndarray]:
    """Load ``cluster_features.parquet`` → ``(feature_columns, X)``.

    Rows are sorted by the integer ``idx`` key so row ``i`` is cluster ``i`` in
    ``[0, N)`` — the same indexing as ``node_features.npy`` / ``edge_index.npy``.

    Returns:
        ``feature_columns``: names of the ``F`` feature columns (order matters —
        stored in the checkpoint and re-checked at score time).
        ``X``: ``(N, F)`` float32 feature matrix.
    """
    table = pq.read_table(path)
    if "idx" not in table.column_names:
        raise ValueError(f"{path} has no 'idx' column; got {table.column_names}")
    idx = table.column("idx").to_numpy(zero_copy_only=False)
    order = np.argsort(idx, kind="stable")
    if not np.array_equal(idx[order], np.arange(len(idx))):
        raise ValueError(f"feature 'idx' in {path} is not a contiguous 0..N-1 range")
    feature_columns = [c for c in table.column_names if c != "idx"]
    cols = [
        table.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
        for c in feature_columns
    ]
    X = np.column_stack(cols)[order] if cols else np.empty((len(idx), 0), np.float32)
    return feature_columns, X


def _allowed_ccids(split_csv: Path, split_name: str) -> set[str]:
    """ccIds assigned to ``split_name`` in ``split.csv`` (columns ``id,label,split``)."""
    with open(split_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "split" not in fields or "id" not in fields:
            raise ValueError(f"{split_csv} must have id,label,split columns")
        return {row["id"] for row in reader if row["split"] == split_name}


def positive_mask(
    subgraphs_path: Path,
    n_nodes: int,
    *,
    split_csv: Path | None = None,
    split_name: str = "train",
) -> np.ndarray:
    """Boolean ``(N,)`` mask of clusters that are members of a suspicious subgraph.

    When ``split_csv`` is given, only suspicious subgraphs assigned to
    ``split_name`` contribute positives — this keeps test-split labels out of
    training. Members are unioned across subgraphs (a cluster may belong to more
    than one).
    """
    table = pq.read_table(subgraphs_path, columns=["ccId", "ccLabel", "member_idx"])
    cc_ids = [str(v) for v in table.column("ccId").to_pylist()]
    cc_labels = table.column("ccLabel").to_pylist()
    members_col = table.column("member_idx").to_pylist()
    allowed = _allowed_ccids(split_csv, split_name) if split_csv is not None else None
    mask = np.zeros(n_nodes, dtype=bool)
    for cc_id, label, members in zip(cc_ids, cc_labels, members_col, strict=True):
        if label != schema.LABEL_SUSPICIOUS:
            continue
        if allowed is not None and cc_id not in allowed:
            continue
        idx = np.asarray(members, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_nodes)]
        mask[idx] = True
    return mask


def build_scorer(in_dim: int, cfg: EncoderConfig) -> ClusterScorer:
    """Assemble a :class:`ClusterScorer` (heterophily encoder + linear head)."""
    encoder = HeterophilyEncoder(
        in_dim,
        cfg.hidden,
        cfg.emb_dim,
        num_layers=cfg.num_layers,
        aggr=cfg.aggr,
        dropout=cfg.dropout,
        normalize=cfg.normalize,
    )
    return ClusterScorer(encoder, emb_dim=cfg.emb_dim)


def estimate_prior(
    X: np.ndarray, mask: np.ndarray, *, sample: int, seed: int
) -> float:
    """TIcE class-prior estimate, subsampling the unlabeled set to bound RAM.

    All positives are kept; the unlabeled set is randomly capped at ``sample``
    rows (49M clusters × F in float64 would not fit in memory). Prior estimation
    is a starting point regardless — a subsample is statistically fine.
    """
    pos_idx = np.flatnonzero(mask)
    unl_idx = np.flatnonzero(~mask)
    rng = np.random.default_rng(seed)
    if unl_idx.size > sample:
        unl_idx = rng.choice(unl_idx, size=sample, replace=False)
    sel = np.concatenate([pos_idx, unl_idx])
    est = tice_prior(X[sel].astype(np.float64), mask[sel])
    print(
        f"[train] estimated prior pi_p={est.prior:.4g} "
        f"(c_hat={est.label_frequency:.4g}, gamma={est.labeled_fraction:.4g}, "
        f"n_labeled={est.n_labeled:,}, sampled_total={est.n_total:,})"
    )
    return est.prior


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: train the cluster-level nnPU scorer.",
    )
    p.add_argument("--features", required=True, type=Path,
                   help="cluster_features.parquet (Stage 1)")
    p.add_argument("--edge-index", required=True, type=Path,
                   help="(2, E) edge_index.npy (Stage 0)")
    p.add_argument("--subgraphs", required=True, type=Path,
                   help="subgraphs.parquet (Stage 0) — supplies positive clusters")
    p.add_argument("--split-csv", type=Path, default=None,
                   help="split.csv; restrict positives to --split-name (no leakage)")
    p.add_argument("--split-name", default="train",
                   help="split to draw training positives from (default: train)")
    p.add_argument("--out", required=True, type=Path, help="output checkpoint .pt")
    p.add_argument("--prior", type=float, default=None,
                   help="class prior pi_p; estimated via TIcE when omitted")
    p.add_argument("--prior-sample", type=int, default=200_000,
                   help="max unlabeled rows used for TIcE prior estimation")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--aggr", default="mean", choices=["mean", "max", "sum", "add"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--normalize", action="store_true",
                   help="L2-normalise encoder output embeddings")
    p.add_argument("--beta", type=float, default=0.0, help="nnPU clamp bound (Kiryo)")
    p.add_argument("--gamma", type=float, default=1.0, help="nnPU ascent scale (Kiryo)")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)

    feature_columns, X = load_features(args.features)
    n_nodes = X.shape[0]
    edge_index = np.load(args.edge_index)
    mask = positive_mask(
        args.subgraphs, n_nodes,
        split_csv=args.split_csv, split_name=args.split_name,
    )
    n_pos = int(mask.sum())
    if n_pos == 0:
        raise SystemExit(
            "no positive clusters found — check --subgraphs / --split-csv / --split-name"
        )

    prior = (
        args.prior
        if args.prior is not None
        else estimate_prior(X, mask, sample=args.prior_sample, seed=args.seed)
    )

    device = torch.device(args.device)
    x = torch.from_numpy(X).to(device)
    ei = torch.from_numpy(np.asarray(edge_index)).long().to(device)
    pos = torch.from_numpy(mask).to(device)

    cfg = EncoderConfig(
        hidden=args.hidden, emb_dim=args.emb_dim, num_layers=args.num_layers,
        aggr=args.aggr, dropout=args.dropout, normalize=args.normalize,
    )
    model = build_scorer(X.shape[1], cfg).to(device)
    print(
        f"[train] N={n_nodes:,} F={X.shape[1]} E={ei.shape[1]:,} "
        f"positives={n_pos:,} prior={prior:.4g} epochs={args.epochs} device={args.device}"
    )

    history, optimizer = train_cluster_nnpu(
        model, x, ei, pos, prior,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        beta=args.beta, gamma=args.gamma,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.out, model, optimizer,
        extra={
            "framing": "cluster_nnpu",
            "in_dim": X.shape[1],
            "feature_columns": feature_columns,
            "encoder": asdict(cfg),
            "prior": float(prior),
            "n_nodes": n_nodes,
            "epochs": args.epochs,
            "loss_first": history.first,
            "loss_last": history.last,
        },
    )
    print(f"[train] risk {history.first:.4f} -> {history.last:.4f}; wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
