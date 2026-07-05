"""Stage 2 (border model) — train the supervised subgraph-border classifier.

plan.md decision #1, "Path A" (the decisive RevClassify insight): represent each subgraph
by border-node Deep Sets — ``DeepSets(senders) ⊕ DeepSets(receivers)`` — plus pooled
internal node features, fed to an MLP trained with weighted BCE
(:class:`ellip2.pu.trainer.SupervisedSubgraphModel` / :func:`train_supervised`). Border sets
are assembled from ``edge_index.npy`` + ``subgraphs.parquet`` (:mod:`ellip2.pu.border_assembly`).

Phase 1 uses border + internal-node (43-d) features only; internal edge (95-d) features are
not in any artifact (would need re-reading background_edges.csv) and are left empty.

Example:
    python scripts/train_border.py \
        --artifacts-dir artifacts/ingest \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/border_model.pt
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
from ellip2.eval.pu_metrics import pr_auc, pu_metric_report
from ellip2.pu.border_assembly import (
    build_subgraph_batch,
    extract_border_sets,
    fit_edge_standardizer,
    fit_node_standardizer,
    load_internal_edge_features,
)
from ellip2.pu.trainer import (
    SubgraphBatch,
    SupervisedSubgraphModel,
    save_checkpoint,
    train_supervised,
)


def load_subgraphs(
    subgraphs_path: Path, split_csv: Path | None
) -> tuple[list[str], list[np.ndarray], np.ndarray, np.ndarray]:
    """Return ``(ccids, members, labels, split)`` from subgraphs.parquet + split.csv."""
    t = pq.read_table(subgraphs_path, columns=["ccId", "ccLabel", "member_idx"])
    ccids = [str(v) for v in t.column("ccId").to_pylist()]
    labels = np.array(
        [1 if v == schema.LABEL_SUSPICIOUS else 0 for v in t.column("ccLabel").to_pylist()],
        dtype=np.int64,
    )
    members = [np.asarray(m, dtype=np.int64) for m in t.column("member_idx").to_pylist()]
    members = [m[m >= 0] for m in members]
    split_of: dict[str, str] = {}
    if split_csv is not None:
        with open(split_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                split_of[str(r["id"])] = r["split"]
    split = np.array([split_of.get(cc, "") for cc in ccids], dtype=str)
    return ccids, members, labels, split


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: train the supervised subgraph-border model.",
    )
    p.add_argument("--artifacts-dir", required=True, type=Path,
                   help="Stage 0 dir with edge_index.npy, node_features.npy")
    p.add_argument("--subgraphs", required=True, type=Path)
    p.add_argument("--split-csv", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output checkpoint .pt")
    p.add_argument("--internal-edges", type=Path, default=None,
                   help="internal_edge_features.parquet (Phase 2 — enables the edge channel)")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--border-cap", type=int, default=64, help="max border nodes/subgraph/side")
    p.add_argument("--set-hidden", type=int, default=64)
    p.add_argument("--set-out", type=int, default=32)
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--restarts", type=int, default=1,
                   help="train N restarts (varying init seed) and keep the best-VAL model; "
                        "rejects the occasional degenerate/collapsed run")
    p.add_argument("--val-split", default="val",
                   help="split used to select the best restart (needs to be non-empty when "
                        "--restarts > 1)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    ccids, members, labels, split = load_subgraphs(args.subgraphs, args.split_csv)
    edge_index = np.load(args.artifacts_dir / "edge_index.npy")
    node_features = np.load(args.artifacts_dir / "node_features.npy", mmap_mode="r")
    n_nodes, node_dim = node_features.shape

    print(f"[train_border] subgraphs={len(ccids):,} extracting border (cap={args.border_cap}) "
          f"over E={edge_index.shape[1]:,}...", flush=True)
    border = extract_border_sets(edge_index, members, n_nodes, cap=args.border_cap, seed=args.seed)
    del edge_index

    tr = np.flatnonzero(split == args.train_split)
    ev = np.flatnonzero(split == args.eval_split)
    n_pos = int(labels[tr].sum())
    if n_pos == 0:
        raise SystemExit(f"no suspicious subgraphs in '{args.train_split}' split")
    pos_weight = float((labels[tr] == 0).sum()) / n_pos

    mean, std = fit_node_standardizer(tr, border, node_features)
    device = torch.device(args.device)
    # edge_dim defaults to the schema constant but is set to the ACTUAL width of the
    # extracted edge features when the channel is on (the real background_edges.csv has
    # 96 feature columns, not the paper's 95 — ingest never validated the count).
    edim = schema.N_EDGE_FEATURES

    edge_feats = None
    edge_mean = edge_std = None
    if args.internal_edges is not None:
        edge_feats = load_internal_edge_features(args.internal_edges)
        for v in edge_feats.values():
            if v.shape[0]:
                edim = int(v.shape[1])
                break
        edge_mean, edge_std = fit_edge_standardizer(tr, edge_feats)
        n_edge_rows = sum(int(v.shape[0]) for v in edge_feats.values())
        print(f"[train_border] internal edges: {n_edge_rows:,} rows over "
              f"{len(edge_feats):,} subgraphs, edge_dim={edim} (edge channel ON)", flush=True)

    def _batch(idx: np.ndarray) -> SubgraphBatch:
        return build_subgraph_batch(
            idx, border, node_features, mean=mean, std=std, edge_dim=edim,
            edge_features_by_sg=edge_feats, edge_mean=edge_mean, edge_std=edge_std,
        )

    train_batch = _batch(tr)
    y = torch.from_numpy(labels[tr].astype(np.float32))
    val = np.flatnonzero(split == args.val_split)
    if args.restarts > 1 and val.size == 0:
        raise SystemExit(
            f"--restarts {args.restarts} needs a non-empty '{args.val_split}' split to select on"
        )
    val_batch = _batch(val) if val.size else None
    print(f"[train_border] train={len(tr):,} (pos={n_pos}, pos_weight={pos_weight:.1f}) "
          f"val={args.val_split}({val.size:,}) eval={args.eval_split}({len(ev):,}) "
          f"restarts={args.restarts} senders={train_batch.sender_x.shape[0]:,} "
          f"receivers={train_batch.receiver_x.shape[0]:,} "
          f"internal={train_batch.node_x.shape[0]:,}", flush=True)

    # Best-of-N restarts with validation-based selection: a collapsed run (degenerate
    # all-positive) has a low val PR-AUC, so it is never selected. Isolates init variance —
    # border sets and the standardizer are shared; only the model init seed varies.
    model: SupervisedSubgraphModel | None = None
    optimizer: torch.optim.Optimizer | None = None
    best_sel = -np.inf
    best_r = -1
    for r in range(args.restarts):
        torch.manual_seed(args.seed + r)
        cand = SupervisedSubgraphModel(
            node_dim, edim, set_hidden=args.set_hidden, set_out=args.set_out,
            mlp_hidden=tuple(args.mlp_hidden),
        ).to(device)
        history, opt = train_supervised(
            cand, train_batch, y, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, pos_weight=pos_weight,
        )
        if val_batch is not None:
            cand.eval()
            with torch.no_grad():
                vscores = torch.sigmoid(cand(val_batch)).cpu().numpy()
            sel = float(pr_auc(labels[val], vscores))
            selmsg = f"val_prauc={sel:.4f}"
        else:
            sel = -float(history.last)  # no val split: prefer the lowest final BCE
            selmsg = f"final_bce={history.last:.4f}"
        print(f"[train_border] restart {r} (seed {args.seed + r}): "
              f"BCE {history.first:.4f}->{history.last:.4f} {selmsg}", flush=True)
        if sel > best_sel:
            best_sel, model, optimizer, best_r = sel, cand, opt, r
    assert model is not None and optimizer is not None
    if args.restarts > 1:
        print(f"[train_border] selected restart {best_r} (best val_prauc={best_sel:.4f})",
              flush=True)

    if ev.size:
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(_batch(ev))).cpu().numpy()
        report = pu_metric_report(scores, labels[ev])
        base = float(labels[ev].mean())
        print(f"[train_border] eval ({args.eval_split}, n={ev.size:,}, base_rate={base:.4f}): "
              + ", ".join(f"{k}={v:.4f}" for k, v in report.items()))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.out, model, optimizer,
        extra={
            "framing": "supervised_subgraph_border",
            "node_dim": node_dim,
            "edge_dim": edim,
            "border_cap": args.border_cap,
            "set_hidden": args.set_hidden,
            "set_out": args.set_out,
            "mlp_hidden": list(args.mlp_hidden),
            "feat_mean": mean.tolist(),
            "feat_std": std.tolist(),
            "use_edges": edge_feats is not None,
            "edge_mean": edge_mean.tolist() if edge_mean is not None else None,
            "edge_std": edge_std.tolist() if edge_std is not None else None,
        },
    )
    print(f"[train_border] wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
