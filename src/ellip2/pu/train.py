"""Stage 2 (train) — cluster-level nnPU scorer over Elliptic2 clusters.

Trains the heterophily-tolerant encoder (:class:`~ellip2.pu.encoder.HeterophilyEncoder`)
plus a linear PU head (:class:`~ellip2.pu.trainer.ClusterScorer`) with the
non-negative PU risk (Kiryo 2017) — the **cluster-level** framing of plan.md
Resolved decision #2. Positives are the clusters that are members of a
*suspicious* subgraph; every other cluster is unlabeled. The checkpoint written
here is consumed by :mod:`ellip2.pu.score`, whose per-cluster scores feed Stage 3
discovery (:mod:`ellip2.discovery.discover`).

**Minibatch training (T-012).** The real background graph is 49M nodes / 196M
edges — a full-graph GNN forward needs ~33GB+ just to gather the first layer's
messages, which OOMs any single GPU. So training runs over **fanout-capped
neighbor-sampled minibatches** (:func:`ellip2.graph.neighbor_sampling.build_neighbor_loader`,
PyG ``NeighborLoader`` backed by ``pyg-lib``): each step seeds on a mix of
positive + unlabeled clusters, samples their k-hop neighborhoods, and computes
the nnPU risk on the seed logits. On a box without the ``pyg-lib``/``torch-sparse``
sampler kernels (the CPU test env) it transparently falls back to the pure-torch
reference sampler, so small graphs still train.

Example:
    python scripts/train.py \
        --features artifacts/features/cluster_features.parquet \
        --edge-index artifacts/ingest/edge_index.npy \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/model.pt --device cuda
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from ellip2.data import schema
from ellip2.pu.encoder import HeterophilyEncoder
from ellip2.pu.nnpu_loss import nnpu_risk, nnpu_risk_for_backward
from ellip2.pu.prior_estimation import tice_prior
from ellip2.pu.trainer import ClusterScorer, save_checkpoint


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
    """Load ``cluster_features.parquet`` → ``(feature_columns, X)`` (float32, N×F),
    rows sorted by the ``idx`` key so row ``i`` is cluster ``i`` in ``[0, N)``."""
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
    ``split_name`` contribute positives — keeping test-split labels out of training.
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
        in_dim, cfg.hidden, cfg.emb_dim,
        num_layers=cfg.num_layers, aggr=cfg.aggr,
        dropout=cfg.dropout, normalize=cfg.normalize,
    )
    return ClusterScorer(encoder, emb_dim=cfg.emb_dim)


def estimate_prior(
    X: np.ndarray, mask: np.ndarray, *, sample: int, seed: int
) -> float:
    """TIcE class-prior estimate, subsampling the unlabeled set to bound RAM."""
    pos_idx = np.flatnonzero(mask)
    unl_idx = np.flatnonzero(~mask)
    rng = np.random.default_rng(seed)
    if unl_idx.size > sample:
        unl_idx = rng.choice(unl_idx, size=sample, replace=False)
    sel = np.concatenate([pos_idx, unl_idx])
    est = tice_prior(X[sel].astype(np.float64), mask[sel])
    print(
        f"[train] estimated prior pi_p={est.prior:.4g} "
        f"(c_hat={est.label_frequency:.4g}, gamma={est.labeled_fraction:.4g})"
    )
    return est.prior


def _sampler_kernels_available() -> bool:
    """True if pyg-lib / torch-sparse are importable (needed by NeighborLoader)."""
    return (
        importlib.util.find_spec("pyg_lib") is not None
        or importlib.util.find_spec("torch_sparse") is not None
    )


class SeedBatcher:
    """Re-iterable source of ``(x_batch, edge_index_batch, n_id, batch_size)`` minibatches.

    Built ONCE over a fixed seed pool — crucially, the PyG ``NeighborLoader`` (which
    materializes a multi-GB CSC of the whole graph) is constructed a single time and
    re-iterated each epoch, rather than rebuilt per epoch (which leaked the CSC and
    OOMed the host). ``x_batch``/``edge_index_batch`` land on ``device``; ``n_id``
    (global idxs, seeds first) stays on CPU for label lookup. Uses ``NeighborLoader``
    when the sampler kernels are present, else the pure-torch reference sampler.
    """

    def __init__(
        self,
        x_all: Tensor,
        edge_index: Tensor,
        seeds: Tensor,
        *,
        num_neighbors: Sequence[int],
        batch_size: int,
        shuffle: bool,
        device: torch.device,
    ) -> None:
        from torch_geometric.data import Data  # noqa: PLC0415

        from ellip2.graph.neighbor_sampling import (  # noqa: PLC0415
            NeighborSamplingConfig,
            build_neighbor_loader,
        )

        self._x_all = x_all
        self._seeds = seeds
        self._device = device
        self._cfg = NeighborSamplingConfig(
            num_neighbors=tuple(num_neighbors), batch_size=batch_size, shuffle=shuffle
        )
        self._data = Data(x=x_all, edge_index=edge_index, num_nodes=x_all.size(0))
        self._use_pyg = _sampler_kernels_available()
        self._loader = (
            build_neighbor_loader(self._data, seeds, self._cfg) if self._use_pyg else None
        )

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor, Tensor, int]]:
        dev = self._device
        if self._loader is not None:
            for b in self._loader:
                yield b.x.to(dev), b.edge_index.to(dev), b.n_id.cpu(), int(b.batch_size)
        else:  # CPU fallback (tests / tiny graphs): kernel-free reference sampler
            from ellip2.graph.neighbor_sampling import iter_subgraph_batches  # noqa: PLC0415

            for sb in iter_subgraph_batches(self._data, self._seeds, self._cfg):
                x_b = self._x_all[sb.n_id].to(dev)
                yield x_b, sb.edge_index.to(dev), sb.n_id.cpu(), sb.batch_size


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 2: train the cluster-level nnPU scorer (minibatch).",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--edge-index", required=True, type=Path, help="(2, E) edge_index.npy")
    p.add_argument("--subgraphs", required=True, type=Path, help="subgraphs.parquet")
    p.add_argument("--split-csv", type=Path, default=None,
                   help="split.csv; restrict positives to --split-name")
    p.add_argument("--split-name", default="train")
    p.add_argument("--out", required=True, type=Path, help="output checkpoint .pt")
    p.add_argument("--prior", type=float, default=None,
                   help="class prior pi_p; estimated via TIcE when omitted")
    p.add_argument("--prior-sample", type=int, default=200_000)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--aggr", default="mean", choices=["mean", "max", "sum", "add"])
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--beta", type=float, default=0.0, help="nnPU clamp bound (Kiryo)")
    p.add_argument("--gamma", type=float, default=1.0, help="nnPU ascent scale (Kiryo)")
    # minibatch neighbor-sampling knobs
    p.add_argument("--batch-size", type=int, default=512, help="seed nodes per minibatch")
    p.add_argument("--num-neighbors", type=int, nargs="+", default=None,
                   help="per-hop fanout (default = [15]*num_layers)")
    p.add_argument("--unlabeled-ratio", type=float, default=20.0,
                   help="unlabeled seeds per positive seed in the training pool")
    p.add_argument("--device", default="cpu", help="cpu or cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    feature_columns, X = load_features(args.features)
    n_nodes = X.shape[0]
    edge_index = np.load(args.edge_index)
    mask = positive_mask(
        args.subgraphs, n_nodes, split_csv=args.split_csv, split_name=args.split_name
    )
    n_pos = int(mask.sum())
    if n_pos == 0:
        raise SystemExit("no positive clusters found — check --subgraphs/--split-csv")

    prior = (
        args.prior if args.prior is not None
        else estimate_prior(X, mask, sample=args.prior_sample, seed=args.seed)
    )

    device = torch.device(args.device)
    x_all = torch.from_numpy(X)                       # CPU; batches move to device
    ei_all = torch.from_numpy(np.asarray(edge_index)).long()
    pos_mask_t = torch.from_numpy(mask)               # CPU bool, for label lookup

    num_layers = args.num_layers
    num_neighbors = args.num_neighbors or [15] * num_layers
    if len(num_neighbors) != num_layers:
        raise SystemExit(
            f"--num-neighbors has {len(num_neighbors)} hops but --num-layers={num_layers}"
        )

    cfg = EncoderConfig(
        hidden=args.hidden, emb_dim=args.emb_dim, num_layers=num_layers,
        aggr=args.aggr, dropout=args.dropout, normalize=args.normalize,
    )
    model = build_scorer(X.shape[1], cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training seed pool: all positives + a balanced unlabeled sample.
    pos_ids = np.flatnonzero(mask)
    unl_all = np.flatnonzero(~mask)
    n_unl = min(unl_all.size, int(args.unlabeled_ratio * n_pos))
    kernels = _sampler_kernels_available()
    print(
        f"[train] N={n_nodes:,} F={X.shape[1]} E={ei_all.shape[1]:,} positives={n_pos:,} "
        f"unl_pool={n_unl:,} prior={prior:.4g} epochs={args.epochs} bs={args.batch_size} "
        f"fanout={num_neighbors} device={args.device} sampler={'pyg' if kernels else 'fallback'}"
    )

    # Fixed training seed pool (all positives + a balanced unlabeled sample), with
    # the sampler built ONCE and reshuffled each epoch (shuffle=True) — rebuilding
    # per epoch leaked the multi-GB CSC and OOMed the host.
    if n_unl < unl_all.size:
        unl_ids = rng.choice(unl_all, size=n_unl, replace=False)
    else:
        unl_ids = unl_all
    seeds = torch.from_numpy(np.concatenate([pos_ids, unl_ids])).long()
    batcher = SeedBatcher(
        x_all, ei_all, seeds, num_neighbors=num_neighbors,
        batch_size=args.batch_size, shuffle=True, device=device,
    )

    first_risk = last_risk = float("nan")
    model.train()
    for epoch in range(args.epochs):
        risks: list[float] = []
        for x_b, ei_b, n_id, bs in batcher:
            seed_pos = pos_mask_t[n_id[:bs]].to(device)
            logits = model(x_b, ei_b)[:bs]
            p_logits, u_logits = logits[seed_pos], logits[~seed_pos]
            if p_logits.numel() == 0 or u_logits.numel() == 0:
                continue
            loss = nnpu_risk_for_backward(
                p_logits, u_logits, prior, beta=args.beta, gamma=args.gamma
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                val = nnpu_risk(p_logits.detach(), u_logits.detach(), prior, beta=args.beta)
            assert isinstance(val, Tensor)  # return_parts=False → single tensor
            risks.append(float(val))
        epoch_risk = float(np.mean(risks)) if risks else float("nan")
        if epoch == 0:
            first_risk = epoch_risk
        last_risk = epoch_risk
        print(f"[train] epoch {epoch + 1}/{args.epochs} nnpu_risk={epoch_risk:.4f} "
              f"({len(risks)} batches)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        args.out, model, optimizer,
        extra={
            "framing": "cluster_nnpu_minibatch",
            "in_dim": X.shape[1],
            "feature_columns": feature_columns,
            "encoder": asdict(cfg),
            "num_neighbors": list(num_neighbors),
            "prior": float(prior),
            "n_nodes": n_nodes,
            "epochs": args.epochs,
            "loss_first": first_risk,
            "loss_last": last_risk,
        },
    )
    print(f"[train] risk {first_risk:.4f} -> {last_risk:.4f}; wrote {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
