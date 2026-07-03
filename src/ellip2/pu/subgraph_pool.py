"""Stage 2 (supervised subgraph model) — pool cluster features to per-subgraph vectors.

plan.md decisions #1/#2: labels exist only at the **subgraph** level, where the reliable
licit labels are bona-fide negatives, so detection is imbalanced *supervised*
classification — the framing every Elliptic2 SOTA uses. A diagnostic on the real data
confirmed a plain supervised classifier over pooled subgraph features beats the
cluster-level nnPU GNN ~10× (PR-AUC 0.30 vs 0.03); see ``docs/stage2-model-choice.md``.

This module represents each labeled subgraph by pooling its member clusters'
``cluster_features`` with order statistics (``mean``/``max``/``min``/``std``), yielding a
fixed-length vector per subgraph for a downstream classifier (:mod:`ellip2.pu.train_subgraph`).

It reads the feature parquet **one column at a time and keeps only member rows**, so it
runs in a couple of GB even though ``cluster_features`` has 49M rows (the full table
decompresses to ~17GB — loading it whole OOMs a 32GB box).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from ellip2.data import schema

# Order statistics pooled over a subgraph's member clusters. sum preserves subgraph
# size, mean normalizes it, min/max/std capture the spread (plan.md §"Subgraph readout").
POOLS: tuple[str, ...] = ("mean", "max", "min", "std")


@dataclass(frozen=True)
class PooledSubgraphs:
    """Per-subgraph pooled feature matrix + labels/splits.

    Attributes:
        ccids: subgraph ids, in ``subgraphs.parquet`` row order.
        X: ``(n_subgraphs, len(pools) * F)`` float32 pooled features.
        y: ``(n_subgraphs,)`` int; 1 = suspicious, 0 = licit.
        split: ``(n_subgraphs,)`` str; the split each ccId belongs to (``""`` if none).
        feature_names: length ``len(pools) * F`` names, ``"{pool}__{col}"``.
    """

    ccids: list[str]
    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.int_]
    split: npt.NDArray[np.str_]
    feature_names: list[str]


def _apply_pools(xm: npt.NDArray[np.float32], pools: Sequence[str]) -> npt.NDArray[np.float32]:
    """Concatenate the requested column-wise pools of member matrix ``xm`` (M×F)."""
    parts: list[npt.NDArray[np.float32]] = []
    for p in pools:
        if p == "mean":
            parts.append(xm.mean(0))
        elif p == "max":
            parts.append(xm.max(0))
        elif p == "min":
            parts.append(xm.min(0))
        elif p == "std":
            parts.append(xm.std(0))
        elif p == "sum":
            parts.append(xm.sum(0))
        else:
            raise ValueError(f"unknown pool {p!r}")
    return np.concatenate(parts).astype(np.float32)


def pool_subgraph_features(
    features_parquet: Path,
    subgraphs_parquet: Path,
    *,
    split_csv: Path | None = None,
    pools: Sequence[str] = POOLS,
) -> PooledSubgraphs:
    """Pool ``cluster_features`` per labeled subgraph → :class:`PooledSubgraphs`."""
    cols = [c for c in pq.read_schema(str(features_parquet)).names if c != "idx"]
    n_feat = len(cols)

    st = pq.read_table(subgraphs_parquet, columns=["ccId", "ccLabel", "member_idx"])
    ccids = [str(v) for v in st.column("ccId").to_pylist()]
    cclabels = st.column("ccLabel").to_pylist()
    members = [
        np.asarray(m, dtype=np.int64) for m in st.column("member_idx").to_pylist()
    ]
    members = [m[m >= 0] for m in members]

    # member-row-only load: unique member idxs, then one feature column at a time.
    concat = np.concatenate(members) if members else np.zeros(0, np.int64)
    uniq = np.unique(concat)
    member_X = np.empty((uniq.size, n_feat), dtype=np.float32)
    for j, c in enumerate(cols):
        col = pq.read_table(str(features_parquet), columns=[c]).column(0).to_numpy(
            zero_copy_only=False
        )
        if uniq.size:
            member_X[:, j] = col[uniq].astype(np.float32)
        del col

    split_of: dict[str, str] = {}
    if split_csv is not None:
        with open(split_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                split_of[str(r["id"])] = r["split"]

    feats: list[npt.NDArray[np.float32]] = []
    ys: list[int] = []
    sp: list[str] = []
    for cc, lab, m in zip(ccids, cclabels, members, strict=True):
        if m.size:
            xm = member_X[np.searchsorted(uniq, m)]
        else:
            xm = np.zeros((1, n_feat), np.float32)
        feats.append(_apply_pools(xm, pools))
        ys.append(1 if lab == schema.LABEL_SUSPICIOUS else 0)
        sp.append(split_of.get(cc, ""))

    X = (
        np.vstack(feats).astype(np.float32)
        if feats
        else np.zeros((0, len(pools) * n_feat), np.float32)
    )
    feature_names = [f"{p}__{c}" for p in pools for c in cols]
    return PooledSubgraphs(
        ccids=ccids,
        X=X,
        y=np.asarray(ys, dtype=int),
        split=np.asarray(sp, dtype=str),
        feature_names=feature_names,
    )
