"""Stage 2 (border model) — assemble per-subgraph SubgraphBatch from Stage-0 artifacts.

The paper's decisive signal is the subgraph **border**: the outside clusters that fund a
subgraph (senders) and that it pays out to (receivers) — licit vs suspicious *internal*
graphlets are nearly identical (plan.md §"Subgraph-level readout"). This module builds the
border sets purely from existing artifacts (no CSV re-read):

* ``member_idx`` (``subgraphs.parquet``) and ``edge_index.npy`` are both in the remapped
  ``[0, N)`` cluster-idx space, so for each labeled subgraph:
  - **senders**  = external ``src`` of edges ``src -> dst`` with ``dst`` internal, ``src`` not
    in the same subgraph;
  - **receivers** = external ``dst`` of edges with ``src`` internal, ``dst`` not in the same
    subgraph.
* node (43-d) features come from ``node_features.npy`` by row index.

No per-subgraph "source/sink" set is defined anywhere in the code, so we use the plain
topological border (external in-/out-neighbours of any internal node). Each set is
**deduped and capped** per subgraph (``cap``) so a member that happens to be a mega-hub
(degree ~1.7e7) can't explode the batch. Internal *edge* features (95-d) are not in any
artifact — left empty here (Phase 1); the model pools an empty edge set to zeros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch

from ellip2.pu.trainer import SubgraphBatch

IdxArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class BorderSets:
    """Per-subgraph border node idxs (row-aligned to ``subgraphs.parquet``).

    Attributes:
        members: length-K list; ``members[j]`` = internal cluster idxs of subgraph ``j``.
        senders: length-K list; external in-neighbour cluster idxs (deduped, capped).
        receivers: length-K list; external out-neighbour cluster idxs (deduped, capped).
    """

    members: list[IdxArray]
    senders: list[IdxArray]
    receivers: list[IdxArray]


def _group_dedup_cap(
    sg: IdxArray, nd: IdxArray, n_sub: int, cap: int, rng: np.random.Generator
) -> list[IdxArray]:
    """Group border ``(subgraph, node)`` pairs → per-subgraph unique node idxs, capped."""
    out: list[IdxArray] = [np.empty(0, np.int64) for _ in range(n_sub)]
    if sg.size == 0:
        return out
    order = np.argsort(sg, kind="stable")
    sg_s, nd_s = sg[order], nd[order]
    uniq, starts = np.unique(sg_s, return_index=True)
    bounds = np.append(starts, sg_s.size)
    for i, g in enumerate(uniq):
        nodes = np.unique(nd_s[bounds[i] : bounds[i + 1]])
        if nodes.size > cap:
            nodes = rng.choice(nodes, size=cap, replace=False)
        out[int(g)] = nodes
    return out


def extract_border_sets(
    edge_index: npt.NDArray[np.integer],
    members: Sequence[IdxArray],
    n_nodes: int,
    *,
    cap: int = 64,
    seed: int = 0,
) -> BorderSets:
    """Extract deduped, capped border sender/receiver sets for every subgraph.

    Args:
        edge_index: ``(2, E)`` background connectivity (Stage-0 ``edge_index.npy``);
            column ``[s, t]`` is the directed edge ``s -> t``.
        members: per-subgraph member cluster idxs (from ``subgraphs.parquet``), each in
            ``[0, n_nodes)``. Assumed near-disjoint (labeled connected components); on the
            rare overlap the later subgraph wins for the ``node -> subgraph`` map.
        n_nodes: total cluster count.
        cap: max border nodes kept per subgraph per side (hub guard).
        seed: RNG for the per-subgraph subsample when a side exceeds ``cap``.

    Returns:
        A :class:`BorderSets`.
    """
    rng = np.random.default_rng(seed)
    node_sg = np.full(n_nodes, -1, dtype=np.int64)
    for j, m in enumerate(members):
        if m.size:
            node_sg[m] = j

    s = edge_index[0].astype(np.int64, copy=False)
    d = edge_index[1].astype(np.int64, copy=False)
    sg_s, sg_d = node_sg[s], node_sg[d]

    send_mask = (sg_d >= 0) & (sg_s != sg_d)          # external src -> internal dst
    recv_mask = (sg_s >= 0) & (sg_d != sg_s)          # internal src -> external dst
    senders = _group_dedup_cap(sg_d[send_mask], s[send_mask], len(members), cap, rng)
    receivers = _group_dedup_cap(sg_s[recv_mask], d[recv_mask], len(members), cap, rng)
    return BorderSets(members=list(members), senders=senders, receivers=receivers)


def build_subgraph_batch(
    positions: Sequence[int] | npt.NDArray[np.integer],
    border: BorderSets,
    node_features: npt.NDArray[np.float32],
    *,
    mean: npt.NDArray[np.float32] | None = None,
    std: npt.NDArray[np.float32] | None = None,
    edge_dim: int = 95,
    edge_features_by_sg: dict[int, npt.NDArray[np.float32]] | None = None,
    edge_mean: npt.NDArray[np.float32] | None = None,
    edge_std: npt.NDArray[np.float32] | None = None,
) -> SubgraphBatch:
    """Build a :class:`SubgraphBatch` for the subgraphs at ``positions`` (batch order).

    Gathers internal-node / sender / receiver 43-d features (one fancy-index each) and tags
    every row with its position in ``positions``; ``mean``/``std`` z-score the node features.
    When ``edge_features_by_sg`` is given (Phase 2 — from
    :mod:`ellip2.features.internal_edges`) the internal 95-d edge features populate ``edge_x``
    (z-scored by ``edge_mean``/``edge_std``); otherwise ``edge_x`` is empty (Phase 1).
    """
    def collect(sets: list[IdxArray]) -> tuple[IdxArray, IdxArray]:
        idx_parts, bat_parts = [], []
        for bi, pos in enumerate(positions):
            a = sets[pos]
            if a.size:
                idx_parts.append(a)
                bat_parts.append(np.full(a.size, bi, dtype=np.int64))
        if idx_parts:
            return np.concatenate(idx_parts), np.concatenate(bat_parts)
        return np.zeros(0, np.int64), np.zeros(0, np.int64)

    n_idx, n_bat = collect(border.members)
    s_idx, s_bat = collect(border.senders)
    r_idx, r_bat = collect(border.receivers)

    def feats(idx: IdxArray) -> torch.Tensor:
        x = np.asarray(node_features[idx], dtype=np.float32)
        if mean is not None and std is not None:
            x = (x - mean) / std
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    edge_x = torch.zeros((0, edge_dim), dtype=torch.float32)
    edge_batch = torch.zeros(0, dtype=torch.long)
    if edge_features_by_sg is not None:
        e_parts, eb_parts = [], []
        for bi, pos in enumerate(positions):
            ef = edge_features_by_sg.get(int(pos))
            if ef is not None and ef.shape[0]:
                e_parts.append(ef)
                eb_parts.append(np.full(ef.shape[0], bi, dtype=np.int64))
        if e_parts:
            ex = np.concatenate(e_parts).astype(np.float32)
            if edge_mean is not None and edge_std is not None:
                ex = (ex - edge_mean) / edge_std
            edge_x = torch.from_numpy(np.ascontiguousarray(ex, dtype=np.float32))
            edge_batch = torch.from_numpy(np.concatenate(eb_parts))

    return SubgraphBatch(
        sender_x=feats(s_idx),
        sender_batch=torch.from_numpy(s_bat),
        receiver_x=feats(r_idx),
        receiver_batch=torch.from_numpy(r_bat),
        node_x=feats(n_idx),
        node_batch=torch.from_numpy(n_bat),
        edge_x=edge_x,
        edge_batch=edge_batch,
        num_graphs=len(positions),
    )


def fit_node_standardizer(
    positions: Sequence[int] | npt.NDArray[np.integer],
    border: BorderSets,
    node_features: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Z-score stats over all internal+border nodes appearing in ``positions`` (eps-guarded)."""
    idxs = []
    for pos in positions:
        for a in (border.members[pos], border.senders[pos], border.receivers[pos]):
            if a.size:
                idxs.append(a)
    if not idxs:
        f = node_features.shape[1]
        return np.zeros(f, np.float32), np.ones(f, np.float32)
    rows = node_features[np.unique(np.concatenate(idxs))]
    mean = rows.mean(0).astype(np.float32)
    std = rows.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def load_internal_edge_features(
    parquet_path: str | Path,
) -> dict[int, npt.NDArray[np.float32]]:
    """Load ``internal_edge_features.parquet`` → ``{subgraph_position: (E, 95) float32}``.

    Produced by :mod:`ellip2.features.internal_edges`; feeds ``build_subgraph_batch``'s
    ``edge_features_by_sg`` so the border model's internal-edge channel is populated.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(parquet_path)
    sg = table.column("subgraph").to_numpy(zero_copy_only=False).astype(np.int64)
    feat_cols = [c for c in table.column_names if c.startswith("f")]
    if not sg.size:
        return {}
    feats = np.column_stack(
        [table.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in feat_cols]
    )
    out: dict[int, npt.NDArray[np.float32]] = {}
    order = np.argsort(sg, kind="stable")
    sg_s, feats_s = sg[order], feats[order]
    uniq, starts = np.unique(sg_s, return_index=True)
    bounds = np.append(starts, sg_s.size)
    for i, g in enumerate(uniq):
        out[int(g)] = feats_s[bounds[i] : bounds[i + 1]]
    return out


def fit_edge_standardizer(
    positions: Sequence[int] | npt.NDArray[np.integer],
    edge_features_by_sg: dict[int, npt.NDArray[np.float32]],
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Z-score stats over all internal edge rows appearing in ``positions`` (eps-guarded)."""
    parts = [
        edge_features_by_sg[int(pos)]
        for pos in positions
        if int(pos) in edge_features_by_sg and edge_features_by_sg[int(pos)].shape[0]
    ]
    if not parts:
        return np.zeros(0, np.float32), np.ones(0, np.float32)
    rows = np.concatenate(parts)
    mean = rows.mean(0).astype(np.float32)
    std = rows.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std
