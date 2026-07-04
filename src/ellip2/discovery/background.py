"""Background discovery — surface NEW suspicious subgraphs in the unlabeled graph.

Everything scored today is one of the 121,810 *labeled* subgraphs; the real goal
is to discover novel laundering structures among the ~48.8M unlabeled background
clusters (plan ``yes-lets-do-it-glittery-coral.md``). This module holds the
discovery machinery; T-026 seeds it with the two small helpers the orchestrator
(T-028 ``discover_background``) needs before it can carve and score candidates:

* :func:`typology_signal_from_features` — Gate 3's structural signal: the
  per-cluster ``source_sink_axis`` column (T-006 ``path_role``) exported as an
  idx-aligned ``(N,)`` array (and written to ``typology_signal.npy`` by
  :func:`main`). Mirrors :func:`ellip2.exit_paths.recover.endpoints_from_features`.
* :func:`known_member_idx` — the exclusion set: the sorted-unique union of every
  labeled subgraph's member cluster idxs from ``subgraphs.parquet``, so Gate 1
  can drop already-known clusters and rank only genuinely NEW structures.

T-027 adds :func:`candidate_member_sets` — the per-candidate subgraph *carve*: it
builds the directed background adjacency once and runs a single **global** backward
BFS from the endpoint set, then, per candidate, a cheap **forward** BFS and a
meet-in-the-middle intersection to recover the member cluster idxs of the ≤k-hop
candidate→endpoint reachability subgraph. Reuses the Stage 3 reachability BFS
(:mod:`ellip2.exit_paths.path_search`); the expensive backward sweep is computed
once and shared across all candidates.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ellip2.exit_paths.path_search import (
    MAX_HOPS,
    _as_node_ids,
    _build_directed_adjacency,
    _hub_mask,
    bfs_reachable,
)


def typology_signal_from_features(
    cluster_features_parquet: Path,
    *,
    score_col: str = "source_sink_axis",
) -> npt.NDArray[np.float64]:
    """Idx-aligned ``(N,)`` typology signal — the ``source_sink_axis`` column of features.

    Rows are sorted by the ``idx`` key so row ``i`` is cluster ``i`` in ``[0, N)`` and
    the returned array can be indexed positionally. Raises if the score column is
    missing or ``idx`` is not a contiguous ``0..N-1`` range.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(cluster_features_parquet, columns=["idx", score_col])
    idx = table.column("idx").to_numpy(zero_copy_only=False).astype(np.int64)
    signal = table.column(score_col).to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.argsort(idx, kind="stable")
    idx, signal = idx[order], signal[order]
    if not np.array_equal(idx, np.arange(idx.size)):
        raise ValueError(
            f"feature 'idx' in {cluster_features_parquet} is not a contiguous 0..N-1 range"
        )
    return signal


def known_member_idx(
    subgraphs_path: Path,
    *,
    n_nodes: int | None = None,
) -> npt.NDArray[np.int64]:
    """Sorted-unique union of every labeled subgraph's member cluster idxs.

    The exclusion set for background discovery: Gate 1 drops these already-known
    clusters so only NEW structures surface. Optionally bounded to ``[0, n_nodes)``.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_path, columns=["member_idx"])
    parts = [
        np.asarray(members, dtype=np.int64)
        for members in table.column("member_idx").to_pylist()
        if members
    ]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    idx = np.unique(np.concatenate(parts))
    if n_nodes is not None:
        idx = idx[(idx >= 0) & (idx < n_nodes)]
    return idx


def candidate_member_sets(
    edge_index: npt.ArrayLike,
    candidate_ids: npt.ArrayLike,
    endpoints: npt.ArrayLike,
    n_nodes: int,
    *,
    max_hops: int = MAX_HOPS,
    frontier_cap: int | None = None,
    hubs: npt.ArrayLike | None = None,
) -> dict[int, npt.NDArray[np.int64]]:
    """Carve the ≤k-hop candidate→endpoint reachability subgraph of each candidate.

    For every candidate we want the node set that lies on some ≤``max_hops``
    directed path from that candidate to an endpoint — the concrete "exit-path
    subgraph" that gets border-scored and reported downstream. This is the
    meet-in-the-middle carve of :func:`ellip2.exit_paths.path_search.reachability`,
    specialised to a *single* source and run per candidate.

    The expensive half is shared: the directed adjacency is built once and a single
    **global** backward BFS is run from the endpoint set (its result is identical for
    every candidate). Only the cheap **forward** BFS — seeded from one candidate —
    runs per candidate; survivors are ``fwd.reached & back.reached &
    (fwd.hops + back.hops <= max_hops)`` (see decision #4). The endpoint ids
    themselves are dropped from each returned member set (endpoints are the target,
    not part of the discovered structure); a candidate that cannot reach any
    endpoint within ``max_hops`` maps to an **empty** array.

    Args:
        edge_index: ``(2, E)`` directed edges (Stage 0 ``edge_index.npy``); not
            symmetrised. ``E == 0`` allowed.
        candidate_ids: candidate source node ids (e.g. top-percentile PU scores,
            already excluding known members). Deduplicated internally.
        endpoints: endpoint node ids (e.g. the T-006 ``endpoint_score`` heuristic).
        n_nodes: total node count.
        max_hops: reachability horizon (default :data:`MAX_HOPS` = 6).
        frontier_cap: optional per-level frontier cap for both sweeps.
        hubs: optional hub bool-mask or id-iterable; hubs are stopped at (not
            transited) in both sweeps.

    Returns:
        ``{candidate_id: member_idx}`` — one entry per (deduplicated) candidate,
        ``member_idx`` a sorted ``int64`` array of the carve's node ids with the
        endpoint ids removed (empty when the candidate reaches no endpoint).

    Raises:
        ValueError: on a malformed ``edge_index``, out-of-range node ids, or an
            invalid ``max_hops`` / ``frontier_cap``.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    ei = ei.astype(np.int64, copy=False)
    if ei.shape[1] and (ei.min() < 0 or ei.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"min {int(ei.min())}, max {int(ei.max())}"
        )

    candidates = _as_node_ids("candidate_ids", candidate_ids, n_nodes)
    endpoint_ids = _as_node_ids("endpoints", endpoints, n_nodes)
    hub_mask = _hub_mask(hubs, n_nodes)

    a = _build_directed_adjacency(ei, n_nodes)
    a_t = a.transpose().tocsr()

    # Global backward sweep from the endpoint set — identical for every candidate,
    # so compute it once and reuse.
    backward = bfs_reachable(
        a_t, endpoint_ids, max_hops=max_hops, frontier_cap=frontier_cap, hub_mask=hub_mask
    )

    drop_endpoint = np.zeros(n_nodes, dtype=bool)
    drop_endpoint[endpoint_ids] = True

    member_sets: dict[int, npt.NDArray[np.int64]] = {}
    for cand in candidates:
        forward = bfs_reachable(
            a, [int(cand)], max_hops=max_hops, frontier_cap=frontier_cap, hub_mask=hub_mask
        )
        both = forward.reached & backward.reached
        total = forward.hops + backward.hops  # valid only where both reached
        survivors = both & (total <= max_hops) & ~drop_endpoint
        member_sets[int(cand)] = np.nonzero(survivors)[0].astype(np.int64)
    return member_sets


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the ``(N,)`` typology signal (``source_sink_axis``) to a ``.npy``."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Background discovery: export the source_sink_axis typology signal (N,).",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--out", required=True, type=Path, help="(N,) typology_signal.npy")
    p.add_argument("--score-col", default="source_sink_axis")
    args = p.parse_args(argv)

    signal = typology_signal_from_features(args.features, score_col=args.score_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, signal)
    print(f"[typology] wrote {signal.size:,} typology signals -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
