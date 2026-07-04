"""Stage 3 — heuristic licit endpoints + one representative exit path per subgraph.

Two thin helpers on top of the reachability engine (:mod:`ellip2.exit_paths.path_search`),
for corroborating a flagged subgraph with a real ≤k-hop exit path to a licit endpoint:

* :func:`endpoints_from_features` — the (missing) ``endpoints.npy`` artifact: the top
  ``endpoint_score`` (T-006 ``path_role``) clusters, i.e. the heuristic licit sinks a
  laundering flow exits into.
* :func:`recover_exit_paths` — for each candidate subgraph, decide (via ``reachability``)
  whether any member reaches the endpoint set within ``max_hops`` and, if so, trace ONE
  representative shortest member→endpoint path by walking down the backward-distance field.
  This stays within the "reachability, not enumeration" contract (SIGN-102): we materialise
  a single path along the already-computed distances, never the exponential path set.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import sparse

from ellip2.exit_paths.path_search import MAX_HOPS, reachability


def endpoints_from_features(
    cluster_features_parquet: Path,
    *,
    percentile: float = 0.999,
    score_col: str = "endpoint_score",
) -> npt.NDArray[np.int64]:
    """Heuristic licit endpoints = cluster idxs whose ``endpoint_score`` is top-``percentile``."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    t = pq.read_table(cluster_features_parquet, columns=["idx", score_col])
    idx = t.column("idx").to_numpy(zero_copy_only=False).astype(np.int64)
    score = t.column(score_col).to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.argsort(idx, kind="stable")
    idx, score = idx[order], score[order]
    thr = float(np.quantile(score, percentile))
    return idx[score >= thr]


def _out_adjacency(edge_index: npt.NDArray[np.integer], n_nodes: int) -> sparse.csr_matrix:
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    a = sparse.coo_matrix(
        (np.ones(src.shape[0], np.int8), (src, dst)), shape=(n_nodes, n_nodes)
    ).tocsr()
    return a


def _trace(
    start: int, back_hops: npt.NDArray[np.int64], adj: sparse.csr_matrix
) -> list[int]:
    """Walk start→endpoint following strictly-decreasing backward distance (shortest path)."""
    d = int(back_hops[start])
    if d < 0:
        return []
    path = [int(start)]
    u = start
    while d > 0:
        nbrs = adj.indices[adj.indptr[u] : adj.indptr[u + 1]]
        nxt = -1
        for w in nbrs:
            if back_hops[w] == d - 1:
                nxt = int(w)
                break
        if nxt < 0:
            return []
        path.append(nxt)
        u = nxt
        d -= 1
    return path


def recover_exit_paths(
    edge_index: npt.NDArray[np.integer],
    members_by_position: Sequence[npt.NDArray[np.int64]],
    positions: Sequence[int],
    endpoints: npt.ArrayLike,
    n_nodes: int,
    *,
    max_hops: int = MAX_HOPS,
    frontier_cap: int | None = None,
    hubs: npt.ArrayLike | None = None,
) -> dict[int, list[int]]:
    """Return ``{position: exit_path}`` — a shortest member→endpoint path, ``[]`` if none ≤k-hop.

    Runs one bounded reachability sweep from the union of the selected subgraphs' members to
    the endpoint set, then traces a single path per subgraph down the backward-distance field.
    """
    cand_parts = [members_by_position[p] for p in positions if members_by_position[p].size]
    candidates = np.unique(np.concatenate(cand_parts)) if cand_parts else np.zeros(0, np.int64)
    if candidates.size == 0:
        return {int(p): [] for p in positions}

    reach = reachability(
        edge_index, candidates, endpoints, n_nodes,
        max_hops=max_hops, frontier_cap=frontier_cap, hubs=hubs,
    )
    back_hops = reach.backward.hops
    adj = _out_adjacency(edge_index, n_nodes)

    out: dict[int, list[int]] = {}
    for p in positions:
        members = members_by_position[p]
        if not members.size:
            out[int(p)] = []
            continue
        mh = back_hops[members]
        valid = mh >= 0
        if not valid.any():
            out[int(p)] = []
            continue
        start = int(members[valid][np.argmin(mh[valid])])  # closest-to-endpoint member
        out[int(p)] = _trace(start, back_hops, adj)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write endpoints.npy from cluster_features' endpoint_score (top percentile)."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Stage 3: write heuristic licit endpoints.npy from endpoint_score.",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--out", required=True, type=Path, help="endpoints.npy")
    p.add_argument("--percentile", type=float, default=0.999)
    p.add_argument("--score-col", default="endpoint_score")
    args = p.parse_args(argv)

    ep = endpoints_from_features(args.features, percentile=args.percentile,
                                 score_col=args.score_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, ep)
    print(f"[endpoints] wrote {ep.size:,} endpoints (top {(1 - args.percentile) * 100:.2f}%) "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
