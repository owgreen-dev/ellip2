"""Stage 3 — bounded ≤k-hop *reachability* between candidates and endpoints.

plan.md Resolved decision #4 + §3. This is the "does illicit value reach a licit
endpoint within ≤6 hops?" test, run as **reachability, NOT path enumeration**
(SIGN-102): we never materialise the (exponentially many) source→endpoint paths;
we only decide, per candidate, whether *some* ≤k-hop directed path to an endpoint
exists, and optionally carve out the induced subgraph of nodes that lie on one.

Two complementary boolean BFS sweeps over the directed background graph, each via
:mod:`scipy.sparse` SpMV (no per-node Python loop, no dense ``N×N``):

* **Backward** multi-source BFS from the **endpoint set** on the *transposed*
  adjacency. A node ``v`` reached at backward-distance ``d`` has a directed path
  ``v → … → endpoint`` of length ``d`` — i.e. ``v`` *reaches* an endpoint in ``d``
  hops. This sweep alone yields the headline output: a candidate reaches the
  endpoint set iff its backward distance is ``≤ k``.
* **Forward** multi-source BFS from the **candidate sources** on the adjacency
  itself. Combined with the backward sweep it pins down the *meet-in-the-middle*
  node set — every node ``m`` with ``d_forward(m) + d_backward(m) ≤ k`` lies on a
  ≤k-hop source→endpoint path. That mask is what the induced-subgraph extraction
  operates on.

Two safeguards on the heavy-tailed background graph (in-degree runs into the
millions), straight from decision #4:

* **Per-level frontier caps.** Each BFS level processes at most ``frontier_cap``
  newly discovered nodes (deterministically, lowest node id first). Bounds the
  cost of a hub explosion; the cost is possible *under*-reporting of reachability,
  which is recorded in :attr:`BFSResult.capped_levels` so callers never mistake a
  capped sweep for an exhaustive one.
* **Hubs excluded from pass-through.** A node flagged a hub is *stopped at*, never
  transited: when the traversal first reaches a hub it is marked reached (recorded
  with its hop) but is **not** expanded to its own neighbours. This matches the
  Elliptic2 construction ("stop at a labeled node / change-of-ownership"): an
  exchange is an *endpoint*, not a conduit, so a path that only reaches an endpoint
  *through* some other hub does not count. Seed nodes are always expanded (they are
  the origin of the sweep), even if flagged hubs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

#: Default reachability horizon (Elliptic2 exit paths are searched to ≤6 hops).
MAX_HOPS = 6


@dataclass(frozen=True)
class BFSResult:
    """Outcome of one bounded, capped, hub-aware BFS sweep.

    Attributes:
        reached: ``(N,)`` bool; True for every node visited within ``max_hops``
            (including seeds and stopped-at hubs).
        hops: ``(N,)`` int; minimum hop distance from the seed set, or ``-1`` for
            unreached nodes. Seeds are ``0``.
        frontiers: per-level arrays of newly reached node ids; ``frontiers[0]`` is
            the seed set, ``frontiers[d]`` the nodes first reached at distance
            ``d``. Each (post-seed) level has at most ``frontier_cap`` entries.
        capped_levels: levels whose discovery was truncated by ``frontier_cap``
            (reachability past these is possibly under-reported).
    """

    reached: npt.NDArray[np.bool_]
    hops: npt.NDArray[np.int64]
    frontiers: list[npt.NDArray[np.int64]]
    capped_levels: list[int]


@dataclass(frozen=True)
class ReachabilityResult:
    """Combined forward+backward reachability between candidates and endpoints.

    Attributes:
        candidates: ``(C,)`` candidate source node ids (deduplicated, sorted).
        endpoints: ``(P,)`` endpoint node ids (deduplicated, sorted).
        candidate_reaches: ``(C,)`` bool aligned to :attr:`candidates`; True iff
            that candidate reaches the endpoint **set** within ``max_hops`` hops.
        survivors: ``(N,)`` bool; the meet-in-the-middle node set — nodes lying on
            at least one ≤``max_hops`` candidate→endpoint directed path. Input to
            :func:`induced_subgraph`.
        forward: the forward BFS sweep from the candidate sources.
        backward: the backward BFS sweep from the endpoint set.
        max_hops: the horizon used for both sweeps.
    """

    candidates: npt.NDArray[np.int64]
    endpoints: npt.NDArray[np.int64]
    candidate_reaches: npt.NDArray[np.bool_]
    survivors: npt.NDArray[np.bool_]
    forward: BFSResult
    backward: BFSResult
    max_hops: int


def _build_directed_adjacency(
    edge_index: npt.NDArray[np.int64], n_nodes: int
) -> sparse.csr_matrix:
    """Binary **directed** adjacency (csr): ``A[i, j] == 1`` iff edge ``i → j``.

    Unlike :mod:`ellip2.features.neighborhood`, the graph is **not** symmetrised —
    exit-path reachability is inherently directional (illicit source → licit sink).
    Parallel edges collapse to one; self-loops are dropped (they add no reach).
    """
    src = edge_index[0]
    dst = edge_index[1]
    data = np.ones(src.shape[0], dtype=np.float64)
    a = sparse.coo_matrix((data, (src, dst)), shape=(n_nodes, n_nodes)).tocsr()
    if a.nnz:
        a.data[:] = 1.0
    a.setdiag(0.0)
    a.eliminate_zeros()
    return a


def _as_node_ids(name: str, ids: npt.ArrayLike, n_nodes: int) -> npt.NDArray[np.int64]:
    arr = np.unique(np.asarray(ids, dtype=np.int64).ravel())
    if arr.size and (arr.min() < 0 or arr.max() >= n_nodes):
        raise ValueError(
            f"{name} contains a node id outside [0, {n_nodes})"
        )
    return arr


def _hub_mask(
    hubs: npt.ArrayLike | None, n_nodes: int
) -> npt.NDArray[np.bool_] | None:
    """Coerce ``hubs`` to a ``(N,)`` bool mask, or ``None`` when unset/empty.

    Accepts either a boolean mask of length ``n_nodes`` or an iterable of hub node
    ids.
    """
    if hubs is None:
        return None
    arr = np.asarray(hubs)
    if arr.dtype == np.bool_:
        if arr.shape[0] != n_nodes:
            raise ValueError(
                f"hub mask has {arr.shape[0]} entries but n_nodes={n_nodes}"
            )
        return arr if arr.any() else None
    mask = np.zeros(n_nodes, dtype=bool)
    ids = np.asarray(arr, dtype=np.int64).ravel()
    if ids.size:
        if ids.min() < 0 or ids.max() >= n_nodes:
            raise ValueError(f"hub ids contain a node outside [0, {n_nodes})")
        mask[ids] = True
    return mask if mask.any() else None


def bfs_reachable(
    adjacency: sparse.spmatrix,
    seeds: npt.ArrayLike,
    *,
    max_hops: int = MAX_HOPS,
    frontier_cap: int | None = None,
    hub_mask: npt.NDArray[np.bool_] | None = None,
) -> BFSResult:
    """Bounded multi-source BFS over the **out-edges** of ``adjacency``.

    Each level advances from the current frontier to its out-neighbours via a
    boolean SpMV (``adjacencyᵀ · frontier_indicator``). For a *backward* sweep,
    pass the transposed adjacency so its out-edges are the original in-edges.

    Args:
        adjacency: ``(N, N)`` scipy sparse adjacency; ``adjacency[i, j] != 0``
            means an edge ``i → j`` is followed from ``i`` to ``j``.
        seeds: source node ids (deduplicated internally); always expanded once.
        max_hops: maximum BFS depth (number of edge steps).
        frontier_cap: if set, at most this many newly discovered nodes are kept
            per level (lowest node id first); truncated levels are recorded.
        hub_mask: ``(N,)`` bool; reached hubs are recorded but not expanded
            (excluded from pass-through). Seeds are exempt — always expanded.

    Returns:
        A :class:`BFSResult`.

    Raises:
        ValueError: on a non-square adjacency, a negative ``max_hops``, a
            non-positive ``frontier_cap``, or an out-of-range seed.
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {adjacency.shape}")
    n_nodes = int(adjacency.shape[0])
    if max_hops < 0:
        raise ValueError(f"max_hops must be non-negative, got {max_hops}")
    if frontier_cap is not None and frontier_cap < 1:
        raise ValueError(f"frontier_cap must be >= 1, got {frontier_cap}")

    seed_ids = _as_node_ids("seeds", seeds, n_nodes)
    # Step matrix: (adjacencyᵀ · f)[k] counts frontier predecessors of k under
    # adjacencyᵀ == frontier nodes i with edge i → k. So `> 0` is the out-neighbour
    # set of the frontier. Transpose once, not per level.
    step = adjacency.transpose().tocsr()

    reached = np.zeros(n_nodes, dtype=bool)
    hops = np.full(n_nodes, -1, dtype=np.int64)
    frontiers: list[npt.NDArray[np.int64]] = []
    capped_levels: list[int] = []

    reached[seed_ids] = True
    hops[seed_ids] = 0
    frontiers.append(seed_ids)
    # Seeds are the origin and are always expanded, even if flagged hubs.
    frontier = seed_ids

    for level in range(1, max_hops + 1):
        if frontier.size == 0:
            break
        fvec = np.zeros(n_nodes, dtype=np.float64)
        fvec[frontier] = 1.0
        counts = np.asarray(step.dot(fvec)).ravel()
        cand = np.nonzero(counts > 0)[0]
        new = np.sort(cand[~reached[cand]])
        if new.size == 0:
            break
        if frontier_cap is not None and new.size > frontier_cap:
            new = new[:frontier_cap]
            capped_levels.append(level)

        reached[new] = True
        hops[new] = level
        frontiers.append(new)

        # Hubs are recorded above but stopped at: drop them from the next frontier.
        frontier = new[~hub_mask[new]] if hub_mask is not None else new

    return BFSResult(
        reached=reached, hops=hops, frontiers=frontiers, capped_levels=capped_levels
    )


def reachability(
    edge_index: npt.ArrayLike,
    sources: npt.ArrayLike,
    endpoints: npt.ArrayLike,
    n_nodes: int,
    *,
    max_hops: int = MAX_HOPS,
    frontier_cap: int | None = None,
    hubs: npt.ArrayLike | None = None,
) -> ReachabilityResult:
    """Bounded candidate→endpoint reachability via two-sided BFS.

    Runs a forward sweep from ``sources`` and a backward sweep from ``endpoints``
    (on the transposed adjacency), each capped and hub-aware, then:

    * decides per candidate whether it reaches the endpoint set within
      ``max_hops`` (backward distance ``≤ max_hops``), and
    * marks the meet-in-the-middle node set
      ``{m : d_forward(m) + d_backward(m) ≤ max_hops}`` — the nodes on at least one
      ≤``max_hops`` source→endpoint path — as :attr:`ReachabilityResult.survivors`.

    Args:
        edge_index: ``(2, E)`` directed edges (Stage 0 ``edge_index.npy``); the
            graph is **not** symmetrised. ``E == 0`` allowed.
        sources: candidate source node ids (e.g. top-percentile PU scores).
        endpoints: endpoint node ids (e.g. the T-006 ``endpoint_score`` heuristic).
        n_nodes: total node count.
        max_hops: reachability horizon (default :data:`MAX_HOPS` = 6).
        frontier_cap: optional per-level frontier cap for both sweeps.
        hubs: optional hub bool-mask or id-iterable; hubs are stopped at (not
            transited) in both sweeps.

    Returns:
        A :class:`ReachabilityResult`.

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

    candidates = _as_node_ids("sources", sources, n_nodes)
    endpoint_ids = _as_node_ids("endpoints", endpoints, n_nodes)
    hub_mask = _hub_mask(hubs, n_nodes)

    a = _build_directed_adjacency(ei, n_nodes)
    forward = bfs_reachable(
        a, candidates, max_hops=max_hops, frontier_cap=frontier_cap, hub_mask=hub_mask
    )
    backward = bfs_reachable(
        a.transpose().tocsr(),
        endpoint_ids,
        max_hops=max_hops,
        frontier_cap=frontier_cap,
        hub_mask=hub_mask,
    )

    # A candidate reaches the endpoint set iff it is within backward distance k of
    # some endpoint (backward sweep = "who reaches an endpoint within d hops").
    if candidates.size:
        b_hops = backward.hops[candidates]
        candidate_reaches = (b_hops >= 0) & (b_hops <= max_hops)
    else:
        candidate_reaches = np.zeros(0, dtype=bool)

    both = forward.reached & backward.reached
    total = forward.hops + backward.hops  # valid only where both reached
    survivors = both & (total <= max_hops)

    return ReachabilityResult(
        candidates=candidates,
        endpoints=endpoint_ids,
        candidate_reaches=candidate_reaches,
        survivors=survivors,
        forward=forward,
        backward=backward,
        max_hops=max_hops,
    )


def induced_subgraph(
    edge_index: npt.ArrayLike,
    node_mask: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Extract the induced subgraph over the survivor nodes.

    Keeps only edges whose **both** endpoints are selected, and relabels node ids
    to a contiguous ``[0, M)`` range in original-id order.

    Args:
        edge_index: ``(2, E)`` directed edges.
        node_mask: either a ``(N,)`` bool mask or an iterable of node ids to keep.

    Returns:
        ``(sub_edge_index, node_ids)`` where ``sub_edge_index`` is ``(2, E')`` in
        the relabelled space and ``node_ids`` maps local id → original node id
        (``node_ids[local] == original``).
    """
    ei = np.asarray(edge_index, dtype=np.int64)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")

    mask = np.asarray(node_mask)
    if mask.dtype == np.bool_:
        node_ids = np.nonzero(mask)[0].astype(np.int64)
    else:
        node_ids = np.unique(np.asarray(mask, dtype=np.int64).ravel())

    # Dense original→local relabel map (-1 for dropped nodes).
    n = int(ei.max()) + 1 if ei.shape[1] else 0
    n = max(n, int(node_ids.max()) + 1 if node_ids.size else 0)
    relabel = np.full(n, -1, dtype=np.int64)
    relabel[node_ids] = np.arange(node_ids.size, dtype=np.int64)

    if ei.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64), node_ids
    keep = (relabel[ei[0]] >= 0) & (relabel[ei[1]] >= 0)
    sub = np.stack([relabel[ei[0][keep]], relabel[ei[1][keep]]])
    return sub.astype(np.int64), node_ids
