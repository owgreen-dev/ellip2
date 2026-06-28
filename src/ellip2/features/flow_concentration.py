"""Stage 1 — per-cluster flow-concentration features.

plan.md §4 (Flow concentration). Peeling chains and layering show up as a node
that pushes most of its value through a single (or a few) counterpart(y/ies)
while sprinkling small change elsewhere. We fingerprint that with three
concentration measures of a cluster's *flow*, computed separately for its
incoming and outgoing edges over the directed background graph:

    gini                    Gini coefficient of the flow distribution (0 = all
                            counterparties equal, → 1 = one dominates)
    hhi                     Herfindahl–Hirschman index = sum of squared shares
                            (1/n for n equal counterparties, → 1 = one dominates)
    max_counterparty_share  share of total flow on the single largest counterparty

The grouping convention mirrors :mod:`ellip2.features.degree`:

    out-flow of node ``v`` == edges with source ``v`` (row 0 of edge_index),
                              grouped by their destination (the counterparty)
    in-flow  of node ``v`` == edges with target ``v`` (row 1 of edge_index),
                              grouped by their source (the counterparty)

**Counterparty aggregation.** "Flow concentration" is about *who* the value goes
to, not how many transactions it took, so the per-edge weights are first summed
per distinct counterparty; parallel edges between the same ordered pair merge
into one share. All three measures are then computed over that per-counterparty
weight distribution, which keeps ``max_counterparty_share`` and the
Gini/HHI consistent.

Definitions for a cluster with per-counterparty weights ``w_1..w_n`` (``w_i ≥ 0``,
``S = Σ w_i > 0``):

    HHI   = Σ (w_i / S)²
    share = max_i w_i / S
    Gini  = ( Σ_i (2·rank_i − n − 1)·w_(i) ) / (n · S),  w_(i) sorted ascending

A single counterparty gives ``gini = 0``, ``hhi = 1``, ``share = 1`` (fully
concentrated). A cluster with no edge in a direction (``S == 0``) takes a
configurable ``empty_value`` (default ``0.0``) rather than NaN, so the assembled
feature frame (T-007) stays finite.

Pure numpy (``unique`` / ``bincount`` / ``lexsort`` segment reductions) — no
pandas, no per-cluster Python loop; the arrays are already materialized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DIRECTIONS: tuple[str, ...] = ("in", "out")
MEASURES: tuple[str, ...] = ("gini", "hhi", "max_counterparty_share")

# Stable column order for the assembled feature frame (T-007 joins on these).
COLUMNS: tuple[str, ...] = tuple(
    f"{d}_{m}" for d in DIRECTIONS for m in MEASURES
)


def _direction_concentration(
    cluster: npt.NDArray[np.int64],
    counterparty: npt.NDArray[np.int64],
    weights: npt.NDArray[np.float64],
    n_nodes: int,
    empty_value: float,
) -> dict[str, npt.NDArray[np.float64]]:
    """Gini/HHI/max-share per cluster over per-counterparty summed weights."""
    def empty() -> npt.NDArray[np.float64]:
        return np.full(n_nodes, float(empty_value), dtype=np.float64)

    if cluster.size == 0:
        return {"gini": empty(), "hhi": empty(), "max_counterparty_share": empty()}

    # Merge parallel edges: sum weight per distinct (cluster, counterparty) pair.
    # counterparty < n_nodes, so cluster * n_nodes + counterparty is collision-free.
    key = cluster * n_nodes + counterparty
    uniq, inv = np.unique(key, return_inverse=True)
    inv = inv.ravel()
    mw = np.bincount(inv, weights=weights).astype(np.float64)  # (U,) merged weight
    mc = (uniq // n_nodes).astype(np.int64)                    # cluster per pair

    total = np.bincount(mc, weights=mw, minlength=n_nodes)
    sumsq = np.bincount(mc, weights=mw * mw, minlength=n_nodes)
    cnt = np.bincount(mc, minlength=n_nodes).astype(np.int64)
    maxw = np.zeros(n_nodes, dtype=np.float64)
    np.maximum.at(maxw, mc, mw)

    has = total > 0
    gini = empty()
    hhi = empty()
    share = empty()
    hhi[has] = sumsq[has] / total[has] ** 2
    share[has] = maxw[has] / total[has]

    # Gini via the sorted-rank closed form, vectorised across clusters.
    order = np.lexsort((mw, mc))  # primary: cluster asc; secondary: weight asc
    sc = mc[order]
    sw = mw[order]
    # Start offset of each cluster's contiguous block in the cluster-sorted order.
    start = np.concatenate(([0], np.cumsum(cnt)[:-1])).astype(np.int64)
    rank = np.arange(sc.size, dtype=np.int64) - start[sc] + 1  # 1-based ascending
    n_row = cnt[sc]
    term = (2 * rank - n_row - 1).astype(np.float64) * sw
    numer = np.bincount(sc, weights=term, minlength=n_nodes)
    gini[has] = numer[has] / (cnt[has].astype(np.float64) * total[has])
    gini[has] = np.clip(gini[has], 0.0, 1.0)  # guard fp drift outside [0, 1]

    return {"gini": gini, "hhi": hhi, "max_counterparty_share": share}


def compute_flow_concentration(
    edge_index: npt.ArrayLike,
    weights: npt.ArrayLike,
    n_nodes: int,
    *,
    empty_value: float = 0.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute per-cluster in/out flow-concentration features.

    Args:
        edge_index: ``(2, E)`` array of int edge endpoints (row 0 = source idx,
            row 1 = target idx), e.g. the Stage 0 ``edge_index.npy`` memmap.
            ``E == 0`` (an empty ``(2, 0)`` array) is allowed.
        weights: ``(E,)`` non-negative edge weights (a volume/value edge feature),
            one per edge in ``edge_index`` column order.
        n_nodes: number of clusters; output arrays are sized ``n_nodes``.
        empty_value: value for a cluster with no edge (zero total flow) in a
            given direction; all three measures are otherwise undefined there.

    Returns:
        Dict keyed by :data:`COLUMNS` (``{direction}_{measure}``); each value is a
        float64 ``(n_nodes,)`` array.

    Raises:
        ValueError: if ``n_nodes`` is negative, ``edge_index`` is not 2 rows,
            ``weights`` length mismatches the edge count, any endpoint is out of
            ``[0, n_nodes)``, or any weight is negative.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")

    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")

    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != ei.shape[1]:
        raise ValueError(
            f"weights has {w.shape[0]} entries but edge_index has "
            f"{ei.shape[1]} edges"
        )
    if w.size and w.min() < 0:
        raise ValueError("weights must be non-negative for flow concentration")

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    if src.size and (src.min() < 0 or src.max() >= n_nodes
                     or dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"src [{src.min()}, {src.max()}], dst [{dst.min()}, {dst.max()}]"
        )

    # out-flow: cluster = source, counterparty = target; in-flow: the reverse.
    cluster = {"out": src, "in": dst}
    other = {"out": dst, "in": src}
    result: dict[str, npt.NDArray[np.float64]] = {}
    for direction in DIRECTIONS:
        aggs = _direction_concentration(
            cluster[direction], other[direction], w, n_nodes, empty_value
        )
        for measure in MEASURES:
            result[f"{direction}_{measure}"] = aggs[measure]
    return result
