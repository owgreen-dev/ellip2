"""Stage 1 — per-cluster temporal features from edge timestamps.

plan.md §4 (Temporal): "activity span, burstiness within the 1-year construction
window". Each background edge carries a (binned, for IP reasons) timestamp. We
summarise *when* a cluster is active over that window with two per-cluster
columns, one row per node idx in ``[0, n_nodes)``:

    activity_span   t_max − t_min over the cluster's incident edges — how long
                    the cluster stays active (0 for a single instantaneous event)
    burstiness      Goh & Barabási burstiness ``B = (σ − μ) / (σ + μ)`` of the
                    inter-event times, where μ and σ are the mean and
                    (population) standard deviation of the gaps between
                    consecutively-timed incident events

**Events = incident edges.** A cluster's activity is every edge that touches it,
regardless of direction, so an edge contributes one event to its source and one
to its destination (a self-loop therefore contributes two events at the same
timestamp). This mirrors ``total_degree`` in :mod:`ellip2.features.degree` as the
incident-event count.

**Burstiness.** ``B`` ranges in ``[−1, 1]`` (Goh & Barabási 2008): ``B = −1`` is a
perfectly regular/periodic train (all gaps equal, σ = 0), ``B = 0`` is
Poisson/random, and ``B → 1`` is bursty (a few long gaps dominate). It needs at
least one inter-event gap, i.e. at least two events; with exactly two events the
single gap has σ = 0 so ``B = −1``. The standard deviation is the *population*
std (matches :mod:`ellip2.features.edge_aggs`), so a node's burstiness depends
only on its own gaps.

A cluster with no incident edge has no defined span or burstiness; so does the
burstiness of a cluster with fewer than two events, or one whose every event
shares a single timestamp (all gaps zero ⇒ ``σ + μ = 0``). Those entries take a
configurable ``empty_value`` (default ``0.0``) rather than NaN, so the assembled
feature frame (T-007) stays finite. ``empty_value`` for ``activity_span`` only
fires for a *zero-event* cluster; a single-event cluster has a genuine span of
``0.0`` (active at one instant), which a non-default ``empty_value`` keeps
distinct.

Pure numpy (``bincount`` / ``lexsort`` segment reductions, ``minimum.at`` /
``maximum.at``) — no pandas, no per-cluster Python loop; the arrays are already
materialized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Stable column order for the assembled feature frame (T-007 joins on these).
COLUMNS: tuple[str, ...] = ("activity_span", "burstiness")


def compute_temporal_features(
    edge_index: npt.ArrayLike,
    timestamps: npt.ArrayLike,
    n_nodes: int,
    *,
    empty_value: float = 0.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute per-cluster activity span and burstiness from edge timestamps.

    Args:
        edge_index: ``(2, E)`` array of int edge endpoints (row 0 = source idx,
            row 1 = destination idx), e.g. the Stage 0 ``edge_index.npy`` memmap.
            ``E == 0`` (an empty ``(2, 0)`` array) is allowed.
        timestamps: ``(E,)`` per-edge timestamps (binned ordinals or any real),
            one per edge in ``edge_index`` column order.
        n_nodes: number of clusters; output arrays are sized ``n_nodes``.
        empty_value: value for an undefined entry — a zero-event cluster's span,
            and any cluster's burstiness when it has fewer than two events or all
            its events share one timestamp.

    Returns:
        Dict keyed by :data:`COLUMNS` (``activity_span``, ``burstiness``); each
        value is a float64 ``(n_nodes,)`` array.

    Raises:
        ValueError: if ``n_nodes`` is negative, ``edge_index`` is not 2 rows,
            ``timestamps`` length mismatches the edge count, or any endpoint is
            out of ``[0, n_nodes)``.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")

    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")

    ts = np.asarray(timestamps, dtype=np.float64).ravel()
    if ts.shape[0] != ei.shape[1]:
        raise ValueError(
            f"timestamps has {ts.shape[0]} entries but edge_index has "
            f"{ei.shape[1]} edges"
        )

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    if src.size and (src.min() < 0 or src.max() >= n_nodes
                     or dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"src [{src.min()}, {src.max()}], dst [{dst.min()}, {dst.max()}]"
        )

    span = np.full(n_nodes, float(empty_value), dtype=np.float64)
    burst = np.full(n_nodes, float(empty_value), dtype=np.float64)

    # Each edge is an incident event for BOTH endpoints (direction-agnostic).
    node = np.concatenate([src, dst])
    event_ts = np.concatenate([ts, ts])
    if node.size == 0:
        return {"activity_span": span, "burstiness": burst}

    # Activity span: per-node max − min of incident timestamps.
    count = np.bincount(node, minlength=n_nodes)
    has = count > 0
    mins = np.full(n_nodes, np.inf, dtype=np.float64)
    maxs = np.full(n_nodes, -np.inf, dtype=np.float64)
    np.minimum.at(mins, node, event_ts)
    np.maximum.at(maxs, node, event_ts)
    span[has] = maxs[has] - mins[has]

    # Inter-event gaps: sort events by (node, timestamp); a gap between two
    # consecutive same-node events belongs to that node and is ≥ 0.
    order = np.lexsort((event_ts, node))  # primary: node asc; secondary: ts asc
    sn = node[order]
    st = event_ts[order]
    same = sn[1:] == sn[:-1]
    gap_node = sn[1:][same]
    gap_val = (st[1:] - st[:-1])[same]

    if gap_node.size:
        gcount = np.bincount(gap_node, minlength=n_nodes).astype(np.float64)
        gsum = np.bincount(gap_node, weights=gap_val, minlength=n_nodes)
        gsumsq = np.bincount(gap_node, weights=gap_val * gap_val, minlength=n_nodes)
        hg = gcount > 0
        mu = np.zeros(n_nodes, dtype=np.float64)
        var = np.zeros(n_nodes, dtype=np.float64)
        mu[hg] = gsum[hg] / gcount[hg]
        var[hg] = gsumsq[hg] / gcount[hg] - mu[hg] ** 2
        sigma = np.sqrt(np.clip(var, 0.0, None))  # clip fp drift below 0
        denom = sigma + mu
        valid = hg & (denom > 0)  # all-zero gaps ⇒ denom 0 ⇒ undefined
        burst[valid] = (sigma[valid] - mu[valid]) / denom[valid]

    return {"activity_span": span, "burstiness": burst}
