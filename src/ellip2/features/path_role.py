"""Stage 1 — per-cluster source/sink (endpoint) heuristic.

plan.md Resolved decision #3 + §4 (Path-role). The Elliptic2 dataset ships **no**
node labels and **no** entity-type labels, so we cannot read off which clusters
are exchanges / licit endpoints. What we *can* do — and what Stage 3 consumes as
its endpoint set — is a **DERIVED, structural heuristic**: an exchange / licit
sink tends to have high in-degree, a large cluster size (#addresses), and high
throughput, while an illicit-side *source* tends to push value outward (out-degree
leaning) from a smaller footprint.

    ┌─────────────────────────────────────────────────────────────────────┐
    │ These scores are a HEURISTIC, NOT a ground-truth entity type. They    │
    │ rank clusters on a source-vs-sink / exchange-like axis; a high        │
    │ endpoint_score means "exchange-like by structure", which still needs  │
    │ corroboration (Stage 3) and human review before any claim.            │
    └─────────────────────────────────────────────────────────────────────┘

**Calibration is by percentile**, not by the raw (anonymized, unknown-unit)
magnitudes: each input array is converted to its in-population percentile rank in
``[0, 1]`` (average rank over ties), then the rank channels are combined. This
makes the heuristic scale-free and monotone — raising any constituent signal can
only raise (never lower) the score it feeds — which is the property Stage 3 relies
on and the tests assert.

Three per-cluster columns, one row per node idx in ``[0, n_nodes)``:

    endpoint_score    exchange / licit-sink likelihood, in ``[0, 1]``:
                      mean percentile rank of (in_degree, size, throughput).
                      High ⇒ a structural endpoint/exchange candidate.
    source_score      illicit-side source likelihood, in ``[0, 1]``:
                      mean percentile rank of (out_degree, size, throughput).
    source_sink_axis  signed lean in ``[-1, 1]``: pct(in_degree) − pct(out_degree).
                      ``+1`` fully sink-leaning (inflow), ``−1`` fully source-leaning.

Inputs are the already-materialized per-cluster signals (e.g. ``in_degree`` /
``out_degree`` from :mod:`ellip2.features.degree`, plus a size and a throughput
column); this module only ranks and combines them, so it stays pure-numpy and
trivially testable.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Stable column order for the assembled feature frame (T-007 joins on these).
COLUMNS: tuple[str, ...] = ("endpoint_score", "source_score", "source_sink_axis")


def _percentile_rank(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """In-population percentile rank in ``[0, 1]`` (average rank over ties).

    The smallest value maps to ``0.0``, the largest to ``1.0``; tied values share
    their mean rank. A single element maps to ``0.5`` (mid). Monotone
    non-decreasing in ``x``, so it is safe to combine additively into a heuristic.
    """
    n = x.size
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.full(1, 0.5, dtype=np.float64)

    # Mean 0-based ordinal rank for each tie group, mapped back to every element.
    _uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg_ordinal = starts + (counts - 1) / 2.0  # mean ordinal rank per group
    return (avg_ordinal[inv.ravel()] / (n - 1)).astype(np.float64)


def _as_channel(name: str, arr: npt.ArrayLike, n_nodes: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape[0] != n_nodes:
        raise ValueError(
            f"{name} has {a.shape[0]} entries but n_nodes is {n_nodes}"
        )
    return a


def compute_path_role(
    in_degree: npt.ArrayLike,
    out_degree: npt.ArrayLike,
    size: npt.ArrayLike,
    throughput: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64]]:
    """Compute the per-cluster source/sink (endpoint) **heuristic**.

    All four inputs are per-cluster arrays of the same length ``n_nodes``:

    Args:
        in_degree: incoming edge count per cluster (e.g. from
            :func:`ellip2.features.degree.compute_degree_features`).
        out_degree: outgoing edge count per cluster.
        size: cluster size — number of addresses in the cluster.
        throughput: total value/volume flowing through the cluster (e.g. summed
            in+out edge weight).

    Returns:
        Dict keyed by :data:`COLUMNS`; each value is a float64 ``(n_nodes,)``
        array. ``endpoint_score`` / ``source_score`` are in ``[0, 1]``;
        ``source_sink_axis`` is in ``[-1, 1]``. **These are derived heuristics,
        not ground-truth entity types.**

    Raises:
        ValueError: if the four inputs do not all share one length.
    """
    n_nodes = np.asarray(in_degree).ravel().shape[0]
    pid = _percentile_rank(_as_channel("in_degree", in_degree, n_nodes))
    pod = _percentile_rank(_as_channel("out_degree", out_degree, n_nodes))
    psz = _percentile_rank(_as_channel("size", size, n_nodes))
    pth = _percentile_rank(_as_channel("throughput", throughput, n_nodes))

    endpoint_score = (pid + psz + pth) / 3.0
    source_score = (pod + psz + pth) / 3.0
    source_sink_axis = pid - pod  # in [-1, 1]

    return {
        "endpoint_score": endpoint_score,
        "source_score": source_score,
        "source_sink_axis": source_sink_axis,
    }
