"""Stage 1 — per-cluster degree features from the Stage 0 edge index.

plan.md §4 (Degree). The Stage 0 ``edge_index.npy`` is a ``(2, E)`` int32 COO
array in the remapped idx space: row 0 holds edge sources, row 1 holds edge
destinations (a directed background graph). From it we derive four per-cluster
columns, one row per node idx in ``[0, n_nodes)``:

    in_degree     number of incoming edges  (idx appears in row 1)
    out_degree    number of outgoing edges  (idx appears in row 0)
    total_degree  in_degree + out_degree
    in_out_ratio  in_degree / out_degree, with the zero-denominator case below

Counting is by edge multiplicity: parallel edges count once each, and a self
loop contributes one to both in_degree and out_degree. Isolated nodes (no
incident edge) get all-zero degrees.

The ratio is undefined when ``out_degree == 0``. We fill those entries with a
configurable ``zero_denom_value`` (default ``0.0``) rather than emitting NaN/inf,
so the assembled feature frame (T-007) stays finite. Note this is distinct from
a *genuine* ratio of ``0.0`` (a node with ``in_degree == 0`` but ``out_degree >
0``): the former is "ratio undefined", the latter is "no inflow".

Pure numpy ``bincount`` — no pandas, no DuckDB; the array is already materialized.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Stable column order for the assembled feature frame (T-007 joins on these).
COLUMNS: tuple[str, ...] = ("in_degree", "out_degree", "total_degree", "in_out_ratio")


def compute_degree_features(
    edge_index: npt.ArrayLike,
    n_nodes: int,
    *,
    zero_denom_value: float = 0.0,
) -> dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]]:
    """Compute per-cluster degree features.

    Args:
        edge_index: ``(2, E)`` array of int edge endpoints (row 0 = source idx,
            row 1 = destination idx), e.g. the Stage 0 ``edge_index.npy`` memmap.
            ``E == 0`` (an empty ``(2, 0)`` array) is allowed.
        n_nodes: number of clusters; output arrays are sized ``n_nodes`` and
            isolated nodes get degree 0.
        zero_denom_value: value assigned to ``in_out_ratio`` where
            ``out_degree == 0`` (the ratio is otherwise undefined).

    Returns:
        Dict keyed by :data:`COLUMNS`. ``in_degree``/``out_degree``/
        ``total_degree`` are int64 ``(n_nodes,)`` arrays; ``in_out_ratio`` is a
        float64 ``(n_nodes,)`` array.

    Raises:
        ValueError: if ``n_nodes`` is negative, ``edge_index`` is not 2 rows, or
            any endpoint falls outside ``[0, n_nodes)``.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")

    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    if src.size and (src.min() < 0 or src.max() >= n_nodes
                     or dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"src [{src.min()}, {src.max()}], dst [{dst.min()}, {dst.max()}]"
        )

    out_degree = np.bincount(src, minlength=n_nodes).astype(np.int64)
    in_degree = np.bincount(dst, minlength=n_nodes).astype(np.int64)
    total_degree = in_degree + out_degree

    in_out_ratio = np.full(n_nodes, float(zero_denom_value), dtype=np.float64)
    has_out = out_degree > 0
    in_out_ratio[has_out] = in_degree[has_out] / out_degree[has_out]

    return {
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": total_degree,
        "in_out_ratio": in_out_ratio,
    }
