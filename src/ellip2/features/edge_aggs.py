"""Stage 1 — per-cluster edge-feature aggregates.

plan.md §4 (Edge-feature aggregates): for each (anonymized) edge feature column
we summarise it per cluster as ``sum`` / ``mean`` / ``max`` / ``std``, computed
*separately* for the cluster's in-edges and out-edges over the directed
background graph. Volume, fee and timestamp spread live among the 95 edge
features, so these aggregates fingerprint how value flows through a cluster.

The grouping convention mirrors :mod:`ellip2.features.degree`:

    out-edges of node ``v``  == edges with source  ``v``  (row 0 of edge_index)
    in-edges  of node ``v``  == edges with target  ``v``  (row 1 of edge_index)

Two entry points share one column convention so they are interchangeable and
join cleanly in T-007:

* :func:`compute_edge_aggregates` — in-memory numpy path. Takes ``edge_index``
  plus an ``(E, F)`` edge-feature array; pure ``bincount`` / ``maximum.at``, no
  pandas. Used by the unit test and by any caller that already has the edge
  features materialised.
* :func:`compute_edge_aggregates_duckdb` — out-of-core path. Streams a DuckDB
  ``GROUP BY`` over ``background_edges`` joined to ``id_map`` (the SIGN-104
  pattern from ``ingest.py``), so the 196M-edge / 95-feature table is never
  loaded into RAM. ``stddev_pop`` matches the *population* std used below.

Each output column is named ``{direction}_{feature}_{stat}`` (e.g.
``out_ef_0_sum``) and is a float64 ``(n_nodes,)`` array. Nodes with no edge in a
given direction take a configurable ``empty_value`` (default ``0.0``) rather than
NaN, so the assembled feature frame stays finite. ``std`` is the population
standard deviation, so a group of one edge has ``std == 0``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt

STATS: tuple[str, ...] = ("sum", "mean", "max", "std")
DIRECTIONS: tuple[str, ...] = ("in", "out")


def aggregate_columns(
    feature_names: list[str],
    *,
    directions: tuple[str, ...] = DIRECTIONS,
    stats: tuple[str, ...] = STATS,
) -> list[str]:
    """Deterministic ordered column names ``{direction}_{feature}_{stat}``."""
    return [
        f"{d}_{name}_{s}"
        for d in directions
        for name in feature_names
        for s in stats
    ]


def _resolve_feature_spec(
    n_feat_cols: int,
    feature_indices: list[int] | None,
    feature_names: list[str] | None,
) -> tuple[list[int], list[str]]:
    idx = list(range(n_feat_cols)) if feature_indices is None else list(feature_indices)
    for j in idx:
        if not 0 <= j < n_feat_cols:
            raise ValueError(
                f"feature index {j} out of range [0, {n_feat_cols})"
            )
    names = [f"ef_{j}" for j in idx] if feature_names is None else list(feature_names)
    if len(names) != len(idx):
        raise ValueError(
            f"feature_names has {len(names)} entries, expected {len(idx)}"
        )
    return idx, names


def _validate_stats(stats: tuple[str, ...]) -> None:
    unknown = [s for s in stats if s not in STATS]
    if unknown:
        raise ValueError(f"unknown stats {unknown}; supported: {STATS}")


def _direction_aggregates(
    key: npt.NDArray[np.int64],
    vals: npt.NDArray[np.float64],
    n_nodes: int,
    stats: tuple[str, ...],
    empty_value: float,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-node aggregates of ``vals`` grouped by endpoint ``key``."""
    count = np.bincount(key, minlength=n_nodes).astype(np.float64)
    has = count > 0

    out: dict[str, npt.NDArray[np.float64]] = {}
    total = np.bincount(key, weights=vals, minlength=n_nodes)

    if "sum" in stats:
        col = np.full(n_nodes, float(empty_value), dtype=np.float64)
        col[has] = total[has]
        out["sum"] = col

    mean = np.full(n_nodes, float(empty_value), dtype=np.float64)
    mean[has] = total[has] / count[has]
    if "mean" in stats:
        out["mean"] = mean.copy()

    if "max" in stats:
        col = np.full(n_nodes, -np.inf, dtype=np.float64)
        np.maximum.at(col, key, vals)
        col[~has] = float(empty_value)
        out["max"] = col

    if "std" in stats:
        sumsq = np.bincount(key, weights=vals * vals, minlength=n_nodes)
        col = np.full(n_nodes, float(empty_value), dtype=np.float64)
        var = sumsq[has] / count[has] - mean[has] ** 2
        col[has] = np.sqrt(np.maximum(var, 0.0))  # clip fp negatives
        out["std"] = col

    return out


def compute_edge_aggregates(
    edge_index: npt.ArrayLike,
    edge_features: npt.ArrayLike,
    n_nodes: int,
    *,
    feature_indices: list[int] | None = None,
    feature_names: list[str] | None = None,
    stats: tuple[str, ...] = STATS,
    empty_value: float = 0.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-cluster edge-feature aggregates from an in-memory edge-feature array.

    Args:
        edge_index: ``(2, E)`` array of int edge endpoints (row 0 = source idx,
            row 1 = target idx), e.g. the Stage 0 ``edge_index.npy`` memmap.
        edge_features: ``(E, F)`` array; row ``k`` holds the features of edge
            ``k`` (same column order as ``edge_index``). ``E == 0`` is allowed.
        n_nodes: number of clusters; output arrays are sized ``n_nodes``.
        feature_indices: which columns of ``edge_features`` to aggregate
            (default all). The 95 Elliptic2 edge features are anonymized, so the
            test passes a few explicit indices.
        feature_names: display names for the selected columns (default
            ``ef_{idx}``); must match ``feature_indices`` in length.
        stats: subset of :data:`STATS` to compute, in output order.
        empty_value: value for a node with no edge in a given direction.

    Returns:
        Dict keyed by ``{direction}_{feature}_{stat}`` (see
        :func:`aggregate_columns`); each value is a float64 ``(n_nodes,)`` array.

    Raises:
        ValueError: on bad shapes, out-of-range endpoints/feature indices, a
            ``feature_names`` length mismatch, or an unknown stat.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    _validate_stats(stats)

    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    ef = np.asarray(edge_features, dtype=np.float64)
    if ef.ndim != 2:
        raise ValueError(f"edge_features must be 2-D (E, F), got shape {ef.shape}")
    if ef.shape[0] != ei.shape[1]:
        raise ValueError(
            f"edge_features has {ef.shape[0]} rows but edge_index has "
            f"{ei.shape[1]} edges"
        )

    idx, names = _resolve_feature_spec(ef.shape[1], feature_indices, feature_names)

    src = ei[0].astype(np.int64, copy=False)
    dst = ei[1].astype(np.int64, copy=False)
    if src.size and (src.min() < 0 or src.max() >= n_nodes
                     or dst.min() < 0 or dst.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"src [{src.min()}, {src.max()}], dst [{dst.min()}, {dst.max()}]"
        )

    keys = {"in": dst, "out": src}
    result: dict[str, npt.NDArray[np.float64]] = {}
    for direction in DIRECTIONS:
        key = keys[direction]
        for j, name in zip(idx, names, strict=True):
            vals = np.ascontiguousarray(ef[:, j], dtype=np.float64)
            aggs = _direction_aggregates(key, vals, n_nodes, stats, empty_value)
            for stat in stats:
                result[f"{direction}_{name}_{stat}"] = aggs[stat]
    return result


# --------------------------------------------------------------------------- #
# Out-of-core DuckDB path (SIGN-104) — streams background_edges via id_map.


def _q(path: Path) -> str:
    """Single-quote a path for inlining into DuckDB SQL."""
    return "'" + str(path).replace("'", "''") + "'"


def _qi(name: str) -> str:
    """Double-quote an identifier for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


# DuckDB aggregate function per stat; stddev_pop matches the population std above.
_DUCK_AGG = {"sum": "sum", "mean": "avg", "max": "max", "std": "stddev_pop"}


def compute_edge_aggregates_duckdb(
    edges_csv: Path,
    id_map_parquet: Path,
    n_nodes: int,
    *,
    feature_indices: list[int] | None = None,
    feature_names: list[str] | None = None,
    stats: tuple[str, ...] = STATS,
    empty_value: float = 0.0,
    src_col: str = "clId1",
    dst_col: str = "clId2",
    threads: int | None = None,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-cluster edge-feature aggregates streamed out-of-core via DuckDB.

    Reads ``background_edges`` (CSV) and the Stage 0 ``id_map.parquet``
    (columns ``idx``, ``orig_id``), remaps each endpoint through a join, and runs
    one ``GROUP BY`` per direction so the 196M-edge table never lands in pandas.
    Results match :func:`compute_edge_aggregates` (population std).

    Args mirror :func:`compute_edge_aggregates`. ``feature_indices`` index into
    the edge-feature columns, i.e. the CSV columns *after* ``src_col``/``dst_col``
    (default first ``len`` features when names are given, else all). ``threads`` /
    ``memory_limit`` / ``temp_dir`` tune DuckDB the same way ``ingest.py`` does.
    """
    import duckdb

    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    _validate_stats(stats)

    con = duckdb.connect()
    try:
        if threads is not None:
            con.execute(f"SET threads TO {int(threads)}")
        if memory_limit is not None:
            con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_dir is not None:
            Path(temp_dir).mkdir(parents=True, exist_ok=True)
            con.execute(f"SET temp_directory = {_q(Path(temp_dir))}")
        con.execute("SET preserve_insertion_order = false")

        all_cols = [
            r[0]
            for r in con.execute(
                f"DESCRIBE SELECT * FROM read_csv_auto({_q(Path(edges_csv))})"
            ).fetchall()
        ]
        feat_cols = [c for c in all_cols if c not in (src_col, dst_col)]
        idx, names = _resolve_feature_spec(
            len(feat_cols), feature_indices, feature_names
        )
        sel_cols = [feat_cols[j] for j in idx]

        result: dict[str, npt.NDArray[np.float64]] = {}
        # endpoint join column per direction: out groups by source, in by target.
        join_col = {"out": src_col, "in": dst_col}
        for direction in DIRECTIONS:
            select_parts = []
            for name, col in zip(names, sel_cols, strict=True):
                for stat in stats:
                    select_parts.append(
                        f"{_DUCK_AGG[stat]}(e.{_qi(col)}) "
                        f"AS {_qi(f'{direction}_{name}_{stat}')}"
                    )
            sql = (
                f"SELECT m.idx AS node, {', '.join(select_parts)} "
                f"FROM read_csv_auto({_q(Path(edges_csv))}) e "
                f"JOIN read_parquet({_q(Path(id_map_parquet))}) m "
                f"ON e.{_qi(join_col[direction])} = m.orig_id "
                "GROUP BY m.idx"
            )
            cols = aggregate_columns(names, directions=(direction,), stats=stats)
            for c in cols:
                result[c] = np.full(n_nodes, float(empty_value), dtype=np.float64)

            reader = con.execute(sql).to_arrow_reader(100_000)
            for batch in reader:
                node = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64)
                for ci, c in enumerate(cols, start=1):
                    vals = batch.column(ci).to_numpy(zero_copy_only=False)
                    result[c][node] = vals.astype(np.float64, copy=False)
        return result
    finally:
        con.close()
