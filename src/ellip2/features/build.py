"""Stage 1 assembly — join every per-cluster feature table into one frame.

plan.md §9 (Stage 1): "DuckDB group-bys keyed on clId1/clId2 produce degree,
in/out ratio, weight-Gini, HHI, edge-feature aggregates; one shallow propagation
pass yields 1-2 hop neighborhood label fractions (leakage-masked). Output
cluster_features.parquet."

This module wires the six feature builders (T-001..T-006) over the Stage 0
artifacts and writes a single ``cluster_features.parquet`` with exactly one row
per cluster idx in ``[0, n_nodes)`` (parquet row order == idx):

    degree              in/out/total degree + in_out_ratio        (degree.py)
    edge aggregates     sum/mean/max/std of edge features, in/out  (edge_aggs.py)
    flow concentration  in/out Gini / HHI / max-counterparty share (flow_concentration.py)
    neighborhood        leakage-masked 1/2-hop label fractions     (neighborhood.py)
    temporal            activity span + burstiness                 (temporal.py)
    path role           source/sink (endpoint) heuristic           (path_role.py)

Data sources & out-of-core posture (SIGN-104):

* Topology-only features (degree, neighborhood) read the canonical
  ``edge_index.npy`` memmap — no CSV pass, no edge features needed.
* Edge-feature aggregates stream out-of-core via
  :func:`edge_aggs.compute_edge_aggregates_duckdb` (one ``GROUP BY`` per
  direction over ``background_edges.csv`` joined to ``id_map.parquet``); the
  95-feature table never lands in RAM.
* Flow concentration and temporal features are pure-numpy and need per-edge
  arrays, so the *single* weight column and *single* timestamp column they use
  are pulled — aligned with their endpoints, in one DuckDB join — into memory.
  Only those two columns (not all 95) are materialised.

Because every feature builder is permutation-invariant in edge order, the
DuckDB-rebuilt endpoint arrays (which carry the pulled weight/timestamp columns)
and the canonical ``edge_index.npy`` describe the same edge set and agree.

The 43 node features and 95 edge features are anonymized binned ordinals
(plan.md §1): which column is volume/weight, timestamp, or cluster size
(#addresses) is not published and must be chosen by inspecting distributions on
real data. So every such column is a configurable index on
:class:`FeatureBuildConfig`, defaulting to ``0`` for the smoke path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..data import schema
from . import degree, edge_aggs, flow_concentration, neighborhood, path_role, temporal

# Fixed-name feature columns guaranteed present (config-independent), asserted
# NaN-free by the builder. Edge-aggregate columns depend on the chosen indices
# and are validated for finiteness too, but are not listed here.
REQUIRED_COLUMNS: tuple[str, ...] = (
    *degree.COLUMNS,
    *flow_concentration.COLUMNS,
    *neighborhood.COLUMNS,
    *temporal.COLUMNS,
    *path_role.COLUMNS,
)

DEFAULT_OUTPUT_NAME = "cluster_features.parquet"


@dataclass(frozen=True)
class FeatureBuildConfig:
    """Inputs and column choices for :func:`build_cluster_features`.

    Attributes:
        artifacts_dir: Stage 0 output dir (``edge_index.npy``,
            ``node_features.npy``, ``id_map.parquet``, ``subgraphs.parquet``,
            ``ingest_manifest.json``).
        raw_dir: dataset root holding ``background_edges.csv`` (the edge
            features live only in the raw CSV; Stage 0 does not materialise them).
        split_csv: persisted ``split.csv`` from :mod:`ellip2.eval.splits`;
            consumed so the neighborhood test-label mask matches the model split
            (SIGN-103).
        out_path: destination parquet (default ``artifacts_dir/cluster_features.parquet``).
        weight_index: edge-feature column used as flow weight / transaction
            volume (indexes the columns *after* ``clId1``/``clId2``).
        timestamp_index: edge-feature column used as the (binned) timestamp.
        size_index: node-feature column used as cluster size (#addresses) for the
            path-role heuristic.
        edge_agg_indices: edge-feature columns to aggregate (sum/mean/max/std).
        edge_agg_names: display names for ``edge_agg_indices`` (default ``ef_{i}``).
        hops: neighborhood hop distances (default ``(1, 2)``).
        positive_label: ccLabel string mapped to the suspicious class.
        empty_value: fill for undefined entries (no edge / empty neighborhood).
    """

    artifacts_dir: Path
    raw_dir: Path
    split_csv: Path
    out_path: Path | None = None
    weight_index: int = 0
    timestamp_index: int = 0
    size_index: int = 0
    edge_agg_indices: tuple[int, ...] = (0, 1, 2)
    edge_agg_names: tuple[str, ...] | None = None
    hops: tuple[int, ...] = (1, 2)
    positive_label: str = "suspicious"
    empty_value: float = 0.0
    # DuckDB tuning for the out-of-core edge-aggregate GROUP BY (T-002). Unbounded
    # by default DuckDB grabs ~80% of RAM and, on the 77GB edge CSV, thrashes a
    # box with no swap to a standstill — set these like ingest.py does.
    duckdb_memory_limit: str | None = None
    duckdb_temp_dir: Path | None = None
    duckdb_threads: int | None = None


@dataclass(frozen=True)
class BuildResult:
    """Outcome of an assembly run.

    Attributes:
        out_path: parquet written.
        n_nodes: rows written (one per cluster idx).
        columns: ordered feature column names (excludes the ``idx`` key column).
    """

    out_path: Path
    n_nodes: int
    columns: list[str] = field(default_factory=list)


def _q(path: Path) -> str:
    """Single-quote a path for inlining into DuckDB SQL."""
    return "'" + str(path).replace("'", "''") + "'"


def _qi(name: str) -> str:
    """Double-quote an identifier for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


def _resolve_n_nodes(artifacts_dir: Path) -> int:
    """Cluster count from the ingest manifest, falling back to the feature shape."""
    manifest = artifacts_dir / "ingest_manifest.json"
    if manifest.is_file():
        n = json.loads(manifest.read_text()).get("n_nodes")
        if isinstance(n, int):
            return n
    feats = np.load(artifacts_dir / "node_features.npy", mmap_mode="r")
    return int(feats.shape[0])


def _pull_edge_columns(
    edges_csv: Path,
    id_map_parquet: Path,
    col_indices: Sequence[int],
) -> tuple[npt.NDArray[np.int64], dict[int, npt.NDArray[np.float64]]]:
    """Remapped endpoints plus selected edge-feature columns, in one DuckDB join.

    Returns a ``(2, E)`` int64 edge_index and a dict mapping each requested
    edge-feature index to its ``(E,)`` float64 column. Endpoints and columns are
    aligned by construction (same SELECT), so the pure-numpy flow/temporal
    builders see consistent edge/feature pairs. Only the requested columns are
    materialised (SIGN-104: the full 95-feature table is never loaded).
    """
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    try:
        con.execute("SET preserve_insertion_order = false")
        all_cols = [
            r[0]
            for r in con.execute(
                f"DESCRIBE SELECT * FROM read_csv_auto({_q(edges_csv)})"
            ).fetchall()
        ]
        src = schema.resolve_column(all_cols, schema.COL_EDGE_SRC, 0)
        dst = schema.resolve_column(all_cols, schema.COL_EDGE_DST, 1)
        feat_cols = [c for c in all_cols if c not in (src, dst)]
        for j in col_indices:
            if not 0 <= j < len(feat_cols):
                raise ValueError(
                    f"edge-feature index {j} out of range [0, {len(feat_cols)})"
                )

        sel = ", ".join(
            f"e.{_qi(feat_cols[j])} AS {_qi(f'f{j}')}" for j in col_indices
        )
        sql = (
            f"SELECT a.idx AS s, b.idx AS d{', ' + sel if sel else ''} "
            f"FROM read_csv_auto({_q(edges_csv)}) e "
            f"JOIN read_parquet({_q(id_map_parquet)}) a ON e.{_qi(src)} = a.orig_id "
            f"JOIN read_parquet({_q(id_map_parquet)}) b ON e.{_qi(dst)} = b.orig_id"
        )

        s_parts: list[npt.NDArray[np.int64]] = []
        d_parts: list[npt.NDArray[np.int64]] = []
        f_parts: dict[int, list[npt.NDArray[np.float64]]] = {j: [] for j in col_indices}
        reader = con.execute(sql).to_arrow_reader(100_000)
        for batch in reader:
            s_parts.append(batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64))
            d_parts.append(batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64))
            for ci, j in enumerate(col_indices, start=2):
                f_parts[j].append(
                    batch.column(ci).to_numpy(zero_copy_only=False).astype(np.float64)
                )

        s = np.concatenate(s_parts) if s_parts else np.empty(0, dtype=np.int64)
        d = np.concatenate(d_parts) if d_parts else np.empty(0, dtype=np.int64)
        cols = {
            j: (np.concatenate(p) if p else np.empty(0, dtype=np.float64))
            for j, p in f_parts.items()
        }
        return np.vstack([s, d]), cols
    finally:
        con.close()


def build_cluster_features(cfg: FeatureBuildConfig) -> BuildResult:
    """Assemble every Stage 1 feature table and write ``cluster_features.parquet``.

    Args:
        cfg: paths and per-feature column choices.

    Returns:
        :class:`BuildResult` with the written path, row count, and column order.

    Raises:
        FileNotFoundError: if a required Stage 0 artifact is missing.
        ValueError: on a NaN/inf in any assembled feature column (assembly bug),
            or a bad column index.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    artifacts = Path(cfg.artifacts_dir)
    n_nodes = _resolve_n_nodes(artifacts)

    edge_index = np.load(artifacts / "edge_index.npy", mmap_mode="r")
    node_features = np.load(artifacts / "node_features.npy", mmap_mode="r")
    id_map_parquet = artifacts / "id_map.parquet"
    subgraphs_parquet = artifacts / "subgraphs.parquet"
    edges_csv = Path(cfg.raw_dir) / schema.F_BACKGROUND_EDGES

    columns: dict[str, npt.NDArray[np.float64] | npt.NDArray[np.int64]] = {}

    # --- T-001 degree (topology only) ------------------------------------- #
    deg = degree.compute_degree_features(edge_index, n_nodes)
    columns.update(deg)

    # --- T-004 neighborhood (topology + leakage-masked labels) ------------ #
    labels = neighborhood.load_subgraph_labels(
        subgraphs_parquet, cfg.split_csv, n_nodes,
        positive_label=cfg.positive_label,
    )
    columns.update(
        neighborhood.compute_neighborhood_features(
            edge_index,
            labels.node_subgraph,
            labels.subgraph_label,
            labels.subgraph_in_test,
            n_nodes,
            hops=cfg.hops,
            empty_value=cfg.empty_value,
        )
    )

    # --- Pull the single weight + timestamp columns for the numpy builders. #
    pull_indices = sorted({cfg.weight_index, cfg.timestamp_index})
    rebuilt_ei, edge_cols = _pull_edge_columns(edges_csv, id_map_parquet, pull_indices)
    weights = edge_cols[cfg.weight_index]
    timestamps = edge_cols[cfg.timestamp_index]

    # --- T-003 flow concentration ----------------------------------------- #
    columns.update(
        flow_concentration.compute_flow_concentration(
            rebuilt_ei, weights, n_nodes, empty_value=cfg.empty_value
        )
    )

    # --- T-005 temporal --------------------------------------------------- #
    columns.update(
        temporal.compute_temporal_features(
            rebuilt_ei, timestamps, n_nodes, empty_value=cfg.empty_value
        )
    )

    # --- T-002 edge-feature aggregates (out-of-core GROUP BY) ------------- #
    agg_names = (
        list(cfg.edge_agg_names) if cfg.edge_agg_names is not None else None
    )
    columns.update(
        edge_aggs.compute_edge_aggregates_duckdb(
            edges_csv,
            id_map_parquet,
            n_nodes,
            feature_indices=list(cfg.edge_agg_indices),
            feature_names=agg_names,
            empty_value=cfg.empty_value,
            threads=cfg.duckdb_threads,
            memory_limit=cfg.duckdb_memory_limit,
            temp_dir=cfg.duckdb_temp_dir,
        )
    )

    # --- T-006 path-role heuristic (derived; needs degree + size + flow) -- #
    size = np.asarray(node_features[:, cfg.size_index], dtype=np.float64)
    throughput = (
        np.bincount(rebuilt_ei[0], weights=weights, minlength=n_nodes)
        + np.bincount(rebuilt_ei[1], weights=weights, minlength=n_nodes)
    )
    columns.update(
        path_role.compute_path_role(
            deg["in_degree"], deg["out_degree"], size, throughput
        )
    )

    # --- Finiteness guard (no NaN/inf may reach the model) ---------------- #
    for name, arr in columns.items():
        a = np.asarray(arr, dtype=np.float64)
        if not np.all(np.isfinite(a)):
            raise ValueError(f"non-finite values in assembled column {name!r}")

    # --- Write one row per cluster idx ------------------------------------ #
    feature_cols = list(columns)
    table = pa.table(
        {"idx": np.arange(n_nodes, dtype=np.int64), **columns}
    )
    out_path = (
        Path(cfg.out_path)
        if cfg.out_path is not None
        else artifacts / DEFAULT_OUTPUT_NAME
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out_path)

    return BuildResult(out_path=out_path, n_nodes=n_nodes, columns=feature_cols)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: build cluster_features.parquet over a Stage 0 artifacts dir."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(description="Assemble Stage 1 cluster_features.parquet")
    p.add_argument("--artifacts-dir", required=True, type=Path,
                   help="Stage 0 output dir (edge_index.npy, etc.)")
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="dataset root holding background_edges.csv")
    p.add_argument("--split-csv", required=True, type=Path,
                   help="persisted split.csv from ellip2.eval.splits")
    p.add_argument("--out", type=Path, default=None,
                   help="output parquet (default <artifacts-dir>/cluster_features.parquet)")
    p.add_argument("--weight-index", type=int, default=0)
    p.add_argument("--timestamp-index", type=int, default=0)
    p.add_argument("--size-index", type=int, default=0)
    p.add_argument("--edge-agg-indices", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--memory-limit", default=None,
                   help="DuckDB memory cap for the edge-aggregate GROUP BY, e.g. '40GB' "
                        "(unbounded default thrashes a no-swap box on the 77GB edge CSV)")
    p.add_argument("--temp-dir", type=Path, default=None,
                   help="DuckDB spill directory (put it on a big disk, e.g. the data volume)")
    p.add_argument("--threads", type=int, default=None, help="DuckDB thread count")
    args = p.parse_args(argv)

    res = build_cluster_features(
        FeatureBuildConfig(
            artifacts_dir=args.artifacts_dir,
            raw_dir=args.raw_dir,
            split_csv=args.split_csv,
            out_path=args.out,
            weight_index=args.weight_index,
            timestamp_index=args.timestamp_index,
            size_index=args.size_index,
            edge_agg_indices=tuple(args.edge_agg_indices),
            duckdb_memory_limit=args.memory_limit,
            duckdb_temp_dir=args.temp_dir,
            duckdb_threads=args.threads,
        )
    )
    print(
        f"[build_features] wrote {res.out_path} "
        f"({res.n_nodes} rows, {len(res.columns)} feature columns)"
    )
    return 0
