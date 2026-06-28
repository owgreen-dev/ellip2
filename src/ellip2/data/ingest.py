"""Stage 0 — Ingest the five Elliptic2 CSVs into model-ready artifacts.

Builds, out-of-core via DuckDB (CPU-only, no GPU, spills to disk):

  id_map.parquet      orig_id -> contiguous int32 idx in [0, N)   (the bijection
                      every other artifact is remapped through)
  node_features.npy   (N, 43) float array, row i = features of cluster idx i
  edge_index.npy      (2, E) int32, COO edge list in remapped idx space (PyG-ready)
  subgraphs.parquet   ccId, ccLabel, n_members, member_idx[]  (subgraph membership)
  ingest_manifest.json counts, shapes, dtypes, integrity results, timings

Design choices tied to plan.md:
* The 49M-node / 196M-edge graph never loads into pandas. DuckDB streams and
  hash-joins out-of-core; the binding constraint is disk, not the 16 GiB g5.xlarge
  RAM. Tune with memory_limit / temp_dir / threads.
* node_features and edge_index are written through numpy memmaps (open_memmap), so
  the full arrays are never resident in RAM during construction.
* The integer remap is a single bijection (id_map); edges and membership are
  remapped through a JOIN against it, so feature rows, edges, and membership are
  guaranteed consistent regardless of DuckDB's scan order.
* Counts are validated against Table 1; integrity checks catch dangling edge
  endpoints and orphan subgraph members. Warn-only unless strict=True.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

from ellip2.data import schema
from ellip2.data.schema import Elliptic2Paths, resolve_column

_BATCH = 1_000_000  # rows per Arrow record batch when scattering to memmap


@dataclass
class IngestConfig:
    raw_dir: Path
    out_dir: Path
    feature_dtype: str = "float32"   # "float16" halves disk (plan suggests it)
    threads: int | None = None       # None -> DuckDB default (n cores)
    memory_limit: str | None = None  # e.g. "12GB"; None -> DuckDB default
    temp_dir: Path | None = None     # DuckDB spill dir; defaults to out_dir/_duckdb_tmp
    strict: bool = False             # hard-fail on count/integrity mismatch
    validate_counts: bool = True     # compare against Table 1
    # Override expected counts (set to None to skip a particular check).
    expected_nodes: int | None = schema.N_NODES
    expected_edges: int | None = schema.N_EDGES
    expected_subgraphs: int | None = schema.N_SUBGRAPHS


@dataclass
class IngestResult:
    out_dir: Path
    manifest: dict = field(default_factory=dict)


class IngestError(RuntimeError):
    pass


def _q(path: Path) -> str:
    """Single-quote a path for inlining into DuckDB SQL."""
    return "'" + str(path).replace("'", "''") + "'"


def _qi(name: str) -> str:
    """Double-quote an identifier (column name) for DuckDB SQL."""
    return '"' + name.replace('"', '""') + '"'


def _warn(report: list[str], msg: str, strict: bool) -> None:
    line = f"[ingest] {'ERROR' if strict else 'WARNING'}: {msg}"
    print(line)
    report.append(msg)
    if strict:
        raise IngestError(msg)


def _columns(con: duckdb.DuckDBPyConnection, path: Path) -> list[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_csv_auto({_q(path)})").fetchall()
    return [r[0] for r in rows]


def _count_csv(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    row = con.execute(f"SELECT count(*) FROM read_csv_auto({_q(path)})").fetchone()
    assert row is not None
    return int(row[0])


def _configure(con: duckdb.DuckDBPyConnection, cfg: IngestConfig) -> Path:
    if cfg.threads is not None:
        con.execute(f"SET threads TO {int(cfg.threads)}")
    if cfg.memory_limit is not None:
        con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    tmp = cfg.temp_dir or (Path(cfg.out_dir) / "_duckdb_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = {_q(tmp)}")
    # Lower peak memory for large streaming queries.
    con.execute("SET preserve_insertion_order = false")
    return tmp


def _build_id_map(con, paths: Elliptic2Paths, report, strict) -> tuple[str, list[str]]:
    """Materialize id_map(orig_id, idx) from background_nodes; return (id_col, cols)."""
    cols = _columns(con, paths.background_nodes)
    id_col = cols[0]
    feat_cols = cols[1:]
    if len(feat_cols) != schema.N_NODE_FEATURES:
        _warn(report,
              f"background_nodes has {len(feat_cols)} feature columns, "
              f"expected {schema.N_NODE_FEATURES}", strict)
    con.execute(
        f"""CREATE TABLE id_map AS
            SELECT CAST(row_number() OVER () - 1 AS BIGINT) AS idx,
                   {_qi(id_col)} AS orig_id
            FROM read_csv_auto({_q(paths.background_nodes)})"""
    )
    # Bijection sanity: idx is 0..N-1 by construction; ensure orig_id is unique.
    n, n_distinct = con.execute(
        "SELECT count(*), count(DISTINCT orig_id) FROM id_map"
    ).fetchone()
    if n != n_distinct:
        _warn(report,
              f"background_nodes has duplicate ids: {n - n_distinct} dupes", strict)
    return id_col, feat_cols


def _write_node_features(con, paths, id_col, feat_cols, n_nodes, cfg) -> dict:
    out = Path(cfg.out_dir) / "node_features.npy"
    dtype = np.dtype(cfg.feature_dtype)
    feats = np.lib.format.open_memmap(
        out, mode="w+", dtype=dtype, shape=(n_nodes, len(feat_cols))
    )
    select = ", ".join(f"n.{_qi(c)}" for c in feat_cols)
    # Join background_nodes back to id_map on the id column so each feature row
    # carries its remapped idx; scatter by idx, so DuckDB scan order is irrelevant.
    sql = (
        f"SELECT m.idx AS idx, {select} "
        f"FROM read_csv_auto({_q(paths.background_nodes)}) AS n "
        f"JOIN id_map m ON n.{_qi(id_col)} = m.orig_id"
    )
    written = 0
    reader = con.execute(sql).to_arrow_reader(_BATCH)
    for batch in reader:
        idx = batch.column(0).to_numpy(zero_copy_only=False)
        mat = np.column_stack([
            batch.column(j).to_numpy(zero_copy_only=False)
            for j in range(1, batch.num_columns)
        ]).astype(dtype, copy=False)
        feats[idx] = mat
        written += idx.shape[0]
    feats.flush()
    del feats
    return {"path": out.name, "shape": [n_nodes, len(feat_cols)],
            "dtype": str(dtype), "rows_written": int(written)}


def _write_edge_index(con, paths, out_dir, report, strict) -> dict:
    cols = _columns(con, paths.background_edges)
    src = resolve_column(cols, schema.COL_EDGE_SRC, 0)
    dst = resolve_column(cols, schema.COL_EDGE_DST, 1)
    join = (
        f"SELECT a.idx AS s, b.idx AS d "
        f"FROM read_csv_auto({_q(paths.background_edges)}) e "
        f"JOIN id_map a ON e.{_qi(src)} = a.orig_id "
        f"JOIN id_map b ON e.{_qi(dst)} = b.orig_id"
    )
    e_total = _count_csv(con, paths.background_edges)
    e_join = con.execute(f"SELECT count(*) FROM ({join})").fetchone()[0]
    dangling = e_total - e_join
    if dangling:
        _warn(report,
              f"{dangling} background edges reference ids absent from "
              "background_nodes (dropped from edge_index)", strict)

    out_path = Path(out_dir) / "edge_index.npy"
    ei = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.int32,
                                   shape=(2, e_join))
    off = 0
    reader = con.execute(join).to_arrow_reader(_BATCH)
    for batch in reader:
        s = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        d = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
        n = s.shape[0]
        ei[0, off:off + n] = s
        ei[1, off:off + n] = d
        off += n
    ei.flush()
    del ei
    return {"path": out_path.name, "shape": [2, int(e_join)], "dtype": "int32",
            "edges_total": int(e_total), "edges_written": int(e_join),
            "dangling": int(dangling), "src_col": src, "dst_col": dst}


def _write_subgraphs(con, paths, cfg, report, strict) -> dict:
    n_cols = _columns(con, paths.nodes)
    node_id_col = n_cols[0]
    node_cc_col = resolve_column(n_cols, schema.COL_CC_ID, 1)
    cc_cols = _columns(con, paths.connected_components)
    cc_id_col = resolve_column(cc_cols, schema.COL_CC_ID, 0)
    cc_label_col = resolve_column(cc_cols, schema.COL_CC_LABEL, 1)

    out_path = Path(cfg.out_dir) / "subgraphs.parquet"
    con.execute(
        f"""COPY (
            SELECT cc.{_qi(cc_id_col)} AS ccId,
                   cc.{_qi(cc_label_col)} AS ccLabel,
                   count(*) AS n_members,
                   list(m.idx) AS member_idx
            FROM read_csv_auto({_q(paths.nodes)}) nm
            JOIN id_map m ON nm.{_qi(node_id_col)} = m.orig_id
            JOIN read_csv_auto({_q(paths.connected_components)}) cc
                 ON nm.{_qi(node_cc_col)} = cc.{_qi(cc_id_col)}
            GROUP BY cc.{_qi(cc_id_col)}, cc.{_qi(cc_label_col)}
        ) TO {_q(out_path)} (FORMAT PARQUET)"""
    )

    # Integrity: orphan members (id not in background) / orphan cc references.
    nm_total = _count_csv(con, paths.nodes)
    nm_mapped = con.execute(
        f"""SELECT count(*) FROM read_csv_auto({_q(paths.nodes)}) nm
            JOIN id_map m ON nm.{_qi(node_id_col)} = m.orig_id"""
    ).fetchone()[0]
    nm_in_cc = con.execute(
        f"""SELECT count(*) FROM read_csv_auto({_q(paths.nodes)}) nm
            JOIN read_csv_auto({_q(paths.connected_components)}) cc
                 ON nm.{_qi(node_cc_col)} = cc.{_qi(cc_id_col)}"""
    ).fetchone()[0]
    orphan_nodes = nm_total - nm_mapped
    orphan_cc = nm_total - nm_in_cc
    if orphan_nodes:
        _warn(report, f"{orphan_nodes} subgraph members not in background_nodes",
              strict)
    if orphan_cc:
        _warn(report, f"{orphan_cc} subgraph members reference unknown ccId", strict)

    n_sub, n_susp = con.execute(
        f"""SELECT count(*),
                   count(*) FILTER (WHERE ccLabel = '{schema.LABEL_SUSPICIOUS}')
            FROM read_parquet({_q(out_path)})"""
    ).fetchone()
    return {"path": out_path.name, "n_subgraphs": int(n_sub),
            "n_suspicious": int(n_susp),
            "base_rate": (n_susp / n_sub) if n_sub else 0.0,
            "members_total": int(nm_total), "orphan_nodes": int(orphan_nodes),
            "orphan_cc": int(orphan_cc),
            "cols": {"node_id": node_id_col, "node_cc": node_cc_col,
                     "cc_id": cc_id_col, "cc_label": cc_label_col}}


def _validate_counts(report, strict, cfg, n_nodes, edge_info, sub_info) -> None:
    if not cfg.validate_counts:
        return
    if cfg.expected_nodes is not None and n_nodes != cfg.expected_nodes:
        _warn(report, f"node count {n_nodes} != Table 1 {cfg.expected_nodes}", strict)
    if cfg.expected_edges is not None and edge_info["edges_total"] != cfg.expected_edges:
        _warn(report,
              f"edge count {edge_info['edges_total']} != Table 1 "
              f"{cfg.expected_edges}", strict)
    if (cfg.expected_subgraphs is not None
            and sub_info["n_subgraphs"] != cfg.expected_subgraphs):
        _warn(report,
              f"subgraph count {sub_info['n_subgraphs']} != Table 1 "
              f"{cfg.expected_subgraphs}", strict)
    if sub_info["n_suspicious"] != schema.N_SUSPICIOUS:
        # Informational only (2,763 paper vs 2,718 RevTrack vs your copy) — never a
        # fatal warning, and kept out of the manifest's `warnings` integrity list.
        print(f"[ingest] INFO: suspicious count {sub_info['n_suspicious']} != paper "
              f"{schema.N_SUSPICIOUS} (RevTrack: 2,718). Use your copy's count.")


def ingest(cfg: IngestConfig) -> IngestResult:
    """Run Stage 0. Returns the result with a written ingest_manifest.json."""
    paths = Elliptic2Paths(Path(cfg.raw_dir))
    paths.require_all()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: list[str] = []
    t0 = time.perf_counter()
    con = duckdb.connect()
    try:
        _configure(con, cfg)

        id_col, feat_cols = _build_id_map(con, paths, report, cfg.strict)
        _n_row = con.execute("SELECT count(*) FROM id_map").fetchone()
        assert _n_row is not None
        n_nodes = _n_row[0]

        con.execute(f"COPY id_map TO {_q(out_dir / 'id_map.parquet')} (FORMAT PARQUET)")

        t_feat = time.perf_counter()
        feat_info = _write_node_features(con, paths, id_col, feat_cols, n_nodes, cfg)
        t_edge = time.perf_counter()
        edge_info = _write_edge_index(con, paths, out_dir, report, cfg.strict)
        t_sub = time.perf_counter()
        sub_info = _write_subgraphs(con, paths, cfg, report, cfg.strict)
        t_end = time.perf_counter()

        _validate_counts(report, cfg.strict, cfg, n_nodes, edge_info, sub_info)
    finally:
        con.close()

    manifest = {
        "dataset": "elliptic2",
        "stage": "0-ingest",
        "raw_dir": str(paths.root.resolve()),
        "out_dir": str(out_dir.resolve()),
        "id_column": id_col,
        "n_nodes": int(n_nodes),
        "id_map": "id_map.parquet",
        "node_features": feat_info,
        "edge_index": edge_info,
        "subgraphs": sub_info,
        "warnings": report,
        "timings_sec": {
            "id_map": round(t_feat - t0, 3),
            "node_features": round(t_edge - t_feat, 3),
            "edge_index": round(t_sub - t_edge, 3),
            "subgraphs": round(t_end - t_sub, 3),
            "total": round(t_end - t0, 3),
        },
    }
    man_path = out_dir / "ingest_manifest.json"
    with man_path.open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return IngestResult(out_dir=out_dir, manifest=manifest)


# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="ellip2-ingest",
        description="Stage 0: ingest Elliptic2 CSVs into model-ready artifacts.",
    )
    p.add_argument("--raw-dir", required=True, type=Path,
                   help="directory containing the five Elliptic2 CSVs")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--feature-dtype", default="float32",
                   choices=["float32", "float16"])
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--memory-limit", default=None, help="e.g. 12GB")
    p.add_argument("--temp-dir", type=Path, default=None,
                   help="DuckDB spill directory (default <out-dir>/_duckdb_tmp)")
    p.add_argument("--strict", action="store_true",
                   help="hard-fail on count/integrity mismatch")
    p.add_argument("--no-validate-counts", action="store_true")
    args = p.parse_args(argv)

    cfg = IngestConfig(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        feature_dtype=args.feature_dtype,
        threads=args.threads,
        memory_limit=args.memory_limit,
        temp_dir=args.temp_dir,
        strict=args.strict,
        validate_counts=not args.no_validate_counts,
    )
    res = ingest(cfg)
    m = res.manifest
    print(f"Ingest complete -> {res.out_dir}")
    print(f"  nodes={m['n_nodes']:,} "
          f"edges={m['edge_index']['edges_written']:,} "
          f"(dangling {m['edge_index']['dangling']:,})")
    print(f"  subgraphs={m['subgraphs']['n_subgraphs']:,} "
          f"suspicious={m['subgraphs']['n_suspicious']:,} "
          f"({m['subgraphs']['base_rate']:.4%})")
    print(f"  node_features {m['node_features']['shape']} "
          f"{m['node_features']['dtype']}")
    print(f"  total {m['timings_sec']['total']}s; "
          f"warnings={len(m['warnings'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
