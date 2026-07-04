"""Stage 2 (border model, Phase 2) — extract per-subgraph internal edge features.

The border model's internal *edge* channel needs the 95-d features of each labeled
subgraph's internal edges. Those features live only in ``background_edges.csv`` (77 GB;
``clId1,clId2`` + 95 features) — ``edge_index.npy`` carries connectivity but no features,
and ``edges.csv`` is a bare edge list. This module streams ``background_edges.csv`` once
through DuckDB, remaps ``clId1/clId2`` to the contiguous ``idx`` space via ``id_map``, and
keeps only edges whose two endpoints are members of the **same** labeled subgraph — writing
``internal_edge_features.parquet`` (``subgraph``, ``src``, ``dst``, ``f0..f94``).

``subgraph`` is the row position in ``subgraphs.parquet`` (matches
:mod:`ellip2.pu.border_assembly`). Out-of-core with a DuckDB ``--memory-limit`` / ``--temp-dir``
(the Stage-1 lesson: unbounded DuckDB thrashes a no-swap box).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ellip2.data import schema


def _q(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def _qi(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _node_sub_table(subgraphs_parquet: Path) -> pa.Table:
    """Build a ``(idx, sg)`` arrow table: cluster idx → its subgraph row position."""
    members = pq.read_table(subgraphs_parquet, columns=["member_idx"]).column(0).to_pylist()
    idx_parts, sg_parts = [], []
    for pos, m in enumerate(members):
        arr = np.asarray(m, dtype=np.int64)
        arr = arr[arr >= 0]
        if arr.size:
            idx_parts.append(arr)
            sg_parts.append(np.full(arr.size, pos, dtype=np.int64))
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64)
    sg = np.concatenate(sg_parts) if sg_parts else np.zeros(0, np.int64)
    return pa.table({"idx": pa.array(idx), "sg": pa.array(sg)})


def extract_internal_edges(
    raw_dir: Path,
    artifacts_dir: Path,
    out_path: Path,
    *,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
    threads: int | None = None,
) -> int:
    """Write ``internal_edge_features.parquet``; return the internal-edge row count."""
    import duckdb  # noqa: PLC0415

    edges_csv = schema.Elliptic2Paths(raw_dir).background_edges
    id_map = artifacts_dir / "id_map.parquet"
    node_sub = _node_sub_table(artifacts_dir / "subgraphs.parquet")

    con = duckdb.connect()
    try:
        if threads is not None:
            con.execute(f"SET threads TO {int(threads)}")
        if memory_limit is not None:
            con.execute(f"SET memory_limit = '{memory_limit}'")
        if temp_dir is not None:
            con.execute(f"SET temp_directory = {_q(Path(temp_dir))}")
        con.execute("SET preserve_insertion_order = false")
        con.register("node_sub", node_sub)

        all_cols = [
            r[0]
            for r in con.execute(
                f"DESCRIBE SELECT * FROM read_csv_auto({_q(edges_csv)})"
            ).fetchall()
        ]
        src = schema.resolve_column(all_cols, schema.COL_EDGE_SRC, 0)
        dst = schema.resolve_column(all_cols, schema.COL_EDGE_DST, 1)
        feat_cols = [c for c in all_cols if c not in (src, dst)]
        sel = ", ".join(
            f"e.{_qi(c)} AS {_qi(f'f{j}')}" for j, c in enumerate(feat_cols)
        )
        sql = (
            f"SELECT nsa.sg AS subgraph, a.idx AS src, b.idx AS dst, {sel} "
            f"FROM read_csv_auto({_q(edges_csv)}) e "
            f"JOIN read_parquet({_q(id_map)}) a ON e.{_qi(src)} = a.orig_id "
            f"JOIN read_parquet({_q(id_map)}) b ON e.{_qi(dst)} = b.orig_id "
            f"JOIN node_sub nsa ON a.idx = nsa.idx "
            f"JOIN node_sub nsb ON b.idx = nsb.idx "
            f"WHERE nsa.sg = nsb.sg"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        reader = con.execute(sql).to_arrow_reader(100_000)
        writer: pq.ParquetWriter | None = None
        n = 0
        try:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                if writer is None:
                    writer = pq.ParquetWriter(out_path, batch.schema)
                writer.write_batch(batch)
                n += batch.num_rows
        finally:
            if writer is not None:
                writer.close()
        if writer is None:  # no internal edges — still emit an empty, typed file
            fields = [("subgraph", pa.int64()), ("src", pa.int64()), ("dst", pa.int64())]
            fields += [(f"f{j}", pa.float64()) for j in range(len(feat_cols))]
            pq.write_table(pa.table({k: pa.array([], type=t) for k, t in fields}), out_path)
        return n
    finally:
        con.close()


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Phase 2: extract per-subgraph internal edge features (95-d).",
    )
    p.add_argument("--raw-dir", required=True, type=Path, help="holds background_edges.csv")
    p.add_argument("--artifacts-dir", required=True, type=Path,
                   help="Stage 0 dir with id_map.parquet, subgraphs.parquet")
    p.add_argument("--out", required=True, type=Path, help="internal_edge_features.parquet")
    p.add_argument("--memory-limit", default=None, help="DuckDB memory cap, e.g. '24GB'")
    p.add_argument("--temp-dir", type=Path, default=None, help="DuckDB spill dir")
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args(argv)

    n = extract_internal_edges(
        args.raw_dir, args.artifacts_dir, args.out,
        memory_limit=args.memory_limit, temp_dir=args.temp_dir, threads=args.threads,
    )
    print(f"[internal_edges] wrote {n:,} internal edges -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
