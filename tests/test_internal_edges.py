"""Tests for Phase-2 internal edge feature extraction (ellip2.features.internal_edges)
and the border-model edge channel (border_assembly edge helpers).

CPU-only, synthetic. Builds tiny background_edges.csv + id_map.parquet + subgraphs.parquet,
runs the DuckDB extraction, and checks only same-subgraph edges (with features) survive.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.features.internal_edges import extract_internal_edges  # noqa: E402
from ellip2.pu.border_assembly import (  # noqa: E402
    build_subgraph_batch,
    extract_border_sets,
    fit_edge_standardizer,
    load_internal_edge_features,
)


def _write_artifacts(tmp: Path) -> tuple[Path, Path]:
    raw = tmp / "raw"
    raw.mkdir()
    art = tmp / "art"
    art.mkdir()
    # clusters c0..c3 -> idx 0..3; subgraph 0={0,1}, subgraph 1={2,3}
    with open(raw / "background_edges.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clId1", "clId2", "f0", "f1"])
        w.writerow(["c0", "c1", 1.0, 2.0])   # internal to subgraph 0
        w.writerow(["c2", "c3", 3.0, 4.0])   # internal to subgraph 1
        w.writerow(["c1", "c2", 9.0, 9.0])   # CROSS-subgraph -> must be dropped
    pq.write_table(
        pa.table({"orig_id": pa.array(["c0", "c1", "c2", "c3"]),
                  "idx": pa.array([0, 1, 2, 3], type=pa.int64())}),
        art / "id_map.parquet",
    )
    pq.write_table(
        pa.table({"ccId": pa.array(["a", "b"]), "ccLabel": pa.array(["suspicious", "licit"]),
                  "n_members": pa.array([2, 2], type=pa.int64()),
                  "member_idx": pa.array([[0, 1], [2, 3]], type=pa.list_(pa.int64()))}),
        art / "subgraphs.parquet",
    )
    return raw, art


def test_extract_keeps_only_internal_edges(tmp_path: Path) -> None:
    raw, art = _write_artifacts(tmp_path)
    out = tmp_path / "internal_edge_features.parquet"
    n = extract_internal_edges(raw, art, out)
    assert n == 2                                   # the cross-subgraph edge is dropped
    t = pq.read_table(out)
    rows = sorted(t.to_pylist(), key=lambda r: r["subgraph"])
    assert [r["subgraph"] for r in rows] == [0, 1]
    assert {r["src"] for r in rows} | {r["dst"] for r in rows} == {0, 1, 2, 3}
    assert rows[0]["f0"] == 1.0 and rows[0]["f1"] == 2.0    # features preserved


def test_load_and_standardize_edges(tmp_path: Path) -> None:
    raw, art = _write_artifacts(tmp_path)
    out = tmp_path / "ie.parquet"
    extract_internal_edges(raw, art, out)
    by_sg = load_internal_edge_features(out)
    assert set(by_sg) == {0, 1}
    assert by_sg[0].shape == (1, 2)                 # one edge, two features
    mean, std = fit_edge_standardizer([0, 1], by_sg)
    assert mean.shape == (2,) and (std > 0).all()


def test_build_batch_populates_edge_channel(tmp_path: Path) -> None:
    # border graph: 4->0 sender, 0->1 internal(=member edge), 1->5 receiver for subgraph 0
    edge_index = np.array([[4, 0, 1], [0, 1, 5]], dtype=np.int64)
    border = extract_border_sets(edge_index, [np.array([0, 1])], n_nodes=6, cap=8)
    nf = np.random.default_rng(0).standard_normal((6, 43)).astype(np.float32)
    edge_feats = {0: np.arange(2 * 95, dtype=np.float32).reshape(2, 95)}
    batch = build_subgraph_batch([0], border, nf, edge_dim=95, edge_features_by_sg=edge_feats)
    assert batch.edge_x.shape == (2, 95)            # populated, not empty
    assert batch.edge_batch.tolist() == [0, 0]
    # without edge features -> empty (Phase 1 behaviour)
    empty = build_subgraph_batch([0], border, nf, edge_dim=95)
    assert empty.edge_x.shape == (0, 95)


if __name__ == "__main__":
    import tempfile
    for t in (test_extract_keeps_only_internal_edges, test_load_and_standardize_edges,
              test_build_batch_populates_edge_channel):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print("ok")
