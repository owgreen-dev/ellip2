"""Unit test for Stage 1 per-cluster edge-feature aggregates (T-002).

Asserts sum/mean/max/std aggregates (separately for in- and out-edges) against
hand-computed values on a tiny synthetic edge table, and checks that the
out-of-core DuckDB path agrees with the in-memory numpy path. Synthetic,
CPU-only, offline (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_edge_aggs.py``.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.features.edge_aggs import (  # noqa: E402
    aggregate_columns,
    compute_edge_aggregates,
    compute_edge_aggregates_duckdb,
)

# Hand-counted directed graph over 4 nodes (node 3 has no edges).
#   0->1  0->2  1->2  2->0
_SRC_ROW = [0, 0, 1, 2]
_DST_ROW = [1, 2, 2, 0]
N = 4
EDGE_INDEX = np.array([_SRC_ROW, _DST_ROW], dtype=np.int32)
# Two edge features per edge (anonymized ordinals stand-in).
EF = np.array(
    [
        [10.0, 1.0],  # 0->1
        [20.0, 2.0],  # 0->2
        [30.0, 3.0],  # 1->2
        [40.0, 4.0],  # 2->0
    ],
    dtype=np.float64,
)


def test_out_edge_aggregates_exact():
    f = compute_edge_aggregates(EDGE_INDEX, EF, N, feature_names=["a", "b"])
    # out-edges grouped by source.
    # node0: feature a = [10, 20]; node1: [30]; node2: [40]; node3: none.
    assert f["out_a_sum"].tolist() == [30.0, 30.0, 40.0, 0.0]
    assert f["out_a_mean"].tolist() == [15.0, 30.0, 40.0, 0.0]
    assert f["out_a_max"].tolist() == [20.0, 30.0, 40.0, 0.0]
    # population std of [10,20] is 5.0; singletons are 0.0; empty node -> 0.0.
    np.testing.assert_allclose(f["out_a_std"], [5.0, 0.0, 0.0, 0.0])
    # second feature on node0: [1, 2] -> sum 3, mean 1.5, max 2, std 0.5
    assert f["out_b_sum"][0] == 3.0
    assert f["out_b_mean"][0] == 1.5
    assert f["out_b_max"][0] == 2.0
    np.testing.assert_allclose(f["out_b_std"][0], 0.5)


def test_in_edge_aggregates_exact():
    f = compute_edge_aggregates(EDGE_INDEX, EF, N, feature_names=["a", "b"])
    # in-edges grouped by target.
    # node0: a=[40]; node1: a=[10]; node2: a=[20,30]; node3: none.
    assert f["in_a_sum"].tolist() == [40.0, 10.0, 50.0, 0.0]
    assert f["in_a_mean"].tolist() == [40.0, 10.0, 25.0, 0.0]
    assert f["in_a_max"].tolist() == [40.0, 10.0, 30.0, 0.0]
    np.testing.assert_allclose(f["in_a_std"], [0.0, 0.0, 5.0, 0.0])
    # node2 second feature in-edges: [2, 3] -> mean 2.5, std 0.5
    assert f["in_b_mean"][2] == 2.5
    np.testing.assert_allclose(f["in_b_std"][2], 0.5)


def test_column_set_and_shapes():
    f = compute_edge_aggregates(EDGE_INDEX, EF, N, feature_names=["a", "b"])
    assert set(f) == set(aggregate_columns(["a", "b"]))
    assert all(v.shape == (N,) for v in f.values())
    assert all(v.dtype == np.float64 for v in f.values())


def test_empty_value_configurable_and_distinct_from_genuine_zero():
    f = compute_edge_aggregates(
        EDGE_INDEX, EF, N, feature_names=["a", "b"], empty_value=-1.0
    )
    # node3 has no edges in either direction -> fill value everywhere.
    assert f["out_a_sum"][3] == -1.0
    assert f["in_a_sum"][3] == -1.0
    assert f["out_a_max"][3] == -1.0
    # node0 has a genuine out-sum of 30 -> unaffected by the fill.
    assert f["out_a_sum"][0] == 30.0


def test_feature_indices_subset():
    f = compute_edge_aggregates(
        EDGE_INDEX, EF, N, feature_indices=[1], feature_names=["b"]
    )
    assert set(f) == set(aggregate_columns(["b"]))
    assert f["out_b_sum"][0] == 3.0


def test_stats_subset_and_order():
    f = compute_edge_aggregates(
        EDGE_INDEX, EF, N, feature_indices=[0], feature_names=["a"],
        stats=("mean", "max"),
    )
    assert set(f) == {"in_a_mean", "in_a_max", "out_a_mean", "out_a_max"}


def test_empty_graph_all_fill():
    f = compute_edge_aggregates(
        np.empty((2, 0), dtype=np.int32),
        np.empty((0, 2), dtype=np.float64),
        3,
        feature_names=["a", "b"],
    )
    assert f["out_a_sum"].tolist() == [0.0, 0.0, 0.0]
    assert f["in_b_max"].tolist() == [0.0, 0.0, 0.0]


def test_bad_shapes_and_indices_rejected():
    for bad, msg in [
        (lambda: compute_edge_aggregates(np.zeros((3, 2)), EF, N), "shape (2, E)"),
        (
            lambda: compute_edge_aggregates(EDGE_INDEX, EF[:2], N),
            "rows but edge_index",
        ),
        (
            lambda: compute_edge_aggregates(EDGE_INDEX, EF, N, feature_indices=[9]),
            "out of range",
        ),
        (
            lambda: compute_edge_aggregates(
                EDGE_INDEX, EF, N, feature_names=["only-one"]
            ),
            "feature_names",
        ),
        (
            lambda: compute_edge_aggregates(EDGE_INDEX, EF, N, stats=("nope",)),
            "unknown stats",
        ),
    ]:
        try:
            bad()
        except ValueError as e:
            assert msg in str(e), f"expected {msg!r} in {e!r}"
        else:
            raise AssertionError(f"expected ValueError containing {msg!r}")


def test_duckdb_path_matches_numpy():
    # Write a tiny background_edges.csv + id_map.parquet and stream the same
    # aggregates out-of-core; assert they match the in-memory result.
    orig = [f"n{i}" for i in range(N)]
    with tempfile.TemporaryDirectory() as d:
        edges_csv = Path(d) / "background_edges.csv"
        with edges_csv.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["clId1", "clId2", "ef_0", "ef_1"])
            for k in range(EDGE_INDEX.shape[1]):
                s, t = int(EDGE_INDEX[0, k]), int(EDGE_INDEX[1, k])
                w.writerow([orig[s], orig[t], EF[k, 0], EF[k, 1]])

        id_map = Path(d) / "id_map.parquet"
        pq.write_table(
            pa.table({"idx": list(range(N)), "orig_id": orig}), id_map
        )

        got = compute_edge_aggregates_duckdb(
            edges_csv, id_map, N, feature_indices=[0, 1], feature_names=["a", "b"]
        )
        want = compute_edge_aggregates(
            EDGE_INDEX, EF, N, feature_indices=[0, 1], feature_names=["a", "b"]
        )
        assert set(got) == set(want)
        for c in want:
            np.testing.assert_allclose(got[c], want[c], atol=1e-9, err_msg=c)


# --------------------------------------------------------------------------- #


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
