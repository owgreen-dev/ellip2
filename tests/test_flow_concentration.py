"""Unit test for Stage 1 per-cluster flow-concentration features (T-003).

Builds a tiny directed weighted graph and asserts Gini / HHI / max-counterparty
share vs values worked out by hand, separately for in- and out-flow. Covers the
diagnostic extremes called out in the acceptance criteria: equal counterparties
-> Gini 0, one dominant counterparty -> Gini near its theoretical ceiling and a
near-1 max share. Also exercises parallel-edge merging and the empty-direction
fill. Synthetic, CPU-only, no external resources (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_flow_concentration.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from ellip2.features.flow_concentration import (  # noqa: E402
    COLUMNS,
    compute_flow_concentration,
)

# Hand-built directed weighted graph over 8 nodes. (src, dst, weight) per edge.
#   node 0 out: two parallel edges to 1 (merge to weight 2) + 2,3 -> three EQUAL
#               counterparties after merging  -> out Gini 0
#   node 4 out: single counterparty           -> fully concentrated
#   node 6 out: one dominant counterparty among four
# In-flow is read off the same edges (grouped by source).
_EDGES = [
    (0, 1, 1.0),
    (0, 1, 1.0),  # parallel edge -> merges with the one above
    (0, 2, 2.0),
    (0, 3, 2.0),
    (4, 5, 7.0),
    (6, 1, 97.0),
    (6, 2, 1.0),
    (6, 3, 1.0),
    (6, 7, 1.0),
]
N = 8
EDGE_INDEX = np.array([[e[0] for e in _EDGES], [e[1] for e in _EDGES]], dtype=np.int32)
WEIGHTS = np.array([e[2] for e in _EDGES], dtype=np.float64)


def _features() -> dict[str, np.ndarray]:
    return compute_flow_concentration(EDGE_INDEX, WEIGHTS, N)


def test_columns_and_shapes():
    f = _features()
    assert set(f) == set(COLUMNS)
    assert all(f[c].shape == (N,) for c in COLUMNS)
    assert all(f[c].dtype == np.float64 for c in COLUMNS)


def test_out_flow_equal_counterparties_gini_zero():
    # Node 0 out: counterparties {1:2 (merged), 2:2, 3:2} -> three equal shares.
    f = _features()
    assert np.isclose(f["out_gini"][0], 0.0)
    assert np.isclose(f["out_hhi"][0], 1.0 / 3.0)            # 3 * (1/3)^2
    assert np.isclose(f["out_max_counterparty_share"][0], 1.0 / 3.0)


def test_out_flow_single_counterparty_fully_concentrated():
    # Node 4 out: {5:7} -> Gini 0, HHI 1, share 1.
    f = _features()
    assert np.isclose(f["out_gini"][4], 0.0)
    assert np.isclose(f["out_hhi"][4], 1.0)
    assert np.isclose(f["out_max_counterparty_share"][4], 1.0)


def test_out_flow_one_dominant_counterparty():
    # Node 6 out: {1:97, 2:1, 3:1, 7:1}, S=100. Gini = 288/(4*100) = 0.72,
    # HHI = 0.97^2 + 3*0.01^2 = 0.9412, share = 0.97.
    f = _features()
    assert np.isclose(f["out_gini"][6], 0.72)
    assert np.isclose(f["out_hhi"][6], 0.9412)
    assert np.isclose(f["out_max_counterparty_share"][6], 0.97)
    # Dominant is far more concentrated than the equal case, and near the
    # n=4 ceiling (n-1)/n = 0.75.
    assert f["out_gini"][6] > f["out_gini"][0]
    assert f["out_gini"][6] <= 0.75


def test_in_flow_hand_computed():
    f = _features()
    # Node 1 in: {0:2 (merged), 6:97}, S=99.
    assert np.isclose(f["in_max_counterparty_share"][1], 97.0 / 99.0)
    assert np.isclose(f["in_hhi"][1], (2.0 / 99) ** 2 + (97.0 / 99) ** 2)
    assert np.isclose(f["in_gini"][1], 95.0 / 198.0)  # (-2 + 97) / (2*99)
    # Node 2 in: {0:2, 6:1}, S=3.
    assert np.isclose(f["in_max_counterparty_share"][2], 2.0 / 3.0)
    assert np.isclose(f["in_hhi"][2], 5.0 / 9.0)       # (2/3)^2 + (1/3)^2
    assert np.isclose(f["in_gini"][2], 1.0 / 6.0)      # (-1 + 2) / (2*3)
    # Node 5 in: {4:7} single counterparty.
    assert np.isclose(f["in_gini"][5], 0.0)
    assert np.isclose(f["in_hhi"][5], 1.0)
    assert np.isclose(f["in_max_counterparty_share"][5], 1.0)


def test_empty_direction_takes_fill_value():
    f = _features()
    # Nodes 1,2,3,5,7 have no out-edges; nodes 0,4,6 have no in-edges.
    for c in ("out_gini", "out_hhi", "out_max_counterparty_share"):
        assert f[c][1] == 0.0
        assert f[c][5] == 0.0
    for c in ("in_gini", "in_hhi", "in_max_counterparty_share"):
        assert f[c][0] == 0.0
        assert f[c][4] == 0.0
    # Configurable fill leaves real values alone.
    g = compute_flow_concentration(EDGE_INDEX, WEIGHTS, N, empty_value=-1.0)
    assert g["out_gini"][1] == -1.0          # no out-edges -> fill
    assert np.isclose(g["out_gini"][0], 0.0)  # real value, unaffected


def test_parallel_edges_merge_into_one_counterparty():
    # Two 0->1 edges must merge: node 0's out distribution is {2,2,2}, not
    # {1,1,2,2}. A non-merging implementation would give a non-zero Gini here.
    f = _features()
    assert np.isclose(f["out_gini"][0], 0.0)
    assert np.isclose(f["out_max_counterparty_share"][0], 1.0 / 3.0)


def test_empty_graph_all_fill():
    f = compute_flow_concentration(np.empty((2, 0), dtype=np.int32), [], 3)
    for c in COLUMNS:
        assert f[c].tolist() == [0.0, 0.0, 0.0]


def test_negative_weight_rejected():
    bad_w = WEIGHTS.copy()
    bad_w[0] = -1.0
    try:
        compute_flow_concentration(EDGE_INDEX, bad_w, N)
    except ValueError as e:
        assert "non-negative" in str(e)
    else:
        raise AssertionError("expected ValueError for a negative weight")


def test_weights_length_mismatch_rejected():
    try:
        compute_flow_concentration(EDGE_INDEX, WEIGHTS[:-1], N)
    except ValueError as e:
        assert "weights" in str(e)
    else:
        raise AssertionError("expected ValueError for a weights length mismatch")


def test_out_of_range_endpoint_rejected():
    bad = np.array([[0, 2], [1, 9]], dtype=np.int32)  # 9 >= n_nodes=3
    try:
        compute_flow_concentration(bad, [1.0, 1.0], 3)
    except ValueError as e:
        assert "out of range" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-range endpoint")


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
