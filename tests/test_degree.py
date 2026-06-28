"""Unit test for Stage 1 per-cluster degree features (T-001).

Builds a tiny hand-counted directed graph and asserts exact in/out/total degree
and the ``in_out_ratio`` convention for the zero-denominator (pure-sink and
isolated) cases. Synthetic, CPU-only, no external resources (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_degree.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from ellip2.features.degree import COLUMNS, compute_degree_features  # noqa: E402

# Hand-counted directed graph over 6 nodes (n_nodes includes isolated node 5).
#   0->1  0->2  1->2  2->0  3->1  1->4
# Node 4 is a pure sink (in=1, out=0); node 5 is isolated (in=0, out=0).
_SRC_ROW = [0, 0, 1, 2, 3, 1]
_DST_ROW = [1, 2, 2, 0, 1, 4]
N = 6
EDGE_INDEX = np.array([_SRC_ROW, _DST_ROW], dtype=np.int32)

# Expectations worked out by hand.
EXP_OUT = [2, 2, 1, 1, 0, 0]
EXP_IN = [1, 2, 2, 0, 1, 0]
EXP_TOTAL = [3, 4, 3, 1, 1, 0]
# in/out: 0->0.5, 1->1.0, 2->2.0, 3->0.0 (genuine: in=0,out>0),
#         4->fill (out=0), 5->fill (out=0).


def test_degree_exact_values():
    f = compute_degree_features(EDGE_INDEX, N)
    assert set(f) == set(COLUMNS)
    assert f["out_degree"].tolist() == EXP_OUT
    assert f["in_degree"].tolist() == EXP_IN
    assert f["total_degree"].tolist() == EXP_TOTAL
    # Sizing + isolated node.
    assert all(f[c].shape == (N,) for c in COLUMNS)
    assert f["total_degree"][5] == 0


def test_in_out_ratio_convention():
    f = compute_degree_features(EDGE_INDEX, N)  # default zero_denom_value=0.0
    r = f["in_out_ratio"]
    assert r.dtype == np.float64
    assert r[0] == 0.5
    assert r[1] == 1.0
    assert r[2] == 2.0
    # Genuine zero ratio (in=0, out>0) — distinct from the undefined case.
    assert r[3] == 0.0
    # Undefined (out==0): pure sink and isolated node both take the fill value.
    assert r[4] == 0.0
    assert r[5] == 0.0


def test_zero_denom_value_is_configurable():
    f = compute_degree_features(EDGE_INDEX, N, zero_denom_value=-1.0)
    r = f["in_out_ratio"]
    assert r[4] == -1.0  # out==0 -> fill
    assert r[5] == -1.0  # out==0 -> fill
    assert r[3] == 0.0   # out>0  -> genuine ratio, unaffected by the fill
    assert r[0] == 0.5


def test_empty_graph_all_zero():
    f = compute_degree_features(np.empty((2, 0), dtype=np.int32), 4)
    assert f["in_degree"].tolist() == [0, 0, 0, 0]
    assert f["out_degree"].tolist() == [0, 0, 0, 0]
    assert f["in_out_ratio"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_self_loop_counts_both_directions():
    # A single self loop on node 1 contributes one in and one out.
    ei = np.array([[1], [1]], dtype=np.int32)
    f = compute_degree_features(ei, 3)
    assert f["in_degree"].tolist() == [0, 1, 0]
    assert f["out_degree"].tolist() == [0, 1, 0]
    assert f["in_out_ratio"][1] == 1.0


def test_out_of_range_endpoint_rejected():
    bad = np.array([[0, 2], [1, 5]], dtype=np.int32)  # 5 >= n_nodes=3
    try:
        compute_degree_features(bad, 3)
    except ValueError as e:
        assert "out of range" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-range endpoint")


def test_bad_shape_rejected():
    try:
        compute_degree_features(np.zeros((3, 4), dtype=np.int32), 4)
    except ValueError as e:
        assert "shape (2, E)" in str(e)
    else:
        raise AssertionError("expected ValueError for non-(2,E) edge_index")


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
