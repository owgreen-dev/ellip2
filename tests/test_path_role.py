"""Unit test for the Stage 1 source/sink (endpoint) heuristic (T-006).

Asserts that high-in-degree + large-size + high-throughput clusters score
endpoint/exchange-like (high ``endpoint_score``, sink-leaning axis) while low ones
do not, on synthetic inputs with hand-computed percentile values. Also pins the
percentile-rank convention (ties, single node) and monotonicity. Synthetic,
CPU-only, no external resources (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_path_role.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from ellip2.features.path_role import (  # noqa: E402
    COLUMNS,
    compute_path_role,
)

# Five clusters, monotonically graded from a clear sink (idx 0) to a clear
# source (idx 4). All four channels are strictly ordered so percentile ranks are
# tie-free and exact: idx0 ranks highest in in_degree/size/throughput and lowest
# in out_degree.
IN_DEGREE = [100, 50, 10, 5, 1]
OUT_DEGREE = [1, 5, 10, 50, 100]
SIZE = [1000, 500, 100, 50, 10]
THROUGHPUT = [10000, 5000, 1000, 500, 100]
N = 5

# Percentile ranks (avg-rank / (n-1)) for a strictly descending channel: idx0 ->
# 1.0, idx4 -> 0.0. For out_degree (strictly ascending) it is the mirror image.
# endpoint_score = mean(pct_in, pct_size, pct_thru):
EXP_ENDPOINT = [1.0, 0.75, 0.5, 0.25, 0.0]
# source_sink_axis = pct_in - pct_out:
EXP_AXIS = [1.0, 0.5, 0.0, -0.5, -1.0]


def test_columns_and_shapes():
    f = compute_path_role(IN_DEGREE, OUT_DEGREE, SIZE, THROUGHPUT)
    assert set(f) == set(COLUMNS)
    for c in COLUMNS:
        assert f[c].shape == (N,)
        assert f[c].dtype == np.float64


def test_endpoint_score_hand_values():
    f = compute_path_role(IN_DEGREE, OUT_DEGREE, SIZE, THROUGHPUT)
    assert np.allclose(f["endpoint_score"], EXP_ENDPOINT)
    # The high-in-degree + large-size + high-throughput cluster scores top...
    assert f["endpoint_score"].argmax() == 0
    assert f["endpoint_score"][0] == 1.0
    # ...and the low-everything cluster scores bottom.
    assert f["endpoint_score"].argmin() == 4
    assert f["endpoint_score"][4] == 0.0
    # Strictly decreasing across the graded clusters.
    assert np.all(np.diff(f["endpoint_score"]) < 0)


def test_source_sink_axis_hand_values():
    f = compute_path_role(IN_DEGREE, OUT_DEGREE, SIZE, THROUGHPUT)
    assert np.allclose(f["source_sink_axis"], EXP_AXIS)
    # Sink-leaning (inflow) at idx0, source-leaning (outflow) at idx4.
    assert f["source_sink_axis"][0] == 1.0
    assert f["source_sink_axis"][4] == -1.0
    assert -1.0 <= f["source_sink_axis"].min() and f["source_sink_axis"].max() <= 1.0


def test_source_score_favors_outflow_end():
    f = compute_path_role(IN_DEGREE, OUT_DEGREE, SIZE, THROUGHPUT)
    s = f["source_score"]
    # source_score weights out_degree instead of in_degree, so relative to
    # endpoint_score the source end (idx4) is lifted and the sink end (idx0) is
    # damped — the heuristic separates the two roles.
    assert s[4] > f["endpoint_score"][4]
    assert s[0] < f["endpoint_score"][0]


def test_all_equal_inputs_are_neutral():
    # No structure -> every percentile rank is 0.5, axis is 0.
    eq = [7, 7, 7, 7]
    f = compute_path_role(eq, eq, eq, eq)
    assert np.allclose(f["endpoint_score"], 0.5)
    assert np.allclose(f["source_score"], 0.5)
    assert np.allclose(f["source_sink_axis"], 0.0)


def test_ties_share_mean_percentile():
    # in_degree [10,10,20]: the two tied 10s share avg ordinal rank 0.5 ->
    # pct 0.25; the 20 -> pct 1.0. Confirms tie handling is the mean rank.
    f = compute_path_role([10, 10, 20], [1, 1, 1], [1, 1, 1], [1, 1, 1])
    # source_sink_axis isolates pct_in - pct_out; pct_out is uniform (all tied ->
    # 0.5), so axis = pct_in - 0.5.
    assert np.allclose(f["source_sink_axis"], [0.25 - 0.5, 0.25 - 0.5, 1.0 - 0.5])


def test_single_cluster_is_midpoint():
    f = compute_path_role([5], [2], [3], [9])
    assert f["endpoint_score"][0] == 0.5
    assert f["source_score"][0] == 0.5
    assert f["source_sink_axis"][0] == 0.0


def test_monotone_in_in_degree():
    # Raising one cluster's in_degree can only raise its endpoint_score.
    base = compute_path_role([1, 2, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1])
    bumped = compute_path_role([1, 9, 3], [1, 1, 1], [1, 1, 1], [1, 1, 1])
    assert bumped["endpoint_score"][1] >= base["endpoint_score"][1]


def test_length_mismatch_rejected():
    try:
        compute_path_role([1, 2, 3], [1, 2], [1, 2, 3], [1, 2, 3])
    except ValueError as e:
        assert "out_degree" in str(e)
    else:
        raise AssertionError("expected ValueError for mismatched input lengths")


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
