"""Unit test for Stage 1 per-cluster temporal features (T-005).

Builds tiny synthetic timestamp sequences with hand-computed answers for the
two columns:

    activity_span = t_max − t_min of a cluster's incident events
    burstiness    = (σ − μ) / (σ + μ) over the inter-event gaps (population σ)

Each measured source node owns a disjoint set of out-edges to dedicated sink
nodes (idx ≥ 10), so its only incident events are the ones we intend. Covers
the burstiness regimes by clean closed-form values: regular train → −1, two
events (single gap) → −1, Poisson-like → 0, the bursty ``(√k−1)/(√k+1)`` family
→ +1/3, plus the single-event / isolated empty cases and incident = in ∪ out.
Synthetic, CPU-only, no external resources (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_temporal.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from ellip2.features.temporal import (  # noqa: E402
    COLUMNS,
    compute_temporal_features,
)

# (src, dst, timestamp). Measured source nodes are 0..7; sinks are 10..41 and
# never asserted on. Node 5 is isolated (appears in no edge).
#
#   node 0: gaps {0,5,10,15} -> [5,5,5]      regular    span 15  B = -1
#   node 1: gaps {0,0,2}     -> [0,2]         Poisson    span  2  B =  0
#   node 2: gaps {0,1,10}    -> [1,9]         span 10, mu 5 sigma 4  B = -1/9
#   node 3: single event {7}                 span  0  B = empty (no gap)
#   node 4: two events {3,8} -> [5]           span  5  B = -1 (single gap)
#   node 6: {0,0,0,0,0,9} -> [0,0,0,0,9]      bursty     span  9  B = (2-1)/(2+1)=1/3
#   node 7: out {2} + in {8} -> incident {2,8} span 6  B = -1  (in ∪ out)
_EDGES = [
    (0, 10, 0.0), (0, 11, 5.0), (0, 12, 10.0), (0, 13, 15.0),
    (1, 14, 0.0), (1, 15, 0.0), (1, 16, 2.0),
    (2, 17, 0.0), (2, 18, 1.0), (2, 19, 10.0),
    (3, 20, 7.0),
    (4, 21, 3.0), (4, 22, 8.0),
    (6, 30, 0.0), (6, 31, 0.0), (6, 32, 0.0), (6, 33, 0.0), (6, 34, 0.0),
    (6, 35, 9.0),
    (7, 40, 2.0), (41, 7, 8.0),
]
N = 42
EDGE_INDEX = np.array(
    [[e[0] for e in _EDGES], [e[1] for e in _EDGES]], dtype=np.int32
)
TIMESTAMPS = np.array([e[2] for e in _EDGES], dtype=np.float64)


def _features(**kw) -> dict[str, np.ndarray]:
    return compute_temporal_features(EDGE_INDEX, TIMESTAMPS, N, **kw)


def test_columns_and_shapes():
    f = _features()
    assert set(f) == set(COLUMNS)
    assert all(f[c].shape == (N,) for c in COLUMNS)
    assert all(f[c].dtype == np.float64 for c in COLUMNS)


def test_activity_span_hand_values():
    f = _features()
    assert np.isclose(f["activity_span"][0], 15.0)
    assert np.isclose(f["activity_span"][1], 2.0)
    assert np.isclose(f["activity_span"][2], 10.0)
    assert np.isclose(f["activity_span"][3], 0.0)   # single event: genuine 0
    assert np.isclose(f["activity_span"][4], 5.0)
    assert np.isclose(f["activity_span"][6], 9.0)
    assert np.isclose(f["activity_span"][7], 6.0)   # incident {2,8}


def test_burstiness_regular_train_is_minus_one():
    # Equal gaps [5,5,5] -> sigma 0 -> B = -1 (perfectly periodic).
    f = _features()
    assert np.isclose(f["burstiness"][0], -1.0)


def test_burstiness_single_gap_is_minus_one():
    # Two events -> one gap -> sigma 0 -> B = -1.
    f = _features()
    assert np.isclose(f["burstiness"][4], -1.0)
    assert np.isclose(f["burstiness"][7], -1.0)


def test_burstiness_poisson_like_is_zero():
    # gaps [0,2]: mu 1, var = 4/2 - 1 = 1, sigma 1 -> B = (1-1)/(1+1) = 0.
    f = _features()
    assert np.isclose(f["burstiness"][1], 0.0)


def test_burstiness_intermediate_hand_value():
    # gaps [1,9]: mu 5, var = 82/2 - 25 = 16, sigma 4 -> B = (4-5)/(4+5) = -1/9.
    f = _features()
    assert np.isclose(f["burstiness"][2], -1.0 / 9.0)


def test_burstiness_bursty_is_positive():
    # gaps [0,0,0,0,9]: k=4 zeros + one spike -> B = (sqrt(4)-1)/(sqrt(4)+1)=1/3.
    f = _features()
    assert np.isclose(f["burstiness"][6], 1.0 / 3.0)
    # Ordering across regimes: bursty > Poisson > regular.
    assert f["burstiness"][6] > f["burstiness"][1] > f["burstiness"][0]


def test_single_and_zero_event_burstiness_empty():
    # Node 3 has one event (no gap); node 5 is isolated (no event).
    f = _features()
    assert f["burstiness"][3] == 0.0   # default empty_value
    assert f["burstiness"][5] == 0.0
    # A non-default empty_value distinguishes "undefined" from real values, and
    # keeps node 3's genuine single-event span of 0.0 separate from no-event.
    g = _features(empty_value=-1.0)
    assert g["burstiness"][3] == -1.0       # no gap -> fill
    assert g["burstiness"][5] == -1.0       # no event -> fill
    assert g["activity_span"][5] == -1.0    # no event -> span fill
    assert np.isclose(g["activity_span"][3], 0.0)   # single event: real 0
    assert np.isclose(g["burstiness"][0], -1.0)     # real value, unaffected


def test_all_same_timestamp_burstiness_empty():
    # Three events all at the same instant: gaps [0,0] -> sigma+mu = 0 -> empty.
    ei = np.array([[0, 0, 0], [1, 2, 3]], dtype=np.int32)
    f = compute_temporal_features(ei, [4.0, 4.0, 4.0], 4, empty_value=-9.0)
    assert f["burstiness"][0] == -9.0
    assert np.isclose(f["activity_span"][0], 0.0)  # span is a genuine 0


def test_empty_graph_all_fill():
    f = compute_temporal_features(np.empty((2, 0), dtype=np.int32), [], 3)
    for c in COLUMNS:
        assert f[c].tolist() == [0.0, 0.0, 0.0]


def test_timestamps_length_mismatch_rejected():
    try:
        compute_temporal_features(EDGE_INDEX, TIMESTAMPS[:-1], N)
    except ValueError as e:
        assert "timestamps" in str(e)
    else:
        raise AssertionError("expected ValueError for a timestamps length mismatch")


def test_bad_shape_rejected():
    try:
        compute_temporal_features(np.zeros((3, 2), dtype=np.int32), [0.0, 0.0], 3)
    except ValueError as e:
        assert "shape (2, E)" in str(e)
    else:
        raise AssertionError("expected ValueError for a non-(2,E) edge_index")


def test_out_of_range_endpoint_rejected():
    bad = np.array([[0, 2], [1, 9]], dtype=np.int32)  # 9 >= n_nodes=3
    try:
        compute_temporal_features(bad, [0.0, 1.0], 3)
    except ValueError as e:
        assert "out of range" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-range endpoint")


def test_negative_n_nodes_rejected():
    try:
        compute_temporal_features(EDGE_INDEX, TIMESTAMPS, -1)
    except ValueError as e:
        assert "n_nodes" in str(e)
    else:
        raise AssertionError("expected ValueError for negative n_nodes")


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
