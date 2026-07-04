"""Tests for background-discovery helpers (ellip2.discovery.background).

CPU-only, synthetic (SIGN-101). T-026: the typology-signal export
(``source_sink_axis`` -> idx-aligned ``(N,)`` array) and the known-member
exclusion union over ``subgraphs.parquet``. T-027: the per-candidate reachability
carve (``candidate_member_sets``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402

from ellip2.discovery import background  # noqa: E402

# a non-trivial permutation of range(5), so idx-alignment can't pass by accident
_SHUFFLE = np.array([2, 0, 4, 1, 3], dtype=np.int64)


def _write_features(tmp: Path, axis: np.ndarray, *, order: np.ndarray | None = None) -> Path:
    """cluster_features.parquet with rows optionally written out of idx order."""
    n = axis.size
    idx = np.arange(n, dtype=np.int64)
    o = idx if order is None else order
    path = tmp / "cluster_features.parquet"
    pq.write_table(
        pa.table({
            "idx": pa.array(idx[o]),
            "endpoint_score": pa.array(np.zeros(n)[o]),
            "source_sink_axis": pa.array(axis[o]),
        }),
        path,
    )
    return path


def _write_subgraphs(tmp: Path, members: list[list[int]], labels: list[str]) -> Path:
    path = tmp / "subgraphs.parquet"
    pq.write_table(
        pa.table({
            "ccId": pa.array([f"cc{i}" for i in range(len(members))]),
            "ccLabel": pa.array(labels),
            "n_members": pa.array([len(m) for m in members], type=pa.int64()),
            "member_idx": pa.array(members, type=pa.list_(pa.int64())),
        }),
        path,
    )
    return path


def test_typology_signal_aligned_to_idx(tmp_path: Path) -> None:
    axis = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    path = _write_features(tmp_path, axis, order=_SHUFFLE)  # rows out of idx order
    signal = background.typology_signal_from_features(path)
    assert signal.shape == (5,)
    # recovered per-idx despite the shuffled file rows
    np.testing.assert_allclose(signal, axis)


def test_typology_signal_custom_column(tmp_path: Path) -> None:
    axis = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    path = _write_features(tmp_path, axis)
    got = background.typology_signal_from_features(path, score_col="endpoint_score")
    np.testing.assert_allclose(got, np.zeros(5))


def test_typology_signal_noncontiguous_idx_raises(tmp_path: Path) -> None:
    path = tmp_path / "cluster_features.parquet"
    pq.write_table(
        pa.table({
            "idx": pa.array(np.array([0, 1, 3], dtype=np.int64)),  # gap at 2
            "source_sink_axis": pa.array([0.1, 0.2, 0.3]),
        }),
        path,
    )
    with pytest.raises(ValueError, match="contiguous"):
        background.typology_signal_from_features(path)


def test_known_member_idx_union(tmp_path: Path) -> None:
    members = [[0, 1, 2], [2, 3], [7, 5]]  # overlap on 2, out of order
    path = _write_subgraphs(tmp_path, members, ["suspicious", "licit", "suspicious"])
    got = background.known_member_idx(path)
    np.testing.assert_array_equal(got, np.array([0, 1, 2, 3, 5, 7], dtype=np.int64))


def test_known_member_idx_bounded(tmp_path: Path) -> None:
    members = [[0, 1, 2], [2, 3], [7, 5]]
    path = _write_subgraphs(tmp_path, members, ["suspicious", "licit", "suspicious"])
    got = background.known_member_idx(path, n_nodes=5)  # drops 5, 7
    np.testing.assert_array_equal(got, np.array([0, 1, 2, 3], dtype=np.int64))


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    """(2, E) int64 edge_index from a list of (src, dst) tuples."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).T
    return np.ascontiguousarray(arr)


# 0 -> 1 -> 2 -> 3 (endpoint), plus a dead-end branch 1 -> 4 that reaches nothing.
_CARVE_EDGES = [(0, 1), (1, 2), (2, 3), (1, 4)]


def test_candidate_member_sets_carve_equals_known(tmp_path: Path) -> None:
    ei = _edge_index(_CARVE_EDGES)
    got = background.candidate_member_sets(ei, [0], [3], 5, max_hops=6)
    # the ≤6-hop 0->3 path is {0,1,2,3}; endpoint 3 dropped; dead-end 4 excluded.
    assert set(got) == {0}
    np.testing.assert_array_equal(got[0], np.array([0, 1, 2], dtype=np.int64))
    assert got[0].dtype == np.int64


def test_candidate_member_sets_drops_endpoint(tmp_path: Path) -> None:
    ei = _edge_index(_CARVE_EDGES)
    got = background.candidate_member_sets(ei, [0], [3], 5, max_hops=6)
    assert 3 not in set(got[0].tolist())  # endpoint never in a member set
    assert 4 not in set(got[0].tolist())  # off-path dead end excluded


def test_candidate_member_sets_respects_max_hops(tmp_path: Path) -> None:
    ei = _edge_index(_CARVE_EDGES)
    # candidate 0 is 3 hops from endpoint 3, candidate 1 is 2 hops.
    got = background.candidate_member_sets(ei, [0, 1], [3], 5, max_hops=2)
    assert set(got) == {0, 1}
    np.testing.assert_array_equal(got[0], np.zeros(0, dtype=np.int64))  # unreachable -> empty
    np.testing.assert_array_equal(got[1], np.array([1, 2], dtype=np.int64))


def test_candidate_member_sets_dedups_candidates(tmp_path: Path) -> None:
    ei = _edge_index(_CARVE_EDGES)
    got = background.candidate_member_sets(ei, [0, 0], [3], 5, max_hops=6)
    assert set(got) == {0}  # deduplicated


def test_candidate_member_sets_bad_edge_index_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shape"):
        background.candidate_member_sets(np.zeros((3, 4)), [0], [1], 5)


def test_main_writes_npy(tmp_path: Path) -> None:
    axis = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    path = _write_features(tmp_path, axis, order=_SHUFFLE)
    out = tmp_path / "typology_signal.npy"
    rc = background.main(["--features", str(path), "--out", str(out)])
    assert rc == 0 and out.is_file()
    np.testing.assert_allclose(np.load(out), axis)


if __name__ == "__main__":
    import tempfile

    for fn in (
        test_typology_signal_aligned_to_idx,
        test_typology_signal_custom_column,
        test_typology_signal_noncontiguous_idx_raises,
        test_known_member_idx_union,
        test_known_member_idx_bounded,
        test_candidate_member_sets_carve_equals_known,
        test_candidate_member_sets_drops_endpoint,
        test_candidate_member_sets_respects_max_hops,
        test_candidate_member_sets_dedups_candidates,
        test_candidate_member_sets_bad_edge_index_raises,
        test_main_writes_npy,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
