"""Tests for background-discovery helpers (ellip2.discovery.background).

CPU-only, synthetic (SIGN-101). T-026: the typology-signal export
(``source_sink_axis`` -> idx-aligned ``(N,)`` array) and the known-member
exclusion union over ``subgraphs.parquet``.
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
        test_main_writes_npy,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
