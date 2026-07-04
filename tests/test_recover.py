"""Tests for Stage-3 endpoint generation + exit-path recovery (ellip2.exit_paths.recover)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.exit_paths.recover import endpoints_from_features, recover_exit_paths  # noqa: E402


def test_endpoints_from_features_top_percentile(tmp_path: Path) -> None:
    idx = np.arange(10)
    score = np.arange(10, dtype=float)          # 0..9, top 20% = idx 8,9
    pq.write_table(
        pa.table({"idx": pa.array(idx), "endpoint_score": pa.array(score)}),
        tmp_path / "cf.parquet",
    )
    ep = endpoints_from_features(tmp_path / "cf.parquet", percentile=0.8)
    assert set(ep.tolist()) == {8, 9}


def test_recover_exit_path_traces_shortest() -> None:
    # 0 -> 1 -> 2 -> 3(endpoint); subgraph 0 = {0}. Also a dead-end 4->5 (no endpoint).
    edge_index = np.array([[0, 1, 2, 4], [1, 2, 3, 5]], dtype=np.int64)
    members = [np.array([0]), np.array([4])]
    paths = recover_exit_paths(edge_index, members, [0, 1], endpoints=[3], n_nodes=6, max_hops=6)
    assert paths[0] == [0, 1, 2, 3]             # traced member -> endpoint
    assert paths[1] == []                        # subgraph 1 can't reach an endpoint


def test_recover_respects_max_hops() -> None:
    edge_index = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    members = [np.array([0])]
    # endpoint 3 is 3 hops away; horizon of 2 must NOT reach it
    assert recover_exit_paths(edge_index, members, [0], [3], 4, max_hops=2)[0] == []
    assert recover_exit_paths(edge_index, members, [0], [3], 4, max_hops=6)[0] == [0, 1, 2, 3]


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_endpoints_from_features_top_percentile(Path(d))
    test_recover_exit_path_traces_shortest()
    test_recover_respects_max_hops()
    print("ok")
