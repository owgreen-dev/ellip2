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


# --- T-029: reuse the transposed step matrix across sweeps -------------------

# A slightly richer graph so several candidates carve non-trivial member sets.
#   0 -> 1 -> 2 -> 9,  3 -> 4 -> 9,  5 -> 6 -> 9,  plus a dead end 1 -> 8.
_OPT_EDGES = [(0, 1), (1, 2), (2, 9), (3, 4), (4, 9), (5, 6), (6, 9), (1, 8)]
_OPT_N = 10


def _carve_without_opt(
    ei: np.ndarray, cands: list[int], endpoints: list[int], n_nodes: int, *, max_hops: int
) -> dict[int, np.ndarray]:
    """Reference carve that never passes a precomputed step matrix (pre-T-029)."""
    from ellip2.exit_paths.path_search import (
        _as_node_ids,
        _build_directed_adjacency,
        bfs_reachable,
    )

    cand_ids = _as_node_ids("c", cands, n_nodes)
    ep_ids = _as_node_ids("e", endpoints, n_nodes)
    a = _build_directed_adjacency(ei.astype(np.int64), n_nodes)
    backward = bfs_reachable(a.transpose().tocsr(), ep_ids, max_hops=max_hops)
    drop = np.zeros(n_nodes, dtype=bool)
    drop[ep_ids] = True
    out: dict[int, np.ndarray] = {}
    for c in cand_ids:
        fwd = bfs_reachable(a, [int(c)], max_hops=max_hops)
        surv = (
            fwd.reached
            & backward.reached
            & ((fwd.hops + backward.hops) <= max_hops)
            & ~drop
        )
        out[int(c)] = np.nonzero(surv)[0].astype(np.int64)
    return out


def test_candidate_member_sets_identical_with_and_without_opt() -> None:
    ei = _edge_index(_OPT_EDGES)
    cands = [0, 3, 5]
    opt = background.candidate_member_sets(ei, cands, [9], _OPT_N, max_hops=6)
    ref = _carve_without_opt(ei, cands, [9], _OPT_N, max_hops=6)
    assert set(opt) == set(ref)
    for c in cands:
        np.testing.assert_array_equal(opt[c], ref[c])


def test_candidate_member_sets_transposes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from scipy import sparse

    counter = {"n": 0}
    orig = sparse.csr_matrix.transpose

    def counting_transpose(self, *a, **k):  # type: ignore[no-untyped-def]
        counter["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(sparse.csr_matrix, "transpose", counting_transpose)

    ei = _edge_index(_OPT_EDGES)
    counter["n"] = 0
    background.candidate_member_sets(ei, [0], [9], _OPT_N, max_hops=6)
    one = counter["n"]
    counter["n"] = 0
    background.candidate_member_sets(ei, [0, 3, 5], [9], _OPT_N, max_hops=6)
    many = counter["n"]

    # The O(nnz) transpose happens once per call regardless of candidate count —
    # it does NOT scale with the number of candidates (pre-T-029 it did).
    assert one == many


def test_main_writes_npy(tmp_path: Path) -> None:
    axis = np.array([0.1, -0.2, 0.3, -0.4, 0.5])
    path = _write_features(tmp_path, axis, order=_SHUFFLE)
    out = tmp_path / "typology_signal.npy"
    rc = background.main(["--features", str(path), "--out", str(out)])
    assert rc == 0 and out.is_file()
    np.testing.assert_allclose(np.load(out), axis)


# --- T-028: discover_background orchestrator ---------------------------------
#
# A 10-node background graph with endpoint 9. Four candidate seeds get high
# suspicion scores; each exercises a different gate:
#   0: novel, reaches 9 via 0->1->2->9, strong typology  -> SURFACES
#   3: reaches 9 via 3->4->9 but weak typology            -> dropped by Gate 3
#   5: reaches 9 via 5->6->9 but is a KNOWN member        -> dropped by Gate 1
#   7: high score, no path to any endpoint                -> dropped by Gate 2
_DISCO_EDGES = [(0, 1), (1, 2), (2, 9), (3, 4), (4, 9), (5, 6), (6, 9)]
_DISCO_N = 10
_DISCO_ENDPOINTS = [9]
_DISCO_F = 6


def _disco_model():  # type: ignore[no-untyped-def]
    import torch

    from ellip2.pu.trainer import SupervisedSubgraphModel

    torch.manual_seed(0)
    return SupervisedSubgraphModel(
        _DISCO_F, 95, set_hidden=8, set_out=4, mlp_hidden=(8,)
    )


def _disco_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scores = np.zeros(_DISCO_N, dtype=np.float64)
    scores[[0, 3, 5, 7]] = 0.9  # the four high-score seeds
    typ = np.zeros(_DISCO_N, dtype=np.float64)
    typ[0] = 1.0  # only seed 0 clears the typology gate
    known = np.array([5, 6], dtype=np.int64)  # seed 5 is an already-known member
    node_features = np.ones((_DISCO_N, _DISCO_F), dtype=np.float32)
    ei = _edge_index(_DISCO_EDGES)
    return scores, typ, known, node_features, ei


def test_discover_background_surfaces_novel_only() -> None:
    scores, typ, known, node_features, ei = _disco_inputs()
    cfg = background.BackgroundDiscoveryConfig(
        score_percentile=0.7, top_k=10, max_hops=6, typology_threshold=0.5
    )
    result = background.discover_background(
        scores, ei, _DISCO_ENDPOINTS, _DISCO_N, node_features, _disco_model(),
        typology_signal=typ, known_members=known, config=cfg,
    )
    # only candidate 0 clears all three gates
    assert [d.candidate for d in result.discovered] == [0]
    d0 = result.discovered[0]
    assert d0.cc_id == "bg0" and d0.rank == 1
    # carved member set = the ≤6-hop 0->9 path minus the endpoint
    np.testing.assert_array_equal(d0.member_idx, np.array([0, 1, 2], dtype=np.int64))
    assert 9 not in set(d0.member_idx.tolist())  # endpoint dropped
    assert 5 not in {d.candidate for d in result.discovered}  # known excluded
    assert 7 not in {d.candidate for d in result.discovered}  # no path (Gate 2)
    assert np.isfinite(d0.border_score)
    # 0, 3, 7 are candidates (5 excluded as known); only 0 & 3 reach an endpoint
    assert result.n_candidates == 3
    assert result.n_reached == 2


def test_discover_background_typology_gate_optional() -> None:
    scores, typ, known, node_features, ei = _disco_inputs()
    # no typology signal + threshold 0 => Gate 3 is a no-op; seed 3 now survives too.
    cfg = background.BackgroundDiscoveryConfig(
        score_percentile=0.7, top_k=10, max_hops=6, typology_threshold=0.0
    )
    result = background.discover_background(
        scores, ei, _DISCO_ENDPOINTS, _DISCO_N, node_features, _disco_model(),
        known_members=known, config=cfg,
    )
    assert {d.candidate for d in result.discovered} == {0, 3}
    # ranks are contiguous 1..k in descending border-score order
    assert sorted(d.rank for d in result.discovered) == [1, 2]


def test_discover_background_excludes_known_even_with_top_score() -> None:
    scores, typ, known, node_features, ei = _disco_inputs()
    scores[5] = 5.0  # make the known member the single highest score
    cfg = background.BackgroundDiscoveryConfig(
        score_percentile=0.7, top_k=1, max_hops=6, typology_threshold=0.0
    )
    result = background.discover_background(
        scores, ei, _DISCO_ENDPOINTS, _DISCO_N, node_features, _disco_model(),
        known_members=known, config=cfg,
    )
    # top_k=1 but the top scorer (5) is known -> the next eligible seed is taken
    assert 5 not in {d.candidate for d in result.discovered}


def test_discover_background_writes_existing_schema(tmp_path: Path) -> None:
    scores, typ, known, node_features, ei = _disco_inputs()
    cfg = background.BackgroundDiscoveryConfig(
        score_percentile=0.7, top_k=10, max_hops=6, typology_threshold=0.5
    )
    result = background.discover_background(
        scores, ei, _DISCO_ENDPOINTS, _DISCO_N, node_features, _disco_model(),
        typology_signal=typ, known_members=known, config=cfg,
    )
    sg_out = tmp_path / "discovered_subgraphs.parquet"
    sc_out = tmp_path / "discovered_scores.parquet"
    background.write_discovered(result, sg_out, sc_out)

    # same schema as the labeled subgraphs.parquet / *_scores.parquet
    sg = pq.read_table(sg_out)
    assert sg.column_names == ["ccId", "ccLabel", "n_members", "member_idx"]
    sc = pq.read_table(sc_out)
    assert sc.column_names == ["ccId", "score", "label", "split"]
    # member_idx round-trips and n_members agrees
    assert sg.column("ccId").to_pylist() == ["bg0"]
    assert sg.column("n_members").to_pylist() == [3]
    assert sg.column("member_idx").to_pylist() == [[0, 1, 2]]
    assert sc.column("split").to_pylist() == ["discovered"]
    assert sc.column("label").to_pylist() == [-1]


def test_discover_main_end_to_end(tmp_path: Path) -> None:
    import torch

    from ellip2.pu.trainer import SupervisedSubgraphModel, save_checkpoint

    scores, typ, known, node_features, ei = _disco_inputs()

    # Stage-0-style artifacts on disk.
    np.save(tmp_path / "cluster_scores.npy", scores)
    np.save(tmp_path / "edge_index.npy", ei)
    np.save(tmp_path / "node_features.npy", node_features)
    np.save(tmp_path / "endpoints.npy", np.array(_DISCO_ENDPOINTS, dtype=np.int64))
    np.save(tmp_path / "typology_signal.npy", typ)
    # known members 5,6 live in a labeled subgraph so known_member_idx picks them up.
    sub = _write_subgraphs(tmp_path, [[5, 6]], ["suspicious"])

    # A node-only border checkpoint (use_edges False) with the extra dict main reads.
    torch.manual_seed(0)
    model = SupervisedSubgraphModel(_DISCO_F, 95, set_hidden=8, set_out=4, mlp_hidden=(8,))
    opt = torch.optim.Adam(model.parameters())
    ckpt = tmp_path / "border_model.pt"
    save_checkpoint(
        ckpt, model, opt,
        extra={
            "framing": "supervised_subgraph_border",
            "node_dim": _DISCO_F, "edge_dim": 95, "border_cap": 64,
            "set_hidden": 8, "set_out": 4, "mlp_hidden": [8],
            "feat_mean": [0.0] * _DISCO_F, "feat_std": [1.0] * _DISCO_F,
            "use_edges": False, "edge_mean": None, "edge_std": None,
        },
    )

    sg_out = tmp_path / "discovered_subgraphs.parquet"
    sc_out = tmp_path / "discovered_scores.parquet"
    rc = background.discover_main([
        "--scores", str(tmp_path / "cluster_scores.npy"),
        "--edge-index", str(tmp_path / "edge_index.npy"),
        "--node-features", str(tmp_path / "node_features.npy"),
        "--endpoints", str(tmp_path / "endpoints.npy"),
        "--model", str(ckpt),
        "--subgraphs", str(sub),
        "--typology-signal", str(tmp_path / "typology_signal.npy"),
        "--out-subgraphs", str(sg_out),
        "--out-scores", str(sc_out),
        "--score-percentile", "0.7",
        "--typology-threshold", "0.5",
    ])
    assert rc == 0 and sg_out.is_file() and sc_out.is_file()
    sg = pq.read_table(sg_out)
    # novel candidate 0 surfaced; known members 5/6 never appear.
    assert sg.column("ccId").to_pylist() == ["bg0"]
    assert sg.column("member_idx").to_pylist() == [[0, 1, 2]]


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
        test_discover_background_writes_existing_schema,
        test_discover_main_end_to_end,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    for fn0 in (
        test_discover_background_surfaces_novel_only,
        test_discover_background_typology_gate_optional,
        test_discover_background_excludes_known_even_with_top_score,
        test_candidate_member_sets_identical_with_and_without_opt,
    ):
        fn0()
    print("ok")
