"""Unit test for Stage 3 bounded exit-path *reachability* (T-016).

Asserts, on tiny hand-built directed graphs (synthetic, CPU-only, no external
resources — SIGN-101), the four properties that define the module
(plan.md Resolved decision #4 + §3):

  * correct candidate->endpoint reachable set within ``k`` hops (and that the
    ``k`` horizon actually bounds it);
  * per-level ``frontier_cap`` is respected (and records ``capped_levels``);
  * a node flagged a hub is *stopped at* — recorded but never used as transit,
    so a path that only reaches the endpoint *through* a hub does not count;
  * backward ∩ forward correctness — the meet-in-the-middle ``survivors`` set
    excludes nodes that are forward-reachable but cannot themselves reach an
    endpoint within ``k`` hops.

Plus a small ``induced_subgraph`` round-trip over the survivors.

Runs under pytest, or standalone: ``python tests/test_path_search.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from scipy import sparse  # noqa: E402

from ellip2.exit_paths.path_search import (  # noqa: E402
    bfs_reachable,
    induced_subgraph,
    reachability,
)


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    """``(2, E)`` int64 edge_index from a ``[(src, dst), ...]`` edge list."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    arr = np.asarray(edges, dtype=np.int64).T
    return np.ascontiguousarray(arr)


def _adjacency(edges: list[tuple[int, int]], n: int) -> sparse.csr_matrix:
    ei = _edge_index(edges)
    data = np.ones(ei.shape[1], dtype=np.float64)
    return sparse.coo_matrix((data, (ei[0], ei[1])), shape=(n, n)).tocsr()


# --------------------------------------------------------------------------- #
# Reachability within k hops, and the k horizon as a hard bound.
# --------------------------------------------------------------------------- #
def test_reachable_within_k_hops() -> None:
    # Simple chain 0 -> 1 -> 2 -> 3, candidate source 0, endpoint 3.
    ei = _edge_index([(0, 1), (1, 2), (2, 3)])
    res = reachability(ei, sources=[0], endpoints=[3], n_nodes=4, max_hops=3)

    assert res.candidates.tolist() == [0]
    assert res.endpoints.tolist() == [3]
    # 0 reaches 3 in exactly 3 hops -> within the horizon.
    assert res.candidate_reaches.tolist() == [True]
    # Every node on the single 0->3 path lies on a <=3-hop path.
    assert res.survivors.tolist() == [True, True, True, True]


def test_k_horizon_bounds_reachability() -> None:
    ei = _edge_index([(0, 1), (1, 2), (2, 3)])
    # With only 2 hops, source 0 cannot reach endpoint 3 (distance 3).
    res = reachability(ei, sources=[0], endpoints=[3], n_nodes=4, max_hops=2)
    assert res.candidate_reaches.tolist() == [False]
    # Node 0 is not on any <=2-hop 0->endpoint path; node 3's backward sweep
    # only reaches 1 and 2, so survivors is empty.
    assert res.survivors.tolist() == [False, False, False, False]


def test_backward_distance_is_reach_distance() -> None:
    # Backward sweep from the endpoint records, for each node, the # of hops to
    # the endpoint. On 0->1->2->3 from endpoint {3}: 3@0, 2@1, 1@2, 0@3.
    a = _adjacency([(0, 1), (1, 2), (2, 3)], 4)
    back = bfs_reachable(a.transpose().tocsr(), seeds=[3], max_hops=6)
    assert back.hops.tolist() == [3, 2, 1, 0]
    assert back.reached.tolist() == [True, True, True, True]


# --------------------------------------------------------------------------- #
# Frontier cap.
# --------------------------------------------------------------------------- #
def test_frontier_cap_respected() -> None:
    # Hub fan-out: 0 -> {1,2,3,4}. A cap of 2 keeps the lowest two ids only.
    a = _adjacency([(0, 1), (0, 2), (0, 3), (0, 4)], 5)
    res = bfs_reachable(a, seeds=[0], max_hops=2, frontier_cap=2)

    assert res.frontiers[0].tolist() == [0]
    # Level 1 discovered 4 nodes but only 2 (lowest ids) survive the cap.
    assert res.frontiers[1].tolist() == [1, 2]
    assert res.capped_levels == [1]
    # 3 and 4 are dropped -> never reached.
    assert res.reached.tolist() == [True, True, True, False, False]


def test_frontier_cap_uncapped_when_below_limit() -> None:
    a = _adjacency([(0, 1), (0, 2)], 3)
    res = bfs_reachable(a, seeds=[0], max_hops=2, frontier_cap=5)
    assert res.frontiers[1].tolist() == [1, 2]
    assert res.capped_levels == []


# --------------------------------------------------------------------------- #
# Hubs are stopped at, never transited.
# --------------------------------------------------------------------------- #
def test_hub_stopped_at_not_transited() -> None:
    # Chain 0 -> 1 -> 2 -> 3 with node 2 a hub. Backward from endpoint {3}
    # reaches 2 (and records it) but must NOT expand through it to 1 / 0.
    a = _adjacency([(0, 1), (1, 2), (2, 3)], 4)
    hub_mask = np.zeros(4, dtype=bool)
    hub_mask[2] = True
    back = bfs_reachable(
        a.transpose().tocsr(), seeds=[3], max_hops=6, hub_mask=hub_mask
    )
    # 3@0, 2@1 (reached, recorded), then stopped: 1 and 0 unreached.
    assert back.reached.tolist() == [False, False, True, True]
    assert back.hops.tolist() == [-1, -1, 1, 0]


def test_hub_transit_blocks_candidate_reach() -> None:
    # 0 -> 1 -> 2 -> 3, endpoint 3, but node 2 is a hub. The only route to the
    # endpoint passes THROUGH the hub, so candidate 0 no longer reaches it.
    ei = _edge_index([(0, 1), (1, 2), (2, 3)])
    res = reachability(
        ei, sources=[0], endpoints=[3], n_nodes=4, max_hops=6, hubs=[2]
    )
    assert res.candidate_reaches.tolist() == [False]


def test_seed_hub_still_expanded() -> None:
    # A hub that is itself a seed is the origin of the sweep -> always expanded.
    a = _adjacency([(0, 1), (1, 2)], 3)
    hub_mask = np.zeros(3, dtype=bool)
    hub_mask[0] = True
    res = bfs_reachable(a, seeds=[0], max_hops=2, hub_mask=hub_mask)
    assert res.reached.tolist() == [True, True, True]


# --------------------------------------------------------------------------- #
# backward ∩ forward (meet-in-the-middle) correctness.
# --------------------------------------------------------------------------- #
def test_survivors_meet_in_the_middle() -> None:
    # 0 -> 1 -> 2 (endpoint), plus a dead-end branch 1 -> 3 (3 reaches no
    # endpoint). Forward from {0} reaches {0,1,2,3}; backward from {2} reaches
    # {0,1,2}. The survivor set is the intersection on a <=k path: {0,1,2}.
    ei = _edge_index([(0, 1), (1, 2), (1, 3)])
    res = reachability(ei, sources=[0], endpoints=[2], n_nodes=4, max_hops=3)

    assert res.forward.reached.tolist() == [True, True, True, True]
    assert res.backward.reached.tolist() == [True, True, True, False]
    # Node 3 is forward-reachable but cannot reach the endpoint -> excluded.
    assert res.survivors.tolist() == [True, True, True, False]
    assert res.candidate_reaches.tolist() == [True]


def test_survivors_respect_total_hop_budget() -> None:
    # Two routes 0->endpoint: short 0->1->4 (2 hops) and long 0->2->3->4
    # (3 hops). With max_hops=2 only the short route's nodes survive.
    ei = _edge_index([(0, 1), (1, 4), (0, 2), (2, 3), (3, 4)])
    res = reachability(ei, sources=[0], endpoints=[4], n_nodes=5, max_hops=2)
    # 0: fwd 0 + back 2 = 2; 1: 1+1=2; 4: 2+0=2 -> survive.
    # 2: fwd 1, back 2 -> 3 > 2; 3: fwd 2, back 1 -> 3 > 2 -> excluded.
    assert res.survivors.tolist() == [True, True, False, False, True]
    assert res.candidate_reaches.tolist() == [True]


# --------------------------------------------------------------------------- #
# Induced subgraph extraction over the survivors.
# --------------------------------------------------------------------------- #
def test_induced_subgraph_relabels_survivors() -> None:
    ei = _edge_index([(0, 1), (1, 2), (1, 3)])
    res = reachability(ei, sources=[0], endpoints=[2], n_nodes=4, max_hops=3)
    sub_ei, node_ids = induced_subgraph(ei, res.survivors)

    # Survivors {0,1,2} relabel to local [0,1,2]; the 1->3 edge is dropped
    # because 3 is not a survivor.
    assert node_ids.tolist() == [0, 1, 2]
    cols = {tuple(sub_ei[:, c].tolist()) for c in range(sub_ei.shape[1])}
    assert cols == {(0, 1), (1, 2)}


# --------------------------------------------------------------------------- #
# Edge cases / validation.
# --------------------------------------------------------------------------- #
def test_empty_edges() -> None:
    ei = np.zeros((2, 0), dtype=np.int64)
    res = reachability(ei, sources=[0], endpoints=[1], n_nodes=2, max_hops=6)
    assert res.candidate_reaches.tolist() == [False]
    assert res.survivors.tolist() == [False, False]


def test_validation_errors() -> None:
    ei = _edge_index([(0, 1), (1, 2)])
    with pytest.raises(ValueError):
        reachability(ei, sources=[0], endpoints=[9], n_nodes=3, max_hops=6)
    with pytest.raises(ValueError):
        reachability(ei, sources=[0], endpoints=[2], n_nodes=3, max_hops=-1)
    a = _adjacency([(0, 1)], 2)
    with pytest.raises(ValueError):
        bfs_reachable(a, seeds=[0], frontier_cap=0)


# --------------------------------------------------------------------------- #
# Precomputed step matrix (T-029): passing adjacencyᵀ skips the per-call
# transpose and must be identical to computing it internally.
# --------------------------------------------------------------------------- #
def test_step_matrix_matches_internal_transpose() -> None:
    a = _adjacency([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)], 5)
    step = a.transpose().tocsr()
    for seeds in ([0], [0, 2], [4]):
        for cap in (None, 2):
            base = bfs_reachable(a, seeds=seeds, max_hops=6, frontier_cap=cap)
            opt = bfs_reachable(
                a, seeds=seeds, max_hops=6, frontier_cap=cap, step_matrix=step
            )
            assert opt.reached.tolist() == base.reached.tolist()
            assert opt.hops.tolist() == base.hops.tolist()
            assert opt.capped_levels == base.capped_levels


def test_step_matrix_shape_mismatch_raises() -> None:
    a = _adjacency([(0, 1)], 2)
    bad = _adjacency([(0, 1)], 3)  # 3x3 step matrix for a 2x2 adjacency
    with pytest.raises(ValueError, match="step_matrix"):
        bfs_reachable(a, seeds=[0], step_matrix=bad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
