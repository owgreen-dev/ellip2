"""Unit test for the candidate-set neighbor sampler (T-012).

Exercises the kernel-free pure-``torch`` reference sampler — seed preservation,
per-hop fanout caps, sampling with replacement, determinism, and mini-batching —
plus the production :func:`build_neighbor_loader` wrapper's configuration (without
iterating it, since ``NeighborLoader``'s sampling kernel needs ``pyg-lib`` /
``torch-sparse`` which are absent from the CPU test env, SIGN-101). CPU-only,
synthetic, no external resources. Runs under pytest, or standalone:
``python tests/test_neighbor_sampling.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import NeighborLoader  # noqa: E402

from ellip2.graph.neighbor_sampling import (  # noqa: E402
    NeighborSamplingConfig,
    SampledBatch,
    build_neighbor_loader,
    iter_subgraph_batches,
    sample_subgraph,
)


def _hub_graph() -> tuple[torch.Tensor, int]:
    """Directed graph with hand-known in-neighbor structure.

    in-neighbors of 0: {1,2,3,4,5,6} (hub, 6 in-edges)
    in-neighbors of 1: {10,11}       (2)
    in-neighbors of 2: {12,13,14}    (3)
    every other node is a pure source (no in-edges).
    """
    edges = []
    for s in (1, 2, 3, 4, 5, 6):
        edges.append((s, 0))
    for s in (10, 11):
        edges.append((s, 1))
    for s in (12, 13, 14):
        edges.append((s, 2))
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long), 15


def _data() -> Data:
    ei, n = _hub_graph()
    data = Data(x=torch.arange(n).float().view(-1, 1), edge_index=ei)
    data.num_nodes = n
    return data


def test_seeds_preserved_first():
    ei, n = _hub_graph()
    batch = sample_subgraph(ei, [0], [3], num_nodes=n)
    assert isinstance(batch, SampledBatch)
    assert batch.batch_size == 1
    # Seed sits first in n_id and is recoverable.
    assert int(batch.n_id[0]) == 0
    assert batch.seed_n_id.tolist() == [0]


def test_first_hop_fanout_cap_exact():
    """Hub seed with 6 in-neighbors, cap 3 -> exactly 3 sampled (no replacement)."""
    ei, n = _hub_graph()
    g = torch.Generator().manual_seed(0)
    batch = sample_subgraph(ei, [0], [3], num_nodes=n, generator=g)
    # Seed local idx is 0; in-degree in the sampled subgraph == neighbors drawn.
    in_deg_seed = int((batch.edge_index[1] == 0).sum())
    assert in_deg_seed == 3
    # n_id = seed + 3 distinct neighbors.
    assert batch.num_nodes == 4
    assert batch.n_id.numel() == 4
    # The drawn neighbors are a subset of the hub's true in-neighbors.
    drawn = set(batch.n_id[1:].tolist())
    assert drawn.issubset({1, 2, 3, 4, 5, 6})


def test_cap_at_most_when_fewer_available():
    """cap 5 but only 2 in-neighbors -> 2 edges (no replacement, no padding)."""
    ei, n = _hub_graph()
    batch = sample_subgraph(ei, [1], [5], num_nodes=n)
    assert int((batch.edge_index[1] == 0).sum()) == 2  # node 1 has in-neighbors {10,11}
    assert set(batch.n_id[1:].tolist()) == {10, 11}


def test_two_hop_per_node_cap_respected():
    """Every expanded node's in-degree is bounded by that hop's fanout cap."""
    ei, n = _hub_graph()
    g = torch.Generator().manual_seed(7)
    cap = (3, 2)
    batch = sample_subgraph(ei, [0], cap, num_nodes=n, generator=g)
    dst = batch.edge_index[1]
    # Seed (local 0) was expanded at hop 0 -> <= cap[0] in-edges.
    assert int((dst == 0).sum()) <= cap[0]
    # Any other node was expanded at hop 1 (or never) -> <= cap[1] in-edges.
    for v in range(1, batch.num_nodes):
        assert int((dst == v).sum()) <= cap[1]
    # Hub's 3 first-hop neighbors include node 1 (in {10,11}) and node 2 (in
    # {12,13,14}); whichever were drawn, their second-hop draws stay within cap[1].


def test_replace_pads_to_cap():
    """With replacement, a node with 2 in-neighbors still yields cap draws."""
    ei, n = _hub_graph()
    g = torch.Generator().manual_seed(1)
    batch = sample_subgraph(ei, [1], [3], num_nodes=n, replace=True, generator=g)
    assert int((batch.edge_index[1] == 0).sum()) == 3  # padded to cap via repeats


def test_isolated_seed_has_no_edges():
    ei, n = _hub_graph()
    batch = sample_subgraph(ei, [9], [3, 2], num_nodes=n)  # node 9 is a pure source
    assert batch.num_nodes == 1
    assert batch.n_id.tolist() == [9]
    assert batch.edge_index.shape == (2, 0)


def test_determinism_with_generator():
    ei, n = _hub_graph()
    a = sample_subgraph(ei, [0], [3, 2], num_nodes=n,
                        generator=torch.Generator().manual_seed(42))
    b = sample_subgraph(ei, [0], [3, 2], num_nodes=n,
                        generator=torch.Generator().manual_seed(42))
    assert torch.equal(a.n_id, b.n_id)
    assert torch.equal(a.edge_index, b.edge_index)


def test_iter_batches_seeds_and_sizes():
    data = _data()
    cfg = NeighborSamplingConfig(num_neighbors=(3,), batch_size=2, shuffle=False)
    batches = list(iter_subgraph_batches(data, [0, 1, 2, 3], cfg))
    assert len(batches) == 2
    # Unshuffled: seeds are split in order, each carried through to the batch.
    assert batches[0].seed_n_id.tolist() == [0, 1]
    assert batches[1].seed_n_id.tolist() == [2, 3]
    for b in batches:
        assert b.batch_size == 2
        # Every sampled edge points into one of the two seeds (1-hop only).
        seed_locals = set(range(b.batch_size))
        assert set(b.edge_index[1].tolist()).issubset(seed_locals)


def test_build_neighbor_loader_config():
    """The production wrapper sets our conventions on a real NeighborLoader."""
    data = _data()
    cfg = NeighborSamplingConfig(num_neighbors=(15, 10), batch_size=7)
    loader = build_neighbor_loader(data, [0, 1], cfg)
    assert isinstance(loader, NeighborLoader)
    assert loader.batch_size == 7
    assert loader.node_sampler.num_neighbors.values == [15, 10]
    assert loader.input_data.node.tolist() == [0, 1]


def test_default_config_is_15_10():
    cfg = NeighborSamplingConfig()
    assert cfg.num_neighbors == (15, 10)
    assert cfg.num_hops == 2


def test_invalid_config_raises():
    for bad in (
        lambda: NeighborSamplingConfig(num_neighbors=()),
        lambda: NeighborSamplingConfig(num_neighbors=(15, -1)),
        lambda: NeighborSamplingConfig(batch_size=0),
    ):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for invalid config")


def test_bad_edge_index_shape_raises():
    try:
        sample_subgraph(torch.zeros((3, 4), dtype=torch.long), [0], [2], num_nodes=4)
    except ValueError:
        pass
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
