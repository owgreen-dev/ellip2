"""Unit test for the heterophily-tolerant GNN encoder (T-011).

CPU-only, synthetic, no external resources (SIGN-101). Asserts the encoder's
output embedding shape, end-to-end differentiability (``loss.backward`` populates
grads on every parameter), and the heterophily-tolerance property that gives the
module its name: the ego path keeps a node's own features even when it has no
neighbors, so an isolated node still gets a non-degenerate embedding that depends
only on itself. Runs under pytest, or standalone:
``python tests/test_encoder.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

from ellip2.pu.encoder import EgoNeighborConv, HeterophilyEncoder  # noqa: E402


def _tiny_graph(n: int = 6, in_dim: int = 43) -> Data:
    """A small directed graph with one deliberately isolated node (idx n-1)."""
    torch.manual_seed(0)
    x = torch.randn(n, in_dim)
    # A little ring 0->1->2->3->4->0 plus a chord; node n-1 has NO edges.
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4, 0],
         [1, 2, 3, 4, 0, 2]],
        dtype=torch.long,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def test_output_shape() -> None:
    g = _tiny_graph()
    enc = HeterophilyEncoder(43, 16, 8, num_layers=2)
    out = enc(g.x, g.edge_index)
    assert out.shape == (g.num_nodes, 8)
    assert out.dtype == torch.float32


def test_single_layer_shape() -> None:
    g = _tiny_graph()
    enc = HeterophilyEncoder(43, 8, num_layers=1)  # out defaults to hidden=8
    out = enc(g.x, g.edge_index)
    assert out.shape == (g.num_nodes, 8)


def test_differentiable_populates_grads() -> None:
    g = _tiny_graph()
    enc = HeterophilyEncoder(43, 16, 4, num_layers=2)
    out = enc(g.x, g.edge_index)
    loss = out.pow(2).mean()
    loss.backward()
    params = list(enc.parameters())
    assert params, "encoder has no parameters"
    # Every parameter receives a gradient (no dead paths).
    for p in params:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    assert any(p.grad.abs().sum() > 0 for p in params)


def test_isolated_node_uses_ego_only() -> None:
    """An isolated node's embedding comes purely from its own (ego) features."""
    g = _tiny_graph(n=6, in_dim=43)
    iso = g.num_nodes - 1
    conv = EgoNeighborConv(43, 5, aggr="mean")
    out = conv(g.x, g.edge_index)
    # With no in-neighbours the aggregated message is zero, so the neighbor map
    # (bias-free) contributes nothing -> output == lin_ego(x_iso).
    expected = conv.lin_ego(g.x[iso])
    torch.testing.assert_close(out[iso], expected)

    # And it must depend on the isolated node's OWN features: perturbing them
    # changes its embedding.
    x2 = g.x.clone()
    x2[iso] += 1.0
    out2 = conv(x2, g.edge_index)
    assert not torch.allclose(out[iso], out2[iso])
    # While untouched, edge-disjoint nodes far from the change are unaffected
    # only through the graph; node iso has no edges so it cannot affect others.
    assert torch.allclose(out[:iso], out2[:iso])


def test_ego_and_neighbor_weights_are_separate() -> None:
    """Zeroing the neighbor weight leaves a pure ego (per-node) transform."""
    g = _tiny_graph()
    conv = EgoNeighborConv(43, 7, aggr="mean")
    with torch.no_grad():
        conv.lin_neigh.weight.zero_()
    out = conv(g.x, g.edge_index)
    expected = conv.lin_ego(g.x)
    torch.testing.assert_close(out, expected)


def test_normalize_unit_norm() -> None:
    g = _tiny_graph()
    enc = HeterophilyEncoder(43, 16, 8, num_layers=2, normalize=True)
    out = enc(g.x, g.edge_index)
    norms = out.norm(p=2, dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


def test_num_layers_validation() -> None:
    try:
        HeterophilyEncoder(43, 16, num_layers=0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for num_layers=0")


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
