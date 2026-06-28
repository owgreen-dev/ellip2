"""Unit + smoke tests for the Stage 2 trainer (T-013).

CPU-only, synthetic, no GPU / real data / S3 (SIGN-101). Covers:

* the supervised subgraph model (border Deep Sets + pooled feats → weighted BCE)
  smoke-trains and the loss decreases on separable data;
* the cluster-level nnPU head smoke-trains and its risk decreases;
* a checkpoint round-trips model **and** optimizer state exactly (identical
  outputs and identical continued-training trajectory after reload);
* the MIL max-pool of cluster scores → subgraph scores matches a hand example;
* Deep Sets / segment pooling handle empty sets.

Runs under pytest, or standalone: ``python tests/test_trainer.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch  # noqa: E402

from ellip2.pu.encoder import HeterophilyEncoder  # noqa: E402
from ellip2.pu.trainer import (  # noqa: E402
    ClusterScorer,
    DeepSets,
    SubgraphBatch,
    SupervisedSubgraphModel,
    load_checkpoint,
    max_pool_to_subgraph,
    save_checkpoint,
    train_cluster_nnpu,
    train_supervised,
)

NODE_DIM = 4
EDGE_DIM = 3


def _make_batch(labels: torch.Tensor, *, k: int = 2, seed: int = 0) -> SubgraphBatch:
    """Build a SubgraphBatch whose BORDER senders encode the label (separable)."""
    torch.manual_seed(seed)
    b = labels.numel()
    sx, sb, rx, rb, nx, nb, ex, eb = [], [], [], [], [], [], [], []
    for i in range(b):
        shift = 3.0 if labels[i] > 0.5 else -3.0
        for _ in range(k):
            sx.append(torch.randn(NODE_DIM) + shift)
            sb.append(i)
            rx.append(torch.randn(NODE_DIM))
            rb.append(i)
            nx.append(torch.randn(NODE_DIM))
            nb.append(i)
        ex.append(torch.randn(EDGE_DIM))
        eb.append(i)
    return SubgraphBatch(
        sender_x=torch.stack(sx),
        sender_batch=torch.tensor(sb, dtype=torch.long),
        receiver_x=torch.stack(rx),
        receiver_batch=torch.tensor(rb, dtype=torch.long),
        node_x=torch.stack(nx),
        node_batch=torch.tensor(nb, dtype=torch.long),
        edge_x=torch.stack(ex),
        edge_batch=torch.tensor(eb, dtype=torch.long),
        num_graphs=b,
    )


def _supervised_model() -> SupervisedSubgraphModel:
    torch.manual_seed(0)
    return SupervisedSubgraphModel(NODE_DIM, EDGE_DIM, set_hidden=16, set_out=8)


# --------------------------------------------------------------------------- #
# Deep Sets / pooling


def test_deepsets_shape_and_empty_set() -> None:
    ds = DeepSets(NODE_DIM, 8, 5)
    # Three sets; the middle one (index 1) is EMPTY.
    x = torch.randn(4, NODE_DIM)
    batch = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    out = ds(x, batch, num_graphs=3)
    assert out.shape == (3, 5)
    assert torch.isfinite(out).all()  # empty set pools to zeros, not NaN/inf


def test_supervised_forward_shape() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0])
    batch = _make_batch(labels)
    model = _supervised_model()
    logits = model(batch)
    assert logits.shape == (3,)


# --------------------------------------------------------------------------- #
# Supervised smoke training


def test_supervised_loss_decreases() -> None:
    labels = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    batch = _make_batch(labels)
    model = _supervised_model()
    n_pos = float(labels.sum())
    pos_weight = (labels.numel() - n_pos) / n_pos
    history, _ = train_supervised(
        model, batch, labels, epochs=80, lr=1e-2, pos_weight=pos_weight
    )
    assert len(history.losses) == 80
    assert all(torch.isfinite(torch.tensor(history.losses)))
    assert history.last < history.first  # learning happened


# --------------------------------------------------------------------------- #
# Cluster-level nnPU smoke training


def _cluster_setup() -> tuple[ClusterScorer, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    n = 12
    pos = torch.zeros(n, dtype=torch.bool)
    pos[:4] = True  # first four clusters are labeled positives
    # Separable features: positives shifted +, unlabeled shifted -.
    x = torch.randn(n, NODE_DIM)
    x[pos] += 3.0
    x[~pos] -= 3.0
    # Edges keep classes apart so the encoder does not smooth them together.
    edge_index = torch.tensor(
        [[0, 1, 2, 4, 5, 6, 7, 8], [1, 2, 3, 5, 6, 7, 8, 9]], dtype=torch.long
    )
    encoder = HeterophilyEncoder(NODE_DIM, 8, 8, num_layers=2)
    scorer = ClusterScorer(encoder, emb_dim=8)
    return scorer, x, edge_index, pos


def test_cluster_nnpu_risk_decreases() -> None:
    scorer, x, edge_index, pos = _cluster_setup()
    history, _ = train_cluster_nnpu(
        scorer, x, edge_index, pos, prior=0.3, epochs=100, lr=1e-2
    )
    assert len(history.losses) == 100
    assert all(torch.isfinite(torch.tensor(history.losses)))
    assert history.last < history.first


# --------------------------------------------------------------------------- #
# Checkpoint round-trip (model + optimizer)


def test_checkpoint_round_trips_model_and_optimizer() -> None:
    labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    batch = _make_batch(labels)
    model = _supervised_model()
    history, opt = train_supervised(model, batch, labels, epochs=15, lr=1e-2)

    with tempfile.TemporaryDirectory() as d:
        ckpt = Path(d) / "ckpt.pt"
        save_checkpoint(ckpt, model, opt, extra={"epoch": 15})

        model2 = _supervised_model()  # fresh (different) init...
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-2)
        meta = load_checkpoint(ckpt, model2, opt2)  # ...overwritten by load

    assert meta["extra"]["epoch"] == 15
    # Identical model: same logits on the same batch.
    with torch.no_grad():
        torch.testing.assert_close(model(batch), model2(batch))

    # Identical OPTIMIZER state: continuing training a few steps gives the exact
    # same loss trajectory only if Adam moments/step counters were restored too.
    h1, _ = train_supervised(model, batch, labels, epochs=5, optimizer=opt)
    h2, _ = train_supervised(model2, batch, labels, epochs=5, optimizer=opt2)
    torch.testing.assert_close(
        torch.tensor(h1.losses), torch.tensor(h2.losses)
    )


# --------------------------------------------------------------------------- #
# MIL max-pool: cluster scores -> subgraph scores


def test_max_pool_to_subgraph_hand_example() -> None:
    # Subgraph 0 has members scoring {0.1, 0.9, 0.3} -> 0.9 (one positive member
    # makes the bag positive); subgraph 1 {0.2, 0.05} -> 0.2; subgraph 2 {0.7};
    # subgraph 3 has NO scored members -> empty_value (0.0).
    scores = torch.tensor([0.1, 0.9, 0.3, 0.2, 0.05, 0.7])
    member_subgraph = torch.tensor([0, 0, 0, 1, 1, 2], dtype=torch.long)
    out = max_pool_to_subgraph(scores, member_subgraph, num_subgraphs=4)
    torch.testing.assert_close(out, torch.tensor([0.9, 0.2, 0.7, 0.0]))


def test_max_pool_single_positive_member_flips_bag() -> None:
    # MIL semantics: a single high-scoring member dominates the bag score.
    scores = torch.tensor([0.01, 0.02, 0.99])
    member_subgraph = torch.tensor([0, 0, 0], dtype=torch.long)
    out = max_pool_to_subgraph(scores, member_subgraph, num_subgraphs=1)
    torch.testing.assert_close(out, torch.tensor([0.99]))


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
