"""Stage 2 — subgraph supervised baseline + cluster-level nnPU head + MIL max-pool.

This module wires together the two framings reconciled in plan.md Resolved
decisions #1 and #2:

* **(a) Default — supervised subgraph model.** At the subgraph level the reliable
  *licit* labels are bona-fide negatives, so this is imbalanced *supervised*
  classification, not PU. The model follows the decisive RevClassify_DS insight
  (plan.md §"Subgraph-level readout"): represent each subgraph by its **border**
  via two Deep Sets — ``DeepSets(senders) ⊕ DeepSets(receivers)`` (the outside
  nodes funding the sources / receiving from the sinks) — concatenated with
  ``[sum, mean, max]``-pooled internal node (43-d) and edge (95-d) features, fed
  to an MLP trained with **weighted BCE**. The border carries the signal because
  licit vs suspicious *internal* graphlet distributions are nearly identical.

* **(b) Optional — cluster-level nnPU head.** Genuine PU only makes sense at the
  cluster level, where the ~49M unlabeled clusters are a true unlabeled set and
  π_p is small. :class:`ClusterScorer` wraps the heterophily-tolerant encoder
  (T-011) with a linear head and is trained with the non-negative PU risk (T-009,
  Kiryo 2017) via :func:`train_cluster_nnpu`.

* **MIL max-pool.** Cluster scores collapse to a subgraph score by **max-pool**
  (:func:`max_pool_to_subgraph`): a subgraph (bag) is positive iff ≥1 member
  cluster is positive — the multiple-instance-learning / noisy-OR rule matching
  how Elliptic2 subgraphs were constructed (a component is suspicious if it
  contains one illicit→licit path). The median subgraph is 3 nodes, so max-pool
  is the right default.

Checkpointing (:func:`save_checkpoint` / :func:`load_checkpoint`) round-trips
model **and** optimizer state so training resumes exactly. Everything is pure CPU
torch — the tests smoke-train a few steps on synthetic data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ellip2.pu.nnpu_loss import (
    SurrogateLoss,
    nnpu_risk,
    nnpu_risk_for_backward,
    sigmoid_loss,
)

# Pooling operators applied over the members of each subgraph / set.
PoolOps = Sequence[str]
_DEFAULT_POOL: PoolOps = ("sum", "mean", "max")


def _mlp(
    in_dim: int, hidden: Sequence[int], out_dim: int, *, activation: type[nn.Module]
) -> nn.Sequential:
    """A plain feed-forward MLP: ``in -> hidden... -> out`` with activations between."""
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(activation())
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


def _segment_pool(
    x: Tensor, batch: Tensor, num_segments: int, ops: PoolOps
) -> Tensor:
    """Pool rows of ``x`` into ``num_segments`` groups given by ``batch``.

    Args:
        x: ``(M, F)`` element features.
        batch: ``(M,)`` long tensor; ``batch[i]`` is the segment of row ``i``.
        num_segments: number of output segments (``>= batch.max() + 1``).
        ops: any of ``"sum"``, ``"mean"``, ``"max"`` — concatenated in order.

    Returns:
        ``(num_segments, len(ops) * F)``. Empty segments contribute zeros for
        every operator (including ``max``, whose ``-inf`` is mapped to ``0``).
    """
    feat_dim = x.size(1)
    out: list[Tensor] = []
    counts = x.new_zeros(num_segments).index_add_(
        0, batch, x.new_ones(x.size(0))
    )
    summed = x.new_zeros(num_segments, feat_dim).index_add_(0, batch, x)
    for op in ops:
        if op == "sum":
            out.append(summed)
        elif op == "mean":
            out.append(summed / counts.clamp(min=1.0).unsqueeze(1))
        elif op == "max":
            m = x.new_full((num_segments, feat_dim), float("-inf"))
            m = m.scatter_reduce(
                0, batch.unsqueeze(1).expand_as(x), x, reduce="amax", include_self=True
            )
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            out.append(m)
        else:  # pragma: no cover - guarded by config
            raise ValueError(f"unknown pool op {op!r}")
    return torch.cat(out, dim=1)


class DeepSets(nn.Module):
    """Permutation-invariant Deep Sets encoder ``ρ(POOL_i φ(x_i))`` (Zaheer 2017).

    A per-element transform ``φ`` (MLP), a multi-statistic pool over the set, then
    a set transform ``ρ`` (MLP). The pool uses ``[sum, mean, max]`` by default:
    ``sum`` preserves set size, ``mean`` normalises it, ``max`` captures the most
    extreme member.

    Args:
        in_dim: per-element feature dim.
        hidden: width of ``φ``'s output (the pooled representation).
        out_dim: dim of the set embedding ``ρ`` produces.
        pool: pooling operators (subset of ``sum``/``mean``/``max``).
        activation: non-linearity for both MLPs.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        out_dim: int,
        *,
        pool: PoolOps = _DEFAULT_POOL,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.pool = tuple(pool)
        self.phi = _mlp(in_dim, [hidden], hidden, activation=activation)
        self.rho = _mlp(len(self.pool) * hidden, [hidden], out_dim, activation=activation)
        self.out_dim = out_dim

    def forward(self, x: Tensor, batch: Tensor, num_graphs: int) -> Tensor:
        """Encode a batch of sets.

        Args:
            x: ``(M, in_dim)`` elements of all sets concatenated.
            batch: ``(M,)`` long set-assignment per element.
            num_graphs: number of sets in the batch (empty sets allowed).

        Returns:
            ``(num_graphs, out_dim)`` set embeddings.
        """
        h = self.phi(x)
        pooled = _segment_pool(h, batch, num_graphs, self.pool)
        return self.rho(pooled)


@dataclass(frozen=True)
class SubgraphBatch:
    """A batch of labeled subgraphs as flat tensors with segment indices.

    Each ``*_x`` array stacks the elements of every subgraph in the batch; the
    matching ``*_batch`` long tensor assigns each element to its subgraph index
    in ``[0, num_graphs)``. Any of the four sets may be empty for a given
    subgraph (e.g. a subgraph with no border senders).

    Attributes:
        sender_x / sender_batch: border senders (outside nodes funding sources).
        receiver_x / receiver_batch: border receivers (outside nodes fed by sinks).
        node_x / node_batch: internal node (43-d) features.
        edge_x / edge_batch: internal edge (95-d) features.
        num_graphs: number of subgraphs in the batch.
    """

    sender_x: Tensor
    sender_batch: Tensor
    receiver_x: Tensor
    receiver_batch: Tensor
    node_x: Tensor
    node_batch: Tensor
    edge_x: Tensor
    edge_batch: Tensor
    num_graphs: int


class SupervisedSubgraphModel(nn.Module):
    """Border Deep Sets + pooled internal features → MLP logit (weighted-BCE model).

    The default, apples-to-apples RevClassify_DS comparator (plan.md decision #1):
    ``DeepSets(senders) ⊕ DeepSets(receivers)`` concatenated with ``[sum,mean,max]``
    pooled internal node and edge features, fed to an MLP producing one logit per
    subgraph.

    Args:
        node_dim: internal/border node feature dim (43 for Elliptic2).
        edge_dim: internal edge feature dim (95 for Elliptic2).
        set_hidden: hidden width of each border Deep Sets.
        set_out: embedding dim each border Deep Sets emits.
        mlp_hidden: hidden widths of the final classifier MLP.
        pool: pooling operators for internal-feature readout (and Deep Sets).
        activation: non-linearity used throughout.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        *,
        set_hidden: int = 32,
        set_out: int = 16,
        mlp_hidden: Sequence[int] = (32,),
        pool: PoolOps = _DEFAULT_POOL,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.pool = tuple(pool)
        self.senders = DeepSets(
            node_dim, set_hidden, set_out, pool=pool, activation=activation
        )
        self.receivers = DeepSets(
            node_dim, set_hidden, set_out, pool=pool, activation=activation
        )
        readout_dim = 2 * set_out + len(self.pool) * node_dim + len(self.pool) * edge_dim
        self.classifier = _mlp(readout_dim, mlp_hidden, 1, activation=activation)

    def forward(self, batch: SubgraphBatch) -> Tensor:
        """Return ``(num_graphs,)`` logits for the subgraphs in ``batch``."""
        b = batch.num_graphs
        s = self.senders(batch.sender_x, batch.sender_batch, b)
        r = self.receivers(batch.receiver_x, batch.receiver_batch, b)
        node_pool = _segment_pool(batch.node_x, batch.node_batch, b, self.pool)
        edge_pool = _segment_pool(batch.edge_x, batch.edge_batch, b, self.pool)
        feat = torch.cat([s, r, node_pool, edge_pool], dim=1)
        return self.classifier(feat).squeeze(-1)


class ClusterScorer(nn.Module):
    """Cluster-level scorer: heterophily encoder (T-011) + linear head → logits.

    Used with the non-negative PU risk (cluster-level framing, decision #2). The
    encoder is any module mapping ``(x, edge_index) -> (N, emb_dim)``; the linear
    head maps each cluster embedding to one logit.

    Args:
        encoder: node encoder, e.g. :class:`ellip2.pu.encoder.HeterophilyEncoder`.
        emb_dim: output dim of ``encoder`` (the head's input dim).
    """

    def __init__(self, encoder: nn.Module, emb_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(emb_dim, 1)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return ``(N,)`` per-cluster logits (larger = more suspicious)."""
        emb = self.encoder(x, edge_index)
        return self.head(emb).squeeze(-1)


@dataclass
class TrainHistory:
    """Loss trajectory of a smoke-training run.

    Attributes:
        losses: scalar loss/risk recorded once per epoch (length == epochs).
    """

    losses: list[float] = field(default_factory=list)

    @property
    def first(self) -> float:
        return self.losses[0]

    @property
    def last(self) -> float:
        return self.losses[-1]


def train_supervised(
    model: SupervisedSubgraphModel,
    batch: SubgraphBatch,
    labels: Tensor,
    *,
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    pos_weight: float | Tensor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[TrainHistory, torch.optim.Optimizer]:
    """Smoke-train the supervised subgraph model with weighted BCE.

    Args:
        model: a :class:`SupervisedSubgraphModel`.
        batch: the training :class:`SubgraphBatch`.
        labels: ``(num_graphs,)`` float targets in ``{0, 1}`` (1 = suspicious).
        epochs: number of full-batch gradient steps.
        lr: Adam learning rate.
        weight_decay: Adam L2 regularisation.
        pos_weight: weight on the positive class for the imbalanced BCE; a scalar
            or 0-dim/1-elem tensor (``count_neg / count_pos`` is a common choice).
        optimizer: reuse an existing optimizer (for resumed training); a fresh
            Adam is created when ``None``.

    Returns:
        ``(history, optimizer)`` — the loss-per-epoch history and the optimizer
        (so its state can be checkpointed).
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pw: Tensor | None
    if pos_weight is None:
        pw = None
    elif isinstance(pos_weight, Tensor):
        pw = pos_weight
    else:
        pw = torch.as_tensor(float(pos_weight))
    target = labels.float()

    history = TrainHistory()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, target, pos_weight=pw
        )
        loss.backward()
        optimizer.step()
        history.losses.append(float(loss.detach()))
    return history, optimizer


def train_cluster_nnpu(
    model: ClusterScorer,
    x: Tensor,
    edge_index: Tensor,
    positive_mask: Tensor,
    prior: float,
    *,
    epochs: int = 50,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    beta: float = 0.0,
    gamma: float = 1.0,
    surrogate: SurrogateLoss = sigmoid_loss,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[TrainHistory, torch.optim.Optimizer]:
    """Smoke-train the cluster-level scorer with the non-negative PU risk.

    Clusters in suspicious subgraphs are the positives (``positive_mask``); every
    other cluster is unlabeled. The backward pass follows Kiryo's training rule
    (:func:`nnpu_risk_for_backward`) so it ascends the over-fit negative term when
    the clamp engages, while the recorded ``history`` holds the value-correct
    :func:`nnpu_risk`.

    Args:
        model: a :class:`ClusterScorer`.
        x: ``(N, F)`` node features.
        edge_index: ``(2, E)`` long edges.
        positive_mask: ``(N,)`` bool; ``True`` for labeled-positive clusters.
        prior: class prior ``π_p ∈ (0, 1)`` (small, cluster-level — decision #2).
        epochs: number of full-graph gradient steps.
        lr / weight_decay: Adam hyper-parameters.
        beta / gamma: nnPU clamp bound and ascent scale (Kiryo 2017).
        surrogate: surrogate loss (defaults to :func:`sigmoid_loss`).
        optimizer: reuse an existing optimizer; a fresh Adam is created when None.

    Returns:
        ``(history, optimizer)``.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos = positive_mask.bool()
    unl = ~pos

    history = TrainHistory()
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(x, edge_index)
        p_logits = logits[pos]
        u_logits = logits[unl]
        loss = nnpu_risk_for_backward(
            p_logits, u_logits, prior, surrogate=surrogate, beta=beta, gamma=gamma
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            value = nnpu_risk(
                p_logits.detach(), u_logits.detach(), prior, surrogate=surrogate, beta=beta
            )
        assert isinstance(value, Tensor)  # return_parts=False → a single tensor
        history.losses.append(float(value))
    return history, optimizer


def max_pool_to_subgraph(
    member_scores: Tensor,
    member_subgraph: Tensor,
    num_subgraphs: int,
    *,
    empty_value: float = 0.0,
) -> Tensor:
    """Max-pool member-cluster scores to a per-subgraph score (MIL / noisy-OR).

    A subgraph (bag) is positive iff ≥1 member cluster is positive, so its score
    is the maximum over member scores — the multiple-instance-learning rule
    matching how Elliptic2 subgraphs were built (plan.md decision #1).

    Args:
        member_scores: ``(M,)`` per-cluster scores (probabilities or logits).
        member_subgraph: ``(M,)`` long; subgraph index of each scored cluster, in
            ``[0, num_subgraphs)``.
        num_subgraphs: number of output subgraphs.
        empty_value: score assigned to a subgraph with no scored members.

    Returns:
        ``(num_subgraphs,)`` subgraph scores.
    """
    out = member_scores.new_full((num_subgraphs,), float("-inf"))
    out = out.scatter_reduce(
        0, member_subgraph, member_scores, reduce="amax", include_self=True
    )
    return torch.where(torch.isfinite(out), out, out.new_full((), empty_value))


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model + optimizer state (and optional ``extra`` metadata) to ``path``.

    The complement of :func:`load_checkpoint`; together they round-trip a training
    run exactly (weights, Adam moments, step counters).
    """
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, str(path))


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint into ``model`` (and ``optimizer`` if given).

    Args:
        path: file written by :func:`save_checkpoint`.
        model: module to load weights into (in place).
        optimizer: optimizer to restore state into (in place); skipped if None.
        map_location: device map for :func:`torch.load` (CPU by default).

    Returns:
        The raw checkpoint dict (so callers can read ``extra``).
    """
    ckpt: dict[str, Any] = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
