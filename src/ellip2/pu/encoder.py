"""Stage 2 — heterophily-tolerant GNN encoder (plan.md §2/§3).

Suspicious clusters in Elliptic2 sit in a **heterophilous** neighborhood: an
illicit cluster routinely transacts with licit exchanges, mixers and ordinary
peers, so its neighbors are often the *opposite* class. A plain GCN (which folds
self-loops into a single symmetric-normalized mean and then smooths) erases the
node's own signal against that mismatched neighborhood — exactly the wrong
inductive bias here.

The fix, following GraphSAGE (Hamilton 2017) and the H2GCN heterophily insight
(Zhu 2020), is to **never mix the ego into the neighbor aggregation**. Each
:class:`EgoNeighborConv` keeps two separate linear maps::

    h_v' = W_ego · x_v  +  W_neigh · AGG_{u∈N(v)} x_u

``W_ego`` carries the node's own features straight through (so an isolated node
still gets a full, non-degenerate embedding), while ``W_neigh`` transforms the
aggregated neighbors independently. No self-loops are added — the two paths stay
distinct, which is what makes the layer heterophily-tolerant. ``AGG`` is a mean
by default (also ``max``/``sum`` via the :class:`~torch_geometric.nn.MessagePassing`
aggregation).

:class:`HeterophilyEncoder` stacks these layers with a non-linearity, dropout and
an optional final L2 normalization, producing per-node embeddings for the PU
head (T-013). Everything is pure CPU torch and runs on a tiny PyG graph.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
from torch import Tensor, nn
from torch_geometric.nn import MessagePassing

Activation = Callable[[Tensor], Tensor]


class EgoNeighborConv(MessagePassing):
    """One ego/neighbor-separated message-passing layer (GraphSAGE-style).

    Computes ``W_ego · x_v + W_neigh · AGG_{u∈N(v)} x_u``. The ego and neighbor
    transforms have independent weights and the ego path bypasses aggregation
    entirely (no self-loops), so the node's own features are preserved even when
    its neighborhood is class-mismatched (heterophily) or empty (isolated node).

    Args:
        in_channels: input feature dim.
        out_channels: output embedding dim.
        aggr: neighbor aggregation, any :class:`MessagePassing` aggr (``"mean"``
            default, also ``"max"``/``"sum"``/``"add"``).
        bias: add a bias term to the ego transform (the neighbor transform is
            kept bias-free so an isolated node's neighbor contribution is exactly
            zero).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        aggr: str = "mean",
        bias: bool = True,
    ) -> None:
        super().__init__(aggr=aggr)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_ego = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Re-initialise both linear maps (and the base aggregation state)."""
        super().reset_parameters()
        self.lin_ego.reset_parameters()
        self.lin_neigh.reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Apply the layer.

        Args:
            x: ``(N, in_channels)`` node features.
            edge_index: ``(2, E)`` long tensor; column ``[s, t]`` is an edge
                ``s -> t`` so ``t`` aggregates messages from its in-neighbour
                ``s`` (PyG flow ``source_to_target``).

        Returns:
            ``(N, out_channels)`` embeddings.
        """
        neigh = self.propagate(edge_index, x=x)
        return self.lin_ego(x) + self.lin_neigh(neigh)

    def message(self, x_j: Tensor) -> Tensor:
        # x_j = features of the source (neighbour) node of each edge; the base
        # class scatter-aggregates these per target node with ``aggr``.
        return x_j

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}({self.in_channels}, {self.out_channels}, "
            f"aggr={self.aggr!r})"
        )


class HeterophilyEncoder(nn.Module):
    """Stacked ego/neighbor-separated encoder producing per-node embeddings.

    A stack of :class:`EgoNeighborConv` layers with a non-linearity and dropout
    between them, optionally L2-normalising the final embeddings (GraphSAGE
    convention; keeps embedding norms comparable for the downstream PU head).

    Args:
        in_channels: input feature dim (43 for Elliptic2 node features).
        hidden_channels: width of the intermediate layers.
        out_channels: embedding dim. Defaults to ``hidden_channels``.
        num_layers: number of conv layers (``>= 1``). With ``1`` layer there is
            no hidden stage and the single layer maps ``in -> out``.
        aggr: neighbor aggregation passed to each layer.
        dropout: dropout probability applied to hidden activations (training
            only; ``eval()`` disables it).
        activation: elementwise non-linearity between layers (default ReLU).
        normalize: L2-normalise the output embeddings along the feature dim.
        bias: bias on each layer's ego transform.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int | None = None,
        *,
        num_layers: int = 2,
        aggr: str = "mean",
        dropout: float = 0.0,
        activation: Activation = torch.relu,
        normalize: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        out_channels = hidden_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.normalize = normalize

        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(
            EgoNeighborConv(dims[i], dims[i + 1], aggr=aggr, bias=bias)
            for i in range(num_layers)
        )

    def reset_parameters(self) -> None:
        """Re-initialise every conv layer."""
        for conv in self.convs:
            cast(EgoNeighborConv, conv).reset_parameters()

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Encode ``(N, in_channels)`` features to ``(N, out_channels)`` embeddings.

        A non-linearity and dropout follow every layer except the last; the final
        layer is linear (so the PU head sees raw embeddings), optionally L2
        normalised.
        """
        h = x
        last = self.num_layers - 1
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < last:
                h = self.activation(h)
                h = nn.functional.dropout(h, p=self.dropout, training=self.training)
        if self.normalize:
            h = nn.functional.normalize(h, p=2.0, dim=-1)
        return h
