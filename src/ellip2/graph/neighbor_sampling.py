"""Stage 2 — neighbor sampling over the candidate set (plan.md §3).

The 49M-node background graph is far too large to run a full-graph GNN, so we
score only the **candidate set** (the subgraphs we want to label) and build each
candidate's computation graph by **fanout-capped neighbor sampling**: from a seed
node, sample at most ``num_neighbors[0]`` of its 1-hop neighbors, then at most
``num_neighbors[1]`` of each of those nodes' neighbors, and so on. The seeds are
always the candidate nodes — we never expand the frontier from arbitrary
background nodes.

Two layers live here:

* :func:`build_neighbor_loader` — the production path. A thin wrapper over
  :class:`torch_geometric.loader.NeighborLoader` that fixes our conventions
  (default fanout ``[15, 10]``, ``input_nodes`` = the candidate set) from a
  single :class:`NeighborSamplingConfig`. On a GPU box with ``pyg-lib`` /
  ``torch-sparse`` installed this is what feeds the trainer.

* :func:`sample_subgraph` / :func:`iter_subgraph_batches` — a small, dependency
  -free reference sampler in pure ``torch``. ``NeighborLoader``'s sampling kernel
  requires the ``pyg-lib`` / ``torch-sparse`` C++ extensions, which are not part
  of the CPU test environment (SIGN-101). This sampler reproduces the same
  fanout-capped, seed-preserving semantics so the sampling logic is exercised on
  CPU with no external kernels. It also documents, in code, exactly what fanout
  caps mean.

Direction convention matches :mod:`ellip2.graph.pyg_data`: an ``edge_index``
column ``[s, t]`` is the directed edge ``s -> t`` and, under PyG's
``source_to_target`` flow, node ``t`` aggregates messages from its in-neighbor
``s``. Sampling a seed ``v`` therefore draws from its **in-neighbors**
``{s : (s -> v) in E}`` — the nodes ``v`` would aggregate from.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy.typing as npt
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

# Default 2-hop fanout: 15 first-hop neighbors, 10 of each at the second hop.
_DEFAULT_NUM_NEIGHBORS = (15, 10)


@dataclass(frozen=True)
class NeighborSamplingConfig:
    """Configuration shared by the production loader and the reference sampler.

    Attributes:
        num_neighbors: per-hop fanout cap; ``len`` is the number of hops. Each
            entry must be ``>= 0`` (``0`` samples no neighbors at that hop, ``-1``
            is rejected — use the production loader directly for "all neighbors").
            Defaults to ``[15, 10]``.
        batch_size: number of seed nodes per mini-batch.
        shuffle: shuffle the seed order each epoch (production loader only).
        replace: sample neighbors with replacement.
    """

    num_neighbors: tuple[int, ...] = _DEFAULT_NUM_NEIGHBORS
    batch_size: int = 512
    shuffle: bool = False
    replace: bool = False

    def __post_init__(self) -> None:
        if not self.num_neighbors:
            raise ValueError("num_neighbors must list at least one hop")
        if any(k < 0 for k in self.num_neighbors):
            raise ValueError(f"num_neighbors must be >= 0, got {self.num_neighbors}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    @property
    def num_hops(self) -> int:
        return len(self.num_neighbors)


@dataclass(frozen=True)
class SampledBatch:
    """A sampled mini-batch in the :class:`NeighborLoader` layout.

    Attributes:
        n_id: ``(num_sampled,)`` global node idxs in the sampled subgraph. The
            first :attr:`batch_size` entries are the seed nodes, in input order
            (the same convention as ``NeighborLoader``).
        batch_size: number of seed nodes.
        edge_index: ``(2, E')`` directed edges **relabeled to local** indices
            (``0..num_sampled-1``), one per sampled neighbor draw; column
            ``[s, t]`` keeps the ``s -> t`` direction.
        num_nodes: ``num_sampled`` (so isolated seeds are still counted).
    """

    n_id: Tensor
    batch_size: int
    edge_index: Tensor
    num_nodes: int

    @property
    def seed_n_id(self) -> Tensor:
        """Global idxs of the seed nodes (``n_id[:batch_size]``)."""
        return self.n_id[: self.batch_size]


def _as_seed_tensor(input_nodes: Tensor | npt.NDArray | list[int]) -> Tensor:
    """Coerce a candidate set to a 1-D ``long`` tensor of node idxs."""
    seeds = torch.as_tensor(input_nodes, dtype=torch.long).reshape(-1)
    return seeds


def build_neighbor_loader(
    data: Data,
    input_nodes: Tensor | npt.NDArray | list[int],
    config: NeighborSamplingConfig | None = None,
) -> NeighborLoader:
    """Build a :class:`~torch_geometric.loader.NeighborLoader` over the candidates.

    The production sampling path. ``input_nodes`` are the **seed** (candidate)
    nodes — the loader expands each seed's neighborhood with the per-hop fanout
    caps in ``config`` and never seeds from arbitrary background nodes.

    Note:
        ``NeighborLoader``'s sampler needs the ``pyg-lib`` / ``torch-sparse``
        kernels, which are absent from the CPU test env. Use
        :func:`iter_subgraph_batches` for kernel-free CPU sampling.

    Args:
        data: homogeneous graph from :func:`ellip2.graph.pyg_data.build_pyg_data`.
        input_nodes: the candidate seed node idxs.
        config: sampling config; defaults to :class:`NeighborSamplingConfig`.

    Returns:
        A configured ``NeighborLoader``.
    """
    cfg = config or NeighborSamplingConfig()
    seeds = _as_seed_tensor(input_nodes)
    return NeighborLoader(
        data,
        num_neighbors=list(cfg.num_neighbors),
        input_nodes=seeds,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        replace=cfg.replace,
    )


def sample_subgraph(
    edge_index: Tensor,
    seeds: Tensor | npt.NDArray | list[int],
    num_neighbors: tuple[int, ...] | list[int],
    *,
    num_nodes: int,
    replace: bool = False,
    generator: torch.Generator | None = None,
) -> SampledBatch:
    """Fanout-capped k-hop neighbor sampling in pure ``torch`` (CPU, kernel-free).

    Reference implementation of ``NeighborLoader``'s directional sampling: the
    seeds are preserved (and placed first in ``n_id``), and for every node in the
    current hop's frontier at most ``num_neighbors[h]`` of its **in-neighbors**
    (``s`` such that ``s -> node``) are drawn at hop ``h``.

    Args:
        edge_index: ``(2, E)`` directed edges, ``long``; column ``[s, t]`` is
            ``s -> t``.
        seeds: candidate seed node idxs.
        num_neighbors: per-hop fanout caps; ``len`` is the number of hops.
        num_nodes: total node count (for validation / adjacency sizing).
        replace: sample with replacement when a node has more in-neighbors than
            the cap.
        generator: optional ``torch.Generator`` for reproducible draws.

    Returns:
        A :class:`SampledBatch`.

    Raises:
        ValueError: if ``edge_index`` is not ``(2, E)`` or ``num_neighbors`` empty.
    """
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(
            f"edge_index must have shape (2, E), got {tuple(edge_index.shape)}"
        )
    if not len(num_neighbors):
        raise ValueError("num_neighbors must list at least one hop")

    src = edge_index[0].to(torch.long)
    dst = edge_index[1].to(torch.long)

    seed_t = _as_seed_tensor(seeds)
    batch_size = int(seed_t.numel())

    # global idx -> local idx; seeds occupy the first ``batch_size`` slots.
    node_order: list[int] = []
    local_of: dict[int, int] = {}
    for g in seed_t.tolist():
        if g not in local_of:
            local_of[g] = len(node_order)
            node_order.append(g)

    edges_src: list[int] = []
    edges_dst: list[int] = []

    frontier = list(local_of.keys())  # global ids currently being expanded
    for cap in num_neighbors:
        next_frontier: list[int] = []
        for v in frontier:
            in_neigh = src[dst == v]  # global in-neighbors of v
            n_avail = int(in_neigh.numel())
            if n_avail == 0 or cap == 0:
                continue
            if replace:
                pick = torch.randint(n_avail, (cap,), generator=generator)
                chosen = in_neigh[pick]
            elif n_avail <= cap:
                chosen = in_neigh
            else:
                perm = torch.randperm(n_avail, generator=generator)[:cap]
                chosen = in_neigh[perm]
            for u in chosen.tolist():
                if u not in local_of:
                    local_of[u] = len(node_order)
                    node_order.append(u)
                    next_frontier.append(u)
                edges_src.append(local_of[u])
                edges_dst.append(local_of[v])
        # de-dup frontier so each node is expanded once at the next hop
        frontier = list(dict.fromkeys(next_frontier))

    n_id = torch.tensor(node_order, dtype=torch.long)
    if edges_src:
        local_edges = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    else:
        local_edges = torch.empty((2, 0), dtype=torch.long)
    return SampledBatch(
        n_id=n_id,
        batch_size=batch_size,
        edge_index=local_edges,
        num_nodes=len(node_order),
    )


def iter_subgraph_batches(
    data: Data,
    input_nodes: Tensor | npt.NDArray | list[int],
    config: NeighborSamplingConfig | None = None,
    *,
    generator: torch.Generator | None = None,
) -> Iterator[SampledBatch]:
    """Yield kernel-free :class:`SampledBatch` mini-batches over the candidate set.

    Splits ``input_nodes`` into batches of ``config.batch_size`` (optionally
    shuffled) and samples each via :func:`sample_subgraph`. The CPU/test-friendly
    counterpart to :func:`build_neighbor_loader`.

    Args:
        data: homogeneous graph (provides ``edge_index`` and ``num_nodes``).
        input_nodes: candidate seed node idxs.
        config: sampling config; defaults to :class:`NeighborSamplingConfig`.
        generator: optional ``torch.Generator`` for reproducible shuffling/draws.

    Yields:
        One :class:`SampledBatch` per mini-batch of seeds.
    """
    cfg = config or NeighborSamplingConfig()
    seeds = _as_seed_tensor(input_nodes)
    if cfg.shuffle:
        seeds = seeds[torch.randperm(seeds.numel(), generator=generator)]

    n_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.x.size(0))
    edge_index = data.edge_index
    for start in range(0, int(seeds.numel()), cfg.batch_size):
        batch_seeds = seeds[start : start + cfg.batch_size]
        yield sample_subgraph(
            edge_index,
            batch_seeds,
            cfg.num_neighbors,
            num_nodes=n_nodes,
            replace=cfg.replace,
            generator=generator,
        )
