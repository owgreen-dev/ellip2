"""Stage 2 — homogeneous PyG ``Data`` from the Stage 0 memmaps (plan.md §3).

The model graph is the 49M-node / 196M-edge background graph as a single node
type and a single edge type. We deliberately build a **homogeneous**
``torch_geometric.data.Data`` (NOT ``HeteroData``): the Elliptic2 background has
one kind of cluster node and one kind of directed flow edge, and the subgraph
labels are a *membership* overlay on those same nodes, not a second node type.
The full-graph GNN is out of scope; this object feeds neighbor sampling
(T-012), so it just needs to expose ``x``, ``edge_index`` and the subgraph
membership accessors.

Two Stage 0 artifacts back it:

    node_features.npy   (N, 43) float  -> ``Data.x``        (cast to float32)
    edge_index.npy      (2, E) int32   -> ``Data.edge_index`` (cast to int64/long)

Both are opened with ``numpy`` memmap (``mmap_mode='r'``) so ``np.load`` does not
eagerly pull the whole array through RAM; the tensors are materialised contiguous
and writable from those views (a plain ``torch.from_numpy`` over a read-only
memmap would emit a non-writable warning and share the mapping). PyG requires
``edge_index`` to be ``torch.long``.

Subgraph membership comes from ``subgraphs.parquet`` (one row per labeled
connected component: ``ccId``, ``ccLabel``, ``member_idx`` = the node idxs in the
remapped Stage 0 idx space). :class:`SubgraphMembership` is the accessor layer —
member idxs by subgraph index or by ``ccId``, and a dense ``node -> subgraph``
inverse (``-1`` for unlabeled background nodes) that round-trips with the member
lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
from torch_geometric.data import Data

# numpy.load's accepted memmap modes (None reads fully into RAM).
MmapMode = Literal["r", "r+", "w+", "c"] | None

_NODE_FEATURES = "node_features.npy"
_EDGE_INDEX = "edge_index.npy"


@dataclass(frozen=True)
class SubgraphMembership:
    """Subgraph membership overlay on the homogeneous node set.

    Subgraph indices follow ``subgraphs.parquet`` row order, matching
    :func:`ellip2.features.neighborhood.load_subgraph_labels`.

    Attributes:
        ccids: ``(K,)`` original ``ccId`` per subgraph index (provenance).
        labels: ``(K,)`` ``ccLabel`` string per subgraph index.
        members: length-``K`` list; ``members[s]`` is the int64 array of node
            idxs belonging to subgraph ``s`` (in the remapped Stage 0 idx space).
    """

    ccids: list[str]
    labels: list[str]
    members: list[npt.NDArray[np.int64]]

    def __len__(self) -> int:
        return len(self.ccids)

    def index_of(self, ccid: str) -> int:
        """Subgraph index for an original ``ccId`` (raises ``KeyError`` if absent)."""
        try:
            return self.ccids.index(ccid)
        except ValueError:
            raise KeyError(ccid) from None

    def member_idx(self, subgraph: int) -> npt.NDArray[np.int64]:
        """Node idxs belonging to subgraph index ``subgraph``."""
        return self.members[subgraph]

    def members_of(self, ccid: str) -> npt.NDArray[np.int64]:
        """Node idxs belonging to the subgraph with original ``ccId``."""
        return self.members[self.index_of(ccid)]

    def node_subgraph(self, n_nodes: int) -> npt.NDArray[np.int64]:
        """Dense ``(n_nodes,)`` inverse map: node idx -> subgraph index (``-1`` if none).

        The inverse of :attr:`members`: ``node_subgraph[members[s]] == s`` for
        every subgraph ``s``. Background nodes in no labeled subgraph stay ``-1``.
        """
        out = np.full(n_nodes, -1, dtype=np.int64)
        for sid, idx in enumerate(self.members):
            if idx.size and (idx.min() < 0 or idx.max() >= n_nodes):
                raise ValueError(
                    f"subgraph {self.ccids[sid]!r} has member idx out of [0, {n_nodes})"
                )
            out[idx] = sid
        return out


def load_node_features(
    artifacts_dir: str | Path, *, mmap_mode: MmapMode = "r"
) -> npt.NDArray[np.floating]:
    """Open Stage 0 ``node_features.npy`` as a memmap (``mmap_mode='r'`` default)."""
    return np.load(Path(artifacts_dir) / _NODE_FEATURES, mmap_mode=mmap_mode)


def load_edge_index(
    artifacts_dir: str | Path, *, mmap_mode: MmapMode = "r"
) -> npt.NDArray[np.integer]:
    """Open Stage 0 ``edge_index.npy`` as a memmap (``mmap_mode='r'`` default)."""
    return np.load(Path(artifacts_dir) / _EDGE_INDEX, mmap_mode=mmap_mode)


def build_pyg_data(
    artifacts_dir: str | Path, *, mmap_mode: MmapMode = "r"
) -> Data:
    """Build a homogeneous :class:`~torch_geometric.data.Data` from Stage 0 memmaps.

    Args:
        artifacts_dir: Stage 0 output dir holding ``node_features.npy`` and
            ``edge_index.npy``.
        mmap_mode: passed to :func:`numpy.load`; ``'r'`` (default) memmaps the
            arrays read-only. Pass ``None`` to read fully into RAM.

    Returns:
        ``Data`` with ``x`` float32 ``(N, F)``, ``edge_index`` int64 ``(2, E)``,
        and ``num_nodes == N`` (so isolated nodes are still counted).

    Raises:
        ValueError: if ``edge_index`` is not ``(2, E)`` or an endpoint falls
            outside ``[0, N)``.
    """
    nf = load_node_features(artifacts_dir, mmap_mode=mmap_mode)
    ei = load_edge_index(artifacts_dir, mmap_mode=mmap_mode)

    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    n_nodes = int(nf.shape[0])
    if ei.shape[1] and (int(ei.min()) < 0 or int(ei.max()) >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"min {int(ei.min())}, max {int(ei.max())}"
        )

    # ascontiguousarray copies the (possibly read-only) memmap views into owned,
    # writable RAM with the dtype PyG expects — float32 features, int64 edges.
    x = torch.from_numpy(np.ascontiguousarray(nf, dtype=np.float32))
    edge_index = torch.from_numpy(np.ascontiguousarray(ei, dtype=np.int64))

    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = n_nodes
    return data


def load_subgraph_membership(
    subgraphs_parquet: str | Path,
) -> SubgraphMembership:
    """Load :class:`SubgraphMembership` from Stage 0 ``subgraphs.parquet``."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(
        subgraphs_parquet, columns=["ccId", "ccLabel", "member_idx"]
    )
    ccids = [str(c) for c in table.column("ccId").to_pylist()]
    labels = [str(v) for v in table.column("ccLabel").to_pylist()]
    members = [
        np.asarray(m, dtype=np.int64) for m in table.column("member_idx").to_pylist()
    ]
    return SubgraphMembership(ccids=ccids, labels=labels, members=members)
