"""Stage 1 — leakage-masked neighbor label fractions (plan.md §4 + §7).

A cluster sitting next to many *known-suspicious* clusters is itself more
suspect; this module turns that intuition into per-cluster features by measuring,
for each node, the label composition of its 1-hop and 2-hop neighborhood over the
background graph. Each neighbor contributes the label of the **subgraph** it
belongs to (licit / suspicious), or ``unknown`` when its label is hidden.

Three label classes per hop, summing to 1 over the (non-empty) neighbor set:

    licit        fraction of neighbors in an observable licit subgraph
    suspicious   fraction of neighbors in an observable suspicious subgraph
    unknown      everything else — unlabeled background nodes, plus every
                 neighbor whose label was masked out for leakage safety

Leakage safety (SIGN-103 / plan.md §7) is the whole point and is enforced two
ways, so a neighbor's label can NEVER reveal the target it is used to predict:

1. **Own-subgraph mask.** When scoring node ``v``, any neighbor in *v's own
   subgraph* carries v's own label by construction; those neighbors are forced to
   ``unknown`` (their licit/suspicious contribution is subtracted out). This is
   inherently per-source: the same node counts as a real label for other sources.
2. **Test-split mask.** Any neighbor whose subgraph is in the persisted TEST
   split (from :mod:`ellip2.eval.splits`) is globally forced to ``unknown`` — test
   labels are unobservable at feature-build time.

The neighborhood is taken **undirected** (label homophily is direction-agnostic);
the directed background edges are symmetrised. ``hop1`` is the immediate neighbor
set; ``hop2`` is the ring at shortest distance *exactly* 2 (distance-1 nodes and
the node itself are excluded). The denominator is the full neighbor set size, so
masked neighbors still appear — as ``unknown`` — and the three fractions sum to 1.

Computed with :mod:`scipy.sparse` propagation (boolean SpMV / SpMM); no per-node
Python loop and no dense ``N×N`` materialisation. A node with an empty neighbor
set in a given hop takes a configurable ``empty_value`` (default ``0.0``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import sparse

# Subgraph label codes (Elliptic2 has two classes; everything else is unknown).
LICIT = 0
SUSPICIOUS = 1

CLASSES: tuple[str, ...] = ("licit", "suspicious", "unknown")
DEFAULT_HOPS: tuple[int, ...] = (1, 2)

# Hub-exclusion for the two-hop adjacency. The two-hop sparse matmul (A @ A) costs
# ~Σ deg(v)² over the whole graph, which a single mega-hub dominates: on real
# Elliptic2 one exchange cluster has degree ~1.7e7, so the naive product would take
# weeks. Nodes with degree above this cap are dropped from BOTH sides of the two-hop
# product — they neither receive nor pass on two-hop signal. That is also the right
# semantics (a cluster wired to millions of others carries no meaningful
# "neighborhood"), and mirrors the hub-exclusion already used in exit_paths.
# `None` disables the cap (small graphs / tests that want the exact product).
# Set to 100 after measuring the real graph: the two-hop matrix must be
# materialized, and its nnz grows steeply with the cap (cap=100 -> ~1.1e9 nnz;
# cap=200 -> ~2.1e9; cap=10000 -> 33e9 / 249GB). scipy's matmul holds several
# transient copies of the result, so peak RAM is ~3x the final matrix: cap=200
# OOMed a 64GB box, cap=100 peaks ~40GB. 100 excludes ~0.24% of nodes (the 120k
# highest-degree hubs) while keeping the product materializable on a 64GB box.
HUB_DEGREE_CAP: int | None = 100

# Stable column order for the assembled feature frame (T-007 joins on these),
# matching the default hops.
COLUMNS: tuple[str, ...] = tuple(
    f"hop{h}_frac_{c}" for h in DEFAULT_HOPS for c in CLASSES
)


@dataclass(frozen=True)
class SubgraphLabels:
    """Per-node / per-subgraph label arrays consumed by the feature builder.

    Attributes:
        node_subgraph: ``(N,)`` int; contiguous subgraph id per node, or ``-1``
            for a background node not in any labeled subgraph.
        subgraph_label: ``(K,)`` int in ``{LICIT, SUSPICIOUS}``, one per subgraph.
        subgraph_in_test: ``(K,)`` bool; True for subgraphs in the TEST split,
            whose labels are masked out everywhere.
        ccids: ``(K,)`` original ccId per subgraph index (provenance only).
    """

    node_subgraph: npt.NDArray[np.int64]
    subgraph_label: npt.NDArray[np.int64]
    subgraph_in_test: npt.NDArray[np.bool_]
    ccids: list[str]


def _build_adjacency(
    edge_index: npt.NDArray[np.int64], n_nodes: int
) -> sparse.csr_matrix:
    """Symmetric boolean adjacency (csr, 1.0 on support, no self-loops)."""
    src = edge_index[0]
    dst = edge_index[1]
    rows = np.concatenate([src, dst])
    cols = np.concatenate([dst, src])
    data = np.ones(rows.shape[0], dtype=np.float64)
    a = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    if a.nnz:
        a.data[:] = 1.0  # binarize: collapse parallel / mirrored edges to one
    a.setdiag(0.0)
    a.eliminate_zeros()
    return a


def _two_hop(
    a1: sparse.csr_matrix, *, hub_degree_cap: int | None = HUB_DEGREE_CAP
) -> sparse.csr_matrix:
    """Boolean adjacency of nodes at shortest distance *exactly* 2.

    Nodes whose degree exceeds ``hub_degree_cap`` are excluded from the two-hop
    product (both endpoints), bounding the ``Σ deg²`` cost that a mega-hub would
    otherwise blow up (see :data:`HUB_DEGREE_CAP`). ``None`` computes the exact,
    uncapped product.
    """
    a = a1
    if hub_degree_cap is not None:
        deg = np.asarray(a1.sum(axis=1)).ravel()
        keep = deg <= hub_degree_cap
        if not keep.all():
            # zero the rows AND columns of hub nodes: keep_diag @ A @ keep_diag
            keep_diag = sparse.diags(keep.astype(np.float64))
            a = (keep_diag @ a1 @ keep_diag).tocsr()
    reach2 = (a @ a) > 0           # length-2 walks: includes diagonal and 1-hop
    p = reach2.astype(np.float64).tocsr()
    p = p - p.multiply(a)          # drop pairs already adjacent (intersection only)
    p.setdiag(0.0)                 # drop self (v -> u -> v)
    p.eliminate_zeros()
    p.data[:] = 1.0
    return p


def _membership(node_subgraph: npt.NDArray[np.int64], n_nodes: int,
                n_subgraphs: int) -> sparse.csr_matrix:
    """One-hot node->subgraph membership (csr, N x K)."""
    valid = node_subgraph >= 0
    rows = np.nonzero(valid)[0]
    cols = node_subgraph[valid]
    data = np.ones(rows.shape[0], dtype=np.float64)
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(n_nodes, max(n_subgraphs, 1))
    ).tocsr()


def compute_neighborhood_features(
    edge_index: npt.ArrayLike,
    node_subgraph: npt.ArrayLike,
    subgraph_label: npt.ArrayLike,
    subgraph_in_test: npt.ArrayLike,
    n_nodes: int,
    *,
    hops: Sequence[int] = DEFAULT_HOPS,
    empty_value: float = 0.0,
    hub_degree_cap: int | None = HUB_DEGREE_CAP,
) -> dict[str, npt.NDArray[np.float64]]:
    """Per-cluster, leakage-masked neighbor label fractions.

    Args:
        edge_index: ``(2, E)`` directed edge endpoints (Stage 0 ``edge_index.npy``);
            symmetrised internally. ``E == 0`` allowed.
        node_subgraph: ``(N,)`` subgraph id per node (``-1`` = unlabeled background).
        subgraph_label: ``(K,)`` label code per subgraph (:data:`LICIT` /
            :data:`SUSPICIOUS`).
        subgraph_in_test: ``(K,)`` bool; subgraphs whose labels are masked out.
        n_nodes: number of clusters; output arrays are sized ``n_nodes``.
        hops: which hop distances to compute (default ``(1, 2)``).
        empty_value: fill for a node with no neighbor at that hop.

    Returns:
        Dict keyed ``hop{h}_frac_{class}`` (see :data:`CLASSES`); each value is a
        float64 ``(n_nodes,)`` array. For a non-empty neighbor set the three
        classes at a hop sum to 1.

    Raises:
        ValueError: on shape/length mismatches, out-of-range endpoints or subgraph
            ids, or a non-positive hop.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")

    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    ei = ei.astype(np.int64, copy=False)

    ns = np.asarray(node_subgraph).astype(np.int64, copy=False).ravel()
    if ns.shape[0] != n_nodes:
        raise ValueError(
            f"node_subgraph has {ns.shape[0]} entries but n_nodes={n_nodes}"
        )

    labels = np.asarray(subgraph_label).astype(np.int64, copy=False).ravel()
    in_test = np.asarray(subgraph_in_test).astype(bool, copy=False).ravel()
    n_sub = labels.shape[0]
    if in_test.shape[0] != n_sub:
        raise ValueError(
            f"subgraph_label ({n_sub}) and subgraph_in_test "
            f"({in_test.shape[0]}) length mismatch"
        )
    if ns.size and ns.max() >= n_sub:
        raise ValueError(
            f"node_subgraph references subgraph {int(ns.max())} but only "
            f"{n_sub} subgraphs were provided"
        )
    if ei.shape[1] and (ei.min() < 0 or ei.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"min {int(ei.min())}, max {int(ei.max())}"
        )
    bad_hops = [h for h in hops if h not in (1, 2)]
    if bad_hops:
        raise ValueError(f"only hops 1 and 2 are supported, got {bad_hops}")

    # Per-node OBSERVABLE label indicators: a node contributes a known licit /
    # suspicious label only if it is labeled AND not in the test split. A node's
    # own indicator doubles as its own-subgraph self-mask (every same-subgraph
    # neighbor shares it), which is why subtracting `same_h * ind` removes exactly
    # the own-subgraph leakage.
    valid = ns >= 0
    node_label = np.full(n_nodes, -1, dtype=np.int64)
    node_test = np.zeros(n_nodes, dtype=bool)
    if labels.size:
        node_label[valid] = labels[ns[valid]]
        node_test[valid] = in_test[ns[valid]]
    observable = valid & ~node_test
    lic_ind = (observable & (node_label == LICIT)).astype(np.float64)
    susp_ind = (observable & (node_label == SUSPICIOUS)).astype(np.float64)

    a1 = _build_adjacency(ei, n_nodes)
    membership = _membership(ns, n_nodes, n_sub)

    adj = {1: a1}
    if 2 in hops:
        adj[2] = _two_hop(a1, hub_degree_cap=hub_degree_cap)

    result: dict[str, npt.NDArray[np.float64]] = {}
    for h in hops:
        ah = adj[h]
        deg = np.asarray(ah.sum(axis=1)).ravel()              # |N_h(v)|
        lic = ah @ lic_ind
        susp = ah @ susp_ind
        # Same-subgraph neighbor count of v = (A_h @ M)[v, subgraph(v)].
        same = np.asarray((ah @ membership).multiply(membership).sum(axis=1)).ravel()
        lic = lic - same * lic_ind                            # remove own-subgraph
        susp = susp - same * susp_ind

        has = deg > 0
        frac_lic = np.full(n_nodes, float(empty_value), dtype=np.float64)
        frac_susp = np.full(n_nodes, float(empty_value), dtype=np.float64)
        frac_unknown = np.full(n_nodes, float(empty_value), dtype=np.float64)
        frac_lic[has] = np.clip(lic[has] / deg[has], 0.0, 1.0)
        frac_susp[has] = np.clip(susp[has] / deg[has], 0.0, 1.0)
        frac_unknown[has] = np.clip(1.0 - frac_lic[has] - frac_susp[has], 0.0, 1.0)

        result[f"hop{h}_frac_licit"] = frac_lic
        result[f"hop{h}_frac_suspicious"] = frac_susp
        result[f"hop{h}_frac_unknown"] = frac_unknown
    return result


def load_subgraph_labels(
    subgraphs_parquet: str | Path,
    split_csv: str | Path,
    n_nodes: int,
    *,
    positive_label: str = "suspicious",
) -> SubgraphLabels:
    """Build :class:`SubgraphLabels` from Stage 0 ``subgraphs.parquet`` + a split.

    Consumes the persisted split written by :mod:`ellip2.eval.splits` (its
    ``split.csv`` with ``id,label,split`` columns), so feature-build and the model
    share one TEST set (SIGN-103). Subgraph indices follow parquet row order.

    Args:
        subgraphs_parquet: path to Stage 0 ``subgraphs.parquet`` (columns
            ``ccId``, ``ccLabel``, ``member_idx`` list of node idxs).
        split_csv: path to ``<method>/split.csv`` from the split generator.
        n_nodes: total node count (sizes ``node_subgraph``).
        positive_label: ccLabel string mapped to :data:`SUSPICIOUS`.

    Raises:
        ValueError: if a member idx is out of ``[0, n_nodes)``.
    """
    import csv  # noqa: PLC0415

    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_parquet, columns=["ccId", "ccLabel", "member_idx"])
    ccids = [str(c) for c in table.column("ccId").to_pylist()]
    cclabels = [str(v) for v in table.column("ccLabel").to_pylist()]
    members = table.column("member_idx").to_pylist()

    test_ids: set[str] = set()
    with Path(split_csv).open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") == "test":
                test_ids.add(str(row["id"]))

    k = len(ccids)
    subgraph_label = np.array(
        [SUSPICIOUS if lab == positive_label else LICIT for lab in cclabels],
        dtype=np.int64,
    )
    subgraph_in_test = np.array([cc in test_ids for cc in ccids], dtype=bool)

    node_subgraph = np.full(n_nodes, -1, dtype=np.int64)
    for sid in range(k):
        idx = np.asarray(members[sid], dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= n_nodes):
            raise ValueError(
                f"subgraph {ccids[sid]!r} has member idx out of [0, {n_nodes})"
            )
        node_subgraph[idx] = sid

    return SubgraphLabels(
        node_subgraph=node_subgraph,
        subgraph_label=subgraph_label,
        subgraph_in_test=subgraph_in_test,
        ccids=ccids,
    )
