"""Background discovery — surface NEW suspicious subgraphs in the unlabeled graph.

Everything scored today is one of the 121,810 *labeled* subgraphs; the real goal
is to discover novel laundering structures among the ~48.8M unlabeled background
clusters (plan ``yes-lets-do-it-glittery-coral.md``). This module holds the
discovery machinery; T-026 seeds it with the two small helpers the orchestrator
(T-028 ``discover_background``) needs before it can carve and score candidates:

* :func:`typology_signal_from_features` — Gate 3's structural signal: the
  per-cluster ``source_sink_axis`` column (T-006 ``path_role``) exported as an
  idx-aligned ``(N,)`` array (and written to ``typology_signal.npy`` by
  :func:`main`). Mirrors :func:`ellip2.exit_paths.recover.endpoints_from_features`.
* :func:`known_member_idx` — the exclusion set: the sorted-unique union of every
  labeled subgraph's member cluster idxs from ``subgraphs.parquet``, so Gate 1
  can drop already-known clusters and rank only genuinely NEW structures.

T-027 adds :func:`candidate_member_sets` — the per-candidate subgraph *carve*: it
builds the directed background adjacency once and runs a single **global** backward
BFS from the endpoint set, then, per candidate, a cheap **forward** BFS and a
meet-in-the-middle intersection to recover the member cluster idxs of the ≤k-hop
candidate→endpoint reachability subgraph. Reuses the Stage 3 reachability BFS
(:mod:`ellip2.exit_paths.path_search`); the expensive backward sweep is computed
once and shared across all candidates.

T-028 adds :func:`discover_background` — the orchestrator that turns those pieces
into ranked, novel leads. It chains three corroboration gates over the background
clusters — **Gate 1** top-percentile cluster suspicion score *excluding* known
members (top-K), **Gate 2** a non-empty ≤k-hop carve to an endpoint (the candidate
actually reaches a licit sink), **Gate 3** a typology signal at or above threshold —
then border-scores each survivor's carved member set one at a time with a trained
node-only :class:`~ellip2.pu.trainer.SupervisedSubgraphModel`, ranks by that border
score, and writes ``discovered_subgraphs.parquet`` /
``discovered_scores.parquet`` in the SAME schema the labeled subgraphs use, so the
Stage-4 leads/investigate layer consumes discovered structures unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ellip2.exit_paths.path_search import (
    MAX_HOPS,
    _as_node_ids,
    _build_directed_adjacency,
    _hub_mask,
    bfs_reachable,
)

if TYPE_CHECKING:
    from ellip2.pu.trainer import SupervisedSubgraphModel


def typology_signal_from_features(
    cluster_features_parquet: Path,
    *,
    score_col: str = "source_sink_axis",
) -> npt.NDArray[np.float64]:
    """Idx-aligned ``(N,)`` typology signal — the ``source_sink_axis`` column of features.

    Rows are sorted by the ``idx`` key so row ``i`` is cluster ``i`` in ``[0, N)`` and
    the returned array can be indexed positionally. Raises if the score column is
    missing or ``idx`` is not a contiguous ``0..N-1`` range.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(cluster_features_parquet, columns=["idx", score_col])
    idx = table.column("idx").to_numpy(zero_copy_only=False).astype(np.int64)
    signal = table.column(score_col).to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.argsort(idx, kind="stable")
    idx, signal = idx[order], signal[order]
    if not np.array_equal(idx, np.arange(idx.size)):
        raise ValueError(
            f"feature 'idx' in {cluster_features_parquet} is not a contiguous 0..N-1 range"
        )
    return signal


def known_member_idx(
    subgraphs_path: Path,
    *,
    n_nodes: int | None = None,
) -> npt.NDArray[np.int64]:
    """Sorted-unique union of every labeled subgraph's member cluster idxs.

    The exclusion set for background discovery: Gate 1 drops these already-known
    clusters so only NEW structures surface. Optionally bounded to ``[0, n_nodes)``.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_path, columns=["member_idx"])
    parts = [
        np.asarray(members, dtype=np.int64)
        for members in table.column("member_idx").to_pylist()
        if members
    ]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    idx = np.unique(np.concatenate(parts))
    if n_nodes is not None:
        idx = idx[(idx >= 0) & (idx < n_nodes)]
    return idx


def candidate_member_sets(
    edge_index: npt.ArrayLike,
    candidate_ids: npt.ArrayLike,
    endpoints: npt.ArrayLike,
    n_nodes: int,
    *,
    max_hops: int = MAX_HOPS,
    frontier_cap: int | None = None,
    hubs: npt.ArrayLike | None = None,
) -> dict[int, npt.NDArray[np.int64]]:
    """Carve the ≤k-hop candidate→endpoint reachability subgraph of each candidate.

    For every candidate we want the node set that lies on some ≤``max_hops``
    directed path from that candidate to an endpoint — the concrete "exit-path
    subgraph" that gets border-scored and reported downstream. This is the
    meet-in-the-middle carve of :func:`ellip2.exit_paths.path_search.reachability`,
    specialised to a *single* source and run per candidate.

    The expensive half is shared: the directed adjacency is built once and a single
    **global** backward BFS is run from the endpoint set (its result is identical for
    every candidate). Only the cheap **forward** BFS — seeded from one candidate —
    runs per candidate; survivors are ``fwd.reached & back.reached &
    (fwd.hops + back.hops <= max_hops)`` (see decision #4). The endpoint ids
    themselves are dropped from each returned member set (endpoints are the target,
    not part of the discovered structure); a candidate that cannot reach any
    endpoint within ``max_hops`` maps to an **empty** array.

    Args:
        edge_index: ``(2, E)`` directed edges (Stage 0 ``edge_index.npy``); not
            symmetrised. ``E == 0`` allowed.
        candidate_ids: candidate source node ids (e.g. top-percentile PU scores,
            already excluding known members). Deduplicated internally.
        endpoints: endpoint node ids (e.g. the T-006 ``endpoint_score`` heuristic).
        n_nodes: total node count.
        max_hops: reachability horizon (default :data:`MAX_HOPS` = 6).
        frontier_cap: optional per-level frontier cap for both sweeps.
        hubs: optional hub bool-mask or id-iterable; hubs are stopped at (not
            transited) in both sweeps.

    Returns:
        ``{candidate_id: member_idx}`` — one entry per (deduplicated) candidate,
        ``member_idx`` a sorted ``int64`` array of the carve's node ids with the
        endpoint ids removed (empty when the candidate reaches no endpoint).

    Raises:
        ValueError: on a malformed ``edge_index``, out-of-range node ids, or an
            invalid ``max_hops`` / ``frontier_cap``.
    """
    if n_nodes < 0:
        raise ValueError(f"n_nodes must be non-negative, got {n_nodes}")
    ei = np.asarray(edge_index)
    if ei.ndim != 2 or ei.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {ei.shape}")
    ei = ei.astype(np.int64, copy=False)
    if ei.shape[1] and (ei.min() < 0 or ei.max() >= n_nodes):
        raise ValueError(
            f"edge endpoints out of range [0, {n_nodes}): "
            f"min {int(ei.min())}, max {int(ei.max())}"
        )

    candidates = _as_node_ids("candidate_ids", candidate_ids, n_nodes)
    endpoint_ids = _as_node_ids("endpoints", endpoints, n_nodes)
    hub_mask = _hub_mask(hubs, n_nodes)

    a = _build_directed_adjacency(ei, n_nodes)
    a_t = a.transpose().tocsr()

    # Global backward sweep from the endpoint set — identical for every candidate,
    # so compute it once and reuse.
    backward = bfs_reachable(
        a_t, endpoint_ids, max_hops=max_hops, frontier_cap=frontier_cap, hub_mask=hub_mask
    )

    drop_endpoint = np.zeros(n_nodes, dtype=bool)
    drop_endpoint[endpoint_ids] = True

    member_sets: dict[int, npt.NDArray[np.int64]] = {}
    for cand in candidates:
        forward = bfs_reachable(
            a, [int(cand)], max_hops=max_hops, frontier_cap=frontier_cap, hub_mask=hub_mask
        )
        both = forward.reached & backward.reached
        total = forward.hops + backward.hops  # valid only where both reached
        survivors = both & (total <= max_hops) & ~drop_endpoint
        member_sets[int(cand)] = np.nonzero(survivors)[0].astype(np.int64)
    return member_sets


def _as_scores(name: str, arr: npt.ArrayLike, n_nodes: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape[0] != n_nodes:
        raise ValueError(f"{name} has {a.shape[0]} entries but n_nodes is {n_nodes}")
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


@dataclass(frozen=True)
class BackgroundDiscoveryConfig:
    """Thresholds for the three-gate background-discovery funnel.

    Attributes:
        score_percentile: Gate 1 cut-off — keep only clusters whose suspicion score
            is at or above this population quantile.
        top_k: cap on the number of Gate-1 candidates carried forward (highest
            scores first).
        max_hops: reachability horizon for the per-candidate carve (Gate 2).
        frontier_cap: optional per-level BFS frontier cap.
        typology_threshold: Gate 3 — keep only candidates whose typology signal is at
            or above this value.
        border_cap: max border nodes kept per carve per side when border-scoring.
    """

    score_percentile: float = 0.9
    top_k: int = 1000
    max_hops: int = MAX_HOPS
    frontier_cap: int | None = None
    typology_threshold: float = 0.0
    border_cap: int = 64


@dataclass(frozen=True)
class DiscoveredSubgraph:
    """One corroborated, border-ranked novel lead.

    Attributes:
        cc_id: synthetic id for the discovered structure (``"bg<candidate>"``).
        candidate: the seed cluster idx (Gate-1 source of the carve).
        member_idx: the carved member cluster idxs (endpoints dropped).
        cluster_score: the candidate's Gate-1 suspicion score.
        typology_signal: the candidate's Gate-3 typology signal.
        border_score: the subgraph-border model's suspicion score for the carve.
        rank: 1-based position in the descending border-score ranking.
    """

    cc_id: str
    candidate: int
    member_idx: npt.NDArray[np.int64]
    cluster_score: float
    typology_signal: float
    border_score: float
    rank: int


@dataclass(frozen=True)
class BackgroundDiscoveryResult:
    """Outcome of :func:`discover_background`.

    Attributes:
        discovered: corroborated leads, ranked by descending border score.
        score_threshold: the absolute Gate-1 score cut-off implied by
            ``score_percentile``.
        n_candidates: Gate-1 candidates carried forward (after top-K, excluding
            known members).
        n_reached: candidates whose carve reached an endpoint (cleared Gate 2).
    """

    discovered: list[DiscoveredSubgraph]
    score_threshold: float
    n_candidates: int
    n_reached: int


def discover_background(
    scores: npt.ArrayLike,
    edge_index: npt.ArrayLike,
    endpoints: npt.ArrayLike,
    n_nodes: int,
    node_features: npt.NDArray[np.float32],
    model: SupervisedSubgraphModel,
    *,
    typology_signal: npt.ArrayLike | None = None,
    known_members: npt.ArrayLike | None = None,
    hubs: npt.ArrayLike | None = None,
    feat_mean: npt.NDArray[np.float32] | None = None,
    feat_std: npt.NDArray[np.float32] | None = None,
    edge_dim: int = 95,
    config: BackgroundDiscoveryConfig | None = None,
) -> BackgroundDiscoveryResult:
    """Surface ranked NOVEL suspicious subgraphs from the unlabeled background graph.

    Chains the three corroboration gates, carves each survivor's exit-path subgraph,
    and border-scores it with the trained node-only subgraph model:

    1. **Gate 1 (score, exclude known):** candidates are clusters whose suspicion
       ``score`` is at or above ``quantile(score, score_percentile)`` and that are NOT
       already-known labeled members (``known_members``). The top ``top_k`` by score
       are carried forward.
    2. **Gate 2 (reachability):** each candidate's ≤``max_hops`` carve to the endpoint
       set is computed (:func:`candidate_member_sets`); a candidate passes iff its
       carve is non-empty (it actually reaches a licit sink).
    3. **Gate 3 (typology):** the candidate's ``typology_signal`` must be at or above
       ``typology_threshold`` (no-op when no signal is supplied).

    Survivors' carved member sets are border-scored ONE AT A TIME with ``model`` (a
    trained :class:`~ellip2.pu.trainer.SupervisedSubgraphModel`, node channel only)
    and ranked by that border score, descending (ties broken by ascending candidate
    id).

    Args:
        scores: ``(n_nodes,)`` per-cluster suspicion scores (Gate 1).
        edge_index: ``(2, E)`` directed background edges.
        endpoints: endpoint node ids (the licit-sink heuristic) the carve must reach.
        n_nodes: total cluster count.
        node_features: ``(n_nodes, F)`` cluster features for the border model.
        model: trained subgraph-border model (already on the target device).
        typology_signal: optional ``(n_nodes,)`` Gate-3 signal (defaults to all-zero,
            i.e. Gate 3 is a no-op with ``typology_threshold <= 0``).
        known_members: cluster idxs to exclude from candidacy (the labeled members).
        hubs: optional hub mask / id-iterable passed through to the carve.
        feat_mean, feat_std: optional z-score stats for ``node_features``.
        edge_dim: internal-edge feature width the model expects (empty edge channel).
        config: :class:`BackgroundDiscoveryConfig` (defaults used when None).

    Returns:
        A :class:`BackgroundDiscoveryResult`.

    Raises:
        ValueError: on a mis-sized ``scores`` / ``typology_signal``, an out-of-range
            ``score_percentile``, a non-positive ``top_k``, or a bad ``edge_index``.
    """
    import torch  # noqa: PLC0415

    from ellip2.pu.border_assembly import (  # noqa: PLC0415
        build_subgraph_batch,
        extract_border_sets,
    )

    cfg = config or BackgroundDiscoveryConfig()
    if not 0.0 <= cfg.score_percentile <= 1.0:
        raise ValueError(f"score_percentile must be in [0, 1], got {cfg.score_percentile}")
    if cfg.top_k <= 0:
        raise ValueError(f"top_k must be positive, got {cfg.top_k}")

    s = _as_scores("scores", scores, n_nodes)
    typ = (
        np.zeros(n_nodes, dtype=np.float64)
        if typology_signal is None
        else _as_scores("typology_signal", typology_signal, n_nodes)
    )
    known_mask = np.zeros(n_nodes, dtype=bool)
    if known_members is not None:
        km = np.asarray(known_members, dtype=np.int64).ravel()
        km = km[(km >= 0) & (km < n_nodes)]
        known_mask[km] = True

    # Gate 1: top-percentile suspicion score, excluding already-known members.
    threshold = float(np.quantile(s, cfg.score_percentile)) if s.size else 0.0
    eligible = np.flatnonzero((s >= threshold) & ~known_mask)
    # top-K by descending score; eligible is ascending, so a stable sort breaks ties
    # by ascending cluster id.
    order = np.argsort(-s[eligible], kind="stable")
    candidates = eligible[order][: cfg.top_k]

    # Gate 2: per-candidate carve; a candidate reaches an endpoint iff its carve is
    # non-empty.
    member_sets = candidate_member_sets(
        edge_index,
        candidates,
        endpoints,
        n_nodes,
        max_hops=cfg.max_hops,
        frontier_cap=cfg.frontier_cap,
        hubs=hubs,
    )

    survivors: list[tuple[int, npt.NDArray[np.int64]]] = []
    n_reached = 0
    for cand in candidates:
        ms = member_sets[int(cand)]
        if ms.size == 0:  # Gate 2: unreachable within max_hops
            continue
        n_reached += 1
        if typ[cand] < cfg.typology_threshold:  # Gate 3: weak typology
            continue
        survivors.append((int(cand), ms))

    # Border-score each survivor's carved member set, one at a time (bounded memory).
    border_scores: list[float] = []
    if survivors:
        border = extract_border_sets(
            np.asarray(edge_index), [ms for _, ms in survivors], n_nodes, cap=cfg.border_cap
        )
        nf = np.asarray(node_features, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for j in range(len(survivors)):
                batch = build_subgraph_batch(
                    [j], border, nf, mean=feat_mean, std=feat_std, edge_dim=edge_dim
                )
                border_scores.append(float(torch.sigmoid(model(batch)).item()))

    # Rank by descending border score, ties broken by ascending candidate id.
    ranked = sorted(range(len(survivors)), key=lambda j: (-border_scores[j], survivors[j][0]))
    discovered = [
        DiscoveredSubgraph(
            cc_id=f"bg{survivors[j][0]}",
            candidate=survivors[j][0],
            member_idx=survivors[j][1],
            cluster_score=float(s[survivors[j][0]]),
            typology_signal=float(typ[survivors[j][0]]),
            border_score=border_scores[j],
            rank=rank,
        )
        for rank, j in enumerate(ranked, start=1)
    ]
    return BackgroundDiscoveryResult(
        discovered=discovered,
        score_threshold=threshold,
        n_candidates=int(candidates.size),
        n_reached=n_reached,
    )


def write_discovered(
    result: BackgroundDiscoveryResult,
    subgraphs_out: Path,
    scores_out: Path,
) -> None:
    """Write ``discovered_subgraphs.parquet`` + ``discovered_scores.parquet``.

    The two tables use the SAME schema as the labeled ``subgraphs.parquet`` /
    ``*_scores.parquet`` (``ccId, ccLabel, n_members, member_idx`` and
    ``ccId, score, label, split``), so the Stage-4 leads/investigate layer consumes
    discovered structures with no changes. Discovered clusters have no ground-truth
    label, so ``ccLabel`` is ``"unknown"``, ``label`` is ``-1``, and ``split`` is
    ``"discovered"``.
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    d = result.discovered
    subgraphs_out.parent.mkdir(parents=True, exist_ok=True)
    scores_out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table({
            "ccId": pa.array([x.cc_id for x in d], type=pa.string()),
            "ccLabel": pa.array(["unknown"] * len(d), type=pa.string()),
            "n_members": pa.array([int(x.member_idx.size) for x in d], type=pa.int64()),
            "member_idx": pa.array(
                [x.member_idx.tolist() for x in d], type=pa.list_(pa.int64())
            ),
        }),
        subgraphs_out,
    )
    pq.write_table(
        pa.table({
            "ccId": pa.array([x.cc_id for x in d], type=pa.string()),
            "score": pa.array([x.border_score for x in d], type=pa.float64()),
            "label": pa.array([-1] * len(d), type=pa.int64()),
            "split": pa.array(["discovered"] * len(d), type=pa.string()),
        }),
        scores_out,
    )


def discover_main(argv: Sequence[str] | None = None) -> int:
    """CLI: run background discovery from saved artifacts → discovered parquets."""
    import argparse  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from ellip2.pu.trainer import SupervisedSubgraphModel  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Background discovery: rank novel subgraphs (score→carve→border).",
    )
    p.add_argument("--scores", required=True, type=Path, help="(N,) cluster_scores.npy")
    p.add_argument("--edge-index", required=True, type=Path, help="edge_index.npy")
    p.add_argument("--node-features", required=True, type=Path, help="node_features.npy")
    p.add_argument("--endpoints", required=True, type=Path, help="(K,) endpoint ids .npy")
    p.add_argument("--model", required=True, type=Path, help="border model checkpoint .pt")
    p.add_argument("--subgraphs", required=True, type=Path,
                   help="subgraphs.parquet (known-member exclusion set)")
    p.add_argument("--typology-signal", type=Path, default=None, help="(N,) typology_signal.npy")
    p.add_argument("--out-subgraphs", required=True, type=Path)
    p.add_argument("--out-scores", required=True, type=Path)
    p.add_argument("--score-percentile", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=1000)
    p.add_argument("--max-hops", type=int, default=MAX_HOPS)
    p.add_argument("--frontier-cap", type=int, default=None)
    p.add_argument("--hub-degree", type=int, default=None,
                   help="clusters with total (in+out) degree above this are stopped at, "
                        "not transited, in the carve — bounds hub explosion on the real graph")
    p.add_argument("--typology-threshold", type=float, default=0.0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    scores = np.load(args.scores)
    n_nodes = int(scores.shape[0])
    edge_index = np.load(args.edge_index)
    node_features = np.load(args.node_features, mmap_mode="r")
    endpoints = np.load(args.endpoints)
    typ = np.load(args.typology_signal) if args.typology_signal is not None else None
    known = known_member_idx(args.subgraphs, n_nodes=n_nodes)

    hubs = None
    if args.hub_degree is not None:
        deg = (np.bincount(edge_index[0], minlength=n_nodes)
               + np.bincount(edge_index[1], minlength=n_nodes))
        hubs = np.flatnonzero(deg > args.hub_degree)
        print(f"[discover] hub-stop: {hubs.size:,} clusters with degree > {args.hub_degree}",
              flush=True)

    device = torch.device(args.device)
    ckpt = torch.load(str(args.model), map_location=device, weights_only=False)
    extra = ckpt["extra"]
    model = SupervisedSubgraphModel(
        int(extra["node_dim"]), int(extra["edge_dim"]),
        set_hidden=int(extra["set_hidden"]), set_out=int(extra["set_out"]),
        mlp_hidden=tuple(extra["mlp_hidden"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = np.asarray(extra["feat_mean"], dtype=np.float32)
    std = np.asarray(extra["feat_std"], dtype=np.float32)

    cfg = BackgroundDiscoveryConfig(
        score_percentile=args.score_percentile,
        top_k=args.top_k,
        max_hops=args.max_hops,
        frontier_cap=args.frontier_cap,
        typology_threshold=args.typology_threshold,
        border_cap=int(extra["border_cap"]),
    )
    result = discover_background(
        scores, edge_index, endpoints, n_nodes, node_features, model,
        typology_signal=typ, known_members=known, hubs=hubs,
        feat_mean=mean, feat_std=std, edge_dim=int(extra["edge_dim"]), config=cfg,
    )
    write_discovered(result, args.out_subgraphs, args.out_scores)
    print(
        f"[discover] N={n_nodes:,} candidates={result.n_candidates:,} "
        f"reached={result.n_reached:,} discovered={len(result.discovered):,} "
        f"-> {args.out_subgraphs}, {args.out_scores}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the ``(N,)`` typology signal (``source_sink_axis``) to a ``.npy``."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Background discovery: export the source_sink_axis typology signal (N,).",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--out", required=True, type=Path, help="(N,) typology_signal.npy")
    p.add_argument("--score-col", default="source_sink_axis")
    args = p.parse_args(argv)

    signal = typology_signal_from_features(args.features, score_col=args.score_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, signal)
    print(f"[typology] wrote {signal.size:,} typology signals -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
