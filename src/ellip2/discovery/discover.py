"""Stage 3 — rank candidates by *corroborated* suspicion (the RevFilter pass).

plan.md §9 (Stage 3) + §7 (RevFilter corroboration; Hit-Rate / NDCG-style
ranking). The PU model (Stage 2) emits a score per cluster, but a high score
alone is a weak lead: the score is a SCAR lower bound and the unlabeled pool is
full of benign clusters. So before a cluster is surfaced to an analyst we demand
**corroboration** from three independent signals, exactly the "RevFilter"
intersection:

  1. **High PU score** — the cluster's score is in the top percentile of the
     population (``score >= quantile(scores, score_percentile)``).
  2. **A valid ≤k-hop exit path** — the cluster can actually reach a (heuristic
     licit) endpoint within ``max_hops`` hops, decided by the Stage 3
     :mod:`ellip2.exit_paths.path_search` reachability sweep (NOT path
     enumeration — SIGN-102).
  3. **A typology signal** — a structural corroborator (e.g. the source-leaning
     ``source_sink_axis`` / flow-concentration signal from Stage 1) clears a
     threshold.

A candidate survives only if **all three** hold. Survivors are then ranked by PU
score, descending (ties broken by node id ascending) — the Hit-Rate / NDCG-style
ordering an analyst works top-down. Corroboration trades recall for precision:
a high-score cluster with no exit path, or no typology signal, is dropped rather
than surfaced, because every surfaced lead costs review time.

The pipeline is pure-numpy + the scipy reachability sweep; it holds no state and
never touches the network, so it is unit-testable on a tiny synthetic graph
(SIGN-101).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ..exit_paths.path_search import MAX_HOPS, reachability


@dataclass(frozen=True)
class DiscoveryConfig:
    """Thresholds for the corroboration filter.

    Attributes:
        score_percentile: keep only clusters whose PU score is at or above this
            population quantile (``0.9`` ⇒ top 10%). In ``[0, 1]``.
        max_hops: reachability horizon for the exit-path test (default
            :data:`ellip2.exit_paths.path_search.MAX_HOPS` = 6).
        frontier_cap: optional per-level frontier cap passed through to the
            reachability sweeps (bounds hub explosion on the real graph).
        typology_threshold: minimum typology signal a survivor must clear.
    """

    score_percentile: float = 0.9
    max_hops: int = MAX_HOPS
    frontier_cap: int | None = None
    typology_threshold: float = 0.0


@dataclass(frozen=True)
class Candidate:
    """One corroborated, ranked candidate.

    Attributes:
        node: cluster (node) id.
        score: its PU score.
        score_pct: fraction of the population with score ≤ this one, in ``[0, 1]``.
        backward_hops: hop distance to the nearest endpoint (``≤ max_hops``).
        typology_signal: the structural corroborator value for this cluster.
        rank: 1-based position in the descending-score ranking.
    """

    node: int
    score: float
    score_pct: float
    backward_hops: int
    typology_signal: float
    rank: int


@dataclass(frozen=True)
class DiscoveryResult:
    """Outcome of a discovery run.

    Attributes:
        candidates: corroborated candidates, ranked by descending PU score
            (ties broken by ascending node id).
        score_threshold: the absolute score cut-off implied by
            ``score_percentile``.
        n_above_threshold: clusters that cleared the score gate (pre-corroboration).
        n_reached: of those, how many also had a valid ≤k-hop exit path.
        max_hops: the reachability horizon used.
    """

    candidates: list[Candidate] = field(default_factory=list)
    score_threshold: float = 0.0
    n_above_threshold: int = 0
    n_reached: int = 0
    max_hops: int = MAX_HOPS


def _as_scores(name: str, arr: npt.ArrayLike, n_nodes: int) -> npt.NDArray[np.float64]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    if a.shape[0] != n_nodes:
        raise ValueError(f"{name} has {a.shape[0]} entries but n_nodes is {n_nodes}")
    if a.size and not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def discover_candidates(
    scores: npt.ArrayLike,
    edge_index: npt.ArrayLike,
    endpoints: npt.ArrayLike,
    n_nodes: int,
    *,
    typology_signal: npt.ArrayLike | None = None,
    hubs: npt.ArrayLike | None = None,
    config: DiscoveryConfig | None = None,
) -> DiscoveryResult:
    """Rank candidates that clear all three corroboration gates.

    Args:
        scores: ``(n_nodes,)`` PU scores, one per cluster (higher = more suspicious).
        edge_index: ``(2, E)`` directed background edges (Stage 0 ``edge_index.npy``).
        endpoints: endpoint node ids — the (heuristic) licit sinks an exit path
            must reach, e.g. the top of the T-006 ``endpoint_score``.
        n_nodes: total cluster count.
        typology_signal: optional ``(n_nodes,)`` structural corroborator (e.g.
            ``source_sink_axis``). When ``None`` the typology gate is a no-op (all
            score-passing, reachable clusters pass it).
        hubs: optional hub bool-mask / id-iterable, passed to the reachability
            sweep (hubs are stopped at, not transited).
        config: :class:`DiscoveryConfig`; defaults applied when ``None``.

    Returns:
        A :class:`DiscoveryResult` with corroborated candidates ranked by
        descending score.

    Raises:
        ValueError: on a bad ``n_nodes``, a mis-sized ``scores`` /
            ``typology_signal``, or an out-of-range ``score_percentile``.
    """
    cfg = config if config is not None else DiscoveryConfig()
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")
    if not 0.0 <= cfg.score_percentile <= 1.0:
        raise ValueError(
            f"score_percentile must be in [0, 1], got {cfg.score_percentile}"
        )

    s = _as_scores("scores", scores, n_nodes)
    typ = (
        np.zeros(n_nodes, dtype=np.float64)
        if typology_signal is None
        else _as_scores("typology_signal", typology_signal, n_nodes)
    )

    # Gate 1: top-percentile PU score. quantile gives the absolute cut-off; a
    # cluster passes iff its score is at or above it.
    threshold = float(np.quantile(s, cfg.score_percentile))
    above = s >= threshold
    above_ids = np.nonzero(above)[0].astype(np.int64)

    # Population percentile of every score (fraction with score <= this one).
    order = np.sort(s)
    score_pct = np.searchsorted(order, s, side="right").astype(np.float64) / n_nodes

    # Gate 2: a valid <=k-hop exit path to an endpoint, decided by reachability.
    reach = reachability(
        edge_index,
        above_ids,
        endpoints,
        n_nodes,
        max_hops=cfg.max_hops,
        frontier_cap=cfg.frontier_cap,
        hubs=hubs,
    )
    reaches_by_node = {
        int(c): bool(r)
        for c, r in zip(reach.candidates, reach.candidate_reaches, strict=True)
    }
    back_hops = reach.backward.hops

    rows: list[tuple[float, int, int, float]] = []  # (score, node, hops, typology)
    n_reached = 0
    for node in above_ids:
        nid = int(node)
        if not reaches_by_node.get(nid, False):
            continue
        n_reached += 1
        # Gate 3: typology corroboration.
        if typ[nid] < cfg.typology_threshold:
            continue
        rows.append((float(s[nid]), nid, int(back_hops[nid]), float(typ[nid])))

    # Rank by descending score, ties broken by ascending node id (Hit-Rate / NDCG
    # ordering: the analyst works the list top-down).
    rows.sort(key=lambda r: (-r[0], r[1]))
    candidates = [
        Candidate(
            node=nid,
            score=score,
            score_pct=float(score_pct[nid]),
            backward_hops=hops,
            typology_signal=tsig,
            rank=i + 1,
        )
        for i, (score, nid, hops, tsig) in enumerate(rows)
    ]

    return DiscoveryResult(
        candidates=candidates,
        score_threshold=threshold,
        n_above_threshold=int(above_ids.size),
        n_reached=n_reached,
        max_hops=cfg.max_hops,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry: rank corroborated candidates from saved score/graph artifacts."""
    import argparse  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Rank corroborated AML candidates (Stage 3 RevFilter)"
    )
    p.add_argument("--scores", required=True, type=Path,
                   help="(N,) PU scores .npy (one per cluster)")
    p.add_argument("--edge-index", required=True, type=Path,
                   help="(2, E) directed edge_index .npy (Stage 0 artifact)")
    p.add_argument("--endpoints", required=True, type=Path,
                   help="endpoint node ids .npy (heuristic licit sinks)")
    p.add_argument("--typology", type=Path, default=None,
                   help="optional (N,) typology signal .npy")
    p.add_argument("--out", type=Path, default=None,
                   help="optional output .csv of ranked candidates")
    p.add_argument("--score-percentile", type=float, default=0.9)
    p.add_argument("--max-hops", type=int, default=MAX_HOPS)
    p.add_argument("--frontier-cap", type=int, default=None)
    p.add_argument("--typology-threshold", type=float, default=0.0)
    args = p.parse_args(argv)

    scores = np.load(args.scores)
    edge_index = np.load(args.edge_index)
    endpoints = np.load(args.endpoints)
    typology = np.load(args.typology) if args.typology is not None else None
    n_nodes = int(scores.shape[0])

    res = discover_candidates(
        scores,
        edge_index,
        endpoints,
        n_nodes,
        typology_signal=typology,
        config=DiscoveryConfig(
            score_percentile=args.score_percentile,
            max_hops=args.max_hops,
            frontier_cap=args.frontier_cap,
            typology_threshold=args.typology_threshold,
        ),
    )

    if args.out is not None:
        import csv  # noqa: PLC0415

        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "node", "score", "score_pct", "backward_hops",
                        "typology_signal"])
            for c in res.candidates:
                w.writerow([c.rank, c.node, c.score, c.score_pct, c.backward_hops,
                            c.typology_signal])

    print(
        f"[discover] {res.n_above_threshold} above score threshold "
        f"{res.score_threshold:.4g}, {res.n_reached} with an exit path, "
        f"{len(res.candidates)} corroborated candidates"
    )
    return 0
