"""Unit test for Stage 3 corroborated candidate ranking (T-017).

Asserts, on a tiny hand-built directed graph + synthetic PU scores (synthetic,
CPU-only, no external resources — SIGN-101), that the RevFilter corroboration
pipeline (plan.md §9 Stage 3 + §7) surfaces ONLY clusters that clear all three
gates — high PU score AND a valid ≤k-hop exit path AND a typology signal — and
returns them ranked by descending score.

Runs under pytest, or standalone: ``python tests/test_discover.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ellip2.discovery.discover import (  # noqa: E402
    DiscoveryConfig,
    discover_candidates,
)


def _edge_index(edges: list[tuple[int, int]]) -> np.ndarray:
    """``(2, E)`` int64 edge_index from a ``[(src, dst), ...]`` edge list."""
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(edges, dtype=np.int64).T)


# Eight clusters. Endpoint (licit sink) = 7.
#   high-score sources: 0, 1, 2, 3  (the top half by score)
#   low-score nodes:    4, 5, 6, 7  (transit / endpoint, below the gate)
# Exit paths (directed, all <= 2 hops, well within k=6):
#   0 -> 7            (1 hop)  reaches
#   3 -> 6 -> 7       (2 hops) reaches
#   2 -> 4 -> 7       (2 hops) reaches
#   1 has NO path to 7                  -> fails the exit-path gate
_EDGES = [(0, 7), (3, 6), (6, 7), (2, 4), (4, 7)]
_N = 8
_ENDPOINTS = [7]

# Top-half (>= median) scores select exactly {0, 1, 2, 3}.
_SCORES = np.array([0.90, 0.85, 0.80, 0.95, 0.10, 0.20, 0.15, 0.00])

# Typology corroborator: node 2 reaches an endpoint but its signal is below the
# threshold, so it must be filtered out; 0 and 3 clear it.
_TYPOLOGY = np.array([0.9, 0.9, 0.1, 0.8, 0.0, 0.0, 0.0, 0.0])


def _run(**overrides: object):
    cfg = DiscoveryConfig(
        score_percentile=overrides.pop("score_percentile", 0.5),  # type: ignore[arg-type]
        max_hops=overrides.pop("max_hops", 6),  # type: ignore[arg-type]
        typology_threshold=overrides.pop("typology_threshold", 0.5),  # type: ignore[arg-type]
    )
    return discover_candidates(
        _SCORES,
        _edge_index(_EDGES),
        _ENDPOINTS,
        _N,
        typology_signal=overrides.pop("typology_signal", _TYPOLOGY),
        config=cfg,
    )


# --------------------------------------------------------------------------- #
# Corroboration: only clusters clearing all three gates survive, ranked.
# --------------------------------------------------------------------------- #
def test_only_corroborated_returned_ranked() -> None:
    res = _run()
    # Gate 1 selects {0,1,2,3}; node 1 fails the exit path; node 2 fails typology.
    assert res.n_above_threshold == 4
    assert res.n_reached == 3  # 0, 2, 3 reach the endpoint; 1 does not
    nodes = [c.node for c in res.candidates]
    assert nodes == [3, 0]  # ranked by descending score (0.95, 0.90)
    assert [c.rank for c in res.candidates] == [1, 2]
    # Every survivor cleared all three gates.
    for c in res.candidates:
        assert c.score >= res.score_threshold
        assert c.typology_signal >= 0.5
        assert 0 <= c.backward_hops <= res.max_hops


def test_high_score_without_exit_path_dropped() -> None:
    # Node 1 has a high score and passes the typology gate, but no path to 7.
    res = _run()
    assert 1 not in {c.node for c in res.candidates}


def test_reaches_but_no_typology_dropped() -> None:
    # Node 2 reaches the endpoint but its typology signal (0.1) < threshold (0.5).
    res = _run()
    assert 2 not in {c.node for c in res.candidates}


def test_below_score_threshold_excluded() -> None:
    res = _run()
    survivors = {c.node for c in res.candidates}
    assert survivors.isdisjoint({4, 5, 6, 7})  # none of the low-score nodes


def test_ranking_is_descending_by_score() -> None:
    res = _run()
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# The k horizon actually bounds the exit-path gate.
# --------------------------------------------------------------------------- #
def test_max_hops_bounds_exit_path() -> None:
    # With max_hops=1 only the 1-hop path 0->7 qualifies; 3->6->7 is too long.
    res = _run(max_hops=1)
    assert [c.node for c in res.candidates] == [0]


# --------------------------------------------------------------------------- #
# Typology gate is a no-op when no signal is supplied.
# --------------------------------------------------------------------------- #
def test_no_typology_signal_passes_all_reachable() -> None:
    res = _run(typology_signal=None, typology_threshold=0.0)
    # Now node 2 is no longer filtered: every score-passing reachable node stays.
    assert [c.node for c in res.candidates] == [3, 0, 2]


# --------------------------------------------------------------------------- #
# A stricter score percentile narrows the candidate pool.
# --------------------------------------------------------------------------- #
def test_stricter_percentile_narrows_pool() -> None:
    # Top 12.5% (one node) -> only node 3 clears the score gate; it also reaches
    # and clears typology.
    res = _run(score_percentile=0.99)
    assert res.n_above_threshold == 1
    assert [c.node for c in res.candidates] == [3]


def test_score_pct_reported() -> None:
    res = _run()
    top = res.candidates[0]
    assert top.node == 3
    # Node 3 has the single largest score -> percentile 1.0 (all scores <= it).
    assert top.score_pct == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def test_bad_scores_length_raises() -> None:
    with pytest.raises(ValueError, match="scores"):
        discover_candidates(np.zeros(3), _edge_index(_EDGES), _ENDPOINTS, _N)


def test_bad_percentile_raises() -> None:
    with pytest.raises(ValueError, match="score_percentile"):
        discover_candidates(
            _SCORES, _edge_index(_EDGES), _ENDPOINTS, _N,
            config=DiscoveryConfig(score_percentile=1.5),
        )


def test_mis_sized_typology_raises() -> None:
    with pytest.raises(ValueError, match="typology_signal"):
        discover_candidates(
            _SCORES, _edge_index(_EDGES), _ENDPOINTS, _N,
            typology_signal=np.zeros(3),
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
