"""Unit tests for Stage 2 leakage-invariant guards (T-015).

Each of the three invariants is exercised with a clean case (must pass) and a
leaky case (must raise :class:`LeakageError`), per plan.md §7 / SIGN-103:

    assert_test_labels_masked        no TEST-split label exposed as a feature
    assert_positive_splits_disjoint  prior / train / recall positives disjoint
    assert_membership_not_encoded    background features don't fingerprint membership

Pure-numpy, CPU-only, no external resources (SIGN-101). Runs under pytest, or
standalone: ``python tests/test_leakage_checks.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ellip2.eval.leakage_checks import (  # noqa: E402
    LeakageError,
    assert_membership_not_encoded,
    assert_positive_splits_disjoint,
    assert_test_labels_masked,
)

# ---------------------------------------------------------------------------
# Invariant 1: no TEST-split label exposed as a feature.
# ---------------------------------------------------------------------------

# 6 nodes across 3 subgraphs; subgraph 2 is in the TEST split.
NODE_SUBGRAPH = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
SUBGRAPH_IN_TEST = np.array([False, False, True], dtype=bool)


def test_test_labels_masked_clean_passes() -> None:
    # Test-subgraph nodes (4, 5) expose nothing; train nodes may.
    exposed = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert_test_labels_masked(exposed, NODE_SUBGRAPH, SUBGRAPH_IN_TEST)


def test_test_labels_masked_background_nodes_ignored() -> None:
    # A background node (-1) exposing a label is not a test leak.
    node_sg = np.array([0, -1, 2, -1], dtype=np.int64)
    exposed = np.array([1.0, 1.0, 0.0, 1.0])
    assert_test_labels_masked(exposed, node_sg, SUBGRAPH_IN_TEST)


def test_test_labels_masked_leak_raises() -> None:
    # Node 5 is in TEST subgraph 2 yet exposes its label -> leak.
    exposed = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    with pytest.raises(LeakageError, match="TEST subgraph"):
        assert_test_labels_masked(exposed, NODE_SUBGRAPH, SUBGRAPH_IN_TEST)


def test_test_labels_masked_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="entries"):
        assert_test_labels_masked([1.0, 0.0], NODE_SUBGRAPH, SUBGRAPH_IN_TEST)


# ---------------------------------------------------------------------------
# Invariant 2: prior / train / recall positive splits are disjoint.
# ---------------------------------------------------------------------------


def test_positive_splits_disjoint_clean_passes() -> None:
    assert_positive_splits_disjoint(
        prior=[1, 2, 3],
        train=[4, 5, 6],
        recall=[7, 8, 9],
    )


def test_positive_splits_disjoint_overlap_raises() -> None:
    # 5 is in both train and recall.
    with pytest.raises(LeakageError, match="overlap"):
        assert_positive_splits_disjoint(
            prior=[1, 2, 3],
            train=[4, 5, 6],
            recall=[5, 8, 9],
        )


def test_positive_splits_disjoint_string_ids() -> None:
    assert_positive_splits_disjoint(prior={"cc1"}, train={"cc2"}, recall={"cc3"})
    with pytest.raises(LeakageError):
        assert_positive_splits_disjoint(prior={"cc1"}, train={"cc1"})


def test_positive_splits_disjoint_needs_two() -> None:
    with pytest.raises(ValueError, match="at least two"):
        assert_positive_splits_disjoint(prior=[1, 2, 3])


# ---------------------------------------------------------------------------
# Invariant 3: background features do not encode subgraph membership.
# ---------------------------------------------------------------------------

# 6 nodes, 3 subgraphs of 2 nodes each.
MEMB = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)


def test_membership_not_encoded_clean_passes() -> None:
    # Features vary within each subgraph -> not a fingerprint.
    feats = np.array(
        [
            [0.1, 9.0],
            [0.7, 9.0],  # col1 collides across subgraphs (all 9.0)
            [0.2, 9.0],
            [0.9, 9.0],
            [0.3, 9.0],
            [0.4, 9.0],
        ]
    )
    assert_membership_not_encoded(feats, MEMB)


def test_membership_not_encoded_fingerprint_raises() -> None:
    # Column 1 == subgraph id: constant within, distinct across -> leak.
    feats = np.array(
        [
            [0.1, 0.0],
            [0.7, 0.0],
            [0.2, 1.0],
            [0.9, 1.0],
            [0.3, 2.0],
            [0.4, 2.0],
        ]
    )
    with pytest.raises(LeakageError, match="fingerprint"):
        assert_membership_not_encoded(feats, MEMB)


def test_membership_not_encoded_constant_column_ok() -> None:
    # A globally constant column is constant within but NOT distinct across -> ok.
    feats = np.full((6, 1), 5.0)
    assert_membership_not_encoded(feats, MEMB)


def test_membership_not_encoded_unlabeled_ignored() -> None:
    # Background nodes (-1) don't count; the labeled rows still vary within.
    node_sg = np.array([0, 0, -1, 1, 1, -1], dtype=np.int64)
    feats = np.array([[0.1], [0.5], [42.0], [0.2], [0.9], [99.0]])
    assert_membership_not_encoded(feats, node_sg)


def test_membership_not_encoded_single_subgraph_noop() -> None:
    # Fewer than two subgraphs -> nothing to fingerprint.
    node_sg = np.array([0, 0, -1], dtype=np.int64)
    feats = np.array([[1.0], [1.0], [2.0]])
    assert_membership_not_encoded(feats, node_sg)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
