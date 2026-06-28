"""Stage 2 — reusable leakage-invariant guards (plan.md §7; pairs with SIGN-103).

Temporal/label leakage is the canonical Elliptic pitfall: a single hidden path
from a TEST label into a training feature silently invalidates every downstream
metric. The feature- and training-building tasks all rely on three structural
invariants, gathered here as pure, deterministic guards so any module can call the
same check (and so the test suite can pin both the clean and the leaky case):

1. **No test-split label exposed as a feature.** A node whose subgraph is in the
   persisted TEST split (from :mod:`ellip2.eval.splits`) must never contribute its
   observable label to any feature. Leakage-masked features (e.g.
   :mod:`ellip2.features.neighborhood`) zero the label indicator for such nodes;
   :func:`assert_test_labels_masked` verifies the zeroing actually happened.
2. **Disjoint positive splits.** The positives used for prior estimation, for
   training, and for held-out recall must be pairwise disjoint (plan.md §7c) —
   otherwise SCAR recall is measured on subgraphs the model already saw.
   :func:`assert_positive_splits_disjoint` enforces it.
3. **Background features do not encode subgraph membership.** A background-node
   feature column that is constant within every subgraph and distinct across them
   is a membership fingerprint: the model could read the answer off the feature
   instead of learning structure. :func:`assert_membership_not_encoded` rejects it.

Every guard raises :class:`LeakageError` (a plain ``Exception``, so it survives
``python -O``, unlike a bare ``assert``) with a message naming the offenders.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt


class LeakageError(Exception):
    """Raised when a leakage invariant is violated."""


def assert_test_labels_masked(
    exposed_label: npt.ArrayLike,
    node_subgraph: npt.ArrayLike,
    subgraph_in_test: npt.ArrayLike,
) -> None:
    """Assert no TEST-split subgraph label is exposed as a feature source.

    Args:
        exposed_label: ``(N,)`` per-node array; a nonzero entry means node ``i``
            contributed its observable subgraph label to some feature (e.g. the
            sum of the licit/suspicious label indicators fed into neighborhood
            propagation). After leakage masking this must be ``0`` for every node
            in a TEST subgraph.
        node_subgraph: ``(N,)`` int; contiguous subgraph id per node, or ``-1``
            for a background node in no labeled subgraph.
        subgraph_in_test: ``(K,)`` bool; True for subgraphs in the TEST split.

    Raises:
        LeakageError: if any node belonging to a TEST subgraph has a nonzero
            ``exposed_label``.
    """
    exposed = np.asarray(exposed_label).ravel()
    node_sg = np.asarray(node_subgraph).ravel()
    in_test = np.asarray(subgraph_in_test, dtype=bool).ravel()
    if exposed.shape[0] != node_sg.shape[0]:
        raise ValueError(
            f"exposed_label has {exposed.shape[0]} entries but node_subgraph has "
            f"{node_sg.shape[0]}"
        )

    labeled = node_sg >= 0
    if labeled.any() and int(node_sg[labeled].max()) >= in_test.shape[0]:
        raise ValueError("node_subgraph references a subgraph id beyond subgraph_in_test")

    is_test_node = np.zeros(node_sg.shape[0], dtype=bool)
    is_test_node[labeled] = in_test[node_sg[labeled]]
    bad = is_test_node & (exposed != 0)
    if bool(bad.any()):
        idx = np.flatnonzero(bad)
        raise LeakageError(
            f"{idx.size} node(s) in TEST subgraphs expose their label as a feature "
            f"(e.g. nodes {idx[:10].tolist()}); test labels must be masked to 0"
        )


def assert_positive_splits_disjoint(**named_splits: Iterable[object]) -> None:
    """Assert the named positive id collections are pairwise disjoint.

    Call as ``assert_positive_splits_disjoint(prior=..., train=..., recall=...)``;
    each value is an iterable of subgraph ids (any hashable type). Order and
    duplicates within a collection are ignored.

    Raises:
        LeakageError: if any two collections share an id, naming the offending
            pair and a few example ids.
        ValueError: if fewer than two collections are supplied.
    """
    if len(named_splits) < 2:
        raise ValueError("need at least two named splits to compare for disjointness")

    sets = {name: set(ids) for name, ids in named_splits.items()}
    names = list(sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = sets[a] & sets[b]
            if overlap:
                example = sorted(map(str, overlap))[:10]
                raise LeakageError(
                    f"positive splits {a!r} and {b!r} overlap on {len(overlap)} "
                    f"id(s) (e.g. {example}); prior / training / recall positives "
                    "must be disjoint"
                )


def assert_membership_not_encoded(
    features: npt.ArrayLike,
    node_subgraph: npt.ArrayLike,
) -> None:
    """Assert no background feature column fingerprints subgraph membership.

    A column "encodes membership" when, restricted to labeled nodes, it is
    constant within every subgraph **and** takes a distinct constant for each
    subgraph — i.e. the subgraph id is recoverable from that one feature. Genuine
    anonymized features vary within a subgraph (or collide across subgraphs), so
    they are not flagged. Only labeled nodes (``node_subgraph >= 0``) participate;
    at least two subgraphs are required for the check to be meaningful.

    Args:
        features: ``(N, F)`` float feature matrix, one row per node.
        node_subgraph: ``(N,)`` int subgraph id per node (``-1`` = background).

    Raises:
        LeakageError: if any column encodes subgraph membership.
    """
    feats = np.asarray(features, dtype=np.float64)
    if feats.ndim == 1:
        feats = feats[:, None]
    node_sg = np.asarray(node_subgraph).ravel()
    if feats.shape[0] != node_sg.shape[0]:
        raise ValueError(
            f"features has {feats.shape[0]} rows but node_subgraph has "
            f"{node_sg.shape[0]} entries"
        )

    labeled = node_sg >= 0
    sg = node_sg[labeled]
    unique_sg = np.unique(sg)
    if unique_sg.shape[0] < 2:
        return  # nothing to fingerprint with fewer than two subgraphs

    rows = feats[labeled]
    for col in range(rows.shape[1]):
        values = rows[:, col]
        per_subgraph = []
        constant_within = True
        for s in unique_sg:
            members = values[sg == s]
            if not np.all(members == members[0]):
                constant_within = False
                break
            per_subgraph.append(members[0])
        if not constant_within:
            continue
        # Constant within each subgraph; leak iff the constants are all distinct.
        if len(set(per_subgraph)) == len(per_subgraph):
            raise LeakageError(
                f"feature column {col} is constant within each subgraph and "
                "distinct across subgraphs — it fingerprints subgraph membership"
            )
