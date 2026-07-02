"""Unit test for Stage 1 leakage-masked neighbor label fractions (T-004).

Builds a tiny labeled graph and asserts 1-hop / 2-hop licit / suspicious /
unknown fractions vs values worked out by hand, then asserts the two leakage
invariants that are the point of this feature (SIGN-103):

  * a neighbor in the node's OWN subgraph never leaks its (shared) label;
  * a neighbor whose subgraph is in the TEST split never leaks its label.

Each invariant is checked against a positive control (flip the mask off and the
fraction changes), so the test fails loudly if masking is ever dropped. A small
end-to-end loader test wires the persisted split from ``ellip2.eval.splits`` into
``load_subgraph_labels``. Synthetic, CPU-only, no external resources (SIGN-101).

Runs under pytest, or standalone: ``python tests/test_neighborhood.py``.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402

from ellip2.features.neighborhood import (  # noqa: E402
    COLUMNS,
    LICIT,
    SUSPICIOUS,
    compute_neighborhood_features,
    load_subgraph_labels,
)

# Hand-built directed graph over 6 nodes (symmetrised for neighborhood).
#   edges: 0->1, 0->2, 0->5, 2->3, 2->4
# Subgraph membership / labels / split:
#   sg0 = {0,1}  licit       train
#   sg1 = {2,3}  suspicious  train
#   sg2 = {4}    suspicious  TEST   (masked everywhere)
#   node 5       unlabeled background (-1)
EDGE_INDEX = np.array([[0, 0, 0, 2, 2], [1, 2, 5, 3, 4]], dtype=np.int32)
N = 6
NODE_SUBGRAPH = np.array([0, 0, 1, 1, 2, -1], dtype=np.int64)
SUBGRAPH_LABEL = np.array([LICIT, SUSPICIOUS, SUSPICIOUS], dtype=np.int64)
SUBGRAPH_IN_TEST = np.array([False, False, True], dtype=bool)


def _features(**kw) -> dict[str, np.ndarray]:
    return compute_neighborhood_features(
        EDGE_INDEX, NODE_SUBGRAPH, SUBGRAPH_LABEL, SUBGRAPH_IN_TEST, N, **kw
    )


def test_hub_exclusion_drops_twohop_through_hub():
    # Star graph: hub 0 <-> leaves 1,2,3,4. Every leaf pair is 2-hop ONLY via hub 0.
    #   leaves 1,2 -> sg0 (licit), leaves 3,4 -> sg1 (suspicious); hub 0 = background.
    ei = np.array([[0, 0, 0, 0], [1, 2, 3, 4]], dtype=np.int64)
    n = 5
    ns = np.array([-1, 0, 0, 1, 1], dtype=np.int64)
    labels = np.array([LICIT, SUSPICIOUS], dtype=np.int64)
    in_test = np.array([False, False], dtype=bool)

    # Uncapped: leaf 1 reaches suspicious leaves 3,4 at two hops through the hub.
    full = compute_neighborhood_features(
        ei, ns, labels, in_test, n, hub_degree_cap=None, empty_value=-1.0
    )
    assert full["hop2_frac_suspicious"][1] > 0.0

    # Cap below the hub's degree (4): hub 0 is excluded from BOTH sides, so no
    # two-hop paths survive -> every node's hop2 neighborhood is empty (fill).
    capped = compute_neighborhood_features(
        ei, ns, labels, in_test, n, hub_degree_cap=3, empty_value=-1.0
    )
    for i in range(n):
        assert capped["hop2_frac_licit"][i] == -1.0
        assert capped["hop2_frac_suspicious"][i] == -1.0
        assert capped["hop2_frac_unknown"][i] == -1.0
    # One-hop features are unaffected by the two-hop cap.
    assert np.allclose(full["hop1_frac_suspicious"], capped["hop1_frac_suspicious"])


def test_columns_and_shapes():
    f = _features()
    assert set(f) == set(COLUMNS)
    assert all(f[c].shape == (N,) for c in COLUMNS)
    assert all(f[c].dtype == np.float64 for c in COLUMNS)


def test_fractions_sum_to_one_on_nonempty():
    f = _features()
    for h in (1, 2):
        tot = (
            f[f"hop{h}_frac_licit"]
            + f[f"hop{h}_frac_suspicious"]
            + f[f"hop{h}_frac_unknown"]
        )
        # Every node here has at least one 1-hop and one 2-hop neighbor.
        assert np.allclose(tot, 1.0)


def test_hop1_hand_computed():
    f = _features()
    # Node 0 (sg0): neighbors {1 own-sg->mask, 2 suspicious, 5 unlabeled}. deg 3.
    assert np.isclose(f["hop1_frac_suspicious"][0], 1 / 3)
    assert np.isclose(f["hop1_frac_licit"][0], 0.0)
    assert np.isclose(f["hop1_frac_unknown"][0], 2 / 3)
    # Node 2 (sg1): neighbors {0 licit, 3 own-sg->mask, 4 test->mask}. deg 3.
    assert np.isclose(f["hop1_frac_licit"][2], 1 / 3)
    assert np.isclose(f["hop1_frac_suspicious"][2], 0.0)
    assert np.isclose(f["hop1_frac_unknown"][2], 2 / 3)
    # Node 5 (unlabeled): neighbor {0 licit, observable}. deg 1.
    assert np.isclose(f["hop1_frac_licit"][5], 1.0)
    # Node 4 (test) still GETS features from its non-test neighbor {2 suspicious}.
    assert np.isclose(f["hop1_frac_suspicious"][4], 1.0)


def test_hop2_hand_computed():
    f = _features()
    # Node 0 2-hop ring = {3 (sg1 suspicious), 4 (sg2 suspicious but TEST->mask)}.
    # deg2 = 2; only node 3 is an observable suspicious label.
    assert np.isclose(f["hop2_frac_suspicious"][0], 1 / 2)
    assert np.isclose(f["hop2_frac_unknown"][0], 1 / 2)
    assert np.isclose(f["hop2_frac_licit"][0], 0.0)


def test_own_subgraph_label_never_leaks():
    """Node 1's only neighbor (0) shares its own subgraph & licit label."""
    f = _features()
    assert np.isclose(f["hop1_frac_licit"][1], 0.0)     # masked, not 1.0
    assert np.isclose(f["hop1_frac_unknown"][1], 1.0)
    # Positive control: move node 1 into its OWN distinct licit subgraph (sg3),
    # so node 0 is no longer same-subgraph -> its licit label becomes observable.
    ns = np.array([0, 3, 1, 1, 2, -1], dtype=np.int64)
    labels = np.array([LICIT, SUSPICIOUS, SUSPICIOUS, LICIT], dtype=np.int64)
    in_test = np.array([False, False, True, False], dtype=bool)
    g = compute_neighborhood_features(EDGE_INDEX, ns, labels, in_test, N)
    assert np.isclose(g["hop1_frac_licit"][1], 1.0)     # different sg -> observable


def test_test_split_label_never_leaks():
    """Node 0's 2-hop neighbor 4 is suspicious but in the TEST split."""
    f = _features()
    assert np.isclose(f["hop2_frac_suspicious"][0], 1 / 2)  # node 4 excluded
    # Positive control: take sg2 OUT of the test split; now both 2-hop neighbors
    # (3 and 4) are observable suspicious -> fraction jumps to 1.0.
    no_test = np.array([False, False, False], dtype=bool)
    g = compute_neighborhood_features(
        EDGE_INDEX, NODE_SUBGRAPH, SUBGRAPH_LABEL, no_test, N
    )
    assert np.isclose(g["hop2_frac_suspicious"][0], 1.0)
    # And on the masked version node 4's label contributes only to unknown.
    assert np.isclose(f["hop2_frac_unknown"][0], 1 / 2)


def test_isolated_node_takes_empty_value():
    # Node 2 is isolated; everything else is one edge 0<->1.
    ei = np.array([[0], [1]], dtype=np.int32)
    ns = np.array([0, 0, -1], dtype=np.int64)
    lab = np.array([LICIT], dtype=np.int64)
    it = np.array([False], dtype=bool)
    f = compute_neighborhood_features(ei, ns, lab, it, 3, empty_value=-1.0)
    for c in COLUMNS:
        assert f[c][2] == -1.0  # isolated -> fill, both hops


def test_empty_graph_all_fill():
    f = compute_neighborhood_features(
        np.empty((2, 0), dtype=np.int32),
        np.array([-1, -1], dtype=np.int64),
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=bool),
        2,
    )
    for c in COLUMNS:
        assert f[c].tolist() == [0.0, 0.0]


def test_hops_param_subset():
    f = compute_neighborhood_features(
        EDGE_INDEX, NODE_SUBGRAPH, SUBGRAPH_LABEL, SUBGRAPH_IN_TEST, N, hops=(1,)
    )
    assert set(f) == {"hop1_frac_licit", "hop1_frac_suspicious", "hop1_frac_unknown"}


def test_out_of_range_endpoint_rejected():
    bad = np.array([[0], [9]], dtype=np.int32)
    try:
        compute_neighborhood_features(
            bad, NODE_SUBGRAPH, SUBGRAPH_LABEL, SUBGRAPH_IN_TEST, N
        )
    except ValueError as e:
        assert "out of range" in str(e)
    else:
        raise AssertionError("expected ValueError for out-of-range endpoint")


def test_loader_consumes_persisted_split():
    """load_subgraph_labels reads subgraphs.parquet + the persisted split.csv."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ellip2.eval.splits import SplitConfig, generate_split

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        # connected_components.csv with enough positives for a stratified split.
        n_sub, n_pos = 40, 12
        comp = tmp / "connected_components.csv"
        with comp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ccId", "ccLabel"])
            for i in range(n_sub):
                lab = "suspicious" if i < n_pos else "licit"
                w.writerow([f"cc{i:03d}", lab])
        cfg = SplitConfig(
            input_csv=comp, out_dir=tmp / "splits",
            method="stratified_random", seed=42, expected_total=None,
        )
        res = generate_split(cfg)
        split_csv = res.out_dir / "split.csv"

        # subgraphs.parquet: one disjoint node block per subgraph, ccIds aligned.
        ccids = [f"cc{i:03d}" for i in range(n_sub)]
        cclabels = ["suspicious" if i < n_pos else "licit" for i in range(n_sub)]
        members = [[2 * i, 2 * i + 1] for i in range(n_sub)]
        n_nodes = 2 * n_sub
        sub_pq = tmp / "subgraphs.parquet"
        pq.write_table(
            pa.table({"ccId": ccids, "ccLabel": cclabels, "member_idx": members}),
            sub_pq,
        )

        sl = load_subgraph_labels(sub_pq, split_csv, n_nodes)

        # Membership round-trips and labels map correctly.
        assert sl.node_subgraph.shape == (n_nodes,)
        for sid in range(n_sub):
            assert sl.node_subgraph[2 * sid] == sid
            assert sl.node_subgraph[2 * sid + 1] == sid
        assert (sl.subgraph_label[:n_pos] == SUSPICIOUS).all()
        assert (sl.subgraph_label[n_pos:] == LICIT).all()

        # in_test mask matches the persisted split exactly.
        with split_csv.open(newline="") as fh:
            test_ids = {r["id"] for r in csv.DictReader(fh) if r["split"] == "test"}
        assert test_ids  # stratified split has test members
        for sid, cc in enumerate(sl.ccids):
            assert sl.subgraph_in_test[sid] == (cc in test_ids)

        # And the loaded arrays drive the feature computation without leaking test.
        f = compute_neighborhood_features(
            np.array([[0, 2], [2, 0]], dtype=np.int32),
            sl.node_subgraph, sl.subgraph_label, sl.subgraph_in_test, n_nodes,
        )
        assert f["hop1_frac_licit"].shape == (n_nodes,)


# --------------------------------------------------------------------------- #


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {t.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
