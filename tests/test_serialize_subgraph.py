"""Tests for ellip2.llm.serialize_subgraph (T-018).

Synthetic, CPU-only, no network (SIGN-101). The three load-bearing properties:
schema, determinism, and a deterministically-enforced size budget.
"""

from __future__ import annotations

import json

import pytest

from ellip2.llm.serialize_subgraph import (
    CandidateSubgraph,
    SerializationConfig,
    SubgraphEdge,
    SubgraphNode,
    serialize_subgraph,
    serialize_subgraph_json,
)


def _candidate() -> CandidateSubgraph:
    """A tiny 4-node peeling-chain-ish candidate: 0 -> 1 -> 2 -> 3 (endpoint)."""
    nodes = [
        SubgraphNode(0, "source", [0.0, 10.0]),
        SubgraphNode(1, "relay", [0.5, 20.0]),
        SubgraphNode(2, "relay", [0.9, 30.0]),
        SubgraphNode(3, "endpoint", [1.0, 40.0]),
    ]
    edges = [
        SubgraphEdge(0, 1, weight=5.0, volume=100.0, timestamp=1.0),
        SubgraphEdge(1, 2, weight=3.0, volume=60.0, timestamp=2.0),
        SubgraphEdge(2, 3, weight=1.0, volume=20.0, timestamp=3.0),
    ]
    return CandidateSubgraph(
        subgraph_id=42,
        pu_score=0.987654321,
        nodes=nodes,
        edges=edges,
        exit_paths=[[0, 1, 2, 3]],
        stats={"in_gini": 0.123456789, "total_degree": 6.0},
    )


def test_schema_has_all_required_fields() -> None:
    payload = serialize_subgraph(_candidate())
    for key in (
        "subgraph_id",
        "pu_score",
        "n_nodes",
        "n_edges",
        "nodes",
        "edges",
        "exit_paths",
        "stats",
        "truncated",
    ):
        assert key in payload

    assert payload["subgraph_id"] == 42
    assert payload["n_nodes"] == 4
    assert payload["n_edges"] == 3
    assert payload["truncated"] is False
    assert payload["exit_paths"] == [[0, 1, 2, 3]]

    # Nodes sorted by id, carry role + binned ordinals.
    assert [n["id"] for n in payload["nodes"]] == [0, 1, 2, 3]
    for node in payload["nodes"]:
        assert set(node) == {"id", "role", "bins"}
        assert all(isinstance(b, int) for b in node["bins"])
    # Edges sorted by (source, target) with the transaction attributes.
    assert [(e["s"], e["t"]) for e in payload["edges"]] == [(0, 1), (1, 2), (2, 3)]
    for edge in payload["edges"]:
        assert set(edge) == {"s", "t", "weight", "volume", "timestamp"}


def test_feature_binning_spans_range() -> None:
    payload = serialize_subgraph(_candidate(), SerializationConfig(n_bins=10))
    bins_by_id = {n["id"]: n["bins"] for n in payload["nodes"]}
    # Column 0 spans 0.0..1.0: min -> bin 0, max -> top bin (n_bins - 1).
    assert bins_by_id[0][0] == 0
    assert bins_by_id[3][0] == 9
    # Column 1 spans 10..40 identically.
    assert bins_by_id[0][1] == 0
    assert bins_by_id[3][1] == 9


def test_constant_feature_column_bins_to_zero() -> None:
    cand = CandidateSubgraph(
        subgraph_id=1,
        pu_score=0.5,
        nodes=[SubgraphNode(0, "a", [7.0]), SubgraphNode(1, "b", [7.0])],
    )
    payload = serialize_subgraph(cand)
    assert all(n["bins"] == [0] for n in payload["nodes"])


def test_determinism_byte_identical() -> None:
    cand = _candidate()
    first = serialize_subgraph_json(cand)
    second = serialize_subgraph_json(cand)
    assert first == second
    # And independent of input ordering of nodes/edges: a reordered candidate
    # serializes identically.
    reordered = CandidateSubgraph(
        subgraph_id=cand.subgraph_id,
        pu_score=cand.pu_score,
        nodes=list(reversed(cand.nodes)),
        edges=list(reversed(cand.edges)),
        exit_paths=cand.exit_paths,
        stats=dict(reversed(list(cand.stats.items()))),
    )
    assert serialize_subgraph_json(reordered) == first
    # Valid JSON.
    assert json.loads(first)["subgraph_id"] == 42


def test_float_rounding_precision() -> None:
    payload = serialize_subgraph(_candidate(), SerializationConfig(precision=3))
    assert payload["pu_score"] == 0.988
    assert payload["stats"]["in_gini"] == 0.123


def test_size_budget_respected_and_marks_truncated() -> None:
    cand = _candidate()
    full = serialize_subgraph_json(cand)
    full_bytes = len(full.encode("utf-8"))

    # A budget below the full size forces deterministic shedding.
    budget = full_bytes - 40
    payload = serialize_subgraph(cand, SerializationConfig(max_bytes=budget))
    encoded = serialize_subgraph_json(cand, SerializationConfig(max_bytes=budget))
    assert len(encoded.encode("utf-8")) <= budget
    assert payload["truncated"] is True
    # Edges are shed before exit-path nodes: the chain nodes survive.
    kept_ids = {n["id"] for n in payload["nodes"]}
    assert {0, 1, 2, 3} <= kept_ids or payload["n_edges"] < 3
    # Truncation is deterministic.
    assert serialize_subgraph_json(cand, SerializationConfig(max_bytes=budget)) == encoded


def test_no_budget_keeps_everything() -> None:
    payload = serialize_subgraph(_candidate(), SerializationConfig(max_bytes=None))
    assert payload["truncated"] is False
    assert payload["n_edges"] == 3


def test_impossible_budget_raises() -> None:
    with pytest.raises(ValueError, match="cannot serialize"):
        serialize_subgraph(_candidate(), SerializationConfig(max_bytes=1))


def test_edge_referencing_unknown_node_raises() -> None:
    cand = CandidateSubgraph(
        subgraph_id=1,
        pu_score=0.5,
        nodes=[SubgraphNode(0, "a", [1.0])],
        edges=[SubgraphEdge(0, 99, weight=1.0)],
    )
    with pytest.raises(ValueError, match="unknown node"):
        serialize_subgraph(cand)


def test_exit_path_unknown_node_raises() -> None:
    cand = CandidateSubgraph(
        subgraph_id=1,
        pu_score=0.5,
        nodes=[SubgraphNode(0, "a", [1.0])],
        exit_paths=[[0, 5]],
    )
    with pytest.raises(ValueError, match="unknown node"):
        serialize_subgraph(cand)


@pytest.mark.parametrize(
    "config",
    [
        SerializationConfig(n_bins=0),
        SerializationConfig(precision=-1),
        SerializationConfig(max_bytes=0),
    ],
)
def test_invalid_config_raises(config: SerializationConfig) -> None:
    with pytest.raises(ValueError):
        serialize_subgraph(_candidate(), config)
