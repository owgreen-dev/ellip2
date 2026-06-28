"""Tests for the LangGraph typology agent (mock LLM node, SIGN-101).

Exercises (1) the end-to-end serialize → classify → validate → report wiring with
a MOCK LLM node, and (2) that rule-based validation can override / flag a typology
the structural signals contradict — without any network or AWS access.
"""

from __future__ import annotations

from typing import Any

import pytest

from ellip2.llm.bedrock_client import TypologyResult
from ellip2.llm.serialize_subgraph import (
    CandidateSubgraph,
    SubgraphEdge,
    SubgraphNode,
)
from ellip2.llm.typology_graph import (
    StructuralSignals,
    build_typology_graph,
    classify_candidate,
    compute_structural_signals,
    predict_typology,
    validate_typology,
)


class _MockClassifier:
    """A stand-in LLM node: records its input and returns a canned verdict."""

    def __init__(self, result: TypologyResult) -> None:
        self.result = result
        self.calls: list[Any] = []

    def __call__(self, serialized: Any) -> TypologyResult:
        self.calls.append(serialized)
        return self.result


def _chain_candidate() -> CandidateSubgraph:
    # 0 -> 1 -> 2 -> 3: a near-linear peeling chain (no hub, linearity ~ 1).
    nodes = [SubgraphNode(node_id=i, role="link", features=[float(i)]) for i in range(4)]
    edges = [SubgraphEdge(source=i, target=i + 1, weight=1.0) for i in range(3)]
    return CandidateSubgraph(
        subgraph_id=7,
        pu_score=0.91,
        nodes=nodes,
        edges=edges,
        exit_paths=[[0, 1, 2, 3]],
        stats={"in_gini": 0.0},
    )


def _consolidation_candidate() -> CandidateSubgraph:
    # 0,1,2 -> 3: a many-to-one funnel (max in-degree 3, no fan-out).
    nodes = [SubgraphNode(node_id=i, role="x", features=[float(i)]) for i in range(4)]
    edges = [SubgraphEdge(source=i, target=3, weight=1.0) for i in range(3)]
    return CandidateSubgraph(subgraph_id=11, pu_score=0.8, nodes=nodes, edges=edges)


def _smurf_candidate() -> CandidateSubgraph:
    # 0 -> 1,2,3: a one-to-many split (max out-degree 3, no funnel).
    nodes = [SubgraphNode(node_id=i, role="x", features=[float(i)]) for i in range(4)]
    edges = [SubgraphEdge(source=0, target=i, weight=1.0) for i in range(1, 4)]
    return CandidateSubgraph(subgraph_id=12, pu_score=0.7, nodes=nodes, edges=edges)


def _hub_candidate() -> CandidateSubgraph:
    # 0,1 -> 2 -> 3,4: an intermediary hub (high in AND out at node 2).
    nodes = [SubgraphNode(node_id=i, role="x", features=[float(i)]) for i in range(5)]
    edges = [
        SubgraphEdge(source=0, target=2, weight=1.0),
        SubgraphEdge(source=1, target=2, weight=1.0),
        SubgraphEdge(source=2, target=3, weight=1.0),
        SubgraphEdge(source=2, target=4, weight=1.0),
    ]
    return CandidateSubgraph(subgraph_id=13, pu_score=0.6, nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- #
# Structural signal computation                                               #
# --------------------------------------------------------------------------- #


def test_structural_signals_chain() -> None:
    sig = compute_structural_signals(_chain_candidate())
    assert sig == StructuralSignals(
        n_nodes=4, n_edges=3, max_in_degree=1, max_out_degree=1, linearity=1.0
    )


def test_structural_signals_funnel() -> None:
    sig = compute_structural_signals(_consolidation_candidate())
    assert sig.max_in_degree == 3
    assert sig.max_out_degree == 1


def test_structural_signals_split() -> None:
    sig = compute_structural_signals(_smurf_candidate())
    assert sig.max_out_degree == 3
    assert sig.max_in_degree == 1


# --------------------------------------------------------------------------- #
# Rule-based structural prediction                                            #
# --------------------------------------------------------------------------- #


def test_predict_typology_each_shape() -> None:
    assert predict_typology(compute_structural_signals(_chain_candidate())) == "peeling_chain"
    assert (
        predict_typology(compute_structural_signals(_consolidation_candidate()))
        == "consolidation"
    )
    assert (
        predict_typology(compute_structural_signals(_smurf_candidate()))
        == "layering_smurfing"
    )
    assert predict_typology(compute_structural_signals(_hub_candidate())) == "nested_service"


def test_predict_typology_inconclusive_singleton() -> None:
    single = CandidateSubgraph(
        subgraph_id=1, pu_score=0.5, nodes=[SubgraphNode(0, "x")], edges=[]
    )
    assert predict_typology(compute_structural_signals(single)) is None


# --------------------------------------------------------------------------- #
# Validation: agreement / flag / override                                     #
# --------------------------------------------------------------------------- #


def test_validate_agreement_not_flagged() -> None:
    signals = compute_structural_signals(_consolidation_candidate())
    result = TypologyResult("consolidation", 0.9, "funnel", ("many senders",))
    outcome = validate_typology(result, signals)
    assert outcome.agrees is True
    assert outcome.flagged is False
    assert outcome.final_typology == "consolidation"
    assert outcome.structural_typology == "consolidation"


def test_validate_contradiction_overrides() -> None:
    # Structure is a clear funnel; the model wrongly says peeling_chain.
    signals = compute_structural_signals(_consolidation_candidate())
    result = TypologyResult("peeling_chain", 0.95, "looks like peeling", ())
    outcome = validate_typology(result, signals, override=True)
    assert outcome.flagged is True
    assert outcome.agrees is False
    assert outcome.claimed_typology == "peeling_chain"
    assert outcome.structural_typology == "consolidation"
    assert outcome.final_typology == "consolidation"  # overridden


def test_validate_contradiction_flag_without_override() -> None:
    signals = compute_structural_signals(_consolidation_candidate())
    result = TypologyResult("peeling_chain", 0.95, "looks like peeling", ())
    outcome = validate_typology(result, signals, override=False)
    assert outcome.flagged is True
    assert outcome.final_typology == "peeling_chain"  # kept but flagged


def test_validate_inconclusive_accepts_claim() -> None:
    single = CandidateSubgraph(
        subgraph_id=1, pu_score=0.5, nodes=[SubgraphNode(0, "x")], edges=[]
    )
    signals = compute_structural_signals(single)
    result = TypologyResult("nested_service", 0.4, "guess", ())
    outcome = validate_typology(result, signals)
    assert outcome.agrees is True
    assert outcome.flagged is False
    assert outcome.final_typology == "nested_service"


# --------------------------------------------------------------------------- #
# End-to-end graph (mock LLM node)                                            #
# --------------------------------------------------------------------------- #


def test_graph_end_to_end_corroborated() -> None:
    candidate = _consolidation_candidate()
    classifier = _MockClassifier(
        TypologyResult("consolidation", 0.88, "many-to-one", ("3 senders into 1",))
    )
    report = classify_candidate(candidate, classifier)

    # The classify node ran against the serialized doc (state transition proof).
    assert len(classifier.calls) == 1
    assert classifier.calls[0]["subgraph_id"] == 11

    assert report["subgraph_id"] == 11
    assert report["typology"] == "consolidation"
    assert report["model_typology"] == "consolidation"
    assert report["confidence"] == 0.88
    assert report["flagged"] is False
    assert report["structural_typology"] == "consolidation"
    assert report["evidence"] == ["3 senders into 1"]


def test_graph_end_to_end_override_flags_contradiction() -> None:
    candidate = _consolidation_candidate()
    # Model is wrong: claims peeling_chain on an obvious funnel.
    classifier = _MockClassifier(TypologyResult("peeling_chain", 0.99, "wrong", ()))
    report = classify_candidate(candidate, classifier)

    assert report["model_typology"] == "peeling_chain"
    assert report["structural_typology"] == "consolidation"
    assert report["flagged"] is True
    assert report["typology"] == "consolidation"  # validation overrode the model
    assert "contradict" in report["validation_reason"]


def test_graph_no_override_keeps_flagged_model_typology() -> None:
    candidate = _consolidation_candidate()
    classifier = _MockClassifier(TypologyResult("peeling_chain", 0.99, "wrong", ()))
    report = classify_candidate(candidate, classifier, override_on_contradiction=False)
    assert report["flagged"] is True
    assert report["typology"] == "peeling_chain"


def test_build_graph_returns_invocable_app() -> None:
    candidate = _chain_candidate()
    classifier = _MockClassifier(TypologyResult("peeling_chain", 0.5, "chain", ()))
    app = build_typology_graph(classifier)
    final_state = app.invoke({"candidate": candidate})
    assert "report" in final_state
    assert "serialized" in final_state
    assert final_state["result"].typology == "peeling_chain"


def test_classifier_not_called_without_invoke() -> None:
    # Building the graph must not invoke the LLM node (no eager classification).
    classifier = _MockClassifier(TypologyResult("consolidation", 0.5, "x", ()))
    build_typology_graph(classifier)
    assert classifier.calls == []


def test_serialize_config_forwarded() -> None:
    from ellip2.llm.serialize_subgraph import SerializationConfig

    candidate = _consolidation_candidate()
    classifier = _MockClassifier(TypologyResult("consolidation", 0.5, "x", ()))
    cfg = SerializationConfig(n_bins=3)
    report = classify_candidate(candidate, classifier, config=cfg)
    # Serialization happened (report produced) and the doc reached the classifier.
    assert report["subgraph_id"] == 11
    assert classifier.calls[0]["n_nodes"] == 4


def test_invalid_candidate_propagates() -> None:
    # serialize_subgraph raises on a duplicate node id; the graph surfaces it.
    bad = CandidateSubgraph(
        subgraph_id=1,
        pu_score=0.5,
        nodes=[SubgraphNode(0, "x"), SubgraphNode(0, "y")],
        edges=[],
    )
    classifier = _MockClassifier(TypologyResult("consolidation", 0.5, "x", ()))
    with pytest.raises(ValueError):
        classify_candidate(bad, classifier)
