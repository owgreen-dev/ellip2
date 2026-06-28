"""Stage 4 (LLM layer) — a LangGraph agent that classifies a candidate's typology.

plan.md §8: a Stage 3 candidate subgraph is run through a small state machine that

  1. **serializes** it to the compact, deterministic JSON the model sees
     (:mod:`ellip2.llm.serialize_subgraph`);
  2. **classifies** its money-laundering typology with an *injectable* LLM node —
     in production :meth:`ellip2.llm.bedrock_client.BedrockTypologyClient.classify`,
     under test a stub returning a canned :class:`~ellip2.llm.bedrock_client.TypologyResult`
     (SIGN-101 — no network);
  3. **validates** the model's verdict against *structural* signals read straight
     off the subgraph topology — a rule-based check that can flag, and optionally
     override, a typology the structure contradicts;
  4. **reports** a single typed verdict bundling the (possibly overridden)
     typology with the validation outcome.

The LLM is the only non-deterministic actor and it is injected, so the whole
graph — wiring, state transitions, and the rule-based override — is exercised on
CPU with a mock classifier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ellip2.llm.bedrock_client import TYPOLOGIES, TypologyResult
from ellip2.llm.serialize_subgraph import (
    CandidateSubgraph,
    SerializationConfig,
    serialize_subgraph,
)

#: A classify node: maps a serialized subgraph (dict or JSON str) to a verdict.
Classifier = Callable[[Any], TypologyResult]


@dataclass(frozen=True)
class StructuralSignals:
    """Interpretable topology signals derived from a candidate subgraph.

    Computed purely from node/edge counts and degrees — no model, no labels — so
    the rule-based validator has an independent view of the subgraph's shape.

    Attributes:
        n_nodes: number of nodes.
        n_edges: number of directed edges.
        max_in_degree: largest in-degree (funnel / consolidation signal).
        max_out_degree: largest out-degree (split / smurfing signal).
        linearity: ``n_edges / (n_nodes - 1)`` — ``~1`` for a simple chain.
    """

    n_nodes: int
    n_edges: int
    max_in_degree: int
    max_out_degree: int
    linearity: float


@dataclass(frozen=True)
class ValidationOutcome:
    """The result of validating an LLM verdict against structural signals.

    Attributes:
        claimed_typology: the typology the LLM returned.
        structural_typology: the typology implied by the topology, or ``None`` if
            the structure is inconclusive.
        agrees: whether the structural reading is consistent with the claim
            (``True`` when the structure is inconclusive — nothing to contradict).
        flagged: ``True`` when the structure actively contradicts the claim.
        final_typology: the typology carried forward — overridden to the
            structural reading on a contradiction when ``override`` is enabled,
            otherwise the claimed one.
        reason: a short human-readable explanation.
    """

    claimed_typology: str
    structural_typology: str | None
    agrees: bool
    flagged: bool
    final_typology: str
    reason: str


class TypologyState(TypedDict, total=False):
    """Mutable state threaded through the LangGraph nodes."""

    candidate: CandidateSubgraph
    serialized: dict[str, Any]
    result: TypologyResult
    validation: ValidationOutcome
    report: dict[str, Any]


def compute_structural_signals(candidate: CandidateSubgraph) -> StructuralSignals:
    """Derive :class:`StructuralSignals` from a candidate's topology."""
    in_deg: dict[int, int] = {}
    out_deg: dict[int, int] = {}
    for edge in candidate.edges:
        out_deg[edge.source] = out_deg.get(edge.source, 0) + 1
        in_deg[edge.target] = in_deg.get(edge.target, 0) + 1
    n_nodes = len(candidate.nodes)
    n_edges = len(candidate.edges)
    denom = n_nodes - 1
    return StructuralSignals(
        n_nodes=n_nodes,
        n_edges=n_edges,
        max_in_degree=max(in_deg.values(), default=0),
        max_out_degree=max(out_deg.values(), default=0),
        linearity=(n_edges / denom) if denom > 0 else 0.0,
    )


def predict_typology(signals: StructuralSignals) -> str | None:
    """Map structural signals to the typology the topology implies.

    Rule-based and deliberately coarse (the signal, not the verdict):

    * **consolidation** — many-to-one funnel: a node gathers from many senders
      (high ``max_in_degree``) while nothing fans back out.
    * **layering_smurfing** — one-to-many split: a node sprays to many receivers
      (high ``max_out_degree``) with no comparable funnel.
    * **nested_service** — a single intermediary with both high in- and
      out-degree (a service hub passing flow through).
    * **peeling_chain** — a near-linear chain (``linearity ~ 1``, no hub).

    Returns ``None`` when the topology is too small or ambiguous to call.
    """
    if signals.n_nodes < 2:
        return None
    fan_in = signals.max_in_degree
    fan_out = signals.max_out_degree
    hub = fan_in >= 2 and fan_out >= 2
    if hub:
        return "nested_service"
    if fan_in >= 2 and fan_out <= 1:
        return "consolidation"
    if fan_out >= 2 and fan_in <= 1:
        return "layering_smurfing"
    if fan_in <= 1 and fan_out <= 1 and signals.linearity >= 0.5:
        return "peeling_chain"
    return None


def validate_typology(
    result: TypologyResult,
    signals: StructuralSignals,
    *,
    override: bool = True,
) -> ValidationOutcome:
    """Validate an LLM verdict against structural signals.

    The structural reading is *advisory*: when it is inconclusive
    (``predict_typology`` returns ``None``) the claim stands unflagged. When it
    disagrees, the verdict is flagged — and, if ``override`` is set, the carried
    typology is replaced with the structural reading.
    """
    claimed = result.typology
    structural = predict_typology(signals)
    if structural is None:
        return ValidationOutcome(
            claimed_typology=claimed,
            structural_typology=None,
            agrees=True,
            flagged=False,
            final_typology=claimed,
            reason="structure inconclusive; accepting model typology",
        )
    if structural == claimed:
        return ValidationOutcome(
            claimed_typology=claimed,
            structural_typology=structural,
            agrees=True,
            flagged=False,
            final_typology=claimed,
            reason="model typology corroborated by structural signals",
        )
    final = structural if override else claimed
    verb = "overridden to" if override else "kept despite"
    return ValidationOutcome(
        claimed_typology=claimed,
        structural_typology=structural,
        agrees=False,
        flagged=True,
        final_typology=final,
        reason=(
            f"structural signals imply {structural!r}, contradicting model "
            f"{claimed!r}; {verb} structural reading"
        ),
    )


def _serialize_node(
    config: SerializationConfig | None,
) -> Callable[[TypologyState], dict[str, Any]]:
    def node(state: TypologyState) -> dict[str, Any]:
        serialized = serialize_subgraph(state["candidate"], config)
        return {"serialized": serialized}

    return node


def _classify_node(
    classifier: Classifier,
) -> Callable[[TypologyState], dict[str, Any]]:
    def node(state: TypologyState) -> dict[str, Any]:
        return {"result": classifier(state["serialized"])}

    return node


def _validate_node(
    override: bool,
) -> Callable[[TypologyState], dict[str, Any]]:
    def node(state: TypologyState) -> dict[str, Any]:
        signals = compute_structural_signals(state["candidate"])
        outcome = validate_typology(state["result"], signals, override=override)
        return {"validation": outcome}

    return node


def _report_node(state: TypologyState) -> dict[str, Any]:
    result = state["result"]
    validation = state["validation"]
    report = {
        "subgraph_id": state["candidate"].subgraph_id,
        "pu_score": state["candidate"].pu_score,
        "typology": validation.final_typology,
        "model_typology": result.typology,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "evidence": list(result.evidence),
        "structural_typology": validation.structural_typology,
        "flagged": validation.flagged,
        "validation_reason": validation.reason,
    }
    return {"report": report}


def build_typology_graph(
    classifier: Classifier,
    *,
    config: SerializationConfig | None = None,
    override_on_contradiction: bool = True,
) -> Any:
    """Build and compile the serialize → classify → validate → report graph.

    Args:
        classifier: the injectable LLM node — a callable mapping the serialized
            subgraph to a :class:`TypologyResult`. In production pass
            ``BedrockTypologyClient(...).classify``; under test pass a stub.
        config: serialization options forwarded to :func:`serialize_subgraph`.
        override_on_contradiction: when ``True`` the validator replaces a
            structurally-contradicted typology with the structural reading.

    Returns:
        A compiled LangGraph app; call ``app.invoke({"candidate": ...})``.
    """
    # Typed Any: langgraph's add_node overloads don't accept a factory-returned
    # Callable[[TypologyState], ...] cleanly, and over-constraining the wiring here
    # adds no safety (the node signatures are checked at their definitions).
    graph: Any = StateGraph(TypologyState)
    graph.add_node("serialize", _serialize_node(config))
    graph.add_node("classify", _classify_node(classifier))
    graph.add_node("validate", _validate_node(override_on_contradiction))
    graph.add_node("report", _report_node)
    graph.add_edge(START, "serialize")
    graph.add_edge("serialize", "classify")
    graph.add_edge("classify", "validate")
    graph.add_edge("validate", "report")
    graph.add_edge("report", END)
    return graph.compile()


def classify_candidate(
    candidate: CandidateSubgraph,
    classifier: Classifier,
    *,
    config: SerializationConfig | None = None,
    override_on_contradiction: bool = True,
) -> dict[str, Any]:
    """Run one candidate through a freshly-built typology graph; return the report."""
    app = build_typology_graph(
        classifier,
        config=config,
        override_on_contradiction=override_on_contradiction,
    )
    final_state: Mapping[str, Any] = app.invoke({"candidate": candidate})
    report: dict[str, Any] = final_state["report"]
    return report


__all__ = [
    "TYPOLOGIES",
    "Classifier",
    "StructuralSignals",
    "TypologyState",
    "ValidationOutcome",
    "build_typology_graph",
    "classify_candidate",
    "compute_structural_signals",
    "predict_typology",
    "validate_typology",
]
