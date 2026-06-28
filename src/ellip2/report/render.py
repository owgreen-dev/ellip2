"""Stage 4 (reporting) — render a per-subgraph investigative report.

plan.md §8: once a candidate subgraph has been scored (Stage 2), corroborated
with an exit path (Stage 3), and assigned a money-laundering typology by the LLM
layer (:mod:`ellip2.llm.typology_graph`), the pipeline emits a human-readable
*lead* for an analyst. This module turns those signals into a single report
carrying everything a reviewer needs to triage the subgraph:

  * the subgraph **id**;
  * its **PU score** and population **percentile** (Stage 2);
  * the recovered **exit path(s)** to the heuristic licit **endpoint type**
    (Stage 3 reachability + the Stage 1 ``path_role`` heuristic);
  * the LLM-assigned **typology** and its **confidence** (Stage 4), together with
    whether the rule-based validator **flagged** a structural contradiction;
  * the **structural evidence** — degree / Gini / flow-concentration stats
    (Stage 1) — backing the call;
  * the model's free-text **rationale** and cited evidence;
  * a mandatory **false-positive caveat block**.

The caveat block is not optional. Every output of this pipeline is an
*investigative lead*, not an accusation: the PU score is a SCAR lower bound, the
endpoint type and node roles are DERIVED heuristics (the dataset ships no
ground-truth entity labels), and the typology is a model verdict. The report
must say so, every time, so a downstream reader never mistakes a lead for a
finding. Rendering is a pure function — no network, no I/O (SIGN-101).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ellip2.llm.serialize_subgraph import CandidateSubgraph

#: The mandatory false-positive caveat. Surfaced on every report (plan.md §8).
CAVEAT_LINES: tuple[str, ...] = (
    "This is an automatically generated INVESTIGATIVE LEAD, not a finding or an "
    "accusation. It requires human review before any action.",
    "The PU suspicion score is a positive-unlabeled (SCAR) lower bound: the "
    "unlabeled pool contains benign clusters, so a high score is not proof of "
    "illicit activity.",
    "Node roles and the licit endpoint type are DERIVED heuristics, not "
    "ground-truth entity labels — the dataset ships none.",
    "The typology is a model verdict; treat a flagged (structurally contradicted) "
    "typology with extra caution.",
    "False positives are expected. Corroborate independently before escalating.",
)


@dataclass(frozen=True)
class SubgraphReport:
    """The structured inputs of a per-subgraph report.

    Attributes:
        subgraph_id: the candidate's subgraph id.
        pu_score: its Stage 2 PU suspicion score.
        pu_percentile: the score's population percentile in ``[0, 1]``.
        exit_paths: recovered ≤k-hop exit path(s); each a node-id sequence from a
            candidate source to the heuristic licit endpoint.
        endpoint_type: the DERIVED heuristic licit endpoint type the path ends at
            (e.g. ``"exchange"``).
        typology: the (final, possibly overridden) money-laundering typology.
        confidence: the LLM's confidence in its verdict, in ``[0, 1]``.
        structural_evidence: degree / Gini / flow-concentration stats (Stage 1).
        rationale: the model's free-text justification.
        evidence: structural signals the model cited.
        flagged: whether the rule-based validator flagged a structural
            contradiction.
        validation_reason: the validator's explanation, if any.
    """

    subgraph_id: int
    pu_score: float
    pu_percentile: float
    exit_paths: Sequence[Sequence[int]]
    endpoint_type: str
    typology: str
    confidence: float
    structural_evidence: Mapping[str, float]
    rationale: str
    evidence: Sequence[str] = field(default_factory=tuple)
    flagged: bool = False
    validation_reason: str | None = None


def build_report(report: SubgraphReport) -> dict[str, Any]:
    """Assemble the structured report dict, including the mandatory caveat block.

    Pure: returns a JSON-able dict; performs no I/O. The ``caveats`` key is always
    present and non-empty — the false-positive caveat block is not optional.
    """
    return {
        "subgraph_id": report.subgraph_id,
        "pu_score": report.pu_score,
        "pu_percentile": report.pu_percentile,
        "exit_paths": [list(path) for path in report.exit_paths],
        "endpoint_type": report.endpoint_type,
        "typology": report.typology,
        "confidence": report.confidence,
        "flagged": report.flagged,
        "validation_reason": report.validation_reason,
        "structural_evidence": dict(report.structural_evidence),
        "rationale": report.rationale,
        "evidence": list(report.evidence),
        "caveats": list(CAVEAT_LINES),
    }


def render_report(report: SubgraphReport) -> str:
    """Render a per-subgraph report as Markdown text.

    Every section is always present, including the trailing **Caveats** block.
    """
    lines: list[str] = []
    lines.append(f"# Suspicious subgraph {report.subgraph_id}")
    lines.append("")
    lines.append(
        f"- PU score: {report.pu_score:.6g} "
        f"(percentile {report.pu_percentile * 100:.1f}%)"
    )

    flag = " — FLAGGED (structural contradiction)" if report.flagged else ""
    lines.append(
        f"- Typology: {report.typology} "
        f"(confidence {report.confidence:.2f}){flag}"
    )
    if report.validation_reason:
        lines.append(f"  - Validation: {report.validation_reason}")

    lines.append("")
    lines.append("## Exit path(s)")
    lines.append(f"Heuristic licit endpoint type: {report.endpoint_type}")
    if report.exit_paths:
        for path in report.exit_paths:
            arrow = " -> ".join(str(node) for node in path)
            lines.append(f"- {arrow}")
    else:
        lines.append("- (none recovered)")

    lines.append("")
    lines.append("## Structural evidence")
    if report.structural_evidence:
        for key in sorted(report.structural_evidence):
            lines.append(f"- {key}: {report.structural_evidence[key]:.6g}")
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Model rationale")
    lines.append(report.rationale or "(none)")
    if report.evidence:
        lines.append("")
        lines.append("Cited evidence:")
        for item in report.evidence:
            lines.append(f"- {item}")

    lines.append("")
    lines.append("## Caveats")
    for caveat in CAVEAT_LINES:
        lines.append(f"- {caveat}")

    return "\n".join(lines)


def report_from_classification(
    candidate: CandidateSubgraph,
    classification: Mapping[str, Any],
    *,
    pu_percentile: float,
    endpoint_type: str,
    structural_evidence: Mapping[str, float] | None = None,
) -> SubgraphReport:
    """Build a :class:`SubgraphReport` from a candidate + a typology-graph report.

    ``classification`` is the dict returned by
    :func:`ellip2.llm.typology_graph.classify_candidate` (keys ``typology``,
    ``confidence``, ``rationale``, ``evidence``, ``flagged``,
    ``validation_reason``). The structural evidence defaults to the candidate's
    own ``stats`` (the Stage 1 degree / Gini / flow summary) when not supplied.
    """
    evidence = (
        structural_evidence
        if structural_evidence is not None
        else dict(candidate.stats)
    )
    return SubgraphReport(
        subgraph_id=candidate.subgraph_id,
        pu_score=candidate.pu_score,
        pu_percentile=pu_percentile,
        exit_paths=[list(path) for path in candidate.exit_paths],
        endpoint_type=endpoint_type,
        typology=str(classification["typology"]),
        confidence=float(classification["confidence"]),
        structural_evidence=evidence,
        rationale=str(classification.get("rationale", "")),
        evidence=list(classification.get("evidence", ())),
        flagged=bool(classification.get("flagged", False)),
        validation_reason=classification.get("validation_reason"),
    )


__all__ = [
    "CAVEAT_LINES",
    "SubgraphReport",
    "build_report",
    "render_report",
    "report_from_classification",
]
