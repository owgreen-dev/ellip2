"""Tests for ellip2.report.render (T-021).

Synthetic, CPU-only, no network (SIGN-101). The report is a pure function of its
inputs; the mandatory false-positive caveat block is the load-bearing property.
"""

from __future__ import annotations

from ellip2.llm.serialize_subgraph import (
    CandidateSubgraph,
    SubgraphEdge,
    SubgraphNode,
)
from ellip2.report.render import (
    CAVEAT_LINES,
    SubgraphReport,
    build_report,
    render_report,
    report_from_classification,
)


def _report(*, flagged: bool = False) -> SubgraphReport:
    return SubgraphReport(
        subgraph_id=42,
        pu_score=0.987654,
        pu_percentile=0.995,
        exit_paths=[[0, 1, 2, 3]],
        endpoint_type="exchange",
        typology="peeling_chain",
        confidence=0.81,
        structural_evidence={
            "total_degree": 6.0,
            "in_gini": 0.72,
            "out_max_counterparty_share": 0.5,
        },
        rationale="Funds hop through single-output relays toward an exchange.",
        evidence=("near-linear chain", "decreasing weights"),
        flagged=flagged,
        validation_reason="model typology corroborated by structural signals",
    )


def test_build_report_has_all_required_fields() -> None:
    out = build_report(_report())
    for key in (
        "subgraph_id",
        "pu_score",
        "pu_percentile",
        "exit_paths",
        "endpoint_type",
        "typology",
        "confidence",
        "structural_evidence",
        "rationale",
        "evidence",
        "flagged",
        "validation_reason",
        "caveats",
    ):
        assert key in out


def test_build_report_values_round_trip() -> None:
    out = build_report(_report())
    assert out["subgraph_id"] == 42
    assert out["pu_score"] == 0.987654
    assert out["pu_percentile"] == 0.995
    assert out["exit_paths"] == [[0, 1, 2, 3]]
    assert out["endpoint_type"] == "exchange"
    assert out["typology"] == "peeling_chain"
    assert out["confidence"] == 0.81
    # Degree / Gini / flow structural evidence carried through.
    assert out["structural_evidence"]["in_gini"] == 0.72
    assert out["structural_evidence"]["total_degree"] == 6.0


def test_caveat_block_present_and_nonempty() -> None:
    out = build_report(_report())
    assert isinstance(out["caveats"], list)
    assert len(out["caveats"]) == len(CAVEAT_LINES)
    assert out["caveats"] == list(CAVEAT_LINES)
    # The false-positive caveat is mandatory: it must mention review + leads.
    joined = " ".join(out["caveats"]).lower()
    assert "investigative lead" in joined
    assert "human review" in joined
    assert "false positive" in joined


def test_render_text_contains_all_sections_and_caveat() -> None:
    text = render_report(_report())
    assert "Suspicious subgraph 42" in text
    assert "PU score" in text
    assert "percentile" in text
    assert "Exit path(s)" in text
    assert "exchange" in text
    assert "0 -> 1 -> 2 -> 3" in text
    assert "Structural evidence" in text
    assert "in_gini" in text
    assert "Model rationale" in text
    assert "single-output relays" in text
    assert "## Caveats" in text
    # Every caveat line is rendered.
    for caveat in CAVEAT_LINES:
        assert caveat in text


def test_flagged_typology_surfaced() -> None:
    clean = render_report(_report(flagged=False))
    flagged = render_report(_report(flagged=True))
    assert "FLAGGED" not in clean
    assert "FLAGGED" in flagged


def test_empty_exit_paths_render_placeholder() -> None:
    report = SubgraphReport(
        subgraph_id=7,
        pu_score=0.5,
        pu_percentile=0.9,
        exit_paths=[],
        endpoint_type="exchange",
        typology="consolidation",
        confidence=0.4,
        structural_evidence={},
        rationale="",
    )
    out = build_report(report)
    assert out["exit_paths"] == []
    # Caveat block is still mandatory even with no exit path / evidence.
    assert out["caveats"] == list(CAVEAT_LINES)
    text = render_report(report)
    assert "(none recovered)" in text
    assert "## Caveats" in text


def test_report_from_classification_uses_candidate_stats() -> None:
    candidate = CandidateSubgraph(
        subgraph_id=99,
        pu_score=0.93,
        nodes=[
            SubgraphNode(0, "source", [0.0]),
            SubgraphNode(1, "relay", [0.5]),
            SubgraphNode(2, "endpoint", [1.0]),
        ],
        edges=[
            SubgraphEdge(0, 1, weight=2.0),
            SubgraphEdge(1, 2, weight=1.0),
        ],
        exit_paths=[[0, 1, 2]],
        stats={"in_gini": 0.3, "total_degree": 4.0},
    )
    classification = {
        "typology": "peeling_chain",
        "confidence": 0.77,
        "rationale": "linear hops",
        "evidence": ["chain"],
        "flagged": False,
        "validation_reason": "corroborated",
    }
    report = report_from_classification(
        candidate,
        classification,
        pu_percentile=0.98,
        endpoint_type="exchange",
    )
    assert report.subgraph_id == 99
    assert report.pu_score == 0.93
    assert report.typology == "peeling_chain"
    assert report.confidence == 0.77
    # Structural evidence defaults to the candidate's own Stage 1 stats.
    assert report.structural_evidence == {"in_gini": 0.3, "total_degree": 4.0}
    assert report.exit_paths == [[0, 1, 2]]

    out = build_report(report)
    assert out["caveats"] == list(CAVEAT_LINES)
    assert out["endpoint_type"] == "exchange"


def test_report_from_classification_override_evidence() -> None:
    candidate = CandidateSubgraph(subgraph_id=1, pu_score=0.6)
    classification = {"typology": "consolidation", "confidence": 0.5}
    report = report_from_classification(
        candidate,
        classification,
        pu_percentile=0.5,
        endpoint_type="mixer",
        structural_evidence={"out_hhi": 0.9},
    )
    assert report.structural_evidence == {"out_hhi": 0.9}
    # Missing optional classification keys default cleanly.
    assert report.rationale == ""
    assert report.evidence == []
    assert report.flagged is False
    assert report.validation_reason is None
