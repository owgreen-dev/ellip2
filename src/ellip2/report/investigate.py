"""Stage 4 — full investigative report cards from border scores.

Composes the whole Stage-4 surface on top of the border model's per-subgraph scores:

1. rank the flagged subgraphs (:mod:`ellip2.report.leads`);
2. reconstruct each one's border graph and turn it into a
   :class:`~ellip2.llm.serialize_subgraph.CandidateSubgraph` (nodes with roles,
   directed edges, a recovered exit path to an external receiver, structural stats);
3. run it through the typology agent — serialize → classify → validate → report
   (:func:`ellip2.llm.typology_graph.classify_candidate`);
4. render the complete Markdown card (:func:`ellip2.report.render.render_report`) — PU
   score + percentile, exit path, LLM typology + confidence + structural validation,
   structural evidence, and the mandatory false-positive caveat — alongside the graph PNG.

The typology **classifier is injectable**. By default an offline, deterministic *heuristic*
classifier (structure → typology) is used, so a full card can be produced with no network /
no AWS. Pass ``--llm bedrock`` to use the real Bedrock Converse client instead (requires the
instance to have Bedrock invoke permissions and model access).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from ellip2.llm.bedrock_client import TYPOLOGIES, TypologyResult
from ellip2.llm.serialize_subgraph import CandidateSubgraph, SubgraphEdge, SubgraphNode
from ellip2.llm.typology_graph import classify_candidate
from ellip2.report.leads import (
    Lead,
    extract_lead_graphs,
    rank_leads,
    render_lead_png,
)
from ellip2.report.render import render_report, report_from_classification

_ENDPOINT_TYPE = "external receiver (DERIVED heuristic)"


def _structural_typology(max_in: int, max_out: int, linearity: float) -> tuple[str, float]:
    """Map degree signals to a typology + a confidence in [0,1] (mirrors typology_graph)."""
    if max_in >= 2 and max_out >= 2:
        return "nested_service", 0.8
    if max_in >= 2 and max_out <= 1:
        return "consolidation", 0.75
    if max_out >= 2 and max_in <= 1:
        return "layering_smurfing", 0.75
    if max_in <= 1 and max_out <= 1 and linearity >= 0.5:
        return "peeling_chain", 0.7
    return "layering_smurfing", 0.4  # inconclusive → low-confidence default


def heuristic_classifier(serialized: Mapping[str, Any]) -> TypologyResult:
    """Offline, deterministic stand-in for the LLM: types the serialized subgraph by shape.

    Operates on the same compact JSON the Bedrock model would see, so it is a drop-in
    :data:`~ellip2.llm.typology_graph.Classifier`. No network.
    """
    edges = serialized.get("edges", [])
    in_deg: dict[int, int] = {}
    out_deg: dict[int, int] = {}
    for e in edges:
        out_deg[e["s"]] = out_deg.get(e["s"], 0) + 1
        in_deg[e["t"]] = in_deg.get(e["t"], 0) + 1
    max_in = max(in_deg.values(), default=0)
    max_out = max(out_deg.values(), default=0)
    n_nodes = int(serialized.get("n_nodes", 0))
    linearity = (len(edges) / (n_nodes - 1)) if n_nodes > 1 else 0.0
    typ, conf = _structural_typology(max_in, max_out, linearity)
    evidence = [
        f"max in-degree {max_in}, max out-degree {max_out}",
        f"linearity {linearity:.2f} over {n_nodes} nodes / {len(edges)} edges",
    ]
    rationale = (
        f"Border/flow shape (fan-in {max_in}, fan-out {max_out}, linearity "
        f"{linearity:.2f}) is most consistent with a {typ.replace('_', ' ')} pattern."
    )
    assert typ in TYPOLOGIES
    return TypologyResult(typology=typ, confidence=conf, rationale=rationale,
                          evidence=tuple(evidence))


def _exit_path(g: nx.DiGraph) -> list[list[int]]:
    """Recover one exit path: an internal 'entry' (funded from outside) → external receiver."""
    receivers = [n for n, d in g.nodes(data=True) if d.get("role") == "receiver"]
    internals = [n for n, d in g.nodes(data=True) if d.get("role") == "internal"]
    if not receivers or not internals:
        return []
    funded = [v for u, v in g.edges() if g.nodes[u].get("role") == "sender"]
    entry = funded[0] if funded else internals[0]
    for r in receivers:
        try:
            return [[int(x) for x in nx.shortest_path(g, source=entry, target=r)]]
        except nx.NetworkXNoPath:
            continue
    return []


def _graph_stats(g: nx.DiGraph) -> dict[str, float]:
    roles = [d.get("role", "internal") for _, d in g.nodes(data=True)]
    in_deg = [dg for _, dg in g.in_degree()]
    out_deg = [dg for _, dg in g.out_degree()]
    return {
        "n_internal": float(roles.count("internal")),
        "n_senders": float(roles.count("sender")),
        "n_receivers": float(roles.count("receiver")),
        "n_edges": float(g.number_of_edges()),
        "max_in_degree": float(max(in_deg, default=0)),
        "max_out_degree": float(max(out_deg, default=0)),
    }


def candidate_from_graph(
    g: nx.DiGraph,
    subgraph_id: int,
    pu_score: float,
    node_features: np.ndarray | None,
    *,
    max_feat: int = 8,
) -> CandidateSubgraph:
    """Build an LLM-layer :class:`CandidateSubgraph` from a lead's border graph."""
    nodes = []
    for n, data in g.nodes(data=True):
        if node_features is not None:
            feats = [float(v) for v in np.asarray(node_features[n])[:max_feat]]
        else:
            feats = []
        nodes.append(SubgraphNode(node_id=int(n), role=str(data.get("role", "internal")),
                                  features=feats))
    edges = [SubgraphEdge(source=int(u), target=int(v)) for u, v in g.edges()]
    return CandidateSubgraph(
        subgraph_id=int(subgraph_id), pu_score=float(pu_score),
        nodes=nodes, edges=edges, exit_paths=_exit_path(g), stats=_graph_stats(g),
    )


def _int_id(cc_id: str, position: int) -> int:
    try:
        return int(cc_id)
    except ValueError:
        return position


def render_card(lead: Lead, g: nx.DiGraph, node_features: np.ndarray | None,
                classifier: Any) -> str:
    """Serialize → classify → validate → render one full investigative card (Markdown)."""
    candidate = candidate_from_graph(g, _int_id(lead.cc_id, lead.position), lead.score,
                                     node_features)
    classification = classify_candidate(candidate, classifier)
    report = report_from_classification(
        candidate, classification, pu_percentile=lead.percentile,
        endpoint_type=_ENDPOINT_TYPE, structural_evidence=candidate.stats,
    )
    return render_report(report)


def investigate(
    border_scores: Path,
    subgraphs: Path,
    edge_index_path: Path,
    node_features_path: Path | None,
    out_dir: Path,
    *,
    classifier: Any,
    split: str | None = None,
    top_k: int = 12,
    max_border: int = 12,
) -> list[Lead]:
    """Write a full investigative card (Markdown + PNG) per top lead into ``out_dir``."""
    leads = rank_leads(border_scores, split=split, top_k=top_k)
    import pyarrow.parquet as pq  # noqa: PLC0415

    members = [
        np.asarray(m, dtype=np.int64)
        for m in pq.read_table(subgraphs, columns=["member_idx"]).column(0).to_pylist()
    ]
    edge_index = np.load(edge_index_path)
    node_features = (
        np.load(node_features_path, mmap_mode="r") if node_features_path else None
    )
    n_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
    n_nodes = max(n_nodes, max((int(m.max()) + 1 for m in members if m.size), default=0))
    graphs = extract_lead_graphs([ld.position for ld in leads], members, edge_index,
                                 n_nodes, max_border=max_border)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Investigative report cards (border model + typology)", "",
             f"Top {len(leads)} subgraphs by border score"
             + (f" in split '{split}'." if split else ".") + " Each is a LEAD, not a finding.",
             ""]
    for rank, ld in enumerate(leads, 1):
        g = graphs[ld.position]
        stem = f"card_{rank:03d}_cc{ld.cc_id}"
        render_lead_png(g, out_dir / f"{stem}.png", f"subgraph {ld.cc_id}  score={ld.score:.3f}")
        card = render_card(ld, g, node_features, classifier)
        card += f"\n\n![subgraph {ld.cc_id}]({stem}.png)\n"
        (out_dir / f"{stem}.md").write_text(card)
        index.append(f"{rank}. **{ld.cc_id}** — score {ld.score:.4g} "
                     f"(pct {ld.percentile * 100:.2f}%) — [card]({stem}.md)")
    (out_dir / "index.md").write_text("\n".join(index) + "\n")
    return leads


def _make_classifier(kind: str) -> Any:
    if kind == "heuristic":
        return heuristic_classifier
    if kind == "bedrock":
        from ellip2.llm.bedrock_client import BedrockTypologyClient  # noqa: PLC0415

        return BedrockTypologyClient().classify
    raise SystemExit(f"unknown --llm {kind!r} (use 'heuristic' or 'bedrock')")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 4: full investigative report cards from border scores.",
    )
    p.add_argument("--border-scores", required=True, type=Path)
    p.add_argument("--subgraphs", required=True, type=Path)
    p.add_argument("--edge-index", required=True, type=Path)
    p.add_argument("--node-features", type=Path, default=None,
                   help="node_features.npy (enables binned node features in the LLM prompt)")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="test")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--max-border", type=int, default=12)
    p.add_argument("--llm", default="heuristic", choices=["heuristic", "bedrock"],
                   help="typology classifier: offline heuristic (default) or Bedrock")
    args = p.parse_args(argv)

    leads = investigate(
        args.border_scores, args.subgraphs, args.edge_index, args.node_features,
        args.out_dir, classifier=_make_classifier(args.llm),
        split=(args.split or None), top_k=args.top_k, max_border=args.max_border,
    )
    print(f"[investigate] wrote {len(leads)} report cards -> {args.out_dir}/index.md "
          f"(classifier={args.llm})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
