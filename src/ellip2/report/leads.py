"""Stage 4 — investigative leads with graph visualizations from border scores.

Turns the Stage-2 border model's per-subgraph scores (``border_scores.parquet``) into a
ranked set of **investigative leads**, each rendered as a graph picture of the flagged
subgraph and its border: internal clusters in the centre, the outside **senders** that
fund it on the left, the outside **receivers** it pays out to on the right — the very
structure the model keys on (plan.md §"Subgraph-level readout"). Each lead also gets a
Markdown card (score, population percentile, structure counts) carrying the mandatory
false-positive caveat from :mod:`ellip2.report.render`.

This is the graph-visualization surface of the pipeline: model choice (trees / border MLP)
is orthogonal — the leads are always shown as graphs, built from ``edge_index.npy`` +
``subgraphs.parquet`` (see ``docs/stage2-model-choice.md``).

Rendering is headless (matplotlib ``Agg``). No network.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display, no network
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.report.render import CAVEAT_LINES  # noqa: E402

# node roles in a lead graph
_ROLE_COLOR = {"internal": "#4c72b0", "sender": "#dd8452", "receiver": "#55a868"}


@dataclass(frozen=True)
class Lead:
    """One ranked investigative lead."""

    cc_id: str
    position: int          # row in subgraphs.parquet
    score: float
    percentile: float      # population percentile in [0, 1]
    label: int             # 1 suspicious, 0 licit (validation only)
    split: str


def rank_leads(
    border_scores: Path, *, split: str | None = None, top_k: int = 20
) -> list[Lead]:
    """Top-``k`` subgraphs by border score (optionally within one split)."""
    t = pq.read_table(border_scores)
    cc = [str(v) for v in t.column("ccId").to_pylist()]
    score = t.column("score").to_numpy(zero_copy_only=False).astype(np.float64)
    label = t.column("label").to_numpy(zero_copy_only=False).astype(int)
    sp = np.array([str(v) for v in t.column("split").to_pylist()])
    # population percentile via average rank
    order = np.argsort(score, kind="stable")
    pct = np.empty(len(score), np.float64)
    pct[order] = (np.arange(len(score)) + 1) / len(score)
    keep = np.ones(len(score), bool) if split is None else (sp == split)
    idx = np.flatnonzero(keep)
    idx = idx[np.argsort(-score[idx], kind="stable")][:top_k]
    return [
        Lead(cc_id=cc[i], position=int(i), score=float(score[i]),
             percentile=float(pct[i]), label=int(label[i]), split=str(sp[i]))
        for i in idx
    ]


def _members_by_position(subgraphs: Path) -> list[np.ndarray]:
    col = pq.read_table(subgraphs, columns=["member_idx"]).column("member_idx").to_pylist()
    return [np.asarray(m, dtype=np.int64) for m in col]


def extract_lead_graphs(
    positions: Sequence[int],
    members: list[np.ndarray],
    edge_index: np.ndarray,
    n_nodes: int,
    *,
    max_border: int = 12,
) -> dict[int, nx.DiGraph]:
    """Build one :class:`networkx.DiGraph` per selected subgraph position.

    A single pass over ``edge_index`` collects each subgraph's internal edges, its border
    **sender** edges (external ``src`` -> internal) and **receiver** edges (internal ->
    external ``dst``); border nodes are capped at ``max_border`` per side for a readable
    picture. Node attribute ``role`` is ``internal`` / ``sender`` / ``receiver``.
    """
    node_sg = np.full(n_nodes, -1, dtype=np.int64)
    for pos in positions:
        node_sg[members[pos]] = pos

    s = edge_index[0].astype(np.int64, copy=False)
    d = edge_index[1].astype(np.int64, copy=False)
    sg_s, sg_d = node_sg[s], node_sg[d]
    touch = (sg_s >= 0) | (sg_d >= 0)
    s, d, sg_s, sg_d = s[touch], d[touch], sg_s[touch], sg_d[touch]

    graphs: dict[int, nx.DiGraph] = {}
    for pos in positions:
        g = nx.DiGraph()
        for m in members[pos].tolist():
            g.add_node(int(m), role="internal")
        internal = (sg_s == pos) & (sg_d == pos)
        for a, b in zip(s[internal].tolist(), d[internal].tolist(), strict=True):
            g.add_edge(int(a), int(b))
        sender = (sg_d == pos) & (sg_s != pos)
        rec = (sg_s == pos) & (sg_d != pos)
        seen_send = 0
        for a, b in zip(s[sender].tolist(), d[sender].tolist(), strict=True):
            if seen_send >= max_border and a not in g:
                continue
            if a not in g:
                g.add_node(int(a), role="sender")
                seen_send += 1
            g.add_edge(int(a), int(b))
        seen_rec = 0
        for a, b in zip(s[rec].tolist(), d[rec].tolist(), strict=True):
            if seen_rec >= max_border and b not in g:
                continue
            if b not in g:
                g.add_node(int(b), role="receiver")
                seen_rec += 1
            g.add_edge(int(a), int(b))
        graphs[pos] = g
    return graphs


def _layout(g: nx.DiGraph) -> dict[int, tuple[float, float]]:
    """Columnar layout: senders left, internal centre, receivers right."""
    cols = {"sender": -1.0, "internal": 0.0, "receiver": 1.0}
    by_role: dict[str, list[int]] = {"sender": [], "internal": [], "receiver": []}
    for n, data in g.nodes(data=True):
        by_role[data.get("role", "internal")].append(n)
    pos: dict[int, tuple[float, float]] = {}
    for role, nodes in by_role.items():
        for i, n in enumerate(sorted(nodes)):
            y = 0.0 if len(nodes) == 1 else 1.0 - 2.0 * i / (len(nodes) - 1)
            pos[n] = (cols[role], y)
    return pos


def render_lead_png(g: nx.DiGraph, out_path: Path, title: str) -> None:
    """Render a lead subgraph to a PNG (senders left, internal centre, receivers right)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    pos = _layout(g)
    colors = [_ROLE_COLOR[g.nodes[n].get("role", "internal")] for n in g.nodes]
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color="#bbbbbb", arrows=True,
                           arrowsize=8, width=0.6, node_size=140)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=140, linewidths=0)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=9, label=r) for r, c in _ROLE_COLOR.items()]
    ax.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=False)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _lead_markdown(lead: Lead, g: nx.DiGraph, png_name: str) -> str:
    roles = [g.nodes[n].get("role") for n in g.nodes]
    n_int = roles.count("internal")
    n_snd = roles.count("sender")
    n_rcv = roles.count("receiver")
    val = "suspicious" if lead.label == 1 else "licit"
    lines = [
        f"## Lead — subgraph {lead.cc_id}",
        "",
        f"- Border score: **{lead.score:.4g}** (percentile {lead.percentile * 100:.2f}%)",
        f"- Structure: {n_int} internal clusters, {n_snd} border senders, {n_rcv} receivers",
        f"- Split: {lead.split}   |   held-out label (validation only): **{val}**",
        "",
        f"![subgraph {lead.cc_id}]({png_name})",
        "",
        "### Caveats",
        *[f"- {c}" for c in CAVEAT_LINES],
        "",
    ]
    return "\n".join(lines)


def generate_leads(
    border_scores: Path,
    subgraphs: Path,
    edge_index_path: Path,
    out_dir: Path,
    *,
    split: str | None = None,
    top_k: int = 20,
    max_border: int = 12,
) -> list[Lead]:
    """Render the top-``k`` leads as PNGs + Markdown into ``out_dir``; return the leads."""
    leads = rank_leads(border_scores, split=split, top_k=top_k)
    members = _members_by_position(subgraphs)
    edge_index = np.load(edge_index_path)
    n_nodes = int(edge_index.max()) + 1 if edge_index.size else 0
    n_nodes = max(n_nodes, max((int(m.max()) + 1 for m in members if m.size), default=0))
    graphs = extract_lead_graphs(
        [ld.position for ld in leads], members, edge_index, n_nodes, max_border=max_border
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    index = ["# Investigative leads (border model)", "",
             f"Top {len(leads)} subgraphs by border score"
             + (f" in split '{split}'." if split else ".") + " Each is a LEAD, not a finding.",
             ""]
    for rank, ld in enumerate(leads, 1):
        g = graphs[ld.position]
        png = f"lead_{rank:03d}_cc{ld.cc_id}.png"
        render_lead_png(g, out_dir / png, f"subgraph {ld.cc_id}  score={ld.score:.3f}")
        (out_dir / f"lead_{rank:03d}_cc{ld.cc_id}.md").write_text(_lead_markdown(ld, g, png))
        index.append(f"{rank}. **{ld.cc_id}** — score {ld.score:.4g} "
                     f"(pct {ld.percentile * 100:.2f}%) — [card](lead_{rank:03d}_cc{ld.cc_id}.md)")
    (out_dir / "index.md").write_text("\n".join(index) + "\n")
    return leads


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Stage 4: render investigative leads (graphs) from border scores.",
    )
    p.add_argument("--border-scores", required=True, type=Path)
    p.add_argument("--subgraphs", required=True, type=Path)
    p.add_argument("--edge-index", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="test", help="rank within this split ('' = all)")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-border", type=int, default=12, help="border nodes/side drawn")
    args = p.parse_args(argv)

    leads = generate_leads(
        args.border_scores, args.subgraphs, args.edge_index, args.out_dir,
        split=(args.split or None), top_k=args.top_k, max_border=args.max_border,
    )
    n_susp = sum(1 for ld in leads if ld.label == 1)
    print(f"[report_leads] wrote {len(leads)} leads -> {args.out_dir}/index.md "
          f"({n_susp}/{len(leads)} held-out-suspicious)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
