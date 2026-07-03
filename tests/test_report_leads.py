"""Tests for the investigative-leads / graph-viz layer (ellip2.report.leads).

CPU-only, synthetic, headless (matplotlib Agg). Builds tiny border_scores + subgraphs +
edge_index and checks that leads are ranked correctly, the per-subgraph border graph has the
right node roles, and PNG + Markdown artifacts are written.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.report.leads import (  # noqa: E402
    extract_lead_graphs,
    generate_leads,
    rank_leads,
)


def _write(tmp: Path) -> dict[str, Path]:
    # 3 subgraphs, 2 members each; ids 0..5 members, 6..11 border nodes.
    members = [[0, 1], [2, 3], [4, 5]]
    # subgraph 0: 6->0 sender, 1->7 receiver, 0->1 internal
    # subgraph 1: 8->2 sender ; subgraph 2: 4->5 internal only
    src = [6, 1, 0, 8, 4]
    dst = [0, 7, 1, 2, 5]
    np.save(tmp / "edge_index.npy", np.array([src, dst], dtype=np.int64))
    pq.write_table(
        pa.table({"ccId": pa.array(["a", "b", "c"]),
                  "ccLabel": pa.array(["suspicious", "licit", "suspicious"]),
                  "n_members": pa.array([2, 2, 2], type=pa.int64()),
                  "member_idx": pa.array(members, type=pa.list_(pa.int64()))}),
        tmp / "subgraphs.parquet",
    )
    pq.write_table(
        pa.table({"ccId": pa.array(["a", "b", "c"]),
                  "score": pa.array([0.9, 0.1, 0.5]),
                  "label": pa.array([1, 0, 1]),
                  "split": pa.array(["test", "test", "test"])}),
        tmp / "border_scores.parquet",
    )
    return {"scores": tmp / "border_scores.parquet",
            "subgraphs": tmp / "subgraphs.parquet",
            "edge_index": tmp / "edge_index.npy"}


def test_rank_leads_orders_by_score(tmp_path: Path) -> None:
    a = _write(tmp_path)
    leads = rank_leads(a["scores"], split="test", top_k=3)
    assert [ld.cc_id for ld in leads] == ["a", "c", "b"]     # 0.9 > 0.5 > 0.1
    assert leads[0].percentile == 1.0                         # top score = 100th pct


def test_extract_lead_graph_roles(tmp_path: Path) -> None:
    a = _write(tmp_path)
    members = [np.array([0, 1]), np.array([2, 3]), np.array([4, 5])]
    edge_index = np.load(a["edge_index"])
    graphs = extract_lead_graphs([0], members, edge_index, n_nodes=12, max_border=12)
    g = graphs[0]
    roles = {n: g.nodes[n]["role"] for n in g.nodes}
    assert roles[0] == "internal" and roles[1] == "internal"
    assert roles[6] == "sender"        # 6 -> 0
    assert roles[7] == "receiver"      # 1 -> 7
    assert g.has_edge(0, 1)            # internal edge


def test_generate_leads_writes_png_and_markdown(tmp_path: Path) -> None:
    a = _write(tmp_path)
    out = tmp_path / "leads"
    leads = generate_leads(a["scores"], a["subgraphs"], a["edge_index"], out,
                           split="test", top_k=2, max_border=12)
    assert len(leads) == 2
    assert (out / "index.md").is_file()
    pngs = list(out.glob("*.png"))
    cards = list(out.glob("lead_*.md"))
    assert len(pngs) == 2 and len(cards) == 2
    assert all(p.stat().st_size > 0 for p in pngs)           # non-empty images
    assert "INVESTIGATIVE LEAD" in (out / cards[0].name).read_text()  # caveat present


if __name__ == "__main__":
    import tempfile
    for t in (test_rank_leads_orders_by_score, test_extract_lead_graph_roles,
              test_generate_leads_writes_png_and_markdown):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print("ok")
