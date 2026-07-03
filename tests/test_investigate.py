"""Tests for the full investigative report cards (ellip2.report.investigate).

CPU-only, offline (heuristic classifier — no Bedrock/network), headless matplotlib. Checks
the candidate assembly, the offline typology classifier, and that a full card (typology +
exit path + structural evidence + caveat) and its graph PNG are written.
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

from ellip2.llm.bedrock_client import TYPOLOGIES  # noqa: E402
from ellip2.report.investigate import (  # noqa: E402
    candidate_from_graph,
    heuristic_classifier,
    investigate,
)
from ellip2.report.leads import extract_lead_graphs  # noqa: E402


def _write(tmp: Path) -> dict[str, Path]:
    # subgraph a: fan-out (1 internal -> 3 receivers) => layering_smurfing
    # members 0,1 (a); 2,3 (b). border nodes 4..9.
    members = [[0, 1], [2, 3]]
    src = [4, 0, 0, 0, 5, 2]
    dst = [0, 5, 6, 7, 2, 8]   # a: 4->0 sender, 0->{5,6,7} receivers; b: 5->2, 2->8
    np.save(tmp / "edge_index.npy", np.array([src, dst], dtype=np.int64))
    np.save(tmp / "node_features.npy",
            np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32))
    pq.write_table(
        pa.table({"ccId": pa.array(["a", "b"]),
                  "ccLabel": pa.array(["suspicious", "licit"]),
                  "n_members": pa.array([2, 2], type=pa.int64()),
                  "member_idx": pa.array(members, type=pa.list_(pa.int64()))}),
        tmp / "subgraphs.parquet",
    )
    pq.write_table(
        pa.table({"ccId": pa.array(["a", "b"]), "score": pa.array([0.95, 0.2]),
                  "label": pa.array([1, 0]), "split": pa.array(["test", "test"])}),
        tmp / "border_scores.parquet",
    )
    return {"scores": tmp / "border_scores.parquet",
            "subgraphs": tmp / "subgraphs.parquet",
            "edge_index": tmp / "edge_index.npy",
            "node_features": tmp / "node_features.npy"}


def test_heuristic_classifier_returns_valid_typology() -> None:
    serialized = {"n_nodes": 4, "edges": [{"s": 0, "t": 5}, {"s": 0, "t": 6}, {"s": 0, "t": 7}]}
    res = heuristic_classifier(serialized)
    assert res.typology in TYPOLOGIES
    assert res.typology == "layering_smurfing"      # 1 -> many fan-out
    assert 0.0 <= res.confidence <= 1.0 and res.rationale


def test_candidate_has_roles_and_exit_path(tmp_path: Path) -> None:
    a = _write(tmp_path)
    members = [np.array([0, 1]), np.array([2, 3])]
    ei = np.load(a["edge_index"])
    nf = np.load(a["node_features"])
    g = extract_lead_graphs([0], members, ei, n_nodes=10, max_border=12)[0]
    cand = candidate_from_graph(g, subgraph_id=0, pu_score=0.95, node_features=nf)
    roles = {n.node_id: n.role for n in cand.nodes}
    assert roles[4] == "sender" and roles[5] == "receiver" and roles[0] == "internal"
    assert cand.exit_paths and cand.exit_paths[0][-1] in (5, 6, 7)   # ends at a receiver
    assert all(len(n.features) == 8 for n in cand.nodes)


def test_investigate_writes_full_cards(tmp_path: Path) -> None:
    a = _write(tmp_path)
    out = tmp_path / "cards"
    leads = investigate(a["scores"], a["subgraphs"], a["edge_index"], a["node_features"],
                        out, classifier=heuristic_classifier, split="test", top_k=2)
    assert len(leads) == 2
    assert (out / "index.md").is_file()
    cards = sorted(out.glob("card_*.md"))
    pngs = sorted(out.glob("card_*.png"))
    assert len(cards) == 2 and len(pngs) == 2
    top = cards[0].read_text()
    assert "Typology:" in top                       # LLM typology section
    assert "Exit path" in top                        # Stage-3 style exit path
    assert "Structural evidence" in top
    assert "INVESTIGATIVE LEAD" in top               # mandatory caveat
    assert "![subgraph" in top                       # embedded graph image


if __name__ == "__main__":
    import tempfile
    test_heuristic_classifier_returns_valid_typology()
    for t in (test_candidate_has_roles_and_exit_path, test_investigate_writes_full_cards):
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
    print("ok")
