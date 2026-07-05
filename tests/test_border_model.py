"""Tests for the supervised subgraph-border model (border_assembly + train/score_border).

CPU-only, synthetic. Builds a graph where a subgraph's BORDER (its external in-neighbours)
encodes its label, and checks that (a) border extraction recovers the right sender/receiver
sets and (b) the trained model separates held-out suspicious from licit subgraphs.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.data import schema  # noqa: E402
from ellip2.pu import score_border, train_border  # noqa: E402
from ellip2.pu.border_assembly import build_subgraph_batch, extract_border_sets  # noqa: E402


def test_extract_border_sets_directionality() -> None:
    # nodes 0,1 = subgraph 0's members; 2 external -> 0 (sender); 1 -> 3 external (receiver).
    edge_index = np.array([[2, 1, 0], [0, 3, 1]], dtype=np.int64)  # 2->0, 1->3, 0->1
    members = [np.array([0, 1], dtype=np.int64)]
    b = extract_border_sets(edge_index, members, n_nodes=4, cap=64)
    assert b.senders[0].tolist() == [2]     # external in-neighbour of an internal node
    assert b.receivers[0].tolist() == [3]   # external out-neighbour of an internal node


def test_border_cap_limits_set_size() -> None:
    # star: 200 external nodes all point into internal node 0 (subgraph 0).
    ext = np.arange(1, 201, dtype=np.int64)
    edge_index = np.array([ext, np.zeros(200, dtype=np.int64)], dtype=np.int64)
    b = extract_border_sets(edge_index, [np.array([0])], n_nodes=201, cap=32, seed=1)
    assert b.senders[0].size == 32          # capped
    assert b.receivers[0].size == 0


def _write(tmp: Path, n_sub: int = 60) -> dict[str, Path]:
    """Synthetic artifacts: each subgraph has 2 members + 2 border senders whose features
    encode the label (suspicious → shifted +, licit → shifted -)."""
    rng = np.random.default_rng(0)
    F = schema.N_NODE_FEATURES
    members = [np.array([2 * k, 2 * k + 1], dtype=np.int64) for k in range(n_sub)]
    # border sender nodes live in a separate id block after the members
    base = 2 * n_sub
    N = base + 2 * n_sub
    X = rng.standard_normal((N, F)).astype(np.float32)
    labels = ["suspicious" if k % 2 == 0 else "licit" for k in range(n_sub)]
    src, dst = [], []
    for k in range(n_sub):
        shift = 3.0 if labels[k] == "suspicious" else -3.0
        for j in range(2):
            snode = base + 2 * k + j
            X[snode] += shift                 # border sender carries the signal
            src.append(snode)
            dst.append(members[k][0])         # sender -> internal
    edge_index = np.array([src, dst], dtype=np.int64)

    ad = tmp / "ingest"
    ad.mkdir()
    np.save(ad / "edge_index.npy", edge_index)
    np.save(ad / "node_features.npy", X)
    pq.write_table(
        pa.table({"ccId": pa.array([f"cc{k}" for k in range(n_sub)]),
                  "ccLabel": pa.array(labels),
                  "n_members": pa.array([2] * n_sub, type=pa.int64()),
                  "member_idx": pa.array([m.tolist() for m in members], type=pa.list_(pa.int64()))}),
        ad / "subgraphs.parquet",
    )
    with open(tmp / "split.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for k in range(n_sub):
            # even k = suspicious, odd = licit; first 60% train, rest test (both classes in both)
            s = "train" if (k % 10) < 6 else "test"
            w.writerow([f"cc{k}", labels[k], s])
    return {"artifacts": ad, "subgraphs": ad / "subgraphs.parquet", "split": tmp / "split.csv"}


def test_build_batch_shapes() -> None:
    edge_index = np.array([[2, 1], [0, 3]], dtype=np.int64)
    members = [np.array([0, 1], dtype=np.int64)]
    b = extract_border_sets(edge_index, members, n_nodes=4, cap=8)
    nf = np.random.default_rng(0).standard_normal((4, schema.N_NODE_FEATURES)).astype(np.float32)
    batch = build_subgraph_batch([0], b, nf, edge_dim=schema.N_EDGE_FEATURES)
    assert batch.num_graphs == 1
    assert batch.node_x.shape == (2, schema.N_NODE_FEATURES)
    assert batch.sender_x.shape[1] == schema.N_NODE_FEATURES
    assert batch.edge_x.shape == (0, schema.N_EDGE_FEATURES)   # Phase 1: empty edges


def test_train_then_score_border_separates(tmp_path: Path) -> None:
    a = _write(tmp_path)
    model, scores = tmp_path / "border.pt", tmp_path / "s.parquet"
    rc = train_border.main([
        "--artifacts-dir", str(a["artifacts"]), "--subgraphs", str(a["subgraphs"]),
        "--split-csv", str(a["split"]), "--out", str(model),
        "--epochs", "150", "--set-hidden", "16", "--set-out", "8", "--border-cap", "8",
    ])
    assert rc == 0 and model.is_file()

    rc = score_border.main([
        "--model", str(model), "--artifacts-dir", str(a["artifacts"]),
        "--subgraphs", str(a["subgraphs"]), "--out", str(scores),
        "--split-csv", str(a["split"]), "--eval-split", "test",
    ])
    assert rc == 0 and scores.is_file()
    rows = pq.read_table(scores).to_pylist()
    te_susp = [r["score"] for r in rows if r["split"] == "test" and r["label"] == 1]
    te_lic = [r["score"] for r in rows if r["split"] == "test" and r["label"] == 0]
    assert np.mean(te_susp) > np.mean(te_lic)   # border signal recovered on held-out


def _rewrite_split_with_val(split_csv: Path, n_sub: int = 60) -> None:
    """Rewrite split.csv as train (k%10<6) / val (6-7) / test (8-9), both classes each."""
    with open(split_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for k in range(n_sub):
            label = "suspicious" if k % 2 == 0 else "licit"
            s = "train" if (k % 10) < 6 else ("val" if (k % 10) < 8 else "test")
            w.writerow([f"cc{k}", label, s])


def test_train_border_restarts_selects_by_val(tmp_path: Path) -> None:
    a = _write(tmp_path)
    _rewrite_split_with_val(a["split"])
    model = tmp_path / "border_r.pt"
    # Few epochs on purpose: keeps val sigmoids non-saturated (continuous, not exact 0/1),
    # so the val-PR-AUC selection call exercises pr_auc(scores, labels) with real args —
    # a swapped argument order would raise on the non-binary "labels".
    rc = train_border.main([
        "--artifacts-dir", str(a["artifacts"]), "--subgraphs", str(a["subgraphs"]),
        "--split-csv", str(a["split"]), "--out", str(model),
        "--epochs", "15", "--set-hidden", "16", "--set-out", "8", "--border-cap", "8",
        "--restarts", "3", "--val-split", "val",
    ])
    assert rc == 0 and model.is_file()   # best-of-3 by val, checkpoint written


def test_train_border_restarts_requires_val(tmp_path: Path) -> None:
    import pytest
    a = _write(tmp_path)   # split has train/test only, no "val"
    with pytest.raises(SystemExit):
        train_border.main([
            "--artifacts-dir", str(a["artifacts"]), "--subgraphs", str(a["subgraphs"]),
            "--split-csv", str(a["split"]), "--out", str(tmp_path / "x.pt"),
            "--epochs", "5", "--restarts", "2", "--val-split", "val",
        ])


if __name__ == "__main__":
    import tempfile
    test_extract_border_sets_directionality()
    test_border_cap_limits_set_size()
    test_build_batch_shapes()
    with tempfile.TemporaryDirectory() as d:
        test_train_then_score_border_separates(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_train_border_restarts_selects_by_val(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_train_border_restarts_requires_val(Path(d))
    print("ok")
