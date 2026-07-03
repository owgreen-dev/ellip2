"""Tests for the supervised subgraph classifier (ellip2.pu.train_subgraph / .score_subgraph).

CPU-only, synthetic. Builds cluster_features + subgraphs + split where suspicious
subgraphs' member clusters are linearly separable from licit ones, then checks that
pooling + the HGBM classifier recover the separation on a held-out split.
"""

from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.pu import score_subgraph, train_subgraph  # noqa: E402
from ellip2.pu.subgraph_pool import pool_subgraph_features  # noqa: E402

N, F, K, M = 200, 6, 20, 5  # 200 clusters, 6 feats, 20 subgraphs × 5 members


def _write(tmp: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, F)).astype(np.float32)
    ccids = [f"cc{k}" for k in range(K)]
    labels = ["suspicious" if k < 10 else "licit" for k in range(K)]
    members = [list(range(k * M, (k + 1) * M)) for k in range(K)]
    for k in range(K):
        X[np.array(members[k])] += 3.0 if k < 10 else -3.0  # separable

    pq.write_table(
        pa.table(
            {"idx": pa.array(np.arange(N, dtype=np.int64)),
             **{f"f{i}": pa.array(X[:, i]) for i in range(F)}}
        ),
        tmp / "cluster_features.parquet",
    )
    pq.write_table(
        pa.table(
            {"ccId": pa.array(ccids), "ccLabel": pa.array(labels),
             "n_members": pa.array([M] * K, type=pa.int64()),
             "member_idx": pa.array(members, type=pa.list_(pa.int64()))}
        ),
        tmp / "subgraphs.parquet",
    )
    # suspicious: 0-4 train, 5-9 test; licit: 10-14 train, 15-19 test
    with open(tmp / "split.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for k in range(K):
            first_half = k < 5 if k < 10 else k < 15
            w.writerow([ccids[k], labels[k], "train" if first_half else "test"])
    return {
        "features": tmp / "cluster_features.parquet",
        "subgraphs": tmp / "subgraphs.parquet",
        "split": tmp / "split.csv",
    }


def test_pool_shapes_and_separation(tmp_path: Path) -> None:
    a = _write(tmp_path)
    pooled = pool_subgraph_features(a["features"], a["subgraphs"], split_csv=a["split"])
    assert pooled.X.shape == (K, 4 * F)          # mean/max/min/std pools
    assert pooled.y.sum() == 10
    assert set(pooled.split.tolist()) == {"train", "test"}
    mean_pool = pooled.X[:, :F]                    # "mean" is the first pool block
    assert mean_pool[pooled.y == 1].mean() > mean_pool[pooled.y == 0].mean()


def test_train_then_score_recovers_separation(tmp_path: Path) -> None:
    a = _write(tmp_path)
    model, scores = tmp_path / "m.pkl", tmp_path / "s.parquet"
    rc = train_subgraph.main([
        "--features", str(a["features"]), "--subgraphs", str(a["subgraphs"]),
        "--split-csv", str(a["split"]), "--out", str(model),
        "--max-iter", "100", "--min-samples-leaf", "2",
    ])
    assert rc == 0 and model.is_file()
    with open(model, "rb") as fh:
        assert pickle.load(fh)["framing"] == "supervised_subgraph_hgbm"

    rc = score_subgraph.main([
        "--model", str(model), "--features", str(a["features"]),
        "--subgraphs", str(a["subgraphs"]), "--out", str(scores),
        "--split-csv", str(a["split"]), "--eval-split", "test",
    ])
    assert rc == 0 and scores.is_file()
    rows = pq.read_table(scores).to_pylist()
    assert len(rows) == K
    assert all(0.0 <= r["score"] <= 1.0 for r in rows)
    te_susp = [r["score"] for r in rows if r["split"] == "test" and r["label"] == 1]
    te_lic = [r["score"] for r in rows if r["split"] == "test" and r["label"] == 0]
    # perfectly separable data → held-out suspicious clearly out-score licit
    assert np.mean(te_susp) > np.mean(te_lic)
    assert min(te_susp) > max(te_lic)


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_pool_shapes_and_separation(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_train_then_score_recovers_separation(Path(d))
    print("ok")
