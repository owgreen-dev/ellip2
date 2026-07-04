"""Tests for the per-cluster suspicion scorer (ellip2.pu.cluster_score).

CPU-only, synthetic (SIGN-101). Builds cluster_features + subgraphs + split where the
member clusters of suspicious subgraphs are linearly separable from licit ones, then
checks train_main -> score_main recovers the separation with high held-out PR-AUC and
writes an (N,) cluster_scores.npy over ALL clusters (labeled + background).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import csv  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.eval.pu_metrics import pr_auc  # noqa: E402
from ellip2.pu import cluster_score  # noqa: E402

N, F, K, M = 300, 6, 20, 5  # 300 clusters; first 100 are labeled (20 subgraphs × 5)


def _write(tmp: Path) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, F)).astype(np.float32)
    ccids = [f"cc{k}" for k in range(K)]
    labels = ["suspicious" if k < 10 else "licit" for k in range(K)]
    members = [list(range(k * M, (k + 1) * M)) for k in range(K)]
    for k in range(K):
        X[np.array(members[k])] += 3.0 if k < 10 else -3.0  # separable per member

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


def test_labeled_member_mask_split_scoped(tmp_path: Path) -> None:
    a = _write(tmp_path)
    train = cluster_score.labeled_member_mask(
        a["subgraphs"], N, split_csv=a["split"], split_name="train"
    )
    test = cluster_score.labeled_member_mask(
        a["subgraphs"], N, split_csv=a["split"], split_name="test"
    )
    # train subgraphs: suspicious 0-4 + licit 10-14 -> members [0,25) ∪ [50,75)
    assert set(np.flatnonzero(train)) == set(range(0, 25)) | set(range(50, 75))
    # every labeled member is in exactly one split; background (idx>=100) never labeled
    assert not (train & test).any()
    assert not train[100:].any() and not test[100:].any()


def test_train_then_score_recovers_separation(tmp_path: Path) -> None:
    a = _write(tmp_path)
    model, scores_npy = tmp_path / "cluster_model.pkl", tmp_path / "cluster_scores.npy"

    rc = cluster_score.train_main([
        "--features", str(a["features"]), "--subgraphs", str(a["subgraphs"]),
        "--split-csv", str(a["split"]), "--out", str(model),
        "--max-iter", "100", "--min-samples-leaf", "2",
    ])
    assert rc == 0 and model.is_file()
    with open(model, "rb") as fh:
        bundle = pickle.load(fh)
    assert bundle["framing"] == cluster_score.FRAMING
    assert bundle["feature_names"] == [f"f{i}" for i in range(F)]

    rc = cluster_score.score_main([
        "--model", str(model), "--features", str(a["features"]), "--out", str(scores_npy),
    ])
    assert rc == 0 and scores_npy.is_file()

    scores = np.load(scores_npy)
    assert scores.shape == (N,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    # held-out (test-split) labeled members: suspicious cc5-9 -> [25,50); licit cc15-19 -> [75,100)
    susp_members = np.arange(25, 50)
    lic_members = np.arange(75, 100)
    members = np.concatenate([susp_members, lic_members])
    y = np.concatenate([np.ones(susp_members.size, int), np.zeros(lic_members.size, int)])
    assert pr_auc(scores[members], y) > 0.9
    assert scores[susp_members].min() > scores[lic_members].max()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_labeled_member_mask_split_scoped(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_train_then_score_recovers_separation(Path(d))
    print("ok")
