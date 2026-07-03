"""End-to-end + unit tests for the Stage 2 CLIs (ellip2.pu.train / .score).

CPU-only, synthetic, no GPU / real data / S3. Builds the four Stage-0/1 artifacts
the drivers consume (cluster_features.parquet, edge_index.npy, subgraphs.parquet,
split.csv) in a tmpdir, then:

* runs train.main → asserts a checkpoint with the expected metadata is written;
* runs score.main → asserts an (N,) scores .npy is written (the discover.py
  contract), finite and in [0, 1], with subgraph metrics on a held-out split;
* unit-checks load_features (idx ordering) and positive_mask (split filtering).

Runs under pytest, or standalone: ``python tests/test_stage2_cli.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import csv  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.pu import score as score_mod  # noqa: E402
from ellip2.pu import train as train_mod  # noqa: E402

N = 30          # clusters
F = 5           # feature columns
_SUSP_TRAIN = list(range(0, 5))     # suspicious, train split
_SUSP_TEST = list(range(5, 10))     # suspicious, test split
_LICIT_TRAIN = list(range(10, 15))  # licit, train split
_LICIT_TEST = list(range(15, 20))   # licit, test split
_POS = set(_SUSP_TRAIN + _SUSP_TEST)


def _write_artifacts(tmp: Path) -> dict[str, Path]:
    """Create separable synthetic Stage-0/1 artifacts; return their paths."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, F)).astype(np.float32)
    pos_idx = np.array(sorted(_POS))
    X[pos_idx] += 3.0          # positives shifted +, so they are learnable
    mask = np.zeros(N, dtype=bool)
    mask[pos_idx] = True
    X[~mask] -= 3.0

    # shuffle rows to prove load_features re-sorts by idx
    perm = np.random.default_rng(1).permutation(N)
    feat_cols = {f"f{i}": pa.array(X[perm, i]) for i in range(F)}
    feats_table = pa.table({"idx": pa.array(perm.astype(np.int64)), **feat_cols})
    features_path = tmp / "cluster_features.parquet"
    pq.write_table(feats_table, features_path)

    # keep classes apart along directed chains within each group
    def _chain(nodes: list[int]) -> list[tuple[int, int]]:
        return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]

    edges = (
        _chain(_SUSP_TRAIN) + _chain(_SUSP_TEST)
        + _chain(_LICIT_TRAIN) + _chain(_LICIT_TEST)
    )
    edge_index = np.array(edges, dtype=np.int32).T  # (2, E)
    edge_index_path = tmp / "edge_index.npy"
    np.save(edge_index_path, edge_index)

    subgraphs_table = pa.table(
        {
            "ccId": pa.array(["s_train", "s_test", "l_train", "l_test"]),
            "ccLabel": pa.array(["suspicious", "suspicious", "licit", "licit"]),
            "n_members": pa.array([5, 5, 5, 5], type=pa.int64()),
            "member_idx": pa.array(
                [_SUSP_TRAIN, _SUSP_TEST, _LICIT_TRAIN, _LICIT_TEST],
                type=pa.list_(pa.int64()),
            ),
        }
    )
    subgraphs_path = tmp / "subgraphs.parquet"
    pq.write_table(subgraphs_table, subgraphs_path)

    split_path = tmp / "split.csv"
    with open(split_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "label", "split"])
        writer.writerows(
            [
                ["s_train", "suspicious", "train"],
                ["s_test", "suspicious", "test"],
                ["l_train", "licit", "train"],
                ["l_test", "licit", "test"],
            ]
        )

    return {
        "features": features_path,
        "edge_index": edge_index_path,
        "subgraphs": subgraphs_path,
        "split": split_path,
    }


def test_load_features_sorts_by_idx(tmp_path: Path) -> None:
    a = _write_artifacts(tmp_path)
    cols, X = train_mod.load_features(a["features"])
    assert cols == [f"f{i}" for i in range(F)]
    assert X.shape == (N, F)
    # row 0 must be cluster 0 (a positive → shifted +), despite shuffled parquet
    assert X[0].mean() > 0.0


def test_positive_mask_split_filtering(tmp_path: Path) -> None:
    a = _write_artifacts(tmp_path)
    full = train_mod.positive_mask(a["subgraphs"], N)
    assert set(np.flatnonzero(full)) == _POS

    train_only = train_mod.positive_mask(
        a["subgraphs"], N, split_csv=a["split"], split_name="train"
    )
    assert set(np.flatnonzero(train_only)) == set(_SUSP_TRAIN)  # no test leakage


def test_train_then_score_end_to_end(tmp_path: Path) -> None:
    a = _write_artifacts(tmp_path)
    model_path = tmp_path / "model.pt"
    scores_path = tmp_path / "scores.npy"

    rc = train_mod.main(
        [
            "--features", str(a["features"]),
            "--edge-index", str(a["edge_index"]),
            "--subgraphs", str(a["subgraphs"]),
            "--split-csv", str(a["split"]),
            "--split-name", "train",
            "--out", str(model_path),
            "--prior", "0.3",
            "--epochs", "150",
            "--lr", "1e-2",
            "--hidden", "16",
            "--emb-dim", "8",
        ]
    )
    assert rc == 0
    assert model_path.is_file()

    import torch

    ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    extra = ckpt["extra"]
    assert extra["framing"] == "cluster_nnpu_minibatch"
    assert extra["in_dim"] == F
    assert extra["feature_columns"] == [f"f{i}" for i in range(F)]
    assert extra["loss_last"] < extra["loss_first"]  # learning happened

    rc = score_mod.main(
        [
            "--model", str(model_path),
            "--features", str(a["features"]),
            "--edge-index", str(a["edge_index"]),
            "--out", str(scores_path),
            "--subgraphs", str(a["subgraphs"]),
            "--split-csv", str(a["split"]),
            "--eval-split", "test",
        ]
    )
    assert rc == 0
    assert scores_path.is_file()

    scores = np.load(scores_path)
    assert scores.shape == (N,)               # discover.py (N,) contract
    assert np.isfinite(scores).all()
    assert ((scores >= 0.0) & (scores <= 1.0)).all()
    # held-out suspicious clusters should out-score held-out licit ones
    assert scores[_SUSP_TEST].mean() > scores[_LICIT_TEST].mean()


def test_score_rejects_mismatched_feature_columns(tmp_path: Path) -> None:
    a = _write_artifacts(tmp_path)
    model_path = tmp_path / "model.pt"
    train_mod.main(
        [
            "--features", str(a["features"]),
            "--edge-index", str(a["edge_index"]),
            "--subgraphs", str(a["subgraphs"]),
            "--out", str(model_path),
            "--prior", "0.3",
            "--epochs", "5",
            "--hidden", "16",
            "--emb-dim", "8",
        ]
    )
    # write features with an extra column -> score must refuse
    table = pq.read_table(a["features"])
    table = table.append_column("f_extra", pa.array(np.zeros(N, dtype=np.float32)))
    bad = tmp_path / "bad_features.parquet"
    pq.write_table(table, bad)
    try:
        score_mod.main(
            [
                "--model", str(model_path),
                "--features", str(bad),
                "--edge-index", str(a["edge_index"]),
                "--out", str(tmp_path / "s.npy"),
            ]
        )
        raise AssertionError("expected SystemExit on column mismatch")
    except SystemExit:
        pass


def _run_standalone() -> int:
    import tempfile

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            try:
                t(Path(d))
                print(f"PASS {t.__name__}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"FAIL {t.__name__}: {e!r}")
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
