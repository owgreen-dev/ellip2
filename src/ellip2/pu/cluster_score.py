"""Per-cluster suspicion scorer — supervised HGBM over ``cluster_features``.

Background-discovery Gate 1 (plan ``yes-lets-do-it-glittery-coral.md``). The old
cluster-level nnPU GNN scored the 49M background clusters at ~random (test PR-AUC
0.03); this replaces it with a **grounded supervised** signal. Each *labeled*
subgraph's member clusters carry that subgraph's verdict: members of a suspicious
subgraph are positives, members of a licit subgraph are bona-fide negatives. A
``HistGradientBoostingClassifier`` (the same estimator the subgraph model uses —
:mod:`ellip2.pu.train_subgraph`) is fit on the labeled member rows of the *train*
split and evaluated on the *test* split's labeled members (PR-AUC). The pickled
model then scores **all** N clusters via :func:`score_main`, writing
``cluster_scores.npy`` ``(N,)`` — the Gate-1 signal
:func:`ellip2.discovery.background.discover_background` ranks candidates by.

Train examples are the small labeled slice, so training loads the full feature
matrix (:func:`ellip2.pu.train.load_features`). *Scoring* must cover all 49M
clusters, whose feature table decompresses to ~17GB, so it streams the parquet in
row batches (:func:`predict_cluster_scores`) — never materializing every column of
every row at once.

Example:
    python scripts/train_cluster.py \
        --features artifacts/features/cluster_features.parquet \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/cluster_model.pkl
    python scripts/score_cluster.py \
        --model artifacts/pu/cluster_model.pkl \
        --features artifacts/features/cluster_features.parquet \
        --out artifacts/pu/cluster_scores.npy
"""

from __future__ import annotations

import argparse
import pickle
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

from ellip2.data import schema
from ellip2.eval.pu_metrics import pu_metric_report
from ellip2.pu.train import _allowed_ccids, load_features, positive_mask

FRAMING = "cluster_suspicion_hgbm"


def labeled_member_mask(
    subgraphs_path: Path,
    n_nodes: int,
    *,
    split_csv: Path | None = None,
    split_name: str = "train",
) -> npt.NDArray[np.bool_]:
    """Boolean ``(N,)`` mask of clusters that are members of ANY labeled subgraph.

    The complement of :func:`ellip2.pu.train.positive_mask`'s suspicious filter:
    every labeled member (licit or suspicious) is marked, so negatives for the
    supervised fit are ``labeled & ~positive``. When ``split_csv`` is given only
    subgraphs assigned to ``split_name`` contribute, keeping the eval split's
    labels out of training.
    """
    table = pq.read_table(subgraphs_path, columns=["ccId", "ccLabel", "member_idx"])
    cc_ids = [str(v) for v in table.column("ccId").to_pylist()]
    cc_labels = table.column("ccLabel").to_pylist()
    members_col = table.column("member_idx").to_pylist()
    allowed = _allowed_ccids(split_csv, split_name) if split_csv is not None else None
    mask = np.zeros(n_nodes, dtype=bool)
    for cc_id, label, members in zip(cc_ids, cc_labels, members_col, strict=True):
        if label not in (schema.LABEL_SUSPICIOUS, schema.LABEL_LICIT):
            continue
        if allowed is not None and cc_id not in allowed:
            continue
        idx = np.asarray(members, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < n_nodes)]
        mask[idx] = True
    return mask


def predict_cluster_scores(
    model: object,
    features_path: Path,
    feature_columns: Sequence[str],
    *,
    batch_size: int = 1_000_000,
) -> npt.NDArray[np.float64]:
    """Score every cluster row of ``features_path`` → ``(N,)`` P(suspicious).

    Streams the parquet in row batches (``iter_batches``) so the whole feature
    matrix never lands in RAM at once, placing each row's score by its ``idx`` key
    so the output is aligned to cluster index regardless of file row order.
    """
    pf = pq.ParquetFile(str(features_path))
    n_nodes = pf.metadata.num_rows
    scores = np.full(n_nodes, np.nan, dtype=np.float64)
    read_cols = ["idx", *feature_columns]
    for batch in pf.iter_batches(batch_size=batch_size, columns=read_cols):
        idx = batch.column("idx").to_numpy(zero_copy_only=False).astype(np.int64)
        cols = [
            batch.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            for c in feature_columns
        ]
        xb = (
            np.column_stack(cols)
            if cols
            else np.empty((len(idx), 0), np.float32)
        )
        proba = model.predict_proba(xb)  # type: ignore[attr-defined]
        scores[idx] = proba[:, 1].astype(np.float64)
    if np.isnan(scores).any():
        raise ValueError(
            f"{int(np.isnan(scores).sum())} clusters left unscored — 'idx' in "
            f"{features_path} is not a contiguous 0..N-1 range"
        )
    return scores


def train_main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Background discovery: train the per-cluster suspicion HGBM.",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--subgraphs", required=True, type=Path, help="subgraphs.parquet")
    p.add_argument("--split-csv", required=True, type=Path, help="split.csv (id,label,split)")
    p.add_argument("--out", required=True, type=Path, help="output model .pkl")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--l2", type=float, default=0.0, help="L2 regularization")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: PLC0415

    feature_columns, X = load_features(args.features)
    n_nodes = X.shape[0]

    def _split_rows(split: str) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int_]]:
        pos = positive_mask(args.subgraphs, n_nodes, split_csv=args.split_csv, split_name=split)
        lab = labeled_member_mask(
            args.subgraphs, n_nodes, split_csv=args.split_csv, split_name=split
        )
        rows = np.flatnonzero(lab)
        return rows, pos[rows].astype(int)

    tr_rows, y_tr = _split_rows(args.train_split)
    if tr_rows.size == 0:
        raise SystemExit(f"no labeled members in '{args.train_split}' split")
    n_pos = int(y_tr.sum())
    n_neg = int((y_tr == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise SystemExit(
            f"train split needs both classes (pos={n_pos}, neg={n_neg}) — "
            "check --subgraphs/--split-csv"
        )

    weights = np.where(y_tr == 1, n_neg / max(1, n_pos), 1.0)
    clf = HistGradientBoostingClassifier(
        max_iter=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2,
        random_state=args.seed,
    )
    print(
        f"[train_cluster] N={n_nodes:,} F={X.shape[1]} train_members={tr_rows.size:,} "
        f"(pos={n_pos:,} neg={n_neg:,})",
        flush=True,
    )
    clf.fit(X[tr_rows], y_tr, sample_weight=weights)

    ev_rows, y_ev = _split_rows(args.eval_split)
    if ev_rows.size and 0 < int(y_ev.sum()) < ev_rows.size:
        scores = clf.predict_proba(X[ev_rows])[:, 1]
        report = pu_metric_report(scores, y_ev)
        print(
            f"[train_cluster] eval ({args.eval_split}, members={ev_rows.size:,}, "
            f"pos={int(y_ev.sum()):,}): "
            + ", ".join(f"{k}={v:.4f}" for k, v in report.items())
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        pickle.dump(
            {"model": clf, "framing": FRAMING, "feature_names": feature_columns},
            fh,
        )
    print(f"[train_cluster] wrote {args.out}")
    return 0


def score_main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Background discovery: score all clusters with the suspicion HGBM.",
    )
    p.add_argument("--model", required=True, type=Path, help="model .pkl from train_cluster")
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--out", required=True, type=Path, help="(N,) cluster_scores.npy")
    p.add_argument("--batch-size", type=int, default=1_000_000,
                   help="feature rows per streamed parquet batch")
    args = p.parse_args(argv)

    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    clf = bundle["model"]
    feature_columns = list(bundle["feature_names"])

    file_cols = [c for c in pq.read_schema(str(args.features)).names if c != "idx"]
    if file_cols != feature_columns:
        raise SystemExit(
            "feature columns differ from training; refusing to score "
            f"(train={len(feature_columns)} cols, now={len(file_cols)})"
        )

    scores = predict_cluster_scores(
        clf, args.features, feature_columns, batch_size=args.batch_size
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, scores)
    print(f"[score_cluster] wrote {scores.shape[0]:,} cluster scores -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(train_main())
