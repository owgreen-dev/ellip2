"""Stage 2 — evaluation metrics for PU (positive-unlabeled) detection.

plan.md §7 (Evaluation). Under the SCAR assumption you **cannot** compute true
precision, because the "negative" pool is unlabeled and contaminated with hidden
positives. The metrics here are the ones that remain meaningful — and they are the
RevClassify reporting table (``final_test/f1``, ``final_test/prauc``) plus the
PU-specific recall and the Lee & Liu lift score:

    pr_auc          PR-AUC = average precision, treating known positives vs. the
                    unlabeled pool as "negatives". This is a LOWER BOUND on the
                    true PR-AUC: some unlabeled subgraphs are truly positive, so
                    they depress apparent precision. Computed exactly as the
                    average-precision sum ``Σ_n (R_n − R_{n−1}) · P_n`` (the
                    sklearn convention) so it is deterministic and hand-checkable.
    binary_f1       Binary F1 on the suspicious (positive) class, from thresholded
                    predictions: ``2·TP / (2·TP + FP + FN)``. RevClassify's F1.
    recall          Recall on held-out positives — the fraction of known positive
                    subgraphs recovered. The one quantity SCAR lets us estimate
                    unbiasedly (keep the recall positive split disjoint from the
                    prior-estimation / training splits; SIGN-103, plan.md §7c).
    lee_liu_score   Lee & Liu (2003) lift-style PU metric ``recall² / Pr(ŷ = 1)``.
                    Ranks models WITHOUT true negatives: it rewards recovering
                    positives (recall²) while penalising predicting positive
                    indiscriminately (Pr of a positive prediction over ALL data).

Label convention (shared with the rest of Stage 2): ``1`` = suspicious / positive,
``0`` = licit / unlabeled. Pure-numpy, deterministic, no external resources.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# Reporting column names mirrored from RevClassify's logged metrics.
COLUMNS: tuple[str, ...] = ("prauc", "f1", "recall", "lee_liu")


def _as_scores(scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(scores, dtype=np.float64).ravel()


def _as_binary(name: str, arr: npt.ArrayLike, n: int | None = None) -> npt.NDArray[np.int64]:
    """Coerce to a 0/1 int64 vector, rejecting anything outside {0, 1}."""
    a = np.asarray(arr).ravel()
    out = a.astype(np.int64)
    if not np.array_equal(out, a) or np.any((out != 0) & (out != 1)):
        raise ValueError(f"{name} must contain only 0/1 values")
    if n is not None and out.shape[0] != n:
        raise ValueError(f"{name} has {out.shape[0]} entries but expected {n}")
    return out


def _confusion(
    y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64]
) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` for the positive (== 1) class."""
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, fn


def pr_auc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """PR-AUC (average precision) of ``scores`` against binary ``labels``.

    Higher ``scores`` rank more positive. Computed as the average-precision sum
    ``Σ_n (R_n − R_{n−1}) · P_n`` over the distinct score thresholds (descending),
    with the curve anchored at ``(recall=0, precision=1)``. This treats the known
    positives vs. the unlabeled pool, so under SCAR it lower-bounds the true value.

    Args:
        scores: real-valued ranking scores, one per example.
        labels: 0/1 labels (``1`` == suspicious / positive).

    Returns:
        Average precision in ``[0, 1]``.

    Raises:
        ValueError: on a length mismatch or if there are no positives.
    """
    s = _as_scores(scores)
    y = _as_binary("labels", labels, s.shape[0])
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("pr_auc is undefined with zero positive labels")

    # Sort by descending score; mergesort keeps ties stable and deterministic.
    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    # Collapse to one point per distinct score (precision/recall only change at a
    # threshold boundary); the last row is always a boundary.
    boundaries = np.r_[np.where(np.diff(s_sorted))[0], s_sorted.size - 1]
    tps_b = tps[boundaries]
    fps_b = fps[boundaries]

    precision = tps_b / np.maximum(tps_b + fps_b, 1)
    recall = tps_b / n_pos

    # Anchor at (R=0, P=1); AP = Σ ΔR · P over the curve.
    recall = np.r_[0.0, recall]
    precision = np.r_[1.0, precision]
    return float(np.sum(np.diff(recall) * precision[1:]))


def binary_f1(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Binary F1 on the suspicious (positive) class from thresholded predictions.

    ``F1 = 2·TP / (2·TP + FP + FN)``. Returns ``0.0`` in the degenerate case where
    there are no positives at all (no true and no predicted positives).
    """
    yt = _as_binary("y_true", y_true)
    yp = _as_binary("y_pred", y_pred, yt.shape[0])
    tp, fp, fn = _confusion(yt, yp)
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return float(2 * tp / denom)


def recall(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Recall on the positive class — fraction of known positives recovered.

    ``recall = TP / (TP + FN)``. Returns ``0.0`` if there are no positives to
    recover (an empty held-out positive set).
    """
    yt = _as_binary("y_true", y_true)
    yp = _as_binary("y_pred", y_pred, yt.shape[0])
    tp, _fp, fn = _confusion(yt, yp)
    pos = tp + fn
    if pos == 0:
        return 0.0
    return float(tp / pos)


def lee_liu_score(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Lee & Liu (2003) PU metric ``recall² / Pr(ŷ = 1)``.

    A lift-style criterion that ranks models without true negatives: ``recall²``
    rewards recovering positives, while ``Pr(ŷ = 1)`` — the fraction of ALL
    examples predicted positive — penalises calling everything positive. Returns
    ``0.0`` when nothing is predicted positive (the metric is degenerate there).
    """
    yt = _as_binary("y_true", y_true)
    yp = _as_binary("y_pred", y_pred, yt.shape[0])
    pr_pred_pos = float(np.mean(yp)) if yp.size else 0.0
    if pr_pred_pos == 0.0:
        return 0.0
    r = recall(yt, yp)
    return float(r * r / pr_pred_pos)


def pu_metric_report(
    scores: npt.ArrayLike,
    labels: npt.ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Convenience: all four metrics at once for a fixed decision ``threshold``.

    ``scores >= threshold`` are taken as positive predictions for the
    threshold-dependent metrics (F1 / recall / Lee & Liu); ``pr_auc`` uses the raw
    scores. Returns a dict keyed by :data:`COLUMNS`.
    """
    s = _as_scores(scores)
    y = _as_binary("labels", labels, s.shape[0])
    y_pred = (s >= threshold).astype(np.int64)
    return {
        "prauc": pr_auc(s, y),
        "f1": binary_f1(y, y_pred),
        "recall": recall(y, y_pred),
        "lee_liu": lee_liu_score(y, y_pred),
    }
