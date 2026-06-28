"""Unit tests for Stage 2 PU evaluation metrics (T-014).

Each metric is checked against a hand-computed value on a small score/label
vector, plus the degenerate guards. Pure-numpy, CPU-only, no external resources
(SIGN-101).

    pr_auc        average precision Σ ΔR·P; the worked example resolves to 5/6.
    binary_f1     2·TP/(2·TP+FP+FN) on the suspicious class.
    recall        TP/(TP+FN) of held-out positives.
    lee_liu_score recall² / Pr(ŷ = 1).

Runs under pytest, or standalone: ``python tests/test_pu_metrics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ellip2.eval.pu_metrics import (  # noqa: E402
    COLUMNS,
    binary_f1,
    lee_liu_score,
    pr_auc,
    pu_metric_report,
    recall,
)


def test_pr_auc_hand_value() -> None:
    # scores descending: labels [1,0,1,0], n_pos=2.
    # precision at boundaries [1, 1/2, 2/3, 1/2], recall [1/2,1/2,1,1]
    # AP = 0.5*1 + 0*0.5 + 0.5*(2/3) + 0*0.5 = 5/6.
    scores = [0.9, 0.8, 0.7, 0.6]
    labels = [1, 0, 1, 0]
    assert pr_auc(scores, labels) == pytest.approx(5.0 / 6.0)


def test_pr_auc_perfect_and_unordered() -> None:
    # A perfect ranking (all positives above all negatives) gives AP == 1.0,
    # regardless of input order (the function sorts internally).
    scores = [0.1, 0.9, 0.2, 0.8]
    labels = [0, 1, 0, 1]
    assert pr_auc(scores, labels) == pytest.approx(1.0)


def test_pr_auc_ties_are_collapsed() -> None:
    # Tied scores form a single threshold boundary: 2 pos / 2 neg all at 0.5 ->
    # precision 2/4 = 0.5 across the whole tied block, recall jumps 0 -> 1.
    assert pr_auc([0.5, 0.5, 0.5, 0.5], [1, 0, 1, 0]) == pytest.approx(0.5)


def test_pr_auc_requires_positive() -> None:
    with pytest.raises(ValueError):
        pr_auc([0.4, 0.6], [0, 0])


def test_pr_auc_length_mismatch() -> None:
    with pytest.raises(ValueError):
        pr_auc([0.4, 0.6, 0.1], [1, 0])


def test_binary_f1_hand_value() -> None:
    # y_true [1,0,1,0,1], y_pred [1,0,1,1,0]: TP=2, FP=1, FN=1 -> F1 = 2/3.
    yt = [1, 0, 1, 0, 1]
    yp = [1, 0, 1, 1, 0]
    assert binary_f1(yt, yp) == pytest.approx(2.0 / 3.0)


def test_binary_f1_perfect_and_empty() -> None:
    assert binary_f1([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    # No positives anywhere -> degenerate, defined as 0.0.
    assert binary_f1([0, 0, 0], [0, 0, 0]) == 0.0


def test_recall_hand_value() -> None:
    # Same vectors: TP=2, FN=1 -> recall = 2/3.
    assert recall([1, 0, 1, 0, 1], [1, 0, 1, 1, 0]) == pytest.approx(2.0 / 3.0)


def test_recall_no_positives() -> None:
    assert recall([0, 0, 0], [1, 1, 0]) == 0.0


def test_lee_liu_hand_value() -> None:
    # recall = 2/3, Pr(pred=1) = 3/5 -> (4/9)/(3/5) = 20/27.
    yt = [1, 0, 1, 0, 1]
    yp = [1, 0, 1, 1, 0]
    assert lee_liu_score(yt, yp) == pytest.approx(20.0 / 27.0)


def test_lee_liu_rewards_recall_penalises_overprediction() -> None:
    yt = [1, 1, 0, 0]
    # Predicting only the true positives beats predicting everything positive,
    # because the denominator Pr(pred=1) grows while recall is already 1.
    selective = lee_liu_score(yt, [1, 1, 0, 0])  # recall 1, Pr 1/2 -> 2.0
    indiscriminate = lee_liu_score(yt, [1, 1, 1, 1])  # recall 1, Pr 1 -> 1.0
    assert selective == pytest.approx(2.0)
    assert indiscriminate == pytest.approx(1.0)
    assert selective > indiscriminate


def test_lee_liu_no_predicted_positive() -> None:
    assert lee_liu_score([1, 0, 1], [0, 0, 0]) == 0.0


def test_rejects_non_binary_labels() -> None:
    with pytest.raises(ValueError):
        binary_f1([0, 2, 1], [0, 1, 1])
    with pytest.raises(ValueError):
        recall([0, 1, 1], [0, 1, 0.5])


def test_report_bundles_all_columns() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random(20)
    labels = (rng.random(20) < 0.4).astype(int)
    report = pu_metric_report(scores, labels, threshold=0.5)
    assert set(report) == set(COLUMNS)
    # Cross-check each bundled value against the standalone functions.
    y_pred = (scores >= 0.5).astype(int)
    assert report["prauc"] == pytest.approx(pr_auc(scores, labels))
    assert report["f1"] == pytest.approx(binary_f1(labels, y_pred))
    assert report["recall"] == pytest.approx(recall(labels, y_pred))
    assert report["lee_liu"] == pytest.approx(lee_liu_score(labels, y_pred))
    for v in report.values():
        assert np.isfinite(v)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
