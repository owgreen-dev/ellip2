"""Unit tests for TIcE class-prior estimation + the sweep grid (T-010).

A synthetic SCAR Positive/Unlabeled mixture is built with a KNOWN class prior
(separable positive/negative feature distributions), and we assert TIcE recovers
the prior and label frequency within tolerance. We also assert the confidence
correction is conservative (larger π̂_p), the input guards raise, and the sweep
grid is well-formed. CPU-only, deterministic via a seeded RNG (SIGN-101). Runs
under pytest, or standalone: ``python tests/test_prior_estimation.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ellip2.pu.prior_estimation import (  # noqa: E402
    PriorEstimate,
    prior_sweep_grid,
    tice_prior,
)


def _scar_mixture(
    *,
    n_pos: int,
    n_neg: int,
    label_freq: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Separable SCAR mixture: positives ~ N(+5,1), negatives ~ N(-5,1).

    A random ``label_freq`` fraction of the positives are marked labeled (s=1);
    everything else is unlabeled. True prior = n_pos / (n_pos + n_neg).
    """
    rng = np.random.default_rng(seed)
    pos = rng.normal(5.0, 1.0, size=n_pos)
    neg = rng.normal(-5.0, 1.0, size=n_neg)
    features = np.concatenate([pos, neg])
    is_positive = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    labeled = is_positive & (rng.random(n_pos + n_neg) < label_freq)
    # Shuffle so labeled/unlabeled aren't contiguous.
    perm = rng.permutation(n_pos + n_neg)
    return features[perm], labeled[perm]


def test_recovers_known_prior_within_tolerance() -> None:
    # True prior π_p = 1000 / 4000 = 0.25; label frequency c = 0.6.
    features, labeled = _scar_mixture(n_pos=1000, n_neg=3000, label_freq=0.6, seed=0)
    est = tice_prior(features, labeled)

    assert isinstance(est, PriorEstimate)
    assert est.n_total == 4000
    # γ = observed labeled fraction ≈ 0.25 * 0.6 = 0.15.
    assert est.labeled_fraction == pytest.approx(0.15, abs=0.02)
    # A near-pure-positive subdomain recovers c ≈ 0.6 ...
    assert est.label_frequency == pytest.approx(0.6, abs=0.1)
    # ... so π̂_p = γ / ĉ ≈ 0.25.
    assert est.prior == pytest.approx(0.25, abs=0.06)
    assert 0.0 < est.prior <= 1.0


def test_prior_in_unit_interval_for_other_priors() -> None:
    # A much smaller prior should still be recovered and stay in (0, 1].
    features, labeled = _scar_mixture(n_pos=400, n_neg=3600, label_freq=0.5, seed=7)
    est = tice_prior(features, labeled)
    assert 0.0 < est.prior <= 1.0
    assert est.prior == pytest.approx(0.10, abs=0.05)


def test_confidence_correction_is_conservative() -> None:
    features, labeled = _scar_mixture(n_pos=1000, n_neg=3000, label_freq=0.6, seed=0)
    plain = tice_prior(features, labeled, delta=0.0)
    conservative = tice_prior(features, labeled, delta=0.05)
    # The Hoeffding correction lowers ĉ, which raises π̂_p (never lowers it).
    assert conservative.label_frequency <= plain.label_frequency + 1e-12
    assert conservative.prior >= plain.prior - 1e-12


def test_input_guards_raise() -> None:
    feats = np.zeros(10)
    with pytest.raises(ValueError):
        tice_prior(feats, np.zeros(10, bool))  # no positives
    with pytest.raises(ValueError):
        tice_prior(feats, np.ones(9, bool))  # length mismatch
    with pytest.raises(ValueError):
        tice_prior(feats, np.ones(10, bool), delta=1.0)  # delta out of range


def test_sweep_grid_default_is_well_formed() -> None:
    grid = prior_sweep_grid()
    assert grid.dtype == np.float64
    assert grid.shape == (5,)
    # Default spans 1e-5 .. 1e-1, one point per decade, ascending.
    np.testing.assert_allclose(grid, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], rtol=1e-9)
    assert np.all(np.diff(grid) > 0)  # strictly increasing
    assert np.all((grid > 0.0) & (grid < 1.0))


def test_sweep_grid_include_and_validation() -> None:
    grid = prior_sweep_grid(1e-4, 1e-1, num=4, include=0.05)
    assert np.all(np.diff(grid) > 0)  # still sorted + unique after insert
    assert any(abs(grid - 0.05) < 1e-12)  # the estimate was merged in
    assert grid[0] == pytest.approx(1e-4)
    assert grid[-1] == pytest.approx(1e-1)

    with pytest.raises(ValueError):
        prior_sweep_grid(0.2, 0.1)  # lo >= hi
    with pytest.raises(ValueError):
        prior_sweep_grid(1e-3, 1e-1, num=1)  # num < 2
    with pytest.raises(ValueError):
        prior_sweep_grid(0.0, 0.1)  # lo not in (0, 1)
    with pytest.raises(ValueError):
        prior_sweep_grid(1e-4, 1e-1, include=1.5)  # include out of range


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
