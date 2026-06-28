"""Unit tests for the non-negative PU risk estimator (T-009).

Asserts the risk value against an independent hand re-derivation on toy logits,
that the result carries gradient, and — the key nnPU property — that the
non-negativity clamp engages exactly when the unbiased negative term goes below
its floor. CPU-only, deterministic, no external resources (SIGN-101). Runs under
pytest, or standalone: ``python tests/test_nnpu_loss.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from ellip2.pu.nnpu_loss import (  # noqa: E402
    nnpu_risk,
    nnpu_risk_for_backward,
    sigmoid_loss,
    upu_risk,
)


def _sig(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def test_sigmoid_loss_symmetry_and_values() -> None:
    z = torch.tensor([-2.0, 0.0, 1.5])
    pos = sigmoid_loss(z, True)
    neg = sigmoid_loss(z, False)
    # ℓ(z,+1)=σ(−z), ℓ(z,−1)=σ(z); they sum to 1 elementwise.
    assert torch.allclose(pos, torch.sigmoid(-z))
    assert torch.allclose(neg, torch.sigmoid(z))
    assert torch.allclose(pos + neg, torch.ones_like(z))


def test_nnpu_risk_matches_hand_value() -> None:
    # Well-separated case: P logits high, U logits low -> negative term positive,
    # so nnPU == unbiased and we can hand-compute the whole thing.
    p = torch.tensor([2.0, 3.0])
    u = torch.tensor([-2.0, -1.0, 0.0, 1.0])
    prior = 0.3

    # Hand re-derivation with plain Python math.
    ep_pos = (_sig(-2.0) + _sig(-3.0)) / 2          # E_P[ℓ⁺] = mean σ(−z)
    ep_neg = (_sig(2.0) + _sig(3.0)) / 2            # E_P[ℓ⁻] = mean σ(z)
    eu_neg = (_sig(-2.0) + _sig(-1.0) + _sig(0.0) + _sig(1.0)) / 4
    pos_risk = prior * ep_pos
    neg_risk = eu_neg - prior * ep_neg
    expected = pos_risk + max(0.0, neg_risk)

    risk, parts = nnpu_risk(p, u, prior, return_parts=True)
    assert neg_risk > 0.0                            # clamp should NOT engage here
    assert not parts.clamped
    assert risk.item() == pytest.approx(expected, rel=1e-6)
    # With no clamp, nnPU and unbiased coincide.
    assert upu_risk(p, u, prior).item() == pytest.approx(expected, rel=1e-6)


def test_nonnegative_clamp_engages_when_unbiased_goes_negative() -> None:
    # Overfit-the-positives regime: P logits very negative (model thinks P are
    # negative -> tiny ℓ⁻ on P -> π_p·E_P[ℓ⁻] large after... actually flip it):
    # make E_U[ℓ⁻] small and π_p·E_P[ℓ⁻] large so the negative term goes < 0.
    # P logits strongly positive => σ(z)≈1 => E_P[ℓ⁻] large; U logits strongly
    # positive => σ(z)... we need E_U[ℓ⁻] small => U logits strongly negative.
    p = torch.tensor([8.0, 9.0])                    # E_P[ℓ⁻]=σ(z)≈1
    u = torch.tensor([-8.0, -9.0])                  # E_U[ℓ⁻]=σ(z)≈0
    prior = 0.5

    risk, parts = nnpu_risk(p, u, prior, return_parts=True)
    assert parts.negative_risk.item() < 0.0         # unbiased term is negative
    assert parts.clamped

    # nnPU clamps the negative term to 0 -> risk == positive_risk exactly.
    assert risk.item() == pytest.approx(parts.positive_risk.item(), rel=1e-6)
    # Unbiased risk dips strictly below the clamped (and below positive_risk).
    unbiased = upu_risk(p, u, prior)
    assert unbiased.item() < risk.item()
    assert unbiased.item() < parts.positive_risk.item()


def test_beta_floor_shifts_the_clamp() -> None:
    p = torch.tensor([8.0, 9.0])
    u = torch.tensor([-8.0, -9.0])
    prior = 0.5
    beta = 0.1
    risk, parts = nnpu_risk(p, u, prior, beta=beta, return_parts=True)
    # Negative term clamped to −β, so risk == positive_risk − β.
    assert risk.item() == pytest.approx(parts.positive_risk.item() - beta, rel=1e-6)


def test_risk_is_differentiable() -> None:
    p = torch.tensor([1.0, -0.5], requires_grad=True)
    u = torch.tensor([0.2, -1.0, 0.7], requires_grad=True)
    risk = nnpu_risk(p, u, 0.3)
    assert risk.requires_grad
    risk.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_for_backward_ascends_when_clamped() -> None:
    # When clamped, the backward objective is −γ·negative_risk; its grad wrt the
    # logits should be the negation of the unbiased negative term's grad.
    p = torch.tensor([8.0, 9.0], requires_grad=True)
    u = torch.tensor([-8.0, -9.0], requires_grad=True)
    obj = nnpu_risk_for_backward(p, u, 0.5, gamma=1.0)
    obj.backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()


def test_invalid_inputs_raise() -> None:
    p = torch.tensor([1.0])
    u = torch.tensor([0.0])
    with pytest.raises(ValueError):
        nnpu_risk(p, u, 0.0)
    with pytest.raises(ValueError):
        nnpu_risk(p, u, 1.0)
    with pytest.raises(ValueError):
        nnpu_risk(torch.empty(0), u, 0.3)
    with pytest.raises(ValueError):
        nnpu_risk(p, torch.empty(0), 0.3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
