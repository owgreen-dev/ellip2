"""Stage 2 — non-negative PU risk estimator (Kiryo, Niu, du Plessis & Sugiyama,
NeurIPS 2017; arXiv:1703.00593).

PU learning sees only positive (P) and unlabeled (U) examples — never explicit
negatives. The *unbiased* PU risk (du Plessis et al. 2015) rewrites the negative
risk using the prior π_p::

    R_unbiased = π_p · E_P[ℓ⁺]  +  ( E_U[ℓ⁻] − π_p · E_P[ℓ⁻] )

where, on a scorer ``g`` (logits, larger = more positive):

    ℓ⁺ = ℓ(g, +1)   loss if the example were labeled positive
    ℓ⁻ = ℓ(g, −1)   loss if the example were labeled negative
    E_P, E_U        means over the positive / unlabeled minibatch

With a flexible model the empirical negative term ``E_U[ℓ⁻] − π_p·E_P[ℓ⁻]`` can go
**negative** — the model overfits the few P points and drives the risk below its
true non-negative floor. Kiryo's fix clamps that term from below::

    R_nn = π_p · E_P[ℓ⁺]  +  max{ −β , E_U[ℓ⁻] − π_p · E_P[ℓ⁻] }

with β = 0 by default (a plain non-negativity clamp). When the clamp engages,
the authors switch to *gradient ascent* on the offending term (scaled by γ) to
"de-overfit"; we expose that as :func:`nnpu_risk_for_backward` while
:func:`nnpu_risk` returns the value-correct clamped risk used for reporting and
model selection.

Pure, deterministic torch — CPU-friendly, fully differentiable. The class prior
``π_p`` is the **cluster-level** prior (small), NOT the 2.27% subgraph base rate
(plan.md Resolved decision #2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

# A surrogate loss: given logits and a target-sign flag, returns the elementwise
# loss. ``positive=True`` asks for ℓ(g, +1); ``positive=False`` for ℓ(g, −1).
SurrogateLoss = Callable[[torch.Tensor, bool], torch.Tensor]


def sigmoid_loss(logits: torch.Tensor, positive: bool) -> torch.Tensor:
    """Sigmoid surrogate ℓ(z, t) = σ(−t·z) (du Plessis et al. 2015).

    Bounded in (0, 1), smooth, and the loss recommended for unbiased/non-negative
    PU because its symmetry ``ℓ⁺ + ℓ⁻ = 1`` keeps the estimator well-behaved.

    Args:
        logits: scorer outputs ``g(x)``; larger = more positive.
        positive: ``True`` for ℓ(·, +1) = σ(−z); ``False`` for ℓ(·, −1) = σ(z).

    Returns:
        Elementwise loss, same shape as ``logits``.
    """
    sign = 1.0 if positive else -1.0
    return torch.sigmoid(-sign * logits)


@dataclass(frozen=True)
class PURiskParts:
    """Decomposed nnPU risk terms (all 0-dim tensors carrying grad).

    Attributes:
        positive_risk: ``π_p · E_P[ℓ⁺]`` — always non-negative, never clamped.
        negative_risk: ``E_U[ℓ⁻] − π_p · E_P[ℓ⁻]`` — the *unbiased* negative
            term, BEFORE the non-negativity clamp. Can be negative.
        clamped: ``True`` iff ``negative_risk < −β`` (the nnPU clamp engaged).
    """

    positive_risk: torch.Tensor
    negative_risk: torch.Tensor
    clamped: bool


def _risk_parts(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    prior: float,
    *,
    surrogate: SurrogateLoss,
    beta: float,
) -> PURiskParts:
    if not 0.0 < prior < 1.0:
        raise ValueError(f"prior (π_p) must be in (0, 1), got {prior}")
    if positive_logits.numel() == 0:
        raise ValueError("positive_logits is empty; need at least one P example")
    if unlabeled_logits.numel() == 0:
        raise ValueError("unlabeled_logits is empty; need at least one U example")

    p = positive_logits.reshape(-1)
    u = unlabeled_logits.reshape(-1)

    # E_P[ℓ⁺], E_P[ℓ⁻], E_U[ℓ⁻]
    positive_risk = prior * surrogate(p, True).mean()
    pos_neg_risk = prior * surrogate(p, False).mean()
    unl_neg_risk = surrogate(u, False).mean()

    negative_risk = unl_neg_risk - pos_neg_risk
    clamped = bool((negative_risk < -beta).item())
    return PURiskParts(positive_risk=positive_risk, negative_risk=negative_risk,
                       clamped=clamped)


def nnpu_risk(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    prior: float,
    *,
    surrogate: SurrogateLoss = sigmoid_loss,
    beta: float = 0.0,
    return_parts: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, PURiskParts]:
    """Non-negative PU risk (Kiryo 2017) — the value used for the loss/report.

    Computes ``R_nn = π_p·E_P[ℓ⁺] + max(−β, E_U[ℓ⁻] − π_p·E_P[ℓ⁻])``. When the
    negative term exceeds ``−β`` this equals the unbiased PU risk; otherwise the
    clamp engages and the returned scalar is the non-negativity-corrected risk.

    Args:
        positive_logits: scorer outputs on the positive (P) minibatch, any shape.
        unlabeled_logits: scorer outputs on the unlabeled (U) minibatch.
        prior: class prior ``π_p ∈ (0, 1)`` (cluster-level, small).
        surrogate: elementwise surrogate loss; defaults to :func:`sigmoid_loss`.
        beta: lower clamp bound (Kiryo's β ≥ 0; default 0 = plain non-negativity).
        return_parts: if ``True``, also return the :class:`PURiskParts` so callers
            can inspect whether the clamp engaged.

    Returns:
        A 0-dim torch tensor (with grad) holding the risk; or, if
        ``return_parts``, a ``(risk, parts)`` tuple.

    Raises:
        ValueError: if ``prior`` is outside ``(0, 1)`` or either minibatch is
            empty.
    """
    parts = _risk_parts(positive_logits, unlabeled_logits, prior,
                        surrogate=surrogate, beta=beta)
    neg = torch.clamp(parts.negative_risk, min=-beta)
    risk = parts.positive_risk + neg
    if return_parts:
        return risk, parts
    return risk


def upu_risk(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    prior: float,
    *,
    surrogate: SurrogateLoss = sigmoid_loss,
) -> torch.Tensor:
    """Unbiased PU risk (du Plessis et al. 2015) — no non-negativity clamp.

    ``R = π_p·E_P[ℓ⁺] + (E_U[ℓ⁻] − π_p·E_P[ℓ⁻])``. Exposed for comparison with
    :func:`nnpu_risk`: when the model overfits, this can dip below the true
    non-negative floor (and below ``nnpu_risk``), which is exactly the pathology
    the non-negative estimator corrects.
    """
    parts = _risk_parts(positive_logits, unlabeled_logits, prior,
                        surrogate=surrogate, beta=0.0)
    return parts.positive_risk + parts.negative_risk


def nnpu_risk_for_backward(
    positive_logits: torch.Tensor,
    unlabeled_logits: torch.Tensor,
    prior: float,
    *,
    surrogate: SurrogateLoss = sigmoid_loss,
    beta: float = 0.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """nnPU objective whose gradient follows Kiryo's training rule.

    When the negative term is at or above ``−β`` this matches :func:`nnpu_risk`.
    When it goes below (overfitting), Kiryo prescribes *gradient ascent* on the
    negative term to de-overfit: the returned surrogate is
    ``−γ · (E_U[ℓ⁻] − π_p·E_P[ℓ⁻])`` so that ``loss.backward()`` ascends that
    term (scaled by ``γ ∈ (0, 1]``). Its forward *value* is not the reported risk
    — use :func:`nnpu_risk` for that.

    Args:
        gamma: ascent scale γ; defaults to 1.0.
    """
    parts = _risk_parts(positive_logits, unlabeled_logits, prior,
                        surrogate=surrogate, beta=beta)
    if parts.clamped:
        return -gamma * parts.negative_risk
    return parts.positive_risk + parts.negative_risk
