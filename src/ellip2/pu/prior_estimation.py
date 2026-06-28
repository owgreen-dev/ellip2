"""Stage 2 — class-prior (π_p) estimation for PU learning + a sweep-grid helper.

The non-negative PU risk (:mod:`ellip2.pu.nnpu_loss`) needs a class prior
``π_p = P(y = +1)`` to weight the positive risk. We never observe negatives, so
``π_p`` must be *estimated* from the Positive (labeled) and Unlabeled samples —
the **mixture-proportion estimation** (MPE) problem.

We implement **TIcE** (Bekker & Davis, *Estimating the Class Prior in Positive
and Unlabeled Data through Decision Tree Induction*, AAAI 2018) in the SCAR
("Selected Completely At Random") setting: each example carries ``s ∈ {0, 1}``
where ``s = 1`` marks a *labeled positive* and ``s = 0`` an unlabeled example,
and the labeled positives are a random ``c``-fraction of the true positives::

    c  = P(s = 1 | y = 1)            label frequency
    γ  = P(s = 1)                    observed labeled fraction
    π_p = P(y = 1) = γ / c           the class prior we want

**TIcE primitive — bounding c from a subdomain.** Within any subdomain ``T`` of
feature space the labeled examples are a ``c``-fraction of the *positives* in
``T``, so ``#labeled_T = c · #pos_T``. Since ``#pos_T ≤ #total_T``::

    #labeled_T / #total_T  ≤  #labeled_T / #pos_T  =  c

i.e. the labeled fraction of *any* subdomain is a **lower bound** on ``c``, and
the bound is tightest on a subdomain that is (nearly) pure-positive. TIcE
therefore induces informative subdomains with a small decision tree and takes::

    ĉ = max_T ( #labeled_T / #total_T  −  correction(|T|, δ) )

The optional Hoeffding-style ``correction`` makes ``ĉ`` a high-confidence lower
bound; with ``delta = 0`` it is omitted (the plain max). ``ĉ`` is floored at the
whole-sample ``γ`` (valid since ``c ≥ γ`` always), which keeps ``π̂_p = γ / ĉ``
in ``(0, 1]``.

The estimate is a **starting point**, not gospel: graph heterophily violates the
irreducibility assumption and biases ``π_p`` upward (plan.md §2; Wu et al., ICML
2024), and ``π_p`` is the small **cluster-level** prior, NOT the 2.27% subgraph
base rate (Resolved decision #2). So we also provide :func:`prior_sweep_grid`,
which builds the log-spaced ``π_p ∈ {1e-1 … 1e-5}`` grid the pipeline sweeps in
both directions around the estimate.

Pure NumPy — CPU-friendly, deterministic, no external resources.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# Smallest label frequency we will divide by, so π̂_p = γ / ĉ never blows up.
_C_FLOOR = 1e-6


@dataclass(frozen=True)
class PriorEstimate:
    """Result of :func:`tice_prior`.

    Attributes:
        prior: estimated class prior ``π̂_p = γ / ĉ ∈ (0, 1]``.
        label_frequency: estimated ``ĉ = P(s = 1 | y = 1)`` (the TIcE primitive).
        labeled_fraction: observed ``γ = P(s = 1)`` in the full sample.
        n_labeled: number of labeled-positive examples (``s = 1``).
        n_total: total number of examples (labeled + unlabeled).
        n_subdomains: number of candidate subdomains the tree induced.
    """

    prior: float
    label_frequency: float
    labeled_fraction: float
    n_labeled: int
    n_total: int
    n_subdomains: int


def _as_2d(features: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Coerce ``(N,)`` or ``(N, D)`` features to a 2-D float64 array."""
    arr = np.asarray(features, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"features must be 1-D or 2-D, got shape {arr.shape}")
    return arr


def _subdomain_masks(
    features: npt.NDArray[np.float64],
    labeled: npt.NDArray[np.bool_],
    *,
    max_depth: int,
    min_subset: int,
) -> list[npt.NDArray[np.bool_]]:
    """Induce candidate subdomains with a small greedy decision tree.

    The full domain is always a candidate. Each tree node is split on the single
    feature/median threshold whose child is richest in labeled examples (TIcE's
    "maximize the labeled fraction" objective); both children become candidates
    and are recursed into, down to ``max_depth``. Returns the boolean membership
    mask of every node.
    """
    n = features.shape[0]
    n_features = features.shape[1]
    masks: list[npt.NDArray[np.bool_]] = [np.ones(n, dtype=bool)]

    def recurse(mask: npt.NDArray[np.bool_], depth: int) -> None:
        if depth >= max_depth:
            return
        idx = np.flatnonzero(mask)
        if idx.size < 2 * min_subset:
            return
        lab = labeled[idx]
        best_frac = -1.0
        best_feature = -1
        best_threshold = 0.0
        for j in range(n_features):
            vals = features[idx, j]
            threshold = float(np.median(vals))
            left = vals <= threshold
            n_left = int(left.sum())
            if n_left < min_subset or n_left > idx.size - min_subset:
                continue  # degenerate split (e.g. constant feature)
            for child in (left, ~left):
                frac = float(lab[child].mean())
                if frac > best_frac:
                    best_frac, best_feature, best_threshold = frac, j, threshold
        if best_feature < 0:
            return
        left = features[idx, best_feature] <= best_threshold
        for child in (left, ~left):
            child_mask = np.zeros(n, dtype=bool)
            child_mask[idx[child]] = True
            masks.append(child_mask)
            recurse(child_mask, depth + 1)

    recurse(masks[0], 0)
    return masks


def tice_prior(
    features: npt.NDArray[np.float64],
    labeled: npt.NDArray[np.bool_],
    *,
    delta: float = 0.0,
    max_depth: int = 4,
    min_subset: int = 5,
) -> PriorEstimate:
    """Estimate the class prior ``π_p`` via TIcE (Bekker & Davis, 2018).

    Args:
        features: per-example features, shape ``(N,)`` or ``(N, D)``. Used only
            to induce subdomains; their units/anonymization do not matter.
        labeled: boolean mask of length ``N``; ``True`` = labeled positive
            (``s = 1``), ``False`` = unlabeled (``s = 0``).
        delta: confidence level in ``[0, 1)`` for the Hoeffding lower-bound
            correction ``sqrt(ln(1/δ) / (2·|T|))`` subtracted from each
            subdomain's labeled fraction. ``0`` (default) disables the
            correction (plain max). Smaller ``delta`` ⇒ more conservative ``ĉ``
            ⇒ larger ``π̂_p``.
        max_depth: depth of the subdomain-inducing tree.
        min_subset: minimum examples a subdomain must contain to be considered
            (guards against tiny, noise-dominated regions).

    Returns:
        A :class:`PriorEstimate`.

    Raises:
        ValueError: if shapes disagree, ``labeled`` has no positives, or
            ``delta`` is outside ``[0, 1)``.
    """
    feats = _as_2d(features)
    lab = np.asarray(labeled, dtype=bool).reshape(-1)
    if lab.shape[0] != feats.shape[0]:
        raise ValueError(
            f"labeled length {lab.shape[0]} != n_examples {feats.shape[0]}"
        )
    if not 0.0 <= delta < 1.0:
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    n_total = lab.shape[0]
    n_labeled = int(lab.sum())
    if n_labeled == 0:
        raise ValueError("labeled has no positives (s=1); cannot estimate prior")

    gamma = n_labeled / n_total
    masks = _subdomain_masks(feats, lab, max_depth=max_depth, min_subset=min_subset)

    # c ≥ γ always (the whole sample is a valid subdomain), so floor ĉ at γ.
    best_c = gamma
    for mask in masks:
        size = int(mask.sum())
        if size < min_subset:
            continue
        frac = float(lab[mask].mean())
        if delta > 0.0:
            frac -= math.sqrt(math.log(1.0 / delta) / (2.0 * size))
        best_c = max(best_c, frac)

    label_frequency = max(best_c, _C_FLOOR)
    prior = min(1.0, gamma / label_frequency)
    return PriorEstimate(
        prior=prior,
        label_frequency=label_frequency,
        labeled_fraction=gamma,
        n_labeled=n_labeled,
        n_total=n_total,
        n_subdomains=len(masks),
    )


def prior_sweep_grid(
    lo: float = 1e-5,
    hi: float = 1e-1,
    *,
    num: int = 5,
    include: float | None = None,
) -> npt.NDArray[np.float64]:
    """Log-spaced grid of candidate priors to sweep (plan.md §2: ``1e-1 … 1e-5``).

    The cluster-level ``π_p`` is genuinely uncertain (heterophily biases any point
    estimate, and it must NOT be anchored to the 2.27% subgraph rate), so the
    pipeline sweeps a wide geometric grid in both directions and selects by
    PR-AUC. This returns that grid, sorted ascending.

    Args:
        lo: smallest prior (inclusive), in ``(0, 1)``.
        hi: largest prior (inclusive), in ``(0, 1)``; must exceed ``lo``.
        num: number of geometrically spaced points (default ``5`` ⇒ one per
            decade for the ``1e-1 … 1e-5`` default).
        include: an extra prior (e.g. a TIcE estimate) to merge into the grid;
            de-duplicated and re-sorted. ``None`` to skip.

    Returns:
        A sorted, unique float64 array of priors, each in ``(0, 1)``.

    Raises:
        ValueError: if ``lo``/``hi`` are not in ``(0, 1)`` with ``lo < hi``,
            ``num < 2``, or ``include`` is outside ``(0, 1)``.
    """
    if not 0.0 < lo < 1.0:
        raise ValueError(f"lo must be in (0, 1), got {lo}")
    if not 0.0 < hi < 1.0:
        raise ValueError(f"hi must be in (0, 1), got {hi}")
    if lo >= hi:
        raise ValueError(f"lo ({lo}) must be < hi ({hi})")
    if num < 2:
        raise ValueError(f"num must be >= 2, got {num}")

    grid = np.geomspace(lo, hi, num=num, dtype=np.float64)
    if include is not None:
        if not 0.0 < include < 1.0:
            raise ValueError(f"include must be in (0, 1), got {include}")
        grid = np.append(grid, np.float64(include))
    return np.unique(grid)
