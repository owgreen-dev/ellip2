"""T-030 — held-out-recovery proxy eval for background discovery.

A label-free yardstick for :func:`ellip2.discovery.background.discover_background`.
Everything discovery surfaces is, by construction, NOT a labeled subgraph — so there
is no ground truth to score it against directly. The proxy: hold the test-split
suspicious subgraphs out of the cluster scorer, run discovery excluding ONLY the
train-split members (so the held-out clusters stay seedable), then measure how many
of those held-out suspicious subgraphs a discovered candidate actually re-covers,
against the expectation of a random set of the same size.

* :func:`recovery_rate` — fraction of TARGET subgraphs with ≥1 member in the union of
  the discovered member sets, plus the random-baseline expectation (a set of the same
  ``|discovered union|`` drawn uniformly from the ``N`` clusters). A recovery rate far
  above baseline is evidence discovery finds real structure, not noise.
* :func:`main` — the CLI: read ``discovered_subgraphs.parquet`` +
  ``subgraphs.parquet`` + ``split.csv``, take the eval-split suspicious subgraphs as
  targets, and report recovery vs baseline.

The real eval RUN is a manual box step after the loop; the ``main`` here is the
offline harness that computes the numbers.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ellip2.data import schema

MemberSets = Iterable[npt.ArrayLike]


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of :func:`recovery_rate`.

    Attributes:
        n_targets: number of target (held-out) subgraphs scored.
        n_recovered: targets with ≥1 member in the discovered union.
        recovery_rate: ``n_recovered / n_targets`` (0.0 when there are no targets).
        discovered_union_size: ``|union of discovered member sets|`` (deduplicated,
            bounded to ``[0, n_nodes)``).
        n_nodes: total cluster count used for the baseline.
        baseline_rate: expected recovery rate of a random set of the same union size.
        lift: ``recovery_rate / baseline_rate`` (inf when baseline is 0 and recovery
            is positive, 0.0 when both are 0).
    """

    n_targets: int
    n_recovered: int
    recovery_rate: float
    discovered_union_size: int
    n_nodes: int
    baseline_rate: float
    lift: float


def _member_arrays(member_sets: MemberSets, n_nodes: int) -> list[npt.NDArray[np.int64]]:
    out: list[npt.NDArray[np.int64]] = []
    for m in member_sets:
        arr = np.asarray(m, dtype=np.int64).ravel()
        arr = arr[(arr >= 0) & (arr < n_nodes)]
        out.append(arr)
    return out


def _baseline_hit_prob(m: int, d: int, n: int) -> float:
    """P(a size-``m`` target shares ≥1 member with a random size-``d`` draw from ``n``).

    Hypergeometric complement: ``1 - C(N-m, D) / C(N, D)`` computed as the running
    product ``prod_i (N-D-i)/(N-i)`` (the chance none of the ``m`` members is drawn).
    A guaranteed hit (``m > N-D``) short-circuits to 1.0; an empty target to 0.0.
    """
    if m <= 0:
        return 0.0
    if m > n - d:  # can't fit all m members outside the drawn set -> certain hit
        return 1.0
    p_miss = 1.0
    for i in range(m):
        p_miss *= (n - d - i) / (n - i)
    return 1.0 - p_miss


def recovery_rate(
    discovered_member_sets: MemberSets,
    target_member_sets: MemberSets,
    n_nodes: int,
) -> RecoveryResult:
    """Fraction of target subgraphs recovered by the discovered member sets vs baseline.

    A target is *recovered* iff at least one of its member clusters appears in the
    union of the discovered member sets. The random baseline is the expected recovery
    of a set of the same union size drawn uniformly from the ``n_nodes`` clusters —
    the yardstick a real discovery run must beat.

    Args:
        discovered_member_sets: iterable of discovered candidates' member idx arrays.
        target_member_sets: iterable of the held-out target subgraphs' member idx
            arrays (e.g. eval-split suspicious subgraphs).
        n_nodes: total cluster count (needed for the baseline).

    Returns:
        A :class:`RecoveryResult`.

    Raises:
        ValueError: when ``n_nodes`` is not positive.
    """
    if n_nodes <= 0:
        raise ValueError(f"n_nodes must be positive, got {n_nodes}")

    discovered = _member_arrays(discovered_member_sets, n_nodes)
    targets = _member_arrays(target_member_sets, n_nodes)

    union = (
        np.unique(np.concatenate(discovered))
        if any(a.size for a in discovered)
        else np.zeros(0, dtype=np.int64)
    )
    d = int(union.size)
    hit = np.zeros(n_nodes, dtype=bool)
    hit[union] = True

    n_targets = len(targets)
    n_recovered = sum(1 for t in targets if t.size and bool(hit[t].any()))
    rate = n_recovered / n_targets if n_targets else 0.0

    baseline = (
        float(np.mean([_baseline_hit_prob(int(t.size), d, n_nodes) for t in targets]))
        if n_targets
        else 0.0
    )
    if baseline > 0.0:
        lift = rate / baseline
    else:
        lift = float("inf") if rate > 0.0 else 0.0

    return RecoveryResult(
        n_targets=n_targets,
        n_recovered=n_recovered,
        recovery_rate=rate,
        discovered_union_size=d,
        n_nodes=n_nodes,
        baseline_rate=baseline,
        lift=lift,
    )


def _allowed_ccids(split_csv: Path, split_name: str) -> set[str]:
    """ccIds assigned to ``split_name`` in ``split.csv`` (columns ``id,label,split``)."""
    with open(split_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        if "split" not in fields or "id" not in fields:
            raise ValueError(f"{split_csv} must have id,label,split columns")
        return {row["id"] for row in reader if row["split"] == split_name}


def load_member_sets(subgraphs_path: Path) -> list[npt.NDArray[np.int64]]:
    """All ``member_idx`` arrays from a subgraphs parquet (``ccId,ccLabel,...`` schema)."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_path, columns=["member_idx"])
    return [np.asarray(m, dtype=np.int64) for m in table.column("member_idx").to_pylist()]


def suspicious_targets(
    subgraphs_path: Path, split_csv: Path, split_name: str
) -> list[npt.NDArray[np.int64]]:
    """Member sets of the SUSPICIOUS subgraphs assigned to ``split_name``.

    The held-out targets for the recovery proxy: the eval-split suspicious subgraphs
    the cluster scorer never trained on.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_path, columns=["ccId", "ccLabel", "member_idx"])
    cc_ids = [str(v) for v in table.column("ccId").to_pylist()]
    labels = table.column("ccLabel").to_pylist()
    members = table.column("member_idx").to_pylist()
    allowed = _allowed_ccids(split_csv, split_name)
    return [
        np.asarray(m, dtype=np.int64)
        for cc, label, m in zip(cc_ids, labels, members, strict=True)
        if label == schema.LABEL_SUSPICIOUS and cc in allowed
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: report recovery of the eval-split suspicious subgraphs vs random baseline."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Held-out-recovery proxy: recovery of eval-split suspicious "
                    "subgraphs by discovered candidates vs random baseline.",
    )
    p.add_argument("--discovered", required=True, type=Path,
                   help="discovered_subgraphs.parquet (candidate member sets)")
    p.add_argument("--subgraphs", required=True, type=Path,
                   help="subgraphs.parquet (labeled subgraphs)")
    p.add_argument("--split-csv", required=True, type=Path, help="split.csv (id,label,split)")
    p.add_argument("--eval-split", default="test",
                   help="split whose suspicious subgraphs are the recovery targets")
    p.add_argument("--n-nodes", required=True, type=int, help="total cluster count N")
    args = p.parse_args(argv)

    discovered = load_member_sets(args.discovered)
    targets = suspicious_targets(args.subgraphs, args.split_csv, args.eval_split)
    result = recovery_rate(discovered, targets, args.n_nodes)

    print(
        f"[recovery] targets({args.eval_split},suspicious)={result.n_targets:,} "
        f"recovered={result.n_recovered:,} rate={result.recovery_rate:.4f} "
        f"baseline={result.baseline_rate:.4f} lift={result.lift:.2f}x "
        f"(discovered_union={result.discovered_union_size:,}/{result.n_nodes:,})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
