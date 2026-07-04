"""Background discovery — surface NEW suspicious subgraphs in the unlabeled graph.

Everything scored today is one of the 121,810 *labeled* subgraphs; the real goal
is to discover novel laundering structures among the ~48.8M unlabeled background
clusters (plan ``yes-lets-do-it-glittery-coral.md``). This module holds the
discovery machinery; T-026 seeds it with the two small helpers the orchestrator
(T-028 ``discover_background``) needs before it can carve and score candidates:

* :func:`typology_signal_from_features` — Gate 3's structural signal: the
  per-cluster ``source_sink_axis`` column (T-006 ``path_role``) exported as an
  idx-aligned ``(N,)`` array (and written to ``typology_signal.npy`` by
  :func:`main`). Mirrors :func:`ellip2.exit_paths.recover.endpoints_from_features`.
* :func:`known_member_idx` — the exclusion set: the sorted-unique union of every
  labeled subgraph's member cluster idxs from ``subgraphs.parquet``, so Gate 1
  can drop already-known clusters and rank only genuinely NEW structures.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt


def typology_signal_from_features(
    cluster_features_parquet: Path,
    *,
    score_col: str = "source_sink_axis",
) -> npt.NDArray[np.float64]:
    """Idx-aligned ``(N,)`` typology signal — the ``source_sink_axis`` column of features.

    Rows are sorted by the ``idx`` key so row ``i`` is cluster ``i`` in ``[0, N)`` and
    the returned array can be indexed positionally. Raises if the score column is
    missing or ``idx`` is not a contiguous ``0..N-1`` range.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(cluster_features_parquet, columns=["idx", score_col])
    idx = table.column("idx").to_numpy(zero_copy_only=False).astype(np.int64)
    signal = table.column(score_col).to_numpy(zero_copy_only=False).astype(np.float64)
    order = np.argsort(idx, kind="stable")
    idx, signal = idx[order], signal[order]
    if not np.array_equal(idx, np.arange(idx.size)):
        raise ValueError(
            f"feature 'idx' in {cluster_features_parquet} is not a contiguous 0..N-1 range"
        )
    return signal


def known_member_idx(
    subgraphs_path: Path,
    *,
    n_nodes: int | None = None,
) -> npt.NDArray[np.int64]:
    """Sorted-unique union of every labeled subgraph's member cluster idxs.

    The exclusion set for background discovery: Gate 1 drops these already-known
    clusters so only NEW structures surface. Optionally bounded to ``[0, n_nodes)``.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    table = pq.read_table(subgraphs_path, columns=["member_idx"])
    parts = [
        np.asarray(members, dtype=np.int64)
        for members in table.column("member_idx").to_pylist()
        if members
    ]
    if not parts:
        return np.zeros(0, dtype=np.int64)
    idx = np.unique(np.concatenate(parts))
    if n_nodes is not None:
        idx = idx[(idx >= 0) & (idx < n_nodes)]
    return idx


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: write the ``(N,)`` typology signal (``source_sink_axis``) to a ``.npy``."""
    import argparse  # noqa: PLC0415

    p = argparse.ArgumentParser(
        description="Background discovery: export the source_sink_axis typology signal (N,).",
    )
    p.add_argument("--features", required=True, type=Path, help="cluster_features.parquet")
    p.add_argument("--out", required=True, type=Path, help="(N,) typology_signal.npy")
    p.add_argument("--score-col", default="source_sink_axis")
    args = p.parse_args(argv)

    signal = typology_signal_from_features(args.features, score_col=args.score_col)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, signal)
    print(f"[typology] wrote {signal.size:,} typology signals -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
