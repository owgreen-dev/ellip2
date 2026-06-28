"""CLI for the persisted subgraph-level split generator (see splits.py).

Exposed as the ``ellip2-make-split`` console script and re-used by
``scripts/make_split.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ellip2.eval.splits import VALID_METHODS, SplitConfig, generate_split


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ellip2-make-split",
        description="Generate a persisted subgraph-level train/val/test split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", required=True, type=Path,
                   help="path to connected_components.csv (id + ccLabel)")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="output root; files land under <out-dir>/<method>/")
    p.add_argument("--method", default="stratified_random", choices=VALID_METHODS)
    p.add_argument("--seed", type=int, default=42,
                   help="ignored for method=round_robin")
    p.add_argument("--ratios", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--id-column", default="0",
                   help="id column index (int) or header name; default first column")
    p.add_argument("--label-column", default="ccLabel")
    p.add_argument("--positive-label", default="suspicious")
    p.add_argument("--negative-label", default="licit")
    p.add_argument("--strict-counts", action="store_true",
                   help="hard-fail (not warn) on count/label mismatches")
    p.add_argument("--no-verify", action="store_true",
                   help="skip invariant checks (not recommended)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    id_column: int | str = (
        int(args.id_column) if str(args.id_column).isdigit() else args.id_column
    )
    cfg = SplitConfig(
        input_csv=args.input,
        out_dir=args.out_dir,
        method=args.method,
        seed=args.seed,
        ratios=tuple(args.ratios),
        id_column=id_column,
        label_column=args.label_column,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        strict_counts=args.strict_counts,
    )
    result = generate_split(cfg, verify=not args.no_verify)
    m = result.manifest
    print(f"Wrote split to {result.out_dir}")
    print(f"  method={m['method']} seed={m['seed']} "
          f"total={m['n_total']} base_rate={m['base_rate']:.4%}")
    for name in ("train", "val", "test"):
        s = m["per_split"][name]
        print(f"  {name:5s}: {s['total']:>7d}  "
              f"pos={s['positive']:>5d} ({s['positive_rate']:.4%})")
    print(f"  content_hash={m['content_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
