"""Tests for the persisted subgraph-level split generator.

Runs with pytest, or standalone: ``python tests/test_splits.py``.
No third-party dependencies required.
"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ellip2.eval.splits import (  # noqa: E402
    SPLIT_TO_MASK,
    SplitConfig,
    assign_splits,
    generate_split,
    load_components,
    verify_split,
)


def _write_components(path: Path, n: int = 1000, n_pos: int = 30,
                      header=("ccId", "ccLabel")) -> None:
    """Synthetic connected_components.csv: n rows, n_pos suspicious, rest licit.
    Positives are interleaved so file order is not class-sorted."""
    rows = []
    pos_every = max(1, n // n_pos)
    pos_written = 0
    for i in range(n):
        is_pos = (i % pos_every == 0) and pos_written < n_pos
        if is_pos:
            pos_written += 1
        rows.append((f"cc{i:06d}", "suspicious" if is_pos else "licit"))
    # top up positives if interleaving under-filled
    j = 0
    while pos_written < n_pos and j < n:
        if rows[j][1] == "licit":
            rows[j] = (rows[j][0], "suspicious")
            pos_written += 1
        j += 1
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        w.writerows(rows)


def _cfg(tmp: Path, **kw) -> SplitConfig:
    defaults = dict(
        input_csv=tmp / "connected_components.csv",
        out_dir=tmp / "splits",
        expected_total=None,  # synthetic data is small; don't warn
    )
    defaults.update(kw)
    return SplitConfig(**defaults)


# --------------------------------------------------------------------------- #


def test_round_robin_matches_glass_reference():
    """round_robin must equal the literal preprocess_glass.py modulo-10 logic."""
    n = 1000
    got = assign_splits([f"x{i}" for i in range(n)], ["licit"] * n,
                        SplitConfig(input_csv=Path("x"), out_dir=Path("y"),
                                    method="round_robin", expected_total=None))
    # Reference: counter = i+1; <=7 train, ==8 val, else test.
    ref = []
    for i in range(n):
        m = (i + 1) % 10
        ref.append("train" if m <= 7 else "val" if m == 8 else "test")
    assert got == ref
    assert got.count("train") == 800
    assert got.count("val") == 100
    assert got.count("test") == 100


def test_stratified_preserves_class_balance_and_covers():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_components(tmp / "connected_components.csv", n=1000, n_pos=30)
        cfg = _cfg(tmp, method="stratified_random", seed=42)
        res = generate_split(cfg)
        verify_split(res, cfg)

        # Coverage + disjoint.
        assert len(res.ids) == 1000
        seen = set()
        for name in ("train", "val", "test"):
            ids = set(res.id_list(name))
            assert not (ids & seen), f"{name} overlaps a prior split"
            seen |= ids
        assert seen == set(res.ids)

        # Each split ~80/10/10 of the positives (stratified).
        m = res.manifest["per_split"]
        assert m["train"]["positive"] == 24
        assert m["val"]["positive"] == 3
        assert m["test"]["positive"] == 3
        assert m["train"]["positive"] + m["val"]["positive"] + m["test"]["positive"] == 30


def test_determinism_same_seed_and_difference_across_seeds():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_components(tmp / "connected_components.csv", n=1000, n_pos=30)
        ids, labels = load_components(_cfg(tmp))

        a = assign_splits(ids, labels, _cfg(tmp, method="stratified_random", seed=42))
        b = assign_splits(ids, labels, _cfg(tmp, method="stratified_random", seed=42))
        c = assign_splits(ids, labels, _cfg(tmp, method="stratified_random", seed=7))
        assert a == b, "same seed must reproduce identical assignment"
        assert a != c, "different seed should change assignment"


def test_outputs_written_and_consistent():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_components(tmp / "connected_components.csv", n=1000, n_pos=30)
        cfg = _cfg(tmp, method="stratified_random", seed=42)
        res = generate_split(cfg)
        out = res.out_dir

        for fname in ("split.csv", "train.txt", "val.txt", "test.txt",
                      "mask.csv", "manifest.json"):
            assert (out / fname).is_file(), f"missing {fname}"

        # split.csv rows align with manifest totals.
        with (out / "split.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1000
        counts = {"train": 0, "val": 0, "test": 0}
        for r in rows:
            counts[r["split"]] += 1
        man = json.loads((out / "manifest.json").read_text())
        for name in counts:
            assert counts[name] == man["per_split"][name]["total"]

        # mask.csv encodes the same assignment via the GLASS convention.
        with (out / "mask.csv").open() as fh:
            mask_rows = {r["id"]: int(r["mask"]) for r in csv.DictReader(fh)}
        for r in rows:
            assert mask_rows[r["id"]] == SPLIT_TO_MASK[r["split"]]

        # per-split id lists match split.csv.
        for name in ("train", "val", "test"):
            listed = (out / f"{name}.txt").read_text().split()
            assert sorted(listed) == sorted(
                r["id"] for r in rows if r["split"] == name
            )


def test_content_hash_stable_across_runs():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_components(tmp / "connected_components.csv", n=1000, n_pos=30)
        h1 = generate_split(_cfg(tmp, method="stratified_random", seed=42)).manifest["content_hash"]
        h2 = generate_split(_cfg(tmp, method="stratified_random", seed=42)).manifest["content_hash"]
        assert h1 == h2


def test_duplicate_ids_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "connected_components.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ccId", "ccLabel"])
            w.writerow(["dup", "licit"])
            w.writerow(["dup", "suspicious"])
        try:
            load_components(_cfg(tmp))
        except ValueError as e:
            assert "duplicate" in str(e).lower()
        else:
            raise AssertionError("expected ValueError on duplicate ccIds")


def test_missing_label_column_rejected():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "connected_components.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ccId", "wrong_name"])
            w.writerow(["cc0", "licit"])
        try:
            load_components(_cfg(tmp))
        except ValueError as e:
            assert "ccLabel" in str(e)
        else:
            raise AssertionError("expected ValueError on missing label column")


def test_named_columns_resolved():
    """Label column found by name even when it is not the second column."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        p = tmp / "connected_components.csv"
        with p.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ccId", "extra", "ccLabel"])
            for i in range(100):
                w.writerow([f"cc{i}", "junk", "suspicious" if i < 10 else "licit"])
        ids, labels = load_components(_cfg(tmp))
        assert len(ids) == 100
        assert labels.count("suspicious") == 10


# --------------------------------------------------------------------------- #


def _run_standalone() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
