"""Persisted subgraph-level train/val/test split generator for Elliptic2.

Why this module exists (plan.md, Resolved design decision #1):

The unit of analysis is the labeled subgraph (connected component), so the split
must be defined over ``connected_components.csv`` rows — ONE assignment per ccId —
and persisted to explicit, content-hashed files that every model reuses. We do NOT
rely on the implicit order produced by GLASS's ``preprocess_glass.py``, whose split
is a deterministic modulo-10 round-robin (``counter % 10 <= 7 -> train, == 8 -> val,
else test``) despite the README calling it "fixed given the random seeds". The
Elliptic2 paper and RevTrack instead describe a *random* 80:10:10. Neither paper's
exact random seed/order is recoverable, so reproducing published splits means
matching each repo's *mechanism*; for our own work we fix one split, persist it, and
report it.

Three mechanisms are supported:

* ``round_robin``       — byte-for-byte reproduction of preprocess_glass.py
                          (file order, unstratified, no seed).
* ``random``            — seeded global shuffle then 80/10/10 (matches the paper's
                          stated mechanism, unstratified).
* ``stratified_random`` — seeded per-class shuffle then 80/10/10 (DEFAULT). At a
                          2.27% suspicious base rate this guarantees every split
                          contains positives and keeps PR-AUC estimates stable.

Pure stdlib by design: the file is only ~121,810 rows, so DuckDB/Polars would be
overkill. numpy (``.npy`` integer mask) and pyarrow (parquet) are used only if
already importable; CSV + JSON + per-split id lists are always written.

Outputs (under ``<out_dir>/<method>/``):
    split.csv     — id,label,split  (split in {train,val,test}); the canonical table
    train.txt     — newline-separated ids (likewise val.txt, test.txt)
    val.txt
    test.txt
    mask.csv      — id,mask  (mask in {0,1,2} == train/val/test, GLASS convention)
    mask.npy      — int8 mask aligned to file order  (only if numpy is importable)
    split.parquet — id,label,split                   (only if pyarrow is importable)
    manifest.json — method, seed, ratios, counts (per split and per class), input
                    sha256, and a content_hash uniquely identifying the assignment
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Canonical Elliptic2 figures (paper Table 1, arXiv:2404.19109). RevTrack reports
# 2,718 suspicious / 119,092 licit — a minor version difference. Counts are
# warn-only by default; set strict_counts=True to hard-fail on mismatch.
EXPECTED_TOTAL = 121_810
EXPECTED_SUSPICIOUS_PAPER = 2_763

VALID_METHODS = ("round_robin", "random", "stratified_random")
# GLASS integer-mask convention: train=0, val=1, test=2.
SPLIT_NAMES = ("train", "val", "test")
SPLIT_TO_MASK = {"train": 0, "val": 1, "test": 2}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for a single, reproducible split run."""

    input_csv: Path
    out_dir: Path
    method: str = "stratified_random"
    seed: int = 42
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    # First column is the ccId by Elliptic2 convention; the label column is
    # literally named "ccLabel". Both are overridable.
    id_column: int | str = 0
    label_column: str = "ccLabel"
    positive_label: str = "suspicious"
    negative_label: str = "licit"
    expected_total: int | None = EXPECTED_TOTAL
    strict_counts: bool = False

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {self.method!r}"
            )
        if len(self.ratios) != 3:
            raise ValueError(f"ratios must be a 3-tuple, got {self.ratios!r}")
        if abs(sum(self.ratios) - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1.0, got {self.ratios!r}")
        if any(r < 0 for r in self.ratios):
            raise ValueError(f"ratios must be non-negative, got {self.ratios!r}")


@dataclass
class SplitResult:
    """In-memory result of a split, plus the manifest written to disk."""

    ids: list[str]
    labels: list[str]
    splits: list[str]  # one of SPLIT_NAMES, aligned to `ids` (file order)
    manifest: dict = field(default_factory=dict)
    out_dir: Path | None = None

    def indices(self, split: str) -> list[int]:
        """Row indices (file order) assigned to `split`."""
        return [i for i, s in enumerate(self.splits) if s == split]

    def id_list(self, split: str) -> list[str]:
        return [self.ids[i] for i in self.indices(split)]


def _warn(msg: str) -> None:
    print(f"[splits] WARNING: {msg}", file=sys.stderr)


def _resolve_columns(header: list[str], cfg: SplitConfig) -> tuple[int, int]:
    """Return (id_col_index, label_col_index) from the CSV header."""
    if isinstance(cfg.id_column, int):
        id_idx = cfg.id_column
        if not 0 <= id_idx < len(header):
            raise ValueError(
                f"id_column index {id_idx} out of range for header {header!r}"
            )
    else:
        if cfg.id_column not in header:
            raise ValueError(
                f"id_column {cfg.id_column!r} not found in header {header!r}"
            )
        id_idx = header.index(cfg.id_column)

    if cfg.label_column not in header:
        raise ValueError(
            f"label_column {cfg.label_column!r} not found in header {header!r}. "
            "Elliptic2 connected_components.csv should contain a 'ccLabel' column."
        )
    return id_idx, header.index(cfg.label_column)


def load_components(cfg: SplitConfig) -> tuple[list[str], list[str]]:
    """Load (ids, labels) from connected_components.csv, preserving file order.

    File order is significant: ``round_robin`` reproduces preprocess_glass.py only
    if rows are consumed in their on-disk order.
    """
    path = Path(cfg.input_csv)
    if not path.is_file():
        raise FileNotFoundError(f"connected_components csv not found: {path}")

    ids: list[str] = []
    labels: list[str] = []
    with path.open("r", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"{path} is empty") from exc
        id_idx, label_idx = _resolve_columns(header, cfg)
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) <= max(id_idx, label_idx):
                raise ValueError(
                    f"{path}:{lineno} has {len(row)} columns; "
                    f"need indices {id_idx} and {label_idx}"
                )
            ids.append(row[id_idx].strip())
            labels.append(row[label_idx].strip())

    if len(ids) != len(set(ids)):
        dupes = _first_duplicates(ids)
        raise ValueError(f"duplicate ccIds in {path} (e.g. {dupes})")

    _validate_counts(labels, cfg)
    return ids, labels


def _first_duplicates(ids: list[str], limit: int = 5) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for i in ids:
        if i in seen and i not in out:
            out.append(i)
            if len(out) >= limit:
                break
        seen.add(i)
    return out


def _validate_counts(labels: list[str], cfg: SplitConfig) -> None:
    total = len(labels)
    n_pos = sum(1 for v in labels if v == cfg.positive_label)
    n_neg = sum(1 for v in labels if v == cfg.negative_label)
    n_other = total - n_pos - n_neg

    if n_other:
        unexpected = sorted(
            {v for v in labels if v not in (cfg.positive_label, cfg.negative_label)}
        )
        msg = (
            f"{n_other} rows have labels outside "
            f"{{{cfg.positive_label!r}, {cfg.negative_label!r}}}: {unexpected[:10]}"
        )
        if cfg.strict_counts:
            raise ValueError(msg)
        _warn(msg)

    if n_pos == 0:
        raise ValueError(
            f"no positive ({cfg.positive_label!r}) rows found — cannot build a "
            "usable split or compute PR-AUC"
        )

    if cfg.expected_total is not None and total != cfg.expected_total:
        msg = (
            f"row count {total} != expected {cfg.expected_total} "
            "(Elliptic2 Table 1); confirm you have the full file"
        )
        if cfg.strict_counts:
            raise ValueError(msg)
        _warn(msg)

    if n_pos != EXPECTED_SUSPICIOUS_PAPER:
        _warn(
            f"suspicious count {n_pos} != paper's {EXPECTED_SUSPICIOUS_PAPER} "
            f"(RevTrack reports 2,718); base rate {n_pos / total:.4%}. "
            "Use the count shipped with your downloaded copy."
        )


def _assign_round_robin(n: int) -> list[str]:
    """preprocess_glass.py logic: counter starts at 1 and increments per subgraph;
    ``counter % 10 <= 7 -> train, == 8 -> val, else test`` (so remainder 0 and 1..7
    are train = 8/10, 8 -> val, 9 -> test)."""
    out: list[str] = []
    for i in range(n):
        counter = i + 1
        m = counter % 10
        if m <= 7:
            out.append("train")
        elif m == 8:
            out.append("val")
        else:
            out.append("test")
    return out


def _partition(count: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split `count` items into (train, val, test) sizes summing exactly to count.
    Floors val/test and gives the remainder to train (the largest split)."""
    n_val = int(count * ratios[1])
    n_test = int(count * ratios[2])
    n_train = count - n_val - n_test
    return n_train, n_val, n_test


def _assign_from_order(order: list[int], n: int,
                       ratios: tuple[float, float, float]) -> list[str]:
    """Assign splits given a (possibly shuffled) `order` of the n indices."""
    n_train, n_val, _ = _partition(len(order), ratios)
    out = ["test"] * n
    for rank, idx in enumerate(order):
        if rank < n_train:
            out[idx] = "train"
        elif rank < n_train + n_val:
            out[idx] = "val"
        else:
            out[idx] = "test"
    return out


def assign_splits(ids: list[str], labels: list[str], cfg: SplitConfig) -> list[str]:
    """Return a split label ('train'|'val'|'test') per row, aligned to file order."""
    n = len(ids)
    if cfg.method == "round_robin":
        return _assign_round_robin(n)

    rng = random.Random(cfg.seed)
    if cfg.method == "random":
        order = list(range(n))
        rng.shuffle(order)
        return _assign_from_order(order, n, cfg.ratios)

    # stratified_random: shuffle within each class, partition each class 80/10/10,
    # then merge. Guarantees positives land in every split.
    out = ["test"] * n
    by_class: dict[str, list[int]] = {}
    for idx, lab in enumerate(labels):
        by_class.setdefault(lab, []).append(idx)
    for lab in sorted(by_class):  # deterministic class iteration order
        members = by_class[lab]
        rng.shuffle(members)
        n_train, n_val, _ = _partition(len(members), cfg.ratios)
        for rank, idx in enumerate(members):
            if rank < n_train:
                out[idx] = "train"
            elif rank < n_train + n_val:
                out[idx] = "val"
            else:
                out[idx] = "test"
    return out


def _content_hash(ids: list[str], splits: list[str]) -> str:
    """Stable hash of the assignment, independent of row order or output format."""
    h = hashlib.sha256()
    for ccid, split in sorted(zip(ids, splits, strict=True)):
        h.update(ccid.encode())
        h.update(b"\t")
        h.update(split.encode())
        h.update(b"\n")
    return h.hexdigest()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _counts(labels: list[str], splits: list[str], cfg: SplitConfig) -> dict:
    per_split: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        idxs = [i for i, s in enumerate(splits) if s == name]
        n_pos = sum(1 for i in idxs if labels[i] == cfg.positive_label)
        total = len(idxs)
        per_split[name] = {
            "total": total,
            "positive": n_pos,
            "negative": total - n_pos,
            "positive_rate": (n_pos / total) if total else 0.0,
        }
    return per_split


def verify_split(result: SplitResult, cfg: SplitConfig) -> None:
    """Assert the structural invariants every downstream model relies on."""
    ids, labels, splits = result.ids, result.labels, result.splits
    n = len(ids)
    assert len(labels) == n and len(splits) == n, "ragged arrays"

    # 1. Every row assigned to exactly one valid split (complete + disjoint coverage).
    assert all(s in SPLIT_NAMES for s in splits), "invalid split label present"
    grouped: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    for ccid, s in zip(ids, splits, strict=True):
        grouped[s].add(ccid)
    union = set().union(*grouped.values())
    assert union == set(ids), "split union does not cover all ids"
    assert sum(len(g) for g in grouped.values()) == len(set(ids)), "splits overlap"

    # 2. Each split holds at least one positive — required for PR-AUC / F1.
    counts = _counts(labels, splits, cfg)
    for name in SPLIT_NAMES:
        assert counts[name]["positive"] >= 1, (
            f"split {name!r} has no positives; use stratified_random or a larger set"
        )

    # 3. Proportions within tolerance of the requested ratios.
    tol = max(0.02, 5.0 / max(n, 1))  # looser for tiny inputs
    for name, ratio in zip(SPLIT_NAMES, cfg.ratios, strict=True):
        observed = counts[name]["total"] / n
        assert abs(observed - ratio) <= tol, (
            f"split {name!r} proportion {observed:.4f} off target {ratio} (tol {tol})"
        )


def _write_outputs(result: SplitResult, cfg: SplitConfig) -> Path:
    out_dir = Path(cfg.out_dir) / cfg.method
    out_dir.mkdir(parents=True, exist_ok=True)
    ids, labels, splits = result.ids, result.labels, result.splits

    # Canonical table.
    with (out_dir / "split.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        w.writerows(zip(ids, labels, splits, strict=True))

    # Per-split id lists.
    for name in SPLIT_NAMES:
        with (out_dir / f"{name}.txt").open("w") as fh:
            fh.write("\n".join(result.id_list(name)))
            fh.write("\n")

    # Integer mask (GLASS convention) as CSV always; .npy if numpy present.
    mask = [SPLIT_TO_MASK[s] for s in splits]
    with (out_dir / "mask.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "mask"])
        w.writerows(zip(ids, mask, strict=True))
    _maybe_write_npy(out_dir / "mask.npy", mask)
    _maybe_write_parquet(out_dir / "split.parquet", ids, labels, splits)

    # Manifest with full provenance.
    manifest = {
        "dataset": "elliptic2",
        "unit": "subgraph (connected_component)",
        "method": cfg.method,
        "seed": None if cfg.method == "round_robin" else cfg.seed,
        "ratios": list(cfg.ratios),
        "positive_label": cfg.positive_label,
        "negative_label": cfg.negative_label,
        "input_csv": str(Path(cfg.input_csv).resolve()),
        "input_sha256": _file_sha256(Path(cfg.input_csv)),
        "n_total": len(ids),
        "n_positive": sum(1 for v in labels if v == cfg.positive_label),
        "base_rate": (
            sum(1 for v in labels if v == cfg.positive_label) / len(ids)
            if ids else 0.0
        ),
        "per_split": _counts(labels, splits, cfg),
        "content_hash": _content_hash(ids, splits),
        "mask_convention": SPLIT_TO_MASK,
    }
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")

    result.manifest = manifest
    result.out_dir = out_dir
    return out_dir


def _maybe_write_npy(path: Path, mask: list[int]) -> None:
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        return
    np.save(path, np.asarray(mask, dtype=np.int8))


def _maybe_write_parquet(path: Path, ids, labels, splits) -> None:
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError:
        return
    table = pa.table({"id": ids, "label": labels, "split": splits})
    pq.write_table(table, path)


def generate_split(cfg: SplitConfig, *, verify: bool = True) -> SplitResult:
    """Build, verify, and persist a subgraph-level split. Returns the result."""
    ids, labels = load_components(cfg)
    splits = assign_splits(ids, labels, cfg)
    result = SplitResult(ids=ids, labels=labels, splits=splits)
    if verify:
        verify_split(result, cfg)
    _write_outputs(result, cfg)
    return result
