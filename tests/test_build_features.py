"""Full Stage 1 assembly test (T-007).

Builds the tiny synthetic Stage 0 artifacts via the ingest fixtures, writes a
matching split.csv, runs ``build_cluster_features``, and asserts the combined
schema, the one-row-per-cluster-idx contract, and that no required column carries
a NaN/inf. Runs with pytest, or standalone: ``python tests/test_build_features.py``.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.data.ingest import ingest  # noqa: E402
from ellip2.features import (  # noqa: E402
    degree,
    flow_concentration,
    neighborhood,
    path_role,
    temporal,
)
from ellip2.features.build import (  # noqa: E402
    REQUIRED_COLUMNS,
    FeatureBuildConfig,
    build_cluster_features,
)
from ellip2.features.edge_aggs import aggregate_columns  # noqa: E402

# Reuse the Stage 0 synthetic-CSV fixtures so artifacts come from the real ingest.
from test_ingest import N, _cfg, _write_raw  # noqa: E402

# ccLabel per synthetic subgraph (matches _write_raw); two are placed in TEST.
_LABELS = {"S0": "suspicious", "S1": "suspicious",
           "L0": "licit", "L1": "licit", "L2": "licit"}
_TEST_SUBGRAPHS = {"S1", "L2"}


def _write_split(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for cc, lab in _LABELS.items():
            split = "test" if cc in _TEST_SUBGRAPHS else "train"
            w.writerow([cc, lab, split])


def _setup(d: Path) -> FeatureBuildConfig:
    raw, out = d / "raw", d / "out"
    _write_raw(raw)
    ingest(_cfg(raw, out))
    split = out / "splits" / "split.csv"
    _write_split(split)
    return FeatureBuildConfig(
        artifacts_dir=out, raw_dir=raw, split_csv=split,
        edge_agg_indices=(0, 1),
    )


def test_assembly_schema_and_row_count():
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)

        assert res.out_path.is_file()
        assert res.n_nodes == N

        table = pq.read_table(res.out_path)
        # One row per cluster idx, in idx order.
        assert table.num_rows == N
        assert table.column("idx").to_pylist() == list(range(N))

        names = set(table.column_names)
        # The key column plus every fixed-name feature group.
        assert "idx" in names
        for col in REQUIRED_COLUMNS:
            assert col in names, f"missing required column {col!r}"
        # Edge aggregates for the chosen indices (sum/mean/max/std, in + out).
        for col in aggregate_columns(["ef_0", "ef_1"]):
            assert col in names, f"missing edge-agg column {col!r}"
        # Reported column list matches the parquet (minus the idx key).
        assert set(res.columns) == names - {"idx"}


def test_no_nan_in_required_columns():
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)
        table = pq.read_table(res.out_path)
        for col in res.columns:
            arr = np.asarray(table.column(col).to_numpy(zero_copy_only=False),
                             dtype=np.float64)
            assert np.all(np.isfinite(arr)), f"non-finite values in {col!r}"


def test_degree_column_matches_standalone():
    """The assembled degree columns equal a direct compute over edge_index.npy."""
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)
        table = pq.read_table(res.out_path)

        ei = np.load(Path(cfg.artifacts_dir) / "edge_index.npy")
        deg = degree.compute_degree_features(ei, N)
        for col in degree.COLUMNS:
            got = np.asarray(table.column(col).to_numpy(zero_copy_only=False))
            np.testing.assert_array_equal(got, deg[col])


def test_neighborhood_leakage_mask_applied():
    """Subgraphs in the TEST split must not contribute observable labels.

    Re-run the neighborhood builder with the SAME loaded labels but with the
    test mask turned OFF; at least one neighborhood fraction must change, proving
    the persisted split was actually consumed (SIGN-103).
    """
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)
        table = pq.read_table(res.out_path)

        ei = np.load(Path(cfg.artifacts_dir) / "edge_index.npy")
        labels = neighborhood.load_subgraph_labels(
            Path(cfg.artifacts_dir) / "subgraphs.parquet", cfg.split_csv, N
        )
        assert labels.subgraph_in_test.any(), "fixture should mark some test subgraphs"

        masked_off = neighborhood.compute_neighborhood_features(
            ei, labels.node_subgraph, labels.subgraph_label,
            np.zeros_like(labels.subgraph_in_test),  # pretend nothing is in test
            N,
        )
        changed = False
        for col in neighborhood.COLUMNS:
            built = np.asarray(table.column(col).to_numpy(zero_copy_only=False))
            if not np.array_equal(built, masked_off[col]):
                changed = True
                break
        assert changed, "test-split mask had no effect — leakage masking inactive"


def test_path_role_in_expected_range():
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)
        table = pq.read_table(res.out_path)

        for col in ("endpoint_score", "source_score"):
            v = np.asarray(table.column(col).to_numpy(zero_copy_only=False))
            assert v.min() >= 0.0 and v.max() <= 1.0
        axis = np.asarray(table.column("source_sink_axis").to_numpy(zero_copy_only=False))
        assert axis.min() >= -1.0 and axis.max() <= 1.0
        assert set(path_role.COLUMNS).issubset(set(table.column_names))


def test_temporal_columns_present():
    with tempfile.TemporaryDirectory() as d:
        cfg = _setup(Path(d))
        res = build_cluster_features(cfg)
        table = pq.read_table(res.out_path)
        for col in (*temporal.COLUMNS, *flow_concentration.COLUMNS):
            assert col in table.column_names


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
            import traceback
            print(f"FAIL {t.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
