"""Integration test for Stage 0 ingest. Requires duckdb + numpy + pyarrow.

Runs with pytest, or standalone: ``python tests/test_ingest.py``.
Builds tiny synthetic versions of the five Elliptic2 CSVs, runs the full ingest,
and verifies the remapping bijection, feature alignment, edge remapping, subgraph
membership, and integrity accounting.
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

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.data import schema  # noqa: E402
from ellip2.data.ingest import IngestConfig, IngestError, ingest  # noqa: E402

# Synthetic dimensions.
N = 50          # background nodes
N_FEAT = schema.N_NODE_FEATURES   # 43
N_EFEAT = schema.N_EDGE_FEATURES  # 95


def _node_id(i: int) -> str:
    return f"n{i:04d}"


def _feat_value(node_i: int, j: int) -> float:
    """Deterministic, invertible: lets us assert feature rows land at the right idx."""
    return float(node_i * 100 + j)


def _write_raw(raw: Path, *, break_edge: bool = False) -> dict:
    raw.mkdir(parents=True, exist_ok=True)

    # background_nodes.csv : clId + feat_0..feat_42
    with (raw / schema.F_BACKGROUND_NODES).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clId"] + [f"feat_{j}" for j in range(N_FEAT)])
        for i in range(N):
            w.writerow([_node_id(i)] + [_feat_value(i, j) for j in range(N_FEAT)])

    # background_edges.csv : clId1,clId2 + 95 edge feats. Ring + a few chords.
    edges = [(i, (i + 1) % N) for i in range(N)] + [(0, 10), (5, 25), (10, 40)]
    if break_edge:
        edges.append((0, 99999))  # endpoint absent from background_nodes
    with (raw / schema.F_BACKGROUND_EDGES).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clId1", "clId2"] + [f"ef_{j}" for j in range(N_EFEAT)])
        for a, b in edges:
            aid = _node_id(a) if a < N else f"n{a:04d}"
            bid = _node_id(b) if b < N else f"n{b:04d}"
            w.writerow([aid, bid] + [0.0] * N_EFEAT)

    # connected_components.csv : ccId + ccLabel
    ccs = [("S0", "suspicious"), ("S1", "suspicious"),
           ("L0", "licit"), ("L1", "licit"), ("L2", "licit")]
    with (raw / schema.F_CONNECTED_COMPONENTS).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ccId", "ccLabel"])
        w.writerows(ccs)

    # nodes.csv : node id -> ccId membership (a subset of background nodes)
    membership = {
        "S0": [1, 2, 3],
        "S1": [10, 11],
        "L0": [20, 21, 22, 23],
        "L1": [30, 31],
        "L2": [40, 41, 42],
    }
    with (raw / schema.F_NODES).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clId", "ccId"])
        for cc, members in membership.items():
            for m in members:
                w.writerow([_node_id(m), cc])

    # edges.csv : labeled intra-subgraph edges (not consumed by Stage 0 artifacts,
    # but part of the shipped schema; present so require_all() passes)
    with (raw / schema.F_EDGES).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["clId1", "clId2"])
        w.writerow([_node_id(1), _node_id(2)])

    return {"edges": edges, "membership": membership, "ccs": dict(ccs)}


def _cfg(raw: Path, out: Path, **kw) -> IngestConfig:
    defaults = dict(
        raw_dir=raw, out_dir=out,
        expected_nodes=None, expected_edges=None, expected_subgraphs=None,
    )
    defaults.update(kw)
    return IngestConfig(**defaults)


def _load_id_map(out: Path) -> dict[str, int]:
    t = pq.read_table(out / "id_map.parquet").to_pydict()
    return {orig: int(idx) for orig, idx in zip(t["orig_id"], t["idx"], strict=True)}


# --------------------------------------------------------------------------- #


def test_ingest_builds_consistent_artifacts():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        truth = _write_raw(raw)
        res = ingest(_cfg(raw, out))
        m = res.manifest

        # id_map is a bijection over [0, N).
        idmap = _load_id_map(out)
        assert len(idmap) == N
        assert sorted(idmap.values()) == list(range(N))
        assert m["n_nodes"] == N

        # node_features.npy: row at idx(node i) holds feat_value(i, j).
        feats = np.load(out / "node_features.npy")
        assert feats.shape == (N, N_FEAT)
        for i in range(N):
            idx = idmap[_node_id(i)]
            assert feats[idx, 0] == _feat_value(i, 0)
            assert feats[idx, 7] == _feat_value(i, 7)
            assert feats[idx, N_FEAT - 1] == _feat_value(i, N_FEAT - 1)

        # edge_index.npy: remapped, valid, and contains a known remapped edge.
        ei = np.load(out / "edge_index.npy")
        assert ei.shape == (2, len(truth["edges"]))
        assert ei.dtype == np.int32
        assert ei.min() >= 0 and ei.max() < N
        edge_set = {(int(ei[0, k]), int(ei[1, k])) for k in range(ei.shape[1])}
        assert (idmap[_node_id(5)], idmap[_node_id(25)]) in edge_set
        assert m["edge_index"]["dangling"] == 0

        # subgraphs.parquet: membership remapped to idx, labels intact.
        sub = pq.read_table(out / "subgraphs.parquet").to_pydict()
        by_cc = {cc: (lab, set(mem)) for cc, lab, mem in
                 zip(sub["ccId"], sub["ccLabel"], sub["member_idx"], strict=True)}
        assert by_cc["S0"][0] == "suspicious"
        assert by_cc["S0"][1] == {idmap[_node_id(i)] for i in truth["membership"]["S0"]}
        assert by_cc["L0"][1] == {idmap[_node_id(i)] for i in truth["membership"]["L0"]}
        assert m["subgraphs"]["n_subgraphs"] == 5
        assert m["subgraphs"]["n_suspicious"] == 2
        assert m["subgraphs"]["orphan_nodes"] == 0
        assert m["subgraphs"]["orphan_cc"] == 0

        # manifest persisted and self-consistent.
        on_disk = json.loads((out / "ingest_manifest.json").read_text())
        assert on_disk["n_nodes"] == N
        assert not on_disk["warnings"]


def test_float16_dtype_option():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        _write_raw(raw)
        ingest(_cfg(raw, out, feature_dtype="float16"))
        feats = np.load(out / "node_features.npy")
        assert feats.dtype == np.float16
        assert feats[ _load_id_map(out)[_node_id(3)], 5] == _feat_value(3, 5)


def test_dangling_edge_warns_but_completes():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        truth = _write_raw(raw, break_edge=True)
        res = ingest(_cfg(raw, out))  # non-strict: warns, drops the bad edge
        ei = np.load(out / "edge_index.npy")
        assert ei.shape[1] == len(truth["edges"]) - 1  # broken edge dropped
        assert res.manifest["edge_index"]["dangling"] == 1
        assert any("absent from background_nodes" in w
                   for w in res.manifest["warnings"])


def test_dangling_edge_strict_raises():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        _write_raw(raw, break_edge=True)
        try:
            ingest(_cfg(raw, out, strict=True))
        except IngestError as e:
            assert "absent from background_nodes" in str(e)
        else:
            raise AssertionError("expected IngestError under strict mode")


def test_missing_file_rejected():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "raw", Path(d) / "out"
        _write_raw(raw)
        (raw / schema.F_NODES).unlink()
        try:
            ingest(_cfg(raw, out))
        except FileNotFoundError as e:
            assert schema.F_NODES in str(e)
        else:
            raise AssertionError("expected FileNotFoundError for missing CSV")


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
