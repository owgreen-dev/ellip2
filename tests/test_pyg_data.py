"""Unit test for the homogeneous PyG ``Data`` builder (T-008).

Builds the tiny synthetic Stage 0 artifacts via the ingest fixtures, constructs
the homogeneous ``torch_geometric.data.Data`` from the memmaps, and asserts the
``x``/``edge_index`` shapes + dtypes and that subgraph membership round-trips
(member idxs -> node_subgraph inverse -> back). CPU-only, synthetic, no external
resources (SIGN-101). Runs under pytest, or standalone:
``python tests/test_pyg_data.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ellip2.data.ingest import ingest  # noqa: E402
from ellip2.graph.pyg_data import (  # noqa: E402
    SubgraphMembership,
    build_pyg_data,
    load_edge_index,
    load_node_features,
    load_subgraph_membership,
)

# Reuse the Stage 0 synthetic-CSV fixtures so artifacts come from the real ingest.
from test_ingest import N_FEAT, N, _cfg, _write_raw  # noqa: E402


def _make_artifacts(d: Path) -> Path:
    raw, out = d / "raw", d / "out"
    _write_raw(raw)
    ingest(_cfg(raw, out))
    return out


def test_data_shapes_and_dtypes():
    with tempfile.TemporaryDirectory() as d:
        out = _make_artifacts(Path(d))
        ei_disk = np.load(out / "edge_index.npy")
        data = build_pyg_data(out)

        # Single node type / single edge type homogeneous Data.
        assert data.x.shape == (N, N_FEAT)       # (N, 43)
        assert data.x.dtype == torch.float32
        assert data.num_nodes == N

        assert tuple(data.edge_index.shape) == (2, ei_disk.shape[1])  # (2, E)
        assert data.edge_index.dtype == torch.long
        # Values preserved through the int32 -> int64 cast.
        np.testing.assert_array_equal(
            data.edge_index.numpy(), ei_disk.astype(np.int64)
        )
        assert int(data.edge_index.min()) >= 0
        assert int(data.edge_index.max()) < N


def test_loaders_use_memmap():
    with tempfile.TemporaryDirectory() as d:
        out = _make_artifacts(Path(d))
        nf = load_node_features(out)           # default mmap_mode='r'
        ei = load_edge_index(out)
        assert isinstance(nf, np.memmap)
        assert isinstance(ei, np.memmap)
        assert nf.shape == (N, N_FEAT)
        assert ei.shape[0] == 2
        # Read-only mapping.
        assert nf.flags.writeable is False


def test_subgraph_membership_round_trips():
    with tempfile.TemporaryDirectory() as d:
        out = _make_artifacts(Path(d))
        mem = load_subgraph_membership(out / "subgraphs.parquet")

        assert isinstance(mem, SubgraphMembership)
        assert len(mem) == 5                    # S0,S1,L0,L1,L2 (see _write_raw)
        assert set(mem.ccids) == {"S0", "S1", "L0", "L1", "L2"}

        node_subgraph = mem.node_subgraph(N)
        assert node_subgraph.shape == (N,)

        # Every subgraph's member idxs map back to that subgraph index.
        for sid in range(len(mem)):
            idx = mem.member_idx(sid)
            assert np.all(node_subgraph[idx] == sid)

        # ccId accessor agrees with the positional accessor, and the inverse map
        # round-trips through it too.
        for cc in mem.ccids:
            idx = mem.members_of(cc)
            np.testing.assert_array_equal(idx, mem.member_idx(mem.index_of(cc)))
            assert np.all(node_subgraph[idx] == mem.index_of(cc))

        # Background nodes outside any labeled subgraph stay unlabeled.
        labeled = np.concatenate([mem.member_idx(s) for s in range(len(mem))])
        unlabeled = np.setdiff1d(np.arange(N), labeled)
        assert unlabeled.size > 0
        assert np.all(node_subgraph[unlabeled] == -1)


def test_data_and_membership_agree_on_n():
    """The membership inverse is sized to the same N the Data exposes."""
    with tempfile.TemporaryDirectory() as d:
        out = _make_artifacts(Path(d))
        data = build_pyg_data(out)
        mem = load_subgraph_membership(out / "subgraphs.parquet")
        node_subgraph = mem.node_subgraph(int(data.num_nodes))
        assert node_subgraph.shape[0] == data.x.shape[0]


def test_missing_ccid_raises():
    with tempfile.TemporaryDirectory() as d:
        out = _make_artifacts(Path(d))
        mem = load_subgraph_membership(out / "subgraphs.parquet")
        try:
            mem.index_of("NOPE")
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for an unknown ccId")


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
