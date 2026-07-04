"""Tests for the held-out-recovery proxy eval (ellip2.discovery.eval_recovery, T-030).

CPU-only, synthetic (SIGN-101). Covers the recovery metric + random baseline, the
split-scoped target selection, the split-filtered known-member exclusion (only the
named split's members are excluded), and the CLI end-to-end.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import pytest  # noqa: E402

from ellip2.discovery import background, eval_recovery  # noqa: E402


def _write_subgraphs(
    tmp: Path, cc_ids: list[str], members: list[list[int]], labels: list[str]
) -> Path:
    path = tmp / "subgraphs.parquet"
    pq.write_table(
        pa.table({
            "ccId": pa.array(cc_ids),
            "ccLabel": pa.array(labels),
            "n_members": pa.array([len(m) for m in members], type=pa.int64()),
            "member_idx": pa.array(members, type=pa.list_(pa.int64())),
        }),
        path,
    )
    return path


def _write_discovered(tmp: Path, members: list[list[int]]) -> Path:
    path = tmp / "discovered_subgraphs.parquet"
    pq.write_table(
        pa.table({
            "ccId": pa.array([f"bg{i}" for i in range(len(members))]),
            "ccLabel": pa.array(["unknown"] * len(members)),
            "n_members": pa.array([len(m) for m in members], type=pa.int64()),
            "member_idx": pa.array(members, type=pa.list_(pa.int64())),
        }),
        path,
    )
    return path


def _write_split(tmp: Path, assignments: dict[str, str]) -> Path:
    path = tmp / "split.csv"
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for cc, sp in assignments.items():
            w.writerow([cc, "", sp])
    return path


# --- recovery_rate + baseline -------------------------------------------------


def test_recovery_rate_and_baseline() -> None:
    # discovered union = {0,1,2,3}; D=4 out of N=10.
    discovered = [[0, 1, 2], [2, 3]]
    targets = [[3, 4], [5, 6], [0]]  # A recovered (3), B missed, C recovered (0)
    r = eval_recovery.recovery_rate(discovered, targets, 10)

    assert r.n_targets == 3
    assert r.n_recovered == 2
    assert r.recovery_rate == pytest.approx(2 / 3)
    assert r.discovered_union_size == 4
    # baseline: P(hit|m=2,D=4,N=10)=1-(6/10)(5/9)=2/3 for A and B; m=1 -> 4/10 for C
    expected_baseline = (2 / 3 + 2 / 3 + 0.4) / 3
    assert r.baseline_rate == pytest.approx(expected_baseline)
    assert r.lift == pytest.approx((2 / 3) / expected_baseline)


def test_recovery_baseline_guaranteed_hit() -> None:
    # N-D = 1 but target has 2 members -> it cannot avoid the drawn set: certain hit.
    assert eval_recovery._baseline_hit_prob(2, 4, 5) == pytest.approx(1.0)
    # empty target never recovers, baseline 0.
    assert eval_recovery._baseline_hit_prob(0, 4, 10) == 0.0


def test_recovery_empty_discovered() -> None:
    r = eval_recovery.recovery_rate([], [[0, 1], [2]], 10)
    assert r.discovered_union_size == 0
    assert r.n_recovered == 0
    assert r.recovery_rate == 0.0
    assert r.baseline_rate == 0.0  # D=0 -> zero chance
    assert r.lift == 0.0


def test_recovery_no_targets() -> None:
    r = eval_recovery.recovery_rate([[0, 1]], [], 10)
    assert r.n_targets == 0
    assert r.recovery_rate == 0.0
    assert r.baseline_rate == 0.0
    assert r.lift == 0.0


def test_recovery_out_of_range_members_dropped() -> None:
    # members >= n_nodes are clipped, so a spurious huge idx doesn't inflate the union.
    r = eval_recovery.recovery_rate([[0, 99]], [[0], [99]], 10)
    assert r.discovered_union_size == 1  # only idx 0 survives
    assert r.n_recovered == 1  # target [0] recovered; target [99] emptied -> not


def test_recovery_bad_n_nodes_raises() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        eval_recovery.recovery_rate([[0]], [[0]], 0)


# --- split-scoped target selection -------------------------------------------


def test_suspicious_targets_split_scoped(tmp_path: Path) -> None:
    subs = _write_subgraphs(
        tmp_path,
        ["cc0", "cc1", "cc2", "cc3"],
        [[0, 1], [2, 3], [4, 5], [6, 7]],
        ["suspicious", "suspicious", "licit", "suspicious"],
    )
    split = _write_split(
        tmp_path, {"cc0": "train", "cc1": "test", "cc2": "test", "cc3": "test"}
    )
    targets = eval_recovery.suspicious_targets(subs, split, "test")
    # only suspicious subgraphs in the test split: cc1 and cc3 (cc0 is train, cc2 licit)
    got = sorted(t.tolist() for t in targets)
    assert got == [[2, 3], [6, 7]]


# --- split-filtered known-member exclusion (background.known_member_idx) -------


def test_known_member_idx_split_filter_excludes_named_split_only(tmp_path: Path) -> None:
    subs = _write_subgraphs(
        tmp_path,
        ["cc0", "cc1", "cc2"],
        [[0, 1], [2, 3], [4, 5]],
        ["suspicious", "suspicious", "licit"],
    )
    split = _write_split(tmp_path, {"cc0": "train", "cc1": "test", "cc2": "val"})

    # unfiltered: every labeled member is excluded
    all_known = background.known_member_idx(subs)
    np.testing.assert_array_equal(all_known, np.array([0, 1, 2, 3, 4, 5]))

    # split-filtered: only train-split members are excluded; test/val stay seedable
    train_only = background.known_member_idx(subs, split_csv=split, split_name="train")
    np.testing.assert_array_equal(train_only, np.array([0, 1]))


def test_known_member_idx_split_filter_requires_both_args(tmp_path: Path) -> None:
    subs = _write_subgraphs(tmp_path, ["cc0"], [[0, 1]], ["suspicious"])
    split = _write_split(tmp_path, {"cc0": "train"})
    # split_csv without split_name -> no filtering (back-compat default)
    got = background.known_member_idx(subs, split_csv=split)
    np.testing.assert_array_equal(got, np.array([0, 1]))


# --- CLI end-to-end -----------------------------------------------------------


def test_eval_recovery_main_end_to_end(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    discovered = _write_discovered(tmp_path, [[0, 1, 2], [2, 3]])
    subs = _write_subgraphs(
        tmp_path,
        ["cc0", "cc1", "cc2"],
        [[0, 5], [6, 7], [8, 9]],
        ["suspicious", "suspicious", "suspicious"],
    )
    split = _write_split(tmp_path, {"cc0": "train", "cc1": "test", "cc2": "test"})

    rc = eval_recovery.main([
        "--discovered", str(discovered),
        "--subgraphs", str(subs),
        "--split-csv", str(split),
        "--eval-split", "test",
        "--n-nodes", "10",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # targets = cc1[6,7], cc2[8,9] (both test-suspicious); discovered union {0,1,2,3}
    # neither test target overlaps the union -> 0 recovered.
    assert "targets(test,suspicious)=2" in out
    assert "recovered=0" in out


if __name__ == "__main__":
    import tempfile

    test_recovery_rate_and_baseline()
    test_recovery_baseline_guaranteed_hit()
    test_recovery_empty_discovered()
    test_recovery_no_targets()
    test_recovery_out_of_range_members_dropped()
    test_recovery_bad_n_nodes_raises()
    for fn in (
        test_suspicious_targets_split_scoped,
        test_known_member_idx_split_filter_excludes_named_split_only,
        test_known_member_idx_split_filter_requires_both_args,
    ):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
    print("ok")
