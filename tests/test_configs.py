"""Validate the Hydra config tree composes and required keys resolve (T-022).

hydra-core / omegaconf are intentionally NOT installed in the offline CPU test
gate, so this test follows the acceptance-criteria fallback: load the YAML with
PyYAML, resolve the top-level ``defaults`` composition manually, and assert each
group's required keys are present and well-typed. The config files are still
authored as composable Hydra groups (a ``defaults`` list + one option YAML per
group), so adding hydra later needs no restructuring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"

# Every group the composition root must wire together (plan.md sec 6).
GROUPS = ("dataset", "features", "pu", "gnn", "llm", "infra")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path} is not a YAML mapping"
    return data


def _parse_defaults(root: dict[str, Any]) -> dict[str, str]:
    """Return {group: option} from the Hydra-style ``defaults`` list."""
    defaults = root.get("defaults")
    assert isinstance(defaults, list), "config.yaml must have a defaults list"
    chosen: dict[str, str] = {}
    for entry in defaults:
        if entry == "_self_":
            continue
        assert isinstance(entry, dict) and len(entry) == 1, f"bad defaults entry: {entry!r}"
        (group, option), = entry.items()
        chosen[group] = option
    return chosen


def _compose() -> dict[str, Any]:
    """Minimal Hydra compose: load each selected group option under its group key,
    then overlay the root's scalar keys (the ``_self_`` step)."""
    root = _load_yaml(CONFIGS_DIR / "config.yaml")
    chosen = _parse_defaults(root)
    composed: dict[str, Any] = {}
    for group, option in chosen.items():
        group_path = CONFIGS_DIR / group / f"{option}.yaml"
        assert group_path.exists(), f"missing group file {group_path}"
        composed[group] = _load_yaml(group_path)
    for key, value in root.items():
        if key != "defaults":
            composed[key] = value
    return composed


def test_config_root_exists() -> None:
    assert (CONFIGS_DIR / "config.yaml").is_file()


def test_every_group_has_a_default_option() -> None:
    root = _load_yaml(CONFIGS_DIR / "config.yaml")
    chosen = _parse_defaults(root)
    for group in GROUPS:
        assert group in chosen, f"defaults list is missing group {group!r}"
        assert (CONFIGS_DIR / group / f"{chosen[group]}.yaml").is_file()


def test_all_group_options_are_valid_yaml_mappings() -> None:
    # Every option file under every group must load as a mapping (incl. the
    # non-default pu/nnpu.yaml alternative).
    for group_dir in (CONFIGS_DIR / g for g in GROUPS):
        yamls = sorted(group_dir.glob("*.yaml"))
        assert yamls, f"group {group_dir.name} has no option files"
        for path in yamls:
            _load_yaml(path)


def test_compose_resolves_all_groups_and_run_scalars() -> None:
    cfg = _compose()
    for group in GROUPS:
        assert group in cfg, f"composed config missing group {group!r}"
        assert isinstance(cfg[group], dict)
    # Run-level scalars overlaid from config.yaml itself.
    assert cfg["seed"] == 42
    assert cfg["run_name"] == "ellip2-baseline"
    assert cfg["artifacts_dir"]


def test_required_keys_resolve_per_group() -> None:
    cfg = _compose()

    ds = cfg["dataset"]
    for key in ("n_node_features", "n_edge_features", "n_subgraphs", "raw_dir", "split_csv"):
        assert key in ds, f"dataset missing {key}"
    assert ds["n_node_features"] == 43
    assert ds["n_edge_features"] == 95

    feats = cfg["features"]
    for key in ("weight_index", "timestamp_index", "size_index", "edge_agg_indices", "hops"):
        assert key in feats, f"features missing {key}"
    assert feats["hops"] == [1, 2]

    pu = cfg["pu"]
    assert pu["framing"] == "supervised"  # default framing (Resolved decision #2)
    for key in ("loss", "epochs", "lr"):
        assert key in pu

    gnn = cfg["gnn"]
    assert gnn["in_channels"] == 43
    assert gnn["num_neighbors"] == [15, 10]
    for key in ("hidden_channels", "out_channels", "num_layers"):
        assert key in gnn

    llm = cfg["llm"]
    for key in ("model_id", "region", "max_tokens", "max_serialize_bytes", "typologies"):
        assert key in llm, f"llm missing {key}"
    assert set(llm["typologies"]) == {
        "peeling_chain",
        "nested_service",
        "layering_smurfing",
        "consolidation",
    }

    infra = cfg["infra"]
    for key in ("instance_type", "spot", "s3_bucket", "checkpoint_interval_min"):
        assert key in infra, f"infra missing {key}"


def test_pu_nnpu_alternative_has_cluster_prior_keys() -> None:
    # The optional cluster-level PU framing must carry a small prior + sweep grid
    # (plan.md Resolved decision #2): pi_p is NOT the 2.27% subgraph base rate.
    nnpu = _load_yaml(CONFIGS_DIR / "pu" / "nnpu.yaml")
    assert nnpu["framing"] == "nnpu"
    assert 0.0 < nnpu["prior"] < 0.1
    grid = nnpu["prior_sweep"]
    assert grid == sorted(grid, reverse=True)
    assert min(grid) <= 1e-4 and max(grid) >= 1e-1


def test_dataset_base_rate_is_subgraph_level_not_cluster_prior() -> None:
    ds = _load_yaml(CONFIGS_DIR / "dataset" / "elliptic2.yaml")
    # 2,763 / 121,810 ~= 0.0227 (sanity-check the documented base rate).
    assert ds["n_suspicious"] / ds["n_subgraphs"] == pytest.approx(ds["subgraph_base_rate"], abs=1e-3)
