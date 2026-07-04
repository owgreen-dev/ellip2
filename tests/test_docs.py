"""Publishing/documentation gate tests.

These assert the repo's front-door docs exist and stay consistent with the canonical
facts in ``plans/publishing_facts.md`` (the single source of truth). No network, no
external resources (SIGN-101) — pure file reads over tracked docs.

Grows across the publishing tasks (T-031..T-036); this iteration covers T-031 (README).
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Required top-level section headers in README.md (T-031 acceptance criteria).
README_SECTIONS = (
    "## Overview",
    "## Results",
    "## Pipeline",
    "## Quickstart",
    "## Data",
    "## Reproduce",
    "## Repo layout",
    "## Limitations",
    "## License",
    "## Citation",
)


def _read(name: str) -> str:
    return (REPO_ROOT / name).read_text(encoding="utf-8")


def test_readme_exists() -> None:
    assert (REPO_ROOT / "README.md").is_file()


def test_readme_has_all_sections() -> None:
    text = _read("README.md")
    for header in README_SECTIONS:
        assert header in text, f"README.md missing section header: {header!r}"


def test_readme_data_size_correct() -> None:
    text = _read("README.md")
    assert "83" in text, "README.md must state the ~83 GB extracted data size"
    assert "26 GB" not in text and "26GB" not in text, (
        "README.md must not use the stale '26 GB' data size"
    )


def test_readme_links_results() -> None:
    text = _read("README.md")
    assert "RESULTS.md" in text, "README.md must link RESULTS.md"


# --- T-032: LICENSE (MIT) + non-commercial data note ---


def test_license_exists_and_is_mit() -> None:
    path = REPO_ROOT / "LICENSE"
    assert path.is_file(), "LICENSE file must exist at repo root"
    assert "MIT" in path.read_text(encoding="utf-8"), "LICENSE must be MIT"


def test_data_noncommercial_note_present() -> None:
    """The CC BY-NC-ND / non-commercial data note must appear in a tracked doc."""
    docs = ("README.md", "DATA.md")
    found = any(
        (REPO_ROOT / d).is_file()
        and (
            "CC BY-NC-ND" in _read(d) or "non-commercial" in _read(d)
        )
        for d in docs
    )
    assert found, "A non-commercial data-license note must be present in a tracked doc"


# --- T-033: RESULTS.md (metrics + baseline comparison) ---


def test_results_exists() -> None:
    assert (REPO_ROOT / "RESULTS.md").is_file(), "RESULTS.md must exist at repo root"


def test_results_has_baseline_comparison() -> None:
    """RESULTS.md must carry the NAMED baseline table (RevClassify + GLASS)."""
    text = _read("RESULTS.md")
    assert "RevClassify" in text, "RESULTS.md must name RevClassify in the baseline table"
    assert "GLASS" in text, "RESULTS.md must name GLASS in the baseline table"


def test_results_has_caveats_section() -> None:
    text = _read("RESULTS.md")
    assert "Limitations" in text or "Caveats" in text or "caveats" in text, (
        "RESULTS.md must have a Limitations/Caveats section"
    )


def test_headline_pr_auc_consistent_across_docs() -> None:
    """The headline PR-AUC 0.942 must appear in BOTH README and RESULTS."""
    assert "0.942" in _read("README.md"), "README.md must state the headline PR-AUC 0.942"
    assert "0.942" in _read("RESULTS.md"), "RESULTS.md must state the headline PR-AUC 0.942"


# --- T-034: fix stale docs (data size + superseded banner) ---


def _tracked_md_files() -> list[str]:
    """Git-tracked *.md paths (limits the scan to repo docs, not .venv packages)."""
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in out.stdout.splitlines() if line]


def test_no_stale_data_size_in_tracked_docs() -> None:
    """No tracked *.md may carry the stale '26 GB'/'26GB' data size (canonical: ~83 GB)."""
    offenders = [
        f
        for f in _tracked_md_files()
        if "26 GB" in _read(f) or "26GB" in _read(f)
    ]
    assert not offenders, f"stale '26 GB'/'26GB' data size found in: {offenders}"


def test_plan_has_superseded_banner() -> None:
    """plan.md must note it is the original design doc, superseded by README/RESULTS."""
    text = _read("plan.md")
    assert "README.md" in text and "RESULTS.md" in text, (
        "plan.md must point to README.md/RESULTS.md"
    )
    assert "superseded" in text.lower() or "supersede" in text.lower(), (
        "plan.md must state it is superseded by the current results docs"
    )


# --- T-035: pyproject metadata + reproduce commands ---


def _load_pyproject() -> dict:
    raw = (REPO_ROOT / "pyproject.toml").read_bytes()
    return tomllib.loads(raw.decode("utf-8"))


def test_pyproject_parses_and_has_metadata_keys() -> None:
    """pyproject.toml must parse and the [project] table must carry the publishing keys."""
    project = _load_pyproject()["project"]
    for key in ("authors", "keywords", "classifiers", "urls"):
        assert project.get(key), f"pyproject [project] missing/empty key: {key!r}"


def test_pyproject_license_is_mit() -> None:
    project = _load_pyproject()["project"]
    license_field = project["license"]
    text = license_field if isinstance(license_field, str) else license_field.get("text", "")
    assert "MIT" in text, "pyproject [project].license must be MIT"


def test_pyproject_core_config_intact() -> None:
    """The metadata additions must not disturb requires-python or the deps list."""
    project = _load_pyproject()["project"]
    assert project["requires-python"] == ">=3.11"
    assert any(dep.startswith("duckdb") for dep in project["dependencies"])


def test_reproduce_documents_cli_sequence() -> None:
    """A repro doc must list the end-to-end script sequence in order."""
    candidates = ("REPRODUCE.md", "RUNBOOK.md")
    text = "".join(_read(d) for d in candidates if (REPO_ROOT / d).is_file())
    ordered_clis = (
        "build_features",
        "train_border",
        "score_border",
        "make_endpoints",
        "investigate",
        "train_cluster",
        "score_cluster",
        "make_typology_signal",
        "discover_subgraphs",
        "eval_recovery",
    )
    positions = []
    for cli in ordered_clis:
        assert cli in text, f"repro doc must mention the {cli} CLI"
        positions.append(text.index(cli))
    assert positions == sorted(positions), "repro doc must list the CLIs in pipeline order"
