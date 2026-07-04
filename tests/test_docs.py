"""Publishing/documentation gate tests.

These assert the repo's front-door docs exist and stay consistent with the canonical
facts in ``plans/publishing_facts.md`` (the single source of truth). No network, no
external resources (SIGN-101) — pure file reads over tracked docs.

Grows across the publishing tasks (T-031..T-036); this iteration covers T-031 (README).
"""

from __future__ import annotations

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
