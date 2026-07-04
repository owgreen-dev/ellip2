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
