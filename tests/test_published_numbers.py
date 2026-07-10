"""Drift-check: every published number is machine-checked against facts.json.

`facts.json` (repo root) is the single machine-readable source of truth for the numbers
quoted in README.md and RESULTS.md. This test asserts each canonical value string appears
verbatim in the doc(s) that must contain it, so the CI gate fails if a hand-edit drifts a
number in the docs (or in facts.json) without the other being updated to match.

Pure stdlib; no data, no network, no GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_FACTS_PATH = _ROOT / "facts.json"


def _load_facts() -> list[dict[str, object]]:
    facts = json.loads(_FACTS_PATH.read_text(encoding="utf-8"))["facts"]
    assert facts, "facts.json has no facts"
    return facts


# One (value, doc) pair per doc a fact must appear in, so a drift points at the exact miss.
_CASES = [
    (str(fact["value"]), doc)
    for fact in _load_facts()
    for doc in fact["in"]  # type: ignore[attr-defined]
]


@pytest.mark.parametrize(("value", "doc"), _CASES, ids=[f"{d}:{v}" for v, d in _CASES])
def test_published_number_present(value: str, doc: str) -> None:
    """Presence check: catches the primary drift mode — facts.json updated (e.g. a re-run
    produces a new PR-AUC) but a doc left stale, so the canonical string is missing."""
    text = (_ROOT / doc).read_text(encoding="utf-8")
    assert value in text, (
        f"canonical value {value!r} from facts.json is not present in {doc}; "
        f"the docs and facts.json have drifted — reconcile them."
    )


# The headline PR-AUC is quoted in many places; a presence check alone would miss a *partial*
# edit (one of several copies drifting). Pin the invariant instead: the std "± 0.009" is unique
# to the robust headline, so every occurrence must be attached to the mean "0.911 ± 0.009".
# Derived from facts.json (the "± 0.0.." fact), so it tracks the source rather than hard-coding.
_HEADLINE = next(
    (str(f["value"]) for f in _load_facts() if str(f["value"]).count("±") == 1),
    "0.911 ± 0.009",
)
_STD_TOKEN = _HEADLINE.split(" ", 1)[1] if " " in _HEADLINE else _HEADLINE  # e.g. "± 0.009"


@pytest.mark.parametrize("doc", ["README.md", "RESULTS.md"])
def test_headline_pr_auc_not_partially_drifted(doc: str) -> None:
    text = (_ROOT / doc).read_text(encoding="utf-8")
    # Every "± 0.009" must be the full "0.911 ± 0.009" — a lone/altered mean means one copy
    # was hand-edited out of sync with the rest.
    stray = [
        m.start()
        for m in re.finditer(re.escape(_STD_TOKEN), text)
        if not text[max(0, m.start() - len(_HEADLINE)) : m.end()].endswith(_HEADLINE)
    ]
    assert not stray, (
        f"{doc} contains a '{_STD_TOKEN}' not attached to '{_HEADLINE}' "
        f"(offsets {stray}); a headline PR-AUC copy has drifted — reconcile it."
    )
