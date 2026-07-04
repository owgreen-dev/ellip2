# Ralph Guardrails (Signs)

Learned constraints that prevent repeated failures. Each "sign" is a rule discovered through iteration failures. Add new signs as you encounter failure patterns.

> "Progress should persist. Failures should evaporate." - The Ralph philosophy

---

## Verification Signs

### SIGN-001: Verify Before Complete
**Trigger:** About to output completion promise
**Instruction:** ALWAYS run the verification command (`bash scripts/verify.sh`) and confirm it passes before outputting `<promise>COMPLETE</promise>`
**Reason:** Models tend to declare victory without proper verification

### SIGN-002: Check All Tasks Before Complete
**Trigger:** Completing a task in multi-task mode
**Instruction:** Re-read prd.json and count remaining `passes: false` tasks. Only output completion promise when ALL tasks pass, not just the current one.
**Reason:** Premature completion exits loop with work remaining

---

## Progress Signs

### SIGN-003: Document Learnings
**Trigger:** Completing any task
**Instruction:** Update progress.md with what was learned (patterns discovered, files modified, decisions made) before ending iteration
**Reason:** Future iterations need context to avoid re-discovering the same patterns

### SIGN-004: Small Focused Changes
**Trigger:** Making changes per iteration
**Instruction:** Keep changes small and focused. Commit incrementally when tests pass. Don't try to solve everything in one iteration.
**Reason:** Large changes are harder to debug when verification fails

---

## Task Management Signs

### SIGN-005: Use Skip for Manual Tasks
**Trigger:** Encountering a task that requires manual human intervention (creating accounts, API keys, dashboard configuration)
**Instruction:** Set `skip: true` and `skipReason` in prd.json for tasks that cannot be automated. The Ralph loop will ignore skipped tasks and can complete without them.
**Reason:** Allows loop to complete automatable work without blocking on manual steps

### SIGN-006: Reference GitHub Issues in Commits
**Trigger:** Committing changes for a prd.json task
**Instruction:** Include `Fixes #N` or `Closes #N` in commit message body (where N is the `github_issue` from prd.json). Format: `fix: description\n\nFixes #61`
**Reason:** Auto-closes GitHub issues when merged to main, maintains traceability

---

## Project-Specific Signs

These are the standing rules for the ellip2 (Bitcoin/Elliptic2 AML) project. Honor
them every iteration. Add more as failure patterns emerge.

### SIGN-100: Always use the project venv; never system pip
**Trigger:** Running any Python, installing any package
**Instruction:** Use `.venv/bin/python` (and `.venv/bin/python -m pytest/ruff/mypy`).
The system `python3` has NO pip and NO ensurepip. To install, use
`uv pip install --python .venv/bin/python <pkg>`. Never call `python3 -m pip`.
**Reason:** The only working interpreter+deps live in `.venv`; system pip is absent.

### SIGN-101: Tests are synthetic / CPU-only / mocked — no external resources
**Trigger:** Writing any test
**Instruction:** Generate tiny fixtures in a tmp dir (follow `tests/test_ingest.py` and
`tests/test_splits.py`). NEVER download the ~83 GB Elliptic2 dataset, NEVER call AWS /
Bedrock over the network (inject/stub the client), NEVER require a GPU
(`torch.cuda.is_available()` is False here — keep everything on CPU), NEVER need real
S3. A test that needs any of these does not belong in the suite; gate the logic with a
mock and leave the real integration as a `skip: true` task.
**Reason:** The verify gate runs offline on CPU; resource-bound tests break the loop.

### SIGN-102: Honor plan.md's Resolved design decisions
**Trigger:** Implementing any Stage 2/3 or feature module
**Instruction:** Read the "Resolved design decisions" block in `plan.md` first. Specifically:
(1) unit = labeled **subgraph**; score clusters then **max-pool** member scores (MIL);
(2) subgraph level is **supervised** (weighted BCE) by default — genuine nnPU is for the
**cluster** level with small π_p (NOT the 2.27% subgraph base rate);
(3) endpoint = the labeled **licit receiver**; entity typing is an explicit **heuristic**;
(4) ≤6-hop path search is **reachability** (backward+forward BFS, frontier caps, hub
exclusion), NOT path enumeration. Do not reintroduce these reconciled mistakes.
**Reason:** These were settled deliberately; regressing them wastes the whole pipeline.

### SIGN-103: Leakage safety is a hard requirement
**Trigger:** Building neighborhood / label-derived features (Stage 1) or eval splits
**Instruction:** Neighborhood-label features must exclude the subgraph's OWN label and
any TEST-split subgraph labels. Add invariant unit tests asserting this. Reuse the
persisted split from `ellip2.eval.splits` — do not invent a new split.
**Reason:** Temporal/label leakage is the known Elliptic pitfall; silent leakage
invalidates every downstream metric.

### SIGN-104: Keep the out-of-core / memmap patterns
**Trigger:** Touching the 49M-node / 196M-edge data path
**Instruction:** Aggregate with DuckDB group-bys; write big arrays via numpy memmap
(`np.lib.format.open_memmap`). NEVER load the full CSVs or 49M rows into pandas.
**Reason:** The target box is 16 GiB; materializing the graph in RAM OOMs.

### SIGN-105: Don't modify finished, tested modules without cause
**Trigger:** Tempted to edit `src/ellip2/data/*` or `src/ellip2/eval/splits*`
**Instruction:** Stage 0 ingest and the split generator are complete and tested. Only
touch them if your task's acceptance criteria explicitly require it; otherwise import
and reuse them.
**Reason:** Avoid breaking green code that later tasks depend on.

### SIGN-106: One module, one responsibility, typed, matching existing idiom
**Trigger:** Creating a new module
**Instruction:** Mirror the structure/docstring density/type-hint style of
`data/ingest.py` and `eval/splits.py`. Add `from __future__ import annotations`. Create
the package `__init__.py` when first needed. Keep ruff (E,F,I,UP,B) and mypy clean.
**Reason:** The gate enforces ruff+mypy; consistent idiom keeps the suite green.

### SIGN-107: This repo has no GitHub — ignore the `Fixes #N` instruction
**Trigger:** Committing a completed task
**Instruction:** Tasks have no `github_issue`. Commit as
`feat: [T-00X] <description>` with no `Fixes #N` line. Do not try to use `gh` or push
to a remote.
**Reason:** Local-only repo; SIGN-006's GitHub step does not apply here.
