# Ralph Progress Log — ellip2 (Bitcoin/Elliptic2 AML)

Cross-session memory. The fresh-context loop reads this each iteration. Append what you
learned; keep it short and high-signal.

## Project orientation (read once)

- **Master spec:** `plan.md` (staged pipeline §9, repo layout §6, Resolved design
  decisions near the top). Tasks live in `plans/prd.json`. Rules in `plans/guardrails.md`.
- **Goal:** subgraph-level AML detection on Elliptic2 + cluster-level PU, then exit-path
  discovery + an LLM typology/report layer.
- **Verify gate:** `bash scripts/verify.sh` = pytest + ruff + mypy, all via `.venv`.
- **Env:** no system pip — use `.venv/bin/python` and `uv`. CPU only (no GPU). Tests are
  synthetic/mocked (see `tests/test_ingest.py`, `tests/test_splits.py`).

## Done before the loop started

- **Stage 0 ingest** (`src/ellip2/data/ingest.py`, `schema.py`): DuckDB out-of-core →
  `id_map.parquet`, `node_features.npy`, `edge_index.npy`, `subgraphs.parquet`,
  `ingest_manifest.json`. Integrity checks. Tested (`tests/test_ingest.py`).
- **Split generator** (`src/ellip2/eval/splits.py`, `splits_cli.py`): persisted
  subgraph-level train/val/test (stratified_random default; round_robin reproduces
  GLASS). Tested (`tests/test_splits.py`).
- Baseline is **green**: 13 tests, ruff clean, mypy clean.

## Stage → modules still to build (see prd.json for the ordered backlog)

- **Stage 1 features** → `src/ellip2/features/{degree,edge_aggs,flow_concentration,neighborhood,temporal,path_role,build}.py`
- **Stage 2 graph+PU** → `src/ellip2/graph/{pyg_data,neighbor_sampling}.py`,
  `src/ellip2/pu/{nnpu_loss,prior_estimation,encoder,trainer}.py`,
  `src/ellip2/eval/{pu_metrics,leakage_checks}.py`
- **Stage 3 discovery** → `src/ellip2/exit_paths/path_search.py`, `scripts/discover.py`
- **LLM + report** → `src/ellip2/llm/{serialize_subgraph,bedrock_client,typology_graph}.py`,
  `src/ellip2/report/render.py`
- **Config** → `configs/` (Hydra). **Skipped:** infra Stages 1-3, real-data e2e.

## Iteration notes

<!-- Append: [T-00X] one line on what was built, key decision, any gotcha. -->

- [T-001] `features/degree.py`: `compute_degree_features(edge_index, n_nodes, *, zero_denom_value=0.0)` → dict of (N,) arrays keyed by `COLUMNS = (in_degree, out_degree, total_degree, in_out_ratio)`. Pure-numpy `bincount` (row0=src→out_degree, row1=dst→in_degree). **Convention to reuse:** feature modules return `dict[str, np.ndarray]` (column→(N,) array) + a `COLUMNS` tuple, so T-007 can join by name. `in_out_ratio` is undefined when out_degree==0 → filled with `zero_denom_value` (default 0.0, finite, no NaN/inf) — kept distinct from a *genuine* 0.0 ratio (in=0,out>0). Tests use a hand-counted 6-node digraph (pure sink + isolated node) and assert the zero-denom fill vs genuine-zero separately. Int cols are int64, ratio float64.
