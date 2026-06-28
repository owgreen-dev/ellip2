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

- [T-004] `features/neighborhood.py`: `compute_neighborhood_features(edge_index, node_subgraph, subgraph_label, subgraph_in_test, n_nodes, *, hops=(1,2), empty_value=0.0)` → dict keyed `hop{h}_frac_{licit,suspicious,unknown}` (float64 (N,), sum to 1 on non-empty neighbor sets), plus loader `load_subgraph_labels(subgraphs_parquet, split_csv, n_nodes)` → `SubgraphLabels(node_subgraph, subgraph_label, subgraph_in_test, ccids)`. **Label codes:** `LICIT=0, SUSPICIOUS=1`; `node_subgraph[i] = -1` for unlabeled background nodes (most of the 49M). **Graph is symmetrised** (undirected) — label homophily is direction-agnostic. `hop1` = immediate neighbors; `hop2` = ring at shortest distance *exactly* 2 (distance-1 + self removed). Denominator = full neighbor-set size, so masked neighbors stay in as `unknown`. **Leakage (SIGN-103), two masks:** (1) own-subgraph — same-subgraph neighbors carry v's own label, removed via `lic = A@lic_ind - same*lic_ind` where `same[v]=(A@M).multiply(M).sum(1)` is the same-subgraph neighbor count and `lic_ind[v]` (the node's own observable indicator) IS the self-mask since members share it; (2) test-split — globally zero a node's licit/suspicious indicator if its subgraph is in the persisted TEST split. **scipy.sparse:** boolean csr adjacency; 2-hop = `(A@A>0)` minus `(A@A>0).multiply(A)` (intersection only, no dense N×N) then `setdiag(0)`. **Gotchas:** K=0 (all-unlabeled) breaks fancy-indexing `labels[ns[valid]]` — guard with `if labels.size`. Test nodes still GET features from their non-test neighbors (only their OWN label is withheld downstream). Leakage tests use positive controls (flip mask off → fraction changes) so silent un-masking fails loudly. 51 tests green, ruff+mypy clean.

- [T-003] `features/flow_concentration.py` was already present (untracked); this iteration added the missing `tests/test_flow_concentration.py`. API: `compute_flow_concentration(edge_index, weights, n_nodes, *, empty_value=0.0)` → dict keyed by `COLUMNS` = `{in,out}_{gini,hhi,max_counterparty_share}`, all float64 (N,). **Conventions (reuse):** out-flow grouped by source (row0), in-flow by target (row1); per-counterparty weights are SUMMED first (parallel edges merge into one share) so Gini/HHI/share are consistent. Gini via sorted-rank closed form `Σ(2·rank−n−1)·w_(i) / (n·S)`, clipped to [0,1]; HHI=Σ(w_i/S)²; share=max w_i/S. Empty direction (S==0) → `empty_value`. **Test gotchas:** `weights` is the 2nd POSITIONAL arg (before `n_nodes`) — don't pass it as kw after a positional n_nodes. Gini ceiling for one-dominant is (n−1)/n, so "near 1" only holds with many counterparties (n=4 dominant → 0.72, not ~1); assert exact hand value + monotone-vs-equal instead. Hand graph: equal-after-merge (Gini 0), single counterparty (Gini 0/HHI 1/share 1), one-dominant-of-4 (Gini 0.72/HHI 0.9412/share 0.97). 40 tests green, ruff+mypy clean.

- [T-002] `features/edge_aggs.py`: two entry points sharing one column convention. `compute_edge_aggregates(edge_index, edge_features(E,F), n_nodes, *, feature_indices, feature_names, stats=("sum","mean","max","std"), empty_value=0.0)` → numpy in-memory path (test). `compute_edge_aggregates_duckdb(edges_csv, id_map_parquet, n_nodes, ...)` → out-of-core: one `GROUP BY` per direction over `read_csv_auto(edges)` JOIN `read_parquet(id_map)` (SIGN-104; mirrors ingest's `_q`/`_qi`/`to_arrow_reader`). **Conventions:** out-edges grouped by src (row0), in-edges by dst (row1). Columns named `{direction}_{feature}_{stat}` (e.g. `out_ef_0_sum`), all float64 (N,). **std is POPULATION std** (group of 1 → 0.0); DuckDB side uses `stddev_pop` to match. numpy std via `sumsq/count - mean²` clipped at 0 for fp. Nodes with no edge in a direction get `empty_value` (default 0.0, distinct from a genuine zero sum). Test asserts hand-computed in/out aggregates AND that the DuckDB path matches numpy (tiny synthetic CSV + id_map). Gotcha: `feature_names` length must equal `feature_indices` (default indices = ALL columns), so pass both together for a subset.
- [T-001] `features/degree.py`: `compute_degree_features(edge_index, n_nodes, *, zero_denom_value=0.0)` → dict of (N,) arrays keyed by `COLUMNS = (in_degree, out_degree, total_degree, in_out_ratio)`. Pure-numpy `bincount` (row0=src→out_degree, row1=dst→in_degree). **Convention to reuse:** feature modules return `dict[str, np.ndarray]` (column→(N,) array) + a `COLUMNS` tuple, so T-007 can join by name. `in_out_ratio` is undefined when out_degree==0 → filled with `zero_denom_value` (default 0.0, finite, no NaN/inf) — kept distinct from a *genuine* 0.0 ratio (in=0,out>0). Tests use a hand-counted 6-node digraph (pure sink + isolated node) and assert the zero-denom fill vs genuine-zero separately. Int cols are int64, ratio float64.
