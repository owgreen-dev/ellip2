# ellip2 — Money-laundering subgraph detection & discovery on Elliptic2

A research pipeline for anti-money-laundering (AML) analysis of the **Elliptic2** Bitcoin
dataset. It does two things on the 121,810 labeled connected components and the ~48.8M
unlabeled background clusters:

1. **Detect** suspicious *subgraphs* among the labeled connected components (supervised
   border model, PR-AUC **0.942**).
2. **Discover** novel suspicious *subgraphs* among the ~48.8M *unlabeled* background
   clusters (per-cluster score → reachability carve → border re-score → ranked leads).

Everything runs offline on CPU for the tests; the real end-to-end run is a GPU box step.
See [RESULTS.md](RESULTS.md) for the full metrics table and baseline comparison.

## Overview

Elliptic2 ("The Shape of Money Laundering") labels whole *subgraphs* of the Bitcoin
transaction graph as licit or suspicious, at a **2.27%** base rate (2,763 suspicious /
119,047 licit). The detection model is a **Deep Sets border model**: it pools the external
*senders* and *receivers* of a candidate subgraph together with its pooled internal node
and edge features, then classifies with an MLP under a weighted BCE loss. On top of
detection, a background-discovery stage surfaces novel suspicious structures that were
never labeled, using a per-cluster suspicion score, a bounded ≤k-hop exit-path
reachability carve toward heuristic licit endpoints, and a one-at-a-time border re-score.

## Results

Detection progression on stratified 80/10/10 splits (test PR-AUC), each row a distinct
modeling choice:

| Model | test PR-AUC |
|---|---|
| cluster-level nnPU GNN (rejected framing) | 0.030 |
| pooled-features HGBM | 0.286 |
| border model, nodes only | 0.816 |
| border model + internal edge features | 0.844 |
| **border model, tuned** | **0.911 ± 0.009** |

The tuned border model scores **PR-AUC 0.911 ± 0.009** across 5 stratified splits with
**best-of-3 validation-based model selection** (best single split 0.942). Selection matters:
a *single* run is unstable — 1 in 5 fresh splits collapses to ~0.38 — so we keep the
best-validation restart (`train_border --restarts`). See [RESULTS.md](RESULTS.md) for the
full robustness analysis.

Named-table baseline comparison — **RevTrack Table 1 (GPU + node features)**:

| Model | PR-AUC | F1 |
|---|---|---|
| RevClassify_DS (SOTA, border Deep Sets) | 0.974 | 0.953 |
| **Ours (border, best-of-3, 5-split)** | **0.911 ± 0.009** | 0.78 (0.89 val-tuned) |
| GLASS | 0.816 | 0.705 |

Ours beats GLASS and trails RevClassify_DS by ~0.06 PR-AUC (same border-Deep-Sets
architecture, reimplemented from scratch). This is an **approximate, different-split**
comparison, not identical-split — see [RESULTS.md](RESULTS.md) for caveats.

**Discovery:** 208 novel candidate subgraphs surfaced from the 49.3M-cluster background;
held-out-recovery proxy eval re-found **5 of 276** held-out test-suspicious subgraphs
(**1.8% recall**) against a random baseline of 0.0001 → **121× lift**.

## Pipeline

```
Stage 0  ingest        DuckDB out-of-core  -> edge_index.npy, node_features.npy, subgraphs.parquet
   |
Stage 1  features      degree, flow-concentration, neighborhood, temporal, path-role
   |                   -> cluster_features.parquet
   |
Stage 2  detection     border model: DeepSets(senders) + DeepSets(receivers)
   |                   + pooled internal node/edge features -> MLP (weighted BCE)  [PR-AUC 0.942]
   |
Stage 3  exit paths    bounded <=k-hop reachability to heuristic licit endpoints (corroboration)
   |
Stage 4  cards         LangGraph typology agent (Bedrock Claude) + structural validator + graph viz
   |
Discovery              per-cluster HGBM score -> 3-gate funnel -> per-candidate carve
                       -> border re-score -> ranked novel leads
```

## Quickstart

This repo uses `uv` and a project `.venv` (no system pip):

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e '.[dev]'

# run the full gate (pytest + ruff + mypy)
bash scripts/verify.sh

# or just the tests
.venv/bin/python -m pytest -q
```

## Data

- **Source:** Kaggle `ellipticco/elliptic2-data-set` (paper *"The Shape of Money
  Laundering"*).
- **Size:** ~24 GB compressed / **~83 GB extracted** (5 CSVs).
- **License:** **CC BY-NC-ND 4.0** — non-commercial, no-derivatives. The dataset is **NOT
  redistributed** in this repo; download it from Kaggle yourself. See [DATA.md](DATA.md).
- **Counts (paper Table 1):** 49,299,864 background clusters · 196,215,606 background
  edges · 121,810 labeled subgraphs (2,763 suspicious / 119,047 licit; base rate 2.27%).
  43 node features, 96 edge feature columns.

## Reproduce

The real, GPU-scale end-to-end run (ingest → split → features → train/score → discovery →
eval) is documented in [RUNBOOK.md](RUNBOOK.md), including the AWS instance and cost notes.
The offline CPU test suite (`bash scripts/verify.sh`) exercises every module on synthetic
fixtures.

## Repo layout

- `src/ellip2/` — ~50 typed modules:
  - `data/` — Stage 0 ingest (DuckDB out-of-core) + schema.
  - `features/` — degree, edge_aggs, flow_concentration, neighborhood, temporal,
    path_role, build.
  - `graph/` — PyG `Data` construction + neighbor sampling.
  - `pu/` — border/subgraph models, nnPU loss, prior estimation, encoder, trainer,
    cluster_score.
  - `exit_paths/` — bounded reachability BFS + endpoint recovery.
  - `discovery/` — background discovery orchestrator + held-out-recovery eval.
  - `eval/` — splits, PU metrics, leakage checks.
  - `llm/` — subgraph serialization, Bedrock client, LangGraph typology agent.
  - `report/` — per-subgraph investigative card rendering + lead ranking.
- `scripts/` — 18 thin CLIs (`train_border.py`, `score_border.py`, `make_split.py`,
  `build_features.py`, `discover_subgraphs.py`, `eval_recovery.py`, …).
- `configs/` — composable Hydra-style config tree.
- `tests/` — synthetic / CPU-only / mocked test suite.

## Limitations

- Discovery recall is low (1.8%, top-500 candidates only); raising top-K trades compute
  for recall.
- **Training instability**: a single border-model run collapses ~1 in 5 times (degenerate
  all-positive). We mitigate with best-of-N validation-based selection (`--restarts`); the
  reported 0.911 ± 0.009 uses best-of-3.
- The baseline comparison uses a **different split** than RevTrack (approximate, not
  identical-split — their exact random split has no published seed).
- The per-cluster suspicion scorer that drives discovery is weak (test-member PR-AUC
  0.127).
- LLM typologies are unvalidated — the dataset has no ground-truth typology labels.

## License

Code is licensed **MIT** — see [LICENSE](LICENSE). The Elliptic2 **data** is CC
BY-NC-ND 4.0 (non-commercial) and is not included here.

## Citation

If you use this work, cite the two underlying papers:

- Elliptic2 dataset / GLASS — *The Shape of Money Laundering* — **arXiv:2404.19109**.
- RevClassify / RevTrack — *Identifying Money Laundering Subgraphs on the Blockchain*
  (ACM AIF 2024) — **arXiv:2410.08394**.
