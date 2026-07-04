# Publishing canonical facts (SINGLE SOURCE OF TRUTH)

The publishing docs (README.md, RESULTS.md, etc.) MUST use these exact numbers. Do NOT
invent or estimate metrics — copy from here. If a number isn't here, omit it rather than guess.

## What the project is
Elliptic2 Bitcoin anti-money-laundering (AML) research pipeline. Two capabilities:
1. **Detect** suspicious *subgraphs* among the 121,810 labeled connected components.
2. **Discover** novel suspicious subgraphs among the ~48.8M *unlabeled* background clusters.

## Dataset
- Source: Kaggle `ellipticco/elliptic2-data-set` (paper "The Shape of Money Laundering").
- Size: **~24 GB compressed / ~83 GB extracted** (5 CSVs). NOT "26 GB".
- License: **CC BY-NC-ND 4.0** (non-commercial, no-derivatives). Data is NOT redistributed in this repo.
- Validated counts (paper Table 1): **49,299,864** background clusters (nodes) ·
  **196,215,606** background edges · **121,810** labeled subgraphs
  (**2,763 suspicious / 119,047 licit**; base rate **2.27%**). 43 node features, 96 edge feature
  columns in the real CSV (paper says 95).

## Papers (cite these)
- Elliptic2 dataset / GLASS: **arXiv:2404.19109** ("The Shape of Money Laundering").
- RevClassify / RevTrack: **arXiv:2410.08394** ("Identifying Money Laundering Subgraphs on the
  Blockchain", ACM AIF 2024).

## Pipeline stages
Stage 0 ingest (DuckDB → edge_index.npy, node_features.npy, subgraphs.parquet) → Stage 1 features
(degree, flow-concentration, neighborhood, temporal, path-role → cluster_features.parquet) →
**Stage 2 detection: the border model** (Deep Sets over external senders ⊕ receivers ⊕ pooled
internal node & edge features → MLP, weighted BCE) → Stage 3 exit-path reachability corroboration
(heuristic licit endpoints + one representative ≤k-hop path) → Stage 4 investigative cards
(LangGraph typology agent over Bedrock Claude + structural validator + networkx/matplotlib graph
viz) → **Background discovery** (per-cluster HGBM score → 3-gate funnel → per-candidate carve →
border-score → ranked novel leads).

## Detection results (test PR-AUC, our OWN stratified 80/10/10 split)
| Model | test PR-AUC |
|---|---|
| cluster-level nnPU GNN (rejected framing) | 0.030 |
| pooled-features HGBM | 0.286 |
| border model, nodes only | 0.816 |
| border model + internal edge features | 0.844 |
| **border model, tuned** | **0.942** |
Tuned border model full metrics: **PR-AUC 0.942, F1 0.854, recall 0.935**.
Tuned config: border-cap 128, set-hidden 128, set-out 64, mlp-hidden [128, 64], epochs 150,
lr 0.01, weight-decay 1e-4. (Node-only variant used for discovery scoring: PR-AUC 0.80.)

## Published baseline comparison — ALWAYS NAME THE TABLE
Comparable setting = **RevTrack Table 1 (GPU + node features)**:
| Model | PR-AUC | F1 |
|---|---|---|
| RevClassify_DS (SOTA, border Deep Sets) | 0.974 | 0.953 |
| **Ours (tuned border)** | **0.942** | **0.854** |
| GLASS | 0.816 | 0.705 |
NOT comparable (Elliptic2-paper Table 2, CPU, features ignored): GLASS F1 0.933 / PR-AUC 0.208.
**Verdict:** ours beats GLASS on both metrics; trails RevClassify_DS by ~0.03 PR-AUC (same
border-Deep-Sets architecture, reimplemented from scratch).
**Caveats (state honestly):** (a) DIFFERENT SPLIT — ours is our own stratified 80/10/10, not
RevTrack's exact split, so this is an approximate not identical-split comparison; (b) F1 is at a
fixed 0.5 threshold and swung across configs — PR-AUC (threshold-free) is the fair headline, and
there we're within 0.03.

## Discovery results
208 novel candidate subgraphs surfaced from the 49.3M-cluster background (top-500 by cluster score,
known members excluded). Held-out-recovery proxy eval: re-found **5 of 276** held-out test-suspicious
subgraphs = **1.8% recall**, random baseline **0.0001** → **lift 121×**. Interpretation: genuine
signal (121× > random) but low recall because only the top ~500 candidates were surfaced.

## Repo
~50 source modules under `src/ellip2/`, 18 CLIs under `scripts/`, **289 tests** (grows with the
doc tests). Gate: `bash scripts/verify.sh` = pytest + ruff + mypy (via `.venv`). venv/uv only, no
system pip. License for CODE: **MIT**. Attribution: git identity `owgreen-dev`
(`ogreenowow@gmail.com`) — adjustable.

## Limitations (for README/RESULTS "Limitations" section)
- Discovery recall is low (1.8%, top-500 only); raising top-K trades compute for recall.
- Single-split evaluation — no multi-seed/fold variance bars yet.
- Baseline comparison uses a different split than RevTrack (approximate).
- Discovery per-candidate BFS is still O(nnz)-bound (a full-matrix SpMV per level) — a perf
  follow-up; the transpose-reuse fix took the real run 49min → ~17min.
- Discovery carves can overlap known labeled clusters (seeds are novel; 3-hop carves traverse
  labeled territory — 154 clusters in the run).
- The per-cluster suspicion scorer that drives discovery is weak (test-member PR-AUC 0.127).
- LLM typologies are unvalidated (no ground-truth typology labels in the dataset).
