# Stage 2 model choice: supervised subgraph classifier, not the cluster-level GNN

**Status:** decided 2026-07-03, from a diagnostic on the full real Elliptic2 graph.
**TL;DR:** The primary Stage-2 detector is a **supervised classifier over per-subgraph
pooled features** (`ellip2.pu.train_subgraph`, gradient-boosted trees). The cluster-level
**GNN + nnPU** scorer (`ellip2.pu.train`) is kept but demoted to a secondary/experimental
path. **We still use the graph** — just not as the classifier backbone (see below).

## Why we built the GNN path first
Stage 3 discovery (`discover.py`) consumes one score **per cluster** (`(N,)`), and the
only framing that natively produces per-cluster scores is the cluster-level nnPU scorer
(plan.md decision #2). It also mapped cleanly onto the ingest artifacts
(`node_features` / `edge_index` / `subgraphs`) without extra data assembly. So it was the
fastest path to an end-to-end run — and it *did* run end-to-end once the scale bugs were
fixed (hub-exclusion, bounded DuckDB, minibatch sampling, standardization).

## Why we moved off it
On real data the GNN scorer is barely better than chance, and a **diagnostic proved the
ceiling is the framing, not the features.** Pooling each subgraph's member
`cluster_features` (mean/max/min/std) and training a plain classifier vs the
suspicious/licit label gives (test PR-AUC, base rate 0.0227):

| Model | Test PR-AUC | × base rate |
|-------|-------------|-------------|
| GNN cluster-nnPU (`train.py`) | 0.030 | 1.3× |
| Logistic regression (pooled) | 0.151 | 6.6× |
| MLP (pooled) | 0.203 | 9× |
| **Gradient-boosted trees (pooled)** | **0.305** | **13×** |

A trivial supervised model on the *same* features beats the GNN **~10×**. The features
have strong subgraph-level signal; the cluster-nnPU framing throws it away. Three reasons:

1. **Label granularity.** Real labels are at the **subgraph** level. Training a cluster
   scorer requires marking *every* member of a suspicious subgraph as positive — but a
   subgraph is suspicious because of *one* illicit→licit path, so most member clusters are
   benign. That is severe positive-label noise. plan.md decision #1 is explicit: at the
   subgraph level the licit labels are bona-fide negatives, so this is imbalanced
   **supervised** classification, "which is what every Elliptic2 SOTA actually does."
2. **Where the signal lives.** RevClassify beats GLASS because the discriminative signal is
   in the subgraph **border** (who funds it / who it pays), not internal cluster shape —
   licit vs suspicious internal graphlet distributions are nearly identical (plan.md
   §"Subgraph-level readout"). A per-cluster scorer can't see the border.
3. **Cluster-level PU is genuinely hard** — tiny/uncertain prior, heterophily. Our prior
   sweep confirmed it: PR-AUC was flat (~0.030) across π_p ∈ [1e-3, 5e-2], i.e. not
   prior-limited. The framing was the wall.

## What the primary model is now
`ellip2.pu.subgraph_pool` → `train_subgraph` (HistGradientBoosting) → `score_subgraph`,
producing `subgraph_scores.parquet` (per-subgraph detection scores — the deliverable that
validates against RevClassify's held-out numbers). Order-statistic pooling
(mean/max/min/std) already captures the size-invariant readout the plan calls for.

**Next lift (Path A, plan.md decision #1 default):** add the **border** signal — Deep Sets
over sender/receiver border nodes + internal edge features from `edges.csv` — via the
existing `SupervisedSubgraphModel` in `ellip2.pu.trainer`. That needs a data-assembly layer
(border extraction from `edge_index` + subgraph membership) and is expected to push past
0.30.

## We still use the graph (this is not "drop graphs")
The graph is central to the pipeline; it is simply not the classifier's backbone:

- **Features:** the pooled inputs already include graph-derived features — degree, flow
  concentration, neighbor/two-hop label fractions, path-role — computed over the real graph
  (`ellip2.features.*`). The classifier eats graph structure, just pre-digested.
- **Border model (next):** sender/receiver sets are a graph computation on `edge_index`.
- **Stage 3 discovery:** exit-path search (`exit_paths.path_search`) is pure graph traversal
  over `edge_index` — reachability from flagged subgraphs to licit endpoints.
- **Visualization:** flagged subgraphs, their border nodes, and corroborating exit paths are
  meant to be **rendered as graphs** in the investigative report (`ellip2.report.render`).
  `edge_index.npy` / `ellip2.graph.pyg_data` remain the substrate for that. Model choice
  (trees vs GNN) is orthogonal to graph visualization — we keep the graph artifacts and the
  PyG data builder precisely so the leads can be shown as graphs.
