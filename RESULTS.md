# Results — ellip2 on Elliptic2

Metrics for the Elliptic2 Bitcoin AML pipeline. All numbers are copied verbatim from the
project's canonical facts (`plans/publishing_facts.md`) — do not edit them here in
isolation. See [README.md](README.md) for the project overview and pipeline.

## Detection

The detection task is subgraph-level binary classification (suspicious vs. licit) over the
121,810 labeled connected components, at a **2.27%** base rate (2,763 suspicious / 119,047
licit). Metrics below are **test PR-AUC on our own stratified 80/10/10 split**.

### Detection progression

Each row is a distinct modeling choice; PR-AUC climbs as the framing and features improve.

| Model | test PR-AUC |
|---|---|
| cluster-level nnPU GNN (rejected framing) | 0.030 |
| pooled-features HGBM | 0.286 |
| border model, nodes only | 0.816 |
| border model + internal edge features | 0.844 |
| **border model, tuned** | **0.911 ± 0.009** |

The **border model** is a Deep Sets network over the external *senders* and *receivers* of
a candidate subgraph, concatenated with its pooled internal node and edge features, then
classified by an MLP under weighted BCE.
Tuned config: border-cap 128, set-hidden 128, set-out 64, mlp-hidden [128, 64], epochs
150, lr 0.01, weight-decay 1e-4. (The node-only variant used for discovery scoring reaches
PR-AUC 0.80.)

### Robustness (why 0.911 ± 0.009, not 0.942)

The headline is the **mean ± std over 5 stratified 80/10/10 splits**, using **best-of-3
validation-based model selection** (`train_border --restarts 3 --val-split val`) — not a
single favorable split. The rigor pass that produced it found, and then fixed, a real
training instability:

| Setup | test PR-AUC (5 splits) |
|---|---|
| single run per split | **0.756 ± 0.213** — 1 of 5 splits **collapsed** to 0.375 (degenerate all-positive) |
| best-of-3 val selection | **0.911 ± 0.009** — no collapse (the previously-collapsed split recovers to 0.912) |

A single training run collapses ~1 in 5 times; a collapsed restart has low *validation*
PR-AUC, so best-of-N selection never keeps it. The best single split reaches **0.942**
(reproduced exactly), but the honest, stable number is **0.911 ± 0.009**. **F1** @ the 0.5
threshold is **0.78 ± 0.05**; tuning the decision threshold on validation lifts test F1 to
**~0.89** (PR-AUC is threshold-free and is the fair headline).

### Baseline comparison — RevTrack Table 1 (GPU + node features)

The comparable published setting is **RevTrack Table 1 (GPU + node features)** from
*Identifying Money Laundering Subgraphs on the Blockchain* (arXiv:2410.08394):

| Model | PR-AUC | F1 |
|---|---|---|
| RevClassify_DS (SOTA, border Deep Sets) | 0.974 | 0.953 |
| **Ours (border, best-of-3, 5-split)** | **0.911 ± 0.009** | 0.78 (0.89 val-tuned) |
| GLASS | 0.816 | 0.705 |

**Verdict:** ours beats **GLASS** and trails **RevClassify_DS** by ~0.06 PR-AUC — the same
border-Deep-Sets architecture, reimplemented from scratch. (Reporting the robust 5-split
mean; the best single split matches 0.942.)

> Not comparable: the Elliptic2-paper Table 2 (CPU, features ignored) reports GLASS F1
> 0.933 / PR-AUC 0.208 — a different setting, not used for this comparison.

## Discovery

Beyond detection, a background-discovery stage surfaces novel suspicious structures among
the ~48.8M *unlabeled* background clusters that were never labeled.

- **208 novel candidate subgraphs** surfaced from the 49.3M-cluster background (top-500 by
  per-cluster score, known labeled members excluded).
- **Held-out-recovery proxy eval:** re-found **5 of 276** held-out test-suspicious
  subgraphs = **1.8% recall**, against a random baseline of **0.0001** → **121× lift**.

**Interpretation:** genuine signal (121× above random) but low absolute recall, because
only the top ~500 candidates were surfaced. Raising the top-K trades compute for recall.

## Limitations & caveats

These qualify every number above; read them before quoting a headline metric.

- **Different split for the baseline comparison.** Ours is our own stratified 80/10/10
  split, **not** RevTrack's exact split, so the comparison above is *approximate*, not an
  identical-split head-to-head.
- **F1 is at a fixed 0.5 threshold** and swings across runs — PR-AUC (threshold-free) is
  the fair headline; a validation-tuned threshold lifts test F1 to ~0.89.
- **Training instability** — a single run collapses ~1/5 of the time; the reported number
  requires best-of-N validation-based selection (`--restarts`) to be reproducible.
- **Discovery recall is low** (1.8%, top-500 candidates only); raising top-K trades
  compute for recall.
- **The per-cluster suspicion scorer that drives discovery is weak** (test-member PR-AUC
  0.127).
- **Discovery carves can overlap known labeled clusters** — seeds are novel, but ≤k-hop
  carves traverse labeled territory (154 clusters in the run).
- **Discovery per-candidate BFS is still O(nnz)-bound** (a full-matrix SpMV per level); the
  transpose-reuse fix took the real run from 49 min → ~17 min, but it remains a perf
  follow-up.
- **LLM typologies are unvalidated** — the dataset has no ground-truth typology labels.

## Sources

- Elliptic2 dataset / GLASS — *The Shape of Money Laundering* — **arXiv:2404.19109**.
- RevClassify / RevTrack — *Identifying Money Laundering Subgraphs on the Blockchain*
  (ACM AIF 2024) — **arXiv:2410.08394**.
