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
| **border model, tuned** | **0.942** |

The **border model** is a Deep Sets network over the external *senders* and *receivers* of
a candidate subgraph, concatenated with its pooled internal node and edge features, then
classified by an MLP under weighted BCE.

Tuned border model, full metrics: **PR-AUC 0.942, F1 0.854, recall 0.935**.
Tuned config: border-cap 128, set-hidden 128, set-out 64, mlp-hidden [128, 64], epochs
150, lr 0.01, weight-decay 1e-4. (The node-only variant used for discovery scoring reaches
PR-AUC 0.80.)

### Baseline comparison — RevTrack Table 1 (GPU + node features)

The comparable published setting is **RevTrack Table 1 (GPU + node features)** from
*Identifying Money Laundering Subgraphs on the Blockchain* (arXiv:2410.08394):

| Model | PR-AUC | F1 |
|---|---|---|
| RevClassify_DS (SOTA, border Deep Sets) | 0.974 | 0.953 |
| **Ours (tuned border)** | **0.942** | **0.854** |
| GLASS | 0.816 | 0.705 |

**Verdict:** ours beats **GLASS** on both metrics and trails **RevClassify_DS** by ~0.03
PR-AUC — the same border-Deep-Sets architecture, reimplemented from scratch.

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
- **F1 is at a fixed 0.5 threshold** and swung across configs — PR-AUC (threshold-free) is
  the fair headline, and there we're within ~0.03 of SOTA.
- **Single-split evaluation** — no multi-seed / multi-fold variance bars yet.
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
