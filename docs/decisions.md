# Architecture decision records (ADRs)

Short, pre-written answers to the "why did you choose X?" questions this project invites.
Each records a decision that was *not* obvious, the alternative that lost, and the evidence
that settled it. Numbers trace back to [`facts.json`](../facts.json) /
[RESULTS.md](../RESULTS.md); the Stage-2 diagnostic is written up in full in
[docs/stage2-model-choice.md](stage2-model-choice.md).

---

## ADR-1 — Detect with a supervised subgraph **border** model, not a cluster-level nnPU GNN

**Decision.** The primary detector is a supervised **border Deep-Sets** classifier over each
candidate subgraph's external *senders* and *receivers* (plus pooled internal node/edge
features), under weighted BCE — `ellip2.pu.train_border`. The cluster-level GNN + nnPU scorer
(`ellip2.pu.train`) is kept only as a secondary path that feeds discovery.

**Why not the GNN we built first?** The GNN was the fastest route to an end-to-end run (it
natively emits one score per cluster, which discovery needs), but on the real graph it scored
**PR-AUC 0.030** — barely above the 2.27% base rate. A diagnostic proved the ceiling was the
*framing*, not the features: a plain gradient-boosted model on the *same* pooled features hit
~10× that, and the border framing reached **0.911 ± 0.009**.

**Why the border, specifically?** Elliptic2's labels are at the **subgraph** level, and a
subgraph is suspicious because of *who funds it and who it pays* — the border — not its
internal graphlet shape (licit and suspicious internals are nearly identical). A per-cluster
scorer structurally cannot see the border; forcing cluster labels also injects severe
positive-label noise (most members of a suspicious subgraph are benign). This mirrors what
every Elliptic2 SOTA (RevClassify, GLASS) actually does. Full diagnostic:
[stage2-model-choice.md](stage2-model-choice.md).

---

## ADR-2 — Select the trained model by **best-of-N validation restart**, not a single run

**Decision.** `train_border --restarts N --val-split val` trains N independent restarts and
keeps the one with the best validation PR-AUC. The reported **0.911 ± 0.009** uses best-of-3.

**Why.** A *single* border-model run is unstable: about **1 in 5** fresh splits collapses to a
degenerate all-positive solution (~PR-AUC 0.38). Reporting one lucky run would be dishonest and
one unlucky run would understate the method. Validation-based selection is a cheap, standard
mitigation that turns an unstable estimator into a reproducible one, and reporting the mean ±
std across 5 splits (rather than a single best) states the real variance. The alternative —
publishing the single best split (0.942) as *the* number — is available in `facts.json` but is
explicitly labelled "best single split," not the headline.

---

## ADR-3 — Ingest out-of-core with **DuckDB**, not in-memory pandas/NumPy

**Decision.** Stage 0 (`ellip2.data`) streams the five raw CSVs through DuckDB into compact
artifacts (`edge_index.npy`, `node_features.npy`, `subgraphs.parquet`).

**Why.** The dataset is **~83 GB extracted** with **49,299,864** background clusters and
**196,215,606** background edges — it does not fit in memory on any reasonable box. DuckDB gives
out-of-core joins/aggregations with a SQL surface and bounded memory, so ingest runs on a normal
instance and only the model steps need a GPU. Loading the raw frames into pandas was never an
option at this scale; a bespoke chunked reader would have re-implemented a slower, buggier DuckDB.

---

## ADR-4 — Discover with a **cheap-gate → carve → expensive-rescore** funnel

**Decision.** Background discovery does not run the border model on all ~48.8M clusters. It
scores every cluster with a cheap per-cluster HGBM, passes only the top candidates through a
3-gate funnel, carves a bounded ≤k-hop neighborhood around each survivor, and only then pays for
a border re-score — producing ranked novel leads.

**Why.** The border model needs a subgraph's full border; assembling and scoring that for tens
of millions of clusters is infeasible. A cascade spends compute where it matters: the weak-but-
cheap cluster scorer (PR-AUC **0.127**) is good enough to *rank* candidates for triage, and the
expensive, accurate border model only adjudicates the survivors. The cost is recall — only the
top ~500 candidates were surfaced, giving **1.8%** held-out recovery (**5 of 276**), but at
**121×** lift over random, which is genuine signal for a lead-generation tool. Raising top-K
trades compute for recall (see [Limitations](../README.md#limitations)).

---

## ADR-5 — Compare against a named baseline table, stated as an **approximate different-split**

**Decision.** RESULTS.md compares against **RevTrack Table 1 (GPU + node features)** by name,
and states plainly that ours is a *different-split, approximate* comparison — not identical-split.

**Why.** RevTrack publishes no split seed, so an identical-split comparison is impossible to
reproduce honestly. Naming the exact table (rather than a vague "beats prior work") lets a reader
check the claim, and flagging the split difference up front is more credible than an
apples-to-apples claim we cannot back. On the threshold-free headline metric we beat GLASS
(0.816) and trail SOTA RevClassify_DS (0.974) by ~0.06 with the *same* architecture reimplemented
from scratch — which is the honest, defensible story.

---

## ADR-6 — Treat exit paths and LLM typologies as **unvalidated corroboration**, human-in-the-loop

**Decision.** Every lead is rendered as an investigative card, but the exit-path endpoints are
*heuristic* and the typology label is produced by a classifier that is **injectable and offline
by default** (`ellip2.report.investigate` uses a deterministic structural classifier; Bedrock
Claude is opt-in). Cards carry a mandatory "this is a LEAD, not a finding" caveat.

**Why.** The dataset ships **no** ground-truth entity labels, licit-endpoint labels, or typology
labels, so neither the exit path nor the typology can be *validated* — presenting them as facts
would be overclaiming. Making the LLM optional keeps the whole pipeline runnable with zero
credentials (the [`make demo`](../README.md#quickstart) path), keeps CI keyless, and keeps the
human reviewer — not the model — as the decision-maker. This is the same "surfaces leads, doesn't
establish wrongdoing" stance stated in the README disclaimer.
