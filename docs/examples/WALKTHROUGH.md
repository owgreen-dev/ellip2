# Walkthrough — reading one discovered laundering subgraph like an investigator

This is a single lead, narrated end to end: `ccbg25364052`, a subgraph the pipeline
**discovered** in the unlabeled background (it was never in the labeled set) and scored **1.0**.
It's the hero card on the [README](../../README.md); here we walk it the way an analyst would,
one signal at a time, and — importantly — show *where the model was overruled*.

The raw card is [`card_005_ccbg25364052.md`](card_005_ccbg25364052.md); the graph:

![Discovered fraud subgraph bg25364052 — senders (orange) fund an internal cluster (blue) that pays out to receivers (green)](card_005_ccbg25364052.png)

> **This is an investigative lead, not a finding or an accusation.** Everything below is
> derived from public/benchmark data and heuristics; it requires human review before any action.

---

## 0. Why this one surfaced at all

It was never labeled. Discovery scored all ~48.8M background clusters with the cheap per-cluster
model, carved a bounded neighborhood around the top candidates, and re-scored each survivor's
full **border** with the detector. This subgraph came back at **PU score 0.9999 (98.1st
percentile)** — top of the queue. So the first fact isn't "it's fraud," it's "of everything
nobody had looked at, the detector is most confident this one has the *shape* of laundering."
That's the whole job of the tool: move the analyst's eye to the right subgraph.

## 1. The border: who funds it, who it pays

The detector keys on the **border** — the external senders and receivers — because that's where
licit and suspicious subgraphs differ (their internal shapes are nearly identical). Here the
structural evidence reads:

- **10 senders** funnel *in*
- **2 internal hub nodes** (`4514246`, `4072593`) sit in the middle
- **10 receivers** are paid *out*, across **51 edges** over just 15 internal nodes

Ten independent sources converging on two hubs is the first tell: **fan-in** consistent with
structuring / smurfing — many small parallel inflows rather than one counterparty.

## 2. The hubs: fan-out and a deliberate loop

`max_in_degree = 14`, `max_out_degree = 16`. The two hubs don't just pass funds through — they
**distribute**: node `4514246` fans out to 16 distinct downstream nodes at once. And the two hubs
feed *each other* (`4514246 ⇄ 4072593`), a mutual cycle that does no economic work but adds
**layering depth** — hops whose only purpose is to lengthen and obscure the trail. Fan-in to a
hub, a loop between hubs, then fan-out is the classic layering signature.

## 3. The intermediaries: fresh, low-history addresses

The 15 internal nodes are mostly *zero-binned* on their features — the fingerprint of freshly
created, low-activity addresses used once and abandoned. Real pass-through businesses accrete
history; layering intermediaries don't. Several of them (`22906283`, `22906284`, `23061436`,
`23451057`, …) each split to **both** the shared sink and a unique secondary receiver — a textbook
smurfing fan-out.

## 4. The sink: convergence after dispersion

After all that dispersion, the flows **re-converge**: a single receiver node, `5383`, collects
from at least 12 different internal nodes. Disperse-then-reconverge is the shape that distinguishes
laundering from ordinary many-to-many payment traffic.

## 5. Corroboration: an exit path to a licit-looking endpoint

Stage 3 asks a separate question — *can funds actually reach cash-out?* — by graph traversal, not
by the model. It found a bounded reachable path to a heuristic licit endpoint:

```
4072593 -> 2350
```

A hub reaches a plausible off-ramp within the hop budget. This is **corroboration, not proof**:
the endpoint type is a derived heuristic (the dataset ships no ground-truth endpoint labels), so
it raises confidence without establishing anything.

## 6. Where the model was overruled — the honest part

The LLM typology agent first called this **`layering_smurfing`** at 0.87 confidence — a
defensible read given the fan-in/fan-out. The **structural validator disagreed**: two persistent
high-degree hubs that *both* aggregate and redistribute (rather than a one-shot smurf-and-scatter)
are the signature of a **`nested_service`** — an intermediary operated *inside* another service's
address space. The card records the override explicitly:

> Typology: **nested_service** (confidence 0.87) — **FLAGGED (structural contradiction)**
> Validation: structural signals imply 'nested_service', contradicting model 'layering_smurfing';
> overridden to structural reading

Two things matter here. First, structure wins over the language model when they disagree —
deterministic signals over a free-text verdict. Second, the disagreement is **surfaced, not
hidden**: a flagged typology is a signal to the human reviewer to look harder, exactly the
posture a lead-generation tool should take. See
[ADR-6](../decisions.md#adr-6--treat-exit-paths-and-llm-typologies-as-unvalidated-corroboration-human-in-the-loop).

## 7. What the analyst does with it

The card hands a reviewer a ranked, evidence-backed starting point: a border graph, a typology
with its contradiction flagged, structural stats, and a corroborating exit path — plus the
standing caveats that the PU score is a lower bound, the roles are heuristics, and false positives
are expected. It says *look here first, and here's why*. It does **not** say "this is fraud." That
line — leads, not findings — is the entire discipline of the project.

---

*Want to generate cards like this yourself with no setup? Run [`make demo`](../../README.md#quickstart)
— it trains the detector and renders cards on a synthetic graph in under two minutes, no
credentials.*
