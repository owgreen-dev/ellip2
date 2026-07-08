#!/usr/bin/env python3
"""Keyless end-to-end demo: synthetic toy graph -> real detector -> investigative card.

Runs the actual Stage-2 (border model) and Stage-4 (typology + card) pipeline on a small,
deterministic synthetic graph so a reviewer gets a working experience in well under two
minutes with **zero setup** — no Kaggle download, no AWS/Bedrock, no GPU. The typology
classifier is the offline deterministic heuristic (the same default as
`ellip2.report.investigate`), so nothing here touches the network.

What it exercises (the real modules, not stubs):
  1. synthesize a labeled toy graph whose border *senders* carry the class signal, and whose
     suspicious subgraphs fan in -> hub -> fan out to receivers and a shared sink (the shape
     the detector keys on);
  2. `ellip2.pu.train_border` — train the border Deep-Sets model (best-of-3 val restart,
     mirroring the shipped default that mitigates the 1-in-5 training collapse);
  3. `ellip2.pu.score_border` — score held-out subgraphs;
  4. `ellip2.report.investigate` — render full investigative cards (border graph PNG +
     structural typology + exit-path corroboration + the mandatory false-positive caveat).

What it does NOT cover: Stage-0 DuckDB ingest and the 49.3M-cluster background *discovery*
run — those are the GPU-scale steps documented in RUNBOOK.md. This demo is the detection +
card surface, on data small enough to run anywhere.

Usage:
    make demo            # or: .venv/bin/python scripts/demo.py
Outputs land in ./demo_out/ (git-ignored): cards/ (card_*.md + card_*.png + index.md).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless matplotlib so the card PNGs render without a display.
os.environ.setdefault("MPLBACKEND", "Agg")

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import csv  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from ellip2.data import schema  # noqa: E402
from ellip2.pu import score_border, train_border  # noqa: E402
from ellip2.report.investigate import heuristic_classifier, investigate  # noqa: E402

N_SUB = 80          # 40 suspicious / 40 licit
SIGNAL = 3.0        # feature shift on border senders that encodes the label
SEED = 0


def build_synthetic(work: Path) -> dict[str, Path]:
    """Write ingest artifacts (edge_index, node_features, subgraphs) + a split + endpoints.

    Suspicious subgraphs: several senders funnel into two mutually-linked hub internals that
    fan out to receivers and a shared sink (fan-in/fan-out layering — a `nested_service`
    shape). Licit subgraphs: a single linear pass-through. Border *senders* carry the class
    signal (+SIGNAL for suspicious, -SIGNAL for licit) so the border model can separate.
    """
    rng = np.random.default_rng(SEED)
    src: list[int] = []
    dst: list[int] = []
    sender_shifts: list[tuple[int, float]] = []
    members_per_sub: list[list[int]] = []
    cc_ids: list[str] = []
    cc_labels: list[str] = []
    splits: list[str] = []
    endpoints: list[int] = []

    counter = 0

    def new_node() -> int:
        nonlocal counter
        node = counter
        counter += 1
        return node

    for k in range(N_SUB):
        suspicious = k % 2 == 0
        r = k % 10
        split = "train" if r < 6 else ("val" if r < 8 else "test")  # 60/20/20, both classes
        if suspicious:
            h1, h2 = new_node(), new_node()          # two hub internals
            for hub, n_send in ((h1, 3), (h2, 2)):   # senders funnel into the hubs
                for _ in range(n_send):
                    s = new_node()
                    src.append(s)
                    dst.append(hub)
                    sender_shifts.append((s, +SIGNAL))
            src += [h1, h2]                           # mutual hub cycle (obfuscation depth)
            dst += [h2, h1]
            for hub, n_recv in ((h1, 3), (h2, 2)):   # fan out to receivers
                for _ in range(n_recv):
                    rcv = new_node()
                    src.append(hub)
                    dst.append(rcv)
            sink = new_node()                         # both hubs converge on a shared sink
            src += [h1, h2]
            dst += [sink, sink]
            endpoints.append(sink)
            members = [h1, h2]
        else:
            m = new_node()
            s = new_node()
            src.append(s)                             # one sender -> internal -> one receiver
            dst.append(m)
            sender_shifts.append((s, -SIGNAL))
            rcv = new_node()
            src.append(m)
            dst.append(rcv)
            endpoints.append(rcv)
            members = [m]
        members_per_sub.append(members)
        cc_ids.append(f"bg{k}")     # bg-style ids so cards read card_NNN_ccbg… like the real leads
        cc_labels.append("suspicious" if suspicious else "licit")
        splits.append(split)

    n_nodes = counter
    feats = rng.standard_normal((n_nodes, schema.N_NODE_FEATURES)).astype(np.float32)
    for node, shift in sender_shifts:
        feats[node] += shift

    work.mkdir(parents=True, exist_ok=True)
    ingest = work / "ingest"
    ingest.mkdir(exist_ok=True)
    np.save(ingest / "edge_index.npy", np.array([src, dst], dtype=np.int64))
    np.save(ingest / "node_features.npy", feats)
    np.save(work / "endpoints.npy", np.array(sorted(set(endpoints)), dtype=np.int64))
    pq.write_table(
        pa.table({
            "ccId": pa.array(cc_ids),
            "ccLabel": pa.array(cc_labels),
            "n_members": pa.array([len(m) for m in members_per_sub], type=pa.int64()),
            "member_idx": pa.array(members_per_sub, type=pa.list_(pa.int64())),
        }),
        ingest / "subgraphs.parquet",
    )
    split_csv = work / "split.csv"
    with open(split_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "label", "split"])
        for cc, label, split in zip(cc_ids, cc_labels, splits, strict=True):
            w.writerow([cc, label, split])

    return {
        "ingest": ingest,
        "subgraphs": ingest / "subgraphs.parquet",
        "edge_index": ingest / "edge_index.npy",
        "node_features": ingest / "node_features.npy",
        "split": split_csv,
        "endpoints": work / "endpoints.npy",
    }


def main() -> int:
    out = _ROOT / "demo_out"
    work = out / "_work"
    cards = out / "cards"
    print("[demo] 1/4  synthesizing a labeled toy graph "
          f"({N_SUB} subgraphs, no download)…", flush=True)
    art = build_synthetic(work)

    model = work / "border.pt"
    print("[demo] 2/4  training the border Deep-Sets detector "
          "(best-of-3 val restart, CPU)…", flush=True)
    rc = train_border.main([
        "--artifacts-dir", str(art["ingest"]), "--subgraphs", str(art["subgraphs"]),
        "--split-csv", str(art["split"]), "--out", str(model),
        "--epochs", "120", "--set-hidden", "16", "--set-out", "8", "--border-cap", "16",
        "--restarts", "3", "--val-split", "val",
    ])
    if rc != 0:
        print("[demo] border training failed", file=sys.stderr)
        return rc

    scores = work / "border_scores.parquet"
    print("[demo] 3/4  scoring held-out subgraphs…", flush=True)
    rc = score_border.main([
        "--model", str(model), "--artifacts-dir", str(art["ingest"]),
        "--subgraphs", str(art["subgraphs"]), "--out", str(scores),
        "--split-csv", str(art["split"]), "--eval-split", "test",
    ])
    if rc != 0:
        print("[demo] scoring failed", file=sys.stderr)
        return rc

    print("[demo] 4/4  rendering investigative cards (offline heuristic typology)…",
          flush=True)
    leads = investigate(
        scores, art["subgraphs"], art["edge_index"], art["node_features"], cards,
        classifier=heuristic_classifier, split="test", top_k=5,
        endpoints_path=art["endpoints"], max_hops=6,
    )

    rel = cards.relative_to(_ROOT)
    print("\n[demo] done — rendered "
          f"{len(leads)} investigative cards from the held-out split.")
    print(f"[demo] open  {rel}/index.md")
    if leads:
        top = leads[0]
        print(f"[demo] top lead: {top.cc_id}  score={top.score:.3f}  "
              f"-> {rel}/card_001_cc{top.cc_id}.md (+ .png)")  # stem prepends 'cc'
    print("[demo] (this is the detection + card surface; the 49.3M-cluster background "
          "discovery run is the GPU step — see RUNBOOK.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
