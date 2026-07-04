# REPRODUCE — end-to-end CLI sequence

The exact command sequence that takes the raw Elliptic2 CSVs to detection scores,
investigative cards, and ranked novel-discovery leads. Every step is a `scripts/*.py`
shim (or a `ellip2-*` console entry point) over a module in `src/ellip2/`. For the AWS
provisioning wrapper around these (image build, EC2 launch, S3 sync) see `RUNBOOK.md`;
this file is the pipeline order and flags.

All commands assume `.venv` is active and `$OUT` is the ingest/artifacts directory:

```bash
source .venv/bin/activate      # venv/uv only — no system pip (see README Quickstart)
export RAW=/mnt/data/elliptic2/raw          # the 5 downloaded CSVs
export OUT=/mnt/data/elliptic2/artifacts    # all pipeline artifacts land here
export SPLIT=$OUT/splits/stratified_random/split.csv
```

## Stage 0 — Ingest & split

```bash
# CSVs -> id_map.parquet, node_features.npy, edge_index.npy, subgraphs.parquet
ellip2-ingest                       --raw-dir $RAW --out-dir $OUT --memory-limit 12GB

# persisted, reused-everywhere train/val/test split at the subgraph level
python scripts/make_split.py        --input $RAW/connected_components.csv --out-dir $OUT/splits
```

## Stage 1 — Features

```bash
# per-cluster degree / edge-agg / flow-concentration / neighborhood / temporal / path-role
python scripts/build_features.py    --artifacts-dir $OUT --raw-dir $RAW --split-csv $SPLIT
# -> $OUT/cluster_features.parquet
```

## Stage 2 — Border detection model (the headline detector)

```bash
# Deep Sets over external senders + receivers + pooled internal feats -> MLP, weighted BCE
python scripts/train_border.py      --artifacts-dir $OUT --subgraphs $OUT/subgraphs.parquet \
                                    --split-csv $SPLIT --out $OUT/border_model.pt
python scripts/score_border.py      --model $OUT/border_model.pt --artifacts-dir $OUT \
                                    --subgraphs $OUT/subgraphs.parquet --split-csv $SPLIT \
                                    --out $OUT/border_scores.parquet
```

## Stage 3 — Exit-path endpoints & investigative cards

```bash
# heuristic licit-receiver endpoint set from the path-role feature
python scripts/make_endpoints.py    --features $OUT/cluster_features.parquet \
                                    --out $OUT/endpoints.npy --percentile 0.99

# per-candidate cards: reachability exit paths + LangGraph typology agent (Bedrock optional)
python scripts/investigate.py       --border-scores $OUT/border_scores.parquet \
                                    --subgraphs $OUT/subgraphs.parquet \
                                    --edge-index $OUT/edge_index.npy \
                                    --node-features $OUT/node_features.npy \
                                    --endpoints $OUT/endpoints.npy \
                                    --out-dir $OUT/cards --top-k 50
```

## Background discovery — novel suspicious subgraphs among the 48.8M unlabeled clusters

```bash
# Gate-1 per-cluster suspicion scorer (HGBM) over all clusters
python scripts/train_cluster.py     --features $OUT/cluster_features.parquet \
                                    --subgraphs $OUT/subgraphs.parquet --split-csv $SPLIT \
                                    --out $OUT/cluster_model.pkl
python scripts/score_cluster.py     --model $OUT/cluster_model.pkl \
                                    --features $OUT/cluster_features.parquet \
                                    --out $OUT/cluster_scores.npy

# Gate-3 typology structural signal (source/sink axis)
python scripts/make_typology_signal.py --features $OUT/cluster_features.parquet \
                                    --out $OUT/typology_signal.npy

# 3-gate funnel (score + reachability carve + typology) -> border-score -> ranked novel leads
python scripts/discover_subgraphs.py --scores $OUT/cluster_scores.npy \
                                    --edge-index $OUT/edge_index.npy \
                                    --node-features $OUT/node_features.npy \
                                    --endpoints $OUT/endpoints.npy \
                                    --model $OUT/border_model.pt \
                                    --subgraphs $OUT/subgraphs.parquet \
                                    --split-csv $SPLIT --exclude-split train \
                                    --typology-signal $OUT/typology_signal.npy \
                                    --out-subgraphs $OUT/discovered_subgraphs.parquet \
                                    --out-scores $OUT/discovered_scores.parquet

# held-out-recovery proxy eval: how many held-out test-suspicious subgraphs did we re-find?
python scripts/eval_recovery.py     --discovered $OUT/discovered_subgraphs.parquet \
                                    --subgraphs $OUT/subgraphs.parquet \
                                    --split-csv $SPLIT --eval-split test --n-nodes 49299864
```

## AWS instance & cost note

The real run is CPU-heavy for Stages 0–1 (DuckDB out-of-core) and GPU-friendly for the
border model. A single **g5.xlarge** (1× A10G, 16 GiB RAM) on the Deep Learning AMI runs
the whole pipeline; ingest + features can be moved to a cheaper CPU box (e.g.
`c7i.2xlarge`) to save GPU hours. Rough cost: g5.xlarge ≈ **$1.00/hr on-demand
(~$0.40/hr spot)**, gp3 150 GB ≈ **$8/mo**, S3 for the ~24 GB compressed raw ≈ **$0.60/mo**
— budget a few dollars for one full run plus teardown. The g5.xlarge has only **16 GiB
RAM**, so always pass `--memory-limit` to ingest and never materialize the CSVs in pandas.
See `RUNBOOK.md` for the full image-build / launch / S3-sync wrapper.
