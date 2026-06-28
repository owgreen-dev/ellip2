Claude Code Implementation Plan: Bitcoin AML Research on Elliptic2 — Subgraph-Level Detection (Imbalanced Supervised) with Optional Cluster-Level PU Learning
TL;DR

This is buildable as specified, but the single most important design decision is to not run a GNN over the full 49M-node graph for scoring; instead precompute behavioral + structural fingerprints per cluster (DuckDB/Polars), train an nnPU classifier (optionally with a sampled-subgraph GNN encoder), and reserve PyG NeighborLoader sampling for the candidate set only — this fits on a single g5.xlarge (24GB A10G) if you keep heavy aggregation out-of-core (see the RAM caveat).
The Elliptic2 schema is concrete: 5 CSVs — background_nodes.csv [id + 43 features], background_edges.csv [clId1,clId2 + 95 features], connected_components.csv [ccId + ccLabel], nodes.csv [node→ccId membership], edges.csv [subgraph edges]; 49,299,864 nodes, 196,215,606 edges, 121,810 labeled subgraphs of which 2,763 are suspicious (vs. 119,047 licit) per arXiv:2404.19109 Table 1 — the suspicious set is your "positives," a base rate of 2.27% (2,763 of 121,810).
The hardest correctness risks are (1) graph-PU heterophily breaking class-prior estimation, (2) data-leakage from labeled-subgraph nodes bleeding into background features, and (3) overstating "newly discovered subgraphs." Mitigate with leakage-safe feature construction, a held-out positive split for recall estimation, and RevFilter-style corroboration before any discovery claim.

Resolved design decisions (reconciles the four open issues)
1. Unit = the labeled subgraph (connected component) — "Path A". Train and evaluate one binary label per subgraph, matching the SOTA comparators (RevClassify, GLASS, GNN-Seg, Sub2Vec) apples-to-apples. If you score individual clusters, MAX-POOL member-cluster suspicious-probabilities to the subgraph — this is the multiple-instance-learning rule ("bag positive iff ≥1 positive instance," equivalently noisy-OR; use log-sum-exp if pooling end-to-end) and matches how Elliptic2 subgraphs were built (a component is suspicious if it contains one illicit→licit path). Median subgraph is 3 nodes, so max-pool is the right default; attention-pool offers little headroom. Persist explicit train/val/test index files — do NOT rely on preprocess_glass.py's split: it is a deterministic modulo-10 round-robin (counter%10<=7→train, ==8→val, else test), NOT a seeded shuffle, despite the README's "fixed given the random seeds" wording; the Elliptic2 paper and RevTrack both describe a *random* 80:10:10, so reproducing published splits means matching each repo's mechanism. Pick one split, persist it, reuse across all models.
2. This is imbalanced SUPERVISED learning by default, not PU. With reliable licit = negative at the subgraph level, use weighted BCE / focal loss with PR-AUC model selection — that is exactly what every Elliptic2 method does; none uses a PU risk estimator. The 2.27% (2,763/121,810) figure is the SUBGRAPH base rate, NOT a cluster-level prior. Reserve genuine nnPU for a CLUSTER-level scorer, where the ~49M unlabeled clusters are a true unlabeled set and π_p is small (~10⁻³ or less). Recommended hybrid: cluster-level nnPU scorer → max-pool → evaluate at subgraph level against RevClassify; use the reliable licit labels for evaluation/calibration, not as the unlabeled set. State which framing each experiment uses.
3. Endpoint = the labeled licit receiver/sink already inside the suspicious subgraph (RevTrack's R set) — the dataset ships NO node labels and NO entity-type labels (paper Fig. 3: "the label categories are not available in the dataset"). So you do not invent the endpoint; what you must DERIVE is the entity type (exchange vs other service), as an explicitly-flagged structural heuristic: high in/out-degree + large cluster size (#addresses) + high throughput + address reuse, calibrated empirically on the anonymized 43-d features (you cannot assume which dimension is degree/size/throughput without inspecting distributions).
4. ≤6-hop search = reachability, not enumeration. Run backward multi-source BFS from the (smaller) endpoint set on the transposed adjacency (boolean SpMV matrix powers or cuGraph BFS via a virtual super-source), with per-level frontier caps and hub nodes EXCLUDED from pass-through (stop at them — an exchange is an endpoint, not transit, matching Elliptic2's "stop at labeled node / change-of-ownership" construction). Intersect with capped forward BFS from candidate sources; extract induced subgraphs only on survivors (PyG k_hop_subgraph, flow="target_to_source"). Average degree ≈3.98 so depth-6 is ~4k nodes in expectation — but heavy-tailed hubs (in-degree in the millions) explode an uncapped frontier, hence the caps.

Key Findings
1. Elliptic2 data structure

Source of truth: paper arXiv:2404.19109 ("The Shape of Money Laundering"), GitHub MITIBMxGraph/Elliptic2, Kaggle ellipticco/elliptic2-data-set. Download is ~26GB unzipped into 5 CSVs. The Hacker News
Exact counts (paper Table 1): 49,299,864 nodes; 196,215,606 edges; 121,810 labeled subgraphs; 43 node features; 95 edge features; 2 classes. Licit subgraphs = 119,047 (avg 3.65 nodes, median 3, max 296); Suspicious = 2,763 (avg 3.79 nodes, median 3, max 30). (RevTrack's paper reports 119,092 licit / 2,718 suspicious — minor version differences; use the split shipped with the data you download.) arXivarXiv
Schema (confirmed from preprocess_glass.py in the Elliptic2 repo):

background_nodes.csv: column 0 = cluster id; columns 1: = 43 node features. Feature columns are anonymized/binned ordinals — names are NOT published; treat as feat_0..feat_42. Documented examples: node size (number of addresses), number of transactions.
background_edges.csv: source/destination columns literally named clId1, clId2, plus 95 edge features. Edge features include transaction volume, fee, timestamp (binned for IP reasons).
connected_components.csv: column 0 = ccId (subgraph id); label column literally named ccLabel holding the class string ("licit"/"suspicious").
nodes.csv: column 0 = node id, column 1 = ccId — the node→subgraph membership list.
edges.csv: the labeled intra-subgraph edge list (same clId1/clId2 convention).


Labels exist ONLY at subgraph level; the vast majority of the 49M nodes are unlabeled. This is a PU setup ONLY at the cluster level (suspicious-subgraph members = positives, the ~49M others = unlabeled). At the SUBGRAPH level, the reliable licit labels are bona fide negatives, so the task is imbalanced supervised classification (weighted BCE) — which is what every Elliptic2 SOTA actually does; see Resolved design decision #2. arXiv
Loading 49M/196M without OOM: do NOT load CSVs with pandas. Use DuckDB or Polars to read/aggregate, store the edge list as a binary CSR (int32 edge_index .npy or torch tensor). 196M edges × 2 × int32 ≈ 1.6GB; 49M × 43 features as float16 ≈ 4.2GB. Avoid materializing all 95 edge features for all 196M edges in RAM simultaneously (≈37GB float16) — stream/aggregate them.
PyG structures: use homogeneous Data (single node type "cluster," single edge type "transaction"). HeteroData is unnecessary. Use NeighborLoader for mini-batch sampling on the candidate set. RevTrack avoids GNN-over-full-graph entirely by tracking senders/receivers and using precomputed node embeddings (raw_emb.pt, served separately on Google Drive). PyTorch GeometricGitHub

2. PU learning on graphs

nnPU (Kiryo, Niu, du Plessis & Sugiyama, NeurIPS 2017, arXiv:1703.00593): a non-negative risk estimator. In the authors' words, with unbiased PU and a flexible model "empirical risks on training data will go negative, and we will suffer from serious overfitting. In this paper, we propose a non-negative risk estimator for PU learning … more robust against overfitting, and thus we are able to use very flexible models (such as deep neural networks) given limited P data." Risk = π_p·E_P[ℓ+] + max{0, E_U[ℓ−] − π_p·E_P[ℓ−]}. PyTorch implementations to reuse: kiryor/nnPUlearning, cimeister/pu-learning. arxiv
Class-prior (π_p) estimation is the crux. Standard methods (du Plessis/Niu/Sugiyama penalized-L1/KM; Garg et al. mixture-proportion estimation) assume "irreducibility."
Graph-PU heterophily problem — Wu, Yao, Han, Yao & Liu, "Unraveling the Impact of Heterophilic Structures on Graph Positive-Unlabeled Learning" (ICML 2024, arXiv:2405.19919): "a critical challenge for PU learning on graph lies on the edge heterophily, which directly violates the irreducibility assumption for Class-Prior Estimation … and degenerates the latent label inference on unlabeled nodes." Heterophily yields an overestimated class prior and entangles positive/negative latent features. AML graphs are intrinsically heterophilic — a laundering path connects illicit→licit by construction. Their fix (GPL: Graph PU Learning with Label Propagation Loss) is a bilevel optimization that "reduces heterophily in the inner loop and efficiently learns a classifier in the outer loop."
Practical implications: (a) prefer subgraph-level classification (RevTrack's framing) where the "shape" is more homophilous than node-level; (b) the 2.27% figure is the SUBGRAPH base rate (2,763/121,810), not a cluster-level prior — at the cluster level π_p is far smaller (~10⁻³ or less); if doing cluster-level PU, estimate π_p with KM2/TIcE, clamp to a small range, and sweep 1e−1…1e−5 in BOTH directions (under- vs over-estimation sensitivity is dataset-dependent per Kiryo 2017 vs imbalanced-PU follow-ups), using reliable licit subgraphs as known negatives for validation; (c) use a heterophily-tolerant GNN encoder (include ego features, separate ego vs. neighbor aggregation à la GraphSAGE; avoid pure GCN smoothing).
Combining GNN + PU head: GNN encoder (GraphSAGE/GIN on sampled subgraphs) → pooled subgraph embedding → linear head trained with nnPU loss. Alternatively decouple (like RevTrack's raw_emb.pt): precompute embeddings, then train an MLP + nnPU head — cheaper and easier to leakage-control.

3. PyTorch Geometric at this scale

You do NOT need a 1.2TB server. Two viable paths:

Decoupled (recommended first): precompute per-cluster features/embeddings offline (DuckDB/Polars + optional shallow propagation), then PU-train on tabular vectors. Scoring 49M rows is a streamed batched forward pass.
Sampled GNN: keep edge_index + node features in host RAM (or disk-backed), use NeighborLoader with num_neighbors=[15,10] (2 hops) and modest batch size; only seed nodes/subgraphs of interest are sampled. For graphs exceeding RAM, PyG supports remote backends (e.g., Kùzu, which streams subgraphs from disk) and torch_geometric.distributed (≥2.5). PyTorch Geometric


RevTrack loading: ships a preprocessed version in data/elliptic/raw; node embeddings raw_emb.pt downloaded separately (Google Drive). RevTrack deliberately uses non-GNN nets on sender/receiver representations because they "scale better"; all paper experiments ran on a single V100. preprocess_glass.py reads CSVs with pandas, remaps ids via dicts, and writes an integer edge_list.txt + subgraphs.pth. GitHub + 2
GPU budget: A10G has 24GB (nvidia-smi shows 23028MiB). Sampled GNN minibatches + the nnPU MLP head use only a few GB; the binding constraint is host RAM for the graph, not GPU memory.

4. Feature engineering (behavioral fingerprinting)
Compute per-cluster aggregates over the directed weighted graph:

Degree: in-degree, out-degree, in/out ratio, total degree.
Flow concentration: Gini coefficient of incoming and of outgoing edge weights; HHI; max-single-counterparty share. Peeling chains produce characteristic 1-in/2-out repeated structures with a dominant "change" output. Merkle Science
Neighborhood label distribution: fraction of 1-hop/2-hop neighbors labeled licit / suspicious / unknown (leakage-controlled — see Eval).
Edge-feature aggregates per node: sum/mean/max/std of volume, fee, timestamp spread, for in- and out-edges.
Temporal: activity span, burstiness within the 1-year construction window.
Path-role (RevTrack insight): is the cluster a likely source (illicit-side) or sink (exchange/licit endpoint)?
Compute at scale with DuckDB SQL group-bys over the edge table (joins on clId1/clId2) or Polars lazy frames; both stream from disk and handle 196M rows. Use PyG transforms only for the sampled candidate set.
Predictive priors from Elliptic1 literature: aggregated/neighborhood features materially boost GNN performance over local-only features; node degree alone surfaces peeling chains (per the Elliptic2 paper's own model analysis); exchange-adjacency and nested-service patterns near final deposits are strong suspicious signals. arxiv
Subgraph-level readout (Path A — the decisive RevClassify insight): represent each subgraph by BORDER-node Deep Sets — DeepSets(senders S) ⊕ DeepSets(receivers R), where S/R are the outside nodes pointing into sources / pointed to by sinks — concatenated with [sum, mean, max] pooled internal node (43-d) and edge (95-d) features. The border (who funds the subgraph, who receives from it) carries more signal than internal shape; RevClassify beats GLASS precisely because licit vs suspicious internal graphlet distributions are nearly identical. Sum preserves subgraph size, mean normalizes, max captures the most extreme member; Deep Sets is the default aggregator (Set2Set/attention only if you need more expressivity). Feed the fixed-length vector to an MLP — weighted/focal BCE for the supervised subgraph model, nnPU only for the cluster-level scorer.

5. Docker + EC2 g5.xlarge

Base image: use an AWS Deep Learning Container (PyTorch GPU on Ubuntu 22.04) matched to a CUDA your PyG wheels support. As of mid-2026 AWS DLCs exist for PyTorch 2.5/CUDA 12.4, 2.7/CUDA 12.8, and 2.9/CUDA 13.0. Pin PyG companion wheels to the exact torch+CUDA: install torch-scatter/torch-sparse/torch-cluster/torch-spline-conv from https://data.pyg.org/whl/torch-${VER}+cu${CUDA}.html. CUDA/PyG mismatch is the #1 install failure.
Alternative: Deep Learning AMI (Ubuntu 22.04) bakes NVIDIA driver + CUDA + Docker; then a slim custom Dockerfile. g5 needs the NVIDIA driver + NVIDIA Container Toolkit; A10G shows as 23028MiB. AWS re:Post
S3: dataset (~26GB) — download once to an EBS gp3 volume (≥100GB) via aws s3 cp/s5cmd rather than streaming per-epoch; keep preprocessed CSR/feature artifacts on EBS and back up to S3.
Spot interruption: poll IMDSv2 http://169.254.169.254/latest/meta-data/spot/instance-action every 5s and/or trap SIGTERM; on the 2-minute notice, checkpoint model+optimizer+RNG+epoch/offset to S3 and exit non-zero so the job requeues. Resume by loading the latest S3 checkpoint. Test locally with Amazon EC2 Metadata Mock (AEMM). Amazon Web Services + 3
User data script: install driver/toolkit (or use DLAMI) → pull image from ECR → aws s3 cp the data → run container with --gpus all, mount EBS → start training with checkpoint-resume.

6. Repository structure

Config: Hydra (RevTrack itself uses Hydra-style YAML configs + wandb sweeps) — composable dataset/, model/, experiment/ groups and CLI overrides, ideal for Claude Code generating one module at a time.
Layout:

ellip2-aml/
  configs/            # hydra: dataset/, features/, pu/, gnn/, llm/, infra/
  src/ellip2/
    data/             # download.py, schema.py, duckdb_loaders.py, csr_build.py
    features/         # degree.py, flow_concentration.py, neighborhood.py, edge_aggs.py
    graph/            # pyg_data.py, neighbor_sampling.py
    pu/               # nnpu_loss.py, prior_estimation.py, trainer.py, encoder.py
    exit_paths/       # path_search.py (<=6 hops to licit endpoints)
    eval/             # pu_metrics.py, splits.py, leakage_checks.py
    llm/              # serialize_subgraph.py, bedrock_client.py, typology_graph.py
    report/           # render.py
  scripts/            # train_pu.py, score_all.py, discover.py, classify_typology.py
  docker/Dockerfile
  infra/userdata.sh
  pyproject.toml

Each module has a single responsibility with typed I/O contracts against the Hydra config schema, so Claude Code can implement them independently.

7. Evaluation

You cannot compute precision without negatives. Under the SCAR assumption, recall is estimable from PU data (fraction of held-out positives recovered); report recall on a held-out positive subgraph split. Use PR-AUC treating known positives vs. unlabeled (a lower bound — some unlabeled are truly positive, depressing apparent precision) and the lift-style PU metric recall² / Pr(predicted positive) (Lee & Liu) which ranks models without true negatives. arxivarxiv
Reproduce RevClassify evaluation: report binary F1 on the suspicious class and PR-AUC (logged as final_test/f1, final_test/prauc), using the dataset's shipped train/val/test split (preprocess_glass.py uses a deterministic modulo-10 ≈80/10/10 round-robin over subgraphs). GitHub
"Newly discovered suspicious subgraphs": follow RevFilter — iteratively filter licit transactions to surface candidates, then require corroboration: (a) high PU score, (b) recognizable typology (peeling chain / nested service), (c) a valid ≤6-hop illicit→licit exit path, (d) ideally an external cross-check. Report Hit-Rate / NDCG ranking metrics as RevFilter does.
Data leakage (Elliptic1 lessons → Elliptic2): Elliptic1's known pitfalls are temporal and feature leakage. For Elliptic2: (a) when building neighborhood-label features, exclude the subgraph's own label and the labels of test-split subgraphs; (b) respect the 1-year construction window — avoid future-derived aggregates; (c) keep the positive split disjoint across prior-estimation, training, and recall evaluation; (d) never let background features encode subgraph membership. Add unit tests asserting these invariants.

8. LangGraph / explainability layer

Serialize each candidate subgraph to compact JSON: nodes with role + key (binned) features, directed edges with weight/volume/timestamp, the detected exit path(s), degree/Gini summary stats, and the PU score. Keep within Bedrock Converse payload limits.
Prompting: few-shot classify into named typologies — peeling chain (chain of similar-degree nodes with a persisting "change" output, small amounts peeled), nested service (multiple illicit paths merging on one "service" node that forwards to an exchange), layering/smurfing, consolidation. Provide the Elliptic2/Elliptic typology definitions and ask for typology + confidence + rationale + which structural evidence supports it. arXiv
Bedrock integration: boto3.client("bedrock-runtime") with the Converse API (converse/converse_stream), an Anthropic Claude model id available in your region. Wrap as a LangGraph node: state = candidate subgraph → serialize → Converse → parse typology → validate against rule-based checks → emit report. Use LangGraph for the multi-step agent (RAG over typology docs → classify → self-check → write report).
Per-subgraph report: subgraph id, PU score + percentile, recovered exit path(s) to a named licit endpoint type, assigned typology + confidence, structural evidence (degree/Gini/flow stats), LLM rationale, and a caveat block on false-positive risk (payroll/treasury patterns mimic peeling chains).

Details
Recommended end-to-end pipeline (stage by stage)
Stage 0 — Ingest. aws s3 cp the archive to EBS; unzip; with DuckDB build (a) an integer-remapped CSR edge list (edge_index.npy, int32), (b) node_features.npy (49M×43), (c) a subgraphs parquet mapping ccId→member node ids→ccLabel from connected_components.csv + nodes.csv. Validate counts against Table 1 (49,299,864 / 196,215,606 / 121,810).
Stage 1 — Feature engineering. DuckDB group-bys keyed on clId1/clId2 produce degree, in/out ratio, weight-Gini, HHI, edge-feature aggregates; one shallow propagation pass (sparse mat-vec via scipy/torch.sparse) yields 1–2 hop neighborhood label fractions (leakage-masked). Output cluster_features.parquet.
Stage 2 — Classification. SUPERVISED subgraph baseline first: border-set + pooled features → weighted-BCE/focal MLP, PR-AUC selection, on a persisted split — this is the apples-to-apples RevClassify comparator. OPTIONAL cluster-level PU: positives = clusters in suspicious ccs; unlabeled = the ~49M rest; estimate π_p with KM2/TIcE and sweep 1e−1…1e−5 (do NOT anchor to the 2.27% subgraph rate); train nnPU (MLP head on features/embeddings, or GNN encoder on sampled subgraphs). Checkpoint to S3. Score clusters in streamed batches → scores.parquet, then MAX-POOL member-cluster scores to each subgraph (MIL: bag positive iff ≥1 positive instance) for subgraph-level evaluation against RevClassify.
Stage 3 — Exit-path discovery (reachability, not enumeration). The endpoint is the labeled licit receiver/sink (RevTrack's R set); the dataset ships no entity types, so any "exchange" typing is an explicitly-flagged structural heuristic (high in-degree + large cluster size + throughput + address reuse). Run backward multi-source BFS from the (smaller) endpoint set on the transposed adjacency (boolean SpMV / cuGraph from a virtual super-source), per-level frontier caps, hubs excluded from pass-through; intersect with capped forward BFS from top-percentile candidate sources; extract induced subgraphs only on survivors (PyG k_hop_subgraph, flow="target_to_source"). A candidate + its path = a newly discovered suspicious subgraph. Rank, then send to the LLM typology/report layer. Corroborate before claiming.
Library versions & compatibility (pin these)

Choose ONE coherent stack, e.g. AWS DLC PyTorch 2.5 + CUDA 12.4 + Python 3.11, PyG ≥2.5 (for distributed/remote-backend support), companion wheels from https://data.pyg.org/whl/torch-2.5.0+cu124.html. RevTrack's own env: conda python=3.10, its requirements.txt, single V100.
DuckDB ≥1.x, Polars ≥1.x, boto3 ≥1.34 (Converse API), current langgraph/langchain, s5cmd for fast S3.
Known gotchas: (1) installing PyG companion libs from PyPI instead of the matching wheel index → CUDA symbol errors; (2) g5 needs the NVIDIA driver before the container toolkit or torch.cuda.is_available() is False; (3) ARM64 DLCs (G5g/Graviton) need a working Triton — stick to x86 g5.xlarge.

Recommendations

Build the subgraph-level SUPERVISED baseline first (border-set + pooled features → weighted-BCE/focal MLP), validating PR-AUC/F1 on a persisted split before adding a GNN encoder or the cluster-level PU scorer. Benchmark to beat: RevClassify_DS — and ALWAYS name the table, because the GLASS numbers differ ~4× by setup: RevTrack Table 1 (GPU + node features) reports RevClassify_DS PR-AUC 0.974 / F1 0.953 and GLASS 0.816 / 0.705, whereas Elliptic2-paper Table 2 (CPU, features ignored) reports GLASS F1 0.933 / PR-AUC 0.208 — not comparable.
Treat the class prior as a CLUSTER-level hyperparameter swept across a wide grid (π_p ∈ {1e−1 … 1e−5}); do NOT anchor it to the 2.27% subgraph base rate, and use reliable licit labels as known negatives for validation/calibration. Threshold to change course: if PR-AUC swings >0.1 across the grid, prefer the subgraph-level supervised framing and keep PU only as a secondary cluster-level signal.
Gate every "discovery" behind 3 corroborating signals (PU score percentile, valid ≤6-hop exit path, recognized typology). Threshold: only report candidates above ~99th percentile PU score with a complete exit path.
Engineer for spot from day one: checkpoint every N minutes to S3, SIGTERM/IMDS handler, idempotent resume. Threshold: if interruption rate makes wall-clock unpredictable, run the (short) PU-training step on on-demand g5 and keep scoring on spot.
Lock leakage controls into the feature layer (unit tests asserting no test-subgraph label touches training features) before scaling to 49M.
Use Hydra + wandb mirroring RevTrack so you can diff your numbers against theirs directly.

Caveats

g5.xlarge specs (important correction): the g5.xlarge is 4 vCPUs, 16 GiB RAM, 1× NVIDIA A10G (24 GB) — per AWS/Vantage specs ("The g5.xlarge instance is in the GPU instance family with 4 vCPUs, 16 GiB of memory"). 16 GiB RAM is tight for holding 49M×43 float32 node features (≈8.4GB) plus a 1.6GB edge index plus working memory. Recommend either g5.2xlarge/g5.4xlarge (the 16 vCPU / 64 GiB config is g5.4xlarge) for the feature-build/scoring steps, or do all heavy aggregation in DuckDB (out-of-core, spills to disk) and load only compact artifacts into RAM. This is the most important infra adjustment to the assumed setup.
Feature column names (43 node / 95 edge) are not published — they are anonymized binned ordinals; any "which features are most predictive" analysis must be empirical on Elliptic2, not transferred verbatim from Elliptic1 (which used 166 differently-defined features).
Suspicious counts differ slightly between sources (2,763 in the Elliptic2 paper vs. 2,718 in RevTrack) — use the count in your downloaded copy.
The graph-PU heterophily result is recent (ICML 2024) and GPL has no widely-adopted PyG implementation; budget time to port the nnPU + label-propagation loss yourself.
"Statistically indistinguishable from confirmed illicit clusters" is a ranking/anomaly claim, not proof of laundering; legitimate payroll/treasury flows mimic peeling chains. All outputs are investigative leads requiring human review.
Bedrock model availability and exact Claude model IDs vary by region; confirm model access in your region before wiring the LLM layer.