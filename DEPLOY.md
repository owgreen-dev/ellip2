# Build & deploy: local → Docker → EC2

Yes — the model is **develop locally, bake into one Docker image, run on EC2**.

## 1. Develop & test locally (no GPU, no ~24 GB download)
Each module is unit-tested against tiny synthetic CSVs in `tests/`. Use a venv:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[dev]'
.venv/bin/python -m pytest -q          # 13 tests, ~1s
```

## 2. One Docker image for the whole pipeline
`docker/Dockerfile` builds on an **AWS Deep Learning Container** (PyTorch GPU,
CUDA pinned to a version the PyG wheels support) and `pip install -e .`s this
package. It's GPU-capable for Stages 2–3, but **Stage 0/1 run CPU-only in the same
image** (DuckDB is CPU). Build and push to ECR:

```bash
docker build -f docker/Dockerfile -t ellip2:latest .
aws ecr create-repository --repository-name ellip2 --region "$REGION" || true
docker tag ellip2:latest "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ellip2:latest"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ellip2:latest"
```

## 3. Run on EC2
`infra/userdata.sh` stages the 5 CSVs from S3 → EBS, pulls the image, runs a stage.
Stage 0:

```bash
docker run --rm -v /mnt/ebs:/mnt/ebs ellip2:latest \
  --raw-dir /mnt/ebs/elliptic2/raw --out-dir /mnt/ebs/elliptic2/artifacts \
  --memory-limit 12GB
```

## Stage → instance mapping (the key cost/RAM decision)

| Stage | Needs GPU? | RAM-sensitive? | Instance |
|-------|-----------|----------------|----------|
| 0 Ingest (DuckDB) | No | No — streams/spills to disk | any CPU box, or g5.xlarge with `--memory-limit` |
| 1 Features (DuckDB) | No | No — out-of-core | same |
| 2 PU train (nnPU) | **Yes (A10G)** | a few GB only | g5.xlarge on-demand |
| 3 Score + path search | Yes | host RAM for the graph | g5.xlarge (spot) |

**Critical caveat (plan.md):** g5.xlarge is only **16 GiB RAM** — too tight to hold
49M×43 float32 (~8.4 GB) plus the edge index plus working set in RAM at once. That's
exactly why Stage 0 keeps everything in DuckDB (out-of-core) and writes results as
numpy **memmaps** (`open_memmap`), never materializing full arrays. Bump to
g5.4xlarge (64 GiB) only if a later stage genuinely needs the graph resident.

## Artifacts produced by Stage 0 (under `--out-dir`)
- `id_map.parquet` — orig_id → contiguous int32 idx (the bijection)
- `node_features.npy` — (49,299,864 × 43)
- `edge_index.npy` — (2 × 196,215,606) int32, PyG-ready, remapped
- `subgraphs.parquet` — ccId, ccLabel, member_idx[]
- `ingest_manifest.json` — counts vs Table 1, integrity (dangling/orphans), timings

## Spot resilience (Stages 2–3, next)
Poll IMDSv2 `spot/instance-action` + trap SIGTERM → checkpoint model/optimizer/RNG
to S3 → exit non-zero to requeue. Run the short PU-train step on on-demand if spot
interruption makes wall-clock unpredictable.
