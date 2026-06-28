# RUNBOOK — running the ellip2 pipeline on AWS (EC2 + Docker)

End-to-end steps to take the built pipeline from this repo to a real run on the
Elliptic2 dataset. Read `DEPLOY.md` first for the why; this is the how.

> **Reality check — what's runnable today**
> | Stage | CLI | Status |
> |-------|-----|--------|
> | 0 Ingest | `ellip2-ingest` | ✅ ready (Docker ENTRYPOINT) |
> | — Split | `ellip2-make-split` | ✅ ready |
> | 1 Features | `python scripts/build_features.py` | ✅ ready |
> | 2 Train / Score | — | ⚠️ **library only** (`ellip2.pu.trainer`), no CLI driver yet |
> | 3 Discover | `python scripts/discover.py` | ✅ ready (needs `scores.parquet` from Stage 2) |
>
> Stage 2 needs a ~40-line `scripts/train.py` / `scripts/score.py` wrapper around the
> existing `train_supervised` / `train_cluster_nnpu` / `max_pool_to_subgraph` functions,
> plus extending `infra/userdata.sh` (it only wires `ingest` today). Ask me to write
> those before the real-data run — the rest below is complete.

---

## Part 0 — Prerequisites (local, one-time)

```bash
# Local tools
aws --version          # AWS CLI v2
docker --version       # Docker (for building the image)
aws configure          # set access key, secret, default region (e.g. us-east-1)

# Pick your identifiers — used throughout
export REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET=ellip2-$ACCOUNT_ID          # S3 bucket name (must be globally unique)
export ECR_REPO=ellip2
echo "acct=$ACCOUNT_ID region=$REGION bucket=$BUCKET"
```

You need an AWS account with permissions for: S3, ECR, EC2, and IAM (to create one role).
**Cost heads-up:** a g5.xlarge is ~$1.00/hr on-demand (~$0.40/hr spot); EBS gp3 100 GB is
~$8/mo; S3 storage for ~26 GB is ~$0.60/mo. Budget a few dollars for a full run + teardown.

---

## Part 1 — Get the dataset onto S3 (~26 GB, one-time)

The 5 CSVs are not redistributed here. Download from Kaggle
(`ellipticco/elliptic2-data-set`) or the MITIBMxGraph/Elliptic2 release, unzip to get:
`background_nodes.csv`, `background_edges.csv`, `connected_components.csv`, `nodes.csv`,
`edges.csv`.

```bash
# Create the bucket (skip if it exists)
aws s3 mb s3://$BUCKET --region $REGION

# Upload the 5 CSVs (run from wherever you unzipped them)
aws s3 cp ./elliptic2/ s3://$BUCKET/elliptic2/raw/ --recursive --exclude "*" --include "*.csv"

# Verify all five landed
aws s3 ls s3://$BUCKET/elliptic2/raw/
```

> Tip: 26 GB over a home connection is slow. If you have the data on another cloud box,
> upload from there. `s5cmd` is much faster than `aws s3 cp` for this.

---

## Part 2 — Build & push the Docker image to ECR

The image bakes in the whole package. Two paths — pick one.

### 2a. Create the ECR repo + log in

```bash
aws ecr create-repository --repository-name $ECR_REPO --region $REGION || true
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

# The base image is an AWS Deep Learning Container in a DIFFERENT account (763104351884).
# You must also log in there to pull it:
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$REGION.amazonaws.com
```

### 2b. Build (from the repo root) and push

```bash
cd /home/ogreenowow/dev/microservices/bitcoin

# If your region isn't us-east-1, override the DLC base account/region in the build arg.
docker build -f docker/Dockerfile -t $ECR_REPO:latest \
  --build-arg BASE_IMAGE=763104351884.dkr.ecr.$REGION.amazonaws.com/pytorch-training:2.5.0-gpu-py311-cu124-ubuntu22.04-ec2 \
  .

docker tag $ECR_REPO:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$ECR_REPO:latest
```

> **Simpler alternative:** skip local build entirely and build *on the EC2 instance*
> (Part 3) after it boots — the DLAMI already has Docker + AWS creds, so you avoid pushing
> a multi-GB image over your home uplink. Just `git clone` (or `scp`) the repo there and
> run the same `docker build`.

---

## Part 3 — Launch the EC2 instance

### 3a. One-time IAM role so the instance can read S3 + ECR

```bash
# Trust policy
cat > /tmp/ec2-trust.json <<'JSON'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow",
 "Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}
JSON

aws iam create-role --role-name ellip2-ec2 \
  --assume-role-policy-document file:///tmp/ec2-trust.json
aws iam attach-role-policy --role-name ellip2-ec2 \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam attach-role-policy --role-name ellip2-ec2 \
  --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
aws iam create-instance-profile --instance-profile-name ellip2-ec2
aws iam add-role-to-instance-profile \
  --instance-profile-name ellip2-ec2 --role-name ellip2-ec2
```

### 3b. Launch a g5.xlarge on the Deep Learning AMI

The **DLAMI** ships the NVIDIA driver + Docker + NVIDIA Container Toolkit pre-baked, so
`--gpus all` works out of the box. Find the current DLAMI id:

```bash
# Latest Ubuntu 22.04 DLAMI (GPU, PyTorch). Names change — list and pick the newest.
aws ec2 describe-images --owners amazon --region $REGION \
  --filters "Name=name,Values=Deep Learning*Ubuntu 22.04*" "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[:3].[ImageId,Name]' --output table

export AMI_ID=ami-xxxxxxxx     # paste one from above
export KEY=my-keypair          # an existing EC2 key pair name (for SSH)
export SG=sg-xxxxxxxx          # a security group allowing inbound SSH (port 22) from your IP
```

Launch with a 150 GB gp3 root volume (holds the 26 GB raw + artifacts with headroom):

```bash
aws ec2 run-instances --region $REGION \
  --image-id $AMI_ID --instance-type g5.xlarge \
  --key-name $KEY --security-group-ids $SG \
  --iam-instance-profile Name=ellip2-ec2 \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":150,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=ellip2}]' \
  --query 'Instances[0].InstanceId' --output text
```

> **Cheaper for Stages 0–1:** ingest + features are CPU-only and DuckDB-bound. You can run
> them on a `c7i.2xlarge` (no GPU) and only spin up the g5 for Stage 2 train/score.
> The g5.xlarge has just **16 GiB RAM**, so always pass `--memory-limit` to ingest.

Get the public IP and SSH in:

```bash
aws ec2 describe-instances --instance-ids <id> \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
ssh -i ~/.ssh/$KEY.pem ubuntu@<public-ip>
```

---

## Part 4 — Run the stages (on the instance)

```bash
# On the instance — set the same env
export REGION=us-east-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export BUCKET=ellip2-$ACCOUNT_ID
export IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/ellip2:latest
export RAW=/mnt/data/elliptic2/raw
export OUT=/mnt/data/elliptic2/artifacts
mkdir -p $RAW $OUT

# Pull data from S3 and the image from ECR
aws s3 cp s3://$BUCKET/elliptic2/raw/ $RAW/ --recursive
aws ecr get-login-password --region $REGION | docker login --username AWS \
  --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker pull $IMAGE
```

### Stage 0 — Ingest (CPU; ~minutes)

```bash
# ENTRYPOINT is ellip2-ingest, so flags go straight on `docker run`
docker run --rm -v /mnt/data:/mnt/data $IMAGE \
  --raw-dir $RAW --out-dir $OUT --memory-limit 12GB --feature-dtype float32
# Produces: id_map.parquet, node_features.npy, edge_index.npy, subgraphs.parquet,
# ingest_manifest.json. Check the manifest's counts match Table 1 (49,299,864 nodes etc.).
```

### Split (CPU; seconds) — override the entrypoint

```bash
docker run --rm -v /mnt/data:/mnt/data --entrypoint ellip2-make-split $IMAGE \
  --input $RAW/connected_components.csv --out-dir $OUT/splits
# -> $OUT/splits/stratified_random/split.csv  (reuse everywhere for comparability)
```

### Stage 1 — Features (CPU; DuckDB out-of-core)

```bash
docker run --rm -v /mnt/data:/mnt/data --entrypoint python $IMAGE \
  scripts/build_features.py \
    --artifacts-dir $OUT \
    --raw-dir $RAW \
    --split-csv $OUT/splits/stratified_random/split.csv
# -> $OUT/cluster_features.parquet
# (edge-feature column meanings are anonymized; --weight-index/--timestamp-index/
#  --size-index/--edge-agg-indices let you point at the right columns once profiled.)
```

### Stage 2 — Train + Score (GPU) ⚠️ needs the driver script first

No CLI yet — `ellip2.pu.trainer` exposes `train_supervised`, `train_cluster_nnpu`,
`max_pool_to_subgraph` as functions. Once `scripts/train.py` + `scripts/score.py` exist
(ask me to write them), the run is:

```bash
docker run --rm --gpus all -v /mnt/data:/mnt/data --entrypoint python $IMAGE \
  scripts/train.py --features $OUT/cluster_features.parquet \
    --subgraphs $OUT/subgraphs.parquet \
    --split-csv $OUT/splits/stratified_random/split.csv \
    --out $OUT/model.pt
docker run --rm --gpus all -v /mnt/data:/mnt/data --entrypoint python $IMAGE \
  scripts/score.py --model $OUT/model.pt --features $OUT/cluster_features.parquet \
    --out $OUT/scores.parquet
# Validate against RevClassify: report final_test/prauc + final_test/f1 (eval/pu_metrics).
```

### Stage 3 — Discover (CPU)

```bash
docker run --rm -v /mnt/data:/mnt/data --entrypoint python $IMAGE \
  scripts/discover.py \
    --scores $OUT/scores.parquet \
    --edge-index $OUT/edge_index.npy \
    --endpoints $OUT/endpoints.npy \
    --out $OUT/candidates.parquet \
    --score-percentile 0.99 --max-hops 6
# Ranked candidate subgraphs with corroborating exit paths.
# (--endpoints = the licit-receiver set from the path_role heuristic; produced in Stage 1/2.)
```

---

## Part 5 — Save results & tear down

```bash
# Push artifacts back to S3 (keep them; delete the box)
aws s3 cp $OUT/ s3://$BUCKET/elliptic2/artifacts/ --recursive \
  --exclude "*.npy"   # the big memmaps are reproducible; keep parquet/model/scores

# From your laptop — STOP (keep EBS, resume later) or TERMINATE (delete everything)
aws ec2 stop-instances --instance-ids <id>        # ~$8/mo EBS keeps state
aws ec2 terminate-instances --instance-ids <id>   # full teardown, no further cost
```

---

## Gotchas (from plan.md, learned the hard way)

- **16 GiB RAM on g5.xlarge** — never load CSVs into pandas; ingest streams via DuckDB and
  always pass `--memory-limit`. For RAM-heavy steps use g5.2xlarge/4xlarge.
- **PyG/CUDA mismatch is the #1 failure** — the Dockerfile pins companion wheels to the
  base's torch+CUDA. Don't `pip install torch-geometric` from PyPI inside the image.
- **Driver before container** — on a non-DLAMI box, install the NVIDIA driver + container
  toolkit *before* `docker run --gpus all`, or `torch.cuda.is_available()` is False.
- **Spot interruptions** — for long runs use spot + checkpoint to S3 and an IMDS
  `spot/instance-action` poll (this is skip-task T-023, not yet wired).
- **Suspicious count** — use the count in *your* downloaded copy (2,763 paper vs 2,718
  RevTrack); ingest warns but doesn't fail on the difference.
