#!/usr/bin/env bash
# EC2 g5 user-data: pull the image from ECR, stage data from S3 to EBS, run a stage.
# Assumes a Deep Learning AMI (NVIDIA driver + Docker + NVIDIA Container Toolkit
# pre-baked) OR that the driver/toolkit are installed before this runs. The driver
# MUST exist before the container, or torch.cuda.is_available() is False.
set -euo pipefail

REGION="${REGION:-us-east-1}"
ACCOUNT_ID="${ACCOUNT_ID:?set ACCOUNT_ID}"
IMAGE="${IMAGE:-${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/ellip2:latest}"
S3_DATA="${S3_DATA:?set S3_DATA, e.g. s3://my-bucket/elliptic2/}"   # holds the 5 CSVs
DATA_DIR="${DATA_DIR:-/mnt/ebs/elliptic2/raw}"
OUT_DIR="${OUT_DIR:-/mnt/ebs/elliptic2/artifacts}"
STAGE="${STAGE:-ingest}"   # ingest | features | train | score

mkdir -p "${DATA_DIR}" "${OUT_DIR}"

# 1. Stage the ~26GB dataset to EBS once (s5cmd is much faster than aws s3 cp).
if ! ls "${DATA_DIR}"/*.csv >/dev/null 2>&1; then
  if command -v s5cmd >/dev/null 2>&1; then
    s5cmd cp "${S3_DATA}*" "${DATA_DIR}/"
  else
    aws s3 cp --recursive "${S3_DATA}" "${DATA_DIR}/"
  fi
fi

# 2. Authenticate to ECR and pull the pipeline image.
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker pull "${IMAGE}"

# 3. Run the requested stage. Ingest/features are CPU-only (no --gpus needed);
#    train/score request the A10G. memory_limit keeps DuckDB under the 16GiB box.
case "${STAGE}" in
  ingest)
    docker run --rm -v /mnt/ebs:/mnt/ebs "${IMAGE}" \
      --raw-dir "${DATA_DIR}" --out-dir "${OUT_DIR}" \
      --memory-limit "${DUCKDB_MEM:-12GB}" --feature-dtype "${FEATURE_DTYPE:-float32}"
    ;;
  features|train|score)
    echo "stage '${STAGE}' not yet implemented (Stages 1-3 land next)"; exit 2
    ;;
  *)
    echo "unknown STAGE=${STAGE}"; exit 2 ;;
esac
