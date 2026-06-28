# One image for the whole pipeline. GPU-capable for Stages 2-3 (nnPU train / score),
# but Stage 0 ingest and Stage 1 features run CPU-only inside this same image.
#
# Base = AWS Deep Learning Container (PyTorch GPU). Pin to a CUDA your PyG wheels
# support (plan.md §"Library versions & compatibility"). As of mid-2026 the
# PyTorch 2.5 / CUDA 12.4 DLC is a known-good pairing. Override at build time:
#   docker build -f docker/Dockerfile \
#     --build-arg BASE_IMAGE=763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:2.5.0-gpu-py311-cu124-ubuntu22.04-ec2 \
#     -t ellip2:latest .
# (ECR DLC pulls need: aws ecr get-login-password ... | docker login 763104351884.dkr.ecr.<region>.amazonaws.com)
ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.0-gpu-py311-cu124-ubuntu22.04-ec2
FROM ${BASE_IMAGE}

# torch+CUDA the base ships, used to pick the matching PyG wheel index.
ARG TORCH_VERSION=2.5.0
ARG CUDA_TAG=cu124

WORKDIR /opt/ellip2

# Install our package's CPU-only data deps first (cached layer). We rely on the
# DLC's pre-installed torch — do NOT reinstall torch here or you risk CUDA mismatch.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# PyG companion wheels MUST come from the matching torch+CUDA index, NOT PyPI
# (PyPI builds -> CUDA symbol errors). This is the #1 install failure.
RUN pip install --no-cache-dir torch-geometric \
        torch-scatter torch-sparse torch-cluster \
        -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html

COPY scripts ./scripts
COPY infra ./infra

# Sanity: package imports and DuckDB is present (verified at build, not just run).
RUN python -c "import ellip2, duckdb, numpy, pyarrow; print('ellip2 image OK', duckdb.__version__)"

# Default entrypoint is the ingest CLI; override `command` for other stages.
ENTRYPOINT ["ellip2-ingest"]
CMD ["--help"]
