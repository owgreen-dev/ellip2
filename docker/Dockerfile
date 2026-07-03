# One image for the whole pipeline. GPU-capable for Stages 2-3 (nnPU train / score),
# but Stage 0 ingest and Stage 1 features run CPU-only inside this same image.
#
# Base = AWS Deep Learning Container (PyTorch GPU). The DLC ships torch built for
# its CUDA; we NEVER reinstall torch (that would clobber the CUDA build). Override
# the base at build time to track a newer DLC:
#   docker build -f docker/Dockerfile \
#     --build-arg BASE_IMAGE=763104351884.dkr.ecr.<region>.amazonaws.com/pytorch-training:<tag> \
#     -t ellip2:latest .
# (ECR DLC pulls need: aws ecr get-login-password ... | docker login 763104351884.dkr.ecr.<region>.amazonaws.com)
#
# torch-geometric is pure-Python and works against whatever torch the DLC ships —
# this codebase does NOT use the torch-scatter/sparse/cluster CUDA companion wheels
# (no imports of them anywhere), so we deliberately do NOT install them. That avoids
# the "PyG/CUDA wheel mismatch" build failure entirely.
ARG BASE_IMAGE=763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.39
FROM ${BASE_IMAGE}

WORKDIR /opt/ellip2

# Install our package's CPU-only data deps first (cached layer). These are the core
# deps only (duckdb/numpy/pyarrow) — NOT the `gpu` extra — so the DLC's torch stays
# untouched.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Runtime deps not provided by the base: torch-geometric (GNN encoder / MessagePassing)
# and scipy (sparse ops in features/neighborhood + exit_paths/path_search). Pure-Python
# / prebuilt wheels from PyPI — no CUDA-specific index needed.
RUN pip install --no-cache-dir torch-geometric scipy

# pyg-lib: the neighbor-sampling kernel PyG's NeighborLoader needs for Stage 2
# minibatch training/scoring (T-012). This is the ONE torch+CUDA-pinned wheel we
# keep — a matching prebuilt exists for the base's torch 2.8.0 / cu129 / cp312.
# (Override PYG_WHL_INDEX if you change BASE_IMAGE to a different torch/CUDA.)
ARG PYG_WHL_INDEX=https://data.pyg.org/whl/torch-2.8.0+cu129.html
RUN pip install --no-cache-dir pyg-lib -f ${PYG_WHL_INDEX}

COPY scripts ./scripts
COPY infra ./infra

# Sanity: the whole runtime import surface resolves at BUILD time, not just at run.
RUN python -c "import ellip2, duckdb, numpy, pyarrow, scipy, torch, torch_geometric; \
print('ellip2 image OK — torch', torch.__version__, 'pyg', torch_geometric.__version__, 'duckdb', duckdb.__version__)"

# Default entrypoint is the ingest CLI; override `command` for other stages.
ENTRYPOINT ["ellip2-ingest"]
CMD ["--help"]
