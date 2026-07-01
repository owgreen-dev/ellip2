#!/usr/bin/env python3
"""Thin shim around ellip2.pu.train so Stage 2 training runs from a source
checkout without installation.

Example:
    python scripts/train.py \
        --features artifacts/features/cluster_features.parquet \
        --edge-index artifacts/ingest/edge_index.npy \
        --subgraphs artifacts/ingest/subgraphs.parquet \
        --split-csv artifacts/splits/stratified_random/split.csv \
        --out artifacts/pu/model.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ellip2.pu.train import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
