#!/usr/bin/env python3
"""Thin shim around ellip2.discovery.discover so Stage 3 ranking runs from a
source checkout without installation.

Example:
    python scripts/discover.py \
        --scores artifacts/pu/scores.npy \
        --edge-index artifacts/ingest/edge_index.npy \
        --endpoints artifacts/stage3/endpoints.npy \
        --typology artifacts/stage3/source_sink_axis.npy \
        --out artifacts/stage3/candidates.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ellip2.discovery.discover import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
