#!/usr/bin/env python3
"""Thin shim around ellip2.features.build so Stage 1 assembly runs from a source
checkout without installation.

Example:
    python scripts/build_features.py \
        --artifacts-dir artifacts/ingest \
        --raw-dir data/elliptic2 \
        --split-csv artifacts/splits/stratified_random/split.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ellip2.features.build import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
