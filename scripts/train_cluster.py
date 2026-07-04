#!/usr/bin/env python3
"""Thin shim around ellip2.pu.cluster_score.train_main (per-cluster suspicion HGBM)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ellip2.pu.cluster_score import train_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(train_main())
