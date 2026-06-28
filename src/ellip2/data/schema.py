"""Canonical Elliptic2 schema: file names, column conventions, Table 1 counts.

Source of truth: arXiv:2404.19109 Table 1, and preprocess_glass.py in the
Elliptic2 repo. Feature column *names* are NOT published (anonymized binned
ordinals) — only their counts (43 node, 95 edge) are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Table 1 (paper) ------------------------------------------------------- #
N_NODES = 49_299_864          # background clusters
N_EDGES = 196_215_606         # background edges
N_SUBGRAPHS = 121_810         # labeled connected components
N_SUSPICIOUS = 2_763          # paper; RevTrack reports 2,718 (version skew)
N_LICIT = 119_047
N_NODE_FEATURES = 43
N_EDGE_FEATURES = 95

# --- File names ------------------------------------------------------------ #
F_BACKGROUND_NODES = "background_nodes.csv"   # clId + 43 node features
F_BACKGROUND_EDGES = "background_edges.csv"   # clId1,clId2 + 95 edge features
F_CONNECTED_COMPONENTS = "connected_components.csv"  # ccId + ccLabel
F_NODES = "nodes.csv"                         # node id -> ccId membership
F_EDGES = "edges.csv"                         # labeled intra-subgraph edges

ALL_FILES = (
    F_BACKGROUND_NODES,
    F_BACKGROUND_EDGES,
    F_CONNECTED_COMPONENTS,
    F_NODES,
    F_EDGES,
)

# --- Known column names (case-insensitive match; positional fallback) ------ #
COL_EDGE_SRC = "clId1"
COL_EDGE_DST = "clId2"
COL_CC_ID = "ccId"
COL_CC_LABEL = "ccLabel"

LABEL_SUSPICIOUS = "suspicious"
LABEL_LICIT = "licit"


@dataclass(frozen=True)
class Elliptic2Paths:
    """Resolved paths to the five raw CSVs under a dataset root."""

    root: Path

    @property
    def background_nodes(self) -> Path:
        return self.root / F_BACKGROUND_NODES

    @property
    def background_edges(self) -> Path:
        return self.root / F_BACKGROUND_EDGES

    @property
    def connected_components(self) -> Path:
        return self.root / F_CONNECTED_COMPONENTS

    @property
    def nodes(self) -> Path:
        return self.root / F_NODES

    @property
    def edges(self) -> Path:
        return self.root / F_EDGES

    def missing(self) -> list[str]:
        """Names of expected CSVs that are absent (empty list == all present)."""
        return [name for name in ALL_FILES if not (self.root / name).is_file()]

    def require_all(self) -> None:
        miss = self.missing()
        if miss:
            raise FileNotFoundError(
                f"missing Elliptic2 CSVs under {self.root}: {miss}"
            )


def expected_counts() -> dict[str, int]:
    """Table 1 counts keyed by artifact, for ingest validation."""
    return {
        "nodes": N_NODES,
        "edges": N_EDGES,
        "subgraphs": N_SUBGRAPHS,
        "suspicious": N_SUSPICIOUS,
        "node_features": N_NODE_FEATURES,
        "edge_features": N_EDGE_FEATURES,
    }


def resolve_column(columns: list[str], preferred: str,
                   positional_fallback: int) -> str:
    """Return a column from `columns` matching `preferred` (case-insensitive),
    else the column at `positional_fallback`. Raises if the fallback is out of
    range."""
    lower = {c.lower(): c for c in columns}
    if preferred.lower() in lower:
        return lower[preferred.lower()]
    if not 0 <= positional_fallback < len(columns):
        raise ValueError(
            f"column {preferred!r} not found and positional fallback "
            f"{positional_fallback} out of range for {columns!r}"
        )
    return columns[positional_fallback]
