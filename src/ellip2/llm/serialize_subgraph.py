"""Stage 4 (LLM layer) — compact, deterministic JSON serialization of a candidate.

plan.md §8: each candidate subgraph surfaced by Stage 3 is handed to the LLM
typology classifier (Bedrock Converse). Before it can be, the subgraph has to be
rendered as a *compact* JSON document that fits inside the Converse payload
budget while preserving the structural signals a model needs to recognise a
typology (peeling chain, nested service, layering, consolidation):

  * **nodes** — id, a (heuristic, DERIVED) ``role`` label, and the node's key
    features *binned* to small ordinals (raw 43-d anonymized vectors are both
    too large and uninformative verbatim — plan.md "binned features");
  * **directed edges** — ``source → target`` with weight / volume / timestamp;
  * the recovered **exit path(s)** to a heuristic licit endpoint (Stage 3);
  * **summary stats** — degree / Gini / flow concentration (Stage 1);
  * the cluster's **PU score** (Stage 2).

Two properties are load-bearing and both are unit-tested (SIGN-101 — pure
function, no network):

* **Determinism.** The same candidate always serializes to byte-identical JSON.
  We sort every collection by a stable key, round every float to a fixed
  precision, dump with ``sort_keys=True`` and compact separators. This matters
  because the serialized text is the LLM prompt: a non-deterministic prompt would
  defeat prompt caching and make runs irreproducible.
* **A configurable size budget, enforced deterministically.** If the full
  document exceeds :attr:`SerializationConfig.max_bytes`, detail is shed in a
  fixed priority order (least-important edges first, then peripheral nodes) until
  it fits — never by random sampling. Exit-path nodes are retained as long as
  possible because they carry the corroborating structure. The result records
  ``truncated`` so a downstream reader never mistakes a trimmed document for a
  complete one.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: Default number of ordinal bins for node features.
DEFAULT_N_BINS = 10
#: Default float rounding precision (decimal places) for deterministic output.
DEFAULT_PRECISION = 6


@dataclass(frozen=True)
class SubgraphNode:
    """One node in the candidate subgraph.

    Attributes:
        node_id: cluster (node) id.
        role: a DERIVED structural role label (e.g. ``"source"`` / ``"sink"`` /
            ``"endpoint"`` from the Stage 1 heuristic) — NOT a ground-truth
            entity type.
        features: the node's selected (anonymized) feature values, binned to
            ordinals during serialization.
    """

    node_id: int
    role: str
    features: Sequence[float] = field(default_factory=tuple)


@dataclass(frozen=True)
class SubgraphEdge:
    """One directed edge ``source → target`` with transaction attributes."""

    source: int
    target: int
    weight: float = 0.0
    volume: float = 0.0
    timestamp: float = 0.0


@dataclass(frozen=True)
class CandidateSubgraph:
    """A Stage 3 candidate ready for serialization.

    Attributes:
        subgraph_id: the candidate's subgraph id.
        pu_score: its Stage 2 PU score.
        nodes: member nodes.
        edges: directed edges among the members.
        exit_paths: recovered ≤k-hop exit path(s); each is a node-id sequence
            from a candidate source to a heuristic licit endpoint.
        stats: structural summary stats (degree / Gini / flow concentration).
    """

    subgraph_id: int
    pu_score: float
    nodes: Sequence[SubgraphNode] = field(default_factory=tuple)
    edges: Sequence[SubgraphEdge] = field(default_factory=tuple)
    exit_paths: Sequence[Sequence[int]] = field(default_factory=tuple)
    stats: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SerializationConfig:
    """Knobs for the serializer.

    Attributes:
        max_bytes: optional UTF-8 byte budget for the encoded JSON. ``None``
            disables the budget. When set, detail is shed deterministically until
            the document fits (see module docstring).
        n_bins: number of ordinal bins for node features (min-max within the
            subgraph, per feature column).
        precision: decimal places every float is rounded to before encoding.
    """

    max_bytes: int | None = None
    n_bins: int = DEFAULT_N_BINS
    precision: int = DEFAULT_PRECISION


def _round(value: float, precision: int) -> float:
    # Normalise -0.0 to 0.0 so the encoded text is stable regardless of sign.
    r = round(float(value), precision)
    return 0.0 if r == 0.0 else r


def _bin_features(
    nodes: Sequence[SubgraphNode], n_bins: int
) -> dict[int, list[int]]:
    """Bin each feature column to ``[0, n_bins)`` ordinals via per-column min-max.

    Deterministic given the node set: a column with no spread maps to all-zeros;
    otherwise ``bin = floor((v - lo) / (hi - lo) * n_bins)`` clamped to the top
    bin. Returns a mapping ``node_id -> list[bin]``.
    """
    if not nodes:
        return {}
    width = max(len(n.features) for n in nodes)
    lo = [float("inf")] * width
    hi = [float("-inf")] * width
    for node in nodes:
        for j, v in enumerate(node.features):
            fv = float(v)
            if fv < lo[j]:
                lo[j] = fv
            if fv > hi[j]:
                hi[j] = fv

    binned: dict[int, list[int]] = {}
    for node in nodes:
        bins: list[int] = []
        for j, v in enumerate(node.features):
            span = hi[j] - lo[j]
            if span <= 0.0:
                bins.append(0)
            else:
                idx = int((float(v) - lo[j]) / span * n_bins)
                bins.append(min(idx, n_bins - 1))
        binned[node.node_id] = bins
    return binned


def _build_payload(
    candidate: CandidateSubgraph,
    config: SerializationConfig,
    *,
    keep_nodes: set[int] | None,
    edge_limit: int | None,
    truncated: bool,
) -> dict[str, Any]:
    """Assemble the JSON-able payload, optionally restricted to a node/edge subset."""
    binned = _bin_features(candidate.nodes, config.n_bins)

    nodes = [n for n in candidate.nodes if keep_nodes is None or n.node_id in keep_nodes]
    node_objs = [
        {"id": int(n.node_id), "role": str(n.role), "bins": binned[n.node_id]}
        for n in sorted(nodes, key=lambda n: n.node_id)
    ]
    kept_ids = {n.node_id for n in nodes}

    edges = [
        e
        for e in candidate.edges
        if e.source in kept_ids and e.target in kept_ids
    ]
    # Most-important edges first (heaviest weight), ties by endpoints — a stable
    # order that survives the edge-limit truncation below.
    edges.sort(key=lambda e: (-float(e.weight), e.source, e.target))
    if edge_limit is not None:
        edges = edges[:edge_limit]
    edge_objs = [
        {
            "s": int(e.source),
            "t": int(e.target),
            "weight": _round(e.weight, config.precision),
            "volume": _round(e.volume, config.precision),
            "timestamp": _round(e.timestamp, config.precision),
        }
        for e in sorted(edges, key=lambda e: (e.source, e.target))
    ]

    paths = [
        [int(x) for x in path]
        for path in candidate.exit_paths
        if all(x in kept_ids for x in path)
    ]

    stats = {
        str(k): _round(v, config.precision)
        for k, v in sorted(candidate.stats.items())
    }

    return {
        "subgraph_id": int(candidate.subgraph_id),
        "pu_score": _round(candidate.pu_score, config.precision),
        "n_nodes": len(node_objs),
        "n_edges": len(edge_objs),
        "nodes": node_objs,
        "edges": edge_objs,
        "exit_paths": paths,
        "stats": stats,
        "truncated": truncated,
    }


def _encode(payload: Mapping[str, Any]) -> str:
    """Deterministic, compact JSON encoding."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _node_priority(candidate: CandidateSubgraph) -> list[int]:
    """Node ids ordered most- to least-important for budget trimming.

    Exit-path nodes first (they carry the corroborating structure), then the rest
    by ascending id. Deterministic.
    """
    on_path: set[int] = set()
    for path in candidate.exit_paths:
        on_path.update(int(x) for x in path)
    ordered = sorted(
        (n.node_id for n in candidate.nodes),
        key=lambda nid: (nid not in on_path, nid),
    )
    return ordered


def serialize_subgraph(
    candidate: CandidateSubgraph,
    config: SerializationConfig | None = None,
) -> dict[str, Any]:
    """Serialize ``candidate`` to a compact, deterministic JSON-able dict.

    Args:
        candidate: the Stage 3 candidate subgraph.
        config: :class:`SerializationConfig`; defaults applied when ``None``.

    Returns:
        A dict ready for :func:`json.dumps` (already rounded / sorted / binned).
        When a ``max_bytes`` budget is set and the full document exceeds it,
        detail is shed deterministically until it fits and ``payload["truncated"]``
        is ``True``.

    Raises:
        ValueError: if ``config`` is invalid, an edge / exit-path references an
            unknown node, or the budget cannot be met even by the minimal
            document.
    """
    cfg = config if config is not None else SerializationConfig()
    if cfg.n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {cfg.n_bins}")
    if cfg.precision < 0:
        raise ValueError(f"precision must be >= 0, got {cfg.precision}")
    if cfg.max_bytes is not None and cfg.max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {cfg.max_bytes}")

    node_ids = {n.node_id for n in candidate.nodes}
    if len(node_ids) != len(candidate.nodes):
        raise ValueError("duplicate node_id in candidate.nodes")
    for e in candidate.edges:
        if e.source not in node_ids or e.target not in node_ids:
            raise ValueError(
                f"edge {e.source}->{e.target} references an unknown node"
            )
    for path in candidate.exit_paths:
        for x in path:
            if int(x) not in node_ids:
                raise ValueError(f"exit path references an unknown node {x}")

    full = _build_payload(
        candidate, cfg, keep_nodes=None, edge_limit=None, truncated=False
    )
    if cfg.max_bytes is None or len(_encode(full).encode("utf-8")) <= cfg.max_bytes:
        return full

    # Over budget: shed detail deterministically. First drop the least-important
    # edges (heaviest kept), then peripheral nodes (exit-path nodes last).
    priority = _node_priority(candidate)
    n_full_edges = full["n_edges"]
    best: dict[str, Any] | None = None
    for n_keep in range(len(priority), 0, -1):
        keep = set(priority[:n_keep])
        for edge_limit in range(n_full_edges, -1, -1):
            payload = _build_payload(
                candidate, cfg, keep_nodes=keep, edge_limit=edge_limit, truncated=True
            )
            if len(_encode(payload).encode("utf-8")) <= cfg.max_bytes:
                best = payload
                break
        if best is not None:
            break

    if best is None:
        raise ValueError(
            f"cannot serialize within max_bytes={cfg.max_bytes}; "
            "the minimal document is still too large"
        )
    return best


def serialize_subgraph_json(
    candidate: CandidateSubgraph,
    config: SerializationConfig | None = None,
) -> str:
    """:func:`serialize_subgraph` then encode to a compact, deterministic JSON string."""
    return _encode(serialize_subgraph(candidate, config))
