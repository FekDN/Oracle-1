# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Graph snapshots and temporal-window persistence."""
from __future__ import annotations

from ._shared import *
from .canonical_keys import ALIAS_TABLE, DOMAIN_TABLE, ENTITY_TYPE_TABLE, GROUP_TABLE
from .graph import RuntimeGraph
from .time_utils import _ts_to_year

# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH SNAPSHOT (DE)SERIALIZATION (TODO #2)
# ══════════════════════════════════════════════════════════════════════════════

GRAPH_SNAPSHOT_FORMAT_VERSION = 1


class _CompatibleSnapshotUnpickler(pickle.Unpickler):
    """Load trusted snapshots across NumPy 1.x and 2.x module paths."""

    def find_class(self, module: str, name: str) -> Any:
        try:
            return super().find_class(module, name)
        except ModuleNotFoundError:
            if module.startswith("numpy._core."):
                compatible_module = module.replace("numpy._core", "numpy.core", 1)
                return super().find_class(compatible_module, name)
            raise


def _load_trusted_pickle(path: Path) -> Any:
    """Load an Oracle snapshot created by a trusted local run."""
    with path.open("rb") as handle:
        return _CompatibleSnapshotUnpickler(handle).load()


def graph_snapshot_summary(system: Dict[str, Any], top_k: int = 20) -> Dict[str, Any]:
    """Build a JSON-safe, inspection-only summary of the live graph.

    The binary snapshot intentionally preserves rich Python objects and is
    therefore only for trusted local artefacts. This companion summary is the
    safe surface for the local dashboard and experiment tracking, without
    unpickling a graph in a web request.
    """
    graph = system["graph"]
    nodes = list(graph.nodes.values())
    edges = list(graph.edges.values())

    def _number(value: Any) -> float:
        try:
            value = float(value)
            return value if math.isfinite(value) else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _year(node: "KnowledgeNode") -> Optional[int]:
        ts = _number(getattr(node, "timestamp_start", 0.0)) or _number(
            getattr(node, "timestamp", 0.0))
        if not ts:
            return None
        return _ts_to_year(ts)

    domain_counts = Counter(str(getattr(node, "domain", "unknown") or "unknown") for node in nodes)
    type_counts = Counter(str(getattr(node, "entity_type", "unknown") or "unknown") for node in nodes)
    origin_counts = Counter(str(getattr(node, "origin_kind", "combinatorial") or "combinatorial") for node in nodes)
    temporal_counts = Counter(str(getattr(node, "temporal_kind", "point") or "point") for node in nodes)
    yearly_counts = Counter(str(year) for node in nodes if (year := _year(node)) is not None)
    relationship_counts = Counter(
        str(getattr(edge, "relationship_type", "related") or "related") for edge in edges
    )

    readiness_values = [_number(getattr(node, "readiness_score", 0.0)) for node in nodes]
    potential_values = [_number(getattr(node, "convergence_potential", 0.0)) for node in nodes]

    def _node_view(node: "KnowledgeNode") -> Dict[str, Any]:
        return {
            "id": str(getattr(node, "id", "")),
            "text": str(getattr(node, "text", ""))[:280],
            "domain": str(getattr(node, "domain", "unknown") or "unknown"),
            "entity_type": str(getattr(node, "entity_type", "unknown") or "unknown"),
            "origin_kind": str(getattr(node, "origin_kind", "combinatorial") or "combinatorial"),
            "temporal_kind": str(getattr(node, "temporal_kind", "point") or "point"),
            "temporal_observations": len(getattr(node, "__dict__", {}).get("temporal_observations", []) or []),
            "temporal_density": [round(_number(value), 5) for value in (
                getattr(node, "temporal_density", []) or [])],
            "year": _year(node),
            "readiness": round(_number(getattr(node, "readiness_score", 0.0)), 6),
            "convergence_potential": round(_number(getattr(node, "convergence_potential", 0.0)), 6),
            "scientific": round(_number(getattr(node, "scientific_score", 0.0)), 6),
            "investment": round(_number(getattr(node, "investment_score", 0.0)), 6),
            "social": round(_number(getattr(node, "social_score", 0.0)), 6),
            "maturity": round(_number(getattr(node, "maturity_score", 0.0)), 6),
            "structural_dependency": round(_number(getattr(node, "structural_dependency_index", 0.0)), 6),
            "cascade_influence": round(_number(getattr(node, "cascade_influence", 0.0)), 6),
            "upstream_pressure": round(_number(getattr(node, "upstream_pressure", 0.0)), 6),
        }

    ranked_nodes = sorted(
        [node for node in nodes if not getattr(node, "is_temporal_zone", False)],
        key=lambda node: (
            _number(getattr(node, "convergence_potential", 0.0)),
            _number(getattr(node, "readiness_score", 0.0)),
            _number(getattr(node, "upstream_pressure", 0.0)),
        ),
        reverse=True,
    )[:max(1, int(top_k))]

    virtual_synth = getattr(system.get("dynamics"), "virtual_synth", None)
    phantom_gen = system.get("phantom_gen")
    return {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "nodes": len(nodes), "edges": len(edges), "zones": len(graph.zones),
            "phantoms": len(getattr(phantom_gen, "phantoms", []) or []),
            "virtual_nodes": len(getattr(virtual_synth, "virtual_nodes", []) or []),
            "family_links": sum(len(value) for value in getattr(graph, "_family_links", {}).values()) // 2,
        },
        "distributions": {
            "domains": dict(domain_counts.most_common()),
            "entity_types": dict(type_counts.most_common()),
            "origin_kinds": dict(origin_counts.most_common()),
            "temporal_kinds": dict(temporal_counts.most_common()),
            "years": dict(sorted(yearly_counts.items())),
            "relationship_types": dict(relationship_counts.most_common()),
        },
        "aggregates": {
            "mean_readiness": round(statistics.fmean(readiness_values), 6) if readiness_values else 0.0,
            "mean_convergence_potential": round(statistics.fmean(potential_values), 6) if potential_values else 0.0,
        },
        "top_nodes": [_node_view(node) for node in ranked_nodes],
        # This is deliberately emitted together with every graph snapshot,
        # rather than requiring a user to submit a separate "forecast".
        # It is the complete current map of structural convergence locations;
        # a UI may simply surface the top K of this always-updated list.
        "convergence_map": [_node_view(node) for node in ranked_nodes],
        "interpretation": (
            "Current graph-derived convergence map: structural places where an "
            "emergence is demanded or enabled by the existing graph. Scores are "
            "diagnostics, not calibrated probabilities; decision use requires "
            "temporal backtesting on independent, provenance-audited evidence."
        ),
    }


def _write_graph_summary(path: Path, system: Dict[str, Any]) -> Path:
    """Atomically write the public JSON companion to a trusted pickle snapshot."""
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    tmp_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(graph_snapshot_summary(system), f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, summary_path)
    return summary_path


def save_graph_snapshot(path, system: Dict[str, Any]) -> None:
    """
    Persist RuntimeGraph state (nodes, edges, zones, and the other
    bookkeeping RuntimeGraph carries — temporal_sorted, domain_clusters, the
    union-find family_parent map) plus phantom/virtual node state, so a
    graph built once can be reused across training resumes and inference
    calls instead of being rebuilt (or, previously, simply lost) on every
    process start.

    Uses pickle — already an established pattern in this codebase (see
    ConceptClassifier's training_data save/load) — rather than a hand-rolled
    JSON schema, since RuntimeGraph.edges is tuple-keyed and
    KnowledgeNode/KnowledgeEdge/PhantomNode/VirtualNode embeddings are numpy
    arrays: both awkward to round-trip through JSON, both trivial in pickle.

    Scope, deliberately: RuntimeGraph's own state plus
    PhantomNodeGenerator.phantoms and DynamicsOrchestrator.virtual_synth.
    virtual_nodes (exactly "nodes, edges, zones, phantom/virtual node state"
    per the TODO). Does NOT snapshot ConvergencePressureField (a pure
    function of the graph, recomputed on the first update_epoch()/
    run_epoch() call after load) or the various orchestrators' own internal
    caches (DormancyTracker, ConvergenceAccelerometer's time series,
    GraphEntropyMonitor, StructuralConflictMeter, etc.) — those rebuild from
    scratch on resume. Widening the snapshot to cover those too is a
    reasonable follow-up if losing that history across resumes turns out to
    matter in practice.

    Writes atomically (tmp file + os.replace) to match save_checkpoint()'s
    crash-safety convention in train.py.
    """
    graph         = system["graph"]
    phantom_gen   = system.get("phantom_gen")
    virtual_synth = getattr(system.get("dynamics"), "virtual_synth", None)

    payload = {
        "format_version":  GRAPH_SNAPSHOT_FORMAT_VERSION,
        "nodes":           graph.nodes,
        "edges":           graph.edges,
        "zones":           graph.zones,
        "temporal_sorted": graph.temporal_sorted,
        "domain_clusters": dict(graph.domain_clusters),
        # TODO #16: _family_parent (union-find) replaced by _family_links
        # (direct symmetric adjacency, Dict[str, Set[str]]) — see the class
        # comment on RuntimeGraph.union_family for why. Sets pickle fine.
        "family_links":    {k: set(v) for k, v in getattr(graph, "_family_links", {}).items()},
        "phantoms":        list(phantom_gen.phantoms) if phantom_gen is not None else [],
        "virtual_nodes":   list(virtual_synth.virtual_nodes) if virtual_synth is not None else [],
        # Queue reconciliation needs an idempotency receipt inside the same
        # crash-safe snapshot as the graph.  If a process dies after applying
        # a document's observations but before it can mark the external queue
        # complete, the next run sees this set and does not ingest that source
        # a second time.
        "processed_document_ids": set(getattr(graph, "processed_document_ids", set())),
        # Manifest: node.id is now the CanonicalKey (hex) everywhere — those
        # bytes only mean anything (resolve to real domain/group/entity_type
        # names) via DOMAIN_TABLE/GROUP_TABLE/ENTITY_TYPE_TABLE/ALIAS_TABLE,
        # which are process-lifetime module-level singletons, not part of
        # `graph`. Without saving them here, reloading a snapshot in a fresh
        # process would silently strand every canonical-key id: the ids
        # would still be valid dict keys in graph.nodes, but any NEW node
        # the classifier assigns after reload could collide with a
        # previously-registered domain/group meaning something different,
        # since a fresh process starts back at _DEFAULT_DOMAINS etc. Reusing
        # to_dict()/restore_from_dict() rather than re-deriving a format —
        # same JSON-safe shape save_lookup_tables() already uses, just
        # folded into this one pickle instead of a second file, so
        # --graph_snapshot stays a single, complete artifact.
        "domain_table":      DOMAIN_TABLE.to_dict(),
        "group_table":       GROUP_TABLE.to_dict(),
        "entity_type_table": ENTITY_TYPE_TABLE.to_dict(),
        # AliasTable has no to_dict()/from_dict() pair (it's plain dicts
        # with tuple keys and bytes values — not JSON-safe, but pickle
        # doesn't care) — copied directly rather than adding a new
        # serialization method that would only ever be used here.
        "alias_map":         dict(ALIAS_TABLE._map),
        "alias_domains":     {k: list(v) for k, v in ALIAS_TABLE._alias_domains.items()},
        # SEMANTIC_HASHER._pq (a trained faiss.ProductQuantizer, when
        # present) is deliberately NOT persisted: faiss is an optional
        # dependency (see "faiss not available — brute-force ANN fallback"
        # at import time) and whether its objects pickle cleanly isn't
        # something to gamble save_graph_snapshot's reliability on. A
        # reload just means the hasher re-accumulates its training buffer
        # and re-trains PQ once enough embeddings pass through again —
        # falls back to its existing SHA-256 hashing path meanwhile, the
        # same graceful degradation it already has for "faiss missing".
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    summary_path = _write_graph_summary(path, system)
    logger.info(
        f"[graph_snapshot] Saved {len(payload['nodes'])} nodes, "
        f"{len(payload['edges'])} edges, {len(payload['zones'])} zones, "
        f"{len(payload['phantoms'])} phantoms, "
        f"{len(payload['virtual_nodes'])} virtual nodes -> {path} "
        f"(summary -> {summary_path})"
    )


def load_graph_snapshot(path, system: Dict[str, Any]) -> Dict[str, int]:
    """
    Load a snapshot written by save_graph_snapshot() into system["graph"]
    (and system["phantom_gen"] / dynamics.virtual_synth, when present) IN
    PLACE, mutating the existing objects rather than replacing them —
    callers that already hold references (e.g. dynamics/topology's shared
    phantom_gen) keep working without needing to re-wire anything.

    Returns a dict of counts loaded. Raises FileNotFoundError if the path
    doesn't exist — callers that want "load if present, else start fresh"
    should check Path(path).exists() first (train.py and run.py both do).
    """
    # Pickle is deliberately used for the rich in-process graph. Do not load
    # snapshots received from an untrusted party: Python pickle can execute
    # code during deserialisation. The dashboard consumes the companion JSON
    # summary and never unpickles snapshots in response to an HTTP request.
    payload = _load_trusted_pickle(Path(path))

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid graph snapshot {path}: payload is not a mapping")
    version = payload.get("format_version")
    if version is None or int(version) > GRAPH_SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported graph snapshot format {version!r}; supported up to "
            f"{GRAPH_SNAPSHOT_FORMAT_VERSION}"
        )
    missing = {"nodes", "edges", "zones"} - set(payload)
    if missing:
        raise ValueError(f"Invalid graph snapshot {path}: missing {sorted(missing)}")

    graph = system["graph"]
    graph.nodes           = payload["nodes"]
    graph.edges           = payload["edges"]
    graph.zones           = payload["zones"]
    graph.temporal_sorted = payload.get("temporal_sorted", [])
    graph.domain_clusters = payload.get("domain_clusters", {})
    graph.processed_document_ids = set(payload.get("processed_document_ids", set()))
    if hasattr(graph, "_family_links"):
        loaded_links = payload.get("family_links")
        if loaded_links is None:
            # TODO #16: pre-migration snapshot written under the old
            # union-find "family_parent" key — no lossless conversion to
            # direct adjacency exists (union-find doesn't record which
            # specific pairs triggered each union, only the final root),
            # so we simply start family links fresh rather than guess.
            loaded_links = {}
        graph._family_links = defaultdict(set, {k: set(v) for k, v in loaded_links.items()})
    if hasattr(graph, "_rebuild_incoming_cache"):
        graph._rebuild_incoming_cache()
    if hasattr(graph, "_rebuild_adjacency"):
        graph._rebuild_adjacency()

    phantom_gen = system.get("phantom_gen")
    if phantom_gen is not None:
        phantom_gen.phantoms = payload.get("phantoms", [])

    virtual_synth = getattr(system.get("dynamics"), "virtual_synth", None)
    if virtual_synth is not None:
        virtual_synth.virtual_nodes = payload.get("virtual_nodes", [])

    # Manifest: restore the module-level canonical-key vocabulary (see the
    # matching comment in save_graph_snapshot for why this has to travel
    # with the graph, not just graph.nodes/edges). Absent entirely on a
    # pre-canonical-key snapshot — restore_from_dict()/direct dict update
    # are no-ops on missing keys, so an old snapshot still loads cleanly,
    # just without any canonical-key vocabulary (fine: nothing in it would
    # have used canonical-key ids yet either).
    if "domain_table" in payload:
        DOMAIN_TABLE.restore_from_dict(payload["domain_table"])
    if "group_table" in payload:
        GROUP_TABLE.restore_from_dict(payload["group_table"])
    if "entity_type_table" in payload:
        ENTITY_TYPE_TABLE.restore_from_dict(payload["entity_type_table"])
    if "alias_map" in payload:
        ALIAS_TABLE._map = dict(payload["alias_map"])
        ALIAS_TABLE._alias_domains = defaultdict(
            list, {k: list(v) for k, v in payload.get("alias_domains", {}).items()})

    counts = {
        "nodes":         len(graph.nodes),
        "edges":         len(graph.edges),
        "zones":         len(graph.zones),
        "phantoms":      len(payload.get("phantoms", [])),
        "virtual_nodes": len(payload.get("virtual_nodes", [])),
        "processed_documents": len(getattr(graph, "processed_document_ids", set())),
        "canonical_domains": len(payload.get("domain_table", {}).get("id2name", {})) or None,
    }
    logger.info(f"[graph_snapshot] Loaded {counts} <- {path}")
    return counts


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL WINDOW (DE)SERIALIZATION — Manifest §4, "Слепки скользящего окна"
#
#  save_graph_snapshot/load_graph_snapshot above checkpoint the single
#  always-current graph, for process resume. The pair below instead persists
#  a *slice* — RuntimeGraph.as_of(cutoff)'s output — as its own standalone
#  memory dump, so a historical state (or an intersection of several
#  windows, loaded as separate files) can be restored to RAM instantly
#  without re-walking the live graph, and historical simulations/backtests
#  run against exactly that frozen slice with no risk of a concurrent
#  ingestion mutating it mid-run.
# ══════════════════════════════════════════════════════════════════════════════

TEMPORAL_WINDOW_FORMAT_VERSION = 1


def save_temporal_window(
    path, graph: "RuntimeGraph",
    window_start: Optional[float] = None,
    window_end:   float = None,
    label: str = "",
) -> None:
    """
    Materialise graph.as_of(window_end) and write it to `path` as its own
    memory dump — a "sliding window" snapshot in the Manifest §4 sense,
    distinct from save_graph_snapshot()'s single live-graph checkpoint.

    `window_start` is optional and purely descriptive (stored in the
    payload so a caller building a *rolling* series of windows — e.g. one
    file per year — can label and later filter them); the graph reachable
    at any timestamp is, by construction, everything with
    timestamp_start <= window_end, so only `window_end` actually drives
    what gets included (as_of() has no lower bound — a node doesn't stop
    existing just because a window's display range starts later than it).

    Raises ValueError if window_end is None — a window has to be anchored
    to a moment, unlike save_graph_snapshot's "just the current state".
    """
    if window_end is None:
        raise ValueError("save_temporal_window requires window_end (a cutoff timestamp)")

    sliced = graph.as_of(window_end)
    payload = {
        "format_version": TEMPORAL_WINDOW_FORMAT_VERSION,
        "window_start":   window_start,
        "window_end":     window_end,
        "label":          label,
        "nodes":          sliced.nodes,
        "edges":          sliced.edges,
        "zones":          sliced.zones,
        "temporal_sorted": sliced.temporal_sorted,
        "domain_clusters": dict(sliced.domain_clusters),
        "family_links":    {k: set(v) for k, v in sliced._family_links.items()},
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)
    logger.info(
        f"[temporal_window] Saved window(end={window_end}, label={label!r}): "
        f"{len(payload['nodes'])} nodes, {len(payload['edges'])} edges, "
        f"{len(payload['zones'])} zones -> {path}"
    )


def load_temporal_window(path) -> "RuntimeGraph":
    """
    Load a window written by save_temporal_window() as a standalone,
    independent RuntimeGraph — does NOT touch any live system's graph
    (contrast load_graph_snapshot, which mutates system["graph"] in place).
    A caller wanting to run a backtest against a historical window builds
    its own system dict around this returned graph instead
    (build_oracle1_system-style wiring, with graph=this return value).

    Returns the reconstructed RuntimeGraph. Raises FileNotFoundError if
    `path` doesn't exist.
    """
    payload = _load_trusted_pickle(Path(path))

    window = RuntimeGraph()
    window.nodes           = payload["nodes"]
    window.edges           = payload["edges"]
    window.zones           = payload["zones"]
    window.temporal_sorted = payload.get("temporal_sorted", [])
    window.domain_clusters = payload.get("domain_clusters", {})
    window._family_links   = defaultdict(
        set, {k: set(v) for k, v in payload.get("family_links", {}).items()})
    window._rebuild_incoming_cache()
    window._rebuild_adjacency()

    logger.info(
        f"[temporal_window] Loaded window(end={payload.get('window_end')}, "
        f"label={payload.get('label', '')!r}): {len(window.nodes)} nodes, "
        f"{len(window.edges)} edges, {len(window.zones)} zones <- {path}"
    )
    return window


# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

__all__ = ['GRAPH_SNAPSHOT_FORMAT_VERSION', 'graph_snapshot_summary', '_write_graph_summary', 'save_graph_snapshot', 'load_graph_snapshot', 'TEMPORAL_WINDOW_FORMAT_VERSION', 'save_temporal_window', 'load_temporal_window']
