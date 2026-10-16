"""Runtime graph storage, ancestry checks, and traversal."""
from __future__ import annotations

from ._shared import *
from .canonical_keys import CanonicalKey, DOMAIN_TABLE, ENTITY_TYPE_TABLE, GROUP_TABLE
from .config import CONFIG, EDGE_WEIGHT_COMPONENTS, INHIBITORY_WEIGHT
from .entities import KnowledgeEdge, KnowledgeNode, NodeFeatureBuilder, TemporalZone
from .time_utils import _ts_to_year


def _safe_timestamp(value: Any) -> float:
    """Return a finite timestamp value, or zero when the value is invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0

def _investment_provenance_token(node: "KnowledgeNode") -> Optional[str]:
    """
    Origin-identity token for one node's own local investment_total_usd
    (Manifest §3, "Закон сохранения ресурса" — Provenance Tracking / the
    "Rhombus problem"): a stand-in for "which grant/round did this money
    come from", used so two different nodes that trace back to the same
    money can be recognised as sharing it rather than each counting it in
    full.

    - "investors:<sorted lead investors>" when investment_lead_investors is
      populated — real shared-origin identity, already extracted/unioned
      across observations via UNION_LIST (see NODE_FIELD_STRATEGIES). Two
      nodes citing the same funding round naturally collide on this token
      once the ingestion pipeline names investors.
    - "unlabeled:<node_id>" otherwise — scoped to this one node alone, so
      an amount with no attributable source is never assumed to be shared
      with anything else. This is the safe default: assuming a *false*
      shared origin would wrongly erase real money at a convergence point,
      which is worse than the conservative "not proven shared, so count it
      here" fallback.

    Returns None when the node has no recorded investment at all (nothing
    to tag).
    """
    amt = float(getattr(node, "investment_total_usd", 0.0))
    if amt <= 0.0:
        return None
    investors = getattr(node, "investment_lead_investors", None)
    if investors:
        cleaned = sorted({inv.strip().lower() for inv in investors if inv and inv.strip()})
        if cleaned:
            return "investors:" + "|".join(cleaned)
    return f"unlabeled:{node.id}"


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME GRAPH
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  ANCESTRY ENFORCER
#
#  "Discoveries don't arise from nothing — they are reorganisations,
#   mutations, and combinations of existing knowledge."
#
#  AncestryEnforcer implements this as a hard invariant on the graph:
#   • The N oldest nodes (by timestamp) are "primordial roots" — exempt,
#     because their true predecessors are simply outside our corpus.
#   • Every other node must have at least one incoming non-containment edge
#     before entering RuntimeGraph.
#   • When a batch of nodes is added without a sufficient incoming edge
#     (common during ingestion when a document mentions a concept for the
#     first time), the enforcer synthesises a lightweight "DERIVED_FROM" edge
#     to the best-matching existing predecessor.
#
#  Three matching strategies, in priority order:
#   1. Embedding cosine similarity (if both nodes have embeddings and
#      similarity ≥ ancestry_synthesiser_sem_threshold).
#   2. Domain affinity: highest (scientific_score + strategic_value) among
#      nodes in the same domain that predate the new node.
#   3. Global fallback: highest (scientific_score + strategic_value) among
#      ANY node that predates the new node.
#
#  The synthesised edge is tagged with relationship_type="DERIVED_FROM" and
#  discovery_method="ancestry_synthesised" so downstream code can distinguish
#  it from evidence-backed edges extracted from documents.  Its component
#  weights are kept modest so it doesn't dominate GNN attention.
# ══════════════════════════════════════════════════════════════════════════════

class AncestryEnforcer:
    """Enforces the ancestry invariant when nodes are added to RuntimeGraph.

    Usage — called automatically by RuntimeGraph methods; no manual
    instantiation needed.  The singleton ANCESTRY_ENFORCER is created at
    module level and shared across the session.
    """

    def is_primordial(self, node: "KnowledgeNode", graph: "RuntimeGraph") -> bool:
        """True if this node is among the N temporally earliest — exempt from the invariant.

        temporal_sorted is maintained in ascending order, so we read the first
        primordial_cutoff_n entries. Temporal-zone nodes are also exempt because
        they are structural containers, not knowledge claims. Axiom nodes
        (temporal_kind == "axiom" — undated ideas, myths, mathematical facts;
        see NodeEdgeAssembler.assemble_node) are exempt for a different
        reason: they are foundational by nature, not "primordial" merely
        because a missing date happens to sort as 1970. Without this, an
        axiom node could only satisfy the ancestry invariant by being
        assigned a predecessor from find_best_predecessor — which, for a
        timestamp-less node, would otherwise search the *entire* graph
        including nodes dated decades later than the axiom itself.
        """
        if node.is_temporal_zone or getattr(node, "temporal_kind", "point") == "axiom":
            return True
        n = CONFIG["primordial_cutoff_n"]
        primordial_ids = {nid for _, nid in graph.temporal_sorted[:n]}
        return node.id in primordial_ids

    def has_incoming(self, node_id: str, graph: "RuntimeGraph") -> bool:
        """True if node_id already has at least one incoming non-containment edge.

        Delegates to RuntimeGraph's O(1) in-degree cache instead of scanning
        all edges (which was O(E) and catastrophic at scale).
        """
        return graph.has_incoming_edge(node_id)

    def has_incoming_in_batch(
        self,
        node_id: str,
        batch_edges: List["KnowledgeEdge"],
    ) -> bool:
        """True if any edge in the current batch points TO node_id (non-containment)."""
        return any(
            e.target_id == node_id and not e.is_containment_edge
            for e in batch_edges
        )

    def find_best_predecessor(
        self,
        node: "KnowledgeNode",
        graph: "RuntimeGraph",
    ) -> Optional["KnowledgeNode"]:
        """Return the best predecessor node to anchor the new node to.

        Predecessor = already in graph AND (timestamp < node.timestamp OR
        node.timestamp == 0).  Returns None if the graph is empty.

        Scalability: uses graph.temporal_sorted (always-sorted list) with
        bisect to locate the eligible prefix in O(log N), then heapq.nlargest
        to find the top-scored candidates in O(P log K) where P is the
        eligible count and K = ancestry_synthesiser_candidates — never
        materialising the full predecessor list into memory.
        """
        if not graph.temporal_sorted:
            return None

        # An axiom has no real date to anchor a predecessor search with —
        # falling through to the "timestamp == 0.0" branch below would treat
        # every node currently in the graph, including ones dated decades
        # after the axiom itself, as an eligible "predecessor" and hand it
        # to whichever one scores highest. is_primordial() already exempts
        # axiom nodes from needing a predecessor at all, so this search
        # shouldn't run for them in the first place.
        if getattr(node, "temporal_kind", "point") == "axiom":
            return None

        max_cands = CONFIG["ancestry_synthesiser_candidates"]

        if node.timestamp == 0.0:
            # All existing nodes are eligible predecessors
            eligible_end = len(graph.temporal_sorted)
        else:
            # Binary search: find insertion point of node.timestamp
            # Everything to the left has timestamp <= node.timestamp
            eligible_end = bisect.bisect_right(
                graph.temporal_sorted, (node.timestamp, "")
            )
            if eligible_end == 0:
                return None

        # Use heapq.nlargest to avoid materialising all eligible candidates.
        # Score function: scientific_score + strategic_value (same as before).
        # We stream from temporal_sorted[0:eligible_end] to stay O(P log K).
        def _score_and_node(idx: int):
            _, nid = graph.temporal_sorted[idx]
            if nid == node.id:
                return None
            n = graph.nodes.get(nid)
            if n is None or n.is_temporal_zone:
                return None
            return (n.scientific_score + n.strategic_value, n)

        top_scored = heapq.nlargest(
            max_cands,
            filter(None, (_score_and_node(i) for i in range(eligible_end))),
            key=lambda x: x[0],
        )
        if not top_scored:
            return None
        candidates = [n for _, n in top_scored]

        threshold = CONFIG["ancestry_synthesiser_sem_threshold"]
        best: Optional["KnowledgeNode"] = None
        best_sim = -1.0

        # Strategy 1: embedding cosine similarity
        if node.embedding is not None:
            new_norm = float(np.linalg.norm(node.embedding))
            if new_norm > 1e-8:
                for cand in candidates:
                    if cand.embedding is None:
                        continue
                    c_norm = float(np.linalg.norm(cand.embedding))
                    if c_norm < 1e-8:
                        continue
                    sim = float(np.dot(node.embedding, cand.embedding) / (new_norm * c_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best = cand

        if best is not None and best_sim >= threshold:
            return best

        # Strategy 2: same-domain affinity
        domain_cands = [c for c in candidates if c.domain == node.domain]
        if domain_cands:
            return max(domain_cands, key=lambda n: n.scientific_score + n.strategic_value)

        # Strategy 3: global best-influence fallback
        return max(candidates, key=lambda n: n.scientific_score + n.strategic_value)

    def synthesise_edge(
        self,
        predecessor: "KnowledgeNode",
        new_node: "KnowledgeNode",
        similarity: float = 0.0,
    ) -> "KnowledgeEdge":
        """Build a synthetic DERIVED_FROM edge.

        Uses fixed modest weights so the synthesised lineage does not drown out
        real evidence-backed edges during GNN training.  The edge is tagged
        with discovery_method="ancestry_synthesised" for auditability.
        """
        sem_sim = float(np.clip(similarity, 0.0, 1.0)) if similarity >= CONFIG["ancestry_synthesiser_sem_threshold"] else CONFIG["ancestry_edge_semantic_sim"]
        edge = KnowledgeEdge(
            id=f"anc_{predecessor.id[:10]}_{new_node.id[:10]}",
            source_id=predecessor.id,
            target_id=new_node.id,
            relationship_type="DERIVED_FROM",
            discovery_method="ancestry_synthesised",
            semantic_similarity=sem_sim,
            temporal_proximity=CONFIG["ancestry_edge_temporal_prox"],
            limitation_resolution=0.0,
            citation_link=0.0,
            investment_correlation=0.0,
            social_correlation=0.0,
            inhibitory_force=0.0,
            timestamp=new_node.timestamp,
            confidence=float(np.clip(similarity if similarity > 0 else 0.3, 0.1, 0.9)),
            evidence=["auto-synthesised ancestry lineage"],
        )
        edge.compute_total_weight()
        return edge

    def enforce(
        self,
        node: "KnowledgeNode",
        graph: "RuntimeGraph",
        batch_edges: Optional[List["KnowledgeEdge"]] = None,
    ) -> Optional["KnowledgeEdge"]:
        """Check the ancestry invariant for *node* and return a synthetic edge if needed.

        Args:
            node:        The node being added.
            graph:       The RuntimeGraph (must already contain the node so that
                         is_primordial() works with the updated temporal_sorted).
            batch_edges: Edges being added in the same call (prevents false
                         positives when a source and target arrive together).

        Returns:
            A synthetic KnowledgeEdge if one was synthesised, else None.
            The caller is responsible for adding this edge to the graph.
        """
        # Temporal zones are structural — exempt
        if node.is_temporal_zone:
            return None

        # Primordial roots are exempt (N oldest by timestamp)
        if self.is_primordial(node, graph):
            return None

        # Incoming edge already in graph?
        if self.has_incoming(node.id, graph):
            return None

        # Incoming edge in same batch?
        if batch_edges and self.has_incoming_in_batch(node.id, batch_edges):
            return None

        # No incoming edge found — find best predecessor and synthesise
        predecessor = self.find_best_predecessor(node, graph)
        if predecessor is None:
            # Empty graph or no earlier node — node becomes an additional root
            logger.info(
                f"[Ancestry] '{node.text[:50]}' has no eligible predecessors — "
                "admitted as additional primordial root"
            )
            return None

        # Compute similarity for the edge weight (if embeddings available)
        similarity = 0.0
        if node.embedding is not None and predecessor.embedding is not None:
            nn = float(np.linalg.norm(node.embedding))
            pn = float(np.linalg.norm(predecessor.embedding))
            if nn > 1e-8 and pn > 1e-8:
                similarity = float(np.dot(node.embedding, predecessor.embedding) / (nn * pn))

        edge = self.synthesise_edge(predecessor, node, similarity)
        logger.info(
            f"[Ancestry] Synthesised DERIVED_FROM: "
            f"'{predecessor.text[:40]}' → '{node.text[:40]}' "
            f"(sim={similarity:.3f})"
        )
        return edge


ANCESTRY_ENFORCER = AncestryEnforcer()


class RuntimeGraph:
    def __init__(self):
        self.nodes:           Dict[str, KnowledgeNode]            = {}
        self.edges:           Dict[Tuple[str,str], KnowledgeEdge] = {}
        self.zones:           Dict[str, TemporalZone]             = {}
        self.temporal_sorted: List[Tuple[float, str]]             = []
        self.domain_clusters: Dict[str, List[str]]                = {}
        # Direct symmetric adjacency for "idea families" — near-duplicate
        # nodes just below the hard merge threshold (see BucketIndex.
        # FAMILY_THRESHOLD) that stay separate nodes (unique details
        # preserved) but whose epistemic_mass should be readable as one
        # aggregate "iceberg" rather than fragments. See union_family /
        # family_members / family_epistemic_mass below.
        #
        # NOT a union-find partition (deliberately — see the long comment on
        # union_family): a node can sit in several, only-loosely-related
        # families at once (A resembles B for one reason and C for an
        # unrelated reason, without B and C resembling each other), and
        # family_members() only walks a bounded number of hops rather than
        # the full transitive closure, so one promiscuous hub node can't
        # silently drag the whole connected component into one "family".
        self._family_links: Dict[str, Set[str]] = defaultdict(set)

        # ── Canonical Key bitmask index (Manifest §1) ─────────────────────────
        # "Идентификация только через Canonical Key... это позволяет мгновенно
        # фильтровать граф на миллиарды узлов битовыми масками (например,
        # найти все узлы типа 'физический предел' в домене 'оптика' — это
        # побитовая операция O(1), а не семантический поиск)." Every node
        # assembled by CanonicalNodeAssembler carries a 12-byte canonical_key
        # (domain_id/group_id/entity_type_id/sem_hash — see CanonicalKey)
        # stamped into node.__dict__["canonical_key"]; these three dicts are
        # the actual "bitmask" index over those fields — domain_id/group_id/
        # entity_type_id -> set of node ids — so query_by_canonical_mask()
        # below is always a handful of dict lookups + set intersections,
        # never a scan of self.nodes and never an embedding/FAISS query.
        # _canon_key_of caches each indexed node's own (d, g, e) triple so a
        # later re-index (e.g. sem_hash gets refined once a real embedding
        # arrives) or removal can deindex in O(1) instead of scanning every
        # bucket. Nodes with no canonical_key (phantom/zone nodes) simply
        # never appear here — they were never reachable by "search the
        # graph by canonical key" in the first place, so that's not a
        # regression.
        self._canon_domain_idx: Dict[int, Set[str]]           = defaultdict(set)
        self._canon_group_idx:  Dict[int, Set[str]]           = defaultdict(set)
        self._canon_entity_idx: Dict[int, Set[str]]           = defaultdict(set)
        self._canon_key_of:     Dict[str, Tuple[int, int, int]] = {}
        # Content identities of documents whose observations have been
        # committed to this graph.  The external document queue reconciles
        # against this receipt after a crash, making source ingestion
        # idempotent across graph snapshots rather than merely per process.
        self.processed_document_ids: Set[str] = set()

        # ── Adjacency index (TODO #20) ─────────────────────────────────────
        # node_id -> set of edge keys (src, tgt) touching it, either side.
        # get_neighbours()/k_hop_neighborhood()/propagate_wave() used to scan
        # ALL of self.edges on every call — O(E) per call regardless of how
        # local the query actually was, which cannot work at the "billions
        # of nodes" scale the manifest targets. Maintained incrementally by
        # add_edge()/remove_node() below so it never needs a full rebuild in
        # normal operation (only _rebuild_adjacency() after a raw/pickled
        # graph load, same pattern as _rebuild_incoming_cache()).
        self._adjacency: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

    # Incremental in-degree cache: set of node IDs that have ≥1 incoming
    # non-containment edge.  Updated by add_edge(); read by AncestryEnforcer.
    # Initialised lazily in _ensure_cache() because __init__ runs before
    # _incoming_ids is declared.
    def _ensure_cache(self):
        if not hasattr(self, "_incoming_ids"):
            self._incoming_ids: set = set()

    def _mark_has_incoming(self, node_id: str) -> None:
        self._ensure_cache()
        self._incoming_ids.add(node_id)

    def has_incoming_edge(self, node_id: str) -> bool:
        """True if node_id has at least one incoming non-containment edge (O(1) via cache)."""
        self._ensure_cache()
        return node_id in self._incoming_ids

    # ── Core mutation API ─────────────────────────────────────────────────────

    def add_node(self, node: KnowledgeNode):
        """Add a node with no ancestry check. Use add_node_checked() for enforcement."""
        self.nodes[node.id] = node
        # bisect.insort keeps temporal_sorted in O(log N) instead of O(N log N) sort()
        bisect.insort(self.temporal_sorted, (node.timestamp, node.id))
        self.domain_clusters.setdefault(node.domain, []).append(node.id)
        self._index_canonical_key(node)

    # ── Canonical Key bitmask index (Manifest §1) ───────────────────────────
    def _index_canonical_key(self, node: KnowledgeNode) -> None:
        """(Re-)register *node* in the canonical-key bitmask index. No-op if
        the node was never assembled with a canonical_key. Safe to call
        repeatedly for the same node (e.g. once on ingest, again after
        GraphConsolidator refines its sem_hash with a real embedding) —
        de-indexes the stale entry first via _canon_key_of."""
        raw = node.__dict__.get("canonical_key")
        if not raw or len(raw) != 12:
            return
        self._deindex_canonical_key(node.id)
        ck = CanonicalKey.from_bytes(raw)
        self._canon_domain_idx[ck.domain_id].add(node.id)
        self._canon_group_idx[ck.group_id].add(node.id)
        self._canon_entity_idx[ck.entity_type_id].add(node.id)
        self._canon_key_of[node.id] = (ck.domain_id, ck.group_id, ck.entity_type_id)

    def _deindex_canonical_key(self, node_id: str) -> None:
        """Remove node_id from all three canonical-key index buckets, O(1)
        via the cached (domain_id, group_id, entity_type_id) triple."""
        prev = self._canon_key_of.pop(node_id, None)
        if prev is None:
            return
        d, g, e = prev
        self._canon_domain_idx[d].discard(node_id)
        self._canon_group_idx[g].discard(node_id)
        self._canon_entity_idx[e].discard(node_id)

    def query_by_canonical_mask(self, domain_id: Optional[int] = None,
                                 group_id: Optional[int] = None,
                                 entity_type_id: Optional[int] = None) -> Set[str]:
        """
        Manifest §1: node lookup through the CanonicalKey's fields, not
        through text or embedding search — "найти все узлы типа X в домене
        Y — это побитовая операция O(1)". Returns every node id whose
        canonical_key matches ALL of the given (non-None) fields: a few
        dict lookups against the index built by _index_canonical_key() plus
        set intersection, never a scan of self.nodes. Passing no arguments
        returns an empty set rather than "everything" — a full-graph query
        is a different, deliberate operation, not an accidental default.
        """
        parts: List[Set[str]] = []
        if domain_id is not None:
            parts.append(self._canon_domain_idx.get(domain_id, set()))
        if group_id is not None:
            parts.append(self._canon_group_idx.get(group_id, set()))
        if entity_type_id is not None:
            parts.append(self._canon_entity_idx.get(entity_type_id, set()))
        if not parts:
            return set()
        result = set(parts[0])
        for s in parts[1:]:
            result &= s
        return result

    def query_by_canonical_names(self, domain: Optional[str] = None,
                                  group: Optional[str] = None,
                                  entity_type: Optional[str] = None) -> Set[str]:
        """Human-readable convenience wrapper around query_by_canonical_mask()
        — this is literally the manifest's own worked example: "найти все
        узлы типа 'физический предел' в домене 'оптика'" resolves through
        the same DOMAIN_TABLE/GROUP_TABLE/ENTITY_TYPE_TABLE used at
        classification time (ConceptClassifier), then does the O(1) lookup."""
        domain_id = DOMAIN_TABLE.get_id(domain) if domain is not None else None
        group_id = GROUP_TABLE.get_id(group) if group is not None else None
        entity_type_id = (ENTITY_TYPE_TABLE.get_id(entity_type)
                           if entity_type is not None else None)
        return self.query_by_canonical_mask(
            domain_id=domain_id, group_id=group_id, entity_type_id=entity_type_id)

    def add_node_checked(
        self,
        node: KnowledgeNode,
        batch_edges: Optional[List["KnowledgeEdge"]] = None,
    ) -> Optional["KnowledgeEdge"]:
        """Add a node and enforce the ancestry invariant.

        If the node has no incoming edge (neither in the graph nor in
        *batch_edges*) and is not a primordial root, AncestryEnforcer
        synthesises a DERIVED_FROM edge to the best-matching predecessor.

        The synthesised edge — if any — is added to the graph automatically.
        It is also returned so the caller can log or inspect it.

        Call this instead of add_node() whenever you're loading nodes that
        must obey the "discoveries don't arise from nothing" invariant.
        """
        self.add_node(node)  # register first so is_primordial() sees it in temporal_sorted
        anchor = ANCESTRY_ENFORCER.enforce(node, self, batch_edges)
        if anchor is not None:
            self.add_edge(anchor)   # add to graph and update cache
        return anchor

    def add_edge(self, edge: KnowledgeEdge):
        key = (edge.source_id, edge.target_id)
        self.edges[key] = edge
        # Keep in-degree cache current so has_incoming_edge() is always O(1)
        if not edge.is_containment_edge:
            self._mark_has_incoming(edge.target_id)
        # Adjacency index (TODO #20) — see __init__ comment. Both endpoints
        # index the same edge key so get_neighbours()/k_hop_neighborhood()/
        # propagate_wave() can look up "edges touching this node" in O(1)
        # regardless of graph size, instead of scanning self.edges.
        self._adjacency[edge.source_id].add(key)
        self._adjacency[edge.target_id].add(key)

    def _deindex_edge(self, key: Tuple[str, str]) -> None:
        """Remove one edge key from the adjacency index on both endpoints.

        Call this everywhere an edge is deleted from self.edges directly
        (i.e. anywhere that doesn't go through add_edge) — currently only
        GraphConsolidator._replace_edges_in_graph — so _adjacency never
        drifts from the true edge set.
        """
        src, tgt = key
        self._adjacency.get(src, set()).discard(key)
        self._adjacency.get(tgt, set()).discard(key)

    def _rebuild_adjacency(self) -> None:
        """Recompute _adjacency from scratch.

        Needed after loading a raw/pickled graph (load_graph_snapshot etc.)
        or any bulk assignment to self.edges that bypassed add_edge() —
        mirrors _rebuild_incoming_cache().
        """
        self._adjacency = defaultdict(set)
        for key in self.edges:
            self._adjacency[key[0]].add(key)
            self._adjacency[key[1]].add(key)

    def add_zone(self, zone: TemporalZone):
        self.zones[zone.id] = zone
        # A newly-added zone can retroactively cover nodes that already
        # exist in the graph (Manifest §2, "Retroactive connectivity" —
        # loading a zone is itself a kind of ingestion event). Sweep
        # membership immediately so callers don't have to remember to.
        self.recompute_zone_membership(zone_ids=[zone.id])

    # ── Temporal zones as FIELDS, not semantic clusters ─────────────────────
    # Manifest §1: "Temporal zones are fields, not nodes: a zone ... is a
    # spatio-temporal region in the graph. Any node whose coordinates
    # (domain + time) intersect this zone automatically comes under its
    # gravity ... Zones can overlap, compounding the effect." The previous
    # implementation (ContinuousFeedEngine._detect_zone_membership) matched
    # nodes to zones by embedding cosine-similarity against the zone's
    # centroid — a semantic judgment, and single-valued (one "best" zone
    # per node even when several matched). Neither matches the manifest:
    # membership here is a pure (domain, timestamp) coordinate test via
    # TemporalZone.covers(), and a node keeps every zone that covers it.

    def zones_covering_node(self, node_id: str) -> List[TemporalZone]:
        """Every zone whose (domain_focus, [start, end]) field covers this
        node's own (domain, timestamp) coordinates. Pure geometry — no
        embeddings involved."""
        node = self.nodes.get(node_id)
        if node is None or node.is_temporal_zone:
            return []
        return [z for z in self.zones.values()
                if z.covers(node.domain, node.timestamp)]

    @staticmethod
    def combined_zone_multiplier(zones: List[TemporalZone]) -> float:
        """
        Compose the acceleration multiplier of several overlapping zones.

        Each zone_multiplier is a "how many times faster than baseline"
        factor (>= 1.0). Multiplying them (rather than taking max()) is
        what makes overlap actually *compound* per the manifest, instead of
        collapsing back to whichever single zone is strongest. Combining
        multiple independent acceleration effects multiplicatively is the
        standard way to stack rate factors (e.g. compounding growth rates);
        summing the excess above 1.0 would work too but multiplication is
        what "compounding" means literally here. Clipped to
        CONFIG["zone_multiplier_max"] so an arbitrary pile-up of zones
        can't blow the feature scale (NodeFeatureBuilder divides this by 5).
        """
        if not zones:
            return 1.0
        combined = 1.0
        for z in zones:
            combined *= max(1.0, float(z.zone_multiplier))
        cap = CONFIG.get("zone_multiplier_max", 5.0)
        return float(np.clip(combined, 1.0, cap))

    def recompute_zone_membership(
        self, node_ids: Optional[List[str]] = None,
        zone_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Sweep geometric zone membership and write it back onto both sides:

          - TemporalZone.contained_node_ids gets every node whose
            coordinates fall in the field (adds; never silently drops a
            node added some other way).
          - KnowledgeNode.zone_ids gets the full list of covering zones.
          - KnowledgeNode.zone_id / acceleration_multiplier are kept as the
            derived single-valued projection every existing scalar reader
            (NodeFeatureBuilder, PathStep, ConvergencePressureField) still
            expects: zone_id = the *strongest* covering zone (highest
            zone_multiplier, ties broken by id for determinism);
            acceleration_multiplier = combined_zone_multiplier() across all
            covering zones, so pressure/feature code sees the compounded
            effect even though it only reads one scalar.

        Pass node_ids and/or zone_ids to restrict the sweep (e.g. just the
        nodes/zones touched by the latest ingestion batch) instead of
        rescanning the whole graph; omit both for a full sweep.

        Returns {zone_id: [newly added member node_ids]}, mirroring the
        return shape of the old _detect_zone_membership so callers that
        used it for affected-node propagation (ContinuousFeedEngine.feed)
        keep working unchanged.
        """
        candidate_zones = (
            [self.zones[z] for z in zone_ids if z in self.zones]
            if zone_ids is not None else list(self.zones.values())
        )
        candidate_node_ids = (
            [n for n in node_ids if n in self.nodes]
            if node_ids is not None else list(self.nodes.keys())
        )

        zone_updates: Dict[str, List[str]] = {}
        for nid in candidate_node_ids:
            node = self.nodes[nid]
            if node.is_temporal_zone:
                continue
            covering = [z for z in candidate_zones if z.covers(node.domain, node.timestamp)]
            if zone_ids is not None:
                # Restricted sweep: only touching a subset of zones, so
                # merge into the node's existing membership rather than
                # replacing it outright (other zones not in this sweep may
                # already cover it).
                existing = {z for z in node.zone_ids if z not in {c.id for c in covering}}
                new_ids = {z.id for z in covering}
                node.zone_ids = sorted(existing | new_ids)
            else:
                node.zone_ids = sorted(z.id for z in covering)

            for z in covering:
                if nid not in z.contained_node_ids:
                    z.contained_node_ids.append(nid)
                    zone_updates.setdefault(z.id, []).append(nid)

            all_covering = [self.zones[zid] for zid in node.zone_ids if zid in self.zones]
            if all_covering:
                best = max(all_covering, key=lambda z: (z.zone_multiplier, z.id))
                node.zone_id = best.id
                node.acceleration_multiplier = self.combined_zone_multiplier(all_covering)
            else:
                node.zone_id = ""
                node.acceleration_multiplier = 1.0
        return zone_updates

    def remove_node(self, node_id: str) -> None:
        """Remove a node and clean up ALL associated indices.

        Removes from: nodes dict, temporal_sorted, domain_clusters,
        and rebuilds _incoming_ids from the remaining edges so that
        AncestryEnforcer.has_incoming_edge() never refers to a dead node.

        Callers must remap/delete all edges touching this node first.
        """
        node = self.nodes.pop(node_id, None)
        # Remove from temporal_sorted (maintains sorted order, just filters)
        self.temporal_sorted = [
            (ts, nid) for ts, nid in self.temporal_sorted if nid != node_id
        ]
        # Remove from domain_clusters
        domain = getattr(node, "domain", None) if node else None
        if domain and domain in self.domain_clusters:
            try:
                self.domain_clusters[domain].remove(node_id)
            except ValueError:
                pass
        else:
            # Slower fallback: scan all clusters
            for id_list in self.domain_clusters.values():
                if node_id in id_list:
                    id_list.remove(node_id)
                    break
        # Rebuild _incoming_ids so the cache is consistent with remaining edges
        self._rebuild_incoming_cache()
        self._adjacency.pop(node_id, None)

    def _rebuild_incoming_cache(self) -> None:
        """Recompute _incoming_ids from scratch.

        Must be called after any operation that adds, removes, or remaps edges
        so that has_incoming_edge() always reflects the true graph state.
        """
        self._ensure_cache()
        self._incoming_ids = {
            tgt
            for (src, tgt), edge in self.edges.items()
            if not edge.is_containment_edge
        }

    def get_node(self, nid: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(nid)

    def get_edge(self, src: str, tgt: str) -> Optional[KnowledgeEdge]:
        return self.edges.get((src, tgt))

    def get_all_zones(self) -> List[TemporalZone]:
        return list(self.zones.values())

    def zone_centroid(self, zone_id: str) -> Optional[np.ndarray]:
        zone = self.zones.get(zone_id)
        if not zone: return None
        embs = [self.nodes[nid].embedding
                for nid in zone.contained_node_ids
                if nid in self.nodes and self.nodes[nid].embedding is not None]
        return np.mean(embs, axis=0) if embs else None

    # ── Idea families (soft/hierarchical consolidation, "iceberg") ──────────
    # GraphConsolidator's merge decision is binary by necessity (either two
    # observations are the same canonical node or they are not — see
    # BucketIndex.MERGE_THRESHOLD). But a near-miss just below that
    # threshold (BucketIndex.FAMILY_THRESHOLD) is not noise — it is very
    # often the same underlying thought stated slightly differently, and
    # collapsing thousands of such near-duplicates into a single node would
    # destroy the unique detail each one carries. Families let both be true
    # at once: nodes stay separate (detail preserved), but their
    # epistemic_mass reads as one aggregate "iceberg" for anything that
    # wants to know how much the underlying idea, not the individual
    # phrasing, actually weighs.
    #
    # Deliberately NOT union-find. A strict partition (every node has
    # exactly one family root) is the wrong model here for two concrete
    # reasons:
    #   1. "Resembles idea X" isn't transitive in practice. FAMILY_THRESHOLD
    #      is a single embedding-similarity cutoff, noisy enough that
    #      A~B and B~C above threshold does not reliably mean A~C too — A
    #      and C can be genuinely different ideas that both happen to
    #      resemble a common "hub" phrasing B for different surface
    #      reasons. Union-find has no choice but to force A, B, C into one
    #      family regardless (this is the classic "chaining" failure mode
    #      of single-linkage clustering); direct adjacency plus a bounded
    #      traversal (family_traversal_max_depth below) contains the damage
    #      instead of letting one promiscuous hub silently merge unrelated
    #      clusters at graph scale.
    #   2. A node legitimately belongs to several, mutually unrelated
    #      families at once — e.g. node A can be a near-duplicate of B for
    #      one reason and, completely separately, a near-duplicate of C for
    #      another, without B and C resembling each other at all. Union-find
    #      cannot represent that (by construction, it's a disjoint-set
    #      structure — one root per node); direct adjacency naturally can.
    #
    # _family_links is keyed by node_id -> its DIRECT family neighbours, so
    # "this node's family" is an O(degree) lookup, not an O(every node ever
    # linked) scan — the graph carries the index, nothing needs to be
    # denormalised onto individual KnowledgeNode objects (which get
    # reconstructed on every NodeConsolidator.consolidate() call anyway, so
    # anything stored ON a node would need re-syncing on every consolidation
    # — storing it on the graph, keyed by id, sidesteps that entirely).
    def union_family(self, a: str, b: str) -> None:
        """Record a and b as directly family-linked (same underlying idea).
        Symmetric; safe to call repeatedly (idempotent — a set, not a list).
        """
        if a == b:
            return
        self._family_links[a].add(b)
        self._family_links[b].add(a)

    def family_members(self, node_id: str,
                       max_depth: Optional[int] = None) -> List[str]:
        """node_id plus every node reachable within max_depth hops of direct
        family links (bounded BFS — see the class comment above for why this
        is bounded rather than a full transitive closure). Defaults to
        CONFIG["family_traversal_max_depth"]. Returns [node_id] (just
        itself) if it has no family links at all — the common case."""
        if max_depth is None:
            max_depth = CONFIG.get("family_traversal_max_depth", 2)
        seen = {node_id}
        frontier = [node_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for cur in frontier:
                for nb in self._family_links.get(cur, ()):
                    if nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
            depth += 1
        return list(seen)

    def family_epistemic_mass(self, node_id: str,
                              max_depth: Optional[int] = None) -> float:
        """
        Aggregate epistemic_mass across node_id's family within max_depth
        hops — the "iceberg" reading. Falls back to the node's own
        epistemic_mass if it has no family (the common case: most nodes are
        singletons).
        """
        members = self.family_members(node_id, max_depth=max_depth)
        if len(members) <= 1:
            node = self.nodes.get(node_id)
            return float(getattr(node, "epistemic_mass", 0.0)) if node else 0.0
        total = 0.0
        for nid in members:
            node = self.nodes.get(nid)
            if node is not None:
                total += float(getattr(node, "epistemic_mass", 0.0))
        return total

    # ── Explicit relation-type slice (node × node × relation "tensor") ──────
    def relation_adjacency(
        self, component: str = "max"
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Undirected adjacency restricted to one edge component — one named
        slice of the node × node × relation structure (see README, "How
        Oracle Differs from LLMs" and the 7 EXTENDED_EDGE_COMPONENTS).

        component: one of EXTENDED_EDGE_COMPONENTS, or "max" (default) for
        the strongest of the 6 *conductive* components at each edge, damped
        by inhibitory_force: cond = max(6 components) * (1 - inhibitory_force).
        inhibitory_force never competes inside the max() itself — it is a
        resistive signal, not a conductive one (Principle 4, Institutional
        Friction), so a high value there must never be mistaken for a strong
        push. But it still has to act on the result: an edge that incumbents
        have built a strong inhibitory_force against should behave like a
        wall, not a transparent bystander — a wave that would otherwise be
        "positive" needs to actually weaken or stop there, not sail through
        at full strength just because semantic_similarity happens to be high.

        This is the single source of truth for "how strongly does A connect
        to B on this particular kind of relation" — CascadeAnalyzer and any
        future per-relation lookup should read it from here rather than
        rebuilding their own scan of self.edges.
        """
        adjacency: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for (src, tgt), edge in self.edges.items():
            if edge.is_containment_edge:
                continue
            if component == "max":
                w = max(getattr(edge, c) for c in EDGE_WEIGHT_COMPONENTS)
                w *= max(0.0, 1.0 - edge.inhibitory_force)
            else:
                w = getattr(edge, component, 0.0)
            if w <= 0.0:
                continue
            adjacency[src].append((tgt, w))
            adjacency[tgt].append((src, w))
        return adjacency

    # ── Type index (already maintained via domain_clusters) + measured
    #    zone strength, replacing the LLM-guessed zone_multiplier with an
    #    empirical citation_velocity_ratio once enough evidence exists ──────
    def domain_baseline_velocity(
        self, domain: str, exclude_ids: Optional[Set[str]] = None
    ) -> float:
        """
        Mean InfluenceTimeline.velocity across reinforced (non-zone) nodes in
        *domain*, using the existing domain_clusters index rather than a
        full graph scan. Nodes that have never been reinforced (no
        epistemic_timeline yet) don't count — they would only dilute the
        baseline toward zero without saying anything about typical velocity.

        `exclude_ids` matters: pass the zone's own members when computing a
        zone's citation_velocity_ratio (see measured_zone_multiplier). If the
        baseline included the very nodes being measured against it, the
        average ratio would be mathematically forced toward 1.0 regardless
        of how fast the zone is actually accelerating — the baseline has to
        represent the rest of the domain, not the zone itself.

        Returns 0.0 if no qualifying node exists in the domain yet.
        """
        exclude_ids = exclude_ids or set()
        ids = self.domain_clusters.get(domain, [])
        velocities = []
        for nid in ids:
            if nid in exclude_ids:
                continue
            node = self.nodes.get(nid)
            if node is None or node.is_temporal_zone:
                continue
            tl_summary = node.__dict__.get("epistemic_timeline")
            if tl_summary is None:
                continue
            velocities.append(node.influence_velocity)
        return float(np.mean(velocities)) if velocities else 0.0

    def measured_zone_multiplier(
        self, zone: "TemporalZone", min_members: int = 3
    ) -> Optional[float]:
        """
        citation_velocity_ratio-based zone_multiplier: for each member with
        a populated timeline, ratio = its velocity / the domain baseline
        computed over the *rest* of that domain (its own zone siblings
        excluded — see domain_baseline_velocity), averaged across members
        and clipped to [1.0, 5.0] (same range the LLM-guessed value was
        already clipped to). Returns None when fewer than `min_members`
        qualifying members exist yet — too little evidence to override the
        LLM's initial estimate (cold start).
        """
        member_ids = set(zone.contained_node_ids)
        ratios = []
        for nid in zone.contained_node_ids:
            node = self.nodes.get(nid)
            if node is None or node.is_temporal_zone:
                continue
            tl_summary = node.__dict__.get("epistemic_timeline")
            if tl_summary is None:
                continue
            baseline = self.domain_baseline_velocity(node.domain, exclude_ids=member_ids)
            if baseline <= 1e-9:
                continue
            ratios.append(node.influence_velocity / baseline)
        if len(ratios) < min_members:
            return None
        return float(np.clip(np.mean(ratios), 1.0, 5.0))

    def lineage_ancestors(self, node_id: str, max_depth: Optional[int] = None) -> FrozenSet[str]:
        """
        node_id plus every node reachable by walking backward along
        lineage-carrying edges (DERIVED_FROM/ENABLES/CAUSED_BY, or whatever
        CONFIG["provenance_ancestry_relationship_types"] names) up to
        max_depth hops. This is the single-node version of the ancestor
        walk compute_ancestor_overlap_discount() performs inline for its
        own per-target bookkeeping; kept as its own method here because
        node_investment_provenance() below needs the same walk starting
        from an arbitrary node, not grouped by convergence target.
        """
        if max_depth is None:
            max_depth = CONFIG.get("provenance_ancestor_max_depth", 6)
        ancestry_types = set(CONFIG.get(
            "provenance_ancestry_relationship_types",
            ["DERIVED_FROM", "ENABLES", "CAUSED_BY"]))
        reverse_lineage: Dict[str, List[str]] = defaultdict(list)
        for (src, tgt), edge in self.edges.items():
            if not edge.is_containment_edge and edge.relationship_type in ancestry_types:
                reverse_lineage[tgt].append(src)

        seen = {node_id}
        frontier = [node_id]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for cur in frontier:
                for parent in reverse_lineage.get(cur, ()):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append(parent)
            frontier = nxt
            depth += 1
        return frozenset(seen)

    def node_investment_provenance(
        self, node_id: str, max_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Exact Set-Union investment provenance for node_id (Manifest §3,
        "Закон сохранения ресурса" — the "Rhombus problem": a grant that
        spawned two research branches later converging on one breakthrough
        node must not be counted twice there).

        Walks node_id's own lineage_ancestors() and, for every ancestor
        (node_id included), records its local funding as one origin token
        via _investment_provenance_token(). When two ancestors resolve to
        the SAME token (the same named funding round, or literally the
        same node reached via two paths), only ONE amount is kept — that
        is the Set Union the manifest describes, as opposed to naively
        summing every node's investment_total_usd along every path.

        max() rather than sum() when the same token shows different
        amounts on different nodes is a deliberate, asymmetric-risk
        choice: ConvergencePressureField exists specifically to flag
        *excess*/over-hyped investment pressure, so silently overcounting
        shared money would manufacture exactly the false signal this
        mechanism exists to prevent. Undercounting a genuine split of one
        round across branches is the safer failure to allow.

        Returns {token: usd_amount}; effective_investment_usd() is the sum
        of this dict's values — the de-duplicated total.
        """
        provenance: Dict[str, float] = {}
        for anc_id in self.lineage_ancestors(node_id, max_depth):
            anc = self.nodes.get(anc_id)
            if anc is None or anc.is_temporal_zone:
                continue
            token = _investment_provenance_token(anc)
            if token is None:
                continue
            amt = float(getattr(anc, "investment_total_usd", 0.0))
            provenance[token] = max(provenance.get(token, 0.0), amt)
        return provenance

    def effective_investment_usd(self, node_id: str, max_depth: Optional[int] = None) -> float:
        """De-duplicated investment total reachable from node_id — the sum
        of node_investment_provenance()'s Set-Union'd tokens. Use this
        instead of raw node.investment_total_usd anywhere that combines
        money across more than one node; a single node's own value alone
        needs no deduplication."""
        return float(sum(self.node_investment_provenance(node_id, max_depth).values()))

    def combined_investment_provenance(
        self, node_ids: Iterable[str], max_depth: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Set-Union of node_investment_provenance() across several nodes that
        all converge on a common downstream target — the literal "merge
        point" from the manifest's Rhombus example. Same combine rule (max
        per token) as the single-node walk, so a funding round visible via
        two different converging branches still counts once at the point
        where they meet.
        """
        combined: Dict[str, float] = {}
        for nid in node_ids:
            for token, amt in self.node_investment_provenance(nid, max_depth).items():
                combined[token] = max(combined.get(token, 0.0), amt)
        return combined

    def k_hop_neighborhood(self, node_ids: List[str], k: int = 3) -> set:
        """BFS out to k hops using the adjacency index (TODO #20) — O(hops
        actually touched) rather than O(E) per hop. Same return value as
        before: the full set of node ids within k hops (seeds included)."""
        frontier = set(node_ids)
        visited  = set(node_ids)
        for _ in range(k):
            new_f = set()
            for nid in frontier:
                for (src, tgt) in self._adjacency.get(nid, ()):
                    other = tgt if src == nid else src
                    if other not in visited:
                        new_f.add(other)
            if not new_f:
                break  # nothing left to expand — no point spinning through remaining hops
            frontier = new_f
            visited.update(frontier)
        return visited

    def get_neighbours(self, node_id: str, min_component: str = None,
                        min_value: float = 0.0, include_future: bool = False,
                        cutoff_ts: float = float("inf")
                        ) -> List[Tuple[str, KnowledgeEdge]]:
        """Edges touching node_id, via the adjacency index (TODO #20) — O(degree)
        instead of a scan over every edge in the graph."""
        result = []
        for key in self._adjacency.get(node_id, ()):
            edge = self.edges.get(key)
            if edge is None:
                continue  # stale key (shouldn't happen — add_edge/_deindex_edge keep this in sync)
            src, tgt = key
            nbr = tgt if src == node_id else src
            if nbr not in self.nodes: continue
            nbr_ts = self.nodes[nbr].timestamp
            if not include_future and nbr_ts > cutoff_ts: continue
            if min_component:
                if getattr(edge, min_component, 0.0) < min_value: continue
            result.append((nbr, edge))
        result.sort(key=lambda x: x[1].total_weight, reverse=True)
        return result


    def propagate_wave(
        self,
        seed_ids: Iterable[str],
        apply_fn: Callable[[str], bool],
        max_visits: Optional[int] = None,
    ) -> Set[str]:
        """
        Self-terminating change propagation (Manifest — the "wave", not a
        fixed-radius cutoff): starting from seed_ids (freshly added/changed
        nodes), visit each reached node and call apply_fn(node_id). apply_fn
        does whatever local recompute the caller cares about (pressure,
        tension, an auxiliary model's local nudge...) and returns True iff
        that node's own state changed *enough to matter*. The wave only
        continues past a node that returned True; a node that returns False
        absorbs the disturbance and the wave simply stops there — "a node
        relays the change to its neighbours only if the change actually
        shifts its own weight or tension".

        There is no `k`/radius parameter on purpose: how far a disturbance
        travels is an emergent property of the graph's own state, not a
        number callers pick in advance. `max_visits` is a hard safety valve
        against a pathological cascade touching the whole graph — not the
        ordinary termination mechanism — and should stay unset in the
        common case where the wave dies out on its own within a few hops.

        Returns every node id actually visited (seeds included, whether or
        not that seed's own apply_fn call returned True).
        """
        visited: Set[str] = set()
        frontier: List[str] = list(dict.fromkeys(seed_ids))
        while frontier:
            next_frontier: List[str] = []
            for nid in frontier:
                if nid in visited:
                    continue
                visited.add(nid)
                if max_visits is not None and len(visited) >= max_visits:
                    return visited
                if not apply_fn(nid):
                    continue  # wave dies here — no relay to neighbours
                for (src, tgt) in self._adjacency.get(nid, ()):
                    other = tgt if src == nid else src
                    if other not in visited:
                        next_frontier.append(other)
            frontier = next_frontier
        return visited

    def audit_ancestry(self) -> Dict[str, Any]:
        """Audit the ancestry invariant across the whole graph.

        Returns a summary dict with:
          - "primordial": list of (node_id, text) for the N exempt roots.
          - "orphans":    list of (node_id, text) for non-primordial nodes
                         that have no incoming edge (invariant violation).
          - "synthesised_edges": count of edges with
                         discovery_method == "ancestry_synthesised".
          - "ok":        True if orphans list is empty.

        Use this in tests, post-ingestion checks, and training data validation.
        Example::

            report = graph.audit_ancestry()
            assert report["ok"], f"Orphan nodes: {report['orphans']}"
        """
        n = CONFIG["primordial_cutoff_n"]
        primordial_ids = {nid for _, nid in self.temporal_sorted[:n]}

        orphans = []
        for nid, node in self.nodes.items():
            if node.is_temporal_zone:
                continue
            if nid in primordial_ids:
                continue
            if not self.has_incoming_edge(nid):
                orphans.append((nid, node.text[:60]))

        primordial = [
            (nid, self.nodes[nid].text[:60])
            for _, nid in self.temporal_sorted[:n]
            if nid in self.nodes
        ]

        synthesised = sum(
            1 for e in self.edges.values()
            if e.discovery_method == "ancestry_synthesised"
        )

        return {
            "primordial":        primordial,
            "orphans":           orphans,
            "synthesised_edges": synthesised,
            "ok":                len(orphans) == 0,
        }

    # ── Temporal point-in-time reconstruction ("sliding window") ────────────
    # Manifest §4, "Слепки скользящего окна": standard graph DBs don't suit
    # a graph that is temporal and continuously retroactively rewritten, so
    # storage/loading here work in terms of a *slice* — the state the graph
    # had as of a given moment — rather than only ever operating on the one
    # always-current graph. as_of() is the in-memory half of that (instant
    # reconstruction of "the graph as it looked in year Y"); the paired
    # save_temporal_window()/load_temporal_window() functions further down
    # this file are the on-disk half, persisting a slice as its own memory
    # dump so it can be reloaded without re-deriving it (the "Memory Dumps
    # скользящих временных окон" storage layer itself).
    def as_of(self, cutoff: float) -> "RuntimeGraph":
        """
        Reconstruct the graph as it existed at `cutoff` (a UNIX timestamp,
        the same units as KnowledgeNode.timestamp / TemporalZone.
        start_timestamp elsewhere in this file).

        A node is included iff its own line had already started by
        `cutoff` (timestamp_start <= cutoff — the same field reinforcement
        already keeps in sync with a node's earliest observation; see
        GraphConsolidator's "node.timestamp_start = tl.points[0].timestamp"
        near the InfluenceTimeline sync). An edge is included iff both
        endpoints survive. A zone is included iff it had already begun
        (start_timestamp <= cutoff) — a zone can still stretch past cutoff
        into this slice's "future", same as it would look to someone
        actually standing at `cutoff`, not yet knowing when the zone ends.

        Surviving nodes get their own history trimmed to what would
        actually have been knowable at `cutoff`, rather than silently
        leaking hindsight into a "historical" slice — see
        _trim_node_to_cutoff() for exactly what that does and does not
        rewind; in short, topology and the epistemic-mass/density signal
        are reconstructed correctly, but scores fed by many different
        subsystems (readiness_score, investment_total_usd, forecast_score,
        LLM-rated physical-axis scores, ...) are carried over at
        last-known-value rather than independently un-computed — there is
        no single formula that inverts all of them, and guessing would be
        worse than being explicit about the limitation. Treat those fields
        as approximate on any slice earlier than a node's final observation.

        Returns a NEW, independent RuntimeGraph; the live graph this is
        called on is never mutated. Family links are carried over
        restricted to pairs that both survive. Rebuilt via add_node/
        add_edge/add_zone (not by copying the dicts directly) so every
        derived index — temporal_sorted, domain_clusters, the incoming-edge
        cache, zone membership — comes out internally consistent for the
        slice automatically instead of hand-copied and possibly stale.
        """
        sliced = RuntimeGraph()
        surviving_ids: Set[str] = set()

        for nid, node in self.nodes.items():
            if node.timestamp_start > cutoff:
                continue
            node_copy = copy.deepcopy(node)
            self._trim_node_to_cutoff(node_copy, cutoff)
            sliced.add_node(node_copy)
            surviving_ids.add(nid)

        for (src, tgt), edge in self.edges.items():
            if src in surviving_ids and tgt in surviving_ids:
                edge_copy = copy.deepcopy(edge)
                edge_history = getattr(edge_copy, "__dict__", {}).get(
                    "source_observations", [])
                if isinstance(edge_history, list) and edge_history:
                    kept_history = []
                    for point in edge_history:
                        if not isinstance(point, dict):
                            continue
                        try:
                            point_ts = float(point.get("timestamp"))
                        except (TypeError, ValueError):
                            continue
                        if point_ts <= cutoff:
                            kept_history.append(point)
                    if not kept_history:
                        continue
                    edge_copy.__dict__["source_observations"] = kept_history
                    edge_copy.timestamp = max(
                        float(point["timestamp"]) for point in kept_history)
                    latest_point = max(kept_history, key=lambda point: float(point["timestamp"]))
                    if latest_point.get("evidence"):
                        edge_copy.evidence = list(latest_point["evidence"])
                elif edge_copy.timestamp > cutoff:
                    continue
                sliced.add_edge(edge_copy)

        for zone in self.zones.values():
            if zone.start_timestamp <= cutoff:
                sliced.add_zone(copy.deepcopy(zone))

        for a, neighbours in self._family_links.items():
            if a not in surviving_ids:
                continue
            for b in neighbours:
                if b in surviving_ids:
                    sliced.union_family(a, b)

        logger.info(
            "[RuntimeGraph.as_of] cutoff=%.0f -> %d/%d nodes, %d/%d edges, "
            "%d/%d zones survive",
            cutoff, len(sliced.nodes), len(self.nodes),
            len(sliced.edges), len(self.edges),
            len(sliced.zones), len(self.zones),
        )
        return sliced

    @staticmethod
    def _trim_node_to_cutoff(node: "KnowledgeNode", cutoff: float) -> None:
        """
        Mutate `node` (already a private deep copy — see as_of()) so its
        own recorded history doesn't reach past `cutoff`. Two things are
        rewound with confidence because each has a single, well-defined
        source of truth:

          1. temporal_density: truncated to the bins that had started by
             cutoff (using density_bin_years to convert cutoff into a bin
             count from timestamp_start), and timestamp_end is clamped
             down to cutoff — the node's "line" simply doesn't extend past
             the slice.
          2. epistemic_timeline (InfluenceTimeline), when present: points
             after cutoff are dropped, reference_ts is pinned to cutoff
             (the exact mechanism InfluenceTimeline documents for
             historical replay — see its docstring), and
             node.epistemic_mass is resynced from the truncated timeline
             so it reflects only what had accumulated by cutoff.

        Everything else is deliberately left alone — see as_of()'s
        docstring for why guessing a rewind formula for the remaining
        scores would be worse than leaving them as last-known-value.
        """
        if node.timestamp_end:
            node.timestamp_end = min(node.timestamp_end, cutoff)
        if node.timestamp > cutoff:
            node.timestamp = cutoff

        if node.temporal_density and node.density_bin_years:
            bins_kept = int((cutoff - node.timestamp_start) // node.density_bin_years) + 1
            bins_kept = max(bins_kept, 0)
            if bins_kept < len(node.temporal_density):
                node.temporal_density = node.temporal_density[:bins_kept]

        source_observations = node.__dict__.get("source_observations")
        if isinstance(source_observations, list):
            node.__dict__["source_observations"] = [
                point for point in source_observations
                if isinstance(point, dict)
                and point.get("timestamp") is not None
                and _safe_timestamp(point.get("timestamp")) <= cutoff
            ]
            dated_statuses = [point for point in node.__dict__["source_observations"]
                              if point.get("label_status_code") is not None]
            if dated_statuses:
                latest_status = dated_statuses[-1]
                node.__dict__["label_status_code"] = latest_status["label_status_code"]
                node.__dict__["label_status"] = latest_status.get("label_status")
                node.__dict__["status_evidence"] = latest_status.get("status_evidence", "")
            else:
                node.__dict__.pop("label_status_code", None)
                node.__dict__.pop("label_status", None)
                node.__dict__.pop("status_evidence", None)

        label_timeline = node.__dict__.get("label_support_timeline")
        if isinstance(label_timeline, list):
            node.__dict__["label_support_timeline"] = [
                point for point in label_timeline
                if isinstance(point, dict)
                and point.get("timestamp") is not None
                and _safe_timestamp(point.get("timestamp")) <= cutoff
            ]
            if node.__dict__["label_support_timeline"]:
                latest_support = node.__dict__["label_support_timeline"][-1]
                node.__dict__["label_support_score"] = latest_support.get(
                    "support_score", 0.0)
                effective_status = latest_support.get("effective_label_status_code")
                if effective_status is not None:
                    node.__dict__["effective_label_status_code"] = effective_status
                    node.__dict__["label_status_code"] = effective_status
                    node.__dict__["label_status"] = latest_support.get(
                        "effective_label_status")
                    node.__dict__["status_evidence"] = latest_support.get(
                        "status_evidence", "")
            else:
                node.__dict__.pop("label_support_score", None)
                node.__dict__.pop("effective_label_status_code", None)

        timeline = node.__dict__.get("epistemic_timeline")
        if timeline is not None and getattr(timeline, "points", None):
            timeline.points = [p for p in timeline.points if p.timestamp <= cutoff]
            timeline.reference_ts = cutoff
            node.epistemic_mass = timeline.epistemic_mass

    def subgraph(self, node_ids: Iterable[str]) -> "RuntimeGraph":
        """
        Node-induced subgraph: `node_ids` plus every edge/zone/family-link
        between members of that set. Unlike as_of(), this filters by
        membership rather than time and does not trim any node's history —
        callers that want a *historical* slice should use as_of(); this is
        the complementary "just this neighbourhood" extraction — e.g.
        TemporalSplitTrainer's wave-bounded local training step (Manifest
        §5), or pulling a subgraph as the base for graph_to_tensors()/
        model.encode() without paying an encode pass over the whole live
        graph.

        Returns a NEW, independent RuntimeGraph, built via add_node/
        add_edge/add_zone for the same internal-consistency reason as
        as_of() (see its docstring) — every derived index comes out
        correct for the subset automatically.
        """
        keep = set(node_ids)
        sub = RuntimeGraph()
        for nid in keep:
            node = self.nodes.get(nid)
            if node is not None:
                sub.add_node(copy.deepcopy(node))
        for (src, tgt), edge in self.edges.items():
            if src in keep and tgt in keep:
                sub.add_edge(copy.deepcopy(edge))
        for zone in self.zones.values():
            if set(zone.contained_node_ids) & keep:
                sub.add_zone(copy.deepcopy(zone))
        for a, neighbours in self._family_links.items():
            if a not in keep:
                continue
            for b in neighbours:
                if b in keep:
                    sub.union_family(a, b)
        return sub


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH WALKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathStep:
    node_id:           str
    node_text:         str
    node_domain:       str
    node_timestamp:    float
    zone_id:           str
    zone_multiplier:   float
    edge_to_next:      Optional[KnowledgeEdge]
    cumulative_weight: float

    @property
    def is_cross_domain(self) -> bool:
        if self.edge_to_next is None: return False
        return self.edge_to_next.relationship_type.startswith("cross_domain")


@dataclass
class Trajectory:
    query:             str
    steps:             List[PathStep]
    total_confidence:  float
    cross_domain_hops: int
    deepest_zone_mult: float
    provenance_urls:   List[str]

    def to_readable(self) -> str:
        lines = [f"Query: {self.query}",
                 f"Confidence: {self.total_confidence:.3f} | "
                 f"Cross-domain hops: {self.cross_domain_hops} | "
                 f"Max zone acceleration: ×{self.deepest_zone_mult:.1f}", ""]
        for i, step in enumerate(self.steps):
            zone_tag = (f" [inside zone ×{step.zone_multiplier:.1f}]"
                        if step.zone_id else "")
            lines.append(f"  Step {i+1}: {step.node_text}{zone_tag}")
            lines.append(f"           domain: {step.node_domain} "
                         f"| year: {_ts_to_year(step.node_timestamp)}")
            if step.edge_to_next:
                e = step.edge_to_next
                dominant = max(EDGE_WEIGHT_COMPONENTS,
                               key=lambda c: getattr(e, c))
                lines.append(f"           → [{e.relationship_type}]  "
                             f"(dominant signal: {dominant}={getattr(e, dominant):.2f})")
        lines.append("")
        lines.append("Sources: " + " | ".join(self.provenance_urls[:5]))
        return "\n".join(lines)


class GraphWalker:
    BEAM_WIDTH   = CONFIG["walker_beam_width"]
    MAX_PATH_LEN = CONFIG["walker_max_path_len"]
    MIN_CONF     = CONFIG["walker_min_conf"]
    LIMIT_BONUS  = CONFIG["walker_limit_bonus"]
    LIMIT_THRESH = CONFIG["walker_limit_thresh"]

    def __init__(self, runtime_graph: RuntimeGraph):
        self.g = runtime_graph

    def query(self, query_text: str, embedder,
              max_trajectories: int = 5,
              component_filter: Optional[str] = None,
              min_component_value: float = 0.0) -> List[Trajectory]:
        query_emb = embedder.encode(query_text)
        start_ids = self._find_start_nodes(query_emb, top_k=CONFIG["walker_top_k_start"])
        if not start_ids: return []

        beams: List[Tuple[List[str], float, float]] = [
            ([nid], 1.0, self.g.nodes[nid].timestamp)
            for nid in start_ids
        ]
        completed = []

        for step in range(self.MAX_PATH_LEN):
            next_beams = []
            for path, score, last_ts in beams:
                current_id = path[-1]
                nbrs = self.g.get_neighbours(
                    current_id, min_component=component_filter,
                    min_value=min_component_value)
                for nbr_id, edge in nbrs[:CONFIG["walker_nbr_scan_cap"]]:
                    if nbr_id in path: continue
                    nbr = self.g.nodes[nbr_id]
                    nbr_ts = nbr.timestamp
                    # Timeless nodes (temporal_kind == "axiom", e.g. undated
                    # ideas/myths/mathematical facts) carry no real date, so
                    # they must not be gated by — or gate — the forward-in-time
                    # constraint the way a timestamp=0.0 (1970!) placeholder
                    # otherwise silently would.
                    if getattr(nbr, "temporal_kind", "point") != "axiom" and nbr_ts < last_ts:
                        continue
                    new_score = score * max(edge.total_weight, 1e-6)
                    if edge.limitation_resolution > self.LIMIT_THRESH:
                        new_score *= self.LIMIT_BONUS
                    nbr_node = self.g.nodes.get(nbr_id)
                    if nbr_node and nbr_node.zone_id:
                        new_score *= nbr_node.acceleration_multiplier
                    if new_score < self.MIN_CONF: continue
                    next_beams.append((path + [nbr_id], new_score, nbr_ts))

            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:self.BEAM_WIDTH]
            if not beams: break
            for path, score, _ in beams:
                if len(path) >= CONFIG["walker_min_path_len"]:
                    completed.append((path, score))

        completed.sort(key=lambda x: x[1], reverse=True)
        trajectories = []
        seen_paths = set()
        for path, score in completed:
            key = tuple(path)
            if key in seen_paths: continue
            seen_paths.add(key)
            traj = self._build_trajectory(query_text, path, score)
            if traj: trajectories.append(traj)
            if len(trajectories) >= max_trajectories: break
        return trajectories

    def _find_start_nodes(self, query_emb: np.ndarray, top_k: int = 5) -> List[str]:
        scores = []
        query_vec = np.asarray(query_emb, dtype=np.float32).reshape(-1)
        text_dim = NodeFeatureBuilder.TEXT_DIM
        if query_vec.size < text_dim:
            return []
        query_vec = query_vec[:text_dim]
        for nid, node in self.g.nodes.items():
            if node.embedding is None or node.is_temporal_zone: continue
            # CanonicalNodeAssembler stores the full graph feature vector on
            # node.embedding (413 dimensions, or 423 with physical axes).
            # Query retrieval is semantic, so compare only its first 384 text
            # dimensions with the same slice of each node vector.
            node_vec = np.asarray(node.embedding, dtype=np.float32).reshape(-1)[:text_dim]
            if node_vec.size != text_dim:
                continue
            sim = float(cos_sim([query_vec], [node_vec])[0][0])
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scores[:top_k]]

    def _build_trajectory(self, query: str, path: List[str],
                           score: float) -> Optional[Trajectory]:
        steps = []
        cross_hops   = 0
        deepest_zone = 1.0
        provenance   = []

        for i, nid in enumerate(path):
            node = self.g.nodes.get(nid)
            if not node: return None
            edge = (self.g.get_edge(path[i], path[i+1])
                    if i < len(path) - 1 else None)
            if edge and edge.is_containment_edge: continue
            zone_mult = node.acceleration_multiplier
            if zone_mult > deepest_zone: deepest_zone = zone_mult
            steps.append(PathStep(
                node_id=nid, node_text=node.text, node_domain=node.domain,
                node_timestamp=node.timestamp, zone_id=node.zone_id,
                zone_multiplier=zone_mult, edge_to_next=edge,
                cumulative_weight=score))
            if node.provenance: provenance.append(node.provenance)

        if len(steps) < 2: return None
        return Trajectory(
            query=query, steps=steps, total_confidence=score,
            cross_domain_hops=cross_hops, deepest_zone_mult=deepest_zone,
            provenance_urls=list(dict.fromkeys(provenance)))


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — INHIBITORY EDGES
# ══════════════════════════════════════════════════════════════════════════════

def extend_edge(edge: KnowledgeEdge, inhibitory_force: float = 0.0) -> KnowledgeEdge:
    edge.inhibitory_force = float(np.clip(inhibitory_force, 0.0, 1.0))
    base = getattr(edge, "total_weight", 0.0)
    edge.total_weight = max(0.0, base + INHIBITORY_WEIGHT * edge.inhibitory_force)
    return edge


def compute_inhibitory_pressure(node_id: str, graph: RuntimeGraph) -> float:
    pressure = 0.0
    for (src, tgt), edge in graph.edges.items():
        if tgt != node_id: continue
        inh = getattr(edge, "inhibitory_force", 0.0)
        if inh <= 0: continue
        attacker = graph.get_node(src)
        attacker_weight = (getattr(attacker, "strategic_value", 5.0) / 10.0
                           if attacker else 0.5)
        pressure += inh * attacker_weight
    return min(1.0, pressure)


# ══════════════════════════════════════════════════════════════════════════════
#  RESOURCE PROVENANCE OVERCOUNTING (TODO #10)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ancestor_overlap_discount(
    graph: "RuntimeGraph",
    targets: Optional[Iterable[str]] = None,
) -> Dict[Tuple[str, str], float]:
    """
    "Diamond topology" double-counting fix.

    If a single external cause (one funding tranche, one foundational
    breakthrough) produced two adjacent technologies A and B that then both
    feed a common downstream node C, naively summing A's and B's reported
    magnitudes at C treats them as independent — double-counting the shared
    resource. This computes, for every non-containment edge (src, tgt),
    a discount in [provenance_min_discount, 1.0] proportional to how much of
    `src`'s own ancestry is *also* reachable from at least one sibling edge
    converging on the same `tgt` — an exact set-intersection over real graph
    reachability (walking backward along DERIVED_FROM / ENABLES / CAUSED_BY
    edges specifically — the same causal/derivation vs. commentary-type
    distinction the edge-validation prompt already draws), not an
    approximate similarity heuristic.

    This is the "Option A" mechanism from the TODO, but grounded in exact
    ancestor-set membership rather than a fuzzy Jaccard-similarity score.
    Full "Option C" (explicit named-provenance dicts, e.g.
    {"pentagon_grant_1959": 1e9}, unioned at merge points) would be more
    precise still, but depends on the LLM extraction pipeline tagging a
    stable origin identifier per funding mention — not attempted here since
    LocalLLMClient.complete() is currently a stub (see TODO #1) and there is
    no way to validate such tags end-to-end yet. "Option B" (rewriting the
    GNN's message passing as literal flow-conservation) is a separate,
    larger architecture decision deferred to the user.

    Used by:
      - ConvergencePressureField._compute_node_pressure() — discounts
        excess_investment_pressure per qualifying incoming edge.
      - graph_to_tensors() — returns an edge_provenance_discount tensor,
        row-aligned with edge_index, that Oracle1Model.encode() multiplies
        into the edge_weights feeding StructuralDependencyUpdater's sdi/casc
        scatter_add (the neural analogue of the same signal).

    Not applied to CascadeAnalyzer.analyze(): that traversal already visits
    each node at most once and keeps the MAX cumulative conductance among
    competing paths rather than summing them (see `next_frontier`), so it
    does not exhibit this bug.

    Returns {(src_id, tgt_id): discount} for every non-containment edge.
    Edges whose target has fewer than two distinct incoming sources, or
    whose source shares no ancestry with any sibling source, get 1.0
    (no discount).
    """
    ancestry_types = set(CONFIG.get(
        "provenance_ancestry_relationship_types",
        ["DERIVED_FROM", "ENABLES", "CAUSED_BY"]))
    max_depth = CONFIG.get("provenance_ancestor_max_depth", 6)
    strength  = CONFIG.get("provenance_overlap_discount_strength", 1.0)
    min_disc  = CONFIG.get("provenance_min_discount", 0.15)

    # Reverse adjacency restricted to lineage-carrying edge types — this
    # defines what "ancestor" means for this computation. Built once, O(E).
    reverse_lineage: Dict[str, List[str]] = defaultdict(list)
    # Sources grouped by target over ALL non-containment edges: the
    # convergence being protected against is general (sdi/casc are not
    # investment-specific), not restricted to lineage-typed edges.
    sources_by_target: Dict[str, List[str]] = defaultdict(list)

    for (src, tgt), edge in graph.edges.items():
        if edge.is_containment_edge:
            continue
        sources_by_target[tgt].append(src)
        if edge.relationship_type in ancestry_types:
            reverse_lineage[tgt].append(src)

    if targets is not None:
        target_set = set(targets)
        sources_by_target = {t: v for t, v in sources_by_target.items() if t in target_set}

    ancestor_cache: Dict[str, FrozenSet[str]] = {}

    def ancestors(nid: str) -> FrozenSet[str]:
        cached = ancestor_cache.get(nid)
        if cached is not None:
            return cached
        seen = {nid}
        frontier = [nid]
        depth = 0
        while frontier and depth < max_depth:
            nxt = []
            for cur in frontier:
                for parent in reverse_lineage.get(cur, ()):
                    if parent not in seen:
                        seen.add(parent)
                        nxt.append(parent)
            frontier = nxt
            depth += 1
        result = frozenset(seen)
        ancestor_cache[nid] = result
        return result

    discount: Dict[Tuple[str, str], float] = {}
    for tgt, srcs in sources_by_target.items():
        unique_srcs = list(dict.fromkeys(srcs))  # de-dup, preserve order
        if len(unique_srcs) < 2:
            for s in unique_srcs:
                discount[(s, tgt)] = 1.0
            continue
        src_ancestors = {s: ancestors(s) for s in unique_srcs}
        for s in unique_srcs:
            own = src_ancestors[s]
            if len(own) <= 1:  # nothing but itself — nowhere to overlap
                discount[(s, tgt)] = 1.0
                continue
            sibling_union: Set[str] = set()
            for other in unique_srcs:
                if other != s:
                    sibling_union |= src_ancestors[other]
            if not sibling_union:
                discount[(s, tgt)] = 1.0
                continue
            overlap_fraction = len(own & sibling_union) / len(own)
            d = 1.0 - strength * overlap_fraction
            discount[(s, tgt)] = max(min_disc, min(1.0, d))

__all__ = ['_investment_provenance_token', 'AncestryEnforcer', 'ANCESTRY_ENFORCER', 'RuntimeGraph', 'PathStep', 'Trajectory', 'GraphWalker', 'extend_edge', 'compute_inhibitory_pressure', 'compute_ancestor_overlap_discount']
