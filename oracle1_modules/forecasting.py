"""Phantom-node, scenario, and convergence forecasting."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, EDGE_WEIGHT_COMPONENTS, N_EDGE_COMPONENTS
from .entities import KnowledgeEdge, KnowledgeNode, PhysicalSubstrateEncoder
from .graph import GraphWalker, RuntimeGraph, Trajectory
from .graph_dynamics import (
    CONFIDENCE_ENGINE,
    ContextualPriorityHead,
    ConvergencePressureField,
    DormancyTracker,
    GRAPH_TRAVERSAL,
    PRESSURE_COMPUTER,
    VoidPressureVector,
)
from .label_status import (
    LabelStatus,
    node_label_status,
    node_label_support_score,
    parse_label_status,
)
from .time_utils import _ts_to_year
from ._shared import _maybe_no_grad

# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — PHANTOM NODE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhantomNode:
    id: str
    gap_description: str
    predicted_entity_type: str
    predicted_domain: str
    predicted_year_range: Tuple[int, int]
    predicted_feature_vec: np.ndarray
    structural_gap_score: float
    source_node_ids: List[str]
    target_node_ids: List[str]
    void_pressure_at_gap: float = 0.0
    is_confirmed: bool = False
    confirmed_by_node_id: Optional[str] = None
    generation_timestamp: float = 0.0
    evidence_count: int = 0
    CONFIRMATION_SIM_THRESHOLD: float = 0.75
    source_label_status_codes: List[int] = field(default_factory=list)
    source_support_scores: List[float] = field(default_factory=list)
    epistemic_basis: List[Dict[str, Any]] = field(default_factory=list)
    social_readiness: Dict[str, Any] = field(default_factory=dict)


class PhantomNodeGenerator:
    EDGE_PROB_THRESHOLD = CONFIG["phantom_edge_prob_threshold"]
    PHANTOM_THRESHOLD   = CONFIG["phantom_gap_threshold"]
    CONFIRMATION_SIM    = CONFIG["phantom_confirmation_sim"]
    MAX_PHANTOMS        = CONFIG["phantom_max_count"]
    SOCIAL_READINESS_TYPES = frozenset({
        "fictional_hype", "social_attention", "unmet_need",
    })

    def __init__(self, neural_enabled: bool = True):
        self.phantoms: List[PhantomNode] = []
        if TORCH_AVAILABLE and neural_enabled:
            latent_dim = CONFIG["model_latent_dim"]
            self.feature_decoder = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Linear(256, PhysicalSubstrateEncoder.EXTENDED_TOTAL_DIM),
                nn.Sigmoid())

    def scan_for_voids(self, node_latents, graph: RuntimeGraph,
                       pressure_field: ConvergencePressureField,
                       edge_predictor=None,
                       seed_ids: Optional[Iterable[str]] = None,
                       top_candidates: int = 100) -> List[PhantomNode]:
        """
        seed_ids (TODO #23): the nodes that were just added or changed —
        i.e. exactly where new edges just appeared in the graph. Defaults
        to pressure_field._last_changed (the same change set
        DormancyTracker.update_wave() is fed — see
        Oracle1Extended.update_epoch()), so callers that don't pass it
        explicitly still get the scoped behaviour rather than silently
        falling back to a graph-wide scan.
        """
        if not graph.nodes:
            return []
        if seed_ids is None:
            seed_ids = pressure_field._last_changed
        raw_candidates = self._find_void_pairs(seed_ids, graph, pressure_field, top_candidates)
        # Candidate pairs have already passed structural and epistemic gates.
        candidate_pairs = raw_candidates
        known_pairs = {
            frozenset(phantom.source_node_ids + phantom.target_node_ids)
            for phantom in self.phantoms
        }
        new_phantoms = []
        for src_id, tgt_id, gap_pressure in candidate_pairs:
            pair_key = frozenset((src_id, tgt_id))
            if pair_key in known_pairs:
                continue
            if gap_pressure < self.PHANTOM_THRESHOLD: continue
            src_idx = self._node_to_idx(src_id, graph)
            tgt_idx = self._node_to_idx(tgt_id, graph)
            if src_idx is None or tgt_idx is None: continue
            if node_latents is not None:
                h_src, h_tgt = node_latents[src_idx], node_latents[tgt_idx]
            else:
                h_src = graph.nodes[src_id].embedding
                h_tgt = graph.nodes[tgt_id].embedding
            if h_src is None or h_tgt is None:
                continue
            if TORCH_AVAILABLE and isinstance(h_src, torch.Tensor):
                bridge = (h_src + h_tgt) / 2.0
                if hasattr(self, "feature_decoder"):
                    with _maybe_no_grad():
                        predicted_features = self.feature_decoder(
                            bridge.unsqueeze(0)).squeeze(0).cpu().numpy()
                else:
                    predicted_features = bridge.detach().cpu().numpy()
            else:
                predicted_features = (
                    np.asarray(h_src, dtype=np.float32)
                    + np.asarray(h_tgt, dtype=np.float32)
                ) / 2.0
            phantom = self._build_phantom(src_id, tgt_id, predicted_features,
                                          gap_pressure, graph)
            if phantom:
                new_phantoms.append(phantom)
                self.phantoms.append(phantom)
                known_pairs.add(pair_key)
        self.phantoms.sort(key=lambda p: p.structural_gap_score, reverse=True)
        self.phantoms = self.phantoms[:self.MAX_PHANTOMS]
        return new_phantoms

    def _local_void_candidates(self, seed_ids: Iterable[str], graph: "RuntimeGraph",
                                pressure_field: "ConvergencePressureField") -> Set[str]:
        """
        Candidate pool for phantom-pair search (TODO #23) around nodes that
        were just added or changed — i.e. exactly when new edges appear in
        the graph, per the manifest. Two sources, neither of them "all
        nodes":
          • graph neighbours — a 2-hop walk from each seed (structural
            holes: a phantom bridge typically closes a triangle the new
            edge just opened, so the missing corner is within 2 hops);
          • same-topic nodes — via the CanonicalKey group/domain index
            (TODO #24, see below). This component is intentionally NOT
            bounded by graph distance — a node found this way can be
            arbitrarily far in the topology; that's the point of having it
            at all (a phantom bridge is just as often between two nodes
            that have simply never been connected), not a flaw. What has
            to be bounded is whether "same topic" is actually meaningful,
            and whether the resulting pool stays small — both handled
            below.

        TODO #24 — two failure modes the plain domain_id lookup had:
          1. domain_id == 0 (== group_id == 0 == entity_type_id == 0, see
             ConceptClassifier._unknown_result()) means "never
             successfully classified", not "genuinely the same domain as
             every other unclassified node in the graph". Expanding
             through _canon_domain_idx[0] would pull in an arbitrary,
             unbounded grab-bag of mutually-unrelated nodes that share
             nothing but having failed classification — nodes that really
             can be domain-unrelated and arbitrarily distant, which is
             exactly the bad kind of "distant", as opposed to the
             deliberate cross-topology reach described above. Skipped
             entirely for an unclassified seed; the 2-hop graph walk above
             still runs regardless.
          2. Even a real domain/group has no upper bound on population —
             a broad, heavily-populated field would otherwise silently
             turn "scoped" back into "graph-wide" for exactly the seeds
             that live in a common domain. Bounded by
             phantom_domain_pool_cap, keeping the highest-pressure members
             of an oversized cluster rather than an arbitrary slice (the
             ones already most likely to be phantom-worthy, by the same
             pressure signal the ranking step below uses anyway) —
             temporal proximity to the seed would be another reasonable
             way to pick which members to keep and isn't done here, so an
             oversized cluster can still surface a temporally-distant
             member as long as its pressure is high enough.
          group_id (a GLOBAL table, GROUP_TABLE — not nested inside
          domain_id) is preferred over domain_id when the classifier
          assigned one: the same group_id can legitimately appear under
          different domain_ids (e.g. "vaccines" tagged under both biology
          and public_health), making it a strictly more specific *and*
          occasionally cross-domain-aware topical match — closer to what
          a phantom bridge is actually looking for than the coarser
          domain bucket.
        """
        pool: Set[str] = set()
        cap = CONFIG["phantom_domain_pool_cap"]
        for seed in seed_ids:
            pool.update(graph.k_hop_neighborhood([seed], k=2))
            key = graph._canon_key_of.get(seed)
            if key is None:
                continue
            domain_id, group_id, _entity_type_id = key
            if domain_id == 0:
                continue  # unclassified — not a meaningful topic match, see TODO #24
            topical = (graph._canon_group_idx.get(group_id, set()) if group_id != 0
                       else graph._canon_domain_idx.get(domain_id, set()))
            if len(topical) > cap:
                topical = set(sorted(
                    topical,
                    key=lambda nid: PRESSURE_COMPUTER.node_pressure(nid, pressure_field),
                    reverse=True)[:cap])
            pool.update(topical)
        return {nid for nid in pool
                if (n := graph.nodes.get(nid)) is not None
                and not n.is_temporal_zone and n.embedding is not None}

    def _find_void_pairs(self, seed_ids: Iterable[str],
                          graph: "RuntimeGraph", pressure_field, top_n):
        """
        TODO #23: scoped to _local_void_candidates(seed_ids) — graph
        neighbours + same-topic nodes around whatever just changed —
        instead of sorting every embedded node in the graph by pressure on
        every call, which cannot work once the graph holds anywhere near
        the node counts the manifest targets. Only runs at all when
        something to scope to actually exists: an empty/exhausted seed set
        (nothing changed since the last scan) means an empty candidate
        pool, so scan_for_voids does no search at all rather than falling
        back to a global one.

        The redundant existing_edges-based neighbour re-check the previous
        version did here (a_nbrs/b_nbrs, scanning every edge in the graph
        to answer a question already answered one line above by the direct
        (a, b) in graph.edges lookup) is gone rather than replaced — it
        never filtered anything the direct lookup didn't already catch.
        """
        pool = self._local_void_candidates(seed_ids, graph, pressure_field)
        if len(pool) < 2:
            return []
        sorted_by_pressure = sorted(
            pool,
            key=lambda nid: PRESSURE_COMPUTER.node_pressure(nid, pressure_field),
            reverse=True)[:CONFIG["phantom_void_top_candidates"]]
        candidates = []
        for i, a in enumerate(sorted_by_pressure):
            for b in sorted_by_pressure[i+1:]:
                if (a, b) in graph.edges or (b, a) in graph.edges: continue
                p_a = PRESSURE_COMPUTER.node_pressure(a, pressure_field)
                p_b = PRESSURE_COMPUTER.node_pressure(b, pressure_field)
                candidates.append((a, b, (p_a + p_b) / 2.0))
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = SPARSITY_REGULATOR.filter_phantom_candidates(
            candidates, graph, pressure_field)
        return candidates[:top_n]

    @classmethod
    def _social_readiness(cls, nodes: Iterable[KnowledgeNode]) -> Dict[str, Any]:
        """Summarize independent, quoted social-readiness sources for a pair."""
        sources_by_type: Dict[str, Set[str]] = defaultdict(set)
        all_sources: Set[str] = set()
        seen_observations: Set[Tuple[str, str, str]] = set()
        observations: List[Dict[str, Any]] = []
        for node in nodes:
            records = getattr(node, "__dict__", {}).get("influence_observations", [])
            for record in records if isinstance(records, list) else []:
                if not isinstance(record, dict):
                    continue
                signal_type = str(record.get("signal_type", "")).strip().casefold()
                if signal_type not in cls.SOCIAL_READINESS_TYPES:
                    continue
                identity = str(record.get("source_identity") or record.get("source", "")) \
                    .strip().casefold()
                evidence = str(record.get("evidence", "")).strip()
                if not identity or not evidence:
                    continue
                key = (identity, signal_type, evidence)
                if key in seen_observations:
                    continue
                seen_observations.add(key)
                all_sources.add(identity)
                sources_by_type[signal_type].add(identity)
                metadata = record.get("source_metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                observations.append({
                    "source": str(record.get("source", "")),
                    "source_identity": str(record.get("source_identity") or identity),
                    "signal_type": signal_type,
                    "title": metadata.get("title"),
                    "authors": list(metadata.get("authors", []) or []),
                    "publication": metadata.get("publication"),
                    "forum_topic": metadata.get("forum_topic"),
                    "source_type": metadata.get("source_type"),
                    "evidence": evidence[:1000],
                })
        return {
            "source_count": len(all_sources),
            "signal_source_counts": {
                signal_type: len(source_ids)
                for signal_type, source_ids in sorted(sources_by_type.items())
            },
            "observations": observations[:10],
        }

    def _build_phantom(self, src_id, tgt_id, predicted_features, gap_pressure,
                        graph, bridge_vec=None) -> Optional[PhantomNode]:
        src = graph.get_node(src_id)
        tgt = graph.get_node(tgt_id)
        if not src or not tgt: return None
        social_readiness = self._social_readiness((src, tgt))
        src_year = _ts_to_year(src.timestamp)
        tgt_year = _ts_to_year(tgt.timestamp)
        predicted_year_min = max(src_year, tgt_year)
        predicted_year_max = predicted_year_min + int(
            CONFIG["phantom_year_range_base"] +
            gap_pressure * CONFIG["phantom_year_range_scale"])
        source_nodes = (src, tgt)
        source_basis = []
        for source_node in source_nodes:
            status = node_label_status(source_node)
            support = node_label_support_score(source_node)
            source_points = getattr(source_node, "__dict__", {}).get(
                "source_observations", [])
            eligible_statuses = (
                {LabelStatus.FACT, LabelStatus.PROOF}
                if status in {LabelStatus.FACT, LabelStatus.PROOF}
                else {status}
            )
            latest = next((point for point in reversed(source_points)
                           if isinstance(point, dict)
                           and (parse_label_status(point.get("label_status_code"))
                                or parse_label_status(point.get("label_status")))
                           in eligible_statuses
                           and point.get("evidence")), {})
            latest_evidence = latest.get("evidence", [])
            if isinstance(latest_evidence, str):
                latest_evidence = [latest_evidence]
            source_basis.append({
                "node_id": source_node.id,
                "text": source_node.text,
                "label_status_code": int(status) if status is not None else None,
                "label_status": status.name.casefold() if status is not None else None,
                "source_label_status_code": (
                    int(parse_label_status(latest.get("label_status_code"))
                        or parse_label_status(latest.get("label_status")))
                    if latest and (parse_label_status(latest.get("label_status_code"))
                                   or parse_label_status(latest.get("label_status")))
                    is not None else None),
                "support_score": round(support, 4),
                "source": latest.get("source", ""),
                "title": (latest.get("source_metadata") or {}).get("title"),
                "evidence": [quote for quote in latest_evidence
                             if isinstance(quote, str) and quote.strip()][:3],
            })
        return PhantomNode(
            id="phantom_" + str(uuid.uuid4())[:8],
            gap_description=(f"Bridge between '{src.text[:40]}' and '{tgt.text[:40]}'"),
            predicted_entity_type="predicted_bridge_technology",
            predicted_domain=f"{src.domain} × {tgt.domain}",
            predicted_year_range=(predicted_year_min, predicted_year_max),
            predicted_feature_vec=predicted_features,
            # Keep this as a monotonic pressure normalization, not a claim
            # that the predicted bridge has this probability of existing.
            structural_gap_score=(gap_pressure /
                                  (gap_pressure + max(self.PHANTOM_THRESHOLD, 1e-8))),
            source_node_ids=[src_id],
            target_node_ids=[tgt_id],
            void_pressure_at_gap=gap_pressure,
            generation_timestamp=max(src.timestamp, tgt.timestamp),
            evidence_count=sum(len(item.get("evidence", [])) for item in source_basis),
            source_label_status_codes=[int(item["label_status_code"])
                                       for item in source_basis
                                       if item.get("label_status_code") is not None],
            source_support_scores=[item["support_score"] for item in source_basis],
            epistemic_basis=source_basis,
            social_readiness=social_readiness)

    def _node_to_idx(self, node_id, graph) -> Optional[int]:
        for i, nid in enumerate(graph.nodes):
            if nid == node_id: return i
        return None

    # Anomaly-based rupture confirmation thresholds
    RUPTURE_ANOMALY_THRESHOLD = CONFIG["phantom_rupture_anomaly_threshold"]
    RUPTURE_PROXIMITY_HOPS    = CONFIG["phantom_rupture_proximity_hops"]

    def check_confirmation(self, new_node: KnowledgeNode,
                            graph: RuntimeGraph) -> List[PhantomNode]:
        """Standard similarity-based confirmation (sim > CONFIRMATION_SIM)."""
        if new_node.embedding is None: return []
        confirmed = []
        for phantom in self.phantoms:
            if phantom.is_confirmed: continue
            predicted_text_emb = phantom.predicted_feature_vec[:384]
            sim = float(cos_sim([new_node.embedding], [predicted_text_emb])[0][0])
            if sim > self.CONFIRMATION_SIM:
                phantom.is_confirmed = True
                phantom.confirmed_by_node_id = new_node.id
                phantom.__dict__["confirmation_mode"] = "similarity"
                confirmed.append(phantom)
                logger.info(f"[PhantomGenerator] CONFIRMED (similarity): "
                            f"{phantom.gap_description[:60]} "
                            f"-> '{new_node.text[:40]}' (sim={sim:.3f})")
        return confirmed

    def check_confirmation_by_rupture(
        self,
        new_node: "KnowledgeNode",
        graph: "RuntimeGraph",
        anomaly_accumulator: "AnomalyAccumulator",
    ) -> List[PhantomNode]:
        """Confirmation by Rupture (anomaly-driven).

        A breakthrough often arrives as an anomaly -- a node that looks nothing
        like the prediction but fills the structural void regardless.
        Standard check_confirmation() misses this because it requires high
        semantic similarity.

        Conditions for confirmation:
          1. new_node.anomaly_density >= RUPTURE_ANOMALY_THRESHOLD, AND
          2. new_node is topologically near the phantom's source/target nodes
             (within RUPTURE_PROXIMITY_HOPS).

        Tagged with confirmation_mode='rupture' so callers can distinguish it.
        """
        node_anomaly = anomaly_accumulator.get_density(new_node.id)
        if node_anomaly < self.RUPTURE_ANOMALY_THRESHOLD:
            return []

        confirmed = []
        for phantom in self.phantoms:
            if phantom.is_confirmed: continue
            phantom_zone_ids = set(phantom.source_node_ids + phantom.target_node_ids)
            if not phantom_zone_ids: continue
            if self._is_near_phantom_zone(new_node.id, phantom_zone_ids, graph,
                                           self.RUPTURE_PROXIMITY_HOPS):
                phantom.is_confirmed = True
                phantom.confirmed_by_node_id = new_node.id
                phantom.__dict__["confirmation_mode"] = "rupture"
                confirmed.append(phantom)
                logger.warning(
                    f"[PhantomGenerator] CONFIRMED by RUPTURE: "
                    f"'{phantom.gap_description[:60]}' filled by anomalous node "
                    f"'{new_node.text[:40]}' (anomaly={node_anomaly:.3f})")
        return confirmed

    def _is_near_phantom_zone(
        self,
        start_id: str,
        target_ids: "Set[str]",
        graph: "RuntimeGraph",
        max_hops: int,
    ) -> bool:
        """Delegate to unified BFS traversal engine."""
        return GRAPH_TRAVERSAL.bfs_reachable(start_id, target_ids, graph, max_hops)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — RETROSPECTIVE CONVERGENCE LOSS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergencePoint:
    id: str
    description: str
    timestamp: float
    convergence_node_ids: List[str]
    prerequisite_paths: List[List[str]] = field(default_factory=list)
    paths_density: float = 0.0


class RetrospectiveConvergenceLoss(nn.Module):
    MAX_PATHS = CONFIG["retrospective_loss_max_paths"]

    def __init__(self):
        super().__init__()

    def forward(self, predicted_convergence_scores, historical_truths, paths_densities):
        if not TORCH_AVAILABLE: return 0.0
        convergence_gap = F.mse_loss(predicted_convergence_scores,
                                      historical_truths, reduction="none")
        return (convergence_gap * (1.0 + paths_densities)).mean()

    def compute_paths_density(self, convergence_point: ConvergencePoint,
                               graph: RuntimeGraph, cutoff_timestamp: float) -> float:
        target_ids = set(convergence_point.convergence_node_ids)
        n_paths = 0
        for nid in list(graph.nodes)[:200]:
            node = graph.get_node(nid)
            if not node or node.timestamp >= cutoff_timestamp: continue
            if nid in target_ids: continue
            n_paths += self._count_paths_to_targets(nid, target_ids, graph, 5, int(self.MAX_PATHS))
            if n_paths >= self.MAX_PATHS: break
        convergence_point.paths_density = n_paths
        return math.log1p(n_paths) / math.log1p(self.MAX_PATHS)

    def _count_paths_to_targets(self, start, targets, graph, max_hops, max_paths):
        """Delegate to unified DFS traversal engine."""
        return GRAPH_TRAVERSAL.dfs_count_paths(start, targets, graph, max_hops, max_paths)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — GRAPH ENTROPY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntropySnapshot:
    region_id: str
    timestamp: float
    entropy: float
    void_pressure_mean: float
    n_nodes: int
    n_edges: int
    entropy_gradient: float = 0.0
    alert: Optional[str] = None


class GraphEntropyMonitor:
    H_MAX = math.log(N_EDGE_COMPONENTS)
    SINGULARITY_DROP_THRESHOLD = CONFIG["entropy_singularity_drop_threshold"]
    PRE_SINGULARITY_ENTROPY    = CONFIG["entropy_pre_singularity_entropy"]
    PRE_SINGULARITY_PRESSURE   = CONFIG["entropy_pre_singularity_pressure"]

    def __init__(self):
        self.snapshots: Dict[str, List[EntropySnapshot]] = defaultdict(list)
        self.current:   Dict[str, EntropySnapshot]       = {}

    def compute_snapshot(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField,
                          timestamp: float) -> Dict[str, EntropySnapshot]:
        domain_edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        for (src, tgt), edge in graph.edges.items():
            if edge.is_containment_edge: continue
            src_node = graph.get_node(src)
            if src_node: domain_edges[src_node.domain].append(edge)

        new_snapshots = {}
        for domain, edges in domain_edges.items():
            entropy = self._compute_entropy(edges)
            domain_node_ids = graph.domain_clusters.get(domain, [])
            pressures = [pressure_field.field.get(nid, VoidPressureVector(nid)).void_pressure
                         for nid in domain_node_ids]
            mean_pressure = float(np.mean(pressures)) if pressures else 0.0
            snap = EntropySnapshot(region_id=domain, timestamp=timestamp,
                                   entropy=entropy, void_pressure_mean=mean_pressure,
                                   n_nodes=len(domain_node_ids), n_edges=len(edges))
            prev = self.current.get(domain)
            if prev: snap.entropy_gradient = entropy - prev.entropy
            snap.alert = self._classify_alert(snap, prev)
            self.snapshots[domain].append(snap)
            self.current[domain] = snap
            new_snapshots[domain] = snap
        return new_snapshots

    def _compute_entropy(self, edges: List[KnowledgeEdge]) -> float:
        if not edges: return 0.0
        component_means = np.zeros(N_EDGE_COMPONENTS)
        for edge in edges:
            for k, comp in enumerate(EDGE_WEIGHT_COMPONENTS):
                component_means[k] += getattr(edge, comp, 0.0)
        component_means /= len(edges)
        total = component_means.sum() + 1e-8
        probs = component_means / total
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy / self.H_MAX)

    def _classify_alert(self, snap: EntropySnapshot,
                         prev: Optional[EntropySnapshot]) -> Optional[str]:
        if (prev and prev.entropy > CONFIG["entropy_high_prev_threshold"]
                and snap.entropy_gradient < -self.SINGULARITY_DROP_THRESHOLD):
            return "singularity_collapse"
        if (snap.entropy > self.PRE_SINGULARITY_ENTROPY
                and snap.void_pressure_mean > self.PRE_SINGULARITY_PRESSURE):
            return "pre_singularity"
        return None

    def get_alerts(self) -> List[Tuple[str, str, EntropySnapshot]]:
        alerts = [(d, snap.alert, snap) for d, snap in self.current.items() if snap.alert]
        alerts.sort(key=lambda x: x[2].void_pressure_mean, reverse=True)
        return alerts

    def predict_entropy_trend(self, window: int = 5) -> Dict[str, float]:
        """Predict per-domain entropy delta for the *next* epoch.

        Uses a simple linear extrapolation over the last *window* snapshots.
        A positive value means entropy is rising (system diversifying);
        negative means it is converging (pre-collapse).

        Returns a dict  {domain: predicted_delta}.  Consumed by
        RecursiveForecastingLoop to modulate branching confidence.
        """
        trends: Dict[str, float] = {}
        for domain, snaps in self.snapshots.items():
            recent = snaps[-window:]
            if len(recent) < 2:
                trends[domain] = 0.0
                continue
            # Least-squares slope of entropy over the last window
            xs = list(range(len(recent)))
            ys = [s.entropy for s in recent]
            n  = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxy = sum(x * y for x, y in zip(xs, ys))
            sxx = sum(x * x for x in xs)
            denom = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / denom if denom != 0 else 0.0
            trends[domain] = float(slope)
        return trends


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: STRUCTURAL SPARSITY REGULATOR
#
#  Prevents exponential blow-up of PhantomNodeGenerator and
#  RecursiveForecastingLoop on large graphs.
#
#  Responsibilities:
#    • prune_weak_nodes  — remove structurally irrelevant nodes below threshold
#    • compress_dense_regions — collapse high-degree hubs into summaries
#    • get_branching_cap — per-node adaptive branching ceiling
# ══════════════════════════════════════════════════════════════════════════════

class StructuralSparsityRegulator:
    """Noise suppressor and branching limiter for large-graph scenarios.

    Without this regulator:
      • PhantomNodeGenerator._find_void_pairs is O(N²) on phantom candidates
      • RecursiveForecastingLoop branches exponentially with MAX_BRANCHES^MAX_DEPTH

    Strategy
    --------
    * Weak nodes — nodes whose void_pressure + forecast_score fall below
      ``weak_node_threshold`` — are marked ineligible for phantom scanning and
      recursive expansion (no hard deletion, to preserve graph integrity).
    * Dense regions — nodes with degree > ``dense_degree_threshold`` — have their
      effective neighbour list capped to the top-K by edge weight when used in
      traversal, preventing fan-out explosions.
    * Branching cap — derived from local pressure: high-pressure nodes get more
      branches; quiet nodes get fewer.
    """

    # Tune in CONFIG or override per-instance
    WEAK_PRESSURE_THRESHOLD = CONFIG["sparsity_weak_pressure_threshold"]   # below this void_pressure → weak
    WEAK_FORECAST_THRESHOLD = CONFIG["sparsity_weak_forecast_threshold"]   # below this forecast_score → weak
    DENSE_DEGREE_THRESHOLD  = CONFIG["sparsity_dense_degree_threshold"]    # edges above this → apply branch cap
    MAX_DENSE_NEIGHBOURS    = CONFIG["sparsity_max_dense_neighbours"]      # top-K neighbours kept for dense nodes
    BRANCH_FLOOR            = CONFIG["sparsity_branch_floor"]
    BRANCH_CEILING          = CONFIG["forecast_max_branches"]

    def is_weak_node(self, node_id: str,
                     graph: "RuntimeGraph",
                     pressure_field: "ConvergencePressureField") -> bool:
        """Return True if the node is structurally insignificant.

        Weak nodes are skipped by PhantomNodeGenerator and RecursiveForecastingLoop
        to avoid polluting the candidate pool.
        """
        node = graph.get_node(node_id)
        if not node or node.is_temporal_zone:
            return True
        pressure = PRESSURE_COMPUTER.node_pressure(node_id, pressure_field)
        if pressure >= self.WEAK_PRESSURE_THRESHOLD:
            return False
        if float(getattr(node, "forecast_score", 0.0)) >= self.WEAK_FORECAST_THRESHOLD:
            return False
        return True

    def get_capped_neighbours(self, node_id: str,
                               graph: "RuntimeGraph") -> List[Tuple[str, "KnowledgeEdge"]]:
        """Return top-K neighbours by edge weight for dense nodes, all for sparse ones."""
        nbrs: List[Tuple[str, "KnowledgeEdge"]] = graph.get_neighbours(node_id)
        if len(nbrs) <= self.DENSE_DEGREE_THRESHOLD:
            return nbrs
        # Dense region: keep only the top MAX_DENSE_NEIGHBOURS by total_weight
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1].total_weight, reverse=True)
        return nbrs_sorted[:self.MAX_DENSE_NEIGHBOURS]

    def get_branching_cap(self, node_id: str,
                          pressure_field: "ConvergencePressureField") -> int:
        """Adaptive branching ceiling proportional to local void_pressure.

        Nodes near singularity get the full BRANCH_CEILING; quiet nodes get only
        BRANCH_FLOOR to prevent silent combinatorial explosion.
        """
        pressure = PRESSURE_COMPUTER.node_pressure(node_id, pressure_field)
        # Normalise to [0, 1] against singularity threshold
        ratio = min(1.0, pressure / (PRESSURE_COMPUTER.SINGULARITY_THRESHOLD + 1e-8))
        cap = self.BRANCH_FLOOR + round(ratio * (self.BRANCH_CEILING - self.BRANCH_FLOOR))
        return max(self.BRANCH_FLOOR, min(self.BRANCH_CEILING, cap))

    def filter_phantom_candidates(
        self,
        candidates: List[Tuple[str, str, float]],
        graph: "RuntimeGraph",
        pressure_field: "ConvergencePressureField",
    ) -> List[Tuple[str, str, float]]:
        """Require factual anchors and cap speculative phantom candidates."""
        factual_pairs = []
        speculative_pairs = []
        min_support = float(CONFIG.get("phantom_min_fact_support", 0.40))
        for a, b, score in candidates:
            if (self.is_weak_node(a, graph, pressure_field)
                    or self.is_weak_node(b, graph, pressure_field)):
                continue
            node_a, node_b = graph.get_node(a), graph.get_node(b)
            if node_a is None or node_b is None:
                continue
            status_a, status_b = node_label_status(node_a), node_label_status(node_b)
            strong_a = (status_a in {LabelStatus.FACT, LabelStatus.PROOF}
                        and node_label_support_score(node_a) >= min_support
                        and self._has_grounded_status_evidence(node_a))
            strong_b = (status_b in {LabelStatus.FACT, LabelStatus.PROOF}
                        and node_label_support_score(node_b) >= min_support
                        and self._has_grounded_status_evidence(node_b))
            ranked_pair = float(score) * max(
                0.1, min(1.0, (node_label_support_score(node_a)
                               + node_label_support_score(node_b)) / 2.0))
            pair = (a, b, float(score))
            if strong_a and strong_b:
                factual_pairs.append((ranked_pair, pair))
            elif (strong_a and status_b in {LabelStatus.IDEA_CONCEPT, LabelStatus.THEORY}) \
                    or (strong_b and status_a in {LabelStatus.IDEA_CONCEPT, LabelStatus.THEORY}):
                speculative_pairs.append((ranked_pair, pair))

        factual_pairs.sort(key=lambda item: item[0], reverse=True)
        speculative_pairs.sort(key=lambda item: item[0], reverse=True)
        rate = min(0.25, max(0.0, float(
            CONFIG.get("phantom_speculative_pair_rate", 0.10))))
        if not speculative_pairs or rate <= 0.0:
            return [pair for _, pair in factual_pairs]
        factual_count = len(factual_pairs)
        if factual_count:
            speculative_limit = max(
                1, int(math.ceil(factual_count * rate / (1.0 - rate))))
        else:
            # A graph may contain only one strong fact plus several weak
            # hypotheses; permit a single exploratory bridge in that case.
            speculative_limit = 1
        return ([pair for _, pair in factual_pairs]
                + [pair for _, pair in speculative_pairs[:speculative_limit]])

    @staticmethod
    def _has_grounded_status_evidence(node: "KnowledgeNode") -> bool:
        """Require a source observation and a supporting quote for factual anchors."""
        for observation in getattr(node, "__dict__", {}).get("source_observations", []) or []:
            if not isinstance(observation, dict):
                continue
            status = (parse_label_status(observation.get("label_status_code"))
                      or parse_label_status(observation.get("label_status")))
            evidence = observation.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if (status in {LabelStatus.FACT, LabelStatus.PROOF}
                    and any(isinstance(quote, str) and quote.strip() for quote in evidence)):
                return True
        return False


# ── Singleton ─────────────────────────────────────────────────────────────────
SPARSITY_REGULATOR = StructuralSparsityRegulator()


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: SCENARIO COHERENCE SCORER
#
#  ForecastScenario was: nodes + edges + probability.
#  Without a structural coherence score, alternative scenarios cannot be ranked
#  meaningfully.  This class computes it and stores it back onto the scenario.
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioCoherenceScorer:
    """Compute a Scenario Structural Coherence Score (SSCS) for ForecastScenario.

    SSCS ∈ [0, 1] is a weighted combination of:
      1. Path continuity  — are consecutive hallucinated nodes connected or
                            near-connected in the base graph?
      2. Temporal monotonicity — do predicted timestamps increase with depth?
      3. Domain coherence — do the hallucinated nodes cluster in few domains?
      4. Confidence gradient — does confidence decay smoothly (no sudden jumps)?

    Higher SSCS → more internally consistent scenario → preferred for ranking.
    """

    W_PATH        = CONFIG["scenario_coherence_w_path"]
    W_TEMPORAL    = CONFIG["scenario_coherence_w_temporal"]
    W_DOMAIN      = CONFIG["scenario_coherence_w_domain"]
    W_CONFIDENCE  = CONFIG["scenario_coherence_w_confidence"]

    def score(self, scenario: "ForecastScenario",
              base_graph: "RuntimeGraph") -> float:
        """Compute and attach SSCS to *scenario.coherence_score*. Returns score."""
        nodes = [h.node for h in scenario.hallucinated_nodes]
        if not nodes:
            scenario.coherence_score = 0.0
            return 0.0

        s_path       = self._path_continuity(scenario, base_graph)
        s_temporal   = self._temporal_monotonicity(scenario)
        s_domain     = self._domain_coherence(scenario)
        s_confidence = self._confidence_smoothness(scenario)

        sscs = (self.W_PATH       * s_path
              + self.W_TEMPORAL   * s_temporal
              + self.W_DOMAIN     * s_domain
              + self.W_CONFIDENCE * s_confidence)
        scenario.coherence_score = float(np.clip(sscs, 0.0, 1.0))
        return scenario.coherence_score

    # ── Sub-scorers ────────────────────────────────────────────────────────────

    def _path_continuity(self, scenario: "ForecastScenario",
                         base_graph: "RuntimeGraph") -> float:
        """Fraction of consecutive (depth-ordered) node pairs that are connected
        or within 2 hops in the base graph."""
        nodes = sorted(scenario.hallucinated_nodes, key=lambda h: h.generation_depth)
        if len(nodes) < 2:
            return 1.0
        connected = 0
        for i in range(len(nodes) - 1):
            a_parents = set(nodes[i].parent_node_ids)
            b_parents = set(nodes[i + 1].parent_node_ids)
            # Overlap in parent IDs → they share a common ancestor in the base graph
            if a_parents & b_parents:
                connected += 1
            elif GRAPH_TRAVERSAL.bfs_reachable(
                nodes[i].node.id, b_parents, base_graph, max_hops=2
            ):
                connected += 1
        return connected / (len(nodes) - 1)

    def _temporal_monotonicity(self, scenario: "ForecastScenario") -> float:
        """Fraction of consecutive timestamp pairs that are non-decreasing."""
        nodes = sorted(scenario.hallucinated_nodes, key=lambda h: h.generation_depth)
        if len(nodes) < 2:
            return 1.0
        mono = sum(
            1 for i in range(len(nodes) - 1)
            if nodes[i + 1].predicted_timestamp >= nodes[i].predicted_timestamp
        )
        return mono / (len(nodes) - 1)

    def _domain_coherence(self, scenario: "ForecastScenario") -> float:
        """Score based on domain diversity: 1 domain = 1.0, many = lower."""
        nodes = scenario.hallucinated_nodes
        if not nodes:
            return 1.0
        domains = set(h.node.domain for h in nodes)
        # Normalise: 1 domain → 1.0; N domains → 1/N
        return 1.0 / len(domains)

    def _confidence_smoothness(self, scenario: "ForecastScenario") -> float:
        """Penalise abrupt drops in confidence_at_depth."""
        c = scenario.confidence_at_depth
        if len(c) < 2:
            return 1.0
        deltas = [abs(c[i] - c[i - 1]) / max(c[i - 1], 1e-8)
                  for i in range(1, len(c))]
        mean_drop = float(np.mean(deltas))
        # mean_drop = 0 → perfectly smooth = 1.0; DECAY_FACTOR ≈ 0.35 drop → ~0.5
        return float(np.exp(-mean_drop))


# ── Singleton ─────────────────────────────────────────────────────────────────
COHERENCE_SCORER = ScenarioCoherenceScorer()


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — RECURSIVE FORECASTING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HallucinatedNode:
    node: KnowledgeNode
    generation_depth: int
    confidence: float
    parent_node_ids: List[str]
    predicted_timestamp: float
    is_secondary_effect: bool


@dataclass
class ForecastScenario:
    scenario_id: str
    description: str
    depth: int
    hallucinated_nodes: List[HallucinatedNode]
    predicted_edges: List[KnowledgeEdge]
    confidence_at_depth: List[float]
    secondary_effects: List[str]
    collapse_probability: float = 0.0
    coherence_score: float = 0.0      # ← NEW: computed by ScenarioCoherenceScorer


class RecursiveForecastingLoop:
    MIN_CONFIDENCE  = CONFIG["forecast_min_confidence"]
    DECAY_FACTOR    = CONFIG["forecast_decay_factor"]
    MAX_DEPTH       = CONFIG["forecast_max_depth"]
    MAX_BRANCHES    = CONFIG["forecast_max_branches"]
    YEARS_PER_DEPTH = CONFIG["forecast_years_per_depth"]

    def __init__(self, graph_walker: GraphWalker, edge_predictor,
                 pressure_field: ConvergencePressureField,
                 phantom_gen: PhantomNodeGenerator,
                 entropy_monitor: Optional["GraphEntropyMonitor"] = None):
        self.walker         = graph_walker
        self.edge_pred      = edge_predictor
        self.pressure       = pressure_field
        self.phantom_gen    = phantom_gen
        # GraphEntropyMonitor is optional for backward compatibility.
        # When provided, predict_entropy_trend() modulates branching confidence.
        self.entropy_monitor = entropy_monitor

    def forecast(self, seed_node_ids: List[str], base_graph: RuntimeGraph,
                  n_scenarios: int = 3) -> List[ForecastScenario]:
        # ── Pull entropy trend once per forecast call ──────────────────────────
        entropy_trend: Dict[str, float] = {}
        if self.entropy_monitor is not None:
            entropy_trend = self.entropy_monitor.predict_entropy_trend()

        scenarios = []
        for seed_id in seed_node_ids[:n_scenarios]:
            scenario = self._run_scenario(seed_id, base_graph, len(scenarios),
                                          entropy_trend)
            if scenario:
                # Score coherence before returning
                COHERENCE_SCORER.score(scenario, base_graph)
                scenarios.append(scenario)

        # Sort primarily by coherence_score, break ties with confidence depth
        scenarios.sort(
            key=lambda s: (s.coherence_score,
                           s.confidence_at_depth[-1] if s.confidence_at_depth else 0),
            reverse=True,
        )
        return scenarios

    def _run_scenario(self, seed_id: str, base_graph: RuntimeGraph,
                       scenario_idx: int,
                       entropy_trend: Optional[Dict[str, float]] = None
                       ) -> Optional[ForecastScenario]:
        seed = base_graph.get_node(seed_id)
        if not seed: return None
        hallucinated_nodes: List[HallucinatedNode] = []
        predicted_edges:    List[KnowledgeEdge]    = []
        confidence_per_depth: List[float]           = [1.0]
        current_confidence = 1.0
        secondary_effects:  List[str]               = []
        hallucinated_ids: Set[str] = set()
        current_frontier = {seed_id}
        entropy_trend = entropy_trend or {}

        for depth in range(1, self.MAX_DEPTH + 1):
            # ── Decay via ConfidenceDecayEngine (geometric step) ───────────────
            current_confidence = CONFIDENCE_ENGINE.geometric_decay(current_confidence)

            # ── Entropy trend modulation ───────────────────────────────────────
            # Negative trend (converging) → boost confidence (system focusing)
            # Positive trend (diverging)  → suppress confidence (system scattering)
            domain_trend = entropy_trend.get(seed.domain, 0.0)
            entropy_mod  = float(np.exp(-max(0.0, domain_trend) * 2.0))
            current_confidence *= entropy_mod

            if current_confidence < self.MIN_CONFIDENCE:
                break

            new_frontier: Set[str] = set()
            for node_id in list(current_frontier)[:self.MAX_BRANCHES]:
                # ── Adaptive branching via SparsityRegulator ──────────────────
                branch_cap = SPARSITY_REGULATOR.get_branching_cap(node_id, self.pressure)
                next_nodes = self._predict_next_nodes(
                    node_id, base_graph, hallucinated_ids, depth, current_confidence)
                for h_node in next_nodes[:branch_cap]:
                    # ── plateau + entropy penalty on each hallucinated node ───
                    node_conf = CONFIDENCE_ENGINE.plateau_penalty(h_node.node, base_graph)
                    if self.entropy_monitor is not None:
                        node_conf *= CONFIDENCE_ENGINE.entropy_alert_factor(
                            h_node.node.domain, self.entropy_monitor)
                    h_node.confidence *= node_conf

                    hallucinated_nodes.append(h_node)
                    hallucinated_ids.add(h_node.node.id)
                    new_frontier.add(h_node.node.id)
                    if depth >= 2:
                        secondary_effects.append(
                            f"Depth {depth}: '{h_node.node.text[:50]}' (conf={h_node.confidence:.2f})")
            if not new_frontier:
                break
            confidence_per_depth.append(current_confidence)
            current_frontier = new_frontier

        if not hallucinated_nodes: return None
        return ForecastScenario(
            scenario_id=f"scenario_{scenario_idx}_{seed_id[:8]}",
            description=f"Forward projection from '{seed.text[:50]}'",
            depth=len(confidence_per_depth) - 1,
            hallucinated_nodes=hallucinated_nodes,
            predicted_edges=predicted_edges,
            confidence_at_depth=confidence_per_depth,
            secondary_effects=secondary_effects,
            collapse_probability=max(0.0, 1.0 - current_confidence / self.MIN_CONFIDENCE))

    # ── Incumbent Reaction constants ─────────────────────────────────────────
    # Threshold forecast_score above which a hallucinated node triggers
    # defensive inhibitory phantom edges from domain incumbents.
    INCUMBENT_THREAT_THRESHOLD = CONFIG["incumbent_threat_threshold"]
    PLATEAU_PENALTY_SCALE      = CONFIG["plateau_penalty_scale"]

    def _apply_incumbent_reaction(
        self,
        hallucinated_node: "KnowledgeNode",
        base_graph: "RuntimeGraph",
    ) -> List["KnowledgeEdge"]:
        """Domain Self-Preservation Instinct.

        For every incumbent node in the same domain whose strategic_value > 8,
        generate a phantom inhibitory KnowledgeEdge proportional to both the
        breakthrough's forecast_score and the incumbent's strategic stake.

        These edges are NOT persisted to the graph; they are attached to
        HallucinatedNode.incumbent_inhibitory_edges so the Lag Model can
        delay predicted_timestamp accordingly.
        """
        phantom_edges: List[KnowledgeEdge] = []
        h_forecast = float(np.clip(hallucinated_node.forecast_score, 0.0, 10.0))
        if h_forecast / 10.0 < self.INCUMBENT_THREAT_THRESHOLD:
            return phantom_edges

        competitors = [
            n for n in base_graph.nodes.values()
            if (not n.is_temporal_zone
                and n.strategic_value > CONFIG["incumbent_sv_threshold"]
                and n.domain == hallucinated_node.domain)
        ]
        for comp in competitors:
            force = (h_forecast / 10.0) * (comp.strategic_value / 10.0)
            inh_edge = KnowledgeEdge(
                id=f"incumbent_inh_{comp.id[:6]}_{hallucinated_node.id[:6]}",
                source_id=comp.id,
                target_id=hallucinated_node.id,
                relationship_type="incumbent_inhibition",
                inhibitory_force=float(np.clip(force, 0.0, 1.0)),
                timestamp=hallucinated_node.timestamp,
                confidence=force,
            )
            inh_edge.compute_total_weight()
            phantom_edges.append(inh_edge)
            logger.info(
                f"[IncumbentReaction] '{comp.text[:40]}' (sv={comp.strategic_value:.1f}) "
                f"-> inhibitory against '{hallucinated_node.text[:40]}' force={force:.3f}"
            )
        return phantom_edges

    def _efficiency_plateau_penalty(
        self,
        hallucinated_node: "KnowledgeNode",
        base_graph: "RuntimeGraph",
    ) -> float:
        """Delegate to ConfidenceDecayEngine (canonical plateau logic)."""
        return CONFIDENCE_ENGINE.plateau_penalty(hallucinated_node, base_graph)

    def _predict_next_nodes(self, node_id, base_graph, already_hallucinated,
                             depth, confidence) -> List[HallucinatedNode]:
        seed = base_graph.get_node(node_id)
        if not seed: return []
        nbrs = base_graph.get_neighbours(node_id)
        phantom_candidates = [p for p in self.phantom_gen.phantoms
                              if node_id in p.source_node_ids and not p.is_confirmed]
        results = []
        for phantom in phantom_candidates[:self.MAX_BRANCHES]:
            h_node = KnowledgeNode(
                id=f"halluc_{phantom.id}_{depth}",
                text=phantom.gap_description[:80],
                node_type="hallucinated",
                domain=phantom.predicted_domain,
                entity_type=phantom.predicted_entity_type,
                timestamp=(seed.timestamp + depth * self.YEARS_PER_DEPTH * 365.25 * 24 * 3600))

            # Incumbent reaction: slow down breakthrough via phantom inhibitory edges
            incumbent_edges = self._apply_incumbent_reaction(h_node, base_graph)
            lag_years = sum(e.inhibitory_force for e in incumbent_edges) * CONFIG["incumbent_lag_years_per_force"]
            h_node.timestamp += lag_years * 365.25 * 24 * 3600

            # Efficiency plateau penalty
            plateau_factor = self._efficiency_plateau_penalty(h_node, base_graph)
            raw_conf = confidence * phantom.structural_gap_score * plateau_factor

            h_item = HallucinatedNode(
                node=h_node, generation_depth=depth,
                confidence=raw_conf,
                parent_node_ids=[node_id],
                predicted_timestamp=h_node.timestamp,
                is_secondary_effect=(depth >= 2))
            if incumbent_edges:
                h_item.__dict__["incumbent_inhibitory_edges"] = incumbent_edges
            results.append(h_item)

        for nbr_id, edge in nbrs[:self.MAX_BRANCHES]:
            if nbr_id in already_hallucinated: continue
            nbr = base_graph.get_node(nbr_id)
            if not nbr: continue
            proj_node = KnowledgeNode(
                id=f"halluc_proj_{nbr_id}_{depth}",
                text=f"Projected: {nbr.text[:60]}",
                node_type="hallucinated_projection",
                domain=nbr.domain, entity_type=nbr.entity_type,
                timestamp=(nbr.timestamp + depth * self.YEARS_PER_DEPTH * 365.25 * 24 * 3600),
                forecast_score=nbr.forecast_score * (self.DECAY_FACTOR ** depth))

            incumbent_edges = self._apply_incumbent_reaction(proj_node, base_graph)
            lag_years = sum(e.inhibitory_force for e in incumbent_edges) * CONFIG["incumbent_lag_years_per_force"]
            proj_node.timestamp += lag_years * 365.25 * 24 * 3600

            plateau_factor = self._efficiency_plateau_penalty(proj_node, base_graph)
            raw_conf = confidence * edge.total_weight * plateau_factor

            p_item = HallucinatedNode(
                node=proj_node, generation_depth=depth,
                confidence=raw_conf,
                parent_node_ids=[node_id, nbr_id],
                predicted_timestamp=proj_node.timestamp,
                is_secondary_effect=(depth >= 2))
            if incumbent_edges:
                p_item.__dict__["incumbent_inhibitory_edges"] = incumbent_edges
            results.append(p_item)

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — PATH-AGNOSTIC INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceCloud:
    query: str
    convergence_domain: str
    predicted_year_range: Tuple[int, int]
    n_independent_paths: int
    redundancy_score: float
    total_attention_flow: float
    convergence_probability: float
    top_trajectories: List[Trajectory]
    void_pressure_at_target: float
    active_phantoms: List[PhantomNode]
    entropy_alerts: List[Tuple[str, str]]

    def to_readable(self) -> str:
        lines = [
            f"QUERY: {self.query}", "",
            f"CONVERGENCE DETECTED: {self.convergence_domain}",
            f"Predicted window:     {self.predicted_year_range[0]}–{self.predicted_year_range[1]}", "",
            f"INEVITABILITY ASSESSMENT",
            f"  Independent paths:    {self.n_independent_paths}",
            f"  Redundancy score:     {self.redundancy_score:.2f}",
            f"  Attention flow:       {self.total_attention_flow:.3f}",
            f"  Convergence prob:     {self.convergence_probability:.1%}",
            f"  Void pressure:        {self.void_pressure_at_target:.2f}", "",
        ]
        if self.active_phantoms:
            lines.append(f"PREDICTED BRIDGE TECHNOLOGIES ({len(self.active_phantoms)}):")
            for p in self.active_phantoms[:3]:
                lines.append(f"  [{p.structural_gap_score:.2f}] {p.gap_description[:60]}")
        if self.entropy_alerts:
            lines.append(f"\nENTROPY ALERTS:")
            for domain, alert_type in self.entropy_alerts[:3]:
                lines.append(f"  {alert_type.upper()}: {domain}")
        if self.top_trajectories:
            lines.append(f"\nBEST PATHS (top {len(self.top_trajectories)}):")
            for i, traj in enumerate(self.top_trajectories):
                lines.append(f"  Path {i+1} [{traj.total_confidence:.3f}]: "
                             + " → ".join(s.node_text[:25] for s in traj.steps))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-CONVERGENCE ALERT  –  rich structured report for nodes that have
#  crossed the emergence threshold.
#
#  Usage:
#      detector = HighConvergenceDetector(threshold=72.5)
#      alerts   = detector.scan(graph)
#      for alert in alerts:
#          print(alert.to_readable())
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentMaturity:
    """Maturity level of a single technical component (0–100 %)."""
    name:    str
    percent: float   # 0–100
    # ✓ if percent >= 80, ⚠ if 60–79, ✗ below 60
    @property
    def status_icon(self) -> str:
        if self.percent >= 80:
            return "✓"
        if self.percent >= 60:
            return "⚠"
        return "✗"

    def format_line(self, width: int = 40) -> str:
        label = f"├─ {self.name}:"
        return f"{label:<{width}} {self.percent:.0f}%  {self.status_icon}"


@dataclass
class BottleneckItem:
    """A single unresolved barrier on the path to emergence."""
    description:    str
    solved_pct:     float   # fraction already resolved, 0–100
    months_to_resolve: str  # e.g. "12-18 months"

    def format_line(self) -> str:
        return (f"├─ {self.description} "
                f"({self.solved_pct:.0f}% solved, {self.months_to_resolve} to resolve)")


@dataclass
class HistoricalAnalogue:
    """A past breakthrough that followed a similar convergence pattern."""
    name:    str    # e.g. "Laser (1958→1960)"
    comment: str    # e.g. "Similar cross-domain convergence pattern"

    def format_line(self) -> str:
        return f"- {self.name}: {self.comment}"


@dataclass
class LikelyBuilder:
    """An organisation most likely to realise the predicted technology."""
    rank:        int
    name:        str
    probability: float   # 0–100 %
    rationale:   str

    def format_line(self) -> str:
        return f"{self.rank}. {self.name} ({self.probability:.0f}%) - {self.rationale}"


@dataclass
class PredictedForm:
    """One expected property of the emerging technology."""
    trait: str   # e.g. "Cloud-based platform (not standalone software)"

    def format_line(self) -> str:
        return f"├─ {self.trait}"


@dataclass
class MonitoringSignal:
    """A leading indicator to watch before full emergence."""
    signal: str   # e.g. "Joint publications (GNN + LLM + Forecasting authors)"

    def format_line(self) -> str:
        return f"├─ {self.signal}"


@dataclass
class HighConvergenceAlert:
    """
    Full-detail alert emitted when a knowledge-graph node crosses the
    convergence threshold.  Covers maturity, bottlenecks, historical
    analogues, likely builders, predicted form, and monitoring signals.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    node_id:     str
    label:       str
    description: str

    # ── Scores ────────────────────────────────────────────────────────────────
    convergence_score: float   # 0–100
    threshold:         float   # 0–100

    # ── Component breakdown ───────────────────────────────────────────────────
    components:  List[ComponentMaturity]

    # ── Barriers ──────────────────────────────────────────────────────────────
    bottlenecks: List[BottleneckItem]

    # ── Historical context ────────────────────────────────────────────────────
    analogues:   List[HistoricalAnalogue]

    # ── Temporal prediction ───────────────────────────────────────────────────
    emergence_window:        str    # e.g. "Q2-Q3 2026"
    confidence_interval:     str    # e.g. "[Q4 2025, Q4 2027]"
    probability:             float  # 0–100 %
    probability_margin:      float  # ± value, e.g. 2.8

    # ── Actor analysis ────────────────────────────────────────────────────────
    likely_builders: List[LikelyBuilder]

    # ── Technology profile ────────────────────────────────────────────────────
    predicted_forms:    List[PredictedForm]

    # ── Surveillance ──────────────────────────────────────────────────────────
    monitoring_signals: List[MonitoringSignal]

    # ── Optional reflexive note ───────────────────────────────────────────────
    self_awareness_note: str = ""

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def above_threshold_pct(self) -> float:
        """How many percent above the threshold the score sits."""
        if self.threshold == 0:
            return 0.0
        return (self.convergence_score - self.threshold) / self.threshold * 100.0

    @property
    def status_label(self) -> str:
        """Human-readable status based on how far above threshold."""
        pct = self.above_threshold_pct
        if pct >= 20:
            return f"CRITICAL MASS ACHIEVED ({pct:.0f}% above threshold)"
        if pct >= 10:
            return f"THRESHOLD EXCEEDED ({pct:.0f}% above threshold)"
        if pct >= 0:
            return f"THRESHOLD MET ({pct:.0f}% above threshold)"
        return "BELOW THRESHOLD"

    def to_readable(self) -> str:
        """
        Render the alert as the canonical multi-section text report.
        Designed to be printed directly or forwarded to a logger.
        """
        W = 70   # total width for separator lines
        SEP = "─" * W

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            "█" * W,
            f"  ALERT: High-Convergence Cluster Detected",
            "█" * W,
            "",
            f"Node ID:     {self.node_id}",
            f"Label:       \"{self.label}\"",
            "",
            "Description:",
        ]
        # Wrap description at ~65 chars with consistent indentation
        for chunk in self._wrap(self.description, 65):
            lines.append(f"  {chunk}")

        # ── Score block ───────────────────────────────────────────────────────
        lines += [
            "",
            SEP,
            f"Convergence Score:  {self.convergence_score:.1f} / 100",
            f"Threshold:          {self.threshold:.1f}",
            f"Status:             {self.status_label}",
        ]

        # ── Component maturity ────────────────────────────────────────────────
        if self.components:
            lines += ["", SEP, "Component Maturity:"]
            for c in self.components:
                lines.append(c.format_line())

        # ── Bottlenecks ───────────────────────────────────────────────────────
        if self.bottlenecks:
            lines += ["", SEP, "Primary Bottlenecks:"]
            for b in self.bottlenecks:
                lines.append(b.format_line())

        # ── Historical analogues ──────────────────────────────────────────────
        if self.analogues:
            lines += ["", SEP, "Historical Analogues:"]
            for a in self.analogues:
                lines.append(a.format_line())

        # ── Temporal forecast ─────────────────────────────────────────────────
        lines += [
            "",
            SEP,
            f"Emergence Window:     {self.emergence_window}",
            f"Confidence Interval:  {self.confidence_interval}",
            f"Probability:          {self.probability:.1f}% (±{self.probability_margin:.1f}%)",
        ]

        # ── Likely builders ───────────────────────────────────────────────────
        if self.likely_builders:
            lines += ["", SEP, "Most Likely Builders:"]
            for b in self.likely_builders:
                lines.append(b.format_line())

        # ── Predicted form ────────────────────────────────────────────────────
        if self.predicted_forms:
            lines += ["", SEP, "Predicted Form:"]
            for f_ in self.predicted_forms:
                lines.append(f_.format_line())

        # ── Monitoring signals ────────────────────────────────────────────────
        if self.monitoring_signals:
            lines += ["", SEP, "Recommended Monitoring Signals (check monthly):"]
            for s in self.monitoring_signals:
                lines.append(s.format_line())

        # ── Self-awareness note ───────────────────────────────────────────────
        if self.self_awareness_note:
            lines += ["", SEP, "System Self-Awareness Note:"]
            for chunk in self._wrap(self.self_awareness_note, 65):
                lines.append(f"  {chunk}")

        lines += ["", "█" * W, ""]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """
        Simple word-wrap: splits on spaces, respects existing newlines,
        produces lines no longer than *width* characters.
        """
        result: List[str] = []
        for paragraph in text.splitlines():
            words  = paragraph.split()
            line   = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) <= width:
                    line = candidate
                else:
                    if line:
                        result.append(line)
                    line = word
            if line:
                result.append(line)
        return result or [""]


class HighConvergenceDetector:
    """
    Scans a RuntimeGraph and emits HighConvergenceAlert objects for every
    node whose convergence_potential (scaled 0–100) exceeds *threshold*.

    The detector derives component maturity from the node's physical-axis
    scores (scientific_score, maturity_score, etc.) so it works without any
    additional input beyond what MotherAnnotator / IngestionPipeline already
    populate.

    Example
    -------
    detector = HighConvergenceDetector(threshold=72.5)
    alerts   = detector.scan(graph)
    for alert in alerts:
        print(alert.to_readable())
    """

    # Mapping: (display_name, KnowledgeNode attribute name)
    _COMPONENT_MAP: List[Tuple[str, str]] = [
        ("LLM semantic understanding",      "scientific_score"),
        ("Graph neural networks",           "structural_dependency_index"),
        ("Knowledge graph infrastructure",  "cascade_influence"),
        ("Temporal modeling",               "maturity_score"),
        ("Multi-modal integration",         "social_score"),
        ("Forecasting methodology",         "readiness_score"),
        ("Interpretability",                "investment_score"),
    ]

    # Scale factor: node scores are 0–10, components expect 0–100
    _SCORE_SCALE = 10.0

    def __init__(
        self,
        threshold: float = 72.5,
        # Bottlenecks, analogues, builders, etc. can be injected at
        # construction time so domain experts can customise them without
        # subclassing.
        default_bottlenecks:       Optional[List[BottleneckItem]]    = None,
        default_analogues:         Optional[List[HistoricalAnalogue]] = None,
        default_likely_builders:   Optional[List[LikelyBuilder]]     = None,
        default_predicted_forms:   Optional[List[PredictedForm]]     = None,
        default_monitoring_signals: Optional[List[MonitoringSignal]] = None,
        default_self_awareness_note: str = "",
    ) -> None:
        self.threshold               = threshold
        self.default_bottlenecks     = default_bottlenecks     or []
        self.default_analogues       = default_analogues       or []
        self.default_likely_builders = default_likely_builders or []
        self.default_predicted_forms = default_predicted_forms or []
        self.default_monitoring_signals = default_monitoring_signals or []
        self.default_self_awareness_note = default_self_awareness_note

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(self, graph: "RuntimeGraph") -> List[HighConvergenceAlert]:
        """
        Return one HighConvergenceAlert per node that exceeds the threshold,
        sorted by convergence score descending (highest priority first).
        """
        alerts: List[HighConvergenceAlert] = []
        for node in graph.nodes.values():
            score = self._node_score(node)
            if score >= self.threshold:
                alerts.append(self._build_alert(node, score))
        alerts.sort(key=lambda a: a.convergence_score, reverse=True)
        return alerts

    def alert_for_node(self, node: "KnowledgeNode") -> Optional[HighConvergenceAlert]:
        """
        Build an alert for a specific node regardless of whether it is
        part of a RuntimeGraph.  Returns None if the node is below threshold.
        """
        score = self._node_score(node)
        if score < self.threshold:
            return None
        return self._build_alert(node, score)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _node_score(node: "KnowledgeNode") -> float:
        """
        Map convergence_potential (0–1 or 0–10) to a 0–100 score.
        If convergence_potential is not set, fall back to the mean of
        scientific_score, maturity_score, and readiness_score.
        """
        pot = node.convergence_potential
        if pot > 0:
            # convergence_potential is stored as 0–1 by most components
            return min(pot * 100.0, 100.0) if pot <= 1.0 else min(pot * 10.0, 100.0)
        # Fallback: average of the three most reliable maturity proxies
        proxies = [node.scientific_score, node.maturity_score, node.readiness_score]
        valid   = [v for v in proxies if v > 0]
        if not valid:
            return 0.0
        return min(sum(valid) / len(valid) * 10.0, 100.0)   # scores are 0–10

    def _build_components(self, node: "KnowledgeNode") -> List[ComponentMaturity]:
        """
        Derive component maturities from the node's scalar attributes.
        Each score is on 0–10 scale → multiply by _SCORE_SCALE for percent.
        """
        result: List[ComponentMaturity] = []
        for name, attr in self._COMPONENT_MAP:
            raw = getattr(node, attr, 0.0) or 0.0
            # Normalise: if score looks like 0–1 range, rescale to 0–100
            pct = (raw * 100.0) if raw <= 1.0 else (raw * self._SCORE_SCALE)
            pct = min(max(pct, 0.0), 100.0)
            result.append(ComponentMaturity(name=name, percent=pct))
        return result

    def _derive_emergence_window(self, node: "KnowledgeNode") -> Tuple[str, str]:
        """
        Estimate a qualitative emergence window from temporal attributes.
        Returns (window_label, confidence_interval_label).
        """
        now   = datetime.utcnow()
        year  = now.year
        month = now.month

        # Use maturity_score as a proxy for how close emergence is:
        # maturity close to 10 → very soon; close to 0 → further out
        maturity = min(max(node.maturity_score or 5.0, 0.0), 10.0)
        months_out = max(1, int((10.0 - maturity) * 4))   # 1–40 months

        def quarter(m: int, y: int) -> str:
            q = (m - 1) // 3 + 1
            return f"Q{q} {y}"

        def add_months(base_month: int, base_year: int, n: int) -> Tuple[int, int]:
            total  = base_month - 1 + n
            return (total % 12) + 1, base_year + total // 12

        mid_m, mid_y   = add_months(month, year, months_out)
        early_m, early_y = add_months(month, year, max(1, months_out - 3))
        late_m, late_y  = add_months(month, year, months_out + 6)

        window = f"{quarter(mid_m, mid_y)}"
        ci     = f"[{quarter(early_m, early_y)}, {quarter(late_m, late_y)}]"
        return window, ci

    def _build_alert(self, node: "KnowledgeNode", score: float) -> HighConvergenceAlert:
        """Assemble a complete HighConvergenceAlert from a KnowledgeNode."""
        window, ci = self._derive_emergence_window(node)

        # Probability: derived from convergence_potential and void_pressure proxy
        probability = min(score + (node.upstream_pressure or 0.0) * 5.0, 99.9)
        margin      = max(1.0, (100.0 - probability) / 5.0)

        return HighConvergenceAlert(
            node_id           = node.id,
            label             = node.text,
            description       = node.full_text or node.text,
            convergence_score = score,
            threshold         = self.threshold,
            components        = self._build_components(node),
            bottlenecks       = self.default_bottlenecks,
            analogues         = self.default_analogues,
            emergence_window  = window,
            confidence_interval = ci,
            probability       = probability,
            probability_margin  = margin,
            likely_builders   = self.default_likely_builders,
            predicted_forms   = self.default_predicted_forms,
            monitoring_signals  = self.default_monitoring_signals,
            self_awareness_note = self.default_self_awareness_note,
        )


class PathAgnosticInference:
    def __init__(self, graph_walker: GraphWalker,
                 pressure_field: ConvergencePressureField,
                 phantom_gen: PhantomNodeGenerator,
                 entropy_monitor: GraphEntropyMonitor,
                 priority_head: ContextualPriorityHead,
                 dormancy_tracker: DormancyTracker):
        self.walker   = graph_walker
        self.pressure = pressure_field
        self.phantoms = phantom_gen
        self.entropy  = entropy_monitor
        self.priority = priority_head
        self.dormancy = dormancy_tracker

    def query(self, query_text: str, embedder, graph: RuntimeGraph,
              n_paths: int = 20, future_horizon: float = 5.0) -> ConvergenceCloud:
        trajectories = self.walker.query(query_text, embedder, max_trajectories=n_paths)
        if not trajectories:
            return ConvergenceCloud(query=query_text, convergence_domain="no convergence detected",
                                    predicted_year_range=(0, 0), n_independent_paths=0,
                                    redundancy_score=0.0, total_attention_flow=0.0,
                                    convergence_probability=0.0, top_trajectories=[],
                                    void_pressure_at_target=0.0, active_phantoms=[],
                                    entropy_alerts=[])
        domain_counts: Dict[str, int] = defaultdict(int)
        for traj in trajectories:
            if traj.steps: domain_counts[traj.steps[-1].node_domain] += 1
        convergence_domain = max(domain_counts, key=domain_counts.get)
        node_sets = [frozenset(s.node_id for s in t.steps) for t in trajectories]
        n_independent = sum(1 for i, ns in enumerate(node_sets)
                            if all(len(ns & prev) / len(ns | prev) < CONFIG["path_agnostic_independence_overlap_threshold"]
                                   for prev in node_sets[:i]))
        redundancy  = math.log1p(n_independent)
        # TODO #10 fix: this used to sum t.total_confidence over every
        # trajectory unconditionally, treating heavily-overlapping paths
        # (which n_independent, right above, already knows how to detect via
        # node_sets) as fully independent contributions — the same
        # "diamond topology" double counting as ConvergencePressureField's
        # excess_investment_pressure, just for path confidence instead of
        # investment. Reuses the SAME node_sets and the SAME processing
        # order as n_independent (so the two stay conceptually consistent):
        # each trajectory's contribution is scaled by the fraction of its
        # own nodes not already covered by an earlier trajectory in the
        # list, rather than an all-or-nothing threshold cut — a trajectory
        # that's 80% redundant with what's already been counted contributes
        # about 20% of its confidence, not 0% or 100%.
        total_flow = 0.0
        _flow_seen_nodes: Set[str] = set()
        for traj, ns in zip(trajectories, node_sets):
            novel_fraction = 1.0 if not ns else len(ns - _flow_seen_nodes) / len(ns)
            total_flow += (traj.total_confidence *
                          (1.0 + CONFIG["path_agnostic_cross_domain_bonus"] * traj.cross_domain_hops) *
                          novel_fraction)
            _flow_seen_nodes |= ns
        conv_prob   = min(0.99, 1.0 - math.exp(-redundancy * CONFIG["path_agnostic_conv_prob_decay"]))
        endpoint_ids = [t.steps[-1].node_id for t in trajectories if t.steps]
        pressures = [self.pressure.field.get(nid, VoidPressureVector(nid)).void_pressure
                     for nid in endpoint_ids]
        mean_pressure = float(np.mean(pressures)) if pressures else 0.0
        end_years = [_ts_to_year(t.steps[-1].node_timestamp) for t in trajectories if t.steps]
        y_min = min(end_years) if end_years else 2025
        y_max = max(end_years) + int(future_horizon) if end_years else 2030
        relevant_phantoms = [p for p in self.phantoms.phantoms
                             if not p.is_confirmed and p.predicted_domain
                             and convergence_domain.lower()[:CONFIG["path_agnostic_domain_match_chars"]] in p.predicted_domain.lower()][:CONFIG["path_agnostic_max_phantoms"]]
        alerts = [(d, t) for d, t, _ in self.entropy.get_alerts()
                  if convergence_domain.lower()[:CONFIG["path_agnostic_alert_match_chars"]] in d.lower()][:CONFIG["path_agnostic_max_alerts"]]
        return ConvergenceCloud(
            query=query_text, convergence_domain=convergence_domain,
            predicted_year_range=(y_min, y_max), n_independent_paths=n_independent,
            redundancy_score=redundancy, total_attention_flow=total_flow,
            convergence_probability=conv_prob,
            top_trajectories=sorted(trajectories, key=lambda t: t.total_confidence,
                                    reverse=True)[:CONFIG["path_agnostic_top_trajectories"]],
            void_pressure_at_target=mean_pressure,
            active_phantoms=relevant_phantoms, entropy_alerts=alerts)

__all__ = ['PhantomNode', 'PhantomNodeGenerator', 'ConvergencePoint', 'RetrospectiveConvergenceLoss', 'EntropySnapshot', 'GraphEntropyMonitor', 'StructuralSparsityRegulator', 'SPARSITY_REGULATOR', 'ScenarioCoherenceScorer', 'COHERENCE_SCORER', 'HallucinatedNode', 'ForecastScenario', 'RecursiveForecastingLoop', 'ConvergenceCloud', 'ComponentMaturity', 'BottleneckItem', 'HistoricalAnalogue', 'LikelyBuilder', 'PredictedForm', 'MonitoringSignal', 'HighConvergenceAlert', 'HighConvergenceDetector', 'PathAgnosticInference']
