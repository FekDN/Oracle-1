"""Convergence pressure, dormancy, and local graph dynamics."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG
from .entities import KnowledgeEdge
from .graph import RuntimeGraph
from .significance import EpistemicLevel
from .time_utils import _ts_to_year

# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — CONVERGENCE PRESSURE FIELD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VoidPressureVector:
    node_id: str
    void_pressure: float = 0.0
    unresolved_limitation_pressure: float = 0.0
    cultural_phantom_pressure:      float = 0.0
    excess_investment_pressure:     float = 0.0
    inhibitory_sink_pressure:       float = 0.0
    temporal_zone_pressure:         float = 0.0
    pressure_history: List[float] = field(default_factory=list)


class ConvergencePressureField:
    SINGULARITY_THRESHOLD = CONFIG["pressure_singularity_threshold"]
    PRESSURE_DECAY_RATE   = CONFIG["pressure_decay_rate"]

    def __init__(self):
        self.field: Dict[str, VoidPressureVector] = {}
        # Fingerprint snapshots from the last compute()/update() call. update()
        # diffs against these to find genuinely-changed nodes/edges (Manifest
        # §3, "Волновое распространение изменений"). None until the first
        # call — that absence is exactly what forces update()'s cold-start
        # fall-through to a full compute() below.
        self._last_node_fp: Optional[Dict[str, tuple]] = None
        self._last_edge_fp: Optional[Dict[Tuple[str, str], tuple]] = None
        # TODO #22: the node ids this class itself identified as genuinely
        # changed on the most recent compute()/update() call — i.e. exactly
        # the seed set a wave should start from. Exposed so other per-epoch
        # subsystems (DormancyTracker.update_wave, ...) that need "what
        # actually changed since last time" don't have to re-derive their
        # own separate fingerprint diff to answer the same question —
        # Oracle1Extended.update_epoch() runs pressure_field.update() first
        # and reuses this for dormancy immediately after.
        self._last_changed: Set[str] = set()

    def compute(self, graph: RuntimeGraph) -> Dict[str, VoidPressureVector]:
        for nid in graph.nodes:
            self.field[nid] = self._compute_node_pressure(nid, graph)
        self._last_node_fp, self._last_edge_fp = self._fingerprint_snapshot(graph)
        self._last_changed = set(graph.nodes)  # cold start: everything is "new"
        return self.field

    @staticmethod
    def _fingerprint_snapshot(
        graph: RuntimeGraph
    ) -> Tuple[Dict[str, tuple], Dict[Tuple[str, str], tuple]]:
        """
        Cheap per-node/per-edge tuples covering exactly the fields
        _compute_node_pressure() actually reads: strategic_value /
        investment_total_usd / zone_id / acceleration_multiplier /
        entity_type / social_gravity / forecast_score / domain for nodes;
        limitation_resolution / total_weight / investment_correlation /
        inhibitory_force / relationship_type / is_containment_edge for
        edges. update() diffs these against the previous snapshot to find
        genuinely-changed nodes without re-walking the whole graph.
        """
        node_fp = {
            nid: (n.strategic_value, n.investment_total_usd, n.zone_id,
                  n.acceleration_multiplier, n.entity_type, n.social_gravity,
                  n.forecast_score, n.domain)
            for nid, n in graph.nodes.items()
        }
        edge_fp = {
            key: (e.limitation_resolution, e.total_weight, e.investment_correlation,
                  e.inhibitory_force, e.relationship_type, e.is_containment_edge)
            for key, e in graph.edges.items()
        }
        return node_fp, edge_fp

    def update(self, graph: RuntimeGraph) -> Dict[str, VoidPressureVector]:
        """
        Incremental counterpart to compute() (Manifest §3, "Волновое
        распространение изменений"): a change ripples outward like a stone
        dropped in water instead of forcing a full O(N) recompute every
        epoch — see also `Oracle1Extended.update_epoch()`, the only caller.

        Cold start (compute()/update() has never run, so there's no prior
        snapshot to diff against) falls straight through to compute().

        Otherwise: diff node/edge fingerprints against the last snapshot to
        find genuinely-changed nodes, exactly recompute the 2-hop closure
        around them — _compute_node_pressure()'s own direct edges and its
        2-hop cultural-phantom scan are both guaranteed to be covered within
        that radius — then keep propagating outward one hop at a time using
        the same self-terminating conductance-decay wave
        `CascadeAnalyzer.analyze()` already uses elsewhere in this file:
        `relation_adjacency("max")` for per-edge conductance, cumulative
        conductance multiplied per hop, and `PRESSURE_DECAY_RATE` — until now
        an unused, orphaned CONFIG hook — as the cutoff below which the wave
        is considered to have died out.

        This is a deliberate approximation past the 2-hop boundary: a node's
        exact pressure can in principle depend on a source up to
        `provenance_ancestor_max_depth` (default 6) hops away via
        `combined_investment_provenance()`, which has no fixed radius.
        Walking that unbounded exact chain for every touched node would
        degenerate back into a near-full recompute, so the wave trades that
        residual exactness for staying proportional to what actually
        changed; compute() remains available as the exact cold-start/
        periodic-resync path.
        """
        if self._last_node_fp is None:
            return self.compute(graph)

        new_node_fp, new_edge_fp = self._fingerprint_snapshot(graph)

        changed: Set[str] = set()
        for nid, fp in new_node_fp.items():
            if self._last_node_fp.get(nid) != fp:
                changed.add(nid)
        for nid in self._last_node_fp:
            if nid not in new_node_fp:
                changed.add(nid)  # node removed since the last update

        all_edge_keys = set(new_edge_fp) | set(self._last_edge_fp)
        for key in all_edge_keys:
            if new_edge_fp.get(key) != self._last_edge_fp.get(key):
                src, tgt = key
                changed.add(src)
                changed.add(tgt)

        self._last_changed = changed  # TODO #22 — see __init__ comment

        # Drop stale field entries for nodes no longer in the graph.
        for nid in list(self.field):
            if nid not in graph.nodes:
                self.field.pop(nid, None)

        if not changed:
            self._last_node_fp, self._last_edge_fp = new_node_fp, new_edge_fp
            return self.field

        # Exact 2-hop closure around every changed node.
        closure = set(changed) | graph.k_hop_neighborhood(list(changed), k=2)
        visited = {nid for nid in closure if nid in graph.nodes}
        for nid in visited:
            self.field[nid] = self._compute_node_pressure(nid, graph)

        # Propagate outward past the exact boundary as a self-terminating
        # conductance-decay wave, mirroring CascadeAnalyzer.analyze().
        adjacency = graph.relation_adjacency("max")
        frontier: Dict[str, float] = {nid: 1.0 for nid in visited}

        while frontier:
            next_frontier: Dict[str, float] = {}
            for origin_id, upstream_cond in frontier.items():
                for nbr_id, edge_cond in adjacency.get(origin_id, ()):
                    if nbr_id in visited:
                        continue
                    cond = upstream_cond * edge_cond
                    if cond < self.PRESSURE_DECAY_RATE:
                        continue  # wave has decayed past the point of mattering
                    if cond > next_frontier.get(nbr_id, 0.0):
                        next_frontier[nbr_id] = cond
            if not next_frontier:
                break  # the wave has died out on its own
            for nbr_id in next_frontier:
                visited.add(nbr_id)
                if nbr_id in graph.nodes:
                    self.field[nbr_id] = self._compute_node_pressure(nbr_id, graph)
            frontier = next_frontier

        self._last_node_fp, self._last_edge_fp = new_node_fp, new_edge_fp
        return self.field

    def _compute_node_pressure(self, nid: str, graph: RuntimeGraph,
                                provenance_discount: Optional[Dict[Tuple[str, str], float]] = None
                                ) -> VoidPressureVector:
        """
        provenance_discount is accepted for backward-compatible call sites
        but no longer used: excess_investment_pressure below now calls
        graph.combined_investment_provenance() for an EXACT Set-Union
        de-duplication of shared funding across converging sources
        (Manifest §3, "Закон сохранения ресурса" — the "Rhombus problem"),
        superseding the approximate per-edge ancestor-overlap discount this
        parameter used to carry. compute_ancestor_overlap_discount() itself
        is unaffected and still lives on: graph_to_tensors() still computes
        and uses it independently for the GNN-facing tensor path, which
        needs one scalar per edge (for StructuralDependencyUpdater's
        scatter_add) rather than a token→amount dict, and so can't consume
        the exact mechanism directly — see that function's docstring for
        why the neural path keeps the approximation while the symbolic
        pressure calc here uses the exact one (Manifest §5: GNN as an
        auxiliary co-processor for the nonlinear/approximate case, the
        deterministic core for anything computable exactly).
        """
        pv = VoidPressureVector(node_id=nid)
        node = graph.get_node(nid)
        if not node: return pv

        lim_thr  = CONFIG["pressure_limitation_threshold"]
        inv_thr  = CONFIG["pressure_investment_threshold"]
        inv_logsc = CONFIG["pressure_invest_log_scale"]
        lim_scale = CONFIG["pressure_limitation_scale"]
        unr_gate  = CONFIG["pressure_unresolved_gate"]
        zn_scale  = CONFIG["pressure_zone_multiplier_scale"]

        # nid-level, not src-level — depends only on nid's own outgoing
        # edges, so compute it once rather than once per qualifying source
        # (the original per-edge loop recomputed this identically on every
        # iteration).
        outgoing_resolution = sum(
            e.limitation_resolution
            for (s2, t2), e in graph.edges.items()
            if s2 == nid and not e.is_containment_edge)

        qualifying_srcs: List[str] = []
        corr_sum = 0.0
        for (src, tgt), edge in graph.edges.items():
            if edge.is_containment_edge: continue
            is_incoming = (tgt == nid)
            if is_incoming and edge.limitation_resolution > lim_thr:
                pv.unresolved_limitation_pressure += (
                    edge.limitation_resolution * lim_scale * (1.0 - edge.total_weight))
            if is_incoming:
                inh = getattr(edge, "inhibitory_force", 0.0)
                if inh > 0:
                    attacker = graph.get_node(src)
                    weight = (getattr(attacker, "strategic_value", 5.0) / 10.0
                              if attacker else 0.5)
                    pv.inhibitory_sink_pressure += inh * weight
            if (is_incoming and edge.investment_correlation > inv_thr
                    and outgoing_resolution < unr_gate):
                qualifying_srcs.append(src)
                corr_sum += edge.investment_correlation

        if qualifying_srcs:
            # Set-Union every qualifying source's investment provenance
            # BEFORE summing: a funding round visible through two
            # different converging branches counts once here, not once
            # per branch. Weighted by the average investment_correlation
            # across qualifying edges — correlation is a per-edge
            # transmission-strength signal, orthogonal to fund identity,
            # so it still matters independently of the dedup.
            deduped = graph.combined_investment_provenance(qualifying_srcs)
            avg_corr = corr_sum / len(qualifying_srcs)
            pv.excess_investment_pressure = (
                avg_corr * math.log1p(sum(deduped.values())) / inv_logsc)

        k2 = graph.k_hop_neighborhood([nid], k=2)
        for nbr_id in k2:
            nbr = graph.get_node(nbr_id)
            if not nbr: continue
            if "cultural_phantom" in getattr(nbr, "entity_type", ""):
                sg = getattr(nbr, "social_gravity", 0.0) or nbr.forecast_score
                pv.cultural_phantom_pressure += sg / 10.0

        if node.zone_id and node.acceleration_multiplier > 1.0:
            pv.temporal_zone_pressure = (node.acceleration_multiplier - 1.0) * zn_scale

        pw = CONFIG["pressure_void_weights"]
        pv.void_pressure = (
            pv.unresolved_limitation_pressure * pw["unresolved_limitation"] +
            pv.cultural_phantom_pressure      * pw["cultural_phantom"]      +
            pv.excess_investment_pressure     * pw["excess_investment"]     +
            pv.inhibitory_sink_pressure       * pw["inhibitory_sink"]       +
            pv.temporal_zone_pressure         * pw["temporal_zone"])
        pv.pressure_history.append(pv.void_pressure)
        return pv

    def pressure_gradient(self, node_id: str, neighbour_id: str) -> float:
        p_src = self.field.get(node_id,    VoidPressureVector(node_id)).void_pressure
        p_tgt = self.field.get(neighbour_id, VoidPressureVector(neighbour_id)).void_pressure
        return p_src - p_tgt


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — DORMANCY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EdgeDormancyState:
    edge_key: Tuple[str, str]
    dormancy_score: float = 1.0
    epochs_dormant: int = 0
    last_contribution: float = 0.0
    awakening_event: Optional[str] = None
    neighbourhood_heat: float = 0.0


class DormancyTracker:
    AWAKENING_THRESHOLD = CONFIG["dormancy_awakening_threshold"]
    ACTIVE_THRESHOLD    = CONFIG["dormancy_active_threshold"]
    HEAT_DECAY_FACTOR   = CONFIG["dormancy_heat_decay_factor"]
    DORMANCY_GROWTH     = CONFIG["dormancy_growth_per_epoch"]
    WAVE_EPSILON        = CONFIG["dormancy_wave_epsilon"]

    def __init__(self):
        self.states: Dict[Tuple[str, str], EdgeDormancyState] = {}

    def initialise_edge(self, edge: KnowledgeEdge):
        key = (edge.source_id, edge.target_id)
        if key not in self.states:
            self.states[key] = EdgeDormancyState(edge_key=key)

    def update_epoch(self, graph: RuntimeGraph) -> List[Tuple[str, str]]:
        """
        Exact full pass over every registered edge — kept as the cold-start
        / periodic-reconciliation path (same role compute() plays next to
        update() on ConvergencePressureField) because it's the only path
        that also evaluates _semantic_twin_heat, which is deliberately not
        hop-bounded (see update_wave()'s docstring) and so cannot be
        reached by any wave. Call this on whatever periodic cadence you can
        still afford at your graph's scale; use update_wave() as the normal
        per-ingestion hot path (TODO #22).
        """
        newly_awake = []
        for key in self.states:
            src, tgt = key
            heat = self._compute_neighbourhood_heat(src, tgt, graph)
            if self._apply_heat(key, heat):
                newly_awake.append(key)
        return newly_awake

    def update_wave(self, seed_ids: Iterable[str], graph: "RuntimeGraph") -> List[Tuple[str, str]]:
        """
        Incremental counterpart to update_epoch() (TODO #22, mirrors
        ConvergencePressureField's compute()/update() split): recompute
        dormancy only for edges a self-terminating wave from seed_ids
        (freshly added/changed nodes) actually reaches, instead of scanning
        every edge in self.states on every call — the update_epoch() cost
        that is "нереально" (unworkable) once self.states holds anywhere
        near the graph sizes the manifest targets.

        Scope note — this recomputes the TOPOLOGICAL heat component only
        (_topological_heat: local, hop-bounded, exactly what a wave can
        cover). _semantic_twin_heat is deliberately NOT hop-bounded by
        design: a dormant edge should be able to wake because an unrelated,
        arbitrarily-distant node just became its semantic twin, and no
        bounded graph-topology wave can discover that trigger — it needs
        its own change detector over the embedding space (an ANN index
        flagging "a new/changed embedding just became someone's nearest
        neighbour"), which this codebase doesn't have yet (see the
        CanonicalKey SemanticHasher "brute-force ANN fallback" comment for
        the same missing infrastructure elsewhere). Keep running
        update_epoch() on whatever periodic cadence you can still afford as
        the exact pass that also covers semantic-twin waking.

        The wave visits a node, recomputes every edge incident to it (via
        the adjacency index, TODO #20), and only relays past that node if
        at least one of those edges' dormancy_score actually moved by more
        than dormancy_wave_epsilon — "a node relays the change to its
        neighbours only if the change actually shifts its own weight" is
        the exact rule the manifest states for pressure/tension, applied
        here to dormancy.
        """
        newly_awake: List[Tuple[str, str]] = []

        def apply_fn(nid: str) -> bool:
            relay = False
            for key in graph._adjacency.get(nid, ()):
                if key not in self.states:
                    continue
                src, tgt = key
                heat = self._topological_heat(src, tgt, graph)
                prev_score = self.states[key].dormancy_score
                if self._apply_heat(key, heat):
                    newly_awake.append(key)
                if abs(self.states[key].dormancy_score - prev_score) > self.WAVE_EPSILON:
                    relay = True
            return relay

        graph.propagate_wave(seed_ids, apply_fn)
        return newly_awake

    def _apply_heat(self, key: Tuple[str, str], heat: float) -> bool:
        """
        Shared update rule for one edge's dormancy state given its heat —
        exactly update_epoch()'s original per-edge body, extracted once
        (TODO #22) so update_epoch() and update_wave() apply the identical
        rule instead of two copies that could silently drift apart, which
        is exactly the kind of duplicate-path confusion the manifest
        already flags elsewhere (CanonicalKey vs MD5, TODO #19).

        Returns True iff this call counts as an awakening event (matches
        update_epoch()'s original return semantics: True on every call
        where the edge is above AWAKENING_THRESHOLD and has decayed below
        ACTIVE_THRESHOLD, not only on the transition into that state).
        """
        state = self.states[key]
        state.neighbourhood_heat = heat
        if heat > self.AWAKENING_THRESHOLD:
            decay = heat * self.HEAT_DECAY_FACTOR
            state.dormancy_score = max(0.0, state.dormancy_score * (1 - decay))
            if state.dormancy_score < self.ACTIVE_THRESHOLD:
                state.awakening_event = f"neighbourhood_heat={heat:.2f}"
                return True
            return False
        state.dormancy_score = min(1.0, state.dormancy_score + self.DORMANCY_GROWTH)
        state.epochs_dormant += 1
        return False


    def _compute_neighbourhood_heat(self, src: str, tgt: str,
                                     graph: RuntimeGraph) -> float:
        """
        Heat that can wake a dormant edge, combining two independent sources:

        1. Topological — src/tgt's 2-hop neighbourhood (_topological_heat).
        2. Semantic — nodes ANYWHERE in the graph whose embedding is close
           to src's or tgt's, regardless of how many hops away they sit
           (_semantic_twin_heat). A 2026 breakthrough and a similar 1920
           paper in an unrelated domain will never be topological
           neighbours — connections aren't bound by when they were created,
           so a dormant edge should be able to wake because its *twin* got
           hot elsewhere in the graph, not only because its own local patch
           did.

        Combined by max, not sum: either source heating up is reason enough
        to look again, and summing would let two weak, unrelated signals
        falsely add up to "hot".
        """
        return max(
            self._topological_heat(src, tgt, graph),
            self._semantic_twin_heat(src, tgt, graph),
        )

    def _topological_heat(self, src: str, tgt: str, graph: RuntimeGraph) -> float:
        k2 = graph.k_hop_neighborhood([src, tgt], k=2)
        if not k2: return 0.0
        scores = []
        current_year = _ts_to_year(time.time())
        for nid in k2:
            node = graph.get_node(nid)
            if node and not node.is_temporal_zone:
                try:
                    year = _ts_to_year(node.timestamp)
                    recency = 1.0 / (1.0 + max(0, current_year - year) / 5.0)
                except Exception:
                    recency = 0.5
                scores.append(node.forecast_score * recency)
        return float(np.mean(scores)) if scores else 0.0

    SEMANTIC_TWIN_THRESHOLD      = CONFIG["dormancy_semantic_twin_threshold"]
    SEMANTIC_TWIN_MAX_CANDIDATES = CONFIG["dormancy_semantic_twin_max_candidates"]  # cap the brute-force similarity scan

    def _semantic_twin_heat(self, src: str, tgt: str, graph: RuntimeGraph) -> float:
        anchors = [n for n in (graph.get_node(src), graph.get_node(tgt))
                   if n is not None and n.embedding is not None]
        if not anchors:
            return 0.0
        candidates = [
            (nid, n) for nid, n in graph.nodes.items()
            if n.embedding is not None and not n.is_temporal_zone
            and nid not in (src, tgt)
        ]
        if not candidates:
            return 0.0
        if len(candidates) > self.SEMANTIC_TWIN_MAX_CANDIDATES:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(candidates), self.SEMANTIC_TWIN_MAX_CANDIDATES, replace=False)
            candidates = [candidates[i] for i in idx]

        cand_mat   = np.stack([n.embedding for _, n in candidates])
        anchor_mat = np.stack([n.embedding for n in anchors])
        sims = cos_sim(anchor_mat, cand_mat)          # (n_anchors, n_candidates)
        best_sim = sims.max(axis=0)                    # closest anchor per candidate

        current_year = _ts_to_year(time.time())
        hot_scores = []
        for (nid, node), sim in zip(candidates, best_sim):
            if sim < self.SEMANTIC_TWIN_THRESHOLD:
                continue
            try:
                year = _ts_to_year(node.timestamp)
                recency = 1.0 / (1.0 + max(0, current_year - year) / 5.0)
            except Exception:
                recency = 0.5
            hot_scores.append(node.forecast_score * recency * float(sim))
        return float(np.mean(hot_scores)) if hot_scores else 0.0

    def get_dormancy(self, src: str, tgt: str) -> float:
        return self.states.get((src, tgt), EdgeDormancyState((src, tgt))).dormancy_score

    def get_awakening_score(self, src: str, tgt: str) -> float:
        return 1.0 - self.get_dormancy(src, tgt)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — STRUCTURAL PRESSURE COMPUTER
#
#  Previously, void_pressure / SDI / cascade_influence / threshold checks were
#  duplicated in ContextualPriorityHead, PhantomNodeGenerator,
#  GraphEntropyMonitor, and RecursiveForecastingLoop.
#  All four now delegate to this single authority.
# ══════════════════════════════════════════════════════════════════════════════

class StructuralPressureComputer:
    """Single source of truth for topological pressure statistics.

    Eliminates duplication across ContextualPriorityHead, PhantomNodeGenerator,
    GraphEntropyMonitor, and RecursiveForecastingLoop.

    All threshold constants are read from CONFIG so that a single edit
    propagates to every consumer automatically.
    """

    SINGULARITY_THRESHOLD = CONFIG["pressure_singularity_threshold"]
    COLLAPSE_THRESHOLD    = CONFIG["collapse_threshold"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def node_pressure(self, node_id: str,
                      pressure_field: "ConvergencePressureField") -> float:
        """Return the void_pressure scalar for a single node."""
        pv = pressure_field.field.get(node_id, VoidPressureVector(node_id))
        return pv.void_pressure

    def sdi_cascade_stats(self, node_ids: "List[str]",
                          graph: "RuntimeGraph") -> "Dict[str, float]":
        """Return mean SDI, mean cascade_influence, and mean zone_multiplier
        for a collection of node IDs (2-hop neighbourhood, domain set, etc.)."""
        sdis, cascades, zmults = [], [], []
        for nid in node_ids:
            node = graph.get_node(nid)
            if not node or node.is_temporal_zone:
                continue
            sdis.append(getattr(node, "structural_dependency_index", 0.0))
            cascades.append(getattr(node, "cascade_influence", 0.0))
            zmults.append(node.acceleration_multiplier)
        return {
            "mean_sdi":    float(np.mean(sdis))    if sdis    else 0.0,
            "mean_cascade": float(np.mean(cascades)) if cascades else 0.0,
            "mean_zmult":  float(np.mean(zmults))  if zmults  else 1.0,
        }

    def is_singularity(self, pressure: float) -> bool:
        return pressure >= self.SINGULARITY_THRESHOLD

    def is_collapse(self, pressure: float) -> bool:
        return pressure >= self.COLLAPSE_THRESHOLD

    def pressure_gradient(self, src_id: str, tgt_id: str,
                          pressure_field: "ConvergencePressureField") -> float:
        return pressure_field.pressure_gradient(src_id, tgt_id)

    def edge_topology_counts(self, k2: "Set[str]",
                              graph: "RuntimeGraph") -> "Dict[str, int]":
        """Count total / limitation / inhibitory edges touching a neighbourhood."""
        n_total = n_lim = n_inh = 0
        for (s2, t2), edge in graph.edges.items():
            if s2 in k2 or t2 in k2:
                n_total += 1
                if edge.limitation_resolution > CONFIG["priority_lim_threshold"]:
                    n_lim += 1
                if getattr(edge, "inhibitory_force", 0.0) > CONFIG["priority_inh_threshold"]:
                    n_inh += 1
        return {"n_edges": n_total, "n_lim": n_lim, "n_inh": n_inh}


# ── Singleton shared by all subsystems ────────────────────────────────────────
PRESSURE_COMPUTER = StructuralPressureComputer()


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — CONFIDENCE DECAY ENGINE
#
#  Previously confidence degradation lived in three separate places:
#    • RecursiveForecastingLoop   (DECAY_FACTOR geometric decay)
#    • PhantomNodeGenerator       (structural_gap_score multiplication)
#    • PathAgnosticInference      (entropy alerts)
#    • RecursiveForecastingLoop   (plateau penalty)
#  Parts of these are mathematically equivalent.  This class owns the logic.
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceDecayEngine:
    """Canonical source for all confidence-degradation arithmetic.

    Consumers receive a single ``decay_step`` call that applies:
      1. Geometric depth decay (DECAY_FACTOR per recursion level)
      2. Efficiency-plateau Christensen penalty
      3. Optional entropy-alert suppression

    This makes it impossible for modules to diverge silently when thresholds
    change.
    """

    DECAY_FACTOR       = CONFIG["forecast_decay_factor"]
    PLATEAU_SCALE      = CONFIG["plateau_penalty_scale"]
    PLATEAU_FLOOR      = CONFIG["plateau_penalty_floor"]

    def geometric_decay(self, confidence: float, steps: int = 1) -> float:
        """Standard exponential decay by depth."""
        return confidence * (self.DECAY_FACTOR ** steps)

    def plateau_penalty(self, node: "KnowledgeNode",
                        graph: "RuntimeGraph") -> float:
        """Return a multiplier in (PLATEAU_FLOOR, 1.0].

        1.0  = no penalty (node exceeds or equals domain plateau)
        < 1.0 = Christensen drag — incumbent efficiency plateau is higher.
        """
        max_plateau = max(
            (n.efficiency_plateau for n in graph.nodes.values()
             if not n.is_temporal_zone and n.domain == node.domain),
            default=0.0,
        )
        if max_plateau <= 0.0:
            return 1.0
        node_potential = float(node.convergence_potential)
        if node_potential >= max_plateau:
            return 1.0
        gap_ratio = (max_plateau - node_potential) / max(max_plateau, 1e-6)
        penalty   = 1.0 - self.PLATEAU_SCALE * float(np.clip(gap_ratio, 0.0, 1.0))
        return float(max(self.PLATEAU_FLOOR, penalty))

    def entropy_alert_factor(self, domain: str,
                             entropy_monitor: "GraphEntropyMonitor") -> float:
        """Suppress confidence when the domain's entropy is in 'pre_singularity'.

        Returns a multiplier [0.5, 1.0].  Normal = 1.0; pre-singularity = 0.5.
        """
        snap = entropy_monitor.current.get(domain)
        if snap and snap.alert == "pre_singularity":
            return 0.5
        return 1.0

    def decay_step(self, confidence: float,
                   node: "KnowledgeNode",
                   graph: "RuntimeGraph",
                   entropy_monitor: "Optional[GraphEntropyMonitor]" = None) -> float:
        """Apply one full decay step: geometric × plateau × entropy."""
        c = self.geometric_decay(confidence)
        c *= self.plateau_penalty(node, graph)
        if entropy_monitor is not None:
            c *= self.entropy_alert_factor(node.domain, entropy_monitor)
        return c


# ── Singleton ─────────────────────────────────────────────────────────────────
CONFIDENCE_ENGINE = ConfidenceDecayEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — GRAPH TRAVERSAL ENGINE
#
#  DFS path counting existed in RetrospectiveConvergenceLoss._count_paths_to_targets
#  and BFS reachability in PhantomNodeGenerator._is_near_phantom_zone.
#  Both are unified here so future changes (e.g., cycle handling or pruning)
#  propagate automatically.
# ══════════════════════════════════════════════════════════════════════════════

class GraphTraversalEngine:
    """Unified DFS / BFS traversal primitives for RuntimeGraph.

    Replaces:
      • RetrospectiveConvergenceLoss._count_paths_to_targets  (DFS)
      • PhantomNodeGenerator._is_near_phantom_zone            (BFS)
    """

    @staticmethod
    def dfs_count_paths(start: str, targets: "Set[str]",
                        graph: "RuntimeGraph",
                        max_hops: int, max_paths: int) -> int:
        """Count distinct simple paths from *start* to any node in *targets*,
        limited to *max_hops* and stopping after *max_paths* are found."""
        count = 0
        stack = [(start, {start}, 0)]
        while stack and count < max_paths:
            current, visited, hops = stack.pop()
            if hops > max_hops:
                continue
            for (src, tgt) in list(graph.edges.keys()):
                if src != current:
                    continue
                nbr = tgt
                if nbr in visited:
                    continue
                if nbr in targets:
                    count += 1
                else:
                    stack.append((nbr, visited | {nbr}, hops + 1))
        return count

    @staticmethod
    def bfs_reachable(start: str, target_ids: "Set[str]",
                      graph: "RuntimeGraph", max_hops: int) -> bool:
        """Return True if any node in *target_ids* is reachable from *start*
        within *max_hops* steps (undirected traversal)."""
        visited: set = {start}
        frontier = [start]
        for _hop in range(max_hops):
            next_frontier = []
            for nid in frontier:
                for (src, tgt) in list(graph.edges.keys()):
                    neighbour = (tgt if src == nid
                                 else (src if tgt == nid else None))
                    if neighbour is None or neighbour in visited:
                        continue
                    if neighbour in target_ids:
                        return True
                    visited.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return False

    @staticmethod
    def _is_dependency_resolved(node: "KnowledgeNode", graph: "RuntimeGraph") -> bool:
        """Local check: has *node* itself crossed from idea to solution?

        Not recursive — this says nothing about node's own requires_node_ids,
        only about node itself. Dispatches on node kind because "resolved"
        means a different thing for each:

          - entity_type == "physical_limit": a constraint, not a claim about
            a built artifact — maturity_score on one is meaningless. It
            resolves when some OTHER node earns an incoming
            limitation_resolution edge above CONFIG["priority_lim_threshold"]
            (Manifest §1, "First-Class Constraints"). Read via
            graph._adjacency (O(degree), incrementally maintained by
            add_edge — same access pattern DormancyTracker.propagate_wave
            uses) rather than RuntimeGraph.relation_adjacency(), which
            rescans every edge in the graph and can't be asked for one
            node's incoming edges only.
          - origin_kind == "fundamental": something nature already contains;
            once independently confirmed to exist (epistemic_level
            PROVEN_DISCOVERY or PEER_REVIEWED) there is nothing left to
            engineer, regardless of maturity_score — Principle 1 is explicit
            that a fundamental node's timing "is not a convergence event".
          - everything else (combinatorial, the default): resolved once its
            own maturity_score crosses CONFIG["dependency_maturity_gate"].

        Fails closed on missing/unscored data (returns False) — a
        diagnostic tool should default to "still an idea", not "assume
        solved", when it doesn't know.
        """
        if getattr(node, "entity_type", "") == "physical_limit":
            lim_threshold = CONFIG["priority_lim_threshold"]
            for key in graph._adjacency.get(node.id, ()):
                if key[1] != node.id:
                    continue  # this node was the source, not the target — not incoming
                edge = graph.edges.get(key)
                if edge is not None and edge.limitation_resolution > lim_threshold:
                    return True
            return False

        if getattr(node, "origin_kind", "combinatorial") == "fundamental":
            return node.epistemic_level in (
                EpistemicLevel.PROVEN_DISCOVERY.name,
                EpistemicLevel.PEER_REVIEWED.name,
            )

        return node.maturity_score >= CONFIG["dependency_maturity_gate"]

    @staticmethod
    def find_unresolved_dependencies(
        start: str,
        graph: "RuntimeGraph",
        max_hops: int = None,
        max_nodes: int = None,
    ) -> List[str]:
        """Walk start's requires_node_ids closure and return the deepest
        node id(s) that actually explain why start is still idea-status —
        the idea/solution boundary, not a single aggregate score.

        This is a SECOND, independent traversal primitive, not a wrapper
        around bfs_reachable/dfs_count_paths above: those walk graph.edges
        (real registered KnowledgeEdge objects). requires_node_ids is a
        plain string-list attribute with no corresponding edge ever created
        for it anywhere in the ingestion or merge path — confirmed nothing
        in this file emits a "requires"-typed KnowledgeEdge from it, so
        graph.edges has nothing to walk here.

        Direction matters and is easy to get backwards (an earlier draft of
        this method had it backwards): the search STOPS at a RESOLVED node
        — its own maturity_score was, by definition, earned with its own
        inputs already functioning, so digging further beneath it is
        redundant, the same "don't relay past a boundary" idiom
        bfs_reachable and DormancyTracker's wave propagation both use. It
        RECURSES through an UNRESOLVED node's own requires_node_ids, because
        an unresolved intermediate ("quantum chips", to use the conceptual
        example this was built for) is rarely the actionable answer by
        itself — the useful diagnosis is what specifically is missing
        beneath it ("room-temperature qubit coherence"). A node is reported
        as a result leaf only when it is itself unresolved AND either
        declares no requires_node_ids, or every one of its declared
        children turns out to already be resolved — i.e. nothing deeper
        explains it, so the gap is AT that node (institutional/temporal,
        Principle 4/5 territory — not a missing precursor beneath it).

        Returns [] if start has no requires_node_ids, or if every entry
        resolves — either way, no *structural* blocker was found, and a low
        readiness_score on start needs a different, non-structural check —
        not evidence that this function failed to find something.

        Correctness depends entirely on IngestionPipeline._process_chunk's
        slug -> canonical-node.id remap (the requires_node_ids fix applied
        alongside this method): an LLM emits these ids as per-document
        slugs, never as the final CanonicalKey hex. Unmapped, virtually
        every entry would silently fail graph.get_node() and this function
        would return [] for nearly any node — a false "fully resolved"
        reading, not a crash, which is the dangerous direction for a
        diagnostic tool to fail in.
        """
        max_hops  = CONFIG["dependency_closure_max_hops"]  if max_hops  is None else max_hops
        max_nodes = CONFIG["dependency_closure_max_nodes"] if max_nodes is None else max_nodes

        start_node = graph.get_node(start)
        if start_node is None:
            return []

        memo: Dict[str, List[str]] = {}
        state = {"visited_count": 0, "budget_exhausted": False}

        def resolve(req_id: str, depth: int, on_path: "Set[str]") -> List[str]:
            if req_id in memo:
                return memo[req_id]
            if req_id in on_path:
                return [req_id]  # cycle in requires_node_ids — surface it, don't loop forever
            if depth > max_hops or state["budget_exhausted"]:
                return [req_id]  # boundary hit — report as open rather than silently dropped

            state["visited_count"] += 1
            if state["visited_count"] > max_nodes:
                if not state["budget_exhausted"]:
                    logger.warning(
                        f"[GraphTraversalEngine] find_unresolved_dependencies("
                        f"'{start}'): max_nodes budget ({max_nodes}) exhausted "
                        "— returned closure may be incomplete."
                    )
                state["budget_exhausted"] = True
                return [req_id]

            node = graph.get_node(req_id)
            if node is None:
                # Dangling even after the slug_map fix (node later removed
                # from the graph, or data ingested before the fix was
                # applied). Report rather than skip silently: "unknown" and
                # "not yet built" both mean the same thing to whatever
                # depends on it.
                result = [req_id]
            elif GraphTraversalEngine._is_dependency_resolved(node, graph):
                result = []
            else:
                children = getattr(node, "requires_node_ids", [])
                deeper: List[str] = []
                for child_id in children:
                    deeper.extend(resolve(child_id, depth + 1, on_path | {req_id}))
                result = deeper if deeper else [req_id]

            memo[req_id] = result
            return result

        unresolved: List[str] = []
        seen: "Set[str]" = set()
        for req_id in getattr(start_node, "requires_node_ids", []):
            for leaf_id in resolve(req_id, 1, {start}):
                if leaf_id not in seen:
                    seen.add(leaf_id)
                    unresolved.append(leaf_id)
        return unresolved


# ── Singleton ─────────────────────────────────────────────────────────────────
GRAPH_TRAVERSAL = GraphTraversalEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — CONTEXTUAL PRIORITY HEAD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocalTopologyStats:
    mean_sdi_2hop:             float = 0.0
    mean_cascade_2hop:         float = 0.0
    void_pressure_src:         float = 0.0
    void_pressure_tgt:         float = 0.0
    pressure_gradient:         float = 0.0
    n_edges_2hop:              int   = 0
    n_limitation_edges_2hop:   int   = 0
    n_inhibitory_edges_2hop:   int   = 0
    mean_zone_multiplier_2hop: float = 1.0
    awakening_score:           float = 0.0
    mean_constraint_proximity_2hop: float = 0.5


class ContextualPriorityHead(nn.Module):
    SINGULARITY_THRESHOLD = CONFIG["priority_singularity_threshold"]
    EXPO_SCALE   = CONFIG["priority_expo_scale"]
    LINEAR_SCALE = CONFIG["priority_linear_scale"]

    def __init__(self, latent_dim: int, n_topology_features: int = 12):
        super().__init__()
        self.latent_dim = latent_dim
        if not TORCH_AVAILABLE: return
        self.pair_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.GELU(),
            nn.LayerNorm(256), nn.Linear(256, 64))
        self.topo_encoder = nn.Sequential(
            nn.Linear(n_topology_features, 64), nn.GELU(), nn.Linear(64, 64))
        self.priority_head = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.GELU(), nn.Linear(64, 1))
        if TORCH_AVAILABLE:
            self.log_expo_scale   = nn.Parameter(torch.tensor(math.log(self.EXPO_SCALE)))
            self.singularity_bias = nn.Parameter(torch.tensor(-self.SINGULARITY_THRESHOLD))

    def topology_to_vector(self, stats: LocalTopologyStats) -> np.ndarray:
        return np.array([
            stats.mean_sdi_2hop,
            stats.mean_cascade_2hop,
            stats.void_pressure_src / (self.SINGULARITY_THRESHOLD + 1e-8),
            stats.void_pressure_tgt / (self.SINGULARITY_THRESHOLD + 1e-8),
            max(-1, min(1, stats.pressure_gradient / self.SINGULARITY_THRESHOLD)),
            min(stats.n_edges_2hop, 50) / 50.0,
            min(stats.n_limitation_edges_2hop, 20) / 20.0,
            min(stats.n_inhibitory_edges_2hop, 10) / 10.0,
            stats.mean_zone_multiplier_2hop / 5.0,
            stats.awakening_score,
            stats.mean_constraint_proximity_2hop,
            1.0 if stats.void_pressure_src > self.SINGULARITY_THRESHOLD else 0.0,
        ], dtype=np.float32)

    def forward(self, h_src, h_tgt, topology_vec, time_delta):
        if not TORCH_AVAILABLE: return topology_vec
        pair_feat = self.pair_encoder(torch.cat([h_src, h_tgt], dim=-1))
        topo_feat = self.topo_encoder(topology_vec)
        combined  = torch.cat([pair_feat, topo_feat], dim=-1)
        raw_priority = self.priority_head(combined)
        expo_scale = torch.exp(self.log_expo_scale)
        above_threshold = torch.relu(raw_priority)
        below_threshold = -torch.relu(-raw_priority)
        priority = (torch.exp(above_threshold * expo_scale) +
                    below_threshold * self.LINEAR_SCALE)
        priority = torch.clamp(priority, CONFIG["priority_clamp_min"], CONFIG["priority_clamp_max"])
        return priority

    def compute_topology_stats(self, src_id, tgt_id, graph, pressure_field,
                                dormancy_tracker) -> LocalTopologyStats:
        # ── Delegate topology arithmetic to PRESSURE_COMPUTER ─────────────────
        k2    = graph.k_hop_neighborhood([src_id, tgt_id], k=2)
        sdc   = PRESSURE_COMPUTER.sdi_cascade_stats(list(k2), graph)
        ecnt  = PRESSURE_COMPUTER.edge_topology_counts(k2, graph)

        stats = LocalTopologyStats()
        stats.mean_sdi_2hop             = sdc["mean_sdi"]
        stats.mean_cascade_2hop         = sdc["mean_cascade"]
        stats.mean_zone_multiplier_2hop = sdc["mean_zmult"]
        stats.n_edges_2hop              = ecnt["n_edges"]
        stats.n_limitation_edges_2hop   = ecnt["n_lim"]
        stats.n_inhibitory_edges_2hop   = ecnt["n_inh"]
        # constraint_proximity is not yet stored on nodes — keep 0.5 default
        stats.mean_constraint_proximity_2hop = 0.5

        stats.void_pressure_src = PRESSURE_COMPUTER.node_pressure(src_id, pressure_field)
        stats.void_pressure_tgt = PRESSURE_COMPUTER.node_pressure(tgt_id, pressure_field)
        stats.pressure_gradient = PRESSURE_COMPUTER.pressure_gradient(src_id, tgt_id, pressure_field)
        stats.awakening_score   = dormancy_tracker.get_awakening_score(src_id, tgt_id)
        return stats

    @staticmethod
    def _topo_stats_to_array(stats: "LocalTopologyStats") -> np.ndarray:
        """Serialise a LocalTopologyStats into a fixed 12-dim numpy vector.
        Used by OracleDataset.from_graph() to build topology_vec without
        needing a live ContextualPriorityHead instance (Fix 3)."""
        SINGULARITY_THRESHOLD = CONFIG["priority_singularity_threshold"]
        EXPO_SCALE = CONFIG["priority_expo_scale"]
        return np.array([
            stats.mean_sdi_2hop,
            stats.mean_cascade_2hop,
            stats.void_pressure_src / (SINGULARITY_THRESHOLD + 1e-8),
            stats.void_pressure_tgt / (SINGULARITY_THRESHOLD + 1e-8),
            max(-1.0, min(1.0, stats.pressure_gradient / (SINGULARITY_THRESHOLD + 1e-8))),
            min(stats.n_edges_2hop, 50) / 50.0,
            min(stats.n_limitation_edges_2hop, 20) / 20.0,
            min(stats.n_inhibitory_edges_2hop, 10) / 10.0,
            stats.mean_zone_multiplier_2hop / 5.0,
            stats.awakening_score,
            stats.mean_constraint_proximity_2hop,
            1.0 if stats.void_pressure_src > SINGULARITY_THRESHOLD else 0.0,
        ], dtype=np.float32)

__all__ = ['VoidPressureVector', 'ConvergencePressureField', 'EdgeDormancyState', 'DormancyTracker', 'StructuralPressureComputer', 'PRESSURE_COMPUTER', 'ConfidenceDecayEngine', 'CONFIDENCE_ENGINE', 'GraphTraversalEngine', 'GRAPH_TRAVERSAL', 'LocalTopologyStats', 'ContextualPriorityHead']
