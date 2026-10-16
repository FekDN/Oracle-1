# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Graph node, edge, temporal, and feature data structures."""
from __future__ import annotations

from ._shared import *
from .config import (
    CONFIG,
    INHIBITORY_WEIGHT,
    N_EXTENDED_COMPONENTS,
    PHYSICAL_AXIS_ORDER,
    PHYSICAL_SUBSTRATE_DIM,
    V6_EDGE_WEIGHTS,
)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeNode:
    id: str
    text: str
    full_text: str = ""
    node_type: str = ""
    domain: str = ""
    entity_type: str = ""
    # "fundamental" — nature already contains it; humans only discover it as
    # measurement/theory improves (a constant, a material property, a limit).
    # Not a matter of investment or engineering effort, and its timing isn't
    # predictable from pressure the way combinatorial timing is — but once
    # discovered it can move an entire cluster of downstream technologies at
    # once, not just its own immediate neighbours.
    # "combinatorial" (default) — an engineered recombination of already-known
    # fundamentals; almost all products and techniques. Timing tracks
    # investment, convergence pressure, and adjacent-domain maturity, which is
    # exactly what Principle 1 (Convergence Determinism) and the
    # Intervention Optimizer's investment injection reason about.
    origin_kind: str = "combinatorial"
    provenance: str = ""
    timestamp: float = 0.0
    publication_date: Optional[str] = None

    # ── Temporal shape ───────────────────────────────────────────────────
    # 'point'    — a single instantaneous observation (one filing, one quoted
    #              remark, one publication); `timestamp` is exact.
    # 'interval' — the idea is restated, argued over, or elaborated across a
    #              meaningful span; `timestamp` then holds the most recent
    #              reinforcement, while timestamp_start/timestamp_end give the
    #              full band. Populated by SignificanceProcessor from the
    #              node's InfluenceTimeline — see _sync_epistemic_to_node.
    temporal_kind:     str   = "point"
    timestamp_start:   float = 0.0
    timestamp_end:     float = 0.0
    # Weighted reinforcement count per `density_bin_years`-wide bucket across
    # [timestamp_start, timestamp_end]. Empty for point-nodes. This curve, not
    # merely the band's existence, is what Temporal Nonlinearity and
    # ConvergenceAccelerometer read to tell a thickening idea from one that has
    # simply persisted at constant background noise for decades.
    temporal_density:    List[float] = field(default_factory=list)
    density_bin_years:   float       = 1.0

    scientific_score:  float = 0.0
    investment_score:  float = 0.0
    social_score:      float = 0.0
    maturity_score:    float = 0.0
    group_size_score:  float = 0.0
    readiness_score:   float = 0.0

    dual_use_risk:        float = 0.0
    strategic_value:      float = 0.0
    legal_risk_score:     float = 0.0
    export_control_risk:  float = 0.0

    structural_dependency_index: float = 0.0
    cascade_influence:           float = 0.0
    upstream_pressure:           float = 0.0

    is_temporal_zone:   bool      = False
    zone_multiplier:    float     = 1.0
    contained_node_ids: List[str] = field(default_factory=list)

    # zone_id/acceleration_multiplier stay as the *derived* strongest-zone
    # projection (kept for every existing reader that expects one scalar —
    # NodeFeatureBuilder, PathStep/GraphWalker, ConvergencePressureField).
    # zone_ids is the actual membership: every TemporalZone whose
    # (domain_focus, [start_timestamp, end_timestamp]) region covers this
    # node's own (domain, timestamp) coordinates — see
    # RuntimeGraph.zones_covering_node(). A node's coordinates can sit
    # inside several overlapping zones at once (Manifest §1: "Zones may
    # overlap, compounding the effect"); zone_id/acceleration_multiplier
    # alone cannot represent that, so zone_ids is the source of truth and
    # the scalar pair is recomputed from it via
    # RuntimeGraph.recompute_zone_membership().
    zone_id:               str   = ""
    acceleration_multiplier: float = 1.0
    zone_ids:               List[str] = field(default_factory=list)

    sentiment_review_score:    float = 0.0
    sentiment_fiction_score:   float = 0.0
    sentiment_forum_score:     float = 0.0
    social_perception_score:   float = 0.0

    investment_rounds:         int   = 0
    investment_total_usd:      float = 0.0
    investment_last_round_usd: float = 0.0
    investment_lead_investors: List[str] = field(default_factory=list)

    forum_post_count:    int        = 0
    forum_sentiment_raw: List[float] = field(default_factory=list)

    solves_limitations: List[str] = field(default_factory=list)
    requires_node_ids:  List[str] = field(default_factory=list)
    enables_node_ids:   List[str] = field(default_factory=list)

    discovery_depth: int       = 0
    discovery_path:  List[str] = field(default_factory=list)
    discovery_query: str       = ""

    group_id:    int = -1
    group_count: int = 1

    convergence_potential: float = 0.0
    forecast_score:        float = 0.0

    embedding: Optional[np.ndarray] = None

    # Extensions
    phantom_weight: float = 0.0
    social_gravity: float = 0.0

    # Incumbent-reaction / efficiency threshold
    # efficiency_plateau: 0–10 scale representing peak efficiency of the
    # *current* dominant technology in this domain.  A hallucinated node whose
    # convergence_potential does not exceed this value will have its
    # convergence_probability penalised (see RecursiveForecastingLoop).
    efficiency_plateau: float = 0.0

    # Set to True after resolve_rupture_physics() has relaxed this node's
    # physical constraints – allows downstream code to skip hard feasibility gates.
    rupture_physics_relaxed: bool = False

    # ── Epistemic / Significance fields (populated by SignificanceProcessor) ──
    # Human-readable epistemic level name (e.g. "PEER_REVIEWED", "PROVEN_DISCOVERY")
    epistemic_level: str = ""
    # Accumulated epistemic mass from InfluenceTimeline (0–10 scale)
    epistemic_mass: float = 0.0
    # Velocity of epistemic reinforcements over the last 5 years (breakthrough signal)
    influence_velocity: float = 0.0
    # Cross-domain source diversity index (0–1)
    source_diversity: float = 0.0

    # ── Canonical key metadata (populated by CanonicalAssembler) ────────────
    # Observation count from all sources that produced this canonical concept
    observation_count: int = 0
    # Source type that produced this node (e.g. "arxiv", "nature", "forum")
    source_type: str = ""


@dataclass
class KnowledgeEdge:
    id: str
    source_id: str
    target_id: str

    semantic_similarity:   float = 0.0
    temporal_proximity:    float = 0.0
    limitation_resolution: float = 0.0
    citation_link:         float = 0.0
    investment_correlation: float = 0.0
    social_correlation:    float = 0.0

    # Extension: inhibitory component
    inhibitory_force: float = 0.0

    total_weight: float = 0.0
    confidence:   float = 0.0

    relationship_type:  str       = "related"
    relation_type:      str       = ""        # alias used by topology module
    discovery_method:   str       = "direct"
    evidence:           List[str] = field(default_factory=list)
    timestamp:          float     = 0.0

    is_containment_edge: bool = False

    def compute_total_weight(self) -> float:
        w = V6_EDGE_WEIGHTS
        raw = (
            self.semantic_similarity    * w["semantic"]              +
            self.temporal_proximity     * w["temporal"]              +
            self.limitation_resolution  * w["limitation_resolution"] +
            self.citation_link          * w["citation"]              +
            self.investment_correlation * w["investment"]            +
            self.social_correlation     * w["social"]               +
            INHIBITORY_WEIGHT           * self.inhibitory_force
        )
        # total_weight is consumed as a conductance/capacity by GraphWalker,
        # ObligatoryBridgeDetector's max-flow, and any future caller — a
        # negative value there isn't "extra-strong resistance", it's a type
        # the algorithm doesn't have a meaning for (max-flow needs
        # non-negative capacities; an attention mechanism reading this as a
        # scale would zero out instead of registering "high conflict here").
        # Clamped at the source rather than relying on every consumer to
        # remember its own max(x, 0.0). The raw signed balance is still
        # available directly via inhibitory_force for anything that wants to
        # reason about resistance as its own quantity rather than net
        # capacity.
        self.total_weight = max(0.0, raw)
        return self.total_weight


@dataclass
class TemporalZone:
    id: str
    description: str
    start_timestamp: float
    end_timestamp: float
    domain_focus: str
    zone_multiplier: float
    contained_node_ids: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

    def covers(self, domain: str, timestamp: float) -> bool:
        """
        Geometric field test (Manifest §1, "Temporal zones are fields, not
        nodes"): a zone is a region in (domain, time), and any node whose
        own coordinates fall inside that region is automatically under its
        gravity — membership is a coordinate check, not a semantic-
        similarity judgment against the zone's centroid.

        Time: [start_timestamp, end_timestamp] inclusive.
        Domain: `domain_focus` is treated as a substring match against the
        node's domain (case-insensitive) so a broad focus like "physics"
        still covers a more specific "physics.optics" node domain, and an
        empty domain_focus covers every domain (a purely temporal zone,
        e.g. "the 1940s" with no domain restriction).
        """
        if not (self.start_timestamp <= timestamp <= self.end_timestamp):
            return False
        if not self.domain_focus:
            return True
        focus = self.domain_focus.lower().strip()
        dom = (domain or "").lower().strip()
        return focus in dom or dom in focus


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

class HashingTextEmbedder:
    """Deterministic local text representation used only for graph features.

    It has no learned world knowledge and never replaces the ingestion LLM.
    Its purpose is operational: every node must have a stable 384-dimensional
    feature representation even when a sentence-transformer is unavailable.
    Stable hashing also means a saved graph does not change merely because a
    different optional embedding package happens to be installed on restart.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = int(dimension)

    def encode(self, text: str) -> np.ndarray:
        tokens = re.findall(r"[\w]+", (text or "").casefold(), flags=re.UNICODE)
        vector = np.zeros(self.dimension, dtype=np.float32)
        if not tokens:
            return vector
        # Unigrams preserve concept identity; adjacent pairs retain a small
        # amount of mechanism/order information without an external model.
        features = tokens + [f"{a}\x1f{b}" for a, b in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            raw = int.from_bytes(digest, "little")
            index = raw % self.dimension
            vector[index] += 1.0 if (raw >> 63) == 0 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else vector


class NodeFeatureBuilder:
    TEXT_DIM       = 384
    READINESS_DIM  = 5
    ACCEL_DIM      = 4
    STRUCT_DIM     = 3
    ZONE_DIM       = 4
    SOCIAL_DIM     = 3
    INVEST_DIM     = 2
    FORUM_DIM      = 1
    PLATEAU_DIM    = 1   # efficiency_plateau — incumbent technology ceiling
    TEMPORAL_DIM   = 6   # temporal_kind (point/interval/axiom) + density shape

    TOTAL_DIM = (TEXT_DIM + READINESS_DIM + ACCEL_DIM + STRUCT_DIM +
                 ZONE_DIM + SOCIAL_DIM + INVEST_DIM + FORUM_DIM + PLATEAU_DIM +
                 TEMPORAL_DIM)  # 413

    SLICES = {
        "text":      slice(0,   384),
        "readiness": slice(384, 389),
        "accel":     slice(389, 393),
        "struct":    slice(393, 396),
        "zone":      slice(396, 400),
        "social":    slice(400, 403),
        "invest":    slice(403, 405),
        "forum":     slice(405, 406),
        "plateau":   slice(406, 407),   # efficiency_plateau / 10.0
        # temporal_kind + density profile shape (see KnowledgeNode and the
        # README note on epistemic plugs / temporal density): appended at
        # the end rather than inserted, so it doesn't shift any existing
        # slice boundary anywhere else that might hardcode one.
        "temporal":  slice(407, 413),
    }

    def __init__(self, sentence_embedder=None):
        self.embedder = sentence_embedder or HashingTextEmbedder(self.TEXT_DIM)

    def build(self, node: KnowledgeNode) -> np.ndarray:
        vec = np.zeros(self.TOTAL_DIM, dtype=np.float32)
        raw_emb = np.asarray(self.embedder.encode(node.text or node.entity_type), dtype=np.float32).reshape(-1)
        emb = np.zeros(self.TEXT_DIM, dtype=np.float32)
        emb[:min(len(raw_emb), self.TEXT_DIM)] = raw_emb[:self.TEXT_DIM]
        norm = float(np.linalg.norm(emb))
        if norm > 1e-12:
            emb /= norm
        vec[self.SLICES["text"]] = emb
        node.embedding = emb
        vec[self.SLICES["readiness"]] = [
            node.scientific_score  / 10.0,
            node.investment_score  / 10.0,
            node.social_score      / 10.0,
            node.maturity_score    / 10.0,
            min(node.group_count, 100) / 100.0,
        ]
        vec[self.SLICES["accel"]] = [
            node.dual_use_risk       / 10.0,
            node.strategic_value     / 10.0,
            node.legal_risk_score    / 10.0,
            node.export_control_risk / 10.0,
        ]
        vec[self.SLICES["zone"]] = [
            1.0 if node.is_temporal_zone else 0.0,
            node.zone_multiplier / 5.0,
            node.acceleration_multiplier / 5.0,
            min(node.discovery_depth, 10) / 10.0,
        ]
        vec[self.SLICES["social"]] = [
            node.sentiment_review_score,
            node.sentiment_fiction_score,
            node.sentiment_forum_score,
        ]
        vec[self.SLICES["invest"]] = [
            math.log1p(node.investment_total_usd) / 20.0,
            min(node.investment_rounds, 20) / 20.0,
        ]
        vec[self.SLICES["forum"]] = [
            math.log1p(node.forum_post_count) / 10.0,
        ]
        vec[self.SLICES["plateau"]] = [
            min(node.efficiency_plateau, 10.0) / 10.0,
        ]

        # Temporal shape: point/interval/axiom + density profile summary.
        # temporal_density/timestamp_start/timestamp_end exist on the node
        # (see SignificanceProcessor._sync_epistemic_to_node) but were
        # otherwise invisible to the GNN — only the single-point `node_years`
        # input reached it, so "a paper" and "a decade-long debate that just
        # happens to average to the same year" looked identical going in.
        kind = getattr(node, "temporal_kind", "point")
        density = getattr(node, "temporal_density", None) or []
        year_secs = 365.25 * 24 * 3600
        span_years = 0.0
        if kind == "interval":
            span_years = max(0.0, (node.timestamp_end - node.timestamp_start) / year_secs)
        peak_density = max(density) if density else 0.0
        trend = 0.0
        if len(density) >= 2:
            mid = len(density) // 2
            early, late = density[:mid] or [0.0], density[mid:] or [0.0]
            trend = float(np.tanh((np.mean(late) - np.mean(early)) / 2.0))
        vec[self.SLICES["temporal"]] = [
            1.0 if kind == "point" else 0.0,
            1.0 if kind == "interval" else 0.0,
            1.0 if kind == "axiom" else 0.0,
            min(1.0, math.log1p(span_years) / 5.0),
            min(1.0, peak_density / 10.0),
            trend,   # -1 fading toward now .. 0 flat/point .. +1 thickening toward now
        ]
        return vec

    def update_structural_section(self, vec: np.ndarray,
                                   sdi: float, cascade: float,
                                   upstream_pressure: float) -> np.ndarray:
        vec = vec.copy()
        vec[self.SLICES["struct"]] = [
            min(sdi, 10) / 10.0,
            min(cascade, 10) / 10.0,
            max(-5, min(upstream_pressure, 5)) / 5.0,
        ]
        return vec


class EdgeFeatureBuilder:
    # Use all 7 components including inhibitory_force (Fix 6)
    DIM = N_EXTENDED_COMPONENTS + 2        # 7 + 2 = 9
    COMP_SLICE    = slice(0, N_EXTENDED_COMPONENTS)  # 0:7
    IS_CONT_IDX   = N_EXTENDED_COMPONENTS   # 7
    ZONE_MULT_IDX = N_EXTENDED_COMPONENTS + 1  # 8

    def build(self, edge: KnowledgeEdge) -> np.ndarray:
        vec = np.zeros(self.DIM, dtype=np.float32)
        if edge.is_containment_edge:
            vec[self.IS_CONT_IDX] = 1.0
            vec[self.ZONE_MULT_IDX] = 1.0
        else:
            vec[self.COMP_SLICE] = [
                edge.semantic_similarity,
                edge.temporal_proximity,
                edge.limitation_resolution,
                edge.citation_link,
                edge.investment_correlation,
                edge.social_correlation,
                edge.inhibitory_force,   # 7th component — the only negative weight signal
            ]
        return vec


class PhysicalSubstrateEncoder:
    EXTENDED_TOTAL_DIM = NodeFeatureBuilder.TOTAL_DIM + PHYSICAL_SUBSTRATE_DIM  # 416

    EXTENDED_SLICES = {
        **NodeFeatureBuilder.SLICES,
        "physical": slice(NodeFeatureBuilder.TOTAL_DIM,
                          NodeFeatureBuilder.TOTAL_DIM + PHYSICAL_SUBSTRATE_DIM),
    }

    PHYSICAL_SUBSTRATE_PROMPT = """Analyse this technology node in terms of
fundamental physical and resource constraints.

NODE TEXT: {node_text}
DOMAIN: {domain}
ENTITY TYPE: {entity_type}

Rate the following 10 abstract constraint axes on a 0.0–1.0 scale.
Return ONLY valid JSON:
{{
  "resource_intensity":     <0.0-1.0>,
  "scale_feasibility":      <0.0-1.0>,
  "constraint_proximity":   <0.0-1.0>,
  "reversibility":          <0.0-1.0>,
  "parallel_deployability": <0.0-1.0>,
  "environmental_coupling": <0.0-1.0>,
  "cross_scale_stability":  <0.0-1.0>,
  "synthesis_complexity":   <0.0-1.0>,
  "longevity":              <0.0-1.0>,
  "cascadability":          <0.0-1.0>
}}"""

    def build_physical_section(self, llm_scores: Dict[str, float]) -> np.ndarray:
        vec = np.zeros(PHYSICAL_SUBSTRATE_DIM, dtype=np.float32)
        for i, axis in enumerate(PHYSICAL_AXIS_ORDER):
            vec[i] = float(np.clip(llm_scores.get(axis, 0.5), 0.0, 1.0))
        return vec

    def extend_node_features(self, base_features: np.ndarray,
                              physical_scores: Dict[str, float]) -> np.ndarray:
        phys = self.build_physical_section(physical_scores)
        return np.concatenate([base_features, phys])

    def feasibility_score(self, physical_vec: np.ndarray) -> float:
        fw = CONFIG["feasibility_axis_weights"]
        cp  = float(physical_vec[PHYSICAL_AXIS_ORDER.index("constraint_proximity")])
        ri  = float(physical_vec[PHYSICAL_AXIS_ORDER.index("resource_intensity")])
        sc  = float(physical_vec[PHYSICAL_AXIS_ORDER.index("synthesis_complexity")])
        return max(CONFIG["feasibility_floor"],
                   1.0 - (cp * fw["constraint_proximity"] +
                          ri * fw["resource_intensity"] +
                          sc * fw["synthesis_complexity"]))

    # ── Rupture Physics Modifier ──────────────────────────────────────────────
    # When an ONTOLOGICAL_RUPTURE is confirmed for a node, old physical barriers
    # in that zone become obsolete. This method collapses constraint_proximity,
    # synthesis_complexity and resource_intensity for all dependent child nodes,
    # opening the door for Recursive Forecast hallucinations that were previously
    # filtered as "physically impossible".
    #
    # collapse_factor: 0.0 = total collapse, 1.0 = no change (default 0.35)
    # affected_axes:   subset of PHYSICAL_AXIS_ORDER to relax; if None → defaults
    RUPTURE_COLLAPSE_FACTOR  = CONFIG["rupture_collapse_factor"]
    RUPTURE_DEFAULT_AXES     = CONFIG["rupture_default_axes"]

    def resolve_rupture_physics(
        self,
        ruptured_node_id: str,
        graph: "RuntimeGraph",
        collapse_factor: float = None,
        affected_axes: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Re-compute physical constraint vectors for all nodes that depend on
        (are reachable from) *ruptured_node_id* in the graph.

        Returns a dict {node_id: relaxed_physical_vec} for every affected node.
        The caller should store these back into the node features / use them to
        override feasibility_score() results.

        The relaxation formula:
            new_axis_value = old_value * collapse_factor
        i.e. barriers shrink toward 0 – meaning "easier", "less constrained".
        """
        if collapse_factor is None:
            collapse_factor = self.RUPTURE_COLLAPSE_FACTOR
        collapse_factor = float(np.clip(collapse_factor, 0.0, 1.0))
        if affected_axes is None:
            affected_axes = self.RUPTURE_DEFAULT_AXES

        axis_indices = [
            PHYSICAL_AXIS_ORDER.index(ax) for ax in affected_axes
            if ax in PHYSICAL_AXIS_ORDER
        ]

        # BFS / DFS to find all child nodes reachable from ruptured_node_id
        visited: Set[str] = set()
        frontier = [ruptured_node_id]
        while frontier:
            nid = frontier.pop()
            if nid in visited:
                continue
            visited.add(nid)
            for (src, tgt) in graph.edges:
                if src == nid and tgt not in visited:
                    edge = graph.edges.get((src, tgt))
                    if edge and not edge.is_containment_edge:
                        frontier.append(tgt)

        relaxed: Dict[str, np.ndarray] = {}
        for nid in visited:
            node = graph.get_node(nid)
            if not node or node.is_temporal_zone:
                continue
            # Build physical vec from node's current feature embedding if available;
            # otherwise start from a neutral 0.5 baseline.
            if node.embedding is not None and len(node.embedding) >= self.EXTENDED_TOTAL_DIM:
                phys_vec = node.embedding[
                    NodeFeatureBuilder.TOTAL_DIM:
                    NodeFeatureBuilder.TOTAL_DIM + PHYSICAL_SUBSTRATE_DIM
                ].copy().astype(np.float32)
            else:
                phys_vec = np.full(PHYSICAL_SUBSTRATE_DIM, 0.5, dtype=np.float32)

            # Relax the chosen constraint axes
            for idx in axis_indices:
                phys_vec[idx] = float(phys_vec[idx] * collapse_factor)

            relaxed[nid] = phys_vec
            logger.info(
                f"[RupturePhysics] Node '{nid}' relaxed "
                f"axes={affected_axes} factor={collapse_factor:.2f} "
                f"new_cp={phys_vec[PHYSICAL_AXIS_ORDER.index('constraint_proximity')]:.3f}"
            )

        logger.warning(
            f"[RupturePhysics] ONTOLOGICAL RUPTURE at '{ruptured_node_id}' "
            f"→ relaxed physics for {len(relaxed)} dependent nodes."
        )
        return relaxed


# ══════════════════════════════════════════════════════════════════════════════
#  GNN MODEL COMPONENTS
# ══════════════════════════════════════════════════════════════════════════════

class TemporalEncoder(nn.Module):
    def __init__(self, latent_dim: int, max_years: int = 300):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.latent_dim = latent_dim
        pe = torch.zeros(max_years, latent_dim)
        log_y = torch.arange(max_years).float().unsqueeze(1)
        div   = torch.exp(torch.arange(0, latent_dim, 2).float()
                          * (-math.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(log_y * div)
        pe[:, 1::2] = torch.cos(log_y * div)
        self.register_buffer("pe", pe)

    def forward(self, x, year, zone_multiplier=None):
        if not TORCH_AVAILABLE:
            return x
        if zone_multiplier is not None:
            effective_year = (year.float() + (zone_multiplier - 1.0) * 5).long()
        else:
            effective_year = year
        effective_year = effective_year.clamp(0, self.pe.size(0) - 1)
        return x + self.pe[effective_year]

__all__ = ['KnowledgeNode', 'KnowledgeEdge', 'TemporalZone', 'HashingTextEmbedder', 'NodeFeatureBuilder', 'EdgeFeatureBuilder', 'PhysicalSubstrateEncoder', 'TemporalEncoder']
