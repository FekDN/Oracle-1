# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Topology, structural tension, and phase-transition analysis."""
from __future__ import annotations

from ._shared import *
from .config import (
    ALERT_ANOMALY_SATURATION,
    ALERT_DOMAIN_TRANSPLANT,
    ALERT_ONTOLOGICAL_RUPTURE,
    ALERT_OVERLOAD,
    ALERT_VOID_COLLAPSE,
    ATTENTION_GAMMA,
    COLLAPSE_THRESHOLD,
    CONFIG,
    EDGE_WEIGHT_COMPONENTS,
    N_EDGE_COMPONENTS,
    PHASE_THRESHOLDS,
    PHYSICAL_AXIS_ORDER,
    VIRTUAL_NODE_TENSION_THRESHOLD,
)
from .entities import KnowledgeEdge, KnowledgeNode
from .forecasting import PhantomNode
from .graph import RuntimeGraph
from .graph_dynamics import ConvergencePressureField, VoidPressureVector
from .time_utils import _ts_to_year

# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS — CONVERGENCE TIME SERIES & ACCELEROMETER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceTimeSeries:
    node_id: str
    window_size: int = CONFIG["conv_series_window_size"]
    _timestamps: Deque[float] = field(default_factory=lambda: deque(maxlen=CONFIG["conv_series_window_size"]))
    _scores:     Deque[float] = field(default_factory=lambda: deque(maxlen=CONFIG["conv_series_window_size"]))

    @property
    def score(self) -> float:
        return self._scores[-1] if self._scores else 0.0

    @property
    def velocity(self) -> float:
        if len(self._scores) < 2: return 0.0
        s = list(self._scores); t = list(self._timestamps)
        dt = t[-1] - t[-2]
        return 0.0 if dt <= 0 else (s[-1] - s[-2]) / dt

    @property
    def acceleration(self) -> float:
        if len(self._scores) < 3: return 0.0
        s = list(self._scores); t = list(self._timestamps)
        dt1, dt2 = t[-2] - t[-3], t[-1] - t[-2]
        if dt1 <= 0 or dt2 <= 0: return 0.0
        v1 = (s[-2] - s[-3]) / dt1; v2 = (s[-1] - s[-2]) / dt2
        return (v2 - v1) / ((dt1 + dt2) / 2.0)

    @property
    def jerk(self) -> float:
        if len(self._scores) < 4: return 0.0
        s = list(self._scores); t = list(self._timestamps)
        def acc_at(i):
            dt1, dt2 = t[i-1] - t[i-2], t[i] - t[i-1]
            if dt1 <= 0 or dt2 <= 0: return 0.0
            return ((s[i] - s[i-1]) / dt2 - (s[i-1] - s[i-2]) / dt1) / ((dt1 + dt2) / 2.0)
        dt = t[-1] - t[-2]
        return 0.0 if dt <= 0 else (acc_at(-1) - acc_at(-2)) / dt

    def push(self, timestamp: float, score: float):
        self._timestamps.append(timestamp); self._scores.append(score)

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.score,
            np.tanh(self.velocity     * CONFIG["conv_series_vel_scale"]),
            np.tanh(self.acceleration * CONFIG["conv_series_accel_scale"]),
            np.tanh(self.jerk         * CONFIG["conv_series_jerk_scale"]),
        ], dtype=np.float32)

    def predicted_score_at(self, delta_t: float) -> float:
        return (self.score + self.velocity * delta_t + 0.5 * self.acceleration * delta_t ** 2)


class ConvergenceAccelerometer:
    WINDOW_EPOCHS = CONFIG["accelerometer_window_epochs"]
    C_WEIGHTS = CONFIG["accelerometer_c_weights"]
    EPOCHS_PER_YEAR = CONFIG["accelerometer_epochs_per_year"]

    def __init__(self):
        self.series: Dict[str, ConvergenceTimeSeries] = {}

    def _get_or_create(self, node_id: str) -> ConvergenceTimeSeries:
        if node_id not in self.series:
            self.series[node_id] = ConvergenceTimeSeries(node_id=node_id,
                                                          window_size=self.WINDOW_EPOCHS)
        return self.series[node_id]

    def update(self, node_id: str, graph: RuntimeGraph,
               pressure_field: ConvergencePressureField,
               all_embeddings: Dict[str, np.ndarray],
               timestamp: float) -> ConvergenceTimeSeries:
        node = graph.get_node(node_id)
        if not node: return self._get_or_create(node_id)
        k2 = graph.k_hop_neighborhood([node_id], k=2)
        k2_nodes = [graph.get_node(nid) for nid in k2
                    if nid != node_id and graph.get_node(nid)]
        sem_score = 0.0
        if node.embedding is not None and k2_nodes:
            nbr_embs = [n.embedding for n in k2_nodes if n.embedding is not None]
            if nbr_embs:
                sims = cos_sim([node.embedding], nbr_embs)[0]
                sem_score = float(np.mean(sims))
        n_k2 = len(k2)
        max_edges = max(1, n_k2 * (n_k2 - 1) // 2)
        actual_edges = sum(1 for (s, t) in graph.edges
                           if s in k2 and t in k2 and not graph.edges[(s, t)].is_containment_edge)
        edge_score = min(1.0, actual_edges / max_edges)
        pv = pressure_field.field.get(node_id, VoidPressureVector(node_id=node_id))
        pressure_score = min(1.0, pv.void_pressure / CONFIG["accelerometer_pressure_norm"])
        incoming = [(s, t) for (s, t) in graph.edges if t == node_id]
        resource_score = float(np.mean([graph.edges[k].investment_correlation for k in incoming
                                         if not graph.edges[k].is_containment_edge] or [0.0]))
        resonance_score = float(np.mean([graph.edges[k].limitation_resolution for k in incoming
                                          if not graph.edges[k].is_containment_edge] or [0.0]))
        w = self.C_WEIGHTS
        c_t = (sem_score * w["semantic"] + edge_score * w["edge"] +
               pressure_score * w["pressure"] + resource_score * w["resource"] +
               resonance_score * w["resonance"])
        series = self._get_or_create(node_id)
        series.push(timestamp, c_t)
        return series

    def update_all(self, graph, pressure_field, all_embeddings, timestamp):
        for nid in list(graph.nodes):
            self.update(nid, graph, pressure_field, all_embeddings, timestamp)
        return self.series

    def top_accelerating(self, n: int = 10) -> List[Tuple[str, float]]:
        scored = [(nid, s.acceleration) for nid, s in self.series.items() if len(s._scores) >= 3]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def pre_breakthrough_candidates(self, min_acceleration=0.01, min_jerk=0.005):
        candidates = [(nid, s) for nid, s in self.series.items()
                      if s.acceleration >= min_acceleration and s.jerk >= min_jerk]
        candidates.sort(key=lambda x: x[1].acceleration, reverse=True)
        return candidates


class DualObjectiveHead(nn.Module):
    HORIZON_EPOCHS = CONFIG["dual_head_horizon_epochs"]

    def __init__(self, latent_dim: int, deriv_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.deriv_dim  = deriv_dim
        if not TORCH_AVAILABLE: return
        self.struct_enc = nn.Sequential(
            nn.Linear(latent_dim * 2, 128), nn.GELU(), nn.LayerNorm(128))
        self.deriv_enc = nn.Sequential(
            nn.Linear(deriv_dim * 2, 64), nn.GELU(), nn.Linear(64, 64))
        self.decoder = nn.Sequential(
            nn.Linear(128 + 64, 128), nn.GELU(), nn.LayerNorm(128),
            nn.Linear(128, 64), nn.GELU())
        self.priority_head = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.eta_head      = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        if TORCH_AVAILABLE:
            self.alpha_score = nn.Parameter(torch.tensor(0.5))
            self.alpha_accel = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_src, h_tgt, deriv_src, deriv_tgt):
        if not TORCH_AVAILABLE: return None, None
        struct = self.struct_enc(torch.cat([h_src, h_tgt], dim=-1))
        deriv  = self.deriv_enc(torch.cat([deriv_src, deriv_tgt], dim=-1))
        shared = self.decoder(torch.cat([struct, deriv], dim=-1))
        priority = self.priority_head(shared)
        eta      = self.eta_head(shared) * self.HORIZON_EPOCHS
        return priority, eta

    def loss(self, priority_pred, eta_pred, breakthrough_labels, actual_etas):
        if not TORCH_AVAILABLE: return 0.0
        bce = F.binary_cross_entropy(priority_pred.squeeze(-1), breakthrough_labels)
        mask = breakthrough_labels > 0.5
        eta_loss = torch.tensor(0.0)
        if mask.sum() > 0:
            eta_loss = F.mse_loss(eta_pred.squeeze(-1)[mask] / self.HORIZON_EPOCHS,
                                   actual_etas[mask] / self.HORIZON_EPOCHS)
        return bce + CONFIG["dual_head_eta_loss_weight"] * eta_loss


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS — STABILIZED CENTROID TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StabilizedCentroid:
    cluster_id: str
    dim: int
    alpha_add:   float = CONFIG["centroid_alpha_add"]
    alpha_epoch: float = CONFIG["centroid_alpha_epoch"]
    _centroid:        np.ndarray = field(default_factory=lambda: None)
    _size:            int        = 0
    _velocity:        np.ndarray = field(default_factory=lambda: None)
    _semantic_drift:  float      = 0.0
    _size_drift:      float      = 0.0
    _history:     Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=CONFIG["centroid_history_len"]))
    _size_history: Deque[int]       = field(default_factory=lambda: deque(maxlen=CONFIG["centroid_history_len"]))

    def __post_init__(self):
        if self._centroid is None: self._centroid = np.zeros(self.dim, dtype=np.float64)
        if self._velocity is None: self._velocity = np.zeros(self.dim, dtype=np.float64)

    def add_node(self, embedding: np.ndarray, weight: float = 1.0) -> np.ndarray:
        emb = embedding.astype(np.float64)
        if self._size == 0:
            self._centroid = emb.copy()
        else:
            w = self.alpha_add / math.sqrt(max(1, self._size)) * weight
            delta = emb - self._centroid
            c_unit = self._centroid / (np.linalg.norm(self._centroid) + 1e-12)
            parallel = np.dot(delta, c_unit) * c_unit
            orthogonal = delta - parallel
            self._size_drift     += float(np.linalg.norm(parallel)) * w
            self._semantic_drift += float(np.linalg.norm(orthogonal)) * w
            self._centroid = self._centroid + w * delta
        self._size += 1
        return self._centroid.copy()

    def epoch_update(self, all_embeddings: List[np.ndarray],
                      weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Weiszfeld-iterated weighted geometric median.

        `weights` should be epistemic_mass (or similar quality signal) per
        point, not left uniform: 10,000 forum posts about phone cameras and
        1 Nature paper about 5D quartz storage are not equally-weighted
        voters for where a domain's centroid sits. Without real weights, the
        median (even inverse-distance-weighted) is still dominated by
        whichever cluster has more *points*, not more *evidentiary mass* —
        VirtualNodeSynthesizer would then chase "the median smartphone"
        instead of the rare, heavy breakthrough signal.

        `weights=None` falls back to uniform 1.0 (previous behaviour,
        preserved for any caller that hasn't been updated yet).
        """
        if not all_embeddings: return self._centroid.copy()
        prev = self._centroid.copy()
        mat = np.stack([e.astype(np.float64) for e in all_embeddings])
        w_arr = (np.asarray(weights, dtype=np.float64) if weights is not None
                 else np.ones(len(all_embeddings), dtype=np.float64))
        w_arr = np.clip(w_arr, 1e-6, None)  # never literally zero out a voter
        current = np.average(mat, axis=0, weights=w_arr)
        for _ in range(CONFIG["centroid_weiszfeld_iterations"]):
            dists = np.linalg.norm(mat - current, axis=1) + 1e-8
            iw = w_arr / dists; iw /= iw.sum()
            current = (mat * iw[:, None]).sum(axis=0)
        self._centroid = (1 - self.alpha_epoch) * self._centroid + self.alpha_epoch * current
        v_new = self._centroid - prev
        self._velocity = CONFIG["centroid_velocity_ema_decay"] * self._velocity + CONFIG["centroid_velocity_ema_alpha"] * v_new
        self._history.append(self._centroid.copy())
        self._size_history.append(len(all_embeddings))
        return self._centroid.copy()

    @property
    def semantic_convergence_velocity(self) -> float:
        if len(self._history) < 2: return 0.0
        displacement = self._history[-1] - self._history[-2]
        disp_norm = np.linalg.norm(displacement)
        if disp_norm < 1e-10: return 0.0
        c_unit = self._centroid / (np.linalg.norm(self._centroid) + 1e-12)
        parallel = np.dot(displacement, c_unit)
        return math.sqrt(max(0, disp_norm**2 - parallel**2))

    @property
    def is_size_inflated(self) -> bool:
        if len(self._size_history) < 2: return False
        size_growth = self._size_history[-1] / max(1, self._size_history[-2])
        return (size_growth > CONFIG["centroid_size_inflation_growth"]
                and self.semantic_convergence_velocity < CONFIG["centroid_size_inflation_vel_cap"])


class StabilizedCentroidTracker:
    def __init__(self, embedding_dim: int = 384):
        self.dim = embedding_dim
        self.centroids: Dict[str, StabilizedCentroid] = {}

    def _get_or_create(self, cluster_id: str) -> StabilizedCentroid:
        if cluster_id not in self.centroids:
            self.centroids[cluster_id] = StabilizedCentroid(cluster_id=cluster_id, dim=self.dim)
        return self.centroids[cluster_id]

    def add_node(self, node: KnowledgeNode) -> Optional[np.ndarray]:
        if node.embedding is None or node.is_temporal_zone: return None
        # Floor at 0.1 rather than 0.0: a brand-new, not-yet-reinforced node
        # should barely move the centroid, not be silently invisible to it —
        # only actual epistemic_mass (up to the 10.0 cap) lets it pull hard.
        weight = max(0.1, getattr(node, "epistemic_mass", 0.0))
        if node.zone_id:
            self._get_or_create(f"zone:{node.zone_id}").add_node(node.embedding[:self.dim], weight)
        return self._get_or_create(node.domain).add_node(node.embedding[:self.dim], weight)

    def epoch_update(self, graph: RuntimeGraph):
        domain_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
        domain_wts:  Dict[str, List[float]]       = defaultdict(list)
        zone_embs:   Dict[str, List[np.ndarray]] = defaultdict(list)
        zone_wts:    Dict[str, List[float]]       = defaultdict(list)
        for nid, node in graph.nodes.items():
            if node.embedding is None or node.is_temporal_zone: continue
            weight = max(0.1, getattr(node, "epistemic_mass", 0.0))
            domain_embs[node.domain].append(node.embedding[:self.dim])
            domain_wts[node.domain].append(weight)
            if node.zone_id:
                zone_embs[f"zone:{node.zone_id}"].append(node.embedding[:self.dim])
                zone_wts[f"zone:{node.zone_id}"].append(weight)
        all_embs = {**domain_embs, **zone_embs}
        all_wts  = {**domain_wts, **zone_wts}
        for cluster_id, embs in all_embs.items():
            self._get_or_create(cluster_id).epoch_update(embs, all_wts[cluster_id])

    def get_centroid(self, cluster_id: str) -> Optional[np.ndarray]:
        c = self.centroids.get(cluster_id)
        return c._centroid.copy() if c is not None else None

    def zone_centroid(self, zone_id: str) -> Optional[np.ndarray]:
        return self.get_centroid(f"zone:{zone_id}")

    def true_convergence_velocity(self, cluster_id: str) -> float:
        c = self.centroids.get(cluster_id)
        return c.semantic_convergence_velocity if c else 0.0

    def false_convergence_nodes(self) -> List[str]:
        return [cid for cid, c in self.centroids.items() if c.is_size_inflated]


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS — OBLIGATORY BRIDGE DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StructuralVoid:
    id: str
    node_a_id: str
    node_b_id: str
    semantic_pull: float
    topological_gap: float
    pressure_gradient: float
    obligatory_bridge_desc: str
    bridge_domain: str
    void_severity: float
    
    # --- Extended fields for Oracle-1 ---
    is_rupture_driven: bool = False # Did the void arise due to an ontological breakthrough?
    inhibitory_drag: float = 0.0     # To what extent do incumbents interfere with the construction of the bridge?
    physical_feasibility: float = 0.5 # Forecast of the physical feasibility of the bridge
    creation_timestamp: float = 0.0


class ObligatoryBridgeDetector:
    SEM_PULL_THRESHOLD  = CONFIG["bridge_sem_pull_threshold"]
    PRESSURE_THRESHOLD  = CONFIG["bridge_pressure_threshold"]
    CAPACITY_THRESHOLD  = CONFIG["bridge_capacity_threshold"]
    MAX_VOIDS_PER_EPOCH = CONFIG["bridge_max_voids_per_epoch"]

    def __init__(self):
        self.voids: List[StructuralVoid] = []

    def scan(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField,
             centroid_tracker: StabilizedCentroidTracker, timestamp: float) -> List[StructuralVoid]:
        
        # Take nodes with high pressure or those where a rupture has recently occurred
        node_list = [(nid, n) for nid, n in graph.nodes.items()
                     if not n.is_temporal_zone and n.embedding is not None]
        
        if len(node_list) < 2: return []

        # Sort by pressure + take into account the break flag (gives 1.5x priority)
        sorted_by_pressure = sorted(
            node_list, 
            key=lambda x: getattr(x[1], "upstream_pressure", 0.0) * (CONFIG["bridge_rupture_priority_multiplier"] if x[1].rupture_physics_relaxed else 1.0),
            reverse=True
        )[:CONFIG["bridge_top_pressure_nodes"]]

        existing_pairs = {(v.node_a_id, v.node_b_id) for v in self.voids}
        new_voids = []

        for i, (aid, a) in enumerate(sorted_by_pressure):
            for (bid, b) in sorted_by_pressure[i+1:i+CONFIG["bridge_pair_lookahead"]]:
                if (aid, bid) in existing_pairs or (bid, aid) in existing_pairs: continue
                
                void = self._evaluate_pair(aid, a, bid, b, graph, pressure_field, timestamp)
                if void:
                    new_voids.append(void)
                    existing_pairs.add((aid, bid))

        new_voids.sort(key=lambda v: v.void_severity, reverse=True)
        self.voids.extend(new_voids[:self.MAX_VOIDS_PER_EPOCH])
        self.voids.sort(key=lambda v: v.void_severity, reverse=True)
        self.voids = self.voids[:self.MAX_VOIDS_PER_EPOCH]
        return new_voids

    def _evaluate_pair(self, aid, a, bid, b, graph, pressure_field, timestamp):
        if a.embedding is None or b.embedding is None: return None
        
        # 1. Semantic traction (how much ideas "want" to connect)
        sem = float(cos_sim([a.embedding[:384]], [b.embedding[:384]])[0][0])
        if sem < self.SEM_PULL_THRESHOLD: return None

        # 2. Void pressure
        pva = pressure_field.field.get(aid, VoidPressureVector(node_id=aid)).void_pressure
        pvb = pressure_field.field.get(bid, VoidPressureVector(node_id=bid)).void_pressure
        p_norm = (pva + pvb) / CONFIG["bridge_pressure_norm"]
        if p_norm < self.PRESSURE_THRESHOLD: return None

        # 3. Topological Gap (Max-Flow)
        # Use the total_weight of the edges, which already includes inhibitory forces
        max_flow = self._compute_max_flow(aid, bid, graph, max_hops=4)
        topo_gap = 1.0 - max_flow
        if topo_gap < (1.0 - self.CAPACITY_THRESHOLD): return None

        # 4. Analysis of alternative paths (redundancy)
        alt_paths = self._count_alternative_paths(aid, bid, graph, max_hops=5)
        density_penalty = math.log1p(alt_paths)

        # 5. Rupture Factor
        # If a rupture occurs in one of the nodes, the void becomes critical
        rupture_boost = 1.0
        if a.rupture_physics_relaxed or b.rupture_physics_relaxed:
            rupture_boost = CONFIG["bridge_rupture_boost"]

        # 6. Severity Calculation
        void_severity = (sem * topo_gap * min(1.0, p_norm) * rupture_boost) / (1.0 + density_penalty)
        
        if void_severity < CONFIG["bridge_severity_floor"]: return None

        drag = (a.strategic_value + b.strategic_value) / 20.0 * CONFIG["bridge_drag_scale"]

        return StructuralVoid(
            id="void_" + str(uuid.uuid4())[:8],
            node_a_id=aid, 
            node_b_id=bid, 
            semantic_pull=sem,
            topological_gap=topo_gap, 
            pressure_gradient=p_norm,
            is_rupture_driven=(rupture_boost > 1.0),
            inhibitory_drag=drag,
            obligatory_bridge_desc=(
                f"Obligatory bridge between '{a.text[:35]}' and '{b.text[:35]}'. "
                f"Filling this void resolves {a.domain} constraints using {b.domain} principles."),
            bridge_domain=f"{a.domain} × {b.domain}",
            void_severity=void_severity, 
            creation_timestamp=timestamp
        )

    def _compute_max_flow(self, src, tgt, graph, max_hops=4) -> float:
        best: Dict[str, float] = {src: float("inf")}
        heap = [(-float("inf"), src, 0)]
        visited: Dict[str, int] = {}
        while heap:
            neg_cap, node, hops = heapq.heappop(heap)
            cap = -neg_cap
            if hops > max_hops: continue
            if node in visited and visited[node] <= hops: continue
            visited[node] = hops
            if node == tgt: return cap
            for (s2, t2), edge in graph.edges.items():
                if edge.is_containment_edge: continue
                nbr = t2 if s2 == node else (s2 if t2 == node else None)
                if nbr is None: continue
                new_cap = min(cap, max(edge.total_weight, 0.0))
                if new_cap > best.get(nbr, 0.0):
                    best[nbr] = new_cap
                    heapq.heappush(heap, (-new_cap, nbr, hops + 1))
        return 0.0

    def _count_alternative_paths(self, src, tgt, graph, max_hops=5, max_count=20) -> int:
        count = 0
        stack = [(src, {src}, 0)]
        while stack and count < max_count:
            cur, visited, hops = stack.pop()
            if hops > max_hops: continue
            for (s2, t2), edge in graph.edges.items():
                if edge.is_containment_edge: continue
                nbr = t2 if s2 == cur else (s2 if t2 == cur else None)
                if nbr is None or nbr in visited: continue
                if nbr == tgt: count += 1
                else: stack.append((nbr, visited | {nbr}, hops + 1))
        return count


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS — PHANTOM PHASE ALIGNER & VIRTUAL NODE SYNTHESIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhantomLagModel:
    phantom_id: str
    phantom_timestamp: float
    physical_readiness: float
    lag_estimate_years: float
    phase_alignment: str
    confidence: float
    LOG_MEAN  = CONFIG["lag_log_mean"]
    LOG_SIGMA = CONFIG["lag_log_sigma"]

    @classmethod
    def expected_lag(cls) -> float:
        return math.exp(cls.LOG_MEAN + 0.5 * cls.LOG_SIGMA ** 2)

    @classmethod
    def lag_probability(cls, observed_lag_years: float) -> float:
        if observed_lag_years <= 0: return 0.0
        log_lag = math.log(observed_lag_years)
        exponent = -((log_lag - cls.LOG_MEAN) ** 2) / (2 * cls.LOG_SIGMA ** 2)
        return (1.0 / (observed_lag_years * cls.LOG_SIGMA * math.sqrt(2 * math.pi))
                * math.exp(exponent))


class PhantomPhaseAligner:
    """
    Analyzes the time gap (lag) between the cultural phantom and reality.
    Uses accelerometer data and physical fitness.
    """
    def __init__(self, accelerometer: ConvergenceAccelerometer):
        self.accel = accelerometer
        self.lag_models: Dict[str, PhantomLagModel] = {}

    def align(self, phantom: PhantomNode, graph: RuntimeGraph,
              pressure_field: ConvergencePressureField,
              current_timestamp: float) -> PhantomLagModel:
        
        # 1. Phantom age analysis
        phantom_year = _ts_to_year(phantom.generation_timestamp)
        current_year = _ts_to_year(current_timestamp)
        phantom_age = max(0, current_year - phantom_year)

        # 2. Calculation of the physical readiness of the substrate
        physical_readiness = self._find_physical_readiness(phantom, graph)
        
        # 3. Dynamic Lag Calculation
        low_b  = CONFIG["lag_readiness_low_boundary"]
        high_b = CONFIG["lag_readiness_high_boundary"]
        if physical_readiness < low_b:
            lag_estimate = PhantomLagModel.expected_lag()
        elif physical_readiness < high_b:
            fraction_done = (physical_readiness - low_b) / (high_b - low_b)
            lag_estimate = PhantomLagModel.expected_lag() * (1.0 - fraction_done * CONFIG["lag_linear_max_reduction"])
        else:
            lag_estimate = max(
                CONFIG["lag_collapse_min_years"],
                PhantomLagModel.expected_lag() * CONFIG["lag_collapse_factor"] * (1.1 - physical_readiness))

        # 4. Taking into account inhibitory pressure (market resistance)
        # If the phantom threatens strong incumbents, Lag increases
        inhibitory_drag = self._estimate_inhibitory_drag(phantom, graph)
        lag_estimate *= (1.0 + inhibitory_drag)

        # 5. Phase determination
        if physical_readiness < CONFIG["lag_phase_stagnant_readiness"] and phantom_age > CONFIG["lag_phase_stagnant_age_years"]: 
            phase = "stagnant_phantom"
        elif physical_readiness > CONFIG["lag_phase_imminent_readiness"] and lag_estimate < CONFIG["lag_phase_imminent_lag_years"]:
            phase = "imminent_collapse"
        elif physical_readiness < CONFIG["lag_phase_early_readiness"] and phantom_age <= CONFIG["lag_phase_early_age_years"]:
            phase = "early_phantom"
        else:
            phase = "aligned"

        # 6. Final forecast confidence
        domain_acceleration = self._domain_acceleration(phantom.predicted_domain, graph)
        confidence = float(np.clip(
            CONFIG["lag_confidence_base"]
            + CONFIG["lag_confidence_readiness_w"] * physical_readiness
            + CONFIG["lag_confidence_accel_w"] * np.tanh(domain_acceleration * CONFIG["lag_confidence_accel_tanh_scale"]),
            0.0, 1.0))

        model = PhantomLagModel(
            phantom_id=phantom.id, 
            phantom_timestamp=phantom.generation_timestamp,
            physical_readiness=physical_readiness, 
            lag_estimate_years=lag_estimate,
            phase_alignment=phase, 
            confidence=confidence
        )
        self.lag_models[phantom.id] = model
        return model

    def _estimate_inhibitory_drag(self, phantom, graph) -> float:
        """Considers how much current domain leaders will interfere with technology."""
        drag = 0.0
        target_domain = phantom.predicted_domain.lower()
        incumbents = [n for n in graph.nodes.values() 
                      if n.domain.lower() in target_domain
                      and n.strategic_value > CONFIG["lag_inhibitory_drag_incumbent_sv_threshold"]]
        for inc in incumbents:
            drag += (inc.strategic_value / 10.0) * CONFIG["lag_inhibitory_drag_per_incumbent"]
        return min(CONFIG["lag_inhibitory_drag_cap"], drag)

    def _find_physical_readiness(self, phantom, graph) -> float:
        # Improved readiness search: looking at the readiness_score of all related technology predecessors
        target_domain = phantom.predicted_domain.lower()
        scores = []
        # Direct connection testing from a phantom
        for nid in phantom.source_node_ids:
            node = graph.get_node(nid)
            if node: scores.append(node.readiness_score)
        
        # Domain check
        for nid, node in graph.nodes.items():
            if target_domain[:8] in node.domain.lower() and not node.is_temporal_zone:
                scores.append(node.readiness_score)
        
        return float(np.mean(scores)) / 10.0 if scores else 0.0

    def _domain_acceleration(self, domain: str, graph: "RuntimeGraph" = None) -> float:
        # Take the average acceleration for all series of the domain from the accelerometer
        if graph is None:
            return 0.0
        accels = [s.acceleration for nid, s in self.accel.series.items()
                  if graph.get_node(nid) and domain.lower() in graph.get_node(nid).domain.lower()]
        return float(np.mean(accels)) if accels else 0.0

    def get_best_phantoms(self, n: int = 5, phase_filter: Optional[str] = "early_phantom"):
        result = []
        for pid, model in self.lag_models.items():
            if phase_filter and model.phase_alignment != phase_filter: continue
            value = model.confidence / (1.0 + model.lag_estimate_years / 50.0)
            result.append((model, value))
        result.sort(key=lambda x: x[1], reverse=True)
        return result[:n]


@dataclass
class VirtualNode:
    id: str
    description: str
    predicted_domain: str
    predicted_entity_type: str
    predicted_embedding: np.ndarray
    predicted_year: int
    predicted_readiness: float
    source_trajectory: List[str]
    trajectory_confidence: float
    generation_method: str
    is_virtual: bool = True

    SYNTHESIS_PROMPT = """
Analyze the strategic frontier of the following knowledge trajectory.
Predict the NEXT inevitable technology that fills the structural void.

TRAJECTORY: {trajectory_descriptions}
DOMAIN: {domain} | PREDICTED BASE YEAR: {year}

Return ONLY valid JSON:
{{
  "description": "Functional summary (e.g., 'Molecular-scale DNA-templated metallic deposition')",
  "entity_type": "Role (e.g., 'frontier_breakthrough', 'bridge_tech')",
  "predicted_readiness": 0.0-1.0,
  "efficiency_plateau": 0.0-1.0, 
  "physical_substrate": {{
      "resource_intensity": 0.0-1.0,      
      "scale_feasibility": 0.0-1.0,       
      "constraint_proximity": 0.0-1.0,    
      "reversibility": 0.0-1.0,           
      "parallel_deployability": 0.0-1.0,  
      "environmental_coupling": 0.0-1.0,  
      "cross_scale_stability": 0.0-1.0,   
      "synthesis_complexity": 0.0-1.0,    
      "longevity": 0.0-1.0,               
      "cascadability": 0.0-1.0            
  }},
  "inhibitory_targets": ["id1", "id2"], 
  "confidence": 0.0-1.0
}}
"""


class VirtualNodeSynthesizer:
    MIN_TRAJECTORY_CONFIDENCE = CONFIG["synth_min_trajectory_confidence"]
    MIN_VOID_SEVERITY         = CONFIG["synth_min_void_severity"]
    MAX_VIRTUAL_NODES         = CONFIG["synth_max_virtual_nodes"]
    PROJECTION_YEARS          = CONFIG["synth_projection_years"]

    def __init__(self, centroid_tracker: StabilizedCentroidTracker,
                 accelerometer: ConvergenceAccelerometer,
                 void_detector: ObligatoryBridgeDetector, llm_client=None):
        self.centroids = centroid_tracker
        self.accel     = accelerometer
        self.voids     = void_detector
        self.llm       = llm_client
        self.virtual_nodes: List[VirtualNode] = []

    def synthesize_all_horizons(self, graph: RuntimeGraph, current_timestamp: float):
        """
        Generates forecasts for all given time horizons.
        """
        for years in self.PROJECTION_YEARS:
            traj_nodes = self.synthesize_from_trajectories(graph, current_timestamp, years)
            self.virtual_nodes.extend(traj_nodes)
        
        void_nodes = self.synthesize_from_voids(graph, current_timestamp)
        self.virtual_nodes.extend(void_nodes)

    def synthesize_from_trajectories(self, graph: RuntimeGraph,
                                      current_timestamp: float,
                                      projection_years: float = 5.0) -> List[VirtualNode]:
        new_nodes: List[VirtualNode] = []
        for domain, centroid_obj in self.centroids.centroids.items():
            if domain.startswith("zone:"): continue
            
            vel = centroid_obj.semantic_convergence_velocity
            if vel < CONFIG["synth_velocity_threshold"] or centroid_obj.is_size_inflated: continue
            
            projected_emb = self._project_centroid(centroid_obj, projection_years * 12.0)
            if projected_emb is None: continue
            
            confidence = min(1.0, vel / CONFIG["synth_velocity_confidence_norm"])
            if confidence < self.MIN_TRAJECTORY_CONFIDENCE: continue
            
            source_ids = self._find_trajectory_nodes(domain, graph, n=5)
            
            # Advanced LLM Challenge for Physics
            desc_data = self._generate_strategic_description(
                projected_emb, domain, source_ids, graph,
                _ts_to_year(current_timestamp) + int(projection_years),
                vel, 0.5
            )
            
            vnode = VirtualNode(
                id="virtual_" + str(uuid.uuid4())[:8],
                description=desc_data['description'],
                predicted_domain=domain,
                predicted_entity_type=desc_data.get('entity_type', "predicted_frontier_technology"),
                predicted_embedding=projected_emb,
                predicted_year=_ts_to_year(current_timestamp) + int(projection_years),
                predicted_readiness=min(CONFIG["synth_readiness_cap"],
                                        CONFIG["synth_readiness_base"] + vel * CONFIG["synth_readiness_from_velocity"]),
                source_trajectory=source_ids,
                trajectory_confidence=confidence,
                generation_method="centroid_projection"
            )
            # Save physical characteristics in additional attributes
            vnode.__dict__['physical_substrate'] = desc_data.get('physical_substrate', {})
            new_nodes.append(vnode)
        return new_nodes

    def synthesize_from_voids(self, graph: RuntimeGraph,
                               current_timestamp: float) -> List[VirtualNode]:
        new_nodes: List[VirtualNode] = []
        for void in self.voids.voids[:CONFIG["synth_void_scan_depth"]]:
            if void.void_severity < self.MIN_VOID_SEVERITY: continue
            
            node_a = graph.get_node(void.node_a_id)
            node_b = graph.get_node(void.node_b_id)
            if not node_a or not node_b: continue
            
            bridge_emb = self._optimal_bridge_embedding(node_a.embedding[:384], node_b.embedding[:384], void)
            target_year = _ts_to_year(current_timestamp) + CONFIG["bridge_void_default_target_years"]
            
            desc_data = self._generate_strategic_description(
                bridge_emb, void.bridge_domain, [void.node_a_id, void.node_b_id], 
                graph, target_year, 0.5, void.pressure_gradient
            )
            
            vnode = VirtualNode(
                id="virtual_void_" + str(uuid.uuid4())[:8],
                description=desc_data['description'],
                predicted_domain=void.bridge_domain,
                predicted_entity_type="predicted_bridge_technology",
                predicted_embedding=bridge_emb,
                predicted_year=target_year,
                predicted_readiness=void.void_severity,
                source_trajectory=[void.node_a_id, void.node_b_id],
                trajectory_confidence=void.void_severity,
                generation_method="void_gradient"
            )
            vnode.__dict__['physical_substrate'] = desc_data.get('physical_substrate', {})
            new_nodes.append(vnode)
        return new_nodes

    def _project_centroid(self, centroid_obj: "StabilizedCentroid",
                           delta_epochs: float) -> Optional[np.ndarray]:
        """
        Project centroid forward by delta_epochs using velocity and the
        last two centroid positions to estimate acceleration.
        Second-order (quadratic) extrapolation; result is renormalised to
        preserve the original embedding magnitude.
        """
        if len(centroid_obj._history) < 2:
            return None
        h = list(centroid_obj._history)
        velocity     = h[-1] - h[-2]
        acceleration = np.zeros_like(velocity)
        if len(h) >= 3:
            v2 = h[-1] - h[-2]
            v1 = h[-2] - h[-3]
            acceleration = v2 - v1

        projected = (centroid_obj._centroid
                     + velocity * delta_epochs
                     + 0.5 * acceleration * delta_epochs ** 2)
        norm = np.linalg.norm(projected)
        if norm > 1e-8:
            projected = projected / norm * np.linalg.norm(centroid_obj._centroid)
        return projected

    def _optimal_bridge_embedding(self, emb_a: np.ndarray,
                                   emb_b: np.ndarray,
                                   void: "StructuralVoid") -> np.ndarray:
        """
        Find the embedding that maximises semantic pull to BOTH A and B.
        Gradient of (sim(bridge,A) + sim(bridge,B)) w.r.t. bridge is
        proportional to A_norm + B_norm, so the optimal direction is their sum.
        Result is scaled to the mean magnitude of A and B.
        """
        a_norm = emb_a / (np.linalg.norm(emb_a) + 1e-12)
        b_norm = emb_b / (np.linalg.norm(emb_b) + 1e-12)
        optimal_direction = a_norm + b_norm
        norm = np.linalg.norm(optimal_direction)
        if norm < 1e-8:
            return (emb_a + emb_b) / 2.0
        mean_mag = (np.linalg.norm(emb_a) + np.linalg.norm(emb_b)) / 2.0
        return (optimal_direction / norm) * mean_mag

    def _find_trajectory_nodes(self, domain: str,
                                graph: "RuntimeGraph",
                                n: int = 5) -> List[str]:
        """Return the n most recent non-zone nodes in the given domain."""
        nodes = [
            (nid, nd) for nid, nd in graph.nodes.items()
            if nd.domain == domain and not nd.is_temporal_zone
        ]
        nodes.sort(key=lambda x: x[1].timestamp, reverse=True)
        return [nid for nid, _ in nodes[:n]]

    def _generate_strategic_description(self, embedding, domain, source_ids, graph, 
                                        projected_year, velocity, void_pressure) -> dict:
        """
        Requests from LLM not only a description, but also a physical profile of the future node.
        Returns a dict with at minimum {"description": str} and optionally
        {"physical_substrate": {...}, "entity_type": str}.
        """
        source_texts = [graph.get_node(nid).text[:50] for nid in source_ids if graph.get_node(nid)]
        
        if self.llm:
            prompt = f"""
            Predict the technology at the frontier of {domain} (~{projected_year}).
            Based on: {source_texts}
            
            Return JSON:
            {{
              "description": "functional description",
              "entity_type": "type",
              "physical_substrate": {{
                  "resource_intensity": 0.0-1.0,      
                  "scale_feasibility": 0.0-1.0,       
                  "constraint_proximity": 0.0-1.0,    
                  "reversibility": 0.0-1.0,           
                  "parallel_deployability": 0.0-1.0,  
                  "environmental_coupling": 0.0-1.0,  
                  "cross_scale_stability": 0.0-1.0,   
                  "synthesis_complexity": 0.0-1.0,    
                  "longevity": 0.0-1.0,               
                  "cascadability": 0.0-1.0 
              }}
            }}
            """
            try:
                raw = self.llm.complete(prompt)
                return json.loads(raw)
            except: pass
            
        return {
            "description": f"Predicted {domain} breakthrough",
            "physical_substrate": {ax: 0.5 for ax in PHYSICAL_AXIS_ORDER}
        }

    # Backward-compatible alias: original oracle1_dynamics.py used _generate_description (returns str).
    # Combined uses the extended _generate_strategic_description (returns dict).
    def _generate_description(self, embedding, domain, source_ids, graph,
                               projected_year, velocity, void_pressure) -> str:
        """Backward-compatible alias; returns only the description string."""
        data = self._generate_strategic_description(
            embedding, domain, source_ids, graph, projected_year, velocity, void_pressure)
        return data.get("description", f"Predicted {domain} breakthrough (~{projected_year})")

    def get_virtual_nodes_as_graph_entries(self) -> Tuple[List[KnowledgeNode], List[KnowledgeEdge]]:
        """
        Returns nodes TOGETHER with their connecting edges.
        """
        nodes = []
        edges = []
        for vn in self.virtual_nodes:
            # 1. Build KnowledgeNode from VirtualNode
            node = KnowledgeNode(
                id=vn.id,
                text=vn.description[:80],
                node_type="virtual_future",
                domain=vn.predicted_domain,
                entity_type=vn.predicted_entity_type,
                timestamp=float(vn.predicted_year - 1970) * 31536000,
                readiness_score=vn.predicted_readiness * 10
            )
            node.embedding = vn.predicted_embedding
            # Transferring physics
            node.__dict__['physical_scores'] = vn.__dict__.get('physical_substrate', {})
            nodes.append(node)

            # 2. Create edges from parents to a virtual node
            for parent_id in vn.source_trajectory:
                edge = KnowledgeEdge(
                    id=str(uuid.uuid4())[:8],
                    source_id=parent_id,
                    target_id=vn.id,
                    relationship_type="projection_of_trajectory",
                    confidence=vn.trajectory_confidence,
                    semantic_similarity=CONFIG["synth_edge_default_sim"]  # base value for projection
                )
                edge.compute_total_weight()
                edges.append(edge)
                
        return nodes, edges


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — STRUCTURAL CONFLICT METER
# ══════════════════════════════════════════════════════════════════════════════

class StructuralConflictMeter:
    def __init__(self):
        self._cache: Dict[str, float] = {}

    def compute_all(self, graph: RuntimeGraph):
        for nid in graph.nodes:
            node = graph.get_node(nid)
            if not node or node.is_temporal_zone:
                self._cache[nid] = 0.0; continue
            self._cache[nid] = self._conflict_for_node(nid, graph)

    def _conflict_for_node(self, nid: str, graph: RuntimeGraph) -> float:
        incoming = [edge for (src, tgt), edge in graph.edges.items()
                    if tgt == nid and not edge.is_containment_edge]
        if len(incoming) < 2: return 0.0
        weights = np.array([e.total_weight for e in incoming])
        norm_var = min(1.0, float(np.var(weights)) / CONFIG["conflict_var_norm_denom"])
        n_edges = len(incoming)
        n_pairs = n_edges * (n_edges - 1) / 2
        if n_pairs < 1: return norm_var * 0.5
        total_inconsistency = 0.0
        for i in range(n_edges):
            for j in range(i + 1, n_edges):
                for comp in EDGE_WEIGHT_COMPONENTS:
                    total_inconsistency += abs(getattr(incoming[i], comp, 0.0) -
                                               getattr(incoming[j], comp, 0.0))
        norm_inconsistency = total_inconsistency / (n_pairs * N_EDGE_COMPONENTS)
        return min(1.0, CONFIG["conflict_norm_var_weight"] * norm_var + CONFIG["conflict_inconsistency_weight"] * norm_inconsistency)

    def get_conflict(self, nid: str) -> float:
        return self._cache.get(nid, 0.0)

    def top_conflict_nodes(self, n: int = 10) -> List[Tuple[str, float]]:
        return sorted(self._cache.items(), key=lambda x: x[1], reverse=True)[:n]


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — PHANTOM PRESSURE LAYER
# ══════════════════════════════════════════════════════════════════════════════

def apply_phantom_pressure(phantom_weight: float) -> float:
    if phantom_weight <= 0: return 0.0
    low  = CONFIG["phantom_pressure_low_bound"]
    if phantom_weight < low:
        raw = (phantom_weight / low) ** 2 * CONFIG["phantom_pressure_low_scale"]
    else:
        raw = phantom_weight * CONFIG["phantom_pressure_high_scale"]
    return min(CONFIG["phantom_pressure_max"], raw)


def update_phantom_weights(graph: RuntimeGraph, llm_client=None):
    for nid, node in graph.nodes.items():
        if node.is_temporal_zone: continue
        etype = getattr(node, "entity_type", "").lower()
        new_pw = getattr(node, "phantom_weight", 0.0)
        if any(k in etype for k in ("cultural_phantom", "desired_effect",
                                     "collective_desire", "cultural_desire")):
            soc = getattr(node, "social_perception_score", 0.6)
            new_pw = max(new_pw, float(np.clip(soc / 10.0, 0.0, 1.0)))
        if any(k in etype for k in ("fiction", "media", "mythology", "folklore", "narrative")):
            new_pw = max(new_pw, CONFIG["phantom_fiction_default_weight"])
        incoming_soc = [edge.social_correlation for (src, tgt), edge in graph.edges.items()
                        if tgt == nid and not edge.is_containment_edge]
        if incoming_soc and np.mean(incoming_soc) > CONFIG["phantom_social_boost_threshold"]:
            new_pw = min(1.0, new_pw + CONFIG["phantom_social_boost_amount"])
        node.phantom_weight = float(np.clip(new_pw, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — CROSS-DOMAIN ISOMORPHISM AMPLIFIER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class IsomorphicPair:
    node_a_id: str
    node_b_id: str
    domain_a: str
    domain_b: str
    structural_similarity: float
    edge_id: Optional[str] = None
    amplification_applied: bool = False


class CrossDomainIsomorphismAmplifier:
    ISOMORPHISM_THRESHOLD = CONFIG["isomorphism_threshold"]
    AMPLIFICATION_FACTOR  = CONFIG["isomorphism_amplification_factor"]
    TOPOLOGY_FEATURE_DIM  = 8

    def __init__(self):
        self.isomorphic_pairs: List[IsomorphicPair] = []
        self._topology_cache: Dict[str, np.ndarray] = {}

    def compute_topology_features(self, nid: str, graph: RuntimeGraph) -> np.ndarray:
        if nid in self._topology_cache: return self._topology_cache[nid]
        k2 = graph.k_hop_neighborhood([nid], k=2)
        out_edges = [(s, t, e) for (s, t), e in graph.edges.items()
                     if s == nid and not e.is_containment_edge]
        in_edges  = [(s, t, e) for (s, t), e in graph.edges.items()
                     if t == nid and not e.is_containment_edge]
        k2_edges  = [(s, t, e) for (s, t), e in graph.edges.items()
                     if s in k2 and t in k2 and not e.is_containment_edge]
        degree       = len(out_edges) + len(in_edges)
        weighted_deg = sum(e.total_weight for _, _, e in out_edges + in_edges)
        k2_n         = len(k2)
        k2_density   = len(k2_edges) / max(1, k2_n * (k2_n - 1) // 2)
        out_weights  = [e.total_weight for _, _, e in out_edges] or [0.0]
        w_var        = float(np.var(out_weights))
        in_out_ratio = len(in_edges) / (degree + 1)
        max_w        = max(out_weights)
        total_w = sum(out_weights) + 1e-8
        probs   = [w / total_w for w in out_weights]
        w_entr  = float(-np.sum([p * np.log(p + 1e-8) for p in probs]))
        w_entr  = min(1.0, w_entr / math.log(max(2, len(out_weights))))
        lim_dens = (sum(1 for _, _, e in out_edges + in_edges if e.limitation_resolution > 0.5)
                    / max(1, degree))
        feat = np.array([
            min(1.0, degree / 20.0), min(1.0, weighted_deg / 10.0),
            min(1.0, k2_density), min(1.0, w_var / 0.25),
            in_out_ratio, max_w, w_entr, lim_dens], dtype=np.float32)
        self._topology_cache[nid] = feat
        return feat

    def compute_structural_similarity(self, nid_a, nid_b, graph) -> float:
        fa = self.compute_topology_features(nid_a, graph)
        fb = self.compute_topology_features(nid_b, graph)
        return float(np.dot(fa / (np.linalg.norm(fa) + 1e-8), fb / (np.linalg.norm(fb) + 1e-8)))

    def amplify_cross_domain_edges(self, graph: RuntimeGraph) -> int:
        self._topology_cache.clear()
        n_amplified = 0
        for (src, tgt), edge in graph.edges.items():
            if edge.is_containment_edge: continue
            rel = getattr(edge, "relation_type", "") or getattr(edge, "relationship_type", "")
            if rel != "cross_domain": continue
            src_node = graph.get_node(src); tgt_node = graph.get_node(tgt)
            if not src_node or not tgt_node: continue
            if src_node.domain == tgt_node.domain: continue
            sim = self.compute_structural_similarity(src, tgt, graph)
            if sim > self.ISOMORPHISM_THRESHOLD:
                edge.total_weight = min(1.0, edge.total_weight * self.AMPLIFICATION_FACTOR)
                self.isomorphic_pairs.append(IsomorphicPair(
                    node_a_id=src, node_b_id=tgt, domain_a=src_node.domain,
                    domain_b=tgt_node.domain, structural_similarity=sim,
                    edge_id=edge.id, amplification_applied=True))
                n_amplified += 1
        return n_amplified

    def detect_latent_isomorphisms(self, graph: RuntimeGraph, top_n: int = 20) -> List[IsomorphicPair]:
        self._topology_cache.clear()
        existing_edges = {(s, t) for (s, t) in graph.edges}
        domains = list(graph.domain_clusters.keys())
        candidates: List[Tuple[str, str, float]] = []
        for i, d_a in enumerate(domains):
            for d_b in domains[i + 1:]:
                if d_a == d_b: continue
                reps_a = graph.domain_clusters.get(d_a, [])[:3]
                reps_b = graph.domain_clusters.get(d_b, [])[:3]
                for na in reps_a:
                    for nb in reps_b:
                        if (na, nb) in existing_edges or (nb, na) in existing_edges: continue
                        sim = self.compute_structural_similarity(na, nb, graph)
                        if sim > self.ISOMORPHISM_THRESHOLD:
                            candidates.append((na, nb, sim))
        candidates.sort(key=lambda x: x[2], reverse=True)
        result = []
        for na, nb, sim in candidates[:top_n]:
            node_a = graph.get_node(na); node_b = graph.get_node(nb)
            if node_a and node_b:
                result.append(IsomorphicPair(
                    node_a_id=na, node_b_id=nb, domain_a=node_a.domain,
                    domain_b=node_b.domain, structural_similarity=sim))
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — ANOMALY ACCUMULATOR
# ══════════════════════════════════════════════════════════════════════════════

class AnomalyAccumulator:
    WINDOW_SIZE = CONFIG["anomaly_window_size"]
    EMA_ALPHA   = CONFIG["anomaly_ema_alpha"]

    def __init__(self):
        self._histories: Dict[str, Deque[float]] = {}
        self._densities: Dict[str, float]        = {}

    def push(self, nid: str, anomaly_value: float):
        anomaly_value = float(np.clip(anomaly_value, 0.0, 1.0))
        if nid not in self._histories:
            self._histories[nid] = deque(maxlen=self.WINDOW_SIZE)
            self._densities[nid] = 0.0
        self._histories[nid].append(anomaly_value)
        self._densities[nid] = (1 - self.EMA_ALPHA) * self._densities[nid] + self.EMA_ALPHA * anomaly_value

    def get_density(self, nid: str) -> float:
        return self._densities.get(nid, 0.0)

    def get_history(self, nid: str) -> List[float]:
        """Return the raw anomaly measurement history for a node."""
        return list(self._histories.get(nid, []))

    def measure_and_push(self, nid: str, graph: RuntimeGraph,
                          domain_degree_stats: Dict[str, Tuple[float, float]]) -> float:
        node = graph.get_node(nid)
        if not node or node.is_temporal_zone: return 0.0
        anomaly = 0.0
        incoming = [(s, t, e) for (s, t), e in graph.edges.items()
                    if t == nid and not e.is_containment_edge]
        if incoming:
            invest     = np.mean([e.investment_correlation for _, _, e in incoming])
            resolution = np.mean([e.limitation_resolution  for _, _, e in incoming])
            if invest > CONFIG["anomaly_invest_high"] and resolution < CONFIG["anomaly_resolution_low"]:
                anomaly += CONFIG["anomaly_invest_bonus"]
        degree = sum(1 for (s, t) in graph.edges if s == nid or t == nid)
        if node.domain in domain_degree_stats:
            mu, sigma = domain_degree_stats[node.domain]
            if sigma > 0: anomaly += min(CONFIG["anomaly_degree_cap"],
                                         abs(degree - mu) / sigma * CONFIG["anomaly_degree_scale"])
        if incoming and len(incoming) >= 3:
            sims = [e.semantic_similarity for _, _, e in incoming]
            if float(np.var(sims)) > CONFIG["anomaly_variance_threshold"]:
                anomaly += CONFIG["anomaly_variance_bonus"]
        anomaly = min(1.0, anomaly)
        self.push(nid, anomaly)
        return anomaly

    def domain_degree_statistics(self, graph: RuntimeGraph) -> Dict[str, Tuple[float, float]]:
        domain_degrees: Dict[str, List[int]] = {}
        for nid, node in graph.nodes.items():
            if node.is_temporal_zone: continue
            degree = sum(1 for (s, t) in graph.edges if s == nid or t == nid)
            domain_degrees.setdefault(node.domain, []).append(degree)
        return {domain: (float(np.mean(degs)), float(np.std(degs)) + 1e-6)
                for domain, degs in domain_degrees.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — CONNECTIVITY ENTROPY CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

class ConnectivityEntropyCalculator:
    def __init__(self):
        self._cache: Dict[str, float] = {}

    def compute_all(self, graph: RuntimeGraph):
        for nid in graph.nodes:
            self._cache[nid] = self._entropy_for_node(nid, graph)

    def _entropy_for_node(self, nid: str, graph: RuntimeGraph) -> float:
        out_edges = [e for (s, t), e in graph.edges.items()
                     if s == nid and not e.is_containment_edge]
        if not out_edges: return 0.0
        weights = [max(0.0, e.total_weight) for e in out_edges]
        total = sum(weights)
        if total < 1e-8: return 0.0
        probs = [w / total for w in weights]
        raw_entropy = -sum(p * math.log(p + 1e-8) for p in probs if p > 0)
        k = len(probs)
        max_entropy = math.log(k) if k > 1 else 1.0
        return float(min(1.0, raw_entropy / max_entropy))

    def get(self, nid: str) -> float:
        return self._cache.get(nid, 0.0)

    def entropy_gradient(self, nid: str, prev_entropy: float) -> float:
        return self._cache.get(nid, 0.0) - prev_entropy

    def brittle_nodes(self, threshold: float = CONFIG["connectivity_entropy_brittle_threshold"]) -> List[str]:
        return [nid for nid, h in self._cache.items() if h < threshold]

    def diffuse_nodes(self, threshold: float = CONFIG["connectivity_entropy_diffuse_threshold"]) -> List[str]:
        return [nid for nid, h in self._cache.items() if h > threshold]


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — ONTOLOGICAL TENSION FIELD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeTensionState:
    node_id: str
    timestamp: float
    upstream_pressure:   float = 0.0
    inhibitory_drag:     float = 0.0
    connectivity_entropy: float = 0.0
    anomaly_density:     float = 0.0
    phantom_pressure:    float = 0.0
    structural_conflict: float = 0.0
    tension:             float = 0.0
    alert_level: str = "none"

    def attention_multiplier(self, gamma: float = ATTENTION_GAMMA) -> float:
        if self.tension < 1e-6: return 1.0
        return float(min(10.0, max(1.0, self.tension ** gamma)))


def compute_ontological_tension(node, upstream_pressure, inhibitory_drag,
                                 connectivity_entropy, anomaly_density,
                                 phantom_pressure, structural_conflict=0.0) -> float:
    tw = CONFIG["tension_weights"]
    base = (tw["upstream_pressure"]    * upstream_pressure    +
            tw["inhibitory_drag"]      * inhibitory_drag      +
            tw["connectivity_entropy"] * connectivity_entropy +
            tw["anomaly_density"]      * anomaly_density      +
            tw["phantom_pressure"]     * phantom_pressure)
    conflict_bonus = min(CONFIG["tension_conflict_bonus_cap"],
                         structural_conflict * CONFIG["tension_conflict_bonus_scale"])
    tension = min(1.0, base + conflict_bonus)
    if tension > COLLAPSE_THRESHOLD:
        tension = tension ** 2
    return min(1.0, tension)


class OntologicalTensionField:
    def compute(self, graph, pressure_field, conflict_meter, anomaly_acc,
                entropy_calc, timestamp) -> Dict[str, NodeTensionState]:
        states: Dict[str, NodeTensionState] = {}
        for nid, node in graph.nodes.items():
            if node.is_temporal_zone: continue
            up_raw  = getattr(node, "upstream_pressure", 0.0)
            up_norm = float(np.tanh(max(0.0, up_raw) / CONFIG["tension_pressure_zone_scale"]))
            inh_drag = self._compute_inhibitory_drag(nid, graph)
            c_entropy = entropy_calc.get(nid)
            anomaly   = anomaly_acc.get_density(nid)
            phantom_w = getattr(node, "phantom_weight", 0.0)
            phantom_p = apply_phantom_pressure(phantom_w)
            conflict  = conflict_meter.get_conflict(nid)
            tension   = compute_ontological_tension(node, up_norm, inh_drag,
                                                     c_entropy, anomaly, phantom_p, conflict)
            state = NodeTensionState(
                node_id=nid, timestamp=timestamp, upstream_pressure=up_norm,
                inhibitory_drag=inh_drag, connectivity_entropy=c_entropy,
                anomaly_density=anomaly, phantom_pressure=phantom_p,
                structural_conflict=conflict, tension=tension,
                alert_level=self._alert_level(tension))
            states[nid] = state
        return states

    def _compute_inhibitory_drag(self, nid, graph) -> float:
        total = 0.0
        for (src, tgt), edge in graph.edges.items():
            if tgt != nid or edge.is_containment_edge: continue
            inh = getattr(edge, "inhibitory_force", 0.0)
            if inh <= 0: continue
            attacker = graph.get_node(src)
            sv = getattr(attacker, "strategic_value", 5.0) / 10.0 if attacker else 0.5
            total += inh * sv
        return min(1.0, total)

    @staticmethod
    def _alert_level(tension: float) -> str:
        thr = CONFIG["alert_thresholds"]
        if tension < thr["none"]:     return "none"
        elif tension < thr["elevated"]: return "elevated"
        elif tension < thr["critical"]: return "critical"
        else: return "rupture"


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — VIRTUAL NODE FACTORY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VirtualNodeSpec:
    id: str
    source_node_id: str
    source_tension: float
    source_conflict: float
    domain: str
    functional_requirements: Dict[str, Any]
    must_reduce_drag:         bool = False
    must_preserve_invariants: bool = True
    must_resolve_anomaly:     bool = False
    must_bridge_domains:      bool = False
    must_reduce_entropy:      bool = False
    projection_edge_type: str = "projection"
    confidence: float = 0.0
    timestamp:  float = 0.0


class VirtualNodeFactory:
    def __init__(self):
        self.virtual_specs: List[VirtualNodeSpec] = []

    def generate_from_tension(self, tension_states: Dict[str, NodeTensionState],
                               graph: RuntimeGraph, timestamp: float) -> List[VirtualNodeSpec]:
        new_specs = []
        for nid, state in tension_states.items():
            if state.tension < VIRTUAL_NODE_TENSION_THRESHOLD: continue
            if any(s.source_node_id == nid for s in self.virtual_specs): continue
            node = graph.get_node(nid)
            if not node: continue
            spec = self._build_spec(node, state, timestamp)
            new_specs.append(spec)
            self.virtual_specs.append(spec)
        return new_specs

    def _build_spec(self, node, state, timestamp) -> VirtualNodeSpec:
        reqs: Dict[str, Any] = {}
        must_reduce_drag     = state.inhibitory_drag    > CONFIG["vfactory_drag_threshold"]
        must_resolve_anomaly = state.anomaly_density    > CONFIG["vfactory_anomaly_threshold"]
        must_bridge_domains  = state.structural_conflict > CONFIG["vfactory_conflict_threshold"]
        must_reduce_entropy  = state.connectivity_entropy < CONFIG["vfactory_entropy_threshold"]
        if must_reduce_drag:   reqs["drag_target"] = "bypass or substitute incumbent"
        if must_resolve_anomaly: reqs["consistency_requirement"] = "must resolve citation conflicts"
        if must_bridge_domains: reqs["cross_domain_bridge"] = True
        if must_reduce_entropy: reqs["diversification"] = f"must connect to at least {CONFIG['vfactory_entropy_min_domains']} separate domains"
        reqs["must_preserve_core_invariants"] = True
        confidence = min(0.95, (state.tension - VIRTUAL_NODE_TENSION_THRESHOLD)
                          / (1.0 - VIRTUAL_NODE_TENSION_THRESHOLD))
        return VirtualNodeSpec(
            id="vspec_" + str(uuid.uuid4())[:8], source_node_id=node.id,
            source_tension=state.tension, source_conflict=state.structural_conflict,
            domain=node.domain, functional_requirements=reqs,
            must_reduce_drag=must_reduce_drag, must_preserve_invariants=True,
            must_resolve_anomaly=must_resolve_anomaly,
            must_bridge_domains=must_bridge_domains,
            must_reduce_entropy=must_reduce_entropy,
            confidence=confidence, timestamp=timestamp)

    def as_knowledge_nodes(self, graph: RuntimeGraph) -> List[KnowledgeNode]:
        result = []
        for spec in self.virtual_specs:
            source = graph.get_node(spec.source_node_id)
            vnode = KnowledgeNode(
                id=spec.id,
                text=f"[VIRTUAL] Required in {spec.domain}: tension={spec.source_tension:.2f}",
                node_type="virtual_required", domain=spec.domain,
                entity_type="ontological_requirement", timestamp=spec.timestamp)
            if source and source.embedding is not None:
                vnode.embedding = source.embedding.copy()
            result.append(vnode)
        return result


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY — PHASE TRANSITION DETECTOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhaseAlert:
    node_id:    str
    alert_type: str
    tension:    float
    pressure:   float
    conflict:   float
    entropy:    float
    timestamp:  float
    message:    str


class PhaseTransitionDetector:
    def __init__(self):
        self.alerts: List[PhaseAlert] = []
        self._fired_for: Set[str] = set()

    def scan(self, tension_states, conflict_meter, entropy_calc, graph, timestamp):
        new_alerts: List[PhaseAlert] = []
        for nid, state in tension_states.items():
            pressure = state.upstream_pressure + state.inhibitory_drag * 0.5
            conflict = state.structural_conflict
            entropy  = state.connectivity_entropy
            fired_type = self._classify(pressure, conflict, entropy)
            if fired_type is None: continue
            if nid in self._fired_for:
                existing_type = next((a.alert_type for a in reversed(self.alerts)
                                      if a.node_id == nid), None)
                if existing_type == fired_type: continue
            node = graph.get_node(nid)
            name = node.text[:40] if node else nid
            domain = node.domain if node else "unknown"
            msg = self._build_message(name, domain, fired_type, state)
            alert = PhaseAlert(node_id=nid, alert_type=fired_type, tension=state.tension,
                               pressure=pressure, conflict=conflict, entropy=entropy,
                               timestamp=timestamp, message=msg)
            new_alerts.append(alert)
            self.alerts.append(alert)
            self._fired_for.add(nid)
        if new_alerts:
            logger.warning(f"[PhaseTransition] {len(new_alerts)} new alerts: "
                           + ", ".join(f"{a.node_id[:8]}:{a.alert_type}" for a in new_alerts[:5]))
        return new_alerts

    def _classify(self, pressure, conflict, entropy) -> Optional[str]:
        for alert_type in [ALERT_ONTOLOGICAL_RUPTURE, ALERT_DOMAIN_TRANSPLANT,
                           ALERT_ANOMALY_SATURATION, ALERT_VOID_COLLAPSE, ALERT_OVERLOAD]:
            θp, θc, θe = PHASE_THRESHOLDS[alert_type]
            if pressure >= θp and conflict >= θc and entropy >= θe:
                return alert_type
        return None

    def _build_message(self, name, domain, alert_type, state) -> str:
        msgs = {
            ALERT_OVERLOAD: f"'{name}' [{domain}]: pressure overload T={state.tension:.2f}.",
            ALERT_VOID_COLLAPSE: f"'{name}' [{domain}]: void collapse imminent.",
            ALERT_DOMAIN_TRANSPLANT: f"'{name}' [{domain}]: cross-domain transplantation candidate.",
            ALERT_ANOMALY_SATURATION: f"'{name}' [{domain}]: anomaly saturation.",
            ALERT_ONTOLOGICAL_RUPTURE: f"*** ONTOLOGICAL RUPTURE *** '{name}' [{domain}]: T={state.tension:.3f}.",
        }
        return msgs.get(alert_type, f"Alert: {alert_type} at '{name}'")

    def rupture_nodes(self) -> List[PhaseAlert]:
        return [a for a in self.alerts if a.alert_type == ALERT_ONTOLOGICAL_RUPTURE]

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for a in self.alerts: counts[a.alert_type] = counts.get(a.alert_type, 0) + 1
        return counts


# TODO #18 (found in TODO_rev3's dead-code audit): MotherAnnotator — the
# original single-pass Mother-LLM annotator this file's docstring and the
# comment below both reference by name — was deleted here. Its own
# docstring self-flagged it `.. deprecated::` (12,000-char truncation, no
# response validation) with zero call sites anywhere in the codebase,
# fully superseded by the IngestionPipeline below (chunking, 3-pass
# extraction, validation, cross-document linking). Not a bug — the
# deprecation was already complete; this just removes the now-inert code.

__all__ = ['ConvergenceTimeSeries', 'ConvergenceAccelerometer', 'DualObjectiveHead', 'StabilizedCentroid', 'StabilizedCentroidTracker', 'StructuralVoid', 'ObligatoryBridgeDetector', 'PhantomLagModel', 'PhantomPhaseAligner', 'VirtualNode', 'VirtualNodeSynthesizer', 'StructuralConflictMeter', 'apply_phantom_pressure', 'update_phantom_weights', 'IsomorphicPair', 'CrossDomainIsomorphismAmplifier', 'AnomalyAccumulator', 'ConnectivityEntropyCalculator', 'NodeTensionState', 'compute_ontological_tension', 'OntologicalTensionField', 'VirtualNodeSpec', 'VirtualNodeFactory', 'PhaseAlert', 'PhaseTransitionDetector']
