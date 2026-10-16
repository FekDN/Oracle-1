# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Intervention definitions, simulation, and optimization."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, V6_READINESS_WEIGHTS
from .entities import KnowledgeNode
from .graph import RuntimeGraph
from .graph_dynamics import ConvergencePressureField
from .topology_dynamics import (
    AnomalyAccumulator,
    ConnectivityEntropyCalculator,
    OntologicalTensionField,
    PhantomLagModel,
    StructuralConflictMeter,
)

# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  INTERVENTION OPTIMIZER MODULE
#  Встроен из intervention_optimizer.py
# ══════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("oracle1.intervention")


# ══════════════════════════════════════════════════════════════════════════════
#  ENUMS & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

class InterventionType(Enum):
    """Типы вмешательств, понимаемых оптимизатором."""
    INVESTMENT_INJECTION    = auto()  # +$X млн инвестиций
    REGULATORY_REMOVAL      = auto()  # снятие регуляторного барьера (inhibitory_force ↓)
    BREAKTHROUGH_ADJACENT   = auto()  # прорыв в смежной технологии (scientific_score соседей ↑)
    TALENT_INFLUX           = auto()  # приток кадров (group_size_score ↑, social_gravity ↑)
    GEOPOLITICAL_ACCEL      = auto()  # геополитический катализатор (zone_multiplier ↑)
    CUSTOM_PARAMS           = auto()  # произвольный набор полей KnowledgeNode


# Поля KnowledgeNode, которые NodePatcher умеет изменять по типу вмешательства.
# Значение — функция (node_copy, params) → None, изменяет node_copy in-place.
_PATCH_REGISTRY: Dict[InterventionType, Any] = {}


# ══════════════════════════════════════════════════════════════════════════════
#  DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class InterventionSpec:
    """Описание одного вмешательства.

    Параметры зависят от intervention_type:

    INVESTMENT_INJECTION
        amount_usd   : float  — объём инвестиций в USD (например 2e8)
        year         : int    — целевой год вмешательства (влияет на timestamp)

    REGULATORY_REMOVAL
        inhibitory_scale : float  — на сколько снизить inhibitory_force входящих рёбер
                                    (0.0 = полностью снять, 0.5 = вдвое меньше)
        edge_ids         : list[str] | None — конкретные id рёбер; если None — все
                                    inhibitory рёбра, входящие в target_node_id

    BREAKTHROUGH_ADJACENT
        scientific_boost : float  — прирост scientific_score для 1-hop соседей
        only_domain      : str | None — ограничить соседей доменом

    TALENT_INFLUX
        headcount_multiplier : float  — множитель group_size_score (например 2.0)
        social_boost         : float  — прирост social_gravity

    GEOPOLITICAL_ACCEL
        zone_multiplier_add : float  — прибавка к acceleration_multiplier
        duration_years      : float  — через сколько лет эффект затухает

    CUSTOM_PARAMS
        fields : dict[str, float]  — произвольные поля KnowledgeNode → новые значения
                                     (используются как абсолютные, не как дельта)
        delta_fields : dict[str, float]  — произвольные поля → дельта (прибавка)
    """
    target_node_id:    str
    intervention_type: InterventionType
    params:            Dict[str, Any] = field(default_factory=dict)
    label:             str = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.intervention_type.name}@{self.target_node_id[:12]}"


@dataclass
class CascadeEffect:
    """Вторичный эффект на один соседний узел."""
    node_id:            str
    node_text:          str
    relation:           str            # "1-hop" / "2-hop" / ... — wave stops on its own
    delta_readiness:    float = 0.0
    delta_forecast:     float = 0.0
    delta_investment:   float = 0.0
    direction:          str = "neutral"  # "positive" / "negative" / "neutral"


@dataclass
class BottleneckChange:
    """Изменение статуса боттлнека (появился / снят)."""
    node_id:   str
    node_text: str
    change:    str   # "NEW" — новый боттлнек, "REMOVED" — снят
    reason:    str = ""


@dataclass
class SimulationRun:
    """Результат одной симуляции (один сэмпл Монте-Карло)."""
    run_id:                str
    noise_sigma:           float
    baseline_readiness:    float
    patched_readiness:     float
    baseline_forecast:     float
    patched_forecast:      float
    baseline_tension:      float
    patched_tension:       float
    baseline_lag_years:    float
    patched_lag_years:     float
    cascade_effects:       List[CascadeEffect] = field(default_factory=list)
    bottleneck_changes:    List[BottleneckChange] = field(default_factory=list)

    @property
    def delta_readiness(self) -> float:
        return self.patched_readiness - self.baseline_readiness

    @property
    def delta_forecast(self) -> float:
        return self.patched_forecast - self.baseline_forecast

    @property
    def years_accelerated(self) -> float:
        """Положительное значение = ускорение; отрицательное = замедление."""
        return self.baseline_lag_years - self.patched_lag_years


@dataclass
class InterventionResult:
    """Агрегированный результат N симуляций."""
    spec:               InterventionSpec
    n_simulations:      int
    elapsed_s:          float
    runs:               List[SimulationRun] = field(default_factory=list)

    # ── Агрегаты ──────────────────────────────────────────────────────────────
    mean_delta_readiness:  float = 0.0
    std_delta_readiness:   float = 0.0
    mean_delta_forecast:   float = 0.0
    std_delta_forecast:    float = 0.0
    mean_years_accelerated: float = 0.0
    std_years_accelerated:  float = 0.0
    p10_years_accelerated:  float = 0.0
    p50_years_accelerated:  float = 0.0
    p90_years_accelerated:  float = 0.0

    # Вероятность того, что технология ускорится (years_accelerated > 0)
    prob_acceleration:     float = 0.0

    # Топ-N каскадных эффектов (усреднённых по прогонам)
    top_cascade_effects:   List[CascadeEffect] = field(default_factory=list)
    bottleneck_changes:    List[BottleneckChange] = field(default_factory=list)

    # Рекомендуемая оптимальная комбинация (заполняется OptimalFinder)
    optimal_combination:   Optional[List[InterventionSpec]] = None
    optimal_expected_gain: float = 0.0

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  INTERVENTION OPTIMIZER — {self.spec.label}",
            f"{'='*60}",
            f"  Target node    : {self.spec.target_node_id}",
            f"  Type           : {self.spec.intervention_type.name}",
            f"  Simulations    : {self.n_simulations}  ({self.elapsed_s:.2f}s)",
            f"",
            f"  Δ readiness_score  : {self.mean_delta_readiness:+.4f}  ±{self.std_delta_readiness:.4f}",
            f"  Δ forecast_score   : {self.mean_delta_forecast:+.4f}  ±{self.std_delta_forecast:.4f}",
            f"",
            f"  Years accelerated  : {self.mean_years_accelerated:+.2f} yr  (p10={self.p10_years_accelerated:+.1f}, p50={self.p50_years_accelerated:+.1f}, p90={self.p90_years_accelerated:+.1f})",
            f"  Prob. acceleration : {self.prob_acceleration*100:.1f}%",
        ]
        if self.top_cascade_effects:
            lines += ["", "  Cascade effects (top):"]
            for ce in self.top_cascade_effects[:5]:
                sign = "▲" if ce.direction == "positive" else ("▼" if ce.direction == "negative" else "─")
                lines.append(
                    f"    {sign} [{ce.relation}] {ce.node_text[:45]:<45}  "
                    f"Δread={ce.delta_readiness:+.3f}  Δfcst={ce.delta_forecast:+.3f}"
                )
        if self.bottleneck_changes:
            lines += ["", f"  Bottleneck changes (top-{min(3, len(self.bottleneck_changes))} of {len(self.bottleneck_changes)}):"]
            for bc in self.bottleneck_changes[:3]:
                lines.append(f"    [{bc.change}] {bc.node_text[:50]}  — {bc.reason}")
        if self.optimal_combination:
            lines += ["", f"  Optimal combination  (expected gain {self.optimal_expected_gain:+.2f} yr):"]
            for s in self.optimal_combination:
                lines.append(f"    • {s.label}")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  NODE PATCHER
#  Создаёт временную копию узла с применёнными параметрами вмешательства.
# ══════════════════════════════════════════════════════════════════════════════

class NodePatcher:
    """Применяет InterventionSpec к копии KnowledgeNode.

    Возвращает новый объект, оригинал не трогается.
    Все параметры масштабируются на noise_scale для Монте-Карло.
    """

    # Лог-нормальный масштаб для перевода USD → investment_score (0–10)
    _INVEST_LOG_SCALE = CONFIG.get("pressure_invest_log_scale", 20.0)

    @classmethod
    def patch(cls,
              original: KnowledgeNode,
              spec: InterventionSpec,
              noise_scale: float = 1.0) -> KnowledgeNode:
        """Вернуть патченную копию, не трогая original."""
        node = copy.deepcopy(original)
        t    = spec.intervention_type
        p    = spec.params
        ns   = float(np.clip(noise_scale, 0.0, 10.0))

        if t == InterventionType.INVESTMENT_INJECTION:
            cls._patch_investment(node, p, ns)

        elif t == InterventionType.REGULATORY_REMOVAL:
            cls._patch_regulatory(node, p, ns)

        elif t == InterventionType.BREAKTHROUGH_ADJACENT:
            # Соседи патчатся отдельно в GraphPatcher; здесь — сам узел
            boost = float(p.get("scientific_boost", 1.0)) * ns
            node.scientific_score  = min(10.0, node.scientific_score + boost * 0.3)
            node.readiness_score   = cls._recompute_readiness(node)

        elif t == InterventionType.TALENT_INFLUX:
            mult  = float(p.get("headcount_multiplier", 1.5)) * ns
            soc   = float(p.get("social_boost", 1.0)) * ns
            node.group_size_score  = min(10.0, node.group_size_score * mult)
            node.group_count       = max(1, int(node.group_count * mult))
            node.social_gravity    = min(10.0, node.social_gravity + soc)
            node.social_score      = min(10.0, node.social_score + soc * 0.5)
            node.readiness_score   = cls._recompute_readiness(node)

        elif t == InterventionType.GEOPOLITICAL_ACCEL:
            add = float(p.get("zone_multiplier_add", 1.0)) * ns
            node.acceleration_multiplier = min(5.0, node.acceleration_multiplier + add)
            node.zone_multiplier         = min(5.0, node.zone_multiplier + add * 0.5)
            node.readiness_score         = cls._recompute_readiness(node)

        elif t == InterventionType.CUSTOM_PARAMS:
            for fld, val in p.get("fields", {}).items():
                if hasattr(node, fld):
                    setattr(node, fld, float(val) * ns)
            for fld, delta in p.get("delta_fields", {}).items():
                if hasattr(node, fld):
                    setattr(node, fld, getattr(node, fld) + float(delta) * ns)
            node.readiness_score = cls._recompute_readiness(node)

        # Всегда пересчитываем upstream_pressure и forecast_score
        node.upstream_pressure = max(
            node.upstream_pressure,
            node.readiness_score * node.strategic_value * 0.1,
        )
        node.forecast_score = (
            0.4 * node.readiness_score
            + 0.3 * node.convergence_potential
            + 0.2 * node.scientific_score / 10.0
            + 0.1 * node.maturity_score / 10.0
        )
        return node

    # ── Внутренние патч-функции ──────────────────────────────────────────────

    @classmethod
    def _patch_investment(cls, node: KnowledgeNode,
                          params: Dict[str, Any], ns: float) -> None:
        amount_usd = float(params.get("amount_usd", 1e8)) * ns

        # feasibility_score gates how much of the money actually converts
        # into progress. Without this, a simulation can inject $100T into a
        # node whose synthesis_complexity makes it physically unbuildable
        # (feasibility_score near 0) and still get a large "accelerated by
        # 50 years" result — money cannot buy past a substrate limit that
        # isn't a limit of funding to begin with. Money spent is still
        # recorded honestly below; only its *effect* on pressure/maturity is
        # scaled by feasibility. A node that never ran through
        # PhysicalSubstrateEncoder has no "feasibility" key yet — default to
        # 1.0 (unknown, not "impossible") rather than silently zeroing it.
        feasibility = float(np.clip(node.__dict__.get("feasibility", 1.0), 0.0, 1.0))

        # origin_kind is a separate axis from feasibility: feasibility asks
        # "is this physically possible at all", origin_kind asks "is
        # progress here even a function of money and effort". A
        # combinatorial bottleneck (engineering a known set of pieces
        # together) responds to funding the way real R&D usually does — more
        # money, more parallel attempts, faster convergence. A fundamental
        # bottleneck (a physical constant, material property, or mechanism
        # nature hasn't given up yet) does not schedule itself around a
        # budget; money buys more people looking, not a discovery on demand.
        # Investment can still nudge it (larger search, better instruments),
        # hence a dampening factor rather than another hard gate to zero.
        origin_dampening = (
            CONFIG["intervention_fundamental_investment_dampening"]
            if node.origin_kind == "fundamental" else 1.0
        )

        # 1. Прямое накопление инвестиций
        node.investment_total_usd      += amount_usd
        node.investment_last_round_usd  = amount_usd
        node.investment_rounds         += 1

        # 2. investment_score: логарифмически масштабированный 0–10
        raw_invest_score = math.log1p(node.investment_total_usd) / cls._INVEST_LOG_SCALE * 10.0
        node.investment_score = float(np.clip(raw_invest_score, 0.0, 10.0))

        # 3. upstream_pressure растёт пропорционально вложениям и стратегической ценности,
        #    но эффект гасится feasibility_score — деньги не пробивают физический предел
        pressure_boost = (
            math.log1p(amount_usd / 1e6)          # в млн USD
            * CONFIG.get("pressure_limitation_scale", 2.0)
            * 0.1
            * feasibility
            * origin_dampening
        )
        node.upstream_pressure = min(
            node.upstream_pressure + pressure_boost,
            200.0,
        )

        # 4. maturity_score косвенно растёт (больше денег → больше экспериментов),
        #    тоже гасится feasibility_score и origin_dampening
        node.maturity_score = min(
            10.0,
            node.maturity_score + math.log1p(amount_usd / 1e8) * 0.5 * feasibility * origin_dampening
        )

        # 5. Пересчёт readiness
        node.readiness_score = cls._recompute_readiness(node)

    @classmethod
    def _patch_regulatory(cls, node: KnowledgeNode,
                           params: Dict[str, Any], ns: float) -> None:
        # Снижение ингибиторного давления самого узла
        # (реальные рёбра патчатся в GraphPatcher)
        scale    = float(params.get("inhibitory_scale", 0.0)) * ns
        red_frac = float(np.clip(1.0 - scale, 0.0, 1.0))
        # Отражаем: legal_risk_score и export_control_risk снижаются
        node.legal_risk_score    *= red_frac
        node.export_control_risk *= red_frac
        # strategic_value немного растёт (барьеры сняты — конкуренты слабее)
        node.strategic_value = min(10.0, node.strategic_value * (1.0 + (1.0 - red_frac) * 0.2))
        node.readiness_score = cls._recompute_readiness(node)

    # ── Пересчёт readiness по v6 формуле из CONFIG ──────────────────────────

    @staticmethod
    def _recompute_readiness(node: KnowledgeNode,
                             weights: Optional[Dict] = None) -> float:
        """Пересчитать readiness_score по формуле v6.

        Args:
            node:    узел для пересчёта.
            weights: словарь весов (по умолчанию V6_READINESS_WEIGHTS из CONFIG).
        """
        w = weights or V6_READINESS_WEIGHTS
        score = (
            w["scientific"]  * node.scientific_score  / 10.0
            + w["investment"]  * node.investment_score  / 10.0
            + w["social"]      * node.social_score      / 10.0
            + w["maturity"]    * node.maturity_score    / 10.0
            + w["group_size"]  * min(node.group_count, 100) / 100.0
        )
        return float(np.clip(score, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH PATCHER
#  Временно патчит граф (узел + рёбра), возвращает контекст-менеджер.
# ══════════════════════════════════════════════════════════════════════════════

class GraphPatcher:
    """Atomic swap: заменяет узел в графе патченной копией, затем откатывает.

    Использование:
        with GraphPatcher(graph, spec, patched_node, noise_scale) as gp:
            # граф содержит patched_node вместо оригинала
            # рёбра тоже пропатчены
            ...
        # здесь граф полностью восстановлен
    """

    def __init__(self, graph: RuntimeGraph, spec: InterventionSpec,
                 patched_node: KnowledgeNode, noise_scale: float = 1.0):
        self._graph        = graph
        self._spec         = spec
        self._patched_node = patched_node
        self._noise_scale  = noise_scale
        self._original_node: Optional[KnowledgeNode] = None
        self._edge_backups: Dict[Tuple[str, str], float] = {}  # (src,tgt) → original inhibitory_force
        self._neighbour_backups: Dict[str, Tuple[float, float]] = {}  # node_id → (scientific_score, readiness_score)
        # ONTOLOGICAL_RUPTURE relaxation (see TopologyInstabilityOrchestrator.
        # run_epoch → PhysicalSubstrateEncoder.resolve_rupture_physics) is a
        # *global* side effect: it walks every non-containment edge reachable
        # from the ruptured node and flips rupture_physics_relaxed on all of
        # them, potentially far outside the single patched node and its
        # direct neighbours that the rest of this class tracks. If a what-if
        # simulation ever triggers a rupture while patched, that flag change
        # would otherwise survive __exit__ and permanently contaminate the
        # real graph. Cheap at this graph's scale (one bool per node), so we
        # snapshot it unconditionally rather than only when we can prove a
        # rupture happened during the patch.
        self._rupture_backup: Dict[str, bool] = {}

    def __enter__(self) -> "GraphPatcher":
        nid = self._spec.target_node_id
        self._original_node = self._graph.nodes.get(nid)
        if self._original_node is None:
            raise KeyError(f"[GraphPatcher] Node '{nid}' not found in graph")

        self._rupture_backup = {
            n_id: n.rupture_physics_relaxed for n_id, n in self._graph.nodes.items()
        }

        # 1. Заменяем узел
        self._graph.nodes[nid] = self._patched_node

        # 2. Патч рёбер при REGULATORY_REMOVAL
        if self._spec.intervention_type == InterventionType.REGULATORY_REMOVAL:
            scale    = float(self._spec.params.get("inhibitory_scale", 0.0)) * self._noise_scale
            red_frac = float(np.clip(1.0 - scale, 0.0, 1.0))
            target_edges = self._spec.params.get("edge_ids", None)
            for (src, tgt), edge in self._graph.edges.items():
                if tgt != nid or edge.is_containment_edge:
                    continue
                inh = getattr(edge, "inhibitory_force", 0.0)
                if inh <= 0.0:
                    continue
                if target_edges is not None and edge.id not in target_edges:
                    continue
                self._edge_backups[(src, tgt)] = inh
                edge.inhibitory_force *= red_frac
                edge.compute_total_weight()

        # 3. Патч соседей при BREAKTHROUGH_ADJACENT
        if self._spec.intervention_type == InterventionType.BREAKTHROUGH_ADJACENT:
            boost       = float(self._spec.params.get("scientific_boost", 1.0)) * self._noise_scale
            only_domain = self._spec.params.get("only_domain", None)
            for (src, tgt), edge in self._graph.edges.items():
                if src != nid or edge.is_containment_edge:
                    continue
                nbr = self._graph.nodes.get(tgt)
                if nbr is None:
                    continue
                if only_domain and nbr.domain != only_domain:
                    continue
                self._neighbour_backups[tgt] = (nbr.scientific_score, nbr.readiness_score)
                nbr.scientific_score = min(10.0, nbr.scientific_score + boost)
                nbr.readiness_score  = NodePatcher._recompute_readiness(nbr)

        return self

    def __exit__(self, *_):
        nid = self._spec.target_node_id
        # Восстановить узел
        if self._original_node is not None:
            self._graph.nodes[nid] = self._original_node
        # Восстановить рёбра
        for (src, tgt), orig_inh in self._edge_backups.items():
            edge = self._graph.edges.get((src, tgt))
            if edge is not None:
                edge.inhibitory_force = orig_inh
                edge.compute_total_weight()
        # Восстановить соседей
        for nbr_id, (orig_sci, orig_ready) in self._neighbour_backups.items():
            nbr = self._graph.nodes.get(nbr_id)
            if nbr is not None:
                nbr.scientific_score = orig_sci
                nbr.readiness_score  = orig_ready
        # Восстановить rupture_physics_relaxed по всему графу
        for n_id, was_relaxed in self._rupture_backup.items():
            n = self._graph.nodes.get(n_id)
            if n is not None:
                n.rupture_physics_relaxed = was_relaxed


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS SNAPSHOT
#  Вычисляет набор метрик для узла в текущем состоянии графа.
# ══════════════════════════════════════════════════════════════════════════════

class MetricsSnapshot:
    """Снимает readiness/forecast/tension/lag для одного узла."""

    def __init__(self, system: Dict[str, Any]):
        self._graph    = system["graph"]
        self._pressure = system["pressure_field"]
        self._tension_field  = OntologicalTensionField()
        self._anomaly_acc    = AnomalyAccumulator()
        self._entropy_calc   = ConnectivityEntropyCalculator()

    def capture(self, node_id: str, timestamp: float) -> Dict[str, float]:
        """Вернуть словарь метрик для node_id."""
        node = self._graph.get_node(node_id)
        if node is None:
            return self._zero_metrics()

        # Обновляем давление
        try:
            self._pressure.update(self._graph, timestamp)
        except Exception:
            pass

        # Энтропия и аномалия (быстро, только для локального узла)
        self._entropy_calc.compute_all(self._graph)
        deg_stats = self._anomaly_acc.domain_degree_statistics(self._graph)
        self._anomaly_acc.measure_and_push(node_id, self._graph, deg_stats)

        # Tension для целевого узла
        try:
            conflict_meter = _make_stub_conflict_meter(self._graph)
            states = self._tension_field.compute(
                self._graph, self._pressure,
                conflict_meter, self._anomaly_acc,
                self._entropy_calc, timestamp,
            )
            tension = states.get(node_id, _ZeroTensionState()).tension
        except Exception:
            tension = 0.0

        # Lag оценка через PhantomLagModel (аппроксимация)
        readiness = node.readiness_score
        lag_years = self._estimate_lag(readiness, node)

        return {
            "readiness":  readiness,
            "forecast":   node.forecast_score,
            "tension":    tension,
            "lag_years":  lag_years,
        }

    @staticmethod
    def _estimate_lag(readiness: float, node: KnowledgeNode) -> float:
        """Оценить лаг до коммерческого появления через логнормальную модель."""
        expected_lag = PhantomLagModel.expected_lag()
        low_b  = CONFIG["lag_readiness_low_boundary"]
        high_b = CONFIG["lag_readiness_high_boundary"]
        if readiness < low_b:
            lag = expected_lag
        elif readiness < high_b:
            fraction = (readiness - low_b) / max(high_b - low_b, 1e-6)
            lag = expected_lag * (1.0 - fraction * CONFIG["lag_linear_max_reduction"])
        else:
            lag = max(
                CONFIG["lag_collapse_min_years"],
                expected_lag * CONFIG["lag_collapse_factor"] * (1.1 - readiness),
            )
        # Inhibitory correction — считаем по стратегическим конкурентам домена
        # (упрощённая версия PhantomPhaseAligner._estimate_inhibitory_drag)
        # Не зависит от графа → нет side effects
        return float(max(0.0, lag))

    @staticmethod
    def _zero_metrics() -> Dict[str, float]:
        return {"readiness": 0.0, "forecast": 0.0, "tension": 0.0, "lag_years": 0.0}


# ══════════════════════════════════════════════════════════════════════════════
#  CASCADE ANALYZER
#  Пропагирует эффект вмешательства на 2-hop соседей.
# ══════════════════════════════════════════════════════════════════════════════

class CascadeAnalyzer:
    """Анализирует вторичные/третичные эффекты на соседей целевого узла."""

    def __init__(self, graph: RuntimeGraph):
        self._graph = graph

    def analyze(self, node_id: str, baseline_metrics: Dict[str, float],
                patched_metrics: Dict[str, float],
                min_delta: float = 1e-5, max_hops: int = 20) -> List[CascadeEffect]:
        """
        Вычислить каскадный эффект на соседних узлах.

        Распространяется по графу как рябь по воде / зацепление шестерён
        (см. README, "Implementation Status" и "How Oracle Differs from
        LLMs"): на каждом шаге накопленная проводимость умножается на
        проводимость *конкретного* следующего ребра — а не на фиксированный
        коэффициент, отдельно прописанный для «1-hop» и «2-hop», — и волна
        останавливается сама, там, где перестаёт заметно менять соседей.
        `max_hops` — это только защитный потолок на случай патологического
        графа (циклов с проводимостью ~1.0), а не расчётный радиус: обычно
        произведение проводимостей < 1.0 гасит сигнал заметно раньше него.
        """
        effects: List[CascadeEffect] = []
        delta_r  = patched_metrics["readiness"] - baseline_metrics["readiness"]
        delta_f  = patched_metrics["forecast"]  - baseline_metrics["forecast"]

        if abs(delta_r) < 1e-6 and abs(delta_f) < 1e-6:
            return effects

        # Adjacency with a scalar conductance per edge — the "max" slice of
        # the node × node × relation structure (RuntimeGraph.relation_adjacency),
        # so this is no longer a private, duplicated scan of self._graph.edges.
        adjacency = self._graph.relation_adjacency("max")

        # A fundamental discovery (origin_kind == "fundamental" — a physical
        # constant, material property, or mechanism nature already contains,
        # not an engineered combination) tends to move a whole cluster of
        # otherwise-unrelated technologies at once, not just its immediate,
        # same-domain neighbours: a superconductivity breakthrough touches
        # power grids, transport, computing, and imaging simultaneously. A
        # combinatorial node's effect stays comparatively local. Modelled as
        # a conductance boost specifically on hops that cross a domain
        # boundary, not a blanket boost everywhere — it should travel
        # *between* clusters more easily, not simply travel further.
        source_node = self._graph.get_node(node_id)
        cross_domain_boost = (
            CONFIG["cascade_fundamental_cross_domain_boost"]
            if source_node is not None and source_node.origin_kind == "fundamental"
            else 1.0
        )

        visited  = {node_id}
        frontier: Dict[str, float] = {node_id: 1.0}   # node_id -> cumulative conductance
        magnitude = max(abs(delta_r), abs(delta_f))
        hop = 0

        while frontier and hop < max_hops:
            hop += 1
            relation = f"{hop}-hop"
            next_frontier: Dict[str, float] = {}

            for origin_id, upstream_cond in frontier.items():
                origin_domain = self._graph.get_node(origin_id)
                origin_domain = origin_domain.domain if origin_domain else None
                for nbr_id, edge_cond in adjacency.get(origin_id, ()):
                    if nbr_id in visited:
                        continue
                    if cross_domain_boost != 1.0:
                        nbr_domain = self._graph.get_node(nbr_id)
                        if nbr_domain is not None and nbr_domain.domain != origin_domain:
                            edge_cond = min(1.0, edge_cond * cross_domain_boost)
                    cond = upstream_cond * edge_cond
                    if cond * magnitude < min_delta:
                        continue  # decayed past the point of mattering here
                    if cond > next_frontier.get(nbr_id, 0.0):
                        next_frontier[nbr_id] = cond

            if not next_frontier:
                break  # the wave has died out on its own — this is the point

            for nbr_id, cond in next_frontier.items():
                visited.add(nbr_id)
                nbr = self._graph.get_node(nbr_id)
                if nbr is None or nbr.is_temporal_zone:
                    continue
                d_r = delta_r * cond
                d_f = delta_f * cond
                direction = "positive" if d_r + d_f > 0 else (
                    "negative" if d_r + d_f < 0 else "neutral"
                )
                effects.append(CascadeEffect(
                    node_id        = nbr_id,
                    node_text      = nbr.text[:60],
                    relation       = relation,
                    delta_readiness = round(d_r, 5),
                    delta_forecast  = round(d_f, 5),
                    delta_investment = round(
                        delta_r * cond * nbr.investment_correlation
                        if hasattr(nbr, "investment_correlation") else 0.0, 5
                    ),
                    direction = direction,
                ))

            frontier = next_frontier

        effects.sort(key=lambda e: abs(e.delta_readiness + e.delta_forecast), reverse=True)
        return effects

    def analyze_multi(
        self,
        sources: List[Tuple[str, Dict[str, float], Dict[str, float]]],
        min_delta: float = 1e-5, max_hops: int = 20,
    ) -> List[CascadeEffect]:
        """
        Superpose several simultaneous "what-if" waves and read off where
        they interfere constructively — two individually modest
        interventions whose waves meet at the same downstream node and add
        up to something neither produces alone (a resonance point, an
        "Эврика!"), not just two side-by-side, independently-analysed
        ripples that happen to be reported together.

        `sources`: list of (node_id, baseline_metrics, patched_metrics),
        one per simultaneous intervention — same shapes analyze() takes.

        Each source propagates from its own origin using the same
        self-terminating conductance rule as analyze(). Propagation from one
        source does not block another source from reaching the same node —
        that overlap is exactly what this method exists to surface. Effects
        from all sources are summed (superposed) per node; a CascadeEffect
        reached by more than one source has "(resonance ×N)" appended to its
        relation label and is sorted to the front.
        """
        if not sources:
            return []
        if len(sources) == 1:
            nid, bm, pm = sources[0]
            return self.analyze(nid, bm, pm, min_delta=min_delta, max_hops=max_hops)

        adjacency = self._graph.relation_adjacency("max")
        # node_id -> [delta_r, delta_f, n_sources_reaching_it, max_hop_reached]
        combined: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0, 0, 0])

        for nid, baseline_metrics, patched_metrics in sources:
            delta_r = patched_metrics["readiness"] - baseline_metrics["readiness"]
            delta_f = patched_metrics["forecast"]  - baseline_metrics["forecast"]
            if abs(delta_r) < 1e-6 and abs(delta_f) < 1e-6:
                continue
            magnitude = max(abs(delta_r), abs(delta_f))
            source_node = self._graph.get_node(nid)
            cross_domain_boost = (
                CONFIG["cascade_fundamental_cross_domain_boost"]
                if source_node is not None and source_node.origin_kind == "fundamental"
                else 1.0
            )
            visited: Set[str] = {nid}
            frontier: Dict[str, float] = {nid: 1.0}
            hop = 0
            while frontier and hop < max_hops:
                hop += 1
                next_frontier: Dict[str, float] = {}
                for origin_id, upstream_cond in frontier.items():
                    origin_domain = self._graph.get_node(origin_id)
                    origin_domain = origin_domain.domain if origin_domain else None
                    for nbr_id, edge_cond in adjacency.get(origin_id, ()):
                        if nbr_id in visited:
                            continue
                        if cross_domain_boost != 1.0:
                            nbr_domain = self._graph.get_node(nbr_id)
                            if nbr_domain is not None and nbr_domain.domain != origin_domain:
                                edge_cond = min(1.0, edge_cond * cross_domain_boost)
                        cond = upstream_cond * edge_cond
                        if cond * magnitude < min_delta:
                            continue
                        if cond > next_frontier.get(nbr_id, 0.0):
                            next_frontier[nbr_id] = cond
                if not next_frontier:
                    break
                for nbr_id, cond in next_frontier.items():
                    visited.add(nbr_id)
                    entry = combined[nbr_id]
                    entry[0] += delta_r * cond
                    entry[1] += delta_f * cond
                    entry[2] += 1
                    entry[3] = max(entry[3], hop)
                frontier = next_frontier

        effects: List[CascadeEffect] = []
        for nbr_id, (d_r, d_f, n_sources, hop) in combined.items():
            nbr = self._graph.get_node(nbr_id)
            if nbr is None or nbr.is_temporal_zone:
                continue
            if abs(d_r) < min_delta and abs(d_f) < min_delta:
                continue
            direction = "positive" if d_r + d_f > 0 else (
                "negative" if d_r + d_f < 0 else "neutral"
            )
            relation = f"{hop}-hop"
            if n_sources > 1:
                relation += f" (resonance ×{n_sources})"
            effects.append(CascadeEffect(
                node_id         = nbr_id,
                node_text       = nbr.text[:60],
                relation        = relation,
                delta_readiness = round(d_r, 5),
                delta_forecast  = round(d_f, 5),
                delta_investment = round(
                    d_r * nbr.investment_correlation
                    if hasattr(nbr, "investment_correlation") else 0.0, 5
                ),
                direction = direction,
            ))
        effects.sort(key=lambda e: (
            0 if "resonance" in e.relation else 1,
            -abs(e.delta_readiness + e.delta_forecast),
        ))
        return effects


# ══════════════════════════════════════════════════════════════════════════════
#  BOTTLENECK DETECTOR
#  Сравнивает структурные боттлнеки до и после вмешательства.
# ══════════════════════════════════════════════════════════════════════════════

class BottleneckDetector:
    """Определяет узлы, ставшие / перставшие быть боттлнеком."""

    # Порог: если tension > этого значения → узел считается боттлнеком
    BOTTLENECK_TENSION_THRESHOLD = CONFIG["bottleneck_tension_threshold"]

    def __init__(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField):
        self._graph    = graph
        self._pressure = pressure_field
        self._tension_field = OntologicalTensionField()

    def _snapshot_tensions(self, timestamp: float) -> Dict[str, float]:
        anomaly_acc  = AnomalyAccumulator()
        entropy_calc = ConnectivityEntropyCalculator()
        entropy_calc.compute_all(self._graph)
        deg_stats = anomaly_acc.domain_degree_statistics(self._graph)
        for nid in self._graph.nodes:
            anomaly_acc.measure_and_push(nid, self._graph, deg_stats)
        conflict_meter = _make_stub_conflict_meter(self._graph)
        states = self._tension_field.compute(
            self._graph, self._pressure,
            conflict_meter, anomaly_acc, entropy_calc, timestamp,
        )
        return {nid: s.tension for nid, s in states.items()}

    def detect_changes(self, baseline_tensions: Dict[str, float],
                        patched_tensions: Dict[str, float]) -> List[BottleneckChange]:
        changes: List[BottleneckChange] = []
        t = self.BOTTLENECK_TENSION_THRESHOLD
        all_ids = set(baseline_tensions) | set(patched_tensions)
        for nid in all_ids:
            b = baseline_tensions.get(nid, 0.0)
            p = patched_tensions.get(nid, 0.0)
            was_bottleneck = b > t
            is_bottleneck  = p > t
            node = self._graph.get_node(nid)
            text = node.text[:55] if node else nid
            if not was_bottleneck and is_bottleneck:
                changes.append(BottleneckChange(
                    node_id=nid, node_text=text,
                    change="NEW",
                    reason=f"tension {b:.2f} → {p:.2f} (exceeded {t})",
                ))
            elif was_bottleneck and not is_bottleneck:
                changes.append(BottleneckChange(
                    node_id=nid, node_text=text,
                    change="REMOVED",
                    reason=f"tension {b:.2f} → {p:.2f} (below {t})",
                ))
        return changes


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMAL INTERVENTION FINDER
#  Перебирает комбинации спецификаций и ищет минимальный набор для цели.
# ══════════════════════════════════════════════════════════════════════════════

class OptimalInterventionFinder:
    """Ищет оптимальную комбинацию вмешательств.

    Параметры
    ─────────
    target_years      : желаемое ускорение (лет)
    target_probability: минимальная вероятность достижения
    max_interventions : максимум вмешательств в комбинации
    budget_usd        : максимальный бюджет инвестиций (если задан)

    Алгоритм: жадный беим-поиск (greedy beam search по gain).
    Полный перебор не нужен — каждое вмешательство независимо по первому порядку.
    """

    def __init__(self, optimizer: "InterventionOptimizer"):
        self._opt = optimizer

    def find(self,
             candidates: List[InterventionSpec],
             target_years: float = 3.0,
             target_probability: float = 0.7,
             max_interventions: int = 3,
             budget_usd: Optional[float] = None,
             n_simulations_per_candidate: int = 100,
             beam_width: int = 3,
             ) -> Tuple[List[InterventionSpec], float]:
        """Вернуть (список спецификаций, ожидаемое суммарное ускорение).

        Если цель не достигнута ни одной комбинацией, возвращает наилучшую
        найденную.

        Args:
            candidates:                  список кандидатов для оценки.
            target_years:                желаемое ускорение (лет).
            target_probability:          минимальная вероятность достижения цели.
            max_interventions:           максимальное число вмешательств в комбинации.
            budget_usd:                  ограничение бюджета (None = без ограничения).
            n_simulations_per_candidate: число прогонов на кандидата.
            beam_width:                  ширина пучка beam-search (топ-N на каждом шаге).
                                         1 = чисто жадный (старое поведение).
        """
        # Шаг 1: оценить каждый кандидат по отдельности
        scored: List[Tuple[float, InterventionSpec]] = []
        for spec in candidates:
            try:
                result = self._opt.simulate(spec, n_simulations=n_simulations_per_candidate)
                gain   = result.mean_years_accelerated
                prob   = result.prob_acceleration
                # Штраф за низкую вероятность
                adjusted_gain = gain * min(1.0, prob / max(target_probability, 0.01))
                scored.append((adjusted_gain, spec))
            except Exception as e:
                logger.warning(f"[OptimalFinder] Candidate '{spec.label}' failed: {e}")

        scored.sort(key=lambda x: x[0], reverse=True)

        # Шаг 2: beam-search: на каждом шаге рассматриваем топ-beam_width кандидатов
        # Каждый «луч» — (total_gain, total_usd, [chosen_specs])
        beams: List[Tuple[float, float, List[InterventionSpec]]] = [(0.0, 0.0, [])]

        for step in range(max_interventions):
            next_beams: List[Tuple[float, float, List[InterventionSpec]]] = []
            for beam_gain, beam_usd, beam_chosen in beams:
                for gain, spec in scored[:max(beam_width * 2, len(scored))]:
                    if spec in beam_chosen:
                        continue
                    if budget_usd is not None:
                        cost = float(spec.params.get("amount_usd", 0))
                        if beam_usd + cost > budget_usd:
                            continue
                    else:
                        cost = 0.0
                    # Subadditive gain: каждое следующее вмешательство даёт меньше
                    decay = CONFIG.get("forecast_decay_factor", 0.65) ** (len(beam_chosen) + 1)
                    new_gain = beam_gain + gain * decay
                    next_beams.append((new_gain, beam_usd + cost, beam_chosen + [spec]))

            if not next_beams:
                break
            next_beams.sort(key=lambda x: x[0], reverse=True)
            beams = next_beams[:beam_width]

            # Ранняя остановка, если лучший луч уже достиг цели
            if beams[0][0] >= target_years:
                break

        best_gain, _, chosen = beams[0] if beams else (0.0, 0.0, [])
        total_gain = best_gain

        return chosen, total_gain


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN INTERFACE — InterventionOptimizer
# ══════════════════════════════════════════════════════════════════════════════

class InterventionOptimizer:
    """Главный класс для запуска What-if симуляций.

    Пример
    ──────
    >>> optimizer = InterventionOptimizer(system)
    >>> spec = InterventionSpec(
    ...     target_node_id="theory_1958",
    ...     intervention_type=InterventionType.INVESTMENT_INJECTION,
    ...     params={"amount_usd": 200_000_000},
    ...     label="$200M coherent-light fund",
    ... )
    >>> result = optimizer.simulate(spec, n_simulations=500)
    >>> print(result.summary())
    """

    # Гауссовый шум вокруг magnitude для Монте-Карло
    DEFAULT_NOISE_SIGMA  = CONFIG["intervention_default_noise_sigma"]   # 20%
    DEFAULT_SIMULATIONS  = CONFIG["intervention_default_simulations"]

    def __init__(self, system: Dict[str, Any]):
        self._system  = system
        self._graph   = system["graph"]
        self._pressure: ConvergencePressureField = system["pressure_field"]
        self._metrics = MetricsSnapshot(system)
        self._cascade = CascadeAnalyzer(self._graph)
        self._bneck   = BottleneckDetector(self._graph, self._pressure)
        self._finder  = OptimalInterventionFinder(self)

    # ── Основной вход ────────────────────────────────────────────────────────

    def simulate(self,
                 spec: InterventionSpec,
                 n_simulations: int = DEFAULT_SIMULATIONS,
                 noise_sigma: float = DEFAULT_NOISE_SIGMA,
                 timestamp: Optional[float] = None,
                 random_seed: Optional[int] = None,
                 ) -> InterventionResult:
        """Запустить N Monte-Carlo симуляций и вернуть агрегированный результат.

        Args:
            spec:          спецификация вмешательства.
            n_simulations: количество прогонов Монте-Карло.
            noise_sigma:   стандартное отклонение лог-нормального шума (0 = детерм.).
            timestamp:     время симуляции (по умолчанию текущее).
            random_seed:   seed для воспроизводимости (None = случайный).
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        if timestamp is None:
            timestamp = _now_ts()

        node = self._graph.get_node(spec.target_node_id)
        if node is None:
            raise KeyError(
                f"[InterventionOptimizer] Node '{spec.target_node_id}' not found in graph. "
                f"Available nodes: {list(self._graph.nodes.keys())[:10]}"
            )

        logger.info(
            f"[Intervention] Starting simulation: {spec.label!r}  "
            f"n={n_simulations}  node='{node.text[:50]}'"
        )
        t0   = time.time()
        runs: List[SimulationRun] = []

        # Базовые метрики (один раз, без шума)
        baseline = self._metrics.capture(spec.target_node_id, timestamp)
        baseline_tensions = self._safe_tension_snapshot(timestamp)

        for i in range(n_simulations):
            run = self._run_one(spec, timestamp, baseline, baseline_tensions,
                                noise_sigma, run_idx=i)
            if run is not None:
                runs.append(run)

        elapsed = time.time() - t0
        result  = self._aggregate(spec, runs, elapsed)
        logger.info(
            f"[Intervention] Done in {elapsed:.2f}s  "
            f"mean_accel={result.mean_years_accelerated:+.2f}yr  "
            f"prob={result.prob_acceleration*100:.1f}%"
        )
        return result

    def find_optimal(self,
                     candidates: List[InterventionSpec],
                     target_years: float = 3.0,
                     target_probability: float = 0.7,
                     max_interventions: int = 3,
                     budget_usd: Optional[float] = None,
                     n_simulations_per_candidate: int = 100,
                     beam_width: int = 3,
                     ) -> Tuple[List[InterventionSpec], float]:
        """Найти оптимальную комбинацию вмешательств (beam-search).

        Args:
            candidates:                  список спецификаций для перебора.
            target_years:                целевое ускорение в годах.
            target_probability:          минимальная вероятность достижения.
            max_interventions:           максимум вмешательств в комбинации.
            budget_usd:                  ограничение бюджета (None = без ограничения).
            n_simulations_per_candidate: прогонов на кандидата при оценке.
            beam_width:                  ширина пучка (1 = жадный).

        Returns:
            (список_спецификаций, ожидаемое_ускорение_лет)
        """
        return self._finder.find(
            candidates=candidates,
            target_years=target_years,
            target_probability=target_probability,
            max_interventions=max_interventions,
            budget_usd=budget_usd,
            n_simulations_per_candidate=n_simulations_per_candidate,
            beam_width=beam_width,
        )

    # ── Один прогон ──────────────────────────────────────────────────────────

    def _run_one(self,
                 spec:               InterventionSpec,
                 timestamp:          float,
                 baseline:           Dict[str, float],
                 baseline_tensions:  Dict[str, float],
                 noise_sigma:        float,
                 run_idx:            int,
                 ) -> Optional[SimulationRun]:
        """Один Monte-Carlo прогон: патч → метрики → откат."""
        # Генерируем noise_scale: log-нормальный вокруг 1.0
        if noise_sigma > 0:
            log_noise = np.random.normal(0.0, noise_sigma)
            noise_scale = float(np.exp(log_noise))
        else:
            noise_scale = 1.0

        try:
            patched_node = NodePatcher.patch(
                self._graph.nodes[spec.target_node_id], spec, noise_scale
            )
        except Exception as e:
            logger.debug(f"[Intervention] Run {run_idx}: patch failed: {e}")
            return None

        try:
            with GraphPatcher(self._graph, spec, patched_node, noise_scale):
                # Метрики с патченным узлом
                patched = self._metrics.capture(spec.target_node_id, timestamp)
                patched_tensions = self._safe_tension_snapshot(timestamp)

                # Каскадные эффекты
                cascade_effects = self._cascade.analyze(
                    spec.target_node_id, baseline, patched
                )
                # Боттлнеки
                bottleneck_changes = self._bneck.detect_changes(
                    baseline_tensions, patched_tensions
                )
        except Exception as e:
            logger.debug(f"[Intervention] Run {run_idx}: simulation failed: {e}")
            return None

        return SimulationRun(
            run_id              = f"{spec.target_node_id[:8]}_{run_idx}",
            noise_sigma         = noise_scale,
            baseline_readiness  = baseline["readiness"],
            patched_readiness   = patched["readiness"],
            baseline_forecast   = baseline["forecast"],
            patched_forecast    = patched["forecast"],
            baseline_tension    = baseline["tension"],
            patched_tension     = patched["tension"],
            baseline_lag_years  = baseline["lag_years"],
            patched_lag_years   = patched["lag_years"],
            cascade_effects     = cascade_effects,
            bottleneck_changes  = bottleneck_changes,
        )

    # ── Агрегация ─────────────────────────────────────────────────────────────

    def _aggregate(self,
                   spec:    InterventionSpec,
                   runs:    List[SimulationRun],
                   elapsed: float) -> InterventionResult:
        result = InterventionResult(
            spec           = spec,
            n_simulations  = len(runs),
            elapsed_s      = round(elapsed, 3),
            runs           = runs,
        )
        if not runs:
            return result

        dr_vals  = [r.delta_readiness  for r in runs]
        df_vals  = [r.delta_forecast   for r in runs]
        acc_vals = [r.years_accelerated for r in runs]

        result.mean_delta_readiness   = float(np.mean(dr_vals))
        result.std_delta_readiness    = float(np.std(dr_vals))
        result.mean_delta_forecast    = float(np.mean(df_vals))
        result.std_delta_forecast     = float(np.std(df_vals))
        result.mean_years_accelerated = float(np.mean(acc_vals))
        result.std_years_accelerated  = float(np.std(acc_vals))
        result.prob_acceleration      = float(np.mean([a > 0 for a in acc_vals]))

        sorted_acc = sorted(acc_vals)
        n = len(sorted_acc)
        result.p10_years_accelerated  = float(sorted_acc[max(0, int(n * 0.10))])
        result.p50_years_accelerated  = float(sorted_acc[max(0, int(n * 0.50))])
        result.p90_years_accelerated  = float(sorted_acc[max(0, int(n * 0.90))])

        # Топ каскадных эффектов: усредняем по прогонам
        cascade_agg: Dict[str, List[CascadeEffect]] = {}
        for run in runs:
            for ce in run.cascade_effects:
                cascade_agg.setdefault(ce.node_id, []).append(ce)
        agg_cascades: List[CascadeEffect] = []
        for nid, ces in cascade_agg.items():
            avg_ce = CascadeEffect(
                node_id         = nid,
                node_text       = ces[0].node_text,
                relation        = ces[0].relation,
                delta_readiness = float(np.mean([c.delta_readiness for c in ces])),
                delta_forecast  = float(np.mean([c.delta_forecast  for c in ces])),
                delta_investment= float(np.mean([c.delta_investment for c in ces])),
            )
            avg_ce.direction = (
                "positive" if avg_ce.delta_readiness + avg_ce.delta_forecast > 0
                else ("negative" if avg_ce.delta_readiness + avg_ce.delta_forecast < 0
                      else "neutral")
            )
            agg_cascades.append(avg_ce)
        agg_cascades.sort(key=lambda c: abs(c.delta_readiness + c.delta_forecast), reverse=True)
        result.top_cascade_effects = agg_cascades[:10]

        # Боттлнеки: мажоритарное голосование (изменение появляется в >50% прогонов)
        bneck_votes: Dict[Tuple[str, str], List[BottleneckChange]] = {}
        for run in runs:
            for bc in run.bottleneck_changes:
                key = (bc.node_id, bc.change)
                bneck_votes.setdefault(key, []).append(bc)
        for (nid, change), bcs in bneck_votes.items():
            if len(bcs) / len(runs) > 0.5:
                result.bottleneck_changes.append(bcs[0])

        return result

    # ── Вспомогательные ──────────────────────────────────────────────────────

    def _safe_tension_snapshot(self, timestamp: float) -> Dict[str, float]:
        try:
            return self._bneck._snapshot_tensions(timestamp)
        except Exception:
            return {}


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE FACTORIES
#  Готовые спецификации для самых частых сценариев.
# ══════════════════════════════════════════════════════════════════════════════

def spec_investment(node_id: str, amount_usd: float,
                    year: int = 2028, label: str = "") -> InterventionSpec:
    """Вмешательство типа 'вложить X USD в технологию узла'."""
    return InterventionSpec(
        target_node_id=node_id,
        intervention_type=InterventionType.INVESTMENT_INJECTION,
        params={"amount_usd": amount_usd, "year": year},
        label=label or f"${amount_usd/1e6:.0f}M investment @ {node_id[:15]}",
    )


def spec_remove_regulation(node_id: str, scale: float = 0.5,
                            label: str = "") -> InterventionSpec:
    """Вмешательство 'снять регуляторный барьер'."""
    return InterventionSpec(
        target_node_id=node_id,
        intervention_type=InterventionType.REGULATORY_REMOVAL,
        params={"inhibitory_scale": scale},
        label=label or f"regulatory_removal({scale:.0%}) @ {node_id[:15]}",
    )


def spec_adjacent_breakthrough(node_id: str, scientific_boost: float = 2.0,
                                 domain: Optional[str] = None,
                                 label: str = "") -> InterventionSpec:
    """Вмешательство 'прорыв в смежной технологии'."""
    params: Dict[str, Any] = {"scientific_boost": scientific_boost}
    if domain:
        params["only_domain"] = domain
    return InterventionSpec(
        target_node_id=node_id,
        intervention_type=InterventionType.BREAKTHROUGH_ADJACENT,
        params=params,
        label=label or f"adjacent_breakthrough(+{scientific_boost:.1f}) @ {node_id[:15]}",
    )


def spec_talent_influx(node_id: str, headcount_multiplier: float = 2.0,
                        social_boost: float = 1.5, label: str = "") -> InterventionSpec:
    """Вмешательство 'приток кадров'."""
    return InterventionSpec(
        target_node_id=node_id,
        intervention_type=InterventionType.TALENT_INFLUX,
        params={"headcount_multiplier": headcount_multiplier,
                "social_boost": social_boost},
        label=label or f"talent_influx(x{headcount_multiplier:.1f}) @ {node_id[:15]}",
    )


def spec_geopolitical(node_id: str, zone_multiplier_add: float = 1.5,
                       duration_years: float = 5.0, label: str = "") -> InterventionSpec:
    """Вмешательство 'геополитический катализатор'."""
    return InterventionSpec(
        target_node_id=node_id,
        intervention_type=InterventionType.GEOPOLITICAL_ACCEL,
        params={"zone_multiplier_add": zone_multiplier_add,
                "duration_years": duration_years},
        label=label or f"geopolitical(+{zone_multiplier_add:.1f}x) @ {node_id[:15]}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _now_ts() -> float:
    return time.time()


class _ZeroTensionState:
    tension = 0.0


def _make_stub_conflict_meter(graph: RuntimeGraph):
    """Создать и наполнить StructuralConflictMeter для текущего состояния графа."""
    try:
        meter = StructuralConflictMeter()
        meter.compute_all(graph)        # compute_all — правильный метод API
        return meter
    except Exception as exc:
        logger.debug(f"[_make_stub_conflict_meter] falling back to stub: {exc}")
        # Duck-typed stub: все конфликты = 0
        class _Stub:
            def get_conflict(self, nid: str) -> float: return 0.0
            def compute_all(self, *a, **kw): pass
        return _Stub()

__all__ = ['logger', 'InterventionType', '_PATCH_REGISTRY', 'InterventionSpec', 'CascadeEffect', 'BottleneckChange', 'SimulationRun', 'InterventionResult', 'NodePatcher', 'GraphPatcher', 'MetricsSnapshot', 'CascadeAnalyzer', 'BottleneckDetector', 'OptimalInterventionFinder', 'InterventionOptimizer', 'spec_investment', 'spec_remove_regulation', 'spec_adjacent_breakthrough', 'spec_talent_influx', 'spec_geopolitical', '_now_ts', '_ZeroTensionState', '_make_stub_conflict_meter']
__all__.remove("logger")
