# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Node and edge merging plus continuous graph updates."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, PHYSICAL_AXIS_ORDER
from .entities import KnowledgeEdge, KnowledgeNode, NodeFeatureBuilder, PhysicalSubstrateEncoder
from .edge_signals import (
    aggregate_unique_source_prevalence,
    mean_unique_source_metric,
    relation_factor_evidence,
)
from .graph import ANCESTRY_ENFORCER
from .ingestion import IngestionPipeline, IngestionResult
from .label_status import (
    LabelStatus,
    build_label_support_timeline,
    label_status_name,
    parse_label_status,
)
from .neural import Oracle1Model
from .time_utils import _ts_to_year
from .training import _local_gradient_step
from .validation import INGESTION_SOURCE_TYPES

# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH CONSOLIDATION ENGINE
#  Deduplication and parameter consolidation for nodes/edges from multiple sources.
#
#  Merge strategies by field type:
#    scientific_score   -> TRUST_WEIGHTED_MEAN  (Nature outweighs a blog)
#    maturity_score     -> TEMPORAL_TREND       (technology matures over time)
#    efficiency_plateau -> MAX                  (best work defines the ceiling)
#    dual_use_risk      -> MAX                  (worst-case for safety)
#    investment_total   -> SUM_UNIQUE           (no double-counting of rounds)
#    embedding          -> EMBEDDING_CENTROID   (semantic centroid)
# ══════════════════════════════════════════════════════════════════════════════

class MergeStrategy(Enum):
    TRUST_WEIGHTED_MEAN = auto()
    TEMPORAL_LATEST     = auto()
    TEMPORAL_TREND      = auto()
    MAX                 = auto()
    MIN                 = auto()
    SUM_UNIQUE          = auto()
    # Like SUM_UNIQUE but graph-prior-aware: the prior's accumulated total is
    # used as the BASE; only new-source values that exceed the current base are
    # added.  This prevents double-counting when apply_to_graph is called
    # repeatedly (prior already contains the previously-summed total).
    # TODO #14: no NODE_FIELD_STRATEGIES/EDGE_FIELD_STRATEGIES entry declares
    # this anymore (see the comment above investment_rounds) — kept as an enum
    # member for now rather than removed outright, since nothing depends on
    # renumbering the other members either way; FieldMerger has no dispatch
    # entry for it, so selecting it would fall through to values[0] (see
    # FieldMerger.apply's fn() if fn else values[0]).
    SUM_ADDITIVE_OVER_PRIOR = auto()
    UNION_LIST          = auto()
    EMBEDDING_CENTROID  = auto()
    MAJORITY_VOTE       = auto()
    KEEP_FIRST          = auto()


NODE_FIELD_STRATEGIES: Dict[str, MergeStrategy] = {
    "text":              MergeStrategy.KEEP_FIRST,
    "full_text":         MergeStrategy.KEEP_FIRST,
    "node_type":         MergeStrategy.MAJORITY_VOTE,
    "domain":            MergeStrategy.MAJORITY_VOTE,
    "entity_type":       MergeStrategy.MAJORITY_VOTE,
    "scientific_score":  MergeStrategy.TRUST_WEIGHTED_MEAN,
    "investment_score":  MergeStrategy.TRUST_WEIGHTED_MEAN,
    "social_score":      MergeStrategy.TRUST_WEIGHTED_MEAN,
    "maturity_score":    MergeStrategy.TEMPORAL_TREND,
    "readiness_score":   MergeStrategy.TEMPORAL_TREND,
    "group_size_score":  MergeStrategy.TRUST_WEIGHTED_MEAN,
    "dual_use_risk":        MergeStrategy.MAX,
    "legal_risk_score":     MergeStrategy.MAX,
    "export_control_risk":  MergeStrategy.MAX,
    "strategic_value":   MergeStrategy.TRUST_WEIGHTED_MEAN,
    "efficiency_plateau":MergeStrategy.MAX,
    # TODO #14 (found in TODO_rev2): these three previously declared
    # SUM_ADDITIVE_OVER_PRIOR here, but NodeConsolidator.consolidate() has
    # always overridden them to SUM_UNIQUE explicitly — the declaration was
    # dead (FieldMerger._sum_additive_over_prior() never executed) and
    # misleading to read. SUM_UNIQUE is the one actually exercised, so it's
    # now also what's declared; the unused method and enum value are removed
    # below rather than wired in, since re-enabling SUM_ADDITIVE_OVER_PRIOR
    # would need its own fix first (its "prior is always the highest-weight
    # entry" assumption no longer holds after TODO #9's prior_trust =
    # sum_real_trust instead of a flat constant — the same class of bug #11
    # found for MAJORITY_VOTE) and nothing currently depends on it.
    "investment_rounds":         MergeStrategy.SUM_UNIQUE,
    "investment_total_usd":      MergeStrategy.SUM_UNIQUE,
    "investment_last_round_usd": MergeStrategy.MAX,
    "investment_lead_investors": MergeStrategy.UNION_LIST,
    "sentiment_review_score":  MergeStrategy.TEMPORAL_TREND,
    "sentiment_fiction_score": MergeStrategy.TEMPORAL_TREND,
    "sentiment_forum_score":   MergeStrategy.TEMPORAL_TREND,
    "social_perception_score": MergeStrategy.TEMPORAL_TREND,
    "forum_post_count":        MergeStrategy.SUM_UNIQUE,
    "solves_limitations":MergeStrategy.UNION_LIST,
    "requires_node_ids": MergeStrategy.UNION_LIST,
    "enables_node_ids":  MergeStrategy.UNION_LIST,
    "embedding":         MergeStrategy.EMBEDDING_CENTROID,
}

PHYSICAL_AXIS_STRATEGIES: Dict[str, MergeStrategy] = {
    "constraint_proximity":   MergeStrategy.MAX,
    "synthesis_complexity":   MergeStrategy.MAX,
    "environmental_coupling": MergeStrategy.MAX,
    "resource_intensity":     MergeStrategy.MAX,
    "scale_feasibility":      MergeStrategy.TRUST_WEIGHTED_MEAN,
    "parallel_deployability": MergeStrategy.TRUST_WEIGHTED_MEAN,
    "cross_scale_stability":  MergeStrategy.TRUST_WEIGHTED_MEAN,
    "cascadability":          MergeStrategy.TRUST_WEIGHTED_MEAN,
    "reversibility":          MergeStrategy.TEMPORAL_TREND,
    "longevity":              MergeStrategy.TEMPORAL_TREND,
}

EDGE_FIELD_STRATEGIES: Dict[str, MergeStrategy] = {
    "semantic_similarity":    MergeStrategy.TRUST_WEIGHTED_MEAN,
    "temporal_proximity":     MergeStrategy.TEMPORAL_LATEST,
    "limitation_resolution":  MergeStrategy.MAX,
    "citation_link":          MergeStrategy.MAX,
    "investment_correlation": MergeStrategy.TRUST_WEIGHTED_MEAN,
    "social_correlation":     MergeStrategy.TRUST_WEIGHTED_MEAN,
    "inhibitory_force":       MergeStrategy.MAX,
    "confidence":             MergeStrategy.MAX,
    "relationship_type":      MergeStrategy.MAJORITY_VOTE,
}

# TODO #11 fix support: fields whose merge is a vote, not an average, need a
# per-candidate accumulated-weight vector as their sufficient statistic
# across sessions — a single (winner, combined_weight) pair (which IS
# correct for TRUST_WEIGHTED_MEAN, see TODO #9) silently gives the current
# winner credit for every vote ever cast, including votes cast for OTHER
# candidates that lost. See GraphConsolidator.apply_to_graph()'s prior-
# seeding and test_consolidation_order.py's MAJORITY_VOTE tests.
NODE_MAJORITY_VOTE_FIELDS = frozenset(
    f for f, s in NODE_FIELD_STRATEGIES.items() if s == MergeStrategy.MAJORITY_VOTE)
EDGE_MAJORITY_VOTE_FIELDS = frozenset(
    f for f, s in EDGE_FIELD_STRATEGIES.items() if s == MergeStrategy.MAJORITY_VOTE)

# TODO #13 fix support (found auditing #9): FieldMerger._temporal_trend()
# needs >=3 (timestamp, value) points to fit anything — below that it just
# returns the latest value. NodeConsolidator.consolidate() only ever sees
# the single collapsed __graph_prior__ point plus this session's real
# observation(s), i.e. exactly 2 points for the common "one document
# mentions this node per session" case — so the np.polyfit branch was
# unreachable in practice and every TEMPORAL_TREND field silently behaved
# like TEMPORAL_LATEST (verified: 5 sessions with maturity_score rising
# 2->3->4->5->6 each converged to exactly the raw input, no extrapolation).
# Fix mirrors NODE_MAJORITY_VOTE_FIELDS: keep a small sufficient-statistic
# per field (a bounded, timestamp-sorted window of real points, not just
# the latest) in consolidation_meta["trend_history"], and re-seed it each
# session as synthetic __graph_prior_trend__ echoes — see
# GraphConsolidator.apply_to_graph(). Only scalar_fields TEMPORAL_TREND
# members are covered; "reversibility"/"longevity" are PHYSICAL_AXIS_ORDER
# axes merged through a separate obs.physical code path and are NOT
# included here — left as-is, out of scope for this pass.
NODE_TREND_FIELDS = frozenset(
    f for f, s in NODE_FIELD_STRATEGIES.items() if s == MergeStrategy.TEMPORAL_TREND)
NODE_TREND_HISTORY_WINDOW = 5  # points kept per field; polyfit only needs a handful


@dataclass
class Observation:
    """Single observation of an entity extracted from one source document."""
    source:    str
    timestamp: float
    trust:     float
    fields:    Dict[str, Any]   = field(default_factory=dict)
    physical:  Dict[str, float] = field(default_factory=dict)


@dataclass
class NodeRecord:
    canonical_id:  str
    observations:  List[Observation] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    @property
    def source_count(self) -> int:
        return len(set(o.source for o in self.observations))

    @property
    def latest_timestamp(self) -> float:
        return max((o.timestamp for o in self.observations), default=0.0)


@dataclass
class EdgeRecord:
    source_id:    str
    target_id:    str
    observations: List[Observation] = field(default_factory=list)

    def add(self, obs: Observation) -> None:
        self.observations.append(obs)

    @property
    def edge_key(self) -> Tuple[str, str]:
        return (self.source_id, self.target_id)


class FieldMerger:
    """Applies a merge strategy to a list of (value, weight, timestamp) tuples. All methods are static."""

    @staticmethod
    def apply(strategy: MergeStrategy, values: List[Any],
              weights: List[float], timestamps: List[float]) -> Any:
        if not values:
            return None
        dispatch = {
            MergeStrategy.TRUST_WEIGHTED_MEAN: lambda: FieldMerger._trust_weighted_mean(values, weights),
            MergeStrategy.TEMPORAL_LATEST:     lambda: FieldMerger._temporal_latest(values, timestamps),
            MergeStrategy.TEMPORAL_TREND:      lambda: FieldMerger._temporal_trend(values, timestamps),
            MergeStrategy.MAX:                 lambda: FieldMerger._safe_max(values),
            MergeStrategy.MIN:                 lambda: FieldMerger._safe_min(values),
            MergeStrategy.SUM_UNIQUE:               lambda: FieldMerger._sum_unique(values),
            MergeStrategy.UNION_LIST:          lambda: FieldMerger._union_list(values),
            MergeStrategy.EMBEDDING_CENTROID:  lambda: FieldMerger._embedding_centroid(values, weights),
            MergeStrategy.MAJORITY_VOTE:       lambda: FieldMerger._majority_vote(values, weights),
            MergeStrategy.KEEP_FIRST:          lambda: FieldMerger._keep_first(values),
        }
        fn = dispatch.get(strategy)
        return fn() if fn else values[0]

    @staticmethod
    def _trust_weighted_mean(values, weights) -> float:
        numeric, w_valid = [], []
        for v, w in zip(values, weights):
            try:
                numeric.append(float(v)); w_valid.append(max(float(w), 1e-6))
            except (TypeError, ValueError):
                pass
        if not numeric:
            return 0.0
        total_w = sum(w_valid)
        return sum(v * w for v, w in zip(numeric, w_valid)) / total_w

    @staticmethod
    def _temporal_latest(values, timestamps) -> Any:
        if not timestamps:
            return values[-1] if values else None
        pairs = [(ts, v) for ts, v in zip(timestamps, values) if v is not None]
        return max(pairs, key=lambda x: x[0])[1] if pairs else None

    @staticmethod
    def _temporal_trend(values, timestamps, value_max: float = 10.0) -> float:
        """Latest value + weak linear extrapolation capped at ±10% of the last value.

        *value_max* controls the upper clip bound:
          - 10.0 for standard node score fields (0-10 range)
          - 1.0  for physical substrate axes (0-1 range)
        """
        numeric_pairs = []
        for ts, v in zip(timestamps, values):
            try:
                pair = (float(ts), float(v))
                if np.isfinite(pair[0]) and np.isfinite(pair[1]):
                    numeric_pairs.append(pair)
            except (TypeError, ValueError):
                pass
        if not numeric_pairs:
            return 0.0
        numeric_pairs.sort(key=lambda x: x[0])
        latest = numeric_pairs[-1][1]
        if len(numeric_pairs) < 3:
            return latest
        ts_arr  = np.array([p[0] for p in numeric_pairs])
        v_arr   = np.array([p[1] for p in numeric_pairs])
        # Multiple papers often carry only a publication year, so several
        # observations can share one timestamp.  There is no temporal slope
        # to fit in that case; returning the latest measured value is safer
        # than passing a rank-deficient design matrix to polyfit.
        ts_std = float(ts_arr.std())
        if len(np.unique(ts_arr)) < 2 or not np.isfinite(ts_std) or ts_std < 1e-8:
            return latest
        ts_norm = (ts_arr - ts_arr.mean()) / ts_std
        try:
            slope = float(np.polyfit(ts_norm, v_arr, 1)[0])
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return latest
        if not np.isfinite(slope):
            return latest
        delta   = np.clip(slope * 0.1, -latest * 0.1, latest * 0.1)
        return float(np.clip(latest + delta, 0.0, value_max))

    @staticmethod
    def _safe_max(values) -> Any:
        numeric = [float(v) for v in values if v is not None]
        return max(numeric) if numeric else 0.0

    @staticmethod
    def _safe_min(values) -> Any:
        numeric = [float(v) for v in values if v is not None]
        return min(numeric) if numeric else 0.0

    @staticmethod
    def _sum_unique(values) -> float:
        seen: set = set()
        total = 0.0
        for v in values:
            try:
                fv = float(v); key = round(fv, 2)
                if key not in seen and key > 0:
                    seen.add(key); total += fv
            except (TypeError, ValueError):
                pass
        return total

    @staticmethod
    def _union_list(values) -> List:
        seen = set(); result = []
        for lst in values:
            if not isinstance(lst, (list, tuple)):
                continue
            for item in lst:
                key = str(item)
                if key not in seen:
                    seen.add(key); result.append(item)
        return result

    @staticmethod
    def _embedding_centroid(values, weights) -> Optional[np.ndarray]:
        arrays, w_valid = [], []
        for v, w in zip(values, weights):
            if v is None:
                continue
            try:
                arr = np.array(v, dtype=np.float32)
                arrays.append(arr); w_valid.append(max(float(w), 1e-6))
            except Exception:
                pass
        if not arrays:
            return None
        total_w  = sum(w_valid)
        centroid = sum(a * w for a, w in zip(arrays, w_valid)) / total_w
        mean_norm = np.mean([np.linalg.norm(a) for a in arrays])
        norm = np.linalg.norm(centroid)
        if norm > 1e-8 and mean_norm > 1e-8:
            centroid = centroid / norm * mean_norm
        return centroid

    @staticmethod
    def _majority_vote(values, weights) -> Any:
        vote_map: Dict[str, float] = {}
        for v, w in zip(values, weights):
            if v is None or v == "":
                continue
            key = str(v)
            vote_map[key] = vote_map.get(key, 0.0) + max(float(w), 1e-6)
        if not vote_map:
            return values[0] if values else ""
        return max(vote_map, key=vote_map.get)

    @staticmethod
    def _keep_first(values) -> Any:
        non_empty = [v for v in values if v is not None and v != ""]
        if not non_empty:
            return values[0] if values else None
        if all(isinstance(v, str) for v in non_empty):
            return max(non_empty, key=len)
        return non_empty[0]


class NodeConsolidator:
    """
    Merges N observations of one node into a single KnowledgeNode.
    Physical axes are handled separately via PHYSICAL_AXIS_STRATEGIES.
    Conflict metadata is stored in node.__dict__['consolidation_meta'].
    """

    def consolidate(self, record: NodeRecord) -> KnowledgeNode:
        obs        = record.observations
        weights    = [o.trust for o in obs]
        timestamps = [o.timestamp for o in obs]

        # Observations from the graph-prior, its per-candidate vote echoes
        # (TODO #11) and its per-field trend-history echoes (TODO #13) are
        # NOT real external sources. They must participate in field merging
        # (as the accumulated knowledge baseline) but must NOT appear in
        # provenance or be treated as independent sources for conflict
        # detection.
        real_obs = [o for o in obs
                    if o.source not in ("__graph_prior__", "__graph_prior_vote__",
                                         "__graph_prior_trend__")]

        def merge(field_name: str, strategy: Optional[MergeStrategy] = None) -> Any:
            s = strategy or NODE_FIELD_STRATEGIES.get(
                field_name, MergeStrategy.TRUST_WEIGHTED_MEAN)
            return FieldMerger.apply(s, [o.fields.get(field_name) for o in obs],
                                     weights, timestamps)

        node = KnowledgeNode(
            id=record.canonical_id,
            text=merge("text"),
            full_text=merge("full_text", MergeStrategy.KEEP_FIRST),
            node_type=merge("node_type"), domain=merge("domain"),
            entity_type=merge("entity_type"),
            # provenance only from real sources, not the internal prior
            provenance=", ".join(sorted({o.source for o in real_obs})) if real_obs
                       else ", ".join(sorted({o.source for o in obs})),
            # timestamp: latest real-source observation; fall back to latest overall
            timestamp=max((o.timestamp for o in real_obs), default=record.latest_timestamp),
            publication_date=merge("publication_date", MergeStrategy.TEMPORAL_LATEST),
            scientific_score=merge("scientific_score"),
            investment_score=merge("investment_score"),
            social_score=merge("social_score"),
            maturity_score=merge("maturity_score"),
            readiness_score=merge("readiness_score"),
            group_size_score=merge("group_size_score"),
            dual_use_risk=merge("dual_use_risk"),
            legal_risk_score=merge("legal_risk_score"),
            export_control_risk=merge("export_control_risk"),
            strategic_value=merge("strategic_value"),
            efficiency_plateau=merge("efficiency_plateau"),
            investment_rounds=int(merge("investment_rounds", MergeStrategy.SUM_UNIQUE) or 0),
            investment_total_usd=merge("investment_total_usd", MergeStrategy.SUM_UNIQUE) or 0.0,
            investment_last_round_usd=merge("investment_last_round_usd", MergeStrategy.MAX) or 0.0,
            investment_lead_investors=merge("investment_lead_investors", MergeStrategy.UNION_LIST) or [],
            sentiment_review_score=merge("sentiment_review_score"),
            sentiment_fiction_score=merge("sentiment_fiction_score"),
            sentiment_forum_score=merge("sentiment_forum_score"),
            social_perception_score=merge("social_perception_score"),
            forum_post_count=int(merge("forum_post_count", MergeStrategy.SUM_UNIQUE) or 0),
            solves_limitations=merge("solves_limitations", MergeStrategy.UNION_LIST) or [],
            requires_node_ids=merge("requires_node_ids", MergeStrategy.UNION_LIST) or [],
            enables_node_ids=merge("enables_node_ids", MergeStrategy.UNION_LIST) or [],
            embedding=merge("embedding", MergeStrategy.EMBEDDING_CENTROID),
        )
        # Physical substrate — merge per-axis, then rebuild the canonical embedding tail
        merged_physical: Dict[str, float] = {}
        has_physical = any(bool(o.physical) for o in obs)
        if has_physical:
            for axis, strategy in PHYSICAL_AXIS_STRATEGIES.items():
                axis_values  = [o.physical.get(axis) for o in obs if axis in o.physical]
                axis_weights = [o.trust for o in obs if axis in o.physical]
                axis_ts      = [o.timestamp for o in obs if axis in o.physical]
                if axis_values:
                    if strategy == MergeStrategy.TEMPORAL_TREND:
                        # Physical axes are 0-1 scale — use tight clip
                        merged_val = FieldMerger._temporal_trend(
                            axis_values, axis_ts, value_max=1.0)
                    else:
                        merged_val = FieldMerger.apply(
                            strategy, axis_values, axis_weights, axis_ts) or 0.5
                    merged_physical[axis] = float(merged_val)
            node.__dict__["raw_physical_scores"] = merged_physical

            # Rebuild the physical section of the embedding using the real encoder so that
            # PhysicalSubstrateEncoder.feasibility_score() and resolve_rupture_physics()
            # always operate on a consistent 416-dim vector.
            phys_encoder = PhysicalSubstrateEncoder()
            phys_vec = phys_encoder.build_physical_section(merged_physical)
            node.__dict__["feasibility"] = phys_encoder.feasibility_score(phys_vec)

            if node.embedding is not None:
                emb = np.array(node.embedding, dtype=np.float32)
                if len(emb) == NodeFeatureBuilder.TOTAL_DIM:
                    # Centroid was computed from 407-dim base vectors — extend to 416
                    node.embedding = phys_encoder.extend_node_features(emb, merged_physical)
                elif len(emb) == PhysicalSubstrateEncoder.EXTENDED_TOTAL_DIM:
                    # Already 416-dim (centroid of extended vectors) — overwrite tail
                    emb[NodeFeatureBuilder.TOTAL_DIM:] = phys_vec
                    node.embedding = emb
        # Conflict metadata
        conflict_map = self._compute_conflicts(obs)
        node.__dict__["consolidation_meta"] = {
            "source_count":     record.source_count,
            # observation_count/sources: real sources only (same real_obs as
            # provenance above) — this previously read the unfiltered obs
            # list, so it counted the graph-prior and every vote echo as if
            # each were an independent source.
            "observation_count": len(real_obs),
            "sources":          [o.source for o in real_obs],
            "timestamp_range":  [min(o.timestamp for o in obs),
                                  max(o.timestamp for o in obs)],
            "conflicts":        conflict_map,
            "trust_weights":    weights,
        }
        logger.debug(f"[NodeConsolidator] '{record.canonical_id}': {len(obs)} obs, "
                     f"{record.source_count} sources, {len(conflict_map)} conflicts")
        return node

    @staticmethod
    def _compute_conflicts(obs: List[Observation],
                            variance_threshold: float = None) -> Dict[str, Dict]:
        """Detect fields where independent real sources disagree significantly.

        Uses separate thresholds for:
          - 0-10 score fields  (std > 2.0 ≈ 20% of range is conflicting)
          - 0-1 physical axes  (std > 0.20 ≈ 20% of range is conflicting)

        The __graph_prior__ observation is intentionally excluded from conflict
        detection — it represents accumulated knowledge, not an independent source.
        Comparing the prior against new data would generate spurious conflicts on
        every re-ingestion cycle.
        """
        # Only count genuinely independent external sources (excludes the
        # graph-prior and its TODO #11 vote / TODO #13 trend-history echoes).
        real_obs = [o for o in obs
                    if o.source not in ("__graph_prior__", "__graph_prior_vote__",
                                         "__graph_prior_trend__")]
        if len(real_obs) < 2:
            return {}

        score_threshold = variance_threshold or CONFIG.get(
            "consolidation_conflict_variance_threshold", 2.0)
        physical_threshold = CONFIG.get(
            "consolidation_physical_conflict_threshold", 0.20)

        conflicts: Dict[str, Dict] = {}

        # ── Standard node score fields ────────────────────────────────────────
        for fname in list(NODE_FIELD_STRATEGIES.keys()):
            vals, srcs = [], []
            for o in real_obs:
                v = o.fields.get(fname)
                try:
                    fv = float(v)
                    vals.append(fv); srcs.append(o.source)
                except (TypeError, ValueError):
                    pass
            if len(vals) >= 2:
                std = statistics.stdev(vals)
                if std > score_threshold:
                    conflicts[fname] = {
                        "std": round(std, 3), "min": min(vals),
                        "max": max(vals), "values": vals, "sources": srcs,
                        "scale": "0-10",
                    }

        # ── Physical substrate axes (0-1 scale, different threshold) ─────────
        for axis in PHYSICAL_AXIS_ORDER:
            vals, srcs = [], []
            for o in real_obs:
                v = o.physical.get(axis)
                if v is not None:
                    try:
                        vals.append(float(v)); srcs.append(o.source)
                    except (TypeError, ValueError):
                        pass
            if len(vals) >= 2:
                std = statistics.stdev(vals)
                if std > physical_threshold:
                    conflicts[f"physical.{axis}"] = {
                        "std": round(std, 3), "min": min(vals),
                        "max": max(vals), "values": vals, "sources": srcs,
                        "scale": "0-1",
                    }

        return conflicts


class EdgeConsolidator:
    """
    Merges N observations of one edge into a single KnowledgeEdge.
    relationship_type is resolved by trust-weighted majority vote.

    Edge component fields (semantic_similarity, inhibitory_force, etc.) use None
    as the "not extracted" sentinel.  Storing 0.0 as default would include
    unextracted components in TRUST_WEIGHTED_MEAN and silently underestimate
    all edge weights.
    """

    # Edge component fields that should be treated as "not extracted" when 0.0.
    # These are all normalised to [0, 1] and default to 0.0 in KnowledgeEdge.
    EDGE_SCORE_FIELDS = {
        "semantic_similarity", "temporal_proximity", "limitation_resolution",
        "citation_link", "investment_correlation", "social_correlation",
        "inhibitory_force", "confidence",
    }

    def consolidate(self, record: EdgeRecord,
                    corpus_document_count: int = 0) -> KnowledgeEdge:
        obs        = record.observations
        weights    = [o.trust for o in obs]
        timestamps = [o.timestamp for o in obs]

        source_points: List[Dict[str, Any]] = []
        legacy_baseline: Dict[str, float] = {}
        for observation in obs:
            fields = observation.fields if isinstance(observation.fields, dict) else {}
            if observation.source == "__graph_prior__":
                historical = fields.get("source_observations", [])
                if isinstance(historical, list) and historical:
                    source_points.extend(point for point in historical if isinstance(point, dict))
                else:
                    # Preserve scores from snapshots created before evidence
                    # receipts were persisted; new observations never use this path.
                    for name in EDGE_SCORE_FIELDS:
                        try:
                            value = float(fields.get(name, 0.0) or 0.0)
                        except (TypeError, ValueError):
                            continue
                        if value > 0.0:
                            legacy_baseline[name] = max(legacy_baseline.get(name, 0.0), value)
                continue
            factor_evidence = fields.get("factor_evidence")
            if not isinstance(factor_evidence, dict):
                evidence = fields.get("evidence", []) or []
                factor_evidence = relation_factor_evidence(
                    str(fields.get("relationship_type", "related")), bool(evidence))
            source_points.append({
                "source": str(observation.source),
                "source_identity": str(fields.get("source_identity") or observation.source),
                "timestamp": float(observation.timestamp) if observation.timestamp else None,
                "trust": float(np.clip(observation.trust, 0.0, 1.0)),
                "relationship_type": fields.get("relationship_type", "related"),
                "signal_type": fields.get("signal_type"),
                "semantic_similarity": fields.get("semantic_similarity", 0.0),
                "temporal_proximity": fields.get("temporal_proximity", 0.0),
                "factor_evidence": dict(factor_evidence),
                "parameter_observations": list(fields.get("parameter_observations", []) or []),
                "evidence": list(fields.get("evidence", []) or []),
                "source_observation": fields.get("source_observation", {}),
                "source_metadata": dict(fields.get("source_metadata", {}) or {}),
            })

        def merge(field_name: str) -> Any:
            s = EDGE_FIELD_STRATEGIES.get(field_name, MergeStrategy.TRUST_WEIGHTED_MEAN)
            # Use None for "not extracted" component values (same as NodeConsolidator).
            # FieldMerger._trust_weighted_mean / _safe_max skip None via float() exception.
            raw_values = []
            for o in obs:
                v = o.fields.get(field_name)
                if field_name in self.EDGE_SCORE_FIELDS and isinstance(v, float) and v == 0.0:
                    raw_values.append(None)
                else:
                    raw_values.append(v if v is not None else o.fields.get(field_name, None))
            return FieldMerger.apply(s, raw_values, weights, timestamps)

        # Bug fix (found auditing TODO #11): a default of "related" here made
        # the regular __graph_prior__ observation (whose relationship_type is
        # deliberately popped before re-seeding, see apply_to_graph()) cast a
        # PHANTOM vote for the literal string "related", weighted by whatever
        # prior_trust has accumulated — a value no real source ever reported.
        # None (the correct "field missing" signal) is safely skipped by
        # _majority_vote(); mirrors how NodeConsolidator.merge() already
        # handles domain/node_type/entity_type.
        rt_values  = [o.fields.get("relationship_type") for o in obs]
        primary_rt = FieldMerger._majority_vote(rt_values, weights) or "related"
        # A single grounded contradiction is structurally meaningful even
        # when later sources vote for a generic relation. Keep that edge
        # visibly marked as contradictory while retaining every source vote
        # in its observation history and conflict metadata below.
        if any(str(value).casefold() == "contradicts" for value in rt_values):
            primary_rt = "contradicts"
        pair_metrics = {
            name: mean_unique_source_metric(source_points, name, legacy_baseline.get(name, 0.0))
            for name in ("semantic_similarity", "temporal_proximity")
        }
        cumulative_factors = {
            name: aggregate_unique_source_prevalence(
                source_points, name, corpus_document_count)
            for name in ("limitation_resolution", "citation_link",
                         "investment_correlation", "social_correlation",
                         "inhibitory_force")
        }
        edge = KnowledgeEdge(
            id=f"cons_{record.source_id[:6]}_{record.target_id[:6]}",
            source_id=record.source_id, target_id=record.target_id,
            semantic_similarity=pair_metrics["semantic_similarity"],
            temporal_proximity=pair_metrics["temporal_proximity"],
            limitation_resolution=cumulative_factors["limitation_resolution"],
            citation_link=cumulative_factors["citation_link"],
            investment_correlation=cumulative_factors["investment_correlation"],
            social_correlation=cumulative_factors["social_correlation"],
            inhibitory_force=cumulative_factors["inhibitory_force"],
            confidence=float(merge("confidence") or 0.0),
            relationship_type=primary_rt,
            evidence=FieldMerger._union_list([o.fields.get("evidence", []) for o in obs]),
            timestamp=max(o.timestamp for o in obs),
        )
        edge.compute_total_weight()
        edge.__dict__["influence_corpus_size"] = int(corpus_document_count)
        # Exclude the graph-prior AND its per-candidate vote echoes (TODO #11)
        # from the "sources" metadata — neither is an independent external
        # source; leaving __graph_prior_vote__ in here previously leaked that
        # bookkeeping string into consolidation_meta["sources"]/observation_count.
        real_obs = [o for o in obs
                    if o.source not in ("__graph_prior__", "__graph_prior_vote__")]
        relationship_type_votes: Dict[str, float] = {}
        for observation in obs:
            relationship = observation.fields.get("relationship_type")
            if relationship not in (None, ""):
                relationship_type_votes[str(relationship)] = (
                    relationship_type_votes.get(str(relationship), 0.0)
                    + max(0.0, float(observation.trust))
                )
        # Keep the minority relationship claims visible across incremental
        # sessions. A weighted winner is useful as a summary, but it is not a
        # retraction mechanism and must not hide sources that disagree.
        conflicts = self._compute_edge_conflicts(real_obs)
        if len(relationship_type_votes) > 1:
            conflicts["relationship_type"] = {
                "values": sorted(relationship_type_votes),
                "trust_weights": {
                    key: round(value, 4)
                    for key, value in relationship_type_votes.items()
                },
            }
        edge.__dict__["alternate_relationship_types"] = [
            rt for rt in sorted(relationship_type_votes) if rt != primary_rt]
        persisted_observations = []
        persisted_keys = set()
        for point in source_points:
            source_name = str(point.get("source", ""))
            if source_name in ("__graph_prior__", "__graph_prior_vote__"):
                continue
            evidence = point.get("evidence", []) or []
            if isinstance(evidence, str):
                evidence = [evidence]
            identity = str(point.get("source_identity") or source_name)
            signal_type = point.get("signal_type")
            relationship = point.get("relationship_type")
            key = (identity.casefold(), str(signal_type or "").casefold(),
                   str(relationship or "").casefold(), tuple(evidence or []))
            if key in persisted_keys:
                continue
            persisted_keys.add(key)
            persisted_observations.append({
                "source": source_name,
                "source_identity": identity,
                "timestamp": point.get("timestamp"),
                "trust": float(max(0.0, float(point.get("trust", 0.0) or 0.0))),
                "relationship_type": relationship,
                "signal_type": signal_type,
                "confidence": float(point.get("confidence") or 0.0),
                "semantic_similarity": point.get("semantic_similarity", 0.0),
                "temporal_proximity": point.get("temporal_proximity", 0.0),
                "factor_evidence": dict(point.get("factor_evidence", {}) or {}),
                "parameter_observations": list(point.get("parameter_observations", []) or []),
                "source_metadata": dict(point.get("source_metadata", {}) or {}),
                "source_label_status_code": point.get("source_label_status_code"),
                "source_observation": point.get("source_observation", {}),
                "evidence": [str(quote)[:1000] for quote in evidence
                             if isinstance(quote, str) and quote.strip()][:5],
            })
        edge.__dict__["source_observations"] = persisted_observations
        edge.__dict__["consolidation_meta"] = {
            "observation_count": len(real_obs),
            "sources": [o.source for o in real_obs],
            "conflicts": conflicts,
            "relationship_type_votes": relationship_type_votes,
        }
        return edge

    @staticmethod
    def _compute_edge_conflicts(real_obs: List[Observation]) -> Dict[str, Dict]:
        """Detect edge component fields where real sources genuinely disagree.

        0.0 values are treated as "not extracted by LLM" (same sentinel logic
        used in EdgeConsolidator.merge and NodeConsolidator._node_to_fields).
        Including 0.0 defaults would generate false conflicts whenever one
        source doesn't mention a field and another source reports e.g. 0.9.
        """
        if len(real_obs) < 2:
            return {}
        threshold = CONFIG.get("consolidation_physical_conflict_threshold", 0.20)
        # Fields where 0.0 means "not observed by this source" (all normalised
        # edge score components).  Only non-zero values from ≥2 sources are
        # considered for conflict detection.
        SENTINEL_ZERO_FIELDS = {
            "semantic_similarity", "temporal_proximity", "limitation_resolution",
            "citation_link", "investment_correlation", "social_correlation",
            "inhibitory_force", "confidence",
        }
        conflicts: Dict[str, Dict] = {}
        for fname in EDGE_FIELD_STRATEGIES:
            if fname == "relationship_type":
                continue
            vals, srcs = [], []
            for o in real_obs:
                v = o.fields.get(fname)
                try:
                    fv = float(v)
                    # Skip 0.0 for score fields — indistinguishable from
                    # "not extracted"; including them creates false conflicts.
                    if fname in SENTINEL_ZERO_FIELDS and fv == 0.0:
                        continue
                    vals.append(fv); srcs.append(o.source)
                except (TypeError, ValueError):
                    pass
            if len(vals) >= 2:
                std = statistics.stdev(vals)
                if std > threshold:
                    conflicts[fname] = {
                        "std": round(std, 3), "min": min(vals),
                        "max": max(vals), "values": vals, "sources": srcs,
                    }
        return conflicts


@dataclass
class ConsolidationStats:
    nodes_before:        int  = 0
    nodes_after:         int  = 0
    edges_before:        int  = 0
    edges_after:         int  = 0
    merges_applied:      int  = 0
    conflicts_found:     int  = 0
    high_conflict_nodes: List[str] = field(default_factory=list)
    # TODO #15: count of DERIVED_FROM edges AncestryEnforcer had to
    # synthesise post-hoc because apply_to_graph() writes nodes directly
    # (graph.nodes[cid] = node) rather than through add_node_checked(),
    # which is where this normally happens automatically.
    ancestry_edges_synthesised: int = 0
    # TODO #16: count of RuntimeGraph.union_family() calls triggered this
    # call by an is_family-tagged edge (see BucketIndex.FAMILY_THRESHOLD).
    family_links_created: int = 0


class GraphConsolidator:
    """
    Orchestrator over RuntimeGraph:
        1. Collects NodeRecord / EdgeRecord for each canonical entity
        2. Runs NodeConsolidator / EdgeConsolidator
        3. Updates the graph (replaces duplicate nodes/edges)
        4. Provides a conflict report (fields with high variance across sources)

    Typical usage:
        consolidator = GraphConsolidator()
        for doc in documents:
            result = pipeline.ingest(doc.text, ...)
            consolidator.observe_from_ingestion_result(result, result.source_type)
        stats = consolidator.apply_to_graph(graph)
        print_conflict_report(consolidator)
    """

    def __init__(self):
        self._node_records: Dict[str, NodeRecord]   = {}
        self._edge_records: Dict[Tuple, EdgeRecord] = {}
        self._node_consolidator = NodeConsolidator()
        self._edge_consolidator = EdgeConsolidator()
        self._id_redirects: Dict[str, str]          = {}
        self._influence_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._document_identities: set = set()
        self._corpus_document_count = 0

    def observe_node(self, node, source: str, timestamp: float,
                     trust: float = 0.7,
                     canonical_id: Optional[str] = None) -> str:
        cid = canonical_id or node.id
        # Guard: a date_unknown observation must only enrich an existing node.
        # If it arrives without a canonical_id mapping to a known graph node
        # (i.e. no one confirmed it matches something already in the graph),
        # reject it to prevent undated nodes from entering the knowledge base.
        if node.__dict__.get("date_unknown") and canonical_id is None:
            logger.warning(
                f"[Consolidator] observe_node rejected '{node.id}' "
                f"('{node.text[:60]}'): no date and no canonical_id — "
                "cannot create a new dateless node. "
                "Provide publication_year or route via load_result_into_graph "
                "with a consolidator so the match resolves first."
            )
            return cid  # return without adding to records
        if cid not in self._node_records:
            self._node_records[cid] = NodeRecord(canonical_id=cid)
        physical = getattr(node, "__dict__", {}).get("raw_physical_scores", {})
        obs = Observation(source=source, timestamp=timestamp, trust=trust,
                          physical=physical or {},
                          fields=self._node_to_fields(node))
        self._node_records[cid].add(obs)
        if node.id != cid:
            self._id_redirects[node.id] = cid
        return cid

    def observe_edge(self, edge, source: str, timestamp: float,
                     trust: float = 0.7,
                     canonical_src: Optional[str] = None,
                     canonical_tgt: Optional[str] = None,
                     source_label_status_code: Optional[int] = None,
                     source_observation: Optional[Dict[str, Any]] = None) -> None:
        src = canonical_src or self._id_redirects.get(edge.source_id, edge.source_id)
        tgt = canonical_tgt or self._id_redirects.get(edge.target_id, edge.target_id)
        key = (src, tgt)
        if key not in self._edge_records:
            self._edge_records[key] = EdgeRecord(source_id=src, target_id=tgt)
        fields = self._edge_to_fields(edge)
        if source_label_status_code is not None:
            fields["source_label_status_code"] = int(source_label_status_code)
        if source_observation:
            fields["source_observation"] = dict(source_observation)
        self._edge_records[key].add(
            Observation(source=source, timestamp=timestamp, trust=trust,
                        fields=fields))

    def observe_from_ingestion_result(self, result: IngestionResult,
                                       source_type: str = "unknown") -> None:
        trust = INGESTION_SOURCE_TYPES.get(
            source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"]
        ts = max((n.timestamp for n in result.nodes), default=0.0)
        for node in result.nodes:
            self.observe_node(node, source=result.source,
                              timestamp=node.timestamp or ts, trust=trust)
        source_nodes = {node.id: node for node in result.nodes}
        for edge in result.edges + result.cross_doc_edges:
            source_node = source_nodes.get(edge.source_id)
            source_private = getattr(source_node, "__dict__", {}) if source_node else {}
            source_status = source_private.get("label_status_code")
            source_observation = {}
            if source_node is not None:
                source_observation = {
                    "source_node_id": source_node.id,
                    "source_node_text": source_node.text,
                    "timestamp": float(source_node.timestamp or ts),
                    "trust": float(source_private.get("source_trust", trust)),
                    "label_status_code": source_status,
                    "label_status": source_private.get("label_status"),
                    "status_evidence": source_private.get("status_evidence", ""),
                    "evidence": list(source_private.get("source_evidence", [])),
                    "source_metadata": dict(source_private.get("source_metadata", {})),
                }
            self.observe_edge(
                edge, source=result.source, timestamp=ts, trust=trust,
                source_label_status_code=source_status,
                source_observation=source_observation,
            )
        source_identity = str(result.document_metadata.get("document_content_hash") or result.source)
        if source_identity:
            self._document_identities.add(source_identity.casefold())
        for signal in result.influence_observations:
            if not isinstance(signal, dict):
                continue
            target_id = self._id_redirects.get(
                str(signal.get("target_node_id", "")), str(signal.get("target_node_id", "")))
            if not target_id or not signal.get("evidence"):
                continue
            self._influence_records[target_id].append({
                "source": str(result.source),
                "source_identity": str(signal.get("source_identity") or source_identity),
                "timestamp": float(signal.get("timestamp") or ts or 0.0),
                "trust": float(np.clip(signal.get("trust", trust), 0.0, 1.0)),
                "signal_type": str(signal.get("signal_type", "")),
                "evidence": str(signal.get("evidence", ""))[:1000],
                "factor_evidence": {},
                "parameter_observations": list(signal.get("parameter_observations", []) or []),
                "source_metadata": dict(result.document_metadata),
            })

    def apply_to_graph(self, graph) -> ConsolidationStats:
        processed_document_ids = getattr(graph, "processed_document_ids", set()) or set()
        self._corpus_document_count = len(
            {str(item).casefold() for item in processed_document_ids if str(item).strip()}
            | self._document_identities)
        stats = ConsolidationStats(
            nodes_before=len(graph.nodes),
            edges_before=len(graph.edges))

        # ── Seed each NodeRecord with the current graph state as a prior ──────
        # Without this, each apply_to_graph call only consolidates observations
        # from the current session.  Existing accumulated data (from previous
        # ingestion sessions, DormancyTracker updates, etc.) would be silently
        # overwritten by a fresh partial extraction.
        #
        # IMPORTANT: we must never double-seed.  If apply_to_graph is called
        # multiple times on the same consolidator, the prior from the previous
        # call is already baked into the graph node.  We purge any leftover
        # __graph_prior__ observations before re-seeding so the prior is always
        # fresh and never accumulates N copies with weight N×1.0.
        #
        # Order-dependence fix (see test_consolidation_order.py): a flat prior
        # trust (formerly a constant 1.0, i.e. higher than *any* real source)
        # made whichever source got ingested first "sticky" — later sessions,
        # even from a more trustworthy source, could only partially move a
        # value that a low-trust source had anchored first, simply because it
        # went first. The fix carries forward the *actual* combined trust
        # weight behind the existing value (consolidation_meta["sum_real_trust"],
        # which for TRUST_WEIGHTED_MEAN-family fields is exactly the
        # sufficient statistic needed for the running weighted mean to match a
        # from-scratch batch recomputation over every session to date,
        # regardless of order) instead of a fixed constant. The CONFIG value
        # remains as the fallback for nodes/edges with no recorded provenance
        # (e.g. written directly by ingestion without ever going through a
        # consolidator) — there we have no information to do better, so we
        # keep the original conservative (maximally-protective) behaviour.
        default_prior_trust = CONFIG.get("consolidation_graph_prior_trust", 1.0)
        for cid, record in self._node_records.items():
            # Strip any stale prior from a previous apply_to_graph call.
            record.observations = [o for o in record.observations
                                    if o.source not in ("__graph_prior__", "__graph_prior_vote__",
                                                         "__graph_prior_trend__")]
            existing = graph.get_node(cid)
            if existing is not None:
                prior_trust = existing.__dict__.get(
                    "consolidation_meta", {}).get(
                        "sum_real_trust", default_prior_trust)
                prior_fields = self._node_to_fields(existing)
                # TODO #11 fix: MAJORITY_VOTE fields are handled separately
                # below via per-candidate echoes — remove them here so they
                # aren't ALSO merged through the flat sum_real_trust-weighted
                # path, which would hand the current winner credit for
                # every vote ever cast (including votes for other
                # candidates that lost — the bug test_consolidation_order.py
                # now regression-tests).
                existing_vote_totals = existing.__dict__.get(
                    "consolidation_meta", {}).get("vote_totals", {})
                for vf in NODE_MAJORITY_VOTE_FIELDS:
                    prior_fields.pop(vf, None)
                # TODO #13 fix, second bug found verifying the first: same
                # reasoning as MAJORITY_VOTE above, but for a different
                # underlying reason. existing.<trend field> is this node's
                # current CONSOLIDATED value — once _temporal_trend()
                # starts extrapolating (>=3 points), that value is already
                # an extrapolation, not a raw data point. Leaving it in
                # prior_fields fed it back into _temporal_trend() as though
                # it WERE a fresh raw observation at existing.timestamp,
                # compounding a little more extrapolation on top of an
                # already-extrapolated number every single session — and
                # since how many such compounding rounds have happened by
                # a given point depends on how many sessions were
                # *processed* before it, not just on real timestamps, this
                # made the result order-dependent (verified: 5! permutations
                # of 5 sessions gave dozens of distinct final values instead
                # of one). trend_history's echoes below carry the real raw
                # (timestamp, value) points instead, so this contaminated
                # shortcut isn't needed.
                for tf in NODE_TREND_FIELDS:
                    prior_fields.pop(tf, None)
                prior_obs = Observation(
                    source="__graph_prior__",
                    timestamp=existing.timestamp,
                    trust=prior_trust,
                    physical=getattr(existing, "__dict__", {}).get("raw_physical_scores", {}),
                    fields=prior_fields,
                )
                # Insert at position 0 so it acts as the baseline that new
                # observations supplement, not override.
                record.observations.insert(0, prior_obs)
                # One tiny synthetic observation per historically-seen
                # candidate, each carrying ONLY that field at exactly its
                # own accumulated weight — the correct sufficient statistic
                # for MAJORITY_VOTE (a vector over candidates), as opposed
                # to TRUST_WEIGHTED_MEAN's scalar (S, W) pair. FieldMerger.
                # _majority_vote() already skips observations missing the
                # field (None), so these single-field echoes only ever
                # contribute to their own candidate's tally.
                for vf in NODE_MAJORITY_VOTE_FIELDS:
                    totals = existing_vote_totals.get(vf)
                    if totals:
                        for candidate, weight in totals.items():
                            record.observations.insert(0, Observation(
                                source="__graph_prior_vote__",
                                timestamp=existing.timestamp,
                                trust=weight,
                                fields={vf: candidate}))
                    else:
                        # No recorded vote history for this field (e.g. the
                        # node was written outside the consolidator) — fall
                        # back to a single echo of the current value at the
                        # conservative default, same as the old behaviour.
                        current_val = getattr(existing, vf, None)
                        if current_val not in (None, ""):
                            record.observations.insert(0, Observation(
                                source="__graph_prior_vote__",
                                timestamp=existing.timestamp,
                                trust=default_prior_trust,
                                fields={vf: current_val}))
                # TODO #13 fix: one synthetic single-field observation per
                # stored (timestamp, value) point in this node's bounded
                # trend history (see the NODE_TREND_FIELDS comment above and
                # the trend_history write-back near the end of this method),
                # so FieldMerger._temporal_trend() actually sees enough
                # points across sessions to fit a slope instead of always
                # collapsing to exactly 2 (one prior + one new observation)
                # and silently degrading to TEMPORAL_LATEST. trust is
                # irrelevant to _temporal_trend (it only reads timestamps
                # and values) but must still be excluded from
                # sum_real_trust — see the source-exclusion tuples above and
                # in the sum_real_trust computation below.
                existing_trend_history = existing.__dict__.get(
                    "consolidation_meta", {}).get("trend_history", {})
                for tf in NODE_TREND_FIELDS:
                    for ts, val in existing_trend_history.get(tf, []):
                        record.observations.insert(0, Observation(
                            source="__graph_prior_trend__",
                            timestamp=ts,
                            trust=default_prior_trust,
                            fields={tf: val}))

        # ── Also seed EdgeRecords with existing graph edges ───────────────────
        for (src, tgt), record in self._edge_records.items():
            record.observations = [o for o in record.observations
                                    if o.source not in ("__graph_prior__", "__graph_prior_vote__")]
            existing_edge = graph.get_edge(src, tgt)
            if existing_edge is not None:
                prior_trust = existing_edge.__dict__.get(
                    "consolidation_meta", {}).get(
                        "sum_real_trust", default_prior_trust)
                prior_fields = self._edge_to_fields(existing_edge)
                existing_vote_totals = existing_edge.__dict__.get(
                    "consolidation_meta", {}).get("vote_totals", {})
                for vf in EDGE_MAJORITY_VOTE_FIELDS:
                    prior_fields.pop(vf, None)
                prior_obs = Observation(
                    source="__graph_prior__",
                    timestamp=existing_edge.timestamp,
                    trust=prior_trust,
                    fields=prior_fields,
                )
                record.observations.insert(0, prior_obs)
                for vf in EDGE_MAJORITY_VOTE_FIELDS:
                    totals = existing_vote_totals.get(vf)
                    if totals:
                        for candidate, weight in totals.items():
                            record.observations.insert(0, Observation(
                                source="__graph_prior_vote__",
                                timestamp=existing_edge.timestamp,
                                trust=weight,
                                fields={vf: candidate}))
                    else:
                        current_val = getattr(existing_edge, vf, None)
                        if current_val not in (None, ""):
                            record.observations.insert(0, Observation(
                                source="__graph_prior_vote__",
                                timestamp=existing_edge.timestamp,
                                trust=default_prior_trust,
                                fields={vf: current_val}))

        def _vote_totals_from_observations(observations, vote_fields) -> Dict[str, Dict[Any, float]]:
            """TODO #11: recompute the per-candidate accumulated-weight
            vector for each MAJORITY_VOTE field from this call's full
            observation list (prior echoes + real observations alike —
            each already carries the correct weight to contribute)."""
            out: Dict[str, Dict[Any, float]] = {}
            for vf in vote_fields:
                tally: Dict[Any, float] = {}
                for o in observations:
                    val = o.fields.get(vf)
                    if val in (None, ""):
                        continue
                    tally[val] = tally.get(val, 0.0) + o.trust
                if tally:
                    out[vf] = tally
            return out

        def _trend_history_from_observations(
                observations, trend_fields) -> Dict[str, List[Tuple[float, float]]]:
            """TODO #13: bounded (timestamp, value) window per TEMPORAL_TREND
            field, recomputed from this call's full observation list —
            __graph_prior_trend__ echoes already carry the old window
            forward (mirrors how _vote_totals_from_observations doesn't need
            a separate "old" parameter either), so this only needs to add
            this session's genuinely new real points and re-truncate.
            __graph_prior__ itself is skipped even though it still carries
            these fields (unlike MAJORITY_VOTE fields, TEMPORAL_TREND fields
            are not popped from prior_fields) — its point is redundant with
            the trend_history echoes and would just add a near-duplicate.
            Keeping the *most recent by timestamp* (not "most recently
            processed") is what makes this order-independent: the final
            window is a pure function of the set of (timestamp, value)
            pairs ever seen, not of session arrival order.
            """
            out: Dict[str, List[Tuple[float, float]]] = {}
            for tf in trend_fields:
                points = []
                for o in observations:
                    if o.source == "__graph_prior__":
                        continue
                    if o.source == "__graph_prior_vote__":
                        continue
                    val = o.fields.get(tf)
                    if val in (None, ""):
                        continue
                    try:
                        points.append((float(o.timestamp), float(val)))
                    except (TypeError, ValueError):
                        continue
                if points:
                    points.sort(key=lambda p: p[0])
                    out[tf] = points[-NODE_TREND_HISTORY_WINDOW:]
            return out

        # 1. Consolidate nodes
        consolidated_nodes = {}
        for cid, record in self._node_records.items():
            node = (self._fields_to_node(cid, record.observations[0])
                    if len(record.observations) == 1
                    else self._node_consolidator.consolidate(record))
            self._sync_temporal_profile(
                node, record.observations, existing_node=graph.get_node(cid))
            if len(record.observations) > 1:
                stats.merges_applied += len(record.observations) - 1
            meta = node.__dict__.setdefault("consolidation_meta", {})
            # Running total of every real (non-prior) trust weight ever behind
            # this node, carried forward via the prior seeded above. Doubles
            # as the sufficient statistic for the order-independence fix and
            # as a provenance signal ("how much has actually corroborated
            # this value, cumulatively") independent of which source arrived
            # in which session.
            #
            # NOT capped — see the long comment on CONFIG
            # "consolidation_sum_real_trust_cap" for why a hard cap here
            # (TODO #12's first attempt) provably reintroduces the exact
            # order-dependence #9 exists to eliminate. __graph_prior_vote__
            # and __graph_prior_trend__ are both excluded: they're separate
            # per-field bookkeeping mechanisms (see vote_totals/
            # trend_history below) and must NOT also feed the scalar
            # TRUST_WEIGHTED_MEAN sufficient statistic, or a node with N
            # majority-vote/trend fields would have its sum_real_trust
            # inflated ~Nx (each field's echoes separately re-add roughly
            # the same historical weight).
            meta["sum_real_trust"] = sum(
                o.trust for o in record.observations
                if o.source not in ("__graph_prior_vote__", "__graph_prior_trend__"))
            meta["vote_totals"] = _vote_totals_from_observations(
                record.observations, NODE_MAJORITY_VOTE_FIELDS)
            meta["trend_history"] = _trend_history_from_observations(
                record.observations, NODE_TREND_FIELDS)
            # Second bug found verifying the first (TODO #13): merge()
            # computed maturity_score/etc. above from the FULL observations
            # list — old trend_history echoes (<= window points) PLUS this
            # session's own new point, i.e. up to window+1 points, not the
            # windowed set. Which point ends up being the "+1 extra" one
            # depends on which session happens to be processed LAST, so
            # np.polyfit's slope came out slightly order-dependent once
            # more real sessions had accumulated than the window holds
            # (verified: 7 sessions, window=5 -> 2 distinct results across
            # 30 shuffles, a small but real and order-dependent spread).
            # Recomputing directly from the just-written, already
            # order-independent trend_history window (exactly window
            # points, never window+1) fixes this — overrides whatever
            # merge() put on the node above for these specific fields only.
            for tf in NODE_TREND_FIELDS:
                hist = meta["trend_history"].get(tf)
                if hist:
                    setattr(node, tf, FieldMerger._temporal_trend(
                        [v for _, v in hist], [ts for ts, _ in hist]))
            if meta.get("conflicts"):
                stats.conflicts_found += len(meta["conflicts"])
                stats.high_conflict_nodes.append(cid)
            consolidated_nodes[cid] = node
        # 2. Replace nodes in graph.
        for old_id in self._id_redirects:
            graph.remove_node(old_id) if hasattr(graph, "remove_node") else graph.nodes.pop(old_id, None)
        # Bug fix (found auditing TODO #15): this used to be a bare
        # `graph.nodes[cid] = node`, bypassing RuntimeGraph.add_node()
        # entirely — which meant temporal_sorted and domain_clusters were
        # NEVER updated for any node that ever went through
        # apply_to_graph() (i.e. every node from the real ingestion
        # pipeline, TODO #1). Concretely verified: temporal_sorted stayed
        # `[]` forever across sessions, which made
        # AncestryEnforcer.find_best_predecessor() (called just below, step
        # 5) always return None — every node was silently admitted as an
        # "additional primordial root" and ancestry_edges_synthesised
        # stayed 0 no matter how many sessions ran. domain_clusters stayed
        # `{}` the same way, silently breaking anything that reads it
        # (e.g. domain-scoped phantom-void scanning).
        #
        # remove_node()+add_node() per node would also fix this, but
        # remove_node() calls _rebuild_incoming_cache() (an O(E) scan)
        # internally — doing that once per node in a large batch would be
        # O(N·E), and apply_to_graph() already rebuilds that cache once
        # after all edges are written (step 5 below), so we inline just the
        # two structural indices remove_node()/add_node() would touch here,
        # without repeating the cache rebuild N times.
        for cid, node in consolidated_nodes.items():
            old_node = graph.nodes.get(cid)
            if old_node is not None:
                old_ts = getattr(old_node, "timestamp", None)
                if old_ts is not None:
                    graph.temporal_sorted = [
                        (ts, nid) for ts, nid in graph.temporal_sorted
                        if not (nid == cid and ts == old_ts)
                    ]
                old_domain = getattr(old_node, "domain", None)
                if old_domain and old_domain in graph.domain_clusters:
                    try:
                        graph.domain_clusters[old_domain].remove(cid)
                    except ValueError:
                        pass
            graph.nodes[cid] = node
            bisect.insort(graph.temporal_sorted, (node.timestamp, cid))
            graph.domain_clusters.setdefault(node.domain, []).append(cid)
            # Currently a no-op in practice: canonical_key is a dynamic
            # __dict__ attribute attached by CanonicalNodeAssembler at
            # ingestion time, and NodeConsolidator._node_to_fields()/
            # _fields_to_node() do not carry it through consolidation (found
            # auditing this same fix, not yet corrected — separate issue
            # from the temporal_sorted/domain_clusters one above). Calling
            # this is still correct and forward-compatible: it's documented
            # as safe to call repeatedly and no-ops cleanly when
            # canonical_key is absent, so it costs nothing today and needs
            # no further change if canonical_key preservation through
            # consolidation is fixed later.
            if hasattr(graph, "_index_canonical_key"):
                graph._index_canonical_key(node)
        # 3. Redirect stale edge keys ATOMICALLY.
        #    Mutating only edge.source_id / edge.target_id leaves the dict key
        #    stale: graph.edges[(old_id, X)] points to an edge with source_id=new_id.
        #    We must rebuild the entire dict so keys always match field values.
        if self._id_redirects:
            new_edge_dict: Dict = {}
            for (src, tgt), edge in list(
                    (graph.edges if isinstance(graph.edges, dict) else {}).items()):
                new_src = self._id_redirects.get(src, src)
                new_tgt = self._id_redirects.get(tgt, tgt)
                if new_src == new_tgt:
                    continue  # self-loop created by redirect — drop it
                edge.source_id = new_src
                edge.target_id = new_tgt
                new_edge_dict[(new_src, new_tgt)] = edge
            graph.edges = new_edge_dict
            # Rebuild _incoming_ids from the remapped edges
            if hasattr(graph, "_rebuild_incoming_cache"):
                graph._rebuild_incoming_cache()
            # Clean temporal_sorted and domain_clusters
            redirected_away = set(self._id_redirects.keys())
            graph.temporal_sorted = [
                (ts, nid) for ts, nid in graph.temporal_sorted
                if nid not in redirected_away
            ]
            for domain in list(graph.domain_clusters.keys()):
                graph.domain_clusters[domain] = [
                    nid for nid in graph.domain_clusters[domain]
                    if nid not in redirected_away
                ]
        # 4. Consolidate edges and write them into the graph.
        #
        # Bug fix: this used to be gated on `len(record.observations) > 1`,
        # i.e. only edges independently corroborated by *two or more*
        # documents/sessions were ever written to the graph at all — an edge
        # seen exactly once (the overwhelming majority on any first ingestion
        # pass into a fresh graph, since the seeding step above only adds a
        # prior when the edge *already exists* in the graph) was silently
        # dropped. EdgeConsolidator.consolidate() already handles a
        # single-observation record correctly (every merge strategy degrades
        # to "return the one value" for a length-1 list), so there was no
        # correctness reason for the guard — call it unconditionally.
        for key, record in self._edge_records.items():
            cons_edge = self._edge_consolidator.consolidate(
                record, corpus_document_count=self._corpus_document_count)
            existing_edge = graph.get_edge(*key)
            old_history = getattr(existing_edge, "__dict__", {}).get(
                "source_observations", []) if existing_edge is not None else []
            new_history = getattr(cons_edge, "__dict__", {}).get(
                "source_observations", [])
            merged_history: Dict[Tuple[str, Optional[float], str, Tuple[str, ...]], Dict[str, Any]] = {}
            for item in list(old_history if isinstance(old_history, list) else []) + list(
                    new_history if isinstance(new_history, list) else []):
                if not isinstance(item, dict):
                    continue
                try:
                    item_timestamp = float(item["timestamp"]) if item.get("timestamp") is not None else None
                except (TypeError, ValueError):
                    item_timestamp = None
                evidence_key = tuple(str(value) for value in item.get("evidence", [])
                                     if isinstance(value, str))
                item_key = (str(item.get("source", "")), item_timestamp,
                            str(item.get("relationship_type", "related")), evidence_key)
                if not item_key[0]:
                    continue
                merged = dict(merged_history.get(item_key, {}))
                for name, value in item.items():
                    if value not in (None, "", []):
                        merged[name] = value
                merged_history[item_key] = merged
            cons_edge.__dict__["source_observations"] = sorted(
                merged_history.values(),
                key=lambda item: (item.get("timestamp") is None,
                                  item.get("timestamp") or 0.0,
                                  item.get("source", "")),
            )
            meta = cons_edge.__dict__.setdefault("consolidation_meta", {})
            # NOT capped — see the long comment on CONFIG
            # "consolidation_sum_real_trust_cap" for why a hard cap here
            # (TODO #12's first attempt) provably reintroduces the exact
            # order-dependence #9 exists to eliminate. __graph_prior_vote__
            # is still excluded: those are a separate per-field bookkeeping
            # mechanism (see vote_totals below) and must NOT also feed the
            # scalar TRUST_WEIGHTED_MEAN sufficient statistic, or an edge
            # with a majority-vote field would have its sum_real_trust
            # inflated (the field's echoes separately re-add roughly the
            # same historical weight).
            meta["sum_real_trust"] = sum(
                o.trust for o in record.observations
                if o.source != "__graph_prior_vote__")
            meta["vote_totals"] = _vote_totals_from_observations(
                record.observations, EDGE_MAJORITY_VOTE_FIELDS)
            # TODO #16 fix ("Вариант 2"): if ANY observation behind this edge
            # (this session's or a previous one's, since the prior seeded
            # above round-trips through _edge_to_fields the same way) was
            # tagged is_family by BucketIndex.find_connections(), register
            # the link on the graph. An OR, not a vote or a trust-weighted
            # merge — one corroborating document is enough evidence that two
            # nodes are worth reading as one "iceberg"; it doesn't need
            # repeated confirmation the way a numeric value does.
            if any(o.fields.get("is_family") for o in record.observations):
                cons_edge.__dict__["is_family"] = True
                graph.union_family(key[0], key[1])
                stats.family_links_created += 1
            self._replace_edges_in_graph(graph, key[0], key[1], cons_edge)

        self._apply_influence_observations(graph)

        # 5. TODO #15 fix: enforce the ancestry invariant.
        #
        # apply_to_graph() writes nodes directly into graph.nodes (step 2
        # above) rather than through RuntimeGraph.add_node_checked(), which
        # is the graph's own documented entry point for this — so
        # AncestryEnforcer never ran on the real ingestion path
        # (IngestionPipeline -> apply_to_graph(), the one TODO #1 wired up),
        # and a batch of documents that each introduce a new idea with no
        # prior art already in the graph would silently produce orphan
        # nodes. Run enforcement now that both nodes AND edges from this
        # call are fully committed to the graph — has_incoming() below
        # therefore already sees same-batch cross-references correctly
        # (e.g. A and B both arriving in this call with a real B->A edge),
        # without needing separate batch_edges plumbing: by this point
        # "in the graph" and "in this batch" are the same thing.
        # Pre-existing nodes that already had ancestry established (on an
        # earlier call, whether synthesised here or real from extraction)
        # short-circuit immediately inside enforce() via has_incoming(), so
        # checking every consolidated node each call is cheap, not just
        # correct.
        for cid, node in consolidated_nodes.items():
            synthetic = ANCESTRY_ENFORCER.enforce(node, graph)
            if synthetic is not None:
                graph.add_edge(synthetic)
                stats.ancestry_edges_synthesised += 1

        self._rebuild_label_support(graph)

        stats.nodes_after = len(graph.nodes)
        stats.edges_after = len(graph.edges) if isinstance(graph.edges, dict) else 0
        logger.info(
            f"[GraphConsolidator] Nodes: {stats.nodes_before} -> {stats.nodes_after} "
            f"(-{stats.nodes_before - stats.nodes_after}), "
            f"Merges: {stats.merges_applied}, Conflicts: {stats.conflicts_found}, "
            f"Ancestry edges synthesised: {stats.ancestry_edges_synthesised}, "
            f"Family links created: {stats.family_links_created}")
        return stats

    def _apply_influence_observations(self, graph) -> None:
        """Attach non-causal source signals to known nodes and their existing edges."""
        if not self._influence_records:
            return
        factor_fields = (
            "limitation_resolution", "citation_link", "investment_correlation",
            "social_correlation", "inhibitory_force")

        def merge_measurements(existing: Dict[str, Any], incoming: Dict[str, Any]) -> None:
            """Merge newly grounded numeric measurements into a repeated source receipt."""
            records = list(existing.get("parameter_observations", []) or [])
            keys = {
                (item.get("factor"), item.get("raw_value"), item.get("unit"), item.get("evidence"))
                for item in records if isinstance(item, dict)
            }
            for item in incoming.get("parameter_observations", []) or []:
                if not isinstance(item, dict):
                    continue
                key = (item.get("factor"), item.get("raw_value"), item.get("unit"), item.get("evidence"))
                if key not in keys:
                    records.append(dict(item))
                    keys.add(key)
            existing["parameter_observations"] = records

        for target_id, incoming in self._influence_records.items():
            node = graph.get_node(target_id)
            if node is None:
                continue
            node_history = getattr(node, "__dict__", {}).setdefault(
                "influence_observations", [])
            if not isinstance(node_history, list):
                node_history = []
                node.__dict__["influence_observations"] = node_history
            node_records = {
                (str(point.get("source_identity") or point.get("source", "")).casefold(),
                 str(point.get("signal_type", "")).casefold(),
                 str(point.get("evidence", ""))): point
                for point in node_history if isinstance(point, dict)
            }
            for point in incoming:
                key = (str(point.get("source_identity", "")).casefold(),
                       str(point.get("signal_type", "")).casefold(),
                       str(point.get("evidence", "")))
                if key not in node_records:
                    node_history.append(dict(point))
                    node_records[key] = node_history[-1]
                else:
                    merge_measurements(node_records[key], point)

            for edge in (graph.edges.values() if isinstance(graph.edges, dict) else []):
                if target_id not in (edge.source_id, edge.target_id):
                    continue
                edge_history = getattr(edge, "__dict__", {}).get("source_observations", [])
                if not isinstance(edge_history, list):
                    edge_history = []
                signal_history = getattr(edge, "__dict__", {}).setdefault(
                    "signal_observations", [])
                if not isinstance(signal_history, list):
                    signal_history = []
                    edge.__dict__["signal_observations"] = signal_history
                signal_keys = {
                    (str(item.get("source_identity") or item.get("source", "")).casefold(),
                     str(item.get("signal_type", "")).casefold(),
                     str(item.get("evidence", "")))
                    for item in signal_history + edge_history
                    if isinstance(item, dict) and item.get("signal_type")
                }
                signal_history_keys = {
                    (str(item.get("source_identity") or item.get("source", "")).casefold(),
                     str(item.get("signal_type", "")).casefold(),
                     str(item.get("evidence", "")))
                    for item in signal_history if isinstance(item, dict)
                }
                for point in incoming:
                    key = (str(point.get("source_identity", "")).casefold(),
                           str(point.get("signal_type", "")).casefold(),
                           str(point.get("evidence", "")))
                    if key not in signal_keys:
                        signal_point = {
                            **point,
                            "relationship_type": None,
                            "semantic_similarity": edge.semantic_similarity,
                            "temporal_proximity": edge.temporal_proximity,
                        }
                        signal_history.append(signal_point)
                        signal_keys.add(key)
                        signal_history_keys.add(key)
                    else:
                        existing = next((item for item in signal_history + edge_history
                                         if isinstance(item, dict)
                                         and (str(item.get("source_identity") or item.get("source", "")).casefold(),
                                              str(item.get("signal_type", "")).casefold(),
                                              str(item.get("evidence", ""))) == key), None)
                        if existing is not None:
                            merge_measurements(existing, point)
                            if key not in signal_history_keys:
                                signal_history.append({**existing})
                                signal_history_keys.add(key)
                all_points = [item for item in edge_history + signal_history
                              if isinstance(item, dict)]
                for field_name in factor_fields:
                    setattr(edge, field_name,
                            aggregate_unique_source_prevalence(
                                all_points, field_name, self._corpus_document_count))
                unique_signal_sources: Dict[str, set] = defaultdict(set)
                for item in edge_history + signal_history:
                    if not isinstance(item, dict):
                        continue
                    signal_type = str(item.get("signal_type", "")).casefold()
                    source_identity = str(
                        item.get("source_identity") or item.get("source", "")).casefold()
                    if signal_type and source_identity:
                        unique_signal_sources[signal_type].add(source_identity)
                edge.__dict__["signal_counts"] = {
                    signal_type: len(source_ids)
                    for signal_type, source_ids in unique_signal_sources.items()
                }
                edge.__dict__["influence_corpus_size"] = int(self._corpus_document_count)
                edge.compute_total_weight()

    def conflict_report(self, top_n: int = 20) -> List[Dict]:
        report = []
        for cid, record in self._node_records.items():
            # Only count truly independent external sources — exclude prior
            real_obs = [o for o in record.observations
                        if o.source != "__graph_prior__"]
            if len(real_obs) < 2:
                continue
            # _compute_conflicts already filters __graph_prior__ internally
            conflicts = NodeConsolidator._compute_conflicts(record.observations)
            if conflicts:
                report.append({
                    "node_id":         cid,
                    "source_count":    len({o.source for o in real_obs}),
                    "conflict_fields": list(conflicts.keys()),
                    "worst_field":     max(conflicts, key=lambda k: conflicts[k]["std"]),
                    "worst_std":       max(c["std"] for c in conflicts.values()),
                    "details":         conflicts,
                })
        report.sort(key=lambda x: x["worst_std"], reverse=True)
        return report[:top_n]

    def get_node_history(self, canonical_id: str) -> List[Dict]:
        record = self._node_records.get(canonical_id)
        if not record:
            return []
        return [
            {"source": o.source, "timestamp": o.timestamp, "trust": o.trust,
             # Include 0.0 values — they can be meaningful (e.g. zero inhibitory_force).
             # Only skip None and empty strings.
             "fields": {k: v for k, v in o.fields.items()
                        if v is not None and v != "" and isinstance(v, (int, float, str))},
             "physical": dict(o.physical) if o.physical else {},
            }
            for o in sorted(record.observations, key=lambda o: o.timestamp)
            # Exclude consolidation bookkeeping — these aren't source events.
            if not _is_internal_consolidation_source(o.source)
        ]

    @staticmethod
    def _sync_temporal_profile(node, observations: List[Observation],
                               existing_node=None) -> None:
        """Persist the temporal *line* of a canonical node across runs.

        Consolidation carries scalar fields through a graph prior, but an
        interval and its density curve need the individual dated source
        events.  Keep a compact, deduplicated receipt list on the node and
        rebuild the curve from that list.  Thus an incremental launch extends
        an existing temporal object; it cannot flatten it back to the latest
        document's point.
        """
        source_history: List[Dict[str, Any]] = []
        if existing_node is not None:
            existing_sources = getattr(existing_node, "__dict__", {}).get(
                "source_observations", [])
            if isinstance(existing_sources, list):
                source_history.extend(
                    item for item in existing_sources
                    if isinstance(item, dict)
                    and not _is_internal_consolidation_source(item.get("source"))
                )

        for observation in observations:
            if _is_internal_consolidation_source(observation.source):
                continue
            try:
                timestamp = float(observation.timestamp)
            except (TypeError, ValueError):
                timestamp = 0.0
            fields = observation.fields if isinstance(observation.fields, dict) else {}
            metadata = fields.get("source_metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            evidence = fields.get("source_evidence", fields.get("evidence", []))
            if isinstance(evidence, str):
                evidence = [evidence]
            evidence = [str(quote)[:1000] for quote in evidence
                        if isinstance(quote, str) and quote.strip()][:5] \
                if isinstance(evidence, list) else []
            source_history.append({
                "source": str(observation.source),
                "timestamp": timestamp if timestamp > 0.0 else None,
                "year": _ts_to_year(timestamp) if timestamp > 0.0 else None,
                "trust": float(max(0.0, observation.trust)),
                "title": metadata.get("title"),
                "authors": list(metadata.get("authors", [])),
                "publication": metadata.get("publication"),
                "forum_topic": metadata.get("forum_topic"),
                "source_type": metadata.get("source_type", "unknown"),
                "publication_date": metadata.get("publication_date"),
                "date_evidence": metadata.get("date_evidence"),
                "source_url": metadata.get("source_url"),
                "evidence": evidence,
                "context": str(fields.get("source_context", ""))[:2000],
                "label_status_code": fields.get("label_status_code"),
                "label_status": label_status_name(fields.get("label_status_code")),
                "status_evidence": str(fields.get("status_evidence") or "")[:1000],
            })

        deduplicated_sources: Dict[Tuple[str, Optional[float], Tuple[str, ...]], Dict[str, Any]] = {}
        for point in source_history:
            try:
                timestamp = float(point["timestamp"]) if point.get("timestamp") is not None else None
                key = (str(point.get("source", "")), timestamp,
                       tuple(str(item) for item in point.get("evidence", []) if item))
            except (TypeError, ValueError):
                continue
            if not key[0]:
                continue
            old = deduplicated_sources.get(key)
            if old is None:
                deduplicated_sources[key] = dict(point)
                continue
            merged = dict(old)
            for field_name, value in point.items():
                if value not in (None, "", []) and merged.get(field_name) in (None, "", []):
                    merged[field_name] = value
            if point.get("label_status_code") is not None:
                merged["label_status_code"] = int(point["label_status_code"])
                merged["label_status"] = label_status_name(point["label_status_code"])
            if point.get("status_evidence"):
                merged["status_evidence"] = point["status_evidence"]
            merged["trust"] = max(float(old.get("trust", 0.0)), float(point.get("trust", 0.0)))
            deduplicated_sources[key] = merged
        source_points = sorted(
            deduplicated_sources.values(),
            key=lambda item: (item.get("timestamp") is None, item.get("timestamp") or 0.0),
        )
        node.__dict__["source_observations"] = source_points
        own_status_points = [point for point in source_points
                             if parse_label_status(point.get("label_status_code")) is not None]
        if own_status_points:
            latest_status = own_status_points[-1]
            node.__dict__["label_status_code"] = int(
                parse_label_status(latest_status["label_status_code"]))
            node.__dict__["label_status"] = label_status_name(
                latest_status["label_status_code"])
            node.__dict__["status_evidence"] = str(
                latest_status.get("status_evidence") or "")
        node.__dict__["label_support_timeline"] = build_label_support_timeline(source_points)
        if node.__dict__["label_support_timeline"]:
            node.__dict__["label_support_score"] = node.__dict__["label_support_timeline"][-1]["support_score"]

        history: List[Dict[str, Any]] = []
        if existing_node is not None:
            existing_history = getattr(existing_node, "__dict__", {}).get(
                "temporal_observations", [])
            if isinstance(existing_history, list):
                history.extend(item for item in existing_history if isinstance(item, dict))
        for observation in observations:
            if _is_internal_consolidation_source(observation.source):
                continue
            try:
                timestamp = float(observation.timestamp)
            except (TypeError, ValueError):
                continue
            if timestamp <= 0.0:
                continue
            history.append({
                "source": str(observation.source),
                "timestamp": timestamp,
                "trust": float(max(0.0, observation.trust)),
            })

        deduplicated: Dict[Tuple[str, float], Dict[str, Any]] = {}
        for point in history:
            try:
                key = (str(point["source"]), float(point["timestamp"]))
            except (KeyError, TypeError, ValueError):
                continue
            # Re-ingestion of the same source/date must not inflate density.
            old = deduplicated.get(key)
            if old is None or float(point.get("trust", 0.0)) > float(old.get("trust", 0.0)):
                deduplicated[key] = {**point, "source": key[0], "timestamp": key[1],
                                     "trust": float(point.get("trust", 0.0))}
        points = sorted(deduplicated.values(), key=lambda item: item["timestamp"])
        node.__dict__["temporal_observations"] = points
        if not points:
            return

        node.timestamp_start = points[0]["timestamp"]
        node.timestamp_end = points[-1]["timestamp"]
        node.timestamp = node.timestamp_end
        span = node.timestamp_end - node.timestamp_start
        min_span = CONFIG["interval_min_span_days"] * 24.0 * 3600.0
        if len(points) < 2 or span < min_span:
            node.temporal_kind = "point"
            node.temporal_density = []
            return

        node.temporal_kind = "interval"
        bin_years = max(float(getattr(node, "density_bin_years", 1.0) or 1.0), 1e-6)
        bin_seconds = bin_years * 365.25 * 24.0 * 3600.0
        n_bins = max(1, int(math.ceil(span / bin_seconds)))
        density = [0.0] * n_bins
        for point in points:
            fraction = ((point["timestamp"] - node.timestamp_start) / span) if span else 0.0
            index = min(n_bins - 1, max(0, int(fraction * n_bins)))
            density[index] += max(0.05, float(point.get("trust", 0.0)))
        node.temporal_density = density

    @staticmethod
    def _rebuild_label_support(graph) -> None:
        """Rebuild each label's support history, including dated refutations."""
        refutations_by_target: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
        support_by_target: Dict[str, Dict[Tuple[Any, ...], Dict[str, Any]]] = {}
        for (source_id, target_id), edge in getattr(graph, "edges", {}).items():
            aggregate_relation = str(
                getattr(edge, "relationship_type", "related")).casefold()
            source_node = graph.get_node(source_id) if hasattr(graph, "get_node") else None
            source_points = getattr(source_node, "__dict__", {}).get(
                "source_observations", []) if source_node is not None else []
            source_by_key = {}
            for point in source_points if isinstance(source_points, list) else []:
                if parse_label_status(point.get("label_status_code")) != LabelStatus.REFUTATION:
                    continue
                source_by_key[(str(point.get("source", "")), point.get("timestamp"))] = point

            edge_points = getattr(edge, "__dict__", {}).get("source_observations", [])
            if not isinstance(edge_points, list) or not edge_points:
                if aggregate_relation not in {"contradicts", "supports", "confirms", "proves"}:
                    continue
                source_private = getattr(source_node, "__dict__", {}) if source_node else {}
                source_observation = getattr(edge, "__dict__", {}).get(
                    "source_observation", {})
                edge_points = [{
                    "source": getattr(source_node, "provenance", ""),
                    "timestamp": getattr(edge, "timestamp", None),
                    "trust": source_private.get("source_trust", 0.5),
                    "confidence": getattr(edge, "confidence", 0.5),
                    "relationship_type": aggregate_relation,
                    "source_label_status_code": getattr(edge, "__dict__", {}).get(
                        "source_label_status_code", source_private.get("label_status_code")),
                    "evidence": list(getattr(edge, "evidence", [])),
                    "source_observation": source_observation,
                }]

            for edge_point in edge_points:
                if not isinstance(edge_point, dict):
                    continue
                relation = str(edge_point.get(
                    "relationship_type", aggregate_relation)).casefold()
                if relation not in {"contradicts", "supports", "confirms", "proves"}:
                    continue
                source = str(edge_point.get("source", ""))
                timestamp = edge_point.get("timestamp")
                source_point = edge_point.get("source_observation")
                if not isinstance(source_point, dict):
                    source_point = source_by_key.get((source, timestamp), {})
                status = parse_label_status(edge_point.get("source_label_status_code"))
                if status is None:
                    status = parse_label_status(source_point.get("label_status_code"))
                if status is None:
                    source_private = getattr(source_node, "__dict__", {}) if source_node else {}
                    status = parse_label_status(source_private.get("label_status_code"))
                if relation == "contradicts" and status != LabelStatus.REFUTATION:
                    continue
                if relation in {"supports", "confirms", "proves"} and status not in {
                        LabelStatus.FACT, LabelStatus.PROOF}:
                    continue

                metadata = source_point.get("source_metadata", {})
                metadata = metadata if isinstance(metadata, dict) else {}
                evidence = edge_point.get("evidence", []) or source_point.get("evidence", [])
                if isinstance(evidence, str):
                    evidence = [evidence]
                event_timestamp = timestamp or source_point.get("timestamp")
                source_metadata = metadata
                if not source_metadata:
                    source_metadata = edge_point.get("source_metadata", {})
                    source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
                if relation == "contradicts":
                    evidence = edge_point.get("evidence", []) or source_point.get("evidence", [])
                event = {
                    "source": source or str(source_point.get("source", "")),
                    "source_node_id": source_point.get("source_node_id", source_id),
                    "source_node_text": source_point.get(
                        "source_node_text", getattr(source_node, "text", "")),
                    "timestamp": event_timestamp,
                    "year": _ts_to_year(event_timestamp or 0.0),
                    "trust": edge_point.get("trust", source_point.get("trust", 0.5)),
                    "confidence": edge_point.get("confidence", getattr(edge, "confidence", 0.5)),
                    "label_status_code": int(LabelStatus.REFUTATION),
                    "label_status": label_status_name(LabelStatus.REFUTATION),
                    "source_label_status_code": int(status),
                    "status_evidence": source_point.get("status_evidence", ""),
                    "context": source_point.get("context", ""),
                    "evidence": [str(value)[:1000] for value in evidence
                                 if isinstance(value, str) and value.strip()][:5],
                    "title": source_metadata.get("title"),
                    "authors": list(source_metadata.get("authors", [])),
                    "publication": source_metadata.get("publication"),
                    "forum_topic": source_metadata.get("forum_topic"),
                    "source_type": source_metadata.get("source_type", "unknown"),
                    "publication_date": source_metadata.get("publication_date"),
                    "source_url": source_metadata.get("source_url"),
                    "source_metadata": source_metadata,
                    "is_refutation_event": relation == "contradicts",
                    "event_type": relation,
                }
                if not event["source"] or not event["evidence"]:
                    continue
                if relation in {"supports", "confirms", "proves"}:
                    if relation == "proves" and status != LabelStatus.PROOF:
                        continue
                    event["independence_key"] = (
                        source_metadata.get("source_url")
                        or "|".join(str(source_metadata.get(field) or "").casefold()
                                    for field in ("title", "publication"))
                        or event["source"]
                    )
                    event["source_label_status_code"] = int(status)
                    event["status_evidence"] = str(event["evidence"][0])
                    event["label_status_code"] = int(
                        LabelStatus.PROOF if relation == "proves" else LabelStatus.FACT)
                    event["label_status"] = label_status_name(event["label_status_code"])
                    key = (relation, event["independence_key"], event_timestamp,
                           event["source_node_id"], tuple(event["evidence"]))
                    support_by_target.setdefault(target_id, {})[key] = event
                    continue
                key = (event["source"], event_timestamp, event["source_node_id"],
                       tuple(event["evidence"]))
                refutations_by_target.setdefault(target_id, {})[key] = event

        for node_id, node in getattr(graph, "nodes", {}).items():
            own_points = getattr(node, "__dict__", {}).get("source_observations", [])
            if not isinstance(own_points, list):
                own_points = []
            refutations = list(refutations_by_target.get(node_id, {}).values())
            corroborations = list(support_by_target.get(node_id, {}).values())
            timeline = build_label_support_timeline(
                own_points, refutations, corroborations)
            node.__dict__["label_support_timeline"] = timeline
            if timeline:
                node.__dict__["label_support_score"] = timeline[-1]["support_score"]
                effective_status = parse_label_status(
                    timeline[-1].get("effective_label_status_code"))
                if effective_status is not None:
                    node.__dict__["effective_label_status_code"] = int(effective_status)
                    node.__dict__["label_status_code"] = int(effective_status)
                    node.__dict__["label_status"] = label_status_name(effective_status)
                    node.__dict__["status_evidence"] = str(
                        timeline[-1].get("status_evidence") or "")
            else:
                node.__dict__.pop("effective_label_status_code", None)

    @staticmethod
    def _node_to_fields(node) -> Dict[str, Any]:
        # Numeric score fields where 0.0 is the *default unset* value.
        # When the LLM does not extract a score, the node field stays 0.0.
        # Storing 0.0 as-is would include it in TRUST_WEIGHTED_MEAN and dilute
        # existing non-zero observations.  We store None instead, which all
        # FieldMerger strategies skip, preserving the "not observed" semantics.
        NUMERIC_SCORE_FIELDS = {
            "scientific_score", "investment_score", "social_score",
            "maturity_score", "readiness_score", "group_size_score",
            "dual_use_risk", "legal_risk_score", "export_control_risk",
            "strategic_value", "efficiency_plateau",
            "investment_last_round_usd",
            "sentiment_review_score", "sentiment_fiction_score",
            "sentiment_forum_score", "social_perception_score",
        }
        scalar_fields = [
            "text", "full_text", "node_type", "domain", "entity_type",
            "publication_date", "scientific_score", "investment_score",
            "social_score", "maturity_score", "readiness_score",
            "group_size_score", "dual_use_risk", "legal_risk_score",
            "export_control_risk", "strategic_value", "efficiency_plateau",
            "investment_rounds", "investment_total_usd", "investment_last_round_usd",
            "sentiment_review_score", "sentiment_fiction_score",
            "sentiment_forum_score", "social_perception_score", "forum_post_count",
        ]
        list_fields = ["investment_lead_investors", "solves_limitations",
                       "requires_node_ids", "enables_node_ids"]
        d: Dict[str, Any] = {}
        for f in scalar_fields:
            v = getattr(node, f, None)
            # Treat 0.0 on numeric score fields as "not extracted by LLM"
            # (all score fields default to 0.0 in KnowledgeNode).
            if f in NUMERIC_SCORE_FIELDS and isinstance(v, float) and v == 0.0:
                d[f] = None
            else:
                d[f] = v
        for f in list_fields:
            v = getattr(node, f, None)
            d[f] = list(v) if v else []
        private = getattr(node, "__dict__", {})
        d["source_evidence"] = list(private.get("source_evidence", []))
        d["source_context"] = str(private.get("source_context", ""))
        d["label_status_code"] = private.get("label_status_code")
        d["label_status"] = private.get("label_status")
        d["status_evidence"] = private.get("status_evidence", "")
        source_metadata = private.get("source_metadata", {})
        d["source_metadata"] = dict(source_metadata) if isinstance(source_metadata, dict) else {}
        emb = getattr(node, "embedding", None)
        if emb is not None:
            d["embedding"] = np.array(emb, dtype=np.float32)
        return d

    @staticmethod
    def _edge_to_fields(edge) -> Dict[str, Any]:
        private = getattr(edge, "__dict__", {})
        source_observations = list(private.get("source_observations", []) or [])
        source_observations.extend(
            item for item in (private.get("signal_observations", []) or [])
            if isinstance(item, dict))
        return {
            "semantic_similarity":    getattr(edge, "semantic_similarity", 0.0),
            "temporal_proximity":     getattr(edge, "temporal_proximity", 0.0),
            "limitation_resolution":  getattr(edge, "limitation_resolution", 0.0),
            "citation_link":          getattr(edge, "citation_link", 0.0),
            "investment_correlation": getattr(edge, "investment_correlation", 0.0),
            "social_correlation":     getattr(edge, "social_correlation", 0.0),
            "inhibitory_force":       getattr(edge, "inhibitory_force", 0.0),
            "confidence":             getattr(edge, "confidence", 0.0),
            "relationship_type":      getattr(edge, "relationship_type", "related"),
            "evidence":               list(getattr(edge, "evidence", [])),
            "factor_evidence":        dict(getattr(edge, "__dict__", {}).get("factor_evidence", {}) or {}),
            "source_observations":    source_observations,
            "signal_observations":    list(private.get("signal_observations", []) or []),
            "parameter_observations": list(private.get("parameter_observations", []) or []),
            "signal_counts":          dict(private.get("signal_counts", {}) or {}),
            "influence_corpus_size":  int(private.get("influence_corpus_size", 0) or 0),
            "source_identity":        getattr(edge, "__dict__", {}).get("source_identity"),
            "source_metadata":        dict(getattr(edge, "__dict__", {}).get("source_metadata", {}) or {}),
            "source_label_status_code": getattr(edge, "__dict__", {}).get(
                "source_label_status_code"),
            "source_observation": getattr(edge, "__dict__", {}).get(
                "source_observation", {}),
            # TODO #16 fix ("Вариант 2"): NOT a declared KnowledgeEdge
            # dataclass field — BucketIndex.find_connections() and
            # NodeEdgeAssembler.assemble_edge() set this as a dynamic
            # edge.__dict__["is_family"] attribute for FAMILY_THRESHOLD-tier
            # matches (see both for the full comment). It used to die right
            # here: _edge_to_fields() is the only place that turns a
            # KnowledgeEdge into something GraphConsolidator looks at, and
            # this dict didn't have a key for it, so apply_to_graph() never
            # saw it and RuntimeGraph.union_family() was never called from
            # the real ingestion path.
            "is_family":              bool(getattr(edge, "__dict__", {}).get("is_family", False)),
        }

    @staticmethod
    def _fields_to_node(canonical_id: str, obs: Observation) -> KnowledgeNode:
        f = obs.fields
        node = KnowledgeNode(
            id=canonical_id, text=f.get("text", ""),
            full_text=f.get("full_text", ""),
            node_type=f.get("node_type", "research"),
            domain=f.get("domain", "unknown"),
            entity_type=f.get("entity_type", "research_frontier"),
            provenance=obs.source, timestamp=obs.timestamp,
            publication_date=f.get("publication_date"),
            scientific_score=float(f.get("scientific_score") or 0.0),
            investment_score=float(f.get("investment_score") or 0.0),
            social_score=float(f.get("social_score") or 0.0),
            maturity_score=float(f.get("maturity_score") or 0.0),
            readiness_score=float(f.get("readiness_score") or 0.0),
            strategic_value=float(f.get("strategic_value") or 0.0),
            efficiency_plateau=float(f.get("efficiency_plateau") or 0.0),
            dual_use_risk=float(f.get("dual_use_risk") or 0.0),
            legal_risk_score=float(f.get("legal_risk_score") or 0.0),
            export_control_risk=float(f.get("export_control_risk") or 0.0),
        )
        # Restore embedding; if raw_physical_scores are present and the stored
        # embedding is only 407-dim (base), extend it to 416 via the real encoder.
        emb = f.get("embedding")
        if emb is not None:
            node.embedding = np.array(emb, dtype=np.float32)
        if obs.physical:
            phys_encoder = PhysicalSubstrateEncoder()
            clean_phys = {ax: float(np.clip(obs.physical.get(ax, 0.5), 0.0, 1.0))
                          for ax in PHYSICAL_AXIS_ORDER}
            node.__dict__["raw_physical_scores"] = clean_phys
            phys_vec = phys_encoder.build_physical_section(clean_phys)
            node.__dict__["feasibility"] = phys_encoder.feasibility_score(phys_vec)
            if node.embedding is not None and len(node.embedding) == NodeFeatureBuilder.TOTAL_DIM:
                node.embedding = phys_encoder.extend_node_features(node.embedding, clean_phys)
        return node

    @staticmethod
    def _replace_edges_in_graph(graph, src_id: str, tgt_id: str, cons_edge) -> None:
        """Replace all edges (src_id→tgt_id) with the consolidated one.

        Guards:
        - Self-loops (created when src/tgt mapped to the same canonical id) are
          dropped silently rather than inserted into the graph.
        - _incoming_ids cache is rebuilt after deletion so AncestryEnforcer
          remains consistent.
        """
        if src_id == tgt_id:
            # Drop self-loop: delete any existing (src, tgt) entries and return
            to_remove = [k for k, e in graph.edges.items()
                         if getattr(e, "source_id", None) == src_id
                         and getattr(e, "target_id", None) == tgt_id]
            for key in to_remove:
                del graph.edges[key]
                if hasattr(graph, "_deindex_edge"):
                    graph._deindex_edge(key)
            if to_remove and hasattr(graph, "_rebuild_incoming_cache"):
                graph._rebuild_incoming_cache()
            return
        edges_dict = graph.edges if isinstance(graph.edges, dict) else {}
        to_remove  = [k for k, e in edges_dict.items()
                      if (getattr(e, "source_id", None) == src_id and
                          getattr(e, "target_id", None) == tgt_id)]
        for key in to_remove:
            del graph.edges[key]
            if hasattr(graph, "_deindex_edge"):
                graph._deindex_edge(key)
        if hasattr(graph, "add_edge"):
            graph.add_edge(cons_edge)   # add_edge updates _incoming_ids + _adjacency
        else:
            graph.edges[(cons_edge.source_id, cons_edge.target_id)] = cons_edge


def print_conflict_report(consolidator: GraphConsolidator, top_n: int = 10) -> None:
    """Human-readable report of fields with high variance across sources."""
    report = consolidator.conflict_report(top_n=top_n)
    if not report:
        print("[Consolidation] No conflicts — all sources agree.")
        return
    print(f"\n{'='*60}")
    print(f"  CONFLICT REPORT ({len(report)} nodes)")
    print(f"{'='*60}")
    for item in report:
        print(f"\n  Node: {item['node_id']}")
        print(f"  Sources: {item['source_count']}")
        for fname, info in sorted(item["details"].items(),
                                   key=lambda x: x[1]["std"], reverse=True):
            vals_str = ", ".join(f"{s}={v:.1f}" for s, v in
                                  zip(info["sources"], info["values"]))
            print(f"    {fname}: std={info['std']:.2f}  [{vals_str}]")
    print(f"{'='*60}\n")


def build_consolidation_pipeline(llm_client, embedder=None,
                                  existing_nodes=None) -> Tuple:
    """
    Factory: creates a paired IngestionPipeline + GraphConsolidator.

    Usage:
        pipeline, consolidator = build_consolidation_pipeline(my_llm, my_embedder)
        for doc in documents:
            result = pipeline.ingest(doc.text, doc.source, doc.timestamp,
                                     source_type=doc.source_type)
            consolidator.observe_from_ingestion_result(result, doc.source_type)
        stats = consolidator.apply_to_graph(graph)
        print_conflict_report(consolidator)
    """
    return IngestionPipeline(llm_client, embedder, existing_nodes), GraphConsolidator()



# ══════════════════════════════════════════════════════════════════════════════
#  CONTINUOUS FEED ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class ContinuousFeedEngine:
    """
    Manifest §5, "Инкрементальное обучение (Online Learning)": adding a new
    node should nudge the already-trained model locally, not force a
    full-graph retrain. feed() finds the affected wave via
    RuntimeGraph.propagate_wave() — a genuinely self-terminating BFS that
    keeps relaying past a node only while that node's own local tension
    (Σ total_weight over its incident edges) actually shifts by more than
    continuous_feed_tension_epsilon — and takes one or more local gradient
    steps restricted to that wave, via the same _local_gradient_step()
    TemporalSplitTrainer._local_step() uses — see that function's docstring
    for what one step actually does.

    TODO #21: this used to bound the affected set with
    RuntimeGraph.k_hop_neighborhood(k=CONFIG["continuous_feed_k_hop"]) — a
    fixed-radius cutoff. That doesn't match "a node relays the change to
    its neighbours only if the change actually shifts its own weight or
    tension" (Manifest §5): a radius has no opinion about whether anything
    actually changed, so it either stops a real disturbance short (a strong
    edge two hops out gets truncated at k=3 for no principled reason) or
    keeps padding a dead wave with unaffected neighbours purely because
    they're within k hops. propagate_wave() replaces it — depth is now an
    emergent property of the disturbance itself, not a number chosen in
    advance. self._last_local_tension persists node -> its local-tension
    reading as of the last time a wave touched it, across feed() calls, so
    "did this shift" is judged against this node's own history rather than
    recomputed from nothing each time.

    Previously this class computed the affected-node set and zone
    membership updates correctly but never took an actual gradient step —
    self.opt was constructed and never called. MAX_STEPS (a class constant
    that was itself never read anywhere) is now used to bound how many
    local steps a single feed() call may take: when the affected wave is
    larger than max_local_nodes, it's split into that many chunks (seed
    nodes always included whole; only the wave's padding around each seed
    is what gets split) rather than either silently truncating real
    changed nodes out of the update or taking one huge, unbounded step
    over the whole wave. max_local_nodes doubles as propagate_wave()'s
    max_visits safety valve — a hard cap on the wave itself, not just on
    how the gradient step over it gets chunked, so a pathological cascade
    (e.g. a hub node whose tension keeps crossing epsilon) can't make one
    feed() call walk the entire graph.
    """
    TENSION_EPSILON = CONFIG["continuous_feed_tension_epsilon"]
    ETA_CONTINUOUS  = CONFIG["continuous_feed_eta_continuous"]
    MAX_STEPS       = CONFIG["continuous_feed_max_steps"]

    def __init__(self, model: Oracle1Model, loss_fn: Optional["Oracle1Loss"] = None,
                 feature_builder: Optional["NodeFeatureBuilder"] = None,
                 edge_builder: Optional["EdgeFeatureBuilder"] = None,
                 device=None, max_local_nodes: int = 500):
        self.model = model
        self.loss_fn = loss_fn
        self.fb = feature_builder
        self.eb = edge_builder
        self.device = device
        self.max_local_nodes = max_local_nodes
        # Local-tension memory the wave's termination test reads/writes —
        # see the class docstring. Node ids age out naturally (nothing ever
        # prunes this dict), which is fine: a stale entry for a node that
        # hasn't been touched in a long time just means the next wave that
        # reaches it compares against an old baseline, which is exactly
        # "did anything change since we last looked", the right question.
        self._last_local_tension: Dict[str, float] = {}
        if TORCH_AVAILABLE:
            import torch.optim as optim
            self.opt = optim.AdamW(model.parameters(), lr=self.ETA_CONTINUOUS, weight_decay=1e-5)

    def _local_tension(self, node_id: str, runtime_graph: "RuntimeGraph") -> float:
        """Σ total_weight over node_id's incident edges — the same 'weight
        or tension' quantity the manifest names as the propagation gate.
        Uses get_neighbours(), which is an O(degree) adjacency-index lookup
        (TODO #20), not a graph-wide scan."""
        return sum(edge.total_weight for _, edge in runtime_graph.get_neighbours(node_id))

    def _wave_gate(self, node_id: str, runtime_graph: "RuntimeGraph") -> bool:
        """propagate_wave()'s apply_fn: recompute this node's local tension,
        compare against its last-known reading, and report whether the
        wave should keep relaying past it. A node with no prior reading
        (never visited before, including every seed — a node's own arrival
        is itself the disturbance) always counts as changed."""
        current = self._local_tension(node_id, runtime_graph)
        previous = self._last_local_tension.get(node_id)
        self._last_local_tension[node_id] = current
        return previous is None or abs(current - previous) > self.TENSION_EPSILON

    def feed(self, new_nodes, new_edges, runtime_graph) -> Dict[str, Any]:
        if not TORCH_AVAILABLE: return {"status": "torch_not_available"}
        zone_updates = self._detect_zone_membership(new_nodes, runtime_graph)
        affected = runtime_graph.propagate_wave(
            [n.id for n in new_nodes],
            apply_fn=lambda nid: self._wave_gate(nid, runtime_graph),
            max_visits=self.max_local_nodes,
        )
        for zone_id, new_members in zone_updates.items():
            zone_node = runtime_graph.get_node(zone_id)
            if zone_node: affected.update(zone_node.contained_node_ids)

        # Local gradient step(s) over the affected wave — the part that was
        # previously missing entirely (self.opt existed, was never used).
        step_losses: List[float] = []
        if self.device is not None and self.loss_fn is not None and affected:
            seed = {n.id for n in new_nodes} & affected
            padding = list(affected - seed)
            chunk_budget = max(1, self.max_local_nodes - len(seed))
            chunks = ([padding[i:i + chunk_budget] for i in range(0, len(padding), chunk_budget)]
                      if padding else [[]])
            for chunk in chunks[:self.MAX_STEPS]:
                local_ids = seed | set(chunk)
                if not local_ids:
                    continue
                local_graph = runtime_graph.subgraph(local_ids)
                if not local_graph.nodes:
                    continue
                loss = _local_gradient_step(
                    self.model, self.loss_fn, local_graph, self.device,
                    feature_builder=self.fb, edge_builder=self.eb, optimizer=self.opt)
                if loss is not None:
                    step_losses.append(loss)

        return {"new_nodes": len(new_nodes), "new_edges": len(new_edges),
                "affected_nodes": len(affected), "zone_updates": len(zone_updates),
                "local_steps_taken": len(step_losses),
                "mean_step_loss": (sum(step_losses) / len(step_losses)) if step_losses else None}


    def _detect_zone_membership(self, new_nodes, runtime_graph) -> Dict[str, List[str]]:
        """
        Geometric field membership (Manifest §1): a zone is a region in
        (domain, time), not a semantic cluster, so membership is decided by
        RuntimeGraph.zones_covering_node() — a coordinate test — rather than
        embedding cosine-similarity against a zone centroid. A node whose
        (domain, timestamp) falls in more than one zone's field keeps every
        covering zone (recompute_zone_membership writes the full list to
        node.zone_ids); zone_id/acceleration_multiplier on the node are the
        derived strongest-zone / compounded-multiplier projection that
        scalar readers (NodeFeatureBuilder, PathStep, pressure fields) use.

        This no longer requires node.embedding at all — geometry doesn't
        need one — so nodes ingested before embedding assignment (or zones
        with too few members for a centroid) are still matched correctly.
        """
        zone_updates = runtime_graph.recompute_zone_membership(
            node_ids=[n.id for n in new_nodes])

        # citation_velocity_ratio: now that membership has changed, try to
        # replace each touched zone's LLM-guessed zone_multiplier with a
        # measured one (RuntimeGraph.measured_zone_multiplier); keep the LLM
        # prior until there's enough evidence (cold start). Re-running
        # recompute_zone_membership afterwards propagates any updated
        # multiplier into acceleration_multiplier for the zone's members.
        touched_zone_ids = list(zone_updates.keys())
        remeasured = False
        for zone_id in touched_zone_ids:
            zone = runtime_graph.zones.get(zone_id)
            if zone is None:
                continue
            measured = runtime_graph.measured_zone_multiplier(zone)
            if measured is not None and measured != zone.zone_multiplier:
                zone.zone_multiplier = measured
                zone_node = runtime_graph.nodes.get(zone.id)
                if zone_node is not None:
                    zone_node.zone_multiplier = measured
                remeasured = True
        if remeasured:
            runtime_graph.recompute_zone_membership(zone_ids=touched_zone_ids)
        return zone_updates




def _is_internal_consolidation_source(source: Any) -> bool:
    """Identify internal merge bookkeeping that is not source evidence."""
    return str(source or "").startswith("__graph_prior")

__all__ = ['MergeStrategy', 'NODE_FIELD_STRATEGIES', 'PHYSICAL_AXIS_STRATEGIES', 'EDGE_FIELD_STRATEGIES', 'NODE_MAJORITY_VOTE_FIELDS', 'EDGE_MAJORITY_VOTE_FIELDS', 'NODE_TREND_FIELDS', 'NODE_TREND_HISTORY_WINDOW', 'Observation', 'NodeRecord', 'EdgeRecord', 'FieldMerger', 'NodeConsolidator', 'EdgeConsolidator', 'ConsolidationStats', 'GraphConsolidator', 'print_conflict_report', 'build_consolidation_pipeline', 'ContinuousFeedEngine', '_is_internal_consolidation_source']
