"""Evidence-based edge factor derivation and source aggregation."""

from __future__ import annotations

from ._shared import *


INFLUENCE_FACTOR_FIELDS = {
    "citation": "citation_link",
    "data_use": "citation_link",
    "investment": "investment_correlation",
    "social_attention": "social_correlation",
    "fictional_hype": "social_correlation",
    "unmet_need": "social_correlation",
}

# Qualitative relation and influence labels are retained as provenance only.
# They never become numeric edge strengths without quoted measurements.


def relation_factor_evidence(relationship_type: str, has_grounded_quote: bool) -> Dict[str, float]:
    """Return no numeric score for a qualitative relationship label."""
    return {}


def derive_pair_metrics(source_node=None, target_node=None,
                        source_text: str = "", target_text: str = "",
                        source_timestamp: float = 0.0,
                        target_timestamp: float = 0.0) -> Dict[str, float]:
    """Derive pair similarity and temporal proximity from endpoint data."""
    def value(obj, name, fallback=None):
        if isinstance(obj, dict):
            return obj.get(name, fallback)
        return getattr(obj, name, fallback)

    semantic = 0.0
    source_embedding = value(source_node, "embedding")
    target_embedding = value(target_node, "embedding")
    if source_embedding is not None and target_embedding is not None:
        try:
            left = np.asarray(source_embedding, dtype=np.float32).reshape(-1)
            right = np.asarray(target_embedding, dtype=np.float32).reshape(-1)
            if left.shape == right.shape:
                denom = float(np.linalg.norm(left) * np.linalg.norm(right))
                if denom > 1e-8:
                    semantic = float(np.clip(np.dot(left, right) / denom, 0.0, 1.0))
        except Exception:
            semantic = 0.0
    if semantic == 0.0:
        left_tokens = set(re.findall(r"[^\W_]+", str(source_text).casefold(), re.UNICODE))
        right_tokens = set(re.findall(r"[^\W_]+", str(target_text).casefold(), re.UNICODE))
        semantic = len(left_tokens & right_tokens) / len(left_tokens | right_tokens) \
            if left_tokens and right_tokens else 0.0

    temporal = 0.0
    try:
        left_ts, right_ts = float(source_timestamp or 0.0), float(target_timestamp or 0.0)
        if left_ts > 0.0 and right_ts > 0.0:
            years_apart = abs(left_ts - right_ts) / (365.2425 * 24.0 * 60.0 * 60.0)
            temporal = math.exp(-years_apart / 50.0)
    except (TypeError, ValueError, OverflowError):
        temporal = 0.0
    return {
        "semantic_similarity": round(float(np.clip(semantic, 0.0, 1.0)), 6),
        "temporal_proximity": round(float(np.clip(temporal, 0.0, 1.0)), 6),
    }


def mean_unique_source_metric(points: List[Dict[str, Any]], field_name: str,
                              fallback: float = 0.0) -> float:
    """Average a pair property across unique source measurements."""
    values: Dict[str, Tuple[float, float]] = {}
    for point in points if isinstance(points, list) else []:
        if not isinstance(point, dict):
            continue
        source = str(point.get("source_identity") or point.get("source", "")).strip().casefold()
        if not source:
            continue
        try:
            value = float(np.clip(float(point.get(field_name)), 0.0, 1.0))
            trust = float(np.clip(float(point.get("trust", 0.0)), 0.0, 1.0))
        except (TypeError, ValueError):
            continue
        prior = values.get(source)
        if prior is None or trust > prior[1]:
            values[source] = (value, trust)
    denominator = sum(weight for _, weight in values.values())
    if denominator <= 1e-8:
        return round(float(np.clip(fallback, 0.0, 1.0)), 6)
    return round(sum(value * weight for value, weight in values.values()) / denominator, 6)


def aggregate_unique_source_prevalence(points: List[Dict[str, Any]], field_name: str,
                                       corpus_document_count: int = 0) -> float:
    """Measure a factor as the share of analyzed documents carrying its signal.

    A document contributes one observed event when it contains a qualitative
    signal. If it also contains an explicit normalized measurement, that
    quoted value replaces the binary event for that document. Each content
    identity contributes once, regardless of replay count.
    """
    per_source: Dict[str, Dict[str, Any]] = {}
    for point in points if isinstance(points, list) else []:
        if not isinstance(point, dict):
            continue
        source = str(point.get("source_identity") or point.get("source", "")).strip().casefold()
        if not source:
            continue
        state = per_source.setdefault(source, {"has_signal": False, "measurements": {}})
        signal_type = str(point.get("signal_type", "")).strip().casefold()
        if INFLUENCE_FACTOR_FIELDS.get(signal_type) == field_name:
            state["has_signal"] = True
        for measurement in point.get("parameter_observations", []) or []:
            if not isinstance(measurement, dict) or measurement.get("factor") != field_name:
                continue
            try:
                value = float(measurement.get("normalized_value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                continue
            evidence = str(measurement.get("evidence", ""))
            state["measurements"][(value, evidence)] = value
    contributions = []
    for state in per_source.values():
        measurements = list(state["measurements"].values())
        if measurements:
            contributions.append(sum(measurements) / len(measurements))
        elif state["has_signal"]:
            contributions.append(1.0)
    denominator = max(int(corpus_document_count or 0), len(per_source))
    if denominator <= 0 or not contributions:
        return 0.0
    return round(float(np.clip(sum(contributions) / denominator, 0.0, 1.0)), 6)
