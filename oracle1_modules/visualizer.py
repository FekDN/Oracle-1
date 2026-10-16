# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Live graph queries and the visualizer HTTP API."""
from __future__ import annotations

from ._shared import *
from .consolidation import _is_internal_consolidation_source
from .label_status import label_status_display, label_status_name, parse_label_status
from .time_utils import _ts_to_year
from .validation import ResponseValidator

# ══════════════════════════════════════════════════════════════════════════════
#  VISUALIZER QUERY LAYER — live queries against the raw graph, not an export.
#
#  A JSON export of the whole graph cannot exist at Oracle-1's real scale
#  (hundreds of billions of nodes don't fit in a file, let alone browser
#  memory) — and doesn't need to: oracle1_visualizer.html only ever shows
#  one node's 1-hop window at a time by design, so the right shape here is
#  a targeted, O(degree) query per interaction, not a bulk dump. The two
#  functions below ARE that query surface; serve_visualizer_api() exposes
#  them over HTTP so the (static, dependency-free) HTML page can fetch()
#  them directly instead of loading a file.
# ══════════════════════════════════════════════════════════════════════════════



def _node_to_visualizer_dict(
    node: "KnowledgeNode",
    consolidator: Optional["GraphConsolidator"] = None,
    canonical_assembler: Optional["CanonicalNodeAssembler"] = None,
) -> Dict[str, Any]:
    """One node's data in oracle1_visualizer.html's shape. Fidelity is
    tiered by what's available. Extracted quotes and source metadata are
    retained with each observation; legacy temporal-density buckets are
    explicitly marked as aggregates and never presented as source evidence.
    """
    def year_of(ts: float) -> Optional[int]:
        return _ts_to_year(ts) if ts else None

    def to_observation(source: str, timestamp: float, trust: Optional[float],
                       fields: Optional[Dict[str, Any]] = None,
                       source_type: str = "unknown") -> Optional[Dict[str, Any]]:
        fields = fields if isinstance(fields, dict) else {}
        metadata = fields.get("source_metadata")
        if not isinstance(metadata, dict):
            metadata = {key: fields.get(key) for key in (
                "title", "authors", "publication", "forum_topic", "source_type",
                "publication_date", "date_evidence", "source_url",
            )}
        evidence = fields.get("source_evidence", fields.get("evidence", []))
        if isinstance(evidence, str):
            evidence = [evidence]
        event_type = str(fields.get("event_type") or "observation").casefold()
        is_refutation_event = bool(fields.get("is_refutation_event"))
        is_status_change = event_type in {"refutation", "supports", "confirms", "proves"}
        if isinstance(evidence, list):
            source_title = str(metadata.get("title") or "")
            evidence = [ResponseValidator._clean_evidence_quote(quote)[:1000]
                        for quote in evidence if isinstance(quote, str) and quote.strip()]
            evidence = [quote for quote in evidence if (
                (is_status_change
                 or ResponseValidator._quote_has_supporting_context(quote, node.text))
                and not re.search(r"(?i)\bprotocol\b", source_title)
                and not ResponseValidator._quote_repeats_source_title(quote, source_title)
            )][:5]
        else:
            evidence = []
        if not evidence:
            # Do not leave a clickable date/trust marker after its only
            # supporting quote has been rejected as a label or metadata.
            return None
        context = fields.get("source_context", fields.get("context", ""))
        if (not is_status_change
                and not ResponseValidator._analysis_context_is_supported(context, node.text, evidence)):
            context = ""
        source_status = (parse_label_status(fields.get("source_label_status_code"))
                         or parse_label_status(fields.get("label_status_code"))
                         or parse_label_status(fields.get("label_status")))
        status = (parse_label_status(fields.get("effective_label_status_code"))
                  or source_status)
        return {
            "year": year_of(timestamp), "timestamp": float(timestamp) if timestamp else None,
            "source": str(source or ""),
            "source_type": metadata.get("source_type") or source_type or "unknown",
            "title": metadata.get("title"),
            "authors": list(metadata.get("authors", [])) if isinstance(metadata.get("authors", []), list) else [],
            "publication": metadata.get("publication"),
            "forum_topic": metadata.get("forum_topic"),
            "publication_date": metadata.get("publication_date"),
            "date_evidence": metadata.get("date_evidence"),
            "source_url": metadata.get("source_url"),
            "trust": round(float(trust), 3) if trust is not None else None,
            "label_status_code": int(status) if status is not None else None,
            "label_status": label_status_name(status) if status is not None else None,
            "label_status_display": label_status_display(status) if status is not None else None,
            "source_label_status_code": int(source_status) if source_status is not None else None,
            "source_label_status": label_status_name(source_status) if source_status is not None else None,
            "source_label_status_display": label_status_display(source_status) if source_status is not None else None,
            "event_type": event_type,
            "is_status_change": is_status_change,
            "status_evidence": str(fields.get("status_evidence") or "")[:1000],
            "support_score": fields.get("support_score"),
            "is_refutation_event": is_refutation_event,
            "source_node_id": fields.get("source_node_id"),
            "source_node_text": fields.get("source_node_text"),
            "evidence": evidence,
            "snippet": evidence[0] if evidence else "",
            "context": str(context)[:1200],
            "is_aggregate": False,
        }

    observations: List[Dict[str, Any]] = []
    source_records_seen = False
    label_timeline = getattr(node, "__dict__", {}).get("label_support_timeline", [])
    if isinstance(label_timeline, list) and label_timeline:
        source_records_seen = True
        for event in label_timeline:
            if not isinstance(event, dict):
                continue
            fields = dict(event)
            fields["source_evidence"] = event.get("evidence", [])
            fields["source_context"] = event.get("context", "")
            metadata = event.get("source_metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            fields["source_metadata"] = {
                **metadata,
                **{name: event.get(name) for name in (
                    "title", "authors", "publication", "forum_topic", "source_type",
                    "publication_date", "date_evidence", "source_url",
                ) if event.get(name) is not None},
            }
            item = to_observation(
                event.get("source", ""), event.get("timestamp") or 0.0,
                event.get("trust"), fields, event.get("source_type", "unknown"),
            )
            if item is not None:
                observations.append(item)
    if consolidator is not None and not (isinstance(label_timeline, list) and label_timeline):
        rec = consolidator._node_records.get(node.id)
        if rec and rec.observations:
            records = [o for o in rec.observations
                       if not _is_internal_consolidation_source(o.source)]
            source_records_seen = bool(records)
            observations = [item for item in (
                to_observation(o.source, o.timestamp, o.trust, o.fields)
                for o in records
            ) if item is not None]
    if not source_records_seen and not observations and canonical_assembler is not None:
        obs_list = canonical_assembler.observations.get(node.id)
        if obs_list:
            records = [o for o in obs_list
                       if not _is_internal_consolidation_source(getattr(o, "source", ""))]
            source_records_seen = bool(records)
            observations = [item for item in (
                to_observation(
                    o.source, o.timestamp, o.trust,
                    getattr(o, "delta_fields", {}), getattr(o, "source_type", "unknown"),
                ) for o in records
            ) if item is not None]
    if not source_records_seen and not observations:
        persisted = getattr(node, "__dict__", {}).get("source_observations", [])
        if isinstance(persisted, list) and persisted:
            records = [item for item in persisted
                       if isinstance(item, dict)
                       and not _is_internal_consolidation_source(item.get("source"))]
            source_records_seen = bool(records)
            observations = [item for item in (
                to_observation(
                    record.get("source", ""), record.get("timestamp") or 0.0,
                    record.get("trust"), record, record.get("source_type", "unknown"),
                ) for record in records
            ) if item is not None]
    if (not source_records_seen and not observations
            and getattr(node, "__dict__", {}).get("source_metadata")
            and not _is_internal_consolidation_source(node.provenance)):
        source_records_seen = True
        item = to_observation(
            node.provenance, node.timestamp,
            getattr(node, "__dict__", {}).get("source_trust"),
            {
                "source_evidence": getattr(node, "__dict__", {}).get("source_evidence", []),
                "source_context": getattr(node, "__dict__", {}).get("source_context", ""),
                "source_metadata": getattr(node, "__dict__", {}).get("source_metadata", {}),
            },
        )
        observations = [item] if item is not None else []
    if not source_records_seen and not observations:
        persisted_timeline = getattr(node, "__dict__", {}).get("temporal_observations", [])
        if isinstance(persisted_timeline, list) and persisted_timeline:
            records = [item for item in persisted_timeline
                       if isinstance(item, dict)
                       and not _is_internal_consolidation_source(item.get("source"))]
            source_records_seen = bool(records)
            observations = [item for item in (
                to_observation(
                    record.get("source", ""), record.get("timestamp") or 0.0,
                    record.get("trust"), record, record.get("source_type", "unknown"),
                ) for record in records
            ) if item is not None]
    density_start_year = year_of(node.timestamp_start or node.timestamp)
    if (not source_records_seen and not observations and node.temporal_kind == "interval"
            and node.temporal_density and density_start_year is not None):
        bin_years = node.density_bin_years or 1.0
        start_year = density_start_year
        peak = max(node.temporal_density) or 1.0
        observations = [
            {"year": start_year + round(i * bin_years), "timestamp": None,
             "source": "Агрегированный временной профиль", "source_type": "aggregate",
             "title": None, "authors": [], "publication": None, "forum_topic": None,
             "publication_date": None, "date_evidence": None, "source_url": None,
             "trust": None, "weight": round(min(1.0, w / peak), 3),
             "evidence": [], "snippet": "", "context": "",
             "is_aggregate": True}
            for i, w in enumerate(node.temporal_density) if w > 0
        ]
    has_supported_observation = any(
        (not obs.get("is_aggregate") and bool(obs.get("evidence")))
        or (obs.get("is_aggregate") and obs.get("source_type") == "aggregate")
        for obs in observations
    )
    raw_influences = getattr(node, "__dict__", {}).get("influence_observations", [])
    influence_observations = []
    seen_influences = set()
    if isinstance(raw_influences, list):
        for record in raw_influences:
            if not isinstance(record, dict):
                continue
            source_metadata = record.get("source_metadata", {})
            source_metadata = source_metadata if isinstance(source_metadata, dict) else {}
            key = (str(record.get("source_identity") or record.get("source", "")).casefold(),
                   str(record.get("signal_type", "")).casefold(),
                   str(record.get("evidence", "")))
            if key in seen_influences:
                continue
            seen_influences.add(key)
            influence_observations.append({
                "source": str(record.get("source", "")),
                "source_identity": str(record.get("source_identity", "")),
                "year": year_of(float(record.get("timestamp") or 0.0)),
                "signal_type": str(record.get("signal_type", "")),
                "title": source_metadata.get("title"),
                "authors": list(source_metadata.get("authors", []) or []),
                "publication": source_metadata.get("publication"),
                "forum_topic": source_metadata.get("forum_topic"),
                "source_type": source_metadata.get("source_type"),
                "trust": record.get("trust"),
                "evidence": str(record.get("evidence", ""))[:1000],
                "parameter_observations": list(record.get("parameter_observations", []) or []),
            })

    return {
        "id": node.id, "text": node.text, "domain": node.domain,
        "entity_type": node.entity_type, "origin_kind": node.origin_kind,
        "isPhantom": False, "isVirtual": bool(getattr(node, "is_virtual", False)),
        "isEvidenceSupported": bool(has_supported_observation or getattr(node, "is_virtual", False)),
        "predicted_year": getattr(node, "predicted_year", None),
        "trajectory_confidence": getattr(node, "trajectory_confidence", None),
        "maturity_score": round(float(node.maturity_score), 2),
        "epistemic_level": node.epistemic_level or None,
        "label_status_code": getattr(node, "__dict__", {}).get("label_status_code"),
        "label_status": getattr(node, "__dict__", {}).get("label_status"),
        "label_status_display": label_status_display(
            getattr(node, "__dict__", {}).get("label_status_code")),
        "label_support_score": getattr(node, "__dict__", {}).get("label_support_score"),
        "observations": observations,
        "influence_observations": influence_observations,
    }


def _edge_to_visualizer_dict(
    edge: "KnowledgeEdge",
    consolidator: Optional["GraphConsolidator"] = None,
) -> Dict[str, Any]:
    """One edge's visualizer data, including source-backed observations.

    KnowledgeEdge carries no source-level history, so do not synthesize a
    clickable observation from its aggregate timestamp or confidence.
    """
    observations: List[Dict[str, Any]] = []
    if consolidator is not None:
        rec = consolidator._edge_records.get((edge.source_id, edge.target_id))
        if rec and rec.observations:
            observations = []
            for observation in rec.observations:
                if _is_internal_consolidation_source(observation.source):
                    continue
                evidence = observation.fields.get("evidence", []) or []
                if isinstance(evidence, str):
                    evidence = [evidence]
                source_metadata = observation.fields.get("source_metadata", {}) or {}
                observations.append({
                    "year": _ts_to_year(observation.timestamp) if observation.timestamp else None,
                    "source": observation.source,
                    "title": source_metadata.get("title"),
                    "authors": list(source_metadata.get("authors", []) or []),
                    "publication": source_metadata.get("publication"),
                    "forum_topic": source_metadata.get("forum_topic"),
                    "source_type": source_metadata.get("source_type"),
                    "trust": round(float(observation.trust), 3),
                    "relationship_type": observation.fields.get("relationship_type"),
                    "signal_type": observation.fields.get("signal_type"),
                    "factor_evidence": dict(observation.fields.get("factor_evidence", {}) or {}),
                    "parameter_observations": list(
                        observation.fields.get("parameter_observations", []) or []),
                    "source_identity": observation.fields.get("source_identity"),
                    "evidence": [str(quote)[:1000] for quote in evidence
                                if isinstance(quote, str) and quote.strip()][:5],
                })
    persisted = getattr(edge, "__dict__", {}).get("source_observations", [])
    if isinstance(persisted, list):
        existing_keys = {
            (str(point.get("source_identity") or point.get("source", "")).casefold(),
             str(point.get("signal_type", "")).casefold(),
             tuple(point.get("evidence", []) or []))
            for point in observations
        }
        for record in persisted:
            if (not isinstance(record, dict) or record.get("signal_type")
                    or _is_internal_consolidation_source(record.get("source"))):
                continue
            source_metadata = record.get("source_metadata", {}) or {}
            evidence = record.get("evidence", []) or []
            if isinstance(evidence, str):
                evidence = [evidence]
            try:
                timestamp = float(record.get("timestamp") or 0.0)
                trust = float(record.get("trust", edge.confidence))
            except (TypeError, ValueError):
                timestamp, trust = 0.0, float(edge.confidence)
            key = (str(record.get("source_identity") or record.get("source", "")).casefold(),
                   str(record.get("signal_type", "")).casefold(), tuple(evidence or []))
            if key in existing_keys:
                continue
            observations.append({
                "year": _ts_to_year(timestamp) if timestamp else None,
                "source": str(record.get("source") or ""),
                "title": source_metadata.get("title"),
                "authors": list(source_metadata.get("authors", []) or []),
                "publication": source_metadata.get("publication"),
                "forum_topic": source_metadata.get("forum_topic"),
                "source_type": source_metadata.get("source_type"),
                "source_identity": record.get("source_identity"),
                "trust": round(trust, 3),
                "relationship_type": record.get("relationship_type"),
                "signal_type": record.get("signal_type"),
                "factor_evidence": dict(record.get("factor_evidence", {}) or {}),
                "parameter_observations": list(record.get("parameter_observations", []) or []),
                "evidence": [str(quote)[:1000] for quote in evidence
                            if isinstance(quote, str) and quote.strip()][:5],
            })
            existing_keys.add(key)
    signal_records = []
    seen_signal_keys = set()
    for record in (getattr(edge, "__dict__", {}).get("signal_observations", []) or []):
        if not isinstance(record, dict):
            continue
        source_metadata = record.get("source_metadata", {}) or {}
        evidence = str(record.get("evidence", ""))
        key = (str(record.get("source_identity") or record.get("source", "")).casefold(),
               str(record.get("signal_type", "")).casefold(), evidence)
        if key in seen_signal_keys:
            continue
        seen_signal_keys.add(key)
        signal_records.append({
            "year": _ts_to_year(float(record.get("timestamp") or 0.0))
                if record.get("timestamp") else None,
            "source": str(record.get("source", "")),
            "source_identity": record.get("source_identity"),
            "title": source_metadata.get("title"),
            "authors": list(source_metadata.get("authors", []) or []),
            "publication": source_metadata.get("publication"),
            "forum_topic": source_metadata.get("forum_topic"),
            "source_type": source_metadata.get("source_type"),
            "trust": record.get("trust"),
            "signal_type": record.get("signal_type"),
            "parameter_observations": list(record.get("parameter_observations", []) or []),
            "evidence": [evidence] if evidence else [],
        })
    # Older graph snapshots may have signal receipts only in source_observations.
    for record in (getattr(edge, "__dict__", {}).get("source_observations", []) or []):
        if not isinstance(record, dict) or not record.get("signal_type"):
            continue
        source_metadata = record.get("source_metadata", {}) or {}
        evidence = record.get("evidence", []) or []
        if isinstance(evidence, str):
            evidence = [evidence]
        evidence_text = str(evidence[0]) if evidence else ""
        key = (str(record.get("source_identity") or record.get("source", "")).casefold(),
               str(record.get("signal_type", "")).casefold(), evidence_text)
        if key in seen_signal_keys:
            continue
        seen_signal_keys.add(key)
        signal_records.append({
            "year": _ts_to_year(float(record.get("timestamp") or 0.0))
                if record.get("timestamp") else None,
            "source": str(record.get("source", "")),
            "source_identity": record.get("source_identity"),
            "title": source_metadata.get("title"),
            "authors": list(source_metadata.get("authors", []) or []),
            "publication": source_metadata.get("publication"),
            "forum_topic": source_metadata.get("forum_topic"),
            "source_type": source_metadata.get("source_type"),
            "trust": record.get("trust"),
            "signal_type": record.get("signal_type"),
            "parameter_observations": list(record.get("parameter_observations", []) or []),
            "evidence": [str(quote)[:1000] for quote in evidence
                         if isinstance(quote, str) and quote.strip()][:5],
        })
    metadata = getattr(edge, "__dict__", {}).get("consolidation_meta", {})
    relationship_votes = metadata.get("relationship_type_votes", {})
    relationship_conflict = metadata.get("conflicts", {}).get("relationship_type", {})
    alternate_relationship_types = list(
        getattr(edge, "__dict__", {}).get("alternate_relationship_types", [])
    )
    signal_counts = dict(getattr(edge, "__dict__", {}).get("signal_counts", {}) or {})
    if not signal_counts:
        unique_sources: Dict[str, set] = defaultdict(set)
        for record in signal_records:
            signal_type = str(record.get("signal_type", "")).casefold()
            identity = str(record.get("source_identity") or record.get("source", "")).casefold()
            if signal_type and identity:
                unique_sources[signal_type].add(identity)
        signal_counts = {name: len(sources) for name, sources in unique_sources.items()}
    return {
        "id": edge.id, "source": edge.source_id, "target": edge.target_id,
        "relationship_type": edge.relationship_type,
        "alternate_relationship_types": alternate_relationship_types,
        "relationship_type_votes": relationship_votes,
        "isContested": bool(relationship_conflict or alternate_relationship_types),
        "confidence": round(float(edge.confidence), 3),
        "semantic_similarity": round(float(edge.semantic_similarity), 3),
        "temporal_proximity": round(float(edge.temporal_proximity), 3),
        "limitation_resolution": round(float(edge.limitation_resolution), 3),
        "inhibitory_force": round(float(edge.inhibitory_force), 3),
        "citation_link": round(float(edge.citation_link), 3),
        "investment_correlation": round(float(edge.investment_correlation), 3),
        "social_correlation": round(float(edge.social_correlation), 3),
        "signal_counts": signal_counts,
        "influence_corpus_size": int(
            getattr(edge, "__dict__", {}).get("influence_corpus_size", 0) or 0),
        "signals": signal_records,
        "isPhantom": bool(getattr(edge, "is_virtual", False)), "observations": observations,
    }


def _add_phantom_to_window(
    ph: "PhantomNode", nodes_out: Dict[str, Any], edges_out: List[Dict[str, Any]],
) -> None:
    epistemic_basis = getattr(ph, "epistemic_basis", []) or []
    basis_by_id = {str(item.get("node_id")): item for item in epistemic_basis
                   if isinstance(item, dict) and item.get("node_id")}
    nodes_out[ph.id] = {
        "id": ph.id, "text": (ph.gap_description or ph.predicted_domain)[:80],
        "domain": ph.predicted_domain, "entity_type": ph.predicted_entity_type,
        "origin_kind": "fundamental", "isPhantom": True,
        "structural_gap_score": round(float(ph.structural_gap_score), 3),
        "void_pressure": round(float(ph.void_pressure_at_gap), 3),
        "gap_description": ph.gap_description,
        "predicted_year_range": [int(ph.predicted_year_range[0]),
                                 int(ph.predicted_year_range[1])],
        "epistemic_basis": list(basis_by_id.values()),
        "social_readiness": dict(getattr(ph, "social_readiness", {}) or {}),
        "observations": [],
    }

    def phantom_edge(src_id: str, tgt_id: str, basis_node_id: str) -> Dict[str, Any]:
        return {
            "id": f"phantom_{ph.id}_{src_id}_{tgt_id}", "source": src_id, "target": tgt_id,
            "relationship_type": "candidate_bridge",
            "confidence": None,
            "semantic_similarity": None,
            "temporal_proximity": None,
            "limitation_resolution": None,
            "inhibitory_force": None, "citation_link": None,
            "investment_correlation": None, "social_correlation": None,
            "isPhantom": True,
            "prediction_basis": [basis_by_id[basis_node_id]]
                if basis_node_id in basis_by_id else [],
            "observations": [],
        }

    for src_id in ph.source_node_ids:
        if src_id in nodes_out:
            edges_out.append(phantom_edge(src_id, ph.id, src_id))
    for tgt_id in ph.target_node_ids:
        if tgt_id in nodes_out:
            edges_out.append(phantom_edge(ph.id, tgt_id, tgt_id))


def get_node_window(
    graph: "RuntimeGraph",
    node_id: str,
    consolidator: Optional["GraphConsolidator"] = None,
    canonical_assembler: Optional["CanonicalNodeAssembler"] = None,
    phantom_index: Optional[Dict[str, List["PhantomNode"]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    The ONE query the visualizer needs per "recenter": node_id itself, its
    direct (1-hop) neighbors, the edges connecting them, any phantom nodes
    touching node_id, and the temporal zones covering it — never more.

    O(degree(node_id)) via graph._adjacency (the same incrementally-
    maintained index GraphTraversalEngine/DormancyTracker already read
    this way — see either for the same access pattern), NOT O(N) or O(E):
    this call's cost depends only on how many edges touch this ONE node,
    never on how large the graph is. That's what makes it the right
    replacement for a whole-graph export rather than just a smaller one —
    "smaller" still doesn't work at hundreds of billions of nodes, "scoped
    to exactly what one window needs" does, regardless of graph size.

    phantom_index: Dict[node_id -> List[PhantomNode]] for phantoms whose
    source_node_ids/target_node_ids include that node. PhantomNodeGenerator
    doesn't expose one directly — build it once from phantom_gen.phantoms
    (e.g. defaultdict(list); for ph in phantom_gen.phantoms: for nid in
    ph.source_node_ids + ph.target_node_ids: index[nid].append(ph)) and
    keep it alongside the graph, same lifetime as the consolidator.

    Returns None if node_id isn't in the graph.
    """
    center = graph.get_node(node_id)
    if center is None:
        return None

    center_payload = _node_to_visualizer_dict(center, consolidator, canonical_assembler)
    if not center_payload.get("isEvidenceSupported"):
        endpoint_has_edge_evidence = bool(getattr(center, "__dict__", {}).get(
            "edge_endpoint_only")) and any(
                any(observation.get("evidence") for observation in
                    _edge_to_visualizer_dict(graph.edges[key], consolidator).get(
                        "observations", []))
                for key in graph._adjacency.get(node_id, ()) if key in graph.edges
            )
        if not endpoint_has_edge_evidence:
            return None
        center_payload["isEvidenceSupported"] = True
    nodes_out: Dict[str, Any] = {
        node_id: center_payload
    }
    edges_out: List[Dict[str, Any]] = []

    for key in graph._adjacency.get(node_id, ()):
        edge = graph.edges.get(key)
        if edge is None:
            continue  # index/edge-dict momentarily out of sync — skip, don't crash a live query
        other_id = key[1] if key[0] == node_id else key[0]
        other = graph.get_node(other_id)
        if other is None:
            continue  # dangling adjacency entry
        if other_id not in nodes_out:
            other_payload = _node_to_visualizer_dict(other, consolidator, canonical_assembler)
            if not other_payload.get("isEvidenceSupported"):
                edge_payload = _edge_to_visualizer_dict(edge, consolidator)
                endpoint_has_edge_evidence = bool(getattr(other, "__dict__", {}).get(
                    "edge_endpoint_only")) and any(
                        observation.get("evidence")
                        for observation in edge_payload.get("observations", []))
                if not endpoint_has_edge_evidence:
                    continue
                other_payload["isEvidenceSupported"] = True
            nodes_out[other_id] = other_payload
        edges_out.append(_edge_to_visualizer_dict(edge, consolidator))

    if phantom_index:
        for ph in phantom_index.get(node_id, []):
            _add_phantom_to_window(ph, nodes_out, edges_out)

    zones_out = [{
        "id": z.id, "description": z.description, "domain_focus": z.domain_focus,
        "start_year": _ts_to_year(z.start_timestamp) if z.start_timestamp else None,
        "end_year": _ts_to_year(z.end_timestamp) if z.end_timestamp else None,
        "zone_multiplier": round(float(z.zone_multiplier), 2),
        "contained_node_ids": [nid for nid in z.contained_node_ids if nid in nodes_out],
    } for z in graph.zones_covering_node(node_id)]

    return {"nodes": nodes_out, "edges": edges_out, "zones": zones_out}


def search_nodes(graph: "RuntimeGraph", query: str, limit: int = 20,
                 consolidator: Optional["GraphConsolidator"] = None,
                 canonical_assembler: Optional["CanonicalNodeAssembler"] = None) -> List[Dict[str, str]]:
    """
    Best-effort text search over node.text for the visualizer's search
    box. Correct at whatever scale the in-memory RuntimeGraph itself can
    hold — but this IS a linear scan (O(N) over graph.nodes), so it does
    NOT itself solve "search across hundreds of billions of nodes" any
    more than get_node_window() tries to export them: BucketIndex is
    embedding-similarity routing for Pass D, not a text index, and isn't
    wired to arbitrary substring queries. At real production scale, swap
    this for whatever full-text/trigram/prefix index actually backs node
    lookup in your deployment; this is the minimum that makes the search
    box work against a live graph at all.
    """
    q = query.strip().lower()
    if not q:
        return []
    results: List[Dict[str, str]] = []
    for nid, node in graph.nodes.items():
        if q in node.text.lower() or q in nid.lower():
            if not _node_to_visualizer_dict(
                    node, consolidator, canonical_assembler).get("isEvidenceSupported"):
                continue
            results.append({"id": nid, "text": node.text, "entity_type": node.entity_type})
            if len(results) >= limit:
                break
    return results


def serve_visualizer_api(
    graph: "RuntimeGraph",
    host: str = "127.0.0.1",
    port: int = 8765,
    consolidator: Optional["GraphConsolidator"] = None,
    canonical_assembler: Optional["CanonicalNodeAssembler"] = None,
    phantom_index: Optional[Dict[str, List["PhantomNode"]]] = None,
) -> None:
    """
    Minimal, dependency-free HTTP API so oracle1_visualizer.html can query
    the live graph directly:
      GET /neighborhood?id=<node_id>  -> get_node_window(...) or 404
      GET /search?q=<text>&limit=<n>  -> {"results": [...], "graph": {counts}}
    Blocks until interrupted (Ctrl-C) — run it in its own process/thread.
    Standard library only (http.server) on purpose, so this runs with no
    new dependency; swap in Flask/FastAPI/an ASGI server if your
    deployment already has one and you'd rather host it there instead.
    """
    import http.server
    import urllib.parse

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, status: int, payload: Any) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)
            try:
                if parsed.path == "/neighborhood":
                    node_id = (params.get("id") or [""])[0]
                    window = get_node_window(graph, node_id, consolidator,
                                              canonical_assembler, phantom_index)
                    if window is None:
                        self._send_json(404, {"error": f"unknown node id '{node_id}'"})
                    else:
                        self._send_json(200, window)
                elif parsed.path == "/search":
                    q = (params.get("q") or [""])[0]
                    limit = int((params.get("limit") or ["20"])[0])
                    graph_status = {
                        "nodes": len(graph.nodes),
                        "edges": len(graph.edges),
                    }
                    visualizer_metadata = getattr(graph, "visualizer_metadata", {})
                    if isinstance(visualizer_metadata, dict):
                        graph_status.update({
                            key: visualizer_metadata[key]
                            for key in ("mode", "label")
                            if key in visualizer_metadata
                        })
                    self._send_json(200, {
                        "results": search_nodes(
                            graph, q, limit, consolidator, canonical_assembler),
                        "graph": graph_status,
                    })
                else:
                    self._send_json(404, {"error": "unknown endpoint"})
            except Exception as exc:
                logger.error(f"[VisualizerAPI] {self.path} failed: {exc}")
                self._send_json(500, {"error": str(exc)})

        def log_message(self, fmt, *args):
            logger.debug("[VisualizerAPI] " + fmt % args)

    server = http.server.ThreadingHTTPServer((host, port), Handler)
    logger.info(f"[VisualizerAPI] serving on http://{host}:{port} "
                f"(/neighborhood?id=..., /search?q=...)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

__all__ = ['_node_to_visualizer_dict', '_edge_to_visualizer_dict', '_add_phantom_to_window', 'get_node_window', 'search_nodes', 'serve_visualizer_api']
