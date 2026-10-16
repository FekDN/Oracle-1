# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Evidence status codes and time-ordered support calculations for graph labels."""
from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict, Iterable, List, Optional


class LabelStatus(IntEnum):
    """Compact per-source classification attached to one extracted graph label."""

    IDEA_CONCEPT = 1
    THEORY = 2
    FACT = 3
    PROOF = 4
    REFUTATION = 5
    # Read older snapshots and model fixtures that used "evidence" as a status.
    EVIDENCE = PROOF


LABEL_STATUS_NAMES = {
    LabelStatus.IDEA_CONCEPT: "idea_concept",
    LabelStatus.THEORY: "theory",
    LabelStatus.FACT: "fact",
    LabelStatus.PROOF: "proof",
    LabelStatus.REFUTATION: "refutation",
}

LABEL_STATUS_DISPLAY = {
    LabelStatus.IDEA_CONCEPT: "идея / концепт",
    LabelStatus.THEORY: "теория",
    LabelStatus.FACT: "факт",
    LabelStatus.PROOF: "доказательство",
    LabelStatus.REFUTATION: "опровержение",
}

LABEL_STATUS_STRENGTH = {
    LabelStatus.IDEA_CONCEPT: 0.10,
    LabelStatus.THEORY: 0.35,
    LabelStatus.FACT: 0.75,
    LabelStatus.PROOF: 0.95,
    LabelStatus.REFUTATION: 0.75,
}


def parse_label_status(value: Any) -> Optional[LabelStatus]:
    """Return a known status from its JSON name or compact integer code."""
    if isinstance(value, LabelStatus):
        return value
    if isinstance(value, str):
        key = value.strip().casefold().replace("-", "_").replace("/", "_")
        aliases = {
            "idea": LabelStatus.IDEA_CONCEPT,
            "concept": LabelStatus.IDEA_CONCEPT,
            "idea_concept": LabelStatus.IDEA_CONCEPT,
            "theory": LabelStatus.THEORY,
            "evidence": LabelStatus.PROOF,
            "proof": LabelStatus.PROOF,
            "fact": LabelStatus.FACT,
            "refutation": LabelStatus.REFUTATION,
            "refute": LabelStatus.REFUTATION,
        }
        return aliases.get(key)
    try:
        return LabelStatus(int(value))
    except (TypeError, ValueError):
        return None


def label_status_name(value: Any) -> Optional[str]:
    """Return the stable JSON-facing name for a status code."""
    status = parse_label_status(value)
    return LABEL_STATUS_NAMES.get(status) if status is not None else None


def label_status_display(value: Any) -> Optional[str]:
    """Return the Russian UI label for a status code."""
    status = parse_label_status(value)
    return LABEL_STATUS_DISPLAY.get(status) if status is not None else None


def node_label_status(node: Any) -> Optional[LabelStatus]:
    """Return a node's current evidence-derived status, including promotions."""
    private = getattr(node, "__dict__", {})
    timeline = private.get("label_support_timeline", [])
    if isinstance(timeline, list) and timeline:
        latest = timeline[-1]
        status = parse_label_status(latest.get("effective_label_status_code"))
        if status is not None:
            return status
    status = parse_label_status(private.get("effective_label_status_code"))
    return status or parse_label_status(private.get("label_status_code")) \
        or parse_label_status(private.get("label_status"))


def node_label_support_score(node: Any) -> float:
    """Return the current source-grounded support strength for phantom gating."""
    private = getattr(node, "__dict__", {})
    timeline = private.get("label_support_timeline", [])
    if isinstance(timeline, list) and timeline:
        try:
            return min(1.0, max(0.0, float(timeline[-1].get("support_score", 0.0))))
        except (TypeError, ValueError):
            return 0.0
    try:
        trust = min(1.0, max(0.0, float(private.get("source_trust", 0.5))))
    except (TypeError, ValueError):
        trust = 0.5
    status = node_label_status(node)
    return LABEL_STATUS_STRENGTH.get(status, 0.0) * trust


def build_label_support_timeline(
    observations: Iterable[Dict[str, Any]],
    refutations: Iterable[Dict[str, Any]] = (),
    corroborations: Iterable[Dict[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Rebuild a label's dated support curve from source events and refutations.

    Source status is assigned from that source's quoted passage. Source trust
    can scale an event's strength, but cannot promote a status to ``fact``.
    A grounded contradiction reduces support from its own date onward; later
    evidence can raise it again. Earlier timeline points remain unchanged.
    """
    events: List[Dict[str, Any]] = []
    for observation in observations:
        if not isinstance(observation, dict):
            continue
        status = parse_label_status(observation.get("label_status_code"))
        if status is None:
            status = parse_label_status(observation.get("label_status"))
        if status is None:
            continue
        event = dict(observation)
        event["label_status_code"] = int(status)
        event["label_status"] = LABEL_STATUS_NAMES[status]
        events.append(event)

    for refutation in refutations:
        if not isinstance(refutation, dict):
            continue
        event = dict(refutation)
        event["label_status_code"] = int(LabelStatus.REFUTATION)
        event["label_status"] = LABEL_STATUS_NAMES[LabelStatus.REFUTATION]
        event["is_refutation_event"] = True
        event["event_type"] = "refutation"
        events.append(event)

    for corroboration in corroborations:
        if not isinstance(corroboration, dict):
            continue
        event = dict(corroboration)
        event["event_type"] = str(event.get("event_type") or "supports")
        events.append(event)

    def event_key(event: Dict[str, Any]):
        try:
            timestamp = float(event.get("timestamp") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        return (timestamp, str(event.get("source", "")),
                str(event.get("source_node_id", "")))

    events.sort(key=event_key)
    strength = 0.0
    effective_status = LabelStatus.IDEA_CONCEPT
    confirmation_sources: Dict[str, float] = {}
    timeline: List[Dict[str, Any]] = []
    for event in events:
        event_type = event.get("event_type", "observation")
        status = parse_label_status(event.get("label_status_code"))
        try:
            trust = min(1.0, max(0.0, float(event.get("trust", 0.5))))
        except (TypeError, ValueError):
            trust = 0.5
        if event_type == "refutation":
            try:
                confidence = min(1.0, max(0.0, float(event.get("confidence", 0.5))))
            except (TypeError, ValueError):
                confidence = 0.5
            strength *= 1.0 - 0.75 * min(trust, confidence)
            confirmation_sources.clear()
        elif event_type in {"supports", "confirms", "proves"}:
            source_status = parse_label_status(event.get("source_label_status_code"))
            if source_status not in {LabelStatus.FACT, LabelStatus.PROOF}:
                continue
            source_strength = LABEL_STATUS_STRENGTH[source_status] * trust
            if source_strength < 0.40:
                continue
            source_key = str(event.get("independence_key") or event.get("source") or "")
            if event_type == "proves" and source_status == LabelStatus.PROOF:
                if trust >= 0.80:
                    effective_status = LabelStatus.PROOF
                    strength = max(
                        strength, LABEL_STATUS_STRENGTH[LabelStatus.PROOF] * trust)
                else:
                    effective_status = max(effective_status, LabelStatus.FACT)
                    strength = max(strength, LABEL_STATUS_STRENGTH[LabelStatus.FACT] * trust)
            elif event_type == "confirms":
                if source_key:
                    confirmation_sources[source_key] = max(
                        confirmation_sources.get(source_key, 0.0), trust)
                effective_status = max(effective_status, LabelStatus.FACT)
                strength = max(strength, LABEL_STATUS_STRENGTH[LabelStatus.FACT] * trust)
                # Two distinct, source-identified confirmations are a proof
                # event; repeated edges or popularity from one paper do not count.
                independent_trust = min(confirmation_sources.values(), default=0.0)
                if len(confirmation_sources) >= 2 and independent_trust >= 0.70:
                    effective_status = LabelStatus.PROOF
                    strength = max(strength, 0.85 * independent_trust)
            elif event_type == "supports":
                effective_status = max(effective_status, LabelStatus.FACT)
                strength = max(strength, LABEL_STATUS_STRENGTH[LabelStatus.FACT] * trust)
        elif status is not None:
            status_strength = LABEL_STATUS_STRENGTH[status]
            strength = max(strength, status_strength * trust)
            effective_status = max(effective_status, status)
        else:
            continue

        # A refutation changes the target's current assessment without
        # rewriting earlier points or the original status on its source record.
        if event_type == "refutation":
            if strength < 0.20:
                effective_status = LabelStatus.IDEA_CONCEPT
            elif strength < 0.55:
                effective_status = LabelStatus.THEORY
            elif strength < 0.82:
                effective_status = LabelStatus.FACT
            else:
                effective_status = LabelStatus.PROOF
        point = dict(event)
        point["support_score"] = round(min(1.0, max(0.0, strength)), 4)
        point["effective_label_status_code"] = int(effective_status)
        point["effective_label_status"] = LABEL_STATUS_NAMES[effective_status]
        timeline.append(point)
    return timeline


__all__ = [
    "LabelStatus", "LABEL_STATUS_NAMES", "LABEL_STATUS_DISPLAY",
    "LABEL_STATUS_STRENGTH", "parse_label_status", "label_status_name",
    "label_status_display", "node_label_status", "node_label_support_score",
    "build_label_support_timeline",
]
