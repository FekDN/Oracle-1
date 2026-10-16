"""Serve the existing one-hop graph visualizer against a trusted snapshot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from oracle1 import (
    KnowledgeEdge,
    KnowledgeNode,
    PhantomNodeGenerator,
    RuntimeGraph,
    _year_to_ts,
    load_graph_snapshot,
    serve_visualizer_api,
)


class _VirtualSynthState:
    def __init__(self) -> None:
        self.virtual_nodes = []


class _DynamicsState:
    def __init__(self) -> None:
        self.virtual_synth = _VirtualSynthState()


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-1 graph window API")
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    graph = RuntimeGraph()
    phantom_gen = PhantomNodeGenerator()
    dynamics = _DynamicsState()
    load_graph_snapshot(args.snapshot, {
        "graph": graph, "phantom_gen": phantom_gen, "dynamics": dynamics,
    })
    manifest_dir = Path(args.snapshot).resolve().parent
    manifest = {}
    for manifest_name in ("manifest.json", "run_manifest.json"):
        try:
            candidate = json.loads(
                (manifest_dir / manifest_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(candidate, dict):
            manifest = candidate
            break
    visualizer_mode = manifest.get("visualizer_mode") or manifest.get("run_mode")
    report_title = ""
    if not visualizer_mode:
        try:
            report = (manifest_dir / "run_report.md").read_text(encoding="utf-8")
        except OSError:
            report = ""
        report_title = next(
            (line[2:].strip() for line in report.splitlines() if line.startswith("# ")),
            "",
        )
        if "Offline LLM simulation" in report or manifest_dir.name.startswith("llm_simulated_test"):
            visualizer_mode = "simulated_llm_response"
    if visualizer_mode in {"citation_preview", "simulated_llm_response"}:
        graph.visualizer_metadata = {
            "mode": visualizer_mode,
            "label": (manifest.get("label") or manifest.get("warning")
                      or manifest.get("description") or report_title),
        }
    for virtual in dynamics.virtual_synth.virtual_nodes:
        if virtual.id in graph.nodes:
            continue
        node = KnowledgeNode(
            id=virtual.id,
            text=virtual.description,
            node_type="virtual_forecast",
            domain=virtual.predicted_domain,
            entity_type=virtual.predicted_entity_type,
            timestamp=_year_to_ts(virtual.predicted_year),
            readiness_score=float(virtual.predicted_readiness) * 10.0,
            forecast_score=float(virtual.trajectory_confidence) * 10.0,
        )
        node.__dict__["is_virtual"] = True
        node.__dict__["predicted_year"] = int(virtual.predicted_year)
        node.__dict__["trajectory_confidence"] = round(float(virtual.trajectory_confidence), 3)
        graph.add_node(node)
        for source_id in virtual.source_trajectory:
            if source_id not in graph.nodes:
                continue
            edge = KnowledgeEdge(
                id=f"vis_{source_id[:8]}_{virtual.id[:8]}",
                source_id=source_id,
                target_id=virtual.id,
                relationship_type="projection",
                discovery_method="virtual_prediction",
                semantic_similarity=0.7,
                confidence=float(virtual.trajectory_confidence),
                timestamp=node.timestamp,
            )
            edge.__dict__["is_virtual"] = True
            edge.compute_total_weight()
            graph.add_edge(edge)
    phantom_index = defaultdict(list)
    for phantom in phantom_gen.phantoms:
        for node_id in phantom.source_node_ids + phantom.target_node_ids:
            phantom_index[node_id].append(phantom)
    serve_visualizer_api(
        graph, host="127.0.0.1", port=args.port,
        phantom_index=phantom_index,
    )


if __name__ == "__main__":
    main()
