# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""High-level dynamics and topology orchestration."""
from __future__ import annotations

from ._shared import *
from .config import ALERT_ONTOLOGICAL_RUPTURE, ATTENTION_GAMMA, CONFIG
from .entities import KnowledgeEdge, KnowledgeNode, PhysicalSubstrateEncoder
from .forecasting import (
    ConvergenceCloud,
    ForecastScenario,
    GraphEntropyMonitor,
    PathAgnosticInference,
    PhantomNode,
    PhantomNodeGenerator,
    RecursiveForecastingLoop,
    RetrospectiveConvergenceLoss,
)
from .graph import GraphWalker, RuntimeGraph, extend_edge
from .graph_dynamics import ContextualPriorityHead, ConvergencePressureField, DormancyTracker
from .neural import Oracle1Model
from .topology_dynamics import (
    AnomalyAccumulator,
    ConnectivityEntropyCalculator,
    ConvergenceAccelerometer,
    CrossDomainIsomorphismAmplifier,
    DualObjectiveHead,
    ObligatoryBridgeDetector,
    OntologicalTensionField,
    PhantomPhaseAligner,
    PhaseTransitionDetector,
    StabilizedCentroidTracker,
    StructuralConflictMeter,
    VirtualNodeFactory,
    VirtualNodeSynthesizer,
    update_phantom_weights,
)

# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  ORACLE-1 EXTENDED (master orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class Oracle1Extended:
    def __init__(self, base_model: Oracle1Model, graph: RuntimeGraph,
                 phantom_gen: Optional["PhantomNodeGenerator"] = None):
        self.model      = base_model
        self.graph      = graph
        self.latent_dim = getattr(base_model, "latent_dim", 512)

        self.physical_encoder = PhysicalSubstrateEncoder()
        self.pressure_field   = ConvergencePressureField()
        self.dormancy_tracker = DormancyTracker()
        # Fix 4: reuse the priority_head that is already registered inside
        # Oracle1Model (so its parameters are trained via backprop).
        # If the model stub doesn't carry one, fall back to a standalone instance.
        self.priority_head = getattr(base_model, "priority_head", None)
        if self.priority_head is None and getattr(base_model, "neural_enabled", True):
            self.priority_head = ContextualPriorityHead(self.latent_dim)
        # TODO #4 fix: this used to always build its own private
        # PhantomNodeGenerator, disconnected from the one build_oracle1_system
        # creates and threads into DynamicsOrchestrator / TopologyInstability-
        # Orchestrator. scan_for_voids() (the only place phantoms are ever
        # produced) is only ever called here, on self.phantom_gen — so every
        # phantom generated used to be invisible to dynamics.phase_aligner
        # and topology's rupture-confirmation, which read a permanently-empty
        # separate instance. Accepting the shared instance (falling back to a
        # private one only for standalone/test use) fixes that split.
        self.phantom_gen      = phantom_gen if phantom_gen is not None else PhantomNodeGenerator()
        self.conv_loss        = RetrospectiveConvergenceLoss()
        self.entropy_monitor  = GraphEntropyMonitor()

        self.walker = GraphWalker(graph)
        self.recursive_forecast = RecursiveForecastingLoop(
            graph_walker=self.walker,
            edge_predictor=getattr(base_model, "edge_pred_head", None),
            pressure_field=self.pressure_field,
            phantom_gen=self.phantom_gen,
            entropy_monitor=self.entropy_monitor)  # ← NEW: feeds predict_entropy_trend()
        self.path_agnostic = PathAgnosticInference(
            graph_walker=self.walker, pressure_field=self.pressure_field,
            phantom_gen=self.phantom_gen, entropy_monitor=self.entropy_monitor,
            priority_head=self.priority_head, dormancy_tracker=self.dormancy_tracker)

        logger.info("Oracle1Extended initialised")

    def register_edge(self, edge: KnowledgeEdge, inhibitory_force: float = 0.0):
        extend_edge(edge, inhibitory_force)
        self.dormancy_tracker.initialise_edge(edge)
        self.graph.add_edge(edge)
        # add_edge() already updates the in-degree cache, which satisfies the
        # ancestry invariant for any node that receives this edge as its first
        # incoming edge.

    def update_epoch(self, timestamp: float,
                      node_latents: Optional["torch.Tensor"] = None,
                      full_dormancy_resync: bool = False) -> Dict[str, Any]:
        """
        node_latents: TODO #4 fix. Optional [N, latent_dim] tensor, row-aligned
        with `list(self.graph.nodes.keys())` at the time of the call (the same
        ordering graph_to_tensors()/compute_node_latents() produce and that
        PhantomNodeGenerator._node_to_idx already assumes). When omitted,
        phantom scanning falls back to the previous behaviour (no void
        detection) rather than failing — callers that don't have a model/
        device handy (e.g. quick symbolic-only smoke tests) keep working.

        full_dormancy_resync (TODO #22): False (default) runs
        DormancyTracker.update_wave() over exactly the node set
        pressure_field.update() just identified as genuinely changed —
        the normal per-ingestion hot path, cost proportional to the
        disturbance rather than to len(self.dormancy_tracker.states). Pass
        True on whatever periodic cadence the caller can still afford
        (nightly, every N calls, ...) to run the exact
        DormancyTracker.update_epoch() pass instead — the only one that
        also evaluates _semantic_twin_heat (deliberately not hop-bounded,
        see update_wave()'s docstring, so no wave can substitute for it).
        """
        # §3 fix: incremental update() (diff + 2-hop exact closure + decaying
        # wave) instead of a full compute() over every node on every epoch.
        # compute() is still what update() itself falls back to on its very
        # first call (cold start, no snapshot to diff against yet).
        self.pressure_field.update(self.graph)
        if full_dormancy_resync:
            awakened = self.dormancy_tracker.update_epoch(self.graph)
        else:
            # Reuse pressure_field's own change-detection (TODO #22) instead
            # of DormancyTracker deriving a second, separate diff to answer
            # the same "what actually changed" question.
            awakened = self.dormancy_tracker.update_wave(
                self.pressure_field._last_changed, self.graph)
        new_phantoms = self.phantom_gen.scan_for_voids(
            node_latents=node_latents, graph=self.graph, pressure_field=self.pressure_field,
            seed_ids=self.pressure_field._last_changed)  # TODO #23 — scoped, not graph-wide
        entropy_snaps = self.entropy_monitor.compute_snapshot(
            self.graph, self.pressure_field, timestamp)
        alerts = self.entropy_monitor.get_alerts()
        return {
            "n_nodes":        len(self.graph.nodes),
            "n_edges":        len(self.graph.edges),
            "n_phantoms":     len(self.phantom_gen.phantoms),
            "awakened_edges": len(awakened),
            "new_phantoms":   len(new_phantoms),
            "entropy_alerts": [(d, t) for d, t, _ in alerts],
            "top_pressure_nodes": self._top_pressure_nodes(5)}

    def _top_pressure_nodes(self, n: int) -> List[Tuple[str, float]]:
        items = [(nid, pv.void_pressure) for nid, pv in self.pressure_field.field.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        return [(self.graph.nodes[nid].text[:40] if nid in self.graph.nodes else nid, p)
                for nid, p in items[:n]]

    def query_convergence(self, query_text: str, embedder, n_paths: int = 20) -> ConvergenceCloud:
        return self.path_agnostic.query(query_text, embedder, self.graph, n_paths)

    def run_recursive_forecast(self, seed_query: str, embedder,
                                max_depth: int = 5) -> List[ForecastScenario]:
        trajs    = self.walker.query(seed_query, embedder, max_trajectories=5)
        seed_ids = [t.steps[0].node_id for t in trajs if t.steps]
        return self.recursive_forecast.forecast(seed_node_ids=seed_ids,
                                                base_graph=self.graph, n_scenarios=3)

    def new_node_arrived(self, node: KnowledgeNode) -> List[PhantomNode]:
        confirmed = self.phantom_gen.check_confirmation(node, self.graph)
        if confirmed:
            logger.info(f"[Oracle1Extended] {len(confirmed)} phantoms confirmed by '{node.text[:40]}'")
        return confirmed


# ══════════════════════════════════════════════════════════════════════════════
#  DYNAMICS ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class DynamicsOrchestrator:
    def __init__(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField,
                 phantom_gen: PhantomNodeGenerator, llm_client=None,
                 neural_enabled: bool = True):
        self.graph        = graph
        self.pressure     = pressure_field
        self.phantom_gen  = phantom_gen
        self.centroid_tracker = StabilizedCentroidTracker()
        self.accelerometer    = ConvergenceAccelerometer()
        self.bridge_detector  = ObligatoryBridgeDetector()
        self.phase_aligner    = PhantomPhaseAligner(self.accelerometer)
        self.virtual_synth    = VirtualNodeSynthesizer(
            self.centroid_tracker, self.accelerometer, self.bridge_detector,
            llm_client=llm_client)
        self.dual_head = (DualObjectiveHead(latent_dim=CONFIG["model_latent_dim"])
                  if neural_enabled else None)

    def run_epoch(self, timestamp: float, all_embeddings=None) -> Dict[str, Any]:
        self.centroid_tracker.epoch_update(self.graph)
        false_conv = self.centroid_tracker.false_convergence_nodes()
        self.accelerometer.update_all(self.graph, self.pressure, all_embeddings or {}, timestamp)
        top_accel  = self.accelerometer.top_accelerating(5)
        pre_break  = self.accelerometer.pre_breakthrough_candidates()
        new_voids  = self.bridge_detector.scan(self.graph, self.pressure, self.centroid_tracker, timestamp)
        for phantom in self.phantom_gen.phantoms:
            self.phase_aligner.align(phantom, self.graph, self.pressure, timestamp)
        best_phantoms = self.phase_aligner.get_best_phantoms(n=3)
        virtual_traj = self.virtual_synth.synthesize_from_trajectories(self.graph, timestamp)
        virtual_void = self.virtual_synth.synthesize_from_voids(self.graph, timestamp)
        self.virtual_synth.virtual_nodes.extend(virtual_traj + virtual_void)
        return {
            "false_convergence_clusters":  false_conv,
            "top_accelerating_nodes":      [(nid[:12], f"{a:.4f}") for nid, a in top_accel],
            "pre_breakthrough_candidates": len(pre_break),
            "new_structural_voids":        len(new_voids),
            "top_void_severity":           (new_voids[0].void_severity if new_voids else 0.0),
            "best_phantom_lags":           [(m.phantom_id[:8], m.phase_alignment,
                                             f"{m.lag_estimate_years:.0f}yr") for m, _ in best_phantoms],
            "new_virtual_nodes_traj":      len(virtual_traj),
            "new_virtual_nodes_void":      len(virtual_void),
            "total_virtual_nodes":         len(self.virtual_synth.virtual_nodes)}

    def get_virtual_nodes_for_recursive_forecast(self) -> List[KnowledgeNode]:
        nodes, _ = self.virtual_synth.get_virtual_nodes_as_graph_entries()
        return nodes


# ══════════════════════════════════════════════════════════════════════════════
#  TOPOLOGY INSTABILITY ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

class TopologyInstabilityOrchestrator:
    def __init__(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField,
                 phantom_gen: Optional["PhantomNodeGenerator"] = None,
                 physical_encoder: Optional["PhysicalSubstrateEncoder"] = None):
        self.graph            = graph
        self.pressure         = pressure_field
        self.entropy_calc     = ConnectivityEntropyCalculator()
        self.anomaly_acc      = AnomalyAccumulator()
        self.conflict_meter   = StructuralConflictMeter()
        self.tension_field    = OntologicalTensionField()
        self.isomorphism      = CrossDomainIsomorphismAmplifier()
        self.virtual_factory  = VirtualNodeFactory()
        self.phase_detector   = PhaseTransitionDetector()
        # Optional hooks for rupture-driven physics relaxation and anomaly confirmation
        self.phantom_gen      = phantom_gen
        self.physical_encoder = physical_encoder or PhysicalSubstrateEncoder()
        # Tracks node_ids for which physics have already been relaxed this session
        self._relaxed_rupture_ids: Set[str] = set()

    def run_epoch(self, timestamp: float) -> Dict[str, Any]:
        update_phantom_weights(self.graph)
        prev_entropy = dict(self.entropy_calc._cache)
        self.entropy_calc.compute_all(self.graph)
        entropy_gradients = {nid: self.entropy_calc.entropy_gradient(nid, prev_entropy.get(nid, 0.0))
                             for nid in self.graph.nodes}
        domain_degree_stats = self.anomaly_acc.domain_degree_statistics(self.graph)
        for nid in self.graph.nodes:
            self.anomaly_acc.measure_and_push(nid, self.graph, domain_degree_stats)
        self.conflict_meter.compute_all(self.graph)
        tension_states = self.tension_field.compute(
            self.graph, self.pressure, self.conflict_meter,
            self.anomaly_acc, self.entropy_calc, timestamp)
        n_amplified = self.isomorphism.amplify_cross_domain_edges(self.graph)
        latent_iso  = self.isomorphism.detect_latent_isomorphisms(self.graph, top_n=10)
        new_virtual_specs = self.virtual_factory.generate_from_tension(tension_states, self.graph, timestamp)
        new_alerts = self.phase_detector.scan(
            tension_states, self.conflict_meter, self.entropy_calc, self.graph, timestamp)

        # ── Post-rupture physics relaxation ──────────────────────────────────
        # For every newly confirmed ONTOLOGICAL_RUPTURE, collapse the physical
        # constraint vectors of dependent nodes so that Recursive Forecasts that
        # were previously blocked as "physically impossible" can now flow through.
        rupture_relaxed: Dict[str, int] = {}
        for alert in new_alerts:
            if alert.alert_type == ALERT_ONTOLOGICAL_RUPTURE:
                if alert.node_id not in self._relaxed_rupture_ids:
                    relaxed = self.physical_encoder.resolve_rupture_physics(
                        alert.node_id, self.graph)
                    # Mark relaxed flag on nodes so downstream feasibility gates skip them
                    for nid in relaxed:
                        node = self.graph.get_node(nid)
                        if node:
                            node.rupture_physics_relaxed = True
                    self._relaxed_rupture_ids.add(alert.node_id)
                    rupture_relaxed[alert.node_id] = len(relaxed)

        # ── Anomaly-driven phantom confirmation (Confirmation by Rupture) ─────
        rupture_confirmations: List[str] = []
        if self.phantom_gen is not None:
            for nid in self.graph.nodes:
                node = self.graph.get_node(nid)
                if not node or node.is_temporal_zone: continue
                confirmed = self.phantom_gen.check_confirmation_by_rupture(
                    node, self.graph, self.anomaly_acc)
                for ph in confirmed:
                    rupture_confirmations.append(ph.id)

        alert_counts = self.phase_detector.summary()
        brittle = self.entropy_calc.brittle_nodes()
        diffuse = self.entropy_calc.diffuse_nodes()
        top_conflict = self.conflict_meter.top_conflict_nodes(5)
        top_tension = sorted([(nid, s.tension) for nid, s in tension_states.items()],
                             key=lambda x: x[1], reverse=True)[:5]
        return {
            "total_tension_states":   len(tension_states),
            "top_tension_nodes":      [(self.graph.nodes[nid].text[:30] if nid in self.graph.nodes else nid,
                                        f"{t:.3f}") for nid, t in top_tension],
            "rupture_count":          len(self.phase_detector.rupture_nodes()),
            "new_phase_alerts":       len(new_alerts),
            "alert_summary":          alert_counts,
            "new_virtual_specs":      len(new_virtual_specs),
            "brittle_nodes_count":    len(brittle),
            "diffuse_nodes_count":    len(diffuse),
            "cross_domain_amplified": n_amplified,
            "latent_isomorphisms":    len(latent_iso),
            "top_conflict_nodes":     [(self.graph.nodes[nid].text[:30] if nid in self.graph.nodes else nid,
                                        f"{c:.3f}") for nid, c in top_conflict],
            "entropy_rising_nodes":   sum(1 for g in entropy_gradients.values() if g > 0.05),
            "entropy_falling_nodes":  sum(1 for g in entropy_gradients.values() if g < -0.05),
            "rupture_physics_relaxed": rupture_relaxed,
            "rupture_confirmations":  len(rupture_confirmations)}

    def get_virtual_seeds(self) -> List[KnowledgeNode]:
        return self.virtual_factory.as_knowledge_nodes(self.graph)

    def get_attention_multipliers(self) -> Dict[str, float]:
        """
        Return the attention multiplier T(n)^γ (capped at 10.0) for every
        node that has fired a phase alert.  Used by ContextualPriorityHead
        to weight edges toward high-tension nodes.

        Note: populated only after run_epoch() has been called at least once;
        returns an empty dict otherwise.
        """
        result: Dict[str, float] = {}
        for alert in self.phase_detector.alerts:
            state_tension = alert.tension
            multiplier = float(min(10.0, max(1.0, state_tension ** ATTENTION_GAMMA)))
            result[alert.node_id] = multiplier
        return result

    def rupture_report(self) -> str:
        ruptures = self.phase_detector.rupture_nodes()
        if not ruptures: return "No ontological ruptures detected in current epoch."
        lines = [f"ONTOLOGICAL RUPTURE REPORT  ({len(ruptures)} active)", "=" * 60]
        for alert in sorted(ruptures, key=lambda a: a.tension, reverse=True):
            node = self.graph.get_node(alert.node_id)
            name   = node.text[:50] if node else alert.node_id
            domain = node.domain if node else "unknown"
            specs  = [s for s in self.virtual_factory.virtual_specs if s.source_node_id == alert.node_id]
            lines += [
                f"\n  NODE:    {name}",
                f"  DOMAIN:  {domain}",
                f"  TENSION: {alert.tension:.3f}  (T^{ATTENTION_GAMMA:.1f} = {alert.tension**ATTENTION_GAMMA:.3f})",
                f"  COMPONENTS: pressure={alert.pressure:.2f}  conflict={alert.conflict:.2f}  entropy={alert.entropy:.2f}",
                f"  MESSAGE: {alert.message}"]
            if specs:
                lines.append("  REQUIREMENTS FOR RESOLVING NODE:")
                for k, v in specs[0].functional_requirements.items():
                    lines.append(f"    {k}: {v}")
        return "\n".join(lines)

__all__ = ['Oracle1Extended', 'DynamicsOrchestrator', 'TopologyInstabilityOrchestrator']
