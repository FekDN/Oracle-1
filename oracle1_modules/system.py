"""Factory for constructing a fully wired Oracle-1 system."""
from __future__ import annotations

from ._shared import *
from .canonical import CanonicalNodeAssembler, ConceptClassifier
from .config import CONFIG
from .entities import HashingTextEmbedder
from .forecasting import PhantomNodeGenerator
from .graph import RuntimeGraph
from .graph_dynamics import ConvergencePressureField
from .llm import get_llm_client
from .neural import Oracle1Loss, Oracle1Model
from .orchestrators import DynamicsOrchestrator, Oracle1Extended, TopologyInstabilityOrchestrator
from .significance import attach_significance_layer

def build_oracle1_system(latent_dim: int = CONFIG["model_latent_dim"],
                          n_gnn_layers: int = CONFIG["model_n_gnn_layers"],
                          n_heads: int = CONFIG["model_n_heads"],
                          llm_client=None,
                          embedder=None,
                          attach_significance: bool = True,
                          attach_canonical: bool = True) -> Dict[str, Any]:
    """
    Convenience factory that wires all modules into a ready-to-use system.
    If ``llm_client`` is None, will attempt to initialize a default local model
    using CONFIG['llm_default_model'].

    Returns a dict with keys:
        graph            — RuntimeGraph (empty, ready to populate)
        model            — Oracle1Model (neural weights)
        extended         — Oracle1Extended (main query interface)
        dynamics         — DynamicsOrchestrator (convergence + virtual nodes)
        topology         — TopologyInstabilityOrchestrator (rupture detection)
        pressure_field   — ConvergencePressureField (shared between orchestrators)
        phantom_gen      — PhantomNodeGenerator (shared between orchestrators)
        loss             — Oracle1Loss
        significance     — SignificanceProcessor (epistemic gradient layer, optional)
        canonical        — CanonicalAssembler (canonical key system, optional)
    """
    # Every graph needs stable text features.  The hashing embedder is a
    # deterministic operational fallback, not another predictive model; a
    # caller may supply a stronger local sentence encoder explicitly.
    if embedder is None:
        embedder = HashingTextEmbedder()

    # Attempt to initialize a default llm_client if not provided
    if llm_client is None:
        try:
            llm_client = get_llm_client()
            logger.info(f"[build_oracle1_system] Initialized default LLM: {llm_client}")
        except ValueError as e:
            logger.warning(f"[build_oracle1_system] Failed to init LLM: {e}")
            llm_client = None

    graph      = RuntimeGraph()
    model      = Oracle1Model(latent_dim=latent_dim, n_gnn_layers=n_gnn_layers, n_heads=n_heads)
    loss       = Oracle1Loss()
    pressure   = ConvergencePressureField()
    phantom_gen = PhantomNodeGenerator()
    # TODO #4 fix: thread the SAME phantom_gen into Oracle1Extended that
    # dynamics/topology receive below — see the comment in
    # Oracle1Extended.__init__ for why this matters (previously extended had
    # its own, disconnected instance and scan_for_voids()'s output never
    # reached dynamics.phase_aligner or topology's rupture-confirmation).
    extended   = Oracle1Extended(model, graph, phantom_gen=phantom_gen)
    dynamics   = DynamicsOrchestrator(graph, pressure, phantom_gen, llm_client=llm_client)
    topology   = TopologyInstabilityOrchestrator(graph, pressure, phantom_gen=phantom_gen)
    system = {
        "graph":         graph,
        "model":         model,
        "extended":      extended,
        "dynamics":      dynamics,
        "topology":      topology,
        "pressure_field": pressure,
        "phantom_gen":   phantom_gen,
        "loss":          loss,
        # TODO #1 support: exposed so callers (e.g. train.py's ingestion
        # phase) can reuse the same client/embedder instead of constructing
        # a second one. May be None (see the try/except above) — ingestion
        # callers must check before use.
        "llm_client":    llm_client,
        "embedder":      embedder,
    }

    # ── Optional: Significance Gradient Layer ────────────────────────────────
    if attach_significance:
        try:
            sig_proc = attach_significance_layer(system,
                                                 llm_client=llm_client,
                                                 embedder=embedder)
            logger.info("[build_oracle1_system] SignificanceProcessor attached")
        except NameError:
            logger.warning("[build_oracle1_system] SignificanceProcessor not available "
                           "(oracle1_significance module not inlined yet)")

    # ── Optional: Canonical Key / Assembler ──────────────────────────────────
    if attach_canonical:
        try:
            classifier = ConceptClassifier()
            assembler = CanonicalNodeAssembler(classifier=classifier,
                                               embedder=embedder,
                                               llm_fallback=llm_client)
            system["canonical"] = assembler
            system["concept_classifier"] = classifier
            logger.info("[build_oracle1_system] CanonicalNodeAssembler attached")
        except NameError:
            logger.warning("[build_oracle1_system] CanonicalNodeAssembler not available "
                           "(oracle1_canonical_key module not inlined yet)")

    return system

__all__ = ['build_oracle1_system']
