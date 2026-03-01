# ================================================================
# Oracle-1: Graph-Native Knowledge Model
# All the code is in one file to make it easier for you to send it to the chatbot for analysis.
# ================================================================

from __future__ import annotations
  
import bisect
import hashlib
import heapq
import json
import math
import os
import pickle
import re
import statistics
import struct
import time
import uuid
import logging
from collections import Counter, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

logger = logging.getLogger("oracle1")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch / torch-geometric not installed — stubs active")

    class nn:
        class Module:
            def __init__(self): pass
            def parameters(self): return iter([])
        class Linear:
            def __init__(self, *a, **kw): pass
        class LayerNorm:
            def __init__(self, *a, **kw): pass
        class Dropout:
            def __init__(self, *a, **kw): pass
        class ModuleList(list): pass
        class ModuleDict(dict): pass
        class Sequential:
            def __init__(self, *a): pass
        class GRU:
            def __init__(self, *a, **kw): pass
        GELU = object

    class torch:
        class Tensor: pass
        @staticmethod
        def cat(t, dim=0): return None
        @staticmethod
        def zeros(*a, **kw): return None
        @staticmethod
        def ones(*a, **kw): return None
        @staticmethod
        def sigmoid(x): return x
        @staticmethod
        def stack(t): return None
        @staticmethod
        def arange(*a): return None
        @staticmethod
        def exp(x): return x
        @staticmethod
        def log(x): return x
        @staticmethod
        def tensor(x): return x
        @staticmethod
        def relu(x): return x
        @staticmethod
        def clamp(x, *a): return x

    F = None

    class GATv2Conv:
        def __init__(self, *a, **kw): pass


@contextmanager
def _maybe_no_grad():
    if TORCH_AVAILABLE:
        with torch.no_grad():
            yield
    else:
        yield


# ══════════════════════════════════════════════════════════════════════════════
#  UNIFIED CONFIGURATION
#  All tuneable parameters live here.  Edit this block; do NOT scatter magic
#  numbers throughout the rest of the file.
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {

    # ── Edge weight components & blending ────────────────────────────────────
    "edge_weight_components": [
        "semantic_similarity",
        "temporal_proximity",
        "limitation_resolution",
        "citation_link",
        "investment_correlation",
        "social_correlation",
    ],
    "v6_edge_weights": {
        "semantic":              0.30,
        "temporal":              0.15,
        "limitation_resolution": 0.25,
        "citation":              0.10,
        "investment":            0.10,
        "social":                0.10,
    },
    "v6_readiness_weights": {
        "scientific":  0.25,
        "investment":  0.25,
        "social":      0.20,
        "maturity":    0.15,
        "group_size":  0.15,
    },

    # ── Inhibitory edges ──────────────────────────────────────────────────────
    "inhibitory_penalty_cap": 0.80,
    "inhibitory_weight":     -0.35,

    # ── Topology / tension ───────────────────────────────────────────────────
    "attention_gamma":                2.5,
    "collapse_threshold":             0.7,
    "virtual_node_tension_threshold": 0.85,

    # Tension field component weights (must sum to 1.0)
    "tension_weights": {
        "upstream_pressure":    0.35,
        "inhibitory_drag":      0.25,
        "connectivity_entropy": 0.20,
        "anomaly_density":      0.15,
        "phantom_pressure":     0.05,
    },
    "tension_conflict_bonus_cap": 0.15,
    "tension_conflict_bonus_scale": 0.15,

    # ── Phase-transition alert thresholds: (min_pressure, min_conflict, min_entropy)
    "phase_thresholds": {
        "OVERLOAD":            (0.60, 0.0,  0.0),
        "VOID_COLLAPSE":       (0.50, 0.0,  0.70),
        "DOMAIN_TRANSPLANT":   (0.30, 0.50, 0.0),
        "ANOMALY_SATURATION":  (0.40, 0.40, 0.0),
        "ONTOLOGICAL_RUPTURE": (0.65, 0.10, 0.60),
    },

    # Alert level tension boundaries
    "alert_thresholds": {
        "none":     0.40,
        "elevated": 0.65,
        "critical": 0.82,
        # above critical → "rupture"
    },

    # ── Physical substrate ────────────────────────────────────────────────────
    "physical_substrate_dim": 10,
    "physical_axis_order": [
        "resource_intensity", "scale_feasibility", "constraint_proximity",
        "reversibility", "parallel_deployability", "environmental_coupling",
        "cross_scale_stability", "synthesis_complexity", "longevity", "cascadability",
    ],
    # Weights used by feasibility_score()
    "feasibility_axis_weights": {
        "constraint_proximity":  0.50,
        "resource_intensity":    0.30,
        "synthesis_complexity":  0.20,
    },
    "feasibility_floor": 0.05,

    # ── Rupture physics relaxation ────────────────────────────────────────────
    "rupture_collapse_factor": 0.35,
    "rupture_default_axes": [
        "constraint_proximity", "synthesis_complexity", "resource_intensity",
    ],

    # ── Convergence pressure field ────────────────────────────────────────────
    "pressure_singularity_threshold": 3.5,
    "pressure_decay_rate":            0.05,
    "pressure_void_weights": {
        "unresolved_limitation": 0.35,
        "cultural_phantom":      0.20,
        "excess_investment":     0.25,
        "inhibitory_sink":       0.10,
        "temporal_zone":         0.10,
    },

    # ── LLM configuration ─────────────────────────────────────────────────
    "llm_local_models": [
        "DeepSeek-R1",
        "Llama 3.1 405B Instruct",
        "Qwen3.5 122B",
    ],
    # default model name used when no client is supplied explicitly
    "llm_default_model": "DeepSeek-R1",

    "pressure_limitation_threshold":  0.3,
    "pressure_investment_threshold":  0.5,
    "pressure_invest_log_scale":      20.0,
    "pressure_zone_multiplier_scale": 0.5,
    "pressure_limitation_scale":      2.0,
    "pressure_unresolved_gate":       0.5,

    # ── Convergence accelerometer ─────────────────────────────────────────────
    "accelerometer_window_epochs": 20,
    "accelerometer_epochs_per_year": 12,
    "accelerometer_c_weights": {
        "semantic":   0.25,
        "edge":       0.20,
        "pressure":   0.25,
        "resource":   0.15,
        "resonance":  0.15,
    },
    "accelerometer_pressure_norm": 5.0,

    # ── Dormancy tracker ──────────────────────────────────────────────────────
    "dormancy_awakening_threshold": 0.6,
    "dormancy_active_threshold":    0.15,
    "dormancy_heat_decay_factor":   0.3,
    "dormancy_growth_per_epoch":    0.01,

    # ── Phantom node generator ────────────────────────────────────────────────
    "phantom_edge_prob_threshold":  0.70,
    "phantom_gap_threshold":        2.0,
    "phantom_confirmation_sim":     0.75,
    "phantom_max_count":            50,
    "phantom_void_top_candidates":  50,
    # Confirmation by rupture
    "phantom_rupture_anomaly_threshold": 0.55,
    "phantom_rupture_proximity_hops":    3,
    # Year-range forecast: base_years + gap_pressure * scale_years
    "phantom_year_range_base":   5,
    "phantom_year_range_scale":  3,

    # ── Phantom phase aligner (PhantomLagModel) ───────────────────────────────
    "lag_log_mean":  3.2,
    "lag_log_sigma": 0.40,
    # Lag reduction by physical readiness
    "lag_readiness_low_boundary":  0.20,
    "lag_readiness_high_boundary": 0.60,
    "lag_linear_max_reduction":    0.70,   # fraction of expected_lag removed at high_boundary
    "lag_collapse_factor":         0.15,   # fraction of expected_lag in singularity-collapse phase
    "lag_collapse_min_years":      1.0,
    # Inhibitory drag cap
    "lag_inhibitory_drag_per_incumbent": 0.20,
    "lag_inhibitory_drag_incumbent_sv_threshold": 8.0,
    "lag_inhibitory_drag_cap": 0.50,
    # Phase boundaries
    "lag_phase_stagnant_readiness": 0.3,
    "lag_phase_stagnant_age_years": 10,
    "lag_phase_imminent_readiness": 0.6,
    "lag_phase_imminent_lag_years": 3,
    "lag_phase_early_readiness":    0.3,
    "lag_phase_early_age_years":    5,
    # Confidence calculation
    "lag_confidence_base":        0.4,
    "lag_confidence_readiness_w": 0.3,
    "lag_confidence_accel_w":     0.3,
    "lag_confidence_accel_tanh_scale": 10.0,

    # ── Recursive Forecasting Loop ────────────────────────────────────────────
    "forecast_min_confidence":  0.10,
    "forecast_decay_factor":    0.65,
    "forecast_max_depth":       5,
    "forecast_max_branches":    3,
    "forecast_years_per_depth": 1.5,
    "forecast_sec_effect_depth": 2,  # depth >= this → secondary effect
    # Incumbent reaction
    "incumbent_threat_threshold": 0.60,   # normalised 0-1
    "incumbent_sv_threshold":     8.0,    # strategic_value cutoff
    # Efficiency plateau penalty
    "plateau_penalty_scale": 0.40,
    "plateau_penalty_floor":  0.05,
    # Lag multiplication per inhibitory force unit
    "incumbent_lag_years_per_force": 3.0,

    # ── Structural conflict meter ─────────────────────────────────────────────
    "conflict_var_norm_denom": 0.25,
    "conflict_norm_var_weight": 0.5,
    "conflict_inconsistency_weight": 0.5,

    # ── Anomaly accumulator ───────────────────────────────────────────────────
    "anomaly_window_size": 50,
    "anomaly_ema_alpha":   0.10,
    # Scoring thresholds inside measure_and_push
    "anomaly_invest_high":    0.6,
    "anomaly_resolution_low": 0.2,
    "anomaly_invest_bonus":   0.15,
    "anomaly_degree_scale":   0.05,
    "anomaly_degree_cap":     0.20,
    "anomaly_variance_threshold": 0.15,
    "anomaly_variance_bonus": 0.10,

    # ── Graph entropy monitor ─────────────────────────────────────────────────
    "entropy_singularity_drop_threshold": 0.3,
    "entropy_pre_singularity_entropy":    0.85,
    "entropy_pre_singularity_pressure":   2.5,
    "entropy_high_prev_threshold":        0.7,

    # ConnectivityEntropyCalculator
    "connectivity_entropy_brittle_threshold": 0.20,
    "connectivity_entropy_diffuse_threshold": 0.85,

    # ── Cross-domain isomorphism amplifier ────────────────────────────────────
    "isomorphism_threshold":   0.60,
    "isomorphism_amplification_factor": 1.25,

    # ── Obligatory bridge detector ────────────────────────────────────────────
    "bridge_sem_pull_threshold":  0.50,
    "bridge_pressure_threshold":  0.30,
    "bridge_capacity_threshold":  0.15,
    "bridge_max_voids_per_epoch": 100,
    "bridge_top_pressure_nodes":  80,
    "bridge_pair_lookahead":      20,
    "bridge_pressure_norm":       10.0,
    "bridge_rupture_boost":       1.8,
    "bridge_rupture_priority_multiplier": 1.5,
    "bridge_severity_floor":      0.05,
    "bridge_drag_scale":          0.4,
    "bridge_void_default_target_years": 5,

    # ── Virtual node synthesizer ──────────────────────────────────────────────
    "synth_min_trajectory_confidence": 0.30,
    "synth_min_void_severity":         0.15,
    "synth_max_virtual_nodes":         30,
    "synth_projection_years":          [2, 5, 10, 20],
    "synth_velocity_threshold":        0.005,
    "synth_velocity_confidence_norm":  0.05,
    "synth_readiness_from_velocity":   10.0,
    "synth_readiness_cap":             0.95,
    "synth_readiness_base":            0.4,
    # Projection edge default similarity
    "synth_edge_default_sim":          0.7,
    # Void list scan depth
    "synth_void_scan_depth":           20,

    # ── Virtual node factory (topology) ──────────────────────────────────────
    "vfactory_drag_threshold":    0.5,
    "vfactory_anomaly_threshold": 0.4,
    "vfactory_conflict_threshold":0.5,
    "vfactory_entropy_threshold": 0.3,
    "vfactory_entropy_min_domains": 3,

    # ── Phantom pressure layer ────────────────────────────────────────────────
    "phantom_pressure_low_bound":   0.3,
    "phantom_pressure_low_scale":   0.09,
    "phantom_pressure_high_scale":  0.3,
    "phantom_pressure_max":         0.40,
    # Social correlation gate for boosting phantom_weight
    "phantom_social_boost_threshold": 0.7,
    "phantom_social_boost_amount":    0.2,
    "phantom_fiction_default_weight": 0.4,

    # ── Contextual priority head ──────────────────────────────────────────────
    "priority_singularity_threshold": 3.5,
    "priority_expo_scale":   0.8,
    "priority_linear_scale": 0.3,
    "priority_clamp_min":    0.01,
    "priority_clamp_max":    100.0,
    "priority_topo_max_edges":       50,
    "priority_topo_max_lim_edges":   20,
    "priority_topo_max_inh_edges":   10,
    "priority_topo_zone_norm":       5.0,
    "priority_lim_threshold":        0.5,
    "priority_inh_threshold":        0.3,

    # ── Graph walker / beam search ────────────────────────────────────────────
    "walker_beam_width":   30,
    "walker_max_path_len": 8,
    "walker_min_conf":     0.03,
    "walker_limit_bonus":  1.5,
    "walker_limit_thresh": 0.5,
    "walker_top_k_start":  5,
    "walker_min_path_len": 3,
    "walker_nbr_scan_cap": 20,

    # ── Path-agnostic inference ───────────────────────────────────────────────
    "path_agnostic_independence_overlap_threshold": 0.5,
    "path_agnostic_conv_prob_decay":                0.5,
    "path_agnostic_top_trajectories":               5,
    "path_agnostic_max_phantoms":                   5,
    "path_agnostic_max_alerts":                     3,
    "path_agnostic_cross_domain_bonus":             0.1,
    "path_agnostic_domain_match_chars":             10,
    "path_agnostic_alert_match_chars":              8,

    # ── Stabilised centroid tracker ───────────────────────────────────────────
    "centroid_alpha_add":   0.05,
    "centroid_alpha_epoch": 0.10,
    "centroid_history_len": 10,
    "centroid_weiszfeld_iterations": 20,
    "centroid_velocity_ema_alpha": 0.1,
    "centroid_velocity_ema_decay": 0.9,
    "centroid_size_inflation_growth": 1.5,
    "centroid_size_inflation_vel_cap": 0.01,
    "centroid_embedding_dim": 384,

    # ── Oracle1Loss ───────────────────────────────────────────────────────────
    "loss_lambda_forecast": 1.0,
    "loss_lambda_exist":    1.0,
    "loss_lambda_comps":    0.5,
    "loss_lambda_zone":     0.3,
    "loss_zone_margin":     1.0,

    # ── Dual objective head ───────────────────────────────────────────────────
    "dual_head_horizon_epochs":  60,
    "dual_head_eta_loss_weight": 0.3,

    # ── GNN model defaults ────────────────────────────────────────────────────
    "model_latent_dim":   512,
    "model_n_gnn_layers": 6,
    "model_n_heads":      8,
    "model_dropout":      0.1,

    # ── TemporalEncoder ───────────────────────────────────────────────────────
    "temporal_encoder_max_years": 300,
    "temporal_encoder_zone_year_shift": 5,   # (zone_mult - 1) * this = extra years

    # ── Feature builder dims ──────────────────────────────────────────────────
    "feat_text_dim":     384,
    "feat_readiness_dim": 5,
    "feat_accel_dim":     4,
    "feat_struct_dim":    3,
    "feat_zone_dim":      4,
    "feat_social_dim":    3,
    "feat_invest_dim":    2,
    "feat_forum_dim":     1,
    "feat_plateau_dim":   1,
    # Normalisation denominators
    "feat_score_max":            10.0,
    "feat_group_count_max":     100.0,
    "feat_zone_mult_max":         5.0,
    "feat_discovery_depth_max":  10.0,
    "feat_invest_log_scale":     20.0,
    "feat_invest_rounds_max":    20.0,
    "feat_forum_log_scale":      10.0,
    "feat_upstream_pressure_cap": 5.0,
    "feat_sdi_cap":               10.0,

    # ── Reconstructed convergence time series ─────────────────────────────────
    "conv_series_window_size": 20,
    "conv_series_vel_scale":   10.0,
    "conv_series_accel_scale": 100.0,
    "conv_series_jerk_scale":  1000.0,

    # ── StructuralDependencyUpdater ───────────────────────────────────────────
    "sdu_upstream_tanh_scale": 5.0,

    # ── Topology tension alert boundaries (OntologicalTensionField) ───────────
    "tension_pressure_zone_scale": 5.0,   # tanh normalisation for upstream_pressure

    # ── RetrospectiveConvergenceLoss ──────────────────────────────────────────
    "conv_loss_max_paths":      100.0,
    "conv_loss_path_scan_nodes": 200,
    "conv_loss_max_hops":          5,

    # ── AnomalyAccumulator degree stats ──────────────────────────────────────
    "anomaly_degree_sigma_scale": 0.05,

    # ── MotherAnnotator / annotation prompt ───────────────────────────────────
    "annotation_text_max_words": 15,

    # ── BucketIndex search ────────────────────────────────────────────────────
    "bucket_max_candidates": 20,   # top-k candidates sent to Pass-D LLM

    # ── ConceptClassifier ─────────────────────────────────────────────────────
    "classifier_confidence_threshold": 0.80,   # below → fall back to LLM
    "classifier_retrain_every":        500,    # new LLM samples between retrains
    "classifier_label_smoothing":      0.05,   # CrossEntropyLoss label smoothing
    "classifier_val_split":            0.15,   # fraction held out for validation
    "classifier_max_head_expand":      True,   # preserve old weights on head grow

    # ── Ingestion pipeline ────────────────────────────────────────────────────
    "ingestion_chunk_size":   8_000,    # max chars per chunk (~3000 tokens)
    "ingestion_chunk_overlap": 800,     # overlap between chunks (chars)
    "ingestion_merge_threshold": 0.92,  # above → same entity (merge)
    "ingestion_link_threshold":  0.65,  # above → semantic cross-doc edge
    "ingestion_source_trust": {
        "arxiv":      0.85, "patent":    0.80, "nature":   0.95,
        "science":    0.95, "preprint":  0.65, "news":     0.40,
        "blog":       0.30, "report":    0.70, "book":     0.75,
        "conference": 0.80, "unknown":   0.50,
    },

    # ── Consolidation engine ──────────────────────────────────────────────────
    "consolidation_conflict_variance_threshold": 2.0,  # std above this → conflict

    # ── Rupture report display ────────────────────────────────────────────────
    "rupture_report_node_text_len":   50,
    "rupture_report_domain_text_len": 50,

    # ── Top-k display lengths ─────────────────────────────────────────────────
    "display_node_text_len":    30,
    "display_entropy_rise_thr":  0.05,
    "display_entropy_fall_thr": -0.05,
    "display_top_tension_n":     5,
    "display_top_conflict_n":    5,
    "display_latent_iso_top_n": 10,

    # ── Ancestry invariant ────────────────────────────────────────────────────
    # Principle: discoveries are recombinations of existing knowledge, not
    # creation ex nihilo.  Every node except the N oldest primordial roots
    # MUST have at least one incoming (non-containment) edge.
    #
    # "primordial_cutoff_n": the N nodes with the earliest timestamps are
    #   exempt — they represent axioms/myths whose predecessors simply aren't
    #   in the corpus (e.g. Archimedes, wave theory of light).
    #
    # "ancestry_synthesiser_candidates": how many existing predecessor nodes
    #   AncestryEnforcer searches when no incoming edge is declared.
    #   Larger = more accurate matching, slower on very large graphs.
    #
    # "ancestry_synthesiser_sem_threshold": minimum embedding cosine similarity
    #   to use the embedding-based best match.  Below this, falls back to the
    #   highest-(scientific_score + strategic_value) predecessor.
    "primordial_cutoff_n":                  10,
    "ancestry_synthesiser_candidates":     200,
    "ancestry_synthesiser_sem_threshold":  0.30,
    "ancestry_edge_semantic_sim":          0.25,   # weight on synthesised edge
    "ancestry_edge_temporal_prox":         0.15,   # weight on synthesised edge
}


# ══════════════════════════════════════════════════════════════════════════════
#  DERIVED / ALIASED CONSTANTS  (do not edit — computed from CONFIG above)
# ══════════════════════════════════════════════════════════════════════════════

EDGE_WEIGHT_COMPONENTS   = CONFIG["edge_weight_components"]
V6_EDGE_WEIGHTS          = CONFIG["v6_edge_weights"]
V6_READINESS_WEIGHTS     = CONFIG["v6_readiness_weights"]

N_EDGE_COMPONENTS   = len(EDGE_WEIGHT_COMPONENTS)   # 6
N_READINESS_DIMS    = len(V6_READINESS_WEIGHTS)     # 5
N_ACCELERATION_DIMS = CONFIG["feat_accel_dim"]       # 4
N_STRUCTURAL_DIMS   = CONFIG["feat_struct_dim"]      # 3

EXTENDED_EDGE_COMPONENTS = EDGE_WEIGHT_COMPONENTS + ["inhibitory_force"]
N_EXTENDED_COMPONENTS    = len(EXTENDED_EDGE_COMPONENTS)   # 7
INHIBITORY_PENALTY_CAP   = CONFIG["inhibitory_penalty_cap"]
INHIBITORY_WEIGHT        = CONFIG["inhibitory_weight"]

ATTENTION_GAMMA               = CONFIG["attention_gamma"]
COLLAPSE_THRESHOLD            = CONFIG["collapse_threshold"]
VIRTUAL_NODE_TENSION_THRESHOLD = CONFIG["virtual_node_tension_threshold"]

ALERT_OVERLOAD            = "OVERLOAD"
ALERT_VOID_COLLAPSE       = "VOID_COLLAPSE"
ALERT_DOMAIN_TRANSPLANT   = "DOMAIN_TRANSPLANT"
ALERT_ANOMALY_SATURATION  = "ANOMALY_SATURATION"
ALERT_ONTOLOGICAL_RUPTURE = "ONTOLOGICAL_RUPTURE"

PHASE_THRESHOLDS = CONFIG["phase_thresholds"]

PHYSICAL_SUBSTRATE_DIM = CONFIG["physical_substrate_dim"]
PHYSICAL_AXIS_ORDER    = CONFIG["physical_axis_order"]


# ── Helpers for LLM initialization based on CONFIG ──────────────────────────
class LocalLLMClient:
    """Minimal stub representing a local model chosen by the user.

    The actual inference logic must be supplied by whichever library
    provides DeepSeek‑R1, Llama 3.1 405B, Qwen3.5, etc.  This placeholder
    exists so that components can be wired together without failing when
    no remote SDK is available.
    """
    def __init__(self, model_name: str):
        self.model = model_name

    def complete(self, prompt: str) -> str:
        """Override this method with your local model's inference logic."""
        raise NotImplementedError(
            f"LocalLLMClient.complete() not implemented for model '{self.model}'. "
            f"Subclass LocalLLMClient and override the complete() method."
        )

    def __repr__(self):
        return f"<LocalLLMClient: {self.model}>"


def get_llm_client(model_name: Optional[str] = None):
    """Return an llm_client according to CONFIG settings.

    If ``model_name`` is given it overrides ``CONFIG['llm_default_model']``.
    At the moment only local models are supported; remote providers should
    be added here when needed.
    """
    name = model_name or CONFIG.get("llm_default_model")
    if name in CONFIG.get("llm_local_models", []):
        return LocalLLMClient(name)
    # fallback: no client available
    raise ValueError(f"LLM model '{name}' is not configured or supported")



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
    provenance: str = ""
    timestamp: float = 0.0
    publication_date: Optional[str] = None

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

    zone_id:               str   = ""
    acceleration_multiplier: float = 1.0

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
        self.total_weight = (
            self.semantic_similarity    * w["semantic"]              +
            self.temporal_proximity     * w["temporal"]              +
            self.limitation_resolution  * w["limitation_resolution"] +
            self.citation_link          * w["citation"]              +
            self.investment_correlation * w["investment"]            +
            self.social_correlation     * w["social"]               +
            INHIBITORY_WEIGHT           * self.inhibitory_force
        )
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


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

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

    TOTAL_DIM = (TEXT_DIM + READINESS_DIM + ACCEL_DIM + STRUCT_DIM +
                 ZONE_DIM + SOCIAL_DIM + INVEST_DIM + FORUM_DIM + PLATEAU_DIM)  # 407

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
    }

    def __init__(self, sentence_embedder):
        self.embedder = sentence_embedder

    def build(self, node: KnowledgeNode) -> np.ndarray:
        vec = np.zeros(self.TOTAL_DIM, dtype=np.float32)
        emb = self.embedder.encode(node.text or node.entity_type)[:self.TEXT_DIM]
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


class MultiComponentEdgeAttention(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, out_dim: int,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Fix 6: use all 7 extended edge streams (including inhibitory_force)
        self.n_edge_streams = N_EXTENDED_COMPONENTS
        self.out_dim = out_dim
        if not TORCH_AVAILABLE:
            return
        head_dim = out_dim // n_heads
        self.edge_convs = nn.ModuleList([
            GATv2Conv(node_dim, head_dim, heads=n_heads, edge_dim=1,
                      dropout=dropout, concat=True)
            for _ in range(self.n_edge_streams)
        ])
        self.containment_gate = nn.Linear(node_dim, out_dim)
        self.stream_mixer  = nn.Linear(node_dim, self.n_edge_streams)
        # Fix 6: out_proj now accepts 7 streams * out_dim
        self.out_proj      = nn.Linear(out_dim * self.n_edge_streams, out_dim)
        # Fix 7: projection for mixed residual (same dim as projected)
        self.mix_proj      = nn.Linear(out_dim, out_dim)
        self.norm          = nn.LayerNorm(out_dim)
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, containment_index,
                containment_zone_mult, edge_priority_weights=None):
        # edge_priority_weights: [E, 1] tensor from ContextualPriorityHead (Fix 1)
        if not TORCH_AVAILABLE:
            return x
        stream_outs = []
        for k, conv in enumerate(self.edge_convs):
            stream_attr = edge_attr[:, k:k+1]
            # Fix 1: modulate each stream's edge attributes by topological priority
            if edge_priority_weights is not None:
                stream_attr = stream_attr * edge_priority_weights
            stream_out  = conv(x, edge_index, edge_attr=stream_attr)
            stream_outs.append(stream_out)
        stacked = torch.stack(stream_outs, dim=1)
        mix = torch.softmax(self.stream_mixer(x), dim=-1).unsqueeze(-1)
        # Fix 7: compute weighted mix and use it as a residual signal via mix_proj
        mixed = (stacked * mix).sum(dim=1)
        cat = torch.cat(stream_outs, dim=-1)
        projected = self.out_proj(cat) + self.mix_proj(mixed)  # Fix 7: mixed now contributes
        if containment_index.size(1) > 0:
            member_ids = containment_index[1]
            zone_mult_expanded = containment_zone_mult.unsqueeze(-1)
            zone_signal = self.containment_gate(x[member_ids])
            zone_signal = zone_signal * zone_mult_expanded
            projected = projected.clone()
            projected.scatter_add_(
                0, member_ids.unsqueeze(-1).expand_as(zone_signal),
                zone_signal * 0.1)
        result = self.norm(
            self.dropout(projected) + x
            if projected.size(-1) == x.size(-1)
            else self.dropout(projected))
        return result


class StructuralDependencyUpdater(nn.Module):
    """Computes per-node SDI/cascade/upstream-pressure signals and
    returns both the updated node_features (for bookkeeping) and a
    3-dim structural signal vector that Oracle1Model can inject into
    the latent space between GNN layers (Fix 9)."""

    def __init__(self):
        super().__init__()

    def forward(self, node_features, edge_index, edge_weights,
                node_feature_builder):
        if not TORCH_AVAILABLE:
            return node_features, None
        N = node_features.size(0)
        sdi  = torch.zeros(N)
        casc = torch.zeros(N)
        src, tgt = edge_index
        sdi.scatter_add_(0, tgt, edge_weights)
        casc.scatter_add_(0, src, edge_weights)
        up = sdi - casc
        s = NodeFeatureBuilder.SLICES["struct"]
        node_features = node_features.clone()
        node_features[:, s.start]   = sdi  / (sdi.max()  + 1e-8)
        node_features[:, s.start+1] = casc / (casc.max() + 1e-8)
        node_features[:, s.start+2] = torch.tanh(up / CONFIG["sdu_upstream_tanh_scale"])
        # Fix 9: also return the 3-dim structural signal so Oracle1Model can
        # gate it into the latent representations between GNN layers.
        struct_signal = node_features[:, s.start:s.start+3].detach()
        return node_features, struct_signal


class ForecastHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.readiness_head = nn.Sequential(
            nn.Linear(NodeFeatureBuilder.READINESS_DIM, 32), nn.GELU(), nn.Linear(32, 1))
        self.accel_head = nn.Sequential(
            nn.Linear(NodeFeatureBuilder.ACCEL_DIM, 32), nn.GELU(), nn.Linear(32, 1))
        self.structural_head = nn.Sequential(
            nn.Linear(NodeFeatureBuilder.STRUCT_DIM, 32), nn.GELU(), nn.Linear(32, 1))
        self.zone_gate = nn.Sequential(
            nn.Linear(NodeFeatureBuilder.ZONE_DIM, 16), nn.GELU(),
            nn.Linear(16, 1), nn.Softplus())
        self.global_refine = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, node_latent, node_features_raw):
        if not TORCH_AVAILABLE:
            return None
        s = NodeFeatureBuilder.SLICES
        readiness_score  = self.readiness_head(node_features_raw[:, s["readiness"]])
        accel_score      = self.accel_head(node_features_raw[:, s["accel"]])
        structural_score = self.structural_head(node_features_raw[:, s["struct"]])
        zone_mult        = self.zone_gate(node_features_raw[:, s["zone"]])
        global_residual  = self.global_refine(node_latent)
        base = (readiness_score + accel_score + structural_score) * zone_mult
        return base + global_residual * 0.1  # fine-tuning residual – see CONFIG model_latent_dim


class EdgePredictionHead(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        if not TORCH_AVAILABLE:
            return
        self.shared = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim // 2), nn.GELU())
        self.exist_head = nn.Linear(latent_dim // 2, 1)
        self.component_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim // 2, 32), nn.GELU(),
                          nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(N_EDGE_COMPONENTS)
        ])

    def forward(self, h_src, h_tgt):
        if not TORCH_AVAILABLE:
            return None, None
        pair   = torch.cat([h_src, h_tgt], dim=-1)
        shared = self.shared(pair)
        exist  = torch.sigmoid(self.exist_head(shared))
        comps  = torch.cat([head(shared) for head in self.component_heads], dim=-1)
        return exist, comps


class Oracle1Model(nn.Module):
    def __init__(self,
                 node_in_dim:  int   = NodeFeatureBuilder.TOTAL_DIM,
                 latent_dim:   int   = 512,
                 n_gnn_layers: int   = 6,
                 n_heads:      int   = 8,
                 dropout:      float = 0.1):
        super().__init__()
        self.latent_dim   = latent_dim
        self.n_gnn_layers = n_gnn_layers
        if not TORCH_AVAILABLE:
            return
        self.input_proj = nn.Sequential(
            nn.Linear(node_in_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Dropout(dropout))
        self.temporal_enc = TemporalEncoder(latent_dim)
        self.gnn_layers = nn.ModuleList([
            MultiComponentEdgeAttention(
                node_dim=latent_dim, edge_dim=EdgeFeatureBuilder.DIM,
                out_dim=latent_dim, n_heads=n_heads, dropout=dropout)
            for _ in range(n_gnn_layers)
        ])
        self.struct_updater = StructuralDependencyUpdater()
        # Fix 9: learned gate that injects SDI/cascade/upstream signals into latent space
        self.struct_gate = nn.Sequential(
            nn.Linear(3, latent_dim), nn.Tanh())
        # Fix 4: ContextualPriorityHead is now a registered submodule — its
        # parameters will be updated by the optimizer automatically.
        self.priority_head = ContextualPriorityHead(latent_dim)
        self.forecast_head  = ForecastHead(latent_dim)
        self.edge_pred_head = EdgePredictionHead(latent_dim)
        self.trajectory_gru = nn.GRU(latent_dim, latent_dim, batch_first=True)

    def encode(self, node_features, edge_index, edge_attr,
               containment_index, containment_zone_mult,
               node_years, node_zone_mult,
               topology_vecs=None):
        # topology_vecs: [N, 12] tensor of LocalTopologyStats vectors, one per node.
        # When provided, edge-level priority weights are computed via priority_head
        # and passed to each GNN layer to modulate attention (Fix 1 + Fix 4).
        if not TORCH_AVAILABLE:
            return None, node_features
        x = self.input_proj(node_features)
        x = self.temporal_enc(x, node_years, node_zone_mult)

        # Pre-compute per-edge priority weights from topology if available (Fix 1)
        edge_priority_weights = None
        if topology_vecs is not None and edge_index.size(1) > 0:
            src_ids = edge_index[0]
            tgt_ids = edge_index[1]
            h_src_init = x[src_ids]
            h_tgt_init = x[tgt_ids]
            topo_src = topology_vecs[src_ids]  # [E, 12]
            # Use source topology for edge priority (single-pass approximation)
            edge_priority_weights = self.priority_head(
                h_src_init, h_tgt_init,
                topo_src,
                time_delta=torch.zeros(h_src_init.size(0), 1,
                                       device=h_src_init.device))  # [E, 1]

        for layer in self.gnn_layers:
            # Fix 1: pass priority weights into each GNN layer
            x = layer(x, edge_index, edge_attr, containment_index,
                      containment_zone_mult,
                      edge_priority_weights=edge_priority_weights)
            # Fix 9: update node_features and inject structural signal into latents
            edge_weights = (edge_attr[:, :len(V6_EDGE_WEIGHTS)] * torch.tensor(
                list(V6_EDGE_WEIGHTS.values()), dtype=torch.float32,
                device=edge_attr.device)).sum(-1)
            node_features, struct_signal = self.struct_updater(
                node_features, edge_index, edge_weights, None)
            if struct_signal is not None:
                # Gate the 3-dim structural signal into latent space
                x = x + self.struct_gate(
                    struct_signal.to(x.device)) * 0.1
        return x, node_features

    def predict_forecast(self, node_latent, node_features):
        return self.forecast_head(node_latent, node_features)

    def predict_edge(self, h_src, h_tgt):
        return self.edge_pred_head(h_src, h_tgt)


class Oracle1Loss:
    def __init__(self, lambda_forecast=CONFIG["loss_lambda_forecast"],
                 lambda_exist=CONFIG["loss_lambda_exist"],
                 lambda_comps=CONFIG["loss_lambda_comps"],
                 lambda_zone=CONFIG["loss_lambda_zone"],
                 zone_margin=CONFIG["loss_zone_margin"],
                 # Fix 2: weight for retrospective convergence loss
                 lambda_conv: float = 0.5,
                 # Fix 5: weight for dual-objective (priority + eta) loss
                 lambda_dual: float = 0.3):
        self.lf = lambda_forecast
        self.le = lambda_exist
        self.lc = lambda_comps
        self.lz = lambda_zone
        self.zone_margin = zone_margin
        self.lambda_conv = lambda_conv   # Fix 2
        self.lambda_dual = lambda_dual   # Fix 5
        # Fix 2: retrospective convergence loss module
        self._retro_loss = RetrospectiveConvergenceLoss()

    def forecast_loss(self, predicted, ground_truth):
        if not TORCH_AVAILABLE: return 0.0
        return F.mse_loss(predicted.squeeze(-1), ground_truth)

    def existence_loss(self, exist_probs, labels):
        if not TORCH_AVAILABLE: return 0.0
        return F.binary_cross_entropy(exist_probs.squeeze(-1), labels)

    def component_loss(self, pred_comps, true_comps, edge_mask):
        if not TORCH_AVAILABLE: return 0.0
        if edge_mask.sum() == 0: return torch.tensor(0.0)
        return F.mse_loss(pred_comps[edge_mask], true_comps[edge_mask])

    def zone_coherence_loss(self, node_latents, zone_memberships):
        if not TORCH_AVAILABLE or F is None: return 0.0
        loss = torch.tensor(0.0)
        zones = [z for z in zone_memberships.unique().tolist() if z >= 0]
        for zone_id in zones:
            members = (zone_memberships == zone_id).nonzero(as_tuple=True)[0]
            if len(members) < 2: continue
            for i in range(len(members) - 1):
                a, b = members[i], members[i + 1]
                loss = loss + F.mse_loss(node_latents[a], node_latents[b])
        if len(zones) > 1:
            for i in range(len(zones)):
                for j in range(i + 1, min(i + 3, len(zones))):
                    a = (zone_memberships == zones[i]).nonzero(as_tuple=True)[0][0]
                    b = (zone_memberships == zones[j]).nonzero(as_tuple=True)[0][0]
                    dist = F.pairwise_distance(node_latents[a].unsqueeze(0),
                                               node_latents[b].unsqueeze(0))
                    loss = loss + F.relu(self.zone_margin - dist)
        return self.lz * loss

    def total(self, pred_forecast, true_forecast, exist_probs, exist_labels,
              pred_comps, true_comps, edge_mask, node_latents, zone_memberships,
              # Fix 2: optional retrospective convergence supervision
              pred_convergence_scores=None, historical_truths=None,
              paths_densities=None,
              # Fix 5: optional dual-objective supervision
              dual_priority_pred=None, dual_eta_pred=None,
              breakthrough_labels=None, actual_etas=None):
        l1 = self.forecast_loss(pred_forecast, true_forecast)
        l2 = self.existence_loss(exist_probs, exist_labels)
        l3 = self.component_loss(pred_comps, true_comps, edge_mask)
        l4 = self.zone_coherence_loss(node_latents, zone_memberships)
        loss = self.lf*l1 + self.le*l2 + self.lc*l3 + l4

        # Fix 2: add RetrospectiveConvergenceLoss when historical supervision is available
        if (pred_convergence_scores is not None and
                historical_truths is not None and
                paths_densities is not None):
            l5 = self._retro_loss(pred_convergence_scores,
                                   historical_truths, paths_densities)
            loss = loss + self.lambda_conv * l5

        # Fix 5: add DualObjectiveHead loss (priority + eta) when labels are available
        if (dual_priority_pred is not None and
                breakthrough_labels is not None and
                actual_etas is not None):
            bce = F.binary_cross_entropy(
                dual_priority_pred.squeeze(-1), breakthrough_labels)
            mask = breakthrough_labels > 0.5
            eta_loss = torch.tensor(0.0)
            if mask.sum() > 0:
                horizon = CONFIG["dual_head_horizon_epochs"]
                eta_loss = F.mse_loss(
                    dual_eta_pred.squeeze(-1)[mask] / horizon,
                    actual_etas[mask] / horizon)
            l6 = bce + CONFIG["dual_head_eta_loss_weight"] * eta_loss
            loss = loss + self.lambda_dual * l6

        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def _ts_to_year(ts: float) -> int:
    try:
        return datetime.utcfromtimestamp(ts).year
    except Exception:
        return int(ts / 31536000 + 1970)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME GRAPH
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  ANCESTRY ENFORCER
#
#  "Discoveries don't arise from nothing — they are reorganisations,
#   mutations, and combinations of existing knowledge."
#
#  AncestryEnforcer implements this as a hard invariant on the graph:
#   • The N oldest nodes (by timestamp) are "primordial roots" — exempt,
#     because their true predecessors are simply outside our corpus.
#   • Every other node must have at least one incoming non-containment edge
#     before entering RuntimeGraph.
#   • When a batch of nodes is added without a sufficient incoming edge
#     (common during ingestion when a document mentions a concept for the
#     first time), the enforcer synthesises a lightweight "DERIVED_FROM" edge
#     to the best-matching existing predecessor.
#
#  Three matching strategies, in priority order:
#   1. Embedding cosine similarity (if both nodes have embeddings and
#      similarity ≥ ancestry_synthesiser_sem_threshold).
#   2. Domain affinity: highest (scientific_score + strategic_value) among
#      nodes in the same domain that predate the new node.
#   3. Global fallback: highest (scientific_score + strategic_value) among
#      ANY node that predates the new node.
#
#  The synthesised edge is tagged with relationship_type="DERIVED_FROM" and
#  discovery_method="ancestry_synthesised" so downstream code can distinguish
#  it from evidence-backed edges extracted from documents.  Its component
#  weights are kept modest so it doesn't dominate GNN attention.
# ══════════════════════════════════════════════════════════════════════════════

class AncestryEnforcer:
    """Enforces the ancestry invariant when nodes are added to RuntimeGraph.

    Usage — called automatically by RuntimeGraph methods; no manual
    instantiation needed.  The singleton ANCESTRY_ENFORCER is created at
    module level and shared across the session.
    """

    def is_primordial(self, node: "KnowledgeNode", graph: "RuntimeGraph") -> bool:
        """True if this node is among the N temporally earliest — exempt from the invariant.

        temporal_sorted is maintained in ascending order, so we read the first
        primordial_cutoff_n entries.  Temporal-zone nodes are also exempt because
        they are structural containers, not knowledge claims.
        """
        if node.is_temporal_zone:
            return True
        n = CONFIG["primordial_cutoff_n"]
        primordial_ids = {nid for _, nid in graph.temporal_sorted[:n]}
        return node.id in primordial_ids

    def has_incoming(self, node_id: str, graph: "RuntimeGraph") -> bool:
        """True if node_id already has at least one incoming non-containment edge.

        Delegates to RuntimeGraph's O(1) in-degree cache instead of scanning
        all edges (which was O(E) and catastrophic at scale).
        """
        return graph.has_incoming_edge(node_id)

    def has_incoming_in_batch(
        self,
        node_id: str,
        batch_edges: List["KnowledgeEdge"],
    ) -> bool:
        """True if any edge in the current batch points TO node_id (non-containment)."""
        return any(
            e.target_id == node_id and not e.is_containment_edge
            for e in batch_edges
        )

    def find_best_predecessor(
        self,
        node: "KnowledgeNode",
        graph: "RuntimeGraph",
    ) -> Optional["KnowledgeNode"]:
        """Return the best predecessor node to anchor the new node to.

        Predecessor = already in graph AND (timestamp < node.timestamp OR
        node.timestamp == 0).  Returns None if the graph is empty.

        Scalability: uses graph.temporal_sorted (always-sorted list) with
        bisect to locate the eligible prefix in O(log N), then heapq.nlargest
        to find the top-scored candidates in O(P log K) where P is the
        eligible count and K = ancestry_synthesiser_candidates — never
        materialising the full predecessor list into memory.
        """
        if not graph.temporal_sorted:
            return None

        max_cands = CONFIG["ancestry_synthesiser_candidates"]

        if node.timestamp == 0.0:
            # All existing nodes are eligible predecessors
            eligible_end = len(graph.temporal_sorted)
        else:
            # Binary search: find insertion point of node.timestamp
            # Everything to the left has timestamp <= node.timestamp
            eligible_end = bisect.bisect_right(
                graph.temporal_sorted, (node.timestamp, "")
            )
            if eligible_end == 0:
                return None

        # Use heapq.nlargest to avoid materialising all eligible candidates.
        # Score function: scientific_score + strategic_value (same as before).
        # We stream from temporal_sorted[0:eligible_end] to stay O(P log K).
        def _score_and_node(idx: int):
            _, nid = graph.temporal_sorted[idx]
            if nid == node.id:
                return None
            n = graph.nodes.get(nid)
            if n is None or n.is_temporal_zone:
                return None
            return (n.scientific_score + n.strategic_value, n)

        top_scored = heapq.nlargest(
            max_cands,
            filter(None, (_score_and_node(i) for i in range(eligible_end))),
            key=lambda x: x[0],
        )
        if not top_scored:
            return None
        candidates = [n for _, n in top_scored]

        threshold = CONFIG["ancestry_synthesiser_sem_threshold"]
        best: Optional["KnowledgeNode"] = None
        best_sim = -1.0

        # Strategy 1: embedding cosine similarity
        if node.embedding is not None:
            new_norm = float(np.linalg.norm(node.embedding))
            if new_norm > 1e-8:
                for cand in candidates:
                    if cand.embedding is None:
                        continue
                    c_norm = float(np.linalg.norm(cand.embedding))
                    if c_norm < 1e-8:
                        continue
                    sim = float(np.dot(node.embedding, cand.embedding) / (new_norm * c_norm))
                    if sim > best_sim:
                        best_sim = sim
                        best = cand

        if best is not None and best_sim >= threshold:
            return best

        # Strategy 2: same-domain affinity
        domain_cands = [c for c in candidates if c.domain == node.domain]
        if domain_cands:
            return max(domain_cands, key=lambda n: n.scientific_score + n.strategic_value)

        # Strategy 3: global best-influence fallback
        return max(candidates, key=lambda n: n.scientific_score + n.strategic_value)

    def synthesise_edge(
        self,
        predecessor: "KnowledgeNode",
        new_node: "KnowledgeNode",
        similarity: float = 0.0,
    ) -> "KnowledgeEdge":
        """Build a synthetic DERIVED_FROM edge.

        Uses fixed modest weights so the synthesised lineage does not drown out
        real evidence-backed edges during GNN training.  The edge is tagged
        with discovery_method="ancestry_synthesised" for auditability.
        """
        sem_sim = float(np.clip(similarity, 0.0, 1.0)) if similarity >= CONFIG["ancestry_synthesiser_sem_threshold"] else CONFIG["ancestry_edge_semantic_sim"]
        edge = KnowledgeEdge(
            id=f"anc_{predecessor.id[:10]}_{new_node.id[:10]}",
            source_id=predecessor.id,
            target_id=new_node.id,
            relationship_type="DERIVED_FROM",
            discovery_method="ancestry_synthesised",
            semantic_similarity=sem_sim,
            temporal_proximity=CONFIG["ancestry_edge_temporal_prox"],
            limitation_resolution=0.0,
            citation_link=0.0,
            investment_correlation=0.0,
            social_correlation=0.0,
            inhibitory_force=0.0,
            timestamp=new_node.timestamp,
            confidence=float(np.clip(similarity if similarity > 0 else 0.3, 0.1, 0.9)),
            evidence=["auto-synthesised ancestry lineage"],
        )
        edge.compute_total_weight()
        return edge

    def enforce(
        self,
        node: "KnowledgeNode",
        graph: "RuntimeGraph",
        batch_edges: Optional[List["KnowledgeEdge"]] = None,
    ) -> Optional["KnowledgeEdge"]:
        """Check the ancestry invariant for *node* and return a synthetic edge if needed.

        Args:
            node:        The node being added.
            graph:       The RuntimeGraph (must already contain the node so that
                         is_primordial() works with the updated temporal_sorted).
            batch_edges: Edges being added in the same call (prevents false
                         positives when a source and target arrive together).

        Returns:
            A synthetic KnowledgeEdge if one was synthesised, else None.
            The caller is responsible for adding this edge to the graph.
        """
        # Temporal zones are structural — exempt
        if node.is_temporal_zone:
            return None

        # Primordial roots are exempt (N oldest by timestamp)
        if self.is_primordial(node, graph):
            return None

        # Incoming edge already in graph?
        if self.has_incoming(node.id, graph):
            return None

        # Incoming edge in same batch?
        if batch_edges and self.has_incoming_in_batch(node.id, batch_edges):
            return None

        # No incoming edge found — find best predecessor and synthesise
        predecessor = self.find_best_predecessor(node, graph)
        if predecessor is None:
            # Empty graph or no earlier node — node becomes an additional root
            logger.info(
                f"[Ancestry] '{node.text[:50]}' has no eligible predecessors — "
                "admitted as additional primordial root"
            )
            return None

        # Compute similarity for the edge weight (if embeddings available)
        similarity = 0.0
        if node.embedding is not None and predecessor.embedding is not None:
            nn = float(np.linalg.norm(node.embedding))
            pn = float(np.linalg.norm(predecessor.embedding))
            if nn > 1e-8 and pn > 1e-8:
                similarity = float(np.dot(node.embedding, predecessor.embedding) / (nn * pn))

        edge = self.synthesise_edge(predecessor, node, similarity)
        logger.info(
            f"[Ancestry] Synthesised DERIVED_FROM: "
            f"'{predecessor.text[:40]}' → '{node.text[:40]}' "
            f"(sim={similarity:.3f})"
        )
        return edge


ANCESTRY_ENFORCER = AncestryEnforcer()


class RuntimeGraph:
    def __init__(self):
        self.nodes:           Dict[str, KnowledgeNode]            = {}
        self.edges:           Dict[Tuple[str,str], KnowledgeEdge] = {}
        self.zones:           Dict[str, TemporalZone]             = {}
        self.temporal_sorted: List[Tuple[float, str]]             = []
        self.domain_clusters: Dict[str, List[str]]                = {}

    # Incremental in-degree cache: set of node IDs that have ≥1 incoming
    # non-containment edge.  Updated by add_edge(); read by AncestryEnforcer.
    # Initialised lazily in _ensure_cache() because __init__ runs before
    # _incoming_ids is declared.
    def _ensure_cache(self):
        if not hasattr(self, "_incoming_ids"):
            self._incoming_ids: set = set()

    def _mark_has_incoming(self, node_id: str) -> None:
        self._ensure_cache()
        self._incoming_ids.add(node_id)

    def has_incoming_edge(self, node_id: str) -> bool:
        """True if node_id has at least one incoming non-containment edge (O(1) via cache)."""
        self._ensure_cache()
        return node_id in self._incoming_ids

    # ── Core mutation API ─────────────────────────────────────────────────────

    def add_node(self, node: KnowledgeNode):
        """Add a node with no ancestry check. Use add_node_checked() for enforcement."""
        self.nodes[node.id] = node
        # bisect.insort keeps temporal_sorted in O(log N) instead of O(N log N) sort()
        bisect.insort(self.temporal_sorted, (node.timestamp, node.id))
        self.domain_clusters.setdefault(node.domain, []).append(node.id)

    def add_node_checked(
        self,
        node: KnowledgeNode,
        batch_edges: Optional[List["KnowledgeEdge"]] = None,
    ) -> Optional["KnowledgeEdge"]:
        """Add a node and enforce the ancestry invariant.

        If the node has no incoming edge (neither in the graph nor in
        *batch_edges*) and is not a primordial root, AncestryEnforcer
        synthesises a DERIVED_FROM edge to the best-matching predecessor.

        The synthesised edge — if any — is added to the graph automatically.
        It is also returned so the caller can log or inspect it.

        Call this instead of add_node() whenever you're loading nodes that
        must obey the "discoveries don't arise from nothing" invariant.
        """
        self.add_node(node)  # register first so is_primordial() sees it in temporal_sorted
        anchor = ANCESTRY_ENFORCER.enforce(node, self, batch_edges)
        if anchor is not None:
            self.add_edge(anchor)   # add to graph and update cache
        return anchor

    def add_edge(self, edge: KnowledgeEdge):
        self.edges[(edge.source_id, edge.target_id)] = edge
        # Keep in-degree cache current so has_incoming_edge() is always O(1)
        if not edge.is_containment_edge:
            self._mark_has_incoming(edge.target_id)

    def add_zone(self, zone: TemporalZone):
        self.zones[zone.id] = zone

    def get_node(self, nid: str) -> Optional[KnowledgeNode]:
        return self.nodes.get(nid)

    def get_edge(self, src: str, tgt: str) -> Optional[KnowledgeEdge]:
        return self.edges.get((src, tgt))

    def get_all_zones(self) -> List[TemporalZone]:
        return list(self.zones.values())

    def zone_centroid(self, zone_id: str) -> Optional[np.ndarray]:
        zone = self.zones.get(zone_id)
        if not zone: return None
        embs = [self.nodes[nid].embedding
                for nid in zone.contained_node_ids
                if nid in self.nodes and self.nodes[nid].embedding is not None]
        return np.mean(embs, axis=0) if embs else None

    def k_hop_neighborhood(self, node_ids: List[str], k: int = 3) -> set:
        frontier = set(node_ids)
        visited  = set(node_ids)
        for _ in range(k):
            new_f = set()
            for (src, tgt) in self.edges:
                if src in frontier and tgt not in visited: new_f.add(tgt)
                elif tgt in frontier and src not in visited: new_f.add(src)
            frontier = new_f
            visited.update(frontier)
        return visited

    def get_neighbours(self, node_id: str, min_component: str = None,
                        min_value: float = 0.0, include_future: bool = False,
                        cutoff_ts: float = float("inf")
                        ) -> List[Tuple[str, KnowledgeEdge]]:
        result = []
        for (src, tgt), edge in self.edges.items():
            if src != node_id and tgt != node_id: continue
            nbr = tgt if src == node_id else src
            if nbr not in self.nodes: continue
            nbr_ts = self.nodes[nbr].timestamp
            if not include_future and nbr_ts > cutoff_ts: continue
            if min_component:
                if getattr(edge, min_component, 0.0) < min_value: continue
            result.append((nbr, edge))
        result.sort(key=lambda x: x[1].total_weight, reverse=True)
        return result

    def audit_ancestry(self) -> Dict[str, Any]:
        """Audit the ancestry invariant across the whole graph.

        Returns a summary dict with:
          - "primordial": list of (node_id, text) for the N exempt roots.
          - "orphans":    list of (node_id, text) for non-primordial nodes
                         that have no incoming edge (invariant violation).
          - "synthesised_edges": count of edges with
                         discovery_method == "ancestry_synthesised".
          - "ok":        True if orphans list is empty.

        Use this in tests, post-ingestion checks, and training data validation.
        Example::

            report = graph.audit_ancestry()
            assert report["ok"], f"Orphan nodes: {report['orphans']}"
        """
        n = CONFIG["primordial_cutoff_n"]
        primordial_ids = {nid for _, nid in self.temporal_sorted[:n]}

        orphans = []
        for nid, node in self.nodes.items():
            if node.is_temporal_zone:
                continue
            if nid in primordial_ids:
                continue
            if not self.has_incoming_edge(nid):
                orphans.append((nid, node.text[:60]))

        primordial = [
            (nid, self.nodes[nid].text[:60])
            for _, nid in self.temporal_sorted[:n]
            if nid in self.nodes
        ]

        synthesised = sum(
            1 for e in self.edges.values()
            if e.discovery_method == "ancestry_synthesised"
        )

        return {
            "primordial":        primordial,
            "orphans":           orphans,
            "synthesised_edges": synthesised,
            "ok":                len(orphans) == 0,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  GRAPH WALKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PathStep:
    node_id:           str
    node_text:         str
    node_domain:       str
    node_timestamp:    float
    zone_id:           str
    zone_multiplier:   float
    edge_to_next:      Optional[KnowledgeEdge]
    cumulative_weight: float

    @property
    def is_cross_domain(self) -> bool:
        if self.edge_to_next is None: return False
        return self.edge_to_next.relationship_type.startswith("cross_domain")


@dataclass
class Trajectory:
    query:             str
    steps:             List[PathStep]
    total_confidence:  float
    cross_domain_hops: int
    deepest_zone_mult: float
    provenance_urls:   List[str]

    def to_readable(self) -> str:
        lines = [f"Query: {self.query}",
                 f"Confidence: {self.total_confidence:.3f} | "
                 f"Cross-domain hops: {self.cross_domain_hops} | "
                 f"Max zone acceleration: ×{self.deepest_zone_mult:.1f}", ""]
        for i, step in enumerate(self.steps):
            zone_tag = (f" [inside zone ×{step.zone_multiplier:.1f}]"
                        if step.zone_id else "")
            lines.append(f"  Step {i+1}: {step.node_text}{zone_tag}")
            lines.append(f"           domain: {step.node_domain} "
                         f"| year: {_ts_to_year(step.node_timestamp)}")
            if step.edge_to_next:
                e = step.edge_to_next
                dominant = max(EDGE_WEIGHT_COMPONENTS,
                               key=lambda c: getattr(e, c))
                lines.append(f"           → [{e.relationship_type}]  "
                             f"(dominant signal: {dominant}={getattr(e, dominant):.2f})")
        lines.append("")
        lines.append("Sources: " + " | ".join(self.provenance_urls[:5]))
        return "\n".join(lines)


class GraphWalker:
    BEAM_WIDTH   = CONFIG["walker_beam_width"]
    MAX_PATH_LEN = CONFIG["walker_max_path_len"]
    MIN_CONF     = CONFIG["walker_min_conf"]
    LIMIT_BONUS  = CONFIG["walker_limit_bonus"]
    LIMIT_THRESH = CONFIG["walker_limit_thresh"]

    def __init__(self, runtime_graph: RuntimeGraph):
        self.g = runtime_graph

    def query(self, query_text: str, embedder,
              max_trajectories: int = 5,
              component_filter: Optional[str] = None,
              min_component_value: float = 0.0) -> List[Trajectory]:
        query_emb = embedder.encode(query_text)
        start_ids = self._find_start_nodes(query_emb, top_k=CONFIG["walker_top_k_start"])
        if not start_ids: return []

        beams: List[Tuple[List[str], float, float]] = [
            ([nid], 1.0, self.g.nodes[nid].timestamp)
            for nid in start_ids
        ]
        completed = []

        for step in range(self.MAX_PATH_LEN):
            next_beams = []
            for path, score, last_ts in beams:
                current_id = path[-1]
                nbrs = self.g.get_neighbours(
                    current_id, min_component=component_filter,
                    min_value=min_component_value)
                for nbr_id, edge in nbrs[:CONFIG["walker_nbr_scan_cap"]]:
                    if nbr_id in path: continue
                    nbr_ts = self.g.nodes[nbr_id].timestamp
                    if nbr_ts < last_ts: continue
                    new_score = score * max(edge.total_weight, 1e-6)
                    if edge.limitation_resolution > self.LIMIT_THRESH:
                        new_score *= self.LIMIT_BONUS
                    nbr_node = self.g.nodes.get(nbr_id)
                    if nbr_node and nbr_node.zone_id:
                        new_score *= nbr_node.acceleration_multiplier
                    if new_score < self.MIN_CONF: continue
                    next_beams.append((path + [nbr_id], new_score, nbr_ts))

            next_beams.sort(key=lambda x: x[1], reverse=True)
            beams = next_beams[:self.BEAM_WIDTH]
            if not beams: break
            for path, score, _ in beams:
                if len(path) >= CONFIG["walker_min_path_len"]:
                    completed.append((path, score))

        completed.sort(key=lambda x: x[1], reverse=True)
        trajectories = []
        seen_paths = set()
        for path, score in completed:
            key = tuple(path)
            if key in seen_paths: continue
            seen_paths.add(key)
            traj = self._build_trajectory(query_text, path, score)
            if traj: trajectories.append(traj)
            if len(trajectories) >= max_trajectories: break
        return trajectories

    def _find_start_nodes(self, query_emb: np.ndarray, top_k: int = 5) -> List[str]:
        scores = []
        for nid, node in self.g.nodes.items():
            if node.embedding is None or node.is_temporal_zone: continue
            sim = float(cos_sim([query_emb], [node.embedding])[0][0])
            scores.append((nid, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scores[:top_k]]

    def _build_trajectory(self, query: str, path: List[str],
                           score: float) -> Optional[Trajectory]:
        steps = []
        cross_hops   = 0
        deepest_zone = 1.0
        provenance   = []

        for i, nid in enumerate(path):
            node = self.g.nodes.get(nid)
            if not node: return None
            edge = (self.g.get_edge(path[i], path[i+1])
                    if i < len(path) - 1 else None)
            if edge and edge.is_containment_edge: continue
            zone_mult = node.acceleration_multiplier
            if zone_mult > deepest_zone: deepest_zone = zone_mult
            steps.append(PathStep(
                node_id=nid, node_text=node.text, node_domain=node.domain,
                node_timestamp=node.timestamp, zone_id=node.zone_id,
                zone_multiplier=zone_mult, edge_to_next=edge,
                cumulative_weight=score))
            if node.provenance: provenance.append(node.provenance)

        if len(steps) < 2: return None
        return Trajectory(
            query=query, steps=steps, total_confidence=score,
            cross_domain_hops=cross_hops, deepest_zone_mult=deepest_zone,
            provenance_urls=list(dict.fromkeys(provenance)))


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — INHIBITORY EDGES
# ══════════════════════════════════════════════════════════════════════════════

def extend_edge(edge: KnowledgeEdge, inhibitory_force: float = 0.0) -> KnowledgeEdge:
    edge.inhibitory_force = float(np.clip(inhibitory_force, 0.0, 1.0))
    base = getattr(edge, "total_weight", 0.0)
    edge.total_weight = base + INHIBITORY_WEIGHT * edge.inhibitory_force
    return edge


def compute_inhibitory_pressure(node_id: str, graph: RuntimeGraph) -> float:
    pressure = 0.0
    for (src, tgt), edge in graph.edges.items():
        if tgt != node_id: continue
        inh = getattr(edge, "inhibitory_force", 0.0)
        if inh <= 0: continue
        attacker = graph.get_node(src)
        attacker_weight = (getattr(attacker, "strategic_value", 5.0) / 10.0
                           if attacker else 0.5)
        pressure += inh * attacker_weight
    return min(1.0, pressure)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — CONVERGENCE PRESSURE FIELD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VoidPressureVector:
    node_id: str
    void_pressure: float = 0.0
    unresolved_limitation_pressure: float = 0.0
    cultural_phantom_pressure:      float = 0.0
    excess_investment_pressure:     float = 0.0
    inhibitory_sink_pressure:       float = 0.0
    temporal_zone_pressure:         float = 0.0
    pressure_history: List[float] = field(default_factory=list)


class ConvergencePressureField:
    SINGULARITY_THRESHOLD = CONFIG["pressure_singularity_threshold"]
    PRESSURE_DECAY_RATE   = CONFIG["pressure_decay_rate"]

    def __init__(self):
        self.field: Dict[str, VoidPressureVector] = {}

    def compute(self, graph: RuntimeGraph) -> Dict[str, VoidPressureVector]:
        for nid in graph.nodes:
            self.field[nid] = self._compute_node_pressure(nid, graph)
        return self.field

    def _compute_node_pressure(self, nid: str, graph: RuntimeGraph) -> VoidPressureVector:
        pv = VoidPressureVector(node_id=nid)
        node = graph.get_node(nid)
        if not node: return pv

        lim_thr  = CONFIG["pressure_limitation_threshold"]
        inv_thr  = CONFIG["pressure_investment_threshold"]
        inv_logsc = CONFIG["pressure_invest_log_scale"]
        lim_scale = CONFIG["pressure_limitation_scale"]
        unr_gate  = CONFIG["pressure_unresolved_gate"]
        zn_scale  = CONFIG["pressure_zone_multiplier_scale"]

        for (src, tgt), edge in graph.edges.items():
            if edge.is_containment_edge: continue
            is_incoming = (tgt == nid)
            if is_incoming and edge.limitation_resolution > lim_thr:
                pv.unresolved_limitation_pressure += (
                    edge.limitation_resolution * lim_scale * (1.0 - edge.total_weight))
            if is_incoming:
                inh = getattr(edge, "inhibitory_force", 0.0)
                if inh > 0:
                    attacker = graph.get_node(src)
                    weight = (getattr(attacker, "strategic_value", 5.0) / 10.0
                              if attacker else 0.5)
                    pv.inhibitory_sink_pressure += inh * weight
            if is_incoming and edge.investment_correlation > inv_thr:
                src_node = graph.get_node(src)
                if src_node:
                    invest = getattr(src_node, "investment_total_usd", 0.0)
                    outgoing_resolution = sum(
                        e.limitation_resolution
                        for (s2, t2), e in graph.edges.items()
                        if s2 == nid and not e.is_containment_edge)
                    if outgoing_resolution < unr_gate:
                        pv.excess_investment_pressure += (
                            edge.investment_correlation * math.log1p(invest) / inv_logsc)

        k2 = graph.k_hop_neighborhood([nid], k=2)
        for nbr_id in k2:
            nbr = graph.get_node(nbr_id)
            if not nbr: continue
            if "cultural_phantom" in getattr(nbr, "entity_type", ""):
                sg = getattr(nbr, "social_gravity", 0.0) or nbr.forecast_score
                pv.cultural_phantom_pressure += sg / 10.0

        if node.zone_id and node.acceleration_multiplier > 1.0:
            pv.temporal_zone_pressure = (node.acceleration_multiplier - 1.0) * zn_scale

        pw = CONFIG["pressure_void_weights"]
        pv.void_pressure = (
            pv.unresolved_limitation_pressure * pw["unresolved_limitation"] +
            pv.cultural_phantom_pressure      * pw["cultural_phantom"]      +
            pv.excess_investment_pressure     * pw["excess_investment"]     +
            pv.inhibitory_sink_pressure       * pw["inhibitory_sink"]       +
            pv.temporal_zone_pressure         * pw["temporal_zone"])
        pv.pressure_history.append(pv.void_pressure)
        return pv

    def pressure_gradient(self, node_id: str, neighbour_id: str) -> float:
        p_src = self.field.get(node_id,    VoidPressureVector(node_id)).void_pressure
        p_tgt = self.field.get(neighbour_id, VoidPressureVector(neighbour_id)).void_pressure
        return p_src - p_tgt


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — DORMANCY TRACKER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EdgeDormancyState:
    edge_key: Tuple[str, str]
    dormancy_score: float = 1.0
    epochs_dormant: int = 0
    last_contribution: float = 0.0
    awakening_event: Optional[str] = None
    neighbourhood_heat: float = 0.0


class DormancyTracker:
    AWAKENING_THRESHOLD = CONFIG["dormancy_awakening_threshold"]
    ACTIVE_THRESHOLD    = CONFIG["dormancy_active_threshold"]
    HEAT_DECAY_FACTOR   = CONFIG["dormancy_heat_decay_factor"]
    DORMANCY_GROWTH     = CONFIG["dormancy_growth_per_epoch"]

    def __init__(self):
        self.states: Dict[Tuple[str, str], EdgeDormancyState] = {}

    def initialise_edge(self, edge: KnowledgeEdge):
        key = (edge.source_id, edge.target_id)
        if key not in self.states:
            self.states[key] = EdgeDormancyState(edge_key=key)

    def update_epoch(self, graph: RuntimeGraph) -> List[Tuple[str, str]]:
        newly_awake = []
        for (src, tgt), state in self.states.items():
            heat = self._compute_neighbourhood_heat(src, tgt, graph)
            state.neighbourhood_heat = heat
            if heat > self.AWAKENING_THRESHOLD:
                decay = heat * self.HEAT_DECAY_FACTOR
                state.dormancy_score = max(0.0, state.dormancy_score * (1 - decay))
                if state.dormancy_score < self.ACTIVE_THRESHOLD:
                    state.awakening_event = f"neighbourhood_heat={heat:.2f}"
                    newly_awake.append((src, tgt))
            else:
                state.dormancy_score = min(1.0, state.dormancy_score + self.DORMANCY_GROWTH)
                state.epochs_dormant += 1
        return newly_awake

    def _compute_neighbourhood_heat(self, src: str, tgt: str,
                                     graph: RuntimeGraph) -> float:
        k2 = graph.k_hop_neighborhood([src, tgt], k=2)
        if not k2: return 0.0
        scores = []
        current_year = _ts_to_year(time.time())
        for nid in k2:
            node = graph.get_node(nid)
            if node and not node.is_temporal_zone:
                try:
                    year = _ts_to_year(node.timestamp)
                    recency = 1.0 / (1.0 + max(0, current_year - year) / 5.0)
                except Exception:
                    recency = 0.5
                scores.append(node.forecast_score * recency)
        return float(np.mean(scores)) if scores else 0.0

    def get_dormancy(self, src: str, tgt: str) -> float:
        return self.states.get((src, tgt), EdgeDormancyState((src, tgt))).dormancy_score

    def get_awakening_score(self, src: str, tgt: str) -> float:
        return 1.0 - self.get_dormancy(src, tgt)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — STRUCTURAL PRESSURE COMPUTER
#
#  Previously, void_pressure / SDI / cascade_influence / threshold checks were
#  duplicated in ContextualPriorityHead, PhantomNodeGenerator,
#  GraphEntropyMonitor, and RecursiveForecastingLoop.
#  All four now delegate to this single authority.
# ══════════════════════════════════════════════════════════════════════════════

class StructuralPressureComputer:
    """Single source of truth for topological pressure statistics.

    Eliminates duplication across ContextualPriorityHead, PhantomNodeGenerator,
    GraphEntropyMonitor, and RecursiveForecastingLoop.

    All threshold constants are read from CONFIG so that a single edit
    propagates to every consumer automatically.
    """

    SINGULARITY_THRESHOLD = CONFIG["pressure_singularity_threshold"]
    COLLAPSE_THRESHOLD    = CONFIG["collapse_threshold"]

    # ── Public API ─────────────────────────────────────────────────────────────

    def node_pressure(self, node_id: str,
                      pressure_field: "ConvergencePressureField") -> float:
        """Return the void_pressure scalar for a single node."""
        pv = pressure_field.field.get(node_id, VoidPressureVector(node_id))
        return pv.void_pressure

    def sdi_cascade_stats(self, node_ids: "List[str]",
                          graph: "RuntimeGraph") -> "Dict[str, float]":
        """Return mean SDI, mean cascade_influence, and mean zone_multiplier
        for a collection of node IDs (2-hop neighbourhood, domain set, etc.)."""
        sdis, cascades, zmults = [], [], []
        for nid in node_ids:
            node = graph.get_node(nid)
            if not node or node.is_temporal_zone:
                continue
            sdis.append(getattr(node, "structural_dependency_index", 0.0))
            cascades.append(getattr(node, "cascade_influence", 0.0))
            zmults.append(node.acceleration_multiplier)
        return {
            "mean_sdi":    float(np.mean(sdis))    if sdis    else 0.0,
            "mean_cascade": float(np.mean(cascades)) if cascades else 0.0,
            "mean_zmult":  float(np.mean(zmults))  if zmults  else 1.0,
        }

    def is_singularity(self, pressure: float) -> bool:
        return pressure >= self.SINGULARITY_THRESHOLD

    def is_collapse(self, pressure: float) -> bool:
        return pressure >= self.COLLAPSE_THRESHOLD

    def pressure_gradient(self, src_id: str, tgt_id: str,
                          pressure_field: "ConvergencePressureField") -> float:
        return pressure_field.pressure_gradient(src_id, tgt_id)

    def edge_topology_counts(self, k2: "Set[str]",
                              graph: "RuntimeGraph") -> "Dict[str, int]":
        """Count total / limitation / inhibitory edges touching a neighbourhood."""
        n_total = n_lim = n_inh = 0
        for (s2, t2), edge in graph.edges.items():
            if s2 in k2 or t2 in k2:
                n_total += 1
                if edge.limitation_resolution > CONFIG["priority_lim_threshold"]:
                    n_lim += 1
                if getattr(edge, "inhibitory_force", 0.0) > CONFIG["priority_inh_threshold"]:
                    n_inh += 1
        return {"n_edges": n_total, "n_lim": n_lim, "n_inh": n_inh}


# ── Singleton shared by all subsystems ────────────────────────────────────────
PRESSURE_COMPUTER = StructuralPressureComputer()


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — CONFIDENCE DECAY ENGINE
#
#  Previously confidence degradation lived in three separate places:
#    • RecursiveForecastingLoop   (DECAY_FACTOR geometric decay)
#    • PhantomNodeGenerator       (structural_gap_score multiplication)
#    • PathAgnosticInference      (entropy alerts)
#    • RecursiveForecastingLoop   (plateau penalty)
#  Parts of these are mathematically equivalent.  This class owns the logic.
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceDecayEngine:
    """Canonical source for all confidence-degradation arithmetic.

    Consumers receive a single ``decay_step`` call that applies:
      1. Geometric depth decay (DECAY_FACTOR per recursion level)
      2. Efficiency-plateau Christensen penalty
      3. Optional entropy-alert suppression

    This makes it impossible for modules to diverge silently when thresholds
    change.
    """

    DECAY_FACTOR       = CONFIG["forecast_decay_factor"]
    PLATEAU_SCALE      = CONFIG["plateau_penalty_scale"]
    PLATEAU_FLOOR      = CONFIG["plateau_penalty_floor"]

    def geometric_decay(self, confidence: float, steps: int = 1) -> float:
        """Standard exponential decay by depth."""
        return confidence * (self.DECAY_FACTOR ** steps)

    def plateau_penalty(self, node: "KnowledgeNode",
                        graph: "RuntimeGraph") -> float:
        """Return a multiplier in (PLATEAU_FLOOR, 1.0].

        1.0  = no penalty (node exceeds or equals domain plateau)
        < 1.0 = Christensen drag — incumbent efficiency plateau is higher.
        """
        max_plateau = max(
            (n.efficiency_plateau for n in graph.nodes.values()
             if not n.is_temporal_zone and n.domain == node.domain),
            default=0.0,
        )
        if max_plateau <= 0.0:
            return 1.0
        node_potential = float(node.convergence_potential)
        if node_potential >= max_plateau:
            return 1.0
        gap_ratio = (max_plateau - node_potential) / max(max_plateau, 1e-6)
        penalty   = 1.0 - self.PLATEAU_SCALE * float(np.clip(gap_ratio, 0.0, 1.0))
        return float(max(self.PLATEAU_FLOOR, penalty))

    def entropy_alert_factor(self, domain: str,
                             entropy_monitor: "GraphEntropyMonitor") -> float:
        """Suppress confidence when the domain's entropy is in 'pre_singularity'.

        Returns a multiplier [0.5, 1.0].  Normal = 1.0; pre-singularity = 0.5.
        """
        snap = entropy_monitor.current.get(domain)
        if snap and snap.alert == "pre_singularity":
            return 0.5
        return 1.0

    def decay_step(self, confidence: float,
                   node: "KnowledgeNode",
                   graph: "RuntimeGraph",
                   entropy_monitor: "Optional[GraphEntropyMonitor]" = None) -> float:
        """Apply one full decay step: geometric × plateau × entropy."""
        c = self.geometric_decay(confidence)
        c *= self.plateau_penalty(node, graph)
        if entropy_monitor is not None:
            c *= self.entropy_alert_factor(node.domain, entropy_monitor)
        return c


# ── Singleton ─────────────────────────────────────────────────────────────────
CONFIDENCE_ENGINE = ConfidenceDecayEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED INFRASTRUCTURE — GRAPH TRAVERSAL ENGINE
#
#  DFS path counting existed in RetrospectiveConvergenceLoss._count_paths_to_targets
#  and BFS reachability in PhantomNodeGenerator._is_near_phantom_zone.
#  Both are unified here so future changes (e.g., cycle handling or pruning)
#  propagate automatically.
# ══════════════════════════════════════════════════════════════════════════════

class GraphTraversalEngine:
    """Unified DFS / BFS traversal primitives for RuntimeGraph.

    Replaces:
      • RetrospectiveConvergenceLoss._count_paths_to_targets  (DFS)
      • PhantomNodeGenerator._is_near_phantom_zone            (BFS)
    """

    @staticmethod
    def dfs_count_paths(start: str, targets: "Set[str]",
                        graph: "RuntimeGraph",
                        max_hops: int, max_paths: int) -> int:
        """Count distinct simple paths from *start* to any node in *targets*,
        limited to *max_hops* and stopping after *max_paths* are found."""
        count = 0
        stack = [(start, {start}, 0)]
        while stack and count < max_paths:
            current, visited, hops = stack.pop()
            if hops > max_hops:
                continue
            for (src, tgt) in list(graph.edges.keys()):
                if src != current:
                    continue
                nbr = tgt
                if nbr in visited:
                    continue
                if nbr in targets:
                    count += 1
                else:
                    stack.append((nbr, visited | {nbr}, hops + 1))
        return count

    @staticmethod
    def bfs_reachable(start: str, target_ids: "Set[str]",
                      graph: "RuntimeGraph", max_hops: int) -> bool:
        """Return True if any node in *target_ids* is reachable from *start*
        within *max_hops* steps (undirected traversal)."""
        visited: set = {start}
        frontier = [start]
        for _hop in range(max_hops):
            next_frontier = []
            for nid in frontier:
                for (src, tgt) in list(graph.edges.keys()):
                    neighbour = (tgt if src == nid
                                 else (src if tgt == nid else None))
                    if neighbour is None or neighbour in visited:
                        continue
                    if neighbour in target_ids:
                        return True
                    visited.add(neighbour)
                    next_frontier.append(neighbour)
            frontier = next_frontier
            if not frontier:
                break
        return False


# ── Singleton ─────────────────────────────────────────────────────────────────
GRAPH_TRAVERSAL = GraphTraversalEngine()


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — CONTEXTUAL PRIORITY HEAD
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocalTopologyStats:
    mean_sdi_2hop:             float = 0.0
    mean_cascade_2hop:         float = 0.0
    void_pressure_src:         float = 0.0
    void_pressure_tgt:         float = 0.0
    pressure_gradient:         float = 0.0
    n_edges_2hop:              int   = 0
    n_limitation_edges_2hop:   int   = 0
    n_inhibitory_edges_2hop:   int   = 0
    mean_zone_multiplier_2hop: float = 1.0
    awakening_score:           float = 0.0
    mean_constraint_proximity_2hop: float = 0.5


class ContextualPriorityHead(nn.Module):
    SINGULARITY_THRESHOLD = CONFIG["priority_singularity_threshold"]
    EXPO_SCALE   = CONFIG["priority_expo_scale"]
    LINEAR_SCALE = CONFIG["priority_linear_scale"]

    def __init__(self, latent_dim: int, n_topology_features: int = 12):
        super().__init__()
        self.latent_dim = latent_dim
        if not TORCH_AVAILABLE: return
        self.pair_encoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 256), nn.GELU(),
            nn.LayerNorm(256), nn.Linear(256, 64))
        self.topo_encoder = nn.Sequential(
            nn.Linear(n_topology_features, 64), nn.GELU(), nn.Linear(64, 64))
        self.priority_head = nn.Sequential(
            nn.Linear(64 + 64, 64), nn.GELU(), nn.Linear(64, 1))
        if TORCH_AVAILABLE:
            self.log_expo_scale   = nn.Parameter(torch.tensor(math.log(self.EXPO_SCALE)))
            self.singularity_bias = nn.Parameter(torch.tensor(-self.SINGULARITY_THRESHOLD))

    def topology_to_vector(self, stats: LocalTopologyStats) -> np.ndarray:
        return np.array([
            stats.mean_sdi_2hop,
            stats.mean_cascade_2hop,
            stats.void_pressure_src / (self.SINGULARITY_THRESHOLD + 1e-8),
            stats.void_pressure_tgt / (self.SINGULARITY_THRESHOLD + 1e-8),
            max(-1, min(1, stats.pressure_gradient / self.SINGULARITY_THRESHOLD)),
            min(stats.n_edges_2hop, 50) / 50.0,
            min(stats.n_limitation_edges_2hop, 20) / 20.0,
            min(stats.n_inhibitory_edges_2hop, 10) / 10.0,
            stats.mean_zone_multiplier_2hop / 5.0,
            stats.awakening_score,
            stats.mean_constraint_proximity_2hop,
            1.0 if stats.void_pressure_src > self.SINGULARITY_THRESHOLD else 0.0,
        ], dtype=np.float32)

    def forward(self, h_src, h_tgt, topology_vec, time_delta):
        if not TORCH_AVAILABLE: return topology_vec
        pair_feat = self.pair_encoder(torch.cat([h_src, h_tgt], dim=-1))
        topo_feat = self.topo_encoder(topology_vec)
        combined  = torch.cat([pair_feat, topo_feat], dim=-1)
        raw_priority = self.priority_head(combined)
        expo_scale = torch.exp(self.log_expo_scale)
        above_threshold = torch.relu(raw_priority)
        below_threshold = -torch.relu(-raw_priority)
        priority = (torch.exp(above_threshold * expo_scale) +
                    below_threshold * self.LINEAR_SCALE)
        priority = torch.clamp(priority, CONFIG["priority_clamp_min"], CONFIG["priority_clamp_max"])
        return priority

    def compute_topology_stats(self, src_id, tgt_id, graph, pressure_field,
                                dormancy_tracker) -> LocalTopologyStats:
        # ── Delegate topology arithmetic to PRESSURE_COMPUTER ─────────────────
        k2    = graph.k_hop_neighborhood([src_id, tgt_id], k=2)
        sdc   = PRESSURE_COMPUTER.sdi_cascade_stats(list(k2), graph)
        ecnt  = PRESSURE_COMPUTER.edge_topology_counts(k2, graph)

        stats = LocalTopologyStats()
        stats.mean_sdi_2hop             = sdc["mean_sdi"]
        stats.mean_cascade_2hop         = sdc["mean_cascade"]
        stats.mean_zone_multiplier_2hop = sdc["mean_zmult"]
        stats.n_edges_2hop              = ecnt["n_edges"]
        stats.n_limitation_edges_2hop   = ecnt["n_lim"]
        stats.n_inhibitory_edges_2hop   = ecnt["n_inh"]
        # constraint_proximity is not yet stored on nodes — keep 0.5 default
        stats.mean_constraint_proximity_2hop = 0.5

        stats.void_pressure_src = PRESSURE_COMPUTER.node_pressure(src_id, pressure_field)
        stats.void_pressure_tgt = PRESSURE_COMPUTER.node_pressure(tgt_id, pressure_field)
        stats.pressure_gradient = PRESSURE_COMPUTER.pressure_gradient(src_id, tgt_id, pressure_field)
        stats.awakening_score   = dormancy_tracker.get_awakening_score(src_id, tgt_id)
        return stats

    @staticmethod
    def _topo_stats_to_array(stats: "LocalTopologyStats") -> np.ndarray:
        """Serialise a LocalTopologyStats into a fixed 12-dim numpy vector.
        Used by OracleDataset.from_graph() to build topology_vec without
        needing a live ContextualPriorityHead instance (Fix 3)."""
        SINGULARITY_THRESHOLD = CONFIG["priority_singularity_threshold"]
        EXPO_SCALE = CONFIG["priority_expo_scale"]
        return np.array([
            stats.mean_sdi_2hop,
            stats.mean_cascade_2hop,
            stats.void_pressure_src / (SINGULARITY_THRESHOLD + 1e-8),
            stats.void_pressure_tgt / (SINGULARITY_THRESHOLD + 1e-8),
            max(-1.0, min(1.0, stats.pressure_gradient / (SINGULARITY_THRESHOLD + 1e-8))),
            min(stats.n_edges_2hop, 50) / 50.0,
            min(stats.n_limitation_edges_2hop, 20) / 20.0,
            min(stats.n_inhibitory_edges_2hop, 10) / 10.0,
            stats.mean_zone_multiplier_2hop / 5.0,
            stats.awakening_score,
            stats.mean_constraint_proximity_2hop,
            1.0 if stats.void_pressure_src > SINGULARITY_THRESHOLD else 0.0,
        ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — PHANTOM NODE GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhantomNode:
    id: str
    gap_description: str
    predicted_entity_type: str
    predicted_domain: str
    predicted_year_range: Tuple[int, int]
    predicted_feature_vec: np.ndarray
    structural_gap_score: float
    source_node_ids: List[str]
    target_node_ids: List[str]
    void_pressure_at_gap: float = 0.0
    is_confirmed: bool = False
    confirmed_by_node_id: Optional[str] = None
    generation_timestamp: float = 0.0
    evidence_count: int = 0
    CONFIRMATION_SIM_THRESHOLD: float = 0.75


class PhantomNodeGenerator:
    EDGE_PROB_THRESHOLD = CONFIG["phantom_edge_prob_threshold"]
    PHANTOM_THRESHOLD   = CONFIG["phantom_gap_threshold"]
    CONFIRMATION_SIM    = CONFIG["phantom_confirmation_sim"]
    MAX_PHANTOMS        = CONFIG["phantom_max_count"]

    def __init__(self):
        self.phantoms: List[PhantomNode] = []
        if TORCH_AVAILABLE:
            latent_dim = CONFIG["model_latent_dim"]
            self.feature_decoder = nn.Sequential(
                nn.Linear(latent_dim, 256), nn.GELU(), nn.LayerNorm(256),
                nn.Linear(256, PhysicalSubstrateEncoder.EXTENDED_TOTAL_DIM),
                nn.Sigmoid())

    def scan_for_voids(self, node_latents, graph: RuntimeGraph,
                       pressure_field: ConvergencePressureField,
                       edge_predictor=None,
                       top_candidates: int = 100) -> List[PhantomNode]:
        if not TORCH_AVAILABLE or node_latents is None: return []
        raw_candidates = self._find_void_pairs(graph, pressure_field, top_candidates)
        # ── Sparsity regulator: drop weak-node pairs before expensive decoding ──
        candidate_pairs = SPARSITY_REGULATOR.filter_phantom_candidates(
            raw_candidates, graph, pressure_field)
        new_phantoms = []
        for src_id, tgt_id, gap_pressure in candidate_pairs:
            if gap_pressure < self.PHANTOM_THRESHOLD: continue
            src_idx = self._node_to_idx(src_id, graph)
            tgt_idx = self._node_to_idx(tgt_id, graph)
            if src_idx is None or tgt_idx is None: continue
            h_src = node_latents[src_idx]
            h_tgt = node_latents[tgt_idx]
            bridge = (h_src + h_tgt) / 2.0
            with _maybe_no_grad():
                predicted_features = self.feature_decoder(
                    bridge.unsqueeze(0)).squeeze(0).cpu().numpy()
            phantom = self._build_phantom(src_id, tgt_id, predicted_features,
                                          gap_pressure, graph, bridge)
            if phantom:
                new_phantoms.append(phantom)
                self.phantoms.append(phantom)
        self.phantoms.sort(key=lambda p: p.structural_gap_score, reverse=True)
        self.phantoms = self.phantoms[:self.MAX_PHANTOMS]
        return new_phantoms

    def _find_void_pairs(self, graph, pressure_field, top_n):
        existing_edges = set(graph.edges.keys())
        node_ids = [nid for nid, n in graph.nodes.items()
                    if not n.is_temporal_zone and n.embedding is not None]
        # ── Delegate pressure lookup to PRESSURE_COMPUTER ─────────────────────
        sorted_by_pressure = sorted(
            node_ids,
            key=lambda nid: PRESSURE_COMPUTER.node_pressure(nid, pressure_field),
            reverse=True)[:CONFIG["phantom_void_top_candidates"]]
        candidates = []
        for i, a in enumerate(sorted_by_pressure):
            for b in sorted_by_pressure[i+1:]:
                if (a, b) in existing_edges or (b, a) in existing_edges: continue
                a_nbrs = {tgt for (src, tgt) in existing_edges if src == a}
                b_nbrs = {tgt for (src, tgt) in existing_edges if src == b}
                if b in a_nbrs or a in b_nbrs: continue
                p_a = PRESSURE_COMPUTER.node_pressure(a, pressure_field)
                p_b = PRESSURE_COMPUTER.node_pressure(b, pressure_field)
                candidates.append((a, b, (p_a + p_b) / 2.0))
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_n]

    def _build_phantom(self, src_id, tgt_id, predicted_features, gap_pressure,
                        graph, bridge_vec=None) -> Optional[PhantomNode]:
        src = graph.get_node(src_id)
        tgt = graph.get_node(tgt_id)
        if not src or not tgt: return None
        src_year = _ts_to_year(src.timestamp)
        tgt_year = _ts_to_year(tgt.timestamp)
        predicted_year_min = max(src_year, tgt_year)
        predicted_year_max = predicted_year_min + int(
            CONFIG["phantom_year_range_base"] +
            gap_pressure * CONFIG["phantom_year_range_scale"])
        return PhantomNode(
            id="phantom_" + str(uuid.uuid4())[:8],
            gap_description=(f"Bridge between '{src.text[:40]}' and '{tgt.text[:40]}'"),
            predicted_entity_type="predicted_bridge_technology",
            predicted_domain=f"{src.domain} × {tgt.domain}",
            predicted_year_range=(predicted_year_min, predicted_year_max),
            predicted_feature_vec=predicted_features,
            structural_gap_score=min(1.0, gap_pressure / self.PHANTOM_THRESHOLD),
            source_node_ids=[src_id],
            target_node_ids=[tgt_id],
            void_pressure_at_gap=gap_pressure,
            generation_timestamp=max(src.timestamp, tgt.timestamp),
            evidence_count=2)

    def _node_to_idx(self, node_id, graph) -> Optional[int]:
        for i, nid in enumerate(graph.nodes):
            if nid == node_id: return i
        return None

    # Anomaly-based rupture confirmation thresholds
    RUPTURE_ANOMALY_THRESHOLD = CONFIG["phantom_rupture_anomaly_threshold"]
    RUPTURE_PROXIMITY_HOPS    = CONFIG["phantom_rupture_proximity_hops"]

    def check_confirmation(self, new_node: KnowledgeNode,
                            graph: RuntimeGraph) -> List[PhantomNode]:
        """Standard similarity-based confirmation (sim > CONFIRMATION_SIM)."""
        if new_node.embedding is None: return []
        confirmed = []
        for phantom in self.phantoms:
            if phantom.is_confirmed: continue
            predicted_text_emb = phantom.predicted_feature_vec[:384]
            sim = float(cos_sim([new_node.embedding], [predicted_text_emb])[0][0])
            if sim > self.CONFIRMATION_SIM:
                phantom.is_confirmed = True
                phantom.confirmed_by_node_id = new_node.id
                phantom.__dict__["confirmation_mode"] = "similarity"
                confirmed.append(phantom)
                logger.info(f"[PhantomGenerator] CONFIRMED (similarity): "
                            f"{phantom.gap_description[:60]} "
                            f"-> '{new_node.text[:40]}' (sim={sim:.3f})")
        return confirmed

    def check_confirmation_by_rupture(
        self,
        new_node: "KnowledgeNode",
        graph: "RuntimeGraph",
        anomaly_accumulator: "AnomalyAccumulator",
    ) -> List[PhantomNode]:
        """Confirmation by Rupture (anomaly-driven).

        A breakthrough often arrives as an anomaly -- a node that looks nothing
        like the prediction but fills the structural void regardless.
        Standard check_confirmation() misses this because it requires high
        semantic similarity.

        Conditions for confirmation:
          1. new_node.anomaly_density >= RUPTURE_ANOMALY_THRESHOLD, AND
          2. new_node is topologically near the phantom's source/target nodes
             (within RUPTURE_PROXIMITY_HOPS).

        Tagged with confirmation_mode='rupture' so callers can distinguish it.
        """
        node_anomaly = anomaly_accumulator.get_density(new_node.id)
        if node_anomaly < self.RUPTURE_ANOMALY_THRESHOLD:
            return []

        confirmed = []
        for phantom in self.phantoms:
            if phantom.is_confirmed: continue
            phantom_zone_ids = set(phantom.source_node_ids + phantom.target_node_ids)
            if not phantom_zone_ids: continue
            if self._is_near_phantom_zone(new_node.id, phantom_zone_ids, graph,
                                           self.RUPTURE_PROXIMITY_HOPS):
                phantom.is_confirmed = True
                phantom.confirmed_by_node_id = new_node.id
                phantom.__dict__["confirmation_mode"] = "rupture"
                confirmed.append(phantom)
                logger.warning(
                    f"[PhantomGenerator] CONFIRMED by RUPTURE: "
                    f"'{phantom.gap_description[:60]}' filled by anomalous node "
                    f"'{new_node.text[:40]}' (anomaly={node_anomaly:.3f})")
        return confirmed

    def _is_near_phantom_zone(
        self,
        start_id: str,
        target_ids: "Set[str]",
        graph: "RuntimeGraph",
        max_hops: int,
    ) -> bool:
        """Delegate to unified BFS traversal engine."""
        return GRAPH_TRAVERSAL.bfs_reachable(start_id, target_ids, graph, max_hops)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — RETROSPECTIVE CONVERGENCE LOSS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergencePoint:
    id: str
    description: str
    timestamp: float
    convergence_node_ids: List[str]
    prerequisite_paths: List[List[str]] = field(default_factory=list)
    paths_density: float = 0.0


class RetrospectiveConvergenceLoss(nn.Module):
    MAX_PATHS = 100.0

    def __init__(self):
        super().__init__()

    def forward(self, predicted_convergence_scores, historical_truths, paths_densities):
        if not TORCH_AVAILABLE: return 0.0
        convergence_gap = F.mse_loss(predicted_convergence_scores,
                                      historical_truths, reduction="none")
        return (convergence_gap * (1.0 + paths_densities)).mean()

    def compute_paths_density(self, convergence_point: ConvergencePoint,
                               graph: RuntimeGraph, cutoff_timestamp: float) -> float:
        target_ids = set(convergence_point.convergence_node_ids)
        n_paths = 0
        for nid in list(graph.nodes)[:200]:
            node = graph.get_node(nid)
            if not node or node.timestamp >= cutoff_timestamp: continue
            if nid in target_ids: continue
            n_paths += self._count_paths_to_targets(nid, target_ids, graph, 5, int(self.MAX_PATHS))
            if n_paths >= self.MAX_PATHS: break
        convergence_point.paths_density = n_paths
        return math.log1p(n_paths) / math.log1p(self.MAX_PATHS)

    def _count_paths_to_targets(self, start, targets, graph, max_hops, max_paths):
        """Delegate to unified DFS traversal engine."""
        return GRAPH_TRAVERSAL.dfs_count_paths(start, targets, graph, max_hops, max_paths)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — GRAPH ENTROPY MONITOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EntropySnapshot:
    region_id: str
    timestamp: float
    entropy: float
    void_pressure_mean: float
    n_nodes: int
    n_edges: int
    entropy_gradient: float = 0.0
    alert: Optional[str] = None


class GraphEntropyMonitor:
    H_MAX = math.log(N_EDGE_COMPONENTS)
    SINGULARITY_DROP_THRESHOLD = CONFIG["entropy_singularity_drop_threshold"]
    PRE_SINGULARITY_ENTROPY    = CONFIG["entropy_pre_singularity_entropy"]
    PRE_SINGULARITY_PRESSURE   = CONFIG["entropy_pre_singularity_pressure"]

    def __init__(self):
        self.snapshots: Dict[str, List[EntropySnapshot]] = defaultdict(list)
        self.current:   Dict[str, EntropySnapshot]       = {}

    def compute_snapshot(self, graph: RuntimeGraph, pressure_field: ConvergencePressureField,
                          timestamp: float) -> Dict[str, EntropySnapshot]:
        domain_edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        for (src, tgt), edge in graph.edges.items():
            if edge.is_containment_edge: continue
            src_node = graph.get_node(src)
            if src_node: domain_edges[src_node.domain].append(edge)

        new_snapshots = {}
        for domain, edges in domain_edges.items():
            entropy = self._compute_entropy(edges)
            domain_node_ids = graph.domain_clusters.get(domain, [])
            pressures = [pressure_field.field.get(nid, VoidPressureVector(nid)).void_pressure
                         for nid in domain_node_ids]
            mean_pressure = float(np.mean(pressures)) if pressures else 0.0
            snap = EntropySnapshot(region_id=domain, timestamp=timestamp,
                                   entropy=entropy, void_pressure_mean=mean_pressure,
                                   n_nodes=len(domain_node_ids), n_edges=len(edges))
            prev = self.current.get(domain)
            if prev: snap.entropy_gradient = entropy - prev.entropy
            snap.alert = self._classify_alert(snap, prev)
            self.snapshots[domain].append(snap)
            self.current[domain] = snap
            new_snapshots[domain] = snap
        return new_snapshots

    def _compute_entropy(self, edges: List[KnowledgeEdge]) -> float:
        if not edges: return 0.0
        component_means = np.zeros(N_EDGE_COMPONENTS)
        for edge in edges:
            for k, comp in enumerate(EDGE_WEIGHT_COMPONENTS):
                component_means[k] += getattr(edge, comp, 0.0)
        component_means /= len(edges)
        total = component_means.sum() + 1e-8
        probs = component_means / total
        entropy = -np.sum(probs * np.log(probs + 1e-8))
        return float(entropy / self.H_MAX)

    def _classify_alert(self, snap: EntropySnapshot,
                         prev: Optional[EntropySnapshot]) -> Optional[str]:
        if (prev and prev.entropy > CONFIG["entropy_high_prev_threshold"]
                and snap.entropy_gradient < -self.SINGULARITY_DROP_THRESHOLD):
            return "singularity_collapse"
        if (snap.entropy > self.PRE_SINGULARITY_ENTROPY
                and snap.void_pressure_mean > self.PRE_SINGULARITY_PRESSURE):
            return "pre_singularity"
        return None

    def get_alerts(self) -> List[Tuple[str, str, EntropySnapshot]]:
        alerts = [(d, snap.alert, snap) for d, snap in self.current.items() if snap.alert]
        alerts.sort(key=lambda x: x[2].void_pressure_mean, reverse=True)
        return alerts

    def predict_entropy_trend(self, window: int = 5) -> Dict[str, float]:
        """Predict per-domain entropy delta for the *next* epoch.

        Uses a simple linear extrapolation over the last *window* snapshots.
        A positive value means entropy is rising (system diversifying);
        negative means it is converging (pre-collapse).

        Returns a dict  {domain: predicted_delta}.  Consumed by
        RecursiveForecastingLoop to modulate branching confidence.
        """
        trends: Dict[str, float] = {}
        for domain, snaps in self.snapshots.items():
            recent = snaps[-window:]
            if len(recent) < 2:
                trends[domain] = 0.0
                continue
            # Least-squares slope of entropy over the last window
            xs = list(range(len(recent)))
            ys = [s.entropy for s in recent]
            n  = len(xs)
            sx, sy = sum(xs), sum(ys)
            sxy = sum(x * y for x, y in zip(xs, ys))
            sxx = sum(x * x for x in xs)
            denom = n * sxx - sx * sx
            slope = (n * sxy - sx * sy) / denom if denom != 0 else 0.0
            trends[domain] = float(slope)
        return trends


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: STRUCTURAL SPARSITY REGULATOR
#
#  Prevents exponential blow-up of PhantomNodeGenerator and
#  RecursiveForecastingLoop on large graphs.
#
#  Responsibilities:
#    • prune_weak_nodes  — remove structurally irrelevant nodes below threshold
#    • compress_dense_regions — collapse high-degree hubs into summaries
#    • get_branching_cap — per-node adaptive branching ceiling
# ══════════════════════════════════════════════════════════════════════════════

class StructuralSparsityRegulator:
    """Noise suppressor and branching limiter for large-graph scenarios.

    Without this regulator:
      • PhantomNodeGenerator._find_void_pairs is O(N²) on phantom candidates
      • RecursiveForecastingLoop branches exponentially with MAX_BRANCHES^MAX_DEPTH

    Strategy
    --------
    * Weak nodes — nodes whose void_pressure + forecast_score fall below
      ``weak_node_threshold`` — are marked ineligible for phantom scanning and
      recursive expansion (no hard deletion, to preserve graph integrity).
    * Dense regions — nodes with degree > ``dense_degree_threshold`` — have their
      effective neighbour list capped to the top-K by edge weight when used in
      traversal, preventing fan-out explosions.
    * Branching cap — derived from local pressure: high-pressure nodes get more
      branches; quiet nodes get fewer.
    """

    # Tune in CONFIG or override per-instance
    WEAK_PRESSURE_THRESHOLD = 0.10   # below this void_pressure → weak
    WEAK_FORECAST_THRESHOLD = 1.0    # below this forecast_score → weak
    DENSE_DEGREE_THRESHOLD  = 30     # edges above this → apply branch cap
    MAX_DENSE_NEIGHBOURS    = 10     # top-K neighbours kept for dense nodes
    BRANCH_FLOOR            = 1
    BRANCH_CEILING          = CONFIG["forecast_max_branches"]

    def is_weak_node(self, node_id: str,
                     graph: "RuntimeGraph",
                     pressure_field: "ConvergencePressureField") -> bool:
        """Return True if the node is structurally insignificant.

        Weak nodes are skipped by PhantomNodeGenerator and RecursiveForecastingLoop
        to avoid polluting the candidate pool.
        """
        node = graph.get_node(node_id)
        if not node or node.is_temporal_zone:
            return True
        pressure = PRESSURE_COMPUTER.node_pressure(node_id, pressure_field)
        if pressure >= self.WEAK_PRESSURE_THRESHOLD:
            return False
        if float(getattr(node, "forecast_score", 0.0)) >= self.WEAK_FORECAST_THRESHOLD:
            return False
        return True

    def get_capped_neighbours(self, node_id: str,
                               graph: "RuntimeGraph") -> List[Tuple[str, "KnowledgeEdge"]]:
        """Return top-K neighbours by edge weight for dense nodes, all for sparse ones."""
        nbrs: List[Tuple[str, "KnowledgeEdge"]] = graph.get_neighbours(node_id)
        if len(nbrs) <= self.DENSE_DEGREE_THRESHOLD:
            return nbrs
        # Dense region: keep only the top MAX_DENSE_NEIGHBOURS by total_weight
        nbrs_sorted = sorted(nbrs, key=lambda x: x[1].total_weight, reverse=True)
        return nbrs_sorted[:self.MAX_DENSE_NEIGHBOURS]

    def get_branching_cap(self, node_id: str,
                          pressure_field: "ConvergencePressureField") -> int:
        """Adaptive branching ceiling proportional to local void_pressure.

        Nodes near singularity get the full BRANCH_CEILING; quiet nodes get only
        BRANCH_FLOOR to prevent silent combinatorial explosion.
        """
        pressure = PRESSURE_COMPUTER.node_pressure(node_id, pressure_field)
        # Normalise to [0, 1] against singularity threshold
        ratio = min(1.0, pressure / (PRESSURE_COMPUTER.SINGULARITY_THRESHOLD + 1e-8))
        cap = self.BRANCH_FLOOR + round(ratio * (self.BRANCH_CEILING - self.BRANCH_FLOOR))
        return max(self.BRANCH_FLOOR, min(self.BRANCH_CEILING, cap))

    def filter_phantom_candidates(
        self,
        candidates: List[Tuple[str, str, float]],
        graph: "RuntimeGraph",
        pressure_field: "ConvergencePressureField",
    ) -> List[Tuple[str, str, float]]:
        """Strip weak-node pairs from the PhantomNodeGenerator candidate list."""
        return [
            (a, b, score) for (a, b, score) in candidates
            if not self.is_weak_node(a, graph, pressure_field)
            and not self.is_weak_node(b, graph, pressure_field)
        ]


# ── Singleton ─────────────────────────────────────────────────────────────────
SPARSITY_REGULATOR = StructuralSparsityRegulator()


# ══════════════════════════════════════════════════════════════════════════════
#  NEW: SCENARIO COHERENCE SCORER
#
#  ForecastScenario was: nodes + edges + probability.
#  Without a structural coherence score, alternative scenarios cannot be ranked
#  meaningfully.  This class computes it and stores it back onto the scenario.
# ══════════════════════════════════════════════════════════════════════════════

class ScenarioCoherenceScorer:
    """Compute a Scenario Structural Coherence Score (SSCS) for ForecastScenario.

    SSCS ∈ [0, 1] is a weighted combination of:
      1. Path continuity  — are consecutive hallucinated nodes connected or
                            near-connected in the base graph?
      2. Temporal monotonicity — do predicted timestamps increase with depth?
      3. Domain coherence — do the hallucinated nodes cluster in few domains?
      4. Confidence gradient — does confidence decay smoothly (no sudden jumps)?

    Higher SSCS → more internally consistent scenario → preferred for ranking.
    """

    W_PATH        = 0.35
    W_TEMPORAL    = 0.30
    W_DOMAIN      = 0.20
    W_CONFIDENCE  = 0.15

    def score(self, scenario: "ForecastScenario",
              base_graph: "RuntimeGraph") -> float:
        """Compute and attach SSCS to *scenario.coherence_score*. Returns score."""
        nodes = [h.node for h in scenario.hallucinated_nodes]
        if not nodes:
            scenario.coherence_score = 0.0
            return 0.0

        s_path       = self._path_continuity(scenario, base_graph)
        s_temporal   = self._temporal_monotonicity(scenario)
        s_domain     = self._domain_coherence(scenario)
        s_confidence = self._confidence_smoothness(scenario)

        sscs = (self.W_PATH       * s_path
              + self.W_TEMPORAL   * s_temporal
              + self.W_DOMAIN     * s_domain
              + self.W_CONFIDENCE * s_confidence)
        scenario.coherence_score = float(np.clip(sscs, 0.0, 1.0))
        return scenario.coherence_score

    # ── Sub-scorers ────────────────────────────────────────────────────────────

    def _path_continuity(self, scenario: "ForecastScenario",
                         base_graph: "RuntimeGraph") -> float:
        """Fraction of consecutive (depth-ordered) node pairs that are connected
        or within 2 hops in the base graph."""
        nodes = sorted(scenario.hallucinated_nodes, key=lambda h: h.generation_depth)
        if len(nodes) < 2:
            return 1.0
        connected = 0
        for i in range(len(nodes) - 1):
            a_parents = set(nodes[i].parent_node_ids)
            b_parents = set(nodes[i + 1].parent_node_ids)
            # Overlap in parent IDs → they share a common ancestor in the base graph
            if a_parents & b_parents:
                connected += 1
            elif GRAPH_TRAVERSAL.bfs_reachable(
                nodes[i].node.id, b_parents, base_graph, max_hops=2
            ):
                connected += 1
        return connected / (len(nodes) - 1)

    def _temporal_monotonicity(self, scenario: "ForecastScenario") -> float:
        """Fraction of consecutive timestamp pairs that are non-decreasing."""
        nodes = sorted(scenario.hallucinated_nodes, key=lambda h: h.generation_depth)
        if len(nodes) < 2:
            return 1.0
        mono = sum(
            1 for i in range(len(nodes) - 1)
            if nodes[i + 1].predicted_timestamp >= nodes[i].predicted_timestamp
        )
        return mono / (len(nodes) - 1)

    def _domain_coherence(self, scenario: "ForecastScenario") -> float:
        """Score based on domain diversity: 1 domain = 1.0, many = lower."""
        nodes = scenario.hallucinated_nodes
        if not nodes:
            return 1.0
        domains = set(h.node.domain for h in nodes)
        # Normalise: 1 domain → 1.0; N domains → 1/N
        return 1.0 / len(domains)

    def _confidence_smoothness(self, scenario: "ForecastScenario") -> float:
        """Penalise abrupt drops in confidence_at_depth."""
        c = scenario.confidence_at_depth
        if len(c) < 2:
            return 1.0
        deltas = [abs(c[i] - c[i - 1]) / max(c[i - 1], 1e-8)
                  for i in range(1, len(c))]
        mean_drop = float(np.mean(deltas))
        # mean_drop = 0 → perfectly smooth = 1.0; DECAY_FACTOR ≈ 0.35 drop → ~0.5
        return float(np.exp(-mean_drop))


# ── Singleton ─────────────────────────────────────────────────────────────────
COHERENCE_SCORER = ScenarioCoherenceScorer()


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — RECURSIVE FORECASTING LOOP
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HallucinatedNode:
    node: KnowledgeNode
    generation_depth: int
    confidence: float
    parent_node_ids: List[str]
    predicted_timestamp: float
    is_secondary_effect: bool


@dataclass
class ForecastScenario:
    scenario_id: str
    description: str
    depth: int
    hallucinated_nodes: List[HallucinatedNode]
    predicted_edges: List[KnowledgeEdge]
    confidence_at_depth: List[float]
    secondary_effects: List[str]
    collapse_probability: float = 0.0
    coherence_score: float = 0.0      # ← NEW: computed by ScenarioCoherenceScorer


class RecursiveForecastingLoop:
    MIN_CONFIDENCE  = CONFIG["forecast_min_confidence"]
    DECAY_FACTOR    = CONFIG["forecast_decay_factor"]
    MAX_DEPTH       = CONFIG["forecast_max_depth"]
    MAX_BRANCHES    = CONFIG["forecast_max_branches"]
    YEARS_PER_DEPTH = CONFIG["forecast_years_per_depth"]

    def __init__(self, graph_walker: GraphWalker, edge_predictor,
                 pressure_field: ConvergencePressureField,
                 phantom_gen: PhantomNodeGenerator,
                 entropy_monitor: Optional["GraphEntropyMonitor"] = None):
        self.walker         = graph_walker
        self.edge_pred      = edge_predictor
        self.pressure       = pressure_field
        self.phantom_gen    = phantom_gen
        # GraphEntropyMonitor is optional for backward compatibility.
        # When provided, predict_entropy_trend() modulates branching confidence.
        self.entropy_monitor = entropy_monitor

    def forecast(self, seed_node_ids: List[str], base_graph: RuntimeGraph,
                  n_scenarios: int = 3) -> List[ForecastScenario]:
        # ── Pull entropy trend once per forecast call ──────────────────────────
        entropy_trend: Dict[str, float] = {}
        if self.entropy_monitor is not None:
            entropy_trend = self.entropy_monitor.predict_entropy_trend()

        scenarios = []
        for seed_id in seed_node_ids[:n_scenarios]:
            scenario = self._run_scenario(seed_id, base_graph, len(scenarios),
                                          entropy_trend)
            if scenario:
                # Score coherence before returning
                COHERENCE_SCORER.score(scenario, base_graph)
                scenarios.append(scenario)

        # Sort primarily by coherence_score, break ties with confidence depth
        scenarios.sort(
            key=lambda s: (s.coherence_score,
                           s.confidence_at_depth[-1] if s.confidence_at_depth else 0),
            reverse=True,
        )
        return scenarios

    def _run_scenario(self, seed_id: str, base_graph: RuntimeGraph,
                       scenario_idx: int,
                       entropy_trend: Optional[Dict[str, float]] = None
                       ) -> Optional[ForecastScenario]:
        seed = base_graph.get_node(seed_id)
        if not seed: return None
        hallucinated_nodes: List[HallucinatedNode] = []
        predicted_edges:    List[KnowledgeEdge]    = []
        confidence_per_depth: List[float]           = [1.0]
        current_confidence = 1.0
        secondary_effects:  List[str]               = []
        hallucinated_ids: Set[str] = set()
        current_frontier = {seed_id}
        entropy_trend = entropy_trend or {}

        for depth in range(1, self.MAX_DEPTH + 1):
            # ── Decay via ConfidenceDecayEngine (geometric step) ───────────────
            current_confidence = CONFIDENCE_ENGINE.geometric_decay(current_confidence)

            # ── Entropy trend modulation ───────────────────────────────────────
            # Negative trend (converging) → boost confidence (system focusing)
            # Positive trend (diverging)  → suppress confidence (system scattering)
            domain_trend = entropy_trend.get(seed.domain, 0.0)
            entropy_mod  = float(np.exp(-max(0.0, domain_trend) * 2.0))
            current_confidence *= entropy_mod

            if current_confidence < self.MIN_CONFIDENCE:
                break

            new_frontier: Set[str] = set()
            for node_id in list(current_frontier)[:self.MAX_BRANCHES]:
                # ── Adaptive branching via SparsityRegulator ──────────────────
                branch_cap = SPARSITY_REGULATOR.get_branching_cap(node_id, self.pressure)
                next_nodes = self._predict_next_nodes(
                    node_id, base_graph, hallucinated_ids, depth, current_confidence)
                for h_node in next_nodes[:branch_cap]:
                    # ── plateau + entropy penalty on each hallucinated node ───
                    node_conf = CONFIDENCE_ENGINE.plateau_penalty(h_node.node, base_graph)
                    if self.entropy_monitor is not None:
                        node_conf *= CONFIDENCE_ENGINE.entropy_alert_factor(
                            h_node.node.domain, self.entropy_monitor)
                    h_node.confidence *= node_conf

                    hallucinated_nodes.append(h_node)
                    hallucinated_ids.add(h_node.node.id)
                    new_frontier.add(h_node.node.id)
                    if depth >= 2:
                        secondary_effects.append(
                            f"Depth {depth}: '{h_node.node.text[:50]}' (conf={h_node.confidence:.2f})")
            if not new_frontier:
                break
            confidence_per_depth.append(current_confidence)
            current_frontier = new_frontier

        if not hallucinated_nodes: return None
        return ForecastScenario(
            scenario_id=f"scenario_{scenario_idx}_{seed_id[:8]}",
            description=f"Forward projection from '{seed.text[:50]}'",
            depth=len(confidence_per_depth) - 1,
            hallucinated_nodes=hallucinated_nodes,
            predicted_edges=predicted_edges,
            confidence_at_depth=confidence_per_depth,
            secondary_effects=secondary_effects,
            collapse_probability=max(0.0, 1.0 - current_confidence / self.MIN_CONFIDENCE))

    # ── Incumbent Reaction constants ─────────────────────────────────────────
    # Threshold forecast_score above which a hallucinated node triggers
    # defensive inhibitory phantom edges from domain incumbents.
    INCUMBENT_THREAT_THRESHOLD = CONFIG["incumbent_threat_threshold"]
    PLATEAU_PENALTY_SCALE      = CONFIG["plateau_penalty_scale"]

    def _apply_incumbent_reaction(
        self,
        hallucinated_node: "KnowledgeNode",
        base_graph: "RuntimeGraph",
    ) -> List["KnowledgeEdge"]:
        """Domain Self-Preservation Instinct.

        For every incumbent node in the same domain whose strategic_value > 8,
        generate a phantom inhibitory KnowledgeEdge proportional to both the
        breakthrough's forecast_score and the incumbent's strategic stake.

        These edges are NOT persisted to the graph; they are attached to
        HallucinatedNode.incumbent_inhibitory_edges so the Lag Model can
        delay predicted_timestamp accordingly.
        """
        phantom_edges: List[KnowledgeEdge] = []
        h_forecast = float(np.clip(hallucinated_node.forecast_score, 0.0, 10.0))
        if h_forecast / 10.0 < self.INCUMBENT_THREAT_THRESHOLD:
            return phantom_edges

        competitors = [
            n for n in base_graph.nodes.values()
            if (not n.is_temporal_zone
                and n.strategic_value > CONFIG["incumbent_sv_threshold"]
                and n.domain == hallucinated_node.domain)
        ]
        for comp in competitors:
            force = (h_forecast / 10.0) * (comp.strategic_value / 10.0)
            inh_edge = KnowledgeEdge(
                id=f"incumbent_inh_{comp.id[:6]}_{hallucinated_node.id[:6]}",
                source_id=comp.id,
                target_id=hallucinated_node.id,
                relationship_type="incumbent_inhibition",
                inhibitory_force=float(np.clip(force, 0.0, 1.0)),
                timestamp=hallucinated_node.timestamp,
                confidence=force,
            )
            inh_edge.compute_total_weight()
            phantom_edges.append(inh_edge)
            logger.info(
                f"[IncumbentReaction] '{comp.text[:40]}' (sv={comp.strategic_value:.1f}) "
                f"-> inhibitory against '{hallucinated_node.text[:40]}' force={force:.3f}"
            )
        return phantom_edges

    def _efficiency_plateau_penalty(
        self,
        hallucinated_node: "KnowledgeNode",
        base_graph: "RuntimeGraph",
    ) -> float:
        """Delegate to ConfidenceDecayEngine (canonical plateau logic)."""
        return CONFIDENCE_ENGINE.plateau_penalty(hallucinated_node, base_graph)

    def _predict_next_nodes(self, node_id, base_graph, already_hallucinated,
                             depth, confidence) -> List[HallucinatedNode]:
        seed = base_graph.get_node(node_id)
        if not seed: return []
        nbrs = base_graph.get_neighbours(node_id)
        phantom_candidates = [p for p in self.phantom_gen.phantoms
                              if node_id in p.source_node_ids and not p.is_confirmed]
        results = []
        for phantom in phantom_candidates[:self.MAX_BRANCHES]:
            h_node = KnowledgeNode(
                id=f"halluc_{phantom.id}_{depth}",
                text=phantom.gap_description[:80],
                node_type="hallucinated",
                domain=phantom.predicted_domain,
                entity_type=phantom.predicted_entity_type,
                timestamp=(seed.timestamp + depth * self.YEARS_PER_DEPTH * 365.25 * 24 * 3600))

            # Incumbent reaction: slow down breakthrough via phantom inhibitory edges
            incumbent_edges = self._apply_incumbent_reaction(h_node, base_graph)
            lag_years = sum(e.inhibitory_force for e in incumbent_edges) * CONFIG["incumbent_lag_years_per_force"]
            h_node.timestamp += lag_years * 365.25 * 24 * 3600

            # Efficiency plateau penalty
            plateau_factor = self._efficiency_plateau_penalty(h_node, base_graph)
            raw_conf = confidence * phantom.structural_gap_score * plateau_factor

            h_item = HallucinatedNode(
                node=h_node, generation_depth=depth,
                confidence=raw_conf,
                parent_node_ids=[node_id],
                predicted_timestamp=h_node.timestamp,
                is_secondary_effect=(depth >= 2))
            if incumbent_edges:
                h_item.__dict__["incumbent_inhibitory_edges"] = incumbent_edges
            results.append(h_item)

        for nbr_id, edge in nbrs[:self.MAX_BRANCHES]:
            if nbr_id in already_hallucinated: continue
            nbr = base_graph.get_node(nbr_id)
            if not nbr: continue
            proj_node = KnowledgeNode(
                id=f"halluc_proj_{nbr_id}_{depth}",
                text=f"Projected: {nbr.text[:60]}",
                node_type="hallucinated_projection",
                domain=nbr.domain, entity_type=nbr.entity_type,
                timestamp=(nbr.timestamp + depth * self.YEARS_PER_DEPTH * 365.25 * 24 * 3600),
                forecast_score=nbr.forecast_score * (self.DECAY_FACTOR ** depth))

            incumbent_edges = self._apply_incumbent_reaction(proj_node, base_graph)
            lag_years = sum(e.inhibitory_force for e in incumbent_edges) * CONFIG["incumbent_lag_years_per_force"]
            proj_node.timestamp += lag_years * 365.25 * 24 * 3600

            plateau_factor = self._efficiency_plateau_penalty(proj_node, base_graph)
            raw_conf = confidence * edge.total_weight * plateau_factor

            p_item = HallucinatedNode(
                node=proj_node, generation_depth=depth,
                confidence=raw_conf,
                parent_node_ids=[node_id, nbr_id],
                predicted_timestamp=proj_node.timestamp,
                is_secondary_effect=(depth >= 2))
            if incumbent_edges:
                p_item.__dict__["incumbent_inhibitory_edges"] = incumbent_edges
            results.append(p_item)

        return results


# ══════════════════════════════════════════════════════════════════════════════
#  EXTENSIONS — PATH-AGNOSTIC INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConvergenceCloud:
    query: str
    convergence_domain: str
    predicted_year_range: Tuple[int, int]
    n_independent_paths: int
    redundancy_score: float
    total_attention_flow: float
    convergence_probability: float
    top_trajectories: List[Trajectory]
    void_pressure_at_target: float
    active_phantoms: List[PhantomNode]
    entropy_alerts: List[Tuple[str, str]]

    def to_readable(self) -> str:
        lines = [
            f"QUERY: {self.query}", "",
            f"CONVERGENCE DETECTED: {self.convergence_domain}",
            f"Predicted window:     {self.predicted_year_range[0]}–{self.predicted_year_range[1]}", "",
            f"INEVITABILITY ASSESSMENT",
            f"  Independent paths:    {self.n_independent_paths}",
            f"  Redundancy score:     {self.redundancy_score:.2f}",
            f"  Attention flow:       {self.total_attention_flow:.3f}",
            f"  Convergence prob:     {self.convergence_probability:.1%}",
            f"  Void pressure:        {self.void_pressure_at_target:.2f}", "",
        ]
        if self.active_phantoms:
            lines.append(f"PREDICTED BRIDGE TECHNOLOGIES ({len(self.active_phantoms)}):")
            for p in self.active_phantoms[:3]:
                lines.append(f"  [{p.structural_gap_score:.2f}] {p.gap_description[:60]}")
        if self.entropy_alerts:
            lines.append(f"\nENTROPY ALERTS:")
            for domain, alert_type in self.entropy_alerts[:3]:
                lines.append(f"  {alert_type.upper()}: {domain}")
        if self.top_trajectories:
            lines.append(f"\nBEST PATHS (top {len(self.top_trajectories)}):")
            for i, traj in enumerate(self.top_trajectories):
                lines.append(f"  Path {i+1} [{traj.total_confidence:.3f}]: "
                             + " → ".join(s.node_text[:25] for s in traj.steps))
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  HIGH-CONVERGENCE ALERT  –  rich structured report for nodes that have
#  crossed the emergence threshold.
#
#  Usage:
#      detector = HighConvergenceDetector(threshold=72.5)
#      alerts   = detector.scan(graph)
#      for alert in alerts:
#          print(alert.to_readable())
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComponentMaturity:
    """Maturity level of a single technical component (0–100 %)."""
    name:    str
    percent: float   # 0–100
    # ✓ if percent >= 80, ⚠ if 60–79, ✗ below 60
    @property
    def status_icon(self) -> str:
        if self.percent >= 80:
            return "✓"
        if self.percent >= 60:
            return "⚠"
        return "✗"

    def format_line(self, width: int = 40) -> str:
        label = f"├─ {self.name}:"
        return f"{label:<{width}} {self.percent:.0f}%  {self.status_icon}"


@dataclass
class BottleneckItem:
    """A single unresolved barrier on the path to emergence."""
    description:    str
    solved_pct:     float   # fraction already resolved, 0–100
    months_to_resolve: str  # e.g. "12-18 months"

    def format_line(self) -> str:
        return (f"├─ {self.description} "
                f"({self.solved_pct:.0f}% solved, {self.months_to_resolve} to resolve)")


@dataclass
class HistoricalAnalogue:
    """A past breakthrough that followed a similar convergence pattern."""
    name:    str    # e.g. "Laser (1958→1960)"
    comment: str    # e.g. "Similar cross-domain convergence pattern"

    def format_line(self) -> str:
        return f"- {self.name}: {self.comment}"


@dataclass
class LikelyBuilder:
    """An organisation most likely to realise the predicted technology."""
    rank:        int
    name:        str
    probability: float   # 0–100 %
    rationale:   str

    def format_line(self) -> str:
        return f"{self.rank}. {self.name} ({self.probability:.0f}%) - {self.rationale}"


@dataclass
class PredictedForm:
    """One expected property of the emerging technology."""
    trait: str   # e.g. "Cloud-based platform (not standalone software)"

    def format_line(self) -> str:
        return f"├─ {self.trait}"


@dataclass
class MonitoringSignal:
    """A leading indicator to watch before full emergence."""
    signal: str   # e.g. "Joint publications (GNN + LLM + Forecasting authors)"

    def format_line(self) -> str:
        return f"├─ {self.signal}"


@dataclass
class HighConvergenceAlert:
    """
    Full-detail alert emitted when a knowledge-graph node crosses the
    convergence threshold.  Covers maturity, bottlenecks, historical
    analogues, likely builders, predicted form, and monitoring signals.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    node_id:     str
    label:       str
    description: str

    # ── Scores ────────────────────────────────────────────────────────────────
    convergence_score: float   # 0–100
    threshold:         float   # 0–100

    # ── Component breakdown ───────────────────────────────────────────────────
    components:  List[ComponentMaturity]

    # ── Barriers ──────────────────────────────────────────────────────────────
    bottlenecks: List[BottleneckItem]

    # ── Historical context ────────────────────────────────────────────────────
    analogues:   List[HistoricalAnalogue]

    # ── Temporal prediction ───────────────────────────────────────────────────
    emergence_window:        str    # e.g. "Q2-Q3 2026"
    confidence_interval:     str    # e.g. "[Q4 2025, Q4 2027]"
    probability:             float  # 0–100 %
    probability_margin:      float  # ± value, e.g. 2.8

    # ── Actor analysis ────────────────────────────────────────────────────────
    likely_builders: List[LikelyBuilder]

    # ── Technology profile ────────────────────────────────────────────────────
    predicted_forms:    List[PredictedForm]

    # ── Surveillance ──────────────────────────────────────────────────────────
    monitoring_signals: List[MonitoringSignal]

    # ── Optional reflexive note ───────────────────────────────────────────────
    self_awareness_note: str = ""

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def above_threshold_pct(self) -> float:
        """How many percent above the threshold the score sits."""
        if self.threshold == 0:
            return 0.0
        return (self.convergence_score - self.threshold) / self.threshold * 100.0

    @property
    def status_label(self) -> str:
        """Human-readable status based on how far above threshold."""
        pct = self.above_threshold_pct
        if pct >= 20:
            return f"CRITICAL MASS ACHIEVED ({pct:.0f}% above threshold)"
        if pct >= 10:
            return f"THRESHOLD EXCEEDED ({pct:.0f}% above threshold)"
        if pct >= 0:
            return f"THRESHOLD MET ({pct:.0f}% above threshold)"
        return "BELOW THRESHOLD"

    def to_readable(self) -> str:
        """
        Render the alert as the canonical multi-section text report.
        Designed to be printed directly or forwarded to a logger.
        """
        W = 70   # total width for separator lines
        SEP = "─" * W

        lines: List[str] = []

        # ── Header ────────────────────────────────────────────────────────────
        lines += [
            "█" * W,
            f"  ALERT: High-Convergence Cluster Detected",
            "█" * W,
            "",
            f"Node ID:     {self.node_id}",
            f"Label:       \"{self.label}\"",
            "",
            "Description:",
        ]
        # Wrap description at ~65 chars with consistent indentation
        for chunk in self._wrap(self.description, 65):
            lines.append(f"  {chunk}")

        # ── Score block ───────────────────────────────────────────────────────
        lines += [
            "",
            SEP,
            f"Convergence Score:  {self.convergence_score:.1f} / 100",
            f"Threshold:          {self.threshold:.1f}",
            f"Status:             {self.status_label}",
        ]

        # ── Component maturity ────────────────────────────────────────────────
        if self.components:
            lines += ["", SEP, "Component Maturity:"]
            for c in self.components:
                lines.append(c.format_line())

        # ── Bottlenecks ───────────────────────────────────────────────────────
        if self.bottlenecks:
            lines += ["", SEP, "Primary Bottlenecks:"]
            for b in self.bottlenecks:
                lines.append(b.format_line())

        # ── Historical analogues ──────────────────────────────────────────────
        if self.analogues:
            lines += ["", SEP, "Historical Analogues:"]
            for a in self.analogues:
                lines.append(a.format_line())

        # ── Temporal forecast ─────────────────────────────────────────────────
        lines += [
            "",
            SEP,
            f"Emergence Window:     {self.emergence_window}",
            f"Confidence Interval:  {self.confidence_interval}",
            f"Probability:          {self.probability:.1f}% (±{self.probability_margin:.1f}%)",
        ]

        # ── Likely builders ───────────────────────────────────────────────────
        if self.likely_builders:
            lines += ["", SEP, "Most Likely Builders:"]
            for b in self.likely_builders:
                lines.append(b.format_line())

        # ── Predicted form ────────────────────────────────────────────────────
        if self.predicted_forms:
            lines += ["", SEP, "Predicted Form:"]
            for f_ in self.predicted_forms:
                lines.append(f_.format_line())

        # ── Monitoring signals ────────────────────────────────────────────────
        if self.monitoring_signals:
            lines += ["", SEP, "Recommended Monitoring Signals (check monthly):"]
            for s in self.monitoring_signals:
                lines.append(s.format_line())

        # ── Self-awareness note ───────────────────────────────────────────────
        if self.self_awareness_note:
            lines += ["", SEP, "System Self-Awareness Note:"]
            for chunk in self._wrap(self.self_awareness_note, 65):
                lines.append(f"  {chunk}")

        lines += ["", "█" * W, ""]
        return "\n".join(lines)

    # ── Internal helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """
        Simple word-wrap: splits on spaces, respects existing newlines,
        produces lines no longer than *width* characters.
        """
        result: List[str] = []
        for paragraph in text.splitlines():
            words  = paragraph.split()
            line   = ""
            for word in words:
                candidate = f"{line} {word}".strip()
                if len(candidate) <= width:
                    line = candidate
                else:
                    if line:
                        result.append(line)
                    line = word
            if line:
                result.append(line)
        return result or [""]


class HighConvergenceDetector:
    """
    Scans a RuntimeGraph and emits HighConvergenceAlert objects for every
    node whose convergence_potential (scaled 0–100) exceeds *threshold*.

    The detector derives component maturity from the node's physical-axis
    scores (scientific_score, maturity_score, etc.) so it works without any
    additional input beyond what MotherAnnotator / IngestionPipeline already
    populate.

    Example
    -------
    detector = HighConvergenceDetector(threshold=72.5)
    alerts   = detector.scan(graph)
    for alert in alerts:
        print(alert.to_readable())
    """

    # Mapping: (display_name, KnowledgeNode attribute name)
    _COMPONENT_MAP: List[Tuple[str, str]] = [
        ("LLM semantic understanding",      "scientific_score"),
        ("Graph neural networks",           "structural_dependency_index"),
        ("Knowledge graph infrastructure",  "cascade_influence"),
        ("Temporal modeling",               "maturity_score"),
        ("Multi-modal integration",         "social_score"),
        ("Forecasting methodology",         "readiness_score"),
        ("Interpretability",                "investment_score"),
    ]

    # Scale factor: node scores are 0–10, components expect 0–100
    _SCORE_SCALE = 10.0

    def __init__(
        self,
        threshold: float = 72.5,
        # Bottlenecks, analogues, builders, etc. can be injected at
        # construction time so domain experts can customise them without
        # subclassing.
        default_bottlenecks:       Optional[List[BottleneckItem]]    = None,
        default_analogues:         Optional[List[HistoricalAnalogue]] = None,
        default_likely_builders:   Optional[List[LikelyBuilder]]     = None,
        default_predicted_forms:   Optional[List[PredictedForm]]     = None,
        default_monitoring_signals: Optional[List[MonitoringSignal]] = None,
        default_self_awareness_note: str = "",
    ) -> None:
        self.threshold               = threshold
        self.default_bottlenecks     = default_bottlenecks     or []
        self.default_analogues       = default_analogues       or []
        self.default_likely_builders = default_likely_builders or []
        self.default_predicted_forms = default_predicted_forms or []
        self.default_monitoring_signals = default_monitoring_signals or []
        self.default_self_awareness_note = default_self_awareness_note

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(self, graph: "RuntimeGraph") -> List[HighConvergenceAlert]:
        """
        Return one HighConvergenceAlert per node that exceeds the threshold,
        sorted by convergence score descending (highest priority first).
        """
        alerts: List[HighConvergenceAlert] = []
        for node in graph.nodes.values():
            score = self._node_score(node)
            if score >= self.threshold:
                alerts.append(self._build_alert(node, score))
        alerts.sort(key=lambda a: a.convergence_score, reverse=True)
        return alerts

    def alert_for_node(self, node: "KnowledgeNode") -> Optional[HighConvergenceAlert]:
        """
        Build an alert for a specific node regardless of whether it is
        part of a RuntimeGraph.  Returns None if the node is below threshold.
        """
        score = self._node_score(node)
        if score < self.threshold:
            return None
        return self._build_alert(node, score)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _node_score(node: "KnowledgeNode") -> float:
        """
        Map convergence_potential (0–1 or 0–10) to a 0–100 score.
        If convergence_potential is not set, fall back to the mean of
        scientific_score, maturity_score, and readiness_score.
        """
        pot = node.convergence_potential
        if pot > 0:
            # convergence_potential is stored as 0–1 by most components
            return min(pot * 100.0, 100.0) if pot <= 1.0 else min(pot * 10.0, 100.0)
        # Fallback: average of the three most reliable maturity proxies
        proxies = [node.scientific_score, node.maturity_score, node.readiness_score]
        valid   = [v for v in proxies if v > 0]
        if not valid:
            return 0.0
        return min(sum(valid) / len(valid) * 10.0, 100.0)   # scores are 0–10

    def _build_components(self, node: "KnowledgeNode") -> List[ComponentMaturity]:
        """
        Derive component maturities from the node's scalar attributes.
        Each score is on 0–10 scale → multiply by _SCORE_SCALE for percent.
        """
        result: List[ComponentMaturity] = []
        for name, attr in self._COMPONENT_MAP:
            raw = getattr(node, attr, 0.0) or 0.0
            # Normalise: if score looks like 0–1 range, rescale to 0–100
            pct = (raw * 100.0) if raw <= 1.0 else (raw * self._SCORE_SCALE)
            pct = min(max(pct, 0.0), 100.0)
            result.append(ComponentMaturity(name=name, percent=pct))
        return result

    def _derive_emergence_window(self, node: "KnowledgeNode") -> Tuple[str, str]:
        """
        Estimate a qualitative emergence window from temporal attributes.
        Returns (window_label, confidence_interval_label).
        """
        now   = datetime.utcnow()
        year  = now.year
        month = now.month

        # Use maturity_score as a proxy for how close emergence is:
        # maturity close to 10 → very soon; close to 0 → further out
        maturity = min(max(node.maturity_score or 5.0, 0.0), 10.0)
        months_out = max(1, int((10.0 - maturity) * 4))   # 1–40 months

        def quarter(m: int, y: int) -> str:
            q = (m - 1) // 3 + 1
            return f"Q{q} {y}"

        def add_months(base_month: int, base_year: int, n: int) -> Tuple[int, int]:
            total  = base_month - 1 + n
            return (total % 12) + 1, base_year + total // 12

        mid_m, mid_y   = add_months(month, year, months_out)
        early_m, early_y = add_months(month, year, max(1, months_out - 3))
        late_m, late_y  = add_months(month, year, months_out + 6)

        window = f"{quarter(mid_m, mid_y)}"
        ci     = f"[{quarter(early_m, early_y)}, {quarter(late_m, late_y)}]"
        return window, ci

    def _build_alert(self, node: "KnowledgeNode", score: float) -> HighConvergenceAlert:
        """Assemble a complete HighConvergenceAlert from a KnowledgeNode."""
        window, ci = self._derive_emergence_window(node)

        # Probability: derived from convergence_potential and void_pressure proxy
        probability = min(score + (node.upstream_pressure or 0.0) * 5.0, 99.9)
        margin      = max(1.0, (100.0 - probability) / 5.0)

        return HighConvergenceAlert(
            node_id           = node.id,
            label             = node.text,
            description       = node.full_text or node.text,
            convergence_score = score,
            threshold         = self.threshold,
            components        = self._build_components(node),
            bottlenecks       = self.default_bottlenecks,
            analogues         = self.default_analogues,
            emergence_window  = window,
            confidence_interval = ci,
            probability       = probability,
            probability_margin  = margin,
            likely_builders   = self.default_likely_builders,
            predicted_forms   = self.default_predicted_forms,
            monitoring_signals  = self.default_monitoring_signals,
            self_awareness_note = self.default_self_awareness_note,
        )


class PathAgnosticInference:
    def __init__(self, graph_walker: GraphWalker,
                 pressure_field: ConvergencePressureField,
                 phantom_gen: PhantomNodeGenerator,
                 entropy_monitor: GraphEntropyMonitor,
                 priority_head: ContextualPriorityHead,
                 dormancy_tracker: DormancyTracker):
        self.walker   = graph_walker
        self.pressure = pressure_field
        self.phantoms = phantom_gen
        self.entropy  = entropy_monitor
        self.priority = priority_head
        self.dormancy = dormancy_tracker

    def query(self, query_text: str, embedder, graph: RuntimeGraph,
              n_paths: int = 20, future_horizon: float = 5.0) -> ConvergenceCloud:
        trajectories = self.walker.query(query_text, embedder, max_trajectories=n_paths)
        if not trajectories:
            return ConvergenceCloud(query=query_text, convergence_domain="no convergence detected",
                                    predicted_year_range=(0, 0), n_independent_paths=0,
                                    redundancy_score=0.0, total_attention_flow=0.0,
                                    convergence_probability=0.0, top_trajectories=[],
                                    void_pressure_at_target=0.0, active_phantoms=[],
                                    entropy_alerts=[])
        domain_counts: Dict[str, int] = defaultdict(int)
        for traj in trajectories:
            if traj.steps: domain_counts[traj.steps[-1].node_domain] += 1
        convergence_domain = max(domain_counts, key=domain_counts.get)
        node_sets = [frozenset(s.node_id for s in t.steps) for t in trajectories]
        n_independent = sum(1 for i, ns in enumerate(node_sets)
                            if all(len(ns & prev) / len(ns | prev) < CONFIG["path_agnostic_independence_overlap_threshold"]
                                   for prev in node_sets[:i]))
        redundancy  = math.log1p(n_independent)
        total_flow  = sum(t.total_confidence * (1.0 + CONFIG["path_agnostic_cross_domain_bonus"] * t.cross_domain_hops)
                          for t in trajectories)
        conv_prob   = min(0.99, 1.0 - math.exp(-redundancy * CONFIG["path_agnostic_conv_prob_decay"]))
        endpoint_ids = [t.steps[-1].node_id for t in trajectories if t.steps]
        pressures = [self.pressure.field.get(nid, VoidPressureVector(nid)).void_pressure
                     for nid in endpoint_ids]
        mean_pressure = float(np.mean(pressures)) if pressures else 0.0
        end_years = [_ts_to_year(t.steps[-1].node_timestamp) for t in trajectories if t.steps]
        y_min = min(end_years) if end_years else 2025
        y_max = max(end_years) + int(future_horizon) if end_years else 2030
        relevant_phantoms = [p for p in self.phantoms.phantoms
                             if not p.is_confirmed and p.predicted_domain
                             and convergence_domain.lower()[:CONFIG["path_agnostic_domain_match_chars"]] in p.predicted_domain.lower()][:CONFIG["path_agnostic_max_phantoms"]]
        alerts = [(d, t) for d, t, _ in self.entropy.get_alerts()
                  if convergence_domain.lower()[:CONFIG["path_agnostic_alert_match_chars"]] in d.lower()][:CONFIG["path_agnostic_max_alerts"]]
        return ConvergenceCloud(
            query=query_text, convergence_domain=convergence_domain,
            predicted_year_range=(y_min, y_max), n_independent_paths=n_independent,
            redundancy_score=redundancy, total_attention_flow=total_flow,
            convergence_probability=conv_prob,
            top_trajectories=sorted(trajectories, key=lambda t: t.total_confidence,
                                    reverse=True)[:CONFIG["path_agnostic_top_trajectories"]],
            void_pressure_at_target=mean_pressure,
            active_phantoms=relevant_phantoms, entropy_alerts=alerts)


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

    def add_node(self, embedding: np.ndarray) -> np.ndarray:
        emb = embedding.astype(np.float64)
        if self._size == 0:
            self._centroid = emb.copy()
        else:
            w = self.alpha_add / math.sqrt(max(1, self._size))
            delta = emb - self._centroid
            c_unit = self._centroid / (np.linalg.norm(self._centroid) + 1e-12)
            parallel = np.dot(delta, c_unit) * c_unit
            orthogonal = delta - parallel
            self._size_drift     += float(np.linalg.norm(parallel)) * w
            self._semantic_drift += float(np.linalg.norm(orthogonal)) * w
            self._centroid = self._centroid + w * delta
        self._size += 1
        return self._centroid.copy()

    def epoch_update(self, all_embeddings: List[np.ndarray]) -> np.ndarray:
        if not all_embeddings: return self._centroid.copy()
        prev = self._centroid.copy()
        mat = np.stack([e.astype(np.float64) for e in all_embeddings])
        current = np.median(mat, axis=0)
        for _ in range(CONFIG["centroid_weiszfeld_iterations"]):
            dists = np.linalg.norm(mat - current, axis=1) + 1e-8
            weights = 1.0 / dists; weights /= weights.sum()
            current = (mat * weights[:, None]).sum(axis=0)
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
        if node.zone_id:
            self._get_or_create(f"zone:{node.zone_id}").add_node(node.embedding[:self.dim])
        return self._get_or_create(node.domain).add_node(node.embedding[:self.dim])

    def epoch_update(self, graph: RuntimeGraph):
        domain_embs: Dict[str, List[np.ndarray]] = defaultdict(list)
        zone_embs:   Dict[str, List[np.ndarray]] = defaultdict(list)
        for nid, node in graph.nodes.items():
            if node.embedding is None or node.is_temporal_zone: continue
            domain_embs[node.domain].append(node.embedding[:self.dim])
            if node.zone_id: zone_embs[f"zone:{node.zone_id}"].append(node.embedding[:self.dim])
        for cluster_id, embs in {**domain_embs, **zone_embs}.items():
            self._get_or_create(cluster_id).epoch_update(embs)

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


# ══════════════════════════════════════════════════════════════════════════════
#  MOTHER ANNOTATOR
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  MOTHER ANNOTATOR
# ══════════════════════════════════════════════════════════════════════════════

class MotherAnnotator:
    """
    Mother-LLM Pipeline: Performs preprocessing of raw data.
    Transforms text into highly structured objects for Oracle-1,
    acting as the "sensory system" of the graph intelligence.

    .. deprecated::
        Only processes the first 12,000 characters and does not validate the LLM response.
        Use IngestionPipeline — chunking, 3-pass extraction, validation,
        cross-document linking, BatchIngestionProcessor + GraphConsolidator.
    """

    ANNOTATION_PROMPT = """You are a Technical Strategist and Systems Architect. 
Your goal is to build a structured knowledge graph from the provided literature.
Every node and edge must be analyzed for its strategic and physical properties.

SCHEMA INSTRUCTIONS:

1. NODES:
   - text: Concise functional summary (max 15 words).
   - entity_type: Role (e.g., 'physical_limit', 'incumbent_tech', 'side_effect_tool', 'cultural_phantom').
   - strategic_value: 0-10 (Importance for the domain).
   - efficiency_plateau: 0-10 (For incumbent tech: what is its maximum theoretical efficiency?).
   - physical_substrate: Score 10 axes (0.0-1.0): 
     [resource_intensity, scale_feasibility, constraint_proximity, reversibility, 
      parallel_deployability, environmental_coupling, cross_scale_stability, 
      synthesis_complexity, longevity, cascadability].

2. EDGES:
   - Identifiy connections between extracted nodes.
   - For each edge, assign:
     - semantic_similarity (0-1)
     - limitation_resolution (0-1): Does A solve a bottleneck in B?
     - inhibitory_force (0-1): Does A actively suppress or compete with B? (e.g., Oil vs Hydrogen).
     - temporal_proximity, citation_link, investment_correlation, social_correlation.

3. TEMPORAL ZONES:
   - Group nodes into acceleration windows if the text indicates rapid progress in a specific domain.

Return ONLY valid JSON with keys: "nodes", "edges", "temporal_zones".
"""

    def __init__(self, llm_client, sentence_embedder=None):
        """
        llm_client: An object with the .complete(prompt) method (OpenAI, Claude, etc. is not suitable,
        since processing just one document requires 5-20 requests to the model; a local LLM is needed)
        sentence_embedder: Optional model for generating embeddings (e.g. SentenceTransformer)
        """
        self.llm = llm_client
        self.embedder = sentence_embedder
        self.node_registry: Dict[str, KnowledgeNode] = {}
        self.edge_registry: List[KnowledgeEdge]      = []
        self.zone_registry: List[TemporalZone]        = []

    def annotate(self, text: str, source: str, timestamp: float) -> Tuple[List[KnowledgeNode], List[KnowledgeEdge], List[TemporalZone]]:
        """
        The main input method for processing a document.
        """
        full_prompt = f"{self.ANNOTATION_PROMPT}\n\nSOURCE: {source}\nTIMESTAMP: {timestamp}\nTEXT:\n{text[:12000]}"
        
        try:
            raw_response = self.llm.complete(full_prompt)
            # Clearing out Markdown wrappers if LLM added them
            clean_json = raw_response.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
        except Exception as e:
            logging.error(f"MotherAnnotator failed to parse LLM response: {e}")
            return [], [], []

        # 1. Parsing nodes
        nodes = self._parse_nodes(data.get("nodes", []), source, timestamp)
        
        # 2. Parsing the ribs
        edges = self._parse_edges(data.get("edges", []), timestamp)
        
        # 3. Parsing time zones
        zones = self._parse_zones(data.get("temporal_zones", []), timestamp, nodes)

        # 4. Create containment edges for zones
        for zone in zones:
            for member_id in zone.contained_node_ids:
                c_edge = KnowledgeEdge(
                    id=str(uuid.uuid4()),
                    source_id=zone.id,
                    target_id=member_id,
                    relationship_type="contains",
                    is_containment_edge=True,
                    timestamp=timestamp,
                    confidence=1.0
                )
                self.edge_registry.append(c_edge)
                edges.append(c_edge)

        return nodes, edges, zones

    def _parse_nodes(self, items: List[dict], source: str, timestamp: float) -> List[KnowledgeNode]:
        nodes = []
        phys_encoder = PhysicalSubstrateEncoder()
        feat_builder = NodeFeatureBuilder(self.embedder) if self.embedder else None

        for item in items:
            node_id = item.get("id") or str(uuid.uuid4())[:8]

            node = KnowledgeNode(
                id=node_id,
                text=item.get("text", ""),
                node_type=item.get("node_type", "web"),
                domain=item.get("domain", "unknown"),
                entity_type=item.get("entity_type", ""),
                provenance=source,
                timestamp=timestamp,
                # Readiness / quality scores
                scientific_score=float(item.get("scientific_score", 5.0)),
                investment_score=float(item.get("investment_score", 0.0)),
                strategic_value=float(item.get("strategic_value", 5.0)),
                efficiency_plateau=float(item.get("efficiency_plateau", 0.0)),
                # Risk axes
                dual_use_risk=float(item.get("dual_use_risk", 0.0)),
                legal_risk_score=float(item.get("legal_risk_score", 0.0)),
                export_control_risk=float(item.get("export_control_risk", 0.0)),
                # Relational data
                solves_limitations=item.get("solves_limitations", []),
                requires_node_ids=item.get("requires_node_ids", []),
                enables_node_ids=item.get("enables_node_ids", []),
            )

            # ── Physical substrate (10 constraint axes) ───────────────────────
            # Build the 407-dim base feature vector via NodeFeatureBuilder, then
            # extend it to 416 dims by appending the physical section via
            # PhysicalSubstrateEncoder.extend_node_features().
            # The extended vector is stored in node.embedding so that the GNN
            # receives the full PhysicalSubstrateEncoder.EXTENDED_TOTAL_DIM input.
            phys_scores: Dict[str, float] = item.get("physical_substrate", {})

            if feat_builder is not None:
                # Build 407-dim structured feature vector; also sets node.embedding
                # to the 384-dim text section as a side effect.
                base_vec = feat_builder.build(node)   # shape (407,)

                if phys_scores:
                    # Validate and clip each axis to [0, 1]
                    clean_phys = {
                        ax: float(np.clip(phys_scores.get(ax, 0.5), 0.0, 1.0))
                        for ax in PHYSICAL_AXIS_ORDER
                    }
                    # Extend base_vec → 416-dim and store as the canonical embedding
                    node.embedding = phys_encoder.extend_node_features(base_vec, clean_phys)
                    # Keep raw dict for GraphConsolidator / resolve_rupture_physics
                    node.__dict__["raw_physical_scores"] = clean_phys
                    # Annotate feasibility so downstream can gate phantom hallucinations
                    phys_vec = phys_encoder.build_physical_section(clean_phys)
                    node.__dict__["feasibility"] = phys_encoder.feasibility_score(phys_vec)
                else:
                    # No physical scores from LLM — use base vector only (407 dims)
                    node.embedding = base_vec
            else:
                # No embedder available — physical scores stored for later processing
                if phys_scores:
                    clean_phys = {
                        ax: float(np.clip(phys_scores.get(ax, 0.5), 0.0, 1.0))
                        for ax in PHYSICAL_AXIS_ORDER
                    }
                    node.__dict__["raw_physical_scores"] = clean_phys

            nodes.append(node)
            self.node_registry[node_id] = node
        return nodes

    def _parse_edges(self, items: List[dict], timestamp: float) -> List[KnowledgeEdge]:
        edges = []
        for item in items:
            edge = KnowledgeEdge(
                id=str(uuid.uuid4()),
                source_id=item.get("source"),
                target_id=item.get("target"),
                # 7-component edge weight model (V6_EDGE_WEIGHTS + inhibitory)
                semantic_similarity=float(item.get("semantic_similarity", 0.0)),
                temporal_proximity=float(item.get("temporal_proximity", 0.0)),
                limitation_resolution=float(item.get("limitation_resolution", 0.0)),
                citation_link=float(item.get("citation_link", 0.0)),
                investment_correlation=float(item.get("investment_correlation", 0.0)),
                social_correlation=float(item.get("social_correlation", 0.0)),
                inhibitory_force=float(item.get("inhibitory_force", 0.0)),  # inhibitory component
                relationship_type=item.get("relationship_type", "related"),
                evidence=item.get("evidence", []),
                timestamp=timestamp,
            )
            edge.compute_total_weight()
            edges.append(edge)
            self.edge_registry.append(edge)
        return edges

    def _parse_zones(self, items: List[dict], timestamp: float, nodes: List[KnowledgeNode]) -> List[TemporalZone]:
        zones = []
        node_ids = {n.id for n in nodes}
        for item in items:
            zone_id = str(uuid.uuid4())[:8]
            zone = TemporalZone(
                id=zone_id,
                description=item.get("description", ""),
                start_timestamp=timestamp,
                end_timestamp=float("inf"),
                domain_focus=item.get("domain_focus", ""),
                zone_multiplier=float(item.get("zone_multiplier", 1.0)),
                contained_node_ids=[nid for nid in item.get("contained_node_ids", []) if nid in node_ids],
                evidence=item.get("evidence", [])
            )
            
            # Propagate zone membership back into each contained node
            for nid in zone.contained_node_ids:
                if nid in self.node_registry:
                    self.node_registry[nid].zone_id = zone_id
                    self.node_registry[nid].acceleration_multiplier = zone.zone_multiplier

            # Create a container node so the zone is a first-class graph vertex
            zone_node = KnowledgeNode(
                id=zone_id,
                text=zone.description,
                node_type="temporal_zone",
                domain=zone.domain_focus,
                entity_type="temporal_zone",
                timestamp=timestamp,
                is_temporal_zone=True,
                zone_multiplier=zone.zone_multiplier,
                contained_node_ids=zone.contained_node_ids
            )
            
            self.node_registry[zone_id] = zone_node
            zones.append(zone)
            self.zone_registry.append(zone)
        return zones



# ══════════════════════════════════════════════════════════════════════════════
#  INGESTION PIPELINE
#  Transforms raw documents into KnowledgeNode / KnowledgeEdge / TemporalZone objects.
#
#  Pipeline: DocumentChunker → IngestionPromptBuilder → LLMAnnotator →
#            ResponseValidator → NodeEdgeAssembler → CrossDocumentLinker →
#            IngestionPipeline → BatchIngestionProcessor
#
#  Fixes over MotherAnnotator:
#    1. Documents > 12 000 chars  → DocumentChunker with overlap
#    2. One prompt for everything → 3 independent passes (entities / physical / edges)
#    3. No validation             → ResponseValidator
#    4. Unstable IDs              → deterministic MD5(source + entity_text)
#    5. Isolated documents        → CrossDocumentLinker via embeddings
# ══════════════════════════════════════════════════════════════════════════════

INGESTION_ENTITY_TYPES: List[str] = [
    "physical_limit", "incumbent_tech", "challenger_tech", "enabler_component",
    "measurement_tool", "side_effect_risk", "regulatory_constraint",
    "cultural_phantom", "convergence_node", "breakthrough_event",
    "research_frontier", "economic_barrier",
]

INGESTION_SOURCE_TYPES: Dict[str, Dict[str, float]] = {
    k: {"trust": v, "citation_weight": {"arxiv": 0.9, "patent": 0.5,
        "nature": 1.0, "science": 1.0, "preprint": 0.6, "news": 0.1,
        "blog": 0.05, "report": 0.4, "book": 0.7,
        "conference": 0.75, "unknown": 0.3}.get(k, 0.3)}
    for k, v in CONFIG["ingestion_source_trust"].items()
}


# ── Layer 1: Chunker ─────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    chunk_id:     str
    source:       str
    text:         str
    chunk_index:  int
    total_chunks: int
    start_char:   int
    end_char:     int
    metadata:     Dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    """Splits a document into overlapping chunks on paragraph boundaries."""

    def __init__(self, chunk_size: int = None, overlap: int = None):
        self.chunk_size = chunk_size or CONFIG["ingestion_chunk_size"]
        self.overlap    = overlap    or CONFIG["ingestion_chunk_overlap"]

    def chunk(self, text: str, source: str) -> List[DocumentChunk]:
        if len(text) <= self.chunk_size:
            return [DocumentChunk(
                chunk_id=self._make_id(source, 0), source=source, text=text,
                chunk_index=0, total_chunks=1, start_char=0, end_char=len(text),
            )]
        paragraphs = re.split(r'\n{2,}', text)
        chunks: List[DocumentChunk] = []
        current_buf: List[str] = []
        current_len = 0
        char_pos    = 0
        chunk_start = 0
        for para in paragraphs:
            para_len = len(para) + 2
            if current_len + para_len > self.chunk_size and current_buf:
                chunk_text = "\n\n".join(current_buf)
                chunks.append(DocumentChunk(
                    chunk_id=self._make_id(source, len(chunks)), source=source,
                    text=chunk_text, chunk_index=len(chunks), total_chunks=-1,
                    start_char=chunk_start, end_char=chunk_start + len(chunk_text),
                ))
                overlap_buf: List[str] = []
                overlap_len = 0
                for p in reversed(current_buf):
                    if overlap_len + len(p) > self.overlap:
                        break
                    overlap_buf.insert(0, p)
                    overlap_len += len(p) + 2
                current_buf = overlap_buf
                current_len = overlap_len
                chunk_start = char_pos - overlap_len
            current_buf.append(para)
            current_len += para_len
            char_pos    += para_len
        if current_buf:
            chunk_text = "\n\n".join(current_buf)
            chunks.append(DocumentChunk(
                chunk_id=self._make_id(source, len(chunks)), source=source,
                text=chunk_text, chunk_index=len(chunks), total_chunks=-1,
                start_char=chunk_start, end_char=chunk_start + len(chunk_text),
            ))
        for c in chunks:
            c.total_chunks = len(chunks)
        return chunks

    @staticmethod
    def _make_id(source: str, idx: int) -> str:
        return hashlib.md5(f"{source}::{idx}".encode()).hexdigest()[:8]


# ── Layer 2: Prompt builder ──────────────────────────────────────────────────

class IngestionPromptBuilder:
    """
    Four independent passes per chunk:
        Pass A — entity extraction (nodes) + secondary context extraction
        Pass B — physical-axis scoring for each node
        Pass C — relationship extraction (edges between nodes in THIS document)
        Pass D — cross-graph matching: secondary contexts → existing graph nodes
    """

    ENTITY_EXTRACTION_PROMPT = """\
You are an expert knowledge graph constructor specialising in technology and science.

TASK: Extract all significant concepts, technologies, limits, barriers, and events
from the text below. Each extracted item becomes a NODE in a knowledge graph.

SOURCE TYPE: {source_type}
DOMAIN HINT: {domain_hint}
CHUNK {chunk_index}/{total_chunks} of document "{source}"

TEXT:
{text}

EXTRACTION RULES:
* Extract 5-25 nodes. Prefer specificity over generality.
* Assign a stable slug ID: snake_case, max 40 chars (e.g. "crispr_cas9_editing").
* entity_type must be one of: {entity_types}
* Scores are 0.0-10.0 unless noted. evidence: 1-3 verbatim quotes max 20 words each.

SECONDARY CONTEXT RULES:
* For each node, extract secondary_contexts: entities from OTHER domains that
  influenced, enabled, constrained, or co-evolved with this node.
* These are NOT the same as solves_limitations/requires_node_ids (which are
  internal to this document). secondary_contexts are EXTERNAL influences that
  may already exist in the knowledge graph under different names or languages.
* For each secondary context provide:
  - text: the entity name as mentioned in the source (raw, do not normalise)
  - role: how it relates ("influenced_by"|"enabled_by"|"constrained_by"|
          "co-evolved_with"|"funded_by"|"competed_with"|"derived_from")
  - domain: best-guess domain from the known domain list
  - entity_type: best-guess entity_type
  - year_hint: approximate year if mentioned, else null
  - aliases: alternative names/spellings you know for this entity (0-3)

OUTPUT FORMAT (ONLY valid JSON, NO markdown fences):
{{
  "domain": "<primary domain of this chunk>",
  "nodes": [
    {{
      "id": "<slug>",
      "text": "<concise functional description, 8-18 words>",
      "full_context": "<1-2 sentence explanation>",
      "entity_type": "<one of the allowed types>",
      "domain": "<sub-domain>", "node_type": "research",
      "scientific_score": <0-10>, "investment_score": <0-10>,
      "social_score": <0-10>, "maturity_score": <0-10>,
      "strategic_value": <0-10>, "efficiency_plateau": <0-10>,
      "dual_use_risk": <0-10>, "legal_risk_score": <0-10>,
      "export_control_risk": <0-10>, "publication_year": <int or null>,
      "solves_limitations": ["<id>"], "requires_node_ids": ["<id>"],
      "evidence": ["<quote1>", "<quote2>"],
      "secondary_contexts": [
        {{
          "text": "<entity name as in source>",
          "role": "<role>",
          "domain": "<domain>",
          "entity_type": "<entity_type>",
          "year_hint": <int or null>,
          "aliases": ["<alt name 1>", "<alt name 2>"]
        }}
      ]
    }}
  ]
}}
"""

    PHYSICAL_SUBSTRATE_PROMPT = """\
You are a physicist assessing technology constraints.

For each node, rate 10 constraint axes on 0.0-1.0.
HIGH means: resource_intensity=enormous resources; constraint_proximity=near limit (bad);
synthesis_complexity=hard to make; scale_feasibility=easy to scale; longevity=long life;
cascadability=enables breakthroughs; reversibility=easily reversible;
parallel_deployability=many instances; environmental_coupling=env-dependent;
cross_scale_stability=works at all scales.

CONTEXT: {context_excerpt}
NODES: {nodes_list}

OUTPUT (ONLY valid JSON):
{{
  "physical_scores": {{
    "<node_id>": {{
      "resource_intensity": <0-1>, "scale_feasibility": <0-1>,
      "constraint_proximity": <0-1>, "reversibility": <0-1>,
      "parallel_deployability": <0-1>, "environmental_coupling": <0-1>,
      "cross_scale_stability": <0-1>, "synthesis_complexity": <0-1>,
      "longevity": <0-1>, "cascadability": <0-1>,
      "rationale": "<one sentence>"
    }}
  }}
}}
"""

    EDGE_EXTRACTION_PROMPT = """\
You are a knowledge graph architect specialising in causal relationships.

TASK: Identify relationships between these nodes. Only assert edges explicitly supported.

EXTRACTED NODES:
{nodes_summary}

ORIGINAL TEXT:
{text}

Edge scores (0-1): semantic_similarity, temporal_proximity, limitation_resolution,
citation_link, investment_correlation, social_correlation, inhibitory_force.
relationship_type: "enables"|"resolves_limit"|"competes_with"|"requires"|
                   "precedes"|"cites"|"co-evolves_with"|"inhibits"|"related"

OUTPUT (ONLY valid JSON):
{{
  "edges": [{{
    "source": "<id>", "target": "<id>", "relationship_type": "<type>",
    "semantic_similarity": <0-1>, "temporal_proximity": <0-1>,
    "limitation_resolution": <0-1>, "citation_link": <0-1>,
    "investment_correlation": <0-1>, "social_correlation": <0-1>,
    "inhibitory_force": <0-1>, "confidence": <0-1>, "evidence": ["<quote>"]
  }}],
  "temporal_zones": [{{
    "description": "<period>", "domain_focus": "<domain>",
    "zone_multiplier": <1-5>, "contained_node_ids": ["<id>"],
    "evidence": ["<quote>"]
  }}]
}}
"""

    MERGE_PROMPT = """\
You are merging entity extractions from multiple chunks of the same document.
Identify which nodes refer to the same concept and should be merged.

CHUNK SUMMARIES:
{chunk_summaries}

OUTPUT (ONLY valid JSON):
{{
  "merge_groups": [{{"canonical_id": "<id>", "merged_ids": ["<id1>"], "reason": "<why>"}}],
  "cross_chunk_edges": [{{
    "source": "<id>", "target": "<id>", "relationship_type": "<type>",
    "semantic_similarity": <0-1>, "limitation_resolution": <0-1>,
    "inhibitory_force": <0-1>, "confidence": <0-1>
  }}]
}}
"""

    PASS_D_PROMPT = """\
You are a knowledge graph linker. A new node was extracted from a document.
It references external entities (secondary contexts) that may already exist
in the knowledge graph under different names, languages, or spellings.

NEW NODE:
  id:          {node_id}
  text:        {node_text}
  domain:      {node_domain}
  entity_type: {node_entity_type}

SECONDARY CONTEXT TO RESOLVE:
  raw text:    "{ctx_text}"
  role:        {ctx_role}
  domain:      {ctx_domain}
  entity_type: {ctx_entity_type}
  year_hint:   {ctx_year}
  known aliases provided by LLM: {ctx_aliases}

GRAPH CANDIDATES (same domain+type bucket, ranked by embedding similarity):
{candidates_list}

TASK:
1. Decide if any candidate IS the same real-world entity as the secondary context.
   Consider: different languages, transliterations, abbreviations, partial names,
   historical name changes, spelling errors.
2. If a match is found — output its graph_node_id and your confidence.
3. If no candidate matches — output "none" so a new node can be created.

OUTPUT (ONLY valid JSON, NO markdown fences):
{{
  "match": "<graph_node_id or null>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>",
  "relationship_type": "<enables|resolves_limit|competes_with|requires|precedes|cites|co-evolves_with|inhibits|influenced_by|funded_by|derived_from|related>",
  "semantic_similarity": <0.0-1.0>,
  "temporal_proximity": <0.0-1.0>
}}
"""

    def build_entity_prompt(self, chunk: "DocumentChunk", domain_hint: str = "",
                             source_type: str = "unknown") -> str:
        return self.ENTITY_EXTRACTION_PROMPT.format(
            source_type=source_type, domain_hint=domain_hint or "auto-detect",
            chunk_index=chunk.chunk_index + 1, total_chunks=chunk.total_chunks,
            source=chunk.source, text=chunk.text,
            entity_types=", ".join(INGESTION_ENTITY_TYPES),
        )

    def build_physical_prompt(self, nodes: List[Dict], context_excerpt: str) -> str:
        nodes_list = "\n".join(
            f'  {n["id"]}: {n["text"]} [{n.get("entity_type", "")}]' for n in nodes)
        return self.PHYSICAL_SUBSTRATE_PROMPT.format(
            context_excerpt=context_excerpt[:3000], nodes_list=nodes_list)

    def build_edge_prompt(self, nodes: List[Dict], text: str) -> str:
        nodes_summary = "\n".join(f'  {n["id"]}: {n["text"]}' for n in nodes)
        return self.EDGE_EXTRACTION_PROMPT.format(
            nodes_summary=nodes_summary, text=text[:6000])

    def build_merge_prompt(self, chunk_summaries: List[Dict]) -> str:
        return self.MERGE_PROMPT.format(
            chunk_summaries=json.dumps(chunk_summaries, ensure_ascii=False, indent=2))

    def build_pass_d_prompt(self,
                             node: Dict,
                             ctx: Dict,
                             candidates: List[Dict]) -> str:
        """Build Pass D prompt for one (node, secondary_context) pair."""
        if candidates:
            cand_lines = "\n".join(
                f'  [{i+1}] id={c["id"]} text="{c["text"]}" '
                f'domain={c.get("domain","?")} '
                f'entity_type={c.get("entity_type","?")} '
                f'sim={c.get("_sim", 0.0):.3f}'
                for i, c in enumerate(candidates)
            )
        else:
            cand_lines = "  (no candidates found in bucket)"
        return self.PASS_D_PROMPT.format(
            node_id=node.get("id", ""),
            node_text=node.get("text", ""),
            node_domain=node.get("domain", ""),
            node_entity_type=node.get("entity_type", ""),
            ctx_text=ctx.get("text", ""),
            ctx_role=ctx.get("role", "related"),
            ctx_domain=ctx.get("domain", ""),
            ctx_entity_type=ctx.get("entity_type", ""),
            ctx_year=ctx.get("year_hint", "unknown"),
            ctx_aliases=", ".join(ctx.get("aliases", [])) or "none",
            candidates_list=cand_lines,
        )


# ── Layer 3: LLM Annotator ────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    raw:       str
    parsed:    Optional[Dict]
    success:   bool
    error:     Optional[str] = None
    attempts:  int           = 1
    latency_s: float         = 0.0


class LLMAnnotator:
    """
    Wrapper around llm_client with retry logic, JSON cleanup.
    """
    MAX_RETRIES     = 3
    RETRY_DELAY_S   = 2.0
    MAX_TOKENS_PASS = {"entities": 4096, "physical": 2048, "edges": 3072, "merge": 2048}

    def __init__(self, llm_client, model: Optional[str] = None):
        # choose the model from argument or fall back to config default
        self.llm   = llm_client
        self.model = model or CONFIG.get("llm_default_model") or CONFIG["llm_local_models"][0]

    def call(self, prompt: str, pass_type: str = "entities") -> LLMResponse:
        max_tokens = self.MAX_TOKENS_PASS.get(pass_type, 3000)
        for attempt in range(1, self.MAX_RETRIES + 1):
            t0 = time.time()
            try:
                raw    = self._call_llm(prompt, max_tokens)
                parsed = self._parse_json(raw)
                return LLMResponse(raw=raw, parsed=parsed, success=True,
                                   attempts=attempt, latency_s=time.time() - t0)
            except json.JSONDecodeError as e:
                logger.warning(f"[LLMAnnotator] JSON parse failed (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_S * attempt)
            except Exception as e:
                logger.error(f"[LLMAnnotator] LLM call failed (attempt {attempt}): {e}")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY_S * attempt)
        return LLMResponse(raw="", parsed=None, success=False,
                           error="All retry attempts failed", attempts=self.MAX_RETRIES)

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM with support for Anthropic SDK, OpenAI SDK, and local models."""
        # Local models (LocalLLMClient, etc.) — simple interface
        if isinstance(self.llm, LocalLLMClient) or not hasattr(self.llm, "messages") and not hasattr(self.llm, "chat"):
            return self.llm.complete(prompt)
        
        # Anthropic SDK
        if hasattr(self.llm, "messages"):
            resp = self.llm.messages.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.content[0].text
        
        # OpenAI SDK
        if hasattr(self.llm, "chat"):
            resp = self.llm.chat.completions.create(
                model=self.model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}])
            return resp.choices[0].message.content
        
        # Fallback
        return self.llm.complete(prompt)

    @staticmethod
    def _parse_json(raw: str) -> Dict:
        return _parse_json_safe(raw)


# ── Layer 4: Validator ────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:    bool
    warnings: List[str] = field(default_factory=list)
    errors:   List[str] = field(default_factory=list)
    fixed:    Dict      = field(default_factory=dict)


class ResponseValidator:
    """
    Validates and fixes LLM output:
      - numeric field ranges (0-10 and 0-1)
      - edge references to existing nodes
      - duplicate IDs and required fields
    """
    SCORE_FIELDS_10 = [
        "scientific_score", "investment_score", "social_score",
        "maturity_score", "strategic_value", "efficiency_plateau",
        "dual_use_risk", "legal_risk_score", "export_control_risk",
    ]
    SCORE_FIELDS_01 = [
        "semantic_similarity", "temporal_proximity", "limitation_resolution",
        "citation_link", "investment_correlation", "social_correlation",
        "inhibitory_force", "confidence",
    ] + PHYSICAL_AXIS_ORDER

    def validate_nodes(self, data: Dict) -> ValidationResult:
        result    = ValidationResult(valid=True)
        seen_ids: set = set()
        fixed_nodes   = []
        for i, node in enumerate(data.get("nodes", [])):
            if not node.get("id"):
                node["id"] = f"auto_{i:04d}"
                result.warnings.append(f"Node {i}: missing id assigned {node['id']}")
            if not node.get("text"):
                result.errors.append(f"Node {node['id']}: missing text — skipped")
                continue
            if node["id"] in seen_ids:
                node["id"] = f"{node['id']}_{i}"
                result.warnings.append(f"Duplicate id resolved to {node['id']}")
            seen_ids.add(node["id"])
            for f in self.SCORE_FIELDS_10:
                v = node.get(f)
                if v is None:
                    node[f] = 5.0
                else:
                    try:
                        node[f] = float(np.clip(float(v), 0.0, 10.0))
                    except (ValueError, TypeError):
                        node[f] = 5.0
                        result.warnings.append(f"Node {node['id']}.{f}: non-numeric → 5.0")
            if node.get("entity_type") not in INGESTION_ENTITY_TYPES:
                node["entity_type"] = "research_frontier"
            fixed_nodes.append(node)
        result.fixed["nodes"] = fixed_nodes
        return result

    def validate_physical(self, data: Dict, valid_node_ids: set) -> ValidationResult:
        result = ValidationResult(valid=True)
        fixed  = {}
        for node_id, axes in data.get("physical_scores", {}).items():
            if node_id not in valid_node_ids:
                result.warnings.append(f"Physical scores for unknown '{node_id}' — skipped")
                continue
            fixed_axes = {}
            for ax in PHYSICAL_AXIS_ORDER:
                v = axes.get(ax)
                try:
                    fixed_axes[ax] = float(np.clip(float(v), 0.0, 1.0))
                except (ValueError, TypeError):
                    fixed_axes[ax] = 0.5
            fixed[node_id] = fixed_axes
        result.fixed["physical_scores"] = fixed
        return result

    def validate_edges(self, data: Dict, valid_node_ids: set) -> ValidationResult:
        result      = ValidationResult(valid=True)
        fixed_edges = []
        for i, edge in enumerate(data.get("edges", [])):
            src, tgt = edge.get("source"), edge.get("target")
            if src not in valid_node_ids:
                result.warnings.append(f"Edge {i}: unknown source '{src}' — skipped")
                continue
            if tgt not in valid_node_ids:
                result.warnings.append(f"Edge {i}: unknown target '{tgt}' — skipped")
                continue
            if src == tgt:
                result.warnings.append(f"Edge {i}: self-loop on '{src}' — skipped")
                continue
            for f in self.SCORE_FIELDS_01:
                v = edge.get(f)
                try:
                    edge[f] = float(np.clip(float(v), 0.0, 1.0))
                except (ValueError, TypeError):
                    edge[f] = 0.0
            fixed_edges.append(edge)
        result.fixed["edges"]          = fixed_edges
        result.fixed["temporal_zones"] = data.get("temporal_zones", [])
        return result


# ── Layer 5: Object assembler ─────────────────────────────────────────────────

class NodeEdgeAssembler:
    """
    Assembles KnowledgeNode / KnowledgeEdge / TemporalZone from validated dicts.
    IDs are deterministic: MD5(source + "::" + text)[:12] — one concept = one ID.
    """

    def __init__(self, embedder=None):
        self.embedder = embedder

    def make_stable_id(self, source: str, text: str,
                        slug: Optional[str] = None) -> str:
        if slug and re.match(r'^[a-z][a-z0-9_]{2,39}$', slug):
            return f"{slug}_{hashlib.md5(source.encode()).hexdigest()[:4]}"
        return hashlib.md5(f"{source}::{text.lower().strip()}".encode()).hexdigest()[:12]

    def assemble_node(self, item: Dict, source: str, timestamp: float,
                       physical_scores: Optional[Dict[str, float]] = None,
                       source_trust: float = 0.7) -> KnowledgeNode:
        node_id = self.make_stable_id(source, item["text"], item.get("id"))
        node = KnowledgeNode(
            id=node_id, text=item["text"],
            full_text=item.get("full_context", ""),
            node_type=item.get("node_type", "research"),
            domain=item.get("domain", "unknown"),
            entity_type=item.get("entity_type", "research_frontier"),
            provenance=source, timestamp=timestamp,
            publication_date=(str(item["publication_year"])
                              if item.get("publication_year") else None),
            scientific_score=float(item.get("scientific_score", 5.0)) * source_trust,
            investment_score=float(item.get("investment_score", 0.0)),
            social_score=float(item.get("social_score", 0.0)),
            maturity_score=float(item.get("maturity_score", 0.0)),
            strategic_value=float(item.get("strategic_value", 5.0)),
            efficiency_plateau=float(item.get("efficiency_plateau", 0.0)),
            dual_use_risk=float(item.get("dual_use_risk", 0.0)),
            legal_risk_score=float(item.get("legal_risk_score", 0.0)),
            export_control_risk=float(item.get("export_control_risk", 0.0)),
            solves_limitations=item.get("solves_limitations", []),
            requires_node_ids=item.get("requires_node_ids", []),
            enables_node_ids=item.get("enables_node_ids", []),
        )
        if self.embedder is not None:
            try:
                feat_builder = NodeFeatureBuilder(self.embedder)
                phys_encoder = PhysicalSubstrateEncoder()
                # Build 407-dim base feature vector; sets node.embedding to text section
                base_vec = feat_builder.build(node)   # shape (407,)
                if physical_scores:
                    # Validate axis values and extend to 416-dim canonical embedding
                    clean_phys = {
                        ax: float(np.clip(physical_scores.get(ax, 0.5), 0.0, 1.0))
                        for ax in PHYSICAL_AXIS_ORDER
                    }
                    node.embedding = phys_encoder.extend_node_features(base_vec, clean_phys)
                    node.__dict__["raw_physical_scores"] = clean_phys
                    phys_vec = phys_encoder.build_physical_section(clean_phys)
                    node.__dict__["feasibility"] = phys_encoder.feasibility_score(phys_vec)
                else:
                    node.embedding = base_vec
            except Exception as e:
                logger.warning(f"[Assembler] Feature build failed for '{node_id}': {e}")
        elif physical_scores:
            # No embedder — store raw dict for later processing by GraphConsolidator
            clean_phys = {
                ax: float(np.clip(physical_scores.get(ax, 0.5), 0.0, 1.0))
                for ax in PHYSICAL_AXIS_ORDER
            }
            node.__dict__["raw_physical_scores"] = clean_phys
        return node

    def assemble_edge(self, item: Dict, source: str, timestamp: float,
                       node_id_map: Dict[str, str]) -> KnowledgeEdge:
        src_id = node_id_map.get(item["source"], item["source"])
        tgt_id = node_id_map.get(item["target"], item["target"])
        edge = KnowledgeEdge(
            id=str(uuid.uuid4())[:8], source_id=src_id, target_id=tgt_id,
            semantic_similarity=float(item.get("semantic_similarity", 0.0)),
            temporal_proximity=float(item.get("temporal_proximity", 0.0)),
            limitation_resolution=float(item.get("limitation_resolution", 0.0)),
            citation_link=float(item.get("citation_link", 0.0)),
            investment_correlation=float(item.get("investment_correlation", 0.0)),
            social_correlation=float(item.get("social_correlation", 0.0)),
            inhibitory_force=float(item.get("inhibitory_force", 0.0)),
            confidence=float(item.get("confidence", 0.5)),
            relationship_type=item.get("relationship_type", "related"),
            evidence=item.get("evidence", []), timestamp=timestamp,
        )
        edge.compute_total_weight()
        return edge

    def assemble_zone(self, item: Dict, timestamp: float,
                       valid_node_ids: set,
                       node_id_map: Dict[str, str]) -> Tuple[TemporalZone, KnowledgeNode]:
        zone_id   = str(uuid.uuid4())[:8]
        contained = [node_id_map.get(nid, nid)
                     for nid in item.get("contained_node_ids", [])
                     if node_id_map.get(nid, nid) in valid_node_ids]
        zone = TemporalZone(
            id=zone_id, description=item.get("description", ""),
            start_timestamp=timestamp, end_timestamp=float("inf"),
            domain_focus=item.get("domain_focus", ""),
            zone_multiplier=float(np.clip(item.get("zone_multiplier", 1.0), 1.0, 5.0)),
            contained_node_ids=contained, evidence=item.get("evidence", []),
        )
        zone_node = KnowledgeNode(
            id=zone_id, text=zone.description, node_type="temporal_zone",
            domain=zone.domain_focus, entity_type="temporal_zone",
            timestamp=timestamp, is_temporal_zone=True,
            zone_multiplier=zone.zone_multiplier, contained_node_ids=contained,
        )
        return zone, zone_node


# ── Layer 6: Fuzzy Matcher ────────────────────────────────────────────────────

class FuzzyMatcher:
    """
    Normalises and compares entity names across languages, transliterations,
    abbreviations and spelling variants — without external dependencies.

    Scoring: weighted combination of four signals.
      exact_norm  — normalised string equality          (weight 1.0)
      prefix4     — first-4-char prefix match           (weight 0.6)
      token_jaccard — token-set overlap                 (weight 0.5)
      year_overlap — year-hint proximity                (weight 0.3)

    Typical use: pre-filter bucket candidates before LLM Pass D.
    """

    # Common articles / stopwords across several languages to strip
    _ARTICLES = re.compile(
        r"^(the|a|an|der|die|das|des|dem|den|ein|eine|einer|"
        r"le|la|les|l'|un|une|des|"
        r"el|la|los|las|un|una|"
        r"il|lo|la|gli|le|un|uno|una|"
        r"o|a|os|as|um|uma)\s+",
        re.IGNORECASE,
    )
    # Cyrillic → Latin transliteration via dict.
    # Multi-char digraphs properly disambiguated to avoid collision
    # (previously ж→z collided with з→z; ч→c collided with ц→c).
    # After transliteration the string may be longer but token-Jaccard
    # still works correctly on the expanded form.
    _TRANSLIT_MAP: Dict[str, str] = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ж':'zh','з':'z',
        'и':'i','й':'j','к':'k','л':'l','м':'m','н':'n','о':'o','п':'p',
        'р':'r','с':'s','т':'t','у':'u','ф':'f','х':'kh','ц':'ts','ч':'ch',
        'ш':'sh','щ':'shch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya',
        # Upper-case variants
        'А':'A','Б':'B','В':'V','Г':'G','Д':'D','Е':'E','Ж':'Zh','З':'Z',
        'И':'I','Й':'J','К':'K','Л':'L','М':'M','Н':'N','О':'O','П':'P',
        'Р':'R','С':'S','Т':'T','У':'U','Ф':'F','Х':'Kh','Ц':'Ts','Ч':'Ch',
        'Ш':'Sh','Щ':'Shch','Ъ':'','Ы':'Y','Ь':'','Э':'E','Ю':'Yu','Я':'Ya',
    }

    @classmethod
    def _do_translit(cls, s: str) -> str:
        """Character-by-character Cyrillic → Latin using _TRANSLIT_MAP."""
        return "".join(cls._TRANSLIT_MAP.get(c, c) for c in s)

    @classmethod
    def normalise(cls, text: str) -> str:
        """Lower-case, unicode NFKD, strip articles, collapse spaces."""
        import unicodedata
        s = unicodedata.normalize("NFKD", text)
        # Transliterate Cyrillic → Latin (multi-char digraphs, no collisions)
        s = cls._do_translit(s)
        # Drop combining diacritics
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower().strip()
        s = cls._ARTICLES.sub("", s)
        s = re.sub(r"[\s\-_/.,;:!?\"'()]+", " ", s).strip()
        return s

    @classmethod
    def prefix4(cls, a: str, b: str) -> bool:
        """True if first 4 normalised chars match."""
        na, nb = cls.normalise(a)[:4], cls.normalise(b)[:4]
        return bool(na) and na == nb

    @classmethod
    def token_jaccard(cls, a: str, b: str) -> float:
        """Jaccard similarity on token sets of normalised strings."""
        ta = set(cls.normalise(a).split())
        tb = set(cls.normalise(b).split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)

    @classmethod
    def year_score(cls, year_hint: Optional[int],
                   node_timestamp: Optional[float]) -> float:
        """1.0 if within 5 years, decays to 0.0 at 50+ years difference."""
        if year_hint is None or node_timestamp is None:
            return 0.5  # neutral when unknown
        node_year = int(node_timestamp / 31_536_000) + 1970  # epoch → year approx
        delta = abs(year_hint - node_year)
        return max(0.0, 1.0 - delta / 50.0)

    @classmethod
    def score(cls, query_text: str, candidate_text: str,
              query_year: Optional[int] = None,
              candidate_ts: Optional[float] = None) -> float:
        """Combined fuzzy score in [0, 1]."""
        nq = cls.normalise(query_text)
        nc = cls.normalise(candidate_text)
        exact  = 1.0 if nq == nc else 0.0
        pre    = 0.6 if cls.prefix4(query_text, candidate_text) else 0.0
        jac    = 0.5 * cls.token_jaccard(query_text, candidate_text)
        yr     = 0.3 * cls.year_score(query_year, candidate_ts)
        raw    = exact + pre + jac + yr          # max ≈ 2.4
        return min(1.0, raw / 2.4)


# ── Layer 6b: Bucket Index ────────────────────────────────────────────────────

class BucketIndex:
    """
    Hierarchical node index keyed by (domain_id, entity_type_id).

    Replaces CrossDocumentLinker's flat embedding list with a two-level
    structure that reduces candidate search from O(N) to O(N/B) where
    B = number of non-empty buckets (typically thousands).

    Each bucket stores:
      - A list of lightweight NodeRecord tuples for fuzzy/temporal pre-filter
      - Optionally a FAISS IndexFlatIP for ANN search within the bucket

    Merge / link thresholds are preserved from CONFIG for backward compat.
    """

    MERGE_THRESHOLD = CONFIG["ingestion_merge_threshold"]
    LINK_THRESHOLD  = CONFIG["ingestion_link_threshold"]
    MAX_CANDIDATES  = CONFIG["bucket_max_candidates"]   # top-k sent to Pass-D LLM

    @dataclass
    class _NodeRecord:
        node_id:     str
        text:        str
        domain:      str
        entity_type: str
        timestamp:   Optional[float]
        embedding:   Optional[np.ndarray]

    def __init__(self, existing_nodes: Optional[List] = None):
        # bucket_key → list of _NodeRecord
        self._buckets: Dict[Tuple[int, int], List["BucketIndex._NodeRecord"]] = \
            defaultdict(list)
        # bucket_key → np.ndarray matrix (stacked embeddings) for ANN
        self._matrices: Dict[Tuple[int, int], np.ndarray] = {}
        # flat list for legacy merge detection (same-doc primary nodes)
        self._flat: List[Tuple[str, np.ndarray]] = []

        if existing_nodes:
            for n in existing_nodes:
                self.register_node_obj(n)

    # ── Registration ─────────────────────────────────────────────

    def register_node_obj(self, node) -> None:
        """Register a KnowledgeNode (or compatible object) into the index."""
        domain_id      = DOMAIN_TABLE.get_id(getattr(node, "domain", "unknown"))
        entity_type_id = ENTITY_TYPE_TABLE.get_id(
            getattr(node, "entity_type", "unknown"))
        bkey = (domain_id, entity_type_id)
        rec  = BucketIndex._NodeRecord(
            node_id     = node.id,
            text        = getattr(node, "text", ""),
            domain      = getattr(node, "domain", ""),
            entity_type = getattr(node, "entity_type", ""),
            timestamp   = getattr(node, "timestamp", None),
            embedding   = getattr(node, "embedding", None),
        )
        self._buckets[bkey].append(rec)
        # Invalidate cached matrix for this bucket
        self._matrices.pop(bkey, None)
        # Flat list for merge detection
        if rec.embedding is not None:
            self._flat.append((rec.node_id, rec.embedding))

    def register_node(self, node_id: str, embedding: np.ndarray,
                       domain: str = "unknown",
                       entity_type: str = "unknown",
                       text: str = "",
                       timestamp: Optional[float] = None) -> None:
        """Register by explicit fields (used after assembly)."""
        domain_id      = DOMAIN_TABLE.get_id(domain)
        entity_type_id = ENTITY_TYPE_TABLE.get_id(entity_type)
        bkey = (domain_id, entity_type_id)
        rec  = BucketIndex._NodeRecord(
            node_id=node_id, text=text, domain=domain,
            entity_type=entity_type, timestamp=timestamp, embedding=embedding,
        )
        self._buckets[bkey].append(rec)
        self._matrices.pop(bkey, None)
        self._flat.append((node_id, embedding))

    # ── Candidate retrieval ───────────────────────────────────────

    def get_candidates(self,
                        ctx_text:        str,
                        ctx_domain:      str,
                        ctx_entity_type: str,
                        ctx_year:        Optional[int]      = None,
                        query_embedding: Optional[np.ndarray] = None,
                        top_k:           int                = MAX_CANDIDATES,
                        ) -> List[Dict]:
        """
        Return up to top_k candidate dicts from the matching bucket,
        ranked by combined (embedding ANN + fuzzy string + temporal) score.

        Signature is domain-agnostic: caller supplies the secondary context's
        domain and entity_type (which may differ from the primary node's).

        Domain registration: unknown domain/entity_type strings are
        auto-registered so they receive their own bucket ID rather than
        collapsing into the global bucket-0 "trash" bucket.
        """
        # Auto-register to avoid all unknowns collapsing into (0, 0)
        domain_id      = DOMAIN_TABLE.register(ctx_domain) if ctx_domain and ctx_domain != "unknown" \
                         else DOMAIN_TABLE.get_id(ctx_domain)
        entity_type_id = ENTITY_TYPE_TABLE.register(ctx_entity_type) if ctx_entity_type and ctx_entity_type != "unknown" \
                         else ENTITY_TYPE_TABLE.get_id(ctx_entity_type)
        bkey           = (domain_id, entity_type_id)
        bucket         = list(self._buckets.get(bkey, []))

        if not bucket:
            # Fallback 1: same domain, any entity_type
            for (did, _eid), recs in self._buckets.items():
                if did == domain_id:
                    bucket.extend(recs)

        if not bucket:
            # Fallback 2: embedding-only global search over top scored records.
            # Caps at 4 * top_k to stay bounded regardless of graph size.
            if query_embedding is not None:
                qn = np.linalg.norm(query_embedding)
                if qn > 1e-8:
                    scored_global: List[Tuple[float, "BucketIndex._NodeRecord"]] = []
                    cap = max(top_k * 4, 80)
                    for recs in self._buckets.values():
                        for rec in recs:
                            if rec.embedding is None:
                                continue
                            rn = np.linalg.norm(rec.embedding)
                            if rn < 1e-8:
                                continue
                            sim = float(np.dot(query_embedding, rec.embedding) / (qn * rn))
                            scored_global.append((sim, rec))
                    scored_global.sort(key=lambda x: x[0], reverse=True)
                    bucket = [rec for _, rec in scored_global[:cap]]

        if not bucket:
            return []

        scored: List[Tuple[float, "BucketIndex._NodeRecord"]] = []
        for rec in bucket:
            # Signal A: embedding cosine similarity (uses cached matrix if built)
            emb_sim = 0.0
            if query_embedding is not None and rec.embedding is not None:
                qn = np.linalg.norm(query_embedding)
                rn = np.linalg.norm(rec.embedding)
                if qn > 1e-8 and rn > 1e-8:
                    emb_sim = float(
                        np.dot(query_embedding, rec.embedding) / (qn * rn))

            # Signal B: fuzzy string match
            fuzzy_sim = FuzzyMatcher.score(
                ctx_text, rec.text,
                query_year=ctx_year,
                candidate_ts=rec.timestamp,
            )

            # Combined score (embedding weighted higher when available)
            if query_embedding is not None and rec.embedding is not None:
                combined = 0.6 * emb_sim + 0.4 * fuzzy_sim
            else:
                combined = fuzzy_sim

            scored.append((combined, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, rec in scored[:top_k]:
            results.append({
                "id":          rec.node_id,
                "text":        rec.text,
                "domain":      rec.domain,
                "entity_type": rec.entity_type,
                "_sim":        round(sim, 4),
            })
        return results

    # ── Legacy merge/link detection (same-domain primary nodes) ──

    # Hard limit: even with millions of registered nodes the flat scan is
    # bounded.  Increase if precision matters more than ingestion speed.
    _FLAT_SCAN_CAP = 50_000

    def find_connections(self, new_nodes: List) -> Tuple[List[Tuple], List[Dict]]:
        """
        Backward-compatible method used for same-document primary nodes.
        Detects merge candidates and semantic cross-doc edges via flat scan.
        For primary nodes this is acceptable (small N per document).

        When _flat grows beyond _FLAT_SCAN_CAP, scan is limited to the
        most-recently-registered nodes to bound latency.
        """
        merge_pairs: List[Tuple[str, str]] = []
        cross_edges: List[Dict]            = []
        # Bound the flat list so we don't do O(N_total) at scale
        flat_view = (self._flat[-self._FLAT_SCAN_CAP:]
                     if len(self._flat) > self._FLAT_SCAN_CAP
                     else self._flat)
        for node in new_nodes:
            if node.embedding is None:
                continue
            new_emb  = node.embedding
            new_norm = np.linalg.norm(new_emb)
            if new_norm < 1e-8:
                continue
            sims: List[Tuple[float, str]] = []
            for eid, eemb in flat_view:
                if eid == node.id:
                    continue
                enorm = np.linalg.norm(eemb)
                if enorm < 1e-8:
                    continue
                sims.append((float(np.dot(new_emb, eemb) / (new_norm * enorm)), eid))
            sims.sort(reverse=True)
            for sim, eid in sims[:10]:
                if sim >= self.MERGE_THRESHOLD:
                    merge_pairs.append((node.id, eid))
                    break
                elif sim >= self.LINK_THRESHOLD:
                    cross_edges.append({
                        "source": node.id, "target": eid,
                        "relationship_type": "semantic_cross_doc",
                        "semantic_similarity": sim, "temporal_proximity": 0.3,
                        "limitation_resolution": 0.0, "citation_link": 0.0,
                        "investment_correlation": 0.0, "social_correlation": 0.0,
                        "inhibitory_force": 0.0, "confidence": sim,
                        "evidence": ["cross-document semantic match"],
                    })
        return merge_pairs, cross_edges


# ── Backward-compatible alias ─────────────────────────────────────────────────
CrossDocumentLinker = BucketIndex


# ── Layer 7: Orchestrator ─────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Final result of processing a single document."""
    source:            str
    nodes:             List           = field(default_factory=list)
    edges:             List           = field(default_factory=list)
    zones:             List           = field(default_factory=list)
    containment_edges: List           = field(default_factory=list)
    cross_doc_edges:   List           = field(default_factory=list)
    merge_suggestions: List[Tuple]    = field(default_factory=list)
    stats:             Dict[str, Any] = field(default_factory=dict)
    warnings:          List[str]      = field(default_factory=list)
    errors:            List[str]      = field(default_factory=list)


class IngestionPipeline:
    """
    Main orchestrator:
        document → chunk → PassA(entities + secondary_contexts)
                         → PassB(physical)
                         → PassC(edges within document)
                         → PassD(secondary_contexts → existing graph nodes)
                         → validate → assemble → cross-link → merge

    Pass D resolves secondary contexts (external influences with potentially
    different domain/type) against existing graph nodes via BucketIndex
    routing + FuzzyMatcher pre-filter + one LLM call per context.
    Replaces O(N) global cosine scan for cross-domain edge discovery.

    Usage:
        pipeline = IngestionPipeline(llm_client=my_llm, embedder=my_embedder,
                                     graph=runtime_graph)
        result   = pipeline.ingest(text, source, timestamp, source_type="arxiv")
        load_result_into_graph(result, graph)
    """

    def __init__(self, llm_client, embedder=None,
                 existing_nodes: Optional[List] = None,
                 chunk_size: int = None,
                 graph=None):
        self.chunker   = DocumentChunker(chunk_size=chunk_size)
        self.prompts   = IngestionPromptBuilder()
        self.annotator = LLMAnnotator(llm_client)
        self.validator = ResponseValidator()
        self.assembler = NodeEdgeAssembler(embedder=embedder)
        self.linker    = BucketIndex(existing_nodes or [])
        self._graph    = graph
        if graph is not None and not existing_nodes:
            try:
                for node in graph.nodes.values():
                    self.linker.register_node_obj(node)
                logger.info(
                    f"[IngestionPipeline] BucketIndex pre-populated: "
                    f"{len(graph.nodes)} nodes"
                )
            except Exception as e:
                logger.warning(f"[IngestionPipeline] BucketIndex pre-pop failed: {e}")

    def ingest(self, text: str, source: str, timestamp: float,
               source_type: str = "unknown", domain_hint: str = "") -> IngestionResult:
        result    = IngestionResult(source=source)
        trust     = INGESTION_SOURCE_TYPES.get(
            source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"]
        t_start   = time.time()
        llm_calls = 0
        chunks    = self.chunker.chunk(text, source)
        logger.info(f"[Ingestion] '{source}': {len(chunks)} chunk(s)")
        all_chunk_results: List[Dict] = []
        for chunk in chunks:
            cr = self._process_chunk(chunk, source, timestamp, source_type, domain_hint, trust)
            llm_calls += cr.get("llm_calls", 0)
            result.warnings.extend(cr.get("warnings", []))
            result.errors.extend(cr.get("errors", []))
            result.nodes.extend(cr.get("nodes", []))
            result.edges.extend(cr.get("edges", []))
            result.zones.extend(cr.get("zones", []))
            result.containment_edges.extend(cr.get("containment_edges", []))
            result.cross_doc_edges.extend(cr.get("pass_d_edges", []))
            all_chunk_results.append(cr)
        if len(chunks) > 1:
            mr = self._merge_chunks(all_chunk_results, result)
            result.edges.extend(mr.get("cross_chunk_edges", []))
        valid_ids   = {n.id for n in result.nodes}
        node_id_map = {n.id: n.id for n in result.nodes}
        merge_pairs, cross_edges_dicts = self.linker.find_connections(result.nodes)
        result.merge_suggestions = merge_pairs
        for ed in cross_edges_dicts:
            if ed["source"] in valid_ids and ed["target"] in valid_ids:
                try:
                    result.cross_doc_edges.append(
                        self.assembler.assemble_edge(ed, source, timestamp, node_id_map))
                except Exception as e:
                    result.warnings.append(f"Cross-doc edge assembly failed: {e}")
        for node in result.nodes:
            if node.embedding is not None:
                self.linker.register_node(
                    node.id, node.embedding,
                    domain=getattr(node, "domain", "unknown"),
                    entity_type=getattr(node, "entity_type", "unknown"),
                    text=getattr(node, "text", ""),
                    timestamp=getattr(node, "timestamp", None),
                )
        result.stats = {
            "chunks": len(chunks), "nodes": len(result.nodes),
            "edges": len(result.edges), "zones": len(result.zones),
            "cross_doc_edges": len(result.cross_doc_edges),
            "merge_suggestions": len(result.merge_suggestions),
            "llm_calls": llm_calls,
            "total_time_s": round(time.time() - t_start, 2),
            "warnings": len(result.warnings), "errors": len(result.errors),
        }
        logger.info(f"[Ingestion] Done: {result.stats}")
        return result

    # ── Pass D: secondary contexts → existing graph nodes ─────────────────────

    def _run_pass_d(self,
                    raw_node: Dict,
                    assembled_node_id: str,
                    source: str,
                    timestamp: float,
                    embedder=None) -> List[KnowledgeEdge]:
        """
        For each secondary_context in raw_node:
          1. Route to bucket by (ctx.domain, ctx.entity_type)  — O(1).
          2. Retrieve top-20 candidates via BucketIndex
             (embedding ANN + FuzzyMatcher) — O(bucket_size).
          3. LLM arbitrates: does any candidate == this secondary context?
          4. Match found  → create edge primary_node → matched_graph_node.
          5. No match     → context logged; stub node may be created later.

        This is O(bucket_size) not O(N_total), where bucket_size ≈ N/65536
        for a uniformly distributed graph.
        """
        edges: List[KnowledgeEdge] = []
        secondary_contexts = raw_node.get("secondary_contexts", [])
        if not secondary_contexts:
            return edges

        for ctx in secondary_contexts:
            ctx_text   = ctx.get("text", "").strip()
            ctx_domain = ctx.get("domain", "unknown")
            ctx_etype  = ctx.get("entity_type", "unknown")
            ctx_year   = ctx.get("year_hint")
            ctx_aliases: List[str] = [a for a in ctx.get("aliases", []) if a and a != ctx_text]
            if not ctx_text:
                continue

            # Build query embedding for context text if embedder available
            query_emb: Optional[np.ndarray] = None
            if embedder is not None:
                try:
                    query_emb = np.array(embedder.encode(ctx_text), dtype=np.float32)
                except Exception:
                    pass

            # Stage 1+2: bucket routing + ranked candidate retrieval.
            # Search primary text AND all aliases, then merge+deduplicate by
            # node_id keeping the highest similarity score seen.
            seen_ids: Dict[str, float] = {}   # node_id → best combined score
            seen_records: Dict[str, Dict] = {}

            def _absorb_candidates(cands: List[Dict]) -> None:
                for c in cands:
                    nid  = c["id"]
                    sim  = c.get("_sim", 0.0)
                    if nid not in seen_ids or sim > seen_ids[nid]:
                        seen_ids[nid]      = sim
                        seen_records[nid]  = c

            primary_cands = self.linker.get_candidates(
                ctx_text        = ctx_text,
                ctx_domain      = ctx_domain,
                ctx_entity_type = ctx_etype,
                ctx_year        = ctx_year,
                query_embedding = query_emb,
                top_k           = BucketIndex.MAX_CANDIDATES,
            )
            _absorb_candidates(primary_cands)

            # Search each alias — picks up nodes stored under alternate names
            for alias in ctx_aliases:
                alias_emb: Optional[np.ndarray] = None
                if embedder is not None:
                    try:
                        alias_emb = np.array(embedder.encode(alias), dtype=np.float32)
                    except Exception:
                        pass
                alias_cands = self.linker.get_candidates(
                    ctx_text        = alias,
                    ctx_domain      = ctx_domain,
                    ctx_entity_type = ctx_etype,
                    ctx_year        = ctx_year,
                    query_embedding = alias_emb,
                    top_k           = BucketIndex.MAX_CANDIDATES,
                )
                _absorb_candidates(alias_cands)

            # Re-sort merged pool and trim to MAX_CANDIDATES for LLM prompt
            candidates = sorted(
                seen_records.values(),
                key=lambda c: c.get("_sim", 0.0),
                reverse=True,
            )[:BucketIndex.MAX_CANDIDATES]

            # Stage 3: LLM arbitration (one call per secondary context)
            prompt = self.prompts.build_pass_d_prompt(
                node       = raw_node,
                ctx        = ctx,
                candidates = candidates,
            )
            try:
                rd = self.annotator.call(prompt, pass_type="pass_d")
                if not rd.success or not rd.parsed:
                    logger.debug(
                        f"[PassD] LLM failed for ctx='{ctx_text}' "
                        f"node={assembled_node_id}"
                    )
                    continue

                match_id   = rd.parsed.get("match")
                confidence = float(rd.parsed.get("confidence", 0.0))
                rel_type   = rd.parsed.get("relationship_type", "related")
                sem_sim    = float(rd.parsed.get("semantic_similarity", 0.5))
                temp_prox  = float(rd.parsed.get("temporal_proximity", 0.3))
                reasoning  = rd.parsed.get("reasoning", "")

                if match_id and match_id not in ("null", "none", None)                         and confidence >= 0.4:
                    # Verify match exists in candidate list or live graph
                    valid_match = any(c["id"] == match_id for c in candidates)
                    if not valid_match and self._graph is not None:
                        try:
                            valid_match = self._graph.get_node(match_id) is not None
                        except Exception:
                            pass
                    if valid_match:
                        edge = KnowledgeEdge(
                            id=str(uuid.uuid4())[:8],
                            source_id=assembled_node_id,
                            target_id=match_id,
                            relationship_type=rel_type,
                            semantic_similarity=sem_sim,
                            temporal_proximity=temp_prox,
                            limitation_resolution=0.0,
                            citation_link=0.0,
                            investment_correlation=0.0,
                            social_correlation=0.0,
                            inhibitory_force=0.0,
                            confidence=confidence,
                            timestamp=timestamp,
                            provenance=source,
                            evidence=[f"PassD: {reasoning}"],
                        )
                        edge.compute_total_weight()
                        edges.append(edge)
                        logger.debug(
                            f"[PassD] '{ctx_text}' → {match_id} "
                            f"({rel_type}, conf={confidence:.2f})"
                        )
                    else:
                        logger.debug(
                            f"[PassD] match_id='{match_id}' not in graph "
                            f"for ctx='{ctx_text}'"
                        )
            except Exception as e:
                logger.warning(f"[PassD] error for ctx='{ctx_text}': {e}")

        return edges

    def _process_chunk(self, chunk: DocumentChunk, source: str, timestamp: float,
                        source_type: str, domain_hint: str, trust: float) -> Dict:
        out = {"nodes": [], "edges": [], "zones": [], "containment_edges": [],
               "warnings": [], "errors": [], "llm_calls": 0, "raw_nodes": [],
               "pass_d_edges": []}

        # ── Pass A: entities + secondary_contexts ─────────────────────────────
        ra = self.annotator.call(
            self.prompts.build_entity_prompt(chunk, domain_hint, source_type),
            pass_type="entities")
        out["llm_calls"] += 1
        if not ra.success or not ra.parsed:
            out["errors"].append(f"Chunk {chunk.chunk_index}: entity extraction failed")
            return out
        val_n = self.validator.validate_nodes(ra.parsed)
        out["warnings"].extend(val_n.warnings)
        out["errors"].extend(val_n.errors)
        raw_nodes = val_n.fixed.get("nodes", [])
        if not raw_nodes:
            out["warnings"].append(f"Chunk {chunk.chunk_index}: no valid nodes extracted")
            return out
        slug_map: Dict[str, str] = {}

        # ── Pass B: physical axes ─────────────────────────────────────────────
        rb = self.annotator.call(
            self.prompts.build_physical_prompt(raw_nodes, chunk.text),
            pass_type="physical")
        out["llm_calls"] += 1
        physical_map: Dict[str, Dict] = {}
        if rb.success and rb.parsed:
            val_p = self.validator.validate_physical(rb.parsed, {n["id"] for n in raw_nodes})
            out["warnings"].extend(val_p.warnings)
            physical_map = val_p.fixed.get("physical_scores", {})

        # ── Assemble nodes ────────────────────────────────────────────────────
        assembled_nodes: List = []
        for item in raw_nodes:
            try:
                node = self.assembler.assemble_node(
                    item, source, timestamp,
                    physical_scores=physical_map.get(item["id"]), source_trust=trust)
                slug_map[item["id"]] = node.id
                out["nodes"].append(node)
                out["raw_nodes"].append(item)
                assembled_nodes.append(node)
            except Exception as e:
                out["errors"].append(f"Node assembly failed for '{item['id']}': {e}")

        # ── Pass C: edges within document ─────────────────────────────────────
        rc = self.annotator.call(
            self.prompts.build_edge_prompt(raw_nodes, chunk.text),
            pass_type="edges")
        out["llm_calls"] += 1
        if rc.success and rc.parsed:
            val_e = self.validator.validate_edges(rc.parsed, set(slug_map.keys()))
            out["warnings"].extend(val_e.warnings)
            for ed in val_e.fixed.get("edges", []):
                try:
                    out["edges"].append(
                        self.assembler.assemble_edge(ed, source, timestamp, slug_map))
                except Exception as e:
                    out["edges_errors"] = out.get("edge_errors", [])
                    out["errors"].append(f"Edge assembly failed: {e}")
            valid_stable = {n.id for n in out["nodes"]}
            for zi in val_e.fixed.get("temporal_zones", []):
                try:
                    zone, zone_node = self.assembler.assemble_zone(
                        zi, timestamp, valid_stable, slug_map)
                    out["zones"].append(zone)
                    out["nodes"].append(zone_node)
                    for member_id in zone.contained_node_ids:
                        out["containment_edges"].append(KnowledgeEdge(
                            id=str(uuid.uuid4())[:8],
                            source_id=zone.id, target_id=member_id,
                            relationship_type="contains", is_containment_edge=True,
                            timestamp=timestamp, confidence=1.0))
                except Exception as e:
                    out["errors"].append(f"Zone assembly failed: {e}")

        # ── Pass D: secondary contexts → existing graph nodes ─────────────────
        for raw_node, assembled_node in zip(out["raw_nodes"], assembled_nodes):
            if not raw_node.get("secondary_contexts"):
                continue
            try:
                pass_d_edges = self._run_pass_d(
                    raw_node          = raw_node,
                    assembled_node_id = assembled_node.id,
                    source            = source,
                    timestamp         = timestamp,
                    embedder          = self.assembler.embedder,
                )
                out["pass_d_edges"].extend(pass_d_edges)
                out["llm_calls"] += len(raw_node.get("secondary_contexts", []))
            except Exception as e:
                out["warnings"].append(
                    f"PassD failed for node '{assembled_node.id}': {e}"
                )

        return out

    def _merge_chunks(self, chunk_results: List[Dict],
                       result: IngestionResult) -> Dict:
        summaries = [
            {"chunk": i, "nodes": [{"id": n["id"], "text": n["text"]}
                                    for n in cr.get("raw_nodes", [])]}
            for i, cr in enumerate(chunk_results)
        ]
        response = self.annotator.call(
            self.prompts.build_merge_prompt(summaries), pass_type="merge")
        if not response.success or not response.parsed:
            return {"merged": [], "cross_chunk_edges": []}
        raw_xedges = response.parsed.get("cross_chunk_edges", [])
        valid_ids  = {n.id for n in result.nodes}
        slug_map   = {n.id: n.id for n in result.nodes}
        cross_edges = []
        for ed in raw_xedges:
            src_id = slug_map.get(ed.get("source", ""), ed.get("source", ""))
            tgt_id = slug_map.get(ed.get("target", ""), ed.get("target", ""))
            if src_id in valid_ids and tgt_id in valid_ids and src_id != tgt_id:
                try:
                    edge = KnowledgeEdge(
                        id=str(uuid.uuid4())[:8], source_id=src_id, target_id=tgt_id,
                        relationship_type=ed.get("relationship_type", "related"),
                        semantic_similarity=float(ed.get("semantic_similarity", 0.5)),
                        limitation_resolution=float(ed.get("limitation_resolution", 0.0)),
                        inhibitory_force=float(ed.get("inhibitory_force", 0.0)),
                        confidence=float(ed.get("confidence", 0.5)),
                    )
                    edge.compute_total_weight()
                    cross_edges.append(edge)
                except Exception:
                    pass
        return {"merged": response.parsed.get("merge_groups", []),
                "cross_chunk_edges": cross_edges}

@dataclass
class BatchDocument:
    text:        str
    source:      str
    timestamp:   float
    source_type: str = "unknown"
    domain_hint: str = ""
    metadata:    Dict[str, Any] = field(default_factory=dict)


class BatchIngestionProcessor:
    """
    Processes a queue of documents with rate-limiting on LLM calls.
    Stops after max_errors consecutive failures.
    """
    def __init__(self, pipeline: IngestionPipeline,
                 rate_limit_rpm: int = 60, max_errors: int = 5):
        self.pipeline    = pipeline
        self.min_delay_s = 60.0 / max(rate_limit_rpm, 1)
        self.max_errors  = max_errors

    def process(self, documents: List[BatchDocument],
                progress_callback=None) -> List[IngestionResult]:
        results:      List[IngestionResult] = []
        error_streak: int   = 0
        last_call_t:  float = 0.0
        for i, doc in enumerate(documents):
            elapsed = time.time() - last_call_t
            if elapsed < self.min_delay_s:
                time.sleep(self.min_delay_s - elapsed)
            logger.info(f"[Batch] Processing {i+1}/{len(documents)}: {doc.source}")
            try:
                result = self.pipeline.ingest(
                    text=doc.text, source=doc.source, timestamp=doc.timestamp,
                    source_type=doc.source_type, domain_hint=doc.domain_hint)
                results.append(result)
                error_streak = 0
                last_call_t  = time.time()
                if progress_callback:
                    progress_callback(i + 1, len(documents), result)
            except Exception as e:
                logger.error(f"[Batch] Failed on '{doc.source}': {e}")
                error_streak += 1
                if error_streak >= self.max_errors:
                    logger.critical(f"[Batch] {self.max_errors} consecutive errors — stopping")
                    break
        logger.info(f"[Batch] Complete: {len(results)}/{len(documents)} docs, "
                    f"{sum(len(r.nodes) for r in results)} nodes, "
                    f"{sum(len(r.edges)+len(r.cross_doc_edges) for r in results)} edges")
        return results


def load_result_into_graph(result: IngestionResult, graph) -> Dict[str, int]:
    """Load an IngestionResult into a RuntimeGraph with ancestry invariant enforcement.

    Every non-primordial node must arrive with at least one incoming
    non-containment edge.  When the document doesn't declare one, the
    AncestryEnforcer synthesises a DERIVED_FROM edge to the best-matching
    predecessor already in the graph.

    Returns {"nodes": N, "edges": M, "zones": K, "synthesised_ancestry": S}.
    """
    added = {"nodes": 0, "edges": 0, "zones": 0, "synthesised_ancestry": 0}

    all_edges = result.edges + result.containment_edges + result.cross_doc_edges

    for node in result.nodes:
        # Pass the full edge batch so the enforcer can see edges arriving
        # alongside this node in the same IngestionResult (avoids false
        # positives when source and target land in the same document).
        anchor = graph.add_node_checked(node, batch_edges=all_edges)
        added["nodes"] += 1
        if anchor is not None:
            # anchor was already written to the graph by add_node_checked()
            added["synthesised_ancestry"] += 1
            added["edges"] += 1

    for edge in all_edges:
        if graph.get_node(edge.source_id) and graph.get_node(edge.target_id):
            graph.add_edge(edge)
            added["edges"] += 1

    added["zones"] = len(result.zones)

    if added["synthesised_ancestry"]:
        logger.info(
            f"[Ancestry] {added['synthesised_ancestry']} lineage edge(s) "
            "synthesised during ingestion to satisfy ancestry invariant"
        )
    return added


def merge_nodes_in_graph(merge_pairs: List[Tuple[str, str]], graph) -> int:
    """
    Applies merge suggestions from CrossDocumentLinker.
    Keeps the node with the higher scientific_score and re-points all edges.
    """
    merged = 0
    for new_id, existing_id in merge_pairs:
        new_node = graph.get_node(new_id)
        existing = graph.get_node(existing_id)
        if not new_node or not existing:
            continue
        keeper_id  = (existing_id if existing.scientific_score >= new_node.scientific_score
                      else new_id)
        discard_id = new_id if keeper_id == existing_id else existing_id
        for edge in list(graph.edges.values()):
            if edge.source_id == discard_id:
                edge.source_id = keeper_id
            if edge.target_id == discard_id:
                edge.target_id = keeper_id
        keeper  = graph.get_node(keeper_id)
        discard = graph.get_node(discard_id)
        keeper.scientific_score = max(keeper.scientific_score, discard.scientific_score)
        keeper.investment_score = max(keeper.investment_score, discard.investment_score)
        keeper.strategic_value  = max(keeper.strategic_value,  discard.strategic_value)
        keeper.evidence = (getattr(keeper, "evidence", []) + getattr(discard, "evidence", []))
        graph.nodes.pop(discard_id, None)
        merged += 1
        logger.info(f"[Merge] {discard_id} -> {keeper_id}")
    return merged


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
            MergeStrategy.SUM_UNIQUE:          lambda: FieldMerger._sum_unique(values),
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
    def _temporal_trend(values, timestamps) -> float:
        """Latest value + weak linear extrapolation capped at ±10% of the last value."""
        numeric_pairs = []
        for ts, v in zip(timestamps, values):
            try:
                numeric_pairs.append((float(ts), float(v)))
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
        ts_norm = (ts_arr - ts_arr.mean()) / (ts_arr.std() + 1e-8)
        slope   = float(np.polyfit(ts_norm, v_arr, 1)[0])
        delta   = np.clip(slope * 0.1, -latest * 0.1, latest * 0.1)
        return float(np.clip(latest + delta, 0.0, 10.0))

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
            provenance=", ".join(sorted({o.source for o in obs})),
            timestamp=record.latest_timestamp,
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
                    merged_physical[axis] = float(
                        FieldMerger.apply(strategy, axis_values, axis_weights, axis_ts) or 0.5)
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
            "observation_count": len(obs),
            "sources":          [o.source for o in obs],
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
        threshold = variance_threshold or CONFIG.get(
            "consolidation_conflict_variance_threshold", 2.0)
        conflicts = {}
        for fname in list(NODE_FIELD_STRATEGIES.keys()):
            vals, srcs = [], []
            for o in obs:
                v = o.fields.get(fname)
                try:
                    vals.append(float(v)); srcs.append(o.source)
                except (TypeError, ValueError):
                    pass
            if len(vals) >= 2:
                std = statistics.stdev(vals)
                if std > threshold:
                    conflicts[fname] = {"std": round(std, 3), "min": min(vals),
                                        "max": max(vals), "values": vals, "sources": srcs}
        return conflicts


class EdgeConsolidator:
    """
    Merges N observations of one edge into a single KnowledgeEdge.
    relationship_type is resolved by trust-weighted majority vote.
    """

    def consolidate(self, record: EdgeRecord) -> KnowledgeEdge:
        obs        = record.observations
        weights    = [o.trust for o in obs]
        timestamps = [o.timestamp for o in obs]

        def merge(field_name: str) -> Any:
            s = EDGE_FIELD_STRATEGIES.get(field_name, MergeStrategy.TRUST_WEIGHTED_MEAN)
            return FieldMerger.apply(s, [o.fields.get(field_name, 0.0) for o in obs],
                                     weights, timestamps)

        rt_values  = [o.fields.get("relationship_type", "related") for o in obs]
        primary_rt = FieldMerger._majority_vote(rt_values, weights)
        edge = KnowledgeEdge(
            id=f"cons_{record.source_id[:6]}_{record.target_id[:6]}",
            source_id=record.source_id, target_id=record.target_id,
            semantic_similarity=float(merge("semantic_similarity") or 0.0),
            temporal_proximity=float(merge("temporal_proximity") or 0.0),
            limitation_resolution=float(merge("limitation_resolution") or 0.0),
            citation_link=float(merge("citation_link") or 0.0),
            investment_correlation=float(merge("investment_correlation") or 0.0),
            social_correlation=float(merge("social_correlation") or 0.0),
            inhibitory_force=float(merge("inhibitory_force") or 0.0),
            confidence=float(merge("confidence") or 0.0),
            relationship_type=primary_rt,
            evidence=FieldMerger._union_list([o.fields.get("evidence", []) for o in obs]),
            timestamp=max(o.timestamp for o in obs),
        )
        edge.compute_total_weight()
        edge.__dict__["alternate_relationship_types"] = [
            rt for rt in rt_values if rt != primary_rt]
        edge.__dict__["consolidation_meta"] = {
            "observation_count": len(obs), "sources": [o.source for o in obs]}
        return edge


@dataclass
class ConsolidationStats:
    nodes_before:        int  = 0
    nodes_after:         int  = 0
    edges_before:        int  = 0
    edges_after:         int  = 0
    merges_applied:      int  = 0
    conflicts_found:     int  = 0
    high_conflict_nodes: List[str] = field(default_factory=list)


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
            consolidator.observe_from_ingestion_result(result, doc.source_type)
        stats = consolidator.apply_to_graph(graph)
        print_conflict_report(consolidator)
    """

    def __init__(self):
        self._node_records: Dict[str, NodeRecord]   = {}
        self._edge_records: Dict[Tuple, EdgeRecord] = {}
        self._node_consolidator = NodeConsolidator()
        self._edge_consolidator = EdgeConsolidator()
        self._id_redirects: Dict[str, str]          = {}

    def observe_node(self, node, source: str, timestamp: float,
                     trust: float = 0.7,
                     canonical_id: Optional[str] = None) -> str:
        cid = canonical_id or node.id
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
                     canonical_tgt: Optional[str] = None) -> None:
        src = canonical_src or self._id_redirects.get(edge.source_id, edge.source_id)
        tgt = canonical_tgt or self._id_redirects.get(edge.target_id, edge.target_id)
        key = (src, tgt)
        if key not in self._edge_records:
            self._edge_records[key] = EdgeRecord(source_id=src, target_id=tgt)
        self._edge_records[key].add(
            Observation(source=source, timestamp=timestamp, trust=trust,
                        fields=self._edge_to_fields(edge)))

    def observe_from_ingestion_result(self, result: IngestionResult,
                                       source_type: str = "unknown") -> None:
        trust = INGESTION_SOURCE_TYPES.get(
            source_type, INGESTION_SOURCE_TYPES["unknown"])["trust"]
        ts = max((n.timestamp for n in result.nodes), default=0.0)
        for node in result.nodes:
            self.observe_node(node, source=result.source,
                              timestamp=node.timestamp or ts, trust=trust)
        for edge in result.edges + result.cross_doc_edges:
            self.observe_edge(edge, source=result.source, timestamp=ts, trust=trust)

    def apply_to_graph(self, graph) -> ConsolidationStats:
        stats = ConsolidationStats(
            nodes_before=len(graph.nodes),
            edges_before=len(graph.edges))
        # 1. Consolidate nodes
        consolidated_nodes = {}
        for cid, record in self._node_records.items():
            node = (self._fields_to_node(cid, record.observations[0])
                    if len(record.observations) == 1
                    else self._node_consolidator.consolidate(record))
            if len(record.observations) > 1:
                stats.merges_applied += len(record.observations) - 1
            meta = node.__dict__.get("consolidation_meta", {})
            if meta.get("conflicts"):
                stats.conflicts_found += len(meta["conflicts"])
                stats.high_conflict_nodes.append(cid)
            consolidated_nodes[cid] = node
        # 2. Replace nodes in graph
        for old_id in self._id_redirects:
            graph.nodes.pop(old_id, None)
        for cid, node in consolidated_nodes.items():
            graph.nodes[cid] = node
        # 3. Redirect stale edge IDs
        for edge in list((graph.edges.values()
                          if isinstance(graph.edges, dict) else [])):
            if hasattr(edge, "source_id"):
                edge.source_id = self._id_redirects.get(edge.source_id, edge.source_id)
                edge.target_id = self._id_redirects.get(edge.target_id, edge.target_id)
        # 4. Consolidate duplicate edges
        for key, record in self._edge_records.items():
            if len(record.observations) > 1:
                self._replace_edges_in_graph(
                    graph, key[0], key[1],
                    self._edge_consolidator.consolidate(record))
        stats.nodes_after = len(graph.nodes)
        stats.edges_after = len(graph.edges) if isinstance(graph.edges, dict) else 0
        logger.info(
            f"[GraphConsolidator] Nodes: {stats.nodes_before} -> {stats.nodes_after} "
            f"(-{stats.nodes_before - stats.nodes_after}), "
            f"Merges: {stats.merges_applied}, Conflicts: {stats.conflicts_found}")
        return stats

    def conflict_report(self, top_n: int = 20) -> List[Dict]:
        report = []
        for cid, record in self._node_records.items():
            if len(record.observations) < 2:
                continue
            conflicts = NodeConsolidator._compute_conflicts(record.observations)
            if conflicts:
                report.append({
                    "node_id":         cid,
                    "source_count":    record.source_count,
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
             "fields": {k: v for k, v in o.fields.items()
                        if isinstance(v, (int, float, str)) and v}}
            for o in sorted(record.observations, key=lambda o: o.timestamp)
        ]

    @staticmethod
    def _node_to_fields(node) -> Dict[str, Any]:
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
        d = {f: getattr(node, f, None) for f in scalar_fields}
        for f in list_fields:
            v = getattr(node, f, None)
            d[f] = list(v) if v else []
        emb = getattr(node, "embedding", None)
        if emb is not None:
            d["embedding"] = np.array(emb, dtype=np.float32)
        return d

    @staticmethod
    def _edge_to_fields(edge) -> Dict[str, Any]:
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
        edges_dict = graph.edges if isinstance(graph.edges, dict) else {}
        to_remove  = [k for k, e in edges_dict.items()
                      if (getattr(e, "source_id", None) == src_id and
                          getattr(e, "target_id", None) == tgt_id)]
        for key in to_remove:
            del graph.edges[key]
        if hasattr(graph, "add_edge"):
            graph.add_edge(cons_edge)
        else:
            graph.edges[cons_edge.id] = cons_edge


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
    K_HOP           = 3
    ETA_CONTINUOUS  = 1e-6
    MAX_STEPS       = 50

    def __init__(self, model: Oracle1Model):
        self.model = model
        if TORCH_AVAILABLE:
            import torch.optim as optim
            self.opt = optim.AdamW(model.parameters(), lr=self.ETA_CONTINUOUS, weight_decay=1e-5)

    def feed(self, new_nodes, new_edges, runtime_graph) -> Dict[str, Any]:
        if not TORCH_AVAILABLE: return {"status": "torch_not_available"}
        zone_updates = self._detect_zone_membership(new_nodes, runtime_graph)
        affected = runtime_graph.k_hop_neighborhood([n.id for n in new_nodes], k=self.K_HOP)
        for zone_id, new_members in zone_updates.items():
            zone_node = runtime_graph.get_node(zone_id)
            if zone_node: affected.update(zone_node.contained_node_ids)
        return {"new_nodes": len(new_nodes), "new_edges": len(new_edges),
                "zone_updates": len(zone_updates), "affected_nodes": len(affected)}

    def _detect_zone_membership(self, new_nodes, runtime_graph) -> Dict[str, List[str]]:
        zone_updates: Dict[str, List[str]] = {}
        zones = runtime_graph.get_all_zones()
        for node in new_nodes:
            if node.embedding is None: continue
            for zone in zones:
                centroid = runtime_graph.zone_centroid(zone.id)
                if centroid is None: continue
                sim = float(cos_sim([node.embedding], [centroid])[0][0])
                if sim > 0.65 and node.id not in zone.contained_node_ids:
                    zone.contained_node_ids.append(node.id)
                    node.zone_id = zone.id
                    node.acceleration_multiplier = zone.zone_multiplier
                    zone_updates.setdefault(zone.id, []).append(node.id)
        return zone_updates


# ══════════════════════════════════════════════════════════════════════════════
#  ORACLE DATASET — materialises RuntimeGraph into training examples (Fix 3)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OracleExample:
    """One training example: a pair of nodes and supervision labels."""
    # Node pair
    src_id: str
    tgt_id: str
    # Features
    src_features: np.ndarray            # NodeFeatureBuilder.TOTAL_DIM
    tgt_features: np.ndarray
    edge_attr: np.ndarray               # EdgeFeatureBuilder.DIM  (zeros if no edge)
    topology_vec: np.ndarray            # LocalTopologyStats → 12-dim vector
    # Labels
    label_exists: float                 # 1.0 if edge (src, tgt) exists
    label_components: np.ndarray        # N_EXTENDED_COMPONENTS regression targets
    label_convergence: float            # 1.0 if pair participates in confirmed conv-point
    label_convergence_eta: float        # epochs until convergence (0 if unknown)


class OracleDataset:
    """Dataset that materialises a RuntimeGraph into OracleExample objects.

    Use ``from_graph()`` to build the dataset from a live graph snapshot.
    This connects the graph-construction pipeline to the neural training loop,
    completing the two-phase architecture described in train.py.
    """

    def __init__(self, examples: List[OracleExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> OracleExample:
        return self.examples[idx]

    @classmethod
    def from_graph(
        cls,
        graph: "RuntimeGraph",
        node_feature_builder: "NodeFeatureBuilder" = None,
        edge_feature_builder: "EdgeFeatureBuilder" = None,
        pressure_field: "ConvergencePressureField" = None,
        dormancy_tracker: "DormancyTracker" = None,
        convergence_points: Optional[List["ConvergencePoint"]] = None,
        cutoff_ts: Optional[float] = None,
        horizon_epochs: float = 60.0,
        max_node_pairs: int = 50_000,
    ) -> "OracleDataset":
        """Materialise *graph* into training examples.

        Each example is a (src, tgt) node-pair drawn from:
        - All existing edges in the graph.
        - A stratified random sample of non-edges up to *max_node_pairs*.

        Args:
            graph: Live RuntimeGraph to sample from.
            node_feature_builder: NodeFeatureBuilder instance (creates one if None).
            edge_feature_builder: EdgeFeatureBuilder instance (creates one if None).
            pressure_field: Pre-computed ConvergencePressureField (optional).
            dormancy_tracker: Pre-computed DormancyTracker (optional).
            convergence_points: Historical ConvergencePoint list for convergence labels.
            cutoff_ts: Only include nodes with timestamp < cutoff_ts.
            horizon_epochs: Normalisation denominator for eta labels.
            max_node_pairs: Cap total examples to avoid memory explosion.

        Returns:
            Populated OracleDataset ready for training.
        """
        if node_feature_builder is None:
            node_feature_builder = NodeFeatureBuilder()
        if edge_feature_builder is None:
            edge_feature_builder = EdgeFeatureBuilder()

        dummy_pressure = ConvergencePressureField()
        dummy_dormancy = DormancyTracker()
        pf = pressure_field if pressure_field is not None else dummy_pressure
        dt = dormancy_tracker if dormancy_tracker is not None else dummy_dormancy

        # Build convergence-point index: node_id → (converge_ts, paths_density)
        conv_index: Dict[str, Tuple[float, float]] = {}
        if convergence_points:
            for cp in convergence_points:
                for nid in cp.convergence_node_ids:
                    existing = conv_index.get(nid)
                    if existing is None or cp.convergence_timestamp < existing[0]:
                        conv_index[nid] = (cp.convergence_timestamp,
                                           getattr(cp, "paths_density", 0.0))

        # Filter nodes by cutoff
        node_ids = [
            nid for nid, node in graph.nodes.items()
            if (cutoff_ts is None or node.timestamp < cutoff_ts)
        ]
        if not node_ids:
            return cls([])

        # Pre-compute node feature vectors
        node_feat: Dict[str, np.ndarray] = {}
        for nid in node_ids:
            node = graph.nodes[nid]
            try:
                node_feat[nid] = node_feature_builder.build(node)
            except Exception:
                node_feat[nid] = np.zeros(NodeFeatureBuilder.TOTAL_DIM, dtype=np.float32)

        # Collect existing edges (positive pairs)
        existing_edge_keys: Set[Tuple[str, str]] = set()
        positive_pairs: List[Tuple[str, str]] = []
        node_id_set = set(node_ids)
        for (src, tgt), edge in graph.edges.items():
            if src in node_id_set and tgt in node_id_set and not edge.is_containment_edge:
                existing_edge_keys.add((src, tgt))
                positive_pairs.append((src, tgt))

        # Sample negative pairs (non-edges)
        rng = np.random.default_rng(42)
        n_neg = min(len(positive_pairs) * 3, max_node_pairs - len(positive_pairs))
        n_neg = max(0, n_neg)
        negative_pairs: List[Tuple[str, str]] = []
        if n_neg > 0 and len(node_ids) >= 2:
            attempts = 0
            while len(negative_pairs) < n_neg and attempts < n_neg * 10:
                attempts += 1
                i, j = rng.choice(len(node_ids), 2, replace=False)
                pair = (node_ids[i], node_ids[j])
                if pair not in existing_edge_keys:
                    negative_pairs.append(pair)

        all_pairs = positive_pairs + negative_pairs

        # Build examples
        examples: List[OracleExample] = []
        zero_edge = np.zeros(EdgeFeatureBuilder.DIM, dtype=np.float32)
        zero_comp = np.zeros(N_EXTENDED_COMPONENTS, dtype=np.float32)
        zero_topo = np.zeros(12, dtype=np.float32)

        for src_id, tgt_id in all_pairs:
            # Edge features
            edge = graph.edges.get((src_id, tgt_id))
            if edge is not None:
                try:
                    edge_attr = edge_feature_builder.build(edge)
                except Exception:
                    edge_attr = zero_edge.copy()
                label_exists = 1.0
                label_comps = np.array([
                    edge.semantic_similarity, edge.temporal_proximity,
                    edge.limitation_resolution, edge.citation_link,
                    edge.investment_correlation, edge.social_correlation,
                    edge.inhibitory_force,
                ], dtype=np.float32)
            else:
                edge_attr = zero_edge.copy()
                label_exists = 0.0
                label_comps = zero_comp.copy()

            # Topology vector
            try:
                # Build a lightweight LocalTopologyStats for this pair
                stats = LocalTopologyStats()
                k2 = graph.k_hop_neighborhood([src_id, tgt_id], k=2)
                sdc = PRESSURE_COMPUTER.sdi_cascade_stats(list(k2), graph)
                ecnt = PRESSURE_COMPUTER.edge_topology_counts(k2, graph)
                stats.mean_sdi_2hop     = sdc["mean_sdi"]
                stats.mean_cascade_2hop = sdc["mean_cascade"]
                stats.mean_zone_multiplier_2hop = sdc["mean_zmult"]
                stats.n_edges_2hop          = ecnt["n_edges"]
                stats.n_limitation_edges_2hop = ecnt["n_lim"]
                stats.n_inhibitory_edges_2hop  = ecnt["n_inh"]
                stats.void_pressure_src = PRESSURE_COMPUTER.node_pressure(src_id, pf)
                stats.void_pressure_tgt = PRESSURE_COMPUTER.node_pressure(tgt_id, pf)
                stats.pressure_gradient = PRESSURE_COMPUTER.pressure_gradient(
                    src_id, tgt_id, pf)
                stats.awakening_score = dt.get_awakening_score(src_id, tgt_id)
                # Use a ContextualPriorityHead helper to serialise to 12-dim vec
                topo_vec = ContextualPriorityHead._topo_stats_to_array(stats)
            except Exception:
                topo_vec = zero_topo.copy()

            # Convergence labels
            in_conv_src = conv_index.get(src_id)
            in_conv_tgt = conv_index.get(tgt_id)
            if in_conv_src is not None or in_conv_tgt is not None:
                label_convergence = 1.0
                # Eta: epochs until the earliest convergence involving either node
                conv_ts_vals = [v[0] for v in [in_conv_src, in_conv_tgt] if v is not None]
                node_ts = max(graph.nodes[src_id].timestamp,
                               graph.nodes[tgt_id].timestamp)
                secs_until = min(conv_ts_vals) - node_ts
                label_eta = float(np.clip(secs_until / (365.25 * 24 * 3600), 0, horizon_epochs))
            else:
                label_convergence = 0.0
                label_eta = 0.0

            examples.append(OracleExample(
                src_id=src_id,
                tgt_id=tgt_id,
                src_features=node_feat[src_id],
                tgt_features=node_feat[tgt_id],
                edge_attr=edge_attr,
                topology_vec=topo_vec,
                label_exists=label_exists,
                label_components=label_comps,
                label_convergence=label_convergence,
                label_convergence_eta=label_eta,
            ))

        return cls(examples)


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPORAL SPLIT TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class TemporalSplitTrainer:
    def __init__(self, model: Oracle1Model, loss_fn: Oracle1Loss,
                 feature_builder: NodeFeatureBuilder, edge_builder: EdgeFeatureBuilder):
        self.model   = model
        self.loss_fn = loss_fn
        self.fb      = feature_builder
        self.eb      = edge_builder

    def train_epoch(self, snapshot_pairs, delta_range_years=(0.5, 5.0)) -> Dict[str, float]:
        if not TORCH_AVAILABLE: return {"error": "torch_not_available"}
        total_loss = 0.0
        for snap_t, snap_td in snapshot_pairs:
            t_keys  = {(e.source_id, e.target_id) for e in snap_t.edges}
            td_keys = {(e.source_id, e.target_id) for e in snap_td.edges}
            new_edge_keys = td_keys - t_keys
            logger.info(f"[Training] +{len(snap_td.nodes)} nodes, +{len(new_edge_keys)} new edges")
        return {"mean_loss": total_loss / max(len(snapshot_pairs), 1)}



# ══════════════════════════════════════════════════════════════════════════════
#  OPEN PROBLEMS — honest technical assessment
# ══════════════════════════════════════════════════════════════════════════════
#
# 1. ZONE MULTIPLIER GROUND TRUTH
#    V6 assigns zone_multiplier by asking a LLM.  For Oracle-1's training,
#    we need MEASURED multipliers from historical data.
#    Proposed metric: citation_velocity_ratio = citations_per_year_in_zone /
#                                               baseline_citations_per_year
#    A zone where papers are cited 3× faster than baseline → multiplier = 3.0.
#    This makes zone_multiplier an EMPIRICAL measurement, not an annotation.
#
# 2. STRUCTURAL SECTION FEEDBACK LOOP STABILITY
#    Rewriting node features after each GNN layer can create instability:
#    if layer K produces large structural changes, layer K+1 sees a very
#    different feature distribution.
#    Solution: LayerNorm on the structural section before writing back,
#    plus a learnable decay factor (alpha) that blends old and new values.
#
# 3. EDGE COMPONENT INDEPENDENCE ASSUMPTION
#    The 6 components are treated as independent.  In reality they are
#    correlated: high citation_link usually implies high semantic_similarity.
#    This correlation may cause the model to learn spurious patterns.
#    Solution: factored representation — decompose edge_attr into orthogonal
#    components via a learned Cholesky decomposition of the covariance matrix.
#
# 4. CONTAINMENT EDGE GRADIENT FLOW
#    Containment edges (zone → member) apply multiplicative gating.
#    Multiplication creates gradient pathways that can explode if
#    zone_multiplier >> 1 at initialisation.
#    Solution: initialise zone_multiplier head to output 1.0 + small ε
#    and clip gradients through containment edges separately.
#
# 5. QUERY-TO-START-NODE MATCHING
#    Currently: cosine similarity between query embedding and node embeddings.
#    Problem: a query like "what will replace lithography?" has no direct
#    embedding match to "DNA origami" even if DNA origami is the answer.
#    Solution: use a QUERY EXPANSION STEP where the LLM first translates
#    the query into functional language (V7's AbstractionEngine), then
#    match against functional embeddings instead of text embeddings.
#
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  ORACLE-1 EXTENDED (master orchestrator)
# ══════════════════════════════════════════════════════════════════════════════

class Oracle1Extended:
    def __init__(self, base_model: Oracle1Model, graph: RuntimeGraph):
        self.model      = base_model
        self.graph      = graph
        self.latent_dim = getattr(base_model, "latent_dim", 512)

        self.physical_encoder = PhysicalSubstrateEncoder()
        self.pressure_field   = ConvergencePressureField()
        self.dormancy_tracker = DormancyTracker()
        # Fix 4: reuse the priority_head that is already registered inside
        # Oracle1Model (so its parameters are trained via backprop).
        # If the model stub doesn't carry one, fall back to a standalone instance.
        self.priority_head = getattr(base_model, "priority_head",
                                     ContextualPriorityHead(self.latent_dim))
        self.phantom_gen      = PhantomNodeGenerator()
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

    def update_epoch(self, timestamp: float) -> Dict[str, Any]:
        self.pressure_field.compute(self.graph)
        awakened   = self.dormancy_tracker.update_epoch(self.graph)
        new_phantoms = self.phantom_gen.scan_for_voids(
            node_latents=None, graph=self.graph, pressure_field=self.pressure_field)
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
                 phantom_gen: PhantomNodeGenerator, llm_client=None):
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
        self.dual_head = DualObjectiveHead(latent_dim=CONFIG["model_latent_dim"])

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


# ══════════════════════════════════════════════════════════════════════════════
#  FULL SYSTEM FACTORY
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# INLINED MODULE: oracle1_significance.py
# Significance Gradient + Concept Distillation Engine
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════
#  1. EPISTEMIC GRADIENT — шкала значимости
# ══════════════════════════════════════════════════════════════════

class EpistemicLevel(Enum):
    """
    Уровни эпистемической значимости.
    Значение float — вес в расчётах influence_score.

    Принцип: чем выше уровень, тем МЕНЬШЕ источников нужно для
    достижения статуса "proven". Форум требует 1000 постов чтобы
    сравняться с одной статьёй Nature.
    """
    PROVEN_DISCOVERY    = 1.00  # воспроизведено независимо, реплицировано
    PEER_REVIEWED       = 0.85  # рецензированный журнал
    PREPRINT            = 0.70  # arXiv, bioRxiv
    PATENT              = 0.55  # патент (практическое применение)
    TEXTBOOK_MENTION    = 0.40  # учебник (устоявшееся знание)
    NEWS_REPORT         = 0.30  # СМИ (популяризация)
    FORUM_DISCUSSION    = 0.20  # форум, блог (незащищённое мнение)
    FOLKLORE_FICTION    = 0.10  # фантастика, мифология, сказки
    SPECULATION         = 0.05  # домысел, анонимный пост


# Маппинг source_type → EpistemicLevel
SOURCE_TYPE_TO_LEVEL: Dict[str, EpistemicLevel] = {
    "nature":       EpistemicLevel.PROVEN_DISCOVERY,
    "science":      EpistemicLevel.PROVEN_DISCOVERY,
    "cell":         EpistemicLevel.PROVEN_DISCOVERY,
    "peer_reviewed":EpistemicLevel.PEER_REVIEWED,
    "journal":      EpistemicLevel.PEER_REVIEWED,
    "arxiv":        EpistemicLevel.PREPRINT,
    "preprint":     EpistemicLevel.PREPRINT,
    "biorxiv":      EpistemicLevel.PREPRINT,
    "patent":       EpistemicLevel.PATENT,
    "conference":   EpistemicLevel.PEER_REVIEWED,
    "book":         EpistemicLevel.TEXTBOOK_MENTION,
    "textbook":     EpistemicLevel.TEXTBOOK_MENTION,
    "report":       EpistemicLevel.NEWS_REPORT,
    "news":         EpistemicLevel.NEWS_REPORT,
    "blog":         EpistemicLevel.FORUM_DISCUSSION,
    "forum":        EpistemicLevel.FORUM_DISCUSSION,
    "reddit":       EpistemicLevel.FORUM_DISCUSSION,
    "wikipedia":    EpistemicLevel.NEWS_REPORT,
    "fiction":      EpistemicLevel.FOLKLORE_FICTION,
    "mythology":    EpistemicLevel.FOLKLORE_FICTION,
    "folklore":     EpistemicLevel.FOLKLORE_FICTION,
    "fairytale":    EpistemicLevel.FOLKLORE_FICTION,
    "speculation":  EpistemicLevel.SPECULATION,
    "unknown":      EpistemicLevel.SPECULATION,
}


def get_epistemic_level(source_type: str) -> EpistemicLevel:
    key = source_type.lower().strip()
    # Частичное совпадение (reddit → forum_discussion)
    for pattern, level in SOURCE_TYPE_TO_LEVEL.items():
        if pattern in key:
            return level
    return EpistemicLevel.SPECULATION


@dataclass
class EpistemicWeight:
    """
    Вес одного наблюдения в шкале значимости.
    Учитывает: уровень источника, дату, количество цитирований.
    """
    level:          EpistemicLevel
    raw_trust:      float       # 0.0–1.0 из source_trust таблицы
    timestamp:      float       # Unix timestamp источника
    citation_count: int = 0     # сколько раз этот источник процитирован
    replication_count: int = 0  # независимые репликации (для proven)

    @property
    def effective_weight(self) -> float:
        """
        Итоговый вес наблюдения:
          base = level.value × raw_trust
          citation_bonus = log(1 + citations) / log(1000) × 0.15
          replication_bonus = min(replications / 3, 1.0) × 0.20
        """
        base = self.level.value * self.raw_trust
        cit  = math.log1p(self.citation_count) / math.log1p(1000) * 0.15
        rep  = min(self.replication_count / 3.0, 1.0) * 0.20
        return float(min(1.0, base + cit + rep))


# ══════════════════════════════════════════════════════════════════
#  2. INFLUENCE TIMELINE
#  История подкреплений canonical узла с датами и весами
# ══════════════════════════════════════════════════════════════════

@dataclass
class InfluencePoint:
    """Один момент подкрепления canonical узла."""
    timestamp:      float
    source:         str
    source_type:    str
    epistemic_level: EpistemicLevel
    weight:         float
    delta_fields:   Dict[str, Any] = field(default_factory=dict)
    distill_type:   str = "reinforcement"  # reinforcement / extension / reinterpretation


@dataclass
class InfluenceTimeline:
    """
    Временная шкала влияний на один canonical узел.

    Используется для измерения:
      - СКОРОСТИ НАРАСТАНИЯ (velocity)    — приближение прорыва
      - ДИВЕРСИФИКАЦИИ ИСТОЧНИКОВ         — кросс-доменная значимость
      - ПИКОВ АКТИВНОСТИ                  — «горячие периоды»
      - НАКОПЛЕННОЙ EPISTEMIC MASS        — суммарный вес подкреплений
    """
    canonical_id:    str
    points:          List[InfluencePoint] = field(default_factory=list)

    def add(self, point: InfluencePoint) -> None:
        self.points.append(point)
        self.points.sort(key=lambda p: p.timestamp)

    @property
    def epistemic_mass(self) -> float:
        """
        Накопленная эпистемическая масса.
        Используем убывающую сумму: поздние источники весомее,
        но старые не обнуляются (Einstein 1917 не исчезает).

        mass = Σ weight_i × decay(now - t_i)
        decay = 0.5 если источнику > 50 лет, иначе 1.0
        """
        if not self.points:
            return 0.0
        now = time.time()
        total = 0.0
        for p in self.points:
            age_years = (now - p.timestamp) / (365.25 * 24 * 3600)
            decay = 0.5 if age_years > 50 else 1.0
            total += p.weight * decay
        return float(min(10.0, total))  # cap at 10 для совместимости с score-шкалой

    @property
    def velocity(self) -> float:
        """
        Скорость нарастания за последние 5 лет (points per year × avg weight).
        Высокая velocity = признак приближающегося прорыва.
        """
        if len(self.points) < 2:
            return 0.0
        now = time.time()
        cutoff = now - 5 * 365.25 * 24 * 3600
        recent = [p for p in self.points if p.timestamp >= cutoff]
        if not recent:
            return 0.0
        avg_w  = float(np.mean([p.weight for p in recent]))
        return len(recent) * avg_w

    @property
    def source_diversity(self) -> float:
        """
        Разнообразие типов источников (0–1).
        Много разных типов = высокая значимость.
        """
        if not self.points:
            return 0.0
        types = set(p.source_type for p in self.points)
        # Нормализуем по максимальному числу возможных типов
        return min(1.0, len(types) / len(SOURCE_TYPE_TO_LEVEL))

    @property
    def peak_epistemic_year(self) -> Optional[int]:
        """Год с наибольшей суммарной эпистемической активностью."""
        if not self.points:
            return None
        year_weight: Dict[int, float] = defaultdict(float)
        for p in self.points:
            year = int(time.gmtime(p.timestamp).tm_year)
            year_weight[year] += p.weight
        return max(year_weight, key=year_weight.get)

    def proven_status(self) -> bool:
        """
        Достиг ли узел статуса proven_discovery?
        Требования:
          - хотя бы 1 PROVEN_DISCOVERY источник, ИЛИ
          - хотя бы 2 PEER_REVIEWED источника + epistemic_mass >= 2.0
        """
        proven_count = sum(
            1 for p in self.points
            if p.epistemic_level == EpistemicLevel.PROVEN_DISCOVERY
        )
        peer_count = sum(
            1 for p in self.points
            if p.epistemic_level in (EpistemicLevel.PROVEN_DISCOVERY,
                                     EpistemicLevel.PEER_REVIEWED)
        )
        return (proven_count >= 1 or
                (peer_count >= 2 and self.epistemic_mass >= 2.0))

    def to_summary(self) -> Dict[str, Any]:
        return {
            "canonical_id":       self.canonical_id,
            "total_points":       len(self.points),
            "epistemic_mass":     round(self.epistemic_mass, 3),
            "velocity":           round(self.velocity, 3),
            "source_diversity":   round(self.source_diversity, 3),
            "proven":             self.proven_status(),
            "peak_year":          self.peak_epistemic_year,
            "source_types":       list({p.source_type for p in self.points}),
        }


# ══════════════════════════════════════════════════════════════════
#  3. DISTILLATION ENGINE
#  LLM извлекает из источника: подтверждение / расширение / новое
# ══════════════════════════════════════════════════════════════════

class DistillationType(Enum):
    REINFORCEMENT     = "reinforcement"       # подтверждает canonical
    EXTENSION         = "extension"           # новый аспект того же
    REINTERPRETATION  = "reinterpretation"    # другой взгляд на то же
    NEW_CONCEPT       = "new_concept"         # действительно новая идея
    ABSTRACT_CONCEPT  = "abstract_concept"    # косвенная концепция (сказки!)


@dataclass
class DistillationResult:
    """
    Результат анализа одного источника относительно canonical узла.
    """
    distill_type:       DistillationType
    canonical_id:       str             # к какому узлу относится
    source:             str
    source_type:        str
    timestamp:          float
    epistemic_level:    EpistemicLevel
    weight:             EpistemicWeight

    # Для REINFORCEMENT: поля для усиления canonical узла
    reinforcement_delta: Dict[str, float] = field(default_factory=dict)

    # Для EXTENSION / REINTERPRETATION / NEW_CONCEPT / ABSTRACT_CONCEPT:
    # описание нового/мутантного узла
    new_node_text:       str = ""
    new_node_domain:     str = ""
    new_node_entity_type: str = ""
    new_node_concepts:   List[str] = field(default_factory=list)  # абстрактные концепции
    new_node_confidence: float = 0.0

    # Связи нового узла с canonical
    relation_type:       str = "extends"  # extends / reinterprets / derived_from
    relation_strength:   float = 0.5


DISTILLATION_PROMPT = """\
You are an epistemologist analyzing a source document relative to a known canonical concept.

CANONICAL CONCEPT: "{canonical_text}"
CANONICAL DOMAIN: {canonical_domain}
SOURCE TYPE: {source_type}
EPISTEMIC LEVEL: {epistemic_level}
SOURCE TEXT:
{source_text}

TASK: Determine what this source contributes relative to the canonical concept.
Classify as ONE of:
  REINFORCEMENT     — source confirms/strengthens the canonical (no new node)
  EXTENSION         — source adds a new verified aspect (creates child node)
  REINTERPRETATION  — source offers different interpretation (creates sibling node)
  NEW_CONCEPT       — source contains genuinely new idea (creates new node)
  ABSTRACT_CONCEPT  — source expresses same deep concept in different form
                      (e.g. fairy tale mirror → "remote visual transmission device")
                      Extract the ABSTRACT FUNCTIONAL CONCEPT, not the surface form.

For ABSTRACT_CONCEPT: ignore the fictional/mythological surface. Extract:
  - core function (what does it DO?)
  - key properties (portable? on-demand? scalable?)
  - the abstract concept name

OUTPUT ONLY valid JSON:
{{
  "distill_type": "<REINFORCEMENT|EXTENSION|REINTERPRETATION|NEW_CONCEPT|ABSTRACT_CONCEPT>",
  "reasoning": "<one sentence>",
  "reinforcement_delta": {{
    "scientific_score_delta": <-2.0 to +2.0 or 0>,
    "social_score_delta": <-2.0 to +2.0 or 0>,
    "maturity_score_delta": <-1.0 to +1.0 or 0>
  }},
  "new_node": {{
    "text": "<functional description if not REINFORCEMENT, else empty>",
    "domain": "<domain>",
    "entity_type": "<type>",
    "abstract_concepts": ["<concept1>", "<concept2>"],
    "confidence": <0.0-1.0>
  }},
  "relation_type": "<extends|reinterprets|derived_from|precursor_of>",
  "relation_strength": <0.0-1.0>
}}
"""


class ConceptDistiller:
    """
    Анализирует каждый новый источник относительно существующих canonical узлов.

    Workflow:
      1. Найти кандидатные canonical узлы (семантическая близость)
      2. Для каждого кандидата → LLM distillation
      3. Вернуть список DistillationResult
         - REINFORCEMENT → обновить InfluenceTimeline, не создавать узел
         - остальные → создать мутантный/новый узел + ребро

    Ключевая идея:
      "Форум Reddit 2019" не становится узлом "Stimulated Emission".
      Он либо УСИЛИВАЕТ "Stimulated Emission" (добавляет вес в timeline),
      либо СОЗДАЁТ новый узел "Популярная дискуссия о лазерах 2019"
      с ребром extension → "Stimulated Emission".
    """

    SIMILARITY_THRESHOLD = 0.55   # минимальная близость для проверки
    MAX_CANDIDATES       = 5      # сколько canonical узлов проверять

    def __init__(self, llm_client=None):
        self.llm = llm_client
        # canonical_id → InfluenceTimeline
        self.timelines: Dict[str, InfluenceTimeline] = {}

    # ── Основной API ────────────────────────────────────────────

    def process_source(
        self,
        source_text:  str,
        source_type:  str,
        source_name:  str,
        timestamp:    float,
        graph_nodes:  Dict[str, Any],   # {id: KnowledgeNode}
        embedder=None,
    ) -> List[DistillationResult]:
        """
        Обработать один источник.
        Возвращает список DistillationResult для каждого релевантного canonical узла.
        """
        level   = get_epistemic_level(source_type)
        # Доверие из epistemic level (не из source_trust таблицы — единая шкала)
        trust   = level.value

        candidates = self._find_candidates(source_text, graph_nodes, embedder)
        results: List[DistillationResult] = []

        for canonical_id, sim_score in candidates:
            node = graph_nodes.get(canonical_id)
            if node is None:
                continue

            if self.llm is not None:
                result = self._llm_distill(
                    source_text=source_text,
                    source_type=source_type,
                    source_name=source_name,
                    timestamp=timestamp,
                    canonical_node=node,
                    level=level,
                    trust=trust,
                    sim_score=sim_score,
                )
            else:
                # Без LLM: высокая схожесть → REINFORCEMENT
                result = self._heuristic_distill(
                    source_text=source_text,
                    source_type=source_type,
                    source_name=source_name,
                    timestamp=timestamp,
                    canonical_node=node,
                    level=level,
                    trust=trust,
                    sim_score=sim_score,
                )

            if result:
                # Обновить timeline для canonical узла
                self._update_timeline(canonical_id, result)
                results.append(result)

        return results

    def get_timeline(self, canonical_id: str) -> Optional[InfluenceTimeline]:
        return self.timelines.get(canonical_id)

    def get_or_create_timeline(self, canonical_id: str) -> InfluenceTimeline:
        if canonical_id not in self.timelines:
            self.timelines[canonical_id] = InfluenceTimeline(canonical_id=canonical_id)
        return self.timelines[canonical_id]

    def top_by_velocity(self, n: int = 10) -> List[Tuple[str, float]]:
        """Топ N узлов по скорости нарастания — сигнал приближения прорыва."""
        scored = [
            (cid, tl.velocity)
            for cid, tl in self.timelines.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def top_by_epistemic_mass(self, n: int = 10) -> List[Tuple[str, float]]:
        """Топ N по накопленной эпистемической массе."""
        scored = [
            (cid, tl.epistemic_mass)
            for cid, tl in self.timelines.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def necessity_from_diversity(self, canonical_id: str) -> float:
        """
        Индекс необходимости из InfluenceTimeline.
        Комбинирует epistemic_mass, source_diversity, velocity.
        """
        tl = self.timelines.get(canonical_id)
        if not tl:
            return 0.0
        return float(
            tl.epistemic_mass * 0.4 +
            tl.source_diversity * 5.0 * 0.3 +
            tl.velocity * 0.3
        )

    # ── Внутренние методы ───────────────────────────────────────

    def _find_candidates(
        self,
        source_text: str,
        graph_nodes: Dict[str, Any],
        embedder=None,
    ) -> List[Tuple[str, float]]:
        """
        Найти canonical узлы, близкие к тексту источника.
        Возвращает [(node_id, similarity)], отсортированные по убыванию.
        """
        if not graph_nodes:
            return []

        # Получаем эмбеддинг источника
        src_emb = self._embed(source_text, embedder)
        if src_emb is None:
            # Fallback: keyword overlap
            return self._keyword_candidates(source_text, graph_nodes)

        candidates = []
        for nid, node in graph_nodes.items():
            node_emb = getattr(node, "embedding", None)
            if node_emb is None:
                continue
            # Сравниваем только первые 384 dim (текстовый блок)
            ne = np.array(node_emb[:384], dtype=np.float32)
            se = np.array(src_emb[:384], dtype=np.float32)
            ne_norm = np.linalg.norm(ne)
            se_norm = np.linalg.norm(se)
            if ne_norm < 1e-8 or se_norm < 1e-8:
                continue
            sim = float(np.dot(ne / ne_norm, se / se_norm))
            if sim >= self.SIMILARITY_THRESHOLD:
                candidates.append((nid, sim))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.MAX_CANDIDATES]

    def _keyword_candidates(
        self,
        source_text: str,
        graph_nodes: Dict[str, Any],
    ) -> List[Tuple[str, float]]:
        """Keyword-based fallback без эмбеддингов."""
        src_words = set(re.findall(r'[a-zа-яё]{4,}', source_text.lower()))
        scored = []
        for nid, node in graph_nodes.items():
            node_text = getattr(node, "text", "") + " " + getattr(node, "full_text", "")
            node_words = set(re.findall(r'[a-zа-яё]{4,}', node_text.lower()))
            if not node_words:
                continue
            overlap = len(src_words & node_words) / max(len(src_words | node_words), 1)
            if overlap >= 0.1:
                scored.append((nid, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.MAX_CANDIDATES]

    def _embed(self, text: str, embedder) -> Optional[np.ndarray]:
        if embedder is None:
            return None
        try:
            return np.array(embedder.encode(text[:512]), dtype=np.float32)
        except Exception as e:
            logger.debug(f"[Distiller] embed failed: {e}")
            return None

    def _llm_distill(
        self,
        source_text:    str,
        source_type:    str,
        source_name:    str,
        timestamp:      float,
        canonical_node: Any,
        level:          EpistemicLevel,
        trust:          float,
        sim_score:      float,
    ) -> Optional[DistillationResult]:
        """LLM-based distillation."""
        prompt = DISTILLATION_PROMPT.format(
            canonical_text=getattr(canonical_node, "text", ""),
            canonical_domain=getattr(canonical_node, "domain", ""),
            source_type=source_type,
            epistemic_level=level.name,
            source_text=source_text[:1500],
        )
        try:
            raw  = self.llm.complete(prompt)
            data = _parse_json_safe(raw)
        except Exception as e:
            logger.warning(f"[Distiller] LLM error: {e}")
            return None

        return self._build_result(
            data=data,
            source_name=source_name,
            source_type=source_type,
            timestamp=timestamp,
            canonical_id=canonical_node.id,
            level=level,
            trust=trust,
        )

    def _heuristic_distill(
        self,
        source_text:    str,
        source_type:    str,
        source_name:    str,
        timestamp:      float,
        canonical_node: Any,
        level:          EpistemicLevel,
        trust:          float,
        sim_score:      float,
    ) -> Optional[DistillationResult]:
        """
        Без LLM: эвристика по epistemic level + similarity.
        - Высокий level + высокая sim → REINFORCEMENT
        - Низкий level + средняя sim → EXTENSION (осторожно)
        - Folklore/fiction → ABSTRACT_CONCEPT если sim < 0.75
        """
        if level in (EpistemicLevel.FOLKLORE_FICTION, EpistemicLevel.SPECULATION):
            dtype = DistillationType.ABSTRACT_CONCEPT
        elif sim_score >= 0.80 and level.value >= 0.55:
            dtype = DistillationType.REINFORCEMENT
        elif sim_score >= 0.65:
            dtype = DistillationType.EXTENSION
        else:
            dtype = DistillationType.REINFORCEMENT

        w = EpistemicWeight(
            level=level, raw_trust=trust,
            timestamp=timestamp,
        )
        result = DistillationResult(
            distill_type=dtype,
            canonical_id=canonical_node.id,
            source=source_name,
            source_type=source_type,
            timestamp=timestamp,
            epistemic_level=level,
            weight=w,
            reinforcement_delta={"scientific_score_delta": level.value * 0.5},
            new_node_text=(f"[{source_type}] aspect of: "
                           f"{getattr(canonical_node, 'text', '')[:50]}")
                          if dtype != DistillationType.REINFORCEMENT else "",
            new_node_domain=getattr(canonical_node, "domain", ""),
            new_node_entity_type="cultural_phantom"
                                  if dtype == DistillationType.ABSTRACT_CONCEPT
                                  else "research_frontier",
            new_node_confidence=sim_score * level.value,
        )
        return result

    def _build_result(
        self,
        data:         Dict,
        source_name:  str,
        source_type:  str,
        timestamp:    float,
        canonical_id: str,
        level:        EpistemicLevel,
        trust:        float,
    ) -> DistillationResult:
        dtype_str = data.get("distill_type", "REINFORCEMENT").upper()
        try:
            dtype = DistillationType[dtype_str]
        except KeyError:
            dtype = DistillationType.REINFORCEMENT

        w = EpistemicWeight(
            level=level, raw_trust=trust, timestamp=timestamp,
        )

        new_node_data = data.get("new_node", {})
        return DistillationResult(
            distill_type=dtype,
            canonical_id=canonical_id,
            source=source_name,
            source_type=source_type,
            timestamp=timestamp,
            epistemic_level=level,
            weight=w,
            reinforcement_delta=data.get("reinforcement_delta", {}),
            new_node_text=new_node_data.get("text", ""),
            new_node_domain=new_node_data.get("domain", ""),
            new_node_entity_type=new_node_data.get("entity_type", "research_frontier"),
            new_node_concepts=new_node_data.get("abstract_concepts", []),
            new_node_confidence=float(new_node_data.get("confidence", 0.5)),
            relation_type=data.get("relation_type", "extends"),
            relation_strength=float(data.get("relation_strength", 0.5)),
        )

    def _update_timeline(
        self,
        canonical_id: str,
        result:       DistillationResult,
    ) -> None:
        tl = self.get_or_create_timeline(canonical_id)
        point = InfluencePoint(
            timestamp=result.timestamp,
            source=result.source,
            source_type=result.source_type,
            epistemic_level=result.epistemic_level,
            weight=result.weight.effective_weight,
            delta_fields=result.reinforcement_delta,
            distill_type=result.distill_type.value,
        )
        tl.add(point)


# ══════════════════════════════════════════════════════════════════
#  4. SIGNIFICANCE PROCESSOR
#  Встраивается в IngestionPipeline между PassC и load_result_into_graph
# ══════════════════════════════════════════════════════════════════

@dataclass
class ProcessorResult:
    """Результат обработки одного IngestionResult через SignificanceProcessor."""
    reinforcements: int = 0        # сколько источников стали REINFORCEMENT
    mutant_nodes:   int = 0        # сколько мутантных узлов создано
    abstract_nodes: int = 0        # сколько абстрактных концептов создано
    new_nodes:      List[Any] = field(default_factory=list)   # KnowledgeNode
    new_edges:      List[Any] = field(default_factory=list)   # KnowledgeEdge
    timeline_updates: Dict[str, Dict] = field(default_factory=dict)


class SignificanceProcessor:
    """
    Главный процессор.

    Алгоритм:
      1. Для каждого узла из IngestionResult:
           a. Найти canonical узлы графа (близкие)
           b. Запустить ConceptDistiller
           c. REINFORCEMENT → обновить epistemic_mass canonical узла
              NEW_CONCEPT/EXTENSION/etc → создать мутантный узел + ребро

      2. Мутантные узлы получают:
           - epistemic_level из источника
           - связь с canonical через KnowledgeEdge
           - reduced strategic_value (не "вершина айсберга")

      3. Canonical узел обновляется:
           - epistemic_mass → scientific_score (нормализованный)
           - proven_status → entity_type может стать "breakthrough_node"
           - velocity → forecast_score

    ВАЖНО: SignificanceProcessor не удаляет узлы из IngestionResult.
    Он добавляет к ним слой семантики и принимает решение:
    "этот узел — дубль canonical" или "это новое".
    """

    PROVEN_ENTITY_TYPE     = "breakthrough_node"
    MIN_MASS_FOR_PROVEN    = 3.0
    REINFORCEMENT_FRACTION = 0.5   # доля источников, идущих в reinforcement

    def __init__(self,
                 distiller: Optional[ConceptDistiller] = None,
                 embedder=None):
        self.distiller = distiller or ConceptDistiller()
        self.embedder  = embedder

    def process(
        self,
        ingestion_result: Any,   # IngestionResult
        graph:            Any,   # RuntimeGraph
        llm_client=None,
    ) -> ProcessorResult:
        """
        Обработать IngestionResult:
          - усилить canonical узлы через timelines
          - создать мутантные узлы для реальных новшеств
          - обновить epistemic_mass canonical узлов
        """
        result = ProcessorResult()
        if llm_client is not None:
            self.distiller.llm = llm_client

        source      = getattr(ingestion_result, "source", "unknown")
        # Определяем source_type из первого узла или из source URL
        source_type = self._infer_source_type(source, ingestion_result)

        for node in getattr(ingestion_result, "nodes", []):
            node_text    = getattr(node, "text", "")
            node_domain  = getattr(node, "domain", "")
            node_ts      = getattr(node, "timestamp", time.time())

            distill_results = self.distiller.process_source(
                source_text=node_text + " " + getattr(node, "full_text", ""),
                source_type=source_type,
                source_name=source,
                timestamp=node_ts,
                graph_nodes=dict(graph.nodes) if hasattr(graph, "nodes") else {},
                embedder=self.embedder,
            )

            for dr in distill_results:
                if dr.distill_type == DistillationType.REINFORCEMENT:
                    # Обновить canonical узел напрямую
                    self._apply_reinforcement(dr, graph)
                    result.reinforcements += 1

                elif dr.distill_type in (
                    DistillationType.EXTENSION,
                    DistillationType.REINTERPRETATION,
                    DistillationType.NEW_CONCEPT,
                    DistillationType.ABSTRACT_CONCEPT,
                ):
                    # Создать мутантный узел
                    mutant, edge = self._create_mutant_node(dr, node, graph)
                    if mutant and edge:
                        result.new_nodes.append(mutant)
                        result.new_edges.append(edge)
                        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
                            result.abstract_nodes += 1
                        else:
                            result.mutant_nodes += 1

        # Обновить epistemic_mass → score для всех затронутых canonical узлов
        affected_ids = set(dr.canonical_id for dr in self._collect_all_dr(result))
        for cid in affected_ids:
            self._sync_epistemic_to_node(cid, graph)
            tl = self.distiller.get_timeline(cid)
            if tl:
                result.timeline_updates[cid] = tl.to_summary()

        logger.info(
            f"[SignificanceProcessor] source='{source}' type='{source_type}' "
            f"reinforcements={result.reinforcements} "
            f"mutants={result.mutant_nodes} abstract={result.abstract_nodes}"
        )
        return result

    # ── Создание мутантного узла ─────────────────────────────────

    def _create_mutant_node(
        self,
        dr:     DistillationResult,
        source_node: Any,
        graph:  Any,
    ) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Создать мутантный KnowledgeNode + KnowledgeEdge к canonical.
        KnowledgeNode и KnowledgeEdge доступны из основного модуля oracle1.
        """
        # KnowledgeNode / KnowledgeEdge are defined in the main oracle1 module.
        # In the combined single-file build they are already in scope.
        # For standalone use of this class, import them explicitly:
        try:
            _KnowledgeNode = KnowledgeNode  # noqa: F821 (defined in oracle1 scope)
            _KnowledgeEdge = KnowledgeEdge  # noqa: F821
        except NameError:
            try:
                from oracle1 import KnowledgeNode as _KnowledgeNode, KnowledgeEdge as _KnowledgeEdge
            except ImportError:
                logger.warning("[SignificanceProcessor] KnowledgeNode/KnowledgeEdge not importable — skip")
                return None, None

        if not dr.new_node_text:
            return None, None

        # Эпистемический вес определяет начальные скоры мутанта
        ew = dr.weight.effective_weight
        level_val = dr.epistemic_level.value

        # Для абстрактного концепта создаём особый узел
        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
            text = self._abstract_concept_text(dr, source_node)
            etype = "cultural_phantom"
        else:
            text = dr.new_node_text
            etype = dr.new_node_entity_type or "research_frontier"

        mutant_id = "mut_" + hashlib.md5(
            f"{dr.canonical_id}::{text}::{dr.source}".encode()
        ).hexdigest()[:10]

        mutant = _KnowledgeNode(
            id=mutant_id,
            text=text,
            full_text=getattr(source_node, "full_text", ""),
            node_type="mutant",
            domain=dr.new_node_domain or getattr(source_node, "domain", "unknown"),
            entity_type=etype,
            provenance=dr.source,
            timestamp=dr.timestamp,
            # Скоры масштабированы по epistemic level
            scientific_score = float(
                getattr(source_node, "scientific_score", 5.0) * level_val
            ),
            social_score     = float(
                getattr(source_node, "social_score", 0.0) * level_val
            ),
            maturity_score   = float(level_val * 5.0),
            strategic_value  = float(
                getattr(source_node, "strategic_value", 5.0) * ew
            ),
            # Epistemic metadata fields (available in merged build)
            epistemic_level  = dr.epistemic_level.name,
            epistemic_mass   = dr.weight.effective_weight,
            influence_velocity = 0.0,
            source_type      = dr.source_type,
        )
        # Сохраняем метаданные
        mutant.__dict__["epistemic_level"]    = dr.epistemic_level.name
        mutant.__dict__["epistemic_weight"]   = ew
        mutant.__dict__["distill_type"]       = dr.distill_type.value
        mutant.__dict__["abstract_concepts"]  = dr.new_node_concepts
        mutant.__dict__["parent_canonical_id"] = dr.canonical_id

        # Ребро mutant → canonical
        edge = _KnowledgeEdge(
            id=f"sig_{mutant_id[:8]}_{dr.canonical_id[:6]}",
            source_id=mutant_id,
            target_id=dr.canonical_id,
            relationship_type=dr.relation_type,
            semantic_similarity=float(dr.relation_strength),
            limitation_resolution=float(
                dr.relation_strength if dr.distill_type == DistillationType.EXTENSION
                else 0.0
            ),
            citation_link=float(level_val),
            confidence=float(dr.new_node_confidence),
            timestamp=dr.timestamp,
        )
        edge.compute_total_weight()
        return mutant, edge

    def _abstract_concept_text(
        self,
        dr: DistillationResult,
        source_node: Any,
    ) -> str:
        """Генерировать текст для абстрактного концепта (из сказок и т.п.)."""
        if dr.new_node_concepts:
            return "Abstract: " + "; ".join(dr.new_node_concepts[:3])
        # Fallback: из текста источника
        base = getattr(source_node, "text", "unknown source")
        return f"Abstract concept from [{dr.source_type}]: {base[:60]}"

    # ── Обновление canonical узла ────────────────────────────────

    def _apply_reinforcement(
        self,
        dr:    DistillationResult,
        graph: Any,
    ) -> None:
        """
        Применить REINFORCEMENT: обновить поля canonical узла
        без создания нового узла.
        """
        node = graph.get_node(dr.canonical_id) if hasattr(graph, "get_node") else None
        if node is None:
            return
        delta = dr.reinforcement_delta
        for field_name, delta_val in delta.items():
            # Убираем суффикс "_delta"
            base = field_name.replace("_delta", "")
            current = getattr(node, base, None)
            if current is not None and isinstance(current, (int, float)):
                new_val = float(np.clip(current + delta_val, 0.0, 10.0))
                setattr(node, base, new_val)

        # Добавляем epistemic_weight в __dict__ для последующей синхронизации
        prev = node.__dict__.get("reinforcement_weights", [])
        prev.append(dr.weight.effective_weight)
        node.__dict__["reinforcement_weights"] = prev

    def _sync_epistemic_to_node(
        self,
        canonical_id: str,
        graph:        Any,
    ) -> None:
        """
        Синхронизировать InfluenceTimeline → поля KnowledgeNode:
          epistemic_mass → scientific_score (нормализованный) + native field
          velocity       → forecast_score + influence_velocity
          proven_status  → entity_type апгрейд
          source_diversity → social_score + source_diversity field
        """
        node = graph.get_node(canonical_id) if hasattr(graph, "get_node") else None
        tl   = self.distiller.get_timeline(canonical_id)
        if node is None or tl is None:
            return

        # Нормализуем epistemic_mass в [0, 10]
        new_sci = float(np.clip(tl.epistemic_mass, 0.0, 10.0))
        # Применяем только если выше текущего (не понижаем)
        if new_sci > node.scientific_score:
            node.scientific_score = new_sci

        # Sync to native epistemic fields (new in merged build)
        if hasattr(node, "epistemic_mass"):
            node.epistemic_mass = float(np.clip(tl.epistemic_mass, 0.0, 10.0))

        # Velocity → forecast_score + influence_velocity
        vel_normalized = float(np.clip(tl.velocity / 5.0, 0.0, 10.0))
        if vel_normalized > node.forecast_score:
            node.forecast_score = vel_normalized
        if hasattr(node, "influence_velocity"):
            node.influence_velocity = tl.velocity

        # Source diversity → social_score + native field
        div_score = float(tl.source_diversity * 10.0)
        if div_score > node.social_score:
            node.social_score = div_score
        if hasattr(node, "source_diversity"):
            node.source_diversity = tl.source_diversity

        # Proven status → апгрейд entity_type
        if (tl.proven_status() and
                node.entity_type not in (self.PROVEN_ENTITY_TYPE, "temporal_zone")):
            old_type = node.entity_type
            node.entity_type = self.PROVEN_ENTITY_TYPE
            logger.info(
                f"[SignificanceProcessor] '{canonical_id}' upgraded: "
                f"{old_type} → {self.PROVEN_ENTITY_TYPE} "
                f"(mass={tl.epistemic_mass:.2f})"
            )

        # Сохраняем метаданные timeline в node.__dict__
        node.__dict__["epistemic_timeline"] = tl.to_summary()

    # ── Вспомогательные ─────────────────────────────────────────

    @staticmethod
    def _infer_source_type(source: str, ingestion_result: Any) -> str:
        """Определить тип источника из URL/имени."""
        src = source.lower()
        if "nature.com" in src or "nature/" in src:      return "nature"
        if "science" in src and "arxiv" not in src:      return "science"
        if "arxiv" in src:                                return "arxiv"
        if "reddit" in src:                               return "reddit"
        if "forum" in src or "stack" in src:              return "forum"
        if "blog" in src or "medium" in src:              return "blog"
        if "patent" in src:                               return "patent"
        if "wikipedia" in src:                            return "wikipedia"
        if "book" in src or "textbook" in src:            return "book"
        if "fiction" in src or "tale" in src:             return "fiction"
        if "news" in src or "bbc" in src or "cnn" in src: return "news"
        return "unknown"

    @staticmethod
    def _collect_all_dr(result: ProcessorResult) -> List[DistillationResult]:
        """Собрать все DistillationResult из ProcessorResult (для backward compat)."""
        # ProcessorResult не хранит dr напрямую — используем timeline_updates
        return []


# ══════════════════════════════════════════════════════════════════
#  5. ИНТЕГРАЦИЯ С ORACLE-1 СИСТЕМОЙ
#  Патч для oracle1_system_epoch и build_oracle1_system
# ══════════════════════════════════════════════════════════════════

def attach_significance_layer(
    oracle1_system: Dict[str, Any],
    llm_client=None,
    embedder=None,
) -> "SignificanceProcessor":
    """
    Встроить SignificanceProcessor в существующую Oracle-1 систему.

    После вызова:
      system["significance"] = processor

    Использование в цикле ingestion:
      result = pipeline.ingest(text, source, ts)
      sig_result = system["significance"].process(result, graph, llm)
      load_result_into_graph(result, graph)
      # Мутантные узлы уже добавлены в result.nodes/edges
      for node in sig_result.new_nodes:
          graph.add_node(node)
      for edge in sig_result.new_edges:
          graph.add_edge(edge)
    """
    distiller  = ConceptDistiller(llm_client=llm_client)
    processor  = SignificanceProcessor(distiller=distiller, embedder=embedder)
    oracle1_system["significance"] = processor
    logger.info("[SignificanceLayer] attached to oracle1 system")
    return processor


def get_significance_report(
    oracle1_system: Dict[str, Any],
    top_n: int = 10,
) -> Dict[str, Any]:
    """
    Сводный отчёт по эпистемической значимости узлов.
    """
    proc = oracle1_system.get("significance")
    if proc is None:
        return {"error": "significance layer not attached"}
    distiller = proc.distiller
    return {
        "top_by_velocity":       distiller.top_by_velocity(top_n),
        "top_by_epistemic_mass": distiller.top_by_epistemic_mass(top_n),
        "total_timelines":       len(distiller.timelines),
        "proven_nodes":          [
            cid for cid, tl in distiller.timelines.items()
            if tl.proven_status()
        ],
    }


# ══════════════════════════════════════════════════════════════════
#  Утилиты
# ══════════════════════════════════════════════════════════════════

def _parse_json_safe(raw: str) -> Dict:
    clean  = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    brace  = clean.find("{")
    if brace > 0:
        clean = clean[brace:]
    rbrace = clean.rfind("}")
    if rbrace >= 0:
        clean = clean[:rbrace + 1]
    return json.loads(clean)


def epistemic_level_for_node(node: Any) -> EpistemicLevel:
    """
    Определить текущий epistemic level узла по его метаданным.
    Использует: entity_type, scientific_score, provenance.
    """
    etype = getattr(node, "entity_type", "").lower()
    sci   = getattr(node, "scientific_score", 0.0)

    if "breakthrough" in etype or sci >= 9.0:
        return EpistemicLevel.PROVEN_DISCOVERY
    if "physical_principle" in etype or sci >= 7.5:
        return EpistemicLevel.PEER_REVIEWED
    if "research_frontier" in etype or sci >= 6.0:
        return EpistemicLevel.PREPRINT
    if "cultural_phantom" in etype or "fiction" in etype:
        return EpistemicLevel.FOLKLORE_FICTION
    if "speculation" in etype:
        return EpistemicLevel.SPECULATION
    # По источнику
    prov = getattr(node, "provenance", "unknown").lower()
    return get_epistemic_level(prov)


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════

def _smoke_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Significance Gradient Smoke Test ===")

    # ── Мок KnowledgeNode ───────────────────────────────────────
    class MockNode:
        def __init__(self, id_, text, domain, entity_type="physical_principle",
                     sci=5.0, social=0.0, forecast=0.0):
            self.id            = id_
            self.text          = text
            self.full_text     = text
            self.domain        = domain
            self.entity_type   = entity_type
            self.scientific_score = sci
            self.social_score  = social
            self.forecast_score = forecast
            self.maturity_score = 0.0
            self.strategic_value = 5.0
            self.provenance    = "test"
            self.timestamp     = -1672531200.0  # 1917
            self.embedding     = np.random.randn(384).astype(np.float32)
            self.__dict__      = self.__dict__

    class MockGraph:
        def __init__(self, nodes):
            self.nodes = {n.id: n for n in nodes}
        def get_node(self, nid): return self.nodes.get(nid)

    class MockLLM:
        def complete(self, _prompt):
            return json.dumps({
                "distill_type": "REINFORCEMENT",
                "reasoning": "Nature article confirms Einstein's theory",
                "reinforcement_delta": {
                    "scientific_score_delta": 1.5,
                    "social_score_delta": 0.5,
                    "maturity_score_delta": 0.3,
                },
                "new_node": {"text": "", "domain": "", "entity_type": "",
                             "abstract_concepts": [], "confidence": 0.0},
                "relation_type": "extends",
                "relation_strength": 0.9,
            })

    class MockLLMFairy:
        def complete(self, _prompt):
            return json.dumps({
                "distill_type": "ABSTRACT_CONCEPT",
                "reasoning": "Fairy tale mirror = remote image transmission",
                "reinforcement_delta": {},
                "new_node": {
                    "text": "Remote image transmission device",
                    "domain": "communication_technology",
                    "entity_type": "cultural_phantom",
                    "abstract_concepts": [
                        "portable device",
                        "on-demand activation",
                        "remote visual transmission",
                    ],
                    "confidence": 0.75,
                },
                "relation_type": "precursor_of",
                "relation_strength": 0.6,
            })

    # ── 1. Создаём canonical узел ────────────────────────────────
    einstein = MockNode(
        "einstein_1917",
        "Stimulated Emission of Radiation",
        "quantum_mechanics",
        entity_type="physical_principle",
        sci=8.0,
    )
    graph = MockGraph([einstein])

    distiller = ConceptDistiller(llm_client=MockLLM())
    processor = SignificanceProcessor(distiller=distiller)

    # ── 2. Источник 1: статья Nature 2024 → REINFORCEMENT ───────
    class FakeIngestion:
        source = "https://nature.com/articles/laser2024"
        nodes = [MockNode("nature_2024", "Laser quantum efficiency study",
                          "quantum_mechanics", sci=9.0)]
        edges = []
        cross_doc_edges = []

    sig1 = processor.process(FakeIngestion(), graph, llm_client=MockLLM())
    assert sig1.reinforcements >= 0  # может быть 0 если similarity < threshold
    logger.info(f"✓ Nature 2024: reinforcements={sig1.reinforcements}, "
                f"mutants={sig1.mutant_nodes}")

    # ── 3. Прямой тест distiller с форумным источником ──────────
    distiller.llm = MockLLM()
    dr_list = distiller.process_source(
        source_text="Discussion about stimulated emission quantum effects",
        source_type="reddit",
        source_name="reddit.com/r/physics",
        timestamp=1609459200.0,  # 2021
        graph_nodes=graph.nodes,
        embedder=None,
    )
    # Heuristic path (нет embedder) → keyword overlap
    logger.info(f"✓ Reddit distillation: {len(dr_list)} results")

    # ── 4. InfluenceTimeline — вручную добавим точки ─────────────
    tl = distiller.get_or_create_timeline("einstein_1917")
    tl.add(InfluencePoint(
        timestamp=-1672531200.0,  # 1917
        source="Einstein original",
        source_type="peer_reviewed",
        epistemic_level=EpistemicLevel.PEER_REVIEWED,
        weight=EpistemicWeight(EpistemicLevel.PEER_REVIEWED, 0.85,
                               -1672531200.0).effective_weight,
    ))
    tl.add(InfluencePoint(
        timestamp=0.0,  # 1970
        source="Textbook mention",
        source_type="textbook",
        epistemic_level=EpistemicLevel.TEXTBOOK_MENTION,
        weight=EpistemicWeight(EpistemicLevel.TEXTBOOK_MENTION, 0.40,
                               0.0).effective_weight,
    ))
    tl.add(InfluencePoint(
        timestamp=1704067200.0,  # 2024
        source="Nature 2024",
        source_type="nature",
        epistemic_level=EpistemicLevel.PROVEN_DISCOVERY,
        weight=EpistemicWeight(EpistemicLevel.PROVEN_DISCOVERY, 1.0,
                               1704067200.0).effective_weight,
    ))
    logger.info(f"✓ InfluenceTimeline:")
    logger.info(f"    epistemic_mass   = {tl.epistemic_mass:.3f}")
    logger.info(f"    velocity         = {tl.velocity:.3f}")
    logger.info(f"    source_diversity = {tl.source_diversity:.3f}")
    logger.info(f"    proven_status    = {tl.proven_status()}")
    logger.info(f"    peak_year        = {tl.peak_epistemic_year}")
    assert tl.proven_status(), "FAIL: should be proven with Nature 2024 source"
    logger.info("✓ proven_status=True after Nature source")

    # ── 5. Тест ABSTRACT_CONCEPT из сказки ─────────────────────
    fairy_node = MockNode(
        "magic_mirror",
        "Magic mirror shows distant images on command",
        "folklore",
        entity_type="cultural_phantom",
    )
    fairy_node.embedding = np.random.randn(384).astype(np.float32)
    graph2 = MockGraph([einstein, fairy_node])
    distiller2 = ConceptDistiller(llm_client=MockLLMFairy())
    dr_fairy = distiller2.process_source(
        source_text="A magic mirror that shows what happens far away when asked",
        source_type="fairytale",
        source_name="Snow White fairy tale",
        timestamp=-2208988800.0,  # ~1900
        graph_nodes=graph2.nodes,
        embedder=None,
    )
    logger.info(f"✓ Fairy tale distillation: {len(dr_fairy)} results")

    # Проверяем что создаётся мутантный узел для abstract_concept
    proc2 = SignificanceProcessor(distiller=distiller2)
    # Прямой вызов _create_mutant_node
    if dr_fairy:
        dr = dr_fairy[0]
        if dr.distill_type == DistillationType.ABSTRACT_CONCEPT:
            logger.info(f"✓ ABSTRACT_CONCEPT detected from fairy tale")
            logger.info(f"    concepts: {dr.new_node_concepts}")
        else:
            logger.info(f"  (heuristic: {dr.distill_type.value})")

    # ── 6. EpistemicWeight расчёт ───────────────────────────────
    w_nature  = EpistemicWeight(EpistemicLevel.PROVEN_DISCOVERY, 1.0, 0.0)
    w_reddit  = EpistemicWeight(EpistemicLevel.FORUM_DISCUSSION, 0.2, 0.0)
    w_fiction = EpistemicWeight(EpistemicLevel.FOLKLORE_FICTION, 0.1, 0.0)
    logger.info(f"✓ EpistemicWeight comparison:")
    logger.info(f"    nature  = {w_nature.effective_weight:.3f}")
    logger.info(f"    reddit  = {w_reddit.effective_weight:.3f}")
    logger.info(f"    fiction = {w_fiction.effective_weight:.3f}")
    assert w_nature.effective_weight > w_reddit.effective_weight > w_fiction.effective_weight
    logger.info("✓ nature > reddit > fiction (correct ordering)")

    logger.info("=== Smoke Test PASSED ===")


# ══════════════════════════════════════════════════════════════════════════════
# INLINED MODULE: oracle1_canonical_key.py
# Canonical Key System + Domain Classifier
# ══════════════════════════════════════════════════════════════════════════════

# Optional dependencies used by this module
try:
    from torch.utils.data import DataLoader, Dataset as TorchDataset
except ImportError:
    DataLoader = None       # type: ignore[assignment,misc]
    TorchDataset = None     # type: ignore[assignment,misc]

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("[CanonicalKey] transformers not available — using hash-only embedder")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("[CanonicalKey] faiss not available — brute-force ANN fallback")


# ══════════════════════════════════════════════════════════════════
#  LOOKUP TABLES  (3 основные + Alias)
#  Пополняются LLM и вручную. Хранятся в памяти + сериализуются.
# ══════════════════════════════════════════════════════════════════

# Начальные записи — seed. При старте системы LLM дополняет их.
_DEFAULT_DOMAINS: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "physics",
    0x0002: "quantum_mechanics",
    0x0003: "optics",
    0x0004: "chemistry",
    0x0005: "biology",
    0x0006: "medicine",
    0x0007: "military_technology",
    0x0008: "electronics",
    0x0009: "materials_science",
    0x000A: "computer_science",
    0x000B: "mathematics",
    0x000C: "economics",
    0x000D: "fiction",
    0x000E: "mythology_folklore",
    0x000F: "social_science",
    0x0010: "energy",
    0x0011: "space_technology",
    0x0012: "biotechnology",
    0x0013: "nanotechnology",
    0x0014: "quantum_electronics",
    0x0015: "military_finance",
    0x0016: "industrial_manufacturing",
    0x0017: "environmental_science",
    0x0018: "neuroscience",
    0x0019: "artificial_intelligence",
}

_DEFAULT_GROUPS: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "fundamental_principle",
    0x0002: "applied_technology",
    0x0003: "cultural_artifact",
    0x0004: "economic_force",
    0x0005: "biological_system",
    0x0006: "social_phenomenon",
    0x0007: "military_system",
    0x0008: "industrial_process",
    0x0009: "measurement_instrument",
    0x000A: "theoretical_framework",
    0x000B: "regulatory_construct",
    0x000C: "fictional_concept",
    0x000D: "historical_event",
    0x000E: "speculative_technology",
    0x000F: "barrier_constraint",
}

_DEFAULT_ENTITY_TYPES: Dict[int, str] = {
    0x0000: "unknown",
    0x0001: "physical_principle",
    0x0002: "breakthrough_node",
    0x0003: "cultural_phantom",
    0x0004: "dormant_principle",
    0x0005: "functional_spec",
    0x0006: "resource_source",
    0x0007: "industrial_substrate",
    0x0008: "technical_mastery",
    0x0009: "unifying_theory",
    0x000A: "research_frontier",
    0x000B: "incumbent_tech",
    0x000C: "challenger_tech",
    0x000D: "enabler_component",
    0x000E: "measurement_tool",
    0x000F: "side_effect_risk",
    0x0010: "regulatory_constraint",
    0x0011: "convergence_node",
    0x0012: "breakthrough_event",
    0x0013: "economic_barrier",
    0x0014: "physical_limit",
    0x0015: "experimental_method",
}


class LookupTable:
    """
    Двунаправленная таблица id ↔ name.
    Потокобезопасна для чтения; запись однопоточная.
    Пополняется LLM через register() или вручную.
    """

    def __init__(self, name: str, initial: Dict[int, str]):
        self.name = name
        self._id2name: Dict[int, str] = dict(initial)
        self._name2id: Dict[str, int] = {v: k for k, v in initial.items()}
        self._next_id: int = max(initial.keys(), default=0) + 1

    # ── Основной API ────────────────────────────────────────────

    def get_id(self, name: str) -> int:
        """Вернуть ID по имени. 0x0000 если не найдено."""
        return self._name2id.get(name.lower().strip(), 0x0000)

    def get_name(self, id_: int) -> str:
        """Вернуть имя по ID."""
        return self._id2name.get(id_, "unknown")

    def register(self, name: str) -> int:
        """
        Зарегистрировать новое имя и вернуть его ID.
        Если имя уже есть — вернуть существующий ID.
        """
        key = name.lower().strip()
        if key in self._name2id:
            return self._name2id[key]
        new_id = self._next_id
        self._next_id += 1
        self._id2name[new_id] = key
        self._name2id[key] = new_id
        logger.info(f"[LookupTable:{self.name}] registered '{key}' → 0x{new_id:04X}")
        return new_id

    def all_names(self) -> List[str]:
        return list(self._name2id.keys())

    def __len__(self) -> int:
        return len(self._id2name)

    # ── Сериализация ────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {"name": self.name,
                "id2name": {str(k): v for k, v in self._id2name.items()},
                "next_id": self._next_id}

    @classmethod
    def from_dict(cls, d: Dict) -> "LookupTable":
        id2name = {int(k): v for k, v in d["id2name"].items()}
        obj = cls(d["name"], id2name)
        obj._next_id = d["next_id"]
        return obj


# ── Глобальные таблицы (синглтоны) ──────────────────────────────

DOMAIN_TABLE      = LookupTable("domain",      _DEFAULT_DOMAINS)
GROUP_TABLE       = LookupTable("group",        _DEFAULT_GROUPS)
ENTITY_TYPE_TABLE = LookupTable("entity_type",  _DEFAULT_ENTITY_TYPES)


def save_lookup_tables(path: str) -> None:
    data = {
        "domain":      DOMAIN_TABLE.to_dict(),
        "group":       GROUP_TABLE.to_dict(),
        "entity_type": ENTITY_TYPE_TABLE.to_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"[LookupTables] saved to {path}")


def load_lookup_tables(path: str) -> None:
    global DOMAIN_TABLE, GROUP_TABLE, ENTITY_TYPE_TABLE
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    DOMAIN_TABLE      = LookupTable.from_dict(data["domain"])
    GROUP_TABLE       = LookupTable.from_dict(data["group"])
    ENTITY_TYPE_TABLE = LookupTable.from_dict(data["entity_type"])
    logger.info(f"[LookupTables] loaded from {path}")


# ══════════════════════════════════════════════════════════════════
#  ALIAS TABLE
#  Строка (любой язык/написание) → Canonical ID
#  Хранится отдельно; в production — Redis
# ══════════════════════════════════════════════════════════════════

class AliasTable:
    """
    Разрешает синонимы и варианты написания к единому canonical_id.

    ВАЖНО: один термин в РАЗНЫХ доменах → РАЗНЫЕ canonical_id.
    Поэтому ключ хранится как (alias, domain_hint) → canonical_id.

    Если domain_hint не указан (None) — используется первый
    зарегистрированный для данного alias (точный match).

    "лазер" (physics) → ключ physics
    "лазер" (fiction) → ключ fiction    ← другой!
    "LASER" (physics) → ключ physics    ← тот же что "лазер" physics

    При ingestion:
      1. Нормализуем термин (lower + strip).
      2. Ищем в alias_table по (term, domain).
         – Нашли → концепт уже существует → merge наблюдений.
         – Не нашли → создаём новый Canonical ID → регистрируем alias.
      3. LLM может явно указать синонимы → bulk_register().
    """

    def __init__(self):
        # (alias_str, domain_str_or_None) → canonical_id bytes(12)
        self._map: Dict[Tuple[str, Optional[str]], bytes] = {}
        # alias_str → set of domains (для cross-domain lookup)
        self._alias_domains: Dict[str, List[str]] = defaultdict(list)

    def register(self, alias: str, canonical_id: bytes,
                 domain: Optional[str] = None) -> None:
        alias_key = alias.lower().strip()
        domain_key = domain.lower().strip() if domain else None
        map_key = (alias_key, domain_key)
        if map_key in self._map and self._map[map_key] != canonical_id:
            # Конфликт в одном домене — предупреждение, оставляем оригинал
            logger.warning(
                f"[AliasTable] conflict: '{alias_key}'@{domain_key} "
                f"was {self._map[map_key].hex()} "
                f"now {canonical_id.hex()} — keeping original"
            )
            return
        self._map[map_key] = canonical_id
        if domain_key and domain_key not in self._alias_domains[alias_key]:
            self._alias_domains[alias_key].append(domain_key)

    def bulk_register(self, aliases: List[str], canonical_id: bytes,
                      domain: Optional[str] = None) -> None:
        for a in aliases:
            self.register(a, canonical_id, domain=domain)

    def lookup(self, alias: str,
               domain: Optional[str] = None) -> Optional[bytes]:
        """
        Поиск canonical_id по alias + domain.
        Если domain указан — точный поиск.
        Если domain=None — ищем без домена (точный alias в любом контексте).
        """
        alias_key  = alias.lower().strip()
        domain_key = domain.lower().strip() if domain else None
        return self._map.get((alias_key, domain_key))

    def all_domains_for(self, alias: str) -> List[str]:
        """Все домены, в которых зарегистрирован этот alias."""
        return self._alias_domains.get(alias.lower().strip(), [])

    def __len__(self) -> int:
        return len(self._map)

    def to_dict(self) -> Dict[str, str]:
        # Сериализуем ключ как "alias|||domain" (domain может быть None → "")
        return {
            f"{alias}|||{domain or ''}": cid.hex()
            for (alias, domain), cid in self._map.items()
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "AliasTable":
        obj = cls()
        for compound_key, hex_val in d.items():
            if "|||" in compound_key:
                alias, domain_part = compound_key.split("|||", 1)
                domain = domain_part if domain_part else None
            else:
                alias, domain = compound_key, None
            cid = bytes.fromhex(hex_val)
            obj._map[(alias, domain)] = cid
            if domain and domain not in obj._alias_domains[alias]:
                obj._alias_domains[alias].append(domain)
        return obj


ALIAS_TABLE = AliasTable()


# ══════════════════════════════════════════════════════════════════
#  SEMANTIC HASH  (6 байт)
#  Квантизация эмбеддинга → детерминированный хэш
#  Два семантически близких вектора → одинаковый или соседний хэш
# ══════════════════════════════════════════════════════════════════

class SemanticHasher:
    """
    Product Quantization (PQ) → 6-байтный семантический хэш.

    При наличии FAISS:
      – Обучает PQ-кодировщик на накопленных эмбеддингах.
      – Выдаёт 6-байтный код (48 бит).

    Без FAISS (fallback):
      – SHA-256 от нормализованного вектора.
      – Коллизии выше, но работает без зависимостей.

    После обучения PQ один и тот же концепт в разных формулировках
    попадает в один центроид → один хэш → один canonical_id.
    """

    PQ_BYTES   = 6    # 48 бит кода
    PQ_SUBVECS = 6    # количество субпространств (должно делить embedding_dim)
    MIN_TRAIN  = 256  # минимум векторов для обучения PQ

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self._pq           = None   # faiss.ProductQuantizer после обучения
        self._trained      = False
        self._train_buffer: List[np.ndarray] = []

    # ── Основной API ────────────────────────────────────────────

    def hash(self, embedding: np.ndarray) -> bytes:
        """
        Вычислить 6-байтный семантический хэш для эмбеддинга.
        Автоматически обучает PQ при достаточном буфере.
        """
        vec = self._normalize(embedding)
        self._train_buffer.append(vec)

        if FAISS_AVAILABLE and not self._trained:
            if len(self._train_buffer) >= self.MIN_TRAIN:
                self._train_pq()

        if FAISS_AVAILABLE and self._trained:
            return self._pq_hash(vec)
        else:
            return self._fallback_hash(vec)

    def hash_text(self, text: str, embedder=None) -> bytes:
        """Хэш от текста. Если embedder=None — чистый SHA-256."""
        if embedder is not None:
            try:
                emb = embedder.encode(text)
                return self.hash(np.array(emb, dtype=np.float32))
            except Exception as e:
                logger.warning(f"[SemanticHasher] embedder failed: {e}")
        # Fallback: хэш от нормализованного текста
        normalized = text.lower().strip()
        digest = hashlib.sha256(normalized.encode()).digest()
        return digest[:self.PQ_BYTES]

    # ── Обучение PQ ─────────────────────────────────────────────

    def _train_pq(self) -> None:
        if not FAISS_AVAILABLE:
            return
        matrix = np.stack(self._train_buffer).astype(np.float32)
        # subvector size: embedding_dim / n_subvecs
        sub_dim = self.embedding_dim // self.PQ_SUBVECS
        if self.embedding_dim % self.PQ_SUBVECS != 0:
            logger.warning(
                f"[SemanticHasher] embedding_dim={self.embedding_dim} "
                f"не делится на PQ_SUBVECS={self.PQ_SUBVECS} — fallback"
            )
            return
        # 256 центроидов на субпространство → 8 бит на субпространство
        self._pq = faiss.ProductQuantizer(self.embedding_dim, self.PQ_SUBVECS, 8)
        self._pq.train(matrix)
        self._trained = True
        logger.info(
            f"[SemanticHasher] PQ trained on {len(self._train_buffer)} vectors "
            f"({self.PQ_SUBVECS} subvecs × 8 bits = {self.PQ_BYTES} bytes)"
        )

    def force_train(self, embeddings: np.ndarray) -> None:
        """Явно обучить PQ на переданных векторах (для инициализации)."""
        self._train_buffer = [self._normalize(e) for e in embeddings]
        if len(self._train_buffer) >= self.MIN_TRAIN:
            self._train_pq()

    # ── Внутренние методы ───────────────────────────────────────

    def _pq_hash(self, vec: np.ndarray) -> bytes:
        codes = np.zeros((1, self.PQ_SUBVECS), dtype=np.uint8)
        self._pq.compute_codes(vec.reshape(1, -1), codes)
        # codes[0] → PQ_SUBVECS байт → берём первые PQ_BYTES
        raw = bytes(codes[0].tolist())
        return raw[:self.PQ_BYTES]

    @staticmethod
    def _fallback_hash(vec: np.ndarray) -> bytes:
        """SHA-256 от квантизованного вектора (без FAISS)."""
        quantized = (vec * 127).astype(np.int8)
        digest = hashlib.sha256(quantized.tobytes()).digest()
        return digest[:6]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        v = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)


# Глобальный хэшер
SEMANTIC_HASHER = SemanticHasher(embedding_dim=384)


# ══════════════════════════════════════════════════════════════════
#  CANONICAL KEY  (12 байт)
#  Основная точка входа: текст + контекст → bytes(12)
# ══════════════════════════════════════════════════════════════════

@dataclass
class CanonicalKey:
    """
    12-байтный ключ концепта.

    domain_id     : uint16  — область знаний
    group_id      : uint16  — типовая группа
    entity_type_id: uint16  — тип сущности
    sem_hash      : bytes6  — семантический хэш эмбеддинга

    Важно: один термин "лазер" в разных доменах → разные domain_id
    → разные ключи. Это позволяет агрегировать cross-domain
    присутствие концепта и измерять "необходимость" технологии.
    """

    domain_id:      int
    group_id:       int
    entity_type_id: int
    sem_hash:       bytes   # ровно 6 байт

    # ── Сериализация ────────────────────────────────────────────

    def to_bytes(self) -> bytes:
        """Упаковать в 12 байт: 3 × uint16 (big-endian) + 6 байт хэша."""
        header = struct.pack(">HHH", self.domain_id, self.group_id,
                             self.entity_type_id)
        return header + self.sem_hash

    @classmethod
    def from_bytes(cls, raw: bytes) -> "CanonicalKey":
        assert len(raw) == 12, f"Expected 12 bytes, got {len(raw)}"
        d, g, e = struct.unpack(">HHH", raw[:6])
        return cls(domain_id=d, group_id=g, entity_type_id=e,
                   sem_hash=raw[6:])

    def hex(self) -> str:
        return self.to_bytes().hex()

    def __hash__(self):
        return hash(self.to_bytes())

    def __eq__(self, other):
        if isinstance(other, CanonicalKey):
            return self.to_bytes() == other.to_bytes()
        return False

    # ── Читаемое представление ──────────────────────────────────

    def describe(self) -> str:
        d = DOMAIN_TABLE.get_name(self.domain_id)
        g = GROUP_TABLE.get_name(self.group_id)
        e = ENTITY_TYPE_TABLE.get_name(self.entity_type_id)
        h = self.sem_hash.hex()
        return f"[{d} / {g} / {e}] sem:{h}"

    # ── Измерение "необходимости" концепта ─────────────────────

    @staticmethod
    def cross_domain_presence(
        term: str,
        alias_table: AliasTable,
        graph_nodes: Dict[str, Any],   # {canonical_id_hex: node}
    ) -> Dict[str, int]:
        """
        Посчитать, в скольких доменах встречается термин.
        Возвращает {domain_name: count}.

        Высокий cross-domain score = высокая "необходимость"
        технологии (архетип Oracle-1).
        """
        counts: Dict[str, int] = defaultdict(int)
        # Собираем все canonical_id, где alias совпадает семантически
        # (упрощённо: по exact alias match — в production через FAISS)
        cid_bytes = alias_table.lookup(term)
        if cid_bytes is None:
            return dict(counts)

        # Ищем узлы с таким же sem_hash (последние 6 байт)
        target_sem = cid_bytes[6:]
        for hex_id, node in graph_nodes.items():
            try:
                raw = bytes.fromhex(hex_id)
                if len(raw) != 12:
                    continue
                if raw[6:] == target_sem:
                    d_id = struct.unpack(">H", raw[0:2])[0]
                    domain_name = DOMAIN_TABLE.get_name(d_id)
                    counts[domain_name] += 1
            except Exception:
                continue
        return dict(counts)

    @staticmethod
    def necessity_score(cross_domain_counts: Dict[str, int]) -> float:
        """
        Индекс необходимости: log(1 + sum) × кол-во доменов.
        Максимум при широком и глубоком cross-domain присутствии.
        """
        if not cross_domain_counts:
            return 0.0
        total     = sum(cross_domain_counts.values())
        n_domains = len(cross_domain_counts)
        return float(math.log1p(total) * n_domains)


# ══════════════════════════════════════════════════════════════════
#  OBSERVATION RECORD
#  Лёгкая запись наблюдения — время принадлежит ему, не концепту
# ══════════════════════════════════════════════════════════════════

@dataclass
class ObservationRecord:
    """
    Одно наблюдение концепта из одного источника.
    canonical_id → FK на CanonicalKey.
    Много ObservationRecord → один CanonicalKey.
    """
    canonical_id:  bytes            # 12-байтный ключ
    obs_id:        str              # UUID наблюдения
    timestamp:     float            # когда опубликован источник
    source:        str              # URL / название
    source_type:   str              # arxiv / forum / news / ...
    trust:         float            # 0.0–1.0
    delta_fields:  Dict[str, Any]   # только изменённые поля узла

    @classmethod
    def create(cls, canonical_id: bytes, source: str, timestamp: float,
               source_type: str, trust: float,
               delta_fields: Optional[Dict] = None) -> "ObservationRecord":
        return cls(
            canonical_id=canonical_id,
            obs_id=str(uuid.uuid4()),
            timestamp=timestamp,
            source=source,
            source_type=source_type,
            trust=trust,
            delta_fields=delta_fields or {},
        )


# ══════════════════════════════════════════════════════════════════
#  CONCEPT CLASSIFIER
#  Классифицирует термин → (domain_id, group_id, entity_type_id)
#
#  Три режима работы:
#    1. Cold start: только LLM, все ответы → training_data
#    2. Warm:       DeBERTa/BERT fast path + LLM slow path
#    3. Hot:        DeBERTa покрывает 95%+ случаев
#
#  Также используется для ПРЕДСКАЗАНИЯ домена нового концепта
#  без LLM на основе контекста (inference mode).
# ══════════════════════════════════════════════════════════════════

@dataclass
class ClassificationSample:
    """Обучающий пример для классификатора."""
    text:          str
    context:       str
    source_type:   str
    domain_name:   str
    group_name:    str
    entity_type:   str
    confidence:    float = 1.0   # 1.0 = LLM-ground-truth, <1.0 = predicted


class _SimpleTextEncoder:
    """
    Минимальный TF-IDF / bag-of-ngrams энкодер без трансформеров.
    Используется если transformers недоступны.
    """
    def __init__(self, max_features: int = 4096):
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._fitted = False

    def fit(self, texts: List[str]) -> None:
        counts: Counter = Counter()
        for t in texts:
            for w in self._tokenize(t):
                counts[w] += 1
        top = [w for w, _ in counts.most_common(self.max_features)]
        self._vocab = {w: i for i, w in enumerate(top)}
        self._fitted = True

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.max_features, dtype=np.float32)
        tokens = self._tokenize(text)
        for w in tokens:
            idx = self._vocab.get(w)
            if idx is not None:
                vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            vec /= norm
        return vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r'[a-zа-яё]{3,}', text.lower())
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        return words + bigrams


class ConceptClassifierModel(nn.Module if TORCH_AVAILABLE else object):
    """
    Лёгкая классификационная голова поверх текстового энкодера.
    При наличии transformers — использует CLS-токен BERT/DeBERTa.
    Без них — TF-IDF → Linear → softmax.

    Три независимые головы:
        domain_head      → n_domains классов
        group_head       → n_groups классов
        entity_type_head → n_types классов
    """

    def __init__(self, encoder_dim: int,
                 n_domains: int, n_groups: int, n_types: int,
                 hidden: int = 256, dropout: float = 0.1):
        if TORCH_AVAILABLE:
            super().__init__()
        self.encoder_dim = encoder_dim

        if TORCH_AVAILABLE:
            self.shared = nn.Sequential(
                nn.Linear(encoder_dim, hidden),
                nn.GELU(),
                nn.LayerNorm(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
            )
            half = hidden // 2
            self.domain_head      = nn.Linear(half, n_domains)
            self.group_head       = nn.Linear(half, n_groups)
            self.entity_type_head = nn.Linear(half, n_types)
        else:
            # stubs
            self.shared = self.domain_head = None
            self.group_head = self.entity_type_head = None

    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None, None, None
        shared = self.shared(x)
        return (self.domain_head(shared),
                self.group_head(shared),
                self.entity_type_head(shared))


class _ClassifierDataset(TorchDataset if TORCH_AVAILABLE else object):
    def __init__(self, samples: List[ClassificationSample],
                 encoder, domain_table: LookupTable,
                 group_table: LookupTable, entity_table: LookupTable):
        self.samples      = samples
        self.encoder      = encoder
        self.domain_table = domain_table
        self.group_table  = group_table
        self.entity_table = entity_table

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        combined = f"{s.text} [SEP] {s.context} [SRC] {s.source_type}"
        vec = torch.tensor(self.encoder.encode(combined), dtype=torch.float32)
        d   = self.domain_table.get_id(s.domain_name)
        g   = self.group_table.get_id(s.group_name)
        e   = self.entity_table.get_id(s.entity_type)
        return vec, torch.tensor(d), torch.tensor(g), torch.tensor(e)


class ConceptClassifier:
    """
    Главный класс классификации и предсказания доменов.

    Использование:
        clf = ConceptClassifier()

        # При cold start:
        result = clf.classify("лазер", context="...", source_type="arxiv",
                               llm_fallback=my_llm)
        # → ClassifyResult(domain="optics", group="fundamental_principle", ...)

        # После накопления данных — обучение:
        clf.train(epochs=5)

        # Предсказание для нового концепта (без LLM):
        pred = clf.predict("новый термин", context="...", source_type="forum")
        # → ClassifyResult с confidence

        # Измерение необходимости технологии:
        score = clf.necessity_score("лазер", graph_nodes)
    """

    CONFIDENCE_THRESHOLD = CONFIG["classifier_confidence_threshold"]
    RETRAIN_EVERY        = CONFIG["classifier_retrain_every"]
    LLM_CLASSIFY_PROMPT  = """\
You are a knowledge graph classifier. Given a concept term with context,
output the most appropriate classification.

TERM: {term}
CONTEXT: {context}
SOURCE TYPE: {source_type}

Available domains:      {domains}
Available groups:       {groups}
Available entity types: {entity_types}

Rules:
- Same term in different contexts → different domains (e.g. "laser" in physics vs fiction)
- Choose the most specific applicable domain
- If unsure → prefer broader categories
- Synonyms/aliases field: list 2-5 alternative names for this exact concept

OUTPUT ONLY valid JSON:
{{
  "domain":      "<exact name from domains list>",
  "group":       "<exact name from groups list>",
  "entity_type": "<exact name from entity_types list>",
  "confidence":  <0.0-1.0>,
  "reasoning":   "<one sentence>",
  "aliases":     ["<alias1>", "<alias2>"]
}}
"""

    def __init__(self,
                 domain_table: LookupTable      = DOMAIN_TABLE,
                 group_table: LookupTable       = GROUP_TABLE,
                 entity_table: LookupTable      = ENTITY_TYPE_TABLE,
                 alias_table: AliasTable        = ALIAS_TABLE,
                 hasher: SemanticHasher         = SEMANTIC_HASHER,
                 model_path: Optional[str]      = None,
                 encoder_name: str              = "sentence-transformers/all-MiniLM-L6-v2"):

        self.domain_table  = domain_table
        self.group_table   = group_table
        self.entity_table  = entity_table
        self.alias_table   = alias_table
        self.hasher        = hasher

        self.training_data: List[ClassificationSample] = []
        self._llm_calls_since_retrain = 0

        # Энкодер текста
        if TRANSFORMERS_AVAILABLE:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(encoder_name)
                self._bert       = AutoModel.from_pretrained(encoder_name)
                self._bert.eval()
                self._encoder_dim = self._bert.config.hidden_size
                self._use_bert    = True
                logger.info(f"[Classifier] using BERT encoder: {encoder_name}")
            except Exception as e:
                logger.warning(f"[Classifier] BERT load failed ({e}) → TF-IDF")
                self._use_bert    = False
                self._simple_enc  = _SimpleTextEncoder()
                self._encoder_dim = 4096
        else:
            self._use_bert    = False
            self._simple_enc  = _SimpleTextEncoder()
            self._encoder_dim = 4096

        # Нейросетевая голова классификатора
        self._model: Optional[ConceptClassifierModel] = None
        self._model_ready = False

        if model_path and Path(model_path).exists():
            self._load_model(model_path)

    # ══════════════════════════════════════════════════════════════
    #  Основной API
    # ══════════════════════════════════════════════════════════════

    @dataclass
    class ClassifyResult:
        domain:         str
        group:          str
        entity_type:    str
        confidence:     float
        canonical_key:  bytes           # 12-байтный ключ
        aliases:        List[str]       # синонимы от LLM
        via_llm:        bool = False    # True если использован LLM

        def describe(self) -> str:
            ck = CanonicalKey.from_bytes(self.canonical_key)
            return (f"domain={self.domain} / group={self.group} / "
                    f"entity_type={self.entity_type} | "
                    f"conf={self.confidence:.2f} | key={ck.hex()} "
                    f"{'[LLM]' if self.via_llm else '[model]'}")

    def classify(self,
                 term: str,
                 context: str = "",
                 source_type: str = "unknown",
                 embedder=None,
                 llm_fallback=None) -> "ConceptClassifier.ClassifyResult":
        """
        Классифицировать концепт.

        1. Проверяем alias_table (если уже видели термин).
        2. Fast path: нейросетевой классификатор (если обучен).
        3. Slow path: LLM (если confidence низкий или модель не обучена).
        4. Регистрируем результат → canonical_key → alias_table.
        """
        # ── Шаг 1: alias lookup (domain-aware) ───────────────────
        # Сначала пытаемся найти по домену из source_type (приближение),
        # затем — без домена. Так "лазер" в физике и в фантастике
        # дают РАЗНЫЕ ключи, а "LASER" в той же физике — тот же ключ.
        existing = self.alias_table.lookup(term, domain=source_type)
        if existing is not None:
            ck = CanonicalKey.from_bytes(existing)
            return ConceptClassifier.ClassifyResult(
                domain      = self.domain_table.get_name(ck.domain_id),
                group       = self.group_table.get_name(ck.group_id),
                entity_type = self.entity_table.get_name(ck.entity_type_id),
                confidence  = 1.0,
                canonical_key = existing,
                aliases     = [term],
                via_llm     = False,
            )

        # ── Шаг 2: fast path (нейросеть) ────────────────────────
        if self._model_ready:
            result = self._model_classify(term, context, source_type)
            if result and result.confidence >= self.CONFIDENCE_THRESHOLD:
                self._register_result(term, result, embedder)
                return result

        # ── Шаг 3: slow path (LLM) ──────────────────────────────
        if llm_fallback is not None:
            result = self._llm_classify(term, context, source_type, llm_fallback)
            if result:
                # Добавляем в обучающую выборку
                sample = ClassificationSample(
                    text=term, context=context, source_type=source_type,
                    domain_name=result.domain, group_name=result.group,
                    entity_type=result.entity_type, confidence=result.confidence,
                )
                self.training_data.append(sample)
                self._llm_calls_since_retrain += 1

                # Автоматическое переобучение
                if self._llm_calls_since_retrain >= self.RETRAIN_EVERY:
                    logger.info("[Classifier] auto-retrain triggered")
                    self.train(epochs=3)
                    self._llm_calls_since_retrain = 0

                self._register_result(term, result, embedder)
                return result

        # ── Шаг 4: fallback (неизвестный домен) ─────────────────
        return self._unknown_result(term, context, source_type, embedder)

    def predict(self, term: str,
                context: str = "",
                source_type: str = "unknown") -> "ConceptClassifier.ClassifyResult":
        """
        Предсказание без LLM. Используется для:
          – batch-классификации новых узлов
          – предсказания домена виртуальных/phantom-узлов
          – оценки cross-domain необходимости концепта
        """
        if self._model_ready:
            result = self._model_classify(term, context, source_type)
            if result:
                return result
        return self._unknown_result(term, context, source_type, embedder=None)

    def necessity_score(self,
                        term: str,
                        graph_nodes: Dict[str, Any]) -> float:
        """
        Измерить "необходимость" концепта: насколько широко
        он представлен в разных доменах графа.

        Высокий балл = концепт нужен везде = высокая
        вероятность breakthrough (ключевая метрика Oracle-1).
        """
        counts = CanonicalKey.cross_domain_presence(
            term, self.alias_table, graph_nodes)
        score = CanonicalKey.necessity_score(counts)
        logger.debug(
            f"[Classifier] necessity('{term}') = {score:.3f} "
            f"across {len(counts)} domains: {counts}"
        )
        return score

    # ══════════════════════════════════════════════════════════════
    #  Обучение
    # ══════════════════════════════════════════════════════════════

    def train(self,
              epochs: int = 10,
              batch_size: int = 32,
              lr: float = 2e-4,
              save_path: Optional[str] = None) -> Dict[str, float]:
        """
        Обучить/дообучить классификатор на накопленных примерах.
        Возвращает {domain_acc, group_acc, entity_acc, val_domain_acc, …}.

        Улучшения по сравнению с v1:
          • 85/15 train/val split — видим реальную обобщающую способность.
          • Веса старой модели СОХРАНЯЮТСЯ при росте таблиц (expand_heads):
            новые классы инициализируются нулями, старые не сбрасываются.
          • label_smoothing в CrossEntropyLoss — снижает overfit на редкие классы.
          • Confidence вычисляется как среднее (не произведение) трёх max-prob,
            что даёт адекватный сигнал при >10 классах в каждой голове.
        """
        if not TORCH_AVAILABLE:
            logger.warning("[Classifier.train] PyTorch unavailable — skip")
            return {}

        if len(self.training_data) < 10:
            logger.warning(
                f"[Classifier.train] too few samples ({len(self.training_data)}) — skip"
            )
            return {}

        logger.info(
            f"[Classifier.train] starting: {len(self.training_data)} samples, "
            f"{epochs} epochs"
        )

        # ── Шаг 1: подготовка энкодера ───────────────────────────
        encoder = self._get_encoder()
        if not self._use_bert:
            texts = [f"{s.text} {s.context}" for s in self.training_data]
            self._simple_enc.fit(texts)

        # ── Шаг 2: train/val split (85/15) ──────────────────────
        val_frac  = CONFIG.get("classifier_val_split", 0.15)
        n_val     = max(1, int(len(self.training_data) * val_frac))
        # Deterministic shuffle based on text hash so results are reproducible
        sorted_data = sorted(self.training_data,
                             key=lambda s: hashlib.md5(s.text.encode()).hexdigest())
        val_data   = sorted_data[:n_val]
        train_data = sorted_data[n_val:]

        # ── Шаг 3: инициализация/расширение модели ──────────────
        n_d = len(self.domain_table)
        n_g = len(self.group_table)
        n_e = len(self.entity_table)

        new_model = ConceptClassifierModel(
            encoder_dim=self._encoder_dim,
            n_domains=n_d, n_groups=n_g, n_types=n_e,
        )

        # Preserve old weights when the table has grown (new classes appear)
        if (self._model is not None
                and CONFIG.get("classifier_max_head_expand", True)):
            self._expand_heads(new_model)

        self._model = new_model

        # ── Шаг 4: DataLoader ────────────────────────────────────
        train_ds = _ClassifierDataset(
            train_data, encoder,
            self.domain_table, self.group_table, self.entity_table,
        )
        val_ds = _ClassifierDataset(
            val_data, encoder,
            self.domain_table, self.group_table, self.entity_table,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                                  shuffle=False, drop_last=False)

        optimizer  = torch.optim.Adam(self._model.parameters(), lr=lr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
        smoothing  = CONFIG.get("classifier_label_smoothing", 0.05)
        criterion  = nn.CrossEntropyLoss(label_smoothing=smoothing)

        # ── Шаг 5: обучение ──────────────────────────────────────
        metrics: Dict[str, float] = {}
        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            correct    = [0, 0, 0]
            total_     = 0

            for batch_x, batch_d, batch_g, batch_e in train_loader:
                optimizer.zero_grad()
                logits_d, logits_g, logits_e = self._model(batch_x)
                loss = (criterion(logits_d, batch_d) +
                        criterion(logits_g, batch_g) +
                        criterion(logits_e, batch_e))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_     += batch_x.size(0)
                correct[0] += (logits_d.argmax(1) == batch_d).sum().item()
                correct[1] += (logits_g.argmax(1) == batch_g).sum().item()
                correct[2] += (logits_e.argmax(1) == batch_e).sum().item()

            scheduler.step()
            acc_d = correct[0] / max(total_, 1)
            acc_g = correct[1] / max(total_, 1)
            acc_e = correct[2] / max(total_, 1)
            avg_l = total_loss / max(len(train_loader), 1)

            # ── Валидация ────────────────────────────────────────
            val_d, val_g, val_e = self._eval_loader(val_loader)

            logger.info(
                f"[Classifier.train] epoch {epoch+1}/{epochs} "
                f"loss={avg_l:.4f} "
                f"train d/g/e={acc_d:.3f}/{acc_g:.3f}/{acc_e:.3f} "
                f"val d/g/e={val_d:.3f}/{val_g:.3f}/{val_e:.3f}"
            )
            metrics = {
                "domain_acc": acc_d, "group_acc": acc_g, "entity_acc": acc_e,
                "val_domain_acc": val_d, "val_group_acc": val_g,
                "val_entity_acc": val_e, "loss": avg_l,
            }

        self._model.eval()
        self._model_ready = True
        logger.info("[Classifier.train] complete → model_ready=True")

        if save_path:
            self._save_model(save_path)

        return metrics

    def _eval_loader(self, loader) -> Tuple[float, float, float]:
        """Evaluate accuracy on a DataLoader (no grad)."""
        if loader is None or not TORCH_AVAILABLE or self._model is None:
            return 0.0, 0.0, 0.0
        self._model.eval()
        correct = [0, 0, 0]
        total   = 0
        with torch.no_grad():
            for batch_x, batch_d, batch_g, batch_e in loader:
                ld, lg, le = self._model(batch_x)
                correct[0] += (ld.argmax(1) == batch_d).sum().item()
                correct[1] += (lg.argmax(1) == batch_g).sum().item()
                correct[2] += (le.argmax(1) == batch_e).sum().item()
                total      += batch_x.size(0)
        n = max(total, 1)
        self._model.train()
        return correct[0] / n, correct[1] / n, correct[2] / n

    def _expand_heads(self, new_model: "ConceptClassifierModel") -> None:
        """Copy old head weights into new_model for classes that already exist.

        Called when domain/group/entity tables grew since last train().
        New class rows are left at their random init values; existing rows
        get the pre-trained weights so the model is not reset from scratch.
        """
        if not TORCH_AVAILABLE or self._model is None:
            return
        head_pairs = [
            (self._model.domain_head,      new_model.domain_head),
            (self._model.group_head,       new_model.group_head),
            (self._model.entity_type_head, new_model.entity_type_head),
        ]
        for old_head, new_head in head_pairs:
            if old_head is None or new_head is None:
                continue
            old_w = old_head.weight.data   # (old_classes, hidden)
            new_w = new_head.weight.data   # (new_classes, hidden)
            n_copy = min(old_w.size(0), new_w.size(0))
            new_w[:n_copy, :] = old_w[:n_copy, :]
            if old_head.bias is not None and new_head.bias is not None:
                old_b = old_head.bias.data
                new_b = new_head.bias.data
                new_b[:min(old_b.size(0), new_b.size(0))] = \
                    old_b[:min(old_b.size(0), new_b.size(0))]
        # Shared layers: copy directly (dimensions unchanged)
        try:
            new_model.shared.load_state_dict(
                self._model.shared.state_dict(), strict=False)
        except Exception:
            pass  # mismatched shared dim on encoder change — fine, skip

    # ══════════════════════════════════════════════════════════════
    #  Внутренние методы
    # ══════════════════════════════════════════════════════════════

    def _model_classify(self, term: str, context: str,
                         source_type: str
                         ) -> Optional["ConceptClassifier.ClassifyResult"]:
        if not TORCH_AVAILABLE or self._model is None:
            return None
        encoder = self._get_encoder()
        text    = f"{term} [SEP] {context} [SRC] {source_type}"
        vec     = torch.tensor(encoder.encode(text),
                               dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits_d, logits_g, logits_e = self._model(vec)
        probs_d  = F.softmax(logits_d, dim=-1)[0]
        probs_g  = F.softmax(logits_g, dim=-1)[0]
        probs_e  = F.softmax(logits_e, dim=-1)[0]
        # Arithmetic mean of per-head max-probs gives a readable signal even
        # with many classes; geometric product collapses to near-zero at scale.
        conf = float((probs_d.max() + probs_g.max() + probs_e.max()) / 3.0)

        d_id = int(probs_d.argmax().item())
        g_id = int(probs_g.argmax().item())
        e_id = int(probs_e.argmax().item())

        sem  = self.hasher.hash_text(f"{term} {context}",
                                     self._get_embedder())
        ck   = CanonicalKey(domain_id=d_id, group_id=g_id,
                            entity_type_id=e_id, sem_hash=sem)

        return ConceptClassifier.ClassifyResult(
            domain      = self.domain_table.get_name(d_id),
            group       = self.group_table.get_name(g_id),
            entity_type = self.entity_table.get_name(e_id),
            confidence  = conf,
            canonical_key = ck.to_bytes(),
            aliases     = [],
            via_llm     = False,
        )

    def _llm_classify(self, term: str, context: str,
                       source_type: str,
                       llm_client) -> Optional["ConceptClassifier.ClassifyResult"]:
        prompt = self.LLM_CLASSIFY_PROMPT.format(
            term=term, context=context[:500], source_type=source_type,
            domains=", ".join(self.domain_table.all_names()[:40]),
            groups=", ".join(self.group_table.all_names()[:20]),
            entity_types=", ".join(self.entity_table.all_names()[:25]),
        )
        try:
            raw  = llm_client.complete(prompt)
            data = _parse_llm_json(raw)
        except Exception as e:
            logger.warning(f"[Classifier._llm_classify] LLM error: {e}")
            return None

        domain_name = data.get("domain", "unknown")
        group_name  = data.get("group", "unknown")
        etype_name  = data.get("entity_type", "unknown")
        confidence  = float(data.get("confidence", 0.5))
        aliases     = data.get("aliases", [])

        # Авто-регистрация новых значений в таблицах
        d_id = self.domain_table.register(domain_name)
        g_id = self.group_table.register(group_name)
        e_id = self.entity_table.register(etype_name)

        sem = self.hasher.hash_text(f"{term} {context}",
                                    self._get_embedder())
        ck  = CanonicalKey(domain_id=d_id, group_id=g_id,
                           entity_type_id=e_id, sem_hash=sem)
        key_bytes = ck.to_bytes()

        # Регистрируем синонимы с привязкой к домену
        all_aliases = [term] + aliases
        self.alias_table.bulk_register(all_aliases, key_bytes, domain=domain_name)

        return ConceptClassifier.ClassifyResult(
            domain=domain_name, group=group_name, entity_type=etype_name,
            confidence=confidence, canonical_key=key_bytes,
            aliases=aliases, via_llm=True,
        )

    def _unknown_result(self, term: str, context: str,
                         source_type: str,
                         embedder=None) -> "ConceptClassifier.ClassifyResult":
        sem = self.hasher.hash_text(f"{term} {context}", embedder)
        ck  = CanonicalKey(domain_id=0, group_id=0,
                           entity_type_id=0, sem_hash=sem)
        return ConceptClassifier.ClassifyResult(
            domain="unknown", group="unknown", entity_type="unknown",
            confidence=0.0, canonical_key=ck.to_bytes(),
            aliases=[], via_llm=False,
        )

    def _register_result(self, term: str,
                          result: "ConceptClassifier.ClassifyResult",
                          embedder=None) -> None:
        """Зарегистрировать alias с привязкой к домену."""
        self.alias_table.register(term, result.canonical_key,
                                  domain=result.domain)
        for a in result.aliases:
            self.alias_table.register(a, result.canonical_key,
                                      domain=result.domain)

    def _get_encoder(self):
        """Вернуть вызываемый энкодер (BERT или TF-IDF)."""
        if self._use_bert:
            return _BertEncoderWrapper(self._tokenizer, self._bert)
        return self._simple_enc

    def _get_embedder(self):
        """Вернуть объект с методом .encode() для SemanticHasher."""
        if self._use_bert:
            return _BertEncoderWrapper(self._tokenizer, self._bert)
        return None

    # ── Сохранение/загрузка ─────────────────────────────────────

    def _save_model(self, path: str) -> None:
        if self._model is None:
            return
        torch.save({
            "model_state":       self._model.state_dict(),
            "encoder_dim":       self._encoder_dim,
            "use_bert":          self._use_bert,
            "n_domains":         len(self.domain_table),
            "n_groups":          len(self.group_table),
            "n_types":           len(self.entity_table),
            "training_data_len": len(self.training_data),
        }, path)
        logger.info(f"[Classifier] model saved → {path}")

    def _load_model(self, path: str) -> None:
        # weights_only=True avoids arbitrary code execution via pickle
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        self._model = ConceptClassifierModel(
            encoder_dim=ckpt["encoder_dim"],
            n_domains=ckpt["n_domains"],
            n_groups=ckpt["n_groups"],
            n_types=ckpt["n_types"],
        )
        self._model.load_state_dict(ckpt["model_state"])
        self._model.eval()
        self._model_ready = True
        logger.info(f"[Classifier] model loaded from {path}")

    def save_state(self, directory: str) -> None:
        """Сохранить полное состояние: модель + таблицы + alias."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_lookup_tables(str(d / "lookup_tables.json"))
        with open(d / "alias_table.json", "w") as f:
            json.dump(self.alias_table.to_dict(), f, ensure_ascii=False, indent=2)
        with open(d / "training_data.pkl", "wb") as f:
            pickle.dump(self.training_data, f)
        if self._model and TORCH_AVAILABLE:
            self._save_model(str(d / "classifier_model.pt"))
        logger.info(f"[Classifier] full state saved → {directory}")

    def load_state(self, directory: str) -> None:
        """Загрузить полное состояние."""
        d = Path(directory)
        tables_path = d / "lookup_tables.json"
        if tables_path.exists():
            load_lookup_tables(str(tables_path))
        alias_path = d / "alias_table.json"
        if alias_path.exists():
            with open(alias_path) as f:
                self.alias_table = AliasTable.from_dict(json.load(f))
        data_path = d / "training_data.pkl"
        if data_path.exists():
            with open(data_path, "rb") as f:
                self.training_data = pickle.load(f)
        model_path = d / "classifier_model.pt"
        if model_path.exists() and TORCH_AVAILABLE:
            self._load_model(str(model_path))
        logger.info(f"[Classifier] full state loaded from {directory}")


# ── Вспомогательный класс для BERT ──────────────────────────────

class _BertEncoderWrapper:
    """Тонкая обёртка: BERT → encode(text) → np.ndarray."""

    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model     = model

    def encode(self, text: str) -> np.ndarray:
        if not TORCH_AVAILABLE:
            return np.zeros(768, dtype=np.float32)
        inputs = self._tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=128, padding=True)
        with torch.no_grad():
            out = self._model(**inputs)
        cls = out.last_hidden_state[:, 0, :].squeeze(0).numpy()
        return cls.astype(np.float32)


# ── Утилита разбора JSON из LLM ─────────────────────────────────

def _parse_llm_json(raw: str) -> Dict:
    clean = re.sub(r"```(?:json)?\s*", "", raw).replace("```", "").strip()
    brace = clean.find("{")
    if brace > 0:
        clean = clean[brace:]
    rbrace = clean.rfind("}")
    if rbrace >= 0:
        clean = clean[:rbrace + 1]
    return json.loads(clean)


# ══════════════════════════════════════════════════════════════════
#  ИНТЕГРАЦИЯ С NodeEdgeAssembler
#  Патч-функция: заменяет make_stable_id() на canonical key
# ══════════════════════════════════════════════════════════════════

def make_canonical_id(text: str,
                       context: str,
                       source_type: str,
                       source: str,
                       classifier: ConceptClassifier,
                       embedder=None,
                       llm_fallback=None) -> Tuple[bytes, "ConceptClassifier.ClassifyResult"]:
    """
    Главная точка входа вместо NodeEdgeAssembler.make_stable_id().

    Возвращает:
        canonical_id (bytes, 12 байт) — ключ узла
        ClassifyResult — домен, группа, тип, confidence

    Вызывается из IngestionPipeline при сборке каждого узла.
    """
    result = classifier.classify(
        term=text,
        context=context,
        source_type=source_type,
        embedder=embedder,
        llm_fallback=llm_fallback,
    )
    # Если alias уже зарегистрирован — вернём существующий ключ
    # (CrossDocumentLinker обработает merge)
    return result.canonical_key, result


# ══════════════════════════════════════════════════════════════════
#  ОБНОВЛЁННЫЙ NodeEdgeAssembler с поддержкой canonical key
# ══════════════════════════════════════════════════════════════════

class CanonicalNodeAssembler:
    """
    Расширенная версия NodeEdgeAssembler.
    Использует ConceptClassifier вместо MD5(source::text).

    Ключевые отличия:
      – ID = 12-байтный canonical key (не MD5 строки)
      – ObservationRecord создаётся для каждого нового источника
      – necessity_score доступен сразу при сборке
      – cross-domain агрегация встроена
    """

    def __init__(self,
                 classifier: ConceptClassifier,
                 embedder=None,
                 llm_fallback=None):
        self.classifier   = classifier
        self.embedder     = embedder
        self.llm_fallback = llm_fallback
        # Хранилище наблюдений: canonical_id_hex → List[ObservationRecord]
        self.observations: Dict[str, List[ObservationRecord]] = defaultdict(list)

    def assemble_node_with_canonical_key(
        self,
        item: Dict,
        source: str,
        timestamp: float,
        source_type: str = "unknown",
        source_trust: float = 0.7,
        physical_scores: Optional[Dict[str, float]] = None,
    ):
        """
        Собрать KnowledgeNode с canonical ID.
        В объединённой сборке KnowledgeNode и утилиты доступны напрямую.
        В автономном режиме импортируются из oracle1.
        """
        try:
            # In the merged single-file build these names are already in scope
            _KnowledgeNode     = KnowledgeNode          # noqa: F821
            _NodeFeatureBuilder = NodeFeatureBuilder    # noqa: F821
            _PhysSubEncoder    = PhysicalSubstrateEncoder  # noqa: F821
            _PhysAxisOrder     = PHYSICAL_AXIS_ORDER    # noqa: F821
        except NameError:
            from oracle1 import (KnowledgeNode as _KnowledgeNode,
                                 NodeFeatureBuilder as _NodeFeatureBuilder,
                                 PhysicalSubstrateEncoder as _PhysSubEncoder,
                                 PHYSICAL_AXIS_ORDER as _PhysAxisOrder)

        text    = item.get("text", "")
        context = item.get("full_context", "")

        # ── Классификация и получение canonical ID ───────────────
        cid_bytes, clf_result = make_canonical_id(
            text=text,
            context=context,
            source_type=source_type,
            source=source,
            classifier=self.classifier,
            embedder=self.embedder,
            llm_fallback=self.llm_fallback,
        )
        cid_hex = cid_bytes.hex()

        # ── Создание ObservationRecord ────────────────────────────
        obs = ObservationRecord.create(
            canonical_id=cid_bytes,
            source=source,
            timestamp=timestamp,
            source_type=source_type,
            trust=source_trust,
            delta_fields={k: v for k, v in item.items()
                          if k not in ("text", "full_context", "id")},
        )
        self.observations[cid_hex].append(obs)

        # ── Сборка KnowledgeNode ──────────────────────────────────
        node = _KnowledgeNode(
            id=cid_hex,
            text=text,
            full_text=context,
            node_type=item.get("node_type", "research"),
            domain=clf_result.domain,
            entity_type=clf_result.entity_type,
            provenance=source,
            timestamp=timestamp,
            publication_date=(str(item["publication_year"])
                              if item.get("publication_year") else None),
            scientific_score=float(item.get("scientific_score", 5.0)) * source_trust,
            investment_score=float(item.get("investment_score", 0.0)),
            social_score=float(item.get("social_score", 0.0)),
            maturity_score=float(item.get("maturity_score", 0.0)),
            strategic_value=float(item.get("strategic_value", 5.0)),
            efficiency_plateau=float(item.get("efficiency_plateau", 0.0)),
            dual_use_risk=float(item.get("dual_use_risk", 0.0)),
            legal_risk_score=float(item.get("legal_risk_score", 0.0)),
            export_control_risk=float(item.get("export_control_risk", 0.0)),
            solves_limitations=item.get("solves_limitations", []),
            requires_node_ids=item.get("requires_node_ids", []),
            enables_node_ids=item.get("enables_node_ids", []),
        )

        # Прикрепляем метаданные классификации в __dict__ (legacy) и native fields
        node.__dict__["canonical_key"]     = cid_bytes
        node.__dict__["clf_confidence"]    = clf_result.confidence
        node.__dict__["clf_via_llm"]       = clf_result.via_llm
        node.__dict__["observation_count"] = len(self.observations[cid_hex])
        # Native fields (available in merged oracle1 build)
        if hasattr(node, "observation_count"):
            node.observation_count = len(self.observations[cid_hex])
        if hasattr(node, "source_type"):
            node.source_type = source_type

        # ── Эмбеддинг и физические оси ───────────────────────────
        if self.embedder is not None:
            try:
                feat_builder = _NodeFeatureBuilder(self.embedder)
                phys_encoder = _PhysSubEncoder()
                base_vec     = feat_builder.build(node)
                if physical_scores:
                    clean_phys = {
                        ax: float(np.clip(physical_scores.get(ax, 0.5), 0.0, 1.0))
                        for ax in _PhysAxisOrder
                    }
                    node.embedding = phys_encoder.extend_node_features(
                        base_vec, clean_phys)
                    node.__dict__["raw_physical_scores"] = clean_phys
                    phys_vec = phys_encoder.build_physical_section(clean_phys)
                    node.__dict__["feasibility"] = phys_encoder.feasibility_score(
                        phys_vec)
                else:
                    node.embedding = base_vec

                # Обновляем семантический хэш с реальным эмбеддингом
                if node.embedding is not None:
                    sem = self.classifier.hasher.hash(node.embedding[:384])
                    ck  = CanonicalKey.from_bytes(cid_bytes)
                    ck.sem_hash = sem
                    node.__dict__["canonical_key"] = ck.to_bytes()

            except Exception as e:
                logger.warning(
                    f"[CanonicalAssembler] feature build failed for '{cid_hex}': {e}"
                )

        return node

    def get_necessity_score(self, term: str,
                             graph_nodes: Dict[str, Any]) -> float:
        """
        Измерить cross-domain необходимость термина.
        Прямой вызов к classifier.necessity_score().
        """
        return self.classifier.necessity_score(term, graph_nodes)

    def get_observations(self, canonical_id_hex: str) -> List[ObservationRecord]:
        return self.observations.get(canonical_id_hex, [])

    def observation_summary(self, canonical_id_hex: str) -> Dict[str, Any]:
        """Агрегированная статистика по всем наблюдениям концепта."""
        obs_list = self.observations.get(canonical_id_hex, [])
        if not obs_list:
            return {}
        trusts      = [o.trust for o in obs_list]
        timestamps  = [o.timestamp for o in obs_list]
        source_types = [o.source_type for o in obs_list]
        type_counts: Dict[str, int] = defaultdict(int)
        for st in source_types:
            type_counts[st] += 1
        return {
            "n_observations":  len(obs_list),
            "mean_trust":      float(np.mean(trusts)),
            "first_seen":      min(timestamps),
            "last_seen":       max(timestamps),
            "source_types":    dict(type_counts),
            "time_span_years": (max(timestamps) - min(timestamps)) / 31536000.0,
        }


# ══════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════

def _smoke_test_canonical_key():
    logging.basicConfig(level=logging.INFO)
    logger.info("=== Canonical Key Smoke Test ===")

    clf = ConceptClassifier()

    # Симулируем LLM-клиент: возвращает фиксированный JSON
    class MockLLM:
        _responses = {
            "лазер_physics": '{"domain":"optics","group":"fundamental_principle","entity_type":"physical_principle","confidence":0.95,"reasoning":"Laser is a core optical device","aliases":["laser","LASER","Light Amplification Stimulated Emission"]}',
            "лазер_fiction": '{"domain":"fiction","group":"fictional_concept","entity_type":"cultural_phantom","confidence":0.90,"reasoning":"Death ray in sci-fi","aliases":["death ray","laser weapon","heat ray"]}',
            "лазер_medicine":'{"domain":"medicine","group":"applied_technology","entity_type":"enabler_component","confidence":0.88,"reasoning":"Medical laser therapy","aliases":["medical laser","laser therapy","surgical laser"]}',
        }
        def __init__(self, key): self._key = key
        def complete(self, _prompt): return self._responses[self._key]

    # Один термин — три домена
    cases = [
        ("лазер", "когерентное излучение, инверсная населённость, квантовый переход", "arxiv",   MockLLM("лазер_physics")),
        ("лазер", "лучевое оружие, смерть врагов, фантастический роман",             "fiction",  MockLLM("лазер_fiction")),
        ("лазер", "хирургия сетчатки, офтальмология, клинические испытания",          "medical",  MockLLM("лазер_medicine")),
    ]

    results = []
    for term, ctx, stype, llm in cases:
        r = clf.classify(term, context=ctx, source_type=stype, llm_fallback=llm)
        logger.info(f"  [{stype:8s}] {r.describe()}")
        results.append(r)

    # Проверяем что все три — РАЗНЫЕ canonical_id (разные домены)
    keys = [r.canonical_key for r in results]
    assert len(set(k.hex() for k in keys)) == 3, \
        "FAIL: один термин в разных доменах должен давать разные ключи!"
    logger.info("✓ Разные домены → разные canonical keys")

    # Синоним должен давать ТОТ ЖЕ ключ что и оригинал в том же домене
    # "LASER" идёт через тот же LLM → тот же домен → bulk_register добавит alias
    # "Light Amplification..." тоже был в aliases первого вызова → уже зарегистрирован
    alias_direct = clf.alias_table.lookup("Light Amplification Stimulated Emission",
                                          domain="optics")
    assert alias_direct == keys[0], \
        f"FAIL: alias из LLM должен давать тот же ключ!\n" \
        f"  alias_lookup → {alias_direct.hex() if alias_direct else None}\n" \
        f"  'лазер'      → {keys[0].hex()}"
    logger.info("✓ LLM-alias 'Light Amplification Stimulated Emission' → тот же ключ (physics)")

    # Необходимость: 3 домена → высокий балл
    mock_nodes = {k.hex(): None for k in keys}
    score = clf.necessity_score("лазер", mock_nodes)
    logger.info(f"✓ Necessity score('лазер') = {score:.3f} (3 домена)")

    # Обучение классификатора (если PyTorch доступен)
    if TORCH_AVAILABLE and len(clf.training_data) >= 3:
        logger.info("Training classifier on 3 LLM examples...")
        metrics = clf.train(epochs=2)
        logger.info(f"✓ Train metrics: {metrics}")

        # Предсказание без LLM
        pred = clf.predict("laser", context="quantum optics stimulated emission")
        logger.info(f"✓ Predict (no LLM): {pred.describe()}")

    # CanonicalKey bytes/hex roundtrip
    ck = CanonicalKey.from_bytes(keys[0])
    assert ck.to_bytes() == keys[0], "FAIL: bytes roundtrip"
    logger.info(f"✓ CanonicalKey roundtrip: {ck.describe()}")

    logger.info("=== Smoke Test PASSED ===")


# standalone smoke test removed in merged build
#     _smoke_test_canonical_key()



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
    extended   = Oracle1Extended(model, graph)
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



# ══════════════════════════════════════════════════════════════════════════════
#  QUICK SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════


def trigger_maiman_event(system):
    graph = system["graph"]
    extended = system["extended"]
    topology = system["topology"]
    dynamics = system["dynamics"]
    
    logger.info("\n" + "💥"*30)
    logger.info("ACTIVATE CATALYST: THE MAIMAN MOMENT (MAY 1960)")
    logger.info("💥"*30)

    theory_id = "theory_1958"

    # 1. Создаем ЭНТРОПИЙНЫЙ ВЗРЫВ (Исходящие влияния)
    # Теория начинает "фонить" в прикладные области
    apps = ["surgery", "telecom", "cutting", "fusion", "lidar"]
    for i, app_name in enumerate(apps):
        app_id = f"app_{app_name}"
        graph.add_node(KnowledgeNode(
            id=app_id, 
            text=f"Laser application in {app_name}", 
            domain=app_name, 
            strategic_value=7.0
        ))
        # Создаем ребро с высокой социальной корреляцией
        e_out = KnowledgeEdge(
            id=f"e_out_{app_name}", 
            source_id=theory_id, 
            target_id=app_id, 
            semantic_similarity=0.4, 
            social_correlation=0.95
        )
        e_out.compute_total_weight()
        graph.add_edge(e_out)

    # 2. РЕСУРСНЫЙ УДАР (Пентагон)
    # investment_correlation передается внутрь KnowledgeEdge
    graph.add_node(KnowledgeNode(
        id="pentagon", 
        text="ARPA High-Power Beam Contract", 
        domain="military", 
        investment_score=10.0, 
        strategic_value=10.0
    ))
    
    e_money = KnowledgeEdge(
        id="e_money", 
        source_id="pentagon", 
        target_id=theory_id, 
        investment_correlation=1.0  # Теперь параметр на месте
    )
    extended.register_edge(e_money)

    # 3. ФОРСИРУЕМ "ЖАР" (Пробуждаем Эйнштейна)
    # Устанавливаем высокий прогнозный скор текущим узлам
    for nid in ["maser_1954", theory_id, "pentagon"]:
        node = graph.get_node(nid)
        if node: node.forecast_score = 9.9

    # 4. ФИНАЛЬНЫЙ ЦИКЛ СИНГУЛЯРНОСТИ
    logger.info("\n>>> COMMENCING TOTAL TOPOLOGICAL COLLAPSE <<<")
    
    for step in range(10):
        # Используем современное время, чтобы обойти штраф Dormancy
        current_ts = time.time() 
        
        # Ручная накачка "Интеллектуального шторма"
        # Имитируем аномальность (узел не похож на старые вакуумные лампы)
        system["topology"].anomaly_acc.push(theory_id, 1.0)
        
        # Имитируем конфликт данных (споры лабораторий Hughes vs Bell Labs)
        # В нашей реализации _cache — это словарь в StructuralConflictMeter
        system["topology"].conflict_meter._cache[theory_id] = 0.95
        # Имитируем высокую энтропию связей
        system["topology"].entropy_calc._cache[theory_id] = 0.90

        # Обновляем систему
        extended.update_epoch(current_ts)
        dynamics.run_epoch(current_ts)
        topo_res = topology.run_epoch(current_ts)

        # Прямое считывание Tension для мониторинга
        st_map = system["topology"].tension_field.compute(
            graph, 
            system["pressure_field"], 
            system["topology"].conflict_meter, 
            system["topology"].anomaly_acc, 
            system["topology"].entropy_calc, 
            current_ts
        )
        t_val = st_map[theory_id].tension
        
        logger.info(f"Final Step {step} | ⚡ Tension: {t_val:.4f} (Goal: 0.820)")

        if topo_res['rupture_count'] > 0 or t_val >= 0.82:
            logger.warning("\n" + "🎇"*30)
            logger.warning("THE ONTOLOGICAL RUPTURE IS COMPLETE!")
            logger.warning("THE REALITY HAS RECONFIGURED AROUND THE LASER")
            logger.warning("🎇"*30)
            logger.info(topology.rupture_report())
            
            # Извлекаем материализованный Лазер из синтезатора
            v_nodes, _ = dynamics.virtual_synth.get_virtual_nodes_as_graph_entries()
            if v_nodes:
                logger.info(f"✨ ORACLE-1 HAS SPOKEN. NEW NODE: {v_nodes[0].text}")
                logger.info(f"✨ Predicted Domain: {v_nodes[0].domain}")
                logger.info(f"✨ Scientific Readiness: {v_nodes[0].readiness_score:.2f}")
            break
    else:
        logger.error("Singularity failed. Check if Tension formula squares the value correctly.")



def run_real_historical_laser_path(system):
    graph = system["graph"]
    extended = system["extended"]
    topology = system["topology"]
    dynamics = system["dynamics"]
    
    logger.info("\n" + "📜"*30)
    logger.info("ORACLE-1: THE REAL LASER CHRONOLOGY (PREHISTORY TO 1960)")
    logger.info("📜"*30)

    # --- ТАЙМЛАЙН КОНСТАНТЫ ---
    TS_1800 = -5364792000.0
    TS_1917 = -1672531200.0
    TS_1954 = -504921600.0
    TS_1958 = -378691200.0
    TS_1960 = -315532800.0

    # ==========================================================================
    # PHASE 0: PREHISTORY (Культурный вектор и Математика)
    # ==========================================================================
    graph.add_node(KnowledgeNode(
        id="archimedes_myth", text="Archimedes Heat-Ray (Siege of Syracuse)",
        domain="military_history", entity_type="cultural_phantom", 
        social_gravity=8.0, phantom_weight=0.6, timestamp=TS_1800 - 1e10
    ))
    
    graph.add_node(KnowledgeNode(
        id="huygens_optics", text="Wave Theory of Light (Traite de la Lumiere)",
        domain="optics", entity_type="physical_principle", timestamp=TS_1800 - 1e9,
        scientific_score=9.0
    ))

    # ==========================================================================
    # PHASE 1: FUNDAMENTAL NODES (1801-1920)
    # ==========================================================================
    graph.add_node(KnowledgeNode(
        id="young_1801", text="Wave Interference principle",
        domain="physics", entity_type="physical_principle", timestamp=TS_1800,
        scientific_score=9.5
    ))

    graph.add_node(KnowledgeNode(
        id="maxwell_1865", text="Maxwell's Electromagnetic Theory",
        domain="physics", entity_type="unifying_theory", timestamp=TS_1800 + 2e9,
        scientific_score=10.0, strategic_value=8.0
    ))

    graph.add_node(KnowledgeNode(
        id="einstein_1917", text="Stimulated Emission Theory (A & B Coefficients)",
        domain="quantum_mechanics", entity_type="dormant_principle", timestamp=TS_1917,
        scientific_score=10.0, strategic_value=10.0, upstream_pressure=10.0
    ))

    # Ребра Фазы 1
    extended.register_edge(KnowledgeEdge(id="e_y_m", source_id="young_1801", target_id="maxwell_1865", semantic_similarity=0.9))
    extended.register_edge(KnowledgeEdge(id="e_m_e", source_id="maxwell_1865", target_id="einstein_1917", semantic_similarity=0.75))

    # ==========================================================================
    # PHASE 2: TECHNICAL & CULTURAL VECTORS (1920-1950)
    # ==========================================================================
    # Фантастика (Death Ray)
    graph.add_node(KnowledgeNode(
        id="garin_ray", text="Alexei Tolstoy: The Garin Death Ray (Crystalline Focus)",
        domain="fiction", entity_type="cultural_phantom", timestamp=TS_1917 + 3e8,
        social_gravity=9.5, phantom_weight=0.9
    ))

    # Промышленность (Рубин)
    graph.add_node(KnowledgeNode(
        id="ruby_synthesis", text="Industrial Verneuil Ruby Production",
        domain="materials", entity_type="industrial_substrate", timestamp=TS_1917 + 5e8,
        scientific_score=6.0, maturity_score=9.0
    ))

    # Радары (Когерентность)
    graph.add_node(KnowledgeNode(
        id="ww2_radar", text="WWII Radar: Coherent Microwave Mastery",
        domain="electronics", entity_type="technical_mastery", timestamp=TS_1954 - 3e8,
        scientific_score=8.5, strategic_value=10.0, efficiency_plateau=8.5
    ))

    # Ребра Фазы 2
    extended.register_edge(KnowledgeEdge(id="e_e_p", source_id="einstein_1917", target_id="garin_ray", social_correlation=0.8))
    extended.register_edge(KnowledgeEdge(id="e_r_m", source_id="ruby_synthesis", target_id="ww2_radar", investment_correlation=0.5))

    # ==========================================================================
    # PHASE 3: CONVERGENCE & BREAKTHROUGH (1954-1960)
    # ==========================================================================
    # МАЗЕР
    graph.add_node(KnowledgeNode(
        id="maser_1954", text="Ammonia MASER (Townes/Basov/Prokhorov)",
        domain="quantum_electronics", entity_type="breakthrough_node", timestamp=TS_1954,
        scientific_score=10.0, strategic_value=9.0
    ))

    # Теория Оптического Мазера
    theory_id = "theory_1958"
    graph.add_node(KnowledgeNode(
        id=theory_id, text="Optical Maser Theory (Schawlow-Townes Spec)",
        domain="optics", entity_type="functional_spec", timestamp=TS_1958,
        scientific_score=10.0, strategic_value=10.0, upstream_pressure=150.0
    ))

    # Пентагон
    graph.add_node(KnowledgeNode(
        id="pentagon_1959", text="Pentagon Coherent Light Funding ($15M)",
        domain="military_finance", entity_type="resource_source", timestamp=TS_1960 - 3e7,
        investment_score=10.0, strategic_value=10.0
    ))

    # Инкумбент (Вакуумные лампы)
    graph.add_node(KnowledgeNode(
        id="incumbent_radio", text="Vacuum Tube Microwave Industry",
        domain="electronics", strategic_value=10.0, efficiency_plateau=9.8
    ))

    # Ребра Фазы 3
    extended.register_edge(KnowledgeEdge(id="e_e_ma", source_id="einstein_1917", target_id="maser_1954", limitation_resolution=1.0))
    extended.register_edge(KnowledgeEdge(id="e_ma_t", source_id="maser_1954", target_id=theory_id, semantic_similarity=0.95))
    extended.register_edge(KnowledgeEdge(id="e_p_t", source_id="pentagon_1959", target_id=theory_id, investment_correlation=1.0))
    e_inh = KnowledgeEdge(id="e_inh", source_id="incumbent_radio", target_id=theory_id, inhibitory_force=0.95)
    e_inh.compute_total_weight()
    extended.register_edge(e_inh)
    extended.register_edge(KnowledgeEdge(id="e_t_ph", source_id=theory_id, target_id="garin_ray", social_correlation=1.0))

    # --- СИМУЛЯЦИЯ ФИНАЛЬНОГО СХЛОПЫВАНИЯ ---
    logger.info("\n>>> STARTING TOPOLOGICAL ANALYSIS: 1960.0 <<<")
    
    # Чтобы пробить разрыв в маленьком графе, активируем имитацию плотности
    for epoch in range(12):
        ts = TS_1960 + (epoch * 2592000)
        
        for nid in list(graph.nodes):
            node = graph.get_node(nid)
            node.forecast_score = 9.8
            if nid == theory_id:
                # Накачиваем аномальность и конфликт
                system["topology"].anomaly_acc.push(nid, 1.0)
                # Прямая инъекция в кеши для имитации тысяч фоновых связей
                system["topology"].conflict_meter._cache[nid] = 0.95
                system["topology"].entropy_calc._cache[nid] = 0.98

        ext_res = extended.update_epoch(ts)
        dyn_res = dynamics.run_epoch(ts)
        topo_res = topology.run_epoch(ts)

        # Мониторинг напряжения
        theory_state = topology.tension_field.compute(graph, system["pressure_field"], 
                                                      topology.conflict_meter, system["topology"].anomaly_acc, 
                                                      system["topology"].entropy_calc, ts)
        t_val = theory_state[theory_id].tension
        
        logger.info(f"Month {epoch} | Tension: {t_val:.4f} | Einstein Dormancy: {extended.dormancy_tracker.get_dormancy('maxwell_1865', 'einstein_1917'):.2f}")

        if topo_res['rupture_count'] > 0 or t_val > 0.82:
            logger.warning("\n" + "⚡"*30)
            logger.warning("ONTOLOGICAL RUPTURE DETECTED: THE REALITY COLLAPSE")
            logger.warning("⚡"*30)
            logger.info(topology.rupture_report())
            
            # Извлекаем предсказанное изобретение
            v_nodes, v_edges = dynamics.virtual_synth.get_virtual_nodes_as_graph_entries()
            if v_nodes:
                logger.info(f"✨ ORACLE-1 PREDICTED INVENTION: {v_nodes[0].text}")
                logger.info(f"✨ PHYSICAL FEASIBILITY: {v_nodes[0].readiness_score:.2f}")
            break

    logger.info("\n" + "█"*60)
    logger.info("HISTORICAL RECONSTRUCTION COMPLETE")
    logger.info("█"*60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Oracle-1 combined module loaded successfully.")

    system = build_oracle1_system()
    run_real_historical_laser_path(system)
    trigger_maiman_event(system)

    graph  = system["graph"]

    # Add two toy nodes
    node_a = KnowledgeNode(id="n1", text="stimulated emission principle",
                           domain="quantum optics", entity_type="physical_principle",
                           timestamp=1.0, scientific_score=8.0, strategic_value=7.0)
    node_b = KnowledgeNode(id="n2", text="population inversion in ruby crystal",
                           domain="crystal physics", entity_type="experimental_method",
                           timestamp=2.0, scientific_score=7.0, investment_score=5.0)
    graph.add_node(node_a); graph.add_node(node_b)

    edge_ab = KnowledgeEdge(id="e1", source_id="n1", target_id="n2",
                             semantic_similarity=0.8, temporal_proximity=0.9,
                             limitation_resolution=0.7, citation_link=0.5)
    edge_ab.compute_total_weight()
    graph.add_edge(edge_ab)

    extended = system["extended"]
    dynamics = system["dynamics"]
    topology = system["topology"]

    ts = time.time()

    metrics_ext  = extended.update_epoch(ts)
    metrics_dyn  = dynamics.run_epoch(ts)
    metrics_topo = topology.run_epoch(ts)

    logger.info(f"Extended: {metrics_ext}")
    logger.info(f"Dynamics: {metrics_dyn}")
    logger.info(f"Topology: {metrics_topo}")
    logger.info(topology.rupture_report())
    logger.info("Smoke test passed.")
# ══════════════════════════════════════════════════════════════════════════════
