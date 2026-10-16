"""Central configuration and derived runtime constants."""
from __future__ import annotations

from ._shared import *

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
        "qwen3.5:test",   # placeholder for a local Ollama test run — see below
    ],
    # default model name used when no client is supplied explicitly
    "llm_default_model": "DeepSeek-R1",

    # Context window (tokens) per model — the basis for ingestion sizing
    # (Manifest §2: "Документ загружается в LLM целиком ... благодаря
    # моделям с контекстом 128k+ токенов"). Used by
    # estimate_ingestion_char_budget() to decide how large a document can
    # go into a single ingestion pass instead of being chunked. All three
    # configured local models support at least 128k tokens; unrecognised
    # model names fall back to "llm_context_window_default" rather than a
    # small, chunking-happy value, since the manifest's premise is that a
    # large-context model is the norm, not the exception.
    "llm_context_windows": {
        "DeepSeek-R1":               128_000,
        "Llama 3.1 405B Instruct":   128_000,
        "Qwen3.5 122B":              256_000,
        # "qwen3.5:test" is a PLACEHOLDER KEY, not a real Ollama tag — this
        # dict is looked up by whatever string ends up as llm_client.model
        # (IngestionPipeline infers model_name from the constructed
        # client, not from this dict), which in turn is whatever
        # ORACLE_LLM_MODEL is set to. Rename this key to match your exact
        # `ollama list` tag (e.g. "qwen3.5:7b") — a mismatch here doesn't
        # error, it silently falls back to llm_context_window_default
        # (128k) below, which is the wrong number for a 256k test.
        # 256_000 is what "up to 256K" asks for; weak hardware may not
        # actually manage that (KV-cache memory scales with num_ctx
        # regardless of model size) — see get_llm_client()'s
        # ORACLE_LLM_NUM_CTX handling and _check_ollama_context_window()
        # before assuming this value is what's actually running.
        "qwen3.5:test":              256_000,
    },
    "llm_context_window_default": 128_000,

    "pressure_limitation_threshold":  0.3,
    "pressure_investment_threshold":  0.5,
    "pressure_invest_log_scale":      20.0,
    "pressure_zone_multiplier_scale": 0.5,
    # Cap on RuntimeGraph.combined_zone_multiplier() — several overlapping
    # temporal zones compound multiplicatively (Manifest §1); this bounds
    # how far that compounding can push acceleration_multiplier before it
    # saturates. Matches the [1.0, 5.0] range single zone_multiplier values
    # were already clipped to (see measured_zone_multiplier).
    "zone_multiplier_max": 5.0,
    "pressure_limitation_scale":      2.0,
    "pressure_unresolved_gate":       0.5,

    # ── Resource provenance overcounting (TODO #10) ───────────────────────────
    # When two nodes converging on the same target share an ancestor (a
    # common funding tranche, a common foundational breakthrough), summing
    # their reported magnitudes independently double-counts that shared
    # resource. compute_ancestor_overlap_discount() computes an exact (real
    # ancestor-set-intersection, not approximate-similarity) per-edge
    # discount used by ConvergencePressureField.excess_investment_pressure,
    # StructuralDependencyUpdater's sdi/casc scatter_add (via
    # graph_to_tensors' edge_provenance_discount), and
    # PathAgnosticInference.total_attention_flow.
    "provenance_ancestor_max_depth":       6,     # bounded BFS depth when walking DERIVED_FROM/ENABLES/CAUSED_BY lineage backward
    "provenance_ancestry_relationship_types": ["DERIVED_FROM", "ENABLES", "CAUSED_BY"],
    "provenance_overlap_discount_strength": 1.0,  # 0 = no discount ever, 1 = full linear discount proportional to ancestor overlap
    "provenance_min_discount":             0.15,  # floor so a shared distant ancestor never fully zeroes an edge's contribution

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
    "dormancy_wave_epsilon":        0.01,  # min |Δ dormancy_score| for update_wave() to keep relaying past a node (TODO #22)

    # ── Phantom node generator ────────────────────────────────────────────────
    "phantom_edge_prob_threshold":  0.70,
    "phantom_gap_threshold":        2.0,
    # Phantom bridges need at least one grounded fact/proof endpoint. A
    # small, configurable share may pair a fact/proof with an idea/theory.
    "phantom_min_fact_support":     0.40,
    "phantom_speculative_pair_rate": 0.10,
    "phantom_confirmation_sim":     0.75,
    "phantom_max_count":            50,
    "phantom_void_top_candidates":  50,
    # TODO #24: cap on how many nodes a single same-domain/-group cluster
    # contributes to _local_void_candidates()'s pool — without this, a
    # large, generic field (or, worse, the domain_id==0 "unclassified"
    # bucket) would silently turn the "scoped" search back into a
    # near-graph-wide one. Kept, not the whole cluster, when it's over cap.
    "phantom_domain_pool_cap":      200,
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

    # (TODO #18: "annotation_text_max_words" removed here — it was only ever
    # read by MotherAnnotator, deleted as confirmed-dead code; see the
    # comment where the class used to be, near IngestionPipeline.)

    # ── BucketIndex search ────────────────────────────────────────────────────
    "bucket_max_candidates": 20,   # top-k candidates sent to Pass-D LLM

    # ── ConceptClassifier ─────────────────────────────────────────────────────
    "classifier_confidence_threshold": 0.80,   # below → fall back to LLM
    "classifier_retrain_every":        500,    # new LLM samples between retrains
    "classifier_label_smoothing":      0.05,   # CrossEntropyLoss label smoothing
    "classifier_val_split":            0.15,   # fraction held out for validation
    "classifier_max_head_expand":      True,   # preserve old weights on head grow

    # ── Ingestion pipeline ────────────────────────────────────────────────────
    # Manifest §2, "Никакого DocumentChunker": documents are loaded into the
    # LLM whole, sized to the *active model's* real context window (see
    # "llm_context_windows" above and estimate_ingestion_char_budget()), not
    # to a fixed small constant. "ingestion_chunk_size" below is now only a
    # conservative FLOOR/fallback — used verbatim only when the LLM client
    # exposes no recognisable model name to size against. In normal
    # operation the effective budget is far larger (~128k-token models →
    # low hundreds of thousands of chars), so a document only actually gets
    # split when it genuinely exceeds the model's context, which should be
    # rare for the articles/papers/patents Oracle-1 ingests.
    "ingestion_chunk_size":   8_000,    # fallback max chars/chunk (~3000 tokens) when model is unknown
    "ingestion_chunk_overlap": 800,     # overlap between chunks (chars) — only matters on the rare fallback split path
    "ingestion_merge_threshold": 0.92,  # above → same entity (merge)
    # TODO #19: "ingestion_use_canonical_key" removed here — MD5(source+text)
    # identity (NodeEdgeAssembler.make_stable_id/assemble_node) was a remnant
    # of the old implementation kept alive behind this flag. node.id is now
    # the 12-byte CanonicalKey (hex) unconditionally — domain_id/group_id/
    # entity_type_id (ConceptClassifier, LLM-fallback, table grows
    # dynamically — see LookupTable.register()) + 6-byte semantic hash of the
    # embedding. IngestionPipeline no longer branches: only
    # CanonicalNodeAssembler.assemble_node_with_canonical_key() runs. The
    # domain/group/entity_type tables and aliases still resolve correctly
    # even before an LLM is available (see ConceptClassifier._unknown_result()
    # — collapses to domain_id=0 but the per-node sem_hash still differs, so
    # IDs stay distinguishable; they just don't carry real domain/group/
    # entity_type signal until a working LLM backend is wired in — see
    # LocalLLMClient). Confirmed dead as of this removal: NodeEdgeAssembler.
    # make_stable_id()/assemble_node() had no other caller.
    "ingestion_family_threshold": 0.85, # above (but below merge) → same "iceberg" family, still separate nodes
    "ingestion_link_threshold":  0.65,  # above → semantic cross-doc edge
    # TODO #16: how many hops RuntimeGraph.family_members()/
    # family_epistemic_mass() walk from a node along direct family links
    # before stopping. Deliberately bounded, not a full transitive closure —
    # see the comment on RuntimeGraph.union_family for the chaining risk a
    # single noisy embedding-similarity threshold creates at graph scale.
    # 1 = direct family only (most conservative, cheapest); 2 = also catches
    # "both matched a common hub phrasing but weren't compared to each other
    # directly" without walking arbitrarily far across the graph.
    "family_traversal_max_depth": 2,
    "ingestion_source_trust": {
        "arxiv":      0.85, "patent":    0.80, "nature":   0.95,
        "science":    0.95, "preprint":  0.65, "news":     0.40,
        "blog":       0.30, "report":    0.70, "book":     0.75,
        "forum":      0.25,
        "conference": 0.80, "unknown":   0.50,
    },

    # ── Consolidation engine ──────────────────────────────────────────────────
    "consolidation_conflict_variance_threshold": 2.0,  # std above this → conflict (0-10 score fields)
    "consolidation_physical_conflict_threshold": 0.20, # std above this → conflict for 0-1 physical axes
    # TODO #9 fix: this used to be applied as a *flat* prior trust on every
    # apply_to_graph() call, making whichever source got ingested first
    # "sticky" regardless of its own trust (see test_consolidation_order.py —
    # a trust=1.0 flat prior produced a ~2-point swing on a 0-10 field
    # depending purely on ingestion order). GraphConsolidator.apply_to_graph()
    # now instead carries forward the actual combined trust weight recorded
    # in consolidation_meta["sum_real_trust"] when available (which makes the
    # result order-independent — verified for up to 3 sessions / 6 orderings).
    # This constant survives as the fallback ONLY for nodes/edges with no
    # recorded provenance (written directly, never through a consolidator) —
    # there we have no information to do better, so we keep the original
    # conservative (maximally-protective) value.
    "consolidation_graph_prior_trust": 1.0,  # fallback trust weight when a node/edge has no consolidation_meta

    # TODO #12 (found auditing #9): sum_real_trust (the running sufficient
    # statistic for TRUST_WEIGHTED_MEAN, see #9 above) grows without bound —
    # every session's trust adds in, whether it's a genuinely independent
    # corroboration or the Nth low-trust blog repeating the same original
    # claim. After enough low/medium-trust sessions, even a single new
    # trust=0.95 (nature/science) correction can barely move the value
    # (verified: 40 sessions at trust=0.5 -> a trust=0.95 correction moves
    # the value only ~4.5% of the way toward its own reading). This is real
    # and still open.
    #
    # A first attempt capped sum_real_trust at this value with min(cap, ...)
    # directly in GraphConsolidator.apply_to_graph(). That attempt was
    # reverted (found auditing #12 itself): capping an *incrementally
    # accumulating* weight and then feeding the capped figure back in as the
    # next session's prior weight is provably order-dependent — the exact
    # defect #9 exists to eliminate, just bounded to a smaller magnitude
    # instead of eliminated. Concretely, with cap=5 and three trust=3
    # sessions reporting values 10, 10, 1: order (10,10,1) converges to
    # 6.625, order (1,10,10) converges to 7.1875 — a real, order-dependent
    # discrepancy despite the cap being active in both runs (see
    # test_consolidation_order.py). The root cause: capping breaks the
    # invariant capped_weight * value == true_partial_sum that #9's
    # induction relies on, and how much that mismatch distorts the final
    # result depends on *which* observations happen to fall inside vs.
    # outside the discounted window — i.e. on arrival order. No pure
    # O(1)-state (single scalar weight) incremental scheme can enforce a
    # hard ceiling on accumulated weight AND stay exactly order-independent;
    # the two goals are in tension by construction, not by a fixable bug.
    #
    # sum_real_trust is therefore uncapped again in apply_to_graph() and this
    # constant is currently NOT read by the merge path — #9's exact
    # order-independence guarantee takes priority over #12's over-anchoring
    # fix until one of these is implemented properly:
    #   (a) exponential time-decay — recency-weighted by design, which means
    #       explicitly giving up strict order-independence (not just bounding
    #       it) and relaxing test_consolidation_order.py's guarantee to match;
    #   (b) source-domain deduplication at Observation.source, reusing TODO
    #       #10's overlap-discount idea so that N low-trust sources repeating
    #       the same original claim don't count as N independent
    #       corroborations in the first place — this fixes the actual root
    #       cause (non-independent evidence) rather than symptom-patching the
    #       running weight, and unlike (a) it can likely stay exactly
    #       order-independent since it discounts individual Observation.trust
    #       at ingestion time rather than the running aggregate.
    # Left unset/unused rather than half-applied; do not re-wire this
    # constant into apply_to_graph() without also re-deriving whether the
    # chosen approach preserves order-independence, and extending
    # test_consolidation_order.py to actually exercise the capped regime
    # (the existing tests never accumulate enough trust to reach this cap).
    "consolidation_sum_real_trust_cap": 5.0,

    # ── Edge semantic validation (optional LLM pass) ─────────────────────────
    # Master switch. Runs a second LLM call per batch of extracted edges to
    # catch cross-domain nonsense connections (e.g. fictional political
    # events linked to real scientific discoveries) before they reach the
    # graph. TODO #7: was False (silently opt-out, meaning unfiltered
    # first-pass LLM extraction went straight into the graph). Now that #1
    # wires IngestionPipeline into train.py for the first time, defaulting
    # this on means the first real ingestion run is validated rather than
    # raw — the deliberate trade-off is roughly double the LLM call volume
    # during ingestion. Set back to False (or pass --disable_edge_validation
    # to train.py) if LLM cost/latency is a binding constraint for your run.
    "edge_validation_enabled":        True,
    # Minimum plausibility score returned by LLM to keep the edge.
    # Edges below this threshold are dropped and logged.
    "edge_validation_min_score":      0.40,
    # Maximum edges per single LLM call (batching reduces API overhead).
    "edge_validation_batch_size":     12,
    # Which edge categories to validate.  Remove entries to skip that pass.
    #   "intra_doc"  — edges within a single document (Pass C)
    #   "cross_doc"  — NO LONGER CHECKED by _process_chunk: Pass D's own
    #                  batched call now returns "plausibility"/"verdict"
    #                  for each match in the same response that decides
    #                  the match itself (see _run_pass_d_batch), so the
    #                  separate LLM pass this used to trigger is gone —
    #                  keeping "cross_doc" in this list is harmless but has
    #                  nothing left to do at chunk level. Left documented
    #                  rather than deleted from the enum in case a future
    #                  pass wants the name back for something else.
    #   "graph_load" — final gate in load_result_into_graph before add_edge(),
    #                  independent of the above — still the place to add an
    #                  extra cross_doc_edges safety net if you want one.
    "edge_validation_passes":         ["intra_doc"],
    # If True, edges that time out or cause LLM errors are KEPT (fail-open).
    # If False, they are dropped (fail-closed / conservative mode).
    "edge_validation_fail_open":      True,

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

    # ── Promoted from hardcoded class-level constants ──────────────────────
    # These used to be plain ALL_CAPS attributes on their owning class,
    # invisible to anyone tuning the system from CONFIG. Left in place:
    # NodeFeatureBuilder's *_DIM slices, PQ_BYTES/PQ_SUBVECS (fixed by the
    # product-quantization scheme), TOPOLOGY_FEATURE_DIM, and EpistemicLevel's
    # member values (an Enum's values aren't meant to be swapped at runtime) —
    # those are schema, not tuning.
    "dormancy_semantic_twin_threshold":       0.75,
    "dormancy_semantic_twin_max_candidates":  2000,   # cap on the brute-force similarity scan
    "retrospective_loss_max_paths":           100.0,
    "sparsity_weak_pressure_threshold":       0.10,   # below this void_pressure → weak
    "sparsity_weak_forecast_threshold":       1.0,    # below this forecast_score → weak
    "sparsity_dense_degree_threshold":        30,     # edges above this → apply branch cap
    "sparsity_max_dense_neighbours":          10,     # top-K neighbours kept for dense nodes
    "sparsity_branch_floor":                  1,
    "scenario_coherence_w_path":              0.35,
    "scenario_coherence_w_temporal":          0.30,
    "scenario_coherence_w_domain":            0.20,
    "scenario_coherence_w_confidence":        0.15,
    "llm_annotator_max_retries":              3,
    "llm_annotator_retry_delay_s":            2.0,
    "edge_validator_max_tokens":              1024,
    # TODO #21: "continuous_feed_k_hop" removed — ContinuousFeedEngine.feed()
    # no longer bounds the affected wave by a fixed hop radius. See
    # continuous_feed_tension_epsilon below.
    "continuous_feed_tension_epsilon":        0.02,   # min |Δ local tension| for the wave to keep relaying past a node
    "continuous_feed_eta_continuous":         1e-6,
    "continuous_feed_max_steps":              50,
    "hard_negative_fraction":                 0.5,    # share of negatives mined as hard (see OracleDataset.from_graph)
    "hard_negative_sim_threshold":            0.6,
    "hard_negative_max_embedded":             2000,   # cap on the O(n²) similarity matrix
    "oracle_dataset_val_temporal_frac":       0.15,   # newest slice of the graph's timeline held out for validation (see train.py build_dataset_from_graph)
    "interval_min_span_days":                 30.0,   # point vs interval cutoff, see InfluenceTimeline.temporal_kind
    "concept_distiller_similarity_threshold": 0.55,   # минимальная близость для проверки
    "concept_distiller_max_candidates":       5,      # сколько canonical узлов проверять
    "significance_min_mass_for_proven":       3.0,
    "significance_reinforcement_fraction":    0.5,    # доля источников, идущих в reinforcement
    "bottleneck_tension_threshold":           0.55,
    "intervention_default_noise_sigma":       0.20,   # 20%
    "intervention_default_simulations":       200,
    "intervention_fundamental_investment_dampening": 0.15,  # money buys engineering effort, not a scheduled discovery
    "cascade_fundamental_cross_domain_boost":  1.8,   # fundamental discoveries jump between domains more readily than combinatorial ones

    # ── Dependency-closure traversal (GraphTraversalEngine.find_unresolved_dependencies) ──
    "dependency_closure_max_hops":  6,     # requires_node_ids depth cap, same role as phantom_rupture_proximity_hops
    "dependency_closure_max_nodes": 500,   # visited-node budget per call — same "catastrophic at scale" concern as has_incoming_edge's O(1) cache
    "dependency_maturity_gate":     6.0,   # maturity_score (0-10) above which a combinatorial node counts as actualized rather than idea-stage.
                                            # Starting guess, not calibrated: HeuristicLLMClient adds 3.5 per matched deploy-keyword ("prototype",
                                            # "deploy", ...), so this requires more than one weak signal before something counts as solved. Tune
                                            # against real ingested data before trusting the diagnosis.

    # ── LLM call-count optimisation ─────────────────────────────────────────
    # Pass A+B+C merged into one call per chunk instead of three sequential
    # ones (B and C each only ever depended on A's own output, never on each
    # other). Kept behind a flag rather than deleting the old 3-call path:
    # the merge trades 2 fewer calls for a longer single generation (higher
    # truncation/malformation surface — see _recover_partial_array), and
    # that trade hasn't been measured yet on real documents. Compare the
    # two by re-ingesting the same corpus with this flipped and diffing (a)
    # how often ResponseValidator falls back to a 5.0 default per score
    # field, (b) how often an "evidence" quote fails to appear verbatim in
    # the source text.
    "ingestion_unified_extraction": True,
    # Batch size for Pass D (secondary-context arbitration), same role as
    # edge_validation_batch_size. Items carry a full candidate list (up to
    # bucket_max_candidates=20 each) so this is kept smaller than that
    # batch's 12 to bound prompt size.
    "pass_d_batch_size": 8,
    # Skip Pass D's LLM identity-arbitration call entirely and auto-accept
    # the top candidate when its embedding similarity is at or above this
    # threshold — mirrors the threshold BucketIndex.find_connections already
    # uses to merge same-document near-duplicates without any LLM. Off by
    # default (>1.0 never fires): auto-accepting skips the chronological/
    # plausibility check the batched LLM call would otherwise still run,
    # which similarity alone cannot catch (two entities can embed as near-
    # identical and still have an impossible causal chronology). Lower it
    # deliberately only once you've decided that trade is worth the extra
    # speed at your volume.
    "pass_d_auto_accept_threshold": 1.01,
    # Cap on how many earlier-chunk nodes are listed verbatim in a later
    # chunk's prompt (see IngestionPromptBuilder's "ALREADY-EXTRACTED
    # NODES" block) when a document was split by the fallback path. Keeps
    # the most RECENT entries when exceeded (nearby context is more likely
    # relevant than something from 50 chunks back) rather than growing
    # unboundedly across a book-length document.
    "ingestion_carryforward_max_nodes": 200,
}


def _apply_config_overrides() -> None:
    override_path = os.environ.get("ORACLE_CONFIG_OVERRIDES", "").strip()
    if not override_path:
        return
    with open(override_path, encoding="utf-8-sig") as handle:
        overrides = json.load(handle)
    if not isinstance(overrides, dict):
        raise ValueError("ORACLE_CONFIG_OVERRIDES must contain a JSON object")

    def merge(target: Dict[str, Any], patch: Dict[str, Any], prefix: str = "") -> None:
        for key, value in patch.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in target:
                raise ValueError(f"Unknown Oracle config key: {path}")
            current = target[key]
            if isinstance(current, dict):
                if not isinstance(value, dict):
                    raise ValueError(f"Oracle config value {path} must be an object")
                merge(current, value, path)
            elif isinstance(current, tuple):
                if not isinstance(value, (list, tuple)):
                    raise ValueError(f"Oracle config value {path} must be an array")
                target[key] = tuple(value)
            elif isinstance(current, bool):
                if not isinstance(value, bool):
                    raise ValueError(f"Oracle config value {path} must be boolean")
                target[key] = value
            elif isinstance(current, int):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"Oracle config value {path} must be an integer")
                target[key] = value
            elif isinstance(current, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Oracle config value {path} must be a finite number")
                target[key] = float(value)
            elif not isinstance(value, type(current)):
                raise ValueError(f"Oracle config value {path} must be {type(current).__name__}")
            else:
                target[key] = value

    merge(CONFIG, overrides)
    logger.info("[CONFIG] Runtime overrides loaded from %s", override_path)


_apply_config_overrides()


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

__all__ = ['CONFIG', '_apply_config_overrides', 'EDGE_WEIGHT_COMPONENTS', 'V6_EDGE_WEIGHTS', 'V6_READINESS_WEIGHTS', 'N_EDGE_COMPONENTS', 'N_READINESS_DIMS', 'N_ACCELERATION_DIMS', 'N_STRUCTURAL_DIMS', 'EXTENDED_EDGE_COMPONENTS', 'N_EXTENDED_COMPONENTS', 'INHIBITORY_PENALTY_CAP', 'INHIBITORY_WEIGHT', 'ATTENTION_GAMMA', 'COLLAPSE_THRESHOLD', 'VIRTUAL_NODE_TENSION_THRESHOLD', 'ALERT_OVERLOAD', 'ALERT_VOID_COLLAPSE', 'ALERT_DOMAIN_TRANSPLANT', 'ALERT_ANOMALY_SATURATION', 'ALERT_ONTOLOGICAL_RUPTURE', 'PHASE_THRESHOLDS', 'PHYSICAL_SUBSTRATE_DIM', 'PHYSICAL_AXIS_ORDER']
