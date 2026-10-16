# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Neural model components and their training losses."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, N_EXTENDED_COMPONENTS, V6_EDGE_WEIGHTS
from .entities import EdgeFeatureBuilder, NodeFeatureBuilder, TemporalEncoder
from .forecasting import RetrospectiveConvergenceLoss
from .graph_dynamics import ContextualPriorityHead

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
        # Manifest §1: "Ребро — это 7-мерный тензор, а не скаляр" — the 6
        # conductive EDGE_WEIGHT_COMPONENTS plus inhibitory_force. One head
        # per EXTENDED_EDGE_COMPONENTS entry (7), not the stale 6-dim
        # N_EDGE_COMPONENTS — see OracleDataset.from_graph's 7-dim
        # label_components and run.py's mode_edge, which already zips
        # EXTENDED_EDGE_COMPONENTS against these predictions.
        self.component_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim // 2, 32), nn.GELU(),
                          nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(N_EXTENDED_COMPONENTS)
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
    """Auxiliary contextual-influence estimator for the Oracle graph.

    This module does not represent Oracle-1's world model and does not learn
    the future from raw text. The persisted RuntimeGraph is primary: interval
    nodes, density curves, temporal zones, provenance and seven-component
    edges are its state. This neural component only learns how local context
    modulates edge existence/components. A retrospective outcome dataset is
    required before a forecast head can be called decision-grade.
    """
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
               topology_vecs=None, edge_provenance_discount=None):
        # topology_vecs: [N, 12] tensor of LocalTopologyStats vectors, one per node.
        # When provided, edge-level priority weights are computed via priority_head
        # and passed to each GNN layer to modulate attention (Fix 1 + Fix 4).
        # edge_provenance_discount: TODO #10, [E] tensor row-aligned with
        # edge_index (see graph_to_tensors/compute_ancestor_overlap_discount).
        # Down-weights sdi/casc's scatter_add for edges whose source shares
        # ancestry with a sibling edge converging on the same target, so a
        # shared upstream cause (funding tranche, foundational breakthrough)
        # isn't counted once per downstream path. None/omitted = no discount
        # (all edges weighted 1.0), preserving old behaviour for any caller
        # not yet passing it.
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
            if edge_provenance_discount is not None and edge_provenance_discount.numel() == edge_weights.numel():
                edge_weights = edge_weights * edge_provenance_discount.to(edge_weights.device)
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

    def forward(self, src_features, tgt_features):
        """
        Pair-level forward path for OracleDataset-style batches (train.py's
        train_one_epoch()/validate() — see Bug 2 in the audit notes: those
        called model(**batch) against a model with no forward() at all).

        Deliberately does NOT route through encode(), the multi-layer GNN
        path: encode() needs edge_index/edge_attr/containment structure for
        MultiComponentEdgeAttention to attend over, but OracleDataset.
        from_graph() samples independent (src, tgt) node pairs with no such
        local adjacency attached — mirroring exactly why
        TemporalSplitTrainer._local_step() builds its own small subgraph and
        calls encode() explicitly there instead of relying on a shared
        forward(). Here there is no subgraph, just two raw feature vectors,
        so input_proj (the same projection encode() itself starts from) is
        the correct depth: enough to give predict_edge two comparable
        latents, not a claim that this replaces graph-contextualised
        encoding.

        Returns a dict with "exist_probs"/"pred_comps" — deliberately not
        the full Oracle1Loss.total() surface (pred_forecast, node_latents,
        zone_memberships, ...), since pair-sampled examples carry no
        forecast or zone-membership supervision for those terms. Callers
        pair this with existence_loss()+lc*component_loss() directly,
        exactly as TemporalSplitTrainer._local_step() already does.
        """
        if not TORCH_AVAILABLE:
            return {"exist_probs": None, "pred_comps": None}
        h_src = self.input_proj(src_features)
        h_tgt = self.input_proj(tgt_features)
        exist_probs, pred_comps = self.predict_edge(h_src, h_tgt)
        return {"exist_probs": exist_probs, "pred_comps": pred_comps}


# Truthful name for new integrations. Oracle1Model stays for compatibility
# with existing local checkpoints and scripts.
ContextInfluenceEstimator = Oracle1Model


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

__all__ = ['TemporalEncoder', 'MultiComponentEdgeAttention', 'StructuralDependencyUpdater', 'ForecastHead', 'EdgePredictionHead', 'Oracle1Model', 'ContextInfluenceEstimator', 'Oracle1Loss']
