"""Graph dataset construction and temporal model training."""
from __future__ import annotations

from ._shared import *
from .config import CONFIG, EXTENDED_EDGE_COMPONENTS, N_EXTENDED_COMPONENTS
from .entities import EdgeFeatureBuilder, NodeFeatureBuilder
from .graph_dynamics import (
    ContextualPriorityHead,
    ConvergencePressureField,
    DormancyTracker,
    LocalTopologyStats,
    PRESSURE_COMPUTER,
)
from .neural import Oracle1Loss, Oracle1Model
from .tensorization import graph_to_tensors

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
        # Sample negative pairs (non-edges): a mix of uniformly-random
        # negatives ("easy" — almost always unrelated pairs, e.g. a
        # compression algorithm and Inca mythology) and HARD negatives
        # (embeddings that already look related — semantic_similarity above
        # a threshold — but with no actual topological edge). Pure random
        # sampling teaches the model to reject obviously-unrelated pairs
        # within a few steps and then stops improving on anything useful;
        # the hard negatives are exactly the interdisciplinary near-misses
        # Oracle is supposed to be good at judging — Structural Vacuum
        # candidates, not noise. Easy negatives are kept too, not replaced:
        # without them the model has nothing anchoring it against a
        # degenerate "everything is somewhat related" collapse.
        HARD_NEG_FRACTION      = CONFIG["hard_negative_fraction"]
        HARD_NEG_SIM_THRESHOLD = CONFIG["hard_negative_sim_threshold"]
        HARD_NEG_MAX_EMBEDDED  = CONFIG["hard_negative_max_embedded"]  # cap the O(n²) similarity matrix

        rng = np.random.default_rng(42)
        n_neg = min(len(positive_pairs) * 3, max_node_pairs - len(positive_pairs))
        n_neg = max(0, n_neg)
        negative_pairs: List[Tuple[str, str]] = []

        if n_neg > 0 and len(node_ids) >= 2:
            n_hard = int(n_neg * HARD_NEG_FRACTION)

            embedded_ids = [nid for nid in node_ids if graph.nodes[nid].embedding is not None]
            if len(embedded_ids) > HARD_NEG_MAX_EMBEDDED:
                embedded_ids = list(
                    rng.choice(embedded_ids, HARD_NEG_MAX_EMBEDDED, replace=False)
                )
            if len(embedded_ids) >= 2 and n_hard > 0:
                emb_mat = np.stack([graph.nodes[nid].embedding for nid in embedded_ids])
                sims = cos_sim(emb_mat, emb_mat)
                np.fill_diagonal(sims, -1.0)
                idx_i, idx_j = np.where(sims >= HARD_NEG_SIM_THRESHOLD)
                candidates = [
                    (embedded_ids[i], embedded_ids[j], float(sims[i, j]))
                    for i, j in zip(idx_i.tolist(), idx_j.tolist()) if i < j
                ]
                candidates.sort(key=lambda c: -c[2])
                for src_c, tgt_c, _ in candidates:
                    if len(negative_pairs) >= n_hard:
                        break
                    pair = (src_c, tgt_c)
                    if pair not in existing_edge_keys and (tgt_c, src_c) not in existing_edge_keys:
                        negative_pairs.append(pair)

            # Easy/random negatives fill the rest.
            attempts = 0
            remaining = max(1, n_neg - len(negative_pairs))
            neg_set = set(negative_pairs)
            while len(negative_pairs) < n_neg and attempts < remaining * 10:
                attempts += 1
                i, j = rng.choice(len(node_ids), 2, replace=False)
                pair = (node_ids[i], node_ids[j])
                if pair not in existing_edge_keys and pair not in neg_set:
                    negative_pairs.append(pair)
                    neg_set.add(pair)

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

def _local_gradient_step(
    model: "Oracle1Model", loss_fn: "Oracle1Loss", local_graph: "RuntimeGraph",
    device, feature_builder: Optional["NodeFeatureBuilder"] = None,
    edge_builder: Optional["EdgeFeatureBuilder"] = None, optimizer=None,
) -> Optional[float]:
    """
    Shared by TemporalSplitTrainer._local_step() and ContinuousFeedEngine.feed()
    — one forward pass restricted to `local_graph`, existence+component loss
    against its actual edges, and (when `optimizer` is given) one
    backward()+optimizer.step(). Returns the scalar loss value, or None if
    there was nothing to learn from (subgraph too small, or encode/predict
    degrade to their TORCH_AVAILABLE=False stubs).

    Pulled out to a module-level function rather than duplicated in both
    callers — the two call sites differ only in how they arrive at
    `local_graph` (a k-hop wave around a snapshot diff vs. a k-hop wave
    around freshly-fed nodes), not in what happens once they have it.
    """
    tensors = graph_to_tensors(local_graph, device, node_fb=feature_builder, edge_fb=edge_builder)
    if tensors is None:
        return None
    node_ids = tensors["node_ids"]
    if len(node_ids) < 2:
        return None
    node_idx = {nid: i for i, nid in enumerate(node_ids)}

    model.train()
    node_latents, _ = model.encode(
        tensors["node_features"], tensors["edge_index"], tensors["edge_attr"],
        tensors["containment_index"], tensors["containment_zone_mult"],
        tensors["node_years"], tensors["node_zone_mult"],
        topology_vecs=tensors.get("topology_vecs"),
        edge_provenance_discount=tensors.get("edge_provenance_discount"),
    )
    if node_latents is None:
        return None

    positive_pairs = [(s, t) for (s, t), e in local_graph.edges.items()
                       if not e.is_containment_edge]

    # Negative pairs: random non-edges among the SAME local nodes.
    # Deliberately simpler than OracleDataset.from_graph's hard/easy split —
    # this is a small, frequent, local update meant to keep the existence
    # head honest about *this* neighbourhood, not a from-scratch training
    # pass that needs to teach global discrimination against unrelated
    # domains.
    existing = set(positive_pairs)
    negative_pairs: List[Tuple[str, str]] = []
    n = len(node_ids)
    if n >= 2:
        target_neg = min(max(len(positive_pairs) * 2, 4), n * (n - 1))
        rng = np.random.default_rng(abs(hash(tuple(node_ids))) % (2**32))
        attempts = 0
        while len(negative_pairs) < target_neg and attempts < target_neg * 10:
            i, j = rng.integers(0, n, size=2)
            if i != j:
                a, b = node_ids[int(i)], node_ids[int(j)]
                if (a, b) not in existing and (a, b) not in negative_pairs:
                    negative_pairs.append((a, b))
            attempts += 1

    pairs = positive_pairs + negative_pairs
    if not pairs:
        return None

    src_idx = torch.tensor([node_idx[s] for s, _ in pairs], dtype=torch.long, device=device)
    tgt_idx = torch.tensor([node_idx[t] for _, t in pairs], dtype=torch.long, device=device)
    h_src = node_latents[src_idx]
    h_tgt = node_latents[tgt_idx]

    exist_probs, pred_comps = model.predict_edge(h_src, h_tgt)
    if exist_probs is None:
        return None

    exist_labels = torch.zeros(len(pairs), dtype=torch.float32, device=device)
    # 7-dim, matching EdgePredictionHead.component_heads (was N_EDGE_COMPONENTS
    # == 6, which excluded inhibitory_force and would mismatch pred_comps' width).
    true_comps = torch.zeros(len(pairs), N_EXTENDED_COMPONENTS, dtype=torch.float32, device=device)
    edge_mask  = torch.zeros(len(pairs), dtype=torch.bool, device=device)
    for i, (s, t) in enumerate(pairs):
        edge = local_graph.edges.get((s, t))
        if edge is not None:
            exist_labels[i] = 1.0
            edge_mask[i]    = True
            true_comps[i]   = torch.tensor(
                [getattr(edge, c) for c in EXTENDED_EDGE_COMPONENTS],
                dtype=torch.float32, device=device)

    loss = (loss_fn.existence_loss(exist_probs, exist_labels)
            + loss_fn.lc * loss_fn.component_loss(pred_comps, true_comps, edge_mask))

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float(loss.item())


class TemporalSplitTrainer:
    """
    Manifest §5, "Инкрементальное обучение (Online Learning)": adding a new
    node to the graph should not trigger retraining over the whole graph.
    train_epoch() is that mechanism — for each (snap_t, snap_td) pair (the
    graph before/after some ingestion delta — see RuntimeGraph.as_of() for
    how to produce these from a single live graph at two timestamps, or
    load_temporal_window() for two persisted windows), it identifies what
    actually changed, takes the k-hop neighbourhood of the change as the
    "wave" (RuntimeGraph.k_hop_neighborhood/subgraph), and runs one
    forward+backward(+optimizer.step, when an optimizer is supplied)
    restricted to that local subgraph — never a full-graph forward pass.

    This is deliberately a lighter-weight, more frequent sibling of
    train.py's Phase-2 batch training, not a replacement for it: a
    randomly-initialised model still needs many full passes over a
    representative sample of the graph to learn anything useful (that's
    what Phase 2 is for — see train.py). TemporalSplitTrainer is for what
    happens *after* that bootstrap, as the graph keeps growing: each new
    increment of real ingestion nudges the already-trained model locally
    instead of waiting for (or forcing) another full retrain.

    Loss scope, deliberately: existence_loss + component_loss only — the
    self-supervised "does this edge exist, and with what weights" signal
    OracleDataset.from_graph() is itself primarily built from (see its
    docstring), which is exactly the part of Oracle1Loss.total() that is
    well-defined for a just-added node with no outcome history yet.
    forecast_loss/zone_coherence_loss/retrospective-convergence supervision
    need ground truth (validated future outcomes, a populated
    zone_memberships map across many nodes, confirmed historical
    convergence points) that a live incremental update — by construction —
    doesn't have for brand-new nodes; those stay Phase-2/backtest concerns,
    where that supervision genuinely exists.
    """

    def __init__(self, model: Oracle1Model, loss_fn: Oracle1Loss,
                 feature_builder: NodeFeatureBuilder, edge_builder: EdgeFeatureBuilder,
                 optimizer=None, device=None,
                 k_hop: int = 2, max_local_nodes: int = 500):
        self.model   = model
        self.loss_fn = loss_fn
        self.fb      = feature_builder
        self.eb      = edge_builder
        # Optional: when given, _local_step() actually applies the
        # gradient (optimizer.step()) rather than just reporting the loss —
        # matching how train.py wires its own optimizer in from outside
        # the trainer/loss objects rather than having them own it.
        self.optimizer = optimizer
        self.device    = device
        self.k_hop           = k_hop
        self.max_local_nodes = max_local_nodes

    def train_epoch(self, snapshot_pairs, delta_range_years=(0.5, 5.0)) -> Dict[str, float]:
        """
        snapshot_pairs: List[Tuple[RuntimeGraph, RuntimeGraph]], each a
        (before, after) pair spanning one ingestion delta. `delta_range_years`
        is accepted for interface stability with earlier callers but is not
        itself enforced here — callers building snapshot_pairs (e.g. from a
        series of as_of() cuts) are the natural place to decide spacing;
        this method only needs the two end states.

        Returns mean_loss over local steps actually taken, plus counts —
        n_steps can be 0 (every pair a no-op delta), in which case
        mean_loss is reported as 0.0 rather than dividing by zero.
        """
        if not TORCH_AVAILABLE:
            return {"error": "torch_not_available"}
        if self.device is None:
            return {"error": "device_not_set"}

        total_loss = 0.0
        n_steps = 0
        total_new_nodes = 0
        total_new_edges = 0

        for snap_t, snap_td in snapshot_pairs:
            t_ids  = set(snap_t.nodes.keys())
            td_ids = set(snap_td.nodes.keys())
            new_node_ids = td_ids - t_ids

            t_keys  = {(e.source_id, e.target_id) for e in snap_t.edges.values()}
            td_keys = {(e.source_id, e.target_id) for e in snap_td.edges.values()}
            new_edge_keys = td_keys - t_keys

            total_new_nodes += len(new_node_ids)
            total_new_edges += len(new_edge_keys)

            changed_seed = set(new_node_ids)
            changed_seed.update(n for pair in new_edge_keys for n in pair)
            changed_seed &= td_ids
            if not changed_seed:
                # Nothing actually changed between the two snapshots — no
                # wave to propagate, so no gradient step for this pair.
                # This IS the "no full-graph retrain on every addition"
                # behaviour the manifest asks for, made visible: most pairs
                # in a densely-sampled series will legitimately hit this.
                continue

            local_ids = snap_td.k_hop_neighborhood(list(changed_seed), k=self.k_hop)
            if len(local_ids) > self.max_local_nodes:
                # Keep the wave bounded even when the change touches a
                # high-degree hub — the seed always stays in, the rest of
                # the neighbourhood is capped rather than silently
                # ballooning this "local" step back into a full-graph pass.
                overflow = list(local_ids - changed_seed)
                budget = max(0, self.max_local_nodes - len(changed_seed))
                local_ids = set(changed_seed) | set(overflow[:budget])

            local_graph = snap_td.subgraph(local_ids)
            if not local_graph.nodes:
                continue

            step_loss = self._local_step(local_graph)
            if step_loss is not None:
                total_loss += step_loss
                n_steps += 1

        logger.info(
            "[TemporalSplitTrainer] %d pair(s): +%d new nodes, +%d new edges, "
            "%d local gradient step(s) taken (k_hop=%d, cap=%d)",
            len(snapshot_pairs), total_new_nodes, total_new_edges, n_steps,
            self.k_hop, self.max_local_nodes,
        )
        return {
            "mean_loss": total_loss / n_steps if n_steps else 0.0,
            "n_steps":   n_steps,
            "new_nodes": total_new_nodes,
            "new_edges": total_new_edges,
        }

    def _local_step(self, local_graph: "RuntimeGraph") -> Optional[float]:
        """One forward+backward(+step) restricted to local_graph — see
        _local_gradient_step() (shared with ContinuousFeedEngine.feed())
        for what this actually does."""
        return _local_gradient_step(
            self.model, self.loss_fn, local_graph, self.device,
            feature_builder=self.fb, edge_builder=self.eb, optimizer=self.optimizer)

__all__ = ['OracleExample', 'OracleDataset', '_local_gradient_step', 'TemporalSplitTrainer']
