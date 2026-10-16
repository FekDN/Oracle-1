# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
"""Conversion between runtime graphs and model tensors."""
from __future__ import annotations

from ._shared import *
from .entities import EdgeFeatureBuilder, NodeFeatureBuilder
from .graph import RuntimeGraph, compute_ancestor_overlap_discount

def graph_to_tensors(
    graph:   RuntimeGraph,
    device:  "torch.device",
    node_fb: Optional[NodeFeatureBuilder] = None,
    edge_fb: Optional[EdgeFeatureBuilder] = None,
    extended=None,          # Bug 7 fix: Oracle1Extended for topology_vecs
) -> Optional[Dict[str, Any]]:
    """Конвертировать RuntimeGraph в тензоры для Oracle1Model.encode().

    Возвращает словарь с ключами:
      node_features, edge_index, edge_attr,
      containment_index, containment_zone_mult,
      node_years, node_zone_mult, topology_vecs (optional),
      edge_provenance_discount, node_ids

    Возвращает None если граф пустой (нет узлов).

    Если передан extended (Oracle1Extended), запрашивает
    get_topology_vecs_tensor() и включает результат в словарь под ключом
    topology_vecs, чтобы encode() мог активировать ContextualPriorityHead.

    TODO #4: this used to live only in run.py (a script, not importable by
    train.py) — relocated here so both entrypoints can build node_latents
    for Oracle1Extended.update_epoch() via compute_node_latents() below.

    TODO #10: also returns edge_provenance_discount, row-aligned 1:1 with
    edge_index's columns (computed in the SAME loop that establishes that
    ordering, so there is no risk of misalignment) — see
    compute_ancestor_overlap_discount(). Oracle1Model.encode() multiplies
    this into the edge_weights feeding StructuralDependencyUpdater's sdi/casc
    scatter_add, the neural analogue of the fix already applied to
    ConvergencePressureField.excess_investment_pressure.
    """
    node_ids = list(graph.nodes.keys())
    if not node_ids:
        return None

    node_fb = node_fb or NodeFeatureBuilder()
    edge_fb = edge_fb or EdgeFeatureBuilder()

    # ── Признаки узлов ────────────────────────────────────────────────────────
    node_idx: Dict[str, int] = {nid: i for i, nid in enumerate(node_ids)}
    node_feat_list = []
    node_years_list = []
    node_zone_mult_list = []

    for nid in node_ids:
        node = graph.nodes[nid]
        try:
            feat = node_fb.build(node)
        except Exception:
            feat = np.zeros(NodeFeatureBuilder.TOTAL_DIM, dtype=np.float32)
        node_feat_list.append(feat)

        # Год из timestamp (Unix → calendar year approximation)
        year = int(node.timestamp / 31_536_000 + 1970) if node.timestamp != 0 else 2000
        node_years_list.append(max(1700, min(year, 2100)))
        node_zone_mult_list.append(float(node.acceleration_multiplier))

    node_features = torch.tensor(
        np.stack(node_feat_list), dtype=torch.float32, device=device
    )
    node_years = torch.tensor(node_years_list, dtype=torch.float32, device=device)
    node_zone_mult = torch.tensor(node_zone_mult_list, dtype=torch.float32, device=device)

    # ── Рёбра (обычные и containment) ─────────────────────────────────────────
    # TODO #10: computed once here (not once per edge elsewhere) so the
    # ordering that edge_index/edge_provenance_discount share is trivially
    # guaranteed correct.
    provenance_discount = compute_ancestor_overlap_discount(graph)
    edge_src, edge_tgt, edge_attr_list, edge_disc_list = [], [], [], []
    cont_src, cont_tgt, cont_mult_list = [], [], []

    for (src_id, tgt_id), edge in graph.edges.items():
        if src_id not in node_idx or tgt_id not in node_idx:
            continue
        si, ti = node_idx[src_id], node_idx[tgt_id]
        if edge.is_containment_edge:
            cont_src.append(si)
            cont_tgt.append(ti)
            cont_mult_list.append(
                float(graph.nodes[src_id].zone_multiplier)
                if src_id in graph.nodes else 1.0
            )
        else:
            try:
                attr = edge_fb.build(edge)
            except Exception:
                attr = np.zeros(EdgeFeatureBuilder.DIM, dtype=np.float32)
            edge_src.append(si)
            edge_tgt.append(ti)
            edge_attr_list.append(attr)
            edge_disc_list.append(provenance_discount.get((src_id, tgt_id), 1.0))

    # edge_index
    if edge_src:
        edge_index = torch.tensor(
            [edge_src, edge_tgt], dtype=torch.long, device=device
        )
        edge_attr = torch.tensor(
            np.stack(edge_attr_list), dtype=torch.float32, device=device
        )
        edge_provenance_discount = torch.tensor(
            edge_disc_list, dtype=torch.float32, device=device
        )
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        edge_attr  = torch.zeros((0, EdgeFeatureBuilder.DIM),
                                 dtype=torch.float32, device=device)
        edge_provenance_discount = torch.zeros(0, dtype=torch.float32, device=device)

    # containment_index
    if cont_src:
        containment_index = torch.tensor(
            [cont_src, cont_tgt], dtype=torch.long, device=device
        )
        containment_zone_mult = torch.tensor(
            cont_mult_list, dtype=torch.float32, device=device
        )
    else:
        containment_index     = torch.zeros((2, 0), dtype=torch.long, device=device)
        containment_zone_mult = torch.zeros(0, dtype=torch.float32, device=device)

    # Attach topology_vecs when Oracle1Extended is available
    topology_vecs = None
    if extended is not None and hasattr(extended, "get_topology_vecs_tensor"):
        topology_vecs = extended.get_topology_vecs_tensor(node_ids, device=device)

    result = dict(
        node_features        = node_features,
        edge_index           = edge_index,
        edge_attr            = edge_attr,
        containment_index    = containment_index,
        containment_zone_mult= containment_zone_mult,
        node_years            = node_years,
        node_zone_mult        = node_zone_mult,
        edge_provenance_discount = edge_provenance_discount,
        node_ids              = node_ids,   # вспомогательное поле, не подаётся в модель
    )
    if topology_vecs is not None:
        result["topology_vecs"] = topology_vecs
    return result


def compute_node_latents(
    model: "Oracle1Model",
    graph: RuntimeGraph,
    device: "torch.device",
    extended=None,
    node_fb: Optional[NodeFeatureBuilder] = None,
    edge_fb: Optional[EdgeFeatureBuilder] = None,
) -> Optional["torch.Tensor"]:
    """TODO #4: run graph_to_tensors() + Oracle1Model.encode() once, in
    inference mode (no_grad — this is a read-only side computation for
    phantom-void scanning, not part of any training step, so it should
    neither build an autograd graph nor perturb the model's train/eval
    flag for the caller's own subsequent forward passes).

    Returns node_latents row-aligned with list(graph.nodes.keys()) at call
    time (matching graph_to_tensors' own node_ids ordering, which is what
    PhantomNodeGenerator._node_to_idx assumes) — or None if the graph is
    empty or TORCH_AVAILABLE is False, in which case callers should pass
    node_latents=None onward (phantom scanning degrades to its previous,
    no-void-detection behaviour rather than failing).
    """
    if not TORCH_AVAILABLE or device is None:
        return None
    tensors = graph_to_tensors(graph, device, node_fb=node_fb, edge_fb=edge_fb, extended=extended)
    if tensors is None:
        return None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            node_latents, _ = model.encode(
                tensors["node_features"], tensors["edge_index"], tensors["edge_attr"],
                tensors["containment_index"], tensors["containment_zone_mult"],
                tensors["node_years"], tensors["node_zone_mult"],
                topology_vecs=tensors.get("topology_vecs"),
                edge_provenance_discount=tensors.get("edge_provenance_discount"),
            )
    finally:
        if was_training:
            model.train()
    return node_latents

__all__ = ['graph_to_tensors', 'compute_node_latents']
