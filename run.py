"""
Oracle-1 · Inference Script
# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
============================

Запускает обученную модель в нескольких режимах:

  query    — найти точки конвергенции по текстовому запросу
  forecast — рекурсивный прогноз будущих изобретений от затравочного запроса
  scan     — полное сканирование графа: прорывы, фантомы, онтологические разрывы
  edge     -- предсказать вероятность связи между двумя концепциями
  report   — комплексный отчёт (query + forecast + scan)

Примеры использования
---------------------
  # Полный отчёт по теме
  python run.py report --query "coherent light amplification"

  # Только точки конвергенции
  python run.py query --query "population inversion mechanism"

  # Рекурсивный прогноз
  python run.py forecast --query "stimulated emission" --depth 4

  # Вероятность связи между двумя идеями
  python run.py edge --src "Einstein stimulated emission" --tgt "ruby crystal maser"

  # Сканирование с кастомным порогом
  python run.py scan --threshold 60.0

  # Загрузить конкретный чекпоинт
  python run.py report --query "laser" --ckpt ./checkpoints/best_ema_weights.pt

  # Запустить N шагов символической симуляции перед инференсом
  python run.py report --query "laser" --graph_steps 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from oracle1 import (
    # Модель и граф
    Oracle1Model,
    build_oracle1_system,
    CONFIG as ORACLE_CONFIG,
    # Типы данных
    RuntimeGraph,
    # Детектор прорывов
    HighConvergenceDetector,
    # TODO #4: graph_to_tensors used to be defined locally in this file
    # (see git history / TODO.md #4) — relocated into oracle1.py so
    # train.py can share it too (needed to compute real phantom-scan
    # node_latents instead of a hardcoded None). compute_node_latents is
    # the new graph_to_tensors()+encode() convenience wrapper.
    graph_to_tensors,
    compute_node_latents,
    HashingTextEmbedder,
    # TODO #2: graph (de)serialization, shared with train.py.
    save_graph_snapshot,
    load_graph_snapshot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle1.run")


# ══════════════════════════════════════════════════════════════════════════════
#  ЗАГРУЗКА ВЕСОВ
# ══════════════════════════════════════════════════════════════════════════════

def load_model_weights(
    model:   Oracle1Model,
    ckpt_path: Path,
    device:  torch.device,
) -> Dict[str, Any]:
    """Загрузить веса из чекпоинта.

    Поддерживает два формата, которые сохраняет train.py:
      • ``best_ema_weights.pt``  — только веса EMA-модели (лёгкий, для инференса)
      • ``best_model.pt``        — полный чекпоинт с оптимайзером и метаданными
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # EMA-only format (создаётся в train.py строка ~979)
    if "model_state_dict" in ckpt and "optimizer_state_dict" not in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {
            "epoch":        ckpt.get("epoch", "?"),
            "ema_val_loss": ckpt.get("ema_val_loss", float("nan")),
            "model_config": ckpt.get("model_config", {}),
        }
        log.info(
            f"[ckpt] EMA weights loaded from {ckpt_path.name}  "
            f"epoch={meta['epoch']}  ema_val_loss={meta['ema_val_loss']:.6f}"
        )
        return meta

    # Full checkpoint format
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        meta = {
            "epoch":        ckpt.get("epoch", "?"),
            "best_val_loss": ckpt.get("best_val_loss", float("nan")),
            "model_config": ckpt.get("model_config", {}),
            "graph_snapshot": ckpt.get("graph_snapshot", {}),
        }
        log.info(
            f"[ckpt] Full checkpoint loaded from {ckpt_path.name}  "
            f"epoch={meta['epoch']}  best_val_loss={meta['best_val_loss']:.6f}"
        )
        return meta

    raise ValueError(
        f"Нераспознанный формат чекпоинта: {ckpt_path.name}  "
        f"(ключи: {list(ckpt.keys())})"
    )


def find_best_checkpoint(save_dir: Path) -> Optional[Path]:
    """Найти лучший чекпоинт в папке (приоритет: EMA → best → latest)."""
    for name in ("best_ema_weights.pt", "best_model.pt", "final_model.pt"):
        p = save_dir / name
        if p.exists():
            return p
    # последний периодический чекпоинт
    periodic = sorted(save_dir.glob("checkpoint_epoch_*.pt"),
                      key=lambda p: p.stat().st_mtime)
    if periodic:
        return periodic[-1]
    return None


# graph_to_tensors() is now imported from oracle1 (see the import block
# above) — relocated there for TODO #4 so train.py can share it too.


# ══════════════════════════════════════════════════════════════════════════════
#  НЕЙРОСЕТЕВОЙ ИНФЕРЕНС
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_neural_inference(
    model:  Oracle1Model,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, Any]:
    """Прогнать вспомогательный оценщик контекстного влияния по графу.

    Возвращает:
      node_latents   — [N, latent_dim]  скрытые представления
      node_features  — [N, feat_dim]    обновлённые признаки
      node_ids       — List[str]        порядок узлов

    ForecastHead deliberately is not read here. Ordinary training supervises
    edge existence and the seven edge components, not future outcomes;
    presenting its untrained output as a breakthrough probability would be a
    category error. Ranking remains a transparent graph operation.
    """
    model.eval()

    # Убираем вспомогательные ключи, не принимаемые encode()
    node_ids = tensors.pop("node_ids")
    try:
        # Bug 7 fix: pass topology_vecs so ContextualPriorityHead modulates attention
        # TODO #10: pass edge_provenance_discount so StructuralDependencyUpdater's
        # sdi/casc scatter_add is discounted the same way excess_investment_pressure
        # already is (see compute_ancestor_overlap_discount in oracle1.py).
        node_latents, node_features_upd = model.encode(
            node_features         = tensors["node_features"],
            edge_index            = tensors["edge_index"],
            edge_attr             = tensors["edge_attr"],
            containment_index     = tensors["containment_index"],
            containment_zone_mult = tensors["containment_zone_mult"],
            node_years            = tensors["node_years"],
            node_zone_mult        = tensors["node_zone_mult"],
            topology_vecs         = tensors.get("topology_vecs"),
            edge_provenance_discount = tensors.get("edge_provenance_discount"),
        )
        return dict(
            node_latents    = node_latents,
            node_features   = node_features_upd,
            node_ids        = node_ids,
        )
    finally:
        # Вернуть ключ обратно, чтобы не сломать вызывающий код
        tensors["node_ids"] = node_ids


def predict_edge_probability(
    model:       Oracle1Model,
    node_latents: torch.Tensor,
    node_ids:    List[str],
    src_text:    str,
    tgt_text:    str,
    graph:       RuntimeGraph,
) -> Tuple[float, np.ndarray]:
    """Предсказать вероятность ребра между двумя узлами (по тексту).

    Возвращает (exist_probability, component_vector).
    """
    # Найти ближайшие узлы по тексту
    def _find(text: str) -> Optional[int]:
        text_lo = text.lower()
        # точное совпадение
        for i, nid in enumerate(node_ids):
            node = graph.get_node(nid)
            if node and node.text.lower() == text_lo:
                return i
        # частичное совпадение
        for i, nid in enumerate(node_ids):
            node = graph.get_node(nid)
            if node and text_lo in node.text.lower():
                return i
        return None

    si = _find(src_text)
    ti = _find(tgt_text)
    if si is None or ti is None:
        raise ValueError(
            f"Не удалось найти узлы для: '{src_text}' / '{tgt_text}'. "
            "Попробуйте --graph_steps чтобы расширить граф."
        )

    with torch.no_grad():
        h_src = node_latents[si].unsqueeze(0)
        h_tgt = node_latents[ti].unsqueeze(0)
        exist_prob, comps = model.predict_edge(h_src, h_tgt)

    return (
        float(exist_prob.squeeze()),
        comps.squeeze().cpu().numpy()
    )


# ══════════════════════════════════════════════════════════════════════════════
#  СИМВОЛИЧЕСКИЙ ИНФЕРЕНС (без нейросети)
# ══════════════════════════════════════════════════════════════════════════════

def run_symbolic_scan(
    system:    Dict[str, Any],
    threshold: float = 72.5,
) -> Dict[str, Any]:
    """Символическая часть: прорывы, фантомы, разрывы, давление."""
    graph    = system["graph"]
    extended = system["extended"]
    dynamics = system["dynamics"]
    topology = system["topology"]

    # HighConvergenceDetector: узлы, пересёкшие порог
    detector = HighConvergenceDetector(threshold=threshold)
    alerts   = detector.scan(graph)

    # Топология: отчёт о разрывах
    rupture_text = topology.rupture_report()

    # Динамика: pre-breakthrough кандидаты
    pre_break = dynamics.accelerometer.pre_breakthrough_candidates()

    # Давление: топ-5 узлов
    top_pressure = extended._top_pressure_nodes(5)

    # Фантомные узлы
    phantoms = extended.phantom_gen.phantoms

    return dict(
        alerts        = alerts,
        rupture_text  = rupture_text,
        pre_break     = pre_break,
        top_pressure  = top_pressure,
        phantoms      = phantoms,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  РЕЖИМЫ ЗАПУСКА
# ══════════════════════════════════════════════════════════════════════════════

def mode_query(args: argparse.Namespace, system: Dict[str, Any],
               neural: Optional[Dict[str, Any]]) -> None:
    """Найти точки конвергенции по текстовому запросу."""
    extended = system["extended"]

    # Простой embedder, если не подключён внешний
    embedder = _get_embedder(system)

    log.info(f"\n{'═'*64}")
    log.info(f"  QUERY MODE: «{args.query}»")
    log.info(f"{'═'*64}\n")

    cloud = extended.query_convergence(
        query_text = args.query,
        embedder   = embedder,
        n_paths    = args.n_paths,
    )
    print("\n" + cloud.to_readable())

    # Latents make edge influence queries possible; node ranking stays a
    # graph-derived diagnostic, never an unsupervised neural forecast.
    if neural is not None:
        _print_structural_top_nodes(system["graph"], k=args.top_k)


def mode_forecast(args: argparse.Namespace, system: Dict[str, Any],
                  neural: Optional[Dict[str, Any]]) -> None:
    """Рекурсивный прогноз будущих технологий."""
    extended = system["extended"]
    embedder = _get_embedder(system)

    log.info(f"\n{'═'*64}")
    log.info(f"  FORECAST MODE: «{args.query}»  depth={args.depth}")
    log.info(f"{'═'*64}\n")

    scenarios = extended.run_recursive_forecast(
        seed_query = args.query,
        embedder   = embedder,
        max_depth  = args.depth,
    )

    if not scenarios:
        print("Сценарии не построены — граф слишком мал. "
              "Увеличьте --graph_steps.")
        return

    for i, sc in enumerate(scenarios, 1):
        print(f"\n── Сценарий {i}/{len(scenarios)} "
              f"(coherence={sc.coherence_score:.3f}, "
              f"collapse_p={sc.collapse_probability:.2f}) ──")
        print(f"  {sc.description}")
        print(f"  Глубина: {sc.depth}  "
              f"Уверенность по глубинам: "
              + " → ".join(f"{c:.2f}" for c in sc.confidence_at_depth))
        if sc.hallucinated_nodes:
            print(f"  Галлюцинированные узлы ({len(sc.hallucinated_nodes)}):")
            for hn in sc.hallucinated_nodes[:5]:
                print(f"    [{hn.confidence:.2f}] {hn.node.text[:60]}")
        if sc.secondary_effects:
            print(f"  Вторичные эффекты:")
            for eff in sc.secondary_effects[:3]:
                print(f"    • {eff}")


def mode_scan(args: argparse.Namespace, system: Dict[str, Any],
              neural: Optional[Dict[str, Any]]) -> None:
    """Полное сканирование графа."""
    log.info(f"\n{'═'*64}")
    log.info(f"  SCAN MODE  (threshold={args.threshold})")
    log.info(f"{'═'*64}\n")

    result = run_symbolic_scan(system, threshold=args.threshold)

    # ── Онтологические разрывы ─────────────────────────────────────────────
    print(result["rupture_text"])

    # ── HighConvergence alerts ─────────────────────────────────────────────
    alerts = result["alerts"]
    print(f"\n{'═'*64}")
    print(f"  HIGH-CONVERGENCE ALERTS  ({len(alerts)} найдено, порог={args.threshold})")
    print(f"{'═'*64}")
    if alerts:
        for alert in alerts[:args.top_k]:
            print("\n" + alert.to_readable())
    else:
        print("  Нет узлов выше порога.")

    # ── Pre-breakthrough кандидаты ─────────────────────────────────────────
    pre_break = result["pre_break"]
    if pre_break:
        print(f"\n── Pre-breakthrough candidates ({len(pre_break)}) ──")
        for nid, score in pre_break[:args.top_k]:
            node = system["graph"].get_node(nid)
            label = node.text[:50] if node else nid
            print(f"  [{score:.4f}] {label}")

    # ── Топ давление ──────────────────────────────────────────────────────
    print(f"\n── Top-5 Pressure Nodes ──")
    for label, pressure in result["top_pressure"]:
        print(f"  [{pressure:.3f}] {label}")

    # ── Фантомы ───────────────────────────────────────────────────────────
    phantoms = result["phantoms"]
    if phantoms:
        print(f"\n── Phantom Nodes ({len(phantoms)}) ──")
        for ph in sorted(phantoms,
                         key=lambda p: p.structural_gap_score, reverse=True)[:args.top_k]:
            print(f"  [{ph.structural_gap_score:.3f}] {ph.gap_description[:60]}")

    # ── Структурный рейтинг ───────────────────────────────────────────────
    if neural is not None:
        _print_structural_top_nodes(system["graph"], k=args.top_k)


def mode_edge(args: argparse.Namespace, system: Dict[str, Any],
              neural: Optional[Dict[str, Any]]) -> None:
    """Предсказать связь между двумя концепциями."""
    if neural is None:
        log.error("Режим 'edge' требует загруженной модели (--ckpt).")
        sys.exit(1)

    log.info(f"\n{'═'*64}")
    log.info(f"  EDGE MODE: «{args.src}» → «{args.tgt}»")
    log.info(f"{'═'*64}\n")

    try:
        prob, comps = predict_edge_probability(
            model        = system["model"],
            node_latents = neural["node_latents"],
            node_ids     = neural["node_ids"],
            src_text     = args.src,
            tgt_text     = args.tgt,
            graph        = system["graph"],
        )
    except ValueError as e:
        log.error(str(e))
        sys.exit(1)

    from oracle1 import EXTENDED_EDGE_COMPONENTS
    print(f"\n  Вероятность ребра:   {prob:.1%}")
    print(f"  Компоненты ребра:")
    for name, val in zip(EXTENDED_EDGE_COMPONENTS, comps):
        bar = "█" * int(val * 20)
        print(f"    {name:<28} {val:.3f}  {bar}")

    # Проверить, есть ли уже ребро в символическом графе
    graph = system["graph"]
    exist_node_ids = [nid for nid, node in graph.nodes.items()
                      if args.src.lower() in node.text.lower()
                      or args.tgt.lower() in node.text.lower()]
    if len(exist_node_ids) >= 2:
        sid, tid = exist_node_ids[0], exist_node_ids[1]
        symbolic = graph.edges.get((sid, tid)) or graph.edges.get((tid, sid))
        if symbolic:
            print(f"\n  ▸ Ребро уже существует в символическом графе "
                  f"(total_weight={symbolic.total_weight:.3f})")
        else:
            print(f"\n  ▸ Ребра пока нет в символическом графе — "
                  f"нейросеть предсказывает его с вероятностью {prob:.1%}")


def mode_report(args: argparse.Namespace, system: Dict[str, Any],
                neural: Optional[Dict[str, Any]]) -> None:
    """Комплексный отчёт: query + scan + forecast."""
    print(f"\n{'█'*64}")
    print(f"  ORACLE-1  FULL REPORT")
    print(f"  Query: «{args.query}»")
    print(f"  Graph: {len(system['graph'].nodes)} nodes / "
          f"{len(system['graph'].edges)} edges")
    print(f"{'█'*64}")

    # 1. Convergence query
    print(f"\n{'═'*64}")
    print(f"  § 1. CONVERGENCE ANALYSIS")
    print(f"{'═'*64}")
    mode_query(args, system, neural)

    # 2. Scan
    print(f"\n{'═'*64}")
    print(f"  § 2. GRAPH SCAN")
    print(f"{'═'*64}")
    mode_scan(args, system, neural)

    # 3. Recursive forecast
    print(f"\n{'═'*64}")
    print(f"  § 3. RECURSIVE FORECAST")
    print(f"{'═'*64}")
    mode_forecast(args, system, neural)


# ══════════════════════════════════════════════════════════════════════════════
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def _get_embedder(system: Optional[Dict[str, Any]] = None):
    """Use the encoder that built the loaded graph's text features."""
    if system is not None and system.get("embedder") is not None:
        return system["embedder"]
    return HashingTextEmbedder()


def _print_structural_top_nodes(graph: RuntimeGraph, k: int = 10) -> None:
    """Show graph-derived diagnostics, explicitly not neural probabilities."""
    rows = sorted(
        graph.nodes.values(),
        key=lambda node: (
            float(getattr(node, "forecast_score", 0.0)),
            float(getattr(node, "convergence_potential", 0.0)),
            float(getattr(node, "upstream_pressure", 0.0)),
        ),
        reverse=True,
    )[:k]
    print(f"\n── Structural Top-{k} Candidates (diagnostic, not probability) ──")
    for rank, node in enumerate(rows, 1):
        print(
            f"  {rank:2}. [struct={node.forecast_score:6.3f}  "
            f"potential={node.convergence_potential:6.3f}] {node.text[:70]}"
        )


def _run_graph_steps(system: Dict[str, Any], n_steps: int,
                     device: Optional[torch.device] = None) -> None:
    """Прогреть символическую часть системы.

    TODO #4 fix: thread real node_latents into update_epoch() (via
    compute_node_latents(), same as train.py's oracle_graph_step()) instead
    of relying on the old hardcoded node_latents=None inside
    Oracle1Extended.update_epoch() — without this, phantom-void scanning
    silently always returned no phantoms.
    """
    if n_steps <= 0:
        return
    log.info(f"[graph] Запускаем {n_steps} шагов символической симуляции…")
    for step in range(n_steps):
        ts = float(time.time())
        node_latents = (compute_node_latents(system["model"], system["graph"], device,
                                             extended=system["extended"])
                        if device is not None else None)
        ext = system["extended"].update_epoch(ts, node_latents=node_latents)
        dyn = system["dynamics"].run_epoch(ts)
        topo = system["topology"].run_epoch(ts)

        n_nodes = len(system["graph"].nodes)
        n_edges = len(system["graph"].edges)
        phantoms = len(system["extended"].phantom_gen.phantoms)
        log.info(
            f"  step {step+1}/{n_steps}: "
            f"nodes={n_nodes}  edges={n_edges}  phantoms={phantoms}  "
            f"pre_break={dyn.get('pre_breakthrough_candidates', 0)}  "
            f"ruptures={topo.get('rupture_count', 0)}"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle-1 inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True,
                           help="Режим запуска")

    # ── Общие аргументы (для всех режимов) ───────────────────────────────────
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--ckpt", default=None,
        help="Путь к чекпоинту (.pt). Если не указан — ищет best_ema_weights.pt "
             "в --save_dir"
    )
    common.add_argument(
        "--save_dir", default="./checkpoints",
        help="Папка с чекпоинтами (используется если --ckpt не задан)"
    )
    common.add_argument(
        "--device", default="cuda",
        help="cuda / cpu / mps"
    )
    common.add_argument(
        "--latent_dim",   type=int, default=ORACLE_CONFIG["model_latent_dim"]
    )
    common.add_argument(
        "--n_gnn_layers", type=int, default=ORACLE_CONFIG["model_n_gnn_layers"]
    )
    common.add_argument(
        "--n_heads",      type=int, default=ORACLE_CONFIG["model_n_heads"]
    )
    common.add_argument(
        "--graph_steps", type=int, default=10,
        help="Шагов символической симуляции перед инференсом"
    )
    common.add_argument(
        "--top_k", type=int, default=10,
        help="Сколько результатов показывать"
    )
    common.add_argument(
        "--no_neural", action="store_true",
        help="Пропустить нейросетевой инференс (только символическая система)"
    )
    common.add_argument(
        "--output_json", default=None,
        help="Сохранить результаты в JSON-файл"
    )
    common.add_argument(
        "--graph_snapshot", default=None,
        help=(
            "Путь к снапшоту графа (nodes/edges/zones/phantoms/virtual_nodes), "
            "тот же формат, что и --graph_snapshot в train.py. Если файл "
            "существует — загружается ДО --graph_steps. Если флаг передан, "
            "состояние сохраняется обратно в тот же файл ПОСЛЕ --graph_steps, "
            "так что несколько запусков run.py тоже накапливают рост графа, "
            "а не только train.py (TODO #2: 'across training resumes and "
            "inference calls')."
        )
    )

    # ── query ─────────────────────────────────────────────────────────────────
    s = sub.add_parser("query", parents=[common],
                       help="Конвергенция по запросу")
    s.add_argument("--query", required=True, help="Текстовый запрос")
    s.add_argument("--n_paths", type=int, default=20,
                   help="Число путей для PathAgnosticInference")

    # ── forecast ──────────────────────────────────────────────────────────────
    s = sub.add_parser("forecast", parents=[common],
                       help="Рекурсивный прогноз")
    s.add_argument("--query", required=True, help="Затравочный запрос")
    s.add_argument("--depth", type=int, default=4,
                   help="Глубина рекурсивного прогноза")

    # ── scan ──────────────────────────────────────────────────────────────────
    s = sub.add_parser("scan", parents=[common],
                       help="Сканирование графа")
    s.add_argument("--threshold", type=float, default=72.5,
                   help="Порог HighConvergenceDetector (0–100)")

    # ── edge ──────────────────────────────────────────────────────────────────
    s = sub.add_parser("edge", parents=[common],
                       help="Вероятность ребра между двумя концепциями")
    s.add_argument("--src", required=True, help="Исходный узел (текст)")
    s.add_argument("--tgt", required=True, help="Целевой узел (текст)")

    # ── report ────────────────────────────────────────────────────────────────
    s = sub.add_parser("report", parents=[common],
                       help="Полный отчёт (query + scan + forecast)")
    s.add_argument("--query", required=True, help="Текстовый запрос")
    s.add_argument("--depth", type=int, default=3,
                   help="Глубина рекурсии для секции forecast")
    s.add_argument("--threshold", type=float, default=72.5)
    s.add_argument("--n_paths",   type=int,   default=20)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Устройство ─────────────────────────────────────────────────────────────
    has_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not has_cuda:
        log.warning("CUDA недоступна — переключаемся на CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    log.info(f"Device: {device}")
    if has_cuda and device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")

    # ── Построить систему Oracle-1 ─────────────────────────────────────────────
    log.info("Инициализация Oracle-1 system…")
    system = build_oracle1_system(
        latent_dim   = args.latent_dim,
        n_gnn_layers = args.n_gnn_layers,
        n_heads      = args.n_heads,
    )
    model: Oracle1Model = system["model"]
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Oracle1Model  параметров: {n_params:,}")

    # ── TODO #2: restore graph snapshot, if given and present ─────────────────
    if args.graph_snapshot and Path(args.graph_snapshot).exists():
        snap_counts = load_graph_snapshot(args.graph_snapshot, system)
        log.info(f"[graph_snapshot] Восстановлено из {args.graph_snapshot}: {snap_counts}")

    # ── Загрузить веса ─────────────────────────────────────────────────────────
    ckpt_path: Optional[Path] = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    elif not args.no_neural:
        ckpt_path = find_best_checkpoint(Path(args.save_dir))
        if ckpt_path:
            log.info(f"Автопоиск чекпоинта: {ckpt_path}")
        else:
            log.warning(
                f"Чекпоинт не найден в {args.save_dir}. "
                "Модель будет с случайными весами. "
                "Используйте --ckpt <path> или --no_neural."
            )

    neural_results: Optional[Dict[str, Any]] = None

    if ckpt_path and not args.no_neural:
        load_model_weights(model, ckpt_path, device)

    # ── Прогреть граф ──────────────────────────────────────────────────────────
    # TODO #4: pass device through so phantom-void scanning gets real node
    # latents (skipped — node_latents=None — when --no_neural, matching the
    # "no neural inference at all" intent of that flag).
    _run_graph_steps(system, args.graph_steps, device=None if args.no_neural else device)

    # TODO #2: persist any growth from --graph_steps back to the same
    # snapshot, so a sequence of run.py invocations accumulates state too,
    # not just train.py runs.
    if args.graph_snapshot:
        save_graph_snapshot(args.graph_snapshot, system)

    graph = system["graph"]
    log.info(
        f"Граф: {len(graph.nodes)} узлов / "
        f"{len(graph.edges)} рёбер"
    )

    # ── Нейросетевой инференс ─────────────────────────────────────────────────
    if not args.no_neural and len(graph.nodes) > 0:
        # Bug 7 fix: pass extended so graph_to_tensors can attach topology_vecs
        tensors = graph_to_tensors(graph, device, extended=system["extended"])
        if tensors is not None:
            log.info("Запуск нейросетевого инференса…")
            t0 = time.time()
            neural_results = run_neural_inference(model, tensors, device)
            log.info(f"Нейросеть: {time.time() - t0:.2f}s  "
                     f"({len(neural_results['node_ids'])} узлов)")
        else:
            log.warning("Граф пустой — нейросетевой инференс пропущен")

    # ── Выбрать режим ─────────────────────────────────────────────────────────
    t_start = time.time()

    dispatch = {
        "query":    mode_query,
        "forecast": mode_forecast,
        "scan":     mode_scan,
        "edge":     mode_edge,
        "report":   mode_report,
    }
    dispatch[args.mode](args, system, neural_results)

    elapsed = time.time() - t_start
    log.info(f"\n[done] Режим '{args.mode}' завершён за {elapsed:.2f}s")

    # ── Сохранить JSON (опционально) ──────────────────────────────────────────
    if args.output_json and neural_results is not None:
        out = {
            "mode":    args.mode,
            "graph": {
                "n_nodes": len(graph.nodes),
                "n_edges": len(graph.edges),
            },
            "top_nodes": [
                {
                    "rank":   i + 1,
                    "node_id": nid,
                    "text":   (graph.get_node(nid).text
                               if graph.get_node(nid) else ""),
                    "structural_score": float(
                        getattr(graph.get_node(nid), "forecast_score", 0.0)
                    ),
                    "convergence_potential": float(
                        getattr(graph.get_node(nid), "convergence_potential", 0.0)
                    ),
                }
                for i, (idx, nid) in enumerate(
                    sorted(
                        zip(
                            range(len(neural_results["node_ids"])),
                            neural_results["node_ids"],
                        ),
                        key=lambda x: (
                            float(getattr(graph.get_node(x[1]), "forecast_score", 0.0)),
                            float(getattr(graph.get_node(x[1]), "convergence_potential", 0.0)),
                        ),
                        reverse=True,
                    )
                )[:args.top_k]
            ],
        }
        Path(args.output_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        log.info(f"[json] результаты сохранены → {args.output_json}")


if __name__ == "__main__":
    main()
