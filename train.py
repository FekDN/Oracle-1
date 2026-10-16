"""
Oracle-1 · Production Training Script  (Two-Phase Architecture)
# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com)
======================================================================

Два явных этапа:
─────────────────────────────────────────────────────────────────────
  ФАЗА 1: ПОСТРОЕНИЕ ГРАФА (graph_build_steps шагов)
    • Только символическая система: extended / dynamics / topology
    • Нейросеть не трогается
    • Граф накапливает структуру, tension, phantom-узлы, void-зоны
    • Логируется размер графа и ключевые метрики после каждого шага
    • Заканчивается когда граф достиг min_nodes_to_train или исчерпан лимит шагов

  ФАЗА 2: ОБУЧЕНИЕ ВСПОМОГАТЕЛЬНОГО ОЦЕНЩИКА ВЛИЯНИЯ (train_epochs эпох)
    • DataLoader строится ПОСЛЕ фазы 1 из живого RuntimeGraph
    • Если граф изменился существенно (rebuild_every шагов фазы 1
      продолжают работать опционально) — датасет перестраивается
    • Стандартный цикл: train_one_epoch → validate → LR step → checkpoint
    • Обучаются только контекстные веса существующих/кандидатных рёбер
      (наличие + 7 компонент). RuntimeGraph не подменяется нейросетью и не
      получает «прогноз» из loss без исторической outcome-разметки.
    • Early stopping, EMA, AMP — без изменений

  ОПЦИОНАЛЬНАЯ ФАЗА 1b: ИНКРЕМЕНТАЛЬНЫЕ ШАГИ ГРАФА
    • Каждые graph_interleave_every эпох обучения:
        oracle_system_epoch() × graph_interleave_steps раз
        затем rebuild датасета из обновлённого графа
    • Позволяет нейросети учиться на растущем графе

Остальная инфраструктура (AMP, EMA, GradScaler, atomic checkpoints,
RNG save/restore, EarlyStopping, torch.compile) — без изменений.

Features
--------
  Core
  ✓ AMP (autocast + GradScaler)
  ✓ Full RNG state save/restore
  ✓ Atomic checkpoint save (write-tmp→rename)
  ✓ Resume from any checkpoint
  ✓ Gradient accumulation
  ✓ Gradient clipping

  Training quality
  ✓ EMA weights
  ✓ EarlyStopping with patience
  ✓ Warmup + CosineAnnealing LR schedule

  Infrastructure
  ✓ Checkpoint rotation
  ✓ JSON config snapshot next to best model
  ✓ Deterministic mode
  ✓ torch.compile
  ✓ Ctrl-C / SIGTERM emergency save

  Oracle-1-specific  (v2)
  ✓ Phase 1: graph build loop (символическая генерация)
  ✓ Phase 2: neural training on graph-derived dataset
  ✓ Dataset rebuild from live RuntimeGraph after graph steps
  ✓ Interleaved graph growth during training (опционально)
  ✓ Graph readiness gate — обучение не начинается до min_nodes_to_train
  ✓ Structural pressure logging per graph step
  ✓ Entropy alert reporting
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

# ── Oracle-1 imports ──────────────────────────────────────────────────────────
# TODO #3 fix: this module is oracle1.py, not oracle1_refactored.py (which
# doesn't exist in this project — the name is a leftover from an earlier
# refactor). The stale import meant train.py could never run at all.
from oracle1 import (
    Oracle1Model,
    Oracle1Loss,
    OracleDataset,
    OracleExample,
    build_oracle1_system,
    CONFIG as ORACLE_CONFIG,
    # TODO #4: graph -> tensors -> Oracle1Model.encode(), needed so the
    # phantom-node scan gets real node latents instead of a hardcoded None.
    # (graph_to_tensors itself isn't called directly here — only via
    # compute_node_latents, which wraps it — so it's not imported by name.)
    compute_node_latents,
    # TODO #2: graph (de)serialization, shared with run.py.
    save_graph_snapshot,
    load_graph_snapshot,
    # TODO #1: text ingestion, never previously wired into either entrypoint.
    IngestionPipeline,
    GraphConsolidator,
    BatchIngestionProcessor,
    BatchDocument,
    INGESTION_SOURCE_TYPES,
    print_conflict_report,
    # (get_llm_client isn't called directly — ingestion reuses
    # system["llm_client"], the one build_oracle1_system() already
    # constructed via get_llm_client() internally, rather than building a
    # second client — so it's not imported by name either.)
    # Manifest §5: incremental local-subgraph training, see --temporal_split_training.
    TemporalSplitTrainer,
    NodeFeatureBuilder,
    EdgeFeatureBuilder,
)
from corpus_library import DocumentQueue, ExtractedDocument, _read_text_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("oracle1.train")


# ══════════════════════════════════════════════════════════════════════════════
#  SEED / DETERMINISM
# ══════════════════════════════════════════════════════════════════════════════

def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
            log.info("[seed] deterministic=STRICT")
        except RuntimeError as exc:
            log.warning(f"[seed] strict deterministic failed ({exc}). Falling back to warn_only=True")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
                log.info("[seed] deterministic=WARN_ONLY")
            except TypeError:
                torch.use_deterministic_algorithms(False)
                log.warning("[seed] warn_only unavailable; deterministic mode disabled.")
    else:
        torch.backends.cudnn.benchmark = True


# ══════════════════════════════════════════════════════════════════════════════
#  RNG STATE CAPTURE / RESTORE
# ══════════════════════════════════════════════════════════════════════════════

def capture_rng_state() -> Dict[str, Any]:
    return {
        "python":     random.getstate(),
        "numpy":      np.random.get_state(),
        "torch":      torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ══════════════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    path:          Path,
    epoch:         int,
    global_step:   int,
    model:         nn.Module,
    optimizer:     optim.Optimizer,
    scheduler,
    scaler:        GradScaler,
    ema:           "EMAWeights",
    best_val_loss: float,
    rng_state:     Optional[Dict[str, Any]] = None,
    extra:         Optional[Dict[str, Any]] = None,
) -> None:
    """Атомарное сохранение: пишет во временный файл, затем переименовывает."""
    path = Path(path)
    tmp  = path.with_suffix(".tmp")

    ckpt: Dict[str, Any] = {
        "epoch":                epoch,
        "global_step":          global_step,
        "best_val_loss":        best_val_loss,
        "total_epochs":         (extra or {}).get("total_epochs"),
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict":    scaler.state_dict(),
        "ema_shadow":           ema.cpu_shadow(),
        "rng_state":            rng_state if rng_state is not None else capture_rng_state(),
    }
    if extra:
        ckpt.update(extra)

    torch.save(ckpt, tmp)
    tmp.rename(path)
    log.info(f"[ckpt] saved → {path.name}")


def load_checkpoint(
    path:        Path,
    model:       nn.Module,
    optimizer:   Optional[optim.Optimizer] = None,
    scheduler=None,
    scaler:      Optional[GradScaler]      = None,
    ema:         Optional["EMAWeights"]    = None,
    device:      str | torch.device        = "cpu",
    restore_rng: bool                      = True,
) -> Tuple[int, int, float, Optional[int]]:
    """Загрузка чекпоинта. Возвращает (start_epoch, global_step, best_val_loss, total_epochs)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer  and "optimizer_state_dict"  in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler  and "scheduler_state_dict"  in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler     and "scaler_state_dict"     in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ema        and "ema_shadow"            in ckpt:
        ema.load_shadow(ckpt["ema_shadow"], device=torch.device(device))
        log.info("[ckpt] EMA shadow weights restored onto device")

    if restore_rng and "rng_state" in ckpt:
        restore_rng_state(ckpt["rng_state"])
        log.info("[ckpt] RNG state restored  (torch / CUDA / numpy / python)")

    epoch         = ckpt.get("epoch",         0)
    global_step   = ckpt.get("global_step",   0)
    best_val_loss = ckpt.get("best_val_loss", float("inf"))

    log.info(
        f"[ckpt] resumed from {path.name}  "
        f"epoch={epoch}  step={global_step}  best_val={best_val_loss:.6f}"
    )
    return epoch, global_step, best_val_loss, ckpt.get("total_epochs")


def rotate_checkpoints(save_dir: Path, prefix: str, keep: int) -> None:
    files = sorted(save_dir.glob(f"{prefix}*.pt"), key=lambda p: p.stat().st_mtime)
    for old in files[:-keep]:
        old.unlink(missing_ok=True)
        log.debug(f"[ckpt] rotated (deleted) {old.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  EMA WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════

class EMAWeights:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay  = decay
        self.shadow: Dict[str, torch.Tensor] = {
            name: param.data.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def cpu_shadow(self) -> Dict[str, torch.Tensor]:
        return {name: t.detach().cpu() for name, t in self.shadow.items()}

    def load_shadow(self, cpu_dict: Dict[str, torch.Tensor],
                    device: torch.device) -> None:
        self.shadow = {name: t.to(device) for name, t in cpu_dict.items()}

    def apply(self, model: nn.Module) -> "_EMAContext":
        return _EMAContext(self, model)

    def copy_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                param.data.copy_(self.shadow[name])


class _EMAContext:
    def __init__(self, ema: EMAWeights, model: nn.Module):
        self.ema    = ema
        self.model  = model
        self._backup: Dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.ema.shadow:
                self._backup[name] = param.data.clone()
                param.data.copy_(self.ema.shadow[name])
        return self

    def __exit__(self, *_):
        for name, param in self.model.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])


# ══════════════════════════════════════════════════════════════════════════════
#  EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5,
                 mode: str = "min"):
        self.patience    = patience
        self.min_delta   = min_delta
        self.mode        = mode
        self.best        = float("inf") if mode == "min" else float("-inf")
        self.counter     = 0
        self.should_stop = False

    def step(self, metric: float) -> bool:
        improved = (
            (metric < self.best - self.min_delta) if self.mode == "min"
            else (metric > self.best + self.min_delta)
        )
        if improved:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            log.info(
                f"[EarlyStopping] patience {self.counter}/{self.patience}  "
                f"(best={self.best:.6f}  current={metric:.6f})"
            )
            if self.counter >= self.patience:
                log.warning("[EarlyStopping] patience exhausted — stopping.")
                self.should_stop = True
        return self.should_stop


# ══════════════════════════════════════════════════════════════════════════════
#  LEARNING RATE SCHEDULE
# ══════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    warmup_epochs = max(1, warmup_epochs)
    cosine_epochs = max(1, total_epochs - warmup_epochs)
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=1e-7)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


# ══════════════════════════════════════════════════════════════════════════════
#  JSON CONFIG SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════

def save_run_config(path: Path, args: argparse.Namespace) -> None:
    oracle_keys = [
        "model_latent_dim", "model_n_gnn_layers", "model_n_heads", "model_dropout",
        "loss_lambda_forecast", "loss_lambda_exist", "loss_lambda_comps",
        "loss_lambda_zone", "forecast_decay_factor", "forecast_max_depth",
        "pressure_singularity_threshold", "collapse_threshold",
        "phantom_max_count", "phantom_gap_threshold",
    ]
    record = {
        "cli_args":       vars(args),
        "oracle_config":  {k: ORACLE_CONFIG[k] for k in oracle_keys if k in ORACLE_CONFIG},
        "torch_version":  torch.__version__,
        "cuda_version":   (torch.version.cuda if torch.cuda.is_available() else None),
        "cuda_devices":   (
            {i: torch.cuda.get_device_name(i)
             for i in range(torch.cuda.device_count())}
            if torch.cuda.is_available() else {}
        ),
        "saved_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    log.info(f"[config] JSON snapshot → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  ФАЗА 0: INGESTION ТЕКСТА В ГРАФ (TODO #1)
#
#  Раньше IngestionPipeline / GraphConsolidator.apply_to_graph() ни разу не
#  вызывались ни отсюда, ни из run.py — Фаза 1 всегда стартовала с пустого
#  графа, и весь текстовый ingestion-пайплайн (Layers 1-4 в oracle1.py)
#  существовал только на бумаге. --corpus_dir / --seed_docs ниже дают явную
#  точку входа перед Фазой 1, используя ИМЕННО те классы, что для этого уже
#  построены: IngestionPipeline (chunk → LLM extract → validate → assemble)
#  + GraphConsolidator.apply_to_graph() (trust-weighted мердж повторных
#  наблюдений — см. docstring GraphConsolidator и test_consolidation_order.py
#  по TODO #9).
#
#  ВАЖНОЕ ОГРАНИЧЕНИЕ: LocalLLMClient.complete() в текущем виде — заглушка
#  (raise NotImplementedError). Ingestion физически не может отработать без
#  реального LLM-бэкенда (подкласс LocalLLMClient или другой провайдер,
#  зарегистрированный через CONFIG['llm_default_model']/get_llm_client()).
#  Это не то, что можно починить внутри этого файла — ниже сделана ранняя,
#  понятная ошибка вместо непрозрачного traceback-а из середины пайплайна.
# ══════════════════════════════════════════════════════════════════════════════

def build_ingestion_documents(
    args: argparse.Namespace,
    processed_document_ids: Optional[set] = None,
) -> Tuple[List["BatchDocument"], Optional[DocumentQueue], Dict[str, int]]:
    """Build documents from a recursive corpus plus optional explicit JSON.

    Directory ingestion is mediated by ``DocumentQueue``.  It extracts
    content and embedded bibliographic metadata before calling an LLM, then
    rejects repeated content hashes and (unless --allow_revisions is passed)
    a changed file carrying the same embedded title+author.  The queue itself
    is reconciled with RuntimeGraph.processed_document_ids, so it remains
    correct across a process crash between graph save and queue update.
    """
    docs: List["BatchDocument"] = []
    queue: Optional[DocumentQueue] = None
    queue_stats: Dict[str, int] = {}

    if args.corpus_dir:
        corpus_path = Path(args.corpus_dir)
        queue_path = Path(args.document_queue)
        queue = DocumentQueue(queue_path)
        try:
            selected, queue_stats = queue.select_new(
                corpus_path,
                processed_document_ids=set(processed_document_ids or set()),
                allow_revisions=args.allow_revisions,
                retry_failed=args.retry_failed_documents,
                document_python=args.document_python or None,
                max_chars=args.max_document_chars,
            )
        except Exception as exc:
            log.error(f"[ingest] не удалось открыть corpus/queue: {exc}")
            selected = []
        for document in selected:
            try:
                source = str(document.path.relative_to(corpus_path))
            except ValueError:
                source = str(document.path)
            docs.append(BatchDocument(
                text=document.text,
                source=source,
                # File mtime is a queue-maintenance detail, never historical
                # evidence.  Ingestion asks the LLM to infer the document's
                # date and source type from the document itself.
                timestamp=0.0,
                metadata={
                    "document_content_hash": document.content_hash,
                    "document_title": document.title,
                    "document_authors": document.authors,
                    "document_path": str(document.path),
                    "document_metadata": document.metadata,
                    "document_warning": document.warning,
                },
            ))
            if document.warning:
                log.warning(f"[ingest] {source}: {document.warning}")
        log.info(
            f"[ingest] corpus {corpus_path}: queued={len(selected)}, "
            + ", ".join(f"{name}={value}" for name, value in sorted(queue_stats.items()))
        )

    if args.seed_docs:
        seed_path = Path(args.seed_docs)
        if not seed_path.is_file():
            log.error(f"[ingest] --seed_docs файл не найден: {seed_path}")
        else:
            raw_entries: List[Dict[str, Any]] = []
            try:
                seed_text = _read_text_file(seed_path)
                if seed_path.suffix == ".jsonl":
                    raw_entries = [json.loads(line) for line in seed_text.splitlines()
                                   if line.strip()]
                else:
                    loaded = json.loads(seed_text)
                    raw_entries = loaded if isinstance(loaded, list) else loaded.get("documents", [])
            except Exception as e:
                log.error(f"[ingest] не удалось разобрать --seed_docs {seed_path}: {e}")
                raw_entries = []

            n_before = len(docs)
            for entry in raw_entries:
                text = entry.get("text")
                if text is None and "path" in entry:
                    try:
                        text = _read_text_file(Path(entry["path"]))
                    except Exception as e:
                        log.warning(f"[ingest] пропускаю {entry.get('path')}: {e}")
                        continue
                if not text or not text.strip():
                    continue
                docs.append(BatchDocument(
                    text=text,
                    source=entry.get("source", entry.get("path", "seed_doc")),
                    # --seed_docs is retained only as a development fixture.
                    # Do not allow a caller to smuggle a manual date/domain/
                    # source type into the physical graph.
                    timestamp=0.0,
                ))
            log.info(f"[ingest] --seed_docs: {seed_path} -> {len(docs) - n_before} документ(ов)")

    return docs, queue, queue_stats


def run_ingestion_phase(system: Dict[str, Any], args: argparse.Namespace) -> None:
    """Фаза 0: прогнать --corpus_dir / --seed_docs через IngestionPipeline +
    GraphConsolidator.apply_to_graph() ДО начала Фазы 1. No-op, если ни один
    из флагов не передан."""
    if not args.corpus_dir and not args.seed_docs:
        return

    documents, queue, queue_stats = build_ingestion_documents(
        args, processed_document_ids=getattr(system["graph"], "processed_document_ids", set()))
    if not documents:
        log.warning("[ingest] --corpus_dir/--seed_docs заданы, но документов для "
                    "ingestion не нашлось — Фаза 0 пропущена")
        return

    llm_client = system.get("llm_client")
    if llm_client is None:
        log.error(
            "[ingest] Нет рабочего локального LLM-бэкенда. Для реального "
            "извлечения задайте ORACLE_LLM_BACKEND=openai_compatible и "
            "ORACLE_LLM_ENDPOINT; для явного offline smoke-test можно задать "
            "ORACLE_LLM_BACKEND=heuristic. "
            f"{len(documents)} документ(ов) НЕ будут проглочены — Фаза 1 "
            "стартует с текущим (возможно пустым) графом."
        )
        return

    log.info(f"\n{'═'*62}")
    log.info(f"  ФАЗА 0 — Text Ingestion  ({len(documents)} документ(ов))")
    log.info(f"{'═'*62}\n")

    pipeline = IngestionPipeline(
        llm_client=llm_client,
        embedder=system.get("embedder"),
        graph=system["graph"],
        # Manifest: reuse the SAME ConceptClassifier build_oracle1_system()
        # already attached (system["concept_classifier"], when
        # attach_canonical=True — the default) rather than let
        # IngestionPipeline.__init__ default to constructing a second,
        # separate one. The classifier's LEARNED neural fast-path model
        # (unlike DOMAIN_TABLE/GROUP_TABLE/ALIAS_TABLE, which are
        # module-level singletons regardless) is per-instance state — two
        # instances would each accumulate their own classification_history
        # and never cross-train.
        classifier=system.get("concept_classifier"),
    )
    consolidator = GraphConsolidator()
    processor = BatchIngestionProcessor(
        pipeline,
        rate_limit_rpm=args.ingestion_rate_limit_rpm,
        max_errors=args.ingestion_max_errors,
    )

    completed_document_keys: List[str] = []
    failed_document_keys: List[str] = []
    document_node_ids: Dict[str, List[str]] = {}
    ready_results = []

    def _progress(done: int, total: int, result) -> None:
        doc = documents[done - 1]
        document_key = str(doc.metadata.get("document_content_hash", ""))
        if result.ready_to_commit():
            ready_results.append(result)
            consolidator.observe_from_ingestion_result(
                result, source_type=getattr(result, "source_type", "unknown"))
            if document_key:
                completed_document_keys.append(document_key)
                document_node_ids[document_key] = [node.id for node in result.nodes]
        elif document_key:
            failed_document_keys.append(document_key)
        log.info(f"  [ingest] {done}/{total}  '{result.source}'  "
                 f"nodes={len(result.nodes)}  "
                 f"edges={len(result.edges) + len(result.cross_doc_edges)}  "
                 f"llm_failures={result.stats.get('llm_failures', 0)}  "
                 f"llm_incomplete={result.stats.get('llm_incomplete_calls', 0)}  "
                 f"warnings={len(result.warnings)}")

    results = processor.process(documents, progress_callback=_progress)
    if not results:
        log.warning("[ingest] Ни один документ не был успешно обработан — "
                    "Фаза 1 стартует с текущим графом без изменений")
        return

    nodes_before = len(system["graph"].nodes)
    edges_before = len(system["graph"].edges)
    stats = consolidator.apply_to_graph(system["graph"])
    print_conflict_report(consolidator)

    # Zone registration: GraphConsolidator.apply_to_graph() (above) only
    # ever consolidates result.nodes/edges — it has no notion of
    # TemporalZone at all, so real zones extracted by IngestionPipeline
    # (result.zones) previously never reached the graph on this path
    # (the only place that called graph.add_zone() was
    # load_result_into_graph(), which Phase 0 does not use). Registered
    # after apply_to_graph() so the nodes a zone's geometric sweep needs to
    # test coverage against already exist. Dedup by zone.id: the same zone
    # (e.g. "AI Boom 2020-2025") is often extracted independently from
    # several documents in one corpus — add_zone() overwrites by id, which
    # is fine for a first pass but means the LAST extraction wins rather
    # than being trust-consolidated like nodes/edges are; a real
    # consolidation strategy for zone fields is a separate follow-up.
    zones_added = 0
    seen_zone_ids = set()
    for result in ready_results:
        for zone in result.zones:
            if zone.id in seen_zone_ids:
                continue
            seen_zone_ids.add(zone.id)
            system["graph"].add_zone(zone)
            zones_added += 1

    graph = system["graph"]
    redirects = consolidator._id_redirects

    def node_is_committed(node_id: str) -> bool:
        visited = set()
        while node_id in redirects and node_id not in visited:
            visited.add(node_id)
            node_id = redirects[node_id]
        return graph.get_node(node_id) is not None

    committed_document_keys = [
        content_id for content_id in completed_document_keys
        if document_node_ids.get(content_id)
        and all(node_is_committed(node_id)
                for node_id in document_node_ids[content_id])
    ]
    uncommitted_document_keys = [
        content_id for content_id in completed_document_keys
        if content_id not in set(committed_document_keys)
    ]
    graph.processed_document_ids.update(committed_document_keys)
    if ready_results:
        save_graph_snapshot(args.graph_snapshot, system)
    if queue is not None:
        if committed_document_keys:
            queue.mark(committed_document_keys, "ingested", "llm_complete_nodes_committed")
        failed_keys = list(dict.fromkeys(
            failed_document_keys + uncommitted_document_keys))
        if failed_keys:
            queue.mark(failed_keys, "failed", "llm_incomplete_or_nodes_not_committed")

    log.info(
        f"[ingest] Фаза 0 завершена: {len(results)}/{len(documents)} документ(ов), "
        f"граф {nodes_before}->{stats.nodes_after} узлов "
        f"(+{stats.nodes_after - nodes_before}), "
        f"{edges_before}->{stats.edges_after} рёбер "
        f"(+{stats.edges_after - edges_before}), "
        f"{stats.merges_applied} слияние(й), "
        f"{stats.conflicts_found} конфликт(ов) полей, "
        f"{stats.ancestry_edges_synthesised} ancestry-рёбер синтезировано, "
        f"{zones_added} временных зон зарегистрировано"
    )
    if queue is not None:
        log.info(f"[ingest] queue state: {queue.counts()}")
    # TODO #15: apply_to_graph() enforces the ancestry invariant per-call now
    # (see the fix + ConsolidationStats.ancestry_edges_synthesised), but a
    # graph-wide audit here is a cheap, independent sanity check for the
    # whole Phase 0 batch at once — visibility if something upstream (e.g. a
    # future code path that still writes nodes directly) reintroduces orphans.
    ancestry_report = system["graph"].audit_ancestry()
    if not ancestry_report["ok"]:
        log.warning(
            f"[ingest] audit_ancestry: {len(ancestry_report['orphans'])} orphan "
            f"node(s) after Фаза 0 — ancestry invariant violated: "
            f"{ancestry_report['orphans'][:5]}"
            + (" …" if len(ancestry_report["orphans"]) > 5 else "")
        )
    else:
        log.info(f"[ingest] audit_ancestry: OK, no orphan nodes "
                 f"({len(ancestry_report['primordial'])} primordial roots)")


# ══════════════════════════════════════════════════════════════════════════════
#  ORACLE-1 GRAPH SYSTEM — один шаг символической симуляции
#
#  Теперь это явно называется "graph step", а не "epoch", чтобы
#  не смешивать с нейросетевыми эпохами.
# ══════════════════════════════════════════════════════════════════════════════

def oracle_graph_step(
    system:    Dict[str, Any],
    timestamp: float,
    device:    Optional["torch.device"] = None,
) -> Dict[str, Any]:
    """Продвинуть символическую систему на один шаг.

    Обновляет extended / dynamics / topology.
    НЕ трогает нейросеть и DataLoader (кроме опционального read-only
    encode() ниже — веса не обновляются, autograd-граф не строится).
    Возвращает метрики для логирования.

    TODO #4 fix: PhantomNodeGenerator.scan_for_voids() used to always be
    called with node_latents=None (hardcoded inside Oracle1Extended.
    update_epoch()), so it always returned []. When a device is supplied,
    this now runs a single no_grad Oracle1Model.encode() pass over the
    current graph (via compute_node_latents(), which handles the empty-
    graph / TORCH_AVAILABLE=False cases by returning None) and threads the
    resulting latents through — matching what run.py's inference path
    already does for its own graph steps.
    """
    node_latents = compute_node_latents(system["model"], system["graph"], device,
                                        extended=system["extended"]) if device is not None else None
    ext_metrics  = system["extended"].update_epoch(timestamp, node_latents=node_latents)
    dyn_metrics  = system["dynamics"].run_epoch(timestamp)
    topo_metrics = system["topology"].run_epoch(timestamp)

    merged: Dict[str, Any] = {"oracle_ts": timestamp}
    merged.update({f"ext/{k}":  v for k, v in ext_metrics.items()})
    merged.update({f"dyn/{k}":  v for k, v in dyn_metrics.items()})
    merged.update({f"topo/{k}": v for k, v in topo_metrics.items()})

    alerts = ext_metrics.get("entropy_alerts", [])
    if alerts:
        log.warning(f"  [GEM] {len(alerts)} entropy alert(s): {alerts}")

    top_p = ext_metrics.get("top_pressure_nodes", [])
    if top_p:
        parts = "  ".join(f"'{t[:30]}'={p:.3f}" for t, p in top_p)
        log.info(f"  [pressure] top-5 → {parts}")

    n_phantoms = len(system["extended"].phantom_gen.phantoms)
    graph      = system["graph"]
    log.info(
        f"  [graph] nodes={len(graph.nodes)}  edges={len(graph.edges)}  "
        f"phantoms={n_phantoms}  "
        f"pre-breakthrough={dyn_metrics.get('pre_breakthrough_candidates', 0)}  "
        f"new_voids={dyn_metrics.get('new_structural_voids', 0)}  "
        f"virtual_nodes={dyn_metrics.get('total_virtual_nodes', 0)}"
    )

    ruptures = topo_metrics.get("rupture_count", 0)
    if ruptures:
        log.warning(f"  [topology] ⚡ {ruptures} ontological rupture(s) detected")

    return merged


def graph_is_ready(system: Dict[str, Any], min_nodes: int) -> bool:
    """Граф готов к нейросетевому обучению, если набрал достаточно узлов."""
    return len(system["graph"].nodes) >= min_nodes


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET BUILDER FROM LIVE GRAPH
#
#  Ключевая функция v2: строит OracleDataset из текущего состояния
#  RuntimeGraph вместо чтения статичного файла на диске. Train/val —
#  temporal split этого единого датасета, а не отдельный файл (см.
#  докстрок функции ниже).
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_from_graph(
    system:      Dict[str, Any],
    args:        argparse.Namespace,
    has_cuda:    bool,
    dl_generator: torch.Generator,
) -> Tuple[DataLoader, DataLoader, int]:
    """Собрать DataLoader из живого графа.

    Возвращает (train_loader, val_loader, n_train_samples).

    Train и val — не два файла, а temporal split ОДНОГО
    OracleDataset.from_graph(graph): самая новая доля узлов графа по
    timestamp_start (args.val_temporal_frac) уходит в валидацию, всё
    остальное — в обучение. Это делает validate() настоящим бэктестом
    (Манифест §4, «Слепки скользящего окна» / RuntimeGraph.as_of()) —
    проверкой, может ли модель, обученная только на прошлом графа,
    правильно оценивать пары, вовлекающие узлы, появившиеся уже ПОСЛЕ
    точки отсечения — вместо случайного сплита, который бы молча давал
    модели подглядеть в будущее графа и обесценил саму идею измерения
    обобщения.

    Если OracleDataset не поддерживает from_graph() — фолбэк на
    файловый датасет (старое поведение, сейчас недостижимо — см. ниже).
    """
    graph = system["graph"]

    if not hasattr(OracleDataset, "from_graph"):
        # Недостижимо на практике: from_graph — classmethod, объявленный
        # безусловно на самом классе OracleDataset (oracle1.py), так что
        # hasattr() выше всегда True. Ветка оставлена как defensive guard
        # на случай будущего рефакторинга, а не удалена молча.
        #
        # Раньше здесь было `OracleDataset(args.data_path)` — тот же класс
        # бага, что чинился ниже для val_ds: OracleDataset.__init__ ждёт
        # List[OracleExample], а не путь. Тихая порча вместо явной ошибки
        # была бы хуже, чем эта ветка вообще есть, поэтому она теперь падает
        # сразу и по делу, а не где-то в середине DataLoader. Чтобы
        # действительно включить этот путь, сначала нужен
        # OracleDataset.from_file() (или эквивалент с чётко определённым
        # форматом) — гадать формат для ветки, до которой код никогда не
        # доходит, уже отклонили в предыдущем аудите этого файла.
        raise NotImplementedError(
            "OracleDataset.from_graph() is unavailable and no file-based "
            "fallback dataset loader exists (OracleDataset.from_file() or "
            "equivalent). Implement one before OracleDataset can lose "
            "from_graph(), or before --data_path is relied on again."
        )

    all_examples = OracleDataset.from_graph(graph).examples
    log.info(
        f"[dataset] built from live graph: "
        f"{len(all_examples):,} pairs  "
        f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
    )

    # ── Temporal train/val split ────────────────────────────────────────────
    val_frac   = float(args.val_temporal_frac)
    node_birth = {nid: node.timestamp_start for nid, node in graph.nodes.items()}
    sorted_ts  = sorted(node_birth.values())

    train_ds: Optional[OracleDataset] = None
    val_ds:   Optional[OracleDataset] = None

    if len(sorted_ts) >= 2 and 0.0 < val_frac < 1.0:
        split_idx = min(max(1, int(len(sorted_ts) * (1.0 - val_frac))),
                         len(sorted_ts) - 1)
        cutoff_ts = sorted_ts[split_idx]

        def pair_ts(ex: OracleExample) -> float:
            # A pair only becomes fully "knowable" once both its endpoints
            # exist, so it's dated by whichever endpoint arrived later.
            return max(node_birth.get(ex.src_id, 0.0), node_birth.get(ex.tgt_id, 0.0))

        train_examples = [ex for ex in all_examples if pair_ts(ex) <  cutoff_ts]
        val_examples   = [ex for ex in all_examples if pair_ts(ex) >= cutoff_ts]

        if train_examples and val_examples:
            train_ds, val_ds = OracleDataset(train_examples), OracleDataset(val_examples)
            log.info(
                f"[dataset] temporal split @ cutoff_ts={cutoff_ts:.0f}: "
                f"train={len(train_examples):,} pairs  |  "
                f"val={len(val_examples):,} pairs (val_temporal_frac={val_frac})"
            )

    if train_ds is None:
        # Graph too young or too uniform in time for a real backtest split
        # (fewer than 2 distinct node timestamps, or the cutoff landed such
        # that one side came out empty). Falls back to a fixed-seed random
        # hold-out so validation is at least non-degenerate, rather than
        # blocking training entirely. This is a weaker signal than the
        # temporal split — it measures held-out pairs from the SAME window
        # of graph history, not genuinely unseen future structure — so it's
        # logged explicitly rather than silently substituted.
        log.warning(
            f"[dataset] not enough temporal spread for a backtest split "
            f"({len(sorted_ts)} distinct node timestamps) — falling back to "
            "a random hold-out. Val loss will understate how well the model "
            "actually generalises to unseen future graph structure."
        )
        shuffled = list(all_examples)
        random.Random(1337).shuffle(shuffled)
        n_val          = max(1, int(len(shuffled) * val_frac)) if shuffled else 0
        val_examples   = shuffled[:n_val]
        train_examples = shuffled[n_val:] or shuffled
        train_ds, val_ds = OracleDataset(train_examples), OracleDataset(val_examples)

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = has_cuda,
        # Bug 2 fix: OracleExample is a plain dataclass — default_collate
        # can't batch it (see oracle_collate_fn's docstring above).
        collate_fn  = oracle_collate_fn,
    )
    train_loader = DataLoader(
        train_ds, shuffle=True, drop_last=True,
        generator=dl_generator,
        **loader_kw,
    )
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)

    log.info(
        f"[dataset] train: {len(train_ds):,} samples / {len(train_loader)} batches  |  "
        f"val: {len(val_ds):,} samples / {len(val_loader)} batches  |  "
        f"effective_batch={args.batch_size * args.grad_accum}"
    )
    return train_loader, val_loader, len(train_ds)


# ══════════════════════════════════════════════════════════════════════════════
#  COLLATE
# ══════════════════════════════════════════════════════════════════════════════

def oracle_collate_fn(examples: List[Any]) -> Dict[str, "torch.Tensor"]:
    """
    Custom collate for OracleExample (Bug 2 in the audit notes).

    OracleExample is a plain dataclass — not a Tensor, Mapping, or a
    Sequence of those — so PyTorch's default collate has no rule for it and
    the DataLoader raises before model(**batch) is ever reached. This stacks
    only the tensor-shaped fields train_one_epoch()/validate() actually
    consume (src_features/tgt_features for the forward pass,
    label_exists/label_components as supervision). The non-tensor
    src_id/tgt_id identifiers are intentionally dropped here rather than
    passed through: batch[k].to(device) has no meaning for a pair of
    strings, which is exactly what would break downstream if they were kept.
    """
    return {
        "src_features": torch.tensor(
            np.stack([ex.src_features for ex in examples]), dtype=torch.float32),
        "tgt_features": torch.tensor(
            np.stack([ex.tgt_features for ex in examples]), dtype=torch.float32),
        "label_exists": torch.tensor(
            [ex.label_exists for ex in examples], dtype=torch.float32),
        "label_components": torch.tensor(
            np.stack([ex.label_components for ex in examples]), dtype=torch.float32),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN / VALIDATE LOOPS
# ══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   Oracle1Loss,
    optimizer:   optim.Optimizer,
    scaler:      GradScaler,
    ema:         EMAWeights,
    device:      torch.device,
    grad_clip:   float,
    grad_accum:  int,
    global_step: int,
    amp_enabled: bool,
    log_window:  int = 50,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)
    t_start    = time.time()

    _recent_losses: list = []
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        is_last_in_accum = (
            (batch_idx + 1) % grad_accum == 0
            or batch_idx == n_batches - 1
        )

        with autocast(enabled=amp_enabled):
            # Bug 2 fix: Oracle1Model has no forward() that accepts the full
            # OracleExample field set, and pair-sampled examples carry no
            # forecast/zone-membership supervision for Oracle1Loss.total()'s
            # other terms anyway — so this calls the new pair-level
            # forward() explicitly (rather than model(**batch), which would
            # also hand it label_exists/label_components as stray kwargs)
            # and combines existence_loss + lc*component_loss directly,
            # mirroring TemporalSplitTrainer._local_step()'s working pattern
            # exactly, since that's the supervision this dataset actually
            # provides.
            outputs   = model(src_features=batch["src_features"],
                              tgt_features=batch["tgt_features"])
            edge_mask = batch["label_exists"] > 0.5
            loss = (criterion.existence_loss(outputs["exist_probs"], batch["label_exists"])
                    + criterion.lc * criterion.component_loss(
                        outputs["pred_comps"], batch["label_components"], edge_mask))
            loss    = loss / grad_accum

        scaler.scale(loss).backward()
        unscaled_loss = loss.item() * grad_accum
        total_loss   += unscaled_loss
        _recent_losses.append(unscaled_loss)
        if len(_recent_losses) > log_window:
            _recent_losses.pop(0)

        if is_last_in_accum:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(model)
            global_step += 1

        report_every = max(1, n_batches // 5)
        if (batch_idx + 1) % report_every == 0 or batch_idx == n_batches - 1:
            elapsed  = time.time() - t_start
            progress = (batch_idx + 1) / n_batches
            eta      = elapsed / progress * (1.0 - progress)
            avg_loss = sum(_recent_losses) / len(_recent_losses)
            log.info(
                f"    {batch_idx+1:>5}/{n_batches}  "
                f"loss(avg{len(_recent_losses)})={avg_loss:.5f}  "
                f"amp_scale={scaler.get_scale():.0f}  "
                f"eta={eta:.0f}s"
            )

    return total_loss / n_batches, global_step


@torch.no_grad()
def validate(
    model:       nn.Module,
    loader:      DataLoader,
    criterion:   Oracle1Loss,
    device:      torch.device,
    ema:         EMAWeights,
    amp_enabled: bool,
) -> Tuple[float, float]:
    def _run(m: nn.Module) -> float:
        m.eval()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast(enabled=amp_enabled):
                # Bug 2 fix: same explicit forward()+existence/component-loss
                # pattern as train_one_epoch() — see the comment there.
                outputs   = m(src_features=batch["src_features"],
                             tgt_features=batch["tgt_features"])
                edge_mask = batch["label_exists"] > 0.5
                batch_loss = (criterion.existence_loss(outputs["exist_probs"], batch["label_exists"])
                              + criterion.lc * criterion.component_loss(
                                  outputs["pred_comps"], batch["label_components"], edge_mask))
                total += batch_loss.item()
        return total / len(loader)

    live_loss = _run(model)
    with ema.apply(model):
        ema_loss = _run(model)

    return live_loss, ema_loss


# ══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY SAVE (SIGINT / SIGTERM)
# ══════════════════════════════════════════════════════════════════════════════

_EMERGENCY: Dict[str, Any] = {}


def _emergency_save(sig, frame):
    log.warning(f"\n[!] Signal {sig} received — saving emergency checkpoint…")
    s = _EMERGENCY
    if s:
        try:
            save_checkpoint(
                path          = Path(s["save_dir"]) / "emergency.pt",
                epoch         = s["epoch"],
                global_step   = s["global_step"],
                model         = s["model"],
                optimizer     = s["optimizer"],
                scheduler     = s["scheduler"],
                scaler        = s["scaler"],
                ema           = s["ema"],
                best_val_loss = s["best_val_loss"],
                rng_state     = capture_rng_state(),
            )
            log.info("[!] Emergency checkpoint saved.  Safe to exit.")
        except Exception as exc:
            log.error(f"[!] Emergency save FAILED: {exc}")
        # TODO #2: protect accumulated graph structure the same way model
        # checkpoints are protected — Phase 1/interleaved graph growth can
        # represent a lot of (possibly LLM-billed) ingestion work that would
        # otherwise be lost on the same SIGINT/SIGTERM a checkpoint survives.
        if s.get("graph_snapshot_path") and s.get("system") is not None:
            try:
                save_graph_snapshot(s["graph_snapshot_path"], s["system"])
                log.info("[!] Emergency graph snapshot saved.")
            except Exception as exc:
                log.error(f"[!] Emergency graph snapshot save FAILED: {exc}")
    sys.exit(0)


# ══════════════════════════════════════════════════════════════════════════════
#  ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oracle-1 production training loop (v2 — two-phase)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    # The dataset is always derived from the live graph after Phase 0/1.
    # ``--data_path`` used to be required even though no code read it, which
    # made the documented ingestion workflow impossible to launch without a
    # fake filename. Keep the optional deprecated flag for shell-script
    # compatibility, but make the source of training data explicit.
    g.add_argument("--data_path", default=None,
                   help="Устарело и игнорируется: набор строится из живого графа после ingestion.")
    g.add_argument("--val_temporal_frac", type=float,
                   default=ORACLE_CONFIG["oracle_dataset_val_temporal_frac"],
                   help="Доля временной шкалы графа (по timestamp_start самых "
                        "новых узлов), отводимая под валидацию вместо обучения")
    g.add_argument("--save_dir",   default="./checkpoints")
    g.add_argument("--resume",     default=None,
                   help="Checkpoint path to resume from")

    g = p.add_argument_group("Model  (Oracle-1 architecture)")
    g.add_argument("--latent_dim",   type=int,   default=ORACLE_CONFIG["model_latent_dim"])
    g.add_argument("--n_gnn_layers", type=int,   default=ORACLE_CONFIG["model_n_gnn_layers"])
    g.add_argument("--n_heads",      type=int,   default=ORACLE_CONFIG["model_n_heads"])
    g.add_argument("--dropout",      type=float, default=ORACLE_CONFIG["model_dropout"])

    # ── ФАЗА 1: построение графа ──────────────────────────────────────────────
    g = p.add_argument_group("Phase 1 — Graph Build")
    g.add_argument("--graph_build_steps", type=int, default=50,
                   help="Число шагов символической симуляции ДО начала нейросетевого обучения")
    g.add_argument("--graph_only", action="store_true",
                   help="Обновить и сохранить физический граф без обучения вспомогательного оценщика")
    g.add_argument("--min_nodes_to_train", type=int, default=20,
                   help="Минимальное число узлов в графе для начала обучения нейросети")
    g.add_argument("--graph_step_delay",  type=float, default=0.0,
                   help="Пауза между шагами фазы 1 (сек). 0 = без паузы")

    # ── ФАЗА 2: нейросетевое обучение ─────────────────────────────────────────
    g = p.add_argument_group("Phase 2 — Neural Training")
    g.add_argument("--train_epochs",   type=int,   default=100,
                   help="Число эпох нейросетевого обучения (фаза 2)")
    g.add_argument("--batch_size",     type=int,   default=16)
    g.add_argument("--lr",             type=float, default=1e-4)
    g.add_argument("--weight_decay",   type=float, default=1e-5)
    g.add_argument("--grad_clip",      type=float, default=1.0)
    g.add_argument("--grad_accum",     type=int,   default=1)
    g.add_argument("--warmup_epochs",  type=int,   default=5)

    # ── Интерливинг (опциональный рост графа во время обучения) ───────────────
    g = p.add_argument_group("Interleaved Graph Growth (optional)")
    g.add_argument("--graph_interleave_every", type=int, default=0,
                   help=(
                       "Каждые N эпох обучения запускать M шагов символической симуляции "
                       "и перестраивать датасет. 0 = отключено"
                   ))
    g.add_argument("--graph_interleave_steps", type=int, default=5,
                   help="Число шагов симуляции на каждый интерлив")
    g.add_argument("--temporal_split_training", action="store_true",
                   help=(
                       "Manifest §5: после каждого интерлива (тот же --graph_interleave_every "
                       "цикл, отдельного каденса нет намеренно — иначе можно рассинхронизировать "
                       "с моментом, когда граф реально меняется) взять снимок графа ДО роста "
                       "(RuntimeGraph.as_of()) и ПОСЛЕ, и сделать TemporalSplitTrainer.train_epoch() "
                       "— локальный шаг дообучения только на k-hop окрестности реально изменившихся "
                       "узлов, а не полный проход по датасету. Требует --graph_interleave_every > 0, "
                       "иначе граф никогда не меняется и делать не на чем."
                   ))

    g = p.add_argument_group("EMA")
    g.add_argument("--ema_decay", type=float, default=0.9999)

    g = p.add_argument_group("Early Stopping")
    g.add_argument("--patience",  type=int,   default=0)
    g.add_argument("--min_delta", type=float, default=1e-5)

    g = p.add_argument_group("Checkpointing")
    g.add_argument("--save_every", type=int, default=5)
    g.add_argument("--keep_last",  type=int, default=3)

    g = p.add_argument_group("Hardware / Reproducibility")
    g.add_argument("--device",        default="cuda")
    g.add_argument("--num_workers",   type=int,  default=4)
    g.add_argument("--no_amp",        action="store_true")
    g.add_argument("--compile",       action="store_true")
    g.add_argument("--deterministic", action="store_true")
    g.add_argument("--seed",          type=int, default=42)

    # ── ФАЗА 0: ingestion текста в граф (TODO #1) ─────────────────────────────
    # Раньше IngestionPipeline/GraphConsolidator ни разу не вызывались ни из
    # train.py, ни из run.py — граф Фазы 1 всегда стартовал пустым. Оба флага
    # опциональны и могут сочетаться; при отсутствии обоих Фаза 0 просто
    # пропускается (текущее поведение по умолчанию не меняется).
    g = p.add_argument_group("Phase 0 — Text Ingestion (optional)")
    g.add_argument("--corpus_dir", default=None,
                   help=(
                       "Корень библиотеки для рекурсивного ingestion перед Фазой 1. "
                       "Поддерживаются txt/md/csv/json/html/xml/fb2/epub/rtf/doc/docx/odt/pdf; "
                       "вложенные каталоги обходятся рекурсивно."
                   ))
    g.add_argument("--document_queue", default=None,
                   help="JSON-очередь/реестр библиотеки. По умолчанию создаётся в --state_dir или artifacts.")
    g.add_argument("--document_python", default=os.environ.get("ORACLE_DOCUMENT_PYTHON", ""),
                   help="Python bundled runtime для извлечения PDF через pypdf (опционально).")
    g.add_argument("--max_document_chars", type=int, default=0,
                   help=("Явный лимит символов документа; 0 = передать LLM весь "
                         "извлечённый документ (стандарт Oracle-1).") )
    g.add_argument("--allow_revisions", action="store_true",
                   help="Обработать новую редакцию с тем же embedded title+author, но другим содержимым")
    g.add_argument("--retry_failed_documents", action="store_true",
                   help="Повторить документы, ранее завершившиеся ошибкой extraction/LLM")
    g.add_argument("--seed_docs", default=None,
                   help=(
                       "JSON или JSONL файл с документами: список объектов "
                       "{text|path, source}. Только fixture для разработки: "
                       "метаданные источника, дату и домен всё равно определяет LLM."
                   ))
    g.add_argument("--ingestion_rate_limit_rpm", type=int, default=60,
                   help="Ограничение частоты LLM-вызовов при ingestion (запросов/мин)")
    g.add_argument("--ingestion_max_errors", type=int, default=5,
                   help="Остановить ingestion после стольких подряд ошибок")
    g.add_argument("--disable_edge_validation", action="store_true",
                   help=(
                       "Отключить CONFIG['edge_validation_enabled'] для этого запуска "
                       "(по умолчанию включена — TODO #7 — что примерно удваивает "
                       "объём LLM-вызовов при ingestion; используйте этот флаг, если "
                       "это критично по стоимости/задержке)."
                   ))

    # ── Graph snapshot (TODO #2) ───────────────────────────────────────────────
    g = p.add_argument_group("Graph Snapshot (optional)")
    g.add_argument("--graph_snapshot", default=None,
                   help=(
                       "Путь для сохранения/загрузки состояния RuntimeGraph "
                       "(nodes/edges/zones/phantoms/virtual_nodes) между запусками. "
                       "Если файл существует — загружается ПЕРЕД Фазой 0/1. "
                       "Сохраняется после Фазы 1 и по завершении обучения. "
                       "Без этого флага граф, как и раньше, не переживает "
                       "перезапуск процесса."
                   ))

    g = p.add_argument_group("Persistent Oracle State")
    g.add_argument("--state_dir", default=None,
                   help=(
                       "Постоянное состояние Oracle: graph_snapshot, очередь документов и "
                       "auxiliary context-influence estimator checkpoints. Каждый запуск "
                       "с этим каталогом продолжает тот же граф, а не создаёт новый."
                   ))

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # A state directory defines one long-lived Oracle: graph, document queue
    # and the *auxiliary* context-influence estimator advance together.  A
    # per-run directory belongs to UI/log artefacts, not to a new knowledge
    # universe.  Explicit CLI paths still win so scripted deployments retain
    # full control.
    if args.state_dir:
        state_dir = Path(args.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        if args.graph_snapshot is None:
            args.graph_snapshot = str(state_dir / "graph_snapshot.pkl")
        if args.document_queue is None:
            args.document_queue = str(state_dir / "document_queue.json")
        if args.save_dir == "./checkpoints":
            args.save_dir = str(state_dir / "context_influence_checkpoints")
        if args.resume is None and not args.graph_only:
            candidate = Path(args.save_dir) / "best_model.pt"
            if candidate.exists():
                args.resume = str(candidate)
                log.info(f"[state] continuing auxiliary estimator from {candidate}")
    if args.document_queue is None:
        args.document_queue = str(Path(args.save_dir).parent / "document_queue.json")
    if args.max_document_chars < 0:
        raise SystemExit("--max_document_chars must be 0 (whole document) or a positive limit")

    seed_everything(args.seed, deterministic=args.deterministic)

    has_cuda    = torch.cuda.is_available()
    device      = torch.device("cpu" if args.graph_only else (args.device if has_cuda else "cpu"))
    amp_enabled = (not args.no_amp) and (device.type == "cuda")
    log.info(f"Device: {device}  |  AMP: {'ON' if amp_enabled else 'OFF'}")
    if has_cuda:
        log.info(f"GPU: {torch.cuda.get_device_name(device)}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Строим систему Oracle-1 ───────────────────────────────────────────────
    system    = build_oracle1_system(
        latent_dim   = args.latent_dim,
        n_gnn_layers = args.n_gnn_layers,
        n_heads      = args.n_heads,
    )
    model     = system["model"]
    criterion = system["loss"]
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"ContextInfluenceEstimator trainable params: {n_params:,} (auxiliary graph-context module)")

    # ── TODO #2: restore a graph snapshot from a previous run, if present ─────
    # Loads in place (mutates system["graph"] / system["phantom_gen"] /
    # dynamics.virtual_synth) — the shared phantom_gen wired up inside
    # build_oracle1_system() above stays correctly shared across
    # extended/dynamics/topology after this (see Oracle1Extended.__init__).
    if args.graph_snapshot and Path(args.graph_snapshot).exists():
        snap_counts = load_graph_snapshot(args.graph_snapshot, system)
        log.info(f"[graph_snapshot] Восстановлено из {args.graph_snapshot}: {snap_counts}")

    # ── TODO #7: allow disabling edge validation for this run without ─────────
    # editing CONFIG directly (it defaults to True — see the CONFIG comment
    # for the LLM-call-volume trade-off this represents).
    if args.disable_edge_validation:
        ORACLE_CONFIG["edge_validation_enabled"] = False
        log.info("[ingest] --disable_edge_validation: CONFIG['edge_validation_enabled'] = False для этого запуска")

    # ── ФАЗА 0: text ingestion, если заданы --corpus_dir/--seed_docs (TODO #1) ─
    run_ingestion_phase(system, args)

    if args.compile:
        if not hasattr(torch, "compile"):
            log.warning("[compile] torch.compile unavailable (need PyTorch >= 2.0) — skipping")
        elif device.type != "cuda":
            log.warning("[compile] torch.compile skipped on CPU")
        else:
            log.info("[compile] compiling Oracle1Model…")
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    # ── Оптимайзер и шедулер (по train_epochs, не по graph_build_steps) ───────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.train_epochs)
    scaler    = GradScaler(enabled=amp_enabled)
    ema       = EMAWeights(model, decay=args.ema_decay)

    # Manifest §5: shares model/criterion/optimizer with Phase 2's main
    # training below by design — both are just different-cadence signal
    # sources nudging the SAME weights (see _local_gradient_step, which
    # ContinuousFeedEngine.feed() also calls this same way). Constructed
    # unconditionally (cheap — no parameters of its own) but only ever
    # invoked below when --temporal_split_training is set.
    temporal_trainer = TemporalSplitTrainer(
        model, criterion, NodeFeatureBuilder(system.get("embedder")), EdgeFeatureBuilder(),
        optimizer=optimizer, device=device,
    )

    early_stop: Optional[EarlyStopping] = (
        EarlyStopping(patience=args.patience, min_delta=args.min_delta)
        if args.patience > 0 else None
    )

    start_epoch   = 0
    global_step   = 0
    best_val_loss = float("inf")

    if args.resume:
        start_epoch, global_step, best_val_loss, _ckpt_total_epochs = load_checkpoint(
            Path(args.resume), model, optimizer, scheduler, scaler, ema, device,
        )
        if _ckpt_total_epochs is not None and _ckpt_total_epochs != args.train_epochs:
            log.warning(
                f"[resume] --train_epochs={args.train_epochs} differs from "
                f"checkpoint's total_epochs={_ckpt_total_epochs}. "
                "CosineAnnealingLR period will be misaligned."
            )

    save_run_config(save_dir / "run_config.json", args)
    # TODO #2: registered before Phase 0/1 so a SIGINT/SIGTERM during graph
    # build (before the Phase 2 epoch loop below sets the rest of _EMERGENCY)
    # can still save a graph snapshot — the model/optimizer keys _emergency_
    # save also wants aren't populated yet at this point in the run, but
    # that failure is caught independently of the graph-snapshot save.
    _EMERGENCY.update(save_dir=str(save_dir), system=system,
                      graph_snapshot_path=args.graph_snapshot)
    signal.signal(signal.SIGINT,  _emergency_save)
    signal.signal(signal.SIGTERM, _emergency_save)

    # DataLoader generator — воспроизводимый shuffle после resume
    _dl_generator = torch.Generator()
    _dl_generator.manual_seed(args.seed)

    # ══════════════════════════════════════════════════════════════════════════
    #  ФАЗА 1: ПОСТРОЕНИЕ ГРАФА
    #  Нейросеть не трогается. Граф накапливает структуру.
    # ══════════════════════════════════════════════════════════════════════════

    log.info(f"\n{'═'*62}")
    log.info(f"  ФАЗА 1 — Построение графа  ({args.graph_build_steps} шагов)")
    log.info(f"  Нейросетевое обучение начнётся после: "
             f"≥{args.min_nodes_to_train} узлов или {args.graph_build_steps} шагов")
    log.info(f"{'═'*62}\n")

    for step in range(args.graph_build_steps):
        oracle_ts = float(time.time())
        log.info(f"── Graph step {step+1}/{args.graph_build_steps} {'─'*40}")

        oracle_graph_step(system, oracle_ts,
                  device=None if args.graph_only else device)

        # Проверяем ворота готовности: если граф уже готов — не ждём конца фазы
        if graph_is_ready(system, args.min_nodes_to_train):
            remaining = args.graph_build_steps - step - 1
            if remaining > 0:
                log.info(
                    f"  [gate] граф достиг {len(system['graph'].nodes)} узлов "
                    f"(>= {args.min_nodes_to_train}). "
                    f"Продолжаем ещё {remaining} шагов для накопления структуры…"
                )

        if args.graph_step_delay > 0:
            time.sleep(args.graph_step_delay)

    # Финальная проверка ворот
    n_nodes_after_build = len(system["graph"].nodes)
    if not args.graph_only and not graph_is_ready(system, args.min_nodes_to_train):
        log.error(
            f"[ABORT] После {args.graph_build_steps} шагов граф содержит только "
            f"{n_nodes_after_build} узлов (нужно >= {args.min_nodes_to_train}). "
            "Увеличьте --graph_build_steps или уменьшите --min_nodes_to_train."
        )
        sys.exit(1)

    log.info(
        f"\n[phase1] Граф построен: "
        f"{n_nodes_after_build} узлов, {len(system['graph'].edges)} рёбер. "
        f"Переходим к нейросетевому обучению.\n"
    )

    # TODO #2: save right after Phase 1 completes — this is potentially
    # expensive (LLM-billed) ingestion + graph-build work, and Phase 2
    # (many epochs of neural training) is a lot of additional time for
    # something to go wrong before the next natural save point.
    if args.graph_snapshot:
        save_graph_snapshot(args.graph_snapshot, system)

    if args.graph_only:
        log.info(
            "[graph_only] Graph update complete: %d nodes / %d edges; "
            "auxiliary neural training was skipped.",
            len(system["graph"].nodes), len(system["graph"].edges),
        )
        return

    # ══════════════════════════════════════════════════════════════════════════
    #  ФАЗА 2: НЕЙРОСЕТЕВОЕ ОБУЧЕНИЕ
    #  DataLoader строится из живого графа ПОСЛЕ фазы 1.
    # ══════════════════════════════════════════════════════════════════════════

    log.info(f"\n{'═'*62}")
    log.info(f"  ФАЗА 2 — Auxiliary Context-Influence Estimator  "
             f"(эпохи {start_epoch+1} → {args.train_epochs})")
    log.info(f"{'═'*62}\n")

    # Строим датасет из живого графа
    train_loader, val_loader, _ = build_dataset_from_graph(
        system, args, has_cuda, _dl_generator
    )

    for epoch in range(start_epoch, args.train_epochs):

        _EMERGENCY.update(
            epoch=epoch, global_step=global_step,
            model=model, optimizer=optimizer, scheduler=scheduler,
            scaler=scaler, ema=ema, best_val_loss=best_val_loss,
            save_dir=str(save_dir),
        )

        t_epoch = time.time()
        log.info(f"── Epoch {epoch+1}/{args.train_epochs} {'─'*44}")

        # ── Интерливинг: опциональные шаги роста графа во время обучения ──────
        # Символическая система продолжает работать, граф эволюционирует.
        # После каждого пакета шагов перестраиваем датасет из обновлённого графа.
        if (
            args.graph_interleave_every > 0
            and epoch > 0
            and epoch % args.graph_interleave_every == 0
        ):
            log.info(
                f"  [interleave] запускаем {args.graph_interleave_steps} шагов "
                f"символической симуляции…"
            )
            # Manifest §5: captured BEFORE the graph changes below, so
            # system["graph"].as_of(t_before_interleave) reconstructs
            # exactly the "before" slice TemporalSplitTrainer needs — every
            # node oracle_graph_step() adds below gets a timestamp_start
            # strictly later than this (they're stamped with
            # float(time.time()) at creation, called only after this line
            # runs), so as_of()'s "<=  cutoff" test correctly excludes them.
            t_before_interleave = time.time()
            for _ in range(args.graph_interleave_steps):
                oracle_graph_step(system, float(time.time()), device=device)

            log.info(
                f"  [interleave] граф вырос до "
                f"{len(system['graph'].nodes)} узлов / "
                f"{len(system['graph'].edges)} рёбер. "
                f"Перестраиваем датасет…"
            )
            train_loader, val_loader, _ = build_dataset_from_graph(
                system, args, has_cuda, _dl_generator
            )

            if args.temporal_split_training:
                # as_of() deepcopies every surviving node/edge/zone — real
                # cost on a large graph, which is exactly why this only
                # runs at the (already infrequent) interleave cadence, not
                # every epoch, and only when explicitly opted into.
                snap_before = system["graph"].as_of(t_before_interleave)
                ts_stats = temporal_trainer.train_epoch([(snap_before, system["graph"])])
                log.info(
                    f"  [temporal_split] {ts_stats.get('n_steps', 0)} локальный(х) шаг(ов), "
                    f"mean_loss={ts_stats.get('mean_loss', 0.0):.4f}  "
                    f"(+{ts_stats.get('new_nodes', 0)} узлов / "
                    f"+{ts_stats.get('new_edges', 0)} рёбер за этот интерлив)"
                )

        # ── Нейросетевой шаг ─────────────────────────────────────────────────
        train_loss, global_step = train_one_epoch(
            model        = model,
            loader       = train_loader,
            criterion    = criterion,
            optimizer    = optimizer,
            scaler       = scaler,
            ema          = ema,
            device       = device,
            grad_clip    = args.grad_clip,
            grad_accum   = args.grad_accum,
            global_step  = global_step,
            amp_enabled  = amp_enabled,
        )

        # ── Валидация ─────────────────────────────────────────────────────────
        live_loss, ema_loss = validate(
            model       = model,
            loader      = val_loader,
            criterion   = criterion,
            device      = device,
            ema         = ema,
            amp_enabled = amp_enabled,
        )

        # ── LR scheduler step ─────────────────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - t_epoch
        log.info(
            f"  train={train_loss:.6f}  live_val={live_loss:.6f}  "
            f"ema_val={ema_loss:.6f}  "
            f"lr={current_lr:.3e}  amp_scale={scaler.get_scale():.0f}  "
            f"time={epoch_time:.1f}s  step={global_step}  "
            f"graph_nodes={len(system['graph'].nodes)}"
        )

        # ── Best-model checkpoint ─────────────────────────────────────────────
        if ema_loss < best_val_loss:
            best_val_loss = ema_loss
            save_checkpoint(
                path          = save_dir / "best_model.pt",
                epoch         = epoch + 1,
                global_step   = global_step,
                model         = model,
                optimizer     = optimizer,
                scheduler     = scheduler,
                scaler        = scaler,
                ema           = ema,
                best_val_loss = best_val_loss,
                rng_state     = capture_rng_state(),
                extra         = {
                    "total_epochs": args.train_epochs,
                    "model_config": {
                        "latent_dim":   args.latent_dim,
                        "n_gnn_layers": args.n_gnn_layers,
                        "n_heads":      args.n_heads,
                        "dropout":      args.dropout,
                    },
                    "graph_snapshot": {
                        "n_nodes": len(system["graph"].nodes),
                        "n_edges": len(system["graph"].edges),
                    },
                },
            )
            log.info(f"  ★ new best EMA val = {best_val_loss:.6f}")

            # Чистые EMA-веса для быстрого инференса
            ema_only_path = save_dir / "best_ema_weights.pt"
            ema_model = Oracle1Model(
                latent_dim   = args.latent_dim,
                n_gnn_layers = args.n_gnn_layers,
                n_heads      = args.n_heads,
                dropout      = args.dropout,
            ).to(device)
            _raw_sd = (
                model._orig_mod.state_dict()
                if hasattr(model, "_orig_mod")
                else model.state_dict()
            )
            ema_model.load_state_dict(_raw_sd)
            ema.copy_to(ema_model)
            torch.save(
                {
                    "model_state_dict": ema_model.state_dict(),
                    "model_config": {
                        "latent_dim":   args.latent_dim,
                        "n_gnn_layers": args.n_gnn_layers,
                        "n_heads":      args.n_heads,
                    },
                    "epoch":        epoch + 1,
                    "ema_val_loss": ema_loss,
                },
                ema_only_path,
            )
            log.info(f"  [ema] pure EMA weights → {ema_only_path.name}")

        # ── Периодический чекпоинт + ротация ──────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_name = f"checkpoint_epoch_{epoch+1:04d}.pt"
            save_checkpoint(
                path          = save_dir / ckpt_name,
                epoch         = epoch + 1,
                global_step   = global_step,
                model         = model,
                optimizer     = optimizer,
                scheduler     = scheduler,
                scaler        = scaler,
                ema           = ema,
                best_val_loss = best_val_loss,
                rng_state     = capture_rng_state(),
            )
            rotate_checkpoints(save_dir, prefix="checkpoint_epoch_",
                               keep=args.keep_last)

        # ── EarlyStopping ──────────────────────────────────────────────────────
        if early_stop is not None and early_stop.step(ema_loss):
            log.warning(f"[EarlyStopping] triggered at epoch {epoch+1}")
            break

        log.info(f"{'─'*62}")

    # ── Финальный чекпоинт ────────────────────────────────────────────────────
    save_checkpoint(
        path          = save_dir / "final_model.pt",
        epoch         = args.train_epochs,
        global_step   = global_step,
        model         = model,
        optimizer     = optimizer,
        scheduler     = scheduler,
        scaler        = scaler,
        ema           = ema,
        best_val_loss = best_val_loss,
        rng_state     = capture_rng_state(),
    )

    # TODO #2: final snapshot — picks up any growth from interleaved graph
    # steps during Phase 2 that happened after the post-Phase-1 save above.
    if args.graph_snapshot:
        save_graph_snapshot(args.graph_snapshot, system)

    log.info(f"\n{'═'*62}")
    log.info(f"  Training complete.  Best EMA val loss: {best_val_loss:.6f}")
    log.info(f"  Graph at end: {len(system['graph'].nodes)} nodes / "
             f"{len(system['graph'].edges)} edges")
    log.info(f"  Artefacts in: {save_dir.resolve()}")
    log.info(f"{'═'*62}")


if __name__ == "__main__":
    main()
