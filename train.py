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

  ФАЗА 2: НЕЙРОСЕТЕВОЕ ОБУЧЕНИЕ (train_epochs эпох)
    • DataLoader строится ПОСЛЕ фазы 1 из живого RuntimeGraph
    • Если граф изменился существенно (rebuild_every шагов фазы 1
      продолжают работать опционально) — датасет перестраивается
    • Стандартный цикл: train_one_epoch → validate → LR step → checkpoint
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
import copy
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
from torch.utils.data import DataLoader, Dataset

# ── Oracle-1 imports ──────────────────────────────────────────────────────────
from oracle1_refactored import (
    Oracle1Model,
    Oracle1Loss,
    OracleDataset,
    build_oracle1_system,
    CONFIG as ORACLE_CONFIG,
)

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
    path.write_text(json.dumps(record, indent=2, default=str))
    log.info(f"[config] JSON snapshot → {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
#  ORACLE-1 GRAPH SYSTEM — один шаг символической симуляции
#
#  Теперь это явно называется "graph step", а не "epoch", чтобы
#  не смешивать с нейросетевыми эпохами.
# ══════════════════════════════════════════════════════════════════════════════

def oracle_graph_step(
    system:    Dict[str, Any],
    timestamp: float,
) -> Dict[str, Any]:
    """Продвинуть символическую систему на один шаг.

    Обновляет extended / dynamics / topology.
    НЕ трогает нейросеть и DataLoader.
    Возвращает метрики для логирования.
    """
    ext_metrics  = system["extended"].update_epoch(timestamp)
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
#  RuntimeGraph вместо чтения статичного файла на диске.
#  Если OracleDataset поддерживает from_graph() — используем его.
#  Иначе — фолбэк на файловый снапшот.
# ══════════════════════════════════════════════════════════════════════════════

def build_dataset_from_graph(
    system:      Dict[str, Any],
    val_path:    str,
    args:        argparse.Namespace,
    has_cuda:    bool,
    dl_generator: torch.Generator,
) -> Tuple[DataLoader, DataLoader, int]:
    """Собрать DataLoader из живого графа.

    Возвращает (train_loader, val_loader, n_train_samples).

    Если OracleDataset не поддерживает from_graph() — фолбэк на
    файловый датасет (старое поведение), с предупреждением.
    """
    graph = system["graph"]

    # Пробуем построить датасет из живого графа
    if hasattr(OracleDataset, "from_graph"):
        train_ds = OracleDataset.from_graph(graph)
        log.info(
            f"[dataset] built from live graph: "
            f"{len(train_ds):,} samples  "
            f"({len(graph.nodes)} nodes, {len(graph.edges)} edges)"
        )
    else:
        # OracleDataset ещё не поддерживает from_graph() —
        # используем файловый путь, но предупреждаем.
        log.warning(
            "[dataset] OracleDataset.from_graph() not available. "
            "Falling back to --data_path snapshot.  "
            "Implement OracleDataset.from_graph(graph) to enable live graph training."
        )
        train_ds = OracleDataset(args.data_path)

    val_ds = OracleDataset(val_path)

    loader_kw = dict(
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        pin_memory  = has_cuda,
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
            outputs = model(**batch)
            loss    = criterion.total(**outputs, **batch)
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
                outputs = m(**batch)
                total  += criterion.total(**outputs, **batch).item()
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
    g.add_argument("--data_path",  required=True,
                   help="Путь к начальному файловому датасету (фолбэк если from_graph недоступен)")
    g.add_argument("--val_path",   required=True,
                   help="Путь к валидационному датасету")
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

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    seed_everything(args.seed, deterministic=args.deterministic)

    has_cuda    = torch.cuda.is_available()
    device      = torch.device(args.device if has_cuda else "cpu")
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
    log.info(f"Oracle1Model trainable params: {n_params:,}")

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

        oracle_graph_step(system, oracle_ts)

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
    if not graph_is_ready(system, args.min_nodes_to_train):
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

    # ══════════════════════════════════════════════════════════════════════════
    #  ФАЗА 2: НЕЙРОСЕТЕВОЕ ОБУЧЕНИЕ
    #  DataLoader строится из живого графа ПОСЛЕ фазы 1.
    # ══════════════════════════════════════════════════════════════════════════

    log.info(f"\n{'═'*62}")
    log.info(f"  ФАЗА 2 — Нейросетевое обучение  "
             f"(эпохи {start_epoch+1} → {args.train_epochs})")
    log.info(f"{'═'*62}\n")

    # Строим датасет из живого графа
    train_loader, val_loader, _ = build_dataset_from_graph(
        system, args.val_path, args, has_cuda, _dl_generator
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
            for _ in range(args.graph_interleave_steps):
                oracle_graph_step(system, float(time.time()))

            log.info(
                f"  [interleave] граф вырос до "
                f"{len(system['graph'].nodes)} узлов / "
                f"{len(system['graph'].edges)} рёбер. "
                f"Перестраиваем датасет…"
            )
            train_loader, val_loader, _ = build_dataset_from_graph(
                system, args.val_path, args, has_cuda, _dl_generator
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

    log.info(f"\n{'═'*62}")
    log.info(f"  Training complete.  Best EMA val loss: {best_val_loss:.6f}")
    log.info(f"  Graph at end: {len(system['graph'].nodes)} nodes / "
             f"{len(system['graph'].edges)} edges")
    log.info(f"  Artefacts in: {save_dir.resolve()}")
    log.info(f"{'═'*62}")


if __name__ == "__main__":
    main()
