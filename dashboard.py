"""Local control room for Oracle-1.

The server binds to 127.0.0.1 by default and starts no external services.
Training and inference each run in their own child process, one at a time.
Per-run logs and JSON results remain reproducible under ``artifacts/runs``
while one long-lived Oracle state lives in ``artifacts/oracle_state``.

Run:
    python dashboard.py
Then open http://127.0.0.1:7861
"""

from __future__ import annotations

import argparse
import atexit
import base64
import hashlib
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import venv
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS_FILE = ROOT / "requirements.txt"
VENV_MARKER = VENV_DIR / ".oracle1-requirements.sha256"


def _venv_python() -> Path:
    executable = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV_DIR / executable


def _inside_project_venv() -> bool:
    return Path(sys.prefix).resolve() == VENV_DIR.resolve()


def _clean_child_environment() -> Dict[str, str]:
    environment = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV"):
        environment.pop(key, None)
    environment["PYTHONUTF8"] = "1"
    return environment


def _check_venv(python: Path, environment: Dict[str, str]):
    pip_check = subprocess.run(
        [str(python), "-m", "pip", "check"], cwd=str(ROOT), env=environment,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if pip_check.returncode != 0:
        return False, (pip_check.stdout + pip_check.stderr).strip()

    smoke_code = (
        "import importlib.util, torch; "
        "bad=[name for name in ('pyg_lib','torch_scatter','torch_sparse',"
        "'torch_cluster','torch_spline_conv') if importlib.util.find_spec(name)]; "
        "assert not bad, 'unmanaged compiled PyG extensions installed: '+', '.join(bad); "
        "import flask, numpy, sklearn, pypdf; "
        "from sklearn.metrics.pairwise import cosine_similarity; "
        "import torch_geometric; from torch_geometric.nn import GATv2Conv; "
        "print('Flask',flask.__version__,'NumPy',numpy.__version__,'Torch',torch.__version__,"
        "'PyG',torch_geometric.__version__,'CUDA',torch.cuda.is_available())"
    )
    smoke = subprocess.run(
        [str(python), "-c", smoke_code], cwd=str(ROOT), env=environment,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if smoke.returncode != 0:
        return False, (smoke.stdout + smoke.stderr).strip()
    return True, smoke.stdout.strip()


def _bootstrap_project_venv() -> int:
    print(
        f"[Oracle-1 setup] OS={platform.platform()} | Python={platform.python_version()} "
        f"| arch={platform.machine()} | CPUs={os.cpu_count() or 'unknown'} "
        f"| free_disk={shutil.disk_usage(ROOT).free // (1024 ** 3)} GiB",
        flush=True,
    )
    if sys.version_info < (3, 11) or sys.version_info >= (3, 13):
        print("[Oracle-1 setup] Python 3.11 or 3.12 (64-bit) is required.", flush=True)
        return 2
    if sys.maxsize <= 2 ** 32:
        print("[Oracle-1 setup] A 64-bit Python installation is required.", flush=True)
        return 2
    if not REQUIREMENTS_FILE.is_file():
        print(f"[Oracle-1 setup] Missing dependency file: {REQUIREMENTS_FILE}", flush=True)
        return 2

    requirements_hash = hashlib.sha256(REQUIREMENTS_FILE.read_bytes()).hexdigest()
    python = _venv_python()
    environment = _clean_child_environment()
    marker_hash = VENV_MARKER.read_text(encoding="ascii").strip() if VENV_MARKER.is_file() else ""
    healthy, details = _check_venv(python, environment) if python.is_file() else (False, "venv missing")

    if marker_hash != requirements_hash or not healthy:
        if not python.is_file():
            print("[Oracle-1 setup] Creating isolated .venv", flush=True)
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
            python = _venv_python()

        print("[Oracle-1 setup] Installing pinned project dependencies", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "uninstall", "-y", "pyg-lib", "torch-scatter",
             "torch-sparse", "torch-cluster", "torch-spline-conv", "torchaudio",
             "torchvision", "torchtext", "tensorflow-intel"],
            cwd=str(ROOT), env=environment, check=False,
        )
        install = subprocess.run(
            [str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
            cwd=str(ROOT), env=environment,
        )
        if install.returncode != 0:
            print("[Oracle-1 setup] Dependency installation failed; dashboard was not started.", flush=True)
            return install.returncode

        healthy, details = _check_venv(python, environment)
        if not healthy:
            print(f"[Oracle-1 setup] Environment validation failed:\n{details}", flush=True)
            return 1
        VENV_MARKER.write_text(requirements_hash, encoding="ascii")

    print(f"[Oracle-1 setup] Environment OK: {details}", flush=True)
    command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
    return subprocess.run(command, cwd=str(ROOT), env=environment).returncode


if __name__ == "__main__" and not _inside_project_venv():
    raise SystemExit(_bootstrap_project_venv())


from flask import Flask, Response, jsonify, request


ARTIFACT_ROOT = ROOT / "artifacts"
ARTIFACTS = ARTIFACT_ROOT / "runs"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
ORACLE_STATE = ROOT / "artifacts" / "oracle_state"
ORACLE_STATE.mkdir(parents=True, exist_ok=True)
STATE_SNAPSHOT = ORACLE_STATE / "graph_snapshot.pkl"
STATE_SUMMARY = ORACLE_STATE / "graph_snapshot.pkl.summary.json"
STATE_QUEUE = ORACLE_STATE / "document_queue.json"
STATE_CHECKPOINTS = ORACLE_STATE / "context_influence_checkpoints"
STATE_MODEL_CONFIG = ORACLE_STATE / "model_architecture.json"
STATE_CONFIG_OVERRIDES = ORACLE_STATE / "config_overrides.json"
ORACLE_GRAPH_API_URL = os.environ.get("ORACLE_GRAPH_API_URL", "").strip().rstrip("/")
try:
    LOCAL_VISUALIZER_NODE_LIMIT = int(os.environ.get("ORACLE_LOCAL_VISUALIZER_NODE_LIMIT", "100000"))
except ValueError:
    LOCAL_VISUALIZER_NODE_LIMIT = 100_000
LOCAL_VISUALIZER_NODE_LIMIT = max(1, min(10_000_000, LOCAL_VISUALIZER_NODE_LIMIT))
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
INFER_MODES = {"scan", "edge"}
logger = logging.getLogger("oracle1.dashboard")
app = Flask(__name__)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError, TypeError):
        return default


def visualizer_snapshot_sources() -> List[Dict[str, Any]]:
    """List saved graph snapshots that the local viewer can load."""
    sources: List[Dict[str, Any]] = []
    candidates = [STATE_SNAPSHOT]
    if ARTIFACT_ROOT.is_dir():
        candidates.extend(ARTIFACT_ROOT.glob("*/graph_snapshot.pkl"))

    seen = set()
    for snapshot_path in candidates:
        resolved = snapshot_path.resolve()
        if resolved in seen:
            continue
        try:
            source_id = resolved.relative_to(ARTIFACT_ROOT.resolve()).as_posix()
        except ValueError:
            continue
        seen.add(resolved)
        summary_path = Path(str(snapshot_path) + ".summary.json")
        summary = read_json(summary_path, {})
        counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
        if not isinstance(counts, dict):
            counts = {}

        manifest: Dict[str, Any] = {}
        for manifest_name in ("manifest.json", "run_manifest.json"):
            candidate = read_json(snapshot_path.parent / manifest_name, {})
            if isinstance(candidate, dict) and candidate:
                manifest = candidate
                break
        mode = manifest.get("visualizer_mode") or manifest.get("run_mode")
        report_path = snapshot_path.parent / "run_report.md"
        if not mode and report_path.is_file():
            report = report_path.read_text(encoding="utf-8", errors="replace")
            if "Offline LLM simulation" in report:
                mode = "simulated_llm_response"
        if not mode and snapshot_path.parent.name.startswith("llm_simulated_test"):
            mode = "simulated_llm_response"

        is_oracle_state = resolved == STATE_SNAPSHOT.resolve()
        title = "Oracle state" if is_oracle_state else snapshot_path.parent.name
        nodes = int(counts.get("nodes", 0) or 0)
        edges = int(counts.get("edges", 0) or 0)
        sources.append({
            "id": source_id,
            "path": resolved,
            "label": f"{title} · {nodes:,} узлов / {edges:,} рёбер",
            "mode": mode or "saved_snapshot",
            "manifest_format": manifest.get("format"),
            "counts": counts,
            "modified": snapshot_path.stat().st_mtime if snapshot_path.is_file() else 0,
            "is_oracle_state": is_oracle_state,
        })

    return sources


def default_visualizer_snapshot() -> Path:
    """Prefer active state, then the richest evidence-backed simulated graph."""
    sources = visualizer_snapshot_sources()
    oracle_state = next((source for source in sources if source["is_oracle_state"]), None)
    if oracle_state and int(oracle_state["counts"].get("nodes", 0) or 0) > 0:
        return oracle_state["path"]

    simulated_previews = [
        source for source in sources
        if source.get("manifest_format") == "oracle1_simulated_llm_preview_v1"
        and int(source["counts"].get("nodes", 0) or 0) > 0
    ]
    if simulated_previews:
        best_preview = max(
            simulated_previews,
            key=lambda source: (
                int(source["counts"].get("nodes", 0) or 0),
                source.get("modified", 0),
            ),
        )
        return best_preview["path"]
    return oracle_state["path"] if oracle_state else STATE_SNAPSHOT


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def valid_run_id(value: Any) -> str:
    run_id = str(value or "")
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError("Invalid run identifier")
    return run_id


def run_dir(run_id: str) -> Path:
    target = (ARTIFACTS / valid_run_id(run_id)).resolve()
    if ARTIFACTS.resolve() not in target.parents:
        raise ValueError("Run path escapes the artifact directory")
    return target


def bounded_int(data: Dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    try:
        value = int(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def bounded_float(data: Dict[str, Any], key: str, default: float, low: float, high: float) -> float:
    try:
        value = float(data.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(low, min(high, value))


def new_run_id() -> str:
    return datetime.now().strftime("run-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


_CONFIG_DEFAULTS: Optional[Dict[str, Any]] = None


def oracle_config_defaults() -> Dict[str, Any]:
    global _CONFIG_DEFAULTS
    if _CONFIG_DEFAULTS is None:
        code = "import json; from oracle1 import CONFIG; print(json.dumps(CONFIG, ensure_ascii=False))"
        environment = os.environ.copy()
        environment.pop("ORACLE_CONFIG_OVERRIDES", None)
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=str(ROOT), capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=180,
            env=environment,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout)[-1200:]
            raise RuntimeError(f"Could not read oracle1.CONFIG: {detail}")
        _CONFIG_DEFAULTS = json.loads(result.stdout)
    return _CONFIG_DEFAULTS


def flatten_config(value: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, dict):
            flattened.update(flatten_config(item, path))
        else:
            flattened[path] = item
    return flattened


def deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_config_value(value: Any, default: Any, path: str) -> Any:
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ValueError(f"{path} must be boolean")
        return value
    if isinstance(default, int) and not isinstance(default, bool):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{path} must be an integer")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{path} must be a finite number")
        return float(value)
    if isinstance(default, (list, tuple)):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a JSON array")
        return value
    if isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{path} must be text")
        return value
    raise ValueError(f"{path} is not editable through the dashboard")


def nested_overrides(values: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    flat_defaults = flatten_config(defaults)
    result: Dict[str, Any] = {}
    for path, value in values.items():
        if path not in flat_defaults:
            raise ValueError(f"Unknown Oracle config key: {path}")
        parsed = parse_config_value(value, flat_defaults[path], path)
        if parsed == flat_defaults[path]:
            continue
        parts = path.split(".")
        cursor = result
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = parsed
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Ollama model discovery / context-window check — dashboard-side mirror of
#  oracle1.get_llm_client()'s ORACLE_LLM_NUM_CTX handling. Kept independent
#  (urllib only, no `import oracle1`) so this always-on lightweight process
#  never pays oracle1's own heavy import cost — that stays inside the short-
#  lived train.py/run.py subprocesses, same separation the rest of this file
#  already uses.
# ══════════════════════════════════════════════════════════════════════════

def ollama_base_url(endpoint: str) -> Optional[str]:
    match = re.match(r"^(https?://[^/]+)", endpoint.strip())
    return match.group(1) if match else None


def fetch_ollama_models(base_url: str) -> List[Dict[str, Any]]:
    req = urllib.request.Request(f"{base_url}/api/tags")
    with urllib.request.urlopen(req, timeout=5) as response:
        data = json.loads(response.read().decode("utf-8"))
    return [
        {"name": m.get("name"), "size": m.get("size")}
        for m in (data.get("models") or []) if m.get("name")
    ]


def fetch_ollama_context_info(base_url: str, model: str) -> Dict[str, Any]:
    body = json.dumps({"model": model}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/api/show", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as response:
        info = json.loads(response.read().decode("utf-8"))
    baked_in = None
    for line in (info.get("parameters") or "").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx":
            try:
                baked_in = int(parts[1])
            except ValueError:
                pass
    ceiling = None
    for key, val in (info.get("model_info") or {}).items():
        if key.endswith(".context_length"):
            try:
                ceiling = int(val)
            except (TypeError, ValueError):
                pass
    return {"baked_in_num_ctx": baked_in, "architectural_ceiling": ceiling}


class JobManager:
    """One local long-running child at a time, with durable tail logs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._job: Optional[Dict[str, Any]] = None
        self._lines: deque[str] = deque(maxlen=600)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            job = dict(self._job) if self._job else None
            if job and self._process is not None:
                job["returncode"] = self._process.poll()
            return {"job": job, "log_tail": list(self._lines)}

    def start(self, kind: str, run_id: str, command: List[str], environment: Dict[str, str]) -> Dict[str, Any]:
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("Another Oracle-1 job is already running")
            directory = run_dir(run_id)
            directory.mkdir(parents=True, exist_ok=True)
            log_name = {
                "training": "training.log",
                "graph_update": "graph-update.log",
                "simulation": "simulation.log",
            }.get(kind, "inference.log")
            log_path = directory / log_name
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            self._process = process
            self._lines.clear()
            self._job = {
                "kind": kind,
                "run_id": run_id,
                "pid": process.pid,
                "started_at": utc_now(),
                "command": command,
                "log": log_path.name,
                "state": "running",
            }
            thread = threading.Thread(
                target=self._collect,
                args=(process, log_path, run_id, kind),
                daemon=True,
            )
            thread.start()
            return dict(self._job)

    def _collect(self, process: subprocess.Popen[str], log_path: Path, run_id: str, kind: str) -> None:
        with log_path.open("w", encoding="utf-8", buffering=1) as log_handle:
            assert process.stdout is not None
            for line in process.stdout:
                log_handle.write(line)
                with self._lock:
                    self._lines.append(line.rstrip())
        exit_code = process.wait()
        manifest_path = run_dir(run_id) / "run.json"
        manifest = read_json(manifest_path, {})
        manifest["updated_at"] = utc_now()
        manifest["last_job"] = kind
        manifest["state"] = "completed" if exit_code == 0 else "failed"
        manifest["exit_code"] = exit_code
        write_json(manifest_path, manifest)
        if not ORACLE_GRAPH_API_URL:
            VISUALIZER.resume_after_job()
        with self._lock:
            if self._process is process:
                self._job = {
                    **(self._job or {}), "state": manifest["state"],
                    "finished_at": manifest["updated_at"], "returncode": exit_code,
                }

    def stop(self) -> Dict[str, Any]:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise RuntimeError("No running job")
            self._process.terminate()
            return {"state": "stopping", "pid": self._process.pid}


JOBS = JobManager()


class VisualizerServerManager:
    """Keep the graph-window API loaded once, refreshing it after snapshot writes."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen[Any]] = None
        self._port: Optional[int] = None
        self._snapshot_stamp: Optional[int] = None
        self._loaded_snapshot: Optional[Path] = None
        self._selected_snapshot = default_visualizer_snapshot()
        self._requested = False
        self._paused = False

    def selected_snapshot(self) -> Path:
        """Return the snapshot currently selected for graph viewing."""
        with self._lock:
            return self._selected_snapshot

    def select_snapshot(self, snapshot_path: Path) -> None:
        """Switch the viewer to an artifact snapshot without changing Oracle state."""
        resolved = snapshot_path.resolve()
        with self._lock:
            if resolved == self._selected_snapshot:
                return
            self._selected_snapshot = resolved
            self._stop_locked()

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            return int(listener.getsockname()[1])

    def _stop_locked(self) -> None:
        process = self._process
        self._process = None
        self._snapshot_stamp = None
        self._loaded_snapshot = None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)

    def ensure_current(self, snapshot_path: Optional[Path] = None) -> int:
        snapshot_path = snapshot_path or self.selected_snapshot()
        if not snapshot_path.is_file():
            raise FileNotFoundError("Graph snapshot is not available yet")
        summary_path = Path(str(snapshot_path) + ".summary.json")
        summary = read_json(summary_path, {})
        node_count = summary.get("counts", {}).get("nodes") if isinstance(summary, dict) else None
        if node_count is None:
            raise RuntimeError("Local visualizer requires a graph summary; connect an indexed graph API for legacy snapshots")
        if int(node_count) > LOCAL_VISUALIZER_NODE_LIMIT:
            raise RuntimeError(
                f"Local snapshot viewer limit is {LOCAL_VISUALIZER_NODE_LIMIT:,} nodes "
                f"(graph has {int(node_count):,}); set ORACLE_GRAPH_API_URL to a live indexed graph API"
            )
        snapshot_stamp = snapshot_path.stat().st_mtime_ns
        with self._lock:
            self._requested = True
            if self._paused:
                raise RuntimeError("Local graph viewer is paused while a graph operation is running")
            if (self._process is not None and self._process.poll() is None
                    and self._loaded_snapshot == snapshot_path.resolve()
                    and self._snapshot_stamp == snapshot_stamp and self._port is not None):
                return self._port

            self._stop_locked()
            ports = [self._port] if self._port is not None else []
            ports.append(self._free_port())
            last_error = "Visualizer API did not become ready"
            for port in dict.fromkeys(ports):
                command = [
                    sys.executable, "visualizer_server.py",
                    "--snapshot", str(snapshot_path), "--port", str(port),
                ]
                process = subprocess.Popen(
                    command, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        with urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/search?q=&limit=1", timeout=0.5
                        ) as response:
                            if response.status == 200:
                                self._process = process
                                self._port = port
                                self._snapshot_stamp = snapshot_stamp
                                self._loaded_snapshot = snapshot_path.resolve()
                                return port
                    except (OSError, urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                last_error = f"Visualizer process failed to bind port {port}"
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
            raise RuntimeError(last_error)

    def refresh_if_running(self) -> None:
        with self._lock:
            should_refresh = self._requested and not self._paused
            snapshot_path = self._selected_snapshot
        if should_refresh and snapshot_path.is_file():
            self.ensure_current()

    def pause_for_job(self) -> None:
        with self._lock:
            self._paused = True
            self._stop_locked()

    def resume_after_job(self) -> None:
        with self._lock:
            self._paused = False
            requested = self._requested
            snapshot_path = self._selected_snapshot
        if requested and snapshot_path.is_file():
            try:
                self.ensure_current()
            except (FileNotFoundError, RuntimeError) as exc:
                logger.warning("[Visualizer] not restarted after graph job: %s", exc)

    def stop(self) -> None:
        with self._lock:
            self._requested = False
            self._paused = True
            self._stop_locked()


VISUALIZER = VisualizerServerManager()
atexit.register(VISUALIZER.stop)


def start_job(kind: str, run_id: str, command: List[str], environment: Dict[str, str]) -> Dict[str, Any]:
    local_viewer = not ORACLE_GRAPH_API_URL
    if local_viewer:
        VISUALIZER.pause_for_job()
    try:
        return JOBS.start(kind, run_id, command, environment)
    except Exception:
        if local_viewer:
            VISUALIZER.resume_after_job()
        raise


def run_overview(run_id: str) -> Dict[str, Any]:
    directory = run_dir(run_id)
    manifest = read_json(directory / "run.json", {})
    # Runs are audit records, not separate knowledge universes.  Every view
    # therefore points at the current persistent Oracle graph.
    summary = read_json(STATE_SUMMARY, {})
    result_files = []
    results_dir = directory / "results"
    if results_dir.is_dir():
        for path in sorted(results_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
            result_files.append({"name": path.name, "modified": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()})
    return {
        "id": run_id,
        "manifest": manifest,
        "summary": summary,
        "has_checkpoint": (STATE_CHECKPOINTS / "best_ema_weights.pt").exists(),
        "results": result_files,
    }


def configure_llm(payload: Dict[str, Any], environment: Dict[str, str]) -> Dict[str, Any]:
    """Translate the dashboard's LLM form into the ORACLE_LLM_* environment
    contract oracle1.get_llm_client() reads. Credentials/endpoints stay
    process-local (never written into a run manifest verbatim below beyond
    what's already non-secret)."""
    backend = str(payload.get("llm_backend", "openai_compatible")).strip().lower()
    if backend not in {"openai_compatible", "heuristic"}:
        raise ValueError("Unsupported local LLM backend")
    environment["ORACLE_LLM_BACKEND"] = backend
    if backend == "heuristic":
        return {"backend": backend, "warning": "Heuristic mode is for pipeline smoke tests only — results are not evidence."}

    endpoint = str(payload.get("llm_endpoint", "")).strip()
    model = str(payload.get("llm_model", "")).strip()
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("A full http(s) local LLM endpoint is required")
    if not model or len(model) > 160:
        raise ValueError("A local LLM model name is required")
    environment["ORACLE_LLM_ENDPOINT"] = endpoint
    environment["ORACLE_LLM_MODEL"] = model

    api_key = str(payload.get("llm_api_key", ""))
    if api_key:
        environment["ORACLE_LLM_API_KEY"] = api_key

    result: Dict[str, Any] = {"backend": backend, "endpoint": endpoint, "model": model}

    num_ctx_raw = str(payload.get("llm_num_ctx", "")).strip()
    if num_ctx_raw:
        try:
            num_ctx = int(num_ctx_raw)
        except ValueError:
            raise ValueError("Context window (num_ctx) must be an integer")
        if not 256 <= num_ctx <= 4_000_000:
            raise ValueError("Context window (num_ctx) is out of a sane range")
        environment["ORACLE_LLM_NUM_CTX"] = str(num_ctx)
        result["num_ctx"] = num_ctx
        # Ollama-specific reminder, mirrors oracle1._check_ollama_context_window:
        # num_ctx only takes effect against Ollama's native /api/chat, not its
        # OpenAI-compatible /v1/chat/completions (Ollama's own docs; also
        # ollama/ollama#11409) — the pipeline itself warns again at start-up,
        # this is just the earliest, cheapest place to catch the mismatch.
        if "/v1" in endpoint and ollama_base_url(endpoint):
            result["warning"] = (
                "num_ctx is set but the endpoint is Ollama's /v1/ path, which "
                "ignores it per request — point at .../api/chat instead, or "
                "bake num_ctx into a Modelfile with `ollama create`."
            )

    timeout_raw = str(payload.get("llm_timeout_s", "0")).strip()
    try:
        timeout_value = float(timeout_raw) if timeout_raw else 0.0
    except ValueError as exc:
        raise ValueError("LLM timeout must be zero (disabled) or a number of seconds") from exc
    if not math.isfinite(timeout_value):
        raise ValueError("LLM timeout must be a finite number")
    if timeout_value <= 0:
        environment.pop("ORACLE_LLM_TIMEOUT_S", None)
        result["timeout_s"] = None
    else:
        timeout_s = max(5.0, min(24 * 3600.0, timeout_value))
        environment["ORACLE_LLM_TIMEOUT_S"] = str(timeout_s)
        result["timeout_s"] = timeout_s

    max_retries = bounded_int(payload, "llm_max_retries", 3, 0, 10)
    environment["ORACLE_LLM_MAX_RETRIES"] = str(max_retries)
    result["max_retries"] = max_retries

    return result


def configure_oracle_environment(environment: Dict[str, str]) -> None:
    if STATE_CONFIG_OVERRIDES.is_file():
        environment["ORACLE_CONFIG_OVERRIDES"] = str(STATE_CONFIG_OVERRIDES)


@app.get("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/visualizer")
def visualizer_page() -> Response:
    if ORACLE_GRAPH_API_URL:
        api_url = ORACLE_GRAPH_API_URL
    else:
        try:
            port = VISUALIZER.ensure_current()
            api_url = f"http://127.0.0.1:{port}"
        except (FileNotFoundError, RuntimeError) as exc:
            return Response(f"Граф-визуализатор недоступен: {exc}", status=503, mimetype="text/plain")
    page = (ROOT / "oracle1_visualizer.html").read_text(encoding="utf-8")
    page = page.replace(
        'value="http://127.0.0.1:8765"',
        f'value="{api_url}"',
    )
    return Response(page, mimetype="text/html")


@app.post("/api/visualizer/source")
def api_visualizer_source() -> Response:
    """Select one of the known artifact snapshots for read-only visualization."""
    payload = request.get_json(silent=True) or {}
    source_id = str(payload.get("source_id", "")).strip()
    source = next(
        (item for item in visualizer_snapshot_sources() if item["id"] == source_id),
        None,
    )
    if source is None:
        return jsonify({"error": "Unknown graph snapshot"}), 404
    if ORACLE_GRAPH_API_URL:
        return jsonify({"error": "Local snapshots are unavailable with the remote graph API"}), 400
    if not source["path"].is_file():
        return jsonify({"error": "Graph snapshot file is missing"}), 404

    VISUALIZER.select_snapshot(source["path"])
    try:
        VISUALIZER.ensure_current()
    except (OSError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"source_id": source["id"], "label": source["label"]})


@app.get("/api/config")
def api_config() -> Response:
    try:
        defaults = oracle_config_defaults()
        effective = deep_merge(defaults, read_json(STATE_CONFIG_OVERRIDES, {}))
        return jsonify({
            "defaults": flatten_config(defaults),
            "values": flatten_config(effective),
            "overrides_path": str(STATE_CONFIG_OVERRIDES),
        })
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return jsonify({"error": str(exc)}), 500


@app.put("/api/config")
def api_config_update() -> Response:
    payload = request.get_json(silent=True) or {}
    values = payload.get("values")
    if not isinstance(values, dict):
        return jsonify({"error": "values must be an object of CONFIG paths to values"}), 400
    try:
        defaults = oracle_config_defaults()
        overrides = nested_overrides(values, defaults)
        write_json(STATE_CONFIG_OVERRIDES, overrides)
        return jsonify({
            "saved": len(values),
            "values": flatten_config(deep_merge(defaults, overrides)),
        })
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        return jsonify({"error": str(exc)}), 400


@app.delete("/api/config")
def api_config_reset() -> Response:
    STATE_CONFIG_OVERRIDES.unlink(missing_ok=True)
    return jsonify({"reset": True})


@app.get("/api/status")
def api_status() -> Response:
    visualizer_graph = None
    sources = visualizer_snapshot_sources()
    selected_path = VISUALIZER.selected_snapshot()
    selected_id = selected_path.resolve().relative_to(ARTIFACT_ROOT.resolve()).as_posix()
    selected_source = next((item for item in sources if item["id"] == selected_id), None)
    if selected_source is None:
        selected_path = default_visualizer_snapshot()
        VISUALIZER.select_snapshot(selected_path)
        selected_id = selected_path.resolve().relative_to(ARTIFACT_ROOT.resolve()).as_posix()
        selected_source = next((item for item in sources if item["id"] == selected_id), None)

    try:
        if not ORACLE_GRAPH_API_URL:
            VISUALIZER.refresh_if_running()
            summary = read_json(Path(str(selected_path) + ".summary.json"), {})
            counts = summary.get("counts", {}) if isinstance(summary, dict) else {}
            visualizer_graph = {
                **(counts if isinstance(counts, dict) else {}),
                "mode": selected_source["mode"] if selected_source else "saved_snapshot",
                "label": selected_source["label"] if selected_source else selected_path.parent.name,
                "source_id": selected_id,
            }
        else:
            with urllib.request.urlopen(
                f"{ORACLE_GRAPH_API_URL}/search?q=&limit=1", timeout=1.5
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict) and isinstance(payload.get("graph"), dict):
                    visualizer_graph = payload["graph"]
    except RuntimeError:
        pass
    except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        logger.warning("[Visualizer] graph status unavailable: %s", exc)
    queue = read_json(STATE_QUEUE, {})
    documents = queue.get("documents", {}) if isinstance(queue, dict) else {}
    queue_counts: Dict[str, int] = {}
    queue_files = set()
    if isinstance(documents, dict):
        for record in documents.values():
            if isinstance(record, dict):
                state = str(record.get("state", "unknown"))
                queue_counts[state] = queue_counts.get(state, 0) + 1
                paths = record.get("paths", [])
                if isinstance(paths, list):
                    queue_files.update(
                        os.path.normcase(os.path.normpath(str(path)))
                        for path in paths if path
                    )
                path = record.get("path")
                if path:
                    queue_files.add(os.path.normcase(os.path.normpath(str(path))))
    return jsonify({
        "root": str(ROOT), "state_dir": str(ORACLE_STATE), "jobs": JOBS.snapshot(),
        "queue_counts": queue_counts,
        "queue_file_count": len(queue_files),
        "summary": read_json(STATE_SUMMARY, {}),
        "has_graph": selected_path.is_file(),
        "visualizer_backend": "remote indexed API" if ORACLE_GRAPH_API_URL else "local snapshot (bounded)",
        "visualizer_graph": visualizer_graph,
        "visualizer_source_id": selected_id,
        "visualizer_sources": [
            {
                "id": source["id"], "label": source["label"],
                "mode": source["mode"], "counts": source["counts"],
                "is_oracle_state": source["is_oracle_state"],
            }
            for source in sources
        ],
        "local_visualizer_node_limit": LOCAL_VISUALIZER_NODE_LIMIT,
    })


@app.get("/api/queue/failed")
def api_queue_failed() -> Response:
    """DocumentQueue (corpus_library.py) already records the real per-file
    exception in last_error — the [ingest] summary log line only ever
    prints an aggregate count. Surfacing the underlying records here is
    what actually answers "which files failed, and why", instead of
    leaving that buried in a JSON file on disk."""
    queue = read_json(STATE_QUEUE, {})
    documents = queue.get("documents", {}) if isinstance(queue, dict) else {}
    failed = []
    if isinstance(documents, dict):
        for record in documents.values():
            if isinstance(record, dict) and record.get("state") == "failed_extract":
                failed.append({
                    "path": record.get("path", ""),
                    "last_error": record.get("last_error", ""),
                    "updated_at": record.get("updated_at", ""),
                })
    failed.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return jsonify({"failed": failed})


@app.get("/api/runs")
def api_runs() -> Response:
    runs = []
    for directory in sorted(ARTIFACTS.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if directory.is_dir() and RUN_ID_RE.fullmatch(directory.name):
            runs.append(run_overview(directory.name))
    return jsonify({"runs": runs})


@app.get("/api/runs/<run_id>")
def api_run(run_id: str) -> Response:
    try:
        return jsonify(run_overview(run_id))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/runs/<run_id>/results/<name>")
def api_result(run_id: str, name: str) -> Response:
    if Path(name).name != name or not name.endswith(".json"):
        return jsonify({"error": "Invalid result filename"}), 400
    try:
        path = run_dir(run_id) / "results" / name
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    if not path.is_file():
        return jsonify({"error": "Result not found"}), 404
    return jsonify(read_json(path, {}))


@app.get("/api/llm/models")
def api_llm_models() -> Response:
    endpoint = request.args.get("endpoint", "").strip()
    base = ollama_base_url(endpoint)
    if not base:
        return jsonify({"error": "Provide a valid http(s) endpoint first"}), 400
    try:
        models = fetch_ollama_models(base)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return jsonify({"error": f"Could not reach {base}/api/tags ({exc}) — is Ollama running there?"}), 502
    return jsonify({"base": base, "models": models})


@app.get("/api/llm/context")
def api_llm_context() -> Response:
    endpoint = request.args.get("endpoint", "").strip()
    model = request.args.get("model", "").strip()
    base = ollama_base_url(endpoint)
    if not base or not model:
        return jsonify({"error": "Provide both endpoint and model"}), 400
    try:
        info = fetch_ollama_context_info(base, model)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return jsonify({"error": f"Could not reach {base}/api/show ({exc})"}), 502
    return jsonify(info)


@app.post("/api/train")
def api_train() -> Response:
    payload = request.get_json(silent=True) or {}
    corpus_dir = Path(str(payload.get("corpus_dir", "")).strip()).expanduser()
    if not corpus_dir.is_dir():
        return jsonify({"error": "Укажите существующую локальную директорию библиотеки"}), 400
    run_id = new_run_id()
    directory = run_dir(run_id)
    directory.mkdir(parents=True)
    environment = os.environ.copy()
    try:
        configure_oracle_environment(environment)
        llm_config = configure_llm(payload, environment)
        train_estimator = bool(payload.get("train_estimator", False))
        latent_dim = bounded_int(payload, "latent_dim", 128, 32, 1024)
        heads = bounded_int(payload, "heads", 4, 1, 16)
        if latent_dim % heads:
            raise ValueError("latent_dim must be divisible by attention heads")
        epochs = bounded_int(payload, "epochs", 10, 1, 10000)
        graph_steps = bounded_int(payload, "graph_steps", 2, 0, 10000)
        min_nodes = bounded_int(payload, "min_nodes", 5, 2, 1000000)
        batch_size = bounded_int(payload, "batch_size", 8, 1, 4096)
        learning_rate = bounded_float(payload, "learning_rate", 0.0003, 0.0000001, 1.0)
        device = str(payload.get("device", "cpu")).lower()
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device")
        layers = bounded_int(payload, "layers", 2, 1, 12)
        architecture = {"latent_dim": latent_dim, "heads": heads, "layers": layers}
        if train_estimator:
            established = read_json(STATE_MODEL_CONFIG, {})
            if established and established != architecture:
                raise ValueError(
                    "Архитектура вспомогательного оценщика уже закреплена для этого "
                    "Oracle state. Создайте новый state directory для другой архитектуры."
                )
            if not established:
                write_json(STATE_MODEL_CONFIG, architecture)
        if train_estimator:
            command = [
                sys.executable, "train.py", "--corpus_dir", str(corpus_dir.resolve()),
                "--state_dir", str(ORACLE_STATE), "--save_dir", str(STATE_CHECKPOINTS),
                "--graph_snapshot", str(STATE_SNAPSHOT), "--document_queue", str(STATE_QUEUE),
                "--graph_build_steps", str(graph_steps), "--min_nodes_to_train", str(min_nodes),
                "--train_epochs", str(epochs), "--batch_size", str(batch_size),
                "--latent_dim", str(latent_dim), "--n_heads", str(heads),
                "--n_gnn_layers", str(layers), "--lr", str(learning_rate),
                "--num_workers", "0", "--device", device,
                "--save_every", str(max(1, min(epochs, 5))),
                "--seed", str(bounded_int(payload, "seed", 42, 0, 2**31 - 1)),
            ]
        else:
            command = [
                sys.executable, "graph_update.py", "--corpus_dir", str(corpus_dir.resolve()),
                "--graph_snapshot", str(STATE_SNAPSHOT), "--document_queue", str(STATE_QUEUE),
                "--graph_steps", str(graph_steps),
            ]
        if not bool(payload.get("edge_validation", False)):
            command.append("--disable_edge_validation")
        if bool(payload.get("retry_failed", False)):
            command.append("--retry_failed_documents")
        job_kind = "training" if train_estimator else "graph_update"
        manifest = {
            "id": run_id, "created_at": utc_now(), "updated_at": utc_now(),
            "state": "queued", "kind": job_kind, "llm": llm_config,
            "source": {"corpus_dir": str(corpus_dir.resolve()), "mode": "recursive_library"},
            "oracle_config_overrides": read_json(STATE_CONFIG_OVERRIDES, {}),
            "parameters": {"epochs": epochs, "graph_steps": graph_steps, "min_nodes": min_nodes,
                           "batch_size": batch_size, "latent_dim": latent_dim, "heads": heads,
                           "layers": layers, "device": device,
                           "train_estimator": train_estimator},
        }
        write_json(directory / "run.json", manifest)
        job = start_job(job_kind, run_id, command, environment)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"run_id": run_id, "job": job}), 202


@app.post("/api/infer")
def api_infer() -> Response:
    """Runs run.py's query/forecast/scan/edge/report modes against the
    current persistent graph. This is the piece the previous /api/forecast
    stub never actually did — it read a static field and never started a
    process, even though JobManager already had an "inference" kind ready
    for exactly this."""
    payload = request.get_json(silent=True) or {}
    mode = str(payload.get("mode", "")).strip().lower()
    if mode not in INFER_MODES:
        return jsonify({"error": f"mode must be one of {sorted(INFER_MODES)}"}), 400
    if not STATE_SNAPSHOT.exists():
        return jsonify({"error": "Граф ещё не создан — сначала запустите обучение с библиотекой."}), 400

    run_id = new_run_id()
    directory = run_dir(run_id)
    directory.mkdir(parents=True)
    results_dir = directory / "results"
    results_dir.mkdir()
    try:
        device = str(payload.get("device", "cpu")).lower()
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("Unsupported device")
        top_k = bounded_int(payload, "top_k", 10, 1, 500)
        graph_steps = bounded_int(payload, "graph_steps", 0, 0, 10000)
        command = [
            sys.executable, "run.py", mode,
            "--graph_snapshot", str(STATE_SNAPSHOT),
            "--save_dir", str(STATE_CHECKPOINTS),
            "--device", device,
            "--graph_steps", str(graph_steps),
            "--top_k", str(top_k),
            "--output_json", str(results_dir / f"{mode}.json"),
        ]
        if bool(payload.get("no_neural", False)):
            command.append("--no_neural")

        parameters: Dict[str, Any] = {"mode": mode, "device": device, "top_k": top_k, "graph_steps": graph_steps}

        if mode in {"query", "forecast", "report"}:
            query = str(payload.get("query", "")).strip()
            if not query:
                raise ValueError("Запрос (query) обязателен для этого режима")
            command += ["--query", query]
            parameters["query"] = query
        if mode in {"forecast", "report"}:
            depth = bounded_int(payload, "depth", 4 if mode == "forecast" else 3, 1, 20)
            command += ["--depth", str(depth)]
            parameters["depth"] = depth
        if mode in {"scan", "report"}:
            threshold = bounded_float(payload, "threshold", 72.5, 0.0, 100.0)
            command += ["--threshold", str(threshold)]
            parameters["threshold"] = threshold
        if mode in {"query", "report"}:
            n_paths = bounded_int(payload, "n_paths", 20, 1, 500)
            command += ["--n_paths", str(n_paths)]
            parameters["n_paths"] = n_paths
        if mode == "edge":
            src = str(payload.get("src", "")).strip()
            tgt = str(payload.get("tgt", "")).strip()
            if not src or not tgt:
                raise ValueError("src и tgt обязательны для режима edge")
            command += ["--src", src, "--tgt", tgt]
            parameters["src"], parameters["tgt"] = src, tgt

        manifest = {
            "id": run_id, "created_at": utc_now(), "updated_at": utc_now(),
            "state": "queued", "kind": "inference",
            "source": {"mode": mode},
            "oracle_config_overrides": read_json(STATE_CONFIG_OVERRIDES, {}),
            "parameters": parameters,
        }
        write_json(directory / "run.json", manifest)
        environment = os.environ.copy()
        configure_oracle_environment(environment)
        job = start_job("inference", run_id, command, environment)
    except (ValueError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"run_id": run_id, "job": job}), 202


@app.post("/api/simulate")
def api_simulate() -> Response:
    payload = request.get_json(silent=True) or {}
    snapshot_path = VISUALIZER.selected_snapshot()
    if not snapshot_path.is_file():
        return jsonify({"error": "Сначала обновите граф из библиотеки."}), 400
    target_node_id = str(payload.get("target_node_id", "")).strip()
    intervention_type = str(payload.get("intervention_type", "")).strip()
    valid_types = {
        "INVESTMENT_INJECTION", "REGULATORY_REMOVAL", "BREAKTHROUGH_ADJACENT",
        "TALENT_INFLUX", "GEOPOLITICAL_ACCEL", "CUSTOM_PARAMS",
    }
    if not target_node_id or len(target_node_id) > 256:
        return jsonify({"error": "Укажите id целевого узла из визуализатора."}), 400
    if intervention_type not in valid_types:
        return jsonify({"error": "Неизвестный тип искусственного условия."}), 400
    params = payload.get("params", {})
    if not isinstance(params, dict):
        return jsonify({"error": "Параметры симуляции должны быть JSON-объектом."}), 400

    run_id = new_run_id()
    directory = run_dir(run_id)
    directory.mkdir(parents=True)
    results_dir = directory / "results"
    results_dir.mkdir()
    n_simulations = bounded_int(payload, "n_simulations", 100, 1, 1000)
    noise_sigma = bounded_float(payload, "noise_sigma", 0.20, 0.0, 2.0)
    seed = bounded_int(payload, "seed", 42, 0, 2**31 - 1)
    label = str(payload.get("label", ""))[:160]
    command = [
        sys.executable, "simulate.py",
        "--snapshot", str(snapshot_path),
        "--target_node_id", target_node_id,
        "--intervention_type", intervention_type,
        "--params_b64", base64.urlsafe_b64encode(
            json.dumps(params, ensure_ascii=False).encode("utf-8")
        ).decode("ascii"),
        "--label", label,
        "--n_simulations", str(n_simulations),
        "--noise_sigma", str(noise_sigma),
        "--seed", str(seed),
        "--output_json", str(results_dir / "simulation.json"),
    ]
    environment = os.environ.copy()
    configure_oracle_environment(environment)
    manifest = {
        "id": run_id, "created_at": utc_now(), "updated_at": utc_now(),
        "state": "queued", "kind": "simulation",
        "source": {"mode": "counterfactual", "graph_snapshot": str(snapshot_path)},
        "oracle_config_overrides": read_json(STATE_CONFIG_OVERRIDES, {}),
        "parameters": {
            "target_node_id": target_node_id, "intervention_type": intervention_type,
            "params": params, "n_simulations": n_simulations,
            "noise_sigma": noise_sigma, "seed": seed,
            "persistent_graph_changed": False,
        },
    }
    write_json(directory / "run.json", manifest)
    try:
        job = start_job("simulation", run_id, command, environment)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"run_id": run_id, "job": job}), 202


@app.post("/api/stop")
def api_stop() -> Response:
    try:
        return jsonify(JOBS.stop())
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400


INDEX_HTML = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Oracle-1 · Local Control Room</title>
<style>
:root{--ink:#e8f0ff;--muted:#9badc7;--bg:#07101f;--card:#0d1a2e;--line:#203654;--accent:#62d5b3;--accent2:#50b9f5;--warn:#ffc66d;--bad:#ff7d87}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 15% 0,#14345a 0,transparent 30%),var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}main{max-width:1500px;margin:auto;padding:26px}h1,h2,h3,p{margin-top:0}h1{font-size:30px;letter-spacing:-.7px}h2{font-size:17px;margin-bottom:14px}h3{font-size:13px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}.sub{color:var(--muted);max-width:900px}.notice{border-left:3px solid var(--warn);background:#211d14;padding:11px 13px;color:#f5dfa7;margin:18px 0;border-radius:4px;font-size:13px}.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:16px;min-width:0}.card{min-width:0;background:linear-gradient(145deg,#10213a,#0b172a);border:1px solid var(--line);border-radius:12px;padding:18px;box-shadow:0 12px 38px #0003}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px;margin:17px 0}.stat{border:1px solid var(--line);border-radius:9px;padding:11px;background:#091629}.stat b{display:block;font-size:20px;color:var(--accent)}label{display:block;color:var(--muted);font-size:12px;margin-top:9px}input,select,textarea,button{font:inherit}input,select,textarea{width:100%;color:var(--ink);background:#071424;border:1px solid #2a4366;border-radius:6px;padding:8px}textarea{min-height:90px;resize:vertical}button{cursor:pointer;color:#062019;background:var(--accent);border:0;border-radius:6px;font-weight:700;padding:10px 14px;margin-top:13px}button.alt{background:#213956;color:var(--ink);margin-left:0;padding:8px 10px;font-size:12px}button.stop{background:var(--bad);color:#2a0610}button.stop{margin-left:7px}button:disabled{opacity:.55;cursor:not-allowed}.inline{display:flex;gap:8px;align-items:flex-end}.inline>div{flex:1}.row{display:grid;grid-template-columns:repeat(3,1fr);gap:9px}.row.two{grid-template-columns:repeat(2,1fr)}.advanced{margin-top:12px}.advanced summary{cursor:pointer;color:var(--accent);font-size:13px}.run{border-top:1px solid var(--line);padding:12px 0}.run:first-child{border-top:0}.run b{display:block}.tag{display:inline-block;margin:4px 5px 0 0;padding:2px 6px;border-radius:10px;border:1px solid #345274;color:var(--muted);font-size:11px}.state{color:var(--warn)}.state.completed{color:var(--accent)}.state.failed{color:var(--bad)}pre{margin:0;background:#040b15;border:1px solid var(--line);padding:12px;border-radius:8px;max-height:280px;overflow:auto;color:#c5d6ee;white-space:pre-wrap;font-size:12px}.barline{display:grid;grid-template-columns:130px 1fr 40px;gap:8px;align-items:center;font-size:12px;margin:4px 0}.bar{height:8px;background:#173151;border-radius:6px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--accent2),var(--accent))}table{width:100%;border-collapse:collapse;font-size:12px}td,th{text-align:left;padding:6px;border-bottom:1px solid var(--line)}th{color:var(--muted)}.hidden{display:none!important}.mode-fields{display:none}.mode-fields.active{display:block}.small{font-size:11px;color:var(--muted)}code{background:#071424;padding:1px 5px;border-radius:4px}@media(max-width:900px){main{padding:16px}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(3,1fr)}.row{grid-template-columns:1fr}}
</style>
<style>
.tabs{display:flex;gap:8px;margin:14px 0}.tabs button{margin:0;background:#213956;color:var(--ink)}.tabs button.active{background:var(--accent);color:#062019}
.config-tools{display:grid;grid-template-columns:minmax(220px,1fr) auto auto;gap:10px;align-items:end;margin:12px 0}.config-group{border-top:1px solid var(--line);padding:8px 0}.config-group summary{cursor:pointer;color:var(--accent);font-weight:700}.config-row{display:grid;grid-template-columns:minmax(220px,1fr) minmax(220px,1.2fr);gap:14px;align-items:center;border-top:1px solid #1a2b42;padding:6px 0}.config-row label{margin:0;color:var(--ink);overflow-wrap:anywhere}.config-row small{display:block;color:var(--muted);font-weight:400}.config-row input,.config-row textarea{margin:0}.config-row textarea{min-height:56px;font-family:ui-monospace,Consolas,monospace;font-size:12px}.config-status{min-height:20px;color:var(--accent)}
.graph-frame{display:block;width:100%;max-width:100%;min-width:0;height:620px;border:1px solid var(--line);border-radius:8px;background:#0a0e14}.sim-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}.sim-note{border-left:3px solid var(--accent2);padding:8px 11px;color:var(--muted);background:#0a1728}
@media(max-width:900px){.config-tools{grid-template-columns:1fr}.config-row{grid-template-columns:1fr}.graph-frame{height:70vh}}
</style></head><body><main>
<h1>Oracle‑1 <span style="color:var(--accent)">Local Control Room</span></h1>
    <p class="sub">Сначала данные перестраивают физический граф. Символическая динамика и узлы-прогнозы обновляются в этом цикле; neural estimator остаётся необязательным вспомогательным слоем.</p>
<div class="notice">Запросы здесь исследуют уже существующий граф. Новые прогнозные узлы появляются при реструктуризации после добавления данных; искусственные условия запускаются отдельно и не меняют постоянный граф. Браузеру передаются только связанные окна, никогда весь граф.</div>
<div class="tabs"><button class="active" data-view="workspaceView">Граф</button><button data-view="configPanel">CONFIG</button></div>
<section class="stats" id="stats"></section>
<details class="card" id="configPanel" style="display:none;margin-bottom:16px">
    <summary style="cursor:pointer;font-weight:700;color:var(--accent)">Параметры Oracle</summary>
    <p class="small">Значения загружаются из CONFIG в oracle1.py. Изменения сохраняются как локальные overrides и применяются при следующем запуске процесса; текущий процесс не меняется.</p>
    <div class="config-tools"><label>Фильтр параметров<input id="configSearch" type="search" placeholder="например: phantom_gap_threshold"></label><button type="button" id="saveConfig">Сохранить настройки</button><button type="button" class="alt" id="resetConfig">Сбросить</button></div>
    <div id="configStatus" class="config-status small"></div><div id="configFields">Загрузка CONFIG…</div>
</details>
<div id="workspaceView">
<section class="card hidden" id="failedCard" style="margin-bottom:16px">
  <h2>Не удалось извлечь текст <span class="small" id="failedCount"></span></h2>
  <div id="failedList"></div>
</section>

<div class="grid">
<section class="card"><h2>Добавить данные и перестроить граф</h2>
<form id="trainForm">
  <label>Путь к корню локальной директории<input name="corpus_dir" required placeholder="/home/user/library или C:\Knowledge\Library"></label>
  <p class="small" style="margin:8px 0 0">Поддерживаются вложенные txt, pdf, fb2, doc/docx, odt, epub, html/xml. Путь и содержимое файлов остаются на этом компьютере — в браузер уходят только метаданные хода выполнения.</p>

  <h3 style="margin-top:16px">LLM для извлечения</h3>
  <div class="row two">
    <label>Backend<select name="llm_backend" id="backend">
      <option value="openai_compatible">OpenAI-compatible (Ollama / vLLM / LM Studio)</option>
      <option value="heuristic">Heuristic — только офлайн smoke-тест</option>
    </select></label>
    <label id="endpointLabel">Endpoint<input name="llm_endpoint" id="endpoint" placeholder="http://127.0.0.1:11434/api/chat" value="http://127.0.0.1:11434/api/chat"></label>
  </div>
  <p class="small hidden" id="endpointHint" style="margin:4px 0 0">Для Ollama используйте нативный <code>/api/chat</code>, если хотите задавать <code>num_ctx</code> — <code>/v1/chat/completions</code> его молча игнорирует (ollama/ollama#11409).</p>

  <div class="inline" id="modelRow">
    <div><label>Модель<input name="llm_model" id="model" placeholder="Найти локальную модель" list="modelList"><datalist id="modelList"></datalist></label></div>
    <div style="flex:0"><button type="button" class="alt" id="discoverModels">Найти модели</button></div>
    <div style="flex:0"><button type="button" class="alt" id="checkContext">Проверить контекст</button></div>
  </div>
  <p class="small" id="contextInfo"></p>
  <label class="hidden" id="apiKeyLabel">API key (не сохраняется на диск)<input name="llm_api_key" type="password" autocomplete="off"></label>

  <details class="advanced" open><summary>Контекст, таймаут, повторы (важно для слабого железа)</summary>
    <div class="row">
      <label>num_ctx (окно контекста)<input name="llm_num_ctx" id="numCtx" type="number" placeholder="256000"></label>
    <label>Таймаут ответа, сек (0 = отключён)<input name="llm_timeout_s" type="number" value="0" min="0"></label>
      <label>Повторов при ошибке<input name="llm_max_retries" type="number" value="3" min="0" max="10"></label>
    </div>
    <p class="small" style="margin-top:4px">На медленном железе увеличьте таймаут (например, 1800 = 30 минут) и уменьшите число повторов — каждый повтор повторяет полное ожидание генерации.</p>
  </details>

    <label><input name="train_estimator" type="checkbox" style="width:auto"> После обновления отдельно обучить вспомогательный neural estimator</label>
    <details class="advanced"><summary>Параметры итерации и опционального estimator</summary>
    <div class="row"><label>Эпохи<input name="epochs" type="number" value="10" min="1"></label><label>Шаги графа<input name="graph_steps" type="number" value="2" min="0"></label><label>Мин. узлов<input name="min_nodes" type="number" value="5" min="2"></label></div>
    <div class="row"><label>Latent dim<input name="latent_dim" type="number" value="128" step="32"></label><label>Heads<input name="heads" type="number" value="4"></label><label>Layers<input name="layers" type="number" value="2"></label></div>
    <div class="row"><label>Batch<input name="batch_size" type="number" value="8"></label><label>Learning rate<input name="learning_rate" type="number" value="0.0003" step="0.0001"></label><label>Device<select name="device"><option>cpu</option><option>cuda</option><option>mps</option></select></label></div>
    <label><input name="edge_validation" type="checkbox" checked style="width:auto"> Включить LLM-проверку извлечённых рёбер (медленнее, точнее)</label>
    <label><input name="retry_failed" type="checkbox" style="width:auto"> Повторить ранее не обработанные документы</label>
  </details>
    <button>Добавить данные и обновить граф</button>
  <button type="button" class="stop" id="stop">Остановить активную задачу</button>
</form>
</section>

<section class="card"><h2>Окно связанной части графа</h2>
    <label>Снимок для просмотра (выбор не меняет постоянное состояние Oracle)<select id="visualizerSource"></select></label>
    <p class="small" id="graphBackendNote">Режим визуализации: локальный snapshot.</p>
    <iframe id="graphFrame" class="graph-frame" src="about:blank" title="Связанная часть физического графа"></iframe>
    <h3 style="margin-top:18px">What-if: искусственные условия</h3>
    <p class="sim-note">Патч применяется временно к копии состояния внутри Monte Carlo прогонов. Смотрите оценку каскадов и боттлнеков; постоянный граф остаётся без изменений.</p>
    <form id="simForm">
        <label>Целевой узел<input name="target_node_id" id="simTargetNode" list="nodeCandidates" required placeholder="Выберите узел в визуализаторе и укажите его id"><datalist id="nodeCandidates"></datalist></label>
        <div class="sim-grid">
            <label>Условие<select name="intervention_type">
                <option value="INVESTMENT_INJECTION">Ввод инвестиций</option>
                <option value="REGULATORY_REMOVAL">Снятие регуляторного барьера</option>
                <option value="BREAKTHROUGH_ADJACENT">Смежный прорыв</option>
                <option value="TALENT_INFLUX">Приток специалистов</option>
                <option value="GEOPOLITICAL_ACCEL">Внешний ускоритель</option>
                <option value="CUSTOM_PARAMS">Пользовательские параметры</option>
            </select></label>
            <label>Число прогонов<input name="n_simulations" type="number" value="100" min="1" max="1000"></label>
        </div>
        <label>Параметры условия (JSON)<textarea name="params" id="simParams">{"amount_usd":1000000,"year":2026}</textarea></label>
        <div class="sim-grid"><label>Шум Monte Carlo<input name="noise_sigma" type="number" value="0.2" min="0" max="2" step="0.05"></label><label>Подпись<input name="label" maxlength="160" placeholder="Опционально"></label></div>
        <button>Симулировать каскадные последствия</button>
    </form>
    <h3 style="margin-top:18px">Активная задача</h3><pre id="liveLog">Нет активной задачи.</pre>
</section>
</div>
</div>

<div class="grid" id="runHistory" style="margin-top:16px">
<section class="card"><h2>Запуски</h2><div id="runs">Загрузка…</div></section>
<section class="card">
  <h2>Выбранный запуск</h2>
  <label>Запуск<select id="runSelect"></select></label>
  <div id="graphStats" class="sub" style="margin-top:10px">Выберите запуск.</div>
  <div id="charts"></div>
  <h3 style="margin-top:16px">Топ структурных узлов</h3>
  <div style="overflow:auto"><table><thead><tr><th>Узел</th><th>Домен</th><th>Potential</th><th>Readiness</th><th>Год</th></tr></thead><tbody id="topNodes"></tbody></table></div>
  <h3 style="margin-top:16px">Последний JSON-результат</h3><pre id="result">—</pre>
</section>
</div>
</main><script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(url,opt){const r=await fetch(url,opt);const d=await r.json();if(!r.ok)throw Error(d.error||'Request failed');return d}
function formData(form){const o={};for(const [k,v] of new FormData(form))o[k]=v;form.querySelectorAll('input[type=checkbox]').forEach(x=>o[x.name]=x.checked);return o}
let cache=[];
function chosen(){return $('#runSelect').value}

function setStats(status,runs){
  const s=status.summary||{},c=s.counts||{},g=status.visualizer_graph||c,q=status.queue_counts||{},job=status.jobs?.job;
    const files=status.queue_file_count??Object.values(q).reduce((a,b)=>a+b,0);
    const sourceSelect=$('#visualizerSource'),sourceOptions=status.visualizer_sources||[];
    const priorSource=sourceSelect.value;
    sourceSelect.innerHTML=sourceOptions.map(x=>`<option value="${esc(x.id)}">${esc(x.label)}</option>`).join('');
    sourceSelect.value=sourceOptions.some(x=>x.id===status.visualizer_source_id)
        ?status.visualizer_source_id:(sourceOptions.some(x=>x.id===priorSource)?priorSource:'');
    sourceSelect.disabled=status.visualizer_backend==='remote indexed API';
    sourceSelect.closest('label').classList.toggle('hidden',sourceSelect.disabled);
  $('#stats').innerHTML=[
    ['runs',runs.length],['files',files],['ingested',q.ingested||0],
        ['nodes',g.nodes||0],['edges',g.edges||0],['zones',g.zones||0],
        ['phantoms',g.phantoms||0],['virtual',g.virtual_nodes||0],
    ['job',job?.state||'idle'],
  ].map(([k,v])=>`<div class=stat><b>${esc(v)}</b>${k}</div>`).join('');
    $('#graphBackendNote').textContent=status.visualizer_graph?.mode==='simulated_llm_response'
        ?`Тестовый граф построен на синтетических ответах, не на выводах реальной LLM или науки. ${status.visualizer_graph.label||''}`
        :status.visualizer_graph?.mode==='citation_preview'
        ?`Citation preview построен по DOI из References, семантического анализа LLM не было. ${status.visualizer_graph.label||''}`
        :status.visualizer_backend==='remote indexed API'
        ?'Окна визуализации запрашиваются у внешнего индексированного API; локальные ingestion/update операции всё ещё используют in-memory RuntimeGraph.'
        :`${status.visualizer_graph?.label||'Локальный snapshot'} · ${Number(g.nodes||0).toLocaleString()} узлов / ${Number(g.edges||0).toLocaleString()} рёбер. Локальный viewer ограничен ${Number(status.local_visualizer_node_limit||0).toLocaleString()} узлами.`;
    const frame=$('#graphFrame');
    const sourceId=status.visualizer_source_id||'';
    if(status.has_graph&&(frame.dataset.loaded!=='1'||frame.dataset.source!==sourceId)){
        frame.src='/visualizer?source='+encodeURIComponent(sourceId);
        frame.dataset.loaded='1';
        frame.dataset.source=sourceId;
    }else if(!status.has_graph){
        frame.src='about:blank';
        frame.dataset.loaded='0';
        frame.dataset.source='';
    }
}

function renderRunDetail(run){
  const sum=run?.summary;
  if(!sum||!sum.counts){$('#graphStats').textContent='Граф ещё не создан, либо запуск не связан с ingestion.';$('#charts').innerHTML='';$('#topNodes').innerHTML='';return}
  const c=sum.counts;
  $('#graphStats').innerHTML=`${esc(sum.interpretation||'')}<br><br><b>${c.nodes??0}</b> узлов · <b>${c.edges??0}</b> рёбер · <b>${c.zones??0}</b> зон · <b>${c.phantoms??0}</b> фантомов · mean readiness: <b>${sum.aggregates?.mean_readiness??'—'}</b>`;
  const entries=Object.entries(sum.distributions?.domains||{}).slice(0,10);
  const max=Math.max(1,...entries.map(x=>x[1]));
  $('#charts').innerHTML=entries.length?('<h3 style="margin-top:14px">Домены</h3>'+entries.map(([k,v])=>`<div class=barline><span>${esc(k)}</span><span class=bar><i style="width:${100*v/max}%"></i></span><span>${v}</span></div>`).join('')):'';
  $('#topNodes').innerHTML=(sum.top_nodes||[]).map(n=>`<tr><td title="${esc(n.id)}">${esc(n.text)}</td><td>${esc(n.domain)}</td><td>${n.convergence_potential??'—'}</td><td>${n.readiness??'—'}</td><td>${n.year??'—'}</td></tr>`).join('')||'<tr><td colspan=5 class=sub>Нет данных.</td></tr>';
    $('#nodeCandidates').innerHTML=(sum.top_nodes||[]).map(n=>`<option value="${esc(n.id)}">${esc(n.text)}</option>`).join('');
}

async function showRun(id){
  if(!id){renderRunDetail(null);$('#result').textContent='—';return}
  const run=await api('/api/runs/'+encodeURIComponent(id));
  renderRunDetail(run);
  const last=run.results?.[0];
  if(last){const data=await api('/api/runs/'+encodeURIComponent(id)+'/results/'+encodeURIComponent(last.name));$('#result').textContent=JSON.stringify(data,null,2)}
    else $('#result').textContent='Для этого обновления отдельный результат не сохранялся.';
}

function renderRuns(runs){
  cache=runs;
  const prior=chosen();
  $('#runSelect').innerHTML=runs.length?runs.map(r=>`<option value="${esc(r.id)}">${esc(r.id)} — ${esc(r.manifest.kind||'?')} — ${esc(r.manifest.state||'unknown')}</option>`).join(''):'<option value="">Нет запусков</option>';
  if(runs.some(r=>r.id===prior))$('#runSelect').value=prior;
  $('#runs').innerHTML=runs.map(r=>{
    const m=r.manifest;
    return `<div class=run><b>${esc(r.id)} <span class="state ${esc(m.state||'')}">${esc(m.state||'unknown')}</span></b>
      <span class=tag>${esc(m.kind||'?')}</span>
    <span class=tag>${esc(m.source?.corpus_dir||m.source?.mode||'')}</span>
      ${m.llm?`<span class=tag>${esc(m.llm.backend||'')}</span>`:''}
      ${r.has_checkpoint?'<span class=tag>checkpoint</span>':''}
      <button class=alt data-run="${esc(r.id)}">Показать</button></div>`;
  }).join('')||'<span class="sub">Запусков пока нет.</span>';
  document.querySelectorAll('[data-run]').forEach(b=>b.onclick=()=>{$('#runSelect').value=b.dataset.run;showRun(b.dataset.run)});
  if(!chosen()&&runs.length)$('#runSelect').value=runs[0].id;
  showRun(chosen());
}

async function refresh(){
  try{
    const [runsResp,status]=await Promise.all([api('/api/runs'),api('/api/status')]);
    setStats(status,runsResp.runs);
    renderRuns(runsResp.runs);
    $('#liveLog').textContent=(status.jobs.log_tail||[]).join('\n')||(status.jobs.job?JSON.stringify(status.jobs.job,null,2):'Нет активной задачи.');
  }catch(e){$('#liveLog').textContent='Ошибка интерфейса: '+e.message}
}

$('#trainForm').onsubmit=async e=>{
  e.preventDefault();
  try{const d=await api('/api/train',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(formData(e.target))});alert('Запуск создан: '+d.run_id);await refresh()}
  catch(x){alert(x.message)}
};
$('#simForm').onsubmit=async e=>{
  e.preventDefault();
    try{
        const values=formData(e.target);
        values.n_simulations=Number(values.n_simulations);
        values.noise_sigma=Number(values.noise_sigma);
        values.params=JSON.parse(values.params||'{}');
        const d=await api('/api/simulate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)});
        alert('What-if симуляция запущена: '+d.run_id);
        await refresh();
    }
  catch(x){alert(x.message)}
};
const simParamExamples={
    INVESTMENT_INJECTION:{amount_usd:1000000,year:2026},
    REGULATORY_REMOVAL:{inhibitory_scale:0.5},
    BREAKTHROUGH_ADJACENT:{scientific_boost:2.0},
    TALENT_INFLUX:{headcount_multiplier:2.0,social_boost:1.5},
    GEOPOLITICAL_ACCEL:{zone_multiplier_add:1.5,duration_years:5.0},
    CUSTOM_PARAMS:{fields:{readiness_score:8.0}},
};
$('#simForm select[name=intervention_type]').onchange=e=>{
    $('#simParams').value=JSON.stringify(simParamExamples[e.target.value]||{},null,2);
};
$('#stop').onclick=async()=>{try{await api('/api/stop',{method:'POST'});await refresh()}catch(x){alert(x.message)}};
$('#runSelect').onchange=()=>showRun(chosen());

function updateBackendUI(){
  const heuristic=$('#backend').value==='heuristic';
  $('#endpointLabel').classList.toggle('hidden',heuristic);
  $('#endpointHint').classList.toggle('hidden',heuristic);
  $('#modelRow').classList.toggle('hidden',heuristic);
  $('#apiKeyLabel').classList.toggle('hidden',heuristic);
}
$('#backend').onchange=updateBackendUI;
updateBackendUI();

$('#discoverModels').onclick=async()=>{
  const endpoint=$('#endpoint').value.trim();
  if(!endpoint){alert('Сначала укажите endpoint');return}
  try{
    const d=await api('/api/llm/models?endpoint='+encodeURIComponent(endpoint));
        const models=d.models.map(m=>m.name).filter(Boolean);
        $('#modelList').innerHTML=models.map(name=>`<option value="${esc(name)}">`).join('');
        if(models.length===1)$('#model').value=models[0];
        else if(!models.includes($('#model').value.trim()))$('#model').value='';
        alert(models.length===1?`Найдена и выбрана модель: ${models[0]}.`:models.length?`Найдено моделей: ${models.length} — выберите одну из подсказок поля "Модель".`:'Ollama ответил, но моделей не найдено (ollama list пуст).');
  }catch(x){alert('Не удалось получить список моделей: '+x.message)}
};
$('#checkContext').onclick=async()=>{
  const endpoint=$('#endpoint').value.trim(),model=$('#model').value.trim();
  if(!endpoint||!model){alert('Укажите endpoint и модель');return}
  try{
    const d=await api('/api/llm/context?endpoint='+encodeURIComponent(endpoint)+'&model='+encodeURIComponent(model));
    const parts=[];
    parts.push(d.baked_in_num_ctx?`запечено в модели: ${d.baked_in_num_ctx}`:'num_ctx не запечён в Modelfile — Ollama использует свой дефолт (обычно значительно меньше, чем нужно)');
    if(d.architectural_ceiling)parts.push(`архитектурный потолок модели: ${d.architectural_ceiling}`);
    $('#contextInfo').textContent=parts.join(' · ');
    if(d.baked_in_num_ctx&&!$('#numCtx').value)$('#numCtx').value=d.baked_in_num_ctx;
  }catch(x){$('#contextInfo').textContent='Не удалось проверить: '+x.message}
};

document.querySelectorAll('.tabs button').forEach(button=>button.addEventListener('click',async()=>{
    const configView=button.dataset.view==='configPanel';
    document.querySelectorAll('.tabs button').forEach(item=>item.classList.toggle('active',item===button));
    $('#workspaceView').style.display=configView?'none':'';
    $('#runHistory').style.display=configView?'none':'';
    $('#configPanel').style.display=configView?'block':'none';
    $('#configPanel').open=configView;
    if(configView)await loadConfig();
}));

let configDefaults={},configValues={},configLoaded=false;
function configGroup(path){return path.includes('.')?path.split('.')[0]:path.split('_')[0]}
function renderConfig(values,defaults){
    configDefaults=defaults;configValues=values;
    const groups={};
    Object.keys(values).sort().forEach(path=>{
        const group=configGroup(path);
        (groups[group]??=[]).push(path);
    });
    $('#configFields').innerHTML=Object.entries(groups).map(([group,paths])=>
        `<details class="config-group"><summary>${esc(group)} <small>(${paths.length})</small></summary>`+
        paths.map(path=>{
            const value=values[path],kind=Array.isArray(value)?'json':(value===null?'json':typeof value);
            let input='';
            if(kind==='boolean')input=`<input type="checkbox" data-config-path="${esc(path)}" ${value?'checked':''} style="width:auto">`;
            else if(kind==='number')input=`<input type="number" step="${Number.isInteger(value)?'1':'any'}" data-config-path="${esc(path)}" value="${esc(value)}">`;
            else if(kind==='json')input=`<textarea data-config-path="${esc(path)}" data-kind="json">${esc(JSON.stringify(value))}</textarea>`;
            else input=`<input type="text" data-config-path="${esc(path)}" value="${esc(value)}">`;
            return `<div class="config-row" data-config-row="${esc(path.toLowerCase())}"><label>${esc(path)}<small>${kind} · default ${esc(JSON.stringify(defaults[path]))}</small></label>${input}</div>`;
        }).join('')+'</details>'
    ).join('');
    filterConfig();
}
function filterConfig(){
    const query=$('#configSearch').value.trim().toLowerCase();
    document.querySelectorAll('.config-group').forEach(group=>{
        let shown=0;
        group.querySelectorAll('.config-row').forEach(row=>{
            const match=!query||row.dataset.configRow.includes(query);
            row.style.display=match?'':'none';
            if(match)shown++;
        });
        group.style.display=shown?'':'none';
        if(query&&shown)group.open=true;
    });
}
async function loadConfig(force=false){
    if(configLoaded&&!force)return;
    $('#configStatus').textContent='Загружаю значения CONFIG…';
    try{
        const data=await api('/api/config');
        renderConfig(data.values,data.defaults);
        configLoaded=true;
        $('#configStatus').textContent=`${Object.keys(data.values).length} параметров · сохраняются в локальный override-файл`;
    }catch(error){$('#configStatus').textContent=error.message}
}
function collectConfig(){
    const values={};
    document.querySelectorAll('[data-config-path]').forEach(input=>{
        const path=input.dataset.configPath;
        if(input.type==='checkbox')values[path]=input.checked;
        else if(input.dataset.kind==='json')values[path]=JSON.parse(input.value);
        else if(input.type==='number')values[path]=Number(input.value);
        else values[path]=input.value;
    });
    return values;
}
$('#configSearch').addEventListener('input',filterConfig);
$('#saveConfig').onclick=async()=>{
    try{
        const response=await api('/api/config',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({values:collectConfig()})});
        renderConfig(response.values,configDefaults);
        $('#configStatus').textContent='Сохранено. Новые значения вступят в силу при следующем запуске процесса.';
    }catch(error){$('#configStatus').textContent=error.message}
};
$('#resetConfig').onclick=async()=>{
    try{await api('/api/config',{method:'DELETE'});configLoaded=false;await loadConfig(true);$('#configStatus').textContent='Сброшено к значениям CONFIG в oracle1.py.'}
    catch(error){$('#configStatus').textContent=error.message}
};
$('#visualizerSource').onchange=async e=>{
    try{
        await api('/api/visualizer/source',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({source_id:e.target.value})});
        await refresh();
    }catch(error){alert(error.message);await refresh()}
};

refresh();
setInterval(refresh,3000);
</script></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle-1 local control room")
    parser.add_argument("--host", default="127.0.0.1", help="Default is loopback only")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be 1..65535")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
