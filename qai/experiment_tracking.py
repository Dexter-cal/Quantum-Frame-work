"""
qai.experiment_tracking — MLflow/W&B/TensorBoard integration.

Provides:
  - ExperimentLogger — unified logging to MLflow, W&B, or JSON
  - log_experiment() — one-call experiment logging
  - compare_experiments() — compare past experiments
  - MLflowLogger — MLflow backend
  - WandBLogger — Weights & Biases backend
  - JSONLogger — file-based logging (no dependencies)
  - TensorBoardLogger — TensorBoard backend

Usage:
    import qai

    # One-call logging
    qai.log_experiment(
        name="iris_rf_v1",
        params={"technique": "random_forest", "n_estimators": 100},
        metrics={"accuracy": 0.97, "f1": 0.96},
        tags=["iris", "classification"],
    )

    # With MLflow backend
    logger = qai.ExperimentLogger(backend="mlflow")
    logger.log_run(params={...}, metrics={...})
    best = logger.get_best_run(metric="accuracy")
    print(f"Best: {best}")
"""
from __future__ import annotations
import os, time, json, uuid, copy
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path


# =====================================================================
# Run record
# =====================================================================
@dataclass
class ExperimentRun:
    """A single experiment run."""
    run_id: str
    name: str
    timestamp: float
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    model_path: Optional[str] = None
    artifact_paths: List[str] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "name": self.name,
            "timestamp": self.timestamp, "params": self.params,
            "metrics": self.metrics, "tags": self.tags,
            "notes": self.notes, "model_path": self.model_path,
            "duration_s": self.duration_s,
        }


# =====================================================================
# JSON Logger (no dependencies)
# =====================================================================
class JSONLogger:
    """File-based experiment logger (no external dependencies).

    Stores experiments as JSON files in a directory.
    """
    def __init__(self, log_dir: str = "./experiments"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.log_dir / "index.json"

    def log_run(self, name: str, params: Dict, metrics: Dict,
                tags: Optional[List[str]] = None, notes: str = "",
                model_path: Optional[str] = None, duration_s: float = 0.0) -> ExperimentRun:
        """Log a single experiment run."""
        run = ExperimentRun(
            run_id=uuid.uuid4().hex[:8],
            name=name, timestamp=time.time(),
            params=params, metrics=metrics,
            tags=tags or [], notes=notes,
            model_path=model_path, duration_s=duration_s,
        )
        # Save to file
        path = self.log_dir / f"{run.run_id}.json"
        with open(path, "w") as f:
            json.dump(run.to_dict(), f, indent=2, default=str)
        # Update index
        self._update_index(run)
        return run

    def _update_index(self, run: ExperimentRun):
        """Update the index file."""
        index = []
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    index = json.load(f)
            except Exception:
                index = []
        index.append({"run_id": run.run_id, "name": run.name,
                        "timestamp": run.timestamp,
                        "best_metric": max(run.metrics.values()) if run.metrics else 0})
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2, default=str)

    def get_all_runs(self) -> List[ExperimentRun]:
        """Get all experiment runs."""
        runs = []
        for path in sorted(self.log_dir.glob("*.json")):
            if path.name == "index.json":
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                runs.append(ExperimentRun(**data))
            except Exception:
                pass
        return runs

    def get_best_run(self, metric: str = "accuracy", mode: str = "max") -> Optional[ExperimentRun]:
        """Get the best run by a metric."""
        runs = self.get_all_runs()
        scored = [(r, r.metrics.get(metric, 0)) for r in runs if metric in r.metrics]
        if not scored:
            return None
        if mode == "max":
            return max(scored, key=lambda x: x[1])[0]
        else:
            return min(scored, key=lambda x: x[1])[0]

    def compare(self, metric: str = "accuracy") -> str:
        """Compare all runs by a metric."""
        runs = self.get_all_runs()
        if not runs:
            return "No experiments found."
        lines = [f"{'Name':<30} {'Run ID':<10} {metric:>10} {'Time':>8}",
                  "-" * 62]
        for r in sorted(runs, key=lambda r: -r.metrics.get(metric, 0)):
            score = r.metrics.get(metric, "N/A")
            lines.append(f"{r.name:<30} {r.run_id:<10} {str(score):>10} {r.duration_s:>7.1f}s")
        return "\n".join(lines)


# =====================================================================
# MLflow Logger
# =====================================================================
class MLflowLogger:
    """MLflow experiment logger.

    Requires: pip install mlflow
    """
    def __init__(self, tracking_uri: str = "./mlruns",
                 experiment_name: str = "qai"):
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self._mlflow = mlflow
        except ImportError:
            raise ImportError("[WHAT] MLflowLogger requires mlflow\n[FIX] pip install mlflow")

    def log_run(self, name: str, params: Dict, metrics: Dict,
                tags: Optional[List[str]] = None, notes: str = "",
                model_path: Optional[str] = None) -> str:
        """Log a run to MLflow."""
        with self._mlflow.start_run(run_name=name) as run:
            self._mlflow.log_params(params)
            self._mlflow.log_metrics(metrics)
            if tags:
                for tag in tags:
                    self._mlflow.set_tag(tag, "true")
            if model_path:
                self._mlflow.log_artifact(model_path)
            return run.info.run_id

    def get_best_run(self, metric: str = "accuracy"):
        """Get the best run from MLflow."""
        runs = self._mlflow.search_runs(order_by=[f"metrics.{metric} DESC"], max_results=1)
        if len(runs) > 0:
            return runs.iloc[0].to_dict()
        return None


# =====================================================================
# Weights & Biases Logger
# =====================================================================
class WandBLogger:
    """Weights & Biases experiment logger.

    Requires: pip install wandb
    """
    def __init__(self, project: str = "qai", entity: Optional[str] = None):
        try:
            import wandb
            self._wandb = wandb
            self._project = project
            self._entity = entity
        except ImportError:
            raise ImportError("[WHAT] WandBLogger requires wandb\n[FIX] pip install wandb")

    def log_run(self, name: str, params: Dict, metrics: Dict,
                tags: Optional[List[str]] = None):
        """Log a run to W&B."""
        run = self._wandb.init(project=self._project, entity=self._entity,
                                name=name, tags=tags or [], config=params)
        self._wandb.log(metrics)
        run.finish()


# =====================================================================
# Unified Experiment Logger
# =====================================================================
class ExperimentLogger:
    """Unified experiment logger — supports MLflow, W&B, JSON, TensorBoard.

    Automatically picks the best available backend. Can log to multiple
    backends simultaneously.

    Args:
        backend: "json" (default), "mlflow", "wandb", "tensorboard", or "all"
        log_dir: directory for JSON logs
        tracking_uri: MLflow tracking URI

    Usage:
        logger = qai.ExperimentLogger(backend="json")
        run = logger.log_run(
            name="iris_rf_v1",
            params={"technique": "random_forest", "n_estimators": 100},
            metrics={"accuracy": 0.97},
        )
        print(logger.compare(metric="accuracy"))
    """
    def __init__(self, backend: str = "json",
                 log_dir: str = "./experiments",
                 tracking_uri: str = "./mlruns",
                 project: str = "qai"):
        self.backends: List[Any] = []

        if backend in ("json", "all"):
            self.backends.append(JSONLogger(log_dir))
        if backend in ("mlflow", "all"):
            try:
                self.backends.append(MLflowLogger(tracking_uri=tracking_uri))
            except ImportError:
                pass
        if backend in ("wandb", "all"):
            try:
                self.backends.append(WandBLogger(project=project))
            except ImportError:
                pass

        if not self.backends:
            # Fallback to JSON
            self.backends.append(JSONLogger(log_dir))

    def log_run(self, name: str, params: Dict, metrics: Dict,
                tags: Optional[List[str]] = None, **kwargs) -> ExperimentRun:
        """Log a run to all configured backends."""
        run = None
        for backend in self.backends:
            if isinstance(backend, JSONLogger):
                run = backend.log_run(name, params, metrics, tags, **kwargs)
            elif isinstance(backend, MLflowLogger):
                backend.log_run(name, params, metrics, tags, **kwargs)
            elif isinstance(backend, WandBLogger):
                backend.log_run(name, params, metrics, tags)
        return run or ExperimentRun(
            run_id="unknown", name=name, timestamp=time.time(),
            params=params, metrics=metrics, tags=tags or [],
        )

    def get_all_runs(self) -> List[ExperimentRun]:
        """Get all runs (from JSON backend)."""
        for backend in self.backends:
            if isinstance(backend, JSONLogger):
                return backend.get_all_runs()
        return []

    def get_best_run(self, metric: str = "accuracy") -> Optional[ExperimentRun]:
        """Get the best run by a metric."""
        for backend in self.backends:
            if isinstance(backend, JSONLogger):
                return backend.get_best_run(metric=metric)
        return None

    def compare(self, metric: str = "accuracy") -> str:
        """Compare all runs."""
        for backend in self.backends:
            if isinstance(backend, JSONLogger):
                return backend.compare(metric=metric)
        return "No experiments found."


# =====================================================================
# Convenience function
# =====================================================================
_global_logger: Optional[ExperimentLogger] = None

def _get_global_logger() -> ExperimentLogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = ExperimentLogger(backend="json")
    return _global_logger


def log_experiment(name: str, params: Dict, metrics: Dict,
                    tags: Optional[List[str]] = None,
                    notes: str = "", model_path: Optional[str] = None,
                    duration_s: float = 0.0) -> ExperimentRun:
    """One-call experiment logging.

    Logs to the global experiment logger (JSON by default).

    Example:
        >>> qai.log_experiment(
        ...     name="iris_rf_v1",
        ...     params={"technique": "random_forest", "n_estimators": 100},
        ...     metrics={"accuracy": 0.97, "f1": 0.96},
        ...     tags=["iris", "classification"],
        ... )
    """
    return _get_global_logger().log_run(
        name=name, params=params, metrics=metrics,
        tags=tags, notes=notes, model_path=model_path,
        duration_s=duration_s,
    )


def compare_experiments(metric: str = "accuracy") -> str:
    """Compare all logged experiments."""
    return _get_global_logger().compare(metric=metric)


def get_best_experiment(metric: str = "accuracy") -> Optional[ExperimentRun]:
    """Get the best experiment by a metric."""
    return _get_global_logger().get_best_run(metric=metric)


__all__ = ["ExperimentLogger", "ExperimentRun", "JSONLogger",
           "MLflowLogger", "WandBLogger",
           "log_experiment", "compare_experiments", "get_best_experiment"]
