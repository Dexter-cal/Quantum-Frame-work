"""
qai.modellog — per-model logging system for production.

Every model gets a log file that tracks:
  - Training events (fit/train calls, loss, epochs)
  - Inference events (forward calls, latency)
  - Errors (exceptions, NaN, OOM)
  - Weight changes (edits, merges, prunes)
  - Deployment events (save, load, convert, deploy)
  - Performance metrics (accuracy, latency over time)
  - Drift detection (input distribution changes)

Logs are stored as structured JSON lines (one event per line) for easy
parsing. Each model's log lives alongside its .qmodel file.

Usage:
    logger = qai.ModelLog("my_model")
    logger.log_train(epoch=0, loss=1.5)
    logger.log_inference(latency_ms=12.3)
    logger.log_error("OOM during forward pass")
    logger.log_weight_edit("scale", layer="W", factor=0.9)
    logger.log_deploy("production", version="1.2.0")

    # Query the log
    events = logger.query(event_type="train")
    summary = logger.summary()
"""
from __future__ import annotations
import json
import os
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# ModelLog — per-model structured logging
# ============================================================================
class ModelLog:
    """Structured logging for a single model.

    Events are stored as JSON lines in a .log file.
    Each event has: timestamp, event_type, model_id, + event-specific fields.
    """

    def __init__(self, model_id: str, log_dir: str = "./model_logs"):
        self.model_id = model_id
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, f"{model_id}.log")
        self._buffer: deque = deque(maxlen=10000)  # in-memory buffer
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing log entries into buffer."""
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        self._buffer.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass

    def _write(self, event: Dict[str, Any]) -> None:
        """Write an event to the log file + buffer."""
        event["timestamp"] = time.time()
        event["timestamp_iso"] = datetime.now().isoformat()
        event["model_id"] = self.model_id
        line = json.dumps(event, default=str)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")
        self._buffer.append(event)

    # === Training events ===
    def log_train(self, epoch: int, loss: float,
                   extra: Optional[Dict] = None) -> None:
        """Log a training event."""
        event = {"event_type": "train", "epoch": epoch, "loss": float(loss)}
        if extra:
            event.update(extra)
        self._write(event)

    def log_fit_start(self, n_samples: int, technique: str) -> None:
        """Log the start of fit()."""
        self._write({"event_type": "fit_start", "n_samples": n_samples,
                       "technique": technique})

    def log_fit_end(self, duration_s: float, final_loss: Optional[float] = None) -> None:
        """Log the end of fit()."""
        self._write({"event_type": "fit_end", "duration_s": duration_s,
                       "final_loss": final_loss})

    # === Inference events ===
    def log_inference(self, latency_ms: float,
                        input_shape: Optional[Tuple] = None,
                        output_shape: Optional[Tuple] = None) -> None:
        """Log an inference event."""
        self._write({"event_type": "inference", "latency_ms": float(latency_ms),
                       "input_shape": list(input_shape) if input_shape else None,
                       "output_shape": list(output_shape) if output_shape else None})

    # === Error events ===
    def log_error(self, message: str, error_type: str = "RuntimeError",
                    traceback: Optional[str] = None) -> None:
        """Log an error."""
        self._write({"event_type": "error", "message": message,
                       "error_type": error_type, "traceback": traceback})

    def log_warning(self, message: str) -> None:
        """Log a warning."""
        self._write({"event_type": "warning", "message": message})

    # === Weight events ===
    def log_weight_edit(self, operation: str, layer: str,
                          **details) -> None:
        """Log a weight edit."""
        self._write({"event_type": "weight_edit", "operation": operation,
                       "layer": layer, **details})

    def log_weight_merge(self, strategy: str, parent_a: str,
                           parent_b: str, output: str) -> None:
        """Log a weight merge."""
        self._write({"event_type": "weight_merge", "strategy": strategy,
                       "parent_a": parent_a, "parent_b": parent_b,
                       "output": output})

    # === Deployment events ===
    def log_deploy(self, environment: str, version: str = "1.0.0",
                     endpoint: Optional[str] = None) -> None:
        """Log a deployment."""
        self._write({"event_type": "deploy", "environment": environment,
                       "version": version, "endpoint": endpoint})

    def log_save(self, path: str, format: str = ".qmodel") -> None:
        """Log a save event."""
        self._write({"event_type": "save", "path": path, "format": format})

    def log_load(self, path: str, format: str = ".qmodel") -> None:
        """Log a load event."""
        self._write({"event_type": "load", "path": path, "format": format})

    def log_convert(self, from_format: str, to_format: str,
                      path: str) -> None:
        """Log a format conversion."""
        self._write({"event_type": "convert", "from": from_format,
                       "to": to_format, "path": path})

    # === Performance events ===
    def log_metric(self, metric_name: str, value: float,
                     step: Optional[int] = None) -> None:
        """Log a performance metric."""
        self._write({"event_type": "metric", "metric": metric_name,
                       "value": float(value), "step": step})

    def log_drift(self, drift_score: float, feature: Optional[str] = None) -> None:
        """Log a data drift detection."""
        self._write({"event_type": "drift", "drift_score": float(drift_score),
                       "feature": feature})

    # === Querying ===
    def query(self, event_type: Optional[str] = None,
               since: Optional[float] = None,
               limit: int = 1000) -> List[Dict[str, Any]]:
        """Query log events."""
        results = []
        for event in reversed(self._buffer):
            if event_type and event.get("event_type") != event_type:
                continue
            if since and event.get("timestamp", 0) < since:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return list(reversed(results))

    def summary(self) -> Dict[str, Any]:
        """Summary of all events."""
        events = list(self._buffer)
        by_type: Dict[str, int] = defaultdict(int)
        for e in events:
            by_type[e.get("event_type", "unknown")] += 1
        # compute stats
        train_events = [e for e in events if e.get("event_type") == "train"]
        losses = [e["loss"] for e in train_events if "loss" in e]
        inference_events = [e for e in events if e.get("event_type") == "inference"]
        latencies = [e["latency_ms"] for e in inference_events if "latency_ms" in e]
        error_events = [e for e in events if e.get("event_type") == "error"]
        return {
            "model_id": self.model_id,
            "log_path": self.log_path,
            "n_events": len(events),
            "by_type": dict(by_type),
            "n_errors": len(error_events),
            "n_train_events": len(train_events),
            "n_inference_events": len(inference_events),
            "first_event": events[0]["timestamp_iso"] if events else None,
            "last_event": events[-1]["timestamp_iso"] if events else None,
            "train_loss_trend": {
                "first": losses[0] if losses else None,
                "last": losses[-1] if losses else None,
                "min": min(losses) if losses else None,
                "max": max(losses) if losses else None,
            } if losses else None,
            "inference_latency_stats": {
                "mean": float(np.mean(latencies)) if latencies else None,
                "p50": float(np.percentile(latencies, 50)) if latencies else None,
                "p95": float(np.percentile(latencies, 95)) if latencies else None,
                "p99": float(np.percentile(latencies, 99)) if latencies else None,
                "n": len(latencies),
            } if latencies else None,
            "recent_errors": [e["message"] for e in error_events[-5:]],
        }

    def export_json(self, path: str) -> str:
        """Export all events as a JSON array."""
        events = list(self._buffer)
        with open(path, "w") as f:
            json.dump(events, f, indent=2, default=str)
        return path

    def clear(self) -> None:
        """Clear all events."""
        self._buffer.clear()
        if os.path.exists(self.log_path):
            os.unlink(self.log_path)


# ============================================================================
# LoggedModel — wraps any model with automatic logging
# ============================================================================
class LoggedModel:
    """Wraps any qai model with automatic logging.

    Every fit(), forward(), and error is automatically logged.

    Usage:
        model = qai.build(technique="regression")
        logged = qai.LoggedModel(model, model_id="reg_v1")
        logged.fit(X, y)    # → auto-logs fit_start + fit_end
        logged.forward(X[0]) # → auto-logs inference latency
    """

    def __init__(self, model: Any, model_id: str = "model",
                 log_dir: str = "./model_logs"):
        self.model = model
        self.logger = ModelLog(model_id, log_dir)
        self._wrap_methods()

    def _wrap_methods(self) -> None:
        """Wrap fit() and forward() with logging."""
        original_fit = self.model.fit
        original_forward = self.model.forward
        logger = self.logger

        def logged_fit(X, y=None):
            n = len(X) if hasattr(X, "__len__") else 0
            technique = getattr(self.model, "name", "unknown")
            logger.log_fit_start(n, technique)
            t0 = time.time()
            try:
                result = original_fit(X, y)
                duration = time.time() - t0
                logger.log_fit_end(duration)
                return result
            except Exception as e:
                logger.log_error(f"fit() failed: {e}", type(e).__name__)
                raise

        def logged_forward(x):
            t0 = time.time()
            try:
                result = original_forward(x)
                latency = (time.time() - t0) * 1000
                input_shape = np.asarray(x).shape if x is not None else None
                output_shape = np.asarray(result).shape if result is not None else None
                logger.log_inference(latency, input_shape, output_shape)
                return result
            except Exception as e:
                logger.log_error(f"forward() failed: {e}", type(e).__name__)
                raise

        self.model.fit = logged_fit
        self.model.forward = logged_forward

    def __getattr__(self, name):
        """Delegate to the underlying model."""
        return getattr(self.model, name)

    def is_trained(self) -> bool:
        return self.model.is_trained()


# ============================================================================
# ModelLogRegistry — manage logs for multiple models
# ============================================================================
class ModelLogRegistry:
    """Registry of model logs across multiple models."""

    def __init__(self, log_dir: str = "./model_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def list_models(self) -> List[str]:
        """List all models that have logs."""
        return [f.replace(".log", "") for f in os.listdir(self.log_dir)
                 if f.endswith(".log")]

    def get_log(self, model_id: str) -> ModelLog:
        """Get the log for a specific model."""
        return ModelLog(model_id, self.log_dir)

    def all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get summaries for all models."""
        return {mid: self.get_log(mid).summary()
                 for mid in self.list_models()}

    def find_errors(self, since: Optional[float] = None) -> Dict[str, List[Dict]]:
        """Find all errors across all models."""
        out: Dict[str, List[Dict]] = {}
        for mid in self.list_models():
            errors = self.get_log(mid).query(event_type="error", since=since)
            if errors:
                out[mid] = errors
        return out

    def stats(self) -> Dict[str, Any]:
        """Overall stats across all model logs."""
        models = self.list_models()
        total_events = 0
        total_errors = 0
        total_size = 0
        for mid in models:
            log = self.get_log(mid)
            summary = log.summary()
            total_events += summary["n_events"]
            total_errors += summary["n_errors"]
            log_path = os.path.join(self.log_dir, f"{mid}.log")
            if os.path.exists(log_path):
                total_size += os.path.getsize(log_path)
        return {
            "n_models": len(models),
            "total_events": total_events,
            "total_errors": total_errors,
            "total_log_size_bytes": total_size,
            "log_dir": self.log_dir,
        }


__all__ = ["ModelLog", "LoggedModel", "ModelLogRegistry"]
