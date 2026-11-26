"""
qai.observability -- production model monitoring / observability system.

Tracks drift, latency, accuracy, and alerts in real-time. Provides a
self-contained HTML dashboard with live charts (Canvas/SVG only -- no
external JS deps).

Classes:
  - MonitoringConfig       : configuration for the monitor
  - MetricSample           : a single metric observation
  - PredictionLog          : a single prediction event with full context
  - ModelMonitor           : the main monitor (ingests predictions/metrics)
  - Alert                  : an alert event
  - AlertRule              : a rule that triggers an alert
  - DriftDetector          : PSI + KS drift detection per feature
  - LatencyTracker         : inference latency tracking with percentiles
  - HealthChecker          : comprehensive health checks
  - MonitoringDashboard    : HTML / JSON / markdown dashboard generator

Top-level convenience functions:
  - monitor(model_name, config=None) -> ModelMonitor
  - quick_health(model, X=None, y=None) -> dict
  - dashboard(monitor) -> str   (HTML)

Storage: JSON-lines format at `storage_path`
         (default ~/.qai/monitoring/<model_name>.jsonl)

Usage:
    import qai.observability as obs
    mon = obs.monitor('my_model')
    pid = mon.log_prediction(input=x, output=y, confidence=0.9, latency_ms=12.3)
    print(mon.health_score())
    html = obs.dashboard(mon)
"""
from __future__ import annotations

import os
import json
import time
import uuid
import hashlib
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_float(v: Any) -> Optional[float]:
    """Convert v to float; return None if NaN/Inf/non-numeric."""
    if v is None:
        return None
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _now() -> float:
    return time.time()


def _hash_input(x: Any) -> str:
    """Stable SHA-256 hash of an input vector (or any picklable object)."""
    try:
        arr = np.asarray(x, dtype=float).flatten()
        return hashlib.sha256(arr.tobytes()).hexdigest()[:16]
    except Exception:
        try:
            return hashlib.sha256(str(x).encode("utf-8")).hexdigest()[:16]
        except Exception:
            return "0000000000000000"


def _summarize_input(x: Any) -> Dict[str, float]:
    """Statistical summary of an input vector."""
    try:
        arr = np.asarray(x, dtype=float).flatten()
        if arr.size == 0:
            return {"n_features": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "n_features": int(arr.shape[0]),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    except Exception:
        return {"n_features": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}


# ---------------------------------------------------------------------------
# 1. MonitoringConfig
# ---------------------------------------------------------------------------
@dataclass
class MonitoringConfig:
    """Configuration for the model monitor.

    Attributes:
        drift_threshold: PSI threshold above which drift is flagged.
        latency_p99_threshold_ms: p99 latency threshold (ms) for SLA.
        accuracy_drop_threshold: accuracy drop (absolute) considered severe.
        alert_cooldown_s: min seconds between repeated alerts of the same rule.
        sample_rate: fraction of predictions to log (0-1).
        retention_days: how long to keep data (informational; not auto-pruned).
        alert_channels: where alerts go. Built-in: 'console'. (Others noop.)
    """
    drift_threshold: float = 0.05
    latency_p99_threshold_ms: float = 100.0
    accuracy_drop_threshold: float = 0.05
    alert_cooldown_s: float = 300.0
    sample_rate: float = 1.0
    retention_days: int = 30
    alert_channels: List[str] = field(default_factory=lambda: ["console"])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "MonitoringConfig":
        d = d or {}
        known = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(**kwargs)

    def validate(self) -> List[str]:
        """Return a list of validation error messages (empty if valid)."""
        errs: List[str] = []
        if not (0.0 < self.drift_threshold < 1.0):
            errs.append("drift_threshold must be in (0, 1)")
        if self.latency_p99_threshold_ms <= 0:
            errs.append("latency_p99_threshold_ms must be > 0")
        if not (0.0 < self.accuracy_drop_threshold < 1.0):
            errs.append("accuracy_drop_threshold must be in (0, 1)")
        if self.alert_cooldown_s < 0:
            errs.append("alert_cooldown_s must be >= 0")
        if not (0.0 < self.sample_rate <= 1.0):
            errs.append("sample_rate must be in (0, 1]")
        if self.retention_days <= 0:
            errs.append("retention_days must be > 0")
        if not isinstance(self.alert_channels, list) or not self.alert_channels:
            errs.append("alert_channels must be a non-empty list")
        return errs


# ---------------------------------------------------------------------------
# 2. MetricSample
# ---------------------------------------------------------------------------
@dataclass
class MetricSample:
    """A single metric observation.

    Attributes:
        timestamp: unix epoch seconds.
        metric_name: dotted name, e.g. "latency.p99", "drift.max_psi".
        value: the numeric value.
        tags: optional dict of string tags.
        prediction_id: optional associated prediction id.
    """
    timestamp: float
    metric_name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    prediction_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "metric_name": self.metric_name,
            "value": self.value,
            "tags": dict(self.tags),
            "prediction_id": self.prediction_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MetricSample":
        return cls(
            timestamp=float(d.get("timestamp", _now())),
            metric_name=str(d.get("metric_name", "")),
            value=float(d.get("value", 0.0)),
            tags=dict(d.get("tags", {}) or {}),
            prediction_id=d.get("prediction_id"),
        )

    def __repr__(self) -> str:
        return (f"MetricSample(t={self.timestamp:.2f}, "
                f"{self.metric_name}={self.value:.4g}, "
                f"tags={self.tags})")


# ---------------------------------------------------------------------------
# 3. PredictionLog
# ---------------------------------------------------------------------------
@dataclass
class PredictionLog:
    """A single prediction event with full context.

    Attributes:
        prediction_id: unique id (returned by log_prediction).
        timestamp: unix epoch seconds.
        input_hash: stable hash of the raw input vector.
        input_summary: statistical summary of the input.
        output: the model's prediction (any JSON-serialisable value).
        confidence: model's confidence in [0, 1] (if applicable).
        latency_ms: prediction latency in milliseconds.
        model_version: optional version string.
        user_id: optional requesting user id.
        metadata: optional dict with extra context (e.g. {"actual": y}).
    """
    prediction_id: str
    timestamp: float
    input_hash: str
    input_summary: Dict[str, float]
    output: Any
    confidence: float
    latency_ms: float
    model_version: Optional[str] = None
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "input_hash": self.input_hash,
            "input_summary": dict(self.input_summary),
            "output": self.output,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
            "user_id": self.user_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PredictionLog":
        return cls(
            prediction_id=str(d.get("prediction_id", "")),
            timestamp=float(d.get("timestamp", _now())),
            input_hash=str(d.get("input_hash", "")),
            input_summary=dict(d.get("input_summary", {}) or {}),
            output=d.get("output"),
            confidence=float(d.get("confidence", 0.0)),
            latency_ms=float(d.get("latency_ms", 0.0)),
            model_version=d.get("model_version"),
            user_id=d.get("user_id"),
            metadata=dict(d.get("metadata", {}) or {}),
        )


# ---------------------------------------------------------------------------
# 4. Alert
# ---------------------------------------------------------------------------
@dataclass
class Alert:
    """An alert event.

    Attributes:
        id: unique alert id.
        severity: "info" | "warning" | "critical".
        title: short human-readable title.
        message: longer description.
        metric_name: the metric that triggered the alert.
        threshold: the threshold value.
        actual_value: the value that triggered the alert.
        timestamp: unix epoch seconds.
        acknowledged: whether a human has ack'd this alert.
    """
    id: str
    severity: str
    title: str
    message: str
    metric_name: str
    threshold: Optional[float]
    actual_value: float
    timestamp: float
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "actual_value": self.actual_value,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Alert":
        return cls(
            id=str(d.get("id", "")),
            severity=str(d.get("severity", "info")),
            title=str(d.get("title", "")),
            message=str(d.get("message", "")),
            metric_name=str(d.get("metric_name", "")),
            threshold=_safe_float(d.get("threshold")),
            actual_value=float(d.get("actual_value", 0.0)),
            timestamp=float(d.get("timestamp", _now())),
            acknowledged=bool(d.get("acknowledged", False)),
        )

    def acknowledge(self) -> None:
        """Mark this alert as acknowledged."""
        self.acknowledged = True

    def __repr__(self) -> str:
        flag = "[ACK]" if self.acknowledged else "[NEW]"
        thr = "" if self.threshold is None else f", threshold={self.threshold}"
        return (f"Alert{flag} {self.severity.upper()} {self.title!r} "
                f"({self.metric_name}={self.actual_value:.4g}{thr})")


# ---------------------------------------------------------------------------
# 5. AlertRule
# ---------------------------------------------------------------------------
@dataclass
class AlertRule:
    """A rule that triggers an alert when a metric crosses a threshold.

    Attributes:
        name: human-readable rule name.
        metric_name: dotted metric name to match against.
        condition: one of ">", "<", ">=", "<=", "==", "!=".
        threshold: threshold value.
        severity: "info" | "warning" | "critical".
        cooldown_s: min seconds between repeated fires.
        last_fired: timestamp of last fire (0 = never).
    """
    name: str
    metric_name: str
    condition: str
    threshold: float
    severity: str = "warning"
    cooldown_s: float = 300.0
    last_fired: float = 0.0

    _OPS = {
        ">":  lambda a, b: a > b,
        "<":  lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def evaluate(self, value: float) -> bool:
        """Return True if `value` crosses the rule's threshold."""
        op = self._OPS.get(self.condition)
        if op is None:
            return False
        try:
            return bool(op(float(value), float(self.threshold)))
        except (TypeError, ValueError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "condition": self.condition,
            "threshold": self.threshold,
            "severity": self.severity,
            "cooldown_s": self.cooldown_s,
            "last_fired": self.last_fired,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlertRule":
        return cls(
            name=str(d.get("name", "")),
            metric_name=str(d.get("metric_name", "")),
            condition=str(d.get("condition", ">")),
            threshold=float(d.get("threshold", 0.0)),
            severity=str(d.get("severity", "warning")),
            cooldown_s=float(d.get("cooldown_s", 300.0)),
            last_fired=float(d.get("last_fired", 0.0)),
        )


# ---------------------------------------------------------------------------
# 6. DriftDetector (specialised for observability)
# ---------------------------------------------------------------------------
class DriftDetector:
    """Specialised drift detection using PSI + KS statistic, per feature.

    Unlike qai.drift.DriftDetector (which returns a DriftReport dataclass),
    this one returns a flat dict[str, float] suitable for metric logging.

    Args:
        reference_data: optional (n_samples, n_features) reference distribution.
        n_bins: number of bins for PSI computation.
    """
    def __init__(self, reference_data: Optional[Any] = None, n_bins: int = 10):
        self.n_bins = int(n_bins)
        self.reference: Optional[np.ndarray] = None
        self._last_result: Dict[str, Any] = {}
        if reference_data is not None:
            self.set_reference(reference_data)

    def set_reference(self, X: Any) -> None:
        """Set the reference distribution."""
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        self.reference = arr
        self._last_result = {}

    def detect(self, X: Any) -> Dict[str, float]:
        """Detect drift in X relative to the reference distribution.

        Returns a flat dict with per-feature PSI/KS scores plus aggregates.
        Returns {"status": "no_reference_set"} if no reference has been set
        (graceful degradation).
        """
        if self.reference is None:
            return {"status": "no_reference_set"}
        try:
            new = np.asarray(X, dtype=float)
            if new.ndim == 1:
                new = new.reshape(-1, 1)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        if new.shape[1] != self.reference.shape[1]:
            return {
                "status": "feature_mismatch",
                "reference_n_features": int(self.reference.shape[1]),
                "new_n_features": int(new.shape[1]),
            }

        n_features = self.reference.shape[1]
        result: Dict[str, float] = {}
        psis: List[float] = []
        kss: List[float] = []
        for i in range(n_features):
            base_col = self.reference[:, i]
            new_col = new[:, i]
            try:
                psi = self._psi(base_col, new_col)
                ks = self._ks_stat(base_col, new_col)
            except Exception:
                psi = 0.0
                ks = 0.0
            result[f"feature_{i}_psi"] = float(psi)
            result[f"feature_{i}_ks"] = float(ks)
            psis.append(float(psi))
            kss.append(float(ks))

        result["mean_psi"] = float(np.mean(psis)) if psis else 0.0
        result["mean_ks"] = float(np.mean(kss)) if kss else 0.0
        result["max_psi"] = float(np.max(psis)) if psis else 0.0
        result["max_ks"] = float(np.max(kss)) if kss else 0.0
        result["n_features"] = float(n_features)
        result["n_samples"] = float(new.shape[0])
        self._last_result = dict(result)
        return result

    def is_drifting(self, X: Any, threshold: float = 0.05) -> bool:
        """Return True if max PSI or max KS exceeds `threshold`."""
        res = self.detect(X)
        if "status" in res:
            return False
        return bool(res.get("max_psi", 0.0) > threshold or
                    res.get("max_ks", 0.0) > threshold)

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the detector state."""
        if self.reference is None:
            return {
                "status": "no_reference_set",
                "n_features": 0,
                "n_samples": 0,
                "last_result": {},
            }
        return {
            "status": "ok",
            "n_features": int(self.reference.shape[1]),
            "n_samples": int(self.reference.shape[0]),
            "last_result": self._last_result,
        }

    # ----- internals -----
    def _psi(self, baseline: np.ndarray, new: np.ndarray) -> float:
        """Population Stability Index."""
        if len(baseline) == 0 or len(new) == 0:
            return 0.0
        edges = np.percentile(baseline, np.linspace(0, 100, self.n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
        new_counts = np.histogram(new, bins=edges)[0].astype(float)
        base_pct = (base_counts + 0.5) / (len(baseline) + 0.5 * self.n_bins)
        new_pct = (new_counts + 0.5) / (len(new) + 0.5 * self.n_bins)
        return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))

    def _ks_stat(self, baseline: np.ndarray, new: np.ndarray) -> float:
        """KS statistic (lazy-import scipy; fall back to manual)."""
        if len(baseline) == 0 or len(new) == 0:
            return 0.0
        try:
            from scipy.stats import ks_2samp  # lazy import
            return float(ks_2samp(baseline, new).statistic)
        except ImportError:
            return self._ks_manual(baseline, new)
        except Exception:
            return self._ks_manual(baseline, new)

    def _ks_manual(self, baseline: np.ndarray, new: np.ndarray) -> float:
        """Manual KS statistic (no scipy)."""
        all_vals = np.concatenate([baseline, new])
        all_vals.sort()
        cdf_base = np.searchsorted(np.sort(baseline), all_vals, side="right") / len(baseline)
        cdf_new = np.searchsorted(np.sort(new), all_vals, side="right") / len(new)
        return float(np.max(np.abs(cdf_base - cdf_new)))


# ---------------------------------------------------------------------------
# 7. LatencyTracker
# ---------------------------------------------------------------------------
class LatencyTracker:
    """Tracks inference latency over time.

    Stores (timestamp, latency_ms) tuples in a bounded deque. Thread-safe.
    """
    def __init__(self, window_size: int = 10000):
        self._samples: deque = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a latency observation."""
        try:
            lat = float(latency_ms)
        except (TypeError, ValueError):
            lat = 0.0
        with self._lock:
            self._samples.append((_now(), lat))

    def percentiles(self) -> Dict[str, float]:
        """Return p50, p90, p95, p99 (plus mean/min/max)."""
        with self._lock:
            if not self._samples:
                return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0,
                        "mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
            vals = np.array([s[1] for s in self._samples], dtype=float)
        return {
            "p50": float(np.percentile(vals, 50)),
            "p90": float(np.percentile(vals, 90)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "n": int(vals.shape[0]),
        }

    def throughput(self) -> float:
        """Return requests per second over the observed window."""
        with self._lock:
            n = len(self._samples)
            if n < 2:
                return 0.0
            t0 = self._samples[0][0]
            t1 = self._samples[-1][0]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return float((n - 1) / dt)

    def sla_compliance(self, p99_threshold_ms: float = 100.0) -> float:
        """Return the fraction (0..1) of requests under the threshold (ms)."""
        with self._lock:
            if not self._samples:
                return 1.0
            vals = np.array([s[1] for s in self._samples], dtype=float)
        return float(np.mean(vals <= float(p99_threshold_ms)))

    def history(self, n: int = 100) -> List[float]:
        """Return the last n latency observations."""
        with self._lock:
            recent = list(self._samples)[-n:]
        return [float(s[1]) for s in recent]


# ---------------------------------------------------------------------------
# 8. HealthCheck + HealthChecker
# ---------------------------------------------------------------------------
@dataclass
class HealthCheck:
    """Result of a single health check."""
    name: str
    status: str  # "pass" | "warn" | "fail"
    score: float  # 0..1
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "score": float(self.score),
            "message": self.message,
            "details": dict(self.details),
        }


class HealthChecker:
    """Comprehensive health checks for a ModelMonitor."""

    def check_drift(self, monitor: "ModelMonitor") -> HealthCheck:
        """Check drift status."""
        drift = monitor.compute_drift()
        if isinstance(drift, dict) and drift.get("status") == "no_reference_set":
            return HealthCheck(
                name="drift", status="pass", score=1.0,
                message="No reference set -- drift detection disabled",
                details=drift,
            )
        if isinstance(drift, dict) and drift.get("status") in ("no_data", "error",
                                                              "feature_mismatch"):
            return HealthCheck(
                name="drift", status="warn", score=0.5,
                message=f"Drift check returned: {drift.get('status')}",
                details=drift,
            )
        max_psi = float(drift.get("max_psi", 0.0))
        threshold = monitor.config.drift_threshold
        if max_psi > threshold * 4:
            status, score = "fail", 0.1
        elif max_psi > threshold:
            status, score = "warn", 0.5
        else:
            status, score = "pass", 1.0
        return HealthCheck(
            name="drift", status=status, score=score,
            message=f"max PSI = {max_psi:.4f} (threshold {threshold})",
            details=drift,
        )

    def check_latency(self, monitor: "ModelMonitor") -> HealthCheck:
        """Check latency SLA compliance."""
        lat = monitor.compute_latency_percentiles()
        p99 = float(lat.get("p99", 0.0))
        threshold = monitor.config.latency_p99_threshold_ms
        sla = monitor._latency_tracker.sla_compliance(threshold)
        if p99 == 0 and lat.get("n", 0) == 0:
            return HealthCheck(
                name="latency", status="pass", score=1.0,
                message="No latency data yet",
                details=lat,
            )
        if p99 > threshold * 2:
            status, score = "fail", 0.2
        elif p99 > threshold:
            status, score = "warn", 0.6
        else:
            status, score = "pass", 1.0
        return HealthCheck(
            name="latency", status=status, score=score,
            message=f"p99 = {p99:.2f}ms (threshold {threshold}ms), "
                    f"SLA compliance = {sla*100:.1f}%",
            details=lat,
        )

    def check_accuracy(self, monitor: "ModelMonitor") -> HealthCheck:
        """Check accuracy drop."""
        acc = monitor.compute_accuracy()
        acc_safe = _safe_float(acc)
        if acc_safe is None:
            return HealthCheck(
                name="accuracy", status="pass", score=1.0,
                message="No ground truth available -- accuracy check skipped",
                details={"accuracy": None},
            )
        drop_threshold = monitor.config.accuracy_drop_threshold
        # If accuracy < (1 - drop_threshold), warn. Below (1 - 2*drop) -> fail.
        if acc_safe < (1.0 - 2 * drop_threshold):
            status, score = "fail", 0.2
        elif acc_safe < (1.0 - drop_threshold):
            status, score = "warn", 0.6
        else:
            status, score = "pass", 1.0
        return HealthCheck(
            name="accuracy", status=status, score=score,
            message=f"accuracy = {acc_safe:.4f}",
            details={"accuracy": acc_safe},
        )

    def check_errors(self, monitor: "ModelMonitor") -> HealthCheck:
        """Check error / alert volume."""
        alerts = monitor.alerts()
        n_critical = sum(1 for a in alerts if a.severity == "critical")
        n_warning = sum(1 for a in alerts if a.severity == "warning")
        if n_critical > 0:
            status, score = "fail", 0.2
        elif n_warning > 5:
            status, score = "warn", 0.5
        elif n_warning > 0:
            status, score = "warn", 0.8
        else:
            status, score = "pass", 1.0
        return HealthCheck(
            name="errors", status=status, score=score,
            message=f"{n_critical} critical, {n_warning} warning alerts",
            details={"n_critical": n_critical, "n_warning": n_warning,
                     "n_total": len(alerts)},
        )

    def check_all(self, monitor: "ModelMonitor") -> Dict[str, Any]:
        """Run all checks and return a unified report."""
        checks = [
            self.check_drift(monitor),
            self.check_latency(monitor),
            self.check_accuracy(monitor),
            self.check_errors(monitor),
        ]
        n_fail = sum(1 for c in checks if c.status == "fail")
        n_warn = sum(1 for c in checks if c.status == "warn")
        if n_fail > 0:
            overall = "critical"
        elif n_warn > 0:
            overall = "warning"
        else:
            overall = "healthy"
        avg_score = float(np.mean([c.score for c in checks])) if checks else 0.0
        return {
            "overall_status": overall,
            "overall_score": round(avg_score, 4),
            "checks": [c.to_dict() for c in checks],
            "summary": {
                "total": len(checks),
                "pass": sum(1 for c in checks if c.status == "pass"),
                "warn": n_warn,
                "fail": n_fail,
            },
        }


# ---------------------------------------------------------------------------
# 9. ModelMonitor
# ---------------------------------------------------------------------------
class ModelMonitor:
    """Production model monitor.

    Ingests predictions and metrics, tracks drift / latency / accuracy,
    fires alerts based on configurable rules, and provides a one-shot
    dashboard snapshot.

    Args:
        model_name: human-readable model name.
        config: MonitoringConfig or dict (optional).
        storage_path: path to JSONL log file. If None,
            ~/.qai/monitoring/<model_name>.jsonl is used.
    """
    DEFAULT_WINDOW = 10000

    def __init__(self,
                 model_name: str,
                 config: Optional[Union[MonitoringConfig, Dict[str, Any]]] = None,
                 storage_path: Optional[str] = None):
        self.model_name = str(model_name)

        if isinstance(config, MonitoringConfig):
            self.config = config
        else:
            self.config = MonitoringConfig.from_dict(config or {})
        errs = self.config.validate()
        if errs:
            raise ValueError(
                f"[WHAT] Invalid MonitoringConfig\n"
                f"[WHY] {errs}\n"
                f"[FIX] Fix the listed config fields"
            )

        self._lock = threading.Lock()
        self._predictions: deque = deque(maxlen=self.DEFAULT_WINDOW)
        self._inputs: deque = deque(maxlen=self.DEFAULT_WINDOW)  # raw vectors for drift
        self._metrics: deque = deque(maxlen=self.DEFAULT_WINDOW)
        self._alerts: List[Alert] = []
        self._alert_rules: List[AlertRule] = []
        self._latency_tracker = LatencyTracker(window_size=self.DEFAULT_WINDOW)
        self._drift_detector = DriftDetector()
        self._start_time = _now()

        # Storage
        if storage_path is None:
            storage_path = os.path.expanduser(
                f"~/.qai/monitoring/{model_name}.jsonl"
            )
        self.storage_path = storage_path
        try:
            d = os.path.dirname(storage_path)
            if d:
                os.makedirs(d, exist_ok=True)
        except Exception:
            pass

        self._init_default_rules()

    def _init_default_rules(self) -> None:
        """Install the default alert rules."""
        self._alert_rules = [
            AlertRule(
                name="drift_high",
                metric_name="drift.max_psi",
                condition=">",
                threshold=self.config.drift_threshold,
                severity="warning",
                cooldown_s=self.config.alert_cooldown_s,
            ),
            AlertRule(
                name="latency_p99_high",
                metric_name="latency.p99",
                condition=">",
                threshold=self.config.latency_p99_threshold_ms,
                severity="warning",
                cooldown_s=self.config.alert_cooldown_s,
            ),
            AlertRule(
                name="accuracy_drop",
                metric_name="accuracy",
                condition="<",
                threshold=1.0 - self.config.accuracy_drop_threshold,
                severity="critical",
                cooldown_s=self.config.alert_cooldown_s,
            ),
        ]

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def log_prediction(self,
                       input: Any,
                       output: Any,
                       confidence: float,
                       latency_ms: float,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Log a single prediction event. Returns the prediction_id."""
        # Sampling
        if self.config.sample_rate < 1.0:
            import random
            if random.random() > self.config.sample_rate:
                return ""

        pid = uuid.uuid4().hex
        now = _now()
        input_hash = _hash_input(input)
        input_summary = _summarize_input(input)
        meta = dict(metadata or {})

        log = PredictionLog(
            prediction_id=pid,
            timestamp=now,
            input_hash=input_hash,
            input_summary=input_summary,
            output=output,
            confidence=float(confidence),
            latency_ms=float(latency_ms),
            model_version=str(meta.get("model_version")) if meta.get("model_version") else None,
            user_id=str(meta.get("user_id")) if meta.get("user_id") else None,
            metadata=meta,
        )

        # Keep raw input vector for drift detection (in-memory only)
        try:
            arr = np.asarray(input, dtype=float).flatten()
            raw_vec = arr if arr.size > 0 else None
        except Exception:
            raw_vec = None

        with self._lock:
            self._predictions.append(log)
            if raw_vec is not None:
                self._inputs.append(raw_vec)
        self._latency_tracker.record(latency_ms)
        self._append_storage({"kind": "prediction", **log.to_dict()})
        return pid

    def log_metric(self,
                   name: str,
                   value: float,
                   tags: Optional[Dict[str, str]] = None) -> None:
        """Log a custom metric observation."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = 0.0
        sample = MetricSample(
            timestamp=_now(),
            metric_name=str(name),
            value=v,
            tags=dict(tags or {}),
            prediction_id=None,
        )
        with self._lock:
            self._metrics.append(sample)
        self._append_storage({"kind": "metric", **sample.to_dict()})
        # Also fire any custom alert rules matching this metric name
        self._maybe_alert(str(name), v, _now())

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    def _append_storage(self, record: Dict[str, Any]) -> None:
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Alerting
    # ------------------------------------------------------------------
    def _maybe_alert(self, metric_name: str, value: float, now: float) -> None:
        """Evaluate all alert rules matching metric_name (with cooldown)."""
        new_alerts: List[Alert] = []
        with self._lock:
            for rule in self._alert_rules:
                if rule.metric_name != metric_name:
                    continue
                if not rule.evaluate(value):
                    continue
                if now - rule.last_fired < rule.cooldown_s:
                    continue
                rule.last_fired = now
                alert = Alert(
                    id=uuid.uuid4().hex[:12],
                    severity=rule.severity,
                    title=f"{rule.name} triggered",
                    message=(f"metric {metric_name} = {value:.4g} "
                             f"({rule.condition} {rule.threshold})"),
                    metric_name=metric_name,
                    threshold=rule.threshold,
                    actual_value=value,
                    timestamp=now,
                    acknowledged=False,
                )
                self._alerts.append(alert)
                new_alerts.append(alert)
        for a in new_alerts:
            self._emit_alert(a)

    def _emit_alert(self, alert: Alert) -> None:
        """Dispatch an alert to configured channels."""
        if "console" in self.config.alert_channels:
            try:
                print(f"[OBSERVABILITY][{alert.severity.upper()}] "
                      f"{alert.title}: {alert.message}")
            except Exception:
                pass

    def add_rule(self, rule: AlertRule) -> None:
        """Add a custom alert rule."""
        with self._lock:
            self._alert_rules.append(rule)

    def alerts(self) -> List[Alert]:
        """Return a snapshot of all alerts (newest last)."""
        with self._lock:
            return list(self._alerts)

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert by id. Returns True if found."""
        with self._lock:
            for a in self._alerts:
                if a.id == alert_id:
                    a.acknowledge()
                    return True
        return False

    # ------------------------------------------------------------------
    # Computation
    # ------------------------------------------------------------------
    def compute_drift(self, window_size: int = 100) -> Dict[str, float]:
        """Compute PSI / KS per feature against the reference set.

        Returns {"status": "no_data"} if no predictions logged,
        {"status": "no_reference_set"} if no reference data has been set,
        or a dict of per-feature scores + aggregates.
        """
        with self._lock:
            recent_inputs = list(self._inputs)[-window_size:]
            detector = self._drift_detector
        if not recent_inputs:
            return {"status": "no_data"}
        if detector is None or detector.reference is None:
            return {"status": "no_reference_set"}
        try:
            X = np.array(recent_inputs, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            result = detector.detect(X)
        except Exception as e:
            return {"status": "error", "message": str(e)}
        # Trigger drift alert if needed
        if isinstance(result, dict) and "max_psi" in result:
            self._maybe_alert("drift.max_psi", float(result["max_psi"]), _now())
        return result

    def compute_latency_percentiles(self) -> Dict[str, float]:
        """Return p50, p90, p95, p99 latency (plus mean/min/max)."""
        return self._latency_tracker.percentiles()

    def compute_accuracy(self,
                         window_size: int = 100,
                         ground_truth: Optional[List[Any]] = None) -> float:
        """Compute rolling accuracy. Returns NaN if no ground truth available.

        Ground truth can be supplied either:
          - At log time via metadata={"actual": y_true}, OR
          - Here via ground_truth=[y0, y1, ...] (aligned to log order).
        """
        with self._lock:
            recent = list(self._predictions)[-window_size:]
        if not recent:
            return float("nan")
        correct = 0
        total = 0
        if ground_truth is not None:
            gt = list(ground_truth)
            for i, log in enumerate(recent):
                if i >= len(gt):
                    break
                try:
                    if log.output == gt[i]:
                        correct += 1
                except Exception:
                    pass
                total += 1
        else:
            for log in recent:
                actual = log.metadata.get("actual")
                if actual is None:
                    continue
                try:
                    if log.output == actual:
                        correct += 1
                except Exception:
                    pass
                total += 1
        if total == 0:
            return float("nan")
        return float(correct) / float(total)

    def _evaluate_alerts(self) -> None:
        """Evaluate all built-in alert rules against current metrics."""
        now = _now()
        # Latency alert
        lat = self._latency_tracker.percentiles()
        self._maybe_alert("latency.p99", float(lat.get("p99", 0.0)), now)
        # Drift alert (compute_drift fires its own alert internally)
        self.compute_drift()
        # Accuracy alert
        acc = self.compute_accuracy()
        acc_safe = _safe_float(acc)
        if acc_safe is not None:
            self._maybe_alert("accuracy", acc_safe, now)

    def health_score(self) -> Dict[str, float]:
        """Return an overall 0-100 health score + per-component scores."""
        self._evaluate_alerts()

        # Latency score
        lat = self._latency_tracker.percentiles()
        p99 = float(lat.get("p99", 0.0))
        threshold = self.config.latency_p99_threshold_ms
        if p99 == 0.0 and lat.get("n", 0) == 0:
            latency_score = 100.0
        else:
            ratio = p99 / max(threshold, 1e-9)
            # 100 when ratio=1, 0 when ratio=2
            latency_score = max(0.0, min(100.0, 100.0 * (2.0 - ratio)))
        latency_score = round(latency_score, 2)

        # Drift score
        drift_res = self.compute_drift()
        if isinstance(drift_res, dict) and "max_psi" in drift_res:
            max_psi = float(drift_res["max_psi"])
            d_threshold = max(self.config.drift_threshold, 1e-9)
            # 100 when max_psi=0, 0 when max_psi=4*threshold
            drift_score = max(0.0, min(100.0,
                100.0 * (1.0 - max_psi / (4.0 * d_threshold))))
        else:
            drift_score = 100.0  # graceful -- no reference set
        drift_score = round(drift_score, 2)

        # Accuracy score
        acc = self.compute_accuracy()
        acc_safe = _safe_float(acc)
        if acc_safe is None:
            accuracy_score = 100.0  # graceful -- no ground truth
        else:
            accuracy_score = max(0.0, min(100.0, acc_safe * 100.0))
        accuracy_score = round(accuracy_score, 2)

        # Errors score (based on alert volume)
        with self._lock:
            n_alerts = len(self._alerts)
            n_critical = sum(1 for a in self._alerts if a.severity == "critical")
            n_warning = sum(1 for a in self._alerts if a.severity == "warning")
        errors_score = max(0.0, 100.0
                            - n_critical * 20.0
                            - n_warning * 5.0
                            - max(0, n_alerts - n_critical - n_warning) * 2.0)
        errors_score = round(errors_score, 2)

        overall = (latency_score * 0.3
                   + drift_score * 0.3
                   + accuracy_score * 0.3
                   + errors_score * 0.1)
        overall = round(overall, 2)
        return {
            "overall": overall,
            "latency": latency_score,
            "drift": drift_score,
            "accuracy": accuracy_score,
            "errors": errors_score,
        }

    def stats(self) -> Dict[str, int]:
        """Return integer stats: n_predictions, n_metrics, n_alerts, ..."""
        with self._lock:
            n_pred = len(self._predictions)
            n_metrics = len(self._metrics)
            n_alerts = len(self._alerts)
            n_ack = sum(1 for a in self._alerts if a.acknowledged)
        uptime = int(_now() - self._start_time)
        return {
            "n_predictions": int(n_pred),
            "n_metrics": int(n_metrics),
            "n_alerts": int(n_alerts),
            "n_alerts_ack": int(n_ack),
            "n_alerts_unack": int(n_alerts - n_ack),
            "uptime_s": int(uptime),
        }

    def dashboard(self) -> Dict[str, Any]:
        """One-shot status snapshot suitable for a UI."""
        with self._lock:
            recent_preds = [p.to_dict() for p in list(self._predictions)[-20:]]
        return {
            "model_name": self.model_name,
            "generated_at": _now(),
            "uptime_s": _now() - self._start_time,
            "stats": self.stats(),
            "latency": self.compute_latency_percentiles(),
            "latency_history": self._latency_tracker.history(100),
            "health_score": self.health_score(),
            "alerts": [a.to_dict() for a in self.alerts()],
            "drift": self.compute_drift(),
            "accuracy": _safe_float(self.compute_accuracy()),
            "recent_predictions": recent_preds,
            "config": self.config.to_dict(),
            "throughput_rps": self._latency_tracker.throughput(),
            "sla_compliance": self._latency_tracker.sla_compliance(
                self.config.latency_p99_threshold_ms),
        }

    def export_json(self, path: str) -> str:
        """Export the full dashboard snapshot to a JSON file. Returns path."""
        payload = self.dashboard()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def set_reference(self, X: Any) -> None:
        """Set the reference distribution for drift detection."""
        self._drift_detector.set_reference(X)

    def health_check(self) -> Dict[str, Any]:
        """Run a full HealthChecker report."""
        return HealthChecker().check_all(self)

    def __repr__(self) -> str:
        s = self.stats()
        return (f"<ModelMonitor name={self.model_name!r} "
                f"preds={s['n_predictions']} alerts={s['n_alerts']} "
                f"uptime={s['uptime_s']}s>")


# ---------------------------------------------------------------------------
# 10. MonitoringDashboard
# ---------------------------------------------------------------------------
_DASHBOARD_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QAI Observability -- __MODEL_NAME__</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 Helvetica, Arial, sans-serif;
    background: #0d1117; color: #c9d1d9; margin: 0; padding: 24px;
  }
  h1 { color: #58a6ff; margin: 0 0 4px 0; font-size: 22px; }
  .subtitle { color: #8b949e; font-size: 12px; margin-bottom: 20px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 20px;
  }
  .card {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 14px;
  }
  .card h3 {
    color: #8b949e; font-size: 11px; text-transform: uppercase;
    margin: 0 0 6px 0; letter-spacing: 0.5px; font-weight: 600;
  }
  .card .value { font-size: 26px; font-weight: 600; color: #f0f6fc; }
  .card .sub { color: #8b949e; font-size: 11px; margin-top: 4px; }
  .section {
    background: #161b22; border: 1px solid #30363d; border-radius: 8px;
    padding: 16px; margin-bottom: 20px;
  }
  .section h2 {
    color: #58a6ff; font-size: 14px; margin: 0 0 12px 0;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  canvas { display: block; max-width: 100%; }
  .alerts { list-style: none; padding: 0; margin: 0; max-height: 320px;
            overflow-y: auto; }
  .alert-item {
    padding: 10px 12px; border-left: 3px solid #30363d; margin-bottom: 6px;
    background: #0d1117; border-radius: 4px; font-size: 13px;
  }
  .alert-item.warning { border-color: #d29922; }
  .alert-item.critical { border-color: #f85149; }
  .alert-item.info { border-color: #58a6ff; }
  .alert-item .title { font-weight: 600; color: #f0f6fc; }
  .alert-item .msg { color: #c9d1d9; margin-top: 2px; }
  .alert-item .meta { color: #8b949e; font-size: 11px; margin-top: 4px; }
  .gauge-row { display: flex; align-items: center; gap: 24px; flex-wrap: wrap; }
  .gauge canvas { width: 140px; height: 140px; }
  .health-bars { flex: 1; min-width: 280px; }
  .health-bar { margin-bottom: 10px; }
  .health-bar .label {
    display: flex; justify-content: space-between; font-size: 12px;
    color: #8b949e; margin-bottom: 4px;
  }
  .health-bar .track {
    height: 6px; background: #21262d; border-radius: 3px; overflow: hidden;
  }
  .health-bar .fill {
    height: 100%; background: #3fb950; transition: width 0.3s;
  }
  .health-bar .fill.warn { background: #d29922; }
  .health-bar .fill.crit { background: #f85149; }
  .empty { color: #8b949e; font-size: 13px; padding: 8px 0; }
</style>
</head>
<body>
<h1>QAI Observability -- <span id="model-name"></span></h1>
<div class="subtitle" id="subtitle"></div>

<div class="grid" id="kpis"></div>

<div class="section">
  <h2>Health Score</h2>
  <div class="gauge-row">
    <div class="gauge"><canvas id="gauge" width="140" height="140"></canvas></div>
    <div class="health-bars" id="health-bars"></div>
  </div>
</div>

<div class="section">
  <h2>Latency Histogram (recent predictions)</h2>
  <canvas id="latency-hist" width="900" height="200"></canvas>
</div>

<div class="section">
  <h2>Drift Heatmap (per-feature PSI / KS)</h2>
  <canvas id="drift-heatmap" width="900" height="220"></canvas>
</div>

<div class="section">
  <h2>Alert Feed</h2>
  <ul class="alerts" id="alerts"></ul>
</div>

<script>
const DATA = __DATA__;

function fmt(v, d) {
  d = (d == null) ? 2 : d;
  if (v == null || (typeof v === 'number' && isNaN(v))) return 'n/a';
  if (typeof v === 'number') return v.toFixed(d);
  return String(v);
}

// Model name + subtitle
document.getElementById('model-name').textContent = DATA.model_name || '';
const gen = new Date((DATA.generated_at || 0) * 1000).toLocaleString();
document.getElementById('subtitle').textContent =
  'Generated ' + gen + '  -  uptime ' + Math.round(DATA.uptime_s || 0) + 's';

// KPIs
const kpis = document.getElementById('kpis');
const stats = DATA.stats || {};
const lat = DATA.latency || {};
const drift = DATA.drift || {};
const cards = [
  {label: 'Predictions', value: stats.n_predictions || 0,
   sub: 'uptime ' + (stats.uptime_s || 0) + 's'},
  {label: 'p99 Latency (ms)', value: fmt(lat.p99, 2),
   sub: 'p50=' + fmt(lat.p50, 2) + '  p95=' + fmt(lat.p95, 2)},
  {label: 'Throughput (rps)', value: fmt(DATA.throughput_rps, 2),
   sub: 'SLA compliance ' + fmt((DATA.sla_compliance || 0) * 100, 1) + '%'},
  {label: 'Alerts', value: stats.n_alerts || 0,
   sub: (stats.n_alerts_unack || 0) + ' unacknowledged'},
  {label: 'Metrics Logged', value: stats.n_metrics || 0, sub: ''},
  {label: 'Accuracy', value: fmt(DATA.accuracy, 4), sub: ''},
  {label: 'Max PSI', value: fmt(drift.max_psi, 4),
   sub: 'max KS=' + fmt(drift.max_ks, 4)},
  {label: 'Health Score', value: fmt((DATA.health_score||{}).overall, 1),
   sub: '/ 100'},
];
cards.forEach(function(c) {
  const el = document.createElement('div');
  el.className = 'card';
  el.innerHTML = '<h3>' + c.label + '</h3>' +
                 '<div class="value">' + c.value + '</div>' +
                 '<div class="sub">' + (c.sub || '') + '</div>';
  kpis.appendChild(el);
});

// Health gauge
function drawGauge(canvas, value) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  const cx = w/2, cy = h/2, r = Math.min(w, h)/2 - 10;
  ctx.clearRect(0, 0, w, h);
  // background arc (270 degrees)
  ctx.beginPath();
  ctx.arc(cx, cy, r, Math.PI*0.75, Math.PI*0.25, false);
  ctx.lineWidth = 10; ctx.strokeStyle = '#21262d'; ctx.lineCap = 'round';
  ctx.stroke();
  // value arc
  const start = Math.PI*0.75;
  const end = start + Math.PI*1.5 * Math.max(0, Math.min(100, value)) / 100;
  ctx.beginPath();
  ctx.arc(cx, cy, r, start, end, false);
  let color = '#3fb950';
  if (value < 50) color = '#f85149';
  else if (value < 75) color = '#d29922';
  ctx.strokeStyle = color; ctx.stroke();
  // text
  ctx.fillStyle = '#f0f6fc';
  ctx.font = 'bold 32px sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(Math.round(value), cx, cy);
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px sans-serif';
  ctx.fillText('/ 100', cx, cy + 22);
}
drawGauge(document.getElementById('gauge'),
          ((DATA.health_score || {}).overall) || 0);

// Health bars
const bars = document.getElementById('health-bars');
const hs = DATA.health_score || {};
['latency', 'drift', 'accuracy', 'errors'].forEach(function(k) {
  const v = hs[k] || 0;
  const cls = v < 50 ? 'crit' : (v < 75 ? 'warn' : '');
  const el = document.createElement('div');
  el.className = 'health-bar';
  el.innerHTML =
    '<div class="label"><span>' + k.toUpperCase() +
    '</span><span>' + Math.round(v) + '</span></div>' +
    '<div class="track"><div class="fill ' + cls +
    '" style="width:' + Math.max(0, Math.min(100, v)) + '%"></div></div>';
  bars.appendChild(el);
});

// Latency histogram (recent_predictions latencies, fall back to latency_history)
function drawHistogram(canvas, values, bins) {
  bins = bins || 20;
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  // axis line
  ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, h - 20); ctx.lineTo(w, h - 20); ctx.stroke();
  if (!values || values.length === 0) {
    ctx.fillStyle = '#8b949e'; ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No latency data', w/2, h/2);
    return;
  }
  let min = Math.min.apply(null, values);
  let max = Math.max.apply(null, values);
  if (max - min < 1e-9) max = min + 1;
  const range = max - min;
  const counts = new Array(bins).fill(0);
  for (let i = 0; i < values.length; i++) {
    let idx = Math.floor((values[i] - min) / range * bins);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  }
  const maxCount = Math.max.apply(null, counts) || 1;
  const barW = w / bins;
  for (let i = 0; i < bins; i++) {
    const barH = (counts[i] / maxCount) * (h - 36);
    ctx.fillStyle = '#58a6ff';
    ctx.fillRect(i * barW + 1, h - barH - 20, Math.max(1, barW - 2), barH);
  }
  ctx.fillStyle = '#8b949e'; ctx.font = '11px sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(min.toFixed(1) + 'ms', 2, h - 4);
  ctx.textAlign = 'right';
  ctx.fillText(max.toFixed(1) + 'ms', w - 2, h - 4);
  ctx.textAlign = 'center';
  ctx.fillText('p99=' + (DATA.latency || {}).p99.toFixed(2) + 'ms',
               w/2, h - 4);
}
let latVals = (DATA.recent_predictions || [])
                .map(function(p) { return p.latency_ms; })
                .filter(function(v) { return v != null && !isNaN(v); });
if (latVals.length === 0 && DATA.latency_history) {
  latVals = DATA.latency_history;
}
drawHistogram(document.getElementById('latency-hist'), latVals);

// Drift heatmap
function drawHeatmap(canvas, driftData) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (!driftData || driftData.status === 'no_reference_set' ||
      driftData.status === 'no_data' || driftData.status === 'error' ||
      driftData.status === 'feature_mismatch') {
    ctx.fillStyle = '#8b949e'; ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(driftData ? ('Drift: ' + driftData.status)
                           : 'No drift data', w/2, h/2);
    return;
  }
  const features = [];
  const keys = Object.keys(driftData);
  for (let i = 0; i < keys.length; i++) {
    const m = keys[i].match(/^feature_(\\d+)_psi$/);
    if (m) {
      const idx = parseInt(m[1]);
      features.push({
        idx: idx,
        psi: driftData['feature_' + idx + '_psi'] || 0,
        ks:  driftData['feature_' + idx + '_ks']  || 0,
      });
    }
  }
  if (features.length === 0) {
    ctx.fillStyle = '#8b949e'; ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No per-feature drift data', w/2, h/2);
    return;
  }
  const threshold = (DATA.config || {}).drift_threshold || 0.05;
  const cellW = w / features.length;
  const labelsH = 24;
  const bottomPad = 24;
  const rowH = (h - labelsH - bottomPad) / 2;
  function colorFor(v) {
    if (v > threshold * 4) return '#f85149';
    if (v > threshold)     return '#d29922';
    return '#3fb950';
  }
  for (let i = 0; i < features.length; i++) {
    // PSI row
    ctx.fillStyle = colorFor(features[i].psi);
    ctx.fillRect(i * cellW + 1, labelsH, cellW - 2, rowH - 2);
    // KS row
    ctx.fillStyle = colorFor(features[i].ks);
    ctx.fillRect(i * cellW + 1, labelsH + rowH, cellW - 2, rowH - 2);
    // value labels
    ctx.fillStyle = '#c9d1d9';
    ctx.font = '10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(features[i].psi.toFixed(3),
                 i * cellW + cellW/2, labelsH + rowH - 4);
    ctx.fillText(features[i].ks.toFixed(3),
                 i * cellW + cellW/2, labelsH + 2 * rowH - 4);
    // x-axis label
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px sans-serif';
    ctx.fillText('f' + features[i].idx,
                 i * cellW + cellW/2, h - 6);
  }
  // row labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '11px sans-serif';
  ctx.textAlign = 'right';
  ctx.fillText('PSI', 28, labelsH + rowH/2 + 4);
  ctx.fillText('KS',  28, labelsH + rowH + rowH/2 + 4);
}
drawHeatmap(document.getElementById('drift-heatmap'), DATA.drift || {});

// Alerts
const alertsEl = document.getElementById('alerts');
const alerts = DATA.alerts || [];
if (alerts.length === 0) {
  const li = document.createElement('li');
  li.className = 'alert-item info';
  li.innerHTML = '<div class="title">No alerts -- all healthy</div>';
  alertsEl.appendChild(li);
} else {
  for (let i = alerts.length - 1; i >= 0; i--) {
    const a = alerts[i];
    const li = document.createElement('li');
    li.className = 'alert-item ' + (a.severity || 'info');
    const ts = new Date((a.timestamp || 0) * 1000).toLocaleString();
    li.innerHTML =
      '<div class="title">[' + (a.severity || '').toUpperCase() + '] ' +
      (a.title || '') + '</div>' +
      '<div class="msg">' + (a.message || '') + '</div>' +
      '<div class="meta">' + (a.metric_name || '') + ' = ' +
      fmt(a.actual_value, 4) + ' (threshold ' + fmt(a.threshold, 4) +
      ') &middot; ' + ts + ' &middot; ' +
      (a.acknowledged ? 'ACK' : 'UNACK') + '</div>';
    alertsEl.appendChild(li);
  }
}
</script>
</body>
</html>
"""


class MonitoringDashboard:
    """Generates a self-contained HTML dashboard (plus JSON / markdown).

    Args:
        monitor: a ModelMonitor instance.
    """
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor

    def to_html(self) -> str:
        """Return a self-contained HTML page with inline Canvas charts."""
        data = self.monitor.dashboard()
        # JSON-encode safely; replace the placeholders
        json_str = json.dumps(data, default=str)
        html = _DASHBOARD_HTML_TEMPLATE.replace("__DATA__", json_str)
        html = html.replace("__MODEL_NAME__",
                            str(self.monitor.model_name or "model"))
        return html

    def to_json(self) -> str:
        """Return a JSON snapshot for external dashboard tools."""
        return json.dumps(self.monitor.dashboard(), default=str, indent=2)

    def to_markdown(self) -> str:
        """Return a markdown summary of the monitor's state."""
        d = self.monitor.dashboard()
        s = d.get("stats", {})
        l = d.get("latency", {})
        h = d.get("health_score", {})
        a = d.get("alerts", [])
        dr = d.get("drift", {})
        lines = [
            f"# Observability Dashboard -- {d.get('model_name', 'model')}",
            "",
            f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S',
                                            time.localtime(d.get('generated_at', 0)))}  ",
            f"**Uptime:** {int(d.get('uptime_s', 0))}s  ",
            f"**Throughput:** {d.get('throughput_rps', 0):.2f} rps  ",
            f"**SLA compliance:** {d.get('sla_compliance', 0)*100:.1f}%",
            "",
            "## KPIs",
            "",
            f"- Predictions: **{s.get('n_predictions', 0)}**",
            f"- Metrics logged: **{s.get('n_metrics', 0)}**",
            f"- Alerts: **{s.get('n_alerts', 0)}** "
            f"({s.get('n_alerts_unack', 0)} unacknowledged)",
            f"- Accuracy: `{d.get('accuracy')}`",
            "",
            "## Latency",
            "",
            f"| p50 | p90 | p95 | p99 | mean | max |",
            f"|-----|-----|-----|-----|------|-----|",
            f"| {l.get('p50', 0):.2f} | {l.get('p90', 0):.2f} | "
            f"{l.get('p95', 0):.2f} | {l.get('p99', 0):.2f} | "
            f"{l.get('mean', 0):.2f} | {l.get('max', 0):.2f} |",
            "",
            "## Health Score",
            "",
            f"- **Overall:** {h.get('overall', 0):.1f} / 100",
            f"- Latency: {h.get('latency', 0):.1f}",
            f"- Drift: {h.get('drift', 0):.1f}",
            f"- Accuracy: {h.get('accuracy', 0):.1f}",
            f"- Errors: {h.get('errors', 0):.1f}",
            "",
            "## Drift",
            "",
        ]
        if isinstance(dr, dict) and "max_psi" in dr:
            lines.append(f"- Max PSI: **{dr.get('max_psi', 0):.4f}**")
            lines.append(f"- Max KS:  **{dr.get('max_ks', 0):.4f}**")
            lines.append(f"- Features: {int(dr.get('n_features', 0))}")
        else:
            lines.append(f"- Status: `{dr.get('status', 'unknown') if isinstance(dr, dict) else 'unknown'}`")
        lines.append("")
        lines.append("## Alerts")
        lines.append("")
        if not a:
            lines.append("_No alerts -- all healthy._")
        else:
            lines.append("| Severity | Title | Metric | Actual | Threshold | Time |")
            lines.append("|----------|-------|--------|--------|-----------|------|")
            for al in a:
                sev = al.get("severity", "info")
                title = al.get("title", "")
                metric = al.get("metric_name", "")
                actual = al.get("actual_value", 0)
                thr = al.get("threshold", "")
                ts = time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(al.get("timestamp", 0)))
                lines.append(f"| {sev} | {title} | {metric} | {actual} | {thr} | {ts} |")
        lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 11. Top-level convenience functions
# ---------------------------------------------------------------------------
def monitor(model_name: str,
            config: Optional[Union[MonitoringConfig, Dict[str, Any]]] = None
            ) -> ModelMonitor:
    """Create a ModelMonitor. Convenience constructor."""
    return ModelMonitor(model_name=model_name, config=config)


def quick_health(model: Any,
                 X: Optional[Any] = None,
                 y: Optional[Any] = None) -> dict:
    """Quick health check on a trained model.

    Runs a few predictions through `model.predict`, logs them to a temporary
    monitor, and returns the health snapshot. Does not persist to disk.

    Args:
        model: any object with a `predict(x)` method.
        X: optional sample inputs.
        y: optional ground-truth labels aligned to X.

    Returns:
        A dict with health_score, stats, and alerts count.
    """
    # Use a throwaway monitor with no on-disk storage
    mon = ModelMonitor(model_name=f"quick_health_{id(model)}",
                       storage_path=os.devnull)
    if X is not None and hasattr(model, "predict"):
        try:
            X_arr = np.asarray(X, dtype=float)
        except Exception:
            X_arr = X
        try:
            preds = model.predict(X_arr)
        except Exception as e:
            return {"error": f"predict() failed: {type(e).__name__}: {e}"}
        # Normalize preds to a list
        if hasattr(preds, "tolist"):
            try:
                preds_list = preds.tolist()
            except Exception:
                preds_list = list(preds)
        elif hasattr(preds, "__iter__") and not isinstance(preds, (str, bytes)):
            preds_list = list(preds)
        else:
            preds_list = [preds] * len(X_arr) if hasattr(X_arr, "__len__") else [preds]
        n = len(X_arr) if hasattr(X_arr, "__len__") else 1
        for i in range(n):
            try:
                x = X_arr[i]
            except Exception:
                x = X_arr
            try:
                pred = preds_list[i] if i < len(preds_list) else preds_list[-1]
            except Exception:
                pred = None
            actual = None
            if y is not None:
                try:
                    y_arr = list(y) if hasattr(y, "__iter__") else [y]
                    actual = y_arr[i] if i < len(y_arr) else None
                except Exception:
                    actual = None
            t0 = _now()
            try:
                _ = model.predict(x)
                lat = (_now() - t0) * 1000.0
            except Exception:
                lat = 0.0
            mon.log_prediction(
                input=np.asarray(x).flatten().tolist() if hasattr(x, "flatten") else x,
                output=pred,
                confidence=1.0,
                latency_ms=lat,
                metadata={"actual": actual} if actual is not None else None,
            )
    hs = mon.health_score()
    return {
        "health_score": hs,
        "stats": mon.stats(),
        "n_alerts": len(mon.alerts()),
        "latency": mon.compute_latency_percentiles(),
        "accuracy": _safe_float(mon.compute_accuracy()),
    }


def dashboard(monitor: ModelMonitor) -> str:
    """Generate a self-contained HTML dashboard for a ModelMonitor."""
    return MonitoringDashboard(monitor).to_html()


__all__ = [
    "MonitoringConfig",
    "MetricSample",
    "PredictionLog",
    "ModelMonitor",
    "Alert",
    "AlertRule",
    "DriftDetector",
    "LatencyTracker",
    "HealthCheck",
    "HealthChecker",
    "MonitoringDashboard",
    "monitor",
    "quick_health",
    "dashboard",
]
