"""
qai.model_monitor — production monitoring that tracks drift, latency, and
accuracy over time.

Combines qai.DriftDetector, latency tracking, and accuracy tracking into
a single production monitoring dashboard.

Usage:
    import qai
    monitor = qai.ModelMonitor(baseline_X=X_train, baseline_y=y_train)

    # In production:
    for x, actual in zip(X_prod, y_prod):
        pred = model.predict(x)
        monitor.record(x, pred, actual, latency_ms=0.5)

    # Check health:
    report = monitor.report()
    print(report)
    if report.alerts:
        print(f"ALERTS: {report.alerts}")
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
import numpy as np

try:
    from .drift import DriftDetector, DriftReport
except ImportError:
    DriftDetector = None


@dataclass
class MonitorReport:
    """A production monitoring report."""
    n_predictions: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    current_accuracy: float = 0.0
    accuracy_trend: str = "stable"  # "improving", "declining", "stable"
    drift_detected: bool = False
    drifted_features: List[int] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)
    uptime_seconds: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"=== Model Monitor Report ===",
            f"predictions:    {self.n_predictions}",
            f"uptime:         {self.uptime_seconds:.0f}s",
            f"avg latency:    {self.avg_latency_ms:.3f}ms",
            f"p95 latency:    {self.p95_latency_ms:.3f}ms",
            f"p99 latency:    {self.p99_latency_ms:.3f}ms",
            f"accuracy:       {self.current_accuracy:.4f}",
            f"accuracy trend: {self.accuracy_trend}",
            f"drift detected: {self.drift_detected}",
        ]
        if self.drifted_features:
            lines.append(f"drifted features: {self.drifted_features}")
        if self.alerts:
            lines.append("")
            lines.append("alerts:")
            for a in self.alerts:
                lines.append(f"  ! {a}")
        else:
            lines.append("")
            lines.append("no alerts — all healthy")
        return "\n".join(lines)


class ModelMonitor:
    """Production model monitor.

    Tracks:
      - Prediction latency (avg, p50, p95, p99)
      - Accuracy over a rolling window
      - Accuracy trend (improving / declining / stable)
      - Data drift (via qai.DriftDetector)
      - Alerts when things go wrong

    Args:
        baseline_X: training data for drift detection.
        baseline_y: training labels for accuracy baseline.
        window_size: rolling window for accuracy/latency tracking.
        drift_check_interval: check drift every N predictions.
        latency_alert_ms: alert if avg latency exceeds this.
        accuracy_alert: alert if accuracy drops below this.
    """
    def __init__(self, baseline_X=None, baseline_y=None,
                 window_size: int = 500,
                 drift_check_interval: int = 100,
                 latency_alert_ms: float = 100.0,
                 accuracy_alert: float = 0.8):
        self.window_size = window_size
        self.drift_check_interval = drift_check_interval
        self.latency_alert_ms = latency_alert_ms
        self.accuracy_alert = accuracy_alert

        self._latencies: deque = deque(maxlen=window_size)
        self._correct: deque = deque(maxlen=window_size)
        self._X_buffer: deque = deque(maxlen=window_size)
        self._n_preds = 0
        self._start_time = time.time()
        self._last_drift_check = 0
        self._last_accuracy = 1.0

        # Set up drift detector
        self.drift_detector = None
        if baseline_X is not None and DriftDetector is not None:
            self.drift_detector = DriftDetector(baseline=baseline_X, method="ks")

    def record(self, x, prediction, actual=None, latency_ms: float = 0.0):
        """Record a prediction event.

        Args:
            x: the input that was predicted.
            prediction: the model's prediction.
            actual: the actual label (None if not available yet).
            latency_ms: prediction latency in milliseconds.
        """
        self._latencies.append(latency_ms)
        self._X_buffer.append(np.asarray(x).flatten())
        if actual is not None:
            self._correct.append(int(prediction == actual))
        self._n_preds += 1

        # Periodic drift check
        if self.drift_detector and self._n_preds - self._last_drift_check >= self.drift_check_interval:
            self._last_drift_check = self._n_preds
            self._check_drift()

    def _check_drift(self):
        """Check for data drift on the recent buffer."""
        if len(self._X_buffer) < 10:
            return
        X_recent = np.array(list(self._X_buffer))
        try:
            report = self.drift_detector.detect(X_recent)
            if report.drifted:
                # Store the alert
                self._drift_alert = report
        except Exception:
            pass

    def report(self) -> MonitorReport:
        """Generate a monitoring report."""
        r = MonitorReport()
        r.n_predictions = self._n_preds
        r.uptime_seconds = time.time() - self._start_time

        # Latency stats
        if self._latencies:
            lats = np.array(self._latencies)
            r.avg_latency_ms = float(np.mean(lats))
            r.p95_latency_ms = float(np.percentile(lats, 95))
            r.p99_latency_ms = float(np.percentile(lats, 99))

        # Accuracy
        if self._correct:
            current_acc = float(np.mean(self._correct))
            r.current_accuracy = current_acc

            # Trend: compare first half vs second half of the window
            if len(self._correct) >= 20:
                half = len(self._correct) // 2
                first_half = float(np.mean(list(self._correct)[:half]))
                second_half = float(np.mean(list(self._correct)[half:]))
                if second_half > first_half + 0.02:
                    r.accuracy_trend = "improving"
                elif second_half < first_half - 0.02:
                    r.accuracy_trend = "declining"
                else:
                    r.accuracy_trend = "stable"

            self._last_accuracy = current_acc

        # Drift
        if hasattr(self, "_drift_alert"):
            r.drift_detected = self._drift_alert.drifted
            r.drifted_features = self._drift_alert.drifted_features

        # Alerts
        if r.avg_latency_ms > self.latency_alert_ms:
            r.alerts.append(f"High latency: {r.avg_latency_ms:.1f}ms > {self.latency_alert_ms}ms threshold")
        if r.current_accuracy < self.accuracy_alert and r.current_accuracy > 0:
            r.alerts.append(f"Low accuracy: {r.current_accuracy:.3f} < {self.accuracy_alert} threshold")
        if r.accuracy_trend == "declining":
            r.alerts.append("Accuracy declining — consider retraining")
        if r.drift_detected:
            r.alerts.append(f"Data drift detected on features: {r.drifted_features}")

        return r

    def reset(self):
        """Reset all counters."""
        self._latencies.clear()
        self._correct.clear()
        self._X_buffer.clear()
        self._n_preds = 0
        self._start_time = time.time()
        self._last_drift_check = 0
        if hasattr(self, "_drift_alert"):
            del self._drift_alert


__all__ = ["ModelMonitor", "MonitorReport"]
