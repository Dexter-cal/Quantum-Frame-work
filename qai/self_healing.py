"""
qai.self_healing — a model wrapper that detects performance degradation
and automatically retrains when accuracy drops below a threshold.

Usage:
    import qai
    m = qai.build(technique="random_forest")
    m.train(X_train, y_train)

    healer = qai.SelfHealingModel(
        model=m,
        X_baseline=X_train,
        y_baseline=y_train,
        accuracy_threshold=0.85,
        check_interval=100,  # check every 100 predictions
    )

    # In production:
    for x in incoming:
        pred = healer.predict(x)
        healer.record_outcome(x, actual_label)
        # healer auto-retrains if accuracy drops
"""
from __future__ import annotations
import time, threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from collections import deque
import numpy as np


@dataclass
class HealingEvent:
    """Record of a self-healing event."""
    timestamp: str
    event_type: str  # "degradation_detected", "retrain_started", "retrain_completed", "retrain_failed"
    old_accuracy: float
    new_accuracy: float
    n_samples_since_last_check: int
    details: str = ""


class SelfHealingModel:
    """A model wrapper that monitors performance and auto-retrains on degradation.

    Args:
        model: the underlying qai.Model.
        X_baseline: training data for retraining.
        y_baseline: training labels for retraining.
        accuracy_threshold: minimum acceptable accuracy (default 0.85).
        check_interval: check accuracy every N predictions (default 100).
        retrain_callback: optional callback called after retraining.
        window_size: number of recent predictions to evaluate accuracy over.
    """
    def __init__(self, model, X_baseline, y_baseline,
                 accuracy_threshold: float = 0.85,
                 check_interval: int = 100,
                 retrain_callback: Optional[Callable] = None,
                 window_size: int = 200):
        self.model = model
        self.X_baseline = np.asarray(X_baseline)
        self.y_baseline = np.asarray(y_baseline)
        self.accuracy_threshold = accuracy_threshold
        self.check_interval = check_interval
        self.retrain_callback = retrain_callback
        self.window_size = window_size

        self._predictions: deque = deque(maxlen=window_size)
        self._outcomes: deque = deque(maxlen=window_size)
        self._n_since_check = 0
        self._current_accuracy = 1.0
        self._events: List[HealingEvent] = []
        self._lock = threading.Lock()
        self._n_retrains = 0

    def predict(self, x):
        """Make a prediction (monitored)."""
        with self._lock:
            pred = self.model.predict(x)
            self._predictions.append(pred)
            self._n_since_check += 1
            return pred

    def record_outcome(self, x, actual_label):
        """Record the actual outcome for a prediction."""
        with self._lock:
            self._outcomes.append(actual_label)
            if self._n_since_check >= self.check_interval and len(self._outcomes) >= 10:
                self._check_and_heal()

    def _check_and_heal(self):
        """Check accuracy and retrain if needed."""
        n = min(len(self._predictions), len(self._outcomes))
        if n < 5:
            return

        preds = list(self._predictions)[-n:]
        actuals = list(self._outcomes)[-n:]
        accuracy = float(np.mean(np.array(preds) == np.array(actuals)))
        old_accuracy = self._current_accuracy
        self._current_accuracy = accuracy
        self._n_since_check = 0

        if accuracy < self.accuracy_threshold:
            self._log_event("degradation_detected", old_accuracy, accuracy,
                             f"accuracy {accuracy:.3f} < threshold {self.accuracy_threshold}")
            self._retrain()

    def _retrain(self):
        """Retrain the model on the baseline data."""
        self._log_event("retrain_started", self._current_accuracy, 0.0,
                         "starting retrain on baseline data")
        try:
            self.model.train(self.X_baseline, self.y_baseline, verbose=False)
            # Evaluate on the recent window
            n = min(len(self._predictions), len(self._outcomes))
            if n >= 5:
                preds = list(self._predictions)[-n:]
                actuals = list(self._outcomes)[-n:]
                new_acc = float(np.mean(np.array(preds) == np.array(actuals)))
            else:
                new_acc = self._current_accuracy

            self._current_accuracy = new_acc
            self._n_retrains += 1
            self._log_event("retrain_completed", 0.0, new_acc,
                             f"retrain #{self._n_retrains} complete, accuracy now {new_acc:.3f}")

            if self.retrain_callback:
                try:
                    self.retrain_callback(self.model, new_acc)
                except Exception:
                    pass
        except Exception as e:
            self._log_event("retrain_failed", self._current_accuracy, 0.0, str(e))

    def _log_event(self, event_type: str, old_acc: float, new_acc: float, details: str):
        self._events.append(HealingEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            event_type=event_type, old_accuracy=old_acc,
            new_accuracy=new_acc, n_samples_since_last_check=self.check_interval,
            details=details,
        ))

    def status(self) -> Dict[str, Any]:
        """Return current health status."""
        return {
            "current_accuracy": round(self._current_accuracy, 4),
            "accuracy_threshold": self.accuracy_threshold,
            "n_predictions": len(self._predictions),
            "n_outcomes": len(self._outcomes),
            "n_retrains": self._n_retrains,
            "n_events": len(self._events),
            "healthy": self._current_accuracy >= self.accuracy_threshold,
        }

    def event_log(self, last_n: int = 10) -> List[Dict]:
        """Return recent healing events."""
        return [
            {"timestamp": e.timestamp, "type": e.event_type,
             "old_accuracy": round(e.old_accuracy, 4),
             "new_accuracy": round(e.new_accuracy, 4),
             "details": e.details}
            for e in self._events[-last_n:]
        ]

    def __repr__(self):
        s = self.status()
        return f"<SelfHealingModel accuracy={s['current_accuracy']:.3f} healthy={s['healthy']} retrains={s['n_retrains']}>"


__all__ = ["SelfHealingModel", "HealingEvent"]
