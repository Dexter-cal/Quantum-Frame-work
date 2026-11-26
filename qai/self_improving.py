"""
qai.self_improving -- production-grade self-improving / continuously-learning models.

This is the production version of the "Self-Improving Models" Phase 6.4 idea
from the roadmap. It bundles five cooperating components:

  - ContinuousLearner    : wraps a model and learns from streaming data
  - AutoRetrainer        : watches a deployed model and retrains on degradation
  - KnowledgeAccumulator : persists knowledge across sessions (a model "memory")
  - SelfEvaluator        : introspects predictions to find blind spots
  - ModelGuardian        : top-level orchestrator wiring the above together

Usage:
    import qai
    from qai.self_improving import wrap

    m = qai.build(technique="gaussian_nb")
    m.train(X, y)
    g = wrap(m)
    pred = g.serve(X[0])
    print(g.dashboard())
    print(g.alerts())

Design notes:
  - numpy + stdlib only (no torch).
  - Each class works with any model exposing .predict(), .train(X, y), and
    optionally .score(X, y) / .predict_proba(x). Missing methods are
    detected and skipped gracefully.
  - Persistence uses json + pickle.
"""
from __future__ import annotations

import os
import json
import time
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers — model-agnostic adapters
# ---------------------------------------------------------------------------

def _safe_train(model, X, y) -> bool:
    """Try to train `model` on (X, y); accept verbose kwarg if present.

    Returns True on success, False on failure. Tries `train()` then `fit()`.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    for method_name in ("train", "fit"):
        fn = getattr(model, method_name, None)
        if fn is None:
            continue
        try:
            import inspect
            sig = inspect.signature(fn)
            if "verbose" in sig.parameters:
                fn(X, y, verbose=False)
            else:
                fn(X, y)
            return True
        except Exception:
            continue
    return False


def _batch_predict(model, X) -> np.ndarray:
    """Predict on a batch. Tries batched predict first, falls back row-by-row.

    Some qai techniques return only the first row's prediction when handed a
    2-D array; the row-by-row fallback guarantees full coverage.
    """
    X = np.asarray(X)
    n = len(X)
    if n == 0:
        return np.array([])
    # Attempt batched prediction (sklearn models accept 2-D directly)
    try:
        preds = model.predict(X)
        if hasattr(preds, "__len__") and len(preds) == n:
            return np.asarray(preds).flatten()
    except Exception:
        pass
    # Row-by-row fallback
    out: List[Any] = []
    for i in range(n):
        try:
            out.append(model.predict(X[i]))
        except Exception:
            out.append(None)
    return np.asarray([p if p is not None else -1 for p in out])


def _score(model, X, y, metric: str = "accuracy") -> float:
    """Compute a metric for `model` on (X, y).

    For known metrics ('accuracy', 'mse', 'mae'), compute directly from
    predictions so the result is unambiguous. For other metric names,
    fall back to model.score(X, y), then model.evaluate(X, y)[metric],
    then manual accuracy.
    """
    y_arr = np.asarray(y).flatten()
    if metric in ("accuracy", "mse", "mae"):
        preds = _batch_predict(model, X)
        preds = np.asarray(preds).flatten()
        if metric == "accuracy":
            return float(np.mean(preds == y_arr))
        if metric == "mse":
            return float(np.mean((preds.astype(float) - y_arr.astype(float)) ** 2))
        return float(np.mean(np.abs(preds.astype(float) - y_arr.astype(float))))
    # Unknown metric: try model.score, then evaluate(), then accuracy fallback
    if hasattr(model, "score"):
        try:
            s = model.score(X, y)
            if s is not None:
                return float(s)
        except Exception:
            pass
    if hasattr(model, "evaluate"):
        try:
            res = model.evaluate(X, y)
            if isinstance(res, dict):
                if metric in res:
                    return float(res[metric])
                if "accuracy" in res:
                    return float(res["accuracy"])
        except Exception:
            pass
    preds = _batch_predict(model, X)
    return float(np.mean(np.asarray(preds).flatten() == y_arr))


def _get_proba(model, x) -> Optional[np.ndarray]:
    """Try several ways to get class probabilities for input x.

    qai's Model wrapper exposes predict_proba when the underlying technique
    has a `.model` attribute, but some techniques store the sklearn object as
    `._model`. We probe both, plus any inner object exposing predict_proba.
    """
    # 1. Top-level wrapper method
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(x)
            if p is not None:
                return np.asarray(p, dtype=float).flatten()
        except Exception:
            pass
    # 2. Probe the technique for inner sklearn-style objects
    tech = getattr(model, "technique", None)
    if tech is not None:
        x_arr = np.asarray(x, dtype=float)
        if x_arr.ndim == 1:
            x_arr = x_arr.reshape(1, -1)
        for attr in ("_model", "model", "sklearn_model", "clf", "estimator"):
            inner = getattr(tech, attr, None)
            if inner is not None and hasattr(inner, "predict_proba"):
                try:
                    return np.asarray(inner.predict_proba(x_arr), dtype=float).flatten()
                except Exception:
                    continue
    return None


def _now_iso() -> str:
    """ISO-8601 UTC timestamp."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# 1. ContinuousLearner
# ---------------------------------------------------------------------------

class ContinuousLearner:
    """Wraps a qai model and learns continuously from streaming data.

    Args:
        model: any object with .predict() and .train(X, y).
        drift_threshold: PSI above this triggers a retrain (default 0.05).
        retrain_window: samples between drift checks (default 100).
        validation_split: fraction of the buffer used to score retrained model.
    """
    def __init__(self, model, drift_threshold: float = 0.05,
                 retrain_window: int = 100, validation_split: float = 0.2):
        self.model = model
        self.drift_threshold = float(drift_threshold)
        self.retrain_window = int(retrain_window)
        self.validation_split = float(validation_split)
        self._buffer_X: List[np.ndarray] = []
        self._buffer_y: List[Any] = []
        self._baseline: Optional[np.ndarray] = None
        self._history: List[Dict[str, Any]] = []
        self._versions: List[bytes] = []
        self._max_versions = 10
        self._n_seen = 0
        self._n_retrains = 0
        self._current_drift = 0.0
        self._last_acc: Optional[float] = None
        self._frozen = False

    def observe(self, x, y_true=None, y_pred=None):
        """Record a new observation; triggers retraining if drift > threshold."""
        try:
            x_arr = np.asarray(x, dtype=float).flatten()
        except Exception:
            x_arr = np.zeros(1, dtype=float)
        self._buffer_X.append(x_arr)
        if y_true is not None:
            self._buffer_y.append(y_true)
        if y_pred is None:
            try:
                y_pred = self.model.predict(x)
            except Exception:
                y_pred = None
        self._n_seen += 1
        # Initialize baseline on the first full window
        if self._baseline is None and len(self._buffer_X) >= self.retrain_window:
            self._baseline = np.array(self._buffer_X[:self.retrain_window])
        # Periodic drift check + possible retrain
        if (not self._frozen and self._baseline is not None
                and len(self._buffer_X) % self.retrain_window == 0):
            self._maybe_retrain()
        return y_pred

    def state(self) -> Dict[str, Any]:
        """Return current state: n_seen, n_retrains, current_drift, last_acc."""
        return {
            "n_seen": self._n_seen,
            "n_retrains": self._n_retrains,
            "current_drift": round(self._current_drift, 4),
            "last_acc": self._last_acc,
            "frozen": self._frozen,
            "buffer_size": len(self._buffer_X),
        }

    def history(self) -> List[Dict[str, Any]]:
        """List of retrain events with timestamps + metrics."""
        return list(self._history)

    def rollback(self, steps: int = 1) -> Dict[str, Any]:
        """Revert the model to N versions ago (if a snapshot exists)."""
        if not self._versions:
            raise RuntimeError("No versions saved yet -- cannot rollback")
        steps = max(1, min(steps, len(self._versions)))
        snapshot = self._versions[-steps]
        try:
            self.model = pickle.loads(snapshot)
        except Exception as e:
            raise RuntimeError(f"Rollback failed: {e}")
        self._versions = self._versions[:-steps]
        self._n_retrains = max(0, self._n_retrains - steps)
        return self.state()

    def freeze(self) -> bool:
        """Pause learning."""
        self._frozen = True
        return self._frozen

    def unfreeze(self) -> bool:
        """Resume learning."""
        self._frozen = False
        return self._frozen

    def health(self) -> Dict[str, Any]:
        """Current model health snapshot."""
        drift_ok = self._current_drift <= self.drift_threshold
        acc_ok = self._last_acc is None or self._last_acc >= 0.5
        return {
            **self.state(),
            "drift_status": "ok" if drift_ok else "drifted",
            "baseline_size": 0 if self._baseline is None else int(self._baseline.shape[0]),
            "versions_saved": len(self._versions),
            "healthy": bool(drift_ok and acc_ok),
        }

    # ----- internals -----

    def _maybe_retrain(self):
        """Compute drift; if above threshold, snapshot and retrain on buffer."""
        self._current_drift = self._compute_drift()
        if self._current_drift <= self.drift_threshold:
            return
        # Snapshot the pre-retrain model so rollback() can restore it
        try:
            snap = pickle.dumps(self.model)
            self._versions.append(snap)
            if len(self._versions) > self._max_versions:
                self._versions = self._versions[-self._max_versions:]
        except Exception:
            pass
        event: Dict[str, Any] = {
            "timestamp": time.time(), "ts_str": _now_iso(),
            "drift": round(self._current_drift, 4),
            "n_samples": len(self._buffer_X),
        }
        if len(self._buffer_y) < 10:
            event["status"] = "skipped"
            event["reason"] = "insufficient labeled samples"
            self._history.append(event)
            return
        n = len(self._buffer_y)
        X = np.array(self._buffer_X[-n:])
        y = np.array(self._buffer_y[-n:])
        if not _safe_train(self.model, X, y):
            event["status"] = "failed"
            event["reason"] = "train() raised"
            self._history.append(event)
            return
        n_val = max(1, int(n * self.validation_split))
        self._last_acc = _score(self.model, X[-n_val:], y[-n_val:], metric="accuracy")
        self._n_retrains += 1
        event["status"] = "completed"
        event["last_acc"] = self._last_acc
        self._history.append(event)

    def _compute_drift(self) -> float:
        """Average PSI across features (baseline vs most recent window)."""
        if self._baseline is None or len(self._buffer_X) < self.retrain_window:
            return 0.0
        recent = np.array(self._buffer_X[-self.retrain_window:])
        if recent.shape[1] != self._baseline.shape[1]:
            return 0.0
        psis: List[float] = []
        for i in range(self._baseline.shape[1]):
            try:
                psis.append(self._psi(self._baseline[:, i], recent[:, i]))
            except Exception:
                continue
        return float(np.mean(psis)) if psis else 0.0

    @staticmethod
    def _psi(baseline: np.ndarray, recent: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index between two 1-D distributions."""
        edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        edges[0] = -np.inf
        edges[-1] = np.inf
        base_counts = np.histogram(baseline, bins=edges)[0].astype(float)
        new_counts = np.histogram(recent, bins=edges)[0].astype(float)
        base_pct = (base_counts + 0.5) / (len(baseline) + 0.5 * n_bins)
        new_pct = (new_counts + 0.5) / (len(recent) + 0.5 * n_bins)
        return float(np.sum((new_pct - base_pct) * np.log(new_pct / base_pct)))

    def __repr__(self):
        return (f"<ContinuousLearner n_seen={self._n_seen} "
                f"n_retrains={self._n_retrains} drift={self._current_drift:.4f}>")


# ---------------------------------------------------------------------------
# 2. AutoRetrainer
# ---------------------------------------------------------------------------

class AutoRetrainer:
    """Watches a deployed model and auto-retrains when performance degrades.

    Args:
        model: the deployed model. Will be retrained in-place on degradation.
        metric: 'accuracy' | 'mse' | 'mae' (default 'accuracy').
        min_samples: minimum labelled samples needed to trigger a retrain.
        degradation_threshold: drop in metric that queues a retrain.
        max_retrains_per_day: hard cap to avoid runaway retraining.
    """
    def __init__(self, model, metric: str = "accuracy", min_samples: int = 50,
                 degradation_threshold: float = 0.05, max_retrains_per_day: int = 3):
        self.model = model
        self.metric = metric
        self.min_samples = int(min_samples)
        self.degradation_threshold = float(degradation_threshold)
        self.max_retrains_per_day = int(max_retrains_per_day)
        self._baseline_metric: Optional[float] = None
        self._pending_retrain = False
        self._pending_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._retrain_log: List[Dict[str, Any]] = []
        self._kill_switch = False

    def evaluate(self, X, y) -> Dict[str, Any]:
        """Score the model on (X, y); queue a retrain if it has degraded."""
        if len(X) < self.min_samples:
            return {"skipped": True, "reason": "insufficient samples", "n": len(X)}
        score = _score(self.model, X, y, metric=self.metric)
        if self._baseline_metric is None:
            self._baseline_metric = score
        # For 'accuracy' higher is better; for mse/mae lower is better.
        if self.metric in ("mse", "mae"):
            degradation = score - self._baseline_metric
        else:
            degradation = self._baseline_metric - score
        result: Dict[str, Any] = {
            "timestamp": time.time(), "ts_str": _now_iso(),
            "metric": self.metric, "score": round(score, 4),
            "baseline": round(self._baseline_metric, 4),
            "degradation": round(degradation, 4),
            "n_samples": len(X),
        }
        if (not self._kill_switch and degradation > self.degradation_threshold):
            self._pending_retrain = True
            self._pending_data = (X, y)
            result["queued_retrain"] = True
        else:
            result["queued_retrain"] = False
        return result

    def tick(self) -> Dict[str, Any]:
        """Process a pending retrain. Safe to call repeatedly (no-op if none)."""
        if self._kill_switch:
            return {"skipped": True, "reason": "kill_switch active"}
        if not self._pending_retrain or self._pending_data is None:
            return {"skipped": True, "reason": "no pending retrain"}
        if self._retrains_today() >= self.max_retrains_per_day:
            return {"skipped": True, "reason": "max_retrains_per_day reached"}
        X, y = self._pending_data
        old_score = _score(self.model, X, y, metric=self.metric)
        ok = _safe_train(self.model, X, y)
        new_score = _score(self.model, X, y, metric=self.metric)
        entry: Dict[str, Any] = {
            "timestamp": time.time(), "ts_str": _now_iso(),
            "old_score": round(old_score, 4), "new_score": round(new_score, 4),
            "status": "completed" if ok else "failed",
            "n_samples": len(X),
        }
        self._retrain_log.append(entry)
        if ok:
            self._baseline_metric = new_score
        self._pending_retrain = False
        self._pending_data = None
        return entry

    def swap_to(self, new_model):
        """Atomic swap of the deployed model. Returns the previous model."""
        old = self.model
        self.model = new_model
        return old

    def retrain_log(self) -> List[Dict[str, Any]]:
        """History of all retrains with old vs new metric."""
        return list(self._retrain_log)

    def kill_switch(self) -> bool:
        """Immediately stop all auto-retraining (safety). Clears pending state."""
        self._kill_switch = True
        self._pending_retrain = False
        self._pending_data = None
        return self._kill_switch

    def _retrains_today(self) -> int:
        """Count retrains in the last 24 hours."""
        cutoff = time.time() - 86400
        return sum(1 for e in self._retrain_log if e["timestamp"] >= cutoff)

    def __repr__(self):
        return (f"<AutoRetrainer metric={self.metric} "
                f"n_retrains={len(self._retrain_log)} "
                f"kill={self._kill_switch}>")


# ---------------------------------------------------------------------------
# 3. KnowledgeAccumulator
# ---------------------------------------------------------------------------

class KnowledgeAccumulator:
    """Accumulates knowledge across sessions; the model 'remembers' what it
    learned even after a restart. Knowledge is stored on disk as JSON.

    Args:
        storage_path: directory for the knowledge base. Default ~/.qai/knowledge/.
    """
    def __init__(self, storage_path: str = "~/.qai/knowledge/"):
        self.storage_path = os.path.expanduser(storage_path)
        os.makedirs(self.storage_path, exist_ok=True)
        self._kb_file = os.path.join(self.storage_path, "kb.json")
        self._kb: List[Dict[str, Any]] = self._load_kb()

    def record(self, topic, content, source: str = "observation",
               confidence: float = 1.0) -> int:
        """Add a knowledge fact. `content` may be a string, dict, or list.

        If `content` is a dict with keys 'X' and 'y', it is treated as a
        training sample pair and used by consolidate().
        """
        entry = {
            "topic": str(topic),
            "content": content,
            "source": str(source),
            "confidence": float(confidence),
            "timestamp": time.time(),
            "ts_str": _now_iso(),
            "id": hashlib.sha1(
                f"{topic}|{time.time()}|{len(self._kb)}".encode()
            ).hexdigest()[:12],
        }
        self._kb.append(entry)
        self._save_kb()
        return len(self._kb)

    def recall(self, topic) -> List[Dict[str, Any]]:
        """Retrieve knowledge whose topic or content contains `topic` (case-insensitive)."""
        needle = str(topic).lower()
        hits = [e for e in self._kb
                if needle in str(e["topic"]).lower()
                or needle in str(e["content"]).lower()]
        hits.sort(key=lambda e: (-e["confidence"], -e["timestamp"]))
        return hits

    def consolidate(self, model) -> Dict[str, Any]:
        """Distill accumulated knowledge into `model` by fine-tuning on
        stored sample pairs (entries whose content is {X, y}).
        """
        X_all: List[Any] = []
        y_all: List[Any] = []
        for e in self._kb:
            c = e["content"]
            if isinstance(c, dict) and "X" in c and "y" in c:
                try:
                    X_all.extend(np.asarray(c["X"]).tolist())
                    y_all.extend(np.asarray(c["y"]).tolist())
                except Exception:
                    continue
        if not X_all:
            return {"consolidated": False, "reason": "no sample-pair facts", "n_samples": 0}
        X = np.asarray(X_all)
        y = np.asarray(y_all)
        ok = _safe_train(model, X, y)
        return {
            "consolidated": bool(ok),
            "n_samples": len(X_all),
            "n_features": int(X.shape[1]) if X.ndim > 1 else 1,
        }

    def forget(self, topic, older_than_days: Optional[float] = None) -> int:
        """Prune knowledge matching `topic`. If older_than_days is set,
        only entries older than that are removed.
        """
        needle = str(topic).lower()
        cutoff = time.time() - (older_than_days or 0) * 86400
        kept: List[Dict[str, Any]] = []
        removed = 0
        for e in self._kb:
            matches = (needle in str(e["topic"]).lower()
                       or needle in str(e["content"]).lower())
            old_enough = (older_than_days is None) or (e["timestamp"] < cutoff)
            if matches and old_enough:
                removed += 1
            else:
                kept.append(e)
        self._kb = kept
        self._save_kb()
        return removed

    def export(self, path: str) -> str:
        """Export the knowledge base to `path` as JSON for sharing."""
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"kb": self._kb, "exported_at": _now_iso()},
                      f, indent=2, default=str)
        return path

    def import_kb(self, path: str) -> int:
        """Load a shared knowledge base from `path`. Returns count imported."""
        path = os.path.expanduser(path)
        with open(path) as f:
            data = json.load(f)
        imported = data.get("kb", [])
        for e in imported:
            self._kb.append(e)
        self._save_kb()
        return len(imported)

    # ----- internals -----

    def _save_kb(self):
        try:
            with open(self._kb_file, "w") as f:
                json.dump({"kb": self._kb}, f, indent=2, default=str)
        except Exception:
            pass

    def _load_kb(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self._kb_file):
            return []
        try:
            with open(self._kb_file) as f:
                data = json.load(f)
            return list(data.get("kb", []))
        except Exception:
            return []

    def __repr__(self):
        return f"<KnowledgeAccumulator n_facts={len(self._kb)}>"


# ---------------------------------------------------------------------------
# 4. SelfEvaluator
# ---------------------------------------------------------------------------

class SelfEvaluator:
    """Model introspects on its own predictions to find weaknesses.

    Args:
        model: a trained classifier (predict_proba optional but useful).
        calibration_threshold: ECE above this raises an alert (default 0.1).
        confidence_floor: confidences below this are 'uncertain' (default 0.5).
    """
    def __init__(self, model, calibration_threshold: float = 0.1,
                 confidence_floor: float = 0.5):
        self.model = model
        self.calibration_threshold = float(calibration_threshold)
        self.confidence_floor = float(confidence_floor)

    def find_blind_spots(self, X, y=None, n_probes: int = 100) -> List[Dict[str, Any]]:
        """Find inputs where the model is overconfident-or-wrong.

        An input is a blind spot if its confidence >= confidence_floor AND
        either (a) no labels provided, or (b) prediction != true label.
        """
        n = min(int(n_probes), len(X))
        spots: List[Dict[str, Any]] = []
        for i in range(n):
            conf, pred = self._confidence(X[i])
            if conf < self.confidence_floor:
                continue
            wrong = False
            actual = None
            if y is not None:
                try:
                    actual = np.asarray(y[i]).item()
                    wrong = (actual != pred)
                except Exception:
                    pass
            if y is None or wrong:
                spots.append({
                    "index": int(i),
                    "prediction": pred,
                    "confidence": round(float(conf), 4),
                    "actual": actual,
                    "correct": None if y is None else (not wrong),
                })
        return spots

    def confidence_audit(self, X) -> Dict[str, Any]:
        """Return summary stats on the model's confidence distribution over X."""
        confs: List[float] = []
        for i in range(min(len(X), 200)):
            conf, _ = self._confidence(X[i])
            confs.append(float(conf))
        if not confs:
            return {"n": 0}
        arr = np.asarray(confs)
        return {
            "n": int(len(arr)),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "overconfident_frac": float(np.mean(arr > 0.9)),
            "underconfident_frac": float(np.mean(arr < 0.5)),
        }

    def calibration_error(self, X, y) -> float:
        """Compute Expected Calibration Error (ECE) over 10 confidence bins."""
        n_bins = 10
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        confs: List[float] = []
        corrects: List[int] = []
        n = min(len(X), 200)
        for i in range(n):
            conf, pred = self._confidence(X[i])
            try:
                actual = np.asarray(y[i]).item()
                corrects.append(int(actual == pred))
            except Exception:
                corrects.append(0)
            confs.append(float(conf))
        if not confs:
            return 0.0
        c = np.asarray(confs)
        ok = np.asarray(corrects)
        ece = 0.0
        for b in range(n_bins):
            mask = (c >= bins[b]) & (c < bins[b + 1])
            if mask.sum() == 0:
                continue
            acc_bin = float(ok[mask].mean())
            conf_bin = float(c[mask].mean())
            ece += abs(acc_bin - conf_bin) * (mask.sum() / len(c))
        return float(ece)

    def suggest_augmentations(self, blind_spots) -> List[Dict[str, Any]]:
        """Suggest data augmentations likely to fix the given blind spots."""
        if not blind_spots:
            return []
        suggestions: List[Dict[str, Any]] = []
        avg_conf = float(np.mean([b["confidence"] for b in blind_spots]))
        n_wrong = sum(1 for b in blind_spots if b.get("correct") is False)
        suggestions.append({
            "name": "jitter",
            "params": {"noise_std": 0.05},
            "reason": "Blind spots suggest local overfitting; small Gaussian noise regularizes.",
        })
        if n_wrong > 0:
            suggestions.append({
                "name": "mixup",
                "params": {"alpha": 0.2},
                "reason": f"{n_wrong} blind spots are misclassified; mixup smooths boundaries.",
            })
        if avg_conf > 0.9:
            suggestions.append({
                "name": "label_smoothing",
                "params": {"epsilon": 0.1},
                "reason": f"avg confidence {avg_conf:.2f} > 0.9: model is overconfident.",
            })
        return suggestions

    # ----- internals -----

    def _confidence(self, x) -> Tuple[float, Any]:
        """Return (confidence, prediction) for input x."""
        proba = _get_proba(self.model, x)
        if proba is not None and len(proba) > 0:
            pred = int(np.argmax(proba))
            conf = float(proba[pred])
            return conf, pred
        # Fallback: predict + uniform confidence
        try:
            pred = self.model.predict(x)
            pred_v = np.asarray(pred).item() if hasattr(pred, "item") else pred
            try:
                return 1.0, int(pred_v)
            except (TypeError, ValueError):
                return 1.0, pred_v
        except Exception:
            return 0.0, None

    def __repr__(self):
        return f"<SelfEvaluator conf_floor={self.confidence_floor}>"


# ---------------------------------------------------------------------------
# 5. ModelGuardian
# ---------------------------------------------------------------------------

class ModelGuardian:
    """Top-level orchestrator that ties ContinuousLearner, AutoRetrainer,
    KnowledgeAccumulator, and SelfEvaluator together.

    Args:
        model: a trained qai model.
        config: optional dict; keys are forwarded to the relevant sub-component.
            Recognized keys:
                drift_threshold, retrain_window, validation_split,
                metric, min_samples, degradation_threshold, max_retrains_per_day,
                knowledge_path, calibration_threshold, confidence_floor,
                state_path.
    """
    def __init__(self, model, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = dict(config or {})
        self.learner = ContinuousLearner(
            model,
            drift_threshold=self.config.get("drift_threshold", 0.05),
            retrain_window=self.config.get("retrain_window", 100),
            validation_split=self.config.get("validation_split", 0.2),
        )
        self.retrainer = AutoRetrainer(
            model,
            metric=self.config.get("metric", "accuracy"),
            min_samples=self.config.get("min_samples", 50),
            degradation_threshold=self.config.get("degradation_threshold", 0.05),
            max_retrains_per_day=self.config.get("max_retrains_per_day", 3),
        )
        self.knowledge = KnowledgeAccumulator(
            storage_path=self.config.get("knowledge_path", "~/.qai/knowledge/")
        )
        self.evaluator = SelfEvaluator(
            model,
            calibration_threshold=self.config.get("calibration_threshold", 0.1),
            confidence_floor=self.config.get("confidence_floor", 0.5),
        )
        self._alerts: List[Dict[str, Any]] = []
        self._started_at = time.time()
        self._shutdown = False
        self._state_path = os.path.expanduser(
            self.config.get("state_path", "~/.qai/guardian_state/")
        )

    def serve(self, x):
        """Predict + observe + maybe retrain + maybe alert. Returns prediction."""
        if self._shutdown:
            raise RuntimeError("ModelGuardian has been shut down")
        try:
            pred = self.model.predict(x)
        except Exception as e:
            self._push_alert("predict_error", message=str(e), severity="critical")
            raise
        self.learner.observe(x, y_pred=pred)
        # Drift alert
        if self.learner._current_drift > self.learner.drift_threshold:
            self._push_alert(
                "drift",
                value=round(self.learner._current_drift, 4),
                threshold=self.learner.drift_threshold,
                severity="warning",
            )
        return pred

    def dashboard(self) -> Dict[str, Any]:
        """One-shot status of the whole self-improving system."""
        state = self.learner.state()
        health = self.learner.health()
        return {
            "uptime_s": round(time.time() - self._started_at, 1),
            "model_technique": getattr(self.model, "technique_name", "unknown"),
            "trained": self._is_model_trained(),
            "n_seen": state["n_seen"],
            "n_retrains": state["n_retrains"],
            "current_drift": state["current_drift"],
            "last_acc": state["last_acc"],
            "frozen": state["frozen"],
            "healthy": bool(health["healthy"]),
            "n_alerts": len(self._alerts),
            "n_pending_retrains": 1 if self.retrainer._pending_retrain else 0,
            "retrains_today": self.retrainer._retrains_today(),
            "knowledge_facts": len(self.knowledge._kb),
            "kill_switch": self.retrainer._kill_switch,
        }

    def alerts(self) -> List[Dict[str, Any]]:
        """Return pending alerts (drift, degradation, blind spots, errors)."""
        return list(self._alerts)

    def shutdown(self) -> str:
        """Graceful stop. Saves state to disk and returns the state path."""
        self._shutdown = True
        os.makedirs(self._state_path, exist_ok=True)
        # Save the model
        try:
            with open(os.path.join(self._state_path, "model.pkl"), "wb") as f:
                pickle.dump(self.model, f)
        except Exception:
            pass
        # Save metadata
        meta = {
            "shutdown_at": _now_iso(),
            "uptime_s": round(time.time() - self._started_at, 1),
            "n_seen": self.learner._n_seen,
            "n_retrains": self.learner._n_retrains,
            "n_alerts": len(self._alerts),
            "learner_history": self.learner._history,
            "retrain_log": self.retrainer._retrain_log,
            "alerts": self._alerts,
        }
        try:
            with open(os.path.join(self._state_path, "guardian_state.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)
        except Exception:
            pass
        return self._state_path

    # ----- extras -----

    def save_state(self, path: str) -> str:
        """Save the guardian (model + metadata) to `path`. Returns the path."""
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(self.model, f)
        meta = {
            "saved_at": _now_iso(),
            "uptime_s": round(time.time() - self._started_at, 1),
            "n_seen": self.learner._n_seen,
            "n_retrains": self.learner._n_retrains,
            "learner_history": self.learner._history,
            "retrain_log": self.retrainer._retrain_log,
        }
        with open(os.path.join(path, "guardian_state.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)
        return path

    def evaluate(self, X, y) -> Dict[str, Any]:
        """Convenience: run AutoRetrainer.evaluate + tick + audit blind spots."""
        ev = self.retrainer.evaluate(X, y)
        tick = self.retrainer.tick()
        # Compute calibration if labels are available
        try:
            ece = self.evaluator.calibration_error(X, y)
        except Exception:
            ece = None
        if ece is not None and ece > self.evaluator.calibration_threshold:
            self._push_alert("calibration", value=round(ece, 4),
                             threshold=self.evaluator.calibration_threshold,
                             severity="warning")
        return {"evaluate": ev, "tick": tick, "ece": ece}

    def find_blind_spots(self, X, y=None, n_probes: int = 100):
        """Convenience: find blind spots and push an alert if any are found."""
        spots = self.evaluator.find_blind_spots(X, y, n_probes=n_probes)
        if spots:
            self._push_alert("blind_spots", count=len(spots),
                             severity="info", sample=spots[:3])
        return spots

    # ----- internals -----

    def _push_alert(self, kind: str, severity: str = "warning", **fields):
        alert = {
            "kind": kind,
            "severity": severity,
            "timestamp": time.time(),
            "ts_str": _now_iso(),
            **fields,
        }
        self._alerts.append(alert)
        # Keep the alert log bounded
        if len(self._alerts) > 100:
            self._alerts = self._alerts[-100:]

    def _is_model_trained(self) -> bool:
        tech = getattr(self.model, "technique", None)
        if tech is not None and hasattr(tech, "is_trained"):
            try:
                return bool(tech.is_trained())
            except Exception:
                return False
        return True  # assume trained if we can't tell

    def __repr__(self):
        d = self.dashboard()
        return (f"<ModelGuardian technique={d['model_technique']} "
                f"n_seen={d['n_seen']} n_retrains={d['n_retrains']} "
                f"healthy={d['healthy']} n_alerts={d['n_alerts']}>")


# ---------------------------------------------------------------------------
# Top-level convenience functions
# ---------------------------------------------------------------------------

def wrap(model, **kwargs) -> ModelGuardian:
    """Wrap `model` in a ModelGuardian with optional config kwargs.

    Example:
        g = wrap(m, drift_threshold=0.1, retrain_window=50)
    """
    return ModelGuardian(model, config=kwargs)


def from_checkpoint(path: str) -> ModelGuardian:
    """Restore a ModelGuardian from a saved-state directory.

    The directory must contain `model.pkl` (and optionally
    `guardian_state.json`). The restored guardian's history and counts are
    populated from the JSON if present.
    """
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"[WHAT] checkpoint directory not found: {path}\n"
            f"[WHY] from_checkpoint expects a directory created by "
            f"ModelGuardian.save_state() or .shutdown()\n"
            f"[FIX] pass the path returned by save_state()/shutdown()"
        )
    model_path = os.path.join(path, "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pkl missing in {path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    g = ModelGuardian(model)
    # Restore metadata
    state_path = os.path.join(path, "guardian_state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path) as f:
                meta = json.load(f)
            g.learner._history = list(meta.get("learner_history", []))
            g.learner._n_seen = int(meta.get("n_seen", 0))
            g.learner._n_retrains = int(meta.get("n_retrains", 0))
            g.retrainer._retrain_log = list(meta.get("retrain_log", []))
            g._alerts = list(meta.get("alerts", []))
            uptime = float(meta.get("uptime_s", 0))
            g._started_at = time.time() - uptime
        except Exception:
            pass
    return g


__all__ = [
    "ContinuousLearner",
    "AutoRetrainer",
    "KnowledgeAccumulator",
    "SelfEvaluator",
    "ModelGuardian",
    "wrap",
    "from_checkpoint",
]
