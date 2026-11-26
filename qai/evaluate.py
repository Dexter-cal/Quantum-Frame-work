"""
qai.evaluate -- one-call evaluation suite.

Picks the right metrics for the technique type and returns a formatted
report. No more remembering whether to call classification_report vs
mean_absolute_error vs silhouette_score -- qai.evaluate() figures it out
from the model + the data shape.

Usage:
    import qai
    m = qai.build(technique="classifier")
    m.train(X_train, y_train)
    report = qai.evaluate(m, X_test, y_test)
    print(report)              # formatted text
    report.as_dict()           # raw metrics

The report includes:
  - The technique name + family
  - Appropriate metrics for the task (classification / regression /
    clustering / reinforcement)
  - A short verdict ("good" / "ok" / "poor") based on metric thresholds
  - Optional: per-class breakdown for classifiers
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np
import qai


class EvaluationReport:
    """A formatted evaluation report.

    Attributes:
        technique: the technique name (e.g. "classifier")
        family: the technique family (e.g. "classical_ml")
        metrics: dict of metric name -> value
        verdict: short verdict string ("excellent" / "good" / "ok" / "poor")
        notes: list of human-readable notes
    """
    def __init__(self, technique: str, family: str, metrics: Dict[str, float],
                 verdict: str = "", notes: Optional[list] = None):
        self.technique = technique
        self.family = family
        self.metrics = metrics
        self.verdict = verdict or "unknown"
        self.notes = notes or []

    def as_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "family": self.family,
            "metrics": self.metrics,
            "verdict": self.verdict,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        lines = [
            f"=== Evaluation Report ===",
            f"technique: {self.technique}  (family: {self.family})",
            f"verdict:   {self.verdict}",
            "",
            "metrics:",
        ]
        for k, v in self.metrics.items():
            if isinstance(v, float):
                lines.append(f"  {k:30s} {v:.4f}")
            else:
                lines.append(f"  {k:30s} {v}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)

    def __repr__(self):
        return f"<EvaluationReport technique={self.technique!r} verdict={self.verdict!r}>"


def evaluate(model, X, y=None, verbose: bool = False) -> EvaluationReport:
    """Evaluate `model` on (X, y) and return an EvaluationReport.

    The metric selection is driven by the model's `family` attribute:
      - "classical_ml" / "deep_learning" with classification-like y
        -> accuracy, precision, recall, f1, confusion_matrix
      - regression-like y (continuous)
        -> MSE, MAE, R², explained_variance
      - "unsupervised" / clustering
        -> silhouette, davies_bouldin, n_clusters
      - "reinforcement"
        -> total_reward, n_episodes, success_rate

    Args:
        model: a qai.Model (from qai.build(...)).
        X: test inputs.
        y: test targets (None for unsupervised).
        verbose: if True, print the report.

    Returns:
        EvaluationReport
    """
    tech = getattr(model, "technique", None)
    technique_name = getattr(tech, "name", "unknown") if tech else "unknown"
    family = getattr(tech, "family", "unknown") if tech else "unknown"
    metrics: Dict[str, Any] = {}
    notes: list = []
    verdict = "unknown"

    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    # ----- Classification -----
    is_classification = (
        y is not None and y.dtype.kind in ("i", "u", "S", "U", "O")
        and (y.ndim == 0 or len(np.unique(y)) <= 20)
    )
    if is_classification and family in ("classical_ml", "deep_learning"):
        try:
            preds = model.predict(X)
            preds = np.asarray(preds).flatten()
            y_flat = y.flatten()
            metrics["accuracy"] = float(np.mean(preds == y_flat))
            # Try sklearn's classification_report
            try:
                report = qai.classification_report(y_flat, preds)
                metrics["classification_report"] = report
            except Exception:
                pass
            try:
                cm = qai.confusion_matrix(y_flat, preds)
                metrics["confusion_matrix"] = cm.tolist()
            except Exception:
                pass
            try:
                metrics["balanced_accuracy"] = float(qai.balanced_accuracy_score(y_flat, preds))
            except Exception:
                pass
            try:
                metrics["cohen_kappa"] = float(qai.cohen_kappa_score(y_flat, preds))
            except Exception:
                pass
            # Verdict
            acc = metrics["accuracy"]
            if acc >= 0.95: verdict = "excellent"
            elif acc >= 0.85: verdict = "good"
            elif acc >= 0.7: verdict = "ok"
            else: verdict = "poor"
            notes.append(f"Verdict based on accuracy = {acc:.3f}")
        except Exception as e:
            notes.append(f"Classification metrics failed: {e}")

    # ----- Regression -----
    elif y is not None and y.dtype.kind == "f" and family in ("classical_ml", "deep_learning"):
        try:
            preds = model.predict(X)
            preds = np.asarray(preds).flatten().astype(float)
            y_flat = y.flatten().astype(float)
            metrics["mse"] = float(np.mean((preds - y_flat) ** 2))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(np.mean(np.abs(preds - y_flat)))
            try: metrics["r2"] = float(qai.r2_score(y_flat, preds))
            except Exception: pass
            try: metrics["explained_variance"] = float(qai.explained_variance_score(y_flat, preds))
            except Exception: pass
            try: metrics["max_error"] = float(qai.max_error(y_flat, preds))
            except Exception: pass
            # Verdict
            r2 = metrics.get("r2", 0)
            if r2 >= 0.9: verdict = "excellent"
            elif r2 >= 0.7: verdict = "good"
            elif r2 >= 0.4: verdict = "ok"
            else: verdict = "poor"
            notes.append(f"Verdict based on R² = {r2:.3f}")
        except Exception as e:
            notes.append(f"Regression metrics failed: {e}")

    # ----- Clustering / Unsupervised -----
    elif family in ("unsupervised",) or y is None:
        try:
            preds = model.predict(X) if hasattr(model, "predict") else None
            if preds is not None:
                preds = np.asarray(preds).flatten()
                metrics["n_clusters_found"] = int(len(np.unique(preds)))
            try:
                sil = float(qai.silhouette_score(X, preds))
                metrics["silhouette"] = sil
                if sil >= 0.7: verdict = "excellent"
                elif sil >= 0.5: verdict = "good"
                elif sil >= 0.25: verdict = "ok"
                else: verdict = "poor"
                notes.append(f"Verdict based on silhouette = {sil:.3f}")
            except Exception:
                pass
            try:
                metrics["davies_bouldin"] = float(qai.davies_bouldin_score(X, preds))
            except Exception:
                pass
        except Exception as e:
            notes.append(f"Clustering metrics failed: {e}")

    # ----- Reinforcement -----
    elif family == "reinforcement":
        if hasattr(model, "status"):
            try:
                s = model.status() if callable(getattr(model, "status")) else {}
                metrics["n_episodes"] = s.get("n_episodes_completed", s.get("episodes_completed", 0))
                metrics["n_successes"] = s.get("n_successes", s.get("successes", 0))
                metrics["total_reward"] = s.get("total_reward", 0.0)
                n_ep = metrics["n_episodes"]
                if n_ep > 0:
                    metrics["success_rate"] = metrics["n_successes"] / n_ep
                    sr = metrics["success_rate"]
                    if sr >= 0.8: verdict = "excellent"
                    elif sr >= 0.5: verdict = "good"
                    elif sr >= 0.2: verdict = "ok"
                    else: verdict = "poor"
                    notes.append(f"Verdict based on success rate = {sr:.3f}")
            except Exception as e:
                notes.append(f"RL metrics failed: {e}")

    # ----- Fallback -----
    if not metrics:
        notes.append("No metrics could be computed -- technique family unknown or unsupported.")
        verdict = "unknown"

    report = EvaluationReport(technique_name, family, metrics, verdict, notes)
    if verbose:
        print(report)
    return report


__all__ = ["evaluate", "EvaluationReport"]
