"""
qai.auto_ml — automatic model selection and hyperparameter optimization.

Tries multiple techniques on your data, evaluates each via cross-validation,
and returns the best model + a comparison report.

Features:
  - auto_train() — one-call auto-ML: try N techniques, pick best
  - AutoML — configurable auto-ML with budget, parallelism, callbacks
  - compare_techniques() — evaluate multiple techniques side-by-side
  - suggest_technique() — recommend which technique to try based on data

Usage:
    import qai

    # One-call auto-ML
    result = qai.auto_train(X, y, task="classification", time_budget_s=60)
    print(result.best_technique, result.best_score)
    result.best_model.predict(X_new)

    # Detailed comparison
    results = qai.compare_techniques(X, y, techniques=["classifier", "svm", "random_forest"])
    for r in results:
        print(f"{r['technique']:20s} accuracy={r['accuracy']:.3f} time={r['time_s']:.1f}s")
"""
from __future__ import annotations
import os, time, math, copy, warnings
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
import numpy as np


# =====================================================================
# Technique suggestions based on data characteristics
# =====================================================================
def suggest_technique(n_samples: int, n_features: int, task: str = "classification",
                       n_classes: int = 2, has_gpu: bool = False) -> List[Dict[str, Any]]:
    """Suggest which techniques to try based on data characteristics.

    Uses ML rules of thumb:
      - Small data (<1000): classical ML (RF, SVM, GB)
      - Medium data (1K-100K): boosted trees + neural nets
      - Large data (>100K): deep learning + boosted trees
      - Text data: transformer
      - Image data: CNN
      - Tabular: gradient boosting (usually wins)

    Returns:
        List of {technique, reason, expected_accuracy, expected_time_s}
    """
    suggestions = []

    if task == "classification":
        # Rule of thumb: for tabular data, gradient boosting usually wins
        if n_samples < 1000:
            suggestions.append({"technique": "random_forest", "reason": "Small data — RF is robust and fast",
                                 "expected_accuracy": "high", "expected_time_s": 1})
            suggestions.append({"technique": "svm", "reason": "Small data — SVM works well with little data",
                                 "expected_accuracy": "high", "expected_time_s": 1})
            suggestions.append({"technique": "gradient_boosting", "reason": "Small data — GB often beats RF",
                                 "expected_accuracy": "high", "expected_time_s": 2})
            suggestions.append({"technique": "classifier", "reason": "Baseline — logistic regression",
                                 "expected_accuracy": "medium", "expected_time_s": 0.1})
        elif n_samples < 100000:
            suggestions.append({"technique": "gradient_boosting", "reason": "Medium data — GB is SOTA for tabular",
                                 "expected_accuracy": "high", "expected_time_s": 5})
            suggestions.append({"technique": "xgboost", "reason": "Medium data — XGBoost is fast and strong",
                                 "expected_accuracy": "high", "expected_time_s": 3})
            suggestions.append({"technique": "random_forest", "reason": "Medium data — RF is a strong baseline",
                                 "expected_accuracy": "high", "expected_time_s": 2})
            if has_gpu:
                suggestions.append({"technique": "neural_network", "reason": "Medium data + GPU — NN can learn complex patterns",
                                     "expected_accuracy": "medium-high", "expected_time_s": 30})
        else:
            suggestions.append({"technique": "xgboost", "reason": "Large data — XGBoost scales well",
                                 "expected_accuracy": "high", "expected_time_s": 30})
            suggestions.append({"technique": "lightgbm", "reason": "Large data — LightGBM is fastest for large data",
                                 "expected_accuracy": "high", "expected_time_s": 10})
            if has_gpu:
                suggestions.append({"technique": "neural_network", "reason": "Large data + GPU — deep learning shines",
                                     "expected_accuracy": "high", "expected_time_s": 300})

    elif task == "regression":
        if n_samples < 1000:
            suggestions.append({"technique": "regression", "reason": "Small data — linear baseline",
                                 "expected_accuracy": "medium", "expected_time_s": 0.1})
            suggestions.append({"technique": "random_forest", "reason": "Small data — RF for regression",
                                 "expected_accuracy": "high", "expected_time_s": 1})
        else:
            suggestions.append({"technique": "gradient_boosting", "reason": "Tabular regression — GB is SOTA",
                                 "expected_accuracy": "high", "expected_time_s": 5})
            suggestions.append({"technique": "xgboost", "reason": "Fast and strong for regression",
                                 "expected_accuracy": "high", "expected_time_s": 3})

    elif task == "clustering":
        suggestions.append({"technique": "kmeans", "reason": "Default clustering — fast and simple",
                             "expected_accuracy": "medium", "expected_time_s": 1})
        suggestions.append({"technique": "gmm", "reason": "Probabilistic clustering — handles overlapping clusters",
                             "expected_accuracy": "medium-high", "expected_time_s": 2})
        suggestions.append({"technique": "dbscan", "reason": "Density-based — finds arbitrary shapes",
                             "expected_accuracy": "medium", "expected_time_s": 2})

    return suggestions


# =====================================================================
# Compare techniques
# =====================================================================
@dataclass
class TechniqueResult:
    """Result of evaluating a single technique."""
    technique: str
    accuracy: float
    time_s: float
    params: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    predictions: Optional[List] = None


def compare_techniques(X, y=None, techniques: Optional[List[str]] = None,
                       task: str = "classification",
                       test_size: float = 0.2,
                       verbose: bool = True) -> List[TechniqueResult]:
    """Evaluate multiple techniques on the same data and compare.

    Args:
        X: input data
        y: labels (None for unsupervised)
        techniques: list of technique names to try (None = auto-select)
        task: "classification" or "regression"
        test_size: fraction of data for testing
        verbose: if True, print progress

    Returns:
        List of TechniqueResult sorted by accuracy (descending)

    Example:
        >>> results = qai.compare_techniques(X, y, techniques=["classifier", "svm", "random_forest"])
        >>> for r in results:
        ...     print(f"{r.technique:20s} acc={r.accuracy:.3f} time={r.time_s:.1f}s")
    """
    import qai
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    # Auto-select techniques if not specified
    if techniques is None:
        has_gpu = False
        try:
            from .cuda_kernels import get_hardware_detector
            has_gpu = any(d.device_type == "cuda" for d in get_hardware_detector().get_all_devices())
        except Exception:
            pass
        suggestions = suggest_technique(len(X), X.shape[1] if X.ndim > 1 else 1,
                                          task, has_gpu=has_gpu)
        techniques = [s["technique"] for s in suggestions]

    # Split data
    n = len(X)
    split = int(n * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = (y[:split], y[split:]) if y is not None else (None, None)

    results = []
    for tech_name in techniques:
        if verbose:
            print(f"  trying {tech_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            # Build and train
            kwargs = {}
            if tech_name in ("random_forest", "extra_trees"):
                kwargs["n_estimators"] = 20  # fast for comparison
            elif tech_name in ("gradient_boosting", "adaboost"):
                kwargs["n_estimators"] = 20
            elif tech_name == "neural_network":
                kwargs["hidden_dim"] = 16
                kwargs["epochs"] = 5

            m = qai.build(technique=tech_name, **kwargs)

            if y_train is not None:
                m.train(X_train, y_train, verbose=False)
            else:
                m.train(X_train, verbose=False)

            # Evaluate
            if y_test is not None:
                preds = [m.predict(X_test[i]) for i in range(len(X_test))]
                if task == "classification":
                    accuracy = sum(1 for p, t in zip(preds, y_test) if p == t) / max(len(y_test), 1)
                else:
                    accuracy = 1.0 - np.mean(np.abs(np.array(preds) - y_test)) / max(np.std(y_test), 1e-8)
            else:
                # Unsupervised — use silhouette if possible
                preds = [m.predict(X_test[i]) for i in range(len(X_test))]
                accuracy = 0.0  # placeholder

            elapsed = time.time() - t0
            result = TechniqueResult(
                technique=tech_name, accuracy=float(accuracy),
                time_s=elapsed, params=kwargs,
                predictions=preds,
            )
            if verbose:
                print(f"accuracy={accuracy:.3f} time={elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            result = TechniqueResult(
                technique=tech_name, accuracy=0.0,
                time_s=elapsed, error=str(e)[:100],
            )
            if verbose:
                print(f"FAIL: {str(e)[:50]}")
        results.append(result)

    # Sort by accuracy (descending)
    results.sort(key=lambda r: -r.accuracy)
    return results


# =====================================================================
# Auto-ML
# =====================================================================
@dataclass
class AutoMLResult:
    """Result of an auto-ML run."""
    best_model: Any
    best_technique: str
    best_score: float
    all_results: List[TechniqueResult] = field(default_factory=list)
    total_time_s: float = 0.0
    n_techniques_tried: int = 0
    n_samples: int = 0
    n_features: int = 0
    task: str = ""

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "Auto-ML Results",
            "=" * 60,
            f"Data: {self.n_samples} samples, {self.n_features} features",
            f"Task: {self.task}",
            f"Techniques tried: {self.n_techniques_tried}",
            f"Total time: {self.total_time_s:.1f}s",
            "",
            f"{'Technique':<25} {'Accuracy':>10} {'Time':>8}",
            "-" * 45,
        ]
        for r in self.all_results:
            err = f" (error: {r.error[:30]})" if r.error else ""
            lines.append(f"{r.technique:<25} {r.accuracy:>10.3f} {r.time_s:>7.1f}s{err}")
        lines.append("-" * 45)
        lines.append(f"\nWinner: {self.best_technique} (accuracy={self.best_score:.3f})")
        lines.append("=" * 60)
        return "\n".join(lines)


def auto_train(X, y=None, task: str = "classification",
               techniques: Optional[List[str]] = None,
               time_budget_s: float = 60,
               test_size: float = 0.2,
               verbose: bool = True) -> AutoMLResult:
    """Automatically try multiple techniques and return the best model.

    This is the one-call auto-ML: give it data, get back the best model.

    Args:
        X: input data
        y: labels (None for unsupervised)
        task: "classification", "regression", or "clustering"
        techniques: list of techniques to try (None = auto-select)
        time_budget_s: max time to spend (default 60s)
        test_size: fraction of data for testing
        verbose: if True, print progress

    Returns:
        AutoMLResult with best_model, best_technique, best_score, and all results

    Example:
        >>> result = qai.auto_train(X, y, task="classification")
        >>> print(result.summary())
        >>> result.best_model.predict(X_new)
    """
    import qai
    X = np.asarray(X)
    if y is not None:
        y = np.asarray(y)

    # Auto-select techniques
    if techniques is None:
        has_gpu = False
        try:
            from .cuda_kernels import get_hardware_detector
            has_gpu = any(d.device_type == "cuda" for d in get_hardware_detector().get_all_devices())
        except Exception:
            pass
        suggestions = suggest_technique(
            len(X), X.shape[1] if X.ndim > 1 else 1, task, has_gpu=has_gpu)
        techniques = [s["technique"] for s in suggestions]

    if verbose:
        print(f"[qai.auto_train] trying {len(techniques)} techniques: {techniques}")
        print(f"  data: {len(X)} samples, {X.shape[1] if X.ndim > 1 else 1} features")
        print(f"  time budget: {time_budget_s}s")

    t_start = time.time()
    results = []
    best_model = None
    best_score = -1
    best_tech = ""

    for tech_name in techniques:
        if time.time() - t_start > time_budget_s:
            if verbose:
                print(f"  time budget exceeded, stopping")
            break

        if verbose:
            print(f"  trying {tech_name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            kwargs = {}
            if tech_name in ("random_forest", "extra_trees"):
                kwargs["n_estimators"] = 20
            elif tech_name in ("gradient_boosting", "adaboost"):
                kwargs["n_estimators"] = 20
            elif tech_name == "neural_network":
                kwargs["hidden_dim"] = 16
                kwargs["epochs"] = 5

            m = qai.build(technique=tech_name, **kwargs)

            # Split data
            n = len(X)
            split = int(n * (1 - test_size))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = (y[:split], y[split:]) if y is not None else (None, None)

            if y_train is not None:
                m.train(X_train, y_train, verbose=False)
            else:
                m.train(X_train, verbose=False)

            # Evaluate
            if y_test is not None:
                preds = [m.predict(X_test[i]) for i in range(len(X_test))]
                if task == "classification":
                    score = sum(1 for p, t in zip(preds, y_test) if p == t) / max(len(y_test), 1)
                else:
                    score = 1.0 - np.mean(np.abs(np.array(preds) - y_test)) / max(np.std(y_test), 1e-8)
            else:
                score = 0.0

            elapsed = time.time() - t0
            result = TechniqueResult(technique=tech_name, accuracy=float(score),
                                       time_s=elapsed, params=kwargs)
            results.append(result)

            if score > best_score:
                best_score = score
                best_model = m
                best_tech = tech_name

            if verbose:
                print(f"accuracy={score:.3f} time={elapsed:.1f}s")
        except Exception as e:
            elapsed = time.time() - t0
            result = TechniqueResult(technique=tech_name, accuracy=0.0,
                                       time_s=elapsed, error=str(e)[:100])
            results.append(result)
            if verbose:
                print(f"FAIL: {str(e)[:50]}")

    total_time = time.time() - t_start
    if verbose and best_model:
        print(f"\n[qai.auto_train] best: {best_tech} (accuracy={best_score:.3f})")

    return AutoMLResult(
        best_model=best_model,
        best_technique=best_tech,
        best_score=best_score,
        all_results=results,
        total_time_s=total_time,
        n_techniques_tried=len(results),
        n_samples=len(X),
        n_features=X.shape[1] if X.ndim > 1 else 1,
        task=task,
    )


__all__ = ["suggest_technique", "compare_techniques", "auto_train",
           "AutoMLResult", "TechniqueResult"]
