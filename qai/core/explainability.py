"""
qai.core.explainability -- Model interpretability & feature importance utilities.
"""
from typing import Dict, Any, List
import numpy as np


def feature_importance(model: Any) -> Dict[str, float]:
    """Extracts feature importances or linear coefficients from a trained QAI model."""
    inner = getattr(model, "model", model)

    if hasattr(inner, "feature_importances_"):
        imps = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        coef = inner.coef_
        imps = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        raise AttributeError("Model does not expose feature_importances_ or coef_")

    imps = np.array(imps)
    norm_imps = imps / (imps.sum() + 1e-12)
    return {f"feature_{i}": float(v) for i, v in enumerate(norm_imps)}


def permutation_importance(model: Any, X: Any, y: Any, metric_fn: Any = None, n_repeats: int = 5) -> Dict[str, float]:
    """Computes permutation feature importance by evaluating metric drop upon shuffling each feature."""
    X_arr = np.array(X)
    y_arr = np.array(y)

    if metric_fn is None:
        def default_metric(m, x_d, y_d):
            preds = m.predict(x_d)
            return float(np.mean(preds == y_d) if y_d.ndim == 1 and np.issubdtype(y_d.dtype, np.integer) else -np.mean((preds - y_d) ** 2))
        metric_fn = default_metric

    baseline = metric_fn(model, X_arr, y_arr)
    n_features = X_arr.shape[1]
    importances = []

    for col in range(n_features):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_arr.copy()
            np.random.shuffle(X_perm[:, col])
            score_perm = metric_fn(model, X_perm, y_arr)
            drops.append(baseline - score_perm)
        importances.append(float(np.mean(drops)))

    norm = np.array(importances)
    norm = np.maximum(0, norm)
    total = norm.sum()
    if total > 0:
        norm = norm / total

    return {f"feature_{i}": float(v) for i, v in enumerate(norm)}
