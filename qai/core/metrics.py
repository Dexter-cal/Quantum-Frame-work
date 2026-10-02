"""
qai.core.metrics -- Classification and regression evaluation metrics.
Provides classification_report(), confusion_matrix(), and mean_squared_error().
"""
from __future__ import annotations
from typing import Dict, Any
import numpy as np


def classification_report(y_true, y_pred) -> Dict[str, Any]:
    """Generates precision, recall, and f1-score metrics for classification targets."""
    from sklearn.metrics import classification_report as sk_report
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch between y_true ({len(y_t)}) and y_pred ({len(y_p)})")
    return sk_report(y_t, y_p, output_dict=True)


def confusion_matrix(y_true, y_pred) -> np.ndarray:
    """Computes confusion matrix for classification targets."""
    from sklearn.metrics import confusion_matrix as sk_cm
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    return sk_cm(y_t, y_p)


def mean_absolute_error(y_true, y_pred) -> float:
    """Computes mean absolute error for regression targets."""
    from sklearn.metrics import mean_absolute_error as sk_mae
    return float(sk_mae(y_true, y_pred))


def r2_score(y_true, y_pred) -> float:
    """Computes R2 coefficient of determination score for regression targets."""
    from sklearn.metrics import r2_score as sk_r2
    return float(sk_r2(y_true, y_pred))


def log_loss(y_true, y_pred_proba) -> float:
    """Computes cross-entropy / log loss for classification probabilities."""
    from sklearn.metrics import log_loss as sk_log_loss
    return float(sk_log_loss(y_true, y_pred_proba))


def roc_auc_score(y_true, y_score, multi_class="ovr") -> float:
    """Computes Area Under the Receiver Operating Characteristic Curve (ROC AUC)."""
    from sklearn.metrics import roc_auc_score as sk_auc
    return float(sk_auc(y_true, y_score, multi_class=multi_class))
