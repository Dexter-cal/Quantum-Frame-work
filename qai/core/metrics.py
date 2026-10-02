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
