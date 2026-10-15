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


def cohen_kappa_score(y_true: Any, y_pred: Any) -> float:
    """Computes Cohen's Kappa score for inter-annotator agreement."""
    from sklearn.metrics import cohen_kappa_score as sk_kappa
    return float(sk_kappa(y_true, y_pred))


def matthews_corrcoef(y_true: Any, y_pred: Any) -> float:
    """Computes Matthews Correlation Coefficient (MCC)."""
    from sklearn.metrics import matthews_corrcoef as sk_mcc
    return float(sk_mcc(y_true, y_pred))


def balanced_accuracy_score(y_true: Any, y_pred: Any) -> float:
    """Computes Balanced Accuracy score."""
    from sklearn.metrics import balanced_accuracy_score as sk_bal_acc
    return float(sk_bal_acc(y_true, y_pred))


def silhouette_score(X: Any, labels: Any) -> float:
    """Computes Silhouette Coefficient for clustering validation."""
    from sklearn.metrics import silhouette_score as sk_sil
    return float(sk_sil(X, labels))


def davies_bouldin_score(X: Any, labels: Any) -> float:
    """Computes Davies-Bouldin score for clustering validation."""
    from sklearn.metrics import davies_bouldin_score as sk_db
    return float(sk_db(X, labels))


def adjusted_rand_score(labels_true: Any, labels_pred: Any) -> float:
    """Computes Adjusted Rand Index (ARI) for clustering similarity."""
    from sklearn.metrics import adjusted_rand_score as sk_ari
    return float(sk_ari(labels_true, labels_pred))


def normalized_mutual_info_score(labels_true: Any, labels_pred: Any) -> float:
    """Computes Normalized Mutual Information (NMI) for clustering."""
    from sklearn.metrics import normalized_mutual_info_score as sk_nmi
    return float(sk_nmi(labels_true, labels_pred))


def mean_squared_log_error(y_true: Any, y_pred: Any) -> float:
    """Computes Mean Squared Logarithmic Error (MSLE)."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    if np.any(yt < 0) or np.any(yp < 0):
        raise ValueError("Mean Squared Logarithmic Error cannot be used when targets/predictions are negative.")
    return float(np.mean((np.log1p(yt) - np.log1p(yp)) ** 2))


def mean_absolute_percentage_error(y_true: Any, y_pred: Any) -> float:
    """Computes Mean Absolute Percentage Error (MAPE)."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    denom = np.abs(yt)
    denom[denom == 0] = 1e-12
    return float(np.mean(np.abs((yt - yp) / denom)))


def median_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Computes Median Absolute Error (MedAE)."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return float(np.median(np.abs(yt - yp)))


def explained_variance_score(y_true: Any, y_pred: Any) -> float:
    """Computes Explained Variance Score."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    var_diff = np.var(yt - yp)
    var_true = np.var(yt)
    return float(1.0 - (var_diff / (var_true + 1e-12)))


def max_error(y_true: Any, y_pred: Any) -> float:
    """Computes Maximum Error across all sample pairs."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return float(np.max(np.abs(yt - yp)))


def brier_score(y_true: Any, y_prob: Any) -> float:
    """Computes Brier Score for probability calibration."""
    from qai.math import brier_score as q_brier
    return q_brier(y_true, y_prob)


def expected_calibration_error(y_true: Any, y_prob: Any, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    from qai.math import expected_calibration_error as q_ece
    return q_ece(y_true, y_prob, n_bins=n_bins)


def precision_recall_curve(y_true: Any, y_score: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes precision-recall pairs for different probability thresholds."""
    from sklearn.metrics import precision_recall_curve as sk_pr
    return sk_pr(y_true, y_score)


def roc_curve(y_true: Any, y_score: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Receiver Operating Characteristic (ROC) curve."""
    from sklearn.metrics import roc_curve as sk_roc
    return sk_roc(y_true, y_score)


def det_curve(y_true: Any, y_score: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes Detection Error Tradeoff (DET) curve."""
    from sklearn.metrics import det_curve as sk_det
    return sk_det(y_true, y_score)


def coverage_error(y_true: Any, y_score: Any) -> float:
    """Computes coverage error measure for multi-label classification."""
    from sklearn.metrics import coverage_error as sk_coverage
    return float(sk_coverage(y_true, y_score))


def label_ranking_average_precision_score(y_true: Any, y_score: Any) -> float:
    """Computes Label Ranking Average Precision (LRAP)."""
    from sklearn.metrics import label_ranking_average_precision_score as sk_lrap
    return float(sk_lrap(y_true, y_score))


def zero_one_loss(y_true: Any, y_pred: Any) -> float:
    """Computes 0-1 Classification Loss (fraction of misclassifications)."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    return float(np.mean(yt != yp))


def hamming_loss(y_true: Any, y_pred: Any) -> float:
    """Computes Hamming Loss for multilabel classification."""
    yt = np.array(y_true)
    yp = np.array(y_pred)
    return float(np.mean(yt != yp))


def jaccard_score(y_true: Any, y_pred: Any, average: str = "binary") -> float:
    """Computes Jaccard Similarity Score (IoU)."""
    from sklearn.metrics import jaccard_score as sk_jaccard
    return float(sk_jaccard(y_true, y_pred, average=average))


def multilabel_confusion_matrix(y_true: Any, y_pred: Any) -> np.ndarray:
    """Computes confusion matrix for each class or multi-label."""
    from sklearn.metrics import multilabel_confusion_matrix as sk_mcm
    return sk_mcm(y_true, y_pred)


def brier_score_loss(y_true: Any, y_prob: Any) -> float:
    """Computes Brier Score Loss for probability calibration."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_prob, dtype=float)
    return float(np.mean((yt - yp) ** 2))
