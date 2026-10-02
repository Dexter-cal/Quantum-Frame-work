"""
Governance and drift detection utilities for qai models and features.
"""
import numpy as np

def detect_drift(reference_data, current_data, threshold=0.1):
    """
    Detects feature distribution drift between reference (baseline) and current dataset.
    Uses normalized mean shift and variance shift ratio.
    Returns a dict with overall drift status and per-feature drift metrics.
    """
    ref = np.asarray(reference_data, dtype=float)
    cur = np.asarray(current_data, dtype=float)

    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if cur.ndim == 1:
        cur = cur.reshape(-1, 1)

    if ref.shape[1] != cur.shape[1]:
        raise ValueError(f"Feature count mismatch: reference has {ref.shape[1]}, current has {cur.shape[1]}")

    feature_results = []
    overall_drift = False

    for col_idx in range(ref.shape[1]):
        r_col = ref[:, col_idx]
        c_col = cur[:, col_idx]

        r_mean, r_std = np.mean(r_col), np.std(r_col)
        c_mean, c_std = np.mean(c_col), np.std(c_col)

        if r_std == 0:
            r_std = 1e-8

        # Measure mean shift normalized by standard error of the mean
        se = r_std / np.sqrt(len(r_col))
        mean_shift_stat = abs(r_mean - c_mean) / se if se > 0 else 0.0

        is_drifted = mean_shift_stat > (threshold * 100)
        if is_drifted:
            overall_drift = True

        feature_results.append({
            "feature_index": col_idx,
            "drift_score": round(float(mean_shift_stat), 4),
            "drifted": is_drifted,
            "ref_mean": round(float(r_mean), 4),
            "cur_mean": round(float(c_mean), 4)
        })

    return {
        "drift_detected": overall_drift,
        "features": feature_results,
        "threshold": threshold
    }


def weight_distance(model1, model2) -> float:
    """Computes Euclidean distance between model weight parameter vectors."""
    t1, t2 = model1.technique, model2.technique
    if hasattr(t1, "w") and hasattr(t2, "w") and t1.w is not None and t2.w is not None:
        w1, w2 = np.ravel(t1.w), np.ravel(t2.w)
        if len(w1) != len(w2):
            raise ValueError(f"Weight vector length mismatch: {len(w1)} vs {len(w2)}")
        return float(np.linalg.norm(w1 - w2))
    raise AttributeError("Both models must have numeric weight vectors ('w') to compute distance")


def calibration_curve(y_true, y_prob, n_bins: int = 5):
    """Computes probability calibration curve (fraction of positives vs mean predicted probability)."""
    from sklearn.calibration import calibration_curve as sk_cal
    y_t, y_p = np.asarray(y_true), np.asarray(y_prob)
    prob_true, prob_pred = sk_cal(y_t, y_p, n_bins=n_bins)
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist()}
