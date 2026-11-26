"""
qai.confidence_calibrator — calibrate prediction confidence.

Production models often have poorly calibrated probabilities (overconfident
or underconfident). This module applies Platt scaling (logistic) or
isotonic regression to fix the calibration.

Usage:
    import qai
    m = qai.build(technique="random_forest", n_estimators=50)
    m.train(X_train, y_train)

    calibrator = qai.ConfidenceCalibrator(method="isotonic")
    calibrator.fit(m, X_cal, y_cal)  # use a calibration set

    # Now get calibrated probabilities
    probs = calibrator.predict_proba(m, X_test)
    print(probs)  # well-calibrated probabilities

    # Check calibration quality
    score = calibrator.score(X_test, y_test)
    print(f"Brier score: {score['brier']:.4f}, ECE: {score['ece']:.4f}")
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import numpy as np


class ConfidenceCalibrator:
    """Calibrate model prediction confidence.

    Args:
        method: "platt" (logistic regression) or "isotonic" (isotonic regression).
    """
    def __init__(self, method: str = "isotonic"):
        if method not in ("platt", "isotonic"):
            raise ValueError("[WHAT] method must be 'platt' or 'isotonic'")
        self.method = method
        self._calibrator = None
        self._fitted = False

    def fit(self, model, X_cal, y_cal) -> "ConfidenceCalibrator":
        """Fit the calibrator on a calibration set.

        Args:
            model: a trained qai classifier with predict_proba.
            X_cal: calibration inputs.
            y_cal: calibration labels (binary 0/1).
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression

        probs = self._get_probs(model, X_cal)
        y_cal = np.asarray(y_cal)

        if probs.ndim > 1 and probs.shape[1] > 1:
            # Multi-class: use the max probability
            probs = probs.max(axis=1)
            y_cal = (y_cal == probs.argmax(axis=1) if probs.ndim > 1 else y_cal).astype(int)

        if self.method == "platt":
            self._calibrator = LogisticRegression()
            self._calibrator.fit(probs.reshape(-1, 1), y_cal)
        else:
            self._calibrator = IsotonicRegression(out_of_bounds="clip")
            self._calibrator.fit(probs, y_cal)

        self._fitted = True
        return self

    def predict_proba(self, model, X) -> np.ndarray:
        """Get calibrated probabilities.

        Args:
            model: the trained model.
            X: inputs.

        Returns:
            Calibrated probability array (n_samples,) for binary,
            or (n_samples, n_classes) for multi-class.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first")
        probs = self._get_probs(model, X)
        if probs.ndim > 1 and probs.shape[1] > 1:
            # Multi-class: calibrate the max probability, redistribute
            max_probs = probs.max(axis=1)
            if self.method == "platt":
                calibrated = self._calibrator.predict_proba(max_probs.reshape(-1, 1))[:, 1]
            else:
                calibrated = self._calibrator.predict(max_probs)
            # Rescale all probabilities
            scale = calibrated / np.clip(max_probs, 1e-8, 1.0)
            return probs * scale[:, np.newaxis]
        else:
            if self.method == "platt":
                return self._calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
            return self._calibrator.predict(probs)

    def score(self, model, X, y) -> Dict[str, float]:
        """Evaluate calibration quality.

        Returns:
            Dict with: brier (Brier score, lower=better), ece (Expected Calibration Error, lower=better).
        """
        y = np.asarray(y)
        probs = self.predict_proba(model, X)

        if probs.ndim > 1:
            # Multi-class: use max prob
            preds = probs.argmax(axis=1)
            confidences = probs.max(axis=1)
        else:
            preds = (probs > 0.5).astype(int)
            confidences = probs

        # Brier score
        if probs.ndim > 1:
            onehot = np.zeros_like(probs)
            y_int = np.asarray(y, dtype=int)
            onehot[np.arange(len(y_int)), y_int] = 1
            brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
        else:
            brier = float(np.mean((probs - y) ** 2))

        # Expected Calibration Error (10 bins)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_acc = float(np.mean(preds[mask] == y[mask]))
                bin_conf = float(np.mean(confidences[mask]))
                ece += abs(bin_acc - bin_conf) * mask.sum() / len(y)

        return {"brier": brier, "ece": ece}

    @staticmethod
    def _get_probs(model, X) -> np.ndarray:
        """Get raw probabilities from the model."""
        tech = getattr(model, "technique", None)
        if tech is not None and hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
            return tech.model.predict_proba(X)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        # Fallback: convert predictions to pseudo-probabilities
        preds = np.asarray(model.predict(X))
        probs = np.zeros((len(preds), 2))
        probs[np.arange(len(preds)), preds.astype(int)] = 0.9
        probs[probs == 0] = 0.1
        return probs


__all__ = ["ConfidenceCalibrator"]
