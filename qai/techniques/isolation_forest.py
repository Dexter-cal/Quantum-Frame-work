from __future__ import annotations
import numpy as np
from .base import Technique

class IsolationForest(Technique):
    """Unsupervised Isolation Forest for anomaly and outlier detection."""
    name = "isolation_forest"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_estimators=100, contamination=0.1, **params):
        super().__init__(n_estimators=n_estimators, contamination=contamination, **params)
        from sklearn.ensemble import IsolationForest as SKIsolationForest
        self.model = SKIsolationForest(n_estimators=n_estimators, contamination=contamination, **params)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
        self._trained = True
        return self

    def forward(self, x):
        """Returns 1 for inliers (normal) and -1 for outliers (anomalies)."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        res = self.model.predict(x)
        return int(res[0]) if single else res

    # --- unique tools ---
    def anomaly_score(self, x):
        """Compute average anomaly score (more negative = more anomalous)."""
        if not self._trained:
            raise RuntimeError("anomaly_score() requires a fitted model.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        scores = self.model.score_samples(x)
        return float(scores[0]) if x.shape[0] == 1 else scores

    def is_anomaly(self, x):
        """Returns True if sample is classified as an anomaly (-1)."""
        pred = self.forward(x)
        if isinstance(pred, (int, np.integer)):
            return pred == -1
        return pred == -1
