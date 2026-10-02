from __future__ import annotations
import numpy as np
from .base import Technique

class TruncatedSVD(Technique):
    """Unsupervised Truncated Singular Value Decomposition for LSA/dimensionality reduction."""
    name = "truncated_svd"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, **params):
        super().__init__(n_components=n_components, **params)
        from sklearn.decomposition import TruncatedSVD as SKTruncatedSVD
        self.model = SKTruncatedSVD(n_components=n_components, **params)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
        self._trained = True
        return self

    def forward(self, x):
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        res = self.model.transform(x)
        return res[0] if single else res

    def explained_variance_ratio(self):
        return self.model.explained_variance_ratio_
