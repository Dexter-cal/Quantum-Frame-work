from __future__ import annotations
import numpy as np
from .base import Technique

class GaussianMixtureModel(Technique):
    """Unsupervised Gaussian Mixture Model (GMM) probabilistic clustering."""
    name = "gmm"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=3, covariance_type="full", **params):
        super().__init__(n_components=n_components, covariance_type=covariance_type, **params)
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(n_components=n_components, covariance_type=covariance_type, **params)

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
        res = self.model.predict(x)
        return int(res[0]) if single else res

    # --- unique tools ---
    def predict_proba(self, x):
        """Return component posterior probabilities for x."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict_proba.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.predict_proba(x)

    def score_samples(self, x):
        """Compute the log-likelihood of each sample under the model."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before score_samples.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.score_samples(x)

    def means(self):
        return self.model.means_

    def covariances(self):
        return self.model.covariances_
