from __future__ import annotations
import numpy as np
from .base import Technique

class DBSCAN(Technique):
    """Unsupervised density-based spatial clustering of applications with noise (DBSCAN).
    Fits cluster labels based on density parameters eps and min_samples."""
    name = "dbscan"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, eps=0.5, min_samples=5, **params):
        super().__init__(eps=eps, min_samples=min_samples, **params)
        from sklearn.cluster import DBSCAN as SKDBSCAN
        self.model = SKDBSCAN(eps=eps, min_samples=min_samples, **params)
        self.labels_ = None
        self.components_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.components_ = self.model.components_
        self._trained = True
        return self

    def forward(self, x):
        """DBSCAN assigns cluster membership based on nearest core point distance."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        if len(self.components_) == 0:
            # All noise
            res = np.full(x.shape[0], -1)
        else:
            # Nearest core sample assignment
            dists = np.linalg.norm(x[:, np.newaxis, :] - self.components_[np.newaxis, :, :], axis=2)
            min_idx = np.argmin(dists, axis=1)
            min_dist = dists[np.arange(len(x)), min_idx]
            core_labels = self.labels_[self.model.core_sample_indices_]
            res = np.where(min_dist <= self.params["eps"], core_labels[min_idx], -1)

        return int(res[0]) if single else res

    # --- unique tools ---
    def n_clusters(self):
        if self.labels_ is None:
            return 0
        return len(set(self.labels_) - {-1})

    def noise_ratio(self):
        if self.labels_ is None or len(self.labels_) == 0:
            return 0.0
        return float(np.sum(self.labels_ == -1) / len(self.labels_))
