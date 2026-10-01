from __future__ import annotations
import numpy as np
from .base import Technique

class HierarchicalClustering(Technique):
    """Unsupervised Agglomerative Hierarchical Clustering."""
    name = "hierarchical"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_clusters=2, linkage="ward", **params):
        super().__init__(n_clusters=n_clusters, linkage=linkage, **params)
        from sklearn.cluster import AgglomerativeClustering
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, **params)
        self.labels_ = None
        self._last_X = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.labels_ = self.model.fit_predict(X)
        self._last_X = X
        self._trained = True
        return self

    def forward(self, x):
        """Hierarchical clustering assigns new samples to the nearest cluster centroid."""
        if not self._trained or self._last_X is None:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        centroids = []
        unique_labels = sorted(list(set(self.labels_)))
        for lbl in unique_labels:
            centroids.append(np.mean(self._last_X[self.labels_ == lbl], axis=0))
        centroids = np.array(centroids)

        dists = np.linalg.norm(x[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        assigned = np.array([unique_labels[idx] for idx in nearest_idx])

        return int(assigned[0]) if single else assigned

    # --- unique tools ---
    def n_leaves(self):
        return getattr(self.model, "n_leaves_", len(self._last_X) if self._last_X is not None else 0)

    def cluster_counts(self):
        if self.labels_ is None:
            return {}
        unique, counts = np.unique(self.labels_, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
