from __future__ import annotations
import numpy as np
from .base import Technique


class KMeans(Technique):
    """Unsupervised -- fit(X, y=None). Tests that the interface handles a
    technique with no labels at all, a real, distinct case from Section 3
    (objective() being generic, not every technique uses loss/labels)."""
    name = "kmeans"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_clusters=3, **params):
        super().__init__(n_clusters=n_clusters, **params)
        from sklearn.cluster import KMeans as SKKMeans
        self.model = SKKMeans(n_clusters=n_clusters, n_init=10, **params)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        result = self.model.predict(x)
        return int(result[0]) if single else result

    # --- unique tools -----------------------------------------------------
    def cluster_centers(self):
        return self.model.cluster_centers_

    def elbow_plot_data(self, X, k_range=range(1, 10)):
        from sklearn.cluster import KMeans as SKKMeans
        inertias = []
        for k in k_range:
            m = SKKMeans(n_clusters=k, n_init=10).fit(np.asarray(X, dtype=float))
            inertias.append(m.inertia_)
        return list(k_range), inertias
