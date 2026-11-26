from __future__ import annotations
import numpy as np
from .base import Technique

class SpectralClustering(Technique):
    """Spectral Clustering technique for non-convex cluster structures."""
    name = "spectral_clustering"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_clusters=2, **params):
        super().__init__(n_clusters=n_clusters, **params)
        from sklearn.cluster import SpectralClustering as SKSpectral
        self.model = SKSpectral(n_clusters=n_clusters, **params)
        self.labels_ = None
        self._last_X = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.labels_ = self.model.fit_predict(X)
        self._last_X = X
        self._trained = True
        return self

    def forward(self, x):
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


class FastICA(Technique):
    """Fast Independent Component Analysis (ICA) for blind source separation."""
    name = "fast_ica"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, **params):
        super().__init__(n_components=n_components, **params)
        from sklearn.decomposition import FastICA as SKICA
        self.model = SKICA(n_components=n_components, **params)

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


class Isomap(Technique):
    """Isometric Feature Mapping (Isomap) manifold learning."""
    name = "isomap"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, n_neighbors=5, **params):
        super().__init__(n_components=n_components, n_neighbors=n_neighbors, **params)
        from sklearn.manifold import Isomap as SKIsomap
        self.model = SKIsomap(n_components=n_components, n_neighbors=n_neighbors, **params)

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


class LocallyLinearEmbedding(Technique):
    """Locally Linear Embedding (LLE) manifold learning."""
    name = "lle"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, n_neighbors=5, **params):
        super().__init__(n_components=n_components, n_neighbors=n_neighbors, **params)
        from sklearn.manifold import LocallyLinearEmbedding as SKLLE
        self.model = SKLLE(n_components=n_components, n_neighbors=n_neighbors, **params)

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
