from __future__ import annotations
import numpy as np
from .base import Technique


class KNearestNeighbors(Technique):
    name = "knn"
    family = "classical_ml"

    def __init__(self, k=5, **params):
        super().__init__(k=k, **params)
        from sklearn.neighbors import KNeighborsClassifier
        self.model = KNeighborsClassifier(n_neighbors=k, **params)
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, np.asarray(y)
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        result = self.model.predict(x)
        return result[0] if single else result

    # --- unique tools -----------------------------------------------------
    def nearest_neighbors(self, x, n=None):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        distances, indices = self.model.kneighbors(x, n_neighbors=n or self.params["k"])
        return list(zip(indices[0].tolist(), distances[0].tolist()))

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
