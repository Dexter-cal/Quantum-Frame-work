from __future__ import annotations
import numpy as np
from .base import Technique


class DecisionTree(Technique):
    name = "decision_tree"
    family = "classical_ml"

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier(**params)
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
    def feature_importance(self):
        return dict(enumerate(self.model.feature_importances_.tolist()))

    def tree_depth(self) -> int:
        return self.model.get_depth()

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
