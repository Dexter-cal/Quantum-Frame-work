"""
qai.techniques.classifier — a Type built on scikit-learn's optimizer, which
is exactly the point of a Python prototype: gradient-based optimization is
inherited for free instead of hand-implemented (Section 99's autodiff point).
"""
from __future__ import annotations
import numpy as np
from .base import Technique


class Classifier(Technique):
    """Logistic-regression-style classifier."""

    name = "classifier"
    family = "classical_ml"

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(max_iter=1000, **params)
        self.classes_ = None
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
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

    # --- unique tools, only Classifier has these --------------------------
    def class_probabilities(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self.model.predict_proba(x)
        return {cls: float(p) for cls, p in zip(self.classes_, probs[0])}

    def confusion_matrix(self):
        from sklearn.metrics import confusion_matrix
        preds = self.forward(self._last_X)
        return confusion_matrix(self._last_y, preds)

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
