from __future__ import annotations
import numpy as np
from .base import Technique

class MultinomialNaiveBayes(Technique):
    """Multinomial Naive Bayes technique for discrete count features."""
    name = "multinomial_naive_bayes"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, **params):
        super().__init__(alpha=alpha, **params)
        from sklearn.naive_bayes import MultinomialNB
        self.model = MultinomialNB(alpha=alpha, **params)
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if (X < 0).any():
            raise ValueError("MultinomialNaiveBayes requires non-negative feature counts (X >= 0)")
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, np.asarray(y)
        return self

    def forward(self, x):
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        res = self.model.predict(x)
        return res[0] if single else res

    def class_log_prior(self):
        return self.model.class_log_prior_

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
