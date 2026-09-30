from __future__ import annotations
import numpy as np
from .base import Technique


class Perceptron(Technique):
    """The original 1958 single-layer neural unit (Section 32, historical/
    foundational techniques) -- genuinely runnable, same interface as
    everything modern, so the design doc's teaching claim is testable."""
    name = "perceptron"
    family = "historical"

    def __init__(self, max_iter=1000, **params):
        super().__init__(max_iter=max_iter, **params)
        from sklearn.linear_model import Perceptron as SKPerceptron
        self.model = SKPerceptron(max_iter=max_iter, **params)
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
    def n_updates(self) -> int:
        """How many training iterations the weight-update rule actually ran."""
        return int(self.model.n_iter_)

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
