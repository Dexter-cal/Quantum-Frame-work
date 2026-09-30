from __future__ import annotations
import numpy as np
from .base import Technique


class SVM(Technique):
    name = "svm"
    family = "classical_ml"

    def __init__(self, kernel="rbf", **params):
        super().__init__(kernel=kernel, **params)
        from sklearn.svm import SVC
        self.model = SVC(kernel=kernel, probability=True, **params)
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
    def support_vectors(self):
        return self.model.support_vectors_

    def margin_width(self) -> float:
        # for a linear kernel, ||w|| relates directly to margin = 2/||w||;
        # for non-linear kernels this is reported via the decision function
        # spread as an approximation, which is the honest thing to expose
        if self.model.kernel == "linear":
            w = self.model.coef_[0]
            return float(2 / np.linalg.norm(w))
        raise NotImplementedError("margin_width() is exact only for a linear kernel")

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
