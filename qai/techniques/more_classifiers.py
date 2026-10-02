from __future__ import annotations
import numpy as np
from .base import Technique

class BernoulliNaiveBayes(Technique):
    name = "bernoulli_naive_bayes"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, binarize=0.0, **params):
        super().__init__(alpha=alpha, binarize=binarize, **params)
        from sklearn.naive_bayes import BernoulliNB
        self.model = BernoulliNB(alpha=alpha, binarize=binarize, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class ComplementNaiveBayes(Technique):
    name = "complement_naive_bayes"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, **params):
        super().__init__(alpha=alpha, **params)
        from sklearn.naive_bayes import ComplementNB
        self.model = ComplementNB(alpha=alpha, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        if (X < 0).any():
            raise ValueError("ComplementNaiveBayes requires non-negative features (X >= 0)")
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class SGDClassifier(Technique):
    name = "sgd_classifier"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, loss="hinge", alpha=0.0001, max_iter=1000, **params):
        super().__init__(loss=loss, alpha=alpha, max_iter=max_iter, **params)
        from sklearn.linear_model import SGDClassifier as SKSGD
        self.model = SKSGD(loss=loss, alpha=alpha, max_iter=max_iter, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_[0]) if len(self.model.intercept_) > 0 else 0.0
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class PassiveAggressiveClassifier(Technique):
    name = "passive_aggressive_classifier"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, C=1.0, max_iter=1000, **params):
        super().__init__(C=C, max_iter=max_iter, **params)
        from sklearn.linear_model import PassiveAggressiveClassifier as SKPA
        self.model = SKPA(C=C, max_iter=max_iter, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class LinearSVC(Technique):
    name = "linear_svc"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, C=1.0, max_iter=1000, **params):
        super().__init__(C=C, max_iter=max_iter, **params)
        from sklearn.svm import LinearSVC as SKLinearSVC
        self.model = SKLinearSVC(C=C, max_iter=max_iter, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_[0]) if len(self.model.intercept_) > 0 else 0.0
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class NuSVC(Technique):
    name = "nu_svc"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, nu=0.5, kernel="rbf", **params):
        super().__init__(nu=nu, kernel=kernel, **params)
        from sklearn.svm import NuSVC as SKNuSVC
        self.model = SKNuSVC(nu=nu, kernel=kernel, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class RadiusNeighborsClassifier(Technique):
    name = "radius_neighbors_classifier"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, radius=1.0, **params):
        super().__init__(radius=radius, **params)
        from sklearn.neighbors import RadiusNeighborsClassifier as SKRNC
        self.model = SKRNC(radius=radius, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class NearestCentroid(Technique):
    name = "nearest_centroid"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.neighbors import NearestCentroid as SKNC
        self.model = SKNC(**params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class BaggingClassifier(Technique):
    name = "bagging_classifier"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=10, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import BaggingClassifier as SKBagging
        self.model = SKBagging(n_estimators=n_estimators, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))


class HistGradientBoostingClassifier(Technique):
    name = "hist_gradient_boosting"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, max_iter=100, **params):
        super().__init__(max_iter=max_iter, **params)
        from sklearn.ensemble import HistGradientBoostingClassifier as SKHGB
        self.model = SKHGB(max_iter=max_iter, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, y
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

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
