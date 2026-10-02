from __future__ import annotations
import numpy as np
from .base import Technique

class LDA(Technique):
    """Linear Discriminant Analysis."""
    name = "lda"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        self.model = LinearDiscriminantAnalysis(**params)
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


class QDA(Technique):
    """Quadratic Discriminant Analysis."""
    name = "qda"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
        self.model = QuadraticDiscriminantAnalysis(**params)
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


class AdaBoost(Technique):
    """AdaBoost Ensemble Classifier."""
    name = "adaboost"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=50, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import AdaBoostClassifier
        self.model = AdaBoostClassifier(n_estimators=n_estimators, **params)
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


class GradientBoosting(Technique):
    """Gradient Boosting Classifier."""
    name = "gradient_boosting"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=100, learning_rate=0.1, **params):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **params)
        from sklearn.ensemble import GradientBoostingClassifier
        self.model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, **params)
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


class ExtraTrees(Technique):
    """Extra Trees Ensemble Classifier."""
    name = "extra_trees"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=100, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import ExtraTreesClassifier
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, **params)
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


class RidgeRegression(Technique):
    """Ridge Linear Regression with L2 Regularization."""
    name = "ridge"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, **params):
        super().__init__(alpha=alpha, **params)
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=alpha, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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
        return float(res[0]) if single else res


class LassoRegression(Technique):
    """Lasso Linear Regression with L1 Regularization."""
    name = "lasso"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, **params):
        super().__init__(alpha=alpha, **params)
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha=alpha, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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
        return float(res[0]) if single else res


class ElasticNet(Technique):
    """ElasticNet Linear Regression with L1 & L2 Regularization."""
    name = "elastic_net"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, l1_ratio=0.5, **params):
        super().__init__(alpha=alpha, l1_ratio=l1_ratio, **params)
        from sklearn.linear_model import ElasticNet as SKElasticNet
        self.model = SKElasticNet(alpha=alpha, l1_ratio=l1_ratio, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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
        return float(res[0]) if single else res


class KernelRidge(Technique):
    """Kernel Ridge Regression."""
    name = "kernel_ridge"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, alpha=1.0, kernel="linear", **params):
        super().__init__(alpha=alpha, kernel=kernel, **params)
        from sklearn.kernel_ridge import KernelRidge as SKKernelRidge
        self.model = SKKernelRidge(alpha=alpha, kernel=kernel, **params)
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
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
        return float(res[0]) if single else res
