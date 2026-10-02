from __future__ import annotations
import numpy as np
from .base import Technique

class BayesianRidge(Technique):
    name = "bayesian_ridge"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import BayesianRidge as SKBayesRidge
        self.model = SKBayesRidge(**params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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


class ARDRegression(Technique):
    name = "ard_regression"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import ARDRegression as SKARD
        self.model = SKARD(**params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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


class HuberRegressor(Technique):
    name = "huber"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import HuberRegressor as SKHuber
        self.model = SKHuber(**params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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


class RANSACRegressor(Technique):
    name = "ransac"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import RANSACRegressor as SKRANSAC
        self.model = SKRANSAC(**params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
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


class TheilSenRegressor(Technique):
    name = "theil_sen"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.linear_model import TheilSenRegressor as SKTheilSen
        self.model = SKTheilSen(**params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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


class QuantileRegressor(Technique):
    name = "quantile_regression"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, quantile=0.5, **params):
        super().__init__(quantile=quantile, **params)
        from sklearn.linear_model import QuantileRegressor as SKQuantile
        self.model = SKQuantile(quantile=quantile, **params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
        self.w = self.model.coef_
        self.b = float(self.model.intercept_)
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


class DecisionTreeRegressor(Technique):
    name = "decision_tree_regressor"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, max_depth=None, **params):
        super().__init__(max_depth=max_depth, **params)
        from sklearn.tree import DecisionTreeRegressor as SKTreeReg
        self.model = SKTreeReg(max_depth=max_depth, **params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
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


class RandomForestRegressor(Technique):
    name = "random_forest_regressor"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=100, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import RandomForestRegressor as SKRFReg
        self.model = SKRFReg(n_estimators=n_estimators, **params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
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


class AdaBoostRegressor(Technique):
    name = "adaboost_regressor"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=50, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import AdaBoostRegressor as SKAdaReg
        self.model = SKAdaReg(n_estimators=n_estimators, **params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
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


class GradientBoostingRegressor(Technique):
    name = "gradient_boosting_regressor"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=100, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import GradientBoostingRegressor as SKGBReg
        self.model = SKGBReg(n_estimators=n_estimators, **params)

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.model.fit(X, y)
        self._trained = True
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
