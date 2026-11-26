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
    """Gradient Boosting Classifier -- with optional modern training upgrades.

    Builds ON TOP of the existing sklearn GradientBoostingClassifier. New
    optional parameters activate modern training improvements on the SAME
    `gradient_boosting` technique:

      early_stopping_rounds=0  -- if > 0, enable sklearn's n_iter_no_change
                                  early stopping (saves estimators + time).
      validation_fraction=0.1  -- fraction of train set held out for early
                                  stopping (only used when early_stopping_rounds>0).
      snapshot_ensemble=False  -- if True, train `n_snapshots` models with
                                  different random seeds and average their
                                  predictions (Huang et al. 2017 style).
      n_snapshots=3            -- number of snapshots when snapshot_ensemble=True.
      calibrated=False         -- if True, wrap the trained classifier with
                                  CalibratedClassifierCV (isotonic / sigmoid).
      calibration_method="sigmoid"  -- "sigmoid" or "isotonic".
    """
    name = "gradient_boosting"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, n_estimators=100, learning_rate=0.1,
                 early_stopping_rounds: int = 0,
                 validation_fraction: float = 0.1,
                 snapshot_ensemble: bool = False,
                 n_snapshots: int = 3,
                 calibrated: bool = False,
                 calibration_method: str = "sigmoid",
                 **params):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate, **params)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.snapshot_ensemble = snapshot_ensemble
        self.n_snapshots = n_snapshots
        self.calibrated = calibrated
        self.calibration_method = calibration_method

        from sklearn.ensemble import GradientBoostingClassifier
        kwargs = dict(n_estimators=n_estimators, learning_rate=learning_rate, **params)
        if early_stopping_rounds > 0:
            kwargs["n_iter_no_change"] = early_stopping_rounds
            kwargs["validation_fraction"] = validation_fraction
        self.model = GradientBoostingClassifier(**kwargs)
        # Container for snapshot-ensemble members (built on the same model type).
        self._snapshots: list = []
        self._calibrated_wrapped = None
        self._last_X, self._last_y = None, None

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.model.fit(X, y)
        self._trained = True
        # Build extra snapshots on the SAME technique (different seeds).
        if self.snapshot_ensemble:
            from sklearn.ensemble import GradientBoostingClassifier
            self._snapshots = []
            for i in range(self.n_snapshots):
                snap = GradientBoostingClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    random_state=i * 13 + 7,
                )
                snap.fit(X, y)
                self._snapshots.append(snap)
        # Optional probability calibration (built on top of the trained model).
        if self.calibrated:
            from sklearn.calibration import CalibratedClassifierCV
            try:
                self._calibrated_wrapped = CalibratedClassifierCV(
                    self.model, method=self.calibration_method, cv=3
                )
                self._calibrated_wrapped.fit(X, y)
            except Exception:
                self._calibrated_wrapped = None
        self._last_X, self._last_y = X, y
        return self

    def forward(self, x):
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        # Calibrated wrapper takes precedence if available.
        if self._calibrated_wrapped is not None:
            res = self._calibrated_wrapped.predict(x)
            return res[0] if single else res
        # Snapshot ensemble averages votes from all snapshots.
        if self._snapshots:
            votes = [snap.predict(x) for snap in self._snapshots]
            votes.append(self.model.predict(x))
            stacked = np.stack(votes, axis=0)
            # mode along axis 0 (majority vote)
            from scipy import stats
            res = stats.mode(stacked, axis=0, keepdims=False).mode
            return res[0] if single else res
        res = self.model.predict(x)
        return res[0] if single else res

    def n_estimators_used(self) -> int:
        """How many estimators were actually used (after early stopping)."""
        return int(getattr(self.model, "n_estimators_", self.n_estimators))

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
        X = np.asarray(X, dtype=float)
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "[WHAT] regressor cannot fit string/categorical targets\n"
                "[WHY] regressors require numeric y. Got non-numeric labels: "
                f"{np.unique(y)[:5]}\n"
                "[FIX] use a classifier technique (qai.build(type='random_forest')) "
                "or encode labels to int first"
            ) from e
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
        X = np.asarray(X, dtype=float)
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "[WHAT] regressor cannot fit string/categorical targets\n"
                "[WHY] regressors require numeric y. Got non-numeric labels: "
                f"{np.unique(y)[:5]}\n"
                "[FIX] use a classifier technique (qai.build(type='random_forest')) "
                "or encode labels to int first"
            ) from e
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
        X = np.asarray(X, dtype=float)
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "[WHAT] regressor cannot fit string/categorical targets\n"
                "[WHY] regressors require numeric y. Got non-numeric labels: "
                f"{np.unique(y)[:5]}\n"
                "[FIX] use a classifier technique (qai.build(type='random_forest')) "
                "or encode labels to int first"
            ) from e
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
        X = np.asarray(X, dtype=float)
        try:
            y = np.asarray(y, dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError(
                "[WHAT] regressor cannot fit string/categorical targets\n"
                "[WHY] regressors require numeric y. Got non-numeric labels: "
                f"{np.unique(y)[:5]}\n"
                "[FIX] use a classifier technique (qai.build(type='random_forest')) "
                "or encode labels to int first"
            ) from e
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
