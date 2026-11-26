from __future__ import annotations
import numpy as np
from .base import Technique

class TimeSeriesForecaster(Technique):
    """Autoregressive time-series forecasting technique using lag features."""
    name = "time_series"
    family = "classical_ml"
    compatible_learning_techniques = ("supervised",)

    def __init__(self, lags=3, horizon=1, **params):
        super().__init__(lags=lags, horizon=horizon, **params)
        from sklearn.linear_model import Ridge
        self.lags = lags
        self.horizon = horizon
        self.model = Ridge(**params)
        self.series_ = None

    def _create_lag_features(self, series):
        X, y = [], []
        for i in range(len(series) - self.lags - self.horizon + 1):
            X.append(series[i:i + self.lags])
            y.append(series[i + self.lags:i + self.lags + self.horizon])
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        if self.horizon == 1:
            y = y.ravel()
        return X, y

    def fit(self, X, y=None):
        series = np.asarray(X, dtype=float).ravel()
        if len(series) <= self.lags + self.horizon:
            raise ValueError(f"Series length ({len(series)}) must be > lags ({self.lags}) + horizon ({self.horizon})")

        X_lags, y_lags = self._create_lag_features(series)
        self.model.fit(X_lags, y_lags)
        self.series_ = series
        self._trained = True
        return self

    def forward(self, x=None, steps=None):
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict/forward.")

        if x is None or (isinstance(x, (list, np.ndarray)) and len(x) == 0):
            recent_lags = self.series_[-self.lags:]
        else:
            recent_lags = np.asarray(x, dtype=float).ravel()[-self.lags:]

        if len(recent_lags) < self.lags:
            raise ValueError(f"Input requires at least {self.lags} lag values.")

        n_steps = steps or self.horizon
        predictions = []
        current_window = list(recent_lags)

        for _ in range(n_steps):
            input_arr = np.array(current_window[-self.lags:]).reshape(1, -1)
            pred = self.model.predict(input_arr)
            next_val = float(pred.ravel()[0])
            predictions.append(next_val)
            current_window.append(next_val)

        res = np.array(predictions)
        return float(res[0]) if n_steps == 1 else res

    # --- unique tools ---
    def forecast(self, steps=5):
        """Generate multi-step future predictions from the end of the training series."""
        return self.forward(steps=steps)

    def residuals(self):
        if not self._trained:
            raise RuntimeError("residuals() requires a fitted model.")
        X_lags, y_lags = self._create_lag_features(self.series_)
        preds = self.model.predict(X_lags)
        return y_lags - preds

    def accuracy(self) -> float:
        if not self._trained or self.series_ is None:
            return 0.0
        X_lags, y_lags = self._create_lag_features(self.series_)
        preds = self.model.predict(X_lags)
        mape = np.mean(np.abs((y_lags - preds) / (np.abs(y_lags) + 1e-8)))
        return float(max(0.0, 1.0 - mape))
