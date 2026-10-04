"""
Data preprocessing, scaling, encoding, missing-value handling, and automated cleaning utilities for qai.
"""
import numpy as np
from typing import Tuple, Optional, Any

class StandardScaler:
    """Scales features by removing the mean and scaling to unit variance."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transforming data.")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class LabelEncoder:
    """Encodes categorical labels into numeric integers and decodes them back."""
    def __init__(self):
        self.classes_ = None
        self._mapping = {}
        self._inverse = {}

    def fit(self, y):
        unique_labels = sorted(list(set(y)))
        self.classes_ = unique_labels
        self._mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self._inverse = {idx: label for idx, label in enumerate(unique_labels)}
        return self

    def transform(self, y):
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder must be fitted before transforming labels.")
        return np.array([self._mapping[item] for item in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y_encoded):
        if self.classes_ is None:
            raise RuntimeError("LabelEncoder must be fitted before inverse transforming.")
        return [self._inverse[idx] for idx in y_encoded]


class SimpleImputer:
    """Imputes missing values (np.nan) with column mean or median strategies."""
    def __init__(self, strategy="mean"):
        if strategy not in ("mean", "median"):
            raise ValueError("strategy must be either 'mean' or 'median'")
        self.strategy = strategy
        self.fill_values_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        fill_vals = []
        for col_idx in range(X.shape[1]):
            col = X[:, col_idx]
            valid_vals = col[~np.isnan(col)]
            if len(valid_vals) == 0:
                fill_vals.append(0.0)
            elif self.strategy == "mean":
                fill_vals.append(np.mean(valid_vals))
            else:
                fill_vals.append(np.median(valid_vals))
        self.fill_values_ = np.array(fill_vals)
        return self

    def transform(self, X):
        X = np.array(X, dtype=float, copy=True)
        if self.fill_values_ is None:
            raise RuntimeError("SimpleImputer must be fitted before transform.")
        for col_idx in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col_idx])
            X[nan_mask, col_idx] = self.fill_values_[col_idx]
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def clean_dataset(X, y=None, impute_strategy="mean", scale=True):
    """Automated dataset cleaning: handles missing values, encodes labels, and scales features in one call."""
    if not isinstance(X, (list, np.ndarray)):
        raise TypeError("X must be a list or numpy array")

    X_arr = np.asarray(X, dtype=float)
    if np.isnan(X_arr).any():
        imputer = SimpleImputer(strategy=impute_strategy)
        X_arr = imputer.fit_transform(X_arr)

    if scale:
        scaler = StandardScaler()
        X_arr = scaler.fit_transform(X_arr)

    y_arr = None
    if y is not None:
        if isinstance(y[0], str):
            encoder = LabelEncoder()
            y_arr = encoder.fit_transform(y)
        else:
            y_arr = np.asarray(y)

    return (X_arr, y_arr) if y is not None else X_arr


class PolynomialFeatures:
    """Generates polynomial and interaction feature combinations."""
    def __init__(self, degree=2, include_bias=False):
        from sklearn.preprocessing import PolynomialFeatures as SKPoly
        self.degree = degree
        self.include_bias = include_bias
        self.model = SKPoly(degree=degree, include_bias=include_bias)

    def fit_transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return self.model.fit_transform(X_arr)

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return self.model.transform(X_arr)


class MinMaxScaler:
    """Transforms features by scaling each feature to a given range [0, 1]."""
    def __init__(self, feature_range=(0, 1)):
        from sklearn.preprocessing import MinMaxScaler as SKMinMax
        self.model = SKMinMax(feature_range=feature_range)

    def fit_transform(self, X):
        return self.model.fit_transform(np.asarray(X, dtype=float))

    def transform(self, X):
        return self.model.transform(np.asarray(X, dtype=float))


class RobustScaler:
    """Scales features using statistics that are robust to outliers."""
    def __init__(self):
        from sklearn.preprocessing import RobustScaler as SKRobust
        self.model = SKRobust()

    def fit_transform(self, X):
        return self.model.fit_transform(np.asarray(X, dtype=float))

    def transform(self, X):
        return self.model.transform(np.asarray(X, dtype=float))


class Normalizer:
    """Normalizes samples individually to unit norm (L2 by default)."""
    def __init__(self, norm="l2"):
        from sklearn.preprocessing import Normalizer as SKNorm
        self.model = SKNorm(norm=norm)

    def fit_transform(self, X):
        return self.model.fit_transform(np.asarray(X, dtype=float))

    def transform(self, X):
        return self.model.transform(np.asarray(X, dtype=float))


class OneHotEncoder:
    """Encodes categorical integer features as a one-hot numeric array."""
    def __init__(self, sparse_output=False):
        from sklearn.preprocessing import OneHotEncoder as SKOHE
        self.model = SKOHE(sparse_output=sparse_output)

    def fit_transform(self, X):
        return self.model.fit_transform(np.asarray(X))

    def transform(self, X):
        return self.model.transform(np.asarray(X))


class Binarizer:
    """Binarizes data (set feature values to 0 or 1) according to a threshold."""
    def __init__(self, threshold=0.0):
        from sklearn.preprocessing import Binarizer as SKBin
        self.model = SKBin(threshold=threshold)

    def fit_transform(self, X):
        return self.model.fit_transform(np.asarray(X, dtype=float))

    def transform(self, X):
        return self.model.transform(np.asarray(X, dtype=float))


class VarianceThreshold:
    """Feature selector that removes all low-variance features."""
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.variances_ = None
        self.selected_indices_ = None

    def fit(self, X: Any, y: Any = None) -> "VarianceThreshold":
        X_arr = np.array(X)
        self.variances_ = np.var(X_arr, axis=0)
        self.selected_indices_ = np.where(self.variances_ > self.threshold)[0]
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        if self.selected_indices_ is None:
            raise RuntimeError("VarianceThreshold is not fitted yet.")
        if len(self.selected_indices_) == 0:
            return np.empty((X_arr.shape[0], 0))
        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


class SelectKBest:
    """Select features according to the k highest scores."""
    def __init__(self, k: int = 10):
        self.k = k
        self.scores_ = None
        self.selected_indices_ = None

    def fit(self, X: Any, y: Any) -> "SelectKBest":
        X_arr = np.array(X)
        y_arr = np.array(y)
        n_features = X_arr.shape[1]
        scores = []
        for col in range(n_features):
            corr = np.corrcoef(X_arr[:, col], y_arr)[0, 1] if X_arr.shape[0] > 1 else 0.0
            scores.append(abs(corr) if not np.isnan(corr) else 0.0)
        self.scores_ = np.array(scores)
        top_k = min(self.k, n_features)
        self.selected_indices_ = np.argsort(self.scores_)[-top_k:]
        return self

    def transform(self, X: Any) -> np.ndarray:
        X_arr = np.array(X)
        if self.selected_indices_ is None:
            raise RuntimeError("SelectKBest is not fitted yet.")
        return X_arr[:, self.selected_indices_]

    def fit_transform(self, X: Any, y: Any) -> np.ndarray:
        return self.fit(X, y).transform(X)
