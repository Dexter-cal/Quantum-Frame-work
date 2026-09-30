"""
Data preprocessing, scaling, encoding, and missing-value handling utilities for qai.
"""
import numpy as np

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
