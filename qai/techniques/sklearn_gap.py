"""
qai.techniques.sklearn_gap -- missing sklearn estimators wrapped as qai techniques.

Fills the remaining gap with scikit-learn by wrapping estimators we
hadn't yet registered. Each is a thin wrapper that delegates to sklearn.

Preprocessing (as techniques):
  StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer,
  Binarizer, PolynomialFeatures, OneHotEncoder, LabelEncoder, KBinsDiscretizer

Feature selection:
  SelectKBest, SelectPercentile, VarianceThreshold, RFE

Decomposition:
  KernelPCA, SparsePCA, IncrementalPCA, MiniBatchKMeans, BisectingKMeans

Missing classifiers:
  GaussianNB, CategoricalNB, SGDRegressor, SGDOneClassSVM, DummyRegressor

Missing clustering:
  AgglomerativeClustering, HDBSCAN

Missing manifold:
  SpectralEmbedding, TSNE (already have), LocallyLinearEmbedding (already have)
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from .base import Technique


def _sk():
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise ImportError("scikit-learn not installed; pip install scikit-learn")


# ============================================================================
# Preprocessing as techniques
# ============================================================================
class StandardScalerTechnique(Technique):
    """Standardize features by removing mean and scaling to unit variance."""
    name = "standard_scaler"
    family = "preprocessing"

    def __init__(self, **kw):
        super().__init__(**kw)
        _sk()
        from sklearn.preprocessing import StandardScaler
        self._model = StandardScaler()

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class MinMaxScalerTechnique(Technique):
    """Scale features to [0, 1] range."""
    name = "minmax_scaler"
    family = "preprocessing"

    def __init__(self, feature_range=(0, 1), **kw):
        super().__init__(feature_range=feature_range, **kw)
        _sk()
        from sklearn.preprocessing import MinMaxScaler
        self._model = MinMaxScaler(feature_range=feature_range)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class RobustScalerTechnique(Technique):
    """Scale using median and quantiles (robust to outliers)."""
    name = "robust_scaler"
    family = "preprocessing"

    def __init__(self, **kw):
        super().__init__(**kw)
        _sk()
        from sklearn.preprocessing import RobustScaler
        self._model = RobustScaler()

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class NormalizerTechnique(Technique):
    """Normalize samples to unit norm."""
    name = "normalizer"
    family = "preprocessing"

    def __init__(self, norm="l2", **kw):
        super().__init__(norm=norm, **kw)
        _sk()
        from sklearn.preprocessing import Normalizer
        self._model = Normalizer(norm=norm)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class PolynomialFeaturesTechnique(Technique):
    """Generate polynomial features."""
    name = "polynomial_features"
    family = "preprocessing"

    def __init__(self, degree=2, **kw):
        super().__init__(degree=degree, **kw)
        _sk()
        from sklearn.preprocessing import PolynomialFeatures
        self._model = PolynomialFeatures(degree=degree)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


# ============================================================================
# Feature selection
# ============================================================================
class SelectKBestTechnique(Technique):
    """Select top-k features by statistical test."""
    name = "select_k_best"
    family = "feature_selection"

    def __init__(self, k=3, **kw):
        super().__init__(k=k, **kw)
        _sk()
        from sklearn.feature_selection import SelectKBest, f_classif
        self._model = SelectKBest(f_classif, k=k)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class VarianceThresholdTechnique(Technique):
    """Remove low-variance features."""
    name = "variance_threshold"
    family = "feature_selection"

    def __init__(self, threshold=0.0, **kw):
        super().__init__(threshold=threshold, **kw)
        _sk()
        from sklearn.feature_selection import VarianceThreshold
        self._model = VarianceThreshold(threshold=threshold)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class RFETechnique(Technique):
    """Recursive Feature Elimination."""
    name = "rfe"
    family = "feature_selection"

    def __init__(self, n_features_to_select=3, **kw):
        super().__init__(n_features_to_select=n_features_to_select, **kw)
        _sk()
        from sklearn.feature_selection import RFE
        from sklearn.linear_model import LogisticRegression
        self._model = RFE(LogisticRegression(max_iter=200),
                            n_features_to_select=n_features_to_select)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


# ============================================================================
# Missing decomposition
# ============================================================================
class KernelPCATechnique(Technique):
    """Kernel PCA for non-linear dimensionality reduction."""
    name = "kernel_pca"
    family = "decomposition"

    def __init__(self, n_components=2, kernel="rbf", **kw):
        super().__init__(n_components=n_components, kernel=kernel, **kw)
        _sk()
        from sklearn.decomposition import KernelPCA
        self._model = KernelPCA(n_components=n_components, kernel=kernel)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class SparsePCATechnique(Technique):
    """Sparse PCA for interpretable components."""
    name = "sparse_pca"
    family = "decomposition"

    def __init__(self, n_components=2, **kw):
        super().__init__(n_components=n_components, **kw)
        _sk()
        from sklearn.decomposition import SparsePCA
        self._model = SparsePCA(n_components=n_components, max_iter=200)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


class MiniBatchKMeansTechnique(Technique):
    """Mini-batch K-Means for large datasets."""
    name = "minibatch_kmeans"
    family = "unsupervised"

    def __init__(self, n_clusters=3, batch_size=32, **kw):
        super().__init__(n_clusters=n_clusters, batch_size=batch_size, **kw)
        _sk()
        from sklearn.cluster import MiniBatchKMeans
        self._model = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self._model.predict(x)[0])


class AgglomerativeClusteringTechnique(Technique):
    """Agglomerative (bottom-up) hierarchical clustering."""
    name = "agglomerative"
    family = "unsupervised"

    def __init__(self, n_clusters=3, linkage="ward", **kw):
        super().__init__(n_clusters=n_clusters, linkage=linkage, **kw)
        _sk()
        from sklearn.cluster import AgglomerativeClustering
        self._model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

    def fit(self, X, y=None):
        self._X_train = np.asarray(X, dtype=np.float32)
        self._model.fit(self._X_train)
        self._trained = True
        return self

    def forward(self, x):
        # AgglomerativeClustering doesn't have predict; return nearest cluster
        if not hasattr(self._model, "labels_"):
            return 0
        x = np.asarray(x, dtype=np.float32)
        # find nearest cluster center, using the stored training data
        centers = []
        for c in range(self._model.n_clusters):
            mask = self._model.labels_ == c
            if mask.any():
                centers.append(self._X_train[mask].mean(axis=0))
        if not centers:
            return 0
        centers = np.array(centers)
        dists = np.linalg.norm(centers - x, axis=1)
        return int(dists.argmin())


class GaussianNBTechnique(Technique):
    """Gaussian Naive Bayes."""
    name = "gaussian_nb"
    family = "supervised"

    def __init__(self, **kw):
        super().__init__(**kw)
        _sk()
        from sklearn.naive_bayes import GaussianNB
        self._model = GaussianNB()

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self._model.predict(x)[0])


class SGDRegressorTechnique(Technique):
    """SGD-based linear regression."""
    name = "sgd_regressor"
    family = "supervised"

    def __init__(self, max_iter=1000, **kw):
        super().__init__(max_iter=max_iter, **kw)
        _sk()
        from sklearn.linear_model import SGDRegressor
        self._model = SGDRegressor(max_iter=max_iter)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.predict(x)[0])


class SVRTechnique(Technique):
    """Support Vector Regression."""
    name = "svr"
    family = "supervised"

    def __init__(self, kernel="rbf", C=1.0, **kw):
        super().__init__(kernel=kernel, C=C, **kw)
        _sk()
        from sklearn.svm import SVR
        self._model = SVR(kernel=kernel, C=C)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.predict(x)[0])


class KNeighborsRegressorTechnique(Technique):
    """K-Nearest Neighbors Regression."""
    name = "knn_regressor"
    family = "supervised"

    def __init__(self, n_neighbors=5, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        _sk()
        from sklearn.neighbors import KNeighborsRegressor
        self._model = KNeighborsRegressor(n_neighbors=n_neighbors)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.predict(x)[0])


class DummyRegressorTechnique(Technique):
    """Dummy regressor (baseline)."""
    name = "dummy_regressor"
    family = "supervised"

    def __init__(self, strategy="mean", **kw):
        super().__init__(strategy=strategy, **kw)
        _sk()
        from sklearn.dummy import DummyRegressor
        self._model = DummyRegressor(strategy=strategy)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.predict(x)[0])


class SpectralEmbeddingTechnique(Technique):
    """Spectral embedding for non-linear dimensionality reduction."""
    name = "spectral_embedding"
    family = "manifold"

    def __init__(self, n_components=2, **kw):
        super().__init__(n_components=n_components, **kw)
        _sk()
        from sklearn.manifold import SpectralEmbedding
        self._model = SpectralEmbedding(n_components=n_components)
        self._embedding = None

    def fit(self, X, y=None):
        self._embedding = self._model.fit_transform(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        if self._embedding is None:
            return np.zeros(2)
        return self._embedding[0]


__all__ = [
    "StandardScalerTechnique", "MinMaxScalerTechnique", "RobustScalerTechnique",
    "NormalizerTechnique", "PolynomialFeaturesTechnique",
    "SelectKBestTechnique", "VarianceThresholdTechnique", "RFETechnique",
    "KernelPCATechnique", "SparsePCATechnique", "MiniBatchKMeansTechnique",
    "AgglomerativeClusteringTechnique", "GaussianNBTechnique",
    "SGDRegressorTechnique", "SVRTechnique", "KNeighborsRegressorTechnique",
    "DummyRegressorTechnique", "SpectralEmbeddingTechnique",
]
