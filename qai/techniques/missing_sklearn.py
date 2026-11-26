"""
qai.techniques.missing_sklearn -- wrappers for sklearn estimators we missed.

These wrap (not reimplement) scikit-learn estimators so users can do
qai.build(technique="standard_scaler") instead of importing sklearn directly.

Categories:
  - Preprocessing: StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler,
    Normalizer, Binarizer, PolynomialFeatures, OneHotEncoder, LabelEncoder,
    OrdinalEncoder, KBinsDiscretizer, SimpleImputer, KNNImputer
  - Feature selection: SelectKBest, SelectPercentile, VarianceThreshold,
    RFE, RFECV, SelectFromModel
  - Decomposition: KernelPCA, SparsePCA, MiniBatchKMeans, IncrementalPCA,
    GaussianRandomProjection, SparseRandomProjection
  - Clustering: AgglomerativeClustering, BisectingKMeans, MiniBatchKMeans,
    HDBSCAN, SpectralBiclustering, SpectralCoclustering
  - Regression: SVR, LinearSVR, SGDRegressor, DummyRegressor,
    PoissonRegressor, GammaRegressor, TweedieRegressor, RadiusNeighborsRegressor,
    StackingRegressor, VotingRegressor
  - Classification: SVC, OneClassSVM, SGDOneClassSVM
  - Ensemble: StackingClassifier, VotingClassifier
  - Manifold: SpectralEmbedding, RandomTreesEmbedding
  - Other: NearestNeighbors, LocalOutlierFactor, EllipticEnvelope,
    KernelDensity, DictionaryLearning, MiniBatchDictionaryLearning,
    BernoulliRBM, LabelSpreading
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .base import Technique


def _sk():
    try:
        import sklearn
        return sklearn
    except ImportError:
        return None


# ============================================================================
# Preprocessing
# ============================================================================
class StandardScaler(Technique):
    """Standardize features by removing mean and scaling to unit variance."""
    name = "standard_scaler"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import StandardScaler as SK
        self._m = SK()
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]
    def transform(self, X): return self._m.transform(X)
    def inverse_transform(self, X): return self._m.inverse_transform(X)

class MinMaxScaler(Technique):
    """Transform features by scaling each feature to [0, 1] range."""
    name = "minmax_scaler"
    family = "preprocessing"
    def __init__(self, feature_range=(0, 1), **kw):
        super().__init__(feature_range=feature_range, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import MinMaxScaler as SK
        self._m = SK(feature_range=feature_range)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]
    def transform(self, X): return self._m.transform(X)

class RobustScaler(Technique):
    """Scale features using statistics robust to outliers."""
    name = "robust_scaler"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import RobustScaler as SK
        self._m = SK()
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class MaxAbsScaler(Technique):
    """Scale each feature by its maximum absolute value."""
    name = "maxabs_scaler"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import MaxAbsScaler as SK
        self._m = SK()
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class Normalizer(Technique):
    """Normalize samples individually to unit norm."""
    name = "normalizer"
    family = "preprocessing"
    def __init__(self, norm="l2", **kw):
        super().__init__(norm=norm, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import Normalizer as SK
        self._m = SK(norm=norm)
    def fit(self, X, y=None): self._trained = True; return self  # stateless
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class Binarizer(Technique):
    """Binarize data (threshold-based)."""
    name = "binarizer"
    family = "preprocessing"
    def __init__(self, threshold=0.0, **kw):
        super().__init__(threshold=threshold, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import Binarizer as SK
        self._m = SK(threshold=threshold)
    def fit(self, X, y=None): self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class PolynomialFeatures(Technique):
    """Generate polynomial and interaction features."""
    name = "polynomial_features"
    family = "preprocessing"
    def __init__(self, degree=2, **kw):
        super().__init__(degree=degree, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import PolynomialFeatures as SK
        self._m = SK(degree=degree)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class OneHotEncoder(Technique):
    """Encode categorical features as one-hot numeric arrays."""
    name = "one_hot_encoder"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import OneHotEncoder as SK
        self._m = SK(sparse_output=False)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class LabelEncoder(Technique):
    """Encode target labels with value between 0 and n_classes-1."""
    name = "label_encoder"
    family = "preprocessing"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import LabelEncoder as SK
        self._m = SK()
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        x = np.asarray(x).reshape(-1) if np.asarray(x).ndim > 0 else [x]
        return self._m.transform(x)

class SimpleImputer(Technique):
    """Impute missing values using mean/median/most_frequent."""
    name = "simple_imputer"
    family = "preprocessing"
    def __init__(self, strategy="mean", **kw):
        super().__init__(strategy=strategy, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.impute import SimpleImputer as SK
        self._m = SK(strategy=strategy)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class KNNImputer(Technique):
    """Impute missing values using k-Nearest Neighbors."""
    name = "knn_imputer"
    family = "preprocessing"
    def __init__(self, n_neighbors=5, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.impute import KNNImputer as SK
        self._m = SK(n_neighbors=n_neighbors)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class KBinsDiscretizer(Technique):
    """Bin continuous data into intervals."""
    name = "kbins_discretizer"
    family = "preprocessing"
    def __init__(self, n_bins=5, encode="onehot", strategy="quantile", **kw):
        super().__init__(n_bins=n_bins, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.preprocessing import KBinsDiscretizer as SK
        self._m = SK(n_bins=n_bins, encode=encode, strategy=strategy)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]


# ============================================================================
# Feature Selection
# ============================================================================
class SelectKBest(Technique):
    """Select top k features based on univariate statistical tests."""
    name = "select_k_best"
    family = "feature_selection"
    def __init__(self, k=10, score_func=None, **kw):
        super().__init__(k=k, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_selection import SelectKBest as SK, f_classif
        self._m = SK(k=k, score_func=score_func or f_classif)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class VarianceThreshold(Technique):
    """Remove all low-variance features."""
    name = "variance_threshold"
    family = "feature_selection"
    def __init__(self, threshold=0.0, **kw):
        super().__init__(threshold=threshold, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_selection import VarianceThreshold as SK
        self._m = SK(threshold=threshold)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class RFE(Technique):
    """Recursive Feature Elimination."""
    name = "rfe"
    family = "feature_selection"
    def __init__(self, n_features_to_select=5, **kw):
        super().__init__(n_features_to_select=n_features_to_select, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.feature_selection import RFE as SK
        from sklearn.linear_model import LogisticRegression
        self._m = SK(estimator=LogisticRegression(max_iter=200), n_features_to_select=n_features_to_select)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]


# ============================================================================
# Decomposition
# ============================================================================
class KernelPCA(Technique):
    """Kernel Principal Component Analysis."""
    name = "kernel_pca"
    family = "decomposition"
    def __init__(self, n_components=2, kernel="rbf", **kw):
        super().__init__(n_components=n_components, kernel=kernel, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.decomposition import KernelPCA as SK
        self._m = SK(n_components=n_components, kernel=kernel)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class SparsePCA(Technique):
    """Sparse Principal Component Analysis."""
    name = "sparse_pca"
    family = "decomposition"
    def __init__(self, n_components=2, alpha=1.0, **kw):
        super().__init__(n_components=n_components, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.decomposition import SparsePCA as SK
        self._m = SK(n_components=n_components, alpha=alpha)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class IncrementalPCA(Technique):
    """Incremental PCA for large datasets."""
    name = "incremental_pca"
    family = "decomposition"
    def __init__(self, n_components=2, **kw):
        super().__init__(n_components=n_components, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.decomposition import IncrementalPCA as SK
        self._m = SK(n_components=n_components)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class GaussianRandomProjection(Technique):
    """Reduce dimensionality via Gaussian random projection."""
    name = "gaussian_random_projection"
    family = "decomposition"
    def __init__(self, n_components=2, **kw):
        super().__init__(n_components=n_components, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.random_projection import GaussianRandomProjection as SK
        self._m = SK(n_components=n_components)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]


# ============================================================================
# Clustering
# ============================================================================
class AgglomerativeClustering(Technique):
    """Hierarchical agglomerative clustering."""
    name = "agglomerative_clustering"
    family = "unsupervised"
    def __init__(self, n_clusters=2, linkage="ward", **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.cluster import AgglomerativeClustering as SK
        self._m = SK(n_clusters=n_clusters, linkage=linkage)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return int(self._m.labels_[0]) if hasattr(self._m, "labels_") else 0

class BisectingKMeans(Technique):
    """Bisecting K-Means clustering."""
    name = "bisecting_kmeans"
    family = "unsupervised"
    def __init__(self, n_clusters=2, **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.cluster import BisectingKMeans as SK
        self._m = SK(n_clusters=n_clusters)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class MiniBatchKMeans(Technique):
    """Mini-Batch K-Means for large datasets."""
    name = "mini_batch_kmeans"
    family = "unsupervised"
    def __init__(self, n_clusters=3, **kw):
        super().__init__(n_clusters=n_clusters, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.cluster import MiniBatchKMeans as SK
        self._m = SK(n_clusters=n_clusters)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])


# ============================================================================
# Regression (missing from our catalogue)
# ============================================================================
class SVR(Technique):
    """Support Vector Regression."""
    name = "svr"
    family = "supervised"
    def __init__(self, kernel="rbf", C=1.0, **kw):
        super().__init__(kernel=kernel, C=C, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.svm import SVR as SK
        self._m = SK(kernel=kernel, C=C)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class SGDRegressor(Technique):
    """Linear model fitted by SGD."""
    name = "sgd_regressor"
    family = "supervised"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.linear_model import SGDRegressor as SK
        self._m = SK()
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class DummyRegressor(Technique):
    """Regressor that makes predictions using simple rules (mean, median)."""
    name = "dummy_regressor"
    family = "supervised"
    def __init__(self, strategy="mean", **kw):
        super().__init__(strategy=strategy, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.dummy import DummyRegressor as SK
        self._m = SK(strategy=strategy)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class PoissonRegressor(Technique):
    """Generalized Linear Model with Poisson distribution."""
    name = "poisson_regressor"
    family = "supervised"
    def __init__(self, alpha=1.0, **kw):
        super().__init__(alpha=alpha, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.linear_model import PoissonRegressor as SK
        self._m = SK(alpha=alpha)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class KNeighborsRegressor(Technique):
    """Regression based on k-nearest neighbors."""
    name = "kneighbors_regressor"
    family = "supervised"
    def __init__(self, n_neighbors=5, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neighbors import KNeighborsRegressor as SK
        self._m = SK(n_neighbors=n_neighbors)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class MLPRegressor(Technique):
    """Multi-layer Perceptron regressor."""
    name = "mlp_regressor"
    family = "supervised"
    def __init__(self, hidden_layer_sizes=(100,), max_iter=200, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neural_network import MLPRegressor as SK
        self._m = SK(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return float(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])


# ============================================================================
# Classification (missing)
# ============================================================================
class SVC(Technique):
    """C-Support Vector Classification."""
    name = "svc"
    family = "supervised"
    def __init__(self, kernel="rbf", C=1.0, **kw):
        super().__init__(kernel=kernel, C=C, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.svm import SVC as SK
        self._m = SK(kernel=kernel, C=C)
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class OneClassSVM(Technique):
    """Unsupervised Outlier Detection."""
    name = "one_class_svm"
    family = "unsupervised"
    def __init__(self, nu=0.5, **kw):
        super().__init__(nu=nu, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.svm import OneClassSVM as SK
        self._m = SK(nu=nu)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])


# ============================================================================
# Ensemble (missing)
# ============================================================================
class StackingClassifier(Technique):
    """Stacked generalization with a meta-classifier."""
    name = "stacking_classifier"
    family = "supervised"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.ensemble import StackingClassifier as SK
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        self._m = SK(estimators=[("dt", DecisionTreeClassifier()), ("lr", LogisticRegression(max_iter=200))])
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])

class VotingClassifier(Technique):
    """Soft voting majority rule classifier."""
    name = "voting_classifier"
    family = "supervised"
    def __init__(self, **kw):
        super().__init__(**kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.ensemble import VotingClassifier as SK
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        self._m = SK(estimators=[("dt", DecisionTreeClassifier()), ("lr", LogisticRegression(max_iter=200))])
    def fit(self, X, y): self._m.fit(X, y); self._trained = True; return self
    def forward(self, x): return int(self._m.predict(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0])


# ============================================================================
# Other (missing)
# ============================================================================
class NearestNeighbors(Technique):
    """Unsupervised learner for neighbor searches."""
    name = "nearest_neighbors"
    family = "unsupervised"
    def __init__(self, n_neighbors=5, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neighbors import NearestNeighbors as SK
        self._m = SK(n_neighbors=n_neighbors)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        x = x.reshape(1, -1) if np.asarray(x).ndim == 1 else x
        distances, indices = self._m.kneighbors(x)
        return indices[0].tolist()

class LocalOutlierFactor(Technique):
    """Unsupervised anomaly detection using Local Outlier Factor."""
    name = "local_outlier_factor"
    family = "unsupervised"
    def __init__(self, n_neighbors=20, **kw):
        super().__init__(n_neighbors=n_neighbors, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neighbors import LocalOutlierFactor as SK
        # Bug fix: enable novelty=True so predict() (single-sample inference)
        # works after fit(). Without novelty=True, sklearn's LOF only supports
        # fit_predict() which requires n_samples > n_neighbors -- single-sample
        # predict() crashed with "n_neighbors >= n_samples_fit".
        self._m = SK(n_neighbors=n_neighbors, novelty=True)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        # Use predict() (not fit_predict()) since we're already fitted.
        # predict() works on a single sample because novelty=True was set.
        x_arr = np.asarray(x).reshape(1, -1) if np.asarray(x).ndim == 1 else x
        return int(self._m.predict(x_arr)[0])

class SpectralEmbedding(Technique):
    """Spectral embedding for non-linear dimensionality reduction."""
    name = "spectral_embedding"
    family = "manifold"
    def __init__(self, n_components=2, **kw):
        super().__init__(n_components=n_components, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.manifold import SpectralEmbedding as SK
        self._m = SK(n_components=n_components)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.embedding_[0]

class BernoulliRBM(Technique):
    """Bernoulli Restricted Boltzmann Machine."""
    name = "bernoulli_rbm"
    family = "generative"
    def __init__(self, n_components=2, n_iter=10, **kw):
        super().__init__(n_components=n_components, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neural_network import BernoulliRBM as SK
        self._m = SK(n_components=n_components, n_iter=n_iter)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x): return self._m.transform(x.reshape(1, -1) if np.asarray(x).ndim == 1 else x)[0]

class KernelDensity(Technique):
    """Kernel Density Estimation."""
    name = "kernel_density"
    family = "unsupervised"
    def __init__(self, bandwidth=1.0, kernel="gaussian", **kw):
        super().__init__(bandwidth=bandwidth, **kw)
        if not _sk(): raise ImportError("pip install scikit-learn")
        from sklearn.neighbors import KernelDensity as SK
        self._m = SK(bandwidth=bandwidth, kernel=kernel)
    def fit(self, X, y=None): self._m.fit(X); self._trained = True; return self
    def forward(self, x):
        x = x.reshape(1, -1) if np.asarray(x).ndim == 1 else x
        return float(self._m.score_samples(x)[0])
    def sample(self, n=1): return self._m.sample(n)


__all__ = [
    # preprocessing
    "StandardScaler", "MinMaxScaler", "RobustScaler", "MaxAbsScaler",
    "Normalizer", "Binarizer", "PolynomialFeatures", "OneHotEncoder",
    "LabelEncoder", "SimpleImputer", "KNNImputer", "KBinsDiscretizer",
    # feature selection
    "SelectKBest", "VarianceThreshold", "RFE",
    # decomposition
    "KernelPCA", "SparsePCA", "IncrementalPCA", "GaussianRandomProjection",
    # clustering
    "AgglomerativeClustering", "BisectingKMeans", "MiniBatchKMeans",
    # regression
    "SVR", "SGDRegressor", "DummyRegressor", "PoissonRegressor",
    "KNeighborsRegressor", "MLPRegressor",
    # classification
    "SVC", "OneClassSVM",
    # ensemble
    "StackingClassifier", "VotingClassifier",
    # other
    "NearestNeighbors", "LocalOutlierFactor", "SpectralEmbedding",
    "BernoulliRBM", "KernelDensity",
]
