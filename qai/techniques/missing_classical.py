"""
qai.techniques.missing_classical -- classical ML techniques we missed.

  - GaussianProcess          : Gaussian Process Regression/Classification
  - NMF                      : Non-negative Matrix Factorization
  - FactorAnalysis           : Factor Analysis
  - IsotonicRegression       : monotonic regression
  - BIRCH                    : Balanced Iterative Reducing and Clustering using Hierarchies
  - OPTICS                   : Ordering Points To Identify the Clustering Structure
  - MeanShift                : mean-shift clustering
  - AffinityPropagation      : message-passing clustering
  - MDS                      : Multi-Dimensional Scaling
  - PLSCanonical / CCA       : Partial Least Squares + Canonical Correlation Analysis
  - KernelDensityEstimator   : KDE
  - CovarianceEstimator      : Empirical/Shrunk/Robust covariance
  - ProbabilityCalibrator    : Platt/Isotonic calibration

All wrap scikit-learn implementations where available.
"""
from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .base import Technique


def _try_sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError:
        return None


_SK = _try_sklearn()


# ============================================================================
# 1. Gaussian Process
# ============================================================================
class GaussianProcess(Technique):
    """Gaussian Process for regression or classification.

    Wraps sklearn.gaussian_process.GaussianProcessRegressor / Classifier.
    """
    name = "gaussian_process"
    family = "supervised"

    def __init__(self, task: str = "regressor",
                 kernel: str = "rbf", alpha: float = 1e-10,
                 n_restarts_optimizer: int = 0, **kw):
        super().__init__(task=task, kernel=kernel, alpha=alpha,
                         n_restarts_optimizer=n_restarts_optimizer, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed; pip install scikit-learn")
        from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic
        kernels = {"rbf": RBF(), "matern": Matern(), "rational_quadratic": RationalQuadratic()}
        kern = kernels.get(kernel, RBF())
        if task == "classifier":
            self._model = GaussianProcessClassifier(kernel=kern, n_restarts_optimizer=n_restarts_optimizer)
        else:
            self._model = GaussianProcessRegressor(kernel=kern, alpha=alpha,
                                                     n_restarts_optimizer=n_restarts_optimizer)

    def fit(self, X, y):
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        pred = self._model.predict(x)
        return pred[0] if len(pred) == 1 else pred

    def predict_with_uncertainty(self, x):
        """For regression: returns (mean, std)."""
        x = np.asarray(x).reshape(1, -1) if np.asarray(x).ndim == 1 else np.asarray(x)
        if hasattr(self._model, "predict") and "return_std" in self._model.predict.__doc__ if self._model.predict.__doc__ else False:
            mean, std = self._model.predict(x, return_std=True)
            return float(mean[0]), float(std[0])
        return self.forward(x), 0.0


# ============================================================================
# 2. NMF (Non-negative Matrix Factorization)
# ============================================================================
class NMF(Technique):
    """Non-negative Matrix Factorization for topic modeling / dim reduction."""
    name = "nmf"
    family = "unsupervised"

    def __init__(self, n_components: int = 2, max_iter: int = 200,
                 init: str = "nndsvda", **kw):
        super().__init__(n_components=n_components, max_iter=max_iter, init=init, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.decomposition import NMF as SKNMF
        self._model = SKNMF(n_components=n_components, max_iter=max_iter, init=init)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        X = np.maximum(X, 0)  # NMF requires non-negative
        self._model.fit(X)
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        x = np.maximum(x, 0)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]

    def reconstruct(self, x):
        """Reconstruct input from latent representation."""
        latent = self.forward(x)
        return self._model.inverse_transform(latent.reshape(1, -1))[0]


# ============================================================================
# 3. Factor Analysis
# ============================================================================
class FactorAnalysis(Technique):
    """Factor Analysis for latent variable discovery."""
    name = "factor_analysis"
    family = "unsupervised"

    def __init__(self, n_components: int = 2, max_iter: int = 1000, **kw):
        super().__init__(n_components=n_components, max_iter=max_iter, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.decomposition import FactorAnalysis as SKFA
        self._model = SKFA(n_components=n_components, max_iter=max_iter)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.transform(x)[0]


# ============================================================================
# 4. Isotonic Regression
# ============================================================================
class IsotonicRegression(Technique):
    """Monotonic regression: ensures predictions are non-decreasing."""
    name = "isotonic_regression"
    family = "supervised"

    def __init__(self, y_min: Optional[float] = None,
                 y_max: Optional[float] = None,
                 increasing: bool = True, **kw):
        super().__init__(y_min=y_min, y_max=y_max, increasing=increasing, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.isotonic import IsotonicRegression as SKIR
        self._model = SKIR(y_min=y_min, y_max=y_max, increasing=increasing)
        self._X_train = None

    def fit(self, X, y):
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()
        # sort by X
        order = np.argsort(X)
        self._X_train = X[order]
        self._model.fit(X, y)
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x).flatten()
        return float(self._model.predict(x)[0])


# ============================================================================
# 5. BIRCH clustering
# ============================================================================
class BIRCH(Technique):
    """BIRCH: memory-efficient hierarchical clustering for large datasets."""
    name = "birch"
    family = "unsupervised"

    def __init__(self, n_clusters: int = 3, threshold: float = 0.5, **kw):
        super().__init__(n_clusters=n_clusters, threshold=threshold, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cluster import Birch
        self._model = Birch(n_clusters=n_clusters, threshold=threshold)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return int(self._model.predict(x)[0])


# ============================================================================
# 6. OPTICS clustering
# ============================================================================
class OPTICS(Technique):
    """OPTICS: density-based clustering that builds a reachability plot."""
    name = "optics"
    family = "unsupervised"

    def __init__(self, min_samples: int = 5, max_eps: float = np.inf, **kw):
        super().__init__(min_samples=min_samples, max_eps=max_eps, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cluster import OPTICS as SKOPTICS
        self._model = SKOPTICS(min_samples=min_samples, max_eps=max_eps)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        # OPTICS doesn't have a clean predict; we use fit_predict labels
        return self._model.labels_[0] if hasattr(self._model, "labels_") else 0


# ============================================================================
# 7. Mean Shift clustering
# ============================================================================
class MeanShift(Technique):
    """Mean-shift: mode-seeking clustering via kernel density estimation."""
    name = "mean_shift"
    family = "unsupervised"

    def __init__(self, bandwidth: Optional[float] = None, **kw):
        super().__init__(bandwidth=bandwidth, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cluster import MeanShift
        self._model = MeanShift(bandwidth=bandwidth)
        self._cluster_centers = None

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._cluster_centers = self._model.cluster_centers_
        self._trained = True
        return self

    def forward(self, x):
        if self._cluster_centers is None:
            return 0
        x = np.asarray(x, dtype=np.float32)
        dists = np.linalg.norm(self._cluster_centers - x, axis=1)
        return int(np.argmin(dists))


# ============================================================================
# 8. Affinity Propagation
# ============================================================================
class AffinityPropagation(Technique):
    """Affinity Propagation: message-passing clustering that auto-selects exemplars."""
    name = "affinity_propagation"
    family = "unsupervised"

    def __init__(self, damping: float = 0.5, max_iter: int = 200, **kw):
        super().__init__(damping=damping, max_iter=max_iter, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cluster import AffinityPropagation as SKAP
        self._model = SKAP(damping=damping, max_iter=max_iter)
        self._cluster_centers = None

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._cluster_centers = self._model.cluster_centers_
        self._trained = True
        return self

    def forward(self, x):
        if self._cluster_centers is None or len(self._cluster_centers) == 0:
            return 0
        x = np.asarray(x, dtype=np.float32)
        dists = np.linalg.norm(self._cluster_centers - x, axis=1)
        return int(np.argmin(dists))


# ============================================================================
# 9. MDS (Multi-Dimensional Scaling)
# ============================================================================
class MDS(Technique):
    """MDS: embed high-D data into low-D preserving pairwise distances."""
    name = "mds"
    family = "manifold"

    def __init__(self, n_components: int = 2, max_iter: int = 300,
                 dissimilarity: str = "euclidean", **kw):
        super().__init__(n_components=n_components, max_iter=max_iter,
                         dissimilarity=dissimilarity, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.manifold import MDS as SKMDS
        self._model = SKMDS(n_components=n_components, max_iter=max_iter,
                              dissimilarity=dissimilarity, normalized_stress="auto")

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        # MDS doesn't have transform; re-fit on combined data
        return self._model.embedding_[0]


# ============================================================================
# 10. PLS / CCA (Cross decomposition)
# ============================================================================
class PLSCanonical(Technique):
    """Partial Least Squares Canonical: finds correlations between two datasets."""
    name = "pls_canonical"
    family = "supervised"

    def __init__(self, n_components: int = 2, **kw):
        super().__init__(n_components=n_components, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cross_decomposition import PLSCanonical as SKPLS
        self._model = SKPLS(n_components=n_components)

    def fit(self, X, y):
        self._model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.predict(x)[0]


class CCA(Technique):
    """Canonical Correlation Analysis."""
    name = "cca"
    family = "supervised"

    def __init__(self, n_components: int = 2, **kw):
        super().__init__(n_components=n_components, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.cross_decomposition import CCA as SKCCA
        self._model = SKCCA(n_components=n_components)

    def fit(self, X, y):
        self._model.fit(np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self._model.predict(x)[0]


# ============================================================================
# 11. KDE (Kernel Density Estimation)
# ============================================================================
class KernelDensityEstimator(Technique):
    """KDE: non-parametric density estimation."""
    name = "kde"
    family = "unsupervised"

    def __init__(self, bandwidth: float = 1.0, kernel: str = "gaussian", **kw):
        super().__init__(bandwidth=bandwidth, kernel=kernel, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.neighbors import KernelDensity
        self._model = KernelDensity(bandwidth=bandwidth, kernel=kernel)

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        """Returns log-density at x."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.score_samples(x)[0])

    def sample(self, n: int = 1):
        return self._model.sample(n)


# ============================================================================
# 12. Covariance Estimation
# ============================================================================
class CovarianceEstimator(Technique):
    """Covariance estimation: empirical / shrunk / robust."""
    name = "covariance"
    family = "unsupervised"

    def __init__(self, method: str = "empirical",
                 shrinkage: float = 0.1, **kw):
        super().__init__(method=method, shrinkage=shrinkage, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        if method == "empirical":
            from sklearn.covariance import EmpiricalCovariance
            self._model = EmpiricalCovariance()
        elif method == "shrunk":
            from sklearn.covariance import ShrunkCovariance
            self._model = ShrunkCovariance(shrinkage=shrinkage)
        elif method == "robust":
            from sklearn.covariance import MinCovDet
            self._model = MinCovDet()
        else:
            from sklearn.covariance import EmpiricalCovariance
            self._model = EmpiricalCovariance()

    def fit(self, X, y=None):
        self._model.fit(np.asarray(X, dtype=np.float32))
        self._trained = True
        return self

    def forward(self, x):
        """Returns Mahalanobis distance."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return float(self._model.mahalanobis(x)[0])

    @property
    def covariance(self):
        return self._model.covariance_ if hasattr(self._model, "covariance_") else None


# ============================================================================
# 13. Probability Calibration
# ============================================================================
class ProbabilityCalibrator(Technique):
    """Calibrate predicted probabilities (Platt sigmoid or isotonic)."""
    name = "probability_calibration"
    family = "supervised"

    def __init__(self, method: str = "isotonic", cv: int = 3, **kw):
        super().__init__(method=method, cv=cv, **kw)
        if _SK is None:
            raise ImportError("scikit-learn not installed")
        from sklearn.calibration import CalibratedClassifierCV
        self._calibrator_cls = CalibratedClassifierCV
        self._method = method
        self._cv = cv
        self._model = None

    def fit(self, X, y):
        # needs a base estimator; use a simple LogisticRegression
        from sklearn.linear_model import LogisticRegression
        base = LogisticRegression(max_iter=200)
        self._model = self._calibrator_cls(base, method=self._method, cv=self._cv)
        self._model.fit(np.asarray(X), np.asarray(y))
        self._trained = True
        return self

    def forward(self, x):
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        probs = self._model.predict_proba(x)
        return probs[0]


__all__ = [
    "GaussianProcess", "NMF", "FactorAnalysis", "IsotonicRegression",
    "BIRCH", "OPTICS", "MeanShift", "AffinityPropagation", "MDS",
    "PLSCanonical", "CCA", "KernelDensityEstimator",
    "CovarianceEstimator", "ProbabilityCalibrator",
]
