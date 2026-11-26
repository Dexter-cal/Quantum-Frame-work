from __future__ import annotations
import numpy as np
from .base import Technique

class GaussianMixtureModel(Technique):
    """Unsupervised Gaussian Mixture Model (GMM) probabilistic clustering.

    Advanced features:
        - Variational Bayesian inference: set `variational_bayes=True` to use
          BayesianGaussianMixture with a Dirichlet process prior. This
          automatically infers the number of active components from the data
          (avoids the "I picked the wrong k" problem). Set
          `weight_concentration_prior` low to favor few active components,
          high to favor many.
        - Multiple initialization methods: `init_params` can be "kmeans"
          (default, most accurate but slow), "kmeans++" (faster), "random"
          (fastest, may not converge), or "random_from_data".
        - Online learning: call `partial_fit(X)` to update the model with new
          data without retraining from scratch.
        - Model selection: use `bic(X)` or `aic(X)` to compare different k
          values and pick the best one.
    """
    name = "gmm"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=3, covariance_type="full",
                 variational_bayes=False, init_params="kmeans",
                 weight_concentration_prior=1e-3, **params):
        # Accept n_clusters as an alias for n_components (for ClusterConfig compat)
        if "n_clusters" in params:
            n_components = params.pop("n_clusters")
        # Filter out params that don't apply to sklearn's GMM/BayesianGaussianMixture
        sklearn_relevant = ["tol", "max_iter", "n_init", "init_params",
                            "reg_covar", "random_state", "warm_start",
                            "verbose", "verbose_interval"]
        sklearn_params = {k: v for k, v in params.items() if k in sklearn_relevant}
        super().__init__(n_components=n_components, covariance_type=covariance_type,
                         variational_bayes=variational_bayes,
                         init_params=init_params,
                         weight_concentration_prior=weight_concentration_prior,
                         **params)
        self.n_components = n_components
        self.variational_bayes = variational_bayes
        self.init_params = init_params
        if variational_bayes:
            from sklearn.mixture import BayesianGaussianMixture
            self.model = BayesianGaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                init_params=init_params,
                weight_concentration_prior=weight_concentration_prior,
                **sklearn_params)
        else:
            from sklearn.mixture import GaussianMixture
            self.model = GaussianMixture(
                n_components=n_components,
                covariance_type=covariance_type,
                init_params=init_params,
                **sklearn_params)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
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
        return int(res[0]) if single else res

    # --- unique tools ---
    def predict_proba(self, x):
        """Return component posterior probabilities for x."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before predict_proba.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.predict_proba(x)

    def score_samples(self, x):
        """Compute the log-likelihood of each sample under the model."""
        if not self._trained:
            raise RuntimeError("Model must be fitted before score_samples.")
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.model.score_samples(x)

    def means(self):
        return self.model.means_

    def covariances(self):
        return self.model.covariances_

    def bic(self, X):
        """Bayesian Information Criterion — lower is better. Use for model selection.
        Only available for standard (non-variational) GMM."""
        if self.variational_bayes:
            raise AttributeError(
                "BayesianGaussianMixture doesn't have bic(). "
                "Use variational_bayes=False for model selection with bic/aic.")
        X = np.asarray(X, dtype=float)
        return float(self.model.bic(X))

    def aic(self, X):
        """Akaike Information Criterion — lower is better. Use for model selection.
        Only available for standard (non-variational) GMM."""
        if self.variational_bayes:
            raise AttributeError(
                "BayesianGaussianMixture doesn't have aic(). "
                "Use variational_bayes=False for model selection with bic/aic.")
        X = np.asarray(X, dtype=float)
        return float(self.model.aic(X))

    def active_components(self):
        """For variational_bayes=True: return the number of components with
        non-negligible weight. For standard GMM, returns n_components."""
        if self.variational_bayes:
            weights = self.model.weights_
            return int(np.sum(weights > 1e-3))
        return self.n_components
