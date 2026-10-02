from __future__ import annotations
import numpy as np
from .base import Technique

class TSNE(Technique):
    """Unsupervised t-Distributed Stochastic Neighbor Embedding (t-SNE)."""
    name = "tsne"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, perplexity=30.0, **params):
        super().__init__(n_components=n_components, perplexity=perplexity, **params)
        from sklearn.manifold import TSNE as SKTSNE
        self.n_components = n_components
        self.perplexity = perplexity
        self.model = SKTSNE(n_components=n_components, perplexity=perplexity, **params)
        self.embedding_ = None
        self._last_X = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        if len(X) <= self.perplexity:
            # Perplexity must be less than n_samples
            self.model.set_params(perplexity=max(1.0, float(len(X) - 1)))
        self.embedding_ = self.model.fit_transform(X)
        self._last_X = X
        self._trained = True
        return self

    def forward(self, x):
        """t-SNE embedding for input points based on nearest trained reference points."""
        if not self._trained or self.embedding_ is None:
            raise RuntimeError("Model must be fitted before predict/forward.")
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)

        dists = np.linalg.norm(x[:, np.newaxis, :] - self._last_X[np.newaxis, :, :], axis=2)
        nearest_idx = np.argmin(dists, axis=1)
        res = self.embedding_[nearest_idx]
        return res[0] if single else res

    # --- unique tools ---
    def embedding(self):
        if not self._trained:
            raise RuntimeError("embedding() requires a fitted model.")
        return self.embedding_

    def kl_divergence(self):
        return getattr(self.model, "kl_divergence_", 0.0)
