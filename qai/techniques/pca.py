from __future__ import annotations
import numpy as np
from .base import Technique


class PCA(Technique):
    """Principal Component Analysis -- finds directions of greatest variance
    (Section 32). Deliberately a DIFFERENT shape of unsupervised technique
    than K-Means: no "cluster label" output at all, forward() TRANSFORMS
    the input into a lower-dimensional space instead of predicting anything.
    A real stress test of whether the Model interface generalizes to this,
    not just to clustering/classification."""
    name = "pca"
    family = "classical_ml"
    compatible_learning_techniques = ("unsupervised",)

    def __init__(self, n_components=2, **params):
        super().__init__(n_components=n_components, **params)
        from sklearn.decomposition import PCA as SKPCA
        self.model = SKPCA(n_components=n_components, **params)

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.model.fit(X)
        self._trained = True
        return self

    def forward(self, x):
        """Returns the TRANSFORMED (reduced-dimension) representation, not
        a class label -- forward() means genuinely different things across
        techniques, exactly as the design doc's Section 3 requires."""
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        result = self.model.transform(x)
        return result[0] if single else result

    # --- unique tools -----------------------------------------------------
    def explained_variance_ratio(self):
        return self.model.explained_variance_ratio_.tolist()

    def find_eigenvectors(self):
        """The design doc's Section 20 formula for PCA: 'finds eigenvectors
        of the covariance matrix' -- this is literally that, exposed."""
        return self.model.components_

    def reconstruction_error(self, x):
        """Compresses then decompresses -- how much information was lost."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        compressed = self.model.transform(x)
        reconstructed = self.model.inverse_transform(compressed)
        return float(np.mean((x - reconstructed) ** 2))
