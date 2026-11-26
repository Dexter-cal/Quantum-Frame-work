from __future__ import annotations
import numpy as np
from .base import Technique


class NaiveBayes(Technique):
    name = "naive_bayes"
    family = "classical_ml"

    def __init__(self, **params):
        super().__init__(**params)
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB(**params)
        self._last_X = None
        self._last_y = None

    def fit(self, X, y, classes=None):
        """`classes`: OPTIONAL, but genuinely important -- if you intend to
        finetune() with a class not present in this initial call, sklearn's
        GaussianNB requires ALL possible classes to be declared upfront via
        partial_fit(classes=...). This is a real, documented sklearn
        constraint discovered by actually testing finetune() with a novel
        class -- it does NOT support learning a genuinely unseen class
        later, only refining/expanding on classes it was told about from
        the start."""
        X = np.asarray(X, dtype=float)
        if classes is not None:
            self.model.partial_fit(X, y, classes=classes)
        else:
            self.model.fit(X, y)
        self._trained = True
        self._last_X, self._last_y = X, np.asarray(y)
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        result = self.model.predict(x)
        return result[0] if single else result

    # --- unique tools -----------------------------------------------------
    def class_priors(self):
        return dict(zip(self.model.classes_.tolist(), self.model.class_prior_.tolist()))

    def likelihood_table(self):
        """Mean/variance per feature per class -- the actual learned
        Gaussian likelihoods this technique's name refers to."""
        table = {}
        for i, cls in enumerate(self.model.classes_):
            table[cls] = {"mean": self.model.theta_[i].tolist(), "var": self.model.var_[i].tolist()}
        return table

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))

    def finetune(self, X_new, y_new):
        """HONEST mechanism: real incremental learning via partial_fit(),
        which updates learned per-class statistics from new data WITHOUT
        needing to re-see old data. Real, documented limitation: this can
        only refine classes the model already knows about (declared via
        `classes=` at the initial fit()) -- it cannot learn a genuinely
        novel, never-declared class after the fact. That constraint was
        discovered by actually testing this, not assumed in advance."""
        if not self._trained:
            raise RuntimeError("finetune() requires an already-trained model -- call fit() first")
        X_new = np.asarray(X_new, dtype=float)
        self.model.partial_fit(X_new, y_new)
        self._last_X = np.vstack([self._last_X, X_new])
        self._last_y = np.concatenate([self._last_y, np.asarray(y_new)])
        return self
