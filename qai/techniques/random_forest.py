from __future__ import annotations
import numpy as np
from .base import Technique


class RandomForest(Technique):
    """Ensemble of Decision Trees (Section 46 pairs these together).
    A real test of technique COMPOSITION -- this Type internally builds
    many of another Type's underlying models, a genuine, working example
    of the ensemble pattern from Section 24/26."""
    name = "random_forest"
    family = "classical_ml"

    def __init__(self, n_estimators=100, **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, **params)
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
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
    def feature_importance(self):
        return dict(enumerate(self.model.feature_importances_.tolist()))

    def tree_count(self) -> int:
        return len(self.model.estimators_)

    def member_agreement(self, x):
        """What fraction of the forest's individual trees agree with the
        final vote -- a real, working example of Section 46's Ensemble
        .voting_breakdown() / .diversity_score() idea.

        Note: sklearn's internal tree estimators return ENCODED integer
        class indices, not the original labels -- must map back via
        self.model.classes_ before comparing to the forest's final
        (label-space) prediction. Caught by a real test on 2026-09-06."""
        x = np.asarray(x, dtype=float).reshape(1, -1)
        encoded_votes = [tree.predict(x)[0] for tree in self.model.estimators_]
        decoded_votes = [self.model.classes_[int(v)] for v in encoded_votes]
        final = self.forward(x[0])
        agreement = sum(1 for v in decoded_votes if v == final) / len(decoded_votes)
        return agreement

    def accuracy(self) -> float:
        preds = self.forward(self._last_X)
        return float(np.mean(preds == self._last_y))
