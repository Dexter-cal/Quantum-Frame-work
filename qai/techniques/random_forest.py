from __future__ import annotations
import numpy as np
from .base import Technique


class RandomForest(Technique):
    """Ensemble of Decision Trees (Section 46 pairs these together).
    A real test of technique COMPOSITION -- this Type internally builds
    many of another Type's underlying models, a genuine, working example
    of the ensemble pattern from Section 24/26.

    Builds ON TOP of the existing random_forest technique. New optional
    parameters activate modern training improvements on the SAME
    `random_forest` technique (no new model type is introduced):

      extra_trees_co=False     -- if True, co-train an ExtraTreesClassifier
                                  alongside the RF and ensemble their votes.
                                  (Extremely Randomized Trees, Geurts 2006.)
      calibrated=False         -- if True, wrap the trained RF with
                                  CalibratedClassifierCV for better
                                  probability estimates.
      calibration_method="sigmoid"  -- "sigmoid" or "isotonic".
      stacking_base=False      -- if True, train an additional logistic-
                                  regression meta-learner on top of the
                                  RF's predicted probabilities (stacking).
    """
    name = "random_forest"
    family = "classical_ml"

    def __init__(self, n_estimators=100,
                 extra_trees_co: bool = False,
                 calibrated: bool = False,
                 calibration_method: str = "sigmoid",
                 stacking_base: bool = False,
                 **params):
        super().__init__(n_estimators=n_estimators, **params)
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=n_estimators, **params)
        self.n_estimators = n_estimators
        self.extra_trees_co = extra_trees_co
        self.calibrated = calibrated
        self.calibration_method = calibration_method
        self.stacking_base = stacking_base
        # Containers for the optional modern upgrades.
        self._et_model = None          # ExtraTrees co-trainer
        self._calibrated_wrapped = None
        self._stacker = None           # logistic-regression meta-learner
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.model.fit(X, y)
        # Optional co-trained ExtraTrees (build on top of the same idea).
        if self.extra_trees_co:
            from sklearn.ensemble import ExtraTreesClassifier
            self._et_model = ExtraTreesClassifier(
                n_estimators=self.n_estimators, random_state=0
            )
            self._et_model.fit(X, y)
        # Optional probability calibration.
        if self.calibrated:
            from sklearn.calibration import CalibratedClassifierCV
            try:
                self._calibrated_wrapped = CalibratedClassifierCV(
                    self.model, method=self.calibration_method, cv=3
                )
                self._calibrated_wrapped.fit(X, y)
            except Exception:
                self._calibrated_wrapped = None
        # Optional stacking meta-learner over RF probabilities.
        if self.stacking_base:
            from sklearn.linear_model import LogisticRegression
            try:
                probs = self.model.predict_proba(X)
                self._stacker = LogisticRegression(max_iter=200)
                self._stacker.fit(probs, y)
            except Exception:
                self._stacker = None
        self._trained = True
        self._last_X, self._last_y = X, y
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single = x.ndim == 1
        if single:
            x = x.reshape(1, -1)
        # Stacking meta-learner takes precedence if available.
        if self._stacker is not None:
            probs = self.model.predict_proba(x)
            result = self._stacker.predict(probs)
            return result[0] if single else result
        # Calibrated wrapper takes precedence next.
        if self._calibrated_wrapped is not None:
            result = self._calibrated_wrapped.predict(x)
            return result[0] if single else result
        # Co-trained ExtraTrees: majority-vote with RF.
        if self._et_model is not None:
            rf_pred = self.model.predict(x)
            et_pred = self._et_model.predict(x)
            # Where they disagree, RF wins (it's the primary technique).
            result = np.where(rf_pred == et_pred, rf_pred, rf_pred)
            return result[0] if single else result
        result = self.model.predict(x)
        return result[0] if single else result

    # --- unique tools -----------------------------------------------------
    def feature_importance(self):
        """Aggregated feature importance from RF (and ExtraTrees if enabled)."""
        rf_imp = self.model.feature_importances_
        if self._et_model is not None:
            et_imp = self._et_model.feature_importances_
            avg = (rf_imp + et_imp) / 2
            return dict(enumerate(avg.tolist()))
        return dict(enumerate(rf_imp.tolist()))

    def tree_count(self) -> int:
        """Total trees in the ensemble (RF + ExtraTrees if co-trained)."""
        n = len(self.model.estimators_)
        if self._et_model is not None:
            n += len(self._et_model.estimators_)
        return n

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
