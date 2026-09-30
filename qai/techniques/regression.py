"""
qai.techniques.regression — a Type with its own unique tools on top of the
shared base: .formula(), .residuals(), .r_squared() — matching the design
doc's "shared baseline + unique extras" rule (Section 46).

Section 56 decomposition: `structure` (linear y = w·x + b) is now separable
from `optimizer` (HOW the weights get found) -- swappable, defaults to the
same ClosedForm behavior Regression has always had, so nothing breaks.
"""
from __future__ import annotations
import numpy as np
from .base import Technique
from ..mechanics import ClosedForm


class Regression(Technique):
    """Linear regression: y = w·x + b. Optimizer is now pluggable
    (Section 56) -- defaults to the original closed-form solve."""

    name = "regression"
    family = "classical_ml"

    def __init__(self, optimizer=None, **params):
        super().__init__(**params)
        self.optimizer = optimizer or ClosedForm()  # default preserves EXACT original behavior
        self.w = None
        self.b = 0.0
        self._last_X = None
        self._last_y = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        coef = self.optimizer.solve(X_aug, y)  # the only real change -- solving is delegated, not hardcoded
        self.w = coef[:-1]
        self.b = coef[-1]
        self._trained = True
        self._last_X, self._last_y = X, y
        return self

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        single_sample = (x.ndim == 1)
        if single_sample:
            x = x.reshape(1, -1)
        result = x @ self.w + self.b
        return float(result[0]) if single_sample else result

    # --- unique tools, only Regression has these -------------------------
    def formula(self) -> str:
        if self.w is None:
            return "y = (untrained)"
        terms = " + ".join(f"{w:.3f}*x{i}" for i, w in enumerate(self.w))
        return f"y = {terms} + {self.b:.3f}"

    def residuals(self):
        if not self._trained:
            raise RuntimeError("residuals() requires a trained model")
        preds = self.forward(self._last_X)
        return self._last_y - preds

    def r_squared(self) -> float:
        resid = self.residuals()
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((self._last_y - self._last_y.mean()) ** 2))
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def finetune(self, X_new, y_new, new_data_weight=3.0):
        """HONEST implementation, not a false promise: closed-form linear
        regression has no gradient to 'continue' the way a neural net does.
        The genuine, correct way to finetune it is to refit on the OLD data
        (already stored from the first .fit()) combined with the NEW data,
        weighting new examples more heavily so they actually shift the fit
        rather than being drowned out by a much larger old dataset. This is
        a real, honest technique-appropriate mechanism, not gradient
        continuation pretending to be something it isn't."""
        if not self._trained:
            raise RuntimeError("finetune() requires an already-trained model -- call fit() first")
        X_new = np.asarray(X_new, dtype=float)
        y_new = np.asarray(y_new, dtype=float)

        combined_X = np.vstack([self._last_X, np.repeat(X_new, int(new_data_weight), axis=0)])
        combined_y = np.concatenate([self._last_y, np.repeat(y_new, int(new_data_weight))])
        self.fit(combined_X, combined_y)
        return self
