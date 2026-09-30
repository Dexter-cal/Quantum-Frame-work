"""
qai.core.weights -- Section 4/78/82: real weight inspection and direct
editing, not just documentation. Works for any technique exposing a real
coefficient vector (Regression's w/b, or any sklearn linear model's
coef_/intercept_).
"""
from __future__ import annotations
import numpy as np


class WeightAccessor:
    """model.weights -- a real, live view onto a technique's actual
    numbers, with get/set that genuinely mutate the underlying model."""

    def __init__(self, technique):
        self.technique = technique

    def _get_coef_and_bias(self):
        if hasattr(self.technique, "w") and self.technique.w is not None:
            return self.technique.w, self.technique.b, "w"  # Regression's own attributes
        if hasattr(self.technique, "model") and hasattr(self.technique.model, "coef_"):
            return self.technique.model.coef_[0] if self.technique.model.coef_.ndim > 1 else self.technique.model.coef_, \
                   self.technique.model.intercept_, "sklearn"
        raise NotImplementedError(f"No inspectable weight vector for technique '{self.technique.name}'")

    def as_table(self):
        coef, bias, _ = self._get_coef_and_bias()
        rows = [{"name": f"w{i}", "value": float(w)} for i, w in enumerate(coef)]
        rows.append({"name": "bias", "value": float(np.asarray(bias).flatten()[0])})
        return rows

    def get(self, index: int) -> float:
        coef, _, _ = self._get_coef_and_bias()
        return float(coef[index])

    def set(self, index: int, value: float):
        coef, bias, kind = self._get_coef_and_bias()
        if kind == "w":
            self.technique.w[index] = value
        else:
            # sklearn linear models: coef_ may be 2D (n_classes, n_features) for classifiers
            if self.technique.model.coef_.ndim > 1:
                self.technique.model.coef_[0][index] = value
            else:
                self.technique.model.coef_[index] = value
        return self

    def raw_flat(self):
        coef, bias, _ = self._get_coef_and_bias()
        return np.concatenate([np.asarray(coef).flatten(), np.asarray(bias).flatten()])
