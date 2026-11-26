"""
qai.mechanics.objective -- the second piece of Section 56's decomposition.
Objective answers "how is wrongness measured," separable from Optimizer
(how weights get found) and structure (what the model IS).
"""
from __future__ import annotations
import numpy as np


class Objective:
    name = "objective"

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """d(loss)/d(predictions), per sample -- what GradientDescent
        actually chains back through X to get d(loss)/d(weights)."""
        raise NotImplementedError


class MSE(Objective):
    """Mean Squared Error -- penalizes large errors quadratically, so it's
    genuinely SENSITIVE to outliers (one bad point can dominate the loss)."""
    name = "mse"

    def compute(self, predictions, targets):
        return float(np.mean((predictions - targets) ** 2))

    def gradient(self, predictions, targets):
        n = len(targets)
        return (2 / n) * (predictions - targets)


class MAE(Objective):
    """Mean Absolute Error -- penalizes all errors linearly, so it's
    genuinely more ROBUST to outliers than MSE. A real, checkable
    behavioral difference, not just a different formula on paper."""
    name = "mae"

    def compute(self, predictions, targets):
        return float(np.mean(np.abs(predictions - targets)))

    def gradient(self, predictions, targets):
        n = len(targets)
        return (1 / n) * np.sign(predictions - targets)
