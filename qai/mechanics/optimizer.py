"""
qai.mechanics.optimizer -- Section 56's real claim under test: a Type's
`structure` (what makes it "Regression") should be separable from HOW its
weights actually get found. Two genuinely different optimizers here,
proving they're swappable without changing what the model IS.
"""
from __future__ import annotations
import numpy as np


class Optimizer:
    name = "optimizer"

    def solve(self, X_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Returns the learned coefficient vector (weights + bias combined,
        bias as the last element, matching Regression's X_aug convention)."""
        raise NotImplementedError


class ClosedForm(Optimizer):
    """What Regression has used all along -- exact, one-shot, via
    np.linalg.lstsq. The ORIGINAL mechanism, now formalized as one
    swappable option instead of hardcoded inside Regression itself."""
    name = "closed_form"

    def solve(self, X_aug, y):
        coef, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
        return coef


class GradientDescent(Optimizer):
    """A REAL, from-scratch gradient descent implementation -- not a
    library call. Now fully decomposed per Section 56: objective (how
    wrongness is measured) and training_loop (when to stop) are BOTH
    swappable, separable from the optimizer mechanism itself."""
    name = "gradient_descent"

    def __init__(self, learning_rate=0.01, objective=None, training_loop=None):
        from .objective import MSE
        from .training_loop import FixedEpochs
        self.learning_rate = learning_rate
        self.objective = objective or MSE()               # default preserves original behavior
        self.training_loop = training_loop or FixedEpochs(1000)  # default preserves original behavior
        self.loss_history = []

    def solve(self, X_aug, y):
        n_samples, n_features = X_aug.shape
        coef = np.zeros(n_features)
        self.loss_history = []
        epoch = 0

        while self.training_loop.should_continue(epoch, self.loss_history):
            predictions = X_aug @ coef
            loss = self.objective.compute(predictions, y)
            self.loss_history.append(loss)

            grad_wrt_predictions = self.objective.gradient(predictions, y)
            gradient = X_aug.T @ grad_wrt_predictions  # chain rule: d(loss)/d(coef) via d(loss)/d(pred)
            coef = coef - self.learning_rate * gradient
            epoch += 1

        return coef
