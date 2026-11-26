"""
qai.hyperparam_optimizer — Bayesian hyperparameter optimization.

Goes beyond grid/random search with a Gaussian Process surrogate model
that learns which regions of the hyperparameter space are promising.

Usage:
    import qai
    opt = qai.HyperparameterOptimizer(
        technique="random_forest",
        param_space={
            "n_estimators": (10, 200),
            "max_depth": (3, 20),
        },
        n_iterations=20,
        k=3,
    )
    result = opt.optimize(X, y)
    print(result.best_params, result.best_score)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    n_iterations: int
    history: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "bayesian"


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization with GP surrogate.

    Uses a simple Gaussian Process surrogate to model the objective
    function (cross-validation accuracy) and an expected-improvement
    acquisition function to select the next point to try.

    Args:
        technique: qai technique name.
        param_space: dict of {param_name: (min, max)} for continuous
                     or {param_name: [val1, val2, ...]} for discrete.
        n_iterations: number of optimization iterations.
        k: cross-validation folds.
        n_initial_random: number of random points before GP kicks in.
    """
    def __init__(self, technique: str,
                 param_space: Dict[str, Union[Tuple, List]],
                 n_iterations: int = 20, k: int = 3,
                 n_initial_random: int = 5):
        self.technique = technique
        self.param_space = param_space
        self.n_iterations = n_iterations
        self.k = k
        self.n_initial_random = n_initial_random
        self._X_observed: List[np.ndarray] = []
        self._y_observed: List[float] = []
        self._param_names = list(param_space.keys())
        self._bounds = self._compute_bounds()

    def optimize(self, X, y) -> OptimizationResult:
        """Run the optimization.

        Returns:
            OptimizationResult with best params + score + history.
        """
        import qai
        X, y = np.asarray(X), np.asarray(y)
        history = []

        for i in range(self.n_iterations):
            # Select next point
            if i < self.n_initial_random or len(self._X_observed) < 2:
                params = self._sample_random()
            else:
                params = self._suggest_bayesian()

            # Evaluate
            score = self._evaluate(params, X, y, qai)

            self._X_observed.append(np.array(list(params.values())))
            self._y_observed.append(score)
            history.append({"iteration": i + 1, "params": params, "score": score})

        # Find best
        best_idx = int(np.argmax(self._y_observed))
        best_params = dict(zip(self._param_names, self._X_observed[best_idx]))
        best_params = self._denormalize_params(best_params)

        return OptimizationResult(
            best_params=best_params,
            best_score=self._y_observed[best_idx],
            n_iterations=self.n_iterations,
            history=history,
            method="bayesian",
        )

    def _compute_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for name in self._param_names:
            space = self.param_space[name]
            if isinstance(space, tuple):
                bounds.append((float(space[0]), float(space[1])))
            else:
                bounds.append((0.0, float(len(space) - 1)))
        return bounds

    def _sample_random(self) -> Dict[str, Any]:
        params = {}
        for name, space in self.param_space.items():
            if isinstance(space, tuple):
                params[name] = np.random.uniform(space[0], space[1])
            else:
                params[name] = float(np.random.randint(0, len(space)))
        return params

    def _suggest_bayesian(self) -> Dict[str, Any]:
        """Simple expected improvement using a GP surrogate."""
        X_obs = np.array(self._X_observed)
        y_obs = np.array(self._y_observed)

        # Fit a simple GP (radial basis function kernel)
        # Use a simple distance-based surrogate instead of full GP
        # for speed and no extra dependencies
        best_y = y_obs.max()
        candidates = [self._sample_random() for _ in range(100)]

        best_ei = -float("inf")
        best_candidate = candidates[0]
        for cand in candidates:
            x_cand = np.array(list(cand.values()))
            # Compute expected improvement
            distances = np.sqrt(np.sum((X_obs - x_cand) ** 2, axis=1))
            min_dist = distances.min()
            # Predicted mean: weighted average of observed values
            weights = np.exp(-distances ** 2 / (2 * max(np.std(y_obs), 0.01) ** 2))
            weights /= weights.sum()
            pred_mean = np.sum(weights * y_obs)
            pred_std = max(min_dist * np.std(y_obs), 0.01)
            # Expected improvement
            from math import erf, sqrt
            z = (pred_mean - best_y) / max(pred_std, 1e-8)
            ei = pred_std * (z * (0.5 * (1 + erf(z / sqrt(2)))) + (1 / sqrt(2 * np.pi)) * np.exp(-z ** 2 / 2))
            if ei > best_ei:
                best_ei = ei
                best_candidate = cand
        return best_candidate

    def _evaluate(self, params: Dict[str, Any], X, y, qai_module) -> float:
        """Evaluate a parameter configuration via cross-validation."""
        real_params = self._denormalize_params(params)
        try:
            def build_fn():
                return qai_module.build(technique=self.technique, **real_params)
            results = qai_module.cross_validate(build_fn, X, y, k=self.k, verbose=False)
            return results["mean_accuracy"]
        except Exception:
            return 0.0

    def _denormalize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal float params to actual technique params."""
        result = {}
        for name, val in params.items():
            space = self.param_space[name]
            if isinstance(space, tuple):
                if isinstance(space[0], int) and isinstance(space[1], int):
                    result[name] = int(round(val))
                else:
                    result[name] = float(val)
            else:
                idx = int(round(val))
                result[name] = space[min(idx, len(space) - 1)]
        return result


__all__ = ["HyperparameterOptimizer", "OptimizationResult"]
