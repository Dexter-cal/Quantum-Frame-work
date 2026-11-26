"""
qai.hparam_optimizer — Bayesian hyperparameter optimization with expected improvement.

More sophisticated than qai.AutoTuner's "bayesian" strategy (which is
explore+exploit on a grid). This does true Bayesian optimization using
a Gaussian Process surrogate model + Expected Improvement acquisition.

Usage:
    import qai
    import numpy as np

    def objective(params):
        m = qai.build(technique="random_forest", **params)
        m.train(X_train, y_train, verbose=False)
        return m.accuracy()

    opt = qai.HyperparameterOptimizer(
        param_space={
            "n_estimators": (10, 200),      # continuous range
            "max_depth": (3, 20),            # continuous range
        },
        objective=objective,
        n_initial=5,      # random exploration
        n_iterations=15,   # total evaluations
    )
    result = opt.optimize()
    print(f"Best params: {result.best_params}")
    print(f"Best score: {result.best_score}")
"""
from __future__ import annotations
import time, math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
import numpy as np


@dataclass
class OptimizationResult:
    """Result of a hyperparameter optimization run."""
    best_params: Dict[str, float] = field(default_factory=dict)
    best_score: float = 0.0
    n_evaluations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        lines = [
            f"=== Optimization Result ===",
            f"best params: {self.best_params}",
            f"best score:  {self.best_score:.4f}",
            f"evaluations: {self.n_evaluations}",
            f"elapsed:     {self.elapsed_seconds:.1f}s",
            "",
            "history:",
        ]
        for h in self.history[-5:]:
            lines.append(f"  iter {h['iter']}: score={h['score']:.4f} params={h['params']}")
        return "\n".join(lines)


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization with Expected Improvement.

    Uses a simple Gaussian Process surrogate model + EI acquisition
    function. No external dependencies (no scikit-optimize, no GPyOpt).

    Args:
        param_space: dict of {param_name: (min, max)} for continuous params.
        objective: callable(params_dict) -> float (higher = better).
        n_initial: number of random initial evaluations (default 5).
        n_iterations: total number of evaluations (default 20).
        minimize: if True, lower objective is better (default False = maximize).
        seed: random seed.
    """
    def __init__(self, param_space: Dict[str, Tuple[float, float]],
                 objective: Callable[[Dict[str, float]], float],
                 n_initial: int = 5,
                 n_iterations: int = 20,
                 minimize: bool = False,
                 seed: int = 42):
        self.param_space = param_space
        self.objective = objective
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.minimize = minimize
        self.rng = np.random.default_rng(seed)
        self._X: List[np.ndarray] = []  # evaluated param vectors
        self._y: List[float] = []       # objective values
        self._params: List[Dict[str, float]] = []  # param dicts

    def optimize(self) -> OptimizationResult:
        """Run the optimization. Returns OptimizationResult."""
        start = time.time()
        result = OptimizationResult()
        param_names = list(self.param_space.keys())
        bounds = np.array([self.param_space[n] for n in param_names])
        dim = len(param_names)

        # Phase 1: random exploration
        for i in range(self.n_initial):
            x = self.rng.uniform(bounds[:, 0], bounds[:, 1])
            params = self._vector_to_params(x, param_names)
            score = self._evaluate(params, i, result)

        # Phase 2: Bayesian optimization
        for i in range(self.n_initial, self.n_iterations):
            # Fit GP surrogate (simple: use RBF kernel + mean)
            X = np.array(self._X)
            y = np.array(self._y)

            # Find next point via Expected Improvement
            best_y = max(y) if not self.minimize else min(y)
            next_x = self._maximize_ei(X, y, bounds, best_y, n_candidates=200)

            params = self._vector_to_params(next_x, param_names)
            score = self._evaluate(params, i, result)

        # Find best
        if self.minimize:
            best_idx = int(np.argmin(self._y))
        else:
            best_idx = int(np.argmax(self._y))
        result.best_params = self._params[best_idx]
        result.best_score = self._y[best_idx]
        result.n_evaluations = len(self._y)
        result.elapsed_seconds = time.time() - start
        return result

    def _evaluate(self, params: Dict[str, float], iteration: int,
                  result: OptimizationResult) -> float:
        """Evaluate the objective at params."""
        # Round integer params
        rounded = {}
        for k, v in params.items():
            rounded[k] = int(round(v)) if "n_" in k or "max_" in k or "min_" in k else v

        score = self.objective(rounded)

        x = self._params_to_vector(rounded)
        self._X.append(x)
        self._y.append(score)
        self._params.append(rounded)

        result.history.append({
            "iter": iteration, "params": rounded, "score": score,
        })
        return score

    def _maximize_ei(self, X: np.ndarray, y: np.ndarray,
                     bounds: np.ndarray, best_y: float,
                     n_candidates: int = 200) -> np.ndarray:
        """Find the next point to evaluate by maximizing Expected Improvement."""
        dim = X.shape[1]

        # Generate candidate points
        candidates = self.rng.uniform(bounds[:, 0], bounds[:, 1], size=(n_candidates, dim))

        # Compute EI for each candidate
        best_ei = -1
        best_x = candidates[0]

        for x_cand in candidates:
            ei = self._expected_improvement(x_cand, X, y, best_y)
            if ei > best_ei:
                best_ei = ei
                best_x = x_cand

        return best_x

    def _expected_improvement(self, x: np.ndarray, X: np.ndarray,
                              y: np.ndarray, best_y: float) -> float:
        """Compute Expected Improvement at point x."""
        # Simple GP: RBF kernel
        # Compute mean + std at x using GP regression
        K = self._rbf_kernel(X, X) + 1e-6 * np.eye(len(X))
        K_s = self._rbf_kernel(X, x.reshape(1, -1))
        K_ss = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1))

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu = K_s.flatten() @ alpha
            v = np.linalg.solve(L, K_s.flatten())
            sigma = np.sqrt(max(K_ss[0, 0] - v @ v, 1e-10))
        except np.linalg.LinAlgError:
            return 0.0

        # Expected Improvement
        if self.minimize:
            improvement = best_y - mu
        else:
            improvement = mu - best_y

        # EI = improvement * Phi(z) + sigma * phi(z)
        if sigma < 1e-10:
            return max(improvement, 0)

        z = improvement / sigma
        ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
        return max(ei, 0)

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sq_dist = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_dist / (2 * length_scale**2))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    @staticmethod
    def _normal_pdf(x: float) -> float:
        """Standard normal PDF."""
        return math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

    def _vector_to_params(self, x: np.ndarray, names: List[str]) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(names, x)}

    def _params_to_vector(self, params: Dict[str, float]) -> np.ndarray:
        return np.array([params[n] for n in self.param_space.keys()])


__all__ = ["HyperparameterOptimizer", "OptimizationResult"]
