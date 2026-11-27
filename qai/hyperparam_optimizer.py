"""
qai.hyperparam_optimizer — THE one Bayesian hyperparameter optimizer.

True Bayesian optimization: a Gaussian-process surrogate (RBF kernel,
Cholesky posterior) + Expected Improvement acquisition. No external
dependencies (no scikit-optimize, no GPyOpt).

One class, two modes — pick by constructor argument:

1. Technique mode — candidates are scored by real k-fold cross-validation
   of a real qai model:

    import qai
    opt = qai.HyperparameterOptimizer(
        technique="random_forest",
        param_space={
            "n_estimators": (10, 200),       # continuous / int range
            "max_depth": [3, 5, 10, None],   # discrete choices
        },
        n_iterations=20, k=3,
    )
    result = opt.optimize(X, y)
    print(result.best_params, result.best_score)

2. Objective mode — you supply the scoring callable (works for anything,
   not just qai techniques):

    opt = qai.HyperparameterOptimizer(
        param_space={"x": (-5.0, 5.0)},
        objective=lambda p: -((p["x"] - 3) ** 2),   # higher = better
        n_iterations=15,
    )
    result = opt.optimize()

Consolidation note (v1.8.7): every "Bayesian" road in qai now leads here —
`qai.bayesian_optimize()`, `AutoTuner(strategy="bayesian")`,
`HyperOptimizer(strategy="bayesian")` and the former duplicate module
`qai.hparam_optimizer` (same class name, second implementation) are all
delegates to this engine. Before v1.8.7 `qai.bayesian_optimize()` was a
random-search stand-in with a fake loss and the "bayesian" strategy of
`HyperOptimizer` was literally the random strategy.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

Space = Union[Tuple[float, float], List[Any]]


@dataclass
class OptimizationResult:
    """Result of a hyperparameter optimization run.

    Attributes:
        best_params: best parameter configuration found (denormalized to
            the types the technique/objective expects).
        best_score: its score (cross-validated accuracy in technique mode,
            the objective value in objective mode).
        n_iterations: number of iterations requested.
        history: one entry per evaluation:
            {"iteration": i, "iter": i, "params": {...}, "score": s[,
             "std": cv_std]}  ("iter" is a compat alias; "std" is present
             in technique mode only).
        method: always "bayesian".
        n_evaluations: number of objective evaluations performed
            (compat alias for len(history)).
        elapsed_seconds: wall-clock time of optimize().
    """
    best_params: Dict[str, Any] = field(default_factory=dict)
    best_score: float = 0.0
    n_iterations: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    method: str = "bayesian"
    n_evaluations: int = 0
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        lines = [
            "=== Optimization Result ===",
            f"best params: {self.best_params}",
            f"best score:  {self.best_score:.4f}",
            f"iterations:  {self.n_iterations} (evaluations: {self.n_evaluations})",
            f"elapsed:     {self.elapsed_seconds:.1f}s",
            "",
            "history (last 5):",
        ]
        for h in self.history[-5:]:
            lines.append(
                f"  iter {h.get('iteration', h.get('iter'))}: "
                f"score={h['score']:.4f} params={h['params']}"
            )
        return "\n".join(lines)


class HyperparameterOptimizer:
    """Bayesian hyperparameter optimization (GP surrogate + Expected Improvement).

    Exactly one of `technique` / `objective` must be given — that choice
    selects the mode:

    - **technique mode**: `technique="random_forest"`, no `objective`.
      Each candidate is evaluated by real k-fold cross-validation via
      `qai.cross_validate`, so the score reflects actual model performance.
      Use `.optimize(X, y)`.

    - **objective mode**: `objective=callable(params_dict) -> float`
      (higher = better unless `minimize=True`). Works for any black-box
      function. Use `.optimize()`.

    Args:
        technique: qai technique name (technique mode).
        param_space: dict of {param_name: (min, max)} for a continuous
            range (int endpoints are rounded back to int) or
            {param_name: [v1, v2, ...]} for discrete choices.
        objective: scoring callable (objective mode).
        n_iterations: total number of evaluations.
        k: cross-validation folds (technique mode).
        n_initial_random: number of random exploration points before the
            GP kicks in (technique mode; `n_initial` is accepted as an
            alias for objective mode).
        n_initial: alias for `n_initial_random` (objective mode).
        minimize: objective mode only — if True, lower objective is better.
        seed: RNG seed for reproducibility (`random_state` accepted as
            alias; None = nondeterministic).

    Raises:
        ValueError: if neither or both of technique/objective are given,
            or the param_space is empty/malformed.
    """
    def __init__(self,
                 technique: Optional[str] = None,
                 param_space: Optional[Dict[str, Space]] = None,
                 objective: Optional[Callable[[Dict[str, Any]], float]] = None,
                 n_iterations: int = 20,
                 k: int = 3,
                 n_initial_random: int = 5,
                 n_initial: Optional[int] = None,
                 minimize: bool = False,
                 seed: Optional[int] = 42,
                 random_state: Optional[int] = None):
        if (technique is None) == (objective is None):
            raise ValueError(
                "[WHAT] HyperparameterOptimizer needs exactly one of "
                "`technique` or `objective`\n"
                "[WHY] The mode is chosen by that argument: technique mode "
                "scores candidates with k-fold CV of a qai model, objective "
                "mode scores them with your callable\n"
                "[FIX] Pass technique='random_forest' (then optimize(X, y)) "
                "OR objective=my_fn (then optimize()) — not both, not neither"
            )
        if not param_space:
            raise ValueError(
                "[WHAT] HyperparameterOptimizer received an empty param_space\n"
                "[WHY] There is nothing to search over\n"
                "[FIX] Pass e.g. {'n_estimators': (10, 200), 'max_depth': [3, 5, 10]}"
            )
        for name, space in param_space.items():
            is_tuple = isinstance(space, tuple) and len(space) == 2
            is_list = isinstance(space, (list,)) and len(space) > 0
            if not (is_tuple or is_list):
                raise ValueError(
                    f"[WHAT] param_space[{name!r}] = {space!r} is not a valid space\n"
                    "[WHY] Spaces are (min, max) tuples or non-empty [choices] lists\n"
                    "[FIX] Use e.g. (10, 200) for a range or [3, 5, 10] for choices"
                )
        if seed is not None and random_state is not None:
            raise ValueError(
                "[WHAT] Both `seed` and `random_state` were given\n"
                "[WHY] They are aliases for the same RNG seed\n"
                "[FIX] Pass only one of them"
            )

        self.technique = technique
        self.objective = objective
        self.param_space = dict(param_space)
        self.n_iterations = int(n_iterations)
        self.k = int(k)
        self.minimize = bool(minimize)
        self.n_initial_random = int(n_initial if n_initial is not None else n_initial_random)
        self.seed = seed if seed is not None else random_state
        self.rng = np.random.default_rng(self.seed)

        self._param_names = list(self.param_space.keys())
        self._bounds = self._compute_bounds()
        # Observations in bound coordinates (discrete lists become indices).
        self._X_observed: List[np.ndarray] = []
        self._y_observed: List[float] = []
        self._std_observed: List[Optional[float]] = []

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def optimize(self, X: Any = None, y: Any = None) -> OptimizationResult:
        """Run the optimization.

        Technique mode requires `X`, `y`; objective mode ignores them.
        Returns an OptimizationResult.
        """
        if self.technique is not None and (X is None or y is None):
            raise ValueError(
                "[WHAT] technique-mode optimize() needs X and y\n"
                "[WHY] Candidates are scored by cross-validating a real "
                "model on your data\n"
                "[FIX] Call opt.optimize(X, y), or construct the optimizer "
                "with objective=... if you score candidates yourself"
            )
        start = time.time()
        self.rng = np.random.default_rng(self.seed)
        self._X_observed, self._y_observed, self._std_observed = [], [], []
        history: List[Dict[str, Any]] = []
        n_initial = max(1, min(self.n_initial_random, self.n_iterations))

        for i in range(self.n_iterations):
            if i < n_initial or len(self._y_observed) < 2:
                vec = self._sample_random()
            else:
                vec = self._suggest_bayesian()
            score, std = self._evaluate(vec, X, y)
            self._X_observed.append(vec)
            self._y_observed.append(score)
            self._std_observed.append(std)
            real_params = self._denormalize(vec)
            entry: Dict[str, Any] = {
                "iteration": i + 1,
                "iter": i + 1,  # compat alias (former qai.hparam_optimizer)
                "params": real_params,
                "score": float(score),
            }
            if std is not None:
                entry["std"] = float(std)
            history.append(entry)

        y_arr = np.asarray(self._y_observed)
        best_idx = int(np.argmin(y_arr)) if self.minimize else int(np.argmax(y_arr))
        elapsed = time.time() - start
        return OptimizationResult(
            best_params=self._denormalize(np.asarray(self._X_observed[best_idx])),
            best_score=float(self._y_observed[best_idx]),
            n_iterations=self.n_iterations,
            history=history,
            method="bayesian",
            n_evaluations=len(history),
            elapsed_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # space helpers
    # ------------------------------------------------------------------
    def _compute_bounds(self) -> List[Tuple[float, float]]:
        bounds = []
        for name in self._param_names:
            space = self.param_space[name]
            if isinstance(space, tuple):
                bounds.append((float(space[0]), float(space[1])))
            else:
                bounds.append((0.0, float(len(space) - 1)))
        return bounds

    def _sample_random(self) -> np.ndarray:
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])
        return self.rng.uniform(lo, hi)

    def _denormalize(self, vec: np.ndarray) -> Dict[str, Any]:
        """Map a bound-coordinate vector back to real parameter values."""
        result: Dict[str, Any] = {}
        for name, val in zip(self._param_names, np.asarray(vec, dtype=float)):
            space = self.param_space[name]
            if isinstance(space, tuple):
                if isinstance(space[0], int) and isinstance(space[1], int):
                    result[name] = int(round(float(val)))
                else:
                    result[name] = float(val)
            else:
                idx = int(round(float(val)))
                idx = max(0, min(idx, len(space) - 1))
                result[name] = space[idx]
        return result

    # ------------------------------------------------------------------
    # evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, vec: np.ndarray, X: Any, y: Any) -> Tuple[float, Optional[float]]:
        """Evaluate one candidate. Returns (score, cv_std_or_None)."""
        real_params = self._denormalize(vec)
        if self.objective is not None:
            # Objective mode: let user-code exceptions propagate — hiding
            # bugs in the user's own objective is worse than crashing.
            score = self.objective(real_params)
            return float(score), None
        # Technique mode: real k-fold CV; invalid param combos score 0.0
        # instead of aborting the whole search.
        import qai
        try:
            def build_fn():
                return qai.build(technique=self.technique, **real_params)
            results = qai.cross_validate(build_fn, X, y, k=self.k, verbose=False)
            return float(results["mean_accuracy"]), float(results.get("std_accuracy", 0.0) or 0.0)
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # GP surrogate + Expected Improvement
    # ------------------------------------------------------------------
    def _suggest_bayesian(self) -> np.ndarray:
        """Maximize Expected Improvement over random candidates using the
        GP posterior (RBF kernel, Cholesky solve)."""
        X_obs = np.asarray(self._X_observed, dtype=float)
        y_obs = np.asarray(self._y_observed, dtype=float)
        # Internal GP always maximizes; flip sign when the user minimizes.
        gp_y = -y_obs if self.minimize else y_obs
        best_y = float(gp_y.max())
        lo = np.array([b[0] for b in self._bounds])
        hi = np.array([b[1] for b in self._bounds])

        best_ei, best_vec = -float("inf"), self._sample_random()
        for _ in range(200):
            cand = self.rng.uniform(lo, hi)
            ei = self._expected_improvement(cand, X_obs, gp_y, best_y)
            if ei > best_ei:
                best_ei, best_vec = ei, cand
        return best_vec

    def _expected_improvement(self, x: np.ndarray, X: np.ndarray,
                              y: np.ndarray, best_y: float) -> float:
        """EI at candidate x under the RBF-kernel GP posterior."""
        K = self._rbf_kernel(X, X) + 1e-6 * np.eye(len(X))
        K_s = self._rbf_kernel(X, x.reshape(1, -1))
        K_ss = self._rbf_kernel(x.reshape(1, -1), x.reshape(1, -1))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            mu = float(K_s.flatten() @ alpha)
            v = np.linalg.solve(L, K_s.flatten())
            sigma = math.sqrt(max(K_ss[0, 0] - float(v @ v), 1e-10))
        except np.linalg.LinAlgError:
            return 0.0

        improvement = mu - best_y  # internal GP is always maximize
        if sigma < 1e-10:
            return max(improvement, 0.0)
        z = improvement / sigma
        ei = improvement * self._normal_cdf(z) + sigma * self._normal_pdf(z)
        return max(ei, 0.0)

    @staticmethod
    def _rbf_kernel(X1: np.ndarray, X2: np.ndarray,
                    length_scale: float = 1.0) -> np.ndarray:
        """RBF (squared exponential) kernel."""
        sq = (np.sum(X1 ** 2, axis=1).reshape(-1, 1)
              + np.sum(X2 ** 2, axis=1)
              - 2.0 * X1 @ X2.T)
        return np.exp(-np.maximum(sq, 0.0) / (2.0 * length_scale ** 2))

    @staticmethod
    def _normal_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    @staticmethod
    def _normal_pdf(x: float) -> float:
        return math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)


__all__ = ["HyperparameterOptimizer", "OptimizationResult"]
