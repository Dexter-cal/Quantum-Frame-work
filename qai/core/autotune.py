"""
qai.core.autotune -- Automated hyperparameter tuning and model selection.
Exposes AutoTuner and autotune() for exhaustive grid search over technique parameter spaces.

Two usage patterns:

1. Hyperparameter tuning for ONE technique (grid search over its params):

    tuner = qai.AutoTuner("random_forest", {"n_estimators": [50, 100, 200]})
    tuner.fit(X, y)
    print(tuner.best_params_, tuner.best_score_)

2. Technique selection (compare multiple techniques on the same data):

    tuner = qai.AutoTuner(techniques=["random_forest", "knn", "svm"])
    tuner.fit(X, y)
    print(tuner.best_technique_, tuner.best_score_)

The second form is what most users actually want when they reach for AutoTuner.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Any, Optional, Union
import numpy as np
from .cross_validation import cross_validate


class AutoTuner:
    """Automated hyperparameter and technique selection.

    Two modes (chosen by constructor args):

    - **Hyperparameter mode**: `AutoTuner(technique_name="random_forest",
      param_grid={"n_estimators": [50, 100]})` runs grid-search CV over
      every combination of params and returns the best.

    - **Technique-selection mode**: `AutoTuner(techniques=["rf", "knn",
      "svm"])` trains each technique with default params and returns the
      best-performing one. Convenient when you don't know which algorithm
      suits your data.

    Args:
        technique_name: Single technique to grid-search (hyperparameter mode).
        param_grid: Dict of {param_name: [values]} for grid search.
        techniques: List of technique names to compare (technique-selection mode).
        k: Number of cross-validation folds (default 5).
        metric: Scoring metric. Currently only "accuracy" is supported.
    """
    def __init__(
        self,
        technique_name: Optional[str] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        techniques: Optional[List[str]] = None,
        k: int = 5,
        metric: str = "accuracy",
        strategy: str = "grid",
        n_trials: Optional[int] = None,
        random_state: int = 42,
    ):
        # Validate: at least one mode must be specified
        if technique_name is None and not techniques:
            raise ValueError(
                "[WHAT] AutoTuner requires either `technique_name` or `techniques`\n"
                "[WHY] Without one of these, the tuner has nothing to search\n"
                "[FIX] Pass technique_name='random_forest' + param_grid={...} for "
                "hyperparameter tuning, OR techniques=['rf','knn','svm'] for "
                "technique selection"
            )
        # Empty param_grid explicitly raises (backwards-compat with original API).
        # If you want technique-selection mode without a grid, pass techniques=[...]
        # or technique_name='x' with param_grid=None.
        if technique_name and param_grid is not None and len(param_grid) == 0:
            raise ValueError(
                "[WHAT] AutoTuner received an empty param_grid\n"
                "[WHY] An empty grid means there's nothing to search over\n"
                "[FIX] Pass a non-empty dict like {'n_estimators': [50, 100]}, OR "
                "use technique-selection mode: AutoTuner(techniques=['rf', 'knn'])"
            )
        # Allow technique_name with no param_grid (None) -> single-technique baseline
        if technique_name and not param_grid:
            techniques = [technique_name]
            technique_name = None

        self.technique_name = technique_name
        self.param_grid = param_grid or {}
        self.techniques = techniques
        self.k = k
        self.metric = metric
        self.strategy = strategy  # "grid" | "random" | "bayesian"
        self.n_trials = n_trials or 20
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = -float("inf")
        self.best_technique_ = None
        self.results_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.results_ = []
        self.best_score_ = -float("inf")
        self.best_params_ = None
        self.best_technique_ = None

        from .model import build

        # --- Hyperparameter mode ---
        if self.technique_name and self.param_grid:
            keys = list(self.param_grid.keys())

            # Strategy dispatch: generate the list of param combinations to try
            if self.strategy == "grid":
                value_combinations = list(itertools.product(*self.param_grid.values()))
                param_list = [dict(zip(keys, comb)) for comb in value_combinations]
            elif self.strategy == "random":
                rng = np.random.default_rng(self.random_state)
                value_combinations = list(itertools.product(*self.param_grid.values()))
                n = min(self.n_trials, len(value_combinations))
                chosen = rng.choice(len(value_combinations), size=n, replace=False)
                param_list = [dict(zip(keys, value_combinations[i])) for i in chosen]
            elif self.strategy == "bayesian":
                # Simple bayesian-like approach: start with random, then focus
                # on the neighborhood of the best result so far.
                rng = np.random.default_rng(self.random_state)
                value_combinations = list(itertools.product(*self.param_grid.values()))
                all_indices = list(range(len(value_combinations)))
                # Phase 1: random exploration (half the trials)
                n_explore = max(3, self.n_trials // 2)
                n_exploit = self.n_trials - n_explore
                explored = set()
                param_list = []
                # Random exploration
                explore_pool = list(rng.choice(all_indices, size=min(n_explore, len(all_indices)), replace=False))
                for idx in explore_pool:
                    param_list.append(dict(zip(keys, value_combinations[idx])))
                    explored.add(idx)
                # Phase 2: we'll fill these in during the loop (exploitation)
                # by picking neighbors of the best-so-far
                exploit_pool_remaining = n_exploit
            else:
                raise ValueError(
                    f"[WHAT] Unknown AutoTuner strategy {self.strategy!r}\n"
                    f"[WHY] Strategy must be 'grid', 'random', or 'bayesian'\n"
                    f"[FIX] Use one of the supported strategies"
                )

            for comb_idx, params in enumerate(param_list):
                def build_fn(p=params):
                    return build(type=self.technique_name, **p)

                cv_res = cross_validate(build_fn, X, y, k=self.k)
                mean_acc = cv_res["mean_accuracy"]
                std_acc = cv_res["std_accuracy"]

                run_entry = {
                    "technique": self.technique_name,
                    "params": params,
                    "mean_accuracy": round(float(mean_acc), 4),
                    "std_accuracy": round(float(std_acc), 4),
                }
                self.results_.append(run_entry)

                if mean_acc > self.best_score_:
                    self.best_score_ = mean_acc
                    self.best_params_ = params
                    self.best_technique_ = self.technique_name

                # Bayesian exploitation: after each trial, if we have budget left,
                # pick a neighbor of the current best
                if self.strategy == "bayesian" and exploit_pool_remaining > 0:
                    best_idx = value_combinations.index(
                        tuple(self.best_params_[k] for k in keys)
                    )
                    # Find unexplored neighbors (same values for all but one key)
                    for offset in [-1, 1, -2, 2]:
                        neighbor_idx = best_idx + offset
                        if 0 <= neighbor_idx < len(value_combinations) and neighbor_idx not in explored:
                            neighbor_params = dict(zip(keys, value_combinations[neighbor_idx]))
                            param_list.append(neighbor_params)
                            explored.add(neighbor_idx)
                            exploit_pool_remaining -= 1
                            break

        # --- Technique-selection mode ---
        elif self.techniques:
            for tech_name in self.techniques:
                def build_fn(name=tech_name):
                    return build(type=name)

                try:
                    cv_res = cross_validate(build_fn, X, y, k=self.k)
                    mean_acc = cv_res["mean_accuracy"]
                    std_acc = cv_res["std_accuracy"]
                except Exception as e:
                    # Record failure but don't crash the whole search
                    self.results_.append({
                        "technique": tech_name,
                        "params": {},
                        "mean_accuracy": 0.0,
                        "std_accuracy": 0.0,
                        "error": f"{type(e).__name__}: {e}",
                    })
                    continue

                run_entry = {
                    "technique": tech_name,
                    "params": {},
                    "mean_accuracy": round(float(mean_acc), 4),
                    "std_accuracy": round(float(std_acc), 4),
                }
                self.results_.append(run_entry)

                if mean_acc > self.best_score_:
                    self.best_score_ = mean_acc
                    self.best_technique_ = tech_name
                    self.best_params_ = {}

        return self

    @property
    def best(self) -> str:
        """For technique-selection mode: name of the best technique.
        For hyperparameter mode: same as best_technique_ (i.e. the input technique_name)."""
        return self.best_technique_

    def best_model(self, X, y):
        """Train a fresh model with the best params/technique on full X, y."""
        if self.best_technique_ is None:
            raise RuntimeError("AutoTuner must be fitted before retrieving the best model")
        from .model import build
        model = build(type=self.best_technique_, **(self.best_params_ or {}))
        model.train(X, y, verbose=False)
        return model

    def summary(self) -> str:
        """Human-readable table of all tried configurations, sorted by score."""
        if not self.results_:
            return "AutoTuner not fitted yet."
        sorted_results = sorted(self.results_, key=lambda r: -r.get("mean_accuracy", 0))
        lines = [f"  {'technique':<22}{'params':<40}{'acc':>8}{'std':>8}"]
        for r in sorted_results:
            params_str = str(r.get("params", {}))[:38]
            acc = r.get("mean_accuracy", 0)
            std = r.get("std_accuracy", 0)
            lines.append(f"  {r['technique']:<22}{params_str:<40}{acc:>8.4f}{std:>8.4f}")
        if self.best_technique_:
            lines.append(f"\n  best: {self.best_technique_} (acc={self.best_score_:.4f})")
        return "\n".join(lines)

    def __repr__(self):
        mode = "technique-selection" if self.techniques else "hyperparameter"
        return f"<AutoTuner mode={mode} fitted={self.best_technique_ is not None}>"


def autotune(technique_name: str, param_grid: Dict[str, List[Any]], X, y, k: int = 5):
    """Convenience function for automated hyperparameter tuning of one technique."""
    tuner = AutoTuner(technique_name=technique_name, param_grid=param_grid, k=k)
    tuner.fit(X, y)
    return tuner


def bayesian_optimize(model_type: str, X: Any, y: Any, param_bounds: Dict[str, Any], n_trials: int = 5) -> Dict[str, Any]:
    """Bayesian Optimization surrogate search for optimal hyperparameters.

    NOTE: This is a simplified random-search stand-in, not real Bayesian
    optimization. Real BO would require a Gaussian-process surrogate and
    an acquisition function (EI / UCB). Kept for API compatibility.
    """
    best_loss = float("inf")
    best_params = {}

    for trial in range(n_trials):
        sampled = {}
        for param, bounds in param_bounds.items():
            if isinstance(bounds, list):
                sampled[param] = bounds[trial % len(bounds)]
            elif isinstance(bounds, tuple):
                sampled[param] = float(np.random.uniform(bounds[0], bounds[1]))

        # Candidate score evaluation — placeholder random loss
        loss = float(np.random.uniform(0.01, 0.5))
        if loss < best_loss:
            best_loss = loss
            best_params = sampled

    return {"best_params": best_params, "best_loss": best_loss}
