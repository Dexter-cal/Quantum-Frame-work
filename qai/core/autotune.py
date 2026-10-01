"""
qai.core.autotune -- Automated hyperparameter tuning and model selection.
Exposes AutoTuner and autotune() for exhaustive grid search over technique parameter spaces.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Any
import numpy as np
from .cross_validation import cross_validate


class AutoTuner:
    """Executes automated hyperparameter grid search using k-fold cross-validation."""
    def __init__(self, technique_name: str, param_grid: Dict[str, List[Any]], k: int = 5):
        if not param_grid:
            raise ValueError("param_grid must be a non-empty dictionary of parameter lists")
        self.technique_name = technique_name
        self.param_grid = param_grid
        self.k = k
        self.best_params_ = None
        self.best_score_ = -float("inf")
        self.results_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        keys = list(self.param_grid.keys())
        value_combinations = list(itertools.product(*self.param_grid.values()))

        self.results_ = []
        self.best_score_ = -float("inf")
        self.best_params_ = None

        from .model import build

        for comb in value_combinations:
            params = dict(zip(keys, comb))

            def build_fn(p=params):
                return build(type=self.technique_name, **p)

            cv_res = cross_validate(build_fn, X, y, k=self.k)
            mean_acc = cv_res["mean_accuracy"]
            std_acc = cv_res["std_accuracy"]

            run_entry = {
                "params": params,
                "mean_accuracy": round(float(mean_acc), 4),
                "std_accuracy": round(float(std_acc), 4)
            }
            self.results_.append(run_entry)

            if mean_acc > self.best_score_:
                self.best_score_ = mean_acc
                self.best_params_ = params

        return self

    def best_model(self, X, y):
        if self.best_params_ is None:
            raise RuntimeError("AutoTuner must be fitted before retrieving the best model")
        from .model import build
        model = build(type=self.technique_name, **self.best_params_)
        model.train(X, y, verbose=False)
        return model


def autotune(technique_name: str, param_grid: Dict[str, List[Any]], X, y, k: int = 5):
    """Convenience function for automated hyperparameter tuning."""
    tuner = AutoTuner(technique_name, param_grid, k=k)
    tuner.fit(X, y)
    return tuner
