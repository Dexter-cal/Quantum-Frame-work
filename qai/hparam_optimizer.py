"""
qai.hparam_optimizer — DELEGATE module (since v1.8.7).

This module used to carry a second, independently-written
`HyperparameterOptimizer` (Gaussian Process + Expected Improvement) —
the same feature as `qai.hyperparam_optimizer`, same class name, different
file. That is exactly the fragmentation the framework is eliminating:
one feature, one implementation, one entry point.

The canonical implementation now lives in `qai.hyperparam_optimizer`
(one class, two modes: technique+CV and objective). This module is kept
only so existing imports keep working:

    from qai.hparam_optimizer import HyperparameterOptimizer   # unchanged

The canonical class is a superset: it supports continuous ranges AND
discrete choice lists, both `optimize(X, y)` and `optimize()`, plus
`minimize`, seeding, and CV std in the history. The old objective-mode
constructor shape (`param_space=`, `objective=`, `n_initial=`,
`n_iterations=`, `minimize=`, `seed=`) is fully preserved.
"""
from qai.hyperparam_optimizer import HyperparameterOptimizer, OptimizationResult

__all__ = ["HyperparameterOptimizer", "OptimizationResult"]
