"""v1.8.7 — One Bayesian optimizer: consolidation regression tests.

Guards the consolidation of the five scattered "Bayesian" implementations:
  1. qai.bayesian_optimate() was a random-search stand-in with a fake
     loss (np.random.uniform(0.01, 0.5) — never touched the model).
  2. AutoTuner(strategy="bayesian") was an explore/exploit grid heuristic.
  3. qai.HyperparameterOptimizer (qai.hyperparam_optimizer) — the
     exported GP+EI engine, now THE canonical one.
  4. qai.hparam_optimizer carried a second class with the SAME NAME
     (objective-mode GP+EI); it is now a delegate shim.
  5. HyperOptimizer(strategy="bayesian") in qai.automl was literally the
     same code as strategy="random".

After v1.8.7 all five roads lead to one engine (RBF-kernel GP posterior +
Expected Improvement), and every candidate in technique mode is scored by
real k-fold cross-validation of a real qai model.
"""
import numpy as np
import pytest

import qai
import qai.hparam_optimizer as hparam_mod
from qai.core.autotune import AutoTuner
from qai.automl import HyperOptimizer
from qai.hyperparam_optimizer import HyperparameterOptimizer


def _make_data(n=120, seed=7):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [3.0, 3.0], [0.0, 3.0], [3.0, 0.0]])
    X = np.vstack([centers[i % 4] + rng.normal(0, 0.5, size=2) for i in range(n)])
    y = np.array([i % 4 for i in range(n)])
    return X, y


# ---------------------------------------------------------------------------
# 1. qai.bayesian_optimize() — the public entry is now real
# ---------------------------------------------------------------------------
def test_bayesian_optimize_returns_measured_score():
    X, y = _make_data()
    res = qai.bayesian_optimize(
        "random_forest", X, y,
        {"n_estimators": (10, 60), "max_depth": (2, 8)},
        n_trials=6, k=2, random_state=42,
    )
    # backward-compatible keys survive
    assert "best_params" in res and "best_loss" in res
    # plus the new honest fields
    assert res["method"] == "bayesian"
    assert res["n_trials"] == 6
    assert len(res["history"]) == 6
    # the score is a real measured accuracy, not a random draw
    assert 0.0 <= res["best_score"] <= 1.0
    assert abs(res["best_loss"] - (1.0 - res["best_score"])) < 1e-12
    # the reported best is the best among recorded evaluations — the loss
    # comes from the model, not from the dice
    assert res["best_score"] == pytest.approx(
        max(h["score"] for h in res["history"]))


def test_bayesian_optimize_is_deterministic_with_seed():
    X, y = _make_data()
    kw = dict(n_trials=6, k=2, random_state=42)
    a = qai.bayesian_optimize("random_forest", X, y,
                              {"n_estimators": (10, 60), "max_depth": (2, 8)}, **kw)
    b = qai.bayesian_optimize("random_forest", X, y,
                              {"n_estimators": (10, 60), "max_depth": (2, 8)}, **kw)
    assert a["best_params"] == b["best_params"]
    assert a["best_score"] == pytest.approx(b["best_score"])


def test_bayesian_optimize_discrete_choices():
    X, y = _make_data()
    choices = {"n_estimators": [5, 10, 20], "max_depth": [2, 4, 8]}
    res = qai.bayesian_optimize("random_forest", X, y, choices,
                                n_trials=5, k=2, random_state=0)
    assert res["best_params"]["n_estimators"] in choices["n_estimators"]
    assert res["best_params"]["max_depth"] in choices["max_depth"]


def test_bayesian_optimize_empty_bounds_raises():
    X, y = _make_data()
    with pytest.raises(ValueError, match="param_bounds"):
        qai.bayesian_optimize("random_forest", X, y, {})


# ---------------------------------------------------------------------------
# 3. canonical engine — objective mode (pure math, proves real optimization)
# ---------------------------------------------------------------------------
def test_objective_mode_finds_maximum():
    opt = HyperparameterOptimizer(
        param_space={"x": (0.0, 10.0)},
        objective=lambda p: -((p["x"] - 3.0) ** 2),
        n_iterations=15,
    )
    res = opt.optimize()
    assert abs(res.best_params["x"] - 3.0) < 1.0
    assert res.best_score > -1.0  # max of -(x-3)^2 is 0 at x=3
    assert res.n_evaluations == 15
    assert res.n_iterations == 15
    assert res.elapsed_seconds >= 0.0
    assert res.method == "bayesian"


def test_objective_mode_minimize():
    opt = HyperparameterOptimizer(
        param_space={"x": (0.0, 10.0)},
        objective=lambda p: (p["x"] - 3.0) ** 2,
        n_iterations=15, minimize=True,
    )
    res = opt.optimize()
    assert abs(res.best_params["x"] - 3.0) < 1.0
    assert res.best_score < 1.0


def test_objective_mode_old_signature_preserved():
    # the pre-consolidation qai.hparam_optimizer constructor shape
    opt = HyperparameterOptimizer(
        param_space={"a": (0.0, 2.0), "b": (0.0, 2.0)},
        objective=lambda p: -(p["a"] - 1.0) ** 2 - (p["b"] - 0.5) ** 2,
        n_initial=4, n_iterations=10, minimize=False, seed=7,
    )
    res = opt.optimize()
    assert abs(res.best_params["a"] - 1.0) < 1.0
    assert abs(res.best_params["b"] - 0.5) < 1.0
    # "iter" compat alias is still in every history entry
    assert all("iter" in h for h in res.history)


def test_engine_technique_mode_contract():
    # the exact shape tests/test_new_features.py pins
    X, y = _make_data()
    opt = HyperparameterOptimizer(
        technique="random_forest",
        param_space={"n_estimators": (10, 50)},
        n_iterations=5, k=2, n_initial_random=3,
    )
    res = opt.optimize(X, y)
    assert res.best_score > 0
    assert "n_estimators" in res.best_params
    assert isinstance(res.best_params["n_estimators"], int)
    assert res.n_iterations == 5
    assert len(res.history) == 5


def test_engine_technique_mode_needs_data():
    opt = HyperparameterOptimizer(
        technique="random_forest", param_space={"n_estimators": (10, 50)})
    with pytest.raises(ValueError, match="X and y"):
        opt.optimize()


def test_engine_mode_validation():
    X, y = _make_data()
    obj = lambda p: 0.5
    # both modes at once -> error
    with pytest.raises(ValueError, match="exactly one"):
        HyperparameterOptimizer(technique="random_forest", objective=obj,
                                param_space={"n_estimators": (10, 50)})
    # neither mode -> error
    with pytest.raises(ValueError, match="exactly one"):
        HyperparameterOptimizer(param_space={"x": (0, 1)})
    # empty space -> error
    with pytest.raises(ValueError, match="param_space"):
        HyperparameterOptimizer(objective=obj, param_space={})
    # invalid space shape -> error
    with pytest.raises(ValueError, match="valid space"):
        HyperparameterOptimizer(objective=obj, param_space={"x": 5})
    # both seed aliases -> error
    with pytest.raises(ValueError, match="aliases"):
        HyperparameterOptimizer(objective=obj, param_space={"x": (0, 1)},
                                seed=1, random_state=2)


# ---------------------------------------------------------------------------
# 4. the duplicate module is now a delegate of the canonical class
# ---------------------------------------------------------------------------
def test_duplicate_module_is_delegate():
    assert hparam_mod.HyperparameterOptimizer is qai.HyperparameterOptimizer
    assert hparam_mod.OptimizationResult is qai.OptimizationResult


# ---------------------------------------------------------------------------
# 2. AutoTuner(strategy="bayesian") delegates to the same engine
# ---------------------------------------------------------------------------
def test_autotuner_bayesian_delegates():
    X, y = _make_data()
    tuner = AutoTuner("random_forest",
                      {"n_estimators": [5, 15, 30], "max_depth": [2, 5]},
                      strategy="bayesian", n_trials=5, k=2,
                      random_state=42).fit(X, y)
    assert tuner.best_params_ is not None
    assert tuner.best_technique_ == "random_forest"
    assert 0.0 < tuner.best_score_ <= 1.0
    assert len(tuner.results_) == 5
    for r in tuner.results_:
        assert "mean_accuracy" in r and "std_accuracy" in r
        assert r["params"]["n_estimators"] in (5, 15, 30)


# ---------------------------------------------------------------------------
# 5. HyperOptimizer(strategy="bayesian") is no longer the random branch
# ---------------------------------------------------------------------------
def test_hyperoptimizer_bayesian_delegates():
    X, y = _make_data()
    opt = HyperOptimizer(technique="random_forest", cv_folds=2)
    res = opt.optimize(X, y,
                       param_grid={"n_estimators": [5, 15],
                                   "max_depth": [2, 5]},
                       strategy="bayesian", n_trials=6)
    assert res["strategy"] == "bayesian"
    assert res["n_trials"] == 6
    assert 0.0 <= res["best_score"] <= 1.0
    assert len([h for h in res["history"] if "error" not in h]) == 6
    assert res["best_params"]["n_estimators"] in (5, 15)
    # best_model is fitted and predicts
    assert res["best_model"] is not None
    preds = res["best_model"].predict(X[:5])
    assert len(preds) == 5


def test_hyperoptimizer_unknown_strategy_raises():
    X, y = _make_data()
    opt = HyperOptimizer(technique="random_forest", cv_folds=2)
    with pytest.raises(ValueError, match="unknown strategy"):
        opt.optimize(X, y, param_grid={"n_estimators": [5]},
                     strategy="magic", n_trials=2)


# ---------------------------------------------------------------------------
# integration: all roads reach the same engine
# ---------------------------------------------------------------------------
def test_all_entries_share_one_engine(monkeypatch):
    X, y = _make_data()
    calls = []

    import qai.hyperparam_optimizer as canon
    orig_optimize = canon.HyperparameterOptimizer.optimize

    def spy(self, *a, **kw):
        calls.append(1)
        return orig_optimize(self, *a, **kw)

    monkeypatch.setattr(canon.HyperparameterOptimizer, "optimize", spy)

    qai.bayesian_optimize("random_forest", X, y,
                          {"n_estimators": [5, 20]}, n_trials=3, k=2)
    AutoTuner("random_forest", {"n_estimators": [5, 20]},
              strategy="bayesian", n_trials=3, k=2).fit(X, y)
    HyperOptimizer(technique="random_forest", cv_folds=2).optimize(
        X, y, param_grid={"n_estimators": [5, 20]},
        strategy="bayesian", n_trials=3)

    assert len(calls) == 3, "every bayesian road must delegate to the canonical engine"
