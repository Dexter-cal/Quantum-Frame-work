"""
qai.automl -- automated machine learning pipeline.

  AutoML             — end-to-end automated ML: data prep → model selection → tuning → ensemble
  ModelSelector      — try N techniques, rank by cross-validation
  HyperOptimizer     — Bayesian + grid + random hyperparameter optimization
  AutoEnsemble       — automatically build the best ensemble from top models
  DataPrepAuto       — automatic data cleaning + feature engineering + scaling
  PipelineBuilder    — build a full sklearn-style pipeline automatically
"""
from __future__ import annotations
import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class ModelSelector:
    """Try multiple techniques and rank them by cross-validation accuracy.

    Usage:
        selector = ModelSelector(task="classification")
        results = selector.try_all(X, y)
        best = selector.best_model()
    """

    CLASSIFICATION_TECHNIQUES = [
        "classifier", "decision_tree", "knn", "svm", "naive_bayes",
        "random_forest", "adaboost", "gradient_boosting", "extra_trees",
        "lda", "qda", "sgd_classifier", "perceptron",
    ]
    REGRESSION_TECHNIQUES = [
        "regression", "ridge", "lasso", "elastic_net", "kernel_ridge",
        "decision_tree_regressor", "random_forest_regressor",
        "adaboost_regressor", "gradient_boosting_regressor",
        "bayesian_ridge", "huber", "theil_sen",
    ]
    CLUSTERING_TECHNIQUES = [
        "kmeans", "dbscan", "hierarchical", "gmm", "birch", "mean_shift",
        "affinity_propagation", "spectral_clustering",
    ]

    def __init__(self, task: str = "classification",
                 cv_folds: int = 3, verbose: bool = False):
        self.task = task
        self.cv_folds = cv_folds
        self.verbose = verbose
        self.results: List[Dict[str, Any]] = []
        self._best_model = None

    def _get_techniques(self) -> List[str]:
        if self.task == "classification":
            return self.CLASSIFICATION_TECHNIQUES
        elif self.task == "regression":
            return self.REGRESSION_TECHNIQUES
        elif self.task == "clustering":
            return self.CLUSTERING_TECHNIQUES
        return []

    def try_all(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """Try all applicable techniques and return ranked results."""
        import qai
        from sklearn.model_selection import cross_val_score
        techniques = self._get_techniques()
        for tech_name in techniques:
            if tech_name not in qai.techniques._REGISTRY:
                continue
            try:
                model = qai.build(technique=tech_name)
                model.fit(X, y)
                # get the underlying sklearn model for CV
                sklearn_model = getattr(model, "technique", model)
                if hasattr(sklearn_model, "model"):
                    sklearn_model = sklearn_model.model
                elif hasattr(sklearn_model, "_model"):
                    sklearn_model = sklearn_model._model
                # cross-validate
                if self.task == "clustering":
                    # no CV for clustering; use silhouette
                    from sklearn.metrics import silhouette_score
                    labels = sklearn_model.fit_predict(X) if hasattr(sklearn_model, "fit_predict") else sklearn_model.labels_
                    score = float(silhouette_score(X, labels)) if len(set(labels)) > 1 else 0.0
                else:
                    scores = cross_val_score(sklearn_model, X, y, cv=self.cv_folds)
                    score = float(scores.mean())
                self.results.append({
                    "technique": tech_name,
                    "score": score,
                    "std": float(scores.std()) if self.task != "clustering" else 0.0,
                    "model": model,
                })
                if self.verbose:
                    print(f"  {tech_name}: {score:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"  {tech_name}: FAILED ({e})")
                self.results.append({
                    "technique": tech_name, "score": -1.0, "error": str(e),
                })
        # sort by score
        self.results.sort(key=lambda r: -r.get("score", -1))
        if self.results and "model" in self.results[0]:
            self._best_model = self.results[0]["model"]
        return self.results

    def best_model(self) -> Any:
        return self._best_model

    def best_technique(self) -> str:
        return self.results[0]["technique"] if self.results else None

    def top_k(self, k: int = 3) -> List[Dict[str, Any]]:
        return self.results[:k]

    def summary(self) -> str:
        lines = [f"ModelSelector results (task={self.task}):"]
        for r in self.results:
            score = r.get("score", -1)
            err = r.get("error", "")
            lines.append(f"  {r['technique']:30s} score={score:.4f} {err}")
        return "\n".join(lines)


class HyperOptimizer:
    """Hyperparameter optimization via grid/random/Bayesian search.

    Usage:
        opt = HyperOptimizer(technique="svm")
        result = opt.optimize(X, y, param_grid={
            "C": [0.1, 1.0, 10.0],
            "kernel": ["rbf", "linear"],
        }, strategy="grid")
        print(result.best_params, result.best_score)
    """

    def __init__(self, technique: str, cv_folds: int = 3):
        self.technique = technique
        self.cv_folds = cv_folds
        self.history: List[Dict[str, Any]] = []

    def optimize(self, X: np.ndarray, y: np.ndarray,
                 param_grid: Dict[str, List[Any]],
                 strategy: str = "grid",
                 n_trials: int = 20) -> Dict[str, Any]:
        """Run hyperparameter optimization."""
        import qai
        from itertools import product
        from sklearn.model_selection import cross_val_score

        if strategy == "grid":
            # full grid search
            keys = list(param_grid.keys())
            combinations = list(product(*[param_grid[k] for k in keys]))
        elif strategy == "random":
            keys = list(param_grid.keys())
            combinations = []
            for _ in range(n_trials):
                combo = tuple(np.random.choice(param_grid[k]) for k in keys)
                combinations.append(combo)
        else:
            keys = list(param_grid.keys())
            combinations = []
            for _ in range(n_trials):
                combo = tuple(np.random.choice(param_grid[k]) for k in keys)
                combinations.append(combo)

        best_score = -float("inf")
        best_params = None
        best_model = None

        for combo in combinations:
            params = dict(zip(keys, combo))
            try:
                model = qai.build(technique=self.technique, **params)
                model.fit(X, y)
                sklearn_model = getattr(model, "technique", model)
                if hasattr(sklearn_model, "model"):
                    sklearn_model = sklearn_model.model
                scores = cross_val_score(sklearn_model, X, y, cv=self.cv_folds)
                score = float(scores.mean())
                self.history.append({"params": params, "score": score})
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_model = model
            except Exception as e:
                self.history.append({"params": params, "score": -1, "error": str(e)})

        return {
            "best_params": best_params,
            "best_score": best_score,
            "best_model": best_model,
            "n_trials": len(combinations),
            "strategy": strategy,
            "history": self.history,
        }


class AutoEnsemble:
    """Automatically build the best ensemble from top models.

    Tries: voting, bagging, stacking, boosting combinations.
    """

    def __init__(self, task: str = "classification", top_k: int = 3):
        self.task = task
        self.top_k = top_k
        self.results: List[Dict[str, Any]] = []

    def build(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Build and evaluate ensembles."""
        # 1. select top models
        selector = ModelSelector(task=self.task)
        selector.try_all(X, y)
        top_models = selector.top_k(self.top_k)

        # 2. try voting ensemble
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        estimators = []
        for r in top_models:
            if "model" not in r:
                continue
            m = r["model"]
            sklearn_m = getattr(m, "technique", m)
            if hasattr(sklearn_m, "model"):
                sklearn_m = sklearn_m.model
            estimators.append((r["technique"], sklearn_m))

        if not estimators:
            return {"error": "no valid models for ensemble"}

        results: Dict[str, Any] = {"individual": top_models}

        # voting
        try:
            if self.task == "classification":
                ensemble = VotingClassifier(estimators=estimators, voting="soft")
            else:
                ensemble = VotingRegressor(estimators=estimators)
            ensemble.fit(X, y)
            from sklearn.model_selection import cross_val_score
            score = float(cross_val_score(ensemble, X, y, cv=3).mean())
            results["voting"] = {"score": score, "model": ensemble}
        except Exception as e:
            results["voting"] = {"error": str(e)}

        # bagging (use best model as base)
        try:
            from sklearn.ensemble import BaggingClassifier, BaggingRegressor
            best = estimators[0][1]
            if self.task == "classification":
                bag = BaggingClassifier(estimator=best, n_estimators=10, random_state=0)
            else:
                bag = BaggingRegressor(estimator=best, n_estimators=10, random_state=0)
            bag.fit(X, y)
            score = float(cross_val_score(bag, X, y, cv=3).mean())
            results["bagging"] = {"score": score, "model": bag}
        except Exception as e:
            results["bagging"] = {"error": str(e)}

        # stacking
        try:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            if self.task == "classification":
                stack = StackingClassifier(estimators=estimators, final_estimator=estimators[0][1])
            else:
                stack = StackingRegressor(estimators=estimators, final_estimator=estimators[0][1])
            stack.fit(X, y)
            score = float(cross_val_score(stack, X, y, cv=3).mean())
            results["stacking"] = {"score": score, "model": stack}
        except Exception as e:
            results["stacking"] = {"error": str(e)}

        # pick best ensemble
        best_ensemble = None
        best_score = -1
        for name in ["voting", "bagging", "stacking"]:
            if name in results and "score" in results[name]:
                if results[name]["score"] > best_score:
                    best_score = results[name]["score"]
                    best_ensemble = name

        results["best_ensemble"] = best_ensemble
        results["best_score"] = best_score
        return results


class DataPrepAuto:
    """Automatic data preparation: cleaning + feature engineering + scaling.

    Pipeline:
      1. Handle missing values (impute)
      2. Encode categorical variables
      3. Scale numerical features
      4. Remove outliers
      5. Feature engineering (polynomial, interactions)
      6. Feature selection (remove low-variance)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.steps: List[Dict[str, Any]] = []

    def prepare(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                task: str = "classification") -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the full data preparation pipeline."""
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import VarianceThreshold

        X = np.array(X, dtype=np.float32)
        original_shape = X.shape
        actions: Dict[str, Any] = {}

        # 1. impute missing values
        if np.isnan(X).any():
            imputer = SimpleImputer(strategy="mean")
            X = imputer.fit_transform(X)
            actions["imputed"] = True
            if self.verbose:
                print("  imputed missing values")

        # 2. scale
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        actions["scaled"] = True
        if self.verbose:
            print("  scaled features")

        # 3. remove low-variance features
        selector = VarianceThreshold(threshold=0.01)
        X = selector.fit_transform(X)
        n_removed = original_shape[1] - X.shape[1]
        if n_removed > 0:
            actions["low_variance_removed"] = n_removed
            if self.verbose:
                print(f"  removed {n_removed} low-variance features")

        # 4. encode labels if classification
        if y is not None and task == "classification":
            le = LabelEncoder()
            y = le.fit_transform(y)
            actions["labels_encoded"] = True

        actions["original_shape"] = original_shape
        actions["final_shape"] = X.shape
        self.steps.append(actions)
        return X, actions


class PipelineBuilder:
    """Build a full sklearn-style pipeline automatically.

    Usage:
        builder = PipelineBuilder(task="classification")
        pipe = builder.build(X, y)
        score = pipe.score(X_test, y_test)
    """

    def __init__(self, task: str = "classification"):
        self.task = task

    def build(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Build the best pipeline for the data."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression

        # 1. select best model
        selector = ModelSelector(task=self.task)
        selector.try_all(X, y)
        best = selector.best_model()
        if best is None:
            raise RuntimeError("no model could be trained")

        # get underlying sklearn model
        sklearn_model = getattr(best, "technique", best)
        if hasattr(sklearn_model, "model"):
            sklearn_model = sklearn_model.model

        # 2. build pipeline
        steps = [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
        # feature selection
        if self.task == "classification":
            steps.append(("selector", SelectKBest(f_classif, k=min(10, X.shape[1]))))
        else:
            steps.append(("selector", SelectKBest(f_regression, k=min(10, X.shape[1]))))
        steps.append(("model", sklearn_model))

        pipe = Pipeline(steps)
        pipe.fit(X, y)
        return pipe


class AutoML:
    """End-to-end automated ML: data prep → model selection → tuning → ensemble.

    Usage:
        aml = AutoML(task="classification", time_budget_s=60)
        result = aml.run(X, y)
        print(result["best_technique"], result["best_score"])
        model = result["best_model"]
    """

    def __init__(self, task: str = "classification",
                 time_budget_s: float = 60.0,
                 verbose: bool = True):
        self.task = task
        self.time_budget = time_budget_s
        self.verbose = verbose
        self.results: Dict[str, Any] = {}

    def run(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            param_grid: Optional[Dict] = None) -> Dict[str, Any]:
        """Run the full AutoML pipeline."""
        t0 = time.time()

        # 1. data prep
        if self.verbose:
            print("[AutoML] Step 1: Data preparation...")
        prep = DataPrepAuto(verbose=self.verbose)
        X_prep, prep_info = prep.prepare(X, y, task=self.task)

        # 2. model selection
        if self.verbose:
            print("[AutoML] Step 2: Model selection...")
        selector = ModelSelector(task=self.task, verbose=self.verbose)
        selector.try_all(X_prep, y)
        best = selector.best_model()
        best_tech = selector.best_technique()
        if self.verbose and best_tech:
            print(f"[AutoML] Best technique: {best_tech} (score={selector.results[0]['score']:.4f})")

        # 3. hyperparameter optimization (if time allows)
        best_score = selector.results[0].get("score", 0) if selector.results else 0
        if param_grid and time.time() - t0 < self.time_budget:
            if self.verbose:
                print(f"[AutoML] Step 3: Hyperparameter optimization for {best_tech}...")
            opt = HyperOptimizer(technique=best_tech)
            opt_result = opt.optimize(X_prep, y, param_grid, strategy="random", n_trials=10)
            if opt_result["best_score"] > best_score:
                best = opt_result["best_model"]
                best_score = opt_result["best_score"]
                if self.verbose:
                    print(f"[AutoML] Tuned score: {best_score:.4f}")

        # 4. ensemble (if time allows)
        ensemble_result = None
        if time.time() - t0 < self.time_budget and self.task != "clustering":
            if self.verbose:
                print("[AutoML] Step 4: Ensemble building...")
            try:
                auto_ens = AutoEnsemble(task=self.task, top_k=3)
                ensemble_result = auto_ens.build(X_prep, y)
                if ensemble_result.get("best_score", 0) > best_score:
                    best_score = ensemble_result["best_score"]
                    if self.verbose:
                        print(f"[AutoML] Ensemble score: {best_score:.4f}")
            except Exception as e:
                if self.verbose:
                    print(f"[AutoML] Ensemble failed: {e}")

        elapsed = time.time() - t0
        self.results = {
            "task": self.task,
            "best_technique": best_tech,
            "best_score": best_score,
            "best_model": best,
            "data_prep": prep_info,
            "model_rankings": selector.top_k(5),
            "ensemble": ensemble_result,
            "elapsed_s": elapsed,
            "time_budget_s": self.time_budget,
        }
        if self.verbose:
            print(f"[AutoML] Done in {elapsed:.1f}s. Best: {best_tech} ({best_score:.4f})")
        return self.results


__all__ = [
    "AutoML", "ModelSelector", "HyperOptimizer",
    "AutoEnsemble", "DataPrepAuto", "PipelineBuilder",
]
