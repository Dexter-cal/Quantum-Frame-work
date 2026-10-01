"""
qai.core.model — the Model wrapper around a Technique.

This is where the SHARED general methods live (Section 74 of the design
doc): train, predict, accuracy, help, export -- available on every model
regardless of which technique it wraps.
"""
from __future__ import annotations
import time
import json
import numpy as np
from ..techniques import get_technique
from .schema import Schema


class CompatibilityError(ValueError):
    """Raised when a Type and a learning_technique are declared together
    but the Type doesn't actually support that learning_technique."""


class Model:
    def __init__(self, technique_name: str, learning_technique: str = None,
                 input_schema=None, output_schema=None,
                 optimizer=None, objective=None, training_loop=None, **params):
        cls = get_technique(technique_name)

        if learning_technique is not None:
            if learning_technique not in cls.compatible_learning_techniques:
                supported = ", ".join(cls.compatible_learning_techniques)
                raise CompatibilityError(
                    f"[WHAT] Type '{technique_name}' is not compatible with learning_technique '{learning_technique}'\n"
                    f"[WHY] '{technique_name}' only supports: {supported}\n"
                    f"[FIX] Either drop learning_technique, use one of [{supported}], "
                    f"or pick a different Type (e.g. 'tabular_policy' for reinforcement)"
                )
        else:
            learning_technique = cls.compatible_learning_techniques[0]  # sensible default, Section 66

        # Section 56 mechanics decomposition wiring
        if optimizer is not None or objective is not None or training_loop is not None:
            if technique_name == "regression":
                if optimizer is None:
                    from ..mechanics import GradientDescent
                    optimizer = GradientDescent(objective=objective, training_loop=training_loop)
                elif hasattr(optimizer, "objective") and objective is not None:
                    optimizer.objective = objective
                if hasattr(optimizer, "training_loop") and training_loop is not None:
                    optimizer.training_loop = training_loop
                params["optimizer"] = optimizer

        self.technique = cls(**params)
        self.technique_name = technique_name
        self.learning_technique = learning_technique
        self.input_schema = Schema(input_schema) if input_schema and not isinstance(input_schema, Schema) else input_schema
        self.output_schema = Schema(output_schema) if output_schema and not isinstance(output_schema, Schema) else output_schema
        self._history = []
        self._last_input = None
        self._last_output = None

    # --- general methods, every model has these --------------------------
    def train(self, X=None, y=None, environment=None, verbose=True, **kwargs):
        from .dataset import Dataset
        if isinstance(X, Dataset):
            X, y = X.to_arrays()  # a Dataset object flows straight into the same train() everything else uses

        if environment is not None:
            start = time.time()
            self.technique.fit(environment, **kwargs)
            elapsed = time.time() - start
            self._history.append({"event": "train", "mode": "environment", "elapsed_s": round(elapsed, 4)})
            if verbose:
                print(f"[qai] Training complete (via Environment). technique={self.technique_name} time={elapsed:.3f}s")
            return self

        # Handle dict-rows training input
        if isinstance(X, dict):
            if self.input_schema is not None:
                X = self.input_schema.dict_to_array(X)
            else:
                X = np.array(list(X.values()), dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        X = np.asarray(X)
        if self.input_schema is not None:
            self.input_schema.validate(X, context="training input")

        start = time.time()
        self.technique.fit(X, y, **kwargs)
        elapsed = time.time() - start
        self._history.append({"event": "train", "n_samples": len(X), "elapsed_s": round(elapsed, 4)})
        if verbose:
            print(f"[qai] Training complete. technique={self.technique_name} "
                  f"n_samples={len(X)} time={elapsed:.3f}s")
        return self

    def predict(self, x, **kwargs):
        if not self.technique.is_trained():
            raise RuntimeError("predict() called before train() -- model has no learned weights yet")

        if isinstance(x, dict):
            if self.input_schema is not None:
                x_arr = self.input_schema.dict_to_array(x)
            else:
                x_arr = np.array(list(x.values()), dtype=float)
            x_val = x_arr
        else:
            x_val = x

        if self.input_schema is not None:
            self.input_schema.validate(x_val, context="prediction input")

        self._last_input = self._snapshot(x)
        result = self.technique.forward(x_val, **kwargs)

        if self.output_schema is not None:
            self.output_schema.validate(result, context="prediction output")

        self._last_output = self._snapshot(result)
        return result

    def input(self):
        """Return the most recent value supplied to predict()."""
        return self._snapshot(self._last_input)

    def output(self):
        """Return the most recent value produced by predict()."""
        return self._snapshot(self._last_output)

    @staticmethod
    def _snapshot(value):
        """Keep inspection state isolated from mutable NumPy values."""
        if isinstance(value, np.ndarray):
            return value.copy()
        return value

    def accuracy(self, X=None, y=None) -> float:
        if hasattr(self.technique, "accuracy"):
            return self.technique.accuracy()
        if X is not None and y is not None:
            preds = self.predict(X)
            return float(np.mean(np.round(preds) == y))
        raise NotImplementedError(f"{self.technique_name} has no accuracy() and no X/y given")

    def help(self) -> str:
        return self.technique.help()

    def training_status(self) -> dict:
        return {
            "technique": self.technique_name,
            "trained": self.technique.is_trained(),
            "history": self._history,
        }

    def explain(self, x):
        from .explain import explain as _explain
        return _explain(self, x)

    def visualize(self, save_path, X=None):
        from .visualize import visualize as _visualize
        return _visualize(self, save_path, X=X)

    @property
    def weights(self):
        from .weights import WeightAccessor
        return WeightAccessor(self.technique)

    def pipe_to(self, other_model):
        from .pipeline import Pipeline
        if isinstance(self, Pipeline):
            return Pipeline(self.models + [other_model])
        return Pipeline([self, other_model])

    def finetune(self, X_new, y_new=None, verbose=True, **kwargs):
        if not hasattr(self.technique, "finetune"):
            raise NotImplementedError(
                f"[WHAT] '{self.technique_name}' has no finetune() implemented\n"
                f"[WHY] Not every technique has a meaningful notion of incremental improvement "
                f"(e.g. KNN just IS its stored training data -- there's nothing to 'continue')\n"
                f"[FIX] Use model.train() again with the combined old+new data instead, "
                f"or check qai/techniques/{self.technique_name}.py for what's actually possible here"
            )
        start = time.time()
        if y_new is not None:
            self.technique.finetune(X_new, y_new, **kwargs)
        else:
            self.technique.finetune(X_new, **kwargs)
        elapsed = time.time() - start
        self._history.append({"event": "finetune", "elapsed_s": round(elapsed, 4)})
        if verbose:
            print(f"[qai] Finetune complete. technique={self.technique_name} time={elapsed:.3f}s")
        return self

    def export(self, path: str):
        """Simplified stand-in for the real .qmodel export -- serializes
        whatever numeric state the technique has, plus metadata."""
        state = {"technique": self.technique_name, "params": self.technique.params}
        if hasattr(self.technique, "w"):
            state["w"] = self.technique.w.tolist()
            state["b"] = float(self.technique.b)
        with open(path, "w") as f:
            json.dump(state, f)
        print(f"[qai] Exported model -> {path}")
        return path

    # --- pass-through to technique-specific unique tools --------------------
    def __getattr__(self, item):
        return getattr(self.technique, item)

    def __repr__(self):
        return f"<qai.Model technique={self.technique_name} trained={self.technique.is_trained()}>"


def build(type: str, learning_technique: str = None, input_schema=None, output_schema=None,
          optimizer=None, objective=None, training_loop=None, **params) -> Model:
    """qai.build(type='regression', learning_technique='supervised') -- the entry point,
    matching the doc exactly, now with real compatibility checking (Section 66) and Section 56 mechanics."""
    return Model(type, learning_technique=learning_technique, input_schema=input_schema, output_schema=output_schema,
                 optimizer=optimizer, objective=objective, training_loop=training_loop, **params)
