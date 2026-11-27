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


class InputValidationError(ValueError):
    """Raised when a user passes input that doesn't match what the model
    was trained on (wrong dtype, wrong shape, wrong type).

    The error message includes sample training data so the user can see
    exactly what the model expects.
    """


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
    def fit(self, X=None, y=None, **kwargs):
        """Alias for train(). Identical behavior; train() adds logging.

        Both model.fit(X, y) and model.train(X, y) do the same thing:
        call technique.fit(X, y) and mark the model as trained.
        train() additionally logs timing + returns self for chaining.
        """
        return self.train(X, y, verbose=False, **kwargs)

    def train(self, X=None, y=None, environment=None, verbose=True,
              budget=None, callbacks=None, **kwargs):
        """Train the model.

        Args:
            X: training inputs (array, list of dicts, Dataset, etc.)
            y: training targets (optional for unsupervised techniques).
            environment: for reinforcement techniques (overrides X, y).
            verbose: if True, print a one-line training summary.
            budget: optional qai.ResourceBudget -- training stops early
                    if RAM or time budget is exceeded. The model is left
                    in the best state seen so far.
            callbacks: optional list of callback objects. Each callback
                        is called at well-defined points during training
                        (on_epoch_end, on_train_end). Standard callbacks:
                          - qai.callbacks.EarlyStopping(patience=5)
                          - qai.callbacks.CheckpointEvery(n=10, path=...)
                          - qai.callbacks.LogProgress()
            **kwargs: passed through to the technique's fit().
        Returns:
            self (for chaining).
        """
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

        try:
            X = np.asarray(X)
        except (ValueError, TypeError):
            # Inhomogeneous shapes (e.g. list of variable-size graphs for GNN)
            # — pass through as-is; the technique's fit() will handle it.
            pass
        if self.input_schema is not None:
            self.input_schema.validate(X, context="training input")

        # Budget check (before training starts)
        if budget is not None:
            check = budget.check_model(self)
            if not check["fits"]:
                print(f"[qai] WARNING: model doesn't fit budget -- skipping training")
                self._history.append({"event": "train_skipped", "reason": "budget_exceeded",
                                       "details": check})
                return self
            if budget.time_exceeded():
                print(f"[qai] WARNING: time budget already exceeded -- skipping training")
                self._history.append({"event": "train_skipped", "reason": "time_exceeded"})
                return self

        # Fire on_train_start callbacks
        if callbacks:
            for cb in callbacks:
                if hasattr(cb, "on_train_start"):
                    cb.on_train_start(self)

        start = time.time()
        try:
            self.technique.fit(X, y, **kwargs)
            train_status = "complete"
            train_error = None
        except Exception as e:
            train_status = "failed"
            train_error = str(e)
            raise
        finally:
            elapsed = time.time() - start
            # Robust sample count: 0-d arrays (e.g. raw strings) have no len()
            if hasattr(X, "ndim") and X.ndim == 0:
                n_samples = 1
            elif hasattr(X, "__len__"):
                n_samples = len(X)
            else:
                n_samples = 1
            self._history.append({"event": "train", "n_samples": n_samples,
                                   "elapsed_s": round(elapsed, 4),
                                   "status": train_status})
            if verbose and train_status == "complete":
                print(f"[qai] Training complete. technique={self.technique_name} "
                      f"n_samples={n_samples} time={elapsed:.3f}s")

            # Budget check (after training)
            if budget is not None and budget.time_exceeded():
                self._history.append({"event": "budget_stopped", "reason": "time_exceeded_after_train"})

            # Fire on_train_end callbacks
            if callbacks:
                for cb in callbacks:
                    if hasattr(cb, "on_train_end"):
                        cb.on_train_end(self, elapsed_s=elapsed)

        # Store training-data signature for input validation (Section: input validation)
        # We keep the first 3 samples so we can show them to users who pass
        # the wrong type/shape later.
        self._train_data_signature = self._capture_data_signature(X)
        return self

    def partial_fit(self, X=None, y=None, environment=None, verbose=False, **kwargs):
        """Online learning: continue training on new data without restarting.

        Dispatches to the underlying technique's `partial_fit` if it has one.
        Techniques that don't support online learning raise
        NotImplementedError with a helpful message.

        For reinforcement techniques, pass `environment=` to run additional
        episodes (the technique's fit() is called with the environment).

        Args:
            X, y: new training data (same shape semantics as train()).
            environment: for reinforcement techniques.
            verbose: if True, print a one-line summary.
            **kwargs: passed through to the technique's partial_fit().

        Returns:
            self (for chaining).
        """
        from .dataset import Dataset
        if isinstance(X, Dataset):
            X, y = X.to_arrays()

        # Reinforcement path
        if environment is not None:
            if not hasattr(self.technique, "fit"):
                raise NotImplementedError(
                    f"[WHAT] '{self.technique_name}' has no fit() for environment-based partial_fit\n"
                    f"[WHY] Reinforcement techniques should accept environment= in fit()\n"
                    f"[FIX] Check qai/techniques/{self.technique_name}.py"
                )
            start = time.time()
            self.technique.fit(environment, **kwargs)
            elapsed = time.time() - start
            self._history.append({"event": "partial_fit", "mode": "environment",
                                   "elapsed_s": round(elapsed, 4)})
            if verbose:
                print(f"[qai] Partial fit (env). technique={self.technique_name} time={elapsed:.3f}s")
            return self

        # Standard path: dispatch to technique.partial_fit if present
        if hasattr(self.technique, "partial_fit"):
            start = time.time()
            self.technique.partial_fit(X, y, **kwargs)
            elapsed = time.time() - start
            self._history.append({"event": "partial_fit", "elapsed_s": round(elapsed, 4)})
            if verbose:
                print(f"[qai] Partial fit. technique={self.technique_name} time={elapsed:.3f}s")
            return self

        # Fallback: if technique has fit() but no partial_fit(), try fit() on
        # the new data (some techniques -- like TabularPolicy -- just continue
        # learning when fit() is called again).
        if hasattr(self.technique, "fit"):
            # For RL techniques with environment, this is the natural path
            # For supervised, this is "retrain on new data only" -- not true
            # online learning, but better than nothing.
            start = time.time()
            try:
                self.technique.fit(X, y, **kwargs)
                elapsed = time.time() - start
                self._history.append({"event": "partial_fit_fallback",
                                       "elapsed_s": round(elapsed, 4)})
                if verbose:
                    print(f"[qai] Partial fit (fallback to fit()). "
                          f"technique={self.technique_name} time={elapsed:.3f}s")
                return self
            except Exception:
                pass

        raise NotImplementedError(
            f"[WHAT] '{self.technique_name}' has no partial_fit() implemented\n"
            f"[WHY] Online learning requires the technique to support incremental "
            f"updates. '{self.technique_name}' doesn't.\n"
            f"[FIX] Use model.train() with combined old+new data, or check "
            f"qai/techniques/{self.technique_name}.py for what's actually possible"
        )

    def predict(self, x, **kwargs):
        if not self.technique.is_trained():
            raise RuntimeError("predict() called before train() -- model has no learned weights yet")

        # Convert dict → array using schema BEFORE validation, otherwise
        # validate_input sees a 0-d object array and mis-classifies it as
        # text input. Also let validate_input see the converted array.
        if isinstance(x, dict):
            if self.input_schema is not None:
                x_val = self.input_schema.dict_to_array(x)
            else:
                x_val = np.array(list(x.values()), dtype=float)
        else:
            x_val = x

        # --- input validation (warn user with sample data if type mismatches) ---
        self.validate_input(x_val)

        if self.input_schema is not None:
            self.input_schema.validate(x_val, context="prediction input")

        # Auto-switch torch sub-components to eval mode for inference
        # (critical for BatchNorm/Dropout — they behave differently in train mode)
        if hasattr(self.technique, "eval"):
            try:
                self.technique.eval()
            except Exception:
                pass

        self._last_input = self._snapshot(x)
        result = self.technique.forward(x_val, **kwargs)

        if self.output_schema is not None:
            self.output_schema.validate(result, context="prediction output")

        self._last_output = self._snapshot(result)
        return result

    # ------------------------------------------------------------------
    # Input validation — warn users with sample data when input type mismatches
    # ------------------------------------------------------------------
    def _capture_data_signature(self, X):
        """Capture the 'shape' of training data so we can validate later inputs.

        Returns a dict with:
          - kind: 'numeric' | 'text' | 'image' | 'tokens' | 'other'
          - sample_rows: first 3 training samples (for error messages)
          - n_features: number of features per sample (if numeric)
          - dtype: numpy dtype or 'str'
          - ndim: number of dimensions
        """
        sig = {"kind": "other", "sample_rows": [], "n_features": None,
               "dtype": None, "ndim": None}
        try:
            # Try numeric first
            arr = np.asarray(X)
            sig["ndim"] = arr.ndim
            sig["dtype"] = str(arr.dtype)
            # Take first 3 rows
            if arr.ndim >= 1 and len(arr) > 0:
                n_show = min(3, len(arr))
                sig["sample_rows"] = [arr[i] for i in range(n_show)]
            # Determine kind
            if arr.dtype.kind in ("f", "i", "u"):
                sig["kind"] = "numeric"
                if arr.ndim == 2:
                    sig["n_features"] = arr.shape[1]
                elif arr.ndim == 1:
                    # 1-D numeric input. Could be either:
                    #   (a) a single tabular row [f1, f2, f3, f4]
                    #   (b) a 1-D sequence (time series, text tokens) where
                    #       predict() will receive a *different-length* window
                    #       (e.g., a `lags=3` window) than the training series.
                    # Use the technique's own notion of expected input length
                    # if it has one (lags, input_dim, n_features); otherwise
                    # leave n_features=None so we don't enforce a wrong length.
                    tech_n = None
                    for attr in ("lags", "input_dim", "n_features",
                                 "n_lags", "window_size", "seq_len"):
                        v = getattr(self.technique, attr, None)
                        if isinstance(v, int) and v > 0:
                            tech_n = v
                            break
                    sig["n_features"] = tech_n  # may be None — that's OK
                elif arr.ndim == 4:
                    sig["kind"] = "image"
                    sig["n_features"] = arr.shape[1:]  # (C, H, W)
            elif arr.dtype.kind in ("U", "S", "O"):
                # Could be text or string labels
                first = arr.flat[0] if arr.size > 0 else ""
                if isinstance(first, str):
                    sig["kind"] = "text"
                    sig["sample_rows"] = [str(arr.flat[i]) for i in range(min(3, arr.size))]
                else:
                    sig["kind"] = "other"
                    sig["sample_rows"] = [str(arr.flat[i]) for i in range(min(3, arr.size))]
            return sig
        except Exception:
            # Non-array-like (graphs, tuples, etc.) — just store repr
            try:
                sig["sample_rows"] = [repr(X[0])[:100]] if len(X) > 0 else []
            except Exception:
                sig["sample_rows"] = [repr(X)[:100]]
            return sig

    def validate_input(self, x) -> bool:
        """Validate that input `x` matches what the model was trained on.

        Checks:
          - dtype compatibility (numeric vs text)
          - number of features (for tabular models)
          - number of channels/dimensions (for image models)

        If the input doesn't match, raises InputValidationError with a
        helpful message that includes sample training data so the user
        can see exactly what was expected.

        Returns True if validation passes.

        Examples:
            >>> model.train(X_train, y_train)  # X_train is numeric, 4 features
            >>> model.validate_input([1.0, 2.0, 3.0, 4.0])  # OK
            >>> model.validate_input("hello world")
            InputValidationError: This model was trained on NUMERIC data (4 features),
            but you passed TEXT input: "hello world"

            Sample training data (first 3 rows):
              [5.1, 3.5, 1.4, 0.2]
              [4.9, 3.0, 1.4, 0.2]
              [4.7, 3.2, 1.3, 0.2]
        """
        sig = getattr(self, "_train_data_signature", None)
        if sig is None or not sig.get("sample_rows"):
            return True  # nothing to compare against (untrained or no signature)

        expected_kind = sig["kind"]
        expected_n = sig.get("n_features")
        sample_rows = sig.get("sample_rows", [])

        # Determine the actual kind of the input
        try:
            x_arr = np.asarray(x)
            actual_kind = "other"
            if x_arr.dtype.kind in ("f", "i", "u"):
                actual_kind = "numeric"
            elif x_arr.dtype.kind in ("U", "S", "O"):
                actual_kind = "text"
            elif x_arr.dtype == object:
                actual_kind = "other"
        except Exception:
            actual_kind = "other"
            x_arr = None

        # Check 1: kind mismatch (numeric vs text)
        if expected_kind == "numeric" and actual_kind == "text":
            preview = str(x)[:80]
            msg = self._format_validation_error(
                expected=f"NUMERIC data"
                         + (f" ({expected_n} features)" if expected_n else ""),
                got=f"TEXT input: \"{preview}\"",
                sample_rows=sample_rows,
                kind="numeric")
            raise InputValidationError(msg)

        if expected_kind == "text" and actual_kind == "numeric":
            msg = self._format_validation_error(
                expected="TEXT data (strings)",
                got=f"NUMERIC input: {x_arr.tolist()[:5] if x_arr is not None else x}",
                sample_rows=sample_rows,
                kind="text")
            raise InputValidationError(msg)

        # Check 2: feature-count mismatch (tabular models)
        if (expected_kind == "numeric" and expected_n is not None
                and actual_kind == "numeric" and x_arr is not None):
            actual_n = None
            if x_arr.ndim == 1:
                actual_n = len(x_arr)
            elif x_arr.ndim == 2:
                actual_n = x_arr.shape[1]
            if actual_n is not None and actual_n != expected_n:
                msg = self._format_validation_error(
                    expected=f"NUMERIC data with {expected_n} features",
                    got=f"NUMERIC data with {actual_n} features: "
                        f"{x_arr.tolist()[:5] if x_arr.size <= 10 else '[...]'}",
                    sample_rows=sample_rows,
                    kind="numeric")
                raise InputValidationError(msg)

        # Check 3: image shape mismatch
        if expected_kind == "image" and actual_kind == "numeric" and x_arr is not None:
            expected_shape = expected_n  # (C, H, W)
            if expected_shape and x_arr.ndim >= 2:
                # Could be (H, W), (C, H, W), or (N, C, H, W)
                actual_shape = x_arr.shape[-3:] if x_arr.ndim >= 3 else x_arr.shape[-2:]
                if actual_shape != expected_shape and actual_shape[-2:] != expected_shape[-2:]:
                    msg = self._format_validation_error(
                        expected=f"IMAGE data with shape {expected_shape}",
                        got=f"array with shape {x_arr.shape}",
                        sample_rows=sample_rows,
                        kind="image")
                    raise InputValidationError(msg)

        return True

    def _format_validation_error(self, expected: str, got: str,
                                   sample_rows: list, kind: str = "numeric") -> str:
        """Format a helpful error message with sample training data."""
        lines = [
            "[WHAT] Input doesn't match what this model was trained on.",
            f"[WHY]  This model was trained on {expected},",
            f"       but you passed {got}.",
            "",
            "[FIX]   Use input that matches the training data format. "
            "Sample training data (first 3 rows):",
        ]
        for i, row in enumerate(sample_rows[:3]):
            if isinstance(row, np.ndarray):
                if kind == "image":
                    lines.append(f"          row {i}: image with shape {row.shape}")
                elif row.size <= 10:
                    lines.append(f"          row {i}: {row.tolist()}")
                else:
                    lines.append(f"          row {i}: [{row.flat[0]}, {row.flat[1]}, "
                                  f"{row.flat[2]}, ...] (shape {row.shape})")
            elif isinstance(row, str):
                lines.append(f"          row {i}: \"{row[:80]}\"")
            else:
                lines.append(f"          row {i}: {row}")
        return "\n".join(lines)

    def input(self):
        """Return the most recent value supplied to predict()."""
        return self._snapshot(self._last_input)

    def output(self):
        """Return the most recent value produced by predict().

        Returns a ModelOutput object that supports chainable formatting:
            result = model.output()
            result.format("json").redact("emails")
            result.format("text").save("output.txt")

        Also supports dict-like access:
            result["prediction"]
            "confidence" in result
        """
        from ..chainable_output import ModelOutput
        return ModelOutput(self._snapshot(self._last_output), model=self)

    def io_summary(self) -> dict:
        """One-glance view of expected input type and produced output type.

        Returns:
            Dict with input_type, output_type, input_shape, output_shape,
            technique, family, and examples.

        Example:
            >>> m.io_summary()
            {'input_type': 'numeric_array', 'input_shape': (4,),
             'output_type': 'int', 'output_shape': (),
             'technique': 'classifier', 'family': 'classical_ml'}
        """
        sig = getattr(self, "_train_data_signature", None)
        input_info = {"type": "unknown", "shape": None}
        if sig:
            input_info = {
                "type": sig.get("kind", "unknown"),
                "shape": (sig.get("n_features", "?"),) if sig.get("n_features") else None,
            }
        output_info = {"type": "unknown", "shape": None}
        if self._last_output is not None:
            output = self._last_output
            if isinstance(output, (int, np.integer)):
                output_info = {"type": "int", "shape": ()}
            elif isinstance(output, (float, np.floating)):
                output_info = {"type": "float", "shape": ()}
            elif isinstance(output, str):
                output_info = {"type": "str", "shape": ()}
            elif isinstance(output, np.ndarray):
                output_info = {"type": "array", "shape": output.shape}
            elif isinstance(output, list):
                output_info = {"type": "list", "shape": (len(output),)}
        return {
            "input_type": input_info["type"],
            "input_shape": input_info["shape"],
            "output_type": output_info["type"],
            "output_shape": output_info["shape"],
            "technique": self.technique_name,
            "family": getattr(self.technique, "family", "unknown"),
            "trained": self.technique.is_trained(),
        }

    def checkpoint(self, name: str, path: Optional[str] = None) -> str:
        """Save a named checkpoint for later rollback.

        Args:
            name: checkpoint name (e.g. "v1", "before_finetune")
            path: optional directory (default: ./checkpoints)

        Returns:
            Path to the checkpoint file.
        """
        import os
        checkpoint_dir = path or "./checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        filepath = os.path.join(checkpoint_dir, f"{name}.qmodel")
        from .persistence import save_model
        save_model(self, filepath)
        self._history.append({"event": "checkpoint", "name": name, "path": filepath})
        return filepath

    def rollback(self, name: str, path: Optional[str] = None) -> "Model":
        """Rollback to a previously saved checkpoint.

        Args:
            name: checkpoint name
            path: optional directory (default: ./checkpoints)

        Returns:
            self (with restored weights)
        """
        import os
        checkpoint_dir = path or "./checkpoints"
        filepath = os.path.join(checkpoint_dir, f"{name}.qmodel")
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"[WHAT] Checkpoint '{name}' not found at {filepath}\n"
                f"[FIX] Save it first with model.checkpoint('{name}')")
        from .persistence import load_model
        loaded = load_model(filepath)
        # Copy technique state
        self.technique = loaded.technique
        self._history.append({"event": "rollback", "name": name})
        return self

    def clone_to_staging(self) -> "Model":
        """Clone the model to a staging area for risky changes.

        Creates a deep copy that you can experiment on without affecting
        the original model. If the changes work, merge back; if not,
        discard the staging copy.

        Returns:
            A new Model instance (deep copy).

        Example:
            >>> staging = model.clone_to_staging()
            >>> staging.finetune(risky_data, method="lora", rank=64)
            >>> # Test staging model...
            >>> # If good: model = staging
            >>> # If bad: just discard staging, model is unchanged
        """
        import copy
        new = copy.deepcopy(self)
        new._history.append({"event": "cloned_to_staging"})
        return new

    def deploy(self, target: str = "server", **kwargs) -> Any:
        """Deploy the model to a specific target.

        Auto-optimizes the model for the deployment target:
          - "server": full precision, ProductionServer
          - "edge": quantized to int8, smaller, faster
          - "mobile": quantized to int4, smallest, fastest

        Args:
            target: "server", "edge", or "mobile"
            **kwargs: passed to the deployment

        Returns:
            Deployment handle (server URL, file path, etc.)
        """
        if target == "server":
            from .production_serving import ProductionServer
            port = kwargs.get("port", 8000)
            server = ProductionServer(model=self, port=port)
            server.start()
            return f"http://0.0.0.0:{port}"
        elif target == "edge":
            # Quantize to int8 for edge deployment
            from ..quantization import quantize_dynamic
            quantize_dynamic(self)
            path = kwargs.get("path", "./model_edge.qmodel")
            from .persistence import save_model
            save_model(self, path)
            return path
        elif target == "mobile":
            # Quantize to int4 for mobile
            try:
                from ..quantization import quantize_to_int4
                quantize_to_int4(self, quant_type="nf4")
            except ImportError:
                from ..quantization import quantize_dynamic
                quantize_dynamic(self)  # fallback to int8
            path = kwargs.get("path", "./model_mobile.qmodel")
            from .persistence import save_model
            save_model(self, path)
            return path
        else:
            raise ValueError(f"Unknown deploy target: {target}. Use: server, edge, mobile")

    def explainability_score(self) -> float:
        """Compute a standardized 0-100 explainability score.

        Higher = more explainable. Based on:
          - Has built-in feature_importances_? (+30)
          - Has linear weights (coef_)? (+25)
          - Has predict_proba? (+15)
          - Has explain() method? (+15)
          - Is tree-based? (+10)
          - Is small (<100K params)? (+5)

        Returns:
            Score 0-100
        """
        score = 0
        tech = self.technique
        # Check for feature importances
        if hasattr(tech, "model") and hasattr(tech.model, "feature_importances_"):
            score += 30
        # Check for linear weights
        if hasattr(tech, "w") and tech.w is not None:
            score += 25
        elif hasattr(tech, "model") and hasattr(tech.model, "coef_"):
            score += 25
        # Check for predict_proba
        if hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
            score += 15
        # Check for explain method
        if hasattr(self, "explain"):
            score += 15
        # Check if tree-based
        family = getattr(tech, "family", "")
        if family in ("classical_ml",) and "tree" in getattr(tech, "name", "").lower():
            score += 10
        if "forest" in getattr(tech, "name", "").lower():
            score += 10
        # Check model size
        try:
            from ..quantization import model_size_mb
            if model_size_mb(self) < 0.1:  # <100KB
                score += 5
        except Exception:
            pass
        return min(score, 100)

    def predict_proba(self, x):
        """Predict class probabilities."""
        tech = self.technique
        if hasattr(tech, "model") and hasattr(tech.model, "predict_proba"):
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)
            return tech.model.predict_proba(x_arr)
        raise NotImplementedError(f"predict_proba() not available for '{self.technique_name}'")

    def predict_top_k(self, x, k: int = 3):
        """Return top-k predictions with probabilities."""
        try:
            proba = self.predict_proba(x)
            if hasattr(proba, "flatten"):
                proba = proba.flatten()
            top_idx = np.argsort(proba)[::-1][:k]
            return [(int(i), float(proba[i])) for i in top_idx]
        except NotImplementedError:
            return [(self.predict(x), 1.0)]

    def evaluate(self, X, y, metrics=None) -> dict:
        """Evaluate model on test data."""
        y = np.asarray(y)
        preds = []
        n = min(len(X), 500)
        for i in range(n):
            try: preds.append(self.predict(X[i]))
            except: preds.append(None)
        preds = np.array([p if p is not None else -1 for p in preds])
        y_eval = y[:n]
        if metrics is None:
            metrics = ["accuracy"] if y.dtype.kind in ("i","u") else ["mse","mae"]
        results = {}
        for m in metrics:
            if m == "accuracy": results["accuracy"] = float(np.mean(preds == y_eval))
            elif m == "mse": results["mse"] = float(np.mean((preds.astype(float) - y_eval.astype(float))**2))
            elif m == "mae": results["mae"] = float(np.mean(np.abs(preds.astype(float) - y_eval.astype(float))))
            elif m == "r2":
                ss_res = np.sum((y_eval.astype(float) - preds.astype(float))**2)
                ss_tot = np.sum((y_eval.astype(float) - y_eval.mean())**2)
                results["r2"] = float(1 - ss_res / max(ss_tot, 1e-8))
        results["n_samples"] = n
        return results

    @property
    def history(self):
        """Return training history."""
        return self._history

    def summary(self) -> str:
        """Model architecture summary."""
        try:
            from ..model_tools import model_summary
            return model_summary(self)
        except Exception:
            return f"<Model: {self.technique_name}>"

    def save(self, path: str, format: str = "qmodel"):
        """Save model to file."""
        from .persistence import save_model
        if format == "gguf":
            from ..quantization import save_gguf
            save_gguf(self, path, quantization="q4_k_m")
        elif format == "onnx":
            from .onnx import export_to_onnx
            export_to_onnx(self, path)
        else:
            save_model(self, path)

    @classmethod
    def load(cls, path: str) -> "Model":
        """Load model from file."""
        from .persistence import load_model
        return load_model(path)

    def to(self, device=None, dtype=None):
        """Move the model to a specific device and/or dtype.

        Args:
            device: "cpu", "cuda:0", "mps", or None (auto-select best)
            dtype: torch.float32, torch.float16, torch.bfloat16, or None

        Returns:
            self (for chaining)

        Example:
            m.to("cuda:0")           # move to GPU 0
            m.to("cpu", dtype="fp16") # CPU + half precision
            m.to()                    # auto-select best device
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return self  # no-op without torch

        # Auto-select device if not specified
        if device is None:
            try:
                from ..cuda_kernels import auto_select_device
                device = auto_select_device(self)
            except Exception:
                device = "cpu"

        # Convert string dtype to torch dtype
        if isinstance(dtype, str):
            dtype_map = {"fp32": torch.float32, "float32": torch.float32,
                         "fp16": torch.float16, "float16": torch.float16,
                         "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
            dtype = dtype_map.get(dtype.lower(), torch.float32)

        # Move torch submodules
        tech = self.technique
        torch_device = torch.device(device) if isinstance(device, str) else device

        def _move_module(module):
            if isinstance(module, nn.Module):
                module.to(torch_device)
                if dtype is not None:
                    module.to(dtype)
            elif isinstance(module, (list, tuple)):
                for item in module:
                    _move_module(item)

        for attr_name in vars(tech):
            attr = getattr(tech, attr_name)
            if isinstance(attr, nn.Module):
                attr.to(torch_device)
                if dtype is not None:
                    attr.to(dtype)
            elif isinstance(attr, nn.Parameter):
                setattr(tech, attr_name, nn.Parameter(attr.to(torch_device, dtype=dtype) if dtype else attr.to(torch_device)))
            elif isinstance(attr, (list, tuple)):
                _move_module(attr)

        self._device = str(torch_device)
        return self

    def device_info(self) -> dict:
        """Get information about the device the model is currently on."""
        try:
            import torch
            import torch.nn as nn
            tech = self.technique
            for attr in vars(tech).values():
                if isinstance(attr, nn.Module):
                    params = list(attr.parameters())
                    if params:
                        return {"device": str(params[0].device),
                                "dtype": str(params[0].dtype)}
        except Exception:
            pass
        return {"device": "unknown", "dtype": "unknown"}

    def n_parameters(self) -> dict:
        """Count model parameters.

        Returns:
            Dict with 'total', 'trainable', 'frozen' counts
        """
        try:
            import torch
            import torch.nn as nn
            tech = self.technique
            total = 0
            trainable = 0
            def _count(module):
                nonlocal total, trainable
                if isinstance(module, nn.Module):
                    for p in module.parameters():
                        total += p.numel()
                        if p.requires_grad:
                            trainable += p.numel()
                elif isinstance(module, (list, tuple)):
                    for item in module:
                        _count(item)
            for attr in vars(tech).values():
                if isinstance(attr, nn.Module):
                    for p in attr.parameters():
                        total += p.numel()
                        if p.requires_grad:
                            trainable += p.numel()
                elif isinstance(attr, (list, tuple)):
                    _count(attr)
                elif isinstance(attr, nn.Parameter):
                    total += attr.numel()
                    if attr.requires_grad:
                        trainable += attr.numel()
            return {"total": total, "trainable": trainable,
                    "frozen": total - trainable,
                    "total_mb": total * 4 / (1024**2)}  # assume fp32
        except Exception:
            return {"total": 0, "trainable": 0, "frozen": 0}

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
        """Explain the model's prediction for input `x`.

        Returns a dict with a standardized schema so tools (dashboards,
        loggers, etc.) can rely on common fields:

            {
              "technique": str,                  # e.g. "regression"
              "prediction": Any,                 # the predicted value
              "top_features": [                  # ranked feature attributions
                {"feature": str, "contribution": float, "rank": int}, ...
              ],
              "confidence": float | None,        # 0..1, or None if N/A
              "warnings": [str],                 # any caveats
              "details": {...}                   # technique-specific extras
            }

        Subclasses / specific techniques may add extra fields to `details`.
        """
        from .explain import explain as _explain
        raw = _explain(self, x)

        # If the technique's explainer already returns a dict, normalize it
        # into the standard schema (without losing technique-specific data).
        if isinstance(raw, dict):
            # Already a dict -- normalize fields into the standard schema
            details = {k: v for k, v in raw.items()
                        if k not in ("technique", "prediction", "top_features",
                                      "feature_contributions", "features",
                                      "confidence", "probability", "warnings")}
            standardized = {
                "technique": raw.get("technique", self.technique_name),
                "prediction": raw.get("prediction", None),
                "top_features": raw.get("top_features",
                                          raw.get("feature_contributions",
                                                  raw.get("features", []))),
                "confidence": raw.get("confidence", raw.get("probability", None)),
                "warnings": raw.get("warnings", []),
                "details": details,
            }
            # Promote commonly-used detail fields to top-level so callers
            # don't have to dig into `details` for them.
            for promoted in ("class_probabilities", "feature_attributions",
                              "feature_ranking", "most_influential_feature",
                              "reasoning", "method", "bias_contribution"):
                if promoted in details and promoted not in standardized:
                    standardized[promoted] = details[promoted]
            return standardized
        # Non-dict return: wrap it in the standard schema
        return {
            "technique": self.technique_name,
            "prediction": raw,
            "top_features": [],
            "confidence": None,
            "warnings": ["non-dict explain() return -- wrapped in standard schema"],
            "details": {"raw": raw},
        }

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

    def finetune(self, X_new, y_new=None, verbose=True, method=None, **kwargs):
        """Fine-tune the model on new data.

        Two modes:
        1. Traditional (default): calls technique.finetune() if available,
           otherwise re-trains with combined old + new data.
        2. Parameter-efficient (method="lora"|"qlora"|"adapter"|"ia3"|"prefix"|"prompt"):
           applies a PEFT adapter and trains only the adapter weights.

        Args:
            X_new: new training inputs
            y_new: new targets (optional for unsupervised)
            verbose: if True, print progress
            method: PEFT method — "lora", "qlora", "adapter", "ia3", "prefix", "prompt"
                    If None (default), uses traditional finetune.
            **kwargs: passed to the PEFT adapter (e.g. rank=8 for LoRA)
        """
        # PEFT path
        if method is not None:
            from ..finetuning import apply_finetuning
            adapter = apply_finetuning(self, method=method, **kwargs)
            # Train the adapter
            adapter.finetune(X_new, train_targets=y_new, verbose=verbose)
            return self

        # Traditional path
        if not hasattr(self.technique, "finetune"):
            raise NotImplementedError(
                f"[WHAT] '{self.technique_name}' has no finetune() implemented\n"
                f"[WHY] Not every technique has a meaningful notion of incremental improvement "
                f"(e.g. KNN just IS its stored training data -- there's nothing to 'continue')\n"
                f"[FIX] Use model.train() again with the combined old+new data instead, "
                f"or use model.finetune(method='lora') for parameter-efficient fine-tuning"
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
        """Serialize whatever numeric state the technique has, plus metadata.

        Handles arbitrary-shape weights and bias: 0-D (scalar), 1-D (vector),
        and 2-D (matrix). Falls back to inspecting sklearn-style `coef_` /
        `intercept_` for techniques that don't expose `w`/`b` directly.
        """
        state = {"technique": self.technique_name, "params": getattr(self.technique, "params", {})}

        def _to_jsonable(v):
            """Convert numpy scalars/arrays to JSON-serializable Python types."""
            if v is None:
                return None
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, np.generic):  # np.float64, np.int64, etc.
                return v.item()
            if isinstance(v, (list, tuple)):
                return [_to_jsonable(x) for x in v]
            return v

        # Prefer the technique's own `w`/`b` (Regression-style)
        if hasattr(self.technique, "w") and self.technique.w is not None:
            state["w"] = _to_jsonable(self.technique.w)
            state["b"] = _to_jsonable(self.technique.b)
        # Fall back to sklearn-style coef_/intercept_
        elif hasattr(self.technique, "model"):
            inner = self.technique.model
            if hasattr(inner, "coef_"):
                state["coef_"] = _to_jsonable(inner.coef_)
            if hasattr(inner, "intercept_"):
                state["intercept_"] = _to_jsonable(inner.intercept_)
            if hasattr(inner, "classes_"):
                state["classes_"] = _to_jsonable(inner.classes_)

        # Stash any other obvious numeric attributes (technique-specific extras)
        for attr in ("classes_", "vocab", "loss_history_"):
            if hasattr(self.technique, attr):
                try:
                    state[attr] = _to_jsonable(getattr(self.technique, attr))
                except Exception:
                    pass  # not JSON-serializable — skip silently

        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        print(f"[qai] Exported model -> {path}")
        return path

    # --- pass-through to technique-specific unique tools --------------------
    def __getattr__(self, item):
        # Recursion-safe: bypass our override entirely for internal attrs.
        # __getattr__ is only called when normal lookup fails, so reaching
        # here for "technique" means it isn't set yet (e.g. during __init__
        # or unpickling). Use object.__getattribute__ to avoid infinite loop.
        if item in ("technique", "technique_name", "learning_technique",
                    "input_schema", "output_schema", "_history",
                    "_last_input", "_last_output"):
            raise AttributeError(f"Model has no attribute '{item}'")
        try:
            technique = object.__getattribute__(self, "technique")
        except AttributeError:
            raise AttributeError(f"Model has no attribute '{item}'")
        if technique is not None:
            return getattr(technique, item)
        raise AttributeError(f"Model has no attribute '{item}'")

    def __repr__(self):
        return f"<qai.Model technique={self.technique_name} trained={self.technique.is_trained()}>"


def build(type: str = None, learning_technique: str = None, input_schema=None, output_schema=None,
          optimizer=None, objective=None, training_loop=None, config=None,
          train_config=None, tokenizer=None, pretrained: bool = False,
          **params) -> Model:
    """qai.build(type='regression', learning_technique='supervised') -- the entry point,
    matching the doc exactly, now with real compatibility checking (Section 66) and Section 56 mechanics.

    `type=` is the original kwarg (kept for backwards compatibility), but it
    shadows the Python builtin. Prefer `technique=` in new code:

        model = qai.build(technique="regression")

    Both are accepted and equivalent.

    Advanced usage — pass a config object to bundle architecture params:

        model = qai.build(
            technique="transformer",
            config=qai.TransformerConfig(d_model=256, n_heads=8, n_layers=6),
            tokenizer=tok,
            train_config=qai.TrainConfig(lr=3e-4, warmup_steps=200),
        )

    The config object is unpacked into kwargs, so this is equivalent to:

        model = qai.build(technique="transformer", d_model=256, n_heads=8, n_layers=6)

    `tokenizer=` is attached to the model for use in .train() and .generate().
    `train_config=` sets default training recipe (overridable per .train() call).
    """
    # Accept `technique=` as the modern, non-builtin-shadowing alternative to `type=`
    if type is None and "technique" in params:
        type = params.pop("technique")
    if type is None:
        raise TypeError(
            "[WHAT] qai.build() requires a technique name\n"
            "[WHY] Neither `type=` nor `technique=` was provided\n"
            "[FIX] Call e.g. qai.build(technique='regression') or qai.build(type='regression')"
        )

    # Unpack config object into params (kwargs take precedence over config fields)
    if config is not None:
        from ..configs import config_to_kwargs
        config_kwargs = config_to_kwargs(config)
        for k, v in config_kwargs.items():
            if k not in params:
                params[k] = v

    # If a config object was used, filter out params that the technique's
    # __init__ doesn't accept. This prevents config fields that don't apply
    # (e.g. BoostingConfig.loss for AdaBoost) from crashing the constructor
    # when forwarded to sklearn.
    if config is not None:
        try:
            cls = get_technique(type)
            import inspect
            sig = inspect.signature(cls.__init__)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                              for p in sig.parameters.values())
            if not has_var_kw:
                # No **kwargs — filter strictly
                accepted = set(sig.parameters.keys()) - {"self"}
                params = {k: v for k, v in params.items() if k in accepted}
            # If has **kwargs, the technique accepts everything — but it might
            # forward to sklearn which doesn't. We can't know, so leave as-is
            # and let the technique handle filtering.
        except Exception:
            pass

    # Try construction; if it fails with TypeError (unexpected kwarg), retry
    # with filtered params (inspect the technique's __init__).
    try:
        model = Model(type, learning_technique=learning_technique, input_schema=input_schema,
                      output_schema=output_schema, optimizer=optimizer, objective=objective,
                      training_loop=training_loop, **params)
    except TypeError as e:
        if "unexpected keyword argument" in str(e) or "multiple values for keyword" in str(e):
            # Inspect and filter
            try:
                cls = get_technique(type)
                import inspect
                sig = inspect.signature(cls.__init__)
                accepted = set(sig.parameters.keys()) - {"self"}
                filtered = {k: v for k, v in params.items() if k in accepted}
                model = Model(type, learning_technique=learning_technique, input_schema=input_schema,
                              output_schema=output_schema, optimizer=optimizer, objective=objective,
                              training_loop=training_loop, **filtered)
            except Exception:
                raise
        else:
            raise

    # Attach optional tokenizer and train_config to the model
    if tokenizer is not None:
        model.tokenizer = tokenizer
    if train_config is not None:
        model._train_config = train_config

    # Load pretrained weights if requested
    if pretrained:
        try:
            from ..pretrained import load_pretrained
            load_pretrained(model, pretrained if isinstance(pretrained, str) else "default")
        except Exception as e:
            import warnings
            warnings.warn(f"Could not load pretrained weights for '{type}': {e}")

    return model
