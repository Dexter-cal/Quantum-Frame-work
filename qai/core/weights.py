"""
qai.core.weights -- Section 4/78/82: real weight inspection and direct
editing, not just documentation.

Works for any technique exposing a real coefficient vector or matrix:
- Regression's `w` (1-D) + `b` (scalar)
- Softmax regression / neural-net output layer (2-D `w` + 1-D `b`)
- Any sklearn linear model with `coef_` (1-D or 2-D) + `intercept_`

The `get(row, col=None)` API is explicit about its 2-D-ness: passing only
`row` returns a whole row (as a list); passing both returns a single float.
Use `flat_index(i)` for 1-D-style access on flattened arrays.
"""
from __future__ import annotations
import numpy as np


class WeightAccessor:
    """model.weights -- a real, live view onto a technique's actual
    numbers, with get/set that genuinely mutate the underlying model."""

    def __init__(self, technique):
        self.technique = technique

    def _get_coef_and_bias(self):
        """Returns (coef, bias, kind) where kind is 'w' or 'sklearn'."""
        if hasattr(self.technique, "w") and self.technique.w is not None:
            return self.technique.w, self.technique.b, "w"  # Regression's own attributes
        if hasattr(self.technique, "model") and hasattr(self.technique.model, "coef_"):
            coef = self.technique.model.coef_
            # sklearn linear models: coef_ may be 2-D (n_classes, n_features) for classifiers
            if coef.ndim > 1 and coef.shape[0] == 1:
                coef = coef[0]  # squeeze binary-classifier shape (1, n) -> (n,)
            return coef, self.technique.model.intercept_, "sklearn"
        raise NotImplementedError(
            f"[WHAT] No inspectable weight vector for technique '{self.technique.name}'\n"
            f"[WHY] The technique doesn't expose `w`/`b` or sklearn-style `coef_`/`intercept_`\n"
            f"[FIX] Use a different technique, or add a `w` attribute to "
            f"qai/techniques/{self.technique.name}.py"
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def as_table(self):
        """Return a flat list of {name, value} dicts for all weights+bias.

        For 2-D weight matrices, each entry is `w{row}_{col}`. Useful for
        serializing to CSV / printing a human-readable table."""
        coef, bias, _ = self._get_coef_and_bias()
        coef = np.asarray(coef)
        rows = []
        if coef.ndim == 1:
            for i, w in enumerate(coef):
                rows.append({"name": f"w{i}", "value": float(w)})
        elif coef.ndim == 2:
            n_rows, n_cols = coef.shape
            for i in range(n_rows):
                for j in range(n_cols):
                    rows.append({"name": f"w{i}_{j}", "value": float(coef[i, j])})
        # Bias: may be scalar or 1-D
        bias_arr = np.asarray(bias).flatten()
        if bias_arr.size == 1:
            rows.append({"name": "bias", "value": float(bias_arr[0])})
        else:
            for i, b in enumerate(bias_arr):
                rows.append({"name": f"b{i}", "value": float(b)})
        return rows

    def shape(self):
        """Return the shape of the weight matrix (excluding bias)."""
        coef, _, _ = self._get_coef_and_bias()
        return np.asarray(coef).shape

    def get(self, row: int, col: int = None) -> float:
        """Get a single weight.

        - `get(i)` on 1-D weights → returns w[i] as float
        - `get(i)` on 2-D weights → returns row i as a list of floats
        - `get(i, j)` on 2-D weights → returns w[i, j] as float
        """
        coef, _, _ = self._get_coef_and_bias()
        coef = np.asarray(coef)
        if col is None:
            if coef.ndim == 1:
                return float(coef[row])
            # 2-D: return whole row
            return coef[row].tolist()
        return float(coef[row, col])

    def get_bias(self, index: int = None) -> float:
        """Get a bias term.

        - `get_bias()` on scalar bias → returns the single bias
        - `get_bias()` on 1-D bias → returns the whole vector as a list
        - `get_bias(i)` on 1-D bias → returns b[i] as float
        """
        _, bias, _ = self._get_coef_and_bias()
        bias_arr = np.asarray(bias).flatten()
        if bias_arr.size == 1:
            return float(bias_arr[0])
        if index is None:
            return bias_arr.tolist()
        return float(bias_arr[index])

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def set(self, *args):
        """Set a single weight. Accepts two calling conventions:

        - `set(i, v)` on 1-D weights → w[i] = v
        - `set(i, v)` on 2-D weights → w[i, :] = v (whole row, v must be list/array)
        - `set(i, j, v)` on 2-D weights → w[i, j] = v  (numpy-natural order)

        Returns self so calls chain: `model.weights.set(0, 1.5).set(1, 2.0)`.
        """
        # Parse args: 2-positional = (row, value); 3-positional = (row, col, value)
        if len(args) == 2:
            row, value = args
            col = None
        elif len(args) == 3:
            row, col, value = args
        else:
            raise TypeError(
                f"[WHAT] WeightAccessor.set() takes 2 or 3 positional args, got {len(args)}\n"
                f"[WHY] Calling conventions are set(i, v) for 1-D or set(i, j, v) for 2-D\n"
                f"[FIX] Use model.weights.set(0, 1.5) for 1-D, or model.weights.set(0, 1, 1.5) for 2-D"
            )

        coef, bias, kind = self._get_coef_and_bias()
        coef = np.asarray(coef)
        if kind == "w":
            target = self.technique.w
        else:
            target = self.technique.model.coef_
            # Handle sklearn 2-D coef_ (n_classes, n_features)
            if target.ndim > 1 and coef.ndim == 1:
                # We squeezed it in _get_coef_and_bias; need to set on the original 2-D
                target = target[0]

        if col is None:
            if coef.ndim == 1:
                target[row] = value
            else:
                # 2-D: set whole row
                target[row, :] = value
        else:
            if coef.ndim == 2:
                target[row, col] = value
            else:
                raise IndexError(
                    f"[WHAT] Tried to set weight at (row={row}, col={col}) on 1-D weights\n"
                    f"[WHY] The technique's weight vector is 1-D (shape {coef.shape})\n"
                    f"[FIX] Use set(row={row}, value=...) instead"
                )
        return self

    def set_bias(self, value, index: int = None):
        """Set a bias term. Same shape rules as get_bias()."""
        _, bias, kind = self._get_coef_and_bias()
        if kind == "w":
            target = self.technique.b
        else:
            target = self.technique.model.intercept_

        bias_arr = np.asarray(bias).flatten()
        if bias_arr.size == 1:
            if index is not None and index != 0:
                raise IndexError(f"Bias is scalar, cannot set index {index}")
            # Write back as scalar
            if kind == "w":
                self.technique.b = float(value)
            else:
                # sklearn intercept_ for binary classifiers is a 1-D array of size 1
                self.technique.model.intercept_[0] = float(value)
        else:
            if index is None:
                # Replace entire bias vector
                if kind == "w":
                    self.technique.b = np.asarray(value, dtype=float)
                else:
                    self.technique.model.intercept_[:] = value
            else:
                target[index] = value
        return self

    # ------------------------------------------------------------------
    # Bulk access
    # ------------------------------------------------------------------
    def raw_flat(self):
        """Return all weights (not bias) as a flat 1-D numpy array."""
        coef, _, _ = self._get_coef_and_bias()
        return np.asarray(coef).flatten()

    def raw(self):
        """Return the raw weight array (any shape) — useful for custom math."""
        coef, _, _ = self._get_coef_and_bias()
        return np.asarray(coef)

    def bias_flat(self):
        """Return bias as a flat 1-D numpy array."""
        _, bias, _ = self._get_coef_and_bias()
        return np.asarray(bias).flatten()

    def flat_index(self, i: int) -> float:
        """Treat weights as a flat array and return the i-th element."""
        return float(self.raw_flat()[i])

    def set_flat_index(self, i: int, value: float):
        """Treat weights as a flat array and set the i-th element."""
        coef, _, kind = self._get_coef_and_bias()
        coef = np.asarray(coef)
        flat = coef.flatten()
        flat[i] = value
        new_shape = coef.reshape(flat.shape) if coef.ndim == 1 else flat.reshape(coef.shape)
        if kind == "w":
            self.technique.w = new_shape
        else:
            if self.technique.model.coef_.ndim > 1 and coef.ndim == 1:
                # Was squeezed; put back into 2-D
                self.technique.model.coef_[0] = new_shape
            else:
                self.technique.model.coef_ = new_shape
        return self

    def __repr__(self):
        try:
            coef, bias, _ = self._get_coef_and_bias()
            coef_shape = np.asarray(coef).shape
            bias_shape = np.asarray(bias).shape
            return f"<WeightAccessor coef_shape={coef_shape} bias_shape={bias_shape}>"
        except NotImplementedError:
            return f"<WeightAccessor technique={self.technique.name} (no inspectable weights)>"
