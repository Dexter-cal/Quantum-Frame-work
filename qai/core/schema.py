"""
qai.core.schema — real input/output schema declaration and validation
(design doc Section 91). This is the honesty mechanism: a model's declared
shape is actually checked against real data, not just documented.
"""
from __future__ import annotations
import numpy as np


class SchemaError(ValueError):
    """Raised when data doesn't match a model's declared schema."""


class Schema:
    def __init__(self, fields: dict):
        """fields: {"name": type} where type is one of float, int, bool, str"""
        self.fields = fields

    def validate_row(self, row: dict, context: str = "input"):
        """Validate dict-based (named field) inputs or outputs."""
        if not isinstance(row, dict):
            raise SchemaError(
                f"[WHAT] Expected dictionary with named fields in {context}\n"
                f"[WHY] Schema declares named fields: {list(self.fields.keys())}, but received {type(row).__name__}\n"
                f"[FIX] Pass a dict with keys matching: {', '.join(self.fields.keys())}"
            )
        for name, expected_type in self.fields.items():
            if name not in row:
                raise SchemaError(
                    f"[WHAT] Missing field '{name}' in {context}\n"
                    f"[WHY] Schema declares {name}: {expected_type.__name__}, but it wasn't provided\n"
                    f"[FIX] Include '{name}' in every {context} row"
                )
            value = row[name]
            if not isinstance(value, expected_type) and not (
                expected_type is float and isinstance(value, (int, float, np.number))
            ) and not (
                expected_type is int and isinstance(value, (int, np.integer))
            ):
                raise SchemaError(
                    f"[WHAT] Field '{name}' has wrong type in {context}\n"
                    f"[WHY] Schema expects {expected_type.__name__}, got {type(value).__name__} ({value!r})\n"
                    f"[FIX] Convert '{name}' to {expected_type.__name__} before passing it in"
                )

    def validate_array(self, X: np.ndarray, context: str = "input"):
        """For plain numeric arrays (the common case), check column count."""
        expected_cols = len(self.fields)
        actual_cols = X.shape[1] if X.ndim == 2 else X.shape[0]
        if actual_cols != expected_cols:
            names = ", ".join(self.fields.keys())
            raise SchemaError(
                f"[WHAT] {context} has {actual_cols} columns, schema declares {expected_cols}\n"
                f"[WHY] Schema fields are: {names}\n"
                f"[FIX] Provide exactly {expected_cols} columns, in the order: {names}"
            )

    def validate(self, value, context: str = "data"):
        """Unified validation dispatcher supporting dicts, list of dicts, and arrays."""
        if isinstance(value, dict):
            self.validate_row(value, context=context)
        elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            for idx, item in enumerate(value):
                self.validate_row(item, context=f"{context}[{idx}]")
        elif isinstance(value, (np.ndarray, list)):
            arr = np.asarray(value)
            self.validate_array(arr, context=context)
        else:
            self.validate_row(value, context=context)

    def dict_to_array(self, row: dict) -> np.ndarray:
        """Convert a named-field dictionary row into an ordered array based on schema fields."""
        self.validate_row(row, context="dict input")
        return np.array([row[k] for k in self.fields.keys()], dtype=float)

    def __repr__(self):
        fields = ", ".join(f"{k}: {v.__name__}" for k, v in self.fields.items())
        return f"Schema({fields})"
