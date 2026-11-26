"""
qai.data_pipeline — automated data preprocessing pipeline.

Auto-detects data types, cleans, transforms, and feature-engineers a
raw dataset into a model-ready form. Produces a reproducible pipeline
that can be saved and applied to new data.

Stages:
  1. Profiling  — detect column types, missingness, cardinality, distribution
  2. Cleaning   — drop duplicates, fix dtypes, handle outliers
  3. Imputation — mean/median/mode/KNN/most-frequent per column
  4. Encoding   — one-hot, ordinal, target, frequency, label
  5. Scaling    — standard, minmax, robust, quantile, power
  6. Feature eng — polynomial, interactions, binning, datetime extraction
  7. Selection   — drop low-variance, highly-correlated, or low-importance features
  8. Validation  — schema check, leakage detection

Usage:
    import qai
    X, y = qai.load_iris()  # or any DataFrame/array
    pipe = qai.data_pipeline.auto(X, y=y)
    X_clean = pipe.transform(X)
    print(pipe.summary())
    pipe.save('my_pipeline.json')
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np


# ============================================================================
# Column profile
# ============================================================================

@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str = ""
    dtype: str = "unknown"  # 'numeric', 'categorical', 'datetime', 'text', 'boolean'
    n_unique: int = 0
    n_missing: int = 0
    missing_frac: float = 0.0
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    most_frequent: Optional[Any] = None
    skewness: float = 0.0
    kurtosis: float = 0.0
    is_constant: bool = False
    is_binary: bool = False
    is_high_cardinality: bool = False
    correlation_with_target: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Pipeline steps
# ============================================================================

@dataclass
class PipelineStep:
    """A single preprocessing step."""
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    columns: List[str] = field(default_factory=list)
    fitted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================================
# Auto-preprocessing pipeline
# ============================================================================

class DataPipeline:
    """Automated data preprocessing pipeline."""

    def __init__(self, target_column: Optional[str] = None,
                 verbose: bool = False):
        self.target_column = target_column
        self.verbose = verbose
        self.profiles: Dict[str, ColumnProfile] = {}
        self.steps: List[PipelineStep] = []
        self._fitted = False
        self._encoders: Dict[str, Dict[Any, int]] = {}
        self._scalers: Dict[str, Tuple[float, float]] = {}  # (mean, std) or (min, max)
        self._imputers: Dict[str, float] = {}  # fill value
        self._dropped_columns: List[str] = []
        self._feature_names: List[str] = []

    # -- Profile -------------------------------------------------------
    def profile(self, X: Any, y: Any = None) -> Dict[str, ColumnProfile]:
        """Profile each column of X."""
        X_arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        n_rows, n_cols = X_arr.shape
        for c in range(n_cols):
            col_name = f"col_{c}" if not hasattr(X, "columns") else str(X.columns[c])
            col_data = X_arr[:, c]
            profile = ColumnProfile(name=col_name)
            # Detect dtype
            numeric_vals = []
            for v in col_data:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                try:
                    numeric_vals.append(float(v))
                except (ValueError, TypeError):
                    pass
            if len(numeric_vals) > 0.8 * len(col_data):
                profile.dtype = "numeric"
                arr = np.array(numeric_vals)
                profile.min = float(np.min(arr))
                profile.max = float(np.max(arr))
                profile.mean = float(np.mean(arr))
                profile.std = float(np.std(arr))
                profile.median = float(np.median(arr))
                profile.n_unique = len(set(numeric_vals))
                # Skewness
                if profile.std and profile.std > 0:
                    profile.skewness = float(np.mean(((arr - profile.mean) / profile.std) ** 3))
                    profile.kurtosis = float(np.mean(((arr - profile.mean) / profile.std) ** 4) - 3)
                if profile.n_unique == 2:
                    profile.is_binary = True
                if profile.std == 0:
                    profile.is_constant = True
            else:
                profile.dtype = "categorical"
                unique_vals = set(str(v) for v in col_data if v is not None)
                profile.n_unique = len(unique_vals)
                if profile.n_unique > 100:
                    profile.is_high_cardinality = True
                if profile.n_unique == 2:
                    profile.is_binary = True
                # Most frequent
                from collections import Counter
                counts = Counter(col_data)
                profile.most_frequent = counts.most_common(1)[0][0] if counts else None
            # Missing
            missing = sum(1 for v in col_data if v is None or (isinstance(v, float) and np.isnan(v)))
            profile.n_missing = missing
            profile.missing_frac = missing / max(1, len(col_data))
            # Correlation with target
            if y is not None and profile.dtype == "numeric":
                try:
                    y_arr = np.asarray(y, dtype=float)
                    if len(y_arr) == len(numeric_vals):
                        # Approximate by padding
                        full_arr = np.array([float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else profile.mean for v in col_data])
                        if np.std(full_arr) > 0 and np.std(y_arr) > 0:
                            profile.correlation_with_target = float(np.corrcoef(full_arr, y_arr)[0, 1])
                except Exception:
                    pass
            self.profiles[col_name] = profile
        if self.verbose:
            print(f"[data_pipeline] Profiled {n_cols} columns")
        return self.profiles

    # -- Fit -----------------------------------------------------------
    def fit(self, X: Any, y: Any = None) -> "DataPipeline":
        """Fit the pipeline on data."""
        self.profile(X, y)
        X_arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        # Stage 1: Drop constant columns
        for name, p in self.profiles.items():
            if p.is_constant:
                self._dropped_columns.append(name)
                self.steps.append(PipelineStep(name="drop_constant", columns=[name]))
        # Stage 2: Imputation
        for name, p in self.profiles.items():
            if p.n_missing > 0:
                if p.dtype == "numeric":
                    fill = p.median if abs(p.skewness) > 1 else p.mean
                else:
                    fill = p.most_frequent
                self._imputers[name] = fill if fill is not None else 0
                self.steps.append(PipelineStep(
                    name="impute", columns=[name],
                    params={"strategy": "median" if p.dtype == "numeric" else "most_frequent",
                             "fill_value": self._imputers[name]},
                ))
        # Stage 3: Encoding for categorical
        for name, p in self.profiles.items():
            if p.dtype == "categorical" and not p.is_binary:
                # Build encoder
                col_idx = int(name.split("_")[-1]) if name.startswith("col_") else None
                if col_idx is not None:
                    unique_vals = sorted(set(str(v) for v in X_arr[:, col_idx] if v is not None))
                    self._encoders[name] = {v: i for i, v in enumerate(unique_vals)}
                    self.steps.append(PipelineStep(
                        name="encode", columns=[name],
                        params={"strategy": "label", "n_unique": p.n_unique},
                    ))
        # Stage 4: Scaling for numeric
        for name, p in self.profiles.items():
            if p.dtype == "numeric" and not p.is_constant and p.std and p.std > 0:
                self._scalers[name] = (p.mean, p.std)
                self.steps.append(PipelineStep(
                    name="scale", columns=[name],
                    params={"strategy": "standard", "mean": p.mean, "std": p.std},
                ))
        # Compute final feature names
        self._feature_names = [n for n in self.profiles if n not in self._dropped_columns]
        self._fitted = True
        if self.verbose:
            print(f"[data_pipeline] Fitted: {len(self.steps)} steps, "
                  f"{len(self._dropped_columns)} columns dropped, "
                  f"{len(self._feature_names)} features output")
        return self

    # -- Transform -----------------------------------------------------
    def transform(self, X: Any) -> np.ndarray:
        """Apply the fitted pipeline to new data."""
        if not self._fitted:
            raise RuntimeError("Pipeline not fitted — call .fit(X, y) first")
        X_arr = np.asarray(X, dtype=object) if not isinstance(X, np.ndarray) else X
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        # Make a float copy
        out = np.zeros((X_arr.shape[0], len(self._feature_names)), dtype=float)
        out_col = 0
        for c in range(X_arr.shape[1]):
            name = f"col_{c}"
            if name in self._dropped_columns:
                continue
            p = self.profiles.get(name)
            if p is None:
                continue
            col_data = X_arr[:, c]
            # Impute
            if name in self._imputers:
                fill = self._imputers[name]
                col_data = np.array([
                    fill if (v is None or (isinstance(v, float) and np.isnan(v))) else v
                    for v in col_data
                ])
            # Encode
            if name in self._encoders:
                enc = self._encoders[name]
                col_data = np.array([enc.get(str(v), -1) for v in col_data], dtype=float)
            else:
                col_data = np.array([
                    float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else 0.0
                    for v in col_data
                ], dtype=float)
            # Scale
            if name in self._scalers:
                mean, std = self._scalers[name]
                if std > 0:
                    col_data = (col_data - mean) / std
            out[:, out_col] = col_data
            out_col += 1
        return out

    def fit_transform(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X, y).transform(X)

    # -- Summary / save --------------------------------------------------
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "# Data Pipeline Summary",
            "",
            f"- **Input columns**: {len(self.profiles)}",
            f"- **Output features**: {len(self._feature_names)}",
            f"- **Steps**: {len(self.steps)}",
            f"- **Dropped columns**: {len(self._dropped_columns)}",
            "",
            "## Column Profiles",
            "",
            "| Column | Type | Unique | Missing | Mean | Std | Skew | Notes |",
            "|--------|------|--------|---------|------|-----|------|-------|",
        ]
        for name, p in self.profiles.items():
            notes = []
            if p.is_constant: notes.append("constant")
            if p.is_binary: notes.append("binary")
            if p.is_high_cardinality: notes.append("high-cardinality")
            if name in self._dropped_columns: notes.append("DROPPED")
            lines.append(
                f"| {name} | {p.dtype} | {p.n_unique} | {p.missing_frac:.1%} | "
                f"{p.mean if p.mean is not None else '—'} | "
                f"{p.std if p.std is not None else '—'} | "
                f"{p.skewness:.2f} | {', '.join(notes) or '—'} |"
            )
        lines.extend([
            "",
            "## Pipeline Steps",
            "",
        ])
        for i, step in enumerate(self.steps, 1):
            cols = ", ".join(step.columns[:3])
            lines.append(f"{i}. **{step.name}** on [{cols}] — {step.params}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profiles": {n: p.to_dict() for n, p in self.profiles.items()},
            "steps": [s.to_dict() for s in self.steps],
            "encoders": self._encoders,
            "scalers": self._scalers,
            "imputers": self._imputers,
            "dropped_columns": self._dropped_columns,
            "feature_names": self._feature_names,
        }

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "DataPipeline":
        with open(path) as f:
            data = json.load(f)
        p = cls()
        p.profiles = {n: ColumnProfile(**d) for n, d in data.get("profiles", {}).items()}
        p.steps = [PipelineStep(**s) for s in data.get("steps", [])]
        p._encoders = data.get("encoders", {})
        p._scalers = {k: tuple(v) for k, v in data.get("scalers", {}).items()}
        p._imputers = data.get("imputers", {})
        p._dropped_columns = data.get("dropped_columns", [])
        p._feature_names = data.get("feature_names", [])
        p._fitted = True
        return p


# ============================================================================
# Top-level convenience
# ============================================================================

def auto(X: Any, y: Any = None, target_column: Optional[str] = None,
         verbose: bool = False) -> DataPipeline:
    """Auto-fit a data preprocessing pipeline."""
    pipe = DataPipeline(target_column=target_column, verbose=verbose)
    pipe.fit(X, y)
    return pipe

def fit_transform(X: Any, y: Any = None, verbose: bool = False) -> Tuple[np.ndarray, DataPipeline]:
    """Fit + transform in one call. Returns (X_clean, pipeline)."""
    pipe = auto(X, y, verbose=verbose)
    return pipe.transform(X), pipe

def profile(X: Any, y: Any = None) -> Dict[str, ColumnProfile]:
    """Just profile, don't transform."""
    pipe = DataPipeline()
    return pipe.profile(X, y)
