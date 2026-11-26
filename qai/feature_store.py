"""
qai.feature_store — feature engineering + storage + reuse across models.

A feature store lets you compute features once, store them, and reuse them
across multiple models — avoiding redundant computation and ensuring
consistency between training and serving.

Usage:
    import qai
    store = qai.FeatureStore("./features")

    # Register a feature group
    store.register("user_stats", compute_fn=lambda df: {
        "avg_purchase": df["amount"].mean(),
        "n_orders": len(df),
    }, dependencies=["raw_orders"])

    # Compute + materialize features
    features = store.compute("user_stats", input_data=my_dataframe)

    # Retrieve cached features
    cached = store.get("user_stats")

    # List all registered features
    for name in store.list_features():
        print(name)
"""
from __future__ import annotations
import os, json, time, hashlib, pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Callable, Set
import numpy as np


@dataclass
class FeatureGroup:
    """A registered feature group."""
    name: str
    description: str = ""
    compute_fn: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    created_at: str = ""
    last_computed: str = ""
    is_materialized: bool = False
    n_features: int = 0
    materialized_path: str = ""


class FeatureStore:
    """Feature store with registration, computation, caching, and reuse.

    Args:
        storage_path: directory for materialized features.
    """
    def __init__(self, storage_path: str = "./feature_store"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._registry: Dict[str, FeatureGroup] = {}
        self._cache: Dict[str, Any] = {}
        self._load_registry()

    def register(self, name: str, compute_fn: Optional[Callable] = None,
                 description: str = "", dependencies: Optional[List[str]] = None,
                 version: str = "1.0.0") -> "FeatureStore":
        """Register a feature group.

        Args:
            name: unique feature group name.
            compute_fn: function that computes features from input data.
            description: human-readable description.
            dependencies: names of other feature groups this one depends on.
            version: semantic version.
        """
        if name in self._registry:
            raise ValueError(f"[WHAT] Feature group '{name}' already registered")
        self._registry[name] = FeatureGroup(
            name=name, description=description, compute_fn=compute_fn,
            dependencies=dependencies or [], version=version,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._save_registry()
        return self

    def compute(self, name: str, input_data: Any = None,
                use_cache: bool = True) -> Any:
        """Compute (or retrieve cached) features for a group.

        Args:
            name: feature group name.
            input_data: data to compute features from.
            use_cache: if True, return cached result if available.

        Returns:
            Computed features (dict or array).
        """
        if name not in self._registry:
            raise KeyError(f"[WHAT] Feature group '{name}' not registered")
        fg = self._registry[name]

        if use_cache and name in self._cache:
            return self._cache[name]

        # Compute dependencies first
        dep_results = {}
        for dep in fg.dependencies:
            if dep in self._registry:
                dep_results[dep] = self.compute(dep, input_data, use_cache)

        # Compute this group
        if fg.compute_fn is None:
            raise ValueError(f"Feature group '{name}' has no compute_fn")
        try:
            result = fg.compute_fn(input_data) if not dep_results else fg.compute_fn(input_data, **dep_results)
        except TypeError:
            result = fg.compute_fn(input_data)

        # Cache + update metadata
        self._cache[name] = result
        fg.last_computed = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if isinstance(result, dict):
            fg.n_features = len(result)
        elif isinstance(result, np.ndarray):
            fg.n_features = result.shape[-1] if result.ndim > 0 else 1
        self._save_registry()
        return result

    def materialize(self, name: str, input_data: Any = None) -> str:
        """Compute + persist features to disk.

        Args:
            name: feature group name.
            input_data: data to compute from (uses cache if None).

        Returns:
            Path to the materialized file.
        """
        if name not in self._registry:
            raise KeyError(f"Feature group '{name}' not registered")
        fg = self._registry[name]

        features = self.compute(name, input_data)
        path = os.path.join(self.storage_path, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(features, f)

        fg.is_materialized = True
        fg.materialized_path = path
        self._save_registry()
        return path

    def get(self, name: str) -> Any:
        """Retrieve cached or materialized features."""
        if name in self._cache:
            return self._cache[name]
        fg = self._registry.get(name)
        if fg and fg.is_materialized and os.path.exists(fg.materialized_path):
            with open(fg.materialized_path, "rb") as f:
                features = pickle.load(f)
            self._cache[name] = features
            return features
        raise KeyError(f"Feature group '{name}' not computed or materialized")

    def list_features(self) -> List[Dict[str, Any]]:
        """List all registered feature groups."""
        return [
            {"name": fg.name, "description": fg.description,
             "version": fg.version, "n_features": fg.n_features,
             "dependencies": fg.dependencies,
             "is_materialized": fg.is_materialized,
             "last_computed": fg.last_computed}
            for fg in self._registry.values()
        ]

    def delete(self, name: str) -> bool:
        """Delete a feature group + its materialized data."""
        if name not in self._registry:
            return False
        fg = self._registry[name]
        if fg.is_materialized and os.path.exists(fg.materialized_path):
            os.remove(fg.materialized_path)
        if name in self._cache:
            del self._cache[name]
        del self._registry[name]
        self._save_registry()
        return True

    def _save_registry(self):
        path = os.path.join(self.storage_path, "registry.json")
        data = {n: {k: v for k, v in asdict(fg).items() if k != "compute_fn"}
                for n, fg in self._registry.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_registry(self):
        path = os.path.join(self.storage_path, "registry.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            for name, info in data.items():
                fg = FeatureGroup(**info)
                self._registry[name] = fg
        except Exception:
            pass

    def __repr__(self):
        return f"<FeatureStore n_groups={len(self._registry)}>"


__all__ = ["FeatureStore", "FeatureGroup"]
