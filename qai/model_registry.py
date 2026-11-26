"""
qai.model_registry -- local model marketplace.

A local registry for publishing, discovering, versioning, and searching
trained qai models. Models are stored as .qmodel bundles in a local
directory with an index file for fast search.

Usage:
    import qai
    reg = qai.ModelRegistry("./my_models")

    # Publish a model
    reg.publish(model, name="iris_classifier", version="1.0.0",
                description="Iris species classifier", tags=["classification", "iris"])

    # Search
    results = reg.search("iris")
    for r in results:
        print(r["name"], r["version"], r["description"])

    # Load
    model = reg.load("iris_classifier", version="1.0.0")

    # List all
    for entry in reg.list_all():
        print(entry["name"], entry["version"])

    # Delete
    reg.delete("iris_classifier", "1.0.0")
"""
from __future__ import annotations
import os, json, time, hashlib, shutil
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RegistryEntry:
    """A single model entry in the registry."""
    name: str
    version: str
    description: str
    tags: List[str] = field(default_factory=list)
    technique: str = ""
    created_at: str = ""
    file_path: str = ""
    file_size: int = 0
    file_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class ModelRegistry:
    """Local model marketplace with publish/discover/search/version/load.

    Args:
        registry_path: directory to store registered models.
    """
    def __init__(self, registry_path: str = "./model_registry"):
        self.registry_path = registry_path
        os.makedirs(registry_path, exist_ok=True)
        self.index_path = os.path.join(registry_path, "registry_index.json")
        self._index: Dict[str, Dict[str, RegistryEntry]] = {}  # {name: {version: entry}}
        self._load_index()

    def publish(self, model, name: str, version: str = "1.0.0",
                description: str = "", tags: Optional[List[str]] = None,
                sign_key: Optional[str] = None, **kwargs) -> str:
        """Publish a model to the registry.

        Args:
            model: a trained qai.Model.
            name: model name (unique within the registry).
            version: semantic version string.
            description: human-readable description.
            tags: list of search tags.
            sign_key: optional HMAC signing key for the bundle.
            **kwargs: passed to qai.save_bundle.

        Returns:
            The path to the saved bundle.
        """
        tags = tags or []
        # Save the model as a .qmodel bundle
        filename = f"{name}_{version}.qmodel"
        filepath = os.path.join(self.registry_path, filename)

        # Import qai lazily (avoids circular import)
        import qai
        qai.save_bundle(model, filepath, sign_key=sign_key,
                         description=description, **kwargs)

        # Compute file hash
        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        # Get technique name
        tech = getattr(model, "technique", None)
        technique_name = getattr(tech, "name", "unknown") if tech else "unknown"

        entry = RegistryEntry(
            name=name, version=version, description=description,
            tags=tags, technique=technique_name,
            file_path=filepath,
            file_size=os.path.getsize(filepath),
            file_hash=file_hash,
            metadata=kwargs,
        )

        # Add to index
        if name not in self._index:
            self._index[name] = {}
        self._index[name][version] = entry
        self._save_index()
        return filepath

    def load(self, name: str, version: Optional[str] = None,
             sign_key: Optional[str] = None):
        """Load a model from the registry.

        Args:
            name: model name.
            version: version string (default: latest).
            sign_key: HMAC signing key.

        Returns:
            The loaded qai.Model.
        """
        if name not in self._index:
            raise KeyError(f"[WHAT] Model '{name}' not found in registry\n"
                           f"[WHY] Registry has {len(self._index)} models\n"
                           f"[FIX] Use reg.list_all() to see available models")
        if version is None:
            # Get the latest version
            versions = sorted(self._index[name].keys())
            version = versions[-1]
        if version not in self._index[name]:
            raise KeyError(f"[WHAT] Version '{version}' not found for model '{name}'\n"
                           f"[WHY] Available versions: {list(self._index[name].keys())}\n"
                           f"[FIX] Use a valid version or omit version= for latest")

        entry = self._index[name][version]
        import qai
        bundle = qai.load_bundle(entry.file_path, sign_key=sign_key)
        return bundle.model

    def search(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """Search the registry by name, description, or tags.

        Args:
            query: search string.
            max_results: maximum number of results.

        Returns:
            List of matching entries as dicts.
        """
        query_lower = query.lower()
        results = []
        for name, versions in self._index.items():
            for version, entry in versions.items():
                score = 0
                if query_lower in name.lower():
                    score += 3
                if query_lower in entry.description.lower():
                    score += 2
                if any(query_lower in tag.lower() for tag in entry.tags):
                    score += 2
                if query_lower in entry.technique.lower():
                    score += 1
                if score > 0:
                    results.append((score, asdict(entry)))
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results[:max_results]]

    def list_all(self) -> List[Dict[str, Any]]:
        """List all models in the registry."""
        results = []
        for name, versions in self._index.items():
            for version, entry in versions.items():
                results.append(asdict(entry))
        return results

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model."""
        if name not in self._index:
            return []
        return sorted(self._index[name].keys())

    def delete(self, name: str, version: str) -> bool:
        """Delete a model version from the registry."""
        if name not in self._index or version not in self._index[name]:
            return False
        entry = self._index[name][version]
        if os.path.exists(entry.file_path):
            os.remove(entry.file_path)
        del self._index[name][version]
        if not self._index[name]:
            del self._index[name]
        self._save_index()
        return True

    def stats(self) -> Dict[str, Any]:
        """Return registry statistics."""
        total_models = sum(len(versions) for versions in self._index.values())
        total_size = sum(e.file_size for versions in self._index.values() for e in versions.values())
        techniques = set(e.technique for versions in self._index.values() for e in versions.values())
        return {
            "n_model_names": len(self._index),
            "n_total_models": total_models,
            "total_size_bytes": total_size,
            "n_techniques": len(techniques),
            "techniques": list(techniques),
        }

    def _save_index(self):
        """Save the index to disk."""
        index_data = {}
        for name, versions in self._index.items():
            index_data[name] = {v: asdict(e) for v, e in versions.items()}
        with open(self.index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)

    def _load_index(self):
        """Load the index from disk."""
        if not os.path.exists(self.index_path):
            return
        try:
            with open(self.index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            self._index = {}
            for name, versions in index_data.items():
                self._index[name] = {v: RegistryEntry(**e) for v, e in versions.items()}
        except Exception:
            self._index = {}


__all__ = ["ModelRegistry", "RegistryEntry"]
