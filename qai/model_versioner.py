"""
qai.model_versioner — track model versions, compare diffs, rollback.

Usage:
    import qai
    ver = qai.ModelVersioner("./model_versions")

    # Save a version
    ver.save(model, version="1.0.0", note="initial training")

    # Make changes and save again
    model.finetune(X_new, y_new)
    ver.save(model, version="1.1.0", note="fine-tuned on new data")

    # List versions
    for v in ver.list_versions():
        print(v["version"], v["note"], v["size_kb"])

    # Diff two versions
    diff = ver.diff("1.0.0", "1.1.0")
    print(diff)

    # Rollback
    model = ver.load("1.0.0")
"""
from __future__ import annotations
import os, json, time, hashlib, shutil, pickle
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class VersionInfo:
    """Metadata for a saved model version."""
    version: str
    note: str
    created_at: str
    file_path: str
    file_size: int
    file_hash: str
    technique: str = ""
    params_count: int = 0


class ModelVersioner:
    """Track model versions with diff and rollback support.

    Args:
        storage_path: directory to store versioned models.
    """
    def __init__(self, storage_path: str = "./model_versions"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        self._index: Dict[str, VersionInfo] = {}
        self._load_index()

    def save(self, model, version: str, note: str = "") -> str:
        """Save a model version.

        Args:
            model: a trained qai.Model.
            version: semantic version string (e.g. "1.0.0").
            note: human-readable note about this version.

        Returns:
            Path to the saved model.
        """
        filename = f"model_{version}.qmodel"
        filepath = os.path.join(self.storage_path, filename)
        import qai
        qai.save_bundle(model, filepath, description=f"v{version}: {note}")

        with open(filepath, "rb") as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        tech = getattr(model, "technique", None)
        tech_name = getattr(tech, "name", "unknown") if tech else "unknown"
        n_params = 0
        if tech is not None:
            if hasattr(tech, "net") and tech.net is not None:
                try:
                    n_params = sum(p.numel() for p in tech.net.parameters())
                except Exception:
                    pass
            elif hasattr(tech, "model") and hasattr(tech.model, "coef_"):
                try:
                    n_params = int(np.asarray(tech.model.coef_).size)
                except Exception:
                    pass

        info = VersionInfo(
            version=version, note=note,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            file_path=filepath, file_size=os.path.getsize(filepath),
            file_hash=file_hash, technique=tech_name, params_count=n_params,
        )
        self._index[version] = info
        self._save_index()
        return filepath

    def load(self, version: str, sign_key: Optional[str] = None):
        """Load a specific model version.

        Args:
            version: the version string to load.
            sign_key: optional signing key.

        Returns:
            The loaded qai.Model.
        """
        if version not in self._index:
            raise KeyError(f"[WHAT] Version '{version}' not found\n"
                           f"[WHY] Available: {list(self._index.keys())}\n"
                           f"[FIX] Use a valid version")
        import qai
        bundle = qai.load_bundle(self._index[version].file_path, sign_key=sign_key)
        return bundle.model

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all saved versions, sorted by creation time."""
        versions = sorted(self._index.values(), key=lambda v: v.created_at)
        return [
            {"version": v.version, "note": v.note, "created_at": v.created_at,
             "size_kb": round(v.file_size / 1024, 1), "technique": v.technique,
             "params": v.params_count}
            for v in versions
        ]

    def diff(self, version_a: str, version_b: str) -> Dict[str, Any]:
        """Compare two model versions.

        Returns:
            Dict with: version_changed, size_diff, hash_match, params_diff.
        """
        if version_a not in self._index or version_b not in self._index:
            raise KeyError("One or both versions not found")
        a, b = self._index[version_a], self._index[version_b]
        return {
            "version_a": version_a, "version_b": version_b,
            "technique_changed": a.technique != b.technique,
            "size_diff_bytes": b.file_size - a.file_size,
            "hash_match": a.file_hash == b.file_hash,
            "params_diff": b.params_count - a.params_count,
            "note_a": a.note, "note_b": b.note,
            "identical": a.file_hash == b.file_hash,
        }

    def rollback(self, version: str):
        """Alias for load()."""
        return self.load(version)

    def delete(self, version: str) -> bool:
        """Delete a version."""
        if version not in self._index:
            return False
        info = self._index[version]
        if os.path.exists(info.file_path):
            os.remove(info.file_path)
        del self._index[version]
        self._save_index()
        return True

    def latest(self) -> Optional[str]:
        """Return the latest version string."""
        if not self._index:
            return None
        return max(self._index.keys(), key=lambda v: self._index[v].created_at)

    def _save_index(self):
        path = os.path.join(self.storage_path, "version_index.json")
        data = {v: asdict(info) for v, info in self._index.items()}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load_index(self):
        path = os.path.join(self.storage_path, "version_index.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                data = json.load(f)
            self._index = {v: VersionInfo(**info) for v, info in data.items()}
        except Exception:
            self._index = {}

    def __repr__(self):
        return f"<ModelVersioner n_versions={len(self._index)}>"


__all__ = ["ModelVersioner", "VersionInfo"]
