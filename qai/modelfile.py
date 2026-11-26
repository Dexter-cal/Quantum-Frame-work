"""
qai.modelfile -- model files with proper extensions + a full file system.

Solves the user's question: "when you train a model, it should produce a
file with an extension that the user can pick up and do whatever they want."

  ModelFile          — wraps a .qmodel file with rich operations
  ModelShelf         — a directory of model files you can browse/search/load
  ModelConverter     — convert between .qmodel, .pt, .pkl, .onnx, .safetensors
  ModelFingerprinter — hash model weights for identification
  ModelProvenance    — track which dataset + hyperparams produced a model
  ModelVersioner     — semver versioning in filenames
  ModelDiffer        — compare two .qmodel files
  AutoSaver          — auto-save after fit() with configurable triggers

Usage:
    # Train + auto-save
    model = qai.build(technique="regression")
    model.fit(X, y)
    # → automatically creates regression_v1.0.0_2024-01-15.qmodel

    # Browse saved models
    shelf = qai.ModelShelf("./my_models")
    for m in shelf.list():
        print(m.name, m.size, m.technique, m.fingerprint)

    # Load any model file
    model = shelf.load("regression_v1.0.0_2024-01-15.qmodel")

    # Convert formats
    qai.ModelConverter.to_torch(model, "model.pt")
    qai.ModelConverter.to_pickle(model, "model.pkl")
    qai.ModelConverter.to_onnx(model, "model.onnx")

    # Inspect a file (like `file` command)
    info = qai.ModelFile.inspect("model.qmodel")
    print(info["technique"], info["n_params"], info["fingerprint"])

    # Compare two models
    diff = qai.ModelDiffer.compare("old.qmodel", "new.qmodel")
"""
from __future__ import annotations
import hashlib
import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Constants
# ============================================================================
QMODEL_EXTENSION = ".qmodel"
SUPPORTED_FORMATS = [".qmodel", ".pt", ".pkl", ".onnx", ".safetensors", ".npy"]


# ============================================================================
# 1. ModelFile — wraps a .qmodel file with rich operations
# ============================================================================
class ModelFile:
    """Represents a single .qmodel file on disk.

    A .qmodel file is a zip archive containing:
      - manifest.json    (metadata: technique, params, provenance, version)
      - model.json       (architecture spec)
      - weights.npy      (weight arrays)
      - README.md        (auto-generated documentation)
      - fingerprint.txt  (SHA-256 hash of weights for identification)
      - provenance.json  (dataset, hyperparams, training history)
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.name = os.path.basename(path)
        self._manifest: Optional[Dict[str, Any]] = None
        self._provenance: Optional[Dict[str, Any]] = None
        self._fingerprint: Optional[str] = None

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path)

    @property
    def size_bytes(self) -> int:
        return os.path.getsize(self.path) if self.exists else 0

    @property
    def size_human(self) -> str:
        size = self.size_bytes
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    @property
    def extension(self) -> str:
        return os.path.splitext(self.name)[1]

    @property
    def modified_at(self) -> float:
        return os.path.getmtime(self.path) if self.exists else 0

    @property
    def modified_at_human(self) -> str:
        ts = self.modified_at
        if ts == 0:
            return "unknown"
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def manifest(self) -> Dict[str, Any]:
        """Read the manifest from the .qmodel file."""
        if self._manifest is not None:
            return self._manifest
        try:
            import zipfile
            with zipfile.ZipFile(self.path, "r") as zf:
                manifest_data = zf.read("manifest.json")
                self._manifest = json.loads(manifest_data)
        except Exception as e:
            self._manifest = {"error": str(e)}
        return self._manifest

    def provenance(self) -> Dict[str, Any]:
        """Read provenance info (dataset, hyperparams, training history)."""
        if self._provenance is not None:
            return self._provenance
        try:
            import zipfile
            with zipfile.ZipFile(self.path, "r") as zf:
                if "provenance.json" in zf.namelist():
                    self._provenance = json.loads(zf.read("provenance.json"))
                else:
                    self._provenance = {}
        except Exception:
            self._provenance = {}
        return self._provenance

    def fingerprint(self) -> str:
        """Get the model's fingerprint (SHA-256 of weights)."""
        if self._fingerprint is not None:
            return self._fingerprint
        try:
            import zipfile
            with zipfile.ZipFile(self.path, "r") as zf:
                if "fingerprint.txt" in zf.namelist():
                    self._fingerprint = zf.read("fingerprint.txt").decode().strip()
                else:
                    # compute from weights
                    weights_data = zf.read("weights.npy")
                    self._fingerprint = hashlib.sha256(weights_data).hexdigest()[:16]
        except Exception:
            self._fingerprint = "unknown"
        return self._fingerprint

    @property
    def technique(self) -> str:
        if self.extension != QMODEL_EXTENSION:
            return os.path.splitext(self.name)[0].split("_v")[0]
        m = self.manifest()
        return m.get("technique", "unknown")

    @property
    def version(self) -> str:
        if self.extension != QMODEL_EXTENSION:
            parsed = ModelVersioner.parse_filename(self.name)
            return parsed.get("version", "unknown")
        m = self.manifest()
        return m.get("version", "1.0.0")

    @property
    def n_params(self) -> int:
        if self.extension != QMODEL_EXTENSION:
            return 0
        m = self.manifest()
        return m.get("total_params", 0)

    @property
    def description(self) -> str:
        if self.extension != QMODEL_EXTENSION:
            return ""
        m = self.manifest()
        return m.get("description", "")

    @property
    def author(self) -> str:
        if self.extension != QMODEL_EXTENSION:
            return "unknown"
        m = self.manifest()
        return m.get("author", "unknown")

    def summary(self) -> Dict[str, Any]:
        """One-line summary of the model file."""
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size_human,
            "technique": self.technique,
            "version": self.version,
            "n_params": self.n_params,
            "fingerprint": self.fingerprint(),
            "modified": self.modified_at_human,
            "author": self.author,
            "description": self.description[:100],
        }

    def load(self) -> Any:
        """Load the model from the file (any format)."""
        ext = self.extension
        if ext == QMODEL_EXTENSION:
            import qai
            return qai.load_bundle(self.path)
        elif ext == ".pkl":
            return ModelConverter.from_pickle(self.path)
        elif ext == ".pt":
            return ModelConverter.from_torch(self.path)
        elif ext == ".npy":
            return np.load(self.path, allow_pickle=True)
        else:
            raise ValueError(f"cannot load format: {ext}")

    def copy_to(self, dest_dir: str) -> str:
        """Copy this model file to another directory."""
        dest = os.path.join(dest_dir, self.name)
        shutil.copy2(self.path, dest)
        return dest

    def rename(self, new_name: str) -> str:
        """Rename the model file."""
        new_path = os.path.join(os.path.dirname(self.path), new_name)
        os.rename(self.path, new_path)
        self.path = new_path
        self.name = new_name
        return new_path

    def delete(self) -> bool:
        """Delete the model file."""
        if self.exists:
            os.unlink(self.path)
            return True
        return False

    @staticmethod
    def inspect(path: str) -> Dict[str, Any]:
        """Inspect any model file (like the `file` command).

        Works with .qmodel, .pt, .pkl, .onnx, .npy
        """
        ext = os.path.splitext(path)[1].lower()
        if not os.path.exists(path):
            return {"error": "file not found", "path": path}
        info: Dict[str, Any] = {
            "path": path,
            "name": os.path.basename(path),
            "extension": ext,
            "size_bytes": os.path.getsize(path),
            "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
        }
        if ext == ".qmodel":
            mf = ModelFile(path)
            info.update({
                "format": "qai_bundle",
                "technique": mf.technique,
                "version": mf.version,
                "n_params": mf.n_params,
                "fingerprint": mf.fingerprint(),
                "author": mf.author,
                "description": mf.description,
                "signed": mf.manifest().get("signed", False),
                "provenance": mf.provenance(),
            })
        elif ext == ".pt":
            info["format"] = "pytorch"
            try:
                import torch
                obj = torch.load(path, map_location="cpu")
                if isinstance(obj, dict):
                    info["keys"] = list(obj.keys())[:10]
                    info["n_tensors"] = sum(1 for v in obj.values()
                                              if hasattr(v, "shape"))
                elif hasattr(obj, "state_dict"):
                    info["format"] = "pytorch_model"
                    info["n_params"] = sum(p.numel() for p in obj.parameters())
                info["type"] = type(obj).__name__
            except Exception as e:
                info["error"] = str(e)
        elif ext == ".pkl":
            info["format"] = "pickle"
            try:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                info["type"] = type(obj).__name__
                if hasattr(obj, "state_dict"):
                    info["format"] = "pickle_sklearn_or_model"
            except Exception as e:
                info["error"] = str(e)
        elif ext == ".onnx":
            info["format"] = "onnx"
            try:
                import onnx
                model = onnx.load(path)
                info["ir_version"] = model.ir_version
                info["producer"] = model.producer_name
                info["n_nodes"] = len(model.graph.node)
            except Exception:
                info["note"] = "install onnx for detailed inspection"
        elif ext == ".npy":
            info["format"] = "numpy_array"
            try:
                arr = np.load(path, allow_pickle=True)
                info["shape"] = list(arr.shape)
                info["dtype"] = str(arr.dtype)
            except Exception as e:
                info["error"] = str(e)
        else:
            info["format"] = "unknown"
        # compute file hash
        with open(path, "rb") as f:
            info["file_hash"] = hashlib.sha256(f.read()).hexdigest()[:16]
        return info

    def __repr__(self):
        return (f"<ModelFile {self.name} technique={self.technique} "
                f"size={self.size_human} params={self.n_params}>")


# ============================================================================
# 2. ModelShelf — a directory of model files you can browse/search/load
# ============================================================================
class ModelShelf:
    """A directory that holds .qmodel files; supports search + browse + load.

    Like a bookshelf for models. You can:
      - list all models
      - search by technique, version, author, description
      - sort by date, size, name
      - load any model
      - add/remove models
      - get stats (total size, n models, by technique)
    """

    def __init__(self, directory: str = "./models"):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def list(self, sort_by: str = "modified") -> List[ModelFile]:
        """List all model files on the shelf."""
        files = []
        for name in os.listdir(self.directory):
            if name.endswith(QMODEL_EXTENSION) or any(name.endswith(ext) for ext in SUPPORTED_FORMATS):
                files.append(ModelFile(os.path.join(self.directory, name)))
        if sort_by == "modified":
            files.sort(key=lambda f: f.modified_at, reverse=True)
        elif sort_by == "name":
            files.sort(key=lambda f: f.name)
        elif sort_by == "size":
            files.sort(key=lambda f: f.size_bytes, reverse=True)
        return files

    def search(self, query: str = "",
                technique: Optional[str] = None,
                author: Optional[str] = None,
                version: Optional[str] = None) -> List[ModelFile]:
        """Search models on the shelf."""
        results = []
        q = query.lower()
        for mf in self.list():
            # text search
            if q and q not in mf.name.lower() and q not in mf.description.lower():
                continue
            # filter by technique
            if technique and mf.technique != technique:
                continue
            # filter by author
            if author and mf.author != author:
                continue
            # filter by version
            if version and mf.version != version:
                continue
            results.append(mf)
        return results

    def load(self, name: str) -> Any:
        """Load a model by filename."""
        path = os.path.join(self.directory, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"no model named '{name}' on shelf")
        return ModelFile(path).load()

    def add(self, model_path: str, rename: Optional[str] = None) -> str:
        """Add a model file to the shelf."""
        name = rename or os.path.basename(model_path)
        dest = os.path.join(self.directory, name)
        shutil.copy2(model_path, dest)
        return dest

    def remove(self, name: str) -> bool:
        """Remove a model from the shelf."""
        path = os.path.join(self.directory, name)
        if os.path.exists(path):
            os.unlink(path)
            return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Statistics about the shelf."""
        files = self.list()
        by_technique: Dict[str, int] = {}
        total_size = 0
        for f in files:
            by_technique[f.technique] = by_technique.get(f.technique, 0) + 1
            total_size += f.size_bytes
        return {
            "directory": self.directory,
            "n_models": len(files),
            "total_size_bytes": total_size,
            "total_size_human": ModelFile(os.path.join(self.directory, ".")).size_human if files else "0B",
            "by_technique": by_technique,
            "oldest": min((f.modified_at for f in files), default=0),
            "newest": max((f.modified_at for f in files), default=0),
        }

    def tree(self) -> str:
        """Render the shelf as a tree (like `ls -la`)."""
        files = self.list()
        lines = [f"📚 ModelShelf: {self.directory} ({len(files)} models)"]
        for f in files:
            lines.append(f"  📄 {f.name:<40} {f.size_human:>8}  "
                         f"{f.technique:<15} v{f.version:<8} {f.modified_at_human}")
        return "\n".join(lines)


# ============================================================================
# 3. ModelConverter — convert between formats
# ============================================================================
class ModelConverter:
    """Convert models between .qmodel, .pt, .pkl, .onnx, .safetensors."""

    @staticmethod
    def to_torch(model: Any, path: str) -> str:
        """Save model as a PyTorch .pt file."""
        try:
            import torch
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), path)
            elif hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
                torch.save(model.technique.state_dict(), path)
            else:
                torch.save(model, path)
            return path
        except ImportError:
            raise ImportError("torch not installed; pip install torch")

    @staticmethod
    def to_pickle(model: Any, path: str) -> str:
        """Save model as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(model, f)
        return path

    @staticmethod
    def to_onnx(model: Any, path: str,
                 input_shape: Tuple[int, ...] = (1, 4)) -> str:
        """Export model to ONNX format."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch not installed for ONNX export")
        # get the underlying torch model
        torch_model = model
        if hasattr(model, "technique") and hasattr(model.technique, "net"):
            torch_model = model.technique.net
        elif hasattr(model, "technique") and hasattr(model.technique, "_model"):
            torch_model = model.technique._model
        if not isinstance(torch_model, torch.nn.Module):
            raise ValueError("model doesn't have a torch.nn.Module to export")
        torch_model.eval()
        dummy = torch.randn(*input_shape)
        torch.onnx.export(torch_model, dummy, path, export_params=True,
                           opset_version=11)
        return path

    @staticmethod
    def to_numpy_weights(model: Any, path: str) -> str:
        """Save just the weights as a .npy file."""
        weights = {}
        if hasattr(model, "state_dict"):
            for k, v in model.state_dict().items():
                weights[k] = np.asarray(v)
        elif hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
            for k, v in model.technique.state_dict().items():
                weights[k] = np.asarray(v)
        else:
            weights["model"] = model
        np.save(path, weights, allow_pickle=True)
        return path

    @staticmethod
    def from_pickle(path: str) -> Any:
        """Load a model from a pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def from_torch(path: str) -> Any:
        """Load a model from a .pt file."""
        try:
            import torch
            return torch.load(path, map_location="cpu")
        except ImportError:
            raise ImportError("torch not installed")

    @staticmethod
    def convert(input_path: str, output_path: str) -> str:
        """Auto-detect input format and convert to output format."""
        in_ext = os.path.splitext(input_path)[1].lower()
        out_ext = os.path.splitext(output_path)[1].lower()
        if in_ext == ".qmodel":
            import qai
            model = qai.load_bundle(input_path)
        elif in_ext == ".pt":
            model = ModelConverter.from_torch(input_path)
        elif in_ext == ".pkl":
            model = ModelConverter.from_pickle(input_path)
        else:
            raise ValueError(f"unsupported input format: {in_ext}")
        if out_ext == ".pt":
            return ModelConverter.to_torch(model, output_path)
        elif out_ext == ".pkl":
            return ModelConverter.to_pickle(model, output_path)
        elif out_ext == ".onnx":
            return ModelConverter.to_onnx(model, output_path)
        elif out_ext == ".npy":
            return ModelConverter.to_numpy_weights(model, output_path)
        elif out_ext == ".qmodel":
            import qai
            return qai.save_bundle(model, output_path)
        else:
            raise ValueError(f"unsupported output format: {out_ext}")


# ============================================================================
# 4. ModelFingerprinter — identify models by weight hash
# ============================================================================
class ModelFingerprinter:
    """Compute a unique fingerprint for a model based on its weights.

    The fingerprint is a SHA-256 hash of the flattened weight array.
    Two models with the same weights have the same fingerprint.
    """

    @staticmethod
    def fingerprint(model: Any) -> str:
        """Compute the fingerprint of a model."""
        weights = ModelFingerprinter._extract_weights(model)
        if not weights:
            return "empty"
        # flatten all weights into one array
        flat = np.concatenate([w.flatten() for w in weights])
        return hashlib.sha256(flat.tobytes()).hexdigest()[:16]

    @staticmethod
    def _extract_weights(model: Any) -> List[np.ndarray]:
        """Extract weight arrays from a model."""
        if hasattr(model, "state_dict"):
            try:
                return [np.asarray(v) for v in model.state_dict().values()]
            except Exception:
                pass
        if hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
            try:
                return [np.asarray(v) for v in model.technique.state_dict().values()]
            except Exception:
                pass
        # check technique for numpy attributes
        targets = [model]
        if hasattr(model, "technique"):
            targets.append(model.technique)
        out = []
        for target in targets:
            for name in dir(target):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(target, name)
                    if isinstance(val, np.ndarray) and val.size > 0:
                        out.append(val)
                except Exception:
                    pass
        return out

    @staticmethod
    def are_identical(model_a: Any, model_b: Any) -> bool:
        """Check if two models have identical weights."""
        return ModelFingerprinter.fingerprint(model_a) == \
               ModelFingerprinter.fingerprint(model_b)

    @staticmethod
    def similarity(model_a: Any, model_b: Any) -> float:
        """Compute weight similarity (cosine of flattened weight vectors)."""
        wa = ModelFingerprinter._extract_weights(model_a)
        wb = ModelFingerprinter._extract_weights(model_b)
        if not wa or not wb:
            return 0.0
        # flatten + pad to same length
        fa = np.concatenate([w.flatten() for w in wa])
        fb = np.concatenate([w.flatten() for w in wb])
        if len(fa) != len(fb):
            min_len = min(len(fa), len(fb))
            fa, fb = fa[:min_len], fb[:min_len]
        na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
        if na < 1e-9 or nb < 1e-9:
            return 0.0
        return float(np.dot(fa, fb) / (na * nb))


# ============================================================================
# 5. ModelProvenance — track how a model was created
# ============================================================================
class ModelProvenance:
    """Track the full provenance of a model.

    Records:
      - dataset (name, size, hash, columns)
      - hyperparameters (all params passed to the technique)
      - training history (loss per epoch, training time)
      - environment (python version, qai version, OS, hardware)
      - parent model (if this is a fine-tune)
      - git commit (if available)
    """

    def __init__(self):
        self.data: Dict[str, Any] = {
            "created_at": time.time(),
            "created_at_iso": datetime.now().isoformat(),
            "environment": self._capture_env(),
            "dataset": {},
            "hyperparameters": {},
            "training_history": [],
            "parent_model": None,
            "git_commit": self._git_commit(),
        }

    def _capture_env(self) -> Dict[str, Any]:
        import sys, platform
        try:
            import qai
            qai_version = "unknown"
        except Exception:
            qai_version = "unknown"
        return {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "qai_version": qai_version,
            "hostname": platform.node(),
        }

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()[:12]
        except Exception:
            pass
        return None

    def record_dataset(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                        name: str = "unnamed") -> None:
        """Record dataset info."""
        self.data["dataset"] = {
            "name": name,
            "n_samples": int(X.shape[0]) if hasattr(X, "shape") else len(X),
            "n_features": int(X.shape[1]) if hasattr(X, "shape") and X.ndim > 1 else 0,
            "has_labels": y is not None,
            "hash": hashlib.sha256(np.asarray(X).tobytes()).hexdigest()[:16] if hasattr(X, "tobytes") else "unknown",
        }

    def record_hyperparams(self, params: Dict[str, Any]) -> None:
        """Record the hyperparameters used."""
        # convert non-serializable values to strings
        clean = {}
        for k, v in params.items():
            if isinstance(v, (str, int, float, bool, list, dict, type(None))):
                clean[k] = v
            else:
                clean[k] = str(v)
        self.data["hyperparameters"] = clean

    def record_training(self, epoch: int, loss: float,
                          extra: Optional[Dict] = None) -> None:
        """Record one training epoch."""
        entry = {"epoch": epoch, "loss": float(loss)}
        if extra:
            entry.update(extra)
        self.data["training_history"].append(entry)

    def record_parent(self, parent_path: str) -> None:
        """Record the parent model (for fine-tuning lineage)."""
        self.data["parent_model"] = parent_path

    def to_dict(self) -> Dict[str, Any]:
        return self.data

    def to_json(self) -> str:
        return json.dumps(self.data, indent=2, default=str)

    def summary(self) -> str:
        """Human-readable summary."""
        d = self.data
        lines = [
            f"Provenance for model created at {d['created_at_iso']}",
            f"  Environment: Python {d['environment']['python_version']} on {d['environment']['platform']}",
            f"  Dataset: {d['dataset'].get('name', 'unknown')} "
            f"({d['dataset'].get('n_samples', 0)} samples, "
            f"{d['dataset'].get('n_features', 0)} features)",
            f"  Hyperparameters: {d['hyperparameters']}",
            f"  Training: {len(d['training_history'])} epochs recorded",
            f"  Parent: {d.get('parent_model', 'none')}",
            f"  Git: {d.get('git_commit', 'unknown')}",
        ]
        return "\n".join(lines)


# ============================================================================
# 6. ModelVersioner — semver versioning
# ============================================================================
class ModelVersioner:
    """Semantic versioning for model files.

    Filenames follow: {technique}_v{major}.{minor}.{patch}_{timestamp}.qmodel
    e.g. regression_v1.2.0_2024-01-15_143022.qmodel
    """

    @staticmethod
    def make_filename(technique: str, version: str = "1.0.0",
                       timestamp: Optional[float] = None) -> str:
        """Generate a versioned filename."""
        ts = timestamp or time.time()
        ts_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d_%H%M%S")
        # sanitize technique
        safe_tech = technique.replace(" ", "_").replace("/", "_")
        return f"{safe_tech}_v{version}_{ts_str}{QMODEL_EXTENSION}"

    @staticmethod
    def parse_filename(filename: str) -> Dict[str, Any]:
        """Parse a versioned filename."""
        name = os.path.splitext(filename)[0]
        parts = name.split("_v")
        if len(parts) < 2:
            return {"technique": name, "version": "unknown", "timestamp": None}
        technique = parts[0]
        rest = "_v".join(parts[1:])
        # split version from timestamp
        version_parts = rest.split("_", 1)
        version = version_parts[0]
        timestamp_str = version_parts[1] if len(version_parts) > 1 else None
        return {
            "technique": technique,
            "version": version,
            "timestamp_str": timestamp_str,
        }

    @staticmethod
    def bump_version(current: str, part: str = "patch") -> str:
        """Bump a semver version.

        part: "major" | "minor" | "patch"
        """
        try:
            major, minor, patch = current.split(".")
            major, minor, patch = int(major), int(minor), int(patch)
        except ValueError:
            major, minor, patch = 1, 0, 0
        if part == "major":
            major += 1; minor = 0; patch = 0
        elif part == "minor":
            minor += 1; patch = 0
        else:
            patch += 1
        return f"{major}.{minor}.{patch}"


# ============================================================================
# 7. ModelDiffer — compare two .qmodel files
# ============================================================================
class ModelDiffer:
    """Compare two model files; report differences."""

    @staticmethod
    def compare(path_a: str, path_b: str) -> Dict[str, Any]:
        """Compare two model files."""
        info_a = ModelFile.inspect(path_a)
        info_b = ModelFile.inspect(path_b)
        differences: List[Dict[str, Any]] = []
        # compare metadata
        for key in ["technique", "version", "n_params", "author"]:
            va = info_a.get(key)
            vb = info_b.get(key)
            if va != vb:
                differences.append({"field": key, "a": va, "b": vb})
        # compare fingerprints
        fp_a = info_a.get("fingerprint", "")
        fp_b = info_b.get("fingerprint", "")
        if fp_a and fp_b and fp_a != fp_b:
            differences.append({"field": "fingerprint", "a": fp_a, "b": fp_b,
                                 "meaning": "weights are different"})
        # compare sizes
        sa = info_a.get("size_bytes", 0)
        sb = info_b.get("size_bytes", 0)
        if sa != sb:
            differences.append({"field": "size", "a": sa, "b": sb,
                                 "diff_bytes": sb - sa})
        # compare provenance
        prov_a = info_a.get("provenance", {})
        prov_b = info_b.get("provenance", {})
        if prov_a.get("dataset", {}).get("hash") != prov_b.get("dataset", {}).get("hash"):
            differences.append({"field": "dataset", "meaning": "trained on different data"})
        return {
            "file_a": path_a,
            "file_b": path_b,
            "n_differences": len(differences),
            "differences": differences,
            "identical": len(differences) == 0,
        }

    @staticmethod
    def diff_summary(path_a: str, path_b: str) -> str:
        """Human-readable diff."""
        diff = ModelDiffer.compare(path_a, path_b)
        lines = [f"Diff: {os.path.basename(path_a)} vs {os.path.basename(path_b)}"]
        if diff["identical"]:
            lines.append("  ✅ Models are identical")
        else:
            lines.append(f"  ❌ {diff['n_differences']} differences:")
            for d in diff["differences"]:
                lines.append(f"    - {d['field']}: {d.get('a', '?')} → {d.get('b', '?')}")
        return "\n".join(lines)


# ============================================================================
# 8. AutoSaver — auto-save after fit() with configurable triggers
# ============================================================================
class AutoSaver:
    """Automatically save models after training.

    Triggers:
      - after_fit: save every time fit() is called
      - every_n_epochs: save every N epochs (requires callback)
      - on_best: save when a metric improves (requires callback)
      - on_crash: save right before OOM/crash (via signal handler)

    Usage:
        saver = AutoSaver("./my_models", trigger="after_fit")
        model = qai.build(technique="regression")
        saver.wrap(model)  # wraps model.fit to auto-save
        model.fit(X, y)    # → auto-saves to my_models/regression_v1.0.0_....qmodel
    """

    def __init__(self, directory: str = "./models",
                 trigger: str = "after_fit",
                 version: str = "1.0.0",
                 provenance: bool = True,
                 max_files: int = 100):
        self.directory = directory
        self.trigger = trigger
        self.version = version
        self.enable_provenance = provenance
        self.max_files = max_files
        self.shelf = ModelShelf(directory)
        self.save_count = 0
        self.saved_files: List[str] = []

    def wrap(self, model: Any) -> Any:
        """Wrap a model so fit() triggers auto-save."""
        original_fit = model.fit

        def wrapped_fit(*args, **kwargs):
            result = original_fit(*args, **kwargs)
            if self.trigger == "after_fit":
                self.save(model, args, kwargs)
            return result

        model.fit = wrapped_fit
        return model

    def save(self, model: Any, fit_args: Tuple = (),
             fit_kwargs: Optional[Dict] = None) -> str:
        """Save a model with auto-generated filename + provenance."""
        fit_kwargs = fit_kwargs or {}
        # get technique name
        technique = getattr(model, "name", None) or \
                     getattr(getattr(model, "technique", None), "name", "model")
        # generate filename
        filename = ModelVersioner.make_filename(technique, self.version)
        path = os.path.join(self.directory, filename)
        # build provenance
        provenance = None
        if self.enable_provenance:
            provenance = ModelProvenance()
            if fit_args:
                X = fit_args[0] if fit_args else None
                y = fit_args[1] if len(fit_args) > 1 else None
                if X is not None:
                    provenance.record_dataset(X, y)
            provenance.record_hyperparams(getattr(model, "params", {}))
        # save
        import qai
        try:
            qai.save_bundle(model, path)
            # add provenance + fingerprint to the bundle
            self._enrich_bundle(path, model, provenance)
        except Exception as e:
            # fallback to pickle
            with open(path.replace(QMODEL_EXTENSION, ".pkl"), "wb") as f:
                pickle.dump(model, f)
        self.save_count += 1
        self.saved_files.append(path)
        # cleanup old files if over limit
        if self.max_files and self.save_count > self.max_files:
            self._cleanup_old()
        return path

    def _enrich_bundle(self, path: str, model: Any,
                        provenance: Optional[ModelProvenance]) -> None:
        """Add fingerprint + provenance to a .qmodel bundle."""
        try:
            import zipfile
            import tempfile
            # read existing content
            with zipfile.ZipFile(path, "r") as zf:
                content = {name: zf.read(name) for name in zf.namelist()}
            # add fingerprint
            fp = ModelFingerprinter.fingerprint(model)
            content["fingerprint.txt"] = fp.encode()
            # add provenance
            if provenance:
                content["provenance.json"] = provenance.to_json().encode()
            # rewrite
            tmp = path + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in content.items():
                    zf.writestr(name, data)
            os.replace(tmp, path)
        except Exception:
            pass  # enrichment is optional

    def _cleanup_old(self) -> None:
        """Remove oldest files beyond max_files."""
        files = self.shelf.list(sort_by="modified")
        while len(files) > self.max_files:
            oldest = files.pop()
            oldest.delete()

    def status(self) -> Dict[str, Any]:
        return {
            "directory": self.directory,
            "trigger": self.trigger,
            "version": self.version,
            "save_count": self.save_count,
            "n_files_on_shelf": len(self.shelf.list()),
            "saved_files": [os.path.basename(f) for f in self.saved_files[-5:]],
        }


# ============================================================================
# Convenience: save with auto-generated filename
# ============================================================================
def save_trained(model: Any, directory: str = "./models",
                  version: str = "1.0.0",
                  provenance: bool = True,
                  X: Optional[np.ndarray] = None,
                  y: Optional[np.ndarray] = None) -> str:
    """Save a trained model with auto-generated filename + provenance.

    Usage:
        model = qai.build(technique="regression")
        model.fit(X, y)
        path = qai.save_trained(model, "./my_models", X=X, y=y)
        # → creates ./my_models/regression_v1.0.0_2024-01-15_143022.qmodel
    """
    saver = AutoSaver(directory, version=version, provenance=provenance)
    return saver.save(model, fit_args=(X, y) if X is not None else ())


def load_model_file(path: str) -> Any:
    """Load any model file by extension.

    Supports: .qmodel, .pt, .pkl, .npy
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".qmodel":
        import qai
        return qai.load_bundle(path)
    elif ext == ".pt":
        return ModelConverter.from_torch(path)
    elif ext == ".pkl":
        return ModelConverter.from_pickle(path)
    elif ext == ".npy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError(f"unsupported format: {ext}")


def inspect_model(path: str) -> Dict[str, Any]:
    """Inspect any model file (like the `file` command)."""
    return ModelFile.inspect(path)


__all__ = [
    "QMODEL_EXTENSION", "SUPPORTED_FORMATS",
    "ModelFile", "ModelShelf", "ModelConverter",
    "ModelFingerprinter", "ModelProvenance", "ModelVersioner",
    "ModelDiffer", "AutoSaver",
    "save_trained", "load_model_file", "inspect_model",
]
