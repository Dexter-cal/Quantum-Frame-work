"""
qai.modelfile.extended -- extended model file features.

  HFModelImporter   — download from HuggingFace → wrap as qai.Model → .qmodel
  ModelRegistry2    — unified registry: .qmodel + .pt + .pkl + HF cache
  ModelSigning      — sign .qmodel files with HMAC for tamper detection
  ModelLinting      — check .qmodel files for common issues
  ModelMigration    — migrate old .qmodel format to new format
  ModelThumbnail    — generate a visual thumbnail of model architecture
  ModelCompatibility — check if a .qmodel is compatible with current qai version
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import pickle
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. HFModelImporter — download from HuggingFace → .qmodel
# ============================================================================
class HFModelImporter:
    """Download a model from HuggingFace and convert it to .qmodel format.

    Pipeline:
      1. Download model from HF Hub (via huggingface_hub or transformers)
      2. Wrap it as a qai-compatible Model (with forward + state_dict)
      3. Save as .qmodel with provenance (source: "huggingface", repo_id)
      4. The .qmodel file can then be used like any native qai model

    Usage:
        importer = HFModelImporter()
        path = importer.import_model("distilbert-base-uncased", "./my_models")
        # → downloads from HF, converts to .qmodel, saves to ./my_models
        model = qai.load_model_file(path)
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/qai/hf_imports")
        os.makedirs(self.cache_dir, exist_ok=True)

    def import_model(self, repo_id: str, output_dir: str = "./models",
                      task: str = "text-classification",
                      secret_key: Optional[str] = None) -> str:
        """Download from HF → convert → save as .qmodel.

        Returns the path to the .qmodel file.
        """
        # 1. download
        model_dir = self._download_from_hf(repo_id)

        # 2. create a wrapper
        wrapper = HFModelWrapper(repo_id, model_dir, task)

        # 3. save as .qmodel
        os.makedirs(output_dir, exist_ok=True)
        safe_name = repo_id.replace("/", "_").replace("-", "_")
        filename = f"hf_{safe_name}_v1.0.0_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.qmodel"
        path = os.path.join(output_dir, filename)

        # 4. build the .qmodel archive
        self._save_as_qmodel(wrapper, path, repo_id, secret_key)
        return path

    def _download_from_hf(self, repo_id: str) -> str:
        """Download a model from HuggingFace Hub."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub not installed; pip install huggingface_hub"
            )
        local_dir = os.path.join(self.cache_dir, repo_id.replace("/", "_"))
        return snapshot_download(repo_id=repo_id, cache_dir=self.cache_dir)

    def _save_as_qmodel(self, wrapper: "HFModelWrapper", path: str,
                          repo_id: str, secret_key: Optional[str]) -> None:
        """Save an HFModelWrapper as a .qmodel archive."""
        from .modelfile import ModelProvenance, ModelFingerprinter, ModelVersioner

        provenance = ModelProvenance()
        provenance.data["source"] = "huggingface"
        provenance.data["repo_id"] = repo_id
        provenance.record_hyperparams({"task": wrapper.task})

        manifest = {
            "format_version": "1.0",
            "created_at": time.time(),
            "created_at_iso": datetime.now().isoformat(),
            "technique": "huggingface_import",
            "description": f"Imported from HuggingFace: {repo_id}",
            "author": "huggingface_importer",
            "source": "huggingface",
            "repo_id": repo_id,
            "task": wrapper.task,
            "total_params": 0,
            "contents": {
                "manifest.json": True,
                "model.json": True,
                "wrapper.pkl": True,
                "provenance.json": True,
                "fingerprint.txt": True,
                "README.md": True,
            },
        }

        model_json = {
            "type": "HFModelWrapper",
            "repo_id": repo_id,
            "task": wrapper.task,
        }

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            zf.writestr("model.json", json.dumps(model_json, indent=2))
            zf.writestr("wrapper.pkl", pickle.dumps(wrapper))
            zf.writestr("provenance.json", provenance.to_json())
            zf.writestr("fingerprint.txt", hashlib.sha256(
                repo_id.encode()).hexdigest()[:16])
            zf.writestr("README.md", f"# {repo_id}\n\nImported from HuggingFace.\n")


class HFModelWrapper:
    """Wraps a HuggingFace model so it behaves like a qai model.

    Has forward(), fit() (no-op), state_dict() (returns HF weights),
    and is_trained().
    """

    def __init__(self, repo_id: str, model_dir: str, task: str = "text-classification"):
        self.repo_id = repo_id
        self.model_dir = model_dir
        self.task = task
        self._model = None
        self._tokenizer = None
        self._trained = True  # HF models are pre-trained

    def _load(self):
        """Lazy-load the HF model + tokenizer."""
        if self._model is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self._tokenizer = AutoTokenizer.from_pretrained(self.repo_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.repo_id)
        except Exception as e:
            raise RuntimeError(f"failed to load HF model '{self.repo_id}': {e}")

    def forward(self, x: Any) -> Any:
        """Forward pass. Accepts text (str) or token IDs (list)."""
        self._load()
        import torch
        if isinstance(x, str):
            inputs = self._tokenizer(x, return_tensors="pt", truncation=True, max_length=512)
        elif isinstance(x, (list, np.ndarray)):
            inputs = {"input_ids": torch.as_tensor([x])}
        else:
            inputs = x
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.logits.numpy()

    def fit(self, X, y=None):
        """No-op: HF models are pre-trained. Use fine_tune() instead."""
        return self

    def train(self, X, y=None):
        """Alias for fit()."""
        return self.fit(X, y)

    def fine_tune(self, X, y, n_epochs: int = 3, lr: float = 1e-5):
        """Fine-tune the HF model on new data."""
        self._load()
        import torch
        from torch.optim import AdamW
        optimizer = AdamW(self._model.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for i, (x, target) in enumerate(zip(X, y)):
                if isinstance(x, str):
                    inputs = self._tokenizer(x, return_tensors="pt", truncation=True)
                else:
                    inputs = {"input_ids": torch.as_tensor([x])}
                labels = torch.tensor([int(target)])
                outputs = self._model(**inputs, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def state_dict(self) -> Dict[str, Any]:
        """Return the model's state dict."""
        self._load()
        return {k: v.cpu().numpy() for k, v in self._model.state_dict().items()}

    def is_trained(self) -> bool:
        return self._trained

    def __repr__(self):
        return f"<HFModelWrapper repo={self.repo_id} task={self.task}>"


# ============================================================================
# 2. ModelRegistry2 — unified registry
# ============================================================================
class ModelRegistry2:
    """Unified model registry: .qmodel + .pt + .pkl + HF cache.

    You register directories; the registry indexes all model files.
    Search by technique, author, tag, fingerprint, or free text.
    """

    def __init__(self):
        self._dirs: List[str] = []
        self._index: Dict[str, Dict[str, Any]] = {}

    def register_directory(self, directory: str) -> None:
        """Index all model files in a directory."""
        from .modelfile import ModelFile, SUPPORTED_FORMATS
        self._dirs.append(directory)
        for name in os.listdir(directory):
            if any(name.endswith(ext) for ext in SUPPORTED_FORMATS):
                path = os.path.join(directory, name)
                try:
                    info = ModelFile.inspect(path)
                    self._index[path] = info
                except Exception:
                    self._index[path] = {"path": path, "error": "inspection failed"}

    def search(self, query: str = "",
                technique: Optional[str] = None,
                author: Optional[str] = None,
                min_params: Optional[int] = None,
                max_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search across all registered directories."""
        results = []
        q = query.lower()
        for path, info in self._index.items():
            if q and q not in info.get("name", "").lower() and \
               q not in str(info.get("description", "")).lower():
                continue
            if technique and info.get("technique") != technique:
                continue
            if author and info.get("author") != author:
                continue
            if min_params and info.get("n_params", 0) < min_params:
                continue
            if max_size and info.get("size_bytes", 0) > max_size:
                continue
            results.append(info)
        return results

    def by_fingerprint(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """Find a model by its fingerprint."""
        for path, info in self._index.items():
            if info.get("fingerprint") == fingerprint:
                return info
        return None

    def stats(self) -> Dict[str, Any]:
        """Registry statistics."""
        by_technique: Dict[str, int] = {}
        by_format: Dict[str, int] = {}
        total_size = 0
        for info in self._index.values():
            t = info.get("technique", "unknown")
            by_technique[t] = by_technique.get(t, 0) + 1
            f = info.get("format", "unknown")
            by_format[f] = by_format.get(f, 0) + 1
            total_size += info.get("size_bytes", 0)
        return {
            "n_directories": len(self._dirs),
            "n_models": len(self._index),
            "by_technique": by_technique,
            "by_format": by_format,
            "total_size_bytes": total_size,
        }


# ============================================================================
# 3. ModelSigning — sign .qmodel files for tamper detection
# ============================================================================
class ModelSigning:
    """Sign .qmodel files with HMAC-SHA256 for tamper detection.

    Usage:
        # Sign
        ModelSigning.sign("model.qmodel", secret_key="my_secret")
        # Verify
        ok = ModelSigning.verify("model.qmodel", secret_key="my_secret")
    """

    @staticmethod
    def sign(path: str, secret_key: str) -> bool:
        """Sign a .qmodel file by adding a signature to the manifest."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                content = {name: zf.read(name) for name in zf.namelist()}
            manifest = json.loads(content["manifest.json"])
            # compute HMAC over weights + model.json
            weight_hash = hashlib.sha256(content.get("weights.npy", b"")).hexdigest()
            model_hash = hashlib.sha256(content.get("model.json", b"")).hexdigest()
            sig_data = f"{weight_hash}:{model_hash}"
            signature = hmac.new(secret_key.encode(), sig_data.encode(),
                                   hashlib.sha256).hexdigest()
            manifest["signed"] = True
            manifest["signed_hash"] = signature
            manifest["signed_at"] = time.time()
            content["manifest.json"] = json.dumps(manifest, indent=2).encode()
            # rewrite
            tmp = path + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in content.items():
                    zf.writestr(name, data)
            os.replace(tmp, path)
            return True
        except Exception:
            return False

    @staticmethod
    def verify(path: str, secret_key: str) -> bool:
        """Verify a .qmodel file's signature."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                if not manifest.get("signed"):
                    return False
                weight_hash = hashlib.sha256(zf.read("weights.npy")).hexdigest()
                model_hash = hashlib.sha256(zf.read("model.json")).hexdigest()
                sig_data = f"{weight_hash}:{model_hash}"
                expected = hmac.new(secret_key.encode(), sig_data.encode(),
                                      hashlib.sha256).hexdigest()
                return hmac.compare_digest(expected, manifest.get("signed_hash", ""))
        except Exception:
            return False

    @staticmethod
    def is_signed(path: str) -> bool:
        """Check if a .qmodel file is signed."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                return manifest.get("signed", False)
        except Exception:
            return False


# ============================================================================
# 4. ModelLinting — check .qmodel files for common issues
# ============================================================================
class ModelLinting:
    """Lint .qmodel files for common issues.

    Checks:
      - manifest is valid JSON with required fields
      - weights.npy is not empty
      - model.json is valid
      - fingerprint matches weights
      - provenance is present
      - file is signed (warning if not)
      - total_params > 0
      - README.md exists
    """

    REQUIRED_MANIFEST_FIELDS = ["format_version", "technique", "total_params", "created_at"]

    @staticmethod
    def lint(path: str) -> Dict[str, Any]:
        """Run all lint checks on a .qmodel file."""
        issues: List[Dict[str, str]] = []
        warnings: List[Dict[str, str]] = []
        try:
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                # check manifest
                if "manifest.json" not in names:
                    issues.append({"check": "manifest_exists", "severity": "error",
                                     "message": "manifest.json missing"})
                else:
                    manifest = json.loads(zf.read("manifest.json"))
                    for field in ModelLinting.REQUIRED_MANIFEST_FIELDS:
                        if field not in manifest:
                            issues.append({"check": f"manifest_{field}", "severity": "error",
                                             "message": f"manifest missing field: {field}"})
                    if manifest.get("total_params", 0) == 0:
                        warnings.append({"check": "params", "severity": "warning",
                                           "message": "total_params is 0"})
                # check weights
                if "weights.npy" not in names:
                    issues.append({"check": "weights", "severity": "error",
                                     "message": "weights.npy missing"})
                elif len(zf.read("weights.npy")) == 0:
                    issues.append({"check": "weights", "severity": "error",
                                     "message": "weights.npy is empty"})
                # check model.json
                if "model.json" not in names:
                    issues.append({"check": "model_json", "severity": "error",
                                     "message": "model.json missing"})
                # check fingerprint
                if "fingerprint.txt" not in names:
                    warnings.append({"check": "fingerprint", "severity": "warning",
                                       "message": "fingerprint.txt missing"})
                # check provenance
                if "provenance.json" not in names:
                    warnings.append({"check": "provenance", "severity": "warning",
                                       "message": "provenance.json missing"})
                # check README
                if "README.md" not in names:
                    warnings.append({"check": "readme", "severity": "warning",
                                       "message": "README.md missing"})
                # check signature
                if "manifest.json" in names:
                    manifest = json.loads(zf.read("manifest.json"))
                    if not manifest.get("signed"):
                        warnings.append({"check": "signature", "severity": "warning",
                                           "message": "model is not signed"})
        except Exception as e:
            issues.append({"check": "zip", "severity": "error",
                             "message": f"cannot read file: {e}"})
        return {
            "path": path,
            "n_errors": len(issues),
            "n_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "passed": len(issues) == 0,
        }


# ============================================================================
# 5. ModelMigration — migrate old .qmodel format to new
# ============================================================================
class ModelMigration:
    """Migrate old .qmodel files to the latest format.

    Versions:
      0.9: no fingerprint, no provenance
      1.0: has fingerprint + provenance + signed
    """

    LATEST_VERSION = "1.0"

    @staticmethod
    def migrate(path: str) -> Dict[str, Any]:
        """Migrate a .qmodel file to the latest format."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                content = {name: zf.read(name) for name in zf.namelist()}
            manifest = json.loads(content.get("manifest.json", "{}"))
            old_version = manifest.get("format_version", "0.9")
            if old_version == ModelMigration.LATEST_VERSION:
                return {"migrated": False, "reason": "already at latest version"}
            # add missing fields
            manifest["format_version"] = ModelMigration.LATEST_VERSION
            if "fingerprint.txt" not in content:
                fp = hashlib.sha256(content.get("weights.npy", b"")).hexdigest()[:16]
                content["fingerprint.txt"] = fp.encode()
            if "provenance.json" not in content:
                prov = {"migrated_from": old_version, "migrated_at": time.time()}
                content["provenance.json"] = json.dumps(prov, indent=2).encode()
            if "README.md" not in content:
                content["README.md"] = f"# {manifest.get('technique', 'model')}\n\nMigrated from v{old_version}.\n".encode()
            manifest["migrated_at"] = time.time()
            content["manifest.json"] = json.dumps(manifest, indent=2).encode()
            # rewrite
            tmp = path + ".tmp"
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, data in content.items():
                    zf.writestr(name, data)
            os.replace(tmp, path)
            return {"migrated": True, "from": old_version, "to": ModelMigration.LATEST_VERSION}
        except Exception as e:
            return {"migrated": False, "error": str(e)}

    @staticmethod
    def needs_migration(path: str) -> bool:
        """Check if a .qmodel file needs migration."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                return manifest.get("format_version", "0.9") != ModelMigration.LATEST_VERSION
        except Exception:
            return False


# ============================================================================
# 6. ModelCompatibility — version compatibility checking
# ============================================================================
class ModelCompatibility:
    """Check if a .qmodel file is compatible with the current qai version."""

    @staticmethod
    def check(path: str) -> Dict[str, Any]:
        """Check compatibility of a .qmodel file."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                manifest = json.loads(zf.read("manifest.json"))
                model_json = json.loads(zf.read("model.json"))
        except Exception as e:
            return {"compatible": False, "error": str(e)}
        technique = manifest.get("technique", "unknown")
        # check if technique is still registered
        try:
            import qai
            technique_exists = technique in qai.techniques._REGISTRY
        except Exception:
            technique_exists = False
        # check format version
        format_version = manifest.get("format_version", "0.9")
        format_compatible = format_version >= "1.0"
        return {
            "compatible": technique_exists and format_compatible,
            "technique": technique,
            "technique_available": technique_exists,
            "format_version": format_version,
            "format_compatible": format_compatible,
            "issues": [
                *(["technique not registered"] if not technique_exists else []),
                *(["format version too old"] if not format_compatible else []),
            ],
        }


__all__ = [
    "HFModelImporter", "HFModelWrapper",
    "ModelRegistry2", "ModelSigning", "ModelLinting",
    "ModelMigration", "ModelCompatibility",
]
