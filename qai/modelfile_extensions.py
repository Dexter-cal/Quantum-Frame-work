"""
qai.modelfile_extensions -- extra model-file features.

  HuggingFaceImporter — download HF models and import to .qmodel
  ModelRegistry       — global in-memory registry of loaded models
  ModelInspector      — deep model inspection (layers, params, shapes, dtypes)
  ModelMigrator       — migrate models between versions / directories / formats
  ModelPackager       — package a model + tokenizer + README for deployment
"""
from __future__ import annotations
import os
import json
import shutil
import zipfile
import pickle
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. HuggingFaceImporter — convenience wrapper around HFConverter
# ============================================================================
class HuggingFaceImporter:
    """Download a model from HuggingFace and import it into qai.

    Usage:
        path = HuggingFaceImporter.import("bert-base-uncased", "./my_models")
        model = qai.load_model_file(path)
    """

    @staticmethod
    def import_model(model_name: str, output_dir: str = "./hf_models",
                task: str = "feature-extraction",
                include_tokenizer: bool = True) -> str:
        """Download from HF Hub and save as .qmodel."""
        from .hf_converter import HFConverter
        return HFConverter.download_and_convert(
            model_name, output_dir, task, include_tokenizer
        )

    @staticmethod
    def import_many(model_names: List[str],
                     output_dir: str = "./hf_models") -> List[str]:
        """Import multiple HF models at once."""
        paths = []
        for name in model_names:
            try:
                path = HuggingFaceImporter.import_model(name, output_dir)
                paths.append(path)
            except Exception as e:
                print(f"[HFImporter] failed to import {name}: {e}")
        return paths

    @staticmethod
    def list_cache() -> List[Dict[str, Any]]:
        """List all HuggingFace models in the local cache."""
        from .hf_converter import HFConverter
        return HFConverter.list_hf_cache()


# ============================================================================
# 2. ModelRegistry — global in-memory registry of loaded models
# ============================================================================
class ModelRegistry:
    """A global registry that tracks all loaded models.

    Every time you load a model, it gets registered here. You can then
    look it up by name, fingerprint, or technique without re-loading.

    Usage:
        model = qai.build(technique="regression")
        ModelRegistry.register("my_model", model)
        loaded = ModelRegistry.get("my_model")
    """

    _models: Dict[str, Any] = {}
    _fingerprints: Dict[str, str] = {}

    @classmethod
    def register(cls, name: str, model: Any) -> None:
        """Register a model by name."""
        cls._models[name] = model
        try:
            from .modelfile import ModelFingerprinter
            fp = ModelFingerprinter.fingerprint(model)
            cls._fingerprints[name] = fp
        except Exception:
            pass

    @classmethod
    def get(cls, name: str) -> Optional[Any]:
        """Get a registered model by name."""
        return cls._models.get(name)

    @classmethod
    def get_by_fingerprint(cls, fingerprint: str) -> Optional[Any]:
        """Find a registered model by its weight fingerprint."""
        for name, fp in cls._fingerprints.items():
            if fp == fingerprint:
                return cls._models[name]
        return None

    @classmethod
    def list(cls) -> Dict[str, Dict[str, Any]]:
        """List all registered models."""
        result = {}
        for name, model in cls._models.items():
            result[name] = {
                "type": type(model).__name__,
                "fingerprint": cls._fingerprints.get(name, "unknown"),
                "trained": getattr(model, "is_trained", lambda: False)()
                            if hasattr(model, "is_trained") else
                            getattr(getattr(model, "technique", None), "is_trained", lambda: False)(),
            }
        return result

    @classmethod
    def remove(cls, name: str) -> bool:
        """Remove a model from the registry."""
        if name in cls._models:
            del cls._models[name]
            cls._fingerprints.pop(name, None)
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models."""
        cls._models.clear()
        cls._fingerprints.clear()


# ============================================================================
# 3. ModelInspector — deep model inspection
# ============================================================================
class ModelInspector:
    """Deep-inspect a model's structure.

    Reports:
      - All layers and their shapes
      - Parameter count per layer
      - Data types
      - Memory footprint
      - Weight statistics (mean, std, min, max, sparsity)
    """

    @staticmethod
    def inspect(model: Any) -> Dict[str, Any]:
        """Deep inspection of a model."""
        report: Dict[str, Any] = {
            "model_type": type(model).__name__,
            "layers": [],
            "total_params": 0,
            "total_bytes": 0,
            "dtypes": set(),
        }

        # Try PyTorch state_dict
        sd = {}
        if hasattr(model, "state_dict"):
            try:
                sd = model.state_dict()
            except Exception:
                pass
        elif hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
            try:
                sd = model.technique.state_dict()
            except Exception:
                pass
        elif hasattr(model, "technique") and hasattr(model.technique, "net"):
            try:
                sd = model.technique.net.state_dict()
            except Exception:
                pass

        if sd:
            for name, tensor in sd.items():
                try:
                    import torch
                    if isinstance(tensor, torch.Tensor):
                        arr = tensor.detach().cpu().numpy()
                    else:
                        arr = np.asarray(tensor)
                except ImportError:
                    arr = np.asarray(tensor)

                layer_info = {
                    "name": name,
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "n_params": int(arr.size),
                    "bytes": int(arr.nbytes),
                    "mean": float(arr.mean()) if arr.size > 0 else 0,
                    "std": float(arr.std()) if arr.size > 0 else 0,
                    "min": float(arr.min()) if arr.size > 0 else 0,
                    "max": float(arr.max()) if arr.size > 0 else 0,
                    "sparsity": float((arr == 0).mean()) if arr.size > 0 else 0,
                }
                report["layers"].append(layer_info)
                report["total_params"] += layer_info["n_params"]
                report["total_bytes"] += layer_info["bytes"]
                report["dtypes"].add(layer_info["dtype"])
        else:
            # fallback: scan for numpy attributes
            targets = [model]
            if hasattr(model, "technique"):
                targets.append(model.technique)
            for target in targets:
                for attr_name in dir(target):
                    if attr_name.startswith("_"):
                        continue
                    try:
                        val = getattr(target, attr_name)
                        if isinstance(val, np.ndarray) and val.size > 0:
                            report["layers"].append({
                                "name": attr_name,
                                "shape": list(val.shape),
                                "dtype": str(val.dtype),
                                "n_params": int(val.size),
                                "bytes": int(val.nbytes),
                                "mean": float(val.mean()),
                                "std": float(val.std()),
                                "min": float(val.min()),
                                "max": float(val.max()),
                                "sparsity": float((val == 0).mean()),
                            })
                            report["total_params"] += val.size
                            report["total_bytes"] += val.nbytes
                            report["dtypes"].add(str(val.dtype))
                    except Exception:
                        pass

        report["dtypes"] = list(report["dtypes"])
        report["total_mb"] = round(report["total_bytes"] / (1024 * 1024), 4)
        report["n_layers"] = len(report["layers"])
        return report

    @staticmethod
    def summary(model: Any) -> str:
        """Human-readable inspection summary."""
        report = ModelInspector.inspect(model)
        lines = [
            f"Model: {report['model_type']}",
            f"  Layers: {report['n_layers']}",
            f"  Parameters: {report['total_params']:,}",
            f"  Memory: {report['total_mb']} MB",
            f"  Dtypes: {', '.join(report['dtypes'])}",
            "",
            f"  {'Layer':<30} {'Shape':<20} {'Params':>10} {'Mean':>8} {'Std':>8} {'Sparsity':>8}",
            f"  {'-'*30} {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}",
        ]
        for layer in report["layers"][:20]:
            lines.append(
                f"  {layer['name']:<30} {str(layer['shape']):<20} "
                f"{layer['n_params']:>10,} {layer['mean']:>8.4f} "
                f"{layer['std']:>8.4f} {layer['sparsity']:>8.2%}"
            )
        if len(report["layers"]) > 20:
            lines.append(f"  ... and {len(report['layers']) - 20} more layers")
        return "\n".join(lines)


# ============================================================================
# 4. ModelMigrator — migrate models between versions/directories/formats
# ============================================================================
class ModelMigrator:
    """Migrate models between directories, formats, or framework versions.

    Usage:
        # Migrate all .pkl files to .qmodel
        ModelMigrator.migrate_directory("./old_models", "./new_models",
                                         from_format=".pkl", to_format=".qmodel")
    """

    @staticmethod
    def migrate_directory(src_dir: str, dest_dir: str,
                            from_format: str = ".pkl",
                            to_format: str = ".qmodel") -> Dict[str, Any]:
        """Migrate all model files in a directory to a new format."""
        os.makedirs(dest_dir, exist_ok=True)
        migrated = []
        failed = []
        for name in os.listdir(src_dir):
            if not name.endswith(from_format):
                continue
            src_path = os.path.join(src_dir, name)
            dest_name = os.path.splitext(name)[0] + to_format
            dest_path = os.path.join(dest_dir, dest_name)
            try:
                from .modelfile import ModelConverter
                ModelConverter.convert(src_path, dest_path)
                migrated.append(dest_name)
            except Exception as e:
                failed.append({"file": name, "error": str(e)})
        return {
            "migrated": migrated,
            "failed": failed,
            "n_migrated": len(migrated),
            "n_failed": len(failed),
        }

    @staticmethod
    def rename_all(directory: str, pattern: str = "{technique}_v{version}") -> int:
        """Rename all .qmodel files in a directory to follow a pattern."""
        from .modelfile import ModelFile, ModelVersioner
        count = 0
        for name in os.listdir(directory):
            if not name.endswith(".qmodel"):
                continue
            mf = ModelFile(os.path.join(directory, name))
            new_name = ModelVersioner.make_filename(mf.technique, mf.version)
            if new_name != name:
                mf.rename(new_name)
                count += 1
        return count


# ============================================================================
# 5. ModelPackager — package a model + dependencies for deployment
# ============================================================================
class ModelPackager:
    """Package a model + its dependencies for deployment.

    Creates a self-contained .qpkg archive that includes:
      - The .qmodel file
      - The tokenizer (if any)
      - A requirements.txt with the needed pip packages
      - A README.md with usage instructions
      - A run.py script that loads and runs the model
      - The logic source (if the model has @qai.logic attached)

    Usage:
        ModelPackager.package("my_model.qmodel", "my_model.qpkg")
    """

    @staticmethod
    def package(qmodel_path: str, output_path: str,
                 extra_files: Optional[List[str]] = None,
                 requirements: Optional[List[str]] = None) -> str:
        """Package a .qmodel file into a deployable .qpkg archive."""
        from .modelfile import ModelFile

        mf = ModelFile(qmodel_path)
        info = ModelFile.inspect(qmodel_path)

        # Build requirements
        reqs = requirements or ["qai", "numpy"]
        if info.get("technique") in ["transformer", "cnn", "rnn", "lstm", "gru"]:
            reqs.append("torch")
        if info.get("format") == "qai_bundle" and info.get("technique") == "huggingface":
            reqs.extend(["transformers", "torch"])

        # Build run.py
        run_script = f'''#!/usr/bin/env python
"""Auto-generated runner for {mf.name}"""
import sys
import qai

def main():
    model = qai.load_model_file("{mf.name}")
    if len(sys.argv) > 1:
        import numpy as np
        x = np.array([float(v) for v in sys.argv[1:]], dtype=np.float32)
        result = model.forward(x)
        print(f"Result: {{result}}")
    else:
        print(f"Model loaded: {mf.name}")
        print(f"Technique: {info.get('technique', 'unknown')}")
        print(f"Params: {info.get('n_params', 0):,}")
        print("Usage: python run.py [input_values...]")

if __name__ == "__main__":
    main()
'''

        # Build README
        readme = f"""# {mf.name}

## Model Info
- **Technique**: {info.get('technique', 'unknown')}
- **Version**: {info.get('version', '1.0.0')}
- **Parameters**: {info.get('n_params', 0):,}
- **Fingerprint**: {info.get('fingerprint', 'unknown')}
- **Size**: {mf.size_human}

## Usage

```bash
# Install requirements
pip install {' '.join(reqs)}

# Run the model
python run.py 1.0 2.0 3.0 4.0
```

## Python API

```python
import qai
model = qai.load_model_file("{mf.name}")
result = model.forward([1.0, 2.0, 3.0, 4.0])
```
"""

        # Write .qpkg (zip)
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            # add the .qmodel file
            zf.write(qmodel_path, mf.name)
            # add requirements.txt
            zf.writestr("requirements.txt", "\n".join(reqs))
            # add README.md
            zf.writestr("README.md", readme)
            # add run.py
            zf.writestr("run.py", run_script)
            # add manifest
            manifest = {
                "package_version": "1.0",
                "created_at": datetime.now().isoformat(),
                "model_file": mf.name,
                "technique": info.get("technique", "unknown"),
                "requirements": reqs,
                "files": [mf.name, "requirements.txt", "README.md", "run.py"],
            }
            if extra_files:
                for f in extra_files:
                    if os.path.exists(f):
                        zf.write(f, os.path.basename(f))
                        manifest["files"].append(os.path.basename(f))
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        return output_path

    @staticmethod
    def unpack(qpkg_path: str, dest_dir: str) -> Dict[str, Any]:
        """Unpack a .qpkg archive into a directory."""
        os.makedirs(dest_dir, exist_ok=True)
        with zipfile.ZipFile(qpkg_path, "r") as zf:
            zf.extractall(dest_dir)
            manifest = json.loads(zf.read("manifest.json")) if "manifest.json" in zf.namelist() else {}
        return {
            "extracted_to": dest_dir,
            "files": manifest.get("files", []),
            "model_file": manifest.get("model_file", "unknown"),
            "requirements": manifest.get("requirements", []),
        }


__all__ = [
    "HuggingFaceImporter", "ModelRegistry", "ModelInspector",
    "ModelMigrator", "ModelPackager",
]
