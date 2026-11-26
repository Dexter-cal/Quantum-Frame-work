"""
qai.hf_converter -- convert HuggingFace models to/from qai format.

Download a model from HuggingFace Hub → convert to .qmodel → use with qai.

Usage:
    # Download from HF and convert
    model = qai.HFConverter.download_and_convert("bert-base-uncased")
    # → saves as bert-base-uncased_v1.0.0_2024-01-15.qmodel

    # Or convert an already-loaded HF model
    from transformers import AutoModel
    hf_model = AutoModel.from_pretrained("bert-base-uncased")
    qai_path = qai.HFConverter.convert(hf_model, "my_model.qmodel")

    # Load the converted model back
    model = qai.load_model_file("my_model.qmodel")
"""
from __future__ import annotations
import os
import pickle
import time
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class HFConverter:
    """Convert HuggingFace models to/from qai .qmodel format.

    Supports:
      - AutoModel (any HF transformer)
      - AutoModelForSequenceClassification
      - AutoModelForCausalLM
      - AutoModelForImageClassification
      - Pipelines (text-classification, etc.)
      - Tokenizers (saved alongside the model)
    """

    @staticmethod
    def download_and_convert(model_name: str,
                               output_dir: str = "./hf_models",
                               task: str = "feature-extraction",
                               include_tokenizer: bool = True) -> str:
        """Download a model from HuggingFace Hub and convert to .qmodel.

        Args:
            model_name: HuggingFace model name (e.g. "bert-base-uncased")
            output_dir: Where to save the .qmodel file
            task: What task the model is for
            include_tokenizer: If True, save the tokenizer alongside

        Returns:
            Path to the .qmodel file
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed; pip install transformers"
            )

        os.makedirs(output_dir, exist_ok=True)

        # Download model + tokenizer
        print(f"[HFConverter] downloading {model_name}...")
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name) if include_tokenizer else None

        # Generate filename
        safe_name = model_name.replace("/", "_").replace("-", "_")
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{safe_name}_v1.0.0_{timestamp}.qmodel"
        path = os.path.join(output_dir, filename)

        # Convert
        HFConverter._save_as_qmodel(model, path, model_name, task, tokenizer)
        print(f"[HFConverter] saved: {path}")
        return path

    @staticmethod
    def convert(hf_model: Any,
                 output_path: str,
                 model_name: str = "hf_model",
                 task: str = "feature-extraction",
                 tokenizer: Any = None) -> str:
        """Convert an already-loaded HuggingFace model to .qmodel.

        Args:
            hf_model: A HuggingFace model (AutoModel, etc.)
            output_path: Where to save the .qmodel file
            model_name: Name for the manifest
            task: Task type
            tokenizer: Optional tokenizer to bundle

        Returns:
            Path to the .qmodel file
        """
        HFConverter._save_as_qmodel(hf_model, output_path, model_name, task, tokenizer)
        return output_path

    @staticmethod
    def _save_as_qmodel(hf_model: Any, path: str, model_name: str,
                          task: str, tokenizer: Any = None) -> None:
        """Internal: save a HF model as a .qmodel zip archive."""
        import zipfile

        # Extract state dict
        if hasattr(hf_model, "state_dict"):
            sd = hf_model.state_dict()
        else:
            sd = {}

        # Convert tensors to numpy
        weights: Dict[str, np.ndarray] = {}
        for k, v in sd.items():
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    weights[k] = v.detach().cpu().numpy()
                else:
                    weights[k] = np.asarray(v)
            except ImportError:
                weights[k] = np.asarray(v)

        # Compute fingerprint
        if weights:
            flat = np.concatenate([w.flatten() for w in weights.values()])
            fingerprint = hashlib.sha256(flat.tobytes()).hexdigest()[:16]
        else:
            fingerprint = "unknown"

        # Count params
        n_params = sum(w.size for w in weights.values())

        # Build manifest
        manifest = {
            "format_version": "2.0",
            "created_at": time.time(),
            "created_at_iso": datetime.now().isoformat(),
            "technique": "huggingface",
            "hf_model_name": model_name,
            "task": task,
            "description": f"HuggingFace model: {model_name}",
            "author": "huggingface",
            "contents": {
                "manifest.json": True,
                "model.json": True,
                "weights.npy": True,
                "README.md": True,
            },
            "weight_format": "pytorch_sd",
            "total_params": n_params,
            "has_tokenizer": tokenizer is not None,
            "signed": False,
        }

        # Model config
        model_config = {}
        if hasattr(hf_model, "config"):
            config = hf_model.config
            for attr in dir(config):
                if not attr.startswith("_"):
                    try:
                        val = getattr(config, attr)
                        if isinstance(val, (str, int, float, bool, list, dict, type(None))):
                            model_config[attr] = val
                    except Exception:
                        pass

        model_json = {
            "technique": "huggingface",
            "hf_model_name": model_name,
            "task": task,
            "config": model_config,
            "weight_keys": list(weights.keys())[:20],
        }

        # README
        readme = f"""# {model_name}

Converted from HuggingFace Hub to qai .qmodel format.

- **Original model**: {model_name}
- **Task**: {task}
- **Parameters**: {n_params:,}
- **Fingerprint**: {fingerprint}
- **Converted at**: {datetime.now().isoformat()}
- **Has tokenizer**: {tokenizer is not None}

## Usage

```python
import qai
model = qai.load_model_file("{os.path.basename(path)}")
```
"""

        # Write zip
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
            zf.writestr("model.json", json.dumps(model_json, indent=2, default=str))
            # save weights as pickled dict (handles variable-size arrays)
            zf.writestr("weights.npy", pickle.dumps(weights))
            zf.writestr("README.md", readme)
            zf.writestr("fingerprint.txt", fingerprint)
            # save tokenizer if provided
            if tokenizer is not None:
                tokenizer_data = pickle.dumps(tokenizer)
                zf.writestr("tokenizer.pkl", tokenizer_data)

    @staticmethod
    def inspect_qmodel(path: str) -> Dict[str, Any]:
        """Inspect a .qmodel file that was converted from HuggingFace."""
        import zipfile, json
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            model_json = json.loads(zf.read("model.json"))
        return {
            "hf_model_name": manifest.get("hf_model_name", "unknown"),
            "task": manifest.get("task", "unknown"),
            "n_params": manifest.get("total_params", 0),
            "fingerprint": zf.read("fingerprint.txt").decode().strip() if "fingerprint.txt" in zf.namelist() else "unknown",
            "has_tokenizer": manifest.get("has_tokenizer", False),
            "config_keys": list(model_json.get("config", {}).keys())[:10],
        }

    @staticmethod
    def list_hf_cache() -> List[Dict[str, Any]]:
        """List models downloaded from HuggingFace (in the default cache)."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        if not os.path.exists(cache_dir):
            return []
        models = []
        for name in os.listdir(cache_dir):
            if name.startswith("models--"):
                model_name = name.replace("models--", "").replace("--", "/")
                full_path = os.path.join(cache_dir, name)
                size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for dirpath, _, filenames in os.walk(full_path)
                    for f in filenames
                )
                models.append({
                    "name": model_name,
                    "cache_path": full_path,
                    "size_mb": round(size / (1024 * 1024), 2),
                })
        return models


__all__ = ["HFConverter"]
