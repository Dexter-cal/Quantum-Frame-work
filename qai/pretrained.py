"""
qai.pretrained — pretrained weight loading for qai models.

Provides:
  - qai.build(technique=..., pretrained=True) — auto-download weights
  - load_pretrained(model, source) — load weights from URL/file/HF Hub
  - download_weights(technique, version) — fetch from HuggingFace Hub

Usage:
    import qai
    # Auto-download ImageNet weights for ResNet
    m = qai.build(technique="resnet50", pretrained=True)
    # Or load from a specific source
    m = qai.build(technique="resnet")
    qai.load_pretrained(m, "https://example.com/resnet50.pth")
    # Or from HuggingFace Hub
    m = qai.build(technique="bert")
    qai.load_pretrained(m, "huggingface:bert-base-uncased")
"""
from __future__ import annotations
import os, hashlib, urllib.request, json
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


# Registry of known pretrained checkpoints
# Format: technique_name → {version: source}
PRETRAINED_REGISTRY: Dict[str, Dict[str, str]] = {
    "resnet": {
        "imagenet": "torchvision:resnet18",
        "imagenet50": "torchvision:resnet50",
        "imagenet101": "torchvision:resnet101",
    },
    "resnet50": {"imagenet": "torchvision:resnet50"},
    "vit": {
        "imagenet": "huggingface:google/vit-base-patch16-224",
        "imagenet21k": "huggingface:google/vit-base-patch16-224-in21k",
    },
    "bert": {
        "base": "huggingface:bert-base-uncased",
        "large": "huggingface:bert-large-uncased",
    },
    "gpt2": {
        "small": "huggingface:gpt2",
        "medium": "huggingface:gpt2-medium",
        "large": "huggingface:gpt2-large",
        "xl": "huggingface:gpt2-xl",
    },
    "t5": {
        "small": "huggingface:t5-small",
        "base": "huggingface:t5-base",
        "large": "huggingface:t5-large",
        "3b": "huggingface:t5-3b",
        "11b": "huggingface:t5-11b",
    },
    "llama": {
        "7b": "huggingface:meta-llama/Llama-2-7b-hf",
        "13b": "huggingface:meta-llama/Llama-2-13b-hf",
        "70b": "huggingface:meta-llama/Llama-2-70b-hf",
    },
    "whisper": {
        "tiny": "huggingface:openai/whisper-tiny",
        "base": "huggingface:openai/whisper-base",
        "small": "huggingface:openai/whisper-small",
        "medium": "huggingface:openai/whisper-medium",
        "large": "huggingface:openai/whisper-large",
    },
    "clip": {
        "base": "huggingface:openai/clip-vit-base-patch32",
        "large": "huggingface:openai/clip-vit-large-patch14",
    },
}


def _cache_dir() -> str:
    """Get the qai cache directory (~/.qai/pretrained)."""
    cache = os.path.expanduser("~/.qai/pretrained")
    os.makedirs(cache, exist_ok=True)
    return cache


def _url_to_filename(url: str) -> str:
    """Hash a URL to a filename."""
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{h}.bin"


def _download(url: str, dest: str, verbose: bool = True) -> str:
    """Download a file with progress reporting."""
    if os.path.exists(dest):
        if verbose:
            print(f"[qai.pretrained] cached at {dest}")
        return dest
    if verbose:
        print(f"[qai.pretrained] downloading {url} → {dest}")
    try:
        urllib.request.urlretrieve(url, dest)
        return dest
    except Exception as e:
        raise RuntimeError(
            f"[WHAT] Failed to download pretrained weights from {url}\n"
            f"[WHY] {e}\n"
            f"[FIX] Check your internet connection, or download manually and "
            f"load with qai.load_pretrained(model, '/path/to/weights.pth')"
        )


def _load_huggingface(repo_id: str, verbose: bool = True) -> Dict:
    """Load weights from HuggingFace Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "[WHAT] HuggingFace Hub integration requires `huggingface_hub`\n"
            "[FIX] pip install huggingface_hub"
        )
    # Try pytorch_model.bin first, then model.safetensors
    for filename in ["pytorch_model.bin", "model.safetensors"]:
        try:
            path = hf_hub_download(repo_id=repo_id, filename=filename)
            if verbose:
                print(f"[qai.pretrained] downloaded {repo_id}/{filename}")
            if filename.endswith(".bin"):
                return torch.load(path, map_location="cpu")
            else:
                # safetensors
                from safetensors.torch import load_file
                return load_file(path)
        except Exception:
            continue
    raise RuntimeError(f"Could not find weights for {repo_id}")


def _load_torchvision(model_name: str, verbose: bool = True) -> Dict:
    """Load torchvision pretrained weights."""
    try:
        import torchvision.models as tv_models
    except ImportError:
        raise ImportError(
            "[WHAT] torchvision integration requires `torchvision`\n"
            "[FIX] pip install torchvision"
        )
    # Map name to torchvision model function
    model_map = {
        "resnet18": tv_models.resnet18,
        "resnet50": tv_models.resnet50,
        "resnet101": tv_models.resnet101,
        "mobilenet_v2": tv_models.mobilenet_v2,
        "mobilenet_v3_small": tv_models.mobilenet_v3_small,
        "mobilenet_v3_large": tv_models.mobilenet_v3_large,
        "efficientnet_b0": tv_models.efficientnet_b0,
        "densenet121": tv_models.densenet121,
    }
    if model_name not in model_map:
        raise ValueError(f"Unknown torchvision model: {model_name}")
    # Get the model with pretrained weights
    try:
        # Newer torchvision API (0.13+)
        from torchvision.models import get_model_weights
        weights = get_model_weights(model_name).DEFAULT
        model = model_map[model_name](weights=weights)
    except (ImportError, AttributeError):
        # Older API
        model = model_map[model_name](pretrained=True)
    return model.state_dict()


def load_pretrained(model, source: str, verbose: bool = True):
    """Load pretrained weights into a qai model.

    Args:
        model: a qai.Model
        source: one of:
            - "huggingface:repo_id" (e.g. "huggingface:bert-base-uncased")
            - "torchvision:model_name" (e.g. "torchvision:resnet50")
            - "http://..." or "https://..." URL
            - "/path/to/weights.pth" local file
        verbose: if True, print progress

    Returns:
        The model with loaded weights (in-place).
    """
    if not _TORCH:
        raise ImportError("[WHAT] load_pretrained requires PyTorch")

    if source.startswith("huggingface:"):
        repo_id = source[len("huggingface:"):]
        state_dict = _load_huggingface(repo_id, verbose)
    elif source.startswith("torchvision:"):
        model_name = source[len("torchvision:"):]
        state_dict = _load_torchvision(model_name, verbose)
    elif source.startswith(("http://", "https://")):
        cache_path = os.path.join(_cache_dir(), _url_to_filename(source))
        _download(source, cache_path, verbose)
        state_dict = torch.load(cache_path, map_location="cpu")
    elif os.path.exists(source):
        if verbose:
            print(f"[qai.pretrained] loading from {source}")
        state_dict = torch.load(source, map_location="cpu")
    else:
        raise ValueError(f"Unknown pretrained source: {source}")

    # Load into model
    tech = model.technique
    _load_state_into_technique(tech, state_dict, verbose)
    tech._trained = True
    return model


def _load_state_into_technique(tech, state_dict, verbose: bool = True):
    """Load a state_dict into a technique's torch submodules."""
    if not hasattr(tech, "parameters"):
        return
    # Try to match keys to submodule state dicts
    # First, try direct load on each submodule
    for attr_name in vars(tech):
        attr = getattr(tech, attr_name)
        if isinstance(attr, torch.nn.Module):
            # Try to load matching keys
            sub_state = {}
            for k, v in state_dict.items():
                if k.startswith(f"{attr_name}."):
                    sub_state[k[len(f"{attr_name}."):]] = v
                elif k in attr.state_dict():
                    sub_state[k] = v
            if sub_state:
                try:
                    attr.load_state_dict(sub_state, strict=False)
                    if verbose:
                        print(f"  loaded {len(sub_state)} keys into {attr_name}")
                except Exception as e:
                    if verbose:
                        print(f"  could not load {attr_name}: {e}")


def download_weights(technique: str, version: str = "default",
                      verbose: bool = True) -> Dict:
    """Download pretrained weights for a technique.

    Args:
        technique: technique name (e.g. "resnet", "bert")
        version: version name (e.g. "imagenet", "base")
        verbose: if True, print progress

    Returns:
        The state dict.
    """
    if technique not in PRETRAINED_REGISTRY:
        raise ValueError(
            f"No pretrained weights registered for '{technique}'. "
            f"Available: {sorted(PRETRAINED_REGISTRY.keys())}"
        )
    versions = PRETRAINED_REGISTRY[technique]
    if version == "default":
        version = next(iter(versions))
    if version not in versions:
        raise ValueError(f"Unknown version '{version}' for '{technique}'. "
                         f"Available: {sorted(versions.keys())}")
    source = versions[version]
    if source.startswith("huggingface:"):
        return _load_huggingface(source[len("huggingface:"):], verbose)
    elif source.startswith("torchvision:"):
        return _load_torchvision(source[len("torchvision:"):], verbose)
    elif source.startswith(("http://", "https://")):
        cache_path = os.path.join(_cache_dir(), _url_to_filename(source))
        _download(source, cache_path, verbose)
        return torch.load(cache_path, map_location="cpu")
    raise ValueError(f"Unknown source type: {source}")


def list_pretrained(technique: Optional[str] = None) -> Dict:
    """List available pretrained weights.

    Args:
        technique: if given, list versions for this technique only

    Returns:
        Dict of technique → {version: source}
    """
    if technique is not None:
        return PRETRAINED_REGISTRY.get(technique, {})
    return PRETRAINED_REGISTRY


__all__ = ["load_pretrained", "download_weights", "list_pretrained",
           "PRETRAINED_REGISTRY"]
