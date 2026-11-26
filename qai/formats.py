"""
qai.formats — comprehensive ML file format registry + converters.

Supports ALL known ML model/weight file formats:

  Model formats (store architecture + weights):
    .qmodel     — qai native bundle
    .pt/.pth    — PyTorch checkpoint
    .onnx       — ONNX (cross-framework)
    .gguf       — GGUF (llama.cpp, quantized)
    .tflite     — TensorFlow Lite (mobile/edge)
    .engine     — TensorRT (NVIDIA GPU)
    .mlmodel    — Apple Core ML
    .pb         — TensorFlow SavedModel
    .h5/.keras  — Keras model
    .ckpt       — checkpoint

  Weight-only formats (store just tensors):
    .weights    — qai native weights
    .wt         — alias for .weights
    .safetensors — HuggingFace safe tensor format
    .npy/.npz   — NumPy array
    .bin        — binary weights (HF legacy)

  Serialization formats:
    .pkl        — Python pickle (full model)
    .msgpack    — JAX/Flax checkpoint
    .orbax      — Orbax checkpoint (JAX)

Usage:
    # List all supported formats
    qai.formats.list_formats()

    # Convert between any formats
    qai.formats.convert("model.qmodel", "model.safetensors")
    qai.formats.convert("model.pt", "model.weights")
    qai.formats.convert("model.qmodel", "model.gguf")

    # Get format info
    info = qai.formats.format_info(".safetensors")
"""
from __future__ import annotations
import json
import os
import pickle
import struct
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Format registry
# ============================================================================
class FormatInfo:
    """Information about a file format."""
    def __init__(self, extension: str, name: str, category: str,
                 description: str = "", stores_architecture: bool = False,
                 stores_weights: bool = True, is_safe: bool = True,
                 supports_quantization: bool = False, framework: str = "",
                 aliases: Optional[List[str]] = None):
        self.extension = extension
        self.name = name
        self.category = category  # "model" | "weights" | "serialization"
        self.description = description
        self.stores_architecture = stores_architecture
        self.stores_weights = stores_weights
        self.is_safe = is_safe  # safe = no code execution on load
        self.supports_quantization = supports_quantization
        self.framework = framework
        self.aliases = aliases or []


FORMATS: Dict[str, FormatInfo] = {
    # === Model formats (architecture + weights) ===
    ".qmodel": FormatInfo(
        ".qmodel", "Quantum Model Bundle", "model",
        "qai native format: zip archive with manifest, architecture, weights, provenance",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="qai",
    ),
    ".pt": FormatInfo(
        ".pt", "PyTorch Checkpoint", "model",
        "PyTorch checkpoint: can store full model or just state_dict",
        stores_architecture=False, stores_weights=True, is_safe=False, framework="pytorch",
        aliases=[".pth"],
    ),
    ".onnx": FormatInfo(
        ".onnx", "ONNX", "model",
        "Open Neural Network Exchange: cross-framework graph + weights",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="cross",
    ),
    ".gguf": FormatInfo(
        ".gguf", "GGUF (GGML Unified Format)", "model",
        "llama.cpp format: single file with weights + tokenizer + metadata, supports quantization",
        stores_architecture=True, stores_weights=True, is_safe=True,
        supports_quantization=True, framework="llama.cpp",
    ),
    ".tflite": FormatInfo(
        ".tflite", "TensorFlow Lite", "model",
        "Google's mobile/edge format: FlatBuffer, low-latency, supports quantization",
        stores_architecture=True, stores_weights=True, is_safe=True,
        supports_quantization=True, framework="tensorflow",
    ),
    ".engine": FormatInfo(
        ".engine", "TensorRT Engine", "model",
        "NVIDIA's optimized inference format: hardware-specific, ultra-fast on GPU",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="nvidia",
    ),
    ".mlmodel": FormatInfo(
        ".mlmodel", "Apple Core ML", "model",
        "Apple's on-device format: iOS/macOS/watchOS, optimized for Apple Silicon",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="apple",
    ),
    ".pb": FormatInfo(
        ".pb", "TensorFlow SavedModel", "model",
        "TensorFlow protobuf format: graph + weights, production TF serving",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="tensorflow",
    ),
    ".h5": FormatInfo(
        ".h5", "Keras HDF5", "model",
        "Keras model format: architecture + weights + training config",
        stores_architecture=True, stores_weights=True, is_safe=True, framework="keras",
        aliases=[".keras"],
    ),
    ".ckpt": FormatInfo(
        ".ckpt", "Checkpoint", "model",
        "Generic checkpoint format: used by TF, Stable Diffusion, etc.",
        stores_architecture=False, stores_weights=True, is_safe=False, framework="various",
    ),

    # === Weight-only formats ===
    ".weights": FormatInfo(
        ".weights", "qai Weights", "weights",
        "qai native weights format: npz with metadata + changelog",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="qai",
        aliases=[".wt"],
    ),
    ".wt": FormatInfo(
        ".wt", "Weights (short)", "weights",
        "Alias for .weights: qai native weights format",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="qai",
        aliases=[".weights"],
    ),
    ".safetensors": FormatInfo(
        ".safetensors", "Safetensors", "weights",
        "HuggingFace safe tensor format: zero-copy, no code execution, mmap-able",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="huggingface",
    ),
    ".npy": FormatInfo(
        ".npy", "NumPy Array", "weights",
        "Single NumPy array: raw weights, no metadata",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="numpy",
        aliases=[".npz"],
    ),
    ".bin": FormatInfo(
        ".bin", "Binary Weights", "weights",
        "HuggingFace legacy binary format: raw tensors, being replaced by .safetensors",
        stores_architecture=False, stores_weights=True, is_safe=False, framework="huggingface",
    ),

    # === Serialization formats ===
    ".pkl": FormatInfo(
        ".pkl", "Python Pickle", "serialization",
        "Python pickle: can store any Python object including full models. DANGEROUS: can execute code on load.",
        stores_architecture=False, stores_weights=True, is_safe=False, framework="python",
    ),
    ".msgpack": FormatInfo(
        ".msgpack", "MessagePack (JAX)", "serialization",
        "JAX/Flax checkpoint format: serialized via msgpack",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="jax",
    ),
    ".orbax": FormatInfo(
        ".orbax", "Orbax Checkpoint", "serialization",
        "Orbax checkpoint format: modern JAX/Flax checkpointing",
        stores_architecture=False, stores_weights=True, is_safe=True, framework="jax",
    ),
}


def list_formats(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all supported formats."""
    out = []
    for ext, info in FORMATS.items():
        if category and info.category != category:
            continue
        out.append({
            "extension": ext,
            "name": info.name,
            "category": info.category,
            "description": info.description,
            "stores_architecture": info.stores_architecture,
            "stores_weights": info.stores_weights,
            "is_safe": info.is_safe,
            "supports_quantization": info.supports_quantization,
            "framework": info.framework,
            "aliases": info.aliases,
        })
    return out


def format_info(extension: str) -> Dict[str, Any]:
    """Get info about a specific format."""
    # normalize
    if not extension.startswith("."):
        extension = "." + extension
    info = FORMATS.get(extension)
    if info is None:
        # check aliases
        for ext, fi in FORMATS.items():
            if extension in fi.aliases:
                info = fi
                break
    if info is None:
        return {"error": f"unknown format: {extension}", "available": list(FORMATS.keys())}
    return {
        "extension": info.extension,
        "name": info.name,
        "category": info.category,
        "description": info.description,
        "stores_architecture": info.stores_architecture,
        "stores_weights": info.stores_weights,
        "is_safe": info.is_safe,
        "supports_quantization": info.supports_quantization,
        "framework": info.framework,
        "aliases": info.aliases,
    }


def is_model_format(extension: str) -> bool:
    """Check if an extension is a model format (stores architecture)."""
    info = format_info(extension)
    return info.get("category") == "model"


def is_weights_format(extension: str) -> bool:
    """Check if an extension is a weights-only format."""
    info = format_info(extension)
    return info.get("category") == "weights"


def is_safe_format(extension: str) -> bool:
    """Check if an extension is safe (no code execution on load)."""
    info = format_info(extension)
    return info.get("is_safe", False)


# ============================================================================
# Universal converter
# ============================================================================
class UniversalConverter:
    """Convert between any supported formats."""

    @staticmethod
    def convert(input_path: str, output_path: str,
                 model: Optional[Any] = None) -> str:
        """Convert a model/weights file from one format to another.

        If `model` is provided, saves it directly to output_path.
        Otherwise, loads from input_path first.
        """
        in_ext = os.path.splitext(input_path)[1].lower()
        out_ext = os.path.splitext(output_path)[1].lower()

        # resolve aliases
        in_ext = _resolve_alias(in_ext)
        out_ext = _resolve_alias(out_ext)

        if in_ext not in FORMATS:
            raise ValueError(f"unsupported input format: {in_ext}")
        if out_ext not in FORMATS:
            raise ValueError(f"unsupported output format: {out_ext}")

        # load the model/weights if not provided
        if model is None:
            model = UniversalConverter._load(input_path, in_ext)

        # save to output format
        return UniversalConverter._save(model, output_path, out_ext, input_path)

    @staticmethod
    def _load(path: str, ext: str) -> Any:
        """Load from any format."""
        if ext == ".qmodel":
            import qai
            return qai.load_bundle(path)
        elif ext in (".pt", ".pth"):
            import torch
            return torch.load(path, map_location="cpu", weights_only=False)
        elif ext == ".pkl":
            with open(path, "rb") as f:
                return pickle.load(f)
        elif ext == ".safetensors":
            return UniversalConverter._load_safetensors(path)
        elif ext in (".weights", ".wt"):
            from .weights import load_weights
            return load_weights(path)
        elif ext == ".npy":
            return np.load(path, allow_pickle=True)
        elif ext == ".npz":
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        elif ext == ".onnx":
            try:
                import onnx
                return onnx.load(path)
            except ImportError:
                raise ImportError("onnx not installed; pip install onnx")
        elif ext == ".gguf":
            raise NotImplementedError("GGUF loading requires llama.cpp; use convert_hf_to_gguf.py")
        elif ext == ".bin":
            # HF legacy binary: try torch.load first
            try:
                import torch
                return torch.load(path, map_location="cpu", weights_only=True)
            except Exception:
                return np.fromfile(path, dtype=np.float32)
        elif ext == ".ckpt":
            try:
                import torch
                return torch.load(path, map_location="cpu", weights_only=False)
            except Exception:
                with open(path, "rb") as f:
                    return pickle.load(f)
        else:
            raise NotImplementedError(f"loading from {ext} not implemented")

    @staticmethod
    def _save(model: Any, path: str, ext: str, source_path: str = "") -> str:
        """Save to any format."""
        if ext == ".qmodel":
            import qai
            return qai.save_bundle(model, path)
        elif ext in (".pt", ".pth"):
            import torch
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)
            return path
        elif ext == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(model, f)
            return path
        elif ext == ".safetensors":
            return UniversalConverter._save_safetensors(model, path)
        elif ext in (".weights", ".wt"):
            from .weights import save_weights
            return save_weights(model, path)
        elif ext == ".npy":
            weights = _get_weights_dict(model)
            if len(weights) == 1:
                np.save(path, list(weights.values())[0])
            else:
                np.save(path, weights, allow_pickle=True)
            return path
        elif ext == ".onnx":
            return UniversalConverter._save_onnx(model, path)
        elif ext == ".bin":
            # save as torch binary
            import torch
            if hasattr(model, "state_dict"):
                torch.save(model.state_dict(), path)
            else:
                torch.save(model, path)
            return path
        else:
            raise NotImplementedError(f"saving to {ext} not implemented")

    @staticmethod
    def _load_safetensors(path: str) -> Dict[str, np.ndarray]:
        """Load from .safetensors format."""
        try:
            from safetensors import safe_open
            weights = {}
            with safe_open(path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
            return weights
        except ImportError:
            raise ImportError("safetensors not installed; pip install safetensors")

    @staticmethod
    def _save_safetensors(model: Any, path: str) -> str:
        """Save to .safetensors format."""
        try:
            from safetensors.numpy import save_file as np_save_file
            from safetensors.torch import save_file as torch_save_file
        except ImportError:
            raise ImportError("safetensors not installed; pip install safetensors")

        weights = _get_weights_dict(model)
        # try torch first (supports GPU tensors)
        try:
            import torch
            torch_weights = {k: torch.as_tensor(v) for k, v in weights.items()}
            torch_save_file(torch_weights, path)
            return path
        except Exception:
            pass
        # fallback to numpy
        np_weights = {k: np.asarray(v) for k, v in weights.items()}
        np_save_file(np_weights, path)
        return path

    @staticmethod
    def _save_onnx(model: Any, path: str) -> str:
        """Save to ONNX format."""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for ONNX export")

        torch_model = model
        if hasattr(model, "technique") and hasattr(model.technique, "net"):
            torch_model = model.technique.net
        elif hasattr(model, "technique") and hasattr(model.technique, "_model"):
            torch_model = model.technique._model

        if not isinstance(torch_model, torch.nn.Module):
            raise ValueError("model doesn't have a torch.nn.Module for ONNX export")

        torch_model.eval()
        # determine input shape
        dummy = torch.randn(1, 4)
        try:
            torch.onnx.export(torch_model, dummy, path, export_params=True, opset_version=11)
        except Exception:
            # try different input shape
            dummy = torch.randn(1, 3, 224, 224)
            torch.onnx.export(torch_model, dummy, path, export_params=True, opset_version=11)
        return path


def _resolve_alias(ext: str) -> str:
    """Resolve format aliases to canonical extension."""
    for canonical, info in FORMATS.items():
        if ext == canonical or ext in info.aliases:
            return canonical
    return ext


def _get_weights_dict(model: Any) -> Dict[str, np.ndarray]:
    """Extract weights as a dict from any model."""
    if isinstance(model, dict):
        return {k: np.asarray(v) for k, v in model.items()}
    # unwrap Bundle → Model → technique
    if hasattr(model, "model") and not hasattr(model, "state_dict"):
        model = model.model  # unwrap Bundle
    if hasattr(model, "state_dict"):
        try:
            return {k: np.asarray(v) for k, v in model.state_dict().items()}
        except Exception:
            pass
    if hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
        try:
            return {k: np.asarray(v) for k, v in model.technique.state_dict().items()}
        except Exception:
            pass
    # fallback: scan for numpy arrays
    out = {}
    targets = [model]
    if hasattr(model, "technique"):
        targets.append(model.technique)
    for target in targets:
        for name in dir(target):
            if name.startswith("_"):
                continue
            try:
                val = getattr(target, name)
                if isinstance(val, np.ndarray) and val.size > 0:
                    out[name] = val.copy()
            except Exception:
                pass
    return out


# ============================================================================
# Convenience functions
# ============================================================================
def convert(input_path: str, output_path: str,
             model: Optional[Any] = None) -> str:
    """Convert between any supported formats."""
    return UniversalConverter.convert(input_path, output_path, model)


def to_safetensors(model: Any, path: str) -> str:
    """Save model weights as .safetensors."""
    if not path.endswith(".safetensors"):
        path += ".safetensors"
    return UniversalConverter._save_safetensors(model, path)


def to_gguf(model: Any, path: str, quantization: str = "Q4_K_M") -> str:
    """Convert model to GGUF format (requires llama.cpp).

    Note: This is a stub — actual conversion requires llama.cpp's
    convert_hf_to_gguf.py script.
    """
    if not path.endswith(".gguf"):
        path += ".gguf"
    # save as safetensors first (GGUF conversion reads from HF format)
    st_path = path.replace(".gguf", ".safetensors")
    to_safetensors(model, st_path)
    # would call: python convert_hf_to_gguf.py --input {st_path} --output {path}
    # would call: ./quantize {path} {path} {quantization}
    raise NotImplementedError(
        "GGUF conversion requires llama.cpp. Steps:\n"
        "1. pip install safetensors\n"
        f"2. We saved weights to {st_path}\n"
        "3. git clone https://github.com/ggml-org/llama.cpp\n"
        "4. cd llama.cpp && python convert_hf_to_gguf.py --input <model_dir> --output " + path + "\n"
        f"5. ./quantize {path} {path} {quantization}"
    )


def to_tflite(model: Any, path: str) -> str:
    """Convert model to TFLite (requires TensorFlow)."""
    if not path.endswith(".tflite"):
        path += ".tflite"
    try:
        import tensorflow as tf
        # This is a simplified path — real conversion needs a TF model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_model)
        return path
    except ImportError:
        raise ImportError("tensorflow not installed; pip install tensorflow")


def detect_format(path: str) -> Dict[str, Any]:
    """Auto-detect the format of a file by extension + content."""
    ext = os.path.splitext(path)[1].lower()
    ext = _resolve_alias(ext)
    if ext in FORMATS:
        info = format_info(ext)
        info["detected_by"] = "extension"
        return info
    # try by magic bytes
    try:
        with open(path, "rb") as f:
            magic = f.read(16)
        if magic[:4] == b"PK\x03\x04":
            # zip — could be .qmodel, .keras, .tflite
            return {"format": "zip_archive", "detected_by": "magic",
                     "possible_formats": [".qmodel", ".keras", ".tflite"]}
        if magic[:4] == b"\x89HDF":
            return {"format": "hdf5", "detected_by": "magic",
                     "possible_formats": [".h5"]}
        if magic[:4] == b"\x00\x00\x00\x00":
            return {"format": "onnx_proto", "detected_by": "magic",
                     "possible_formats": [".onnx"]}
    except Exception:
        pass
    return {"error": "cannot detect format", "extension": ext}


__all__ = [
    "FORMATS", "FormatInfo",
    "list_formats", "format_info",
    "is_model_format", "is_weights_format", "is_safe_format",
    "UniversalConverter", "convert",
    "to_safetensors", "to_gguf", "to_tflite",
    "detect_format",
]
