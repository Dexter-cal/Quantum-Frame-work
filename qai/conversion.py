"""
qai.conversion -- unified bidirectional model conversion API.

Convert any qai.Model (or sklearn estimator, or torch.nn.Module) to any of
10 supported formats:

    onnx, torchscript, tflite, coreml, tensorrt, openvino,
    gguf, safetensors, pickle, qai_native

Top-level API:

    qai.conversion.export(model, format, path=None, **kwargs) -> str
    qai.conversion.import_model(path, format=None, **kwargs) -> Any
    qai.conversion.supported_formats() -> list[str]
    qai.conversion.available_backends() -> dict[str, bool]
    qai.conversion.compare(model, format, sample_input=None) -> dict
    qai.conversion.benchmark_formats(model, formats=None, sample_input=None) -> list
    qai.conversion.ConversionReport(...)

Per-format exporters are also callable classes you can use directly:

    ONNXExporter, TorchScriptExporter, TFLiteExporter, CoreMLExporter,
    TensorRTExporter, OpenVINOExporter, GGUFExporter, SafeTensorsExporter,
    PickleExporter, QAINativeExporter

Design rules (very important):

  * No hard external dependencies. Every conversion library (onnx, skl2onnx,
    torch, tensorflow, coremltools, openvino, tensorrt, gguf, safetensors)
    is imported LAZILY inside the exporter that needs it.
  * If the library isn't installed, the exporter prints a helpful install
    message and raises ImportError -- it never crashes with a bare stack trace.
  * `pickle` and `qai_native` ALWAYS work (no external deps).
  * Each exporter auto-detects what the model is (sklearn estimator,
    torch.nn.Module, or qai.Model wrapping a Technique) and applies the right
    conversion path.
"""
from __future__ import annotations

import os
import sys
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ============================================================================
# Format registry
# ============================================================================

#: All formats supported by this module, in canonical order.
SUPPORTED_FORMATS: List[str] = [
    "onnx",
    "torchscript",
    "tflite",
    "coreml",
    "tensorrt",
    "openvino",
    "gguf",
    "safetensors",
    "pickle",
    "qai_native",
]

#: Maps each format to its canonical file extension.
_FORMAT_EXTENSIONS: Dict[str, str] = {
    "onnx":        ".onnx",
    "torchscript": ".pt",
    "tflite":      ".tflite",
    "coreml":      ".mlpackage",
    "tensorrt":    ".engine",
    "openvino":    ".xml",
    "gguf":        ".gguf",
    "safetensors": ".safetensors",
    "pickle":      ".pkl",
    "qai_native":  ".qmodel",
}

#: Aliases that callers may pass for `format=`.
_FORMAT_ALIASES: Dict[str, str] = {
    "pkl":      "pickle",
    "pickle":   "pickle",
    "qmodel":   "qai_native",
    "qai":      "qai_native",
    "ts":       "torchscript",
    "torch":    "torchscript",
    "pt":       "torchscript",
    "jit":      "torchscript",
    "tf":       "tflite",
    "mlmodel":  "coreml",
    "mlpackage": "coreml",
    "engine":   "tensorrt",
    "trt":      "tensorrt",
    "ir":       "openvino",
    "st":       "safetensors",
    "safe":     "safetensors",
}


def supported_formats() -> list:
    """Return a copy of the list of all supported export formats.

    Example:
        >>> from qai.conversion import supported_formats
        >>> supported_formats()
        ['onnx', 'torchscript', 'tflite', 'coreml', 'tensorrt',
         'openvino', 'gguf', 'safetensors', 'pickle', 'qai_native']
    """
    return list(SUPPORTED_FORMATS)


def _normalize_format(fmt: Optional[str]) -> str:
    """Normalize a user-supplied format string to a canonical key.

    Accepts the canonical names in SUPPORTED_FORMATS, plus aliases listed in
    _FORMAT_ALIASES. Strips a leading "." if present (so ".onnx" → "onnx").
    """
    if fmt is None:
        raise ValueError(
            "[qai.conversion] format is None. "
            f"Supported formats: {supported_formats()}"
        )
    key = fmt.strip().lower().lstrip(".")
    if key in SUPPORTED_FORMATS:
        return key
    if key in _FORMAT_ALIASES:
        return _FORMAT_ALIASES[key]
    raise ValueError(
        f"[qai.conversion] Unsupported format '{fmt}'. "
        f"Supported formats: {supported_formats()}"
    )


# ============================================================================
# Helpers: model type detection
# ============================================================================

def _is_qai_model(model: Any) -> bool:
    """Detect if `model` is a qai.Model (has a `.technique` wrapping a Technique)."""
    return (
        hasattr(model, "technique")
        and hasattr(model.technique, "forward")
        and hasattr(model.technique, "fit")
    )


def _get_qai_technique(model: Any):
    """Return the underlying Technique from a qai.Model (or self if already one)."""
    if _is_qai_model(model):
        return model.technique
    return model


def _get_torch_module(model: Any):
    """Return a torch.nn.Module if the model wraps one, else None.

    Recognised patterns:
      * the model is itself a torch.nn.Module
      * qai.Model whose technique.net / .model / .net is a torch.nn.Module
      * qai.Model whose technique._model is a torch.nn.Module (rare)
    """
    try:
        import torch  # noqa: F401
        import torch.nn as nn
    except ImportError:
        return None

    if isinstance(model, nn.Module):
        return model

    if _is_qai_model(model):
        t = model.technique
        # Check most-common attribute names first.
        for attr_name in ("net", "model", "_model", "module", "_module"):
            attr = getattr(t, attr_name, None)
            if isinstance(attr, nn.Module):
                return attr
        # Some techniques store the torch module inside a list/tuple attribute.
        for attr_name in vars(t):
            if attr_name.startswith("_") and attr_name in ("_trained", "_last_X", "_last_y"):
                continue
            try:
                val = getattr(t, attr_name)
            except Exception:
                continue
            if isinstance(val, nn.Module):
                return val
    return None


def _get_sklearn_estimator(model: Any):
    """Return a sklearn BaseEstimator if the model wraps one, else None."""
    try:
        from sklearn.base import BaseEstimator
    except ImportError:
        return None

    if isinstance(model, BaseEstimator):
        return model

    if _is_qai_model(model):
        t = model.technique
        # Most common attribute names used by qai techniques that wrap sklearn.
        for attr_name in ("_model", "model", "estimator", "clf", "reg", "sk_model"):
            attr = getattr(t, attr_name, None)
            if isinstance(attr, BaseEstimator):
                return attr
    return None


def _detect_model_kind(model: Any) -> str:
    """Return one of: 'qai', 'torch', 'sklearn', 'unknown'."""
    if _is_qai_model(model):
        return "qai"
    try:
        import torch.nn as nn
        if isinstance(model, nn.Module):
            return "torch"
    except ImportError:
        pass
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return "sklearn"
    except ImportError:
        pass
    return "unknown"


def _infer_input_shape(model: Any) -> Tuple[int, ...]:
    """Best-effort guess at the model's input shape (batch=1)."""
    # qai.Model stores a training-data signature.
    sig = getattr(model, "_train_data_signature", None) if _is_qai_model(model) else None
    if sig is None and hasattr(model, "technique"):
        sig = getattr(model.technique, "_train_data_signature", None)
    if sig and sig.get("n_features"):
        n = sig["n_features"]
        if isinstance(n, (tuple, list)):
            return (1,) + tuple(n)
        return (1, int(n))
    # Sklearn estimators know n_features_in_.
    sk = _get_sklearn_estimator(model)
    if sk is not None:
        n = getattr(sk, "n_features_in_", None)
        if n is not None:
            return (1, int(n))
    # Torch model — try a sample tensor.
    tm = _get_torch_module(model)
    if tm is not None:
        for name, param in tm.named_parameters():
            if name.endswith("weight") and param.dim() == 2:
                return (1, int(param.shape[1]))
    return (1, 4)


def _human_size(n_bytes: int) -> str:
    """Render a byte count as a friendly human-readable string."""
    if n_bytes is None:
        return "?"
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024.0
    return f"{n:.1f} TB"


# ============================================================================
# Base exporter
# ============================================================================

class _BaseExporter:
    """Common functionality for every per-format exporter.

    Subclasses override `export(model, path, **kwargs)` and set `format_name`.
    Instances are callable:  ONNXExporter()(model, path)  ->  path string.
    """
    format_name: str = ""

    def __call__(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        return self.export(model, path, **kwargs)

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        raise NotImplementedError(f"{self.__class__.__name__}.export() not implemented")

    # -- helpers --------------------------------------------------------
    def _ensure_path(self, path: Optional[str], default_ext: str) -> str:
        """Return a usable path: defaults to ./qai_model.<ext> in CWD."""
        if path is None or path == "":
            path = os.path.join(os.getcwd(), f"qai_model{default_ext}")
        # If path is a directory, append a default filename.
        if os.path.isdir(path):
            path = os.path.join(path, f"qai_model{default_ext}")
        # If no extension, append the canonical one.
        _, ext = os.path.splitext(path)
        if not ext:
            path = path + default_ext
        return path

    @staticmethod
    def _missing_dep(lib: str, hint: str = "") -> ImportError:
        """Build (and print) a helpful ImportError for a missing conversion library."""
        msg = (
            f"[qai.conversion] library '{lib}' is not installed -- "
            f"cannot convert to this format.\n"
            f"  FIX: pip install {lib}"
        )
        if hint:
            msg += f"\n  NOTE: {hint}"
        print(msg, file=sys.stderr)
        return ImportError(msg)


# ============================================================================
# Pickle -- always works (no external deps)
# ============================================================================

class PickleExporter(_BaseExporter):
    """Export any model as a Python pickle file.

    Works for every Python object (qai.Model, sklearn estimator, torch module,
    pipeline, etc.). Pickle is NOT portable across Python versions and can
    execute arbitrary code on load -- prefer qai_native or onnx for production.
    """
    format_name = "pickle"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["pickle"])
        protocol = kwargs.get("protocol", pickle.HIGHEST_PROTOCOL)
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=protocol)
        return path


# ============================================================================
# QAI native -- always works (uses qai.core.persistence)
# ============================================================================

class QAINativeExporter(_BaseExporter):
    """Save as a .qmodel file using qai's own persistence layer.

    Internally uses qai.core.persistence.save_model (which is pickle-based
    for the moment but keeps the door open for a real archive format later).
    """
    format_name = "qai_native"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["qai_native"])
        try:
            # Lazy import so users without the rest of qai still get pickle.
            from .core.persistence import save_model
        except Exception:
            # Fall back to plain pickle if persistence is somehow unavailable.
            with open(path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            return path
        return save_model(model, path)


# ============================================================================
# ONNX -- torch.onnx.export for torch models, skl2onnx for sklearn models
# ============================================================================

class ONNXExporter(_BaseExporter):
    """Export to ONNX.

    Decision tree:
      * If the model wraps a torch.nn.Module  ->  torch.onnx.export
      * Elif the model wraps a sklearn estimator ->  skl2onnx.convert_sklearn
      * Else ->  TypeError
    """
    format_name = "onnx"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["onnx"])

        # --- torch path ---
        torch_model = _get_torch_module(model)
        if torch_model is not None:
            try:
                import torch
            except ImportError:
                raise self._missing_dep("torch", "needed for ONNX export of torch models")

            torch_model.eval()
            input_shape = kwargs.get("input_shape") or _infer_input_shape(model)
            try:
                dummy = torch.randn(*input_shape)
            except Exception:
                dummy = torch.randn(1, 4)

            opset_version = kwargs.get("opset_version", 13)
            export_kwargs = dict(
                export_params=kwargs.get("export_params", True),
                opset_version=opset_version,
                do_constant_folding=kwargs.get("do_constant_folding", True),
                input_names=kwargs.get("input_names", ["input"]),
                output_names=kwargs.get("output_names", ["output"]),
                dynamic_axes=kwargs.get("dynamic_axes", {
                    "input": {0: "batch"},
                    "output": {0: "batch"},
                }),
            )

            # Newer torch (>=2.9) defaults to the dynamo-based exporter,
            # which requires `onnxscript` and `onnx`. Older torch uses the
            # TorchScript-based exporter which only needs torch itself. We
            # try the legacy exporter first when `onnx` isn't installed, so
            # that torch-only environments still produce ONNX files.
            try:
                # Inspect whether `dynamo` is a parameter to export().
                import inspect
                _sig_params = inspect.signature(torch.onnx.export).parameters
                if "dynamo" in _sig_params and not _try_import("onnx"):
                    export_kwargs["dynamo"] = False
            except Exception:
                pass

            try:
                torch.onnx.export(torch_model, dummy, path, **export_kwargs)
            except TypeError:
                # Very old torch (<2.0) doesn't have `dynamo` -- retry without it.
                export_kwargs.pop("dynamo", None)
                torch.onnx.export(torch_model, dummy, path, **export_kwargs)
            except Exception as e:
                # Modern torch (>=2.9) requires the `onnx` package even for
                # dynamo=False. If onnx itself is missing, raise a clear
                # install message rather than a cryptic OnnxExporterError.
                if "onnx" in str(e).lower() and not _try_import("onnx"):
                    raise self._missing_dep(
                        "onnx",
                        "torch.onnx.export requires the `onnx` Python package "
                        "(even when dynamo=False). pip install onnx"
                    )
                raise
            return path

        # --- sklearn path ---
        sk_model = _get_sklearn_estimator(model)
        if sk_model is not None:
            try:
                from skl2onnx import convert_sklearn
            except ImportError:
                raise self._missing_dep(
                    "skl2onnx",
                    "needed for ONNX export of sklearn models. Also requires onnx."
                )
            try:
                from skl2onnx.common.data_types import FloatTensorType
            except ImportError:
                raise self._missing_dep("skl2onnx", "FloatTensorType not found in skl2onnx")

            n_features = getattr(sk_model, "n_features_in_", None) or kwargs.get("n_features", 4)
            initial_types = kwargs.get("initial_types") or [
                ("input", FloatTensorType([None, int(n_features)]))
            ]
            onnx_model = convert_sklearn(
                sk_model,
                initial_types=initial_types,
                target_opset=kwargs.get("opset_version", 13),
            )
            with open(path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            return path

        # --- unknown ---
        kind = _detect_model_kind(model)
        raise TypeError(
            f"[qai.conversion] Cannot export {type(model).__name__} (kind='{kind}') to ONNX. "
            f"No torch.nn.Module or sklearn estimator was detected on the model."
        )


# ============================================================================
# TorchScript -- torch.jit.script / torch.jit.trace
# ============================================================================

class TorchScriptExporter(_BaseExporter):
    """Export a torch-based model as TorchScript (.pt).

    Tries torch.jit.script first (works for static-graph modules); falls back
    to torch.jit.trace if scripting fails or if a sample_input is provided.
    """
    format_name = "torchscript"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["torchscript"])
        try:
            import torch
        except ImportError:
            raise self._missing_dep("torch", "needed for TorchScript export")

        torch_model = _get_torch_module(model)
        if torch_model is None:
            kind = _detect_model_kind(model)
            raise TypeError(
                f"[qai.conversion] Cannot export {type(model).__name__} (kind='{kind}') "
                f"to TorchScript: no torch.nn.Module was detected."
            )
        torch_model.eval()

        sample = kwargs.get("sample_input")
        mode = kwargs.get("mode", "auto")  # "auto" | "script" | "trace"

        scripted = None
        if mode in ("auto", "script") and sample is None:
            try:
                scripted = torch.jit.script(torch_model)
            except Exception as e:
                if mode == "script":
                    raise
                # fall through to trace
                warnings.warn(
                    f"[qai.conversion] torch.jit.script failed ({e}); falling back to trace."
                )

        if scripted is None:
            if sample is None:
                input_shape = _infer_input_shape(model)
                try:
                    sample = torch.randn(*input_shape)
                except Exception:
                    sample = torch.randn(1, 4)
            if not torch.is_tensor(sample):
                sample = torch.as_tensor(np.asarray(sample), dtype=torch.float32)
            scripted = torch.jit.trace(torch_model, sample)

        torch.jit.save(scripted, path)
        return path


# ============================================================================
# SafeTensors -- save model.state_dict() with the safetensors library
# ============================================================================

class SafeTensorsExporter(_BaseExporter):
    """Save the model's state_dict() as a .safetensors file.

    Uses safetensors.torch.save_file if torch is available, else
    safetensors.numpy.save_file. The model must expose a state_dict() either
    directly or via its underlying technique.
    """
    format_name = "safetensors"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["safetensors"])
        try:
            from safetensors.numpy import save_file as np_save_file
        except ImportError:
            raise self._missing_dep(
                "safetensors",
                "needed for SafeTensors export. pip install safetensors"
            )
        # safetensors.torch needs torch itself -- treat it as optional; the
        # numpy backend below is fully functional without it (the old code
        # imported both in one try/except, so torch-less environments got a
        # misleading "safetensors not installed" error).
        try:
            from safetensors.torch import save_file as torch_save_file
        except Exception:
            torch_save_file = None

        weights = self._extract_weights(model)
        if not weights:
            raise ValueError(
                f"[qai.conversion] Cannot export {type(model).__name__} to safetensors: "
                f"no extractable weights (state_dict() returned nothing useful)."
            )

        # Prefer torch tensors (preserves dtype/device info) when possible.
        try:
            import torch
            torch_weights = {}
            for k, v in weights.items():
                if torch.is_tensor(v):
                    torch_weights[k] = v.detach().cpu().clone()
                else:
                    torch_weights[k] = torch.as_tensor(np.asarray(v))
            torch_save_file(torch_weights, path)
            return path
        except Exception as e:
            # Fall back to numpy
            np_weights = {k: np.asarray(v) for k, v in weights.items()}
            # safetensors numpy backend is limited to a few dtypes -- cast safely.
            for k, v in np_weights.items():
                if v.dtype not in (np.float16, np.float32, np.float64,
                                   np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64,
                                   np.bool_):
                    np_weights[k] = v.astype(np.float32)
            np_save_file(np_weights, path)
            return path

    @staticmethod
    def _extract_weights(model: Any) -> Dict[str, Any]:
        """Pull a {name: tensor} state_dict out of any model."""
        # Plain {name: tensor} dicts pass straight through (the old fallback
        # scanned vars(model), which raises TypeError on dicts -- so every
        # exporter crashed on plain weight dicts).
        if isinstance(model, dict):
            return {str(k): v for k, v in model.items()}

        # Unwrap qai.Bundle if we ever get one
        if hasattr(model, "model") and not hasattr(model, "state_dict") \
                and not _is_qai_model(model):
            model = model.model

        # Direct state_dict()
        if hasattr(model, "state_dict"):
            try:
                sd = model.state_dict()
                if sd:
                    return dict(sd)
            except Exception:
                pass

        # qai.Model -> technique.state_dict()
        if _is_qai_model(model):
            t = model.technique
            if hasattr(t, "state_dict"):
                try:
                    sd = t.state_dict()
                    if sd:
                        return dict(sd)
                except Exception:
                    pass
            # Pull torch module's state_dict
            tm = _get_torch_module(model)
            if tm is not None and hasattr(tm, "state_dict"):
                try:
                    sd = tm.state_dict()
                    if sd:
                        return dict(sd)
                except Exception:
                    pass

        # Fallback: scan instance __dict__ for numpy arrays / torch tensors.
        out: Dict[str, Any] = {}
        targets = [model]
        if _is_qai_model(model):
            targets.append(model.technique)
        for target in targets:
            for name in vars(target):
                if name.startswith("__"):
                    continue
                try:
                    val = getattr(target, name)
                except Exception:
                    continue
                if isinstance(val, np.ndarray) and val.size > 0:
                    out[name] = val.copy()
                else:
                    try:
                        import torch
                        if torch.is_tensor(val) and val.numel() > 0:
                            out[name] = val.detach().cpu().clone()
                    except ImportError:
                        pass
        return out


# ============================================================================
# TFLite -- requires tensorflow (or ai-edge-torch for torch models)
# ============================================================================

class TFLiteExporter(_BaseExporter):
    """Export to TensorFlow Lite (.tflite).

    For Keras / TF models: uses tf.lite.TFLiteConverter directly.
    For torch models: tries ai-edge-torch if installed; otherwise instructs
    the user to install it.
    """
    format_name = "tflite"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["tflite"])

        # --- torch -> tflite via ai-edge-torch ---
        torch_model = _get_torch_module(model)
        if torch_model is not None:
            try:
                import ai_edge_torch
            except ImportError:
                raise self._missing_dep(
                    "ai-edge-torch",
                    "for torch → TFLite conversion. pip install ai-edge-torch"
                )
            try:
                import torch
            except ImportError:
                raise self._missing_dep("torch")
            torch_model.eval()
            sample = kwargs.get("sample_input")
            if sample is None:
                input_shape = kwargs.get("input_shape") or _infer_input_shape(model)
                try:
                    sample = torch.randn(*input_shape)
                except Exception:
                    sample = torch.randn(1, 4)
            if not torch.is_tensor(sample):
                sample = torch.as_tensor(np.asarray(sample), dtype=torch.float32)
            tflite_model = ai_edge_torch.convert(torch_model, sample)
            tflite_model.export(path)
            return path

        # --- TF / Keras direct ---
        try:
            import tensorflow as tf
        except ImportError:
            raise self._missing_dep(
                "tensorflow",
                "for TFLite export. pip install tensorflow OR ai-edge-torch (for torch models)."
            )

        # Look for common attributes that hold a Keras model.
        keras_model = model
        if _is_qai_model(model):
            for attr_name in ("keras_model", "_model", "model"):
                attr = getattr(model.technique, attr_name, None)
                if attr is not None and hasattr(attr, "save"):
                    keras_model = attr
                    break

        if hasattr(keras_model, "save") and not callable(getattr(keras_model, "save", None)):
            # If save is a method/property of a non-Keras object, skip
            pass

        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        except Exception:
            # Maybe it's a SavedModel path
            if isinstance(model, str) and os.path.isdir(model):
                converter = tf.lite.TFLiteConverter.from_saved_model(model)
            else:
                raise TypeError(
                    f"[qai.conversion] Cannot export {type(model).__name__} to TFLite: "
                    f"no Keras model or SavedModel path was found."
                )

        if kwargs.get("quantize", False):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_bytes = converter.convert()
        with open(path, "wb") as f:
            f.write(tflite_bytes)
        return path


# ============================================================================
# Core ML -- requires coremltools
# ============================================================================

class CoreMLExporter(_BaseExporter):
    """Export to Apple Core ML (.mlpackage) using coremltools.

    For torch models: traces the module, then converts via coremltools.convert.
    For sklearn models: uses coremltools.converters.sklearn.convert when available.
    """
    format_name = "coreml"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["coreml"])
        try:
            import coremltools as ct
        except ImportError:
            raise self._missing_dep(
                "coremltools",
                "for Apple Core ML export. pip install coremltools"
            )

        # --- torch path ---
        torch_model = _get_torch_module(model)
        if torch_model is not None:
            try:
                import torch
            except ImportError:
                raise self._missing_dep("torch")
            torch_model.eval()
            sample = kwargs.get("sample_input")
            if sample is None:
                input_shape = kwargs.get("input_shape") or _infer_input_shape(model)
                try:
                    sample = torch.randn(*input_shape)
                except Exception:
                    sample = torch.randn(1, 4)
            if not torch.is_tensor(sample):
                sample = torch.as_tensor(np.asarray(sample), dtype=torch.float32)
            try:
                traced = torch.jit.trace(torch_model, sample)
                input_shape_tuple = tuple(sample.shape)
                mlmodel = ct.convert(
                    traced,
                    inputs=[ct.TensorType(name="input", shape=input_shape_tuple)],
                    convert_to=kwargs.get("convert_to", "mlprogram"),
                )
                mlmodel.save(path)
                return path
            except Exception as e:
                raise RuntimeError(
                    f"[qai.conversion] CoreML conversion (torch path) failed: {e}"
                )

        # --- sklearn path ---
        sk_model = _get_sklearn_estimator(model)
        if sk_model is not None:
            try:
                from coremltools.converters import sklearn as sk_converter
                mlmodel = sk_converter.convert(sk_model)
                mlmodel.save(path)
                return path
            except Exception as e:
                raise RuntimeError(
                    f"[qai.conversion] CoreML conversion (sklearn path) failed: {e}"
                )

        kind = _detect_model_kind(model)
        raise TypeError(
            f"[qai.conversion] Cannot export {type(model).__name__} (kind='{kind}') "
            f"to CoreML: no torch.nn.Module or sklearn estimator detected."
        )


# ============================================================================
# TensorRT -- requires tensorrt + (optionally) pycuda
# ============================================================================

class TensorRTExporter(_BaseExporter):
    """Export to a TensorRT engine (.engine) via an ONNX intermediate.

    Pipeline:
      1. Export model to ONNX (uses ONNXExporter).
      2. Parse the ONNX file with tensorrt.OnnxParser.
      3. Build a serialized engine and write it to disk.

    Requires NVIDIA TensorRT and CUDA. Use `available_backends()` to check.
    """
    format_name = "tensorrt"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["tensorrt"])
        try:
            import tensorrt as trt
        except ImportError:
            raise self._missing_dep(
                "tensorrt",
                "NVIDIA TensorRT requires a CUDA install + TensorRT SDK. "
                "See https://developer.nvidia.com/tensorrt"
            )

        # Step 1: produce an ONNX intermediate in the same directory.
        onnx_path = path + ".onnx"
        try:
            ONNXExporter().export(model, onnx_path, **kwargs)
        except Exception as e:
            raise RuntimeError(
                f"[qai.conversion] TensorRT needs an ONNX intermediate, but "
                f"ONNX export failed: {e}"
            )

        # Step 2: parse + build the engine.
        try:
            import pycuda.driver as cuda  # noqa: F401
            import pycuda.autoinit  # noqa: F401  (initialises CUDA context)
        except ImportError:
            raise self._missing_dep(
                "pycuda",
                "TensorRT engine build requires pycuda for CUDA context init. "
                "pip install pycuda"
            )

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(getattr(
            trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH",
            getattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH", 0)
        ))
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            onnx_bytes = f.read()
        if not parser.parse(onnx_bytes):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"[qai.conversion] TensorRT failed to parse ONNX:\n{errors}")

        config = builder.create_builder_config()
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                         kwargs.get("workspace_bytes", 1 << 30))
        except Exception:
            # Older API
            try:
                config.max_workspace_size = kwargs.get("workspace_bytes", 1 << 30)
            except Exception:
                pass

        if kwargs.get("fp16", False):
            try:
                config.set_flag(trt.BuilderFlag.FP16)
            except Exception:
                pass

        try:
            serialized = builder.build_serialized_network(network, config)
        except AttributeError:
            # Older TensorRT API
            engine = builder.build_cuda_engine(network)
            serialized = engine.serialize()

        with open(path, "wb") as f:
            f.write(serialized)
        return path


# ============================================================================
# OpenVINO -- requires openvino (uses ov.convert_model on torch modules)
# ============================================================================

class OpenVINOExporter(_BaseExporter):
    """Export to OpenVINO IR (.xml + .bin) using openvino.convert_model.

    Currently only the torch path is implemented (openvino supports torch
    modules directly). For sklearn models, the user would need to first
    convert to ONNX and then use OpenVINO's ONNX reader (not yet wired here).
    """
    format_name = "openvino"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        if path is None or path == "":
            path = os.path.join(os.getcwd(), "qai_model_openvino.xml")
        else:
            # If a directory is given, append a default filename.
            if os.path.isdir(path):
                path = os.path.join(path, "qai_model_openvino.xml")
            # Make sure it ends in .xml
            if not path.endswith(".xml"):
                path = path + ".xml"

        try:
            import openvino as ov
        except ImportError:
            raise self._missing_dep(
                "openvino",
                "for OpenVINO IR export. pip install openvino"
            )

        torch_model = _get_torch_module(model)
        if torch_model is None:
            raise TypeError(
                f"[qai.conversion] OpenVINO export currently only supports torch models. "
                f"Got {type(model).__name__} (kind='{_detect_model_kind(model)}')."
            )

        try:
            import torch
        except ImportError:
            raise self._missing_dep("torch")

        torch_model.eval()
        sample = kwargs.get("sample_input")
        if sample is None:
            input_shape = kwargs.get("input_shape") or _infer_input_shape(model)
            try:
                sample = torch.randn(*input_shape)
            except Exception:
                sample = torch.randn(1, 4)
        if not torch.is_tensor(sample):
            sample = torch.as_tensor(np.asarray(sample), dtype=torch.float32)

        try:
            ov_model = ov.convert_model(torch_model, example_input=sample)
        except Exception as e:
            raise RuntimeError(
                f"[qai.conversion] OpenVINO convert_model failed: {e}"
            )

        ov.save_model(ov_model, path)
        return path


# ============================================================================
# GGUF -- THE one GGUF writer (metadata-aware) + THE one GGUF reader
# ============================================================================

def _add_metadata_kv(writer: Any, key: str, value: Any, vt: Any) -> None:
    """Write one metadata key/value pair with the right GGUF value type.

    GGUF headers are typed key/value stores; python's bool is a subclass of
    int, so bool MUST be tested before int or every True/False becomes 1/0.
    """
    if isinstance(value, bool):
        writer.add_key_value(key, bool(value), vt.BOOL)
    elif isinstance(value, int):
        writer.add_key_value(key, int(value), vt.INT64)
    elif isinstance(value, float):
        writer.add_key_value(key, float(value), vt.FLOAT64)
    elif isinstance(value, str):
        writer.add_key_value(key, str(value), vt.STRING)
    elif isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(
                "GGUF cannot store an empty array; drop the key or use a "
                "placeholder string")
        else:
            first = value[0]
            if isinstance(first, bool):
                writer.add_key_value(
                    key, [bool(v) for v in value], vt.ARRAY, sub_type=vt.BOOL)
            elif isinstance(first, int):
                writer.add_key_value(
                    key, [int(v) for v in value], vt.ARRAY, sub_type=vt.INT64)
            elif isinstance(first, float):
                writer.add_key_value(
                    key, [float(v) for v in value], vt.ARRAY, sub_type=vt.FLOAT64)
            else:
                writer.add_key_value(
                    key, [str(v) for v in value], vt.ARRAY, sub_type=vt.STRING)
    else:
        writer.add_key_value(key, str(value), vt.STRING)


class GGUFExporter(_BaseExporter):
    """Export a model's weights to a GGUF (.gguf) file -- metadata-aware.

    This is the ONE GGUF writer in qai (qai.formats.to_gguf and
    qai.formats.convert(target=".gguf") are delegates to it).

    What it writes:
      * every extractable weight tensor, stored as float32 (GGUF needs
        contiguous arrays),
      * typed metadata: pass ``metadata={...}`` with str / int / float /
        bool / list values and each pair lands in the GGUF header, readable
        back by ``gguf.GGUFReader.get_field()`` (or any llama.cpp tooling),
      * automatic metadata: ``general.name``, ``general.description``,
        ``qai.tensor_count``, ``qai.exported_by``.

    Backward-compatible kwargs:
      * ``arch``        -- GGUF architecture string (default "qai-model")
      * ``description`` -- general.description text
      * ``block_type``  -- now recorded as metadata ``qai.block_type``
        (the old writer called ``GGUFWriter.add_block_type``, a method that
        no longer exists in the gguf package -- every export crashed)
      * ``quantization``-- recorded as ``qai.quantization_requested`` plus an
        honest ``qai.weights_stored_as`` note. Actual quantization (Q4_K_M
        etc.) is llama.cpp's quantize step, not this container writer.

    NOTE on scope: you get a valid .gguf container holding your weights and
    metadata. A fully *runnable* llama.cpp LLM (tokenizer, chat template,
    quantized tensors) still requires llama.cpp's convert_hf_to_gguf.py on a
    HuggingFace checkpoint.
    """
    format_name = "gguf"

    def export(self, model: Any, path: Optional[str] = None, **kwargs) -> str:
        path = self._ensure_path(path, _FORMAT_EXTENSIONS["gguf"])
        try:
            import gguf
            from gguf import GGUFValueType
        except ImportError:
            raise self._missing_dep(
                "gguf",
                "for writing GGUF files. pip install gguf. "
                "For full llama.cpp conversion you'll also need llama.cpp's "
                "convert_hf_to_gguf.py script."
            )

        weights = SafeTensorsExporter._extract_weights(model)
        if not weights:
            raise ValueError(
                f"[qai.conversion] Cannot export {type(model).__name__} to GGUF: "
                f"no extractable weights."
            )

        writer = gguf.GGUFWriter(path, kwargs.get("arch", "qai-model"))

        # ---- automatic metadata ----
        model_name = kwargs.get("model_name")
        if not model_name:
            model_name = "qai-weights" if isinstance(model, dict) else type(model).__name__
        try:
            writer.add_name(str(model_name))
        except Exception:
            pass
        try:
            writer.add_description(kwargs.get(
                "description",
                "qai framework model (weights + metadata container)"))
        except Exception:
            pass

        # ---- metadata: explicit dict wins, legacy kwargs fold into it ----
        metadata = dict(kwargs.pop("metadata", None) or {})
        block_type = kwargs.pop("block_type", None)
        if block_type is not None:
            metadata.setdefault("qai.block_type", str(block_type))
        quantization = kwargs.pop("quantization", None)
        want_f16 = bool(quantization) and str(quantization).lower() in (
            "f16", "fp16", "half")
        if quantization is not None:
            metadata.setdefault("qai.quantization_requested", str(quantization))
            if want_f16:
                metadata.setdefault("qai.weights_stored_as", "float16")
            else:
                metadata.setdefault(
                    "qai.weights_stored_as",
                    "float32 (unquantized); producing " + str(quantization)
                    + " weights is llama.cpp's quantize step")
        metadata.setdefault("qai.tensor_count", len(weights))
        metadata.setdefault("qai.exported_by", "qai.conversion.GGUFExporter")

        for key, value in metadata.items():
            try:
                _add_metadata_kv(writer, key, value, GGUFValueType)
            except Exception as e:
                warnings.warn(
                    f"[qai.conversion] GGUF: skipping metadata '{key}': {e}")

        # Add each tensor. gguf expects contiguous numpy arrays.
        # (One GGUF writer here; no other module may import gguf to write.)
        store_dtype = np.float16 if want_f16 else np.float32
        for name, tensor in weights.items():
            try:
                import torch
                if torch.is_tensor(tensor):
                    arr = tensor.detach().cpu().float().numpy()
                    arr = arr.astype(store_dtype)
                else:
                    arr = np.asarray(tensor, dtype=store_dtype)
            except ImportError:
                arr = np.asarray(tensor, dtype=store_dtype)

            # gguf requires contiguous arrays
            arr = np.ascontiguousarray(arr)
            safe_name = name.replace(":", "_").replace("/", "_")
            try:
                writer.add_tensor(safe_name, arr)
            except Exception as e:
                warnings.warn(
                    f"[qai.conversion] GGUF: skipping tensor '{name}': {e}"
                )

        # Finalize the file.
        try:
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
        finally:
            try:
                writer.close()
            except Exception:
                pass
        return path


# ============================================================================
# Exporter registry
# ============================================================================

_EXPORTER_REGISTRY: Dict[str, _BaseExporter] = {
    "onnx":        ONNXExporter(),
    "torchscript": TorchScriptExporter(),
    "tflite":      TFLiteExporter(),
    "coreml":      CoreMLExporter(),
    "tensorrt":    TensorRTExporter(),
    "openvino":    OpenVINOExporter(),
    "gguf":        GGUFExporter(),
    "safetensors": SafeTensorsExporter(),
    "pickle":      PickleExporter(),
    "qai_native":  QAINativeExporter(),
}


# ============================================================================
# Top-level export / import
# ============================================================================

def export(model: Any, format: str, path: str = None, **kwargs) -> str:
    """Export `model` to the requested `format`, writing it to `path`.

    Args:
        model:      A qai.Model, a sklearn estimator, a torch.nn.Module,
                    or anything picklable.
        format:     One of SUPPORTED_FORMATS (case-insensitive, with aliases).
        path:       Output file path. If None, defaults to ./qai_model.<ext>
                    in the current working directory.
        **kwargs:   Format-specific options. Common ones:
                      - input_shape: tuple, e.g. (1, 4)
                      - sample_input: array/tensor for tracing
                      - opset_version: ONNX opset (default 13)
                      - dynamic_axes: ONNX dynamic axes
                      - quantize: bool, for TFLite
                      - fp16: bool, for TensorRT

    Returns:
        The output path (string). Useful for chaining:
            path = qai.conversion.export(m, "onnx")
            qai.conversion.import_model(path)

    Raises:
        ValueError:   if `format` is not in SUPPORTED_FORMATS.
        ImportError:  if the conversion library for `format` isn't installed
                      (with a helpful install message printed to stderr).
        TypeError:    if the model type isn't supported by the chosen format.
    """
    fmt = _normalize_format(format)
    exporter = _EXPORTER_REGISTRY[fmt]
    return exporter(model, path, **kwargs)


def import_model(path: str, format: str = None, **kwargs) -> Any:
    """Load a model from `path`, auto-detecting the format if `format=None`.

    Args:
        path:    Path to the saved model file.
        format:  Optional explicit format (one of SUPPORTED_FORMATS, or an
                 alias like "pkl", "qmodel", "pt", "onnx"). If None, the
                 format is detected from the file extension.
        **kwargs: Format-specific options:
                    - map_location: "cpu" | "cuda" (for torch files)
                    - framework:    "numpy" | "torch" | "pt" (for safetensors)

    Returns:
        The loaded model object. Type depends on the format:
          * pickle/qai_native -> the original Python object (qai.Model, etc.)
          * torchscript       -> torch.jit.ScriptModule
          * onnx              -> onnx.ModelProto
          * safetensors       -> dict[str, np.ndarray]
          * coreml            -> coremltools.models.MLModel
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[qai.conversion] file not found: '{path}'")

    if format is None:
        format = _detect_format_from_path(path)
    fmt = _normalize_format(format)

    if fmt == "pickle":
        with open(path, "rb") as f:
            return pickle.load(f)

    if fmt == "qai_native":
        try:
            from .core.persistence import load_model
            return load_model(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)

    if fmt == "torchscript":
        try:
            import torch
        except ImportError:
            raise ImportError(
                "[qai.conversion] torch not installed; cannot load TorchScript file. "
                "pip install torch"
            )
        return torch.jit.load(path, map_location=kwargs.get("map_location", "cpu"))

    if fmt == "onnx":
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "[qai.conversion] onnx not installed; cannot load ONNX file. "
                "pip install onnx"
            )
        return onnx.load(path)

    if fmt == "safetensors":
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError(
                "[qai.conversion] safetensors not installed. pip install safetensors"
            )
        framework = kwargs.get("framework", "numpy")
        weights = {}
        with safe_open(path, framework=framework) as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights

    if fmt == "coreml":
        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "[qai.conversion] coremltools not installed. pip install coremltools"
            )
        return ct.models.MLModel(path)

    if fmt == "gguf":
        try:
            import gguf
        except ImportError:
            raise ImportError(
                "[qai.conversion] gguf not installed. pip install gguf"
            )
        reader = gguf.GGUFReader(path)
        # Return a simple dict of tensor_name -> ndarray
        return {t.name: t.data for t in reader.tensors}

    # tensorrt / openvino / tflite don't have a simple "load" path here.
    raise NotImplementedError(
        f"[qai.conversion] import_model for format '{fmt}' is not implemented. "
        f"Use the format's native runtime (e.g. tensorflow.lite.Interpreter "
        f"for tflite, openvino.runtime.Core for openvino, tensorrt.Runtime for "
        f"tensorrt engines)."
    )


def _detect_format_from_path(path: str) -> str:
    """Guess the format from the file extension."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    direct_map = {
        "pkl": "pickle",
        "pickle": "pickle",
        "qmodel": "qai_native",
        "pt": "torchscript",
        "pth": "torchscript",
        "onnx": "onnx",
        "safetensors": "safetensors",
        "tflite": "tflite",
        "mlmodel": "coreml",
        "mlpackage": "coreml",
        "gguf": "gguf",
        "engine": "tensorrt",
        "trt": "tensorrt",
    }
    if ext in direct_map:
        return direct_map[ext]
    # OpenVINO IR uses .xml; only treat as openvino if there's a .bin next to it.
    if ext == "xml":
        bin_path = os.path.splitext(path)[0] + ".bin"
        if os.path.exists(bin_path):
            return "openvino"
    # Last resort: try pickle.
    return "pickle"


# ============================================================================
# available_backends() -- introspection
# ============================================================================

def _try_import(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def available_backends() -> dict:
    """Return a dict of {format: bool} indicating which backends are installed.

    A True value means the conversion library for that format is available
    and the corresponding exporter should work. False means the user needs to
    `pip install` something.

    `pickle` and `qai_native` are always True (they have no external deps).

    Example:
        >>> from qai.conversion import available_backends
        >>> available_backends()
        {'onnx': True, 'torchscript': True, 'tflite': False, 'coreml': False,
         'tensorrt': False, 'openvino': False, 'gguf': False,
         'safetensors': False, 'pickle': True, 'qai_native': True}
    """
    # ONNX export works if:
    #   - skl2onnx is installed (sklearn path -- skl2onnx depends on onnx), OR
    #   - onnx is installed and torch is installed (modern torch.onnx.export).
    # In legacy torch (<2.9) torch.onnx.export worked without `onnx`, but we
    # don't bother special-casing that anymore -- if the user has torch, we
    # also expect `onnx` to be installed.
    onnx_ok = _try_import("skl2onnx") or (_try_import("onnx") and _try_import("torch"))

    return {
        "onnx":        onnx_ok,
        "torchscript": _try_import("torch"),
        "tflite":      _try_import("tensorflow") or _try_import("ai_edge_torch"),
        "coreml":      _try_import("coremltools"),
        "tensorrt":    _try_import("tensorrt"),
        "openvino":    _try_import("openvino"),
        "gguf":        _try_import("gguf"),
        "safetensors": _try_import("safetensors"),
        "pickle":      True,
        "qai_native":  True,
    }


# ============================================================================
# ConversionReport
# ============================================================================

class ConversionReport:
    """Structured report describing the result of a conversion.

    Attributes:
        format:              One of SUPPORTED_FORMATS.
        path:                Where the converted model was written.
        size_bytes:          File size on disk.
        input_shapes:        List of input tensor shapes (best effort).
        output_shapes:       List of output tensor shapes (best effort).
        supported_ops:       Operations the target format supports
                             (relevant for ONNX/TFLite op sets).
        unsupported_ops:     Operations that were dropped or approximated.
        requires_runtime:    True if loading the file needs the original
                             framework runtime (e.g. pickle/qai_native).
        conversion_warnings: List of warning strings produced during export.
    """
    def __init__(self,
                 format: str,
                 path: Optional[str] = None,
                 size_bytes: int = 0,
                 input_shapes: Optional[List[Tuple[int, ...]]] = None,
                 output_shapes: Optional[List[Tuple[int, ...]]] = None,
                 supported_ops: Optional[List[str]] = None,
                 unsupported_ops: Optional[List[str]] = None,
                 requires_runtime: bool = False,
                 conversion_warnings: Optional[List[str]] = None):
        self.format = format
        self.path = path
        self.size_bytes = size_bytes or 0
        self.input_shapes = input_shapes or []
        self.output_shapes = output_shapes or []
        self.supported_ops = supported_ops or []
        self.unsupported_ops = unsupported_ops or []
        self.requires_runtime = requires_runtime
        self.conversion_warnings = conversion_warnings or []

    def to_dict(self) -> dict:
        return {
            "format": self.format,
            "path": self.path,
            "size_bytes": self.size_bytes,
            "size_human": _human_size(self.size_bytes),
            "input_shapes": [tuple(s) for s in self.input_shapes],
            "output_shapes": [tuple(s) for s in self.output_shapes],
            "supported_ops": list(self.supported_ops),
            "unsupported_ops": list(self.unsupported_ops),
            "requires_runtime": self.requires_runtime,
            "conversion_warnings": list(self.conversion_warnings),
        }

    def __repr__(self) -> str:
        return (
            f"<ConversionReport format={self.format!r} "
            f"size={_human_size(self.size_bytes)} "
            f"warnings={len(self.conversion_warnings)}>"
        )


# ============================================================================
# compare() -- run prediction on both original and converted models
# ============================================================================

def _infer_sample_input(model: Any) -> Optional[np.ndarray]:
    """Try to invent a reasonable sample input for `model`."""
    # qai.Model keeps a training-data signature.
    sig = getattr(model, "_train_data_signature", None) if _is_qai_model(model) else None
    if sig is None and _is_qai_model(model):
        sig = getattr(model.technique, "_train_data_signature", None)
    if sig and sig.get("n_features"):
        n = sig["n_features"]
        if isinstance(n, (tuple, list)):
            shape = (1,) + tuple(n)
        else:
            shape = (1, int(n))
        return np.random.randn(*shape).astype(np.float32)

    # Sklearn estimators know n_features_in_.
    sk = _get_sklearn_estimator(model)
    if sk is not None:
        n = getattr(sk, "n_features_in_", None)
        if n is not None:
            return np.random.randn(1, int(n)).astype(np.float32)

    # Try the technique's last training X.
    if _is_qai_model(model):
        for attr_name in ("_last_X", "_last_x", "X_train"):
            val = getattr(model.technique, attr_name, None)
            if val is not None:
                try:
                    arr = np.asarray(val)
                    if arr.ndim >= 2 and arr.shape[0] > 0:
                        return arr[:1].astype(np.float32)
                except Exception:
                    pass
    return None


def _flatten_to_vector(x: Any) -> np.ndarray:
    """Flatten any prediction output to a 1-D float vector for comparison."""
    if isinstance(x, (int, float)):
        return np.array([float(x)], dtype=np.float64)
    arr = np.asarray(x)
    if arr.dtype.kind in ("U", "S", "O"):
        # Non-numeric (e.g. string predictions). Hash to ints for cosine sim.
        try:
            return np.array([hash(s) for s in arr.ravel()], dtype=np.float64)
        except Exception:
            return np.array([hash(str(x))], dtype=np.float64)
    return arr.astype(np.float64).ravel()


def _run_original_model(model: Any, sample_input: Any) -> np.ndarray:
    """Run a forward pass on the original model and return a 1-D vector."""
    if hasattr(model, "predict"):
        out = model.predict(sample_input)
    elif hasattr(model, "forward"):
        out = model.forward(sample_input)
    else:
        raise RuntimeError(
            f"Model {type(model).__name__} has neither predict() nor forward()"
        )
    return _flatten_to_vector(out)


def _run_converted_file(path: str, fmt: str, sample_input: Any,
                        loaded: Any = None) -> np.ndarray:
    """Run a forward pass on a converted model file/object.

    `loaded` may be the result of import_model(); if None we'll load on demand.
    """
    if fmt in ("pickle", "qai_native"):
        if loaded is None:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
        if hasattr(loaded, "predict"):
            return _flatten_to_vector(loaded.predict(sample_input))
        if hasattr(loaded, "forward"):
            return _flatten_to_vector(loaded.forward(sample_input))
        raise RuntimeError("Loaded pickle/qmodel has no predict() or forward()")

    if fmt == "onnx":
        try:
            import onnxruntime as rt
        except ImportError:
            raise RuntimeError(
                "onnxruntime not installed; cannot run ONNX inference for compare()"
            )
        sess = rt.InferenceSession(path)
        input_name = sess.get_inputs()[0].name
        arr = np.asarray(sample_input, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        outs = sess.run(None, {input_name: arr})
        return _flatten_to_vector(outs[0])

    if fmt == "torchscript":
        try:
            import torch
        except ImportError:
            raise RuntimeError("torch not installed; cannot run TorchScript inference")
        if loaded is None:
            loaded = torch.jit.load(path, map_location="cpu")
        loaded.eval()
        arr = np.asarray(sample_input, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        t = torch.as_tensor(arr)
        with torch.no_grad():
            out = loaded(t)
        return _flatten_to_vector(out.detach().cpu().numpy())

    if fmt == "safetensors":
        # Safetensors stores weights only -- can't run inference directly.
        raise RuntimeError(
            "safetensors stores weights only; runtime comparison requires "
            "loading weights into a model skeleton first."
        )

    raise NotImplementedError(
        f"Runtime comparison for format '{fmt}' is not implemented."
    )


def compare(model: Any, format: str, sample_input: Any = None) -> dict:
    """Convert `model` to `format`, run both, and return similarity metrics.

    Metrics returned:
      * cosine_similarity: 1.0 = identical direction, 0.0 = orthogonal.
      * max_abs_diff:       Largest absolute element-wise difference.
      * relative_error:     Mean(|orig - conv| / max(|orig|, eps)).

    If the conversion library isn't installed, or the conversion/inference
    fails, the returned dict will have an `error` key and the metrics set
    to None.

    Args:
        model:        Any qai.Model / sklearn estimator / torch module.
        format:       One of SUPPORTED_FORMATS.
        sample_input: Optional input to feed both models. If None, we try
                      to infer one from the model's training signature.

    Returns:
        dict with keys: format, path, cosine_similarity, max_abs_diff,
        relative_error, orig_output, conv_output, (error on failure).
    """
    fmt = _normalize_format(format)

    # 1. Figure out a sample input.
    if sample_input is None:
        sample_input = _infer_sample_input(model)
    if sample_input is None:
        return {
            "format": fmt,
            "error": "no sample_input provided and none could be inferred",
            "cosine_similarity": None,
            "max_abs_diff": None,
            "relative_error": None,
        }

    # 2. Run the original.
    try:
        orig_vec = _run_original_model(model, sample_input)
    except Exception as e:
        return {
            "format": fmt,
            "error": f"original model predict failed: {e}",
            "cosine_similarity": None,
            "max_abs_diff": None,
            "relative_error": None,
        }

    # 3. Export to the target format.
    ext = _FORMAT_EXTENSIONS.get(fmt, ".bin")
    tmp_dir = "/tmp" if os.path.exists("/tmp") else os.getcwd()
    tmp_path = os.path.join(tmp_dir, f"_qai_compare_{fmt}{ext}")
    try:
        export_path = export(model, fmt, tmp_path, sample_input=sample_input)
    except ImportError as e:
        return {
            "format": fmt,
            "error": f"backend not installed: {e}",
            "cosine_similarity": None,
            "max_abs_diff": None,
            "relative_error": None,
        }
    except Exception as e:
        return {
            "format": fmt,
            "error": f"export failed: {e}",
            "cosine_similarity": None,
            "max_abs_diff": None,
            "relative_error": None,
        }

    # 4. Run the converted model.
    try:
        conv_vec = _run_converted_file(export_path, fmt, sample_input)
    except Exception as e:
        return {
            "format": fmt,
            "path": export_path,
            "error": f"converted model inference failed: {e}",
            "cosine_similarity": None,
            "max_abs_diff": None,
            "relative_error": None,
        }

    # 5. Compute similarity metrics.
    # Align shapes if necessary.
    if orig_vec.shape != conv_vec.shape:
        n = min(orig_vec.size, conv_vec.size)
        orig_vec = orig_vec[:n]
        conv_vec = conv_vec[:n]

    if orig_vec.size == 0:
        cos = 1.0
        max_diff = 0.0
        rel = 0.0
    else:
        norm_o = float(np.linalg.norm(orig_vec))
        norm_c = float(np.linalg.norm(conv_vec))
        if norm_o > 0 and norm_c > 0:
            cos = float(np.dot(orig_vec, conv_vec) / (norm_o * norm_c))
        elif norm_o == 0 and norm_c == 0:
            cos = 1.0
        else:
            cos = 0.0
        max_diff = float(np.max(np.abs(orig_vec - conv_vec)))
        denom = np.maximum(np.abs(orig_vec), 1e-8)
        rel = float(np.mean(np.abs(orig_vec - conv_vec) / denom))

    return {
        "format": fmt,
        "path": export_path,
        "cosine_similarity": cos,
        "max_abs_diff": max_diff,
        "relative_error": rel,
        "orig_output": orig_vec.tolist() if orig_vec.size <= 10 else f"array{orig_vec.shape}",
        "conv_output": conv_vec.tolist() if conv_vec.size <= 10 else f"array{conv_vec.shape}",
    }


# ============================================================================
# benchmark_formats() -- convert to each format and time load+inference
# ============================================================================

def benchmark_formats(model: Any,
                      formats: list = None,
                      sample_input: Any = None) -> list:
    """Convert `model` to each format in `formats` and return a leaderboard.

    For each format, measures:
      * exported:        True if the export succeeded.
      * size_bytes:      Size of the output file.
      * load_time_s:      Time to import_model() the file back.
      * inference_time_s: Time for one forward pass on the loaded model.
      * total_time_s:     load_time_s + inference_time_s (used for sorting).
      * error:           None on success, else an error string.

    Returns a list of dicts sorted by total_time_s (fastest first). Failed
    entries appear last.

    Args:
        model:        Any model that export() accepts.
        formats:      List of formats to benchmark. Defaults to
                      ['onnx', 'torchscript', 'pickle', 'qai_native'].
        sample_input: Input to time inference with. If None, we try to infer.
    """
    if formats is None:
        formats = ["onnx", "torchscript", "pickle", "qai_native"]
    if sample_input is None:
        sample_input = _infer_sample_input(model)

    tmp_dir = "/tmp" if os.path.exists("/tmp") else os.getcwd()
    results: List[Dict[str, Any]] = []

    for raw_fmt in formats:
        try:
            fmt = _normalize_format(raw_fmt)
        except ValueError as e:
            results.append({
                "format": str(raw_fmt), "exported": False, "size_bytes": 0,
                "load_time_s": None, "inference_time_s": None,
                "total_time_s": None, "error": str(e),
            })
            continue

        entry: Dict[str, Any] = {
            "format": fmt,
            "exported": False,
            "size_bytes": 0,
            "load_time_s": None,
            "inference_time_s": None,
            "total_time_s": None,
            "error": None,
        }

        ext = _FORMAT_EXTENSIONS.get(fmt, ".bin")
        tmp_path = os.path.join(tmp_dir, f"_qai_bench_{fmt}{ext}")

        # --- 1. Export ---
        try:
            export_path = export(model, fmt, tmp_path, sample_input=sample_input)
            entry["exported"] = True
            try:
                entry["size_bytes"] = os.path.getsize(export_path)
            except OSError:
                entry["size_bytes"] = 0
        except ImportError as e:
            entry["error"] = f"backend not installed: {e}"
            results.append(entry)
            continue
        except Exception as e:
            entry["error"] = f"export failed: {e}"
            results.append(entry)
            continue

        # --- 2. Load + time ---
        loaded = None
        try:
            t0 = time.perf_counter()
            # For pickle/qai_native/torchscript/safetensors we can call import_model.
            # For onnx, import_model returns an onnx.ModelProto which isn't directly
            # runnable, so we'll skip the load timing there and let _run_converted_file
            # build the InferenceSession itself.
            if fmt in ("pickle", "qai_native", "torchscript", "safetensors", "coreml", "gguf"):
                loaded = import_model(export_path, format=fmt)
                t1 = time.perf_counter()
                entry["load_time_s"] = t1 - t0
            else:
                # onnx / tflite / tensorrt / openvino -- loading == opening the file
                # via a runtime session, which we fold into inference_time.
                t1 = time.perf_counter()
                entry["load_time_s"] = t1 - t0
        except Exception as e:
            entry["error"] = f"load failed: {e}"
            results.append(entry)
            continue

        # --- 3. Inference time ---
        if sample_input is None:
            entry["error"] = "no sample_input; skipped inference timing"
            results.append(entry)
            continue

        try:
            # Warm-up (not measured)
            try:
                _run_converted_file(export_path, fmt, sample_input, loaded=loaded)
            except NotImplementedError:
                # Some formats can't run inference here (safetensors, gguf, etc.)
                entry["error"] = "no runtime for this format"
                results.append(entry)
                continue

            t2 = time.perf_counter()
            _run_converted_file(export_path, fmt, sample_input, loaded=loaded)
            t3 = time.perf_counter()
            entry["inference_time_s"] = t3 - t2
            entry["total_time_s"] = (entry["load_time_s"] or 0.0) + (entry["inference_time_s"] or 0.0)
        except Exception as e:
            entry["error"] = f"inference failed: {e}"

        results.append(entry)

    # Sort: successful entries first by total_time_s, then failures.
    def _sort_key(e):
        if e.get("error") and not e.get("inference_time_s"):
            return (1, 1e18, e["format"])
        return (0, e.get("total_time_s") or 1e18, e["format"])

    results.sort(key=_sort_key)
    return results


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    # Top-level API
    "export",
    "import_model",
    "supported_formats",
    "available_backends",
    # Per-format exporters
    "ONNXExporter",
    "TorchScriptExporter",
    "TFLiteExporter",
    "CoreMLExporter",
    "TensorRTExporter",
    "OpenVINOExporter",
    "GGUFExporter",
    "SafeTensorsExporter",
    "PickleExporter",
    "QAINativeExporter",
    # Reports and comparison
    "ConversionReport",
    "compare",
    "benchmark_formats",
    # Constants
    "SUPPORTED_FORMATS",
]
