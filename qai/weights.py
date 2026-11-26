"""
qai.weights -- weights file system: inspect, edit, merge, visualize, compare.

Weights are the "source code" of a model. This module treats them as
first-class files that can be:
  - Inspected (values, shapes, stats, distributions)
  - Edited (with warnings about manual editing being dangerous)
  - Merged (combine weights from two models)
  - Visualized (heatmaps, histograms, dimension views)
  - Compared (diff two weight sets)
  - Versioned (track changes over time)
  - Validated (check for NaN, explosion, death)

Each model's weights are saved as a .weights file (npz archive) that
can be opened independently of the model.

Usage:
    # Save weights separately
    qai.weights.save_weights(model, "model.weights")

    # Inspect
    info = qai.weights.inspect("model.weights")
    print(info["layers"]["W"]["shape"], info["layers"]["W"]["mean"])

    # Edit (with warning)
    qai.weights.edit("model.weights", layer="W", operation="scale", factor=0.9)

    # Merge two models' weights
    qai.weights.merge("model_a.weights", "model_b.weights", "merged.weights", alpha=0.5)

    # Visualize as heatmap
    qai.weights.heatmap("model.weights", layer="W", save_path="heatmap.png")

    # Compare
    diff = qai.weights.compare("old.weights", "new.weights")
"""
from __future__ import annotations
import hashlib
import json
import os
import time
import warnings
import zipfile
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. WeightsFile — represents a .weights file
# ============================================================================
WEIGHTS_EXTENSION = ".weights"


class WeightsFile:
    """A .weights file is an npz archive containing:
      - Each layer's weight array (keyed by layer name)
      - metadata.json with stats, fingerprint, timestamps
      - changelog.json tracking all edits
    """

    def __init__(self, path: str):
        self.path = str(path)
        self.name = os.path.basename(path)
        self._metadata: Optional[Dict[str, Any]] = None
        self._changelog: Optional[List[Dict[str, Any]]] = None

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

    def load(self) -> Dict[str, np.ndarray]:
        """Load all weight arrays from the file."""
        data = np.load(self.path, allow_pickle=True)
        weights = {}
        for key in data.files:
            if key not in ("metadata.json", "changelog.json"):
                weights[key] = data[key]
        return weights

    def metadata(self) -> Dict[str, Any]:
        """Load metadata."""
        if self._metadata is not None:
            return self._metadata
        try:
            data = np.load(self.path, allow_pickle=True)
            if "metadata.json" in data.files:
                self._metadata = json.loads(str(data["metadata.json"]))
            else:
                self._metadata = {}
        except Exception:
            self._metadata = {}
        return self._metadata

    def changelog(self) -> List[Dict[str, Any]]:
        """Load the edit changelog."""
        if self._changelog is not None:
            return self._changelog
        try:
            data = np.load(self.path, allow_pickle=True)
            if "changelog.json" in data.files:
                self._changelog = json.loads(str(data["changelog.json"]))
            else:
                self._changelog = []
        except Exception:
            self._changelog = []
        return self._changelog

    @property
    def n_layers(self) -> int:
        return len(self.load())

    @property
    def n_params(self) -> int:
        return sum(arr.size for arr in self.load().values())

    @property
    def fingerprint(self) -> str:
        """SHA-256 fingerprint of all weights."""
        weights = self.load()
        flat = np.concatenate([arr.flatten() for arr in weights.values()])
        return hashlib.sha256(flat.tobytes()).hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        """Summary of the weights file."""
        weights = self.load()
        layers = {}
        for name, arr in weights.items():
            layers[name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "n_params": int(arr.size),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "has_nan": bool(np.isnan(arr).any()),
                "has_inf": bool(np.isinf(arr).any()),
                "sparsity": float((arr == 0).mean()),
            }
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size_human,
            "n_layers": len(weights),
            "n_params": self.n_params,
            "fingerprint": self.fingerprint,
            "layers": layers,
            "n_edits": len(self.changelog()),
            "created_at": self.metadata().get("created_at", "unknown"),
        }


# ============================================================================
# 2. save_weights / load_weights
# ============================================================================
def save_weights(model: Any, path: str,
                  include_metadata: bool = True,
                  provenance: Optional[Dict] = None) -> str:
    """Save a model's weights as a .weights file.

    The file is an npz archive with:
      - Each layer's weight array
      - metadata.json (stats, fingerprint, timestamps)
      - changelog.json (empty initially)
    """
    # ensure .weights or .wt extension (np.savez adds .npz, so we use a temp then rename)
    if not (path.endswith(WEIGHTS_EXTENSION) or path.endswith(".wt")):
        path += WEIGHTS_EXTENSION

    weights = _extract_weights(model)
    if not weights:
        raise ValueError("model has no extractable weights")

    # build metadata
    metadata = {
        "created_at": time.time(),
        "created_at_iso": datetime.now().isoformat(),
        "n_layers": len(weights),
        "n_params": sum(arr.size for arr in weights.values()),
        "provenance": provenance or {},
    }
    # compute per-layer stats
    layer_stats = {}
    for name, arr in weights.items():
        layer_stats[name] = {
            "shape": list(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    metadata["layer_stats"] = layer_stats

    # save as npz (np.savez adds .npz, so save to temp then rename)
    save_dict = {**weights,
                  "metadata.json": np.array(json.dumps(metadata, indent=2)),
                  "changelog.json": np.array(json.dumps([], indent=2))}
    tmp_path = path + ".tmp"
    np.savez(tmp_path, **save_dict)
    # np.savez creates tmp_path + ".npz", rename to our .weights
    os.rename(tmp_path + ".npz", path)
    return path


def load_weights(path: str) -> Dict[str, np.ndarray]:
    """Load weights from a .weights or .wt file."""
    wf = WeightsFile(path)
    return wf.load()


def _extract_weights(model: Any) -> Dict[str, np.ndarray]:
    """Extract weight arrays from any model type."""
    # unwrap Bundle → Model
    if hasattr(model, "model") and not hasattr(model, "state_dict") and not hasattr(model, "technique"):
        model = model.model  # unwrap Bundle
    # try state_dict
    if hasattr(model, "state_dict"):
        try:
            return {k: np.asarray(v) for k, v in model.state_dict().items()}
        except Exception:
            pass
    # try technique.state_dict
    if hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
        try:
            return {k: np.asarray(v) for k, v in model.technique.state_dict().items()}
        except Exception:
            pass
    # fallback: scan for numpy attributes
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


def _apply_weights(model: Any, weights: Dict[str, np.ndarray]) -> Any:
    """Apply weight arrays back to a model."""
    if hasattr(model, "load_state_dict"):
        try:
            model.load_state_dict(weights)
            return model
        except Exception:
            pass
    if hasattr(model, "technique") and hasattr(model.technique, "load_state_dict"):
        try:
            model.technique.load_state_dict(weights)
            return model
        except Exception:
            pass
    # fallback: set attributes directly
    for name, arr in weights.items():
        try:
            if hasattr(model, name):
                setattr(model, name, arr)
            elif hasattr(model, "technique") and hasattr(model.technique, name):
                setattr(model.technique, name, arr)
        except Exception:
            pass
    return model


# ============================================================================
# 3. inspect_weights — detailed inspection
# ============================================================================
def inspect_weights(path: str) -> Dict[str, Any]:
    """Inspect a .weights file. Returns detailed stats per layer."""
    wf = WeightsFile(path)
    return wf.summary()


def inspect_layer(path: str, layer_name: str) -> Dict[str, Any]:
    """Deep inspection of a single layer's weights."""
    weights = load_weights(path)
    if layer_name not in weights:
        available = list(weights.keys())
        raise KeyError(f"layer '{layer_name}' not found. Available: {available}")
    arr = weights[layer_name]
    return {
        "name": layer_name,
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "n_params": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "n_zeros": int((arr == 0).sum()),
        "sparsity": float((arr == 0).mean()),
        "n_negative": int((arr < 0).sum()),
        "n_positive": int((arr > 0).sum()),
        "frobenius_norm": float(np.linalg.norm(arr)),
        "sample_values": arr.flatten()[:10].tolist(),
    }


# ============================================================================
# 4. edit_weights — edit with WARNING
# ============================================================================
class WeightsEditWarning(UserWarning):
    """Warning raised when manually editing weights."""
    pass


def edit_weights(path: str, layer: str,
                  operation: str = "set",
                  value: Optional[float] = None,
                  factor: Optional[float] = None,
                  indices: Optional[Tuple] = None,
                  new_values: Optional[np.ndarray] = None,
                  confirm: bool = False) -> Dict[str, Any]:
    """Edit weights in a .weights file.

    ⚠️  WARNING: Manually editing weights is DANGEROUS.
    - Can destroy learned features
    - Can cause NaN/explosion in forward pass
    - Can make the model produce garbage
    Always backup before editing.

    Operations:
      - "set": set entire layer to `value`
      - "scale": multiply all weights by `factor`
      - "add": add `value` to all weights
      - "zero": zero out specific `indices`
      - "replace": replace `indices` with `new_values`
      - "noise": add gaussian noise (std=`factor`)
      - "prune": zero out smallest-magnitude weights (fraction=`factor`)
      - "quantize": round to `factor` decimal places

    Args:
        confirm: if True, suppress the warning (user acknowledged risk)
    """
    if not confirm:
        warnings.warn(
            "⚠️  Manually editing model weights is DANGEROUS. "
            "This can destroy learned features, cause NaN, or produce garbage output. "
            "Always backup your .weights file before editing. "
            "Pass confirm=True to suppress this warning.",
            WeightsEditWarning,
            stacklevel=2,
        )

    weights = load_weights(path)
    if layer not in weights:
        raise KeyError(f"layer '{layer}' not found. Available: {list(weights.keys())}")

    arr = weights[layer].copy()
    original_stats = {"mean": float(arr.mean()), "std": float(arr.std()),
                       "min": float(arr.min()), "max": float(arr.max())}

    if operation == "set":
        if value is None:
            raise ValueError("operation 'set' requires `value`")
        arr[:] = value
    elif operation == "scale":
        if factor is None:
            raise ValueError("operation 'scale' requires `factor`")
        arr *= factor
    elif operation == "add":
        if value is None:
            raise ValueError("operation 'add' requires `value`")
        arr += value
    elif operation == "zero":
        if indices is None:
            raise ValueError("operation 'zero' requires `indices`")
        arr[indices] = 0
    elif operation == "replace":
        if indices is None or new_values is None:
            raise ValueError("operation 'replace' requires `indices` and `new_values`")
        arr[indices] = new_values
    elif operation == "noise":
        if factor is None:
            raise ValueError("operation 'noise' requires `factor` (std)")
        arr += np.random.normal(0, factor, arr.shape).astype(arr.dtype)
    elif operation == "prune":
        if factor is None:
            raise ValueError("operation 'prune' requires `factor` (fraction to prune)")
        threshold = np.percentile(np.abs(arr), factor * 100)
        arr[np.abs(arr) < threshold] = 0
    elif operation == "quantize":
        if factor is None:
            raise ValueError("operation 'quantize' requires `factor` (decimal places)")
        arr = np.round(arr, decimals=int(factor))
    else:
        raise ValueError(f"unknown operation '{operation}'")

    # validate: check for NaN/Inf
    if np.isnan(arr).any():
        raise ValueError("edit produced NaN values — aborting (weights not saved)")
    if np.isinf(arr).any():
        raise ValueError("edit produced Inf values — aborting (weights not saved)")

    weights[layer] = arr
    new_stats = {"mean": float(arr.mean()), "std": float(arr.std()),
                  "min": float(arr.min()), "max": float(arr.max())}

    # record in changelog
    _append_changelog(path, {
        "operation": operation,
        "layer": layer,
        "timestamp": time.time(),
        "original_stats": original_stats,
        "new_stats": new_stats,
        "params": {"value": value, "factor": factor, "indices": str(indices)},
    })

    # save
    _save_weights_dict(path, weights)
    return {
        "layer": layer,
        "operation": operation,
        "original_stats": original_stats,
        "new_stats": new_stats,
        "warning": "weights edited manually — verify model still works correctly",
    }


def _save_weights_dict(path: str, weights: Dict[str, np.ndarray]) -> None:
    """Save a dict of weights back to a .weights file, preserving metadata + changelog."""
    wf = WeightsFile(path)
    metadata = wf.metadata()
    changelog = wf.changelog()
    # update metadata stats
    layer_stats = {}
    for name, arr in weights.items():
        layer_stats[name] = {
            "shape": list(arr.shape),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    metadata["layer_stats"] = layer_stats
    metadata["last_edited_at"] = time.time()
    save_dict = {**weights,
                  "metadata.json": np.array(json.dumps(metadata, indent=2)),
                  "changelog.json": np.array(json.dumps(changelog, indent=2))}
    tmp_path = path + ".tmp"
    np.savez(tmp_path, **save_dict)
    os.rename(tmp_path + ".npz", path)


def _append_changelog(path: str, entry: Dict[str, Any]) -> None:
    """Append an entry to the changelog in a .weights file."""
    wf = WeightsFile(path)
    changelog = wf.changelog()
    changelog.append(entry)
    # rewrite with updated changelog
    weights = wf.load()
    metadata = wf.metadata()
    save_dict = {**weights,
                  "metadata.json": np.array(json.dumps(metadata, indent=2)),
                  "changelog.json": np.array(json.dumps(changelog, indent=2))}
    tmp_path = path + ".tmp"
    np.savez(tmp_path, **save_dict)
    os.rename(tmp_path + ".npz", path)


# ============================================================================
# 5. merge_weights — merge weights from two models
# ============================================================================
def merge_weights(path_a: str, path_b: str, output_path: str,
                   strategy: str = "lerp",
                   alpha: float = 0.5,
                   layers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Merge weights from two .weights files.

    Strategies:
      - "lerp": linear interpolation: alpha * A + (1-alpha) * B
      - "ties": TIES merging (trim + sign-resolve + merge)
      - "max_norm": keep the weight with larger magnitude per element
      - "average": simple average
      - "union": take A for some layers, B for others (specify `layers`)

    Args:
        alpha: weight for model A (0 = all B, 1 = all A)
        layers: for "union" strategy, which layers to take from A
    """
    weights_a = load_weights(path_a)
    weights_b = load_weights(path_b)
    merged: Dict[str, np.ndarray] = {}
    conflicts = 0

    all_keys = set(weights_a.keys()) | set(weights_b.keys())
    for key in all_keys:
        if key in ("metadata.json", "changelog.json"):
            continue
        if key not in weights_a:
            merged[key] = weights_b[key].copy()
            continue
        if key not in weights_b:
            merged[key] = weights_a[key].copy()
            continue
        a = weights_a[key]
        b = weights_b[key]
        if a.shape != b.shape:
            merged[key] = a.copy()  # can't merge different shapes
            conflicts += 1
            continue
        if strategy == "lerp":
            merged[key] = (alpha * a + (1 - alpha) * b).astype(a.dtype)
        elif strategy == "average":
            merged[key] = ((a + b) / 2).astype(a.dtype)
        elif strategy == "max_norm":
            mask = (np.abs(a) >= np.abs(b)).astype(a.dtype)
            merged[key] = (mask * a + (1 - mask) * b).astype(a.dtype)
        elif strategy == "ties":
            # TIES: trim 20%, resolve sign conflicts, average
            threshold_a = np.percentile(np.abs(a), 80)
            threshold_b = np.percentile(np.abs(b), 80)
            a_trimmed = np.where(np.abs(a) < threshold_a, 0, a)
            b_trimmed = np.where(np.abs(b) < threshold_b, 0, b)
            total = a_trimmed + b_trimmed
            sign = np.sign(total)
            a_aligned = np.where(np.sign(a_trimmed) == sign, a_trimmed, 0)
            b_aligned = np.where(np.sign(b_trimmed) == sign, b_trimmed, 0)
            count = ((a_trimmed != 0).astype(int) + (b_trimmed != 0).astype(int))
            count = np.maximum(count, 1)
            merged[key] = ((a_aligned + b_aligned) / count).astype(a.dtype)
        elif strategy == "union":
            if layers and key in layers:
                merged[key] = a.copy()
            else:
                merged[key] = b.copy()
        else:
            merged[key] = (alpha * a + (1 - alpha) * b).astype(a.dtype)

    # save merged
    if not output_path.endswith(WEIGHTS_EXTENSION):
        output_path += WEIGHTS_EXTENSION
    metadata = {
        "created_at": time.time(),
        "created_at_iso": datetime.now().isoformat(),
        "source": "merge",
        "merge_strategy": strategy,
        "alpha": alpha,
        "parent_a": path_a,
        "parent_b": path_b,
        "n_conflicts": conflicts,
    }
    save_dict = {**merged,
                  "metadata.json": np.array(json.dumps(metadata, indent=2)),
                  "changelog.json": np.array(json.dumps([{
                      "operation": "merge",
                      "strategy": strategy,
                      "alpha": alpha,
                      "parents": [path_a, path_b],
                      "timestamp": time.time(),
                  }], indent=2))}
    tmp_path = output_path + ".tmp"
    np.savez(tmp_path, **save_dict)
    os.rename(tmp_path + ".npz", output_path)
    return {
        "output_path": output_path,
        "strategy": strategy,
        "n_layers_merged": len(merged),
        "n_conflicts": conflicts,
    }


# ============================================================================
# 6. visualize_weights — heatmaps, histograms, dimension views
# ============================================================================
def visualize_weights(path: str, layer: Optional[str] = None,
                       plot_type: str = "heatmap",
                       save_path: Optional[str] = None) -> Dict[str, Any]:
    """Visualize weights as heatmaps, histograms, or dimension views.

    plot_type: "heatmap" | "histogram" | "stats" | "dimension_view"

    For "dimension_view": reshapes weights to 2D and shows as image.
    """
    weights = load_weights(path)
    if layer is None:
        # pick the largest layer
        layer = max(weights.keys(), key=lambda k: weights[k].size)
    if layer not in weights:
        raise KeyError(f"layer '{layer}' not found")
    arr = weights[layer]

    result: Dict[str, Any] = {"layer": layer, "plot_type": plot_type}

    if plot_type == "heatmap":
        # generate a text-based heatmap (or save PNG if matplotlib available)
        result["shape"] = list(arr.shape)
        result["stats"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        # try to save as PNG
        if save_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                # reshape to 2D for visualization
                if arr.ndim > 2:
                    flat = arr.reshape(arr.shape[0], -1)
                elif arr.ndim == 1:
                    flat = arr.reshape(1, -1)
                else:
                    flat = arr
                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(flat, cmap="RdBu_r", aspect="auto",
                                vmin=-np.std(arr), vmax=np.std(arr))
                ax.set_title(f"Weight heatmap: {layer}")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"
        else:
            # text-based heatmap (first 10x10 block)
            flat = arr.flatten()[:100].reshape(10, 10) if arr.size >= 100 else arr.reshape(1, -1)
            result["text_heatmap"] = _text_heatmap(flat)

    elif plot_type == "histogram":
        # generate histogram data
        hist, bin_edges = np.histogram(arr.flatten(), bins=50)
        result["histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
        if save_path:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(arr.flatten(), bins=50, color="steelblue", edgecolor="white")
                ax.set_title(f"Weight distribution: {layer}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Count")
                fig.tight_layout()
                fig.savefig(save_path, dpi=100)
                plt.close(fig)
                result["saved_to"] = save_path
            except ImportError:
                result["note"] = "install matplotlib for PNG output"

    elif plot_type == "dimension_view":
        # reshape weights to different dimensions and show stats
        views = []
        total = arr.size
        for dim in [1, 2, 4, 8, 16]:
            if total % dim == 0:
                reshaped = arr.reshape(dim, -1)
                views.append({
                    "shape": list(reshaped.shape),
                    "row_means": reshaped.mean(axis=1).tolist(),
                    "row_stds": reshaped.std(axis=1).tolist(),
                })
        result["dimension_views"] = views

    elif plot_type == "stats":
        result.update(inspect_layer(path, layer))

    return result


def _text_heatmap(arr: np.ndarray, width: int = 40) -> str:
    """Generate a text-based heatmap."""
    chars = " .:-=+*#%@"
    normalized = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-8)
    lines = []
    for row in normalized:
        line = "".join(chars[min(int(v * (len(chars) - 1)), len(chars) - 1)] for v in row)
        lines.append(line)
    return "\n".join(lines)


# ============================================================================
# 7. compare_weights — diff two weight files
# ============================================================================
def compare_weights(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare two .weights files."""
    weights_a = load_weights(path_a)
    weights_b = load_weights(path_b)
    differences: List[Dict[str, Any]] = []
    all_keys = set(weights_a.keys()) | set(weights_b.keys())
    for key in all_keys:
        if key in ("metadata.json", "changelog.json"):
            continue
        if key not in weights_a:
            differences.append({"layer": key, "change": "added_in_b"})
            continue
        if key not in weights_b:
            differences.append({"layer": key, "change": "removed_in_b"})
            continue
        a, b = weights_a[key], weights_b[key]
        if a.shape != b.shape:
            differences.append({"layer": key, "change": "shape_mismatch",
                                 "shape_a": list(a.shape), "shape_b": list(b.shape)})
            continue
        diff = b - a
        differences.append({
            "layer": key,
            "change": "modified",
            "n_changed": int((diff != 0).sum()),
            "pct_changed": float((diff != 0).mean()),
            "mean_diff": float(diff.mean()),
            "max_diff": float(np.abs(diff).max()),
            "l2_diff": float(np.linalg.norm(diff)),
        })
    return {
        "file_a": path_a,
        "file_b": path_b,
        "n_layers_compared": len(all_keys - {"metadata.json", "changelog.json"}),
        "n_different": len(differences),
        "differences": differences,
        "identical": len(differences) == 0,
    }


# ============================================================================
# 8. validate_weights — check for problems
# ============================================================================
def validate_weights(path: str) -> Dict[str, Any]:
    """Validate weights for common problems."""
    weights = load_weights(path)
    issues: List[Dict[str, str]] = []
    warnings_list: List[Dict[str, str]] = []
    for name, arr in weights.items():
        if np.isnan(arr).any():
            issues.append({"layer": name, "issue": "NaN values detected",
                             "severity": "critical"})
        if np.isinf(arr).any():
            issues.append({"layer": name, "issue": "Inf values detected",
                             "severity": "critical"})
        if arr.std() == 0:
            issues.append({"layer": name, "issue": "all weights identical (std=0)",
                             "severity": "error"})
        if arr.max() > 1e6:
            warnings_list.append({"layer": name, "issue": "very large weights (>1e6)",
                                    "severity": "warning"})
        if arr.min() < -1e6:
            warnings_list.append({"layer": name, "issue": "very negative weights (<-1e6)",
                                    "severity": "warning"})
        sparsity = (arr == 0).mean()
        if sparsity > 0.9:
            warnings_list.append({"layer": name, "issue": f"high sparsity ({sparsity:.0%} zeros)",
                                    "severity": "warning"})
    return {
        "path": path,
        "n_layers": len(weights),
        "n_issues": len(issues),
        "n_warnings": len(warnings_list),
        "issues": issues,
        "warnings": warnings_list,
        "valid": len(issues) == 0,
    }


# ============================================================================
# 9. WeightsRepository — default storage for weights
# ============================================================================
class WeightsRepository:
    """A default repository for weights files.

    When a model is trained without specifying a save path, weights
    are auto-stored here. Like a "weights cache" that's always available.
    """

    def __init__(self, root: str = "~/.qai/weights"):
        self.root = os.path.expanduser(root)
        os.makedirs(self.root, exist_ok=True)

    def auto_save(self, model: Any, technique: str = "model",
                   provenance: Optional[Dict] = None) -> str:
        """Auto-save weights with auto-generated filename."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        filename = f"{technique}_{timestamp}{WEIGHTS_EXTENSION}"
        path = os.path.join(self.root, filename)
        return save_weights(model, path, provenance=provenance)

    def list(self) -> List[WeightsFile]:
        """List all weights files in the repository."""
        files = []
        for name in os.listdir(self.root):
            if name.endswith(WEIGHTS_EXTENSION):
                files.append(WeightsFile(os.path.join(self.root, name)))
        files.sort(key=lambda f: f.size_bytes, reverse=True)
        return files

    def search(self, query: str = "") -> List[WeightsFile]:
        """Search weights files by name."""
        q = query.lower()
        return [f for f in self.list() if q in f.name.lower()]

    def load_latest(self) -> Optional[Dict[str, np.ndarray]]:
        """Load the most recently saved weights."""
        files = sorted(self.list(), key=lambda f: os.path.getmtime(f.path), reverse=True)
        if not files:
            return None
        return files[0].load()

    def cleanup(self, max_files: int = 100) -> int:
        """Remove oldest files beyond max_files."""
        files = sorted(self.list(), key=lambda f: os.path.getmtime(f.path))
        removed = 0
        while len(files) > max_files:
            oldest = files.pop(0)
            os.unlink(oldest.path)
            removed += 1
        return removed

    def stats(self) -> Dict[str, Any]:
        """Repository stats."""
        files = self.list()
        total_size = sum(f.size_bytes for f in files)
        return {
            "root": self.root,
            "n_files": len(files),
            "total_size_bytes": total_size,
            "total_size_human": f"{total_size / (1024**2):.1f}MB" if total_size > 0 else "0B",
        }


# ============================================================================
# 10. WeightsEditor — interactive editor (programmatic)
# ============================================================================
class WeightsEditor:
    """A programmatic weights editor with safety checks.

    Usage:
        editor = WeightsEditor("model.weights")
        editor.view("layer_W")           # inspect a layer
        editor.scale("layer_W", 0.9)     # scale by 0.9 (with warning)
        editor.prune("layer_W", 0.3)     # prune 30% smallest
        editor.save()                     # save changes
        editor.undo()                     # undo last change
        editor.diff()                     # show what changed
    """

    def __init__(self, path: str, confirm: bool = False):
        self.path = path
        self.confirm = confirm
        self.weights = load_weights(path)
        self._original = {k: v.copy() for k, v in self.weights.items()}
        self._history: List[Dict[str, Any]] = []

    def view(self, layer: str) -> Dict[str, Any]:
        """View a layer's weights."""
        if layer not in self.weights:
            raise KeyError(f"layer '{layer}' not found. Available: {list(self.weights.keys())}")
        return inspect_layer(self.path, layer)

    def set(self, layer: str, value: float) -> "WeightsEditor":
        """Set all weights in a layer to a value."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "set", "layer": layer, "old": self.weights[layer].copy()})
        self.weights[layer][:] = value
        return self

    def scale(self, layer: str, factor: float) -> "WeightsEditor":
        """Scale all weights in a layer."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "scale", "layer": layer, "old": self.weights[layer].copy()})
        self.weights[layer] *= factor
        return self

    def add(self, layer: str, value: float) -> "WeightsEditor":
        """Add a value to all weights in a layer."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "add", "layer": layer, "old": self.weights[layer].copy()})
        self.weights[layer] += value
        return self

    def prune(self, layer: str, fraction: float) -> "WeightsEditor":
        """Prune the smallest-magnitude weights."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "prune", "layer": layer, "old": self.weights[layer].copy()})
        arr = self.weights[layer]
        threshold = np.percentile(np.abs(arr), fraction * 100)
        arr[np.abs(arr) < threshold] = 0
        return self

    def noise(self, layer: str, std: float) -> "WeightsEditor":
        """Add Gaussian noise to a layer."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "noise", "layer": layer, "old": self.weights[layer].copy()})
        self.weights[layer] += np.random.normal(0, std, self.weights[layer].shape)
        return self

    def zero(self, layer: str, indices: Tuple) -> "WeightsEditor":
        """Zero out specific indices."""
        if not self.confirm:
            warnings.warn("⚠️ Manual weight editing is dangerous. Pass confirm=True.",
                            WeightsEditWarning, stacklevel=2)
        self._history.append({"op": "zero", "layer": layer, "old": self.weights[layer].copy()})
        self.weights[layer][indices] = 0
        return self

    def save(self) -> str:
        """Save changes to the file."""
        # validate before saving
        for name, arr in self.weights.items():
            if np.isnan(arr).any():
                raise ValueError(f"layer '{name}' has NaN — cannot save")
            if np.isinf(arr).any():
                raise ValueError(f"layer '{name}' has Inf — cannot save")
        _save_weights_dict(self.path, self.weights)
        return self.path

    def save_as(self, path: str) -> str:
        """Save to a new file."""
        for name, arr in self.weights.items():
            if np.isnan(arr).any():
                raise ValueError(f"layer '{name}' has NaN — cannot save")
            if np.isinf(arr).any():
                raise ValueError(f"layer '{name}' has Inf — cannot save")
        if not path.endswith(WEIGHTS_EXTENSION):
            path += WEIGHTS_EXTENSION
        _save_weights_dict(path, self.weights)
        return path

    def undo(self) -> bool:
        """Undo the last edit."""
        if not self._history:
            return False
        entry = self._history.pop()
        self.weights[entry["layer"]] = entry["old"]
        return True

    def diff(self) -> List[Dict[str, Any]]:
        """Show what changed from the original."""
        changes = []
        for name, arr in self.weights.items():
            orig = self._original.get(name)
            if orig is None:
                changes.append({"layer": name, "change": "added"})
                continue
            if not np.array_equal(arr, orig):
                d = arr - orig
                changes.append({
                    "layer": name,
                    "n_changed": int((d != 0).sum()),
                    "pct_changed": float((d != 0).mean()),
                    "max_diff": float(np.abs(d).max()),
                })
        return changes

    def reset(self) -> None:
        """Reset to original weights."""
        self.weights = {k: v.copy() for k, v in self._original.items()}
        self._history.clear()


__all__ = [
    "WEIGHTS_EXTENSION", "WeightsFile",
    "save_weights", "load_weights", "inspect_weights", "inspect_layer",
    "edit_weights", "merge_weights", "visualize_weights", "compare_weights",
    "validate_weights", "WeightsRepository", "WeightsEditor",
    "WeightsEditWarning",
]
