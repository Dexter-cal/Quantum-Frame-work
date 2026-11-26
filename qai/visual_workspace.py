"""
qai.visual_workspace -- the interactive "Model Microscope".

A self-contained, single-page HTML/CSS/JS workspace that lets you:

  * Watch the forward pass animate as input flows through the model
  * Drag a weight slider and see the prediction change in real time
  * Click any neuron and see what concept it represents
  * Compare two inputs side by side
  * Probe for adversarial vulnerabilities
  * Replay training history (epoch scrubber)
  * Export a self-contained "story" HTML

Usage:
    qai.workspace(model)         # blocking; visit http://localhost:7788
    qai.workspace(model, port=8000, open_browser=True)
    qai.generate_workspace(model, path="workspace.html")  # save & open later

The workspace is the visual realization of the design in
``docs/VISUAL_DESIGN.md``.  It is intentionally framework-free (no React,
no build step): one HTML file with embedded CSS+JS, talking to a tiny
Python HTTP server that exposes the loaded model.

The workspace supports three modes:

  * LIVE  -- connected to a real qai model via the embedded HTTP server
  * DEMO  -- pure static synthetic model (open the HTML file directly)
  * STORY -- a recorded session exported for sharing

This module is the foundation of Phase 4 of the roadmap.
"""
from __future__ import annotations

import html
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# Model introspection helpers
# ============================================================================

def _safe_call(model, x):
    """Call a model on input x, returning (output, latency_ms)."""
    t0 = time.perf_counter()
    try:
        if hasattr(model, "predict"):
            out = model.predict(x)
        elif hasattr(model, "__call__"):
            out = model(x)
        else:
            out = None
        latency = (time.perf_counter() - t0) * 1000
        return out, latency
    except Exception as e:
        return {"_error": str(e)}, (time.perf_counter() - t0) * 1000


def _get_weights(model) -> List[Dict[str, Any]]:
    """Extract a normalized weight structure for visualization.

    Returns a list of layer dicts: ``[{name, shape, weights, biases}]``.
    Works against qai Models (with `.technique` or `.weights`), sklearn
    estimators, torch modules, and plain numpy arrays.
    """
    layers: List[Dict[str, Any]] = []

    # 1) qai Model
    tech = getattr(model, "technique", None)
    if tech is not None:
        # Try common attribute names
        for attr in ("net", "model", "_model", "module"):
            sub = getattr(tech, attr, None)
            if sub is not None:
                return _extract_torch_layers(sub) or _extract_sklearn_layers(tech) or layers

    # 2) Direct torch module
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return _extract_torch_layers(model)
    except ImportError:
        pass

    # 3) Direct sklearn estimator
    try:
        from sklearn.base import BaseEstimator
        if isinstance(model, BaseEstimator):
            return _extract_sklearn_layers(model)
    except ImportError:
        pass

    # 4) weights accessor on qai Model
    if hasattr(model, "weights"):
        w = model.weights
        if hasattr(w, "to_dict"):
            w = w.to_dict()
        if isinstance(w, dict):
            for name, arr in w.items():
                try:
                    a = np.asarray(arr, dtype=float)
                    layers.append({
                        "name": str(name),
                        "shape": list(a.shape),
                        "weights": a.flatten()[:512].tolist(),  # cap for transport
                        "n_params": int(a.size),
                        "min": float(np.min(a)),
                        "max": float(np.max(a)),
                        "mean": float(np.mean(a)),
                        "std": float(np.std(a)),
                    })
                except Exception:
                    continue

    # 5) Synthetic fallback
    if not layers:
        layers = _synthetic_layers(architecture=[4, 8, 6, 2])
    return layers


def _extract_torch_layers(module) -> List[Dict[str, Any]]:
    """Walk a torch module's named_parameters and bucket them by layer prefix."""
    layers: Dict[str, Dict[str, Any]] = {}
    try:
        for name, param in module.named_parameters():
            try:
                arr = param.detach().cpu().numpy().astype(float)
            except Exception:
                continue
            prefix = name.rsplit(".", 1)[0] if "." in name else name
            slot = layers.setdefault(prefix, {
                "name": prefix, "params": {}, "n_params": 0, "_weights": [],
                "_biases": [], "min": float("inf"), "max": float("-inf"),
            })
            slot["n_params"] += int(arr.size)
            slot["min"] = min(slot["min"], float(arr.min()))
            slot["max"] = max(slot["max"], float(arr.max()))
            if "weight" in name:
                slot["_weights"] = arr
            elif "bias" in name:
                slot["_biases"] = arr
        out: List[Dict[str, Any]] = []
        for name, slot in layers.items():
            w = slot["_weights"]
            if isinstance(w, np.ndarray) and w.size:
                a = w
                out.append({
                    "name": name,
                    "shape": list(a.shape),
                    "weights": a.flatten()[:512].tolist(),
                    "n_params": slot["n_params"],
                    "min": float(np.min(a)),
                    "max": float(np.max(a)),
                    "mean": float(np.mean(a)),
                    "std": float(np.std(a)),
                })
        return out
    except Exception:
        return []


def _extract_sklearn_layers(est) -> List[Dict[str, Any]]:
    """Pull coefs_/intercept_ from sklearn estimators."""
    layers: List[Dict[str, Any]] = []
    try:
        if hasattr(est, "coefs_") and hasattr(est, "intercepts_"):
            # MLPClassifier / MLPRegressor
            for i, (w, b) in enumerate(zip(est.coefs_, est.intercepts_)):
                layers.append({
                    "name": f"layer{i}",
                    "shape": list(w.shape),
                    "weights": np.asarray(w, dtype=float).flatten()[:512].tolist(),
                    "n_params": int(np.asarray(w).size + np.asarray(b).size),
                    "min": float(np.min(w)),
                    "max": float(np.max(w)),
                    "mean": float(np.mean(w)),
                    "std": float(np.std(w)),
                })
        elif hasattr(est, "coef_"):
            w = np.asarray(est.coef_, dtype=float)
            if w.ndim == 1:
                w = w.reshape(-1, 1)
            layers.append({
                "name": "coef",
                "shape": list(w.shape),
                "weights": w.flatten()[:512].tolist(),
                "n_params": int(w.size + np.asarray(getattr(est, "intercept_", 0)).size),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
            })
        elif hasattr(est, "feature_importances_"):
            w = np.asarray(est.feature_importances_, dtype=float).reshape(1, -1)
            layers.append({
                "name": "feature_importance",
                "shape": list(w.shape),
                "weights": w.flatten()[:512].tolist(),
                "n_params": int(w.size),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
            })
        elif hasattr(est, "tree_") and hasattr(est.tree_, "feature"):
            t = est.tree_
            layers.append({
                "name": "decision_tree",
                "shape": [int(t.node_count)],
                "weights": list(t.feature.astype(float))[:512],
                "n_params": int(t.node_count),
                "min": 0.0,
                "max": float(max(t.feature)),
                "mean": float(np.mean(t.feature)),
                "std": float(np.std(t.feature)),
            })
    except Exception:
        pass
    return layers


def _synthetic_layers(architecture: List[int]) -> List[Dict[str, Any]]:
    """Generate a synthetic model architecture (for demo / story mode)."""
    layers: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for i, (n_in, n_out) in enumerate(zip(architecture[:-1], architecture[1:])):
        w = rng.normal(0, 0.5, size=(n_in, n_out))
        layers.append({
            "name": f"dense_{i}",
            "shape": list(w.shape),
            "weights": w.flatten()[:512].tolist(),
            "n_params": int(w.size),
            "min": float(np.min(w)),
            "max": float(np.max(w)),
            "mean": float(np.mean(w)),
            "std": float(np.std(w)),
        })
    return layers


def _model_metadata(model) -> Dict[str, Any]:
    """One-shot summary of the loaded model."""
    info: Dict[str, Any] = {
        "name": getattr(model, "technique_name", None) or type(model).__name__,
        "type": type(getattr(model, "technique", model)).__name__,
        "trained": bool(getattr(model, "_trained", True)),
        "version": getattr(model, "version", "0.0.0"),
    }
    try:
        layers = _get_weights(model)
        info["n_layers"] = len(layers)
        info["n_params"] = sum(L.get("n_params", 0) for L in layers)
        info["layers"] = [
            {"name": L["name"], "shape": L["shape"], "n_params": L["n_params"]}
            for L in layers
        ]
    except Exception:
        info["layers"] = []
        info["n_params"] = 0
        info["n_layers"] = 0
    # Try qai's own summary() if it exists
    if hasattr(model, "summary"):
        try:
            s = model.summary()
            if isinstance(s, str):
                info["summary_text"] = s
            elif isinstance(s, dict):
                info["summary"] = s
        except Exception:
            pass
    return info


def _parse_input(raw: str) -> Any:
    """Parse a chat input string into model-ready form.

    Accepts JSON lists, comma-separated numbers, raw text, and "key=value" pairs.
    """
    raw = raw.strip()
    if not raw:
        return None
    # JSON array
    if raw.startswith("[") and raw.endswith("]"):
        try:
            return np.array(json.loads(raw), dtype=float)
        except Exception:
            pass
    # CSV numbers
    parts = [p.strip() for p in raw.split(",")]
    try:
        nums = [float(p) for p in parts]
        return np.array(nums)
    except ValueError:
        pass
    # Raw text (string input)
    return raw


def _format_output(out: Any) -> Dict[str, Any]:
    """Normalize a model output for the UI."""
    result: Dict[str, Any] = {"raw": None, "prediction": None, "proba": None,
                              "confidence": None, "classes": None}
    if out is None:
        return result
    if isinstance(out, dict) and "_error" in out:
        result["error"] = out["_error"]
        return result
    # numpy
    try:
        if isinstance(out, np.ndarray):
            arr = out
            if arr.ndim == 0:
                result["prediction"] = float(arr)
                result["confidence"] = 1.0
            elif arr.ndim == 1 and arr.size == 1:
                result["prediction"] = float(arr[0])
                result["confidence"] = 1.0
            elif arr.ndim == 1:
                # Probability distribution or class scores
                if arr.size <= 32:
                    result["proba"] = arr.tolist()
                    idx = int(np.argmax(arr))
                    result["prediction"] = idx
                    result["classes"] = list(range(arr.size))
                    # Softmax for confidence
                    e = np.exp(arr - np.max(arr))
                    s = e / e.sum()
                    result["confidence"] = float(s[idx])
                else:
                    result["prediction"] = float(arr[0])
            elif arr.ndim == 2 and arr.shape[0] == 1:
                return _format_output(arr[0])
            else:
                result["raw"] = arr.tolist()[:50]
            return result
    except Exception:
        pass
    # Tuple (proba, pred) common in sklearn
    if isinstance(out, (tuple, list)) and len(out) == 2:
        try:
            proba, pred = out
            return _format_output(proba) | {"prediction": pred}
        except Exception:
            pass
    # Plain Python
    if isinstance(out, (int, float)):
        result["prediction"] = float(out)
        result["confidence"] = 1.0
    elif isinstance(out, str):
        result["prediction"] = out
        result["confidence"] = 0.5
    else:
        try:
            result["raw"] = str(out)[:200]
        except Exception:
            pass
    return result


# ============================================================================
# The HTTP backend that the workspace talks to
# ============================================================================

class _WorkspaceBackend:
    """Tiny JSON API the workspace frontend calls."""

    def __init__(self, model, story_mode: bool = False):
        self.model = model
        self.story_mode = story_mode
        self._lock = threading.RLock()
        self._history: List[Dict[str, Any]] = []  # weight-edit history
        self._predictions: List[Dict[str, Any]] = []  # recent predictions
        self._pinned_input: Optional[Any] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._layers_cache: Optional[List[Dict[str, Any]]] = None

    # -- metadata ------------------------------------------------------
    def metadata(self) -> Dict[str, Any]:
        if self._metadata is None:
            with self._lock:
                self._metadata = _model_metadata(self.model)
        return self._metadata

    # -- predict -------------------------------------------------------
    def predict(self, raw_input: str) -> Dict[str, Any]:
        x = _parse_input(raw_input)
        out, latency = _safe_call(self.model, x)
        formatted = _format_output(out)
        formatted["latency_ms"] = round(latency, 2)
        formatted["input"] = raw_input
        with self._lock:
            self._predictions.append(formatted)
            if len(self._predictions) > 50:
                self._predictions = self._predictions[-50:]
        return formatted

    # -- weights -------------------------------------------------------
    def get_layers(self) -> List[Dict[str, Any]]:
        if self._layers_cache is None:
            with self._lock:
                self._layers_cache = _get_weights(self.model)
        return self._layers_cache

    def get_weights(self, layer_idx: int) -> Dict[str, Any]:
        layers = self.get_layers()
        if 0 <= layer_idx < len(layers):
            return layers[layer_idx]
        return {"error": "layer index out of range"}

    def edit_weight(self, layer_idx: int, flat_idx: int, new_value: float) -> Dict[str, Any]:
        """Record an edit. Real weight writes are best-effort."""
        with self._lock:
            self._history.append({
                "layer": layer_idx, "flat_idx": flat_idx,
                "new_value": float(new_value), "ts": time.time(),
            })
        # Best-effort write
        try:
            self._apply_weight_edit(layer_idx, flat_idx, new_value)
        except Exception:
            pass
        return {"ok": True, "history_size": len(self._history)}

    def _apply_weight_edit(self, layer_idx: int, flat_idx: int, new_value: float) -> None:
        """Try to mutate an actual weight. Works for sklearn coefs_; torch needs hook."""
        model = self.model
        tech = getattr(model, "technique", model)
        # sklearn-style: coefs_ is a list of 2D arrays
        if hasattr(tech, "coefs_") and isinstance(tech.coefs_, list):
            if 0 <= layer_idx < len(tech.coefs_):
                w = tech.coefs_[layer_idx]
                flat = w.flatten()
                if 0 <= flat_idx < flat.size:
                    flat[flat_idx] = new_value
                    tech.coefs_[layer_idx] = flat.reshape(w.shape)
                    return
        if hasattr(tech, "coef_"):
            w = np.asarray(tech.coef_, dtype=float)
            flat = w.flatten()
            if 0 <= flat_idx < flat.size:
                flat[flat_idx] = new_value
                tech.coef_ = flat.reshape(w.shape)
                return
        # Otherwise just record (no-op write)
        return

    def reset_weights(self) -> Dict[str, Any]:
        """Clear edit history. (Does not undo writes; user should reload.)"""
        with self._lock:
            n = len(self._history)
            self._history.clear()
        return {"ok": True, "cleared": n}

    def weight_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._history)

    # -- compare / counterfactual -------------------------------------
    def pin_input(self, raw_input: str) -> Dict[str, Any]:
        self._pinned_input = _parse_input(raw_input)
        return {"ok": True}

    def compare(self, raw_a: str, raw_b: str) -> Dict[str, Any]:
        a, la = _safe_call(self.model, _parse_input(raw_a))
        b, lb = _safe_call(self.model, _parse_input(raw_b))
        fa = _format_output(a); fa["latency_ms"] = round(la, 2)
        fb = _format_output(b); fb["latency_ms"] = round(lb, 2)
        # Diff
        try:
            pa = np.asarray(fa.get("proba") or fa.get("prediction") or 0.0, dtype=float)
            pb = np.asarray(fb.get("proba") or fb.get("prediction") or 0.0, dtype=float)
            if pa.shape == pb.shape:
                diff = float(np.max(np.abs(pa - pb)))
            else:
                diff = 1.0
        except Exception:
            diff = 0.0
        return {"a": fa, "b": fb, "max_diff": diff}

    # -- adversarial probe --------------------------------------------
    def adversarial_probe(self, raw_input: str, epsilon: float = 0.05) -> Dict[str, Any]:
        """Apply small random noise to the input and see if prediction flips."""
        x = _parse_input(raw_input)
        if not isinstance(x, np.ndarray) or x.size == 0:
            return {"error": "adversarial probe needs numeric input"}
        baseline, _ = _safe_call(self.model, x)
        fb = _format_output(baseline)
        n_trials = 20
        flips = 0
        worst_perturbation = None
        worst_diff = 0.0
        rng = np.random.default_rng(int(time.time()))
        for _ in range(n_trials):
            noise = rng.normal(0, epsilon, size=x.shape)
            x_adv = x + noise
            adv_out, _ = _safe_call(self.model, x_adv)
            fa = _format_output(adv_out)
            try:
                pa = np.asarray(fb.get("proba") or fb.get("prediction") or 0.0, dtype=float)
                pb = np.asarray(fa.get("proba") or fa.get("prediction") or 0.0, dtype=float)
                if pa.shape == pb.shape:
                    diff = float(np.max(np.abs(pa - pb)))
                    if diff > worst_diff:
                        worst_diff = diff
                        worst_perturbation = noise.tolist()
                    if fa.get("prediction") != fb.get("prediction"):
                        flips += 1
            except Exception:
                continue
        return {
            "baseline": fb,
            "n_trials": n_trials,
            "n_flips": flips,
            "flip_rate": flips / n_trials if n_trials else 0.0,
            "worst_diff": worst_diff,
            "worst_perturbation": worst_perturbation,
            "epsilon": epsilon,
        }

    # -- health / blind spots -----------------------------------------
    def health(self) -> Dict[str, Any]:
        layers = self.get_layers()
        dead = 0; saturated = 0; total = 0
        for L in layers:
            try:
                w = np.array(L.get("weights", []), dtype=float)
                if w.size == 0:
                    continue
                total += w.size
                dead += int((np.abs(w) < 1e-4).sum())
                saturated += int((np.abs(w) > 3.0 * (L.get("std", 1.0) + 1e-9)).sum())
            except Exception:
                continue
        return {
            "n_layers": len(layers),
            "n_params": sum(L.get("n_params", 0) for L in layers),
            "dead_neurons": dead,
            "saturated_neurons": saturated,
            "total_weights_inspected": total,
            "dead_pct": round(dead / max(total, 1) * 100, 2),
            "saturated_pct": round(saturated / max(total, 1) * 100, 2),
            "overall_health": "good" if dead / max(total, 1) < 0.05 else "warning",
        }

    # -- story export --------------------------------------------------
    def export_story(self) -> Dict[str, Any]:
        """Export a self-contained story (model snapshot + recent predictions)."""
        return {
            "model": self.metadata(),
            "layers": self.get_layers(),
            "predictions": self._predictions[-20:],
            "weight_history": list(self._history),
            "exported_at": time.time(),
        }


# ============================================================================
# The frontend (single-file HTML/CSS/JS workspace)
# ============================================================================

WORKSPACE_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QAI Model Microscope</title>
<style>
  :root {
    --bg: #0a0e1a;
    --bg-elev: #131826;
    --bg-elev2: #1a2030;
    --border: #2a3142;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #5eead4;
    --accent-dim: #2dd4bf;
    --warn: #fbbf24;
    --danger: #f87171;
    --good: #4ade80;
    --info: #60a5fa;
    --purple: #c084fc;
    --pink: #f472b6;
    --shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
    background: var(--bg);
    color: var(--text);
    overflow: hidden;
    font-size: 13px;
    line-height: 1.5;
  }
  /* ---------- top bar ---------- */
  .topbar {
    height: 48px;
    background: var(--bg-elev);
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 16px;
    gap: 12px;
    flex-shrink: 0;
  }
  .topbar .brand {
    font-weight: 700;
    color: var(--accent);
    font-size: 14px;
    letter-spacing: 0.5px;
  }
  .topbar .brand .micro { color: var(--text-dim); font-weight: 400; font-size: 11px; }
  .topbar .sep { width: 1px; height: 24px; background: var(--border); }
  .topbar .model-info {
    color: var(--text-dim);
    font-size: 12px;
    display: flex; align-items: center; gap: 8px;
  }
  .topbar .model-info b { color: var(--text); }
  .topbar .badge {
    background: var(--bg-elev2);
    border: 1px solid var(--border);
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 10px;
    color: var(--accent);
  }
  .topbar .spacer { flex: 1; }
  .topbar button {
    background: var(--bg-elev2);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 12px;
    transition: all 0.15s;
  }
  .topbar button:hover { background: var(--accent); color: var(--bg); border-color: var(--accent); }
  .topbar button.danger:hover { background: var(--danger); color: white; border-color: var(--danger); }

  /* ---------- main layout ---------- */
  .main {
    display: grid;
    grid-template-columns: 320px 1fr 320px;
    height: calc(100vh - 48px - 28px - 240px);
  }
  .col {
    background: var(--bg-elev);
    border-right: 1px solid var(--border);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .col:last-child { border-right: none; }
  .col-header {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: 1px;
    background: var(--bg-elev2);
  }
  .col-body { padding: 12px; flex: 1; overflow-y: auto; }

  /* ---------- input panel ---------- */
  .chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .chat-msg {
    background: var(--bg-elev2);
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 6px;
    font-size: 12px;
    word-wrap: break-word;
  }
  .chat-msg.user { border-left: 2px solid var(--accent); }
  .chat-msg.pred { border-left: 2px solid var(--purple); }
  .chat-msg .lbl { color: var(--text-dim); font-size: 10px; margin-bottom: 2px; text-transform: uppercase; letter-spacing: 0.5px; }
  .chat-input-row {
    padding: 10px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 8px;
  }
  .chat-input {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 10px;
    border-radius: 6px;
    font-family: 'SF Mono', Consolas, monospace;
    font-size: 12px;
    outline: none;
  }
  .chat-input:focus { border-color: var(--accent); }
  .send-btn {
    background: var(--accent);
    color: var(--bg);
    border: none;
    padding: 0 14px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
  }
  .send-btn:hover { background: var(--accent-dim); }
  .send-btn:disabled { background: var(--border); color: var(--text-dim); cursor: not-allowed; }
  .examples {
    padding: 8px 12px;
    border-top: 1px solid var(--border);
  }
  .examples-title { color: var(--text-dim); font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
  .example-btns { display: flex; flex-wrap: wrap; gap: 4px; }
  .example-btn {
    background: var(--bg-elev2);
    border: 1px solid var(--border);
    color: var(--text-dim);
    padding: 3px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 10px;
    font-family: monospace;
  }
  .example-btn:hover { border-color: var(--accent); color: var(--accent); }
  .file-drop {
    margin: 8px 12px;
    padding: 16px;
    border: 2px dashed var(--border);
    border-radius: 6px;
    text-align: center;
    color: var(--text-dim);
    font-size: 11px;
    cursor: pointer;
    transition: all 0.15s;
  }
  .file-drop:hover, .file-drop.dragover { border-color: var(--accent); color: var(--accent); }

  /* ---------- canvas panel ---------- */
  .canvas-container {
    flex: 1;
    position: relative;
    background: var(--bg);
    background-image:
      radial-gradient(circle at 1px 1px, var(--border) 1px, transparent 0);
    background-size: 24px 24px;
    overflow: hidden;
  }
  #archCanvas {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    display: block;
  }
  .canvas-controls {
    position: absolute;
    top: 10px; right: 10px;
    display: flex; gap: 4px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px;
  }
  .canvas-controls button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 12px;
  }
  .canvas-controls button:hover, .canvas-controls button.active { background: var(--accent); color: var(--bg); }
  .canvas-legend {
    position: absolute;
    bottom: 10px; left: 10px;
    background: var(--bg-elev);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 10px;
    color: var(--text-dim);
    display: flex; flex-direction: column; gap: 4px;
  }
  .legend-row { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; }

  /* ---------- output panel ---------- */
  .output-section {
    padding: 12px;
    border-bottom: 1px solid var(--border);
  }
  .output-section:last-child { border-bottom: none; }
  .output-section-title {
    color: var(--text-dim);
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .prediction-display {
    font-size: 32px;
    font-weight: 700;
    color: var(--accent);
    text-align: center;
    padding: 8px;
    font-family: 'SF Mono', Consolas, monospace;
  }
  .prediction-label {
    text-align: center;
    color: var(--text-dim);
    font-size: 11px;
    margin-bottom: 8px;
  }
  .confidence-bar-container {
    background: var(--bg);
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
    margin: 8px 0;
  }
  .confidence-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--danger), var(--warn), var(--good));
    width: 0%;
    transition: width 0.4s ease;
  }
  .confidence-label {
    display: flex; justify-content: space-between;
    font-size: 10px; color: var(--text-dim);
  }
  .metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
  }
  .metric-card {
    background: var(--bg-elev2);
    border: 1px solid var(--border);
    padding: 6px 8px;
    border-radius: 4px;
  }
  .metric-card .label { color: var(--text-dim); font-size: 9px; text-transform: uppercase; }
  .metric-card .value { font-size: 14px; font-weight: 600; color: var(--text); font-family: monospace; }
  .metric-card .value .unit { color: var(--text-dim); font-size: 10px; font-weight: 400; }

  .explanation-list { font-size: 11px; }
  .explanation-row {
    display: flex; align-items: center; gap: 6px;
    margin: 4px 0;
  }
  .explanation-row .name { width: 90px; color: var(--text-dim); }
  .explanation-row .bar {
    flex: 1;
    background: var(--bg-elev2);
    height: 6px;
    border-radius: 3px;
    overflow: hidden;
  }
  .explanation-row .fill { height: 100%; background: var(--info); }
  .explanation-row .val { color: var(--text); font-family: monospace; font-size: 10px; }

  .health-list { font-size: 11px; }
  .health-row {
    display: flex; justify-content: space-between;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
  }
  .health-row:last-child { border-bottom: none; }
  .health-row .name { color: var(--text-dim); }
  .health-row .val { color: var(--text); font-family: monospace; font-weight: 600; }
  .health-row .val.warn { color: var(--warn); }
  .health-row .val.danger { color: var(--danger); }
  .health-row .val.good { color: var(--good); }

  /* ---------- tabs (output panel) ---------- */
  .tabs {
    display: flex;
    border-bottom: 1px solid var(--border);
    background: var(--bg-elev2);
  }
  .tab {
    flex: 1;
    padding: 8px 4px;
    text-align: center;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    cursor: pointer;
    border-bottom: 2px solid transparent;
  }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ---------- weight inspector (bottom) ---------- */
  .inspector {
    height: 240px;
    background: var(--bg-elev);
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
  }
  .inspector-tabs {
    display: flex;
    background: var(--bg-elev2);
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .inspector-tab {
    padding: 8px 16px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    cursor: pointer;
    border-right: 1px solid var(--border);
  }
  .inspector-tab:hover { color: var(--text); }
  .inspector-tab.active { color: var(--accent); background: var(--bg-elev); }
  .inspector-body {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
    display: none;
  }
  .inspector-body.active { display: block; }
  .inspector-body.active.flex { display: flex; gap: 8px; }

  .heatmap {
    display: grid;
    gap: 1px;
    background: var(--border);
    padding: 1px;
    width: fit-content;
  }
  .heatmap-cell {
    width: 12px; height: 12px;
    cursor: pointer;
  }
  .heatmap-cell:hover { outline: 2px solid var(--accent); outline-offset: -1px; }

  .slider-row {
    display: grid;
    grid-template-columns: 110px 1fr 60px;
    gap: 8px;
    align-items: center;
    margin: 6px 0;
    font-size: 11px;
  }
  .slider-row .name { color: var(--text-dim); font-family: monospace; }
  .slider-row input[type="range"] {
    -webkit-appearance: none;
    background: var(--bg-elev2);
    height: 4px;
    border-radius: 2px;
    outline: none;
  }
  .slider-row input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: var(--accent);
    cursor: pointer;
  }
  .slider-row .val { color: var(--text); font-family: monospace; text-align: right; }

  .sensitivity-canvas { width: 100%; height: 180px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; }

  .history-list { font-size: 11px; }
  .history-row {
    display: grid;
    grid-template-columns: 80px 1fr 80px 80px;
    gap: 8px;
    padding: 4px 0;
    border-bottom: 1px solid var(--border);
    font-family: monospace;
  }
  .history-row .ts { color: var(--text-dim); }
  .history-row .name { color: var(--text); }
  .history-row .old { color: var(--warn); }
  .history-row .new { color: var(--accent); }

  /* ---------- status bar ---------- */
  .statusbar {
    height: 28px;
    background: var(--bg-elev);
    border-top: 1px solid var(--border);
    display: flex;
    align-items: center;
    padding: 0 12px;
    gap: 16px;
    font-size: 10px;
    color: var(--text-dim);
    flex-shrink: 0;
    font-family: 'SF Mono', Consolas, monospace;
  }
  .statusbar .stat { display: flex; align-items: center; gap: 4px; }
  .statusbar .stat b { color: var(--text); }
  .statusbar .pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--good);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }

  /* ---------- story mode ---------- */
  .story-tour {
    position: fixed;
    bottom: 270px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-elev);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: var(--shadow);
    max-width: 500px;
    z-index: 1000;
    display: none;
  }
  .story-tour.active { display: block; }
  .story-tour .step-num { color: var(--accent); font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
  .story-tour .step-text { color: var(--text); font-size: 13px; margin-bottom: 8px; }
  .story-tour .step-nav { display: flex; justify-content: space-between; align-items: center; }

  /* ---------- compare overlay ---------- */
  .compare-overlay {
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    background: var(--bg-elev);
    border: 1px solid var(--accent);
    border-radius: 8px;
    padding: 16px;
    box-shadow: var(--shadow);
    width: 480px;
    display: none;
    z-index: 500;
  }
  .compare-overlay.active { display: block; }
  .compare-overlay h3 { color: var(--accent); margin-bottom: 12px; font-size: 13px; }
  .compare-overlay .compare-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .compare-overlay .compare-cell { background: var(--bg-elev2); padding: 8px; border-radius: 4px; border: 1px solid var(--border); }
  .compare-overlay .compare-cell h4 { color: var(--text-dim); font-size: 10px; text-transform: uppercase; margin-bottom: 6px; }
  .compare-overlay .compare-cell .pred { font-size: 18px; color: var(--text); font-family: monospace; }
  .compare-overlay .diff { margin-top: 8px; padding: 8px; background: var(--bg-elev2); border-radius: 4px; text-align: center; font-family: monospace; }
  .compare-overlay .close { position: absolute; top: 8px; right: 12px; cursor: pointer; color: var(--text-dim); }
  .compare-overlay .close:hover { color: var(--danger); }

  /* ---------- responsive ---------- */
  @media (max-width: 1199px) {
    .main { grid-template-columns: 50% 50%; }
    .col.output { display: none; }
    .col.output.mobile-active { display: flex; position: absolute; top: 48px; bottom: 268px; left: 50%; right: 0; z-index: 100; }
  }
  @media (max-width: 767px) {
    .main { grid-template-columns: 1fr; }
    .col.canvas, .col.output { display: none; }
    .col.input.mobile-active { display: flex; }
    .col.canvas.mobile-active { display: flex; }
    .col.output.mobile-active { display: flex; position: static; }
    .mobile-tab-bar { display: flex !important; }
  }
  .mobile-tab-bar {
    display: none;
    background: var(--bg-elev2);
    border-bottom: 1px solid var(--border);
    height: 36px;
    align-items: center;
    justify-content: space-around;
  }
  .mobile-tab-bar button {
    background: transparent;
    color: var(--text-dim);
    border: none;
    padding: 8px 16px;
    font-size: 11px;
    cursor: pointer;
  }
  .mobile-tab-bar button.active { color: var(--accent); border-bottom: 2px solid var(--accent); }
</style>
</head>
<body>

<!-- ============= TOP BAR ============= -->
<div class="topbar">
  <div class="brand">QAI<span class="micro"> model microscope</span></div>
  <div class="sep"></div>
  <div class="model-info">
    model: <b id="modelName">—</b>
    <span class="badge" id="modelType">—</span>
    <span class="badge" id="paramCount" style="color: var(--purple)">—</span>
  </div>
  <div class="spacer"></div>
  <button onclick="exportStory()" title="Export self-contained story">Export Story</button>
  <button onclick="toggleStoryMode()" id="storyBtn">Story Mode</button>
  <button onclick="openCompare()" title="Compare two inputs side by side">Compare</button>
  <button onclick="runAdversarial()" title="One-click adversarial probe">Adversarial</button>
  <button onclick="resetWeights()" class="danger" title="Reset weight edits">Reset</button>
</div>

<!-- ============= MAIN 3-COLUMN ============= -->
<div class="main">

  <!-- COLUMN 1: INPUT -->
  <div class="col input mobile-active" id="colInput">
    <div class="col-header">Input</div>
    <div class="chat-history" id="chatHistory">
      <div class="chat-msg pred"><div class="lbl">Welcome</div>Type an input below to send it through the model. Try <code>[1.0, 2.0, 3.0, 4.0]</code> or pick an example.</div>
    </div>
    <div class="examples">
      <div class="examples-title">Examples</div>
      <div class="example-btns" id="exampleBtns">
        <button class="example-btn" onclick="sendExample('[5.1, 3.5, 1.4, 0.2]')">iris setosa</button>
        <button class="example-btn" onclick="sendExample('[6.2, 2.9, 4.3, 1.3]')">iris versicolor</button>
        <button class="example-btn" onclick="sendExample('[7.7, 3.8, 6.7, 2.2]')">iris virginica</button>
        <button class="example-btn" onclick="sendExample('[1.0, 2.0, 3.0, 4.0]')">[1,2,3,4]</button>
        <button class="example-btn" onclick="sendExample('Hello world')">text</button>
        <button class="example-btn" onclick="sendRandom()">random</button>
      </div>
    </div>
    <div class="file-drop" id="fileDrop">Drop CSV / image / text here<br><span style="font-size:10px">or click to upload</span></div>
    <div class="chat-input-row">
      <input class="chat-input" id="chatInput" placeholder="[1.0, 2.0, ...] or text" onkeydown="if(event.key==='Enter')sendInput()">
      <button class="send-btn" id="sendBtn" onclick="sendInput()">Send</button>
    </div>
  </div>

  <!-- COLUMN 2: ARCHITECTURE CANVAS -->
  <div class="col canvas" id="colCanvas">
    <div class="col-header">Architecture Canvas</div>
    <div class="canvas-container">
      <canvas id="archCanvas"></canvas>
      <div class="canvas-controls">
        <button onclick="setRenderMode('full')" id="modeFull" class="active" title="Show every node">Full</button>
        <button onclick="setRenderMode('summary')" id="modeSummary" title="Layer summaries">Summary</button>
        <button onclick="setRenderMode('block')" id="modeBlock" title="Block-level boxes">Block</button>
        <button onclick="toggleAnim()" id="animBtn" title="Replay forward-pass animation">▶</button>
        <button onclick="zoomBy(1.2)" title="Zoom in">+</button>
        <button onclick="zoomBy(0.8)" title="Zoom out">−</button>
      </div>
      <div class="canvas-legend">
        <div class="legend-row"><div class="legend-swatch" style="background:var(--info)"></div>positive weight</div>
        <div class="legend-row"><div class="legend-swatch" style="background:var(--pink)"></div>negative weight</div>
        <div class="legend-row"><div class="legend-swatch" style="background:var(--accent)"></div>active path</div>
        <div class="legend-row"><div class="legend-swatch" style="background:var(--text-dim)"></div>inactive / weak</div>
      </div>
    </div>
  </div>

  <!-- COLUMN 3: OUTPUT -->
  <div class="col output" id="colOutput">
    <div class="tabs">
      <div class="tab active" onclick="setOutputTab('prediction', this)">Prediction</div>
      <div class="tab" onclick="setOutputTab('explanation', this)">Explain</div>
      <div class="tab" onclick="setOutputTab('health', this)">Health</div>
    </div>
    <div class="output-section" id="outputPrediction">
      <div class="output-section-title">Predicted Output</div>
      <div class="prediction-display" id="predictionValue">—</div>
      <div class="prediction-label" id="predictionLabel">no input yet</div>
      <div class="confidence-bar-container"><div class="confidence-bar" id="confBar"></div></div>
      <div class="confidence-label">
        <span>confidence</span>
        <span id="confVal">—</span>
      </div>
    </div>
    <div class="output-section">
      <div class="output-section-title">Metrics</div>
      <div class="metric-grid">
        <div class="metric-card"><div class="label">Latency</div><div class="value" id="mLatency">—<span class="unit">ms</span></div></div>
        <div class="metric-card"><div class="label">Tokens/s</div><div class="value" id="mTps">—</div></div>
        <div class="metric-card"><div class="label">Params active</div><div class="value" id="mActive">—</div></div>
        <div class="metric-card"><div class="label">Surprise</div><div class="value" id="mSurprise">—</div></div>
      </div>
    </div>
    <div class="output-section" id="outputExplanation" style="display:none">
      <div class="output-section-title">Top contributing features</div>
      <div class="explanation-list" id="explanationList"><div style="color:var(--text-dim);font-size:11px">No prediction yet.</div></div>
    </div>
    <div class="output-section" id="outputHealth" style="display:none">
      <div class="output-section-title">Blind spot detection</div>
      <div class="health-list" id="healthList"><div style="color:var(--text-dim);font-size:11px">Click "Health" to inspect.</div></div>
    </div>
  </div>
</div>

<!-- ============= WEIGHT INSPECTOR (bottom) ============= -->
<div class="inspector">
  <div class="inspector-tabs">
    <div class="inspector-tab active" onclick="setInspectorTab('heatmap', this)">Heatmap</div>
    <div class="inspector-tab" onclick="setInspectorTab('sliders', this)">Sliders</div>
    <div class="inspector-tab" onclick="setInspectorTab('sensitivity', this)">Sensitivity</div>
    <div class="inspector-tab" onclick="setInspectorTab('history', this)">History</div>
  </div>
  <div class="inspector-body active" id="inspectorHeatmap">
    <div id="heatmapContainer"><div style="color:var(--text-dim);font-size:11px">Loading weights…</div></div>
  </div>
  <div class="inspector-body" id="inspectorSliders">
    <div id="slidersContainer"><div style="color:var(--text-dim);font-size:11px">Loading top-K weights…</div></div>
  </div>
  <div class="inspector-body" id="inspectorSensitivity">
    <canvas class="sensitivity-canvas" id="sensitivityCanvas"></canvas>
    <div style="font-size:10px;color:var(--text-dim);margin-top:6px">Pick a weight slider above to sweep its value from min→max and see how the prediction responds.</div>
  </div>
  <div class="inspector-body" id="inspectorHistory">
    <div class="history-list" id="historyList"><div style="color:var(--text-dim);font-size:11px">No weight edits yet. Drag a slider to start.</div></div>
  </div>
</div>

<!-- ============= STATUS BAR ============= -->
<div class="statusbar">
  <div class="stat"><div class="pulse"></div><span id="statusMode">live</span></div>
  <div class="stat">tokens: <b id="sTokens">0 in / 0 out</b></div>
  <div class="stat">latency: <b id="sLatency">—ms</b></div>
  <div class="stat">conf: <b id="sConf">—%</b></div>
  <div class="stat">params active: <b id="sActive">— / —</b></div>
  <div class="stat" id="sExperts" style="display:none">experts: <b>—</b></div>
  <div class="spacer" style="flex:1"></div>
  <div class="stat">qai v1.1.0</div>
</div>

<!-- ============= STORY MODE OVERLAY ============= -->
<div class="story-tour" id="storyTour">
  <div class="step-num">Step <span id="storyStep">1</span> of <span id="storyTotal">5</span></div>
  <div class="step-text" id="storyText">Welcome to the model microscope.</div>
  <div class="step-nav">
    <button onclick="prevStoryStep()">Prev</button>
    <button onclick="nextStoryStep()" id="storyNextBtn">Next</button>
    <button onclick="closeStory()">Close</button>
  </div>
</div>

<!-- ============= COMPARE OVERLAY ============= -->
<div class="compare-overlay" id="compareOverlay">
  <span class="close" onclick="closeCompare()">✕</span>
  <h3>Compare two inputs</h3>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:8px">
    <input class="chat-input" id="compareA" placeholder="input A" value="[5.1, 3.5, 1.4, 0.2]">
    <input class="chat-input" id="compareB" placeholder="input B" value="[7.7, 3.8, 6.7, 2.2]">
  </div>
  <button class="send-btn" onclick="runCompare()" style="width:100%">Compare</button>
  <div class="compare-row" id="compareResult" style="margin-top:12px;display:none">
    <div class="compare-cell"><h4>Input A</h4><div class="pred" id="predA">—</div></div>
    <div class="compare-cell"><h4>Input B</h4><div class="pred" id="predB">—</div></div>
  </div>
  <div class="diff" id="compareDiff" style="display:none">max diff: —</div>
</div>

<script>
// ============================================================================
// State
// ============================================================================
const STATE = {
  mode: 'live',           // 'live' | 'demo' | 'story'
  model: null,            // metadata from /api/model
  layers: [],             // weight layers
  renderMode: 'full',
  zoom: 1.0,
  pan: { x: 0, y: 0 },
  animating: false,
  activeInput: null,
  lastOutput: null,
  selectedLayer: 0,
  selectedWeight: 0,
  storyStep: 0,
  storySteps: [
    "Welcome to the model microscope. We're about to walk through what happens when an input flows through your model.",
    "Step 1: Your input arrives at the input layer (left side of canvas). Each node glows brighter when its value is larger.",
    "Step 2: Activations propagate through hidden layers. The strongest paths are highlighted — these are the weights doing the work.",
    "Step 3: The winning output node gets a golden ring. The confidence bar on the right shows how sure the model is.",
    "Step 4: Drag a slider in the Weight Inspector below to mutate a weight — watch the prediction change in real time. This is causal debugging.",
    "Done. Try the Compare button to overlay two inputs, or Adversarial to probe for vulnerabilities."
  ],
  tokensIn: 0, tokensOut: 0,
};

// ============================================================================
// API client (LIVE mode talks to backend; DEMO/STORY mode uses synthetic data)
// ============================================================================
const API = {
  async get(url) {
    if (STATE.mode !== 'live') return null;
    const r = await fetch(url);
    return r.json();
  },
  async post(url, body) {
    if (STATE.mode !== 'live') return null;
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    return r.json();
  },
};

// ============================================================================
// Boot
// ============================================================================
window.addEventListener('DOMContentLoaded', async () => {
  // Detect mode: if backend not reachable, switch to DEMO
  try {
    const meta = await API.get('/api/model');
    if (meta) {
      STATE.model = meta;
      STATE.mode = 'live';
      document.getElementById('statusMode').textContent = 'live';
    } else {
      throw new Error('no backend');
    }
  } catch (e) {
    STATE.mode = 'demo';
    STATE.model = syntheticModel();
    document.getElementById('statusMode').textContent = 'demo';
    document.getElementById('statusMode').style.color = 'var(--warn)';
  }
  renderModelInfo();
  await loadLayers();
  await loadHealth();
  setupCanvas();
  setupFileDrop();
  drawCanvas();
  drawHeatmap();
  drawSliders();
});

function syntheticModel() {
  // Fallback synthetic model when no backend is connected
  return {
    name: 'DemoMLP',
    type: 'MLPClassifier',
    trained: true,
    n_layers: 4,
    n_params: 86,
    layers: [
      {name: 'dense_0', shape: [4, 8], n_params: 32, min: -1.2, max: 1.3, mean: 0.05, std: 0.6, weights: Array.from({length: 32}, (_, i) => Math.sin(i * 0.7) * 0.8)},
      {name: 'dense_1', shape: [8, 6], n_params: 48, min: -1.5, max: 1.6, mean: -0.02, std: 0.7, weights: Array.from({length: 48}, (_, i) => Math.cos(i * 0.5) * 0.9)},
      {name: 'dense_2', shape: [6, 2], n_params: 12, min: -2.0, max: 2.1, mean: 0.1, std: 1.0, weights: Array.from({length: 12}, (_, i) => Math.tan(i * 0.3) * 0.5)},
    ],
  };
}

function renderModelInfo() {
  document.getElementById('modelName').textContent = STATE.model.name;
  document.getElementById('modelType').textContent = STATE.model.type;
  document.getElementById('paramCount').textContent = STATE.model.n_params.toLocaleString() + ' params';
}

// ============================================================================
// Load weights and health
// ============================================================================
async function loadLayers() {
  if (STATE.mode === 'live') {
    const data = await API.get('/api/layers');
    STATE.layers = data.layers || [];
  } else {
    STATE.layers = STATE.model.layers || [];
  }
}

async function loadHealth() {
  let h;
  if (STATE.mode === 'live') {
    h = await API.get('/api/health');
  } else {
    h = {
      n_layers: 3, n_params: 92,
      dead_neurons: 4, saturated_neurons: 2, total_weights_inspected: 92,
      dead_pct: 4.3, saturated_pct: 2.2, overall_health: 'good',
    };
  }
  STATE.health = h;
  renderHealth();
}

function renderHealth() {
  const h = STATE.health;
  if (!h) return;
  const el = document.getElementById('healthList');
  const cls = (v, thresh) => v > thresh * 2 ? 'danger' : (v > thresh ? 'warn' : 'good');
  el.innerHTML = `
    <div class="health-row"><span class="name">Layers inspected</span><span class="val good">${h.n_layers}</span></div>
    <div class="health-row"><span class="name">Total params</span><span class="val">${h.n_params.toLocaleString()}</span></div>
    <div class="health-row"><span class="name">Dead neurons</span><span class="val ${cls(h.dead_pct, 3)}">${h.dead_neurons} (${h.dead_pct}%)</span></div>
    <div class="health-row"><span class="name">Saturated neurons</span><span class="val ${cls(h.saturated_pct, 5)}">${h.saturated_neurons} (${h.saturated_pct}%)</span></div>
    <div class="health-row"><span class="name">Overall health</span><span class="val ${h.overall_health === 'good' ? 'good' : 'warn'}">${h.overall_health.toUpperCase()}</span></div>
  `;
}

// ============================================================================
// Canvas / Architecture rendering
// ============================================================================
let canvas, ctx;
let nodePositions = []; // {x, y, layer, idx, label, value}

function setupCanvas() {
  canvas = document.getElementById('archCanvas');
  ctx = canvas.getContext('2d');
  resizeCanvas();
  window.addEventListener('resize', () => { resizeCanvas(); drawCanvas(); });
  // Pan & zoom
  let dragging = false, lastX = 0, lastY = 0;
  canvas.addEventListener('mousedown', (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    STATE.pan.x += (e.clientX - lastX); STATE.pan.y += (e.clientY - lastY);
    lastX = e.clientX; lastY = e.clientY;
    drawCanvas();
  });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    zoomBy(e.deltaY < 0 ? 1.1 : 0.9);
  });
  canvas.addEventListener('click', (e) => {
    // Click a node → load its weights into inspector
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    for (const n of nodePositions) {
      const dx = n.sx - cx, dy = n.sy - cy;
      if (dx*dx + dy*dy < 100) {
        STATE.selectedLayer = n.layer;
        document.getElementById('inspectorHeatmap').scrollIntoView({behavior:'smooth'});
        setInspectorTab('heatmap', document.querySelector('.inspector-tab'));
        return;
      }
    }
  });
}

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  ctx.scale(dpr, dpr);
}

function zoomBy(f) {
  STATE.zoom *= f;
  STATE.zoom = Math.max(0.3, Math.min(3, STATE.zoom));
  drawCanvas();
}

function setRenderMode(mode) {
  STATE.renderMode = mode;
  ['Full', 'Summary', 'Block'].forEach(m => {
    document.getElementById('mode' + m).classList.toggle('active', m.toLowerCase() === mode);
  });
  drawCanvas();
}

function toggleAnim() {
  STATE.animating = !STATE.animating;
  document.getElementById('animBtn').textContent = STATE.animating ? '⏸' : '▶';
  if (STATE.animating) animateForwardPass();
}

function animateForwardPass() {
  if (!STATE.animating) return;
  let layer = 0;
  const step = () => {
    if (!STATE.animating) return;
    STATE.activeLayer = layer;
    drawCanvas();
    layer++;
    if (layer >= STATE.layers.length + 1) {
      STATE.animating = false;
      STATE.activeLayer = null;
      document.getElementById('animBtn').textContent = '▶';
      drawCanvas();
      return;
    }
    setTimeout(step, 350);
  };
  step();
}

function colorForWeight(w, min, max) {
  if (w === undefined || w === null || isNaN(w)) return getCss('--text-dim');
  const mag = Math.max(-1, Math.min(1, (w - min) / Math.max(1e-9, max - min) * 2 - 1));
  if (mag > 0) {
    const a = Math.abs(mag);
    return `rgba(96, 165, 250, ${0.2 + a * 0.8})`;  // info blue
  } else {
    const a = Math.abs(mag);
    return `rgba(244, 114, 182, ${0.2 + a * 0.8})`;  // pink
  }
}

function getCss(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawCanvas() {
  if (!ctx) return;
  const W = canvas.clientWidth, H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);
  ctx.save();
  ctx.translate(STATE.pan.x, STATE.pan.y);
  ctx.scale(STATE.zoom, STATE.zoom);

  nodePositions = [];
  const layers = STATE.layers;
  if (!layers.length) {
    ctx.fillStyle = getCss('--text-dim');
    ctx.font = '14px sans-serif';
    ctx.fillText('No weights to display', 20, 30);
    ctx.restore();
    return;
  }
  const nCols = layers.length + 1;
  const colW = (W - 80) / nCols;
  const maxNodes = Math.max(...layers.map(L => Math.max(L.shape[0], L.shape[1])), 8);

  layers.forEach((L, i) => {
    const x = 40 + colW * (i + 1);
    const inN = Math.min(L.shape[0], STATE.renderMode === 'full' ? 16 : 6);
    const outN = Math.min(L.shape[1], STATE.renderMode === 'full' ? 16 : 6);
    const startY = (H - (inN * 22 + outN * 22) / 2 * 0) / 2;

    // input nodes
    for (let k = 0; k < inN; k++) {
      const y = H / 2 - (inN * 22) / 2 + k * 22 + 11;
      const w = L.weights ? L.weights[k * (L.weights.length / Math.max(1, inN)) | 0] : 0;
      ctx.fillStyle = colorForWeight(w, L.min, L.max);
      ctx.beginPath();
      ctx.arc(x - colW * 0.4, y, 6, 0, Math.PI * 2);
      ctx.fill();
      if (k === 0) {
        ctx.fillStyle = getCss('--text-dim');
        ctx.font = '9px monospace';
        ctx.fillText(L.name, x - colW * 0.4 - 30, y - 18);
      }
      nodePositions.push({sx: x - colW * 0.4, sy: y, layer: i, idx: k, label: L.name + '[' + k + ']'});
    }
    // output nodes (next layer)
    for (let k = 0; k < outN; k++) {
      const y = H / 2 - (outN * 22) / 2 + k * 22 + 11;
      const w = L.weights ? L.weights[k * (L.weights.length / Math.max(1, outN)) | 0] : 0;
      const isActive = STATE.activeLayer === i + 1;
      ctx.fillStyle = colorForWeight(w, L.min, L.max);
      if (isActive) {
        ctx.strokeStyle = getCss('--accent');
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x + colW * 0.4, y, 8, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(x + colW * 0.4, y, 6, 0, Math.PI * 2);
      ctx.fill();
      nodePositions.push({sx: x + colW * 0.4, sy: y, layer: i + 1, idx: k, label: 'out[' + k + ']'});
    }
    // Edges (sample)
    const maxEdges = STATE.renderMode === 'full' ? 40 : 12;
    for (let e = 0; e < maxEdges; e++) {
      const a = e % inN, b = (e * 7) % outN;
      const y1 = H / 2 - (inN * 22) / 2 + a * 22 + 11;
      const y2 = H / 2 - (outN * 22) / 2 + b * 22 + 11;
      const wIdx = (a * L.shape[1] + b) % L.weights.length;
      const w = L.weights ? L.weights[wIdx] : 0;
      ctx.strokeStyle = colorForWeight(w, L.min, L.max);
      ctx.lineWidth = Math.abs(w) * 1.5 + 0.2;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.moveTo(x - colW * 0.4 + 6, y1);
      ctx.lineTo(x + colW * 0.4 - 6, y2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
  });

  ctx.restore();
}

// ============================================================================
// Input panel
// ============================================================================
async function sendInput() {
  const inp = document.getElementById('chatInput');
  const val = inp.value.trim();
  if (!val) return;
  document.getElementById('sendBtn').disabled = true;
  appendChat('user', 'input: ' + val);
  STATE.tokensIn += val.length;
  let result;
  if (STATE.mode === 'live') {
    result = await API.post('/api/predict', {input: val});
  } else {
    result = demoPredict(val);
  }
  STATE.lastOutput = result;
  renderOutput(result, val);
  document.getElementById('chatInput').value = '';
  document.getElementById('sendBtn').disabled = false;
  STATE.animating = true;
  toggleAnim();
  setTimeout(() => { toggleAnim(); STATE.animating = true; toggleAnim(); }, 0);
  // Restart animation
  STATE.animating = true;
  document.getElementById('animBtn').textContent = '⏸';
  animateForwardPass();
}

function sendExample(val) {
  document.getElementById('chatInput').value = val;
  sendInput();
}

function sendRandom() {
  const arr = Array.from({length: 4}, () => +(Math.random() * 10).toFixed(2));
  sendExample('[' + arr.join(', ') + ']');
}

function appendChat(kind, text) {
  const div = document.createElement('div');
  div.className = 'chat-msg ' + kind;
  div.innerHTML = '<div class="lbl">' + (kind === 'user' ? 'You' : 'Model') + '</div>' + escapeHtml(text);
  const hist = document.getElementById('chatHistory');
  hist.appendChild(div);
  hist.scrollTop = hist.scrollHeight;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function demoPredict(input) {
  // Synthetic prediction for demo mode
  const seed = input.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
  const r = (seed * 9301 + 49297) % 233280 / 233280;
  const classes = ['setosa', 'versicolor', 'virginica'];
  const idx = Math.floor(r * 3);
  const conf = 0.6 + r * 0.39;
  return {
    prediction: idx,
    classes: classes,
    proba: classes.map((_, i) => i === idx ? conf : (1 - conf) / 2),
    confidence: conf,
    latency_ms: 4 + r * 8,
  };
}

// ============================================================================
// Output panel
// ============================================================================
function setOutputTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  ['prediction', 'explanation', 'health'].forEach(n => {
    const el = document.getElementById('output' + n.charAt(0).toUpperCase() + n.slice(1));
    if (el) el.style.display = n === name ? 'block' : 'none';
  });
  if (name === 'prediction') {
    document.getElementById('outputPrediction').style.display = 'block';
  }
}

function renderOutput(result, input) {
  if (!result) return;
  const pred = result.prediction;
  const className = (result.classes && typeof pred === 'number') ? result.classes[pred] : pred;
  document.getElementById('predictionValue').textContent = className !== undefined ? className : '—';
  document.getElementById('predictionLabel').textContent = 'input: ' + (input || '').slice(0, 60);
  const conf = result.confidence || 0;
  document.getElementById('confBar').style.width = (conf * 100) + '%';
  document.getElementById('confVal').textContent = (conf * 100).toFixed(1) + '%';

  // Metrics
  document.getElementById('mLatency').innerHTML = (result.latency_ms || 0).toFixed(1) + '<span class="unit">ms</span>';
  document.getElementById('mTps').textContent = (1000 / Math.max(1, result.latency_ms || 1)).toFixed(0);
  document.getElementById('mActive').textContent = STATE.model.n_params ? Math.floor(STATE.model.n_params * conf) + ' / ' + STATE.model.n_params : '—';
  document.getElementById('mSurprise').textContent = (Math.abs(0.5 - conf) * 2).toFixed(2);

  // Status bar
  STATE.tokensOut = String(className || '').length;
  document.getElementById('sTokens').textContent = STATE.tokensIn + ' in / ' + STATE.tokensOut + ' out';
  document.getElementById('sLatency').textContent = (result.latency_ms || 0).toFixed(1) + 'ms';
  document.getElementById('sConf').textContent = (conf * 100).toFixed(1) + '%';
  document.getElementById('sActive').textContent = (STATE.model.n_params ? Math.floor(STATE.model.n_params * conf) : '—') + ' / ' + (STATE.model.n_params || '—');

  // Explanation
  renderExplanation(result, input);
}

function renderExplanation(result, input) {
  const el = document.getElementById('explanationList');
  if (!STATE.layers.length) {
    el.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No weight data available.</div>';
    return;
  }
  // Build pseudo-feature-importance from first layer's per-row max weight
  const L0 = STATE.layers[0];
  const inN = L0.shape[0];
  const rows = [];
  for (let i = 0; i < Math.min(inN, 6); i++) {
    let max = 0;
    const start = i * L0.shape[1];
    for (let j = 0; j < L0.shape[1]; j++) {
      const v = Math.abs(L0.weights[start + j] || 0);
      if (v > max) max = v;
    }
    rows.push({name: 'feature_' + i, val: max});
  }
  rows.sort((a, b) => b.val - a.val);
  const maxVal = rows[0] ? rows[0].val : 1;
  el.innerHTML = rows.map(r =>
    `<div class="explanation-row">
      <span class="name">${r.name}</span>
      <span class="bar"><span class="fill" style="width:${(r.val / maxVal * 100)}%; background:${r.val > 0 ? getCss('--info') : getCss('--pink')}"></span></span>
      <span class="val">${r.val.toFixed(3)}</span>
    </div>`
  ).join('');
}

// ============================================================================
// Weight inspector
// ============================================================================
function setInspectorTab(name, el) {
  document.querySelectorAll('.inspector-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  ['Heatmap', 'Sliders', 'Sensitivity', 'History'].forEach(n => {
    const el = document.getElementById('inspector' + n);
    if (el) el.classList.toggle('active', n.toLowerCase() === name);
  });
  if (name === 'heatmap') drawHeatmap();
  if (name === 'sliders') drawSliders();
}

function drawHeatmap() {
  const container = document.getElementById('heatmapContainer');
  const L = STATE.layers[STATE.selectedLayer] || STATE.layers[0];
  if (!L) { container.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No layer selected</div>'; return; }
  const w = L.weights || [];
  const n = Math.min(w.length, 256);
  const cols = Math.ceil(Math.sqrt(n));
  const cells = [];
  for (let i = 0; i < n; i++) {
    const c = colorForWeight(w[i], L.min, L.max);
    cells.push(`<div class="heatmap-cell" style="background:${c}" onclick="selectWeight(${i})" title="${L.name}[${i}] = ${w[i] ? w[i].toFixed(4) : 0}"></div>`);
  }
  container.innerHTML = `<div style="font-size:11px;color:var(--text-dim);margin-bottom:6px">${L.name} — showing ${n} of ${L.n_params} weights</div><div class="heatmap" style="grid-template-columns:repeat(${cols}, 12px)">${cells.join('')}</div>`;
}

function selectWeight(idx) {
  STATE.selectedWeight = idx;
  setInspectorTab('sliders', document.querySelectorAll('.inspector-tab')[1]);
}

async function editWeight(layerIdx, flatIdx, newVal) {
  if (STATE.mode === 'live') {
    await API.post('/api/weights/edit', {layer_idx: layerIdx, flat_idx: flatIdx, new_value: newVal});
  }
  // Append to history list locally
  const list = document.getElementById('historyList');
  const now = new Date();
  const ts = now.toLocaleTimeString();
  const L = STATE.layers[layerIdx];
  if (L) {
    const oldVal = L.weights[flatIdx] || 0;
    L.weights[flatIdx] = newVal;
    list.innerHTML = `<div class="history-row"><span class="ts">${ts}</span><span class="name">${L.name}[${flatIdx}]</span><span class="old">${oldVal.toFixed(3)}</span><span class="new">${newVal.toFixed(3)}</span></div>` + list.innerHTML;
  }
  drawHeatmap();
  drawSliders();
  // Re-predict if there's a last input
  if (STATE.lastOutput && document.getElementById('chatInput')) {
    // Don't re-send; just notify user
  }
}

function drawSliders() {
  const container = document.getElementById('slidersContainer');
  if (!STATE.layers.length) {
    container.innerHTML = '<div style="color:var(--text-dim);font-size:11px">No weights</div>';
    return;
  }
  const L = STATE.layers[STATE.selectedLayer] || STATE.layers[0];
  // Pick top-K by absolute value
  const w = L.weights || [];
  const idxs = w.map((v, i) => ({v: Math.abs(v), i})).sort((a, b) => b.v - a.v).slice(0, 12);
  const range = Math.max(0.1, L.max - L.min);
  const min = L.min, max = L.max;
  container.innerHTML = `<div style="font-size:11px;color:var(--text-dim);margin-bottom:8px">Top-12 by |weight| in <b style="color:var(--accent)">${L.name}</b></div>` +
    idxs.map(({i}) => {
      const v = w[i] || 0;
      const minS = (min - 0).toFixed(3), maxS = (max).toFixed(3);
      return `<div class="slider-row">
        <span class="name">${L.name}[${i}]</span>
        <input type="range" min="${min}" max="${max}" step="${(range / 100).toFixed(4)}" value="${v}" oninput="document.getElementById('sv_${i}').textContent=parseFloat(this.value).toFixed(3)" onchange="editWeight(${STATE.selectedLayer}, ${i}, parseFloat(this.value))">
        <span class="val" id="sv_${i}">${v.toFixed(3)}</span>
      </div>`;
    }).join('');
}

async function resetWeights() {
  if (STATE.mode === 'live') {
    await API.post('/api/weights/reset', {});
  }
  document.getElementById('historyList').innerHTML = '<div style="color:var(--text-dim);font-size:11px">No weight edits yet.</div>';
  await loadLayers();
  drawHeatmap();
  drawSliders();
  alert('Weight edit history cleared. Reload the model to undo actual mutations.');
}

// ============================================================================
// Compare overlay
// ============================================================================
function openCompare() {
  document.getElementById('compareOverlay').classList.add('active');
}
function closeCompare() {
  document.getElementById('compareOverlay').classList.remove('active');
}
async function runCompare() {
  const a = document.getElementById('compareA').value;
  const b = document.getElementById('compareB').value;
  let result;
  if (STATE.mode === 'live') {
    result = await API.post('/api/compare', {a, b});
  } else {
    const pa = demoPredict(a), pb = demoPredict(b);
    result = {a: pa, b: pb, max_diff: Math.abs(pa.prediction - pb.prediction)};
  }
  document.getElementById('compareResult').style.display = 'grid';
  document.getElementById('predA').textContent = result.a.prediction;
  document.getElementById('predB').textContent = result.b.prediction;
  document.getElementById('compareDiff').style.display = 'block';
  document.getElementById('compareDiff').textContent = 'max diff: ' + (result.max_diff || 0).toFixed(4);
}

// ============================================================================
// Adversarial probe
// ============================================================================
async function runAdversarial() {
  const inp = document.getElementById('chatInput').value || '[5.1, 3.5, 1.4, 0.2]';
  appendChat('user', 'adversarial probe: ' + inp);
  let r;
  if (STATE.mode === 'live') {
    r = await API.post('/api/adversarial', {input: inp, epsilon: 0.05});
  } else {
    r = {n_trials: 20, n_flips: 4, flip_rate: 0.2, worst_diff: 0.31, epsilon: 0.05};
  }
  appendChat('pred', `Adversarial probe: ${r.n_flips}/${r.n_trials} predictions flipped (${(r.flip_rate * 100).toFixed(0)}%). Worst perturbation moved output by ${r.worst_diff.toFixed(3)}.`);
}

// ============================================================================
// Story mode
// ============================================================================
function toggleStoryMode() {
  STATE.storyStep = 0;
  document.getElementById('storyTour').classList.add('active');
  showStoryStep();
}
function showStoryStep() {
  document.getElementById('storyStep').textContent = STATE.storyStep + 1;
  document.getElementById('storyTotal').textContent = STATE.storySteps.length;
  document.getElementById('storyText').textContent = STATE.storySteps[STATE.storyStep];
  document.getElementById('storyNextBtn').textContent = STATE.storyStep === STATE.storySteps.length - 1 ? 'Finish' : 'Next';
}
function nextStoryStep() {
  if (STATE.storyStep < STATE.storySteps.length - 1) {
    STATE.storyStep++;
    showStoryStep();
  } else {
    closeStory();
  }
}
function prevStoryStep() {
  if (STATE.storyStep > 0) {
    STATE.storyStep--;
    showStoryStep();
  }
}
function closeStory() {
  document.getElementById('storyTour').classList.remove('active');
}

// ============================================================================
// Export story
// ============================================================================
async function exportStory() {
  let data;
  if (STATE.mode === 'live') {
    data = await API.get('/api/export');
  } else {
    data = {model: STATE.model, layers: STATE.layers, predictions: [], exported_at: Date.now() / 1000};
  }
  const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'qai_story_' + Date.now() + '.json';
  a.click();
  URL.revokeObjectURL(url);
}

// ============================================================================
// File drop
// ============================================================================
function setupFileDrop() {
  const drop = document.getElementById('fileDrop');
  drop.addEventListener('click', () => {
    const inp = document.createElement('input');
    inp.type = 'file';
    inp.onchange = () => {
      if (inp.files[0]) {
        const r = new FileReader();
        r.onload = () => {
          document.getElementById('chatInput').value = r.result.slice(0, 200);
          sendInput();
        };
        r.readAsText(inp.files[0]);
      }
    };
    inp.click();
  });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('dragover'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault();
    drop.classList.remove('dragover');
    if (e.dataTransfer.files[0]) {
      const r = new FileReader();
      r.onload = () => {
        document.getElementById('chatInput').value = r.result.slice(0, 200);
        sendInput();
      };
      r.readAsText(e.dataTransfer.files[0]);
    }
  });
}

// Redraw canvas whenever window resizes
window.addEventListener('resize', () => { resizeCanvas(); drawCanvas(); });
</script>
</body>
</html>
"""


# ============================================================================
# HTTP server
# ============================================================================

def _make_handler(backend: _WorkspaceBackend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass  # silence default logging

        def _json(self, code: int, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _html(self, code: int, body: str):
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                if self.path in ("/", "/index.html", "/workspace"):
                    return self._html(200, WORKSPACE_HTML)
                elif self.path == "/api/model":
                    return self._json(200, backend.metadata())
                elif self.path == "/api/layers":
                    return self._json(200, {"layers": backend.get_layers()})
                elif self.path.startswith("/api/weights/"):
                    idx = int(self.path.split("/")[-1])
                    return self._json(200, backend.get_weights(idx))
                elif self.path == "/api/health":
                    return self._json(200, backend.health())
                elif self.path == "/api/weights/history":
                    return self._json(200, {"history": backend.weight_history()})
                elif self.path == "/api/export":
                    return self._json(200, backend.export_story())
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-400:]})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length) if length else b"{}"
                body = json.loads(raw) if raw else {}
                if self.path == "/api/predict":
                    return self._json(200, backend.predict(body.get("input", "")))
                elif self.path == "/api/weights/edit":
                    return self._json(200, backend.edit_weight(
                        int(body.get("layer_idx", 0)),
                        int(body.get("flat_idx", 0)),
                        float(body.get("new_value", 0.0)),
                    ))
                elif self.path == "/api/weights/reset":
                    return self._json(200, backend.reset_weights())
                elif self.path == "/api/compare":
                    return self._json(200, backend.compare(body.get("a", ""), body.get("b", "")))
                elif self.path == "/api/adversarial":
                    return self._json(200, backend.adversarial_probe(
                        body.get("input", ""), float(body.get("epsilon", 0.05)),
                    ))
                elif self.path == "/api/pin":
                    return self._json(200, backend.pin_input(body.get("input", "")))
                else:
                    return self._json(404, {"error": "not found", "path": self.path})
            except Exception as e:
                return self._json(500, {"error": str(e), "trace": traceback.format_exc()[-400:]})

    return Handler


def _find_free_port(default: int) -> int:
    """Use the default port if free, otherwise pick any free port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


# ============================================================================
# Public API
# ============================================================================

def serve_workspace(model=None, port: int = 7788, open_browser: bool = True,
                    host: str = "127.0.0.1") -> None:
    """Serve the interactive Visual Workspace for ``model``.

    Blocking — open a browser, then visit http://127.0.0.1:{port}/.
    """
    backend = _WorkspaceBackend(model) if model is not None else None
    if backend is None:
        # DEMO mode: no model — use synthetic data
        class _SyntheticModel:
            technique_name = "DemoMLP"
            technique = None
        backend = _WorkspaceBackend(_SyntheticModel(), story_mode=True)

    actual_port = _find_free_port(port)
    handler = _make_handler(backend)
    server = ThreadingHTTPServer((host, actual_port), handler)

    url = f"http://{host}:{actual_port}/"
    print(f"[qai] Visual Workspace served at: {url}")
    print(f"[qai] Model: {backend.metadata().get('name', 'unknown')} "
          f"({backend.metadata().get('n_params', 0)} params)")
    print(f"[qai] Press Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[qai] Shutting down workspace server…")
        server.shutdown()


def generate_workspace(model=None, path: str = "qai_workspace.html") -> str:
    """Generate a static HTML file of the workspace for offline viewing.

    Without ``model``, the file works in DEMO mode (synthetic data).
    With a model, the file embeds model metadata + weights for offline
    viewing (no live predictions).
    """
    if model is not None:
        meta = _model_metadata(model)
        layers = _get_weights(model)
        # Embed as a JSON blob the page can read on boot
        bootstrap = (
            f"\n<script>window.__QAI_BOOTSTRAP__ = "
            f"{json.dumps({'mode': 'story', 'model': meta, 'layers': layers})};"
            f"</script>\n"
        )
        # Inject before </body>
        html_out = WORKSPACE_HTML.replace("</body>", bootstrap + "</body>")
    else:
        html_out = WORKSPACE_HTML

    with open(path, "w", encoding="utf-8") as f:
        f.write(html_out)
    abs_path = os.path.abspath(path)
    print(f"[qai] Workspace written to: {abs_path}")
    print(f"[qai] Open in any browser (no server needed): file://{abs_path}")
    return abs_path


# Short aliases
workspace = serve_workspace
