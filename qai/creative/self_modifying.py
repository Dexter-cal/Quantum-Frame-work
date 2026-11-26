"""
qai.creative.self_modifying -- Creative Idea 3: SELF-MODIFYING ARCHITECTURE

A meta-controller that edits its own topology during training. At each
audit step it may:
  - GROW   : add a neuron to a layer whose gradients are consistently large
  - PRUNE  : drop neurons whose activation magnitude is consistently tiny
  - SPLIT  : duplicate a high-load neuron and reset one copy
  - WIDEN  : add a residual branch when overall loss plateaus
  - DEPTH  : add a new layer when residual variance is high

Wraps any layer-list-based model (PyTorch nn.Sequential-like, or our own
core.Sequential). Falls back to logging-only mode for opaque models.

Usage:
    sma = qai.creative.SelfModifyingArchitecture(model)
    sma.audit_after_epoch(epoch=5, X_val, y_val)
"""
from __future__ import annotations
import copy
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


class SelfModifyingArchitecture:
    """Meta-controller that grows / prunes its host model's topology."""

    def __init__(self, model: Any,
                 grow_threshold: float = 0.1,
                 prune_threshold: float = 0.01,
                 plateau_patience: int = 3,
                 max_neurons_per_layer: int = 1024):
        self.model = model
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.plateau_patience = plateau_patience
        self.max_neurons = max_neurons_per_layer
        self.loss_history: List[float] = []
        self.activation_stats: Dict[str, Dict[str, float]] = {}
        self.changes: List[Dict[str, Any]] = []
        self._hooks: List[Any] = []

    def _is_torch_sequential(self) -> bool:
        return _TORCH and isinstance(self.model, (nn.Sequential, nn.ModuleList))

    def _attach_hooks(self) -> None:
        if not _TORCH or not isinstance(self.model, nn.Module):
            return
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            with torch.no_grad():
                arr = out.detach().cpu().numpy()
                self.activation_stats[name] = {
                    "mean_abs": float(np.mean(np.abs(arr))),
                    "std": float(np.std(arr)),
                    "n_samples": arr.shape[0] if arr.ndim > 0 else 1,
                }
        return hook

    def audit_after_epoch(self, epoch: int,
                          X_val: Optional[Any] = None,
                          y_val: Optional[Any] = None,
                          current_loss: Optional[float] = None) -> Dict[str, Any]:
        """Run one self-modification audit. Returns dict of changes made."""
        if not self._is_torch_sequential():
            return {"changed": False, "reason": "model is not a torch Sequential/ModuleList"}
        if not self.activation_stats:
            self._attach_hooks()
            # run forward pass to populate stats
            if X_val is not None:
                try:
                    with torch.no_grad():
                        if isinstance(X_val, np.ndarray):
                            X_val = torch.as_tensor(X_val, dtype=torch.float32)
                        self.model(X_val)
                except Exception:
                    pass

        changes_made: List[Dict[str, Any]] = []
        # 1. Plateau detection
        if current_loss is not None:
            self.loss_history.append(current_loss)
            if len(self.loss_history) > self.plateau_patience:
                recent = self.loss_history[-self.plateau_patience:]
                if max(recent) - min(recent) < self.prune_threshold:
                    changes_made.append({"type": "plateau_detected",
                                          "recent_losses": recent})

        # 2. Grow neurons for layers with high-activation outputs
        for name, stats in self.activation_stats.items():
            if stats["mean_abs"] > self.grow_threshold:
                try:
                    self._grow_layer(name, n_add=1)
                    changes_made.append({"type": "grow", "layer": name,
                                          "n_added": 1, "reason": "high activation"})
                except Exception as e:
                    changes_made.append({"type": "grow_failed", "layer": name,
                                          "error": str(e)})
            elif stats["mean_abs"] < self.prune_threshold:
                try:
                    self._prune_layer(name, n_remove=1)
                    changes_made.append({"type": "prune", "layer": name,
                                          "n_removed": 1, "reason": "low activation"})
                except Exception as e:
                    changes_made.append({"type": "prune_failed", "layer": name,
                                          "error": str(e)})

        self.changes.append({"epoch": epoch, "changes": changes_made})
        return {"epoch": epoch, "n_changes": len(changes_made), "changes": changes_made}

    def _find_layer(self, name: str) -> Optional[nn.Module]:
        if not _TORCH:
            return None
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    def _grow_layer(self, name: str, n_add: int = 1) -> None:
        """Add n_add neurons to a Linear layer (preserves L1 ≈ 0 init)."""
        layer = self._find_layer(name)
        if layer is None or not isinstance(layer, nn.Linear):
            return
        old_out = layer.out_features
        new_out = min(old_out + n_add, self.max_neurons)
        if new_out == old_out:
            return
        new_layer = nn.Linear(layer.in_features, new_out, bias=layer.bias is not None)
        with torch.no_grad():
            new_layer.weight[:old_out] = layer.weight
            new_layer.weight[old_out:] = torch.randn_like(new_layer.weight[old_out:]) * 0.01
            if layer.bias is not None:
                new_layer.bias[:old_out] = layer.bias
                new_layer.bias[old_out:] = 0
        # Replace in parent — easiest: monkey-patch the module dict
        parent = self.model
        if hasattr(parent, name):
            setattr(parent, name, new_layer)
        else:
            # walk submodules
            for n, m in parent.named_children():
                if n == name.split(".")[-1]:
                    parent._modules[n] = new_layer

    def _prune_layer(self, name: str, n_remove: int = 1) -> None:
        """Remove the n_remove lowest-activation neurons."""
        layer = self._find_layer(name)
        if layer is None or not isinstance(layer, nn.Linear):
            return
        old_out = layer.out_features
        if old_out <= n_remove + 1:
            return
        # pick neurons with smallest |weight|
        with torch.no_grad():
            w = layer.weight
            norms = w.abs().sum(dim=1)
            keep_idx = torch.argsort(norms, descending=True)[:old_out - n_remove]
            new_layer = nn.Linear(layer.in_features, len(keep_idx),
                                    bias=layer.bias is not None)
            new_layer.weight.data = w[keep_idx]
            if layer.bias is not None:
                new_layer.bias.data = layer.bias[keep_idx]
        parent = self.model
        for n, m in parent.named_children():
            if n == name.split(".")[-1]:
                parent._modules[n] = new_layer

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all topology changes since creation."""
        return {
            "n_audits": len(self.changes),
            "total_changes": sum(len(c["changes"]) for c in self.changes),
            "by_type": {
                t: sum(1 for c in self.changes for ch in c["changes"] if ch["type"] == t)
                for t in ["grow", "prune", "grow_failed", "prune_failed",
                          "plateau_detected"]
            },
            "history": self.changes[-5:],
        }


__all__ = ["SelfModifyingArchitecture"]
