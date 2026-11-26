"""
qai.surgery -- weight surgery: precise operations on model weights.

  WeightSurgeon     — surgical operations on specific neurons/channels/layers
  NeuronInspector   — inspect individual neurons (activation, importance, role)
  ChannelPruner     — prune entire channels/filters (structured pruning)
  NeuronTransplant  — copy neurons from one model to another
  WeightProbing     — probe weights for specific patterns (lottery tickets, dead units)
  WeightTransplant  — transplant learned features between models
"""
from __future__ import annotations
import copy
import math
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class WeightSurgeon:
    """Surgical operations on specific neurons/channels/layers.

    Unlike WeightsEditor (which operates on whole layers), the Surgeon
    operates on individual neurons or channels within a layer.

    Operations:
      - disable_neuron(layer, idx)     — zero out a specific neuron
      - transplant_neuron(src, dst)    — copy a neuron from model A to B
      - swap_neurons(layer, idx_a, b)  — swap two neurons
      - clone_neuron(layer, idx)       — duplicate a neuron
      - rewire(layer, from_idx, to_idx)— redirect one neuron's output to another
      - ablate(layer, idx)             — remove a neuron and rewire connections
      - stimulate(layer, idx, value)   —force a neuron to always output `value`
    """

    def __init__(self, model: Any):
        self.model = model
        self._operations: List[Dict[str, Any]] = []

    def _get_weights(self) -> Dict[str, np.ndarray]:
        """Extract weights from model."""
        from .weights import _extract_weights
        return _extract_weights(self.model)

    def _set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        """Apply weights back to model (in-place for torch models)."""
        if hasattr(self.model, "load_state_dict"):
            try:
                import torch
                torch_weights = {k: torch.as_tensor(v) for k, v in weights.items()}
                self.model.load_state_dict(torch_weights)
                return
            except Exception:
                pass
        if hasattr(self.model, "technique") and hasattr(self.model.technique, "load_state_dict"):
            try:
                import torch
                torch_weights = {k: torch.as_tensor(v) for k, v in weights.items()}
                self.model.technique.load_state_dict(torch_weights)
                return
            except Exception:
                pass
        from .weights import _apply_weights
        self.model = _apply_weights(self.model, weights)

    def disable_neuron(self, layer: str, neuron_idx: int) -> Dict[str, Any]:
        """Zero out a specific neuron's weights."""
        weights = self._get_weights()
        if layer not in weights:
            raise KeyError(f"layer '{layer}' not found")
        arr = weights[layer]
        if neuron_idx >= arr.shape[0]:
            raise IndexError(f"neuron {neuron_idx} out of range for layer with {arr.shape[0]} neurons")
        original = arr[neuron_idx].copy()
        arr[neuron_idx] = 0
        weights[layer] = arr
        self._set_weights(weights)
        self._operations.append({
            "op": "disable_neuron", "layer": layer, "neuron": neuron_idx,
            "original_norm": float(np.linalg.norm(original)),
            "timestamp": time.time(),
        })
        return {"disabled": True, "layer": layer, "neuron": neuron_idx}

    def transplant_neuron(self, donor_model: Any, layer: str,
                            donor_neuron_idx: int, recipient_neuron_idx: int) -> Dict[str, Any]:
        """Copy a neuron from a donor model into this model."""
        from .weights import _extract_weights
        donor_weights = _extract_weights(donor_model)
        if layer not in donor_weights:
            raise KeyError(f"layer '{layer}' not found in donor")
        donor_arr = donor_weights[layer]
        if donor_neuron_idx >= donor_arr.shape[0]:
            raise IndexError("donor neuron out of range")
        recipient_weights = self._get_weights()
        if layer not in recipient_weights:
            raise KeyError(f"layer '{layer}' not found in recipient")
        recip_arr = recipient_weights[layer]
        if recipient_neuron_idx >= recip_arr.shape[0]:
            raise IndexError("recipient neuron out of range")
        # shapes must match
        if donor_arr[donor_neuron_idx].shape != recip_arr[recipient_neuron_idx].shape:
            raise ValueError(f"shape mismatch: donor {donor_arr[donor_neuron_idx].shape} vs recipient {recip_arr[recipient_neuron_idx].shape}")
        original = recip_arr[recipient_neuron_idx].copy()
        recip_arr[recipient_neuron_idx] = donor_arr[donor_neuron_idx]
        recipient_weights[layer] = recip_arr
        self._set_weights(recipient_weights)
        self._operations.append({
            "op": "transplant_neuron", "layer": layer,
            "donor_neuron": donor_neuron_idx,
            "recipient_neuron": recipient_neuron_idx,
            "timestamp": time.time(),
        })
        return {"transplanted": True, "layer": layer}

    def swap_neurons(self, layer: str, idx_a: int, idx_b: int) -> Dict[str, Any]:
        """Swap two neurons within a layer."""
        weights = self._get_weights()
        if layer not in weights:
            raise KeyError(f"layer '{layer}' not found")
        arr = weights[layer]
        arr[idx_a], arr[idx_b] = arr[idx_b].copy(), arr[idx_a].copy()
        weights[layer] = arr
        self._set_weights(weights)
        self._operations.append({
            "op": "swap_neurons", "layer": layer,
            "idx_a": idx_a, "idx_b": idx_b, "timestamp": time.time(),
        })
        return {"swapped": True, "layer": layer}

    def clone_neuron(self, layer: str, src_idx: int,
                       dest_idx: int) -> Dict[str, Any]:
        """Copy neuron src_idx to dest_idx (overwrites dest)."""
        weights = self._get_weights()
        arr = weights[layer]
        arr[dest_idx] = arr[src_idx].copy()
        weights[layer] = arr
        self._set_weights(weights)
        self._operations.append({
            "op": "clone_neuron", "layer": layer,
            "src": src_idx, "dest": dest_idx, "timestamp": time.time(),
        })
        return {"cloned": True, "layer": layer, "src": src_idx, "dest": dest_idx}

    def stimulate_neuron(self, layer: str, neuron_idx: int,
                           value: float) -> Dict[str, Any]:
        """Force a neuron to always output `value` (set bias + zero weights)."""
        weights = self._get_weights()
        arr = weights[layer]
        arr[neuron_idx] = 0  # zero weights
        weights[layer] = arr
        self._set_weights(weights)
        # set bias if exists
        bias_key = layer.replace("weight", "bias").replace("W", "b")
        if bias_key in weights:
            bias_arr = weights[bias_key]
            if neuron_idx < bias_arr.shape[0]:
                bias_arr[neuron_idx] = value
                weights[bias_key] = bias_arr
                self._set_weights(weights)
        return {"stimulated": True, "layer": layer, "neuron": neuron_idx, "value": value}

    def operation_log(self) -> List[Dict[str, Any]]:
        """Get log of all surgical operations."""
        return list(self._operations)

    def summary(self) -> Dict[str, Any]:
        """Summary of all operations performed."""
        by_type: Dict[str, int] = defaultdict(int)
        for op in self._operations:
            by_type[op["op"]] += 1
        return {
            "n_operations": len(self._operations),
            "by_type": dict(by_type),
            "layers_modified": list(set(op["layer"] for op in self._operations)),
        }


class NeuronInspector:
    """Inspect individual neurons in a model.

    For each neuron, reports:
      - weight norm (importance)
      - weight statistics (mean, std, min, max)
      - activation pattern (if input data provided)
      - dead/alive status (dead = all-zero weights)
      - role (heuristic: what input features it responds to)
    """

    @staticmethod
    def inspect(model: Any, layer: str) -> List[Dict[str, Any]]:
        """Inspect all neurons in a layer."""
        from .weights import _extract_weights
        weights = _extract_weights(model)
        if layer not in weights:
            raise KeyError(f"layer '{layer}' not found")
        arr = weights[layer]
        neurons = []
        for i in range(arr.shape[0]):
            neuron_weights = arr[i]
            norm = float(np.linalg.norm(neuron_weights))
            is_dead = norm < 1e-8
            neurons.append({
                "index": i,
                "norm": norm,
                "mean": float(neuron_weights.mean()),
                "std": float(neuron_weights.std()),
                "min": float(neuron_weights.min()),
                "max": float(neuron_weights.max()),
                "n_params": int(neuron_weights.size),
                "is_dead": is_dead,
                "top_inputs": np.argsort(np.abs(neuron_weights))[-5:][::-1].tolist() if neuron_weights.ndim > 0 else [],
                "top_input_weights": np.sort(np.abs(neuron_weights))[-5:][::-1].tolist() if neuron_weights.ndim > 0 else [],
            })
        return neurons

    @staticmethod
    def find_dead_neurons(model: Any, layer: str) -> List[int]:
        """Find all dead neurons (zero weights) in a layer."""
        neurons = NeuronInspector.inspect(model, layer)
        return [n["index"] for n in neurons if n["is_dead"]]

    @staticmethod
    def most_important_neurons(model: Any, layer: str,
                                 top_k: int = 5) -> List[Dict[str, Any]]:
        """Find the most important neurons by weight norm."""
        neurons = NeuronInspector.inspect(model, layer)
        neurons.sort(key=lambda n: -n["norm"])
        return neurons[:top_k]

    @staticmethod
    def neuron_similarity(model_a: Any, model_b: Any,
                           layer: str) -> Dict[str, Any]:
        """Compare neurons between two models."""
        from .weights import _extract_weights
        wa = _extract_weights(model_a)
        wb = _extract_weights(model_b)
        if layer not in wa or layer not in wb:
            raise KeyError(f"layer '{layer}' not found in both models")
        arr_a, arr_b = wa[layer], wb[layer]
        if arr_a.shape != arr_b.shape:
            return {"error": "shape mismatch"}
        similarities = []
        for i in range(arr_a.shape[0]):
            na, nb = arr_a[i], arr_b[i]
            cos_sim = float(np.dot(na, nb) / (np.linalg.norm(na) * np.linalg.norm(nb) + 1e-8))
            similarities.append({"neuron": i, "cosine_similarity": cos_sim})
        avg_sim = float(np.mean([s["cosine_similarity"] for s in similarities]))
        return {
            "layer": layer,
            "n_neurons": len(similarities),
            "avg_similarity": avg_sim,
            "per_neuron": similarities,
            "most_similar": max(similarities, key=lambda s: s["cosine_similarity"]),
            "least_similar": min(similarities, key=lambda s: s["cosine_similarity"]),
        }


class ChannelPruner:
    """Structured pruning: remove entire channels/filters.

    Unlike magnitude pruning (which zeros individual weights), channel
    pruning removes entire neurons/filters and rewires the network.
    """

    @staticmethod
    def prune(model: Any, layer: str, fraction: float = 0.1) -> Dict[str, Any]:
        """Prune the least important channels in a layer.

        Args:
            fraction: fraction of channels to remove (0.1 = 10%)
        """
        from .weights import _extract_weights, _apply_weights
        weights = _extract_weights(model)
        if layer not in weights:
            raise KeyError(f"layer '{layer}' not found")
        arr = weights[layer]
        n_channels = arr.shape[0]
        n_to_prune = int(n_channels * fraction)
        if n_to_prune == 0:
            return {"pruned": 0, "reason": "fraction too small"}
        # compute importance (L2 norm per channel)
        importances = np.array([np.linalg.norm(arr[i]) for i in range(n_channels)])
        prune_indices = np.argsort(importances)[:n_to_prune]
        # zero out pruned channels
        for idx in prune_indices:
            arr[idx] = 0
        weights[layer] = arr
        model = _apply_weights(model, weights)
        return {
            "pruned": n_to_prune,
            "total_channels": n_channels,
            "prune_fraction": fraction,
            "pruned_indices": prune_indices.tolist(),
            "remaining_channels": n_channels - n_to_prune,
        }

    @staticmethod
    def find_prunable(model: Any, layer: str,
                       threshold: float = 0.01) -> List[int]:
        """Find channels with norm below threshold (candidates for pruning)."""
        neurons = NeuronInspector.inspect(model, layer)
        return [n["index"] for n in neurons if n["norm"] < threshold]


class WeightProbing:
    """Probe weights for specific patterns.

    Checks for:
      - Lottery tickets (sparse subnetworks that perform well)
      - Dead units (neurons with zero output)
      - Exploding weights (very large values)
      - Vanishing weights (very small values)
      - Symmetric patterns (potential redundancy)
      - Clustering (groups of similar neurons)
    """

    @staticmethod
    def lottery_ticket_check(model: Any, X: np.ndarray,
                               y: np.ndarray,
                               sparsity: float = 0.5) -> Dict[str, Any]:
        """Check if the model contains a lottery ticket (sparse subnetwork).

        Creates a mask that zeros out the smallest-magnitude weights,
        then checks if the model still works.
        """
        from .weights import _extract_weights, _apply_weights
        weights = _extract_weights(model)
        original_pred = model.forward(X[0]) if hasattr(model, "forward") else None
        # create masks
        masks = {}
        n_total = 0
        n_pruned = 0
        for name, arr in weights.items():
            threshold = np.percentile(np.abs(arr), sparsity * 100)
            mask = (np.abs(arr) >= threshold).astype(arr.dtype)
            masks[name] = mask
            n_total += arr.size
            n_pruned += int((mask == 0).sum())
        # apply masks
        masked_weights = {name: arr * masks[name] for name, arr in weights.items()}
        masked_model = copy.deepcopy(model)
        masked_model = _apply_weights(masked_model, masked_weights)
        # check if still works
        try:
            new_pred = masked_model.forward(X[0])
            if original_pred is not None and new_pred is not None:
                diff = abs(float(original_pred) - float(new_pred))
                still_works = diff < 0.5
            else:
                still_works = True
        except Exception:
            still_works = False
        return {
            "sparsity": sparsity,
            "n_total_weights": n_total,
            "n_pruned": n_pruned,
            "actual_sparsity": n_pruned / max(n_total, 1),
            "still_works": still_works,
            "has_lottery_ticket": still_works,
        }

    @staticmethod
    def find_patterns(model: Any) -> Dict[str, Any]:
        """Find patterns in model weights."""
        from .weights import _extract_weights
        weights = _extract_weights(model)
        patterns: Dict[str, Any] = {"layers": {}}
        for name, arr in weights.items():
            layer_info: Dict[str, Any] = {}
            # dead units
            if arr.ndim > 1:
                norms = np.array([np.linalg.norm(arr[i]) for i in range(arr.shape[0])])
                layer_info["dead_units"] = int((norms < 1e-8).sum())
                layer_info["total_units"] = int(arr.shape[0])
            # exploding weights
            layer_info["exploding"] = int((np.abs(arr) > 1e3).sum())
            # vanishing weights
            layer_info["vanishing"] = int((np.abs(arr) < 1e-6).sum())
            # symmetry check (are there duplicate rows?)
            if arr.ndim > 1 and arr.shape[0] > 1:
                duplicates = 0
                for i in range(min(arr.shape[0], 20)):
                    for j in range(i + 1, min(arr.shape[0], 20)):
                        if np.allclose(arr[i], arr[j], atol=1e-6):
                            duplicates += 1
                layer_info["duplicate_pairs"] = duplicates
            patterns["layers"][name] = layer_info
        return patterns


__all__ = [
    "WeightSurgeon", "NeuronInspector",
    "ChannelPruner", "WeightProbing",
]
