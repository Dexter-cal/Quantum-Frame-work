"""
qai.model_tools — creative model utilities and introspection tools.

Provides:
  - suggest_hyperparameters() — auto-suggest good hyperparams per technique
  - estimate_training_cost() — estimate time/memory before training
  - ModelProfiler — profile inference latency, memory, FLOPs per layer
  - count_parameters() — total/trainable/non-trainable parameter counts
  - model_summary() — full architecture summary (like torchsummary)
  - model_to_device() — move a qai model to a specific device (CPU/GPU/MPS)
  - compare_models() — side-by-side comparison of two models
  - auto_configure() — auto-pick best params based on data + hardware

Usage:
    import qai
    m = qai.build(technique="transformer", vocab_size=1000)

    # Count parameters
    info = qai.count_parameters(m)
    print(f"{info['total']} total, {info['trainable']} trainable")

    # Model summary
    print(qai.model_summary(m))

    # Suggest hyperparameters
    suggestions = qai.suggest_hyperparameters("transformer", n_samples=10000)
    print(suggestions)

    # Estimate training cost
    cost = qai.estimate_training_cost(m, n_samples=10000, epochs=10)
    print(f"Estimated: {cost['estimated_time_min']:.0f} min, {cost['estimated_memory_mb']:.0f} MB")

    # Profile
    profiler = qai.ModelProfiler(m)
    results = profiler.profile(input_shape=(1, 8))
    print(results)

    # Move to device
    qai.model_to_device(m, "cuda:0")
"""
from __future__ import annotations
import os, time, math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch, torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False


# =====================================================================
# Parameter counting
# =====================================================================
def count_parameters(model) -> Dict[str, int]:
    """Count total, trainable, and non-trainable parameters.

    Returns:
        Dict with 'total', 'trainable', 'non_trainable', 'total_mb'
    """
    if not _TORCH:
        return {"total": 0, "trainable": 0, "non_trainable": 0, "total_mb": 0.0}
    tech = model.technique if hasattr(model, "technique") else model
    total, trainable = 0, 0
    for attr in vars(tech).values():
        if isinstance(attr, (nn.Parameter, torch.Tensor)):
            n = attr.nelement()
            total += n
            if attr.requires_grad:
                trainable += n
        elif isinstance(attr, nn.Module):
            for p in attr.parameters():
                total += p.nelement()
                if p.requires_grad:
                    trainable += p.nelement()
        elif isinstance(attr, (list, tuple)):
            for x in attr:
                if isinstance(x, nn.Module):
                    for p in x.parameters():
                        total += p.nelement()
                        if p.requires_grad:
                            trainable += p.nelement()
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
        "total_mb": total * 4 / (1024**2),  # assume float32
    }


# =====================================================================
# Model summary (like torchsummary)
# =====================================================================
def model_summary(model, input_shape: Optional[Tuple] = None) -> str:
    """Generate a human-readable model architecture summary.

    Shows: technique name, family, parameter count, layer breakdown,
    memory footprint, and (if torch) per-layer parameter counts.

    Args:
        model: a qai.Model
        input_shape: optional input shape for FLOPs estimation

    Returns:
        Formatted string
    """
    info = count_parameters(model)
    tech = model.technique if hasattr(model, "technique") else model
    name = getattr(tech, "name", "?")
    family = getattr(tech, "family", "?")
    trained = "trained" if tech.is_trained() else "untrained"

    lines = [
        "=" * 70,
        f"Model Summary: {name} ({family}) — {trained}",
        "=" * 70,
        f"Total parameters:     {info['total']:>12,}",
        f"Trainable parameters: {info['trainable']:>12,}",
        f"Non-trainable:        {info['non_trainable']:>12,}",
        f"Memory (float32):     {info['total_mb']:>12.2f} MB",
    ]

    # Per-layer breakdown (for torch models)
    if _TORCH:
        lines.append("-" * 70)
        lines.append(f"{'Layer':<40} {'Parameters':>12} {'Shape'}")
        lines.append("-" * 70)
        for attr_name in vars(tech):
            attr = getattr(tech, attr_name)
            if isinstance(attr, nn.Module):
                n_params = sum(p.nelement() for p in attr.parameters())
                if n_params > 0:
                    lines.append(f"  {attr_name:<38} {n_params:>12,}")
            elif isinstance(attr, nn.Parameter):
                lines.append(f"  {attr_name:<38} {attr.nelement():>12,} {list(attr.shape)}")
            elif isinstance(attr, (list, tuple)):
                for i, item in enumerate(attr):
                    if isinstance(item, nn.Module):
                        n_params = sum(p.nelement() for p in item.parameters())
                        if n_params > 0:
                            lines.append(f"  {attr_name}[{i}]:{'':<32} {n_params:>12,}")

    # Params info
    params = tech.list_params() if hasattr(tech, "list_params") else []
    if params:
        lines.append("-" * 70)
        lines.append(f"Key parameters ({len(params)} total):")
        for p in params[:10]:
            lines.append(f"  {p['name']}: {p['current']} (default: {p['default']})")
        if len(params) > 10:
            lines.append(f"  ... and {len(params) - 10} more")

    lines.append("=" * 70)
    return "\n".join(lines)


# =====================================================================
# Hyperparameter suggestions
# =====================================================================
# Curated suggestions based on ML literature + best practices
_HYPERPARAM_SUGGESTIONS = {
    "transformer": {
        "small": {"d_model": 128, "n_heads": 4, "n_layers": 2, "max_seq_len": 256,
                   "dropout": 0.1, "lr": 3e-4, "batch_size": 32, "epochs": 10},
        "medium": {"d_model": 256, "n_heads": 8, "n_layers": 6, "max_seq_len": 512,
                    "dropout": 0.1, "lr": 3e-4, "batch_size": 32, "epochs": 20},
        "large": {"d_model": 512, "n_heads": 8, "n_layers": 12, "max_seq_len": 1024,
                   "dropout": 0.1, "lr": 2e-4, "batch_size": 16, "epochs": 50},
    },
    "cnn": {
        "small": {"conv_channels": [16, 32], "fc_hidden": 32, "lr": 1e-3, "batch_size": 32},
        "medium": {"conv_channels": [32, 64, 128], "fc_hidden": 128, "lr": 1e-3, "batch_size": 64},
        "large": {"conv_channels": [64, 128, 256, 512], "fc_hidden": 512, "lr": 1e-3, "batch_size": 128},
    },
    "random_forest": {
        "small": {"n_estimators": 50, "max_depth": 10},
        "medium": {"n_estimators": 200, "max_depth": 20},
        "large": {"n_estimators": 500, "max_depth": None},
    },
    "gradient_boosting": {
        "small": {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
        "medium": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 5},
        "large": {"n_estimators": 500, "learning_rate": 0.01, "max_depth": 7},
    },
    "kmeans": {
        "small": {"n_clusters": 3, "n_init": 3},
        "medium": {"n_clusters": 8, "n_init": 5},
        "large": {"n_clusters": 20, "n_init": 10},
    },
    "svm": {
        "small": {"C": 1.0, "kernel": "rbf"},
        "medium": {"C": 10.0, "kernel": "rbf"},
        "large": {"C": 100.0, "kernel": "rbf"},
    },
    "regression": {
        "any": {"fit_intercept": True},
    },
    "classifier": {
        "any": {"max_iter": 1000},
    },
}


def suggest_hyperparameters(technique_name: str, n_samples: int = 1000,
                             n_features: int = 10, task: str = "classification"
                             ) -> Dict[str, Any]:
    """Suggest good hyperparameters for a technique based on data size.

    Uses curated recommendations from ML literature + best practices.

    Args:
        technique_name: e.g. "transformer", "random_forest"
        n_samples: number of training samples
        n_features: number of input features
        task: "classification" or "regression"

    Returns:
        Dict with suggested params + reasoning
    """
    # Determine scale
    if n_samples < 1000:
        scale = "small"
    elif n_samples < 100000:
        scale = "medium"
    else:
        scale = "large"

    suggestions = _HYPERPARAM_SUGGESTIONS.get(technique_name, {})
    if scale in suggestions:
        params = suggestions[scale]
    elif "any" in suggestions:
        params = suggestions["any"]
    else:
        params = {}

    # Add reasoning
    reasoning = []
    if technique_name == "transformer":
        reasoning.append(f"d_model={params.get('d_model')} for {scale} dataset ({n_samples} samples)")
        reasoning.append(f"n_layers={params.get('n_layers')} — more layers for larger data")
        reasoning.append(f"lr={params.get('lr')} — standard for AdamW + cosine schedule")
        reasoning.append(f"batch_size={params.get('batch_size')} — adjust to fit GPU memory")
    elif technique_name == "random_forest":
        reasoning.append(f"n_estimators={params.get('n_estimators')} — more trees = better but slower")
        reasoning.append(f"max_depth={params.get('max_depth')} — None means unlimited (may overfit)")
    elif technique_name == "gradient_boosting":
        reasoning.append(f"learning_rate={params.get('learning_rate')} — lower = slower but better")
        reasoning.append(f"n_estimators={params.get('n_estimators')} — more rounds for lower LR")
    else:
        reasoning.append(f"Suggested for {technique_name} at {scale} scale")

    return {
        "technique": technique_name,
        "scale": scale,
        "n_samples": n_samples,
        "suggested_params": params,
        "reasoning": reasoning,
        "note": "These are starting points — always validate with cross-validation",
    }


# =====================================================================
# Training cost estimator
# =====================================================================
def estimate_training_cost(model, n_samples: int, epochs: int = 10,
                            batch_size: int = 32, device: str = "cpu"
                            ) -> Dict[str, Any]:
    """Estimate training time and memory before starting.

    Uses parameter count + data size + empirical throughput estimates.

    Args:
        model: a qai.Model
        n_samples: number of training samples
        epochs: number of epochs
        batch_size: batch size
        device: "cpu" or "cuda"

    Returns:
        Dict with estimated time, memory, and FLOPs
    """
    info = count_parameters(model)
    n_params = info["total"]

    # Estimate memory
    # Model weights: n_params * 4 bytes (float32)
    # Gradients: same as weights
    # Optimizer state (AdamW): 2x weights (momentum + variance)
    # Activations: ~2x batch_size * model_size_per_sample
    model_memory_mb = n_params * 4 / (1024**2)
    gradient_memory_mb = model_memory_mb
    optimizer_memory_mb = 2 * model_memory_mb  # AdamW
    activation_memory_mb = batch_size * model_memory_mb * 0.1  # rough estimate
    total_memory_mb = model_memory_mb + gradient_memory_mb + optimizer_memory_mb + activation_memory_mb

    # Estimate time (very rough)
    # CPU: ~1M params * 1 batch in ~0.1s
    # GPU: ~10x faster
    steps_per_epoch = math.ceil(n_samples / batch_size)
    total_steps = steps_per_epoch * epochs
    if device == "cuda":
        time_per_step = n_params / 1e8  # GPU: ~100M params/s
    else:
        time_per_step = n_params / 1e7  # CPU: ~10M params/s
    estimated_time_s = total_steps * time_per_step
    estimated_time_min = estimated_time_s / 60

    # Estimate FLOPs (rough: 2 * n_params per forward pass)
    flops_per_forward = 2 * n_params
    total_flops = flops_per_forward * total_steps * 2  # forward + backward

    return {
        "n_parameters": n_params,
        "n_trainable": info["trainable"],
        "n_samples": n_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "estimated_time_min": estimated_time_min,
        "estimated_time_str": _format_time(estimated_time_s),
        "estimated_memory_mb": total_memory_mb,
        "estimated_memory_str": f"{total_memory_mb:.1f} MB",
        "total_flops": total_flops,
        "total_flops_str": _format_flops(total_flops),
        "device": device,
        "warning": "Estimates are rough — actual time depends on architecture, data loading, etc." if n_params > 0 else "Model has 0 parameters",
    }


def _format_time(seconds: float) -> str:
    """Format seconds into human-readable time."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def _format_flops(flops: float) -> str:
    """Format FLOPs into human-readable string."""
    if flops < 1e6:
        return f"{flops:.0f} FLOPs"
    elif flops < 1e9:
        return f"{flops/1e6:.1f} MFLOPs"
    elif flops < 1e12:
        return f"{flops/1e9:.1f} GFLOPs"
    elif flops < 1e15:
        return f"{flops/1e12:.1f} TFLOPs"
    else:
        return f"{flops/1e15:.1f} PFLOPs"


# =====================================================================
# Model Profiler
# =====================================================================
class ModelProfiler:
    """Profile a model's inference performance.

    Measures:
      - Latency (mean, p50, p95, p99)
      - Throughput (queries per second)
      - Memory usage (before/after)
      - Parameter count per layer

    Args:
        model: a qai.Model
        warmup: number of warmup iterations before profiling
    """
    def __init__(self, model, warmup: int = 5):
        self.model = model
        self.warmup = warmup

    def profile(self, input_data=None, input_shape=None, n_runs: int = 100) -> Dict[str, Any]:
        """Run profiling.

        Args:
            input_data: optional input data (if None, generates random from input_shape)
            input_shape: shape of input (if input_data is None)
            n_runs: number of profiling iterations

        Returns:
            Dict with latency, throughput, memory, parameter info
        """
        # Generate input if not provided
        if input_data is None:
            if input_shape is None:
                input_shape = (1, 4)
            input_data = np.random.randn(*input_shape).astype(np.float32)
        elif not isinstance(input_data, np.ndarray):
            input_data = np.asarray(input_data, dtype=np.float32)

        # Warmup
        for _ in range(self.warmup):
            try:
                self.model.predict(input_data[0] if input_data.ndim > 1 else input_data)
            except Exception:
                pass

        # Profile
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                self.model.predict(input_data[0] if input_data.ndim > 1 else input_data)
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)
        param_info = count_parameters(self.model)

        # Memory
        mem_before = 0
        mem_after = 0
        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated() / (1024**2)
            try:
                self.model.predict(input_data[0] if input_data.ndim > 1 else input_data)
            except Exception:
                pass
            mem_after = torch.cuda.max_memory_allocated() / (1024**2)

        return {
            "latency_ms_mean": float(latencies.mean()),
            "latency_ms_p50": float(np.percentile(latencies, 50)),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "latency_ms_p99": float(np.percentile(latencies, 99)),
            "throughput_qps": float(1000 / max(latencies.mean(), 0.001)),
            "n_parameters": param_info["total"],
            "n_trainable": param_info["trainable"],
            "model_size_mb": param_info["total_mb"],
            "gpu_memory_before_mb": mem_before,
            "gpu_memory_after_mb": mem_after,
            "n_runs": n_runs,
        }


# =====================================================================
# Move model to device
# =====================================================================
def model_to_device(model, device: str):
    """Move a qai model to a specific device (CPU/GPU/MPS).

    Args:
        model: a qai.Model
        device: "cpu", "cuda:0", "cuda:1", "mps"
    """
    if not _TORCH:
        return model
    tech = model.technique
    target = torch.device(device)
    for attr_name in vars(tech):
        attr = getattr(tech, attr_name)
        if isinstance(attr, (nn.Module, nn.Parameter, torch.Tensor)):
            attr.data = attr.data.to(target) if isinstance(attr, (nn.Parameter, torch.Tensor)) else attr.to(target)
        elif isinstance(attr, (list, tuple)):
            for i, item in enumerate(attr):
                if isinstance(item, nn.Module):
                    attr[i] = item.to(target)
    # Store device
    tech._device = device
    if hasattr(tech, "device"):
        tech.device = target
    return model


# =====================================================================
# Compare models
# =====================================================================
def compare_models(model_a, model_b, labels: Optional[Tuple[str, str]] = None) -> str:
    """Side-by-side comparison of two models.

    Args:
        model_a, model_b: two qai.Model objects
        labels: optional (label_a, label_b) tuple

    Returns:
        Formatted comparison string
    """
    info_a = count_parameters(model_a)
    info_b = count_parameters(model_b)
    tech_a = model_a.technique
    tech_b = model_b.technique
    label_a = labels[0] if labels else tech_a.name
    label_b = labels[1] if labels else tech_b.name

    lines = [
        "=" * 70,
        f"{'Metric':<25} {label_a:>20} {label_b:>20}",
        "=" * 70,
        f"{'Technique':<25} {tech_a.name:>20} {tech_b.name:>20}",
        f"{'Family':<25} {tech_a.family:>20} {tech_b.family:>20}",
        f"{'Trained':<25} {str(tech_a.is_trained()):>20} {str(tech_b.is_trained()):>20}",
        f"{'Total params':<25} {info_a['total']:>20,} {info_b['total']:>20,}",
        f"{'Trainable params':<25} {info_a['trainable']:>20,} {info_b['trainable']:>20,}",
        f"{'Model size (MB)':<25} {info_a['total_mb']:>20.2f} {info_b['total_mb']:>20.2f}",
        "=" * 70,
    ]
    return "\n".join(lines)


# =====================================================================
# Auto-configure model based on data + hardware
# =====================================================================
def auto_configure(technique_name: str, n_samples: int = 1000,
                    n_features: int = 10, task: str = "classification",
                    device: Optional[str] = None) -> Dict[str, Any]:
    """Auto-pick the best technique + params based on data + hardware.

    Combines:
      - suggest_hyperparameters() for technique-specific params
      - recommend_device_for_technique() for hardware
      - estimate_training_cost() for cost projection

    Args:
        technique_name: the technique to configure
        n_samples: number of training samples
        n_features: number of features
        task: "classification" or "regression"
        device: optional device override

    Returns:
        Dict with recommended params, device, and cost estimate
    """
    import qai
    # 1. Suggest hyperparameters
    hp = suggest_hyperparameters(technique_name, n_samples, n_features, task)
    # 2. Recommend device
    if device is None:
        try:
            rec = qai.recommend_device_for_technique(technique_name)
            device = rec["recommended_device"]
        except Exception:
            device = "cpu"
    # 3. Estimate cost
    try:
        m = qai.build(technique=technique_name, **hp["suggested_params"])
        cost = estimate_training_cost(m, n_samples, epochs=10, device=device)
    except Exception:
        cost = {"estimated_time_min": 0, "estimated_memory_mb": 0}
    return {
        "technique": technique_name,
        "suggested_params": hp["suggested_params"],
        "reasoning": hp["reasoning"],
        "recommended_device": device,
        "estimated_cost": cost,
    }


__all__ = ["count_parameters", "model_summary", "suggest_hyperparameters",
           "estimate_training_cost", "ModelProfiler", "model_to_device",
           "compare_models", "auto_configure"]
