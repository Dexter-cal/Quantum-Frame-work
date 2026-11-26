"""
qai.profiler -- model performance profiler.

Measures latency, memory, parameter count, and (where possible) FLOPs
estimate for any qai model. Useful for:
  - Pre-deployment sizing ("will this fit on a Pi?")
  - Comparing model variants
  - Regression-detecting changes ("did the new layer make it slower?")

Usage:
    import qai
    m = qai.build(technique="neural_network", hidden_dim=64)
    m.train(X, y)
    profile = qai.Profiler(m).profile(X_sample)
    print(profile)              # formatted text
    profile.as_dict()           # raw metrics
"""
from __future__ import annotations
import time, gc, sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class Profile:
    """A model performance profile.

    Attributes:
        technique: model technique name
        n_params: total parameter count
        model_size_mb: estimated model size in RAM (MB)
        avg_latency_ms: mean prediction latency per sample (ms)
        p50_latency_ms: 50th-percentile latency (ms)
        p95_latency_ms: 95th-percentile latency (ms)
        p99_latency_ms: 99th-percentile latency (ms)
        throughput_per_s: predictions per second (1 / avg_latency)
        memory_peak_mb: peak memory during prediction (MB) -- best-effort
        flops_estimate: rough FLOPs estimate (where possible)
    """
    technique: str
    n_params: int = 0
    model_size_mb: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_per_s: float = 0.0
    memory_peak_mb: float = 0.0
    flops_estimate: Optional[int] = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "technique": self.technique,
            "n_params": self.n_params,
            "model_size_mb": round(self.model_size_mb, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "p50_latency_ms": round(self.p50_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "p99_latency_ms": round(self.p99_latency_ms, 3),
            "throughput_per_s": round(self.throughput_per_s, 1),
            "memory_peak_mb": round(self.memory_peak_mb, 3),
            "flops_estimate": self.flops_estimate,
            "notes": self.notes,
        }

    def __str__(self) -> str:
        lines = [
            f"=== Profile: {self.technique} ===",
            f"parameters:        {self.n_params:,}",
            f"model size:        {self.model_size_mb:.3f} MB",
            f"avg latency:       {self.avg_latency_ms:.3f} ms/sample",
            f"p50 latency:       {self.p50_latency_ms:.3f} ms",
            f"p95 latency:       {self.p95_latency_ms:.3f} ms",
            f"p99 latency:       {self.p99_latency_ms:.3f} ms",
            f"throughput:        {self.throughput_per_s:.1f} predictions/sec",
            f"peak memory:       {self.memory_peak_mb:.3f} MB",
        ]
        if self.flops_estimate is not None:
            lines.append(f"FLOPs estimate:    {self.flops_estimate:,}")
        if self.notes:
            lines.append("")
            lines.append("notes:")
            for n in self.notes:
                lines.append(f"  - {n}")
        return "\n".join(lines)


class Profiler:
    """Profile a qai model's inference performance.

    Args:
        model: a qai.Model (from qai.build(...)).
    """
    def __init__(self, model):
        self.model = model
        tech = getattr(model, "technique", None)
        self.technique_name = getattr(tech, "name", type(model).__name__) if tech else type(model).__name__

    def profile(self, X_sample, n_warmup: int = 3, n_runs: int = 20) -> Profile:
        """Run inference benchmarks and return a Profile.

        Args:
            X_sample: a sample input (single prediction worth of data).
            n_warmup: warmup runs (not timed).
            n_runs: timed runs.

        Returns:
            Profile dataclass.
        """
        X_arr = np.asarray(X_sample)
        # Warmup
        for _ in range(n_warmup):
            try:
                self.model.predict(X_arr)
            except Exception:
                pass

        # Timed runs
        latencies = []
        gc.collect()
        mem_before = self._get_process_mem_mb()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                self.model.predict(X_arr)
            except Exception:
                pass
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms
        mem_after = self._get_process_mem_mb()

        latencies = np.array(latencies)
        n_params = self._count_params()
        size_mb = self._estimate_size_mb()
        flops = self._estimate_flops(X_arr)

        return Profile(
            technique=self.technique_name,
            n_params=n_params,
            model_size_mb=size_mb,
            avg_latency_ms=float(np.mean(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_per_s=1000.0 / max(float(np.mean(latencies)), 1e-6),
            memory_peak_mb=max(mem_after - mem_before, 0.0),
            flops_estimate=flops,
            notes=self._notes(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _count_params(self) -> int:
        """Estimate parameter count for the model."""
        tech = getattr(self.model, "technique", None)
        if tech is None:
            return 0
        # PyTorch-backed
        if hasattr(tech, "net") and tech.net is not None:
            try:
                return sum(p.numel() for p in tech.net.parameters())
            except Exception:
                pass
        # sklearn-backed: count coef_ + intercept_
        n = 0
        if hasattr(tech, "model"):
            for attr in ("coef_", "intercept_", "feature_importances_",
                         "estimators_", "classes_", "cluster_centers_"):
                v = getattr(tech.model, attr, None)
                if v is not None and hasattr(v, "size"):
                    try:
                        n += int(v.size)
                    except Exception:
                        pass
                elif v is not None and hasattr(v, "__len__"):
                    try:
                        n += len(v)
                    except Exception:
                        pass
        # Q-learning Q-table
        if hasattr(tech, "agent") and hasattr(tech.agent, "q_table"):
            n += len(tech.agent.q_table)
        return n

    def _estimate_size_mb(self) -> float:
        """Estimate model size in MB."""
        n_params = self._count_params()
        # Assume float32 (4 bytes) for most models, float64 (8 bytes) for sklearn
        tech = getattr(self.model, "technique", None)
        if tech is not None and hasattr(tech, "model"):
            # sklearn -- usually float64
            return (n_params * 8) / (1024 * 1024)
        return (n_params * 4) / (1024 * 1024)

    def _estimate_flops(self, X_sample) -> Optional[int]:
        """Rough FLOPs estimate for a single forward pass."""
        tech = getattr(self.model, "technique", None)
        if tech is None:
            return None
        X_arr = np.atleast_2d(np.asarray(X_sample))
        n_features = X_arr.shape[-1] if X_arr.ndim > 0 else 1
        # PyTorch MLP: 2 * sum(input * output) per linear layer
        if hasattr(tech, "net") and tech.net is not None:
            try:
                import torch.nn as nn
                flops = 0
                for module in tech.net.modules():
                    if isinstance(module, nn.Linear):
                        # 2 * (in_features * out_features) for multiply-add
                        flops += 2 * module.in_features * module.out_features
                return flops
            except Exception:
                return None
        # sklearn linear models: 2 * n_features
        if hasattr(tech, "model") and hasattr(tech.model, "coef_"):
            try:
                return 2 * int(np.asarray(tech.model.coef_).size)
            except Exception:
                return None
        # Random forest: 2 * n_features * n_estimators * avg_depth
        if hasattr(tech, "model") and hasattr(tech.model, "estimators_"):
            try:
                n_trees = len(tech.model.estimators_)
                avg_depth = 10  # rough default
                return 2 * n_features * n_trees * avg_depth
            except Exception:
                return None
        return None

    def _get_process_mem_mb(self) -> float:
        """Best-effort process RSS in MB."""
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except Exception:
            return 0.0

    def _notes(self) -> list:
        notes = []
        tech = getattr(self.model, "technique", None)
        if tech is None:
            notes.append("model has no technique attribute; profiling is limited")
            return notes
        if not getattr(tech, "_trained", False):
            notes.append("model is not trained; predictions may be misleading")
        if hasattr(tech, "net"):
            try:
                import torch
                if torch.cuda.is_available():
                    notes.append("GPU available but CPU inference profiled")
            except ImportError:
                pass
        return notes


__all__ = ["Profiler", "Profile"]
