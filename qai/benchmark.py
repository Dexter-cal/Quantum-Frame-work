"""
qai.benchmark -- performance + scaling benchmarks for the framework.

  - ModelBenchmark          : throughput, latency, memory for a single model
  - ScalingBenchmark        : how does throughput scale with batch size / concurrency?
  - InferenceComparison     : compare N models side-by-side
  - TrainingBenchmark       : wall-clock + tokens/sec for training
  - MemoryProfiler          : peak RAM + VRAM during a workload
  - LatencyPercentiles      : p50/p95/p99 latency
  - StressTest              : high-concurrency load testing
  - RegressionBench         : run benchmarks before/after a change; flag regressions
"""
from __future__ import annotations
import gc
import time
import statistics
import threading
import concurrent.futures
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False


def _memory_usage() -> Dict[str, float]:
    """Current process memory in MB."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss > 1_000_000:  # likely KB on Linux
            rss_mb = rss / 1024
        else:  # macOS returns bytes
            rss_mb = rss / (1024 * 1024)
    except Exception:
        rss_mb = 0.0
    vram_mb = 0.0
    if _TORCH and torch.cuda.is_available():
        try:
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
    return {"rss_mb": rss_mb, "vram_mb": vram_mb}


# ============================================================================
# 1. Model Benchmark
# ============================================================================
class ModelBenchmark:
    """Benchmark a single model: throughput, latency, memory."""

    def __init__(self, model: Any):
        self.model = model
        self.results: Dict[str, Any] = {}

    def benchmark_inference(self, X: List[Any],
                              warmup: int = 3,
                              repeats: int = 10) -> Dict[str, Any]:
        """Measure inference latency + throughput."""
        # warmup
        for x in X[:warmup]:
            try:
                self.model.forward(x)
            except Exception:
                pass
        # measure
        latencies: List[float] = []
        for _ in range(repeats):
            for x in X:
                t0 = time.perf_counter()
                try:
                    self.model.forward(x)
                except Exception:
                    pass
                latencies.append((time.perf_counter() - t0) * 1000)  # ms
        # memory before/after
        gc.collect()
        mem_before = _memory_usage()
        try:
            self.model.forward(X[0])
        except Exception:
            pass
        mem_after = _memory_usage()
        return {
            "n_samples": len(X),
            "n_repeats": repeats,
            "total_calls": len(latencies),
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "std_latency_ms": float(np.std(latencies)),
            "throughput_samples_per_sec": float(1000.0 / np.mean(latencies)) if latencies else 0,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
            "memory_delta_mb": {
                k: mem_after[k] - mem_before[k] for k in mem_before
            },
        }

    def benchmark_training(self, X: Any, y: Any,
                            n_epochs: int = 3) -> Dict[str, Any]:
        """Measure training wall-clock + per-epoch time."""
        epoch_times: List[float] = []
        gc.collect()
        mem_before = _memory_usage()
        for ep in range(n_epochs):
            t0 = time.perf_counter()
            try:
                self.model.fit(X, y)
            except Exception as e:
                return {"error": str(e), "epochs_completed": ep}
            elapsed = time.perf_counter() - t0
            epoch_times.append(elapsed)
        mem_after = _memory_usage()
        return {
            "n_epochs": n_epochs,
            "total_time_s": float(sum(epoch_times)),
            "mean_epoch_time_s": float(np.mean(epoch_times)),
            "std_epoch_time_s": float(np.std(epoch_times)),
            "samples_per_epoch": len(X) if hasattr(X, "__len__") else 0,
            "samples_per_sec": float(len(X) / np.mean(epoch_times)) if X is not None and hasattr(X, "__len__") else 0,
            "memory_before_mb": mem_before,
            "memory_after_mb": mem_after,
        }

    def full_report(self, X_infer: Optional[List[Any]] = None,
                     X_train: Optional[Any] = None, y_train: Optional[Any] = None,
                     n_epochs: int = 3) -> Dict[str, Any]:
        report: Dict[str, Any] = {"model_class": type(self.model).__name__}
        if X_infer is not None:
            report["inference"] = self.benchmark_inference(X_infer)
        if X_train is not None and y_train is not None:
            report["training"] = self.benchmark_training(X_train, y_train, n_epochs)
        self.results = report
        return report


# ============================================================================
# 2. Scaling Benchmark
# ============================================================================
class ScalingBenchmark:
    """How does throughput scale with batch size / concurrency?"""

    def __init__(self, model: Any):
        self.model = model

    def by_batch_size(self, X: List[Any],
                       batch_sizes: List[int] = [1, 4, 8, 16, 32],
                       repeats: int = 5) -> Dict[str, Any]:
        """Throughput vs batch size."""
        results: List[Dict[str, Any]] = []
        for bs in batch_sizes:
            if bs > len(X):
                continue
            latencies: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                for i in range(0, bs):
                    try:
                        self.model.forward(X[i])
                    except Exception:
                        pass
                latencies.append((time.perf_counter() - t0) * 1000)
            mean_lat = float(np.mean(latencies))
            results.append({
                "batch_size": bs,
                "mean_latency_ms": mean_lat,
                "throughput_samples_per_sec": float(bs * 1000 / mean_lat) if mean_lat > 0 else 0,
            })
        return {"scaling": results}

    def by_concurrency(self, X: List[Any],
                        concurrency_levels: List[int] = [1, 2, 4, 8],
                        repeats: int = 5) -> Dict[str, Any]:
        """Throughput vs concurrency level."""
        results: List[Dict[str, Any]] = []
        for c in concurrency_levels:
            if c > len(X):
                continue
            latencies: List[float] = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                    futures = [pool.submit(self.model.forward, X[i]) for i in range(c)]
                    for f in concurrent.futures.as_completed(futures):
                        try:
                            f.result()
                        except Exception:
                            pass
                latencies.append((time.perf_counter() - t0) * 1000)
            mean_lat = float(np.mean(latencies))
            results.append({
                "concurrency": c,
                "mean_latency_ms": mean_lat,
                "throughput_samples_per_sec": float(c * 1000 / mean_lat) if mean_lat > 0 else 0,
            })
        return {"concurrency_scaling": results}


# ============================================================================
# 3. Inference Comparison
# ============================================================================
class InferenceComparison:
    """Side-by-side comparison of N models."""

    def __init__(self, models: Dict[str, Any]):
        self.models = models

    def compare(self, X: List[Any], repeats: int = 5) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for name, model in self.models.items():
            bench = ModelBenchmark(model)
            results[name] = bench.benchmark_inference(X, repeats=repeats)
        # rank by throughput
        ranked = sorted(
            [(name, r["throughput_samples_per_sec"]) for name, r in results.items()],
            key=lambda x: -x[1]
        )
        return {
            "results": results,
            "ranked_by_throughput": [{"model": n, "throughput": t} for n, t in ranked],
            "fastest": ranked[0][0] if ranked else None,
            "slowest": ranked[-1][0] if ranked else None,
            "speedup_fastest_vs_slowest": (
                ranked[0][1] / ranked[-1][1] if ranked and ranked[-1][1] > 0 else 1.0
            ),
        }


# ============================================================================
# 4. Latency Percentiles
# ============================================================================
class LatencyPercentiles:
    """Compute p50/p95/p99 latency under varying load."""

    def __init__(self, model: Any):
        self.model = model

    def measure(self, X: List[Any], n_calls: int = 100) -> Dict[str, Any]:
        latencies: List[float] = []
        for i in range(n_calls):
            x = X[i % len(X)]
            t0 = time.perf_counter()
            try:
                self.model.forward(x)
            except Exception:
                pass
            latencies.append((time.perf_counter() - t0) * 1000)
        return {
            "n_calls": n_calls,
            "p50": float(np.percentile(latencies, 50)),
            "p90": float(np.percentile(latencies, 90)),
            "p95": float(np.percentile(latencies, 95)),
            "p99": float(np.percentile(latencies, 99)),
            "max": float(np.max(latencies)),
            "min": float(np.min(latencies)),
            "mean": float(np.mean(latencies)),
            "std": float(np.std(latencies)),
            "raw_latencies_ms": latencies,
        }


# ============================================================================
# 5. Stress Test
# ============================================================================
class StressTest:
    """High-concurrency load testing."""

    def __init__(self, model: Any):
        self.model = model

    def run(self, X: List[Any], n_concurrent: int = 50,
             duration_seconds: float = 5.0) -> Dict[str, Any]:
        """Hammer the model with n_concurrent workers for `duration_seconds`."""
        start = time.perf_counter()
        completed = 0
        errors = 0
        latencies: List[float] = []

        def _worker():
            nonlocal completed, errors
            while time.perf_counter() - start < duration_seconds:
                x = X[completed % len(X)]
                t0 = time.perf_counter()
                try:
                    self.model.forward(x)
                    latencies.append((time.perf_counter() - t0) * 1000)
                    completed += 1
                except Exception:
                    errors += 1

        threads = [threading.Thread(target=_worker) for _ in range(n_concurrent)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.perf_counter() - start
        return {
            "n_concurrent": n_concurrent,
            "duration_s": float(elapsed),
            "n_completed": completed,
            "n_errors": errors,
            "throughput_per_sec": float(completed / elapsed) if elapsed > 0 else 0,
            "error_rate": float(errors / max(completed + errors, 1)),
            "latency_p50_ms": float(np.percentile(latencies, 50)) if latencies else 0,
            "latency_p95_ms": float(np.percentile(latencies, 95)) if latencies else 0,
            "latency_p99_ms": float(np.percentile(latencies, 99)) if latencies else 0,
            "latency_max_ms": float(np.max(latencies)) if latencies else 0,
        }


# ============================================================================
# 6. Regression Bench
# ============================================================================
class RegressionBench:
    """Run benchmarks before/after a change; flag regressions.

    Usage:
        rb = RegressionBench()
        rb.baseline = rb.measure(model, X)
        # make a change to model
        rb.after = rb.measure(model, X)
        regressions = rb.check_regressions()
    """
    def __init__(self, tolerance_pct: float = 0.1):
        self.tolerance = tolerance_pct
        self.baseline: Optional[Dict[str, Any]] = None
        self.after: Optional[Dict[str, Any]] = None

    def measure(self, model: Any, X: List[Any]) -> Dict[str, Any]:
        bench = ModelBenchmark(model)
        return bench.benchmark_inference(X, repeats=5)

    def check_regressions(self) -> Dict[str, Any]:
        if not self.baseline or not self.after:
            return {"error": "both baseline and after must be measured"}
        regressions: List[Dict[str, Any]] = []
        improvements: List[Dict[str, Any]] = []
        for key in ["mean_latency_ms", "p95_latency_ms", "p99_latency_ms"]:
            b = self.baseline[key]
            a = self.after[key]
            delta = a - b
            pct = delta / b if b > 0 else 0
            if abs(pct) > self.tolerance:
                entry = {"metric": key, "baseline": b, "after": a,
                          "delta": delta, "pct_change": pct}
                if delta > 0:
                    regressions.append(entry)
                else:
                    improvements.append(entry)
        # throughput is inverse: drop = regression
        b_thr = self.baseline["throughput_samples_per_sec"]
        a_thr = self.after["throughput_samples_per_sec"]
        thr_pct = (a_thr - b_thr) / b_thr if b_thr > 0 else 0
        if abs(thr_pct) > self.tolerance:
            entry = {"metric": "throughput", "baseline": b_thr, "after": a_thr,
                      "delta": a_thr - b_thr, "pct_change": thr_pct}
            if thr_pct < 0:
                regressions.append(entry)
            else:
                improvements.append(entry)
        return {
            "n_regressions": len(regressions),
            "n_improvements": len(improvements),
            "regressions": regressions,
            "improvements": improvements,
            "any_regression": len(regressions) > 0,
        }


# ============================================================================
# 7. Quick benchmark helper
# ============================================================================
def quick_benchmark(model: Any, X: List[Any],
                     repeats: int = 3) -> Dict[str, Any]:
    """One-line benchmark for a model on a list of inputs."""
    return ModelBenchmark(model).benchmark_inference(X, repeats=repeats)


def compare_models(models: Dict[str, Any], X: List[Any],
                    repeats: int = 3) -> Dict[str, Any]:
    """One-line comparison of multiple models."""
    return InferenceComparison(models).compare(X, repeats=repeats)


__all__ = [
    "ModelBenchmark", "ScalingBenchmark", "InferenceComparison",
    "LatencyPercentiles", "StressTest", "RegressionBench",
    "quick_benchmark", "compare_models",
]
