"""
qai.budget — resource budget management for ML workloads.

Manages RAM, GPU VRAM, CPU, and disk budgets so you can:
  - Set a global RAM budget (e.g., 8GB) and the system refuses to load
    models that would exceed it
  - Load multiple models simultaneously within budget
  - Automatically pick the smallest model variant that fits
  - Swap models between RAM and disk when budget is tight
  - Train within a time budget (stop after N seconds)
  - Run inference within a cost budget (max tokens, max API calls)

Usage:
    budget = qai.ResourceBudget(ram_mb=4096, gpu_mb=8192, time_s=3600)
    budget.check_model(my_model)  # → fits or not
    budget.load(model, "model_a")  # loads if fits, raises if not
    budget.load(model2, "model_b") # loads second model if budget allows
    budget.status()  # current usage
    budget.evict("model_a")  # free up RAM

    # Auto-select model variant that fits budget
    model = budget.best_fit([model_small, model_medium, model_large])
"""
from __future__ import annotations
import gc
import os
import sys
import time
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. ResourceBudget — the main budget manager
# ============================================================================
class ResourceBudget:
    """Resource budget manager for RAM, GPU, CPU, disk, time, and cost.

    Tracks how much of each resource is currently in use by qai models.
    Refuses operations that would exceed the budget.

    Args:
        ram_mb: max RAM in MB (None = unlimited)
        gpu_mb: max GPU VRAM in MB (None = unlimited)
        cpu_cores: max CPU cores (None = unlimited)
        disk_mb: max disk space in MB (None = unlimited)
        time_s: max wall-clock time in seconds (None = unlimited)
        cost_usd: max cost in USD (for API calls) (None = unlimited)
    """

    def __init__(self,
                 ram_mb: Optional[float] = None,
                 gpu_mb: Optional[float] = None,
                 cpu_cores: Optional[int] = None,
                 disk_mb: Optional[float] = None,
                 time_s: Optional[float] = None,
                 cost_usd: Optional[float] = None):
        self.limits = {
            "ram_mb": ram_mb,
            "gpu_mb": gpu_mb,
            "cpu_cores": cpu_cores,
            "disk_mb": disk_mb,
            "time_s": time_s,
            "cost_usd": cost_usd,
        }
        self.usage = {
            "ram_mb": 0.0,
            "gpu_mb": 0.0,
            "cpu_cores": 0,
            "disk_mb": 0.0,
            "time_s": 0.0,
            "cost_usd": 0.0,
        }
        self._loaded_models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._start_time = time.time()
        self._lock = threading.Lock()
        self._violations: List[Dict[str, Any]] = []

    # === Model size estimation ===
    @staticmethod
    def estimate_model_size(model: Any) -> Dict[str, float]:
        """Estimate the RAM and GPU memory a model will use."""
        ram_mb = 0.0
        gpu_mb = 0.0
        n_params = 0
        # try state_dict
        if hasattr(model, "state_dict"):
            try:
                for v in model.state_dict().values():
                    arr = np.asarray(v)
                    n_params += arr.size
                    ram_mb += arr.nbytes / (1024 * 1024)
            except Exception:
                pass
        elif hasattr(model, "technique") and hasattr(model.technique, "state_dict"):
            try:
                for v in model.technique.state_dict().values():
                    arr = np.asarray(v)
                    n_params += arr.size
                    ram_mb += arr.nbytes / (1024 * 1024)
            except Exception:
                pass
        else:
            # fallback: scan for numpy arrays
            targets = [model]
            if hasattr(model, "technique"):
                targets.append(model.technique)
            for target in targets:
                for name in dir(target):
                    if name.startswith("_"):
                        continue
                    try:
                        val = getattr(target, name)
                        if isinstance(val, np.ndarray):
                            n_params += val.size
                            ram_mb += val.nbytes / (1024 * 1024)
                    except Exception:
                        pass
        # add overhead (Python object, gradients, optimizer state ≈ 3x weights)
        ram_mb = ram_mb * 3 + 50  # 50MB base overhead
        # GPU usage (if model is on GPU)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mb = float(torch.cuda.memory_allocated()) / (1024 * 1024)
        except Exception:
            pass
        return {
            "ram_mb": ram_mb,
            "gpu_mb": gpu_mb,
            "n_params": n_params,
            "n_params_human": f"{n_params / 1e6:.1f}M" if n_params > 1e6 else f"{n_params / 1e3:.1f}K",
        }

    # === Budget checking ===
    def check_model(self, model: Any) -> Dict[str, Any]:
        """Check if a model fits within the current budget."""
        size = self.estimate_model_size(model)
        fits_ram = (self.limits["ram_mb"] is None or
                     self.usage["ram_mb"] + size["ram_mb"] <= self.limits["ram_mb"])
        fits_gpu = (self.limits["gpu_mb"] is None or
                     self.usage["gpu_mb"] + size["gpu_mb"] <= self.limits["gpu_mb"])
        fits = fits_ram and fits_gpu
        return {
            "fits": fits,
            "fits_ram": fits_ram,
            "fits_gpu": fits_gpu,
            "model_size": size,
            "current_usage": dict(self.usage),
            "remaining_ram_mb": (self.limits["ram_mb"] - self.usage["ram_mb"])
                                if self.limits["ram_mb"] else None,
            "remaining_gpu_mb": (self.limits["gpu_mb"] - self.usage["gpu_mb"])
                                if self.limits["gpu_mb"] else None,
        }

    def can_load(self, model: Any) -> bool:
        """Quick check: can we load this model?"""
        return self.check_model(model)["fits"]

    # === Model loading within budget ===
    def load(self, model: Any, name: str,
             force: bool = False) -> Dict[str, Any]:
        """Load a model into RAM within budget.

        Args:
            model: the model to load
            name: a name to reference this model later
            force: if True, evict other models to make room
        """
        with self._lock:
            check = self.check_model(model)
            if not check["fits"] and not force:
                self._violations.append({
                    "type": "load_exceeded",
                    "name": name,
                    "model_size": check["model_size"],
                    "remaining": check["remaining_ram_mb"],
                    "timestamp": time.time(),
                })
                return {
                    "loaded": False,
                    "reason": "budget exceeded",
                    "details": check,
                }
            if not check["fits"] and force:
                # evict models until we fit
                while not self.check_model(model)["fits"] and self._loaded_models:
                    evicted = self.evict_oldest()
                    if not evicted:
                        break
            # load
            size = check["model_size"]
            self._loaded_models[name] = {
                "model": model,
                "ram_mb": size["ram_mb"],
                "gpu_mb": size["gpu_mb"],
                "loaded_at": time.time(),
                "n_accesses": 0,
            }
            self.usage["ram_mb"] += size["ram_mb"]
            self.usage["gpu_mb"] += size["gpu_mb"]
            return {
                "loaded": True,
                "name": name,
                "ram_used_mb": size["ram_mb"],
                "gpu_used_mb": size["gpu_mb"],
                "total_ram_mb": self.usage["ram_mb"],
                "total_gpu_mb": self.usage["gpu_mb"],
            }

    def unload(self, name: str) -> Dict[str, Any]:
        """Unload a model from RAM."""
        with self._lock:
            if name not in self._loaded_models:
                return {"unloaded": False, "reason": "not loaded"}
            entry = self._loaded_models.pop(name)
            self.usage["ram_mb"] -= entry["ram_mb"]
            self.usage["gpu_mb"] -= entry["gpu_mb"]
            gc.collect()
            return {
                "unloaded": True,
                "name": name,
                "freed_ram_mb": entry["ram_mb"],
                "freed_gpu_mb": entry["gpu_mb"],
            }

    def evict_oldest(self) -> Optional[str]:
        """Evict the oldest loaded model (LRU)."""
        if not self._loaded_models:
            return None
        name = next(iter(self._loaded_models))
        self.unload(name)
        return name

    def evict_largest(self) -> Optional[str]:
        """Evict the largest loaded model."""
        if not self._loaded_models:
            return None
        name = max(self._loaded_models, key=lambda k: self._loaded_models[k]["ram_mb"])
        self.unload(name)
        return name

    def get(self, name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        if name in self._loaded_models:
            self._loaded_models[name]["n_accesses"] += 1
            return self._loaded_models[name]["model"]
        return None

    def list_loaded(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        return [
            {
                "name": name,
                "ram_mb": info["ram_mb"],
                "gpu_mb": info["gpu_mb"],
                "loaded_at": info["loaded_at"],
                "n_accesses": info["n_accesses"],
                "age_s": time.time() - info["loaded_at"],
            }
            for name, info in self._loaded_models.items()
        ]

    # === Auto-selection ===
    def best_fit(self, models: List[Any]) -> Tuple[int, Any]:
        """Pick the largest model that fits within the remaining budget."""
        best_idx = -1
        best_model = None
        best_size = 0
        for i, model in enumerate(models):
            check = self.check_model(model)
            if check["fits"]:
                size = check["model_size"]["ram_mb"]
                if size > best_size:
                    best_size = size
                    best_idx = i
                    best_model = model
        if best_idx == -1:
            # none fit — pick the smallest
            sizes = [(i, self.estimate_model_size(m)["ram_mb"]) for i, m in enumerate(models)]
            sizes.sort(key=lambda x: x[1])
            best_idx = sizes[0][0]
            best_model = models[best_idx]
        return best_idx, best_model

    # === Time budget ===
    def time_remaining(self) -> Optional[float]:
        """How much time is left in the budget?"""
        if self.limits["time_s"] is None:
            return None
        elapsed = time.time() - self._start_time
        return max(0, self.limits["time_s"] - elapsed)

    def time_exceeded(self) -> bool:
        """Has the time budget been exceeded?"""
        if self.limits["time_s"] is None:
            return False
        return self.time_remaining() <= 0

    def record_cost(self, cost_usd: float) -> Dict[str, Any]:
        """Record an API cost."""
        self.usage["cost_usd"] += cost_usd
        if self.limits["cost_usd"] is not None and self.usage["cost_usd"] > self.limits["cost_usd"]:
            self._violations.append({
                "type": "cost_exceeded",
                "total_cost": self.usage["cost_usd"],
                "limit": self.limits["cost_usd"],
                "timestamp": time.time(),
            })
            return {"exceeded": True, "total_cost": self.usage["cost_usd"]}
        return {"exceeded": False, "total_cost": self.usage["cost_usd"]}

    # === Status ===
    def status(self) -> Dict[str, Any]:
        """Full status report."""
        return {
            "limits": self.limits,
            "usage": self.usage,
            "n_loaded_models": len(self._loaded_models),
            "loaded_models": self.list_loaded(),
            "ram_remaining_mb": (self.limits["ram_mb"] - self.usage["ram_mb"])
                                if self.limits["ram_mb"] else None,
            "gpu_remaining_mb": (self.limits["gpu_mb"] - self.usage["gpu_mb"])
                                if self.limits["gpu_mb"] else None,
            "time_remaining_s": self.time_remaining(),
            "cost_remaining_usd": (self.limits["cost_usd"] - self.usage["cost_usd"])
                                   if self.limits["cost_usd"] else None,
            "n_violations": len(self._violations),
            "violations": self._violations[-5:],
        }

    def summary(self) -> str:
        """Human-readable summary."""
        s = self.status()
        lines = ["📊 Resource Budget Status"]
        lines.append(f"  RAM: {s['usage']['ram_mb']:.0f}MB / {s['limits']['ram_mb'] or '∞'}MB")
        lines.append(f"  GPU: {s['usage']['gpu_mb']:.0f}MB / {s['limits']['gpu_mb'] or '∞'}MB")
        lines.append(f"  Models loaded: {s['n_loaded_models']}")
        if s["time_remaining_s"] is not None:
            lines.append(f"  Time remaining: {s['time_remaining_s']:.0f}s")
        if s["cost_remaining_usd"] is not None:
            lines.append(f"  Cost remaining: ${s['cost_remaining_usd']:.2f}")
        if s["n_violations"] > 0:
            lines.append(f"  ⚠️ Violations: {s['n_violations']}")
        return "\n".join(lines)


# ============================================================================
# 2. ModelSwapper — swap models between RAM and disk
# ============================================================================
class ModelSwapper:
    """Swap models between RAM and disk based on access patterns.

    When RAM is tight, least-recently-used models are swapped to disk.
    When accessed again, they're swapped back in (possibly evicting another).

    Usage:
        swapper = ModelSwapper(ram_budget_mb=2048, swap_dir="./swap")
        swapper.add("model_a", model_a)
        swapper.add("model_b", model_b)
        model = swapper.get("model_a")  # auto-swaps back if needed
    """

    def __init__(self, ram_budget_mb: float = 2048,
                 swap_dir: str = "/tmp/qai_swap"):
        self.ram_budget = ram_budget_mb
        self.swap_dir = swap_dir
        os.makedirs(swap_dir, exist_ok=True)
        self._in_ram: OrderedDict[str, Any] = OrderedDict()
        self._on_disk: Dict[str, str] = {}  # name → file path
        self._sizes: Dict[str, float] = {}
        self._current_ram_mb = 0.0

    def add(self, name: str, model: Any) -> None:
        """Add a model to the swapper."""
        size = ResourceBudget.estimate_model_size(model)["ram_mb"]
        self._sizes[name] = size
        # try to load into RAM
        if self._current_ram_mb + size <= self.ram_budget:
            self._in_ram[name] = model
            self._current_ram_mb += size
        else:
            # swap to disk
            self._swap_to_disk(name, model)

    def get(self, name: str) -> Any:
        """Get a model (swap back from disk if needed)."""
        if name in self._in_ram:
            # move to end (most recently used)
            self._in_ram.move_to_end(name)
            return self._in_ram[name]
        if name in self._on_disk:
            # swap back from disk
            return self._swap_from_disk(name)
        raise KeyError(f"model '{name}' not found")

    def _swap_to_disk(self, name: str, model: Any) -> None:
        """Swap a model to disk."""
        import pickle
        path = os.path.join(self.swap_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(model, f)
        self._on_disk[name] = path
        if name in self._in_ram:
            self._current_ram_mb -= self._sizes.get(name, 0)
            del self._in_ram[name]

    def _swap_from_disk(self, name: str) -> Any:
        """Swap a model back from disk."""
        import pickle
        path = self._on_disk[name]
        with open(path, "rb") as f:
            model = pickle.load(f)
        size = self._sizes.get(name, 0)
        # make room if needed
        while self._current_ram_mb + size > self.ram_budget and self._in_ram:
            oldest_name = next(iter(self._in_ram))
            oldest_model = self._in_ram.pop(oldest_name)
            self._current_ram_mb -= self._sizes.get(oldest_name, 0)
            self._swap_to_disk(oldest_name, oldest_model)
        self._in_ram[name] = model
        self._current_ram_mb += size
        del self._on_disk[name]
        return model

    def status(self) -> Dict[str, Any]:
        return {
            "ram_budget_mb": self.ram_budget,
            "current_ram_mb": self._current_ram_mb,
            "n_in_ram": len(self._in_ram),
            "n_on_disk": len(self._on_disk),
            "in_ram": list(self._in_ram.keys()),
            "on_disk": list(self._on_disk.keys()),
        }


# ============================================================================
# 3. TimeBudget — run operations within a time limit
# ============================================================================
class TimeBudget:
    """Run operations within a time budget.

    Two complementary usage patterns:

    **Pattern A — manual polling** (cooperative). The function being limited
    must call :meth:`exceeded` or :meth:`should_stop` itself::

        budget = TimeBudget(seconds=60)
        for epoch in range(100):
            if budget.exceeded():
                break
            train_epoch()

    **Pattern B — automatic interruption** (preemptive, POSIX-only). Use
    :meth:`run` or the context-manager interface to actually interrupt a
    function that runs past the budget — including infinite loops::

        budget = TimeBudget(seconds=0.5)
        try:
            budget.run(lambda: time.sleep(2.0))   # raises TimeoutError
        except TimeoutError:
            print("interrupted after ~0.5s, not 2.0s")

        # Or as a context manager:
        with TimeBudget(seconds=1.0) as tb:
            tb.run(lambda: expensive_call())

    On POSIX (Linux, macOS) interruption uses ``SIGALRM``. On Windows or when
    called from a non-main thread (where signals can't be delivered) it falls
    back to a watchdog thread that calls ``PyThreadState_SetAsyncExc`` to
    raise ``TimeoutError`` in the worker — best effort, may not interrupt
    C-extension calls like ``time.sleep`` on Windows.
    """

    def __init__(self, seconds: Optional[float] = None):
        self.limit = seconds
        self.start = time.time()
        # For watchdog-thread fallback (Windows / non-main thread)
        self._watchdog: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._target_thread_id: Optional[int] = None
        self._armed: bool = False

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining(self) -> Optional[float]:
        if self.limit is None:
            return None
        return max(0, self.limit - self.elapsed())

    def exceeded(self) -> bool:
        if self.limit is None:
            return False
        return self.elapsed() >= self.limit

    def should_stop(self) -> bool:
        """Check if we should stop (with 5% margin for cleanup)."""
        if self.limit is None:
            return False
        return self.elapsed() >= self.limit * 0.95

    def reset(self) -> None:
        self.start = time.time()
        self._disarm()

    # ------------------------------------------------------------------
    # Preemptive interruption — POSIX SIGALRM + watchdog-thread fallback
    # ------------------------------------------------------------------
    def _arm_signal(self) -> bool:
        """Arm SIGALRM in the current (main) thread. Returns True if armed."""
        if self.limit is None:
            return False
        try:
            import signal
            # signal.alarm only works in the main thread
            signal.signal(signal.SIGALRM, self._on_alarm)
            # setitimer supports sub-second resolution; alarm() is integer-only
            signal.setitimer(signal.ITIMER_REAL, float(self.limit))
            self._armed = True
            return True
        except (ValueError, OSError, ImportError):
            # ValueError: not in main thread
            # OSError: platform doesn't support SIGALRM (Windows)
            return False

    def _disarm_signal(self) -> None:
        if not self._armed:
            return
        try:
            import signal
            signal.setitimer(signal.ITIMER_REAL, 0)
        except (ValueError, OSError, ImportError):
            pass
        self._armed = False

    @staticmethod
    def _on_alarm(signum, frame):
        raise TimeoutError(
            "[WHAT] TimeBudget exceeded — function interrupted\n"
            "[WHY] A TimeBudget limit was set and the running function "
            "exceeded it. The SIGALRM handler raised to abort it.\n"
            "[FIX] Either raise the TimeBudget(seconds=...) limit, or make "
            "the function more efficient so it finishes within budget"
        )

    def _arm_watchdog(self) -> None:
        """Best-effort watchdog thread for non-main-thread / Windows use."""
        if self.limit is None:
            return
        self._stop_event = threading.Event()
        # Capture the *current* thread id so the watchdog can raise in it
        self._target_thread_id = threading.current_thread().ident
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog.start()

    def _watchdog_loop(self) -> None:
        assert self._stop_event is not None and self.limit is not None
        # Wait until the budget is exhausted, then raise TimeoutError in
        # the target thread via PyThreadState_SetAsyncExc.
        if self._stop_event.wait(self.limit):
            return  # stop_event was set — disarm
        self._async_raise(self._target_thread_id, TimeoutError(
            "[WHAT] TimeBudget exceeded — function interrupted by watchdog\n"
            "[WHY] A TimeBudget limit was set and the running function "
            "exceeded it. The watchdog thread raised to abort it.\n"
            "[FIX] Either raise the TimeBudget(seconds=...) limit, or make "
            "the function more efficient so it finishes within budget"
        ))

    @staticmethod
    def _async_raise(thread_id: Optional[int], exc: BaseException) -> None:
        """Raise `exc` in the thread with id `thread_id` (best effort)."""
        if thread_id is None:
            return
        import ctypes
        # PyThreadState_SetAsyncExc is CPython-only and best-effort — it
        # may not interrupt a thread that's in a C call (e.g. time.sleep
        # on Windows). It's the only option when SIGALRM isn't available.
        if not hasattr(ctypes, "pythonapi") or not hasattr(ctypes.pythonapi, "PyThreadState_SetAsyncExc"):
            return
        exc_addr = ctypes.cast(ctypes.py_object(exc), ctypes.c_void_p).value
        if exc_addr is None:
            return
        # Signature: PyThreadState_SetAsyncExc(long thread_id, PyObject* exc)
        ctypes.pythonapi.PyThreadState_SetAsyncExc.restype = ctypes.c_long
        ctypes.pythonapi.PyThreadState_SetAsyncExc.argtypes = (
            ctypes.c_ulong, ctypes.py_object)
        try:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(int(thread_id), exc)
        except Exception:
            pass

    def _disarm_watchdog(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self._watchdog = None
        self._stop_event = None
        self._target_thread_id = None

    def _disarm(self) -> None:
        self._disarm_signal()
        self._disarm_watchdog()

    def run(self, fn: Callable, *args, **kwargs):
        """Run `fn(*args, **kwargs)` and interrupt it if it exceeds the budget.

        On POSIX main thread, uses SIGALRM for accurate interruption (works
        even inside `time.sleep`, NumPy calls, and most C extensions).
        Otherwise falls back to a watchdog thread that raises TimeoutError
        in the worker via PyThreadState_SetAsyncExc (best effort — may not
        interrupt blocking C calls on Windows).

        Raises
        ------
        TimeoutError
            If `fn` runs past the budget.
        """
        if self.limit is None:
            return fn(*args, **kwargs)

        # Reset start time so the budget counts from this call
        self.start = time.time()
        armed_signal = self._arm_signal()
        if not armed_signal:
            self._arm_watchdog()
        try:
            return fn(*args, **kwargs)
        finally:
            self._disarm()

    def __enter__(self) -> "TimeBudget":
        if self.limit is not None:
            self.start = time.time()
            if not self._arm_signal():
                self._arm_watchdog()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._disarm()
        # Don't swallow TimeoutError — let it propagate so the caller knows
        # the budget was hit. (Use try/except TimeBudget(...) as tb: ... if
        # you want to handle it.)
        return False


# ============================================================================
# 4. CostTracker — track API costs within a budget
# ============================================================================
class CostTracker:
    """Track API costs (OpenAI, Anthropic, etc.) within a budget.

    Usage:
        tracker = CostTracker(monthly_budget_usd=100.0)
        tracker.record("openai", "gpt-4", tokens=1500, cost=0.045)
        tracker.record("anthropic", "claude-3", tokens=2000, cost=0.030)
        print(tracker.summary())
        if tracker.exceeded():
            print("budget exceeded!")
    """

    # Pricing per 1K tokens (as of 2025)
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }

    def __init__(self, monthly_budget_usd: Optional[float] = None):
        self.budget = monthly_budget_usd
        self._records: List[Dict[str, Any]] = []
        self._total_cost = 0.0

    def record(self, provider: str, model: str,
               input_tokens: int = 0, output_tokens: int = 0,
               cost: Optional[float] = None) -> Dict[str, Any]:
        """Record an API call."""
        if cost is None:
            pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
            cost = (input_tokens / 1000 * pricing["input"] +
                    output_tokens / 1000 * pricing["output"])
        entry = {
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "timestamp": time.time(),
        }
        self._records.append(entry)
        self._total_cost += cost
        return {
            "cost": cost,
            "total_cost": self._total_cost,
            "remaining_budget": (self.budget - self._total_cost) if self.budget else None,
            "exceeded": self.budget is not None and self._total_cost > self.budget,
        }

    def estimate_cost(self, model: str, input_tokens: int,
                       output_tokens: int) -> float:
        """Estimate cost without recording."""
        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        return input_tokens / 1000 * pricing["input"] + output_tokens / 1000 * pricing["output"]

    def exceeded(self) -> bool:
        return self.budget is not None and self._total_cost > self.budget

    def summary(self) -> Dict[str, Any]:
        by_provider: Dict[str, float] = {}
        by_model: Dict[str, float] = {}
        for r in self._records:
            by_provider[r["provider"]] = by_provider.get(r["provider"], 0) + r["cost"]
            by_model[r["model"]] = by_model.get(r["model"], 0) + r["cost"]
        return {
            "total_cost": self._total_cost,
            "budget": self.budget,
            "remaining": (self.budget - self._total_cost) if self.budget else None,
            "exceeded": self.exceeded(),
            "n_calls": len(self._records),
            "by_provider": by_provider,
            "by_model": by_model,
        }


# ============================================================================
# 5. ModelSwarm — manage a swarm of models within budget
# ============================================================================
class ModelSwarm:
    """Manage a swarm of models that share a resource budget.

    Automatically:
      - Loads models when needed
      - Evicts LRU models when budget is tight
      - Routes requests to the right model
      - Tracks per-model usage stats

    Usage:
        swarm = ModelSwarm(ram_budget_mb=4096)
        swarm.register("classifier", model_a)
        swarm.register("regressor", model_b)
        result = swarm.predict("classifier", X[0])
    """

    def __init__(self, ram_budget_mb: float = 4096,
                 gpu_budget_mb: Optional[float] = None):
        self.budget = ResourceBudget(ram_mb=ram_budget_mb, gpu_mb=gpu_budget_mb)
        self._registry: Dict[str, Any] = {}  # name → model (not yet loaded)
        self._stats: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, model: Any) -> None:
        """Register a model (doesn't load yet)."""
        self._registry[name] = model
        self._stats[name] = {
            "n_predictions": 0, "n_loads": 0, "n_evictions": 0,
            "total_latency_ms": 0.0,
        }

    def predict(self, name: str, x: Any) -> Any:
        """Get a prediction from a model (auto-loads if needed)."""
        if name not in self._registry:
            raise KeyError(f"model '{name}' not registered")
        # check if loaded
        loaded = self.budget.get(name)
        if loaded is None:
            # need to load
            model = self._registry[name]
            check = self.budget.check_model(model)
            if not check["fits"]:
                # evict LRU
                evicted = self.budget.evict_oldest()
                if evicted:
                    self._stats.get(evicted, {}).setdefault("n_evictions", 0)
                    self._stats[evicted]["n_evictions"] = self._stats[evicted].get("n_evictions", 0) + 1
            result = self.budget.load(model, name, force=True)
            if not result["loaded"]:
                raise RuntimeError(f"cannot load '{name}' — budget too tight")
            self._stats[name]["n_loads"] += 1
            loaded = self.budget.get(name)
        # predict
        t0 = time.perf_counter()
        result = loaded.forward(x)
        latency = (time.perf_counter() - t0) * 1000
        self._stats[name]["n_predictions"] += 1
        self._stats[name]["total_latency_ms"] += latency
        return result

    def status(self) -> Dict[str, Any]:
        return {
            "budget": self.budget.status(),
            "registered_models": list(self._registry.keys()),
            "loaded_models": [m["name"] for m in self.budget.list_loaded()],
            "per_model_stats": self._stats,
        }


__all__ = [
    "ResourceBudget", "ModelSwapper", "TimeBudget",
    "CostTracker", "ModelSwarm",
]
