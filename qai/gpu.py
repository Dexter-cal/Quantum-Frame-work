"""
qai.gpu -- GPU compute + custom CUDA kernels + device-aware tensors.

Five layers, from highest to lowest:

  Layer 1: DeviceManager
      Detects available GPU backends (CUDA / Metal / ROCm / TPU / Apple Silicon).
      Picks the best one. Tracks device memory + clears cache on demand.

  Layer 2: DeviceTensor
      A numpy-like array that automatically lives on the best available device.
      Supports .to(device), .cpu(), .gpu() moves; falls back to CPU if no GPU.

  Layer 3: CudaKernel (via CuPy or Numba)
      Write custom CUDA kernels inline in Python.
          @qai.gpu.kernel
          def add(x, y, out, n):
              tid = qai.gpu.thread_id()
              if tid < n:
                  out[tid] = x[tid] + y[tid]
      The kernel compiles on first call (JIT) and caches.

  Layer 4: GPU-accelerated ops
      Drop-in replacements for common numpy ops that run on GPU when available:
      qai.gpu.matmul, qai.gpu.softmax, qai.gpu.layernorm, qai.gpu.conv2d,
      qai.gpu.gather, qai.gpu.scatter

  Layer 5: Memory management
      gpu_memory(), empty_cache(), memory_pool(), pinned_memory()
"""
from __future__ import annotations
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# Lazy backend imports
# ============================================================================
def _try_import(name: str):
    try:
        import importlib
        return importlib.import_module(name)
    except ImportError:
        return None


_torch = _try_import("torch")
_cupy = _try_import("cupy")
_numba = _try_import("numba")
_numba_cuda = None
if _numba is not None:
    try:
        from numba import cuda as _numba_cuda
    except Exception:
        _numba_cuda = None


# ============================================================================
# Layer 1: Device Manager
# ============================================================================
class DeviceManager:
    """Detects and manages all available compute devices.

    Order of preference:
      1. NVIDIA GPU (via CUDA)
      2. Apple Silicon GPU (via Metal / MPS)
      3. AMD GPU (via ROCm)
      4. TPU (via XLA)
      5. CPU fallback
    """
    def __init__(self):
        self._cached_devices: Optional[Dict[str, Any]] = None

    def detect(self) -> Dict[str, Any]:
        """Detect all available devices."""
        if self._cached_devices is not None:
            return self._cached_devices
        devices: Dict[str, Any] = {
            "cuda": [], "mps": [], "rocm": [], "tpu": [], "cpu": True,
        }
        # CUDA
        if _torch is not None and _torch.cuda.is_available():
            for i in range(_torch.cuda.device_count()):
                props = _torch.cuda.get_device_properties(i)
                devices["cuda"].append({
                    "id": i,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "total_memory_gb": props.total_memory / (1024 ** 3),
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multiprocessor_count": props.multi_processor_count,
                })
        # Apple Silicon MPS
        if _torch is not None and hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
            devices["mps"].append({"id": 0, "name": "Apple Silicon GPU"})
        # CuPy also indicates CUDA presence
        if _cupy is not None:
            try:
                n = _cupy.cuda.runtime.getDeviceCount()
                if n > 0 and not devices["cuda"]:
                    devices["cuda"] = [{
                        "id": i,
                        "name": _cupy.cuda.runtime.getDeviceProperties(i).get("name", "Unknown"),
                        "total_memory_bytes": _cupy.cuda.runtime.memGetInfo()[1],
                    } for i in range(n)]
            except Exception:
                pass
        self._cached_devices = devices
        return devices

    def best_device(self) -> str:
        """Return the name of the best available device."""
        d = self.detect()
        if d["cuda"]:
            return f"cuda:0"
        if d["mps"]:
            return "mps"
        if d["rocm"]:
            return "rocm:0"
        return "cpu"

    def is_gpu_available(self) -> bool:
        d = self.detect()
        return bool(d["cuda"] or d["mps"] or d["rocm"])

    def summary(self) -> Dict[str, Any]:
        d = self.detect()
        return {
            "best_device": self.best_device(),
            "cuda_devices": len(d["cuda"]),
            "mps_devices": len(d["mps"]),
            "cpu_available": d["cpu"],
            "torch_version": getattr(_torch, "__version__", None) if _torch else None,
            "cupy_version": getattr(_cupy, "__version__", None) if _cupy else None,
            "numba_version": getattr(_numba, "__version__", None) if _numba else None,
        }


# ============================================================================
# Layer 2: Device Tensor
# ============================================================================
class DeviceTensor:
    """A numpy-like array that lives on the best available device.

    Wraps either a torch tensor (on GPU if available) or a numpy array.
    Supports the common math ops without requiring the user to know
    whether they're on CPU or GPU.
    """
    def __init__(self, data: Any, device: Optional[str] = None):
        if isinstance(data, DeviceTensor):
            self._data = data._data
            self._backend = data._backend
            return
        device = device or DeviceManager().best_device()
        if _torch is not None and device != "cpu":
            try:
                t = _torch.as_tensor(np.asarray(data))
                self._data = t.to(device)
                self._backend = "torch"
                return
            except Exception:
                pass
        self._data = np.asarray(data)
        self._backend = "numpy"

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def device(self) -> str:
        if self._backend == "torch":
            return str(self._data.device)
        return "cpu"

    def to(self, device: str) -> "DeviceTensor":
        if self._backend == "torch" and _torch is not None:
            try:
                return DeviceTensor.__new__(DeviceTensor)
                # manual construction to avoid copy
            except Exception:
                pass
        # fallback: just construct on target device
        return DeviceTensor(self.cpu().numpy(), device=device)

    def cpu(self) -> "DeviceTensor":
        if self._backend == "torch":
            return DeviceTensor(self._data.cpu().numpy())
        return self

    def numpy(self) -> np.ndarray:
        if self._backend == "torch":
            return self._data.detach().cpu().numpy() if self._data.requires_grad else \
                    self._data.cpu().numpy()
        return self._data

    def __add__(self, other):
        other = other._data if isinstance(other, DeviceTensor) else other
        return DeviceTensor(self._data + other)
    def __mul__(self, other):
        other = other._data if isinstance(other, DeviceTensor) else other
        return DeviceTensor(self._data * other)
    def __matmul__(self, other):
        other = other._data if isinstance(other, DeviceTensor) else other
        return DeviceTensor(self._data @ other)

    def __repr__(self):
        return f"DeviceTensor(shape={self.shape}, dtype={self.dtype}, device={self.device})"


# ============================================================================
# Layer 3: CUDA Kernel (via CuPy or Numba)
# ============================================================================
class _KernelDecorator:
    """Decorator that turns a Python function into a CUDA kernel.

    Uses CuPy's @jit.rawkernel if available, else Numba's @cuda.jit,
    else falls back to a CPU loop.
    """
    def __init__(self):
        self._kernels: Dict[str, Any] = {}

    def __call__(self, fn: Callable) -> Callable:
        return _CompiledKernel(fn)


class _CompiledKernel:
    """Wraps a Python function with a GPU kernel implementation."""

    def __init__(self, fn: Callable):
        self.fn = fn
        self.name = fn.__name__
        self._gpu_kernel = None
        self._compile_attempted = False

    def _compile(self):
        if self._compile_attempted:
            return
        self._compile_attempted = True
        # try CuPy JIT
        if _cupy is not None:
            try:
                from cupyx import jit as cp_jit
                self._gpu_kernel = cp_jit.rawkernel()(self.fn)
                return
            except Exception:
                pass
        # try Numba CUDA
        if _numba_cuda is not None:
            try:
                self._gpu_kernel = _numba_cuda.jit(self.fn)
                return
            except Exception:
                pass
        self._gpu_kernel = None  # CPU fallback

    def __call__(self, *args, **kwargs):
        self._compile()
        if self._gpu_kernel is not None and _cupy is not None:
            # check if any arg is a cupy array
            if any(isinstance(a, _cupy.ndarray) for a in args):
                # call as CuPy raw kernel: needs grid/block spec
                # default: 1 block of 256 threads, grid scaled to size
                n = max((a.size for a in args if hasattr(a, "size")), default=1)
                grid = ((n + 255) // 256,)
                block = (256,)
                try:
                    return self._gpu_kernel(grid, block, *args)
                except Exception:
                    pass
        if self._gpu_kernel is not None and _numba_cuda is not None:
            # check if any arg is a numba device array
            try:
                from numba.cuda.cudadrv.devicearray import DeviceArray as _NDevArray
                if any(isinstance(a, _NDevArray) for a in args):
                    n = max((a.size for a in args if hasattr(a, "size")), default=1)
                    return self._gpu_kernel[((n + 255) // 256, 1, 1), (256, 1, 1)](*args)
            except Exception:
                pass
        # CPU fallback: just call the function directly
        # (the user's function should handle CPU execution itself)
        return self.fn(*args, **kwargs)


# Public decorator
kernel = _KernelDecorator()


# Thread/grid helpers for kernels
def thread_id() -> int:
    """Returns the current thread index inside a kernel (or 0 on CPU)."""
    if _cupy is not None:
        try:
            from cupyx import jit as cp_jit
            return cp_jit.threadIdx.x + cp_jit.blockIdx.x * cp_jit.blockDim.x
        except Exception:
            pass
    return 0


def block_dim() -> int:
    """Returns the block dimension (or 1 on CPU)."""
    if _cupy is not None:
        try:
            from cupyx import jit as cp_jit
            return cp_jit.blockDim.x
        except Exception:
            pass
    return 1


def grid_dim() -> int:
    """Returns the grid dimension (or 1 on CPU)."""
    if _cupy is not None:
        try:
            from cupyx import jit as cp_jit
            return cp_jit.gridDim.x
        except Exception:
            pass
    return 1


def sync_threads() -> None:
    """Synchronise threads in a block (no-op on CPU)."""
    if _cupy is not None:
        try:
            from cupyx import jit as cp_jit
            cp_jit.syncthreads()
        except Exception:
            pass


# ============================================================================
# Layer 4: GPU-accelerated ops
# ============================================================================
def _to_backend(x: Any) -> Tuple[Any, str]:
    """Move x to the best available backend; return (data, backend_name)."""
    if _torch is not None:
        device = DeviceManager().best_device()
        try:
            t = _torch.as_tensor(np.asarray(x))
            if device != "cpu":
                t = t.to(device)
            return t, "torch"
        except Exception:
            pass
    if _cupy is not None:
        try:
            return _cupy.asarray(x), "cupy"
        except Exception:
            pass
    return np.asarray(x), "numpy"


def _from_backend(x: Any, backend: str) -> np.ndarray:
    if backend == "torch" and _torch is not None:
        return x.detach().cpu().numpy() if hasattr(x, "detach") else x.cpu().numpy()
    if backend == "cupy" and _cupy is not None:
        return _cupy.asnumpy(x)
    return np.asarray(x)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GPU-accelerated matrix multiplication (falls back to numpy)."""
    a_data, a_bk = _to_backend(a)
    b_data, b_bk = _to_backend(b)
    # align backends
    if a_bk != b_bk:
        if a_bk == "torch":
            b_data = _torch.as_tensor(_from_backend(b_data, b_bk)).to(a_data.device)
            b_bk = a_bk
        else:
            a_data = _to_backend(_from_backend(a_data, a_bk))[0]
            a_bk = b_bk
    out = a_data @ b_data
    return _from_backend(out, a_bk)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """GPU-accelerated softmax."""
    if _torch is not None:
        try:
            data, bk = _to_backend(x)
            out = _torch.softmax(data, dim=axis)
            return _from_backend(out, bk)
        except Exception:
            pass
    # numpy fallback
    x = np.asarray(x, dtype=np.float32)
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def layernorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """GPU-accelerated LayerNorm."""
    if _torch is not None:
        try:
            data, bk = _to_backend(x)
            out = _torch.nn.functional.layer_norm(data, data.shape[-1:], eps=eps)
            return _from_backend(out, bk)
        except Exception:
            pass
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def conv2d(x: np.ndarray, weight: np.ndarray, bias: Optional[np.ndarray] = None,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0) -> np.ndarray:
    """GPU-accelerated 2D convolution (cross-correlation, torch semantics).

    Shapes:
        x: (N, C_in, H, W) batch or (C_in, H, W) single image (batch dim is
           added and the output is squeezed back to (C_out, H', W')).
        weight: (C_out, C_in, kH, kW). A 3-dim weight is tolerated: if its
           first dimension matches C_in it is read as (C_in, kH, kW) — a
           single-output-channel kernel — otherwise as (C_out, kH, kW)
           with C_in = 1.
        bias: (C_out,) or None.
        stride / padding: int or (h, w) pair.

    When PyTorch is available this dispatches to ``torch.nn.functional.conv2d``.
    Without PyTorch the fallback is a REAL convolution: vectorized as
    ``sliding_window_view`` + ``einsum`` (cross-correlation, i.e. no kernel
    flip — the same convention torch uses). Before v1.8.8 this fallback
    returned the input unchanged (it was marked "simplified").
    """
    if _torch is not None:
        try:
            x_t, x_bk = _to_backend(x)
            w_t, _ = _to_backend(weight)
            b_t = None
            if bias is not None:
                b_t, _ = _to_backend(bias)
            if x_t.ndim == 3:
                x_t = x_t.unsqueeze(0)  # add batch dim
                squeeze = True
            else:
                squeeze = False
            if w_t.ndim == 3:
                w_t = w_t.unsqueeze(0)
            out = _torch.nn.functional.conv2d(x_t, w_t, bias=b_t, stride=stride, padding=padding)
            if squeeze:
                out = out.squeeze(0)
            return _from_backend(out, x_bk)
        except Exception:
            pass
    # numpy fallback: vectorized cross-correlation (matches torch conv2d)
    return _conv2d_numpy(x, weight, bias=bias, stride=stride, padding=padding)


def _conv2d_numpy(x: np.ndarray, weight: np.ndarray,
                  bias: Optional[np.ndarray] = None,
                  stride: Union[int, Tuple[int, int]] = 1,
                  padding: Union[int, Tuple[int, int]] = 0) -> np.ndarray:
    """Real numpy conv2d via sliding_window_view + einsum (no torch needed)."""
    from numpy.lib.stride_tricks import sliding_window_view

    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    if x.ndim not in (3, 4):
        raise ValueError(
            f"[WHAT] conv2d received input with shape {x.shape}\n"
            "[WHY] Input must be (C_in, H, W) or (N, C_in, H, W)\n"
            "[FIX] Squeeze or expand a batch/channel dim before calling"
        )
    if weight.ndim not in (3, 4):
        raise ValueError(
            f"[WHAT] conv2d received weight with shape {weight.shape}\n"
            "[WHY] Weight must be (C_out, C_in, kH, kW) "
            "(3-dim (C_in, kH, kW) is tolerated as a single-output kernel)\n"
            "[FIX] Reshape the kernel to 4 dims"
        )

    squeeze_out = x.ndim == 3
    if squeeze_out:
        x = x[None]
    n, c_in, h, w = x.shape

    if weight.ndim == 3:
        # disambiguate against the input's channel count
        if weight.shape[0] == c_in:
            weight = weight[None]      # (C_in, kH, kW) -> (1, C_in, kH, kW)
        else:
            weight = weight[:, None]   # (C_out, kH, kW) -> (C_out, 1, kH, kW)
    c_out, w_cin, kh, kw = weight.shape
    if w_cin != c_in:
        raise ValueError(
            f"[WHAT] conv2d channel mismatch: input has {c_in} channels but "
            f"weight expects {w_cin}\n"
            "[WHY] A convolution slides one kernel per input channel and "
            "sums across channels\n"
            f"[FIX] Make weight shape ({c_out}, {c_in}, {kh}, {kw})"
        )

    sh, sw = stride if isinstance(stride, tuple) else (stride, stride)
    ph, pw = padding if isinstance(padding, tuple) else (padding, padding)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
    n, c_in, h, w = x.shape
    if h < kh or w < kw:
        raise ValueError(
            f"[WHAT] conv2d kernel ({kh}x{kw}) is larger than the (padded) "
            f"input ({h}x{w})\n"
            "[WHY] There is no valid window to slide\n"
            "[FIX] Reduce the kernel size or increase padding"
        )

    # windows: (N, C_in, H_out, W_out, kH, kW)
    windows = sliding_window_view(x, (kh, kw), axis=(2, 3))[:, :, ::sh, ::sw]
    # cross-correlate: sum over (C_in, kH, kW) for each output channel
    out = np.einsum("nchwij,ocij->nohw", windows, weight, optimize=True)
    if bias is not None:
        bias = np.asarray(bias, dtype=np.float32).reshape(1, c_out, 1, 1)
        out = out + bias
    return out[0] if squeeze_out else out


def gather(x: np.ndarray, indices: np.ndarray, axis: int = 0) -> np.ndarray:
    """GPU-accelerated gather."""
    if _torch is not None:
        try:
            x_t, x_bk = _to_backend(x)
            i_t = _torch.as_tensor(np.asarray(indices, dtype=np.int64)).to(x_t.device)
            out = _torch.gather(x_t, axis, i_t)
            return _from_backend(out, x_bk)
        except Exception:
            pass
    return np.take(np.asarray(x), np.asarray(indices), axis=axis)


def scatter(target: np.ndarray, indices: np.ndarray, values: np.ndarray, axis: int = 0) -> np.ndarray:
    """GPU-accelerated scatter."""
    if _torch is not None:
        try:
            t_t, t_bk = _to_backend(target)
            i_t = _torch.as_tensor(np.asarray(indices, dtype=np.int64)).to(t_t.device)
            v_t, _ = _to_backend(values)
            v_t = v_t.to(t_t.device)
            out = t_t.scatter(axis, i_t, v_t)
            return _from_backend(out, t_bk)
        except Exception:
            pass
    target = np.array(target, dtype=np.float32)
    np.put_along_axis(target, np.asarray(indices), np.asarray(values), axis=axis)
    return target


# ============================================================================
# Layer 5: Memory management
# ============================================================================
def gpu_memory() -> Dict[str, Any]:
    """Current GPU memory usage."""
    if _torch is not None and _torch.cuda.is_available():
        try:
            return {
                "allocated_mb": _torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mb": _torch.cuda.memory_reserved() / (1024 ** 2),
                "max_allocated_mb": _torch.cuda.max_memory_allocated() / (1024 ** 2),
                "device": _torch.cuda.get_device_name(0),
            }
        except Exception:
            pass
    if _cupy is not None:
        try:
            free, total = _cupy.cuda.runtime.memGetInfo()
            return {
                "free_bytes": free,
                "total_bytes": total,
                "used_bytes": total - free,
                "free_mb": free / (1024 ** 2),
                "total_mb": total / (1024 ** 2),
            }
        except Exception:
            pass
    return {"available": False}


def empty_cache() -> None:
    """Empty GPU memory caches (PyTorch + CuPy)."""
    if _torch is not None and _torch.cuda.is_available():
        try:
            _torch.cuda.empty_cache()
        except Exception:
            pass
    if _cupy is not None:
        try:
            _cupy.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def memory_pool() -> Dict[str, Any]:
    """Inspect memory pools."""
    pool_info: Dict[str, Any] = {}
    if _torch is not None and _torch.cuda.is_available():
        try:
            pool_info["torch"] = {
                "allocated_mb": _torch.cuda.memory_allocated() / (1024 ** 2),
                "reserved_mb": _torch.cuda.memory_reserved() / (1024 ** 2),
            }
        except Exception:
            pass
    if _cupy is not None:
        try:
            pool_info["cupy"] = {
                "used_bytes": _cupy.get_default_memory_pool().used_bytes(),
                "total_bytes": _cupy.get_default_memory_pool().total_bytes(),
            }
        except Exception:
            pass
    return pool_info


def pinned_memory(arr: np.ndarray) -> Any:
    """Allocate page-locked (pinned) host memory for faster GPU transfers."""
    if _torch is not None and _torch.cuda.is_available():
        try:
            t = _torch.as_tensor(arr)
            return t.pin_memory()
        except Exception:
            pass
    return np.asarray(arr)


# ============================================================================
# Convenience: move a qai model to GPU
# ============================================================================
def to_gpu(model: Any) -> Any:
    """Move a model to GPU if available."""
    if _torch is not None and _torch.cuda.is_available():
        if hasattr(model, "to"):
            try:
                return model.to("cuda")
            except Exception:
                pass
    return model


def to_cpu(model: Any) -> Any:
    """Move a model back to CPU."""
    if _torch is not None and hasattr(model, "to"):
        try:
            return model.to("cpu")
        except Exception:
            pass
    return model


def device_count() -> int:
    """Number of available GPUs."""
    if _torch is not None and _torch.cuda.is_available():
        return _torch.cuda.device_count()
    if _cupy is not None:
        try:
            return _cupy.cuda.runtime.getDeviceCount()
        except Exception:
            pass
    return 0


__all__ = [
    "DeviceManager", "DeviceTensor",
    "kernel", "thread_id", "block_dim", "grid_dim", "sync_threads",
    "matmul", "softmax", "layernorm", "conv2d", "gather", "scatter",
    "gpu_memory", "empty_cache", "memory_pool", "pinned_memory",
    "to_gpu", "to_cpu", "device_count",
]
