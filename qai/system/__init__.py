"""
qai.system -- System environment, resource monitoring, and process utilities for AI workloads.
"""
from __future__ import annotations
import os
import sys
import platform
import psutil
from typing import Dict, Any


def get_gpu_memory() -> Dict[str, Any]:
    """Returns GPU VRAM total and allocated memory if PyTorch CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
                "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
                "max_memory_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), 2)
            }
    except Exception:
        pass
    return {
        "cuda_available": False,
        "device_name": None,
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "max_memory_mb": 0.0
    }


def get_process_memory() -> Dict[str, float]:
    """Returns current Python process RSS and VMS RAM usage in MB."""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()
    return {
        "rss_mb": round(mem.rss / (1024 ** 2), 2),
        "vms_mb": round(mem.vms / (1024 ** 2), 2)
    }


def get_disk_usage(path: str = ".") -> Dict[str, float]:
    """Returns disk total, used, and free space in GB for a given path."""
    usage = psutil.disk_usage(path)
    return {
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "percent_used": usage.percent
    }


def set_num_threads(n_threads: int) -> int:
    """Sets CPU thread parallelism limit for NumPy / OpenMP / PyTorch."""
    if not isinstance(n_threads, int) or n_threads <= 0:
        raise ValueError("n_threads must be a positive integer")

    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)

    try:
        import torch
        torch.set_num_threads(n_threads)
    except Exception:
        pass

    return n_threads


def get_env_summary() -> Dict[str, Any]:
    """Returns Python environment version, platform details, and active environment variables."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid()
    }
