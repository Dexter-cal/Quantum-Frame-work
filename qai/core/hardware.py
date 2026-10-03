"""
qai.core.hardware -- System, RAM, CPU, and CUDA GPU hardware profiling.
Provides get_hardware_info() and get_optimal_device() for AI resource optimization.
"""
from __future__ import annotations
import platform
import os
import psutil
from typing import Dict, Any


def get_hardware_info() -> Dict[str, Any]:
    """Inspects local hardware: CPU cores, total/available RAM, OS architecture, and CUDA availability."""
    ram = psutil.virtual_memory()
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count_logical": os.cpu_count() or 1,
        "ram_total_gb": round(ram.total / (1024 ** 3), 2),
        "ram_available_gb": round(ram.available / (1024 ** 3), 2),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def get_optimal_device() -> str:
    """Returns 'cuda' if NVIDIA CUDA GPU is available, else 'cpu'."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
