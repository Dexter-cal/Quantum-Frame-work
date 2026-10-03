"""
qai.system -- System environment, resource monitoring, RAM/ROM automation,
time/date utilities, and automated memory safeguards for AI workloads.
"""
from __future__ import annotations
import os
import sys
import platform
import psutil
import gc
import time
from datetime import datetime, timezone
from typing import Dict, Any, Callable, Optional


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
    """Returns storage (ROM/Disk) total, used, and free space in GB for a given path."""
    usage = psutil.disk_usage(path)
    return {
        "total_gb": round(usage.total / (1024 ** 3), 2),
        "used_gb": round(usage.used / (1024 ** 3), 2),
        "free_gb": round(usage.free / (1024 ** 3), 2),
        "percent_used": usage.percent
    }


def get_ram_info() -> Dict[str, Any]:
    """Returns system RAM statistics including percent usage, available, and total capacity."""
    ram = psutil.virtual_memory()
    return {
        "total_gb": round(ram.total / (1024 ** 3), 2),
        "available_gb": round(ram.available / (1024 ** 3), 2),
        "used_gb": round(ram.used / (1024 ** 3), 2),
        "percent_used": ram.percent
    }


def get_system_time() -> Dict[str, str]:
    """Returns current system timestamp, UTC ISO time, local time, and uptime."""
    now = datetime.now()
    now_utc = datetime.now(timezone.utc)
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = str(now - boot_time).split(".")[0]
    return {
        "local_iso": now.isoformat(),
        "utc_iso": now_utc.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "uptime": uptime
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


def check_ram_threshold(max_percent: float = 90.0) -> bool:
    """Returns True if current system RAM usage exceeds max_percent."""
    ram = psutil.virtual_memory()
    return ram.percent >= max_percent


def auto_clean_memory(threshold_percent: float = 85.0) -> Dict[str, Any]:
    """Triggers garbage collection and PyTorch CUDA cache clearing if RAM/VRAM exceeds threshold."""
    ram = psutil.virtual_memory()
    cleaned = False
    freed_objects = 0

    if ram.percent >= threshold_percent:
        freed_objects = gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        cleaned = True

    return {
        "triggered": cleaned,
        "freed_objects": freed_objects,
        "ram_percent_after": psutil.virtual_memory().percent
    }


class ResourceGuard:
    """Automated safeguard watching RAM and storage limits during AI workflows."""
    def __init__(self, max_ram_percent: float = 90.0, max_disk_percent: float = 95.0, action: Optional[Callable] = None):
        self.max_ram_percent = max_ram_percent
        self.max_disk_percent = max_disk_percent
        self.action = action or auto_clean_memory

    def inspect_and_protect(self) -> Dict[str, Any]:
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage(".")

        ram_exceeded = ram.percent >= self.max_ram_percent
        disk_exceeded = disk.percent >= self.max_disk_percent

        action_result = None
        if ram_exceeded or disk_exceeded:
            if callable(self.action):
                action_result = self.action()

        return {
            "ram_percent": ram.percent,
            "disk_percent": disk.percent,
            "ram_exceeded": ram_exceeded,
            "disk_exceeded": disk_exceeded,
            "action_taken": action_result
        }


def get_env_summary() -> Dict[str, Any]:
    """Returns Python environment version, platform details, and active environment variables."""
    return {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
        "pid": os.getpid()
    }
