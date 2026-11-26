"""
qai.system.extended -- extended system monitoring (CPU/RAM/ROM/disk/network/sensors).

Wraps psutil + standard library for cross-platform system introspection.
Aimed at ML practitioners who need to:
  - check available RAM before loading a model
  - monitor disk IOPS during training data loading
  - watch network throughput for distributed training
  - read CPU temperature / fan speed
  - inspect battery state on laptops
  - get system uptime + load average
  - list running processes + their resource usage
  - send signals to processes (SIGTERM/SIGKILL)
  - read / write shared memory segments
  - time operations with high-precision clocks
"""
from __future__ import annotations
import os
import sys
import time
import socket
import platform
import threading
import signal as _signal
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False


def _require_psutil():
    if not _PSUTIL:
        raise ImportError("psutil not installed; pip install psutil")


# ============================================================================
# 1. CPU
# ============================================================================
class CPUInfo:
    """CPU info: cores, frequency, architecture, cache."""

    @staticmethod
    def info() -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_bits": platform.architecture()[0],
        }
        if _PSUTIL:
            out["physical_cores"] = psutil.cpu_count(logical=False) or 1
            out["logical_cores"] = psutil.cpu_count(logical=True) or 1
            try:
                freq = psutil.cpu_freq()
                if freq:
                    out["frequency_mhz"] = {
                        "current": freq.current, "min": freq.min, "max": freq.max,
                    }
            except Exception:
                pass
            try:
                out["cache_sizes"] = {
                    "l1": getattr(psutil, "cpu_cache_l1", None),
                    "l2": getattr(psutil, "cpu_cache_l2", None),
                    "l3": getattr(psutil, "cpu_cache_l3", None),
                }
            except Exception:
                pass
        else:
            out["logical_cores"] = os.cpu_count() or 1
            out["physical_cores"] = os.cpu_count() or 1
        return out

    @staticmethod
    def usage(interval: float = 0.1) -> Dict[str, Any]:
        """Current CPU usage per core."""
        if not _PSUTIL:
            return {"available": False}
        overall = psutil.cpu_percent(interval=interval)
        per_core = psutil.cpu_percent(interval=interval, percpu=True)
        return {
            "overall_pct": overall,
            "per_core_pct": per_core,
            "n_cores": len(per_core),
            "load_average": CPUInfo.load_average(),
        }

    @staticmethod
    def load_average() -> Optional[Tuple[float, float, float]]:
        """1/5/15-minute load average (POSIX only)."""
        if hasattr(os, "getloadavg"):
            try:
                return tuple(os.getloadavg())
            except Exception:
                pass
        return None


# ============================================================================
# 2. RAM (memory)
# ============================================================================
class RAMInfo:
    """RAM info: total / available / used / swap."""

    @staticmethod
    def info() -> Dict[str, Any]:
        if not _PSUTIL:
            # fallback to os.sysconf on POSIX
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                return {"process_rss_mb": rss / 1024 if rss > 1_000_000 else rss}
            except Exception:
                return {"available": False}
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()
        return {
            "total_bytes": vm.total,
            "total_gb": vm.total / (1024 ** 3),
            "available_bytes": vm.available,
            "available_gb": vm.available / (1024 ** 3),
            "used_bytes": vm.used,
            "used_pct": vm.percent,
            "free_bytes": vm.free,
            "swap_total_bytes": sm.total,
            "swap_used_bytes": sm.used,
            "swap_pct": sm.percent,
        }

    @staticmethod
    def process_memory(pid: Optional[int] = None) -> Dict[str, Any]:
        """Memory usage of a specific process (or current process)."""
        if not _PSUTIL:
            return {"available": False}
        try:
            p = psutil.Process(pid) if pid else psutil.Process()
            mem = p.memory_info()
            return {
                "rss_bytes": mem.rss,
                "rss_mb": mem.rss / (1024 ** 2),
                "vms_bytes": mem.vms,
                "vms_mb": mem.vms / (1024 ** 2),
                "shared_bytes": getattr(mem, "shared", 0),
                "pid": p.pid,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def enough_for(total_bytes: int, safety_factor: float = 1.5) -> bool:
        """Check if there's enough RAM for an allocation."""
        info = RAMInfo.info()
        if "available_bytes" not in info:
            return True
        return info["available_bytes"] >= total_bytes * safety_factor


# ============================================================================
# 3. Disk / Storage
# ============================================================================
class DiskInfo:
    """Disk info: partitions, usage, IOPS, throughput."""

    @staticmethod
    def partitions() -> List[Dict[str, Any]]:
        if not _PSUTIL:
            return []
        out = []
        for p in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(p.mountpoint)
                out.append({
                    "device": p.device,
                    "mountpoint": p.mountpoint,
                    "fstype": p.fstype,
                    "total_bytes": usage.total,
                    "used_bytes": usage.used,
                    "free_bytes": usage.free,
                    "used_pct": usage.percent,
                })
            except (PermissionError, OSError):
                continue
        return out

    @staticmethod
    def usage(path: str = "/") -> Dict[str, Any]:
        if not _PSUTIL:
            try:
                st = os.statvfs(path)
                total = st.f_blocks * st.f_frsize
                free = st.f_bavail * st.f_frsize
                return {"total_bytes": total, "free_bytes": free,
                         "used_bytes": total - free,
                         "used_pct": (total - free) / total * 100 if total > 0 else 0}
            except Exception:
                return {"available": False}
        u = psutil.disk_usage(path)
        return {"total_bytes": u.total, "used_bytes": u.used,
                 "free_bytes": u.free, "used_pct": u.percent}

    @staticmethod
    def io_counters() -> Dict[str, Any]:
        """Disk I/O counters (read/write bytes + ops)."""
        if not _PSUTIL:
            return {"available": False}
        try:
            io = psutil.disk_io_counters(perdisk=False)
            if io is None:
                return {"available": False}
            return {
                "read_count": io.read_count,
                "write_count": io.write_count,
                "read_bytes": io.read_bytes,
                "write_bytes": io.write_bytes,
                "read_time_ms": getattr(io, "read_time", 0),
                "write_time_ms": getattr(io, "write_time", 0),
            }
        except Exception:
            return {"available": False}

    @staticmethod
    def benchmark_write(path: str = "/tmp/qai_disk_bench",
                         size_mb: int = 100) -> Dict[str, Any]:
        """Quick disk write benchmark."""
        data = os.urandom(size_mb * 1024 * 1024)
        t0 = time.perf_counter()
        with open(path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        elapsed = time.perf_counter() - t0
        os.unlink(path)
        return {
            "size_mb": size_mb,
            "elapsed_s": elapsed,
            "throughput_mb_s": size_mb / elapsed if elapsed > 0 else 0,
        }

    @staticmethod
    def benchmark_read(path: str = "/tmp/qai_disk_bench",
                        size_mb: int = 100) -> Dict[str, Any]:
        """Quick disk read benchmark."""
        # write first
        data = os.urandom(size_mb * 1024 * 1024)
        with open(path, "wb") as f:
            f.write(data); os.fsync(f.fileno())
        # read
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            _ = f.read()
        elapsed = time.perf_counter() - t0
        os.unlink(path)
        return {
            "size_mb": size_mb,
            "elapsed_s": elapsed,
            "throughput_mb_s": size_mb / elapsed if elapsed > 0 else 0,
        }


# ============================================================================
# 4. Network
# ============================================================================
class NetworkInfo:
    """Network interfaces, throughput, latency."""

    @staticmethod
    def interfaces() -> List[Dict[str, Any]]:
        if not _PSUTIL:
            return []
        out = []
        for name, addrs in psutil.net_if_addrs().items():
            entry = {"name": name, "addresses": []}
            for a in addrs:
                entry["addresses"].append({
                    "family": str(a.family),
                    "address": a.address,
                    "netmask": a.netmask,
                    "broadcast": a.broadcast,
                })
            out.append(entry)
        return out

    @staticmethod
    def io_counters() -> Dict[str, Any]:
        if not _PSUTIL:
            return {"available": False}
        io = psutil.net_io_counters(pernic=False)
        return {
            "bytes_sent": io.bytes_sent,
            "bytes_recv": io.bytes_recv,
            "packets_sent": io.packets_sent,
            "packets_recv": io.packets_recv,
            "errin": io.errin, "errout": io.errout,
            "dropin": io.dropin, "dropout": io.dropout,
        }

    @staticmethod
    def throughput(interval: float = 1.0) -> Dict[str, float]:
        """Measure network throughput over `interval` seconds."""
        if not _PSUTIL:
            return {"available": False}
        io1 = psutil.net_io_counters()
        time.sleep(interval)
        io2 = psutil.net_io_counters()
        sent_per_sec = (io2.bytes_sent - io1.bytes_sent) / interval
        recv_per_sec = (io2.bytes_recv - io1.bytes_recv) / interval
        return {
            "send_bytes_per_sec": sent_per_sec,
            "recv_bytes_per_sec": recv_per_sec,
            "send_mb_per_sec": sent_per_sec / (1024 ** 2),
            "recv_mb_per_sec": recv_per_sec / (1024 ** 2),
            "interval_s": interval,
        }

    @staticmethod
    def latency(host: str = "8.8.8.8", port: int = 53,
                 timeout: float = 2.0) -> Dict[str, Any]:
        """TCP connect latency to a remote host."""
        t0 = time.perf_counter()
        try:
            s = socket.create_connection((host, port), timeout=timeout)
            elapsed = (time.perf_counter() - t0) * 1000
            s.close()
            return {"host": host, "latency_ms": elapsed, "reachable": True}
        except Exception as e:
            return {"host": host, "latency_ms": None, "reachable": False,
                     "error": str(e)}


# ============================================================================
# 5. Sensors (temperature, fans, battery)
# ============================================================================
class SensorInfo:
    """Hardware sensors: temperature, fan speed, battery."""

    @staticmethod
    def temperatures() -> Dict[str, Any]:
        """CPU/motherboard temperatures (Linux only)."""
        if not _PSUTIL or not hasattr(psutil, "sensors_temperatures"):
            return {"available": False}
        try:
            temps = psutil.sensors_temperatures()
            if not temps:
                return {"available": True, "sensors": {}}
            out = {}
            for name, entries in temps.items():
                out[name] = [{"label": e.label or "n/a",
                              "current": e.current,
                              "high": e.high, "critical": e.critical}
                              for e in entries]
            return {"available": True, "sensors": out}
        except Exception as e:
            return {"available": False, "error": str(e)}

    @staticmethod
    def fans() -> Dict[str, Any]:
        """Fan speeds (Linux only)."""
        if not _PSUTIL or not hasattr(psutil, "sensors_fans"):
            return {"available": False}
        try:
            fans = psutil.sensors_fans()
            return {"available": True, "fans": {k: [{"label": f.label, "current": f.current}
                                                       for f in v]
                                                  for k, v in fans.items()}}
        except Exception:
            return {"available": False}

    @staticmethod
    def battery() -> Dict[str, Any]:
        """Battery state (laptops / UPS)."""
        if not _PSUTIL or not hasattr(psutil, "sensors_battery"):
            return {"available": False}
        try:
            bat = psutil.sensors_battery()
            if bat is None:
                return {"available": True, "present": False}
            return {
                "available": True, "present": True,
                "percent": bat.percent,
                "secs_left": bat.secsleft if bat.secsleft != psutil.POWER_TIME_UNLIMITED else -1,
                "plugged_in": bat.power_plugged,
            }
        except Exception:
            return {"available": False}


# ============================================================================
# 6. Uptime / Boot time / Load average
# ============================================================================
class TimeInfo:
    """System time / uptime / boot time."""

    @staticmethod
    def boot_time() -> float:
        if _PSUTIL:
            try:
                return psutil.boot_time()
            except Exception:
                pass
        # fallback: read /proc on Linux
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime"):
                        return float(line.split()[1])
        except Exception:
            pass
        return 0.0

    @staticmethod
    def uptime() -> float:
        bt = TimeInfo.boot_time()
        if bt == 0:
            return 0.0
        return time.time() - bt

    @staticmethod
    def uptime_human() -> str:
        up = TimeInfo.uptime()
        if up == 0:
            return "unknown"
        days = int(up // 86400)
        hours = int((up % 86400) // 3600)
        mins = int((up % 3600) // 60)
        if days > 0:
            return f"{days}d {hours}h {mins}m"
        if hours > 0:
            return f"{hours}h {mins}m"
        return f"{mins}m"

    @staticmethod
    def load_average() -> Optional[Tuple[float, float, float]]:
        return CPUInfo.load_average()

    @staticmethod
    def monotonic() -> float:
        """Monotonic clock (not affected by system time changes)."""
        return time.monotonic()

    @staticmethod
    def perf_counter() -> float:
        """High-precision performance counter."""
        return time.perf_counter()

    @staticmethod
    def now_iso() -> str:
        """Current UTC time in ISO format."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


# ============================================================================
# 7. Processes
# ============================================================================
class ProcessInfo:
    """Inspect and manage running processes."""

    @staticmethod
    def list_processes(sort_by: str = "memory",
                        limit: int = 20) -> List[Dict[str, Any]]:
        """List top processes by CPU or memory."""
        if not _PSUTIL:
            return []
        procs = []
        for p in psutil.process_iter(["pid", "name", "username",
                                        "memory_info", "cpu_percent",
                                        "status", "create_time"]):
            try:
                info = p.info
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "user": info["username"],
                    "rss_mb": info["memory_info"].rss / (1024 ** 2) if info["memory_info"] else 0,
                    "cpu_pct": info["cpu_percent"] or 0,
                    "status": info["status"],
                    "uptime_s": time.time() - info["create_time"] if info["create_time"] else 0,
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if sort_by == "memory":
            procs.sort(key=lambda p: -p["rss_mb"])
        elif sort_by == "cpu":
            procs.sort(key=lambda p: -p["cpu_pct"])
        return procs[:limit]

    @staticmethod
    def get(pid: int) -> Dict[str, Any]:
        if not _PSUTIL:
            return {"available": False}
        try:
            p = psutil.Process(pid)
            return {
                "pid": pid,
                "name": p.name(),
                "exe": p.exe(),
                "cmdline": p.cmdline(),
                "cwd": p.cwd(),
                "status": p.status(),
                "create_time": p.create_time(),
                "cpu_percent": p.cpu_percent(),
                "memory_info": {
                    "rss": p.memory_info().rss,
                    "vms": p.memory_info().vms,
                },
                "num_threads": p.num_threads(),
                "num_fds": p.num_fds() if hasattr(p, "num_fds") else None,
                "connections": len(p.connections()) if hasattr(p, "connections") else None,
            }
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def kill(pid: int, signal: str = "SIGTERM") -> bool:
        """Send a signal to a process."""
        sig = getattr(_signal, signal, _signal.SIGTERM)
        try:
            os.kill(pid, sig)
            return True
        except Exception:
            return False

    @staticmethod
    def children(pid: int) -> List[int]:
        """Get child process IDs."""
        if not _PSUTIL:
            return []
        try:
            p = psutil.Process(pid)
            return [c.pid for c in p.children(recursive=True)]
        except Exception:
            return []


# ============================================================================
# 8. IPC / Shared memory
# ============================================================================
class SharedMemoryHelper:
    """Wrapper around multiprocessing.shared_memory for cross-process data."""

    @staticmethod
    def create(name: str, size: int) -> Any:
        """Create a named shared memory block."""
        from multiprocessing import shared_memory
        try:
            return shared_memory.SharedMemory(name=name, size=size, create=True)
        except FileExistsError:
            return shared_memory.SharedMemory(name=name, size=size, create=False)

    @staticmethod
    def open(name: str) -> Any:
        from multiprocessing import shared_memory
        return shared_memory.SharedMemory(name=name, create=False)

    @staticmethod
    def unlink(name: str) -> None:
        from multiprocessing import shared_memory
        try:
            shm = shared_memory.SharedMemory(name=name, create=False)
            shm.close(); shm.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def write(name: str, data: bytes) -> None:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=name, create=True, size=len(data))
        shm.buf[:len(data)] = data
        return shm

    @staticmethod
    def read(name: str, size: int) -> bytes:
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=name, create=False)
        return bytes(shm.buf[:size])


# ============================================================================
# 9. High-precision timer for benchmarking
# ============================================================================
class Timer:
    """Context manager + standalone timer with nanosecond precision."""

    def __init__(self, name: str = "block"):
        self.name = name
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self.start is None:
            return 0.0
        end = self.end or time.perf_counter()
        return (end - self.start) * 1000

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000

    def __repr__(self):
        return f"<Timer {self.name}: {self.elapsed_ms:.2f}ms>"


class MultiTimer:
    """Track multiple named timers in a single session."""

    def __init__(self):
        self.timers: Dict[str, List[float]] = defaultdict(list)

    def time(self, name: str) -> Timer:
        """Context manager for a named timer."""
        t = Timer(name)
        self.timers[name].append(t)
        return t

    def summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for name, ts in self.timers.items():
            elapsed = [t.elapsed_ms for t in ts if t.end is not None]
            if elapsed:
                out[name] = {
                    "n_calls": len(elapsed),
                    "total_ms": sum(elapsed),
                    "mean_ms": float(np.mean(elapsed)),
                    "min_ms": min(elapsed),
                    "max_ms": max(elapsed),
                }
        return out


# ============================================================================
# 10. Combined system snapshot
# ============================================================================
def system_snapshot() -> Dict[str, Any]:
    """One-shot snapshot of the whole system."""
    return {
        "timestamp": TimeInfo.now_iso(),
        "host": platform.node(),
        "platform": platform.platform(),
        "uptime_s": TimeInfo.uptime(),
        "uptime_human": TimeInfo.uptime_human(),
        "load_average": TimeInfo.load_average(),
        "cpu": CPUInfo.info(),
        "cpu_usage": CPUInfo.usage(interval=0.05) if _PSUTIL else None,
        "ram": RAMInfo.info(),
        "disk": DiskInfo.usage("/"),
        "disk_io": DiskInfo.io_counters(),
        "network_io": NetworkInfo.io_counters(),
        "temperatures": SensorInfo.temperatures(),
        "battery": SensorInfo.battery(),
        "process_count": len(ProcessInfo.list_processes(limit=9999)) if _PSUTIL else 0,
    }


def health_check() -> Dict[str, Any]:
    """Quick system health check."""
    issues: List[str] = []
    if _PSUTIL:
        ram = RAMInfo.info()
        if ram.get("used_pct", 0) > 90:
            issues.append(f"RAM usage {ram['used_pct']:.1f}% > 90%")
        cpu_pct = psutil.cpu_percent(interval=0.1)
        if cpu_pct > 90:
            issues.append(f"CPU usage {cpu_pct:.1f}% > 90%")
        for part in DiskInfo.partitions():
            if part.get("used_pct", 0) > 95:
                issues.append(f"Disk {part['mountpoint']} {part['used_pct']:.1f}% > 95%")
        bat = SensorInfo.battery()
        if bat.get("present") and not bat.get("plugged_in") and bat.get("percent", 100) < 15:
            issues.append(f"Battery low: {bat['percent']:.0f}%")
    return {
        "healthy": not issues,
        "issues": issues,
        "n_issues": len(issues),
        "snapshot": system_snapshot(),
    }


__all__ = [
    "CPUInfo", "RAMInfo", "DiskInfo", "NetworkInfo", "SensorInfo",
    "TimeInfo", "ProcessInfo", "SharedMemoryHelper",
    "Timer", "MultiTimer",
    "system_snapshot", "health_check",
]
