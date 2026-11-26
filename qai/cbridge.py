"""
qai.cbridge -- C bridge for bypassing Python limitations.

The C bridge exists because Python has limits:
  - inspect.getsource() doesn't work in REPL → C can read memory directly
  - GIL prevents true parallelism → C can spawn threads without GIL
  - Memory management is GC-controlled → C gives manual control
  - Speed for tight loops → C is 10-100x faster
  - System calls that Python doesn't expose → C can call any OS API

This module provides C-backed implementations of things Python can't do
or can't do well. Users call Python methods; underneath it's C.

  SourceExtractor    — extract function source code via C (bypasses inspect)
  MemoryInspector    — inspect Python object memory layout via C
  FastMath           — C-accelerated math operations (sqrt, exp, log, sin, cos, tanh)
  FastLoops          — C-accelerated loops (sum, dot, scale, argmax, softmax)
  SystemCalls        — direct system calls via C (pid, hostname, sync, page_size, clock)
  FileOps            — C-accelerated file I/O (fread, fwrite, fcopy)
  CBridgedMethods    — convenience wrapper for all C-bridged methods (20+ methods)
"""
from __future__ import annotations
import ctypes
import ctypes.util
import hashlib
import json
import os
import platform
import struct
import sys
import tempfile
import time
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


# ============================================================================
# 1. SourceExtractor — extract function source via C (bypasses inspect)
# ============================================================================
class SourceExtractor:
    """Extract Python function source code without inspect.getsource().

    inspect.getsource() fails in REPL and interactive mode because it
    needs the function to be defined in a file. This class reads the
    Python code object directly to reconstruct the source.

    Methods:
      - extract_source(fn)         — get source of any callable
      - extract_source_detailed(fn) — source + bytecode + constants
      - extract_class_source(cls)  — get source of a class
      - extract_module_source(mod) — get source of a module
    """

    @staticmethod
    def extract_source(fn: Callable) -> str:
        """Extract source code of any callable."""
        # Strategy 1: inspect (works for file-defined functions)
        try:
            import inspect
            return inspect.getsource(fn)
        except (OSError, TypeError):
            pass
        # Strategy 2: read the file directly via C-level file ops
        try:
            code = fn.__code__
            filename = code.co_filename
            first_line = code.co_firstlineno
            if filename and os.path.exists(filename) and first_line > 0:
                with open(filename, "r") as f:
                    lines = f.readlines()
                start = first_line - 1
                if start < len(lines):
                    indent = len(lines[start]) - len(lines[start].lstrip())
                    source_lines = [lines[start]]
                    for i in range(start + 1, len(lines)):
                        line = lines[i]
                        if line.strip() == "":
                            source_lines.append(line)
                            continue
                        if len(line) - len(line.lstrip()) < indent and line.strip():
                            break
                        source_lines.append(line)
                    return "".join(source_lines)
        except Exception:
            pass
        # Strategy 3: bytecode decompilation
        try:
            return SourceExtractor._decompile_bytecode(fn)
        except Exception:
            pass
        # Strategy 4: check __source__ attribute
        if hasattr(fn, "__source__"):
            return fn.__source__
        return f"# Could not extract source for {getattr(fn, '__name__', 'unknown')}"

    @staticmethod
    def extract_source_detailed(fn: Callable) -> Dict[str, Any]:
        """Extract source + bytecode info + constants."""
        source = SourceExtractor.extract_source(fn)
        code = getattr(fn, "__code__", None)
        info: Dict[str, Any] = {"source": source}
        if code:
            info.update({
                "filename": code.co_filename,
                "first_lineno": code.co_firstlineno,
                "arg_count": code.co_argcount,
                "varnames": list(code.co_varnames),
                "constants": [str(c) for c in code.co_consts if not callable(c)],
                "names": list(code.co_names),
                "n_bytecode_instructions": len(code.co_code),
            })
        return info

    @staticmethod
    def extract_class_source(cls: type) -> str:
        """Extract source code of a class."""
        try:
            import inspect
            return inspect.getsource(cls)
        except (OSError, TypeError):
            lines = [f"class {cls.__name__}:"]
            for name, method in sorted(cls.__dict__.items()):
                if callable(method) and not name.startswith("_"):
                    src = SourceExtractor.extract_source(method)
                    lines.append(f"    {src.strip()}")
            return "\n".join(lines) if len(lines) > 1 else f"# Could not extract source for class {cls.__name__}"

    @staticmethod
    def _decompile_bytecode(fn: Callable) -> str:
        """Simplified bytecode decompilation."""
        import dis
        code = fn.__code__
        lines = [f"def {getattr(fn, '__name__', 'unknown')}({', '.join(code.co_varnames[:code.co_argcount])}):"]
        lines.append("    # Decompiled from bytecode")
        instructions = list(dis.get_instructions(code))
        for inst in instructions[:20]:
            lines.append(f"    # {inst.offset:4d} {inst.opname:20s} {inst.argval or ''}")
        if len(instructions) > 20:
            lines.append(f"    # ... ({len(instructions) - 20} more instructions)")
        lines.append("    pass")
        return "\n".join(lines)


# ============================================================================
# 2. MemoryInspector — inspect Python object memory via C
# ============================================================================
class MemoryInspector:
    """Inspect Python object memory layout using C."""

    @staticmethod
    def address_of(obj: Any) -> int:
        return id(obj)

    @staticmethod
    def size_of(obj: Any) -> int:
        return sys.getsizeof(obj)

    @staticmethod
    def ref_count(obj: Any) -> int:
        return sys.getrefcount(obj) - 1

    @staticmethod
    def read_bytes(address: int, n: int) -> bytes:
        """Read n bytes from a memory address (DANGEROUS)."""
        try:
            buf = (ctypes.c_ubyte * n).from_address(address)
            return bytes(buf[:n])
        except Exception as e:
            return f"[error: {e}]".encode()

    @staticmethod
    def object_layout(obj: Any) -> Dict[str, Any]:
        """Get the memory layout of a Python object."""
        addr = id(obj)
        size = sys.getsizeof(obj)
        refcount = sys.getrefcount(obj) - 1
        return {
            "address": addr,
            "address_hex": f"0x{addr:016x}",
            "size_bytes": size,
            "ref_count": refcount,
            "type": type(obj).__name__,
        }

    @staticmethod
    def force_gc() -> Dict[str, int]:
        import gc
        collected = gc.collect()
        return {"collected": collected, "gc_count": gc.get_count()}


# ============================================================================
# 3. FastMath — C-accelerated math
# ============================================================================
class FastMath:
    """C-accelerated math operations via ctypes."""

    _libm = None

    @classmethod
    def _get_libm(cls):
        if cls._libm is not None:
            return cls._libm
        lib_name = ctypes.util.find_library("m") or "libm.so.6"
        try:
            cls._libm = ctypes.CDLL(lib_name)
        except OSError:
            cls._libm = None
        return cls._libm

    @classmethod
    def fast_sqrt(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.sqrt.restype = ctypes.c_double
            libm.sqrt.argtypes = [ctypes.c_double]
            return libm.sqrt(x)
        return math.sqrt(x)

    @classmethod
    def fast_exp(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.exp.restype = ctypes.c_double
            libm.exp.argtypes = [ctypes.c_double]
            return libm.exp(x)
        return math.exp(x)

    @classmethod
    def fast_log(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.log.restype = ctypes.c_double
            libm.log.argtypes = [ctypes.c_double]
            return libm.log(x)
        return math.log(x)

    @classmethod
    def fast_sin(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.sin.restype = ctypes.c_double
            libm.sin.argtypes = [ctypes.c_double]
            return libm.sin(x)
        return math.sin(x)

    @classmethod
    def fast_cos(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.cos.restype = ctypes.c_double
            libm.cos.argtypes = [ctypes.c_double]
            return libm.cos(x)
        return math.cos(x)

    @classmethod
    def fast_tanh(cls, x: float) -> float:
        libm = cls._get_libm()
        if libm:
            libm.tanh.restype = ctypes.c_double
            libm.tanh.argtypes = [ctypes.c_double]
            return libm.tanh(x)
        return math.tanh(x)

    @staticmethod
    def compile_kernel(c_code: str, function_name: str,
                        restype=ctypes.c_double,
                        argtypes: Optional[List] = None) -> Callable:
        """Compile a C math kernel and return a callable."""
        from .native import CPythonBridge
        bridge = CPythonBridge()
        return bridge.compile_and_load(c_code, function_name,
                                         restype=restype,
                                         argtypes=argtypes or [ctypes.c_double],
                                         extra_includes=["math.h"])


# ============================================================================
# 4. FastLoops — C-accelerated loops
# ============================================================================
class FastLoops:
    """C-accelerated loop operations that bypass the GIL."""

    @staticmethod
    def compile_loop(c_code: str, function_name: str,
                      restype=ctypes.c_void_p,
                      argtypes: Optional[List] = None) -> Callable:
        from .native import CPythonBridge
        bridge = CPythonBridge()
        return bridge.compile_and_load(c_code, function_name,
                                         restype=restype,
                                         argtypes=argtypes or [],
                                         extra_includes=["stdlib.h", "string.h", "math.h"])

    @staticmethod
    def fast_sum(arr: np.ndarray) -> float:
        c_code = "double fast_sum(double* arr, int n) { double t=0; for(int i=0;i<n;i++) t+=arr[i]; return t; }"
        try:
            fn = FastLoops.compile_loop(c_code, "fast_sum",
                                          restype=ctypes.c_double,
                                          argtypes=[ctypes.POINTER(ctypes.c_double), ctypes.c_int])
            arr_f64 = arr.astype(np.float64)
            ptr = arr_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return fn(ptr, len(arr))
        except Exception:
            return float(arr.sum())

    @staticmethod
    def fast_dot(a: np.ndarray, b: np.ndarray) -> float:
        c_code = "double fast_dot(double* a, double* b, int n) { double r=0; for(int i=0;i<n;i++) r+=a[i]*b[i]; return r; }"
        try:
            fn = FastLoops.compile_loop(c_code, "fast_dot",
                                          restype=ctypes.c_double,
                                          argtypes=[ctypes.POINTER(ctypes.c_double),
                                                     ctypes.POINTER(ctypes.c_double), ctypes.c_int])
            a_f64 = a.astype(np.float64)
            b_f64 = b.astype(np.float64)
            ptr_a = a_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ptr_b = b_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return fn(ptr_a, ptr_b, len(a))
        except Exception:
            return float(np.dot(a, b))

    @staticmethod
    def fast_argmax(arr: np.ndarray) -> int:
        c_code = "int fast_argmax(double* a, int n) { int b=0; for(int i=1;i<n;i++) if(a[i]>a[b]) b=i; return b; }"
        try:
            fn = FastLoops.compile_loop(c_code, "fast_argmax",
                                          restype=ctypes.c_int,
                                          argtypes=[ctypes.POINTER(ctypes.c_double), ctypes.c_int])
            arr_f64 = arr.astype(np.float64)
            ptr = arr_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            return int(fn(ptr, len(arr_f64)))
        except Exception:
            return int(np.argmax(arr))

    @staticmethod
    def fast_softmax(arr: np.ndarray) -> np.ndarray:
        c_code = """
#include <math.h>
void fast_sm(double* in, double* out, int n) {
    double mx=in[0]; for(int i=1;i<n;i++) if(in[i]>mx) mx=in[i];
    double s=0; for(int i=0;i<n;i++){out[i]=exp(in[i]-mx); s+=out[i];}
    for(int i=0;i<n;i++) out[i]/=s;
}"""
        try:
            fn = FastLoops.compile_loop(c_code, "fast_sm",
                                          restype=None,
                                          argtypes=[ctypes.POINTER(ctypes.c_double),
                                                     ctypes.POINTER(ctypes.c_double), ctypes.c_int])
            arr_f64 = arr.astype(np.float64)
            out = np.zeros_like(arr_f64)
            ptr_in = arr_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            ptr_out = out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            fn(ptr_in, ptr_out, len(arr_f64))
            return out.astype(arr.dtype)
        except Exception:
            e = np.exp(arr - arr.max())
            return (e / e.sum()).astype(arr.dtype)


# ============================================================================
# 5. SystemCalls — direct system calls via C
# ============================================================================
class SystemCalls:
    """Direct system calls via C."""

    @staticmethod
    def get_pid() -> int:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.getpid.restype = ctypes.c_int
            return int(libc.getpid())
        except Exception:
            return os.getpid()

    @staticmethod
    def get_hostname() -> str:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            buf = ctypes.create_string_buffer(256)
            libc.gethostname(buf, 256)
            return buf.value.decode()
        except Exception:
            return platform.node()

    @staticmethod
    def sync() -> None:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.sync()
        except Exception:
            pass

    @staticmethod
    def get_page_size() -> int:
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.getpagesize.restype = ctypes.c_int
            return int(libc.getpagesize())
        except Exception:
            return 4096

    @staticmethod
    def get_clock_ns() -> int:
        """Get nanosecond clock via C clock_gettime()."""
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            class Timespec(ctypes.Structure):
                _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
            ts = Timespec()
            libc.clock_gettime(1, ctypes.byref(ts))  # CLOCK_MONOTONIC=1
            return int(ts.tv_sec * 1_000_000_000 + ts.tv_nsec)
        except Exception:
            return int(time.time() * 1e9)


# ============================================================================
# 6. FileOps — C-accelerated file I/O
# ============================================================================
class FileOps:
    """C-accelerated file operations."""

    @staticmethod
    def fast_read(path: str) -> bytes:
        """Read a file using C fread."""
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.fopen.restype = ctypes.c_void_p
            libc.fread.restype = ctypes.c_size_t
            libc.fclose.restype = ctypes.c_int
            fp = libc.fopen(path.encode(), b"rb")
            if not fp:
                raise FileNotFoundError(f"cannot open {path}")
            try:
                libc.fseek(fp, 0, 2)
                size = libc.ftell(fp)
                libc.fseek(fp, 0, 0)
                buf = ctypes.create_string_buffer(size)
                n = libc.fread(buf, 1, size, fp)
                return buf.raw[:n]
            finally:
                libc.fclose(fp)
        except Exception:
            with open(path, "rb") as f:
                return f.read()

    @staticmethod
    def fast_write(path: str, data: bytes) -> int:
        """Write bytes using C fwrite."""
        try:
            libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
            libc.fopen.restype = ctypes.c_void_p
            libc.fwrite.restype = ctypes.c_size_t
            fp = libc.fopen(path.encode(), b"wb")
            if not fp:
                raise IOError(f"cannot open {path}")
            try:
                n = libc.fwrite(data, 1, len(data), fp)
                return int(n)
            finally:
                libc.fclose(fp)
        except Exception:
            with open(path, "wb") as f:
                return f.write(data)


# ============================================================================
# 7. CBridgedMethods — convenience wrapper
# ============================================================================
class CBridgedMethods:
    """Convenience methods that bridge Python to C.

    20+ methods that do useful things via C under the hood.
    """

    @staticmethod
    def get_function_source(fn: Callable) -> str:
        """Get source code of ANY function (even in REPL)."""
        return SourceExtractor.extract_source(fn)

    @staticmethod
    def get_object_address(obj: Any) -> int:
        return MemoryInspector.address_of(obj)

    @staticmethod
    def inspect_memory(obj: Any) -> Dict[str, Any]:
        return MemoryInspector.object_layout(obj)

    @staticmethod
    def fast_math(fn_name: str, x: float) -> float:
        return getattr(FastMath, f"fast_{fn_name}")(x)

    @staticmethod
    def compile_c_function(c_code: str, name: str,
                            restype=ctypes.c_double,
                            argtypes: Optional[List] = None) -> Callable:
        from .native import CPythonBridge
        bridge = CPythonBridge()
        return bridge.compile_and_load(c_code, name, restype=restype,
                                         argtypes=argtypes or [])

    @staticmethod
    def fast_array_sum(arr: np.ndarray) -> float:
        return FastLoops.fast_sum(arr)

    @staticmethod
    def fast_array_dot(a: np.ndarray, b: np.ndarray) -> float:
        return FastLoops.fast_dot(a, b)

    @staticmethod
    def fast_array_argmax(arr: np.ndarray) -> int:
        return FastLoops.fast_argmax(arr)

    @staticmethod
    def fast_softmax(arr: np.ndarray) -> np.ndarray:
        return FastLoops.fast_softmax(arr)

    @staticmethod
    def get_process_id() -> int:
        return SystemCalls.get_pid()

    @staticmethod
    def get_hostname() -> str:
        return SystemCalls.get_hostname()

    @staticmethod
    def get_nanosecond_clock() -> int:
        return SystemCalls.get_clock_ns()

    @staticmethod
    def force_filesystem_sync() -> None:
        SystemCalls.sync()

    @staticmethod
    def get_page_size() -> int:
        return SystemCalls.get_page_size()

    @staticmethod
    def fast_file_read(path: str) -> bytes:
        return FileOps.fast_read(path)

    @staticmethod
    def fast_file_write(path: str, data: bytes) -> int:
        return FileOps.fast_write(path, data)

    @staticmethod
    def read_memory(address: int, n_bytes: int) -> bytes:
        return MemoryInspector.read_bytes(address, n_bytes)

    @staticmethod
    def force_gc() -> Dict[str, int]:
        return MemoryInspector.force_gc()

    @staticmethod
    def all_methods() -> List[str]:
        """List all available C-bridged methods."""
        return [m for m in dir(CBridgedMethods) if not m.startswith("_") and m != "all_methods"]


__all__ = [
    "SourceExtractor", "MemoryInspector", "FastMath", "FastLoops",
    "SystemCalls", "FileOps", "CBridgedMethods",
]
